"""Interface to the separately developed OH model package.

The test bench does not implement an OH model. It produces
:class:`OHMeasurement` objects (OH-only fluxes in a set of apertures, plus the
geometry) and fits a production rate with any object that satisfies the
:class:`OHModel` protocol below.

What the OH package needs to provide
------------------------------------
One object with an ``aperture_flux`` method::

    class MyOHModel:
        name = "my-oh-model"        # optional, recorded with results
        version = "0.1"             # optional

        def aperture_flux(self, production_rate, radius_km, geometry):
            '''Predicted OH (0-0)+(1-1) band flux in erg s-1 cm-2 inside
            circular apertures of projected radius ``radius_km`` (array),
            for water/OH production rate ``production_rate`` (molecules/s)
            and ``geometry`` (ObservingGeometry: r_h, r_dot, delta, phase...).'''

It can be plugged in from Python or named on the command line as
``package.module:ClassOrFactory`` (see :func:`load_model`).

Getting OH-only fluxes from Swift
---------------------------------
UVOT UVW1 contains the OH 308 nm band plus dust-scattered sunlight; V is almost
pure dust continuum. :func:`subtract_continuum` forms
``OH = UVW1 - beta * V`` aperture by aperture, then converts the OH count rate
to a band flux with ``oh_conversion`` (erg s-1 cm-2 per count s-1 for an OH
emission spectrum, not the broadband factor used for stars). Both ``beta``
(depends on dust colour) and ``oh_conversion`` must be supplied by the user
from their adopted calibration.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from .ephemeris import ObservingGeometry


@runtime_checkable
class OHModel(Protocol):
    def aperture_flux(self, production_rate: float, radius_km: np.ndarray,
                      geometry: ObservingGeometry) -> np.ndarray:
        ...


@dataclass
class OHMeasurement:
    geometry: ObservingGeometry
    radius_km: np.ndarray
    oh_rate: np.ndarray  # counts/s attributed to OH
    oh_rate_err: np.ndarray
    oh_flux: np.ndarray  # erg s-1 cm-2
    oh_flux_err: np.ndarray
    meta: dict = field(default_factory=dict)


@dataclass
class ProductionRateFit:
    q: float  # molecules / s
    q_err: float
    chi2: float
    dof: int
    q_per_aperture: np.ndarray
    model_name: str = ""


def subtract_continuum(oh_band: Sequence, continuum: Sequence, beta: float, oh_conversion: float,
                       geometry: ObservingGeometry, rtol: float = 1e-3, meta=None) -> OHMeasurement:
    """OH-only fluxes from matched apertures in an OH filter and a continuum filter.

    ``oh_band`` and ``continuum`` are lists of
    :class:`~comet_testbench.photometry.ApertureResult` measured with the same
    radii in km (e.g. both from ``fixed_aperture_photometry(..., unit='km')``).
    """
    r1 = np.array([a.radius_km for a in oh_band])
    r2 = np.array([a.radius_km for a in continuum])
    if r1.shape != r2.shape or not np.allclose(r1, r2, rtol=rtol):
        raise ValueError("OH-band and continuum apertures must have the same radii in km")
    w = np.array([a.rate for a in oh_band])
    we = np.array([a.rate_err for a in oh_band])
    v = np.array([a.rate for a in continuum])
    ve = np.array([a.rate_err for a in continuum])
    oh = w - beta * v
    oh_err = np.sqrt(we**2 + (beta * ve) ** 2)
    m = {"beta": beta, "oh_conversion": oh_conversion}
    m.update(meta or {})
    return OHMeasurement(geometry, r1, oh, oh_err, oh * oh_conversion, oh_err * oh_conversion, m)


def _q_for_flux(model, target, radius_km, geometry, log_bounds):
    from scipy.optimize import brentq

    def f(logq):
        return float(np.atleast_1d(model.aperture_flux(10**logq, np.atleast_1d(radius_km), geometry))[0]) - target

    lo, hi = log_bounds
    if f(lo) * f(hi) > 0:
        return float("nan")
    return 10 ** brentq(f, lo, hi, xtol=1e-6)


def fit_production_rate(model: OHModel, meas: OHMeasurement,
                        log_q_bounds=(22.0, 32.0)) -> ProductionRateFit:
    """Best-fit production rate across all apertures (chi-square in log Q).

    Also returns the production rate implied by each aperture alone, which is
    the usual consistency check: a good model gives a flat Q versus aperture.
    """
    from scipy.optimize import minimize_scalar

    r = np.asarray(meas.radius_km, float)
    y = np.asarray(meas.oh_flux, float)
    e = np.asarray(meas.oh_flux_err, float)
    good = np.isfinite(y) & np.isfinite(e) & (e > 0)

    def chi2(logq):
        pred = np.asarray(model.aperture_flux(10**logq, r[good], meas.geometry), float)
        return float(np.sum(((y[good] - pred) / e[good]) ** 2))

    res = minimize_scalar(chi2, bounds=log_q_bounds, method="bounded",
                          options={"xatol": 1e-6})
    logq = float(res.x)
    # 1-sigma from delta-chi2 = 1, by curvature.
    h = 1e-3
    c0, cp, cm = chi2(logq), chi2(logq + h), chi2(logq - h)
    curv = (cp - 2 * c0 + cm) / h**2
    sig_log = np.sqrt(2 / curv) if curv > 0 else float("nan")
    q = 10**logq
    per_ap = np.array([_q_for_flux(model, yi, ri, meas.geometry, log_q_bounds)
                       for yi, ri in zip(y, r)])
    return ProductionRateFit(q, q * np.log(10) * sig_log, c0, int(good.sum()) - 1, per_ap,
                             getattr(model, "name", type(model).__name__))


def load_model(spec: str, **kwargs) -> OHModel:
    """Import ``'package.module:Name'`` and instantiate (or call) it with kwargs."""
    mod_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError("Model spec must look like 'package.module:ClassOrFactory'")
    obj = getattr(importlib.import_module(mod_name), attr)
    model = obj(**kwargs) if callable(obj) else obj
    if not isinstance(model, OHModel):
        raise TypeError(f"{spec} does not provide aperture_flux(production_rate, radius_km, geometry)")
    return model
