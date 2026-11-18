"""The OH -> H2O steps, each a thin wrapper over swift_comet_pipeline.

Values with uncertainties are swift_comet_pipeline's ``ValueAndStandardDev``
(``.value``, ``.sigma``), so they pass straight into its functions.

    rate_OH_band, rate_continuum
      -> continuum_subtract   OH signal (ct/s) = OH band - beta * continuum
      -> oh_flux              erg cm^-2 s^-1    (instrument calibration)
      -> oh_molecules         number of OH in the aperture
                              (swift_comet_pipeline oh_flux_to_num_oh:
                               L = 4 pi Delta^2 F, N = L r_h^2 / g_1au(r_dot));
                              g is all OH bands summed (Swift/UVW1) or only
                              the bands the instrument measures, e.g. (1-0)
                              at 282 nm for ULTRASAT
      -> q_h2o_in_aperture    Q(H2O), s^-1, vectorial model column density
                              integrated over the aperture radius
                              (num_oh_within_r_to_q_h2o_vectorial)
      -> q_h2o_total          Q(H2O) treating N as the whole coma
                              (num_oh_to_q_h2o_vectorial)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .backend import require_data
from .calibration import OHCalibration


def _vsd(value, sigma):
    from swift_comet_pipeline.scp_types.primitive.error_propogation import ValueAndStandardDev

    return ValueAndStandardDev(value=float(value), sigma=float(sigma))


@dataclass(frozen=True)
class Geometry:
    """What the model needs about the comet's geometry at the image time."""

    r_h: float  # heliocentric distance, au
    r_dot: float  # heliocentric radial velocity, km/s (Swings effect on the g-factor)
    delta: float  # observer distance, au

    @classmethod
    def from_observing_geometry(cls, g) -> "Geometry":
        """From comet_testbench.ephemeris.ObservingGeometry (or anything with these attributes)."""
        return cls(float(g.r_h), float(g.r_dot), float(g.delta))


def continuum_subtract(oh_band_rate, oh_band_err, continuum_rate, continuum_err, beta):
    """OH signal in the OH band, ct/s: ``oh_band - beta * continuum``.

    The two inputs must be measured in apertures of the same physical radius.
    Errors add in quadrature (the convention swift_comet_pipeline uses).
    """
    value = oh_band_rate - beta * continuum_rate
    sigma = np.hypot(oh_band_err, beta * continuum_err)
    return _vsd(value, sigma)


def oh_flux(oh_rate, calibration: OHCalibration):
    """OH emission flux, erg cm^-2 s^-1, from the OH signal count rate."""
    return oh_rate * calibration.oh_flux_per_count_rate()


def oh_molecules(flux, geometry: Geometry, bands=None):
    """Number of OH molecules that give this flux (fluorescence g-factor).

    bands: which OH A-X bands the flux covers, e.g. ``("1-0",)`` for the
        282 nm band. ``None`` (default) means all bands in the table summed,
        which is what swift_comet_pipeline does for UVW1, and then its own
        ``oh_flux_to_num_oh`` is called. N = 4 pi Delta^2 F r_h^2 / g_1au(r_dot).
    """
    require_data()
    if bands is None:
        from swift_comet_pipeline.modeling.water_production.fluorescence_OH import (
            oh_flux_to_num_oh,
        )

        n = oh_flux_to_num_oh(flux, rh_au=float(geometry.r_h),
                              helio_v_kms=float(geometry.r_dot), delta_au=float(geometry.delta))
        return _vsd(n.value, n.sigma)
    # Same formula as swift_comet_pipeline's oh_flux_to_num_oh, with the band's g-factor.
    delta_cm = float(geometry.delta) * AU_CM
    g = g_factor_1au(geometry.r_dot, bands) / float(geometry.r_h) ** 2
    n = flux * (4 * np.pi * delta_cm**2 / g)
    return _vsd(n.value, n.sigma)


def q_h2o_in_aperture(num_oh, geometry: Geometry, radius_km: float):
    """Q(H2O), s^-1, whose vectorial-model OH column inside ``radius_km`` equals ``num_oh``."""
    import astropy.units as u
    from swift_comet_pipeline.modeling.water_production.num_OH_to_Q import (
        num_oh_within_r_to_q_h2o_vectorial,
    )

    rh, r = float(geometry.r_h), float(radius_km) * u.km
    try:
        q = num_oh_within_r_to_q_h2o_vectorial(rh_au=rh, num_oh=num_oh, within_r=r)
        return _vsd(q.value, q.sigma)
    except ZeroDivisionError:
        # swift_comet_pipeline divides by N, N - sigma and N + sigma, so N = 0 or
        # |N| = sigma fails. Q is linear in N: scale its Q for N = 1 instead,
        # keeping its error convention (sigma_Q = sqrt(2) * sigma_N * dQ/dN).
        q1 = num_oh_within_r_to_q_h2o_vectorial(rh_au=rh, num_oh=_vsd(1.0, 0.0), within_r=r).value
        return _vsd(num_oh.value * q1, np.sqrt(2.0) * num_oh.sigma * q1)


def q_h2o_total(num_oh, geometry: Geometry):
    """Q(H2O), s^-1, whose whole vectorial-model coma holds ``num_oh`` OH molecules.

    Only meaningful when the aperture contains essentially all the OH; for
    smaller apertures use :func:`q_h2o_in_aperture`.
    """
    from swift_comet_pipeline.modeling.water_production.num_OH_to_Q import (
        num_oh_to_q_h2o_vectorial,
    )

    rh = float(geometry.r_h)
    try:
        q = num_oh_to_q_h2o_vectorial(rh_au=rh, num_oh=num_oh)
        return _vsd(q.value, q.sigma)
    except ZeroDivisionError:
        # N = 0: swift_comet_pipeline divides by N. Q is linear in N.
        q1 = num_oh_to_q_h2o_vectorial(rh_au=rh, num_oh=_vsd(1.0, 0.0)).value
        return _vsd(num_oh.value * q1, num_oh.sigma * q1)


# OH A-X bands in swift_comet_pipeline's g-factor table (Schleicher & A'Hearn 1988,
# Table 5): column name and scale. Band centres: 0-0 309 nm, 1-0 282 nm,
# 1-1 314 nm, 2-2 ~320 nm.
OH_BANDS = {
    "0-0": ("L00_1e-15", 1e-15),
    "1-0": ("L10_1e-16", 1e-16),
    "1-1": ("L11_1e-16", 1e-16),
    "2-2": ("L22_1e-18", 1e-18),
}
AU_CM = 1.495978707e13


def _band_key(b) -> str:
    k = str(b).strip().replace("(", "").replace(")", "").replace(" ", "")
    if k not in OH_BANDS:
        raise ValueError(f"unknown OH band {b!r}; known: {sorted(OH_BANDS)}")
    return k


@lru_cache(maxsize=None)
def _band_interpolator(bands: tuple):
    import pandas as pd
    from scipy.interpolate import interp1d
    from swift_comet_pipeline.modeling.fluorescence import hydroxyl_gfactor as hg

    df = pd.read_csv(hg.read_swift_pipeline_config().oh_fluorescence_path)
    g = sum(df[OH_BANDS[b][0]].to_numpy() * OH_BANDS[b][1] for b in bands)
    # Cubic, like swift_comet_pipeline's hydroxyl_gfactor_1au.
    return interp1d(df["v_kms"].to_numpy(), g, kind="cubic")


def g_factor_1au(r_dot: float, bands=None) -> float:
    """OH fluorescence g-factor at 1 au, erg s^-1 molecule^-1 (Schleicher & A'Hearn 1988).

    bands: None for all bands summed (swift_comet_pipeline's
    ``hydroxyl_gfactor_1au``), or a list such as ``["1-0"]`` for single bands,
    read from the same table.
    """
    require_data()
    if bands is None:
        from swift_comet_pipeline.modeling.fluorescence.hydroxyl_gfactor import (
            hydroxyl_gfactor_1au,
        )

        return float(hydroxyl_gfactor_1au(helio_v_kms=float(r_dot)))
    keys = tuple(sorted({_band_key(b) for b in bands}))
    return float(_band_interpolator(keys)(float(r_dot)))
