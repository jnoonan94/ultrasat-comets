"""The OH -> H2O steps, each a thin wrapper over swift_comet_pipeline.

Values with uncertainties are swift_comet_pipeline's ``ValueAndStandardDev``
(``.value``, ``.sigma``), so they pass straight into its functions.

    rate_OH_band, rate_continuum
      -> continuum_subtract   OH signal (ct/s) = OH band - beta * continuum
      -> oh_flux              erg cm^-2 s^-1    (instrument calibration)
      -> oh_molecules         number of OH in the aperture
                              (swift_comet_pipeline oh_flux_to_num_oh:
                               L = 4 pi Delta^2 F, N = L r_h^2 / g_1au(r_dot))
      -> q_h2o_in_aperture    Q(H2O), s^-1, vectorial model column density
                              integrated over the aperture radius
                              (num_oh_within_r_to_q_h2o_vectorial)
      -> q_h2o_total          Q(H2O) treating N as the whole coma
                              (num_oh_to_q_h2o_vectorial)
"""

from __future__ import annotations

from dataclasses import dataclass

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


def oh_molecules(flux, geometry: Geometry):
    """Number of OH molecules that give this flux (fluorescence g-factor)."""
    from swift_comet_pipeline.modeling.water_production.fluorescence_OH import oh_flux_to_num_oh

    require_data()
    n = oh_flux_to_num_oh(flux, rh_au=float(geometry.r_h), helio_v_kms=float(geometry.r_dot),
                          delta_au=float(geometry.delta))
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


def g_factor_1au(r_dot: float) -> float:
    """OH fluorescence g-factor at 1 au, erg s^-1 molecule^-1 (Schleicher & A'Hearn 1988)."""
    from swift_comet_pipeline.modeling.fluorescence.hydroxyl_gfactor import hydroxyl_gfactor_1au

    require_data()
    return float(hydroxyl_gfactor_1au(helio_v_kms=float(r_dot)))
