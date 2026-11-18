"""Instrument-specific numbers the OH model needs, behind one small interface.

Only two things in the OH -> H2O chain depend on the telescope:

* ``beta(dust_redness)``: the continuum scale. A dust coma that gives 1 count/s
  in the continuum band gives ``beta`` counts/s in the OH band. The OH signal
  is ``rate_OH_band - beta * rate_continuum``.
* ``oh_flux_per_count_rate``: OH emission-band flux (erg cm^-2 s^-1) per
  count/s of pure OH signal in the OH band.

Everything after that (fluorescence g-factor, OH molecule count, vectorial
model, production rate) is instrument-independent.

``SwiftUVOTCalibration`` gets both numbers from swift_comet_pipeline.
``FixedCalibration`` takes them as plain numbers, which is how a new
instrument such as ULTRASAT plugs in until it has its own calibration class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Union, runtime_checkable


@runtime_checkable
class OHCalibration(Protocol):
    """What the model needs to know about an instrument."""

    name: str
    oh_band: str  # filter name as it appears in the hand-off 'filter' column
    continuum_band: str

    def accepts(self, instrument: str) -> bool:
        """True if hand-off rows with this 'instrument' value can use this calibration."""

    def beta(self, dust_redness: float) -> float:
        """Continuum scale for dust reddening in percent per 100 nm."""

    def oh_flux_per_count_rate(self) -> float:
        """erg cm^-2 s^-1 of OH emission per count/s of OH signal in the OH band."""


# Hand-off filter names (FITS FILTER keyword) -> swift_comet_pipeline UvotFilter names.
_UVOT_FILTER_NAMES = {
    "UVW1": "uw1", "UVW2": "uw2", "UVM2": "um2", "U": "uuu",
    "B": "ubb", "V": "uvv", "WHITE": "white",
}


def _uvot_filter(name: str):
    from swift_comet_pipeline.scp_types.primitive.uvot_filter import UvotFilter

    key = _UVOT_FILTER_NAMES.get(str(name).strip().upper())
    if key is None:
        raise ValueError(f"unknown UVOT filter {name!r}; known: {sorted(_UVOT_FILTER_NAMES)}")
    return UvotFilter[key]


@dataclass
class SwiftUVOTCalibration:
    """Swift/UVOT, via swift_comet_pipeline.

    * beta: ``swift_comet_pipeline.photometry.dust.beta_parameter`` (solar
      spectrum through the two UVOT effective-area curves, with the reddening
      correction).
    * flux factor: ``swift_comet_pipeline.modeling.water_production.flux_OH``
      (OH spectrum of Bodewits et al. 2019 through UVW1; UVW2, U and UVM2 are
      rough scalings of it in swift_comet_pipeline).

    Needs swift_comet_pipeline's data files; call
    ``oh_h2o.configure(data_dir=...)`` first if they are not found.
    """

    oh_band: str = "UVW1"
    continuum_band: str = "V"
    name: str = "swift-uvot"

    def accepts(self, instrument: str) -> bool:
        s = str(instrument).upper()
        return "UVOT" in s or "SWIFT" in s

    def beta(self, dust_redness: float) -> float:
        from swift_comet_pipeline.photometry.dust.beta_parameter import beta_parameter

        return float(beta_parameter(float(dust_redness), _uvot_filter(self.oh_band),
                                    _uvot_filter(self.continuum_band)))

    def oh_flux_per_count_rate(self) -> float:
        import astropy.units as u
        from swift_comet_pipeline.modeling.water_production.flux_OH import (
            oh_count_rates_to_flux_factor,
        )

        f = oh_count_rates_to_flux_factor(_uvot_filter(self.oh_band))
        value = float(f.to_value(u.erg / (u.cm**2 * u.s)))
        if value <= 0:
            raise ValueError(f"swift_comet_pipeline has no OH flux factor for {self.oh_band}")
        return value


@dataclass
class FixedCalibration:
    """Any instrument, from numbers you supply.

    ``beta`` is a number (no reddening dependence) or a function of dust
    redness in %/100 nm. ``instruments`` lists substrings of the hand-off
    'instrument' column this applies to; empty means any.
    """

    name: str
    oh_band: str
    continuum_band: str
    beta_value: Union[float, Callable[[float], float]]
    flux_per_count_rate: float
    instruments: tuple = field(default_factory=tuple)

    def accepts(self, instrument: str) -> bool:
        if not self.instruments:
            return True
        s = str(instrument).upper()
        return any(i.upper() in s for i in self.instruments)

    def beta(self, dust_redness: float) -> float:
        if callable(self.beta_value):
            return float(self.beta_value(float(dust_redness)))
        return float(self.beta_value)

    def oh_flux_per_count_rate(self) -> float:
        return float(self.flux_per_count_rate)


def calibration_for(name: str, oh_band=None, continuum_band=None) -> OHCalibration:
    """Built-in calibration by name (currently only 'swift-uvot')."""
    if name.lower() in ("swift-uvot", "swift", "uvot"):
        kw = {}
        if oh_band:
            kw["oh_band"] = oh_band
        if continuum_band:
            kw["continuum_band"] = continuum_band
        return SwiftUVOTCalibration(**kw)
    raise ValueError(
        f"no built-in calibration {name!r}; use FixedCalibration for other instruments"
    )
