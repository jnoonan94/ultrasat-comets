"""Instrument adapters.

The search and photometry code is instrument-agnostic. Anything that depends on
the instrument lives here: where the observer is (for the ephemeris), how to
turn raw pixel values into a count rate, and how to calibrate a count rate.

Add a new dataset by subclassing :class:`InstrumentAdapter`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .image import ImageFrame


class InstrumentAdapter:
    """Generic adapter: geocentric observer, image already in counts."""

    name = "generic"

    def __init__(self, observer: str = "500@399", mpc_code: str = "500"):
        self._observer = observer
        self._mpc_code = mpc_code

    def matches(self, frame: ImageFrame) -> bool:
        return True

    # --- geometry ---------------------------------------------------------
    def observer(self, frame: ImageFrame) -> str:
        """Horizons CENTER string for the observer of this frame."""
        return self._observer

    def mpc_code(self, frame: ImageFrame) -> str:
        """MPC observatory code used for the sb_ident discovery query."""
        return self._mpc_code

    def sb_ident_site(self, frame: ImageFrame):
        """(lat_deg, lon_deg, alt_km) for sb_ident, or None to use the MPC code."""
        return None

    # --- radiometry -------------------------------------------------------
    def count_rate_image(self, frame: ImageFrame) -> np.ndarray:
        """Image in counts/s per pixel.

        BUNIT is honoured when it says the image is already a rate. Otherwise
        the image is divided by the exposure map (if any) or EXPOSURE.
        """
        bunit = str(frame.header.get("BUNIT", "")).lower().replace(" ", "")
        if bunit in ("counts/s", "count/s", "ct/s", "cts/s", "electrons/s", "e-/s", "adu/s"):
            return frame.data.copy()
        if frame.exposure_map is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                rate = np.where(frame.exposure_map > 0, frame.data / frame.exposure_map, np.nan)
            return rate
        return frame.data / frame.exposure

    def flux_conversion(self, frame: ImageFrame) -> Optional[float]:
        """Flux density per count rate (erg s-1 cm-2 A-1 per count s-1), if known."""
        val = frame.header.get("PHOTFLAM")
        return float(val) if val is not None else None

    def zero_point(self, frame: ImageFrame) -> Optional[float]:
        """Magnitude zero point for 1 count/s, if known."""
        for key in ("MAGZP", "ZEROPT", "PHOTZP", "MAGZERO"):
            if key in frame.header:
                return float(frame.header[key])
        return None


@dataclass(frozen=True)
class UVOTFilter:
    zero_point_vega: float  # mag for 1 count/s
    flux_factor: float  # erg s-1 cm-2 A-1 per count s-1 (Vega-like spectrum)


# Poole et al. (2008) / Breeveld et al. (2011) in-orbit calibration.
# Verify against the current Swift CALDB before using for science.
UVOT_FILTERS = {
    "V": UVOTFilter(17.89, 2.61e-16),
    "B": UVOTFilter(19.11, 1.32e-16),
    "U": UVOTFilter(18.34, 1.50e-16),
    "UVW1": UVOTFilter(17.44, 4.30e-16),
    "UVM2": UVOTFilter(16.85, 7.50e-16),
    "UVW2": UVOTFilter(17.38, 6.00e-16),
    "WHITE": UVOTFilter(20.29, 3.70e-17),
}

# Poole et al. (2008) coincidence-loss polynomial and dead-time fraction.
_COI_POLY = (0.066, -0.091, 0.029, 0.031)
_DEADTIME_FRAC = 0.0155


def uvot_coincidence_correction(raw_rate, frame_time: float = 0.0110322):
    """Point-source coincidence-loss correction (Poole et al. 2008).

    ``raw_rate`` is the count rate inside the standard 5 arcsec aperture.
    Returns the corrected rate. Rates above the saturation limit return NaN.
    """
    raw_rate = np.asarray(raw_rate, dtype=float)
    ftd = frame_time * (1 - _DEADTIME_FRAC)
    x = raw_rate * frame_time
    with np.errstate(invalid="ignore", divide="ignore"):
        arg = 1 - raw_rate * ftd
        theory = np.where(arg > 0, -np.log(np.where(arg > 0, arg, 1)) / ftd, np.nan)
    a1, a2, a3, a4 = _COI_POLY
    poly = 1 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4
    return theory * poly


class SwiftUVOT(InstrumentAdapter):
    """Swift/UVOT sky images (``*_sk.img``) from the HEASARC archive.

    * Observer is the Swift spacecraft (Horizons ID -128485); sb_ident has no
      spacecraft option, so discovery uses the geocenter with a wider margin.
    * Images are counts; divide by the exposure map (``*_ex.img``) or EXPOSURE.
    * Optional pixel-level coincidence-loss correction: each pixel's local
      rate in a 5 arcsec circle is corrected and the ratio applied to the pixel.
      This is the usual approximation for extended sources; it is not exact.
    """

    name = "swift-uvot"

    def __init__(self, coincidence_correction: bool = True):
        super().__init__(observer="500@-128485", mpc_code="500")
        self.coincidence_correction = coincidence_correction

    def matches(self, frame: ImageFrame) -> bool:
        tel = str(frame.header.get("TELESCOP", "")).upper()
        inst = str(frame.header.get("INSTRUME", "")).upper()
        return tel == "SWIFT" and inst.startswith("UVOT")

    def count_rate_image(self, frame: ImageFrame) -> np.ndarray:
        rate = super().count_rate_image(frame)
        if not self.coincidence_correction:
            return rate
        from scipy.ndimage import convolve

        r_pix = 5.0 / frame.pixel_scale_arcsec
        n = int(np.ceil(r_pix))
        yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
        kernel = (xx**2 + yy**2 <= r_pix**2).astype(float)
        filled = np.nan_to_num(rate, nan=0.0)
        local = convolve(filled, kernel, mode="constant", cval=0.0)
        frame_time = float(frame.header.get("FRAMTIME", 0.0110322))
        corrected = uvot_coincidence_correction(local, frame_time)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(local > 0, corrected / local, 1.0)
        return rate * factor

    def _filter(self, frame):
        return UVOT_FILTERS.get(frame.filter.upper())

    def flux_conversion(self, frame):
        f = self._filter(frame)
        return f.flux_factor if f else None

    def zero_point(self, frame):
        f = self._filter(frame)
        return f.zero_point_vega if f else None


# Observer used by the field-of-regard simulator (ultrasat_object_search.api_query):
# a site 35,785 km above (32.02814 N, 34.89832 E).
ULTRASAT_SIM_SITE = (32.028140, 34.898320, 35785.0)


class UltraSat(InstrumentAdapter):
    """UltraSat images, real or simulated (see :mod:`comet_testbench.pointing`).

    Until UltraSat has an ephemeris in Horizons the observer defaults to the
    same site the field-of-regard simulator uses, so simulated and real
    searches agree. Pass ``horizons_id`` (e.g. ``'500@-NNNNNN'``) once the
    spacecraft is in Horizons; sb_ident still uses ``site`` for discovery.
    """

    name = "ultrasat"

    def __init__(self, site=ULTRASAT_SIM_SITE, horizons_id=None):
        super().__init__()
        self.site = tuple(site)
        self.horizons_id = horizons_id

    def matches(self, frame: ImageFrame) -> bool:
        return "ULTRASAT" in str(frame.header.get("TELESCOP", "")).upper()

    def observer(self, frame):
        if self.horizons_id:
            return self.horizons_id
        lat, lon, alt = self.site
        return f"site:{lon},{lat},{alt}"

    def sb_ident_site(self, frame):
        return self.site


DEFAULT_ADAPTERS = (SwiftUVOT(), UltraSat())


def adapter_for(frame: ImageFrame, adapters=DEFAULT_ADAPTERS) -> InstrumentAdapter:
    """Pick the first adapter that recognises the frame, else the generic one."""
    for a in adapters:
        if a.matches(frame):
            return a
    return InstrumentAdapter()
