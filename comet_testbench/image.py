"""Instrument-agnostic image container and FITS loader.

Everything downstream (comet search, photometry) works on :class:`ImageFrame`,
which only needs a 2-D array, a celestial WCS, a mid-exposure time and an
exposure time. Filter and instrument are carried along as metadata but never
change the geometry or the search.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS, FITSFixedWarning
import astropy.units as u


@dataclass
class ImageFrame:
    """One 2-D image with the metadata the test bench needs."""

    data: np.ndarray
    wcs: WCS
    mid_time: Time
    exposure: float  # seconds
    header: fits.Header
    filter: str = "UNKNOWN"
    instrument: str = "UNKNOWN"
    source: str = ""  # file path and extension, for bookkeeping
    exposure_map: Optional[np.ndarray] = None  # seconds per pixel, if known
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.data.shape

    @property
    def pixel_scale_arcsec(self) -> float:
        """Mean pixel scale in arcsec/pixel."""
        scales = self.wcs.celestial.proj_plane_pixel_scales()
        return float(np.mean([s.to_value(u.arcsec) for s in scales]))

    def footprint(self) -> np.ndarray:
        """Sky corners (RA, Dec in deg) of the image, shape (4, 2)."""
        return self.wcs.celestial.calc_footprint(axes=self.data.shape[::-1])

    def center(self):
        """Sky coordinate of the image center as (ra, dec) degrees."""
        ny, nx = self.data.shape
        ra, dec = self.wcs.celestial.pixel_to_world_values((nx - 1) / 2, (ny - 1) / 2)
        return float(ra), float(dec)

    def contains_pixel(self, x: float, y: float, margin: float = 0.0) -> bool:
        """True if (x, y) lands on the image with real exposure behind it."""
        ny, nx = self.data.shape
        if not (-0.5 + margin <= x <= nx - 0.5 - margin and -0.5 + margin <= y <= ny - 0.5 - margin):
            return False
        ix, iy = int(round(x)), int(round(y))
        ix = min(max(ix, 0), nx - 1)
        iy = min(max(iy, 0), ny - 1)
        if self.exposure_map is not None:
            return bool(self.exposure_map[iy, ix] > 0)
        val = self.data[iy, ix]
        return bool(np.isfinite(val))


# --------------------------------------------------------------------------
# Header helpers (generic FITS conventions first, instrument quirks via adapters)
# --------------------------------------------------------------------------

_TIME_SCALES = {"UTC": "utc", "UT": "utc", "TT": "tt", "TDB": "tdb", "TAI": "tai", "GPS": "tai"}


def _header_time(header: fits.Header) -> Optional[Time]:
    """Mid-exposure time from common FITS keywords, honouring TIMESYS."""
    scale = _TIME_SCALES.get(str(header.get("TIMESYS", "UTC")).strip().upper(), "utc")

    def _t(key):
        val = header.get(key)
        if val is None:
            return None
        if key.startswith("MJD"):
            return Time(float(val), format="mjd", scale=scale)
        return Time(str(val).replace(" ", "T"), format="isot", scale=scale)

    start = _t("DATE-OBS") if "DATE-OBS" in header else _t("MJD-OBS") if "MJD-OBS" in header else None
    if "MJD-AVG" in header:
        return _t("MJD-AVG")
    if "DATE-AVG" in header:
        return _t("DATE-AVG")
    end = _t("DATE-END") if "DATE-END" in header else _t("MJD-END") if "MJD-END" in header else None
    if start is not None and end is not None:
        return start + (end - start) / 2
    if start is not None:
        span = header.get("TELAPSE", header.get("EXPTIME", header.get("EXPOSURE")))
        if span is not None:
            return start + TimeDelta(float(span) / 2, format="sec")
        return start
    return None


def _header_exposure(header: fits.Header) -> float:
    for key in ("EXPOSURE", "EXPTIME", "ONTIME", "TELAPSE"):
        if key in header:
            return float(header[key])
    return float("nan")


def _header_filter(header: fits.Header) -> str:
    for key in ("FILTER", "FILTNAM", "FILTER1", "BAND", "FILTNAME"):
        if key in header and str(header[key]).strip():
            return str(header[key]).strip().upper()
    return "UNKNOWN"


def _celestial_wcs(header: fits.Header) -> Optional[WCS]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        try:
            w = WCS(header)
        except Exception:
            return None
    if not w.has_celestial:
        return None
    return w.celestial


def iter_frames(path, exposure_map_path=None) -> Iterator[ImageFrame]:
    """Yield an :class:`ImageFrame` for every 2-D image HDU with a celestial WCS.

    Works for single-image FITS files and multi-extension files such as Swift
    UVOT ``*_sk.img`` products (one extension per snapshot). If an exposure map
    file is given, its HDUs are matched by position (same order as images).
    """
    path = Path(path)
    exp_hdus = None
    if exposure_map_path is not None:
        exp_hdus = [h for h in fits.open(exposure_map_path) if h.data is not None and h.data.ndim == 2]

    with fits.open(path) as hdul:
        primary = hdul[0].header
        img_index = 0
        for i, hdu in enumerate(hdul):
            if hdu.data is None or getattr(hdu.data, "ndim", 0) != 2:
                continue
            header = hdu.header.copy()
            # Fall back to primary-header keywords the extension lacks.
            for key in ("DATE-OBS", "DATE-END", "MJD-OBS", "TIMESYS", "EXPTIME", "EXPOSURE",
                        "FILTER", "INSTRUME", "TELESCOP", "OBJECT"):
                if key not in header and key in primary:
                    header[key] = primary[key]
            wcs = _celestial_wcs(header)
            if wcs is None:
                continue
            mid = _header_time(header)
            if mid is None:
                raise ValueError(f"{path}[{i}]: no observation time keywords found")
            exp_map = None
            if exp_hdus is not None and img_index < len(exp_hdus):
                exp_map = np.asarray(exp_hdus[img_index].data, dtype=float)
            yield ImageFrame(
                data=np.asarray(hdu.data, dtype=float),
                wcs=wcs,
                mid_time=mid,
                exposure=_header_exposure(header),
                header=header,
                filter=_header_filter(header),
                instrument=str(header.get("INSTRUME", header.get("TELESCOP", "UNKNOWN"))).strip(),
                source=f"{path.name}[{i}]",
                exposure_map=exp_map,
            )
            img_index += 1


def load_frames(path, exposure_map_path=None) -> list:
    """List version of :func:`iter_frames`."""
    return list(iter_frames(path, exposure_map_path))
