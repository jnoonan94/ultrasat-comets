"""Fixed and iterative (growth-curve) aperture photometry of a coma.

All measurements are made on the count-rate image the instrument adapter
provides, so the same code serves any filter or dataset. Radii can be given in
pixels, arcsec or km at the comet (km needs the observing geometry).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.centroids import centroid_com

from .ephemeris import ObservingGeometry
from .image import ImageFrame
from .instruments import InstrumentAdapter, adapter_for


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------

def to_pixels(radii, unit: str, frame: ImageFrame, geometry: Optional[ObservingGeometry] = None):
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    if unit in ("pix", "pixel", "pixels"):
        return radii
    if unit == "arcsec":
        return radii / frame.pixel_scale_arcsec
    if unit == "km":
        if geometry is None:
            raise ValueError("Radii in km need the comet's observing geometry")
        return geometry.km_to_arcsec(radii) / frame.pixel_scale_arcsec
    raise ValueError(f"Unknown radius unit {unit!r}; use 'pix', 'arcsec' or 'km'")


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass
class Background:
    level: float  # count rate per pixel
    sigma: float  # pixel-to-pixel scatter (rate)
    n_pix: int
    method: str

    @property
    def error_of_mean(self) -> float:
        return self.sigma / np.sqrt(max(self.n_pix, 1))


@dataclass
class ApertureResult:
    x: float
    y: float
    radius_pix: float
    radius_arcsec: float
    radius_km: float
    area_pix: float
    coverage: float  # fraction of the aperture on valid pixels
    rate: float  # background-subtracted count rate, counts/s
    rate_err: float
    background: float  # per pixel
    mag: float = float("nan")
    mag_err: float = float("nan")
    flux_density: float = float("nan")  # erg s-1 cm-2 A-1
    flux_density_err: float = float("nan")

    def row(self):
        return asdict(self)


@dataclass
class GrowthCurve:
    apertures: list
    converged: bool
    converged_index: Optional[int]
    background: Background
    centroid_shift_pix: float
    stop_reason: str

    @property
    def best(self) -> ApertureResult:
        i = self.converged_index if self.converged_index is not None else -1
        return self.apertures[i]

    def table(self):
        from astropy.table import Table

        return Table(rows=[a.row() for a in self.apertures])


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def estimate_background(rate: np.ndarray, x: float, y: float, method: str = "annulus",
                        r_in_pix: Optional[float] = None, r_out_pix: Optional[float] = None,
                        mask_radius_pix: Optional[float] = None, sigma: float = 3.0,
                        value: Optional[float] = None) -> Background:
    """Sky level around the comet.

    ``annulus``: sigma-clipped median in an annulus centred on the comet.
    ``global``: sigma-clipped median of the whole frame with the coma
    (``mask_radius_pix``) masked. ``fixed``: use ``value`` as given.
    """
    good = np.isfinite(rate)
    if method == "fixed":
        return Background(float(value), 0.0, 1, "fixed")
    if method == "annulus":
        if r_in_pix is None or r_out_pix is None:
            raise ValueError("annulus background needs r_in and r_out")
        ann = CircularAnnulus((x, y), r_in_pix, r_out_pix)
        m = ann.to_mask(method="center").to_image(rate.shape)
        sel = (m > 0) & good
    elif method == "global":
        yy, xx = np.indices(rate.shape)
        sel = good
        if mask_radius_pix:
            sel = sel & ((xx - x) ** 2 + (yy - y) ** 2 > mask_radius_pix**2)
    else:
        raise ValueError(f"Unknown background method {method!r}")
    vals = rate[sel]
    if vals.size < 10:
        raise ValueError("Too few background pixels")
    _, med, std = sigma_clipped_stats(vals, sigma=sigma)
    return Background(float(med), float(std), int(vals.size), method)


def recenter(rate: np.ndarray, x: float, y: float, box_pix: float, bg: float = 0.0,
             max_iter: int = 10, tol_pix: float = 0.05, max_shift_pix: Optional[float] = None):
    """Iterative centre-of-light within a circular box; returns (x, y, shift)."""
    x0, y0 = x, y
    ny, nx = rate.shape
    for _ in range(max_iter):
        r = int(np.ceil(box_pix))
        ix, iy = int(round(x)), int(round(y))
        x1, x2 = max(ix - r, 0), min(ix + r + 1, nx)
        y1, y2 = max(iy - r, 0), min(iy + r + 1, ny)
        cut = np.nan_to_num(rate[y1:y2, x1:x2] - bg, nan=0.0)
        yy, xx = np.indices(cut.shape)
        circ = (xx + x1 - x) ** 2 + (yy + y1 - y) ** 2 <= box_pix**2
        cut = np.where(circ & (cut > 0), cut, 0.0)
        if cut.sum() <= 0:
            break
        cx, cy = centroid_com(cut)
        nxp, nyp = cx + x1, cy + y1
        step = np.hypot(nxp - x, nyp - y)
        x, y = nxp, nyp
        if step < tol_pix:
            break
    shift = float(np.hypot(x - x0, y - y0))
    if max_shift_pix is not None and shift > max_shift_pix:
        return x0, y0, 0.0
    return float(x), float(y), shift


def _measure(rate, exposure_map, exposure, x, y, radii_pix, bg: Background):
    apertures = [CircularAperture((x, y), r) for r in radii_pix]
    mask = ~np.isfinite(rate)
    tab = aperture_photometry(np.nan_to_num(rate), apertures, mask=mask)
    # Effective exposure per pixel for Poisson errors.
    expmap = exposure_map if exposure_map is not None else np.full(rate.shape, exposure)
    with np.errstate(divide="ignore", invalid="ignore"):
        var_pix = np.where(expmap > 0, np.clip(rate, 0, None) / expmap, 0.0)
    vtab = aperture_photometry(np.nan_to_num(var_pix), apertures, mask=mask)
    out = []
    for k, (ap, r) in enumerate(zip(apertures, radii_pix)):
        area = float(ap.area_overlap(rate, mask=mask))
        total = float(tab[f"aperture_sum_{k}"][0])
        net = total - bg.level * area
        # Pixel noise: Poisson from source+sky counts, or the measured sky
        # scatter if larger (non-Poisson data, read noise). Plus the error in
        # the background level itself, which is fully correlated over the area.
        poisson = float(vtab[f"aperture_sum_{k}"][0])
        var = max(poisson, area * bg.sigma**2) + (area * bg.error_of_mean) ** 2
        out.append((r, area, net, np.sqrt(max(var, 0.0)), area / (np.pi * r**2)))
    return out


def _calibrate(res: ApertureResult, frame, adapter):
    zp = adapter.zero_point(frame)
    if zp is not None and res.rate > 0:
        res.mag = zp - 2.5 * np.log10(res.rate)
        res.mag_err = 2.5 / np.log(10) * res.rate_err / res.rate
    fc = adapter.flux_conversion(frame)
    if fc is not None:
        res.flux_density = res.rate * fc
        res.flux_density_err = res.rate_err * fc
    return res


def _results(frame, adapter, geometry, x, y, measured, bg):
    out = []
    for r, area, net, err, cov in measured:
        r_as = r * frame.pixel_scale_arcsec
        r_km = float(geometry.arcsec_to_km(r_as)) if geometry is not None else float("nan")
        out.append(_calibrate(ApertureResult(x, y, float(r), float(r_as), r_km, area, float(cov), float(net),
                                             float(err), bg.level), frame, adapter))
    return out


def _background_for(rate, x, y, frame, geometry, background, r_max_pix):
    if isinstance(background, Background):
        return background
    background = dict(background or {})
    method = background.pop("method", "annulus")
    unit = background.pop("unit", "pix")
    if method == "annulus":
        r_in = background.pop("r_in", None)
        r_out = background.pop("r_out", None)
        if r_in is None:
            r_in_pix, r_out_pix = 1.5 * r_max_pix, 2.0 * r_max_pix
        else:
            r_in_pix, r_out_pix = to_pixels([r_in, r_out], unit, frame, geometry)
        return estimate_background(rate, x, y, "annulus", r_in_pix, r_out_pix, **background)
    if method == "global":
        mr = background.pop("mask_radius", None)
        mr_pix = to_pixels(mr, unit, frame, geometry)[0] if mr is not None else 1.5 * r_max_pix
        return estimate_background(rate, x, y, "global", mask_radius_pix=mr_pix, **background)
    return estimate_background(rate, x, y, method, **background)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def fixed_aperture_photometry(frame: ImageFrame, x: float, y: float, radii: Sequence[float],
                              unit: str = "arcsec", geometry: Optional[ObservingGeometry] = None,
                              adapter: Optional[InstrumentAdapter] = None,
                              background=None, recenter_box: Optional[float] = None,
                              rate_image: Optional[np.ndarray] = None) -> list:
    """Photometry in a fixed list of apertures.

    ``background`` is a :class:`Background` or a dict such as
    ``{'method': 'annulus', 'r_in': 200, 'r_out': 250, 'unit': 'arcsec'}`` or
    ``{'method': 'global'}``. With no annulus radii the annulus defaults to
    1.5-2x the largest aperture. ``recenter_box`` (same unit as radii)
    turns on centre-of-light recentring before measuring.
    """
    adapter = adapter or adapter_for(frame)
    rate = adapter.count_rate_image(frame) if rate_image is None else rate_image
    radii_pix = to_pixels(radii, unit, frame, geometry)
    bg = _background_for(rate, x, y, frame, geometry, background, float(np.max(radii_pix)))
    if recenter_box:
        box = to_pixels(recenter_box, unit, frame, geometry)[0]
        x, y, _ = recenter(rate, x, y, box, bg.level)
    measured = _measure(rate, frame.exposure_map, frame.exposure, x, y, radii_pix, bg)
    return _results(frame, adapter, geometry, x, y, measured, bg)


def iterative_aperture_photometry(frame: ImageFrame, x: float, y: float,
                                  r_start: float, r_step: float, r_max: float,
                                  unit: str = "arcsec",
                                  geometry: Optional[ObservingGeometry] = None,
                                  adapter: Optional[InstrumentAdapter] = None,
                                  background=None, tol: float = 0.01, n_converge: int = 3,
                                  recenter_box: Optional[float] = None,
                                  max_shift: Optional[float] = None,
                                  rate_image: Optional[np.ndarray] = None) -> GrowthCurve:
    """Grow the aperture until the coma flux converges.

    1. Fix the background (annulus beyond ``r_max`` by default, or as given).
    2. Optionally recentre on the coma's centre of light (``recenter_box``),
       rejecting shifts larger than ``max_shift``.
    3. Step the radius from ``r_start`` by ``r_step``. The flux has
       converged once the fractional change per step stays below ``tol`` for
       ``n_converge`` consecutive steps; the converged radius is the first of
       that run. The loop also stops if the curve turns over (background
       over-subtracted or a neighbouring source) or reaches ``r_max``.
    """
    adapter = adapter or adapter_for(frame)
    rate = adapter.count_rate_image(frame) if rate_image is None else rate_image
    r_max_pix = to_pixels(r_max, unit, frame, geometry)[0]
    bg = _background_for(rate, x, y, frame, geometry, background, r_max_pix)

    shift = 0.0
    if recenter_box:
        box = to_pixels(recenter_box, unit, frame, geometry)[0]
        ms = to_pixels(max_shift, unit, frame, geometry)[0] if max_shift is not None else None
        x, y, shift = recenter(rate, x, y, box, bg.level, max_shift_pix=ms)

    radii = np.arange(r_start, r_max + 0.5 * r_step, r_step)
    radii_pix = to_pixels(radii, unit, frame, geometry)
    measured = _measure(rate, frame.exposure_map, frame.exposure, x, y, radii_pix, bg)
    results = _results(frame, adapter, geometry, x, y, measured, bg)

    converged_index, run, reason, n_used = None, 0, "reached r_max", len(results)
    for k in range(1, len(results)):
        prev, cur = results[k - 1].rate, results[k].rate
        if prev <= 0:
            run = 0
            continue
        frac = (cur - prev) / prev
        if frac < -tol:
            reason, n_used = "growth curve turned over", k + 1
            converged_index = converged_index if converged_index is not None else k - 1
            break
        if abs(frac) < tol:
            run += 1
            if run >= n_converge:
                converged_index = k - n_converge
                reason, n_used = "converged", k + 1
                break
        else:
            run = 0
    return GrowthCurve(results[:n_used], reason == "converged", converged_index, bg, shift, reason)
