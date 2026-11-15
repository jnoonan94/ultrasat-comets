"""Benchmark metrics for the comet search and photometry.

These are the checks used to show that the search built for the UltraSat field
of regard carries over to real images. They take an
:class:`~comet_testbench.image.ImageFrame` and the bench's own outputs, so the
same functions run on Swift/UVOT archive frames now and on downlinked UltraSat
frames later.

* :func:`ephemeris_offset`: where the ephemeris put the comet versus where the
  light is, split into along-track and cross-track components.
* :func:`inject_coma` and :func:`completeness_curve`: plant synthetic comae of
  known brightness in a real background and measure the recovery rate.
* :func:`aperture_stability`: how much the coma count rate moves with
  aperture size and background choice.

Nothing here is calibration: all fluxes are count rates on the adapter's
count-rate image.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np

from .field_search import CometInField
from .image import ImageFrame
from .instruments import InstrumentAdapter, adapter_for
from .photometry import Background, estimate_background, fixed_aperture_photometry, recenter


# --------------------------------------------------------------------------
# 1. Ephemeris recovery
# --------------------------------------------------------------------------

def motion_direction_pix(frame: ImageFrame, comet: CometInField):
    """Unit vector (dx, dy) of the comet's apparent motion on the detector.

    Returns (nan, nan) when the ephemeris has no rates (e.g. sb_ident only).
    """
    g = comet.geometry
    if not (np.isfinite(g.ra_rate) and np.isfinite(g.dec_rate)) or (g.ra_rate == 0 and g.dec_rate == 0):
        return float("nan"), float("nan")
    dt_h = 0.01
    dra = g.ra_rate * dt_h / 3600 / np.cos(np.deg2rad(g.dec))
    ddec = g.dec_rate * dt_h / 3600
    x1, y1 = frame.wcs.world_to_pixel_values(g.ra + dra, g.dec + ddec)
    v = np.array([float(x1) - comet.x, float(y1) - comet.y])
    n = np.hypot(*v)
    return float(v[0] / n), float(v[1] / n)


@dataclass
class EphemerisOffset:
    designation: str
    frame: str
    filter: str
    x_pred: float
    y_pred: float
    x_obs: float
    y_obs: float
    offset_pix: float
    offset_arcsec: float
    along_track_arcsec: float  # + is ahead of the prediction
    cross_track_arcsec: float
    trail_pix: float
    peak_snr: float  # centroid-box signal over background noise

    def row(self):
        return asdict(self)


def ephemeris_offset(frame: ImageFrame, comet: CometInField, box_arcsec: float = 10.0,
                     adapter: Optional[InstrumentAdapter] = None,
                     rate_image: Optional[np.ndarray] = None,
                     background: Optional[Background] = None) -> EphemerisOffset:
    """Offset between the predicted position and the coma's centre of light.

    The centroid is an iterative centre of light in a ``box_arcsec`` circle
    started at the prediction, on the background-subtracted count-rate image.
    ``peak_snr`` says whether there was anything to centroid on; treat offsets
    with low SNR as unmeasured, not as misses.
    """
    adapter = adapter or adapter_for(frame)
    rate = adapter.count_rate_image(frame) if rate_image is None else rate_image
    box = box_arcsec / frame.pixel_scale_arcsec
    if background is None:
        background = estimate_background(rate, comet.x, comet.y, "global",
                                         mask_radius_pix=max(3 * box, 20))
    x, y, _ = recenter(rate, comet.x, comet.y, box, background.level)
    dx, dy = x - comet.x, y - comet.y
    ux, uy = motion_direction_pix(frame, comet)
    s = frame.pixel_scale_arcsec
    along = (dx * ux + dy * uy) * s if np.isfinite(ux) else float("nan")
    cross = (-dx * uy + dy * ux) * s if np.isfinite(ux) else float("nan")
    m = fixed_aperture_photometry(frame, x, y, [box], "pix", adapter=adapter,
                                  background=background, rate_image=rate)[0]
    snr = m.rate / m.rate_err if m.rate_err > 0 else float("nan")
    return EphemerisOffset(comet.designation, frame.source, frame.filter, comet.x, comet.y, x, y,
                           float(np.hypot(dx, dy)), float(np.hypot(dx, dy) * s), float(along),
                           float(cross), comet.trail_pix, float(snr))


# --------------------------------------------------------------------------
# 4. Faint-limit recovery (synthetic comae in real backgrounds)
# --------------------------------------------------------------------------

def coma_model(shape, x0: float, y0: float, rate_in_r: float, r_ref_pix: float,
               r_max_pix: float, profile: str = "rho") -> np.ndarray:
    """Count-rate image of a synthetic coma.

    ``profile='rho'``: surface brightness proportional to 1/rho (a steady-state
    coma), normalised so the rate inside ``r_ref_pix`` is ``rate_in_r``, and
    cut off at ``r_max_pix``. ``profile='gaussian'``: a Gaussian with sigma
    ``r_ref_pix`` whose total rate is ``rate_in_r``.
    """
    ny, nx = shape
    r = int(np.ceil(r_max_pix)) + 2
    x1, x2 = max(int(x0) - r, 0), min(int(x0) + r + 1, nx)
    y1, y2 = max(int(y0) - r, 0), min(int(y0) + r + 1, ny)
    out = np.zeros(shape)
    if x1 >= x2 or y1 >= y2:
        return out
    # 5x5 sub-sampling so the cusp is integrated, not point-sampled.
    sub = (np.arange(5) + 0.5) / 5 - 0.5
    yy, xx = np.mgrid[y1:y2, x1:x2].astype(float)
    acc = np.zeros(yy.shape)
    for sy in sub:
        for sx in sub:
            rho = np.hypot(xx + sx - x0, yy + sy - y0)
            if profile == "rho":
                acc += np.where(rho <= r_max_pix, 1.0 / np.maximum(rho, 0.05), 0.0)
            elif profile == "gaussian":
                acc += np.exp(-0.5 * (rho / r_ref_pix) ** 2)
            else:
                raise ValueError(f"Unknown profile {profile!r}")
    acc /= sub.size**2
    if profile == "rho":
        amp = rate_in_r / (2 * np.pi * r_ref_pix)  # rate inside r is 2*pi*amp*r
    else:
        amp = rate_in_r / (2 * np.pi * r_ref_pix**2)
    out[y1:y2, x1:x2] = amp * acc
    return out


def inject_coma(frame: ImageFrame, x0: float, y0: float, rate_in_r: float, r_ref_pix: float,
                r_max_pix: Optional[float] = None, profile: str = "rho",
                rng: Optional[np.random.Generator] = None) -> ImageFrame:
    """Copy of ``frame`` with a Poisson-sampled synthetic coma added.

    The model is converted to counts with the exposure map (or EXPOSURE), so
    it only works on frames whose data are counts. Frames that are already a
    rate (BUNIT counts/s) get the noiseless model added.
    """
    rng = rng or np.random.default_rng()
    r_max_pix = r_max_pix if r_max_pix is not None else 4 * r_ref_pix
    model = coma_model(frame.shape, x0, y0, rate_in_r, r_ref_pix, r_max_pix, profile)
    new = copy.copy(frame)
    bunit = str(frame.header.get("BUNIT", "")).lower().replace(" ", "")
    if bunit.endswith("/s"):
        new.data = frame.data + model
        return new
    exp = frame.exposure_map if frame.exposure_map is not None else np.full(frame.shape, frame.exposure)
    counts = rng.poisson(np.clip(model * np.nan_to_num(exp), 0, None))
    new.data = frame.data + counts
    return new


@dataclass
class Recovery:
    rate_in_r: float  # injected count rate inside r_ref
    x: float
    y: float
    recovered: bool
    snr: float
    offset_pix: float
    measured_rate: float


def recover(frame: ImageFrame, x0: float, y0: float, r_ref_pix: float,
            adapter: Optional[InstrumentAdapter] = None, snr_min: float = 5.0,
            max_offset_pix: Optional[float] = None, box_pix: Optional[float] = None,
            rate_image: Optional[np.ndarray] = None,
            background: Optional[Background] = None):
    """Recentre near (x0, y0) and decide whether a coma is detected there.

    Detected means SNR >= ``snr_min`` in an ``r_ref_pix`` aperture at the
    recentred position *and* the centroid within ``max_offset_pix`` of the
    start (default: ``r_ref_pix``), so a nearby star does not count.
    Returns (recovered, snr, offset_pix, rate).
    """
    adapter = adapter or adapter_for(frame)
    rate = adapter.count_rate_image(frame) if rate_image is None else rate_image
    box = box_pix if box_pix is not None else r_ref_pix
    max_off = max_offset_pix if max_offset_pix is not None else r_ref_pix
    if background is None:
        background = estimate_background(rate, x0, y0, "annulus", 4 * r_ref_pix, 6 * r_ref_pix)
    x, y, shift = recenter(rate, x0, y0, box, background.level)
    m = fixed_aperture_photometry(frame, x, y, [r_ref_pix], "pix", adapter=adapter,
                                  background=background, rate_image=rate)[0]
    snr = m.rate / m.rate_err if m.rate_err > 0 else 0.0
    return bool(snr >= snr_min and shift <= max_off), float(snr), float(shift), float(m.rate)


def completeness_curve(frame: ImageFrame, positions: Sequence, rates: Sequence[float],
                       r_ref_pix: float, adapter: Optional[InstrumentAdapter] = None,
                       profile: str = "rho", snr_min: float = 5.0, seed: int = 0,
                       control: bool = True):
    """Fraction of injected comae recovered at each brightness.

    Each (position, rate) pair is injected on its own into a fresh copy of
    ``frame``, so injections never overlap. Positions should be on empty sky
    away from the real comet. With ``control=True`` each position is also
    measured *without* an injection; those false positives are returned so
    the curve can be read against the field's own detection rate.

    Returns ``(table, false_positive_rate)`` where ``table`` is a list of dicts
    with ``rate_in_r``, ``n``, ``n_recovered``, ``completeness`` and
    ``median_snr``, plus the per-injection :class:`Recovery` list.
    """
    adapter = adapter or adapter_for(frame)
    rng = np.random.default_rng(seed)
    trials, table = [], []
    for rate in rates:
        rec = []
        for (x, y) in positions:
            f = inject_coma(frame, x, y, rate, r_ref_pix, profile=profile, rng=rng)
            ok, snr, off, meas = recover(f, x, y, r_ref_pix, adapter, snr_min)
            rec.append(Recovery(float(rate), float(x), float(y), ok, snr, off, meas))
        trials += rec
        table.append({"rate_in_r": float(rate), "n": len(rec),
                      "n_recovered": sum(r.recovered for r in rec),
                      "completeness": sum(r.recovered for r in rec) / len(rec),
                      "median_snr": float(np.median([r.snr for r in rec]))})
    fp = float("nan")
    if control:
        base = adapter.count_rate_image(frame)
        fp = float(np.mean([recover(frame, x, y, r_ref_pix, adapter, snr_min, rate_image=base)[0]
                            for (x, y) in positions]))
    return table, fp, trials


def blank_sky_positions(frame: ImageFrame, n: int, avoid: Sequence = (), avoid_radius_pix: float = 100.0,
                        edge_pix: float = 50.0, min_spacing_pix: float = 0.0,
                        max_star_rate: Optional[float] = None, r_check_pix: float = 10.0,
                        adapter: Optional[InstrumentAdapter] = None, seed: int = 0) -> list:
    """Random exposed positions away from ``avoid`` points and the frame edge.

    With ``max_star_rate`` set, positions whose ``r_check_pix`` aperture already
    holds more than that net rate (a star) are skipped.
    """
    adapter = adapter or adapter_for(frame)
    rate = adapter.count_rate_image(frame)
    bg = None
    if max_star_rate is not None:
        bg = estimate_background(rate, *[(s - 1) / 2 for s in frame.shape[::-1]], "global")
    rng = np.random.default_rng(seed)
    ny, nx = frame.shape
    out = []
    for _ in range(200 * n):
        if len(out) >= n:
            break
        x, y = rng.uniform(edge_pix, nx - edge_pix), rng.uniform(edge_pix, ny - edge_pix)
        if any(np.hypot(x - ax, y - ay) < avoid_radius_pix for ax, ay in avoid):
            continue
        if any(np.hypot(x - ox, y - oy) < min_spacing_pix for ox, oy in out):
            continue
        # Whole check aperture must be exposed.
        if not all(frame.contains_pixel(x + dx, y + dy) for dx, dy in
                   [(0, 0), (r_check_pix, 0), (-r_check_pix, 0), (0, r_check_pix), (0, -r_check_pix)]):
            continue
        if max_star_rate is not None:
            m = fixed_aperture_photometry(frame, x, y, [r_check_pix], "pix", adapter=adapter,
                                          background=bg, rate_image=rate)[0]
            if m.rate > max_star_rate:
                continue
        out.append((float(x), float(y)))
    return out


def fifty_percent_rate(table) -> float:
    """Injected rate at 50% completeness, by log-linear interpolation (nan if not bracketed)."""
    rates = np.array([r["rate_in_r"] for r in table])
    comp = np.array([r["completeness"] for r in table])
    order = np.argsort(rates)
    rates, comp = rates[order], comp[order]
    for k in range(1, len(rates)):
        if comp[k - 1] < 0.5 <= comp[k]:
            f = (0.5 - comp[k - 1]) / (comp[k] - comp[k - 1])
            return float(10 ** (np.log10(rates[k - 1]) + f * (np.log10(rates[k]) - np.log10(rates[k - 1]))))
    return float("nan")


# --------------------------------------------------------------------------
# 5. Aperture photometry stability
# --------------------------------------------------------------------------

@dataclass
class StabilityReport:
    radii_arcsec: list
    rates: list  # rate at each radius, reference background
    rate_errs: list
    worst_step_sigma: float  # largest drop between adjacent radii, in sigma
    background_rates: dict  # label -> rate at r_test
    background_spread: float  # (max - min) / median of background_rates
    r_test_arcsec: float

    def row(self):
        return asdict(self)


def aperture_stability(frame: ImageFrame, x: float, y: float, radii_arcsec: Sequence[float],
                       backgrounds: dict, r_test_arcsec: Optional[float] = None,
                       reference: Optional[str] = None,
                       adapter: Optional[InstrumentAdapter] = None,
                       rate_image: Optional[np.ndarray] = None) -> StabilityReport:
    """Check that the coma rate behaves smoothly with aperture and background.

    * Along the radius list (with the ``reference`` background) the enclosed
      rate of a coma should not *fall* by more than its noise between steps;
      ``worst_step_sigma`` is the largest drop in units of the step's error.
    * Across background choices (``backgrounds``: label -> background dict as
      accepted by :func:`fixed_aperture_photometry`), the rate at
      ``r_test_arcsec`` should agree; ``background_spread`` is the fractional
      range.
    """
    adapter = adapter or adapter_for(frame)
    rate = adapter.count_rate_image(frame) if rate_image is None else rate_image
    reference = reference or next(iter(backgrounds))
    r_test = r_test_arcsec if r_test_arcsec is not None else float(np.median(radii_arcsec))
    ref = fixed_aperture_photometry(frame, x, y, list(radii_arcsec), "arcsec", adapter=adapter,
                                    background=dict(backgrounds[reference]), rate_image=rate)
    rates = [a.rate for a in ref]
    errs = [a.rate_err for a in ref]
    worst = 0.0
    for k in range(1, len(rates)):
        # Noise of the annulus between the two radii (outer error dominates).
        step_err = np.sqrt(max(errs[k] ** 2 - errs[k - 1] ** 2, 0.0)) or errs[k]
        drop = (rates[k - 1] - rates[k]) / step_err if step_err > 0 else 0.0
        worst = max(worst, float(drop))
    by_bg = {}
    for label, bgd in backgrounds.items():
        a = fixed_aperture_photometry(frame, x, y, [r_test], "arcsec", adapter=adapter,
                                      background=dict(bgd), rate_image=rate)[0]
        by_bg[label] = a.rate
    vals = np.array(list(by_bg.values()))
    spread = float((vals.max() - vals.min()) / abs(np.median(vals))) if np.median(vals) != 0 else float("inf")
    return StabilityReport(list(map(float, radii_arcsec)), rates, errs, worst, by_bg, spread, float(r_test))


# --------------------------------------------------------------------------
# Swift/UVOT benchmark set
# --------------------------------------------------------------------------

# obsid -> (comet, filter, why it is in the set). All public in the HEASARC archive.
SWIFT_BENCHMARK_FRAMES = {
    "00094318002": ("46P", "V", "46P before perigee, moderate motion"),
    "00094319001": ("46P", "UVW1", "46P before perigee, moderate motion"),
    "00094421002": ("46P", "V", "46P three days before perigee (0.08 au), fastest motion"),
    "00094422001": ("46P", "UVW1", "46P near perigee; exposure is under half the elapsed time"),
    "00094431002": ("46P", "V", "46P after perigee; bright star ~35 arcsec from the coma"),
    "00094432001": ("46P", "UVW1", "46P after perigee; bright star ~20 arcsec from the coma"),
    "00010086001": ("41P", "UVW1", "41P, slow motion, binned 1 arcsec pixels, star field"),
    "00010086002": ("41P", "V", "41P, slow motion, binned 1 arcsec pixels, star field"),
}

# Archive visits labelled as 41P that point at the Galactic Centre: no comet in the field.
SWIFT_NULL_FRAMES = {
    "00093389002": ("41P", "UVW1", "labelled COMET41P, pointed at the Galactic Centre"),
    "00093389003": ("41P", "UVW1", "labelled COMET41P, pointed at the Galactic Centre"),
}

_SEARCH = {"46P": ("%WIRTANEN%", "2018-11-27", "2019-01-14"),
           "41P": ("%41P%", "2017-04-12", "2017-05-08")}


def fetch_swift_benchmark(dest="data/swift", frames=None, session=None) -> list:
    """Download the benchmark frames; returns (sky, exposure_map, obsid, filter) tuples."""
    from .swift_archive import download_images, find_observations

    frames = frames if frames is not None else {**SWIFT_BENCHMARK_FRAMES, **SWIFT_NULL_FRAMES}
    out = []
    for comet in sorted({c for c, _, _ in frames.values()}):
        pattern, start, stop = _SEARCH[comet]
        for obs in find_observations(pattern, start, stop, session=session):
            if obs.obsid in frames and frames[obs.obsid][0] == comet:
                for sky, ex, o, filt in download_images([obs], [frames[obs.obsid][1]], dest,
                                                        session=session):
                    out.append((sky, ex, o.obsid, filt))
    return out


def timing_tolerance_arcsec(frame: ImageFrame, comet: CometInField) -> float:
    """Position error expected from timing alone.

    The ephemeris is evaluated at the midpoint of DATE-OBS/DATE-END. When the
    exposure covers only part of that span, the light-weighted time can sit up
    to half the dead time either side. A trailed coma also centroids anywhere
    along its trail if the exposure was not uniform; allow a quarter of it.
    """
    g = comet.geometry
    rate = np.hypot(g.ra_rate, g.dec_rate) / 3600  # arcsec/s
    if not np.isfinite(rate):
        return float("nan")
    span = float(frame.header.get("TELAPSE", frame.exposure))
    dead = max(span - frame.exposure, 0.0)
    trail = comet.trail_pix * frame.pixel_scale_arcsec if np.isfinite(comet.trail_pix) else 0.0
    return float(rate * dead / 2 + 0.25 * trail)
