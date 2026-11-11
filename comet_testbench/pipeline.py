"""End-to-end test bench: images -> comets in field -> photometry -> OH hand-off."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .field_search import CometInField, find_comets
from .image import ImageFrame, iter_frames
from .instruments import InstrumentAdapter, adapter_for
from .oh import OHModel, fit_production_rate, subtract_continuum
from .photometry import GrowthCurve, fixed_aperture_photometry, iterative_aperture_photometry

log = logging.getLogger(__name__)


@dataclass
class BenchConfig:
    """Settings for one test-bench run. All radii units are explicit."""

    candidates: Sequence[str] = ()
    use_header_object: bool = True
    use_sb_ident: bool = True
    edge_margin_pix: float = 0.0
    # Fixed apertures
    fixed_radii: Sequence[float] = (5000, 10000, 20000, 40000)
    fixed_unit: str = "km"
    # Iterative apertures
    iter_start: float = 3.0
    iter_step: float = 3.0
    iter_max: float = 150.0
    iter_unit: str = "arcsec"
    iter_tol: float = 0.01
    iter_n_converge: int = 3
    # Background, e.g. {"method": "annulus", "r_in": 200, "r_out": 300, "unit": "arcsec"}
    background: dict = field(default_factory=lambda: {"method": "global"})
    # Recentring (arcsec); None turns it off
    recenter_box_arcsec: Optional[float] = 10.0
    max_shift_arcsec: Optional[float] = 20.0

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text()))


@dataclass
class CometMeasurement:
    frame: ImageFrame
    comet: CometInField
    fixed: list
    growth: Optional[GrowthCurve]
    error: str = ""

    @property
    def filter(self):
        return self.frame.filter


def _from_arcsec(value, unit, frame, comet):
    if value is None:
        return None
    if unit == "arcsec":
        return value
    if unit == "km":
        return float(comet.geometry.arcsec_to_km(value))
    return value / frame.pixel_scale_arcsec


def measure_frame(frame: ImageFrame, cfg: BenchConfig, adapter: Optional[InstrumentAdapter] = None,
                  ephemeris=None, sb_ident=None) -> list:
    """Find every comet in one frame and run fixed + iterative photometry on it."""
    adapter = adapter or adapter_for(frame)
    comets = find_comets(frame, cfg.candidates, adapter, cfg.use_header_object, cfg.use_sb_ident,
                         edge_margin_pix=cfg.edge_margin_pix, ephemeris=ephemeris, sb_ident=sb_ident)
    if not comets:
        return []
    rate = adapter.count_rate_image(frame)
    out = []
    for c in comets:
        bg = dict(cfg.background)
        try:
            # Recentre once with the iterative run; reuse that centre for fixed apertures.
            growth = iterative_aperture_photometry(
                frame, c.x, c.y, cfg.iter_start, cfg.iter_step, cfg.iter_max, cfg.iter_unit,
                c.geometry, adapter, dict(bg), cfg.iter_tol, cfg.iter_n_converge,
                recenter_box=_from_arcsec(cfg.recenter_box_arcsec, cfg.iter_unit, frame, c),
                max_shift=_from_arcsec(cfg.max_shift_arcsec, cfg.iter_unit, frame, c),
                rate_image=rate)
            x, y = growth.apertures[0].x, growth.apertures[0].y
            fixed = fixed_aperture_photometry(frame, x, y, cfg.fixed_radii, cfg.fixed_unit,
                                              c.geometry, adapter, growth.background,
                                              rate_image=rate)
            out.append(CometMeasurement(frame, c, fixed, growth))
        except Exception as exc:
            log.warning("Photometry failed for %s in %s: %s", c.designation, frame.source, exc)
            out.append(CometMeasurement(frame, c, [], None, str(exc)))
    return out


def run(images: Sequence, cfg: BenchConfig, adapters=None, ephemeris=None, sb_ident=None) -> list:
    """Run the bench on a list of images.

    ``images`` items are FITS paths or ``(sky_path, exposure_map_path)`` pairs.
    """
    results = []
    for item in images:
        sky, exp = (item, None) if isinstance(item, (str, Path)) else (item[0], item[1])
        for frame in iter_frames(sky, exp):
            adapter = adapter_for(frame, adapters) if adapters else adapter_for(frame)
            res = measure_frame(frame, cfg, adapter, ephemeris, sb_ident)
            log.info("%s (%s): %d comet(s)", frame.source, frame.filter, len(res))
            results.extend(res)
    return results


# --------------------------------------------------------------------------
# OH hand-off
# --------------------------------------------------------------------------

def pair_for_oh(results: Sequence[CometMeasurement], oh_filter="UVW1", continuum_filter="V",
                max_dt_hours: float = 2.0):
    """Pair each OH-band measurement with the nearest-in-time continuum one."""
    pairs = []
    for m in results:
        if m.filter != oh_filter or not m.fixed:
            continue
        best, best_dt = None, None
        for n in results:
            if (n.filter != continuum_filter or not n.fixed
                    or n.comet.designation != m.comet.designation):
                continue
            dt = abs((n.frame.mid_time - m.frame.mid_time).to_value("hour"))
            if dt <= max_dt_hours and (best_dt is None or dt < best_dt):
                best, best_dt = n, dt
        if best is not None:
            pairs.append((m, best, best_dt))
    return pairs


def oh_measurements(results, beta: float, oh_conversion: float, **pair_kw):
    """Continuum-subtracted OH fluxes for every UVW1/V pair (fixed apertures)."""
    out = []
    for w1, v, dt in pair_for_oh(results, **pair_kw):
        out.append(subtract_continuum(w1.fixed, v.fixed, beta, oh_conversion, w1.comet.geometry,
                                      meta={"oh_frame": w1.frame.source,
                                            "continuum_frame": v.frame.source, "dt_hours": dt}))
    return out


def fit_all(model: OHModel, measurements):
    return [(m, fit_production_rate(model, m)) for m in measurements]


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def _write(path, rows):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_results(results: Sequence[CometMeasurement], out_dir, oh=None, fits_=None):
    """Write comets.csv, fixed_apertures.csv, growth_curves.csv (and OH tables)."""
    out_dir = Path(out_dir)

    def base(m):
        return {"frame": m.frame.source, "filter": m.frame.filter,
                "instrument": m.frame.instrument, "mid_time_utc": m.frame.mid_time.utc.isot,
                "exposure_s": m.frame.exposure}

    _write(out_dir / "comets.csv",
           ({**base(m), **m.comet.row(),
             "growth_converged": m.growth.converged if m.growth else None,
             "growth_stop": m.growth.stop_reason if m.growth else m.error,
             "converged_radius_arcsec": m.growth.best.radius_arcsec if m.growth else None,
             "converged_rate": m.growth.best.rate if m.growth else None,
             "centroid_shift_pix": m.growth.centroid_shift_pix if m.growth else None}
            for m in results))
    _write(out_dir / "fixed_apertures.csv",
           ({**base(m), "designation": m.comet.designation, **a.row()} for m in results for a in m.fixed))
    _write(out_dir / "growth_curves.csv",
           ({**base(m), "designation": m.comet.designation, "step": i, **a.row()}
            for m in results if m.growth for i, a in enumerate(m.growth.apertures)))
    if oh:
        _write(out_dir / "oh_fluxes.csv",
               ({"designation": o.geometry.target, "time_jd_utc": o.geometry.time_jd_utc,
                 "r_h_au": o.geometry.r_h, "delta_au": o.geometry.delta, "radius_km": r,
                 "oh_rate": c, "oh_rate_err": ce, "oh_flux": f, "oh_flux_err": fe,
                 **{k: v for k, v in o.meta.items()}}
                for o in oh for r, c, ce, f, fe in
                zip(o.radius_km, o.oh_rate, o.oh_rate_err, o.oh_flux, o.oh_flux_err)))
    if fits_:
        _write(out_dir / "production_rates.csv",
               ({"designation": o.geometry.target, "time_jd_utc": o.geometry.time_jd_utc,
                 "r_h_au": o.geometry.r_h, "model": f.model_name, "q": f.q, "q_err": f.q_err,
                 "chi2": f.chi2, "dof": f.dof,
                 "q_per_aperture": ";".join(f"{q:.4g}" for q in f.q_per_aperture)}
                for o, f in fits_))
