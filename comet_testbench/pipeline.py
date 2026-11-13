"""End-to-end test bench: images -> comets in field -> photometry -> hand-off file."""

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
from .oh import write_handoff
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


def write_results(results: Sequence[CometMeasurement], out_dir, handoff: bool = True):
    """Write comets.csv, fixed_apertures.csv, growth_curves.csv and the hand-off files.

    Hand-off files (one ECSV per comet, see :mod:`comet_testbench.oh`) go to
    ``out_dir/handoff`` and are what the OH/H2O package reads.
    """
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
    if handoff:
        return write_handoff(results, out_dir / "handoff")
    return []
