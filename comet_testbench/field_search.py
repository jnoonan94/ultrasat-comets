"""Find comets inside an image from its WCS and observation time.

Works on any :class:`~comet_testbench.image.ImageFrame`, whatever the filter
or instrument. Candidates come from two places and are merged:

1. named candidates: the header OBJECT keyword (if it looks like a comet) plus
   any designations passed in, and
2. a JPL sb_ident query over the image footprint (optional; needs network).

Every candidate is then placed with a Horizons ephemeris for the actual
observer and kept only if it falls on exposed pixels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from .ephemeris import (EphemerisProvider, ObservingGeometry, horizons_provider,
                        normalize_designation, sb_ident_query)
from .image import ImageFrame
from .instruments import InstrumentAdapter, adapter_for

log = logging.getLogger(__name__)


@dataclass
class CometInField:
    designation: str
    x: float  # 0-based pixel column of the predicted position
    y: float  # 0-based pixel row
    geometry: ObservingGeometry
    found_by: str  # 'header', 'candidate list', 'sb_ident'
    trail_pix: float  # predicted motion during the exposure, pixels
    vmag: float = float("nan")  # sb_ident predicted total V magnitude, if any

    def row(self):
        g = self.geometry
        return {
            "designation": self.designation, "x": self.x, "y": self.y,
            "ra": g.ra, "dec": g.dec, "r_h_au": g.r_h, "r_dot_kms": g.r_dot,
            "delta_au": g.delta, "phase_deg": g.phase, "trail_pix": self.trail_pix, "vmag": self.vmag,
            "found_by": self.found_by,
        }


def _field_box(frame: ImageFrame, margin_deg: float, n_edge: int = 20):
    """Centre and half-widths (deg) of an RA/Dec box enclosing the image.

    The perimeter is sampled (not just the corners, since edge midpoints of a
    wide TAN field bulge outward) and the extents are taken in RA and Dec
    *coordinates*, which is never narrower than the on-sky extent. The pixel
    test in :func:`find_comets` trims anything outside the real footprint.
    """
    ny, nx = frame.data.shape
    t = np.linspace(0, 1, n_edge)
    xs = np.concatenate([t * (nx - 1), np.full(n_edge, nx - 1), t[::-1] * (nx - 1), np.zeros(n_edge)])
    ys = np.concatenate([np.zeros(n_edge), t * (ny - 1), np.full(n_edge, ny - 1), t[::-1] * (ny - 1)])
    ra, dec = frame.wcs.celestial.pixel_to_world_values(xs - 0.5 * np.sign(xs - (nx - 1) / 2),
                                                        ys - 0.5 * np.sign(ys - (ny - 1) / 2))
    ra0, dec0 = frame.center()
    dra = (np.asarray(ra) - ra0 + 180) % 360 - 180
    hw_ra = min(float(np.max(np.abs(dra))) + margin_deg, 180.0)
    hw_dec = float(np.max(np.abs(np.asarray(dec) - dec0))) + margin_deg
    return ra0, dec0, hw_ra, hw_dec


def find_comets(frame: ImageFrame,
                candidates: Sequence[str] = (),
                adapter: Optional[InstrumentAdapter] = None,
                use_header_object: bool = True,
                use_sb_ident: bool = True,
                sb_ident_margin_deg: float = 0.1,
                edge_margin_pix: float = 0.0,
                ephemeris: Optional[EphemerisProvider] = None,
                sb_ident=None,
                refine_with_horizons: bool = True,
                vmag_lim: float = 25.0,
                no_magnitude: str = "periodic") -> list:
    """Comets whose predicted position lands on exposed pixels of ``frame``.

    The same function serves real images and simulated UltraSat pointings
    (:func:`comet_testbench.pointing.virtual_frame`), so the field-of-regard
    method and the image search share one code path.

    Parameters
    ----------
    candidates : designations to test regardless of sb_ident (e.g. ``['46P']``).
    adapter : instrument adapter; chosen automatically when None.
    use_header_object : also test the comet named in the OBJECT keyword.
    use_sb_ident : run a JPL sb_ident discovery query over the footprint.
    sb_ident_margin_deg : extra half-width for the sb_ident box. It covers
        observer parallax and sb_ident propagation error.
    edge_margin_pix : require the comet to be at least this far inside.
    refine_with_horizons : place every candidate with a Horizons ephemeris for
        the real observer (needed for photometry). With False, sb_ident
        positions are used as they are, as the simulator does; named
        candidates not returned by sb_ident still go to Horizons.
    vmag_lim, no_magnitude : passed to :func:`~comet_testbench.ephemeris.sb_ident_query`.
        ``no_magnitude='none'`` with the simulator's ``v_lim`` reproduces its
        selection exactly.
    ephemeris, sb_ident : injectable callables (used by offline tests).
    """
    adapter = adapter or adapter_for(frame)
    ephemeris = ephemeris or horizons_provider()
    sb_ident = sb_ident or sb_ident_query

    named = []  # (designation, how, SbIdentRecord or None)
    if use_header_object:
        des = normalize_designation(frame.header.get("OBJECT", ""))
        if des:
            named.append((des, "header", None))
    for c in candidates:
        des = normalize_designation(c) or c
        named.append((des, "candidate list", None))

    if use_sb_ident:
        ra0, dec0, hw_ra, hw_dec = _field_box(frame, sb_ident_margin_deg)
        site = adapter.sb_ident_site(frame)
        loc = {"site": site} if site is not None else {"mpc_code": adapter.mpc_code(frame)}
        try:
            for rec in sb_ident(ra0, dec0, hw_ra, hw_dec, frame.mid_time,
                                vmag_lim=vmag_lim, no_magnitude=no_magnitude, **loc):
                if isinstance(rec, str):
                    named.append((rec, "sb_ident", None))
                else:
                    named.append((rec.designation, "sb_ident", rec))
        except Exception as exc:  # network or service errors should not kill a run
            log.warning("sb_ident query failed for %s: %s", frame.source, exc)

    records = {des: rec for des, _, rec in named if rec is not None}
    seen, found = set(), []
    observer = adapter.observer(frame)
    for des, how, _ in named:
        if des in seen:
            continue
        seen.add(des)
        rec = records.get(des)
        if rec is not None and not refine_with_horizons:
            geom = ObservingGeometry(des, float(frame.mid_time.utc.jd), rec.ra, rec.dec,
                                     *([float("nan")] * 5), rec.ra_rate, rec.dec_rate,
                                     observer="sb_ident")
        else:
            try:
                geom = ephemeris(des, frame.mid_time, observer)
            except Exception as exc:
                log.warning("No ephemeris for %s at %s: %s", des, frame.source, exc)
                continue
        x, y = frame.wcs.world_to_pixel_values(geom.ra, geom.dec)
        x, y = float(x), float(y)
        if not frame.contains_pixel(x, y, margin=edge_margin_pix):
            continue
        rate = np.hypot(geom.ra_rate, geom.dec_rate)  # arcsec / hour
        span = float(frame.header.get("TELAPSE", frame.exposure))
        trail = rate * span / 3600 / frame.pixel_scale_arcsec if np.isfinite(rate) else float("nan")
        c = CometInField(des, x, y, geom, how, trail)
        c.vmag = rec.vmag if rec is not None else float("nan")
        found.append(c)
    return found
