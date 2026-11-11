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
                        normalize_designation, sb_ident_comets)
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

    def row(self):
        g = self.geometry
        return {
            "designation": self.designation, "x": self.x, "y": self.y,
            "ra": g.ra, "dec": g.dec, "r_h_au": g.r_h, "r_dot_kms": g.r_dot,
            "delta_au": g.delta, "phase_deg": g.phase, "trail_pix": self.trail_pix,
            "found_by": self.found_by,
        }


def _field_box(frame: ImageFrame, margin_deg: float):
    """Centre and half-widths (deg) of a box enclosing the image footprint."""
    ra0, dec0 = frame.center()
    corners = SkyCoord(frame.footprint() * u.deg)
    center = SkyCoord(ra0 * u.deg, dec0 * u.deg)
    off = center.spherical_offsets_to(corners)
    hw_ra = float(np.max(np.abs(off[0].deg))) + margin_deg
    hw_dec = float(np.max(np.abs(off[1].deg))) + margin_deg
    return ra0, dec0, hw_ra, hw_dec


def find_comets(frame: ImageFrame,
                candidates: Sequence[str] = (),
                adapter: Optional[InstrumentAdapter] = None,
                use_header_object: bool = True,
                use_sb_ident: bool = True,
                sb_ident_margin_deg: float = 0.1,
                edge_margin_pix: float = 0.0,
                ephemeris: Optional[EphemerisProvider] = None,
                sb_ident=None) -> list:
    """Comets whose predicted position lands on exposed pixels of ``frame``.

    Parameters
    ----------
    candidates : designations to test regardless of sb_ident (e.g. ``['46P']``).
    adapter : instrument adapter; chosen automatically when None.
    use_header_object : also test the comet named in the OBJECT keyword.
    use_sb_ident : run a JPL sb_ident discovery query over the footprint.
    sb_ident_margin_deg : extra half-width for the sb_ident box. It covers
        geocentric-vs-spacecraft parallax and sb_ident propagation error.
    edge_margin_pix : require the comet to be at least this far inside.
    ephemeris, sb_ident : injectable callables (used by offline tests).
    """
    adapter = adapter or adapter_for(frame)
    ephemeris = ephemeris or horizons_provider()
    sb_ident = sb_ident or sb_ident_comets

    named = []
    if use_header_object:
        des = normalize_designation(frame.header.get("OBJECT", ""))
        if des:
            named.append((des, "header"))
    for c in candidates:
        des = normalize_designation(c) or c
        named.append((des, "candidate list"))

    if use_sb_ident:
        ra0, dec0, hw_ra, hw_dec = _field_box(frame, sb_ident_margin_deg)
        try:
            for des in sb_ident(ra0, dec0, hw_ra, hw_dec, frame.mid_time,
                                mpc_code=adapter.mpc_code(frame)):
                named.append((des, "sb_ident"))
        except Exception as exc:  # network or service errors should not kill a run
            log.warning("sb_ident query failed for %s: %s", frame.source, exc)

    seen, found = set(), []
    observer = adapter.observer(frame)
    for des, how in named:
        if des in seen:
            continue
        seen.add(des)
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
        found.append(CometInField(des, x, y, geom, how, trail))
    return found
