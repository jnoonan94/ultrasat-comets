"""Run the field-of-regard search through the same code as real images.

The field-of-regard simulator (``ultrasat_object_search.py``) asks JPL sb_ident
for comets in a box around each pointing centre. Here a pointing becomes a
*virtual frame*: an :class:`~comet_testbench.image.ImageFrame` with a WCS built
from the pointing centre, roll and field size, and no pixel data. It then goes
through :func:`comet_testbench.field_search.find_comets` exactly like a
downlinked image. When real UltraSat images arrive, swap the virtual frames for
the FITS files and nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from astropy.coordinates import SkyCoord, get_body, get_sun
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
import astropy.units as u

from .field_search import find_comets
from .image import ImageFrame
from .instruments import UltraSat

# Field used by the simulator (ultrasat_object_search.ultrasat_fov).
ULTRASAT_SIM_FOV_DEG = (20.0, 10.0)


def virtual_frame(ra: float, dec: float, time, fov_deg=ULTRASAT_SIM_FOV_DEG, roll_deg: float = 0.0,
                  pixel_scale_arcsec: float = 60.0, exposure: float = 0.0,
                  telescope: str = "ULTRASAT", label: str = "") -> ImageFrame:
    """An image-shaped stand-in for one pointing.

    The WCS is a gnomonic (TAN) projection centred on (ra, dec) with the
    field's long axis along RA at ``roll_deg = 0``. ``pixel_scale_arcsec``
    only sets the resolution of the footprint test; it does not need to match
    the detector. Use a real UltraSat WCS (or real FITS file) instead when one
    exists, which also captures chip gaps.
    """
    nx = int(round(fov_deg[0] * 3600 / pixel_scale_arcsec))
    ny = int(round(fov_deg[1] * 3600 / pixel_scale_arcsec))
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [(nx + 1) / 2, (ny + 1) / 2]
    s = pixel_scale_arcsec / 3600
    th = np.deg2rad(roll_deg)
    w.wcs.cd = [[-s * np.cos(th), s * np.sin(th)], [s * np.sin(th), s * np.cos(th)]]
    t = Time(time)
    header = w.to_header()
    header["DATE-OBS"] = t.utc.isot
    header["TIMESYS"] = "UTC"
    header["EXPOSURE"] = exposure
    header["TELESCOP"] = telescope
    header["OBJECT"] = label
    return ImageFrame(data=np.zeros((ny, nx), dtype=np.float32), wcs=w, mid_time=t.utc,
                      exposure=exposure, header=fits.Header(header), filter="NONE",
                      instrument=telescope, source=label or f"pointing({ra:.3f},{dec:.3f})")


def parse_pointing(p):
    """Accept (ra, dec) pairs or the simulator's ``'ra dec'`` strings (degrees)."""
    if isinstance(p, str):
        ra, dec = p.split()[:2]
        return float(ra), float(dec)
    return float(p[0]), float(p[1])


@dataclass
class PointingConstraints:
    """Flags applied by ``gen_total_observability`` in the simulator."""

    max_abs_gal_lat: float = 30.0
    min_sun_sep: float = 70.0
    min_moon_sep: float = 30.0

    def evaluate(self, ra, dec, time) -> dict:
        c = SkyCoord(ra * u.deg, dec * u.deg)
        t = Time(time)
        sun = get_sun(t)
        moon = get_body("moon", t)
        b = float(c.galactic.b.deg)
        sun_sep = float(c.separation(SkyCoord(sun.ra, sun.dec)).deg)
        moon_sep = float(c.separation(SkyCoord(moon.ra, moon.dec)).deg)
        return {"gal_b_deg": b, "sun_sep_deg": sun_sep, "moon_sep_deg": moon_sep,
                "passes_constraints": abs(b) < self.max_abs_gal_lat and sun_sep > self.min_sun_sep
                and moon_sep > self.min_moon_sep}


def search_pointings(pointings: Sequence, times: Iterable, adapter=None,
                     fov_deg=ULTRASAT_SIM_FOV_DEG, roll_deg: float = 0.0,
                     vmag_lim: float = 25.0, no_magnitude: str = "periodic",
                     refine_with_horizons: bool = False,
                     constraints: Optional[PointingConstraints] = PointingConstraints(),
                     **find_kw) -> list:
    """Field-of-regard search over pointings x times, one row per comet found.

    Equivalent to ``ultrasat_object_search.api_loop`` but through the image
    search. ``vmag_lim=v_lim, no_magnitude='none'`` reproduces the simulator's
    sb_ident selection; the default also keeps numbered periodic comets that
    have no magnitude model.
    Constraint flags are reported per comet rather than used to drop rows.
    """
    adapter = adapter or UltraSat()
    rows = []
    for t in times:
        t = Time(t)
        for i, p in enumerate(pointings):
            ra, dec = parse_pointing(p)
            frame = virtual_frame(ra, dec, t, fov_deg, roll_deg, label=f"pointing {i}")
            comets = find_comets(frame, adapter=adapter, use_header_object=False,
                                 vmag_lim=vmag_lim, no_magnitude=no_magnitude,
                                 refine_with_horizons=refine_with_horizons, **find_kw)
            for c in comets:
                row = {"time_utc": t.utc.isot, "pointing": i, "pointing_ra": ra,
                       "pointing_dec": dec, **c.row()}
                if constraints is not None:
                    row.update(constraints.evaluate(c.geometry.ra, c.geometry.dec, t))
                rows.append(row)
    return rows
