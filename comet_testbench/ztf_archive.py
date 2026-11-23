"""Public Zwicky Transient Facility (ZTF) science images from IRSA.

ZTF images (Palomar 48-inch, ~1"/px, 3072 x 3080 px CCD quadrants) give the
test bench a large public set of ground-based frames with and without comets.
This module has three parts:

* :class:`~comet_testbench.instruments.ZTF`, the instrument adapter (Palomar observer, DN images, MAGZP),
* :func:`search_frames`, a query of the IRSA Image Browse Engine (IBE)
  metadata table at a sky position and time window (no login; public data only),
* :func:`download`, which fetches the ``sciimg.fits`` file for one frame.

IBE docs: https://irsa.ipac.caltech.edu/docs/program_interface/ztf_api.html
"""

from __future__ import annotations

import csv
import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .image import ImageFrame
from .instruments import ZTF, ZTF_MPC_CODE  # noqa: F401  (re-exported)

log = logging.getLogger(__name__)

IBE_SEARCH = "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci"
IBE_DATA = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci"


@dataclass
class ZTFFrame:
    """One row of the IBE ``ztf.ztf_current_meta_sci`` table."""

    filefracday: str
    field: int
    ccdid: int
    qid: int
    filtercode: str
    imgtypecode: str
    obsjd: float
    exptime: float
    seeing: float
    maglimit: float
    corners: tuple  # ((ra1, dec1), ..., (ra4, dec4)) degrees
    ra: float
    dec: float
    tan: tuple = ()  # (crpix1, crpix2, crval1, crval2, cd11, cd12, cd21, cd22) from IBE

    def wcs(self):
        """Linear TAN WCS from the IBE metadata (no TPV distortion; good to a few px)."""
        from astropy.wcs import WCS

        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crpix = self.tan[:2]
        w.wcs.crval = self.tan[2:4]
        w.wcs.cd = np.array(self.tan[4:]).reshape(2, 2)
        return w

    def contains_pixel(self, ra: float, dec: float, margin_pix: float = 0.0,
                       shape=(3080, 3072)) -> bool:
        """Footprint test in pixels via :meth:`wcs` (negative margin = outside by that much)."""
        x, y = self.wcs().world_to_pixel_values(ra, dec)
        ny, nx = shape
        return bool(margin_pix - 0.5 <= x <= nx - 0.5 - margin_pix
                    and margin_pix - 0.5 <= y <= ny - 0.5 - margin_pix)

    @property
    def rcid(self) -> int:
        return 4 * (self.ccdid - 1) + self.qid - 1

    @property
    def stem(self) -> str:
        return (f"ztf_{self.filefracday}_{self.field:06d}_{self.filtercode}"
                f"_c{self.ccdid:02d}_{self.imgtypecode}_q{self.qid}")

    @property
    def url(self) -> str:
        f = self.filefracday
        return f"{IBE_DATA}/{f[:4]}/{f[4:8]}/{f[8:]}/{self.stem}_sciimg.fits"

    def contains(self, ra: float, dec: float, margin_deg: float = 0.0) -> bool:
        """Rough footprint test from the corner coordinates (an RA/Dec box).

        The box is wider than the rotated quadrant; use :meth:`contains_pixel`
        to confirm a position is really on the image.
        """
        c = np.array(self.corners)
        cosd = np.cos(np.deg2rad(self.dec))
        px = ((c[:, 0] - self.ra + 180) % 360 - 180) * cosd
        py = c[:, 1] - self.dec
        qx = ((ra - self.ra + 180) % 360 - 180) * cosd
        qy = dec - self.dec
        return (px.min() + margin_deg <= qx <= px.max() - margin_deg
                and py.min() + margin_deg <= qy <= py.max() - margin_deg)


def _get(url, params=None, session=None, timeout=300, retries=4, stream=False):
    import requests

    getter = session.get if session is not None else requests.get
    delay = 5.0
    for attempt in range(retries + 1):
        try:
            r = getter(url, params=params, timeout=timeout, stream=stream)
            if r.status_code == 404:  # not public / not archived: no point retrying
                r.raise_for_status()
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            if attempt >= retries or getattr(getattr(exc, "response", None), "status_code", 0) == 404:
                raise
            log.info("IRSA request failed (%s); retrying in %.0f s", exc, delay)
            time.sleep(delay)
            delay *= 2


def search_frames(ra: float, dec: float, jd_start: float, jd_end: float,
                  imgtype: str = "o", session=None) -> list:
    """Public ZTF science frames covering (ra, dec) between two UTC JDs."""
    where = f"obsjd>={jd_start:.6f} AND obsjd<={jd_end:.6f} AND imgtypecode='{imgtype}'"
    r = _get(IBE_SEARCH, {"POS": f"{ra:.6f},{dec:.6f}", "ct": "csv", "WHERE": where}, session)
    out = []
    for row in csv.DictReader(io.StringIO(r.text)):
        corners = tuple((float(row[f"ra{k}"]), float(row[f"dec{k}"])) for k in range(1, 5))
        out.append(ZTFFrame(row["filefracday"], int(row["field"]), int(row["ccdid"]), int(row["qid"]),
                            row["filtercode"], row["imgtypecode"], float(row["obsjd"]),
                            float(row["exptime"]), float(row["seeing"] or "nan"),
                            float(row["maglimit"] or "nan"), corners,
                            float(row["ra"]), float(row["dec"]),
                            tuple(float(row[k]) for k in ("crpix1", "crpix2", "crval1", "crval2",
                                                          "cd11", "cd12", "cd21", "cd22"))))
    return out


def download(frame: ZTFFrame, dest, session=None, overwrite: bool = False) -> Path:
    """Fetch the science image for ``frame`` into ``dest``; returns its path."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{frame.stem}_sciimg.fits"
    if path.exists() and not overwrite:
        return path
    r = _get(frame.url, session=session, stream=True)
    tmp = path.with_suffix(".part")
    with open(tmp, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    tmp.rename(path)
    return path
