"""Known-comet ephemeris cache: a second candidate source next to sb_ident.

JPL sb_ident propagates the orbit solution stored in the SBDB. For some
periodic comets that solution belongs to an earlier apparition (67P: the 2015
perihelion; 62P: 2017), and sb_ident then returns nothing for them near their
next perihelion, even in a 20 deg box. Horizons picks the right apparition
(``CAP``), so the fix is to ask Horizons directly for a list of comets that
could be observable:

1. :func:`candidate_comets` lists comets from the SBDB query API with
   q < ``q_max`` and a perihelion within ``window_yr`` of the period. For
   periodic comets the perihelion times are stepped by the period from the
   stored ``tp``, because the stored ``tp`` can be an old apparition.
2. :meth:`EphemerisCache.build` gets one Horizons ephemeris per comet on a
   grid (default daily) for the whole period.
3. :meth:`EphemerisCache.in_frame` interpolates every track to a frame's
   mid-time and returns the comets that land on (or near) it. The field
   search then places each with an exact Horizons query, as for sb_ident.

Building costs one Horizons call per comet per period instead of per frame.
The cache saves to JSON so a campaign builds it once.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import astropy.units as u
from astropy.time import Time

from .ephemeris import HORIZONS_URL, _center_params, _get, horizons_command

log = logging.getLogger(__name__)

SBDB_QUERY_URL = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
DAY = 1.0

# Comets active far from perihelion that a perihelion window would miss.
DISTANT_ACTIVE = ("29P",)


def _perihelion_near(tp_jd: float, period_yr: float, jd0: float, jd1: float, window_d: float) -> bool:
    """True if any perihelion (tp + k*P) falls within window_d of [jd0, jd1]."""
    if not np.isfinite(tp_jd):
        return False
    lo, hi = jd0 - window_d, jd1 + window_d
    if not np.isfinite(period_yr) or period_yr <= 0:
        return lo <= tp_jd <= hi
    p = period_yr * 365.25
    k = np.ceil((lo - tp_jd) / p)
    return tp_jd + k * p <= hi


def candidate_comets(t_start, t_end, q_max: float = 5.0, window_yr: float = 2.0,
                     max_period_yr: float = 200.0, extra: Sequence[str] = DISTANT_ACTIVE,
                     session=None) -> list:
    """Comet designations that may be observable between ``t_start`` and ``t_end``.

    Selection: perihelion distance below ``q_max`` and a perihelion within
    ``window_yr`` of the period, plus ``extra``. Periodic comets (period below
    ``max_period_yr``) are tested on every return, not only the stored ``tp``.
    """
    import requests

    jd0, jd1 = Time(t_start).tdb.jd, Time(t_end).tdb.jd
    params = {"fields": "pdes,q,tp,per_y,e", "sb-kind": "c",
              "sb-cdata": json.dumps({"AND": [f"q|LT|{q_max}"]})}
    getter = session.get if session is not None else requests.get
    r = getter(SBDB_QUERY_URL, params=params, timeout=300)
    r.raise_for_status()
    data = r.json()
    names = []
    for pdes, q, tp, per_y, e in data.get("data", []):
        try:
            tp = float(tp) if tp is not None else float("nan")
            per = float(per_y) if per_y is not None else float("nan")
        except ValueError:
            continue
        if np.isfinite(per) and per > max_period_yr:
            per = float("nan")  # treat as a single passage
        if _perihelion_near(tp, per, jd0, jd1, window_yr * 365.25):
            names.append(pdes)
    for e in extra:
        if e not in names:
            names.append(e)
    return names


def horizons_track(designation: str, t_start, t_end, observer: str = "500@399",
                   step: str = "1d", session=None) -> dict:
    """Horizons RA/Dec (deg) and total magnitude T-mag on a regular grid."""
    t0, t1 = Time(t_start).utc, Time(t_end).utc
    params = {
        "format": "json", "COMMAND": horizons_command(designation), "EPHEM_TYPE": "OBSERVER",
        **_center_params(observer),
        "START_TIME": f"'JD {t0.jd:.6f}'", "STOP_TIME": f"'JD {t1.jd:.6f}'", "STEP_SIZE": f"'{step}'",
        "TIME_TYPE": "UT", "QUANTITIES": "'1,9'", "ANG_FORMAT": "DEG", "CSV_FORMAT": "YES",
        "CAL_FORMAT": "JD", "MAKE_EPHEM": "YES", "OBJ_DATA": "NO", "EXTRA_PREC": "YES",
    }
    text = _get(HORIZONS_URL, params, session=session).get("result", "")
    if "$$SOE" not in text:
        raise RuntimeError(f"Horizons returned no ephemeris for {designation!r}: {text[:300]}")
    header = next(l for l in text.splitlines() if l.strip().startswith("Date_"))
    cols = [c.strip() for c in header.split(",")]
    i_ra = next(i for i, c in enumerate(cols) if c.startswith("R.A."))
    i_dec = next(i for i, c in enumerate(cols) if c.startswith("DEC"))
    i_t = next((i for i, c in enumerate(cols) if c.startswith("T-mag")), None)
    jd, ra, dec, tmag = [], [], [], []
    for line in text.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        try:
            jd.append(float(f[0]))
            ra.append(float(f[i_ra]))
            dec.append(float(f[i_dec]))
        except ValueError:
            continue
        try:
            tmag.append(float(f[i_t]) if i_t is not None else float("nan"))
        except ValueError:
            tmag.append(float("nan"))
    return {"jd": jd, "ra": ra, "dec": dec, "tmag": tmag}


def _unit(ra, dec):
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


@dataclass
class EphemerisCache:
    """Gridded ephemerides of known comets for one observer and period."""

    observer: str
    tracks: dict = field(default_factory=dict)  # designation -> {"jd","ra","dec","tmag"} lists
    failed: list = field(default_factory=list)

    @classmethod
    def build(cls, designations: Iterable[str], t_start, t_end, observer: str = "500@399",
              step: str = "1d", pad_days: float = 1.0, workers: int = 4, session=None):
        """Query Horizons for every designation (``workers`` in parallel)."""
        t0 = Time(t_start) - pad_days * u.day
        t1 = Time(t_end) + pad_days * u.day
        cache = cls(observer)

        def one(des):
            try:
                return des, horizons_track(des, t0, t1, observer, step, session)
            except Exception as exc:
                return des, exc

        with ThreadPoolExecutor(workers) as pool:
            for des, res in pool.map(one, list(designations)):
                if isinstance(res, Exception) or len(res["jd"]) < 2:
                    log.info("No Horizons track for %s: %s", des, res)
                    cache.failed.append(des)
                else:
                    cache.tracks[des] = res
        return cache

    def save(self, path):
        Path(path).write_text(json.dumps({"observer": self.observer, "tracks": self.tracks,
                                          "failed": self.failed}))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["observer"], d["tracks"], d.get("failed", []))

    def positions(self, time):
        """Interpolated (designation, ra, dec, tmag) for every track covering ``time``.

        Positions are cubic-interpolated on unit vectors, so RA wrap and the
        poles are handled; error on a daily grid is well under the default
        0.5 deg search margin except for the closest approaches (< 0.05 au).
        """
        from scipy.interpolate import CubicSpline

        jd = float(Time(time).utc.jd)
        out = []
        for des, t in self.tracks.items():
            tj = np.asarray(t["jd"])
            if not (tj[0] <= jd <= tj[-1]):
                continue
            v = _unit(np.asarray(t["ra"]), np.asarray(t["dec"]))
            if len(tj) >= 4:
                p = CubicSpline(tj, v, axis=0)(jd)
            else:
                p = np.array([np.interp(jd, tj, v[:, k]) for k in range(3)])
            p = p / np.linalg.norm(p)
            ra = float(np.rad2deg(np.arctan2(p[1], p[0])) % 360)
            dec = float(np.rad2deg(np.arcsin(np.clip(p[2], -1, 1))))
            tmag = float(np.interp(jd, tj, np.asarray(t["tmag"], dtype=float)))
            out.append((des, ra, dec, tmag))
        return out

    def in_frame(self, frame, margin_deg: float = 0.5, vmag_lim: Optional[float] = None) -> list:
        """Designations whose interpolated position lies within ``margin_deg`` of the frame."""
        ny, nx = frame.shape
        m = margin_deg / (frame.pixel_scale_arcsec / 3600)
        names = []
        for des, ra, dec, tmag in self.positions(frame.mid_time):
            if vmag_lim is not None and np.isfinite(tmag) and tmag > vmag_lim:
                continue
            x, y = frame.wcs.world_to_pixel_values(ra, dec)
            x, y = float(x), float(y)
            if np.isfinite(x) and -m <= x <= nx + m and -m <= y <= ny + m:
                names.append(des)
        return names
