"""Comet ephemerides and observing geometry.

Two JPL services are used:

* the Small-Body Identification API (via the existing ``sbdb_ident_query``
  module) to discover *which* comets might be in a field, and
* the Horizons API to get an accurate position and geometry for each candidate
  as seen from the actual observer (e.g. the Swift spacecraft, not the
  geocenter: at Delta ~ 0.1 au the parallax from low Earth orbit is ~1 arcmin).

sb_ident can miss comets: with its default ``mag-required`` it drops comets
with no predicted magnitude (that alone hides 46P/Wirtanen in its Nov 2018
Swift images), and it propagates current osculating elements, which can be
poor for archival epochs. The query here turns ``mag-required`` off, and the
field search also checks named candidates directly with Horizons.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Callable, Iterable, Optional

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
SB_IDENT_URL = "https://ssd-api.jpl.nasa.gov/sb_ident.api"
AU_KM = 149597870.7


@dataclass
class ObservingGeometry:
    """Comet geometry at one instant, as seen by the observer.

    This is also the geometry object handed to an OH model.
    """

    target: str
    time_jd_utc: float
    ra: float  # deg, astrometric ICRF
    dec: float  # deg
    r_h: float  # heliocentric distance, au
    r_dot: float  # heliocentric radial velocity, km/s (Swings effect)
    delta: float  # observer distance, au
    delta_dot: float  # km/s
    phase: float  # Sun-target-observer angle, deg
    ra_rate: float = float("nan")  # dRA*cos(Dec)/dt, arcsec/hour
    dec_rate: float = float("nan")  # arcsec/hour
    observer: str = "500@399"

    @property
    def km_per_arcsec(self) -> float:
        return self.delta * AU_KM * np.deg2rad(1 / 3600)

    def arcsec_to_km(self, arcsec):
        return np.asarray(arcsec) * self.km_per_arcsec

    def km_to_arcsec(self, km):
        return np.asarray(km) / self.km_per_arcsec

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------
# Designations
# --------------------------------------------------------------------------

_PERIODIC = re.compile(r"(?<![A-Z0-9])([1-9]\d{0,3})([PD])(?![0-9])(?:-([A-Z]{1,2}))?", re.I)
_NONPERIODIC = re.compile(r"\b([CPDXI])/\s?(\d{4})\s?([A-Z]{1,2}\d{0,3})", re.I)
_INTERSTELLAR = re.compile(r"(?<![A-Z0-9])(\d{1,2})I\b", re.I)


def normalize_designation(name: str) -> Optional[str]:
    """Pull a comet designation out of a free-form name.

    ``'46P/Wirtanen'`` -> ``'46P'``; ``'COMET46P'`` -> ``'46P'``;
    ``'COMET41P/T-G-K'`` -> ``'41P'``; ``'C/2020 F3 (NEOWISE)'`` -> ``'C/2020 F3'``.
    Returns None when nothing comet-like is found.
    """
    if not name:
        return None
    s = str(name).upper().replace("_", " ")
    m = _NONPERIODIC.search(s)
    if m:
        return f"{m.group(1)}/{m.group(2)} {m.group(3)}"
    m = _PERIODIC.search(s.replace("COMET", " "))
    if m:
        return f"{int(m.group(1))}{m.group(2)}" + (f"-{m.group(3)}" if m.group(3) else "")
    m = _INTERSTELLAR.search(s.replace("COMET", " "))
    if m:
        return f"{int(m.group(1))}I"
    return None


def horizons_command(designation: str) -> str:
    """Horizons COMMAND string selecting the orbit solution closest in time.

    ``CAP`` picks the apparition closest to the requested epoch and
    ``NOFRAG`` excludes fragments, which is what archival work needs.
    """
    return f"'DES={designation};CAP;NOFRAG'"


# --------------------------------------------------------------------------
# Horizons
# --------------------------------------------------------------------------

def _get(url, params, session=None, timeout=120, retries=5):
    """GET JSON from a JPL API, backing off when the service rate-limits.

    sb_ident answers a burst of requests (a field-of-regard loop) with HTTP
    429 / "Rate limit exceeded"; wait and retry rather than fail the loop.
    """
    import time
    import requests

    getter = session.get if session is not None else requests.get
    delay = 2.0
    for attempt in range(retries + 1):
        r = getter(url, params=params, timeout=timeout)
        limited = r.status_code in (429, 503) or (
            r.status_code == 200 and "rate limit" in r.text[:300].lower())
        if limited and attempt < retries:
            try:
                wait = min(float(r.headers.get("Retry-After", delay)), 900.0)
            except ValueError:
                wait = delay
            time.sleep(max(wait, delay))
            delay *= 2
            continue
        r.raise_for_status()
        return r.json()


def _center_params(observer: str) -> dict:
    if observer.startswith("site:"):
        return {"CENTER": "'coord@399'", "COORD_TYPE": "GEODETIC",
                "SITE_COORD": f"'{observer[5:]}'"}
    return {"CENTER": f"'{observer}'"}


def query_horizons(designation: str, times: Time, observer: str = "500@399",
                   session=None) -> list:
    """Observer-table ephemeris for one comet at one or more times.

    Parameters
    ----------
    designation : str
        Comet designation, e.g. ``'46P'`` or ``'C/2020 F3'``.
    times : astropy.time.Time
        Scalar or array of times (any scale; converted to UTC JD).
    observer : str
        Horizons CENTER, e.g. ``'500@399'`` (geocenter), ``'500@-128485'``
        (Swift spacecraft) or an MPC code such as ``'568'``. A site above the
        Earth can be given as ``'site:lon_deg,lat_deg,alt_km'`` (geodetic),
        which is how the UltraSat field-of-regard simulations place the
        observer.

    Returns
    -------
    list of ObservingGeometry
    """
    times = Time(times)
    jds = np.atleast_1d(times.utc.jd)
    tlist = " ".join(f"{jd:.8f}" for jd in jds)
    params = {
        "format": "json",
        "COMMAND": horizons_command(designation),
        "EPHEM_TYPE": "OBSERVER",
        **_center_params(observer),
        "TLIST": f"'{tlist}'",
        "TLIST_TYPE": "JD",
        "TIME_TYPE": "UT",
        "QUANTITIES": "'1,3,19,20,24'",
        "ANG_FORMAT": "DEG",
        "CSV_FORMAT": "YES",
        "MAKE_EPHEM": "YES",
        "OBJ_DATA": "NO",
        "EXTRA_PREC": "YES",
    }
    result = _get(HORIZONS_URL, params, session=session).get("result", "")
    return _parse_horizons(result, designation, observer, jds)


def _parse_horizons(text: str, designation: str, observer: str, jds) -> list:
    if "$$SOE" not in text:
        raise RuntimeError(f"Horizons returned no ephemeris for {designation!r}:\n{text[:1500]}")
    header_line = next(l for l in text.splitlines() if l.strip().startswith("Date__"))
    cols = [c.strip() for c in header_line.split(",")]
    body = text.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines()

    def col(name):
        return next(i for i, c in enumerate(cols) if c.startswith(name))

    i_ra, i_dec = col("R.A."), col("DEC")
    i_dra, i_ddec = col("dRA*cosD"), col("d(DEC)/dt")
    i_r, i_rdot = col("r"), col("rdot")
    i_d, i_ddot = col("delta"), col("deldot")
    i_ph = col("S-T-O")
    out = []
    for jd, line in zip(jds, body):
        f = [x.strip() for x in line.split(",")]
        out.append(ObservingGeometry(
            target=designation, time_jd_utc=float(jd),
            ra=float(f[i_ra]), dec=float(f[i_dec]),
            r_h=float(f[i_r]), r_dot=float(f[i_rdot]),
            delta=float(f[i_d]), delta_dot=float(f[i_ddot]),
            phase=float(f[i_ph]),
            ra_rate=float(f[i_dra]), dec_rate=float(f[i_ddec]),
            observer=observer,
        ))
    return out


# --------------------------------------------------------------------------
# Small-Body Identification (discovery of unexpected comets in a field)
# --------------------------------------------------------------------------

def _sexagesimal(value_deg: float, hours: bool) -> str:
    """Format for sb_ident: hh-mm-ss.ss / dd-mm-ss.s with 'M' for minus."""
    c = SkyCoord(value_deg * u.deg, 0 * u.deg) if hours else None
    if hours:
        h, m, s = c.ra.hms
        return f"{int(h):02d}-{int(m):02d}-{s:05.2f}"
    sign = "M" if value_deg < 0 else ""
    v = abs(value_deg)
    d = int(v)
    m = int((v - d) * 60)
    s = (v - d - m / 60) * 3600
    return f"{sign}{d:02d}-{m:02d}-{s:04.1f}"


@dataclass
class SbIdentRecord:
    """One comet from an sb_ident query (second-pass, n-body positions)."""

    designation: str
    name: str
    ra: float  # deg
    dec: float  # deg
    vmag: float  # NaN when sb_ident has no magnitude model for the comet
    ra_rate: float = float("nan")  # arcsec/hour
    dec_rate: float = float("nan")


def _parse_sb_ident_row(row) -> Optional[SbIdentRecord]:
    des = normalize_designation(row[0])
    if not des:
        return None
    dec = row[2].replace("'", " ").replace('"', "").replace("\\", "")
    c = SkyCoord(row[1], dec, unit=(u.hourangle, u.deg))

    def num(v):
        try:
            return float(str(v).rstrip("NT"))
        except (TypeError, ValueError):
            return float("nan")

    vmag = num(row[6]) if len(row) > 6 else float("nan")
    return SbIdentRecord(des, row[0], float(c.ra.deg), float(c.dec.deg), vmag,
                         num(row[7]) if len(row) > 7 else float("nan"),
                         num(row[8]) if len(row) > 8 else float("nan"))


def sb_ident_query(ra_center: float, dec_center: float, ra_hwidth: float, dec_hwidth: float,
                   time: Time, mpc_code: Optional[str] = "500", site=None,
                   vmag_lim: float = 25.0, no_magnitude: str = "periodic",
                   session=None) -> list:
    """Comets JPL sb_ident places inside a centre/half-width field (degrees).

    Reuses ``sbdb_ident_query.param_build`` (comet classes only), the same call
    the field-of-regard simulator (``ultrasat_object_search.api_query``) makes.
    The observer is an MPC code, or ``site=(lat_deg, lon_deg, alt_km)`` as in
    the simulator.

    ``no_magnitude`` decides what happens to comets sb_ident has no
    magnitude model for (so ``vmag_lim`` cannot apply to them):

    * ``'none'``: drop them. Same as the simulator (``mag-required=true``),
      which loses e.g. 46P/Wirtanen in its Nov 2018 Swift images.
    * ``'periodic'`` (default): keep numbered periodic comets such as 46P,
      drop the rest, which are mostly long-gone SOHO sungrazer fragments.
    * ``'all'``: keep everything.
    """
    if no_magnitude not in ("none", "periodic", "all"):
        raise ValueError("no_magnitude must be 'none', 'periodic' or 'all'")
    import sbdb_ident_query as sbq

    date = Time(time).utc.strftime("%Y-%m-%d_%H:%M:%S")
    ra = [_sexagesimal(ra_center, hours=True), f"{ra_hwidth:.4f}"]
    dec = [_sexagesimal(dec_center, hours=False), f"{dec_hwidth:.4f}"]
    if site is not None:
        lat, lon, alt = site
        params = sbq.param_build(date, vmag_lim, ra, dec, lat=lat, lon=lon, alt=alt, fov_center=True)
    else:
        params = sbq.param_build(date, vmag_lim, ra, dec, loc=mpc_code, fov_center=True)
    params["mag-required"] = "true" if no_magnitude == "none" else "false"
    data = _get(SB_IDENT_URL, params, session=session)
    rows = data.get("data_second_pass") or data.get("data") or []
    out, seen = [], set()
    for row in rows:
        rec = _parse_sb_ident_row(row)
        if not rec or rec.designation in seen:
            continue
        if not np.isfinite(rec.vmag):
            if no_magnitude == "none":
                continue
            if no_magnitude == "periodic" and not re.match(r"^\d+[PD]", rec.designation):
                continue
        elif rec.vmag > vmag_lim:
            continue
        seen.add(rec.designation)
        out.append(rec)
    return out


def sb_ident_comets(*args, **kwargs) -> list:
    """Designations only; see :func:`sb_ident_query` for the arguments."""
    return [r.designation for r in sb_ident_query(*args, **kwargs)]


EphemerisProvider = Callable[[str, Time, str], ObservingGeometry]


def horizons_provider(session=None) -> EphemerisProvider:
    """Default ephemeris provider with an in-memory cache."""
    cache = {}

    def provider(designation: str, time: Time, observer: str) -> ObservingGeometry:
        key = (designation, round(float(Time(time).utc.jd), 7), observer)
        if key not in cache:
            cache[key] = query_horizons(designation, time, observer, session=session)[0]
        return cache[key]

    return provider
