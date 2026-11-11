"""Find and download public Swift/UVOT images from the HEASARC archive.

Uses the HEASARC TAP service (``swiftmastr`` table) to list observations and
fetches the sky images (``*_sk.img.gz``) and exposure maps (``*_ex.img.gz``)
from the public archive tree.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from astropy.time import Time

log = logging.getLogger(__name__)

TAP_SYNC = "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync"
ARCHIVE = "https://heasarc.gsfc.nasa.gov/FTP/swift/data/obs"

# Filter name -> code used in UVOT file names.
FILTER_CODES = {"UVW1": "uw1", "UVW2": "uw2", "UVM2": "um2", "U": "uuu",
                "B": "ubb", "V": "uvv", "WHITE": "uwh"}


@dataclass
class SwiftObservation:
    obsid: str
    target: str
    ra: float
    dec: float
    start: Time
    uvot_exposure: float

    @property
    def archive_dir(self) -> str:
        return f"{ARCHIVE}/{self.start.strftime('%Y_%m')}/{self.obsid}/uvot/image"


def _tap(adql: str, session=None):
    import requests
    from astropy.io.votable import parse_single_table

    get = session.get if session is not None else requests.get
    r = get(TAP_SYNC, params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable",
                              "QUERY": adql}, timeout=180)
    r.raise_for_status()
    return parse_single_table(io.BytesIO(r.content)).to_table()


def find_observations(name_pattern: str, start: Optional[str] = None, stop: Optional[str] = None,
                      min_exposure: float = 0.0, session=None) -> list:
    """Swift observations whose target name matches ``name_pattern``.

    ``name_pattern`` is an SQL LIKE pattern on the upper-cased target name,
    e.g. ``'%46P%'`` or ``'%WIRTANEN%'``. ``start``/``stop`` are ISO dates.
    """
    where = [f"upper(name) LIKE '{name_pattern.upper()}'", f"uvot_exposure > {float(min_exposure)}"]
    if start:
        where.append(f"start_time >= {Time(start).mjd}")
    if stop:
        where.append(f"start_time <= {Time(stop).mjd}")
    adql = ("SELECT obsid, name, ra, dec, start_time, uvot_exposure FROM swiftmastr WHERE "
            + " AND ".join(where) + " ORDER BY start_time")
    t = _tap(adql, session)
    # The VOTable parser may name the obsid column by its field ID ("DataLinkID").
    obs_col = t.colnames[0]
    return [SwiftObservation(str(r[obs_col]).zfill(11), str(r["name"]).strip(), float(r["ra"]),
                             float(r["dec"]), Time(float(r["start_time"]), format="mjd"),
                             float(r["uvot_exposure"])) for r in t]


def image_urls(obs: SwiftObservation, filt: str):
    code = FILTER_CODES.get(filt.upper(), filt.lower())
    base = f"{obs.archive_dir}/sw{obs.obsid}{code}"
    return f"{base}_sk.img.gz", f"{base}_ex.img.gz"


def download_images(observations: Iterable[SwiftObservation], filters=("UVW1", "V"),
                    dest="data/swift", overwrite=False, session=None) -> list:
    """Download sky images and exposure maps; returns (sky_path, exp_path, obs, filter).

    Observations without a given filter are skipped quietly (most visits only
    use some filters).
    """
    import requests

    sess = session or requests.Session()
    out = []
    for obs in observations:
        for filt in filters:
            sk_url, ex_url = image_urls(obs, filt)
            paths = []
            for url in (sk_url, ex_url):
                p = Path(dest) / obs.obsid / url.rsplit("/", 1)[1]
                if not p.exists() or overwrite:
                    r = sess.get(url, timeout=600)
                    if r.status_code == 404:
                        paths = None
                        break
                    r.raise_for_status()
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(r.content)
                paths.append(p)
            if paths:
                out.append((paths[0], paths[1], obs, filt.upper()))
                log.info("Have %s %s", obs.obsid, filt)
    return out
