"""Hand-off to the separately developed OH/H2O package.

The test bench stops at aperture photometry plus observing geometry. Everything
after that (continuum removal, OH band flux, the coma model, production rates)
belongs to the OH/H2O package. The two sides meet at a file, so neither has to
import the other.

File format
-----------
One ECSV file (astropy's Enhanced Character Separated Values, a plain-text
table) per comet, named ``<designation>_photometry.ecsv``:

* a commented YAML header holding column units, descriptions and run metadata
  (``schema``, ``designation``, ``created_utc``, ``testbench_version``), and
* comma-separated rows, one per aperture per image.

Each row carries the image (frame, filter, instrument, mid-exposure time), the
aperture (kind, radius in pix / arcsec / km, coverage), the photometry (count
rate, background, calibrated magnitude and flux density where the instrument
adapter knows them) and the comet's geometry at that image's time (r_h, r_dot,
Delta, phase...). Filters are left as separate rows; pairing an OH band with a
continuum band is the package's job.

Reading it::

    from astropy.table import Table
    t = Table.read("46P_photometry.ecsv")      # units and metadata included
    t.meta["schema"]                           # 'ultrasat-comet-photometry/1'

Any CSV reader that skips lines starting with ``#`` also works (e.g.
``pandas.read_csv(path, comment="#")``), without units.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.table import Table
import astropy.units as u

SCHEMA = "ultrasat-comet-photometry/1"
VERSION = "0.2.0"

# name: (unit, description)
COLUMNS = {
    "frame": (None, "Source file and FITS extension"),
    "instrument": (None, "INSTRUME/TELESCOP from the header"),
    "filter": (None, "Filter name from the header"),
    "mid_time_utc": (None, "Mid-exposure time, UTC, ISO 8601"),
    "exposure": (u.s, "Exposure time"),
    "aperture_kind": (None, "'fixed' or 'iterative' (one growth-curve step)"),
    "step": (None, "Aperture index within its kind"),
    "is_converged_radius": (None, "True on the growth-curve step chosen as converged"),
    "x": (u.pix, "Aperture centre, 0-based column"),
    "y": (u.pix, "Aperture centre, 0-based row"),
    "radius_pix": (u.pix, "Aperture radius"),
    "radius_arcsec": (u.arcsec, "Aperture radius"),
    "radius_km": (u.km, "Aperture radius projected at the comet"),
    "coverage": (None, "Fraction of the aperture on valid pixels"),
    "rate": (u.ct / u.s, "Background-subtracted count rate in the aperture"),
    "rate_err": (u.ct / u.s, "1-sigma uncertainty on rate"),
    "background": (u.ct / u.s / u.pix, "Background level per pixel"),
    "mag": (u.mag, "Instrument-system magnitude (NaN if no zero point)"),
    "mag_err": (u.mag, "1-sigma uncertainty on mag"),
    "flux_density": (u.erg / u.s / u.cm**2 / u.AA, "Broadband flux density (NaN if unknown)"),
    "flux_density_err": (u.erg / u.s / u.cm**2 / u.AA, "1-sigma uncertainty on flux_density"),
    "ra": (u.deg, "Comet RA (ephemeris) at mid-exposure"),
    "dec": (u.deg, "Comet Dec (ephemeris) at mid-exposure"),
    "r_h": (u.au, "Heliocentric distance"),
    "r_dot": (u.km / u.s, "Heliocentric radial velocity"),
    "delta": (u.au, "Observer distance"),
    "delta_dot": (u.km / u.s, "Observer range rate"),
    "phase": (u.deg, "Sun-comet-observer angle"),
    "observer": (None, "Ephemeris observer (Horizons CENTER or site)"),
}


def _rows(m):
    """Table rows for one CometMeasurement (fixed apertures, then growth curve)."""
    g = m.comet.geometry
    common = {
        "frame": m.frame.source, "instrument": m.frame.instrument, "filter": m.frame.filter,
        "mid_time_utc": m.frame.mid_time.utc.isot, "exposure": float(m.frame.exposure),
        "ra": g.ra, "dec": g.dec, "r_h": g.r_h, "r_dot": g.r_dot, "delta": g.delta,
        "delta_dot": g.delta_dot, "phase": g.phase, "observer": g.observer,
    }
    sets = [("fixed", m.fixed, None)]
    if m.growth is not None:
        sets.append(("iterative", m.growth.apertures, m.growth.converged_index))
    for kind, aps, best in sets:
        for i, a in enumerate(aps):
            yield {**common, "aperture_kind": kind, "step": i,
                   "is_converged_radius": best is not None and i == best,
                   "x": a.x, "y": a.y, "radius_pix": a.radius_pix,
                   "radius_arcsec": a.radius_arcsec, "radius_km": a.radius_km,
                   "coverage": a.coverage, "rate": a.rate, "rate_err": a.rate_err,
                   "background": a.background, "mag": a.mag, "mag_err": a.mag_err,
                   "flux_density": a.flux_density, "flux_density_err": a.flux_density_err}


def handoff_table(measurements: Sequence, designation: str) -> Table:
    """One comet's photometry and geometry as an astropy Table with units."""
    rows = [r for m in measurements if m.comet.designation == designation for r in _rows(m)]
    t = Table(rows=rows, names=list(COLUMNS)) if rows else Table(names=list(COLUMNS))
    for name, (unit, desc) in COLUMNS.items():
        t[name].description = desc
        if unit is not None:
            t[name].unit = unit
    t.meta.update({
        "schema": SCHEMA,
        "designation": designation,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "testbench_version": VERSION,
    })
    return t


def _safe(designation: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", designation).strip("_")


def write_handoff(measurements: Sequence, out_dir) -> list:
    """Write one ECSV per comet; returns the paths written.

    Measurements whose photometry failed (no apertures) are skipped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    good = [m for m in measurements if m.fixed or m.growth is not None]
    paths = []
    for des in dict.fromkeys(m.comet.designation for m in good):
        p = out_dir / f"{_safe(des)}_photometry.ecsv"
        handoff_table(good, des).write(p, format="ascii.ecsv", delimiter=",", overwrite=True)
        paths.append(p)
    return paths


def read_handoff(path) -> Table:
    """Read a hand-off file and check its schema."""
    t = Table.read(path, format="ascii.ecsv")
    schema = t.meta.get("schema", "")
    if not str(schema).startswith("ultrasat-comet-photometry/"):
        raise ValueError(f"{path}: not a comet photometry hand-off file (schema={schema!r})")
    return t
