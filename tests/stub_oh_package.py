"""Stand-in for the separately developed OH/H2O package.

It does only what the real package must do at the interface: read a hand-off
file, check it carries what the package needs in the units it expects, pair an
OH-band image with a continuum image, and return one result per pair. The
"science" is a placeholder (a fixed continuum scale and a geometric factor), so
the tests exercise the contract, not the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.table import Table
from astropy.time import Time

ACCEPTED_SCHEMAS = ("ultrasat-comet-photometry/1",)

# Columns the package reads, with the unit it expects (None = no unit).
REQUIRED = {
    "frame": None, "filter": None, "mid_time_utc": None, "aperture_kind": None,
    "is_converged_radius": None, "radius_km": u.km, "radius_arcsec": u.arcsec,
    "rate": u.ct / u.s, "rate_err": u.ct / u.s, "r_h": u.au, "r_dot": u.km / u.s,
    "delta": u.au, "phase": u.deg,
}


class ContractError(ValueError):
    pass


@dataclass
class OHResult:
    designation: str
    oh_frame: str
    continuum_frame: str
    dt_minutes: float
    radius_km: float
    oh_rate: float  # placeholder: OH-band rate minus scaled continuum
    oh_rate_err: float
    r_h: float
    delta: float
    r_dot: float
    q_proxy: float  # placeholder production-rate proxy


def load(path) -> Table:
    t = Table.read(path, format="ascii.ecsv")
    schema = str(t.meta.get("schema", ""))
    if schema not in ACCEPTED_SCHEMAS:
        raise ContractError(f"unsupported hand-off schema {schema!r}")
    missing = [c for c in REQUIRED if c not in t.colnames]
    if missing:
        raise ContractError(f"hand-off file lacks columns {missing}")
    for name, unit in REQUIRED.items():
        if unit is None:
            continue
        if t[name].unit is None:
            raise ContractError(f"column {name!r} has no unit")
        t[name].convert_unit_to(unit)  # raises UnitConversionError if incompatible
    return t


def model(path, oh_filter="UVW1", continuum_filter="V", continuum_scale=0.1,
          radius_km=None, max_dt_minutes=60.0) -> list:
    """Pair each OH-band image with the nearest-in-time continuum image.

    Uses the converged growth-curve aperture of the OH image, or the fixed
    aperture closest to ``radius_km`` if given; the continuum is measured in the
    aperture with the same physical radius.
    """
    t = load(path)
    des = t.meta["designation"]

    def pick(rows):
        if radius_km is None:
            sel = rows[(rows["aperture_kind"] == "iterative") & rows["is_converged_radius"]]
            if len(sel) == 0:
                return None
            return sel[0]
        fixed = rows[rows["aperture_kind"] == "fixed"]
        if len(fixed) == 0:
            return None
        return fixed[int(np.argmin(np.abs(fixed["radius_km"] - radius_km)))]

    def at_radius(rows, r_km):
        fixed = rows[rows["aperture_kind"] == "fixed"]
        pool = fixed if len(fixed) else rows
        return pool[int(np.argmin(np.abs(pool["radius_km"] - r_km)))]

    frames = {f: t[t["frame"] == f] for f in dict.fromkeys(t["frame"])}
    oh = [f for f, rows in frames.items() if rows["filter"][0] == oh_filter]
    cont = [f for f, rows in frames.items() if rows["filter"][0] == continuum_filter]
    out = []
    for f in oh:
        rows = frames[f]
        best = pick(rows)
        if best is None or not cont:
            continue
        t_oh = Time(best["mid_time_utc"])
        dts = [abs((Time(frames[c]["mid_time_utc"][0]) - t_oh).to_value(u.min)) for c in cont]
        k = int(np.argmin(dts))
        if dts[k] > max_dt_minutes:
            continue
        r_km = float(best["radius_km"])
        c = at_radius(frames[cont[k]], r_km)
        oh_rate = float(best["rate"] - continuum_scale * c["rate"])
        err = float(np.hypot(best["rate_err"], continuum_scale * c["rate_err"]))
        r_h, delta = float(best["r_h"]), float(best["delta"])
        out.append(OHResult(des, f, cont[k], float(dts[k]), r_km, oh_rate, err, r_h, delta,
                            float(best["r_dot"]), oh_rate * r_h**2 * delta**2 / r_km))
    return out
