"""Run the OH -> H2O chain on a comet test bench hand-off file.

Input: the ECSV the test bench writes (schema ``ultrasat-comet-photometry/1``),
or the same table in memory from ``comet_testbench.oh.handoff_table``. Rows
are apertures; each image has fixed apertures and/or a growth curve.

For every OH-band image the nearest continuum-band image in time (within
``max_dt_minutes``) is used. Every OH-band aperture is matched to the
continuum count rate at the *same physical radius* (km at the comet),
interpolated along the continuum image's apertures; radii outside the
continuum image's range are skipped rather than extrapolated.

Output: one row per OH aperture per dust redness, with every intermediate
quantity, as an astropy Table (write it with :func:`write_results`).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence, Union

import astropy.units as u
import numpy as np
from astropy.table import Table
from astropy.time import Time

from . import physics
from .backend import settings
from .calibration import OHCalibration

ACCEPTED_SCHEMAS = ("ultrasat-comet-photometry/1",)
RESULT_SCHEMA = "ultrasat-oh-h2o/1"

# Hand-off columns this package reads, and the unit it converts them to.
REQUIRED = {
    "frame": None, "instrument": None, "filter": None, "mid_time_utc": None,
    "aperture_kind": None, "step": None, "is_converged_radius": None,
    "radius_km": u.km, "rate": u.ct / u.s, "rate_err": u.ct / u.s,
    "r_h": u.au, "r_dot": u.km / u.s, "delta": u.au,
}


class ContractError(ValueError):
    """The hand-off file is not what this package expects."""


def load_handoff(source: Union[str, Path, Table]) -> Table:
    """Read and check a hand-off file (or table). Units are converted in place."""
    t = source.copy() if isinstance(source, Table) else Table.read(source, format="ascii.ecsv")
    schema = str(t.meta.get("schema", ""))
    if schema not in ACCEPTED_SCHEMAS:
        raise ContractError(f"unsupported hand-off schema {schema!r}; accepted {ACCEPTED_SCHEMAS}")
    missing = [c for c in REQUIRED if c not in t.colnames]
    if missing:
        raise ContractError(f"hand-off lacks columns {missing}")
    for name, unit in REQUIRED.items():
        if unit is None:
            continue
        if t[name].unit is None:
            raise ContractError(f"column {name!r} has no unit")
        try:
            t[name].convert_unit_to(unit)
        except u.UnitConversionError as e:
            raise ContractError(f"column {name!r}: {e}") from e
    return t


def _frames(t: Table) -> dict:
    return {f: t[t["frame"] == f] for f in dict.fromkeys(t["frame"])}


def pair_frames(t: Table, oh_band: str, continuum_band: str, max_dt_minutes: float = 60.0):
    """[(oh_frame, continuum_frame, dt_minutes)], nearest continuum image in time."""
    frames = _frames(t)
    band = lambda rows: str(rows["filter"][0]).strip().upper()  # noqa: E731
    oh = [f for f, r in frames.items() if band(r) == oh_band.upper()]
    cont = [f for f, r in frames.items() if band(r) == continuum_band.upper()]
    if not cont:
        return []
    t_cont = Time([str(frames[c]["mid_time_utc"][0]) for c in cont])
    pairs = []
    for f in oh:
        dt = np.abs((t_cont - Time(str(frames[f]["mid_time_utc"][0]))).to_value(u.min))
        k = int(np.argmin(dt))
        if dt[k] <= max_dt_minutes:
            pairs.append((f, cont[k], float(dt[k])))
    return pairs


def continuum_at_radius(rows: Table, radius_km: float):
    """(rate, rate_err) of a continuum image at a physical radius, by linear interpolation.

    Uses all of the image's apertures (fixed and growth curve). NaN if the
    radius is outside the measured range.
    """
    r = np.asarray(rows["radius_km"], float)
    rate = np.asarray(rows["rate"], float)
    err = np.asarray(rows["rate_err"], float)
    ok = np.isfinite(r) & np.isfinite(rate) & np.isfinite(err)
    r, rate, err = r[ok], rate[ok], err[ok]
    if r.size == 0:
        return np.nan, np.nan
    r, idx = np.unique(r, return_index=True)
    rate, err = rate[idx], err[idx]
    if not (r[0] - 1e-6 * r[0] <= radius_km <= r[-1] + 1e-6 * r[-1]):
        return np.nan, np.nan
    if r.size == 1:
        return float(rate[0]), float(err[0])
    return float(np.interp(radius_km, r, rate)), float(np.interp(radius_km, r, err))


_COLUMNS = {
    "designation": (None, "Comet designation"),
    "oh_frame": (None, "OH-band image"),
    "continuum_frame": (None, "Continuum-band image paired with it"),
    "dt": (u.min, "Time between the two images"),
    "mid_time_utc": (None, "OH-band image mid-exposure, UTC"),
    "aperture_kind": (None, "'fixed' or 'iterative' (growth-curve step)"),
    "step": (None, "Aperture index within its kind"),
    "is_converged_radius": (None, "True on the growth-curve step the bench chose"),
    "radius_km": (u.km, "Aperture radius at the comet"),
    "dust_redness": (u.percent / (100 * u.nm), "Assumed dust reddening"),
    "beta": (None, "Continuum scale: OH-band ct/s per continuum ct/s for dust"),
    "oh_band_rate": (u.ct / u.s, "OH-band count rate"),
    "oh_band_rate_err": (u.ct / u.s, ""),
    "continuum_rate": (u.ct / u.s, "Continuum count rate at the same radius"),
    "continuum_rate_err": (u.ct / u.s, ""),
    "oh_rate": (u.ct / u.s, "OH signal: OH band - beta * continuum"),
    "oh_rate_err": (u.ct / u.s, ""),
    "oh_flux": (u.erg / u.cm**2 / u.s, "OH emission flux in the calibration's OH bands"),
    "oh_flux_err": (u.erg / u.cm**2 / u.s, ""),
    "num_oh": (None, "OH molecules in the aperture"),
    "num_oh_err": (None, ""),
    "q_h2o": (1 / u.s, "H2O production rate, vectorial model matched to the aperture"),
    "q_h2o_err": (1 / u.s, ""),
    "q_h2o_total": (1 / u.s, "H2O production rate if the aperture held the whole OH coma"),
    "q_h2o_total_err": (1 / u.s, ""),
    "r_h": (u.au, "Heliocentric distance"),
    "r_dot": (u.km / u.s, "Heliocentric radial velocity"),
    "delta": (u.au, "Observer distance"),
}


def run(source: Union[str, Path, Table], calibration: OHCalibration,
        dust_redness: Union[float, Sequence[float]] = 0.0, max_dt_minutes: float = 60.0,
        aperture_kinds: Iterable[str] = ("fixed", "iterative"), total_q: bool = True,
        check_instrument: bool = True) -> Table:
    """OH flux and Q(H2O) for every OH-band aperture in a hand-off.

    calibration: e.g. ``SwiftUVOTCalibration()`` or a ``FixedCalibration``;
        its ``oh_band`` / ``continuum_band`` pick the filters.
    dust_redness: one value or several, in % per 100 nm, for the beta used.
    total_q: also compute the whole-coma Q (``q_h2o_total``).
    check_instrument: refuse rows whose 'instrument' the calibration does not accept.
    """
    t = load_handoff(source)
    reds = [float(x) for x in np.atleast_1d(dust_redness)]
    kinds = set(aperture_kinds)
    flux_factor = calibration.oh_flux_per_count_rate()
    betas = {d: calibration.beta(d) for d in reds}
    oh_bands = getattr(calibration, "oh_bands", None)
    frames = _frames(t)
    designation = str(t.meta.get("designation", ""))

    rows = []
    for oh_f, cont_f, dt in pair_frames(t, calibration.oh_band, calibration.continuum_band,
                                         max_dt_minutes):
        oh_rows, cont_rows = frames[oh_f], frames[cont_f]
        if check_instrument:
            for f, rr in ((oh_f, oh_rows), (cont_f, cont_rows)):
                if not calibration.accepts(str(rr["instrument"][0])):
                    raise ContractError(
                        f"{f}: instrument {rr['instrument'][0]!r} is not covered by "
                        f"calibration {calibration.name!r} (pass check_instrument=False to override)"
                    )
        geo = physics.Geometry(float(oh_rows["r_h"][0]), float(oh_rows["r_dot"][0]),
                               float(oh_rows["delta"][0]))
        for a in oh_rows:
            if str(a["aperture_kind"]) not in kinds:
                continue
            r_km = float(a["radius_km"])
            c, c_err = continuum_at_radius(cont_rows, r_km)
            if not np.isfinite(c) or not np.isfinite(float(a["rate"])):
                continue
            for d in reds:
                beta = betas[d]
                s = physics.continuum_subtract(float(a["rate"]), float(a["rate_err"]), c, c_err, beta)
                flux = s * flux_factor
                n = physics.oh_molecules(flux, geo, bands=oh_bands)
                q = physics.q_h2o_in_aperture(n, geo, r_km)
                qt = physics.q_h2o_total(n, geo) if total_q else None
                rows.append({
                    "designation": designation, "oh_frame": oh_f, "continuum_frame": cont_f,
                    "dt": dt, "mid_time_utc": str(a["mid_time_utc"]),
                    "aperture_kind": str(a["aperture_kind"]), "step": int(a["step"]),
                    "is_converged_radius": bool(a["is_converged_radius"]), "radius_km": r_km,
                    "dust_redness": d, "beta": beta,
                    "oh_band_rate": float(a["rate"]), "oh_band_rate_err": float(a["rate_err"]),
                    "continuum_rate": c, "continuum_rate_err": c_err,
                    "oh_rate": s.value, "oh_rate_err": s.sigma,
                    "oh_flux": flux.value, "oh_flux_err": flux.sigma,
                    "num_oh": n.value, "num_oh_err": n.sigma,
                    "q_h2o": q.value, "q_h2o_err": q.sigma,
                    "q_h2o_total": qt.value if qt else np.nan,
                    "q_h2o_total_err": qt.sigma if qt else np.nan,
                    "r_h": geo.r_h, "r_dot": geo.r_dot, "delta": geo.delta,
                })

    out = Table(rows=rows, names=list(_COLUMNS)) if rows else Table(names=list(_COLUMNS))
    for name, (unit, desc) in _COLUMNS.items():
        out[name].description = desc
        if unit is not None:
            out[name].unit = unit
    try:
        from importlib.metadata import version

        scp_version = version("swift_comet_pipeline")
    except Exception:  # pragma: no cover
        scp_version = "unknown"
    from . import __version__

    out.meta.update({
        "schema": RESULT_SCHEMA,
        "designation": designation,
        "input_schema": str(t.meta.get("schema")),
        "calibration": calibration.name,
        "mission_parameters": getattr(calibration, "mission_parameters", None),
        "oh_band": calibration.oh_band,
        "continuum_band": calibration.continuum_band,
        "oh_flux_per_count_rate": flux_factor,
        "oh_bands": "all" if oh_bands is None else ",".join(oh_bands),
        "swift_comet_pipeline_version": scp_version,
        "oh_h2o_version": __version__,
        "vectorial_model": {k: settings()[k] for k in ("quality", "backend")},
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    })
    return out


def write_results(table: Table, path) -> Path:
    """Write a result table as ECSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(path, format="ascii.ecsv", delimiter=",", overwrite=True)
    return path
