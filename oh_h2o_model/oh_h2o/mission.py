"""Versioned mission parameter files (``mission_parameters/*.json``).

One JSON file per instrument, schema ``ultrasat-mission-parameters/1``::

    {
      "schema": "ultrasat-mission-parameters/1",
      "mission": "ULTRASAT",
      "version": "0.1.0",                  # bump on every change, add a changelog entry
      "updated": "2026-11-18",
      "instrument_match": ["ULTRASAT"],    # substrings of the hand-off 'instrument' column
      "calibration_source": "this_file",   # or "swift_comet_pipeline"
      "changelog": [{"version": ..., "date": ..., "notes": ...}],
      "parameters": {
        "<name>": {"value": ..., "unit": ..., "status": ..., "source": ...},
        ...
      }
    }

``status`` is one of measured, nominal, reference, simulator, unconfirmed or
TBD. A parameter whose value is null is not known yet. The OH model reads
``oh_band``, ``continuum_band``, ``beta``, ``beta_reddening_slope`` (optional)
and ``oh_flux_per_count_rate``; the rest are there for the test bench and the
future tests in ``future_tests.md``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .calibration import FixedCalibration, OHCalibration, SwiftUVOTCalibration

SCHEMA = "ultrasat-mission-parameters/1"
STATUSES = {"measured", "nominal", "reference", "simulator", "unconfirmed", "TBD"}
_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


class MissionParameterError(ValueError):
    """The file is malformed, or a parameter the model needs is missing."""


@dataclass
class MissionParameters:
    mission: str
    version: str
    updated: str
    instrument_match: tuple
    calibration_source: str
    parameters: dict
    path: Optional[str] = None

    def value(self, name: str):
        p = self.parameters.get(name)
        return None if p is None else p.get("value")

    def status(self, name: str) -> Optional[str]:
        p = self.parameters.get(name)
        return None if p is None else p.get("status")

    def missing(self, names) -> list:
        """Names among ``names`` with no value yet."""
        return [n for n in names if self.value(n) is None]

    @property
    def label(self) -> str:
        return f"{self.mission} parameters v{self.version}"


def load_mission_parameters(path) -> MissionParameters:
    """Read and check a mission parameter file."""
    path = Path(path)
    with open(path) as f:
        d = json.load(f)
    if d.get("schema") != SCHEMA:
        raise MissionParameterError(f"{path}: schema {d.get('schema')!r}, expected {SCHEMA!r}")
    for key in ("mission", "version", "updated", "parameters", "changelog"):
        if key not in d:
            raise MissionParameterError(f"{path}: missing {key!r}")
    if not _SEMVER.match(str(d["version"])):
        raise MissionParameterError(f"{path}: version {d['version']!r} is not MAJOR.MINOR.PATCH")
    if not any(c.get("version") == d["version"] for c in d["changelog"]):
        raise MissionParameterError(f"{path}: no changelog entry for version {d['version']}")
    for name, p in d["parameters"].items():
        if not isinstance(p, dict) or "value" not in p or "status" not in p:
            raise MissionParameterError(f"{path}: parameter {name!r} needs 'value' and 'status'")
        if p["status"] not in STATUSES:
            raise MissionParameterError(f"{path}: parameter {name!r} has status {p['status']!r}")
        if p["value"] is None and p["status"] != "TBD":
            raise MissionParameterError(f"{path}: parameter {name!r} is null but not 'TBD'")
    return MissionParameters(
        mission=d["mission"], version=d["version"], updated=d["updated"],
        instrument_match=tuple(d.get("instrument_match", ())),
        calibration_source=d.get("calibration_source", "this_file"),
        parameters=d["parameters"], path=str(path),
    )


_NEEDED = ("oh_band", "continuum_band", "beta", "oh_flux_per_count_rate")


def calibration_from_mission(source) -> OHCalibration:
    """OHCalibration from a mission parameter file (path or MissionParameters).

    ``calibration_source: "swift_comet_pipeline"`` gives a SwiftUVOTCalibration
    for the file's bands; ``"this_file"`` gives a FixedCalibration from the
    file's numbers, and fails with the list of TBD parameters if any are missing.
    """
    mp = source if isinstance(source, MissionParameters) else load_mission_parameters(source)
    if mp.calibration_source == "swift_comet_pipeline":
        cal = SwiftUVOTCalibration(oh_band=mp.value("oh_band") or "UVW1",
                                   continuum_band=mp.value("continuum_band") or "V")
    elif mp.calibration_source == "this_file":
        missing = mp.missing(_NEEDED)
        if missing:
            raise MissionParameterError(
                f"{mp.label}: OH calibration incomplete, still TBD: {', '.join(missing)}"
            )
        beta0 = float(mp.value("beta"))
        slope = mp.value("beta_reddening_slope")
        beta = beta0 if slope is None else (lambda d, b=beta0, s=float(slope): b * (1 + s * d))
        cal = FixedCalibration(
            name=mp.mission.lower(), oh_band=mp.value("oh_band"),
            continuum_band=mp.value("continuum_band"), beta_value=beta,
            flux_per_count_rate=float(mp.value("oh_flux_per_count_rate")),
            instruments=mp.instrument_match,
        )
    else:
        raise MissionParameterError(f"{mp.label}: unknown calibration_source {mp.calibration_source!r}")
    cal.mission_parameters = mp.label
    return cal
