"""Set-up swift_comet_pipeline needs before its functions can be called.

swift_comet_pipeline keeps three things in module-level state or on disk:

1. **Data files** (OH fluorescence g-factor table, UVOT effective areas,
   sensitivity). The PyPI wheel (0.3.0) does not ship them: it looks for them
   at ``../../../../data`` from its own source, which only exists in a git
   checkout installed with ``pip install -e``. ``configure(data_dir=...)``
   points it at the ``data/`` folder of a checkout of
   https://github.com/sjoset/swift_comet_pipeline instead (or set the
   ``SWIFT_COMET_PIPELINE_DATA`` environment variable).
2. **Vectorial model cache**: a SQLite file of model runs (pyvectorial_au),
   written to ``<cache_dir>/cache/vectorial_model_cache.sqlite3``. Only the
   first ``configure`` in a process sets it (a swift_comet_pipeline limit).
3. **Vectorial model grid quality and backend**: low / medium / high /
   very_high, and ``sbpy`` (pure Python) or ``rust``.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Optional

DATA_ENV = "SWIFT_COMET_PIPELINE_DATA"
CACHE_ENV = "OH_H2O_CACHE"
DEFAULT_CACHE = Path.home() / ".cache" / "oh_h2o"

# Modules that did `from ...pipeline_internal_config import read_swift_pipeline_config`.
_CONFIG_USERS = (
    "swift_comet_pipeline.pipeline.internal_config.pipeline_internal_config",
    "swift_comet_pipeline.swift.uvot_sensitivity",
    "swift_comet_pipeline.swift.filters.read_filter_effective_area",
    "swift_comet_pipeline.modeling.spectra.solar.solar_count_rate",
    "swift_comet_pipeline.modeling.fluorescence.hydroxyl_gfactor",
    "swift_comet_pipeline.photometry.dust.beta_parameter",
)

_state = {"data_dir": None, "cache_dir": None, "quality": None, "backend": None}


def _default_config():
    from swift_comet_pipeline.pipeline.internal_config import pipeline_internal_config as pic

    try:
        return pic.read_swift_pipeline_config()
    except Exception:
        return None


def _config_from(data_dir: Path):
    from swift_comet_pipeline.pipeline.internal_config import pipeline_internal_config as pic
    from swift_comet_pipeline.scp_types.primitive.uvot_filter import UvotFilter

    ea = data_dir / "swift_effective_areas"
    names = {
        UvotFilter.uuu: "swuuu_20041120v104.arf", UvotFilter.ubb: "swubb_20041120v104.arf",
        UvotFilter.uvv: "swuvv_20041120v104.arf", UvotFilter.uw1: "swuw1_20041120v106.arf",
        UvotFilter.uw2: "swuw2_20041120v105.arf", UvotFilter.um2: "swum2_20041120v105.arf",
        UvotFilter.white: "swuwh_20041120v104.arf",
        UvotFilter.vgrism: "swugv1000_20041120v101.arf",
        UvotFilter.ugrism: "swugu0200_20041120v101.arf",
    }
    return pic.SwiftPipelineConfig(
        uvot_sensitivity_path=data_dir / "sensitivity" / "swusenscorr20041120v007.fits",
        oh_fluorescence_path=data_dir / "fluorescence" / "Schleicher_Ahearn_1988_Table5.csv",
        effective_areas={k: ea / v for k, v in names.items()},
    )


def _data_ok(cfg) -> bool:
    return cfg is not None and Path(cfg.oh_fluorescence_path).is_file() and Path(
        next(iter(cfg.effective_areas.values()))
    ).parent.is_dir()


def _install_config(cfg) -> None:
    def read_swift_pipeline_config():
        return cfg

    for name in _CONFIG_USERS:
        mod = importlib.import_module(name)
        mod.read_swift_pipeline_config = read_swift_pipeline_config
    # Drop anything cached against the old configuration.
    for name, fn in (
        ("swift_comet_pipeline.modeling.fluorescence.hydroxyl_gfactor", "read_hydroxyl_gfactor_1au_data"),
        ("swift_comet_pipeline.modeling.fluorescence.hydroxyl_gfactor", "hydroxyl_gfactor_1au"),
        ("swift_comet_pipeline.modeling.spectra.solar.solar_count_rate", "solar_count_rate_in_filter_1au"),
        ("swift_comet_pipeline.photometry.dust.beta_parameter", "beta_parameter"),
    ):
        f = getattr(importlib.import_module(name), fn, None)
        if hasattr(f, "cache_clear"):
            f.cache_clear()
    from . import physics

    physics._band_interpolator.cache_clear()


def configure(data_dir: Optional[os.PathLike] = None, cache_dir: Optional[os.PathLike] = None,
              quality: str = "medium", backend: str = "sbpy") -> dict:
    """Prepare swift_comet_pipeline. Safe to call more than once.

    data_dir: swift_comet_pipeline's ``data/`` folder. Defaults to
        ``$SWIFT_COMET_PIPELINE_DATA``, else the location the package itself
        expects (works for an editable git install).
    cache_dir: where the vectorial model cache goes (``$OH_H2O_CACHE`` or
        ``~/.cache/oh_h2o``).
    quality: vectorial model grid: 'low', 'medium', 'high' or 'very_high'.
    backend: 'sbpy' (pure Python) or 'rust' (needs pyvectorial_au's Rust binary).

    Returns the settings in effect.
    """
    from swift_comet_pipeline.modeling.vectorial.vectorial_model_backend import (
        vectorial_model_backend_init,
    )
    from swift_comet_pipeline.modeling.vectorial.vectorial_model_cache import (
        vectorial_model_cache_init,
    )
    from swift_comet_pipeline.modeling.vectorial.vectorial_model_grid import (
        vectorial_model_grid_quality_init,
    )
    from swift_comet_pipeline.scp_types.primitive.vectorial_model_backend import (
        VectorialModelBackend,
    )
    from swift_comet_pipeline.scp_types.primitive.vectorial_model_grid_quality import (
        VectorialModelGridQuality,
    )

    data_dir = data_dir or os.environ.get(DATA_ENV)
    if data_dir:
        cfg = _config_from(Path(data_dir).expanduser().resolve())
        if not _data_ok(cfg):
            raise FileNotFoundError(
                f"{data_dir} does not look like swift_comet_pipeline's data/ folder "
                "(need fluorescence/Schleicher_Ahearn_1988_Table5.csv and swift_effective_areas/)"
            )
        _install_config(cfg)
        _state["data_dir"] = str(Path(data_dir).expanduser().resolve())
    elif _state["data_dir"] is None:
        cfg = _default_config()
        _state["data_dir"] = (str(Path(cfg.oh_fluorescence_path).parent.parent)
                              if _data_ok(cfg) else None)

    if _state["cache_dir"] is None:
        cache_dir = Path(cache_dir or os.environ.get(CACHE_ENV) or DEFAULT_CACHE).expanduser()
        vectorial_model_cache_init(project_path=cache_dir)
        _state["cache_dir"] = str(cache_dir)

    vectorial_model_grid_quality_init(VectorialModelGridQuality(quality))
    vectorial_model_backend_init(VectorialModelBackend(backend))
    _state["quality"], _state["backend"] = quality, backend
    return dict(_state)


def require_data() -> None:
    """Raise a clear error if the Swift data files are not available."""
    if _state["data_dir"] is None:
        raise FileNotFoundError(
            "swift_comet_pipeline's data files were not found. Clone "
            "https://github.com/sjoset/swift_comet_pipeline and call "
            f"oh_h2o.configure(data_dir='<clone>/data') or set ${DATA_ENV}."
        )


def settings() -> dict:
    return dict(_state)
