import os
from pathlib import Path

import pytest

scp = pytest.importorskip("swift_comet_pipeline")

import oh_h2o  # noqa: E402


def _data_dir():
    d = os.environ.get(oh_h2o.backend.DATA_ENV)
    return Path(d) if d else None


@pytest.fixture(scope="session")
def configured(tmp_path_factory):
    """swift_comet_pipeline set up with low-quality vectorial models and a temp cache."""
    d = _data_dir()
    try:
        s = oh_h2o.configure(data_dir=d, cache_dir=tmp_path_factory.mktemp("vmcache"),
                             quality="low", backend="sbpy")
    except FileNotFoundError as e:
        pytest.skip(str(e))
    if s["data_dir"] is None:
        pytest.skip("swift_comet_pipeline data/ not found: set SWIFT_COMET_PIPELINE_DATA")
    return s
