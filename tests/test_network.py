"""Live checks against HEASARC and JPL. Run with: pytest -m network"""

import pytest

pytestmark = pytest.mark.network


def test_46p_in_swift_uvw1_image(tmp_path):
    from comet_testbench import BenchConfig, run
    from comet_testbench.swift_archive import download_images, find_observations

    obs = [o for o in find_observations("%WIRTANEN%", "2018-11-28", "2018-11-29")
           if o.obsid == "00094319001"]
    got = download_images(obs, ["UVW1"], tmp_path)
    assert got, "no UVW1 image downloaded"
    sky, ex, _, _ = got[0]
    res = run([(sky, ex)], BenchConfig(iter_max=60))
    assert any(m.comet.designation == "46P" for m in res)
    m = next(m for m in res if m.comet.designation == "46P")
    # Horizons (Swift-centred) prediction should land within a few pixels of the coma peak.
    assert m.growth.centroid_shift_pix < 5
