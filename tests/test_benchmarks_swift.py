"""Benchmark tests on real Swift/UVOT frames of 46P and 41P (UVW1 and V).

Run with ``pytest -m network``. Frames (about 150 MB) are cached in
``$COMET_BENCH_CACHE`` (default ``data/swift``), so only the first run downloads.
The frame list and why each is in it are in
:data:`comet_testbench.benchmarks.SWIFT_BENCHMARK_FRAMES`.
"""

import os

import numpy as np
import pytest

from comet_testbench import BenchConfig, adapter_for, find_comets, load_frames, run, write_results
from comet_testbench.benchmarks import (SWIFT_BENCHMARK_FRAMES, SWIFT_NULL_FRAMES, aperture_stability,
                                        blank_sky_positions, completeness_curve, ephemeris_offset,
                                        fetch_swift_benchmark, fifty_percent_rate,
                                        timing_tolerance_arcsec)
from comet_testbench.ephemeris import horizons_provider

import stub_oh_package as oh_stub

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def swift_frames():
    dest = os.environ.get("COMET_BENCH_CACHE", "data/swift")
    got = fetch_swift_benchmark(dest)
    have = {obsid for _, _, obsid, _ in got}
    missing = set(SWIFT_BENCHMARK_FRAMES) | set(SWIFT_NULL_FRAMES)
    assert have == missing, f"missing from archive: {missing - have}"
    return {obsid: (sky, ex, filt) for sky, ex, obsid, filt in got}


@pytest.fixture(scope="module")
def comets(swift_frames):
    """(frame, adapter, comet) for every benchmark frame, placed by Horizons for Swift."""
    eph = horizons_provider()
    out = {}
    for obsid, (comet, filt, _) in SWIFT_BENCHMARK_FRAMES.items():
        sky, ex, _ = swift_frames[obsid]
        frame = load_frames(sky, ex)[0]
        assert frame.filter == filt
        a = adapter_for(frame)
        found = find_comets(frame, adapter=a, use_sb_ident=False, ephemeris=eph)
        assert [c.designation for c in found] == [comet], obsid
        out[obsid] = (frame, a, found[0])
    return out


# --- 1. ephemeris recovery on real frames ----------------------------------------

def test_ephemeris_recovery(comets):
    """Horizons (Swift-centred) lands on the coma in every frame, both filters."""
    offsets = []
    for obsid, (frame, a, c) in comets.items():
        o = ephemeris_offset(frame, c, box_arcsec=5, adapter=a)
        tol = 1.5 + timing_tolerance_arcsec(frame, c)
        assert o.peak_snr > 20, obsid
        assert o.offset_arcsec < tol, (obsid, o.offset_arcsec, tol)
        offsets.append(o.offset_arcsec)
    # Typical offset is at the level of UVOT's own astrometry (~1 arcsec).
    assert np.median(offsets) < 1.1


def test_recovery_same_in_both_filters(comets):
    """UVW1 and V of the same visit agree; the search does not depend on filter."""
    pairs = [("00094318002", "00094319001"), ("00010086002", "00010086001")]
    for v_id, w_id in pairs:
        ov = ephemeris_offset(*[comets[v_id][i] for i in (0, 2)], box_arcsec=5, adapter=comets[v_id][1])
        ow = ephemeris_offset(*[comets[w_id][i] for i in (0, 2)], box_arcsec=5, adapter=comets[w_id][1])
        assert abs(ov.offset_arcsec - ow.offset_arcsec) < 1.0, (v_id, w_id)


# --- 2. non-sidereal motion ------------------------------------------------------

def test_fast_mover_is_still_found_and_measured(comets):
    frame, a, c = comets["00094421002"]  # 46P, 0.08 au, 996 s elapsed
    assert c.trail_pix > 100  # the bench reports the long trail
    o = ephemeris_offset(frame, c, box_arcsec=5, adapter=a)
    assert o.offset_arcsec < 1.5
    res = run([_paths(frame)], _cfg(), ephemeris=horizons_provider(),
              sb_ident=lambda *a, **k: [])
    (m,) = res
    assert m.growth is not None and len(m.fixed) == 4


def test_offset_on_fast_mover_is_along_track(comets):
    """Near perigee the residual lies along the motion: a timing effect, not a pointing one."""
    frame, a, c = comets["00094422001"]  # exposure is 122 s of 276 s elapsed
    o = ephemeris_offset(frame, c, box_arcsec=5, adapter=a)
    assert abs(o.along_track_arcsec) > 2 * abs(o.cross_track_arcsec)


# --- 3. crowded fields ------------------------------------------------------------

@pytest.mark.parametrize("obsid", ["00094431002", "00094432001"])
def test_bright_star_near_coma_does_not_capture_bench(comets, obsid):
    frame, a, c = comets[obsid]
    (m,) = run([_paths(frame)], _cfg(), ephemeris=horizons_provider(), sb_ident=lambda *a, **k: [])
    o = ephemeris_offset(frame, c, box_arcsec=5, adapter=a)
    # The bench's own recentring (10" box, 20" cap) ends on the coma, not the star.
    x, y = m.growth.apertures[0].x, m.growth.apertures[0].y
    assert np.hypot(x - o.x_obs, y - o.y_obs) * frame.pixel_scale_arcsec < 2.0


# --- 4. faint limit in real backgrounds ------------------------------------------------

@pytest.mark.parametrize("obsid", ["00010086001", "00010086002"])
def test_completeness_in_real_background(comets, obsid):
    frame, a, c = comets[obsid]
    r = 5.0 / frame.pixel_scale_arcsec
    pos = blank_sky_positions(frame, 25, avoid=[(c.x, c.y)], avoid_radius_pix=250, edge_pix=60,
                              min_spacing_pix=30, max_star_rate=0.5, r_check_pix=10, adapter=a, seed=1)
    assert len(pos) >= 20
    bg = float(np.nanmedian(a.count_rate_image(frame)))
    B, t = bg * np.pi * r**2, frame.exposure
    s5 = (25 + np.sqrt(625 + 100 * B * t)) / (2 * t)  # Poisson-only SNR=5 rate
    table, fp, _ = completeness_curve(frame, pos, s5 * np.array([0.25, 0.5, 1, 2, 4, 8]), r,
                                      adapter=a, seed=3)
    comp = [row["completeness"] for row in table]
    assert comp[0] <= 0.15 and comp[-1] == 1.0 and comp[-2] >= 0.95
    assert all(b >= a_ - 0.15 for a_, b in zip(comp, comp[1:]))
    assert fp <= 0.1
    # Real backgrounds (faint stars, coma gradient) cost at most a factor 3 over pure Poisson.
    assert s5 / 3 < fifty_percent_rate(table) < 3 * s5


# --- 5. aperture photometry stability ------------------------------------------------

BACKGROUNDS = {
    "annulus_150_200": {"method": "annulus", "r_in": 150, "r_out": 200, "unit": "arcsec"},
    "annulus_200_250": {"method": "annulus", "r_in": 200, "r_out": 250, "unit": "arcsec"},
    "global": {"method": "global", "mask_radius": 150, "unit": "arcsec"},
}


def test_aperture_stability_on_real_comae(comets):
    for obsid, (frame, a, c) in comets.items():
        o = ephemeris_offset(frame, c, box_arcsec=5, adapter=a)
        rep = aperture_stability(frame, o.x_obs, o.y_obs, [5, 10, 20, 30, 40, 60, 80], BACKGROUNDS,
                                 r_test_arcsec=20, adapter=a)
        # Enclosed rate never drops with radius beyond its noise.
        assert rep.worst_step_sigma < 3, obsid
        # For these nearby comets (0.08-0.21 au) the coma fills the UVOT field,
        # so every background choice contains coma light. Nearer annuli give
        # higher backgrounds and lower coma rates; the spread is a property of
        # the data that the OH package must budget for, recorded here so a
        # change in it is noticed.
        r = rep.background_rates
        assert r["annulus_150_200"] <= r["annulus_200_250"] * 1.01, obsid
        assert rep.background_spread < 0.4, (obsid, rep.background_spread)


# --- 6. round trip through the OH package stub -----------------------------------------

def test_oh_stub_round_trip_on_real_pairs(swift_frames, tmp_path):
    pairs = [("00094319001", "00094318002"), ("00010086001", "00010086002")]
    for w_id, v_id in pairs:
        imgs = [swift_frames[w_id][:2], swift_frames[v_id][:2]]
        res = run(imgs, _cfg(), ephemeris=horizons_provider(), sb_ident=lambda *a, **k: [])
        (path,) = write_results(res, tmp_path / w_id)
        (r,) = oh_stub.model(path, radius_km=10000)
        assert r.dt_minutes < 15
        assert r.radius_km == pytest.approx(10000, rel=1e-6)
        assert 0.05 < r.delta < 0.3 and 1.0 < r.r_h < 1.2


# --- 7. fields that should return nothing ------------------------------------------------

@pytest.mark.parametrize("obsid", list(SWIFT_NULL_FRAMES))
def test_mislabelled_visits_return_nothing(swift_frames, obsid):
    res = run([swift_frames[obsid][:2]], _cfg(), ephemeris=horizons_provider(),
              sb_ident=lambda *a, **k: [])
    assert res == []


# --- helpers -------------------------------------------------------------------------------

def _cfg():
    return BenchConfig(use_sb_ident=False, iter_max=60)


def _paths(frame):
    """(sky, exposure map) paths for a loaded frame."""
    from pathlib import Path

    cache = Path(os.environ.get("COMET_BENCH_CACHE", "data/swift"))
    name = frame.source.split("[")[0]
    sky = next(cache.glob(f"*/{name}"))
    return sky, sky.with_name(name.replace("_sk", "_ex"))
