"""Tests for the OH -> H2O package.

They need swift_comet_pipeline and its data/ folder
(SWIFT_COMET_PIPELINE_DATA=<clone>/data); otherwise they are skipped.
"""

import math
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

import oh_h2o
from oh_h2o import physics

AU_CM = (1 * u.au).to_value(u.cm)
GEO = oh_h2o.Geometry(r_h=1.06, r_dot=-3.0, delta=0.08)  # 46P-like, Dec 2018


# ---------------------------------------------------------------- hand-off fixtures

def _handoff(oh_rates=(0.5, 1.0, 1.5), cont_rates=(2.0, 4.0, 6.0), radii=(1000, 2000, 3000),
             oh_filter="UVW1", cont_filter="V", instrument="UVOT", cont_minutes=5,
             schema="ultrasat-comet-photometry/1"):
    rows = []
    for frame, filt, rates, minute in (("w1.fits[1]", oh_filter, oh_rates, 0),
                                       ("v.fits[1]", cont_filter, cont_rates, cont_minutes)):
        for i, (r, rate) in enumerate(zip(radii, rates)):
            rows.append(dict(frame=frame, instrument=instrument, filter=filt,
                             mid_time_utc=f"2018-12-16T00:{minute:02d}:00.000",
                             aperture_kind="fixed", step=i, is_converged_radius=False,
                             radius_km=float(r), rate=float(rate), rate_err=0.1 * rate,
                             r_h=GEO.r_h, r_dot=GEO.r_dot, delta=GEO.delta))
    t = Table(rows=rows)
    for c, unit in (("radius_km", u.km), ("rate", u.ct / u.s), ("rate_err", u.ct / u.s),
                    ("r_h", u.au), ("r_dot", u.km / u.s), ("delta", u.au)):
        t[c].unit = unit
    t.meta.update(schema=schema, designation="46P")
    return t


# ---------------------------------------------------------------- calibration

def test_swift_beta_matches_uvw1_v_literature(configured):
    cal = oh_h2o.SwiftUVOTCalibration()
    b0 = cal.beta(0.0)
    # Solar-colour dust: UVW1/V count ratio ~0.09 (Xing et al. 2020 use 0.09-0.10).
    assert 0.08 < b0 < 0.10
    # Redder dust puts relatively less light in UVW1.
    assert cal.beta(20.0) < cal.beta(10.0) < b0


def test_swift_flux_factor_is_scp_value(configured):
    assert oh_h2o.SwiftUVOTCalibration().oh_flux_per_count_rate() == pytest.approx(1.2750906e-12)
    with pytest.raises(ValueError):
        oh_h2o.SwiftUVOTCalibration(oh_band="V").oh_flux_per_count_rate()
    with pytest.raises(ValueError, match="unknown UVOT filter"):
        oh_h2o.SwiftUVOTCalibration(oh_band="NUV").beta(0)


def test_fixed_calibration_is_instrument_agnostic():
    cal = oh_h2o.FixedCalibration("ultrasat", "ULTRASAT-OH", "ULTRASAT-CONT", 0.2, 3e-12,
                                  instruments=("ULTRASAT",))
    assert isinstance(cal, oh_h2o.OHCalibration)
    assert cal.accepts("ULTRASAT") and not cal.accepts("UVOT")
    red = oh_h2o.FixedCalibration("x", "A", "B", lambda d: 0.1 * (1 - d / 100), 1.0)
    assert red.beta(10) == pytest.approx(0.09)


# ---------------------------------------------------------------- physics steps

def test_continuum_subtract():
    s = oh_h2o.continuum_subtract(1.0, 0.1, 4.0, 0.4, beta=0.1)
    assert s.value == pytest.approx(0.6)
    assert s.sigma == pytest.approx(math.hypot(0.1, 0.04))


def test_oh_molecules_follows_fluorescence_formula(configured):
    flux = physics._vsd(1e-12, 1e-13)
    n = oh_h2o.oh_molecules(flux, GEO)
    g = oh_h2o.g_factor_1au(GEO.r_dot) / GEO.r_h**2
    expected = 4 * np.pi * (GEO.delta * AU_CM) ** 2 * 1e-12 / g
    assert n.value == pytest.approx(expected, rel=1e-9)
    assert n.sigma / n.value == pytest.approx(0.1)


def test_g_factor_has_swings_effect(configured):
    # Schleicher & A'Hearn 1988: g varies strongly with heliocentric velocity.
    g_in, g_out = oh_h2o.g_factor_1au(-3.0), oh_h2o.g_factor_1au(+3.0)
    assert abs(g_in - g_out) / g_out > 0.05
    assert 1e-15 < oh_h2o.g_factor_1au(0.0) < 1e-14


def test_q_scales_linearly_with_oh(configured):
    n1 = physics._vsd(1e31, 1e30)
    n2 = physics._vsd(2e31, 2e30)
    q1 = oh_h2o.q_h2o_in_aperture(n1, GEO, 20000)
    q2 = oh_h2o.q_h2o_in_aperture(n2, GEO, 20000)
    assert q2.value == pytest.approx(2 * q1.value, rel=1e-6)
    # Same OH count in a smaller aperture means a larger Q; the whole coma gives the smallest.
    q_small = oh_h2o.q_h2o_in_aperture(n1, GEO, 5000)
    q_tot = oh_h2o.q_h2o_total(n1, GEO)
    assert q_small.value > q1.value > q_tot.value > 0


# ---------------------------------------------------------------- hand-off and run

def test_contract_errors():
    with pytest.raises(oh_h2o.ContractError, match="schema"):
        oh_h2o.load_handoff(_handoff(schema="ultrasat-comet-photometry/99"))
    t = _handoff()
    t.remove_column("r_dot")
    with pytest.raises(oh_h2o.ContractError, match="r_dot"):
        oh_h2o.load_handoff(t)
    t = _handoff()
    t["radius_km"].unit = None
    with pytest.raises(oh_h2o.ContractError, match="radius_km"):
        oh_h2o.load_handoff(t)
    t = _handoff()
    t["delta"].unit = u.s
    with pytest.raises(oh_h2o.ContractError, match="delta"):
        oh_h2o.load_handoff(t)


def test_continuum_interpolated_at_same_radius():
    t = oh_h2o.load_handoff(_handoff())
    v = t[t["filter"] == "V"]
    assert oh_h2o.continuum_at_radius(v, 1500) == pytest.approx((3.0, 0.3))
    assert np.isnan(oh_h2o.continuum_at_radius(v, 5000)[0])


def test_pairing_respects_time_window():
    t = oh_h2o.load_handoff(_handoff(cont_minutes=30))
    assert oh_h2o.pair_frames(t, "UVW1", "V", max_dt_minutes=60)[0][2] == pytest.approx(30)
    assert oh_h2o.pair_frames(t, "UVW1", "V", max_dt_minutes=10) == []


def test_run_swift(configured, tmp_path):
    cal = oh_h2o.SwiftUVOTCalibration()
    res = oh_h2o.run(_handoff(), cal, dust_redness=[0.0, 10.0])
    assert len(res) == 6  # 3 apertures x 2 rednesses
    r = res[(res["radius_km"] == 2000) & (res["dust_redness"] == 0.0)][0]
    beta = cal.beta(0.0)
    assert r["oh_rate"] == pytest.approx(1.0 - beta * 4.0)
    assert r["oh_flux"] == pytest.approx(r["oh_rate"] * 1.2750906e-12, rel=1e-6)
    assert r["q_h2o"] > 0 and r["q_h2o_err"] > 0
    assert res["q_h2o"].unit == 1 / u.s and res.meta["calibration"] == "swift-uvot"
    path = oh_h2o.write_results(res, tmp_path / "46P_water.ecsv")
    back = Table.read(path, format="ascii.ecsv")
    assert back.meta["schema"] == "ultrasat-oh-h2o/1"
    assert back["q_h2o"].quantity[0].unit == 1 / u.s


def test_run_rejects_wrong_instrument(configured):
    with pytest.raises(oh_h2o.ContractError, match="instrument"):
        oh_h2o.run(_handoff(instrument="TESTSCOPE"), oh_h2o.SwiftUVOTCalibration())
    res = oh_h2o.run(_handoff(instrument="TESTSCOPE"), oh_h2o.SwiftUVOTCalibration(),
                     check_instrument=False, total_q=False)
    assert len(res) == 3 and np.all(np.isnan(res["q_h2o_total"]))


def test_run_with_another_instrument(configured):
    cal = oh_h2o.FixedCalibration("ultrasat", "OH", "CONT", beta_value=0.25,
                                  flux_per_count_rate=2e-12, instruments=("ULTRASAT",))
    res = oh_h2o.run(_handoff(oh_filter="OH", cont_filter="CONT", instrument="ULTRASAT"), cal)
    assert len(res) == 3
    r = res[res["radius_km"] == 2000][0]
    assert r["oh_rate"] == pytest.approx(1.0 - 0.25 * 4.0)  # exactly zero OH signal
    assert r["q_h2o"] == 0 and r["q_h2o_err"] > 0
    r = res[res["radius_km"] == 3000][0]
    assert r["oh_rate"] == pytest.approx(1.5 - 0.25 * 6.0)


def test_q_degenerate_inputs_match_linear_scaling(configured):
    q = oh_h2o.q_h2o_in_aperture(physics._vsd(2e31, 1e30), GEO, 20000)
    q_edge = oh_h2o.q_h2o_in_aperture(physics._vsd(1e30, 1e30), GEO, 20000)  # N = sigma
    assert q_edge.value == pytest.approx(q.value / 20, rel=1e-6)
    assert q_edge.sigma == pytest.approx(q.sigma, rel=1e-3)
    q0 = oh_h2o.q_h2o_total(physics._vsd(0.0, 1e30), GEO)
    assert q0.value == 0 and q0.sigma == pytest.approx(oh_h2o.q_h2o_total(physics._vsd(2e31, 1e30), GEO).sigma)


def test_end_to_end_from_test_bench(configured, tmp_path):
    """Synthetic UVW1 + V images through comet_testbench, then through this package."""
    import importlib.util

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    tb = pytest.importorskip("comet_testbench")
    spec = importlib.util.spec_from_file_location("_bench_helpers", root / "tests" / "conftest.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    coma, write_fits = helpers.coma_image, helpers.write_fits
    w1 = write_fits(tmp_path / "w1.fits", coma(200, 200, amp=40, sigma=5, seed=2), filt="UVW1")
    v = write_fits(tmp_path / "v.fits", coma(200, 200, amp=60, sigma=5, seed=3), filt="V")
    cfg = tb.BenchConfig(fixed_radii=[1000, 2000, 4000], fixed_unit="km", iter_start=2,
                         iter_step=2, iter_max=60,
                         background={"method": "annulus", "r_in": 90, "r_out": 130,
                                     "unit": "arcsec"})
    eph = helpers.StubEphemeris({"99P": (200, 200)})
    meas = tb.run([w1, v], cfg, ephemeris=eph, sb_ident=lambda *a, **k: [])
    (path,) = tb.write_results(meas, tmp_path / "out")
    res = oh_h2o.run(path, oh_h2o.SwiftUVOTCalibration(), check_instrument=False)
    assert len(res) > 0
    assert set(res["aperture_kind"]) <= {"fixed", "iterative"}
    assert np.all(np.isfinite(res["q_h2o"]))
