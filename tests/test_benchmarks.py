"""Benchmark tests on synthetic images (offline).

Numbered as in the benchmark plan:
1 ephemeris recovery, 2 non-sidereal motion, 3 edge and crowded fields,
4 faint-limit completeness, 5 aperture stability, 6 OH package round trip,
7 fields that should return nothing.
The same checks on real Swift frames are in test_benchmarks_swift.py.
"""

import numpy as np
import pytest

from comet_testbench import (BenchConfig, InstrumentAdapter, find_comets, iterative_aperture_photometry,
                             load_frames, run, write_results)
from comet_testbench.benchmarks import (aperture_stability, blank_sky_positions, coma_model,
                                        completeness_curve, ephemeris_offset, fifty_percent_rate,
                                        inject_coma, motion_direction_pix, recover)
from comet_testbench.ephemeris import ObservingGeometry
from comet_testbench.photometry import fixed_aperture_photometry

from conftest import SHAPE, StubEphemeris, coma_image, geometry_at, make_wcs, write_fits
import stub_oh_package as oh_stub

GENERIC = InstrumentAdapter()


def moving_geometry(x, y, ra_rate, dec_rate, target="99P"):
    g = geometry_at(x, y, target=target)
    g.ra_rate, g.dec_rate = ra_rate, dec_rate
    return g


class MovingEphemeris(StubEphemeris):
    def __init__(self, positions, rates):
        super().__init__(positions)
        self.rates = rates

    def __call__(self, des, time, observer):
        g = super().__call__(des, time, observer)
        g.ra_rate, g.dec_rate = self.rates.get(des, (0.0, 0.0))
        return g


def trailed_coma(x_mid, y_mid, length_pix, angle_deg, amp=40.0, sigma=4.0, bg=2.0, exposure=100.0,
                 seed=5, n=41):
    """A Gaussian coma smeared along a line centred on (x_mid, y_mid)."""
    th = np.deg2rad(angle_deg)
    yy, xx = np.indices(SHAPE, dtype=float)
    rate = np.zeros(SHAPE)
    for s in np.linspace(-0.5, 0.5, n) * length_pix:
        rate += amp / n * np.exp(-0.5 * ((xx - x_mid - s * np.cos(th)) ** 2
                                         + (yy - y_mid - s * np.sin(th)) ** 2) / sigma**2)
    return np.random.default_rng(seed).poisson((rate + bg) * exposure).astype(float)


def add_star(counts, x, y, total_rate, sigma=1.5, exposure=100.0, seed=9):
    yy, xx = np.indices(SHAPE, dtype=float)
    star = total_rate / (2 * np.pi * sigma**2) * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / sigma**2)
    return counts + np.random.default_rng(seed).poisson(star * exposure)


# --- 1. ephemeris recovery ----------------------------------------------------

@pytest.mark.parametrize("dx,dy", [(0.0, 0.0), (2.5, -1.5), (-4.0, 3.0)])
def test_ephemeris_offset_measures_known_error(tmp_path, dx, dy):
    p = write_fits(tmp_path / "img.fits", coma_image(200 + dx, 200 + dy, amp=60, sigma=3, seed=11))
    frame = load_frames(p)[0]
    eph = StubEphemeris({"99P": (200, 200)})
    (c,) = find_comets(frame, adapter=GENERIC, ephemeris=eph, use_sb_ident=False)
    o = ephemeris_offset(frame, c, box_arcsec=8, adapter=GENERIC)
    assert o.x_obs - 200 == pytest.approx(dx, abs=0.15)
    assert o.y_obs - 200 == pytest.approx(dy, abs=0.15)
    assert o.offset_arcsec == pytest.approx(np.hypot(dx, dy), abs=0.2)
    assert o.peak_snr > 50


def test_ephemeris_offset_splits_along_and_cross_track(tmp_path):
    # Comet moves +x on the detector (RA decreasing on this east-left WCS),
    # and sits 3 px ahead and 1 px to the side of the prediction.
    p = write_fits(tmp_path / "img.fits", coma_image(203, 201, amp=60, sigma=3, seed=12))
    frame = load_frames(p)[0]
    eph = MovingEphemeris({"99P": (200, 200)}, {"99P": (-100.0, 0.0)})
    (c,) = find_comets(frame, adapter=GENERIC, ephemeris=eph, use_sb_ident=False)
    ux, uy = motion_direction_pix(frame, c)
    assert ux == pytest.approx(1.0, abs=1e-3) and uy == pytest.approx(0.0, abs=1e-3)
    o = ephemeris_offset(frame, c, box_arcsec=8, adapter=GENERIC)
    assert o.along_track_arcsec == pytest.approx(3.0, abs=0.2)
    assert o.cross_track_arcsec == pytest.approx(1.0, abs=0.2)


# --- 2. non-sidereal motion -----------------------------------------------------

def test_trail_length_from_rates_and_elapsed_time(tmp_path):
    p = write_fits(tmp_path / "img.fits", coma_image(200, 200), extra={"TELAPSE": 360.0})
    frame = load_frames(p)[0]
    # 30 arcsec/h * 360 s = 3 arcsec = 3 px; hypot(18, 24) = 30.
    eph = MovingEphemeris({"99P": (200, 200)}, {"99P": (18.0, 24.0)})
    (c,) = find_comets(frame, adapter=GENERIC, ephemeris=eph, use_sb_ident=False)
    assert c.trail_pix == pytest.approx(3.0, rel=1e-6)


@pytest.mark.parametrize("length", [0, 10, 30])
def test_trailed_coma_centroid_is_mid_exposure(tmp_path, length):
    """A mid-exposure ephemeris lands on the trail centre, however long the trail."""
    p = write_fits(tmp_path / "img.fits", trailed_coma(200, 200, length, 30.0))
    frame = load_frames(p)[0]
    eph = MovingEphemeris({"99P": (200, 200)}, {"99P": (-60.0, 35.0)})
    (c,) = find_comets(frame, adapter=GENERIC, ephemeris=eph, use_sb_ident=False)
    o = ephemeris_offset(frame, c, box_arcsec=max(8, length), adapter=GENERIC)
    assert o.offset_pix < 0.5


def test_start_of_exposure_time_shows_as_along_track_offset(tmp_path):
    """Using the start time instead of mid-exposure puts the prediction half a trail behind."""
    L, ang = 20.0, 0.0
    p = write_fits(tmp_path / "img.fits", trailed_coma(200, 200, L, ang))
    frame = load_frames(p)[0]
    # Prediction at the trail start; motion along +x.
    eph = MovingEphemeris({"99P": (200 - L / 2, 200)}, {"99P": (-100.0, 0.0)})
    (c,) = find_comets(frame, adapter=GENERIC, ephemeris=eph, use_sb_ident=False)
    o = ephemeris_offset(frame, c, box_arcsec=L, adapter=GENERIC)
    assert o.along_track_arcsec == pytest.approx(L / 2, abs=1.0)
    assert abs(o.cross_track_arcsec) < 0.5


@pytest.mark.parametrize("length", [0, 15, 40])
def test_growth_curve_recovers_total_flux_of_trailed_coma(tmp_path, length):
    amp, sigma = 40.0, 4.0
    p = write_fits(tmp_path / "img.fits", trailed_coma(200, 200, length, 45.0, amp=amp, sigma=sigma))
    frame = load_frames(p)[0]
    gc = iterative_aperture_photometry(frame, 200, 200, 2, 2, 70, "pix", adapter=GENERIC,
                                       background={"method": "annulus", "r_in": 100, "r_out": 140})
    assert gc.converged
    # The converged aperture must be larger than the half-trail.
    assert gc.best.radius_pix > length / 2
    assert gc.best.rate == pytest.approx(2 * np.pi * sigma**2 * amp, rel=0.03)


# --- 3. edge and crowded fields -------------------------------------------------

def test_comet_near_edge_is_found_and_coverage_reported(tmp_path):
    p = write_fits(tmp_path / "img.fits", coma_image(8, 200, amp=40, sigma=4))
    frame = load_frames(p)[0]
    eph = StubEphemeris({"99P": (8, 200)})
    assert len(find_comets(frame, adapter=GENERIC, ephemeris=eph, use_sb_ident=False)) == 1
    assert find_comets(frame, adapter=GENERIC, ephemeris=eph, use_sb_ident=False,
                       edge_margin_pix=10) == []
    res = fixed_aperture_photometry(frame, 8, 200, [5, 20], "pix", adapter=GENERIC,
                                    background={"method": "fixed", "value": 2.0})
    assert res[0].coverage == pytest.approx(1.0)
    # A 20 px aperture whose centre is 8.5 px inside the edge: the cut-off
    # circular segment is 23.8% of the area.
    assert res[1].coverage == pytest.approx(0.762, abs=0.01)


def test_comet_on_unexposed_pixels_is_not_found(tmp_path):
    counts = coma_image(200, 200)
    counts[:, 180:220] = np.nan  # e.g. a chip gap
    p = write_fits(tmp_path / "img.fits", counts)
    frame = load_frames(p)[0]
    eph = StubEphemeris({"99P": (200, 200)})
    assert find_comets(frame, adapter=GENERIC, ephemeris=eph, use_sb_ident=False) == []


def test_partial_coverage_flagged_next_to_gap(tmp_path):
    counts = coma_image(200, 200, amp=40, sigma=4)
    counts[:, 210:] = np.nan
    p = write_fits(tmp_path / "img.fits", counts)
    frame = load_frames(p)[0]
    res = fixed_aperture_photometry(frame, 200, 200, [5, 20], "pix", adapter=GENERIC,
                                    background={"method": "fixed", "value": 2.0})
    assert res[0].coverage == pytest.approx(1.0)
    assert res[1].coverage < 0.9


@pytest.mark.parametrize("star_dist,star_rate", [(25, 2000.0), (40, 20000.0)])
def test_bright_star_nearby_does_not_capture_centroid(tmp_path, star_dist, star_rate):
    amp, sigma = 40.0, 4.0
    counts = add_star(coma_image(200, 200, amp=amp, sigma=sigma), 200 + star_dist, 200, star_rate)
    p = write_fits(tmp_path / "img.fits", counts)
    frame = load_frames(p)[0]
    eph = StubEphemeris({"99P": (201.5, 199.0)})  # prediction a little off
    (c,) = find_comets(frame, adapter=GENERIC, ephemeris=eph, use_sb_ident=False)
    o = ephemeris_offset(frame, c, box_arcsec=8, adapter=GENERIC)
    assert np.hypot(o.x_obs - 200, o.y_obs - 200) < 0.5
    gc = iterative_aperture_photometry(frame, c.x, c.y, 2, 2, 60, "pix", adapter=GENERIC,
                                       background={"method": "annulus", "r_in": 100, "r_out": 140},
                                       recenter_box=8, max_shift=6)
    assert np.hypot(gc.best.x - 200, gc.best.y - 200) < 0.5
    assert gc.best.radius_pix < star_dist
    assert gc.best.rate == pytest.approx(2 * np.pi * sigma**2 * amp, rel=0.03)


def test_crowded_background_annulus_is_robust(tmp_path):
    amp, sigma, bg = 40.0, 4.0, 2.0
    counts = coma_image(200, 200, amp=amp, sigma=sigma, bg=bg)
    rng = np.random.default_rng(3)
    for k in range(60):  # stars scattered through the background annulus
        r, th = rng.uniform(95, 135), rng.uniform(0, 2 * np.pi)
        counts = add_star(counts, 200 + r * np.cos(th), 200 + r * np.sin(th), rng.uniform(5, 200), seed=k)
    p = write_fits(tmp_path / "img.fits", counts)
    frame = load_frames(p)[0]
    res = fixed_aperture_photometry(frame, 200, 200, [20], "pix", adapter=GENERIC,
                                    background={"method": "annulus", "r_in": 90, "r_out": 140})
    assert res[0].background == pytest.approx(bg, abs=0.05)
    assert res[0].rate == pytest.approx(2 * np.pi * sigma**2 * amp, rel=0.03)


# --- 4. faint-limit completeness ------------------------------------------------

def _blank_frame(tmp_path, bg=2.0, exposure=100.0):
    counts = np.random.default_rng(21).poisson(np.full(SHAPE, bg * exposure)).astype(float)
    return load_frames(write_fits(tmp_path / "blank.fits", counts, exposure=exposure))[0]


def test_injected_coma_has_the_requested_rate(tmp_path):
    frame = _blank_frame(tmp_path)
    f = inject_coma(frame, 200, 200, rate_in_r=500.0, r_ref_pix=10, r_max_pix=60,
                    rng=np.random.default_rng(1))
    res = fixed_aperture_photometry(f, 200, 200, [10, 20], "pix", adapter=GENERIC,
                                    background={"method": "fixed", "value": 2.0})
    assert res[0].rate == pytest.approx(500.0, rel=0.05)
    assert res[1].rate == pytest.approx(1000.0, rel=0.05)  # 1/rho: rate grows linearly with r
    m = coma_model(SHAPE, 200, 200, 300.0, 5.0, 40.0, profile="gaussian")
    assert m.sum() == pytest.approx(300.0, rel=0.01)


def test_completeness_curve_on_flat_background(tmp_path):
    bg, t, r = 2.0, 100.0, 6.0
    frame = _blank_frame(tmp_path, bg, t)
    pos = blank_sky_positions(frame, 12, edge_pix=60, min_spacing_pix=40, adapter=GENERIC, seed=2)
    assert len(pos) == 12
    # Rate inside r giving SNR = 5 from Poisson noise alone:
    # S t / sqrt(S t + B t) = 5 with B = bg * pi r^2.
    B = bg * np.pi * r**2
    s5 = (25 + np.sqrt(625 + 100 * B * t)) / (2 * t)
    rates = s5 * np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    table, fp, trials = completeness_curve(frame, pos, rates, r, adapter=GENERIC, seed=4)
    comp = [row["completeness"] for row in table]
    assert comp[0] <= 0.1 and comp[-1] == 1.0
    assert all(b >= a - 0.1 for a, b in zip(comp, comp[1:]))  # non-decreasing within noise
    assert fp == 0.0
    # Measured 50% point within a factor of 2 of the Poisson expectation.
    r50 = fifty_percent_rate(table)
    assert s5 / 2 < r50 < s5 * 2
    assert len(trials) == len(pos) * len(rates)


def test_recover_rejects_star_next_to_injection(tmp_path):
    frame = _blank_frame(tmp_path)
    counts = add_star(frame.data, 215, 200, 3000.0)
    frame.data = counts
    ok, snr, off, _ = recover(frame, 200, 200, r_ref_pix=6, adapter=GENERIC, box_pix=20)
    assert not ok and off > 6


# --- 5. aperture stability --------------------------------------------------------

BACKGROUNDS = {
    "annulus_100_140": {"method": "annulus", "r_in": 100, "r_out": 140, "unit": "arcsec"},
    "annulus_140_180": {"method": "annulus", "r_in": 140, "r_out": 180, "unit": "arcsec"},
    "global": {"method": "global", "mask_radius": 100, "unit": "arcsec"},
}


def test_stability_on_one_over_rho_coma(tmp_path):
    amp, bg = 5.0, 2.0
    p = write_fits(tmp_path / "img.fits", coma_image(200, 200, "rho", amp=amp, bg=bg))
    frame = load_frames(p)[0]
    backgrounds = {"true": {"method": "fixed", "value": bg}, **BACKGROUNDS}
    rep = aperture_stability(frame, 200, 200, [5, 10, 20, 30, 40, 60], backgrounds, r_test_arcsec=20,
                             adapter=GENERIC)
    assert rep.worst_step_sigma < 3
    # Rate / radius is flat for a 1/rho coma when the background is right.
    per_r = np.array(rep.rates) / np.array(rep.radii_arcsec)
    assert per_r == pytest.approx(2 * np.pi * amp, rel=0.02)
    # A 1/rho coma still puts amp/rho of light into any sky annulus, so a
    # measured background is biased high by about that much. The spread across
    # background choices should match that bias, not exceed it.
    bias = 2 * np.pi * amp * 20 - min(v for k, v in rep.background_rates.items() if k != "true")
    expected = amp / 120 * np.pi * 20**2  # coma light at ~120" times the aperture area
    assert bias == pytest.approx(expected, rel=0.5)
    assert rep.background_spread < 0.1


def test_stability_catches_oversubtracted_background(tmp_path):
    p = write_fits(tmp_path / "img.fits", coma_image(200, 200, amp=40, sigma=4, bg=2.0))
    frame = load_frames(p)[0]
    good = aperture_stability(frame, 200, 200, [10, 20, 40, 60, 80], {"ok": BACKGROUNDS["global"]},
                              adapter=GENERIC)
    bad = aperture_stability(frame, 200, 200, [10, 20, 40, 60, 80],
                             {"high": {"method": "fixed", "value": 2.1}}, adapter=GENERIC)
    assert good.worst_step_sigma < 3
    assert bad.worst_step_sigma > 3


# --- 6. round trip through a stub of the OH/H2O package -----------------------------

def _two_filter_handoff(tmp_path, stub_sb_ident, **cfg_kw):
    w1 = write_fits(tmp_path / "w1.fits", coma_image(200, 200, amp=40, sigma=5, seed=2), filt="UVW1")
    v = write_fits(tmp_path / "v.fits", coma_image(200, 200, amp=60, sigma=5, seed=3), filt="V")
    kw = dict(fixed_radii=[1000, 2000, 4000], fixed_unit="km", iter_start=2, iter_step=2, iter_max=60,
              background={"method": "annulus", "r_in": 90, "r_out": 130, "unit": "arcsec"})
    kw.update(cfg_kw)
    res = run([w1, v], BenchConfig(**kw), ephemeris=StubEphemeris({"99P": (200, 200)}),
              sb_ident=stub_sb_ident)
    (path,) = write_results(res, tmp_path / "out")
    return res, path


def test_oh_stub_round_trip(tmp_path, stub_sb_ident):
    res, path = _two_filter_handoff(tmp_path, stub_sb_ident)
    (r,) = oh_stub.model(path)
    w1 = next(m for m in res if m.filter == "UVW1")
    v = next(m for m in res if m.filter == "V")
    assert r.designation == "99P" and r.oh_frame == w1.frame.source and r.continuum_frame == v.frame.source
    # What went in comes out: rate, radius and geometry.
    assert r.radius_km == pytest.approx(w1.growth.best.radius_km)
    v_same_r = min(v.fixed, key=lambda a: abs(a.radius_km - r.radius_km))
    assert r.oh_rate == pytest.approx(w1.growth.best.rate - 0.1 * v_same_r.rate)
    g = w1.comet.geometry
    assert (r.r_h, r.delta, r.r_dot) == pytest.approx((g.r_h, g.delta, g.r_dot))
    # Physical radius matching works across filters.
    (rk,) = oh_stub.model(path, radius_km=2000)
    assert rk.radius_km == pytest.approx(2000, rel=1e-6)


def test_oh_stub_rejects_broken_contract(tmp_path, stub_sb_ident):
    from astropy.table import Table

    _, path = _two_filter_handoff(tmp_path, stub_sb_ident)
    t = Table.read(path, format="ascii.ecsv")
    t.meta["schema"] = "ultrasat-comet-photometry/99"
    bad_schema = tmp_path / "bad_schema.ecsv"
    t.write(bad_schema, format="ascii.ecsv")
    with pytest.raises(oh_stub.ContractError, match="schema"):
        oh_stub.load(bad_schema)

    t = Table.read(path, format="ascii.ecsv")
    t.remove_column("r_dot")
    no_col = tmp_path / "no_col.ecsv"
    t.write(no_col, format="ascii.ecsv")
    with pytest.raises(oh_stub.ContractError, match="r_dot"):
        oh_stub.load(no_col)

    t = Table.read(path, format="ascii.ecsv")
    t["radius_km"].unit = None
    no_unit = tmp_path / "no_unit.ecsv"
    t.write(no_unit, format="ascii.ecsv")
    with pytest.raises(oh_stub.ContractError, match="radius_km"):
        oh_stub.load(no_unit)


def test_handoff_units_survive_conversion(tmp_path, stub_sb_ident):
    import astropy.units as u

    _, path = _two_filter_handoff(tmp_path, stub_sb_ident)
    t = oh_stub.load(path)
    assert t["delta"].unit == u.au and t["radius_km"].unit == u.km
    assert t["radius_km"].quantity.to(u.m)[0].value == pytest.approx(1000 * t["radius_km"][0])


def test_oh_stub_skips_unpaired_frames(tmp_path, stub_sb_ident):
    w1 = write_fits(tmp_path / "w1.fits", coma_image(200, 200, amp=40, sigma=5, seed=2), filt="UVW1")
    res = run([w1], BenchConfig(iter_start=2, iter_step=2, iter_max=60, fixed_radii=[1000],
                                background={"method": "annulus", "r_in": 90, "r_out": 130,
                                            "unit": "arcsec"}),
              ephemeris=StubEphemeris({"99P": (200, 200)}), sb_ident=stub_sb_ident)
    (path,) = write_results(res, tmp_path / "out")
    assert oh_stub.model(path) == []


# --- 7. fields that should return nothing -------------------------------------------

def test_empty_field_returns_nothing(tmp_path):
    p = write_fits(tmp_path / "img.fits", coma_image(200, 200, amp=0.0), obj="GRB 200101A")
    eph = StubEphemeris({"12P": (5000, 5000)})
    res = run([p], BenchConfig(), ephemeris=eph, sb_ident=lambda *a, **k: ["12P"])
    assert res == []
    assert write_results(res, tmp_path / "out") == []
    assert not list((tmp_path / "out").rglob("*.ecsv"))


def test_mislabelled_target_returns_nothing(tmp_path, stub_sb_ident):
    """OBJECT names a comet but the telescope pointed elsewhere (as in two Swift '41P' visits)."""
    p = write_fits(tmp_path / "img.fits", coma_image(200, 200), obj="COMET41P/T-G-K")
    eph = StubEphemeris({"41P": (-90000, 400000)})
    res = run([p], BenchConfig(), ephemeris=eph, sb_ident=stub_sb_ident)
    assert res == [] and eph.calls  # it was looked up, then rejected


def test_blank_sky_gives_no_false_detections(tmp_path):
    frame = _blank_frame(tmp_path)
    pos = blank_sky_positions(frame, 30, edge_pix=60, adapter=GENERIC, seed=8)
    assert sum(recover(frame, x, y, 6.0, adapter=GENERIC)[0] for x, y in pos) == 0
