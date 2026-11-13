import json

import numpy as np
import pytest
from astropy.time import Time

from comet_testbench import (HANDOFF_SCHEMA, BenchConfig, InstrumentAdapter, estimate_background,
                             find_comets, fixed_aperture_photometry, iterative_aperture_photometry,
                             load_frames, normalize_designation, read_handoff, run, write_results)
from comet_testbench.instruments import SwiftUVOT, adapter_for, uvot_coincidence_correction
from comet_testbench.photometry import ApertureResult

from conftest import SHAPE, StubEphemeris, coma_image, geometry_at, write_fits

GENERIC = InstrumentAdapter()


# --- designations -----------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("46P/Wirtanen", "46P"), ("COMET46P", "46P"), ("COMET41P/T-G-K", "41P"),
    ("C/2020 F3 (NEOWISE)", "C/2020 F3"), ("Comet8P/Tuttle", "8P"), ("2I/Borisov", "2I"),
    ("73P-B", "73P-B"), ("PKS_0446p11", None), ("NGC 1234", None), ("", None),
])
def test_normalize_designation(name, expected):
    assert normalize_designation(name) == expected


# --- image loading ----------------------------------------------------------

def test_frames_time_and_extensions(tmp_path):
    p = write_fits(tmp_path / "img.fits", coma_image(200, 200), n_ext=2)
    frames = load_frames(p)
    assert len(frames) == 2
    f = frames[0]
    # Mid-exposure, converted from TT (TT - UTC = 69.184 s in 2020).
    assert abs((f.mid_time.utc - Time("2020-01-01T00:01:00", scale="utc")).sec + 69.184) < 0.01
    assert f.filter == "V" and f.exposure == 100.0
    assert abs(f.pixel_scale_arcsec - 1.0) < 1e-6


# --- comet search -----------------------------------------------------------

def test_find_comets_in_and_out_of_field(tmp_path, stub_sb_ident):
    p = write_fits(tmp_path / "img.fits", coma_image(120.3, 250.7))
    frame = load_frames(p)[0]
    eph = StubEphemeris({"99P": (120.3, 250.7), "12P": (900, 900)})
    found = find_comets(frame, candidates=["12P"], adapter=GENERIC, ephemeris=eph,
                        sb_ident=stub_sb_ident)
    assert [c.designation for c in found] == ["99P"]
    assert found[0].found_by == "header"
    assert abs(found[0].x - 120.3) < 1e-6 and abs(found[0].y - 250.7) < 1e-6
    assert eph.calls[0][1] == "500@399"


def test_find_comets_uses_sb_ident_and_survives_failure(tmp_path):
    p = write_fits(tmp_path / "img.fits", coma_image(200, 200), obj="SOME FIELD")
    frame = load_frames(p)[0]
    eph = StubEphemeris({"C/2020 F3": (50, 60)})
    found = find_comets(frame, adapter=GENERIC, ephemeris=eph,
                        sb_ident=lambda *a, **k: ["C/2020 F3"])
    assert [(c.designation, c.found_by) for c in found] == [("C/2020 F3", "sb_ident")]

    def broken(*a, **k):
        raise ConnectionError("offline")
    assert find_comets(frame, adapter=GENERIC, ephemeris=eph, sb_ident=broken) == []


def test_find_comets_respects_exposure_map(tmp_path, stub_sb_ident):
    p = write_fits(tmp_path / "img.fits", coma_image(200, 200))
    frame = load_frames(p)[0]
    frame.exposure_map = np.full(SHAPE, 100.0)
    frame.exposure_map[:, :100] = 0  # unexposed strip
    eph = StubEphemeris({"99P": (50, 200)})
    assert find_comets(frame, adapter=GENERIC, ephemeris=eph, sb_ident=stub_sb_ident) == []
    eph.positions["99P"] = (150, 200)
    assert len(find_comets(frame, adapter=GENERIC, ephemeris=eph, sb_ident=stub_sb_ident)) == 1


# --- photometry -------------------------------------------------------------

def test_fixed_apertures_match_one_over_rho(tmp_path):
    amp, bg = 20.0, 1.5
    p = write_fits(tmp_path / "img.fits", coma_image(200, 200, "rho", amp=amp, bg=bg, noise=False))
    frame = load_frames(p)[0]
    radii = [10, 20, 40]
    res = fixed_aperture_photometry(frame, 200, 200, radii, "pix", adapter=GENERIC,
                                    background={"method": "fixed", "value": bg})
    for r, a in zip(radii, res):
        assert a.rate == pytest.approx(2 * np.pi * amp * r, rel=0.02)
        assert a.coverage == pytest.approx(1.0)


def test_background_estimate(tmp_path):
    p = write_fits(tmp_path / "img.fits", coma_image(200, 200, amp=50, sigma=5, bg=3.0))
    frame = load_frames(p)[0]
    rate = GENERIC.count_rate_image(frame)
    b = estimate_background(rate, 200, 200, "annulus", 80, 120)
    assert b.level == pytest.approx(3.0, abs=0.02)
    g = estimate_background(rate, 200, 200, "global", mask_radius_pix=60)
    assert g.level == pytest.approx(3.0, abs=0.02)


def test_km_radii_use_geometry(tmp_path):
    p = write_fits(tmp_path / "img.fits", coma_image(200, 200, noise=False))
    frame = load_frames(p)[0]
    geom = geometry_at(200, 200, delta=0.5)
    km = 10000.0
    res = fixed_aperture_photometry(frame, 200, 200, [km], "km", geom, GENERIC,
                                    {"method": "fixed", "value": 2.0})
    assert res[0].radius_km == pytest.approx(km)
    assert res[0].radius_arcsec == pytest.approx(km / geom.km_per_arcsec)


def test_iterative_converges_and_recenters(tmp_path):
    amp, sigma, bg = 40.0, 5.0, 2.0
    p = write_fits(tmp_path / "img.fits", coma_image(203.4, 196.2, amp=amp, sigma=sigma, bg=bg))
    frame = load_frames(p)[0]
    gc = iterative_aperture_photometry(frame, 200, 200, 2, 2, 60, "pix", adapter=GENERIC,
                                       background={"method": "annulus", "r_in": 90, "r_out": 130},
                                       tol=0.005, n_converge=3, recenter_box=12, max_shift=10)
    assert gc.converged
    assert abs(gc.best.x - 203.4) < 0.2 and abs(gc.best.y - 196.2) < 0.2
    total = 2 * np.pi * sigma**2 * amp
    assert gc.best.rate == pytest.approx(total, rel=0.02)
    assert 10 <= gc.best.radius_pix <= 30


def test_iterative_flags_nonconvergence(tmp_path):
    p = write_fits(tmp_path / "img.fits", coma_image(200, 200, "rho", amp=20, bg=1, noise=False))
    frame = load_frames(p)[0]
    gc = iterative_aperture_photometry(frame, 200, 200, 5, 5, 60, "pix", adapter=GENERIC,
                                       background={"method": "fixed", "value": 1.0})
    assert not gc.converged and gc.stop_reason == "reached r_max"


def test_rejects_large_recenter_shift(tmp_path):
    p = write_fits(tmp_path / "img.fits", coma_image(230, 200, amp=40, sigma=4))
    frame = load_frames(p)[0]
    gc = iterative_aperture_photometry(frame, 200, 200, 2, 2, 20, "pix", adapter=GENERIC,
                                       background={"method": "fixed", "value": 2.0},
                                       recenter_box=40, max_shift=5)
    assert gc.centroid_shift_pix == 0.0 and gc.best.x == 200


# --- instruments ------------------------------------------------------------

def test_uvot_coincidence_correction():
    assert uvot_coincidence_correction(0.01) == pytest.approx(0.01, rel=1e-3)
    assert uvot_coincidence_correction(30.0) > 30.0
    assert np.isnan(uvot_coincidence_correction(200.0))


def test_adapter_selection(tmp_path):
    p1 = write_fits(tmp_path / "a.fits", coma_image(200, 200), telescope="SWIFT",
                    extra={"INSTRUME": "UVOTA"})
    p2 = write_fits(tmp_path / "b.fits", coma_image(200, 200))
    assert isinstance(adapter_for(load_frames(p1)[0]), SwiftUVOT)
    assert type(adapter_for(load_frames(p2)[0])) is InstrumentAdapter
    assert SwiftUVOT().observer(load_frames(p1)[0]) == "500@-128485"


# --- hand-off to the OH/H2O package -----------------------------------------

def _two_filter_run(tmp_path, stub_sb_ident):
    w1 = write_fits(tmp_path / "w1.fits", coma_image(200, 200, amp=40, sigma=5, seed=2), filt="UVW1")
    v = write_fits(tmp_path / "v.fits", coma_image(200, 200, amp=60, sigma=5, seed=3), filt="V")
    cfg = BenchConfig(fixed_radii=[10, 20], fixed_unit="arcsec", iter_start=2, iter_step=2,
                      iter_max=60, background={"method": "annulus", "r_in": 90, "r_out": 130,
                                               "unit": "arcsec"})
    eph = StubEphemeris({"99P": (200, 200)})
    return run([w1, v], cfg, ephemeris=eph, sb_ident=stub_sb_ident)


def test_pipeline_end_to_end(tmp_path, stub_sb_ident):
    res = _two_filter_run(tmp_path, stub_sb_ident)
    assert len(res) == 2 and all(m.growth.converged for m in res)
    paths = write_results(res, tmp_path / "out")
    for name in ("comets", "fixed_apertures", "growth_curves"):
        assert (tmp_path / "out" / f"{name}.csv").exists()
    assert [p.name for p in paths] == ["99P_photometry.ecsv"]
    assert not (tmp_path / "out" / "oh_fluxes.csv").exists()


def test_handoff_file_contents(tmp_path, stub_sb_ident):
    import astropy.units as u

    res = _two_filter_run(tmp_path, stub_sb_ident)
    (path,) = write_results(res, tmp_path / "out")
    t = read_handoff(path)
    assert t.meta["schema"] == HANDOFF_SCHEMA and t.meta["designation"] == "99P"
    assert set(t["filter"]) == {"UVW1", "V"}
    fixed = t[t["aperture_kind"] == "fixed"]
    assert len(fixed) == 4 and sorted(set(fixed["radius_arcsec"].round(6))) == [10.0, 20.0]
    # Exactly one converged growth-curve step per image.
    it = t[t["aperture_kind"] == "iterative"]
    assert it["is_converged_radius"].sum() == 2
    # Units and geometry travel with the photometry.
    assert t["radius_km"].unit == u.km and t["r_h"].unit == u.au and t["rate"].unit == u.ct / u.s
    assert np.allclose(t["r_h"], 1.2) and np.allclose(t["delta"], 0.5)
    # Matches the in-memory measurement.
    w1 = next(m for m in res if m.filter == "UVW1")
    row = fixed[(fixed["filter"] == "UVW1") & (fixed["step"] == 0)][0]
    assert row["rate"] == pytest.approx(w1.fixed[0].rate)
    assert row["radius_km"] == pytest.approx(w1.fixed[0].radius_km)


def test_handoff_readable_without_astropy(tmp_path, stub_sb_ident):
    import csv

    res = _two_filter_run(tmp_path, stub_sb_ident)
    (path,) = write_results(res, tmp_path / "out")
    lines = [l for l in path.read_text().splitlines() if not l.startswith("#")]
    rows = list(csv.DictReader(lines))
    assert rows and {"filter", "radius_km", "rate", "r_h", "delta", "r_dot"} <= set(rows[0])


def test_read_handoff_rejects_other_files(tmp_path):
    from astropy.table import Table

    p = tmp_path / "x.ecsv"
    Table({"a": [1]}).write(p, format="ascii.ecsv")
    with pytest.raises(ValueError):
        read_handoff(p)


def test_config_from_json(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"candidates": ["46P"], "fixed_radii": [1e4], "use_sb_ident": False}))
    cfg = BenchConfig.from_json(p)
    assert cfg.candidates == ["46P"] and cfg.use_sb_ident is False
