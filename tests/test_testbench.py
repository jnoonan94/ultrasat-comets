import json
import sys
import types

import numpy as np
import pytest
from astropy.time import Time

from comet_testbench import (BenchConfig, InstrumentAdapter, OHMeasurement, estimate_background,
                             find_comets, fit_production_rate, fixed_aperture_photometry,
                             iterative_aperture_photometry, load_frames, load_model,
                             normalize_designation, run, subtract_continuum, write_results)
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


# --- OH interface -----------------------------------------------------------

class LinearToyModel:
    """Stand-in for the real OH package: flux grows linearly with Q and radius."""
    name = "toy"

    def aperture_flux(self, production_rate, radius_km, geometry):
        return 1e-41 * production_rate * np.asarray(radius_km) / geometry.r_h**2


def _ap(r_km, rate, err):
    return ApertureResult(0, 0, 0, 0, r_km, 0, 1, rate, err, 0)


def test_continuum_subtraction_and_fit():
    geom = geometry_at(200, 200, r_h=1.2)
    model = LinearToyModel()
    q_true = 3e28
    radii = np.array([5000.0, 10000.0, 20000.0])
    conv = 1e-12
    oh_flux = model.aperture_flux(q_true, radii, geom)
    beta = 0.1
    v = [_ap(r, 100.0, 1.0) for r in radii]
    w1 = [_ap(r, f / conv + beta * 100.0, 0.01 * f / conv) for r, f in zip(radii, oh_flux)]
    meas = subtract_continuum(w1, v, beta, conv, geom)
    assert np.allclose(meas.oh_flux, oh_flux, rtol=1e-9)
    fit = fit_production_rate(model, meas)
    assert fit.q == pytest.approx(q_true, rel=1e-4)
    assert np.allclose(fit.q_per_aperture, q_true, rtol=1e-4)
    assert fit.q_err > 0 and fit.model_name == "toy"


def test_continuum_requires_matching_radii():
    geom = geometry_at(200, 200)
    with pytest.raises(ValueError):
        subtract_continuum([_ap(1000, 1, 1)], [_ap(2000, 1, 1)], 0.1, 1.0, geom)


def test_load_model_from_spec(monkeypatch):
    mod = types.ModuleType("fake_oh_pkg")
    mod.Model = LinearToyModel
    mod.NotAModel = object
    monkeypatch.setitem(sys.modules, "fake_oh_pkg", mod)
    assert isinstance(load_model("fake_oh_pkg:Model"), LinearToyModel)
    with pytest.raises(TypeError):
        load_model("fake_oh_pkg:NotAModel")


# --- end to end -------------------------------------------------------------

def test_pipeline_end_to_end(tmp_path, stub_sb_ident):
    from comet_testbench.pipeline import fit_all, oh_measurements

    w1 = write_fits(tmp_path / "w1.fits", coma_image(200, 200, amp=40, sigma=5, seed=2), filt="UVW1")
    v = write_fits(tmp_path / "v.fits", coma_image(200, 200, amp=60, sigma=5, seed=3), filt="V")
    cfg = BenchConfig(fixed_radii=[10, 20], fixed_unit="arcsec", iter_start=2, iter_step=2,
                      iter_max=60, background={"method": "annulus", "r_in": 90, "r_out": 130,
                                               "unit": "arcsec"})
    eph = StubEphemeris({"99P": (200, 200)})
    res = run([w1, v], cfg, ephemeris=eph, sb_ident=stub_sb_ident)
    assert len(res) == 2 and all(m.growth.converged for m in res)
    oh = oh_measurements(res, beta=0.1, oh_conversion=1e-12)
    assert len(oh) == 1
    fits_ = fit_all(LinearToyModel(), oh)
    write_results(res, tmp_path / "out", oh, fits_)
    for name in ("comets", "fixed_apertures", "growth_curves", "oh_fluxes", "production_rates"):
        assert (tmp_path / "out" / f"{name}.csv").exists()


def test_config_from_json(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"candidates": ["46P"], "fixed_radii": [1e4], "use_sb_ident": False}))
    cfg = BenchConfig.from_json(p)
    assert cfg.candidates == ["46P"] and cfg.use_sb_ident is False
