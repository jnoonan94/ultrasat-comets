import numpy as np
import pytest
from astropy.time import Time

from comet_testbench import SbIdentRecord, UltraSat, find_comets, search_pointings, virtual_frame
from comet_testbench.ephemeris import _center_params, _parse_sb_ident_row
from comet_testbench.instruments import ULTRASAT_SIM_SITE, adapter_for
from comet_testbench.pointing import PointingConstraints, parse_pointing

T = Time("2025-03-01T00:00:00")


def test_virtual_frame_geometry():
    f = virtual_frame(120.0, 30.0, T, fov_deg=(20, 10), pixel_scale_arcsec=120)
    assert f.data.shape == (300, 600)
    assert f.center() == pytest.approx((120.0, 30.0), abs=1e-6)
    assert abs(f.pixel_scale_arcsec - 120) < 1e-6
    # Long axis along RA: a point 9 deg east on the sky is inside, 6 deg north is not.
    x, y = f.wcs.world_to_pixel_values(120 + 9 / np.cos(np.deg2rad(30)), 30.0)
    assert f.contains_pixel(float(x), float(y))
    x, y = f.wcs.world_to_pixel_values(120.0, 36.0)
    assert not f.contains_pixel(float(x), float(y))
    assert isinstance(adapter_for(f), UltraSat)


def test_virtual_frame_roll():
    f = virtual_frame(0.0, 0.0, T, fov_deg=(20, 10), roll_deg=90, pixel_scale_arcsec=120)
    x, y = f.wcs.world_to_pixel_values(0.0, 9.0)  # long axis now along Dec
    assert f.contains_pixel(float(x), float(y))


def test_parse_pointing_accepts_simulator_strings():
    assert parse_pointing("266.4 -29.0") == (266.4, -29.0)
    assert parse_pointing((1, 2)) == (1.0, 2.0)


def test_ultrasat_observer_matches_simulator():
    a = UltraSat()
    f = virtual_frame(0, 0, T)
    assert a.sb_ident_site(f) == ULTRASAT_SIM_SITE
    lat, lon, alt = ULTRASAT_SIM_SITE
    assert a.observer(f) == f"site:{lon},{lat},{alt}"
    p = _center_params(a.observer(f))
    assert p["CENTER"] == "'coord@399'" and p["SITE_COORD"] == f"'{lon},{lat},{alt}'"
    assert UltraSat(horizons_id="500@-999").observer(f) == "500@-999"


def test_parse_sb_ident_row():
    row = ["46P/Wirtanen", "02:26:01.35", "-23 18'12.9\"", "1.E1", "2.E1", "3.E1", "12.3N",
           "-6.8E+00", "1.3E+00"]
    r = _parse_sb_ident_row(row)
    assert r.designation == "46P" and r.vmag == pytest.approx(12.3)
    assert r.ra == pytest.approx(36.505625) and r.dec == pytest.approx(-23.30358, abs=1e-5)
    assert _parse_sb_ident_row(["(2005 AB) asteroid", "00:00:00", "+00 00'00\""]) is None


def _sb_stub(records):
    calls = []

    def f(ra0, dec0, hw_ra, hw_dec, time, **kw):
        calls.append(dict(ra0=ra0, dec0=dec0, hw_ra=hw_ra, hw_dec=hw_dec, **kw))
        return records
    f.calls = calls
    return f


def test_search_pointings_without_horizons():
    inside = SbIdentRecord("99P", "99P/Test", 121.0, 31.0, 14.0)
    outside = SbIdentRecord("98P", "98P/Far", 140.0, 31.0, 13.0)
    sb = _sb_stub([inside, outside])

    def no_horizons(*a):
        raise AssertionError("Horizons should not be called")

    rows = search_pointings(["120.0 30.0"], [T], sb_ident=sb, ephemeris=no_horizons,
                            constraints=None)
    assert [r["designation"] for r in rows] == ["99P"]
    assert rows[0]["vmag"] == 14.0 and rows[0]["found_by"] == "sb_ident"
    # Simulator observer and magnitude settings reach sb_ident.
    assert sb.calls[0]["site"] == ULTRASAT_SIM_SITE
    assert sb.calls[0]["no_magnitude"] == "periodic"


def test_search_pointings_simulator_compatible_mode():
    sb = _sb_stub([])
    search_pointings([(10, 10)], [T], sb_ident=sb, vmag_lim=18, no_magnitude="none",
                     constraints=None, ephemeris=lambda *a: None)
    assert sb.calls[0]["vmag_lim"] == 18 and sb.calls[0]["no_magnitude"] == "none"
    # Box covers the 20 x 10 deg field plus margin.
    assert sb.calls[0]["hw_ra"] >= 10 and sb.calls[0]["hw_dec"] >= 5


def test_constraint_flags():
    c = PointingConstraints()
    sun = c.evaluate(0, 0, T)
    assert set(sun) == {"gal_b_deg", "sun_sep_deg", "moon_sep_deg", "passes_constraints"}
    # Galactic centre is within 30 deg of the plane.
    assert abs(c.evaluate(266.4, -28.9, T)["gal_b_deg"]) < 2


def test_sb_ident_no_magnitude_policy(monkeypatch):
    from comet_testbench import ephemeris

    rows = [["46P/Wirtanen", "02:26:01", "-23 18'12\"", "", "", "", "", "", ""],
            ["C/1997 H4 (SOHO)", "02:26:01", "-23 18'12\"", "", "", "", "", "", ""],
            ["C/2020 F3 (NEOWISE)", "02:26:01", "-23 18'12\"", "", "", "", "12.0", "", ""],
            ["C/2009 R1 (McNaught)", "02:26:01", "-23 18'12\"", "", "", "", "27.8", "", ""]]
    sent = {}

    def fake_get(url, params, session=None):
        sent.update(params)
        return {"data_second_pass": rows}
    monkeypatch.setattr(ephemeris, "_get", fake_get)
    q = lambda nm: [r.designation for r in ephemeris.sb_ident_query(36.5, -23.3, 1, 1, T,
                                                                     vmag_lim=25, no_magnitude=nm)]
    assert q("periodic") == ["46P", "C/2020 F3"] and sent["mag-required"] == "false"
    assert q("all") == ["46P", "C/1997 H4", "C/2020 F3"]
    assert q("none") == ["C/2020 F3"] and sent["mag-required"] == "true"
