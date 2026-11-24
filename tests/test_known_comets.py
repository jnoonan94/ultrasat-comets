"""Known-comet ephemeris cache and its use in find_comets (offline)."""

import numpy as np

from comet_testbench import InstrumentAdapter, find_comets, load_frames
from comet_testbench.ephemeris import SbIdentRecord
from comet_testbench.known_comets import EphemerisCache, _perihelion_near

from conftest import StubEphemeris, coma_image, make_wcs, write_fits

GENERIC = InstrumentAdapter()


def _cache_through(x, y, des="67P", jd_mid=2458849.5 + 0.0, tmag=12.0):
    """A cache whose track passes pixel (x, y) at the test frame's mid-time."""
    ra, dec = make_wcs().pixel_to_world_values(x, y)
    jd = jd_mid + np.arange(-3, 4)
    return EphemerisCache("500@399", {des: {"jd": list(jd), "ra": [float(ra) + 0.01 * k for k in range(-3, 4)],
                                            "dec": [float(dec)] * 7, "tmag": [tmag] * 7}})


def test_perihelion_window_steps_by_period():
    # Stored tp from an earlier apparition (67P: 2015 tp, 6.44 yr period). Stepping
    # by the mean period drifts ~80 d per orbit, hence windows of months to years.
    assert _perihelion_near(2457247.59, 6.44, 2459500, 2459530, 365)
    assert not _perihelion_near(2457247.59, 6.44, 2458400, 2458500, 365)
    assert _perihelion_near(2459000.0, float("nan"), 2459010, 2459020, 30)


def test_cache_interpolates_and_selects(tmp_path):
    frame = load_frames(write_fits(tmp_path / "a.fits", coma_image(200, 200)))[0]
    cache = _cache_through(100, 150, jd_mid=frame.mid_time.utc.jd)
    (des, ra, dec, tmag), = cache.positions(frame.mid_time)
    x, y = frame.wcs.world_to_pixel_values(ra, dec)
    assert des == "67P" and abs(float(x) - 100) < 0.5 and abs(float(y) - 150) < 0.5
    assert cache.in_frame(frame) == ["67P"]
    assert cache.in_frame(frame, vmag_lim=10.0) == []
    cache.save(tmp_path / "c.json")
    assert EphemerisCache.load(tmp_path / "c.json").in_frame(frame) == ["67P"]


def test_find_comets_uses_cache_when_sb_ident_misses(tmp_path, stub_sb_ident):
    frame = load_frames(write_fits(tmp_path / "a.fits", coma_image(200, 200), obj="FIELD"))[0]
    cache = _cache_through(100, 150, jd_mid=frame.mid_time.utc.jd)
    eph = StubEphemeris({"67P": (100, 150)})
    found = find_comets(frame, adapter=GENERIC, ephemeris=eph, sb_ident=stub_sb_ident, known=cache)
    assert [(c.designation, c.found_by) for c in found] == [("67P", "ephemeris cache")]
    both = find_comets(frame, adapter=GENERIC, ephemeris=eph, known=cache,
                       sb_ident=lambda *a, **k: ["67P"])
    assert both[0].found_by == "sb_ident+ephemeris cache"


def test_sb_ident_records_far_off_frame_skip_horizons(tmp_path):
    frame = load_frames(write_fits(tmp_path / "a.fits", coma_image(200, 200), obj="FIELD"))[0]
    ra, dec = make_wcs().pixel_to_world_values(200, 200)
    near = SbIdentRecord("99P", "99P/Near", float(ra), float(dec), 15.0)
    far = SbIdentRecord("98P", "98P/Far", float(ra) + 3.0, float(dec), 15.0)
    eph = StubEphemeris({"99P": (200, 200), "98P": (200, 200)})
    found = find_comets(frame, adapter=GENERIC, ephemeris=eph, sb_ident=lambda *a, **k: [near, far])
    assert [c.designation for c in found] == ["99P"]
    assert [d for d, _ in eph.calls] == ["99P"]
