"""ZTF adapter and ZTF-style headers (offline)."""

import numpy as np
from astropy.io import fits

from comet_testbench import ZTF, adapter_for, load_frames
from comet_testbench.ztf_archive import ZTFFrame

from conftest import coma_image, make_wcs


def _ztf_file(path, exposure=30.0, magzp=26.2, saturate=1e6):
    counts = coma_image(200, 200, exposure=exposure)
    h = make_wcs().to_header()
    # ZTF has no DATE-OBS: only OBSJD/OBSMJD (shutter open) and SHUTOPEN/SHUTCLSD.
    h["OBSJD"] = 2458598.6874884
    h["OBSMJD"] = 58598.1874884
    h["SHUTOPEN"] = "2019-04-25T04:30:00.213"
    h["SHUTCLSD"] = "2019-04-25T04:30:30.175"
    h["EXPTIME"] = exposure
    h["EXPOSURE"] = exposure
    h["FILTER"] = "ZTF_r"
    h["ORIGIN"] = "Zwicky Transient Facility"
    h["INSTRUME"] = "ZTF/MOSAIC"
    h["TELESCOP"] = "Palomar 48-inch"
    h["MAGZP"] = magzp
    h["SATURATE"] = saturate
    fits.PrimaryHDU(counts, header=h).writeto(path)
    return path


def test_ztf_header_time_and_adapter(tmp_path):
    f = load_frames(_ztf_file(tmp_path / "z.fits"))[0]
    assert f.mid_time.utc.isot.startswith("2019-04-25T04:30:15.1")
    a = adapter_for(f)
    assert isinstance(a, ZTF)
    assert a.observer(f) == "I41@399" and a.mpc_code(f) == "I41"
    # MAGZP is for DN per exposure; per DN/s it gains 2.5 log10(EXPOSURE).
    assert np.isclose(a.zero_point(f), 26.2 + 2.5 * np.log10(30.0))


def test_ztf_saturated_pixels_masked(tmp_path):
    f = load_frames(_ztf_file(tmp_path / "z.fits", saturate=1000.0))[0]
    rate = ZTF().count_rate_image(f)
    assert np.isnan(rate[200, 200]) and np.isfinite(rate[0, 0])


def test_ztf_frame_url_and_footprint():
    fr = ZTFFrame("20190425187488", 571, 12, 1, "zr", "o", 2458598.69, 30.0, 1.8, 21.1,
                  ((150.06, 20.76), (149.14, 20.75), (149.16, 19.88), (150.08, 19.90)),
                  149.61, 20.32)
    assert fr.url.endswith("/sci/2019/0425/187488/ztf_20190425187488_000571_zr_c12_o_q1_sciimg.fits")
    assert fr.rcid == 44
    assert fr.contains(149.6, 20.3) and not fr.contains(151.0, 20.3)
