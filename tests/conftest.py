"""Synthetic comet images so the bench can be tested offline."""

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from comet_testbench.ephemeris import ObservingGeometry

RA0, DEC0 = 150.0, 20.0
SCALE = 1.0 / 3600  # 1 arcsec pixels
SHAPE = (400, 400)


def make_wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [SHAPE[1] / 2 + 0.5, SHAPE[0] / 2 + 0.5]
    w.wcs.cdelt = [-SCALE, SCALE]
    return w


def coma_image(x0, y0, kind="gaussian", amp=50.0, sigma=6.0, bg=2.0, exposure=100.0,
               seed=1, noise=True):
    """Counts image of a coma at pixel (x0, y0) on a flat sky.

    'gaussian': total rate = 2*pi*sigma^2*amp. 'rho': surface rate = amp/rho,
    so the rate inside radius r is 2*pi*amp*r.
    """
    yy, xx = np.indices(SHAPE, dtype=float)
    rho = np.hypot(xx - x0, yy - y0)
    if kind == "gaussian":
        rate = amp * np.exp(-0.5 * (rho / sigma) ** 2)
    else:
        # Sub-sample the central pixels where 1/rho varies fast.
        rate = amp / np.maximum(rho, 0.5)
        near = rho < 3
        for iy, ix in zip(*np.nonzero(near)):
            sub = (np.arange(10) + 0.5) / 10 - 0.5
            sx, sy = np.meshgrid(ix + sub, iy + sub)
            rate[iy, ix] = np.mean(amp / np.hypot(sx - x0, sy - y0))
    rate = rate + bg
    counts = rate * exposure
    if noise:
        counts = np.random.default_rng(seed).poisson(counts).astype(float)
    return counts


def write_fits(path, counts, exposure=100.0, obj="COMET 99P", filt="V", extra=None,
               n_ext=1, telescope="TESTSCOPE"):
    w = make_wcs()
    hdus = [fits.PrimaryHDU()]
    for i in range(n_ext):
        h = w.to_header()
        h["DATE-OBS"] = "2020-01-01T00:00:00"
        h["DATE-END"] = "2020-01-01T00:02:00"
        h["TIMESYS"] = "TT"
        h["EXPOSURE"] = exposure
        h["FILTER"] = filt
        h["OBJECT"] = obj
        h["TELESCOP"] = telescope
        h["INSTRUME"] = "TESTCAM"
        for k, v in (extra or {}).items():
            h[k] = v
        hdus.append(fits.ImageHDU(counts, header=h))
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def geometry_at(x, y, target="99P", delta=0.5, r_h=1.2, r_dot=-5.0):
    ra, dec = make_wcs().pixel_to_world_values(x, y)
    return ObservingGeometry(target, 2458849.5, float(ra), float(dec), r_h, r_dot, delta, 0.0,
                             30.0, 10.0, 0.0)


class StubEphemeris:
    """Ephemeris provider returning fixed positions per designation."""

    def __init__(self, positions):
        self.positions = positions  # designation -> (x, y)
        self.calls = []

    def __call__(self, des, time, observer):
        self.calls.append((des, observer))
        if des not in self.positions:
            raise KeyError(des)
        return geometry_at(*self.positions[des], target=des)


@pytest.fixture
def stub_sb_ident():
    def f(*args, **kwargs):
        return []
    return f
