"""Pick public ZTF frames with known comets, plus comet-free control frames.

Comet frames: Horizons positions (observer Palomar, I41) on a grid of dates,
IRSA IBE search around each, then a Horizons check at each frame's own time
that the comet sits on the quadrant. Control frames: the same field, CCD
quadrant and filter on another night, when the comet is far away. A control
therefore images the same sky as its comet frame, which makes it a direct
null test for apertures placed at the comet's pixel position.

Usage: python select_frames.py OUT.csv
"""

import csv
import sys

import numpy as np
from astropy.time import Time

from comet_testbench.ephemeris import query_horizons
from comet_testbench.ztf_archive import search_frames

# designation, start, stop. A spread of brightness (V ~ 6 to ~18), distance
# (0.08 au 46P to 6 au 29P) and rate of motion.
COMETS = [
    ("46P", "2018-11-25", "2019-01-10"),
    ("21P", "2018-08-20", "2018-10-01"),
    ("38P", "2018-10-15", "2018-12-15"),
    ("64P", "2018-10-15", "2018-12-15"),
    ("260P", "2019-08-01", "2019-10-15"),
    ("29P", "2019-10-01", "2020-02-01"),
    ("C/2019 Y4", "2020-02-15", "2020-04-15"),
    ("7P", "2021-05-01", "2021-07-15"),
    ("8P", "2021-07-01", "2021-09-15"),
    ("4P", "2021-08-01", "2021-10-31"),
    ("57P", "2021-09-01", "2021-11-15"),
    ("67P", "2021-10-01", "2021-12-31"),
    ("C/2021 A1", "2021-09-01", "2021-11-30"),
    ("C/2019 L3", "2021-11-15", "2022-02-28"),
    ("C/2017 K2", "2022-05-01", "2022-08-31"),
    ("C/2022 E3", "2022-11-01", "2023-02-15"),
    ("12P", "2023-07-15", "2023-10-31"),
    ("2P", "2023-08-20", "2023-10-10"),
    ("C/2020 F3", "2020-07-15", "2020-08-15"),
    ("141P", "2020-11-01", "2021-01-31"),
    ("19P", "2021-12-01", "2022-02-28"),
    ("22P", "2022-08-01", "2022-10-31"),
    ("C/2022 A2", "2022-12-01", "2023-03-31"),
    ("62P", "2023-11-01", "2024-01-31"),
    ("C/2023 A3", "2024-02-01", "2024-04-30"),
    ("29P", "2022-10-01", "2023-01-31"),
]
PER_COMET = 4
N_DATES = 16


def pick(frames, n):
    """Up to n frames on different nights, alternating filters."""
    frames = sorted(frames, key=lambda f: f.obsjd)
    nights, out = set(), []
    for want in ("zg", "zr", "zi") * n:
        for f in frames:
            night = int(f.obsjd - 0.2)
            if f.filtercode == want and night not in nights:
                nights.add(night)
                out.append(f)
                break
        if len(out) >= n:
            break
    for f in frames:  # fill from any filter
        if len(out) >= n:
            break
        if int(f.obsjd - 0.2) not in nights:
            nights.add(int(f.obsjd - 0.2))
            out.append(f)
    return out


def select_comet(des, t0, t1):
    rows = []
    if True:
        jds = np.linspace(Time(t0).jd, Time(t1).jd, N_DATES) + 0.25  # ~midnight at Palomar
        geoms = query_horizons(des, Time(jds, format="jd"), "I41@399")
        # Fast movers leave a fixed search position within the night, so cover
        # the night (+-0.4 d) with steps short enough to move < 0.3 deg each.
        queries = []
        for g in geoms:
            rate_deg_day = np.hypot(g.ra_rate, g.dec_rate) * 24 / 3600
            n = int(np.clip(np.ceil(0.8 * rate_deg_day / 0.3), 1, 16))
            if n == 1:
                queries.append((g, 0.4))
            else:
                ts = g.time_jd_utc + np.linspace(-0.4, 0.4, n + 1)[:-1] + 0.4 / n
                queries += [(gg, 0.4 / n) for gg in query_horizons(des, Time(ts, format="jd"), "I41@399")]
        hits = {}
        for g, hw in queries:
            try:
                for f in search_frames(g.ra, g.dec, g.time_jd_utc - hw, g.time_jd_utc + hw):
                    hits[f.stem] = f
            except Exception as exc:
                print(f"  {des}: IBE search failed: {exc}")
        cands = list(hits.values())
        if not cands:
            print(f"{des}: no frames", flush=True)
            return rows
        # Confirm on the quadrant at each frame's own time.
        good = []
        for k in range(0, len(cands), 50):
            chunk = cands[k:k + 50]
            gs = query_horizons(des, Time([f.obsjd for f in chunk], format="jd"), "I41@399")
            good += [(f, g) for f, g in zip(chunk, gs) if f.contains_pixel(g.ra, g.dec, margin_pix=100)]
        chosen = pick([f for f, _ in good], PER_COMET)
        gmap = {f.stem: g for f, g in good}
        print(f"{des}: {len(cands)} frames near track, {len(good)} on quadrant, {len(chosen)} chosen", flush=True)
        for f in chosen:
            g = gmap[f.stem]
            rows.append(dict(kind="comet", target=des, stem=f.stem, url=f.url, obsjd=f.obsjd,
                             filtercode=f.filtercode, field=f.field, rcid=f.rcid, seeing=f.seeing,
                             maglimit=f.maglimit, comet_ra=g.ra, comet_dec=g.dec, pair=""))
            # Control: same field/quadrant/filter, 20-200 days away, comet off the quadrant.
            ctrl = None
            for lo, hi in ((20, 200), (-200, -20)):
                try:
                    opts = [c for c in search_frames(f.ra, f.dec, f.obsjd + lo, f.obsjd + hi)
                            if (c.field, c.rcid, c.filtercode) == (f.field, f.rcid, f.filtercode)]
                except Exception as exc:
                    print(f"  control search failed: {exc}")
                    opts = []
                opts.sort(key=lambda c: abs(c.obsjd - f.obsjd))
                for c in opts[:5]:
                    gc = query_horizons(des, Time(c.obsjd, format="jd"), "I41@399")[0]
                    if not c.contains_pixel(gc.ra, gc.dec, margin_pix=-1000):  # >= 1000 px off
                        ctrl = c
                        break
                if ctrl:
                    break
            if ctrl is None:
                print(f"  no control for {f.stem}")
                continue
            rows.append(dict(kind="control", target=des, stem=ctrl.stem, url=ctrl.url, obsjd=ctrl.obsjd,
                             filtercode=ctrl.filtercode, field=ctrl.field, rcid=ctrl.rcid,
                             seeing=ctrl.seeing, maglimit=ctrl.maglimit, comet_ra=g.ra,
                             comet_dec=g.dec, pair=f.stem))
    return rows


def main(out_path):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    fields = ["kind", "target", "stem", "url", "obsjd", "filtercode", "field", "rcid", "seeing",
              "maglimit", "comet_ra", "comet_dec", "pair"]
    rows = []
    with open(out_path, "w", newline="") as fh, ThreadPoolExecutor(6) as pool:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        futs = {pool.submit(select_comet, *c): c[0] for c in COMETS}
        for fut in as_completed(futs):
            try:
                new = fut.result()
            except Exception as exc:
                print(f"{futs[fut]}: failed: {exc}", flush=True)
                continue
            w.writerows(new)
            fh.flush()
            rows += new
    print(f"{sum(r['kind'] == 'comet' for r in rows)} comet frames, "
          f"{sum(r['kind'] == 'control' for r in rows)} controls -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
