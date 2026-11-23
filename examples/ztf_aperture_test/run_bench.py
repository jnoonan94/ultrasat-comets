"""Run the test bench's field search and aperture extraction on ZTF frames.

Input: the CSV from select_frames.py and a folder of downloaded sciimg files.
For every frame:

1. Field search exactly as for UltraSat: sb_ident over the footprint (observer
   Palomar), no candidate list, then Horizons placement. For comet frames the
   target is also checked by name, so an sb_ident miss and an ephemeris miss
   can be told apart.
2. For each comet on the frame: the default BenchConfig photometry (iterative
   growth curve with recentring + fixed km apertures, global background), and
   the benchmarks' centroid offset and SNR in a 5" aperture.
3. For each control frame: the same measurements at the pixel where the paired
   frame's comet sat (same field and quadrant, so the same sky without the
   comet). These are the null results.

Writes results.csv (one row per frame x comet/position), growth.csv and
stamps/*.png. Usage: python run_bench.py frames.csv RAW_DIR OUT_DIR
"""

import csv
import json
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
from astropy.time import Time

from comet_testbench import BenchConfig, ZTF, find_comets, load_frames
from comet_testbench.benchmarks import ephemeris_offset, recover
from comet_testbench.ephemeris import horizons_provider, ObservingGeometry
from comet_testbench.field_search import CometInField
from comet_testbench.photometry import fixed_aperture_photometry, iterative_aperture_photometry

log = logging.getLogger("ztf")
CFG = BenchConfig()
ADAPTER = ZTF()
EPH = horizons_provider()


def measure(frame, rate, comet, stamp_path=None):
    """Bench photometry at comet.x, comet.y plus diagnostics."""
    out = {"x_pred": comet.x, "y_pred": comet.y, "trail_pix": comet.trail_pix}
    s = frame.pixel_scale_arcsec
    try:
        g = iterative_aperture_photometry(
            frame, comet.x, comet.y, CFG.iter_start, CFG.iter_step, CFG.iter_max, CFG.iter_unit,
            comet.geometry, ADAPTER, dict(CFG.background), CFG.iter_tol, CFG.iter_n_converge,
            recenter_box=CFG.recenter_box_arcsec, max_shift=CFG.max_shift_arcsec, rate_image=rate)
        b = g.best
        fixed = fixed_aperture_photometry(frame, b.x, b.y, CFG.fixed_radii, CFG.fixed_unit,
                                          comet.geometry, ADAPTER, g.background, rate_image=rate)
        out.update(growth_stop=g.stop_reason, converged=g.converged,
                   n_steps=len(g.apertures), r_conv_arcsec=b.radius_arcsec,
                   rate_conv=b.rate, rate_err_conv=b.rate_err, mag_conv=b.mag,
                   snr_conv=b.rate / b.rate_err if b.rate_err > 0 else np.nan,
                   coverage_conv=b.coverage, recenter_shift_arcsec=g.centroid_shift_pix * s,
                   bg_level=g.background.level, bg_sigma=g.background.sigma,
                   growth=[(a.radius_arcsec, a.rate, a.rate_err) for a in g.apertures])
        reason, r_med, rate_med = median_profile_growth(rate, b.x, b.y, g.background.level, s)
        out.update(med_stop=reason, med_r_conv_arcsec=r_med, med_rate_conv=rate_med,
                   med_mag_conv=ADAPTER.zero_point(frame) - 2.5 * np.log10(rate_med) if rate_med > 0 else np.nan)
        for a, r in zip(fixed, CFG.fixed_radii):
            out[f"mag_{int(r)}km"] = a.mag
            out[f"snr_{int(r)}km"] = a.rate / a.rate_err if a.rate_err > 0 else np.nan
            out[f"r_{int(r)}km_arcsec"] = a.radius_arcsec
            out[f"cov_{int(r)}km"] = a.coverage
    except Exception as exc:
        out["error"] = f"growth: {exc}"
    try:
        e = ephemeris_offset(frame, comet, 10.0, ADAPTER, rate_image=rate)
        out.update(offset_arcsec=e.offset_arcsec, along_arcsec=e.along_track_arcsec,
                   cross_arcsec=e.cross_track_arcsec, peak_snr_10as=e.peak_snr,
                   x_obs=e.x_obs, y_obs=e.y_obs)
    except Exception as exc:
        out["error"] = out.get("error", "") + f" offset: {exc}"
    try:
        ok, snr, off, _ = recover(frame, comet.x, comet.y, 5.0 / s, ADAPTER, 5.0, rate_image=rate)
        out.update(detected_5as=ok, snr_5as=snr, recover_shift_arcsec=off * s)
    except Exception as exc:
        out["error"] = out.get("error", "") + f" recover: {exc}"
    if stamp_path is not None:
        try:
            stamp(frame, rate, comet, out, stamp_path)
        except Exception as exc:
            log.warning("stamp failed: %s", exc)
    return out


def median_profile_growth(rate, x, y, bg, scale, r_step=3.0, r_max=150.0, tol=0.01, n_converge=3):
    """Growth curve built from azimuthal *medians* in annuli (experimental).

    Each 3" annulus contributes median(surface rate - bg) x its pixel count,
    so stars inside an annulus are rejected unless they cover half of it.
    Same convergence rule as the bench. Returns (stop_reason, r_conv, rates).
    """
    n = int(np.ceil(r_max / scale)) + 2
    ny, nx = rate.shape
    ix, iy = int(round(x)), int(round(y))
    y1, y2, x1, x2 = max(iy - n, 0), min(iy + n + 1, ny), max(ix - n, 0), min(ix + n + 1, nx)
    cut = rate[y1:y2, x1:x2] - bg
    yy, xx = np.indices(cut.shape)
    rho = np.hypot(xx + x1 - x, yy + y1 - y) * scale
    edges = np.arange(0, r_max + r_step / 2, r_step)
    cum, rates = 0.0, []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (rho >= a) & (rho < b) & np.isfinite(cut)
        if sel.sum() == 0:
            rates.append(cum)
            continue
        vals = cut[sel]
        level = np.mean(vals) if sel.sum() < 30 else np.median(vals)
        area = np.pi * (b**2 - a**2) / scale**2
        cum += level * area
        rates.append(cum)
    run, conv, reason = 0, None, "reached r_max"
    for k in range(1, len(rates)):
        prev, cur = rates[k - 1], rates[k]
        if prev <= 0:
            run = 0
            continue
        frac = (cur - prev) / prev
        if frac < -tol:
            reason, conv = "growth curve turned over", k - 1
            break
        if abs(frac) < tol:
            run += 1
            if run >= n_converge:
                reason, conv = "converged", k - n_converge
                break
        else:
            run = 0
    r_conv = float(edges[1:][conv]) if conv is not None else float(r_max)
    rate_conv = rates[conv] if conv is not None else rates[-1]
    return reason, r_conv, float(rate_conv)


def stamp(frame, rate, comet, res, path, half_arcsec=120):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    h = int(half_arcsec / frame.pixel_scale_arcsec)
    x0, y0 = int(round(comet.x)), int(round(comet.y))
    ny, nx = rate.shape
    y1, y2, x1, x2 = max(y0 - h, 0), min(y0 + h, ny), max(x0 - h, 0), min(x0 + h, nx)
    cut = rate[y1:y2, x1:x2]
    bg, sig = res.get("bg_level", np.nanmedian(cut)), res.get("bg_sigma", np.nanstd(cut))
    fig, ax = plt.subplots(figsize=(3, 3), dpi=110)
    ax.imshow(cut, origin="lower", cmap="gray_r", vmin=bg - 2 * sig, vmax=bg + 15 * sig,
              extent=(x1 - 0.5, x2 - 0.5, y1 - 0.5, y2 - 0.5))
    ax.plot(comet.x, comet.y, "+", color="tab:blue", ms=12, mew=1.5)
    if "x_obs" in res:
        ax.plot(res["x_obs"], res["y_obs"], "x", color="tab:red", ms=8)
    if "r_conv_arcsec" in res and np.isfinite(res["r_conv_arcsec"]):
        r = res["r_conv_arcsec"] / frame.pixel_scale_arcsec
        ax.add_patch(Circle((res.get("x_obs", comet.x), res.get("y_obs", comet.y)), r, fill=False,
                            color="tab:orange", lw=1.2))
    ax.set_xlim(x1 - 0.5, x2 - 0.5), ax.set_ylim(y1 - 0.5, y2 - 0.5)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"{comet.designation} {frame.filter} {frame.mid_time.utc.iso[:10]}", fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main(frames_csv, raw_dir, out_dir):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_in = list(csv.DictReader(open(frames_csv)))
    done_path = out_dir / "results.jsonl"
    done = set()
    if done_path.exists():
        done = {json.loads(l)["stem"] for l in open(done_path)}
    for k, row in enumerate(rows_in):
        if row["stem"] in done:
            continue
        path = raw_dir / f"{row['stem']}_sciimg.fits"
        if not path.exists():
            log.warning("missing %s", path)
            continue
        recs = []
        try:
            frame = load_frames(path)[0]
            rate = ADAPTER.count_rate_image(frame)
            base = dict(kind=row["kind"], target=row["target"], stem=row["stem"],
                        filter=frame.filter, mid_time=frame.mid_time.utc.isot,
                        seeing=float(row["seeing"]), maglimit=float(row["maglimit"]),
                        pixel_scale=frame.pixel_scale_arcsec)
            # 1. Blind field search (the UltraSat method), then a named check.
            blind = find_comets(frame, candidates=(), adapter=ADAPTER, use_header_object=False,
                                use_sb_ident=True, ephemeris=EPH, vmag_lim=22.0)
            blind_names = sorted(c.designation for c in blind)
            # Same search with a 2 deg sb_ident margin: sb_ident's first pass
            # can misplace fast movers by more than the default 0.1 deg.
            wide = find_comets(frame, candidates=(), adapter=ADAPTER, use_header_object=False,
                               use_sb_ident=True, ephemeris=EPH, vmag_lim=22.0, sb_ident_margin_deg=2.0)
            wide_names = sorted(c.designation for c in wide)
            base["wide_list"] = ";".join(wide_names)
            named = [c for c in wide if c.designation not in blind_names]
            if row["kind"] == "comet" and row["target"] not in wide_names:
                named = find_comets(frame, candidates=[row["target"]], adapter=ADAPTER,
                                    use_header_object=False, use_sb_ident=False, ephemeris=EPH)
            for c in blind + named:
                res = measure(frame, rate, c, out_dir / "stamps" / f"{row['stem']}_{c.designation.replace('/', '').replace(' ', '')}.png")
                g = c.geometry
                recs.append({**base, "role": "target" if c.designation == row["target"] else "other",
                             "designation": c.designation, "found_blind": c.designation in blind_names,
                             "found_wide": c.designation in wide_names,
                             "sb_ident_vmag": c.vmag, "r_h": g.r_h, "delta": g.delta,
                             "rate_arcsec_hr": float(np.hypot(g.ra_rate, g.dec_rate)),
                             "blind_list": ";".join(blind_names), **res})
            if row["kind"] == "comet" and not any(r["role"] == "target" for r in recs):
                recs.append({**base, "role": "target", "designation": row["target"],
                             "found_blind": False, "blind_list": ";".join(blind_names),
                             "error": "target not placed on frame"})
            if row["kind"] == "control":
                # 2. Null aperture at the paired comet's sky position.
                x, y = frame.wcs.world_to_pixel_values(float(row["comet_ra"]), float(row["comet_dec"]))
                geom = EPH(row["target"], frame.mid_time, ADAPTER.observer(frame))
                fake_geom = ObservingGeometry(row["target"], geom.time_jd_utc, float(row["comet_ra"]),
                                              float(row["comet_dec"]), geom.r_h, geom.r_dot, geom.delta,
                                              geom.delta_dot, geom.phase, 0.0, 0.0, geom.observer)
                c = CometInField(row["target"], float(x), float(y), fake_geom, "null", 0.0)
                if frame.contains_pixel(c.x, c.y):
                    res = measure(frame, rate, c, out_dir / "stamps" / f"{row['stem']}_null.png")
                    recs.append({**base, "role": "null", "designation": row["target"],
                                 "found_blind": False, "blind_list": ";".join(blind_names),
                                 "pair": row["pair"], **res})
                else:
                    recs.append({**base, "role": "null", "designation": row["target"],
                                 "blind_list": ";".join(blind_names), "error": "null position off frame"})
        except Exception as exc:
            traceback.print_exc()
            recs = [dict(kind=row["kind"], target=row["target"], stem=row["stem"], role="error",
                         error=str(exc))]
        with open(done_path, "a") as fh:
            for r in recs:
                fh.write(json.dumps(r, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
        log.info("[%d/%d] %s %s: %s", k + 1, len(rows_in), row["kind"], row["stem"],
                 ", ".join(f"{r.get('designation')}({r.get('role')}, snr5={r.get('snr_5as', np.nan):.1f}, "
                           f"{r.get('growth_stop', r.get('error', ''))})" for r in recs) or "no comets")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    main(*sys.argv[1:4])
