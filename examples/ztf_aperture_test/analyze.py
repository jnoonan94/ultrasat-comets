"""Summarise run_bench.py output: field search, detection, apertures.

Usage: python analyze.py frames.csv results.jsonl OUT_DIR
Writes summary.json, per_frame.csv and figures into OUT_DIR.
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def load(frames_csv, results):
    stems = {r["stem"] for r in csv.DictReader(open(frames_csv))}
    rows = [json.loads(l) for l in open(results)]
    return [r for r in rows if r["stem"] in stems]


def pct(a, q):
    a = np.asarray([v for v in a if v is not None and np.isfinite(v)])
    return float(np.percentile(a, q)) if a.size else float("nan")


def detected(r, snr_min=5.0, max_shift=5.0):
    return bool(r.get("snr_5as") is not None and r["snr_5as"] >= snr_min
                and r.get("recover_shift_arcsec", 99) <= max_shift)


def main(frames_csv, results, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load(frames_csv, results)
    frames = list(csv.DictReader(open(frames_csv)))
    tgt = [r for r in rows if r["role"] == "target"]
    placed = [r for r in tgt if "x_pred" in r]
    null = [r for r in rows if r["role"] == "null" and "x_pred" in r]
    other = [r for r in rows if r["role"] == "other"]
    S = {}
    S["frames"] = {"comet": sum(f["kind"] == "comet" for f in frames),
                   "control": sum(f["kind"] == "control" for f in frames),
                   "comets": len({f["target"] for f in frames}),
                   "filters": dict(Counter(r.get("filter") for r in rows if r["role"] in ("target", "null"))),
                   "errors": [r for r in rows if r["role"] == "error"]}
    # A. Field search
    S["field_search"] = {
        "targets": len(tgt),
        "placed_on_frame": len(placed),
        "found_blind_default_margin": sum(bool(r.get("found_blind")) for r in tgt),
        "found_blind_2deg_margin": sum(bool(r.get("found_wide")) for r in tgt),
        "missed_default": sorted({(r["designation"], r["stem"]) for r in placed if not r.get("found_blind")}),
        "missed_2deg": sorted({(r["designation"], r["stem"]) for r in placed if not r.get("found_wide")}),
        "other_comets": [(r["designation"], r["kind"], r["stem"], r.get("sb_ident_vmag"),
                          round(r.get("snr_5as", float("nan")), 1)) for r in other],
        "comets_in_controls": sorted({(r["stem"], r["blind_list"]) for r in rows
                                      if r["kind"] == "control" and r.get("blind_list")}),
    }
    # B. Position
    good = [r for r in placed if r.get("peak_snr_10as", 0) >= 10]
    S["position"] = {
        "n_with_signal": len(good),
        "offset_arcsec_median": pct([r["offset_arcsec"] for r in good], 50),
        "offset_arcsec_p90": pct([r["offset_arcsec"] for r in good], 90),
        "offset_arcsec_max": pct([r["offset_arcsec"] for r in good], 100),
        "along_median": pct([r["along_arcsec"] for r in good], 50),
        "cross_median": pct([r["cross_arcsec"] for r in good], 50),
        "null_recenter_shift_median": pct([r.get("recenter_shift_arcsec") for r in null], 50),
        "null_recenter_shift_p90": pct([r.get("recenter_shift_arcsec") for r in null], 90),
    }
    # C. Detection at the predicted position
    S["detection_5as"] = {
        "targets_detected": sum(detected(r) for r in placed), "targets": len(placed),
        "nulls_detected": sum(detected(r) for r in null), "nulls": len(null),
        "target_snr_min": pct([r.get("snr_5as") for r in placed], 0),
        "null_snr_max": pct([r.get("snr_5as") for r in null], 100),
        "false_null": [(r["stem"], round(r["snr_5as"], 1), round(r["recover_shift_arcsec"], 1))
                       for r in null if detected(r)],
        "missed_targets": [(r["designation"], r["stem"], round(r.get("snr_5as", np.nan), 1))
                           for r in placed if not detected(r)],
    }
    # D. Growth curves
    def growth_summary(rs, stop_key, r_key):
        return {"stop": dict(Counter(r.get(stop_key) for r in rs)),
                "r_conv_median": pct([r.get(r_key) for r in rs], 50)}
    S["growth_bench"] = {"targets": growth_summary(placed, "growth_stop", "r_conv_arcsec"),
                         "nulls": growth_summary(null, "growth_stop", "r_conv_arcsec"),
                         "nulls_with_snr_conv_gt5": sum((r.get("snr_conv") or 0) > 5 for r in null),
                         "nulls_converged_flag": sum(bool(r.get("converged")) for r in null)}
    S["growth_median_profile"] = {"targets": growth_summary(placed, "med_stop", "med_r_conv_arcsec"),
                                  "nulls": growth_summary(null, "med_stop", "med_r_conv_arcsec")}
    dm = [r["mag_conv"] - r["med_mag_conv"] for r in placed
          if np.isfinite(r.get("mag_conv", np.nan)) and np.isfinite(r.get("med_mag_conv", np.nan))]
    S["growth_median_profile"]["mag_bench_minus_median_median"] = pct(dm, 50)
    # E. Fixed apertures
    fixed = {}
    for km in (5000, 10000, 20000, 40000):
        rs = [r for r in placed if f"r_{km}km_arcsec" in r]
        fixed[km] = {"r_arcsec_median": pct([r[f"r_{km}km_arcsec"] for r in rs], 50),
                     "n_below_2x_seeing": sum(r[f"r_{km}km_arcsec"] < 2 * r["seeing"] for r in rs),
                     "n_partial_coverage": sum(r[f"cov_{km}km"] < 0.999 for r in rs),
                     "snr_median": pct([r[f"snr_{km}km"] for r in rs], 50), "n": len(rs)}
    S["fixed_km"] = fixed
    json.dump(S, open(out / "summary.json", "w"), indent=1, default=str)

    keys = ["kind", "role", "target", "designation", "stem", "filter", "mid_time", "seeing", "delta",
            "r_h", "rate_arcsec_hr", "sb_ident_vmag", "found_blind", "found_wide", "x_pred", "y_pred",
            "offset_arcsec", "along_arcsec", "cross_arcsec", "peak_snr_10as", "snr_5as",
            "recover_shift_arcsec", "growth_stop", "r_conv_arcsec", "snr_conv", "mag_conv",
            "med_stop", "med_r_conv_arcsec", "med_mag_conv", "mag_10000km", "snr_10000km",
            "r_10000km_arcsec", "error"]
    with open(out / "per_frame.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    figures(placed, null, out)
    print(json.dumps(S, indent=1, default=str))


def figures(placed, null, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    bins = np.logspace(-1, 3.5, 40)
    t = np.clip([abs(r["snr_5as"]) for r in placed if "snr_5as" in r], 0.1, None)
    n = np.clip([abs(r["snr_5as"]) for r in null if "snr_5as" in r], 0.1, None)
    axs[0].hist(t, bins, color="#2a6fdb", alpha=0.8, label=f"comet frames (n={len(t)})")
    axs[0].hist(n, bins, color="#e07b39", alpha=0.8, label=f"same sky, no comet (n={len(n)})")
    axs[0].axvline(5, color="k", ls="--", lw=1)
    axs[0].set_xscale("log"), axs[0].set_xlabel("|SNR| in 5\" aperture at predicted position")
    axs[0].legend(fontsize=8, frameon=False)
    good = [r for r in placed if r.get("peak_snr_10as", 0) >= 10]
    axs[1].scatter([r["along_arcsec"] for r in good], [r["cross_arcsec"] for r in good],
                   s=14, color="#2a6fdb")
    axs[1].axhline(0, color="0.7", lw=0.8), axs[1].axvline(0, color="0.7", lw=0.8)
    axs[1].set_xlabel("along-track offset (\")"), axs[1].set_ylabel("cross-track offset (\")")
    axs[1].set_title("centroid - Horizons prediction", fontsize=9)
    lab = ["converged", "growth curve turned over", "reached r_max"]
    x = np.arange(3)
    for k, (rs, c, name) in enumerate([(placed, "#2a6fdb", "comet"), (null, "#e07b39", "no comet")]):
        cnt = Counter(r.get("growth_stop") for r in rs)
        axs[2].bar(x + 0.38 * k, [cnt.get(l, 0) for l in lab], 0.38, color=c, label=name)
    axs[2].set_xticks(x + 0.19, ["converged", "turned over", "hit r_max"])
    axs[2].set_ylabel("frames"), axs[2].set_title("bench growth-curve stop reason", fontsize=9)
    axs[2].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "ztf_aperture_summary.png", dpi=130)


if __name__ == "__main__":
    main(*sys.argv[1:4])
