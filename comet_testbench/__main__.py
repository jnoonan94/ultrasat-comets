"""Command line entry point: ``python -m comet_testbench {fetch,run} ...``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import BenchConfig, fit_all, oh_measurements, run, write_results


def _pairs(paths):
    """Match each *_sk.img(.gz) with its *_ex.img(.gz) exposure map if present."""
    out = []
    for p in map(Path, paths):
        if "_ex." in p.name:
            continue
        ex = p.with_name(p.name.replace("_sk.", "_ex."))
        out.append((p, ex if ex != p and ex.exists() else None))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="comet_testbench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download Swift/UVOT images for a comet")
    f.add_argument("pattern", help="SQL LIKE pattern on the Swift target name, e.g. '%%46P%%'")
    f.add_argument("--start")
    f.add_argument("--stop")
    f.add_argument("--filters", default="UVW1,V")
    f.add_argument("--dest", default="data/swift")
    f.add_argument("--limit", type=int, default=None, help="at most this many observations")

    r = sub.add_parser("run", help="find comets and measure them in FITS images")
    r.add_argument("images", nargs="+", help="FITS images (Swift *_ex maps are paired automatically)")
    r.add_argument("--out", default="results")
    r.add_argument("--config", help="JSON file with BenchConfig fields")
    r.add_argument("--candidate", action="append", default=[], help="comet to test, e.g. 46P")
    r.add_argument("--no-sb-ident", action="store_true", help="skip the JPL sb_ident discovery query")
    r.add_argument("--beta", type=float, help="dust continuum scaling UVW1 = beta * V")
    r.add_argument("--oh-conversion", type=float, help="erg s-1 cm-2 per UVW1 count/s for OH")
    r.add_argument("--oh-model", help="OH model as package.module:ClassOrFactory")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "fetch":
        from .swift_archive import download_images, find_observations

        obs = find_observations(args.pattern, args.start, args.stop, min_exposure=1)
        if args.limit:
            obs = obs[: args.limit]
        got = download_images(obs, args.filters.split(","), args.dest)
        print(f"{len(obs)} observations, {len(got)} images in {args.dest}")
        return

    cfg = BenchConfig.from_json(args.config) if args.config else BenchConfig()
    if args.candidate:
        cfg.candidates = list(cfg.candidates) + args.candidate
    if args.no_sb_ident:
        cfg.use_sb_ident = False
    results = run(_pairs(args.images), cfg)
    oh = fits_ = None
    if args.beta is not None and args.oh_conversion is not None:
        oh = oh_measurements(results, args.beta, args.oh_conversion)
        if args.oh_model:
            from .oh import load_model

            fits_ = fit_all(load_model(args.oh_model), oh)
    write_results(results, args.out, oh, fits_)
    print(f"{len(results)} comet measurements written to {args.out}/")


if __name__ == "__main__":
    main()
