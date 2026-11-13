"""Command line entry point: ``python -m comet_testbench {fetch,run} ...``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import BenchConfig, run, write_results


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
    r.add_argument("--no-handoff", action="store_true", help="skip the OH/H2O hand-off files")

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
    paths = write_results(results, args.out, handoff=not args.no_handoff)
    for p in paths:
        print(f"hand-off: {p}")
    print(f"{len(results)} comet measurements written to {args.out}/")


if __name__ == "__main__":
    main()
