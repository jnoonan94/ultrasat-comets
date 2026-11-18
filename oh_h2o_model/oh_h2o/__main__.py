"""python -m oh_h2o HANDOFF.ecsv [...]"""

import argparse
from pathlib import Path

from . import calibration_for, configure, run, write_results


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m oh_h2o", description=__doc__)
    p.add_argument("handoff", nargs="+", help="test bench hand-off ECSV file(s)")
    p.add_argument("--instrument", default="swift-uvot")
    p.add_argument("--oh-band", default=None, help="default: the calibration's (UVW1 for Swift)")
    p.add_argument("--continuum-band", default=None, help="default: the calibration's (V for Swift)")
    p.add_argument("--redness", type=float, nargs="+", default=[0.0], help="dust reddening, %%/100 nm")
    p.add_argument("--max-dt", type=float, default=60.0, help="max minutes between paired images")
    p.add_argument("--data-dir", default=None, help="swift_comet_pipeline data/ folder")
    p.add_argument("--cache-dir", default=None, help="vectorial model cache folder")
    p.add_argument("--quality", default="medium", choices=["low", "medium", "high", "very_high"])
    p.add_argument("--backend", default="sbpy", choices=["sbpy", "rust"])
    p.add_argument("--out", default="results/oh", help="output folder")
    a = p.parse_args(argv)

    configure(data_dir=a.data_dir, cache_dir=a.cache_dir, quality=a.quality, backend=a.backend)
    cal = calibration_for(a.instrument, a.oh_band, a.continuum_band)
    for h in a.handoff:
        t = run(h, cal, dust_redness=a.redness, max_dt_minutes=a.max_dt)
        name = Path(h).name.replace("_photometry.ecsv", "") + "_water.ecsv"
        path = write_results(t, Path(a.out) / name)
        print(f"{h}: {len(t)} rows -> {path}")


if __name__ == "__main__":
    main()
