"""Re-run the field search on the frames the first run missed, with the fixes.

Fixes: the default sb_ident margin is now 5 deg (was 0.1), and a known-comet
EphemerisCache (SBDB candidate list + Horizons tracks) is a second source.
The paired control frames are searched too, to check that the extra sources
do not put the target comet on the controls.

Usage: python rerun_missed.py frames.csv results.jsonl RAW_DIR OUT.csv
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from astropy.time import Time

from comet_testbench import ZTF, find_comets, load_frames
from comet_testbench.ephemeris import horizons_provider
from comet_testbench.known_comets import EphemerisCache, candidate_comets


def main(frames_csv, results, raw_dir, out_csv):
    frames = {r["stem"]: r for r in csv.DictReader(open(frames_csv))}
    rows = [json.loads(l) for l in open(results)]
    missed = [r for r in rows if r["role"] == "target" and "x_pred" in r and not r.get("found_blind")]
    controls = [s for s, f in frames.items() if f["kind"] == "control" and f["pair"] in {m["stem"] for m in missed}]
    stems = [m["stem"] for m in missed] + controls
    adapter, eph = ZTF(), horizons_provider()
    loaded = {s: load_frames(Path(raw_dir) / f"{s}_sciimg.fits")[0] for s in stems}
    # One cache per calendar month of frames: candidates within 2 yr of perihelion.
    by_month = defaultdict(list)
    for s, f in loaded.items():
        by_month[f.mid_time.utc.strftime("%Y-%m")].append(f)
    caches = {}
    for month, fs in by_month.items():
        t0 = min(f.mid_time for f in fs)
        t1 = max(f.mid_time for f in fs)
        names = candidate_comets(t0, t1)
        caches[month] = EphemerisCache.build(names, t0, t1, adapter.observer(fs[0]))
        print(f"{month}: {len(names)} candidates, {len(caches[month].tracks)} tracks", flush=True)
    out = []
    for s in stems:
        f, meta = loaded[s], frames[s]
        cache = caches[f.mid_time.utc.strftime("%Y-%m")]
        found = find_comets(f, adapter=adapter, use_header_object=False, ephemeris=eph,
                            vmag_lim=22.0, known=cache)
        tgt = [c for c in found if c.designation == meta["target"]]
        out.append({"kind": meta["kind"], "target": meta["target"], "stem": s,
                    "target_found": bool(tgt), "found_by": tgt[0].found_by if tgt else "",
                    "all_found": ";".join(f"{c.designation}[{c.found_by}]" for c in found)})
        print(out[-1], flush=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main(*sys.argv[1:5])
