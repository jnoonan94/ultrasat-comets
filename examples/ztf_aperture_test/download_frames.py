"""Download the sciimg files listed in frames.csv (2 at a time; IRSA answers more with HTTP 503).

With --follow, keep re-reading the CSV until select_frames.py has finished
(its process is gone and every row is downloaded).
Usage: python download_frames.py frames.csv RAW_DIR [--follow]
"""

import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from comet_testbench.ztf_archive import _get


def fetch(url, path):
    if path.exists():
        return
    try:
        r = _get(url, stream=True, timeout=600, retries=3)
    except Exception as exc:  # IRSA answers bursts with 503; report and move on
        print("FAILED", path.name, exc, flush=True)
        return
    tmp = path.with_suffix(".part")
    with open(tmp, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    tmp.rename(path)
    print("got", path.name, flush=True)


def main(frames_csv, raw_dir, follow=False):
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    queued = set()
    with ThreadPoolExecutor(2) as pool:
        while True:
            rows = list(csv.DictReader(open(frames_csv))) if Path(frames_csv).exists() else []
            for r in rows:
                if r["stem"] not in queued:
                    queued.add(r["stem"])
                    pool.submit(fetch, r["url"], raw / f"{r['stem']}_sciimg.fits")
            running = subprocess.run(["pgrep", "-f", "select_frames.py"], capture_output=True).returncode == 0
            if not follow or not running:
                break
            time.sleep(20)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--follow" in sys.argv)
