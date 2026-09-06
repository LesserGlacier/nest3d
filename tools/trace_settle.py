"""Record the closing settle, pass by pass, as a viewer track.

The search tracks show arrangements being *chosen*; this one shows a single
arrangement being *tightened*.  Nothing here is a new packing -- every frame
is the same nine parts in the same orientations, nudged onto a finer lattice
until the box stops giving.

Every rung of the ladder is traced, serially and in-process, and the frames
kept are the winner's.  That is slower than the ladder itself, which runs
its rungs in parallel and keeps only a box, but a trace is an offline job
and running the rungs here is the only way to know which one to record.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nest3d import pipeline                                 # noqa: E402
from nest3d.settle import _bbox, _rungs, _settle_once       # noqa: E402
from nest3d.solve import part_volume                        # noqa: E402
from nest3d.viewerdata import write                         # noqa: E402


def _frame(fine, offsets, label, t0, order):
    lo, hi = _bbox(fine, offsets)
    ext = hi - lo
    out = []
    for i in order:
        m = fine[i].matrix(offsets[i])
        row = []
        for r in range(3):
            row.extend(round(float(m[r, c]), 6) for c in range(3))
            row.append(round(float(m[r, 3]), 2))
        out.append(row)
    return {
        "m": out,
        "order": [int(i) for i in order],
        "ext": [round(float(v), 2) for v in ext],
        "lo": [round(float(v), 2) for v in lo],
        "vol": float(np.prod(ext)),
        "phase": label,
        "best": False,
        "t": round(time.time() - t0, 2),
    }


def _mark_best(frames):
    """Flag the frames that improved on everything before them."""
    best = float("inf")
    for f in frames:
        if f["vol"] < best - 1e-9:
            best = f["vol"]
            f["best"] = True
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--time", type=float, default=240.0,
                    help="budget for the search that produces the arrangement")
    ap.add_argument("--workers", type=int, default=12,
                    help="search workers, and how many ladder rungs to trace")
    ap.add_argument("--settle-resolution", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=120.0,
                    help="seconds allowed per rung while tracing")
    ap.add_argument("--out", default=str(ROOT / "viewer_settle.json"))
    args = ap.parse_args()

    paths = []
    for pat in args.parts:
        hits = sorted(glob.glob(pat))
        paths.extend(hits if hits else [pat])

    res = pipeline.run(paths, time_budget=args.time, seed=args.seed,
                       workers=args.workers, settle=False, verbose=True)
    packed = res.result
    parts = res.parts
    order = sorted(range(len(parts)))
    print("packed  %s mm   vol %.4g   density %.1f%%"
          % (np.round(packed.extents, 1), packed.volume, 100 * packed.density),
          flush=True)

    best = None
    for resolution, shake in _rungs(args.settle_resolution, args.workers):
        frames = []
        t0 = time.time()
        got = _settle_once(
            [p.mesh for p in parts], packed.packer, packed.packing, resolution,
            0.0, 8, t0 + args.budget, shake,
            lambda fine, offsets, label: frames.append(
                _frame(fine, offsets, label, t0, order)))
        vol = float(np.prod(got[3])) if got else float("inf")
        print("  rung res=%-4d tip=%-5s  %d frames   vol %.4g   (%.1fs)"
              % (resolution, shake, len(frames), vol, time.time() - t0),
              flush=True)
        if got is not None and (best is None or vol < best[0]):
            best = (vol, frames, resolution, shake)

    if best is None or len(best[1]) < 2:
        raise SystemExit("no rung improved on the packed arrangement")

    vol, frames, resolution, shake = best
    frames = _mark_best(frames)
    payload = {
        "parts": [{"name": p.name} for p in parts],
        "part_volume": float(part_volume(packed.packer.poses)),
        "settle": {
            "resolution": resolution,
            "tipped": bool(shake),
            "frames": frames,
            "start_vol": float(packed.volume),
            "end_vol": vol,
        },
    }
    write(payload, args.out)
    print("wrote %s   winner res=%d tip=%s   %d frames   %.4g -> %.4g mm^3"
          % (args.out, resolution, shake, len(frames), packed.volume, vol))


if __name__ == "__main__":
    main()
