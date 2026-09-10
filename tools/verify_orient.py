"""Check that a re-oriented settle still cannot output an interfering pack.

The no-overlap guarantee is the one property this tool must not trade away
for a tighter box, and the orientation pass is the first thing in the
settle that changes a part's *mask* rather than only its offset.  A mask
built for a tilted pose is conservative in exactly the same way as any
other -- the same separating-axis test, the same fill -- so the argument
carries over; but an argument is not a measurement, and this is the
measurement.

Every pair of placed parts is taken back to its triangle mesh at the
transform the settle reports, and intersected exactly.  Any non-zero
intersection volume is a failure of the guarantee, not a rounding
difference.
"""
from __future__ import annotations

import argparse
import glob
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nest3d import settle as settle_mod                   # noqa: E402


def placed_meshes(meshes, packer, packing):
    out = []
    for pl in packing.placements:
        pose = packer.poses[pl.part_index][pl.pose_index]
        m = meshes[pl.part_index].copy()
        m.apply_transform(pose.matrix(pl.offset))
        out.append((pl.part_index, m))
    return out


def worst_overlap(placed):
    """Largest exact intersection volume over every pair, in model units^3."""
    worst = 0.0
    culprit = None
    for a in range(len(placed)):
        ia, ma = placed[a]
        for b in range(a + 1, len(placed)):
            ib, mb = placed[b]
            # Skip pairs whose AABBs are already apart -- an exact boolean
            # is expensive and cannot find anything there.
            lo = np.maximum(ma.bounds[0], mb.bounds[0])
            hi = np.minimum(ma.bounds[1], mb.bounds[1])
            if np.any(hi <= lo):
                continue
            inter = ma.intersection(mb)
            v = float(inter.volume) if inter is not None and not inter.is_empty else 0.0
            if v > worst:
                worst, culprit = v, (ia, ib)
    return worst, culprit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "nest3d_bench"))
    ap.add_argument("--settle-resolution", type=int, default=96)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    files = sorted(Path(args.cache).glob("arr*.pkl"))
    if not files:
        raise SystemExit("no cached arrangements -- run bench_settle.py --pack")

    bad = 0
    for path in files:
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        for orient in (False, True):
            t0 = time.time()
            got = settle_mod.settle(
                blob["meshes"], blob["packer"], blob["packing"],
                resolution=args.settle_resolution, workers=args.workers,
                orient=orient)
            if got is None:
                print("%s orient=%-5s  no improvement" % (path.name, orient),
                  flush=True)
                continue
            s_packer, s_packing, ext = got
            placed = placed_meshes(blob["meshes"], s_packer, s_packing)
            worst, culprit = worst_overlap(placed)
            vol = float(np.prod(ext))
            ok = "OK" if worst <= 0.0 else "FAIL"
            if worst > 0.0:
                bad += 1
            print("%s orient=%-5s  density %5.2f%%  worst pair overlap "
                  "%.6g mm^3  %s%s  (%.1fs)"
                  % (path.name, orient, 100 * blob["part_volume"] / vol,
                     worst, ok, "" if culprit is None else " %s" % (culprit,),
                     time.time() - t0), flush=True)
    print("\n%s" % ("all arrangements clear" if not bad
                    else "%d INTERFERING ARRANGEMENTS" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
