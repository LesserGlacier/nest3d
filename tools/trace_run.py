"""Run a single recorded solve and emit the viewer payload.

Deliberately single-process: the recording lives in memory alongside the
search, and a worker pool would have to ship every frame back through a
pipe.  One chain is also what you want to watch -- twenty interleaved
chains are not a story.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nest3d.geometry import load_parts                      # noqa: E402
from nest3d.orient import candidate_rotations               # noqa: E402
from nest3d.solve import Solver                             # noqa: E402
from nest3d.trace import Recorder                           # noqa: E402
from nest3d.viewerdata import build, write                  # noqa: E402
from nest3d.voxel import build_poses, suggest_pitch         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--resolution", type=int, default=40)
    ap.add_argument("--free-budget", type=float, default=90.0)
    ap.add_argument("--squeeze-budget", type=float, default=150.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--out", default=str(ROOT / "viewer_data.json"))
    args = ap.parse_args()

    paths = []
    for pat in args.parts:
        hits = sorted(glob.glob(pat))
        paths.extend(hits if hits else [pat])
    parts = load_parts(paths)
    meshes = [p.mesh for p in parts]
    pitch = suggest_pitch(meshes, args.resolution)
    print("loaded %d parts, pitch %.2f mm" % (len(parts), pitch), flush=True)

    poses = [build_poses(p.mesh, candidate_rotations(p.mesh, use_rest=True),
                         pitch) for p in parts]
    print("poses: %s" % [len(p) for p in poses], flush=True)

    rec = Recorder()
    solver = Solver(poses, meshes=meshes, seed=args.seed, verbose=True,
                    recorder=rec)

    t = time.time()
    packer, best = solver.free_phase(starts=3, iterations=4000,
                                     budget=args.free_budget)
    print("free done  %.0fs  %d frames" % (time.time() - t, len(rec.frames)),
          flush=True)

    ext = np.asarray(best.packing.extents, dtype=float)
    sq_packer, sq_best, sq_ext, _hist = solver.squeeze(
        ext, budget=args.squeeze_budget, starts=2, iterations=4000,
        workers=1, rounds=3, warm=[(best.order, best.pose_choice)])
    print("squeeze done  %.0fs  %d frames" % (time.time() - t, len(rec.frames)),
          flush=True)

    final = sq_best if sq_best is not None else best
    final_ext = sq_ext if sq_best is not None else ext
    print("best %s mm  vol %.4g  density %.1f%%"
          % (np.round(final_ext, 1), float(np.prod(final_ext)),
             100 * solver.part_volume / float(np.prod(final_ext))), flush=True)

    payload = build(parts, poses, rec, solver.part_volume,
                    best_packing=final.packing,
                    max_search_frames=args.max_frames)
    write(payload, args.out)
    size = Path(args.out).stat().st_size / 1e6
    print("wrote %s  (%.1f MB)  %d best / %d search frames from %d evaluations"
          % (args.out, size, len(payload["tracks"]["best"]),
             len(payload["tracks"]["search"]), payload["stats"]["evaluated"]),
          flush=True)


if __name__ == "__main__":
    main()
