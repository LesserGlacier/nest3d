"""Run the real multi-chain search with recording, and emit viewer data.

This is the production search -- the same 20 parallel chains that produce
the answer -- with each chain returning its improvement trace.  The viewer
built from this ends on the arrangement that was actually chosen, rather
than on whatever a single watchable chain happened to reach.
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
from nest3d.solve import solve_multistart                   # noqa: E402
from nest3d.trace import Recorder                           # noqa: E402
from nest3d.viewerdata import build, write                  # noqa: E402
from nest3d.voxel import build_poses, suggest_pitch         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--resolution", type=int, default=48)
    ap.add_argument("--free-budget", type=float, default=110.0)
    ap.add_argument("--squeeze-budget", type=float, default=200.0)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=420)
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

    t = time.time()
    result = solve_multistart(
        poses, meshes=meshes, seed=args.seed,
        free_budget=args.free_budget, budget=args.squeeze_budget,
        workers=args.workers, chains=max(args.workers, 4),
        record=True, say=lambda m: print(m, flush=True))
    print("search done in %.0fs" % (time.time() - t), flush=True)

    ext = np.asarray(result.extents, dtype=float)
    print("WINNER %s mm  vol %.4g  density %.1f%%"
          % (np.round(ext, 1), result.volume, 100 * result.density), flush=True)

    # Merge every chain's improvements into one timeline, then append the
    # winning arrangement so the track ends on the chosen answer.
    merged = Recorder()
    for tr in result.traces:
        merged.frames.extend(tr)
    merged.frames.sort(key=lambda f: f.t)
    merged.frames.sort(key=lambda f: -f.volume)   # worst first, best last

    winner = Recorder()
    winner.phase = "winner"
    winner.record(result.winner_packing, accepted=True, is_best=True)
    merged.frames.extend(winner.frames)

    print("frames: %d from %d chains (+ winner)"
          % (len(merged.frames), len(result.traces)), flush=True)

    payload = build(parts, poses, merged, result.lower_bound,
                    max_search_frames=args.max_frames)
    # This ladder is ordered by quality across all chains, not by time --
    # it is not the annealer's walk, and mislabelling it as one would be a
    # lie about what you are watching. The walk comes from a single
    # recorded chain and is merged in separately.
    payload["tracks"]["search"] = []
    payload["stats"]["chains"] = len(result.traces)
    write(payload, args.out)
    print("wrote %s (%.1f MB) %d frames"
          % (args.out, Path(args.out).stat().st_size / 1e6,
             len(payload["tracks"]["best"])), flush=True)


if __name__ == "__main__":
    main()
