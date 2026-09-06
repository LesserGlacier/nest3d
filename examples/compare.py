"""Compare the packer against the simpler things you might do instead.

Runs the same nine parts four ways and prints the resulting boxes, then
verifies the winning arrangement with exact boolean intersections.
"""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nest3d.geometry import load_parts                     # noqa: E402
from nest3d.orient import candidate_rotations              # noqa: E402
from nest3d.pack import Packer                             # noqa: E402
from nest3d.solve import solve_multistart                  # noqa: E402
from nest3d.voxel import build_poses, suggest_pitch        # noqa: E402

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 12


def pack(meshes, rotations_fn, pitch, label, part_vol, seed=5):
    poses = [build_poses(m, rotations_fn(m), pitch) for m in meshes]
    t = time.time()
    res = solve_multistart(poses, meshes=meshes, seed=seed, budget=BUDGET,
                           workers=WORKERS, chains=max(WORKERS, 4))
    ext = np.sort(res.extents)[::-1]
    print("  %-28s %6.1f x %6.1f x %6.1f mm   %9.4g mm3   %5.1f%%   %5.1fs"
          % (label, ext[0], ext[1], ext[2], res.volume,
             100 * part_vol / res.volume, time.time() - t))
    return res, poses


def main():
    files = sorted(glob.glob(str(ROOT / "examples" / "sample_parts" / "*.step")))
    parts = load_parts(files)
    meshes = [p.mesh for p in parts]
    part_vol = sum(abs(float(m.volume)) for m in meshes)
    pitch = suggest_pitch(meshes, 28)

    print("\n%d parts, %.4g mm3 of solid, voxel pitch %.2f mm, %.0fs/chain "
          "on %d workers" % (len(parts), part_vol, pitch, BUDGET, WORKERS))
    print("\n  %-28s %-32s %-14s %-8s %s"
          % ("method", "bounding box", "volume", "density", "time"))

    # 1. No rotation at all: parts as they arrive in their files.
    pack(meshes, lambda m: np.eye(3)[None, :, :], pitch,
         "as-supplied orientation", part_vol)

    # 2. Bounding boxes instead of real shapes -- the usual hand estimate.
    boxes = []
    for m in meshes:
        b = trimesh.creation.box(extents=m.extents)
        boxes.append(b)
    box_vol = sum(abs(float(b.volume)) for b in boxes)
    res_b, _ = pack(boxes, lambda m: candidate_rotations(m, use_rest=False),
                    pitch, "each part as its own AABB", box_vol)
    print("      (that box holds %.4g mm3 of real part, %.1f%% dense)"
          % (part_vol, 100 * part_vol / res_b.volume))

    # 3. The 24 axis-aligned rotations.
    pack(meshes, lambda m: candidate_rotations(m, use_rest=False), pitch,
         "24 box rotations", part_vol)

    # 4. Full candidate set: box rotations plus stable resting poses.
    best, poses = pack(meshes, lambda m: candidate_rotations(m, use_rest=True),
                       pitch, "rotations + resting poses", part_vol)

    # Verify the winner against exact geometry.
    print("\n  verifying the best arrangement with boolean intersections...")
    placed = []
    for pl in best.packing.placements:
        pose = best.packer.poses[pl.part_index][pl.pose_index]
        m = meshes[pl.part_index].copy()
        m.apply_transform(pose.matrix(pl.offset))
        placed.append((parts[pl.part_index].name, m))

    worst = 0.0
    pairs = 0
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            inter = placed[i][1].intersection(placed[j][1])
            pairs += 1
            v = abs(float(inter.volume)) if inter is not None and len(inter.vertices) else 0.0
            if v > worst:
                worst = v
    print("  %d pairs checked, largest intersection %.6g mm3" % (pairs, worst))
    print("  %s" % ("PASS - no interference" if worst <= 1e-6
                    else "FAIL - parts interfere"))


if __name__ == "__main__":
    main()
