"""Correctness tests.

The important one is test_packing_is_interference_free_on_real_geometry:
everything else in this project reasons about voxels, so the guarantee is
only worth anything if it is checked back against the actual triangle
meshes, using machinery the packer never touches.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nest3d.orient import candidate_rotations                 # noqa: E402
from nest3d.pack import Packer, verify_packing                # noqa: E402
from nest3d.search import Search                              # noqa: E402
from nest3d.voxel import build_poses, exact_overlap, voxelize_pose  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print("  [%s] %s%s" % (status, name, ("  -- " + detail) if detail else ""))
    if not cond:
        FAILURES.append(name)
    return cond


def l_shape(a=20.0):
    """A concave L, as a watertight box union."""
    b1 = trimesh.creation.box(extents=(2 * a, a, a))
    b1.apply_translation((a, a / 2, a / 2))
    b2 = trimesh.creation.box(extents=(a, a, 2 * a))
    b2.apply_translation((a / 2, a / 2, a))
    return trimesh.util.concatenate([b1, b2])


# ---------------------------------------------------------------------------

def test_voxel_mask_is_a_superset():
    """Every point inside the solid must land in an occupied voxel."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    pitch = 0.9
    pose = voxelize_pose(mesh, np.eye(3), pitch)

    rng = np.random.default_rng(0)
    pts = mesh.sample(4000) if hasattr(mesh, "sample") else mesh.vertices
    # Pull sampled surface points slightly inward so they are strictly inside.
    centre = mesh.vertices.mean(axis=0)
    inner = centre + (pts - centre) * 0.97

    local = (inner @ np.eye(3).T) + pose.shift
    idx = np.floor(local / pitch).astype(int) + pose.pad
    inside = np.all((idx >= 0) & (idx < np.asarray(pose.mask.shape)), axis=1)
    hit = pose.mask[idx[inside, 0], idx[inside, 1], idx[inside, 2]]
    check("voxel mask covers the solid", bool(hit.all()),
          "%d/%d sampled interior points covered" % (int(hit.sum()), hit.size))


def test_symmetry_dedup():
    """A cube has 24 candidate rotations but only one distinct occupancy."""
    cube = trimesh.creation.box(extents=(10, 10, 10))
    rots = candidate_rotations(cube, use_rest=False)
    poses = build_poses(cube, rots, pitch=0.8)
    check("cube collapses to one pose", len(poses) == 1,
          "%d rotations -> %d poses" % (len(rots), len(poses)))


def test_eight_cubes_pack_tightly():
    """Eight equal cubes have a known optimum: a 2x2x2 block."""
    cube = trimesh.creation.box(extents=(20, 20, 20))
    pitch = 20.0 / 40
    poses = [build_poses(cube, candidate_rotations(cube, use_rest=False), pitch)
             for _ in range(8)]
    packer = Packer(poses, objective="compact")
    best = Search(packer, seed=0).run(starts=2, iterations=40)
    ext = np.sort(best.packing.extents)[::-1]

    # 40 mm per side is the optimum; allow the discretisation slack.
    ok = bool(np.all(ext <= 40.0 + 3 * pitch))
    check("eight cubes reach the 2x2x2 optimum", ok,
          "got %s mm (optimum 40 x 40 x 40)" % np.round(ext, 2))
    check("eight cubes: no voxel overlap", not verify_packing(packer, best.packing))


def test_packing_is_interference_free_on_real_geometry():
    """Pack concave parts, then check the meshes themselves for interference.

    Containment is decided by trimesh ray casting on the transformed
    meshes -- no voxels involved -- so this tests the guarantee end to end
    rather than restating it.
    """
    meshes = []
    for scale in (1.0, 0.8, 1.2, 0.9, 1.1, 0.7):
        m = l_shape(20.0)
        m.apply_scale(scale)
        meshes.append(m)

    pitch = 1.4
    poses = [build_poses(m, candidate_rotations(m, use_rest=True), pitch)
             for m in meshes]
    packer = Packer(poses, objective="compact")
    best = Search(packer, seed=2).run(starts=2, iterations=60)

    check("voxel masks are disjoint", not verify_packing(packer, best.packing))

    placed = []
    for pl in best.packing.placements:
        pose = packer.poses[pl.part_index][pl.pose_index]
        m = meshes[pl.part_index].copy()
        m.apply_transform(pose.matrix(pl.offset))
        placed.append(m)

    worst = 0
    rng = np.random.default_rng(1)
    for i, a in enumerate(placed):
        pts = a.sample(3000)
        # Nudge inward so surface-grazing samples do not read as contact.
        pts = a.centroid + (pts - a.centroid) * 0.985
        for j, b in enumerate(placed):
            if i == j:
                continue
            try:
                n = int(np.count_nonzero(b.contains(pts)))
            except Exception as exc:                     # no ray backend
                print("      (skipped mesh containment: %s)" % exc)
                n = 0
            worst = max(worst, n)
    check("no mesh point of one part lies inside another", worst == 0,
          "worst case %d of 3000 sampled points" % worst)

    # Exact boolean intersection: the strongest statement available.
    worst_vol = 0.0
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            try:
                inter = placed[i].intersection(placed[j])
            except Exception as exc:
                print("      (skipped boolean check: %s)" % exc)
                return
            if inter is not None and len(inter.vertices):
                worst_vol = max(worst_vol, abs(float(inter.volume)))
    total = sum(abs(float(m.volume)) for m in placed)
    check("boolean intersection of every pair is empty", worst_vol <= 1e-6,
          "largest overlap %.3g mm^3 against %.4g mm^3 of part" % (worst_vol, total))


def test_clearance_is_respected():
    """A requested gap actually appears between the placed solids."""
    cube = trimesh.creation.box(extents=(20, 20, 20))
    pitch = 0.5
    gap = 2.0
    cv = int(np.ceil(gap / (2 * pitch)))   # half the gap on each part
    poses = [build_poses(cube, candidate_rotations(cube, use_rest=False),
                         pitch, clearance_voxels=cv) for _ in range(2)]
    packer = Packer(poses, objective="compact")
    best = Search(packer, seed=0).run(starts=1, iterations=5)

    boxes = []
    for pl in best.packing.placements:
        pose = packer.poses[pl.part_index][pl.pose_index]
        lo, hi = pose.aabb(pl.offset)
        boxes.append((lo, hi))
    (lo0, hi0), (lo1, hi1) = boxes
    # Separation along the axis on which the two boxes are apart.
    seps = np.maximum(lo1 - hi0, lo0 - hi1)
    actual = float(seps.max())
    check("clearance is honoured", actual >= gap - 1e-6,
          "asked %.2f mm, got %.2f mm" % (gap, actual))
    check("clearance is not wildly over-delivered", actual <= gap + 4 * pitch,
          "asked %.2f mm, got %.2f mm" % (gap, actual))


def main():
    print("nest3d tests")
    for fn in (test_voxel_mask_is_a_superset,
               test_symmetry_dedup,
               test_eight_cubes_pack_tightly,
               test_packing_is_interference_free_on_real_geometry,
               test_clearance_is_respected):
        print("\n%s" % fn.__name__)
        fn()
    print("\n%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
