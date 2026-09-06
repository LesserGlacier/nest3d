"""Candidate orientations for a part.

Full SO(3) is continuous, so "allowing all rotations" in practice means
choosing a finite candidate set that is dense where good solutions live.
Three families are generated and merged:

  axis     the 24 proper rotations of the cube, applied on top of the
           part's own minimum-volume oriented bounding box.  For machined
           and sheet-metal parts this alone is usually near optimal.
  rest     "flat face down" poses taken from the convex hull's large
           faces, each spun about the vertical by a few angles.  These are
           the stable resting poses, and the ones that matter if the pack
           has to be physically stacked or printed.
  random   a quasi-uniform sample of SO(3), so that genuinely oblique
           orientations are reachable.

Candidates are de-duplicated here only by rotation distance.  Exact
de-duplication happens later against the voxel occupancy (voxel.py), which
is what actually catches a part's own symmetry -- footprint alone does not,
since an L-bracket and that bracket turned 180 degrees share a footprint
but pack completely differently.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial.transform import Rotation


def _proper_cube_rotations() -> np.ndarray:
    """The 24 rotation matrices of the octahedral group."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for row, col in enumerate(perm):
                m[row, col] = signs[row]
            if abs(np.linalg.det(m) - 1.0) < 1e-9:
                mats.append(m)
    return np.array(mats)


CUBE_ROTATIONS = _proper_cube_rotations()


def obb_frame(mesh) -> np.ndarray:
    """Rotation taking the part's minimum-volume OBB axes onto world axes."""
    try:
        obb = mesh.bounding_box_oriented
        r = np.asarray(obb.primitive.transform)[:3, :3]
        # Strip any scale trimesh may have folded in, then invert.
        norms = np.linalg.norm(r, axis=0)
        norms[norms == 0] = 1.0
        r = r / norms
        if np.linalg.det(r) < 0:
            r[:, 2] *= -1
        return r.T
    except Exception:
        return np.eye(3)


def _hull_rest_rotations(mesh, max_faces=12, spins=4, area_frac=0.02):
    """Poses where a large convex-hull facet lies flat on the XY plane."""
    try:
        hull = mesh.convex_hull
    except Exception:
        return np.zeros((0, 3, 3))

    normals = np.asarray(hull.face_normals, dtype=float)
    areas = np.asarray(hull.area_faces, dtype=float)
    if len(normals) == 0:
        return np.zeros((0, 3, 3))

    # Merge coplanar facets: bucket by normal direction, sum their area.
    keys = np.round(normals, 3)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    merged = np.zeros(len(uniq))
    np.add.at(merged, inverse, areas)

    total = merged.sum()
    order = np.argsort(merged)[::-1]
    order = [i for i in order if merged[i] >= area_frac * total][:max_faces]

    down = np.array([0.0, 0.0, -1.0])
    out = []
    for i in order:
        n = uniq[i] / max(np.linalg.norm(uniq[i]), 1e-12)
        # Rotation carrying this facet normal to point straight down, so the
        # facet becomes the resting face.
        v = np.cross(n, down)
        c = float(np.dot(n, down))
        s = float(np.linalg.norm(v))
        if s < 1e-9:
            base = np.eye(3) if c > 0 else Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
        else:
            base = Rotation.from_rotvec(v / s * np.arctan2(s, c)).as_matrix()
        for k in range(spins):
            spin = Rotation.from_rotvec([0, 0, 2 * np.pi * k / spins]).as_matrix()
            out.append(spin @ base)
    return np.array(out) if out else np.zeros((0, 3, 3))


def _random_rotations(n, seed):
    """Quasi-uniform SO(3) sample, tolerant of the scipy 1.14 rename."""
    rng = np.random.default_rng(seed)
    try:
        return Rotation.random(n, rng=rng).as_matrix()
    except TypeError:
        return Rotation.random(n, random_state=seed).as_matrix()


def candidate_rotations(mesh, n_random=0, use_obb=True, use_rest=True,
                        use_cube=True, spins=4, seed=0,
                        dedup_angle_deg=8.0):
    """Build the candidate rotation set for one part.

    Returns an (N, 3, 3) array of rotation matrices, ordered so that the
    cheapest and most likely candidates come first.
    """
    groups = []
    base = obb_frame(mesh) if use_obb else np.eye(3)

    if use_cube:
        groups.append(np.einsum("nij,jk->nik", CUBE_ROTATIONS, base))
    else:
        groups.append(base[None, :, :])

    if use_rest:
        rest = _hull_rest_rotations(mesh, spins=spins)
        if len(rest):
            groups.append(rest)

    if n_random > 0:
        groups.append(_random_rotations(n_random, seed))

    cands = np.vstack([g for g in groups if len(g)])

    # Drop candidates that are within dedup_angle_deg of one already kept.
    # This is a cheap pre-filter; the exact occupancy-based dedup happens
    # once the masks are voxelized.
    kept = []
    cos_tol = np.cos(np.radians(dedup_angle_deg))
    for r in cands:
        dup = False
        for other in kept:
            ang_cos = (np.trace(r @ other.T) - 1.0) / 2.0
            if ang_cos > cos_tol:
                dup = True
                break
        if not dup:
            kept.append(r)

    return np.array(kept)
