"""Voxel discretisation of oriented parts.

Why voxels
----------
The parts are irregular and may be concave, so the interesting solutions
are the ones where parts nest into each other's hollows.  Convex-hull or
bounding-box packing throws exactly that away.  A voxel occupancy grid
keeps arbitrary concavity, makes the overlap test a bitwise AND, and gives
one honest quality/time knob (the pitch).

The no-overlap guarantee
------------------------
Each part's collision mask is built to be a *superset* of the true solid.
A voxel is marked when a triangle genuinely intersects that voxel's cube,
decided by the exact Akenine-Moller separating-axis test rather than by
sampling; the enclosed interior is then filled.  So the mask is exactly
"every voxel the solid touches", which is the tightest superset a lattice
of this pitch admits.

Because mask_i is a superset of solid_i, two parts whose masks are
disjoint have solids that are certainly disjoint.  The packer therefore
cannot produce an interfering arrangement -- it can only be slightly
looser than the true optimum, by under one pitch per contact.  Reported
bounding boxes are measured from the exact mesh vertices, not from the
voxels, so that looseness never inflates the reported answer.

The earlier approach here sampled the surface and dilated the result by a
voxel to cover what sampling might have missed.  That is also sound, but
the blanket dilation inflated every part by a full voxel in all
directions -- 2 to 3 times the true volume at a working pitch -- which
wastes far more space than the discretisation itself.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Pose:
    """One part in one orientation, discretised.

    mask       conservative occupancy on a pose-local voxel lattice
    pad        voxels of padding between the lattice origin and the exact
               mesh AABB corner; voxel index ``pad`` sits at pose-local 0
    rotation   3x3 rotation applied to the original mesh
    shift      translation that brings the rotated mesh AABB corner to 0
    extents    exact AABB extents of the rotated mesh (model units)
    """

    rotation: np.ndarray
    mask: np.ndarray
    pad: int
    shift: np.ndarray
    extents: np.ndarray
    pitch: float

    @property
    def shape(self) -> np.ndarray:
        return np.asarray(self.mask.shape, dtype=np.int64)

    @property
    def filled(self) -> int:
        return int(self.mask.sum())

    def matrix(self, offset) -> np.ndarray:
        """4x4 world transform for this pose placed at integer voxel offset."""
        m = np.eye(4)
        m[:3, :3] = self.rotation
        m[:3, 3] = self.shift + (np.asarray(offset, dtype=float) + self.pad) * self.pitch
        return m

    def aabb(self, offset):
        """Exact world AABB (min, max) of the mesh at this offset."""
        lo = (np.asarray(offset, dtype=float) + self.pad) * self.pitch
        return lo, lo + self.extents


def _tri_box_overlap(tri, vox):
    """Akenine-Moller SAT test, vectorised over (triangle, voxel) pairs.

    ``tri`` is (P, 3, 3) in lattice units (one voxel = one unit) and ``vox``
    is (P, 3) integer voxel corners, so each box has centre vox + 0.5 and
    half-extent 0.5 on every axis.
    """
    h = 0.5
    centre = vox.astype(np.float64) + 0.5
    u = tri - centre[:, None, :]
    u0, u1, u2 = u[:, 0], u[:, 1], u[:, 2]

    # 3 box-face axes: the triangle's own AABB against the voxel.
    tmin = np.minimum(np.minimum(u0, u1), u2)
    tmax = np.maximum(np.maximum(u0, u1), u2)
    ok = ~np.any((tmin > h) | (tmax < -h), axis=1)
    if not ok.any():
        return ok

    edges = (u1 - u0, u2 - u1, u0 - u2)

    # 9 edge-cross axes.  For box axis e_i the cross product with edge f has
    # a closed form, so no explicit cross product is needed.
    for f in edges:
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            # a = e_i x f  ->  a[j] = -f[k], a[k] = f[j], a[i] = 0
            aj, ak = -f[:, k], f[:, j]
            p0 = u0[:, j] * aj + u0[:, k] * ak
            p1 = u1[:, j] * aj + u1[:, k] * ak
            p2 = u2[:, j] * aj + u2[:, k] * ak
            r = h * (np.abs(aj) + np.abs(ak))
            pmin = np.minimum(np.minimum(p0, p1), p2)
            pmax = np.maximum(np.maximum(p0, p1), p2)
            ok &= ~((pmin > r + 1e-12) | (pmax < -r - 1e-12))
            if not ok.any():
                return ok

    # 1 triangle-plane axis.
    n = np.cross(edges[0], edges[1])
    d = np.einsum("ij,ij->i", n, u0)
    r = h * np.abs(n).sum(axis=1)
    ok &= np.abs(d) <= r + 1e-12
    return ok


def _surface_voxels(vertices, faces, dims, chunk=1 << 21):
    """Every voxel whose cube is genuinely intersected by the surface."""
    grid = np.zeros(tuple(dims), dtype=bool)
    tris = vertices[faces]                       # (T, 3, 3), lattice units
    lo = np.floor(tris.min(axis=1)).astype(np.int64)
    hi = np.floor(tris.max(axis=1)).astype(np.int64)
    np.clip(lo, 0, dims - 1, out=lo)
    np.clip(hi, 0, dims - 1, out=hi)

    span = hi - lo + 1                           # voxels per axis per triangle
    counts = span.prod(axis=1)
    total = int(counts.sum())
    if total == 0:
        return grid

    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    for begin in range(0, total, chunk):
        end = min(begin + chunk, total)
        flat = np.arange(begin, end, dtype=np.int64)
        ti = np.searchsorted(starts, flat, side="right") - 1
        local = flat - starts[ti]

        sx, sy, sz = span[ti, 0], span[ti, 1], span[ti, 2]
        kz = local % sz
        rest = local // sz
        ky = rest % sy
        kx = rest // sy

        vox = lo[ti] + np.stack([kx, ky, kz], axis=1)
        hit = _tri_box_overlap(tris[ti], vox)
        if hit.any():
            v = vox[hit]
            grid[v[:, 0], v[:, 1], v[:, 2]] = True

    return grid


def voxelize_pose(mesh, rotation, pitch, safety=0, fill=True):
    """Discretise ``mesh`` rotated by ``rotation`` onto a pitch-sized lattice.

    ``safety`` adds extra dilation on top of the exact result.  It is not
    needed for correctness and defaults to none; raise it only to force a
    deliberate extra gap measured in whole voxels.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64) @ np.asarray(rotation).T
    lo = verts.min(axis=0)
    extents = verts.max(axis=0) - lo

    pad = int(safety) + 1
    dims = np.ceil(extents / pitch).astype(np.int64) + 1 + 2 * pad

    lattice_verts = (verts - lo) / pitch + pad
    grid = _surface_voxels(lattice_verts, np.asarray(mesh.faces, dtype=np.int64),
                           dims)

    if safety > 0:
        grid = ndimage.binary_dilation(grid, np.ones((3, 3, 3), bool),
                                       iterations=int(safety))
    if fill:
        # Fills enclosed cavities only.  A through-bore stays open, because
        # it is connected to the exterior -- so another part may still nest
        # inside it.
        grid = ndimage.binary_fill_holes(grid)

    return Pose(rotation=np.asarray(rotation, dtype=float), mask=grid, pad=pad,
                shift=-lo, extents=extents, pitch=float(pitch))


def dilate(mask, voxels):
    """Grow a mask by ``voxels`` in the Chebyshev metric (for clearance)."""
    if voxels <= 0:
        return mask
    out = np.pad(mask, voxels, mode="constant", constant_values=False)
    return ndimage.binary_dilation(out, np.ones((3, 3, 3), bool), iterations=int(voxels))


def mask_key(pose) -> str:
    h = hashlib.blake2b(np.packbits(pose.mask).tobytes(), digest_size=12)
    h.update(np.asarray(pose.mask.shape, dtype=np.int64).tobytes())
    return h.hexdigest()


def build_poses(mesh, rotations, pitch, safety=0, clearance_voxels=0,
                dedup=True, progress=None):
    """Voxelize every candidate rotation, dropping exact duplicates.

    Exact duplicates are what a part's own symmetry produces -- a cube has
    24 candidate rotations but only one distinct mask.  Removing them here
    shrinks the search space with no loss.
    """
    poses = []
    seen = set()
    for i, r in enumerate(rotations):
        pose = voxelize_pose(mesh, r, pitch, safety=safety)
        if clearance_voxels > 0:
            pose.mask = dilate(pose.mask, clearance_voxels)
            pose.pad += clearance_voxels
        if dedup:
            key = mask_key(pose)
            if key in seen:
                continue
            seen.add(key)
        poses.append(pose)
        if progress is not None:
            progress(i + 1, len(rotations))
    return poses


def exact_overlap(pose_a, off_a, pose_b, off_b) -> int:
    """Number of voxels shared by two placed poses (0 means disjoint)."""
    off_a = np.asarray(off_a, dtype=np.int64)
    off_b = np.asarray(off_b, dtype=np.int64)
    lo = np.maximum(off_a, off_b)
    hi = np.minimum(off_a + pose_a.shape, off_b + pose_b.shape)
    if np.any(hi <= lo):
        return 0
    sa = pose_a.mask[lo[0] - off_a[0]:hi[0] - off_a[0],
                     lo[1] - off_a[1]:hi[1] - off_a[1],
                     lo[2] - off_a[2]:hi[2] - off_a[2]]
    sb = pose_b.mask[lo[0] - off_b[0]:hi[0] - off_b[0],
                     lo[1] - off_b[1]:hi[1] - off_b[1],
                     lo[2] - off_b[2]:hi[2] - off_b[2]]
    return int(np.count_nonzero(sa & sb))


def suggest_pitch(meshes, resolution=28):
    """Pitch giving roughly ``resolution`` voxels across the largest part."""
    span = max(float(np.max(m.extents)) for m in meshes)
    return span / float(resolution)
