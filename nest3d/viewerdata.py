"""Turn parts plus a recorded search into the payload a viewer needs.

The whole reason a few thousand arrangements fit in one page: geometry is
shipped once and every frame is just one 4x4 per part.  A frame costs
about 150 numbers regardless of how detailed the parts are.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import ndimage

from .trace import frame_matrices


def _geometry(mesh, round_to=2):
    """Indexed positions for one part, centred on its own AABB corner."""
    v = np.asarray(mesh.vertices, dtype=float)
    f = np.asarray(mesh.faces, dtype=np.int64)
    return {
        "positions": [round(float(x), round_to) for x in v.reshape(-1)],
        "indices": [int(i) for i in f.reshape(-1)],
        "vertex_count": int(len(v)),
        "triangle_count": int(len(f)),
    }


def voxel_shell(pose, mesh):
    """The collision mask's outer skin, in the part's own mesh coordinates.

    What this is for: the mask is a conservative *superset* of the solid --
    a voxel is filled when any triangle touches it -- and that skin is the
    whole reason a pack is looser than the parts really are.  Drawing it
    next to the mesh is the only way to see how much of a box is slack the
    algorithm could not have known was there.

    Only the skin is shipped, not the solid interior: a filled mask is
    twenty times the voxels and every one of them is hidden behind the
    ones sent here.

    Voxel indices go out as integers rather than positions.  The viewer
    rebuilds a centre as ``rot . ((idx - pad + 0.5) * pitch - shift)``,
    which is the pose-local lattice mapped back through the pose's own
    rotation, so the boxes land in the same coordinates as the geometry
    and ride the same per-frame matrix.
    """
    shell = pose.mask & ~ndimage.binary_erosion(pose.mask)
    idx = np.argwhere(shell).astype(np.int32)
    return {
        "pitch": round(float(pose.pitch), 4),
        "pad": int(pose.pad),
        "shift": [round(float(v), 4) for v in pose.shift],
        # Transposed on the way out: the viewer needs mask -> mesh, and the
        # pose stores mesh -> mask.
        "rot": [round(float(v), 6) for v in np.asarray(pose.rotation).T.reshape(-1)],
        "idx": [int(v) for v in idx.reshape(-1)],
        "shell": int(len(idx)),
        "filled": int(pose.filled),
        "ratio": round(float(pose.filled * pose.pitch ** 3 / abs(mesh.volume)), 3),
    }


def _frame_payload(frame, poses, n_parts, rot_dp=6, pos_dp=2):
    """Rotation and translation need different precision.

    A translation is in millimetres, so 2 decimals is far finer than the
    voxel pitch.  A rotation element is in [-1, 1], where 2 decimals leaves
    up to 0.005 of error -- about 2 mm of swing at the far corner of a
    380 mm part, which is enough to make parts visibly interpenetrate in
    the viewer even though the packing itself is sound.
    """
    mats, order = frame_matrices(frame, poses, n_parts)
    out = []
    for m in mats:
        if m is None:
            out.append(None)
            continue
        row = []
        for r in range(3):
            row.extend(round(float(m[r, c]), rot_dp) for c in range(3))
            row.append(round(float(m[r, 3]), pos_dp))
        out.append(row)
    return {
        "m": out,                       # 3x4 row-major per part, null if unplaced
        "order": [int(i) for i in order],
        "ext": [round(float(v), 2) for v in frame.extents],
        "lo": [round(float(v), 2) for v in frame.lo],
        "vol": float(frame.volume),
        "phase": frame.phase,
        "best": bool(frame.is_best),
        "t": round(float(frame.t), 2),
    }


def build(parts, poses, recorder, part_volume, best_packing=None,
          max_search_frames=400):
    """Assemble the viewer payload.

    Three tracks come out of the same recording:
      best     only the arrangements that improved on all before them
      search   the annealer's actual walk, backward steps included
      build    the winning arrangement assembled one part at a time
    """
    n = len(parts)
    meshes = [p.mesh for p in parts]

    best_frames = recorder.best_frames()
    search_frames = recorder.accepted_frames(limit=max_search_frames)

    payload = {
        "parts": [
            {
                "name": p.name,
                "volume": float(abs(p.mesh.volume)),
                "extents": [round(float(v), 2) for v in p.mesh.extents],
                "geometry": _geometry(p.mesh),
            }
            for p in parts
        ],
        "part_volume": float(part_volume),
        "tracks": {
            "best": [_frame_payload(f, poses, n) for f in best_frames],
            "search": [_frame_payload(f, poses, n) for f in search_frames],
        },
        "stats": {
            "evaluated": len(recorder.frames),
            "dropped": int(recorder.dropped),
            "improvements": len(best_frames),
            "accepted": sum(1 for f in recorder.frames if f.accepted),
        },
    }

    # The build track needs no extra frames: it is the final arrangement
    # revealed in placement order, so the viewer just draws the first k
    # parts of the last best frame.
    if best_frames:
        payload["tracks"]["build_of"] = payload["tracks"]["best"][-1]
    return payload


def write(payload, path):
    with open(path, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    return path
