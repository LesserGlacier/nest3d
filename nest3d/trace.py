"""Recording the search so it can be watched afterwards.

Every evaluation the packer does is a complete, valid arrangement -- all
parts placed, oriented and non-overlapping.  The search throws away all
but the running best, which is right for solving and useless for seeing
what happened.  This records them.

What gets stored is deliberately tiny: for each frame, one
(part, pose, offset) triple per part.  The offset is an integer voxel
coordinate and the pose index selects a rotation that already exists, so
a frame is a few dozen bytes and the geometry is never duplicated.  The
4x4 transforms are reconstructed at export time from the poses that are
still in memory, which is also why nothing needs re-packing to replay.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    phase: str
    placements: tuple          # ((part_index, pose_index, ox, oy, oz), ...)
    volume: float
    extents: tuple
    lo: tuple
    accepted: bool
    is_best: bool
    t: float


@dataclass
class Recorder:
    """Collects frames during a search.  Cheap enough to leave switched on."""

    max_frames: int = 20000
    frames: list = field(default_factory=list)
    t0: float = field(default_factory=time.time)
    phase: str = "free"
    dropped: int = 0

    def record(self, packing, accepted=False, is_best=False):
        if not packing.feasible:
            return
        if len(self.frames) >= self.max_frames:
            self.dropped += 1
            return
        placements = tuple(
            (int(p.part_index), int(p.pose_index),
             int(p.offset[0]), int(p.offset[1]), int(p.offset[2]))
            for p in packing.placements
        )
        ext = np.asarray(packing.extents, dtype=float)
        lo = np.asarray(packing.bbox_lo, dtype=float)
        self.frames.append(Frame(
            phase=self.phase, placements=placements,
            volume=float(np.prod(ext)), extents=tuple(float(v) for v in ext),
            lo=tuple(float(v) for v in lo),
            accepted=bool(accepted), is_best=bool(is_best),
            t=time.time() - self.t0,
        ))

    # ------------------------------------------------------------------
    def best_frames(self):
        """Only the arrangements that improved on everything before them."""
        out = []
        best = float("inf")
        for f in self.frames:
            if f.volume < best - 1e-9:
                best = f.volume
                out.append(f)
        return out

    def accepted_frames(self, limit=None):
        """The annealer's actual walk, including the steps it took backwards.

        These are the interesting ones: simulated annealing accepts a worse
        arrangement on purpose so it can climb out of a local optimum, so
        this trace shows it settle, escape and re-settle.  A best-only
        trace hides exactly that.
        """
        out = [f for f in self.frames if f.accepted]
        return _subsample(out, limit)

    def all_frames(self, limit=None):
        return _subsample(list(self.frames), limit)


def _subsample(frames, limit):
    if not limit or len(frames) <= limit:
        return frames
    idx = np.linspace(0, len(frames) - 1, limit).round().astype(int)
    seen = set()
    out = []
    for i in idx:
        if i not in seen:
            seen.add(int(i))
            out.append(frames[int(i)])
    return out


def frame_matrices(frame, poses_per_part, n_parts):
    """Reconstruct one frame's 4x4 transforms, in placement order."""
    mats = [None] * n_parts
    order = []
    for part_i, pose_i, ox, oy, oz in frame.placements:
        pose = poses_per_part[part_i][pose_i]
        mats[part_i] = pose.matrix((ox, oy, oz))
        order.append(part_i)
    return mats, order
