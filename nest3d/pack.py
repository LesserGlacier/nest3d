"""Constructive placement: put parts down one at a time, optimally each time.

For each part the packer evaluates *every* integer voxel offset in the
relevant region at once, using an FFT cross-correlation between the pile's
occupancy and the part's mask.  Offsets whose correlation is zero are
exactly the collision-free ones.  Those are then scored by the objective
(how much the bounding box would grow) and the best is taken.

This is a greedy step, but it is a greedy step over the whole continuum of
positions rather than over a handful of corner points, which is what makes
it work for concave parts: if a part can drop into another part's hollow,
the correlation finds that position without anyone having to nominate it
as a candidate.  Global quality then comes from searching the placement
order and orientation choices on top (search.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, signal

from .voxel import exact_overlap

OBJECTIVES = ("volume", "compact", "maxdim", "height", "fit")

# Weight on the cube-penalty term of the "compact" objective.  Pure minimum
# volume is a trap for a greedy placer: extending the box along its longest
# axis costs the smallest cross-section, so the cheapest step is always to
# make the rod longer, and the pack degenerates into a chain.  Charging for
# the longest dimension as well removes that bias.
COMPACT_WEIGHT = 0.35


@dataclass
class Placement:
    part_index: int
    pose_index: int
    offset: np.ndarray


@dataclass
class Packing:
    placements: list = field(default_factory=list)
    feasible: bool = True
    failed_part: int | None = None
    n_unplaced: int = 0
    # Exact extents measured from mesh AABBs, in model units.
    bbox_lo: np.ndarray | None = None
    bbox_hi: np.ndarray | None = None
    score: float = float("inf")

    @property
    def extents(self):
        if self.bbox_lo is None:
            return None
        return self.bbox_hi - self.bbox_lo

    @property
    def volume(self) -> float:
        e = self.extents
        return float("inf") if e is None else float(np.prod(e))


def _correlate(pile_sub, mask):
    """Overlap count for every offset of ``mask`` inside ``pile_sub``."""
    a = pile_sub.astype(np.float32)
    b = mask[::-1, ::-1, ::-1].astype(np.float32)
    out = signal.fftconvolve(a, b, mode="valid")
    return out


def _axis_span(cur_lo, cur_hi, offs, pad, ext_v):
    """Bounding-box span along one axis, for every offset on that axis."""
    lo = np.minimum(cur_lo, offs + pad)
    hi = np.maximum(cur_hi, offs + pad + ext_v)
    return hi - lo, lo, hi


def _score_field(objective, spans, container=None):
    """Combine per-axis spans into a score array over the offset grid."""
    dx, dy, dz = spans
    if objective == "volume":
        score = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    elif objective == "compact":
        vol = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
        longest = np.maximum(np.maximum(dx[:, None, None], dy[None, :, None]),
                             dz[None, None, :])
        score = vol + COMPACT_WEIGHT * longest ** 3
    elif objective == "maxdim":
        score = np.maximum(np.maximum(dx[:, None, None], dy[None, :, None]),
                           dz[None, None, :])
    elif objective in ("height", "fit"):
        # Height packs into a fixed footprint; fit into a fixed box.  Both
        # score by the vertical span, with the constrained axes masked off.
        score = np.broadcast_to(dz[None, None, :],
                                (dx.size, dy.size, dz.size)).copy()
    else:
        raise ValueError("unknown objective %r" % objective)

    if container is not None:
        bad = np.zeros_like(score, dtype=bool)
        if np.isfinite(container[0]):
            bad |= (dx > container[0] + 1e-9)[:, None, None]
        if np.isfinite(container[1]):
            bad |= (dy > container[1] + 1e-9)[None, :, None]
        if np.isfinite(container[2]):
            bad |= (dz > container[2] + 1e-9)[None, None, :]
        score = np.where(bad, np.inf, score)
    return score


class Packer:
    """Places a sequence of (part, pose) pairs onto a shared voxel lattice."""

    def __init__(self, poses_per_part, objective="volume", container=None,
                 contact_weight=0.0, contact_pool=192, lattice_scale=2.4,
                 verify=True):
        self.poses = poses_per_part
        self.objective = objective
        self.pitch = poses_per_part[0][0].pitch
        self.verify = verify
        self.contact_weight = float(contact_weight)
        self.contact_pool = int(contact_pool)

        if container is None:
            self.container_v = None
        else:
            c = np.asarray(container, dtype=float) / self.pitch
            self.container_v = np.where(np.isfinite(c), c, np.inf)

        self.max_shape = np.max(
            [p.shape for poses in poses_per_part for p in poses], axis=0)
        self.lattice = self._lattice_shape(lattice_scale)
        self._dilated_cache = {}

    def _lattice_shape(self, scale):
        """A working lattice comfortably larger than any sane packing."""
        if self.container_v is not None and np.all(np.isfinite(self.container_v)):
            # Sized to the container plus room for a part to overhang the
            # search region; growing it later would be pointless, since a
            # container failure is a real failure, not a lattice shortage.
            n = np.ceil(self.container_v).astype(np.int64) + self.max_shape + 2
            return n

        total = 0.0
        for poses in self.poses:
            e = poses[0].extents / self.pitch
            total += float(np.prod(e))
        side = total ** (1.0 / 3.0)
        n = int(np.ceil(scale * side)) + int(self.max_shape.max())
        n = max(n, int(3 * self.max_shape.max()))
        return np.array([n, n, n], dtype=np.int64)

    @property
    def bounded(self) -> bool:
        return self.container_v is not None and np.all(np.isfinite(self.container_v))

    # ------------------------------------------------------------------
    def pack(self, order, pose_choice):
        """Place parts in ``order`` using ``pose_choice[part] = pose index``."""
        lattice = self.lattice.copy()
        while True:
            result = self._pack_on(lattice, order, pose_choice)
            if result.feasible or self.bounded:
                # Inside a fixed container there is nothing to grow into:
                # a failure there means the parts genuinely do not fit.
                return result
            lattice = (lattice * 1.5).astype(np.int64)
            if lattice.max() > 900:
                return result

    def _pack_on(self, lattice, order, pose_choice):
        pile = np.zeros(tuple(lattice), dtype=bool)
        placed = []
        cur_lo = None
        cur_hi = None

        for step, part_i in enumerate(order):
            pose = self.poses[part_i][pose_choice[part_i]]
            s = pose.shape
            ext_v = pose.extents / self.pitch

            if step == 0:
                off = (lattice // 2 - s // 2).astype(np.int64)
                cur_lo = off + pose.pad
                cur_hi = cur_lo + ext_v
            else:
                off = self._best_offset(pile, pose, cur_lo, cur_hi, lattice, placed)
                if off is None:
                    # Report how far we got rather than giving up flat: the
                    # annealer needs a gradient to climb out of an
                    # over-tight container, and "eight of nine placed" is a
                    # far more useful signal than a bare failure.
                    return self.finalise(placed, unplaced=len(order) - step,
                                         failed=part_i)
                cur_lo = np.minimum(cur_lo, off + pose.pad)
                cur_hi = np.maximum(cur_hi, off + pose.pad + ext_v)

            pile[off[0]:off[0] + s[0], off[1]:off[1] + s[1],
                 off[2]:off[2] + s[2]] |= pose.mask
            placed.append(Placement(part_i, pose_choice[part_i], off))

        return self.finalise(placed)

    # ------------------------------------------------------------------
    def _best_offset(self, pile, pose, cur_lo, cur_hi, lattice, placed):
        s = pose.shape
        pad = pose.pad
        ext_v = pose.extents / self.pitch

        # An optimal placement always touches the current pile, so offsets
        # need only range from "part's far corner at the pile's near edge"
        # to "part's near corner at the pile's far edge".
        pile_lo = np.floor(cur_lo).astype(np.int64) - pad
        pile_hi = np.ceil(cur_hi).astype(np.int64) - pad
        lo = np.maximum(pile_lo - s + 1, 0)
        hi = np.minimum(pile_hi + 1, lattice - s)
        if np.any(hi < lo):
            return None

        sub = pile[lo[0]:hi[0] + s[0], lo[1]:hi[1] + s[1], lo[2]:hi[2] + s[2]]
        overlap = _correlate(sub, pose.mask)
        free = overlap < 0.5
        if not free.any():
            return None

        offs = [np.arange(lo[a], hi[a] + 1, dtype=np.int64) for a in range(3)]
        spans = []
        for a in range(3):
            d, _, _ = _axis_span(cur_lo[a], cur_hi[a], offs[a].astype(float),
                                 pad, ext_v[a])
            spans.append(d)

        score = _score_field(self.objective, spans, self.container_v)
        score = np.where(free, score, np.inf)
        if not np.isfinite(score).any():
            return None

        best = float(score.min())
        # Break ties towards the lattice origin: among equally good boxes
        # this is the classic bottom-left rule, and it keeps the pile from
        # drifting into a staircase.
        tol = max(abs(best) * 1e-9, 1e-9)
        cand = np.argwhere(score <= best + tol)

        if self.contact_weight > 0 and len(cand) > 1:
            cand = self._rank_by_contact(pile, pose, cand, lo)

        keys = cand.sum(axis=1) * 1e6 + cand[:, 2] * 1e3 + cand[:, 1]
        pick = cand[int(np.argmin(keys))]
        off = lo + pick

        if self.verify:
            for p in placed:
                other = self.poses[p.part_index][p.pose_index]
                if exact_overlap(pose, off, other, p.offset):
                    return self._fallback_offset(pile, pose, lo, hi, score, placed)
        return off

    def _fallback_offset(self, pile, pose, lo, hi, score, placed):
        """Exhaustive exact search, used only if the FFT test disagrees."""
        order = np.argsort(score, axis=None)
        for flat in order[: min(4000, order.size)]:
            if not np.isfinite(score.flat[flat]):
                break
            idx = np.array(np.unravel_index(flat, score.shape), dtype=np.int64)
            off = lo + idx
            ok = True
            for p in placed:
                other = self.poses[p.part_index][p.pose_index]
                if exact_overlap(pose, off, other, p.offset):
                    ok = False
                    break
            if ok:
                return off
        return None

    def _rank_by_contact(self, pile, pose, cand, lo):
        """Among tied offsets, prefer the one hugging the pile most closely."""
        cand = cand[: self.contact_pool]
        key = id(pose)
        dil = self._dilated_cache.get(key)
        if dil is None:
            dil = ndimage.binary_dilation(pose.mask, np.ones((3, 3, 3), bool))
            self._dilated_cache[key] = dil
        s = np.asarray(dil.shape, dtype=np.int64)

        scores = np.empty(len(cand), dtype=np.int64)
        for i, c in enumerate(cand):
            o = lo + c
            sub = pile[o[0]:o[0] + s[0], o[1]:o[1] + s[1], o[2]:o[2] + s[2]]
            if sub.shape != dil.shape:
                scores[i] = 0
            else:
                scores[i] = -int(np.count_nonzero(sub & dil))
        return cand[np.argsort(scores, kind="stable")]

    # ------------------------------------------------------------------
    def finalise(self, placed, unplaced=0, failed=None):
        """Package placements as a Packing, measured from exact mesh AABBs.

        Public because the settle builds its arrangement by moving parts
        rather than by placing them, and still has to be measured and
        scored the same way.
        """
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for p in placed:
            pose = self.poses[p.part_index][p.pose_index]
            a, b = pose.aabb(p.offset)
            lo = np.minimum(lo, a)
            hi = np.maximum(hi, b)
        if not placed:
            lo = np.zeros(3)
            hi = np.zeros(3)

        pack = Packing(placements=placed, feasible=(unplaced == 0),
                       failed_part=failed, n_unplaced=unplaced,
                       bbox_lo=lo, bbox_hi=hi)
        ext = hi - lo
        if self.objective == "volume":
            score = float(np.prod(ext))
        elif self.objective == "compact":
            score = float(np.prod(ext) + COMPACT_WEIGHT * np.max(ext) ** 3)
        elif self.objective == "maxdim":
            score = float(np.max(ext))
        elif self.objective in ("height", "fit"):
            score = float(ext[2])
        else:
            score = float(np.prod(ext))

        if unplaced:
            # Every unplaced part costs more than any arrangement of the
            # placed ones ever could, so a feasible solution always beats an
            # infeasible one, while fewer leftovers still reads as progress.
            score = self._penalty * unplaced + score
        pack.score = score
        return pack

    @property
    def _penalty(self) -> float:
        if self.container_v is not None and np.all(np.isfinite(self.container_v)):
            span = float(np.prod(self.container_v)) * self.pitch ** 3
        else:
            span = float(np.prod(self.lattice)) * self.pitch ** 3
        return 4.0 * max(span, 1.0)


def verify_packing(packer, packing, tolerance=0):
    """Re-check every pair for overlap.  Returns a list of offending pairs."""
    bad = []
    ps = packing.placements
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            a = packer.poses[ps[i].part_index][ps[i].pose_index]
            b = packer.poses[ps[j].part_index][ps[j].pose_index]
            n = exact_overlap(a, ps[i].offset, b, ps[j].offset)
            if n > tolerance:
                bad.append((ps[i].part_index, ps[j].part_index, n))
    return bad
