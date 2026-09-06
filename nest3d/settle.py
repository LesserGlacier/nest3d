"""Shake the finished box down: a local settle on a much finer lattice.

Why this exists
---------------
The search runs at a coarse pitch because that is what makes thousands of
arrangements affordable, and at that pitch each part's collision mask is
about 1.35x its true volume.  The extra is a conservative skin, and where
two parts meet, both skins are between them: every contact in a finished
pack is loose by up to a voxel, and the part positions themselves are
quantised to that same lattice.  The result is an arrangement that would
visibly condense if you could pick the box up and shake it.

The search cannot fix this, for two separate reasons.  It only ever moves
whole arrangements -- a placement is a deterministic function of the
placement order and the poses, so there is no move in its search space
that nudges one part two millimetres -- and re-running the whole search at
a fine pitch costs roughly the cube of the resolution ratio.

So do the cheap half only.  Keep the arrangement the search found, keep
every orientation, re-voxelise at a much finer pitch, and let each part
slide to the best nearby position on that finer lattice.  There is no
combinatorial search here at all: it is a handful of correlations per
part, seconds rather than minutes, and it recovers 5 to 7 per cent of the
box on the sample parts -- more than the fine-pitch refinement pass buys
for twenty times the time.

Guarantees
----------
The settle inherits the packer's no-overlap guarantee unchanged: the fine
masks are conservative supersets of the true solids in exactly the same
way, and a part is only ever moved to an offset whose mask overlap with
the rest of the pile is zero.

It is also monotone.  Each part is lifted out and re-seated by minimising
the objective over a window centred on where it already is, so the
position it is standing in is always one of the candidates -- the box can
only shrink or stay the same, never grow.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import ndimage

from .pack import Packer, Placement, _correlate, _score_field
from .voxel import build_poses, suggest_pitch

# Offsets searched around a part's current seat, in fine voxels, on top of
# the coarse pitch it may have to travel.  Three is enough to cross the
# conservative skin on both parts at any sane resolution.
EXTRA_WINDOW = 3

# Candidate offsets scored for contact when several tie on the objective.
CONTACT_POOL = 256


def _measure(objective, ext):
    """The quantity being minimised, with volume as the tie-break.

    Under --objective height the stack height is what counts, but a
    smaller footprint at the same height is still an improvement, so
    volume decides the ties.
    """
    ext = np.asarray(ext, dtype=float)
    vol = float(np.prod(ext))
    return ((float(ext[2]), vol) if objective == "height" else (vol, vol))


def _bbox(poses, offsets, skip=None):
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for i, off in offsets.items():
        if i == skip:
            continue
        a, b = poses[i].aabb(off)
        lo = np.minimum(lo, a)
        hi = np.maximum(hi, b)
    return lo, hi


class _Pile:
    """The occupancy lattice, with parts painted in and out individually."""

    def __init__(self, shape):
        self.grid = np.zeros(tuple(shape), dtype=bool)
        self.shape = np.asarray(shape, dtype=np.int64)

    def paint(self, pose, off, on=True):
        s = pose.shape
        region = self.grid[off[0]:off[0] + s[0], off[1]:off[1] + s[1],
                           off[2]:off[2] + s[2]]
        if on:
            region |= pose.mask
        else:
            region &= ~pose.mask

    def free_offsets(self, pose, lo, hi):
        """Boolean field over the offset box [lo, hi] (inclusive)."""
        s = pose.shape
        sub = self.grid[lo[0]:hi[0] + s[0], lo[1]:hi[1] + s[1],
                        lo[2]:hi[2] + s[2]]
        return _correlate(sub, pose.mask) < 0.5

    def contact(self, dilated, off):
        s = np.asarray(dilated.shape, dtype=np.int64)
        sub = self.grid[off[0]:off[0] + s[0], off[1]:off[1] + s[1],
                        off[2]:off[2] + s[2]]
        if sub.shape != dilated.shape:
            return 0
        return int(np.count_nonzero(sub & dilated))


def _fine_poses(meshes, packing, packer, pitch, clearance_voxels):
    """One pose per placed part: the orientation the search chose, refined."""
    out = {}
    for pl in packing.placements:
        coarse = packer.poses[pl.part_index][pl.pose_index]
        out[pl.part_index] = build_poses(
            meshes[pl.part_index], [coarse.rotation], pitch, dedup=False,
            clearance_voxels=clearance_voxels)[0]
    return out


def _transfer(packing, packer, fine, pile, margin):
    """Re-seat the arrangement on the fine lattice, as close to home as it can.

    Rounding a position onto a finer lattice moves it by up to half a fine
    voxel, so two parts that were touching can round into each other.  Rather
    than assume that away, each part takes the free offset nearest its
    rounded target; if some part has none, the caller keeps the arrangement
    it already had.
    """
    offsets = {}
    for pl in packing.placements:
        coarse = packer.poses[pl.part_index][pl.pose_index]
        pose = fine[pl.part_index]
        world = coarse.shift + (np.asarray(pl.offset, float) + coarse.pad) * packer.pitch
        target = np.round((world - pose.shift) / pose.pitch - pose.pad)
        target = target.astype(np.int64) + margin

        win = 2
        lo = np.maximum(target - win, 0)
        hi = np.minimum(target + win, pile.shape - pose.shape)
        if np.any(hi < lo):
            return None
        free = pile.free_offsets(pose, lo, hi)
        if not free.any():
            return None
        cand = np.argwhere(free) + lo
        # Nearest to the rounded target, so the arrangement is preserved.
        dist = np.abs(cand - target).sum(axis=1) * 8 + np.abs(cand - target).max(axis=1)
        off = cand[int(np.argmin(dist))]
        offsets[pl.part_index] = off
        pile.paint(pose, off)
    return offsets


def _reseat(pile, fine, offsets, part, window, objective, container, dilated):
    """Best offset for one part, over a window centred on where it is now."""
    pose = fine[part]
    pad = pose.pad
    ext_v = pose.extents / pose.pitch
    here = offsets[part]

    lo = np.maximum(here - window, 0)
    hi = np.minimum(here + window, pile.shape - pose.shape)
    if np.any(hi < lo):
        return here
    free = pile.free_offsets(pose, lo, hi)
    if not free.any():
        return here

    keep_lo, keep_hi = _bbox(fine, offsets, skip=part)
    spans = []
    for a in range(3):
        offs = np.arange(lo[a], hi[a] + 1, dtype=float)
        l = np.minimum(keep_lo[a] / pose.pitch, offs + pad)
        h = np.maximum(keep_hi[a] / pose.pitch, offs + pad + ext_v[a])
        spans.append((h - l) * pose.pitch)

    score = np.where(free, _score_field(objective, spans, container), np.inf)
    best = float(score.min())
    if not np.isfinite(best):
        return here

    cand = np.argwhere(score <= best * (1 + 1e-12)) + lo
    if len(cand) == 1:
        return cand[0]
    # Among positions that leave the same box, take the one hugging the
    # pile hardest.  That is what makes this a settle rather than a
    # re-shuffle: it consolidates the slack instead of moving it around.
    cand = cand[:CONTACT_POOL]
    touch = np.array([pile.contact(dilated, c) for c in cand])
    tied = np.argwhere(touch == touch.max()).ravel()
    pick = cand[tied[int(np.argmin(np.abs(cand[tied] - here).sum(axis=1)))]]
    return pick


def settle(meshes, packer, packing, resolution=96, clearance=0.0, sweeps=8,
           budget=None, say=None):
    """Shake a finished arrangement down onto a finer lattice.

    Returns ``(packer, packing, extents)`` for the settled arrangement, or
    ``None`` if it could not improve on the one it was given.
    """
    if len(packing.placements) < 2:
        return None
    if len(packing.placements) != len(packer.poses):
        # A partial packing has no settled arrangement to speak of, and the
        # fine pose list below would not line up with the part list.
        return None

    pitch = min(suggest_pitch(meshes, resolution), packer.pitch)
    clearance_voxels = (int(np.ceil(clearance / (2 * pitch)))
                        if clearance > 0 else 0)
    objective = "height" if packer.objective in ("height", "fit") else "volume"
    container = None
    if packer.container_v is not None:
        container = np.asarray(packer.container_v) * packer.pitch

    t0 = time.time()
    fine = _fine_poses(meshes, packing, packer, pitch, clearance_voxels)

    # Room for every part to move outward as well as in, on both sides.
    window = int(np.ceil(packer.pitch / pitch)) + EXTRA_WINDOW
    margin = np.full(3, window + 2, dtype=np.int64)
    span = np.zeros(3, dtype=np.int64)
    for pl in packing.placements:
        coarse = packer.poses[pl.part_index][pl.pose_index]
        pose = fine[pl.part_index]
        world = coarse.shift + (np.asarray(pl.offset, float) + coarse.pad) * packer.pitch
        top = np.round((world - pose.shift) / pitch - pose.pad) + pose.shape
        span = np.maximum(span, top.astype(np.int64))
    pile = _Pile(span + 2 * margin)

    offsets = _transfer(packing, packer, fine, pile, margin)
    if offsets is None:
        if say is not None:
            say("  settle      could not transfer to the fine lattice, skipped")
        return None

    lo, hi = _bbox(fine, offsets)
    start = _measure(objective, hi - lo)
    dilated = {i: ndimage.binary_dilation(p.mask, np.ones((3, 3, 3), bool))
               for i, p in fine.items()}
    deadline = None if budget is None else t0 + budget

    for _ in range(sweeps):
        moved = 0
        lo, hi = _bbox(fine, offsets)

        def on_face(i):
            a, b = fine[i].aabb(offsets[i])
            return -int(np.count_nonzero((a <= lo + 1e-6) | (b >= hi - 1e-6)))

        # Parts on the box's faces first: they are the only ones that can
        # shrink it, and freeing space behind them lets the rest follow.
        for part in sorted(offsets, key=lambda i: (on_face(i), i)):
            if deadline is not None and time.time() > deadline:
                break
            pose = fine[part]
            pile.paint(pose, offsets[part], on=False)
            new = _reseat(pile, fine, offsets, part, window, objective,
                          container, dilated[part])
            if not np.array_equal(new, offsets[part]):
                moved += 1
                offsets[part] = new
            pile.paint(pose, offsets[part])
        if moved == 0 or (deadline is not None and time.time() > deadline):
            break

    lo, hi = _bbox(fine, offsets)
    ext = hi - lo
    vol = float(np.prod(ext))
    now = _measure(objective, ext)
    # Judged on the objective that was actually being minimised: under
    # --objective height a shorter stack wins even if the footprint grows
    # inside its container, and a smaller footprint at the same height is
    # still worth having.
    if now >= start:
        if say is not None:
            say("  settle      no improvement (pitch %.2f mm, %.1fs)"
                % (pitch, time.time() - t0))
        return None

    out_packer = Packer([[fine[i]] for i in range(len(packer.poses))],
                        objective=packer.objective,
                        contact_weight=packer.contact_weight)
    out = out_packer.finalise(
        [Placement(i, 0, np.asarray(offsets[i], dtype=np.int64))
         for i in sorted(offsets)])

    if say is not None:
        gain = 100 * (1 - now[0] / start[0]) if start[0] else 0.0
        what, shown = ("height" if objective == "height" else "vol"), now[0]
        if gain <= 1e-9 and start[1]:
            # The objective tied and volume broke the tie; report that.
            what, shown, gain = "vol", vol, 100 * (1 - now[1] / start[1])
        say("  settle      %s mm   %s %.4g   (-%.1f%%, pitch %.2f mm, %.1fs)"
            % (" x ".join("%7.1f" % v for v in ext), what, shown, gain,
               pitch, time.time() - t0))
    return out_packer, out, ext
