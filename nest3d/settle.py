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

Two things the sweep alone will not do
-------------------------------------
It stops at a one-part-move optimum: a box can be held open by three parts
that each need one of the others to move first, and no single re-seat sees
that.  Tipping the box is the cheap half of an answer -- the objective is
flat while a part stays inside the box, so a part can be slid to one wall
for free, and the slack it leaves behind is somewhere a face part can then
move into.  It is worth about a point of density when it works and nothing
at all when it does not, which is why it is the last thing tried.

And it converges somewhere different on every lattice.  A finer one has a
thinner skin to give back, but the parts also round onto it differently,
so finer is a tendency and not a rule.  Rather than pick, run several and
keep the smallest box -- see LADDER.

Guarantees
----------
The settle inherits the packer's no-overlap guarantee unchanged: the fine
masks are conservative supersets of the true solids in exactly the same
way, and a part is only ever moved to an offset whose mask overlap with
the rest of the pile is zero.  That holds on every lattice tried, each of
which re-earns it independently.

It is also monotone.  Each part is lifted out and re-seated by minimising
the objective over a window centred on where it already is, so the
position it is standing in is always one of the candidates -- the box can
only shrink or stay the same, never grow.  The tipping passes are held to
the same rule: an offset is a candidate only if it leaves the box no worse
than it already is.
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

# Directions the box is tipped in, in order, once the objective sweeps have
# run out of moves.  Faces first, then the corners: tipping along an axis
# consolidates the slack against one wall, and the diagonals are what
# actually break an arch, where three parts hold each other up over a void
# and no single one of them can move on its own.
SHAKE_CYCLES = 2

SHAKE_DIRS = (
    (0, 0, -1), (-1, 0, 0), (0, -1, 0),
    (0, 0, 1), (1, 0, 0), (0, 1, 0),
    (-1, -1, -1), (1, 1, 1),
)

# Tilts tried by the orientation pass, in degrees, about each lattice axis
# and in both directions.  Small on purpose: the search already chose the
# pose, and this pass exists to correct that choice by the amount the
# coarse mask could have got it wrong, not to search SO(3) again.  Two
# magnitudes rather than one because the useful correction is not the same
# size on every part -- a flat plate cares about a degree, a stubby hub
# does not notice five.
ORIENT_ANGLES = (1.5, 4.0)

# Rounds of the orientation pass allowed per settle.  A second round pays
# because the first one's re-seats change what a tilt can reach; a third
# has never moved anything on the sample parts.
ORIENT_ROUNDS = 2


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


def _fields(pile, fine, offsets, part, window, objective, container):
    """Score the free offsets in a window around a part's current seat.

    Returns ``(lo, score, volume)`` over the offset box, with ``inf`` where
    the part would overlap the rest of the pile.  ``volume`` is the box
    volume the same offsets produce, which is what breaks ties under
    ``--objective height`` -- see ``_measure``.
    """
    pose = fine[part]
    pad = pose.pad
    ext_v = pose.extents / pose.pitch
    here = offsets[part]

    lo = np.maximum(here - window, 0)
    hi = np.minimum(here + window, pile.shape - pose.shape)
    if np.any(hi < lo):
        return None
    free = pile.free_offsets(pose, lo, hi)
    if not free.any():
        return None

    keep_lo, keep_hi = _bbox(fine, offsets, skip=part)
    spans = []
    for a in range(3):
        offs = np.arange(lo[a], hi[a] + 1, dtype=float)
        l = np.minimum(keep_lo[a] / pose.pitch, offs + pad)
        h = np.maximum(keep_hi[a] / pose.pitch, offs + pad + ext_v[a])
        spans.append((h - l) * pose.pitch)

    score = np.where(free, _score_field(objective, spans, container), np.inf)
    volume = np.where(free, _score_field("volume", spans, None), np.inf)
    return lo, score, volume


def _hug(pile, dilated, cand, here):
    """Of several equal offsets, the one hugging the pile hardest.

    That is what makes this a settle rather than a re-shuffle: it
    consolidates the slack instead of moving it around.  Distance from the
    part's current seat breaks the remaining ties, so a part that has
    nothing to gain stays where it is.
    """
    if len(cand) == 1:
        return cand[0]
    cand = cand[:CONTACT_POOL]
    touch = np.array([pile.contact(dilated, c) for c in cand])
    tied = np.argwhere(touch == touch.max()).ravel()
    return cand[tied[int(np.argmin(np.abs(cand[tied] - here).sum(axis=1)))]]


def _reseat(pile, fine, offsets, part, window, objective, container, dilated):
    """Best offset for one part, over a window centred on where it is now."""
    here = offsets[part]
    got = _fields(pile, fine, offsets, part, window, objective, container)
    if got is None:
        return here
    lo, score, volume = got

    best = float(score.min())
    if not np.isfinite(best):
        return here

    at = score <= best * (1 + 1e-12)
    # Under --objective height many offsets tie on the stack height, and a
    # smaller footprint at the same height is a real improvement -- but a
    # larger one is a real loss, and _measure would throw the whole settle
    # away for it.  So volume decides here too, exactly as it does there.
    vbest = float(volume[at].min())
    at &= volume <= vbest * (1 + 1e-12)
    return _hug(pile, dilated, np.argwhere(at) + lo, here)


def _shake(pile, fine, offsets, part, window, objective, container, dilated,
           direction):
    """Slide one part as far as it will go along ``direction``, for free.

    The objective is flat inside the box: while a part stays within the
    current bounding box, every position it could take scores the same, so
    the objective sweep has no reason to prefer any of them and stops.  The
    slack it leaves is real all the same, and where it sits decides whether
    the next part can move.

    So tip the box.  Only offsets that leave the box no worse than it is
    now are candidates, which keeps the pass monotone; among those, take
    the one furthest along ``direction``.  Nothing improves on this pass by
    itself -- it moves the voids to one wall so that the objective sweep
    after it has somewhere to shrink into.
    """
    here = offsets[part]
    got = _fields(pile, fine, offsets, part, window, objective, container)
    if got is None:
        return here
    lo, score, volume = got

    at_here = tuple(here - lo)
    now, now_vol = float(score[at_here]), float(volume[at_here])
    if not np.isfinite(now):
        return here

    ok = (score <= now * (1 + 1e-12)) & (volume <= now_vol * (1 + 1e-12))
    cand = np.argwhere(ok) + lo
    if len(cand) <= 1:
        return here

    travel = cand @ np.asarray(direction, dtype=float)
    tied = cand[travel <= float(travel.min()) + 1e-9]
    return _hug(pile, dilated, tied, here)


def _tilt(axis, degrees):
    """Rotation about one lattice axis, as a 3x3 matrix."""
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    m = np.eye(3)
    j, k = (axis + 1) % 3, (axis + 2) % 3
    m[j, j] = c
    m[j, k] = -s
    m[k, j] = s
    m[k, k] = c
    return m


def _orient_rotations(rot):
    """Small tilts of an already-chosen pose, about the lattice axes.

    Applied on the left, so the tilt is about the axes of the lattice the
    part is sitting on rather than about the part's own axes -- the box is
    axis-aligned, and it is the part's silhouette against those axes that
    decides the box.
    """
    for axis in range(3):
        for deg in ORIENT_ANGLES:
            for sign in (1, -1):
                yield _tilt(axis, sign * deg) @ rot


def _seat_for(pose, here, old_pose, pile):
    """Where a re-voxelised pose sits if the part does not move.

    A tilted pose has its own lattice -- different padding, different
    extents -- so "the same place" has to be restated rather than reused.
    Anchoring the AABB centre keeps the part where it stands instead of
    letting it drift toward whichever corner the new pose happens to pad.
    """
    lo = (np.asarray(here, dtype=float) + old_pose.pad) * old_pose.pitch
    centre = lo + old_pose.extents / 2.0
    target = (centre - pose.extents / 2.0) / pose.pitch - pose.pad
    seat = np.round(target).astype(np.int64)
    return np.clip(seat, 0, pile.shape - pose.shape)


def _reorient(pile, fine, offsets, part, window, objective, container,
              meshes, pitch, clearance_voxels):
    """Re-choose one part's orientation on the settle's own lattice.

    Every orientation in a finished pack was chosen by the search, at the
    coarse pitch, where a part's mask runs 1.2 to 2.2 times the solid
    inside it.  At that fatness the mask's *shape* is substantially not
    the part's shape, so the pose that scored best there was ranked on
    geometry that is not quite the part.  Here the masks are within a few
    per cent of the solid, and the ranking can be redone on something much
    closer to the real thing.

    This is deliberately a tilt and not a re-search: the arrangement is
    worth keeping, and a pose far from the current one would land the part
    somewhere the rest of the pile is not expecting it.  Each candidate is
    voxelised at the settle pitch, seated where the part already stands,
    and then given the same windowed re-seat every other pass uses.

    Monotone by the same rule as the rest of the settle: the incumbent
    pose at its current offset is one of the things measured, and a
    candidate is taken only if it makes the box strictly smaller.  Returns
    ``(pose, offset)`` or ``None`` if nothing beat standing still.
    """
    here = offsets[part]
    incumbent = fine[part]

    got = _fields(pile, fine, offsets, part, window, objective, container)
    if got is None:
        # The part has no free seat even where it is, which only happens
        # against the edge of the pile.  Leave it to the sweeps.
        return None
    lo, score, volume = got
    seat_ix = tuple(here - lo)
    best = (float(score[seat_ix]), float(volume[seat_ix]))
    if not np.isfinite(best[0]):
        return None

    found = None
    for rot in _orient_rotations(incumbent.rotation):
        pose = build_poses(meshes[part], [rot], pitch, dedup=False,
                           clearance_voxels=clearance_voxels)[0]
        if np.any(pose.shape >= pile.shape):
            # A tilt grows the part's AABB, and the pile is only sized for
            # the poses it was built with.  Rather than grow the lattice
            # for a candidate that probably loses anyway, drop it.
            continue
        seat = _seat_for(pose, here, incumbent, pile)
        fine[part], offsets[part] = pose, seat
        cand = _fields(pile, fine, offsets, part, window, objective, container)
        if cand is None:
            continue
        c_lo, c_score, c_vol = cand
        m = float(c_score.min())
        if not np.isfinite(m) or m > best[0] * (1 + 1e-12):
            continue
        at = c_score <= m * (1 + 1e-12)
        v = float(c_vol[at].min())
        # Volume decides a tie on the objective, exactly as it does in
        # ``_reseat``: under --objective height a shorter footprint at the
        # same height is worth taking, and a taller one never is.
        if (m, v) >= best:
            continue
        at &= c_vol <= v * (1 + 1e-12)
        best = (m, v)
        cands = np.argwhere(at) + c_lo
        # Of the offsets that tie, the one nearest where the part was
        # seated: a tilt is already a change, and there is no reason to
        # add a translation the objective did not ask for.
        near = int(np.argmin(np.abs(cands - seat).sum(axis=1)))
        found = (pose, cands[near])

    fine[part], offsets[part] = incumbent, here
    return found


def _settle_once(meshes, packer, packing, resolution, clearance, sweeps,
                 deadline, shake, orient=False, on_frame=None):
    """One settle, on one lattice.  The unit of work the ladder runs.

    Returns ``(measure, packer, packing, extents, pitch)`` for the settled
    arrangement, or ``None`` if this lattice could not improve on the
    arrangement it was given.
    """
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
    grow = 0
    if orient:
        # A tilted part's AABB is larger than the upright one's, and the
        # pile has to hold the largest candidate rather than the pose it
        # was built from.  Turning a box of diagonal d by theta cannot add
        # more than d*sin(theta) to any axis, which is the bound used here.
        widest = max(float(np.linalg.norm(m.extents)) for m in meshes)
        grow = int(np.ceil(widest * np.sin(np.deg2rad(max(ORIENT_ANGLES)))
                           / pitch)) + 1
    margin = np.full(3, window + 2 + grow, dtype=np.int64)
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
        # Rounding onto this lattice put two parts through each other and
        # there was no free seat nearby.  Another rung of the ladder will
        # not have the same trouble, so this one just drops out.
        return None

    lo, hi = _bbox(fine, offsets)
    start = _measure(objective, hi - lo)
    if on_frame is not None:
        on_frame(fine, offsets, "transfer")
    if deadline is not None and time.time() > deadline:
        # Voxelising this rung alone ran the clock out; there is no time
        # left to sweep with, and an unswept transfer is never an
        # improvement on what came in.
        return None
    dilated = {i: ndimage.binary_dilation(p.mask, np.ones((3, 3, 3), bool))
               for i, p in fine.items()}

    def out_of_time():
        return deadline is not None and time.time() > deadline

    def sweep(pick):
        """One pass over every part, re-seating each with ``pick``."""
        moved = 0
        lo, hi = _bbox(fine, offsets)

        def on_face(i):
            a, b = fine[i].aabb(offsets[i])
            return -int(np.count_nonzero((a <= lo + 1e-6) | (b >= hi - 1e-6)))

        # Parts on the box's faces first: they are the only ones that can
        # shrink it, and freeing space behind them lets the rest follow.
        for part in sorted(offsets, key=lambda i: (on_face(i), i)):
            if out_of_time():
                break
            pose = fine[part]
            pile.paint(pose, offsets[part], on=False)
            new = pick(part)
            if not np.array_equal(new, offsets[part]):
                moved += 1
                offsets[part] = new
            pile.paint(pose, offsets[part])
        return moved

    def reseat_pass():
        return sweep(lambda part: _reseat(pile, fine, offsets, part, window,
                                          objective, container, dilated[part]))

    def shake_pass(d):
        return sweep(lambda part: _shake(pile, fine, offsets, part, window,
                                         objective, container, dilated[part], d))

    def orient_pass():
        """Re-choose every part's orientation once, face parts first.

        Unlike the sweeps this replaces the pose as well as the offset, so
        the part's mask and its dilation both have to be rebuilt when one
        is taken.  The pile is repainted with whichever pose ends up
        winning, so the lattice is never left holding a mask that no part
        is standing in.
        """
        moved = 0
        lo, hi = _bbox(fine, offsets)

        def on_face(i):
            a, b = fine[i].aabb(offsets[i])
            return -int(np.count_nonzero((a <= lo + 1e-6) | (b >= hi - 1e-6)))

        for part in sorted(offsets, key=lambda i: (on_face(i), i)):
            if out_of_time():
                break
            pile.paint(fine[part], offsets[part], on=False)
            got = _reorient(pile, fine, offsets, part, window, objective,
                            container, meshes, pitch, clearance_voxels)
            if got is not None:
                pose, off = got
                fine[part], offsets[part] = pose, off
                dilated[part] = ndimage.binary_dilation(
                    pose.mask, np.ones((3, 3, 3), bool))
                moved += 1
            pile.paint(fine[part], offsets[part])
        return moved

    # Each direction is tipped at most twice: a second cycle is worth having
    # because the first one's re-seats change what the same tip can reach,
    # and a third almost never moves anything.
    tips = list(SHAKE_DIRS) * SHAKE_CYCLES if shake else []
    tipped = 0
    rounds = ORIENT_ROUNDS if orient else 0
    oriented = 0

    def note(label):
        if on_frame is not None:
            on_frame(fine, offsets, label)

    for _ in range(sweeps + len(tips) + rounds):
        if out_of_time():
            break
        if reseat_pass():
            note("settle")
            continue
        # Sliding has run out of moves.  Before tipping -- which only
        # relocates slack and needs another sweep to cash it in -- try
        # turning the parts, which can shrink the box on its own.
        if oriented < rounds:
            oriented += 1
            if orient_pass():
                note("orient")
                continue
        # The objective sweep has run out of moves.  That does not mean the
        # arrangement is tight -- only that no part can shrink the box on
        # its own from where it stands.  Tip the box to change where it
        # stands, and try again; stop once no tip left moves anything.
        while tipped < len(tips) and not out_of_time():
            d = tips[tipped]
            tipped += 1
            if shake_pass(d):
                note("tip")
                break
        else:
            break

    lo, hi = _bbox(fine, offsets)
    ext = hi - lo
    now = _measure(objective, ext)
    # Judged on the objective that was actually being minimised: under
    # --objective height a shorter stack wins even if the footprint grows
    # inside its container, and a smaller footprint at the same height is
    # still worth having.
    if now >= start:
        return None

    out_packer = Packer([[fine[i]] for i in range(len(packer.poses))],
                        objective=packer.objective,
                        contact_weight=packer.contact_weight)
    out = out_packer.finalise(
        [Placement(i, 0, np.asarray(offsets[i], dtype=np.int64))
         for i in sorted(offsets)])
    return now, out_packer, out, ext, pitch


# ----------------------------------------------------------------------
# The ladder
#
# Which lattice settles best is not predictable from the arrangement.  A
# finer one has a thinner conservative skin and so more room to give, but
# it is also a different lattice: the parts round onto it differently, and
# the sweep converges somewhere else.  Measured over four arrangements of
# the sample parts, the finest rung won twice, the middle two once each,
# and the coarsest -- the pitch this pass used to run at alone -- never.
#
# So do not pick.  Run several rungs from the same arrangement, keep
# whichever lands smallest, and let the cores decide how many to try: they
# are independent, single-threaded, and none of them can return anything
# worse than what it was handed.
#
# The tipped rungs are last because they cost about five times a plain one
# and only sometimes pay -- on one arrangement they were the best result
# by a full point of density, on another they moved 37 parts and changed
# the box by nothing at all.
#
# No rung re-orients.  That pass exists and is sound (--settle-orient), but
# adding rungs for it measured worse than leaving it out: see the note in
# README.md.  The rungs share one wall-clock deadline, and a dear rung that
# rarely pays takes time from cheap ones that usually do.
#
# Each entry is (pitch multiplier, tip, re-orient).
LADDER = ((1.0, False, False), (4 / 3, False, False), (5 / 3, False, False),
          (2.0, False, False), (4 / 3, True, False), (5 / 3, True, False))


def _rungs(resolution, workers, orient=None):
    """The ladder, as far up it as there are cores to climb.

    ``orient`` of None lets each rung carry its own setting, which is what
    a normal run wants; True or False forces every rung one way, which is
    how the two are measured against each other on the same arrangements.
    """
    n = max(1, min(int(workers), len(LADDER)))
    rungs = []
    for m, shake, turn in LADDER[:n]:
        if orient is not None:
            turn = bool(orient)
        rungs.append((max(8, int(round(resolution * m))), shake, turn))
    if orient is not None:
        # Forcing the flag can make two rungs identical; running the same
        # lattice twice measures nothing.
        seen, out = set(), []
        for r in rungs:
            if r not in seen:
                seen.add(r)
                out.append(r)
        rungs = out
    return rungs


_SHARED = {}


def _init_worker(shared):
    _SHARED["v"] = shared


def _worker(rung):
    meshes, packer, packing, clearance, sweeps, deadline = _SHARED["v"]
    resolution, shake, turn = rung
    try:
        return _settle_once(meshes, packer, packing, resolution, clearance,
                            sweeps, deadline, shake, turn)
    except Exception:
        # One rung failing is not a reason to lose the others, and the
        # arrangement handed in is still there to fall back on.
        return None


def settle(meshes, packer, packing, resolution=96, clearance=0.0, sweeps=8,
           budget=None, workers=1, orient=None, say=None, on_frame=None):
    """Shake a finished arrangement down onto a finer lattice.

    ``resolution`` is the first rung of the ladder in voxels across the
    largest part; ``workers`` decides how many rungs above it are tried.

    ``orient`` of None lets each rung decide whether to re-choose the
    parts' orientations; True or False forces it on or off everywhere,
    which is how the pass is measured rather than how it is used.

    ``on_frame(poses, offsets, label)`` is called after every pass that
    moved something, for tracing.  It only applies to a single-rung run:
    the rungs of a full ladder run in other processes, and shipping every
    intermediate arrangement back through a pipe to throw all but one away
    is not worth the wire.

    Returns ``(packer, packing, extents)`` for the settled arrangement, or
    ``None`` if no rung could improve on the one it was given.
    """
    if len(packing.placements) < 2:
        return None
    if len(packing.placements) != len(packer.poses):
        # A partial packing has no settled arrangement to speak of, and the
        # fine pose lists would not line up with the part list.
        return None

    t0 = time.time()
    objective = "height" if packer.objective in ("height", "fit") else "volume"
    start = _measure(objective, packing.extents)
    rungs = _rungs(resolution, workers, orient)
    # An absolute deadline, not a duration: the rungs run in processes that
    # take a second or two to start, and a duration would hand each of them
    # a fresh full budget from whenever it happened to get going.
    deadline = None if budget is None else t0 + budget
    shared = (meshes, packer, packing, clearance, sweeps, deadline)

    if len(rungs) == 1:
        results = [_settle_once(meshes, packer, packing, rungs[0][0], clearance,
                                sweeps, deadline, rungs[0][1], rungs[0][2],
                                on_frame)]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=len(rungs),
                                 initializer=_init_worker,
                                 initargs=(shared,)) as pool:
            results = list(pool.map(_worker, rungs))

    best = None
    for got in results:
        if got is not None and (best is None or got[0] < best[0]):
            best = got
    if best is None:
        if say is not None:
            say("  settle      no improvement (%d %s, %.1fs)"
                % (len(rungs), "lattice" if len(rungs) == 1 else "lattices",
                   time.time() - t0))
        return None

    now, out_packer, out, ext, pitch = best
    if say is not None:
        gain = 100 * (1 - now[0] / start[0]) if start[0] else 0.0
        what, shown = ("height" if objective == "height" else "vol"), now[0]
        if gain <= 1e-9 and start[1]:
            # The objective tied and volume broke the tie; report that.
            what, shown, gain = "vol", now[1], 100 * (1 - now[1] / start[1])
        say("  settle      %s mm   %s %.4g   (-%.1f%%, pitch %.2f mm of %d, %.1fs)"
            % (" x ".join("%7.1f" % v for v in ext), what, shown, gain,
               pitch, len(rungs), time.time() - t0))
    return out_packer, out, ext
