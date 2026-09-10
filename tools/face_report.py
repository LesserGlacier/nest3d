"""Why is this box the size it is?  A per-face account of what holds it open.

The settle stops at a one-part-move optimum: no single part can be re-seated
in a way that shrinks the box, and yet the box is plainly loose.  Two escapes
have already been tried and rejected on measurements (a wider search window,
and a wall-push relaxation -- see the README).  The next one should be chosen
on numbers, so this tool produces the numbers.

What it answers
---------------
A box face can only move inward if *every* part standing on it moves inward,
so each face is a small dependency problem of its own:

1. which parts touch the face -- the only ones that can shrink the box along
   that axis at all;
2. how far each of those could slide straight inward on its own, with the
   rest of the pile held still.  Zero is the interesting answer: it means the
   part is not merely disinclined to move, it is jammed;
3. for the jammed ones, *who* is in the way, and whether that blocker could
   itself get out of the way.  Taking that to a fixpoint gives the jam: the
   parts that are stuck on each other rather than merely standing still.  A
   jam of two is one part against one other, and a pair move frees it; three
   or more is a genuine arch, and only a coordinated multi-part move helps.

The last number is the one that decides what to build next.  A pile of pairs
argues for a cheap two-part exchange; a pile of arches argues for something
that moves a whole group under contact, and the README has already priced
that.

Two counts are reported per face and they are not the same count.  The *jam*
is the mutual-blocking cluster above.  The *move* is everything that would
have to translate for the face to come in by a voxel -- every part standing
on the face, jammed or not, plus whatever they are jammed against.  A face
can have an empty jam and still need a four-part move, which is a different
complaint entirely: nothing is stuck, but no single re-seat shrinks the box,
so a one-part-at-a-time sweep has no reason to make any of the moves.

Reading the numbers honestly
----------------------------
Blocking is measured on the same conservative masks the settle moves parts
on, at the same pitch, because that is the geometry the settle actually
faces: two parts whose masks touch may have a fraction of a voxel of real air
between them, but the settle cannot use it either, so a jam reported here is
a jam the settle really has.  Everything reported in millimetres -- face
positions, slide distances, box dimensions -- comes from the exact mesh AABBs
carried on each pose, never from a voxel count, exactly as the rest of the
tool reports its answer.

The pile is rebuilt with settle.py's own helpers (``_fine_poses``,
``_transfer``, ``_Pile``) so that the lattice under this report is the same
lattice the settle converged on, rather than a second opinion about it.

    python tools/face_report.py                  # all cached arrangements
    python tools/face_report.py --state packed   # skip the settle
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nest3d import settle as settle_mod                     # noqa: E402
from nest3d.settle import _Pile, _bbox, _fine_poses, _transfer  # noqa: E402
from nest3d.voxel import exact_overlap, suggest_pitch       # noqa: E402

# Where the parts live is a property of the machine, not of the repo: the real
# workload sits outside the tree and at a different path on every computer,
# and the sample parts are a fixture rather than the thing being measured.  So
# the location comes from the environment and the samples are only the
# fallback -- a path written into this file would be wrong on the next machine
# and would put someone's directory layout in a public repository.
PARTS_ENV = "NEST3D_PARTS"
SAMPLE_PARTS = str(ROOT / "examples" / "sample_parts" / "*.step")


def default_parts():
    """The part set to measure: whatever ``NEST3D_PARTS`` names, else the samples."""
    return os.environ.get(PARTS_ENV) or SAMPLE_PARTS


def find_parts(pattern):
    """Expand a parts glob, or explain how to point the tool at a real one."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(
            "no parts matched %r\n"
            "Set %s to a glob of the parts to measure, or pass --parts."
            % (pattern, PARTS_ENV))
    return paths

# A part counts as standing on a face if its exact AABB reaches within half a
# fine voxel of it.  Half a voxel rather than zero because the arrangement is
# quantised to this lattice: two parts flush against the same wall can differ
# by a rounding, and calling only one of them a face part would blame the
# wrong geometry.  Anything further back is not ignored -- it shows up as the
# "next" gap, and it limits the gain a face move would actually realise.
TOUCH_TOL_VOXELS = 0.5

# How far inward a part is probed for free space.  A part that can slide this
# far is unambiguously loose; the exact figure past that point tells us
# nothing we would act on.
PROBE_VOXELS = 12

# A part counts as nested when this much of it sits inside another's
# concavity.  A quarter is low enough to catch a vessel that is only half
# swallowed and high enough that a part merely leaning into a recess does not
# qualify -- and the fraction is printed, so the reader can judge each case.
NEST_MIN_FRACTION = 0.25

# A shrink smaller than this is arithmetic on the AABB corners, not a move
# anyone could make; counting it as a gain would report faces as improvable
# that are not.
GAIN_EPS_MM = 1e-6

AXES = "xyz"

# (axis, side): side -1 is the low face, whose inward direction is +1.
FACES = tuple((axis, side) for axis in range(3) for side in (-1, +1))


def face_name(axis, side):
    return ("-" if side < 0 else "+") + AXES[axis]


# ----------------------------------------------------------------------
# Rebuilding the settle's lattice


def build_lattice(meshes, packer, packing, resolution):
    """Re-seat an arrangement on the fine lattice, as ``_settle_once`` does.

    This repeats settle.py's lattice setup rather than calling into it,
    because the only entry point there settles as well as builds, and a
    diagnostic that silently improved the arrangement it was asked to explain
    would be worse than useless.  The pitch rule is copied verbatim so that a
    report on an already-settled arrangement lands on the lattice that
    arrangement is standing on, not a coarser one it would have to round onto.
    """
    pitch = min(suggest_pitch(meshes, resolution), packer.pitch)
    fine = _fine_poses(meshes, packing, packer, pitch, 0)

    window = int(np.ceil(packer.pitch / pitch)) + settle_mod.EXTRA_WINDOW
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
        return None
    return fine, pile, offsets, pitch


# ----------------------------------------------------------------------
# The three questions


def free_slide(pile, fine, offsets, part, axis, inward, probe):
    """Voxels ``part`` may translate straight inward before it hits something.

    Pure translation along one axis, nothing else: the question is whether
    there is air immediately behind the part, not whether some clever
    re-seating exists.  The pile's own correlation answers it for the whole
    line of offsets at once, which is also the exact test the settle uses to
    decide where a part may stand -- so a zero here is a zero there.
    """
    pose = fine[part]
    here = offsets[part]
    step = np.zeros(3, dtype=np.int64)
    step[axis] = inward
    far = here + step * probe

    lo = np.maximum(np.minimum(here, far), 0)
    hi = np.minimum(np.maximum(here, far), pile.shape - pose.shape)
    if np.any(hi < lo):
        return 0, True

    # The part must be lifted out first, or it collides with itself.
    pile.paint(pose, here, on=False)
    free = pile.free_offsets(pose, lo, hi)
    pile.paint(pose, here, on=True)

    # Cheap proof that the lattice really is the arrangement: with the part
    # lifted out, the seat it just left has to be free.  If it is not, the
    # transfer or the painting is wrong and every number below it is fiction.
    if not free[tuple(here - lo)]:
        raise RuntimeError("part %d overlaps the pile at its own seat" % part)

    reach = 0
    for k in range(1, probe + 1):
        idx = here + step * k - lo
        if np.any(idx < 0) or np.any(idx >= np.asarray(free.shape)):
            return reach, True          # ran into the lattice, not into a part
        if not free[tuple(idx)]:
            return reach, False
        reach = k
    return reach, True


def blockers(fine, offsets, part, delta):
    """Which parts a unit inward shift of ``part`` would run into, and by how much.

    The pile knows *that* an offset is occupied but not by whom, so blame is
    attributed pairwise on the same masks.  The voxel counts are kept because
    they separate "resting against" from "would have to pass through".
    """
    moved = offsets[part] + delta
    out = []
    for other in offsets:
        if other == part:
            continue
        n = exact_overlap(fine[part], moved, fine[other], offsets[other])
        if n:
            out.append((other, n))
    out.sort(key=lambda t: -t[1])
    return out


def move_group(fine, pile, offsets, seeds, delta):
    """Smallest set of parts that can take the shift ``delta`` together.

    Start from ``seeds`` and close under "is in the way of": anything a
    member would collide with has to come along, and then whatever *that* one
    collides with, and so on.

    Seeded with every part standing on the face, the fixpoint is the
    coordinated move the face is asking for.  Seeded with only the jammed
    ones, it is the mutual-blocking cluster -- the arch, if there is one.
    Both are wanted, and they are the same closure over different seeds.

    The closure is feasible by construction, since a common translation
    cannot make two members overlap each other and nothing outside is left in
    the way.  It is still checked against the lattice bounds, which the
    closure knows nothing about.
    """
    group = set(seeds)
    deps = {}
    queue = list(seeds)
    while queue:
        part = queue.pop()
        hits = blockers(fine, offsets, part, delta)
        deps[part] = hits
        for other, _ in hits:
            if other not in group:
                group.add(other)
                queue.append(other)

    ok = True
    for part in group:
        pose = fine[part]
        new = offsets[part] + delta
        if np.any(new < 0) or np.any(new + pose.shape > pile.shape):
            ok = False
    return group, deps, ok


def nesting(fine, offsets, min_fraction=None):
    """Pairs where one part genuinely sits in another's concavity.

    Hollow vessels are the whole reason this packer voxelises instead of
    working on hulls, so "did the cup end up inside the bowl" is a fact about
    the arrangement worth stating outright -- and a nested part is a different
    kind of neighbour from an adjacent one, since it is enclosed on several
    sides at once rather than resting against one face.

    Inside-ness is decided on the lattice by bracketing: a voxel counts as
    being in B's concavity when B has material on *both* sides of it along at
    least two of the three axes.  Two rather than three, because the
    interesting case here is an open vessel -- a cup's cavity is bracketed by
    the walls in x and y and by the base below, but open at the top, so
    demanding all three axes would report that nothing ever nests.
    """
    if min_fraction is None:
        min_fraction = NEST_MIN_FRACTION
    out = []
    for inner in sorted(offsets):
        a_pose, a_off = fine[inner], offsets[inner]
        for outer in sorted(offsets):
            if outer == inner:
                continue
            b_pose, b_off = fine[outer], offsets[outer]
            lo = np.minimum(a_off, b_off)
            hi = np.maximum(a_off + a_pose.shape, b_off + b_pose.shape)
            if np.any(np.minimum(a_off + a_pose.shape, b_off + b_pose.shape)
                      <= np.maximum(a_off, b_off)):
                # The two bounding boxes are disjoint, so neither is inside
                # the other and the bracketing test would only cost time.
                continue

            occ = np.zeros(tuple(hi - lo), dtype=bool)
            s = b_pose.shape
            o = b_off - lo
            occ[o[0]:o[0] + s[0], o[1]:o[1] + s[1], o[2]:o[2] + s[2]] |= b_pose.mask

            brackets = np.zeros(occ.shape, dtype=np.uint8)
            for axis in range(3):
                pre = np.maximum.accumulate(occ, axis=axis)
                post = np.flip(np.maximum.accumulate(np.flip(occ, axis=axis),
                                                     axis=axis), axis=axis)
                brackets += (pre & post & ~occ).view(np.uint8)
            inside = brackets >= 2

            s = a_pose.shape
            o = a_off - lo
            sub = inside[o[0]:o[0] + s[0], o[1]:o[1] + s[1], o[2]:o[2] + s[2]]
            n = int(np.count_nonzero(sub & a_pose.mask))
            frac = n / max(a_pose.filled, 1)
            if frac >= min_fraction:
                out.append((inner, outer, frac))
    return out


def analyse(fine, pile, offsets, pitch, probe=PROBE_VOXELS,
            touch_tol=TOUCH_TOL_VOXELS):
    """One record per face: who stands on it, who is jammed, and against whom."""
    lo, hi = _bbox(fine, offsets)
    ext = hi - lo
    tol = touch_tol * pitch

    corner = {i: fine[i].aabb(offsets[i]) for i in offsets}
    records = []

    for axis, side in FACES:
        if side < 0:
            plane = lo[axis]
            dist = {i: corner[i][0][axis] - plane for i in offsets}
        else:
            plane = hi[axis]
            dist = {i: plane - corner[i][1][axis] for i in offsets}
        inward = -side

        touching = sorted(i for i in offsets if dist[i] <= tol)
        behind = [dist[i] for i in offsets if i not in touching]
        next_gap = min(behind) if behind else float("inf")

        slides = {}
        for part in touching:
            reach, capped = free_slide(pile, fine, offsets, part, axis, inward,
                                       probe)
            slides[part] = (reach, capped)

        delta = np.zeros(3, dtype=np.int64)
        delta[axis] = inward
        group, deps, feasible = move_group(fine, pile, offsets, touching, delta)

        # The jam proper: only the parts that cannot move, and only what they
        # are stuck on.  Seeding this with the whole touching set -- as the
        # move above does -- would count parts that are perfectly free to
        # slide as members of an arch, and every face would look jammed.
        culprits = [p for p in touching if slides[p][0] == 0]
        jam, jam_deps, _ = (move_group(fine, pile, offsets, culprits, delta)
                            if culprits else (set(), {}, True))

        # A jam of three is not automatically an arch in the load-bearing
        # sense: it can be a chain, where each part waits on the next and the
        # last one is free.  Genuine mutual blocking -- p in q's way and q in
        # p's, which interlocking parts can manage -- is worth separating out,
        # because a chain can be unwound one part at a time from the far end
        # and a cycle cannot.
        mutual = []
        for p in sorted(jam):
            for other, _n in jam_deps.get(p, ()):
                if other > p and p in [o for o, _ in jam_deps.get(other, ())]:
                    mutual.append((p, other))

        # What the coordinated move would actually buy.  Not assumed to be a
        # voxel: a part half a voxel behind the face becomes the new face, and
        # a group member standing on the *opposite* wall pushes that one out,
        # which can cancel the gain entirely.  Both are worth seeing.
        new_lo = np.full(3, np.inf)
        new_hi = np.full(3, -np.inf)
        for i in offsets:
            off = offsets[i] + delta if i in group else offsets[i]
            a, b = fine[i].aabb(off)
            new_lo = np.minimum(new_lo, a)
            new_hi = np.maximum(new_hi, b)
        new_ext = new_hi - new_lo
        gain_mm = float(ext[axis] - new_ext[axis])
        gain_vol = float(np.prod(ext) - np.prod(new_ext))

        # Members the shift pushes past the *far* face, which is where the
        # gain goes when a face turns out to be un-shrinkable: the group that
        # has to move to bring one wall in reaches the other wall and takes it
        # out with it.  The test is a voxel of clearance, not the touching
        # tolerance, because that is exactly the distance a one-voxel shift
        # eats -- a member half a voxel short of the far wall gives back half
        # the gain, and no advisory based on "is it touching" would say so.
        if side < 0:
            far = {i: hi[axis] - corner[i][1][axis] for i in group}
        else:
            far = {i: corner[i][0][axis] - lo[axis] for i in group}
        opposite = sorted(i for i in group if far[i] < pitch - 1e-9)

        records.append(dict(
            axis=axis, side=side, name=face_name(axis, side), plane=float(plane),
            touching=touching, next_gap=float(next_gap), slides=slides,
            group=sorted(group), deps=deps, feasible=feasible,
            culprits=culprits, jam=sorted(jam), jam_deps=jam_deps, mutual=mutual,
            gain_mm=gain_mm, gain_vol=gain_vol, gain_pct=100 * gain_vol / float(np.prod(ext)),
            opposite=opposite, pitch=pitch, ext=ext,
        ))
    return records, ext


def classify(rec):
    """The one-word answer for the tally at the end.

    Named for the move that would free the face, since that is the thing a
    future optimisation would have to be able to make.  "loose" is the case
    where nothing is stuck at all and the face is held open only because no
    *single* part can shrink the box by moving -- which is precisely the
    one-part-move optimum, and needs no arch to explain it.
    """
    n = len(rec["jam"])
    if n == 0:
        return "loose"
    if n == 1:
        # A jammed part with nothing in its way is jammed against the edge of
        # the analysis lattice, which means the margin is too small and the
        # face's numbers should not be believed.  It has never happened, but
        # silently filing it as an arch would hide the day it does.
        return "edge"
    if n == 2:
        return "pair"
    return "arch"


# ----------------------------------------------------------------------
# Reporting


def print_report(label, offsets, ext, records, part_volume=None,
                 reported=None, nested=()):
    print()
    print("=" * 74)
    print("%s" % label)
    dims = " x ".join("%7.2f" % v for v in ext)
    line = "  box %s mm   vol %.5g mm^3" % (dims, float(np.prod(ext)))
    if part_volume:
        line += "   density %5.2f%%" % (100 * part_volume / float(np.prod(ext)))
    print(line)
    if reported is not None and np.max(np.abs(np.asarray(reported) - ext)) > 1e-6:
        # The transfer onto the analysis lattice may nudge a part that rounded
        # into a neighbour, so say so rather than quietly reporting a box the
        # arrangement never had.
        print("  (arrangement as handed in: %s mm -- the analysis lattice moved it)"
              % " x ".join("%7.2f" % v for v in np.asarray(reported)))
    print("  %d parts, fine pitch %.3f mm" % (len(offsets), records[0]["pitch"]))
    if nested:
        print("  nested: " + "; ".join(
            "%d is %.0f%% inside %d" % (a, 100 * f, b) for a, b, f in nested))
    else:
        print("  nested: none -- no part sits inside another's concavity")
    print()

    for rec in records:
        pitch = rec["pitch"]
        gap = ("%.2f mm" % rec["next_gap"]) if np.isfinite(rec["next_gap"]) else "-"
        print("  face %-2s at %8.2f mm   %d touching   next part %s behind"
              % (rec["name"], rec["plane"], len(rec["touching"]), gap))
        for part in rec["touching"]:
            reach, capped = rec["slides"][part]
            how = "%5.2f mm (%s%d vox)" % (reach * pitch, ">=" if capped else "", reach)
            if reach:
                print("      part %-2d  free inward %s" % (part, how))
            else:
                who = ", ".join("%d[%d vox]" % (o, n) for o, n in rec["deps"][part])
                print("      part %-2d  free inward %s  blocked by %s"
                      % (part, how, who or "the lattice edge"))
        # Anything in the jam that is not itself on the face: these are the
        # parts the face is leaning on, and whether they in turn are stuck is
        # what separates a pair from an arch.
        for part in rec["jam"]:
            if part in rec["touching"]:
                continue
            who = ", ".join("%d[%d vox]" % (o, n) for o, n in rec["jam_deps"][part])
            print("        via %-2d  %s" % (part, ("blocked by " + who) if who
                                            else "free to follow"))
        kind = classify(rec)
        if kind == "loose":
            print("      -> jam none: every face part can slide, but no one of "
                  "them shrinks the box alone")
        else:
            shape = ("chain" if not rec["mutual"] else "mutual " + ", ".join(
                "%d<->%d" % pq for pq in rec["mutual"]))
            print("      -> jam %s of %d: {%s}  (%s)"
                  % (kind, len(rec["jam"]),
                     ", ".join(str(i) for i in rec["jam"]), shape))
        note = ""
        if rec["opposite"]:
            note = "  (pushes the far wall out: %s)" % ", ".join(
                str(i) for i in rec["opposite"])
        print("      -> move %d part%s {%s} one voxel in:  %+.3f mm on %s, "
              "volume %+.3f%%%s%s"
              % (len(rec["group"]), "" if len(rec["group"]) == 1 else "s",
                 ", ".join(str(i) for i in rec["group"]),
                 -rec["gain_mm"], AXES[rec["axis"]], -rec["gain_pct"], note,
                 "" if rec["feasible"] else "  [runs off the lattice]"))
        print()


def summarise(rows):
    """The bottom line: pair moves or arches, counted rather than eyeballed."""
    print()
    print("=" * 74)
    print("SUMMARY  (%d faces over %d arrangement states)"
          % (len(rows), len({(r["label"]) for r in rows})))
    print()

    for state in sorted({r["state"] for r in rows}):
        sub = [r for r in rows if r["state"] == state]
        print("  %s: %d faces" % (state, len(sub)))
        for kind in ("loose", "edge", "pair", "arch"):
            k = [r for r in sub if r["kind"] == kind]
            if not k:
                continue
            paying = [r for r in k if r["pays"]]
            print("    jam %-5s %2d faces   jam size %-8s  %d of them still "
                  "shrink the box on a group move"
                  % (kind, len(k),
                     "/".join(str(s) for s in sorted({r["jam"] for r in k})),
                     len(paying)))
        sizes = sorted(r["move"] for r in sub)
        print("    parts per group move: %s   (median %d, max %d)"
              % (", ".join(str(s) for s in sizes), int(np.median(sizes)), max(sizes)))
        print("    parts touching each face: %s"
              % ", ".join(str(s) for s in sorted(r["touching"] for r in sub)))
        print("    jammed face parts per face: %s"
              % ", ".join(str(s) for s in sorted(r["culprits"] for r in sub)))
        print("    faces held by exactly one jammed part: %d of %d"
              % (len([r for r in sub if r["culprits"] == 1]), len(sub)))
        print("    faces with a genuine mutual block (not just a chain): %d of %d"
              % (len([r for r in sub if r["mutual"]]), len(sub)))
        print("    faces whose group move shrinks the box: %d of %d"
              % (len([r for r in sub if r["pays"]]), len(sub)))
        # The interesting failure: the group that has to move to bring one
        # wall in reaches all the way to the opposite wall, so the shift that
        # frees this face grows the other one and the box does not change.
        print("    faces whose group reaches the opposite wall (gain cancelled): "
              "%d of %d" % (len([r for r in sub if r["spans"]]), len(sub)))
        print()

    # The face that would pay best is the one a future optimisation would aim
    # at, so it gets counted separately from the other five.
    # Ranked by what the move buys, with the smallest jam winning ties: when
    # no face pays anything -- which is what a well-settled arrangement looks
    # like -- the honest answer is the cheapest jam to attack, not an
    # arbitrary one of six zeroes.
    def rank(r):
        return (r["gain_mm"] if r["pays"] else 0.0, -r["jam"], -r["move"])

    print("  limiting face per arrangement (the one whose move pays most):")
    for label in sorted({r["label"] for r in rows}):
        sub = [r for r in rows if r["label"] == label]
        best = max(sub, key=rank)
        jam = "loose" if best["kind"] == "loose" else "%s of %d" % (best["kind"],
                                                                    best["jam"])
        print("    %-26s %-2s  jam %-11s move %d parts   %+.3f mm  %+.3f%% vol"
              % (label, best["face"], jam, best["move"],
                 -best["gain_mm"], -best["gain_pct"]))
    print()
    best_rows = [max((r for r in rows if r["label"] == label), key=rank)
                 for label in sorted({r["label"] for r in rows})]
    for kind in ("loose", "edge", "pair", "arch"):
        n = len([r for r in best_rows if r["kind"] == kind])
        print("    jam %-5s %d of %d limiting faces" % (kind, n, len(best_rows)))
    print()


# ----------------------------------------------------------------------


def fill_cache(pattern, cache, seeds, workers, resolution, orientations,
               budget):
    """Pack any missing seed into ``cache``, in the format bench_settle uses.

    The report is only as good as the arrangements it is handed, and asking
    someone to run a second tool first is how a diagnostic ends up being run
    on whatever happened to be lying in the cache.  Seeds already present are
    left alone: packing is the expensive, noisy half, and re-rolling it would
    change the subject between runs.
    """
    from nest3d import pipeline

    paths = find_parts(pattern)
    cache.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        path = cache / ("arr%02d.pkl" % seed)
        if path.exists():
            continue
        print("packing seed %d from %d parts (%.0fs budget)"
              % (seed, len(paths), budget), flush=True)
        res = pipeline.run(paths, time_budget=budget, seed=seed,
                           workers=workers, resolution=resolution,
                           orientations=orientations, settle=False,
                           verbose=False)
        r = res.result
        blob = {"meshes": [p.mesh for p in res.parts], "packer": r.packer,
                "packing": r.packing,
                "extents": np.asarray(r.extents, dtype=float),
                "part_volume": float(r.lower_bound), "seed": seed}
        with open(path, "wb") as fh:
            pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load(cache, wait):
    """Load the cached arrangements, waiting if the cache is still filling."""
    deadline = time.time() + wait
    while True:
        paths = sorted(Path(cache).glob("arr*.pkl"))
        blobs = []
        for path in paths:
            try:
                with open(path, "rb") as fh:
                    blobs.append((path, pickle.load(fh)))
            except Exception:
                # Half-written: the packer is still dumping into it.
                blobs = None
                break
        if blobs:
            return blobs
        if time.time() > deadline:
            raise SystemExit("no complete arrangements in %s" % cache)
        time.sleep(10)


def run_state(label, state, meshes, packer, packing, resolution, part_volume,
              reported, rows, probe=PROBE_VOXELS, touch_tol=TOUCH_TOL_VOXELS):
    got = build_lattice(meshes, packer, packing, resolution)
    if got is None:
        print("  %s: could not be re-seated on the analysis lattice" % label)
        return
    fine, pile, offsets, pitch = got
    records, ext = analyse(fine, pile, offsets, pitch, probe, touch_tol)
    nested = nesting(fine, offsets)
    print_report(label, offsets, ext, records, part_volume, reported, nested)
    for rec in records:
        rows.append(dict(label=label, state=state, face=rec["name"],
                         kind=classify(rec), jam=len(rec["jam"]),
                         move=len(rec["group"]), culprits=len(rec["culprits"]),
                         mutual=len(rec["mutual"]), spans=bool(rec["opposite"]),
                         pays=rec["gain_mm"] > GAIN_EPS_MM,
                         touching=len(rec["touching"]), gain_mm=rec["gain_mm"],
                         gain_vol=rec["gain_vol"], gain_pct=rec["gain_pct"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parts", default=default_parts(),
                    help="glob of the parts to measure, used when a seed has to "
                         "be packed (default: $%s, else the sample parts)"
                         % PARTS_ENV)
    ap.add_argument("--seeds", default=None,
                    type=lambda s: [int(x) for x in s.split(",")],
                    help="pack these seeds into the cache first if they are "
                         "missing; without it, whatever is cached is analysed")
    ap.add_argument("--pack-time", type=float, default=120.0,
                    help="search budget per packed seed, in seconds")
    ap.add_argument("--pack-resolution", type=int, default=28)
    ap.add_argument("--orientations", default="rest")
    ap.add_argument("--cache", default=None,
                    help="directory of arr*.pkl (default: the bench cache)")
    ap.add_argument("--resolution", type=int, default=96,
                    help="voxels across the largest part for the analysis lattice")
    ap.add_argument("--settle-resolution", type=int, default=96)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--state", default="both",
                    choices=("both", "packed", "settled"))
    ap.add_argument("--touch-tol", type=float, default=TOUCH_TOL_VOXELS,
                    help="fine voxels of slack that still counts as touching a face")
    ap.add_argument("--probe", type=int, default=PROBE_VOXELS,
                    help="fine voxels of inward free space probed per face part")
    ap.add_argument("--wait", type=float, default=600.0,
                    help="seconds to wait for the cache to finish filling")
    args = ap.parse_args()

    cache = Path(args.cache) if args.cache else None
    if cache is None:
        import tempfile
        cache = Path(tempfile.gettempdir()) / "nest3d_bench"

    if args.seeds:
        fill_cache(args.parts, cache, args.seeds, args.workers,
                   args.pack_resolution, args.orientations, args.pack_time)

    rows = []
    for path, blob in load(cache, args.wait):
        meshes = blob["meshes"]
        pv = blob["part_volume"]
        name = path.stem

        if args.state in ("both", "packed"):
            run_state("%s  packed (seed %d)" % (name, blob["seed"]), "packed",
                      meshes, blob["packer"], blob["packing"], args.resolution,
                      pv, blob["extents"], rows, args.probe, args.touch_tol)

        if args.state in ("both", "settled"):
            t0 = time.time()
            got = settle_mod.settle(meshes, blob["packer"], blob["packing"],
                                    resolution=args.settle_resolution,
                                    workers=args.workers)
            if got is None:
                print("\n%s: settle found no improvement (%.0fs)"
                      % (name, time.time() - t0))
                continue
            s_packer, s_packing, s_ext = got
            run_state("%s  settled (seed %d)" % (name, blob["seed"]), "settled",
                      meshes, s_packer, s_packing, args.resolution, pv, s_ext,
                      rows, args.probe, args.touch_tol)

    summarise(rows)


if __name__ == "__main__":
    main()
