"""Does the settled box, fed back in as a container, buy anything?

The settle hands back a box several per cent smaller than the one the
search itself could reach, and README argues at length that a fixed
container is the better-posed question for this solver: "do these nine fit
in 118 x 121 x 113?" makes every part compete for the same space, where
"make the box small" lets a greedy placer hang parts off the end.  Nobody
has tried closing that loop -- taking the settled box as a fresh container
target, re-running the search against it, and settling whatever comes out.

The loop cannot be warm-started, and that is the whole difficulty.
``_settled_result`` drops ``config`` deliberately: a settled arrangement is
not something the constructive placer can replay from an (order, poses)
pair, so nothing of the first pass survives except the box dimensions and
their aspect ratio.  The second search therefore starts cold and pays for
itself twice over -- a fresh pose build, a fresh anneal, a fresh settle.
The question is only whether the better-posed target repays that.

Equal wall clock, not equal phases
----------------------------------
A loop that runs two searches has to be measured against a baseline given
the same *total* seconds, or it has proven nothing.  So the baseline runs
first for each seed and its measured wall clock becomes the loop's hard
deadline.  Load and voxelisation are charged honestly: the baseline pays
them once, the loop pays the load once and the voxelisation twice (the
re-squeeze rebuilds the coarse poses, since the pipeline does not hand its
own back), and the search budget the loop hands out is what is left after
subtracting both.

The second search runs through ``Solver.squeeze`` rather than
``pipeline.run(container=...)`` on purpose.  The pipeline's fixed-container
branch is a single annealed ``Search`` on one core, so pitting it against
an eight-worker baseline would measure that asymmetry rather than the
hypothesis.  ``squeeze`` is the same feasibility oracle fanned out over
aspect ratios, which is the parallel form of the same question -- and it
seeds its aspect candidates from the box handed in, so the settled aspect
ratio carries over along with the size.

    python tools/exp_resqueeze.py --seeds 0,1,2 --time 120 --workers 8
    python tools/exp_resqueeze.py --calibrate    # wall clock is not budget
    python tools/exp_resqueeze.py --probe --cache DIR   # how big is the ask?

Parts come from ``$NEST3D_PARTS`` or ``--parts``, falling back to the
sample parts in the repo.  Calibrate before comparing on a part set you
have not measured before: the cost model below is fitted rather than
universal, and a set an order of magnitude larger than the samples costs
its pools quite different money.

``--probe`` answers a cheaper question off the cached arrangements: how far
below the packed box does the settle land?  That gap is exactly what the
re-search is being asked to reach on the coarse lattice, where the masks
are half again the size of the parts inside them.
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

# Which parts to pack, as a glob.  The sample parts are the fallback
# because they are the only ones that live in the repo; a real part set is
# named by ``$NEST3D_PARTS`` or ``--parts`` and never written down here.
# This repo is public and is worked on from several machines, so a
# hardcoded path would leak a directory name on top of failing on three
# machines out of four.
SAMPLE_PARTS = str(ROOT / "examples" / "sample_parts" / "*.step")


def default_parts():
    return os.environ.get("NEST3D_PARTS") or SAMPLE_PARTS


def resolve_parts(pattern):
    """The files to pack, or an error that says how to point at them."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(
            "no parts matched %r.\n"
            "Set NEST3D_PARTS to a glob for the parts you want packed, or "
            "pass --parts.\nOn Windows use the drive-letter form "
            "(X:/dir/*.step); an MSYS /x/dir path matches nothing."
            % pattern)
    return paths

from nest3d import pipeline                                    # noqa: E402
from nest3d.geometry import load_parts                         # noqa: E402
from nest3d.pack import verify_packing                         # noqa: E402
from nest3d.settle import settle as settle_arrangement         # noqa: E402
from nest3d.solve import Solver, part_volume                   # noqa: E402
from nest3d.voxel import build_poses, suggest_pitch            # noqa: E402

# ``Solver.squeeze`` proposes ``hi * (1 - shrink)`` before it proposes
# anything else, so handing it a box one step *above* the target makes the
# very first container it asks about the settled box itself.  Anything
# looser would spend the first annealing run re-establishing a box the
# first pass already had.
SHRINK = 0.05

# Of the loop's second half, the share spent searching rather than
# settling.  The settle is usually finished well inside its slice -- the
# pipeline gives it 10% and it rarely wants that -- but a re-squeeze that
# leaves it nothing has thrown away the cheapest points on the table.
RESQ_SEARCH_SHARE = 0.80

# Wall clock a ``pipeline.run`` costs, as ``FIXED + RATE * time_budget``.
# ``time_budget`` is not the wall clock: the refine and the settle each
# stand up a process pool, and on Windows that means spawning interpreters
# and pickling the poses through to them, which costs the same handful of
# seconds whether the budget is thirty seconds or three hundred.  The loop
# has to invert this to work out what budget buys its first pass the share
# of the wall clock it is meant to have -- guessing gets pass A a budget it
# overruns by the entire deadline, leaving pass B nothing.
#
# These are the sample parts' numbers, and they are a default rather than a
# constant of nature: the fixed part is pool startup and pickling the poses
# through it, both of which scale with how big the masks are.  Run
# ``--calibrate`` on any new part set and pass what it prints.
FIXED_COST = 26.7
RATE = 0.699


def say(msg):
    # Output is redirected to a file for the long runs; unflushed prints
    # would leave it empty until the process exits.
    print(msg, flush=True)


def _fmt(ext):
    return " x ".join("%6.1f" % v for v in np.asarray(ext, dtype=float))


# ---------------------------------------------------------------------------
# Setup shared by both arms
# ---------------------------------------------------------------------------

def measure_overhead(paths, args):
    """Seconds of load and voxelisation, which neither arm can search with.

    Both are outside ``time_budget`` -- the pipeline starts its clock after
    them -- so a fair wall-clock comparison has to know their size before it
    can hand out search seconds.
    """
    t0 = time.time()
    parts = load_parts(paths, split_solids=False)
    load_s = time.time() - t0

    meshes = [p.mesh for p in parts]
    pitch = suggest_pitch(meshes, args.resolution)
    t0 = time.time()
    rotations = pipeline.build_orientations(parts, args.orientations, None, 0)
    poses = [build_poses(p.mesh, r, pitch) for p, r in zip(parts, rotations)]
    voxel_s = time.time() - t0

    pv = part_volume(poses, meshes)
    say("setup: %d parts, pitch %.2f mm, load %.1fs, voxelise %.1fs, "
        "part volume %.4g mm^3" % (len(parts), pitch, load_s, voxel_s, pv))
    return load_s, voxel_s, pv


def coarse_poses(parts, args, seed=0):
    """The pipeline's own coarse poses, rebuilt.

    ``pipeline.run`` keeps these to itself -- what it returns after a settle
    is the settled fine poses -- so the second half of the loop has to make
    them again.  That cost is real and is charged to the loop.
    """
    meshes = [p.mesh for p in parts]
    pitch = suggest_pitch(meshes, args.resolution)
    rotations = pipeline.build_orientations(parts, args.orientations, None, seed)
    return [build_poses(p.mesh, r, pitch) for p, r in zip(parts, rotations)]


# ---------------------------------------------------------------------------
# The two arms
# ---------------------------------------------------------------------------

def baseline_arm(paths, seed, budget, args):
    """The ordinary pipeline, all of the budget in one pass."""
    t0 = time.time()
    res = pipeline.run(paths, time_budget=budget, seed=seed,
                       workers=args.workers, resolution=args.resolution,
                       refine_resolution=args.refine_resolution,
                       orientations=args.orientations,
                       settle_resolution=args.settle_resolution,
                       verbose=False)
    wall = time.time() - t0
    ext = np.asarray(res.result.extents, dtype=float)
    return {
        "wall": wall,
        "ext": ext,
        "volume": float(np.prod(ext)),
        "pv": float(res.result.lower_bound),
        "timings": dict(res.timings),
        "overlaps": len(res.overlaps),
        "parts": res.parts,
    }


def resqueeze(meshes, poses, target, seed, budget, args):
    """Re-ask the search as a fixed-container question aimed at ``target``.

    Returns a dict describing what the second pass achieved, including the
    cases where it achieved nothing: an infeasible target burns the whole
    budget and is the outcome the hypothesis most needs measuring.
    """
    out = {"target": np.asarray(target, dtype=float), "fit": False,
           "ext": None, "volume": float("inf"), "squeeze_s": 0.0,
           "settle_s": 0.0, "settled": False, "overlaps": 0}

    solver = Solver(poses, meshes=meshes, seed=seed, verbose=False)
    solver._say = lambda _msg: None
    start = np.asarray(target, dtype=float) / (1.0 - SHRINK) ** (1.0 / 3.0)

    t0 = time.time()
    packer, sol, ext, history = solver.squeeze(
        start, budget=budget * RESQ_SEARCH_SHARE, starts=2, iterations=60,
        workers=args.workers, rounds=2, shrink=SHRINK)
    out["squeeze_s"] = time.time() - t0
    out["history"] = history
    if sol is None:
        # Every container the descent proposed was infeasible, so nothing
        # was ever banked and there is no arrangement to settle.
        return out

    out["fit"] = True
    ext = np.asarray(ext, dtype=float)
    out["ext"] = ext
    out["volume"] = float(np.prod(ext))

    left = budget - out["squeeze_s"]
    if left < 3.0:
        return out
    t0 = time.time()
    settled = settle_arrangement(meshes, packer, sol.packing,
                                 resolution=args.settle_resolution,
                                 budget=left, workers=args.workers)
    out["settle_s"] = time.time() - t0
    if settled is not None:
        s_packer, s_packing, s_ext = settled
        out["settled"] = True
        out["ext"] = np.asarray(s_ext, dtype=float)
        out["volume"] = float(np.prod(s_ext))
        out["overlaps"] = len(verify_packing(s_packer, s_packing))
    else:
        out["overlaps"] = len(verify_packing(packer, sol.packing))
    return out


def loop_arm(paths, seed, wall_budget, args):
    """Pipeline, then the settled box back in as a container, then settle.

    ``wall_budget`` is a hard deadline on the whole thing, taken from what
    the baseline actually spent on this seed.  The first pass gets a share
    of what is left once both voxelisations and the load are paid for; the
    second pass gets whatever wall clock genuinely remains, which is the
    only way the two arms end up comparable.
    """
    t_start = time.time()
    end = t_start + wall_budget

    # The wall clock the loop has to spend, once the load and the second
    # voxelisation -- neither of which is searching -- are paid for.
    pool = wall_budget - args.load_s - args.voxel_s
    # Pass A is asked for a *share of the wall clock*, so its budget has to
    # be de-rated through the cost model: handing it ``split * pool``
    # directly is how a 24-second budget spent 50 seconds and left pass B
    # nothing at all.
    a_wall_target = args.split * pool
    a_budget = max(10.0, (a_wall_target - args.fixed_cost) / args.rate)

    res_a = pipeline.run(paths, time_budget=a_budget, seed=seed,
                         workers=args.workers, resolution=args.resolution,
                         refine_resolution=args.refine_resolution,
                         orientations=args.orientations,
                         settle_resolution=args.settle_resolution,
                         verbose=False)
    a_wall = time.time() - t_start
    ext_a = np.asarray(res_a.result.extents, dtype=float)
    pv = float(res_a.result.lower_bound)
    row = {
        "a_budget": a_budget, "a_wall": a_wall, "a_wall_target": a_wall_target,
        "a_ext": ext_a, "a_volume": float(np.prod(ext_a)),
        "a_timings": dict(res_a.timings), "pv": pv,
        "b": None, "poses_s": 0.0,
    }

    t0 = time.time()
    poses = coarse_poses(res_a.parts, args, seed=seed)
    row["poses_s"] = time.time() - t0

    b_budget = end - time.time()
    row["b_budget"] = b_budget
    if b_budget < 8.0:
        # Nothing useful can happen in the residue; say so rather than
        # letting a two-second squeeze stand in for the idea.
        row["ext"] = ext_a
        row["volume"] = row["a_volume"]
        row["wall"] = time.time() - t_start
        return row

    meshes = [p.mesh for p in res_a.parts]
    # The volume scale is applied to the settled box: 1.0 asks the coarse
    # lattice for exactly what the fine one achieved, which may simply be
    # out of reach there -- the coarse masks are about 1.35x the solids.
    target = ext_a * args.scale ** (1.0 / 3.0)
    row["b"] = resqueeze(meshes, poses, target, seed + 4001, b_budget, args)

    # The loop can always keep what the first pass gave it, so the answer
    # is the better of the two; a lost re-squeeze costs time, never quality.
    if row["b"]["volume"] < row["a_volume"]:
        row["ext"] = np.asarray(row["b"]["ext"], dtype=float)
        row["volume"] = row["b"]["volume"]
    else:
        row["ext"] = ext_a
        row["volume"] = row["a_volume"]
    row["wall"] = time.time() - t_start
    return row


# ---------------------------------------------------------------------------
# The cheap question: how far below the packed box does the settle land?
# ---------------------------------------------------------------------------

def do_probe(args):
    """Measure the size of the ask, off the cached packed arrangements.

    The re-squeeze has to reach, on the coarse lattice, a box the fine
    lattice only reached by sliding parts the search could not slide.  How
    much smaller that box is bounds what the loop could possibly gain, and
    costs seconds to find out rather than the half hour a full comparison
    takes.
    """
    import tempfile
    d = Path(args.cache) if args.cache else Path(tempfile.gettempdir()) / "nest3d_bench"
    blobs = []
    for path in sorted(d.glob("arr*.pkl")):
        with open(path, "rb") as fh:
            blobs.append(pickle.load(fh))
    if not blobs:
        raise SystemExit("no cached arrangements in %s" % d)

    say("\n%-5s %28s %8s   %28s %8s   %7s" %
        ("seed", "packed box (mm)", "dens%", "settled box (mm)", "dens%", "vol -%"))
    gaps = []
    rows = []
    for blob in blobs:
        pv = float(blob["part_volume"])
        packed = np.asarray(blob["extents"], dtype=float)
        got = settle_arrangement(blob["meshes"], blob["packer"], blob["packing"],
                                 resolution=args.settle_resolution,
                                 workers=args.workers)
        settled = packed if got is None else np.asarray(got[2], dtype=float)
        v0, v1 = float(np.prod(packed)), float(np.prod(settled))
        gaps.append(100 * (1 - v1 / v0))
        rows.append((blob, settled))
        say("%-5d %28s %7.2f%%   %28s %7.2f%%   %6.2f%%"
            % (blob["seed"], _fmt(packed), 100 * pv / v0,
               _fmt(settled), 100 * pv / v1, 100 * (1 - v1 / v0)))
    say("\nthe settle takes %.2f%% of the box on average (min %.2f, max %.2f)."
        % (float(np.mean(gaps)), min(gaps), max(gaps)))
    say("that is the gap a re-squeeze has to close on the search's own")
    say("lattice, on top of whatever the first pass's squeeze already failed to.")

    if args.probe_budget <= 0:
        return

    # And now the question the whole experiment turns on, asked directly
    # and for pennies: hand that settled box back as a container and see
    # whether the search can reach it at all.  The cached packer's own
    # poses are reused, so this skips both the first search and the
    # voxelisation -- it is the feasibility oracle and nothing else.
    say("\nfeeding each settled box back as a container, %.0fs apiece:"
        % args.probe_budget)
    fits = 0
    for blob, settled in rows:
        pv = float(blob["part_volume"])
        poses = blob["packer"].poses
        pitch = float(blob["packer"].pitch)
        got = resqueeze(blob["meshes"], poses, settled, blob["seed"] + 9001,
                        args.probe_budget, args)
        if not got["fit"]:
            say("  seed %-2d  pitch %.2f mm   target %s  DOES NOT FIT  (%.1fs)"
                % (blob["seed"], pitch, _fmt(settled), got["squeeze_s"]))
            continue
        fits += 1
        say("  seed %-2d  pitch %.2f mm   target %s  fits: %s  %6.2f%%  (%.1fs)"
            % (blob["seed"], pitch, _fmt(settled), _fmt(got["ext"]),
               100 * pv / got["volume"], got["squeeze_s"] + got["settle_s"]))
    say("\n%d of %d settled boxes were reachable by the search that produced"
        % (fits, len(rows)))
    say("them.  A settled box the search cannot re-enter is a loop that can")
    say("only ever spend its second half and come back empty-handed.")


# ---------------------------------------------------------------------------

def do_calibrate(args):
    """Fit wall clock against ``time_budget`` for a plain pipeline run.

    Only the loop needs this, and it needs it badly: it hands its first
    pass a slice of the wall clock rather than a slice of the budget, and
    the two are not proportional.  A straight line through three budgets is
    plenty -- the fixed part is process-pool startup and the variable part
    is annealing, and neither is subtle.
    """
    paths = resolve_parts(args.parts)
    budgets = [args.time * f for f in (0.25, 0.5, 1.0)]
    walls = []
    for b in budgets:
        got = baseline_arm(paths, args.seeds[0], b, args)
        walls.append(got["wall"])
        say("budget %6.1fs -> wall %6.1fs   (%s)"
            % (b, got["wall"],
               ", ".join("%s %.1fs" % kv for kv in got["timings"].items())))
    rate, fixed = np.polyfit(np.array(budgets), np.array(walls), 1)
    say("\nwall ~= %.1f + %.3f * budget    "
        "(pass --fixed-cost %.1f --rate %.3f)" % (fixed, rate, fixed, rate))


def do_compare(args):
    paths = resolve_parts(args.parts)
    args.load_s, args.voxel_s, _pv = measure_overhead(paths, args)
    say("budget %.0fs per seed per arm, %d workers, split %.2f, "
        "container scale %.3f\n" % (args.time, args.workers, args.split,
                                    args.scale))

    rows = []
    for seed in args.seeds:
        say("seed %d ------------------------------------------------" % seed)
        base = baseline_arm(paths, seed, args.time, args)
        say("  baseline  %s   vol %.4g   density %6.2f%%   (%.1fs wall: %s)"
            % (_fmt(base["ext"]), base["volume"], 100 * base["pv"] / base["volume"],
               base["wall"],
               ", ".join("%s %.1fs" % kv for kv in base["timings"].items())))
        if base["overlaps"]:
            say("  baseline  WARNING: %d overlapping pairs" % base["overlaps"])

        loop = loop_arm(paths, seed, base["wall"], args)
        pv = loop["pv"]
        say("  loop A    %s   vol %.4g   density %6.2f%%   "
            "(%.1fs wall, aimed at %.1fs from a %.0fs budget)"
            % (_fmt(loop["a_ext"]), loop["a_volume"], 100 * pv / loop["a_volume"],
               loop["a_wall"], loop["a_wall_target"], loop["a_budget"]))
        b = loop["b"]
        if b is None:
            say("  loop B    skipped, no wall clock left")
        elif not b["fit"]:
            say("  loop B    target %s DOES NOT FIT   (squeeze %.1fs, poses %.1fs)"
                % (_fmt(b["target"]), b["squeeze_s"], loop["poses_s"]))
        else:
            say("  loop B    %s   vol %.4g   density %6.2f%%   "
                "(target %s, squeeze %.1fs, settle %.1fs%s)"
                % (_fmt(b["ext"]), b["volume"], 100 * pv / b["volume"],
                   _fmt(b["target"]), b["squeeze_s"], b["settle_s"],
                   "" if b["settled"] else ", settle no gain"))
            if b["overlaps"]:
                say("  loop B    WARNING: %d overlapping pairs" % b["overlaps"])
        say("  loop      %s   vol %.4g   density %6.2f%%   (%.1fs wall)\n"
            % (_fmt(loop["ext"]), loop["volume"], 100 * pv / loop["volume"],
               loop["wall"]))
        rows.append((seed, base, loop))

    say("\n%-5s %10s %10s %10s %9s   %8s %8s"
        % ("seed", "base%", "loopA%", "loop%", "delta", "base_s", "loop_s"))
    deltas = []
    for seed, base, loop in rows:
        pv = loop["pv"]
        b_d = 100 * base["pv"] / base["volume"]
        a_d = 100 * pv / loop["a_volume"]
        l_d = 100 * pv / loop["volume"]
        deltas.append(l_d - b_d)
        say("%-5d %9.2f%% %9.2f%% %9.2f%% %+8.2f   %7.1fs %7.1fs"
            % (seed, b_d, a_d, l_d, l_d - b_d, base["wall"], loop["wall"]))
    say("\nmean delta %+.2f density points (loop minus baseline), "
        "%d of %d seeds won" % (float(np.mean(deltas)),
                                sum(1 for d in deltas if d > 0), len(deltas)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=default_parts(),
                    help="glob for the parts to pack; defaults to "
                         "$NEST3D_PARTS, else the repo's sample parts")
    ap.add_argument("--seeds", default="0,1,2",
                    type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--time", type=float, default=120.0,
                    help="search budget handed to the baseline, in seconds")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--split", type=float, default=0.55,
                    help="share of the loop's search seconds given to pass A")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="volume scale on the settled box before feeding it back")
    ap.add_argument("--resolution", type=int, default=28)
    ap.add_argument("--refine-resolution", type=int, default=48)
    ap.add_argument("--settle-resolution", type=int, default=96)
    ap.add_argument("--orientations", default="rest")
    ap.add_argument("--fixed-cost", type=float, default=FIXED_COST,
                    help="seconds a pipeline run costs regardless of budget")
    ap.add_argument("--rate", type=float, default=RATE,
                    help="wall seconds per budget second on top of that")
    ap.add_argument("--probe", action="store_true",
                    help="measure the settle's gain off the cached arrangements")
    ap.add_argument("--probe-budget", type=float, default=60.0,
                    help="seconds per arrangement for the probe's re-squeeze; "
                         "0 to measure the settle gap only")
    ap.add_argument("--calibrate", action="store_true",
                    help="fit wall clock against time_budget, then stop")
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()
    if args.probe:
        do_probe(args)
    elif args.calibrate:
        do_calibrate(args)
    else:
        do_compare(args)


if __name__ == "__main__":
    main()
