"""End-to-end pipeline: files in, packed arrangement out.

Search coarse, refine fine, then settle.  The coarse pass does the real
searching, because that is where thousands of candidate arrangements are
affordable; the fine pass re-voxelises only the orientations that survived
and tightens the answer.  Orientation indices are kept stable between the
two, so a solution found coarse can be handed straight to the fine pass as
a starting point.

The settle is different in kind.  It searches nothing at all: it takes the
arrangement as found and lets each part slide to the best position within
a voxel or two of where it stands, on a lattice several times finer than
either search pass could afford.  That is what takes back the slack the
search's own pitch left at every contact -- see settle.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .geometry import load_parts
from .orient import candidate_rotations
from .pack import Packer, verify_packing
from .search import Search
from .settle import settle as settle_arrangement
from .solve import Solver, part_volume, solve_multistart
from .voxel import build_poses, suggest_pitch

ORIENTATION_PRESETS = {
    # name: (n_random, use_rest, spins)
    "axis": (0, False, 0),
    "rest": (0, True, 4),
    "fine": (24, True, 6),
    "full": (64, True, 8),
}


@dataclass
class PipelineResult:
    parts: list
    result: object
    poses: list
    pitch: float
    transforms: list = field(default_factory=list)
    overlaps: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    log: list = field(default_factory=list)


def build_orientations(parts, preset="rest", n_random=None, seed=0, log=None):
    """Candidate rotations per part, sharing work between identical parts."""
    n_rand, use_rest, spins = ORIENTATION_PRESETS[preset]
    if n_random is not None:
        n_rand = n_random

    cache = {}
    rotations = []
    for part in parts:
        key = part.shape_key
        if key and key in cache:
            rotations.append(cache[key])
            continue
        rots = candidate_rotations(part.mesh, n_random=n_rand,
                                   use_rest=use_rest, spins=max(spins, 1),
                                   seed=seed)
        if key:
            cache[key] = rots
        rotations.append(rots)
    return rotations


def run(paths, objective="volume", resolution=28, refine_resolution=48,
        pitch=None, clearance=0.0, orientations="rest", n_random=None,
        container=None, time_budget=120.0, seed=0, split_solids=False,
        refine=True, settle=True, settle_resolution=96, settle_orient=None,
        contact_weight=0.0, workers=1, verbose=True, on_phase=None):
    """Load, orient, pack, refine and settle.  Returns a PipelineResult."""
    log = []

    def say(msg):
        log.append(msg)
        if verbose:
            print(msg, flush=True)

    timings = {}
    t0 = time.time()
    parts = load_parts(paths, split_solids=split_solids)
    timings["load"] = time.time() - t0
    if len(parts) < 2:
        raise ValueError("need at least two parts to pack")

    meshes = [p.mesh for p in parts]
    say("loaded %d parts in %.1fs" % (len(parts), timings["load"]))

    coarse_pitch = pitch if pitch else suggest_pitch(meshes, resolution)
    # Half the gap on each part: both masks are dilated, so they stay
    # disjoint only once the true solids are two dilations apart.
    clearance_voxels = (int(np.ceil(clearance / (2 * coarse_pitch)))
                        if clearance > 0 else 0)

    t0 = time.time()
    rotations = build_orientations(parts, orientations, n_random, seed)
    poses = [build_poses(p.mesh, r, coarse_pitch,
                         clearance_voxels=clearance_voxels)
             for p, r in zip(parts, rotations)]
    timings["voxelize"] = time.time() - t0
    say("orientations: %s   (pitch %.2f mm, %.1fs)"
        % (", ".join(str(len(p)) for p in poses), coarse_pitch,
           timings["voxelize"]))

    # Budget split: the coarse pass earns most of it, since it is where
    # arrangements are cheap enough to explore properly.  The settle is
    # given a small slice of its own because it buys more per second than
    # either search pass does -- but it is also usually done long before it
    # spends the slice.
    # A fixed container is answered at one pitch, so there is no refinement
    # to reserve for -- and reserving it anyway is how a --container run
    # came to spend only two thirds of the budget it was given.
    refine = refine and container is None
    settle_share = 0.10 if settle else 0.0
    refine_share = (0.35 - settle_share / 2) if refine else 0.0
    coarse_budget = time_budget * (1.0 - refine_share - settle_share)

    t0 = time.time()
    if container is not None:
        result = _solve_fixed_container(poses, meshes, container, objective,
                                        seed, coarse_budget, contact_weight, say)
    else:
        # Fan out independent solve chains: different seeds land in
        # different basins, and that is the axis of this problem that
        # actually scales with cores.
        # Chains run concurrently, so each gets the whole coarse budget:
        # roughly a third to find a starting box, the rest to squeeze it.
        result = solve_multistart(
            poses, meshes=meshes, seed=seed,
            free_budget=max(3.0, coarse_budget * 0.35),
            budget=max(5.0, coarse_budget * 0.6),
            workers=workers, chains=max(workers, 3),
            contact_weight=contact_weight, say=say)
        say("  coarse best %s mm   vol %.4g   density %4.1f%%"
            % (" x ".join("%7.1f" % v for v in result.extents), result.volume,
               100 * result.density))
    timings["coarse"] = time.time() - t0
    if on_phase is not None:
        on_phase("coarse", _assemble(parts, result))

    kept_rotations = [[p.rotation for p in ps] for ps in poses]
    final_poses = poses
    final_pitch = coarse_pitch

    if refine:
        t0 = time.time()
        fine_pitch = pitch / 1.7 if pitch else suggest_pitch(meshes, refine_resolution)
        fine_clearance = (int(np.ceil(clearance / (2 * fine_pitch)))
                          if clearance > 0 else 0)
        fine_poses = [build_poses(p.mesh, r, fine_pitch, dedup=False,
                                  clearance_voxels=fine_clearance)
                      for p, r in zip(parts, kept_rotations)]
        refined = _refine(fine_poses, meshes, result, seed,
                          time_budget * refine_share, contact_weight,
                          workers, say)
        timings["refine"] = time.time() - t0
        if refined is not None and refined.volume < result.volume:
            result = refined
            final_poses = fine_poses
            final_pitch = fine_pitch
            if on_phase is not None:
                on_phase("refine", _assemble(parts, result, fine_pitch))
        else:
            say("  refine      no improvement, keeping the coarse result")

    if settle:
        # Shake the box down.  The search leaves every contact loose by up
        # to a voxel of its own pitch, and no move in its search space can
        # take that back -- but a local re-seat on a finer lattice can, in
        # seconds rather than minutes.
        t0 = time.time()
        settled = settle_arrangement(
            meshes, result.packer, result.packing,
            resolution=settle_resolution, clearance=clearance,
            budget=max(5.0, time_budget * settle_share), workers=workers,
            orient=settle_orient, say=say)
        timings["settle"] = time.time() - t0
        if settled is not None:
            s_packer, s_packing, s_ext = settled
            result = _settled_result(result, s_packer, s_packing, s_ext)
            final_poses = s_packer.poses
            final_pitch = s_packer.pitch
            if on_phase is not None:
                on_phase("settle", _assemble(parts, result, final_pitch))

    transforms = [None] * len(parts)
    for pl in result.packing.placements:
        pose = result.packer.poses[pl.part_index][pl.pose_index]
        transforms[pl.part_index] = pose.matrix(pl.offset)

    overlaps = verify_packing(result.packer, result.packing)

    return PipelineResult(parts=parts, result=result, poses=final_poses,
                          pitch=final_pitch, transforms=transforms,
                          overlaps=overlaps, timings=timings, log=log)


def _settled_result(previous, packer, packing, extents):
    """The same Result, re-pointed at the settled arrangement.

    ``config`` is dropped deliberately: an (order, pose_choice) pair only
    reconstructs an arrangement by replaying the constructive placer, and
    the settled positions are not something that replay can produce.
    """
    from .solve import Result

    vol = float(np.prod(extents))
    container = None
    if packer.bounded:
        container = np.asarray(packer.container_v) * packer.pitch
    return Result(extents=np.asarray(extents, dtype=float), volume=vol,
                  density=previous.lower_bound / vol,
                  packing=packing, packer=packer,
                  history=previous.history, lower_bound=previous.lower_bound,
                  container=container, objective=packer.objective,
                  config=None, traces=previous.traces,
                  winner_packing=packing)


def _assemble(parts, result, pitch=None):
    """Package a result as a PipelineResult, for snapshots mid-run."""
    transforms = [None] * len(parts)
    for pl in result.packing.placements:
        pose = result.packer.poses[pl.part_index][pl.pose_index]
        transforms[pl.part_index] = pose.matrix(pl.offset)
    return PipelineResult(
        parts=parts, result=result, poses=result.packer.poses,
        pitch=pitch if pitch is not None else result.packer.pitch,
        transforms=transforms,
        overlaps=verify_packing(result.packer, result.packing))


def _solve_fixed_container(poses, meshes, container, objective, seed,
                           budget, contact_weight, say):
    """User supplied the box: answer whether the parts fit, and how well."""
    from .solve import Result

    packer = Packer(poses, objective=objective if objective in ("height", "fit")
                    else "volume", container=container,
                    contact_weight=contact_weight)
    search = Search(packer, seed=seed)

    # Time one arrangement rather than assuming what one costs.  An
    # iteration count derived from a guess is what made --time advisory
    # here: at 80 ms an evaluation it overran a small budget several times
    # over, while the 600-iteration cap left a large one unspent.  The
    # count still has to be right, not just bounded -- the annealing
    # temperature is scheduled against it, so a loop cut short by the
    # clock alone would never cool.
    starts = 4
    probe = time.time()
    search.evaluate(tuple(range(len(poses))), tuple(0 for _ in poses))
    per_eval = max(time.time() - probe, 1e-4)
    iters = int(min(20000, max(60, (budget / starts) / per_eval)))

    best = search.run(starts=starts, iterations=iters,
                      deadline=time.time() + budget)
    pv = part_volume(poses, meshes)

    if not best.packing.feasible:
        say("  container   DOES NOT FIT (%d of %d parts placed)"
            % (len(best.packing.placements), len(poses)))
    else:
        ext = best.packing.extents
        say("  container   FITS   used %s mm of %s"
            % (" x ".join("%.1f" % v for v in ext),
               " x ".join("%.1f" % v for v in np.asarray(container, float))))
    ext = best.packing.extents
    vol = float(np.prod(ext)) if best.packing.feasible else float("inf")
    return Result(extents=ext, volume=vol,
                  density=pv / vol if np.isfinite(vol) else 0.0,
                  packing=best.packing, packer=packer, lower_bound=pv)


def _refine(fine_poses, meshes, coarse, seed, budget, contact_weight,
            workers, say):
    """Re-solve at the fine pitch, starting from the coarse arrangement.

    The coarse configuration is replayed under the objective and container
    it was built with -- replaying it under anything else reconstructs a
    different arrangement, and the fine pass would then start from a worse
    place than the one it is supposed to be improving on.
    """
    if coarse.config is None:
        return None
    order, choice = coarse.config
    if len(order) != len(fine_poses):
        return None

    # A generous cap, actually bounded by the clock below.  Deriving an
    # iteration count from an assumed cost per evaluation is what made
    # --time advisory rather than binding: an evaluation costs an order of
    # magnitude more at the fine pitch than at the coarse one.
    iterations = 5000
    deadline = time.time() + budget * 0.35

    # Try the coarse box itself as the container first.  The fine masks are
    # strictly tighter than the coarse ones, so if anything fits there it
    # fits here, and starting inside that box means the fine pass can only
    # match or beat the coarse answer instead of wandering off to a
    # different arrangement that happens to be worse.  A voxel and a half
    # of slack is allowed, because the fine lattice quantises positions
    # differently and an exactly-zero-slack box is often unreachable.
    fine_pitch = fine_poses[0][0].pitch
    near = np.asarray(coarse.extents, dtype=float) + 1.5 * fine_pitch
    attempts = [(coarse.objective, np.asarray(coarse.extents, dtype=float),
                 iterations),
                (coarse.objective, near, iterations),
                ("volume", near, iterations),
                (coarse.objective, coarse.container, iterations),
                ("compact", None, iterations)]

    seeded = None
    packer = None
    for objective, container, iters in attempts:
        packer = Packer(fine_poses, objective=objective, container=container,
                        contact_weight=contact_weight)
        seeded = Search(packer, seed=seed + 7).anneal(
            tuple(order), tuple(choice), iterations=iters, deadline=deadline)
        if seeded.packing.feasible:
            break
    if seeded is None or not seeded.packing.feasible:
        return None

    ext = np.asarray(seeded.packing.extents, dtype=float)
    say("  fine pack   %s mm   vol %.4g"
        % (" x ".join("%7.1f" % v for v in ext), float(np.prod(ext))))

    solver = Solver(fine_poses, meshes=meshes, seed=seed + 7, verbose=False,
                    contact_weight=contact_weight)
    solver._say = say
    sq_packer, sq_best, sq_ext, history = solver.squeeze(
        ext, budget=budget * 0.65, starts=2, iterations=50, workers=workers,
        warm=[(seeded.order, seeded.pose_choice)])

    if sq_best is not None and float(np.prod(sq_ext)) < float(np.prod(ext)):
        return _as_result(sq_packer, sq_best, sq_ext, solver.part_volume, history)
    return _as_result(packer, seeded, ext, solver.part_volume, [])


def _as_result(packer, sol, ext, pv, history):
    from .solve import Result
    vol = float(np.prod(ext))
    container = None
    if packer.bounded:
        container = np.asarray(packer.container_v) * packer.pitch
    return Result(extents=ext, volume=vol, density=pv / vol,
                  packing=sol.packing, packer=packer, history=history,
                  lower_bound=pv, container=container,
                  objective=packer.objective,
                  config=(tuple(sol.order), tuple(sol.pose_choice)))
