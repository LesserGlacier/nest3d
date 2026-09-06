"""Two-phase solver: grow a box freely, then squeeze it.

Phase 1 (free)      packs with no container at all, under the "compact"
                    objective, to find *some* box that works and, more
                    usefully, a sensible aspect ratio to aim at.

Phase 2 (squeeze)   repeatedly proposes a smaller container and asks
                    whether the parts fit in it.  Feasibility is decided by
                    the same annealed constructive packer, now with the
                    container's limits masking out any placement that would
                    burst the box.

The split matters.  Minimising the box directly is what the user asked
for, but a free-growth packer optimises the box only one part at a time
and cannot undo an early bad commitment.  Asking "do these fit in 120 x
120 x 115?" is a much better-posed question: every part now competes for
the same fixed space, so the packer is pushed into filling hollows instead
of hanging parts off the end.  Binary search over the box size turns that
feasibility oracle back into an optimiser.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .pack import Packer
from .search import Search


@dataclass
class Result:
    extents: np.ndarray
    volume: float
    density: float
    packing: object
    packer: object
    history: list = field(default_factory=list)
    lower_bound: float = 0.0
    container: object = None
    config: object = None
    objective: str = "volume"

    @property
    def efficiency(self) -> float:
        """How close the box is to the volumetric lower bound."""
        return self.lower_bound / self.volume if self.volume else 0.0


def aspect_candidates(box, extra=6):
    """Unit-volume aspect ratios to try, seeded from a known-good box."""
    box = np.asarray(box, dtype=float)
    box = box / np.prod(box) ** (1.0 / 3.0)
    cube = np.ones(3)

    cands = [box, cube]
    # Blends between the free-pack shape and a cube: the truth is usually
    # between "whatever fell out" and "as cubic as the parts allow".
    for t in np.linspace(0.25, 0.75, 3):
        cands.append(box * (1 - t) + cube * t)
    # Mild stretches of the best shape, in case a slab or a column suits
    # these particular parts better.
    for k in range(3):
        for f in (1.2, 0.83):
            v = box.copy()
            v[k] *= f
            cands.append(v)

    out = []
    seen = []
    for c in cands[: 2 + extra + 6]:
        c = np.sort(c)[::-1]
        c = c / np.prod(c) ** (1.0 / 3.0)
        if not any(np.allclose(c, s, rtol=0.02) for s in seen):
            seen.append(c)
            out.append(c)
    return out


def part_volume(poses_per_part, meshes=None):
    if meshes is not None:
        return float(sum(abs(m.volume) for m in meshes))
    return float(sum(p[0].filled * p[0].pitch ** 3 for p in poses_per_part))


@dataclass
class _AspectJob:
    shape: np.ndarray
    hi: float
    lo: float
    seed: int
    starts: int
    iterations: int
    steps: int
    min_gain: float
    shrink: float
    contact_weight: float
    warm: list | None
    deadline: float


def _search_aspect(poses, job):
    """Shrink one aspect ratio as far as it will go.  Runs in a worker.

    Descend first, bisect second.  Bisecting straight away between the
    known-good box and the volumetric lower bound sounds tidy, but that
    lower bound is wildly optimistic for irregular parts -- the first few
    midpoints ask for 60-odd per cent density and simply fail, burning a
    full annealing run each time.  Stepping down by a few per cent at a
    time banks a real improvement on every success, and only once
    something has actually failed is there a bracket worth bisecting.
    """
    hi, lo = job.hi, job.lo
    warm = list(job.warm) if job.warm else None
    best_ext = None
    best_conf = None
    best_dims = None
    bracketed = False
    t0 = time.time()

    for step in range(job.steps):
        if time.time() - t0 > job.deadline:
            break
        if hi / max(lo, 1e-9) < 1.0 + job.min_gain:
            break

        target = math.sqrt(hi * lo) if bracketed else hi * (1.0 - job.shrink)
        target = max(target, lo * (1.0 + 1e-6))
        dims = job.shape * (target ** (1.0 / 3.0))

        packer = Packer(poses, objective="volume", container=dims,
                        contact_weight=job.contact_weight)
        sol = Search(packer, seed=job.seed + step).run(
            starts=job.starts, iterations=job.iterations, warm=warm)

        if sol is None or not sol.packing.feasible:
            lo = target
            bracketed = True
            continue

        # The achieved box is usually tighter than the container allowed,
        # so take what was actually achieved and carry it forward.
        ext = np.asarray(sol.packing.extents, dtype=float)
        vol = float(np.prod(ext))
        hi = min(target, vol)
        if best_ext is None or vol < float(np.prod(best_ext)):
            best_ext = ext
            best_conf = (tuple(sol.order), tuple(sol.pose_choice))
            best_dims = dims
            warm = [best_conf]

    if best_ext is None:
        return None
    # The container travels with the configuration: replaying the same
    # order and orientations without it would construct a different
    # arrangement, because the container is what masked out the placements
    # that would have burst the box.
    return best_ext, best_conf, float(np.prod(best_ext)), best_dims


_WORKER_POSES = None


def _init_worker(poses):
    global _WORKER_POSES
    _WORKER_POSES = poses


def _worker(job):
    try:
        return _search_aspect(_WORKER_POSES, job)
    except Exception:
        return None


def _run_jobs(poses, jobs, workers):
    if workers <= 1 or len(jobs) == 1:
        return [_search_aspect(poses, j) for j in jobs]
    from concurrent.futures import ProcessPoolExecutor
    n = min(workers, len(jobs))
    with ProcessPoolExecutor(max_workers=n, initializer=_init_worker,
                             initargs=(poses,)) as pool:
        return list(pool.map(_worker, jobs))


def _orient_aspect(aspect, box):
    """Assign the aspect's three magnitudes to axes the way ``box`` is shaped."""
    order = np.argsort(np.asarray(box, dtype=float))[::-1]
    out = np.empty(3)
    out[order] = np.sort(np.asarray(aspect, dtype=float))[::-1]
    return out


class Solver:
    def __init__(self, poses_per_part, meshes=None, seed=0, verbose=True,
                 contact_weight=0.0):
        self.poses = poses_per_part
        self.meshes = meshes
        self.seed = seed
        self.verbose = verbose
        self.contact_weight = contact_weight
        self.part_volume = part_volume(poses_per_part, meshes)
        self.log = []

    def _say(self, msg):
        self.log.append(msg)
        if self.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------------
    def free_phase(self, starts=3, iterations=120, objective="compact"):
        packer = Packer(self.poses, objective=objective,
                        contact_weight=self.contact_weight)
        search = Search(packer, seed=self.seed)
        best = search.run(starts=starts, iterations=iterations)
        if best is None or not best.packing.feasible:
            raise RuntimeError("free packing failed; check the input geometry")
        ext = best.packing.extents
        self._say("  free pack   %s mm   vol %.4g   density %4.1f%%"
                  % (_fmt(ext), float(np.prod(ext)),
                     100 * self.part_volume / float(np.prod(ext))))
        return packer, best

    # ------------------------------------------------------------------
    def feasible(self, dims, starts=2, iterations=60, seed_offset=0, warm=None):
        """Can the parts be packed inside a box of these dimensions?"""
        packer = Packer(self.poses, objective="volume", container=dims,
                        contact_weight=self.contact_weight)
        search = Search(packer, seed=self.seed + 1000 + seed_offset)
        best = search.run(starts=starts, iterations=iterations, warm=warm)
        if best is None or not best.packing.feasible:
            return None
        return packer, best

    def squeeze(self, box, budget=90.0, starts=2, iterations=60,
                min_gain=0.004, warm=None, workers=1, rounds=2,
                steps=10, shrink=0.05):
        """Shrink the box by binary search on volume, over several aspects.

        Each aspect ratio is an independent binary search, so they fan out
        across processes.  Rounds let a later round start from whatever the
        best aspect achieved in the earlier one.
        """
        t0 = time.time()
        best_ext = np.asarray(box, dtype=float)
        best_conf = warm[0] if warm else None
        best_dims = None
        history = []

        for rnd in range(rounds):
            remaining = budget - (time.time() - t0)
            if remaining <= 1.0:
                break
            aspects = aspect_candidates(best_ext)
            per_round = remaining / max(1, rounds - rnd)
            # Parallel jobs each get the whole round; serial ones share it.
            deadline = per_round if workers > 1 else per_round / len(aspects)
            jobs = [
                _AspectJob(
                    shape=_orient_aspect(aspect, best_ext),
                    hi=float(np.prod(best_ext)),
                    lo=self.part_volume,
                    seed=self.seed + 1000 + rnd * 101 + ai * 17,
                    starts=starts, iterations=iterations,
                    steps=steps, min_gain=min_gain, shrink=shrink,
                    contact_weight=self.contact_weight,
                    warm=[best_conf] if best_conf else None,
                    deadline=deadline,
                )
                for ai, aspect in enumerate(aspects)
            ]

            outcomes = _run_jobs(self.poses, jobs, workers)
            improved = False
            for out in outcomes:
                if out is None:
                    continue
                ext, conf, vol, dims = out
                history.append((tuple(np.round(ext, 2)), vol))
                if vol < float(np.prod(best_ext)) * (1 - 1e-9):
                    best_ext = np.asarray(ext, dtype=float)
                    best_conf = conf
                    best_dims = dims
                    improved = True

            if improved:
                self._say("  squeeze %-2d  %s mm   vol %.4g   density %4.1f%%"
                          % (rnd + 1, _fmt(best_ext), float(np.prod(best_ext)),
                             100 * self.part_volume / float(np.prod(best_ext))))
            else:
                break

        # Only materialise when an aspect actually won.  ``best_conf`` may
        # still be the warm start handed in, and that was built under a
        # different objective and container, so replaying it here would
        # construct something else entirely.
        if best_dims is None or best_conf is None:
            return None, None, best_ext, history

        packer = Packer(self.poses, objective="volume", container=best_dims,
                        contact_weight=self.contact_weight)
        sol = Search(packer, seed=self.seed).evaluate(tuple(best_conf[0]),
                                                      tuple(best_conf[1]))
        if not sol.packing.feasible:
            return None, None, best_ext, history
        return packer, sol, sol.packing.extents, history

    # ------------------------------------------------------------------
    def solve(self, free_starts=3, free_iterations=120, squeeze_budget=90.0,
              squeeze_starts=2, squeeze_iterations=60, workers=1, rounds=3):
        packer, best = self.free_phase(starts=free_starts,
                                       iterations=free_iterations)
        ext = best.packing.extents

        sq_packer, sq_best, sq_ext, history = self.squeeze(
            ext, budget=squeeze_budget, starts=squeeze_starts,
            iterations=squeeze_iterations, workers=workers, rounds=rounds,
            warm=[(best.order, best.pose_choice)])

        if sq_best is not None and float(np.prod(sq_ext)) < float(np.prod(ext)):
            packer, best, ext = sq_packer, sq_best, sq_ext

        vol = float(np.prod(ext))
        container = None
        if packer.bounded:
            container = np.asarray(packer.container_v) * packer.pitch
        return Result(extents=ext, volume=vol,
                      density=self.part_volume / vol,
                      packing=best.packing, packer=packer,
                      history=history, lower_bound=self.part_volume,
                      container=container, objective=packer.objective,
                      config=(tuple(best.order), tuple(best.pose_choice)))


def _fmt(ext):
    return " x ".join("%7.1f" % v for v in np.asarray(ext, dtype=float))


# ---------------------------------------------------------------------------
# Independent solve chains in parallel
# ---------------------------------------------------------------------------

def _chain(poses, part_vol, seed, budget, contact_weight, free_starts,
           free_iterations, squeeze_starts, squeeze_iterations, rounds):
    solver = Solver(poses, meshes=None, seed=seed, verbose=False,
                    contact_weight=contact_weight)
    solver.part_volume = part_vol
    res = solver.solve(free_starts=free_starts,
                       free_iterations=free_iterations,
                       squeeze_budget=budget,
                       squeeze_starts=squeeze_starts,
                       squeeze_iterations=squeeze_iterations,
                       workers=1, rounds=rounds)
    container = None if res.container is None else [float(v) for v in res.container]
    return float(res.volume), res.config, container, res.objective


def _chain_worker(args):
    try:
        return _chain(_WORKER_POSES, *args)
    except Exception:
        return None


def solve_multistart(poses, meshes=None, seed=0, budget=60.0, workers=1,
                     chains=None, contact_weight=0.0, free_starts=3,
                     free_iterations=120, squeeze_starts=2,
                     squeeze_iterations=60, rounds=2, say=None):
    """Run several independent solve chains and keep the best.

    Each chain is a complete free-then-squeeze solve from its own random
    seed.  Because the constructive packer is a greedy heuristic wrapped in
    annealing, different seeds land in genuinely different basins, so a
    wide fan-out buys more than a single long chain does -- and it is the
    one axis of this problem that scales linearly with cores.
    """
    pv = part_volume(poses, meshes)
    chains = chains or max(workers, 1)
    args = [(pv, seed + 977 * k, budget, contact_weight, free_starts,
             free_iterations, squeeze_starts, squeeze_iterations, rounds)
            for k in range(chains)]

    if workers <= 1 or chains == 1:
        outs = [_chain(poses, *a) for a in args]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(workers, chains),
                                 initializer=_init_worker,
                                 initargs=(poses,)) as pool:
            outs = list(pool.map(_chain_worker, args))

    outs = [o for o in outs if o is not None]
    if not outs:
        raise RuntimeError("every solve chain failed")

    # Replay every chain here and rank on what actually reproduces.  A
    # configuration only reconstructs the same arrangement under the same
    # objective and container it was built with, and ranking on the number
    # a worker reported would trust a replay that never happened.
    best = None
    replayed = []
    for _vol, config, container, objective in outs:
        packer = Packer(poses, objective=objective, container=container,
                        contact_weight=contact_weight)
        sol = Search(packer, seed=seed).evaluate(tuple(config[0]),
                                                 tuple(config[1]))
        if not sol.packing.feasible:
            continue
        ext = np.asarray(sol.packing.extents, dtype=float)
        vol = float(np.prod(ext))
        replayed.append(vol)
        if best is None or vol < best[0]:
            best = (vol, ext, sol, packer, container, objective, config)

    if best is None:
        raise RuntimeError("no solve chain produced a reproducible packing")

    if say is not None:
        vols = sorted(replayed)
        say("  %d chains    best %.4g   median %.4g   worst %.4g"
            % (len(vols), vols[0], vols[len(vols) // 2], vols[-1]))

    vol, ext, sol, packer, container, objective, config = best
    return Result(extents=ext, volume=vol, density=pv / vol,
                  packing=sol.packing, packer=packer, lower_bound=pv,
                  container=container, objective=objective, config=config)
