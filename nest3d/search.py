"""Global search over placement order and orientation choice.

The constructive packer is deterministic given a placement order and one
pose per part, so the search space is exactly those two things:

    order         a permutation of the parts
    pose_choice   an index into each part's candidate orientation list

Simulated annealing over that space, from several seeded starts, is the
workhorse.  It is a good fit here because a single move (swapping two
parts, or re-orienting one) changes the resulting box in a way that is
neither smooth nor monotonic, so gradient-style methods have nothing to
hold on to, while the evaluation is cheap enough to afford thousands of
samples.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Solution:
    order: tuple
    pose_choice: tuple
    score: float
    packing: object = None


def seed_orders(poses_per_part, rng, n):
    """Sensible starting permutations, biggest-first being the classic one."""
    n_parts = len(poses_per_part)
    vols = np.array([float(np.prod(p[0].extents)) for p in poses_per_part])
    diags = np.array([float(np.linalg.norm(p[0].extents)) for p in poses_per_part])
    fills = np.array([p[0].filled for p in poses_per_part], dtype=float)

    seeds = [
        tuple(np.argsort(vols)[::-1]),
        tuple(np.argsort(diags)[::-1]),
        tuple(np.argsort(fills)[::-1]),
    ]
    while len(seeds) < n:
        p = list(range(n_parts))
        rng.shuffle(p)
        seeds.append(tuple(p))
    return seeds[:n]


class Search:
    def __init__(self, packer, seed=0, cache=True):
        self.packer = packer
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.cache = {} if cache else None
        self.evaluations = 0

    def evaluate(self, order, pose_choice):
        key = (order, pose_choice)
        if self.cache is not None and key in self.cache:
            return self.cache[key]
        packing = self.packer.pack(list(order), list(pose_choice))
        self.evaluations += 1
        # packing.score already carries the unplaced-part penalty, so an
        # infeasible arrangement compares as merely very bad rather than
        # as a wall the annealer cannot see past.
        result = Solution(order, pose_choice, packing.score, packing)
        if self.cache is not None:
            self.cache[key] = result
        return result

    # ------------------------------------------------------------------
    def _mutate(self, order, pose_choice):
        order = list(order)
        pose_choice = list(pose_choice)
        n = len(order)
        n_poses = [len(p) for p in self.packer.poses]
        multi = [i for i in range(n) if n_poses[i] > 1]

        r = self.rng.random()
        if r < 0.30 and n > 1:
            i, j = self.rng.sample(range(n), 2)
            order[i], order[j] = order[j], order[i]
        elif r < 0.50 and n > 2:
            i = self.rng.randrange(n)
            j = self.rng.randrange(n)
            item = order.pop(i)
            order.insert(j, item)
        elif r < 0.60 and n > 2:
            i, j = sorted(self.rng.sample(range(n), 2))
            order[i:j + 1] = order[i:j + 1][::-1]
        elif multi:
            # Re-orient one or two parts.
            for _ in range(1 if self.rng.random() < 0.7 else 2):
                p = self.rng.choice(multi)
                pose_choice[p] = self.rng.randrange(n_poses[p])
        elif n > 1:
            i, j = self.rng.sample(range(n), 2)
            order[i], order[j] = order[j], order[i]

        return tuple(order), tuple(pose_choice)

    def anneal(self, order, pose_choice, iterations=400, t_start=0.06,
               t_end=0.002, callback=None, deadline=None):
        """Anneal from one start.  Temperatures are fractions of the score.

        ``deadline`` is an absolute time.time() value.  It exists because
        an iteration count is not a time budget: the cost of one evaluation
        varies by an order of magnitude with the voxel pitch, so a loop
        sized for the coarse pass will overrun badly at the fine one.
        """
        cur = self.evaluate(order, pose_choice)
        best = cur

        for it in range(iterations):
            if deadline is not None and time.time() > deadline:
                break
            frac = it / max(iterations - 1, 1)
            temp = t_start * (t_end / t_start) ** frac
            cand_order, cand_poses = self._mutate(cur.order, cur.pose_choice)
            cand = self.evaluate(cand_order, cand_poses)

            if math.isfinite(cand.score):
                delta = (cand.score - cur.score) / max(abs(cur.score), 1e-12)
                if delta <= 0 or self.rng.random() < math.exp(-delta / temp):
                    cur = cand
                if cand.score < best.score:
                    best = cand
            if callback is not None:
                callback(it, iterations, best.score, cur.score)
        return best

    def run(self, starts=4, iterations=400, callback=None, warm=None,
            deadline=None):
        """Anneal from several starts and keep the best.

        ``warm`` is a list of (order, pose_choice) pairs to start from --
        typically the best arrangement found at a looser box or a coarser
        pitch.  Warm starts matter a lot in the squeeze loop, where each
        successive container is only slightly tighter than the last and the
        previous answer is nearly always the right neighbourhood.
        """
        n_parts = len(self.packer.poses)
        entries = []
        for w in (warm or []):
            entries.append((tuple(w[0]), tuple(w[1])))

        for si, order in enumerate(seed_orders(self.packer.poses, self.rng,
                                               max(starts - len(entries), 0))):
            choice = tuple(0 for _ in range(n_parts))
            if si >= 3:
                choice = tuple(self.rng.randrange(len(p))
                               for p in self.packer.poses)
            entries.append((tuple(order), choice))

        best = None
        for si, (order, choice) in enumerate(entries):
            if deadline is not None and time.time() > deadline and best is not None:
                break

            def cb(it, tot, b, c, si=si):
                if callback is not None:
                    callback(si, len(entries), it, tot, b, c)

            result = self.anneal(order, choice, iterations=iterations,
                                 callback=cb, deadline=deadline)
            if best is None or result.score < best.score:
                best = result
        return best
