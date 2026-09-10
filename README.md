# nest3d

Find the smallest axis-aligned bounding box that holds a set of irregular
3D solids, allowing every rotation and position. Reads STEP directly.

```
python -m nest3d parts/*.step --time 300 --workers 12 --out results/
```

```
  BOUNDING BOX   111.25 x 126.21 x 120.71 mm
  volume         1.695e+06 mm^3
  part volume    7.677e+05 mm^3  (9 parts)
  fill density   45.3%
  interference   none (verified pairwise on the voxel masks)
```

It writes `packing.json` (a 4x4 transform per part), `packed.step` (the
arrangement as a real B-rep assembly you can open in CAD) and
`packed.stl`.

## Why it is built this way

**The problem is NP-hard**, so this is a heuristic that gets a good answer
in a set time budget, not a proof of optimality. Everything below is about
making the heuristic strong rather than pretending it isn't one.

**Voxels, not convex hulls.** The parts are concave, so the whole prize is
letting one part sit inside another's hollow. Convex-hull or
bounding-box packing throws that away before the search even starts. An
occupancy grid keeps arbitrary concavity and makes overlap a bitwise AND.

**Every position is tested, not a shortlist of corners.** For each part
the packer computes the overlap between the pile and the part's mask at
*every* integer offset at once, as an FFT cross-correlation. Zero-overlap
offsets are exactly the collision-free placements. That is what lets a
part drop into a pocket nobody nominated as a candidate point — the usual
"extreme point" heuristics can only place parts at corners of what is
already down.

**Minimum volume is a trap for a greedy placer.** Extending the box along
its longest axis costs the smallest cross-section, so the cheapest next
step is always to make the rod longer, and a pure min-volume greedy
degenerates into a chain. On the sample parts it produced a
75 x 153 x 208 box. Charging for the longest dimension as well (the
`compact` objective) gives 137 x 134 x 111 for the same search effort.

**Then squeeze.** "Minimise the box" is badly posed for a one-part-at-a-time
placer; "do these nine fit in 120 x 120 x 115?" is well posed, because
every part now competes for the same fixed space. The solver packs freely
once to get a starting box and an aspect ratio, then repeatedly proposes
smaller containers and asks the same annealed packer whether they work.
Descend a few per cent at a time first, and only bisect once something has
actually failed — bisecting straight to the volumetric lower bound just
burns annealing runs on boxes that ask for 60% density.

**Then shake it down.** Everything above happens on the search's own
lattice, where a part's mask is a conservative superset about 1.35x its
true volume and positions are quantised to the pitch. Both skins sit
between any two touching parts, so every contact in a finished pack is
loose by up to a voxel — the arrangement really would condense if you
could pick the box up and shake it. The search cannot take that back: a
placement is a deterministic function of the order and the poses, so
nothing in its search space nudges one part by two millimetres, and
re-running the whole search at a fine pitch costs the cube of the
resolution ratio.

So the last pass searches nothing at all. It keeps the arrangement and
every orientation, re-voxelises at 96 voxels across the largest part
(mask 1.10x true), and lets each part slide to the best position within a
voxel or two of where it stands, the parts on the box's faces first. On
the sample parts that is 5 to 7 per cent of the box for about two seconds
of work — more than the fine-pitch refinement buys in a minute. It is
monotone by construction: a part's current seat is always one of the
candidates, so the box can only shrink.

**Search order and orientation, in parallel.** The placer is deterministic
given a placement order and one pose per part, so that pair *is* the search
space, explored by simulated annealing. Independent chains from different
seeds land in different basins, so the fan-out across cores buys more than
one long chain — this is the axis that scales with hardware.

**Coarse then fine.** The search runs at a pitch of ~28 voxels across the
largest part, where an arrangement costs about 50 ms to evaluate. The
winner is then re-voxelised at ~48 and tightened, starting inside the
coarse box so it can only match or beat it.

## The no-overlap guarantee

Each part's collision mask is a **superset** of the true solid: a voxel is
marked when a triangle genuinely intersects its cube, by the exact
Akenine-Möller separating-axis test, and enclosed cavities are then filled.
Since `mask ⊇ solid`, two parts with disjoint masks have disjoint solids.
The packer therefore cannot output an interfering arrangement — it can only
be slightly *looser* than the true optimum, by under one pitch per contact.

Reported box dimensions are measured from the exact mesh vertices, never
from the voxels, so that looseness never inflates the reported answer.

The settle moves parts on a lattice the search never saw, so it re-earns
the guarantee there rather than inheriting it: the fine masks are
conservative in exactly the same way, and a part only ever moves to an
offset whose overlap with the rest of the pile is zero. That holds for
every lattice it tries, and the arrangement it reports is whichever one
came out smallest.

`tests/test_nest3d.py` checks this end to end by taking the packed result
back to the triangle meshes and running exact boolean intersections on
every pair — for the settled arrangement as well as the packed one.

A through-bore stays open during the fill, so another part can genuinely
nest inside a tube. Only fully enclosed cavities are filled, which is right
— nothing can reach them anyway.

## Options that matter

| flag | what it does |
|---|---|
| `--time N` | total search budget in seconds. The single biggest quality knob. |
| `--workers N` | parallel solve chains. Use most of your cores. |
| `--resolution N` | voxels across the largest part during search (default 28). Cost is roughly N³. |
| `--refine-resolution N` | pitch for the final tightening pass (default 48). |
| `--settle-resolution N` | first lattice for the closing settle (default 96). Finer ones are tried alongside it, one per worker, and the smallest box wins — so `--workers` buys settle quality too. |
| `--no-settle` | skip the settle and report the arrangement exactly as the search left it. |
| `--settle-orient` | let the settle re-choose each part's orientation, not just its position. Off by default, and see below for why. |
| `--clearance MM` | minimum gap to hold between parts. Half is applied to each part, so the gap you ask for is the gap you get. |
| `--orientations` | `axis` = the 24 box rotations; `rest` (default) adds stable resting poses; `fine`/`full` add sampled SO(3) for genuinely oblique placements. |
| `--container W D H` | fixed box: answers "do these fit?" instead of minimising. `inf` leaves an axis free. |
| `--objective height` | with `--container W D inf`, minimise stack height in a fixed footprint — the build-plate / shelf case. |
| `--contact 1` | tie-break toward placements that hug the pile. Slower, sometimes tighter. |
| `--split-solids` | treat each solid inside one STEP file as a separate part. |

## Accuracy and cost

The voxel pitch is the one real trade-off. At 28 voxels across the largest
part each mask is about 1.2–1.7x the true part volume (the conservative
skin); at 48 it is 1.08–1.34. Finer means a tighter answer and cubically
more time, which is why the search runs coarse, the refinement runs fine,
and only the settle — which evaluates a few positions per part instead of
thousands of arrangements — runs finer still.

Which fine lattice settles best is not predictable, though. A finer one has
a thinner skin to give back, but it is also a different lattice: the parts
round onto it differently and the sweep converges somewhere else. Over four
arrangements of the sample parts the finest lattice won twice, the middle
two once each, and 96 — the pitch the settle used to run at on its own —
never. So the settle runs several, one per worker, and keeps the smallest
box; on those four that is worth 0.8 to 1.8 points of density over 96
alone. None of them can return anything worse than the arrangement it was
handed, so trying more only costs cores.

Because the masks are conservative, a reported density of 45% means the box
is genuinely 45% solid part — the slack is real clearance between parts,
not measurement error.

Four fifths of a run is inside the placement correlation, and the mask
half of it is the same transform over and over: a pose is built once and
placed thousands of times, so 77% of the correlations in a nine-part run
ask for a (mask, transform shape) pair that process has already done.
Caching those is worth **1.16x on identical work** -- measured by giving
the solver a budget it cannot reach, so every configuration does the same
work and lands on the same box, agreeing to the last cubic millimetre.
`tools/bench_fixedwork.py` is that measurement:

| cache per process | wall, 24 chains | |
|---|---|---|
| off | 83.4 s | |
| 16 MB | 73.8 s | 1.13x |
| 32 MB (default) | 72.0 s | 1.16x |
| 128 MB | 70.0 s | 1.19x |

The cache is bounded in bytes rather than entries, because the transform
is padded out to the shape the *pile* needs: one mask has as many
transforms as it has neighbourhood sizes, each the size of a pile
transform rather than of the mask. Keeping all of them costs 6.6 GB, and
counting entries is no use when they differ in size by two orders of
magnitude -- 32 of them is 193 MB in one process, and there are two dozen
processes. At the default 32 MB per process the cache takes 60% of the
transforms a perfect one would, against a ceiling of 77%. The table above
is where that default comes from: most of the win is bought by the first
32 MB, and the last 96 MB buys 3% more for four times the memory in every
worker. `NEST3D_FFT_CACHE_MB` moves it.

## Things that were tried and are not here

**A bigger search window for the settle.** Tripling and quintupling the
radius each part may move within changes the answer by exactly nothing on
the sample parts. The settle is not reach-limited; it is stuck at a
*one-part-move* optimum, where the box is held open by parts that each
need another to move first.

**A wall push: force a face inward and relax.** The lattice version of what
a physics engine would do — pull one face of the box in by a voxel, declare
that the box, and let the parts shove each other with Gauss-Seidel
relaxation (each part in turn to the offset that overlaps the others least
while staying inside), randomised restarts from the best state seen when
descent stalls. It works, and it is dominated. Measured on three packed
arrangements, at 160 voxels across the largest part:

| arrangement | sweep only | + tipping | + wall push | + both |
|---|---|---|---|---|
| A | 54.79% | 54.79% | 54.79% | 54.79% |
| B | 54.87% | 55.77% | 55.17% | 55.77% |
| C | 58.26% | 59.25% | 59.19% | 59.25% |

The push beats a bare sweep, never beats tipping, and adds nothing at all
on top of it — for fifteen times the run time against tipping's five. Both
escape the same trap, and tipping is the cheaper way out. Once the sweeps
and the tips have run, a one-voxel shrink on any face is genuinely
infeasible: the relaxation gets within about seventy voxels of overlap and
five times the effort does not close them.

**Re-choosing the orientations during the settle.** Every orientation in a
finished pack was picked by the search, at the coarse pitch, where a
part's mask runs 1.2 to 2.2 times the solid inside it -- so the pose that
scored best was ranked on a shape that is substantially not the part's.
The settle has masks within a few per cent of the solid and could rank
them again. It does exactly that under `--settle-orient`: each part is
re-voxelised at 1.5 and 4 degrees about each lattice axis, in both
directions, re-seated by the same windowed sweep as everything else, and
kept only if the box strictly shrinks. It is monotone for the same reason
the rest of the settle is, and it re-earns the no-overlap guarantee the
same way -- verified by exact pairwise mesh booleans on every arrangement
below, all clear.

It also does not pay. On one arrangement, on one lattice, in isolation, it
is worth a full point of density. Against the whole ladder that collapses
to almost nothing, because the finer rungs were already reaching the same
place by another route. On three arrangements of nine real parts --
thin-walled hollow vessels, which nest into each other and are the case
this tool exists for:

| arrangement | ladder | ladder, every rung tilted |
|---|---|---|
| A | 50.85% (43s) | 51.09% (126s) |
| B | 52.65% (38s) | 53.08% (169s) |
| C | 54.21% (51s) | 54.24% (159s) |

0.23 points for 3.4x the time. The nine sample parts agree, at 0.16 points
for 4.6x -- so this is not an artefact of either part set. And adding
tilted rungs *alongside* the
plain ones is worse than not having them, which is the part worth
remembering: the rungs share one wall-clock deadline, so six dear rungs
take time from six cheap ones that pay more often. At the settle's normal
budget twelve rungs gave 44.67% against the plain six's 44.91%; at five
times that budget, 45.22% against 45.25%. A ladder that keeps the smallest
box cannot lose on merit -- it lost on contention.

There is a plainer symptom of the same thing. Asked for `--settle-orient`
on a 60-second budget, the run reports every part at a clean multiple of 45
degrees -- the settle's 8.8-second slice ran out before a single tilt was
accepted. The pass needs a settle budget several times the default even to
fire, and the default is what a 60-second run gives it.

So the pass stays, behind a flag that is off by default, rather than being
deleted. The effect it corrects is largest for parts whose bounding box
swings hardest on a degree or two of tilt -- long flat plates. Neither part
set measured here is that shape, so the flag is left in reach rather than
thrown away.

**Running the placement correlation on a GPU.** Four fifths of a run's wall
clock is inside `scipy.signal.fftconvolve`, which reads like a
GPU-shaped problem and is not one. The baseline is the trap: the search
already fans out over every core, so the comparison is not one CPU core
against the GPU but *all* of them against it.

Measured on an RTX 4060 against 24 cores, at the (pile, mask) shapes an
actual nine-part run asks for -- `tools/fft_bench.py`:

| pile | mask | 1 core | 24 cores | GPU | vs all |
|---|---|---|---|---|---|
| 46x55x85 | 16x17x29 | 10.72ms | 0.45ms | 0.38ms | 1.2x |
| 50x59x83 | 16x16x28 | 19.58ms | 0.82ms | 0.33ms | 2.5x |
| 48x46x58 | 9x9x15 | 3.72ms | 0.16ms | 0.46ms | 0.3x |
| 40x40x29 | 13x7x8 | 0.56ms | 0.02ms | 0.34ms | 0.1x |

Read the GPU column rather than the ratios: it is flat at 0.3 to 0.5 ms
whatever the size. That is not compute, it is plan lookup and kernel
launch. The transforms here average 1.6e5 voxels, far too small to occupy
the device, so a per-call port pays a fixed floor and gets nothing for it
-- and on the commoner small shapes it is a straight loss.

Batching is the only thing that could amortise that floor, and it tops out
too: per correlation 0.21 ms at a batch of 4, rising to 0.42 ms and
plateauing from 16 up, where the device is genuinely saturated. Against 24
cores' 0.82 ms that is 1.9x on the most favourable shape in the workload.
Through Amdahl, a whole port is worth about 1.5x end to end -- and only
after the annealing chains are restructured into one process stepping in
lockstep on a common padded shape, because placement *within* an
arrangement is sequential and cannot be batched at all.

Of the two cheaper things this section used to point at, one paid and one
did not. The mask side of every correlation was being re-transformed on
every call although the pose had not changed; it is now cached, and it is
worth 1.16x on identical work -- see "Accuracy and cost". The process-pool
startup is the entry below.

None of this is a statement about GPUs in general. It is a statement about
many small transforms on one mid-range card: pack forty parts instead of
nine, or settle at a much finer pitch, and the arrays grow into the regime
where the answer flips.

**Bringing the worker pool up before giving it work.** A
`ProcessPoolExecutor` is not ready when it is constructed. Workers are
spawned as tasks are submitted, and each then re-imports nest3d and
unpickles the poses. Instrumented on a nine-part run at 24 workers, the
first worker was ready in 0.6 s and the last in 27.2 s -- more than half
of a 52 s phase spent at less than full width. An idle pool of the same
size costs 4.5 s, so the other 22 s is the workers still importing while
the ones already up saturate the machine. That reads like 20 s to be had
for nothing, and it was where "roughly 25 s of every run is fixed
process-pool startup" came from.

It is not. Holding every worker at a common instant before submitting any
real work does exactly what it promises -- the ramp shortens, and the
later pools come up in about 3 s instead of 8 -- and the run gets
*slower*. On identical work it is 0.92x, and at a fixed 60 s budget it
costs 1.4 density points over six seeds, never winning on any of them:

| | density, mean of 6 seeds | wall, identical work |
|---|---|---|
| as it is | 53.42% | 63.4 s |
| pool warmed first | 51.99% | 68.6 s (0.92x) |

(Density at a 60 s budget and 24 workers; wall on fixed work, at the
lighter settings `bench_fixedwork.py` defaults to. A longer hold was worse
on both counts.)

The reason is that a chain is given a *duration*, not a deadline, and the
run keeps the best of twenty-four of them rather than the mean. Left
alone, the workers arrive staggered, so the early chains run against a
half-empty machine and get more iterations inside their budget than they
would have otherwise. Warming the pool takes that away: all twenty-four
start together, contend equally, and every one of them does less. The
ramp was not waste, it was a head start -- and the fan-out is only as good
as its luckiest chain.

The same reasoning says what would change the answer: chains cut off by a
common absolute deadline, or many more chains than cores, would both make
the staggering worthless and the warm-up worth having. Neither is how this
runs today.

Reusing one pool across the rounds of a squeeze, rather than building a
fresh one per round, was measured at the same time and is not here either
-- for the duller reason that it changes nothing. A squeeze almost always
stops after one round, and on the runs where refine took two, the second
wanted more workers than the first, so the pool was rebuilt regardless.

**Which is also the answer to "why not simulate the physics".** A rigid-body
sim buys exactly one thing over the geometry here — parts moving together,
under contact, instead of one at a time. That is worth having, and the two
cheap versions of it above are already at the point of diminishing returns.
Against that it costs convex decomposition of every part, tens of thousands
of collision steps per trial where the settle evaluates thousands of
candidate positions per part per second, a stochastic result that needs
many trials, and soft contacts that permit interpenetration — which would
forfeit the no-overlap guarantee this tool is built on and need an exact
re-check and a push-apart pass that grows the box again. Settling under
gravity also minimises height along one axis, which is not the objective.

## Install

```
pip install numpy scipy trimesh cadquery-ocp
pip install rtree manifold3d      # only for the verification tests
```

`cadquery-ocp` is OpenCascade, used for reading STEP and writing the packed
assembly back out. Without it, mesh formats (STL/OBJ/PLY/3MF) still work.

## Layout

```
nest3d/geometry.py   STEP + mesh loading, tessellation, STEP/STL export
nest3d/voxel.py      exact conservative voxelisation, the overlap test
nest3d/orient.py     candidate rotation sets
nest3d/pack.py       the constructive placer (FFT correlation + scoring)
nest3d/search.py     simulated annealing over order and orientation
nest3d/solve.py      free phase, squeeze phase, parallel chains
nest3d/settle.py     the closing settle on a much finer lattice
nest3d/pipeline.py   coarse -> fine -> settle orchestration
nest3d/report.py     summary, JSON, exports
tools/               benchmarks and diagnostics, none of them on the
                     import path of a run
tests/               correctness tests and the sample-part generator
examples/compare.py  this packer against the simpler alternatives
```

`python tests/make_samples.py` regenerates the nine sample STEP parts.
