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
tests/               correctness tests and the sample-part generator
examples/compare.py  this packer against the simpler alternatives
```

`python tests/make_samples.py` regenerates the nine sample STEP parts.
