"""Measure settle variants against the same packed arrangements.

The settle is judged on what it does to an arrangement it was handed, so a
fair comparison holds the arrangement fixed and varies only the settle.
Packing is the expensive half and it is also the noisy half -- different
seeds land in different basins -- so pack once per seed, cache that, and
replay every variant against the cache.

    python tools/bench_settle.py --pack --seeds 0,1,2,3   # fill the cache
    python tools/bench_settle.py --variant base,orient    # compare
"""
from __future__ import annotations

import argparse
import glob
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nest3d import pipeline                              # noqa: E402
from nest3d import settle as settle_mod                  # noqa: E402
from nest3d.solve import part_volume                     # noqa: E402


def cache_dir(explicit=None):
    if explicit:
        return Path(explicit)
    import tempfile
    return Path(tempfile.gettempdir()) / "nest3d_bench"


def pack_one(paths, seed, time_budget, workers, resolution, orientations):
    """One packed, unsettled arrangement."""
    res = pipeline.run(paths, time_budget=time_budget, seed=seed,
                       workers=workers, resolution=resolution,
                       orientations=orientations, settle=False, verbose=False)
    return res


def do_pack(args):
    paths = sorted(glob.glob(args.parts))
    if not paths:
        raise SystemExit("no parts matched %r" % args.parts)
    out = cache_dir(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        t0 = time.time()
        res = pack_one(paths, seed, args.time, args.workers, args.resolution,
                       args.orientations)
        r = res.result
        blob = {
            "meshes": [p.mesh for p in res.parts],
            "packer": r.packer,
            "packing": r.packing,
            "extents": np.asarray(r.extents, dtype=float),
            "part_volume": float(r.lower_bound),
            "seed": seed,
        }
        with open(out / ("arr%02d.pkl" % seed), "wb") as fh:
            pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
        vol = float(np.prod(r.extents))
        print("seed %2d  %s mm  vol %.4g  density %5.2f%%  (%.0fs)"
              % (seed, " x ".join("%7.1f" % v for v in r.extents), vol,
                 100 * r.lower_bound / vol, time.time() - t0), flush=True)


def load_arrangements(args):
    d = cache_dir(args.cache)
    out = []
    for path in sorted(d.glob("arr*.pkl")):
        with open(path, "rb") as fh:
            out.append(pickle.load(fh))
    if not out:
        raise SystemExit("cache is empty -- run with --pack first")
    return out


# name -> kwargs handed to settle_mod.settle.  ``workers`` is part of the
# variant because the ladder's width is exactly what is under test: the
# re-oriented rungs are extra rungs, not a replacement for the plain ones,
# so "does orient pay" and "does a wider ladder pay" are the same question
# and have to be asked together.
VARIANTS = {
    "base":     dict(workers=6, orient=False),   # the ladder as it shipped
    "orient":   dict(workers=6, orient=True),    # the same rungs, all tilted
    "ladder12": dict(workers=12, orient=None),   # plain rungs plus tilted ones
}
# There is deliberately no "wider plain ladder" variant.  The ladder holds
# only six distinct lattices, and the tilted rungs are those same six over
# again -- so forcing orient off at twelve workers deduplicates straight
# back down to the six of ``base`` and measures nothing.  Widening the
# ladder for its own sake would mean inventing new pitches, which is a
# different experiment from this one.


def do_compare(args):
    arrs = load_arrangements(args)
    names = args.variant
    rows = []
    for blob in arrs:
        start_vol = float(np.prod(blob["extents"]))
        pv = blob["part_volume"]
        row = {"seed": blob["seed"], "start": start_vol, "pv": pv}
        for name in names:
            kw = dict(VARIANTS[name])
            kw.setdefault("workers", args.workers)
            t0 = time.time()
            got = settle_mod.settle(
                blob["meshes"], blob["packer"], blob["packing"],
                resolution=args.settle_resolution,
                budget=args.budget, **kw)
            dt = time.time() - t0
            if got is None:
                vol, ext = start_vol, blob["extents"]
            else:
                _, _, ext = got
                vol = float(np.prod(ext))
            row[name] = (vol, ext, dt)
        rows.append(row)

    w = max(len(n) for n in names)
    print("\n%-5s %10s  %s" % ("seed", "start%", "   ".join(
        "%-*s" % (w + 16, n) for n in names)))
    for row in rows:
        cells = []
        for name in names:
            vol, ext, dt = row[name]
            cells.append("%-*s" % (w + 16, "%6.2f%%  %5.1fs" %
                                   (100 * row["pv"] / vol, dt)))
        print("%-5d %9.2f%%  %s" % (row["seed"], 100 * row["pv"] / row["start"],
                                    "   ".join(cells)))
    print()
    for name in names:
        d = [100 * r["pv"] / r[name][0] for r in rows]
        base = [100 * r["pv"] / r["start"] for r in rows]
        print("%-*s  mean density %6.3f%%   mean gain %+6.3f pts   mean %5.1fs"
              % (w, name, float(np.mean(d)),
                 float(np.mean(np.array(d) - np.array(base))),
                 float(np.mean([r[name][2] for r in rows]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=str(ROOT / "examples" / "sample_parts" / "*.step"))
    ap.add_argument("--pack", action="store_true", help="fill the arrangement cache")
    ap.add_argument("--seeds", default="0,1,2,3",
                    type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--time", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--resolution", type=int, default=28)
    ap.add_argument("--orientations", default="rest")
    ap.add_argument("--settle-resolution", type=int, default=96)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--variant", default="base",
                    type=lambda s: [x for x in s.split(",")])
    args = ap.parse_args()
    if args.pack:
        do_pack(args)
    else:
        do_compare(args)


if __name__ == "__main__":
    main()
