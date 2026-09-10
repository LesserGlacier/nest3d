"""Command line entry point:  python -m nest3d PARTS... [options]"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .pipeline import ORIENTATION_PRESETS, run
from .report import boolean_verify, summary, write_outputs
from .shipping import report as shipping_report


def _expand(patterns):
    out = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            out.extend(hits)
        elif Path(pat).exists():
            out.append(pat)
        else:
            raise SystemExit("no such file: %s" % pat)
    return out


def build_parser():
    p = argparse.ArgumentParser(
        prog="nest3d",
        description="Find the smallest bounding box that holds a set of "
                    "irregular solids, allowing any orientation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python -m nest3d parts/*.step
  python -m nest3d parts/*.step --time 600 --resolution 36 --workers 12
  python -m nest3d parts/*.step --container 300 200 150   # will they fit?
  python -m nest3d parts/*.step --objective height --container 250 250 inf
  python -m nest3d parts/*.step --clearance 2 --orientations fine
""")
    p.add_argument("parts", nargs="+",
                   help="STEP/STL/OBJ/PLY files (globs allowed)")
    p.add_argument("--objective", default="volume",
                   choices=("volume", "compact", "maxdim", "height", "fit"),
                   help="what to minimise (default: volume)")
    p.add_argument("--container", nargs=3, metavar=("W", "D", "H"),
                   help="fixed box to pack into; use 'inf' for a free axis")
    p.add_argument("--time", dest="time_budget", type=float, default=120.0,
                   help="search budget in seconds (default: 120)")
    p.add_argument("--resolution", type=int, default=28,
                   help="voxels across the largest part, search pass "
                        "(default: 28)")
    p.add_argument("--refine-resolution", type=int, default=48,
                   help="voxels across the largest part, refine pass "
                        "(default: 48)")
    p.add_argument("--pitch", type=float,
                   help="explicit voxel pitch in model units, overrides "
                        "--resolution")
    p.add_argument("--clearance", type=float, default=0.0,
                   help="minimum gap to hold between parts, model units")
    p.add_argument("--orientations", default="rest",
                   choices=tuple(ORIENTATION_PRESETS),
                   help="orientation candidate set (default: rest). "
                        "axis = 24 box rotations; rest = those plus stable "
                        "resting poses; fine/full add sampled SO(3)")
    p.add_argument("--n-random", type=int,
                   help="override the number of sampled SO(3) orientations")
    p.add_argument("--no-refine", action="store_true",
                   help="skip the fine-pitch pass")
    p.add_argument("--no-settle", action="store_true",
                   help="skip the final settle, which shakes the finished "
                        "arrangement down onto a much finer lattice")
    p.add_argument("--settle-resolution", type=int, default=96,
                   help="voxels across the largest part for the settle's "
                        "first lattice (default 96). Finer ones are tried "
                        "alongside it, one per worker, and the smallest box "
                        "wins -- so --workers buys settle quality too.")
    p.add_argument("--settle-orient", dest="settle_orient",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="re-choose each part's orientation during the settle, "
                        "on the settle's own lattice rather than the coarse "
                        "one the search picked it on. Left alone, some rungs "
                        "of the ladder do this and some do not; the flag "
                        "forces it on or off everywhere.")
    p.add_argument("--contact", dest="contact_weight", type=float, default=0.0,
                   help="tie-break weight favouring snug placements "
                        "(0 disables, 1 is a good value to try)")
    p.add_argument("--split-solids", action="store_true",
                   help="treat each solid in a STEP file as a separate part")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel processes for the squeeze phase "
                        "(default 1; try %d here)" % max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="directory for packing.json / packed.step / packed.stl")
    p.add_argument("--formats", default="json,step,stl",
                   help="comma-separated subset of json,step,stl")
    p.add_argument("--unit", default="mm", help="label only (default: mm)")
    p.add_argument("--snapshot", action="store_true",
                   help="with --out, write results after each phase instead "
                        "of only at the end, so a long run is inspectable "
                        "while it is still going")
    p.add_argument("--shipping", action="store_true",
                   help="report carton size, girth and dimensional weight")
    p.add_argument("--padding", type=float, default=0.0,
                   help="padding all round inside the carton, model units")
    p.add_argument("--wall", type=float, default=5.0,
                   help="carton wall thickness, model units (default 5)")
    p.add_argument("--weight", type=float,
                   help="actual shipped weight in kg, to see whether "
                        "dimensional weight governs")
    p.add_argument("--verify", action="store_true",
                   help="additionally check the placed solids for "
                        "interference with exact CAD booleans")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--version", action="version", version="nest3d " + __version__)
    return p


def parse_container(values):
    if not values:
        return None
    out = []
    for v in values:
        v = v.strip().lower()
        out.append(float("inf") if v in ("inf", "free", "-", "none")
                   else float(v))
    return np.array(out, dtype=float)


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = _expand(args.parts)
    container = parse_container(args.container)

    if args.objective in ("height", "fit") and container is None:
        raise SystemExit("--objective %s needs --container" % args.objective)

    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())

    def on_phase(name, partial):
        ext = np.asarray(partial.result.extents, dtype=float)
        print("  [%s] %.1f x %.1f x %.1f mm   vol %.4g"
              % (name, ext[0], ext[1], ext[2], partial.result.volume),
              flush=True)
        if args.out and args.snapshot:
            written = write_outputs(partial, args.out, formats, unit=args.unit)
            print("  [%s] snapshot -> %s"
                  % (name, ", ".join(str(w) for w in written)), flush=True)

    t0 = time.time()
    pr = run(paths,
             objective=args.objective,
             resolution=args.resolution,
             refine_resolution=args.refine_resolution,
             pitch=args.pitch,
             clearance=args.clearance,
             orientations=args.orientations,
             n_random=args.n_random,
             container=container,
             time_budget=args.time_budget,
             seed=args.seed,
             split_solids=args.split_solids,
             refine=not args.no_refine,
             settle=not args.no_settle,
             settle_resolution=args.settle_resolution,
             settle_orient=args.settle_orient,
             contact_weight=args.contact_weight,
             workers=args.workers,
             verbose=not args.quiet,
             on_phase=on_phase)

    print(summary(pr, unit=args.unit))
    print("\n  elapsed        %.1fs  (%s)"
          % (time.time() - t0,
             ", ".join("%s %.1fs" % kv for kv in pr.timings.items())))

    if args.shipping:
        print(shipping_report(np.asarray(pr.result.extents, dtype=float),
                              pr.result.lower_bound,
                              padding_mm=args.padding, wall_mm=args.wall,
                              actual_weight_kg=args.weight, unit=args.unit))

    if args.verify:
        check = boolean_verify(pr)
        print()
        if check is None:
            print("  boolean check unavailable (pip install manifold3d)")
        elif check["offenders"]:
            print("  BOOLEAN CHECK FAILED: %d interfering pair(s), largest %.6g"
                  % (len(check["offenders"]), check["worst_volume"]))
            for a, b, v in check["offenders"]:
                print("      %s / %s : %.6g" % (a, b, v))
        else:
            print("  boolean check  %d pairs, no interference "
                  "(largest intersection %.3g)"
                  % (check["pairs"], check["worst_volume"]))

    if args.out:
        written = write_outputs(pr, args.out, formats, unit=args.unit)
        print()
        for w in written:
            print("  wrote %s" % w)

    return 1 if pr.overlaps else 0


if __name__ == "__main__":
    sys.exit(main())
