"""Is a GPU worth it for the placement correlation?

The packer spends about four fifths of its wall clock inside
``scipy.signal.fftconvolve`` (nest3d/pack.py), which reads like an
open-and-shut case for a GPU.  It is not, and the reason is the baseline:
the search already fans out across every core, so the comparison is not
one CPU core against a GPU but *all* of them against it.  A transform that
is forty times faster on the device is worth well under two if the host
was doing twenty-four of them at once.

That number decides whether porting the placer is worth days of work, and
it is cheap to measure, so measure it before writing any of it.  Sizes
default to a ladder spanning what a real run actually asks for; pass
--pile and --mask to pin an exact pair.

    python tools/fft_bench.py --cores 24
"""
from __future__ import annotations

import argparse
import time

import os

import numpy as np
from scipy import signal

def _add_cuda_dll_dirs():
    """Make the pip-installed CUDA libraries findable on Windows.

    CuPy's wheels do not carry cuFFT and friends; the nvidia-*-cu12 wheels
    do, but nothing puts their bin directories on the DLL search path, so
    importing cupy.fft fails with a bare "known cufft DLL" error even
    though the file is sitting in site-packages.  Harmless elsewhere.
    """
    if not hasattr(os, "add_dll_directory"):
        return
    import glob
    import site
    roots = list(site.getsitepackages())
    roots.append(site.getusersitepackages())
    for root in roots:
        for d in glob.glob(os.path.join(root, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass


_add_cuda_dll_dirs()

try:
    import cupy as cp
except ImportError:  # measured on the CPU alone, which is still informative
    cp = None


def _gpu_fftconvolve_valid(a, b, out_shape, fshape):
    """``fftconvolve(a, b, mode='valid')`` on the device, via cuFFT alone.

    Written out rather than taken from cupyx.scipy.signal because that
    module pulls in cuBLAS through an unrelated import chain, and because
    a port would want this level anyway -- the plan and the transforms of
    the part mask are what a real implementation caches.
    """
    fa = cp.fft.rfftn(a, fshape)
    fb = cp.fft.rfftn(b, fshape)
    full = cp.fft.irfftn(fa * fb, fshape)
    sl = tuple(slice(bs - 1, bs - 1 + o)
               for bs, o in zip(b.shape, out_shape))
    return full[sl]


# Cube sides spanning a real run: the coarse search sits at the low end,
# the refine pass in the middle, the settle's finer rungs at the top.
DEFAULT_SIDES = (48, 64, 96, 128, 160)

# Captured by instrumenting _correlate over a real nine-part run: the ten
# commonest (pile, mask) pairs, which between them are most of the calls.
# The mean pile is 1.59e5 voxels and the largest seen is 9.8e5.
REAL_SHAPES = (
    ((46, 55, 85), (16, 17, 29)),
    ((53, 63, 89), (13, 18, 31)),
    ((50, 59, 83), (16, 16, 28)),
    ((46, 54, 74), (9, 13, 23)),
    ((49, 59, 83), (16, 16, 28)),
    ((48, 46, 58), (9, 9, 15)),
    ((28, 43, 49), (7, 11, 11)),
    ((40, 40, 29), (13, 7, 8)),
)


def _time(fn, repeat, warmup=2):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) / repeat


def _time_gpu(fn, repeat, warmup=3):
    for _ in range(warmup):
        fn()
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t0) / repeat


def bench(pile_shape, mask_shape, cores, repeat):
    rng = np.random.default_rng(0)
    # Occupancy is sparse and boolean in the real thing, but fftconvolve
    # casts to float32 either way, so density does not change the cost.
    pile = (rng.random(pile_shape) < 0.3).astype(np.float32)
    mask = (rng.random(mask_shape) < 0.3).astype(np.float32)

    row = {"pile": pile_shape, "mask": mask_shape}
    row["cpu1"] = _time(lambda: signal.fftconvolve(pile, mask, mode="valid"),
                        repeat)
    # What the whole host manages when every core is busy on its own chain:
    # the chains are independent, so aggregate throughput is what a GPU has
    # to beat, not the latency of one transform.
    row["cpu_all"] = row["cpu1"] / cores

    if cp is None:
        return row

    d_pile = cp.asarray(pile)
    d_mask = cp.asarray(mask)
    out_shape = tuple(p - m + 1 for p, m in zip(pile_shape, mask_shape))
    fshape = tuple(p + m - 1 for p, m in zip(pile_shape, mask_shape))

    # Check the device agrees with the host before timing it: a fast wrong
    # answer is not a result.
    ref = signal.fftconvolve(pile, mask, mode="valid")
    got = cp.asnumpy(_gpu_fftconvolve_valid(d_pile, d_mask, out_shape, fshape))
    row["err"] = float(np.abs(ref - got).max() / max(1.0, np.abs(ref).max()))

    row["gpu"] = _time_gpu(
        lambda: _gpu_fftconvolve_valid(d_pile, d_mask, out_shape, fshape),
        repeat)

    # The same call with the host round trip left in, which is what a naive
    # drop-in port would actually pay on every placement.
    def with_transfer():
        a = cp.asarray(pile)
        b = cp.asarray(mask)
        return cp.asnumpy(_gpu_fftconvolve_valid(a, b, out_shape, fshape))

    row["gpu_xfer"] = _time_gpu(with_transfer, repeat)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, default=0,
                    help="cores the search fans out over (default: detected)")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--pile", type=int, nargs=3, default=None)
    ap.add_argument("--mask", type=int, nargs=3, default=None)
    ap.add_argument("--sides", type=int, nargs="*", default=DEFAULT_SIDES)
    ap.add_argument("--real", action="store_true",
                    help="use shapes captured from an actual run")
    args = ap.parse_args()

    cores = args.cores or (__import__("os").cpu_count() or 1)
    if cp is None:
        print("cupy not importable -- CPU columns only\n")
    else:
        name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        print("device: %s\n" % (name.decode() if isinstance(name, bytes) else name))

    if args.pile and args.mask:
        pairs = [(tuple(args.pile), tuple(args.mask))]
    elif args.real:
        # Shapes actually seen in a nine-part run, in order of how much of
        # the run's time they account for.  Round cubes badly overstate the
        # sizes involved: the mean pile is about 1.6e5 voxels, not 1e6.
        pairs = REAL_SHAPES
    else:
        # A mask about a third of the pile per axis is what a nine-part
        # arrangement looks like once a few parts are down.
        pairs = [((s, s, s), (s // 3, s // 3, s // 3)) for s in args.sides]

    print("%-16s %-14s %9s %9s %9s %9s %8s %8s"
          % ("pile", "mask", "1 core", "%d cores" % cores, "gpu", "gpu+xfer",
             "vs 1", "vs all"))
    for pile_shape, mask_shape in pairs:
        r = bench(pile_shape, mask_shape, cores, args.repeat)
        line = "%-16s %-14s %8.2fms %8.2fms" % (
            "x".join(map(str, pile_shape)), "x".join(map(str, mask_shape)),
            r["cpu1"] * 1e3, r["cpu_all"] * 1e3)
        if "gpu" in r:
            line += " %8.2fms %8.2fms %7.1fx %7.1fx" % (
                r["gpu"] * 1e3, r["gpu_xfer"] * 1e3,
                r["cpu1"] / r["gpu"], r["cpu_all"] / r["gpu"])
            if r["err"] > 1e-4:
                line += "  MISMATCH %.2g" % r["err"]
        else:
            line += " %8s %8s %7s %7s" % ("-", "-", "-", "-")
        print(line, flush=True)

    print("\n'vs all' is the number that matters: a GPU replaces the whole\n"
          "host, not one core.  Under about 2x, a port is not worth writing.")


if __name__ == "__main__":
    main()
