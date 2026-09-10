"""Time the solver on fixed work, to price a change to the inner loop.

Density at a fixed wall-clock budget is the wrong instrument for this.
A run is budgeted by the clock, so a faster inner loop does not finish
sooner, and whether it lands on a better arrangement is mostly luck: over
six seeds the spread was 48.7% to 55.0%, far wider than anything a change
of this kind moves.  Two configurations can differ by a density point and
mean nothing at all.

So take the deadline out of it.  Give the chains a budget they cannot
reach and let the iteration counts decide the work; the work is then
identical across configurations, and wall clock measures exactly what
changed.  ``distinct_volumes`` is the check -- it must hold a single
value, or the configurations did not do the same work and their times are
not comparable.

Parts come from ``$NEST3D_PARTS``.  Measure on the real parts, not the
samples -- see CLAUDE.md.

    python tools/bench_fixedwork.py <workers> <chains> <reps>
"""
import os, sys, json, glob, time, pathlib, subprocess, statistics

REPO = str(pathlib.Path(__file__).resolve().parents[1])

ONE = r'''
import os, sys, json, glob, time
sys.path.insert(0, r"{repo}")
def main():
    from nest3d.geometry import load_parts
    from nest3d.voxel import build_poses, suggest_pitch
    from nest3d.pipeline import build_orientations
    from nest3d.solve import solve_multistart

    parts = load_parts(sorted(glob.glob(os.environ["NEST3D_PARTS"])))
    meshes = [p.mesh for p in parts]
    pitch = suggest_pitch(meshes, 28)
    rots = build_orientations(parts, "rest", None, 0)
    poses = [build_poses(p.mesh, r, pitch) for p, r in zip(parts, rots)]

    t = time.perf_counter()
    res = solve_multistart(poses, meshes=meshes, seed={seed},
                           free_budget=1e6, budget=1e6,
                           workers={workers}, chains={chains},
                           free_starts=1, free_iterations=30,
                           squeeze_starts=1, squeeze_iterations=20,
                           rounds=2, say=None)
    print("OUT " + json.dumps({{"wall_s": round(time.perf_counter() - t, 2),
                               "volume": float(res.volume)}}))
if __name__ == "__main__":
    main()
'''

# Each column is one size of the mask-transform cache, in MB per process.
# 0 turns it off, which is the baseline to beat.
CONFIGS = {
    "no_cache":  {"NEST3D_FFT_CACHE_MB": "0"},
    "cache_16":  {"NEST3D_FFT_CACHE_MB": "16"},
    "cache_32":  {"NEST3D_FFT_CACHE_MB": "32"},
    "cache_128": {"NEST3D_FFT_CACHE_MB": "128"},
}


def main():
    workers = int(sys.argv[1]); chains = int(sys.argv[2])
    reps = int(sys.argv[3])
    out = {}
    for name, envs in CONFIGS.items():
        walls, vols = [], set()
        for r in range(reps):
            env = dict(os.environ, **envs)
            src = ONE.format(repo=REPO, seed=0, workers=workers, chains=chains)
            p = subprocess.run([sys.executable, "-c", src], capture_output=True,
                               text=True, env=env)
            line = [l for l in p.stdout.splitlines() if l.startswith("OUT ")]
            if not line:
                print("FAILED", name, p.stderr[-900:], flush=True)
                continue
            got = json.loads(line[0][4:])
            walls.append(got["wall_s"]); vols.add(round(got["volume"], 3))
            print("%-6s rep %d  wall %6.2fs  vol %.4g"
                  % (name, r, got["wall_s"], got["volume"]), flush=True)
        if walls:
            out[name] = {"wall_min": min(walls),
                         "wall_median": round(statistics.median(walls), 2),
                         "walls": walls,
                         "distinct_volumes": sorted(vols)}
    if "no_cache" in out:
        for k in out:
            out[k]["speedup_vs_no_cache"] = round(
                out["no_cache"]["wall_median"] / out[k]["wall_median"], 3)
    print("\nRESULT " + json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
