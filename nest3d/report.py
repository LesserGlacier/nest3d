"""Reporting and export of a finished packing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import export_mesh, export_step


def summary(pr, unit="mm") -> str:
    """Human-readable result block."""
    res = pr.result
    ext = np.asarray(res.extents, dtype=float)
    lines = []
    lines.append("")
    lines.append("  BOUNDING BOX   %.2f x %.2f x %.2f %s"
                 % (ext[0], ext[1], ext[2], unit))
    lines.append("  volume         %.4g %s^3" % (res.volume, unit))
    lines.append("  part volume    %.4g %s^3  (%d parts)"
                 % (res.lower_bound, unit, len(pr.parts)))
    lines.append("  fill density   %.1f%%   (of the box that is solid part)"
                 % (100 * res.density))
    lines.append("  diagonal       %.2f %s" % (float(np.linalg.norm(ext)), unit))
    lines.append("  voxel pitch    %.3f %s" % (pr.pitch, unit))

    if pr.overlaps:
        lines.append("  INTERFERENCE   %d overlapping pair(s) -- see below"
                     % len(pr.overlaps))
        for a, b, n in pr.overlaps:
            lines.append("      %s / %s : %d voxels"
                         % (pr.parts[a].name, pr.parts[b].name, n))
    else:
        lines.append("  interference   none (verified pairwise on the voxel masks)")

    lines.append("")
    lines.append("  positions are of the part origin, measured from the "
                 "box's minimum corner")
    lines.append("  %-22s %-34s %s" % ("part", "position (x y z)", "rotation"))
    origin = np.asarray(res.packing.bbox_lo, dtype=float)
    for i, part in enumerate(pr.parts):
        m = pr.transforms[i]
        if m is None:
            lines.append("  %-22s NOT PLACED" % part.name)
            continue
        t = m[:3, 3] - origin
        rpy = np.degrees(_rpy(m[:3, :3]))
        lines.append("  %-22s %9.2f %9.2f %9.2f     %7.1f %7.1f %7.1f deg"
                     % (part.name, t[0], t[1], t[2], rpy[0], rpy[1], rpy[2]))
    return "\n".join(lines)


def _rpy(r):
    """Intrinsic Z-Y-X Euler angles, purely for a readable listing."""
    sy = float(np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
    if sy > 1e-9:
        return np.array([np.arctan2(r[2, 1], r[2, 2]),
                         np.arctan2(-r[2, 0], sy),
                         np.arctan2(r[1, 0], r[0, 0])])
    return np.array([np.arctan2(-r[1, 2], r[1, 1]),
                     np.arctan2(-r[2, 0], sy), 0.0])


def to_json(pr, unit="mm") -> dict:
    res = pr.result
    ext = np.asarray(res.extents, dtype=float)
    out = {
        "unit": unit,
        "bounding_box": {
            "extents": [float(v) for v in ext],
            "volume": float(res.volume),
            "origin": [float(v) for v in np.asarray(res.packing.bbox_lo)],
        },
        "part_volume": float(res.lower_bound),
        "fill_density": float(res.density),
        "voxel_pitch": float(pr.pitch),
        "interference": [
            {"a": pr.parts[a].name, "b": pr.parts[b].name, "voxels": int(n)}
            for a, b, n in pr.overlaps
        ],
        "parts": [],
    }
    origin = np.asarray(res.packing.bbox_lo, dtype=float)
    for i, part in enumerate(pr.parts):
        m = pr.transforms[i]
        entry = {
            "name": part.name,
            "source": str(part.source) if part.source else None,
            "placed": m is not None,
        }
        if m is not None:
            # Re-reference to the box corner so the numbers are usable
            # straight away as "put the part here inside the crate".
            local = m.copy()
            local[:3, 3] -= origin
            entry["transform"] = [[float(v) for v in row] for row in local]
            entry["rotation"] = [[float(v) for v in row] for row in m[:3, :3]]
            entry["translation"] = [float(v) for v in local[:3, 3]]
        out["parts"].append(entry)
    return out


def write_outputs(pr, out_dir, formats=("json", "step", "stl"), unit="mm"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    placed = [(p, m) for p, m in zip(pr.parts, pr.transforms) if m is not None]

    if "json" in formats:
        path = out_dir / "packing.json"
        path.write_text(json.dumps(to_json(pr, unit), indent=2))
        written.append(path)

    if "step" in formats and placed:
        try:
            path = export_step([p for p, _ in placed], [m for _, m in placed],
                               out_dir / "packed.step")
            written.append(Path(path))
        except Exception as exc:
            written.append("step export skipped: %s" % exc)

    if "stl" in formats and placed:
        path = export_mesh([p for p, _ in placed], [m for _, m in placed],
                           out_dir / "packed.stl")
        written.append(Path(path))

    return written
