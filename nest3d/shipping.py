"""Turn a packed bounding box into carton numbers.

A shipping carton is not the same object as the minimal bounding box: it
has walls, it usually wants padding, carriers bill on a volumetric weight
rather than the real one, and past a certain size they stop quoting normal
rates at all.  This module does that arithmetic, so the packing result can
be read as "order this box" rather than "here are three numbers".

Carrier rules change and vary by contract.  The thresholds below are the
commonly published ones and are exposed as parameters, not baked in --
treat them as a prompt to check your own rate card, not as authority.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MM_PER_IN = 25.4

# Commonly published US domestic values.  Verify against your own rate card.
DIM_DIVISOR_IN3_PER_LB = 139.0     # UPS/FedEx US domestic retail
DIM_DIVISOR_CM3_PER_KG = 5000.0    # common international metric divisor

MAX_LENGTH_PLUS_GIRTH_IN = 165.0   # over this, most carriers refuse
LARGE_PACKAGE_GIRTH_IN = 130.0     # over this, "large package" surcharges
MAX_LENGTH_IN = 108.0


@dataclass
class Carton:
    inner_mm: np.ndarray
    outer_mm: np.ndarray
    padding_mm: float
    wall_mm: float

    @property
    def inner_in(self):
        return self.inner_mm / MM_PER_IN

    @property
    def outer_in(self):
        return self.outer_mm / MM_PER_IN

    @property
    def length_girth_in(self) -> float:
        d = np.sort(self.outer_in)[::-1]
        return float(d[0] + 2 * (d[1] + d[2]))

    def dim_weight_lb(self, divisor=DIM_DIVISOR_IN3_PER_LB) -> float:
        return float(np.prod(self.outer_in) / divisor)

    def dim_weight_kg(self, divisor=DIM_DIVISOR_CM3_PER_KG) -> float:
        return float(np.prod(self.outer_mm / 10.0) / divisor)


def build_carton(extents_mm, padding_mm=0.0, wall_mm=0.0, round_to_mm=0.0):
    """Grow the packed box into a carton: padding all round, then walls."""
    inner = np.asarray(extents_mm, dtype=float) + 2.0 * float(padding_mm)
    if round_to_mm > 0:
        inner = np.ceil(inner / round_to_mm) * round_to_mm
    outer = inner + 2.0 * float(wall_mm)
    return Carton(inner_mm=inner, outer_mm=outer,
                  padding_mm=float(padding_mm), wall_mm=float(wall_mm))


def report(extents_mm, part_volume_mm3, padding_mm=0.0, wall_mm=5.0,
           actual_weight_kg=None, unit="mm"):
    """Human-readable carton block for a packed result."""
    lines = []
    carton = build_carton(extents_mm, padding_mm, wall_mm)

    inner_in = carton.inner_in
    outer_in = carton.outer_in
    lines.append("")
    lines.append("  SHIPPING")
    lines.append("    packed contents   %.0f x %.0f x %.0f mm   (%.1f x %.1f x %.1f in)"
                 % (extents_mm[0], extents_mm[1], extents_mm[2],
                    extents_mm[0] / MM_PER_IN, extents_mm[1] / MM_PER_IN,
                    extents_mm[2] / MM_PER_IN))
    if padding_mm:
        lines.append("    + %.0f mm padding   %.0f x %.0f x %.0f mm inner"
                     % (padding_mm, *carton.inner_mm))
    lines.append("    carton inner      %.0f x %.0f x %.0f mm   (%.1f x %.1f x %.1f in)"
                 % (*carton.inner_mm, *inner_in))
    lines.append("    carton outer      %.0f x %.0f x %.0f mm   (%.1f x %.1f x %.1f in)"
                 % (*carton.outer_mm, *outer_in))

    lg = carton.length_girth_in
    lines.append("")
    lines.append("    length + girth    %.1f in" % lg)
    if lg > MAX_LENGTH_PLUS_GIRTH_IN:
        lines.append("      OVER the %.0f in limit most carriers will accept at all"
                     % MAX_LENGTH_PLUS_GIRTH_IN)
    elif lg > LARGE_PACKAGE_GIRTH_IN:
        lines.append("      over %.0f in: expect large-package / oversize surcharges"
                     % LARGE_PACKAGE_GIRTH_IN)
    else:
        lines.append("      under the %.0f in large-package threshold"
                     % LARGE_PACKAGE_GIRTH_IN)

    longest = float(np.max(outer_in))
    if longest > MAX_LENGTH_IN:
        lines.append("      longest side %.1f in exceeds the %.0f in maximum"
                     % (longest, MAX_LENGTH_IN))

    dw_lb = carton.dim_weight_lb()
    dw_kg = carton.dim_weight_kg()
    lines.append("")
    lines.append("    dimensional wt    %.1f lb  (US domestic, /%.0f)"
                 % (dw_lb, DIM_DIVISOR_IN3_PER_LB))
    lines.append("                      %.1f kg  (metric, /%.0f)"
                 % (dw_kg, DIM_DIVISOR_CM3_PER_KG))
    if actual_weight_kg is not None:
        billed = max(actual_weight_kg, dw_kg)
        driver = "dimensional" if dw_kg > actual_weight_kg else "actual"
        lines.append("    actual weight     %.1f kg  ->  billed on the %s weight, %.1f kg"
                     % (actual_weight_kg, driver, billed))
    else:
        lines.append("    (billed weight is the greater of actual and dimensional;")
        lines.append("     pass the real weight to see which one governs)")

    fill = part_volume_mm3 / float(np.prod(carton.inner_mm))
    lines.append("")
    lines.append("    carton is %.0f%% glass by volume; the other %.0f%% is air "
                 "and packing" % (100 * fill, 100 * (1 - fill)))
    lines.append("    NOTE carrier limits and divisors above are the commonly")
    lines.append("         published ones - check them against your own rate card.")
    return "\n".join(lines)
