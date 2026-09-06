"""Generate nine irregular sample parts as STEP files.

These exist so the packer can be exercised end to end -- through the same
STEP reader, tessellator and voxeliser that real parts go through -- without
needing the real files.  Dimensions are in millimetres.
"""
from __future__ import annotations

import sys
from pathlib import Path

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepPrimAPI import (BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder,
                             BRepPrimAPI_MakeWedge)
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.STEPControl import STEPControl_StepModelType, STEPControl_Writer


def box(dx, dy, dz, at=(0, 0, 0)):
    return BRepPrimAPI_MakeBox(gp_Pnt(*at), dx, dy, dz).Shape()


def cyl(r, h, at=(0, 0, 0), axis=(0, 0, 1)):
    ax = gp_Ax2(gp_Pnt(*at), gp_Dir(*axis))
    return BRepPrimAPI_MakeCylinder(ax, r, h).Shape()


def cut(a, b):
    return BRepAlgoAPI_Cut(a, b).Shape()


def fuse(a, b):
    return BRepAlgoAPI_Fuse(a, b).Shape()


def moved(shape, dx, dy, dz):
    t = gp_Trsf()
    t.SetTranslation(gp_Vec(dx, dy, dz))
    return BRepBuilderAPI_Transform(shape, t, True).Shape()


def l_bracket():
    s = box(80, 60, 20)
    s = fuse(s, box(20, 60, 70))
    return cut(s, cyl(6, 30, at=(55, 30, -5)))


def u_channel():
    s = box(90, 50, 40)
    return cut(s, box(70, 60, 26, at=(10, -5, 14)))


def t_block():
    s = box(70, 24, 24)
    s = fuse(s, box(24, 24, 60, at=(23, 0, 0)))
    return cut(s, cyl(7, 40, at=(35, 12, 20), axis=(0, 1, 0)))


def flanged_hub():
    s = cyl(34, 12)
    s = fuse(s, cyl(18, 55))
    s = cut(s, cyl(9, 80, at=(0, 0, -5)))
    for x, y in ((24, 0), (-24, 0), (0, 24), (0, -24)):
        s = cut(s, cyl(4.5, 30, at=(x, y, -5)))
    return s


def stepped_shaft():
    s = cyl(14, 70)
    s = fuse(s, cyl(24, 18))
    s = fuse(s, cyl(20, 14, at=(0, 0, 56)))
    return cut(s, box(10, 40, 20, at=(-5, -20, 60)))


def wedge_rib():
    s = BRepPrimAPI_MakeWedge(70.0, 40.0, 50.0, 22.0).Shape()
    s = fuse(s, box(70, 10, 12))
    return cut(s, cyl(6, 30, at=(35, 5, -5)))


def notched_plate():
    s = box(110, 70, 12)
    s = cut(s, box(30, 30, 20, at=(-2, 40, -4)))
    s = cut(s, cyl(8, 30, at=(25, 25, -5)))
    s = cut(s, cyl(8, 30, at=(85, 45, -5)))
    return fuse(s, box(12, 70, 34, at=(98, 0, 12)))


def cross_prism():
    s = box(84, 22, 26)
    s = fuse(s, box(22, 84, 26, at=(31, -31, 0)))
    return cut(s, cyl(7, 40, at=(42, 11, -5)))


def half_shell():
    s = cyl(32, 60)
    s = cut(s, cyl(24, 60, at=(0, 0, 8)))
    s = cut(s, box(70, 40, 70, at=(-35, 0, -5)))
    return fuse(s, box(64, 12, 10, at=(-32, -12, 0)))


PARTS = [
    ("01_l_bracket", l_bracket),
    ("02_u_channel", u_channel),
    ("03_t_block", t_block),
    ("04_flanged_hub", flanged_hub),
    ("05_stepped_shaft", stepped_shaft),
    ("06_wedge_rib", wedge_rib),
    ("07_notched_plate", notched_plate),
    ("08_cross_prism", cross_prism),
    ("09_half_shell", half_shell),
]


def write_step(shape, path):
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_StepModelType.STEPControl_AsIs)
    if writer.Write(str(path)) != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise IOError("failed to write %s" % path)


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in PARTS:
        path = out_dir / (name + ".step")
        write_step(fn(), path)
        print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         Path(__file__).resolve().parents[1] / "examples" / "sample_parts")
