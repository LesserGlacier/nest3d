"""Loading solids (STEP / mesh formats) and exporting packed arrangements.

STEP is read through OpenCascade (OCP) and tessellated to a triangle mesh.
Everything downstream of this module works on meshes only, so any format
trimesh can read is equally acceptable.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

MESH_SUFFIXES = {".stl", ".obj", ".ply", ".off", ".3mf", ".glb", ".gltf", ".dae"}
STEP_SUFFIXES = {".step", ".stp"}


@dataclass
class Part:
    """One rigid body to be packed."""

    name: str
    mesh: trimesh.Trimesh
    source: Path | None = None
    shape_key: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def volume(self) -> float:
        return float(abs(self.mesh.volume))

    @property
    def extents(self) -> np.ndarray:
        return np.asarray(self.mesh.extents, dtype=float)

    def transformed(self, matrix: np.ndarray) -> trimesh.Trimesh:
        m = self.mesh.copy()
        m.apply_transform(matrix)
        return m


# ---------------------------------------------------------------------------
# STEP
# ---------------------------------------------------------------------------

def _caster(cls, name):
    """OCP exposes static casters as ``Face`` or ``Face_s`` depending on build."""
    return getattr(cls, name + "_s", None) or getattr(cls, name)


def _step_shapes(path: Path):
    """Return the top level TopoDS_Shapes read from a STEP file."""
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise IOError("OpenCascade could not read STEP file: %s" % path)
    reader.TransferRoots()
    n = reader.NbShapes()
    if n == 0:
        raise IOError("STEP file contains no transferable shapes: %s" % path)
    return [reader.Shape(i) for i in range(1, n + 1)]


def _tessellate(shape, linear_deflection, angular_deflection):
    """Tessellate a TopoDS_Shape into (vertices, faces) arrays."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_Orientation
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)

    verts = []
    faces = []
    offset = 0

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = _caster(TopoDS, "Face")(exp.Current())
        loc = TopLoc_Location()
        tri = _caster(BRep_Tool, "Triangulation")(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            nb_nodes = tri.NbNodes()
            pts = np.empty((nb_nodes, 3), dtype=np.float64)
            for i in range(1, nb_nodes + 1):
                p = tri.Node(i).Transformed(trsf)
                pts[i - 1] = (p.X(), p.Y(), p.Z())

            nb_tris = tri.NbTriangles()
            idx = np.empty((nb_tris, 3), dtype=np.int64)
            for i in range(1, nb_tris + 1):
                a, b, c = tri.Triangle(i).Get()
                idx[i - 1] = (a - 1, b - 1, c - 1)

            # A STEP face carries an orientation flag; a reversed face needs
            # its winding flipped so the assembled mesh has outward normals.
            if face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED:
                idx = idx[:, ::-1]

            verts.append(pts)
            faces.append(idx + offset)
            offset += nb_nodes
        exp.Next()

    if not verts:
        raise IOError("Tessellation produced no triangles")
    return np.vstack(verts), np.vstack(faces)


def _drop_degenerate(mesh):
    """Remove zero-area faces left behind by tessellation.

    A single sliver is enough to make trimesh call an otherwise perfectly
    closed solid non-watertight, and to split it into a phantom second
    component.  That in turn would stop the interior fill, so the part
    would voxelise as a hollow shell and other parts could be packed
    inside it.  Cheap to strip, expensive to miss.
    """
    try:
        keep = mesh.nondegenerate_faces()
    except Exception:
        try:
            areas = np.asarray(mesh.area_faces, dtype=float)
            keep = areas > max(float(areas.max()) * 1e-12, 1e-12)
        except Exception:
            return mesh
    keep = np.asarray(keep)
    if keep.dtype == bool:
        dropped = int((~keep).sum())
    else:
        dropped = len(mesh.faces) - len(keep)
    if dropped:
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
    return mesh


def _auto_deflection(shape) -> float:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    _caster(BRepBndLib, "Add")(shape, box)
    lo, hi = box.CornerMin(), box.CornerMax()
    diag = float(np.linalg.norm([hi.X() - lo.X(), hi.Y() - lo.Y(), hi.Z() - lo.Z()]))
    return max(diag / 800.0, 1e-6)


def _explode_solids(shapes):
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    out = []
    for shape in shapes:
        exp = TopExp_Explorer(shape, TopAbs_SOLID)
        found = False
        while exp.More():
            out.append(_caster(TopoDS, "Solid")(exp.Current()))
            found = True
            exp.Next()
        if not found:
            out.append(shape)
    return out


def load_step(path, deflection=None, angular_deflection=0.35,
              split_solids=False):
    """Read a STEP file into one or more Parts.

    ``deflection`` is the chord tolerance for tessellation in model units;
    when None it is derived from the shape's diagonal.
    """
    path = Path(path)
    shapes = _step_shapes(path)
    if split_solids:
        shapes = _explode_solids(shapes)

    parts = []
    for i, shape in enumerate(shapes):
        defl = deflection if deflection is not None else _auto_deflection(shape)
        v, f = _tessellate(shape, defl, angular_deflection)
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)
        mesh.merge_vertices()
        _drop_degenerate(mesh)
        try:
            mesh.fix_normals()
        except Exception:
            pass
        name = path.stem if len(shapes) == 1 else "%s#%d" % (path.stem, i + 1)
        parts.append(Part(name=name, mesh=mesh, source=path))
    return parts


# ---------------------------------------------------------------------------
# Generic loading
# ---------------------------------------------------------------------------

def load_part(path, split_solids=False, **step_kwargs):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in STEP_SUFFIXES:
        return load_step(path, split_solids=split_solids, **step_kwargs)
    if suffix in MESH_SUFFIXES:
        loaded = trimesh.load(path, force="mesh")
        if isinstance(loaded, trimesh.Scene):
            loaded = loaded.to_mesh()
        return [Part(name=path.stem, mesh=loaded, source=path)]
    raise ValueError("Unsupported file type %r for %s" % (suffix, path))


def load_parts(paths, split_solids=False, **step_kwargs):
    parts = []
    for p in paths:
        parts.extend(load_part(p, split_solids=split_solids, **step_kwargs))
    for part in parts:
        part.shape_key = shape_signature(part.mesh)
    return parts


def shape_signature(mesh, digits=5) -> str:
    """A pose-invariant fingerprint, used to spot duplicate parts.

    Volume, area and the principal moments of inertia are all invariant
    under rigid motion, so two copies of the same solid hash identically
    however they happen to be positioned in their own files.
    """
    try:
        inertia = np.linalg.eigvalsh(mesh.moment_inertia)
    except Exception:
        inertia = np.zeros(3)
    scale = max(abs(float(mesh.volume)), 1e-12)
    feats = np.concatenate([
        [abs(float(mesh.volume)), float(mesh.area)],
        np.sort(inertia) / (scale ** (5.0 / 3.0)),
        np.sort(np.asarray(mesh.extents, dtype=float)),
    ])
    text = ",".join(("%." + str(digits) + "g") % v for v in feats)
    return hashlib.sha1(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_step(parts, transforms, out_path):
    """Write the packed arrangement back out as a single STEP assembly.

    Each part is re-read from its source file so the export carries exact
    B-rep geometry, not the tessellation used for packing.
    """
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_StepModelType, STEPControl_Writer
    from OCP.TopoDS import TopoDS_Compound

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    cache = {}
    for part, mat in zip(parts, transforms):
        if part.source is None or part.source.suffix.lower() not in STEP_SUFFIXES:
            raise ValueError(
                "STEP export needs STEP sources; %s came from %s" % (part.name, part.source)
            )
        key = str(part.source)
        if key not in cache:
            cache[key] = _step_shapes(part.source)
        shapes = cache[key]
        shape = shapes[0] if len(shapes) == 1 else shapes[int(part.meta.get("shape_index", 0))]

        trsf = gp_Trsf()
        r = mat[:3, :3]
        t = mat[:3, 3]
        trsf.SetValues(
            float(r[0, 0]), float(r[0, 1]), float(r[0, 2]), float(t[0]),
            float(r[1, 0]), float(r[1, 1]), float(r[1, 2]), float(t[1]),
            float(r[2, 0]), float(r[2, 1]), float(r[2, 2]), float(t[2]),
        )
        moved = BRepBuilderAPI_Transform(shape, trsf, True).Shape()
        builder.Add(compound, moved)

    writer = STEPControl_Writer()
    writer.Transfer(compound, STEPControl_StepModelType.STEPControl_AsIs)
    status = writer.Write(str(out_path))
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise IOError("Failed to write STEP file: %s" % out_path)
    return out_path


def export_mesh(parts, transforms, out_path):
    """Write the packed arrangement as a single mesh (STL/OBJ/PLY/3MF)."""
    scene = trimesh.Scene()
    for i, (part, mat) in enumerate(zip(parts, transforms)):
        scene.add_geometry(part.transformed(mat), node_name="%s_%d" % (part.name, i))
    combined = scene.to_mesh()
    combined.export(out_path)
    return out_path
