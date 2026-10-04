"""Bake subject-specific knee geometry into its STL files.

The COMAK knee bodies (femur_distal_r, tibia_proximal_r, patella_r, the two
menisci) carry the subject-specific anatomy. Their geometry is baked into the
STL files on disk at the WA scale, and the visual `scale_factors` reset to
[1, 1, 1] — so the STL on disk *is* the subject's true geometry, self-describing
for ParaView, the cartilage-thickness/pressure pipeline, JointMechanics VTP
output, and any other raw-STL consumer.

This is mandatory for the cartilage / meniscus `Smith2018ContactMesh` STLs
anyway — the JAM contact loader reads raw vertices and ignores XML
`scale_factors` (setting them breaks COMAK-IK; see Process_Pipeline/README.md).
Extending the bake to the knee bone + whole-meniscus visual meshes makes the
entire knee region consistent: every knee STL is real subject geometry,
every knee `scale_factors` is [1, 1, 1].

The generic body (torso, pelvis, limbs) is left on ScaleTool's `scale_factors`
— it is not subject-specific and has no raw-STL consumers.
"""

from pathlib import Path
from typing import Dict, Iterable, Set

import numpy as np
import opensim as osim
import pymskt as mskt


def knee_geometry_files(model: "osim.Model", knee_bodies: Iterable[str]) -> Set[str]:
    """The mesh files ``bake_knee_geometry`` rewrites, for a loaded model.

    Every ``attached_geometry`` Mesh on a knee body, plus the ``mesh_file`` of every
    ``Smith2018ContactMesh`` whose frame is a knee body. Body names, frames and file
    names are the same before and after ScaleTool, so this can be read from the
    unscaled model too -- which is how ``scale_comak_model`` knows, before the scaled
    model can load, which files must be real copies rather than links.
    """
    knee_bodies = set(knee_bodies)
    files = set()
    bs = model.getBodySet()
    for bi in range(bs.getSize()):
        body = bs.get(bi)
        if body.getName() not in knee_bodies:
            continue
        for gi in range(body.getPropertyByName("attached_geometry").size()):
            mesh = osim.Mesh.safeDownCast(body.get_attached_geometry(gi))
            if mesh is not None:
                files.add(mesh.get_mesh_file())
    cg = model.getContactGeometrySet()
    for i in range(cg.getSize()):
        g = cg.get(i)
        if g.getConcreteClassName() != "Smith2018ContactMesh":
            continue
        scm = osim.Smith2018ContactMesh.safeDownCast(g)
        if scm.getFrame().getName() in knee_bodies:
            files.add(scm.get_mesh_file())
    return files


def bake_knee_geometry(
    scaled_osim: Path,
    knee_bodies: Iterable[str],
    scale: float,
    geometry_dir: Path,
) -> Dict[str, Path]:
    """Pre-scale every STL attached to a knee body and zero its scale_factors.

    For each body in `knee_bodies`, every `attached_geometry` Mesh STL is
    multiplied by `scale` on disk and its visual `scale_factors` reset to
    [1, 1, 1]. Every `Smith2018ContactMesh` whose parent frame is a knee body
    has its STL pre-scaled too — the cartilage contact STLs are shared with the
    visual meshes; the meniscus superior/inferior surfaces are contact-only.

    `scale` is the isotropic, dimensionless ``s_wa`` ratio from
    ``build_scale_set``. STL vertices are in meters (OSIM space). Each vertex is
    multiplied by `scale` about the body-local origin (0, 0, 0) — i.e.
    ``v_out = v_in * scale``. For the knee subbodies the body-local origin is the
    knee joint center (where knee_r / pf_r sit), NOT the bone centroid: the
    geometry sits offset from the origin (measured centroid-to-origin distances:
    femur 24.8 mm, tibia 54 mm, patella ~1 mm), so the scaled STL bounding box
    shifts slightly toward the origin. The JAM contact loader reads raw vertices
    and ignores XML `scale_factors`, which is why the scale must be baked into
    the STL on disk rather than left on `scale_factors` (the reset to [1, 1, 1]
    avoids double-scaling and makes the on-disk STL self-describing).

    Each unique STL file is scaled exactly once (a file referenced by both a
    contact mesh and a visual mesh is not double-scaled). The model is saved
    back to `scaled_osim`. Returns {stl_filename: stl_path} for files written.

    Every file it rewrites must be a real file in `geometry_dir`: the rewrite is a
    VTK (C++) write in place, which would go straight through a symlink and
    overwrite the file it points at (the unscaled model's mesh, or a shared
    input). A symlink raises before anything is written.
    """
    scaled_osim = Path(scaled_osim)
    geometry_dir = Path(geometry_dir)
    knee_bodies = set(knee_bodies)

    model = osim.Model(str(scaled_osim))
    model.initSystem()

    # Visual attached_geometry on knee bodies plus Smith2018ContactMesh STLs
    # parented to a knee body (the JAM contact loader ignores scale_factors, so
    # these must be pre-scaled regardless).
    stl_files = knee_geometry_files(model, knee_bodies)

    # Zero the visual scale_factors on the knee bodies.
    bs = model.getBodySet()
    for bi in range(bs.getSize()):
        body = bs.get(bi)
        if body.getName() not in knee_bodies:
            continue
        for gi in range(body.getPropertyByName("attached_geometry").size()):
            mesh = osim.Mesh.safeDownCast(body.upd_attached_geometry(gi))
            if mesh is not None:
                mesh.set_scale_factors(osim.Vec3(1.0, 1.0, 1.0))

    linked = sorted(f for f in stl_files if (geometry_dir / f).is_symlink())
    if linked:
        raise RuntimeError(
            f"refusing to bake through symlinks in {geometry_dir}: {linked}. The bake rewrites "
            "each file in place, which would overwrite the file the link points at."
        )

    # NOTE: an STL stores a triangle soup with no vertex indexing, so this
    # read -> scale -> write -> (later) read round-trip can permute the vertex
    # ARRAY ORDER — observed on ~1.5% of the tibia-bone vertices. The geometry
    # is bit-identical (verified: nearest-neighbour distance 0.0 both ways);
    # only the order of the points array may change. This is harmless: OpenSim
    # re-reads each STL as triangle soup and does not depend on array order.
    # Any "did the mesh change" check must therefore compare geometry as a
    # point SET (e.g. symmetric nearest-neighbour), not element-wise.
    written: Dict[str, Path] = {}
    for fname in sorted(stl_files):
        stl_path = geometry_dir / fname
        if not stl_path.is_file():
            raise FileNotFoundError(f"Knee geometry STL {fname!r} not found in {geometry_dir}")
        m = mskt.mesh.Mesh(str(stl_path))
        m.point_coords = np.asarray(m.point_coords) * float(scale)
        m.save_mesh(str(stl_path))
        written[fname] = stl_path

    model.finalizeConnections()
    model.printToXML(str(scaled_osim))
    return written
