"""Identity-scaling canary: s=1 in every body → spatial state matches base.

If this fails, something in the COMAK body-scaling wrapper is non-identity at identity.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import opensim as osim
import pytest


def _gather_blankevoort_slack(model: osim.Model) -> Dict[str, float]:
    out = {}
    fs = model.upd_ForceSet()
    for i in range(fs.getSize()):
        force = fs.get(i)
        lig = osim.Blankevoort1991Ligament.safeDownCast(force)
        if lig is None:
            continue
        out[force.getName()] = float(lig.get_slack_length())
    return out


def _gather_wrap_dims(model: osim.Model) -> Dict[str, Tuple]:
    """Each wrap → ('cyl'|'ell', primary_scalar_or_dims tuple, translation tuple)."""
    out = {}
    bs = model.getBodySet()
    for bi in range(bs.getSize()):
        body = bs.get(bi)
        wos = body.getWrapObjectSet()
        for wi in range(wos.getSize()):
            w = wos.get(wi)
            tag = f"{body.getName()}/{w.getName()}"
            cls = w.getConcreteClassName()
            t = w.get_translation()
            trans = (float(t[0]), float(t[1]), float(t[2]))
            if cls == "WrapCylinder":
                cyl = osim.WrapCylinder.safeDownCast(w)
                out[tag] = ("cyl", (float(cyl.get_radius()), float(cyl.get_length())), trans)
            elif cls == "WrapEllipsoid":
                ell = osim.WrapEllipsoid.safeDownCast(w)
                d = ell.get_dimensions()
                out[tag] = (
                    "ell",
                    (float(d[0]), float(d[1]), float(d[2])),
                    trans,
                )
    return out


def _gather_joint_translations(model: osim.Model) -> Dict[str, Tuple]:
    """For each Joint, frame[0] translation (the offset on parent side)."""
    out = {}
    js = model.getJointSet()
    for i in range(js.getSize()):
        j = js.get(i)
        try:
            fr = j.get_frames(0)
        except Exception:
            continue
        t = fr.get_translation()
        out[j.getName()] = (float(t[0]), float(t[1]), float(t[2]))
    return out


@pytest.mark.slow
class TestIdentityRoundtrip:
    @pytest.fixture(scope="class")
    def base_model(self, base_comak_path):
        m = osim.Model(str(base_comak_path))
        m.initSystem()
        return m

    @pytest.fixture(scope="class")
    def scaled_model(self, identity_scaled_model):
        m = osim.Model(str(identity_scaled_model))
        m.initSystem()
        return m

    def test_blankevoort_slack_lengths_match(self, base_model, scaled_model):
        base = _gather_blankevoort_slack(base_model)
        scaled = _gather_blankevoort_slack(scaled_model)
        assert set(base) == set(scaled)
        for name in base:
            assert scaled[name] == pytest.approx(base[name], rel=1e-9, abs=1e-9), (
                f"slack_length changed at identity for {name}: " f"{base[name]} → {scaled[name]}"
            )

    def test_wrap_surface_dims_match(self, base_model, scaled_model):
        base = _gather_wrap_dims(base_model)
        scaled = _gather_wrap_dims(scaled_model)
        assert set(base) == set(scaled)
        for tag in base:
            kind_b, dims_b, trans_b = base[tag]
            kind_s, dims_s, trans_s = scaled[tag]
            assert kind_b == kind_s, f"{tag} class changed"
            assert dims_s == pytest.approx(dims_b, abs=1e-9), f"{tag} dims changed"
            assert trans_s == pytest.approx(trans_b, abs=1e-9), f"{tag} translation changed"

    def test_joint_offset_translations_match(self, base_model, scaled_model):
        base = _gather_joint_translations(base_model)
        scaled = _gather_joint_translations(scaled_model)
        for name in base:
            assert name in scaled
            assert scaled[name] == pytest.approx(
                base[name], abs=1e-9
            ), f"joint {name} parent-offset translation changed at identity"

    def test_smith2018_mesh_file_names_preserved(self, base_model, scaled_model):
        cg_b = base_model.getContactGeometrySet()
        cg_s = scaled_model.getContactGeometrySet()
        files_b, files_s = {}, {}
        for i in range(cg_b.getSize()):
            g = cg_b.get(i)
            if g.getConcreteClassName() != "Smith2018ContactMesh":
                continue
            scm = osim.Smith2018ContactMesh.safeDownCast(g)
            files_b[scm.getName()] = scm.get_mesh_file()
        for i in range(cg_s.getSize()):
            g = cg_s.get(i)
            if g.getConcreteClassName() != "Smith2018ContactMesh":
                continue
            scm = osim.Smith2018ContactMesh.safeDownCast(g)
            files_s[scm.getName()] = scm.get_mesh_file()
        assert files_b == files_s

    def test_knee_stl_vertices_match(
        self,
        identity_scaled_model: Path,
        base_comak_path: Path,
    ):
        """At s=1, the knee-geometry bake must preserve geometry exactly —
        bones and meniscus visuals as well as the cartilage/contact surfaces.

        Compared as a point SET via symmetric nearest-neighbour distance, not
        element-wise: an STL has no vertex indexing, so a read→write→read
        round-trip can permute the vertex array while the geometry is
        bit-identical. Element-wise comparison would flag that harmless
        permutation as a failure.
        """
        import pymskt as mskt
        from scipy.spatial import cKDTree

        base_geom = base_comak_path.parent / "Geometry"
        scaled_geom = identity_scaled_model.parent / "Geometry"
        for fname in (
            "smith2019-R-femur-cartilage.stl",
            "smith2019-R-tibia-cartilage.stl",
            "smith2019-R-patella-cartilage.stl",
            "smith2019-R-femur-bone.stl",
            "smith2019-R-tibia-bone.stl",
            "smith2019-R-patella-bone.stl",
            "smith2019-R-medial-meniscus.stl",
            "smith2019-R-lateral-meniscus.stl",
        ):
            pb = np.asarray(mskt.mesh.Mesh(str(base_geom / fname)).point_coords)
            ps = np.asarray(mskt.mesh.Mesh(str(scaled_geom / fname)).point_coords)
            assert pb.shape == ps.shape, f"{fname}: vertex count changed"
            d_sb, _ = cKDTree(pb).query(ps, k=1)
            d_bs, _ = cKDTree(ps).query(pb, k=1)
            max_nn = max(float(d_sb.max()), float(d_bs.max()))
            assert max_nn < 1e-6, (
                f"{fname} geometry drifted at identity "
                f"(max nearest-neighbour distance {max_nn:.3e} m)"
            )
