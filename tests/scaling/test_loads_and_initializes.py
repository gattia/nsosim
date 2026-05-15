"""Smoke tests: pipeline output is a loadable OpenSim model with all expected bits."""

from pathlib import Path

import opensim as osim
import pytest

from nsosim.scaling.config import WA_KNEE_BODIES


@pytest.mark.slow
class TestIdentityModelLoads:
    def test_model_loads_and_initSystem_succeeds(self, identity_scaled_model: Path):
        m = osim.Model(str(identity_scaled_model))
        m.initSystem()
        assert m.getBodySet().getSize() == 27

    def test_seven_contact_meshes_preserved(self, identity_scaled_model: Path):
        m = osim.Model(str(identity_scaled_model))
        m.initSystem()
        cg = m.getContactGeometrySet()
        smith_count = sum(
            1
            for i in range(cg.getSize())
            if cg.get(i).getConcreteClassName() == "Smith2018ContactMesh"
        )
        assert smith_count == 7

    def test_markers_transplanted_from_ab(self, identity_scaled_model: Path):
        m = osim.Model(str(identity_scaled_model))
        m.initSystem()
        # The identity AB input is the stripped Tian base, which carries 41 markers.
        assert m.getMarkerSet().getSize() == 41

    def test_itb1_distal_on_tibia_proximal_r(self, identity_scaled_model: Path):
        m = osim.Model(str(identity_scaled_model))
        m.initSystem()
        itb1 = osim.Blankevoort1991Ligament.safeDownCast(m.upd_ForceSet().get("ITB1"))
        pp2 = osim.PathPoint.safeDownCast(itb1.updGeometryPath().getPathPointSet().get(2))
        assert pp2.getParentFrame().getName() == "tibia_proximal_r"


@pytest.mark.slow
class TestSubjectModelLoads:
    def test_model_loads_and_initSystem_succeeds(self, subject_scaled_model: Path):
        m = osim.Model(str(subject_scaled_model))
        m.initSystem()
        assert m.getBodySet().getSize() == 27
        assert m.getMarkerSet().getSize() == 41
        cg = m.getContactGeometrySet()
        smith_count = sum(
            1
            for i in range(cg.getSize())
            if cg.get(i).getConcreteClassName() == "Smith2018ContactMesh"
        )
        assert smith_count == 7

    def test_knee_visual_geometry_baked(self, subject_scaled_model: Path):
        """Every visual mesh on a knee body must have scale_factors == [1,1,1]:
        knee geometry is baked into the STLs, not scaled by OpenSim. A non-unit
        factor here means the rendered knee would be double-scaled."""
        m = osim.Model(str(subject_scaled_model))
        m.initSystem()
        bs = m.getBodySet()
        n_checked = 0
        for bi in range(bs.getSize()):
            body = bs.get(bi)
            if body.getName() not in WA_KNEE_BODIES:
                continue
            for gi in range(body.getPropertyByName("attached_geometry").size()):
                mesh = osim.Mesh.safeDownCast(body.upd_attached_geometry(gi))
                if mesh is None:
                    continue
                sf = mesh.get_scale_factors()
                assert (sf[0], sf[1], sf[2]) == pytest.approx(
                    (1.0, 1.0, 1.0), abs=1e-12
                ), f"{body.getName()}/{mesh.get_mesh_file()} not baked: {sf}"
                n_checked += 1
        assert n_checked >= 6, f"expected >=6 knee visual meshes, checked {n_checked}"
