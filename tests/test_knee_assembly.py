"""
Tests for knee_assembly data classes, serialization, and strip/add.

Phase 1: dataclass construction, to_dict/from_dict, JSON round-trip.
Phase 3: strip_comak_knee() against the real Smith2019 model.
"""

import json
import math
from pathlib import Path

import pytest

from nsosim.knee_assembly import (
    ComakBody,
    ComakContactForce,
    ComakContactMesh,
    ComakCoordinate,
    ComakCustomJoint,
    ComakKneeConfig,
    ComakLigament,
    ComakMuscle,
    ComakSpring,
    ComakWeldJoint,
    ComakWrapSurface,
)

# ---------------------------------------------------------------------------
# Fixtures — representative data from Phase 0 audit of Smith2019
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_body():
    return ComakBody(
        name="femur_distal_r",
        mass=0.008166,
        inertia=[0.001, 0.001, 0.001, 0.0, 0.0, 0.0],
        mass_center=[0.0, 0.0, 0.0],
        attached_geometry=[{"name": "femur_distal_r_geom", "mesh_file": "femur_distal_r.vtp"}],
    )


@pytest.fixture
def sample_weld_joint():
    return ComakWeldJoint(
        name="femur_femur_distal_r",
        parent_body="femur_r",
        child_body="femur_distal_r",
        parent_offset_translation=[-0.0056, -0.3742, -0.0012],
        parent_offset_orientation=[0.0, 0.0, 0.0],
        child_offset_translation=[0.0, 0.0, 0.0],
        child_offset_orientation=[0.0, 0.0, 0.0],
    )


@pytest.fixture
def sample_coordinate():
    return ComakCoordinate(
        name="knee_flex_r",
        default_value=0.0,
        range_min=-2.094,
        range_max=0.175,
        locked=False,
        clamped=True,
    )


@pytest.fixture
def sample_custom_joint(sample_coordinate):
    return ComakCustomJoint(
        name="knee_r",
        parent_body="femur_distal_r",
        child_body="tibia_proximal_r",
        parent_offset_translation=[0.0, 0.0, 0.0],
        parent_offset_orientation=[0.0, 0.0, 0.0],
        child_offset_translation=[0.0, 0.0, 0.0],
        child_offset_orientation=[0.0, 0.0, 0.0],
        coordinates=[sample_coordinate],
        spatial_transform={
            "rotation1": {
                "axis": [0, 0, 1],
                "coordinate": "knee_flex_r",
                "function": {"type": "LinearFunction", "slope": 1.0, "intercept": 0.0},
            },
        },
    )


@pytest.fixture
def sample_ligament():
    return ComakLigament(
        name="MCLd1",
        linear_stiffness=5000.0,
        transition_strain=0.06,
        damping_coefficient=0.003,
        slack_length=0.05,
        path_points=[
            {"name": "MCLd1-P1", "body": "femur_distal_r", "location": [0.01, -0.02, 0.03]},
            {"name": "MCLd1-P2", "body": "tibia_proximal_r", "location": [-0.01, 0.02, -0.03]},
        ],
    )


@pytest.fixture
def sample_spring():
    return ComakSpring(
        name="knee_add_spring",
        coordinate="knee_add_r",
        stiffness=1.0,
        rest_length=0.0,
        viscosity=0.0,
    )


@pytest.fixture
def sample_contact_mesh():
    return ComakContactMesh(
        name="femur_cartilage",
        parent_frame="femur_distal_r",
        mesh_file="femur_cartilage.stl",
        elastic_modulus=1e6,
        poissons_ratio=0.5,
        thickness=0.005,
        location=[0.0, 0.0, 0.0],
        orientation=[0.0, 0.0, 0.0],
    )


@pytest.fixture
def sample_contact_force():
    return ComakContactForce(
        name="tf_contact",
        target_mesh="femur_cartilage",
        casting_mesh="tibia_cartilage",
    )


@pytest.fixture
def sample_wrap_surface():
    return ComakWrapSurface(
        name="Capsule_r",
        parent_body="femur_distal_r",
        type="WrapCylinder",
        translation=[0.001, -0.005, 0.0],
        xyz_body_rotation=[0.0, 0.0, 1.57],
        quadrant="x",
        radius=0.02,
        length=0.1,
    )


@pytest.fixture
def sample_muscle():
    return ComakMuscle(
        name="recfem_r",
        max_isometric_force=848.8,
        optimal_fiber_length=0.07097,
        tendon_slack_length=0.34053,
        pennation_angle_at_optimal=0.243124,
        max_contraction_velocity=10.0,
        path_points=[
            {"name": "recfem_r-P1", "body": "pelvis", "location": [-0.0287, -0.0411, 0.1116]},
            {"name": "recfem_r-P2", "body": "patella_r", "location": [-0.0001, 0.0150, 0.0011]},
            {"name": "recfem_r-P3", "body": "patella_r", "location": [0.0035, 0.0126, 0.0008]},
        ],
        wrap_objects=["KnExt_at_fem_r"],
    )


@pytest.fixture
def sample_config(
    sample_body,
    sample_weld_joint,
    sample_custom_joint,
    sample_ligament,
    sample_spring,
    sample_contact_mesh,
    sample_contact_force,
    sample_wrap_surface,
    sample_muscle,
):
    return ComakKneeConfig(
        side="r",
        bodies=[sample_body],
        weld_joints=[sample_weld_joint],
        custom_joints=[sample_custom_joint],
        ligaments=[sample_ligament],
        springs=[sample_spring],
        contact_meshes=[sample_contact_mesh],
        contact_forces=[sample_contact_force],
        wrap_surfaces=[sample_wrap_surface],
        spanning_muscles=[sample_muscle],
        ref_femur_length=0.377,
        ref_tibia_length=0.403,
    )


# ---------------------------------------------------------------------------
# Tests — individual dataclasses
# ---------------------------------------------------------------------------


class TestComakBody:
    def test_construction(self, sample_body):
        assert sample_body.name == "femur_distal_r"
        assert sample_body.mass == 0.008166
        assert len(sample_body.inertia) == 6
        assert len(sample_body.mass_center) == 3
        assert len(sample_body.attached_geometry) == 1

    def test_default_geometry(self):
        body = ComakBody(name="test", mass=1.0, inertia=[0] * 6, mass_center=[0, 0, 0])
        assert body.attached_geometry == []


class TestComakWeldJoint:
    def test_construction(self, sample_weld_joint):
        assert sample_weld_joint.name == "femur_femur_distal_r"
        assert sample_weld_joint.parent_body == "femur_r"
        assert sample_weld_joint.child_body == "femur_distal_r"
        assert sample_weld_joint.parent_offset_translation[1] == pytest.approx(-0.3742)


class TestComakCustomJoint:
    def test_construction(self, sample_custom_joint):
        assert sample_custom_joint.name == "knee_r"
        assert len(sample_custom_joint.coordinates) == 1
        assert sample_custom_joint.coordinates[0].name == "knee_flex_r"
        assert "rotation1" in sample_custom_joint.spatial_transform


class TestComakLigament:
    def test_construction(self, sample_ligament):
        assert sample_ligament.name == "MCLd1"
        assert sample_ligament.linear_stiffness == 5000.0
        assert len(sample_ligament.path_points) == 2


class TestComakContactMesh:
    def test_defaults(self):
        mesh = ComakContactMesh(
            name="test",
            parent_frame="body",
            mesh_file="test.stl",
            elastic_modulus=1e6,
            poissons_ratio=0.5,
            thickness=0.005,
            location=[0, 0, 0],
            orientation=[0, 0, 0],
        )
        assert mesh.use_variable_thickness is False
        assert mesh.mesh_back_file == ""
        assert mesh.scale_factors == [1.0, 1.0, 1.0]


class TestComakContactForce:
    def test_defaults(self):
        force = ComakContactForce(name="test", target_mesh="a", casting_mesh="b")
        assert force.min_proximity == 0.0
        assert force.max_proximity == 0.01
        assert force.elastic_foundation_formulation == "linear"
        assert force.use_lumped_contact_model is True


class TestComakWrapSurface:
    def test_cylinder(self, sample_wrap_surface):
        assert sample_wrap_surface.type == "WrapCylinder"
        assert sample_wrap_surface.radius == 0.02
        assert sample_wrap_surface.length == 0.1
        assert sample_wrap_surface.dimensions is None

    def test_ellipsoid(self):
        ell = ComakWrapSurface(
            name="test_ell",
            parent_body="tibia_proximal_r",
            type="WrapEllipsoid",
            translation=[0, 0, 0],
            xyz_body_rotation=[0, 0, 0],
            quadrant="all",
            dimensions=[0.02, 0.03, 0.04],
        )
        assert ell.type == "WrapEllipsoid"
        assert ell.dimensions == [0.02, 0.03, 0.04]
        assert ell.radius is None


class TestComakMuscle:
    def test_construction(self, sample_muscle):
        assert sample_muscle.name == "recfem_r"
        assert sample_muscle.max_isometric_force == 848.8
        assert len(sample_muscle.path_points) == 3
        assert sample_muscle.wrap_objects == ["KnExt_at_fem_r"]

    def test_defaults(self):
        m = ComakMuscle(
            name="test",
            max_isometric_force=100.0,
            optimal_fiber_length=0.1,
            tendon_slack_length=0.2,
            pennation_angle_at_optimal=0.0,
        )
        assert m.max_contraction_velocity == 10.0
        assert m.fiber_damping == 0.1
        assert m.minimum_activation == 0.01
        assert m.maximum_pennation_angle == pytest.approx(1.47063)


# ---------------------------------------------------------------------------
# Tests — ComakKneeConfig serialization
# ---------------------------------------------------------------------------


class TestComakKneeConfigSerialization:
    def test_to_dict_returns_plain_dict(self, sample_config):
        d = sample_config.to_dict()
        assert isinstance(d, dict)
        assert d["side"] == "r"
        assert isinstance(d["bodies"][0], dict)
        assert d["bodies"][0]["name"] == "femur_distal_r"

    def test_dict_roundtrip(self, sample_config):
        d = sample_config.to_dict()
        restored = ComakKneeConfig.from_dict(d)

        assert restored.side == sample_config.side
        assert restored.ref_femur_length == sample_config.ref_femur_length
        assert restored.ref_tibia_length == sample_config.ref_tibia_length

        # Bodies
        assert len(restored.bodies) == len(sample_config.bodies)
        assert restored.bodies[0].name == sample_config.bodies[0].name
        assert restored.bodies[0].mass == sample_config.bodies[0].mass

        # Weld joints
        assert len(restored.weld_joints) == len(sample_config.weld_joints)
        assert restored.weld_joints[0].parent_body == sample_config.weld_joints[0].parent_body

        # Custom joints (nested coordinates)
        assert len(restored.custom_joints) == len(sample_config.custom_joints)
        orig_coord = sample_config.custom_joints[0].coordinates[0]
        rest_coord = restored.custom_joints[0].coordinates[0]
        assert rest_coord.name == orig_coord.name
        assert rest_coord.locked == orig_coord.locked

        # Ligaments
        assert len(restored.ligaments) == len(sample_config.ligaments)
        assert restored.ligaments[0].linear_stiffness == sample_config.ligaments[0].linear_stiffness

        # Springs
        assert restored.springs[0].coordinate == sample_config.springs[0].coordinate

        # Contact meshes
        assert (
            restored.contact_meshes[0].elastic_modulus
            == sample_config.contact_meshes[0].elastic_modulus
        )

        # Contact forces
        assert restored.contact_forces[0].target_mesh == sample_config.contact_forces[0].target_mesh

        # Wrap surfaces
        assert restored.wrap_surfaces[0].type == sample_config.wrap_surfaces[0].type
        assert restored.wrap_surfaces[0].radius == sample_config.wrap_surfaces[0].radius

        # Spanning muscles
        assert restored.spanning_muscles[0].name == sample_config.spanning_muscles[0].name
        assert len(restored.spanning_muscles[0].path_points) == 3
        assert restored.spanning_muscles[0].wrap_objects == ["KnExt_at_fem_r"]

    def test_json_roundtrip(self, sample_config, tmp_path):
        json_path = str(tmp_path / "test_config.json")
        sample_config.to_json(json_path)

        restored = ComakKneeConfig.from_json(json_path)

        # Spot-check across all component types
        assert restored.side == "r"
        assert restored.bodies[0].mass == pytest.approx(0.008166)
        assert restored.weld_joints[0].parent_offset_translation[1] == pytest.approx(-0.3742)
        assert restored.custom_joints[0].spatial_transform["rotation1"]["function"]["slope"] == 1.0
        assert restored.ligaments[0].slack_length == pytest.approx(0.05)
        assert restored.springs[0].stiffness == 1.0
        assert restored.contact_meshes[0].mesh_file == "femur_cartilage.stl"
        assert restored.contact_forces[0].casting_mesh == "tibia_cartilage"
        assert restored.wrap_surfaces[0].radius == pytest.approx(0.02)
        assert restored.spanning_muscles[0].max_isometric_force == pytest.approx(848.8)
        assert restored.ref_femur_length == pytest.approx(0.377)
        assert restored.ref_tibia_length == pytest.approx(0.403)

    def test_json_is_valid_json(self, sample_config, tmp_path):
        json_path = str(tmp_path / "test_config.json")
        sample_config.to_json(json_path)

        with open(json_path) as f:
            raw = json.load(f)

        assert isinstance(raw, dict)
        assert "bodies" in raw
        assert "spanning_muscles" in raw

    def test_from_dict_types_are_dataclasses(self, sample_config):
        """Verify from_dict produces actual dataclass instances, not plain dicts."""
        d = sample_config.to_dict()
        restored = ComakKneeConfig.from_dict(d)

        assert isinstance(restored.bodies[0], ComakBody)
        assert isinstance(restored.weld_joints[0], ComakWeldJoint)
        assert isinstance(restored.custom_joints[0], ComakCustomJoint)
        assert isinstance(restored.custom_joints[0].coordinates[0], ComakCoordinate)
        assert isinstance(restored.ligaments[0], ComakLigament)
        assert isinstance(restored.springs[0], ComakSpring)
        assert isinstance(restored.contact_meshes[0], ComakContactMesh)
        assert isinstance(restored.contact_forces[0], ComakContactForce)
        assert isinstance(restored.wrap_surfaces[0], ComakWrapSurface)
        assert isinstance(restored.spanning_muscles[0], ComakMuscle)


# ---------------------------------------------------------------------------
# Phase 3: strip_comak_knee() tests
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SMITH2019_MODEL_PATH = FIXTURES_DIR / "osim_models" / "full_body_healthy_knee.osim"

requires_smith2019 = pytest.mark.skipif(
    not SMITH2019_MODEL_PATH.exists(),
    reason=f"Smith2019 model not found at {SMITH2019_MODEL_PATH}",
)

try:
    import opensim as osim

    HAS_OPENSIM = True
except ImportError:
    HAS_OPENSIM = False

requires_opensim = pytest.mark.skipif(not HAS_OPENSIM, reason="opensim not available")


@requires_opensim
@requires_smith2019
class TestStripComakKnee:
    """Test strip_comak_knee() against the real Smith2019 model."""

    @pytest.fixture(scope="class")
    def strip_result(self):
        """Run strip once and share across all tests in this class."""
        from nsosim.knee_assembly import strip_comak_knee

        model = osim.Model(str(SMITH2019_MODEL_PATH))
        stripped_model, config = strip_comak_knee(model, side="r")
        return stripped_model, config

    @pytest.fixture
    def stripped_model(self, strip_result):
        return strip_result[0]

    @pytest.fixture
    def config(self, strip_result):
        return strip_result[1]

    # --- Config component counts (from Phase 0 audit) ---

    def test_config_side(self, config):
        assert config.side == "r"

    def test_config_body_count(self, config):
        assert len(config.bodies) == 5

    def test_config_body_names(self, config):
        names = {b.name for b in config.bodies}
        assert names == {
            "femur_distal_r",
            "tibia_proximal_r",
            "patella_r",
            "meniscus_medial_r",
            "meniscus_lateral_r",
        }

    def test_config_weld_joint_count(self, config):
        assert len(config.weld_joints) == 2

    def test_config_custom_joint_count(self, config):
        assert len(config.custom_joints) == 4

    def test_config_custom_joint_names(self, config):
        names = {j.name for j in config.custom_joints}
        assert names == {"knee_r", "pf_r", "meniscus_medial_r", "meniscus_lateral_r"}

    def test_config_ligament_count(self, config):
        assert len(config.ligaments) == 91

    def test_config_spring_count(self, config):
        assert len(config.springs) == 24

    def test_config_contact_mesh_count(self, config):
        assert len(config.contact_meshes) == 7

    def test_config_contact_force_count(self, config):
        assert len(config.contact_forces) == 6

    def test_config_wrap_surface_count(self, config):
        assert len(config.wrap_surfaces) == 4

    def test_config_spanning_muscle_count(self, config):
        assert len(config.spanning_muscles) == 4

    def test_config_spanning_muscle_names(self, config):
        names = {m.name for m in config.spanning_muscles}
        assert names == {"recfem_r", "vasint_r", "vaslat_r", "vasmed_r"}

    # --- Known values from Phase 0 audit ---

    def test_femur_distal_mass(self, config):
        fem_dist = next(b for b in config.bodies if b.name == "femur_distal_r")
        assert fem_dist.mass == pytest.approx(0.008166, abs=1e-5)

    def test_patella_mass(self, config):
        patella = next(b for b in config.bodies if b.name == "patella_r")
        assert patella.mass == pytest.approx(0.398116, abs=1e-5)

    def test_weld_joint_offset(self, config):
        fem_weld = next(wj for wj in config.weld_joints if wj.name == "femur_femur_distal_r")
        assert fem_weld.parent_body == "femur_r"
        assert fem_weld.child_body == "femur_distal_r"
        assert fem_weld.parent_offset_translation[1] == pytest.approx(-0.374181, abs=1e-4)

    def test_knee_joint_has_6_coords(self, config):
        knee = next(j for j in config.custom_joints if j.name == "knee_r")
        assert len(knee.coordinates) == 6

    def test_knee_joint_spatial_transform_all_linear(self, config):
        knee = next(j for j in config.custom_joints if j.name == "knee_r")
        for axis_name, axis_data in knee.spatial_transform.items():
            assert axis_data["function"]["type"] == "LinearFunction"

    def test_recfem_properties(self, config):
        recfem = next(m for m in config.spanning_muscles if m.name == "recfem_r")
        assert recfem.max_isometric_force == pytest.approx(848.8)
        assert len(recfem.path_points) == 3
        assert recfem.wrap_objects == ["KnExt_at_fem_r"]

    def test_segment_lengths(self, config):
        assert config.ref_femur_length == pytest.approx(0.377, abs=0.001)
        assert config.ref_tibia_length == pytest.approx(0.403, abs=0.001)

    def test_wrap_surface_capsule(self, config):
        capsule = next(ws for ws in config.wrap_surfaces if ws.name == "Capsule_r")
        assert capsule.parent_body == "femur_distal_r"
        assert capsule.type == "WrapCylinder"
        assert capsule.radius == pytest.approx(0.018, abs=1e-4)

    # --- Stripped model validation ---

    def test_stripped_model_has_no_comak_ligaments(self, stripped_model):
        force_set = stripped_model.getForceSet()
        for i in range(force_set.getSize()):
            assert force_set.get(i).getConcreteClassName() != "Blankevoort1991Ligament"

    def test_stripped_model_has_no_comak_springs(self, stripped_model):
        force_set = stripped_model.getForceSet()
        for i in range(force_set.getSize()):
            assert force_set.get(i).getConcreteClassName() != "SpringGeneralizedForce"

    def test_stripped_model_has_no_comak_contact_forces(self, stripped_model):
        force_set = stripped_model.getForceSet()
        for i in range(force_set.getSize()):
            assert force_set.get(i).getConcreteClassName() != "Smith2018ArticularContactForce"

    def test_stripped_model_has_no_comak_contact_meshes(self, stripped_model):
        cg_set = stripped_model.getContactGeometrySet()
        for i in range(cg_set.getSize()):
            assert cg_set.get(i).getConcreteClassName() != "Smith2018ContactMesh"

    def test_stripped_model_has_no_spanning_muscles(self, stripped_model):
        force_set = stripped_model.getForceSet()
        spanning = {"recfem_r", "vasint_r", "vaslat_r", "vasmed_r"}
        for i in range(force_set.getSize()):
            assert force_set.get(i).getName() not in spanning

    def test_stripped_model_has_no_comak_bodies(self, stripped_model):
        body_set = stripped_model.getBodySet()
        comak_bodies = {
            "femur_distal_r",
            "tibia_proximal_r",
            "patella_r",
            "meniscus_medial_r",
            "meniscus_lateral_r",
        }
        for i in range(body_set.getSize()):
            assert body_set.get(i).getName() not in comak_bodies

    def test_stripped_model_retains_non_comak_muscles(self, stripped_model):
        """Non-COMAK muscles (hamstrings, gastroc, etc.) should remain."""
        force_set = stripped_model.getForceSet()
        muscle_names = set()
        for i in range(force_set.getSize()):
            f = force_set.get(i)
            if f.getConcreteClassName() == "Millard2012EquilibriumMuscle":
                muscle_names.add(f.getName())
        assert "gaslat_r" in muscle_names
        assert "gasmed_r" in muscle_names
        assert "bflh_r" in muscle_names
        assert "semimem_r" in muscle_names

    def test_stripped_model_retains_femur_wrap_surfaces(self, stripped_model):
        """Wrap surfaces on femur_r (used by non-COMAK muscles) should remain."""
        body_set = stripped_model.getBodySet()
        femur = None
        for i in range(body_set.getSize()):
            if body_set.get(i).getName() == "femur_r":
                femur = body_set.get(i)
                break
        assert femur is not None
        ws = femur.get_WrapObjectSet()
        wrap_names = {ws.get(i).getName() for i in range(ws.getSize())}
        assert "Gastroc_at_Condyles_r" in wrap_names
        assert "KnExt_at_fem_r" in wrap_names

    # --- Config JSON round-trip ---

    def test_config_json_roundtrip(self, config, tmp_path):
        json_path = str(tmp_path / "smith2019_knee_config.json")
        config.to_json(json_path)
        restored = ComakKneeConfig.from_json(json_path)

        assert len(restored.bodies) == len(config.bodies)
        assert len(restored.ligaments) == len(config.ligaments)
        assert len(restored.springs) == len(config.springs)
        assert len(restored.spanning_muscles) == len(config.spanning_muscles)
        assert restored.ref_femur_length == pytest.approx(config.ref_femur_length)


# ---------------------------------------------------------------------------
# Phase 5: Full strip → add round-trip test
# ---------------------------------------------------------------------------


@requires_opensim
@requires_smith2019
class TestRoundTrip:
    """Strip Smith2019, add back, compare component by component."""

    @pytest.fixture(scope="class")
    def roundtrip_models(self):
        """Load original, strip, add back. Shared across all tests."""
        from nsosim.knee_assembly import add_comak_knee, strip_comak_knee

        original = osim.Model(str(SMITH2019_MODEL_PATH))
        original.initSystem()

        to_strip = osim.Model(str(SMITH2019_MODEL_PATH))
        stripped, config = strip_comak_knee(to_strip, side="r")
        rebuilt = add_comak_knee(stripped, config)
        rebuilt.finalizeConnections()

        return original, rebuilt, config

    @pytest.fixture
    def original(self, roundtrip_models):
        return roundtrip_models[0]

    @pytest.fixture
    def rebuilt(self, roundtrip_models):
        return roundtrip_models[1]

    @pytest.fixture
    def config(self, roundtrip_models):
        return roundtrip_models[2]

    # --- Component counts ---

    def test_body_count(self, original, rebuilt):
        assert rebuilt.getBodySet().getSize() == original.getBodySet().getSize()

    def test_joint_count(self, original, rebuilt):
        assert rebuilt.getJointSet().getSize() == original.getJointSet().getSize()

    def test_force_count(self, original, rebuilt):
        assert rebuilt.getForceSet().getSize() == original.getForceSet().getSize()

    def test_contact_geometry_count(self, original, rebuilt):
        assert (
            rebuilt.getContactGeometrySet().getSize() == original.getContactGeometrySet().getSize()
        )

    def test_coordinate_count(self, original, rebuilt):
        assert rebuilt.getCoordinateSet().getSize() == original.getCoordinateSet().getSize()

    # --- Body properties ---

    def test_body_names_match(self, original, rebuilt):
        orig_names = {
            original.getBodySet().get(i).getName() for i in range(original.getBodySet().getSize())
        }
        rebuilt_names = {
            rebuilt.getBodySet().get(i).getName() for i in range(rebuilt.getBodySet().getSize())
        }
        assert rebuilt_names == orig_names

    def test_body_masses_match(self, original, rebuilt):
        for i in range(original.getBodySet().getSize()):
            orig = original.getBodySet().get(i)
            name = orig.getName()
            # Find in rebuilt
            for j in range(rebuilt.getBodySet().getSize()):
                if rebuilt.getBodySet().get(j).getName() == name:
                    assert rebuilt.getBodySet().get(j).getMass() == pytest.approx(
                        orig.getMass(), abs=1e-6
                    ), f"Mass mismatch for {name}"
                    break

    # --- Joint properties ---

    def test_joint_names_match(self, original, rebuilt):
        orig_names = {
            original.getJointSet().get(i).getName() for i in range(original.getJointSet().getSize())
        }
        rebuilt_names = {
            rebuilt.getJointSet().get(i).getName() for i in range(rebuilt.getJointSet().getSize())
        }
        assert rebuilt_names == orig_names

    def test_joint_types_match(self, original, rebuilt):
        for i in range(original.getJointSet().getSize()):
            orig = original.getJointSet().get(i)
            name = orig.getName()
            for j in range(rebuilt.getJointSet().getSize()):
                rb = rebuilt.getJointSet().get(j)
                if rb.getName() == name:
                    assert rb.getConcreteClassName() == orig.getConcreteClassName(), (
                        f"Joint type mismatch for {name}: "
                        f"{rb.getConcreteClassName()} vs {orig.getConcreteClassName()}"
                    )
                    break

    # --- Coordinate properties ---

    def test_coordinate_names_match(self, original, rebuilt):
        orig_names = {
            original.getCoordinateSet().get(i).getName()
            for i in range(original.getCoordinateSet().getSize())
        }
        rebuilt_names = {
            rebuilt.getCoordinateSet().get(i).getName()
            for i in range(rebuilt.getCoordinateSet().getSize())
        }
        assert rebuilt_names == orig_names

    def test_coordinate_defaults_match(self, original, rebuilt):
        for i in range(original.getCoordinateSet().getSize()):
            orig = original.getCoordinateSet().get(i)
            name = orig.getName()
            for j in range(rebuilt.getCoordinateSet().getSize()):
                rb = rebuilt.getCoordinateSet().get(j)
                if rb.getName() == name:
                    assert rb.getDefaultValue() == pytest.approx(
                        orig.getDefaultValue(), abs=1e-6
                    ), f"Default value mismatch for {name}"
                    break

    # --- Force properties ---

    def test_force_names_match(self, original, rebuilt):
        orig_names = {
            original.getForceSet().get(i).getName() for i in range(original.getForceSet().getSize())
        }
        rebuilt_names = {
            rebuilt.getForceSet().get(i).getName() for i in range(rebuilt.getForceSet().getSize())
        }
        assert rebuilt_names == orig_names

    def test_ligament_properties_match(self, original, rebuilt):
        """Spot-check a few ligaments for stiffness and slack length."""
        for name in ["MCLd1", "ACLam1", "PT1", "ITB1"]:
            orig_lig = osim.Blankevoort1991Ligament.safeDownCast(original.getForceSet().get(name))
            rebuilt_lig = osim.Blankevoort1991Ligament.safeDownCast(rebuilt.getForceSet().get(name))
            assert rebuilt_lig is not None, f"Ligament {name} not found in rebuilt"
            assert rebuilt_lig.get_linear_stiffness() == pytest.approx(
                orig_lig.get_linear_stiffness(), abs=1e-6
            ), f"Stiffness mismatch for {name}"
            assert rebuilt_lig.get_slack_length() == pytest.approx(
                orig_lig.get_slack_length(), abs=1e-6
            ), f"Slack length mismatch for {name}"

    def test_spring_properties_match(self, original, rebuilt):
        """Spot-check springs."""
        for name in ["knee_flex_r", "pf_flex_r"]:
            orig_sp = osim.SpringGeneralizedForce.safeDownCast(original.getForceSet().get(name))
            rebuilt_sp = osim.SpringGeneralizedForce.safeDownCast(rebuilt.getForceSet().get(name))
            assert rebuilt_sp is not None, f"Spring {name} not found"
            assert rebuilt_sp.get_stiffness() == pytest.approx(
                orig_sp.get_stiffness(), abs=1e-6
            ), f"Stiffness mismatch for {name}"

    def test_spanning_muscle_properties_match(self, original, rebuilt):
        """Check recfem_r round-trips correctly."""
        orig = osim.Millard2012EquilibriumMuscle.safeDownCast(
            original.getForceSet().get("recfem_r")
        )
        rb = osim.Millard2012EquilibriumMuscle.safeDownCast(rebuilt.getForceSet().get("recfem_r"))
        assert rb is not None
        assert rb.get_max_isometric_force() == pytest.approx(
            orig.get_max_isometric_force(), abs=1e-6
        )
        assert rb.get_optimal_fiber_length() == pytest.approx(
            orig.get_optimal_fiber_length(), abs=1e-6
        )
        assert rb.get_tendon_slack_length() == pytest.approx(
            orig.get_tendon_slack_length(), abs=1e-6
        )

    # --- Contact ---

    def test_contact_mesh_names_match(self, original, rebuilt):
        orig_cg = original.getContactGeometrySet()
        rebuilt_cg = rebuilt.getContactGeometrySet()
        orig_names = {orig_cg.get(i).getName() for i in range(orig_cg.getSize())}
        rebuilt_names = {rebuilt_cg.get(i).getName() for i in range(rebuilt_cg.getSize())}
        assert rebuilt_names == orig_names

    # --- Wrap surfaces on COMAK bodies ---

    def test_comak_body_wrap_surfaces_match(self, original, rebuilt):
        """Wrap surfaces on COMAK bodies should be restored."""
        for body_name in ["femur_distal_r", "tibia_proximal_r", "patella_r"]:
            orig_body = original.getBodySet().get(body_name)
            rebuilt_body = rebuilt.getBodySet().get(body_name)
            orig_ws = orig_body.get_WrapObjectSet()
            rebuilt_ws = rebuilt_body.get_WrapObjectSet()
            assert rebuilt_ws.getSize() == orig_ws.getSize(), (
                f"Wrap count mismatch on {body_name}: "
                f"{rebuilt_ws.getSize()} vs {orig_ws.getSize()}"
            )
            orig_names = {orig_ws.get(i).getName() for i in range(orig_ws.getSize())}
            rebuilt_names = {rebuilt_ws.get(i).getName() for i in range(rebuilt_ws.getSize())}
            assert rebuilt_names == orig_names, f"Wrap names mismatch on {body_name}"

    # --- initSystem succeeds ---

    def test_rebuilt_model_init_system(self, rebuilt):
        """The rebuilt model should pass initSystem without error."""
        rebuilt.initSystem()
