"""Tests for meniscal ligament tibia attachment projection."""

import numpy as np
import pytest
import pyvista as pv

from nsosim.meniscal_ligaments import (
    _identify_tibia_meniscus_points,
    _is_meniscal_tibia_ligament,
    project_meniscal_attachments_to_tibia,
)

# ---------------------------------------------------------------------------
# Helper to build a flat tibia mesh (plane at Y=0, spanning XZ)
# ---------------------------------------------------------------------------


def _make_flat_tibia(y=0.0, extent=0.1):
    """Create a flat square mesh at the given Y height."""
    plane = pv.Plane(
        center=(0, y, 0),
        direction=(0, 1, 0),
        i_size=extent * 2,
        j_size=extent * 2,
        i_resolution=10,
        j_resolution=10,
    )
    return plane


def _make_ligament_entry(
    name,
    tibia_xyz,
    meniscus_xyz,
    tibia_frame="tibia_proximal_r",
    meniscus_frame="meniscus_medial_r",
):
    """Build a minimal ligament dict entry for testing."""
    return {
        "name": name,
        "class": "Blankevoort1991Ligament",
        "points": [
            {
                "name": f"{name}-P1",
                "parent_frame": tibia_frame,
                "include": True,
                "xyz_mesh_updated": np.array(tibia_xyz, dtype=float),
            },
            {
                "name": f"{name}-P2",
                "parent_frame": meniscus_frame,
                "include": True,
                "xyz_mesh_updated": np.array(meniscus_xyz, dtype=float),
            },
        ],
    }


# ---------------------------------------------------------------------------
# Tests for _is_meniscal_tibia_ligament
# ---------------------------------------------------------------------------


class TestIsMeniscalTibiaLigament:
    def test_coronary_medial(self):
        assert _is_meniscal_tibia_ligament("meniscus_medial_COR1") is True

    def test_anterior_horn_lateral(self):
        assert _is_meniscal_tibia_ligament("meniscus_lateral_AHORN1") is True

    def test_posterior_horn_medial(self):
        assert _is_meniscal_tibia_ligament("meniscus_medial_PHORN2") is True

    def test_transverse_excluded(self):
        assert _is_meniscal_tibia_ligament("meniscus_TRANSVLIG1") is False

    def test_non_meniscal(self):
        assert _is_meniscal_tibia_ligament("MCLd1") is False
        assert _is_meniscal_tibia_ligament("ACLam") is False


# ---------------------------------------------------------------------------
# Tests for _identify_tibia_meniscus_points
# ---------------------------------------------------------------------------


class TestIdentifyTibiaMeniscusPoints:
    def test_standard_order(self):
        points = [
            {"parent_frame": "tibia_proximal_r"},
            {"parent_frame": "meniscus_medial_r"},
        ]
        assert _identify_tibia_meniscus_points(points) == (0, 1)

    def test_reversed_order(self):
        points = [
            {"parent_frame": "meniscus_lateral_r"},
            {"parent_frame": "tibia_proximal_r"},
        ]
        assert _identify_tibia_meniscus_points(points) == (1, 0)

    def test_unknown_frames(self):
        points = [
            {"parent_frame": "femur_r"},
            {"parent_frame": "patella_r"},
        ]
        assert _identify_tibia_meniscus_points(points) == (None, None)


# ---------------------------------------------------------------------------
# Tests for project_meniscal_attachments_to_tibia
# ---------------------------------------------------------------------------


class TestProjectMeniscalAttachments:
    @pytest.fixture
    def flat_tibia(self):
        """Flat tibia mesh at Y=0."""
        return _make_flat_tibia(y=0.0)

    def test_ray_hit_flat_plane(self, flat_tibia):
        """Meniscus point above a flat tibia → ray hits at same XZ, Y=0."""
        meniscus_xyz = [0.02, 0.005, 0.01]  # 5mm above the plane
        tibia_xyz_orig = [0.03, 0.0, 0.02]  # original (wrong) tibia point

        attach = {
            "meniscus_medial_COR1": _make_ligament_entry(
                "meniscus_medial_COR1", tibia_xyz_orig, meniscus_xyz
            ),
        }

        results = project_meniscal_attachments_to_tibia(attach, flat_tibia)

        assert "meniscus_medial_COR1" in results
        assert results["meniscus_medial_COR1"]["method"] == "ray"

        # Tibia point should now be directly below meniscus point
        new_tibia = attach["meniscus_medial_COR1"]["points"][0]["xyz_mesh_updated"]
        assert abs(new_tibia[0] - meniscus_xyz[0]) < 1e-3  # same X
        assert abs(new_tibia[2] - meniscus_xyz[2]) < 1e-3  # same Z
        assert abs(new_tibia[1] - 0.0) < 1e-3  # at the plane Y=0

    def test_nearest_fallback(self, flat_tibia):
        """Meniscus point laterally outside tibia → ray misses, nearest fallback."""
        # Point far outside the tibia extent (plane is ±0.1m)
        meniscus_xyz = [0.5, 0.005, 0.5]
        tibia_xyz_orig = [0.5, 0.0, 0.5]

        attach = {
            "meniscus_lateral_COR1": _make_ligament_entry(
                "meniscus_lateral_COR1",
                tibia_xyz_orig,
                meniscus_xyz,
                meniscus_frame="meniscus_lateral_r",
            ),
        }

        results = project_meniscal_attachments_to_tibia(attach, flat_tibia)

        assert results["meniscus_lateral_COR1"]["method"] == "nearest"

    def test_transverse_ligament_excluded(self, flat_tibia):
        """TRANSVLIG ligaments should not be modified."""
        orig_tibia_xyz = np.array([0.01, 0.0, 0.01])
        attach = {
            "meniscus_TRANSVLIG1": {
                "name": "meniscus_TRANSVLIG1",
                "class": "Blankevoort1991Ligament",
                "points": [
                    {
                        "name": "P1",
                        "parent_frame": "meniscus_lateral_r",
                        "include": True,
                        "xyz_mesh_updated": orig_tibia_xyz.copy(),
                    },
                    {
                        "name": "P2",
                        "parent_frame": "meniscus_medial_r",
                        "include": True,
                        "xyz_mesh_updated": np.array([0.02, 0.0, 0.01]),
                    },
                ],
            },
        }

        results = project_meniscal_attachments_to_tibia(attach, flat_tibia)

        assert len(results) == 0
        # Original point unchanged
        np.testing.assert_array_equal(
            attach["meniscus_TRANSVLIG1"]["points"][0]["xyz_mesh_updated"],
            orig_tibia_xyz,
        )

    def test_non_meniscal_ligaments_untouched(self, flat_tibia):
        """Non-meniscal ligaments should not be modified."""
        orig_xyz = np.array([0.01, 0.02, 0.03])
        attach = {
            "MCLd1": {
                "name": "MCLd1",
                "class": "Blankevoort1991Ligament",
                "points": [
                    {
                        "name": "P1",
                        "parent_frame": "femur_distal_r",
                        "include": True,
                        "xyz_mesh_updated": orig_xyz.copy(),
                    },
                    {
                        "name": "P2",
                        "parent_frame": "tibia_proximal_r",
                        "include": True,
                        "xyz_mesh_updated": np.array([0.01, 0.0, 0.03]),
                    },
                ],
            },
            "meniscus_medial_COR1": _make_ligament_entry(
                "meniscus_medial_COR1",
                [0.03, 0.0, 0.02],
                [0.02, 0.005, 0.01],
            ),
        }

        results = project_meniscal_attachments_to_tibia(attach, flat_tibia)

        # Only the meniscal ligament was processed
        assert "MCLd1" not in results
        assert "meniscus_medial_COR1" in results

        # MCL point is unchanged
        np.testing.assert_array_equal(attach["MCLd1"]["points"][0]["xyz_mesh_updated"], orig_xyz)

    def test_multiple_ligaments(self, flat_tibia):
        """All meniscal ligaments in a dict are processed."""
        attach = {}
        lig_names = [
            "meniscus_medial_COR1",
            "meniscus_medial_COR2",
            "meniscus_lateral_AHORN1",
            "meniscus_lateral_PHORN2",
        ]
        for name in lig_names:
            side = "medial" if "medial" in name else "lateral"
            attach[name] = _make_ligament_entry(
                name,
                [0.02, 0.0, 0.01],
                [0.02, 0.005, 0.01],
                meniscus_frame=f"meniscus_{side}_r",
            )

        results = project_meniscal_attachments_to_tibia(attach, flat_tibia)

        assert len(results) == 4
        for name in lig_names:
            assert name in results

    def test_custom_ray_direction_miss(self, flat_tibia):
        """Custom ray direction that misses the flat plane."""
        # Meniscus point offset in X, ray going in +X direction
        # With a flat plane at Y=0, a horizontal ray won't hit it
        meniscus_xyz = [-0.2, 0.005, 0.0]

        attach = {
            "meniscus_medial_COR1": _make_ligament_entry(
                "meniscus_medial_COR1",
                [0.0, 0.0, 0.0],
                meniscus_xyz,
            ),
        }

        results = project_meniscal_attachments_to_tibia(attach, flat_tibia, ray_direction=[1, 0, 0])

        # Horizontal ray on a horizontal plane — should miss
        assert results["meniscus_medial_COR1"]["method"] == "nearest"

    def test_custom_ray_direction_hit(self):
        """Custom ray direction that hits a vertical wall mesh."""
        # Vertical wall at X=0.02, spanning Y and Z
        wall = pv.Plane(
            center=(0.02, 0, 0),
            direction=(1, 0, 0),  # normal in +X
            i_size=0.1,
            j_size=0.1,
            i_resolution=10,
            j_resolution=10,
        )

        meniscus_xyz = [0.01, 0.005, 0.0]  # 10mm to the left of the wall

        attach = {
            "meniscus_medial_COR1": _make_ligament_entry(
                "meniscus_medial_COR1",
                [0.03, 0.005, 0.0],  # original tibia point (will be overwritten)
                meniscus_xyz,
            ),
        }

        results = project_meniscal_attachments_to_tibia(attach, wall, ray_direction=[1, 0, 0])

        assert results["meniscus_medial_COR1"]["method"] == "ray"
        new_tibia = attach["meniscus_medial_COR1"]["points"][0]["xyz_mesh_updated"]
        assert abs(new_tibia[0] - 0.02) < 1e-3  # hit the wall at X=0.02
        assert abs(new_tibia[1] - meniscus_xyz[1]) < 1e-3  # same Y
        assert abs(new_tibia[2] - meniscus_xyz[2]) < 1e-3  # same Z

    def test_distance_reported(self, flat_tibia):
        """Result distance matches expected geometry."""
        height = 0.008  # 8mm above plane
        attach = {
            "meniscus_medial_COR1": _make_ligament_entry(
                "meniscus_medial_COR1",
                [0.02, 0.0, 0.01],
                [0.02, height, 0.01],
            ),
        }

        results = project_meniscal_attachments_to_tibia(attach, flat_tibia)

        assert abs(results["meniscus_medial_COR1"]["distance"] - height) < 1e-3

    def test_zero_ray_direction_raises(self, flat_tibia):
        """Zero-length ray direction should raise ValueError."""
        attach = {
            "meniscus_medial_COR1": _make_ligament_entry(
                "meniscus_medial_COR1", [0, 0, 0], [0, 0.005, 0]
            ),
        }
        with pytest.raises(ValueError, match="non-zero"):
            project_meniscal_attachments_to_tibia(attach, flat_tibia, ray_direction=[0, 0, 0])
