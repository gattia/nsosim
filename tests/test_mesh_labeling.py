"""Tests for mesh labeling functions.

Phase 2.3 of the repo-hardening plan.

Covers:
- classify_points(): SDF -> binary inside/outside
- classify_near_surface(): SDF -> near/far binary labels
- Synthetic cylinder labeling pipeline (SDF + classify)
"""

import numpy as np
import pytest
import pyvista as pv
from pymskt.mesh import Mesh

from nsosim.wrap_surface_fitting.utils import classify_near_surface, classify_points

# ---------------------------------------------------------------------------
# classify_points
# ---------------------------------------------------------------------------


class TestClassifyPoints:
    """classify_points(sdf, threshold) -> binary labels (1=inside, 0=outside)."""

    def test_basic_classification(self):
        """Negative SDF -> inside (1), non-negative SDF -> outside (0)."""
        sdf = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = classify_points(sdf, threshold=0.0)
        expected = np.array([1, 1, 0, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_custom_threshold(self):
        """With threshold=0.5, SDF < 0.5 -> inside."""
        sdf = np.array([-1.0, 0.0, 0.3, 0.5, 1.0])
        result = classify_points(sdf, threshold=0.5)
        # sdf < 0.5: -1, 0, 0.3 -> 1;  0.5, 1.0 -> 0
        expected = np.array([1, 1, 1, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_all_inside(self):
        """All negative SDF -> all inside."""
        sdf = np.array([-5.0, -3.0, -0.001])
        result = classify_points(sdf, threshold=0.0)
        np.testing.assert_array_equal(result, [1, 1, 1])

    def test_all_outside(self):
        """All positive SDF -> all outside."""
        sdf = np.array([0.001, 1.0, 10.0])
        result = classify_points(sdf, threshold=0.0)
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_empty_array(self):
        """Empty input should return empty output."""
        sdf = np.array([])
        result = classify_points(sdf, threshold=0.0)
        assert len(result) == 0

    def test_output_dtype_is_int(self):
        """Result should be integer type."""
        sdf = np.array([-1.0, 1.0])
        result = classify_points(sdf, threshold=0.0)
        assert result.dtype == int


# ---------------------------------------------------------------------------
# classify_near_surface
# ---------------------------------------------------------------------------


class TestClassifyNearSurface:
    """classify_near_surface(sdf, threshold) -> binary near/far labels."""

    def test_basic_near_surface(self):
        """|SDF| <= threshold -> near surface (1)."""
        sdf = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        result = classify_near_surface(sdf, threshold=3.0)
        # |sdf|: 5, 1, 0, 1, 5
        # near (<= 3): 0, 1, 1, 1, 0
        expected = np.array([0, 1, 1, 1, 0])
        np.testing.assert_array_equal(result, expected)

    def test_boundary_at_threshold(self):
        """Point exactly at threshold should be classified as near-surface."""
        sdf = np.array([3.0, -3.0, 3.001])
        result = classify_near_surface(sdf, threshold=3.0)
        expected = np.array([1, 1, 0])
        np.testing.assert_array_equal(result, expected)

    def test_all_near(self):
        """All points within threshold."""
        sdf = np.array([-0.5, 0.0, 0.5])
        result = classify_near_surface(sdf, threshold=1.0)
        np.testing.assert_array_equal(result, [1, 1, 1])

    def test_all_far(self):
        """All points outside threshold."""
        sdf = np.array([-10.0, 10.0])
        result = classify_near_surface(sdf, threshold=1.0)
        np.testing.assert_array_equal(result, [0, 0])

    def test_zero_threshold(self):
        """With threshold=0, only points exactly on surface are near."""
        sdf = np.array([0.0, 0.001, -0.001])
        result = classify_near_surface(sdf, threshold=0.0)
        expected = np.array([1, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_empty_array(self):
        """Empty input should return empty output."""
        sdf = np.array([])
        result = classify_near_surface(sdf, threshold=1.0)
        assert len(result) == 0

    def test_output_dtype_is_int(self):
        """Result should be integer type."""
        sdf = np.array([-1.0, 1.0])
        result = classify_near_surface(sdf, threshold=3.0)
        assert result.dtype == int


# ---------------------------------------------------------------------------
# Synthetic cylinder labeling pipeline
# ---------------------------------------------------------------------------


class TestSyntheticCylinderLabeling:
    """End-to-end: create a cylinder, compute SDF, classify points.

    This tests the full labeling pipeline that would normally be driven by
    prepare_fitting_data() but uses synthetic geometry instead of XML files.
    """

    @staticmethod
    def _make_cylinder_mesh(center, radius, height, direction=(0, 0, 1), resolution=50):
        """Create a PyVista cylinder mesh and wrap as pymskt Mesh."""
        cyl = pv.Cylinder(
            center=center, direction=direction, radius=radius, height=height, resolution=resolution
        )
        return Mesh(cyl.triangulate())

    def test_inside_outside_sdf(self):
        """PCU SDF: inside -> SDF < 0, on surface -> ~0, outside -> SDF > 0.

        Note: pymskt's PCU method returns wrong results for single-point queries
        (a known pymskt/PCU bug). Always pass >= 2 points.
        """
        center = np.array([0.0, 0.0, 0.0])
        radius = 5.0
        height = 20.0

        cyl_mesh = self._make_cylinder_mesh(center, radius, height)

        test_points = np.array(
            [
                [0.0, 0.0, 0.0],  # center of cylinder -> inside
                [radius, 0.0, 0.0],  # on radial surface -> ~0
                [radius + 1, 0.0, 0.0],  # just outside
                [20.0, 0.0, 0.0],  # far from axis -> outside
            ]
        )

        sdf = cyl_mesh.get_sdf_pts(test_points)
        binary = classify_points(sdf, threshold=0.0)

        # Sign and approximate value checks
        assert sdf[0] < 0, f"Center should have SDF<0, got {sdf[0]}"
        assert abs(sdf[1]) < 0.5, f"Surface point SDF should be ~0, got {sdf[1]}"
        assert sdf[2] > 0, f"Outside point should have SDF>0, got {sdf[2]}"
        assert sdf[3] > 0, f"Far point should have SDF>0, got {sdf[3]}"

        # Binary classification
        assert binary[0] == 1, "Center should be classified as inside"
        assert binary[3] == 0, "Far point should be classified as outside"

    def test_near_surface_sdf(self):
        """classify_near_surface on PCU SDF: surface points are near, far points are far."""
        center = np.array([0.0, 0.0, 0.0])
        radius = 5.0
        height = 20.0

        cyl_mesh = self._make_cylinder_mesh(center, radius, height)

        test_points = np.array(
            [
                [radius, 0.0, 0.0],  # on radial surface
                [20.0, 0.0, 0.0],  # far outside
            ]
        )

        sdf = cyl_mesh.get_sdf_pts(test_points)
        near = classify_near_surface(sdf, threshold=1.0)

        assert near[0] == 1, f"Surface point should be near (|SDF|={abs(sdf[0]):.2f} <= 1.0)"
        assert near[1] == 0, f"Far point should be far (|SDF|={abs(sdf[1]):.2f} > 1.0)"

    def test_classify_is_consistent_with_sdf_sign(self):
        """classify_points and SDF sign should agree on inside/outside."""
        center = np.array([0.0, 0.0, 0.0])
        radius = 3.0
        height = 10.0

        cyl_mesh = self._make_cylinder_mesh(center, radius, height)

        # Grid of test points
        rng = np.random.default_rng(42)
        test_points = rng.uniform(-8, 8, (200, 3))

        sdf = cyl_mesh.get_sdf_pts(test_points)
        binary = classify_points(sdf, threshold=0.0)

        # Every point with sdf < 0 should have binary == 1
        inside_mask = sdf < 0
        np.testing.assert_array_equal(binary[inside_mask], 1)

        # Every point with sdf >= 0 should have binary == 0
        outside_mask = sdf >= 0
        np.testing.assert_array_equal(binary[outside_mask], 0)
