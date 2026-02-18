"""Tests for signed distance functions (SDF) for ellipsoids and cylinders."""

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Ellipsoid SDF tests
# ---------------------------------------------------------------------------


class TestSdEllipsoidImproved:
    """Tests for sd_ellipsoid_improved from wrap_signed_distances.py."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from nsosim.wrap_surface_fitting.wrap_signed_distances import sd_ellipsoid_improved

        self.sd_ellipsoid = sd_ellipsoid_improved

    # -- basic sign tests --------------------------------------------------

    def test_center_is_negative(self):
        """Point at center should have negative SDF (inside)."""
        center = torch.zeros(3, dtype=torch.float64)
        axes = torch.ones(3, dtype=torch.float64)  # unit sphere
        R = torch.eye(3, dtype=torch.float64)
        point = torch.zeros(1, 3, dtype=torch.float64)

        sdf = self.sd_ellipsoid(point, center, axes, R)
        assert sdf.item() < 0, f"Center SDF should be negative, got {sdf.item()}"

    def test_surface_point_near_zero(self):
        """Point on the surface of a unit sphere should have SDF ~ 0."""
        center = torch.zeros(3, dtype=torch.float64)
        axes = torch.ones(3, dtype=torch.float64)
        R = torch.eye(3, dtype=torch.float64)
        point = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)

        sdf = self.sd_ellipsoid(point, center, axes, R)
        assert abs(sdf.item()) < 1e-4, f"Surface SDF should be ~0, got {sdf.item()}"

    def test_outside_is_positive(self):
        """Point well outside should have positive SDF."""
        center = torch.zeros(3, dtype=torch.float64)
        axes = torch.ones(3, dtype=torch.float64)
        R = torch.eye(3, dtype=torch.float64)
        point = torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float64)

        sdf = self.sd_ellipsoid(point, center, axes, R)
        assert sdf.item() > 0, f"Outside SDF should be positive, got {sdf.item()}"

    # -- symmetry tests ----------------------------------------------------

    def test_symmetry_unit_sphere(self):
        """SDF should be the same for symmetric points on a sphere."""
        center = torch.zeros(3, dtype=torch.float64)
        axes = torch.ones(3, dtype=torch.float64)
        R = torch.eye(3, dtype=torch.float64)

        points = torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [-2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, -2.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, -2.0],
            ],
            dtype=torch.float64,
        )

        sdf = self.sd_ellipsoid(points, center, axes, R)
        assert torch.allclose(
            sdf, sdf[0].expand_as(sdf), atol=1e-6
        ), "SDF should be equal for symmetric points on a unit sphere"

    # -- known analytic value for unit sphere --------------------------------

    def test_unit_sphere_distance_approximation(self):
        """For a unit sphere, SDF at distance d from center along an axis should be ~ d-1.

        Note: sd_ellipsoid_improved uses a gradient-based approximation (|F(p)|/||grad F(p)||)
        which is accurate near the surface but underestimates for points far away.
        At d=3 the exact distance is 2.0 but the approximation returns ~1.33.
        We use loose tolerances for far points and tight for near-surface points.
        """
        center = torch.zeros(3, dtype=torch.float64)
        axes = torch.ones(3, dtype=torch.float64)
        R = torch.eye(3, dtype=torch.float64)

        # Near-surface points: approximation is accurate
        for d in [0.5, 1.5]:
            point = torch.tensor([[d, 0.0, 0.0]], dtype=torch.float64)
            sdf = self.sd_ellipsoid(point, center, axes, R)
            expected = d - 1.0
            assert (
                abs(sdf.item() - expected) < 0.3
            ), f"At d={d}, SDF={sdf.item():.4f}, expected ~{expected}"

        # Far points: sign is correct, but magnitude underestimates
        for d in [2.0, 3.0]:
            point = torch.tensor([[d, 0.0, 0.0]], dtype=torch.float64)
            sdf = self.sd_ellipsoid(point, center, axes, R)
            assert sdf.item() > 0, f"At d={d}, SDF should be positive (outside)"

    # -- non-axis-aligned ellipsoid ----------------------------------------

    def test_ellipsoid_different_axes(self):
        """An ellipsoid with different semi-axes should still give correct inside/outside."""
        center = torch.zeros(3, dtype=torch.float64)
        axes = torch.tensor([2.0, 1.0, 0.5], dtype=torch.float64)
        R = torch.eye(3, dtype=torch.float64)

        # Inside along longest axis
        inside = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        assert self.sd_ellipsoid(inside, center, axes, R).item() < 0

        # Outside along shortest axis
        outside = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        assert self.sd_ellipsoid(outside, center, axes, R).item() > 0

    # -- translated / rotated ellipsoid ------------------------------------

    def test_translated_center(self):
        """SDF should respect translation of center."""
        center = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        axes = torch.ones(3, dtype=torch.float64)
        R = torch.eye(3, dtype=torch.float64)

        # Point at center should be inside
        at_center = center.unsqueeze(0)
        assert self.sd_ellipsoid(at_center, center, axes, R).item() < 0

        # Point far from center should be outside
        far = torch.tensor([[100.0, 100.0, 100.0]], dtype=torch.float64)
        assert self.sd_ellipsoid(far, center, axes, R).item() > 0

    def test_batch_of_points(self):
        """Should handle batches of points correctly."""
        center = torch.zeros(3, dtype=torch.float64)
        axes = torch.ones(3, dtype=torch.float64)
        R = torch.eye(3, dtype=torch.float64)

        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # inside
                [0.5, 0.0, 0.0],  # inside
                [2.0, 0.0, 0.0],  # outside
                [5.0, 5.0, 5.0],  # outside
            ],
            dtype=torch.float64,
        )

        sdf = self.sd_ellipsoid(points, center, axes, R)
        assert sdf.shape == (4,)
        assert sdf[0].item() < 0  # center
        assert sdf[1].item() < 0  # inside
        assert sdf[2].item() > 0  # outside
        assert sdf[3].item() > 0  # far outside


# ---------------------------------------------------------------------------
# Cylinder SDF tests
# ---------------------------------------------------------------------------


class TestSdCylinderWithAxis:
    """Tests for sd_cylinder_with_axis from fitting.py."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from nsosim.wrap_surface_fitting.fitting import sd_cylinder_with_axis

        self.sd_cylinder = sd_cylinder_with_axis

    def test_center_is_negative(self):
        """Point at cylinder center should have negative SDF."""
        center = torch.zeros(3, dtype=torch.float64)
        radius = torch.tensor(1.0, dtype=torch.float64)
        half_length = torch.tensor(2.0, dtype=torch.float64)
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        point = torch.zeros(1, 3, dtype=torch.float64)

        sdf = self.sd_cylinder(point, center, radius, half_length, axis)
        assert sdf.item() < 0, f"Center SDF should be negative, got {sdf.item()}"

    def test_surface_radial_near_zero(self):
        """Point on the radial surface should have SDF ~ 0."""
        center = torch.zeros(3, dtype=torch.float64)
        radius = torch.tensor(1.0, dtype=torch.float64)
        half_length = torch.tensor(2.0, dtype=torch.float64)
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        # Point on the radial surface, at z=0
        point = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)

        sdf = self.sd_cylinder(point, center, radius, half_length, axis)
        assert abs(sdf.item()) < 1e-4, f"Surface SDF should be ~0, got {sdf.item()}"

    def test_outside_radial_positive(self):
        """Point well outside radially should have positive SDF."""
        center = torch.zeros(3, dtype=torch.float64)
        radius = torch.tensor(1.0, dtype=torch.float64)
        half_length = torch.tensor(2.0, dtype=torch.float64)
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        point = torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float64)

        sdf = self.sd_cylinder(point, center, radius, half_length, axis)
        assert sdf.item() > 0

    def test_outside_axial_positive(self):
        """Point beyond the cap should have positive SDF."""
        center = torch.zeros(3, dtype=torch.float64)
        radius = torch.tensor(1.0, dtype=torch.float64)
        half_length = torch.tensor(2.0, dtype=torch.float64)
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        # Beyond the cap at z = half_length
        point = torch.tensor([[0.0, 0.0, 5.0]], dtype=torch.float64)

        sdf = self.sd_cylinder(point, center, radius, half_length, axis)
        assert sdf.item() > 0

    def test_symmetry_radial(self):
        """Symmetric radial points should have equal SDF."""
        center = torch.zeros(3, dtype=torch.float64)
        radius = torch.tensor(1.0, dtype=torch.float64)
        half_length = torch.tensor(5.0, dtype=torch.float64)
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

        points = torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [-2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, -2.0, 0.0],
            ],
            dtype=torch.float64,
        )

        sdf = self.sd_cylinder(points, center, radius, half_length, axis)
        assert torch.allclose(sdf, sdf[0].expand_as(sdf), atol=1e-6)

    def test_translated_cylinder(self):
        """SDF should respect translation."""
        center = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        radius = torch.tensor(1.0, dtype=torch.float64)
        half_length = torch.tensor(2.0, dtype=torch.float64)
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

        at_center = center.unsqueeze(0)
        sdf = self.sd_cylinder(at_center, center, radius, half_length, axis)
        assert sdf.item() < 0

    def test_known_radial_distance(self):
        """For an axis-aligned cylinder, radial SDF at (r+d, 0, 0) should be ~ d."""
        center = torch.zeros(3, dtype=torch.float64)
        radius = torch.tensor(1.0, dtype=torch.float64)
        half_length = torch.tensor(10.0, dtype=torch.float64)
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

        d = 3.0
        point = torch.tensor([[radius.item() + d, 0.0, 0.0]], dtype=torch.float64)
        sdf = self.sd_cylinder(point, center, radius, half_length, axis)
        assert abs(sdf.item() - d) < 0.1, f"Expected ~{d}, got {sdf.item()}"
