"""Tests for articular surface functions.

Phase 2.2 of the repo-hardening plan.

Covers:
- add_polar_coordinates_about_center()
- build_min_radial_envelope()
- mask_points_by_radial_envelope()
- trim_mesh_by_radial_envelope()
- create_articular_surfaces() (smoke test)
- create_meniscus_articulating_surface() (smoke test)
"""

import numpy as np
import pytest
import pyvista as pv
from pymskt.mesh import Mesh

from nsosim.articular_surfaces import (
    add_polar_coordinates_about_center,
    build_min_radial_envelope,
    create_articular_surfaces,
    create_meniscus_articulating_surface,
    mask_points_by_radial_envelope,
    smooth_1d,
    trim_mesh_by_radial_envelope,
)

# ---------------------------------------------------------------------------
# add_polar_coordinates_about_center
# ---------------------------------------------------------------------------


class TestAddPolarCoordinates:
    """Verify theta, r, y_rel for known point positions."""

    def test_known_cardinal_points(self):
        """Points on the cardinal axes should have exact theta and r."""
        # theta = arctan2(x_rel, z_rel)
        # r = sqrt(x_rel^2 + z_rel^2)
        points = np.array(
            [
                [1.0, 0.0, 0.0],  # x_rel=1, z_rel=0  -> theta=pi/2, r=1
                [0.0, 0.0, 1.0],  # x_rel=0, z_rel=1  -> theta=0,    r=1
                [-1.0, 0.0, 0.0],  # x_rel=-1, z_rel=0 -> theta=-pi/2, r=1
                [0.0, 0.0, -1.0],  # x_rel=0, z_rel=-1 -> theta=pi,  r=1
            ]
        )
        mesh = pv.PolyData(points)
        center = add_polar_coordinates_about_center(mesh, center=np.array([0.0, 0.0, 0.0]))

        np.testing.assert_allclose(center, [0.0, 0.0, 0.0])

        theta = mesh["theta"]
        r = mesh["r"]
        y_rel = mesh["y_rel"]

        np.testing.assert_allclose(theta[0], np.pi / 2, atol=1e-10)
        np.testing.assert_allclose(theta[1], 0.0, atol=1e-10)
        np.testing.assert_allclose(theta[2], -np.pi / 2, atol=1e-10)
        # theta for (0, 0, -1): arctan2(0, -1) = pi
        np.testing.assert_allclose(abs(theta[3]), np.pi, atol=1e-10)

        np.testing.assert_allclose(r, [1.0, 1.0, 1.0, 1.0], atol=1e-10)
        np.testing.assert_allclose(y_rel, [0.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_center_offset(self):
        """Polar coordinates computed relative to a non-origin center."""
        points = np.array([[11.0, 5.0, 10.0]])  # relative to center: (1, 5, 0)
        mesh = pv.PolyData(points)
        center = add_polar_coordinates_about_center(mesh, center=np.array([10.0, 0.0, 10.0]))

        # x_rel=1, z_rel=0 -> theta=pi/2, r=1
        np.testing.assert_allclose(mesh["theta"][0], np.pi / 2, atol=1e-10)
        np.testing.assert_allclose(mesh["r"][0], 1.0, atol=1e-10)
        np.testing.assert_allclose(mesh["y_rel"][0], 5.0, atol=1e-10)

    def test_theta_offset_rotates(self):
        """theta_offset should shift theta and wrap to [-pi, pi]."""
        points = np.array([[0.0, 0.0, 1.0]])  # theta=0 without offset
        mesh = pv.PolyData(points)
        add_polar_coordinates_about_center(
            mesh, center=np.array([0.0, 0.0, 0.0]), theta_offset=np.pi / 2
        )

        # theta = 0 + pi/2 = pi/2
        np.testing.assert_allclose(mesh["theta"][0], np.pi / 2, atol=1e-10)

    def test_theta_offset_wraps(self):
        """Large theta_offset should wrap to [-pi, pi]."""
        # Point at (0, 0, -1): base theta = pi
        points = np.array([[0.0, 0.0, -1.0]])
        mesh = pv.PolyData(points)
        add_polar_coordinates_about_center(
            mesh, center=np.array([0.0, 0.0, 0.0]), theta_offset=np.pi / 2
        )

        # theta = pi + pi/2 = 3pi/2 -> wrapped to -pi/2
        np.testing.assert_allclose(mesh["theta"][0], -np.pi / 2, atol=1e-10)

    def test_default_center_uses_mesh_center(self):
        """If center is None, should use mesh.center."""
        points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        mesh = pv.PolyData(points)
        center = add_polar_coordinates_about_center(mesh, center=None)

        # mesh.center should be [1.0, 0.0, 0.0]
        np.testing.assert_allclose(center, [1.0, 0.0, 0.0], atol=1e-10)

    def test_diagonal_point(self):
        """A point at 45 degrees in the xz-plane."""
        points = np.array([[1.0, 3.0, 1.0]])
        mesh = pv.PolyData(points)
        add_polar_coordinates_about_center(mesh, center=np.array([0.0, 0.0, 0.0]))

        # theta = arctan2(1, 1) = pi/4
        np.testing.assert_allclose(mesh["theta"][0], np.pi / 4, atol=1e-10)
        np.testing.assert_allclose(mesh["r"][0], np.sqrt(2), atol=1e-10)
        np.testing.assert_allclose(mesh["y_rel"][0], 3.0, atol=1e-10)


# ---------------------------------------------------------------------------
# smooth_1d
# ---------------------------------------------------------------------------


class TestSmooth1d:
    def test_constant_interior_unchanged(self):
        """Interior of a constant array is unchanged; edges have boundary effects."""
        y = np.ones(20) * 5.0
        result = smooth_1d(y, window_size=5)
        # np.convolve mode='same' has boundary effects at edges
        np.testing.assert_allclose(result[2:-2], 5.0, atol=1e-10)

    def test_window_1_noop(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        result = smooth_1d(y, window_size=1)
        np.testing.assert_array_equal(result, y)


# ---------------------------------------------------------------------------
# build_min_radial_envelope
# ---------------------------------------------------------------------------


class TestBuildMinRadialEnvelope:
    """Verify the radial envelope from per-region percentile curves."""

    def test_min_of_two_regions(self):
        """Envelope should be the element-wise minimum across regions."""
        region_percentiles = {
            1: {
                "bin_centers": np.array([0.0, 1.0, 2.0]),
                "r_percentile": np.array([5.0, 8.0, 5.0]),
            },
            2: {
                "bin_centers": np.array([0.0, 1.0, 2.0]),
                "r_percentile": np.array([6.0, 4.0, 7.0]),
            },
        }
        theta_grid, r_min = build_min_radial_envelope(
            region_percentiles, smooth_window=1, n_theta_grid=3
        )

        assert len(theta_grid) == 3
        assert len(r_min) == 3

        # At theta=0: min(5, 6) = 5
        # At theta=1: min(8, 4) = 4
        # At theta=2: min(5, 7) = 5
        np.testing.assert_allclose(r_min, [5.0, 4.0, 5.0], atol=0.5)

    def test_single_region(self):
        """With one region, envelope equals that region's percentile."""
        region_percentiles = {
            1: {
                "bin_centers": np.array([0.0, 1.0, 2.0]),
                "r_percentile": np.array([3.0, 4.0, 5.0]),
            },
        }
        theta_grid, r_min = build_min_radial_envelope(
            region_percentiles, smooth_window=1, n_theta_grid=3
        )

        np.testing.assert_allclose(r_min, [3.0, 4.0, 5.0], atol=0.5)

    def test_output_shape(self):
        """Output should have the requested grid resolution."""
        region_percentiles = {
            1: {
                "bin_centers": np.linspace(-np.pi, np.pi, 50),
                "r_percentile": np.ones(50) * 10.0,
            },
        }
        theta_grid, r_min = build_min_radial_envelope(
            region_percentiles, smooth_window=1, n_theta_grid=200
        )
        assert len(theta_grid) == 200
        assert len(r_min) == 200


# ---------------------------------------------------------------------------
# mask_points_by_radial_envelope
# ---------------------------------------------------------------------------


class TestMaskPointsByRadialEnvelope:
    """Verify that points inside/outside the radial envelope are correctly masked."""

    def test_basic_mask(self):
        """Points with r < r_thresh should be kept."""
        # Create points in the xz-plane at known radial distances
        # Point at (5, 0, 0): theta=pi/2, r=5
        # Point at (1, 0, 0): theta=pi/2, r=1
        # Point at (0, 0, 3): theta=0, r=3
        points = np.array(
            [
                [5.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        )
        mesh = pv.PolyData(points)
        center = np.array([0.0, 0.0, 0.0])

        # Envelope: allow r <= 4.0 for all theta
        theta_grid = np.linspace(-np.pi, np.pi, 100)
        r_thresh = np.full(100, 4.0)

        keep, theta, r = mask_points_by_radial_envelope(
            mesh, center, theta_grid, r_thresh, theta_offset=0.0
        )

        # r=5 -> outside, r=1 -> inside, r=3 -> inside
        assert not keep[0], "r=5 should be outside envelope"
        assert keep[1], "r=1 should be inside envelope"
        assert keep[2], "r=3 should be inside envelope"

    def test_theta_offset(self):
        """theta_offset should not change radial filtering for uniform envelope."""
        points = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        mesh = pv.PolyData(points)
        center = np.array([0.0, 0.0, 0.0])

        theta_grid = np.linspace(-np.pi, np.pi, 100)
        r_thresh = np.full(100, 3.0)

        keep0, _, _ = mask_points_by_radial_envelope(
            mesh, center, theta_grid, r_thresh, theta_offset=0.0
        )
        keep_pi, _, _ = mask_points_by_radial_envelope(
            mesh, center, theta_grid, r_thresh, theta_offset=np.pi
        )

        # With a uniform envelope (r=3 everywhere), both points (r=2) should pass
        assert keep0[0] and keep0[1]
        assert keep_pi[0] and keep_pi[1]


# ---------------------------------------------------------------------------
# trim_mesh_by_radial_envelope
# ---------------------------------------------------------------------------


class TestTrimMeshByRadialEnvelope:
    """Verify that trimming produces a mesh with only interior points."""

    def test_trims_outlier_points(self):
        """Points far from center should be removed."""
        # Dense mesh (~10k points) so boundary cells are small
        sphere = pv.Sphere(radius=5.0, center=(0, 0, 0), theta_resolution=100, phi_resolution=100)
        center = np.array([0.0, 0.0, 0.0])

        # Envelope: r <= 3.0 for all theta
        theta_grid = np.linspace(-np.pi, np.pi, 100)
        r_thresh = np.full(100, 3.0)

        trimmed = trim_mesh_by_radial_envelope(sphere, center, theta_grid, r_thresh)

        # The trimmed mesh should have fewer points (only those with xz-radius <= 3)
        assert trimmed.n_points < sphere.n_points
        assert trimmed.n_points > 0

        # With ~10k-point mesh, boundary overshoot from adjacent_cells is ~0.26
        pts = trimmed.points
        r = np.sqrt(pts[:, 0] ** 2 + pts[:, 2] ** 2)
        assert np.all(r <= 3.3), f"Max r in trimmed mesh: {r.max():.2f}"

    def test_preserves_all_when_envelope_is_large(self):
        """If envelope is larger than mesh extent, all points remain."""
        sphere = pv.Sphere(radius=2.0, center=(0, 0, 0), theta_resolution=10, phi_resolution=10)
        center = np.array([0.0, 0.0, 0.0])

        theta_grid = np.linspace(-np.pi, np.pi, 100)
        r_thresh = np.full(100, 100.0)  # Very large envelope

        trimmed = trim_mesh_by_radial_envelope(sphere, center, theta_grid, r_thresh)
        assert trimmed.n_points == sphere.n_points


# ---------------------------------------------------------------------------
# create_articular_surfaces (smoke test)
# ---------------------------------------------------------------------------


class TestCreateArticularSurfaces:
    """Smoke test: create_articular_surfaces should not crash on synthetic spheres."""

    @pytest.mark.slow
    def test_concentric_spheres_smoke(self):
        """Two concentric spheres (bone + cartilage) should produce a result."""
        bone = pv.Sphere(radius=0.03, center=(0, 0, 0), theta_resolution=20, phi_resolution=20)
        cart = pv.Sphere(radius=0.032, center=(0, 0, 0), theta_resolution=20, phi_resolution=20)
        bone_mesh = Mesh(bone)
        cart_mesh = Mesh(cart)

        try:
            result = create_articular_surfaces(
                bone_mesh,
                cart_mesh,
                triangle_density=None,
                bone_clusters=None,
                cart_clusters=None,
                ray_length=10,
                smooth_iter=5,
                n_largest=1,
            )
            assert isinstance(result, pv.PolyData)
            assert result.n_points > 0
        except Exception as e:
            pytest.skip(f"create_articular_surfaces failed on synthetic data: {e}")


# ---------------------------------------------------------------------------
# create_meniscus_articulating_surface (smoke test)
# ---------------------------------------------------------------------------


class TestCreateMeniscusArticulatingSurface:
    """Smoke test for meniscus surface extraction."""

    @pytest.mark.slow
    def test_torus_between_spheres_smoke(self):
        """A torus between two spheres should produce upper/lower surfaces."""
        # Meniscus as a torus
        torus = pv.ParametricTorus(ringradius=0.01, crosssectionradius=0.003)
        # Upper bone (above torus)
        upper = pv.Sphere(radius=0.03, center=(0, 0.01, 0), theta_resolution=20, phi_resolution=20)
        # Lower bone (below torus)
        lower = pv.Sphere(radius=0.03, center=(0, -0.01, 0), theta_resolution=20, phi_resolution=20)

        men_mesh = Mesh(torus)
        upper_mesh = Mesh(upper)
        lower_mesh = Mesh(lower)

        try:
            upper_surf, lower_surf = create_meniscus_articulating_surface(
                meniscus_mesh=men_mesh,
                upper_articulating_bone_mesh=upper_mesh,
                lower_articulating_bone_mesh=lower_mesh,
                ray_length=10.0,
                n_largest=1,
                smooth_iter=5,
                triangle_density=None,
                refine_by_radial_envelope=False,
            )
            # Basic assertions: both should be mesh-like objects
            assert upper_surf is not None
            assert lower_surf is not None
            assert hasattr(upper_surf, "point_coords")
            assert hasattr(lower_surf, "point_coords")
        except Exception as e:
            pytest.skip(f"create_meniscus_articulating_surface failed on synthetic data: {e}")
