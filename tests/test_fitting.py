"""Tests for wrap surface fitters (CylinderFitter, EllipsoidFitter, PatellaFitter).

Phase 2.1 of the repo-hardening plan.

These tests fit synthetic point clouds with known geometry and verify that the
recovered parameters are within tolerance of the ground truth.
"""

import numpy as np
import pytest
import pyvista as pv
import torch
from pymskt.mesh import Mesh

from nsosim.wrap_surface_fitting.fitting import (
    CylinderFitter,
    EllipsoidFitter,
    construct_cylinder_basis,
    sd_cylinder_with_axis,
)
from nsosim.wrap_surface_fitting.main import wrap_surface
from nsosim.wrap_surface_fitting.patella import (
    PatellaFitter,
    compute_ellipsoid_parameters_from_labeled_mesh,
    label_patella_within_wrap_extents,
)
from nsosim.wrap_surface_fitting.wrap_signed_distances import sd_ellipsoid_improved

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _cylinder_surface_points(center, radius, half_length, n=800, noise=0.0003, seed=42):
    """Points on a Z-aligned cylinder surface with optional Gaussian noise."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-half_length, half_length, n)
    pts = np.column_stack(
        [
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            center[2] + z,
        ]
    )
    if noise > 0:
        pts += rng.normal(0, noise, pts.shape)
    return pts


def _ellipsoid_labeled_points(center, axes, n_per_side=600, seed=42):
    """Near-surface inside/outside labeled points for an axis-aligned ellipsoid.

    Generates points in a thin shell around the ellipsoid surface (r_scale 0.85-0.99
    inside, 1.01-1.15 outside), which is representative of real bone mesh data where
    points cluster near the wrap surface boundary.

    Labels: 1 = inside, 0 = outside (based on exact ellipsoid equation).
    Also returns approximate SDF from sd_ellipsoid_improved for distance-based fitting.
    """
    rng = np.random.default_rng(seed)

    u = rng.uniform(0, 2 * np.pi, n_per_side)
    v = rng.uniform(0, np.pi, n_per_side)

    # Thin shell just inside the surface
    r_inside = rng.uniform(0.85, 0.99, n_per_side)
    pts_in = (
        np.column_stack(
            [
                axes[0] * r_inside * np.sin(v) * np.cos(u),
                axes[1] * r_inside * np.sin(v) * np.sin(u),
                axes[2] * r_inside * np.cos(v),
            ]
        )
        + center
    )

    # Thin shell just outside the surface
    r_outside = rng.uniform(1.01, 1.15, n_per_side)
    pts_out = (
        np.column_stack(
            [
                axes[0] * r_outside * np.sin(v) * np.cos(u),
                axes[1] * r_outside * np.sin(v) * np.sin(u),
                axes[2] * r_outside * np.cos(v),
            ]
        )
        + center
    )

    all_pts = np.vstack([pts_in, pts_out])
    labels = np.concatenate([np.ones(n_per_side), np.zeros(n_per_side)])

    # Compute approximate SDF for distance-based loss
    pts_t = torch.tensor(all_pts, dtype=torch.float32)
    center_t = torch.tensor(center, dtype=torch.float32)
    axes_t = torch.tensor(axes, dtype=torch.float32)
    R_t = torch.eye(3, dtype=torch.float32)
    with torch.no_grad():
        sdf = sd_ellipsoid_improved(pts_t, center_t, axes_t, R_t).numpy()

    return all_pts, labels, sdf


# ---------------------------------------------------------------------------
# CylinderFitter
# ---------------------------------------------------------------------------


class TestCylinderFitter:
    """Fit a Z-aligned cylinder and verify recovered parameters."""

    TRUE_CENTER = np.array([0.05, 0.06, -0.03])
    TRUE_RADIUS = 0.02
    TRUE_HALF_LENGTH = 0.04
    TRUE_AXIS = np.array([0.0, 0.0, 1.0])

    @pytest.fixture(scope="class")
    def fitted(self):
        """Fit once, share across all tests in the class."""
        pts = _cylinder_surface_points(
            self.TRUE_CENTER, self.TRUE_RADIUS, self.TRUE_HALF_LENGTH, n=800
        )
        labels = np.ones(len(pts))

        fitter = CylinderFitter(
            epochs=0,
            use_lbfgs=True,
            lbfgs_epochs=20,
            alpha=1.0,
            beta=0.0,
            gamma=0.0,
            initialization="pca",
            center_transform="linear",
            margin_decay_type=None,
        )
        result = fitter.fit(points=pts, labels=labels, margin=1e-6, plot=False)
        return fitter, result

    def test_recovers_center(self, fitted):
        fitter, _ = fitted
        wp = fitter.wrap_params
        np.testing.assert_allclose(
            wp.translation, self.TRUE_CENTER, atol=0.002, err_msg="Center off by >2 mm"
        )

    def test_recovers_radius(self, fitted):
        fitter, _ = fitted
        wp = fitter.wrap_params
        rel_err = abs(float(wp.radius) - self.TRUE_RADIUS) / self.TRUE_RADIUS
        assert rel_err < 0.05, f"Radius relative error {rel_err:.2%} exceeds 5%"

    def test_recovers_axis(self, fitted):
        _, result = fitted
        R = result[2].detach().cpu().numpy()
        recovered_axis = R[:, 2]
        cos_sim = abs(np.dot(recovered_axis, self.TRUE_AXIS))
        assert cos_sim > 0.999, f"Axis cosine similarity {cos_sim:.6f} < 0.999"

    def test_wrap_params_structure(self, fitted):
        fitter, _ = fitted
        wp = fitter.wrap_params
        assert isinstance(wp, wrap_surface)
        assert wp.type_ == "WrapCylinder"
        assert wp.radius is not None
        assert wp.length is not None
        assert wp.translation is not None
        assert wp.xyz_body_rotation is not None
        assert len(wp.translation) == 3
        assert len(wp.xyz_body_rotation) == 3

    def test_pca_init_reasonable(self):
        """PCA initialization alone (before fitting) should be in the right ballpark."""
        pts = _cylinder_surface_points(
            self.TRUE_CENTER, self.TRUE_RADIUS, self.TRUE_HALF_LENGTH, n=500
        )
        labels = np.ones(len(pts))

        fitter = CylinderFitter(
            epochs=0,
            use_lbfgs=True,
            lbfgs_epochs=1,
            initialization="pca",
            center_transform="linear",
        )
        result = fitter.fit(points=pts, labels=labels, margin=1e-6, plot=False)
        center_init = result[0].detach().cpu().numpy()

        # PCA center should be within 10 mm of true center
        np.testing.assert_allclose(center_init, self.TRUE_CENTER, atol=0.01)


# ---------------------------------------------------------------------------
# EllipsoidFitter
# ---------------------------------------------------------------------------


class TestEllipsoidFitter:
    """Fit an axis-aligned ellipsoid and verify recovered parameters."""

    TRUE_CENTER = np.array([0.05, 0.06, -0.03])
    TRUE_AXES = np.array([0.025, 0.035, 0.02])
    TRUE_ROTATION = np.eye(3)

    @pytest.fixture(scope="class")
    def fitted(self):
        pts, labels, sdf = _ellipsoid_labeled_points(self.TRUE_CENTER, self.TRUE_AXES, seed=42)
        fitter = EllipsoidFitter(
            lr=5e-3,
            epochs=500,
            use_lbfgs=True,
            lbfgs_epochs=30,
            alpha=1.0,
            beta=0.5,
            gamma=0.0,
            initialization="pca",
            center_transform="linear",
            margin_decay_type="linear",
        )
        result = fitter.fit(points=pts, labels=labels, sdf=sdf, margin=0.005, plot=False)
        return fitter, result

    def test_recovers_center(self, fitted):
        fitter, _ = fitted
        wp = fitter.wrap_params
        np.testing.assert_allclose(
            wp.translation, self.TRUE_CENTER, atol=0.001, err_msg="Center off by >1 mm"
        )

    def test_recovers_axes(self, fitted):
        fitter, _ = fitted
        wp = fitter.wrap_params
        recovered = np.sort(wp.dimensions)
        expected = np.sort(self.TRUE_AXES)
        for r, e in zip(recovered, expected):
            rel_err = abs(r - e) / e
            assert rel_err < 0.05, f"Axis {r:.4f} vs {e:.4f}: relative error {rel_err:.2%} > 5%"

    def test_wrap_params_structure(self, fitted):
        fitter, _ = fitted
        wp = fitter.wrap_params
        assert isinstance(wp, wrap_surface)
        assert wp.type_ == "WrapEllipsoid"
        assert wp.dimensions is not None
        assert len(wp.dimensions) == 3
        assert wp.translation is not None
        assert wp.xyz_body_rotation is not None

    def test_fit_with_labels_only(self):
        """Fitting with labels only (no SDF) should still produce reasonable results."""
        pts, labels, _sdf = _ellipsoid_labeled_points(self.TRUE_CENTER, self.TRUE_AXES, seed=99)

        fitter = EllipsoidFitter(
            lr=5e-3,
            epochs=100,
            use_lbfgs=False,
            alpha=1.0,
            beta=0.0,
            gamma=0.0,
            initialization="pca",
            center_transform="linear",
        )
        fitter.fit(points=pts, labels=labels, margin=0.005, plot=False)
        wp = fitter.wrap_params
        assert wp.dimensions is not None
        # Without SDF, accuracy is lower but center should still be reasonable
        center_err = np.linalg.norm(wp.translation - self.TRUE_CENTER)
        assert center_err < 0.01, f"Center error {center_err*1000:.1f}mm > 10mm"

    def test_pca_init_reasonable(self):
        """PCA initialization alone should estimate the center near truth."""
        pts, labels, _sdf = _ellipsoid_labeled_points(self.TRUE_CENTER, self.TRUE_AXES, seed=42)

        fitter = EllipsoidFitter(
            lr=1e-3,
            epochs=1,
            use_lbfgs=False,
            initialization="pca",
            center_transform="linear",
        )
        result = fitter.fit(points=pts, labels=labels, margin=0.005, plot=False)
        center_init = result[0].detach().cpu().numpy()

        # PCA center should be within 15 mm (it uses only inside-labeled points)
        np.testing.assert_allclose(center_init, self.TRUE_CENTER, atol=0.015)


# ---------------------------------------------------------------------------
# PatellaFitter
# ---------------------------------------------------------------------------


class TestPatellaFitter:
    """Test PatellaFitter with synthetic labeled meshes."""

    @staticmethod
    def _make_patella_and_wrap():
        """Create synthetic patella sphere and wrap-surface ellipsoid."""
        sphere = pv.Sphere(
            radius=0.02, center=(0.05, 0.06, 0.0), theta_resolution=30, phi_resolution=30
        )
        pat_mesh = Mesh(sphere)

        ellipsoid = pv.ParametricEllipsoid(xradius=0.015, yradius=0.012, zradius=0.01)
        ellipsoid = ellipsoid.translate([0.05, 0.06, 0.0])
        wrap_mesh = Mesh(ellipsoid)

        return pat_mesh, wrap_mesh

    def test_label_patella_within_wrap_extents(self):
        """label_patella_within_wrap_extents adds binary extent labels."""
        pat, wrap = self._make_patella_and_wrap()
        labeled = label_patella_within_wrap_extents(pat, wrap)

        for key in ("within_x_ellipse", "within_y_ellipse", "within_z_ellipse"):
            assert key in labeled.mesh.point_data, f"Missing label array: {key}"
            vals = labeled.mesh.point_data[key]
            assert set(np.unique(vals)).issubset({0, 1}), f"{key} is not binary"

    def test_patella_fitter_smoke(self):
        """PatellaFitter runs and returns center + radii."""
        pat, wrap = self._make_patella_and_wrap()
        labeled = label_patella_within_wrap_extents(pat, wrap)

        fitter = PatellaFitter(labeled)
        fitter.fit()

        params = fitter.fitted_params
        assert "center" in params
        assert "radii" in params
        assert len(params["center"]) == 3
        assert len(params["radii"]) == 3
        assert all(r >= 0 for r in params["radii"])

    def test_patella_fitter_wrap_params(self):
        """PatellaFitter.wrap_params returns a wrap_surface with correct type."""
        pat, wrap = self._make_patella_and_wrap()
        labeled = label_patella_within_wrap_extents(pat, wrap)

        fitter = PatellaFitter(labeled)
        fitter.fit()

        wp = fitter.wrap_params
        assert isinstance(wp, wrap_surface)
        assert wp.type_ == "WrapEllipsoid"
        assert wp.dimensions is not None
        assert wp.translation is not None
        # PatellaFitter always uses zero rotation
        np.testing.assert_array_equal(wp.xyz_body_rotation, [0, 0, 0])

    def test_compute_ellipsoid_parameters_known_sphere(self):
        """compute_ellipsoid_parameters_from_labeled_mesh on unit sphere."""
        sphere = pv.Sphere(
            radius=0.02, center=(0.0, 0.0, 0.0), theta_resolution=50, phi_resolution=50
        )
        mesh = Mesh(sphere)

        # Label every point as within extents
        mesh["within_x_ellipse"] = np.ones(mesh.mesh.n_points)
        mesh["within_y_ellipse"] = np.ones(mesh.mesh.n_points)
        mesh["within_z_ellipse"] = np.ones(mesh.mesh.n_points)

        params = compute_ellipsoid_parameters_from_labeled_mesh(mesh)

        # Center should be near origin
        np.testing.assert_allclose(params["center"], [0.0, 0.0, 0.0], atol=0.002)

        # Radii should match the sphere radius (~0.02)
        for r in params["radii"]:
            assert abs(r - 0.02) < 0.005, f"Radius {r:.4f} not close to 0.02"


# ---------------------------------------------------------------------------
# construct_cylinder_basis
# ---------------------------------------------------------------------------


class TestConstructCylinderBasis:
    """Unit tests for the construct_cylinder_basis helper."""

    def test_identity_for_z_axis(self):
        """Z-aligned axis should give rotation near identity."""
        axis = torch.tensor([0.0, 0.0, 1.0])
        R = construct_cylinder_basis(axis)
        np.testing.assert_allclose(
            R.numpy(), np.eye(3), atol=1e-6, err_msg="Z-axis basis not identity"
        )

    def test_orthonormal(self):
        """Basis should be orthonormal with determinant +1."""
        axis = torch.tensor([1.0, 2.0, 3.0])
        R = construct_cylinder_basis(axis)
        R_np = R.numpy()

        np.testing.assert_allclose(R_np.T @ R_np, np.eye(3), atol=1e-6)
        assert abs(np.linalg.det(R_np) - 1.0) < 1e-6

    def test_z_column_is_axis(self):
        """Third column of the rotation matrix should be the unit axis."""
        axis = torch.tensor([0.0, 1.0, 0.0])
        R = construct_cylinder_basis(axis)
        z_col = R[:, 2].numpy()
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(z_col, expected, atol=1e-6)

    def test_custom_reference_x_axis(self):
        """Providing a custom reference_x_axis should orient x_local accordingly."""
        axis = torch.tensor([0.0, 0.0, 1.0])
        ref_x = torch.tensor([0.0, 1.0, 0.0])
        R = construct_cylinder_basis(axis, reference_x_axis=ref_x)
        x_local = R[:, 0].numpy()
        # x_local should be close to [0, 1, 0] (projected onto plane perp to Z)
        assert np.dot(x_local, [0.0, 1.0, 0.0]) > 0.99
