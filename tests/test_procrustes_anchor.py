"""Tests for the Procrustes-anchor module that builds Smith2019-derived
wrap-surface anchors for the wrap fitter (iter4 of WRAP_FITTER_ROBUSTNESS).

The anchor pipeline samples the Smith2019 wrap surface, transforms its points
via a similarity (rigid+scale) into subject frame, and algebraic-refits a
parametric wrap to recover anchor parameters. These tests cover:

  1. Umeyama recovers a known similarity exactly (rigid, scaled, translated).
  2. Sampled points actually lie on the wrap surface (ellipsoid + cylinder).
  3. Identity-transform roundtrip: sample → fit recovers the source params
     up to algebraic-fit precision.
  4. Known similarity transform: applying a (s, R, t) before refit yields
     anchor center/radius scaled and translated accordingly.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as ScipyRotation

from nsosim.wrap_surface_fitting.procrustes_anchor import (
    procrustes_anchor_for_wrap,
    sample_cylinder_surface_points,
    sample_ellipsoid_surface_points,
    transform_points,
    umeyama_similarity,
)

# ---------------------------------------------------------------------------
# Umeyama
# ---------------------------------------------------------------------------


def _random_similarity(seed=0):
    rng = np.random.default_rng(seed)
    s = float(rng.uniform(0.5, 2.0))
    R = ScipyRotation.from_euler("XYZ", rng.uniform(-np.pi / 3, np.pi / 3, size=3)).as_matrix()
    t = rng.uniform(-1.0, 1.0, size=3)
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T, s, R, t


def test_umeyama_recovers_identity_with_noise_free_corr():
    rng = np.random.default_rng(42)
    src = rng.normal(size=(100, 3))
    T = umeyama_similarity(src, src)
    np.testing.assert_allclose(T, np.eye(4), atol=1e-9)


def test_umeyama_recovers_known_similarity():
    T_true, s, R, t = _random_similarity(seed=7)
    rng = np.random.default_rng(13)
    src = rng.normal(size=(200, 3))
    dst = transform_points(T_true, src)
    T_est = umeyama_similarity(src, dst)
    np.testing.assert_allclose(T_est, T_true, atol=1e-9)


def test_umeyama_handles_pure_scale():
    rng = np.random.default_rng(9)
    src = rng.normal(size=(50, 3))
    dst = src * 1.7
    T = umeyama_similarity(src, dst)
    np.testing.assert_allclose(T[:3, :3], 1.7 * np.eye(3), atol=1e-9)
    np.testing.assert_allclose(T[:3, 3], np.zeros(3), atol=1e-9)


def test_umeyama_handles_reflection_input_safely():
    # If dst is a reflected version of src, output should still be a proper
    # rotation (det = +1) — Umeyama's S correction handles this.
    rng = np.random.default_rng(11)
    src = rng.normal(size=(80, 3))
    reflect = np.diag([1.0, 1.0, -1.0])
    dst = src @ reflect.T
    T = umeyama_similarity(src, dst)
    # The fitted similarity will not match exactly (it's a reflection, not
    # a rotation), but the rotation part should remain a proper rotation.
    s = np.linalg.norm(T[:3, 0])
    R_only = T[:3, :3] / s
    assert np.linalg.det(R_only) > 0


def test_umeyama_input_validation():
    src = np.zeros((4, 3))
    dst = np.zeros((4, 2))
    with pytest.raises(ValueError):
        umeyama_similarity(src, dst)

    with pytest.raises(ValueError):
        umeyama_similarity(np.zeros((2, 3)), np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# Point sampling
# ---------------------------------------------------------------------------


def test_sample_ellipsoid_points_lie_on_surface():
    params = {
        "type": "WrapEllipsoid",
        "dimensions": [0.025, 0.020, 0.012],
        "translation": [0.05, -0.08, 0.01],
        "xyz_body_rotation": [0.1, -0.2, 0.3],
    }
    pts = sample_ellipsoid_surface_points(params, n_points=2000)
    # Transform points back to local frame and check x²/a² + y²/b² + z²/c² ≈ 1.
    dims = np.asarray(params["dimensions"])
    trans = np.asarray(params["translation"])
    R = ScipyRotation.from_euler("XYZ", params["xyz_body_rotation"]).as_matrix()
    local = (pts - trans) @ R  # equivalent to R.T applied row-wise then transposed
    residual = (local / dims) ** 2
    np.testing.assert_allclose(residual.sum(axis=1), np.ones(2000), atol=1e-6)


def test_sample_cylinder_points_lie_on_surface():
    params = {
        "type": "WrapCylinder",
        "radius": 0.018,
        "length": 0.06,
        "translation": [0.0, -0.4, 0.0],
        "xyz_body_rotation": [0.05, 0.1, -0.07],
    }
    pts = sample_cylinder_surface_points(params, n_points=1000, n_along=40)
    R = ScipyRotation.from_euler("XYZ", params["xyz_body_rotation"]).as_matrix()
    trans = np.asarray(params["translation"])
    local = (pts - trans) @ R
    # Radial distance ≈ radius
    radial = np.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2)
    np.testing.assert_allclose(radial, np.full(pts.shape[0], params["radius"]), atol=1e-6)
    # Axial extent within +/- L/2
    assert local[:, 2].max() <= params["length"] / 2 + 1e-9
    assert local[:, 2].min() >= -params["length"] / 2 - 1e-9


# ---------------------------------------------------------------------------
# Anchor roundtrip
# ---------------------------------------------------------------------------


def test_anchor_ellipsoid_identity_roundtrip_recovers_params():
    """Identity bone_transform → refit should recover Smith2019 params (up to
    canonical-pose permutation/sign and algebraic-fit precision)."""
    params = {
        "type": "WrapEllipsoid",
        "dimensions": [0.025, 0.020, 0.012],
        "translation": [0.05, -0.08, 0.01],
        "xyz_body_rotation": [0.1, -0.2, 0.3],
    }
    anchor = procrustes_anchor_for_wrap("TestEll", params, bone_transform=None, n_points=4000)
    np.testing.assert_allclose(anchor.translation, params["translation"], atol=5e-5)
    # Dimensions: should match params sorted descending (canonical pose sorts).
    np.testing.assert_allclose(
        np.sort(anchor.dimensions)[::-1],
        np.sort(params["dimensions"])[::-1],
        atol=5e-5,
    )


def test_anchor_cylinder_identity_roundtrip_recovers_radius_and_center():
    params = {
        "type": "WrapCylinder",
        "radius": 0.018,
        "length": 0.06,
        "translation": [0.0, -0.4, 0.0],
        "xyz_body_rotation": [0.05, 0.1, -0.07],
    }
    anchor = procrustes_anchor_for_wrap("TestCyl", params, bone_transform=None, n_points=4000)
    # Center is geometrically meaningful (axis can slide → check perpendicular distance instead).
    # The cylinder's center should lie close to the original axis. Project
    # the recovered center onto the original axis and compare to the original
    # center projected the same way.
    R = ScipyRotation.from_euler("XYZ", params["xyz_body_rotation"]).as_matrix()
    axis = R[:, 2]
    trans = np.asarray(params["translation"])
    perp = (anchor.translation - trans) - np.dot(anchor.translation - trans, axis) * axis
    assert np.linalg.norm(perp) < 5e-4, f"perpendicular center drift {np.linalg.norm(perp):.2e}"
    # Radius recovered
    assert abs(anchor.radius - params["radius"]) < 5e-4


def test_anchor_ellipsoid_under_known_similarity_transforms_center_and_dims():
    """Apply (s, R, t) similarity → anchor.translation = T·c, dims = s·dims."""
    params = {
        "type": "WrapEllipsoid",
        "dimensions": [0.030, 0.022, 0.014],
        "translation": [0.0, 0.0, 0.0],
        "xyz_body_rotation": [0.0, 0.0, 0.0],
    }
    s = 1.4
    R = ScipyRotation.from_euler("XYZ", [0.2, -0.1, 0.05]).as_matrix()
    t = np.array([0.01, -0.02, 0.005])
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t

    anchor = procrustes_anchor_for_wrap("TestEll", params, bone_transform=T, n_points=4000)
    # Translation should be s*R @ 0 + t = t
    np.testing.assert_allclose(anchor.translation, t, atol=5e-5)
    # Dimensions should be s * original (sort first because canonical pose reorders).
    np.testing.assert_allclose(
        np.sort(anchor.dimensions)[::-1],
        np.sort(np.asarray(params["dimensions"]) * s)[::-1],
        atol=5e-5,
    )


def test_anchor_cylinder_under_known_similarity_scales_radius():
    params = {
        "type": "WrapCylinder",
        "radius": 0.02,
        "length": 0.10,
        "translation": [0.0, 0.0, 0.0],
        "xyz_body_rotation": [0.0, 0.0, 0.0],
    }
    s = 1.3
    R = ScipyRotation.from_euler("XYZ", [0.0, 0.0, 0.0]).as_matrix()
    t = np.array([0.0, 0.0, 0.0])
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    anchor = procrustes_anchor_for_wrap("TestCyl", params, bone_transform=T, n_points=4000)
    assert abs(anchor.radius - s * params["radius"]) < 5e-4
