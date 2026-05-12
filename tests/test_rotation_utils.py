"""Tests for rotation_utils.py: quaternion, Euler, axis-angle conversions."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as ScipyR

from nsosim.wrap_surface_fitting.rotation_utils import RotationUtils

# ---------------------------------------------------------------------------
# Quaternion <-> rotation matrix round-trips
# ---------------------------------------------------------------------------


class TestQuatRotRoundTrip:
    """quat_from_rot and rot_from_quat should be inverses of each other."""

    def test_identity(self, identity_rotation_torch):
        q = RotationUtils.quat_from_rot(identity_rotation_torch)
        R_back = RotationUtils.rot_from_quat(q)
        assert torch.allclose(R_back, identity_rotation_torch, atol=1e-6)

    def test_random_rotation(self, random_rotation_torch):
        q = RotationUtils.quat_from_rot(random_rotation_torch)
        R_back = RotationUtils.rot_from_quat(q)
        assert torch.allclose(R_back, random_rotation_torch, atol=1e-6)

    def test_90deg_x(self, known_90deg_x_rotation_np):
        R = torch.tensor(known_90deg_x_rotation_np, dtype=torch.float64)
        q = RotationUtils.quat_from_rot(R)
        R_back = RotationUtils.rot_from_quat(q)
        assert torch.allclose(R_back, R, atol=1e-6)

    def test_unit_quaternion_output(self, random_rotation_torch):
        """quat_from_rot should always return a unit quaternion."""
        q = RotationUtils.quat_from_rot(random_rotation_torch)
        assert torch.allclose(q.norm(), torch.tensor(1.0, dtype=torch.float64), atol=1e-6)

    @pytest.mark.parametrize("angle_deg", [0, 30, 45, 90, 120, 180])
    def test_z_rotations(self, angle_deg):
        """Round-trip through several Z-axis rotations."""
        angle = np.radians(angle_deg)
        R = torch.tensor(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ],
            dtype=torch.float64,
        )
        q = RotationUtils.quat_from_rot(R)
        R_back = RotationUtils.rot_from_quat(q)
        assert torch.allclose(R_back, R, atol=1e-6)


# ---------------------------------------------------------------------------
# Euler angle <-> rotation matrix round-trips
# ---------------------------------------------------------------------------


class TestEulerRoundTrip:
    """rot_to_euler_xyz_body should be consistent with scipy."""

    def test_identity_gives_zero_euler(self, identity_rotation_torch):
        euler = RotationUtils.rot_to_euler_xyz_body(identity_rotation_torch)
        assert torch.allclose(euler, torch.zeros(3, dtype=torch.float64), atol=1e-6)

    def test_matches_scipy(self, random_rotation_torch):
        """Euler angles should match scipy's intrinsic XYZ decomposition."""
        R_np = random_rotation_torch.numpy()
        euler_ours = RotationUtils.rot_to_euler_xyz_body(R_np)
        euler_scipy = ScipyR.from_matrix(R_np).as_euler("XYZ")
        assert np.allclose(euler_ours, euler_scipy, atol=1e-6)

    def test_euler_to_matrix_roundtrip(self, random_rotation_torch):
        """Euler -> matrix -> Euler should recover original angles (away from gimbal lock)."""
        euler = RotationUtils.rot_to_euler_xyz_body(random_rotation_torch)
        # Reconstruct via scipy to get the matrix back
        R_reconstructed = ScipyR.from_euler("XYZ", euler.numpy()).as_matrix()
        assert np.allclose(R_reconstructed, random_rotation_torch.numpy(), atol=1e-6)

    def test_numpy_input_returns_numpy(self, identity_rotation_np):
        euler = RotationUtils.rot_to_euler_xyz_body(identity_rotation_np)
        assert isinstance(euler, np.ndarray)

    def test_torch_input_returns_torch(self, identity_rotation_torch):
        euler = RotationUtils.rot_to_euler_xyz_body(identity_rotation_torch)
        assert isinstance(euler, torch.Tensor)

    @pytest.mark.parametrize("angle_deg", [30, 45, 60, 89])
    def test_single_axis_rotations_match_scipy(self, angle_deg):
        """Single-axis rotations should match scipy across a range of angles."""
        angle = np.radians(angle_deg)
        # X rotation
        R_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)],
            ]
        )
        euler_ours = RotationUtils.rot_to_euler_xyz_body(R_x)
        euler_scipy = ScipyR.from_matrix(R_x).as_euler("XYZ")
        assert np.allclose(
            euler_ours, euler_scipy, atol=1e-6
        ), f"Mismatch at {angle_deg}deg X rotation"

    @pytest.mark.parametrize("sign", [1, -1])
    def test_gimbal_lock_roundtrip(self, sign):
        """At Y=+/-90deg (gimbal lock), euler->matrix should still recover the rotation.

        The Euler angle decomposition is not unique at gimbal lock (X and Z are
        coupled), but reconstructing a matrix from the extracted angles must still
        yield the original rotation matrix.
        """
        # Y = +/-90 degrees -> R[0,2] = +/-1 (gimbal lock)
        R = ScipyR.from_euler("XYZ", [0.3, sign * np.pi / 2, 0.5]).as_matrix()
        euler_ours = RotationUtils.rot_to_euler_xyz_body(R)
        R_reconstructed = ScipyR.from_euler("XYZ", np.asarray(euler_ours)).as_matrix()
        assert np.allclose(
            R_reconstructed, R, atol=1e-6
        ), f"Gimbal lock roundtrip failed for Y={sign}*90deg"


# ---------------------------------------------------------------------------
# Axis-angle <-> rotation matrix round-trips
# ---------------------------------------------------------------------------


class TestAxisAngleRoundTrip:
    def test_identity(self, identity_rotation_torch):
        aa = RotationUtils.axis_angle_from_rot(identity_rotation_torch)
        assert torch.allclose(aa, torch.zeros(3, dtype=torch.float64), atol=1e-6)

    def test_roundtrip(self, random_rotation_torch):
        aa = RotationUtils.axis_angle_from_rot(random_rotation_torch)
        R_back = RotationUtils.rot_from_axis_angle(aa)
        assert torch.allclose(R_back, random_rotation_torch, atol=1e-6)

    def test_known_90deg_z(self):
        """90-degree rotation about Z -> axis-angle should be [0, 0, pi/2]."""
        R = torch.tensor(
            [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ],
            dtype=torch.float64,
        )
        aa = RotationUtils.axis_angle_from_rot(R)
        expected = torch.tensor([0.0, 0.0, np.pi / 2], dtype=torch.float64)
        assert torch.allclose(aa, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# enforce_sign_convention idempotency
# ---------------------------------------------------------------------------


class TestEnforceSignConventionIdempotency:
    """enforce_sign_convention applied twice should give the same result as once."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_idempotent_random(self, seed):
        R = torch.tensor(ScipyR.random(random_state=seed).as_matrix(), dtype=torch.float64)
        R1 = RotationUtils.enforce_sign_convention(R)
        R2 = RotationUtils.enforce_sign_convention(R1)
        assert torch.allclose(
            R1, R2, atol=1e-10
        ), f"enforce_sign_convention is not idempotent for seed={seed}"


# ---------------------------------------------------------------------------
# canonical_ellipsoid_pose
# ---------------------------------------------------------------------------


def _ellipsoid_matrix(R, axes):
    """Reconstruct the ellipsoid's quadratic-form matrix Q = R diag(a²) R^T."""
    R = np.asarray(R, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    return R @ np.diag(axes**2) @ R.T


class TestCanonicalEllipsoidPose:
    """canonical_ellipsoid_pose must preserve geometry and be stable to perturbation."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_geometric_invariance(self, seed):
        # Random rotation + random distinct axes — canonicalization must return
        # the same geometric ellipsoid.
        rng = np.random.default_rng(seed)
        R = ScipyR.random(random_state=seed).as_matrix()
        axes = rng.uniform(0.005, 0.05, size=3)
        R2, axes2 = RotationUtils.canonical_ellipsoid_pose(R, axes)
        np.testing.assert_allclose(
            _ellipsoid_matrix(R, axes), _ellipsoid_matrix(R2, axes2), atol=1e-14
        )

    def test_idempotent(self):
        # Applying canonicalization twice should produce the same result.
        R = ScipyR.random(random_state=42).as_matrix()
        axes = np.array([0.012, 0.017, 0.032])
        R1, a1 = RotationUtils.canonical_ellipsoid_pose(R, axes)
        R2, a2 = RotationUtils.canonical_ellipsoid_pose(R1, a1)
        np.testing.assert_allclose(R1, R2, atol=1e-14)
        np.testing.assert_allclose(a1, a2, atol=1e-14)

    def test_axes_sorted_descending(self):
        R = ScipyR.random(random_state=7).as_matrix()
        axes = np.array([0.012, 0.017, 0.032])
        _, axes_canon = RotationUtils.canonical_ellipsoid_pose(R, axes)
        assert axes_canon[0] >= axes_canon[1] >= axes_canon[2]

    def test_right_handed(self):
        R = ScipyR.random(random_state=11).as_matrix()
        R[:, 1] *= -1  # make left-handed
        axes = np.array([0.01, 0.02, 0.03])
        R2, _ = RotationUtils.canonical_ellipsoid_pose(R, axes)
        assert np.linalg.det(R2) > 0

    def test_stable_near_gimbal_lock(self):
        """The whole point: small perturbations of a near-gimbal-lock R give
        a small canonical-pose change in Euler-angle output."""
        # Med_Lig_r-like: middle Euler ~ 83° (near gimbal lock at 90°)
        euler_A = np.array([-0.681942, 1.451605, 0.692665])
        euler_B = np.array([-0.800769, 1.437294, 0.816801])
        axes_A = np.array([0.012032, 0.017258, 0.032061])
        axes_B = np.array([0.012010, 0.017308, 0.032030])
        RA = ScipyR.from_euler("XYZ", euler_A).as_matrix()
        RB = ScipyR.from_euler("XYZ", euler_B).as_matrix()
        # The geodesic angle between RA and RB is ~1.24°
        geodesic_deg = np.degrees(np.arccos(np.clip((np.trace(RA.T @ RB) - 1) / 2, -1, 1)))
        assert geodesic_deg < 1.5, f"setup sanity: geodesic should be ~1.24°, got {geodesic_deg}"

        RA_c, _ = RotationUtils.canonical_ellipsoid_pose(RA, axes_A)
        RB_c, _ = RotationUtils.canonical_ellipsoid_pose(RB, axes_B)
        eA = RotationUtils.rot_to_euler_xyz_body(RA_c)
        eB = RotationUtils.rot_to_euler_xyz_body(RB_c)
        max_euler_diff_deg = np.max(np.abs(np.degrees(eB - eA)))
        # Old enforce_sign_convention gave 7.1° here; new canonical should be ≤ 2× geodesic.
        assert max_euler_diff_deg < 2.0, (
            f"Canonical pose Euler drift should be near geodesic ({geodesic_deg:.2f}°), "
            f"got {max_euler_diff_deg:.2f}°"
        )

    def test_numpy_input_returns_numpy(self):
        R = np.eye(3)
        axes = np.array([0.01, 0.02, 0.03])
        R2, a2 = RotationUtils.canonical_ellipsoid_pose(R, axes)
        assert isinstance(R2, np.ndarray)
        assert isinstance(a2, np.ndarray)

    def test_torch_input_returns_torch(self):
        R = torch.eye(3, dtype=torch.float64)
        axes = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)
        R2, a2 = RotationUtils.canonical_ellipsoid_pose(R, axes)
        assert isinstance(R2, torch.Tensor)
        assert isinstance(a2, torch.Tensor)
