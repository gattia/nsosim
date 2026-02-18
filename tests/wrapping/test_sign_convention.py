"""Tests for rotation matrix sign convention enforcement.

Uses the real RotationUtils.enforce_sign_convention from the codebase
instead of a mock copy.
"""

import numpy as np
import torch

from nsosim.wrap_surface_fitting.rotation_utils import RotationUtils


class TestEnforceSignConvention:
    """Tests for RotationUtils.enforce_sign_convention."""

    def test_positive_rotation_unchanged(self):
        """A rotation with positive diagonal entries should pass through mostly unchanged."""
        R = np.array(
            [
                [0.866, -0.5, 0.0],
                [0.5, 0.866, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        R_fixed = RotationUtils.enforce_sign_convention(R)
        assert np.allclose(R, R_fixed, atol=1e-6)

    def test_negative_rotation_gets_canonical_signs(self):
        """A rotation with negative diagonal should have X/Y columns flipped to positive."""
        R_negative = np.array(
            [
                [-0.866, 0.5, 0.0],
                [-0.5, -0.866, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        R_fixed = RotationUtils.enforce_sign_convention(R_negative)
        # After fix: X-axis (col 0) should point mostly +X, Y-axis (col 1) mostly +Y
        assert R_fixed[0, 0] >= 0, "X-axis should point in +X direction"
        assert R_fixed[1, 1] >= 0, "Y-axis should point in +Y direction"

    def test_idempotent(self):
        """Applying enforce_sign_convention twice gives the same result."""
        R = np.array(
            [
                [-0.866, 0.5, 0.0],
                [-0.5, -0.866, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        R_fixed = RotationUtils.enforce_sign_convention(R)
        R_fixed_twice = RotationUtils.enforce_sign_convention(R_fixed)
        assert np.allclose(R_fixed, R_fixed_twice, atol=1e-6)

    def test_right_handedness_preserved(self):
        """Output should always be right-handed (det = +1)."""
        R_left = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        R_fixed = RotationUtils.enforce_sign_convention(R_left)
        assert np.linalg.det(R_fixed) > 0, "Result must be right-handed"

    def test_torch_input(self):
        """Should accept and return torch tensors."""
        R = torch.tensor(
            [
                [0.866, -0.5, 0.0],
                [0.5, 0.866, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        R_fixed = RotationUtils.enforce_sign_convention(R)
        assert isinstance(R_fixed, torch.Tensor)

    def test_numpy_input(self):
        """Should accept and return numpy arrays."""
        R = np.eye(3)
        R_fixed = RotationUtils.enforce_sign_convention(R)
        assert isinstance(R_fixed, np.ndarray)
