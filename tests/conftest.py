"""Shared pytest fixtures for nsosim tests."""

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Rotation matrices
# ---------------------------------------------------------------------------


@pytest.fixture
def identity_rotation_np():
    """3x3 identity rotation matrix (numpy)."""
    return np.eye(3)


@pytest.fixture
def identity_rotation_torch():
    """3x3 identity rotation matrix (torch, float64)."""
    return torch.eye(3, dtype=torch.float64)


@pytest.fixture
def known_90deg_x_rotation_np():
    """90-degree rotation about X axis (numpy).

    Rotates Y -> Z, Z -> -Y.
    """
    return np.array(
        [
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def random_rotation_torch():
    """A deterministic 'random' rotation matrix (torch, float64).

    Built from a fixed axis-angle so tests are reproducible.
    """
    axis = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    axis = axis / torch.norm(axis)
    angle = torch.tensor(1.2, dtype=torch.float64)

    K = torch.tensor(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ],
        dtype=torch.float64,
    )

    R = torch.eye(3, dtype=torch.float64) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    return R
