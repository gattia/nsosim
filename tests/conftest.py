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


# ---------------------------------------------------------------------------
# Simple point clouds
# ---------------------------------------------------------------------------


@pytest.fixture
def unit_sphere_points():
    """100 points uniformly-ish sampled on a unit sphere (numpy)."""
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((100, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


@pytest.fixture
def origin_point():
    """Single point at the origin."""
    return np.zeros((1, 3))


# ---------------------------------------------------------------------------
# dict_bones skeleton (minimal, for validation tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dict_bones():
    """Minimal dict_bones structure with required keys for each bone."""
    return {
        "femur": {
            "ref": {
                "folder": "/tmp/ref",
                "bone_filename": "femur_bone.vtk",
            },
            "subject": {
                "folder": "/tmp/subj",
                "bone_filename": "femur_bone.vtk",
                "cart_filename": "femur_cart.vtk",
                "med_men_filename": "femur_med_men.vtk",
                "lat_men_filename": "femur_lat_men.vtk",
            },
            "model": {
                "path_model_state": "/tmp/model.pth",
                "path_model_config": "/tmp/config.json",
            },
        },
        "tibia": {
            "ref": {
                "folder": "/tmp/ref",
                "bone_filename": "tibia_bone.vtk",
            },
            "subject": {
                "folder": "/tmp/subj",
                "bone_filename": "tibia_bone.vtk",
                "cart_filename": "tibia_cart.vtk",
            },
            "model": {
                "path_model_state": "/tmp/model.pth",
                "path_model_config": "/tmp/config.json",
            },
        },
        "patella": {
            "ref": {
                "folder": "/tmp/ref",
                "bone_filename": "patella_bone.vtk",
            },
            "subject": {
                "folder": "/tmp/subj",
                "bone_filename": "patella_bone.vtk",
                "cart_filename": "patella_cart.vtk",
            },
            "model": {
                "path_model_state": "/tmp/model.pth",
                "path_model_config": "/tmp/config.json",
            },
        },
    }
