"""Tests for coordinate system transforms between NSM space and OpenSim space."""

import numpy as np
import pytest

from nsosim.nsm_fitting import (
    OSIM_TO_NSM_TRANSFORM,
    convert_nsm_recon_to_OSIM_,
    convert_OSIM_to_nsm_,
)

# ---------------------------------------------------------------------------
# Transform matrix properties
# ---------------------------------------------------------------------------


class TestOsimToNsmTransform:
    """Verify algebraic properties of the OSIM_TO_NSM_TRANSFORM matrix."""

    def test_is_orthogonal(self):
        """R^T @ R should be identity (orthogonal matrix)."""
        RtR = OSIM_TO_NSM_TRANSFORM.T @ OSIM_TO_NSM_TRANSFORM
        assert np.allclose(
            RtR, np.eye(3), atol=1e-10
        ), f"OSIM_TO_NSM_TRANSFORM is not orthogonal: R^T R =\n{RtR}"

    def test_determinant_plus_or_minus_one(self):
        """Determinant should be +1 (proper rotation) or -1 (improper / reflection)."""
        det = np.linalg.det(OSIM_TO_NSM_TRANSFORM)
        assert abs(abs(det) - 1.0) < 1e-10, f"det = {det}, expected +/-1"

    def test_shape(self):
        assert OSIM_TO_NSM_TRANSFORM.shape == (3, 3)


# ---------------------------------------------------------------------------
# Round-trip tests (NSM -> OSIM -> NSM and vice versa)
# ---------------------------------------------------------------------------


class TestCoordinateRoundTrip:
    """convert_nsm_recon_to_OSIM_ and convert_OSIM_to_nsm_ should be inverses."""

    @pytest.fixture
    def ref_center(self):
        """A fixed reference mesh center bias vector (mm)."""
        return np.array([100.0, 200.0, 300.0])

    def test_roundtrip_nsm_to_osim_to_nsm(self, ref_center):
        """NSM -> OSIM -> NSM should recover original points."""
        points_original = np.array(
            [
                [10.0, 20.0, 30.0],
                [-5.0, 15.0, -25.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        # Note: convert_nsm_recon_to_OSIM_ modifies in-place (+=, /=), so copy
        pts = points_original.copy()
        pts_osim = convert_nsm_recon_to_OSIM_(pts, ref_center.copy())
        pts_back = convert_OSIM_to_nsm_(pts_osim, ref_center.copy())

        assert np.allclose(
            pts_back, points_original, atol=1e-8
        ), f"Round-trip failed.\nOriginal:\n{points_original}\nRecovered:\n{pts_back}"

    def test_roundtrip_osim_to_nsm_to_osim(self, ref_center):
        """OSIM -> NSM -> OSIM should recover original points."""
        points_osim_original = np.array(
            [
                [0.1, 0.2, 0.3],
                [-0.05, 0.0, 0.15],
            ],
            dtype=np.float64,
        )

        pts = points_osim_original.copy()
        pts_nsm = convert_OSIM_to_nsm_(pts, ref_center.copy())
        pts_back = convert_nsm_recon_to_OSIM_(pts_nsm, ref_center.copy())

        assert np.allclose(pts_back, points_osim_original, atol=1e-8)

    def test_origin_with_zero_bias(self):
        """With zero bias, origin should map to origin (after mm->m scaling)."""
        ref_center = np.zeros(3)
        pts = np.zeros((1, 3), dtype=np.float64)
        pts_osim = convert_nsm_recon_to_OSIM_(pts.copy(), ref_center.copy())
        assert np.allclose(pts_osim, np.zeros((1, 3)), atol=1e-10)


# ---------------------------------------------------------------------------
# Unit conversion consistency
# ---------------------------------------------------------------------------


class TestUnitConversion:
    """NSM space is in mm, OpenSim space is in m.  1 m = 1000 mm."""

    def test_mm_to_m_scaling(self):
        """A 1000mm offset in NSM space should be 1m in OSIM space (modulo rotation)."""
        ref_center = np.zeros(3)
        # 1000 mm along one axis
        pts = np.array([[1000.0, 0.0, 0.0]], dtype=np.float64)
        pts_osim = convert_nsm_recon_to_OSIM_(pts.copy(), ref_center.copy())
        # The magnitude should be 1.0 m (rotation preserves norm)
        assert np.allclose(np.linalg.norm(pts_osim), 1.0, atol=1e-8)

    def test_m_to_mm_scaling(self):
        """1m in OSIM space -> 1000mm magnitude in NSM space."""
        ref_center = np.zeros(3)
        pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        pts_nsm = convert_OSIM_to_nsm_(pts.copy(), ref_center.copy())
        assert np.allclose(np.linalg.norm(pts_nsm), 1000.0, atol=1e-5)
