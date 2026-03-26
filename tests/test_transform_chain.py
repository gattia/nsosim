"""Tests verifying the coordinate transform chain using reference and subject fixtures.

Two fixture sets test complementary aspects:

1. **Reference fixtures** (tests/fixtures/transforms/ref_*.json + nsm_recon_ref_*.vtk):
   Each bone was processed independently by script 1_Fit_NSM_models_to_ref_surfaces.
   Each bone's OSIM mesh was generated using that bone's own mean_orig.
   Tests that the conversion functions are mathematically correct.

2. **Subject fixtures** (tests/fixtures/transforms/subject_9003316/):
   Production subject where tibia/patella had femur_transform applied.
   All bones share fem_ref_center for OSIM conversion.
   Tests the production pipeline convention.

All reference meshes come from single deterministic reconstructions — same points
across coordinate spaces, so transforms should match to floating-point precision.

Fixtures are large (~130MB VTK meshes) and stored in GitHub Releases, not in git.
They are auto-downloaded on first test run. If download fails, mesh-dependent tests
are skipped. See tests/fixtures/transforms/download_fixtures.sh.
"""

import json
import logging
import os
import subprocess

import numpy as np
import pytest
from pymskt.mesh import Mesh

from nsosim.nsm_fitting import (
    apply_transform,
    convert_nsm_recon_to_OSIM,
    convert_nsm_recon_to_OSIM_,
    convert_OSIM_to_nsm,
    convert_OSIM_to_nsm_,
    undo_transform,
)

logger = logging.getLogger(__name__)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "transforms")
SUBJECT_DIR = os.path.join(FIXTURES_DIR, "subject_9003316")
DOWNLOAD_SCRIPT = os.path.join(FIXTURES_DIR, "download_fixtures.sh")

BONES = ["femur", "tibia", "patella"]

# Sentinel file to check if mesh fixtures are present
_REF_MESH_SENTINEL = os.path.join(FIXTURES_DIR, "nsm_recon_ref_femur.vtk")
_SUBJ_MESH_SENTINEL = os.path.join(SUBJECT_DIR, "femur_nsm_recon_mm.vtk")


def _fixtures_present():
    """Check if the large mesh fixtures are downloaded."""
    return os.path.isfile(_REF_MESH_SENTINEL) and os.path.isfile(_SUBJ_MESH_SENTINEL)


def _try_download_fixtures():
    """Attempt to auto-download fixtures via the download script."""
    if _fixtures_present():
        return True
    if not os.path.isfile(DOWNLOAD_SCRIPT):
        return False
    try:
        logger.info("Auto-downloading transform test fixtures...")
        subprocess.run(
            ["bash", DOWNLOAD_SCRIPT],
            check=True,
            capture_output=True,
            timeout=120,
        )
        return _fixtures_present()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        logger.warning("Failed to download fixtures: %s", e)
        return False


# Auto-download on module import
_MESH_FIXTURES_AVAILABLE = _try_download_fixtures()

requires_mesh_fixtures = pytest.mark.skipif(
    not _MESH_FIXTURES_AVAILABLE,
    reason="Mesh fixtures not available. Run: tests/fixtures/transforms/download_fixtures.sh",
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fem_ref_center():
    """Femur reference center (mean_orig) from the reference femur alignment."""
    with open(os.path.join(FIXTURES_DIR, "ref_femur_alignment.json")) as f:
        align = json.load(f)
    return np.array(align["mean_orig"])


# ---------------------------------------------------------------------------
# Reference bone fixtures (each bone processed independently)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=BONES)
def ref_bone_align(request):
    """Load reference alignment JSON only (no meshes, always available)."""
    bone = request.param
    with open(os.path.join(FIXTURES_DIR, f"ref_{bone}_alignment.json")) as f:
        align = json.load(f)
    return {
        "bone": bone,
        "transform_matrix": np.array(align["transform_matrix"]),
        "scale": align["scale"],
        "center": np.array(align["center"]),
        "mean_orig": np.array(align["mean_orig"]),
    }


@pytest.fixture(scope="module", params=BONES)
def ref_bone_data(request):
    """Load reference alignment + meshes at all 3 coordinate spaces for one bone."""
    bone = request.param
    with open(os.path.join(FIXTURES_DIR, f"ref_{bone}_alignment.json")) as f:
        align = json.load(f)

    return {
        "bone": bone,
        "transform_matrix": np.array(align["transform_matrix"]),
        "scale": align["scale"],
        "center": np.array(align["center"]),
        "mean_orig": np.array(align["mean_orig"]),
        "canonical": Mesh(os.path.join(FIXTURES_DIR, f"nsm_recon_ref_{bone}_nsm_space.vtk")),
        "aligned_mm": Mesh(os.path.join(FIXTURES_DIR, f"nsm_recon_ref_{bone}.vtk")),
        "osim": Mesh(os.path.join(FIXTURES_DIR, f"nsm_recon_ref_{bone}_osim_space.vtk")),
    }


# ---------------------------------------------------------------------------
# Subject bone fixtures (femur-aligned space, production convention)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=BONES)
def subject_bone_align(request):
    """Load subject alignment JSON only (no meshes, always available)."""
    bone = request.param
    with open(os.path.join(SUBJECT_DIR, f"{bone}_alignment.json")) as f:
        align = json.load(f)
    return {
        "bone": bone,
        "linear_transform": np.array(align["linear_transform"]),
        "scale": align["scale"],
        "center": np.array(align["center"]),
    }


@pytest.fixture(scope="module", params=BONES)
def subject_bone_data(request):
    """Load subject alignment + mm-space mesh for one bone."""
    bone = request.param
    with open(os.path.join(SUBJECT_DIR, f"{bone}_alignment.json")) as f:
        align = json.load(f)

    return {
        "bone": bone,
        "linear_transform": np.array(align["linear_transform"]),
        "scale": align["scale"],
        "center": np.array(align["center"]),
        "aligned_mm": Mesh(os.path.join(SUBJECT_DIR, f"{bone}_nsm_recon_mm.vtk")),
    }


# ==========================================================================
# Reference bone tests: conversion functions are mathematically correct
# ==========================================================================


@requires_mesh_fixtures
class TestRefAlignedMmToOsim:
    """Reference path: aligned mm → OSIM using each bone's own mean_orig."""

    def test_points_match(self, ref_bone_data):
        """convert_nsm_recon_to_OSIM_ should exactly recover the OSIM mesh."""
        pts_mm = ref_bone_data["aligned_mm"].point_coords.copy()
        pts_osim = convert_nsm_recon_to_OSIM_(pts_mm, ref_bone_data["mean_orig"])
        np.testing.assert_allclose(
            pts_osim,
            ref_bone_data["osim"].point_coords,
            atol=1e-12,
            err_msg=f"{ref_bone_data['bone']}: aligned mm → OSIM mismatch",
        )

    def test_point_count_preserved(self, ref_bone_data):
        """Conversion should not add or remove points."""
        assert (
            ref_bone_data["aligned_mm"].point_coords.shape
            == ref_bone_data["osim"].point_coords.shape
        )


@requires_mesh_fixtures
class TestRefCanonicalToOsim:
    """Reference path: canonical → OSIM (the synthetic decode path)."""

    def test_points_match(self, ref_bone_data):
        """Full chain: canonical → undo_transform → +mean_orig → /1000 → rotate."""
        pts_canonical = ref_bone_data["canonical"].point_coords.copy()
        pts_osim = convert_nsm_recon_to_OSIM(
            pts_canonical,
            ref_bone_data["transform_matrix"],
            ref_bone_data["scale"],
            ref_bone_data["center"],
            ref_bone_data["mean_orig"],
        )
        # Matrix inversion in undo_transform introduces ~1e-12 m error
        np.testing.assert_allclose(
            pts_osim,
            ref_bone_data["osim"].point_coords,
            atol=1e-10,
            err_msg=f"{ref_bone_data['bone']}: canonical → OSIM mismatch",
        )


@requires_mesh_fixtures
class TestRefCanonicalToAlignedMm:
    """Reference path: canonical → aligned mm (undo_transform only)."""

    def test_points_match(self, ref_bone_data):
        """undo_transform should recover the aligned mm mesh."""
        pts_canonical = ref_bone_data["canonical"].point_coords.copy()
        pts_mm = undo_transform(
            pts_canonical,
            ref_bone_data["transform_matrix"],
            ref_bone_data["scale"],
            ref_bone_data["center"],
        )
        # Matrix inversion introduces sub-nanometer error (~1e-9 mm)
        np.testing.assert_allclose(
            pts_mm,
            ref_bone_data["aligned_mm"].point_coords,
            atol=1e-8,
            err_msg=f"{ref_bone_data['bone']}: canonical → aligned mm mismatch",
        )


@requires_mesh_fixtures
class TestRefRoundtrip:
    """Reference: forward + inverse transforms should be identity."""

    def test_osim_to_canonical_to_osim(self, ref_bone_data):
        """OSIM → canonical → OSIM roundtrip."""
        pts_orig = ref_bone_data["osim"].point_coords.copy()
        pts_canonical = convert_OSIM_to_nsm(
            pts_orig.copy(),
            ref_bone_data["transform_matrix"],
            ref_bone_data["scale"],
            ref_bone_data["center"],
            ref_bone_data["mean_orig"],
        )
        pts_roundtrip = convert_nsm_recon_to_OSIM(
            pts_canonical,
            ref_bone_data["transform_matrix"],
            ref_bone_data["scale"],
            ref_bone_data["center"],
            ref_bone_data["mean_orig"],
        )
        np.testing.assert_allclose(
            pts_roundtrip,
            pts_orig,
            atol=1e-10,
            err_msg=f"{ref_bone_data['bone']}: OSIM roundtrip mismatch",
        )

    def test_osim_to_aligned_mm_to_osim(self, ref_bone_data):
        """OSIM → aligned mm → OSIM roundtrip (no matrix inversion needed)."""
        pts_orig = ref_bone_data["osim"].point_coords.copy()
        pts_mm = convert_OSIM_to_nsm_(pts_orig.copy(), ref_bone_data["mean_orig"])
        pts_roundtrip = convert_nsm_recon_to_OSIM_(pts_mm, ref_bone_data["mean_orig"])
        np.testing.assert_allclose(
            pts_roundtrip,
            pts_orig,
            atol=1e-12,
            err_msg=f"{ref_bone_data['bone']}: OSIM ↔ aligned mm roundtrip mismatch",
        )


# ==========================================================================
# Subject tests: production convention (fem_ref_center for all bones)
# ==========================================================================


@requires_mesh_fixtures
class TestSubjectCanonicalToOsimRoundtrip:
    """Subject: mm → canonical → OSIM roundtrip using fem_ref_center.

    This tests the synthetic decode path with production data:
    take the _nsm_recon_mm.vtk (already in femur-aligned space), push it to
    canonical via apply_transform, then recover OSIM via convert_nsm_recon_to_OSIM.
    The result should match the direct mm → OSIM conversion.
    """

    def test_roundtrip_matches_direct(self, subject_bone_data, fem_ref_center):
        """Roundtrip through canonical space should match direct conversion."""
        pts_mm = subject_bone_data["aligned_mm"].point_coords.copy()

        # Direct path: mm → OSIM
        pts_osim_direct = convert_nsm_recon_to_OSIM_(pts_mm.copy(), fem_ref_center)

        # Roundtrip: mm → canonical → OSIM
        pts_canonical = apply_transform(
            pts_mm.copy(),
            subject_bone_data["linear_transform"],
            subject_bone_data["scale"],
            subject_bone_data["center"],
        )
        pts_osim_roundtrip = convert_nsm_recon_to_OSIM(
            pts_canonical,
            subject_bone_data["linear_transform"],
            subject_bone_data["scale"],
            subject_bone_data["center"],
            fem_ref_center,
        )
        np.testing.assert_allclose(
            pts_osim_roundtrip,
            pts_osim_direct,
            atol=1e-10,
            err_msg=f"{subject_bone_data['bone']}: canonical roundtrip mismatch",
        )


@requires_mesh_fixtures
class TestSubjectFemRefCenterConvention:
    """Verify the production convention: all bones use fem_ref_center."""

    def test_all_bones_in_meter_range(self, subject_bone_data, fem_ref_center):
        """All bones should be in a plausible meter range after conversion."""
        pts_mm = subject_bone_data["aligned_mm"].point_coords.copy()
        pts_osim = convert_nsm_recon_to_OSIM_(pts_mm, fem_ref_center)
        # Knee bones should fit within a ~0.3m cube
        extent = pts_osim.max(axis=0) - pts_osim.min(axis=0)
        assert np.all(extent < 0.3), (
            f"{subject_bone_data['bone']} extent {extent} — too large for a knee bone"
        )
        assert np.all(extent > 0.01), (
            f"{subject_bone_data['bone']} extent {extent} — too small, units may be wrong"
        )


@requires_mesh_fixtures
class TestSubjectBonePositions:
    """Verify bones land in correct spatial positions when using fem_ref_center.

    The femur center (condyles) is near the OSIM origin. The tibia sits
    distal (~50mm below in Y). These tests use dedicated fixtures to avoid
    parameterized skips.
    """

    @pytest.fixture(scope="class")
    def femur_osim_center(self, fem_ref_center):
        with open(os.path.join(SUBJECT_DIR, "femur_alignment.json")) as f:
            align = json.load(f)
        mesh = Mesh(os.path.join(SUBJECT_DIR, "femur_nsm_recon_mm.vtk"))
        pts_osim = convert_nsm_recon_to_OSIM_(mesh.point_coords.copy(), fem_ref_center)
        return pts_osim.mean(axis=0)

    @pytest.fixture(scope="class")
    def tibia_osim_center(self, fem_ref_center):
        mesh = Mesh(os.path.join(SUBJECT_DIR, "tibia_nsm_recon_mm.vtk"))
        pts_osim = convert_nsm_recon_to_OSIM_(mesh.point_coords.copy(), fem_ref_center)
        return pts_osim.mean(axis=0)

    def test_femur_near_origin(self, femur_osim_center):
        """Femur center should be within ~50mm of origin in all axes."""
        assert np.all(np.abs(femur_osim_center) < 0.05), (
            f"Femur center {femur_osim_center} — expected near origin"
        )

    def test_tibia_distal_to_femur(self, tibia_osim_center):
        """Tibia center Y should be negative (distal to femur condyles at ~0)."""
        assert tibia_osim_center[1] < -0.02, (
            f"Tibia center Y={tibia_osim_center[1]:.4f}m — expected < -0.02m (distal to femur)"
        )


# ==========================================================================
# Alignment JSON structure tests
# ==========================================================================


class TestAlignmentJsonStructure:
    """Verify alignment JSON structure matches documented expectations.

    See CLAUDE.md "Per-bone linear_transform (alignment JSONs)".
    Uses lightweight JSON-only fixtures — runs without mesh downloads.
    """

    def test_ref_scale_is_one(self, ref_bone_align):
        assert ref_bone_align["scale"] == 1

    def test_ref_center_is_zero(self, ref_bone_align):
        np.testing.assert_array_equal(ref_bone_align["center"], [0.0, 0.0, 0.0])

    def test_ref_transform_is_4x4(self, ref_bone_align):
        T = ref_bone_align["transform_matrix"]
        assert T.shape == (4, 4)
        np.testing.assert_array_equal(T[3, :], [0, 0, 0, 1])

    def test_ref_uniform_scaling(self, ref_bone_align):
        """The 3x3 submatrix should encode uniform scaling * rotation."""
        T = ref_bone_align["transform_matrix"]
        col_norms = np.linalg.norm(T[:3, :3], axis=0)
        np.testing.assert_allclose(col_norms, col_norms[0], rtol=1e-3)

    def test_subject_scale_is_one(self, subject_bone_align):
        assert subject_bone_align["scale"] == 1

    def test_subject_center_is_zero(self, subject_bone_align):
        np.testing.assert_array_equal(subject_bone_align["center"], [0.0, 0.0, 0.0])

    def test_subject_transform_is_4x4(self, subject_bone_align):
        T = subject_bone_align["linear_transform"]
        assert T.shape == (4, 4)
        np.testing.assert_array_equal(T[3, :], [0, 0, 0, 1])

    def test_subject_uniform_scaling(self, subject_bone_align):
        """Subject transforms should also encode uniform scaling * rotation."""
        T = subject_bone_align["linear_transform"]
        col_norms = np.linalg.norm(T[:3, :3], axis=0)
        np.testing.assert_allclose(col_norms, col_norms[0], rtol=1e-3)


class TestAlignmentJsonKeyNames:
    """Verify the documented difference in JSON key names.

    Reference JSONs use 'transform_matrix' and include 'mean_orig'.
    Subject JSONs use 'linear_transform' and do NOT include 'mean_orig'.
    See CLAUDE.md note under "Per-bone linear_transform".
    """

    def test_ref_uses_transform_matrix_key(self):
        for bone in BONES:
            with open(os.path.join(FIXTURES_DIR, f"ref_{bone}_alignment.json")) as f:
                align = json.load(f)
            assert "transform_matrix" in align, f"ref_{bone} missing 'transform_matrix'"
            assert "mean_orig" in align, f"ref_{bone} missing 'mean_orig'"

    def test_subject_uses_linear_transform_key(self):
        for bone in BONES:
            with open(os.path.join(SUBJECT_DIR, f"{bone}_alignment.json")) as f:
                align = json.load(f)
            assert "linear_transform" in align, f"subject {bone} missing 'linear_transform'"
            assert "mean_orig" not in align, f"subject {bone} should not have 'mean_orig'"


class TestFemRefCenter:
    """Verify fem_ref_center matches the documented value."""

    def test_value(self, fem_ref_center):
        np.testing.assert_allclose(
            fem_ref_center, [-1.2176, -10.938, 8.1977], atol=0.001
        )

    def test_differs_from_tibia(self, fem_ref_center):
        """fem_ref_center (femur's mean_orig) differs from tibia's mean_orig."""
        with open(os.path.join(FIXTURES_DIR, "ref_tibia_alignment.json")) as f:
            tib_align = json.load(f)
        tib_mean_orig = np.array(tib_align["mean_orig"])
        assert not np.allclose(fem_ref_center, tib_mean_orig)
