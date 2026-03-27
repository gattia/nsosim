"""Tests for nsosim.transforms — similarity transform utilities."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nsosim.transforms import (
    compute_T_rel,
    compute_transform_deviations,
    decompose_similarity,
    deviations_to_transform,
    mean_rotation,
    recover_bone_transform,
)

FIXTURES_DIR = "tests/fixtures/transforms"
SUBJECT_DIR = f"{FIXTURES_DIR}/subject_9003316"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_similarity(scale, euler_deg, translation):
    """Build a 4x4 similarity transform from components."""
    R = Rotation.from_euler("XYZ", euler_deg, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = translation
    return T


def _load_alignment(path, key="linear_transform"):
    with open(path) as f:
        return np.array(json.load(f)[key])


# ---------------------------------------------------------------------------
# decompose_similarity
# ---------------------------------------------------------------------------


class TestDecomposeSimilarity:
    def test_identity(self):
        s, R, t = decompose_similarity(np.eye(4))
        assert s == pytest.approx(1.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(t, [0, 0, 0], atol=1e-14)

    def test_pure_scale(self):
        T = np.eye(4)
        T[:3, :3] *= 3.5
        s, R, t = decompose_similarity(T)
        assert s == pytest.approx(3.5)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-14)

    def test_roundtrip_random(self):
        """Compose → decompose recovers original components."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            scale = rng.uniform(0.5, 2.0)
            euler = rng.uniform(-30, 30, size=3)
            trans = rng.uniform(-10, 10, size=3)
            T = _make_similarity(scale, euler, trans)

            s_out, R_out, t_out = decompose_similarity(T)
            assert s_out == pytest.approx(scale, rel=1e-10)
            np.testing.assert_allclose(t_out, trans, atol=1e-12)

            R_expected = Rotation.from_euler("XYZ", euler, degrees=True).as_matrix()
            np.testing.assert_allclose(R_out, R_expected, atol=1e-10)

    def test_proper_rotation(self):
        """Returned R always has det = +1."""
        T = _make_similarity(0.013, [5, -2, 3], [0.01, -0.02, -0.7])
        _, R, _ = decompose_similarity(T)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)

    def test_subject_fixture(self):
        """Decompose a real subject femur transform — scale ~0.013."""
        T_fem = _load_alignment(f"{SUBJECT_DIR}/femur_alignment.json")
        s, R, t = decompose_similarity(T_fem)
        assert 0.012 < s < 0.015  # femur scale is ~0.013
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)
        # Femur rotation should be near-identity (bones are roughly aligned before ICP)
        angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        assert angle < 10  # less than 10 degrees from identity


# ---------------------------------------------------------------------------
# mean_rotation
# ---------------------------------------------------------------------------


class TestMeanRotation:
    def test_identical_rotations(self):
        """Mean of N identical rotations is that rotation."""
        R = Rotation.from_euler("XYZ", [15, -10, 20], degrees=True).as_matrix()
        rotations = np.stack([R] * 20)
        R_mean = mean_rotation(rotations)
        np.testing.assert_allclose(R_mean, R, atol=1e-12)

    def test_identity_mean(self):
        """Mean of identity rotations is identity."""
        rotations = np.stack([np.eye(3)] * 5)
        R_mean = mean_rotation(rotations)
        np.testing.assert_allclose(R_mean, np.eye(3), atol=1e-14)

    def test_output_is_proper_rotation(self):
        """Output always has det = +1 and is orthogonal."""
        rng = np.random.default_rng(123)
        rotations = np.array(
            [Rotation.from_euler("XYZ", rng.uniform(-20, 20, 3), degrees=True).as_matrix() for _ in range(30)]
        )
        R_mean = mean_rotation(rotations)
        assert np.linalg.det(R_mean) == pytest.approx(1.0, abs=1e-10)
        np.testing.assert_allclose(R_mean @ R_mean.T, np.eye(3), atol=1e-10)

    def test_symmetric_pair(self):
        """Mean of +θ and -θ rotations about one axis is identity."""
        R_pos = Rotation.from_euler("Z", 30, degrees=True).as_matrix()
        R_neg = Rotation.from_euler("Z", -30, degrees=True).as_matrix()
        R_mean = mean_rotation(np.stack([R_pos, R_neg]))
        np.testing.assert_allclose(R_mean, np.eye(3), atol=1e-10)


# ---------------------------------------------------------------------------
# compute_T_rel / recover_bone_transform roundtrip
# ---------------------------------------------------------------------------


class TestTRelRoundtrip:
    def test_synthetic(self):
        """T_rel → recover roundtrips exactly."""
        T_fem = _make_similarity(0.013, [1, -0.5, 0.3], [0.002, 0.003, -0.013])
        T_tib = _make_similarity(0.019, [2, 1, -1], [0.08, -0.22, 0.92])

        T_rel = compute_T_rel(T_fem, T_tib)
        T_tib_recovered = recover_bone_transform(T_rel, T_fem)
        np.testing.assert_allclose(T_tib_recovered, T_tib, atol=1e-12)

    def test_subject_fixture(self):
        """Roundtrip with real subject alignment JSONs."""
        T_fem = _load_alignment(f"{SUBJECT_DIR}/femur_alignment.json")
        T_tib = _load_alignment(f"{SUBJECT_DIR}/tibia_alignment.json")
        T_pat = _load_alignment(f"{SUBJECT_DIR}/patella_alignment.json")

        for T_other, name in [(T_tib, "tibia"), (T_pat, "patella")]:
            T_rel = compute_T_rel(T_fem, T_other)
            T_recovered = recover_bone_transform(T_rel, T_fem)
            np.testing.assert_allclose(T_recovered, T_other, atol=1e-10, err_msg=f"Failed for {name}")

    def test_T_rel_with_identity_femur(self):
        """When T_fem = I, T_rel = inv(T_other)."""
        T_other = _make_similarity(0.019, [3, -2, 1], [0.1, -0.2, 0.9])
        T_rel = compute_T_rel(np.eye(4), T_other)
        np.testing.assert_allclose(T_rel, np.linalg.inv(T_other), atol=1e-12)


# ---------------------------------------------------------------------------
# compute_transform_deviations
# ---------------------------------------------------------------------------


class TestComputeTransformDeviations:
    def test_single_transform_zero_deviations(self):
        """Single transform → deviations are all zero (mean = the only value)."""
        T = _make_similarity(0.71, [5, -3, 2], [-0.05, -0.02, -0.68])
        result = compute_transform_deviations(T[np.newaxis], mean_fem_scale=0.013)

        np.testing.assert_allclose(result["euler_angles_deg"], [[0, 0, 0]], atol=1e-10)
        np.testing.assert_allclose(result["translations_mm"], [[0, 0, 0]], atol=1e-10)
        np.testing.assert_allclose(result["scale_ratios"], [1.0], atol=1e-14)

    def test_deviations_center_at_zero(self):
        """Mean of deviations should be ~0 for angles and translations."""
        rng = np.random.default_rng(99)
        transforms = np.array(
            [_make_similarity(0.71 + rng.normal(0, 0.01), rng.normal(0, 5, 3), rng.normal([-0.05, -0.02, -0.68], 0.01))
             for _ in range(50)]
        )
        result = compute_transform_deviations(transforms, mean_fem_scale=0.013)

        # Euler angle mean doesn't perfectly center at zero due to nonlinearity,
        # but with 50 samples at ±5° std it should be well within 0.15°
        np.testing.assert_allclose(result["euler_angles_deg"].mean(axis=0), [0, 0, 0], atol=0.15)
        np.testing.assert_allclose(result["translations_mm"].mean(axis=0), [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(result["scale_ratios"].mean(), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# deviations_to_transform roundtrip
# ---------------------------------------------------------------------------


class TestDeviationsRoundtrip:
    def test_zero_deviations_give_mean(self):
        """Zero deviations reconstruct the mean transform."""
        s_mean = 0.71
        R_mean = Rotation.from_euler("XYZ", [5, -3, 2], degrees=True).as_matrix()
        t_mean = np.array([-0.05, -0.02, -0.68])

        T = deviations_to_transform(
            euler_angles_deg=[0, 0, 0],
            translation_mm=[0, 0, 0],
            scale_ratio=1.0,
            R_mean=R_mean,
            t_mean=t_mean,
            s_mean=s_mean,
            mean_fem_scale=0.013,
        )

        T_expected = np.eye(4)
        T_expected[:3, :3] = s_mean * R_mean
        T_expected[:3, 3] = t_mean
        np.testing.assert_allclose(T, T_expected, atol=1e-14)

    def test_decompose_recompose_roundtrip(self):
        """decompose → compute deviations → recompose recovers original transforms."""
        rng = np.random.default_rng(77)
        N = 20
        transforms = np.array(
            [_make_similarity(
                0.71 + rng.normal(0, 0.01),
                rng.normal(0, 5, 3),
                np.array([-0.05, -0.02, -0.68]) + rng.normal(0, 0.01, 3),
            ) for _ in range(N)]
        )
        mean_fem_scale = 0.013

        result = compute_transform_deviations(transforms, mean_fem_scale)

        for i in range(N):
            T_recomposed = deviations_to_transform(
                euler_angles_deg=result["euler_angles_deg"][i],
                translation_mm=result["translations_mm"][i],
                scale_ratio=result["scale_ratios"][i],
                R_mean=result["R_mean"],
                t_mean=result["t_mean"],
                s_mean=result["s_mean"],
                mean_fem_scale=mean_fem_scale,
            )
            np.testing.assert_allclose(T_recomposed, transforms[i], atol=1e-10, err_msg=f"Failed for subject {i}")


