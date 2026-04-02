"""Tests for nsosim.decode — latent vector → OSIM-space mesh decoding.

These are integration tests requiring:
- CUDA-capable GPU (NSM decoder runs on GPU)
- NSM model weights in tests/fixtures/models/{bone}/model.pth
- Mesh fixtures in tests/fixtures/transforms/ (auto-downloaded)

Tests that cannot run due to missing GPU or model weights are skipped gracefully.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TRANSFORMS_DIR = FIXTURES_DIR / "transforms"
MODELS_DIR = FIXTURES_DIR / "models"

BONES = ["femur", "tibia", "patella"]

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

try:
    import torch

    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False

_MODELS_AVAILABLE = all((MODELS_DIR / bone / "model.pth").exists() for bone in BONES)

# Mesh fixtures for comparison (auto-downloaded in test_transform_chain.py)
_MESH_FIXTURES_AVAILABLE = (TRANSFORMS_DIR / "nsm_recon_ref_femur_osim_space.vtk").exists()

requires_gpu = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
requires_nsm_models = pytest.mark.skipif(
    not _MODELS_AVAILABLE,
    reason="NSM model weights not in tests/fixtures/models/",
)
requires_mesh_fixtures = pytest.mark.skipif(
    not _MESH_FIXTURES_AVAILABLE,
    reason="Mesh fixtures not available. Run: tests/fixtures/transforms/download_fixtures.sh",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _load_model_config(bone):
    config_path = MODELS_DIR / bone / "model_params_config.json"
    with open(config_path) as f:
        return json.load(f)


def _load_ref_alignment(bone):
    path = TRANSFORMS_DIR / f"ref_{bone}_alignment.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def fem_ref_center():
    align = _load_ref_alignment("femur")
    return np.array(align["mean_orig"])


@pytest.fixture(scope="module")
def loaded_models():
    """Load all 3 NSM models (expensive — module scope)."""
    from nsosim.utils import load_model

    models = {}
    configs = {}
    for bone in BONES:
        config = _load_model_config(bone)
        model = load_model(
            config,
            str(MODELS_DIR / bone / "model.pth"),
            model_type="triplanar",
        )
        models[bone] = model
        configs[bone] = config
    return models, configs


@pytest.fixture(scope="module")
def ref_transforms():
    """Load reference alignment transforms for all bones."""
    transforms = {}
    for bone in BONES:
        align = _load_ref_alignment(bone)
        transforms[bone] = np.array(align["transform_matrix"])
    return transforms


@pytest.fixture(scope="module")
def ref_latents():
    """Load reference latent vectors for all bones."""
    latents = {}
    for bone in BONES:
        latents[bone] = np.load(TRANSFORMS_DIR / f"latent_{bone}.npy")
    return latents


# ---------------------------------------------------------------------------
# Single-bone decode: decode_latent_to_osim
# ---------------------------------------------------------------------------


@requires_gpu
@requires_nsm_models
class TestDecodeLatentToOsim:
    """Test decode_latent_to_osim with reference data."""

    @pytest.fixture(scope="class")
    def decoded_femur(self, loaded_models, ref_transforms, ref_latents, fem_ref_center):
        from nsosim.decode import decode_latent_to_osim

        models, configs = loaded_models
        return decode_latent_to_osim(
            latent_vector=ref_latents["femur"],
            model=models["femur"],
            linear_transform=ref_transforms["femur"],
            fem_ref_center=fem_ref_center,
            model_config=configs["femur"],
            n_pts_per_axis=256,
        )

    @pytest.fixture(scope="class")
    def decoded_tibia(self, loaded_models, ref_transforms, ref_latents, fem_ref_center):
        from nsosim.decode import decode_latent_to_osim

        models, configs = loaded_models
        # Tibia reference was processed with its own mean_orig, not fem_ref_center.
        tibia_align = _load_ref_alignment("tibia")
        tibia_ref_center = np.array(tibia_align["mean_orig"])
        return decode_latent_to_osim(
            latent_vector=ref_latents["tibia"],
            model=models["tibia"],
            linear_transform=ref_transforms["tibia"],
            fem_ref_center=tibia_ref_center,
            model_config=configs["tibia"],
            n_pts_per_axis=256,
        )

    @pytest.fixture(scope="class")
    def decoded_patella(self, loaded_models, ref_transforms, ref_latents, fem_ref_center):
        from nsosim.decode import decode_latent_to_osim

        models, configs = loaded_models
        patella_align = _load_ref_alignment("patella")
        patella_ref_center = np.array(patella_align["mean_orig"])
        return decode_latent_to_osim(
            latent_vector=ref_latents["patella"],
            model=models["patella"],
            linear_transform=ref_transforms["patella"],
            fem_ref_center=patella_ref_center,
            model_config=configs["patella"],
            n_pts_per_axis=256,
        )

    def test_femur_returns_expected_keys(self, decoded_femur):
        """Femur model with 4 objects should return bone, cart, med_men, lat_men."""
        assert set(decoded_femur.keys()) == {"bone", "cart", "med_men", "lat_men"}

    def test_tibia_returns_expected_keys(self, decoded_tibia):
        """Tibia model with 2 objects should return bone, cart."""
        assert set(decoded_tibia.keys()) == {"bone", "cart"}

    def test_patella_returns_expected_keys(self, decoded_patella):
        """Patella model with 2 objects should return bone, cart."""
        assert set(decoded_patella.keys()) == {"bone", "cart"}

    def test_femur_meshes_have_points(self, decoded_femur):
        for name, mesh in decoded_femur.items():
            assert mesh.point_coords.shape[0] > 100, f"{name} has too few points"
            assert mesh.point_coords.shape[1] == 3

    def test_tibia_meshes_have_points(self, decoded_tibia):
        for name, mesh in decoded_tibia.items():
            assert mesh.point_coords.shape[0] > 100, f"{name} has too few points"
            assert mesh.point_coords.shape[1] == 3

    def test_patella_meshes_have_points(self, decoded_patella):
        for name, mesh in decoded_patella.items():
            assert mesh.point_coords.shape[0] > 100, f"{name} has too few points"
            assert mesh.point_coords.shape[1] == 3

    def test_femur_bone_in_osim_spatial_range(self, decoded_femur):
        """OSIM-space femur bone should be within plausible spatial bounds (meters)."""
        pts = decoded_femur["bone"].point_coords
        assert np.all(np.abs(pts) < 0.3), "Femur bone points outside plausible OSIM range"

    def test_tibia_bone_in_osim_spatial_range(self, decoded_tibia):
        pts = decoded_tibia["bone"].point_coords
        assert np.all(np.abs(pts) < 0.3), "Tibia bone points outside plausible OSIM range"

    def test_patella_bone_in_osim_spatial_range(self, decoded_patella):
        pts = decoded_patella["bone"].point_coords
        assert np.all(np.abs(pts) < 0.3), "Patella bone points outside plausible OSIM range"

    @requires_mesh_fixtures
    def test_femur_bone_close_to_reference_osim(self, decoded_femur):
        """Same model + latent + transform, independent marching cubes runs.

        Observed: centroid <0.02mm, extent <0.02%, ASSD ~0.005mm.
        """
        from pymskt.mesh import Mesh

        ref_mesh = Mesh(str(TRANSFORMS_DIR / "nsm_recon_ref_femur_osim_space.vtk"))
        decoded_pts = decoded_femur["bone"].point_coords
        ref_pts = ref_mesh.point_coords

        decoded_centroid = decoded_pts.mean(axis=0)
        ref_centroid = ref_pts.mean(axis=0)
        np.testing.assert_allclose(decoded_centroid, ref_centroid, atol=0.0001)

        decoded_extent = decoded_pts.max(axis=0) - decoded_pts.min(axis=0)
        ref_extent = ref_pts.max(axis=0) - ref_pts.min(axis=0)
        np.testing.assert_allclose(decoded_extent, ref_extent, rtol=0.005)

        assd = decoded_femur["bone"].get_assd_mesh(ref_mesh)
        assert assd < 0.00005, f"Femur ASSD={assd * 1000:.4f}mm, expected <0.05mm"

    @requires_mesh_fixtures
    def test_tibia_bone_close_to_reference_osim(self, decoded_tibia):
        """Observed: centroid <0.008mm, extent <0.007%, ASSD ~0.001mm."""
        from pymskt.mesh import Mesh

        ref_mesh = Mesh(str(TRANSFORMS_DIR / "nsm_recon_ref_tibia_osim_space.vtk"))
        decoded_pts = decoded_tibia["bone"].point_coords
        ref_pts = ref_mesh.point_coords

        decoded_centroid = decoded_pts.mean(axis=0)
        ref_centroid = ref_pts.mean(axis=0)
        np.testing.assert_allclose(decoded_centroid, ref_centroid, atol=0.0001)

        decoded_extent = decoded_pts.max(axis=0) - decoded_pts.min(axis=0)
        ref_extent = ref_pts.max(axis=0) - ref_pts.min(axis=0)
        np.testing.assert_allclose(decoded_extent, ref_extent, rtol=0.005)

        assd = decoded_tibia["bone"].get_assd_mesh(ref_mesh)
        assert assd < 0.00005, f"Tibia ASSD={assd * 1000:.4f}mm, expected <0.05mm"

    @requires_mesh_fixtures
    def test_patella_bone_close_to_reference_osim(self, decoded_patella):
        """Same model + latent + transform as reference fixture."""
        from pymskt.mesh import Mesh

        ref_mesh = Mesh(str(TRANSFORMS_DIR / "nsm_recon_ref_patella_osim_space.vtk"))
        decoded_pts = decoded_patella["bone"].point_coords
        ref_pts = ref_mesh.point_coords

        decoded_centroid = decoded_pts.mean(axis=0)
        ref_centroid = ref_pts.mean(axis=0)
        np.testing.assert_allclose(decoded_centroid, ref_centroid, atol=0.0001)

        decoded_extent = decoded_pts.max(axis=0) - decoded_pts.min(axis=0)
        ref_extent = ref_pts.max(axis=0) - ref_pts.min(axis=0)
        np.testing.assert_allclose(decoded_extent, ref_extent, rtol=0.005)

        assd = decoded_patella["bone"].get_assd_mesh(ref_mesh)
        assert assd < 0.00005, f"Patella ASSD={assd * 1000:.4f}mm, expected <0.05mm"


# ---------------------------------------------------------------------------
# Full joint decode: decode_joint_from_descriptors
# ---------------------------------------------------------------------------


@requires_gpu
@requires_nsm_models
class TestDecodeJointFromDescriptors:
    """Test decode_joint_from_descriptors with reference data."""

    @pytest.fixture(scope="class")
    def decoded_joint(self, loaded_models, ref_transforms, ref_latents, fem_ref_center):
        """Decode a full joint using reference latents and transforms.

        For this test we use each bone's own mean_orig via fem_ref_center=femur's,
        and compute T_rel from the reference transforms. Since reference bones were
        each processed independently with their own mean_orig, we use the femur's
        mean_orig for this joint test — the spatial positions won't match production
        reference exactly, but the joint assembly test checks relative positions.
        """
        from nsosim.decode import decode_joint_from_descriptors
        from nsosim.transforms import compute_T_rel

        models, configs = loaded_models

        T_fem = ref_transforms["femur"]
        T_rel_tib = compute_T_rel(T_fem, ref_transforms["tibia"])
        T_rel_pat = compute_T_rel(T_fem, ref_transforms["patella"])

        return decode_joint_from_descriptors(
            femur_latent=ref_latents["femur"],
            tibia_latent=ref_latents["tibia"],
            patella_latent=ref_latents["patella"],
            T_fem=T_fem,
            T_rel_tib=T_rel_tib,
            T_rel_pat=T_rel_pat,
            models=models,
            model_configs=configs,
            fem_ref_center=fem_ref_center,
            n_pts_per_axis=256,
        )

    def test_returns_all_bones(self, decoded_joint):
        assert set(decoded_joint.keys()) == {"femur", "tibia", "patella"}

    def test_each_bone_has_meshes(self, decoded_joint):
        for bone_name, meshes in decoded_joint.items():
            assert "bone" in meshes, f"{bone_name} missing 'bone' mesh"
            assert meshes["bone"].point_coords.shape[0] > 100

    def test_femur_mesh_keys(self, decoded_joint):
        """Femur model (4 objects) should have bone, cart, med_men, lat_men."""
        assert set(decoded_joint["femur"].keys()) == {"bone", "cart", "med_men", "lat_men"}

    def test_tibia_mesh_keys(self, decoded_joint):
        """Tibia model (2 objects) should have bone, cart."""
        assert set(decoded_joint["tibia"].keys()) == {"bone", "cart"}

    def test_patella_mesh_keys(self, decoded_joint):
        """Patella model (2 objects) should have bone, cart."""
        assert set(decoded_joint["patella"].keys()) == {"bone", "cart"}

    def test_all_meshes_in_osim_range(self, decoded_joint):
        """All decoded meshes should be in plausible OSIM spatial range."""
        for bone_name, meshes in decoded_joint.items():
            for mesh_name, mesh in meshes.items():
                pts = mesh.point_coords
                assert np.all(
                    np.abs(pts) < 0.3
                ), f"{bone_name}/{mesh_name} outside plausible OSIM range"


# ---------------------------------------------------------------------------
# Phase D: Subject decode via decode_joint_from_descriptors vs production
# ---------------------------------------------------------------------------

SUBJECT_DIR = TRANSFORMS_DIR / "subject_9003316"


def _load_subject_alignment(bone):
    path = SUBJECT_DIR / f"{bone}_alignment.json"
    with open(path) as f:
        return json.load(f)


@requires_gpu
@requires_nsm_models
class TestSubjectDecodeVsProduction:
    """Decode subject 9003316 via decode_joint_from_descriptors and compare
    against production *_nsm_recon_mm.vtk meshes converted to OSIM space.

    This validates the full T_rel recovery path: subject alignment JSONs →
    compute_T_rel → decode_joint_from_descriptors → OSIM meshes, compared
    against the independently-produced production meshes (which went through
    the fitting pipeline: reconstruct_mesh → nsm_recon_to_osim).
    """

    @pytest.fixture(scope="class")
    def subject_transforms(self):
        transforms = {}
        for bone in BONES:
            align = _load_subject_alignment(bone)
            transforms[bone] = np.array(align["linear_transform"])
        return transforms

    @pytest.fixture(scope="class")
    def subject_latents(self):
        latents = {}
        for bone in BONES:
            latents[bone] = np.load(SUBJECT_DIR / f"{bone}_latent.npy")
        return latents

    @pytest.fixture(scope="class")
    def decoded_subject_joint(
        self, loaded_models, subject_transforms, subject_latents, fem_ref_center
    ):
        from nsosim.decode import decode_joint_from_descriptors
        from nsosim.transforms import compute_T_rel

        models, configs = loaded_models

        T_fem = subject_transforms["femur"]
        T_rel_tib = compute_T_rel(T_fem, subject_transforms["tibia"])
        T_rel_pat = compute_T_rel(T_fem, subject_transforms["patella"])

        return decode_joint_from_descriptors(
            femur_latent=subject_latents["femur"],
            tibia_latent=subject_latents["tibia"],
            patella_latent=subject_latents["patella"],
            T_fem=T_fem,
            T_rel_tib=T_rel_tib,
            T_rel_pat=T_rel_pat,
            models=models,
            model_configs=configs,
            fem_ref_center=fem_ref_center,
            n_pts_per_axis=256,
        )

    @pytest.fixture(scope="class")
    def production_osim_meshes(self, fem_ref_center):
        """Load production mm-space meshes and convert to OSIM for comparison."""
        from pymskt.mesh import Mesh

        from nsosim.nsm_fitting import convert_nsm_recon_to_OSIM_

        meshes = {}
        for bone in BONES:
            vtk_path = SUBJECT_DIR / f"{bone}_nsm_recon_mm.vtk"
            mesh = Mesh(str(vtk_path))
            pts_osim = convert_nsm_recon_to_OSIM_(mesh.point_coords.copy(), fem_ref_center)
            mesh.point_coords = pts_osim
            meshes[bone] = mesh
        return meshes

    @pytest.fixture(scope="class", params=BONES)
    def bone_comparison(self, request, decoded_subject_joint, production_osim_meshes):
        bone = request.param
        return bone, decoded_subject_joint[bone]["bone"], production_osim_meshes[bone]

    def test_centroid_close(self, bone_comparison):
        """Decoded and production centroids should be within 0.1mm."""
        bone, decoded_mesh, prod_mesh = bone_comparison
        decoded_centroid = decoded_mesh.point_coords.mean(axis=0)
        prod_centroid = prod_mesh.point_coords.mean(axis=0)
        np.testing.assert_allclose(
            decoded_centroid,
            prod_centroid,
            atol=0.0001,  # 0.1mm in meters
            err_msg=f"{bone}: centroid mismatch between decode and production",
        )

    def test_extent_close(self, bone_comparison):
        """Bounding box extents should match within 0.5%."""
        bone, decoded_mesh, prod_mesh = bone_comparison
        decoded_extent = decoded_mesh.point_coords.max(axis=0) - decoded_mesh.point_coords.min(
            axis=0
        )
        prod_extent = prod_mesh.point_coords.max(axis=0) - prod_mesh.point_coords.min(axis=0)
        np.testing.assert_allclose(
            decoded_extent,
            prod_extent,
            rtol=0.005,
            err_msg=f"{bone}: bounding box extent mismatch",
        )

    def test_assd_below_threshold(self, bone_comparison):
        """Surface distance (ASSD) should be <0.05mm.

        Same decoder + same latent, only difference is marching cubes stochasticity.
        """
        bone, decoded_mesh, prod_mesh = bone_comparison
        assd = decoded_mesh.get_assd_mesh(prod_mesh)
        assert (
            assd < 0.00005
        ), f"{bone}: ASSD={assd * 1000:.4f}mm between decode and production, expected <0.05mm"

    def test_tibia_distal_to_femur(self, decoded_subject_joint):
        """In OSIM coords, tibia centroid should be distal (lower Y) than femur."""
        fem_y = decoded_subject_joint["femur"]["bone"].point_coords[:, 1].mean()
        tib_y = decoded_subject_joint["tibia"]["bone"].point_coords[:, 1].mean()
        assert tib_y < fem_y, f"Tibia centroid Y={tib_y:.4f} not distal to femur Y={fem_y:.4f}"

    def test_patella_anterior_to_femur(self, decoded_subject_joint):
        """In OSIM coords, patella centroid should be anterior (positive X) relative to femur."""
        fem_x = decoded_subject_joint["femur"]["bone"].point_coords[:, 0].mean()
        pat_x = decoded_subject_joint["patella"]["bone"].point_coords[:, 0].mean()
        assert pat_x > fem_x, f"Patella centroid X={pat_x:.4f} not anterior to femur X={fem_x:.4f}"


# ---------------------------------------------------------------------------
# Smoke tests with zero latent
# ---------------------------------------------------------------------------


@requires_gpu
@requires_nsm_models
class TestDecodeZeroLatent:
    """Smoke test: decode a zero latent vector (mean shape)."""

    @pytest.fixture(scope="class", params=BONES)
    def decoded_zero(self, request, loaded_models, fem_ref_center):
        from nsosim.decode import decode_latent_to_osim

        bone = request.param
        models, configs = loaded_models
        align = _load_ref_alignment(bone)
        T = np.array(align["transform_matrix"])
        # For zero-latent test, use each bone's own mean_orig
        ref_center = np.array(align["mean_orig"])

        latent_dim = configs[bone].get("latent_size", 512)
        zero_latent = np.zeros(latent_dim)

        result = decode_latent_to_osim(
            latent_vector=zero_latent,
            model=models[bone],
            linear_transform=T,
            fem_ref_center=ref_center,
            model_config=configs[bone],
            n_pts_per_axis=256,
        )
        return bone, result

    def test_has_bone_mesh(self, decoded_zero):
        bone_name, result = decoded_zero
        assert "bone" in result, f"{bone_name} zero-latent decode missing 'bone'"

    def test_bone_has_points(self, decoded_zero):
        bone_name, result = decoded_zero
        pts = result["bone"].point_coords
        assert pts.shape[0] > 50, f"{bone_name} zero-latent bone has too few points"

    def test_bone_in_plausible_range(self, decoded_zero):
        bone_name, result = decoded_zero
        pts = result["bone"].point_coords
        assert np.all(np.abs(pts) < 0.5), f"{bone_name} zero-latent outside plausible range"
