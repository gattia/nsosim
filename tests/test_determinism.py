"""Determinism tests for NSM fitting and decode pipelines.

Verifies that, with a pinned seed, two runs of the same public API on the
same inputs produce bit-identical outputs.

The unit-level test of ``set_global_seed`` runs without GPU. The integration
tests of ``decode_latent_to_osim`` and ``decode_joint_from_descriptors``
require a CUDA GPU and the NSM model weights.
"""

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TRANSFORMS_DIR = FIXTURES_DIR / "transforms"
MODELS_DIR = FIXTURES_DIR / "models"
BONES = ["femur", "tibia", "patella"]

try:
    import torch

    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False

_MODELS_AVAILABLE = all((MODELS_DIR / bone / "model.pth").exists() for bone in BONES)
_MESH_FIXTURES_AVAILABLE = (TRANSFORMS_DIR / "nsm_recon_ref_femur_osim_space.vtk").exists()

requires_gpu = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
requires_nsm_models = pytest.mark.skipif(
    not _MODELS_AVAILABLE,
    reason="NSM model weights not in tests/fixtures/models/",
)
requires_mesh_fixtures = pytest.mark.skipif(
    not _MESH_FIXTURES_AVAILABLE,
    reason="Mesh fixtures not available. Run tests/fixtures/transforms/download_fixtures.sh",
)


# ---------------------------------------------------------------------------
# Unit test: set_global_seed itself
# ---------------------------------------------------------------------------


class TestSetGlobalSeed:
    """set_global_seed pins NumPy/torch/Python RNGs to a known state."""

    def test_numpy_reproducible(self):
        from nsosim._determinism import set_global_seed

        set_global_seed(123)
        a = np.random.rand(10)
        set_global_seed(123)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_torch_cpu_reproducible(self):
        from nsosim._determinism import set_global_seed

        set_global_seed(7)
        a = torch.randn(20)
        set_global_seed(7)
        b = torch.randn(20)
        assert torch.equal(a, b)

    def test_python_random_reproducible(self):
        import random

        from nsosim._determinism import set_global_seed

        set_global_seed(42)
        a = [random.random() for _ in range(10)]
        set_global_seed(42)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_cudnn_flags_set(self):
        from nsosim._determinism import set_global_seed

        set_global_seed(0)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_idempotent(self):
        """Calling twice with the same seed gives the same RNG state."""
        from nsosim._determinism import set_global_seed

        set_global_seed(99)
        set_global_seed(99)
        a = np.random.rand(5)
        set_global_seed(99)
        set_global_seed(99)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Helpers for GPU integration tests
# ---------------------------------------------------------------------------


def _load_model_config(bone):
    config_path = MODELS_DIR / bone / "model_params_config.json"
    with open(config_path) as f:
        return json.load(f)


def _load_ref_alignment(bone):
    path = TRANSFORMS_DIR / f"ref_{bone}_alignment.json"
    with open(path) as f:
        return json.load(f)


def _mesh_arrays_equal(m1, m2):
    """Compare meshes by point coordinates exactly. pymskt.Mesh wraps
    pyvista.PolyData, so ``.points`` works for both."""
    p1 = np.asarray(m1.points)
    p2 = np.asarray(m2.points)
    if p1.shape != p2.shape:
        return False, f"shape mismatch {p1.shape} vs {p2.shape}"
    if not np.array_equal(p1, p2):
        max_diff = float(np.max(np.abs(p1 - p2)))
        return False, f"max abs diff = {max_diff}"
    return True, "equal"


# ---------------------------------------------------------------------------
# GPU tests: decode_latent_to_osim
# ---------------------------------------------------------------------------


@requires_gpu
@requires_nsm_models
@requires_mesh_fixtures
class TestDecodeLatentToOsimDeterminism:
    """Two calls with the same seed produce bit-identical meshes."""

    @pytest.fixture(scope="class")
    def femur_inputs(self):
        from nsosim.utils import load_model

        config = _load_model_config("femur")
        model = load_model(
            config,
            str(MODELS_DIR / "femur" / "model.pth"),
            model_type="triplanar",
        )
        align = _load_ref_alignment("femur")
        T = np.array(align["transform_matrix"])
        fem_ref_center = np.array(align["mean_orig"])
        latent = np.load(TRANSFORMS_DIR / "latent_femur.npy")
        return model, config, T, fem_ref_center, latent

    def test_same_seed_bit_identical(self, femur_inputs):
        from nsosim.decode import decode_latent_to_osim

        model, config, T, center, latent = femur_inputs

        out_a = decode_latent_to_osim(
            latent_vector=latent,
            model=model,
            linear_transform=T,
            fem_ref_center=center,
            model_config=config,
            n_pts_per_axis=128,
            seed=42,
        )
        out_b = decode_latent_to_osim(
            latent_vector=latent,
            model=model,
            linear_transform=T,
            fem_ref_center=center,
            model_config=config,
            n_pts_per_axis=128,
            seed=42,
        )

        assert set(out_a.keys()) == set(out_b.keys())
        for name in out_a:
            ok, msg = _mesh_arrays_equal(out_a[name], out_b[name])
            assert ok, f"{name} not bit-identical: {msg}"

    def test_resampled_outputs_bit_identical(self, femur_inputs):
        """ACVD resampling is sensitive to upstream noise — confirm it's
        deterministic too once the seed is pinned."""
        from nsosim.decode import decode_latent_to_osim

        model, config, T, center, latent = femur_inputs
        clusters = {"bone": 5_000}

        out_a = decode_latent_to_osim(
            latent_vector=latent,
            model=model,
            linear_transform=T,
            fem_ref_center=center,
            model_config=config,
            n_pts_per_axis=128,
            clusters=clusters,
            seed=42,
        )
        out_b = decode_latent_to_osim(
            latent_vector=latent,
            model=model,
            linear_transform=T,
            fem_ref_center=center,
            model_config=config,
            n_pts_per_axis=128,
            clusters=clusters,
            seed=42,
        )

        ok, msg = _mesh_arrays_equal(out_a["bone"], out_b["bone"])
        assert ok, f"resampled bone not bit-identical: {msg}"


# ---------------------------------------------------------------------------
# GPU tests: decode_joint_from_descriptors (full joint)
# ---------------------------------------------------------------------------


@requires_gpu
@requires_nsm_models
@requires_mesh_fixtures
class TestDecodeJointDeterminism:
    """Multi-bone joint decode is bit-identical with same seed."""

    @pytest.fixture(scope="class")
    def joint_inputs(self):
        from nsosim.transforms import compute_T_rel
        from nsosim.utils import load_model

        models = {}
        configs = {}
        transforms = {}
        latents = {}
        for bone in BONES:
            cfg = _load_model_config(bone)
            models[bone] = load_model(
                cfg,
                str(MODELS_DIR / bone / "model.pth"),
                model_type="triplanar",
            )
            configs[bone] = cfg
            align = _load_ref_alignment(bone)
            transforms[bone] = np.array(align["transform_matrix"])
            latents[bone] = np.load(TRANSFORMS_DIR / f"latent_{bone}.npy")

        fem_ref_center = np.array(_load_ref_alignment("femur")["mean_orig"])
        T_rel_tib = compute_T_rel(transforms["femur"], transforms["tibia"])
        T_rel_pat = compute_T_rel(transforms["femur"], transforms["patella"])

        return {
            "models": models,
            "configs": configs,
            "T_fem": transforms["femur"],
            "T_rel_tib": T_rel_tib,
            "T_rel_pat": T_rel_pat,
            "fem_ref_center": fem_ref_center,
            "latents": latents,
        }

    def test_same_seed_bit_identical(self, joint_inputs):
        from nsosim.decode import decode_joint_from_descriptors

        kwargs = dict(
            femur_latent=joint_inputs["latents"]["femur"],
            tibia_latent=joint_inputs["latents"]["tibia"],
            patella_latent=joint_inputs["latents"]["patella"],
            T_fem=joint_inputs["T_fem"],
            T_rel_tib=joint_inputs["T_rel_tib"],
            T_rel_pat=joint_inputs["T_rel_pat"],
            models=joint_inputs["models"],
            model_configs=joint_inputs["configs"],
            fem_ref_center=joint_inputs["fem_ref_center"],
            n_pts_per_axis=128,
            seed=42,
        )

        out_a = decode_joint_from_descriptors(**kwargs)
        out_b = decode_joint_from_descriptors(**kwargs)

        for bone in BONES:
            assert set(out_a[bone].keys()) == set(out_b[bone].keys())
            for name in out_a[bone]:
                ok, msg = _mesh_arrays_equal(out_a[bone][name], out_b[bone][name])
                assert ok, f"{bone}.{name} not bit-identical: {msg}"


# ---------------------------------------------------------------------------
# Post-processing determinism — these are the steps the plan flagged as the
# big stochasticity drivers: meniscus articular surface extraction,
# articular surface extraction, fat pad, wrap surface fitting.
#
# The strategy: decode meshes deterministically (already verified above),
# then run each post-processing function twice on identical copies of the
# decoded meshes with the seed reset before each call. If outputs are
# bit-identical, the post-processing step itself is deterministic given
# pinned RNG state.
# ---------------------------------------------------------------------------


@requires_gpu
@requires_nsm_models
@requires_mesh_fixtures
class TestPostProcessingDeterminism:
    """Build-pipeline post-processing must produce bit-identical output
    across two runs with the same seed.

    The plan calls out ``create_meniscus_articulating_surface`` as the
    persistent verification failure (322 vs 437–465 triangles, ASSD ~0.455
    mm). Once the inputs are deterministic and the seed is pinned, this
    should drop to bit-identical."""

    @pytest.fixture(scope="class")
    def decoded_meshes(self):
        """Decode all bones once; return two independent deep copies so each
        test can mutate freely without affecting the other."""
        import copy

        from nsosim.decode import decode_joint_from_descriptors
        from nsosim.transforms import compute_T_rel
        from nsosim.utils import load_model

        models = {}
        configs = {}
        transforms = {}
        latents = {}
        for bone in BONES:
            cfg = _load_model_config(bone)
            models[bone] = load_model(
                cfg, str(MODELS_DIR / bone / "model.pth"), model_type="triplanar"
            )
            configs[bone] = cfg
            align = _load_ref_alignment(bone)
            transforms[bone] = np.array(align["transform_matrix"])
            latents[bone] = np.load(TRANSFORMS_DIR / f"latent_{bone}.npy")
        fem_ref_center = np.array(_load_ref_alignment("femur")["mean_orig"])

        decoded = decode_joint_from_descriptors(
            femur_latent=latents["femur"],
            tibia_latent=latents["tibia"],
            patella_latent=latents["patella"],
            T_fem=transforms["femur"],
            T_rel_tib=compute_T_rel(transforms["femur"], transforms["tibia"]),
            T_rel_pat=compute_T_rel(transforms["femur"], transforms["patella"]),
            models=models,
            model_configs=configs,
            fem_ref_center=fem_ref_center,
            n_pts_per_axis=128,
            seed=42,
        )
        return copy.deepcopy(decoded), copy.deepcopy(decoded)

    def test_create_articular_surfaces_bit_identical(self, decoded_meshes):
        """Cartilage articular surface extraction is deterministic."""
        from nsosim._determinism import set_global_seed
        from nsosim.articular_surfaces import create_articular_surfaces

        a, b = decoded_meshes

        set_global_seed(42)
        art_a = create_articular_surfaces(
            a["femur"]["bone"], a["femur"]["cart"], n_largest=1
        )

        set_global_seed(42)
        art_b = create_articular_surfaces(
            b["femur"]["bone"], b["femur"]["cart"], n_largest=1
        )

        ok, msg = _mesh_arrays_equal(art_a, art_b)
        assert ok, f"femur articular surface not bit-identical: {msg}"

    def test_create_meniscus_articulating_surface_bit_identical(self, decoded_meshes):
        """Meniscus articulating surface extraction — the key failing step
        from the plan. Runs medial meniscus (theta_offset=pi)."""
        from nsosim._determinism import set_global_seed
        from nsosim.articular_surfaces import create_meniscus_articulating_surface

        a, b = decoded_meshes

        set_global_seed(42)
        upper_a, lower_a = create_meniscus_articulating_surface(
            meniscus_mesh=a["femur"]["med_men"],
            upper_articulating_bone_mesh=a["femur"]["bone"],
            lower_articulating_bone_mesh=a["tibia"]["bone"],
            theta_offset=np.pi,
            ray_length=15.0,
            n_largest=1,
        )

        set_global_seed(42)
        upper_b, lower_b = create_meniscus_articulating_surface(
            meniscus_mesh=b["femur"]["med_men"],
            upper_articulating_bone_mesh=b["femur"]["bone"],
            lower_articulating_bone_mesh=b["tibia"]["bone"],
            theta_offset=np.pi,
            ray_length=15.0,
            n_largest=1,
        )

        ok, msg = _mesh_arrays_equal(upper_a, upper_b)
        assert ok, f"medial meniscus upper surface not bit-identical: {msg}"

        ok, msg = _mesh_arrays_equal(lower_a, lower_b)
        assert ok, f"medial meniscus LOWER surface not bit-identical: {msg}"

    def test_create_meniscus_lateral_bit_identical(self, decoded_meshes):
        """Lateral meniscus uses theta_offset=0 — separate code path
        (different polar discontinuity) so verify it independently."""
        from nsosim._determinism import set_global_seed
        from nsosim.articular_surfaces import create_meniscus_articulating_surface

        a, b = decoded_meshes

        set_global_seed(42)
        upper_a, lower_a = create_meniscus_articulating_surface(
            meniscus_mesh=a["femur"]["lat_men"],
            upper_articulating_bone_mesh=a["femur"]["bone"],
            lower_articulating_bone_mesh=a["tibia"]["bone"],
            theta_offset=0.0,
            ray_length=15.0,
            n_largest=1,
        )

        set_global_seed(42)
        upper_b, lower_b = create_meniscus_articulating_surface(
            meniscus_mesh=b["femur"]["lat_men"],
            upper_articulating_bone_mesh=b["femur"]["bone"],
            lower_articulating_bone_mesh=b["tibia"]["bone"],
            theta_offset=0.0,
            ray_length=15.0,
            n_largest=1,
        )

        ok, msg = _mesh_arrays_equal(upper_a, upper_b)
        assert ok, f"lateral meniscus upper surface not bit-identical: {msg}"

        ok, msg = _mesh_arrays_equal(lower_a, lower_b)
        assert ok, f"lateral meniscus lower surface not bit-identical: {msg}"

    def test_prefemoral_fatpad_bit_identical(self, decoded_meshes):
        """Fat pad creation uses ray-casting + dilation + ACVD resampling."""
        from nsosim._determinism import set_global_seed
        from nsosim.articular_surfaces import create_prefemoral_fatpad_noboolean

        a, b = decoded_meshes

        set_global_seed(42)
        fp_a = create_prefemoral_fatpad_noboolean(
            femur_bone_mesh=a["femur"]["bone"],
            femur_cart_mesh=a["femur"]["cart"],
            patella_bone_mesh=a["patella"]["bone"],
            patella_cart_mesh=a["patella"]["cart"],
            units="m",
        )

        set_global_seed(42)
        fp_b = create_prefemoral_fatpad_noboolean(
            femur_bone_mesh=b["femur"]["bone"],
            femur_cart_mesh=b["femur"]["cart"],
            patella_bone_mesh=b["patella"]["bone"],
            patella_cart_mesh=b["patella"]["cart"],
            units="m",
        )

        ok, msg = _mesh_arrays_equal(fp_a, fp_b)
        assert ok, f"prefemoral fat pad not bit-identical: {msg}"
