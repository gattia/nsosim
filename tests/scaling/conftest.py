"""Fixtures for tests/scaling/."""

import os
from pathlib import Path
from typing import Optional

import opensim as osim
import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
BASE_COMAK = FIXTURES / "osim_models" / "full_body_healthy_knee.osim"
# NOTE: despite the name, this is AB's stock ArmlessRajagopal (the generic the
# Paper-1 v2 AB runs scaled), NOT a stripped COMAK model. See fixtures README.
STRIPPED_TIAN = FIXTURES / "osim_models" / "unscaled_generic_tian.osim"
RSUBJECT121_AB = (
    Path(__file__).resolve().parents[2]
    / "untracked"
    / "ab_outputs"
    / "RSubject_121"
    / "subject_data"
    / "Models"
    / "match_markers_and_physics.osim"
)

# A fully-built Mode-1 model (a personalized recon knee swapped into a
# reference-size full-body COMAK model) — the input the "build, then scale"
# route (deviations.md Mode 3 / Mode 5) consumes. It is too large for git
# (~51 MB of full-res recon STLs), so it lives untracked and the dependent
# tests skip gracefully when it is absent (same convention as RSUBJECT121_AB
# and the @requires_nsm_models tests). Override with NSOSIM_BUILT_MODEL_OSIM
# to point at any other built model (its Geometry/ must sit beside it).
#
# NOTE (2026-10-03, comak rerun-v3 sub-plan 01): this file's BODY is not
# reference-size. Its skeleton (femur 0.3606, tibia 0.4090, hip half-width
# 0.1069 m) equals the COMAK reference scaled by RSubject_121's v2 AB factors
# (comak audit A11: 0.36064 / 0.40899 / 0.10686), not the reference (0.3742 /
# 0.4033 / 0.0924). So the "subject" fixtures below scale an already-scaled
# body. The knee-coherence assertions compare ratios and still hold. A
# production build from the reference base keeps the reference skeleton
# exactly. nsosim.scaling.validate flags the mismatch, hence the overrides below.
BUILT_MODE1 = (
    Path(__file__).resolve().parents[2]
    / "untracked"
    / "built_models"
    / "mode1_9003175_00m_RIGHT"
    / "built_mode1.osim"
)


def _env_built_model() -> Optional[Path]:
    p = os.environ.get("NSOSIM_BUILT_MODEL_OSIM")
    return Path(p) if p else None


@pytest.fixture(scope="session")
def base_comak_path() -> Path:
    if not BASE_COMAK.is_file():
        pytest.skip(f"Missing base COMAK fixture: {BASE_COMAK}")
    return BASE_COMAK


@pytest.fixture(scope="session")
def stripped_tian_path() -> Path:
    if not STRIPPED_TIAN.is_file():
        pytest.skip(f"Missing stripped Tian base: {STRIPPED_TIAN}")
    return STRIPPED_TIAN


@pytest.fixture(scope="session")
def rsubject121_ab_path() -> Path:
    if not RSUBJECT121_AB.is_file():
        pytest.skip(f"Missing RSubject_121 AB output: {RSUBJECT121_AB}")
    return RSUBJECT121_AB


def place_ab_generic(ab_dir: Path, generic_osim: Path) -> Path:
    """Put ``generic_osim`` beside an AB-shaped model as AB's ``unscaled_generic.osim``.

    ``scale_comak_model`` checks that file against the base model's skeleton. A
    COMAK model only loads with its Geometry/ beside it, so that is linked too.
    """
    target = Path(ab_dir) / "unscaled_generic.osim"
    target.symlink_to(Path(generic_osim).resolve())
    geometry = Path(generic_osim).resolve().parent / "Geometry"
    if geometry.is_dir() and not (Path(ab_dir) / "Geometry").exists():
        (Path(ab_dir) / "Geometry").symlink_to(geometry)
    return target


@pytest.fixture(scope="session")
def identity_ab_osim(stripped_tian_path: Path, base_comak_path: Path, tmp_path_factory) -> Path:
    """Build a synthetic AB-shaped model with every body's scale_factors = (1,1,1).

    Mirrors AB's output convention so scale_comak_model treats it as an
    identity-scale run. The factors are synthetic, so the Rajagopal model they
    are written into does not matter; AB's generic beside it is the COMAK base,
    i.e. this stands for an AB run on the COMAK skeleton.
    """
    out = tmp_path_factory.mktemp("identity_ab") / "match_markers_and_physics.osim"
    place_ab_generic(out.parent, base_comak_path)
    model = osim.Model(str(stripped_tian_path))
    model.initSystem()
    bs = model.getBodySet()
    for i in range(bs.getSize()):
        body = bs.get(i)
        n_geom = body.getPropertyByName("attached_geometry").size()
        if n_geom == 0:
            continue
        body.upd_attached_geometry(0).set_scale_factors(osim.Vec3(1.0, 1.0, 1.0))
    model.finalizeConnections()
    model.printToXML(str(out))
    return out


@pytest.fixture(scope="session")
def synthetic_wa_ab_osim(stripped_tian_path: Path, base_comak_path: Path, tmp_path_factory) -> Path:
    """AB-shaped model with a non-trivial, in-repo-derived knee scale.

    Like ``identity_ab_osim`` but with ``femur_r`` and ``tibia_r`` set to an
    isotropic 0.9 — so ``build_scale_set`` yields ``s_wa = 0.9`` with no
    dependence on the untracked RSubject_121 AB output. Lets the origin-scaling
    placement regression run on the reference base wherever the reference
    Geometry is present. AB's generic beside it is the COMAK base, as for
    ``identity_ab_osim``.
    """
    out = tmp_path_factory.mktemp("synthetic_wa_ab") / "match_markers_and_physics.osim"
    place_ab_generic(out.parent, base_comak_path)
    model = osim.Model(str(stripped_tian_path))
    model.initSystem()
    bs = model.getBodySet()
    for i in range(bs.getSize()):
        body = bs.get(i)
        if body.getPropertyByName("attached_geometry").size() == 0:
            continue
        sf = (0.9, 0.9, 0.9) if body.getName() in ("femur_r", "tibia_r") else (1.0, 1.0, 1.0)
        body.upd_attached_geometry(0).set_scale_factors(osim.Vec3(*sf))
    model.finalizeConnections()
    model.printToXML(str(out))
    return out


def _base_geometry_has_knee_stls(base_osim: Path) -> bool:
    geom = base_osim.parent / "Geometry"
    needed = ("smith2019-R-femur-bone.stl", "smith2019-R-tibia-bone.stl")
    return all((geom / f).is_file() for f in needed)


@pytest.fixture(scope="session")
def synthetic_scaled_reference_model(
    base_comak_path: Path,
    synthetic_wa_ab_osim: Path,
    tmp_path_factory,
) -> Path:
    """Scale the reference base by the synthetic s_wa = 0.9 (no untracked deps)."""
    if not _base_geometry_has_knee_stls(base_comak_path):
        pytest.skip("Reference base Geometry STLs absent — cannot run reference scaling")
    from nsosim.scaling import scale_comak_model

    out_dir = tmp_path_factory.mktemp("synthetic_scaled_ref")
    out_osim = out_dir / "scaled.osim"
    scale_comak_model(
        base_osim=base_comak_path,
        ab_scaled_osim=synthetic_wa_ab_osim,
        output_osim=out_osim,
        output_geometry_dir=out_dir / "Geometry",
        mode="WA",
    )
    return out_osim


@pytest.fixture(scope="session")
def identity_scaled_model(
    base_comak_path: Path,
    identity_ab_osim: Path,
    tmp_path_factory,
) -> Path:
    """Run scale_comak_model with the identity AB fixture once per session."""
    from nsosim.scaling import scale_comak_model

    out_dir = tmp_path_factory.mktemp("identity_scaled")
    out_osim = out_dir / "scaled.osim"
    scale_comak_model(
        base_osim=base_comak_path,
        ab_scaled_osim=identity_ab_osim,
        output_osim=out_osim,
        output_geometry_dir=out_dir / "Geometry",
        mode="WA",
    )
    return out_osim


@pytest.fixture(scope="session")
def subject_scaled_model(
    base_comak_path: Path,
    rsubject121_ab_path: Path,
    tmp_path_factory,
) -> Path:
    """Run scale_comak_model on RSubject_121's AB output once per session.

    That AB run scaled ArmlessRajagopal (Paper-1 v2), so the AB-input check is
    overridden: the legitimate use, reproducing results built on that generic.
    """
    from nsosim.scaling import scale_comak_model

    out_dir = tmp_path_factory.mktemp("subject_scaled")
    out_osim = out_dir / "scaled.osim"
    scale_comak_model(
        base_osim=base_comak_path,
        ab_scaled_osim=rsubject121_ab_path,
        output_osim=out_osim,
        output_geometry_dir=out_dir / "Geometry",
        mode="WA",
        allow_generic_mismatch=True,
    )
    return out_osim


# --- Build-then-scale (Mode 3 / Mode 5) -------------------------------------
# The fixtures above scale the *reference* base (Mode 2). The ones below scale a
# *built* Mode-1 model — the "build, then scale" route that amortizes the GPU
# NSM fit across gait subjects. The only difference is what `base_osim` points
# at; the operation is identical.


@pytest.fixture(scope="session")
def built_mode1_model_path() -> Path:
    """Path to a built Mode-1 model (recon knee in a reference-size body).

    Skips if absent (the model is too large for git). Set
    ``NSOSIM_BUILT_MODEL_OSIM`` to use a different built model.
    """
    cand = _env_built_model() or BUILT_MODE1
    if not cand.is_file():
        pytest.skip(
            f"Missing built Mode-1 model: {cand} " f"(set NSOSIM_BUILT_MODEL_OSIM to override)"
        )
    if not (cand.parent / "Geometry").is_dir():
        pytest.skip(f"Built Mode-1 model has no Geometry/ beside it: {cand}")
    return cand


@pytest.fixture(scope="session")
def built_identity_scaled_model(
    built_mode1_model_path: Path,
    identity_ab_osim: Path,
    tmp_path_factory,
) -> Path:
    """Scale the built Mode-1 model by an identity AB run (s_wa == 1).

    The s_wa = 1 baseline processed through the *same* pipeline as the subject
    case — so per-component comparisons are apples-to-apples. The AB-input check
    is overridden: this built model's body is not the reference skeleton that
    ``identity_ab_osim``'s generic is (see the BUILT_MODE1 note), and identity
    factors do not depend on it.
    """
    from nsosim.scaling import scale_comak_model

    out_dir = tmp_path_factory.mktemp("built_identity_scaled")
    out_osim = out_dir / "scaled.osim"
    scale_comak_model(
        base_osim=built_mode1_model_path,
        ab_scaled_osim=identity_ab_osim,
        output_osim=out_osim,
        output_geometry_dir=out_dir / "Geometry",
        mode="WA",
        allow_generic_mismatch=True,
    )
    return out_osim


@pytest.fixture(scope="session")
def built_subject_scaled_model(
    built_mode1_model_path: Path,
    rsubject121_ab_path: Path,
    tmp_path_factory,
) -> Path:
    """Scale the built Mode-1 model to RSubject_121's gait body (s_wa != 1).

    This is the real Mode-3 scenario: a personalized (OAI) knee scaled to a
    *different* subject's gait body. RSubject_121 gives s_wa ~= 0.973.
    Its AB run scaled ArmlessRajagopal, so the AB-input check is overridden
    (see ``subject_scaled_model``).
    """
    from nsosim.scaling import scale_comak_model

    out_dir = tmp_path_factory.mktemp("built_subject_scaled")
    out_osim = out_dir / "scaled.osim"
    scale_comak_model(
        base_osim=built_mode1_model_path,
        ab_scaled_osim=rsubject121_ab_path,
        output_osim=out_osim,
        output_geometry_dir=out_dir / "Geometry",
        mode="WA",
        allow_generic_mismatch=True,
    )
    return out_osim
