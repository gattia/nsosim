"""Fixtures for tests/scaling/."""

from pathlib import Path

import opensim as osim
import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
BASE_COMAK = FIXTURES / "osim_models" / "full_body_healthy_knee.osim"
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


@pytest.fixture(scope="session")
def identity_ab_osim(stripped_tian_path: Path, tmp_path_factory) -> Path:
    """Build a synthetic AB-shaped model with every body's scale_factors = (1,1,1).

    Mirrors AB's output convention so scale_comak_model treats it as an
    identity-scale run.
    """
    out = tmp_path_factory.mktemp("identity_ab") / "match_markers_and_physics.osim"
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
    """Run scale_comak_model on RSubject_121's AB output once per session."""
    from nsosim.scaling import scale_comak_model

    out_dir = tmp_path_factory.mktemp("subject_scaled")
    out_osim = out_dir / "scaled.osim"
    scale_comak_model(
        base_osim=base_comak_path,
        ab_scaled_osim=rsubject121_ab_path,
        output_osim=out_osim,
        output_geometry_dir=out_dir / "Geometry",
        mode="WA",
    )
    return out_osim
