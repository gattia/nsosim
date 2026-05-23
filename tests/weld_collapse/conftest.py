"""Shared fixtures for the Stage Z weld-collapse test suite."""

from pathlib import Path

import opensim as osim
import pytest

# The base fixture carries both intermediate welds in the right-knee chain.
BASE_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "osim_models" / "full_body_healthy_knee.osim"
)
# Geometry/ lives alongside the fixture model; register it so a collapsed model
# written to a tmp dir still resolves visual + Smith2018ContactMesh meshes.
_GEOMETRY_DIR = BASE_MODEL_PATH.parent / "Geometry"


@pytest.fixture(scope="session", autouse=True)
def _register_geometry_search_path():
    """Register the fixture Geometry/ dir on OpenSim's global search path."""
    osim.ModelVisualizer.addDirToGeometrySearchPaths(str(_GEOMETRY_DIR.resolve()))


@pytest.fixture(scope="module")
def base_model():
    """A freshly loaded + initialized copy of the base (welded) model."""
    model = osim.Model(str(BASE_MODEL_PATH))
    model.initSystem()
    return model


@pytest.fixture(scope="session")
def collapsed_osim(tmp_path_factory):
    """Run Stage Z once; return (input_path, collapsed_path, report)."""
    from nsosim.weld_collapse import collapse_welds

    out = tmp_path_factory.mktemp("weld_collapse") / "collapsed.osim"
    report = collapse_welds(
        BASE_MODEL_PATH,
        out,
        report_json=out.with_suffix(".report.json"),
    )
    return BASE_MODEL_PATH, out, report


@pytest.fixture(scope="module")
def models(collapsed_osim):
    """(input_model, collapsed_model), both loaded + initialized."""
    base_path, collapsed_path, _ = collapsed_osim
    m_in = osim.Model(str(base_path))
    m_in.initSystem()
    m_out = osim.Model(str(collapsed_path))
    m_out.initSystem()
    return m_in, m_out
