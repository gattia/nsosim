"""The collapsed .osim must load and initialize in OpenSim 4.5 with JAM."""

import opensim as osim


def test_collapsed_model_loads(collapsed_osim):
    _, collapsed_path, _ = collapsed_osim
    model = osim.Model(str(collapsed_path))
    assert model.getName() != ""


def test_collapsed_model_initializes(collapsed_osim):
    """initSystem() must succeed -- exercises JAM contact meshes + wraps."""
    _, collapsed_path, _ = collapsed_osim
    model = osim.Model(str(collapsed_path))
    state = model.initSystem()
    assert state is not None


def test_collapsed_model_realizes_dynamics(collapsed_osim):
    """The collapsed model must realize through to the Dynamics stage."""
    _, collapsed_path, _ = collapsed_osim
    model = osim.Model(str(collapsed_path))
    state = model.initSystem()
    model.realizeAcceleration(state)
    assert model.getBodySet().getSize() > 0
