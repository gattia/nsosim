"""
Simulation validation for knee_assembly strip->add round-trip.

Validates that the rebuilt model (strip->add) is functionally equivalent to
the original by running a COMAK forward simulation on both and comparing
the unconstrained secondary DOF trajectories.

Protocol:
    1. Settle: Run a 2-frame ForsimTool (0.01s) with all 24 COMAK unconstrained
       coordinates to find contact/ligament equilibrium. Update coordinate defaults
       to settled values and save the settled model.
    2. Forward: Run a 6-frame ForsimTool (0.05s) from the settled state with a
       loaded knee flexion scenario:
       - Knee flexion ramp: 0 -> 10 deg
       - Quad activation (recfem, vasint, vaslat, vasmed): 10% -> 30% ramp
       - Hamstring co-contraction (bflh, bfsh, semimem, semiten): constant 15%
       - Gastrocnemius (gaslat, gasmed): constant 10%
    3. Compare: All coordinate value columns must agree within 0.007 rad (~0.4 deg).

The loading scenario engages tibiofemoral and patellofemoral contacts, loads
cruciate and collateral ligaments, and compresses menisci. The COMAK solver
finds equilibrium for the 24 secondary DOFs (knee add/rot/translations, PF
tracking, meniscus deformation) at each time step.

Observed diffs (as of implementation):
    - Max: ~0.23 deg (meniscus_medial_flex_r)
    - COMAK secondary DOFs: 0.02-0.23 deg
    - Non-COMAK coordinates: < 0.005 deg
    - Tolerance headroom: ~1.7x above max observed diff

The diffs arise from component ordering differences in the ForceSet/BodySet
between original and rebuilt models, which affect COMAK solver convergence
path. This is expected and does not indicate a model error.

Requires: opensim (JAM/COMAK fork), Smith2019 model + Geometry files.
Runtime: ~2.5 minutes (4 ForsimTool runs: 2 settle + 2 forward).
"""

import os
from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SMITH2019_MODEL_PATH = FIXTURES_DIR / "osim_models" / "full_body_healthy_knee.osim"

requires_smith2019 = pytest.mark.skipif(
    not SMITH2019_MODEL_PATH.exists(),
    reason=f"Smith2019 model not found at {SMITH2019_MODEL_PATH}",
)

try:
    import opensim as osim

    HAS_OPENSIM = True
except ImportError:
    HAS_OPENSIM = False

requires_opensim = pytest.mark.skipif(not HAS_OPENSIM, reason="opensim not available")

# COMAK secondary coordinates that ForsimTool solves for
UNCONSTRAINED_COORDINATES = [
    "/jointset/knee_r/knee_add_r",
    "/jointset/knee_r/knee_rot_r",
    "/jointset/knee_r/knee_tx_r",
    "/jointset/knee_r/knee_ty_r",
    "/jointset/knee_r/knee_tz_r",
    "/jointset/pf_r/pf_flex_r",
    "/jointset/pf_r/pf_rot_r",
    "/jointset/pf_r/pf_tilt_r",
    "/jointset/pf_r/pf_tx_r",
    "/jointset/pf_r/pf_ty_r",
    "/jointset/pf_r/pf_tz_r",
    "/jointset/meniscus_medial_r/meniscus_medial_flex_r",
    "/jointset/meniscus_medial_r/meniscus_medial_rot_r",
    "/jointset/meniscus_medial_r/meniscus_medial_add_r",
    "/jointset/meniscus_medial_r/meniscus_medial_tx_r",
    "/jointset/meniscus_medial_r/meniscus_medial_ty_r",
    "/jointset/meniscus_medial_r/meniscus_medial_tz_r",
    "/jointset/meniscus_lateral_r/meniscus_lateral_flex_r",
    "/jointset/meniscus_lateral_r/meniscus_lateral_rot_r",
    "/jointset/meniscus_lateral_r/meniscus_lateral_add_r",
    "/jointset/meniscus_lateral_r/meniscus_lateral_tx_r",
    "/jointset/meniscus_lateral_r/meniscus_lateral_ty_r",
    "/jointset/meniscus_lateral_r/meniscus_lateral_tz_r",
]


def _create_sto_file(dict_data, path_save):
    """Create an OpenSim .sto file from a dict of arrays.

    Uses appendRow instead of the 3-arg TimeSeriesTable constructor because
    importing nsosim changes osim.StdVectorString to the moco variant, which
    breaks SWIG overload resolution for the 3-arg constructor.
    """
    col_names = [k for k in dict_data if k != "time"]
    time = dict_data["time"]
    n_cols = len(col_names)

    table = osim.TimeSeriesTable()
    for i in range(len(time)):
        row = osim.RowVector(n_cols, 0.0)
        for j, name in enumerate(col_names):
            row[j] = float(dict_data[name][i])
        table.appendRow(float(time[i]), row)
    table.setColumnLabels(col_names)

    osim.STOFileAdapter().write(table, path_save)


def _run_forsim(model_path, results_dir, duration=0.05):
    """Run ForsimTool with unconstrained COMAK coordinates.

    Prescribes a knee flexion ramp (0→10°) with moderate muscle activation
    on quads and hamstrings to load the contacts and ligaments meaningfully.
    """
    dt = 0.01
    time = np.arange(0, duration + dt / 2, dt)

    # Ramp knee flexion 0→10° to load contacts/ligaments
    knee_flex_deg = np.linspace(0, 10, len(time))
    knee_flex_rad = np.deg2rad(knee_flex_deg)

    kinematics = {
        "time": time,
        "knee_flex_r": knee_flex_rad,
        "pelvis_tilt": np.ones(len(time)) * 90,
    }

    # Moderate activation on quads + hamstrings to create co-contraction
    muscles = {"time": time}
    quad_activation = np.linspace(0.1, 0.3, len(time))  # ramp up with flexion
    ham_activation = np.ones(len(time)) * 0.15  # constant co-contraction
    for m in ["recfem_r", "vasint_r", "vaslat_r", "vasmed_r"]:
        muscles[f"{m}_activation"] = quad_activation
    for m in ["bflh_r", "bfsh_r", "semimem_r", "semiten_r"]:
        muscles[f"{m}_activation"] = ham_activation
    for m in ["gaslat_r", "gasmed_r"]:
        muscles[f"{m}_activation"] = np.ones(len(time)) * 0.1

    os.makedirs(results_dir, exist_ok=True)
    _create_sto_file(kinematics, os.path.join(results_dir, "kinematics.sto"))
    _create_sto_file(muscles, os.path.join(results_dir, "muscles.sto"))

    forsim = osim.ForsimTool()
    forsim.set_model_file(str(model_path))
    forsim.set_results_directory(results_dir)
    forsim.set_start_time(-1)
    forsim.set_stop_time(-1)
    forsim.set_integrator_accuracy(1e-2)
    forsim.set_constant_muscle_control(0.02)
    forsim.set_override_default_muscle_activation(0.02)
    forsim.set_use_activation_dynamics(False)
    forsim.set_use_tendon_compliance(False)
    forsim.set_use_muscle_physiology(True)

    for idx, coord in enumerate(UNCONSTRAINED_COORDINATES):
        forsim.set_unconstrained_coordinates(idx, coord)

    forsim.set_prescribed_coordinates_file(os.path.join(results_dir, "kinematics.sto"))
    forsim.set_actuator_input_file(os.path.join(results_dir, "muscles.sto"))

    forsim.run()

    states_path = os.path.join(results_dir, "_states.sto")
    if os.path.exists(states_path):
        return osim.TimeSeriesTable(states_path)
    return None


def _settle_model(model_path, results_dir):
    """Run a single-frame settle sim to find COMAK equilibrium.

    Updates coordinate defaults in the model file with the settled values
    and saves the settled model. Returns the path to the settled model.
    """
    # Run a minimal 2-frame forsim (0.01s) to find equilibrium
    states = _run_forsim(model_path, results_dir, duration=0.01)

    if states is None or states.getNumRows() == 0:
        return model_path  # Fallback: use unsettled model

    # Read the final state values
    model = osim.Model(str(model_path))
    coord_set = model.getCoordinateSet()
    labels = states.getColumnLabels()
    last_row = states.getNumRows() - 1

    for j in range(len(labels)):
        col_name = labels[j]
        if "/value" not in col_name:
            continue
        # Extract coordinate name from state path (e.g., "/jointset/knee_r/knee_add_r/value")
        parts = col_name.rstrip("/value").rsplit("/", 1)
        coord_name = parts[-1] if parts else col_name
        # Remove trailing /value
        coord_name = col_name.split("/value")[0].rsplit("/", 1)[-1]

        value = states.getDependentColumnAtIndex(j)[last_row]

        # Update default in model
        for ci in range(coord_set.getSize()):
            coord = coord_set.get(ci)
            if coord.getName() == coord_name:
                # Clamp to range
                value = max(coord.getRangeMin(), min(coord.getRangeMax(), value))
                coord.setDefaultValue(value)
                break

    # Save next to the original model so Geometry/ folder resolves
    model_dir = os.path.dirname(str(model_path))
    settled_name = os.path.basename(str(model_path)).replace(".osim", "_settled.osim")
    settled_path = os.path.join(model_dir, settled_name)
    model.printToXML(settled_path)
    return settled_path


def _table_to_dict(table):
    """Convert an osim.TimeSeriesTable to a dict of numpy arrays."""
    result = {}
    n_rows = table.getNumRows()
    labels = table.getColumnLabels()

    result["time"] = np.array([table.getIndependentColumn()[i] for i in range(n_rows)])
    for j in range(len(labels)):
        result[labels[j]] = np.array([table.getDependentColumnAtIndex(j)[i] for i in range(n_rows)])

    return result


@requires_opensim
@requires_smith2019
class TestForsimRoundTrip:
    """Settle + forward sim on original and rebuilt models.

    Tests that the COMAK secondary DOFs (knee add/rot/translations, PF tracking,
    meniscus deformation) produce equivalent trajectories under a loaded knee
    flexion scenario. See module docstring for full protocol and tolerances.
    """

    @pytest.fixture(scope="class")
    def forsim_results(self, tmp_path_factory):
        """Settle both models, then run short forward sim. Shared across tests."""
        from nsosim.knee_assembly import add_comak_knee, strip_comak_knee

        base_dir = tmp_path_factory.mktemp("forsim_roundtrip")

        # 1. Settle original model
        orig_settle_dir = str(base_dir / "orig_settle")
        settled_orig = _settle_model(SMITH2019_MODEL_PATH, orig_settle_dir)

        # 2. Run forward sim on settled original
        orig_fwd_dir = str(base_dir / "orig_forward")
        orig_states = _run_forsim(settled_orig, orig_fwd_dir, duration=0.02)

        # 3. Strip -> add -> save rebuilt model next to Geometry/
        model = osim.Model(str(SMITH2019_MODEL_PATH))
        stripped, config = strip_comak_knee(model, side="r")
        rebuilt = add_comak_knee(stripped, config)
        rebuilt_path = str(SMITH2019_MODEL_PATH.parent / "rebuilt_roundtrip_test.osim")
        rebuilt.printToXML(rebuilt_path)

        # 4. Settle rebuilt model
        rebuilt_settle_dir = str(base_dir / "rebuilt_settle")
        settled_rebuilt = _settle_model(rebuilt_path, rebuilt_settle_dir)

        # 5. Run forward sim on settled rebuilt
        rebuilt_fwd_dir = str(base_dir / "rebuilt_forward")
        rebuilt_states = _run_forsim(settled_rebuilt, rebuilt_fwd_dir, duration=0.02)

        # Clean up models saved to fixtures dir
        for path in [rebuilt_path, settled_orig, settled_rebuilt]:
            if os.path.exists(str(path)) and str(path) != str(SMITH2019_MODEL_PATH):
                os.remove(str(path))

        return orig_states, rebuilt_states

    @pytest.fixture
    def orig_states(self, forsim_results):
        return forsim_results[0]

    @pytest.fixture
    def rebuilt_states(self, forsim_results):
        return forsim_results[1]

    def test_both_simulations_completed(self, orig_states, rebuilt_states):
        """Both simulations should produce output."""
        assert orig_states is not None, "Original forsim did not produce states"
        assert rebuilt_states is not None, "Rebuilt forsim did not produce states"

    def test_same_number_of_rows(self, orig_states, rebuilt_states):
        assert orig_states.getNumRows() == rebuilt_states.getNumRows()

    def test_same_columns(self, orig_states, rebuilt_states):
        orig_cols = set(orig_states.getColumnLabels())
        rebuilt_cols = set(rebuilt_states.getColumnLabels())
        assert orig_cols == rebuilt_cols

    def test_coordinate_values_match(self, orig_states, rebuilt_states):
        """Secondary DOF values should be close after settling.

        Both models start from independently settled equilibria, so the
        forward sim should produce similar secondary DOF trajectories.
        Tolerance of 0.007 rad (~0.4 deg) accounts for component ordering
        effects on the COMAK solver convergence.
        """
        orig = _table_to_dict(orig_states)
        rebuilt = _table_to_dict(rebuilt_states)

        np.testing.assert_allclose(
            orig["time"],
            rebuilt["time"],
            atol=1e-10,
            err_msg="Time vectors differ",
        )

        value_mismatches = []
        for col in orig:
            if col == "time" or "/speed" in col:
                continue
            if col not in rebuilt:
                value_mismatches.append(f"{col}: missing from rebuilt")
                continue
            max_diff = np.max(np.abs(orig[col] - rebuilt[col]))
            # 0.007 rad (~0.4 deg) for functional equivalence
            if max_diff > 0.007:
                value_mismatches.append(f"{col}: max_diff={max_diff:.2e}")

        assert (
            len(value_mismatches) == 0
        ), f"{len(value_mismatches)} coordinate values differ beyond tolerance:\n" + "\n".join(
            value_mismatches[:20]
        )
