"""Phase 6B: Run settle + forward sim on Rajagopal+COMAK model.

Runs the same protocol as test_knee_assembly_forsim.py:
1. Settle: 2-frame ForsimTool to find COMAK equilibrium
2. Forward: 6-frame ForsimTool with loaded knee flexion

Also runs on the original Smith2019 model for comparison.

Results are saved to scripts/phase6_rajagopal_audit/forsim_results/ for
review and download.

Usage:
    conda run -n comak python scripts/phase6_rajagopal_audit/run_forsim_rajagopal.py
"""

import logging
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)

import opensim as osim

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAJAGOPAL_COMAK = os.path.join(
    REPO_ROOT,
    "tests/fixtures/osim_models/rajagopal/RajagopalLaiUhlrich2023_comak_knee.osim",
)
SMITH2019_MODEL = os.path.join(REPO_ROOT, "tests/fixtures/osim_models/full_body_healthy_knee.osim")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "forsim_results")

# COMAK secondary coordinates
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


def run_forsim(model_path, results_dir, duration=0.05):
    """Run ForsimTool with COMAK unconstrained coordinates."""
    dt = 0.01
    time = np.arange(0, duration + dt / 2, dt)

    knee_flex_deg = np.linspace(0, 10, len(time))
    kinematics = {
        "time": time,
        "knee_flex_r": knee_flex_deg,  # ForsimTool expects degrees
        "pelvis_tilt": np.ones(len(time)) * 90,
    }

    muscles = {"time": time}
    quad_activation = np.linspace(0.1, 0.3, len(time))
    ham_activation = np.ones(len(time)) * 0.15
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


def run_settle(model_path, results_dir):
    """Run a settle-only ForsimTool: no flexion, just find COMAK equilibrium."""
    dt = 0.01
    time = np.array([0.0, dt])

    # No flexion — hold at 0°. Just let COMAK find contact/ligament equilibrium.
    # No pelvis_tilt prescription — leave at model default (standing).
    kinematics = {
        "time": time,
        "knee_flex_r": np.zeros(len(time)),
    }

    # No muscle activation during settle — just passive forces
    muscles = {"time": time}

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


def settle_model(model_path, results_dir):
    """Run a settle sim, update COMAK secondary DOF defaults, save settled model.

    Only updates coordinates that are COMAK unconstrained DOFs — does NOT touch
    pelvis, hip, ankle, or other non-COMAK coordinates.
    """
    states = run_settle(model_path, results_dir)

    if states is None or states.getNumRows() == 0:
        logger.warning("Settle failed — using unsettled model")
        return model_path

    # Only update COMAK secondary DOF defaults, not pelvis/hip/etc.
    comak_coord_names = set()
    for coord_path in UNCONSTRAINED_COORDINATES:
        # Extract coord name from path like "/jointset/knee_r/knee_add_r"
        comak_coord_names.add(coord_path.rsplit("/", 1)[-1])

    model = osim.Model(str(model_path))
    coord_set = model.getCoordinateSet()
    labels = states.getColumnLabels()
    last_row = states.getNumRows() - 1

    for j in range(len(labels)):
        col_name = labels[j]
        if "/value" not in col_name:
            continue
        coord_name = col_name.split("/value")[0].rsplit("/", 1)[-1]

        if coord_name not in comak_coord_names:
            continue

        value = states.getDependentColumnAtIndex(j)[last_row]

        for ci in range(coord_set.getSize()):
            coord = coord_set.get(ci)
            if coord.getName() == coord_name:
                value = max(coord.getRangeMin(), min(coord.getRangeMax(), value))
                coord.setDefaultValue(value)
                break

    model_dir = os.path.dirname(str(model_path))
    settled_name = os.path.basename(str(model_path)).replace(".osim", "_settled.osim")
    settled_path = os.path.join(model_dir, settled_name)
    model.printToXML(settled_path)
    logger.info(f"Settled model saved: {settled_path}")
    return settled_path


def table_to_dict(table):
    result = {}
    n_rows = table.getNumRows()
    labels = table.getColumnLabels()
    result["time"] = np.array([table.getIndependentColumn()[i] for i in range(n_rows)])
    for j in range(len(labels)):
        result[labels[j]] = np.array([table.getDependentColumnAtIndex(j)[i] for i in range(n_rows)])
    return result


def compare_results(smith_states, raj_states, output_path):
    """Compare coordinate trajectories and save summary."""
    smith = table_to_dict(smith_states)
    raj = table_to_dict(raj_states)

    lines = []
    lines.append("COMAK Forward Sim Comparison: Smith2019 vs Rajagopal+COMAK")
    lines.append("=" * 70)
    lines.append(f"Smith2019 time steps: {len(smith['time'])}")
    lines.append(f"Rajagopal time steps: {len(raj['time'])}")
    lines.append("")

    # Find common value columns
    smith_value_cols = [c for c in smith if "/value" in c and c != "time"]
    raj_value_cols = [c for c in raj if "/value" in c and c != "time"]
    common = sorted(set(smith_value_cols) & set(raj_value_cols))
    smith_only = sorted(set(smith_value_cols) - set(raj_value_cols))
    raj_only = sorted(set(raj_value_cols) - set(smith_value_cols))

    lines.append(f"Common coordinates: {len(common)}")
    lines.append(f"Smith2019 only: {len(smith_only)}")
    lines.append(f"Rajagopal only: {len(raj_only)}")

    if smith_only:
        lines.append(f"\nSmith2019-only coordinates:")
        for c in smith_only:
            lines.append(f"  {c}")

    if raj_only:
        lines.append(f"\nRajagopal-only coordinates:")
        for c in raj_only:
            lines.append(f"  {c}")

    lines.append(f"\n{'Coordinate':<60s} {'Max Diff':>10s} {'Max Deg':>10s}")
    lines.append("-" * 82)

    diffs = []
    for col in common:
        if "/speed" in col:
            continue
        # Use min length in case time steps differ
        n = min(len(smith[col]), len(raj[col]))
        max_diff = np.max(np.abs(smith[col][:n] - raj[col][:n]))
        max_deg = np.degrees(max_diff)
        diffs.append((col, max_diff, max_deg))

    # Sort by max diff descending
    diffs.sort(key=lambda x: x[1], reverse=True)
    for col, max_diff, max_deg in diffs:
        lines.append(f"{col:<60s} {max_diff:10.6f} {max_deg:10.4f}")

    summary = "\n".join(lines)
    print(summary)

    with open(output_path, "w") as f:
        f.write(summary)
    logger.info(f"Comparison saved: {output_path}")


def main():
    print("Phase 6B: COMAK Forward Sim — Smith2019 vs Rajagopal+COMAK")
    print("=" * 70)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Smith2019 ---
    logger.info("=== Smith2019: Settle ===")
    smith_settle_dir = os.path.join(RESULTS_DIR, "smith2019_settle")
    settled_smith = settle_model(SMITH2019_MODEL, smith_settle_dir)

    logger.info("=== Smith2019: Forward sim ===")
    smith_fwd_dir = os.path.join(RESULTS_DIR, "smith2019_forward")
    smith_states = run_forsim(settled_smith, smith_fwd_dir, duration=0.05)

    # --- Rajagopal+COMAK ---
    logger.info("=== Rajagopal+COMAK: Settle ===")
    raj_settle_dir = os.path.join(RESULTS_DIR, "rajagopal_settle")
    settled_raj = settle_model(RAJAGOPAL_COMAK, raj_settle_dir)

    logger.info("=== Rajagopal+COMAK: Forward sim ===")
    raj_fwd_dir = os.path.join(RESULTS_DIR, "rajagopal_forward")
    raj_states = run_forsim(settled_raj, raj_fwd_dir, duration=0.05)

    # --- Compare ---
    if smith_states is not None and raj_states is not None:
        compare_results(
            smith_states,
            raj_states,
            os.path.join(RESULTS_DIR, "comparison_summary.txt"),
        )
    else:
        if smith_states is None:
            logger.error("Smith2019 forward sim failed")
        if raj_states is None:
            logger.error("Rajagopal forward sim failed")

    # List all output files
    print(f"\n=== Results saved to: {RESULTS_DIR} ===")
    for root, dirs, files in os.walk(RESULTS_DIR):
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(root, f), RESULTS_DIR)
            size = os.path.getsize(os.path.join(root, f))
            print(f"  {rel} ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
