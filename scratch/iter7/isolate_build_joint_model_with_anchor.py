#!/usr/bin/env python
"""Iter7 isolation test: same as iter3's isolate_build_joint_model.py BUT with
the new ``smith2019_osim_path`` config key set so build_joint_model wires
Procrustes-from-Smith2019 anchors into each wrap fit.

Comparison gates (plan rev 2, NOTES.md):
  1. All 10 wraps ≤ 0.10 mm A vs B (criterion 1)
  2. No regression on axis-aligned wraps (criterion 2)
  3. Capsule_r center within 0.5 mm of Smith2019 anchor (criterion 3, new gate)
  4. Determinism preserved: A_v1 vs A_v2 ≤ 1 differing line (criterion 4)

Usage
-----
This is a thin shim around the upstream
``comak_gait_simulation/tests/swap_experiments/isolate_build_joint_model.py``
that re-uses its helpers but adds the smith2019_osim_path config key. Run via
the iter7 SLURM submission script (``submit_iter.sh``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Re-use the existing upstream isolation harness for everything except the
# build_joint_model call itself.
_UPSTREAM_DIR = Path(
    "/dataNAS/people/aagatti/projects/comak_gait_simulation/tests/swap_experiments"
)
sys.path.insert(0, str(_UPSTREAM_DIR))
_VERIFY_MB_DIR = _UPSTREAM_DIR.parent / "verify_model_building"
sys.path.insert(0, str(_VERIFY_MB_DIR))

from isolate_build_joint_model import compare_osim_outputs  # noqa: E402


def print_report(label, r):
    """Inlined from upstream isolate_build_joint_model.main()."""
    print(f"\n--- {label} ---")
    if r.get("status") != "OK":
        print(f"  STATUS: {r.get('status')}")
        return
    print(f"  Total differing lines: {r['n_diff_lines']} / {r['n_total_lines']}")
    for owner_type, info in sorted(
        r["by_owner_type"].items(), key=lambda x: -x[1]["max_abs"]
    ):
        print(
            f"  {owner_type:<30}  count={info['count']:>3}  "
            f"max_abs={info['max_abs']:.4g}  max_rel={info['max_rel']:.2%}"
        )
from verify_model_building import (  # noqa: E402
    BASE_COMAK_SIMULATION_PARMS_FOLDER,
    build_dict_bones_from_production,
    load_production_meshes,
)


SMITH2019_OSIM = (
    "/dataNAS/people/aagatti/projects/comak_gait_simulation/COMAK_SIMULATION_REQUIREMENTS/"
    "data/reference_data/comak_models/smith2019/smith2019.osim"
)


def run_build_joint_model(run_dir: Path, subject_name: str, output_dir: Path, seed: int = 0) -> Path:
    """Run build_joint_model with Procrustes anchors from Smith2019."""
    from nsosim.model_building import build_joint_model

    output_dir.mkdir(parents=True, exist_ok=True)
    production_base = run_dir / subject_name
    production_geom_dir = production_base / "geometries_nsm_similarity"

    print(f"  Loading bone meshes from {production_geom_dir}")
    bone_meshes = load_production_meshes(str(production_geom_dir))
    dict_bones = build_dict_bones_from_production(str(production_geom_dir))

    lig_attach_path = os.path.join(
        BASE_COMAK_SIMULATION_PARMS_FOLDER,
        "ligament_tendon_attachment",
        "ligament_tendon_attachment_meniscus_new.json",
    )
    with open(lig_attach_path) as f:
        dict_lig_musc_attach_params = json.load(f)

    fem_transform_path = os.path.join(
        BASE_COMAK_SIMULATION_PARMS_FOLDER, "nsm_meshes", "femur", "ref_femur_alignment.json"
    )
    with open(fem_transform_path) as f:
        fem_ref_center = np.array(json.load(f)["mean_orig"])

    folder_ref_recons = os.path.join(BASE_COMAK_SIMULATION_PARMS_FOLDER, "nsm_meshes")
    path_base_osim = os.path.join(
        BASE_COMAK_SIMULATION_PARMS_FOLDER, "data", "reference_data", "comak_models", "current"
    )

    config = {
        "triangle_density": 3_000_000,
        "lig_normal_shift": 5e-4,
        "fatpad_elastic_modulus": 4e6,
        "fatpad_poissons_ratio": 0.45,
        "fatpad_thickness": 0.01,
        "fatpad_min_proximity": 0.0,
        "fatpad_max_proximity": 0.015,
        "wrap_n_restarts": 1,
        "wrap_jitter_scale": 1e-6,
        # iter7 key change: wire Procrustes-from-Smith2019 anchors.
        "smith2019_osim_path": SMITH2019_OSIM,
    }

    model_name = f"isolation_test_{output_dir.name}"
    save_dir = output_dir / "custom_nsm_full_body_healthy_knee_model"

    print(f"  Running build_joint_model → {save_dir}/{model_name}")
    path_save_model = build_joint_model(
        bone_meshes={
            "femur": bone_meshes["femur"],
            "tibia": bone_meshes["tibia"],
            "patella": bone_meshes["patella"],
        },
        dict_bones=dict_bones,
        ref_data_paths={"folder_ref_recons": folder_ref_recons},
        dict_lig_musc_attach_params=dict_lig_musc_attach_params,
        fem_ref_center=fem_ref_center,
        save_dir=str(save_dir),
        folder_save_bones=str(output_dir / "bones"),
        model_name=model_name,
        path_base_osim_model=path_base_osim,
        config=config,
        project_meniscal_to_tibia=False,
        project_coronary=False,
        seed=seed,
    )
    return Path(path_save_model)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a-run", required=True)
    p.add_argument("--b-run", required=True)
    p.add_argument("--output-root", required=True)
    args = p.parse_args()

    a_run = Path(args.a_run)
    b_run = Path(args.b_run)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Discover subject name from A's run dir
    subj_dirs = [d for d in a_run.iterdir() if d.is_dir()]
    if not subj_dirs:
        raise RuntimeError(f"No subject directory under {a_run}")
    subj_name = subj_dirs[0].name
    print(f"Subject: {subj_name}\n")

    print("Run 1: build_joint_model(A's meshes, seed=0)")
    osim_a_v1 = run_build_joint_model(a_run, subj_name, out_root / "A_v1", seed=0)
    print(f"  → {osim_a_v1}\n")

    print("Run 2: build_joint_model(A's meshes, seed=0) — repeat")
    osim_a_v2 = run_build_joint_model(a_run, subj_name, out_root / "A_v2", seed=0)
    print(f"  → {osim_a_v2}\n")

    print("Run 3: build_joint_model(B's meshes, seed=0)")
    osim_b = run_build_joint_model(b_run, subj_name, out_root / "B", seed=0)
    print(f"  → {osim_b}\n")

    det = compare_osim_outputs("A_v1 vs A_v2", osim_a_v1, osim_a_v2)
    amp = compare_osim_outputs("A_v1 vs B   ", osim_a_v1, osim_b)
    print_report("A_v1 vs A_v2 (build_joint_model deterministic?)", det)
    print_report("A_v1 vs B    (anchor effect on sensitivity)", amp)


if __name__ == "__main__":
    main()
