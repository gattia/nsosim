#!/usr/bin/env python
"""Fit-quality check: did iter7 (Procrustes-anchored) preserve the wrap fit
quality of iter3 (algebraic-init)?

Metric: at each labeled bone vertex, evaluate the SDF of the fitted wrap
surface (in the wrap's parent-body frame, after correcting for the
ADDITIONAL_OFFSETS shift). Threshold at 0 to predict inside/outside,
compare to the Smith2019-derived binary label. Report:

  - classification accuracy (correct predictions / total)
  - margin-violation residual (sum of |SDF| where prediction is wrong) in mm
  - mean |SDF| within the labelled "inside" region (closeness of fit
    boundary to the labelled boundary)

A faithful fit has accuracy ~1.0 and small margin violations. If iter7's
accuracy is materially lower than iter3's, the Smith2019 anchor pulled the
fit away from the subject-bone's data optimum — bad. If they're comparable,
iter7 preserved fit quality while improving reproducibility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from pymskt.mesh import Mesh
from scipy.spatial.transform import Rotation as ScipyR

from nsosim.wrap_surface_fitting.fitting import sd_cylinder_with_axis
from nsosim.wrap_surface_fitting.parameter_extraction import (
    extract_wrap_parameters_from_osim,
)
from nsosim.wrap_surface_fitting.utils import ADDITIONAL_OFFSETS
from nsosim.wrap_surface_fitting.wrap_signed_distances import sd_ellipsoid_improved


# Map osim bone names → labeled-mesh path inside an iter SLURM output
BONE_PATHS = {
    "femur": "A_v1/bones/femur/femur_labeled_mesh_updated.vtk",
    "tibia": "A_v1/bones/tibia/tibia_labeled_mesh_updated.vtk",
    "patella": "A_v1/bones/patella/patella_labeled_mesh_updated.vtk",
}

# Wrap → parent body (from DEFAULT_SMITH2019_BONES)
WRAP_TO_BODY = {
    "Gastroc_at_Condyles_r": "femur_r",
    "KnExt_at_fem_r": "femur_r",
    "KnExt_vasint_at_fem_r": "femur_r",
    "Capsule_r": "femur_distal_r",
    "Med_Lig_r": "tibia_proximal_r",
    "Med_LigP_r": "tibia_proximal_r",
    "PatTen_r": "patella_r",
}

# Wrap → bone the labeled mesh lives on
WRAP_TO_BONE = {
    "Gastroc_at_Condyles_r": "femur",
    "KnExt_at_fem_r": "femur",
    "KnExt_vasint_at_fem_r": "femur",
    "Capsule_r": "femur",
    "Med_Lig_r": "tibia",
    "Med_LigP_r": "tibia",
    "PatTen_r": "patella",
}


def sdf_at_points_in_body_frame(wrap_params: dict, points_body: np.ndarray) -> np.ndarray:
    """Compute the fitted wrap's SDF at bone-vertex positions in the wrap's
    parent-body frame.

    The labeled mesh is in the body frame already (the labelling pipeline
    applies ADDITIONAL_OFFSETS for femur_r), so no further shift is required —
    extract_wrap_parameters_from_osim applies the same offset symmetrically.
    """
    pts = torch.as_tensor(points_body, dtype=torch.float32)
    center = torch.as_tensor(wrap_params["translation"], dtype=torch.float32)
    R = torch.as_tensor(
        ScipyR.from_euler("XYZ", wrap_params["xyz_body_rotation"]).as_matrix(),
        dtype=torch.float32,
    )
    if wrap_params["type"] == "WrapEllipsoid":
        axes = torch.as_tensor(wrap_params["dimensions"], dtype=torch.float32)
        with torch.no_grad():
            return sd_ellipsoid_improved(pts, center, axes, R).cpu().numpy()
    if wrap_params["type"] == "WrapCylinder":
        radius = torch.as_tensor(wrap_params["radius"], dtype=torch.float32)
        half_length = torch.as_tensor(wrap_params["length"] / 2.0, dtype=torch.float32)
        axis = R[:, 2]
        with torch.no_grad():
            return sd_cylinder_with_axis(pts, center, radius, half_length, axis).cpu().numpy()
    raise ValueError(f"Unknown wrap type: {wrap_params['type']}")


def evaluate_fit_quality(osim_path: Path, labeled_mesh_root: Path) -> dict:
    """For every wrap, compute classification accuracy on the labelled bone."""
    params = extract_wrap_parameters_from_osim(str(osim_path))
    results = {}
    # cache loaded labeled meshes
    meshes = {}
    for bone, rel in BONE_PATHS.items():
        p = labeled_mesh_root / rel
        if p.exists():
            meshes[bone] = pv.read(str(p))

    for bone, bd in params.items():
        for body, body_d in bd.items():
            for wn, wp in body_d.items():
                bone_for_label = WRAP_TO_BONE.get(wn)
                if bone_for_label is None or bone_for_label not in meshes:
                    continue
                m = meshes[bone_for_label]
                # Bone vertices in the labelled-mesh frame. For femur this is
                # the "knee local" frame (femur_r origin minus
                # ADDITIONAL_OFFSETS); extract_wrap_parameters_from_osim
                # symmetrically subtracts the same offset, so wp["translation"]
                # and the vertex positions are in the same frame. No correction
                # needed.
                pts = np.asarray(m.points, dtype=np.float32)
                # The labels are float 0/1 from the labelling pipeline. Threshold
                # at 0.5 to be safe.
                label_key = f"{wn}_binary"
                if label_key not in m.point_data:
                    continue
                labels = np.asarray(m.point_data[label_key]).astype(np.float32)
                inside_truth = labels > 0.5  # bool

                sdf = sdf_at_points_in_body_frame(wp, pts)
                pred_inside = sdf < 0.0

                # Classification accuracy + per-class breakdown
                correct = (pred_inside == inside_truth)
                n_inside_truth = int(inside_truth.sum())
                n_outside_truth = int((~inside_truth).sum())
                n_inside_correct = int((pred_inside & inside_truth).sum())
                n_outside_correct = int((~pred_inside & ~inside_truth).sum())

                # Margin violation magnitude (only on misclassified points)
                wrong = ~correct
                if wrong.any():
                    margin_violation_mm = 1000.0 * float(np.abs(sdf[wrong]).mean())
                    max_violation_mm = 1000.0 * float(np.abs(sdf[wrong]).max())
                else:
                    margin_violation_mm = 0.0
                    max_violation_mm = 0.0

                results[wn] = {
                    "type": wp["type"],
                    "n_total": int(labels.size),
                    "n_inside_truth": n_inside_truth,
                    "n_outside_truth": n_outside_truth,
                    "accuracy": float(correct.mean()),
                    "recall_inside": (n_inside_correct / n_inside_truth)
                                      if n_inside_truth else 1.0,
                    "specificity_outside": (n_outside_correct / n_outside_truth)
                                            if n_outside_truth else 1.0,
                    "mean_margin_violation_mm": margin_violation_mm,
                    "max_margin_violation_mm": max_violation_mm,
                }
    return results


def main():
    iter3_osim = Path("scratch/iter3/build_isolation_20260511_234741/A_v1/"
                       "custom_nsm_full_body_healthy_knee_model/isolation_test_A_v1/"
                       "isolation_test_A_v1.osim")
    iter7_osim = Path("scratch/iter7/build_isolation_20260512_124021/A_v1/"
                       "custom_nsm_full_body_healthy_knee_model/isolation_test_A_v1/"
                       "isolation_test_A_v1.osim")
    iter9_osim = Path("scratch/iter9/build_isolation_20260512_144421/A_v1/"
                       "custom_nsm_full_body_healthy_knee_model/isolation_test_A_v1/"
                       "isolation_test_A_v1.osim")
    iter10_osim = Path("scratch/iter10/build_isolation_20260512_153622/A_v1/"
                        "custom_nsm_full_body_healthy_knee_model/isolation_test_A_v1/"
                        "isolation_test_A_v1.osim")
    # Use iter7's labelled meshes for evaluation — they're geometrically the
    # same A-run input meshes for both iters.
    labeled_root = Path("scratch/iter7/build_isolation_20260512_124021")

    iter3 = evaluate_fit_quality(iter3_osim, labeled_root)
    iter7 = evaluate_fit_quality(iter7_osim, labeled_root)
    iter9 = evaluate_fit_quality(iter9_osim, labeled_root)
    iter10 = evaluate_fit_quality(iter10_osim, labeled_root)

    print(f"\n{'Wrap':<26}{'type':<5}{'iter3':>9}{'iter7':>9}{'iter9':>9}{'iter10':>9}"
          f"{'Δ(10-3)':>10}")
    for wn in sorted(iter3.keys()):
        a3 = iter3.get(wn); a7 = iter7.get(wn); a9 = iter9.get(wn); a10 = iter10.get(wn)
        if not (a3 and a7 and a9 and a10):
            continue
        kind = 'ell' if a3['type'] == 'WrapEllipsoid' else 'cyl'
        delta = a10['accuracy'] - a3['accuracy']
        sign = '↑' if delta > 0 else ('↓' if delta < 0 else '·')
        print(f"  {wn:<24}{kind:<5}"
              f"{a3['accuracy']:>9.4f}{a7['accuracy']:>9.4f}{a9['accuracy']:>9.4f}{a10['accuracy']:>9.4f}"
              f"{delta:>+9.5f}{sign}")

    m3 = np.mean([v["accuracy"] for v in iter3.values()])
    m7 = np.mean([v["accuracy"] for v in iter7.values()])
    m9 = np.mean([v["accuracy"] for v in iter9.values()])
    m10 = np.mean([v["accuracy"] for v in iter10.values()])
    print(f"\nMean accuracy across {len(iter3)} wraps: "
          f"iter3={m3:.4f}  iter7={m7:.4f}  iter9={m9:.4f}  iter10={m10:.4f}  Δ(10-3)={m10-m3:+.4f}")


if __name__ == "__main__":
    main()
