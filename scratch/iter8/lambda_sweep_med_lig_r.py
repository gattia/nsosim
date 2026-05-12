#!/usr/bin/env python
"""Quick λ-sweep on Med_Lig_r: does even lower regularization recover fit
quality without hurting reproducibility?

For each λ setting in {full iter7, 10× lower, 100× lower, 0}:
  - Fit Med_Lig_r on A's tibia labeled mesh (Procrustes anchor as init).
  - Fit Med_Lig_r on B's tibia labeled mesh (same anchor).
  - Report:
      - classification accuracy on A
      - center drift A vs B in microns
      - center distance from Smith2019 anchor in mm

If lower λ improves accuracy without inflating A-vs-B drift, that's the
fix for the Med_Lig_r regression.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from scipy.spatial.transform import Rotation as ScipyR

from nsosim.wrap_surface_fitting.fitting import EllipsoidFitter
from nsosim.wrap_surface_fitting.parameter_extraction import (
    extract_wrap_parameters_from_osim,
)
from nsosim.wrap_surface_fitting.procrustes_anchor import procrustes_anchor_for_wrap
from nsosim.wrap_surface_fitting.wrap_signed_distances import sd_ellipsoid_improved
from nsosim._determinism import set_global_seed

ITER7 = Path("scratch/iter7/build_isolation_20260512_124021")
SMITH2019 = (
    "/dataNAS/people/aagatti/projects/comak_gait_simulation/"
    "COMAK_SIMULATION_REQUIREMENTS/data/reference_data/comak_models/smith2019/"
    "smith2019.osim"
)
WRAP_NAME = "Med_Lig_r"


def fit_one(labeled_mesh_path: Path, anchor, *, lam_c, lam_a, lam_q, seed=0):
    set_global_seed(seed)
    from pymskt.mesh import Mesh as PMeshtKt

    m = PMeshtKt(str(labeled_mesh_path))  # wrap for fitter's get_near_surface_points
    pts = np.asarray(m.point_coords, dtype=np.float32)
    labels = np.asarray(m.point_data[f"{WRAP_NAME}_binary"], dtype=np.float32)
    sdf = np.asarray(m.point_data[f"{WRAP_NAME}_sdf"], dtype=np.float32)

    fitter = EllipsoidFitter(
        lr=1e-2,
        epochs=10,
        use_lbfgs=True,
        lbfgs_epochs=100,
        alpha=1.0,
        beta=0.0,
        gamma=0.0,
        margin_decay_type="linear",
        initialization="geometric",
        center_transform="linear",
        anchor_params=anchor,
        lambda_center_reg=lam_c,
        lambda_axes_reg=lam_a,
        lambda_quat_reg=lam_q,
    )
    fitter.fit(points=pts, labels=labels, sdf=sdf, mesh=m, surface_name=WRAP_NAME, margin=0.0002)
    wp = fitter.wrap_params

    # Compute fit-quality accuracy on labels
    R = torch.as_tensor(
        ScipyR.from_euler("XYZ", wp.xyz_body_rotation).as_matrix(), dtype=torch.float32
    )
    center = torch.as_tensor(wp.translation, dtype=torch.float32)
    axes = torch.as_tensor(wp.dimensions, dtype=torch.float32)
    pts_t = torch.as_tensor(pts, dtype=torch.float32)
    with torch.no_grad():
        fitted_sdf = sd_ellipsoid_improved(pts_t, center, axes, R).cpu().numpy()
    pred_inside = fitted_sdf < 0.0
    truth_inside = labels > 0.5
    accuracy = float((pred_inside == truth_inside).mean())
    return wp, accuracy


def main():
    # Build the anchor once (same for A and B — Smith2019-derived, identity-frame)
    smith_params = extract_wrap_parameters_from_osim(SMITH2019)
    smith_med_lig = smith_params["tibia"]["tibia_proximal_r"][WRAP_NAME]
    anchor = procrustes_anchor_for_wrap(
        WRAP_NAME, smith_med_lig, bone_transform=None, n_points=4000,
        body="tibia_proximal_r",
    )
    smith_center = np.asarray(smith_med_lig["translation"])

    label_a = ITER7 / "A_v1" / "bones" / "tibia" / "tibia_labeled_mesh_updated.vtk"
    label_b = ITER7 / "B" / "bones" / "tibia" / "tibia_labeled_mesh_updated.vtk"

    settings = [
        ("iter7 default", anchor, 0.05, 0.005, 0.005),
        ("10x lower",    anchor, 0.005, 0.0005, 0.0005),
        ("100x lower",   anchor, 0.0005, 0.00005, 0.00005),
        ("zero",         anchor, 0.0,   0.0,    0.0),
        # Control: no anchor (algebraic init) at iter3-level lambdas — should
        # reproduce iter3's 99.4% accuracy on Med_Lig_r if the multi-minima
        # hypothesis is right.
        ("algebraic iter3-λ", None, 1.0, 0.1, 0.1),
        ("algebraic λ=0",    None, 0.0, 0.0, 0.0),
    ]

    print(f"\n{'setting':<22}{'acc(A)':>10}{'A↔B center µm':>18}{'A-to-Smith mm':>17}")
    for name, anc, lc, la, lq in settings:
        wp_a, acc_a = fit_one(label_a, anc, lam_c=lc, lam_a=la, lam_q=lq)
        wp_b, acc_b = fit_one(label_b, anc, lam_c=lc, lam_a=la, lam_q=lq)
        ab_drift_um = 1e6 * float(np.linalg.norm(np.asarray(wp_a.translation) - np.asarray(wp_b.translation)))
        d_smith_mm = 1000 * float(np.linalg.norm(np.asarray(wp_a.translation) - smith_center))
        print(f"  {name:<20}{acc_a:>10.5f}{ab_drift_um:>18.2f}{d_smith_mm:>17.4f}")


if __name__ == "__main__":
    main()
