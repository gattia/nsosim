#!/usr/bin/env python
"""Per-wrap λ sweep across all 6 fittable wraps (ellipsoids + cylinders),
each with both Procrustes-anchor init and algebraic init.

For each wrap × setting:
  - Fit on A's labeled mesh.
  - Fit on B's labeled mesh (separate run, same seed).
  - Report:
      acc(A)        — classification accuracy on Smith2019 labels (A run)
      ↔drift(µm)    — center drift between A and B fits
      Δ_Smith(mm)   — final fit center vs Smith2019 reference center

Goal: identify which wraps benefit from anchor (good init basin) and which
get stuck in worse minima (multi-minima problem → per-wrap opt-out).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pymskt.mesh import Mesh as PMesh
from scipy.spatial.transform import Rotation as ScipyR

from nsosim._determinism import set_global_seed
from nsosim.wrap_surface_fitting.fitting import (
    CylinderFitter,
    EllipsoidFitter,
    sd_cylinder_with_axis,
)
from nsosim.wrap_surface_fitting.parameter_extraction import (
    extract_wrap_parameters_from_osim,
)
from nsosim.wrap_surface_fitting.procrustes_anchor import procrustes_anchor_for_wrap
from nsosim.wrap_surface_fitting.wrap_signed_distances import sd_ellipsoid_improved

ITER7 = Path("scratch/iter7/build_isolation_20260512_124021")
SMITH2019 = (
    "/dataNAS/people/aagatti/projects/comak_gait_simulation/"
    "COMAK_SIMULATION_REQUIREMENTS/data/reference_data/comak_models/smith2019/"
    "smith2019.osim"
)

# wrap → (bone, body, kind, [optional bone offset]). For femur_r-attached
# wraps we subtract the ADDITIONAL_OFFSETS shift the labeling pipeline
# applies symmetrically.
WRAPS = [
    ("Gastroc_at_Condyles_r", "femur", "femur_r",        "ellipsoid"),
    ("KnExt_at_fem_r",        "femur", "femur_r",        "cylinder"),
    ("KnExt_vasint_at_fem_r", "femur", "femur_r",        "cylinder"),
    ("Capsule_r",             "femur", "femur_distal_r", "cylinder"),
    ("Med_Lig_r",             "tibia", "tibia_proximal_r","ellipsoid"),
    ("Med_LigP_r",            "tibia", "tibia_proximal_r","ellipsoid"),
]


# ---------------------------------------------------------------------------


def _fit_ellipsoid(labeled_mesh_path, wrap_name, anchor, lam_c, lam_a, lam_q, seed=0):
    set_global_seed(seed)
    m = PMesh(str(labeled_mesh_path))
    pts = np.asarray(m.point_coords, dtype=np.float32)
    labels = np.asarray(m.point_data[f"{wrap_name}_binary"], dtype=np.float32)
    sdf = np.asarray(m.point_data[f"{wrap_name}_sdf"], dtype=np.float32)
    fitter = EllipsoidFitter(
        lr=1e-2, epochs=10, use_lbfgs=True, lbfgs_epochs=100,
        alpha=1.0, beta=0.0, gamma=0.0, margin_decay_type="linear",
        initialization="geometric", center_transform="linear",
        anchor_params=anchor,
        lambda_center_reg=lam_c, lambda_axes_reg=lam_a, lambda_quat_reg=lam_q,
    )
    fitter.fit(points=pts, labels=labels, sdf=sdf, mesh=m, surface_name=wrap_name, margin=0.0002)
    wp = fitter.wrap_params
    return wp, _accuracy_ellipsoid(wp, pts, labels)


def _fit_cylinder(labeled_mesh_path, wrap_name, anchor, lam_c, lam_axis, seed=0):
    set_global_seed(seed)
    m = PMesh(str(labeled_mesh_path))
    pts_all = np.asarray(m.point_coords, dtype=np.float32)
    labels_all = np.asarray(m.point_data[f"{wrap_name}_binary"], dtype=np.float32)
    sdf_all = np.asarray(m.point_data[f"{wrap_name}_sdf"], dtype=np.float32)
    near_bool = np.asarray(m.point_data[f"{wrap_name}_near_surface"], dtype=np.float32) > 0.5
    near_pts = pts_all[near_bool]
    near_labels = labels_all[near_bool]
    near_sdf = sdf_all[near_bool]
    fitter = CylinderFitter(
        lr=0.0, epochs=0, use_lbfgs=True, lbfgs_epochs=100,
        alpha=1.0, beta=0.0, gamma=0.0, margin_decay_type=None,
        initialization="geometric", center_transform="linear",
        anchor_params=anchor,
        lambda_center_reg=lam_c, lambda_axis_reg=lam_axis,
    )
    fitter.fit(points=near_pts, labels=near_labels, sdf=near_sdf, mesh=m,
                surface_name=wrap_name, near_surface_points=near_pts, margin=1e-10)
    wp = fitter.wrap_params
    return wp, _accuracy_cylinder(wp, pts_all, labels_all)


def _accuracy_ellipsoid(wp, pts, labels):
    R = torch.as_tensor(ScipyR.from_euler("XYZ", wp.xyz_body_rotation).as_matrix(), dtype=torch.float32)
    c = torch.as_tensor(wp.translation, dtype=torch.float32)
    a = torch.as_tensor(wp.dimensions, dtype=torch.float32)
    with torch.no_grad():
        sdf_fitted = sd_ellipsoid_improved(torch.as_tensor(pts, dtype=torch.float32), c, a, R).cpu().numpy()
    return float(((sdf_fitted < 0) == (labels > 0.5)).mean())


def _accuracy_cylinder(wp, pts, labels):
    R = torch.as_tensor(ScipyR.from_euler("XYZ", wp.xyz_body_rotation).as_matrix(), dtype=torch.float32)
    c = torch.as_tensor(wp.translation, dtype=torch.float32)
    r = torch.as_tensor(wp.radius, dtype=torch.float32)
    h = torch.as_tensor(wp.length / 2.0, dtype=torch.float32)
    axis = R[:, 2]
    with torch.no_grad():
        sdf_fitted = sd_cylinder_with_axis(torch.as_tensor(pts, dtype=torch.float32), c, r, h, axis).cpu().numpy()
    return float(((sdf_fitted < 0) == (labels > 0.5)).mean())


# ---------------------------------------------------------------------------


def main():
    smith_params = extract_wrap_parameters_from_osim(SMITH2019)

    # 6 settings: 4 with Procrustes anchor + 2 algebraic-init controls.
    # For ellipsoid: (lam_c, lam_a, lam_q). For cylinder: (lam_c, lam_axis).
    # Ellipsoid settings:
    ellipsoid_settings = [
        ("iter9-default", "anchor", 0.05, 0.005, 0.005),
        ("10x lower",     "anchor", 0.005, 0.0005, 0.0005),
        ("100x lower",    "anchor", 0.0005, 0.00005, 0.00005),
        ("zero",          "anchor", 0.0,   0.0,    0.0),
        ("algebraic iter3-λ", None, 1.0,  0.1,    0.1),
        ("algebraic λ=0",     None, 0.0,  0.0,    0.0),
    ]
    cylinder_settings = [
        ("iter9-default", "anchor", 0.05, 0.1),
        ("10x lower",     "anchor", 0.005, 0.01),
        ("100x lower",    "anchor", 0.0005, 0.001),
        ("zero",          "anchor", 0.0,   0.0),
        ("algebraic iter3-λ", None, 1.0,  0.0),  # iter3 had lambda_axis=0
        ("algebraic λ=0",     None, 0.0,  0.0),
    ]

    for wrap_name, bone, body, kind in WRAPS:
        smith_wp = smith_params[bone][body][wrap_name]
        smith_center = np.asarray(smith_wp["translation"])
        anchor = procrustes_anchor_for_wrap(
            wrap_name, smith_wp, bone_transform=None, n_points=4000, body=body
        )
        label_a = ITER7 / "A_v1" / "bones" / bone / f"{bone}_labeled_mesh_updated.vtk"
        label_b = ITER7 / "B" / "bones" / bone / f"{bone}_labeled_mesh_updated.vtk"

        print(f"\n=== {wrap_name}  ({kind}, {bone}/{body}) ===")
        print(f"{'setting':<24}{'acc(A)':>10}{'↔drift µm':>14}{'Δ_Smith mm':>14}")

        settings = ellipsoid_settings if kind == "ellipsoid" else cylinder_settings
        for row in settings:
            name = row[0]; init = row[1]; args = row[2:]
            anc = anchor if init == "anchor" else None
            if kind == "ellipsoid":
                lc, la, lq = args
                wp_a, acc_a = _fit_ellipsoid(label_a, wrap_name, anc, lc, la, lq)
                wp_b, _      = _fit_ellipsoid(label_b, wrap_name, anc, lc, la, lq)
            else:
                lc, la = args
                wp_a, acc_a = _fit_cylinder(label_a, wrap_name, anc, lc, la)
                wp_b, _      = _fit_cylinder(label_b, wrap_name, anc, lc, la)
            ab_um = 1e6 * float(np.linalg.norm(np.asarray(wp_a.translation) - np.asarray(wp_b.translation)))
            d_smith = 1000 * float(np.linalg.norm(np.asarray(wp_a.translation) - smith_center))
            print(f"  {name:<22}{acc_a:>10.5f}{ab_um:>14.2f}{d_smith:>14.4f}")


if __name__ == "__main__":
    main()
