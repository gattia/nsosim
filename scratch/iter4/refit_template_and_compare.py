#!/usr/bin/env python
"""Acceptance criterion 3: refit wraps to the reference (template) labeled bone
with the iter3 code and compare ASSD against original Smith2019 references.

The requirement (from WRAP_FITTER_ROBUSTNESS.md): ASSD of my refit vs original
must be within 10% of the current production refit vs original.
"""
import os
import sys

import numpy as np
import pyvista as pv
from pymskt.mesh import Mesh
from scipy.spatial import cKDTree

# Use the worktree's nsosim
sys.path.insert(
    0, "/dataNAS/people/aagatti/programming/nsosim/.claude/worktrees/wrap-fitter-robustness"
)

from nsosim.model_building import fit_bone_wrap_surfaces
from nsosim.wrap_surface_fitting.parameter_extraction import create_meshes_from_wrap_parameters

BASE = "/dataNAS/people/aagatti/projects/comak_gait_simulation/COMAK_SIMULATION_REQUIREMENTS/fitted_base_wrap_surfaces"


def assd(m1, m2):
    p1 = np.asarray(m1.points)
    p2 = np.asarray(m2.points)
    d12 = cKDTree(p2).query(p1)[0]
    d21 = cKDTree(p1).query(p2)[0]
    return (d12.mean() + d21.mean()) / 2


def load_labeled_bone(bone_name):
    path = f"{BASE}/labeled_bones/{bone_name}_labeled.vtk"
    return Mesh(path)


def refit_all():
    """Refit wraps for femur/tibia/patella with iter3 code."""
    results = {}
    for bone_name in ["femur", "tibia", "patella"]:
        print(f"=== Fitting {bone_name} ===")
        mesh = load_labeled_bone(bone_name)
        points = mesh.point_coords.copy()
        fitted = fit_bone_wrap_surfaces(
            bone_name=bone_name,
            labeled_mesh=mesh,
            labeled_mesh_points=points,
            n_restarts=1,
            jitter_scale=1e-6,
        )
        results[bone_name] = fitted
    return results


def to_param_dict(fitted):
    """Convert nested fitted dict into the input format for create_meshes_from_wrap_parameters."""
    out = {}
    for bone, bodies in fitted.items():
        out[bone] = {}
        for body, surfaces in bodies.items():
            out[bone][body] = {}
            for stype, wraps in surfaces.items():
                for wname, wparams in wraps.items():
                    # wparams is a wrap_surface dataclass instance
                    d = wparams.to_dict()
                    d["type"] = wparams.type_
                    out[bone][body][wname] = d
    return out


def main():
    fitted = refit_all()
    pd = to_param_dict(fitted)
    meshes = create_meshes_from_wrap_parameters(pd)

    print()
    print("Comparison (ASSD in mm):")
    print(f"  {'wrap':<28} {'my_iter3 vs orig':>16} {'curr_prod vs orig':>17} {'ratio':>7}")
    print("  " + "-" * 76)
    rows = []
    for bone, bodies in meshes.items():
        for body, wraps in bodies.items():
            for wname, my_mesh in wraps.items():
                orig_path = f"{BASE}/original_surfaces/{bone}/{wname}_original.vtk"
                fit_path = f"{BASE}/fitted_surfaces/{bone}/{wname}_fitted.vtk"
                if not (os.path.exists(orig_path) and os.path.exists(fit_path)):
                    continue
                orig = pv.read(orig_path)
                cur_fit = pv.read(fit_path)
                # my_mesh is a pv.PolyData
                a_my = assd(my_mesh, orig) * 1000  # mm
                a_cp = assd(cur_fit, orig) * 1000
                ratio = a_my / a_cp if a_cp > 1e-9 else float("nan")
                rows.append((wname, a_my, a_cp, ratio))
                marker = "OK" if ratio <= 1.1 else "FAIL"
                print(f"  {wname:<28} {a_my:>16.4f} {a_cp:>17.4f} {ratio:>7.3f}  {marker}")
    print()
    fail = [r for r in rows if r[3] > 1.1]
    if fail:
        print(f"FAIL: {len(fail)} wraps exceed 1.1× current production ASSD:")
        for n, m, c, r in fail:
            print(f"  - {n}: my={m:.4f} mm, prod={c:.4f} mm, ratio={r:.2f}×")
        sys.exit(1)
    else:
        print(f"PASS: all {len(rows)} wraps within 1.1× current production ASSD.")


if __name__ == "__main__":
    main()
