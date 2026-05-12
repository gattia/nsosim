#!/usr/bin/env python
"""NSM correspondence / labeling diagnostic for runs A vs B.

Replaces the v1 of this script which incorrectly used vertex-to-vertex
distance for "ASSD" (bounded below by edge length) and reported per-vertex
displacement for ACVD-resampled meshes (which have arbitrary vertex layouts
by construction).

Two real tests:

  1. **Point-to-surface ASSD** between A's NSM-fit bone mesh and B's, via
     pymskt's `Mesh.get_assd_mesh()` (uses point-to-triangle SDF). Answers:
     how different are the actual surfaces?

  2. **Spatial label-transfer test.** Independent labeling pipelines produce
     mesh-A-with-labels-A and mesh-B-with-labels-B (SDF, binary, near_surface
     fields per wrap). Use pymskt's `Mesh.copy_scalars_from_other_mesh_to_current`
     to transfer A's scalars onto B's vertices. Compare transferred-A vs
     independent-B per-scalar:
       - For continuous SDFs: mean / max |Δ|.
       - For binary fields: count of vertices where the label flips.

What the post-ACVD meshes can / can't tell us
---------------------------------------------
- They CAN tell us how different the surfaces are between runs (ASSD).
- They CAN tell us whether labels are stable under that surface drift.
- They CANNOT tell us about NSM canonical-space correspondence — ACVD
  re-samples vertices independently per run, so vertex i has no
  cross-run anatomical meaning. That diagnostic needs the pre-resample
  decoder output, which build_joint_model doesn't currently persist.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from pymskt.mesh import Mesh

ITER7_BASE = Path(
    "/dataNAS/people/aagatti/programming/nsosim/.claude/worktrees/"
    "wrap-fitter-robustness/scratch/iter7/build_isolation_20260512_124021"
)


def compare_surfaces(mesh_a_path: Path, mesh_b_path: Path) -> dict:
    """Point-to-surface ASSD between two meshes via pymskt."""
    a = Mesh(str(mesh_a_path))
    b = Mesh(str(mesh_b_path))
    assd = a.get_assd_mesh(b)
    return {
        "n_pts_a": a.point_coords.shape[0],
        "n_pts_b": b.point_coords.shape[0],
        "assd_mm": 1000.0 * float(assd),
    }


def label_transfer_test(labeled_a_path: Path, labeled_b_path: Path,
                         n_closest: int = 3) -> dict:
    """Transfer A's labels onto B's vertices spatially (pymskt weighted-NN)
    and compare to B's own labels."""
    # Load B twice: one keeps its own labels (the truth), one receives A's.
    b_native = Mesh(str(labeled_b_path))
    b_target = Mesh(str(labeled_b_path))
    a = Mesh(str(labeled_a_path))

    # pymskt's helper. Auto-detects categorical-vs-continuous per array.
    # Rename incoming arrays so they don't overwrite B's own labels on b_target.
    incoming = list(b_native.point_data.keys())
    renamed = [f"{n}__fromA" for n in incoming]
    b_target.copy_scalars_from_other_mesh_to_current(
        a,
        orig_scalars_name=incoming,
        new_scalars_name=renamed,
        weighted_avg=True,
        n_closest=n_closest,
    )

    report = {"n_arrays": len(incoming), "by_array": {}}
    for arr in incoming:
        b_truth = np.asarray(b_native.point_data[arr], dtype=np.float64)
        try:
            b_from_a = np.asarray(b_target.point_data[f"{arr}__fromA"], dtype=np.float64)
        except Exception:
            # Array didn't make it through transfer for some reason (e.g.
            # categorical handling); skip.
            continue
        if b_truth.shape != b_from_a.shape:
            continue
        diff = b_from_a - b_truth

        unique = np.unique(b_truth)
        is_binary = (unique.size <= 2) and set(unique.tolist()).issubset({0.0, 1.0})
        info = {"is_binary": is_binary, "n_vertices": b_truth.size,
                "frac_positive_native": float(b_truth.mean())}
        if is_binary:
            n_flipped = int(np.sum(np.abs(diff) > 0.5))
            info["n_flipped"] = n_flipped
            info["pct_flipped"] = 100.0 * n_flipped / b_truth.size
        else:
            info["mean_abs_diff_mm"] = 1000.0 * float(np.abs(diff).mean())
            info["max_abs_diff_mm"] = 1000.0 * float(np.abs(diff).max())
            info["p95_abs_diff_mm"] = 1000.0 * float(np.percentile(np.abs(diff), 95))
        report["by_array"][arr] = info
    return report


def fmt_surface(label: str, r: dict) -> str:
    return (
        f"\n=== {label} ===\n"
        f"  n_pts (A, B):  {r['n_pts_a']}, {r['n_pts_b']}\n"
        f"  point-to-surface ASSD:  {r['assd_mm']:.6f} mm"
    )


def fmt_labels(label: str, r: dict) -> str:
    lines = [f"\n=== {label} ===",
             f"  arrays compared: {r.get('n_arrays', 0)}"]
    if "by_array" not in r:
        return "\n".join(lines)
    for arr in sorted(r["by_array"]):
        info = r["by_array"][arr]
        if info["is_binary"]:
            lines.append(
                f"  {arr:<42}  binary  flipped={info['n_flipped']:>6}/"
                f"{info['n_vertices']} ({info['pct_flipped']:.4f}%)"
            )
        else:
            lines.append(
                f"  {arr:<42}  cont    "
                f"mean|Δ|={info['mean_abs_diff_mm']:.6f} mm  "
                f"p95={info['p95_abs_diff_mm']:.6f} mm  "
                f"max={info['max_abs_diff_mm']:.6f} mm"
            )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bone", default="femur", choices=["femur", "tibia", "patella"])
    p.add_argument("--n-closest", type=int, default=3)
    args = p.parse_args()

    surf_a = ITER7_BASE / "A_v1" / "bones" / args.bone / f"{args.bone}_nsm_recon_osim.vtk"
    surf_b = ITER7_BASE / "B" / "bones" / args.bone / f"{args.bone}_nsm_recon_osim.vtk"
    labeled_a = ITER7_BASE / "A_v1" / "bones" / args.bone / f"{args.bone}_labeled_mesh_updated.vtk"
    labeled_b = ITER7_BASE / "B" / "bones" / args.bone / f"{args.bone}_labeled_mesh_updated.vtk"
    for p_ in (surf_a, surf_b, labeled_a, labeled_b):
        if not p_.exists():
            raise FileNotFoundError(p_)

    print(f"Bone: {args.bone}")
    print(f"  A surface: {surf_a}")
    print(f"  B surface: {surf_b}")

    print(fmt_surface("TEST 1: point-to-surface ASSD (A vs B)",
                       compare_surfaces(surf_a, surf_b)))

    print(fmt_labels(
        "TEST 2: spatial label-transfer A→B vs independent B labels",
        label_transfer_test(labeled_a, labeled_b, n_closest=args.n_closest),
    ))


if __name__ == "__main__":
    main()
