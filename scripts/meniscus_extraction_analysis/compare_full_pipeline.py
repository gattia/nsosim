#!/usr/bin/env python
"""Compare scored vs ray-casting through the IDENTICAL full pipeline.

Both methods get the same processing:
  ACVD resample → extract (method differs) → get_n_largest → remove_isolated
  → smooth → radial envelope trim

Only the extraction step (step 4) differs. Everything else is identical.

Usage:
    conda run -n comak python scripts/meniscus_extraction_analysis/compare_full_pipeline.py
    conda run -n comak python scripts/meniscus_extraction_analysis/compare_full_pipeline.py --n-subjects 20
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nsosim.articular_surfaces import create_meniscus_articulating_surface

logging.basicConfig(level=logging.WARNING)

# ============================================================================
# Configuration
# ============================================================================
RESULTS_ROOT = Path(
    "/dataNAS/people/aagatti/projects/comak_gait_simulation_results/OARSI_menisci_pfp_v1"
)

SCORE_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]

# Production pipeline params (from comak_1_nsm_fitting.py lines 628-652)
SHARED_PARAMS = {
    "ray_length": 15.0,
    "n_largest": 1,
    "smooth_iter": 10,
    "boundary_smoothing": False,
    "radial_percentile": 95.0,
}


def get_subject_dirs(n_subjects, seed=42):
    """Get n_subjects random subject directories with all required files."""
    rng = np.random.default_rng(seed)
    all_dirs = sorted(RESULTS_ROOT.iterdir())
    rng.shuffle(all_dirs)

    valid = []
    for d in all_dirs:
        if not d.is_dir():
            continue
        geom = d / "geometries_nsm_similarity"
        required = [
            geom / "femur" / "med_men_osim.stl",
            geom / "femur" / "lat_men_osim.stl",
            geom / "femur" / "femur_nsm_recon_osim.stl",
            geom / "tibia" / "tibia_nsm_recon_osim.stl",
            geom / "tibia" / "tibia_labeled_mesh_updated.vtk",
        ]
        if all(f.exists() for f in required):
            valid.append(d)
        if len(valid) >= n_subjects:
            break
    return valid


def load_subject(subject_dir):
    """Load all meshes for one subject (in meters, OSIM space)."""
    geom = subject_dir / "geometries_nsm_similarity"

    data = {
        "name": subject_dir.name,
        "med_men": pv.read(str(geom / "femur" / "med_men_osim.stl")),
        "lat_men": pv.read(str(geom / "femur" / "lat_men_osim.stl")),
        "femur": pv.read(str(geom / "femur" / "femur_nsm_recon_osim.stl")),
        "tibia": pv.read(str(geom / "tibia" / "tibia_nsm_recon_osim.stl")),
    }

    tib_labeled = pv.read(str(geom / "tibia" / "tibia_labeled_mesh_updated.vtk"))
    tib_pts = np.array(tib_labeled.points)
    for side in ("med", "lat"):
        key = f"{side}_meniscus_center_binary"
        if key in tib_labeled.point_data:
            labels = np.array(tib_labeled.point_data[key])
            data[f"{side}_center"] = tib_pts[labels == 1].mean(axis=0) if np.any(labels == 1) else None
        else:
            data[f"{side}_center"] = None

    return data


def compute_surface_metrics(mesh_a, mesh_b):
    """Compute detailed comparison metrics between two surfaces.

    Returns dict with ASSD, directed distances, area comparison, Hausdorff.
    All distances in the same units as input meshes (meters).
    """
    from scipy.spatial import KDTree

    pts_a = np.array(mesh_a.points)
    pts_b = np.array(mesh_b.points)

    tree_a = KDTree(pts_a)
    tree_b = KDTree(pts_b)

    dists_a_to_b, _ = tree_b.query(pts_a)
    dists_b_to_a, _ = tree_a.query(pts_b)

    area_a = float(mesh_a.area)
    area_b = float(mesh_b.area)

    return {
        "assd": (dists_a_to_b.mean() + dists_b_to_a.mean()) / 2,
        "a_to_b_mean": float(dists_a_to_b.mean()),
        "a_to_b_p95": float(np.percentile(dists_a_to_b, 95)),
        "a_to_b_max": float(dists_a_to_b.max()),
        "b_to_a_mean": float(dists_b_to_a.mean()),
        "b_to_a_p95": float(np.percentile(dists_b_to_a, 95)),
        "b_to_a_max": float(dists_b_to_a.max()),
        "hausdorff": float(max(dists_a_to_b.max(), dists_b_to_a.max())),
        "area_a": area_a,
        "area_b": area_b,
        "area_ratio": area_a / area_b if area_b > 0 else float("inf"),
        "n_pts_a": mesh_a.n_points,
        "n_pts_b": mesh_b.n_points,
    }


def run_full_pipeline(meniscus_m, femur_m, tibia_m, meniscus_center, theta_offset,
                      extraction_method, score_threshold=0.5, score_smooth_iterations=0):
    """Run the full pipeline with specified extraction method."""
    from pymskt.mesh import Mesh

    upper, lower = create_meniscus_articulating_surface(
        meniscus_mesh=Mesh(mesh=meniscus_m.copy()),
        upper_articulating_bone_mesh=Mesh(mesh=femur_m.copy()),
        lower_articulating_bone_mesh=Mesh(mesh=tibia_m.copy()),
        meniscus_center=meniscus_center,
        theta_offset=theta_offset,
        extraction_method=extraction_method,
        score_threshold=score_threshold,
        score_smooth_iterations=score_smooth_iterations,
        **SHARED_PARAMS,
    )
    upper_pv = upper.mesh if hasattr(upper, "mesh") else upper
    lower_pv = lower.mesh if hasattr(lower, "mesh") else lower
    return upper_pv, lower_pv


def main():
    parser = argparse.ArgumentParser(
        description="Compare scored vs ray-casting through identical full pipeline"
    )
    parser.add_argument("--n-subjects", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Finding {args.n_subjects} subjects...")
    subject_dirs = get_subject_dirs(args.n_subjects, seed=args.seed)
    print(f"Found {len(subject_dirs)} subjects\n")

    # Results: {threshold: {"upper": [metrics_dicts], "lower": [metrics_dicts]}}
    all_results = {t: {"upper": [], "lower": []} for t in SCORE_THRESHOLDS}

    meniscus_configs = [
        ("medial", "med_men", "med_center", np.pi),
        ("lateral", "lat_men", "lat_center", 0.0),
    ]

    for i, subject_dir in enumerate(subject_dirs):
        t0 = time.time()
        print(f"[{i+1}/{len(subject_dirs)}] {subject_dir.name}")

        try:
            data = load_subject(subject_dir)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        for men_name, men_key, center_key, theta_offset in meniscus_configs:
            meniscus_m = data[men_key]
            center = data[center_key]

            # --- Ray-casting (reference) ---
            try:
                ray_upper, ray_lower = run_full_pipeline(
                    meniscus_m, data["femur"], data["tibia"],
                    center, theta_offset, extraction_method="ray_casting",
                )
            except Exception as e:
                print(f"  {men_name}: ray-casting FAILED: {e}")
                continue

            print(f"  {men_name} ray: upper={ray_upper.n_points}pts, lower={ray_lower.n_points}pts")

            # --- Scored at each threshold ---
            for threshold in SCORE_THRESHOLDS:
                try:
                    scored_upper, scored_lower = run_full_pipeline(
                        meniscus_m, data["femur"], data["tibia"],
                        center, theta_offset,
                        extraction_method="scored",
                        score_threshold=threshold,
                        score_smooth_iterations=0,  # no smoothing (calibration showed it hurts)
                    )
                except Exception as e:
                    print(f"  {men_name} scored t={threshold}: FAILED: {e}")
                    continue

                for surf_type, scored_surf, ray_surf in [
                    ("upper", scored_upper, ray_upper),
                    ("lower", scored_lower, ray_lower),
                ]:
                    if scored_surf.n_points == 0 or ray_surf.n_points == 0:
                        continue

                    m = compute_surface_metrics(scored_surf, ray_surf)
                    m["subject"] = data["name"]
                    m["meniscus"] = men_name
                    m["threshold"] = threshold
                    all_results[threshold][surf_type].append(m)

                print(
                    f"    t={threshold:.1f}: upper={scored_upper.n_points}pts, "
                    f"lower={scored_lower.n_points}pts"
                )

        elapsed = time.time() - t0
        print(f"  ({elapsed:.1f}s)\n")

    # ========================================================================
    # Report
    # ========================================================================
    print("=" * 120)
    print("FULL PIPELINE COMPARISON: Scored vs Ray-casting (identical post-processing)")
    print("=" * 120)
    print(f"Subjects: {len(subject_dirs)}")
    print(f"Thresholds: {SCORE_THRESHOLDS}")
    print()
    print("Both methods: ACVD resample → EXTRACT → get_n_largest → remove_isolated → smooth(10) → radial trim(p95)")
    print("Only the EXTRACT step differs (ray-casting vs score threshold).")
    print("Score smoothing = 0 (disabled).")
    print()

    for surf_type in ("upper", "lower"):
        print(f"--- {surf_type.upper()} articular surface ---")
        print(
            f"  {'Thresh':>6} | {'ASSD':>6} | {'S→R mean':>8} | {'S→R p95':>7} | {'S→R max':>7} | "
            f"{'R→S mean':>8} | {'R→S p95':>7} | {'R→S max':>7} | "
            f"{'Area':>6} | {'S pts':>6} | {'R pts':>6} | {'N':>3}"
        )
        print(
            f"  {'':>6} | {'(mm)':>6} | {'(mm)':>8} | {'(mm)':>7} | {'(mm)':>7} | "
            f"{'(mm)':>8} | {'(mm)':>7} | {'(mm)':>7} | "
            f"{'ratio':>6} | {'':>6} | {'':>6} | {'':>3}"
        )
        print(f"  {'-' * 110}")

        for threshold in SCORE_THRESHOLDS:
            entries = all_results[threshold][surf_type]
            if not entries:
                print(f"  {threshold:>6.1f} | {'N/A':>6}")
                continue

            def m(key):
                return np.mean([e[key] for e in entries])

            print(
                f"  {threshold:>6.1f} | {m('assd')*1000:>6.3f} | "
                f"{m('a_to_b_mean')*1000:>8.3f} | {m('a_to_b_p95')*1000:>7.3f} | "
                f"{m('a_to_b_max')*1000:>7.3f} | "
                f"{m('b_to_a_mean')*1000:>8.3f} | {m('b_to_a_p95')*1000:>7.3f} | "
                f"{m('b_to_a_max')*1000:>7.3f} | "
                f"{m('area_ratio'):>6.2f} | {m('n_pts_a'):>6.0f} | {m('n_pts_b'):>6.0f} | "
                f"{len(entries):>3}"
            )
        print()

    # Interpretation
    print("=" * 120)
    print("METRIC INTERPRETATION")
    print("=" * 120)
    print("  S→R (scored→ray):  How far scored points are from nearest ray-casting point.")
    print("  R→S (ray→scored):  How far ray-casting points are from nearest scored point.")
    print("  Area ratio:        scored_area / ray_area.")
    print("  S pts / R pts:     Mean point count for scored / ray-casting surfaces.")
    print()

    # Breakdown by meniscus type
    print("=" * 120)
    print("BREAKDOWN BY MENISCUS TYPE")
    print("=" * 120)
    for threshold in SCORE_THRESHOLDS:
        print(f"\n  threshold = {threshold}")
        for surf_type in ("upper", "lower"):
            entries = all_results[threshold][surf_type]
            for men_type in ("medial", "lateral"):
                subset = [e for e in entries if e["meniscus"] == men_type]
                if not subset:
                    continue

                def sm(key):
                    return np.mean([e[key] for e in subset])

                print(
                    f"    {men_type:>8} {surf_type:>5}: "
                    f"ASSD={sm('assd')*1000:.3f}mm, "
                    f"S→R={sm('a_to_b_mean')*1000:.3f}/{sm('a_to_b_p95')*1000:.3f}mm, "
                    f"R→S={sm('b_to_a_mean')*1000:.3f}/{sm('b_to_a_p95')*1000:.3f}mm, "
                    f"area={sm('area_ratio'):.2f}x, "
                    f"pts={sm('n_pts_a'):.0f}/{sm('n_pts_b'):.0f}, "
                    f"n={len(subset)}"
                )


if __name__ == "__main__":
    main()
