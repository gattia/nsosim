#!/usr/bin/env python
"""Compare scored extraction (minimal cleanup) vs ray-casting (full pipeline).

For each subject, compares:
  - Ray-casting FULL pipeline: ray-cast → get_n_largest → remove_isolated → smooth → radial envelope
  - Scored (minimal):          threshold → get_n_largest → remove_isolated_cells (no smooth, no radial)

Tests whether a high score threshold can replace all the post-processing steps.

Reports ASSD, point counts, number of connected components before filtering, etc.

Usage:
    conda run -n comak python scripts/meniscus_extraction_analysis/compare_extraction_quality.py
    conda run -n comak python scripts/meniscus_extraction_analysis/compare_extraction_quality.py --n-subjects 20
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pymskt.mesh.meshCartilage import get_n_largest, remove_isolated_cells

from nsosim.articular_surfaces import (
    create_meniscus_articulating_surface,
    score_meniscus_vertices,
)

logging.basicConfig(level=logging.WARNING)

# ============================================================================
# Configuration
# ============================================================================
RESULTS_ROOT = Path(
    "/dataNAS/people/aagatti/projects/comak_gait_simulation_results/OARSI_menisci_pfp_v1"
)

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]

# Production pipeline params (from comak_1_nsm_fitting.py)
PRODUCTION_PARAMS = {
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

    # Meniscus centers from tibia labeled mesh
    tib_labeled = pv.read(str(geom / "tibia" / "tibia_labeled_mesh_updated.vtk"))
    tib_pts = np.array(tib_labeled.points)
    for side in ("med", "lat"):
        key = f"{side}_meniscus_center_binary"
        if key in tib_labeled.point_data:
            labels = np.array(tib_labeled.point_data[key])
            if np.any(labels == 1):
                data[f"{side}_center"] = tib_pts[labels == 1].mean(axis=0)
            else:
                data[f"{side}_center"] = None
        else:
            data[f"{side}_center"] = None

    return data


def compute_surface_metrics(scored_mesh, ray_mesh):
    """Compute detailed surface comparison metrics.

    Args:
        scored_mesh: The scored extraction surface
        ray_mesh: The ray-casting full pipeline surface

    Returns dict with:
        assd: Average Symmetric Surface Distance
        area_scored: Surface area of scored mesh
        area_ray: Surface area of ray mesh
        area_ratio: scored/ray area ratio
        scored_to_ray_mean: Mean distance from each scored point to nearest ray point
        scored_to_ray_max: Max distance from scored to ray (Hausdorff one-way)
        scored_to_ray_p95: 95th percentile distance scored→ray
        ray_to_scored_mean: Mean distance from each ray point to nearest scored point
        ray_to_scored_max: Max distance from ray to scored (Hausdorff one-way)
        ray_to_scored_p95: 95th percentile distance ray→scored
        hausdorff: Max of both one-way Hausdorff distances
    """
    from scipy.spatial import KDTree

    pts_scored = np.array(scored_mesh.points)
    pts_ray = np.array(ray_mesh.points)

    tree_scored = KDTree(pts_scored)
    tree_ray = KDTree(pts_ray)

    dists_scored_to_ray, _ = tree_ray.query(pts_scored)
    dists_ray_to_scored, _ = tree_scored.query(pts_ray)

    assd = (dists_scored_to_ray.mean() + dists_ray_to_scored.mean()) / 2

    area_scored = float(scored_mesh.area)
    area_ray = float(ray_mesh.area)

    return {
        "assd": assd,
        "area_scored": area_scored,
        "area_ray": area_ray,
        "area_ratio": area_scored / area_ray if area_ray > 0 else float("inf"),
        "scored_to_ray_mean": float(dists_scored_to_ray.mean()),
        "scored_to_ray_max": float(dists_scored_to_ray.max()),
        "scored_to_ray_p95": float(np.percentile(dists_scored_to_ray, 95)),
        "ray_to_scored_mean": float(dists_ray_to_scored.mean()),
        "ray_to_scored_max": float(dists_ray_to_scored.max()),
        "ray_to_scored_p95": float(np.percentile(dists_ray_to_scored, 95)),
        "hausdorff": float(max(dists_scored_to_ray.max(), dists_ray_to_scored.max())),
    }


def count_components(mesh):
    """Count connected components in a mesh."""
    if mesh.n_points == 0:
        return 0
    conn = mesh.connectivity(extraction_mode="all")
    if "RegionId" in conn.point_data:
        return int(conn.point_data["RegionId"].max()) + 1
    return 1


def scored_extraction_minimal(meniscus_mm, bone_mm, threshold):
    """Score-based extraction with minimal post-processing.

    Steps: compute scores → threshold → get_n_largest → remove_isolated_cells.
    No smoothing, no radial envelope.

    Returns (surface, n_components_before_filter, raw_n_points_before_filter).
    """
    # Compute normals and scores
    meniscus_mm.compute_normals(
        point_normals=True, auto_orient_normals=True, inplace=True
    )
    scores = score_meniscus_vertices(meniscus_mm, bone_mm)

    # Threshold
    mask = scores > threshold
    if not np.any(mask):
        return None, 0, 0

    surface = meniscus_mm.extract_points(mask, adjacent_cells=True)
    if not isinstance(surface, pv.PolyData):
        surface = surface.extract_surface()

    raw_n_points = surface.n_points
    n_components = count_components(surface)

    # Minimal cleanup: largest component + remove isolated
    surface = get_n_largest(surface, n=1)
    if not isinstance(surface, pv.PolyData):
        surface = surface.extract_surface()

    surface = remove_isolated_cells(surface)
    if not isinstance(surface, pv.PolyData):
        surface = surface.extract_surface()

    return surface, n_components, raw_n_points


def run_ray_casting_full(meniscus_m, femur_m, tibia_m, meniscus_center, theta_offset):
    """Run the full production pipeline with ray-casting."""
    from pymskt.mesh import Mesh

    upper, lower = create_meniscus_articulating_surface(
        meniscus_mesh=Mesh(mesh=meniscus_m.copy()),
        upper_articulating_bone_mesh=Mesh(mesh=femur_m.copy()),
        lower_articulating_bone_mesh=Mesh(mesh=tibia_m.copy()),
        meniscus_center=meniscus_center,
        theta_offset=theta_offset,
        extraction_method="ray_casting",
        **PRODUCTION_PARAMS,
    )
    # create_meniscus_articulating_surface returns pymskt Mesh objects — get the pyvista mesh
    upper_pv = upper.mesh if hasattr(upper, "mesh") else upper
    lower_pv = lower.mesh if hasattr(lower, "mesh") else lower
    return upper_pv, lower_pv


def run_scored_minimal(meniscus_m, femur_m, tibia_m):
    """Run scored extraction for upper and lower surfaces at all thresholds.

    Returns dict: {threshold: {"upper": (surface, n_comp, raw_n), "lower": (surface, n_comp, raw_n)}}
    """
    # Convert to mm (matching pipeline)
    men_mm = meniscus_m.copy()
    men_mm.points = men_mm.points * 1000
    fem_mm = femur_m.copy()
    fem_mm.points = fem_mm.points * 1000
    tib_mm = tibia_m.copy()
    tib_mm.points = tib_mm.points * 1000

    results = {}
    for threshold in THRESHOLDS:
        upper_result = scored_extraction_minimal(men_mm.copy(), fem_mm, threshold)
        lower_result = scored_extraction_minimal(men_mm.copy(), tib_mm, threshold)

        # Convert back to meters
        upper_surf, upper_nc, upper_rn = upper_result
        lower_surf, lower_nc, lower_rn = lower_result
        if upper_surf is not None:
            upper_surf.points = upper_surf.points / 1000
        if lower_surf is not None:
            lower_surf.points = lower_surf.points / 1000

        results[threshold] = {
            "upper": (upper_surf, upper_nc, upper_rn),
            "lower": (lower_surf, lower_nc, lower_rn),
        }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare scored (minimal cleanup) vs ray-casting (full pipeline)"
    )
    parser.add_argument("--n-subjects", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Finding {args.n_subjects} subjects...")
    subject_dirs = get_subject_dirs(args.n_subjects, seed=args.seed)
    print(f"Found {len(subject_dirs)} subjects\n")

    # Collect results: per threshold, per surface type → list of metrics dicts
    all_results = {t: {"upper": [], "lower": []} for t in THRESHOLDS}
    # Also collect ray-casting stats
    ray_stats = {"upper": [], "lower": []}

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

            # --- Full ray-casting pipeline ---
            try:
                ray_upper, ray_lower = run_ray_casting_full(
                    meniscus_m, data["femur"], data["tibia"], center, theta_offset
                )
            except Exception as e:
                print(f"  {men_name}: ray-casting FAILED: {e}")
                continue

            ray_upper_n = ray_upper.n_points if ray_upper is not None else 0
            ray_lower_n = ray_lower.n_points if ray_lower is not None else 0
            print(f"  {men_name} ray-casting: upper={ray_upper_n} pts, lower={ray_lower_n} pts")

            ray_stats["upper"].append(ray_upper_n)
            ray_stats["lower"].append(ray_lower_n)

            # --- Scored minimal at each threshold ---
            try:
                scored_results = run_scored_minimal(meniscus_m, data["femur"], data["tibia"])
            except Exception as e:
                print(f"  {men_name}: scored FAILED: {e}")
                continue

            for threshold in THRESHOLDS:
                sr = scored_results[threshold]

                for surf_type, ray_ref in [("upper", ray_upper), ("lower", ray_lower)]:
                    scored_surf, n_comp, raw_n = sr[surf_type]

                    if scored_surf is None or scored_surf.n_points == 0:
                        nan_entry = {
                            "subject": data["name"], "meniscus": men_name,
                            "n_components": n_comp,
                            "scored_n_pts": 0,
                            "ray_n_pts": ray_ref.n_points if ray_ref is not None else 0,
                        }
                        for k in ("assd_mm", "area_ratio", "area_scored_mm2", "area_ray_mm2",
                                  "scored_to_ray_mean_mm", "scored_to_ray_max_mm",
                                  "scored_to_ray_p95_mm", "ray_to_scored_mean_mm",
                                  "ray_to_scored_max_mm", "ray_to_scored_p95_mm",
                                  "hausdorff_mm"):
                            nan_entry[k] = float("nan")
                        all_results[threshold][surf_type].append(nan_entry)
                        continue

                    if ray_ref is None or ray_ref.n_points == 0:
                        continue

                    metrics = compute_surface_metrics(scored_surf, ray_ref)
                    # Convert distances to mm for reporting
                    entry = {
                        "subject": data["name"],
                        "meniscus": men_name,
                        "n_components": n_comp,
                        "scored_n_pts": scored_surf.n_points,
                        "ray_n_pts": ray_ref.n_points,
                        "assd_mm": metrics["assd"] * 1000,
                        "area_ratio": metrics["area_ratio"],
                        "area_scored_mm2": metrics["area_scored"] * 1e6,  # m² → mm²
                        "area_ray_mm2": metrics["area_ray"] * 1e6,
                        "scored_to_ray_mean_mm": metrics["scored_to_ray_mean"] * 1000,
                        "scored_to_ray_max_mm": metrics["scored_to_ray_max"] * 1000,
                        "scored_to_ray_p95_mm": metrics["scored_to_ray_p95"] * 1000,
                        "ray_to_scored_mean_mm": metrics["ray_to_scored_mean"] * 1000,
                        "ray_to_scored_max_mm": metrics["ray_to_scored_max"] * 1000,
                        "ray_to_scored_p95_mm": metrics["ray_to_scored_p95"] * 1000,
                        "hausdorff_mm": metrics["hausdorff"] * 1000,
                    }
                    all_results[threshold][surf_type].append(entry)

                # Print per-threshold summary for this meniscus
                u = scored_results[threshold]["upper"]
                l = scored_results[threshold]["lower"]
                u_n = u[0].n_points if u[0] is not None else 0
                l_n = l[0].n_points if l[0] is not None else 0
                print(
                    f"    t={threshold:.1f}: upper={u_n} pts ({u[1]} comp), "
                    f"lower={l_n} pts ({l[1]} comp)"
                )

        elapsed = time.time() - t0
        print(f"  ({elapsed:.1f}s)\n")

    # ========================================================================
    # Summary Report
    # ========================================================================
    print("=" * 120)
    print("COMPARISON: Scored (minimal cleanup) vs Ray-casting (full pipeline)")
    print("=" * 120)
    print(f"Subjects: {len(subject_dirs)}")
    print(f"Thresholds: {THRESHOLDS}")
    print()
    print("Scored pipeline:      threshold → get_n_largest → remove_isolated_cells")
    print("Ray-casting pipeline: ACVD resample → ray-cast → get_n_largest → remove_isolated")
    print("                      → smooth(10 iters) → radial envelope trim(p95)")
    print()

    for surf_type in ("upper", "lower"):
        print(f"--- {surf_type.upper()} articular surface ---")
        print(
            f"  {'Thresh':>6} | {'ASSD':>6} | {'S→R mean':>8} | {'S→R p95':>7} | {'S→R max':>7} | "
            f"{'R→S mean':>8} | {'R→S p95':>7} | {'R→S max':>7} | "
            f"{'Area ratio':>10} | {'1-comp%':>7} | {'N':>3}"
        )
        print(f"  {'':>6} | {'(mm)':>6} | {'(mm)':>8} | {'(mm)':>7} | {'(mm)':>7} | "
              f"{'(mm)':>8} | {'(mm)':>7} | {'(mm)':>7} | "
              f"{'scor/ray':>10} | {'':>7} | {'':>3}")
        print(f"  {'-' * 110}")

        for threshold in THRESHOLDS:
            entries = all_results[threshold][surf_type]
            valid = [e for e in entries if not np.isnan(e["assd_mm"])]
            if not valid:
                print(f"  {threshold:>6.1f} | {'N/A':>6}")
                continue

            def mean(key):
                return np.mean([e[key] for e in valid])

            comps = [e["n_components"] for e in valid]
            single_pct = sum(1 for c in comps if c == 1) / len(comps) * 100

            print(
                f"  {threshold:>6.1f} | {mean('assd_mm'):>6.3f} | "
                f"{mean('scored_to_ray_mean_mm'):>8.3f} | {mean('scored_to_ray_p95_mm'):>7.3f} | "
                f"{mean('scored_to_ray_max_mm'):>7.3f} | "
                f"{mean('ray_to_scored_mean_mm'):>8.3f} | {mean('ray_to_scored_p95_mm'):>7.3f} | "
                f"{mean('ray_to_scored_max_mm'):>7.3f} | "
                f"{mean('area_ratio'):>10.2f} | {single_pct:>6.0f}% | {len(valid):>3}"
            )

        print()

    # ========================================================================
    # Interpretation guide
    # ========================================================================
    print("=" * 120)
    print("METRIC INTERPRETATION")
    print("=" * 120)
    print("  S→R (scored→ray):  How far scored surface extends BEYOND ray-casting output.")
    print("                     High S→R = scored surface is bigger / has extra fringe.")
    print("  R→S (ray→scored):  How far ray-casting points are from scored surface.")
    print("                     High R→S = scored surface MISSES regions that ray-casting kept.")
    print("  Area ratio:        scored_area / ray_area. >1 = scored is bigger.")
    print("  1-comp%:           Percentage of surfaces that are a single connected component")
    print("                     before get_n_largest (higher = cleaner extraction).")
    print()

    # ========================================================================
    # Breakdown by meniscus type
    # ========================================================================
    print("=" * 120)
    print("BREAKDOWN BY MENISCUS TYPE")
    print("=" * 120)
    for threshold in THRESHOLDS:
        print(f"\n  threshold = {threshold}")
        for surf_type in ("upper", "lower"):
            entries = all_results[threshold][surf_type]
            for men_type in ("medial", "lateral"):
                subset = [e for e in entries if e["meniscus"] == men_type and not np.isnan(e["assd_mm"])]
                if not subset:
                    continue

                def smean(key):
                    return np.mean([e[key] for e in subset])

                comps = [e["n_components"] for e in subset]
                single_pct = sum(1 for c in comps if c == 1) / len(comps) * 100
                print(
                    f"    {men_type:>8} {surf_type:>5}: "
                    f"ASSD={smean('assd_mm'):.3f}mm, "
                    f"S→R={smean('scored_to_ray_mean_mm'):.3f}/{smean('scored_to_ray_p95_mm'):.3f}mm, "
                    f"R→S={smean('ray_to_scored_mean_mm'):.3f}/{smean('ray_to_scored_p95_mm'):.3f}mm, "
                    f"area={smean('area_ratio'):.2f}x, "
                    f"1-comp={single_pct:.0f}%, n={len(subset)}"
                )


if __name__ == "__main__":
    main()
