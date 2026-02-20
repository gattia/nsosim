#!/usr/bin/env python
"""Calibrate scored extraction threshold against ray-casting reference.

For each subject, loads meniscus + bone meshes, runs ray-casting ONCE to get the
reference binary mask, computes scores ONCE, then evaluates all thresholds via
simple numpy array operations.

The key insight: scores are deterministic for a given mesh, so we compute them
once and just sweep thresholds on the score array. No need to repeatedly call
the full extraction pipeline.

Also reports score distributions for ray-kept vs ray-removed vertices, which
directly shows where to set the threshold.

Usage:
    conda run -n comak python scripts/meniscus_extraction_analysis/calibrate_score_threshold.py
    conda run -n comak python scripts/meniscus_extraction_analysis/calibrate_score_threshold.py --n-subjects 20
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nsosim.articular_surfaces import (
    _smooth_scores_on_mesh,
    extract_meniscus_articulating_surface,
    score_meniscus_vertices,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================
RESULTS_ROOT = Path(
    "/dataNAS/people/aagatti/projects/comak_gait_simulation_results/OARSI_menisci_pfp_v1"
)

# Thresholds to sweep (fine grid)
THRESHOLDS = np.arange(0.0, 0.85, 0.025)
SMOOTH_ITERATIONS_LIST = [0, 5, 10, 20, 30]

# Pipeline params (match production: comak_1_nsm_fitting.py lines 628-652)
RAY_LENGTH = 15.0
N_LARGEST = 1
BOUNDARY_SMOOTHING = False


def get_subject_dirs(n_subjects, seed=42):
    """Get n_subjects random subject directories that have all required files."""
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
        ]
        if all(f.exists() for f in required):
            valid.append(d)
        if len(valid) >= n_subjects:
            break

    return valid


def load_subject_meshes_mm(subject_dir):
    """Load meniscus + bone meshes, convert to mm."""
    geom = subject_dir / "geometries_nsm_similarity"

    meshes = {}
    files = {
        "med_men": geom / "femur" / "med_men_osim.stl",
        "lat_men": geom / "femur" / "lat_men_osim.stl",
        "femur": geom / "femur" / "femur_nsm_recon_osim.stl",
        "tibia": geom / "tibia" / "tibia_nsm_recon_osim.stl",
    }
    for name, path in files.items():
        m = pv.read(str(path))
        m.points = m.points * 1000  # meters → mm
        meshes[name] = m

    return meshes


def get_ray_mask(meniscus_mesh_mm, bone_mesh_mm):
    """Run ray-casting extraction, map result back to binary mask on full mesh.

    Returns boolean mask over meniscus_mesh_mm.n_points, or None if extraction fails.
    """
    from scipy.spatial import KDTree

    try:
        ray_surface = extract_meniscus_articulating_surface(
            meniscus_mesh=meniscus_mesh_mm,
            articulating_bone_mesh=bone_mesh_mm,
            ray_length=RAY_LENGTH,
            n_largest=N_LARGEST,
            smooth_iter=0,  # No post-smoothing — raw extraction
            boundary_smoothing=BOUNDARY_SMOOTHING,
        )
    except Exception as e:
        logger.warning(f"Ray-casting failed: {e}")
        return None

    if ray_surface.n_points == 0:
        return None

    # Map extracted vertices back to original mesh indices
    tree = KDTree(meniscus_mesh_mm.points)
    dists, indices = tree.query(ray_surface.points)

    mask = np.zeros(meniscus_mesh_mm.n_points, dtype=bool)
    # Use tolerance — vertices should match exactly since they came from same mesh
    valid = dists < 0.01  # 0.01mm tolerance
    mask[indices[valid]] = True
    return mask


def get_scores(meniscus_mesh_mm, bone_mesh_mm):
    """Compute raw per-vertex scores. Returns 1D array of scores in [-1, 1]."""
    meniscus_mesh_mm.compute_normals(
        point_normals=True, auto_orient_normals=True, inplace=True
    )
    return score_meniscus_vertices(meniscus_mesh_mm, bone_mesh_mm)


def analyze_one_surface(meniscus_mm, bone_mm, surface_name):
    """Analyze one meniscus surface: compute scores, get ray mask, compare.

    Returns dict with score distributions, per-threshold metrics, and raw arrays.
    """
    # 1. Ray-casting reference (once)
    ray_mask = get_ray_mask(meniscus_mm, bone_mm)
    if ray_mask is None:
        return None

    n_total = meniscus_mm.n_points
    n_ray_kept = int(np.sum(ray_mask))

    # 2. Compute raw scores (once)
    raw_scores = get_scores(meniscus_mm, bone_mm)

    # 3. Score distributions for kept vs removed
    scores_kept = raw_scores[ray_mask]
    scores_removed = raw_scores[~ray_mask]

    result = {
        "surface_name": surface_name,
        "n_total": n_total,
        "n_ray_kept": n_ray_kept,
        "n_ray_removed": n_total - n_ray_kept,
        "scores_kept": scores_kept,
        "scores_removed": scores_removed,
        "raw_scores": {
            "kept_mean": float(np.mean(scores_kept)),
            "kept_std": float(np.std(scores_kept)),
            "kept_median": float(np.median(scores_kept)),
            "kept_p5": float(np.percentile(scores_kept, 5)),
            "kept_p25": float(np.percentile(scores_kept, 25)),
            "removed_mean": float(np.mean(scores_removed)) if len(scores_removed) > 0 else 0,
            "removed_std": float(np.std(scores_removed)) if len(scores_removed) > 0 else 0,
            "removed_median": float(np.median(scores_removed)) if len(scores_removed) > 0 else 0,
            "removed_p75": float(np.percentile(scores_removed, 75)) if len(scores_removed) > 0 else 0,
            "removed_p95": float(np.percentile(scores_removed, 95)) if len(scores_removed) > 0 else 0,
        },
        "thresholds": {},  # keyed by (threshold, smooth_iters)
    }

    # 4. For each smooth_iterations variant, smooth scores then sweep thresholds
    for smooth_iters in SMOOTH_ITERATIONS_LIST:
        if smooth_iters == 0:
            scores = raw_scores
        else:
            scores = _smooth_scores_on_mesh(meniscus_mm, raw_scores.copy(), n_iter=smooth_iters)

        for threshold in THRESHOLDS:
            scored_mask = scores > threshold

            # Compute agreement metrics
            tp = np.sum(ray_mask & scored_mask)  # both keep
            fp = np.sum(~ray_mask & scored_mask)  # scored keeps, ray removes
            fn = np.sum(ray_mask & ~scored_mask)  # ray keeps, scored removes
            tn = np.sum(~ray_mask & ~scored_mask)  # both remove

            union = tp + fp + fn
            iou = float(tp / union) if union > 0 else 0.0
            dice = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

            result["thresholds"][(threshold, smooth_iters)] = {
                "iou": iou,
                "dice": dice,
                "precision": precision,
                "recall": recall,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "n_scored_kept": int(np.sum(scored_mask)),
            }

    return result


def main():
    parser = argparse.ArgumentParser(description="Calibrate scored extraction threshold")
    parser.add_argument("--n-subjects", type=int, default=10, help="Number of subjects")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subject selection")
    args = parser.parse_args()

    print(f"Finding {args.n_subjects} subjects with complete data...")
    subject_dirs = get_subject_dirs(args.n_subjects, seed=args.seed)
    print(f"Found {len(subject_dirs)} subjects")

    if len(subject_dirs) == 0:
        print("ERROR: No valid subjects found")
        sys.exit(1)

    # Collect per-threshold metrics across all surfaces
    all_metrics = {}
    for s in SMOOTH_ITERATIONS_LIST:
        for t in THRESHOLDS:
            all_metrics[(t, s)] = {"iou": [], "dice": [], "precision": [], "recall": []}

    # Collect score distributions
    all_kept_scores = []
    all_removed_scores = []
    n_surfaces_total = 0

    for i, subject_dir in enumerate(subject_dirs):
        t0 = time.time()
        print(f"\n[{i+1}/{len(subject_dirs)}] {subject_dir.name}")

        try:
            meshes = load_subject_meshes_mm(subject_dir)
        except Exception as e:
            print(f"  SKIP: Failed to load: {e}")
            continue

        # 4 surfaces per subject: med×{upper,lower}, lat×{upper,lower}
        surface_configs = [
            ("med_upper", meshes["med_men"], meshes["femur"]),
            ("med_lower", meshes["med_men"], meshes["tibia"]),
            ("lat_upper", meshes["lat_men"], meshes["femur"]),
            ("lat_lower", meshes["lat_men"], meshes["tibia"]),
        ]

        for surface_name, men_mesh, bone_mesh in surface_configs:
            result = analyze_one_surface(men_mesh.copy(), bone_mesh, surface_name)
            if result is None:
                print(f"  {surface_name}: FAILED (ray-casting returned empty)")
                continue

            n_surfaces_total += 1
            raw = result["raw_scores"]
            print(
                f"  {surface_name}: {result['n_ray_kept']}/{result['n_total']} ray-kept | "
                f"scores kept=[{raw['kept_p5']:.3f}..{raw['kept_median']:.3f}..{raw['kept_mean']:.3f}] "
                f"removed=[{raw['removed_median']:.3f}..{raw['removed_p95']:.3f}]"
            )

            # Collect raw scores for global distribution (already computed)
            all_kept_scores.extend(result["scores_kept"].tolist())
            all_removed_scores.extend(result["scores_removed"].tolist())

            # Collect threshold metrics
            for key, vals in result["thresholds"].items():
                all_metrics[key]["iou"].append(vals["iou"])
                all_metrics[key]["dice"].append(vals["dice"])
                all_metrics[key]["precision"].append(vals["precision"])
                all_metrics[key]["recall"].append(vals["recall"])

        elapsed = time.time() - t0
        print(f"  ({elapsed:.1f}s)")

    # ========================================================================
    # Report: Score Distributions
    # ========================================================================
    print("\n" + "=" * 90)
    print("SCORE DISTRIBUTIONS (raw, unsmoothed)")
    print("=" * 90)
    kept = np.array(all_kept_scores)
    removed = np.array(all_removed_scores)
    print(f"  Surfaces analyzed: {n_surfaces_total}")
    print(f"  Total vertices: {len(kept) + len(removed)}")
    print(f"  Ray-kept vertices:    {len(kept)}")
    print(f"  Ray-removed vertices: {len(removed)}")
    print()
    print(f"  {'':>20} {'Mean':>7} {'Std':>7} {'P5':>7} {'P25':>7} {'Median':>7} {'P75':>7} {'P95':>7}")
    print(f"  {'Ray-kept':>20} {kept.mean():>7.3f} {kept.std():>7.3f} "
          f"{np.percentile(kept, 5):>7.3f} {np.percentile(kept, 25):>7.3f} "
          f"{np.percentile(kept, 50):>7.3f} {np.percentile(kept, 75):>7.3f} "
          f"{np.percentile(kept, 95):>7.3f}")
    if len(removed) > 0:
        print(f"  {'Ray-removed':>20} {removed.mean():>7.3f} {removed.std():>7.3f} "
              f"{np.percentile(removed, 5):>7.3f} {np.percentile(removed, 25):>7.3f} "
              f"{np.percentile(removed, 50):>7.3f} {np.percentile(removed, 75):>7.3f} "
              f"{np.percentile(removed, 95):>7.3f}")

    # ========================================================================
    # Report: Threshold Sweep (best smooth_iters for each threshold)
    # ========================================================================
    print("\n" + "=" * 90)
    print("THRESHOLD SWEEP (mean across all surfaces)")
    print("=" * 90)

    # Print compact table: for each smooth_iters, show IoU at each threshold
    for smooth_iters in SMOOTH_ITERATIONS_LIST:
        print(f"\n  smooth_iterations = {smooth_iters}")
        print(f"  {'Thresh':>6} | {'IoU':>6} | {'Dice':>6} | {'Prec':>6} | {'Recall':>6}")
        print(f"  {'-'*45}")
        for threshold in THRESHOLDS:
            key = (threshold, smooth_iters)
            m = all_metrics[key]
            if len(m["iou"]) == 0:
                continue
            print(
                f"  {threshold:>6.3f} | {np.mean(m['iou']):>6.3f} | {np.mean(m['dice']):>6.3f} | "
                f"{np.mean(m['precision']):>6.3f} | {np.mean(m['recall']):>6.3f}"
            )

    # ========================================================================
    # Report: Top configurations
    # ========================================================================
    print("\n" + "=" * 90)
    print("TOP 10 CONFIGURATIONS (by mean IoU)")
    print("=" * 90)
    ranked = sorted(
        [(k, np.mean(v["iou"])) for k, v in all_metrics.items() if len(v["iou"]) > 0],
        key=lambda x: -x[1],
    )
    print(f"  {'Rank':>4} | {'Thresh':>6} | {'Smooth':>6} | {'IoU':>6} | {'Dice':>6} | "
          f"{'Prec':>6} | {'Recall':>6} | {'Min IoU':>7}")
    print(f"  {'-'*65}")
    for rank, (key, mean_iou) in enumerate(ranked[:10], 1):
        t, s = key
        m = all_metrics[key]
        print(
            f"  {rank:>4} | {t:>6.3f} | {s:>6d} | {np.mean(m['iou']):>6.3f} | "
            f"{np.mean(m['dice']):>6.3f} | {np.mean(m['precision']):>6.3f} | "
            f"{np.mean(m['recall']):>6.3f} | {np.min(m['iou']):>7.3f}"
        )

    best_key = ranked[0][0]
    print(f"\n  RECOMMENDED: threshold={best_key[0]:.3f}, smooth_iterations={best_key[1]}")


if __name__ == "__main__":
    main()
