# Meniscus Extraction Method Analysis

Comparison of **scored extraction** vs **ray-casting** for meniscus articular surface extraction in `create_meniscus_articulating_surface()`.

Motivated by the instability documented in [`MENISCUS_ARTICULAR_SURFACE_INSTABILITY.md`](../../MENISCUS_ARTICULAR_SURFACE_INSTABILITY.md) — investigating whether a continuous scoring approach could replace the binary ray-casting method.

## Report

[SCORED_VS_RAYCASTING_ANALYSIS.md](SCORED_VS_RAYCASTING_ANALYSIS.md) — full analysis with score distributions, threshold calibration, and conclusions.

## Scripts

All scripts run on production data in `OARSI_menisci_pfp_v1` (780 subjects). Run from the repo root.

| Script | Purpose |
|--------|---------|
| `calibrate_score_threshold.py` | Computes per-vertex scores once, runs ray-casting once, sweeps thresholds via numpy. Reports score distributions (kept vs removed) and IoU/Dice/Precision/Recall at each threshold. |
| `compare_extraction_quality.py` | Scored (minimal cleanup) vs ray-casting (full pipeline). Reports ASSD, directed distances, area ratios, Hausdorff distance. |
| `compare_full_pipeline.py` | Both methods through the identical full pipeline (ACVD → extract → get_n_largest → remove_isolated → smooth → radial trim). Shows that the pipeline is tuned for ray-casting. |

### Usage

```bash
# Threshold calibration (default 10 subjects)
conda run -n comak python scripts/meniscus_extraction_analysis/calibrate_score_threshold.py

# With more subjects
conda run -n comak python scripts/meniscus_extraction_analysis/calibrate_score_threshold.py --n-subjects 20

# Minimal cleanup comparison
conda run -n comak python scripts/meniscus_extraction_analysis/compare_extraction_quality.py --n-subjects 10

# Full pipeline comparison
conda run -n comak python scripts/meniscus_extraction_analysis/compare_full_pipeline.py --n-subjects 10
```

## Key Finding

Ray-casting remains the recommended extraction method. The scored method captures the same geometric signal (IoU = 0.941 at threshold 0.5) but the full post-processing pipeline is co-designed with ray-casting's selection characteristics. See the report for details.
