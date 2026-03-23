# Meniscus Articular Surface Stability — COMPLETED

**Date:** 2026-02-20
**Branch:** `menisci` (merged to `main`)
**Key commits:** `6460e2c` (tests + fixtures), `e6026f9` (scored extraction), `2748d84` (analysis + docstring fix)

---

## What was done vs. the plan

### Test Fixtures — DONE
All 5 real meshes copied to `tests/fixtures/meniscus/` from subject 9683366 LEFT:
- `femur_nsm_recon_osim.stl`, `tibia_nsm_recon_osim.stl`
- `med_men_osim.stl`, `lat_men_osim.stl`
- `tibia_labeled_mesh_updated.vtk`

### Category A: Unit Tests — DONE (4/4)

| Test | Class | Status |
|------|-------|--------|
| A1. Label meniscus regions with SDF | `TestLabelMeniscusRegionsWithSDF` | Done (4 tests) |
| A2. Compute region radial percentiles | `TestComputeRegionRadialPercentiles` | Done (3 tests) |
| A3. Refine end-to-end | `TestRefineEndToEnd` | Done |
| A4. Full pipeline with refinement | `TestFullPipelineWithRefinement` | Done |

### Category B: Characterization Tests — DONE (4/5)

| Test | Class | Status |
|------|-------|--------|
| B1. Topology perturbation stability | `TestTopologyPerturbationStability` | Done |
| B2. Topology perturbation with refinement | `TestTopologyPerturbationWithRefinement` | Done |
| B3. Small surface proportional stability | — | Skipped — covered by real-data tests (C1) which test actual small/large surfaces |
| B4. No rim vertices in extraction | `TestNoRimVerticesInExtraction` | Done (upper + lower) |
| B5. Determinism same inputs | `TestDeterminismSameInputs` | Done |

### Category C: Integration Tests — DONE (2/2)

| Test | Class | Status |
|------|-------|--------|
| C1. Real meniscus extraction stability | `TestRealMeniscusExtractionStability` | Done (upper + lower) |
| C2. Real meniscus no rim points | `TestRealMeniscusNoRimPoints` | Done |

### Part 2: Fix — Method 1 (Scored Extraction) — DONE

| Planned | Done | Notes |
|---------|------|-------|
| `score_meniscus_vertices()` | Yes | `articular_surfaces.py:460` — dot-product scoring |
| `extract_meniscus_articulating_surface_scored()` | Yes | `articular_surfaces.py:496` — full scored pipeline |
| Wire into `create_meniscus_articulating_surface()` | Yes | `extraction_method` param, default `"ray_casting"` for backwards compat |
| Keep radial envelope as second layer | Yes | Refinement still runs after either extraction method |
| Parameter tuning | Yes | Analysis scripts in `scripts/meniscus_extraction_analysis/` |

### Additional work (not in original plan)
- `scripts/meniscus_extraction_analysis/` — calibration and comparison scripts for scored vs ray-casting
- `SCORED_VS_RAYCASTING_ANALYSIS.md` — documents threshold sensitivity and method comparison
- `MENISCUS_ARTICULAR_SURFACE_INSTABILITY.md` — root cause analysis document

---

## Summary

17 tests added in `tests/test_meniscus_stability.py` (804 lines). Scored extraction implemented as an alternative to ray-casting, selectable via `extraction_method` parameter. Default remains `"ray_casting"` so existing pipelines are unaffected.
