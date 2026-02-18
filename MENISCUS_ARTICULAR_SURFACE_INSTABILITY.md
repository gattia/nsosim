# Meniscus Articular Surface Extraction Instability

**Date:** 2026-02-13
**Affected function:** `nsosim.articular_surfaces.create_meniscus_articulating_surface()`
**Severity:** Moderate — surfaces are used as contact geometry in COMAK simulations
**Discovered during:** Pipeline verification testing (comak_gait_simulation)

## Summary

The medial meniscus **inferior (lower) articular surface** exhibits significant stochastic instability between pipeline runs. Running the same subject (9018389_RIGHT) twice with identical code and parameters produces surfaces with **40% different point counts** and **0.45mm ASSD** — well above the 0.1mm tolerance that works for all other geometry outputs. The full meniscus mesh is nearly identical (ASSD 0.002mm); the instability is isolated to the articular surface *extraction*.

## Quantitative Evidence

### Surface-level comparison across two identical pipeline runs

| Surface | Ref pts | New pts | Change | ASSD (mm) | Full meniscus ASSD |
|---------|---------|---------|--------|-----------|-------------------|
| **med_men_lower** | **322** | **451** | **+40%** | **0.455** | 0.002 |
| med_men_upper | 386 | 350 | -9% | 0.051 | 0.002 |
| lat_men_lower | 1002 | 1069 | +7% | 0.082 | 0.002 |
| lat_men_upper | 1090 | 1118 | +3% | 0.038 | 0.002 |

### Directional distance analysis (med_men_lower, in mm)

| Direction | Mean | Median | P95 | P99 | Max | Pts >1mm |
|-----------|------|--------|-----|-----|-----|----------|
| Ref → New | 0.236 | 0.221 | 0.472 | 0.766 | 1.335 | 1/322 (0.3%) |
| **New → Ref** | **0.910** | **0.311** | **4.040** | **4.853** | **5.147** | **110/451 (24.4%)** |

The asymmetry reveals the problem: every reference point has a nearby new point (Ref→New is fine), but **110 new points (24.4%) are >1mm from any reference point** — the new surface has a large region that doesn't exist in the reference.

### Spatial location of outlier points

| Axis | Ref range (mm) | New range | Outlier range | Interpretation |
|------|---------------|-----------|---------------|----------------|
| Z | [-30.2, -8.7] | [-29.8, -4.8] | [-14.6, -4.8] | 78 outliers extend **3.9mm beyond** ref Z-max |
| Y | [-22.6, -17.8] | [-22.6, -16.7] | [-20.8, -16.7] | 15 outliers extend **1.1mm beyond** ref Y-max |

The outlier region is a contiguous extension at one end of the surface — not scattered noise. The extraction boundary has shifted, causing a tongue of extra surface to be included.

## Root Cause Analysis

### Call chain

1. `comak_1_nsm_fitting.py:628` calls `create_meniscus_articulating_surface()`
2. `articular_surfaces.py:568-569` — **Resamples** meniscus mesh (stochastic: clustering-based)
3. `articular_surfaces.py:594-601` — **Extracts** lower surface via `extract_meniscus_articulating_surface()`
4. `articular_surfaces.py:428-432` — calls `remove_intersecting_vertices(mesh1=meniscus, mesh2=tibia, ray_length=-15.0)`
5. `pymskt/mesh/meshCartilage.py:8-43` — **Ray-casting**: for each meniscus vertex, cast ray along surface normal toward tibia; vertices whose rays hit tibia are the articular surface
6. `articular_surfaces.py:606-621` — **Radial envelope refinement** trims surface using 95th percentile radial envelope

### Three compounding sources of stochasticity

1. **NSM reconstruction** (marching cubes): Different mesh topology every run, even though the surface geometry is nearly identical (ASSD <0.005mm). Point counts, connectivity, and vertex positions all vary.

2. **Mesh resampling** (`resample_surface` at line 569): Uses clustering-based decimation, which is itself non-deterministic. Produces different vertex positions and connectivity from the same input mesh.

3. **Ray-casting on different topology** (`remove_intersecting_vertices`): Point normals are computed from local mesh connectivity (neighboring faces). Different topology → different normals → different ray directions → different intersection results. Near the boundary of the articular surface, small normal perturbations flip the ray intersection decision for/against individual vertices.

### Why med_men_lower specifically?

- It is the **smallest** extracted surface: only 5.3% of the full meniscus (322/6036 pts). Compare to lat_men_lower at 14.7% (1002/6814).
- Smaller surfaces have proportionally more boundary vertices relative to interior vertices.
- Boundary vertices are where the extraction is most sensitive — they're at the threshold where rays barely intersect or miss the tibia.
- A small shift in normals or tibia position flips many boundary vertices, causing a large proportional change in total surface area.
- The radial envelope refinement (95th percentile) helps stabilize the boundary but cannot fully compensate when the initial extraction varies this much.

## Relevant Code Locations

| Component | File | Lines | Function |
|-----------|------|-------|----------|
| Pipeline call | `comak_gait_simulation/.../comak_1_nsm_fitting.py` | 628-653 | (main body) |
| Orchestrator | `nsosim/articular_surfaces.py` | 471-655 | `create_meniscus_articulating_surface()` |
| Resampling | `nsosim/articular_surfaces.py` | 568-569 | `meniscus_mesh_.resample_surface(...)` |
| Core extraction | `nsosim/articular_surfaces.py` | 389-469 | `extract_meniscus_articulating_surface()` |
| Ray-casting | `pymskt/mesh/meshCartilage.py` | 8-43 | `remove_intersecting_vertices()` |
| Radial refinement | `nsosim/articular_surfaces.py` | 306-382 | `refine_meniscus_articular_surfaces()` |
| Radial trimming | `nsosim/articular_surfaces.py` | 270-303 | `trim_mesh_by_radial_envelope()` |

## Parameters Used (from pipeline)

```python
create_meniscus_articulating_surface(
    meniscus_mesh=fem_med_men_mesh_osim,
    upper_articulating_bone_mesh=fem_mesh_osim,
    lower_articulating_bone_mesh=tib_mesh_osim,
    meniscus_center=med_meniscus_center,
    theta_offset=np.pi,          # Medial: rotate to avoid polar discontinuity
    ray_length=15.0,             # mm (passed as -15.0 to remove_intersecting_vertices)
    n_largest=1,
    smooth_iter=10,
    boundary_smoothing=False,
    radial_percentile=95.0,
)
```

## Confirmed NOT the Cause

- **Config system changes (Parts 3-4):** All 41 pipeline parameters verified identical before and after config refactoring. Pure refactoring, no value changes.
- **Unit handling:** All `_osim` files correctly detected as meters, ASSD computation verified: native meters × 1000 = mm.
- **pymskt ASSD computation:** Manual vertex-based ASSD matches pymskt surface-based ASSD in direction (pymskt uses surface-to-surface which is more accurate for coarse meshes; values are consistent).

## Confounding Factor: Reference Data Regeneration

The production results (`OARSI_menisci_pfp_v1`) were generated with the *original* reference meshes in `COMAK_SIMULATION_REQUIREMENTS/nsm_meshes/`. During Part 2 reproducibility verification (2026-02-05), the preprocessing scripts were re-run and the regenerated data **overwrote the originals** (backups were deleted after verification passed). The regenerated reference meshes differ slightly from the originals (e.g., `nsm_recon_ref_femur_med_men.vtk`: 5418→5476 pts, ASSD 0.041mm). This means the pipeline verification is not a pure same-inputs comparison — part of the measured 0.455mm ASSD includes drift from different reference data, not just extraction stochasticity. However, the extraction boundary instability remains the primary issue: the other meniscus surfaces use the same regenerated reference data and stay well within tolerance.

## Potential Fixes (Discussion)

### 1. Stabilize the extraction boundary
- Use a **distance-to-bone** criterion instead of ray-casting. Signed distance fields are less sensitive to normal perturbations than individual ray directions.
- Compute SDF from meniscus vertices to bone surface and threshold — deterministic for a given pair of meshes, less dependent on local normal direction.

### 2. Seed resampling for reproducibility
- If `resample_surface` uses random clustering initialization, set a fixed seed.
- This would eliminate source #2 (resampling stochasticity) but not #1 (NSM stochasticity).

### 3. Tighten the radial envelope
- Lower `radial_percentile` from 95.0 to e.g. 90.0 for medial meniscus inferior surface.
- More aggressive trimming would clip the unstable boundary region.
- Risk: may remove legitimate articular surface for some subjects.

### 4. Post-hoc boundary regularization
- After extraction, project the boundary onto a smooth curve (e.g., spline fit in polar coordinates).
- Would produce consistent boundary shape regardless of which specific vertices are selected.

### 5. Accept and document
- Increase ASSD tolerance for meniscus articular surfaces to 0.5–1.0mm in verification.
- The full meniscus mesh (ASSD 0.002mm) already validates the underlying geometry.
- Articular surfaces are used for contact — small boundary shifts may have negligible biomechanical impact.

## Reproduction

```bash
cd /dataNAS/people/aagatti/projects/comak_gait_simulation

# Run pipeline twice on the same subject
./tests/verify_pipeline/submit_verification.sh

# Compare against production reference
python tests/verify_pipeline/verify_pipeline.py

# The med_men_lower_art_surf_osim files will show ASSD ~0.3-0.5mm
# while all other geometry files show ASSD <0.1mm
```

## Related Notebooks

- `comak_gait_simulation/notebooks/simulations_failing_meniscus_edges/improve_meniscus_sup_inf_extraction_Nov.15.2025_2.ipynb`
- `comak_gait_simulation/notebooks/simulations_failing_meniscus_edges/debug_meniscus_extraction_9683366_Nov.16.2025.ipynb`
