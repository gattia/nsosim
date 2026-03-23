# Meniscus Articular Surface Stability: Tests & Fix Plan

## Context

### The Problem
`create_meniscus_articulating_surface()` produces stochastic variation in extracted
articular surfaces — especially the **medial meniscus inferior (lower) surface**.
Across identical pipeline runs, this surface shows ~40% point count variation and
~0.45mm ASSD (vs <0.1mm for all other geometry outputs).

### Why It Matters
Meniscus articular surfaces are used as **contact geometry in COMAK simulations**.
The contact model casts rays to detect penetration, then computes spring-based forces.
If the extracted surface includes side/rim vertices (not flat contact faces), these
can intersect cartilage in unphysical ways, producing bogus forces and simulation
failures. **Only the flat top/bottom faces should be included.**

### Root Cause (3 compounding stochastic sources)
1. **NSM reconstruction** (marching cubes): different mesh topology each run
   (ASSD <0.005mm, but connectivity/vertex positions differ)
2. **ACVD resampling** (`resample_surface`): non-deterministic clustering initialization
3. **Ray-casting** (`remove_intersecting_vertices`): binary hit/miss per vertex,
   depends on normals which depend on local mesh connectivity. Near the boundary
   between flat face and rim, small normal perturbations flip the decision.

### Why med_men_lower specifically?
- Smallest extracted surface: ~322 pts (5.3% of full meniscus)
- Proportionally more boundary vertices than interior
- Boundary vertices are where ray-casting is most sensitive
- Small normal shifts flip many boundary vertices → large proportional change

### Key Constraint
We can **only** include points on the flat superior/inferior contact faces.
The inner rim, outer rim, and horn tips must be excluded. Any method that simply
splits the meniscus into "top half" and "bottom half" (e.g., pure distance-to-bone)
will include rim vertices and cause simulation failures.

---

## Current Architecture

### Call chain
```
create_meniscus_articulating_surface()          # articular_surfaces.py:498
  ├─ resample_surface(subdivisions=1, clusters=N)  # pymskt ACVD — NON-DETERMINISTIC
  ├─ extract_meniscus_articulating_surface()     # articular_surfaces.py:412
  │    ├─ remove_intersecting_vertices()          # pymskt — BINARY RAY-CASTING
  │    ├─ get_n_largest(n=1)                      # pymskt — largest connected component
  │    ├─ remove_isolated_cells()                 # pymskt — strip dangling triangles
  │    └─ smooth(n_iter=10)                       # PyVista Laplacian smoothing
  └─ refine_meniscus_articular_surfaces()        # articular_surfaces.py:324
       ├─ add_polar_coordinates_about_center()    # articular_surfaces.py:35
       ├─ label_meniscus_regions_with_sdf()       # articular_surfaces.py:74
       ├─ compute_region_radial_percentiles()     # articular_surfaces.py:120
       ├─ build_min_radial_envelope()             # articular_surfaces.py:197
       └─ trim_mesh_by_radial_envelope()          # articular_surfaces.py:295
```

### How `remove_intersecting_vertices` works (pymskt)
File: `pymskt/mesh/meshCartilage.py:8-62`

For each vertex of `mesh1` (meniscus):
1. Compute outward normal
2. Cast ray in **negative normal** direction (inward) for `ray_length` distance
3. Check if ray hits any triangle of `mesh2` (bone) via VTK OBBTree
4. If hit → vertex removed; if miss → vertex kept

In nsosim, called with `ray_length=-15.0` (sign flip), so rays go **outward** along
normal. Effect: keeps meniscus vertices whose outward normal points toward the bone.
This correctly selects flat faces (normals face bone) and excludes rim (normals face
sideways). But it's a binary decision — hit or miss — making it topology-sensitive.

### Production parameters (from comak_1_nsm_fitting.py)
```python
create_meniscus_articulating_surface(
    meniscus_mesh=fem_med_men_mesh_osim,       # from femur NSM model, surface_idx=2/3
    upper_articulating_bone_mesh=fem_mesh_osim,
    lower_articulating_bone_mesh=tib_mesh_osim,
    meniscus_center=med_meniscus_center,        # from tibia labeled mesh
    theta_offset=np.pi,      # medial: rotate polar coords to avoid discontinuity
    ray_length=15.0,          # mm
    n_largest=1,
    smooth_iter=10,
    boundary_smoothing=False,
    radial_percentile=95.0,
)
# Lateral uses theta_offset=0.0, otherwise same parameters
```

---

## Prior Art: What Was Already Tried (Notebooks, Nov 14-16, 2025)

Location: `/dataNAS/people/aagatti/projects/comak_gait_simulation/notebooks/simulations_failing_meniscus_edges/`

### Approaches that DID NOT work

| Approach | Notebook | Result | Why it failed |
|----------|----------|--------|---------------|
| **Cartilage penetration filtering** | Nov 14 | Removed 62% of points at 1mm threshold | Too destructive, not selective |
| **Curvature filtering** (max, Gaussian, mean) | Nov 14 | Distributions overlap | Articular vs rim curvature ranges overlap too much for clean separation |
| **Heavy smoothing + curvature diff** | Nov 14 | Visualized but inconclusive | Smoothing 10k iterations + curvature difference didn't cleanly separate |

### Approach that DID work: Radial Envelope Refinement
Developed across Nov 14-15 notebooks, validated Nov 16 on a second subject.

1. Extract raw surfaces via ray-casting (as before)
2. Convert meniscus to polar coords (theta, r) about meniscus center
3. Label meniscus points by SDF proximity to extracted surfaces (threshold 0.1mm)
4. Compute 95th percentile of r per theta bin per region
5. Smooth percentile curves, take element-wise minimum across regions
6. Trim surfaces: remove points where r > r_min(theta)

**This is what's currently implemented.** It stabilizes the boundary significantly
but cannot fully compensate because the envelope is computed FROM the stochastic
extracted surfaces — if the extraction varies, the envelope varies.

### Key finding: PCA normal-direction approach (Nov 14)
- Computed PCA on meniscus point cloud → smallest variance axis = thickness direction
- `|dot(cell_normal, thickness_dir)| >= 0.6` → articular face; < 0.6 → rim
- **This cleanly separated articular from rim on the full meniscus**
- Was NOT applied as the extraction method — only used for visualization/analysis

### Percentile sensitivity (Nov 16, subject 9683366)
| Percentile | Upper pts | Upper cells |
|------------|-----------|-------------|
| 90th | 1,013 | 1,835 |
| 93rd | 1,028 | 1,870 |
| 95th | 1,050 | 1,910 |
| 97th | 1,068 | 1,944 |
| 99th | 1,081 | 1,968 |

~7% variation from 90th to 99th — method is not highly sensitive to this parameter.

---

## Test Data

### Real meshes (copy into repo as test fixtures)
Source (subject 9683366 LEFT, from debug notebook):
- Medial meniscus: `.../OARSI_menisci_pfp_v1/9683366_00m_LEFT/geometries_nsm_similarity/femur/med_men_osim.stl` (7,550 pts)
- Lateral meniscus: `.../femur/lat_men_osim.stl` (6,600 pts)
- Femur bone: `.../femur/femur_nsm_recon_osim.stl` (20,000 pts)
- Tibia bone: `.../tibia/tibia_nsm_recon_osim.stl` (20,000 pts)
- Tibia labeled: `.../tibia/tibia_labeled_mesh_updated.vtk` (99,698 pts)

Full base path: `/dataNAS/people/aagatti/projects/comak_gait_simulation_results/OARSI_menisci_pfp_v1/9683366_00m_LEFT/geometries_nsm_similarity/`

**Decision:** Copy full-resolution meshes into `tests/fixtures/meniscus/`.
Mark integration tests as `@pytest.mark.slow`.

### Synthetic half-ring geometry
Create a C-shaped half-ring (meniscus-like) with:
- Outer radius ~15mm, inner radius ~8mm, height ~5mm
- Clear flat top/bottom faces and curved inner/outer rims
- Bone meshes as flat planes above and below

Use both: synthetic for unit/characterization tests, real meshes for integration tests.

---

## Part 1: Tests

### Branch: `menisci-articular-stability` (off `menisci`)

All new tests go in `tests/test_meniscus_stability.py` (new file).

### Category A: Unit tests for currently untested functions
These fill genuine coverage gaps and should pass with current code.

#### A1. `test_label_meniscus_regions_with_sdf`
**What:** Create a half-ring meniscus between two flat planes (upper bone, lower bone).
Extract upper/lower surfaces (known ground truth for the synthetic geometry).
Call `label_meniscus_regions_with_sdf()`.
**Assert:**
- Vertices near top face get label=2 (near upper surface)
- Vertices near bottom face get label=1 (near lower surface)
- Vertices near both (thin edges) get label=3
- Interior vertices far from both get label=0
- All labels are in {0, 1, 2, 3}

#### A2. `test_compute_region_radial_percentiles`
**What:** Create a mesh with known polar coordinates and region labels.
Set up so region 1 has r values in [3, 7] and region 2 has r values in [4, 8].
Call `compute_region_radial_percentiles(percentile=95.0)`.
**Assert:**
- Returns entries for regions 1 and 2 (not region 0)
- bin_centers are within expected theta range
- r_percentile values are between 95th percentile of the known r distributions
- Number of bins matches expectation

#### A3. `test_refine_meniscus_articular_surfaces_end_to_end`
**What:** Create a synthetic meniscus with known articular regions. Create initial
extracted surfaces that include deliberate outlier "tongue" extending beyond the
expected radial boundary.
Call `refine_meniscus_articular_surfaces()`.
**Assert:**
- Outlier tongue points are removed (not in trimmed surface)
- Core articular surface points are preserved
- Returned envelope (theta_grid, r_min_grid) has expected shape

#### A4. `test_full_pipeline_with_radial_refinement`
**What:** Upgrade existing smoke test to use `refine_by_radial_envelope=True`.
Use torus-between-spheres geometry (or half-ring if torus is too simple).
**Assert:**
- Both surfaces are non-None with >0 points
- Both surfaces are PolyData (not UnstructuredGrid)
- Surfaces are smaller than the full meniscus (refinement trimmed something)

### Category B: Characterization tests (expose the instability)
These should **FAIL with current ray-casting code** and **PASS after the fix**.

#### B1. `test_topology_perturbation_stability` (KEY TEST)
**What:** Create a half-ring meniscus and flat bone planes. Remesh the meniscus
twice using ACVD (`resample_surface`) to get two different topologies of the same
surface. Run `extract_meniscus_articulating_surface()` on both.
**Assert:**
- Extracted surfaces have ASSD < 0.1mm
- Point counts are within 15% of each other
**Why it should fail now:** Different ACVD topologies → different normals at
boundary → different ray-casting results → different extracted surfaces.

#### B2. `test_topology_perturbation_stability_with_refinement`
**What:** Same as B1 but run the full `create_meniscus_articulating_surface()`
with `refine_by_radial_envelope=True`.
**Assert:** Same ASSD and point count thresholds as B1.
**Why:** Tests whether the radial envelope compensates for extraction instability.

#### B3. `test_small_surface_proportional_stability`
**What:** Create geometry that produces both a large and small articular region
(mimicking lat_men_lower ~1000pts vs med_men_lower ~322pts). Apply ACVD
remeshing perturbation. Run extraction on both.
**Assert:**
- The small region is not disproportionately degraded
- Both regions meet the same relative ASSD threshold
**Why:** Validates that the fix addresses the size-dependent sensitivity.

#### B4. `test_no_rim_vertices_in_extraction` (GROUND TRUTH TEST)
**What:** Create a half-ring where we analytically know which vertices are on the
flat top/bottom faces vs the curved rim. The test geometry should have vertices
tagged with their ground-truth classification (top_face, bottom_face, inner_rim,
outer_rim).
Run extraction.
**Assert:**
- No rim-tagged vertices appear in the extracted surface
- At least 80% of flat-face interior vertices are included
- This is a correctness test, not just a stability test

#### B5. `test_determinism_same_inputs`
**What:** Run `extract_meniscus_articulating_surface()` twice with identical inputs
(no resampling).
**Assert:** Identical output (same points, same connectivity).
**Why:** Confirms the function is deterministic when given deterministic inputs.
This should PASS with current code (the stochasticity comes from upstream resampling).

### Category C: Integration tests with real data (marked `@pytest.mark.slow`)

#### C1. `test_real_meniscus_extraction_stability`
**What:** Load real meniscus + bone meshes (from test fixtures). Remesh meniscus
twice with ACVD. Run full extraction pipeline on both.
**Assert:** ASSD < 0.1mm, point count within 15%.
**Requires:** Mesh files in `tests/fixtures/meniscus/`.

#### C2. `test_real_meniscus_no_rim_points`
**What:** Load real meniscus. Run extraction. Verify no extracted points have
normals that are roughly perpendicular to the bone direction (rim check).
**Assert:** For all extracted vertices,
`dot(vertex_normal, direction_to_nearest_bone_point) > min_threshold` (e.g., 0.2).

---

## Part 2: Fix Methods (Ranked by Expected Effectiveness)

### Method 1: Normal Dot-Product Scoring (RECOMMENDED — replaces ray-casting)

**Concept:** Replace the binary ray-casting (`remove_intersecting_vertices`) with a
continuous scoring function. For each meniscus vertex:

```python
from scipy.spatial import KDTree

def score_meniscus_vertices(meniscus_mesh, bone_mesh):
    """Score each meniscus vertex by how much its normal faces the bone."""
    # 1. Compute outward normals on the meniscus
    meniscus_mesh.compute_normals(point_normals=True, auto_orient_normals=True, inplace=True)
    normals = meniscus_mesh.point_normals  # (N, 3)
    points = meniscus_mesh.points           # (N, 3)

    # 2. Find closest point on bone for each meniscus vertex
    bone_tree = KDTree(bone_mesh.points)
    distances, indices = bone_tree.query(points)
    closest_bone_pts = bone_mesh.points[indices]  # (N, 3)

    # 3. Compute direction from meniscus vertex to closest bone point
    direction = closest_bone_pts - points  # (N, 3)
    direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
    direction_norm[direction_norm == 0] = 1  # avoid div by zero
    direction = direction / direction_norm

    # 4. Score = dot(outward_normal, direction_to_bone)
    # High score → normal points toward bone → flat articular face
    # Low score → normal perpendicular to bone direction → rim/edge
    scores = np.sum(normals * direction, axis=1)  # (N,)

    return scores
```

**Why this is the best option:**
- **Continuous, not binary:** Small normal perturbations cause small score changes
  (0.29→0.31), not binary flips (hit→miss). The boundary in score-space is smooth.
- **Preserves flat-face selection:** Only vertices whose normals face the bone score
  high. Rim vertices have normals perpendicular to the bone direction → score ≈ 0.
- **Generalizes the PCA approach that already worked:** The Nov 14 notebook showed
  `|dot(normal, PCA_thickness_dir)| >= 0.6` cleanly separated articular from rim.
  This method replaces the global PCA direction with a per-vertex direction-to-bone,
  which adapts to local meniscus-bone curvature.
- **Can be smoothed on the surface:** After computing raw scores, apply Laplacian
  smoothing to the score field → even more stable classification boundary.
- **Self-correcting at bone edges:** If the meniscus vertex is near the bone edge
  (where bone is curving away), the direction-to-nearest-bone-point becomes more
  tangential → lower score → correctly marginal/excluded.

**Implementation sketch for `extract_meniscus_articulating_surface_scored()`:**
```python
def extract_meniscus_articulating_surface_scored(
    meniscus_mesh, bone_mesh, score_threshold=0.3, smooth_score=True,
    smooth_iterations=20, n_largest=1, smooth_iter=15, boundary_smoothing=False,
):
    # 1. Compute scores
    scores = score_meniscus_vertices(meniscus_mesh, bone_mesh)

    # 2. (Optional) Smooth scores on the mesh surface
    if smooth_score:
        meniscus_mesh["_scores"] = scores
        smoothed = meniscus_mesh.smooth(n_iter=smooth_iterations,
                                         boundary_smoothing=boundary_smoothing)
        # PyVista smooth interpolates point data too
        scores = smoothed["_scores"]

    # 3. Threshold
    mask = scores > score_threshold
    surface = meniscus_mesh.extract_points(mask, adjacent_cells=True)

    # 4. Standard cleanup: largest component, remove isolated cells, smooth
    surface = get_n_largest(surface.extract_surface(), n=n_largest)
    surface = remove_isolated_cells(surface)
    surface = surface.smooth(n_iter=smooth_iter, boundary_smoothing=boundary_smoothing)

    return Mesh(surface)
```

**Parameters to tune:**
- `score_threshold`: Start with 0.3 (from PCA experiment's 0.6 threshold on
  |dot|, but we use signed dot so 0.3 is comparable). Test range: 0.2–0.5.
- `smooth_iterations`: 10-50. More smoothing → more stable but may over-erode.
- Whether to use `smooth()` (Laplacian, moves vertices) or a custom score-only
  smoothing (averages score values without moving vertices).

**Risk:** PyVista's `smooth()` moves vertices AND interpolates point data. We want
to smooth only the score field, not the geometry. May need a custom neighbor-averaging
step instead. This is a known gotcha to watch for.

**Alternative for score smoothing (safer):**
```python
# Average scores over vertex neighbors without moving geometry
from scipy.sparse import csr_matrix
adj = meniscus_mesh.extract_surface().adjacency  # or build from faces
# Score_smoothed[i] = mean(scores[neighbors_of_i])
```

---

### Method 2: Smooth Normals Before Ray-Casting (quick fallback)

**Concept:** Keep the existing `remove_intersecting_vertices` but smooth the meniscus
mesh before computing normals. This averages out topology-dependent normal variation.

```python
# Before calling remove_intersecting_vertices:
meniscus_smoothed = meniscus_mesh.smooth(n_iter=20, boundary_smoothing=False)
# Copy smoothed normals back to original mesh, or just use smoothed mesh
```

**Pros:** ~3-line code change. Minimal risk of breaking anything.
**Cons:** Still binary hit/miss underneath. Smoothing may erode boundary inward
(boundary normals get pulled toward rim direction). But the user noted that
over-eroding is better than including edge artifacts.

**When to use:** If Method 1 proves tricky to tune, this is the fast fallback.

---

### Method 3: Morphological Opening on Extracted Region

**Concept:** After current ray-casting extraction, apply morphological opening:
1. Erode: remove boundary vertices (vertices with <N neighbors also in the set)
2. Dilate: grow back from stable interior

**Pros:** Doesn't change extraction at all — just smooths the boundary.
**Cons:** May remove legitimate narrow parts. Hard to control the erosion depth.
Doesn't address root cause.

**When to use:** As a supplementary cleanup, not as the primary fix.

---

### Method 4: Curvature-Based Pre-Filtering — RULED OUT

Already tested in Nov 14 notebook. Curvature distributions of articular vs rim
regions overlap too much. Maximum curvature, Gaussian curvature, and mean curvature
were all tested. None provided clean separation. **Do not revisit.**

---

## Implementation Order

1. **Create branch** `menisci-articular-stability` off `menisci`
2. **Set up test fixtures** — synthetic half-ring geometry helper + copy real meshes
3. **Write Category A tests** (coverage gaps) — should all pass now
4. **Write Category B tests** (characterization) — B5 should pass, B1-B4 should fail
5. **Run all tests** to confirm expected pass/fail pattern
6. **Implement Method 1** (`extract_meniscus_articulating_surface_scored`)
7. **Wire into `create_meniscus_articulating_surface`** as the extraction step
8. **Tune parameters** (score_threshold, smoothing) until B tests pass
9. **If Method 1 is difficult:** fall back to Method 2 (smooth normals)
10. **Run full test suite** to confirm no regressions
11. **Run pipeline verification** on real data
12. **Commit code changes, then autoformat separately**

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `tests/test_meniscus_stability.py` | CREATE | All new meniscus tests |
| `tests/fixtures/meniscus/` | CREATE (dir) | Real mesh test fixtures (full-resolution) |
| `nsosim/articular_surfaces.py` | MODIFY | Add scored extraction function, wire into pipeline |
| `tests/conftest.py` | MODIFY | Add shared meniscus test fixtures if needed |

---

## Gotchas & Notes

1. **PyVista `smooth()` moves vertices AND interpolates point data.** If we smooth
   to get better scores, we need to either (a) smooth only the score array via
   neighbor averaging, or (b) smooth a copy and transfer scores back to the
   unsmoothed mesh.

2. **`auto_orient_normals=True` may fail on open meshes.** The meniscus articular
   surface is an open surface — normals may not be consistently oriented. Need to
   verify this works or use a fallback normal computation.

3. **Units: the extraction happens in mm space** (after meters→mm conversion in
   `create_meniscus_articulating_surface`). The `score_threshold` is unitless
   (dot product), but `ray_length` and distance thresholds are in mm.

4. **The radial envelope refinement should still be kept** even with Method 1.
   It provides a second layer of boundary stabilization. The combination of
   continuous scoring + radial envelope should be very stable.

5. **ACVD resampling is still non-deterministic.** Even with a better extraction
   method, the input mesh will vary between runs. The fix makes extraction
   *robust to* this variation, not eliminate it.

6. **Commit discipline:** Commit code changes BEFORE running `make autoformat`.
   Autoformat touches the whole repo.

7. **Test environment:** `conda run -n comak python -m pytest tests/ -v`

8. **Real mesh fixture files** are from subject 9683366 LEFT. Full base path:
   `/dataNAS/people/aagatti/projects/comak_gait_simulation_results/OARSI_menisci_pfp_v1/9683366_00m_LEFT/geometries_nsm_similarity/`
   Files to copy:
   - `femur/med_men_osim.stl` (7,550 pts)
   - `femur/lat_men_osim.stl` (6,600 pts)
   - `femur/femur_nsm_recon_osim.stl` (20,000 pts)
   - `tibia/tibia_nsm_recon_osim.stl` (20,000 pts)
   - `tibia/tibia_labeled_mesh_updated.vtk` (99,698 pts)
