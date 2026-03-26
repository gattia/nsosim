# Plan: Synthetic Joint Decoding — Transform Utilities + Decode-from-Latent API

**Created:** 2026-03-25
**Status:** Ready to implement
**Context:** The comak_gait_simulation project needs to generate synthetic knee joints from arbitrary latent vectors and joint poses (for Paper 1, Analysis #8). The core decode + transform logic belongs in nsosim as reusable library functionality.
**Parent plan:** `comak_gait_simulation/.claude/plans/SYNTHETIC_JOINT_SIMULATION.md` — describes the full end-to-end pipeline; this plan covers only the nsosim-side work.

---

## Motivation

Currently, nsosim supports:
- **Fitting** subject meshes → latent vectors + transforms (`align_bone_osim_fit_nsm`, `reconstruct_mesh`)
- **Converting** fitted results to OSIM space (`nsm_recon_to_osim`, `convert_nsm_recon_to_OSIM`)
- **Interpolating** reference points through latent space (`interp_ref_to_subject_to_osim`)

What's missing:
- **Decoding** arbitrary latent vectors (not from fitting) to OSIM-space meshes
- **Transform utilities** for analyzing and reconstructing the per-bone alignment transforms (decompose, compute relative transforms, mean rotation, recompose from deviations)

These are general-purpose capabilities needed by any downstream project that wants to work with the NSM latent space directly (synthetic joints, shape mode visualization, latent interpolation, etc.).

---

## Understanding of the Transform Chain (TO BE VERIFIED)

**IMPORTANT: The transform chain described below was reverse-engineered by reading the code. Before implementing, verify this understanding is correct by running the validation tests described in Phase C. If any step is wrong, update this document and the CLAUDE.md documentation accordingly.**

### Coordinate spaces

1. **NSM canonical space** — the decoder's output space. Unit cube ~[-1, 1]. Each bone has its own canonical space (femur canonical ≠ tibia canonical).
2. **Femur-aligned space** — mm, NSM-oriented (rotated from OSIM by `OSIM_TO_NSM_TRANSFORM`), centered at the femur reference center. All bones share this space after the femur registration is applied.
3. **OSIM space** — meters, OSIM coordinate orientation. The final output space for OpenSim models.

### Per-bone `linear_transform` (from alignment JSONs)

Each bone's alignment JSON stores a 4x4 similarity `linear_transform` that maps **femur-aligned space → that bone's NSM canonical space**. Both `scale` and `center` fields are always `1` and `[0,0,0]` — the similarity ICP embeds centering + scaling + rotation into the 4x4 matrix.

```
Femur:   scale factor (column norm of 3x3) ≈ 0.013  (nearly identical across subjects)
Tibia:   scale factor ≈ 0.019  (varies — encodes joint configuration)
Patella: scale factor ≈ 0.031  (varies — patella is smaller, needs larger scale)
```

The scale factors reflect the mm → [-1,1] canonical range mapping. Tibia/patella transforms vary across subjects because they encode where each bone sits relative to the femur (joint configuration), not just the NSM registration. The 3x3 submatrix = `scale * R` where R is a near-identity rotation (bones are roughly aligned before ICP).

### Production pipeline: subject mesh → OSIM

```
Subject MRI mesh (mm, arbitrary orientation)
  → T_fem_reg: similarity registration to reference femur (subject_bone.rigidly_register)
  → [For tibia/patella: apply T_fem_reg instead of their own registration]
  → Save aligned VTK in femur-aligned space

  → reconstruct_mesh() [NSM library]:
      internally: center → scale → ICP to mean NSM shape → optimize latent → decode
      → scale_mesh_() reverses: ×scale, +offset, inverse ICP
      → returns mesh in aligned input space (NOT canonical space)

  → nsm_recon_to_osim() → convert_nsm_recon_to_OSIM_():
      points_ += fem_ref_center    # un-center (femur's mean_orig, used for ALL bones)
      points_ /= 1000              # mm → m
      points_ @ OSIM_TO_NSM_TRANSFORM.T  # rotate to OSIM orientation
  → OSIM space (meters)
```

Key: `reconstruct_mesh` returns meshes **already transformed back to the aligned input space** (it applies inverse ICP, un-scale, un-offset internally via `scale_mesh_`). So `nsm_recon_to_osim` only needs `convert_nsm_recon_to_OSIM_` (with underscore), not the full `convert_nsm_recon_to_OSIM`.

### Synthetic pipeline: latent → OSIM (what we're building)

`create_mesh(model, latent)` called **without** registration params returns mesh in **NSM canonical space**. To get to OSIM:

```python
convert_nsm_recon_to_OSIM(
    points=decoded_pts,                    # from create_mesh, in canonical bone space
    icp_transform=bone_linear_transform,   # 4x4 for this bone
    scale=1,
    center=[0, 0, 0],
    ref_mesh_orig_center=fem_ref_center    # always femur's mean_orig
)
```

This chains:
1. `undo_transform`: `pts @ inv(linear_transform).T` → femur-aligned space
2. `convert_nsm_recon_to_OSIM_`: `+fem_ref_center`, `/1000`, `@ OSIM_TO_NSM.T` → OSIM

The existing `convert_nsm_recon_to_OSIM` (without underscore) already implements this exact chain. No core conversion changes needed.

### The two conversion functions (clarification for documentation)

| Function | Input space | Output space | When to use |
|----------|-------------|--------------|-------------|
| `convert_nsm_recon_to_OSIM_` (underscore) | Femur-aligned space (mm, NSM-oriented) | OSIM (meters) | After `reconstruct_mesh` (ICP already undone internally) |
| `convert_nsm_recon_to_OSIM` (no underscore) | NSM canonical space | OSIM (meters) | After `create_mesh` (need to undo ICP via `undo_transform` first) |
| `convert_OSIM_to_nsm_` (underscore) | OSIM (meters) | Femur-aligned space (mm) | Going back to pre-OSIM space |
| `convert_OSIM_to_nsm` (no underscore) | OSIM (meters) | NSM canonical space | Going all the way back to canonical |

### Relative transforms (T_rel)

For analysis (regression on joint configuration) and reconstruction (synthetic joints):

```
T_rel_tib = T_fem @ inv(T_tib)    # canonical tibia → canonical femur
T_rel_pat = T_fem @ inv(T_pat)    # canonical patella → canonical femur
```

Recovering per-bone transforms from T_rel:
```
T_tib = inv(T_rel_tib) @ T_fem
T_pat = inv(T_rel_pat) @ T_fem
```

Decomposition into interpretable components (proven on 52 subjects in ACL project):
- Scale: `norm(T[:3, 0])` (uniform scaling embedded in 3x3)
- Rotation: `T[:3,:3] / scale` → proper rotation (det=+1)
- Translation: `T[:3, 3]` (in canonical femur units, convert to mm via `/ mean_fem_scale`)

Mean rotation: element-wise mean of rotation matrices + SVD projection to nearest proper rotation.

Deviations from mean:
- Rotation: `R_mean.T @ R_i` → Euler XYZ (degrees)
- Translation: `(t_i - t_mean) / mean_fem_scale` (mm)
- Scale: `s_i / s_mean` (ratio)

Recomposition (proven roundtrip):
```python
R_full = R_mean @ Rotation.from_euler("XYZ", angles_deg).as_matrix()
t_full = t_mean + dt_mm * mean_fem_scale
s_full = s_mean * scale_ratio
T[:3,:3] = s_full * R_full; T[:3,3] = t_full
```

### `fem_ref_center` — what it is and where it comes from

`fem_ref_center` = `mean_orig` from `ref_femur_alignment.json`. It is the mean position of the reference femur mesh in the NSM-oriented mm space before centering. Value: `[-1.22, -10.94, 8.20]`.

It is used for **ALL bones** (not just femur) because all bones are in the femur-aligned space (tibia/patella had `femur_transform` applied, not their own registration). Adding it back un-centers them all correctly.

It is loaded once at `comak_1_nsm_fitting.py:369` and passed everywhere.

### Reference implementation for T_rel

The ACL project has proven, tested code for T_rel computation and recomposition:
- `/dataNAS/people/aagatti/projects/pratham_ACL_wcb/scripts/Tibia_rotations/compute_relative_tibia_transforms.py`
- `/dataNAS/people/aagatti/projects/pratham_ACL_wcb/scripts/Tibia_rotations/recompose_tibia_transform.py`
- README with results from 52 subjects, coordinate conventions, and usage examples

Adapt this code into `nsosim/transforms.py`.

---

## Implementation Plan

### Phase A: Verify Transform Chain + Document Current State

**Before writing any new code**, verify the transform chain described above and document it in `nsosim/CLAUDE.md`.

**A1: Read and trace the code**
- Read `nsm_fitting.py` end-to-end: `align_bone_osim_fit_nsm`, `convert_nsm_recon_to_OSIM`, `convert_nsm_recon_to_OSIM_`, `undo_transform`, `apply_transform`, `nsm_recon_to_osim`, `interp_ref_to_subject_to_osim`
- Read `utils.py`: `fit_nsm`, `recon_mesh` — trace what `reconstruct_mesh` (NSM library) returns and what `scale_mesh_` does internally
- Read `NSM/sdf/reconstruct/main.py` and `NSM/sdf/mesh/main.py` (`create_mesh`, `scale_mesh_`) in the NSM library (GenerativeAnatomy) to confirm decoder output space and what registration params do
- Confirm: does `create_mesh` without params return mesh in [-1,1] canonical space? Does `create_mesh` WITH params (offset, scale, icp_transform) return mesh in the aligned input space?

**A2: Empirical verification**

All verification data is available — no GPU or simulation needed, just loading meshes and comparing.

**Reference data** (per bone, at `COMAK_SIMULATION_REQUIREMENTS/nsm_meshes/{bone}/`):
- `ref_{bone}_alignment.json` — reference transform (`transform_matrix`, `mean_orig`, `orig_scale`, `scale=1`, `center=[0,0,0]`)
- `latent_{bone}.npy` — reference latent vector
- `nsm_recon_ref_{bone}_nsm_space.vtk` — reference mesh in NSM canonical space (decoder output)
- `nsm_recon_ref_{bone}.vtk` — reference mesh in mm (aligned space, after `reconstruct_mesh` undid ICP)
- `nsm_recon_ref_{bone}_osim_space.vtk` — reference mesh in OSIM meters (the final output)
- `smith2019-R-{bone}-bone_processed.vtk` — the original reference bone mesh

`COMAK_SIMULATION_REQUIREMENTS` is at: `/dataNAS/people/aagatti/projects/comak_gait_simulation/COMAK_SIMULATION_REQUIREMENTS`

**Production subject data** (e.g., subject 9003316_RIGHT at `comak_gait_simulation_results/OARSI_menisci_pfp_v1/9003316_00m_RIGHT/geometries_nsm_similarity/{bone}/`):
- `{bone}_alignment.json` — subject's `linear_transform`, `scale=1`, `center=[0,0,0]`
- `{bone}_latent.npy` — subject's fitted latent vector
- `*_nsm_recon_mm.vtk` — mesh in mm (aligned space, output of `reconstruct_mesh`)
- `{bone}_nsm_recon_osim.stl` — mesh in OSIM meters (final output)
- `{bone}_cartilage_nsm_recon_osim.vtk` — cartilage in OSIM meters

Results base path: `/dataNAS/people/aagatti/projects/comak_gait_simulation_results`

**Verification steps:**

1. Load reference and subject alignment JSONs. Confirm `scale=1, center=[0,0,0]` for all.

2. **Test the aligned-space → OSIM path** (existing production path):
   Load `nsm_recon_ref_femur.vtk` (mm, aligned space) → apply `convert_nsm_recon_to_OSIM_(pts, fem_ref_center)` → compare against `nsm_recon_ref_femur_osim_space.vtk`. Should be identical.

3. **Test the canonical → OSIM path** (the synthetic path):
   Load `nsm_recon_ref_femur_nsm_space.vtk` (canonical) → apply `convert_nsm_recon_to_OSIM(pts, ref_transform_matrix, 1, [0,0,0], fem_ref_center)` → compare against `nsm_recon_ref_femur_osim_space.vtk`. Should be identical (same mesh, just different transform path).

4. **Test with a subject** (same canonical → OSIM path):
   Load subject 9003316's `femur_latent.npy` + `femur_alignment.json` (`linear_transform`). Decode latent with `create_mesh` (no params) → apply `convert_nsm_recon_to_OSIM(pts, linear_transform, 1, [0,0,0], fem_ref_center)` → compare against `femur_nsm_recon_osim.stl`. Expected: same surface geometry, possible marching cubes differences (different grid sampling in `create_mesh` vs the original `reconstruct_mesh` call).

5. **Repeat steps 3-4 for tibia and patella** to confirm the chain works for all bones (tibia is the most important — its transform encodes joint configuration).

**A3: Document in `nsosim/CLAUDE.md`**

Add a comprehensive "Coordinate Spaces & Transform Chain" section covering:
1. The 3 coordinate spaces (NSM canonical, femur-aligned, OSIM) with units and orientation
2. What each `linear_transform`/`transform_matrix` represents per bone
3. Why `fem_ref_center` is used for ALL bones (not just femur)
4. The function reference table: which function operates on which space, when to use each
5. The distinction between `convert_nsm_recon_to_OSIM_` (underscore) vs `convert_nsm_recon_to_OSIM` (no underscore)
6. How `reconstruct_mesh` returns meshes already in aligned space (ICP undone internally) vs `create_mesh` which returns canonical space

This documentation captures the **current state** of the library. It will be updated again at the end (Phase E) to include the new decode functions and T_rel concepts.

**If verification fails:** Stop. Debug the transform chain, correct this document, and re-verify before proceeding.

### Phase B: `nsosim/transforms.py` — New Module

Create a new module for similarity transform utilities. Adapt from the proven ACL project code.

**Functions:**

```python
def decompose_similarity(T: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Decompose 4x4 similarity transform into (scale, R, t).
    T[:3,:3] = scale * R where R is proper rotation (det=+1).
    """

def mean_rotation(rotations: np.ndarray) -> np.ndarray:
    """Mean rotation via element-wise average + SVD projection.
    Args: rotations (N, 3, 3). Returns: (3, 3) proper rotation.
    """

def compute_T_rel(T_fem: np.ndarray, T_other: np.ndarray) -> np.ndarray:
    """Compute relative transform: T_fem @ inv(T_other).
    Maps canonical other-bone → canonical femur space.
    """

def recover_bone_transform(T_rel: np.ndarray, T_fem: np.ndarray) -> np.ndarray:
    """Recover per-bone transform from relative: inv(T_rel) @ T_fem.
    Returns the bone's linear_transform (canonical bone → femur-aligned).
    """

def compute_transform_deviations(
    transforms: np.ndarray,     # (N, 4, 4) array of similarity transforms
    mean_fem_scale: float,       # for mm conversion
) -> dict:
    """Decompose transforms, compute mean, return per-subject deviations.
    Returns dict with: R_mean, t_mean, s_mean, euler_angles_deg (N,3),
    translations_mm (N,3), scale_ratios (N,).
    """

def deviations_to_transform(
    euler_angles_deg: np.ndarray,   # (3,) rotation deviation
    translation_mm: np.ndarray,     # (3,) translation deviation
    scale_ratio: float,             # scale deviation
    R_mean: np.ndarray,             # (3,3) mean rotation
    t_mean: np.ndarray,             # (3,) mean translation (canonical units)
    s_mean: float,                  # mean scale
    mean_fem_scale: float,          # for mm ↔ canonical conversion
) -> np.ndarray:
    """Recompose deviation values into a full 4x4 similarity transform.
    Pass [0,0,0], [0,0,0], 1.0 to get the mean transform.
    Proven roundtrip on 52 subjects (ACL project).
    """
```

**Tests (`tests/test_transforms.py`):**
- Roundtrip: `decompose_similarity(compose(s, R, t))` recovers (s, R, t)
- `mean_rotation` of N identical rotations returns that rotation
- `deviations_to_transform` with zero deviations returns mean transform
- `deviations_to_transform` → `decompose_similarity` → `compute_transform_deviations` roundtrip
- `compute_T_rel` + `recover_bone_transform` roundtrip: `recover(compute(T_fem, T_tib), T_fem) == T_tib`
- Known values from ACL project (52-subject summary statistics)

### Phase C: Decode Functions in `nsosim/decode.py` (NEW MODULE)

Decode functions go in a **new** `nsosim/decode.py` module rather than appending to `nsm_fitting.py`. Rationale: decoding (latent→mesh) is conceptually the inverse of fitting (mesh→latent). Keeping them separate avoids growing `nsm_fitting.py` (already 954 lines) and gives the new functionality a clean home. `decode.py` imports coordinate transform functions from `nsm_fitting.py`.

**C1: `decode_latent_to_osim()`** — single bone

```python
def decode_latent_to_osim(
    latent_vector: np.ndarray,
    model: torch.nn.Module,
    linear_transform: np.ndarray,   # 4x4: bone's linear_transform (or recovered from T_rel)
    fem_ref_center: np.ndarray,     # femur's mean_orig from ref alignment JSON
    n_pts_per_axis: int = 256,
    bone_clusters: int = 20_000,
    cart_clusters: int = None,
    men_clusters: int = None,
) -> dict:
    """Decode arbitrary latent vector to OSIM-space meshes.

    Steps:
    1. create_mesh(model, latent_tensor, objects=N) → canonical bone space
    2. convert_nsm_recon_to_OSIM(pts, linear_transform, 1, [0,0,0], fem_ref_center) → OSIM
    3. Optional resampling (resample_surface with clusters)

    Returns dict: {'bone': Mesh, 'cart': Mesh, optionally 'med_men': Mesh, 'lat_men': Mesh}
    """
```

**C2: `decode_joint_from_descriptors()`** — full joint

```python
def decode_joint_from_descriptors(
    femur_latent: np.ndarray,       # (1024,)
    tibia_latent: np.ndarray,       # (512,)
    patella_latent: np.ndarray,     # (512,)
    T_fem: np.ndarray,              # 4x4 femur linear_transform (required — caller chooses value: reference, population mean, or subject-specific)
    T_rel_tib: np.ndarray,          # 4x4 relative tibia transform
    T_rel_pat: np.ndarray,          # 4x4 relative patella transform
    models: dict,                   # {'femur': model, 'tibia': model, 'patella': model}
    fem_ref_center: np.ndarray,     # femur's mean_orig from ref alignment JSON
    bone_clusters: dict = None,     # e.g., {'femur': 20000, 'tibia': 20000, 'patella': 10000}
    cart_clusters: dict = None,
    men_clusters: dict = None,
) -> dict:
    """Decode full joint from latent vectors and pose transforms.

    This function does not need population statistics — it takes pre-computed
    transforms directly. If the caller wants to go from deviation values to
    T_rel, they call deviations_to_transform() from transforms.py first.

    T_fem is always required. The caller decides what value to provide
    (reference T_fem, population mean, or a specific subject's).

    1. Recover per-bone transforms: T_tib = recover_bone_transform(T_rel_tib, T_fem)
    2. decode_latent_to_osim() for each bone with its transform
    3. Returns {'femur': {bone, cart, med_men, lat_men}, 'tibia': {bone, cart}, 'patella': {bone, cart}}
    """
```

**Tests (`tests/test_decode.py`):**
- Decode reference femur latent with reference transform → compare against stored `nsm_recon_ref_femur_osim_space.vtk` (if exists, or generate and save as reference)
- Decode reference tibia latent with reference transform → compare against stored reference
- Smoke test: decode zero latent for each bone → mesh exists, has reasonable point count, is within expected spatial bounds

### Phase D: Validation (Verify New Decode Functions)

Verify the new decode functions produce correct results.

**D1: Decode known subject → compare against production**
1. Load a production subject's latent vectors and alignment JSONs (e.g., 9003316_RIGHT)
2. Compute T_fem, T_rel_tib, T_rel_pat
3. Call `decode_joint_from_descriptors()`
4. Compare each bone mesh against the production `*_nsm_recon_mm.vtk` files (after converting to same space)
5. **Expected:** Shapes match (same decoder + same latent). Positions match (same transforms). Only difference: marching cubes stochasticity (different grid sampling → slightly different triangulation, but same surface).

**D2: Verify `convert_nsm_recon_to_OSIM` equivalence**
1. Take a production subject's `recon_dict` (with ICP, scale, center)
2. Also take their `linear_transform` from alignment JSON
3. Decode their latent with `create_mesh` (no registration params) → canonical space
4. Convert via `convert_nsm_recon_to_OSIM(pts, linear_transform, 1, [0,0,0], fem_ref_center)`
5. Compare against production OSIM mesh
6. **Expected:** Close match (same surface, possible marching cubes differences)

**If validation fails:** The transform understanding documented above is wrong. Debug by tracing exact values through the chain, fix the documentation, and update this plan.

### Phase E: Final Documentation Update

**Update `nsosim/CLAUDE.md`** to add documentation for the new functionality:
1. The T_rel concept: what it is, how to compute/decompose/recompose it
2. The `transforms.py` module API and usage examples
3. The `decode_latent_to_osim()` and `decode_joint_from_descriptors()` usage examples
4. Update the "Complete Pipeline Workflow" section with a new "Synthetic Joint Decode" subsection

Phase A already documented the existing transform chain. This phase adds the new capabilities on top.

---

## File Changes Summary

| File | Change |
|------|--------|
| `nsosim/transforms.py` | **NEW** — similarity transform decomposition, T_rel, deviations, recomposition |
| `nsosim/decode.py` | **NEW** — `decode_latent_to_osim()` and `decode_joint_from_descriptors()` |
| `nsosim/CLAUDE.md` | Major update: coordinate spaces, transform chain documentation, module organization |
| `tests/fixtures/transforms/` | **NEW** — reference alignment JSONs + meshes at all 3 coordinate spaces |
| `tests/test_transforms.py` | **NEW** — roundtrip tests, known-value tests |
| `tests/test_decode.py` | **NEW** — decode reference latent, known-subject comparison |

## Test Fixtures

Fixture mesh files (~57MB uncompressed, 30MB tarball) are stored in GitHub Releases (`gattia/nsosim`, tag `test-fixtures-v1`), not in git. Alignment JSONs are kept in git (tiny). Tests auto-download meshes on first `pytest` run via `tests/fixtures/transforms/download_fixtures.sh`. Tests that need meshes use `@requires_mesh_fixtures` and skip gracefully if download fails.

**Two fixture sets:**

1. **Reference fixtures** (from `COMAK_SIMULATION_REQUIREMENTS/nsm_meshes/`):
   Each bone processed independently by `1_Fit_NSM_models_to_ref_surfaces`. OSIM meshes use each bone's own `mean_orig`. Per bone (femur, tibia, patella):
   - `ref_{bone}_alignment.json` — `transform_matrix`, `mean_orig`, `scale=1`, `center=[0,0,0]` (in git)
   - `latent_{bone}.npy` — reference latent vector
   - `nsm_recon_ref_{bone}_nsm_space.vtk` — NSM canonical space
   - `nsm_recon_ref_{bone}.vtk` — aligned mm space
   - `nsm_recon_ref_{bone}_osim_space.vtk` — OSIM meters

2. **Subject fixtures** (from `comak_gait_simulation_results/.../9003316_00m_RIGHT/`):
   Production subject where tibia/patella had femur_transform applied. All bones use `fem_ref_center`. Per bone:
   - `subject_9003316/{bone}_alignment.json` — `linear_transform`, `scale=1`, `center=[0,0,0]` (in git)
   - `subject_9003316/{bone}_nsm_recon_mm.vtk` — aligned mm space

Reference fixtures allow exact point-to-point comparison (same points, different transform paths). Subject fixtures test the production `fem_ref_center`-for-all-bones convention via roundtrip.

Decode function tests (Phase C/D) that require the NSM model + GPU are separate from these pure transform tests.

## Design Decisions

### Why `decode.py` instead of adding to `nsm_fitting.py`
`nsm_fitting.py` is 954 lines with two concerns: the fitting pipeline (mesh→latent, lines 28–437) and coordinate transforms (lines 607–954). Decoding (latent→mesh) is conceptually the inverse of fitting. Rather than growing an already-large file, decode functions live in `nsosim/decode.py` which imports the coordinate transform functions from `nsm_fitting.py`.

### Why `decode_joint_from_descriptors` takes T_rel + T_fem (not raw per-bone transforms)
- The caller (regression model / synthetic pipeline) naturally produces T_rel or deviation values
- The function doesn't need population statistics — if going from deviations → T_rel, the caller uses `deviations_to_transform()` from `transforms.py` beforehand
- Each function does one thing: `transforms.py` handles decomposition/recomposition, `decode.py` handles latent→mesh

### Why T_fem is always required (not defaulted)
T_fem is nearly constant across subjects (~0.0132 diagonal), but the caller always provides it explicitly. The choice of value (reference T_fem, population mean, subject-specific) is the caller's responsibility. The decode function doesn't assume or prefer any particular T_fem.

## Dependencies

| Dependency | Status | Notes |
|-----------|--------|-------|
| NSM library (`create_mesh`) | Available | Used by decode functions |
| `convert_nsm_recon_to_OSIM` (existing) | Available | No changes needed |
| `undo_transform` (existing) | Available | Called internally by `convert_nsm_recon_to_OSIM` |
| Test fixtures (meshes) | GitHub Releases tag `test-fixtures-v1` | Auto-downloaded by pytest, 30MB tarball |
| Production subject data | Available | Needed for validation (Phase D) |
| ACL project scripts | Available at `pratham_ACL_wcb/scripts/Tibia_rotations/` | Reference for Phase B |
| scipy | Already a dependency | For `Rotation.from_euler` / `.as_euler` |
