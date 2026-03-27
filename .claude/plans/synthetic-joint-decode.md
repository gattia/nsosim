# Plan: Synthetic Joint Decoding — Transform Utilities + Decode-from-Latent API

**Created:** 2026-03-25
**Status:** Phase B complete (not yet committed), Phase C next
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

## Understanding of the Transform Chain (VERIFIED)

**Verified 2026-03-26 in Phase A.** All paths confirmed with 54 tests in `tests/test_transform_chain.py`. See Phase A completion notes for key findings and corrections made during verification.

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

### Phase A: Verify Transform Chain + Document Current State — DONE

**Completed 2026-03-26.** Commit `5c10f25`. All deliverables:
- `tests/test_transform_chain.py` — 54 tests, all passing
- `CLAUDE.md` — full "Coordinate Systems & Units" section with test cross-references
- Test fixtures in GitHub Releases (tag `test-fixtures-v1`) with auto-download

**Key findings during Phase A:**

1. **Transform chain confirmed correct.** Both paths (production: mm→OSIM, synthetic: canonical→OSIM) verified with exact point-to-point comparison against reference meshes.

2. **Reference vs subject `mean_orig` distinction.** The reference OSIM meshes (from `1_Fit_NSM_models_to_ref_surfaces`) were generated using each bone's **own** `mean_orig`, because each bone was processed independently. In the production subject pipeline, all bones use `fem_ref_center` (the femur's `mean_orig`) because tibia/patella had `femur_transform` applied. Both are correct — the test suite covers both conventions.

3. **Alignment JSON key names differ.** Reference JSONs use `transform_matrix` and include `mean_orig`. Subject JSONs use `linear_transform` and do NOT include `mean_orig`. Verified in `TestAlignmentJsonKeyNames`.

4. **Scale factor values corrected.** The plan originally said "diagonal ≈ 0.0132" but the diagonal is not the scale — it only approximates scale when rotation is near-identity. The actual scale is the column norm of the 3x3 submatrix: femur ~0.013, tibia ~0.019, patella ~0.031 (was incorrectly described as "varies similarly to tibia").

5. **NSM library confirmed.** `create_mesh` without registration params returns mesh in NSM canonical space (~[-1,1]). `scale_mesh_` reverses centering/scaling/ICP in that order. `reconstruct_mesh` calls `scale_mesh_` internally so its output is in aligned input space.

**Test fixture architecture:**
- Alignment JSONs in git (tiny, tests always run)
- Mesh VTKs in GitHub Releases (`test-fixtures-v1`, 30MB tarball)
- Auto-downloaded on first `pytest` run via `download_fixtures.sh`
- `@requires_mesh_fixtures` decorator — mesh tests skip gracefully if download fails

### Phase B: `nsosim/transforms.py` — New Module — DONE

**Completed 2026-03-26.** Not yet committed (pending autoformat). All deliverables:
- `nsosim/transforms.py` — 6 functions, adapted from ACL project reference code
- `tests/test_transforms.py` — 16 tests, all passing
- `nsosim/__init__.py` — updated imports and `__all__`

**Functions implemented:**
- `decompose_similarity(T)` → `(scale, R, t)` — handles det < 0 (improper rotation) case
- `mean_rotation(rotations)` → `R_mean` — SVD projection with `diag([1, 1, d])` reflection correction
- `compute_T_rel(T_fem, T_other)` → `T_rel` — `T_fem @ inv(T_other)`
- `recover_bone_transform(T_rel, T_fem)` → `T_bone` — `inv(T_rel) @ T_fem`
- `compute_transform_deviations(transforms, mean_fem_scale)` → dict of means + per-subject deviations
- `deviations_to_transform(euler_deg, trans_mm, scale_ratio, R_mean, t_mean, s_mean, mean_fem_scale)` → 4x4

**Tests (16 total across 5 classes):**
- `TestDecomposeSimilarity` (5): identity, pure scale, 10 random roundtrips, proper rotation check, subject_9003316 fixture
- `TestMeanRotation` (4): identical rotations, identity, proper rotation output, symmetric ±θ pair
- `TestTRelRoundtrip` (3): synthetic roundtrip, subject_9003316 fixture (tibia + patella), identity-femur edge case
- `TestComputeTransformDeviations` (2): single transform → zero deviations, 50-sample center-at-zero property
- `TestDeviationsRoundtrip` (2): zero deviations = mean transform, full decompose→recompose roundtrip (20 transforms, atol=1e-10)

**Design decisions during implementation:**

1. **Dropped `TestACLKnownValues` class.** Originally planned to cross-validate against hardcoded numbers from the 52-subject ACL analysis. Decided against it: the roundtrip tests prove mathematical correctness, and hardcoded numbers from another project are brittle and opaque. Subject_9003316 fixtures (already in repo) provide sufficient "real data" coverage.

2. **Tightened tolerances on `test_deviations_center_at_zero`.** Initially had `atol=0.5` for all three checks. Translation mean and scale ratio mean are exact by construction (linear operations), so tightened to `1e-10`. Euler angle mean is not exactly zero due to Euler nonlinearity — measured ~0.08° worst case with 50 samples at ±5° std. Set `atol=0.15` with an explanatory comment. The real precision test is the roundtrip at `1e-10`.

3. **No type annotations.** Matches the style of the rest of the codebase (ACL reference code and existing nsosim modules don't use them).

**Adapted from:** `pratham_ACL_wcb/scripts/Tibia_rotations/compute_relative_tibia_transforms.py` (helpers: `decompose_similarity`, `mean_rotation`, deviation computation loop) and `recompose_tibia_transform.py` (`deviations_to_transform`). The script-level `main()` logic (loading files, printing summaries, saving JSON) was not carried over — only the pure math functions.

**What's needed before committing:**
1. Commit code changes
2. Run `make autoformat` and commit separately (per repo convention)

### Phase C: Decode Functions in `nsosim/decode.py` (NEW MODULE)

Decode functions go in a **new** `nsosim/decode.py` module rather than appending to `nsm_fitting.py`. Rationale: decoding (latent→mesh) is conceptually the inverse of fitting (mesh→latent). Keeping them separate avoids growing `nsm_fitting.py` (already 954 lines) and gives the new functionality a clean home. `decode.py` imports coordinate transform functions from `nsm_fitting.py`.

#### Pre-implementation notes (2026-03-26 review)

**Dependencies verified:**
- `NSM.mesh.create_mesh(decoder, latent_vector, objects=N, ...)` → returns `pymskt.mesh.Mesh` (single) or `list[Mesh]` (multiple). Called without registration params (`icp_transform`, `scale`, `offset`), it returns meshes in **NSM canonical space** (~[-1, 1]) with `scale_to_original_mesh=True` being a no-op (no original mesh to scale to).
- `convert_nsm_recon_to_OSIM(pts, icp_transform, scale, center, fem_ref_center)` — existing, works with `scale=1, center=[0,0,0]` and `icp_transform=linear_transform` for synthetic decode path. Already verified in Phase A tests.
- `pymskt.mesh.Mesh.resample_surface(subdivisions=1, clusters=N)` — method on the mesh object, used in `_nsm_recon_to_osim_single_surface`.

**Mesh name mapping:** `get_mesh_names(model_config)` from `utils.py` is now available (commit `29a32c2`). It reads `mesh_names` from model config if present, falls back to `(bone, objects_per_decoder)` lookup. All 7 production model configs have been updated with explicit `mesh_names`. Use `get_mesh_names()` in `decode_latent_to_osim()` to map decoder outputs to named meshes — same pattern as the refactored `recon_mesh()`.

**GPU requirement:** `create_mesh` runs the decoder on CUDA. The function should document this. Tests that call `create_mesh` need `@pytest.mark.skipif(not torch.cuda.is_available(), ...)` or similar.

**C1: `decode_latent_to_osim()`** — single bone

```python
def decode_latent_to_osim(
    latent_vector,           # np.ndarray
    model,                   # torch.nn.Module (on GPU)
    linear_transform,        # np.ndarray, 4x4: bone's linear_transform
    fem_ref_center,          # np.ndarray, femur's mean_orig
    model_config,            # dict — needs objects_per_decoder (and mesh_names if available)
    n_pts_per_axis=256,
    clusters=None,           # dict e.g. {'bone': 20000, 'cart': None} or None for no resampling
):
    """Decode arbitrary latent vector to OSIM-space meshes.

    Steps:
    1. create_mesh(model, latent_tensor, objects=N) → list of Mesh in canonical space
    2. For each mesh: convert_nsm_recon_to_OSIM(pts, linear_transform, 1, [0,0,0], fem_ref_center)
    3. Optional resampling per mesh type

    Returns dict: {'bone': Mesh, 'cart': Mesh, ...} — keys from mesh_names
    """
```

**C2: `decode_joint_from_descriptors()`** — full joint

```python
def decode_joint_from_descriptors(
    femur_latent,            # np.ndarray (1024,)
    tibia_latent,            # np.ndarray (512,)
    patella_latent,          # np.ndarray (512,)
    T_fem,                   # np.ndarray, 4x4 femur linear_transform
    T_rel_tib,               # np.ndarray, 4x4 relative tibia transform
    T_rel_pat,               # np.ndarray, 4x4 relative patella transform
    models,                  # dict {'femur': model, 'tibia': model, 'patella': model}
    model_configs,           # dict {'femur': config, 'tibia': config, 'patella': config}
    fem_ref_center,          # np.ndarray, femur's mean_orig
    n_pts_per_axis=256,
    clusters=None,           # dict of dicts, e.g. {'femur': {'bone': 20000}, ...} or None
):
    """Decode full joint from latent vectors and pose transforms.

    T_fem is always required. The caller decides what value to provide
    (reference T_fem, population mean, or a specific subject's).

    1. Recover per-bone transforms: T_tib = recover_bone_transform(T_rel_tib, T_fem)
    2. decode_latent_to_osim() for each bone with its transform
    3. Returns {'femur': {'bone': Mesh, 'cart': Mesh, ...}, 'tibia': {...}, 'patella': {...}}
    """
```

**Tests (`tests/test_decode.py`):**
- Decode reference femur latent with reference transform → compare against stored `nsm_recon_ref_femur_osim_space.vtk` (if exists, or generate and save as reference)
- Decode reference tibia latent with reference transform → compare against stored reference
- Smoke test: decode zero latent for each bone → mesh exists, has reasonable point count, is within expected spatial bounds
- All decode tests require GPU — mark with appropriate skip decorator

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
