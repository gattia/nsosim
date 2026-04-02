# Plan: Synthetic Joint Decoding — Transform Utilities + Decode-from-Latent API

**Created:** 2026-03-25
**Status:** Complete (2026-04-02)
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

**Completed 2026-03-26.** Commit `516eceb` (code), `e86ae68` (autoformat). All deliverables:
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

### Interstitial: Mesh Name Mapping Refactor — DONE

**Completed 2026-03-27.** Commit `29a32c2`. Before starting Phase C, we identified that both `recon_mesh()` and the planned `decode_latent_to_osim()` would need to map positional decoder outputs to named meshes (bone, cart, med_men, etc.). The existing code used a fragile count-based heuristic duplicated wherever meshes were decoded.

**What was done:**
- Added `get_mesh_names(model_config)` to `nsosim/utils.py` — reads `mesh_names` from config if present, falls back to `(bone, objects_per_decoder)` lookup with warning
- Refactored `recon_mesh()` to use `get_mesh_names()` instead of count-based branching
- Added `mesh_names` as an optional config parameter in the NSM library's training code (commit `709b818` in NSM repo, `adaptive_marching_cubes` branch)
- Updated all 7 production model config JSONs with explicit `mesh_names`
- 14 tests in `tests/test_mesh_names.py`, full suite (272 tests) passes

**Full details:** `.claude/plans/mesh-name-mapping.md`

### Phase C: Decode Functions in `nsosim/decode.py` (NEW MODULE) — DONE

**Completed 2026-04-01.** Commit `bd8bc41` (code), `531261f` (autoformat). All deliverables:
- `nsosim/decode.py` — 2 functions: `decode_latent_to_osim()`, `decode_joint_from_descriptors()`
- `tests/test_decode.py` — 27 tests across 3 classes, all passing on GPU
- `tests/fixtures/models/{femur,tibia,patella}/` — model configs in git, `.pth` weights gitignored
- `nsosim/__init__.py` — added `decode` to imports and `__all__`

**Functions implemented:**
- `decode_latent_to_osim(latent_vector, model, linear_transform, fem_ref_center, model_config, ...)` — decodes a single bone's latent to OSIM-space meshes. Handles numpy→torch conversion, normalizes `create_mesh` return to list, uses `get_mesh_names()` for output keys, optional per-mesh resampling.
- `decode_joint_from_descriptors(femur_latent, tibia_latent, patella_latent, T_fem, T_rel_tib, T_rel_pat, models, model_configs, fem_ref_center, ...)` — recovers per-bone transforms via `recover_bone_transform()`, then calls `decode_latent_to_osim()` for each bone.

**Tests (27 total across 3 classes):**
- `TestDecodeLatentToOsim` (12): Per-bone decode for femur/tibia/patella — correct keys, point counts, OSIM spatial range, and reference comparison with point-to-surface ASSD <0.05mm, centroid <0.1mm, extent rtol <0.5%
- `TestDecodeJointFromDescriptors` (6): Joint decode — all bones present, correct mesh keys per bone, all meshes in OSIM spatial range
- `TestDecodeZeroLatent` (9): Smoke tests — zero latent (mean shape) for each bone, verify mesh exists with plausible points

**Key findings during Phase C:**

1. **Wrong models initially copied.** The reference fixtures (latents + OSIM meshes) were generated by `1_Fit_NSM_models_to_ref_surfaces` using models 568/femur, 650/tibia, 648/patella. Initially copied 75/tibia and 77/patella instead (same latent dim and objects_per_decoder, but different trained weights). Caused ~28% bounding box extent mismatch. Fixed by copying the correct models.

2. **Point-to-surface vs point-to-point ASSD.** Initial ASSD calculation used KDTree nearest-neighbor (point-to-point), which gave 0.074–0.228mm. Switched to `pymskt.Mesh.get_assd_mesh()` which uses point-to-surface SDF via `pcu_sdf`, giving 0.001–0.005mm — consistent with same model+latent decoded independently.

3. **Reference transforms don't encode joint configuration.** Reference bones were each processed independently with their own `mean_orig`. T_rel computed from reference transforms doesn't represent a real joint pose, so spatial relationship tests (tibia distal to femur, patella anterior) are invalid for reference data. These tests would need subject data where all bones share `fem_ref_center`. Replaced with structural tests (correct keys, OSIM range).

4. **Canonical→OSIM uses no-underscore conversion.** `decode_latent_to_osim()` uses `convert_nsm_recon_to_OSIM` (no underscore) which chains `undo_transform` + `convert_nsm_recon_to_OSIM_`. Cannot reuse `_nsm_recon_to_osim_single_surface()` (which expects femur-aligned input, not canonical).

**Model fixture architecture:**
- `tests/fixtures/models/{bone}/model_params_config.json` — in git (trimmed: `list_mesh_paths` and `val_paths` stripped, ~240 lines each vs original ~26k–38k)
- `tests/fixtures/models/{bone}/model.pth` — gitignored (263–299 MB each)
- `@requires_gpu` and `@requires_nsm_models` decorators — tests skip gracefully without GPU or weights

Decode functions go in a **new** `nsosim/decode.py` module rather than appending to `nsm_fitting.py`. Rationale: decoding (latent→mesh) is conceptually the inverse of fitting (mesh→latent). Keeping them separate avoids growing `nsm_fitting.py` (already 954 lines) and gives the new functionality a clean home. `decode.py` imports coordinate transform functions from `nsm_fitting.py`.

#### Pre-implementation notes (2026-03-26 review, updated 2026-03-27)

**Dependencies verified:**
- `NSM.mesh.create_mesh(decoder, latent_vector, n_pts_per_axis=256, ..., objects=1, ...)` — full signature confirmed. Returns `pymskt.mesh.Mesh` (single) or `list[Mesh]` (multiple). Called without registration params (`icp_transform`, `scale`, `offset`), it returns meshes in **NSM canonical space** (~[-1, 1]).
- `latent_vector` must be a **`torch.float` tensor on CUDA** — the function should handle numpy→torch conversion internally: `torch.tensor(latent_vector, dtype=torch.float).cuda()`.
- `objects` comes from `model_config['objects_per_decoder']`.
- `convert_nsm_recon_to_OSIM(pts, icp_transform, scale, center, fem_ref_center)` — existing, works with `scale=1, center=[0,0,0]` and `icp_transform=linear_transform` for synthetic decode path. Already verified in Phase A tests. Does NOT mutate the input array (undo_transform returns new array; underscore version mutates that intermediate).
- `pymskt.mesh.Mesh.resample_surface(subdivisions=1, clusters=N)` — method on the mesh object, used in `_nsm_recon_to_osim_single_surface`.

**Mesh name mapping:** `get_mesh_names(model_config)` from `utils.py` is now available (commit `29a32c2`). It reads `mesh_names` from model config if present, falls back to `(bone, objects_per_decoder)` lookup. All 7 production model configs have been updated with explicit `mesh_names`. Use `get_mesh_names()` in `decode_latent_to_osim()` to map decoder outputs to named meshes — same pattern as the refactored `recon_mesh()`.

**Canonical→OSIM conversion:** Cannot reuse `_nsm_recon_to_osim_single_surface()` (uses underscore version for femur-aligned input). For synthetic path, use `convert_nsm_recon_to_OSIM` (no underscore) which chains `undo_transform` (canonical→femur-aligned) then `convert_nsm_recon_to_OSIM_` (femur-aligned→OSIM). Pattern per mesh:
```python
mesh_osim = mesh.copy()
mesh_osim.point_coords = convert_nsm_recon_to_OSIM(
    mesh.point_coords.copy(), linear_transform, 1, [0, 0, 0], fem_ref_center
)
```

**Return key convention:** Use bare names (`'bone'`, `'cart'`, etc.) — consistent with `nsm_recon_to_osim()` (line 917). Note: `recon_mesh()` uses `'{name}_mesh'` suffix, but that's a different function with different semantics.

**GPU requirement:** `create_mesh` runs the decoder on CUDA. The function should document this. Tests require GPU + model weights on disk.

**Model callers responsibility:** Callers load models via `load_model()` from `utils.py` before calling decode functions. The decode functions take already-loaded models, not paths.

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

**Fixture strategy:** Model weights (`.pth`, 263–299 MB each) are copied to `tests/fixtures/models/{femur,tibia,patella}/` with `model.pth` + `model_params_config.json` per bone. The `.pth` files are gitignored; config JSONs are tracked. Skip decorator:
```python
MODELS_DIR = Path(__file__).parent / "fixtures" / "models"
requires_nsm_models = pytest.mark.skipif(
    not (MODELS_DIR / "femur" / "model.pth").exists(),
    reason="NSM model weights not available in tests/fixtures/models/"
)
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)
```

**Model source:** Copied from `COMAK_SIMULATION_REQUIREMENTS/nsm_models/`:
| Bone | Source model | `objects_per_decoder` | `.pth` size |
|------|-------------|----------------------|-------------|
| Femur | `568_nsm_femur_bone_cart_men_v0.0.1` | 4 (bone, cart, med_men, lat_men) | 299 MB |
| Tibia | `75_nsm_tibia_cartilage_v0.0.1` | 2 (bone, cart) | 263 MB |
| Patella | `77_nsm_patella_cartilage_v0.0.1` | 2 (bone, cart) | 263 MB |

**Test structure:** Use `scope="module"` fixtures for expensive decode calls. Multiple test methods assert on the pre-computed result.

**Tests:**
- Decode reference femur latent with reference transform → compare against stored `nsm_recon_ref_femur_osim_space.vtk` (same surface, allow marching cubes stochasticity)
- Decode reference tibia latent with reference transform → compare against stored reference
- Smoke test: decode zero latent for each bone → mesh exists, has reasonable point count, is within expected spatial bounds
- `decode_joint_from_descriptors()`: decode reference joint using all 3 models + reference transforms → verify all bones present and spatially plausible (tibia distal to femur, patella anterior)
- All decode tests require GPU + model weights — mark with `@requires_gpu` and `@requires_nsm_models`

### Phase D: Validation (Verify New Decode Functions)

Verify the decode functions produce correct results for a real production subject.

**Scope reduction (2026-04-02 review):** The original D2 (verify `convert_nsm_recon_to_OSIM` equivalence by decoding to canonical space and converting) is already covered by Phase A tests: `TestRefCanonicalToOsim` and `TestSubjectCanonicalToOsimRoundtrip` verify this exact path with point-to-point comparison for both reference and subject data. The original D1's reference-latent decode comparison is already covered by Phase C tests: `test_{bone}_bone_close_to_reference_osim` decodes reference latents through `decode_latent_to_osim()` and compares against stored OSIM meshes (ASSD <0.05mm). What remains untested is the **subject** path through `decode_joint_from_descriptors()` — i.e., using T_rel to recover per-bone transforms and comparing against production output.

**D1: Decode known subject via `decode_joint_from_descriptors()` → compare against production**
1. Load subject 9003316_RIGHT's latent vectors and alignment JSONs (already in `tests/fixtures/transforms/subject_9003316/`)
2. Compute T_fem, T_rel_tib, T_rel_pat from the alignment JSONs
3. Call `decode_joint_from_descriptors()`
4. Compare each bone mesh against the production `*_nsm_recon_mm.vtk` files (after converting to OSIM space) using ASSD
5. **Expected:** Shapes match (same decoder + same latent). Positions match (same transforms). Only difference: marching cubes stochasticity (different grid sampling → slightly different triangulation, but same surface).

**If validation fails:** The T_rel recovery path or the `decode_joint_from_descriptors` orchestration is wrong. Debug by comparing per-bone transforms against the raw alignment JSON values, then trace through the decode chain.

**Completed 2026-04-02.** All 11 tests pass. Added `TestSubjectDecodeVsProduction` to `tests/test_decode.py`:

**Tests (11 total: 9 parametrized across 3 bones + 2 spatial relationship):**
- `test_centroid_close[femur/tibia/patella]` (3): decoded vs production centroids within 0.1mm
- `test_extent_close[femur/tibia/patella]` (3): bounding box extents within 0.5%
- `test_assd_below_threshold[femur/tibia/patella]` (3): point-to-surface ASSD <0.05mm
- `test_tibia_distal_to_femur` (1): tibia centroid Y < femur centroid Y in OSIM coords
- `test_patella_anterior_to_femur` (1): patella centroid X > femur centroid X in OSIM coords

**What this validates:** The full T_rel recovery path — subject alignment JSONs → `compute_T_rel()` → `decode_joint_from_descriptors()` (which calls `recover_bone_transform()` internally) → OSIM meshes — produces the same surfaces as the production pipeline (which went through `reconstruct_mesh()` → `nsm_recon_to_osim()`). The spatial relationship tests (tibia distal, patella anterior) confirm the production convention (`fem_ref_center` for all bones) preserves correct joint anatomy, which was not testable with reference data (Phase C finding #3).

**No new fixtures needed.** All subject data was already in `tests/fixtures/transforms/subject_9003316/` from Phase A. Production mm-space meshes are converted to OSIM on the fly via `convert_nsm_recon_to_OSIM_()`.

### Phase E: Final Documentation Update — DONE

**Completed 2026-04-02.** Added "Relative Transforms (T_rel)" subsection to `CLAUDE.md` under "Coordinate Systems & Units".

Items 2–4 were already completed during earlier phases:
- Item 2 (`transforms.py` API): Added during Phase B, visible at line ~358 in CLAUDE.md
- Item 3 (decode function examples): Added during Phase C, visible at line ~322
- Item 4 (Synthetic Joint Decode subsection): Added during Phase C, visible at line ~322

Item 1 (T_rel concept) was the remaining gap. Added a new subsection covering:
- What T_rel represents (joint configuration independent of femur alignment)
- Computing T_rel from per-bone transforms
- Recovering per-bone transforms from T_rel (used by `decode_joint_from_descriptors`)
- Decomposition into interpretable components (scale, rotation, translation)
- Population statistics via `compute_transform_deviations()`
- Recomposition via `deviations_to_transform()` with code examples
- Cross-references to test verification

> **Future consideration (2026-04-02):** When writing `comak_1_synthetic.py` (parent plan Phase C), the post-decode orchestration (articular surfaces → wrap surfaces → ligaments → meniscus → fat pad → model assembly) duplicates `comak_1_nsm_fitting.py` Stages 2–5. Consider extracting the shared orchestration into an nsosim function (e.g., `build_osim_model_from_meshes()`) so both pipelines share the same code. See note in `SYNTHETIC_JOINT_SIMULATION.md` Phase C.

---

## File Changes Summary

| File | Change |
|------|--------|
| `nsosim/transforms.py` | **NEW** — similarity transform decomposition, T_rel, deviations, recomposition |
| `nsosim/decode.py` | **NEW** — `decode_latent_to_osim()` and `decode_joint_from_descriptors()` |
| `nsosim/CLAUDE.md` | Major update: coordinate spaces, transform chain documentation, module organization |
| `tests/fixtures/transforms/` | **NEW** — reference alignment JSONs + meshes at all 3 coordinate spaces |
| `tests/test_transforms.py` | **NEW** — roundtrip tests, known-value tests |
| `tests/test_decode.py` | **NEW** — decode reference latent, known-subject comparison, subject-vs-production validation |

## Test Fixtures

Fixture mesh files (~57MB uncompressed, 30MB tarball) are stored in GitHub Releases (`gattia/nsosim`, tag `test-fixtures-v1`), not in git. Alignment JSONs are kept in git (tiny). Tests auto-download meshes on first `pytest` run via `tests/fixtures/transforms/download_fixtures.sh`. Tests that need meshes use `@requires_mesh_fixtures` and skip gracefully if download fails.

**NSM model weights** (`tests/fixtures/models/{femur,tibia,patella}/`): `.pth` files (263–299 MB each) are gitignored and must be copied manually for now. Config JSONs are tracked in git. Tests that need models use `@requires_nsm_models` and skip gracefully.

**Future consolidation (TODO):** All large test fixtures (mesh VTKs from GitHub Releases, model `.pth` files, latent `.npy` files) should be consolidated into a single downloadable location (Google Drive, HuggingFace dataset, or similar) with a unified download script. Currently split across GitHub Releases (meshes) and manual copy (models). See `.claude/plans/` for a future plan on this.

**Three fixture sets:**

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

3. **Model fixtures** (copied from `COMAK_SIMULATION_REQUIREMENTS/nsm_models/`):
   Per bone (femur, tibia, patella):
   - `models/{bone}/model.pth` — decoder weights (gitignored, 263–299 MB)
   - `models/{bone}/model_params_config.json` — model config with `mesh_names` (in git)

Reference fixtures allow exact point-to-point comparison (same points, different transform paths). Subject fixtures test the production `fem_ref_center`-for-all-bones convention via roundtrip. Model fixtures enable decode integration tests (require GPU).

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

---

## Completion Notes

**Date completed:** 2026-04-02

**Summary:** Added two new modules to nsosim — `transforms.py` (similarity transform decomposition, relative transforms, deviation analysis/recomposition) and `decode.py` (decode arbitrary latent vectors to OSIM-space meshes). Together these enable synthetic joint generation from latent vectors and joint pose descriptors, which was the missing capability needed by `comak_gait_simulation` for Paper 1 Analysis #8.

**Changes made:**

| File | Change | Commit |
|------|--------|--------|
| `nsosim/transforms.py` | NEW — 6 functions for similarity transform math | `516eceb` |
| `nsosim/decode.py` | NEW — `decode_latent_to_osim()`, `decode_joint_from_descriptors()` | `bd8bc41` |
| `nsosim/utils.py` | Added `get_mesh_names()` for decoder output → mesh name mapping | `29a32c2` |
| `nsosim/__init__.py` | Updated imports and `__all__` for new modules | `516eceb`, `bd8bc41` |
| `nsosim/CLAUDE.md` | Coordinate Systems docs, Synthetic Joint Decode section, T_rel section | `5c10f25`, `bd8bc41`, current |
| `tests/test_transform_chain.py` | NEW — 54 tests verifying both transform pipelines | `5c10f25` |
| `tests/test_transforms.py` | NEW — 16 tests for transforms.py | `516eceb` |
| `tests/test_mesh_names.py` | NEW — 14 tests for mesh name mapping | `29a32c2` |
| `tests/test_decode.py` | NEW — 38 tests (27 Phase C + 11 Phase D) for decode functions | `bd8bc41`, `bc1f979` |
| `tests/fixtures/transforms/` | Reference + subject alignment JSONs, mesh VTKs (in GitHub Releases) | `5c10f25` |
| `tests/fixtures/models/` | Model configs in git, `.pth` weights gitignored | `bd8bc41` |

**Tests:** 122 new tests total (54 + 16 + 14 + 27 + 11), all passing. No regressions to existing test suite.

**Additional issues resolved:**
- Mesh name mapping refactor (`get_mesh_names()`) — eliminated fragile count-based heuristic in `recon_mesh()` and prevented duplication in new decode code
- Updated all 7 production model config JSONs with explicit `mesh_names` field
- Added `mesh_names` config parameter upstream in NSM library (commit `709b818` in NSM repo)

**Challenges / Design decisions:**
- Reference vs subject transforms use different conventions (`mean_orig` per-bone vs shared `fem_ref_center`) — tests cover both paths explicitly
- Point-to-surface ASSD (via `pcu_sdf`) needed instead of point-to-point KDTree for meaningful decode comparison (0.001–0.005mm vs 0.074–0.228mm)
- Reference transforms don't encode real joint configuration, so spatial relationship tests (tibia distal, patella anterior) only work with subject data
- `decode.py` placed in new module rather than appending to `nsm_fitting.py` (already 954 lines) — decode is conceptually the inverse of fitting

**Things to note for future work:**
- Post-decode orchestration (articular surfaces → wrap surfaces → ligaments → meniscus → fat pad → model assembly) is duplicated between production and synthetic pipelines. Consider extracting into a shared `build_osim_model_from_meshes()` function when writing `comak_1_synthetic.py`
- NSM model weights (`tests/fixtures/models/*/model.pth`, 263–299 MB each) must be copied manually. Future: consolidate all large fixtures into a single downloadable location
- Parent plan: `comak_gait_simulation/.claude/plans/SYNTHETIC_JOINT_SIMULATION.md` — this plan covers nsosim-side work only; the parent plan tracks the full end-to-end synthetic pipeline
