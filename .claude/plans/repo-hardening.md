# Repository Hardening Plan

Goal: Reduce error potential during fast AI-assisted development by improving test coverage, validation, documentation accuracy, and code hygiene.

---

## Execution Instructions

This plan is designed to be handed to an AI agent phase-by-phase. Follow these rules:

1. **Execute one phase at a time.** Complete a phase, verify it works (`pytest`, `make lint`), then move on. Do not combine multiple phases in one pass.
2. **Phase 1 is pure additive** — new test files only, zero risk to existing code. Start here.
3. **Phases 3 and 4 modify existing source code.** Work on a branch so diffs can be reviewed before merge. Especially Phase 3.1 (new `schemas.py` module that other code will import).
4. **Phase 6 (CI) depends on repo infrastructure.** Before executing, confirm: private vs public repo, target Python version, whether self-hosted runners are available. Ask the user if this context is missing.
5. **After each phase**, run `pytest` and `make lint` to confirm nothing is broken before proceeding.

---

## Phase 1: Test Infrastructure & Core Math Tests (Critical)

### 1.1 Set up test infrastructure
- Add `tests/__init__.py` and `tests/wrapping/__init__.py`
- Create `tests/conftest.py` with shared fixtures (synthetic meshes, known rotation matrices, sample `dict_bones` structures)
- Uncomment `make test` target in Makefile (currently lines 36-38)
- Fix existing `tests/wrapping/test_sign_convention.py` to use proper pytest assertions instead of `print()` statements

### 1.2 Rotation & coordinate transform tests (`tests/test_rotation_utils.py`)
- Round-trip: quaternion -> rotation matrix -> quaternion = identity
- Round-trip: rotation matrix -> Euler angles -> rotation matrix = identity
- `enforce_sign_convention()` idempotency (applying twice gives same result)
- Known-value tests against scipy.spatial.transform.Rotation for validation

### 1.3 SDF tests (`tests/test_wrap_signed_distances.py`)
- `sd_ellipsoid_improved()`: point at center is negative, point on surface ~0, point outside is positive
- `sd_cylinder()`: same inside/surface/outside checks
- Symmetry: SDF should be symmetric for symmetric shapes
- Known analytic values for simple cases (unit sphere, axis-aligned cylinder)

### 1.4 Coordinate system transform tests (`tests/test_coordinate_transforms.py`)
- Round-trip: `convert_nsm_recon_to_OSIM_()` then `convert_OSIM_to_nsm_()` recovers original points
- `OSIM_TO_NSM_TRANSFORM` is orthogonal (R^T R = I) and has determinant +1 or -1
- Unit conversion: mm <-> m consistency checks

### 1.5 `wrap_surface` data class tests (`tests/test_wrap_surface.py`)
- Construction with all field types
- `to_dict()` output contains all expected keys
- Round-trip: construct -> `to_dict()` -> reconstruct preserves values

---

## Phase 2: Synthetic Mesh Tests (High)

### 2.1 Fitter tests (`tests/test_fitting.py`)
- `CylinderFitter`: fit to a known pyvista cylinder point cloud, verify recovered center/axis/radius within tolerance
- `EllipsoidFitter`: fit to a known pyvista ellipsoid point cloud, verify recovered center/radii within tolerance
- `PatellaFitter`: basic smoke test with labeled synthetic mesh

### 2.2 Articular surface tests (`tests/test_articular_surfaces.py`)
- `add_polar_coordinates_about_center()`: verify r/theta values for known point positions
- `create_articular_surfaces()`: smoke test with synthetic sphere mesh pair (bone + cartilage)
- `create_meniscus_articulating_surface()`: verify upper/lower split on a synthetic torus
- Radial envelope functions: `build_min_radial_envelope()`, `trim_mesh_by_radial_envelope()` with known inputs

### 2.3 Mesh labeling tests (`tests/test_mesh_labeling.py`)
- `label_mesh_vertices_for_wrap_surfaces()`: verify binary labels match expected inside/outside for a mesh with a known cylinder through it
- `classify_points()` and `classify_near_surface()`: threshold behavior at boundary

---

## Phase 3: Validation & Safety Guards (High)

### 3.1 Define data structure schemas
- Create `nsosim/schemas.py` with `TypedDict` or `dataclass` definitions for:
  - `DictBones` (with per-bone variants: femur has `med_men_filename`/`lat_men_filename`, others don't)
  - `FittedWrapParameters` (bone -> body -> type -> name -> wrap_surface)
  - `AlignmentTransform` (keys: `transform_matrix`, `mean_orig`, `orig_scale`)
- Add a `validate_dict_bones(dict_bones)` function that checks required keys/subkeys exist before pipeline entry
- Add a `validate_fitted_wrap_parameters(params)` function

### 3.2 Add validation calls at pipeline entry points
- `align_knee_osim_fit_nsm()`: validate `dict_bones` on entry
- `update_osim_model()`: validate `dict_wrap_objects` and `dict_lig_mus_attach`
- `nsm_recon_to_osim()`: validate `surface_idx` is in valid range

### 3.3 Make `recon_mesh()` internal index mapping explicit
- **Context:** `recon_mesh()` (utils.py:323-338) already returns a named dict with keys
  `bone_mesh`, `cart_mesh`, `med_men_mesh`, `lat_men_mesh`, `fibula_mesh`. All downstream code
  (`nsm_recon_to_osim()`, pipeline scripts) already uses named keys — no external breaking change.
- The internal mapping (lines 325-335) uses `mesh_result['mesh'][0..3]` and infers identity by count
  (3 meshes = tibia+fibula, 4 meshes = femur+menisci). This is fragile.
- **Action:** Replace count-based heuristic with an explicit mapping driven by `model_config['bone']`.
  Use the bone type to determine which indices map to which surface names. Add an assertion that
  the mesh count matches the expected count for that bone type, with a clear error message if it doesn't.
- **Impact on pipeline scripts:** None. The returned dict keys are unchanged. This is an internal-only
  improvement to how those keys get assigned.

---

## Phase 4: Code Hygiene (Medium)

### 4.1 Remove deprecated `wraps.py`
- Remove `from . import wraps` from `nsosim/__init__.py` (line 5)
- Remove `'wraps'` from `__all__` in `__init__.py` (line 14)
- Delete `nsosim/wraps.py` or add `raise DeprecationWarning` on import

### 4.2 Fix print() -> logging
- `nsm_fitting.py`: `check_icp_transform()` uses `print()` — switch to `logger.warning()`/`logger.info()`
- `utils.py`: bone clipping output uses `print()` — switch to `logger.info()`

### 4.3 Promote magic numbers to named constants
- In `articular_surfaces.py`, create a constants section at module top:
  ```python
  # Prefemoral fat pad defaults
  DEFAULT_FATPAD_BASE_MM = 0.8
  DEFAULT_FATPAD_TOP_MM = 5.5
  DEFAULT_MAX_DISTANCE_TO_PATELLA_MM = 30.0
  # Meniscus defaults
  DEFAULT_RADIAL_PERCENTILE = 95.0
  DEFAULT_MENISCUS_SDF_THRESHOLD = 0.1
  ```
- In `utils.py`: name the `0.95` bone clipping factor
- Reference these constants in function signatures as defaults

### 4.4 Resolve or expand TODOs
- `nsm_fitting.py:110` — `# TODO: get rid of this extra alignment transform later?` — either resolve or expand with context: is this safe to remove? Under what conditions?
- `nsm_fitting.py:170` — `# TODO: Update NSM recon function to` — complete the sentence
- `comak_osim_update.py:112` — `# TODO: update dictionaries to include parent/child info` — expand with what the target structure would look like

---

## Phase 5: Documentation Accuracy (Medium)

### 5.1 Update `readme.md`
- Remove/update reference to `wraps.py` as active module — note it's deprecated
- Add `wrap_surface_fitting/` submodule to the architecture section
- Update "Step 6: Definition of wrapping surfaces" to point to new approach
- Fix case: `pyproject.toml` says `readme = "README.md"` but file is `readme.md`

### 5.2 Update `CLAUDE.md`
- Document the `LOC_SDF_CACHE` environment variable hack in `nsm_fitting.py:10`
- Note the `print()` vs `logging` inconsistency (until Phase 4.2 fixes it)
- Add note about meniscus instability issue and link to `MENISCUS_ARTICULAR_SURFACE_INSTABILITY.md`
- Document which functions have type annotations and which don't (so AI agents know where to be extra careful)

### 5.3 Add module docstrings to files missing them
- `nsm_fitting.py`, `utils.py`, `osim_utils.py`, `articular_surfaces.py` (core modules)
- `wrap_surface_fitting/fitting.py`, `wrap_surface_fitting/utils.py`, `wrap_surface_fitting/rotation_utils.py`, `wrap_surface_fitting/patella.py`, `wrap_surface_fitting/surface_param_estimation.py`, `wrap_surface_fitting/wrap_signed_distances.py` (submodule)
- Keep docstrings to 1-3 lines: what the module does, not a full API listing

---

## Phase 6: CI/CD (Medium)

**Prerequisites:** Before executing this phase, confirm with the user: private vs public repo, target Python version, whether self-hosted runners are available. Do not guess.

### 6.1 Add GitHub Actions workflow
- `.github/workflows/ci.yml`
- Trigger on push and PR to `main`
- Steps: install deps, `make lint`, `pytest`
- Use a lightweight runner (no GPU needed for Tier 1-2 tests)

### 6.2 Add pre-commit hooks (optional)
- `.pre-commit-config.yaml` with `black`, `isort`, basic file checks
- Keeps formatting consistent without manual `make autoformat`

---

## Phase 7: Type Annotations on Public API (Low)

### 7.1 Add type annotations to pipeline entry points
- `align_knee_osim_fit_nsm()`, `nsm_recon_to_osim()` in `nsm_fitting.py`
- `create_articular_surfaces()`, `create_meniscus_articulating_surface()`, `create_prefemoral_fatpad_noboolean()` in `articular_surfaces.py`
- `update_osim_model()` in `comak_osim_update.py`
- Focus on function signatures only — don't annotate internals

---

## Execution Order

1. **Phase 1** (test infra + math tests) — highest value, lowest risk, pure additive
2. **Phase 3.1-3.2** (validation schemas) — prevents most common AI-agent errors, do on a branch
3. **Phase 2** (synthetic mesh tests) — catches regression in core algorithms
4. **Phase 4** (code hygiene) — small cleanup items, do on a branch
5. **Phase 5** (docs accuracy) — keeps AI agents from being misled
6. **Phase 3.3** (recon_mesh explicit mapping) — internal-only change, do after tests exist to verify no regression
7. **Phase 6** (CI) — automates everything above, requires repo infrastructure context
8. **Phase 7** (type annotations) — nice-to-have, lowest priority
