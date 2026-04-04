# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`nsosim` is a library for creating personalized biomechanical knee models by fitting Neural Shape Models (NSM) to MRI segmentations and integrating them into OpenSim/COMAK simulations. It bridges raw imaging data → NSM-derived geometries → subject-specific OpenSim models.

## Development Commands

```bash
# Install
pip install -e .              # Editable install
make install-dev              # Same as above

# Dependencies
pip install -r requirements.txt       # Production deps
pip install -r requirements-dev.txt   # Dev deps (pytest, black, isort)

# Linting and formatting
make lint          # Check formatting (isort + black)
make autoformat    # Auto-format code

# Tests
pytest             # ~314 tests covering math, SDF, fitters, schemas, articular surfaces, transforms, decode

# Test fixtures (large mesh files, not in git — auto-downloaded on first pytest run)
tests/fixtures/transforms/download_fixtures.sh   # Manual download if needed

# Build
make build         # Build wheel to wheelhouse/
make clean         # Remove build artifacts
```

**Code style:** Black with 100 char line length, isort with black profile.

### Commit Discipline for Formatting

**Always commit code changes BEFORE running `make autoformat`.** Autoformat reformats the entire repo, not just your changes. If you mix code changes with repo-wide reformatting in one commit, the diff becomes unreadable and hard to review. The workflow is:

1. Make your code changes
2. `git add` and `git commit` the code changes
3. Run `make autoformat`
4. `git add` and `git commit` the formatting changes separately (e.g., "Apply autoformat")

This keeps the meaningful diff in one commit and the mechanical formatting in another.

## Testing Guidelines

When writing or modifying tests for this repo, follow these rules strictly.

### Core Principle: Tests Must Be Able to Fail

A test that cannot fail when the code is wrong is worse than no test — it gives false confidence. Every test you write should have a clear scenario where incorrect code would cause it to fail.

### Anti-Patterns (DO NOT DO THESE)

**Never wrap assertions in try/except/pytest.skip.** This silently hides real failures:
```python
# BAD — this test can never fail
try:
    result = create_articular_surfaces(bone, cart, ...)
    assert result.n_points > 0
except Exception as e:
    pytest.skip(f"failed: {e}")  # silently passes even if code is broken

# GOOD — failures are reported
result = create_articular_surfaces(bone, cart, ...)
assert result.n_points > 0
```

**Never loosen tolerances just to make a test pass.** If a test fails, fix the code or understand why the tolerance needs to be what it is. Document the reasoning if a loose tolerance is genuinely required:
```python
# BAD — atol=0.5 on a function that interpolates 3 exact values
np.testing.assert_allclose(r_min, [5.0, 4.0, 5.0], atol=0.5)

# GOOD — tolerance matches actual function precision
np.testing.assert_allclose(r_min, [5.0, 4.0, 5.0], atol=0.01)

# GOOD — loose tolerance with documented reason
# SDF gradient approximation underestimates distance far from surface (see docstring)
assert abs(sdf.item() - expected) < 0.3, f"At d={d}, SDF={sdf.item():.4f}"
```

**Never over-test trivial code.** A one-line function (e.g., `np.convolve(y, kernel, 'same')`) does not need 10+ tests. Two or three tests covering the key behaviors are sufficient. Spend test-writing effort on functions with complex logic, edge cases, or numerical sensitivity.

**Never create fixtures "just in case."** Every fixture in conftest.py should be used by at least one test. Remove unused fixtures promptly.

### What to Test

**Prioritize by risk, not by simplicity.** Functions that are easy to test but never break are low value. Focus on:
1. **Numerical edge cases**: gimbal lock, degenerate inputs, near-zero denominators
2. **Non-trivial configurations**: rotated/translated/scaled inputs, not just axis-aligned defaults
3. **Roundtrip consistency**: encode → decode should recover the original (e.g., rotation matrix → Euler → rotation matrix)
4. **Cross-validation against reference implementations**: compare against SciPy, numpy, or analytical solutions where possible

For this repo specifically, wrap surface fitters should always be tested with non-axis-aligned geometry. Real OpenSim wrap surfaces are never perfectly aligned with coordinate axes. An axis-aligned-only test suite will miss orientation bugs.

### Practical Patterns for This Codebase

**Fitter tests**: Use `scope="class"` fixtures for expensive fitting operations so the optimizer runs once and multiple test methods check different properties of the result:
```python
class TestCylinderFitterRotated:
    @pytest.fixture(scope="class")
    def fitted(self):
        # Expensive: runs optimizer once
        fitter = CylinderFitter(...)
        fitter.fit(points=pts, labels=labels, ...)
        return fitter

    def test_recovers_center(self, fitted):  # cheap: just checks a property
        ...
    def test_recovers_radius(self, fitted):  # cheap: just checks a property
        ...
```

**Rotation/transform tests**: Always validate against `scipy.spatial.transform.Rotation` as the reference. Test both the standard case AND gimbal lock (Y = ±90°).

**Smoke tests**: Mark expensive integration tests with `@pytest.mark.slow`. These should verify the function runs without error and produces plausible output shape/type — but they must NOT swallow exceptions.

**SDF tests**: Surface points should have SDF ≈ 0 with tight tolerance (1e-4). Interior points negative, exterior positive. Test with translated and rotated geometry, not just origin-centered axis-aligned shapes.

**Large test fixtures**: Mesh files (VTK) are stored in GitHub Releases (`gattia/nsosim`, tag `test-fixtures-v1`), not in git. Tests auto-download them on first run via `tests/fixtures/transforms/download_fixtures.sh`. Tests that need mesh fixtures use `@requires_mesh_fixtures` and skip gracefully if download fails. Alignment JSONs are kept in git (tiny) so JSON-only tests always run.

**NSM model fixtures** (`tests/fixtures/models/{femur,tibia,patella}/`): Model configs (`model_params_config.json`, trimmed) are in git. Model weights (`model.pth`, 263–299 MB) are gitignored and must be present locally for decode tests. Models: 568/femur, 650/tibia, 648/patella — must match those used to generate reference fixture latents. Tests use `@requires_gpu` and `@requires_nsm_models` decorators.

## Architecture

### Core Modules

| Module | Purpose |
|--------|---------|
| `nsm_fitting.py` | NSM fitting pipeline: align meshes → fit NSM → convert to OpenSim coords. Also contains coordinate transform functions (`convert_nsm_recon_to_OSIM`, `undo_transform`, etc.) |
| `transforms.py` | Similarity transform utilities: decomposition, relative transforms (T_rel), mean rotation, deviation analysis/recomposition |
| `decode.py` | Decode arbitrary latent vectors to OSIM-space meshes (synthetic joints, shape mode visualization) |
| `model_building.py` | Model-building orchestration: OSIM-space meshes → subject-specific OpenSim model. Step functions (ligament interpolation, wrap fitting, patella centering) + `build_joint_model()` orchestrator. Used by both fitting and synthetic pipelines. |
| `articular_surfaces.py` | Extract/refine cartilage contact surfaces, meniscus processing, prefemoral fatpad, patella optimization |
| `comak_osim_update.py` | Update OpenSim XML with subject-specific meshes, attachments, wrap surfaces |
| `meniscal_ligaments.py` | Post-interpolation projection of meniscal ligament tibia attachments onto tibia surface |
| `osim_utils.py` | Low-level OpenSim XML manipulation via Python API |
| `utils.py` | NSM model loading, mesh I/O, anatomical coordinate system (ACS) alignment utilities |

### Module Organization Principles

`nsm_fitting.py` contains both the fitting pipeline (mesh→latent) and the coordinate transform functions. These could be separate modules, but the transforms are tightly coupled to the fitting workflow and splitting now would break imports across downstream projects. New functionality should go in new modules rather than growing `nsm_fitting.py` further:

- **Fitting (mesh→latent):** `nsm_fitting.py` — existing, don't append to it
- **Decoding (latent→mesh):** `decode.py` — conceptually the inverse of fitting, imports transform functions from `nsm_fitting.py`
- **Model building (meshes→OpenSim):** `model_building.py` — orchestrates articular surface extraction, ligament interpolation, wrap surface fitting, meniscus surfaces, fat pad, and OpenSim model assembly. Extracted from `comak_1_nsm_fitting.py` so both fitting and synthetic pipelines share the same code.
- **Transform analysis:** `transforms.py` — pure math (similarity decomposition, T_rel, deviations), no NSM/model dependencies
- **Coordinate conversions** (`convert_nsm_recon_to_OSIM`, `undo_transform`, etc.): stay in `nsm_fitting.py` for now — a future refactor could extract these into a `coordinates.py` module


### Wrap Surface Fitting Submodule (`wrap_surface_fitting/`)

PyTorch-based optimization to adapt OpenSim wrap surfaces to new bone geometries using signed distance functions (SDF).

| File | Purpose |
|------|---------|
| `fitting.py` | `CylinderFitter`, `EllipsoidFitter` - PyTorch optimizers for shape fitting |
| `patella.py` | `PatellaFitter` - specialized ellipsoid fitting for patella wrap surfaces |
| `config.py` | `DEFAULT_SMITH2019_BONES`, `DEFAULT_FITTING_CONFIG` - default configurations |
| `main.py` | `wrap_surface` data class - container for fitted wrap surface parameters |
| `parameter_extraction.py` | `extract_wrap_parameters_from_osim()`, `create_meshes_from_wrap_parameters()` - extract/visualize reference wrap surfaces |
| `mesh_labeling.py` | `label_mesh_vertices_for_wrap_surfaces()`, `label_multiple_meshes()` - label bone meshes with SDF values |
| `utils.py` | Data loading, SDF computation, point classification, `prepare_multi_bone_fitting_data()` |
| `surface_param_estimation.py` | Geometric initialization for fitters (`fit_cylinder_geometric()`, `fit_ellipsoid_algebraic()`) - internal use |
| `rotation_utils.py` | Quaternion/Euler angle conversions, `RotationUtils` class |
| `wrap_signed_distances.py` | SDF functions for cylinders and ellipsoids |

## Coordinate Systems & Units

### Three Coordinate Spaces

1. **NSM canonical space** (~[-1, 1] unit cube): The decoder's output space. Each bone has its own canonical space (femur canonical ≠ tibia canonical). This is what `create_mesh(model, latent)` returns when called without registration params.

2. **Femur-aligned space** (mm, NSM-oriented): All bones share this space after the femur registration is applied. Orientation is rotated from OSIM by `OSIM_TO_NSM_TRANSFORM`. Centered around the reference femur's centroid (subtracted during preprocessing). In the production pipeline, tibia/patella have `femur_transform` applied instead of their own registration, placing them in this shared space.

3. **OSIM space** (meters, OpenSim orientation): The final output space for OpenSim models. Obtained from femur-aligned space by: `+fem_ref_center`, `/1000`, `@ OSIM_TO_NSM_TRANSFORM.T`.

### Transform Chain

**Production pipeline** (subject mesh → OSIM):
```
Subject mesh (mm, arbitrary orientation)
  → femur registration (or femur_transform for tibia/patella) → femur-aligned mm
  → reconstruct_mesh() → internally: center, scale, ICP, optimize, decode, scale_mesh_()
  → returns mesh in femur-aligned mm (ICP already undone internally)
  → nsm_recon_to_osim() → convert_nsm_recon_to_OSIM_(pts, fem_ref_center) → OSIM
```

**Synthetic decode pipeline** (arbitrary latent → OSIM):
```
create_mesh(model, latent) → NSM canonical space
  → convert_nsm_recon_to_OSIM(pts, linear_transform, 1, [0,0,0], fem_ref_center)
    chains: undo_transform() → femur-aligned mm → convert_nsm_recon_to_OSIM_() → OSIM
```

Both pipelines are verified in `tests/test_transform_chain.py` using reference meshes at all 3 coordinate spaces (exact point-to-point comparison) and subject data (roundtrip through canonical space matches direct conversion).

### Conversion Function Reference

| Function | Input space | Output space | When to use | Verified by |
|----------|-------------|--------------|-------------|-------------|
| `convert_nsm_recon_to_OSIM_` (underscore) | Femur-aligned mm | OSIM meters | After `reconstruct_mesh` (ICP already undone) | `TestRefAlignedMmToOsim` |
| `convert_nsm_recon_to_OSIM` (no underscore) | NSM canonical | OSIM meters | After `create_mesh` (undoes ICP via `undo_transform` first) | `TestRefCanonicalToOsim` |
| `convert_OSIM_to_nsm_` (underscore) | OSIM meters | Femur-aligned mm | Inverse of underscore version | `TestRefRoundtrip` |
| `convert_OSIM_to_nsm` (no underscore) | OSIM meters | NSM canonical | Full inverse (OSIM → canonical) | `TestRefRoundtrip` |

All functions tested in `tests/test_transform_chain.py` with both reference and subject data.

### Per-bone `linear_transform` (alignment JSONs)

Each bone's alignment JSON stores a 4x4 similarity `linear_transform` that maps femur-aligned space → that bone's NSM canonical space. Both `scale` and `center` fields are always `1` and `[0,0,0]` — the similarity ICP embeds centering + scaling + rotation into the 4x4 matrix. The 3x3 submatrix encodes `scale * R` where `scale = norm(T[:3, 0])` (uniform across columns) and `R` is a proper rotation.

Typical scale factors (column norms of the 3x3 submatrix — maps mm to [-1,1] canonical range):
- Femur: ~0.013 (nearly constant across subjects — femur is the registration anchor)
- Tibia: ~0.019 (varies across subjects — encodes joint configuration relative to femur)
- Patella: ~0.031 (varies — patella is smaller so needs larger scaling to fill [-1,1])

Scale=1, center=[0,0,0], 4x4 structure, and uniform scaling verified in `TestAlignmentJsonStructure` in `tests/test_transform_chain.py`.

**Note:** Subject alignment JSONs use `linear_transform` as the key. Reference alignment JSONs use `transform_matrix` and also include `mean_orig`. Verified in `TestAlignmentJsonKeyNames`.

### `fem_ref_center`

`fem_ref_center` = `mean_orig` from `ref_femur_alignment.json`. It is the centroid of the reference femur mesh in NSM-oriented mm space before centering. Value: `[-1.22, -10.94, 8.20]`. Verified in `TestFemRefCenter`.

In the **production subject pipeline**, it is used for ALL bones because all bones are in the femur-aligned space (tibia/patella had `femur_transform` applied). Adding it back un-centers them correctly, preserving their spatial relationships (e.g., tibia ~50mm distal to femur condyles). Verified in `TestSubjectFemRefCenterConvention`.

**Reference reconstructions** (from `1_Fit_NSM_models_to_ref_surfaces`) are different: each bone was processed independently using its own `mean_orig`. The reference alignment JSONs store per-bone `mean_orig` values.

### Relative Transforms (T_rel)

Relative transforms capture how tibia/patella sit relative to the femur in canonical space — encoding joint configuration (flexion, varus/valgus, relative bone size) independently of the femur's own alignment.

**Computing T_rel** from per-bone alignment transforms:
```
T_rel_tib = T_fem @ inv(T_tib)    # canonical tibia → canonical femur
T_rel_pat = T_fem @ inv(T_pat)    # canonical patella → canonical femur
```

**Recovering per-bone transforms** from T_rel (for synthetic decode):
```
T_tib = inv(T_rel_tib) @ T_fem
T_pat = inv(T_rel_pat) @ T_fem
```

This is the path used by `decode_joint_from_descriptors()`: given a femur transform and relative transforms, it recovers per-bone transforms then decodes each bone independently. Verified in `TestSubjectDecodeVsProduction` (Phase D).

**Decomposition into interpretable components** (for regression / analysis):
- Scale: `norm(T_rel[:3, 0])` — uniform scaling embedded in 3x3 submatrix
- Rotation: `T_rel[:3, :3] / scale` → proper rotation matrix (det = +1)
- Translation: `T_rel[:3, 3]` — in canonical femur units; convert to mm via `/ mean_fem_scale`

**Population statistics** (for synthetic generation):
```python
deviations = compute_transform_deviations(list_of_T_rel, mean_fem_scale)
# Returns: R_mean, t_mean, s_mean, and per-subject deviations as:
#   euler_deg (XYZ Euler angles relative to R_mean)
#   trans_mm (translation offset from t_mean, in mm)
#   scale_ratio (s_i / s_mean)
```

Mean rotation uses element-wise mean of rotation matrices + SVD projection to the nearest proper rotation (`mean_rotation()`).

**Recomposition** (from deviations back to a full 4x4 transform):
```python
T_rel = deviations_to_transform(
    euler_deg, trans_mm, scale_ratio,
    R_mean, t_mean, s_mean, mean_fem_scale
)
# Internally: R = R_mean @ Rotation.from_euler("XYZ", euler_deg).as_matrix()
#             t = t_mean + trans_mm * mean_fem_scale
#             s = s_mean * scale_ratio
#             T[:3,:3] = s * R; T[:3,3] = t
```

Decompose→recompose roundtrip verified to `atol=1e-10` in `TestDeviationsRoundtrip`. All functions in `nsosim/transforms.py`, tested in `tests/test_transforms.py` (16 tests).

## Complete Pipeline Workflow

The typical processing pipeline follows this sequence:

### Stage 1: NSM Model Fitting
```python
dict_bones = align_knee_osim_fit_nsm(
    dict_bones=dict_bones,
    folder_save_bones=folder_save_bones,
    n_samples_latent_recon=20_000,
    convergence_patience=10,
    rigid_reg_type='similarity',  # or 'rigid'
    acs_align=False
)
```

### Stage 2: Per-Bone Processing (Femur → Tibia → Patella)
For each bone:
1. **NSM Reconstruction**: `nsm_recon_to_osim()` - convert to OpenSim coords
2. **Articular Surface Extraction**: `create_articular_surfaces()` - extract cartilage contact surfaces
3. **Ligament Attachment Interpolation**: `interp_ref_to_subject_to_osim()` - map reference points to subject
4. **Wrap Surface Fitting**: `CylinderFitter` / `EllipsoidFitter` - fit wrap surfaces to labeled mesh

### Stage 3: Meniscus Processing
```python
med_men_upper, med_men_lower = create_meniscus_articulating_surface(
    meniscus_mesh=fem_med_men_mesh_osim,
    upper_articulating_bone_mesh=fem_mesh_osim,
    lower_articulating_bone_mesh=tib_mesh_osim,
    meniscus_center=med_meniscus_center,  # from tibia labeled mesh
    theta_offset=np.pi,  # CRITICAL: rotate polar coords to avoid discontinuity
    ray_length=15.0,
    n_largest=1,
)
```
**Note**: Menisci use the femur NSM model with `surface_idx` parameter (0=bone, 1=cart, 2=med_men, 3=lat_men).

### Stage 3b: Meniscus Surface Refinement (Optional)
The radial envelope functions can refine meniscus articulating surfaces:
```python
from nsosim.articular_surfaces import refine_meniscus_articular_surfaces

refined_upper, refined_lower = refine_meniscus_articular_surfaces(
    meniscus_mesh=meniscus_mesh,
    upper_surface=upper_art_surf,
    lower_surface=lower_art_surf,
    meniscus_center=meniscus_center,
    theta_offset=np.pi,  # same as original extraction
    radial_percentile=95.0,
)
```

### Stage 4: Prefemoral Fat Pad
```python
prefemoral_fat_pad = create_prefemoral_fatpad_noboolean(
    femur_bone_mesh=fem_mesh_osim,
    femur_cart_mesh=fem_cart_mesh_osim,
    patella_bone_mesh=pat_mesh_osim,
    patella_cart_mesh=pat_cart_mesh_osim,
    base_mm=1.0,
    top_mm=6,
    max_distance_to_patella_mm=25,
    units='m',
)
```

### Stage 4b: Meniscal Ligament Tibia Attachment Projection (Optional)
An alternative to using NSM-interpolated tibia attachments for meniscal ligaments. Instead of relying on `interp_ref_to_subject_to_osim()` for the tibia-side points, this projects from the meniscus attachment straight down onto the tibia surface via ray-casting, producing near-vertical ligaments.
```python
from nsosim.meniscal_ligaments import project_meniscal_attachments_to_tibia

projection_results = project_meniscal_attachments_to_tibia(
    dict_lig_mus_attach=dict_lig_musc_attach_params,
    tibia_mesh=tib_mesh_osim,
    # ray_direction defaults to [0, -1, 0] (-Y = distal)
    # max_ray_length defaults to 0.015 (15mm)
)
```

### Stage 5: OpenSim Model Update
```python
update_osim_model(
    model=osim_model,
    dict_wrap_objects=fitted_wrap_parameters,
    dict_lig_mus_attach=dict_lig_musc_attach_params,
    tibia_mesh_osim=tib_mesh_osim,
    mean_patella=mean_patella,
    lig_musc_xyz_key='xyz_mesh_updated',
)
```

### Synthetic Joint Decode (Latent → OSIM)

For generating meshes from arbitrary latent vectors (synthetic joints, shape mode visualization, latent interpolation) without fitting to a target mesh:

```python
from nsosim.decode import decode_latent_to_osim, decode_joint_from_descriptors
from nsosim.transforms import compute_T_rel, recover_bone_transform
from nsosim.utils import load_model

# Single bone decode
result = decode_latent_to_osim(
    latent_vector=latent,          # np.ndarray, e.g. (1024,) for femur
    model=model,                   # loaded via load_model(), on GPU
    linear_transform=T_bone,       # 4x4 from alignment JSON
    fem_ref_center=fem_ref_center, # from ref_femur_alignment.json['mean_orig']
    model_config=config,           # must have objects_per_decoder and mesh_names
    n_pts_per_axis=256,
    clusters={'bone': 20000},      # optional resampling
)
# result = {'bone': Mesh, 'cart': Mesh, ...}

# Full joint decode from relative transforms
joint = decode_joint_from_descriptors(
    femur_latent=fem_latent,
    tibia_latent=tib_latent,
    patella_latent=pat_latent,
    T_fem=T_fem,                   # femur's linear_transform
    T_rel_tib=T_rel_tib,           # from compute_T_rel(T_fem, T_tib)
    T_rel_pat=T_rel_pat,           # from compute_T_rel(T_fem, T_pat)
    models={'femur': fem_model, 'tibia': tib_model, 'patella': pat_model},
    model_configs={'femur': fem_config, 'tibia': tib_config, 'patella': pat_config},
    fem_ref_center=fem_ref_center,
)
# joint = {'femur': {'bone': Mesh, ...}, 'tibia': {...}, 'patella': {...}}
```

**Transform utilities** (`nsosim.transforms`):
```python
from nsosim.transforms import (
    decompose_similarity,          # T → (scale, R, t)
    mean_rotation,                 # list of R matrices → mean R (SVD-projected)
    compute_T_rel,                 # T_fem, T_other → T_rel = T_fem @ inv(T_other)
    recover_bone_transform,        # T_rel, T_fem → T_bone = inv(T_rel) @ T_fem
    compute_transform_deviations,  # list of T_rel → means + per-subject deviations
    deviations_to_transform,       # euler_deg, trans_mm, scale_ratio + means → T
)
```

**Key difference from production pipeline:** The production pipeline uses `reconstruct_mesh()` which fits a latent to a target mesh and undoes ICP internally — its output is in femur-aligned mm space, converted via `convert_nsm_recon_to_OSIM_` (underscore). The synthetic decode path uses `create_mesh()` which returns NSM canonical space, converted via `convert_nsm_recon_to_OSIM` (no underscore, chains `undo_transform` + underscore version).

## Reference Surface Preparation (One-Time Setup)

Before fitting subjects, reference labeled meshes must be created. This is done once per reference model:

### Extract and Label Reference Wrap Surfaces
```python
from nsosim.wrap_surface_fitting.parameter_extraction import (
    extract_wrap_parameters_from_osim,
    create_meshes_from_wrap_parameters
)
from nsosim.wrap_surface_fitting.mesh_labeling import label_multiple_meshes
from nsosim.wrap_surface_fitting.utils import prepare_multi_bone_fitting_data

# Step 1: Extract original wrap parameters from OpenSim model
original_params = extract_wrap_parameters_from_osim(osim_path)

# Step 2: Create mesh visualizations of wrap surfaces (optional, for verification)
original_surfaces = create_meshes_from_wrap_parameters(original_params)

# Step 3: Label bone meshes with SDF values for each wrap surface
labeled_bones = prepare_multi_bone_fitting_data(
    geometry_folder=dict_bone_folders,  # {bone_name: folder_path}
    xml_path=osim_path,
    bone_dict=DEFAULT_SMITH2019_BONES,
    near_surface_threshold=DEFAULT_SMITH2019_THRESHOLDS
)

# Step 4: Save labeled bones for use in subject fitting
for bone_name, mesh in labeled_bones.items():
    mesh.save(f'{bone_name}_labeled.vtk')
```

### Patella Wrap Surface Labeling
Patella uses a specialized labeling function:
```python
from nsosim.wrap_surface_fitting.patella import label_patella_within_wrap_extents

labeled_patella = label_patella_within_wrap_extents(
    patella_mesh=patella_mesh,
    wrap_surface_mesh=PatTen_r_mesh  # from create_meshes_from_wrap_parameters
)
```

## Critical Data Structures

### dict_bones Structure
```python
dict_bones = {
    'femur': {
        'ref': {
            'folder': '/path/to/ref/meshes',
            'bone_filename': 'smith2019-R-femur-bone_processed.vtk'
        },
        'subject': {
            'folder': '/path/to/subject/meshes',
            'bone_filename': 'SUBJ_RIGHT_femur.vtk',
            'cart_filename': 'SUBJ_RIGHT_femur_cart.vtk',
            'med_men_filename': 'SUBJ_RIGHT_med_men.vtk',  # femur only
            'lat_men_filename': 'SUBJ_RIGHT_lat_men.vtk',  # femur only
        },
        'model': {
            'path_model_state': '/path/to/nsm/model.pth',
            'path_model_config': '/path/to/model_params_config.json'
        },
        'wrap': {
            'path_labeled_bone': '/path/to/femur_labeled.vtk'
        }
    },
    'tibia': { ... },
    'patella': { ... },
    'meniscus': {  # reference menisci only
        'ref': {
            'folder': '/path/to/ref/menisci',
            'med_men_filename': 'med_meniscus_processed.vtk',
            'lat_men_filename': 'lat_meniscus_processed.vtk'
        }
    }
}
```

### Labeled Mesh Data Arrays
Labeled bone meshes store wrap surface classifications as point data:
- `{wrap_name}_binary` - inside (1) / outside (0) classification
- `{wrap_name}_sdf` - signed distance values (positive = outside)
- `{wrap_name}_near_surface` - boolean for points near wrap surface (for cylinders)
- `med_meniscus_center_binary` / `lat_meniscus_center_binary` - meniscus center regions (tibia only)

### Wrap Surface Fitting Pattern
```python
# Get labels from labeled mesh
labels = labeled_mesh[f'{wrap_name}_binary'].copy()
sdf = labeled_mesh[f'{wrap_name}_sdf'].copy()

# For cylinders, also get near-surface mask
near_surface = labeled_mesh[f'{wrap_name}_near_surface'].copy()

# Fit and extract parameters
fitter = CylinderFitter(**CYLINDER_CONSTRUCTOR_CONFIG)
fitter.fit(points=..., labels=..., sdf=..., **CYLINDER_FIT_CONFIG)
wrap_params = fitter.wrap_params
wrap_params.name = wrap_name
wrap_params.body = body_name
```

### wrap_surface Data Class
The `wrap_surface` class (from `wrap_surface_fitting/main.py`) stores fitted wrap parameters:
```python
wrap_params = fitter.wrap_params  # Returns wrap_surface instance
wrap_params.name = 'KnExt_at_fem_r'
wrap_params.body = 'femur_r'
wrap_params.type_        # 'WrapCylinder' or 'WrapEllipsoid'
wrap_params.translation  # np.ndarray, center position (meters)
wrap_params.xyz_body_rotation  # np.ndarray, Euler angles (radians)
wrap_params.radius       # float, cylinder radius (meters)
wrap_params.length       # float, cylinder length (meters)
wrap_params.dimensions   # np.ndarray, ellipsoid radii [x, y, z] (meters)
```

### fitted_wrap_parameters Structure
```python
fitted_wrap_parameters = {
    'femur': {
        'femur_r': {
            'ellipsoid': {'Gastroc_at_Condyles_r': wrap_params},
            'cylinder': {'KnExt_at_fem_r': wrap_params, ...}
        },
        'femur_distal_r': {
            'cylinder': {'Capsule_r': wrap_params}
        }
    },
    'tibia': {...},
    'patella': {...}
}
```

## Key Design Decisions

### `LOC_SDF_CACHE` Environment Variable
`nsm_fitting.py:13` sets `os.environ["LOC_SDF_CACHE"] = ""` at import time. This is a workaround — the upstream NSM library expects this env var to exist but nsosim doesn't use it. Without this line, importing from `NSM.mesh.interpolate` would fail.

### Patella Centering
The patella is centered by subtracting its mean position before saving to OpenSim. The offset is saved as `patella_offset.json` and used to update the default patella joint coordinates:
```python
mean_patella = np.mean(pat_mesh_osim.point_coords, axis=0)
pat_mesh_centered.point_coords -= mean_patella
```

**`_original_position` file convention:** The production pipeline saves patella meshes in two forms:
- `patella_nsm_recon_osim.stl` — **centered** (mean subtracted), used by OpenSim
- `patella_nsm_recon_osim_original_position.vtk` — **non-centered**, in OSIM space as output by `nsm_recon_to_osim()`

"Original position" means "OSIM space before centering", NOT the MRI/subject space. The same applies to `patella_cartilage_nsm_recon_osim_original_position.vtk` and `patella_articular_surface_osim_original_position.vtk`. When re-running model building from saved meshes (e.g., `build_joint_model`), use the `_original_position` VTKs as input — the orchestrator does its own centering. Femur and tibia do not have this distinction because they are not centered.

### Articular Surface Functions
Key functions in `articular_surfaces.py`:
- `create_articular_surfaces()` - Extract bone/cartilage contact surfaces
- `create_meniscus_articulating_surface()` - Extract upper/lower meniscus surfaces
- `create_prefemoral_fatpad_noboolean()` - Create prefemoral fat pad contact surface
- `optimize_patella_position()` - Optimize patella position relative to femur (optional)
- `refine_meniscus_articular_surfaces()` - Refine meniscus surfaces using radial envelope

**Radial envelope functions** (for meniscus refinement):
- `build_min_radial_envelope()` - Build minimum radial envelope from polar coordinates
- `trim_mesh_by_radial_envelope()` - Trim mesh points outside radial envelope
- `mask_points_by_radial_envelope()` - Mask points based on radial envelope

### Meniscus theta_offset Parameter
Meniscus surfaces are extracted using polar coordinates. The `theta_offset` parameter rotates the coordinate system to prevent the polar discontinuity (at θ=0/2π) from cutting through tissue:
- **Medial meniscus**: use `theta_offset=np.pi`
- **Lateral meniscus**: use `theta_offset=0.0`

### Cylinder Quadrant Orientation (commit 629e02d)
Cylinders in OpenSim have a "quadrant" parameter controlling which side is active for muscle wrapping:
- `construct_cylinder_basis()` uses a `reference_x_axis` (defaults to global +X) for consistent orientation
- **Cylinders skip `enforce_sign_convention()`** to preserve quadrant semantics
- Ellipsoids still use `enforce_sign_convention()` as they don't have quadrant sensitivity

See `nsosim/wrap_surface_fitting/CLAUDE.md` for detailed explanation.

### SDF-Based Optimization
Wrap surface fitting uses:
- Signed distance functions to classify points as inside/outside wrap surfaces
- PCA or geometric initialization for faster convergence
- Squared-hinge margin loss for classification
- L-BFGS optimizer for final refinement

### Contact Mesh and Force Creation
For adding new contact surfaces (e.g., prefemoral fat pad):
```python
from nsosim.osim_utils import (
    create_contact_mesh, add_contact_mesh_to_model,
    create_articular_contact_force, add_contact_force_to_model
)

contact_mesh = create_contact_mesh(
    name='femur_bone_mesh',
    parent_frame='/bodyset/femur_distal_r',
    mesh_file='femur_prefemoral_fat_pad.stl',
    elastic_modulus=4e6,
    poissons_ratio=0.45,
    thickness=0.01,
)
add_contact_mesh_to_model(osim_model, contact_mesh)

contact_force = create_articular_contact_force(
    name='prefemoral_fat_pad_contact',
    socket_target_mesh='/contactgeometryset/femur_bone_mesh',
    socket_casting_mesh='/contactgeometryset/patella_cartilage',
)
add_contact_force_to_model(osim_model, contact_force)
```

## File Conventions

### Output File Formats
- `.vtk` - intermediate meshes (preserve point data arrays)
- `.stl` - OpenSim geometry files (final output)
- `.json` - alignment transforms, offsets, configuration
- `.npy` - NSM latent vectors

### Geometry Naming
```
{bone}_nsm_recon_osim.stl          # bone surface for OpenSim
{bone}_articular_surface_osim.stl   # cartilage contact surface
{side}_men_osim.stl                 # meniscus mesh
{side}_men_upper_art_surf_osim.stl  # meniscus superior surface
{side}_men_lower_art_surf_osim.stl  # meniscus inferior surface
femur_prefemoral_fat_pad.stl        # fat pad contact surface
```

## Environment & Dependencies

**Python 3.9** — pinned because opensim (JAM/COMAK fork) was built from source against numpy 2.0.2 on Python 3.9. Changing Python or numpy version requires a full opensim rebuild.

Lock files for reproducibility:
- `requirements-lock.txt` — full pip freeze of the validated `comak` conda environment
- `conda-env-lock.txt` — conda package list including system-level deps

### Key External Dependencies

- **opensim 4.5** (JAM/COMAK fork): Built from source, not pip-installable. Pinned to numpy 2.0.2.
- **NSM** (`github.com/gattia/nsm`): Neural Shape Model library, installed from source
- **pymskt** (`github.com/gattia/pymskt`): Medical Shape and Kinematics Toolkit, installed from source
- **PyTorch 2.3.1**: Used in wrap surface fitting optimization (optional dep via `pip install -e ".[fitting]"`)
- **PyVista 0.45.3**: 3D mesh visualization and manipulation

## Configuration Objects

### DEFAULT_SMITH2019_BONES
Defines wrap surfaces per bone body (from `wrap_surface_fitting/config.py`):
```python
DEFAULT_SMITH2019_BONES = {
    'femur': {
        'wrap_surfaces': {
            'femur_r': {'ellipsoid': [...], 'cylinder': [...]},
            'femur_distal_r': {'cylinder': [...]}
        }
    },
    'tibia': {...},
    'patella': {...}
}
```

### DEFAULT_FITTING_CONFIG
Fitting parameters for ellipsoid and cylinder optimizers:
- `constructor` params: lr, epochs, use_lbfgs, alpha/beta/gamma loss weights
- `fit` params: margin, plot

## Ligament Stiffness Updates
For modifying ligament properties (e.g., patellar tendon stiffness):
```python
DICT_LIGAMENTS_UPDATE_STIFFNESS = {
    'PT1': {'default_stiffness': 3000, 'update_factor': 4.0},
    ...
}
```


## Known Issues

### `recon_mesh()` Mesh Name Mapping (RESOLVED)
`recon_mesh()` previously used a count-based heuristic to infer mesh names from decoder output count. This was resolved by adding `get_mesh_names(model_config)` to `utils.py` (commit `29a32c2`), which reads `mesh_names` from the model config if present, with a fallback to the legacy `(bone, objects_per_decoder)` lookup. All 7 production model configs now have explicit `mesh_names`. Both `recon_mesh()` and `decode_latent_to_osim()` use `get_mesh_names()`. See `.claude/plans/completed/mesh-name-mapping.md`.

### Meniscus Articular Surface Instability
`create_meniscus_articulating_surface()` produces stochastic variation in the medial meniscus inferior surface (~40% point count variation, ASSD ~0.455mm across identical runs). Root causes: non-deterministic marching cubes in NSM reconstruction, non-deterministic mesh resampling, and ray-casting sensitivity to topology changes. See `MENISCUS_ARTICULAR_SURFACE_INSTABILITY.md` for full analysis and proposed fixes.