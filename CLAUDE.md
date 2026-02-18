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
pytest             # Test suite is minimal — being built out (see .claude/plans/repo-hardening.md)
                   # `make test` is currently commented out in the Makefile

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

## Architecture

### Core Modules

| Module | Purpose |
|--------|---------|
| `nsm_fitting.py` | NSM fitting pipeline: align meshes → fit NSM → convert to OpenSim coords |
| `articular_surfaces.py` | Extract/refine cartilage contact surfaces, meniscus processing, prefemoral fatpad, patella optimization |
| `comak_osim_update.py` | Update OpenSim XML with subject-specific meshes, attachments, wrap surfaces |
| `osim_utils.py` | Low-level OpenSim XML manipulation via Python API |
| `utils.py` | NSM model loading, mesh I/O, anatomical coordinate system (ACS) alignment utilities |


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

Two coordinate systems and unit conventions:
- **NSM space**: Millimeters (mm), coordinate system used by Neural Shape Models during fitting
- **OpenSim space**: Meters (m), target coordinate system for OpenSim models

Key conversion functions:
- `convert_OSIM_to_nsm()` / `convert_nsm_recon_to_OSIM()` - transform between spaces
- `nsm_recon_to_osim()` - reconstruct mesh in OpenSim coordinates (returns dict with 'bone', 'cart', 'med_men', 'lat_men' keys)

**Important**: `fem_ref_center` (femoral reference center) from the alignment file is used for ALL bones to maintain consistent spatial relationships.

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

## External Dependencies

- **NSM** (Neural Shape Model): Custom library at `github.com/gattia/nsm` - must be installed first
- **OpenSim**: Python bindings for biomechanical simulation
- **PyTorch 2.0**: Used in wrap surface fitting optimization
- **pymskt**: Medical Shape and Kinematics Toolkit
- **PyVista**: 3D mesh visualization and manipulation

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

### Meniscus Articular Surface Instability
`create_meniscus_articulating_surface()` produces stochastic variation in the medial meniscus inferior surface (~40% point count variation, ASSD ~0.455mm across identical runs). Root causes: non-deterministic marching cubes in NSM reconstruction, non-deterministic mesh resampling, and ray-casting sensitivity to topology changes. See `MENISCUS_ARTICULAR_SURFACE_INSTABILITY.md` for full analysis and proposed fixes.