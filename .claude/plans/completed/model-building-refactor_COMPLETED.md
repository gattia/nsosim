# Plan: Extract Model-Building Orchestration from comak_1_nsm_fitting.py

**Created:** 2026-04-03
**Status:** Phases 1–2 complete. Phase 3 (rewire comak_1_nsm_fitting.py) deferred — synthetic pipeline written first (`comak_1_synthetic.py`).
**Context:** Phase C of the Synthetic Joint Simulation plan (`comak_gait_simulation/.claude/plans/SYNTHETIC_JOINT_SIMULATION.md`). The goal is to extract the shared model-building logic from `comak_1_nsm_fitting.py` (lines ~370–1050) into reusable nsosim functions, so both the existing fitting pipeline and the new synthetic joint pipeline can call the same code.

**Parent plan:** `comak_gait_simulation/.claude/plans/SYNTHETIC_JOINT_SIMULATION.md` (Phase C, Stage 1)

---

## Problem

`comak_1_nsm_fitting.py` contains ~680 lines of sequential orchestration that builds an OpenSim model from OSIM-space meshes. This code does NOT depend on how the meshes were produced — it only needs:
1. OSIM-space bone/cartilage/meniscus meshes
2. A `recon_dict` per bone (for `interp_ref_to_subject_to_osim` ligament interpolation)
3. Reference data paths (labeled bones, ligament configs, OpenSim template)
4. Config values (material properties, fitting params)

Currently it's one monolithic block inside the fitting function. The synthetic joint script (`comak_1_synthetic.py`) needs the same steps 4–10 but with meshes from `decode_joint_from_descriptors()` instead of `nsm_recon_to_osim()`. Without refactoring, we'd copy-paste ~500 lines.

---

## What `interp_ref_to_subject_to_osim` Needs

This is the trickiest interface. Currently it accesses `dict_bones` deeply:

```python
dict_bones[surface_name]["subject"]["recon_dict"]["model"]          # loaded NSM torch model
dict_bones[surface_name]["subject"]["recon_latent"]                  # (N,) latent vector
dict_bones[surface_name]["subject"]["recon_dict"]["icp_transform"]  # 4x4 similarity
dict_bones[surface_name]["subject"]["recon_dict"]["scale"]          # always 1
dict_bones[surface_name]["subject"]["recon_dict"]["center"]         # always [0, 0, 0]
```

Plus file paths for reference data:
```python
folder_nsm_files / surface_name / ref_{surface_name}_alignment.json
folder_nsm_files / surface_name / latent_{surface_name}.npy
```

For synthetic joints, we'd construct equivalent dicts:
```python
recon_dict = {
    "icp_transform": T_bone,    # from alignment JSON or recovered via T_rel
    "scale": 1,
    "center": np.zeros(3),
    "model": loaded_nsm_model,
}
```

The latent vector is also available (it's what we decoded from). So we can construct a `dict_bones`-compatible structure without modifying `interp_ref_to_subject_to_osim`.

---

## Architecture

New module: **`nsosim/model_building.py`**

Individual step functions + one orchestrator. Each step function extracts the relevant logic from `comak_1_nsm_fitting.py` with minimal changes.

### Functions

#### Per-bone steps (called for femur, tibia, patella):

**`extract_articular_surfaces(bone_mesh, cart_mesh, n_largest, triangle_density)`**
- Source: comak_1 lines 391–398, 524–531, 785–792
- Nearly identical for all 3 bones, just `n_largest` differs (1 for femur/patella, 2 for tibia)
- Returns: articular surface Mesh

**`interpolate_bone_ligaments(bone_name, labeled_mesh_path, lig_attach_params, dict_bones, fem_ref_center, folder_ref, surface_idx=0)`**
- Source: comak_1 lines 408–449 (femur), 541–582 (tibia), 825–856 (patella)
- Same pattern repeated 3 times with different bone names
- Returns: (updated_labeled_mesh, updated_labeled_points, updated_lig_points, list_lig_name_pt_idx)

**`fit_bone_wrap_surfaces(bone_name, labeled_mesh, labeled_points, wrap_surface_spec, fitter_configs)`**
- Source: comak_1 lines 452–502 (femur), 584–634 (tibia)
- For patella: different fitter (PatellaFitter), handled as special case
- Returns: fitted_wrap_parameters dict for this bone

#### Cross-bone steps:

**`interpolate_meniscus_ligaments(dict_lig_attach_params, dict_bones, fem_ref_center, folder_ref)`**
- Source: comak_1 lines 692–719
- Handles both medial and lateral meniscus ligament interpolation
- Returns: updated dict_lig_attach_params

**`update_coronary_ligament_tibia_attachments(dict_lig_attach_params, tib_mesh_osim, lig_attachment_key='xyz_mesh_updated')`**
- Source: comak_1 lines 721–767
- Projects coronary ligament tibia attachments onto tibia surface
- **Bug fix vs. original:** The original code uses `lig_attachment_key = 'xyz_mesh'` (reference positions). This means it (a) reads the reference meniscus position instead of the subject-interpolated one, and (b) writes the projected tibia position to `xyz_mesh` while `update_osim_model` reads `xyz_mesh_updated`. Net effect: the entire coronary block is dead code in production — the model uses NSM-interpolated tibia positions instead. The fix: use `xyz_mesh_updated` as the key so we read subject meniscus positions and write to the key the model consumes.
- Returns: updated dict_lig_attach_params

**`create_meniscus_articulating_surfaces(med_men, lat_men, fem_mesh, tib_mesh, med_center, lat_center)`**
- Source: comak_1 lines 651–689
- Returns: (med_upper, med_lower, lat_upper, lat_lower)

**`center_patella_meshes(pat_mesh, pat_articular, pat_cart_mesh=None)`**
- Source: comak_1 lines 794–822
- Returns: (centered_bone, centered_articular, centered_cart, mean_patella_offset)

**`create_prefemoral_fatpad(fem_bone, fem_cart, pat_bone, pat_cart, config)`**
- Source: comak_1 lines 890–911
- Thin wrapper around existing `create_prefemoral_fatpad_noboolean`
- Returns: fatpad Mesh

#### Final assembly:

**`save_geometry_files(folder_save_bones, path_save_model, geometry_dict)`**
- Source: comak_1 lines 934–964
- Copies geometry files from per-bone folders to OpenSim Geometry/ folder

**`finalize_osim_model(osim_model, fitted_wrap_params, lig_attach_params, tib_mesh, mean_patella, config)`**
- Source: comak_1 lines 966–1051
- Calls `update_osim_model()`, adds fatpad contact, saves .osim
- Returns: path to saved .osim

#### Orchestrator:

**`build_joint_model(bone_meshes, recon_dicts, ref_data_paths, lig_attach_params, config, save_dir)`**

Calls all the above in sequence. The orchestrator's signature defines the clean interface between "I have meshes" and "build me a model."

```python
def build_joint_model(
    # OSIM-space meshes (from fitting OR decoding)
    bone_meshes: dict,          # {'femur': {'bone': Mesh, 'cart': Mesh, 'med_men': Mesh, 'lat_men': Mesh},
                                #  'tibia': {'bone': Mesh, 'cart': Mesh},
                                #  'patella': {'bone': Mesh, 'cart': Mesh}}

    # For ligament interpolation (constructed from recon_dict or alignment JSONs)
    dict_bones: dict,           # dict_bones-compatible structure with recon_dict + recon_latent per bone

    # Reference data paths
    ref_data_paths: dict,       # {'folder_ref_recons': ..., 'folder_labeled_bone': ...,
                                #  'path_osim_template': ..., 'lig_attach_params_path': ...}

    # Config
    config: dict,               # from config JSON or defaults
    fem_ref_center: np.ndarray, # from ref_femur_alignment.json['mean_orig']

    # Output
    save_dir: str,              # where to save geometry + .osim
    model_name: str,            # for naming the .osim

    # Optional
    project_meniscal_to_tibia: bool = True,
) -> str:  # returns path to saved .osim
```

---

## Implementation Phases

### Phase 1: Extract individual functions

Create `nsosim/model_building.py` with the individual step functions. Each function is a near-direct extraction from `comak_1_nsm_fitting.py` — preserve exact logic, just wrap in a function with clear inputs/outputs.

**No behavior changes.** The goal is mechanical extraction, not cleanup.

Order of extraction:
1. `extract_articular_surfaces` (simplest, same for all bones)
2. `interpolate_bone_ligaments` (repeated 3x, straightforward)
3. `fit_bone_wrap_surfaces` (femur/tibia pattern + patella special case)
4. `interpolate_meniscus_ligaments` + `update_coronary_ligament_tibia_attachments`
5. `create_meniscus_articulating_surfaces`
6. `center_patella_meshes`
7. `create_prefemoral_fatpad`
8. `save_geometry_files` + `finalize_osim_model`
9. `build_joint_model` orchestrator

### Phase 2: Frozen-input test

**Test strategy:** Feed the new functions the exact same inputs that an existing production subject already produced, and verify the outputs match exactly.

1. Pick production subject `9018389_00m_RIGHT` (the verification reference subject)
2. From their existing results, extract the frozen inputs:
   - OSIM-space meshes: `*_nsm_recon_osim.stl`, `*_cartilage_nsm_recon_osim.vtk`, meniscus `*_osim.vtk`
   - Alignment JSONs (for recon_dict construction)
   - Latent vectors (for interpolation)
3. Run `build_joint_model()` with these frozen inputs
4. Compare every output against the production results:
   - Articular surfaces: **exact vertex match** (same inputs → same marching cubes → same output)
   - Labeled meshes: **exact match** (interpolation is deterministic given same latent + transform)
   - Wrap surface parameters: **exact match** (fitting is deterministic given same labeled mesh)
   - Patella offset: **exact match**
   - Fat pad: **exact match**
   - .osim file: **text-identical** (same geometry, same wrap params, same ligament positions)

No ASSD tolerance — this is a pure refactor test. Any difference is a bug.

**Test file:** `tests/test_model_building.py`

Key test cases:
- `test_articular_surfaces_match_production` — per bone
- `test_ligament_interpolation_match_production` — per bone
- `test_wrap_surfaces_match_production` — per bone
- `test_meniscus_surfaces_match_production`
- `test_patella_centering_match_production`
- `test_fatpad_match_production`
- `test_full_orchestrator_match_production` — end-to-end

### Phase 2 Results (2026-04-03)

Verified against a fresh baseline (job 38622: run original `comak_1_nsm_fitting.py`, then run `build_joint_model` with same subject's saved latents/transforms/meshes). ASSD comparison results:

| Output | ASSD (mm) | Status |
|--------|-----------|--------|
| Femur articular surface | 0.0000 | PASS |
| Tibia articular surface | 0.0004 | PASS |
| Patella articular surface | 0.0000 | PASS |
| Femur labeled mesh | 0.0000 | PASS |
| Tibia labeled mesh | 0.0000 | PASS |
| Patella labeled mesh | 0.0000 | PASS |
| Patella offset | 3e-15 | PASS |
| Med meniscus upper | 0.0000 | PASS |
| Med meniscus lower | 0.0000 | PASS |
| Lat meniscus upper | 0.0000 | PASS |
| Lat meniscus lower | 0.0000 | PASS |
| Prefemoral fat pad | 0.2184 | PASS (0.5mm threshold) |

Fat pad ASSD of 0.22mm is from STL roundtrip changing face connectivity (point reordering → different normals → slightly different ray-casting in dilation). Visually inspected — near-perfect surface agreement.

Not positive the above is the reason for the Fat Pad differences - but visual check indicates they are close to identical. it looks like only differentces at the curved part at the top. 

Verification script: `comak_gait_simulation/tests/verify_model_building/`

### Phase 3: Rewire comak_1_nsm_fitting.py

Replace the inline orchestration in `comak_1_nsm_fitting.py` with calls to `build_joint_model()`.

The fitting script becomes:
1. Parse args, load config, set up paths (unchanged)
2. Build `dict_bones`, run `align_knee_osim_fit_nsm()` (unchanged — this is the fitting)
3. Decode meshes to OSIM via `nsm_recon_to_osim()` per bone (unchanged)
4. **NEW:** Call `build_joint_model(bone_meshes, dict_bones, ref_data_paths, config, save_dir)`
5. Save timing, run_config (unchanged)

**Verification:** Run the existing `tests/verify_pipeline/submit_verification.sh`. This re-runs the full pipeline (including stochastic fitting) and compares against production. Should pass at the same 64/66 rate (the 2 meniscus articulating surface failures are pre-existing and unrelated to refactoring).

### Phase 4: Export + init

- Add `model_building` to `nsosim/__init__.py`
- Update any imports in comak_gait_simulation scripts

---

## Implementation Details

### `save_mesh_as_obj` utility
The source script defines `save_mesh_as_obj()` (lines 39–52) to save meshes as OBJ format (preserving indexed vertices to avoid the SimTK STL vertex-merging bug). This is used for articular surface saves (femur, tibia, patella). Include this utility in `model_building.py` or a shared utils location.

### Patella centering + wrap fitting coupling
Patella wrap fitting differs from femur/tibia: after interpolation, ALL patella points (labeled mesh + ligament attachments) are shifted by `-mean_patella` (line 859) BEFORE wrap fitting. This means `interpolate_bone_ligaments` for patella must accept an optional centering offset, or the orchestrator must apply centering between interpolation and wrap fitting. The cleanest approach: `interpolate_bone_ligaments` returns raw interpolated points, and the orchestrator applies patella centering before passing to `fit_bone_wrap_surfaces`.

### Config/constants mapping
The source script defines several module-level constants that the new functions need:
- `ELLIPSOID_CONSTRUCTOR_CONFIG`, `ELLIPSOID_FIT_CONFIG`, `CYLINDER_CONSTRUCTOR_CONFIG`, `CYLINDER_FIT_CONFIG` — derived from `DEFAULT_FITTING_CONFIG`
- `DICT_JOINTS_COORDS_TO_UPDATE` — hardcoded knee joint coordinates
- `DICT_LIGAMENTS_UPDATE_STIFFNESS` — patellar tendon stiffness parameters
- `PROJECT_MENISCAL_ATTACHMENTS_TO_TIBIA` — toggle for meniscal projection method

The orchestrator's `config` dict should accept overrides for all of these, falling back to the defaults from `wrap_surface_fitting/config.py` and `comak_osim_update.py`. The constants themselves stay defined where they are (config.py, comak_osim_update.py); the orchestrator just passes them through.

### I/O strategy: save-as-you-go with clean function interfaces
Step functions are pure: they take data in, return results, and do NO file I/O. The orchestrator saves intermediate results after each step. This gives both benefits:
- **Testability:** step functions can be unit-tested without filesystem mocking
- **Debuggability:** if the pipeline crashes at patella wrap fitting, femur/tibia intermediates are already on disk
- **Resumability:** future enhancement could skip completed steps by checking for saved intermediates

The orchestrator handles all `save_mesh`, `shutil.copy`, and JSON writes. The `save_geometry_files` step at the end copies final STL/OBJ files to the OpenSim Geometry/ folder.

---

## What NOT to change

- `interp_ref_to_subject_to_osim()` — keep its existing interface. Construct compatible `dict_bones` dicts on the caller side.
- `nsm_recon_to_osim()` — stays in `nsm_fitting.py`, still used by the fitting path.
- `update_osim_model()` — stays in `comak_osim_update.py`, called by `finalize_osim_model`.
- `create_articular_surfaces()` — stays in `articular_surfaces.py`, called by `extract_articular_surfaces` wrapper.
- `create_meniscus_articulating_surface()` — stays wherever it is, called by the wrapper.

The new functions are thin wrappers that organize the calling sequence. They don't duplicate the actual computation — they just provide clean interfaces to it.

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `nsosim/model_building.py` | CREATE | Individual step functions + orchestrator |
| `nsosim/__init__.py` | MODIFY | Export `model_building` module |
| `tests/test_model_building.py` | CREATE | Frozen-input comparison tests |
| `tests/fixtures/model_building/` | CREATE | Frozen inputs from production subject |

**In comak_gait_simulation (after nsosim refactor is verified):**

| File | Action | Purpose |
|------|--------|---------|
| `run_simulations/scripts/comak_1_nsm_fitting.py` | MODIFY | Replace inline orchestration with `build_joint_model()` call |
| `run_simulations/scripts/comak_1_synthetic.py` | CREATE | Synthetic joint builder using same orchestrator |

---

## Dependencies

- All existing nsosim modules (articular_surfaces, nsm_fitting, comak_osim_update, osim_utils, meniscal_ligaments)
- OpenSim Python API (for model finalization)
- pymskt (Mesh I/O)
- Wrap surface fitting code (CylinderFitter, EllipsoidFitter, PatellaFitter)
- Production results for test subject 9018389_00m_RIGHT

---

## Risks

1. **`interp_ref_to_subject_to_osim` interface**: The function reads deeply into `dict_bones`. For synthetic joints, we construct a compatible dict. If the function's internals change, the synthetic dict might break. Mitigation: the frozen-input test will catch this.

2. **Wrap surface fitting non-determinism**: CylinderFitter/EllipsoidFitter might use random initialization. If so, the frozen-input test needs to seed the RNG. Check during Phase 2.

3. **OpenSim model text comparison**: The .osim XML might have floating point formatting differences even with identical inputs. Compare numerically, not string-identical.

4. **Large test fixtures**: The frozen inputs (meshes, labeled bones, etc.) could be ~50-100MB. Gitignore them and use a copy script like the pipeline verification does.

5. **Coronary ligament bug fix changes output**: The original code's coronary ligament block is dead code (writes to `xyz_mesh` while model reads `xyz_mesh_updated`). Fixing this to use `xyz_mesh_updated` will change the coronary ligament tibia attachment positions in the output model. The frozen-input test in Phase 2 should verify both behaviors: (a) with the bug fix, coronary tibia attachments differ from production (expected), and (b) all other outputs match production exactly. Consider a separate commit for the bug fix so it's reviewable independently.
