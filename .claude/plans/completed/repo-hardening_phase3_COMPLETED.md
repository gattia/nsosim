# Phase 3: Validation & Safety Guards — COMPLETED

**Date:** 2026-02-18
**Result:** 154 passed, 0 failed (excl. pre-existing flaky test), lint clean
**New tests added:** 33 (Phase 1: 65, Phase 2: 51, Phase 3: 33, Total: 149+)

---

## What was done vs. the plan

### 3.1 Define data structure schemas — DONE

| Planned | Done | Notes |
|---------|------|-------|
| Create `nsosim/schemas.py` | Yes | New module with `ValidationError` + 3 validation functions |
| `DictBones` TypedDict/dataclass | Done as validation function | `validate_dict_bones()` checks structure at runtime; TypedDicts add type-checker overhead without runtime benefit for this project's stage |
| `FittedWrapParameters` schema | Done as validation function | `validate_fitted_wrap_parameters()` walks the 4-level nested dict |
| `AlignmentTransform` schema | Skipped | Structure is simple (3 keys), always created inline in `align_knee_osim_fit_nsm()`, not passed between modules — validation here adds no safety |
| `validate_dict_bones()` | Yes | Checks: top-level types, required sections (subject/ref/model), required keys per section, bone-specific keys (femur needs meniscus filenames), meniscus entry (ref-only) |
| `validate_fitted_wrap_parameters()` | Yes | Checks: 4-level nesting (bone→body→wrap_type→wrap_name), valid wrap_type ∈ {cylinder, ellipsoid}, wrap_surface attributes (translation, rotation, radius/length for cylinder, dimensions for ellipsoid) |
| `validate_surface_idx()` | Yes (bonus) | Validates int type and range [0, max_idx] |

**Design decisions:**
- `ValidationError` inherits from `ValueError` for backwards compatibility with existing `except ValueError` handlers
- Validation functions use descriptive error messages with full key paths (e.g., `"dict_bones['femur']['subject'] missing keys: ['bone_filename']"`)
- No TypedDicts or dataclasses — runtime validation functions are more appropriate here since the dicts are built dynamically from JSON/file paths and TypedDicts only help static type checkers

### 3.2 Add validation calls at pipeline entry points — DONE

| Planned | Done | Notes |
|---------|------|-------|
| `align_knee_osim_fit_nsm()`: validate `dict_bones` | Yes | Added at top of function, before bone ordering |
| `update_osim_model()`: validate `dict_wrap_objects` | Yes | Added at top of function, before geometry updates |
| `nsm_recon_to_osim()`: validate `surface_idx` | Adjusted | `nsm_recon_to_osim()` doesn't take `surface_idx`. Added validation to `interpolate_ref_points_nsm_space()` which does |

### 3.3 Make `recon_mesh()` internal index mapping explicit — DONE

| Planned | Done | Notes |
|---------|------|-------|
| Replace count-based heuristic | Yes | Uses `model_config.get("bone")` to determine expected mesh count and index mapping |
| Assert mesh count matches expected | Yes | Raises `ValueError` with descriptive message if count doesn't match |
| Keep fallback for unknown bone types | Yes | Falls back to old count-based heuristic for forward compatibility |

**Implementation:**
```python
_EXPECTED_EXTRA_MESHES = {
    "femur": {"count": 4, "names": ["med_men_mesh", "lat_men_mesh"]},
    "tibia": {"count": 3, "names": ["fibula_mesh"]},
    "patella": {"count": 2, "names": []},
}
```
The first two meshes (bone, cart) are always assigned. Additional meshes are mapped by name based on bone type. If `model_config["bone"]` is not in the lookup, the old heuristic is used as fallback.

---

## Tests added (33 total)

`tests/test_schemas.py`:

- **TestValidateDictBones** (13 tests): valid full dict, meniscus entry, empty dict, not-a-dict, missing required bone, missing subject/ref/model sections, missing subject key, femur missing meniscus keys, meniscus missing ref, bone dict not-a-dict, section not-a-dict
- **TestValidateFittedWrapParameters** (11 tests): valid params, not-a-dict, invalid wrap type, cylinder missing radius/length, ellipsoid missing dimensions, missing translation/rotation, body/wrap dicts not-a-dict, cylinder radius wrong type
- **TestValidateSurfaceIdx** (8 tests): valid indices [0-3], negative, too large, float type, custom max_idx
- **TestValidationError** (1 test): confirms it's a ValueError subclass

---

## Things to know for future work

### Fixed: Flaky cylinder center tests → surface overlap tests

Both `TestCylinderFitter::test_recovers_center` and `TestCylinderFitterRotated::test_recovers_center` were flaky because the cylinder center along its axis is geometrically ambiguous for SDF-based fitting (SDF is constant along the barrel axis). Replaced with `test_fitted_surface_covers_true_points` which evaluates the fitted cylinder's SDF at points on the true surface (interior 80% to avoid cap edges). This tests what actually matters — surface overlap — and is invariant to axial sliding. Passes consistently across multiple runs.

### `ValidationError` is importable from `nsosim.schemas`

Other modules can use:
```python
from nsosim.schemas import ValidationError
```
It's already imported in `nsm_fitting.py` (though not currently used directly — the import is there for API convenience if callers want to catch it).

### Unused `ValidationError` import in `nsm_fitting.py`

The `from .schemas import ValidationError` import in `nsm_fitting.py` is currently unused directly (the functions called raise it internally). This is intentional — it makes `ValidationError` accessible at the `nsm_fitting` module level for callers who want to catch it. If isort/linting flags it later, either add `# noqa: F401` or remove it and let callers import from `schemas` directly.
