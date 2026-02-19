# Post-Review Hardening Plan

After completing Phases 1–5 of the original repo-hardening plan, a second review of the
full codebase identified remaining issues. This plan addresses them in priority order.

**Relationship to original plan:** Original Phase 6 (CI) is partially done — the GitHub
Actions workflow exists but needs config fixes (handled in Phase 8 below). Original
Phase 7 (type annotations on public API) was deprioritized and is not included here.

**Context constraints:**
- OpenSim is custom-built from source (JAM/COMAK fork). Standard conda `opensim` won't work
  for running the full pipeline. GitHub Actions CI **cannot** run integration tests that
  touch `opensim` or the full NSM→OSIM pipeline.
- CI is currently lint-only + unit tests that don't require opensim or real mesh data.
- Any refactoring of core fitting logic (`fit()` decomposition) should wait until the
  current code has been validated against real pipeline scripts.

---

## Execution Instructions

Same as the original plan: execute one phase at a time, verify with `pytest` and `make lint`
after each phase, commit before running `make autoformat`.

---

## Phase 7: Bug Fixes & Dead Code Cleanup (High — Zero Risk)

These are confirmed bugs or dead code found during review. Pure cleanup, no behavior change.

### 7.1 Replace debug `print()` with `logger.debug()` in wrap_surface_fitting

Phase 4 converted `print()` → `logging` in the core modules but missed the submodule.
These are unconditional `print()` calls that bypass the logging system and can't be turned off.

**Files and locations:**

| File | Lines | Content |
|------|-------|---------|
| `fitting.py` | 718–726 | `print(f"DEBUG: Final axis vector...")` — 4 print calls inside `fit()` |
| `surface_param_estimation.py` | 96–100, 146–227 | ~20 `print(f"[DEBUG]...")` calls in `fit_cylinder_geometric()` |
| `surface_param_estimation.py` | 359–399 | ~8 `print(f"[DEBUG]...")` calls in `fit_ellipsoid_algebraic()` |
| `utils.py` | 416, 421 | 2 `print(f"[INFO]...")` calls in unit detection logic |
| `parameter_extraction.py` | 90 | 1 `print()` call |

**Action:** Replace all `print(f"[DEBUG]...")` with `logger.debug(...)`. Replace
`print(f"[INFO]...")` with `logger.info(...)`. Each file already has
`logger = logging.getLogger(__name__)` so no new imports needed except in
`surface_param_estimation.py` and `parameter_extraction.py` where logger setup may need adding.

### 7.2 Remove dead code in `fitting.py`

| Location | What | Action |
|----------|------|--------|
| `fitting.py:70–96` | `_compute_loss()` has `pass` + entire implementation commented out | Remove the `pass` and comments; raise `NotImplementedError` instead (same pattern as `wrap_params` at line 732) |
| `fitting.py:1290` | Commented `# return (d**2).mean()` in EllipsoidFitter | Delete the comment |
| `fitting.py:1399–1400` | Commented `# 5e-7,` parameter scale | Delete the comment |

### 7.3 Remove duplicate import in `nsm_fitting.py`

`from pymskt.mesh import Mesh` appears at both line 10 and line 18. Remove the duplicate.

### 7.4 Remove unused `scale_factor` in `wrap_surface_fitting/utils.py`

Lines 415–426: `scale_factor = 1000.0` is computed but never used. The threshold is applied
directly without scaling. Remove the dead variable and the misleading print to avoid
confusion about whether thresholds are in meters or mm.

---

## Phase 8: Packaging & CI Fixes (High — Config Only)

### 8.1 Populate `pyproject.toml` dependencies

Currently `dependencies = []` (empty). Anyone doing `pip install nsosim` gets zero
dependencies installed. Move the actual requirements into `pyproject.toml`.

**Dependencies to add:**
```toml
dependencies = [
    "numpy",
    "scipy",
    "pyvista",
    "vtk",
    "pymskt",
    "nibabel",
]

[project.optional-dependencies]
fitting = ["torch>=2.0"]
```

`torch` is an optional extra (large, has CPU/CUDA/ROCm variants — users need to
install the right build for their hardware). Document install as
`pip install nsosim[fitting]` or manual torch install.

**NOT included (must be installed separately — document this):**
- `opensim` — requires JAM/COMAK fork built from source, not pip-installable
- `NSM` — custom library from `github.com/gattia/nsm`, installed via
  `pip install git+https://github.com/gattia/nsm.git` or from local source

Add a note in `pyproject.toml` comments and update the install section of `README.md`
to document these manual prerequisites. Also add a comment block at the top of
`.github/workflows/ci.yml` documenting CI limitations (lint + unit tests only, no
opensim or real data).

### 8.2 Fix `requires-python` to match CI reality

`pyproject.toml` says `requires-python = ">=3.8"` but CI only tests 3.10 and 3.11.
The code uses `list[float]` syntax (`osim_utils.py:480`) which requires Python 3.9+.
Change to `requires-python = ">=3.10"` to match what's actually tested and supported.

### 8.3 Sync `requirements.txt` with `pyproject.toml`

After 8.1, `requirements.txt` becomes redundant for pip installs but is still used by
CI and conda workflows. Update it to match:
- Fix `mskt` → `pymskt` (the pip package name is `pymskt`, not `mskt`)
- Add `torch` (currently missing, CI installs it separately)
- Keep `#NSM` commented with a note about manual installation

---

## Phase 9: Validation Hardening (Medium)

### 9.1 Replace `assert` with proper validation in `osim_utils.py`

`assert` statements are disabled by `python -O`. These are used for input validation
on data coming from external sources (OpenSim models, user dicts), so they should be
real `if/raise` checks.

**Locations (13 total):**
- Lines 401–473: 10 `assert isinstance(...)` calls in contact mesh/force creation functions
- Line 692: `assert force_dict["name"] == force_name`
- Line 704: `assert path_point_set.getSize() == len(force_dict["points"])`
- Line 885: `assert force_dict_["class"] == ...`

**Action:** Replace each with `if not ...: raise ValueError(...)` or `TypeError(...)` as
appropriate. Keep the same error semantics, just make them survive `-O` mode.

### 9.2 Add positive-value validation in `schemas.py`

`validate_fitted_wrap_parameters()` checks that `radius` and `length` are numeric but
doesn't check they're positive. Add `if val <= 0` checks for cylinder `radius`/`length`
and ellipsoid `dimensions` elements.

---

## Phase 10: Documentation Fixes (Low — Quick Wins)

### 10.1 Document in-place mutation and clarify constants

These functions mutate their inputs but don't say so in the docstring. Add a
`.. warning:: Modifies input in-place.` or `Note: Mutates ...` line to each.

| Function | File | What it mutates |
|----------|------|-----------------|
| `convert_nsm_recon_to_OSIM_()` | `nsm_fitting.py` | Input point arrays via `+=`, `/=` |
| `convert_OSIM_to_nsm_()` | `nsm_fitting.py` | Input point arrays via `*=`, `-=` |
| `add_polar_coordinates_about_center()` | `articular_surfaces.py` | Adds point data arrays to mesh |

Also add a one-line comment to `ROUND_DIGITS = 6` in `osim_utils.py:11` explaining
this is the decimal precision for OpenSim XML coordinates (6 digits ≈ 1 micrometer in meters).

---

## Phase 11: Nice-to-Have (Low Priority — Do Later)

These are good ideas but not urgent. Tackle after validating the current code works
with real pipeline scripts.

### 11.1 Decompose `CylinderFitter.fit()` (347 lines)

Extract into helper methods: `_adam_training_loop()`, `_lbfgs_refinement_stage()`,
`_validate_final_rotation()`. **Wait until the current code is tested against production
scripts** — refactoring a working optimizer before validation is risky.

### 11.2 Add degenerate input tests

Test fitters with: NaN inputs, all-identical points, collinear points, contradictory
labels, empty point clouds. These catch edge cases but aren't blocking.

### 11.3 Add end-to-end integration test

Label → fit → validate SDF pipeline test with synthetic data. Valuable but requires
careful design to avoid being fragile.

---

## Execution Order

1. **Phase 7** (bug fixes + dead code) — highest value, zero risk, pure cleanup
2. **Phase 8** (packaging + CI) — config-only, prevents broken installs
3. **Phase 9** (validation hardening) — assert→raise, positive-value checks
4. **Phase 10** (docs) — quick docstring additions
5. **Phase 11** (nice-to-have) — after production validation

---

## Verified Non-Issues

These were flagged during review but confirmed to be safe after inspection:

| Flag | Verdict | Reason |
|------|---------|--------|
| `utils.py:327` `NameError` on `temp_bone_mesh_path` | **Not a bug** | Guard at line 325 (`clip_bone and ... and mesh_paths[0] is not None`) exactly mirrors the guard at line 274 that defines the variable. All code paths are safe. |
| `torch.cross()` in `surface_param_estimation.py:238` | **Not a bug** | Uses `torch.cross()` correctly for 1D 3-element vectors. The deprecated form was only for batched input without `dim=`. |
