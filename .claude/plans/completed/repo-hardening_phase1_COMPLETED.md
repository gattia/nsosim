# Phase 1: Test Infrastructure & Core Math Tests — COMPLETED

**Date:** 2026-02-17
**Result:** 65 passed, 0 failed (5.67s, `conda run -n comak python -m pytest tests/ -v`)

---

## What was done vs. the plan

### 1.1 Test infrastructure — DONE

| Planned | Done | Notes |
|---------|------|-------|
| Add `tests/__init__.py` | Yes | Empty file for pytest discovery |
| Add `tests/wrapping/__init__.py` | Yes | Empty file for pytest discovery |
| Create `tests/conftest.py` with shared fixtures | Yes | Fixtures: `identity_rotation_np`, `identity_rotation_torch`, `known_90deg_x_rotation_np`, `random_rotation_torch`, `unit_sphere_points`, `origin_point`, `sample_dict_bones` |
| Uncomment `make test` target in Makefile | Yes | Lines 36-38 uncommented |
| Fix `test_sign_convention.py` to use real imports + pytest assertions | Yes | Replaced `MockRotationUtils` with real `RotationUtils` import; replaced all `print()` with `assert` |

### 1.2 Rotation & coordinate transform tests — DONE (27 tests)

`tests/test_rotation_utils.py` covers:

- **TestQuatRotRoundTrip** (6 tests + 6 parametrized = 10 total): identity, random rotation, 90deg X, unit quaternion check, Z-rotations at 0/30/45/90/120/180 degrees
- **TestEulerRoundTrip** (5 tests + 4 parametrized = 8 total): identity→zero euler, matches scipy intrinsic XYZ, euler→matrix→euler roundtrip, numpy/torch return types, single-axis rotations at 30/45/60/89 degrees validated against scipy
- **TestAxisAngleRoundTrip** (3 tests): identity, roundtrip, known 90deg Z
- **TestEnforceSignConventionIdempotency** (5 parametrized tests): 5 random scipy rotations, apply twice = same result

### 1.3 SDF tests — DONE (15 tests)

`tests/test_wrap_signed_distances.py` covers:

- **TestSdEllipsoidImproved** (8 tests): center negative, surface ~0, outside positive, symmetry on unit sphere, distance approximation (near + far), different axes, translated center, batch of points
- **TestSdCylinderWithAxis** (7 tests): center negative, radial surface ~0, outside radial, outside axial (cap), radial symmetry, translated, known radial distance

### 1.4 Coordinate system transform tests — DONE (8 tests)

`tests/test_coordinate_transforms.py` covers:

- **TestOsimToNsmTransform** (3 tests): orthogonality (R^T R = I), determinant ±1, shape
- **TestCoordinateRoundTrip** (3 tests): NSM→OSIM→NSM, OSIM→NSM→OSIM, origin with zero bias
- **TestUnitConversion** (2 tests): 1000mm→1m, 1m→1000mm

### 1.5 wrap_surface data class tests — DONE (9 tests)

`tests/test_wrap_surface.py` covers:

- **TestWrapSurfaceConstruction** (3 tests): cylinder, ellipsoid, numpy array storage
- **TestToDict** (4 tests): all keys present, numpy→list conversion, None preserved, scalars preserved
- **TestRoundTrip** (2 tests): cylinder construct→to_dict→reconstruct, ellipsoid same

---

## Things to know for future work

### Environment

- Tests must be run with `conda run -n comak python -m pytest tests/ -v`. The base conda env does not have pytest or the project dependencies.
- The `make test` target now works but assumes `comak` env is active. If running via `make test`, activate the env first: `conda activate comak && make test`.

### Ellipsoid SDF approximation behavior

`sd_ellipsoid_improved()` uses a gradient-based approximation (`|F(p)| / ||∇F(p)||`) that is:
- **Exact at the surface** (SDF = 0)
- **Accurate near the surface** (within ~0.3 of true distance for points within ~1.5× the radius)
- **Underestimates for far points** (at d=3 from a unit sphere: returns 1.33 instead of exact 2.0)

This is intentional — the function is used for inside/outside classification and optimization of shapes toward nearby mesh points. The test documents this limitation explicitly in its docstring. If someone later swaps in an exact SDF, the test tolerances for near-surface points can be tightened.

### Cylinder SDF location

The plan references `sd_cylinder()` but the actual function is `sd_cylinder_with_axis()` in `fitting.py` (not in `wrap_signed_distances.py`). There is no standalone `sd_cylinder` in the codebase — the old quaternion-based one from `WRAP_SURFACE_FITTING_REFACTORING.md` was replaced by the axis-vector version.

### `convert_nsm_recon_to_OSIM_` modifies in-place

Both `convert_nsm_recon_to_OSIM_()` and `convert_OSIM_to_nsm_()` mutate their input arrays in-place (`+=`, `/=`, `*=`). The round-trip tests work around this by passing `.copy()`. This is worth knowing if writing downstream tests that reuse point arrays.

### Deprecation warning

`construct_cylinder_basis()` in `fitting.py` uses `torch.cross()` without the `dim` argument, which is deprecated. This should be changed to `torch.linalg.cross()` in a future cleanup (Phase 4 or similar). The warning shows up in test output but doesn't affect correctness.

### conftest fixtures available

Any future test files can use these fixtures by just naming them as function parameters:
- `identity_rotation_np` / `identity_rotation_torch` — 3×3 identity
- `known_90deg_x_rotation_np` — 90° about X
- `random_rotation_torch` — deterministic "random" rotation (axis=[1,2,3], angle=1.2rad)
- `unit_sphere_points` — 100 points on unit sphere (numpy, seeded rng)
- `origin_point` — single (1,3) zero point
- `sample_dict_bones` — minimal dict_bones with femur/tibia/patella keys

### What Phase 2 should build on

Phase 2 (Synthetic Mesh Tests) will test the fitters (`CylinderFitter`, `EllipsoidFitter`) and articular surface functions. These tests will likely need:
- PyVista for generating synthetic meshes (cylinders, ellipsoids, tori) — verify it's in the comak env
- Larger tolerances since fitting is iterative/stochastic
- The conftest can be extended with mesh-generating fixtures as needed
