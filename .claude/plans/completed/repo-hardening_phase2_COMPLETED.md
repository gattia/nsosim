# Phase 2: Synthetic Mesh Tests — COMPLETED

**Date:** 2026-02-17
**Result:** 116 passed, 0 failed (14.84s, `conda run -n comak python -m pytest tests/ -v`)
**New tests added:** 51 (Phase 1: 65, Phase 2: 51, Total: 116)

---

## What was done vs. the plan

### 2.1 Fitter tests (`tests/test_fitting.py`) — DONE (18 tests)

| Planned | Done | Notes |
|---------|------|-------|
| `CylinderFitter`: fit to known cylinder, verify center/axis/radius | Yes | 5 tests: center (atol=2mm), radius (<5%), axis (cos_sim>0.999), wrap_params structure, PCA init |
| `EllipsoidFitter`: fit to known ellipsoid, verify center/radii | Yes | 5 tests: center (atol=1mm), axes (<5%), wrap_params structure, labels-only fit, PCA init |
| `PatellaFitter`: basic smoke test | Yes | 4 tests: label_patella_within_wrap_extents, smoke, wrap_params, known sphere geometry |
| `construct_cylinder_basis` tests | Yes (bonus) | 4 tests: identity for z-axis, orthonormal, z-column matches axis, custom reference_x_axis |

**Actual fitter accuracy (measured):**
- CylinderFitter: center 0.4mm, radius 0.1%, axis cos_sim 1.000
- EllipsoidFitter (with SDF + beta=0.5): center <0.01mm, dimension errors 1-2%

**Key insight:** EllipsoidFitter needs SDF values + distance loss (`beta > 0`) for accurate dimension recovery. With only margin-based classification loss (`beta=0`), the fitter finds a separating boundary but dimensions can be 5-16% off. The real pipeline provides SDF values, so the test now matches real usage.

### 2.2 Articular surface tests (`tests/test_articular_surfaces.py`) — DONE (17 tests)

| Planned | Done | Notes |
|---------|------|-------|
| `add_polar_coordinates_about_center()` | Yes | 6 tests: cardinal points, center offset, theta_offset, wrapping, default center, diagonal |
| `smooth_1d()` | Yes | 2 tests: constant interior, window=1 noop |
| `build_min_radial_envelope()` | Yes | 3 tests: min of two regions, single region, output shape |
| `mask_points_by_radial_envelope()` | Yes | 2 tests: basic mask, theta_offset |
| `trim_mesh_by_radial_envelope()` | Yes | 2 tests: trims outliers (~10k pt mesh, tol=3.3), preserves all when envelope large |
| `create_articular_surfaces()` smoke test | Yes | 1 test (marked `@pytest.mark.slow`) |
| `create_meniscus_articulating_surface()` smoke test | Yes | 1 test (marked `@pytest.mark.slow`) |

### 2.3 Mesh labeling tests (`tests/test_mesh_labeling.py`) — DONE (16 tests)

| Planned | Done | Notes |
|---------|------|-------|
| `classify_points()`: threshold behavior | Yes | 6 tests: basic, custom threshold, all inside, all outside, empty, dtype |
| `classify_near_surface()`: threshold behavior | Yes | 7 tests: basic, boundary, all near, all far, zero threshold, empty, dtype |
| Synthetic cylinder labeling pipeline | Yes | 3 tests using PCU SDF: inside/outside, near-surface, sign consistency |

---

## Things to know for future work

### pymskt PCU SDF single-point bug (needs issue filed)

`pymskt.mesh.Mesh.get_sdf_pts()` with the default PCU method returns wrong results for single-point queries. With 2+ points, results are correct.

| Query size | Surface point SDF | Expected |
|-----------|-------------------|----------|
| 1 point   | -11.84           | ~0       |
| 2+ points | 0.01             | ~0       |

**Action needed:** File issue on `github.com/gattia/pymskt` (`gh` CLI not available on this machine). Reproduction script:

```python
import numpy as np
import pyvista as pv
from pymskt.mesh import Mesh

cyl = pv.Cylinder(center=[0,0,0], direction=(0,0,1), radius=5.0, height=20.0, resolution=50)
mesh = Mesh(cyl.triangulate())

# Single point — WRONG
sdf_1 = mesh.get_sdf_pts(np.array([[5.0, 0.0, 0.0]]))  # Returns -11.84 (should be ~0)

# Two points — CORRECT
sdf_2 = mesh.get_sdf_pts(np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))  # Returns [0.01, -4.99]
```

### `trim_mesh_by_radial_envelope()` and mesh resolution

The function uses `adjacent_cells=True` in `extract_points()`, which includes all vertices of any cell touching the boundary. Boundary overshoot scales with cell size:

| Mesh resolution | n_points | Overshoot |
|----------------|----------|-----------|
| 10             | 82       | 1.63      |
| 20             | 362      | 1.11      |
| 50             | 2,402    | 0.46      |
| 100            | 9,802    | 0.26      |

Test uses resolution=100 (~10k points) with tolerance 3.3 (0.3 overshoot budget).

### `smooth_1d()` boundary effects

Uses `np.convolve(y, kernel, mode='same')` — boundary values are affected by zero-padding. Tests check interior only (`result[2:-2]`).

### `make autoformat` was run

`make autoformat` (isort + black) was run repo-wide to fix pre-existing formatting issues. `black` and `isort` were installed in the `comak` env from `requirements-dev.txt`.

### Warnings

- `@pytest.mark.slow` — register in `pyproject.toml` to suppress warning
- `torch.cross()` deprecation — Phase 4 cleanup item
