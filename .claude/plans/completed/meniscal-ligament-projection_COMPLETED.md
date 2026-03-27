# Meniscal Ligament Tibia Attachment Projection

## Status: DONE

## Problem

Meniscal ligaments (coronary, anterior/posterior horn) connect the meniscus to the tibia plateau and should be roughly vertical. The current pipeline transfers both attachment points from reference to subject using NSM interpolation (`interp_ref_to_subject_to_osim()`), but the femur+meniscus and tibia are modeled by **separate NSMs**. Small differences in how these models deform cause the two endpoints to diverge laterally, producing angled ligaments instead of vertical ones.

**Consequences of angled meniscal ligaments:**
- Slack must be taken up before the ligament can engage and resist translation
- Longer effective ligament length → more meniscal mobility in med/lat directions
- Excess meniscal translation → excess joint laxity

## Proposed Solution

A post-interpolation correction step: for each meniscal ligament, keep the meniscus-side attachment as-is and re-derive the tibia-side attachment by casting a short ray in the distal direction (-Y in OpenSim space) from the meniscus point onto the tibia mesh surface.

- **Ray hit** → use intersection as new tibia attachment (near-vertical ligament)
- **No ray hit** → fallback to nearest point on tibia surface (handles extrusion cases); log a warning

## Scope

This library provides building-block functions — the main orchestration pipeline lives outside `nsosim`. So we:
1. **Create the projection function** as a new module in `nsosim`
2. **Export it** from the package
3. The user's pipeline script calls it between `interp_ref_to_subject_to_osim()` and `update_osim_model()`

We do NOT need to modify any existing orchestration within `nsosim`.

## Affected Ligaments (14 total, from Smith2019 COMAK model)

All have P1 on `tibia_proximal_r` and P2 on `meniscus_{medial,lateral}_r`:

| Ligament | Type | Side |
|----------|------|------|
| `meniscus_lateral_COR1/2/3` | Coronary | Lateral |
| `meniscus_lateral_AHORN1/2` | Anterior horn | Lateral |
| `meniscus_lateral_PHORN1/2` | Posterior horn | Lateral |
| `meniscus_medial_COR1/2/3` | Coronary | Medial |
| `meniscus_medial_AHORN1/2` | Anterior horn | Medial |
| `meniscus_medial_PHORN1/2` | Posterior horn | Medial |

**Excluded:** `meniscus_TRANSVLIG1` — connects lateral meniscus to medial meniscus (no tibia point).

## Key Design Decisions

### Coordinate system
- All work happens in OpenSim meters space, after `interp_ref_to_subject_to_osim()` has produced `xyz_mesh_updated` values
- Both `tibia_proximal_r` and `meniscus_*_r` frames have zero offset from each other and same rotation convention, so body-local coordinates are in essentially the same space
- "Distal" = -Y direction in OpenSim space

### Ray-casting approach
- Use pyvista's `PolyData.ray_trace(origin, end_point)` — pymskt Mesh inherits from `pv.PolyData` so this is directly available
- Short max ray length (default 15mm / 0.015m) to avoid spurious hits
- Nearest-neighbor fallback via `PolyData.find_closest_point()`

### Module location
- New file: `nsosim/meniscal_ligaments.py` — doesn't fit in `articular_surfaces.py` (that's about extracting contact surfaces) or `osim_utils.py` (that's OpenSim XML manipulation). This is a geometric correction on interpolated attachment points.

### Ligament identification
- Match by ligament name prefix: `meniscus_medial_` or `meniscus_lateral_`
- Exclude names containing `TRANSVLIG`
- Identify tibia vs meniscus point by `parent_frame` field in the point dict

## Implementation Steps

### Step 1: Create `nsosim/meniscal_ligaments.py`

Single public function:

```python
def project_meniscal_attachments_to_tibia(
    dict_lig_mus_attach,   # standard ligament attachment dict
    tibia_mesh,            # pymskt Mesh / pyvista PolyData in OpenSim space (m)
    ray_direction=None,    # default [0, -1, 0] (-Y = distal)
    max_ray_length=0.015,  # 15mm in meters
    xyz_key="xyz_mesh_updated",
) -> dict:
    """Modifies dict_lig_mus_attach in-place. Returns summary dict."""
```

Internal helpers:
- `_is_meniscal_tibia_ligament(name)` — prefix check, exclude TRANSVLIG
- `_identify_tibia_meniscus_points(points)` — find which point index is tibia vs meniscus by parent_frame

Algorithm per ligament:
1. Get meniscus point coords from `points[meniscus_idx][xyz_key]`
2. `ray_end = meniscus_point + ray_direction * max_ray_length`
3. `intersection_points, _ = tibia_mesh.ray_trace(meniscus_point, ray_end)`
4. If hits → use first intersection as new tibia point
5. If no hits → `tibia_mesh.find_closest_point(meniscus_point)` as fallback
6. Warn if meniscus point Y < tibia closest point Y (below surface = possible extrusion)
7. Update `points[tibia_idx][xyz_key]` in-place

Return dict: `{ligament_name: {'method': 'ray'|'nearest', 'distance': float}}` for diagnostics.

### Step 2: Export from package

Add to `nsosim/__init__.py` (or confirm it auto-imports).

### Step 3: Tests

- Test with a simple flat tibia mesh (plane at Y=0) and meniscus points above it — verify ray hits at correct XZ with Y=0
- Test nearest-neighbor fallback when ray misses (meniscus point laterally outside tibia extent)
- Test TRANSVLIG exclusion
- Test that only meniscal ligaments are modified (other ligaments untouched)

### Step 4: Update CLAUDE.md

Add the new module to the architecture table and document where it fits in the pipeline workflow.

## Usage in Pipeline (outside nsosim)

```python
# After interpolation, before update_osim_model:
from nsosim.meniscal_ligaments import project_meniscal_attachments_to_tibia

projection_results = project_meniscal_attachments_to_tibia(
    dict_lig_mus_attach=dict_lig_mus_attach,
    tibia_mesh=tib_mesh_osim,
    # ray_direction defaults to [0, -1, 0]
    # max_ray_length defaults to 0.015 (15mm)
)

# Then proceed as normal
update_osim_model(
    model=osim_model,
    dict_lig_mus_attach=dict_lig_mus_attach,  # now has corrected tibia points
    ...
)
```

---

## Completion Summary

**Completed:** 2026-03-25
**Commit:** `5055669` — "Add meniscal ligament tibia attachment projection"

### What Was Delivered

1. **`nsosim/meniscal_ligaments.py`** — single public function `project_meniscal_attachments_to_tibia()` plus two internal helpers (`_is_meniscal_tibia_ligament`, `_identify_tibia_meniscus_points`)
2. **`tests/test_meniscal_ligaments.py`** — 16 tests across 3 test classes
3. **`nsosim/__init__.py`** — module exported
4. **`CLAUDE.md`** — architecture table updated, Stage 4b documented
5. **Pipeline integration** — done separately in the external pipeline repo

### Deviations from Plan

- **Ray direction normalization + validation**: The plan didn't mention it, but the implementation normalizes the `ray_direction` vector and raises `ValueError` on zero-length input. Prevents subtle bugs from non-unit direction vectors.
- **`test_custom_ray_direction` split into two tests**: The plan had a single custom direction test. During review, we split it into `test_custom_ray_direction_miss` (horizontal ray on flat plane) and `test_custom_ray_direction_hit` (ray hitting a vertical wall) to confirm the direction is actually used for intersection, not just that it misses.
- **Extra tests beyond plan**: Plan called for 4 test scenarios. Implementation has 16 tests covering: helper functions directly, multiple ligaments, distance reporting, zero direction error, both hit and miss with custom direction.

### Gotchas / Notes

- **In-place mutation**: `project_meniscal_attachments_to_tibia()` modifies `dict_lig_mus_attach` in-place (writes new numpy arrays to `xyz_key`). If you need the original values, copy before calling.
- **`find_closest_point` returns an index**: The nearest-neighbor fallback uses `tibia_mesh.find_closest_point()` which returns a point index, not coordinates. The implementation correctly indexes into `tibia_mesh.points[idx]` to get the actual position.
- **Extrusion warning uses Y-axis only**: The "meniscus below tibia" warning compares raw Y values. This is correct for OpenSim space where Y=superior, but would need adjustment if someone used a different coordinate convention.
- **Frame names are hardcoded**: `_TIBIA_FRAMES` and `_MENISCUS_FRAMES` match the Smith2019 COMAK model naming (`tibia_proximal_r`, `meniscus_medial_r`, `meniscus_lateral_r`). A different OpenSim model with different frame names would need these updated.
