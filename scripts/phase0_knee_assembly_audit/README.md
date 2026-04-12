# Phase 0: COMAK Knee Assembly Model Audit

Phase 0 of the [knee assembly plan](../../.claude/plans/knee-assembly.md). Audits the Smith2019 COMAK model to resolve empirical unknowns before writing production code.

## Scripts

| Script | Purpose |
|--------|---------|
| `phase0_model_audit.py` | Enumerates all components, dumps property schemas, checks API constructors, extracts reference segment lengths |
| `spike_add_api.py` | Verifies property setter API for every COMAK component type (ligaments, springs, contacts, muscles, wraps, joints) |

Run from the repo root:

```bash
conda run -n comak python scripts/phase0_knee_assembly_audit/phase0_model_audit.py
conda run -n comak python scripts/phase0_knee_assembly_audit/spike_add_api.py
```

Results are written to `phase0_model_audit_results.txt` and `spike_add_api_results.txt` in this directory.

## Key Findings

Audited model: `full_body_healthy_knee.osim` (Smith2019)

### Component Counts

| Component | Count |
|-----------|-------|
| Bodies (COMAK) | 5 (`femur_distal_r`, `tibia_proximal_r`, `patella_r`, `meniscus_medial_r`, `meniscus_lateral_r`) |
| Joints (COMAK) | 6 (2 weld + 4 custom: `knee_r`, `pf_r`, 2 meniscus) |
| Ligaments (`Blankevoort1991Ligament`) | 91 |
| Springs (`SpringGeneralizedForce`) | 24 (6 TF + 6 PF + 6 med meniscus + 6 lat meniscus) |
| Contact forces (`Smith2018ArticularContactForce`) | 6 (TF, PF, + 4 meniscus) |
| Contact meshes (`Smith2018ContactMesh`) | 7 |
| Constraints | 0 |
| Markers on COMAK bodies | 0 |
| Spanning muscles | 4 (`recfem_r`, `vasint_r`, `vaslat_r`, `vasmed_r`) |
| Wrap surfaces on COMAK bodies | 4 (1 cylinder on `femur_distal_r`, 2 ellipsoids on `tibia_proximal_r`, 1 ellipsoid on `patella_r`) |
| Wrap surfaces on `femur_r` | 10 (7 ellipsoids + 3 cylinders, including `Gastroc_at_Condyles_r`) |
| Wrap surfaces on `tibia_r` | 5 (3 ellipsoids + 2 cylinders) |

### Simplifications (vs plan assumptions)

- **All spatial transforms are `LinearFunction`** — no `SimmSpline`. Every COMAK joint axis is `slope * coord + intercept`. This eliminates the need for complex spline serialization.
- **All ligament path points are standard `PathPoint`** — no `ConditionalPathPoint` or `MovingPathPoint`.
- **0 constraints, 0 markers on COMAK bodies** — can be dropped from the extraction/strip/add checklist.
- **All API constructors confirmed** — `addBody`, `addJoint`, `Blankevoort1991Ligament`, `SpringGeneralizedForce`, `Smith2018ContactMesh`, `Smith2018ArticularContactForce`, `Millard2012EquilibriumMuscle`.

### Dataclass Updates Needed (from property dumps)

- **`ComakLigament`**: add `transition_strain` (default 0.06) and `damping_coefficient` (default 0.003). Only 4 properties total beyond `path`: `linear_stiffness`, `transition_strain`, `damping_coefficient`, `slack_length`.
- **`ComakContactMesh`**: add `location`, `orientation`, `use_variable_thickness`, `mesh_back_file`, `min_thickness`, `max_thickness`, `scale_factors`. 10 settable properties total.
- **`ComakContactForce`**: add `elastic_foundation_formulation` (str) and `use_lumped_contact_model` (bool). 4 settable properties total.
- **`ComakSpring`**: complete as-is — only 3 settable properties: `stiffness`, `rest_length`, `viscosity` (plus `coordinate` name).
- **`ComakMuscle`**: 24 properties including sub-component curves. For the 4 spanning muscles, key properties are: `max_isometric_force`, `optimal_fiber_length`, `tendon_slack_length`, `pennation_angle_at_optimal`, `max_contraction_velocity`, plus path points and wrap references.

### Reference Segment Lengths (for scaling)

| Measurement | Value |
|-------------|-------|
| Femur (hip → knee) | 0.377 m |
| Tibia (knee → ankle) | 0.403 m |
| Total leg | 0.781 m |

### Weld Joint Offsets

- `femur_femur_distal_r`: parent offset `t=[-0.0056, -0.3742, -0.0012]` (nearly pure Y-translation — femur_distal_r sits at the distal end of femur_r). No orientation offset.
- `tibia_tibia_proximal_r`: parent offset `t=[0.006, 0, 0]` (tiny X offset only). No orientation offset. Note: parent frame is on `tibia_proximal_r`, child frame is `tibia_r` — i.e., `tibia_proximal_r` is parent, `tibia_r` is child.

### Spanning Muscle Details

All 4 spanning muscles have exactly 3 `PathPoint`s, all on main chain bodies (`pelvis`, `femur_r`) or `patella_r`. All wrap on cylinders attached to `femur_r` (not a COMAK body):
- `recfem_r`: pelvis → patella_r (2 pts), wraps on `KnExt_at_fem_r`
- `vasint_r`: femur_r (2 pts) → patella_r, wraps on `KnExt_vasint_at_fem_r`
- `vaslat_r`: femur_r (2 pts) → patella_r, wraps on `KnExt_at_fem_r`
- `vasmed_r`: femur_r (2 pts) → patella_r, wraps on `KnExt_at_fem_r`

### Body Masses (notable)

- `femur_distal_r`: 0.008166 kg (near-zero placeholder)
- `tibia_proximal_r`: 0.008166 kg (near-zero placeholder)
- `patella_r`: 0.398116 kg (non-zero — has real mass)
- `meniscus_medial_r` / `meniscus_lateral_r`: 0.1 kg each

### Left Knee Asymmetry

The left knee has `pf_l` and `knee_l` joints but NO weld joints (`femur_femur_distal_l`, `tibia_tibia_proximal_l` do not exist) and no intermediate bodies. This is a simpler structure than the right knee COMAK setup.

## Spike: Add-API Setter Verification

**Script:** `spike_add_api.py` | **Results:** `spike_add_api_results.txt`

Verifies that every component type needed by `add_comak_knee()` can be created programmatically, have its properties set, and pass `initSystem()`. This resolves the key unknown from Phase 0E (constructors confirmed, but setters untested).

### Results: All 10 tests pass

**Pattern A (direct setters) works for ALL component types.** No need for Pattern B (`updPropertyByName` + `PropertyHelper`). Summary:

| Test | Component | Setters | Path/Wrap | initSystem |
|------|-----------|---------|-----------|------------|
| 1 | `Blankevoort1991Ligament` | `set_linear_stiffness`, `set_slack_length`, etc. | `updGeometryPath()` → `appendNewPathPoint()` | OK |
| 2 | `SpringGeneralizedForce` | `set_stiffness`, `set_rest_length`, `set_viscosity`, `set_coordinate` | N/A | OK |
| 3 | `Smith2018ContactMesh` | `set_mesh_file`, `set_elastic_modulus`, `set_location(Vec3)`, etc. | `updSocket("frame").setConnecteePath(...)` | (skipped — no mesh file) |
| 4 | `Smith2018ArticularContactForce` | `set_min_proximity`, `set_elastic_foundation_formulation`, etc. | `updSocket("target_mesh")` / `updSocket("casting_mesh")` | (setter-only) |
| 5 | `Millard2012EquilibriumMuscle` | Constructor with args + `set_max_isometric_force`, etc. | `updGeometryPath()` → `appendNewPathPoint()` + `addPathWrap()` | OK |
| 6 | `WrapCylinder` / `WrapEllipsoid` | `set_radius`, `set_length`, `set_dimensions(Vec3)`, `set_translation(Vec3)`, `set_xyz_body_rotation(Vec3)`, `set_quadrant` | `body.addWrapObject()` | OK |
| 7 | `CustomJoint` + `SpatialTransform` | `axis.set_axis(Vec3)`, `axis.set_coordinates(0, name)`, `axis.set_function(LinearFunction)` | 7-arg constructor with SpatialTransform | OK |
| 8 | `WeldJoint` with offsets | 7-arg constructor with translation/orientation Vec3 | Creates `PhysicalOffsetFrame` automatically | OK |
| 9 | Full mini-model | All of the above combined | Body chain + weld + custom joint + lig + spring + muscle + wrap | OK |
| 10 | Extract `recfem_r` from Smith2019 | Read all 24 properties | 3 PathPoints, 1 wrap (`KnExt_at_fem_r`) | OK |

### Gotcha discovered: CustomJoint collinear axis rejection

OpenSim **segfaults** (not just an exception) if you create a `CustomJoint` with axes that don't all have unique coordinate assignments. Axes left at default (no coordinate) are treated as collinear and the constructor calls `abort()` inside the C++ layer. **All 6 axes (rotation1-3, translation1-3) must have explicit coordinates and functions**, even for DOFs that are effectively locked (use `LinearFunction(1.0, 0.0)` + a locked coordinate).

### Spanning muscle properties (recfem_r)

24 total properties. Key ones for round-trip: `max_isometric_force`, `optimal_fiber_length`, `tendon_slack_length`, `pennation_angle_at_optimal`, `max_contraction_velocity`. The remaining 19 include sub-component curves (`ActiveForceLengthCurve`, `ForceVelocityCurve`, `FiberForceLengthCurve`, `TendonForceLengthCurve`) and dynamics params (`fiber_damping`, `activation_time_constant`, etc.) — most have standard defaults but must be captured for exact round-trip.
