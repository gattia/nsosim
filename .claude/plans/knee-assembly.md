# COMAK Knee Assembly: Strip, Add, and Round-Trip Validation

## Quick Start for Implementer

**Environment:** All code must run in the `comak` conda environment. OpenSim (JAM/COMAK fork) is not pip-installable — it's a source build pinned to Python 3.9 + numpy 2.0.2.

```bash
conda run -n comak python your_script.py          # run scripts
conda run -n comak python -m pytest tests/ -v      # run tests
conda run -n comak make lint                        # check formatting
```

**Reference model:**
```
/dataNAS/people/aagatti/projects/comak_gait_simulation/COMAK_SIMULATION_REQUIREMENTS/data/reference_data/comak_models/current/full_body_healthy_knee.osim
```

**Phase 0 results — READ FIRST:** `scripts/phase0_knee_assembly_audit/phase0_model_audit_results.txt` contains the complete component enumeration, property dumps for every COMAK class, and reference segment lengths. The `README.md` in that folder has the interpreted findings. These results are the ground truth for what the dataclasses must capture and what the strip/add must handle.

**Implementation strategy:**
- Start with **hardcoded component names** for the Smith2019 round-trip (the names are known from Phase 0). Dynamic class-based discovery is a refinement — get it working with explicit names first, generalize later.
- Phase 0E confirmed constructors exist but did NOT test setting individual properties. The first implementation task is verifying the setter API — e.g., is it `lig.set_linear_stiffness(val)` or `lig.setPropertyByName('linear_stiffness', val)`? Run a quick spike against the model before writing production code.
- Build and test **one component type at a time** (Tier 1 leaf round-trips), not all at once. See the incremental development strategy section.

**OpenSim API gotchas discovered during Phase 0:**
- `axis.get_coordinates()` requires an index argument — use `axis.getCoordinateNamesInArray().getSize()` to check count, then `axis.get_coordinates(0)` to get name
- `Blankevoort1991Ligament` path access: use `lig.getGeometryPath()`, NOT `lig.get_GeometryPath()`
- Path points: `pp.get(j)` returns `AbstractPathPoint` — must downcast via `osim.PathPoint.safeDownCast(pt)` to access `get_location()`
- Weld joint frames: `joint.get_frames(0)` only works if there's a parent offset frame; use `joint.getParentFrame()` / `joint.getChildFrame()` and check for `PhysicalOffsetFrame` via `getConcreteClassName()`
- All removals use reverse-index order, then one `finalizeConnections()` at the end — no `initSystem()` between removals

**OpenSim API gotchas discovered during spike (`spike_add_api.py`):**
- **CustomJoint segfaults on collinear axes (CRITICAL):** If a `CustomJoint` is created without assigning coordinates to all 6 axes (rotation1-3, translation1-3), OpenSim detects "collinear axes" and calls `abort()` inside the C++ layer. This is a **segfault, not a catchable Python exception** — no try/except can recover from it, and it kills the process silently. Production code must **always configure all 6 axes** with explicit coordinate names and functions, even for DOFs that are effectively locked (use `LinearFunction(1.0, 0.0)` + a locked coordinate with `default_value=0`).
- Pattern A (direct setters like `set_linear_stiffness()`) works for **all** component types — no need for the `updPropertyByName` + `PropertyHelper` fallback
- Ligament/muscle path access: `updGeometryPath()` is the correct method (returns mutable `GeometryPath`); `getGeometryPath()` also works for read-only access
- Muscle wrap attachment: `geometry_path.addPathWrap(wrap_object)` works — the wrap object must already be attached to a body via `body.addWrapObject()` before calling this
- WeldJoint 7-arg constructor auto-creates `PhysicalOffsetFrame` instances (named `{body}_offset`) — no manual frame creation needed
- Contact mesh socket: use `mesh.updSocket("frame").setConnecteePath("/bodyset/body_name")` — string path, not object reference
- Contact force sockets: `force.updSocket("target_mesh")` and `force.updSocket("casting_mesh")` — set via `setConnecteePath()`

---

## Context

`nsosim` currently assumes the COMAK knee already exists in the target model — it only *updates* existing components (geometry, wrap surfaces, ligaments, contacts) via `comak_osim_update.py` and `osim_utils.py`. It cannot add a COMAK knee to a model that doesn't have one, nor can it remove one.

The combined COMAK + predictive simulation plan (Track 2) requires placing a COMAK knee into *any* OpenSim model (e.g., Rajagopal), not just Smith2019. This means nsosim needs three new capabilities:

1. **Extract** — Read all COMAK knee components from a reference model into a structured config
2. **Strip** — Remove all COMAK knee components from a model (leaving a generic knee)
3. **Add** — Insert COMAK knee components into any model that has a femur and tibia

The round-trip test (strip Smith2019 → re-add → compare) validates the machinery before generalizing to other models.

**Scope**: Smith2019 round-trip only. Model-agnostic placement (Rajagopal) and bilateral support are future phases. Design around Rajagopal and the new Ulrich model(s) as primary targets — don't over-generalize.

**Source plan**: `/dataNAS/people/aagatti/projects/comak_gait_simulation/NOTES/combined_comak_predsim_plan.md` (Track 2)

**Prior art**: Stripping scripts at `/dataNAS/people/aagatti/projects/gait_opensim_jam_2023/stanford_jam_gait_2023/AddBiomechanics/` — `replace_comak_knee_with_generic.py`, v1/v2 notebooks. The v2 notebook is the production version (confirmed by `Process_Pipeline/README.md`). It strips ALL muscles (intentional — the output was a throwaway skeleton for AddBiomechanics scaling, not a functioning simulation model). For nsosim's purposes, only the 4 spanning muscles should be removed.

**Prior art (scaling)**: `Process_Pipeline/scaleModel.py` in the same AddBiomechanics directory. Supports three scaling modes: "AB" (full AddBiomechanics 3D factors), "LA" (long-axis only), "WA" (weighted average of femur+tibia long-axis). Applies scale factors to COMAK bodies, weld joint offsets, ligament attachment points, and patella joint defaults. See "Scaling Strategy" section below for how this informs the plan.

---

## Prior Art: Confirmed OpenSim API Patterns

Deep-dive into the prior stripping scripts confirms the following API patterns are **proven and executed**:

### Removal API
- `model.getJointSet().remove(idx)` — works
- `model.getBodySet().remove(idx)` — works
- `model.getForceSet().remove(idx)` — works
- `model.getContactGeometrySet().remove(idx)` — works

No `initSystem()` needed between removals. All removals happen first, then `finalizeConnections()` once at the end.

### Two removal patterns (both work)
**Pattern 1 (preferred):** Collect indices → reverse → remove:
```python
indices = [i for i in range(joints.getSize()) if joints.get(i).getName() in to_delete]
indices.reverse()
for idx in indices:
    joints.remove(idx)
```

**Pattern 2:** While-loop with conditional increment:
```python
idx = 0
while idx < joints.getSize():
    if joints.get(idx).getName() in to_delete:
        joints.remove(idx)  # don't increment — next element slides in
    else:
        idx += 1
```

### Offset frame cloning (critical)
Must clone the weld joint offset frame BEFORE removing the joint:
```python
fem_fem = joints.get(joint_idx)     # femur_femur_distal_r weld joint
femur_offset = fem_fem.get_frames(0).clone()  # clone parent offset frame
# ... then remove the joint ...
# ... later, attach cloned frame to knee_r:
knee_r.getSocket('parent_frame').setConnecteePath(femur_offset.getName())
knee_r.append_frames(femur_offset)
```

### Finalization
Use `finalizeConnections()` after removals (not `finalizeFromProperties()`). No `initSystem()` required before `printToXML()`.

### What has NO prior art (untested) — constructors confirmed in Phase 0E, but property setters need verification

Phase 0E confirmed all constructors work. The add API spike (Phase 1 first task) should verify this pattern:

```python
# Adding a body + weld joint (confirmed working in Phase 0E)
body = osim.Body("test_body", 1.0, osim.Vec3(0), osim.Inertia(1))
model.addBody(body)
joint = osim.WeldJoint("test_weld", model.getGround(), body)
model.addJoint(joint)

# Adding a ligament — PROPERTY SETTERS NEED VERIFICATION
# Try both patterns and see which works:
lig = osim.Blankevoort1991Ligament()
lig.setName("MCLd1")
# Pattern A: direct setter
lig.set_linear_stiffness(5000.0)
lig.set_slack_length(0.05)
lig.set_transition_strain(0.06)
lig.set_damping_coefficient(0.003)
# Pattern B: property-based (fallback if Pattern A fails)
# lig.setPropertyByName('linear_stiffness', osim.PropertyHelper.makeValueDouble(5000.0))

# Adding path points to a ligament
gp = lig.updGeometryPath()  # or lig.getGeometryPath()? Test both
gp.appendNewPathPoint("MCLd1-P1", model.getBodySet().get("femur_distal_r"), osim.Vec3(x, y, z))
gp.appendNewPathPoint("MCLd1-P2", model.getBodySet().get("tibia_proximal_r"), osim.Vec3(x, y, z))
model.addForce(lig)

# Adding a spring
spring = osim.SpringGeneralizedForce("knee_flex_r")  # or setName after?
spring.set_stiffness(1.0)
spring.set_rest_length(0.0)
spring.set_viscosity(0.0)
model.addForce(spring)

# Finalize
model.finalizeConnections()
model.initSystem()  # this is the real test
```

Write this spike as `scripts/phase0_knee_assembly_audit/spike_add_api.py` and run it before writing production code. The spike should add one of each component type to a minimal model and verify `initSystem()` succeeds.

- Selective muscle removal by name (prior art removed ALL muscles by class) — straightforward, filter by name in ForceSet.

---

## COMAK Knee Component Inventory

**Phase 0 audit completed.** Script: `scripts/phase0_knee_assembly_audit/phase0_model_audit.py`. Full results: `scripts/phase0_knee_assembly_audit/phase0_model_audit_results.txt`. See `scripts/phase0_knee_assembly_audit/README.md` for interpretation.

The COMAK knee in Smith2019 consists of (exact counts from Phase 0):

| Category | Count | Details |
|----------|-------|---------|
| **Bodies** | 5 | `femur_distal_r` (0.008kg), `tibia_proximal_r` (0.008kg), `patella_r` (0.398kg), `meniscus_medial_r` (0.1kg), `meniscus_lateral_r` (0.1kg) |
| **Joints** | 6 | 2 weld (`femur_femur_distal_r`, `tibia_tibia_proximal_r`), 4 custom (`knee_r`, `pf_r`, `meniscus_medial_r`, `meniscus_lateral_r`) |
| **Coordinates** | 24 | 6 TF + 6 PF + 6 medial meniscus + 6 lateral meniscus |
| **Ligaments** | 91 | `Blankevoort1991Ligament` — MCL(16), ACL(12), PCL(10), LCL(4), PT(6), PFL(13), pCAP(8), ITB(1), meniscus(15), transverse(1) |
| **Contact forces** | 6 | `tf_contact`, `pf_contact`, + 4 meniscus contacts (`femur_meniscus_{med,lat}_contact`, `tibia_meniscus_{med,lat}_contact`) |
| **Contact meshes** | 7 | `Smith2018ContactMesh` — femur/tibia/patella cartilage + 4 meniscus surfaces (med/lat superior/inferior) |
| **Springs** | 24 | `SpringGeneralizedForce` — 6 TF + 6 PF + 6 med meniscus + 6 lat meniscus |
| **Wrap surfaces (COMAK bodies)** | 4 | 1 cylinder (`Capsule_r` on `femur_distal_r`), 2 ellipsoids on `tibia_proximal_r`, 1 ellipsoid (`PatTen_r` on `patella_r`) |
| **Constraints** | 0 | None |
| **Markers on COMAK bodies** | 0 | None |
| **Spanning muscles** | 4 | `recfem_r`, `vasint_r`, `vaslat_r`, `vasmed_r` — all 3 PathPoints, wrap on `femur_r` cylinders |

### Phase 0 simplifications

- **All spatial transforms use `LinearFunction`** — no `SimmSpline`. Every COMAK joint axis is `slope * coord + intercept`. Eliminates complex spline serialization.
- **All ligament path points are standard `PathPoint`** — no `ConditionalPathPoint` or `MovingPathPoint`.
- **0 constraints, 0 markers on COMAK bodies** — dropped from checklist.
- **All API constructors confirmed** (addBody, addJoint, Blankevoort1991Ligament, SpringGeneralizedForce, Smith2018ContactMesh, Smith2018ArticularContactForce, Millard2012EquilibriumMuscle).

### Patella mass

`patella_r` has mass=0.398 kg in the reference model. This is non-trivial (unlike `femur_distal_r`/`tibia_proximal_r` which are ~0.008 kg placeholders). Confirmed: fine as-is — extract/add preserves the model's value.

### Left knee asymmetry

The left knee (`pf_l`, `knee_l`) has NO weld joints and no intermediate bodies (`femur_distal_l`, `tibia_proximal_l` do not exist). It's a simpler structure than the right COMAK knee. Bilateral support will need to account for this asymmetry.

---

## Cross-Joint Spanning Forces

Not all forces are cleanly contained within the COMAK knee. Some span between COMAK-specific bodies and the main body chain. These require special handling during strip/add.

### Forces that attach to COMAK bodies from outside

| Force | Type | Proximal | Distal | Issue |
|-------|------|----------|--------|-------|
| recfem_r | Muscle | pelvis | patella_r | Distal attachment on COMAK body |
| vasint_r | Muscle | femur_r (via femur_distal_r) | patella_r | Path through femur_distal_r, inserts on patella_r |
| vaslat_r | Muscle | femur_r (via femur_distal_r) | patella_r | Same as vasint |
| vasmed_r | Muscle | femur_r (via femur_distal_r) | patella_r | Same as vasint |

These 4 muscles are functionally part of the knee extensor mechanism. During **strip**, removing `patella_r` orphans their distal attachments. During **add**, they need to be reconnected.

**Recommended approach:** Include these 4 muscles in `ComakKneeConfig` as a `spanning_muscles` field. During strip, remove them along with the knee. During add, re-add them with correct path points. This keeps the knee as a self-contained unit. The alternative (leaving them in place with placeholder attachments during strip) is fragile and breaks `initSystem()`.

**Existing infrastructure for spanning muscles:** The NSM interpolation pipeline already handles patella attachment points. `interpolate_bone_ligaments()` in `model_building.py:1046-1058` interpolates attachment points onto the subject-specific patella via NSM, then centers them by subtracting `mean_patella`. The same pipeline used for ligaments works for spanning muscle path points. Slack length renormalization is also already implemented: muscles use `scale_factor = new_length / ref_length` to scale optimal fiber length and tendon slack length (`osim_utils.py:879-898`); ligaments use `setSlackLengthFromReferenceStrain(reference_strain, state)`. So for the model-agnostic add phase, attaching spanning muscles to the new patella follows the same pattern as attaching ligaments — no new machinery needed for the attachment or slack length calculation.

### Forces that bypass COMAK bodies entirely

| Force | Type | Proximal | Distal | Notes |
|-------|------|----------|--------|-------|
| ITB1 | Blankevoort1991Ligament | pelvis → femur_r | tibia_r | Defined in attachment JSON but all points on main chain bodies |
| bflh_r | Muscle | pelvis | tibia_r | Hamstring, crosses knee but no COMAK body contact |
| bfsh_r | Muscle | femur_r | tibia_r | Hamstring short head |
| gaslat_r | Muscle | femur_r | calcn_r | Gastrocnemius, uses wrap on femur_r |
| gasmed_r | Muscle | femur_r | calcn_r | Same |
| semimem_r | Muscle | pelvis | tibia_r | Semimembranosus |
| semiten_r | Muscle | pelvis | tibia_r | Semitendinosus |

These muscles are NOT part of the COMAK knee — they stay in the model during strip/add. However:
- **ITB1** is a `Blankevoort1991Ligament` whose path points are all on main chain bodies (pelvis, femur_r, tibia_r). It IS conceptually part of the COMAK knee package — it should be extracted, removed, and re-added like all other COMAK ligaments. It just happens to not touch any COMAK-specific bodies.
- **Gastrocnemius** muscles use a wrap surface (`Gastroc_at_Condyles_r`) attached to `femur_r`. This wrap surface must be preserved during strip (it's on a main chain body, not a COMAK body).

### Impact on Phase 1 (Data Classes)

Add to `ComakKneeConfig`:
```python
spanning_muscles: list[ComakMuscle]  # recfem_r, vasint_r, vaslat_r, vasmed_r
```

New data class needed:
```python
@dataclass
class ComakMuscle:
    name: str
    max_isometric_force: float
    optimal_fiber_length: float
    tendon_slack_length: float
    pennation_angle: float
    path_points: list[dict]        # [{name, body, location}]
    wrap_objects: list[str]         # wrap surface references
```

### Impact on Phase 2 (Extract)

- Extract the 4 spanning muscles from the model's ForceSet (filter by name, not class)
- Store in `spanning_muscles` field

### Impact on Phase 3 (Strip)

- Remove the 4 spanning muscles along with COMAK ligaments
- Remove ITB1 along with other COMAK ligaments (it's part of the COMAK package even though its points are on main chain bodies)
- Do NOT remove wrap surfaces on `femur_r` (gastrocnemius wraps stay — they belong to non-COMAK muscles)
- **Filtering strategy:** Removing all `Blankevoort1991Ligament` instances IS correct — ITB1 is included. But non-COMAK muscles (hamstrings, gastroc, etc.) that happen to use `Millard2012EquilibriumMuscle` must NOT be removed. Filter by: all `Blankevoort1991Ligament` + the 4 named spanning muscles.

### Impact on Phase 4 (Add)

- Re-add the 4 spanning muscles with correct path points and wrap references
- Verify wrap surfaces they reference (on `femur_r`, `femur_distal_r`) exist in the target model

---

## Consolidated Checklist: Everything Strip/Add Must Handle

| # | Item | Strip | Add | Notes |
|---|------|-------|-----|-------|
| 1 | 5 COMAK bodies | Remove | Re-add | femur_distal_r, tibia_proximal_r, patella_r, 2 menisci |
| 2 | 6 COMAK joints | Remove | Re-add | 2 weld, knee_r, pf_r, 2 meniscus |
| 3 | 24 COMAK coordinates | Remove (with joints) | Re-add (with joints) | 6 TF + 6 PF + 12 meniscus |
| 4 | ~90 COMAK ligaments | Remove | Re-add | All Blankevoort1991Ligament instances including ITB1 |
| 5 | ~12 springs | Remove | Re-add | SpringGeneralizedForce per DOF |
| 6 | 2 contact forces | Remove | Re-add | tf_contact, pf_contact |
| 7 | 7+ contact meshes | Remove | Re-add | Smith2018ContactMesh instances |
| 8 | 7 wrap surfaces on COMAK bodies | Remove | Re-add | Wraps on femur_distal_r, tibia_proximal_r, patella_r |
| 9 | 4 spanning muscles | Remove | Re-add | recfem_r, vasint_r, vaslat_r, vasmed_r (attach to patella_r) |
| 10 | Attached geometry (STL/VTP) | Files orphaned | Copy to target Geometry/ | Visual meshes for COMAK bodies |
| ~~11~~ | ~~Markers on COMAK bodies~~ | — | — | ~~Phase 0: none found~~ |
| 12 | Weld joint offset frames | Clone before removing | Use cloned frame | femur_femur_distal_r defines femur→knee offset |
| 13 | Body mass | Lost during strip | Restored on add | Consider adding to parent during strip-only use |
| ~~14~~ | ~~Coordinate coupling/constraints~~ | — | — | ~~Phase 0: none found~~ |
| 15 | Wrap surfaces on main chain bodies | DO NOT remove | N/A | Gastroc wraps on femur_r belong to non-COMAK muscles |
| 16 | Wrap surfaces on COMAK bodies | Extract before body removal, then remove with body | Re-add to re-created bodies | Wraps on femur_distal_r, tibia_proximal_r, patella_r — must be extracted before the parent body is deleted |

---

## Other Currently Missed Items

### 1. Attached geometry (visual meshes)

The COMAK bodies have attached visual geometry (STL/VTP meshes for rendering). The plan mentions `attached_geometry` in `ComakBody` but doesn't discuss:
- **File management:** The geometry files (.stl/.vtp) live in a `Geometry/` subfolder relative to the .osim. During strip, these files become orphaned. During add to a new model, the files need to be copied to the target model's `Geometry/` folder.
- **Round-trip test:** Need to verify geometry file paths resolve correctly after add.

### 2. Coordinate default values and coupling

Some coordinates have **coupled constraints** (e.g., patellofemoral secondary kinematics may be driven by spline functions of knee flexion). The spatial transform extraction captures this, but the plan should explicitly note:
- Coordinate `is_free_to_satisfy_constraints` property
- Coordinate `prescribed` and `prescribed_function` properties
- Some coordinates may reference other coordinates by name — these references break if coordinate names change in a different model.

### 3. Body mass redistribution

When stripping, the mass of the 5 removed bodies needs to go somewhere (or be documented as lost). For a round-trip this doesn't matter (they come back), but for strip-only use cases (AB scaling), the model's total mass will be wrong. Consider:
- Adding removed body masses to the parent body (`femur_r` or `tibia_r`)
- Or documenting that mass is not conserved during strip

### ~~4. Marker handling~~ — RESOLVED (Phase 0)

~~The Smith2019 model likely has markers attached to COMAK-specific bodies.~~ Phase 0 confirmed: **no markers on any COMAK body.** Dropped from checklist.

### 5. Model-agnostic add: coordinate name conflicts

When adding a COMAK knee to Rajagopal, the target model already has a `knee_angle_r` coordinate (or similar). The add function needs to:
- Remove the existing simple knee joint
- Handle the naming difference (`knee_angle_r` in Rajagopal vs `knee_flex_r` in COMAK)
- Update any muscle path wrapping or force references that use the old coordinate name
- This is flagged as "future phase" but worth noting the specific conflict

### ~~6. Constraint set~~ — RESOLVED (Phase 0)

~~Check whether the Smith2019 model has any `CoordinateCouplerConstraint`.~~ Phase 0 confirmed: **0 constraints in the model.** Dropped from checklist.

---

## Architecture

New module: **`nsosim/knee_assembly.py`**

**Two public functions** (not three — extract is folded into strip):

```
strip_comak_knee(model, side='r') → (model, ComakKneeConfig)
add_comak_knee(model, knee_config, ...) → model
```

**Why not a separate `extract_comak_knee()`?** Strip must discover and remove components in dependency order. Extracting each component's data *as it's removed* guarantees the config captures exactly what was stripped — no mismatch possible. If you need extract-only (read without modifying), call strip on a `model.clone()`.

Plus a data class for the config:
```
ComakKneeConfig — stores all extracted component data, serializable to/from JSON
```

---

## Phase 1: Data Structures (`ComakKneeConfig`)

Define dataclasses to hold all COMAK knee components. These represent the "stored" form of the knee — extracted from a model and ready to be written back.

### Data classes needed:

```python
@dataclass
class ComakBody:
    name: str
    mass: float
    inertia: list[float]        # [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    mass_center: list[float]    # [x, y, z]
    attached_geometry: list[dict]  # [{name, mesh_file, ...}]

@dataclass
class ComakWeldJoint:
    name: str
    parent_body: str            # e.g., "femur_r"
    child_body: str             # e.g., "femur_distal_r"
    parent_offset_translation: list[float]
    parent_offset_orientation: list[float]
    child_offset_translation: list[float]
    child_offset_orientation: list[float]

@dataclass
class ComakCoordinate:
    name: str
    default_value: float
    range_min: float
    range_max: float
    locked: bool
    clamped: bool

@dataclass
class ComakCustomJoint:
    name: str
    parent_body: str
    child_body: str
    parent_offset_translation: list[float]
    parent_offset_orientation: list[float]
    child_offset_translation: list[float]
    child_offset_orientation: list[float]
    coordinates: list[ComakCoordinate]
    spatial_transform: dict     # rotation/translation function definitions

@dataclass
class ComakLigament:
    name: str
    linear_stiffness: float
    transition_strain: float        # Phase 0E: default 0.06
    damping_coefficient: float      # Phase 0E: default 0.003
    slack_length: float
    path_points: list[dict]         # [{name, body, location: [x,y,z]}]

@dataclass
class ComakSpring:
    name: str
    coordinate: str
    stiffness: float
    rest_length: float
    viscosity: float

@dataclass
class ComakContactMesh:
    name: str
    parent_frame: str               # socket_frame path
    mesh_file: str
    elastic_modulus: float           # default 1e6
    poissons_ratio: float            # default 0.5
    thickness: float                 # default 0.005
    location: list[float]            # Phase 0E: [x,y,z], default [0,0,0]
    orientation: list[float]         # Phase 0E: [x,y,z], default [0,0,0]
    use_variable_thickness: bool     # Phase 0E: default false
    mesh_back_file: str              # Phase 0E: for variable thickness
    min_thickness: float             # Phase 0E: default 0.001
    max_thickness: float             # Phase 0E: default 0.01
    scale_factors: list[float]       # Phase 0E: [x,y,z], default [1,1,1]

@dataclass
class ComakContactForce:
    name: str
    target_mesh: str
    casting_mesh: str
    min_proximity: float             # default 0
    max_proximity: float             # default 0.01
    elastic_foundation_formulation: str  # Phase 0E: default "linear"
    use_lumped_contact_model: bool       # Phase 0E: default true

@dataclass
class ComakKneeConfig:
    """Complete description of a COMAK knee, extracted from a reference model."""
    side: str                           # 'r' or 'l'
    bodies: list[ComakBody]
    weld_joints: list[ComakWeldJoint]
    custom_joints: list[ComakCustomJoint]
    ligaments: list[ComakLigament]
    springs: list[ComakSpring]
    contact_meshes: list[ComakContactMesh]
    contact_forces: list[ComakContactForce]
    wrap_surfaces: list[dict]           # reuse existing wrap_surface format, keyed by parent body name
    spanning_muscles: list[ComakMuscle] # recfem_r, vasint_r, vaslat_r, vasmed_r
    ref_femur_length: float             # hip center → knee center in reference model (meters)
    ref_tibia_length: float             # knee center → ankle center in reference model (meters)

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> 'ComakKneeConfig': ...
    def to_json(self, path: str): ...
    @classmethod
    def from_json(cls, path: str) -> 'ComakKneeConfig': ...
```

### Key design notes:
- Socket paths (e.g., `/bodyset/femur_distal_r`) are stored as body *names*, not full paths. The add function resolves paths relative to the target model.
- Spatial transform functions are stored as serializable dicts (function type + coefficients), not OpenSim objects.
- `wrap_surfaces` reuse the existing `wrap_surface.to_dict()` format from `wrap_surface_fitting/main.py`. Each entry includes the parent body name so they can be re-attached to the correct body during add.
- Wrap surfaces on COMAK bodies (femur_distal_r, tibia_proximal_r, patella_r) must be extracted BEFORE those bodies are deleted in strip. Wraps on main chain bodies (femur_r) are NOT included — they belong to non-COMAK muscles.

### Dataclass completeness warning:
Phase 0 dumped ALL properties — dataclasses above are now complete. `ComakLigament` includes `transition_strain` and `damping_coefficient`. `ComakSpring` is complete as-is (3 settable properties). `ComakContactMesh` and `ComakContactForce` include all Phase 0E fields.

---

## Phase 2: Extract (merged into strip — see Phase 3)

Extract is NOT a separate function. Instead, `strip_comak_knee()` extracts each component's data into `ComakKneeConfig` as it removes it. This guarantees the config captures exactly what was stripped. For extract-only (read without modifying), call strip on `model.clone()`.

### Extraction logic (used by strip internally):

Uses the OpenSim Python API (not XML parsing) to iterate over model component sets and extract COMAK-specific components. Identification is by **concrete class name** and **naming convention** (suffix `_r` or `_l`). Discovery is dynamic — no hardcoded component name lists. The strip function discovers COMAK components by class type and body parentage.

**Existing functions to reuse:**
- `get_osim_muscle_ligament_reference_lengths()` (`osim_utils.py:503`) — extracts force properties (partially covers ligaments)
- `extract_wrap_parameters_from_osim()` (`parameter_extraction.py:22`) — extracts wrap surface params

**Dynamic discovery — no hardcoded name lists:**

The strip function discovers COMAK components by class type, not by name. This makes it work on any model that has COMAK components, not just Smith2019:

- **Ligaments:** all `Blankevoort1991Ligament` in ForceSet → extract data → remove
- **Springs:** all `SpringGeneralizedForce` in ForceSet → extract data → remove
- **Contact forces:** all `Smith2018ArticularContactForce` in ForceSet → extract data → remove
- **Contact meshes:** all `Smith2018ContactMesh` in ContactGeometrySet → extract data → remove
- **Spanning muscles:** by name list (these are the only name-based filter, since they're `Millard2012EquilibriumMuscle` and we can't filter all muscles by class)
- **Bodies/Joints:** identified by tracing the joint tree from the user-specified `target_joint` — find all child bodies downstream that don't exist in a standard model

Each component is serialized to its dataclass *before* `remove(idx)` is called. The resulting `ComakKneeConfig` is returned alongside the stripped model.

**Reference names (Smith2019, for validation only — not used as filters):**
```python
# These are used in tests to verify discovery found the right components
EXPECTED_COMAK_BODIES = ['femur_distal_r', 'tibia_proximal_r', 'patella_r', 'meniscus_medial_r', 'meniscus_lateral_r']
EXPECTED_COMAK_JOINTS = ['femur_femur_distal_r', 'tibia_tibia_proximal_r', 'knee_r', 'pf_r', 'meniscus_medial_r', 'meniscus_lateral_r']
```

### Spatial transform extraction (simpler than expected):

**Phase 0 finding: all COMAK joint axes use `LinearFunction` only.** No `SimmSpline` or `Constant`. This greatly simplifies serialization — each axis is just slope + intercept.

```python
joint = osim.CustomJoint.safeDownCast(joint_obj)
st = joint.getSpatialTransform()
# For each of rotation1-3, translation1-3:
axis = st.get_rotation1()
func = axis.get_function()  # Always LinearFunction for COMAK joints
coord_name = axis.get_coordinates(0) if axis.getCoordinateNamesInArray().getSize() > 0 else None
# Note: use axis.getCoordinateNamesInArray().getSize(), NOT axis.get_coordinates().getSize()
```

Store as:
```python
{
    'rotation1': {'axis': [0,0,1], 'function': {'type': 'LinearFunction', 'slope': 1.0, 'intercept': 0.0}, 'coordinate': 'knee_flex_r'},
    'rotation2': {'axis': [1,0,0], 'function': {'type': 'LinearFunction', 'slope': 1.0, 'intercept': 0.0}, 'coordinate': 'knee_add_r'},
    ...
}
```

**Still support SimmSpline in deserialization** for future-proofing (other models may use splines), but the Smith2019 round-trip only needs LinearFunction.

---

## Phase 3: Strip + Extract (`strip_comak_knee`)

```python
def strip_comak_knee(model: osim.Model, side: str = 'r') -> tuple[osim.Model, ComakKneeConfig]:
    """Remove all COMAK knee components, returning stripped model + extracted config.
    
    Each component is serialized to ComakKneeConfig before removal.
    The config captures exactly what was stripped — no mismatch possible.
    """
```

### Implementation:

Follows the proven pattern from prior stripping scripts. Key lessons:

1. **Clone offset frame BEFORE deleting joints** — the weld joint `femur_femur_distal_r` defines the spatial offset between `femur_r` and `femur_distal_r`. This offset must be preserved as the new parent frame for the simplified `knee_r` joint.

2. **Remove in reverse index order** — prevents index shift bugs:
   ```python
   indices_to_remove.sort(reverse=True)
   for idx in indices_to_remove:
       component_set.remove(idx)
   ```

3. **Removal order**: joints → bodies → forces → contact geometry

### Detailed steps:

1. Clone offset frame from `femur_femur_distal_r` weld joint (BEFORE any removals)
2. Extract wrap surfaces from COMAK bodies (BEFORE body removal — once the body is deleted, its wraps are gone)
3. Remove COMAK joints (6) in reverse index order
4. Remove COMAK bodies (5) in reverse index order
5. Remove the 4 spanning muscles by name: `recfem_r`, `vasint_r`, `vaslat_r`, `vasmed_r`
6. Remove COMAK forces: all `Blankevoort1991Ligament`, all `SpringGeneralizedForce`, all `Smith2018ArticularContactForce`
7. Remove `Smith2018ContactMesh` entries from contact geometry set
8. Optionally re-create a simplified `knee_r` joint (hinge or locked) using the cloned offset frame
9. `model.finalizeConnections()`

**Note:** Wrap surfaces on COMAK bodies do NOT need explicit removal — they are destroyed when the parent body is removed. Wrap surfaces on main chain bodies (e.g., `Gastroc_at_Condyles_r` on `femur_r`) are untouched.

### Strip modes:

- **`mode='clean'`**: Remove everything including `knee_r`. Model has no knee joint.
- **`mode='hinge'`**: Keep `knee_r` as a flexion-only hinge with secondary DOFs locked.

Default: `mode='hinge'` (more useful for round-trip test and for predsim stripping).

---

## Phase 4: Add (`add_comak_knee`)

```python
def add_comak_knee(
    model: osim.Model,
    knee_config: ComakKneeConfig,
    target_joint: str = 'knee_r',       # existing joint to replace
    parent_femur_body: str = 'femur_r',
    parent_tibia_body: str = 'tibia_r',
    side: str = 'r',
    scale: float = 1.0,                 # isotropic scale factor (see Scaling Strategy)
) -> osim.Model:
    """Add COMAK knee components to a model that has femur and tibia bodies.
    
    The caller specifies which existing joint to replace and the parent/child
    bodies. The function removes the existing joint and inserts the full COMAK
    knee in its place.
    """
```

### Implementation:

Inverse of strip. Order of operations:

1. **Add bodies** (5) to the model's BodySet
2. **Add weld joints** — connect `femur_r → femur_distal_r` and `tibia_r → tibia_proximal_r` with correct offset transforms
3. **Replace/create `knee_r` joint** — CustomJoint with full spatial transform (all 24 TF DOFs)
4. **Add PF joint** — CustomJoint connecting `femur_distal_r → patella_r`
5. **Add meniscus joints** — CustomJoints connecting `tibia_proximal_r → meniscus_*_r`
6. **Add contact meshes** to ContactGeometrySet
7. **Add contact forces** to ForceSet
8. **Add ligaments** (~80+) to ForceSet with correct path points and stiffness
9. **Add springs** (~12) to ForceSet
10. **Add wrap surfaces** to appropriate bodies
11. **Add attached geometry** (visual meshes) to bodies
12. `model.finalizeConnections()`

### Incremental development strategy: remove-one, add-back-one

**Don't strip everything and then add back incrementally.** That puts the model in a deeply broken state where `initSystem()` failures could be strip artifacts, not add bugs. Instead, for each component type: remove just that type from the full model, add it back using stored config, verify `initSystem()` still passes. The model stays near-valid the whole time.

**Dependency constraints** determine what can be tested independently vs. what must be grouped:

**Two tiers** based on dependency structure:

**Tier 1 — Leaf components (remove-one, add-back-one):**

These have no dependents. Removing them leaves the model fully valid. Each test proves one add+remove helper pair in isolation against a near-complete model.

| Test | Remove from full model | Add back | Notes |
|------|----------------------|----------|-------|
| 1 | all ligaments | `_add_ligaments()` | All ~80 at once — same class, same code path. Error names which one failed. |
| 2 | all springs | `_add_springs()` | All ~12 at once |
| 3 | spanning muscles only | `_add_spanning_muscles()` | 4 muscles |
| 4 | wrap surfaces only | `_add_wrap_surfaces()` | Near-leaf — muscles reference wraps but model still loads without them |
| 5 | contact forces + meshes | `_add_contact_meshes()` then `_add_contact_forces()` | Coupled pair — forces reference meshes by socket path |

```python
def test_roundtrip_ligaments(full_model, config):
    """Remove ligaments from full model, add back, verify initSystem()."""
    model = full_model.clone()
    _remove_ligaments(model)           # remove just ligaments
    model.finalizeConnections()
    model.initSystem()                 # model without ligaments should still be valid
    _add_ligaments(model, config.ligaments)
    model.finalizeConnections()
    model.initSystem()                 # model with ligaments restored should match original
```

**Tier 2 — Structural core (bodies + joints):**

You CANNOT remove a body (`femur_distal_r`) without first removing every ligament, wrap, contact, and force that references it — OpenSim will segfault or throw socket errors. So testing the structural core in isolation requires removing almost everything first. This is effectively a full strip→add, but by the time you get here, all Tier 1 helpers are proven. If it fails, the bug is in body/joint logic, not ligaments/springs/etc.

| Test | Steps | What it proves |
|------|-------|----------------|
| 6 | full `strip_comak_knee()` → full `add_comak_knee()` | Structural core (bodies, joints, offsets) + integration of all helpers |

This is the final round-trip test. There's no way to test body/joint add in isolation because you can't strip bodies without first stripping their dependents.

**Development order:** Build and test Tier 1 helpers first (easiest, most isolated). Then build the structural helpers (bodies, joints) and test them via the full round-trip. Each helper needs a matching `_remove_*()` function — these are simple (collect indices by name/class, reverse, remove) and are reused by `strip_comak_knee()`.

`add_comak_knee()` is a thin wrapper that calls all helpers in order — easy to debug because each piece is independently verified.

### Existing functions to reuse:
- `create_contact_mesh()` and `add_contact_mesh_to_model()` (`osim_utils.py:159-260`)
- `create_articular_contact_force()` and `add_contact_force_to_model()` (`osim_utils.py:264-352`)

### New helper functions needed:
- `_add_bodies(model, bodies)` — create and add OpenSim Bodies
- `_add_weld_joints(model, weld_joints)` — create WeldJoints with offset frames
- `_add_custom_joints(model, custom_joints)` — create CustomJoints with SpatialTransform
- `_add_ligaments(model, ligaments)` — create Blankevoort1991Ligaments with path points
- `_add_springs(model, springs)` — create SpringGeneralizedForces
- `_add_wrap_surfaces(model, wrap_surfaces)` — create WrapCylinder or WrapEllipsoid on appropriate bodies
- `_add_contact_meshes(model, contact_meshes)` — create Smith2018ContactMesh entries
- `_add_contact_forces(model, contact_forces)` — create Smith2018ArticularContactForces
- `_add_spanning_muscles(model, muscles)` — create Millard2012EquilibriumMuscles with path points
- `_rebuild_spatial_transform(transform_dict)` — reconstruct SpatialTransform from serialized dict

---

## Scaling Strategy

The COMAK knee is extracted from Smith2019 at a specific size. When adding to a different model (Rajagopal, Ulrich), everything needs to be scaled isotropically. The femur, tibia, and patella should all scale by the same factor to preserve joint geometry.

### Approach: store reference lengths, compute scale automatically

Store reference segment lengths in `ComakKneeConfig` (values from Phase 0F):
```python
@dataclass
class ComakKneeConfig:
    # ... existing fields ...
    ref_femur_length: float   # hip center → knee center = 0.377 m in Smith2019
    ref_tibia_length: float   # knee center → ankle center = 0.403 m in Smith2019
```

`add_comak_knee()` computes the scale factor from the target model's segment lengths:
```python
def compute_knee_scale_factor(model, ref_femur_length, ref_tibia_length):
    """Compute isotropic scale = mean(femur_scale, tibia_scale)."""
    target_fem_length = ...  # from model joint positions
    target_tib_length = ...  # from model joint positions
    femur_scale = target_fem_length / ref_femur_length
    tibia_scale = target_tib_length / ref_tibia_length
    return np.mean([femur_scale, tibia_scale])
```

**Why average of individual scale factors, not hip-to-ankle ratio?** If the model is in a flexed pose, the hip-to-ankle distance shrinks even though the bones haven't changed size. Computing femur and tibia scale factors independently (from their own joint-to-joint distances) and then averaging is robust to pose artifacts. It also handles the case where femur and tibia scale slightly differently — the average gives a single isotropic factor that's the best compromise for the knee region.

### What scales and what doesn't

| Component | Scale? | Notes |
|-----------|--------|-------|
| Joint offset translations (weld, custom) | YES | |
| Ligament path point locations | YES | |
| Ligament slack lengths | YES | geometric lengths |
| Wrap surface translations | YES | |
| Wrap surface radii/lengths/dimensions | YES | |
| Spatial transform spline Y-values (translation axes) | YES | |
| Spatial transform spline Y-values (rotation axes) | NO | angles are scale-invariant |
| Contact mesh thickness | NO | material property |
| Spring stiffness | NO (initially) | debatable — moment arms change |
| Spring rest_length | YES | |
| Body mass | YES (s^3) | scales with volume |
| Body inertia | YES (s^5) | scales with mass * length^2 |
| STL/VTP geometry files | YES | must write scaled mesh files |

### Prior art

`scaleModel.py` in the AddBiomechanics pipeline implements a "WA" (weighted average) mode that uses `mean(femur_long_axis_scale, tibia_long_axis_scale)` — exactly this approach. It applies the factor to COMAK bodies, weld offsets, ligament points, and patella joint defaults. The main addition here is also scaling the spatial transform spline translation Y-values and the geometry mesh files.

### Round-trip phase does NOT need scaling

Scaling is only relevant when adding to a model with different segment lengths. The Smith2019 round-trip (Phase 5) uses `scale=1.0` by default.

---

### Weld Joints Are Kept (Not Consolidated)

The Smith2019 COMAK knee uses intermediate bodies connected by weld joints: `femur_r → [weld] → femur_distal_r → [knee_r] → tibia_proximal_r → [weld] → tibia_r`. These weld joints are preserved in the config and added to target models because they serve as **local coordinate frames** for the COMAK knee assembly:

- All ligament/wrap/contact positions are expressed relative to `femur_distal_r` or `tibia_proximal_r`
- The weld offset is the **single value** that positions the knee on the parent body
- Scaling only needs to touch the weld offset — internal geometry stays in its local frame
- Different models only change the weld offset, not every position in the config
- The NSM pipeline outputs naturally align with these intermediate body frames

Consolidation (baking weld offsets into every ligament point, wrap position, etc.) was considered and rejected because it would make every position in the config dependent on the absolute frame, complicating scaling and model-agnostic placement.

---

## Phase 5: Tests

New test file: **`tests/test_knee_assembly.py`**

### What the round-trip proves

The round-trip (extract → strip → re-add → compare to original) validates that extract captures everything and add reconstructs it correctly. It proves the machinery works on the known-good Smith2019 model. Once proven, the same `add_comak_knee()` can target other models (Rajagopal) in a future phase — but that's out of scope here.

### Per-phase tests (don't wait until Phase 5)

**Phase 3 tests** (strip + extract — now merged):
- `strip_comak_knee()` returns `(model, config)` where:
  - `model.initSystem()` succeeds
  - Model has correct body/joint/force counts (original minus COMAK components)
  - `knee_r` joint exists (if mode='hinge') and has correct parent/child
  - No `Blankevoort1991Ligament`, `SpringGeneralizedForce`, or `Smith2018ArticularContactForce` remain
  - No `Smith2018ContactMesh` remains
  - Spanning muscles (recfem, vasti) are gone; other muscles (gastroc, hamstrings) remain
- `config` has exact counts: 5 bodies, 6 joints, 91 ligaments, 24 springs, 6 contact forces, 7 contact meshes, 4 spanning muscles, 4 wrap surfaces on COMAK bodies
- Assert specific known values from Phase 0: `femur_distal_r` mass=0.008166, patella_r mass=0.398116, etc.
- Serialize to JSON and back, compare all fields: `config → to_json() → from_json() → compare`

**Phase 4 tests** (add — two tiers, matching the incremental strategy):

*Tier 1 — leaf component round-trips (each tested against the full model):*

| Test | Remove | Add back | What it proves |
|------|--------|----------|----------------|
| `test_roundtrip_ligaments` | all ligaments | `_add_ligaments()` | All 91 at once — same class, same helper |
| `test_roundtrip_springs` | all springs | `_add_springs()` | All 24 at once |
| `test_roundtrip_spanning_muscles` | 4 muscles | `_add_spanning_muscles()` | Muscle extraction + insertion |
| `test_roundtrip_wrap_surfaces` | wrap surfaces | `_add_wrap_surfaces()` | Wrap extraction + insertion |
| `test_roundtrip_contacts` | forces + meshes | `_add_contact_meshes()` + `_add_contact_forces()` | Coupled pair |

Each test calls `initSystem()` after removal AND after add-back. If the add-back fails, the bug is in that specific helper — not contaminated by a broken model state.

*Tier 2 — structural core (requires full strip because you can't remove a body without removing everything attached to it):*

| Test | What it does | What it proves |
|------|-------------|----------------|
| `test_roundtrip_full` | full `strip_comak_knee()` → full `add_comak_knee()` | Bodies, joints, offset frames, plus integration of all helpers |

By this point, all Tier 1 helpers are proven. If the full round-trip fails, the bug is in the body/joint add logic.

**Phase 5 test** (round-trip comparison):

1. Load the Smith2019 production model (`full_body_healthy_knee.osim`)
2. Strip + extract: `stripped, config = strip_comak_knee(model.clone())`
3. Re-add: `rebuilt = add_comak_knee(stripped, config)`
4. Compare `original` vs `rebuilt` component by component

### What to compare:

| Component | Comparison | Tolerance |
|-----------|-----------|-----------|
| Bodies | name, mass, inertia, mass_center | exact (6 decimal places) |
| Joints | name, type, parent/child bodies | exact |
| Joint offsets | translation, orientation | atol=1e-6 |
| Spatial transform functions | type, coefficients | atol=1e-6 |
| Coordinates | name, default value, range, locked | exact |
| Ligaments | name, stiffness, slack_length, path point locations | atol=1e-6 |
| Contact meshes | name, parent_frame, mesh_file, material props | exact |
| Contact forces | name, target/casting mesh paths | exact |
| Springs | coordinate, stiffness, rest_length | exact |
| Wrap surfaces | translation, rotation, dimensions | atol=1e-6 |

### Additional tests:

- **Extract + serialize + deserialize round-trip**: `config → to_json() → from_json() → compare`
- **Strip produces valid model**: stripped model can `initSystem()` without error
- **Strip + add produces valid model**: rebuilt model can `initSystem()` without error
- **Smoke test: update after add**: after adding knee, `update_osim_model()` from `comak_osim_update.py` still works

### Test data:

Production model:
```
/dataNAS/people/aagatti/projects/comak_gait_simulation/COMAK_SIMULATION_REQUIREMENTS/data/reference_data/comak_models/current/full_body_healthy_knee.osim
```

---

## Phase 0: Model Audit & API Verification — COMPLETE

**Script:** `scripts/phase0_knee_assembly_audit/phase0_model_audit.py`
**Results:** `scripts/phase0_knee_assembly_audit/phase0_model_audit_results.txt`
**Interpretation:** `scripts/phase0_knee_assembly_audit/README.md`

All unknowns resolved. Key findings incorporated into Component Inventory and dataclass definitions above. No surprises that require plan changes — the LinearFunction-only spatial transforms and standard-PathPoint-only ligaments are significant simplifications.

### 0A: Enumerate all components by class

Load the model, iterate every component set, dump class names and counts. This tells us exactly what we're dealing with — no guessing.

```python
import opensim as osim

model = osim.Model(path_to_smith2019)

# Joints: what types exist?
for i in range(model.getJointSet().getSize()):
    j = model.getJointSet().get(i)
    print(f"Joint: {j.getName()} -> {j.getConcreteClassName()}")

# Forces: what classes exist?
from collections import Counter
force_classes = Counter()
for i in range(model.getForceSet().getSize()):
    f = model.getForceSet().get(i)
    force_classes[f.getConcreteClassName()] += 1
    print(f"Force: {f.getName()} -> {f.getConcreteClassName()}")
print("\nForce class counts:", dict(force_classes))

# Contact geometry
for i in range(model.getContactGeometrySet().getSize()):
    c = model.getContactGeometrySet().get(i)
    print(f"Contact: {c.getName()} -> {c.getConcreteClassName()}")

# Constraints
cs = model.getConstraintSet()
for i in range(cs.getSize()):
    print(f"Constraint: {cs.get(i).getName()} -> {cs.get(i).getConcreteClassName()}")

# Markers on COMAK bodies
comak_bodies = {'femur_distal_r', 'tibia_proximal_r', 'patella_r', 'meniscus_medial_r', 'meniscus_lateral_r'}
ms = model.getMarkerSet()
for i in range(ms.getSize()):
    m = ms.get(i)
    parent = m.getParentFrameName()
    if any(cb in parent for cb in comak_bodies):
        print(f"Marker on COMAK body: {m.getName()} -> {parent}")
```

### 0B: Check spatial transform function types

```python
for i in range(model.getJointSet().getSize()):
    joint = model.getJointSet().get(i)
    if joint.getConcreteClassName() == 'CustomJoint':
        cj = osim.CustomJoint.safeDownCast(joint)
        st = cj.getSpatialTransform()
        for comp in ['rotation1','rotation2','rotation3','translation1','translation2','translation3']:
            axis = getattr(st, f'get_{comp}')()
            func = axis.get_function()
            print(f"{joint.getName()}.{comp}: {func.getConcreteClassName()}")
```

### 0C: Check ligament path point types

```python
for i in range(model.getForceSet().getSize()):
    f = model.getForceSet().get(i)
    if f.getConcreteClassName() == 'Blankevoort1991Ligament':
        lig = osim.Blankevoort1991Ligament.safeDownCast(f)
        pp = lig.get_GeometryPath().getPathPointSet()
        for j in range(pp.getSize()):
            pt = pp.get(j)
            if pt.getConcreteClassName() != 'PathPoint':
                print(f"NON-STANDARD: {f.getName()} point {j}: {pt.getConcreteClassName()}")
```

### 0D: Check wrap surfaces on COMAK bodies

```python
for body_name in comak_bodies:
    body = model.getBodySet().get(body_name)
    ws = body.getPropertyByName('WrapObjectSet')
    # Or iterate:
    n_wraps = body.getWrapObjectSet().getSize() if hasattr(body, 'getWrapObjectSet') else 0
    print(f"{body_name}: {n_wraps} wrap objects")
```

### 0E: Verify add-back API exists

```python
# Minimal spike: can we add a body + joint from scratch?
test_model = osim.Model()
body = osim.Body("test_body", 1.0, osim.Vec3(0), osim.Inertia(1))
test_model.addBody(body)
joint = osim.WeldJoint("test_weld", test_model.getGround(), body)
test_model.addJoint(joint)
test_model.finalizeConnections()
test_model.initSystem()
print("addBody + addJoint: CONFIRMED")

# Can we create a Blankevoort1991Ligament?
lig = osim.Blankevoort1991Ligament()
lig.setName("test_lig")
print(f"Blankevoort1991Ligament constructor: CONFIRMED")
print(f"  Methods: {[m for m in dir(lig) if 'slack' in m.lower() or 'stiff' in m.lower() or 'path' in m.lower()]}")

# IMPORTANT: Also dump ALL property names so dataclasses can capture every field
print(f"  All properties:")
for i in range(lig.getNumProperties()):
    prop = lig.getPropertyByIndex(i)
    print(f"    {prop.getName()} = {prop.toString()}")

# Can we create a SpringGeneralizedForce?
spring = osim.SpringGeneralizedForce()
print(f"SpringGeneralizedForce constructor: CONFIRMED")
print(f"  Methods: {[m for m in dir(spring) if 'stiff' in m.lower() or 'coord' in m.lower()]}")

# Dump all SpringGeneralizedForce properties too
print(f"  All properties:")
for i in range(spring.getNumProperties()):
    prop = spring.getPropertyByIndex(i)
    print(f"    {prop.getName()} = {prop.toString()}")
```

**Critical:** The property dump is essential. `ComakLigament` and `ComakSpring` dataclasses must capture every property — a missed property reverts to its default in the round-trip. Update the dataclasses after running this.

### 0F: Extract reference segment lengths (for scaling)

```python
# Get hip→knee and knee→ankle distances for ComakKneeConfig.ref_femur_length / ref_tibia_length
state = model.initSystem()
hip_joint = model.getJointSet().get('hip_r')
knee_joint = model.getJointSet().get('knee_r')
ankle_joint = model.getJointSet().get('ankle_r')
# Extract joint center positions in ground frame, compute distances
# Store in ComakKneeConfig for automatic scaling when adding to other models
```

### Phase 0 deliverable

A text dump answering:
1. Exact component counts by class (bodies, joints, forces, contacts, constraints, markers)
2. Spatial transform function types used (SimmSpline? LinearFunction? Others?)
3. Ligament path point types (all PathPoint? Any ConditionalPathPoint/MovingPathPoint?)
4. Wrap surfaces per COMAK body (names and types)
5. Whether `addBody()`, `addJoint()`, `Blankevoort1991Ligament()`, `SpringGeneralizedForce()` constructors exist
6. Any `CoordinateLimitForce` or `CoordinateCouplerConstraint` present
7. Complete property list for `Blankevoort1991Ligament` and `SpringGeneralizedForce` (for dataclass completeness)
8. Reference femur/tibia segment lengths (hip→knee, knee→ankle) for scaling config

If any answer is surprising (e.g., a function type we didn't expect, or `addBody()` doesn't exist), the plan must be updated before proceeding.

---

## Implementation Order

0. **Phase 0**: Model audit & API verification — **COMPLETE** (see `scripts/phase0_knee_assembly_audit/`). Includes `spike_add_api.py` — all 10 setter tests pass, Pattern A (direct setters) confirmed for all component types.
1. **Phase 1**: Data classes + serialization — **COMPLETE** (see implementation notes below)
2. **Phase 2**: ~~`extract_comak_knee()`~~ — **merged into Phase 3** (extract-as-you-strip)
3. **Phase 3**: `strip_comak_knee()` — **COMPLETE** (see implementation notes below)
4. **Phase 4**: `add_comak_knee()` — **COMPLETE** (see implementation notes below)
5. **Phase 5**: Full round-trip test (structural + simulation) — **COMPLETE** (see implementation notes below)
6. **Phase 6**: Add COMAK knee to Rajagopal — 6A: audit both models + mapping (**COMPLETE**), 6B: scale Rajagopal down + add full COMAK knee (5 bodies + weld joints) + test with gait data
7. **Phase 7**: `scale_comak_knee_config` + `compute_knee_scale_factor` — uniform scaling of configs
8. **Phase 8**: `build_comak_knee_config` — bridge model_building → ComakKneeConfig (enables synthetic + MRI paths)
9. **Phase 9**: NSM pipeline integration — preserve subject bone size through similarity registration (rigid-only + scale undo)
10. **Commit code changes, then autoformat separately**

---

## Files Created/Modified

| File | Action | Purpose |
|------|--------|---------|
| `nsosim/knee_assembly/` | CREATE | Subpackage (plan originally had flat `knee_assembly.py`, refactored to subpackage) |
| `nsosim/knee_assembly/__init__.py` | CREATE | Re-exports all dataclasses + `strip_comak_knee` |
| `nsosim/knee_assembly/config.py` | CREATE | Data classes (`ComakKneeConfig`, `ComakBody`, etc.) + JSON serialization |
| `nsosim/knee_assembly/strip.py` | CREATE | `strip_comak_knee()` — extract-as-you-strip + replacement joint |
| `nsosim/knee_assembly/add.py` | CREATE | `add_comak_knee()` + all `_add_*` helpers |
| `nsosim/__init__.py` | MODIFY | Export `knee_assembly` subpackage |
| `tests/test_knee_assembly.py` | CREATE | 64 tests: dataclass construction, serialization, strip validation, structural round-trip |
| `tests/test_knee_assembly_forsim.py` | CREATE | 4 tests: COMAK simulation round-trip (settle + loaded forward sim) |
| `tests/fixtures/osim_models/full_body_healthy_knee.osim` | COPY | Smith2019 reference model for strip tests (~1MB, checked into git) |
| `tests/fixtures/osim_models/Geometry/` | COPY | Mesh files needed by Smith2018ContactMesh at model load time (24MB, gitignored) |
| `.gitignore` | MODIFY | Added `tests/fixtures/osim_models/Geometry/` |

---

## Phase 1 Implementation Notes

**Structural change:** The plan specified a flat `nsosim/knee_assembly.py`. During implementation, we refactored to a subpackage `nsosim/knee_assembly/` with `config.py`, `strip.py`, and (planned) `add.py` — mirroring the `wrap_surface_fitting/` subpackage pattern. The `__init__.py` re-exports everything, so `from nsosim.knee_assembly import ComakBody` works unchanged.

**Added dataclass:** `ComakWrapSurface` — the plan had `wrap_surfaces: list[dict]` on `ComakKneeConfig`. We made it a proper typed dataclass instead, with cylinder-specific (`radius`, `length`) and ellipsoid-specific (`dimensions`) optional fields. Cleaner than untyped dicts.

**Muscle property defaults:** The plan's `ComakMuscle` dataclass (line 237-238) listed 6 fields. The spike revealed `recfem_r` has 24 properties. We captured all 19 scalar properties for exact round-trip (5 key + 14 dynamics/control with defaults). The 4 sub-component curves (`ActiveForceLengthCurve`, etc.) were left as OpenSim defaults — they can be added if round-trip diffs appear.

---

## Phase 3 Implementation Notes

### What changed vs the plan

**Removal order was wrong in the plan.** The plan said "joints → bodies → forces → contact geometry" (line 557). The actual working order is **forces → contacts → joints → bodies**. You must remove all forces/contacts that reference COMAK bodies BEFORE removing those bodies, otherwise `finalizeConnections()` segfaults on dangling socket references.

**Replacement knee joint is mandatory, not optional.** The plan said "optionally re-create a simplified `knee_r` joint" (line 568). In practice, removing the COMAK weld joints disconnects `tibia_r` from the joint tree (it was connected via `tibia_tibia_proximal_r` → `tibia_proximal_r` → `knee_r` → `femur_distal_r` → `femur_femur_distal_r` → `femur_r`). Without a replacement joint, `finalizeConnections()` segfaults — OpenSim can't handle an orphaned body in the joint tree. We always create a `PinJoint` named `knee_r` connecting `femur_r → tibia_r` with weld offsets baked into the parent/child offset frames.

**No `mode` parameter.** The plan proposed `mode='clean'` vs `mode='hinge'`. We implemented only the hinge path (PinJoint) since clean mode would require the caller to supply a replacement joint. Can be added later if needed.

**No offset frame cloning.** The plan emphasized cloning offset frames before joint removal. In practice, we don't need to clone — we already extracted the weld joint offset values into `ComakWeldJoint` dataclasses before any removals, and we use those stored values to construct the replacement PinJoint. The extraction-before-removal pattern makes cloning unnecessary.

**`Smith2018ContactMesh` validates mesh files at `Model()` load time.** The plan assumed mesh files were only needed at `initSystem()`. The JAM/COMAK fork's `Smith2018ContactMesh` calls `findMeshFile()` during model construction, so the `Geometry/` folder must be present even for pure extraction tests. We copied it to `tests/fixtures/osim_models/Geometry/` (gitignored, 24MB).

### OpenSim API gotchas discovered during implementation

These are in addition to the gotchas documented in the Quick Start section:

- **`body.get_inertia()` returns `Vec6`, not `Inertia`** — use `.get(i)` for element access, not subscripting `[i]`
- **`LinearFunction.getCoefficients()` returns `ArrayDouble`** — use `.get(i)` for element access, not subscripting `[i]`
- **`Coordinate` has no `getRange()` method** — use `getRangeMin()` and `getRangeMax()` separately
- **`PhysicalOffsetFrame.safeDownCast()` returns `None` (not exception)** for non-offset frames — must null-check before accessing translation/orientation
- **Removing all COMAK joints orphans `tibia_r`** — causes segfault at `finalizeConnections()`, not a catchable exception. Must add a replacement joint before finalizing.

### Test fixture setup

The Smith2019 `.osim` file (~1MB) is checked into git at `tests/fixtures/osim_models/`. The `Geometry/` folder (24MB, 163 VTP/STL files) is gitignored. Tests skip with `requires_smith2019` if the model file is missing. To set up fixtures locally:

```bash
# Copy .osim (already in git)
# Copy Geometry/ (not in git — must be done manually)
cp -r /path/to/comak_models/current/Geometry tests/fixtures/osim_models/
```

---

## Phase 4 Implementation Notes

### What changed vs the plan

**Skipped incremental Tier 1 testing.** The plan called for remove-one/add-back-one tests for each component type before attempting the full round-trip. In practice, the spike had already verified every setter API, so we wrote all `_add_*` helpers at once and went straight to the full round-trip. When things failed, the error messages pointed directly to the broken helper — no need for intermediate scaffolding.

### OpenSim API gotchas discovered during implementation

- **`SpringGeneralizedForce(name_str)` treats the arg as the coordinate name, not the force name.** The constructor silently sets `coordinate=name_str` and leaves the force name empty. All 24 springs ended up with duplicate names (`springgeneralizedforce`). Fix: use default constructor + `setName()`.
- **`Smith2018ContactMesh` has a `scale_frame` socket** that must be set (defaults to `/ground` in Smith2019). Without it, `finalizeConnections()` fails with "Connectee for Socket 'scale_frame' is unspecified".
- **`Inertia` 6-arg constructor works:** `osim.Inertia(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)` — no need for separate `setProductsOfInertia()`.
- **Importing `nsosim` changes `osim.StdVectorString`** to `opensim.moco.StdVectorString`, breaking the 3-arg `TimeSeriesTable(time, matrix, labels)` constructor. Fix: build tables with `appendRow()` instead.
- **Settled/rebuilt model files must be saved next to `Geometry/`** for `Smith2018ContactMesh.findMeshFile()` to resolve mesh paths at model load time.

---

## Phase 5 Implementation Notes

### Structural round-trip (test_knee_assembly.py::TestRoundTrip)

18 tests comparing original vs strip→add rebuilt model component-by-component:
- Component counts match (bodies, joints, forces, contacts, coordinates)
- Names match for all component types
- Spot-checked property values (ligament stiffness/slack length, spring stiffness, muscle force/fiber length, coordinate defaults)
- Wrap surfaces on COMAK bodies restored
- `initSystem()` succeeds on rebuilt model

### Simulation round-trip (test_knee_assembly_forsim.py::TestForsimRoundTrip)

4 tests validating functional equivalence via COMAK forward simulation:

**Protocol:**
1. **Settle** (independently for each model): 2-frame ForsimTool (0.01s) with all 24 COMAK unconstrained coordinates. Update coordinate defaults to settled values.
2. **Forward sim**: 6-frame ForsimTool (0.05s) from settled state under loaded knee flexion:
   - Knee flexion ramp: 0 → 10°
   - Quad activation (recfem, vasint, vaslat, vasmed): 10% → 30% ramp
   - Hamstring co-contraction (bflh, bfsh, semimem, semiten): constant 15%
   - Gastrocnemius (gaslat, gasmed): constant 10%
3. **Compare**: All coordinate value columns agree within 0.007 rad (~0.4°)

**Observed diffs:**
| Coordinate | Max diff |
|-----------|---------|
| meniscus_medial_flex_r | 0.23° |
| meniscus_lateral_flex_r | 0.08° |
| knee_add_r | 0.08° |
| pf_flex_r | 0.06° |
| All others | < 0.04° |

**Why diffs exist:** Component ordering in the rebuilt model's ForceSet/BodySet differs from the original. The COMAK solver iterates over contacts and ligaments to find equilibrium, so iteration order affects convergence path. The independent settle step reduces but doesn't eliminate this. The diffs are all in COMAK secondary DOFs — prescribed coordinates and non-COMAK coordinates match to machine precision.

**Runtime:** ~2.5 minutes total (4 ForsimTool runs: 2 settle + 2 forward). Uses `integrator_accuracy=1e-2` for speed.

---

## Scaling & Placement Strategy

### Architecture: Three Paths, One Insertion Function

All paths produce a `ComakKneeConfig` and pass it to `add_comak_knee()`. They differ in how the config is built and whether scaling is needed.

**Path A: Generic swap (no subject data)**
```
Reference Smith2019 config (from strip or JSON)
  → scale_comak_knee_config(config, segment_length_ratio)
  → add_comak_knee(target_model, scaled_config)
```
Use case: Put a COMAK knee into any generic model for testing or teaching. The reference config is pre-built and stored as JSON.

**Path B: Synthetic knee (no MRI, NSM-generated geometry)**
```
Decode from latent vectors using reference T_fem → meshes at reference size
  → model_building pipeline (wrap fitting, ligament interp, contacts, menisci, fat pad)
  → build_comak_knee_config(meshes, reference_structural_config) → ComakKneeConfig
  → scale_comak_knee_config(config, segment_length_ratio)
  → add_comak_knee(target_model, scaled_config)
```
Use case: Subject-specific simulation without MRI. Latent vectors may be sampled, mean, or interpolated. All model_building work happens at reference size (decoded and reference meshes match), then the final config is scaled to the target model.

**Path C: Subject-specific with MRI**
```
Subject MRI → similarity register to reference → decompose into (scale, R, t)
  → apply rigid only (rotation + translation), then undo scale → aligned at subject's actual size
  → NSM fit → fitted latents + subject's linear_transform (encodes subject's actual bone size)
  → decode using subject's linear_transform → meshes at subject's actual anatomical size
  → model_building pipeline → wrap surfaces, ligament attachments, contacts at actual size
  → build_comak_knee_config(meshes, reference_structural_config) → ComakKneeConfig
  → add_comak_knee(AB_scaled_model, config, scale=1.0)
```
Use case: Full subject-specific simulation with MRI-derived knee geometry. The key difference from the current production pipeline: similarity registration preserves the subject's bone size by undoing the scale component after alignment. The `linear_transform` then encodes the mapping from subject-sized mm → canonical space, so decode produces subject-sized output directly. No additional scaling needed.

**Why Path C doesn't need scaling:** In the current production pipeline, similarity registration normalizes all subjects to reference size (femur `linear_transform` column norms are ~0.013 and nearly constant across subjects). The corrected pipeline decomposes the similarity transform, applies rigid alignment only, and scales back to original size before NSM fitting. The `linear_transform` then has subject-specific column norms, and `inv(linear_transform)` during decode expands to the subject's actual bone size.

### `build_comak_knee_config`: Bridging model_building → ComakKneeConfig

Currently, `model_building.py` feeds outputs directly to `comak_osim_update.py` which modifies an existing model in place. For the new architecture, a `build_comak_knee_config()` function packages model_building outputs into a `ComakKneeConfig`:

```python
def build_comak_knee_config(
    meshes: dict,                    # decoded bone/cart/meniscus meshes (any size)
    reference_config: ComakKneeConfig,  # structural skeleton from strip
) -> ComakKneeConfig:
    """Run model_building pipeline, package results as ComakKneeConfig.

    The reference_config provides structural components not derivable from
    geometry: joint topology, coordinate definitions/ranges, spring stiffnesses,
    DOF coupling. The model_building pipeline fills in geometric components:
    ligament attachment XYZ, wrap surface parameters, contact mesh STLs.
    """
```

This separates **structural** data (from reference config: joints, coordinates, springs, DOFs) from **geometric** data (from model_building: ligament XYZ, wrap parameters, contact mesh STLs). Both paths B and C use this function — they differ only in the input mesh sizes.

### `scale_comak_knee_config`: Uniform Scaling

```python
def scale_comak_knee_config(config: ComakKneeConfig, scale: float) -> ComakKneeConfig:
    """Scale all spatial quantities in a config by a uniform factor."""
```

| Component | Scale by | Notes |
|-----------|----------|-------|
| Joint offset translations | s | Weld offsets, custom joint offsets |
| Ligament path point locations | s | |
| Ligament slack lengths | s | Geometric lengths |
| Wrap surface translations | s | |
| Wrap surface radii/lengths/dimensions | s | |
| Spatial transform translation Y-values | s | LinearFunction intercepts for translation axes |
| Spring rest_length | s | |
| Body mass | s³ | Scales with volume |
| Body inertia | s⁵ | Scales with mass × length² |
| STL/VTP geometry files | s | Must write scaled mesh files to Geometry/ |
| Contact mesh thickness | — | Material property, scale-invariant |
| Spatial transform rotation Y-values | — | Angles are scale-invariant |
| Spring stiffness | — | Initially; may need tuning |

Scale factor source:
- Paths A/B: `mean(target_femur_length / ref_femur_length, target_tibia_length / ref_tibia_length)` — from segment lengths stored in `ComakKneeConfig`
- Path C: `scale=1.0` — geometry is already at subject size from NSM decode

### Contact mesh scaling: always write correctly-sized STLs

The legacy pipeline used `socket_scale_frame` to apply body scale factors to reference-sized contact mesh STLs at runtime, because physically scaling STLs required a library not in the `comak` environment. The legacy README noted: "Scale factors cannot be changed in xml file since the comak ik will not run if that's the case."

The new pipeline avoids this workaround entirely: `scale_comak_knee_config` physically scales the STL files and writes them to `Geometry/`. Contact meshes use identity scale (`scale_factors=[1,1,1]`) and `socket_scale_frame` pointing to their parent body (which should have identity scale for the COMAK-specific bodies). This is simpler and more predictable than runtime scaling.

### Legacy prior art

The legacy pipeline at `stanford_jam_gait_2023/AddBiomechanics/Process_Pipeline/` had two completely separate paths:

**Without MRI:** `scaleModel.py` extended AddBiomechanics' `rescaling_setup.xml` with COMAK body entries (femur_distal_r → femur scale, tibia_proximal_r → tibia scale, patella_r → femur scale), ran OpenSim ScaleTool on the original lenhart2015.osim, then fixed contact mesh scale frames. Three scaling modes: "AB" (anisotropic XYZ), "LA" (long-axis only), "WA" (weighted average). `scaleModel2.py` additionally rescaled muscle fiber/tendon lengths by path length ratios. `update_the_model_wrapping.ipynb` added a hardcoded quadriceps wrap ellipsoid with absolute coordinates (added after scaling, so not auto-scaled). The COMAK knee was never stripped and re-added — the original model was used directly as ScaleTool input.

**With MRI (August 2024):** `scaleModel_function` was commented out entirely. NSM pipeline produced subject-specific meshes at reference size, `comak_osim_update.py` swapped geometry in place on the Smith2019 model. No scaling needed because both model and meshes were at reference size. This is the limitation the new architecture addresses — it was locked to Smith2019 as the base model.

### Coordinate system and joint placement

The COMAK knee config stores joint offsets relative to parent/child bodies. When placing into a different model (Rajagopal), the target model's knee joint may have:
- Different offset frame orientations
- Different coordinate names (`knee_angle_r` vs `knee_flex_r`)
- Different joint types (PinJoint vs CustomJoint)
- Constraints or muscles that reference the old coordinate name

These are model-specific nuances handled during `add_comak_knee()`. The function signature accepts explicit body/joint names for mapping:

```python
add_comak_knee(
    model,
    config,
    femur_body="femur_r",
    tibia_body="tibia_r",
    knee_joint="knee_r",           # existing joint to replace
    knee_coordinate="knee_angle_r", # existing coordinate to remap to knee_flex_r
    hip_joint="hip_r",             # for coordinate frame context
    ankle_joint="ankle_r",         # for coordinate frame context
)
```

### Open questions for Phase 6A audit

1. What are Rajagopal's knee joint offset frames? Any extra rotations vs Smith2019?
2. What coordinates reference `knee_angle_r`? (constraints, muscles, coupled coordinates)
3. Do femur_r/tibia_r body frames have the same orientation convention as Smith2019?
4. Are there `CoordinateCouplerConstraint` entries that reference knee coordinates?
5. What spanning muscles exist and what are their path points?

---

## Phase 6: Add COMAK Knee to Rajagopal

Add a COMAK knee to the RajagopalLaiUhlrich2023 model. Start by scaling Rajagopal down to Smith2019 reference size so that the COMAK knee config can be added at scale=1.0 — this allows direct comparison with existing Smith2019 forward sim results.

**Target model:** [RajagopalLaiUhlrich2023.osim](https://github.com/opensim-org/opensim-models/blob/master/Models/Rajagopal/RajagopalLaiUhlrich2023.osim) from `opensim-org/opensim-models` (includes Geometry folder).

### Prerequisites

- Download the Rajagopal model + Geometry to `tests/fixtures/osim_models/rajagopal/`
- Smith2019 COMAK knee config (from Phase 3 strip, already available)

### Approach: Scale Entire Rajagopal Model Down to Smith2019 Size

Scale the **entire** Rajagopal model (whole body, not just legs) down to match Smith2019 dimensions. This keeps the COMAK knee at its validated reference size (scale=1.0) and ensures consistent body proportions for dynamics.

**Step 1 — Audit and store model data:** Phase 6A produces structured data files (JSON/CSV) for both models containing:
- Every body: name, mass, inertia, mass center
- Every joint: name, type, parent body, child body, parent/child offset translations and orientations
- Per-body segment lengths (joint-to-joint distances along each axis)
- Every coordinate: name, default value, range, locked/clamped
- Every muscle: name, type, max force, path points
- Every constraint: name, type, referenced coordinates
- Every wrap surface: name, type, parent body, dimensions, position

These files are the ground truth for computing scale factors and identifying model differences. Store in `scripts/phase6_rajagopal_audit/`.

**Step 2 — Compute scale factors from stored data:** Scale factors are computed dynamically from the audit data. Default scaling strategy:
- **Long bones** (femur, tibia, humerus, radius, etc.): long-axis (Y) scale factor only, applied isotropically to all 3 axes
- **Pelvis**: 2 scale factors — medial/lateral (X) and anterior/posterior (Z). Y (superior/inferior) can match one of these or be computed separately
- **Other bodies** (trunk, foot segments, etc.): long-axis only, isotropic

```python
# From stored audit data:
smith_data = load_model_audit("scripts/phase6_rajagopal_audit/smith2019_audit.json")
raj_data = load_model_audit("scripts/phase6_rajagopal_audit/rajagopal_audit.json")

# Compute per-body scale factors from segment lengths
scale_factors = compute_model_scale_factors(smith_data, raj_data, strategy="long_axis")
# Returns: {body_name: [sx, sy, sz]} for each body

# Build ScaleSet XML and run OpenSim ScaleTool
build_and_run_scale_tool(rajagopal_model, scale_factors, output_path)
```

The audit data files allow recomputing scale factors with different strategies without re-running the audit. OpenSim's ScaleTool handles the cascading effects (muscle attachments, joint offsets, wrap surfaces, bone meshes).

Benefits:
- COMAK knee at scale=1.0 — same as Smith2019 round-trip tests
- Whole-body consistency for dynamics (mass distribution, moments of inertia)
- Forward sim results should be comparable to Smith2019 baseline
- Can test with existing gait data
- Validates add machinery without scaling complexity

### Key challenges

1. **Remove existing knee joint and add intermediate bodies.** Rajagopal has `femur_r → knee_r → tibia_r` directly. The COMAK knee needs intermediate bodies (`femur_distal_r`, `tibia_proximal_r`) connected by weld joints. The add function must:
   - Remove Rajagopal's existing `knee_r` joint
   - Add `femur_distal_r` body + `femur_femur_distal_r` weld joint (connects `femur_r` → `femur_distal_r`)
   - Add `tibia_proximal_r` body + `tibia_tibia_proximal_r` weld joint (connects `tibia_r` → `tibia_proximal_r`)
   - Add the COMAK `knee_r` CustomJoint (connects `femur_distal_r` → `tibia_proximal_r`)
   - Add remaining COMAK bodies (`patella_r`, menisci) and their joints (`pf_r`, meniscus joints)

   The weld joints are kept (not consolidated) because they serve as local coordinate frames for the COMAK knee assembly. All ligament/wrap/contact positions are expressed relative to these intermediate bodies, which keeps the config clean for scaling and model-agnostic placement. The weld offset is the single value that positions the knee on the parent body.

2. **Coordinate name conflict.** Rajagopal uses `knee_angle_r`, COMAK uses `knee_flex_r`. The `patellofemoral_knee_angle_r_con` constraint references `knee_angle_r`. Start empirically (see what breaks), then implement renaming or reference updates.

3. **Spanning muscles.** Rajagopal already has `recfem_r`, `vasint_r`, `vaslat_r`, `vasmed_r`. Remove Rajagopal's versions, add Smith2019's versions (which attach to `patella_r`).

4. **Geometry files.** COMAK contact mesh STLs need to be copied to Rajagopal's `Geometry/` folder.

### Implementation plan

**Phase 6A: Audit both models → structured data files + mapping**

Write an audit script that extracts comprehensive model data into JSON/CSV for both Smith2019 and Rajagopal. Store all outputs in `scripts/phase6_rajagopal_audit/`.

**Per-model audit files** (one set per model):
- Every body: name, mass, inertia, mass center
- Every joint: name, type, parent/child body, offset translations/orientations
- Per-body segment lengths (joint-to-joint distances, all 3 axes)
- Every coordinate: name, default value, range, locked/clamped
- Every muscle: name, type, max force, path points (body + location for each)
- Every constraint: name, type, referenced coordinates
- Every wrap surface: name, type, parent body, dimensions, position

**Cross-model mapping file** (the key deliverable):
A single JSON/CSV that maps between the two models:
- **Body mapping:** which Rajagopal body corresponds to which Smith2019 body (e.g., `femur_r` ↔ `femur_r`, noting any that exist in one model but not the other)
- **Joint mapping:** which Rajagopal joint corresponds to which Smith2019 joint (e.g., `knee_r` ↔ `knee_r`, noting type differences like PinJoint vs CustomJoint)
- **Coordinate mapping:** which coordinates correspond (e.g., `knee_angle_r` ↔ `knee_flex_r`)
- **Proposed scaling axes per body:** which axis/axes to use for computing scale factors:
  - Long bones (femur, tibia, humerus, radius, ulna, etc.): `Y` (long axis), applied isotropically
  - Pelvis: `X` (medial/lateral) + `Z` (anterior/posterior)
  - Foot segments, hand segments: `Y` (long axis), isotropic
  - Trunk segments: `Y`, isotropic
- **Per-body scale factors:** computed from segment lengths using the proposed axes, stored as `[sx, sy, sz]`
- **Coordinate conflicts:** coordinates that exist in Rajagopal but not in COMAK (or vice versa), and any constraints/muscles that reference them

This mapping file is both human-readable (for review) and machine-readable (for `compute_model_scale_factors()` to consume). Scale factors can be recomputed with different axis choices by editing the mapping file without re-running the audit.

**Phase 6B: Scale Rajagopal + add COMAK knee**
- Build ScaleSet XML from mapping data, run OpenSim ScaleTool to scale Rajagopal down to Smith2019 size
- Remove Rajagopal's existing `knee_r` joint and spanning muscles
- Add all 5 COMAK bodies (including intermediate `femur_distal_r`, `tibia_proximal_r`) with weld joints
- Add COMAK `knee_r` CustomJoint connecting `femur_distal_r` → `tibia_proximal_r`
- Add remaining joints, ligaments, springs, contacts, wrap surfaces, spanning muscles
- Handle coordinate name conflicts (`knee_angle_r` → `knee_flex_r`)
- Copy COMAK geometry files to Rajagopal's `Geometry/` folder
- Test: `initSystem()`, settle sim, forward sim with gait data

### Tests

- `test_rajagopal_audit` — enumerate model, verify expected structure
- `test_add_to_rajagopal_downscaled` — scale Rajagopal down, add full COMAK knee (5 bodies + weld joints), `initSystem()`
- `test_rajagopal_forsim` — settle + forward sim, reasonable secondary DOF values
- `test_rajagopal_gait` — run with existing gait data, compare to Smith2019 baseline

---

## Phase 7: `scale_comak_knee_config` + `compute_knee_scale_factor`

Implement the scaling function so configs can be scaled to any target model's segment lengths.

### Prerequisites

- Phase 6 working (COMAK knee in downscaled Rajagopal)

### Implementation

```python
def compute_knee_scale_factor(model, ref_femur_length, ref_tibia_length):
    target_fem = measure_femur_length(model)  # hip → knee distance
    target_tib = measure_tibia_length(model)  # knee → ankle distance
    fem_scale = target_fem / ref_femur_length
    tib_scale = target_tib / ref_tibia_length
    return np.mean([fem_scale, tib_scale])
```

Apply `scale_comak_knee_config(config, scale)` before `add_comak_knee()`. This enables:
- Adding COMAK knee to unscaled Rajagopal (scale up from reference)
- Adding COMAK knee to AB-scaled models (scale by AB segment lengths)
- Synthetic knees at any target size

### Tests

- `test_scale_factor_computation` — known segment lengths → expected scale factor
- `test_scaled_add_to_rajagopal` — add with scaling to unscaled Rajagopal, `initSystem()`
- `test_scaled_rajagopal_forsim` — settle + forward sim, reasonable DOF values
- `test_scale_identity` — scale=1.0 matches unscaled add (regression)

---

## Phase 8: `build_comak_knee_config` — Bridge model_building → ComakKneeConfig

Package `model_building.py` outputs into a `ComakKneeConfig` instead of feeding them to `comak_osim_update.py`. This enables Paths B and C (synthetic and MRI-derived knees).

### Prerequisites

- Phase 7 working (scaling)
- Existing `model_building.py` pipeline

### Implementation

```python
def build_comak_knee_config(
    meshes: dict,                       # decoded bone/cart/meniscus meshes
    reference_config: ComakKneeConfig,  # structural skeleton from strip
    dict_bones: dict,                   # bone info for model_building pipeline
) -> ComakKneeConfig:
    """Run model_building pipeline, package results as ComakKneeConfig."""
    # 1. Extract articular surfaces → contact mesh STLs
    # 2. Fit wrap surfaces → ComakWrapSurface entries
    # 3. Interpolate ligament attachments → update ligament path points
    # 4. Process meniscus surfaces
    # 5. Create prefemoral fat pad
    # 6. Return config with structural data from reference + geometric data from pipeline
```

### What comes from where

| Data | Source |
|------|--------|
| Joint topology, DOF definitions | `reference_config` (from Smith2019 strip) |
| Coordinate ranges, defaults | `reference_config` |
| Spring stiffnesses | `reference_config` |
| Ligament stiffness, damping, transition_strain | `reference_config` |
| Ligament path point XYZ | `model_building` pipeline (interpolated to decoded meshes) |
| Ligament slack lengths | Recomputed from new path point positions |
| Wrap surface parameters | `model_building` pipeline (fitted to decoded meshes) |
| Contact mesh STLs | `model_building` pipeline (articular surfaces from decoded meshes) |
| Spanning muscle path points | `model_building` pipeline (interpolated to patella) |

### Tests

- `test_build_config_from_reference_meshes` — decode reference latents → build config → compare to stripped Smith2019 config (geometry should match closely)
- `test_build_config_synthetic` — decode from sampled latents → build config → add to model → `initSystem()`
- `test_build_config_scaled` — build at reference size → scale → add to unscaled Rajagopal → `initSystem()` + forsim

---

## Phase 9: NSM Pipeline Integration — Subject-Specific with Size Preservation

Modify the NSM fitting pipeline to preserve the subject's actual bone size through the decode path.

### Prerequisites

- Phase 8 working (build_comak_knee_config)

### Changes to NSM fitting pipeline

Currently, similarity registration normalizes subjects to reference size. The change:

```python
# Current: similarity register → subject scaled to reference size
transform = similarity_icp(subject_mesh, reference_mesh)
aligned_mesh = apply_transform(subject_mesh, transform)

# New: similarity register → decompose → apply rigid only → restore size
transform = similarity_icp(subject_mesh, reference_mesh)
scale, R, t = decompose_similarity(transform)
rigid_transform = compose_rigid(R, t)
aligned_mesh = apply_transform(subject_mesh, rigid_transform)
# aligned_mesh is now: properly aligned to reference orientation, at subject's actual size
```

The internal `linear_transform` from `reconstruct_mesh()` will now encode subject-sized-mm → canonical, with subject-specific column norms. `decode_latent_to_osim()` with this transform produces subject-sized output.

### Full Path C flow

```
AB-scale Rajagopal to subject → body dimensions match subject
  → Subject MRI → similarity register (rigid only) → NSM fit at subject's actual size
  → decode with subject's linear_transform → meshes at subject's actual size
  → build_comak_knee_config(meshes, reference_config) → config at subject size
  → add_comak_knee(AB_scaled_rajagopal, config)  # scale=1.0
```

### Tests

- `test_rigid_registration_preserves_size` — verify aligned mesh has same bounding box as original
- `test_decode_at_subject_size` — decode with subject's transform, verify output size matches subject anatomy
- `test_path_c_end_to_end` — full pipeline: MRI → config → add to AB-scaled model → `initSystem()` + forsim

---

## Future Phases (Beyond Phase 9)

- **Bilateral support**: Side parameter (`'l'`) with name mirroring and geometry reflection
- **Strip for predsim**: `strip_for_predsim()` that also converts splines to polynomials and adds foot-ground contacts
- **Gait validation**: Run full gait simulation pipeline (COMAK IK → COMAK Tool → COMAK ID → Joint Mechanics) on Rajagopal+COMAK models and compare to Smith2019 baseline

---

## Dependencies & Prerequisites

- **OpenSim Python API** (JAM/COMAK fork) — required for all component access. Tests need the `comak` conda env.
- **Access to Smith2019 .osim model** — needed as the reference for extraction and round-trip testing.
- **RajagopalLaiUhlrich2023 model + Geometry** — needed for Phase 6+.
- The existing `osim_utils.py` helper functions (`create_contact_mesh`, `add_contact_mesh_to_model`, etc.) will be reused directly.
