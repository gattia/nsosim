# COMAK Knee Assembly: Strip, Add, and Round-Trip Validation

## Context

`nsosim` currently assumes the COMAK knee already exists in the target model — it only *updates* existing components (geometry, wrap surfaces, ligaments, contacts) via `comak_osim_update.py` and `osim_utils.py`. It cannot add a COMAK knee to a model that doesn't have one, nor can it remove one.

The combined COMAK + predictive simulation plan (Track 2) requires placing a COMAK knee into *any* OpenSim model (e.g., Rajagopal), not just Smith2019. This means nsosim needs three new capabilities:

1. **Extract** — Read all COMAK knee components from a reference model into a structured config
2. **Strip** — Remove all COMAK knee components from a model (leaving a generic knee)
3. **Add** — Insert COMAK knee components into any model that has a femur and tibia

The round-trip test (strip Smith2019 → re-add → compare) validates the machinery before generalizing to other models.

**Scope**: Smith2019 round-trip only. Model-agnostic placement (Rajagopal) and bilateral support are future phases.

**Source plan**: `/dataNAS/people/aagatti/projects/comak_gait_simulation/NOTES/combined_comak_predsim_plan.md` (Track 2)

**Prior art**: Stripping scripts at `/dataNAS/people/aagatti/projects/gait_opensim_jam_2023/stanford_jam_gait_2023/AddBiomechanics/` — `replace_comak_knee_with_generic.py`, v1/v2 notebooks.

---

## COMAK Knee Component Inventory

The COMAK knee in Smith2019 consists of:

| Category | Count | Examples |
|----------|-------|---------|
| **Bodies** | 5 | `femur_distal_r`, `tibia_proximal_r`, `patella_r`, `meniscus_medial_r`, `meniscus_lateral_r` |
| **Joints** | 6 | 2 weld (`femur_femur_distal_r`, `tibia_tibia_proximal_r`), 1 TF (`knee_r`), 1 PF (`pf_r`), 2 meniscus |
| **Coordinates** | 24 | 6 TF + 6 PF + 6 medial meniscus + 6 lateral meniscus |
| **Ligaments** | ~80+ | `Blankevoort1991Ligament` (MCL, ACL, PCL, LCL, PT, PFL, etc.) |
| **Contact forces** | 2 | `tf_contact`, `pf_contact` (`Smith2018ArticularContactForce`) |
| **Contact meshes** | 7+ | `Smith2018ContactMesh` (femur/tibia/patella cartilage, 4 meniscus surfaces) |
| **Springs** | ~12 | `SpringGeneralizedForce` (passive stiffness per DOF) |
| **Wrap surfaces** | 7 | 3 cylinders + 4 ellipsoids (see `wrap_surface_fitting/config.py`) |

---

## Architecture

New module: **`nsosim/knee_assembly.py`**

Three public functions matching the three-step decomposition:

```
extract_comak_knee(model) → ComakKneeConfig
strip_comak_knee(model, side='r') → model
add_comak_knee(model, knee_config, side='r') → model
```

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
    slack_length: float
    path_points: list[dict]     # [{name, body, location: [x,y,z]}]

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
    parent_frame: str
    mesh_file: str
    elastic_modulus: float
    poissons_ratio: float
    thickness: float
    # + variable thickness params

@dataclass
class ComakContactForce:
    name: str
    target_mesh: str
    casting_mesh: str
    min_proximity: float
    max_proximity: float

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
    wrap_surfaces: list[dict]           # reuse existing wrap_surface format

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
- `wrap_surfaces` reuse the existing `wrap_surface.to_dict()` format from `wrap_surface_fitting/main.py`.

---

## Phase 2: Extract (`extract_comak_knee`)

```python
def extract_comak_knee(model: osim.Model, side: str = 'r') -> ComakKneeConfig:
    """Extract all COMAK knee components from an OpenSim model."""
```

### Implementation:

Uses the OpenSim Python API (not XML parsing) to iterate over model component sets and extract COMAK-specific components. Identification is by **concrete class name** and **naming convention** (suffix `_r` or `_l`).

**Existing functions to reuse:**
- `get_osim_muscle_ligament_reference_lengths()` (`osim_utils.py:503`) — extracts force properties (partially covers ligaments)
- `extract_wrap_parameters_from_osim()` (`parameter_extraction.py:22`) — extracts wrap surface params

**New extraction logic needed:**
- Bodies: iterate `model.getBodySet()`, match COMAK body names
- Joints: iterate `model.getJointSet()`, match COMAK joint names, extract spatial transform functions and coordinates
- Ligaments: iterate `model.getForceSet()`, filter by `getConcreteClassName() == 'Blankevoort1991Ligament'`, extract path points + stiffness + slack length
- Springs: filter by `getConcreteClassName() == 'SpringGeneralizedForce'`
- Contact meshes: iterate `model.getContactGeometrySet()`, filter by `Smith2018ContactMesh`
- Contact forces: filter by `Smith2018ArticularContactForce`

**COMAK component names (Smith2019, right side):**
```python
COMAK_BODIES = ['femur_distal_r', 'tibia_proximal_r', 'patella_r', 'meniscus_medial_r', 'meniscus_lateral_r']
COMAK_JOINTS = ['femur_femur_distal_r', 'tibia_tibia_proximal_r', 'knee_r', 'pf_r', 'meniscus_medial_r', 'meniscus_lateral_r']
COMAK_FORCE_CLASSES = ['Blankevoort1991Ligament', 'SpringGeneralizedForce', 'Smith2018ArticularContactForce']
COMAK_CONTACT_CLASS = 'Smith2018ContactMesh'
```

### Spatial transform extraction (most complex part):

For each `CustomJoint`, extract the 6-component `SpatialTransform`:
```python
joint = osim.CustomJoint.safeDownCast(joint_obj)
st = joint.getSpatialTransform()
# For each of rotation1-3, translation1-3:
axis = st.get_rotation1()
func = axis.get_function()
coord_name = axis.get_coordinates(0) if axis.get_coordinates().getSize() > 0 else None
func_type = func.getConcreteClassName()  # 'SimmSpline', 'LinearFunction', 'Constant', etc.
# Extract function parameters based on type
```

Store as:
```python
{
    'rotation1': {'axis': [x,y,z], 'function': {'type': 'SimmSpline', 'x': [...], 'y': [...]}, 'coordinate': 'knee_flex_r'},
    'rotation2': {'axis': [x,y,z], 'function': {'type': 'LinearFunction', 'slope': 1.0, 'intercept': 0.0}, 'coordinate': 'knee_add_r'},
    ...
}
```

---

## Phase 3: Strip (`strip_comak_knee`)

```python
def strip_comak_knee(model: osim.Model, side: str = 'r') -> osim.Model:
    """Remove all COMAK knee components, leaving a generic femur→tibia joint."""
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

1. Clone offset frame from `femur_femur_distal_r` weld joint
2. Remove COMAK joints (6) in reverse index order
3. Remove COMAK bodies (5) in reverse index order
4. Remove COMAK forces: all `Blankevoort1991Ligament`, `SpringGeneralizedForce`, `Smith2018ArticularContactForce`
5. Remove `Smith2018ContactMesh` entries from contact geometry set
6. Remove wrap surfaces from bodies that were deleted (wraps on `femur_r` stay since the body remains)
7. Optionally re-create a simplified `knee_r` joint (hinge or locked) using the cloned offset frame
8. `model.finalizeConnections()`

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
    parent_femur_body: str = 'femur_r',
    parent_tibia_body: str = 'tibia_r',
    side: str = 'r',
) -> osim.Model:
    """Add COMAK knee components to a model that has femur and tibia bodies."""
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

### Existing functions to reuse:
- `create_contact_mesh()` and `add_contact_mesh_to_model()` (`osim_utils.py:159-260`)
- `create_articular_contact_force()` and `add_contact_force_to_model()` (`osim_utils.py:264-352`)

### New helper functions needed:
- `_add_body(model, body_config)` — create and add an OpenSim Body
- `_add_weld_joint(model, joint_config)` — create WeldJoint with offset frames
- `_add_custom_joint(model, joint_config)` — create CustomJoint with SpatialTransform
- `_add_ligament(model, lig_config)` — create Blankevoort1991Ligament with path points
- `_add_spring(model, spring_config)` — create SpringGeneralizedForce
- `_add_wrap_surface(model, body_name, wrap_config)` — create WrapCylinder or WrapEllipsoid
- `_rebuild_spatial_transform(transform_dict)` — reconstruct SpatialTransform from serialized dict

---

## Phase 5: Round-Trip Test

New test file: **`tests/test_knee_assembly.py`**

### Test strategy:

1. Load the Smith2019 production model (`full_body_healthy_knee.osim`)
2. Extract: `config = extract_comak_knee(model)`
3. Strip: `stripped = strip_comak_knee(model.clone())`
4. Re-add: `rebuilt = add_comak_knee(stripped, config)`
5. Compare `original` vs `rebuilt` component by component

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

## Implementation Order

1. **Phase 1**: Data classes in `knee_assembly.py` + serialization (to_dict/from_dict/JSON)
2. **Phase 2**: `extract_comak_knee()` — extract from Smith2019 and verify config captures everything
3. **Phase 3**: `strip_comak_knee()` — strip Smith2019 and verify result loads
4. **Phase 4**: `add_comak_knee()` — re-add to stripped model
5. **Phase 5**: Round-trip test comparing original vs rebuilt
6. **Commit code changes, then autoformat separately**

Each phase is independently testable.

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `nsosim/knee_assembly.py` | CREATE | Data classes + extract/strip/add functions |
| `nsosim/__init__.py` | MODIFY | Export `knee_assembly` module |
| `tests/test_knee_assembly.py` | CREATE | Round-trip and component tests |

---

## Future Phases (Not in This Scope)

- **Model-agnostic placement**: Parameterize `parent_femur_body`, `parent_tibia_body`, and offset transforms for Rajagopal/other models
- **Bilateral support**: Side parameter (`'l'`) with name mirroring and geometry reflection
- **Strip for predsim**: `strip_for_predsim()` that also converts splines to polynomials and adds foot-ground contacts
- **Integration with existing update pipeline**: Wire `add_comak_knee()` into the NSM fitting pipeline as an alternative to the current "update existing model" path

---

## Dependencies & Prerequisites

- **OpenSim Python API** (JAM/COMAK fork) — required for all component access. Tests need the `comak` conda env.
- **Access to Smith2019 .osim model** — needed as the reference for extraction and round-trip testing.
- The existing `osim_utils.py` helper functions (`create_contact_mesh`, `add_contact_mesh_to_model`, etc.) will be reused directly.
