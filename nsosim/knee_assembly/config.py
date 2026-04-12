"""
Data classes for COMAK knee assembly configuration.

These dataclasses represent the "stored" form of a COMAK knee — extracted from
a reference model and ready to be serialized to JSON or written back into any
OpenSim model.

Socket paths are stored as body *names* (not full paths) so they resolve
correctly when added to a different model.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ComakBody:
    """A body in the COMAK knee (e.g., femur_distal_r, patella_r)."""

    name: str
    mass: float
    inertia: list  # [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    mass_center: list  # [x, y, z]
    attached_geometry: list = field(default_factory=list)  # [{name, mesh_file, ...}]


@dataclass
class ComakWeldJoint:
    """A weld joint connecting a main-chain body to a COMAK body."""

    name: str
    parent_body: str
    child_body: str
    parent_offset_translation: list  # [x, y, z]
    parent_offset_orientation: list  # [x, y, z] Euler angles (radians)
    child_offset_translation: list  # [x, y, z]
    child_offset_orientation: list  # [x, y, z] Euler angles (radians)


@dataclass
class ComakCoordinate:
    """A single coordinate (DOF) within a custom joint."""

    name: str
    default_value: float
    range_min: float
    range_max: float
    locked: bool
    clamped: bool


@dataclass
class ComakCustomJoint:
    """A custom joint with up to 6 DOFs and a spatial transform.

    The spatial_transform dict has keys 'rotation1' through 'rotation3' and
    'translation1' through 'translation3'. Each value is a dict with:
        - 'axis': [x, y, z]
        - 'coordinate': str or None (None = no coordinate assigned)
        - 'function': {'type': 'LinearFunction', 'slope': float, 'intercept': float}

    All COMAK joints in Smith2019 use LinearFunction only. SimmSpline support
    is deferred to future phases (other models may use splines).
    """

    name: str
    parent_body: str
    child_body: str
    parent_offset_translation: list  # [x, y, z]
    parent_offset_orientation: list  # [x, y, z] Euler angles (radians)
    child_offset_translation: list  # [x, y, z]
    child_offset_orientation: list  # [x, y, z] Euler angles (radians)
    coordinates: list  # list of ComakCoordinate dicts
    spatial_transform: dict  # see docstring


@dataclass
class ComakLigament:
    """A Blankevoort1991Ligament with path points."""

    name: str
    linear_stiffness: float
    transition_strain: float  # default 0.06
    damping_coefficient: float  # default 0.003
    slack_length: float
    path_points: list  # [{name: str, body: str, location: [x, y, z]}]


@dataclass
class ComakSpring:
    """A SpringGeneralizedForce acting on a single coordinate."""

    name: str
    coordinate: str
    stiffness: float
    rest_length: float
    viscosity: float


@dataclass
class ComakContactMesh:
    """A Smith2018ContactMesh for articular contact."""

    name: str
    parent_frame: str  # body name (resolved to socket path during add)
    mesh_file: str
    elastic_modulus: float  # default 1e6
    poissons_ratio: float  # default 0.5
    thickness: float  # default 0.005
    location: list  # [x, y, z], default [0, 0, 0]
    orientation: list  # [x, y, z], default [0, 0, 0]
    use_variable_thickness: bool = False
    mesh_back_file: str = ""
    min_thickness: float = 0.001
    max_thickness: float = 0.01
    scale_factors: list = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class ComakContactForce:
    """A Smith2018ArticularContactForce between two contact meshes."""

    name: str
    target_mesh: str  # contact mesh name
    casting_mesh: str  # contact mesh name
    min_proximity: float = 0.0
    max_proximity: float = 0.01
    elastic_foundation_formulation: str = "linear"
    use_lumped_contact_model: bool = True


@dataclass
class ComakWrapSurface:
    """A wrap surface (cylinder or ellipsoid) attached to a COMAK body.

    Reuses the same parameter conventions as wrap_surface_fitting/main.py.
    """

    name: str
    parent_body: str  # body the wrap is attached to
    type: str  # 'WrapCylinder' or 'WrapEllipsoid'
    translation: list  # [x, y, z] (meters)
    xyz_body_rotation: list  # [x, y, z] Euler angles (radians)
    quadrant: str  # e.g., 'x', '-x', 'all'
    # Cylinder-specific
    radius: Optional[float] = None  # meters
    length: Optional[float] = None  # meters
    # Ellipsoid-specific
    dimensions: Optional[list] = None  # [x, y, z] radii (meters)


@dataclass
class ComakMuscle:
    """A spanning muscle (Millard2012EquilibriumMuscle) that crosses the COMAK knee.

    Captures all scalar properties needed for exact round-trip. Sub-component
    curves (ActiveForceLengthCurve, etc.) use OpenSim defaults — if a future
    round-trip test shows diffs on curves, they can be added here.
    """

    name: str
    # Key mechanical properties
    max_isometric_force: float
    optimal_fiber_length: float
    tendon_slack_length: float
    pennation_angle_at_optimal: float
    max_contraction_velocity: float = 10.0
    # Path
    path_points: list = field(default_factory=list)  # [{name, body, location}]
    wrap_objects: list = field(default_factory=list)  # [wrap_object_name, ...]
    # Dynamics / control (usually defaults, but captured for exact round-trip)
    min_control: float = 0.01
    max_control: float = 1.0
    optimal_force: float = 1.0
    ignore_tendon_compliance: bool = False
    ignore_activation_dynamics: bool = False
    fiber_damping: float = 0.1
    default_activation: float = 0.01
    default_fiber_length: float = 0.1
    activation_time_constant: float = 0.01
    deactivation_time_constant: float = 0.03
    minimum_activation: float = 0.01
    maximum_pennation_angle: float = 1.47063


@dataclass
class ComakKneeConfig:
    """Complete description of a COMAK knee, extracted from a reference model.

    This is the top-level container that holds all COMAK knee components.
    It can be serialized to/from JSON for storage and transfer between models.
    """

    side: str  # 'r' or 'l'
    bodies: list  # list of ComakBody
    weld_joints: list  # list of ComakWeldJoint
    custom_joints: list  # list of ComakCustomJoint
    ligaments: list  # list of ComakLigament
    springs: list  # list of ComakSpring
    contact_meshes: list  # list of ComakContactMesh
    contact_forces: list  # list of ComakContactForce
    wrap_surfaces: list  # list of ComakWrapSurface
    spanning_muscles: list  # list of ComakMuscle
    ref_femur_length: float  # hip center -> knee center (meters)
    ref_tibia_length: float  # knee center -> ankle center (meters)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to a plain dict (all nested dataclasses become dicts)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ComakKneeConfig":
        """Reconstruct from a plain dict (e.g., loaded from JSON)."""
        return cls(
            side=d["side"],
            bodies=[ComakBody(**b) for b in d["bodies"]],
            weld_joints=[ComakWeldJoint(**wj) for wj in d["weld_joints"]],
            custom_joints=[_custom_joint_from_dict(cj) for cj in d["custom_joints"]],
            ligaments=[ComakLigament(**lig) for lig in d["ligaments"]],
            springs=[ComakSpring(**sp) for sp in d["springs"]],
            contact_meshes=[ComakContactMesh(**cm) for cm in d["contact_meshes"]],
            contact_forces=[ComakContactForce(**cf) for cf in d["contact_forces"]],
            wrap_surfaces=[ComakWrapSurface(**ws) for ws in d["wrap_surfaces"]],
            spanning_muscles=[ComakMuscle(**m) for m in d["spanning_muscles"]],
            ref_femur_length=d["ref_femur_length"],
            ref_tibia_length=d["ref_tibia_length"],
        )

    def to_json(self, path: str) -> None:
        """Serialize to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ComakKneeConfig":
        """Deserialize from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _custom_joint_from_dict(d: dict) -> ComakCustomJoint:
    """Reconstruct a ComakCustomJoint, converting nested coordinate dicts."""
    coords = [ComakCoordinate(**c) for c in d["coordinates"]]
    return ComakCustomJoint(
        name=d["name"],
        parent_body=d["parent_body"],
        child_body=d["child_body"],
        parent_offset_translation=d["parent_offset_translation"],
        parent_offset_orientation=d["parent_offset_orientation"],
        child_offset_translation=d["child_offset_translation"],
        child_offset_orientation=d["child_offset_orientation"],
        coordinates=coords,
        spatial_transform=d["spatial_transform"],
    )
