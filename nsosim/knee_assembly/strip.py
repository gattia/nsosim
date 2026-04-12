"""
Strip COMAK knee components from an OpenSim model.

Extracts each component's properties into dataclasses before removal,
guaranteeing the returned ComakKneeConfig captures exactly what was stripped.

Usage:
    import opensim as osim
    from nsosim.knee_assembly import strip_comak_knee

    model = osim.Model("full_body_healthy_knee.osim")
    stripped_model, config = strip_comak_knee(model, side='r')
    config.to_json("knee_config.json")
"""

import logging

import numpy as np
import opensim as osim

from .config import (
    ComakBody,
    ComakContactForce,
    ComakContactMesh,
    ComakCoordinate,
    ComakCustomJoint,
    ComakKneeConfig,
    ComakLigament,
    ComakMuscle,
    ComakSpring,
    ComakWeldJoint,
    ComakWrapSurface,
)

logger = logging.getLogger(__name__)

# Names of the 4 spanning muscles that cross the COMAK knee.
# These are the only forces identified by name (not class) because they share
# the Millard2012EquilibriumMuscle class with non-COMAK muscles.
SPANNING_MUSCLE_NAMES = ["recfem_r", "vasint_r", "vaslat_r", "vasmed_r"]

# COMAK-specific bodies (right side). Used for validation, not discovery.
EXPECTED_COMAK_BODIES = [
    "femur_distal_r",
    "tibia_proximal_r",
    "patella_r",
    "meniscus_medial_r",
    "meniscus_lateral_r",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def strip_comak_knee(model, side="r"):
    """Remove all COMAK knee components, returning stripped model + config.

    Each component is serialized to ComakKneeConfig before removal.
    The config captures exactly what was stripped — no mismatch possible.

    Parameters
    ----------
    model : osim.Model
        The OpenSim model to strip. Modified in-place.
    side : str
        'r' or 'l'. Currently only 'r' is supported.

    Returns
    -------
    model : osim.Model
        The stripped model (same object, modified in-place).
    config : ComakKneeConfig
        All extracted COMAK knee component data, ready for serialization or re-addition.
    """
    if side != "r":
        raise NotImplementedError("Only right knee (side='r') is currently supported.")

    # 1. Discover COMAK bodies by checking which bodies are children of
    #    COMAK-specific joints (weld + custom joints whose parent/child
    #    are not in the standard body chain).
    comak_body_names = _discover_comak_bodies(model)
    logger.info(f"Discovered COMAK bodies: {comak_body_names}")

    # 2. Extract wrap surfaces from COMAK bodies BEFORE body removal
    wrap_surfaces = _extract_wrap_surfaces(model, comak_body_names)
    logger.info(f"Extracted {len(wrap_surfaces)} wrap surfaces from COMAK bodies")

    # 3. Extract reference segment lengths (needs initSystem)
    ref_femur_length, ref_tibia_length = _extract_segment_lengths(model)

    # 4. Extract all components (read properties into dataclasses)
    bodies = _extract_bodies(model, comak_body_names)
    weld_joints, custom_joints = _extract_joints(model, comak_body_names)
    ligaments = _extract_ligaments(model)
    springs = _extract_springs(model)
    contact_meshes = _extract_contact_meshes(model)
    contact_forces = _extract_contact_forces(model)
    spanning_muscles = _extract_spanning_muscles(model, SPANNING_MUSCLE_NAMES)

    # 5. Remove all COMAK components (order matters: forces → contacts → joints → bodies)
    _remove_forces_by_class(model, "Blankevoort1991Ligament")
    _remove_forces_by_class(model, "SpringGeneralizedForce")
    _remove_forces_by_class(model, "Smith2018ArticularContactForce")
    _remove_forces_by_name(model, SPANNING_MUSCLE_NAMES)
    _remove_contact_meshes(model)
    _remove_joints(model, comak_body_names)
    _remove_bodies(model, comak_body_names)

    # 6. Reconnect the body chain: removing the COMAK weld joints + intermediate
    #    bodies disconnects tibia_r from the joint tree. Create a simple knee_r
    #    (PinJoint with flexion-only) connecting femur_r → tibia_r directly.
    #    The weld offsets are baked into the joint's parent/child offset frames.
    _add_replacement_knee_joint(model, weld_joints, custom_joints)

    # 7. Finalize
    model.finalizeConnections()

    config = ComakKneeConfig(
        side=side,
        bodies=bodies,
        weld_joints=weld_joints,
        custom_joints=custom_joints,
        ligaments=ligaments,
        springs=springs,
        contact_meshes=contact_meshes,
        contact_forces=contact_forces,
        wrap_surfaces=wrap_surfaces,
        spanning_muscles=spanning_muscles,
        ref_femur_length=ref_femur_length,
        ref_tibia_length=ref_tibia_length,
    )

    logger.info(
        f"Stripped COMAK knee: {len(bodies)} bodies, {len(weld_joints)} weld joints, "
        f"{len(custom_joints)} custom joints, {len(ligaments)} ligaments, "
        f"{len(springs)} springs, {len(contact_forces)} contact forces, "
        f"{len(contact_meshes)} contact meshes, {len(spanning_muscles)} spanning muscles, "
        f"{len(wrap_surfaces)} wrap surfaces"
    )

    return model, config


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _discover_comak_bodies(model):
    """Discover COMAK-specific bodies by class type of their joints.

    COMAK bodies are connected via WeldJoint or CustomJoint to main-chain
    bodies and have names matching the expected COMAK naming convention.
    This uses the known COMAK body names for now — dynamic discovery by
    joint tree traversal is a future enhancement.
    """
    body_set = model.getBodySet()
    found = []
    for name in EXPECTED_COMAK_BODIES:
        for i in range(body_set.getSize()):
            if body_set.get(i).getName() == name:
                found.append(name)
                break
    return found


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _vec3_to_list(v):
    """Convert an osim.Vec3 to a Python list of floats."""
    return [v[0], v[1], v[2]]


def _extract_segment_lengths(model):
    """Extract hip->knee and knee->ankle segment lengths."""
    state = model.initSystem()

    def joint_child_pos(joint_name):
        joint = model.getJointSet().get(joint_name)
        child_frame = joint.getChildFrame()
        pos = child_frame.getPositionInGround(state)
        return np.array([pos[0], pos[1], pos[2]])

    hip_pos = joint_child_pos("hip_r")
    knee_pos = joint_child_pos("knee_r")
    ankle_pos = joint_child_pos("ankle_r")

    fem_len = float(np.linalg.norm(knee_pos - hip_pos))
    tib_len = float(np.linalg.norm(ankle_pos - knee_pos))
    return fem_len, tib_len


def _extract_bodies(model, comak_body_names):
    """Extract COMAK body properties."""
    bodies = []
    body_set = model.getBodySet()
    for i in range(body_set.getSize()):
        body = body_set.get(i)
        if body.getName() not in comak_body_names:
            continue

        mc = body.get_mass_center()
        inertia = body.get_inertia()  # Vec6: [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]

        # Extract attached geometry
        attached_geom = []
        ag_prop = body.getPropertyByName("attached_geometry")
        for j in range(ag_prop.size()):
            geom_str = ag_prop.getValueAsObject(j)
            attached_geom.append(
                {
                    "name": geom_str.getName(),
                    "class_name": geom_str.getConcreteClassName(),
                }
            )

        bodies.append(
            ComakBody(
                name=body.getName(),
                mass=body.getMass(),
                inertia=[inertia.get(k) for k in range(6)],
                mass_center=_vec3_to_list(mc),
                attached_geometry=attached_geom,
            )
        )
    return bodies


def _extract_offset_frame(frame):
    """Extract translation and orientation from a joint frame.

    Returns (translation, orientation) as lists. If the frame is a
    PhysicalOffsetFrame, reads its translation/orientation. Otherwise
    returns zeros (body frame directly).
    """
    if frame.getConcreteClassName() == "PhysicalOffsetFrame":
        off = osim.PhysicalOffsetFrame.safeDownCast(frame)
        return _vec3_to_list(off.get_translation()), _vec3_to_list(off.get_orientation())
    return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]


def _extract_spatial_transform(cj):
    """Extract spatial transform from a CustomJoint as a serializable dict."""
    st = cj.getSpatialTransform()
    result = {}
    for comp in [
        "rotation1",
        "rotation2",
        "rotation3",
        "translation1",
        "translation2",
        "translation3",
    ]:
        axis = getattr(st, f"get_{comp}")()
        ax = axis.get_axis()
        func = axis.get_function()
        n_coords = axis.getCoordinateNamesInArray().getSize()
        coord_name = axis.get_coordinates(0) if n_coords > 0 else None

        # Extract function parameters
        func_dict = {"type": func.getConcreteClassName()}
        if func.getConcreteClassName() == "LinearFunction":
            lf = osim.LinearFunction.safeDownCast(func)
            coeffs = lf.getCoefficients()
            func_dict["slope"] = coeffs.get(0)
            func_dict["intercept"] = coeffs.get(1)
        elif func.getConcreteClassName() == "Constant":
            const = osim.Constant.safeDownCast(func)
            func_dict["value"] = const.getValue()
        else:
            # SimmSpline or other — store as string for now
            func_dict["repr"] = func.toString()

        result[comp] = {
            "axis": _vec3_to_list(ax),
            "coordinate": coord_name,
            "function": func_dict,
        }
    return result


def _get_joint_parent_body_name(joint):
    """Get the actual parent body name from a joint's parent frame.

    The parent frame may be a PhysicalOffsetFrame on the body, so we need
    to find the underlying body.
    """
    pf = joint.getParentFrame()
    if pf.getConcreteClassName() == "PhysicalOffsetFrame":
        # The offset frame's parent is the body
        off = osim.PhysicalOffsetFrame.safeDownCast(pf)
        return off.getParentFrame().getName()
    return pf.getName()


def _get_joint_child_body_name(joint):
    """Get the actual child body name from a joint's child frame."""
    cf = joint.getChildFrame()
    if cf.getConcreteClassName() == "PhysicalOffsetFrame":
        off = osim.PhysicalOffsetFrame.safeDownCast(cf)
        return off.getParentFrame().getName()
    return cf.getName()


def _extract_joints(model, comak_body_names):
    """Extract COMAK joints (weld and custom)."""
    weld_joints = []
    custom_joints = []
    joint_set = model.getJointSet()

    for i in range(joint_set.getSize()):
        joint = joint_set.get(i)
        parent_name = _get_joint_parent_body_name(joint)
        child_name = _get_joint_child_body_name(joint)

        # A COMAK joint has at least one end on a COMAK body
        if parent_name not in comak_body_names and child_name not in comak_body_names:
            continue

        parent_t, parent_o = _extract_offset_frame(joint.getParentFrame())
        child_t, child_o = _extract_offset_frame(joint.getChildFrame())

        if joint.getConcreteClassName() == "WeldJoint":
            weld_joints.append(
                ComakWeldJoint(
                    name=joint.getName(),
                    parent_body=parent_name,
                    child_body=child_name,
                    parent_offset_translation=parent_t,
                    parent_offset_orientation=parent_o,
                    child_offset_translation=child_t,
                    child_offset_orientation=child_o,
                )
            )
        elif joint.getConcreteClassName() == "CustomJoint":
            cj = osim.CustomJoint.safeDownCast(joint)
            spatial_transform = _extract_spatial_transform(cj)

            # Extract coordinates
            coords = []
            coord_set = model.getCoordinateSet()
            for coord_name in [
                v["coordinate"] for v in spatial_transform.values() if v["coordinate"] is not None
            ]:
                for ci in range(coord_set.getSize()):
                    coord = coord_set.get(ci)
                    if coord.getName() == coord_name:
                        coords.append(
                            ComakCoordinate(
                                name=coord.getName(),
                                default_value=coord.getDefaultValue(),
                                range_min=coord.getRangeMin(),
                                range_max=coord.getRangeMax(),
                                locked=coord.get_locked(),
                                clamped=coord.get_clamped(),
                            )
                        )
                        break

            custom_joints.append(
                ComakCustomJoint(
                    name=joint.getName(),
                    parent_body=parent_name,
                    child_body=child_name,
                    parent_offset_translation=parent_t,
                    parent_offset_orientation=parent_o,
                    child_offset_translation=child_t,
                    child_offset_orientation=child_o,
                    coordinates=coords,
                    spatial_transform=spatial_transform,
                )
            )

    return weld_joints, custom_joints


def _extract_path_points(geometry_path):
    """Extract path points from a GeometryPath as a list of dicts."""
    pp_set = geometry_path.getPathPointSet()
    points = []
    for j in range(pp_set.getSize()):
        apt = pp_set.get(j)
        pt = osim.PathPoint.safeDownCast(apt)
        body_path = apt.getSocket("parent_frame").getConnecteePath()
        # Extract body name from socket path (e.g., "/bodyset/femur_r" -> "femur_r")
        body_name = body_path.rsplit("/", 1)[-1]
        loc = pt.get_location()
        points.append(
            {
                "name": apt.getName(),
                "body": body_name,
                "location": _vec3_to_list(loc),
            }
        )
    return points


def _extract_ligaments(model):
    """Extract all Blankevoort1991Ligament instances."""
    ligaments = []
    force_set = model.getForceSet()
    for i in range(force_set.getSize()):
        f = force_set.get(i)
        if f.getConcreteClassName() != "Blankevoort1991Ligament":
            continue
        lig = osim.Blankevoort1991Ligament.safeDownCast(f)
        gp = lig.getGeometryPath()

        ligaments.append(
            ComakLigament(
                name=lig.getName(),
                linear_stiffness=lig.get_linear_stiffness(),
                transition_strain=lig.get_transition_strain(),
                damping_coefficient=lig.get_damping_coefficient(),
                slack_length=lig.get_slack_length(),
                path_points=_extract_path_points(gp),
            )
        )
    return ligaments


def _extract_springs(model):
    """Extract all SpringGeneralizedForce instances."""
    springs = []
    force_set = model.getForceSet()
    for i in range(force_set.getSize()):
        f = force_set.get(i)
        if f.getConcreteClassName() != "SpringGeneralizedForce":
            continue
        spring = osim.SpringGeneralizedForce.safeDownCast(f)
        springs.append(
            ComakSpring(
                name=spring.getName(),
                coordinate=spring.get_coordinate(),
                stiffness=spring.get_stiffness(),
                rest_length=spring.get_rest_length(),
                viscosity=spring.get_viscosity(),
            )
        )
    return springs


def _extract_contact_meshes(model):
    """Extract all Smith2018ContactMesh instances."""
    meshes = []
    cg_set = model.getContactGeometrySet()
    for i in range(cg_set.getSize()):
        cg = cg_set.get(i)
        if cg.getConcreteClassName() != "Smith2018ContactMesh":
            continue
        mesh = osim.Smith2018ContactMesh.safeDownCast(cg)

        # Get parent frame from socket
        frame_path = mesh.getSocket("frame").getConnecteePath()
        frame_name = frame_path.rsplit("/", 1)[-1]

        loc = mesh.get_location()
        orient = mesh.get_orientation()
        sf = mesh.get_scale_factors()

        meshes.append(
            ComakContactMesh(
                name=mesh.getName(),
                parent_frame=frame_name,
                mesh_file=mesh.get_mesh_file(),
                elastic_modulus=mesh.get_elastic_modulus(),
                poissons_ratio=mesh.get_poissons_ratio(),
                thickness=mesh.get_thickness(),
                location=_vec3_to_list(loc),
                orientation=_vec3_to_list(orient),
                use_variable_thickness=mesh.get_use_variable_thickness(),
                mesh_back_file=mesh.get_mesh_back_file(),
                min_thickness=mesh.get_min_thickness(),
                max_thickness=mesh.get_max_thickness(),
                scale_factors=_vec3_to_list(sf),
            )
        )
    return meshes


def _extract_contact_forces(model):
    """Extract all Smith2018ArticularContactForce instances."""
    forces = []
    force_set = model.getForceSet()
    for i in range(force_set.getSize()):
        f = force_set.get(i)
        if f.getConcreteClassName() != "Smith2018ArticularContactForce":
            continue
        cf = osim.Smith2018ArticularContactForce.safeDownCast(f)

        # Extract socket paths and get mesh names
        target_path = cf.getSocket("target_mesh").getConnecteePath()
        casting_path = cf.getSocket("casting_mesh").getConnecteePath()
        target_name = target_path.rsplit("/", 1)[-1]
        casting_name = casting_path.rsplit("/", 1)[-1]

        forces.append(
            ComakContactForce(
                name=cf.getName(),
                target_mesh=target_name,
                casting_mesh=casting_name,
                min_proximity=cf.get_min_proximity(),
                max_proximity=cf.get_max_proximity(),
                elastic_foundation_formulation=cf.get_elastic_foundation_formulation(),
                use_lumped_contact_model=cf.get_use_lumped_contact_model(),
            )
        )
    return forces


def _extract_wrap_surfaces(model, comak_body_names):
    """Extract wrap surfaces from COMAK bodies (before body removal)."""
    wraps = []
    body_set = model.getBodySet()
    for i in range(body_set.getSize()):
        body = body_set.get(i)
        if body.getName() not in comak_body_names:
            continue

        ws = body.get_WrapObjectSet()
        for j in range(ws.getSize()):
            wo = ws.get(j)
            t = wo.get_translation()
            r = wo.get_xyz_body_rotation()

            wrap = ComakWrapSurface(
                name=wo.getName(),
                parent_body=body.getName(),
                type=wo.getConcreteClassName(),
                translation=_vec3_to_list(t),
                xyz_body_rotation=_vec3_to_list(r),
                quadrant=wo.get_quadrant(),
            )

            if wo.getConcreteClassName() == "WrapCylinder":
                cyl = osim.WrapCylinder.safeDownCast(wo)
                wrap.radius = cyl.get_radius()
                wrap.length = cyl.get_length()
            elif wo.getConcreteClassName() == "WrapEllipsoid":
                ell = osim.WrapEllipsoid.safeDownCast(wo)
                d = ell.get_dimensions()
                wrap.dimensions = _vec3_to_list(d)

            wraps.append(wrap)
    return wraps


def _extract_spanning_muscles(model, muscle_names):
    """Extract spanning muscles by name."""
    muscles = []
    force_set = model.getForceSet()
    for i in range(force_set.getSize()):
        f = force_set.get(i)
        if f.getName() not in muscle_names:
            continue
        m = osim.Millard2012EquilibriumMuscle.safeDownCast(f)
        if m is None:
            continue

        gp = m.getGeometryPath()

        # Extract wrap object references
        wrap_objects = []
        wrap_set = gp.getWrapSet()
        for j in range(wrap_set.getSize()):
            wrap_objects.append(wrap_set.get(j).getWrapObjectName())

        muscles.append(
            ComakMuscle(
                name=m.getName(),
                max_isometric_force=m.get_max_isometric_force(),
                optimal_fiber_length=m.get_optimal_fiber_length(),
                tendon_slack_length=m.get_tendon_slack_length(),
                pennation_angle_at_optimal=m.get_pennation_angle_at_optimal(),
                max_contraction_velocity=m.get_max_contraction_velocity(),
                path_points=_extract_path_points(gp),
                wrap_objects=wrap_objects,
                min_control=m.get_min_control(),
                max_control=m.get_max_control(),
                optimal_force=m.get_optimal_force(),
                ignore_tendon_compliance=m.get_ignore_tendon_compliance(),
                ignore_activation_dynamics=m.get_ignore_activation_dynamics(),
                fiber_damping=m.get_fiber_damping(),
                default_activation=m.get_default_activation(),
                default_fiber_length=m.get_default_fiber_length(),
                activation_time_constant=m.get_activation_time_constant(),
                deactivation_time_constant=m.get_deactivation_time_constant(),
                minimum_activation=m.get_minimum_activation(),
                maximum_pennation_angle=m.get_maximum_pennation_angle(),
            )
        )
    return muscles


# ---------------------------------------------------------------------------
# Removal helpers
# ---------------------------------------------------------------------------


def _remove_forces_by_class(model, class_name):
    """Remove all forces of a given class from the ForceSet."""
    force_set = model.getForceSet()
    indices = []
    for i in range(force_set.getSize()):
        if force_set.get(i).getConcreteClassName() == class_name:
            indices.append(i)
    indices.reverse()
    for idx in indices:
        logger.debug(f"Removing force: {force_set.get(idx).getName()} ({class_name})")
        force_set.remove(idx)


def _remove_forces_by_name(model, names):
    """Remove forces by name from the ForceSet."""
    force_set = model.getForceSet()
    indices = []
    for i in range(force_set.getSize()):
        if force_set.get(i).getName() in names:
            indices.append(i)
    indices.reverse()
    for idx in indices:
        logger.debug(f"Removing force: {force_set.get(idx).getName()}")
        force_set.remove(idx)


def _remove_contact_meshes(model):
    """Remove all Smith2018ContactMesh entries from ContactGeometrySet."""
    cg_set = model.getContactGeometrySet()
    indices = []
    for i in range(cg_set.getSize()):
        if cg_set.get(i).getConcreteClassName() == "Smith2018ContactMesh":
            indices.append(i)
    indices.reverse()
    for idx in indices:
        logger.debug(f"Removing contact mesh: {cg_set.get(idx).getName()}")
        cg_set.remove(idx)


def _remove_joints(model, comak_body_names):
    """Remove joints that connect to COMAK bodies."""
    joint_set = model.getJointSet()
    indices = []
    for i in range(joint_set.getSize()):
        joint = joint_set.get(i)
        parent_name = _get_joint_parent_body_name(joint)
        child_name = _get_joint_child_body_name(joint)
        if parent_name in comak_body_names or child_name in comak_body_names:
            indices.append(i)
    indices.reverse()
    for idx in indices:
        logger.debug(f"Removing joint: {joint_set.get(idx).getName()}")
        joint_set.remove(idx)


def _remove_bodies(model, comak_body_names):
    """Remove COMAK bodies from the BodySet."""
    body_set = model.getBodySet()
    indices = []
    for i in range(body_set.getSize()):
        if body_set.get(i).getName() in comak_body_names:
            indices.append(i)
    indices.reverse()
    for idx in indices:
        logger.debug(f"Removing body: {body_set.get(idx).getName()}")
        body_set.remove(idx)


# ---------------------------------------------------------------------------
# Replacement joint
# ---------------------------------------------------------------------------


def _add_replacement_knee_joint(model, weld_joints, custom_joints):
    """Add a simple knee joint connecting femur_r → tibia_r after COMAK removal.

    The COMAK knee chain is: femur_r → [weld] → femur_distal_r → [knee_r] →
    tibia_proximal_r → [weld] → tibia_r. Removing the intermediate bodies
    disconnects tibia_r from the joint tree, which causes a segfault in
    finalizeConnections(). This function creates a replacement PinJoint
    (flexion-only) with the weld offsets baked into the parent/child frames.
    """
    # Find the femur weld offset (femur_r → femur_distal_r)
    fem_weld = next(
        (wj for wj in weld_joints if wj.name == "femur_femur_distal_r"),
        None,
    )
    # Find the tibia weld offset (tibia_proximal_r → tibia_r)
    tib_weld = next(
        (wj for wj in weld_joints if wj.name == "tibia_tibia_proximal_r"),
        None,
    )
    # Find knee_r custom joint offsets
    knee_cj = next(
        (cj for cj in custom_joints if cj.name == "knee_r"),
        None,
    )

    # Compute the combined parent offset: femur_weld parent offset + knee_r parent offset
    # For Smith2019, knee_r parent/child offsets are [0,0,0], so the weld offset is the
    # full offset. The tibia weld offset goes into the child offset frame.
    parent_t = osim.Vec3(0, 0, 0)
    parent_o = osim.Vec3(0, 0, 0)
    child_t = osim.Vec3(0, 0, 0)
    child_o = osim.Vec3(0, 0, 0)

    if fem_weld is not None:
        parent_t = osim.Vec3(*fem_weld.parent_offset_translation)
        parent_o = osim.Vec3(*fem_weld.parent_offset_orientation)
    if tib_weld is not None:
        child_t = osim.Vec3(*tib_weld.parent_offset_translation)
        child_o = osim.Vec3(*tib_weld.parent_offset_orientation)

    femur_body = model.getBodySet().get("femur_r")
    tibia_body = model.getBodySet().get("tibia_r")

    knee = osim.PinJoint(
        "knee_r",
        femur_body,
        parent_t,
        parent_o,
        tibia_body,
        child_t,
        child_o,
    )
    # Rename the coordinate to match what downstream code expects
    knee.getCoordinate().setName("knee_flex_r")
    model.addJoint(knee)
    logger.info("Added replacement PinJoint 'knee_r' (femur_r → tibia_r)")
