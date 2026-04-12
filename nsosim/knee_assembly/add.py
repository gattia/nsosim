"""
Add COMAK knee components to an OpenSim model.

Takes a ComakKneeConfig (from strip or JSON) and inserts all COMAK knee
components into a model that has femur_r and tibia_r bodies.

Usage:
    import opensim as osim
    from nsosim.knee_assembly import add_comak_knee, ComakKneeConfig

    model = osim.Model("stripped_model.osim")
    config = ComakKneeConfig.from_json("knee_config.json")
    model = add_comak_knee(model, config)
"""

import logging

import opensim as osim

from .config import (
    ComakContactForce,
    ComakContactMesh,
    ComakCustomJoint,
    ComakKneeConfig,
    ComakLigament,
    ComakMuscle,
    ComakSpring,
    ComakWeldJoint,
    ComakWrapSurface,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Knee frame orientation presets
# ---------------------------------------------------------------------------

# Each target model may use a different body frame convention for femur_r.
# The COMAK knee (from Smith2019) assumes zero orientation on the femur weld
# joint. When adding to a model with a different convention, the weld joint
# parent offset orientation must be set to match the target model's knee
# frame orientation.
#
# These presets store the XYZ Euler angles (radians) of the knee joint's
# parent offset frame in each model. The values come from the original
# knee joint's parent_offset_orientation in the target model.
#
# To find the correct values for a new model, run the Phase 6A audit script
# and look at the knee joint's parent_offset_orientation.

KNEE_FRAME_PRESETS = {
    # Smith2019: zero orientation (COMAK native convention)
    "smith2019": [0.0, 0.0, 0.0],
    # RajagopalLaiUhlrich2023: rotation mapping Smith2019 body frame axes
    # to Rajagopal body frame axes. Derived from walker_knee_r orientation
    # which encodes the relationship between Rajagopal's femur_r frame and
    # the anatomical knee joint axes.
    #
    # Smith2019 body frame: X=anterior, Y=proximal, Z=lateral (flexion axis)
    # Rajagopal body frame: X≈anterior (slightly rotated), Y≈proximal, Z≈medial
    # The Z axes point in opposite directions (lat vs med), so the mapping
    # requires negating the walker flexion direction.
    #
    # Computed from walker_knee_r orientation [-1.64157, 1.44618, 1.5708]:
    #   R_walker = Rotation.from_euler('XYZ', walker_orient).as_matrix()
    #   R_weld[:, 0] = R_walker[:, 2]   # COMAK X (adduction) = Walker Z
    #   R_weld[:, 1] = R_walker[:, 1]   # COMAK Y (rotation)  = Walker Y
    #   R_weld[:, 2] = -R_walker[:, 0]  # COMAK Z (flexion)   = -Walker X
    #   euler = Rotation.from_matrix(R_weld).as_euler('XYZ')
    "rajagopal": [-0.070770, 0.0, 0.124616],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_comak_knee(model, knee_config, target_joint="knee_r", knee_frame_orientation=None):
    """Add COMAK knee components to a model.

    The model must have femur_r and tibia_r bodies. If a replacement knee joint
    (e.g., PinJoint from strip) exists at `target_joint`, it is removed first.

    Parameters
    ----------
    model : osim.Model
        The OpenSim model to modify. Modified in-place.
    knee_config : ComakKneeConfig
        Complete COMAK knee description (from strip or JSON).
    target_joint : str
        Name of the existing knee joint to remove before adding COMAK components.
        Set to None to skip removal (if no knee joint exists).
    knee_frame_orientation : list of float, str, or None
        XYZ Euler angles (radians) for the femur weld joint parent offset
        orientation. This rotates the COMAK knee assembly to match the target
        model's femur body frame convention.

        - None or [0, 0, 0]: no rotation (Smith2019 convention)
        - "rajagopal": use the RajagopalLaiUhlrich2023 preset
        - "smith2019": use the Smith2019 preset (identity, same as None)
        - [x, y, z]: explicit Euler angles in radians

        To find the correct values for a new model, inspect the original knee
        joint's parent_offset_orientation from the Phase 6A audit data.

    Returns
    -------
    model : osim.Model
        The model with COMAK knee added (same object, modified in-place).
    """
    # Resolve knee frame orientation
    if isinstance(knee_frame_orientation, str):
        if knee_frame_orientation not in KNEE_FRAME_PRESETS:
            raise ValueError(
                f"Unknown knee_frame_orientation preset: '{knee_frame_orientation}'. "
                f"Available: {list(KNEE_FRAME_PRESETS.keys())}"
            )
        knee_frame_orientation = KNEE_FRAME_PRESETS[knee_frame_orientation]

    # 1. Remove the replacement knee joint (if present)
    if target_joint is not None:
        _remove_joint_by_name(model, target_joint)

    # 2. Add bodies
    _add_bodies(model, knee_config.bodies)

    # 3. Add joints (weld first, then custom — order matters for joint tree)
    _add_weld_joints(model, knee_config.weld_joints, knee_frame_orientation)
    _add_custom_joints(model, knee_config.custom_joints)

    # 4. Add wrap surfaces to bodies (before muscles/ligaments that reference them)
    _add_wrap_surfaces(model, knee_config.wrap_surfaces)

    # 5. Add forces
    _add_ligaments(model, knee_config.ligaments)
    _add_springs(model, knee_config.springs)
    _add_spanning_muscles(model, knee_config.spanning_muscles)

    # 6. Add contact
    _add_contact_meshes(model, knee_config.contact_meshes)
    _add_contact_forces(model, knee_config.contact_forces)

    # 7. Finalize
    model.finalizeConnections()

    logger.info(
        f"Added COMAK knee: {len(knee_config.bodies)} bodies, "
        f"{len(knee_config.weld_joints)} weld joints, "
        f"{len(knee_config.custom_joints)} custom joints, "
        f"{len(knee_config.ligaments)} ligaments, "
        f"{len(knee_config.springs)} springs, "
        f"{len(knee_config.contact_forces)} contact forces, "
        f"{len(knee_config.contact_meshes)} contact meshes, "
        f"{len(knee_config.spanning_muscles)} spanning muscles, "
        f"{len(knee_config.wrap_surfaces)} wrap surfaces"
    )

    return model


# ---------------------------------------------------------------------------
# Removal helper
# ---------------------------------------------------------------------------


def _remove_joint_by_name(model, joint_name):
    """Remove a joint by name from the JointSet."""
    joint_set = model.getJointSet()
    for i in range(joint_set.getSize()):
        if joint_set.get(i).getName() == joint_name:
            logger.debug(f"Removing existing joint: {joint_name}")
            joint_set.remove(i)
            return
    logger.debug(f"Joint '{joint_name}' not found — skipping removal")


# ---------------------------------------------------------------------------
# Add helpers
# ---------------------------------------------------------------------------


def _add_bodies(model, bodies):
    """Add COMAK bodies to the model."""
    for b in bodies:
        # Inertia(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
        inertia = osim.Inertia(*b.inertia[:6])
        body = osim.Body(
            b.name,
            b.mass,
            osim.Vec3(*b.mass_center),
            inertia,
        )
        model.addBody(body)
        logger.debug(f"Added body: {b.name}")


def _add_weld_joints(model, weld_joints, knee_frame_orientation=None):
    """Add weld joints connecting main-chain bodies to COMAK bodies.

    Parameters
    ----------
    knee_frame_orientation : list of float or None
        If provided, overrides the parent offset orientation on the femur
        weld joint (the one connecting femur_r → femur_distal_r). This
        rotates the COMAK knee assembly to match the target model's femur
        body frame convention.
    """
    for wj in weld_joints:
        parent_body = model.getBodySet().get(wj.parent_body)
        child_body = model.getBodySet().get(wj.child_body)

        parent_orientation = list(wj.parent_offset_orientation)
        child_orientation = list(wj.child_offset_orientation)

        # Apply knee frame orientation to BOTH weld joints (femur and tibia).
        # Both intermediate bodies must be in the same rotated frame so the
        # COMAK knee_r joint has no built-in twist at default pose.
        #
        # The femur weld is femur_r → femur_distal_r:
        #   Rotation goes on PARENT offset (femur_r side) to rotate femur_distal_r.
        # The tibia weld is tibia_proximal_r → tibia_r:
        #   Rotation goes on CHILD offset (tibia_r side) to rotate tibia_proximal_r
        #   into the same frame (weld direction is reversed).
        if knee_frame_orientation is not None:
            if wj.parent_body == "femur_r" and wj.child_body == "femur_distal_r":
                parent_orientation = list(knee_frame_orientation)
                logger.info(
                    f"Applying knee_frame_orientation {knee_frame_orientation} "
                    f"to weld joint '{wj.name}' parent offset (femur_r side)"
                )
            elif wj.child_body == "tibia_r" or wj.parent_body == "tibia_proximal_r":
                child_orientation = list(knee_frame_orientation)
                logger.info(
                    f"Applying knee_frame_orientation {knee_frame_orientation} "
                    f"to weld joint '{wj.name}' child offset (tibia_r side)"
                )

        joint = osim.WeldJoint(
            wj.name,
            parent_body,
            osim.Vec3(*wj.parent_offset_translation),
            osim.Vec3(*parent_orientation),
            child_body,
            osim.Vec3(*wj.child_offset_translation),
            osim.Vec3(*child_orientation),
        )
        model.addJoint(joint)
        logger.debug(f"Added weld joint: {wj.name}")


def _build_spatial_transform(st_dict):
    """Build a SpatialTransform from a serialized dict.

    CRITICAL: All 6 axes must have coordinates assigned to avoid the
    collinear-axis segfault. See plan gotchas for details.
    """
    st = osim.SpatialTransform()
    for comp_name in [
        "rotation1",
        "rotation2",
        "rotation3",
        "translation1",
        "translation2",
        "translation3",
    ]:
        axis_data = st_dict[comp_name]
        axis = getattr(st, f"get_{comp_name}")()
        axis.set_axis(osim.Vec3(*axis_data["axis"]))

        if axis_data["coordinate"] is not None:
            axis.set_coordinates(0, axis_data["coordinate"])

        func_data = axis_data["function"]
        if func_data["type"] == "LinearFunction":
            axis.set_function(osim.LinearFunction(func_data["slope"], func_data["intercept"]))
        elif func_data["type"] == "Constant":
            axis.set_function(osim.Constant(func_data["value"]))
        else:
            raise ValueError(f"Unsupported function type: {func_data['type']}")

    return st


def _add_custom_joints(model, custom_joints):
    """Add custom joints with spatial transforms and coordinates."""
    for cj in custom_joints:
        parent_body = model.getBodySet().get(cj.parent_body)
        child_body = model.getBodySet().get(cj.child_body)

        st = _build_spatial_transform(cj.spatial_transform)

        joint = osim.CustomJoint(
            cj.name,
            parent_body,
            osim.Vec3(*cj.parent_offset_translation),
            osim.Vec3(*cj.parent_offset_orientation),
            child_body,
            osim.Vec3(*cj.child_offset_translation),
            osim.Vec3(*cj.child_offset_orientation),
            st,
        )
        model.addJoint(joint)

        # Set coordinate properties
        coord_set = model.getCoordinateSet()
        for cc in cj.coordinates:
            for i in range(coord_set.getSize()):
                coord = coord_set.get(i)
                if coord.getName() == cc.name:
                    coord.setDefaultValue(cc.default_value)
                    coord.setRangeMin(cc.range_min)
                    coord.setRangeMax(cc.range_max)
                    coord.set_locked(cc.locked)
                    coord.set_clamped(cc.clamped)
                    break

        logger.debug(f"Added custom joint: {cj.name} ({len(cj.coordinates)} coords)")


def _add_ligaments(model, ligaments):
    """Add Blankevoort1991Ligaments with path points."""
    for lig_data in ligaments:
        lig = osim.Blankevoort1991Ligament()
        lig.setName(lig_data.name)
        lig.set_linear_stiffness(lig_data.linear_stiffness)
        lig.set_transition_strain(lig_data.transition_strain)
        lig.set_damping_coefficient(lig_data.damping_coefficient)
        lig.set_slack_length(lig_data.slack_length)

        gp = lig.updGeometryPath()
        for pp in lig_data.path_points:
            body = model.getBodySet().get(pp["body"])
            gp.appendNewPathPoint(pp["name"], body, osim.Vec3(*pp["location"]))

        model.addForce(lig)

    logger.debug(f"Added {len(ligaments)} ligaments")


def _add_springs(model, springs):
    """Add SpringGeneralizedForces."""
    for sp in springs:
        # NOTE: SpringGeneralizedForce(str) treats the arg as the coordinate name,
        # not the force name. Use default constructor + setName().
        spring = osim.SpringGeneralizedForce()
        spring.setName(sp.name)
        spring.set_coordinate(sp.coordinate)
        spring.set_stiffness(sp.stiffness)
        spring.set_rest_length(sp.rest_length)
        spring.set_viscosity(sp.viscosity)
        model.addForce(spring)

    logger.debug(f"Added {len(springs)} springs")


def _add_contact_meshes(model, contact_meshes):
    """Add Smith2018ContactMesh entries to ContactGeometrySet."""
    for cm in contact_meshes:
        mesh = osim.Smith2018ContactMesh()
        mesh.setName(cm.name)
        mesh.set_mesh_file(cm.mesh_file)
        mesh.set_elastic_modulus(cm.elastic_modulus)
        mesh.set_poissons_ratio(cm.poissons_ratio)
        mesh.set_thickness(cm.thickness)
        mesh.set_location(osim.Vec3(*cm.location))
        mesh.set_orientation(osim.Vec3(*cm.orientation))
        mesh.set_use_variable_thickness(cm.use_variable_thickness)
        mesh.set_mesh_back_file(cm.mesh_back_file)
        mesh.set_min_thickness(cm.min_thickness)
        mesh.set_max_thickness(cm.max_thickness)
        mesh.set_scale_factors(osim.Vec3(*cm.scale_factors))

        # Set socket to parent frame
        mesh.updSocket("frame").setConnecteePath(f"/bodyset/{cm.parent_frame}")
        # scale_frame defaults to /ground in Smith2019
        mesh.updSocket("scale_frame").setConnecteePath("/ground")

        model.addContactGeometry(mesh)

    logger.debug(f"Added {len(contact_meshes)} contact meshes")


def _add_contact_forces(model, contact_forces):
    """Add Smith2018ArticularContactForces."""
    for cf in contact_forces:
        force = osim.Smith2018ArticularContactForce()
        force.setName(cf.name)
        force.set_min_proximity(cf.min_proximity)
        force.set_max_proximity(cf.max_proximity)
        force.set_elastic_foundation_formulation(cf.elastic_foundation_formulation)
        force.set_use_lumped_contact_model(cf.use_lumped_contact_model)

        # Set socket paths to contact meshes
        force.updSocket("target_mesh").setConnecteePath(f"/contactgeometryset/{cf.target_mesh}")
        force.updSocket("casting_mesh").setConnecteePath(f"/contactgeometryset/{cf.casting_mesh}")

        model.addForce(force)

    logger.debug(f"Added {len(contact_forces)} contact forces")


def _add_wrap_surfaces(model, wrap_surfaces):
    """Add wrap surfaces (cylinders and ellipsoids) to their parent bodies."""
    for ws in wrap_surfaces:
        body = model.getBodySet().get(ws.parent_body)

        if ws.type == "WrapCylinder":
            wrap = osim.WrapCylinder()
            wrap.setName(ws.name)
            wrap.set_radius(ws.radius)
            wrap.set_length(ws.length)
            wrap.set_quadrant(ws.quadrant)
            wrap.set_translation(osim.Vec3(*ws.translation))
            wrap.set_xyz_body_rotation(osim.Vec3(*ws.xyz_body_rotation))
            body.addWrapObject(wrap)

        elif ws.type == "WrapEllipsoid":
            wrap = osim.WrapEllipsoid()
            wrap.setName(ws.name)
            wrap.set_dimensions(osim.Vec3(*ws.dimensions))
            wrap.set_quadrant(ws.quadrant)
            wrap.set_translation(osim.Vec3(*ws.translation))
            wrap.set_xyz_body_rotation(osim.Vec3(*ws.xyz_body_rotation))
            body.addWrapObject(wrap)

        else:
            raise ValueError(f"Unknown wrap surface type: {ws.type}")

    logger.debug(f"Added {len(wrap_surfaces)} wrap surfaces")


def _add_spanning_muscles(model, muscles):
    """Add Millard2012EquilibriumMuscle spanning muscles with path points and wraps."""
    for m in muscles:
        muscle = osim.Millard2012EquilibriumMuscle(
            m.name,
            m.max_isometric_force,
            m.optimal_fiber_length,
            m.tendon_slack_length,
            m.pennation_angle_at_optimal,
        )
        muscle.set_max_contraction_velocity(m.max_contraction_velocity)
        muscle.set_min_control(m.min_control)
        muscle.set_max_control(m.max_control)
        muscle.set_optimal_force(m.optimal_force)
        muscle.set_ignore_tendon_compliance(m.ignore_tendon_compliance)
        muscle.set_ignore_activation_dynamics(m.ignore_activation_dynamics)
        muscle.set_fiber_damping(m.fiber_damping)
        muscle.set_default_activation(m.default_activation)
        muscle.set_default_fiber_length(m.default_fiber_length)
        muscle.set_activation_time_constant(m.activation_time_constant)
        muscle.set_deactivation_time_constant(m.deactivation_time_constant)
        muscle.set_minimum_activation(m.minimum_activation)
        muscle.set_maximum_pennation_angle(m.maximum_pennation_angle)

        # Add path points
        gp = muscle.updGeometryPath()
        for pp in m.path_points:
            body = model.getBodySet().get(pp["body"])
            gp.appendNewPathPoint(pp["name"], body, osim.Vec3(*pp["location"]))

        # Add wrap references
        for wrap_name in m.wrap_objects:
            # Find the wrap object in the model
            wrap_obj = _find_wrap_object(model, wrap_name)
            if wrap_obj is not None:
                gp.addPathWrap(wrap_obj)
            else:
                logger.warning(f"Wrap object '{wrap_name}' not found for muscle '{m.name}'")

        model.addForce(muscle)

    logger.debug(f"Added {len(muscles)} spanning muscles")


def _find_wrap_object(model, wrap_name):
    """Find a wrap object by name across all bodies."""
    body_set = model.getBodySet()
    for i in range(body_set.getSize()):
        body = body_set.get(i)
        ws = body.get_WrapObjectSet()
        for j in range(ws.getSize()):
            if ws.get(j).getName() == wrap_name:
                return ws.get(j)
    return None
