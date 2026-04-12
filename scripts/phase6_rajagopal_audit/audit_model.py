"""Phase 6A: Audit an OpenSim model and extract comprehensive structured data.

Produces a JSON file containing every body, joint, coordinate, muscle,
constraint, and wrap surface in the model, plus per-body segment lengths.

Usage:
    conda run -n comak python scripts/phase6_rajagopal_audit/audit_model.py <model_path> <output_path>

Examples:
    # Audit Smith2019
    conda run -n comak python scripts/phase6_rajagopal_audit/audit_model.py \
        tests/fixtures/osim_models/full_body_healthy_knee.osim \
        scripts/phase6_rajagopal_audit/smith2019_audit.json

    # Audit Rajagopal
    conda run -n comak python scripts/phase6_rajagopal_audit/audit_model.py \
        tests/fixtures/osim_models/rajagopal/RajagopalLaiUhlrich2023.osim \
        scripts/phase6_rajagopal_audit/rajagopal_audit.json
"""

import argparse
import json
import sys

import numpy as np
import opensim as osim


def vec3_to_list(v):
    """Convert opensim.Vec3 to Python list."""
    return [v.get(i) for i in range(3)]


def extract_bodies(model):
    """Extract all bodies with mass, inertia, mass center."""
    bodies = []
    body_set = model.getBodySet()
    for i in range(body_set.getSize()):
        body = body_set.get(i)
        inertia = body.get_inertia()
        bodies.append(
            {
                "name": body.getName(),
                "mass": body.getMass(),
                "mass_center": vec3_to_list(body.getMassCenter()),
                "inertia": [inertia.get(j) for j in range(6)],
            }
        )
    return bodies


def extract_offset_frame_data(frame):
    """Extract translation/orientation from a joint frame, handling offset frames."""
    pof = osim.PhysicalOffsetFrame.safeDownCast(frame)
    if pof is not None:
        return {
            "translation": vec3_to_list(pof.get_translation()),
            "orientation": vec3_to_list(pof.get_orientation()),
            "frame_name": pof.getName(),
            "parent_frame_path": pof.getParentFrame().getName(),
        }
    return {
        "translation": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0],
        "frame_name": frame.getName(),
        "parent_frame_path": frame.getName(),
    }


def extract_spatial_transform(joint):
    """Extract spatial transform from a CustomJoint."""
    cj = osim.CustomJoint.safeDownCast(joint)
    if cj is None:
        return None

    st = cj.getSpatialTransform()
    transform = {}
    for comp_name in [
        "rotation1",
        "rotation2",
        "rotation3",
        "translation1",
        "translation2",
        "translation3",
    ]:
        axis = getattr(st, f"get_{comp_name}")()
        axis_vec = [axis.get_axis().get(j) for j in range(3)]

        # Get coordinate name if any
        coord_names = axis.getCoordinateNamesInArray()
        coord_name = None
        if coord_names.getSize() > 0:
            coord_name = coord_names.get(0)

        # Get function info
        func = axis.get_function()
        func_class = func.getConcreteClassName()
        func_info = {"type": func_class}

        if func_class == "LinearFunction":
            lf = osim.LinearFunction.safeDownCast(func)
            coeffs = lf.getCoefficients()
            func_info["slope"] = coeffs.get(0)
            func_info["intercept"] = coeffs.get(1)
        elif func_class == "Constant":
            cf = osim.Constant.safeDownCast(func)
            func_info["value"] = cf.getValue()
        elif func_class == "SimmSpline":
            ss = osim.SimmSpline.safeDownCast(func)
            n = ss.getSize()
            func_info["n_points"] = n
            func_info["x"] = [ss.getX(j) for j in range(n)]
            func_info["y"] = [ss.getY(j) for j in range(n)]
        elif func_class == "MultiplierFunction":
            func_info["note"] = "MultiplierFunction - complex, storing class name only"
        elif func_class == "PolynomialFunction":
            pf = osim.PolynomialFunction.safeDownCast(func)
            coeffs = pf.getCoefficients()
            try:
                n = coeffs.getSize()
            except AttributeError:
                n = coeffs.size()
            func_info["coefficients"] = [coeffs.get(j) for j in range(n)]

        transform[comp_name] = {
            "axis": axis_vec,
            "coordinate": coord_name,
            "function": func_info,
        }

    return transform


def extract_joints(model):
    """Extract all joints with types, parent/child, offsets, and spatial transforms."""
    joints = []
    joint_set = model.getJointSet()
    for i in range(joint_set.getSize()):
        joint = joint_set.get(i)
        joint_data = {
            "name": joint.getName(),
            "type": joint.getConcreteClassName(),
            "parent_body": joint.getParentFrame().findBaseFrame().getName(),
            "child_body": joint.getChildFrame().findBaseFrame().getName(),
            "parent_offset": extract_offset_frame_data(joint.getParentFrame()),
            "child_offset": extract_offset_frame_data(joint.getChildFrame()),
        }

        # Extract spatial transform for CustomJoints
        spatial_transform = extract_spatial_transform(joint)
        if spatial_transform is not None:
            joint_data["spatial_transform"] = spatial_transform

        joints.append(joint_data)
    return joints


def extract_coordinates(model):
    """Extract all coordinates with properties."""
    coords = []
    coord_set = model.getCoordinateSet()
    for i in range(coord_set.getSize()):
        coord = coord_set.get(i)
        coords.append(
            {
                "name": coord.getName(),
                "default_value": coord.getDefaultValue(),
                "range_min": coord.getRangeMin(),
                "range_max": coord.getRangeMax(),
                "locked": coord.getDefaultLocked(),
                "clamped": coord.getDefaultClamped(),
                "motion_type": coord.getMotionType(),
            }
        )
    return coords


def extract_path_points(force):
    """Extract path points from a force with a GeometryPath."""
    points = []
    try:
        gp = force.getGeometryPath()
    except Exception:
        try:
            gp = force.get_GeometryPath()
        except Exception:
            return points

    pp_set = gp.getPathPointSet()
    for j in range(pp_set.getSize()):
        pt = pp_set.get(j)
        pt_data = {
            "name": pt.getName(),
            "type": pt.getConcreteClassName(),
            "body": pt.getParentFrame().findBaseFrame().getName(),
        }
        # Try to get location
        pp = osim.PathPoint.safeDownCast(pt)
        if pp is not None:
            pt_data["location"] = vec3_to_list(pp.get_location())
        else:
            pt_data["location"] = None
        points.append(pt_data)
    return points


def extract_wrap_set(force):
    """Extract wrap object references from a force's geometry path."""
    wraps = []
    try:
        gp = force.getGeometryPath()
    except Exception:
        try:
            gp = force.get_GeometryPath()
        except Exception:
            return wraps

    wrap_set = gp.getWrapSet()
    for j in range(wrap_set.getSize()):
        pw = wrap_set.get(j)
        wraps.append(pw.getWrapObjectName())
    return wraps


def extract_muscles(model):
    """Extract all muscles with properties and path points."""
    muscles = []
    force_set = model.getForceSet()
    for i in range(force_set.getSize()):
        f = force_set.get(i)
        class_name = f.getConcreteClassName()

        if class_name not in (
            "Millard2012EquilibriumMuscle",
            "Thelen2003Muscle",
            "DeGrooteFregly2016Muscle",
        ):
            continue

        muscle = osim.Muscle.safeDownCast(f)
        if muscle is None:
            continue

        muscles.append(
            {
                "name": muscle.getName(),
                "type": class_name,
                "max_isometric_force": muscle.getMaxIsometricForce(),
                "optimal_fiber_length": muscle.getOptimalFiberLength(),
                "tendon_slack_length": muscle.getTendonSlackLength(),
                "pennation_angle": muscle.getPennationAngleAtOptimalFiberLength(),
                "path_points": extract_path_points(f),
                "wrap_objects": extract_wrap_set(f),
            }
        )
    return muscles


def extract_ligaments(model):
    """Extract all Blankevoort1991Ligament instances."""
    ligaments = []
    force_set = model.getForceSet()
    for i in range(force_set.getSize()):
        f = force_set.get(i)
        if f.getConcreteClassName() != "Blankevoort1991Ligament":
            continue

        lig = osim.Blankevoort1991Ligament.safeDownCast(f)
        ligaments.append(
            {
                "name": lig.getName(),
                "linear_stiffness": lig.get_linear_stiffness(),
                "transition_strain": lig.get_transition_strain(),
                "damping_coefficient": lig.get_damping_coefficient(),
                "slack_length": lig.get_slack_length(),
                "path_points": extract_path_points(f),
            }
        )
    return ligaments


def extract_springs(model):
    """Extract all SpringGeneralizedForce instances."""
    springs = []
    force_set = model.getForceSet()
    for i in range(force_set.getSize()):
        f = force_set.get(i)
        if f.getConcreteClassName() != "SpringGeneralizedForce":
            continue

        spring = osim.SpringGeneralizedForce.safeDownCast(f)
        springs.append(
            {
                "name": spring.getName(),
                "coordinate": spring.get_coordinate(),
                "stiffness": spring.get_stiffness(),
                "rest_length": spring.get_rest_length(),
                "viscosity": spring.get_viscosity(),
            }
        )
    return springs


def extract_contact_geometry(model):
    """Extract all contact geometry entries."""
    contacts = []
    cg_set = model.getContactGeometrySet()
    for i in range(cg_set.getSize()):
        cg = cg_set.get(i)
        contacts.append(
            {
                "name": cg.getName(),
                "type": cg.getConcreteClassName(),
                "frame": cg.getFrame().getName(),
            }
        )
    return contacts


def extract_contact_forces(model):
    """Extract contact force entries from ForceSet."""
    forces = []
    force_set = model.getForceSet()
    for i in range(force_set.getSize()):
        f = force_set.get(i)
        class_name = f.getConcreteClassName()
        if "Contact" not in class_name and "contact" not in class_name:
            continue
        # Skip muscles/ligaments/springs already extracted
        if class_name in (
            "Blankevoort1991Ligament",
            "SpringGeneralizedForce",
            "Millard2012EquilibriumMuscle",
            "Thelen2003Muscle",
            "DeGrooteFregly2016Muscle",
        ):
            continue

        forces.append(
            {
                "name": f.getName(),
                "type": class_name,
            }
        )
    return forces


def extract_constraints(model):
    """Extract all constraints."""
    constraints = []
    cs = model.getConstraintSet()
    for i in range(cs.getSize()):
        c = cs.get(i)
        constraint_data = {
            "name": c.getName(),
            "type": c.getConcreteClassName(),
        }

        # For CoordinateCouplerConstraint, extract referenced coordinates
        ccc = osim.CoordinateCouplerConstraint.safeDownCast(c)
        if ccc is not None:
            ind_coords = ccc.getIndependentCoordinateNames()
            constraint_data["independent_coordinates"] = [
                ind_coords.get(j) for j in range(ind_coords.getSize())
            ]
            constraint_data["dependent_coordinate"] = ccc.getDependentCoordinateName()

        constraints.append(constraint_data)
    return constraints


def extract_wrap_surfaces(model):
    """Extract all wrap surfaces from all bodies."""
    wraps = []
    body_set = model.getBodySet()
    for i in range(body_set.getSize()):
        body = body_set.get(i)
        ws_prop = body.getPropertyByName("WrapObjectSet")
        ws_set = osim.WrapObjectSet.safeDownCast(ws_prop.getValueAsObject())
        if ws_set is None:
            continue
        for j in range(ws_set.getSize()):
            wo = ws_set.get(j)
            wrap_data = {
                "name": wo.getName(),
                "type": wo.getConcreteClassName(),
                "parent_body": body.getName(),
                "translation": vec3_to_list(wo.get_translation()),
                "xyz_body_rotation": vec3_to_list(wo.get_xyz_body_rotation()),
                "quadrant": wo.get_quadrant(),
            }

            # Type-specific dimensions
            if wo.getConcreteClassName() == "WrapCylinder":
                cyl = osim.WrapCylinder.safeDownCast(wo)
                wrap_data["radius"] = cyl.get_radius()
                wrap_data["length"] = cyl.get_length()
            elif wo.getConcreteClassName() == "WrapEllipsoid":
                ell = osim.WrapEllipsoid.safeDownCast(wo)
                wrap_data["dimensions"] = vec3_to_list(ell.get_dimensions())
            elif wo.getConcreteClassName() == "WrapSphere":
                sph = osim.WrapSphere.safeDownCast(wo)
                wrap_data["radius"] = sph.get_radius()
            elif wo.getConcreteClassName() == "WrapTorus":
                tor = osim.WrapTorus.safeDownCast(wo)
                wrap_data["inner_radius"] = tor.get_inner_radius()
                wrap_data["outer_radius"] = tor.get_outer_radius()

            wraps.append(wrap_data)
    return wraps


def compute_segment_lengths(model, joints):
    """Compute per-body segment lengths from joint-to-joint distances.

    For each body, find the joint connecting it to its parent and the joint(s)
    connecting it to its children. The segment length is the distance between
    the parent joint's child offset and each child joint's parent offset,
    measured in the body's local frame.
    """
    # Build parent→children map from joints
    body_children = {}  # body_name → list of (joint_name, child_body, parent_offset_in_body)
    body_parent = {}  # body_name → (joint_name, parent_body, child_offset_in_body)

    for j in joints:
        child_body = j["child_body"]
        parent_body = j["parent_body"]

        # The child offset translation is where the joint sits in the child body's frame
        child_offset_t = j["child_offset"]["translation"]
        # The parent offset translation is where the joint sits in the parent body's frame
        parent_offset_t = j["parent_offset"]["translation"]

        body_parent[child_body] = {
            "joint": j["name"],
            "parent_body": parent_body,
            "offset_in_child": child_offset_t,
        }

        if parent_body not in body_children:
            body_children[parent_body] = []
        body_children[parent_body].append(
            {
                "joint": j["name"],
                "child_body": child_body,
                "offset_in_parent": parent_offset_t,
            }
        )

    # Compute segment lengths
    segments = {}
    for j in joints:
        child_body = j["child_body"]
        if child_body == "ground":
            continue

        # Where the parent joint enters this body (child offset of parent joint)
        proximal = np.array(j["child_offset"]["translation"])

        # Where child joints leave this body (parent offset of child joints)
        if child_body in body_children:
            for child_info in body_children[child_body]:
                distal = np.array(child_info["offset_in_parent"])
                diff = distal - proximal
                length = float(np.linalg.norm(diff))

                key = f"{child_body}_to_{child_info['child_body']}"
                segments[key] = {
                    "body": child_body,
                    "proximal_joint": j["name"],
                    "distal_joint": child_info["joint"],
                    "proximal_point": proximal.tolist(),
                    "distal_point": distal.tolist(),
                    "diff_xyz": diff.tolist(),
                    "length": length,
                }

    return segments


def extract_other_forces(model):
    """Extract any forces not captured by muscles/ligaments/springs/contacts."""
    other = []
    force_set = model.getForceSet()
    known_types = {
        "Millard2012EquilibriumMuscle",
        "Thelen2003Muscle",
        "DeGrooteFregly2016Muscle",
        "Blankevoort1991Ligament",
        "SpringGeneralizedForce",
        "Smith2018ArticularContactForce",
    }
    for i in range(force_set.getSize()):
        f = force_set.get(i)
        class_name = f.getConcreteClassName()
        if class_name not in known_types:
            other.append(
                {
                    "name": f.getName(),
                    "type": class_name,
                }
            )
    return other


def audit_model(model_path):
    """Run full audit on an OpenSim model, return structured dict."""
    print(f"Loading model: {model_path}")
    model = osim.Model(model_path)
    model.initSystem()
    print(f"Model loaded: {model.getName()}")

    print("Extracting bodies...")
    bodies = extract_bodies(model)

    print("Extracting joints...")
    joints = extract_joints(model)

    print("Extracting coordinates...")
    coordinates = extract_coordinates(model)

    print("Extracting muscles...")
    muscles = extract_muscles(model)

    print("Extracting ligaments...")
    ligaments = extract_ligaments(model)

    print("Extracting springs...")
    springs = extract_springs(model)

    print("Extracting contact geometry...")
    contact_geometry = extract_contact_geometry(model)

    print("Extracting contact forces...")
    contact_forces = extract_contact_forces(model)

    print("Extracting constraints...")
    constraints = extract_constraints(model)

    print("Extracting wrap surfaces...")
    wrap_surfaces = extract_wrap_surfaces(model)

    print("Extracting other forces...")
    other_forces = extract_other_forces(model)

    print("Computing segment lengths...")
    segment_lengths = compute_segment_lengths(model, joints)

    result = {
        "model_name": model.getName(),
        "model_path": model_path,
        "summary": {
            "n_bodies": len(bodies),
            "n_joints": len(joints),
            "n_coordinates": len(coordinates),
            "n_muscles": len(muscles),
            "n_ligaments": len(ligaments),
            "n_springs": len(springs),
            "n_contact_geometry": len(contact_geometry),
            "n_contact_forces": len(contact_forces),
            "n_constraints": len(constraints),
            "n_wrap_surfaces": len(wrap_surfaces),
            "n_other_forces": len(other_forces),
        },
        "bodies": bodies,
        "joints": joints,
        "coordinates": coordinates,
        "muscles": muscles,
        "ligaments": ligaments,
        "springs": springs,
        "contact_geometry": contact_geometry,
        "contact_forces": contact_forces,
        "constraints": constraints,
        "wrap_surfaces": wrap_surfaces,
        "other_forces": other_forces,
        "segment_lengths": segment_lengths,
    }

    print(f"\nSummary:")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Audit an OpenSim model")
    parser.add_argument("model_path", help="Path to .osim file")
    parser.add_argument("output_path", help="Path for output JSON")
    args = parser.parse_args()

    result = audit_model(args.model_path)

    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nAudit written to: {args.output_path}")


if __name__ == "__main__":
    main()
