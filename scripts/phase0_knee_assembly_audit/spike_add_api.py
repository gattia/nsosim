"""
Spike: Verify OpenSim property setter API for COMAK knee assembly.

Phase 0E confirmed constructors exist. This script tests the SETTER API —
i.e., can we programmatically set individual properties on each component type
and have initSystem() succeed? This resolves the biggest unknown before writing
production code.

Run with: conda run -n comak python scripts/phase0_knee_assembly_audit/spike_add_api.py

Tests:
  1. Blankevoort1991Ligament — set properties + add path points + addForce
  2. SpringGeneralizedForce — set properties + addForce
  3. Smith2018ContactMesh — set properties + add to ContactGeometrySet
  4. Smith2018ArticularContactForce — set properties + addForce
  5. Millard2012EquilibriumMuscle — set properties + add path points + wrap + addForce
  6. WrapCylinder / WrapEllipsoid — set properties + attach to body
  7. CustomJoint with SpatialTransform — set LinearFunction on axes
  8. WeldJoint with offset frames — set translation/orientation
  9. Full mini-model: body + weld + custom joint + ligament + spring + contact + muscle

For each test, we try Pattern A (direct setters like lig.set_linear_stiffness(val))
first. If that fails, we try Pattern B (lig.setPropertyByName(...)).

Results are written to spike_add_api_results.txt in this directory.
"""

import sys
import traceback
from io import StringIO
from pathlib import Path

import opensim as osim

MODEL_PATH = (
    "/dataNAS/people/aagatti/projects/comak_gait_simulation/"
    "COMAK_SIMULATION_REQUIREMENTS/data/reference_data/comak_models/"
    "current/full_body_healthy_knee.osim"
)

OUTPUT_PATH = Path(__file__).parent / "spike_add_api_results.txt"


def tee_print(msg, buf):
    print(msg)
    buf.write(msg + "\n")


def test_ligament_setters(buf):
    """Test Blankevoort1991Ligament property setters and path point addition."""
    tee_print("\n" + "=" * 70, buf)
    tee_print("TEST 1: Blankevoort1991Ligament setters + path points", buf)
    tee_print("=" * 70, buf)

    # Build a minimal model with two bodies
    model = osim.Model()
    model.setName("spike_ligament")

    body_a = osim.Body("body_a", 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(body_a)
    joint_a = osim.WeldJoint("weld_a", model.getGround(), body_a)
    model.addJoint(joint_a)

    body_b = osim.Body("body_b", 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(body_b)
    joint_b = osim.WeldJoint("weld_b", model.getGround(), body_b)
    model.addJoint(joint_b)

    lig = osim.Blankevoort1991Ligament()
    lig.setName("test_MCLd1")

    # Pattern A: direct setters
    setter_results = {}
    for prop_name, value in [
        ("linear_stiffness", 5000.0),
        ("transition_strain", 0.06),
        ("damping_coefficient", 0.003),
        ("slack_length", 0.05),
    ]:
        setter = f"set_{prop_name}"
        try:
            getattr(lig, setter)(value)
            setter_results[prop_name] = f"Pattern A ({setter}): OK"
        except AttributeError:
            # Try Pattern B
            try:
                prop = lig.updPropertyByName(prop_name)
                osim.PropertyHelper.setValueDouble(value, prop)
                setter_results[prop_name] = f"Pattern B (updPropertyByName): OK"
            except Exception as e:
                setter_results[prop_name] = f"FAILED: {e}"

    for prop_name, result in setter_results.items():
        tee_print(f"  {prop_name}: {result}", buf)

    # Test path point addition — try updGeometryPath vs getGeometryPath
    tee_print("\n  Path point addition:", buf)
    gp = None
    for method_name in ["updGeometryPath", "getGeometryPath", "get_GeometryPath"]:
        try:
            gp = getattr(lig, method_name)()
            tee_print(f"    {method_name}(): OK (type={type(gp).__name__})", buf)
            break
        except Exception as e:
            tee_print(f"    {method_name}(): FAILED ({e})", buf)

    if gp is not None:
        try:
            gp.appendNewPathPoint(
                "test_MCLd1-P1", body_a, osim.Vec3(0.01, -0.02, 0.03)
            )
            gp.appendNewPathPoint(
                "test_MCLd1-P2", body_b, osim.Vec3(-0.01, 0.02, -0.03)
            )
            tee_print("    appendNewPathPoint(): OK (2 points added)", buf)
        except Exception as e:
            tee_print(f"    appendNewPathPoint(): FAILED ({e})", buf)

    model.addForce(lig)
    tee_print("  addForce(lig): OK", buf)

    # Finalize and test
    try:
        model.finalizeConnections()
        state = model.initSystem()
        tee_print("  finalizeConnections() + initSystem(): OK", buf)
    except Exception as e:
        tee_print(f"  finalizeConnections() + initSystem(): FAILED ({e})", buf)

    # Verify values round-trip
    tee_print("\n  Value verification (read back after set):", buf)
    for prop_name, expected in [
        ("linear_stiffness", 5000.0),
        ("transition_strain", 0.06),
        ("damping_coefficient", 0.003),
        ("slack_length", 0.05),
    ]:
        getter = f"get_{prop_name}"
        try:
            actual = getattr(lig, getter)()
            match = "MATCH" if abs(actual - expected) < 1e-10 else f"MISMATCH (got {actual})"
            tee_print(f"    {prop_name}: {actual} — {match}", buf)
        except AttributeError:
            try:
                prop = lig.getPropertyByName(prop_name)
                actual = osim.PropertyHelper.getValueDouble(prop)
                match = "MATCH" if abs(actual - expected) < 1e-10 else f"MISMATCH (got {actual})"
                tee_print(f"    {prop_name} (via PropertyHelper): {actual} — {match}", buf)
            except Exception as e:
                tee_print(f"    {prop_name}: cannot read back ({e})", buf)


def test_spring_setters(buf):
    """Test SpringGeneralizedForce property setters."""
    tee_print("\n" + "=" * 70, buf)
    tee_print("TEST 2: SpringGeneralizedForce setters", buf)
    tee_print("=" * 70, buf)

    model = osim.Model()
    model.setName("spike_spring")

    # Need a body with a coordinate for the spring to reference
    body = osim.Body("body", 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(body)
    joint = osim.PinJoint("pin", model.getGround(), body)
    model.addJoint(joint)

    # Get the coordinate name from the pin joint
    coord_name = joint.getCoordinate().getName()
    tee_print(f"  Pin joint coordinate name: {coord_name}", buf)

    spring = osim.SpringGeneralizedForce()
    spring.setName("test_spring")

    setter_results = {}
    for prop_name, value in [
        ("stiffness", 1.0),
        ("rest_length", 0.0),
        ("viscosity", 0.0),
    ]:
        setter = f"set_{prop_name}"
        try:
            getattr(spring, setter)(value)
            setter_results[prop_name] = f"Pattern A ({setter}): OK"
        except AttributeError:
            try:
                prop = spring.updPropertyByName(prop_name)
                osim.PropertyHelper.setValueDouble(value, prop)
                setter_results[prop_name] = f"Pattern B (updPropertyByName): OK"
            except Exception as e:
                setter_results[prop_name] = f"FAILED: {e}"

    for prop_name, result in setter_results.items():
        tee_print(f"  {prop_name}: {result}", buf)

    # Set coordinate — try set_coordinate vs setPropertyByName
    try:
        spring.set_coordinate(coord_name)
        tee_print(f"  coordinate: Pattern A (set_coordinate): OK", buf)
    except AttributeError:
        try:
            prop = spring.updPropertyByName("coordinate")
            osim.PropertyHelper.setValueString(coord_name, prop)
            tee_print(f"  coordinate: Pattern B (updPropertyByName): OK", buf)
        except Exception as e:
            tee_print(f"  coordinate: FAILED ({e})", buf)

    model.addForce(spring)
    tee_print("  addForce(spring): OK", buf)

    try:
        model.finalizeConnections()
        state = model.initSystem()
        tee_print("  finalizeConnections() + initSystem(): OK", buf)
    except Exception as e:
        tee_print(f"  finalizeConnections() + initSystem(): FAILED ({e})", buf)


def test_contact_mesh_setters(buf):
    """Test Smith2018ContactMesh property setters."""
    tee_print("\n" + "=" * 70, buf)
    tee_print("TEST 3: Smith2018ContactMesh setters", buf)
    tee_print("=" * 70, buf)

    # Use the real model so we have valid bodies and geometry paths
    model = osim.Model(MODEL_PATH)

    mesh = osim.Smith2018ContactMesh()
    mesh.setName("test_contact_mesh")

    setter_results = {}
    for prop_name, value, val_type in [
        ("mesh_file", "femur_cartilage.stl", "string"),
        ("elastic_modulus", 1e6, "double"),
        ("poissons_ratio", 0.5, "double"),
        ("thickness", 0.005, "double"),
        ("use_variable_thickness", False, "bool"),
        ("min_thickness", 0.001, "double"),
        ("max_thickness", 0.01, "double"),
    ]:
        setter = f"set_{prop_name}"
        try:
            getattr(mesh, setter)(value)
            setter_results[prop_name] = f"Pattern A ({setter}): OK"
        except AttributeError:
            try:
                prop = mesh.updPropertyByName(prop_name)
                if val_type == "double":
                    osim.PropertyHelper.setValueDouble(value, prop)
                elif val_type == "string":
                    osim.PropertyHelper.setValueString(value, prop)
                elif val_type == "bool":
                    osim.PropertyHelper.setValueBool(value, prop)
                setter_results[prop_name] = f"Pattern B (updPropertyByName): OK"
            except Exception as e:
                setter_results[prop_name] = f"FAILED: {e}"

    for prop_name, result in setter_results.items():
        tee_print(f"  {prop_name}: {result}", buf)

    # Test socket_frame connection
    try:
        mesh.updSocket("frame").setConnecteePath("/bodyset/femur_distal_r")
        tee_print("  socket_frame (updSocket): OK", buf)
    except Exception as e:
        tee_print(f"  socket_frame (updSocket): FAILED ({e})", buf)
        # Try alternative
        try:
            mesh.connectSocket_frame(model.getBodySet().get("femur_distal_r"))
            tee_print("  socket_frame (connectSocket_frame): OK", buf)
        except Exception as e2:
            tee_print(f"  socket_frame (connectSocket_frame): FAILED ({e2})", buf)

    # Test location and orientation (Vec3 properties)
    for prop_name in ["location", "orientation"]:
        setter = f"set_{prop_name}"
        try:
            getattr(mesh, setter)(osim.Vec3(0, 0, 0))
            tee_print(f"  {prop_name}: Pattern A ({setter}(Vec3)): OK", buf)
        except (AttributeError, TypeError) as e:
            try:
                prop = mesh.updPropertyByName(prop_name)
                tee_print(f"  {prop_name}: Property exists, type={prop.getTypeName()}", buf)
            except Exception as e2:
                tee_print(f"  {prop_name}: FAILED ({e}, {e2})", buf)

    # Test scale_factors (Vec3 property)
    try:
        mesh.set_scale_factors(osim.Vec3(1, 1, 1))
        tee_print("  scale_factors: Pattern A (set_scale_factors(Vec3)): OK", buf)
    except (AttributeError, TypeError) as e:
        tee_print(f"  scale_factors: Pattern A failed ({e})", buf)
        try:
            prop = mesh.updPropertyByName("scale_factors")
            tee_print(f"  scale_factors: Property exists, type={prop.getTypeName()}", buf)
        except Exception as e2:
            tee_print(f"  scale_factors: FAILED ({e2})", buf)

    # Don't try initSystem — we don't have a valid mesh file, just testing setters
    tee_print("  (skipping initSystem — no valid mesh file for test)", buf)


def test_contact_force_setters(buf):
    """Test Smith2018ArticularContactForce property setters."""
    tee_print("\n" + "=" * 70, buf)
    tee_print("TEST 4: Smith2018ArticularContactForce setters", buf)
    tee_print("=" * 70, buf)

    force = osim.Smith2018ArticularContactForce()
    force.setName("test_contact_force")

    setter_results = {}
    for prop_name, value, val_type in [
        ("min_proximity", 0.0, "double"),
        ("max_proximity", 0.01, "double"),
        ("elastic_foundation_formulation", "linear", "string"),
        ("use_lumped_contact_model", True, "bool"),
    ]:
        setter = f"set_{prop_name}"
        try:
            getattr(force, setter)(value)
            setter_results[prop_name] = f"Pattern A ({setter}): OK"
        except AttributeError:
            try:
                prop = force.updPropertyByName(prop_name)
                if val_type == "double":
                    osim.PropertyHelper.setValueDouble(value, prop)
                elif val_type == "string":
                    osim.PropertyHelper.setValueString(value, prop)
                elif val_type == "bool":
                    osim.PropertyHelper.setValueBool(value, prop)
                setter_results[prop_name] = f"Pattern B (updPropertyByName): OK"
            except Exception as e:
                setter_results[prop_name] = f"FAILED: {e}"

    for prop_name, result in setter_results.items():
        tee_print(f"  {prop_name}: {result}", buf)

    # Test socket connections for target/casting mesh
    for socket_name in ["target_mesh", "casting_mesh"]:
        try:
            force.updSocket(socket_name).setConnecteePath(f"/contactgeometryset/dummy_{socket_name}")
            tee_print(f"  socket {socket_name} (updSocket): OK", buf)
        except Exception as e:
            tee_print(f"  socket {socket_name}: FAILED ({e})", buf)


def test_muscle_setters(buf):
    """Test Millard2012EquilibriumMuscle property setters, path points, and wrap."""
    tee_print("\n" + "=" * 70, buf)
    tee_print("TEST 5: Millard2012EquilibriumMuscle setters + path + wrap", buf)
    tee_print("=" * 70, buf)

    model = osim.Model()
    model.setName("spike_muscle")

    body_a = osim.Body("body_a", 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(body_a)
    joint_a = osim.WeldJoint("weld_a", model.getGround(), body_a)
    model.addJoint(joint_a)

    body_b = osim.Body("body_b", 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(body_b)
    joint_b = osim.WeldJoint("weld_b", model.getGround(), body_b)
    model.addJoint(joint_b)

    # Try constructor with args first, then default
    muscle = None
    try:
        muscle = osim.Millard2012EquilibriumMuscle("test_recfem", 1000.0, 0.08, 0.35, 0.0)
        tee_print("  Constructor(name, fmax, ofl, tsl, pennation): OK", buf)
    except Exception as e:
        tee_print(f"  Constructor with args: FAILED ({e})", buf)
        muscle = osim.Millard2012EquilibriumMuscle()
        muscle.setName("test_recfem")
        tee_print("  Default constructor + setName: OK", buf)

    # Test key property setters
    setter_results = {}
    for prop_name, value in [
        ("max_isometric_force", 1000.0),
        ("optimal_fiber_length", 0.08),
        ("tendon_slack_length", 0.35),
        ("pennation_angle_at_optimal", 0.05),
        ("max_contraction_velocity", 10.0),
    ]:
        setter = f"set_{prop_name}"
        try:
            getattr(muscle, setter)(value)
            setter_results[prop_name] = f"Pattern A ({setter}): OK"
        except AttributeError:
            try:
                prop = muscle.updPropertyByName(prop_name)
                osim.PropertyHelper.setValueDouble(value, prop)
                setter_results[prop_name] = f"Pattern B (updPropertyByName): OK"
            except Exception as e:
                setter_results[prop_name] = f"FAILED: {e}"

    for prop_name, result in setter_results.items():
        tee_print(f"  {prop_name}: {result}", buf)

    # Test path point addition
    tee_print("\n  Path point addition:", buf)
    gp = None
    for method_name in ["updGeometryPath", "getGeometryPath", "get_GeometryPath"]:
        try:
            gp = getattr(muscle, method_name)()
            tee_print(f"    {method_name}(): OK (type={type(gp).__name__})", buf)
            break
        except Exception as e:
            tee_print(f"    {method_name}(): FAILED ({e})", buf)

    if gp is not None:
        try:
            gp.appendNewPathPoint("test_recfem-P1", body_a, osim.Vec3(0.01, -0.02, 0.03))
            gp.appendNewPathPoint("test_recfem-P2", body_b, osim.Vec3(-0.01, 0.02, -0.03))
            tee_print("    appendNewPathPoint(): OK (2 points added)", buf)
        except Exception as e:
            tee_print(f"    appendNewPathPoint(): FAILED ({e})", buf)

    # Test wrap object reference
    tee_print("\n  Wrap object addition:", buf)
    # Add a wrap cylinder to body_a first
    cyl = osim.WrapCylinder()
    cyl.setName("test_wrap_cyl")
    cyl.set_radius(0.02)
    cyl.set_length(0.1)
    body_a.addWrapObject(cyl)
    tee_print("    Added WrapCylinder to body_a: OK", buf)

    if gp is not None:
        try:
            # Try addPathWrap
            gp.addPathWrap(cyl)
            tee_print("    addPathWrap(cyl): OK", buf)
        except Exception as e:
            tee_print(f"    addPathWrap(cyl): FAILED ({e})", buf)

    model.addForce(muscle)
    tee_print("  addForce(muscle): OK", buf)

    try:
        model.finalizeConnections()
        state = model.initSystem()
        tee_print("  finalizeConnections() + initSystem(): OK", buf)
    except Exception as e:
        tee_print(f"  finalizeConnections() + initSystem(): FAILED ({e})", buf)


def test_wrap_surface_setters(buf):
    """Test WrapCylinder and WrapEllipsoid property setters."""
    tee_print("\n" + "=" * 70, buf)
    tee_print("TEST 6: WrapCylinder / WrapEllipsoid setters", buf)
    tee_print("=" * 70, buf)

    model = osim.Model()
    model.setName("spike_wrap")

    body = osim.Body("body", 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(body)
    joint = osim.WeldJoint("weld", model.getGround(), body)
    model.addJoint(joint)

    # WrapCylinder
    tee_print("\n  WrapCylinder:", buf)
    cyl = osim.WrapCylinder()
    cyl.setName("test_cyl")

    for prop_name, value in [
        ("radius", 0.02),
        ("length", 0.1),
        ("quadrant", "x"),
    ]:
        setter = f"set_{prop_name}"
        try:
            getattr(cyl, setter)(value)
            tee_print(f"    {prop_name}: Pattern A ({setter}): OK", buf)
        except AttributeError:
            try:
                prop = cyl.updPropertyByName(prop_name)
                if isinstance(value, str):
                    osim.PropertyHelper.setValueString(value, prop)
                else:
                    osim.PropertyHelper.setValueDouble(value, prop)
                tee_print(f"    {prop_name}: Pattern B (updPropertyByName): OK", buf)
            except Exception as e:
                tee_print(f"    {prop_name}: FAILED ({e})", buf)

    # Translation and rotation (Vec3)
    for prop_name in ["translation", "xyz_body_rotation"]:
        setter = f"set_{prop_name}"
        try:
            getattr(cyl, setter)(osim.Vec3(0.01, 0.02, 0.03))
            tee_print(f"    {prop_name}: Pattern A ({setter}(Vec3)): OK", buf)
        except (AttributeError, TypeError) as e:
            tee_print(f"    {prop_name}: Pattern A failed ({e})", buf)
            try:
                prop = cyl.updPropertyByName(prop_name)
                tee_print(f"    {prop_name}: Property exists, type={prop.getTypeName()}", buf)
            except Exception as e2:
                tee_print(f"    {prop_name}: FAILED ({e2})", buf)

    body.addWrapObject(cyl)
    tee_print("    addWrapObject(cyl): OK", buf)

    # WrapEllipsoid
    tee_print("\n  WrapEllipsoid:", buf)
    ell = osim.WrapEllipsoid()
    ell.setName("test_ell")

    # Dimensions (Vec3)
    try:
        ell.set_dimensions(osim.Vec3(0.02, 0.03, 0.04))
        tee_print("    dimensions: Pattern A (set_dimensions(Vec3)): OK", buf)
    except (AttributeError, TypeError) as e:
        tee_print(f"    dimensions: Pattern A failed ({e})", buf)
        try:
            prop = ell.updPropertyByName("dimensions")
            tee_print(f"    dimensions: Property exists, type={prop.getTypeName()}", buf)
        except Exception as e2:
            tee_print(f"    dimensions: FAILED ({e2})", buf)

    for prop_name in ["translation", "xyz_body_rotation"]:
        setter = f"set_{prop_name}"
        try:
            getattr(ell, setter)(osim.Vec3(0.01, 0.02, 0.03))
            tee_print(f"    {prop_name}: Pattern A ({setter}(Vec3)): OK", buf)
        except (AttributeError, TypeError) as e:
            tee_print(f"    {prop_name}: Pattern A failed ({e})", buf)

    try:
        ell.set_quadrant("x")
        tee_print("    quadrant: Pattern A (set_quadrant): OK", buf)
    except AttributeError:
        tee_print("    quadrant: Pattern A failed", buf)

    body.addWrapObject(ell)
    tee_print("    addWrapObject(ell): OK", buf)

    try:
        model.finalizeConnections()
        state = model.initSystem()
        tee_print("  finalizeConnections() + initSystem(): OK", buf)
    except Exception as e:
        tee_print(f"  finalizeConnections() + initSystem(): FAILED ({e})", buf)


def test_custom_joint_spatial_transform(buf):
    """Test CustomJoint creation with SpatialTransform and LinearFunction."""
    tee_print("\n" + "=" * 70, buf)
    tee_print("TEST 7: CustomJoint + SpatialTransform + LinearFunction", buf)
    tee_print("=" * 70, buf)

    model = osim.Model()
    model.setName("spike_custom_joint")

    body_a = osim.Body("body_a", 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(body_a)
    weld = osim.WeldJoint("weld_a", model.getGround(), body_a)
    model.addJoint(weld)

    body_b = osim.Body("body_b", 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(body_b)

    # Create a CustomJoint — try constructor patterns
    tee_print("\n  CustomJoint construction:", buf)

    # First, try creating the SpatialTransform
    st = osim.SpatialTransform()
    tee_print("    SpatialTransform(): OK", buf)

    # Set up rotation1 axis with a LinearFunction
    axis_names = ["rotation1", "rotation2", "rotation3",
                  "translation1", "translation2", "translation3"]
    coord_names = ["flex", "add", "rot", "tx", "ty", "tz"]

    for i, (axis_name, coord_name) in enumerate(zip(axis_names, coord_names)):
        try:
            axis = getattr(st, f"get_{axis_name}")()
            tee_print(f"    get_{axis_name}(): OK (type={type(axis).__name__})", buf)

            # Set axis direction
            if "rotation" in axis_name:
                idx = int(axis_name[-1]) - 1
                vec = [0.0, 0.0, 0.0]
                vec[idx] = 1.0
                axis.set_axis(osim.Vec3(*vec))
                tee_print(f"      set_axis(): OK", buf)
            else:
                idx = int(axis_name[-1]) - 1
                vec = [0.0, 0.0, 0.0]
                vec[idx] = 1.0
                axis.set_axis(osim.Vec3(*vec))

            # Set coordinate name
            try:
                axis.set_coordinates(0, f"test_{coord_name}")
                tee_print(f"      set_coordinates(0, name): OK", buf)
            except Exception as e:
                tee_print(f"      set_coordinates(0, name): FAILED ({e})", buf)

            # Set function — LinearFunction(slope, intercept)
            try:
                lf = osim.LinearFunction(1.0, 0.0)
                axis.set_function(lf)
                tee_print(f"      set_function(LinearFunction): OK", buf)
            except Exception as e:
                tee_print(f"      set_function(LinearFunction): FAILED ({e})", buf)

            # Only print details for first axis to keep output manageable
            if i == 0:
                tee_print(f"      (remaining axes follow same pattern — details suppressed)", buf)
                break

        except Exception as e:
            tee_print(f"    get_{axis_name}(): FAILED ({e})", buf)
            break

    # Apply remaining axes silently
    for i, (axis_name, coord_name) in enumerate(zip(axis_names[1:], coord_names[1:])):
        try:
            axis = getattr(st, f"get_{axis_name}")()
            idx_map = {"rotation": int(axis_name[-1]) - 1, "translation": int(axis_name[-1]) - 1}
            vec = [0.0, 0.0, 0.0]
            if "rotation" in axis_name:
                vec[int(axis_name[-1]) - 1] = 1.0
            else:
                vec[int(axis_name[-1]) - 1] = 1.0
            axis.set_axis(osim.Vec3(*vec))
            axis.set_coordinates(0, f"test_{coord_name}")
            axis.set_function(osim.LinearFunction(1.0, 0.0))
        except Exception:
            pass

    # Create joint with the spatial transform
    try:
        joint = osim.CustomJoint(
            "test_knee",
            body_a,             # parent
            osim.Vec3(0),       # parent offset translation
            osim.Vec3(0),       # parent offset orientation
            body_b,             # child
            osim.Vec3(0),       # child offset translation
            osim.Vec3(0),       # child offset orientation
            st,                 # spatial transform
        )
        model.addJoint(joint)
        tee_print("  CustomJoint constructor (7 args + st): OK", buf)
    except Exception as e:
        tee_print(f"  CustomJoint constructor: FAILED ({e})", buf)
        # Try alternative: create joint then set transform
        try:
            joint = osim.CustomJoint("test_knee", body_a, body_b)
            model.addJoint(joint)
            tee_print("  CustomJoint constructor (3 args): OK", buf)
        except Exception as e2:
            tee_print(f"  CustomJoint alt constructor: FAILED ({e2})", buf)
            return

    try:
        model.finalizeConnections()
        state = model.initSystem()
        tee_print("  finalizeConnections() + initSystem(): OK", buf)

        # Verify coordinate names
        coords = model.getCoordinateSet()
        coord_list = [coords.get(i).getName() for i in range(coords.getSize())]
        tee_print(f"  Coordinates created: {coord_list}", buf)
    except Exception as e:
        tee_print(f"  finalizeConnections() + initSystem(): FAILED ({e})", buf)
        tee_print(f"    {traceback.format_exc()}", buf)


def test_weld_joint_offsets(buf):
    """Test WeldJoint with offset frames (translation + orientation)."""
    tee_print("\n" + "=" * 70, buf)
    tee_print("TEST 8: WeldJoint with offset frames", buf)
    tee_print("=" * 70, buf)

    model = osim.Model()
    model.setName("spike_weld_offset")

    body_a = osim.Body("femur_r", 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(body_a)
    joint_a = osim.WeldJoint("hip_r", model.getGround(), body_a)
    model.addJoint(joint_a)

    body_b = osim.Body("femur_distal_r", 0.008, osim.Vec3(0), osim.Inertia(0.001))
    model.addBody(body_b)

    # Create WeldJoint with offset translation (like femur_femur_distal_r)
    try:
        weld = osim.WeldJoint(
            "femur_femur_distal_r",
            body_a,                                      # parent
            osim.Vec3(-0.0056, -0.3742, -0.0012),       # parent offset translation
            osim.Vec3(0, 0, 0),                          # parent offset orientation
            body_b,                                      # child
            osim.Vec3(0, 0, 0),                          # child offset translation
            osim.Vec3(0, 0, 0),                          # child offset orientation
        )
        model.addJoint(weld)
        tee_print("  WeldJoint constructor (7 args): OK", buf)
    except Exception as e:
        tee_print(f"  WeldJoint constructor (7 args): FAILED ({e})", buf)
        return

    try:
        model.finalizeConnections()
        state = model.initSystem()
        tee_print("  finalizeConnections() + initSystem(): OK", buf)
    except Exception as e:
        tee_print(f"  finalizeConnections() + initSystem(): FAILED ({e})", buf)

    # Verify offset frame was created
    try:
        parent_frame = weld.getParentFrame()
        child_frame = weld.getChildFrame()
        tee_print(f"  Parent frame: {parent_frame.getName()} ({parent_frame.getConcreteClassName()})", buf)
        tee_print(f"  Child frame: {child_frame.getName()} ({child_frame.getConcreteClassName()})", buf)
    except Exception as e:
        tee_print(f"  Frame inspection: FAILED ({e})", buf)


def test_full_mini_model(buf):
    """Integration test: build a mini COMAK-like model from scratch."""
    tee_print("\n" + "=" * 70, buf)
    tee_print("TEST 9: Full mini-model (body + weld + custom + lig + spring + muscle)", buf)
    tee_print("=" * 70, buf)

    model = osim.Model()
    model.setName("spike_mini_comak")

    # Ground → femur_r (pin joint for knee flexion upstream)
    femur = osim.Body("femur_r", 5.0, osim.Vec3(0), osim.Inertia(0.1))
    model.addBody(femur)
    hip = osim.WeldJoint("hip_r", model.getGround(), femur)
    model.addJoint(hip)

    # femur_r → femur_distal_r (weld with offset)
    fem_dist = osim.Body("femur_distal_r", 0.008, osim.Vec3(0), osim.Inertia(0.001))
    model.addBody(fem_dist)
    fem_weld = osim.WeldJoint(
        "femur_femur_distal_r",
        femur,
        osim.Vec3(0, -0.374, 0),
        osim.Vec3(0, 0, 0),
        fem_dist,
        osim.Vec3(0, 0, 0),
        osim.Vec3(0, 0, 0),
    )
    model.addJoint(fem_weld)

    # tibia_proximal_r (will be child of knee_r)
    tib_prox = osim.Body("tibia_proximal_r", 0.008, osim.Vec3(0), osim.Inertia(0.001))
    model.addBody(tib_prox)

    # tibia_r (child of tibia_tibia_proximal_r weld)
    tibia = osim.Body("tibia_r", 3.0, osim.Vec3(0), osim.Inertia(0.05))
    model.addBody(tibia)

    # patella_r
    patella = osim.Body("patella_r", 0.4, osim.Vec3(0), osim.Inertia(0.005))
    model.addBody(patella)

    # CustomJoint: knee_r (femur_distal_r → tibia_proximal_r) with 6 DOFs
    # ALL 6 axes must have unique coordinates — OpenSim rejects collinear axes
    st_knee = osim.SpatialTransform()
    knee_axes = [
        ("rotation1", [0, 0, 1], "knee_flex_r"),
        ("rotation2", [1, 0, 0], "knee_add_r"),
        ("rotation3", [0, 1, 0], "knee_rot_r"),
        ("translation1", [1, 0, 0], "knee_tx_r"),
        ("translation2", [0, 1, 0], "knee_ty_r"),
        ("translation3", [0, 0, 1], "knee_tz_r"),
    ]
    for axis_name, axis_vec, coord_name in knee_axes:
        axis = getattr(st_knee, f"get_{axis_name}")()
        axis.set_axis(osim.Vec3(*axis_vec))
        axis.set_coordinates(0, coord_name)
        axis.set_function(osim.LinearFunction(1.0, 0.0))

    knee = osim.CustomJoint(
        "knee_r",
        fem_dist, osim.Vec3(0), osim.Vec3(0),
        tib_prox, osim.Vec3(0), osim.Vec3(0),
        st_knee,
    )
    model.addJoint(knee)

    # Weld: tibia_proximal_r → tibia_r
    tib_weld = osim.WeldJoint(
        "tibia_tibia_proximal_r",
        tib_prox,
        osim.Vec3(0.006, 0, 0),
        osim.Vec3(0, 0, 0),
        tibia,
        osim.Vec3(0, 0, 0),
        osim.Vec3(0, 0, 0),
    )
    model.addJoint(tib_weld)

    # PF joint: femur_distal_r → patella_r (6 DOFs — same pattern)
    st_pf = osim.SpatialTransform()
    pf_axes = [
        ("rotation1", [0, 0, 1], "pf_flex_r"),
        ("rotation2", [1, 0, 0], "pf_add_r"),
        ("rotation3", [0, 1, 0], "pf_rot_r"),
        ("translation1", [1, 0, 0], "pf_tx_r"),
        ("translation2", [0, 1, 0], "pf_ty_r"),
        ("translation3", [0, 0, 1], "pf_tz_r"),
    ]
    for axis_name, axis_vec, coord_name in pf_axes:
        axis = getattr(st_pf, f"get_{axis_name}")()
        axis.set_axis(osim.Vec3(*axis_vec))
        axis.set_coordinates(0, coord_name)
        axis.set_function(osim.LinearFunction(1.0, 0.0))

    pf = osim.CustomJoint(
        "pf_r",
        fem_dist, osim.Vec3(0, 0, 0.03), osim.Vec3(0, 0, 0),
        patella, osim.Vec3(0, 0, 0), osim.Vec3(0, 0, 0),
        st_pf,
    )
    model.addJoint(pf)

    tee_print("  Bodies + joints added: OK", buf)

    # Add a ligament
    lig = osim.Blankevoort1991Ligament()
    lig.setName("test_MCLd1")
    lig.set_linear_stiffness(5000.0)
    lig.set_slack_length(0.05)
    gp_lig = lig.updGeometryPath()
    gp_lig.appendNewPathPoint("MCLd1-P1", fem_dist, osim.Vec3(0.01, -0.02, 0.03))
    gp_lig.appendNewPathPoint("MCLd1-P2", tib_prox, osim.Vec3(-0.01, 0.02, -0.03))
    model.addForce(lig)
    tee_print("  Ligament added: OK", buf)

    # Add a spring on knee_add_r (a secondary DOF — typical COMAK pattern)
    spring = osim.SpringGeneralizedForce("knee_add_spring")
    spring.set_coordinate("knee_add_r")
    spring.set_stiffness(1.0)
    spring.set_rest_length(0.0)
    spring.set_viscosity(0.0)
    model.addForce(spring)
    tee_print("  Spring added: OK", buf)

    # Add a muscle spanning femur_r → patella_r
    muscle = osim.Millard2012EquilibriumMuscle("test_recfem", 1000.0, 0.08, 0.35, 0.0)
    gp_musc = muscle.updGeometryPath()
    gp_musc.appendNewPathPoint("recfem-P1", femur, osim.Vec3(0, -0.1, 0))
    gp_musc.appendNewPathPoint("recfem-P2", patella, osim.Vec3(0, 0.02, 0))
    model.addForce(muscle)
    tee_print("  Muscle added: OK", buf)

    # Add wrap surface to femur_distal_r
    cyl = osim.WrapCylinder()
    cyl.setName("test_Capsule_r")
    cyl.set_radius(0.02)
    cyl.set_length(0.1)
    cyl.set_translation(osim.Vec3(0, 0, 0))
    cyl.set_xyz_body_rotation(osim.Vec3(0, 0, 0))
    fem_dist.addWrapObject(cyl)
    tee_print("  Wrap surface added to femur_distal_r: OK", buf)

    # Finalize
    try:
        model.finalizeConnections()
        state = model.initSystem()
        tee_print("  finalizeConnections() + initSystem(): OK", buf)
    except Exception as e:
        tee_print(f"  finalizeConnections() + initSystem(): FAILED", buf)
        tee_print(f"    {traceback.format_exc()}", buf)
        return

    # Summary
    tee_print(f"\n  Model summary:", buf)
    tee_print(f"    Bodies: {model.getBodySet().getSize()}", buf)
    tee_print(f"    Joints: {model.getJointSet().getSize()}", buf)
    tee_print(f"    Forces: {model.getForceSet().getSize()}", buf)
    tee_print(f"    Coordinates: {model.getCoordinateSet().getSize()}", buf)

    coords = model.getCoordinateSet()
    for i in range(coords.getSize()):
        tee_print(f"      {coords.get(i).getName()}", buf)


def test_extract_spanning_muscle_from_real_model(buf):
    """Extract a spanning muscle from the real Smith2019 model and verify all properties."""
    tee_print("\n" + "=" * 70, buf)
    tee_print("TEST 10: Extract spanning muscle properties from Smith2019", buf)
    tee_print("=" * 70, buf)

    model = osim.Model(MODEL_PATH)
    state = model.initSystem()

    muscle_name = "recfem_r"
    force = model.getForceSet().get(muscle_name)
    muscle = osim.Millard2012EquilibriumMuscle.safeDownCast(force)

    if muscle is None:
        tee_print(f"  {muscle_name}: safeDownCast returned None", buf)
        return

    tee_print(f"  {muscle_name}: safeDownCast OK", buf)

    # Key properties
    props = {
        "max_isometric_force": muscle.get_max_isometric_force(),
        "optimal_fiber_length": muscle.get_optimal_fiber_length(),
        "tendon_slack_length": muscle.get_tendon_slack_length(),
        "pennation_angle_at_optimal": muscle.get_pennation_angle_at_optimal(),
        "max_contraction_velocity": muscle.get_max_contraction_velocity(),
    }
    for name, val in props.items():
        tee_print(f"    {name}: {val}", buf)

    # Path points
    gp = muscle.getGeometryPath()
    pp_set = gp.getPathPointSet()
    tee_print(f"\n  Path points ({pp_set.getSize()}):", buf)
    for i in range(pp_set.getSize()):
        pt = pp_set.get(i)
        pp = osim.PathPoint.safeDownCast(pt)
        if pp is not None:
            loc = pp.get_location()
            body_name = pp.getParentFrame().getName()
            tee_print(f"    {pp.getName()}: body={body_name}, loc=[{loc[0]:.6f}, {loc[1]:.6f}, {loc[2]:.6f}]", buf)
        else:
            tee_print(f"    {pt.getName()}: type={pt.getConcreteClassName()} (not PathPoint)", buf)

    # Wrap objects
    wrap_set = gp.getWrapSet()
    tee_print(f"\n  Wrap objects ({wrap_set.getSize()}):", buf)
    for i in range(wrap_set.getSize()):
        wrap = wrap_set.get(i)
        tee_print(f"    {wrap.getName()}: wrap_object={wrap.getWrapObjectName()}", buf)

    # Full property dump for completeness
    tee_print(f"\n  All properties ({muscle.getNumProperties()}):", buf)
    for i in range(muscle.getNumProperties()):
        prop = muscle.getPropertyByIndex(i)
        val_str = prop.toString()
        if len(val_str) > 100:
            val_str = val_str[:100] + "..."
        tee_print(f"    {prop.getName()} = {val_str}", buf)


def main():
    buf = StringIO()
    tee_print("SPIKE: OpenSim Property Setter API Verification", buf)
    tee_print(f"Model: {MODEL_PATH}", buf)
    tee_print("=" * 70, buf)

    tests = [
        test_ligament_setters,
        test_spring_setters,
        test_contact_mesh_setters,
        test_contact_force_setters,
        test_muscle_setters,
        test_wrap_surface_setters,
        test_custom_joint_spatial_transform,
        test_weld_joint_offsets,
        test_full_mini_model,
        test_extract_spanning_muscle_from_real_model,
    ]

    passed = 0
    failed = 0
    for test_func in tests:
        try:
            test_func(buf)
            passed += 1
        except Exception as e:
            tee_print(f"\n  UNEXPECTED ERROR: {e}", buf)
            tee_print(traceback.format_exc(), buf)
            failed += 1

    tee_print("\n" + "=" * 70, buf)
    tee_print(f"SUMMARY: {passed} tests ran, {failed} unexpected errors", buf)
    tee_print("=" * 70, buf)

    OUTPUT_PATH.write_text(buf.getvalue())
    print(f"\nResults written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
