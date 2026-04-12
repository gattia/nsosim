"""
Phase 0: Model Audit & API Verification for COMAK Knee Assembly plan.

Run with: conda run -n comak python scripts/phase0_knee_assembly_audit/phase0_model_audit.py

Outputs results to scripts/phase0_knee_assembly_audit/phase0_model_audit_results.txt

This script resolves all empirical unknowns before writing production code:
  0A: Component enumeration (bodies, joints, forces, contacts, constraints, markers)
  0B: Spatial transform function types for COMAK joints
  0C: Ligament path point types + spanning muscle details
  0D: Wrap surfaces on COMAK and main chain bodies
  0E: API constructor verification + full property dumps
  0F: Reference segment lengths + weld joint offsets
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import opensim as osim

MODEL_PATH = (
    "/dataNAS/people/aagatti/projects/comak_gait_simulation/"
    "COMAK_SIMULATION_REQUIREMENTS/data/reference_data/comak_models/"
    "current/full_body_healthy_knee.osim"
)

COMAK_BODIES = [
    "femur_distal_r",
    "tibia_proximal_r",
    "patella_r",
    "meniscus_medial_r",
    "meniscus_lateral_r",
]

COMAK_JOINTS = [
    "knee_r",
    "pf_r",
    "meniscus_medial_r",
    "meniscus_lateral_r",
]

SPANNING_MUSCLES = ["recfem_r", "vasint_r", "vaslat_r", "vasmed_r"]

OUTPUT_PATH = Path(__file__).parent / "phase0_model_audit_results.txt"


def section(title):
    return f"\n{'=' * 60}\n{title}\n{'=' * 60}\n"


def run_0a(model):
    """Enumerate all components by class."""
    lines = [section("0A: COMPONENT ENUMERATION")]

    # Bodies
    n = model.getBodySet().getSize()
    lines.append(f"--- Bodies ({n}) ---")
    for i in range(n):
        b = model.getBodySet().get(i)
        lines.append(f"  {b.getName()} (mass={b.getMass():.6f})")

    # Joints
    n = model.getJointSet().getSize()
    lines.append(f"\n--- Joints ({n}) ---")
    for i in range(n):
        j = model.getJointSet().get(i)
        lines.append(f"  {j.getName()} -> {j.getConcreteClassName()}")

    # Forces
    n = model.getForceSet().getSize()
    lines.append(f"\n--- Forces ({n}) ---")
    force_classes = Counter()
    for i in range(n):
        f = model.getForceSet().get(i)
        force_classes[f.getConcreteClassName()] += 1
        lines.append(f"  {f.getName()} -> {f.getConcreteClassName()}")
    lines.append(f"\nForce class counts: {dict(force_classes)}")

    # Contact geometry
    n = model.getContactGeometrySet().getSize()
    lines.append(f"\n--- Contact Geometry ({n}) ---")
    for i in range(n):
        c = model.getContactGeometrySet().get(i)
        lines.append(f"  {c.getName()} -> {c.getConcreteClassName()}")

    # Constraints
    cs = model.getConstraintSet()
    lines.append(f"\n--- Constraints ({cs.getSize()}) ---")
    for i in range(cs.getSize()):
        lines.append(f"  {cs.get(i).getName()} -> {cs.get(i).getConcreteClassName()}")
    if cs.getSize() == 0:
        lines.append("  (none)")

    # Markers on COMAK bodies
    ms = model.getMarkerSet()
    lines.append(f"\n--- Markers on COMAK bodies (of {ms.getSize()} total) ---")
    found = False
    for i in range(ms.getSize()):
        m = ms.get(i)
        parent = m.getParentFrameName()
        if any(cb in parent for cb in COMAK_BODIES):
            lines.append(f"  {m.getName()} -> {parent}")
            found = True
    if not found:
        lines.append("  (none)")

    return "\n".join(lines)


def run_0b(model):
    """Check spatial transform function types for COMAK joints."""
    lines = [section("0B: SPATIAL TRANSFORM FUNCTION TYPES")]

    for i in range(model.getJointSet().getSize()):
        joint = model.getJointSet().get(i)
        if joint.getConcreteClassName() != "CustomJoint":
            continue
        if joint.getName() not in COMAK_JOINTS:
            continue

        cj = osim.CustomJoint.safeDownCast(joint)
        st = cj.getSpatialTransform()
        components = [
            "rotation1", "rotation2", "rotation3",
            "translation1", "translation2", "translation3",
        ]
        for comp in components:
            axis = getattr(st, f"get_{comp}")()
            func = axis.get_function()
            n_coords = axis.getCoordinateNamesInArray().getSize()
            coord_name = axis.get_coordinates(0) if n_coords > 0 else "(none)"
            ax = axis.get_axis()
            lines.append(
                f"  {joint.getName()}.{comp}: "
                f"func={func.getConcreteClassName()}, "
                f"coord={coord_name}, "
                f"axis=[{ax[0]:.1f},{ax[1]:.1f},{ax[2]:.1f}]"
            )
        lines.append("")

    return "\n".join(lines)


def run_0c(model):
    """Check ligament path point types and spanning muscle details."""
    lines = [section("0C: LIGAMENT PATH POINT TYPES + SPANNING MUSCLES")]

    # Ligament path points
    non_standard = []
    total_points = 0
    for i in range(model.getForceSet().getSize()):
        f = model.getForceSet().get(i)
        if f.getConcreteClassName() == "Blankevoort1991Ligament":
            lig = osim.Blankevoort1991Ligament.safeDownCast(f)
            gp = lig.getGeometryPath()
            pp = gp.getPathPointSet()
            for j in range(pp.getSize()):
                pt = pp.get(j)
                total_points += 1
                if pt.getConcreteClassName() != "PathPoint":
                    non_standard.append(
                        f"{f.getName()} point {j}: {pt.getConcreteClassName()}"
                    )

    lines.append(f"Total path points across 91 ligaments: {total_points}")
    if non_standard:
        lines.append("NON-STANDARD path points:")
        for ns in non_standard:
            lines.append(f"  {ns}")
    else:
        lines.append("All path points are standard PathPoint")

    # Spanning muscles
    lines.append("\n--- Spanning muscle path points ---")
    for i in range(model.getForceSet().getSize()):
        f = model.getForceSet().get(i)
        if f.getName() in SPANNING_MUSCLES:
            m = osim.Millard2012EquilibriumMuscle.safeDownCast(f)
            gp = m.getGeometryPath()
            pp = gp.getPathPointSet()
            lines.append(f"  {f.getName()}: {pp.getSize()} points")
            for j in range(pp.getSize()):
                apt = pp.get(j)
                class_name = apt.getConcreteClassName()
                parent = apt.getSocket("parent_frame").getConnecteePath()
                if class_name == "PathPoint":
                    pt = osim.PathPoint.safeDownCast(apt)
                    loc = pt.get_location()
                    lines.append(
                        f"    {apt.getName()} ({class_name}) "
                        f"on {parent} "
                        f"@ [{loc[0]:.6f}, {loc[1]:.6f}, {loc[2]:.6f}]"
                    )
                else:
                    lines.append(
                        f"    {apt.getName()} ({class_name}) on {parent}"
                    )
            pwset = gp.getWrapSet()
            lines.append(f"    Wrap objects: {pwset.getSize()}")
            for j in range(pwset.getSize()):
                lines.append(f"      {pwset.get(j).getWrapObjectName()}")

    return "\n".join(lines)


def run_0d(model):
    """Check wrap surfaces on COMAK and main chain bodies."""
    lines = [section("0D: WRAP SURFACES ON COMAK AND MAIN CHAIN BODIES")]

    bodies_to_check = COMAK_BODIES + ["femur_r", "tibia_r"]
    for body_name in bodies_to_check:
        body = model.getBodySet().get(body_name)
        try:
            ws = body.get_WrapObjectSet()
            n = ws.getSize()
            lines.append(f"{body_name}: {n} wrap objects")
            for j in range(n):
                wo = ws.get(j)
                t = wo.get_translation()
                r = wo.get_xyz_body_rotation()
                lines.append(f"  {wo.getName()} -> {wo.getConcreteClassName()}")
                lines.append(
                    f"    translation: [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]"
                )
                lines.append(
                    f"    rotation: [{r[0]:.6f}, {r[1]:.6f}, {r[2]:.6f}]"
                )
                if wo.getConcreteClassName() == "WrapCylinder":
                    cyl = osim.WrapCylinder.safeDownCast(wo)
                    lines.append(
                        f"    radius={cyl.get_radius():.6f}, "
                        f"length={cyl.get_length():.6f}, "
                        f"quadrant={cyl.get_quadrant()}"
                    )
                elif wo.getConcreteClassName() == "WrapEllipsoid":
                    ell = osim.WrapEllipsoid.safeDownCast(wo)
                    d = ell.get_dimensions()
                    lines.append(
                        f"    dimensions=[{d[0]:.6f}, {d[1]:.6f}, {d[2]:.6f}]"
                    )
        except Exception as e:
            lines.append(f"{body_name}: error getting wrap set: {e}")

    return "\n".join(lines)


def run_0e():
    """Verify API constructors and dump all properties."""
    lines = [section("0E: API CONSTRUCTOR VERIFICATION + PROPERTY DUMPS")]

    # addBody + addJoint
    test_model = osim.Model()
    body = osim.Body("test_body", 1.0, osim.Vec3(0), osim.Inertia(1))
    test_model.addBody(body)
    joint = osim.WeldJoint("test_weld", test_model.getGround(), body)
    test_model.addJoint(joint)
    test_model.finalizeConnections()
    test_model.initSystem()
    lines.append("addBody + addJoint + initSystem: CONFIRMED")

    # Dump properties for each COMAK force class
    classes = [
        ("Blankevoort1991Ligament", osim.Blankevoort1991Ligament),
        ("SpringGeneralizedForce", osim.SpringGeneralizedForce),
        ("Smith2018ContactMesh", osim.Smith2018ContactMesh),
        ("Smith2018ArticularContactForce", osim.Smith2018ArticularContactForce),
        ("Millard2012EquilibriumMuscle", osim.Millard2012EquilibriumMuscle),
    ]

    for class_name, cls in classes:
        lines.append(f"\n--- {class_name} ---")
        try:
            obj = cls()
            lines.append("Constructor: CONFIRMED")
            lines.append("All properties:")
            for i in range(obj.getNumProperties()):
                prop = obj.getPropertyByIndex(i)
                lines.append(f"  [{i}] {prop.getName()} = {prop.toString()}")
        except Exception as e:
            lines.append(f"Constructor FAILED: {e}")

    return "\n".join(lines)


def run_0f(model):
    """Extract reference segment lengths and weld joint offsets."""
    lines = [section("0F: REFERENCE SEGMENT LENGTHS + WELD JOINT OFFSETS")]

    state = model.initSystem()

    # Segment lengths from joint positions
    def get_joint_child_pos(joint_name):
        joint = model.getJointSet().get(joint_name)
        child_frame = joint.getChildFrame()
        pos = child_frame.getPositionInGround(state)
        return np.array([pos[0], pos[1], pos[2]])

    hip_pos = get_joint_child_pos("hip_r")
    knee_pos = get_joint_child_pos("knee_r")
    ankle_pos = get_joint_child_pos("ankle_r")

    fem_len = np.linalg.norm(knee_pos - hip_pos)
    tib_len = np.linalg.norm(ankle_pos - knee_pos)

    lines.append(
        f"Hip center (ground):   "
        f"[{hip_pos[0]:.6f}, {hip_pos[1]:.6f}, {hip_pos[2]:.6f}]"
    )
    lines.append(
        f"Knee center (ground):  "
        f"[{knee_pos[0]:.6f}, {knee_pos[1]:.6f}, {knee_pos[2]:.6f}]"
    )
    lines.append(
        f"Ankle center (ground): "
        f"[{ankle_pos[0]:.6f}, {ankle_pos[1]:.6f}, {ankle_pos[2]:.6f}]"
    )
    lines.append(f"Femur length (hip->knee):    {fem_len:.6f} m")
    lines.append(f"Tibia length (knee->ankle):  {tib_len:.6f} m")
    lines.append(f"Total leg length:            {fem_len + tib_len:.6f} m")

    # Weld joint offsets
    lines.append("\n--- Weld joint offsets ---")
    for jname in ["femur_femur_distal_r", "tibia_tibia_proximal_r"]:
        j = model.getJointSet().get(jname)
        pf = j.getParentFrame()
        cf = j.getChildFrame()
        lines.append(f"{jname}:")
        lines.append(
            f"  parent frame: {pf.getName()} ({pf.getConcreteClassName()})"
        )
        lines.append(
            f"  child frame: {cf.getName()} ({cf.getConcreteClassName()})"
        )
        for label, frame in [("parent", pf), ("child", cf)]:
            if "Offset" in frame.getConcreteClassName():
                off = osim.PhysicalOffsetFrame.safeDownCast(frame)
                t = off.get_translation()
                o = off.get_orientation()
                lines.append(
                    f"  {label} offset: "
                    f"t=[{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}], "
                    f"o=[{o[0]:.6f}, {o[1]:.6f}, {o[2]:.6f}]"
                )

    return "\n".join(lines)


def main():
    print(f"Loading model: {MODEL_PATH}")
    model = osim.Model(MODEL_PATH)

    results = []
    results.append(f"Phase 0 Model Audit: {MODEL_PATH}")
    results.append(f"Model name: {model.getName()}")

    print("Running 0A: Component enumeration...")
    results.append(run_0a(model))

    print("Running 0B: Spatial transform function types...")
    results.append(run_0b(model))

    print("Running 0C: Ligament path point types...")
    results.append(run_0c(model))

    print("Running 0D: Wrap surfaces...")
    results.append(run_0d(model))

    print("Running 0E: API constructor verification...")
    results.append(run_0e())

    print("Running 0F: Reference segment lengths...")
    results.append(run_0f(model))

    output = "\n".join(results)

    OUTPUT_PATH.write_text(output)
    print(f"\nResults written to: {OUTPUT_PATH}")
    print(output)


if __name__ == "__main__":
    main()
