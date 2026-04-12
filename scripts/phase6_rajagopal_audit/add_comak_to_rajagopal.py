"""Phase 6B: Strip Rajagopal's knee and add COMAK knee from Smith2019 config.

Takes the scaled Rajagopal model and:
1. Removes Rajagopal's existing right knee joint, patellofemoral joint, patella body
2. Removes the patellofemoral constraint
3. Removes spanning muscles (recfem_r, vasint_r, vaslat_r, vasmed_r)
4. Adds the full COMAK knee from Smith2019 config

Usage:
    conda run -n comak python scripts/phase6_rajagopal_audit/add_comak_to_rajagopal.py
"""

import logging
import os
import shutil
import sys

import opensim as osim

# Add repo root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)

from nsosim.knee_assembly import ComakKneeConfig, add_comak_knee

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SCALED_RAJAGOPAL = os.path.join(
    REPO_ROOT,
    "tests/fixtures/osim_models/rajagopal/RajagopalLaiUhlrich2023_scaled_to_smith2019.osim",
)
SMITH2019_MODEL = os.path.join(
    REPO_ROOT, "tests/fixtures/osim_models/full_body_healthy_knee.osim"
)
OUTPUT_MODEL = os.path.join(
    REPO_ROOT,
    "tests/fixtures/osim_models/rajagopal/RajagopalLaiUhlrich2023_comak_knee.osim",
)
SMITH2019_GEOMETRY = os.path.join(REPO_ROOT, "tests/fixtures/osim_models/Geometry")
RAJAGOPAL_GEOMETRY = os.path.join(
    REPO_ROOT, "tests/fixtures/osim_models/rajagopal/Geometry"
)

# Components to remove from Rajagopal before adding COMAK knee
RAJAGOPAL_KNEE_JOINTS = ["walker_knee_r", "patellofemoral_r"]
RAJAGOPAL_KNEE_BODIES = ["patella_r"]  # tibia_r and femur_r stay
RAJAGOPAL_KNEE_CONSTRAINTS = ["patellofemoral_knee_angle_r_con"]
RAJAGOPAL_SPANNING_MUSCLES = ["recfem_r", "vasint_r", "vaslat_r", "vasmed_r"]

# Bodies that are "proximal" to the knee — path points on these bodies
# come from Rajagopal. Path points on patella_r and tibia_r come from
# the COMAK config (Smith2019).
PROXIMAL_BODIES = {"pelvis", "femur_r", "femur_distal_r"}

# Rajagopal quad wrap surfaces to remove from femur_r (replaced by COMAK ones)
RAJAGOPAL_QUAD_WRAPS_TO_REMOVE = ["KnExt_at_fem_r", "KnExtVL_at_fem_r"]

# Smith2019 quad wrap surfaces to add to femur_r (personalized from bone mesh)
# These are NOT in the COMAK config (they're on femur_r, a main chain body).
# Values extracted from the Smith2019 reference model.
# Whether to remove Rajagopal's original wraps that are being replaced.
# False = keep both for comparison. True = remove originals (production).
REMOVE_REPLACED_WRAPS = False

# Rajagopal condyle wraps on femur_r to optionally remove
RAJAGOPAL_CONDYLE_WRAPS = ["GasLat_at_condyles_r", "GasMed_at_condyles_r"]

# Smith2019 wrap surfaces to add to femur_r (personalized from bone mesh).
# These are NOT in the COMAK config (they're on femur_r, a main chain body).
SMITH2019_FEMUR_WRAPS = [
    {
        "name": "KnExt_at_fem_r",
        "type": "WrapCylinder",
        "translation": [0.009252183627908708, -0.3673116900279757, 0.0],
        "xyz_body_rotation": [0.0, 0.0, 0.0],
        "quadrant": "x",
        "radius": 0.025906114158144384,
        "length": 0.18504367255817417,
    },
    {
        "name": "KnExt_vasint_at_fem_r",
        "type": "WrapCylinder",
        "translation": [0.009252183627908708, -0.3673116900279757, 0.0],
        "xyz_body_rotation": [0.0, 0.0, 0.0],
        "quadrant": "x",
        "radius": 0.02405567743256264,
        "length": 0.18504367255817417,
    },
    {
        "name": "Gastroc_at_Condyles_r",
        "type": "WrapEllipsoid",
        "translation": [0.009252183627908708, -0.3770264828372798, 0.0002775655088372612],
        "xyz_body_rotation": [0.0589921, -0.0872665, 0.0232129],
        "quadrant": "all",
        "dimensions": [0.0277566, 0.0231305, 0.138783],
    },
]


def extract_rajagopal_spanning_muscles(model):
    """Extract spanning muscle properties + proximal path points from Rajagopal.

    Returns a dict of {muscle_name: {properties, proximal_points, wrap_objects}}.
    Proximal points are those on bodies above the knee (pelvis, femur_r).
    Patella and tibia points are discarded — they'll come from the COMAK config.
    """
    muscles = {}
    fs = model.getForceSet()
    for name in RAJAGOPAL_SPANNING_MUSCLES:
        f = fs.get(name)
        m = osim.Muscle.safeDownCast(f)
        mm = osim.Millard2012EquilibriumMuscle.safeDownCast(f)

        # Extract proximal path points only
        gp = m.getGeometryPath()
        pps = gp.getPathPointSet()
        proximal_points = []
        for j in range(pps.getSize()):
            pt = pps.get(j)
            body = pt.getParentFrame().findBaseFrame().getName()
            if body in PROXIMAL_BODIES:
                pp = osim.PathPoint.safeDownCast(pt)
                if pp is not None:
                    loc = [pp.get_location().get(k) for k in range(3)]
                    proximal_points.append({
                        "name": pt.getName(),
                        "body": body,
                        "location": loc,
                    })

        # Extract wrap objects
        ws = gp.getWrapSet()
        wrap_objects = [ws.get(j).getWrapObjectName() for j in range(ws.getSize())]

        muscles[name] = {
            "max_isometric_force": m.getMaxIsometricForce(),
            "optimal_fiber_length": m.getOptimalFiberLength(),
            "tendon_slack_length": m.getTendonSlackLength(),
            "pennation_angle": m.getPennationAngleAtOptimalFiberLength(),
            "max_contraction_velocity": mm.get_max_contraction_velocity(),
            "proximal_points": proximal_points,
            "wrap_objects": wrap_objects,
        }
        logger.info(
            f"Extracted {name}: {len(proximal_points)} proximal points, "
            f"{len(wrap_objects)} wraps"
        )

    return muscles


def build_hybrid_spanning_muscles(rajagopal_muscles, comak_config):
    """Build hybrid spanning muscles: Rajagopal proximal + COMAK patella points.

    For each spanning muscle:
    - Proximal path points (pelvis, femur_r): from Rajagopal
    - Distal path points (patella_r): from COMAK config (Smith2019)
    - Force properties (max force, fiber length, etc.): from Rajagopal
    - Wrap objects: from Rajagopal (with name mapping to COMAK wraps)
    - Tibia points: REMOVED (COMAK patellar tendon ligaments handle patella→tibia)

    Returns a list of ComakMuscle with hybrid path points.
    """
    from nsosim.knee_assembly.config import ComakMuscle

    # Map Rajagopal wrap names → COMAK wrap names (for wraps we swapped on femur_r)
    WRAP_NAME_MAP = {
        "KnExt_at_fem_r": "KnExt_at_fem_r",  # same name, different geometry
        "KnExtVL_at_fem_r": "KnExt_at_fem_r",  # Rajagopal VL-specific → COMAK shared
    }

    # Smith2019 wrap assignments per muscle (which COMAK wrap each muscle should use)
    COMAK_WRAP_ASSIGNMENTS = {
        "recfem_r": ["KnExt_at_fem_r"],
        "vasint_r": ["KnExt_vasint_at_fem_r"],
        "vaslat_r": ["KnExt_at_fem_r"],
        "vasmed_r": ["KnExt_at_fem_r"],
    }

    # Build lookup of COMAK spanning muscle patella points
    comak_patella_points = {}
    for cm in comak_config.spanning_muscles:
        patella_pts = [
            pp for pp in cm.path_points if pp["body"] == "patella_r"
        ]
        comak_patella_points[cm.name] = patella_pts

    hybrid_muscles = []
    for name, raj_data in rajagopal_muscles.items():
        # Merge: Rajagopal proximal + COMAK patella
        merged_points = list(raj_data["proximal_points"])

        comak_pts = comak_patella_points.get(name, [])
        if comak_pts:
            merged_points.extend(comak_pts)
            logger.info(
                f"  {name}: {len(raj_data['proximal_points'])} Rajagopal proximal + "
                f"{len(comak_pts)} COMAK patella points"
            )
        else:
            logger.warning(f"  {name}: no COMAK patella points found — muscle may not work")

        # Use COMAK wrap assignments for the quad wraps on femur_r
        wrap_objects = COMAK_WRAP_ASSIGNMENTS.get(name, raj_data["wrap_objects"])

        hybrid = ComakMuscle(
            name=name,
            max_isometric_force=raj_data["max_isometric_force"],
            optimal_fiber_length=raj_data["optimal_fiber_length"],
            tendon_slack_length=raj_data["tendon_slack_length"],
            pennation_angle_at_optimal=raj_data["pennation_angle"],
            max_contraction_velocity=raj_data["max_contraction_velocity"],
            path_points=merged_points,
            wrap_objects=wrap_objects,
        )
        hybrid_muscles.append(hybrid)

    return hybrid_muscles


def strip_rajagopal_knee(model):
    """Remove Rajagopal's right knee components to prepare for COMAK knee.

    Removes: walker_knee_r joint, patellofemoral_r joint, patella_r body,
    patellofemoral constraint, and spanning muscles.

    The model will be invalid after this (tibia_r is orphaned) until the
    COMAK knee is added.
    """
    # 1. Remove constraints first (they reference coordinates that will go away)
    cs = model.getConstraintSet()
    indices = []
    for i in range(cs.getSize()):
        if cs.get(i).getName() in RAJAGOPAL_KNEE_CONSTRAINTS:
            indices.append(i)
            logger.info(f"Removing constraint: {cs.get(i).getName()}")
    for i in sorted(indices, reverse=True):
        cs.remove(i)

    # 2. Remove spanning muscles — they'll be re-added with hybrid path points
    # (Rajagopal proximal + Smith2019 patella attachments)
    fs = model.getForceSet()
    indices = []
    for i in range(fs.getSize()):
        if fs.get(i).getName() in RAJAGOPAL_SPANNING_MUSCLES:
            indices.append(i)
            logger.info(f"Removing spanning muscle: {fs.get(i).getName()}")
    for i in sorted(indices, reverse=True):
        fs.remove(i)

    # 3. Remove joints (must happen before body removal)
    js = model.getJointSet()
    indices = []
    for i in range(js.getSize()):
        if js.get(i).getName() in RAJAGOPAL_KNEE_JOINTS:
            indices.append(i)
            logger.info(f"Removing joint: {js.get(i).getName()}")
    for i in sorted(indices, reverse=True):
        js.remove(i)

    # 4. Remove bodies
    bs = model.getBodySet()
    indices = []
    for i in range(bs.getSize()):
        if bs.get(i).getName() in RAJAGOPAL_KNEE_BODIES:
            indices.append(i)
            logger.info(f"Removing body: {bs.get(i).getName()}")
    for i in sorted(indices, reverse=True):
        bs.remove(i)

    # 5. Swap wrap surfaces on femur_r:
    #    - Remove Rajagopal's quad wraps (KnExt_at_fem_r, KnExtVL_at_fem_r)
    #    - Optionally remove Rajagopal's gastroc condyle wraps
    #    - Add COMAK wraps (personalized from bone mesh in the NSM pipeline)
    femur_r = model.getBodySet().get("femur_r")
    ws_prop = femur_r.getPropertyByName("WrapObjectSet")
    ws_set = osim.WrapObjectSet.safeDownCast(ws_prop.getValueAsObject())

    # Always remove quad wraps (being replaced by COMAK equivalents with same name)
    wraps_to_remove = set(RAJAGOPAL_QUAD_WRAPS_TO_REMOVE)
    if REMOVE_REPLACED_WRAPS:
        wraps_to_remove |= set(RAJAGOPAL_CONDYLE_WRAPS)

    indices_to_remove = []
    for i in range(ws_set.getSize()):
        if ws_set.get(i).getName() in wraps_to_remove:
            indices_to_remove.append(i)
            logger.info(f"Removing Rajagopal wrap: {ws_set.get(i).getName()}")
    for i in sorted(indices_to_remove, reverse=True):
        ws_set.remove(i)

    # Add COMAK wraps (cylinders and ellipsoids)
    for wrap_def in SMITH2019_FEMUR_WRAPS:
        if wrap_def["type"] == "WrapCylinder":
            wo = osim.WrapCylinder()
            wo.set_radius(wrap_def["radius"])
            wo.set_length(wrap_def["length"])
        elif wrap_def["type"] == "WrapEllipsoid":
            wo = osim.WrapEllipsoid()
            wo.set_dimensions(osim.Vec3(*wrap_def["dimensions"]))
        else:
            raise ValueError(f"Unknown wrap type: {wrap_def['type']}")
        wo.setName(wrap_def["name"])
        wo.set_translation(osim.Vec3(*wrap_def["translation"]))
        wo.set_xyz_body_rotation(osim.Vec3(*wrap_def["xyz_body_rotation"]))
        wo.set_quadrant(wrap_def["quadrant"])
        femur_r.addWrapObject(wo)
        logger.info(f"Added COMAK wrap: {wrap_def['name']} ({wrap_def['type']}) on femur_r")

    # 6. Do NOT call finalizeConnections() here — the spanning muscles still
    # reference patella_r which was just removed. Finalizing now would segfault
    # on the dangling socket. The COMAK add will create a new patella_r body
    # (same name), and finalizeConnections() is called at the end of add_comak_knee().
    logger.info(
        "Strip complete — model has dangling references (patella_r removed, "
        "tibia_r orphaned). COMAK add will resolve these."
    )

    return model


def copy_comak_geometry(src_geometry_dir, dst_geometry_dir):
    """Copy COMAK-specific STL/VTP files to the target Geometry folder."""
    os.makedirs(dst_geometry_dir, exist_ok=True)

    # Copy all STL files (COMAK contact meshes)
    copied = 0
    if os.path.exists(src_geometry_dir):
        for f in os.listdir(src_geometry_dir):
            if f.endswith((".stl", ".STL")):
                src = os.path.join(src_geometry_dir, f)
                dst = os.path.join(dst_geometry_dir, f)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied += 1
    logger.info(f"Copied {copied} geometry files to {dst_geometry_dir}")
    return copied


def extract_comak_config():
    """Extract COMAK knee config from Smith2019 by stripping it."""
    from nsosim.knee_assembly import strip_comak_knee

    logger.info(f"Loading Smith2019 model: {SMITH2019_MODEL}")
    smith_model = osim.Model(SMITH2019_MODEL)
    smith_model.initSystem()

    logger.info("Stripping COMAK knee from Smith2019...")
    _, config = strip_comak_knee(smith_model)
    logger.info(
        f"Extracted config: {len(config.bodies)} bodies, "
        f"{len(config.ligaments)} ligaments, "
        f"{len(config.spanning_muscles)} spanning muscles"
    )
    return config


def rescale_muscle_properties_to_path_length(model, muscle_names, reference_model):
    """Rescale muscle optimal fiber length and tendon slack length to match new path.

    When muscle path points change (e.g., hybrid Rajagopal proximal + COMAK patella),
    the path length changes but the muscle properties (optimal fiber length, tendon
    slack length) still reflect the original path. This creates a mismatch: if the
    new path is shorter, the muscle-tendon unit is slack and can't generate force.

    This function computes the ratio of new path length to reference path length,
    then scales both optimal_fiber_length and tendon_slack_length by that ratio.
    This preserves the muscle's force-length relationship at the new path length.

    Parameters
    ----------
    model : osim.Model
        The assembled model with muscles at their new path points.
        Must have had initSystem() called.
    muscle_names : list of str
        Names of muscles to rescale.
    reference_model : osim.Model
        The original model with muscles at their original path points,
        used to compute reference path lengths.
        Must have had initSystem() called.

    Returns
    -------
    model : osim.Model
        The model with rescaled muscle properties (modified in-place).
    """
    state = model.initSystem()
    ref_state = reference_model.initSystem()

    for name in muscle_names:
        # New path length
        f_new = model.getForceSet().get(name)
        m_new = osim.Muscle.safeDownCast(f_new)
        new_path_len = m_new.getGeometryPath().getLength(state)

        # Reference path length
        f_ref = reference_model.getForceSet().get(name)
        m_ref = osim.Muscle.safeDownCast(f_ref)
        ref_path_len = m_ref.getGeometryPath().getLength(ref_state)

        if ref_path_len < 1e-8:
            logger.warning(f"  {name}: reference path length is ~0, skipping")
            continue

        scale_factor = new_path_len / ref_path_len

        old_ofl = m_new.getOptimalFiberLength()
        old_tsl = m_new.getTendonSlackLength()
        new_ofl = old_ofl * scale_factor
        new_tsl = old_tsl * scale_factor

        mm = osim.Millard2012EquilibriumMuscle.safeDownCast(f_new)
        mm.set_optimal_fiber_length(new_ofl)
        mm.set_tendon_slack_length(new_tsl)

        logger.info(
            f"  {name}: scale={scale_factor:.3f} "
            f"(path: {ref_path_len:.4f} → {new_path_len:.4f}), "
            f"ofl: {old_ofl:.4f} → {new_ofl:.4f}, "
            f"tsl: {old_tsl:.4f} → {new_tsl:.4f}"
        )

    return model


def _update_gastroc_wraps(model):
    """Update gastroc muscles to use the COMAK condyle wrap.

    Replaces the Rajagopal per-muscle condyle wraps (GasLat_at_condyles_r,
    GasMed_at_condyles_r) with the shared COMAK ellipsoid (Gastroc_at_Condyles_r).
    Keeps the shank wraps (on tibia_r) unchanged.

    Since PathWrap has no setter for the wrap object name, we remove the old
    PathWrap and add a new one referencing the COMAK wrap.
    """
    condyle_wraps_to_replace = {"GasLat_at_condyles_r", "GasMed_at_condyles_r"}
    new_wrap_name = "Gastroc_at_Condyles_r"

    fs = model.getForceSet()
    for name in ["gaslat_r", "gasmed_r"]:
        f = fs.get(name)
        m = osim.Muscle.safeDownCast(f)
        gp = m.updGeometryPath()
        ws = gp.getWrapSet()

        # Find and remove old condyle wrap (reverse order)
        indices_to_remove = []
        for j in range(ws.getSize()):
            if ws.get(j).getWrapObjectName() in condyle_wraps_to_replace:
                logger.info(f"  {name}: removing wrap {ws.get(j).getWrapObjectName()}")
                indices_to_remove.append(j)
        for j in sorted(indices_to_remove, reverse=True):
            ws.remove(j)

        # Add new COMAK condyle wrap via the body's wrap object
        femur_body = model.getBodySet().get("femur_r")
        wrap_obj = femur_body.getWrapObject(new_wrap_name)
        gp.addPathWrap(wrap_obj)
        logger.info(f"  {name}: added wrap → {new_wrap_name}")


def main():
    print("Phase 6B: Add COMAK knee to scaled Rajagopal")
    print("=" * 60)

    # 1. Extract COMAK config from Smith2019
    config = extract_comak_config()

    # 2. Load scaled Rajagopal
    logger.info(f"Loading scaled Rajagopal: {SCALED_RAJAGOPAL}")
    model = osim.Model(SCALED_RAJAGOPAL)
    model.initSystem()
    logger.info(
        f"Rajagopal loaded: {model.getBodySet().getSize()} bodies, "
        f"{model.getJointSet().getSize()} joints, "
        f"{model.getForceSet().getSize()} forces"
    )

    # 3. Extract Rajagopal spanning muscle info BEFORE stripping
    logger.info("Extracting Rajagopal spanning muscle info...")
    raj_muscles = extract_rajagopal_spanning_muscles(model)

    # 4. Strip Rajagopal's knee
    logger.info("Stripping Rajagopal's knee components...")
    model = strip_rajagopal_knee(model)

    # 5. Copy COMAK geometry files
    copy_comak_geometry(SMITH2019_GEOMETRY, RAJAGOPAL_GEOMETRY)

    # 6. Build hybrid spanning muscles (Rajagopal proximal + COMAK patella)
    logger.info("Building hybrid spanning muscles...")
    hybrid_muscles = build_hybrid_spanning_muscles(raj_muscles, config)

    # 7. Add COMAK knee with hybrid spanning muscles
    logger.info("Adding COMAK knee to Rajagopal...")
    config.spanning_muscles = hybrid_muscles
    model = add_comak_knee(
        model,
        config,
        target_joint=None,  # No temp joint to remove
        knee_frame_orientation="rajagopal",  # Rotate COMAK frame to match Rajagopal body frame
    )

    # 8. Update gastroc muscles to use COMAK condyle wrap
    # Gastrocs stay as Rajagopal muscles but swap their condyle wrap reference
    # from Rajagopal's per-muscle wraps to the shared COMAK ellipsoid.
    # Keep the shank wraps (on tibia_r) from Rajagopal unchanged.
    logger.info("Updating gastroc condyle wrap references...")
    _update_gastroc_wraps(model)

    # 9. initSystem (needed before path length computation)
    logger.info("Testing initSystem()...")
    try:
        model.initSystem()
        logger.info("initSystem() SUCCEEDED")
    except Exception as e:
        logger.error(f"initSystem() FAILED: {e}")
        model.printToXML(OUTPUT_MODEL)
        logger.info(f"Model saved (pre-initSystem) to: {OUTPUT_MODEL}")
        return

    # 10. Rescale spanning muscle properties to match new path lengths.
    # The hybrid muscles have Rajagopal's optimal_fiber_length and tendon_slack_length
    # but shorter paths (COMAK patella points are closer than Rajagopal's original
    # patella+tibia path). Without rescaling, the muscles are slack and can't
    # generate force. Use the scaled Rajagopal (pre-strip) as the reference.
    logger.info("Rescaling spanning muscle properties to new path lengths...")
    reference_model = osim.Model(SCALED_RAJAGOPAL)
    model = rescale_muscle_properties_to_path_length(
        model,
        muscle_names=RAJAGOPAL_SPANNING_MUSCLES,
        reference_model=reference_model,
    )

    # 11. Save
    model.printToXML(OUTPUT_MODEL)
    logger.info(f"Model saved to: {OUTPUT_MODEL}")

    # 8. Verify by reloading
    logger.info("Verifying by reloading...")
    model2 = osim.Model(OUTPUT_MODEL)
    model2.initSystem()
    logger.info(
        f"Reload successful: {model2.getBodySet().getSize()} bodies, "
        f"{model2.getJointSet().getSize()} joints, "
        f"{model2.getForceSet().getSize()} forces, "
        f"{model2.getCoordinateSet().getSize()} coordinates"
    )


if __name__ == "__main__":
    main()
