"""Source-model bug fix-ups applied during COMAK body scaling.

These are not scaling artifacts — they are permanent corrections to the base
COMAK model that the legacy pipeline applied post-scaling and that downstream
infrastructure expects to be present.
"""

from typing import Optional

import opensim as osim


def fix_itb1_attachment(model: osim.Model) -> bool:
    """Reparent ITB1's distal attachment from tibia_r to tibia_proximal_r.

    The source model authored the ITB1 distal path-point (Gerdy's tubercle) on
    tibia_r, but the anatomical site sits on the proximal tibia plateau. After
    body scaling, the wrong parent would drift the line of action. This
    function back-corrects the local coordinate by the weld offset so the
    global position is preserved at the moment of reparenting.

    Operates in-place on the model. Caller must `printToXML` to persist.

    Returns True if applied; False if not present or already fixed.
    """
    forces = model.upd_ForceSet()
    try:
        force = forces.get("ITB1")
    except Exception:
        return False

    itb1 = osim.Blankevoort1991Ligament.safeDownCast(force)
    if itb1 is None:
        return False

    pps = itb1.updGeometryPath().getPathPointSet()
    pp_obj = pps.get(2)
    pp = osim.PathPoint.safeDownCast(pp_obj)
    if pp is None:
        return False

    if pp.getParentFrame().getName() != "tibia_r":
        return False  # already fixed upstream

    weld = osim.WeldJoint.safeDownCast(model.getJointSet().get("tibia_tibia_proximal_r"))
    if weld is None:
        return False
    # The weld's parent frame is a PhysicalOffsetFrame on tibia_proximal_r;
    # its translation is the offset of tibia_r origin in tibia_proximal_r coords.
    offset_frame = weld.get_frames(0)
    t = offset_frame.get_translation()

    orig = pp.get_location()
    new_loc = osim.Vec3(orig[0] + t[0], orig[1] + t[1], orig[2] + t[2])

    tibia_proximal = model.getBodySet().get("tibia_proximal_r")
    pp.setParentFrame(tibia_proximal)
    pp.set_location(new_loc)
    return True


def apply_model_fixes(model: osim.Model) -> dict:
    """Run all permanent model fix-ups. Returns a dict of which fixes applied."""
    return {"itb1_fixup_applied": fix_itb1_attachment(model)}


# Convenience for the orchestrator: load → fix → save.
def fix_in_place(scaled_osim_path) -> dict:
    """Load, apply fixes, save back. Returns the per-fix status dict."""
    import pathlib

    p = pathlib.Path(scaled_osim_path)
    model = osim.Model(str(p))
    model.initSystem()
    status = apply_model_fixes(model)
    model.finalizeConnections()
    model.printToXML(str(p))
    return status
