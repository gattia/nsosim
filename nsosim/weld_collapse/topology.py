"""Weld-joint topology discovery for the Stage Z weld collapse.

Finds the intermediate (non-root) ``WeldJoint``s that trigger Simbody's slow
gradient code path, and identifies which welded body is the placeholder
sub-body vs. the load-bearing main body.
"""

import opensim as osim

__all__ = ["find_collapsible_welds", "identify_sub_main", "weld_base_bodies"]


def _is_ground(frame: osim.Frame, model: osim.Model) -> bool:
    """True if ``frame`` resolves (through any offset chain) to the model ground."""
    base = frame.findBaseFrame()
    return base.getAbsolutePathString() == model.getGround().getAbsolutePathString()


def weld_base_bodies(model: osim.Model, weld_name: str):
    """Return the two ``Body`` objects a weld connects, as (parent_body, child_body).

    Each is the base body the weld's parent/child frame resolves to (offset
    frames are followed via ``findBaseFrame``). Raises if either side does not
    resolve to a ``Body``.
    """
    joint = model.getJointSet().get(weld_name)
    if osim.WeldJoint.safeDownCast(joint) is None:
        raise ValueError(f"joint '{weld_name}' is not a WeldJoint")

    parent_base = joint.getParentFrame().findBaseFrame()
    child_base = joint.getChildFrame().findBaseFrame()
    parent_body = osim.Body.safeDownCast(parent_base)
    child_body = osim.Body.safeDownCast(child_base)
    if parent_body is None or child_body is None:
        raise ValueError(
            f"weld '{weld_name}' does not connect exactly two bodies "
            f"(parent base '{parent_base.getName()}', "
            f"child base '{child_base.getName()}')"
        )
    return parent_body, child_body


def find_collapsible_welds(model: osim.Model):
    """Return names of every intermediate (non-root) ``WeldJoint`` in the model.

    A ``WeldJoint`` is collapsible iff neither of its connected frames resolves
    to ground -- a root weld (body welded to ground) is not a slow-gradient
    source and is left alone. For ``full_body_healthy_knee.osim`` this returns
    ``['femur_femur_distal_r', 'tibia_tibia_proximal_r']``.

    Order follows the model's ``JointSet`` order, so the result is deterministic.
    """
    welds = []
    joint_set = model.getJointSet()
    for i in range(joint_set.getSize()):
        joint = joint_set.get(i)
        if osim.WeldJoint.safeDownCast(joint) is None:
            continue
        parent_is_ground = _is_ground(joint.getParentFrame(), model)
        child_is_ground = _is_ground(joint.getChildFrame(), model)
        if not parent_is_ground and not child_is_ground:
            welds.append(joint.getName())
    return welds


def identify_sub_main(model: osim.Model, weld_name: str):
    """Return ``(sub_body_name, main_body_name)`` for a weld.

    The placeholder sub-body is the lower-mass of the two welded bodies; the
    main body is the load-bearing other. The weld's parent/child orientation is
    intentionally ignored -- the two welds in the base model have *opposite*
    parent/child order, so mass is the reliable discriminator.

    Raises if the weld does not connect exactly two bodies, or if the two
    bodies have equal mass (ambiguous -- no placeholder to identify).
    """
    parent_body, child_body = weld_base_bodies(model, weld_name)
    if parent_body.getMass() == child_body.getMass():
        raise ValueError(
            f"weld '{weld_name}' connects two equal-mass bodies "
            f"('{parent_body.getName()}', '{child_body.getName()}') -- "
            "cannot identify a placeholder sub-body by mass"
        )
    if parent_body.getMass() < child_body.getMass():
        return parent_body.getName(), child_body.getName()
    return child_body.getName(), parent_body.getName()
