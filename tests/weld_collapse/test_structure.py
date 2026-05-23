"""Structural tests for the collapsed model: no welds, correct counts, reparenting."""

import opensim as osim

PLACEHOLDER_BODIES = {"femur_distal_r", "tibia_proximal_r"}


def _joint_names(model):
    js = model.getJointSet()
    return {js.get(i).getName() for i in range(js.getSize())}


def _body_names(model):
    bs = model.getBodySet()
    return {bs.get(i).getName() for i in range(bs.getSize())}


def _coordinate_names(model):
    cs = model.getCoordinateSet()
    return {cs.get(i).getName() for i in range(cs.getSize())}


def _n_weldjoints(model):
    js = model.getJointSet()
    return sum(1 for i in range(js.getSize()) if osim.WeldJoint.safeDownCast(js.get(i)) is not None)


class TestCollapsedStructure:
    def test_no_weldjoint_remains(self, models):
        _, collapsed = models
        assert _n_weldjoints(collapsed) == 0

    def test_input_has_two_intermediate_welds(self, models):
        """Guards the test premise: the base model really does carry 2 welds."""
        base, _ = models
        assert _n_weldjoints(base) == 2

    def test_exactly_placeholder_bodies_removed(self, models):
        base, collapsed = models
        removed = _body_names(base) - _body_names(collapsed)
        assert removed == PLACEHOLDER_BODIES

    def test_all_other_bodies_preserved(self, models):
        base, collapsed = models
        # Every non-placeholder body in the input survives.
        assert _body_names(base) - PLACEHOLDER_BODIES == _body_names(collapsed)

    def test_two_joints_removed(self, models):
        base, collapsed = models
        assert collapsed.getJointSet().getSize() == base.getJointSet().getSize() - 2

    def test_removed_joints_are_the_welds(self, models):
        base, collapsed = models
        removed = _joint_names(base) - _joint_names(collapsed)
        assert removed == {"femur_femur_distal_r", "tibia_tibia_proximal_r"}

    def test_coordinates_preserved_exactly(self, models):
        """The collapse must not add, drop, or rename any coordinate."""
        base, collapsed = models
        assert _coordinate_names(collapsed) == _coordinate_names(base)

    def test_knee_r_reparented_onto_femur_and_tibia(self, models):
        _, collapsed = models
        knee = collapsed.getJointSet().get("knee_r")
        assert knee.getParentFrame().findBaseFrame().getName() == "femur_r"
        assert knee.getChildFrame().findBaseFrame().getName() == "tibia_r"

    def test_pf_r_reparented_onto_femur(self, models):
        _, collapsed = models
        pf = collapsed.getJointSet().get("pf_r")
        assert pf.getParentFrame().findBaseFrame().getName() == "femur_r"
        assert pf.getChildFrame().findBaseFrame().getName() == "patella_r"

    def test_offset_frames_created_on_main_bodies(self, models):
        """Each collapse leaves one PhysicalOffsetFrame on the main body."""
        _, collapsed = models
        assert collapsed.hasComponent("/bodyset/femur_r/femur_distal_r_collapsed")
        assert collapsed.hasComponent("/bodyset/tibia_r/tibia_proximal_r_collapsed")

    def test_report_counts(self, collapsed_osim):
        _, _, report = collapsed_osim
        assert report["n_welds_collapsed"] == 2
        assert sorted(report["welds_collapsed"]) == [
            "femur_femur_distal_r",
            "tibia_tibia_proximal_r",
        ]
        per = {r["weld"]: r for r in report["per_weld"]}
        assert per["femur_femur_distal_r"]["sub_body"] == "femur_distal_r"
        assert per["femur_femur_distal_r"]["wraps_moved"] == ["Capsule_r"]
        assert per["tibia_tibia_proximal_r"]["sub_body"] == "tibia_proximal_r"
        assert set(per["tibia_tibia_proximal_r"]["wraps_moved"]) == {
            "Med_Lig_r",
            "Med_LigP_r",
        }
