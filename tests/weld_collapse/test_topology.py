"""Tests for nsosim.weld_collapse.topology — weld detection and sub/main ID."""

import opensim as osim
import pytest

from nsosim.weld_collapse.topology import (
    find_collapsible_welds,
    identify_sub_main,
    weld_base_bodies,
)


class TestFindCollapsibleWelds:
    def test_base_model_returns_the_two_knee_welds(self, base_model):
        welds = find_collapsible_welds(base_model)
        assert welds == ["femur_femur_distal_r", "tibia_tibia_proximal_r"]

    def test_root_weld_is_not_collapsible(self):
        """A body welded directly to ground must NOT be reported as collapsible."""
        model = osim.Model()
        body = osim.Body("welded_body", 1.0, osim.Vec3(0), osim.Inertia(1, 1, 1, 0, 0, 0))
        model.addBody(body)
        weld = osim.WeldJoint("root_weld", model.getGround(), body)
        model.addJoint(weld)
        model.finalizeConnections()
        model.initSystem()
        assert find_collapsible_welds(model) == []

    def test_model_with_no_welds_returns_empty(self):
        """A model whose only joint is a non-weld returns no collapsible welds."""
        model = osim.Model()
        body = osim.Body("b", 1.0, osim.Vec3(0), osim.Inertia(1, 1, 1, 0, 0, 0))
        model.addBody(body)
        joint = osim.PinJoint("pin", model.getGround(), body)
        model.addJoint(joint)
        model.finalizeConnections()
        model.initSystem()
        assert find_collapsible_welds(model) == []


class TestIdentifySubMain:
    def test_femur_weld(self, base_model):
        sub, main = identify_sub_main(base_model, "femur_femur_distal_r")
        assert (sub, main) == ("femur_distal_r", "femur_r")

    def test_tibia_weld(self, base_model):
        """Tibia weld has the opposite parent/child order; mass still wins."""
        sub, main = identify_sub_main(base_model, "tibia_tibia_proximal_r")
        assert (sub, main) == ("tibia_proximal_r", "tibia_r")

    def test_sub_body_is_lower_mass(self, base_model):
        for weld in ("femur_femur_distal_r", "tibia_tibia_proximal_r"):
            sub, main = identify_sub_main(base_model, weld)
            bodies = base_model.getBodySet()
            assert bodies.get(sub).getMass() < bodies.get(main).getMass()

    def test_non_weld_joint_raises(self, base_model):
        with pytest.raises(ValueError, match="not a WeldJoint"):
            weld_base_bodies(base_model, "knee_r")

    def test_equal_mass_weld_is_ambiguous(self):
        """Two equal-mass welded bodies cannot be told apart -- must raise."""
        model = osim.Model()
        inertia = osim.Inertia(1, 1, 1, 0, 0, 0)
        b1 = osim.Body("b1", 2.0, osim.Vec3(0), inertia)
        b2 = osim.Body("b2", 2.0, osim.Vec3(0), inertia)
        model.addBody(b1)
        model.addBody(b2)
        model.addJoint(osim.PinJoint("pin", model.getGround(), b1))
        model.addJoint(osim.WeldJoint("mid_weld", b1, b2))
        model.finalizeConnections()
        model.initSystem()
        with pytest.raises(ValueError, match="equal-mass"):
            identify_sub_main(model, "mid_weld")
