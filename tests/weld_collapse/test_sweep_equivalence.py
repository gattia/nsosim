"""Sweep-equivalence test: the collapsed model is physically identical.

The weld collapse changes the multibody-tree *structure* but not the *physics*.
Across a knee range-of-motion sweep, every surviving body's ground transform,
every marker's ground position, every ligament/muscle path length + moment arm,
and total mass + whole-model COM must match the input model.

Measured residual on this fixture is ~1e-15 m (machine precision -- the collapse
is bit-exact). The assertions use ``atol = 1e-10 m``: generous margin over the
measured residual, yet orders of magnitude tighter than any real
coordinate-shift bug (a missed retarget moves a component by >= millimetres).
"""

import numpy as np
import opensim as osim
import pytest

# Equivalence tolerance. Measured sweep residual is ~1e-15 m; 1e-10 leaves
# headroom while still catching any genuine coordinate shift.
ATOL = 1e-10
# Contact-force comparison: the JAM contact solver is separate from kinematics,
# so it gets its own (looser, documented) relative tolerance.
CONTACT_RTOL = 1e-6

DEG = np.pi / 180.0

# Pose grid. Each dict maps coordinate name -> value (radians / metres).
# knee_flex_r is swept; other entries exercise non-axis-aligned, off-default
# poses at the collapsed seam (secondary knee DOFs, patellofemoral, hip, ankle).
_POSES = [{"knee_flex_r": kf} for kf in np.linspace(0.0, 90.0 * DEG, 10)]
_POSES += [
    {
        "knee_flex_r": 35.0 * DEG,
        "knee_add_r": 5.0 * DEG,
        "knee_rot_r": -8.0 * DEG,
        "knee_tx_r": 0.003,
        "knee_ty_r": -0.002,
        "hip_flex_r": 20.0 * DEG,
        "ankle_flex_r": -10.0 * DEG,
    },
    {
        "knee_flex_r": 70.0 * DEG,
        "knee_add_r": -6.0 * DEG,
        "pf_flex_r": 25.0 * DEG,
        "pf_rot_r": 4.0 * DEG,
        "hip_flex_r": -15.0 * DEG,
    },
]

KNEE_COORDS_FOR_MOMENT_ARMS = ["knee_flex_r", "knee_add_r"]


def _geometry_paths(model):
    """Map absolute path -> GeometryPath for every GeometryPath in the model."""
    paths = {}
    for comp in model.getComponentsList():
        if comp.getConcreteClassName() == "GeometryPath":
            paths[comp.getAbsolutePathString()] = osim.GeometryPath.safeDownCast(comp)
    return paths


def _body_names(model):
    bs = model.getBodySet()
    return [bs.get(i).getName() for i in range(bs.getSize())]


def _marker_names(model):
    ms = model.getMarkerSet()
    return [ms.get(i).getName() for i in range(ms.getSize())]


def _apply_pose(model, state, pose):
    """Set the named coordinates, leave the rest at default, assemble + realize."""
    cs = model.getCoordinateSet()
    for name, value in pose.items():
        cs.get(name).setValue(state, float(value), False)
    model.assemble(state)
    model.realizePosition(state)


def _vec3_np(v):
    return np.array([v.get(0), v.get(1), v.get(2)], dtype=float)


@pytest.mark.slow
class TestSweepEquivalence:
    """Input vs. collapsed model, swept through a knee range of motion."""

    @pytest.fixture(scope="class")
    def swept(self, models):
        """Pre-compute, for every pose, the quantities to compare on both models.

        Returns a dict of max residuals so each test method just asserts one.
        """
        m_in, m_out = models
        s_in = m_in.initSystem()
        s_out = m_out.initSystem()

        common_bodies = sorted(set(_body_names(m_in)) & set(_body_names(m_out)))
        common_markers = sorted(set(_marker_names(m_in)) & set(_marker_names(m_out)))
        gp_in = _geometry_paths(m_in)
        gp_out = _geometry_paths(m_out)
        common_paths = sorted(set(gp_in) & set(gp_out))

        res = {
            "body_translation": 0.0,
            "body_rotation": 0.0,
            "marker": 0.0,
            "path_length": 0.0,
            "moment_arm": 0.0,
            "total_mass": 0.0,
            "com": 0.0,
        }
        itb1_seen = False

        for pose in _POSES:
            _apply_pose(m_in, s_in, pose)
            _apply_pose(m_out, s_out, pose)

            for name in common_bodies:
                t_in = m_in.getBodySet().get(name).getTransformInGround(s_in)
                t_out = m_out.getBodySet().get(name).getTransformInGround(s_out)
                res["body_translation"] = max(
                    res["body_translation"],
                    np.abs(_vec3_np(t_in.p()) - _vec3_np(t_out.p())).max(),
                )
                r_in = np.array([[t_in.R().get(i, j) for j in range(3)] for i in range(3)])
                r_out = np.array([[t_out.R().get(i, j) for j in range(3)] for i in range(3)])
                res["body_rotation"] = max(res["body_rotation"], np.abs(r_in - r_out).max())

            for name in common_markers:
                loc_in = m_in.getMarkerSet().get(name).getLocationInGround(s_in)
                loc_out = m_out.getMarkerSet().get(name).getLocationInGround(s_out)
                res["marker"] = max(
                    res["marker"],
                    np.abs(_vec3_np(loc_in) - _vec3_np(loc_out)).max(),
                )

            for name in common_paths:
                if "ITB1" in name:
                    itb1_seen = True
                res["path_length"] = max(
                    res["path_length"],
                    abs(gp_in[name].getLength(s_in) - gp_out[name].getLength(s_out)),
                )

            for coord_name in KNEE_COORDS_FOR_MOMENT_ARMS:
                c_in = m_in.getCoordinateSet().get(coord_name)
                c_out = m_out.getCoordinateSet().get(coord_name)
                for name in common_paths:
                    res["moment_arm"] = max(
                        res["moment_arm"],
                        abs(
                            gp_in[name].computeMomentArm(s_in, c_in)
                            - gp_out[name].computeMomentArm(s_out, c_out)
                        ),
                    )

            res["total_mass"] = max(
                res["total_mass"],
                abs(m_in.getTotalMass(s_in) - m_out.getTotalMass(s_out)),
            )
            res["com"] = max(
                res["com"],
                np.abs(
                    _vec3_np(m_in.calcMassCenterPosition(s_in))
                    - _vec3_np(m_out.calcMassCenterPosition(s_out))
                ).max(),
            )

        res["n_bodies"] = len(common_bodies)
        res["n_markers"] = len(common_markers)
        res["n_paths"] = len(common_paths)
        res["itb1_seen"] = itb1_seen
        return res

    def test_downstream_body_transforms_translation(self, swept):
        assert swept["body_translation"] < ATOL, swept["body_translation"]

    def test_downstream_body_transforms_rotation(self, swept):
        assert swept["body_rotation"] < ATOL, swept["body_rotation"]

    def test_marker_ground_positions(self, swept):
        assert swept["marker"] < ATOL, swept["marker"]

    def test_path_lengths(self, swept):
        assert swept["path_length"] < ATOL, swept["path_length"]

    def test_itb1_path_present_in_comparison(self, swept):
        """ITB1 spans the collapsed seam -- it must be among the compared paths."""
        assert swept["itb1_seen"]

    def test_moment_arms(self, swept):
        """Catches a missed PathWrap or a mis-moved wrap object."""
        assert swept["moment_arm"] < ATOL, swept["moment_arm"]

    def test_total_mass(self, swept):
        assert swept["total_mass"] < ATOL, swept["total_mass"]

    def test_whole_model_com(self, swept):
        assert swept["com"] < ATOL, swept["com"]

    def test_comparison_actually_covered_the_model(self, swept):
        """Guard: the sweep must have compared a substantial set of components."""
        assert swept["n_bodies"] >= 20
        assert swept["n_markers"] >= 1
        assert swept["n_paths"] >= 100

    def test_contact_forces_match_at_a_stance_pose(self, models):
        """At a representative stance pose, JAM contact forces must agree."""
        m_in, m_out = models
        s_in = m_in.initSystem()
        s_out = m_out.initSystem()
        stance = {"knee_flex_r": 15.0 * DEG, "hip_flex_r": 10.0 * DEG}
        for model, state in ((m_in, s_in), (m_out, s_out)):
            cs = model.getCoordinateSet()
            for name, value in stance.items():
                cs.get(name).setValue(state, float(value), False)
            model.assemble(state)
            model.realizeAcceleration(state)

        contact_names = [
            f.getName()
            for f in m_in.getForceSet()
            if "Smith2018ArticularContactForce" in f.getConcreteClassName()
        ]
        assert contact_names, "no Smith2018ArticularContactForce in the model"

        for name in contact_names:
            f_in = m_in.getForceSet().get(name)
            f_out = m_out.getForceSet().get(name)
            v_in = f_in.getOutput("casting_total_contact_force").getValueAsString(s_in)
            v_out = f_out.getOutput("casting_total_contact_force").getValueAsString(s_out)
            # SimTK Vec3 prints as "~[x,y,z]" (the ~ marks a row vector).
            a_in = np.array([float(x) for x in v_in.strip("~()[] ").split(",")])
            a_out = np.array([float(x) for x in v_out.strip("~()[] ").split(",")])
            np.testing.assert_allclose(a_in, a_out, rtol=CONTACT_RTOL, atol=1e-9)
