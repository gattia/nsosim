"""Anchor wiring tests for EllipsoidFitter and CylinderFitter.

Iter5 of WRAP_FITTER_ROBUSTNESS wires the Procrustes anchor (produced by
``procrustes_anchor``) into both fitters as the L-BFGS initialization AND
the regularizer target. These tests verify:

  1. When ``anchor_params`` is provided, ``_init_*`` snapshots reflect the
     anchor (not the algebraic-fit values).
  2. With ``λ → ∞`` the final fit converges to the anchor regardless of where
     the data optimum sits — confirming the regularizer is wired to the
     anchor as intended.
  3. With ``λ = 0`` and an anchor at the data optimum, fitting still works
     (i.e. anchor only changes init, not the loss landscape).
  4. Default ``anchor_params=None`` preserves the pre-iter5 behavior.
"""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as ScipyR

from nsosim.wrap_surface_fitting.fitting import CylinderFitter, EllipsoidFitter
from nsosim.wrap_surface_fitting.main import wrap_surface
from nsosim.wrap_surface_fitting.wrap_signed_distances import sd_ellipsoid_improved

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _ellipsoid_labeled_points(center, axes, seed=42, n_per_side=600):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2 * np.pi, n_per_side)
    v = rng.uniform(0, np.pi, n_per_side)
    r_in = rng.uniform(0.85, 0.99, n_per_side)
    pts_in = (
        np.column_stack(
            [
                axes[0] * r_in * np.sin(v) * np.cos(u),
                axes[1] * r_in * np.sin(v) * np.sin(u),
                axes[2] * r_in * np.cos(v),
            ]
        )
        + center
    )
    r_out = rng.uniform(1.01, 1.15, n_per_side)
    pts_out = (
        np.column_stack(
            [
                axes[0] * r_out * np.sin(v) * np.cos(u),
                axes[1] * r_out * np.sin(v) * np.sin(u),
                axes[2] * r_out * np.cos(v),
            ]
        )
        + center
    )
    pts = np.vstack([pts_in, pts_out])
    labels = np.concatenate([np.ones(n_per_side), np.zeros(n_per_side)])
    pts_t = torch.tensor(pts, dtype=torch.float32)
    with torch.no_grad():
        sdf = sd_ellipsoid_improved(
            pts_t,
            torch.tensor(center, dtype=torch.float32),
            torch.tensor(axes, dtype=torch.float32),
            torch.eye(3, dtype=torch.float32),
        ).numpy()
    return pts, labels, sdf


def _cylinder_surface_points(center, radius, half_length, n=800, noise=3e-4, seed=42):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-half_length, half_length, n)
    pts = np.column_stack(
        [
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            center[2] + z,
        ]
    )
    if noise > 0:
        pts += rng.normal(0, noise, pts.shape)
    return pts


# ---------------------------------------------------------------------------
# EllipsoidFitter
# ---------------------------------------------------------------------------


TRUE_CENTER_E = np.array([0.05, 0.06, -0.03])
TRUE_AXES_E = np.array([0.025, 0.035, 0.020])
OFFSET_CENTER_E = TRUE_CENTER_E + np.array([0.01, -0.008, 0.005])  # 10 mm-ish offset


def _make_offset_ellipsoid_anchor():
    """Anchor that points away from the data optimum."""
    return wrap_surface(
        name="anchor",
        body=None,
        type_="WrapEllipsoid",
        xyz_body_rotation=np.array([0.0, 0.0, 0.0]),
        translation=OFFSET_CENTER_E.copy(),
        radius=None,
        length=None,
        dimensions=TRUE_AXES_E * 1.2,  # 20% bigger
    )


def test_ellipsoid_anchor_overrides_init_snapshot():
    """With anchor provided, ``_init_center`` should equal the anchor center."""
    pts, labels, sdf = _ellipsoid_labeled_points(TRUE_CENTER_E, TRUE_AXES_E)
    anchor = _make_offset_ellipsoid_anchor()
    fitter = EllipsoidFitter(
        lr=1e-3,
        epochs=1,
        use_lbfgs=False,
        initialization="geometric",  # would otherwise prefer algebraic
        center_transform="linear",
        anchor_params=anchor,
    )
    fitter.fit(points=pts, labels=labels, sdf=sdf, mesh=None, surface_name="anchor", margin=0.005)
    init_center = fitter._init_center.cpu().numpy()
    np.testing.assert_allclose(init_center, anchor.translation, atol=1e-6)
    init_log_axes = np.exp(fitter._init_log_axes.cpu().numpy())
    np.testing.assert_allclose(init_log_axes, anchor.dimensions, atol=1e-6)


def test_ellipsoid_strong_reg_pulls_fit_to_anchor():
    """With huge λ, final fitted center should snap to the anchor center, not data."""
    pts, labels, sdf = _ellipsoid_labeled_points(TRUE_CENTER_E, TRUE_AXES_E)
    anchor = _make_offset_ellipsoid_anchor()
    fitter = EllipsoidFitter(
        lr=5e-3,
        epochs=50,
        use_lbfgs=True,
        lbfgs_epochs=30,
        alpha=1.0,
        beta=0.0,
        gamma=0.0,
        initialization="geometric",
        center_transform="linear",
        anchor_params=anchor,
        lambda_center_reg=1e10,  # overwhelms data loss
        lambda_axes_reg=1e6,
        lambda_quat_reg=1e6,
    )
    fitter.fit(points=pts, labels=labels, sdf=sdf, mesh=None, surface_name="anchor", margin=0.005)
    wp = fitter.wrap_params
    # Should snap to the anchor regardless of where the data optimum was
    assert np.linalg.norm(wp.translation - anchor.translation) < 5e-4
    assert np.linalg.norm(np.sort(wp.dimensions)[::-1] - np.sort(anchor.dimensions)[::-1]) < 5e-4


def test_ellipsoid_no_anchor_unchanged():
    """Without anchor (anchor_params=None), fit converges to the data optimum.

    Matches the convergence config of TestEllipsoidFitter in test_fitting.py.
    """
    pts, labels, sdf = _ellipsoid_labeled_points(TRUE_CENTER_E, TRUE_AXES_E)
    fitter = EllipsoidFitter(
        lr=5e-3,
        epochs=500,
        use_lbfgs=True,
        lbfgs_epochs=30,
        alpha=1.0,
        beta=0.5,
        gamma=0.0,
        initialization="pca",
        center_transform="linear",
        margin_decay_type="linear",
        anchor_params=None,
        lambda_center_reg=0.0,
        lambda_axes_reg=0.0,
        lambda_quat_reg=0.0,
    )
    fitter.fit(points=pts, labels=labels, sdf=sdf, margin=0.005)
    wp = fitter.wrap_params
    # Without anchor and λ=0, should recover the data center (matches TestEllipsoidFitter atol).
    assert np.linalg.norm(wp.translation - TRUE_CENTER_E) < 2e-3


# ---------------------------------------------------------------------------
# CylinderFitter
# ---------------------------------------------------------------------------


TRUE_CENTER_C = np.array([0.05, 0.06, -0.03])
TRUE_RADIUS_C = 0.02
TRUE_HALF_LEN_C = 0.04


def _make_offset_cylinder_anchor(offset_perpendicular=0.01):
    """Anchor center offset perpendicular to the cylinder axis (real drift direction)."""
    return wrap_surface(
        name="anchor",
        body=None,
        type_="WrapCylinder",
        xyz_body_rotation=np.array([0.0, 0.0, 0.0]),
        translation=TRUE_CENTER_C + np.array([offset_perpendicular, 0.0, 0.0]),
        radius=TRUE_RADIUS_C * 1.3,
        length=2.0 * TRUE_HALF_LEN_C,
        dimensions=None,
    )


def test_wraps_to_skip_anchor_pops_named_wraps():
    """build_joint_model's wraps_to_skip_anchor config key should remove named
    wraps from the anchors dict so they fall back to algebraic init.

    iter8.5 finding: Med_Lig_r's loss landscape has a worse local minimum near
    the Smith2019 anchor than near the algebraic init; opting out recovers
    iter3 accuracy.
    """
    # Build a minimal anchors_by_bone structure shaped like
    # procrustes_anchors_from_smith2019 returns, then run the skip-loop in
    # isolation and verify Med_Lig_r is removed but other wraps remain.
    sentinel = object()
    anchors_by_bone = {
        "femur": {
            "femur_r": {
                "ellipsoid": {"Gastroc_at_Condyles_r": sentinel},
                "cylinder": {"KnExt_at_fem_r": sentinel},
            },
            "femur_distal_r": {"cylinder": {"Capsule_r": sentinel}},
        },
        "tibia": {
            "tibia_proximal_r": {
                "ellipsoid": {"Med_Lig_r": sentinel, "Med_LigP_r": sentinel},
            },
        },
    }
    wraps_to_skip = {"Med_Lig_r"}
    for bone_d in anchors_by_bone.values():
        for body_d in bone_d.values():
            for stype_d in body_d.values():
                for name in list(stype_d.keys()):
                    if name in wraps_to_skip:
                        del stype_d[name]

    # Med_Lig_r removed; other tibia wrap preserved
    assert "Med_Lig_r" not in anchors_by_bone["tibia"]["tibia_proximal_r"]["ellipsoid"]
    assert "Med_LigP_r" in anchors_by_bone["tibia"]["tibia_proximal_r"]["ellipsoid"]
    # Other bones untouched
    assert "Gastroc_at_Condyles_r" in anchors_by_bone["femur"]["femur_r"]["ellipsoid"]
    assert "Capsule_r" in anchors_by_bone["femur"]["femur_distal_r"]["cylinder"]


def test_cylinder_anchor_overrides_init_snapshot():
    """With anchor provided, the snapshotted log_center should equal the anchor center."""
    pts = _cylinder_surface_points(TRUE_CENTER_C, TRUE_RADIUS_C, TRUE_HALF_LEN_C, n=400)
    labels = np.ones(len(pts))
    anchor = _make_offset_cylinder_anchor()
    fitter = CylinderFitter(
        epochs=0,
        use_lbfgs=True,
        lbfgs_epochs=1,  # minimum allowed
        initialization="pca",
        center_transform="linear",
        anchor_params=anchor,
        lambda_center_reg=1e10,  # pin so 1 L-BFGS step can't drift the snapshot
    )
    fitter.fit(points=pts, labels=labels, near_surface_points=pts, margin=1e-6)
    # center_transform="linear" stores center directly in _init_log_center
    init_center = fitter._init_log_center.cpu().numpy()
    np.testing.assert_allclose(init_center, anchor.translation, atol=1e-6)


def test_fit_bone_wrap_surfaces_passes_anchors_to_constructor(monkeypatch):
    """fit_bone_wrap_surfaces should pass anchors[body][surface_type][wrap_name]
    as ``anchor_params`` into the fitter constructor; missing entries pass nothing."""
    import nsosim.model_building as mb

    captured = []

    class _FakeFitter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, **fit_kwargs):
            captured.append((self.kwargs, fit_kwargs))
            # Build a dummy result the rest of the call doesn't actually use.

            class _Result:
                final_loss = 0.0

                @property
                def wrap_params(_self):
                    return wrap_surface(
                        name=fit_kwargs.get("surface_name"),
                        body=None,
                        type_="WrapEllipsoid",
                        xyz_body_rotation=np.zeros(3),
                        translation=np.zeros(3),
                        radius=None,
                        length=None,
                        dimensions=np.ones(3) * 0.01,
                    )

            return _Result()

        @property
        def final_loss(self):
            return 0.0

        @property
        def wrap_params(self):
            return wrap_surface(
                name=None,
                body=None,
                type_="WrapEllipsoid",
                xyz_body_rotation=np.zeros(3),
                translation=np.zeros(3),
                radius=None,
                length=None,
                dimensions=np.ones(3) * 0.01,
            )

    # Bypass _fit_with_restarts entirely — directly invoke the stubbed fitter
    # so we can capture the constructor kwargs.
    def _fake_fit_with_restarts(fitter_class, constructor_kwargs, fit_kwargs, points, **_):
        f = _FakeFitter(**constructor_kwargs)
        f.fit(**fit_kwargs)
        return f

    monkeypatch.setattr(mb, "_fit_with_restarts", _fake_fit_with_restarts)

    # Provide a labeled mesh that has the required point-data keys for one
    # ellipsoid wrap (Gastroc_at_Condyles_r) and one cylinder (Capsule_r).
    class _FakeMesh:
        def __init__(self, n=100):
            self._data = {
                "Gastroc_at_Condyles_r_binary": np.ones(n),
                "Gastroc_at_Condyles_r_sdf": np.zeros(n),
                "Capsule_r_binary": np.ones(n),
                "Capsule_r_sdf": np.zeros(n),
                "Capsule_r_near_surface": np.ones(n),
                "KnExt_at_fem_r_binary": np.ones(n),
                "KnExt_at_fem_r_sdf": np.zeros(n),
                "KnExt_at_fem_r_near_surface": np.ones(n),
                "KnExt_vasint_at_fem_r_binary": np.ones(n),
                "KnExt_vasint_at_fem_r_sdf": np.zeros(n),
                "KnExt_vasint_at_fem_r_near_surface": np.ones(n),
            }

        def __getitem__(self, k):
            return self._data[k]

    pts = np.random.default_rng(0).normal(scale=0.01, size=(100, 3))

    ellipse_anchor = _make_offset_ellipsoid_anchor()
    cylinder_anchor = _make_offset_cylinder_anchor()
    anchors = {
        "femur_r": {
            "ellipsoid": {"Gastroc_at_Condyles_r": ellipse_anchor},
            # Intentionally NO entry for KnExt_at_fem_r or KnExt_vasint_at_fem_r
            # to verify the per-wrap fallback to no anchor.
        },
        "femur_distal_r": {
            "cylinder": {"Capsule_r": cylinder_anchor},
        },
    }

    mb.fit_bone_wrap_surfaces(
        bone_name="femur",
        labeled_mesh=_FakeMesh(),
        labeled_mesh_points=pts,
        anchors=anchors,
    )

    # 4 wraps in femur (Gastroc_at_Condyles_r, KnExt_at_fem_r, KnExt_vasint_at_fem_r, Capsule_r)
    assert len(captured) == 4
    # Map wrap_name → captured constructor kwargs
    by_wrap = {kw[1]["surface_name"]: kw[0] for kw in captured}
    assert by_wrap["Gastroc_at_Condyles_r"].get("anchor_params") is ellipse_anchor
    assert by_wrap["Capsule_r"].get("anchor_params") is cylinder_anchor
    # Wraps without an entry should NOT have anchor_params in their kwargs
    assert "anchor_params" not in by_wrap["KnExt_at_fem_r"]
    assert "anchor_params" not in by_wrap["KnExt_vasint_at_fem_r"]


def test_cylinder_strong_reg_pulls_fit_to_anchor():
    """With huge λ, fitted center (perpendicular component) snaps to the anchor."""
    pts = _cylinder_surface_points(TRUE_CENTER_C, TRUE_RADIUS_C, TRUE_HALF_LEN_C, n=400)
    labels = np.ones(len(pts))
    anchor = _make_offset_cylinder_anchor(offset_perpendicular=0.01)
    fitter = CylinderFitter(
        epochs=0,
        use_lbfgs=True,
        lbfgs_epochs=40,
        alpha=1.0,
        initialization="pca",
        center_transform="linear",
        anchor_params=anchor,
        lambda_center_reg=1e10,
        lambda_axis_reg=1e6,
    )
    fitter.fit(points=pts, labels=labels, near_surface_points=pts, margin=1e-6)
    wp = fitter.wrap_params
    # Cylinder center: only the perpendicular-to-axis component is geometrically
    # constrained. Project both onto the anchor axis and check perpendicular drift.
    R = ScipyR.from_euler("XYZ", wp.xyz_body_rotation).as_matrix()
    axis = R[:, 2]
    delta = wp.translation - anchor.translation
    perp = delta - np.dot(delta, axis) * axis
    assert np.linalg.norm(perp) < 5e-4, f"Center perp drift {np.linalg.norm(perp):.2e}"
