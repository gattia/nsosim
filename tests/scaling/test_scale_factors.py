"""Unit tests for nsosim.scaling.scale_factors."""

import json
from pathlib import Path

import numpy as np
import opensim as osim
import pytest

from nsosim.scaling.config import WA_KNEE_BODIES
from nsosim.scaling.scale_factors import build_scale_set, read_ab_factors


def _ss_to_dict(ss):
    out = {}
    for i in range(ss.getSize()):
        s = ss.get(i)
        sf = s.getScaleFactors()
        out[s.getSegmentName()] = (float(sf[0]), float(sf[1]), float(sf[2]))
    return out


class TestReadAbFactors:
    def test_returns_dict_with_femur_tibia(self, rsubject121_ab_path: Path):
        factors = read_ab_factors(rsubject121_ab_path)
        assert "femur_r" in factors
        assert "tibia_r" in factors

    def test_known_rsubject121_values(self, rsubject121_ab_path: Path):
        factors = read_ab_factors(rsubject121_ab_path)
        # Known values per the plan's worked example.
        assert factors["femur_r"] == pytest.approx((0.9812, 0.9637, 1.0049), abs=1e-3)
        assert factors["tibia_r"] == pytest.approx((0.9610, 1.0140, 0.9410), abs=1e-3)
        assert factors["patella_r"] == pytest.approx((1.0, 1.0, 1.0), abs=1e-6)

    def test_skips_zero_geom_bodies(self, base_comak_path):
        # The unscaled COMAK base has femur_r/tibia_r with no attached_geometry.
        # read_ab_factors must skip these silently.
        factors = read_ab_factors(base_comak_path)
        assert "femur_r" not in factors
        assert "tibia_r" not in factors
        # Bodies that do have attached_geometry should appear
        assert "femur_distal_r" in factors
        assert "tibia_proximal_r" in factors


class TestBuildScaleSetWA:
    def test_rejects_missing_femur_or_tibia(self):
        with pytest.raises(ValueError, match="femur_r or tibia_r"):
            build_scale_set({"pelvis": (1.0, 1.0, 1.0)}, mode="WA")

    def test_identity_yields_identity(self):
        bodies = {
            "femur_r": (1.0, 1.0, 1.0),
            "tibia_r": (1.0, 1.0, 1.0),
            "patella_r": (1.0, 1.0, 1.0),
            "pelvis": (1.0, 1.0, 1.0),
            "torso": (1.0, 1.0, 1.0),
        }
        ss, s_wa = build_scale_set(bodies, mode="WA")
        d = _ss_to_dict(ss)
        for body in bodies:
            assert d[body] == pytest.approx((1.0, 1.0, 1.0), abs=1e-12)
        for knee_body in WA_KNEE_BODIES:
            assert d[knee_body] == pytest.approx((1.0, 1.0, 1.0), abs=1e-12)
        assert s_wa == pytest.approx(1.0, abs=1e-12)

    def test_wa_factor_matches_worked_example(self):
        # NOTE: all three axes are deliberately DISTINCT so this test can tell
        # index 0 / 1 / 2 apart. The previous fixture used (0.945, 0.890, 0.945)
        # -- x == z -- which could not distinguish the mediolateral axis from the
        # anteroposterior one, and so silently passed while LONG_AXIS_INDEX was
        # wrong (see config.py HISTORY).
        ab = {
            "femur_r": (0.945, 0.890, 0.960),
            "tibia_r": (0.920, 0.870, 0.935),
            "patella_r": (1.0, 1.0, 1.0),
        }
        ss, s_wa = build_scale_set(ab, mode="WA")
        d = _ss_to_dict(ss)
        expected = (0.890 + 0.870) / 2  # 0.880 -- mean of the LONG (y) axis
        assert s_wa == pytest.approx(expected, abs=1e-9)
        # and it must not be either transverse axis
        assert s_wa != pytest.approx((0.945 + 0.920) / 2, abs=1e-6)
        assert s_wa != pytest.approx((0.960 + 0.935) / 2, abs=1e-6)
        # AB rows pass through unchanged
        assert d["femur_r"] == pytest.approx((0.945, 0.890, 0.960), abs=1e-9)
        assert d["tibia_r"] == pytest.approx((0.920, 0.870, 0.935), abs=1e-9)
        # patella overridden with WA isotropic
        assert d["patella_r"] == pytest.approx((s_wa, s_wa, s_wa), abs=1e-9)
        # knee subbodies get WA isotropic
        for body in (
            "femur_distal_r",
            "tibia_proximal_r",
            "meniscus_medial_r",
            "meniscus_lateral_r",
        ):
            assert d[body] == pytest.approx((s_wa, s_wa, s_wa), abs=1e-9)

    def test_anisotropic_passthrough_for_non_knee(self):
        ab = {
            "femur_r": (1.0, 1.0, 1.0),
            "tibia_r": (1.0, 1.0, 1.0),
            "torso": (1.5, 0.8, 1.2),
            "talus_r": (0.7, 0.8, 0.9),
        }
        ss, _ = build_scale_set(ab, mode="WA")
        d = _ss_to_dict(ss)
        assert d["torso"] == pytest.approx((1.5, 0.8, 1.2), abs=1e-9)
        assert d["talus_r"] == pytest.approx((0.7, 0.8, 0.9), abs=1e-9)

    def test_patella_gets_wa_when_ab_has_no_patella(self):
        """A stripped COMAK skeleton has no patella_r, so AB returns no factor for it."""
        ab = {
            "femur_r": (0.95, 1.03, 0.97),
            "tibia_r": (0.96, 1.01, 0.94),
            "pelvis": (1.1, 1.0, 1.2),
        }
        ss, s_wa = build_scale_set(ab, mode="WA")
        d = _ss_to_dict(ss)
        assert d["patella_r"] == pytest.approx((s_wa, s_wa, s_wa), abs=1e-12)

    def test_each_body_scaled_once(self):
        """Knee bodies AB happens to return (patella_r, or any other) are not added twice."""
        ab = {
            "femur_r": (1.0, 0.9, 1.0),
            "tibia_r": (1.0, 0.9, 1.0),
            "patella_r": (1.0, 1.0, 1.0),
            "femur_distal_r": (1.2, 1.2, 1.2),
        }
        ss, s_wa = build_scale_set(ab, mode="WA")
        names = [ss.get(i).getSegmentName() for i in range(ss.getSize())]
        assert sorted(names) == sorted(set(names))
        assert _ss_to_dict(ss)["femur_distal_r"] == pytest.approx((s_wa,) * 3, abs=1e-12)


def _path_points_on(osim_path: Path, body: str) -> dict:
    """{(force, point): location} for every path point attached to `body`."""
    model = osim.Model(str(osim_path))
    model.initSystem()
    out = {}
    fs = model.getForceSet()
    for i in range(fs.getSize()):
        force = fs.get(i)
        prop = next((p for p in ("GeometryPath", "path") if force.hasProperty(p)), None)
        path = prop and osim.GeometryPath.safeDownCast(
            force.getPropertyByName(prop).getValueAsObject()
        )
        if path is None:
            continue
        points = path.getPathPointSet()
        for k in range(points.getSize()):
            pp = osim.PathPoint.safeDownCast(points.get(k))
            if pp is not None and pp.getParentFrame().getName() == body:
                loc = pp.get_location()
                out[(force.getName(), pp.getName())] = np.array([loc.get(j) for j in range(3)])
    return out


@pytest.mark.slow
def test_patella_attachments_scale_when_ab_has_no_patella(
    base_comak_path: Path, synthetic_wa_ab_osim: Path, tmp_path: Path, monkeypatch
):
    """Rerun-v3 task 1.3: with no AB patella factor, patella attachments still scale by s_wa.

    Before the fix, build_scale_set added no Scale for patella_r in this case, so
    ScaleTool left the tendon and ligament attachments on it at reference size
    while bake_knee_geometry resized the patella mesh.
    """
    from .conftest import _base_geometry_has_knee_stls

    if not _base_geometry_has_knee_stls(base_comak_path):
        pytest.skip("Reference base Geometry STLs absent")
    from nsosim.scaling import orchestrator

    real = orchestrator.read_ab_factors
    monkeypatch.setattr(
        orchestrator,
        "read_ab_factors",
        lambda p: {k: v for k, v in real(p).items() if k != "patella_r"},
    )
    out = tmp_path / "scaled.osim"
    report = json.loads(
        Path(
            orchestrator.scale_comak_model(
                base_osim=base_comak_path,
                ab_scaled_osim=synthetic_wa_ab_osim,
                output_osim=out,
                output_geometry_dir=tmp_path / "Geometry",
                mode="WA",
            )
        ).read_text()
    )
    assert "patella_r" not in report["ab_factors"]
    s_wa = report["wa_scale"]
    assert s_wa == pytest.approx(0.9, abs=1e-12)

    base, scaled = _path_points_on(base_comak_path, "patella_r"), _path_points_on(out, "patella_r")
    assert len(base) >= 20 and scaled.keys() == base.keys()
    for key, loc in base.items():
        assert scaled[key] == pytest.approx(s_wa * loc, abs=1e-12), key
