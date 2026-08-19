"""Unit tests for nsosim.scaling.scale_factors."""

from pathlib import Path

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
