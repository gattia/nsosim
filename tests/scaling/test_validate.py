"""Tests for nsosim.scaling.validate: AB's generic must be the skeleton being scaled.

The Paper-1 v2 measured arm scaled COMAK by factors AB had fitted to its stock
ArmlessRajagopal (E2E audit F15). These tests pin the check that now raises on
that, the override that reproduces it knowingly, and the landmark definitions.
"""

import json
import shutil
from pathlib import Path

import opensim as osim
import pytest

from nsosim.scaling import scale_comak_model
from nsosim.scaling.validate import (
    GenericSkeletonMismatchError,
    check_ab_generic,
    compare_skeletons,
    measure_skeleton,
)

from .conftest import _base_geometry_has_knee_stls, place_ab_generic

# Body-fixed landmarks (m), measured 2026-10-03 (rerun-v3 sub-plan 01, task 1.1).
COMAK = dict(femur=0.374224, tibia=0.403327, hip=0.092392)
RAJAGOPAL = dict(femur=0.408049, tibia=0.396473, hip=0.077260)


def _write_variant(base: Path, out: Path, edit) -> Path:
    """Save an edited copy of a COMAK model, with its Geometry/ linked beside it."""
    model = osim.Model(str(base))
    model.initSystem()
    edit(model)
    model.finalizeConnections()
    model.printToXML(str(out))
    (out.parent / "Geometry").symlink_to(base.resolve().parent / "Geometry")
    return out


@pytest.fixture
def comak_base(base_comak_path: Path) -> Path:
    """The COMAK base; it loads only with its (untracked) Geometry/ beside it."""
    if not _base_geometry_has_knee_stls(base_comak_path):
        pytest.skip("Reference base Geometry STLs absent")
    return base_comak_path


@pytest.fixture
def ab_models_dir(tmp_path: Path) -> Path:
    """An AB-shaped ``Models/`` folder; the check reads only the generic in it."""
    d = tmp_path / "Models"
    d.mkdir()
    return d


class TestMeasureSkeleton:
    def test_comak_reference(self, comak_base: Path):
        lm = measure_skeleton(comak_base)
        assert lm.femur_len == pytest.approx(COMAK["femur"], abs=1e-5)
        assert lm.tibia_len == pytest.approx(COMAK["tibia"], abs=1e-5)
        assert lm.hip_halfwidth == pytest.approx(COMAK["hip"], abs=1e-5)
        assert lm.knee_joint == "knee_r (CustomJoint)"

    def test_unscaled_generic_tian_fixture_is_rajagopal(self, stripped_tian_path: Path):
        # Despite its name, this fixture is AB's stock generic, not COMAK.
        lm = measure_skeleton(stripped_tian_path)
        assert lm.model_name == "ArmlessRajagopal"
        assert lm.knee_joint == "walker_knee_r (CustomJoint)"
        assert lm.femur_len == pytest.approx(RAJAGOPAL["femur"], abs=1e-5)
        assert lm.tibia_len == pytest.approx(RAJAGOPAL["tibia"], abs=1e-5)
        assert lm.hip_halfwidth == pytest.approx(RAJAGOPAL["hip"], abs=1e-5)

    def test_independent_of_coordinate_defaults(self, comak_base: Path, tmp_path: Path):
        """Landmarks are body-fixed: a flexed, translated default knee changes nothing."""

        def flex(model):
            cs = model.updCoordinateSet()
            for name, value in (("knee_flex_r", 1.0), ("knee_ty_r", -0.01), ("knee_add_r", 0.1)):
                cs.get(name).setDefaultValue(value)

        variant = _write_variant(comak_base, tmp_path / "flexed.osim", flex)
        a, b = measure_skeleton(comak_base), measure_skeleton(variant)
        assert b.femur_vec == pytest.approx(a.femur_vec, abs=1e-12)
        assert b.tibia_vec == pytest.approx(a.tibia_vec, abs=1e-12)


class TestCompareSkeletons:
    def test_rajagopal_fails_all_three(self, comak_base: Path, stripped_tian_path: Path):
        rows = compare_skeletons(measure_skeleton(stripped_tian_path), measure_skeleton(comak_base))
        diffs = {r["landmark"]: r["rel_diff"] for r in rows}
        assert diffs == pytest.approx(
            {"femur": 0.0906, "tibia": 0.0182, "hip_halfwidth": 0.1638}, abs=5e-4
        )
        assert not any(r["ok"] for r in rows)

    def test_offset_tibia_caught_though_length_is_not(self, comak_base: Path, tmp_path: Path):
        """A 12 mm sideways tibia offset barely changes its length; the vector check catches it.

        This is the defect found in the v3 AB upload skeleton (the stripped knee's
        tibia frame at +6 mm instead of -6 mm).
        """

        def shift(model):
            weld = model.updJointSet().get("tibia_tibia_proximal_r")
            frame = osim.PhysicalOffsetFrame.safeDownCast(weld.upd_frames(0))
            frame.set_translation(osim.Vec3(0.018, 0, 0))  # was 0.006

        variant = _write_variant(comak_base, tmp_path / "shifted.osim", shift)
        ref, lm = measure_skeleton(comak_base), measure_skeleton(variant)
        assert abs(lm.tibia_len / ref.tibia_len - 1) < 1e-3  # a length check would pass
        rows = {r["landmark"]: r for r in compare_skeletons(lm, ref)}
        assert rows["tibia"]["rel_diff"] == pytest.approx(0.012 / COMAK["tibia"], abs=1e-4)
        assert not rows["tibia"]["ok"]
        assert rows["femur"]["ok"] and rows["hip_halfwidth"]["ok"]


class TestCheckAbGeneric:
    def test_mismatch_raises_and_names_the_override(
        self, ab_models_dir: Path, comak_base: Path, stripped_tian_path: Path
    ):
        shutil.copy(stripped_tian_path, ab_models_dir / "unscaled_generic.osim")
        with pytest.raises(GenericSkeletonMismatchError) as err:
            check_ab_generic(ab_models_dir / "match_markers_and_physics.osim", comak_base)
        msg = str(err.value)
        assert "allow_generic_mismatch=True" in msg
        assert "reproducing results" in msg  # says when the override is legitimate
        assert "recorded in the scaling report" in msg
        assert "ArmlessRajagopal" in msg and "smith2019" in msg
        for value in ("0.4080", "0.3742", "0.3965", "0.4033", "0.0773", "0.0924"):
            assert value in msg  # both sets of skeleton lengths
        assert msg.count("MISMATCH") == 3

    def test_mismatch_override_is_recorded(
        self, ab_models_dir: Path, comak_base: Path, stripped_tian_path: Path
    ):
        shutil.copy(stripped_tian_path, ab_models_dir / "unscaled_generic.osim")
        record = check_ab_generic(
            ab_models_dir / "match_markers_and_physics.osim",
            comak_base,
            allow_generic_mismatch=True,
        )
        assert record["status"] == "mismatch_overridden"
        assert record["allow_generic_mismatch"] is True
        assert record["ab_generic"]["model_name"] == "ArmlessRajagopal"
        assert record["reference"]["femur_len"] == pytest.approx(COMAK["femur"], abs=1e-5)
        assert [r["ok"] for r in record["landmarks"]] == [False, False, False]
        json.dumps(record)  # goes into the scaling report

    def test_missing_generic_raises_and_names_the_override(
        self, ab_models_dir: Path, comak_base: Path
    ):
        with pytest.raises(GenericSkeletonMismatchError) as err:
            check_ab_generic(ab_models_dir / "match_markers_and_physics.osim", comak_base)
        assert "missing" in str(err.value)
        assert "allow_generic_mismatch=True" in str(err.value)

    def test_missing_generic_override_is_recorded(self, ab_models_dir: Path, comak_base: Path):
        record = check_ab_generic(
            ab_models_dir / "match_markers_and_physics.osim",
            comak_base,
            allow_generic_mismatch=True,
        )
        assert record["status"] == "missing_overridden"
        assert record["ab_generic"] is None

    def test_comak_generic_passes(self, ab_models_dir: Path, comak_base: Path):
        place_ab_generic(ab_models_dir, comak_base)
        record = check_ab_generic(ab_models_dir / "match_markers_and_physics.osim", comak_base)
        assert record["status"] == "match"
        assert all(r["ok"] for r in record["landmarks"])

    def test_real_v2_ab_output_raises(self, rsubject121_ab_path: Path, comak_base: Path):
        """The as-run v2 AB output (Paper-1 measured arm) is rejected by default."""
        with pytest.raises(GenericSkeletonMismatchError, match="ArmlessRajagopal"):
            check_ab_generic(rsubject121_ab_path, comak_base)


class TestScaleComakModelRunsTheCheck:
    def test_raises_before_writing_anything(
        self, ab_models_dir: Path, comak_base: Path, stripped_tian_path: Path, tmp_path: Path
    ):
        shutil.copy(stripped_tian_path, ab_models_dir / "unscaled_generic.osim")
        out_dir = tmp_path / "scaled"
        with pytest.raises(GenericSkeletonMismatchError):
            scale_comak_model(
                base_osim=comak_base,
                ab_scaled_osim=ab_models_dir / "match_markers_and_physics.osim",
                output_osim=out_dir / "scaled.osim",
                output_geometry_dir=out_dir / "Geometry",
            )
        assert not out_dir.exists()

    def test_match_is_recorded_in_report(self, identity_scaled_model: Path):
        report = json.loads(identity_scaled_model.with_suffix(".scaling.json").read_text())
        assert report["ab_generic_check"]["status"] == "match"
        assert report["ab_generic_check"]["allow_generic_mismatch"] is False

    def test_override_is_recorded_in_report(
        self,
        comak_base: Path,
        stripped_tian_path: Path,
        tmp_path: Path,
    ):
        """A v2-style AB output (Rajagopal at identity, beside its Rajagopal generic), with the override."""
        models = tmp_path / "Models"
        models.mkdir()
        ab = models / "match_markers_and_physics.osim"
        shutil.copy(stripped_tian_path, ab)
        shutil.copy(stripped_tian_path, models / "unscaled_generic.osim")
        out_dir = tmp_path / "scaled"
        report_path = scale_comak_model(
            base_osim=comak_base,
            ab_scaled_osim=ab,
            output_osim=out_dir / "scaled.osim",
            output_geometry_dir=out_dir / "Geometry",
            allow_generic_mismatch=True,
        )
        check = json.loads(Path(report_path).read_text())["ab_generic_check"]
        assert check["status"] == "mismatch_overridden"
        assert check["allow_generic_mismatch"] is True
        assert check["ab_generic"]["model_name"] == "ArmlessRajagopal"
        assert len(check["landmarks"]) == 3
