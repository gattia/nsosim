"""COMAK body-scaling subject-mass + per-body mass transfer (the post-2026-05-31 fix).

Before the fix, ``scale_comak_model`` transferred only per-body geometric
scale factors from the AB output to the COMAK base — body masses inherited
the base's total mass even though AB sets per-body masses that sum to the
subject's actual measured mass. The result: COMAK simulations on a
multigait cohort applied (e.g.) RSubject_121's gait to bodies with the
wrong total mass AND the wrong per-body distribution.

The fix uses a two-pass strategy:

  Pass 1 — provisional targets:
    - For AB-characterised bodies (non-identity scale_factors AND present
      in the COMAK model): use AB's per-body mass.
    - For all other bodies (knee subbodies, AB-default patella): use
      base_mass × global_ratio.

  Pass 2 — renormalise so the total exactly equals ``subject_mass`` (which
  defaults to the AB total).

These tests pin that behaviour:

  * auto-detect: ``subject_mass=None`` → output total mass equals AB total
  * override:   explicit ``subject_mass`` wins for the total
  * report:     scaling report records subject mass, source, per-body audit
  * per-body:   AB-tuned bodies' final masses preserve AB's proportional
                bias relative to the global-ratio fallback bodies

Slow because each test calls ``scale_comak_model`` end-to-end (ScaleTool +
geometry bake + marker swap + mass transfer).
"""

import json
from pathlib import Path

import opensim as osim
import pytest

from nsosim.scaling import scale_comak_model


def _total_body_mass(p: Path) -> float:
    m = osim.Model(str(p))
    m.initSystem()
    bs = m.getBodySet()
    return float(sum(bs.get(i).getMass() for i in range(bs.getSize())))


def _per_body(p: Path) -> dict:
    m = osim.Model(str(p))
    m.initSystem()
    bs = m.getBodySet()
    return {bs.get(i).getName(): float(bs.get(i).getMass()) for i in range(bs.getSize())}


@pytest.mark.slow
class TestSubjectMassTransfer:
    def test_total_matches_ab_total_by_default(
        self,
        base_comak_path: Path,
        rsubject121_ab_path: Path,
        tmp_path: Path,
    ):
        """With ``subject_mass=None``, output total mass = AB total."""
        ab_total = _total_body_mass(rsubject121_ab_path)

        out_osim = tmp_path / "scaled.osim"
        scale_comak_model(
            base_osim=base_comak_path,
            ab_scaled_osim=rsubject121_ab_path,
            output_osim=out_osim,
            output_geometry_dir=tmp_path / "Geometry",
            mode="WA",
        )

        out_total = _total_body_mass(out_osim)
        assert out_total == pytest.approx(
            ab_total, rel=1e-5
        ), f"output total {out_total} kg should match AB total {ab_total} kg"

    def test_explicit_subject_mass_overrides_total(
        self,
        base_comak_path: Path,
        rsubject121_ab_path: Path,
        tmp_path: Path,
    ):
        """Explicit ``subject_mass`` becomes the final total."""
        target = 99.5
        out_osim = tmp_path / "scaled.osim"
        scale_comak_model(
            base_osim=base_comak_path,
            ab_scaled_osim=rsubject121_ab_path,
            output_osim=out_osim,
            output_geometry_dir=tmp_path / "Geometry",
            mode="WA",
            subject_mass=target,
        )
        assert _total_body_mass(out_osim) == pytest.approx(target, rel=1e-5)

    def test_report_records_audit_fields(
        self,
        base_comak_path: Path,
        rsubject121_ab_path: Path,
        tmp_path: Path,
    ):
        """Scaling report should capture subject mass + per-body audit."""
        out_osim = tmp_path / "scaled.osim"
        report_path = scale_comak_model(
            base_osim=base_comak_path,
            ab_scaled_osim=rsubject121_ab_path,
            output_osim=out_osim,
            output_geometry_dir=tmp_path / "Geometry",
            mode="WA",
        )
        with open(report_path) as f:
            r = json.load(f)
        for k in (
            "subject_mass_kg",
            "subject_mass_source",
            "base_total_mass_kg",
            "output_total_mass_kg",
            "mass_transfer_provisional_total_kg",
            "mass_transfer_renormalize_correction",
            "per_body_mass_audit",
        ):
            assert k in r, f"report missing field: {k}"
        assert r["subject_mass_source"] == "ab_total_body_mass"
        assert r["output_total_mass_kg"] == pytest.approx(r["subject_mass_kg"], rel=1e-5)

    def test_ab_per_body_preserved_in_distribution(
        self,
        base_comak_path: Path,
        rsubject121_ab_path: Path,
        tmp_path: Path,
    ):
        """The AB-characterised bodies should have non-uniform mass scaling
        relative to base — they shouldn't all scale by the same global ratio.

        This is the central per-body-vs-uniform test: with the pre-fix
        ScaleTool.setSubjectMass path, every body's mass ratio = global_ratio.
        Under the new per-body transfer, AB-tuned bodies (e.g. pelvis with
        AB/base ratio 2.6×) should retain a higher final/base ratio than
        bodies that went through the global_ratio fallback (e.g. patella).
        """
        ab_total = _total_body_mass(rsubject121_ab_path)
        base_masses = _per_body(base_comak_path)

        out_osim = tmp_path / "scaled.osim"
        scale_comak_model(
            base_osim=base_comak_path,
            ab_scaled_osim=rsubject121_ab_path,
            output_osim=out_osim,
            output_geometry_dir=tmp_path / "Geometry",
            mode="WA",
        )
        final_masses = _per_body(out_osim)

        # Pelvis: AB-tuned with AB/base = 17.84/6.84 = 2.6
        pelvis_ratio = final_masses["pelvis"] / base_masses["pelvis"]
        # Patella: AB returns identity → falls back to global ratio
        patella_ratio = final_masses["patella_r"] / base_masses["patella_r"]

        # Pelvis ratio should be MUCH higher than patella ratio — that's the
        # whole point of per-body transfer.
        assert pelvis_ratio > patella_ratio * 1.5, (
            f"pelvis ratio {pelvis_ratio:.3f} should be >> patella ratio "
            f"{patella_ratio:.3f}; if not, the AB per-body transfer didn't "
            f"happen (regressed to uniform scaling?)"
        )

    def test_per_body_audit_marks_ab_vs_global(
        self,
        base_comak_path: Path,
        rsubject121_ab_path: Path,
        tmp_path: Path,
    ):
        """Audit dict should label each body's decision path."""
        out_osim = tmp_path / "scaled.osim"
        report_path = scale_comak_model(
            base_osim=base_comak_path,
            ab_scaled_osim=rsubject121_ab_path,
            output_osim=out_osim,
            output_geometry_dir=tmp_path / "Geometry",
            mode="WA",
        )
        with open(report_path) as f:
            audit = json.load(f)["per_body_mass_audit"]

        # Pelvis — AB-tuned, expected ab_per_body
        assert audit["pelvis"]["decision"] == "ab_per_body"
        # Patella — AB returns identity scale, expected global_ratio
        assert audit["patella_r"]["decision"] == "global_ratio"
        # Each audit entry has the four expected keys
        for name, entry in audit.items():
            for k in ("decision", "base_mass_kg", "provisional_kg", "final_kg"):
                assert k in entry, f"body {name} missing audit field {k}"
