"""``mass_rule="volume"``: per-body masses by OpenSim's volume rule (comak rerun-v3, decisions D4a/D4b).

The test list is the spec in comak ``.claude/reports/rerun_v3/mass_decisions.md`` ("Code spec for 01 task 1.5"):

  (i)   the masses sum to the subject mass;
  (ii)  with an identity scale set every body keeps the same final/base ratio;
  (iii) the rule IS AddBiomechanics' rule: applied to AB's own generic with AB's factors it reproduces AB's
        ``match_markers_and_physics.osim`` masses to machine precision (real AB outputs; skipped when absent);
  (iv)  the knee sub-bodies, patella and menisci end at base * s_wa**3 * c;
  (v)   bodies AB models keep AB's *ratios* to each other exactly (their absolute masses are smaller than AB's,
        because the bodies AB lacks take their share out of the same total).

Plus: the default (``mass_rule="as_run"``) is the v2 code path, unchanged; the two rules give the same geometry
and the same inertia per unit mass; and ``"volume"`` refuses ``preserve_mass_distribution=False``.
"""

import hashlib
import json
from pathlib import Path

import opensim as osim
import pytest

from nsosim.scaling import WA_KNEE_BODIES, scale_comak_model, volume_rule_masses
from nsosim.scaling.scale_factors import read_ab_factors

from .conftest import ARMLESS_TRC, _base_geometry_has_knee_stls, make_comak_ab_output

UNTRACKED_AB = Path(__file__).resolve().parents[2] / "untracked" / "ab_outputs"
# Real AB outputs: (Models folder, label). Subject_119_v7 is AB on the stripped COMAK skeleton (v7); RSubject_121 is
# a Paper-1 v2 output on AB's ArmlessRajagopal.
REAL_AB = (
    (UNTRACKED_AB / "Subject_119_v7" / "Models", "COMAK v7 skeleton"),
    (UNTRACKED_AB / "RSubject_121" / "subject_data" / "Models", "v2 ArmlessRajagopal"),
)

# Non-trivial, anisotropic factors on the bodies a stripped COMAK skeleton gives AB.
FACTORS = {
    "pelvis": (1.08, 1.02, 1.15),
    "femur_r": (1.01, 1.04, 0.97),
    "tibia_r": (0.99, 1.06, 1.02),
    "femur_l": (1.02, 1.03, 0.98),
    "tibia_l": (1.00, 1.05, 1.01),
    "torso": (0.94, 1.01, 0.90),
    "calcn_r": (1.05, 0.95, 1.00),
}
SUBJECT_MASS = 80.0
RECORDED_MASS = 79.2


def _masses(path: Path) -> dict:
    m = osim.Model(str(path))
    m.initSystem()
    bs = m.getBodySet()
    return {bs.get(i).getName(): float(bs.get(i).getMass()) for i in range(bs.getSize())}


def _inertia(path: Path) -> dict:
    m = osim.Model(str(path))
    m.initSystem()
    bs = m.getBodySet()
    out = {}
    for i in range(bs.getSize()):
        inertia = bs.get(i).getInertia()
        out[bs.get(i).getName()] = [inertia.getMoments()[k] for k in range(3)] + [
            inertia.getProducts()[k] for k in range(3)
        ]
    return out


def _volume(sf) -> float:
    return sf[0] * sf[1] * sf[2]


def _md5_tree(folder: Path) -> dict:
    return {
        p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(Path(folder).iterdir()) if p.is_file()
    }


# ----------------------------------------------------------------- pure rule ----


def test_volume_rule_sums_to_total():
    """(i) on the pure function."""
    base = {"a": 2.0, "b": 5.0, "c": 0.5}
    final, c = volume_rule_masses(base, {"a": 1.1**3, "b": 0.9}, 70.0)
    assert sum(final.values()) == pytest.approx(70.0, rel=1e-12)
    assert final["c"] == pytest.approx(0.5 * c, rel=1e-15)  # no factor -> V = 1


def test_volume_rule_identity_is_uniform():
    """(ii) with no scaling every body takes the same ratio, total / base total."""
    base = {"a": 2.0, "b": 5.0, "c": 0.5}
    final, c = volume_rule_masses(base, {"a": 1.0, "b": 1.0, "c": 1.0}, 15.0)
    assert {k: final[k] / base[k] for k in base} == pytest.approx({k: 2.0 for k in base}, rel=1e-12)
    assert c == pytest.approx(2.0, rel=1e-12)


@pytest.mark.parametrize("models,label", REAL_AB, ids=[r[1] for r in REAL_AB])
def test_volume_rule_reproduces_ab_masses(models: Path, label: str):
    """(iii) AB's per-body masses are the volume rule on AB's generic with AB's factors."""
    ab, generic = models / "match_markers_and_physics.osim", models / "unscaled_generic.osim"
    if not (ab.is_file() and generic.is_file()):
        pytest.skip(f"real AB output absent ({label}): {models}")
    factors = read_ab_factors(ab, generic_osim=generic)
    ab_masses, generic_masses = _masses(ab), _masses(generic)
    final, _ = volume_rule_masses(
        generic_masses, {k: _volume(v) for k, v in factors.items()}, sum(ab_masses.values())
    )
    assert set(final) == set(ab_masses)
    for name, m in ab_masses.items():
        assert final[name] == pytest.approx(m, rel=1e-12, abs=1e-15), f"{label}: {name}"


# ------------------------------------------------------------ orchestrator ----


@pytest.fixture(scope="session")
def mass_rule_ab(base_comak_path: Path, tmp_path_factory) -> Path:
    if not _base_geometry_has_knee_stls(base_comak_path):
        pytest.skip("Reference base Geometry STLs absent")
    return make_comak_ab_output(tmp_path_factory.mktemp("mass_rule_ab"), FACTORS)


def _scale(base: Path, ab: Path, out_dir: Path, **kwargs) -> tuple:
    out = out_dir / "scaled.osim"
    report = scale_comak_model(
        base_osim=base,
        ab_scaled_osim=ab,
        output_osim=out,
        output_geometry_dir=out_dir / "Geometry",
        mode="WA",
        subject_mass=SUBJECT_MASS,
        recorded_mass_kg=RECORDED_MASS,
        **kwargs,
    )
    return out, json.loads(Path(report).read_text())


@pytest.fixture(scope="session")
def volume_scaled(base_comak_path: Path, mass_rule_ab: Path, tmp_path_factory):
    return _scale(base_comak_path, mass_rule_ab, tmp_path_factory.mktemp("volume"), mass_rule="volume")


@pytest.fixture(scope="session")
def as_run_scaled(base_comak_path: Path, mass_rule_ab: Path, tmp_path_factory):
    return _scale(base_comak_path, mass_rule_ab, tmp_path_factory.mktemp("as_run"), mass_rule="as_run")


@pytest.fixture(scope="session")
def default_scaled(base_comak_path: Path, mass_rule_ab: Path, tmp_path_factory):
    return _scale(base_comak_path, mass_rule_ab, tmp_path_factory.mktemp("default"))


@pytest.mark.slow
class TestVolumeRule:
    def test_total_equals_subject_mass(self, volume_scaled):
        """(i)"""
        out, report = volume_scaled
        assert sum(_masses(out).values()) == pytest.approx(SUBJECT_MASS, rel=1e-9)
        assert report["output_total_mass_kg"] == pytest.approx(SUBJECT_MASS, rel=1e-9)

    def test_knee_bodies_take_s_wa_cubed(self, base_comak_path, volume_scaled):
        """(iv) knee sub-bodies, patella and menisci end at base * s_wa**3 * c."""
        out, report = volume_scaled
        base, final = _masses(base_comak_path), _masses(out)
        s_wa, c = report["wa_scale"], report["mass_normalisation_c"]
        assert s_wa == pytest.approx((FACTORS["femur_r"][1] + FACTORS["tibia_r"][1]) / 2, rel=1e-12)
        for name in WA_KNEE_BODIES:
            assert final[name] == pytest.approx(base[name] * s_wa**3 * c, rel=1e-9), name

    def test_ab_bodies_keep_ab_ratios(self, volume_scaled):
        """(v) bodies AB models keep the ratios AB itself would give them; their absolutes are smaller."""
        out, report = volume_scaled
        final = _masses(out)
        generic = _masses(ARMLESS_TRC)  # what AB would scale: the stripped COMAK skeleton
        ab_rule, _ = volume_rule_masses(
            generic, {k: _volume(v) for k, v in FACTORS.items()}, SUBJECT_MASS
        )
        # This fixture's AB skeleton is the armless COMAK model *with* its knee bodies (a stripped upload has none);
        # whatever AB returns for those, they take s_wa, so they are checked by (iv), not here.
        ab_bodies = [
            b
            for b, entry in report["per_body_mass_audit"].items()
            if entry["in_ab_skeleton"] and b not in WA_KNEE_BODIES
        ]
        assert "pelvis" in ab_bodies and "humerus_r" not in ab_bodies
        for name in ab_bodies:
            assert final[name] / final["pelvis"] == pytest.approx(
                ab_rule[name] / ab_rule["pelvis"], rel=1e-9
            ), name
            assert final[name] < ab_rule[name]  # the arms and knee bodies are carved out, not added on top

    def test_arms_keep_volume_factor_one(self, base_comak_path, volume_scaled):
        out, report = volume_scaled
        base, final = _masses(base_comak_path), _masses(out)
        for name in ("humerus_r", "ulna_l", "hand_r"):
            assert final[name] == pytest.approx(base[name] * report["mass_normalisation_c"], rel=1e-9)
            assert report["per_body_mass_audit"][name]["volume_factor"] == 1.0

    def test_report_fields(self, volume_scaled):
        _, report = volume_scaled
        assert report["mass_rule"] == "volume"
        assert "mass_transfer_renormalize_correction" not in report  # no v2-style number to misread
        assert 0.05 < report["mass_fraction_in_bodies_absent_from_ab"] < 0.2
        assert report["recorded_mass_kg"] == RECORDED_MASS
        assert report["ab_total_minus_recorded_kg"] == pytest.approx(
            report["ab_total_mass_kg"] - RECORDED_MASS, rel=1e-12
        )
        for entry in report["per_body_mass_audit"].values():
            assert set(entry) == {"base_kg", "volume_factor", "final_kg", "in_ab_skeleton"}

    def test_same_geometry_and_inertia_per_mass_as_as_run(self, volume_scaled, as_run_scaled):
        """Only the masses differ: the baked geometry is identical, and inertia / mass is ScaleTool's geometric
        scaling in both."""
        (vol, _), (old, _) = volume_scaled, as_run_scaled
        assert _md5_tree(vol.parent / "Geometry") == _md5_tree(old.parent / "Geometry")
        mv, mo, iv, io = _masses(vol), _masses(old), _inertia(vol), _inertia(old)
        for name in mv:
            if mo[name] > 1e-12:
                assert [x / mv[name] for x in iv[name]] == pytest.approx(
                    [x / mo[name] for x in io[name]], rel=1e-9, abs=1e-15
                ), name

    def test_default_is_as_run(self, default_scaled, as_run_scaled):
        """Switch off (the default) is the v2 code path: the model file is byte-identical to mass_rule='as_run'
        and the report carries the v2 mass fields."""
        (default, report), (as_run, _) = default_scaled, as_run_scaled
        assert default.read_bytes() == as_run.read_bytes()
        assert report["mass_rule"] == "as_run"
        assert "mass_transfer_renormalize_correction" in report

    def test_volume_requires_preserved_mass_distribution(self, base_comak_path, mass_rule_ab, tmp_path):
        with pytest.raises(ValueError, match="preserve_mass_distribution"):
            _scale(base_comak_path, mass_rule_ab, tmp_path, mass_rule="volume", preserve_mass_distribution=False)
