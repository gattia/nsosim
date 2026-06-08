"""Build-then-scale (Mode 3 / Mode 5): scale an already-built personalized knee.

The expensive NSM fit (GPU) builds a Mode-1 model once — a personalized recon
knee in a reference-size full-body COMAK model. To put that knee into a
gait-scaled body for any number of gait subjects, run ``scale_comak_model`` on
the *built* model itself (``base_osim`` = the built model) instead of on the
generic reference base. By the time the model is built, both knee-geometry
generators have run and their outputs are attached to the knee bodies, so the
ordinary body-scaling machinery scales the whole knee:

  * ``bake_knee_geometry`` bakes *whatever STL each knee body points at* — here
    the recon STLs — by ``s_wa`` about the joint-center origin;
  * OpenSim ScaleTool scales the wraps, ligaments, joint frames, muscles, and
    the patella placement offset (``pf_tx/ty/tz_r``) by ``s_wa``.

These tests confirm every knee component scales by ``s_wa`` on a *real* built
model (a personalized OAI knee scaled to a different subject's gait body — the
Pathway-B scenario), and that coherence (cartilage-bone proximity, ligament
reference strains) survives the scale exactly as it does for the reference
knee. See ``docs/deviations.md`` Mode 3.

The built model is too large for git, so the recon end-to-end tests skip when it
is absent (``built_mode1_model_path`` fixture). The same scaling operation is
guarded without that fixture by ``TestReferenceOriginScalingPlacement`` below
(reference base + an in-repo-derived s_wa) and by ``test_nontrivial.py`` (Mode 2)
— the operation is identical; only the attached geometry differs (recon vs
reference).
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import opensim as osim
import pymskt as mskt
import pytest

from nsosim.scaling.scale_factors import read_ab_factors

# Reuse the coherence helpers that already guard Mode 2 — the checks are
# identical, only the base model differs.
from .test_nontrivial import _cart_to_bone_distances, _gather_blankevoort

KNEE_BODIES = (
    "femur_distal_r",
    "tibia_proximal_r",
    "patella_r",
    "meniscus_medial_r",
    "meniscus_lateral_r",
)

PF_KNEE_COORDS = ("pf_tx_r", "pf_ty_r", "pf_tz_r")

# Structural path-point counts on the knee bodies of a built Mode-1 model
# (measured on the 9003175 fixture: 181 Blankevoort ligament points + 5 muscle
# points). Scaling is a pure resize and must preserve the point set exactly, so
# the tests assert before==after set-equality (catches any silent drop/add) AND
# a floor at these counts. A loose floor like ">= 50" would let a silent loss of
# 100+ points pass unnoticed.
N_KNEE_LIGAMENT_POINTS = 181
N_KNEE_MUSCLE_POINTS = 5

# Tolerances are pinned to the MEASURED numerical precision of each mechanism
# (verified on the real built model, RSubject_121 s_wa), NOT loose round numbers.
# Use direct abs() comparisons, never pytest.approx(abs=...): approx keeps a
# default rel=1e-6 that would silently override any tighter abs bound.
#
#   mechanism                                   measured dev      tol (margin)
#   direct coord scaling (points, wrap trans/   ~2.2e-16          1e-12  (exact)
#     size/rotation, patella offset, strain)    (machine eps)
#   pure-knee slack ratio (path/wrap solver)    1.9e-6            1e-5
#   STL vertex (ASCII mesh I/O), meters         1.5e-8 m          1e-7 m
#   cart-bone proximity aggregate, meters       ~2e-10 m          1e-8 m
#   body-centroid placement ratio (STL mean)    ~4e-7 (est)       1e-5
TOL_EXACT = 1e-12  # direct-coordinate scaling: clean to ~machine epsilon
TOL_SLACK = 1e-5  # path-solver-derived slack ratio vs s_wa
TOL_STL_M = 1e-7  # per-vertex STL scale error, meters
TOL_CARTBONE_M = 1e-8  # cart-bone mean/p95 scaling residual, meters
TOL_CENTROID = 1e-5  # STL-mean body-centroid placement ratio vs s_wa


# --- helpers ----------------------------------------------------------------


def _geom_dir(model_path: Path) -> Path:
    return Path(model_path).parent / "Geometry"


def _knee_stl_files(model_path: Path) -> List[str]:
    """Every STL attached to (or contact-meshed on) a knee body."""
    m = osim.Model(str(model_path))
    m.initSystem()
    files = set()
    bs = m.getBodySet()
    for bi in range(bs.getSize()):
        body = bs.get(bi)
        if body.getName() not in KNEE_BODIES:
            continue
        for gi in range(body.getPropertyByName("attached_geometry").size()):
            mesh = osim.Mesh.safeDownCast(body.upd_attached_geometry(gi))
            if mesh is not None and mesh.get_mesh_file():
                files.add(mesh.get_mesh_file())
    cg = m.getContactGeometrySet()
    for i in range(cg.getSize()):
        g = cg.get(i)
        if g.getConcreteClassName() != "Smith2018ContactMesh":
            continue
        scm = osim.Smith2018ContactMesh.safeDownCast(g)
        if scm.getFrame().getName() in KNEE_BODIES:
            files.add(scm.get_mesh_file())
    return sorted(files)


def _sorted_vertex_norms(stl_path: Path) -> np.ndarray:
    """Sorted ||v|| of every vertex. Order-invariant — robust to the vertex
    reordering an STL read->write round-trip can introduce (see
    bake_knee_geometry docstring), while still being an exact per-vertex
    fingerprint of an origin-centered scale."""
    m = mskt.mesh.Mesh(str(stl_path))
    return np.sort(np.linalg.norm(np.asarray(m.point_coords), axis=1))


def _knee_centroid_by_body(model_path: Path) -> Dict[str, np.ndarray]:
    """Body-local centroid of each knee body's combined geometry (STL vertices
    with the model's stored scale_factors applied)."""
    geom = _geom_dir(model_path)
    m = osim.Model(str(model_path))
    m.initSystem()
    out: Dict[str, np.ndarray] = {}
    bs = m.getBodySet()
    # attached visual geometry
    body_files: Dict[str, set] = {b: set() for b in KNEE_BODIES}
    for bi in range(bs.getSize()):
        body = bs.get(bi)
        if body.getName() not in KNEE_BODIES:
            continue
        for gi in range(body.getPropertyByName("attached_geometry").size()):
            mesh = osim.Mesh.safeDownCast(body.upd_attached_geometry(gi))
            if mesh is not None and mesh.get_mesh_file():
                body_files[body.getName()].add(mesh.get_mesh_file())
    # contact meshes (menisci carry only contact STLs)
    cg = m.getContactGeometrySet()
    for i in range(cg.getSize()):
        g = cg.get(i)
        if g.getConcreteClassName() != "Smith2018ContactMesh":
            continue
        scm = osim.Smith2018ContactMesh.safeDownCast(g)
        fr = scm.getFrame().getName()
        if fr in KNEE_BODIES and scm.get_mesh_file():
            body_files[fr].add(scm.get_mesh_file())

    for body_name, files in body_files.items():
        pts = []
        for f in files:
            p = geom / f
            if p.is_file():
                pts.append(np.asarray(mskt.mesh.Mesh(str(p)).point_coords))
        if pts:
            out[body_name] = np.concatenate(pts, axis=0).mean(axis=0)
    return out


def _gather_knee_wraps(model_path: Path) -> Dict[str, dict]:
    """{wrap_name: {body, translation, size, rotation}} for wraps on knee bodies.
    `size` is the cylinder radius or the mean ellipsoid dimension; `rotation`
    is the wrap's xyz_body_rotation (radians)."""
    m = osim.Model(str(model_path))
    m.initSystem()
    out: Dict[str, dict] = {}
    bs = m.getBodySet()
    for bi in range(bs.getSize()):
        body = bs.get(bi)
        if body.getName() not in KNEE_BODIES:
            continue
        wos = body.getWrapObjectSet()
        for wi in range(wos.getSize()):
            w = wos.get(wi)
            t = w.get_translation()
            r = w.get_xyz_body_rotation()
            cyl = osim.WrapCylinder.safeDownCast(w)
            ell = osim.WrapEllipsoid.safeDownCast(w)
            if cyl is not None:
                size = float(cyl.get_radius())
            elif ell is not None:
                d = ell.get_dimensions()
                size = float((d[0] + d[1] + d[2]) / 3.0)
            else:
                size = float("nan")
            out[w.getName()] = {
                "body": body.getName(),
                "translation": np.array([t[0], t[1], t[2]]),
                "size": size,
                "rotation": np.array([r[0], r[1], r[2]]),
            }
    return out


def _gather_knee_ligament_slacks(model_path: Path) -> Dict[str, dict]:
    """{lig_name: {slack, all_knee}} for every Blankevoort ligament. ``all_knee``
    is True only when *every* path point is on a knee body — those slacks must
    scale by exactly s_wa. A ligament spanning a knee body and an
    anisotropically-scaled parent (e.g. ITB1: pelvis -> femur_r -> tibia) scales
    its slack by the actual mixed path-length change, NOT clean s_wa — for those,
    the invariant is reference strain preserved, not slack x s_wa."""
    m = osim.Model(str(model_path))
    m.initSystem()
    fs = m.upd_ForceSet()
    out: Dict[str, dict] = {}
    for i in range(fs.getSize()):
        lig = osim.Blankevoort1991Ligament.safeDownCast(fs.get(i))
        if lig is None:
            continue
        pps = lig.updGeometryPath().getPathPointSet()
        frames = [pps.get(pi).getParentFrame().getName() for pi in range(pps.getSize())]
        out[fs.get(i).getName()] = {
            "slack": float(lig.get_slack_length()),
            "all_knee": all(f in KNEE_BODIES for f in frames),
        }
    return out


def _gather_knee_muscle_points(model_path: Path) -> Dict[str, np.ndarray]:
    """{muscle::point: body-local location} for muscle path points whose parent
    frame is a knee body (e.g. the quadriceps inserting on the patella)."""
    m = osim.Model(str(model_path))
    m.initSystem()
    fs = m.upd_ForceSet()
    out: Dict[str, np.ndarray] = {}
    for i in range(fs.getSize()):
        musc = osim.Muscle.safeDownCast(fs.get(i))
        if musc is None:
            continue
        pps = musc.getGeometryPath().getPathPointSet()
        for pi in range(pps.getSize()):
            p = pps.get(pi)
            if p.getParentFrame().getName() not in KNEE_BODIES:
                continue
            pp = osim.PathPoint.safeDownCast(p)
            if pp is None:
                continue
            loc = pp.get_location()
            out[f"{musc.getName()}::{p.getName()}"] = np.array([loc[0], loc[1], loc[2]])
    return out


def _gather_knee_ligament_points(model_path: Path) -> Dict[str, np.ndarray]:
    """{lig::point: body-local location} for Blankevoort path points whose
    parent frame is a knee body."""
    m = osim.Model(str(model_path))
    m.initSystem()
    out: Dict[str, np.ndarray] = {}
    for comp in m.getComponentsList():
        if "Blankevoort1991Ligament" not in comp.getConcreteClassName():
            continue
        try:
            path_obj = comp.getGeometryPath()
        except Exception:
            continue
        pps = path_obj.getPathPointSet()
        for pi in range(pps.getSize()):
            p = pps.get(pi)
            if p.getParentFrame().getName() not in KNEE_BODIES:
                continue
            pp = osim.PathPoint.safeDownCast(p)
            if pp is None:
                continue
            loc = pp.get_location()
            out[f"{comp.getName()}::{p.getName()}"] = np.array([loc[0], loc[1], loc[2]])
    return out


def _coord_defaults(model_path: Path, names) -> Dict[str, float]:
    m = osim.Model(str(model_path))
    m.initSystem()
    cs = m.getCoordinateSet()
    return {n: float(cs.get(n).getDefaultValue()) for n in names}


def _gather_knee_mass_inertia(model_path: Path) -> Dict[str, Tuple[float, np.ndarray]]:
    """{body: (mass_kg, diagonal_moments[3])} for each knee body. The diagonal
    moments of inertia (kg·m²) are enough for the radius-of-gyration check."""
    m = osim.Model(str(model_path))
    m.initSystem()
    out: Dict[str, Tuple[float, np.ndarray]] = {}
    bs = m.getBodySet()
    for i in range(bs.getSize()):
        b = bs.get(i)
        if b.getName() not in KNEE_BODIES:
            continue
        mom = b.getInertia().getMoments()
        out[b.getName()] = (float(b.getMass()), np.array([mom[0], mom[1], mom[2]]))
    return out


# --- per-component scaling --------------------------------------------------


@pytest.mark.slow
class TestBuiltModelScalesByWA:
    """A real built Mode-1 model scaled to RSubject_121's gait body. Every knee
    component must scale by s_wa relative to the unscaled built model."""

    @pytest.fixture(scope="class")
    def s_wa(self, rsubject121_ab_path) -> float:
        f = read_ab_factors(rsubject121_ab_path)
        s = (f["femur_r"][2] + f["tibia_r"][2]) / 2.0
        # The whole suite is vacuous if s_wa == 1 (everything trivially matches).
        assert abs(s - 1.0) > 0.01, f"need a non-trivial s_wa to test scaling, got {s}"
        return s

    def test_every_knee_stl_scales_by_s_wa(
        self, built_mode1_model_path, built_subject_scaled_model, s_wa
    ):
        orig_geom = _geom_dir(built_mode1_model_path)
        scaled_geom = _geom_dir(built_subject_scaled_model)
        files = _knee_stl_files(built_mode1_model_path)
        assert len(files) >= 9, f"expected the full recon knee STL set, got {files}"
        for f in files:
            o, s = orig_geom / f, scaled_geom / f
            assert o.is_file() and s.is_file(), f"missing {f}"
            no = _sorted_vertex_norms(o)
            ns = _sorted_vertex_norms(s)
            assert len(no) == len(ns), f"{f}: vertex count changed ({len(no)} -> {len(ns)})"
            # Every vertex norm must equal the original * s_wa (origin-centered
            # isotropic scale). Measured worst per-vertex error ~1.5e-8 m (ASCII
            # STL write precision); TOL_STL_M = 1e-7 m keeps ~7x margin.
            err = float(np.max(np.abs(ns - no * s_wa)))
            assert err < TOL_STL_M, f"{f}: per-vertex scale error {err:.2e} m (s_wa={s_wa:.5f})"

    def test_patella_offset_scales_by_s_wa(
        self, built_mode1_model_path, built_subject_scaled_model, s_wa
    ):
        """The centered patella's placement offset (mean_patella, written as the
        pf_r translation defaults) must scale by s_wa so the patella stays put
        relative to the femur after the knee shrinks/grows."""
        before = _coord_defaults(built_mode1_model_path, PF_KNEE_COORDS)
        after = _coord_defaults(built_subject_scaled_model, PF_KNEE_COORDS)
        # The built model must actually have a non-zero patella offset, else this
        # check proves nothing.
        assert any(
            abs(before[c]) > 1e-4 for c in PF_KNEE_COORDS
        ), f"built model has ~zero patella offset {before}; cannot test its scaling"
        for c in PF_KNEE_COORDS:
            if abs(before[c]) < 1e-9:
                continue
            ratio = after[c] / before[c]
            assert (
                abs(ratio - s_wa) < TOL_EXACT
            ), f"{c}: patella offset scaled by {ratio:.12f}, expected s_wa={s_wa:.12f}"

    def test_knee_wraps_scale_by_s_wa(
        self, built_mode1_model_path, built_subject_scaled_model, s_wa
    ):
        before = _gather_knee_wraps(built_mode1_model_path)
        after = _gather_knee_wraps(built_subject_scaled_model)
        assert len(before) >= 3, f"expected knee wraps to compare, got {list(before)}"
        for name, b in before.items():
            a = after[name]
            tb, ta = np.linalg.norm(b["translation"]), np.linalg.norm(a["translation"])
            if tb > 1e-6:
                assert (
                    abs(ta / tb - s_wa) < TOL_EXACT
                ), f"wrap {name}: translation scaled by {ta/tb:.12f}, expected {s_wa:.12f}"
            if b["size"] > 1e-6:
                assert (
                    abs(a["size"] / b["size"] - s_wa) < TOL_EXACT
                ), f"wrap {name}: size scaled by {a['size']/b['size']:.12f}, expected {s_wa:.12f}"
            # An isotropic scale about the origin must NOT rotate the wrap
            # (measured change: exactly 0).
            assert np.allclose(a["rotation"], b["rotation"], atol=TOL_EXACT), (
                f"wrap {name}: rotation changed under scaling "
                f"({b['rotation']} -> {a['rotation']})"
            )

    def test_pure_knee_ligament_slacks_scale_by_s_wa(
        self, built_mode1_model_path, built_subject_scaled_model, s_wa
    ):
        """Slack lengths of ligaments wholly on knee bodies must scale by s_wa
        (so their *absolute* rest length tracks the knee). Mixed-body ligaments
        (e.g. ITB1) are excluded here — their slack scales by the mixed path
        change, and their invariant is checked by
        ``test_blankevoort_reference_strain_preserved`` instead."""
        before = _gather_knee_ligament_slacks(built_mode1_model_path)
        after = _gather_knee_ligament_slacks(built_subject_scaled_model)
        pure = [n for n in (set(before) & set(after)) if before[n]["all_knee"]]
        assert len(pure) >= 80, f"expected many pure-knee ligaments, got {len(pure)}"
        for n in pure:
            sb, sa = before[n]["slack"], after[n]["slack"]
            if sb <= 1e-9:
                continue
            # Slack rides the path-length/wrap solver, whose floor is ~1.9e-6
            # (vs the machine-eps clean scaling of the geometry); TOL_SLACK=1e-5.
            assert (
                abs(sa / sb - s_wa) < TOL_SLACK
            ), f"{n}: slack scaled by {sa/sb:.7f}, expected s_wa={s_wa:.7f}"

    def test_knee_muscle_points_scale_by_s_wa(
        self, built_mode1_model_path, built_subject_scaled_model, s_wa
    ):
        """Muscle path points on knee bodies (e.g. the quadriceps inserting on
        the patella) must scale by s_wa like every other knee attachment."""
        before = _gather_knee_muscle_points(built_mode1_model_path)
        after = _gather_knee_muscle_points(built_subject_scaled_model)
        # Scaling is a pure resize: the point set must be invariant (no silent
        # drop/add/rename), and every point present must scale by s_wa.
        assert set(before) == set(after), (
            f"scaling changed the knee muscle path-point set "
            f"(before {len(before)}, after {len(after)})"
        )
        assert (
            len(before) >= N_KNEE_MUSCLE_POINTS
        ), f"expected >= {N_KNEE_MUSCLE_POINTS} knee muscle points, got {sorted(before)}"
        for k in before:
            nb = np.linalg.norm(before[k])
            if nb <= 1e-6:
                continue
            ratio = np.linalg.norm(after[k]) / nb
            assert (
                abs(ratio - s_wa) < TOL_EXACT
            ), f"muscle point {k}: scaled by {ratio:.12f}, expected {s_wa:.12f}"

    def test_knee_ligament_points_scale_by_s_wa(
        self, built_mode1_model_path, built_subject_scaled_model, s_wa
    ):
        before = _gather_knee_ligament_points(built_mode1_model_path)
        after = _gather_knee_ligament_points(built_subject_scaled_model)
        # Scaling is a pure resize: the full ligament point set must survive
        # unchanged. before==after set-equality catches any silent point loss
        # (a ">= 50" floor would have let a 100+ point drop pass), and np.all
        # over the *whole* set requires every one to scale by exactly s_wa.
        assert set(before) == set(after), (
            f"scaling changed the knee ligament path-point set "
            f"(before {len(before)}, after {len(after)})"
        )
        assert (
            len(before) >= N_KNEE_LIGAMENT_POINTS
        ), f"expected >= {N_KNEE_LIGAMENT_POINTS} knee ligament points, got {len(before)}"
        ratios = np.array(
            [
                np.linalg.norm(after[k]) / np.linalg.norm(before[k])
                for k in before
                if np.linalg.norm(before[k]) > 1e-6
            ]
        )
        assert np.all(np.abs(ratios - s_wa) < TOL_EXACT), (
            f"ligament points did not all scale by s_wa={s_wa:.12f} "
            f"(max dev {np.max(np.abs(ratios - s_wa)):.2e})"
        )

    def test_knee_inertia_scales_by_s_wa_squared(
        self, built_mode1_model_path, built_subject_scaled_model, s_wa
    ):
        """Each knee body's *specific* inertia (I/m = squared radius of gyration)
        must scale by exactly s_wa**2 — the physically correct inertia change for
        an isotropic s_wa resize, independent of the mass the two-pass assigns.

        ScaleTool applies the s_wa geometric factor to the inertia tensor; the
        orchestrator two-pass then rescales inertia only by the mass ratio
        (orchestrator.py:121-126). So I/m isolates the geometry and the mass
        bookkeeping cancels: (I/m)_scaled / (I/m)_base == s_wa**2 for every knee
        body (measured to ~2e-16 on the RSubject_121 fixture). This guards that
        inertia is not left unscaled (or double-scaled) when the knee is resized.
        """
        before = _gather_knee_mass_inertia(built_mode1_model_path)
        after = _gather_knee_mass_inertia(built_subject_scaled_model)
        s2 = s_wa * s_wa
        checked = []
        for body in KNEE_BODIES:
            assert body in before and body in after, f"no inertia for knee body {body}"
            m0, i0 = before[body]
            m1, i1 = after[body]
            assert m0 > 1e-9 and m1 > 1e-9, f"{body}: non-positive mass ({m0} -> {m1})"
            assert np.all(i0 > 0), f"{body}: non-positive base inertia {i0}"
            ratio = (i1 / m1) / (i0 / m0)
            assert np.all(np.abs(ratio - s2) < TOL_EXACT), (
                f"{body}: specific inertia (I/m) scaled by {ratio}, expected "
                f"s_wa^2={s2:.12f} (max dev {np.max(np.abs(ratio - s2)):.2e})"
            )
            checked.append(body)
        assert len(checked) == len(KNEE_BODIES), f"only inertia-checked {checked}"

    def test_scaled_model_initializes(self, built_subject_scaled_model):
        """A scaled built model must still load and realize — i.e. be runnable."""
        m = osim.Model(str(built_subject_scaled_model))
        state = m.initSystem()
        assert m.getBodySet().getSize() > 0
        total_mass = sum(m.getBodySet().get(i).getMass() for i in range(m.getBodySet().getSize()))
        assert total_mass > 0, "scaled model has non-positive total mass"
        m.realizePosition(state)


# --- the plan's flagged regression: origin-scaling placement ----------------


@pytest.mark.slow
class TestOriginScalingPlacement:
    """Scaling a built knee about OSIM (0,0,0) by s_wa must preserve the intended
    placement for EVERY knee body (the joint-center origin is shared, so an
    origin-centered scale carries the body-local centroid by exactly s_wa). This
    is the regression the fix plan flagged as missing."""

    @pytest.fixture(scope="class")
    def s_wa(self, rsubject121_ab_path) -> float:
        f = read_ab_factors(rsubject121_ab_path)
        return (f["femur_r"][2] + f["tibia_r"][2]) / 2.0

    def test_every_knee_body_centroid_scales_by_s_wa(
        self, built_mode1_model_path, built_subject_scaled_model, s_wa
    ):
        before = _knee_centroid_by_body(built_mode1_model_path)
        after = _knee_centroid_by_body(built_subject_scaled_model)
        checked = []
        for body in KNEE_BODIES:
            assert body in before, f"no geometry found for knee body {body}"
            cb = before[body]
            ca = after[body]
            # The patella is centered (centroid ~ 0); an origin-scale leaves a
            # ~0 centroid ~0, so its ratio is ill-defined — verify the centroid
            # stays small instead, and rely on the STL/offset tests for it.
            if np.linalg.norm(cb) < 5e-3:
                assert np.linalg.norm(ca) < 5e-3, f"{body}: centered-body centroid drifted to {ca}"
                continue
            ratio = np.linalg.norm(ca) / np.linalg.norm(cb)
            assert abs(ratio - s_wa) < TOL_CENTROID, (
                f"{body}: centroid placement scaled by {ratio:.7f}, expected s_wa={s_wa:.7f} "
                f"(before={cb}, after={ca})"
            )
            checked.append(body)
        assert (
            len(checked) >= 3
        ), f"expected to placement-check >=3 offset knee bodies, got {checked}"


# --- origin-scaling placement on the reference base (no large fixtures) -----


@pytest.mark.slow
class TestReferenceOriginScalingPlacement:
    """Same origin-scaling placement regression, but on the *reference* base with
    an in-repo-derived s_wa = 0.9 — independent of the untracked AB output and
    the large built model. The scaling operation is identical to the built-model
    case; only the attached geometry differs (reference STLs vs recon)."""

    S_WA = 0.9

    def test_every_knee_body_centroid_scales_by_s_wa(
        self, base_comak_path, synthetic_scaled_reference_model
    ):
        before = _knee_centroid_by_body(base_comak_path)
        after = _knee_centroid_by_body(synthetic_scaled_reference_model)
        checked = []
        for body in KNEE_BODIES:
            if body not in before:
                continue
            cb, ca = before[body], after[body]
            if np.linalg.norm(cb) < 5e-3:  # centered body (patella)
                assert np.linalg.norm(ca) < 5e-3, f"{body}: centered centroid drifted to {ca}"
                continue
            ratio = np.linalg.norm(ca) / np.linalg.norm(cb)
            assert (
                abs(ratio - self.S_WA) < TOL_CENTROID
            ), f"{body}: centroid placement scaled by {ratio:.7f}, expected {self.S_WA}"
            checked.append(body)
        assert len(checked) >= 3, f"expected >=3 offset knee bodies checked, got {checked}"


# --- coherence survives the scale (mirrors Mode 2's test_nontrivial) --------


@pytest.mark.slow
class TestBuiltModelScalingCoherence:
    """Identity-scaled vs subject-scaled BUILT model: the same coherence the
    reference knee preserves under scaling must hold for the recon knee."""

    @pytest.fixture(scope="class")
    def ident_model(self, built_identity_scaled_model):
        m = osim.Model(str(built_identity_scaled_model))
        m.initSystem()
        return m

    @pytest.fixture(scope="class")
    def subj_model(self, built_subject_scaled_model):
        m = osim.Model(str(built_subject_scaled_model))
        m.initSystem()
        return m

    @pytest.fixture(scope="class")
    def s_wa(self, rsubject121_ab_path) -> float:
        f = read_ab_factors(rsubject121_ab_path)
        return (f["femur_r"][2] + f["tibia_r"][2]) / 2.0

    @pytest.mark.parametrize(
        "bone_body,bone_file,cart_file",
        [
            ("femur_distal_r", "femur_nsm_recon_osim.stl", "femur_articular_surface_osim.stl"),
            ("tibia_proximal_r", "tibia_nsm_recon_osim.stl", "tibia_articular_surface_osim.stl"),
            ("patella_r", "patella_nsm_recon_osim.stl", "patella_articular_surface_osim.stl"),
        ],
    )
    def test_cartilage_bone_proximity_preserved(
        self,
        ident_model,
        subj_model,
        built_identity_scaled_model,
        built_subject_scaled_model,
        s_wa,
        bone_body,
        bone_file,
        cart_file,
    ):
        d_ident = _cart_to_bone_distances(
            ident_model, _geom_dir(built_identity_scaled_model), bone_body, bone_file, cart_file
        )
        d_subj = _cart_to_bone_distances(
            subj_model, _geom_dir(built_subject_scaled_model), bone_body, bone_file, cart_file
        )
        mean_i, mean_s = float(d_ident.mean()), float(d_subj.mean())
        p95_i, p95_s = float(np.percentile(d_ident, 95)), float(np.percentile(d_subj, 95))
        # Cart-bone distances scale by s_wa to ~2e-10 m (STL/kdtree); the nearest-
        # neighbour pairing is scale-invariant. TOL_CARTBONE_M = 1e-8 m keeps a
        # ~50x margin while being 5e4x tighter than a "0.5 mm physical" bound.
        assert abs(mean_s - mean_i * s_wa) < TOL_CARTBONE_M, (
            f"{bone_body}: cart-bone mean drifted "
            f"(ident={mean_i:.8f}, subj={mean_s:.8f}, expected={mean_i*s_wa:.8f})"
        )
        assert abs(p95_s - p95_i * s_wa) < TOL_CARTBONE_M, (
            f"{bone_body}: cart-bone p95 drifted "
            f"(ident={p95_i:.8f}, subj={p95_s:.8f}, expected={p95_i*s_wa:.8f})"
        )

    def test_blankevoort_reference_strain_preserved(self, ident_model, subj_model):
        b = _gather_blankevoort(ident_model)
        s = _gather_blankevoort(subj_model)
        common = set(b) & set(s)
        assert len(common) >= 80, "expected at least 80 ligaments to compare"
        # Reference strain (path - slack)/slack is preserved EXACTLY: path and
        # slack scale by the same per-ligament factor, so it cancels. Measured
        # max drift 2.2e-16 (machine eps) across all 91, incl. mixed-body ITB1.
        for name in common:
            drift = abs(s[name]["ref_strain"] - b[name]["ref_strain"])
            assert drift < TOL_EXACT, (
                f"{name}: reference strain drifted by {drift:.2e} "
                f"({b[name]['ref_strain']:.6f} -> {s[name]['ref_strain']:.6f})"
            )
