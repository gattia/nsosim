"""Check that AddBiomechanics scaled the skeleton its factors will be applied to.

AB's per-body scale factors are ratios to the generic model AB was given, which
AB copies into its output as ``Models/unscaled_generic.osim``. ``build_scale_set``
applies them to the COMAK model, so they mean what the pipeline assumes only if
that generic *is* the COMAK skeleton.

In the Paper-1 v2 run it was not: AB scaled its stock ``ArmlessRajagopal``, whose
femur is 9 % longer and hips 16 % narrower than COMAK's. The upload carried the
same marker names, so nothing errored (E2E audit F15/A11). This check makes that
mix-up loud.

Three body-fixed skeleton landmarks are compared, none of them knee geometry
(which is subject-specific):

- femur: hip joint centre -> knee joint centre, in the ``femur_r`` frame;
- tibia: knee joint centre -> ankle joint centre, in the ``tibia_r`` frame;
- hip half-width: |z| of the hip joint centre in the ``pelvis`` frame.

Femur and tibia are compared as VECTORS: the difference ``|v_ab - v_ref|`` must
be within ``rtol`` of ``|v_ref|``. This bounds the length difference too, and also
catches a segment that has the right length but is offset sideways from its
joint. Such a 12 mm anterior-posterior tibia offset was found in a stripped
upload skeleton, with a length change of only 0.07 %.

Joint centres are found from the joint tree, not by joint name, so Rajagopal's
``walker_knee_r``, COMAK's ``knee_r`` (``femur_distal_r`` -> ``tibia_proximal_r``,
welded to the long bones) and a stripped skeleton's ``knee_r`` PinJoint are all
measured the same way. A frame is expressed in a body's frame only through weld
joints, so every landmark is independent of pose and coordinate defaults.
"""

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import opensim as osim

AB_GENERIC_FILENAME = "unscaled_generic.osim"
DEFAULT_RTOL = 0.01


class GenericSkeletonMismatchError(ValueError):
    """AB's generic model is missing or is not the skeleton being scaled."""


@dataclass
class SkeletonLandmarks:
    """Body-fixed skeleton landmarks of one model (metres, kg)."""

    path: str
    md5: str
    model_name: str
    n_bodies: int
    total_mass_kg: float
    knee_joint: str
    femur_vec: Tuple[float, float, float]
    tibia_vec: Tuple[float, float, float]
    hip_halfwidth: float

    @property
    def femur_len(self) -> float:
        return float(np.linalg.norm(self.femur_vec))

    @property
    def tibia_len(self) -> float:
        return float(np.linalg.norm(self.tibia_vec))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["femur_len"] = self.femur_len
        d["tibia_len"] = self.tibia_len
        return d


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rigid_groups(model: osim.Model) -> Dict[str, str]:
    """Map each body name to a representative of its weld-connected group."""
    parent: Dict[str, str] = {}

    def find(b: str) -> str:
        parent.setdefault(b, b)
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        return b

    js = model.getJointSet()
    for i in range(js.getSize()):
        j = js.get(i)
        p = j.getParentFrame().findBaseFrame().getName()
        c = j.getChildFrame().findBaseFrame().getName()
        find(p)
        find(c)
        if j.getConcreteClassName() == "WeldJoint":
            parent[find(p)] = find(c)
    return {b: find(b) for b in list(parent)}


def _joint_between(model: osim.Model, groups: Dict[str, str], body_a: str, body_b: str):
    """The single non-weld joint linking the rigid groups of body_a and body_b.

    Returns (joint, frame on body_a's side, frame on body_b's side).
    """
    ga, gb = groups.get(body_a), groups.get(body_b)
    if ga is None or gb is None:
        raise ValueError(f"model has no body {body_a if ga is None else body_b!r}")
    found = []
    js = model.getJointSet()
    for i in range(js.getSize()):
        j = js.get(i)
        if j.getConcreteClassName() == "WeldJoint":
            continue
        pf, cf = j.getParentFrame(), j.getChildFrame()
        pg = groups[pf.findBaseFrame().getName()]
        cg = groups[cf.findBaseFrame().getName()]
        if (pg, cg) == (ga, gb):
            found.append((j, pf, cf))
        elif (pg, cg) == (gb, ga):
            found.append((j, cf, pf))
    if len(found) != 1:
        names = [f[0].getName() for f in found]
        raise ValueError(f"expected one joint between {body_a} and {body_b}, found {names}")
    return found[0]


def _origin_in(state, frame, body) -> np.ndarray:
    p = frame.findStationLocationInAnotherFrame(state, osim.Vec3(0, 0, 0), body)
    return np.array([p.get(k) for k in range(3)])


def measure_skeleton(osim_path: Path, side: str = "r") -> SkeletonLandmarks:
    """Measure the femur, tibia and hip landmarks of one model (see module doc)."""
    osim_path = Path(osim_path)
    model = osim.Model(str(osim_path))
    state = model.initSystem()
    model.realizePosition(state)
    bs = model.getBodySet()
    groups = _rigid_groups(model)

    pelvis, femur, tibia, talus = "pelvis", f"femur_{side}", f"tibia_{side}", f"talus_{side}"
    _, hip_pelvis, hip_femur = _joint_between(model, groups, pelvis, femur)
    knee, knee_femur, knee_tibia = _joint_between(model, groups, femur, tibia)
    _, ankle_tibia, _ = _joint_between(model, groups, tibia, talus)

    femur_vec = _origin_in(state, knee_femur, bs.get(femur)) - _origin_in(
        state, hip_femur, bs.get(femur)
    )
    tibia_vec = _origin_in(state, ankle_tibia, bs.get(tibia)) - _origin_in(
        state, knee_tibia, bs.get(tibia)
    )
    hip_halfwidth = abs(float(_origin_in(state, hip_pelvis, bs.get(pelvis))[2]))

    return SkeletonLandmarks(
        path=str(osim_path),
        md5=_md5(osim_path),
        model_name=model.getName(),
        n_bodies=bs.getSize(),
        total_mass_kg=float(model.getTotalMass(state)),
        knee_joint=f"{knee.getName()} ({knee.getConcreteClassName()})",
        femur_vec=tuple(float(v) for v in femur_vec),
        tibia_vec=tuple(float(v) for v in tibia_vec),
        hip_halfwidth=hip_halfwidth,
    )


def compare_skeletons(
    generic: SkeletonLandmarks, reference: SkeletonLandmarks, rtol: float = DEFAULT_RTOL
) -> List[dict]:
    """One row per landmark: both values, the relative difference, and pass/fail."""
    rows = []
    for name, label, g_vec, r_vec in (
        ("femur", "femur hip->knee", generic.femur_vec, reference.femur_vec),
        ("tibia", "tibia knee->ankle", generic.tibia_vec, reference.tibia_vec),
    ):
        g, r = np.asarray(g_vec), np.asarray(r_vec)
        rel = float(np.linalg.norm(g - r) / np.linalg.norm(r))
        rows.append(
            dict(
                landmark=name,
                label=label,
                ab_generic_m=float(np.linalg.norm(g)),
                reference_m=float(np.linalg.norm(r)),
                rel_diff=rel,
                ok=rel <= rtol,
            )
        )
    rel = abs(generic.hip_halfwidth - reference.hip_halfwidth) / reference.hip_halfwidth
    rows.append(
        dict(
            landmark="hip_halfwidth",
            label="hip half-width",
            ab_generic_m=generic.hip_halfwidth,
            reference_m=reference.hip_halfwidth,
            rel_diff=float(rel),
            ok=rel <= rtol,
        )
    )
    return rows


def _format_table(rows: List[dict]) -> str:
    lines = [f"  {'landmark':20s} {'AB generic':>11s} {'reference':>11s} {'difference':>11s}"]
    for r in rows:
        flag = "" if r["ok"] else "  MISMATCH"
        lines.append(
            f"  {r['label'] + ' (m)':20s} {r['ab_generic_m']:11.4f} {r['reference_m']:11.4f} "
            f"{100 * r['rel_diff']:10.1f}%{flag}"
        )
    return "\n".join(lines)


_OVERRIDE_HINT = (
    "If this is intended -- e.g. reproducing results that were built on that generic "
    "model -- pass allow_generic_mismatch=True; the override and both sets of landmarks "
    "are recorded in the scaling report."
)


def check_ab_generic(
    ab_scaled_osim: Path,
    reference_osim: Path,
    allow_generic_mismatch: bool = False,
    rtol: float = DEFAULT_RTOL,
    side: str = "r",
) -> dict:
    """Raise unless AB's generic model is the skeleton of ``reference_osim``.

    ``ab_generic`` is ``unscaled_generic.osim`` beside ``ab_scaled_osim`` (AB's
    output layout). ``reference_osim`` is the model the factors will be applied
    to (``scale_comak_model``'s ``base_osim``).

    Raises ``GenericSkeletonMismatchError`` if the generic is missing or any
    landmark differs by more than ``rtol``, unless ``allow_generic_mismatch``.
    Returns a JSON-serialisable record for the scaling report, whose ``status``
    is ``"match"``, ``"mismatch_overridden"`` or ``"missing_overridden"``.
    """
    ab_scaled_osim = Path(ab_scaled_osim)
    generic_path = ab_scaled_osim.parent / AB_GENERIC_FILENAME
    reference = measure_skeleton(reference_osim, side=side)
    record = dict(
        ab_generic_path=str(generic_path),
        allow_generic_mismatch=bool(allow_generic_mismatch),
        rtol=rtol,
        reference=reference.to_dict(),
    )

    if not generic_path.is_file():
        if not allow_generic_mismatch:
            raise GenericSkeletonMismatchError(
                f"AddBiomechanics' generic model {generic_path} is missing, so there is no way "
                f"to check that the scale factors in {ab_scaled_osim.name} are relative to the "
                f"skeleton being scaled ({reference.model_name}, {reference_osim}). Keep AB's "
                f"Models/ folder intact. {_OVERRIDE_HINT}"
            )
        record.update(status="missing_overridden", ab_generic=None, landmarks=None)
        return record

    generic = measure_skeleton(generic_path, side=side)
    rows = compare_skeletons(generic, reference, rtol=rtol)
    record.update(ab_generic=generic.to_dict(), landmarks=rows)

    if all(r["ok"] for r in rows):
        record["status"] = "match"
        return record
    if allow_generic_mismatch:
        record["status"] = "mismatch_overridden"
        return record
    raise GenericSkeletonMismatchError(
        "AddBiomechanics' generic model does not match the skeleton being scaled.\n"
        f"  AB generic: {generic_path}\n"
        f"              model '{generic.model_name}', {generic.n_bodies} bodies, "
        f"knee {generic.knee_joint}\n"
        f"  reference : {reference_osim}\n"
        f"              model '{reference.model_name}', {reference.n_bodies} bodies, "
        f"knee {reference.knee_joint}\n"
        f"{_format_table(rows)}\n"
        f"(Tolerance {100 * rtol:.0f} %. For femur and tibia the difference is between the "
        "joint-centre vectors, so it includes any sideways offset, not only length.)\n"
        "AB's scale factors are ratios to its generic model. Applied here they would size "
        "the body relative to a different skeleton. Run AB on the COMAK skeleton itself, "
        f"with the knee stripped to a hinge. {_OVERRIDE_HINT}"
    )
