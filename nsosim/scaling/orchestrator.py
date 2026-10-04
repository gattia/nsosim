"""Top-level orchestrator for COMAK body scaling (a.k.a. "Stage X")."""

import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import opensim as osim

from ..geometry_staging import check_mode, stage_geometry
from .config import WA_KNEE_BODIES, MassRule, ScalingMode
from .knee_geometry import bake_knee_geometry, knee_geometry_files
from .markers import swap_markerset
from .model_fixes import fix_in_place
from .report import write_report
from .scale_factors import build_scale_set, read_ab_factors
from .scaletool import apply_scaletool
from .validate import check_ab_generic

log = logging.getLogger(__name__)


def _resolve_base_geometry_dir(base_osim: Path) -> Path:
    """Find the Geometry/ folder that the base model's mesh_file refs resolve to."""
    candidate = base_osim.parent / "Geometry"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"Could not locate Geometry/ next to base model {base_osim}")


def _total_body_mass(osim_path: Path) -> float:
    """Sum body masses in an .osim. Used to auto-detect the AB subject mass."""
    m = osim.Model(str(osim_path))
    m.initSystem()
    bs = m.getBodySet()
    return float(sum(bs.get(i).getMass() for i in range(bs.getSize())))


def _read_ab_per_body_masses(ab_osim: Path) -> Dict[str, float]:
    """Read {body_name: mass_kg} from an AB ``match_markers_and_physics.osim``.

    These are AB's physics-residual-tuned per-body masses, which sum to the
    patient's measured total mass.
    """
    m = osim.Model(str(ab_osim))
    m.initSystem()
    bs = m.getBodySet()
    return {bs.get(i).getName(): float(bs.get(i).getMass()) for i in range(bs.getSize())}


def _is_identity(sf: Tuple[float, float, float], atol: float = 1e-6) -> bool:
    return all(abs(s - 1.0) < atol for s in sf)


def _apply_per_body_masses(
    output_osim: Path,
    ab_per_body_masses: Dict[str, float],
    ab_factors: Dict[str, Tuple[float, float, float]],
    target_total_mass: float,
) -> Dict[str, dict]:
    """Two-pass mass transfer on the just-built scaled .osim.

    Pass 1 — provisional targets:
      - For bodies AB characterized (non-identity scale_factors AND present
        in the scaled model): use the AB mass.
      - For all other bodies (knee subbodies, patella with AB-identity scale,
        anything AB doesn't own): use base_mass × global_ratio where
        global_ratio = target_total_mass / sum(base_masses).

    Pass 2 — renormalisation:
      - Compute the provisional total and multiply every body's provisional
        target by (target_total_mass / provisional_total). The final total
        equals ``target_total_mass`` exactly, while the AB-per-body bias is
        preserved.

    Inertia for each body is scaled by (final_mass / current_mass) — the
    geometric scaling is already baked into the current inertia by ScaleTool.

    Returns a per-body audit dict.
    """
    model = osim.Model(str(output_osim))
    model.initSystem()
    bs = model.getBodySet()

    n_bodies = bs.getSize()
    names = [bs.get(i).getName() for i in range(n_bodies)]
    current_masses = [bs.get(i).getMass() for i in range(n_bodies)]
    base_total = sum(current_masses)
    if base_total <= 1e-12:
        raise RuntimeError("Scaled model has zero total body mass — cannot rescale.")

    global_ratio = target_total_mass / base_total

    # Pass 1: provisional target per body.
    provisional = []
    decision = []
    for i, name in enumerate(names):
        sf = ab_factors.get(name)
        if (
            name in ab_per_body_masses
            and sf is not None
            and not _is_identity(sf)
            and current_masses[i] > 1e-12
        ):
            provisional.append(ab_per_body_masses[name])
            decision.append("ab_per_body")
        else:
            provisional.append(current_masses[i] * global_ratio)
            decision.append("global_ratio")

    # Pass 2: renormalise so total exactly matches target.
    provisional_total = sum(provisional)
    correction = target_total_mass / provisional_total if provisional_total > 0 else 1.0
    final_masses = [p * correction for p in provisional]

    audit: Dict[str, dict] = {}
    for i, name in enumerate(names):
        body = bs.get(i)
        new_mass = final_masses[i]
        old_mass = current_masses[i]
        body.setMass(new_mass)

        if old_mass > 1e-12:
            ratio = new_mass / old_mass
            old_inertia = body.getInertia()
            mo = [old_inertia.getMoments()[k] * ratio for k in range(3)]
            po = [old_inertia.getProducts()[k] * ratio for k in range(3)]
            body.setInertia(osim.Inertia(mo[0], mo[1], mo[2], po[0], po[1], po[2]))

        audit[name] = {
            "decision": decision[i],
            "base_mass_kg": old_mass,
            "provisional_kg": provisional[i],
            "final_kg": new_mass,
        }

    model.finalizeConnections()
    model.printToXML(str(output_osim))
    audit["__renormalize_correction"] = correction  # type: ignore[assignment]
    audit["__provisional_total_kg"] = provisional_total  # type: ignore[assignment]
    return audit


def volume_rule_masses(
    base_masses: Dict[str, float],
    volume_factors: Dict[str, float],
    total_mass: float,
) -> Tuple[Dict[str, float], float]:
    """Per-body masses by OpenSim's volume rule. Returns ``({body: kg}, c)``.

    ``m_i = base_i * V_i * c`` with ``c = total_mass / sum_j(base_j * V_j)``, so the masses sum to
    ``total_mass``. ``V_i`` is the product of body *i*'s three scale factors (a body absent from
    ``volume_factors`` has ``V = 1``). This is what ScaleTool does with ``preserve_mass_distribution=False`` and a
    subject mass, and what AddBiomechanics' output masses are: AB's per-body masses equal this rule applied to AB's
    generic with AB's factors, to machine precision (comak rerun-v3 sub-plan 02, ``ab_mass_rule.py``).
    """
    weighted = {name: m * abs(volume_factors.get(name, 1.0)) for name, m in base_masses.items()}
    denominator = sum(weighted.values())
    if denominator <= 1e-12:
        raise RuntimeError("Model has zero total body mass -- cannot apply the volume rule.")
    c = float(total_mass) / denominator
    return {name: w * c for name, w in weighted.items()}, c


def derive_mass_proxies(ab_scaled_osim: Path, ab_bodies, exclude=()) -> Tuple[Dict[str, str], list]:
    """Bodies of AB's skeleton that no marker observes, each mapped to an observed neighbour.

    AB still returns factors for a body with no marker on it, but nothing constrains them: on the COMAK v7 skeleton
    the head (``neckhead``) comes back at 1.4-1.9x its volume in every donor. Under the volume rule that factor would
    set the body's mass. So, for mass and inertia only, such a body borrows an observed neighbour's factors
    (comak rerun-v3, owner ruling 2026-10-04): an observed child if it has one, otherwise its nearest observed
    ancestor. On the v7 skeleton that gives talus -> calcaneus, toes -> calcaneus, neckhead -> torso and
    patella_l -> femur_l.

    Observed = carries at least one marker of AB's MarkerSet. The topology is AB's model's joints. ``exclude``
    (the knee bodies, which take ``s_wa`` regardless) is left out. Returns ``({body: proxy}, sorted observed bodies)``;
    raises if an unobserved body has no observed neighbour.
    """
    model = osim.Model(str(ab_scaled_osim))
    model.initSystem()
    ms = model.getMarkerSet()
    observed = {
        ms.get(i).getParentFrame().findBaseFrame().getName() for i in range(ms.getSize())
    }
    parent: Dict[str, str] = {}
    js = model.getJointSet()
    for i in range(js.getSize()):
        j = js.get(i)
        parent[j.getChildFrame().findBaseFrame().getName()] = j.getParentFrame().findBaseFrame().getName()
    children: Dict[str, list] = {}
    for child, par in parent.items():
        children.setdefault(par, []).append(child)

    proxies: Dict[str, str] = {}
    for body in sorted(set(ab_bodies) - observed - set(exclude)):
        seen = [c for c in sorted(children.get(body, [])) if c in observed]
        if seen:
            proxies[body] = seen[0]
            continue
        up = parent.get(body)
        while up is not None and up not in observed:
            up = parent.get(up)
        if up is None or up == "ground":
            raise ValueError(f"{body}: no marker observes it, and no observed neighbour was found in AB's skeleton")
        proxies[body] = up
    return proxies, sorted(observed)


def _apply_volume_rule_masses(
    output_osim: Path,
    base_osim: Path,
    scale_factors: Dict[str, Tuple[float, float, float]],
    ab_bodies: set,
    target_total_mass: float,
    proxies: Optional[Dict[str, str]] = None,
) -> Dict[str, dict]:
    """Set the scaled model's masses by the volume rule (``mass_rule="volume"``).

    Runs on ScaleTool's output, made with ``preserve_mass_distribution=True``: there every body still has its base
    mass and its inertia is scaled geometrically only. Each body's mass becomes ``volume_rule_masses``' value and its
    inertia (moments and products) is multiplied by ``final / base``, so the inertia per unit mass stays ScaleTool's
    geometric scaling. AB's own per-body masses are not read: they are this same rule applied to AB's generic, and
    on a COMAK model they would leave out the bodies AB does not model.

    ``proxies`` ({body: observed neighbour}, see ``derive_mass_proxies``): a proxied body's mass uses its neighbour's
    volume factor, and its inertia is re-scaled from ``base_osim``'s with the neighbour's factors
    (``Body.scaleInertialProperties``, the call ScaleTool itself makes). Its geometry and joint frames keep AB's
    factors, so AB's marker fit is untouched.

    Returns a per-body audit, plus ``__c`` (the normalisation) and ``__fraction_absent_from_ab`` (the share of the
    final mass in bodies that AB's skeleton does not have: the arms on an armless skeleton, the knee sub-bodies).
    """
    proxies = dict(proxies or {})
    model = osim.Model(str(output_osim))
    model.initSystem()
    bs = model.updBodySet()
    names = [bs.get(i).getName() for i in range(bs.getSize())]
    missing = [b for b in list(proxies) + list(proxies.values()) if b not in names or b not in scale_factors]
    if missing:
        raise ValueError(f"mass proxies name bodies the scaled model or the scale set lacks: {missing}")
    base = {name: float(bs.get(name).getMass()) for name in names}

    def volume(name: str) -> float:
        sf = scale_factors.get(proxies.get(name, name))
        return abs(sf[0] * sf[1] * sf[2]) if sf is not None else 1.0

    final, c = volume_rule_masses(base, {n: volume(n) for n in names}, target_total_mass)

    base_bodies = None
    if proxies:
        base_model = osim.Model(str(base_osim))
        base_model.initSystem()
        base_bodies = base_model.getBodySet()

    audit: Dict[str, dict] = {}
    for name in names:
        body = bs.get(name)
        if name in proxies:
            # Geometric inertia from the neighbour's factors instead of AB's, then the mass ratio below.
            body.setInertia(base_bodies.get(name).getInertia())
            body.scaleInertialProperties(osim.Vec3(*scale_factors[proxies[name]]), False)
        body.setMass(final[name])
        if base[name] > 1e-12:
            ratio = final[name] / base[name]
            inertia = body.getInertia()
            mo = [inertia.getMoments()[k] * ratio for k in range(3)]
            po = [inertia.getProducts()[k] * ratio for k in range(3)]
            body.setInertia(osim.Inertia(mo[0], mo[1], mo[2], po[0], po[1], po[2]))
        entry = {
            "base_kg": base[name],
            "volume_factor": volume(name),
            "final_kg": final[name],
            "in_ab_skeleton": name in ab_bodies,
        }
        if name in proxies:
            sf = scale_factors[name]
            entry.update(mass_proxy=proxies[name], ab_volume_factor=abs(sf[0] * sf[1] * sf[2]))
        audit[name] = entry

    model.finalizeConnections()
    model.printToXML(str(output_osim))
    absent = sum(final[n] for n in names if n not in ab_bodies)
    audit["__c"] = c  # type: ignore[assignment]
    audit["__fraction_absent_from_ab"] = absent / float(target_total_mass)  # type: ignore[assignment]
    return audit


def _read_ab_factors_for_mass(ab_osim: Path) -> Dict[str, Tuple[float, float, float]]:
    """Like read_ab_factors() but kept local so the orchestrator can call it
    without import-cycle risk. AB reports ``(1,1,1)`` for bodies it didn't
    subject-characterise (typically the patella) — used as a signal that the
    body's mass shouldn't be transferred from AB either."""
    return read_ab_factors(ab_osim)


def scale_comak_model(
    base_osim: Path,
    ab_scaled_osim: Path,
    output_osim: Path,
    output_geometry_dir: Path,
    mode: ScalingMode = "WA",
    preserve_mass_distribution: bool = True,
    subject_mass: Optional[float] = None,
    report_json: Optional[Path] = None,
    backup_markerset_xml: Optional[Path] = None,
    allow_generic_mismatch: bool = False,
    mass_rule: MassRule = "as_run",
    recorded_mass_kg: Optional[float] = None,
    mass_proxies: Optional[Dict[str, str]] = None,
    geometry_staging: str = "copy",
) -> Path:
    """Produce a scaled COMAK model from a base + AB output.

    Place in the pipeline (COMAK body scaling):
        This is the COMAK body-scaling orchestrator. It consumes AB's
        ``match_markers_and_physics.osim`` (``ab_scaled_osim`` — source of the
        per-body scale factors in dimensionless ratios and the per-body
        physics-tuned masses in kg) plus a COMAK base model (``base_osim`` —
        carries the subject-specific knee geometry, JAM contact meshes,
        ligaments, wraps). It produces a scaled COMAK ``.osim`` at
        ``output_osim`` with its meshes in ``output_geometry_dir``, plus a JSON
        report. Everything operates in OSIM space: STL vertices and frame
        translations are in meters, masses in kg, inertia in kg·m², and all
        scale factors are dimensionless ratios (1.0 = no change).

    Build, then scale (the personalized-knee path):
        ``base_osim`` may be the *generic reference* COMAK model (scales the
        reference knee to the gait body — "Mode 2") OR an already-built
        personalized model: a Mode-1 MRI build or a synthetic decode whose recon
        knee has been swapped onto the knee bodies. Because ``bake_knee_geometry``
        bakes *whatever STL each knee body points at* (it reads ``mesh_file`` off
        the model) and ScaleTool scales the wraps / ligaments / joint frames /
        muscles and the patella placement offset (``pf_tx/ty/tz_r``), running
        this on a *built* model scales the whole personalized knee by ``s_wa`` —
        no mid-build interception and no GPU re-fit. That is the "build, then
        scale" route (``docs/deviations.md`` Mode 3 for MRI builds, Mode 5 for
        synthetic), which amortizes the expensive NSM fit across gait subjects:
        build once on a reference base, then scale the built model per subject.
        Verified end-to-end in ``tests/scaling/test_build_then_scale.py``. Worked
        examples (scale existing models / build-once-scale-many / single pair)
        are in the docs: "Knee sizing modes" → Mode 3. The build step is
        ``nsosim.model_building.build_joint_model``.

    Pipeline:
      0. Check that AB's generic model (``unscaled_generic.osim`` beside
         ``ab_scaled_osim``) is the skeleton of ``base_osim``
         (``validate.check_ab_generic``). AB's factors are ratios to that
         generic, so on another skeleton they would size the body wrongly. This
         raises unless ``allow_generic_mismatch=True``; it runs before anything
         is written.
      1. Read AB per-body scale factors: each body's mesh scale over the same
         mesh's scale in AB's generic (``read_ab_factors``).
      2. Build a ScaleSet according to `mode`.
      3. ScaleTool.run() → output_osim (geometry-only scaling in meters via the
         dimensionless ScaleSet; the knee weld translation is scaled per-axis by
         the parent bone's anisotropic AB factors, not by s_wa). Body masses
         (kg) pass through unchanged on ScaleTool's preserve-mass-distribution
         path; inertia (kg·m²) is scaled geometrically by scale².
      4. Per-body masses, by ``mass_rule`` (see "Mass rule" below); the total
         body mass equals ``subject_mass`` exactly either way.
      5. Copy base Geometry/ → output_geometry_dir/ (``geometry_staging``).
      6. Bake the knee-body geometry into its STLs: multiply each knee STL's
         vertices (meters) by the isotropic, dimensionless ``s_wa`` about the
         body-local origin (= knee joint center, not the bone centroid), then
         reset the visual scale_factors to [1, 1, 1]. This is mandatory for the
         JAM contact meshes (the loader ignores XML scale_factors) and makes the
         on-disk STL self-describing.
      7. Swap MarkerSet for AB's static-trial-placed markers.
      8. Apply permanent model fix-ups (ITB1 reparent).
      9. Write the JSON report (always — see below).

    The output `.osim` is at `output_osim`; the orchestrator requires
    `output_geometry_dir == output_osim.parent / "Geometry"` so OpenSim's default
    relative-path resolution finds the meshes.

    Geometry staging (``geometry_staging``):
        ``"copy"`` (default) copies the whole base Geometry folder, then step 6
        rewrites the knee meshes in it. ``"link"`` writes real copies only of the
        files step 6 rewrites (``knee_geometry_files``, read from ``base_osim``;
        17 for a COMAK knee) and relative symlinks to ``base_osim``'s Geometry for
        the rest (``nsosim.geometry_staging``). The scaled model and its results
        are identical either way; the report records the mode and the counts. The
        bake raises on a symlink, and the files it wrote must equal the staged set.

    Subject mass handling:
        ``subject_mass``, when None (the default), is auto-detected by summing
        the AB model's body masses. AddBiomechanics tunes per-body masses to
        match the subject's actual measured mass, so this sum IS the subject's
        scaled mass and is the canonical source of truth. Pass an explicit
        float to override (e.g. for unit tests or when the AB tuning is
        suspect).

        ``recorded_mass_kg`` (optional) is the mass typed into AB, e.g. the
        participant's recorded body mass. It never sets a mass: it is written
        to the report beside AB's total (``ab_total_mass_kg``) and their
        difference, as a sanity check on AB's physics-pass estimate.

    Mass rule (``mass_rule``):
        ``"as_run"`` (default; how the Paper-1 v2 models were made) -- the
        two-pass transfer in ``_apply_per_body_masses``. Pass 1 targets each
        AB-scaled body to AB's per-body mass and the rest to base_mass ×
        global_ratio; pass 2 renormalises to ``subject_mass``. Because AB's
        masses already sum to the subject mass, the bodies AB does not model
        (arms, knee sub-bodies) are counted on top and everything is shrunk
        (v2: ×0.848; report ``mass_transfer_renormalize_correction``).

        ``"volume"`` -- ``_apply_volume_rule_masses``: OpenSim's volume rule on
        this model's own base masses, ``m_i = base_i · (sx·sy·sz)_i · c``,
        normalised to ``subject_mass``. AB's per-body masses are this same rule
        on AB's generic, so where the generic and ``base_osim`` carry the same
        masses (the stripped COMAK skeleton) the AB bodies keep AB's ratios
        exactly; bodies AB lacks take their share out of the same total (arms:
        volume factor 1; knee sub-bodies, patella and menisci: ``s_wa³``).
        Requires ``preserve_mass_distribution=True``, so ScaleTool scales the
        geometry only. Report: ``mass_normalisation_c``,
        ``mass_fraction_in_bodies_absent_from_ab`` and per body
        ``{base_kg, volume_factor, final_kg, in_ab_skeleton}``.

        Bodies no marker observes (``mass_proxies``, volume rule only): AB
        returns factors for them that nothing constrains (the head comes back
        at 1.4-1.9x its volume). With ``mass_proxies=None`` (the default) they
        are derived from AB's marker set (``derive_mass_proxies``: an observed
        child, else the nearest observed ancestor) and each such body takes its
        proxy's factors for mass and inertia only; its geometry and joint
        frames keep AB's factors. Pass a dict to set them explicitly, or ``{}``
        for none. Report: ``unobserved_bodies``, ``mass_proxies``,
        ``mass_proxy_source``, ``marker_observed_bodies``; per proxied body
        ``mass_proxy`` and ``ab_volume_factor``.

    AB-input check (``allow_generic_mismatch``):
        Raises ``GenericSkeletonMismatchError`` if AB's generic model is missing
        or differs from ``base_osim``'s skeleton by more than 1 % (femur and
        tibia joint-centre vectors, hip half-width). Pass
        ``allow_generic_mismatch=True`` only when the mismatch is intended, e.g.
        to reproduce results built on another generic model (the Paper-1 v2
        measured arm used AB's ArmlessRajagopal). The check's outcome, including
        any override and both sets of landmarks, is the report's
        ``ab_generic_check``.

    The report is always written so the scaling is recoverable from disk — the
    baked STLs and the model's reset ``scale_factors = [1, 1, 1]`` would
    otherwise hide how much the knee was scaled. The JSON records the isotropic
    ``s_wa`` ratio, the full per-body scale set (dimensionless ratios), and the
    per-body mass audit (base/provisional/final masses in kg). If ``report_json``
    is None, it defaults to ``output_osim.with_suffix(".scaling.json")``.
    Returns the report path.
    """
    check_mode(geometry_staging)
    base_osim = Path(base_osim)
    ab_scaled_osim = Path(ab_scaled_osim)
    output_osim = Path(output_osim)
    output_geometry_dir = Path(output_geometry_dir)

    expected_geom = output_osim.parent / "Geometry"
    if output_geometry_dir.resolve() != expected_geom.resolve():
        raise ValueError(
            f"output_geometry_dir must be {expected_geom} "
            f"(output_osim.parent / 'Geometry'), got {output_geometry_dir}"
        )

    if mass_rule not in ("as_run", "volume"):
        raise ValueError(f"mass_rule must be 'as_run' or 'volume', got {mass_rule!r}")
    if mass_rule != "volume" and mass_proxies:
        raise ValueError("mass_proxies apply only to mass_rule='volume'; as_run reproduces v2 and takes none")
    if mass_rule == "volume" and not preserve_mass_distribution:
        raise ValueError(
            "mass_rule='volume' sets every mass itself and needs ScaleTool to scale the geometry only: "
            "pass preserve_mass_distribution=True"
        )

    ab_generic_check = check_ab_generic(
        ab_scaled_osim, base_osim, allow_generic_mismatch=allow_generic_mismatch
    )
    if ab_generic_check["status"] != "match":
        log.warning(
            "AB-input check overridden (%s): scale factors are not relative to %s",
            ab_generic_check["status"],
            base_osim.name,
        )

    output_osim.parent.mkdir(parents=True, exist_ok=True)

    # AB writes (generic mesh scale) x (body scale): divide by the generic's, when there is one.
    generic = Path(ab_generic_check["ab_generic_path"])
    generic = generic if generic.is_file() else None
    factors = read_ab_factors(ab_scaled_osim, generic_osim=generic)
    scale_set, s_wa = build_scale_set(factors, mode=mode)

    ab_total_mass = _total_body_mass(ab_scaled_osim)
    if subject_mass is None:
        subject_mass_effective = ab_total_mass
        log.info(
            "Auto-detected subject_mass=%.4f kg from AB model %s",
            subject_mass_effective,
            ab_scaled_osim.name,
        )
    else:
        subject_mass_effective = float(subject_mass)
        log.info("Using explicit subject_mass=%.4f kg", subject_mass_effective)

    base_mass = _total_body_mass(base_osim)

    apply_scaletool(
        base_osim=base_osim,
        scale_set=scale_set,
        out_osim=output_osim,
        preserve_mass_distribution=preserve_mass_distribution,
    )

    base_geom = _resolve_base_geometry_dir(base_osim)
    if output_geometry_dir.exists():
        shutil.rmtree(output_geometry_dir)
    rewritten = set()
    if geometry_staging == "link":
        # The scaled model cannot load until its meshes resolve, so the files the
        # bake rewrites are read off the base model (same bodies, same file names).
        base_model = osim.Model(str(base_osim))
        base_model.initSystem()
        rewritten = knee_geometry_files(base_model, WA_KNEE_BODIES)
    staging = stage_geometry(base_geom, output_geometry_dir, real=rewritten, mode=geometry_staging)

    baked_geometry = bake_knee_geometry(
        scaled_osim=output_osim,
        knee_bodies=WA_KNEE_BODIES,
        scale=s_wa,
        geometry_dir=output_geometry_dir,
    )
    if geometry_staging == "link" and set(baked_geometry) != rewritten:
        raise RuntimeError(
            "the bake wrote a different file set than was staged as real copies: "
            f"baked only {sorted(set(baked_geometry) - rewritten)}, "
            f"staged only {sorted(rewritten - set(baked_geometry))}"
        )

    scale_set_dump = {}
    for i in range(scale_set.getSize()):
        s = scale_set.get(i)
        sf = s.getScaleFactors()
        scale_set_dump[s.getSegmentName()] = (float(sf[0]), float(sf[1]), float(sf[2]))

    # Per-body masses -- AFTER the geometry copy + bake so the model loads
    # cleanly (contact meshes are resolvable).
    proxy_fields: dict = {}
    if mass_rule == "volume":
        observed = None
        if mass_proxies is None:
            proxies, observed = derive_mass_proxies(ab_scaled_osim, factors, exclude=WA_KNEE_BODIES)
            source = "derived from AB's marker set"
        else:
            proxies, source = dict(mass_proxies), "explicit argument"
        mass_audit = _apply_volume_rule_masses(
            output_osim=output_osim,
            base_osim=base_osim,
            scale_factors=scale_set_dump,
            ab_bodies=set(factors),
            target_total_mass=subject_mass_effective,
            proxies=proxies,
        )
        proxy_fields = {
            "unobserved_bodies": sorted(proxies),
            "mass_proxies": proxies,
            "mass_proxy_source": source,
            "marker_observed_bodies": observed,
        }
    else:
        ab_masses = _read_ab_per_body_masses(ab_scaled_osim)
        mass_audit = _apply_per_body_masses(
            output_osim=output_osim,
            ab_per_body_masses=ab_masses,
            ab_factors=factors,
            target_total_mass=subject_mass_effective,
        )

    n_added, n_dropped = swap_markerset(
        scaled_osim=output_osim,
        ab_scaled_osim=ab_scaled_osim,
        backup_xml_path=backup_markerset_xml,
    )

    fix_status = fix_in_place(output_osim)

    report_path = (
        Path(report_json) if report_json is not None else output_osim.with_suffix(".scaling.json")
    )
    output_mass = _total_body_mass(output_osim)
    # Split the audit into the per-body table + scalar metadata before serialising.
    per_body_audit = {k: v for k, v in mass_audit.items() if not str(k).startswith("__")}
    if mass_rule == "volume":
        mass_fields = {
            "mass_normalisation_c": mass_audit["__c"],
            "mass_fraction_in_bodies_absent_from_ab": mass_audit["__fraction_absent_from_ab"],
            **proxy_fields,
        }
    else:
        mass_fields = {
            "mass_transfer_provisional_total_kg": mass_audit.get(
                "__provisional_total_kg", subject_mass_effective
            ),
            "mass_transfer_renormalize_correction": mass_audit.get("__renormalize_correction", 1.0),
        }
    write_report(
        report_path,
        mode=mode,
        ab_generic_check=ab_generic_check,
        ab_factors_relative_to=(
            f"generic mesh scales ({generic})" if generic else "raw scale_factors (no generic)"
        ),
        ab_scaled_osim=str(ab_scaled_osim),
        base_osim=str(base_osim),
        output_osim=str(output_osim),
        output_geometry_dir=str(output_geometry_dir),
        preserve_mass_distribution=preserve_mass_distribution,
        subject_mass_kg=subject_mass_effective,
        subject_mass_source=("override" if subject_mass is not None else "ab_total_body_mass"),
        ab_total_mass_kg=ab_total_mass,
        recorded_mass_kg=recorded_mass_kg,
        ab_total_minus_recorded_kg=(
            ab_total_mass - float(recorded_mass_kg) if recorded_mass_kg is not None else None
        ),
        base_total_mass_kg=base_mass,
        output_total_mass_kg=output_mass,
        mass_rule=mass_rule,
        **mass_fields,
        per_body_mass_audit=per_body_audit,
        ab_factors={k: list(v) for k, v in factors.items()},
        scale_set=scale_set_dump,
        wa_scale=s_wa,
        knee_geometry_baked={k: str(v) for k, v in baked_geometry.items()},
        geometry_staging=staging,
        marker_added=n_added,
        marker_dropped=n_dropped,
        backup_markerset_xml=str(backup_markerset_xml) if backup_markerset_xml else None,
        **fix_status,
    )
    return report_path
