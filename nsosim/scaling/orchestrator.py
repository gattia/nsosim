"""Top-level orchestrator for COMAK body scaling (a.k.a. "Stage X")."""

import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import opensim as osim

from .config import WA_KNEE_BODIES, ScalingMode
from .knee_geometry import bake_knee_geometry
from .markers import swap_markerset
from .model_fixes import fix_in_place
from .report import write_report
from .scale_factors import build_scale_set, read_ab_factors
from .scaletool import apply_scaletool

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

    Pipeline:
      1. Read AB per-body scale factors.
      2. Build a ScaleSet according to `mode`.
      3. ScaleTool.run() → output_osim (geometry-only scaling in meters via the
         dimensionless ScaleSet; the knee weld translation is scaled per-axis by
         the parent bone's anisotropic AB factors, not by s_wa). Body masses
         (kg) pass through unchanged on ScaleTool's preserve-mass-distribution
         path; inertia (kg·m²) is scaled geometrically by scale².
      4. Two-pass per-body mass transfer (the post-2026-05-31 fix):
           Pass 1: target each AB-characterised body to AB's per-body mass;
                   target the rest to base_mass × global_ratio.
           Pass 2: renormalise so the total body mass equals
                   ``subject_mass`` exactly.
         This preserves AB's per-body physics-tuned distribution (pelvis ≈
         21 % BW, etc.) while keeping the COMAK extras (menisci, fatpad,
         knee subbodies) from inflating the total.
      5. Copy base Geometry/ → output_geometry_dir/.
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

    Subject mass handling:
        ``subject_mass``, when None (the default), is auto-detected by summing
        the AB model's body masses. AddBiomechanics tunes per-body masses to
        match the subject's actual measured mass, so this sum IS the subject's
        scaled mass and is the canonical source of truth. Pass an explicit
        float to override (e.g. for unit tests or when the AB tuning is
        suspect).

        AB's per-body masses are also transferred (not just the total) — see
        pipeline step 4 above and ``_apply_per_body_masses``. The two-pass
        renormalisation ensures the final total matches ``subject_mass``
        exactly even though COMAK has bodies that AB doesn't model (which
        otherwise causes a ~18% overcount).

    The report is always written so the scaling is recoverable from disk — the
    baked STLs and the model's reset ``scale_factors = [1, 1, 1]`` would
    otherwise hide how much the knee was scaled. The JSON records the isotropic
    ``s_wa`` ratio, the full per-body scale set (dimensionless ratios), and the
    per-body mass audit (base/provisional/final masses in kg). If ``report_json``
    is None, it defaults to ``output_osim.with_suffix(".scaling.json")``.
    Returns the report path.
    """
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

    output_osim.parent.mkdir(parents=True, exist_ok=True)

    factors = read_ab_factors(ab_scaled_osim)
    scale_set, s_wa = build_scale_set(factors, mode=mode)

    if subject_mass is None:
        subject_mass_effective = _total_body_mass(ab_scaled_osim)
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
    shutil.copytree(base_geom, output_geometry_dir)

    baked_geometry = bake_knee_geometry(
        scaled_osim=output_osim,
        knee_bodies=WA_KNEE_BODIES,
        scale=s_wa,
        geometry_dir=output_geometry_dir,
    )

    # Two-pass per-body mass transfer — runs AFTER geometry copy + bake so
    # the model loads cleanly (contact meshes are resolvable).
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
    scale_set_dump = {}
    for i in range(scale_set.getSize()):
        s = scale_set.get(i)
        sf = s.getScaleFactors()
        scale_set_dump[s.getSegmentName()] = (float(sf[0]), float(sf[1]), float(sf[2]))
    output_mass = _total_body_mass(output_osim)
    # Split the audit into the per-body table + scalar metadata before serialising.
    per_body_audit = {k: v for k, v in mass_audit.items() if not str(k).startswith("__")}
    renormalize_correction = mass_audit.get("__renormalize_correction", 1.0)
    provisional_total = mass_audit.get("__provisional_total_kg", subject_mass_effective)
    write_report(
        report_path,
        mode=mode,
        ab_scaled_osim=str(ab_scaled_osim),
        base_osim=str(base_osim),
        output_osim=str(output_osim),
        output_geometry_dir=str(output_geometry_dir),
        preserve_mass_distribution=preserve_mass_distribution,
        subject_mass_kg=subject_mass_effective,
        subject_mass_source=("override" if subject_mass is not None else "ab_total_body_mass"),
        base_total_mass_kg=base_mass,
        output_total_mass_kg=output_mass,
        mass_transfer_provisional_total_kg=provisional_total,
        mass_transfer_renormalize_correction=renormalize_correction,
        per_body_mass_audit=per_body_audit,
        ab_factors={k: list(v) for k, v in factors.items()},
        scale_set=scale_set_dump,
        wa_scale=s_wa,
        knee_geometry_baked={k: str(v) for k, v in baked_geometry.items()},
        marker_added=n_added,
        marker_dropped=n_dropped,
        backup_markerset_xml=str(backup_markerset_xml) if backup_markerset_xml else None,
        **fix_status,
    )
    return report_path
