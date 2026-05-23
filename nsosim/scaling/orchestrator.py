"""Top-level orchestrator for Stage X (COMAK body scaling)."""

import logging
import shutil
from pathlib import Path
from typing import Optional

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


def scale_comak_model(
    base_osim: Path,
    ab_scaled_osim: Path,
    output_osim: Path,
    output_geometry_dir: Path,
    mode: ScalingMode = "WA",
    preserve_mass_distribution: bool = True,
    report_json: Optional[Path] = None,
    backup_markerset_xml: Optional[Path] = None,
) -> Path:
    """Produce a scaled COMAK model from a base + AB output.

    Pipeline:
      1. Read AB per-body scale factors.
      2. Build a ScaleSet according to `mode`.
      3. ScaleTool.run() → output_osim.
      4. Copy base Geometry/ → output_geometry_dir/.
      5. Bake the knee-body geometry into its STLs (pymskt vertex scale,
         visual scale_factors reset to 1).
      6. Swap MarkerSet for AB's static-trial-placed markers.
      7. Apply permanent model fix-ups (ITB1 reparent).
      8. Write the JSON report (always — see below).

    The output `.osim` is at `output_osim`; the orchestrator requires
    `output_geometry_dir == output_osim.parent / "Geometry"` so OpenSim's default
    relative-path resolution finds the meshes.

    The report is always written so the WA scale and per-body factors are
    recoverable from disk — the baked STLs and the model's reset
    ``scale_factors = [1, 1, 1]`` would otherwise hide how much the knee was
    scaled. If ``report_json`` is None, it defaults to
    ``output_osim.with_suffix(".scaling.json")``. Returns the report path.
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

    n_added, n_dropped = swap_markerset(
        scaled_osim=output_osim,
        ab_scaled_osim=ab_scaled_osim,
        backup_xml_path=backup_markerset_xml,
    )

    fix_status = fix_in_place(output_osim)

    report_path = Path(report_json) if report_json is not None else output_osim.with_suffix(
        ".scaling.json"
    )
    scale_set_dump = {}
    for i in range(scale_set.getSize()):
        s = scale_set.get(i)
        sf = s.getScaleFactors()
        scale_set_dump[s.getSegmentName()] = (float(sf[0]), float(sf[1]), float(sf[2]))
    write_report(
        report_path,
        mode=mode,
        ab_scaled_osim=str(ab_scaled_osim),
        base_osim=str(base_osim),
        output_osim=str(output_osim),
        output_geometry_dir=str(output_geometry_dir),
        preserve_mass_distribution=preserve_mass_distribution,
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
