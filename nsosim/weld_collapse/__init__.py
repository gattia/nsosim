"""Stage Z: COMAK weld-joint collapse.

Flattens a final, ready-to-simulate COMAK model by collapsing the intermediate
``WeldJoint``s out of the right-knee assembly. The collapsed model is physically
identical to the input -- same coordinates, forces, geometry, markers, contact
meshes -- but has 2 fewer bodies and 2 fewer joints and no intermediate
``WeldJoint``, removing the slow Simbody gradient code path that intermediate
welds trigger.

Public API::

    from nsosim.weld_collapse import collapse_welds
    collapse_welds(input_osim, output_osim)

See ``.claude/plans/comak-weld-collapse.md`` for the full design.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import opensim as osim

from .collapse import collapse_weld
from .report import write_report
from .topology import find_collapsible_welds, identify_sub_main

__all__ = ["collapse_welds", "find_collapsible_welds", "identify_sub_main", "collapse_weld"]

# The sub->main transform of a collapsible weld must be a pure translation:
# Stage Z only handles translation-only welds (no inertia-tensor rotation).
_ROTATION_IDENTITY_ATOL = 1e-9


def _rotation_error_from_identity(transform: osim.Transform) -> float:
    """Max absolute deviation of a transform's rotation matrix from identity."""
    rot = transform.R()
    mat = np.array([[rot.get(i, j) for j in range(3)] for i in range(3)])
    return float(np.abs(mat - np.eye(3)).max())


def collapse_welds(
    input_osim,
    output_osim,
    weld_names: Optional[list] = None,
    report_json=None,
) -> dict:
    """Top-level Stage Z orchestrator: collapse intermediate welds in a model.

    Parameters
    ----------
    input_osim
        Path to a final COMAK ``.osim`` carrying the welded right-knee seam.
    output_osim
        Path to write the collapsed ``.osim``. Geometry files are NOT copied --
        ``output_osim`` must be written where the input's relative ``Geometry/``
        references still resolve, or with that directory registered via
        ``osim.ModelVisualizer.addDirToGeometrySearchPaths``.
    weld_names
        Explicit list of intermediate ``WeldJoint``s to collapse. Default
        ``None`` auto-detects via :func:`find_collapsible_welds`.
    report_json
        Optional path for a JSON report sidecar.

    Returns
    -------
    dict
        Aggregate report: welds collapsed, per-weld details, body/joint counts.

    Steps
    -----
    1. Load a throwaway copy of ``input_osim`` and ``initSystem()`` it -- used
       only to detect welds and precompute, for each, the sub/main bodies and
       translation ``d`` (``findTransformBetween``). Each weld is asserted
       translation-only. The welds are independent, so all ``d`` come from this
       one realized state.
    2. Load a *fresh* copy of ``input_osim`` for the surgery and only
       ``finalizeFromProperties()`` it. ``initSystem()`` builds a Simbody
       system and transient components that cannot survive the structural
       edits, so the mutated model is never ``initSystem()``-ed.
    3. Collapse each weld in ``JointSet`` order (femur before tibia, for
       determinism) via :func:`collapse_weld` -- property edits only.
    4. ``finalizeConnections()`` once, on the final valid structure;
       ``printToXML(output_osim)``.
    5. Reload + ``initSystem()`` as a self-check; assert no ``WeldJoint``.
    6. Write the JSON report if requested.
    """
    input_osim = Path(input_osim)
    output_osim = Path(output_osim)

    # --- 1. throwaway initialized model: detect welds + precompute d -------
    probe = osim.Model(str(input_osim))
    state = probe.initSystem()

    if weld_names is None:
        weld_names = find_collapsible_welds(probe)

    specs = []
    for weld_name in weld_names:
        sub_name, main_name = identify_sub_main(probe, weld_name)
        sub = probe.getBodySet().get(sub_name)
        main = probe.getBodySet().get(main_name)
        transform = sub.findTransformBetween(state, main)
        rotation_error = _rotation_error_from_identity(transform)
        if rotation_error > _ROTATION_IDENTITY_ATOL:
            raise ValueError(
                f"weld '{weld_name}' is not translation-only (rotation deviates "
                f"from identity by {rotation_error:.2e}); Stage Z only collapses "
                "translation-only welds"
            )
        translation = np.array(
            [transform.p().get(0), transform.p().get(1), transform.p().get(2)],
            dtype=float,
        )
        specs.append((weld_name, sub_name, main_name, translation))

    # --- 2. fresh model for surgery: finalizeFromProperties only -----------
    model = osim.Model(str(input_osim))
    model.finalizeFromProperties()

    # --- 3. collapse each weld -- property edits only, no finalizeConnections.
    per_weld_reports = [
        collapse_weld(model, weld_name, sub_name, main_name, translation)
        for (weld_name, sub_name, main_name, translation) in specs
    ]

    # Single finalizeConnections on the final, valid structure, then write.
    model.finalizeConnections()
    model.printToXML(str(output_osim))

    # Self-check: the written model must reload, initialize, and be weld-free.
    reloaded = osim.Model(str(output_osim))
    reloaded.initSystem()
    n_welds_remaining = sum(
        1
        for i in range(reloaded.getJointSet().getSize())
        if osim.WeldJoint.safeDownCast(reloaded.getJointSet().get(i)) is not None
    )
    if n_welds_remaining != 0:
        raise RuntimeError(f"collapsed model still contains {n_welds_remaining} WeldJoint(s)")

    aggregate = {
        "input_osim": str(input_osim),
        "output_osim": str(output_osim),
        "n_welds_collapsed": len(per_weld_reports),
        "welds_collapsed": [r["weld"] for r in per_weld_reports],
        "n_bodies": reloaded.getBodySet().getSize(),
        "n_joints": reloaded.getJointSet().getSize(),
        "per_weld": per_weld_reports,
    }
    if report_json is not None:
        write_report(report_json, aggregate)
    return aggregate
