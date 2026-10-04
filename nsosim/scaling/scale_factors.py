"""Read AB scale factors and assemble a COMAK ScaleSet."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import opensim as osim

from .config import LONG_AXIS_INDEX, WA_KNEE_BODIES, ScalingMode


def _first_geometry(osim_path: Path) -> Dict[str, Tuple[str, Tuple[float, float, float]]]:
    """{body: (mesh file or geometry type, scale_factors)} of each body's first attached geometry."""
    model = osim.Model(str(osim_path))
    model.initSystem()
    out = {}
    bs = model.getBodySet()
    for i in range(bs.getSize()):
        body = bs.get(i)
        if body.getPropertyByName("attached_geometry").size() == 0:
            continue
        geom = body.upd_attached_geometry(0)
        mesh = osim.Mesh.safeDownCast(geom)
        sf = geom.get_scale_factors()
        key = mesh.get_mesh_file() if mesh is not None else geom.getConcreteClassName()
        out[body.getName()] = (key, (float(sf[0]), float(sf[1]), float(sf[2])))
    return out


def read_ab_factors(
    ab_scaled_osim: Path, generic_osim: Optional[Path] = None
) -> Dict[str, Tuple[float, float, float]]:
    """Extract per-body scale factors from AB's match_markers_and_physics.osim.

    AB records a body's scale only in its attached geometry: it writes each
    mesh's ``scale_factors`` as (generic mesh scale) x (body scale). Bodies with
    no attached geometry carry no record and are skipped.

    ``generic_osim`` is AB's generic, i.e. ``Models/unscaled_generic.osim``. With
    it, each factor is the ratio of a body's first mesh scale to that same mesh's
    scale in the generic. This is needed for a COMAK skeleton, whose meshes carry
    generic scales such as femur 0.925 and torso 0.80-0.93. The mesh must match
    by file, or the call raises.

    Without ``generic_osim``, the raw ``scale_factors`` are returned. That is
    right only when every generic mesh has scale 1, as in AB's stock Rajagopal
    models.
    """
    scaled = _first_geometry(ab_scaled_osim)
    if generic_osim is None:
        return {body: sf for body, (_, sf) in scaled.items()}
    generic = _first_geometry(generic_osim)
    factors: Dict[str, Tuple[float, float, float]] = {}
    for body, (key, sf) in scaled.items():
        if body not in generic or generic[body][0] != key:
            raise ValueError(
                f"{body}: AB's scaled mesh {key!r} has no counterpart in the generic {generic_osim} "
                f"(it has {generic.get(body, ('nothing',))[0]!r}); the scale cannot be recovered"
            )
        g = generic[body][1]
        factors[body] = (sf[0] / g[0], sf[1] / g[1], sf[2] / g[2])
    return factors


def _make_scale(body_name: str, sf: Tuple[float, float, float]) -> osim.Scale:
    s = osim.Scale()
    s.setSegmentName(body_name)
    s.setScaleFactors(osim.Vec3(*sf))
    s.setApply(True)
    return s


def build_scale_set(
    ab_factors: Dict[str, Tuple[float, float, float]],
    mode: ScalingMode = "WA",
) -> Tuple[osim.ScaleSet, float]:
    """Construct a ScaleSet per mode. Returns (scale_set, s_wa).

    All scale factors here are dimensionless ratios (mm/mm). ``s_wa`` is the
    isotropic weighted-average knee factor:

        s_wa = (ab_factors['femur_r'][1] + ab_factors['tibia_r'][1]) / 2

    i.e. the mean of the femur and tibia LONG-AXIS factors (index 1 =
    ``LONG_AXIS_INDEX``). In the femur_r/tibia_r body frames x = anterior-
    posterior, y = superior-inferior, z = mediolateral, so the long axis is
    index 1 -- see the HISTORY note in ``config.py``, this was index 2
    (mediolateral) until 2026-07-25. It is a single
    isotropic, dimensionless ratio (1.0 = no change). It is returned so
    ``bake_knee_geometry`` can multiply the knee STL vertices (meters) on disk
    by it, and so ``scale_comak_model`` can record it in the report.

    Modes
    -----
    "WA" (weighted-average, implemented):
        Every knee subbody (femur_distal_r, tibia_proximal_r, patella_r, the
        two menisci) is scaled isotropically by ``s_wa = mean(femur_r.y,
        tibia_r.y)``, whether or not AB returned a factor for it. AB-provided
        bodies pass their per-axis factors through unchanged. patella_r needs
        this in both AB setups: a Rajagopal-style skeleton has a patella and AB
        returns identity for it (no static-trial signal); a stripped COMAK
        skeleton has none, so AB returns nothing. Without a Scale, ScaleTool
        would leave the patella frames, and the tendon and ligament attachments
        on it, at reference size while ``bake_knee_geometry`` resizes its mesh.

    "LA" (long-axis, NOT IMPLEMENTED):
        Each knee subbody would receive ``(s_long, s_long, s_long)`` from its
        own parent bone's long-axis factor — i.e. femur subbodies use
        femur_r.y, tibia subbodies use tibia_r.y, patella uses its own (or
        the femur's). To add: branch in this function on ``mode == "LA"`` and
        build per-body isotropic Scales from the appropriate long-axis index
        in ``ab_factors``.

    "AB" (anisotropic pass-through, NOT IMPLEMENTED):
        AB's per-axis factors would propagate to knee subbodies — e.g.
        femur_distal_r inherits femur_r's full ``(sx, sy, sz)``. To add: pull
        the parent bone's full triplet from ``ab_factors`` and apply
        anisotropically. Note: bake_knee_geometry currently assumes a single
        scalar; supporting AB requires generalising it to per-axis scaling
        (the STL bake would also need to be anisotropic).

    LA and AB are documented for future resurrection only; the current
    pipeline targets WA exclusively.
    """
    if mode != "WA":
        raise NotImplementedError(f"Scaling mode {mode!r} not yet implemented")

    if "femur_r" not in ab_factors or "tibia_r" not in ab_factors:
        raise ValueError("AB factors missing femur_r or tibia_r — cannot compute WA factor")

    s_wa = (ab_factors["femur_r"][LONG_AXIS_INDEX] + ab_factors["tibia_r"][LONG_AXIS_INDEX]) / 2.0

    scale_set = osim.ScaleSet()

    # 1. AB-provided bodies pass through unchanged. Knee bodies are left to step 2,
    #    whatever AB returned for them (patella_r: identity, or nothing).
    for body_name, sf in ab_factors.items():
        if body_name not in WA_KNEE_BODIES:
            scale_set.adoptAndAppend(_make_scale(body_name, sf))

    # 2. Every knee subbody → isotropic WA.
    for body_name in WA_KNEE_BODIES:
        scale_set.adoptAndAppend(_make_scale(body_name, (s_wa, s_wa, s_wa)))

    return scale_set, s_wa
