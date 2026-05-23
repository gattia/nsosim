"""Read AB scale factors and assemble a COMAK ScaleSet."""

from pathlib import Path
from typing import Dict, Tuple

import opensim as osim

from .config import LONG_AXIS_INDEX, WA_KNEE_BODIES, ScalingMode


def read_ab_factors(ab_scaled_osim: Path) -> Dict[str, Tuple[float, float, float]]:
    """Extract per-body scale factors from AB's match_markers_and_physics.osim.

    Reads each Body's first <attached_geometry>/scale_factors field. Bodies
    with zero attached geometries are skipped (no signal to read).
    """
    model = osim.Model(str(ab_scaled_osim))
    model.initSystem()

    factors: Dict[str, Tuple[float, float, float]] = {}
    bs = model.getBodySet()
    for i in range(bs.getSize()):
        body = bs.get(i)
        n_geom = body.getPropertyByName("attached_geometry").size()
        if n_geom == 0:
            continue
        sf = body.upd_attached_geometry(0).get_scale_factors()
        factors[body.getName()] = (float(sf[0]), float(sf[1]), float(sf[2]))
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

    s_wa is the isotropic weighted-average knee factor — used downstream by
    bake_knee_geometry to scale the knee STLs on disk.

    Modes
    -----
    "WA" (weighted-average, implemented):
        Every knee subbody (femur_distal_r, tibia_proximal_r, patella_r, the
        two menisci) is scaled isotropically by ``s_wa = mean(femur_r.z,
        tibia_r.z)``. AB-provided bodies pass their per-axis factors through
        unchanged, except patella_r (AB returns identity → use s_wa).

    "LA" (long-axis, NOT IMPLEMENTED):
        Each knee subbody would receive ``(s_long, s_long, s_long)`` from its
        own parent bone's long-axis factor — i.e. femur subbodies use
        femur_r.z, tibia subbodies use tibia_r.z, patella uses its own (or
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

    # 1. AB-provided bodies pass through unchanged, except patella_r (AB returns
    #    identity, no static-trial signal).
    for body_name, sf in ab_factors.items():
        if body_name == "patella_r":
            scale_set.adoptAndAppend(_make_scale(body_name, (s_wa, s_wa, s_wa)))
        else:
            scale_set.adoptAndAppend(_make_scale(body_name, sf))

    # 2. Knee subbodies stripped before AB upload → isotropic WA.
    for body_name in WA_KNEE_BODIES:
        if body_name == "patella_r":
            continue  # already added above
        scale_set.adoptAndAppend(_make_scale(body_name, (s_wa, s_wa, s_wa)))

    return scale_set, s_wa
