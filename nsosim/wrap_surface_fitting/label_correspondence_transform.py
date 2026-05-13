"""Subject-adapted wrap surfaces via label-correspondence Procrustes.

Background
----------
The Smith2019-anchor path (``procrustes_anchor.procrustes_anchors_from_smith2019``)
places every subject's wrap at Smith2019's literal body-frame coordinates,
regardless of how the subject's bone shape differs from Smith2019. When the
subject's anatomy diverges, the labeled "inside-wrap" region transfers via
NSM correspondence to a different position on the subject's bone than where
Smith2019 puts its wrap — and the LBFGS fitter, regularized toward the
non-adapted anchor, can't move far enough to follow the labels (catastrophic
on Med_LigP_r for some subjects; ~1.5 pp F1 above naive baseline on average).

This module computes a per-wrap subject-adapted wrap surface directly from
the available label correspondence:

  1. For each wrap, take the reference labeled bone's ``{wrap}_near_surface``
     mask: these are the bone vertices that lie close to Smith2019's wrap
     on the reference (NSM mean) bone.
  2. Pull the corresponding indices on the subject's NSM-adapted labeled
     bone — they're the anatomically matching points on this subject.
  3. Solve similarity Procrustes (Umeyama) on ``ref → subj`` near-surface
     points. The result is a 4×4 similarity transform that maps the
     reference wrap-region into the subject's wrap-region.
  4. Apply that transform to Smith2019's wrap parameters via
     ``procrustes_anchor_for_wrap`` (which samples Smith2019's surface,
     applies the transform, refits the parametric wrap). The output is a
     subject-adapted ``wrap_surface``.

The result is used **directly** as the fitted wrap — no LBFGS — when called
from ``build_joint_model`` with ``wrap_fit_mode='label_correspondence'``.
Validation on 10 OAI subjects (Step-1 A/B determinism set) shows:

  - F1 across the 5 anchor-fittable ellipsoids + cylinder beats the
    Smith2019-anchor + LBFGS approach by 0.07–12.27 pp.
  - A↔B reproducibility under sub-µm bone-mesh drift is uniformly < 16 µm
    on every wrap, < 11 µm on the previously catastrophic 9168709 Med_LigP_r.
  - No multi-minima (Med_Lig_r's iter8.5 opt-out is unnecessary under this
    approach because the anchor is now in the data-optimum basin by
    construction).

Patella wraps (``PatTen_r``) are NOT handled here — they use
``PatellaFitter`` in ``fit_bone_wrap_surfaces``, which has its own approach.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .main import wrap_surface
from .parameter_extraction import extract_wrap_parameters_from_osim
from .procrustes_anchor import procrustes_anchor_for_wrap, umeyama_similarity


_NEAR_SURFACE_MIN_POINTS = 20


def _per_wrap_similarity_from_labels(
    ref_labeled_mesh: Any,
    subj_labeled_mesh: Any,
    wrap_name: str,
) -> Optional[np.ndarray]:
    """Compute the 4×4 similarity that aligns reference near-surface points
    to subject near-surface points for one wrap.

    Returns None if the wrap's ``_near_surface`` array is missing or has too
    few points (< 20). The reference and subject meshes must have matching
    vertex indices (NSM correspondence).

    Args:
        ref_labeled_mesh: PyVista PolyData for the reference (mean) bone
            with ``{wrap}_near_surface`` int/bool point data arrays.
        subj_labeled_mesh: PyVista PolyData for the subject's NSM-adapted
            bone with the same point count and vertex correspondence.
        wrap_name: e.g. ``"Med_LigP_r"``.

    Returns:
        4×4 homogeneous similarity transform, or None if not computable.
    """
    key = f"{wrap_name}_near_surface"
    if key not in ref_labeled_mesh.point_data:
        return None

    mask = np.asarray(ref_labeled_mesh.point_data[key]).astype(bool)
    if int(mask.sum()) < _NEAR_SURFACE_MIN_POINTS:
        return None

    ref_pts = np.asarray(ref_labeled_mesh.points)[mask]
    subj_pts = np.asarray(subj_labeled_mesh.points)[mask]
    if ref_pts.shape != subj_pts.shape:
        raise ValueError(
            f"Vertex count mismatch for {wrap_name}: "
            f"ref={ref_pts.shape}, subj={subj_pts.shape}. "
            "Reference and subject labeled meshes must share NSM correspondence."
        )

    return umeyama_similarity(ref_pts, subj_pts)


def label_correspondence_transforms_for_bone(
    smith2019_osim_path: str,
    bone_name: str,
    ref_labeled_mesh: Any,
    subj_labeled_mesh: Any,
) -> Dict[str, Dict[str, Dict[str, wrap_surface]]]:
    """Build subject-adapted wrap surfaces for every wrap on one bone.

    Args:
        smith2019_osim_path: path to ``smith2019.osim`` (provides the
            reference wrap parameters that get transformed per-subject).
        bone_name: ``"femur"``, ``"tibia"``, or ``"patella"``. The function
            only processes wraps Smith2019 lists under this bone.
        ref_labeled_mesh: reference labeled mesh for this bone (PyVista
            PolyData with ``{wrap}_near_surface`` arrays).
        subj_labeled_mesh: subject's NSM-adapted labeled mesh for this bone
            (same vertex count + index correspondence as ref).

    Returns:
        Nested dict ``{body_name: {surface_type: {wrap_name: wrap_surface}}}``
        — the same per-bone shape returned by
        ``procrustes_anchors_from_smith2019()[bone_name]``. Wraps that lack
        a ``_near_surface`` label are omitted (caller can fall back).
    """
    params = extract_wrap_parameters_from_osim(smith2019_osim_path)
    bone_params = params.get(bone_name, {})

    out: Dict[str, Dict[str, Dict[str, wrap_surface]]] = {}
    for body_name, body_data in bone_params.items():
        for wrap_name, wrap_p in body_data.items():
            T = _per_wrap_similarity_from_labels(
                ref_labeled_mesh, subj_labeled_mesh, wrap_name
            )
            if T is None:
                continue
            ws = procrustes_anchor_for_wrap(
                wrap_name=wrap_name,
                smith2019_wrap_params=wrap_p,
                bone_transform=T,
                body=body_name,
            )
            stype = "ellipsoid" if ws.type_ == "WrapEllipsoid" else "cylinder"
            out.setdefault(body_name, {}).setdefault(stype, {})[wrap_name] = ws
    return out
