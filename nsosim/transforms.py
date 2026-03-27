"""Similarity transform utilities for NSM alignment transforms.

Functions for decomposing, analyzing, and recomposing the 4x4 similarity
transforms stored in NSM alignment JSONs. These transforms map between
femur-aligned space (mm) and per-bone NSM canonical space (~[-1, 1]).

Key concepts:
    - ``linear_transform`` in alignment JSONs: 4x4 similarity matrix where
      T[:3,:3] = scale * R (uniform scaling embedded in the rotation block).
    - ``T_rel = T_fem @ inv(T_other)``: relative transform mapping canonical
      other-bone space → canonical femur space.
    - Deviations from the population mean are expressed as Euler angles (deg),
      translation (mm), and scale ratio — suitable for statistical modeling.

Adapted from proven code in the ACL project (52 subjects, validated roundtrip).
See ``pratham_ACL_wcb/scripts/Tibia_rotations/`` for the original scripts.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def decompose_similarity(T):
    """Decompose a 4x4 similarity transform into (scale, R, t).

    The 3x3 upper-left submatrix encodes ``scale * R`` where scale is uniform
    (same column norm for all three columns) and R is a proper rotation (det +1).

    Args:
        T: (4, 4) similarity transform matrix.

    Returns:
        scale (float): Uniform scale factor (column norm of the 3x3 submatrix).
        R (ndarray, (3, 3)): Proper rotation matrix (det = +1).
        t (ndarray, (3,)): Translation vector (copy of T[:3, 3]).
    """
    upper = T[:3, :3]
    t = T[:3, 3].copy()
    scale = np.linalg.norm(upper[:, 0])
    R = upper / scale
    if np.linalg.det(R) < 0:
        R = -R
        scale = -scale
    return scale, R, t


def mean_rotation(rotations):
    """Compute mean rotation via element-wise average + SVD projection.

    The arithmetic mean of rotation matrices is not generally a valid rotation.
    This projects the mean onto the nearest proper rotation (det = +1) using SVD.

    Args:
        rotations: (N, 3, 3) array of proper rotation matrices.

    Returns:
        (3, 3) proper rotation matrix nearest to the arithmetic mean.
    """
    R_raw = np.mean(rotations, axis=0)
    U, _, Vt = np.linalg.svd(R_raw)
    d = np.linalg.det(U @ Vt)
    return U @ np.diag([1, 1, d]) @ Vt


def compute_T_rel(T_fem, T_other):
    """Compute relative transform: T_fem @ inv(T_other).

    The result maps points from canonical other-bone space to canonical femur
    space. This captures the joint configuration (where the other bone sits
    relative to the femur) independently of each bone's shape.

    Args:
        T_fem: (4, 4) femur's linear_transform (femur-aligned → canonical femur).
        T_other: (4, 4) other bone's linear_transform (femur-aligned → canonical other).

    Returns:
        (4, 4) relative transform (canonical other → canonical femur).
    """
    return T_fem @ np.linalg.inv(T_other)


def recover_bone_transform(T_rel, T_fem):
    """Recover a bone's linear_transform from relative transform and T_fem.

    Inverse of ``compute_T_rel``: given ``T_rel = T_fem @ inv(T_bone)``,
    recovers ``T_bone = inv(T_rel) @ T_fem``.

    Args:
        T_rel: (4, 4) relative transform (from ``compute_T_rel``).
        T_fem: (4, 4) femur's linear_transform.

    Returns:
        (4, 4) the bone's linear_transform (femur-aligned → canonical bone).
    """
    return np.linalg.inv(T_rel) @ T_fem


def compute_transform_deviations(transforms, mean_fem_scale):
    """Decompose an array of similarity transforms into per-subject deviations from the mean.

    Each transform is decomposed into (scale, R, t). The mean of each component
    is computed independently (arithmetic mean for scale/translation, SVD-projected
    mean for rotation). Deviations are expressed relative to the mean:
        - Rotation: Euler XYZ angles (degrees) of ``R_mean.T @ R_i``
        - Translation: ``(t_i - t_mean) / mean_fem_scale`` (mm)
        - Scale: ``s_i / s_mean`` (dimensionless ratio, ~1.0)

    Args:
        transforms: (N, 4, 4) array of similarity transforms (e.g., T_rel values).
        mean_fem_scale (float): Mean femur scale factor for canonical ↔ mm conversion.
            Typically ~0.013, obtained from ``decompose_similarity(T_fem)`` averaged
            across subjects.

    Returns:
        dict with keys:
            R_mean (ndarray, (3, 3)): Mean rotation matrix.
            t_mean (ndarray, (3,)): Mean translation in canonical units.
            s_mean (float): Mean scale factor.
            euler_angles_deg (ndarray, (N, 3)): Per-subject rotation deviations (XYZ degrees).
            translations_mm (ndarray, (N, 3)): Per-subject translation deviations (mm).
            scale_ratios (ndarray, (N,)): Per-subject scale ratios (subject / mean).
    """
    N = len(transforms)
    scales = np.empty(N)
    rotations = np.empty((N, 3, 3))
    translations = np.empty((N, 3))

    for i in range(N):
        scales[i], rotations[i], translations[i] = decompose_similarity(transforms[i])

    s_mean = float(np.mean(scales))
    R_mean = mean_rotation(rotations)
    t_mean = np.mean(translations, axis=0)

    euler_angles_deg = np.empty((N, 3))
    translations_mm = np.empty((N, 3))
    scale_ratios = np.empty(N)

    for i in range(N):
        R_dev = R_mean.T @ rotations[i]
        euler_angles_deg[i] = Rotation.from_matrix(R_dev).as_euler("XYZ", degrees=True)
        translations_mm[i] = (translations[i] - t_mean) / mean_fem_scale
        scale_ratios[i] = scales[i] / s_mean

    return {
        "R_mean": R_mean,
        "t_mean": t_mean,
        "s_mean": s_mean,
        "euler_angles_deg": euler_angles_deg,
        "translations_mm": translations_mm,
        "scale_ratios": scale_ratios,
    }


def deviations_to_transform(
    euler_angles_deg,
    translation_mm,
    scale_ratio,
    R_mean,
    t_mean,
    s_mean,
    mean_fem_scale,
):
    """Recompose deviation values into a full 4x4 similarity transform.

    Inverse of the decomposition in ``compute_transform_deviations``. Pass
    ``[0, 0, 0]`` for angles/translation and ``1.0`` for scale_ratio to
    recover the mean transform.

    Math::

        R_full   = R_mean @ Rotation.from_euler("XYZ", angles_deg).as_matrix()
        t_full   = t_mean + translation_mm * mean_fem_scale
        s_full   = s_mean * scale_ratio
        T[:3,:3] = s_full * R_full
        T[:3, 3] = t_full

    Args:
        euler_angles_deg (array-like, (3,)): Rotation deviation (XYZ degrees).
        translation_mm (array-like, (3,)): Translation deviation (mm).
        scale_ratio (float): Scale deviation (subject / mean).
        R_mean (ndarray, (3, 3)): Mean rotation matrix.
        t_mean (ndarray, (3,)): Mean translation (canonical units).
        s_mean (float): Mean scale factor.
        mean_fem_scale (float): Femur scale for mm ↔ canonical conversion.

    Returns:
        (4, 4) similarity transform.
    """
    euler_angles_deg = np.asarray(euler_angles_deg, dtype=float)
    translation_mm = np.asarray(translation_mm, dtype=float)

    R_dev = Rotation.from_euler("XYZ", euler_angles_deg, degrees=True).as_matrix()
    R_full = R_mean @ R_dev

    t_full = t_mean + translation_mm * mean_fem_scale

    s_full = s_mean * scale_ratio

    T = np.eye(4)
    T[:3, :3] = s_full * R_full
    T[:3, 3] = t_full
    return T
