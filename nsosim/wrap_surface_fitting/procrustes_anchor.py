"""Procrustes anchors for wrap surface fitting.

The wrap-fitter robustness plan (rev 2) replaces the algebraic geometric
init that the existing regularizer pins toward with a Procrustes-from-Smith2019
anchor: take Smith2019 reference wrap surface, sample its surface points,
transform them into the subject bone frame, then algebraic-fit a parametric
wrap to that transformed point cloud. The result is BOTH the L-BFGS
initialization AND the regularizer target — a single source of truth that
biases the fit toward the trusted reference geometry, not toward whatever
biased estimate the algebraic init happens to give on the subject bone.

This module provides the building blocks; iter5 wires them into the fitter.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from .main import wrap_surface

# ---------------------------------------------------------------------------
# Similarity Procrustes (Umeyama)
# ---------------------------------------------------------------------------


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute the rigid+isotropic-scale (similarity) transform mapping src → dst.

    Implements Umeyama 1991 in closed form. Both arrays must have the same
    shape ``(N, 3)`` with point correspondence row-by-row. The returned 4×4
    homogeneous matrix ``T`` satisfies ``T @ [src_i; 1] ≈ [dst_i; 1]``
    in the least-squares sense.

    Args:
        src: ``(N, 3)`` source points.
        dst: ``(N, 3)`` target points (same row order as src).

    Returns:
        4×4 homogeneous similarity transform.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape:
        raise ValueError(f"src and dst must have same shape, got {src.shape} vs {dst.shape}")
    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src/dst must be (N, 3), got {src.shape}")
    if src.shape[0] < 3:
        raise ValueError(f"Need at least 3 correspondences, got {src.shape[0]}")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # Variance of source (denominator for scale)
    var_src = (src_c**2).sum() / src.shape[0]

    # Cross-covariance dst·srcᵀ / N
    cov = dst_c.T @ src_c / src.shape[0]

    U, S, Vt = np.linalg.svd(cov)
    # Reflection correction so resulting rotation has det = +1
    d = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[2, 2] = -1.0
    R = U @ d @ Vt
    scale = (S * np.diag(d)).sum() / max(var_src, 1e-30)
    t = mu_dst - scale * R @ mu_src

    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4×4 affine transform to N×3 points and return N×3 points."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")
    return points @ T[:3, :3].T + T[:3, 3]


# ---------------------------------------------------------------------------
# Surface point sampling from wrap parameters
# ---------------------------------------------------------------------------


def _xyz_body_rotation_matrix(xyz_body_rotation) -> np.ndarray:
    """Convert OpenSim XYZ body-fixed Euler angles (radians) to a 3×3 rotation.

    Body-fixed XYZ = intrinsic XYZ in scipy = ``Rx @ Ry @ Rz`` as a matrix,
    matching ``RotationUtils.rot_to_euler_xyz_body``'s decomposition.
    """
    return ScipyRotation.from_euler(
        "XYZ", np.asarray(xyz_body_rotation, dtype=np.float64)
    ).as_matrix()


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Deterministic near-uniform unit-sphere sampling via the Fibonacci spiral."""
    idx = np.arange(n, dtype=np.float64)
    phi = (1.0 + 5.0**0.5) / 2.0
    z = 1.0 - 2.0 * (idx + 0.5) / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    theta = 2.0 * np.pi * idx / phi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)


def sample_ellipsoid_surface_points(wrap_params: Dict, n_points: int = 5000) -> np.ndarray:
    """Sample N points on the surface of an ellipsoid wrap surface.

    Args:
        wrap_params: dict with ``dimensions`` (3,), ``translation`` (3,),
            ``xyz_body_rotation`` (3,) — OpenSim wrap-surface schema.
        n_points: number of points to sample.

    Returns:
        ``(N, 3)`` points in the wrap's parent-body frame.
    """
    dims = np.asarray(wrap_params["dimensions"], dtype=np.float64)
    if dims.shape != (3,):
        raise ValueError(f"dimensions must be (3,), got {dims.shape}")
    trans = np.asarray(wrap_params["translation"], dtype=np.float64).reshape(3)
    R = _xyz_body_rotation_matrix(wrap_params["xyz_body_rotation"])

    unit = _fibonacci_sphere(n_points)  # (N, 3) on unit sphere
    scaled = unit * dims  # broadcast: per-axis radii in local frame
    return scaled @ R.T + trans


def sample_cylinder_surface_points(
    wrap_params: Dict, n_points: int = 5000, n_along: int = 50
) -> np.ndarray:
    """Sample N points on the lateral surface of a cylinder wrap surface.

    The cylinder local frame has its axis along +Z; the lateral surface is
    ``x = r cosθ, y = r sinθ, z ∈ [-L/2, L/2]``. Points are gridded
    deterministically: ``n_along`` rings × ``ceil(n_points / n_along)`` per
    ring, then trimmed to ``n_points``.

    Args:
        wrap_params: dict with ``radius``, ``length``, ``translation`` (3,),
            ``xyz_body_rotation`` (3,).
        n_points: target number of points.
        n_along: number of axial rings.

    Returns:
        ``(N, 3)`` points in the wrap's parent-body frame.
    """
    r = float(wrap_params["radius"])
    L = float(wrap_params["length"])
    trans = np.asarray(wrap_params["translation"], dtype=np.float64).reshape(3)
    R = _xyz_body_rotation_matrix(wrap_params["xyz_body_rotation"])

    n_around = int(np.ceil(n_points / max(n_along, 1)))
    z_levels = np.linspace(-L / 2.0, L / 2.0, n_along)
    # Phase-shift each ring deterministically to avoid co-aligned columns
    phi_shifts = (np.arange(n_along) / max(n_along, 1)) * (2 * np.pi / n_around)
    theta = np.linspace(0.0, 2.0 * np.pi, n_around, endpoint=False)

    pts = []
    for z, ds in zip(z_levels, phi_shifts):
        t = theta + ds
        ring = np.stack([r * np.cos(t), r * np.sin(t), np.full_like(t, z)], axis=1)
        pts.append(ring)
    pts = np.concatenate(pts, axis=0)[:n_points]
    return pts @ R.T + trans


def sample_wrap_surface_points(wrap_params: Dict, n_points: int = 5000) -> np.ndarray:
    """Dispatch to ellipsoid/cylinder sampler based on the wrap ``type``."""
    surface_type = wrap_params.get("type", "")
    if surface_type == "WrapEllipsoid":
        return sample_ellipsoid_surface_points(wrap_params, n_points)
    if surface_type == "WrapCylinder":
        return sample_cylinder_surface_points(wrap_params, n_points)
    raise ValueError(f"Unsupported wrap surface type: {surface_type!r}")


# ---------------------------------------------------------------------------
# Algebraic refit of transformed point clouds → wrap_surface anchor
# ---------------------------------------------------------------------------


def _fit_ellipsoid_anchor(points: np.ndarray) -> Dict[str, np.ndarray]:
    """Algebraic ellipsoid fit on a point cloud, returning numpy arrays.

    Wraps ``surface_param_estimation.fit_ellipsoid_algebraic`` and returns a
    plain dict in body-frame Cartesian. The returned ``rotation`` is a 3×3
    matrix with columns = local axes of the ellipsoid.
    """
    import torch

    from . import surface_param_estimation

    pts_t = torch.from_numpy(np.asarray(points, dtype=np.float64)).float()
    out = surface_param_estimation.fit_ellipsoid_algebraic(pts_t)
    if not bool(out["success"]):
        raise RuntimeError("Algebraic ellipsoid fit failed on Procrustes anchor points")

    return {
        "center": out["center"].detach().cpu().numpy().astype(np.float64),
        "axes": out["axes"].detach().cpu().numpy().astype(np.float64),
        "rotation": out["rotation"].detach().cpu().numpy().astype(np.float64),
    }


def _fit_cylinder_anchor(points: np.ndarray) -> Dict[str, np.ndarray]:
    """Slice-PCA cylinder fit on a point cloud, returning numpy arrays."""
    import torch

    from . import surface_param_estimation

    pts_t = torch.from_numpy(np.asarray(points, dtype=np.float64)).float()
    out = surface_param_estimation.fit_cylinder_geometric(pts_t)
    if not bool(out["success"]):
        raise RuntimeError("Geometric cylinder fit failed on Procrustes anchor points")
    return {
        "center": out["center"].detach().cpu().numpy().astype(np.float64),
        "radius": float(out["radius"]),
        "half_length": float(out["half_length"]),
        "axis": out["axis"].detach().cpu().numpy().astype(np.float64),
        "rotation": out["rotation"].detach().cpu().numpy().astype(np.float64),
    }


def _canonical_ellipsoid_pose(axes: np.ndarray, R: np.ndarray) -> tuple:
    """Sort axes descending and apply dominant-component sign convention.

    Matches ``RotationUtils.canonical_ellipsoid_pose`` (iter 1) so the anchor's
    Euler representation is bit-stable across the gimbal-lock region. Returns
    ``(axes_sorted, R_canonical)``.
    """
    order = np.argsort(-axes)  # descending
    axes_sorted = axes[order]
    R_sorted = R[:, order]
    # Sign by dominant component of each column
    for c in range(3):
        col = R_sorted[:, c]
        dom = int(np.argmax(np.abs(col)))
        if col[dom] < 0:
            R_sorted[:, c] = -col
    # If determinant flipped to -1, flip the third column (it has the smallest
    # axis so its sign carries the least geometric meaning).
    if np.linalg.det(R_sorted) < 0:
        R_sorted[:, 2] = -R_sorted[:, 2]
    return axes_sorted, R_sorted


def procrustes_anchor_for_wrap(
    wrap_name: str,
    smith2019_wrap_params: Dict,
    bone_transform: Optional[np.ndarray] = None,
    n_points: int = 5000,
    body: Optional[str] = None,
) -> wrap_surface:
    """Build a wrap_surface anchor from Smith2019 reference for one wrap.

    Pipeline:
      1. Sample N points on the Smith2019 wrap surface (in Smith2019 body frame).
      2. Apply ``bone_transform`` (similarity Procrustes from Smith2019 bone →
         subject bone). Defaults to identity — the OSIM body frame is shared
         by construction; only bone-shape variation requires a non-identity
         transform.
      3. Algebraic-refit a parametric wrap to the transformed points.
      4. Wrap parameters in a ``wrap_surface`` data class object.

    Args:
        wrap_name: name of the wrap (e.g. ``"Capsule_r"``).
        smith2019_wrap_params: per-wrap parameter dict, exactly the value
            stored at ``wrap_params[bone][body_name][wrap_name]`` from
            ``extract_wrap_parameters_from_osim``.
        bone_transform: optional 4×4 similarity from Smith2019 → subject
            (in the wrap's parent-body frame). ``None`` ⇒ identity.
        n_points: number of surface points to sample for the refit.
        body: parent body name (stored on the returned ``wrap_surface``).

    Returns:
        ``wrap_surface`` object with the fitted anchor parameters. The
        ``quadrant`` of the original Smith2019 wrap is NOT carried over — the
        anchor is consumed by the optimizer; quadrant handling lives downstream
        in ``construct_cylinder_basis`` / `enforce_sign_convention`.
    """
    if bone_transform is None:
        T = np.eye(4)
    else:
        T = np.asarray(bone_transform, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"bone_transform must be 4×4, got {T.shape}")

    smith_points = sample_wrap_surface_points(smith2019_wrap_params, n_points=n_points)
    subj_points = transform_points(T, smith_points)

    surface_type = smith2019_wrap_params.get("type", "")
    if surface_type == "WrapEllipsoid":
        fit = _fit_ellipsoid_anchor(subj_points)
        axes, R = _canonical_ellipsoid_pose(fit["axes"], fit["rotation"])
        euler = ScipyRotation.from_matrix(R).as_euler("XYZ")
        return wrap_surface(
            name=wrap_name,
            body=body,
            type_="WrapEllipsoid",
            xyz_body_rotation=euler,
            translation=fit["center"],
            radius=None,
            length=None,
            dimensions=axes,
        )

    if surface_type == "WrapCylinder":
        fit = _fit_cylinder_anchor(subj_points)
        R = fit["rotation"]
        euler = ScipyRotation.from_matrix(R).as_euler("XYZ")
        return wrap_surface(
            name=wrap_name,
            body=body,
            type_="WrapCylinder",
            xyz_body_rotation=euler,
            translation=fit["center"],
            radius=fit["radius"],
            length=2.0 * fit["half_length"],
            dimensions=None,
        )

    raise ValueError(f"Unsupported wrap surface type: {surface_type!r}")


def procrustes_anchors_from_smith2019(
    smith2019_osim_path: str,
    bone_transforms: Optional[Dict[str, np.ndarray]] = None,
    n_points: int = 5000,
) -> Dict[str, Dict[str, Dict[str, Dict[str, wrap_surface]]]]:
    """Build Procrustes anchors for every wrap in ``DEFAULT_SMITH2019_BONES``.

    Returns the same nested-dict structure as ``fit_bone_wrap_surfaces``
    output: ``{bone: {body: {surface_type: {wrap_name: wrap_surface}}}}``.

    Args:
        smith2019_osim_path: path to ``smith2019.osim``.
        bone_transforms: optional ``{bone_name: 4×4 similarity}`` mapping. Any
            missing bone defaults to identity. The transform is applied in the
            wrap's parent-body frame.
        n_points: surface points per wrap to sample for the algebraic refit.
    """
    from .parameter_extraction import extract_wrap_parameters_from_osim

    bone_transforms = bone_transforms or {}
    params = extract_wrap_parameters_from_osim(smith2019_osim_path)

    out: Dict[str, Dict[str, Dict[str, Dict[str, wrap_surface]]]] = {}
    for bone_name, bone_data in params.items():
        T = bone_transforms.get(bone_name)
        out[bone_name] = {}
        for body_name, body_data in bone_data.items():
            out[bone_name][body_name] = {}
            for wrap_name, wrap_p in body_data.items():
                stype = wrap_p.get("type", "")
                key = "ellipsoid" if stype == "WrapEllipsoid" else "cylinder"
                out[bone_name][body_name].setdefault(key, {})
                anchor = procrustes_anchor_for_wrap(
                    wrap_name,
                    wrap_p,
                    bone_transform=T,
                    n_points=n_points,
                    body=body_name,
                )
                out[bone_name][body_name][key][wrap_name] = anchor
    return out
