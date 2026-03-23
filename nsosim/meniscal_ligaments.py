"""
Post-interpolation correction for meniscal ligament tibia attachments.

Meniscal ligaments (coronary, anterior/posterior horn) connect the meniscus to the
tibia plateau and should be roughly vertical. Because the meniscus and tibia are modeled
by separate NSMs, interpolated attachment points can diverge laterally. This module
re-derives the tibia-side attachment by casting a short ray from the meniscus point
onto the tibia mesh surface.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Meniscal ligament name prefixes (P1 on tibia, P2 on meniscus)
_MENISCAL_PREFIXES = ("meniscus_medial_", "meniscus_lateral_")
_EXCLUDE_PATTERNS = ("TRANSVLIG",)

# Parent frame names for tibia vs meniscus bodies
_TIBIA_FRAMES = ("tibia_proximal_r",)
_MENISCUS_FRAMES = ("meniscus_medial_r", "meniscus_lateral_r")


def _is_meniscal_tibia_ligament(name):
    """Check if a ligament name is a meniscal-tibia ligament (not transverse)."""
    if not any(name.startswith(prefix) for prefix in _MENISCAL_PREFIXES):
        return False
    if any(pattern in name for pattern in _EXCLUDE_PATTERNS):
        return False
    return True


def _identify_tibia_meniscus_points(points):
    """Find which point index is tibia vs meniscus by parent_frame.

    Returns:
        tuple: (tibia_idx, meniscus_idx) or (None, None) if not identifiable
    """
    tibia_idx = None
    meniscus_idx = None

    for i, point in enumerate(points):
        frame = point.get("parent_frame", "")
        if frame in _TIBIA_FRAMES:
            tibia_idx = i
        elif frame in _MENISCUS_FRAMES:
            meniscus_idx = i

    return tibia_idx, meniscus_idx


def project_meniscal_attachments_to_tibia(
    dict_lig_mus_attach,
    tibia_mesh,
    ray_direction=None,
    max_ray_length=0.015,
    xyz_key="xyz_mesh_updated",
):
    """Project meniscal ligament tibia attachments onto the tibia surface.

    For each meniscal ligament, keeps the meniscus-side attachment as-is and
    re-derives the tibia-side attachment by casting a short ray in the distal
    direction from the meniscus point onto the tibia mesh surface.

    Modifies dict_lig_mus_attach in-place.

    Args:
        dict_lig_mus_attach: Standard ligament/muscle attachment dict. Each entry
            has 'points' list with 'parent_frame' and xyz_key fields.
        tibia_mesh: PyVista PolyData or pymskt Mesh of the tibia in OpenSim space (meters).
        ray_direction: Ray direction vector. Default [0, -1, 0] (-Y = distal in OpenSim).
        max_ray_length: Maximum ray length in meters. Default 0.015 (15mm).
        xyz_key: Key in point dict for the coordinates to read/update.
            Default 'xyz_mesh_updated'.

    Returns:
        dict: Diagnostic summary per ligament:
            {ligament_name: {'method': 'ray'|'nearest', 'distance': float}}
    """
    if ray_direction is None:
        ray_direction = np.array([0.0, -1.0, 0.0])
    else:
        ray_direction = np.asarray(ray_direction, dtype=float)

    # Normalize direction
    norm = np.linalg.norm(ray_direction)
    if norm < 1e-12:
        raise ValueError("ray_direction must be a non-zero vector")
    ray_direction = ray_direction / norm

    results = {}

    for lig_name, lig_dict in dict_lig_mus_attach.items():
        if not _is_meniscal_tibia_ligament(lig_name):
            continue

        points = lig_dict.get("points", [])
        tibia_idx, meniscus_idx = _identify_tibia_meniscus_points(points)

        if tibia_idx is None or meniscus_idx is None:
            logger.warning(
                f"Could not identify tibia/meniscus points for '{lig_name}', skipping"
            )
            continue

        meniscus_point = np.asarray(points[meniscus_idx][xyz_key], dtype=float)
        origin = meniscus_point.copy()
        end_point = origin + ray_direction * max_ray_length

        # Cast ray from meniscus point toward tibia
        intersection_points, _ = tibia_mesh.ray_trace(origin, end_point)

        if len(intersection_points) > 0:
            # Use the first (closest) intersection
            new_tibia_point = np.array(intersection_points[0])
            method = "ray"
            distance = np.linalg.norm(new_tibia_point - meniscus_point)
            logger.debug(
                f"{lig_name}: ray hit at distance {distance * 1000:.2f}mm"
            )
        else:
            # Fallback: nearest point on tibia surface
            closest_idx = tibia_mesh.find_closest_point(meniscus_point)
            new_tibia_point = np.array(tibia_mesh.points[closest_idx])
            method = "nearest"
            distance = np.linalg.norm(new_tibia_point - meniscus_point)
            logger.warning(
                f"{lig_name}: ray missed tibia, using nearest point "
                f"(distance={distance * 1000:.2f}mm)"
            )

        # Warn if meniscus point is below the tibia surface (possible extrusion)
        if meniscus_point[1] < new_tibia_point[1]:
            logger.warning(
                f"{lig_name}: meniscus point Y ({meniscus_point[1]:.6f}) is below "
                f"tibia surface Y ({new_tibia_point[1]:.6f}) — possible extrusion"
            )

        # Update tibia point in-place
        points[tibia_idx][xyz_key] = new_tibia_point

        results[lig_name] = {"method": method, "distance": distance}

    logger.info(
        f"Projected {len(results)} meniscal ligament attachments to tibia surface"
    )
    return results
