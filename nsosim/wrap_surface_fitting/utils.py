"""Data loading, SDF computation, and point classification for wrap surface fitting."""

import logging
import xml.etree.ElementTree as ET
from typing import Dict, Tuple, Union

import numpy as np
import pyvista as pv
from pymskt.mesh import Mesh

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Constants
ADDITIONAL_OFFSETS = {
    "femur_r": [-0.0055513564376633642, -0.37418143637169787, -0.0011706232813375212]
}


def convert_fitted_ellipsoid_to_wrap_params(center, axes=None, rotation_matrix=None):
    """
    Convert fitted ellipsoid parameters to the format expected by create_ellipsoid_polydata.

    Args:
        center: Either:
            - (3,) tensor or array - ellipsoid center coordinates (if axes and rotation_matrix provided)
            - tuple of (center, axes, rotation_matrix) from EllipsoidFitter.fit() output
        axes: (3,) tensor or array - ellipsoid semi-axes lengths [a, b, c] (optional if center is tuple)
        rotation_matrix: (3,3) tensor or array - rotation matrix (optional if center is tuple)

    Returns:
        Dictionary with keys expected by create_ellipsoid_polydata:
        - 'translation': center coordinates as list
        - 'dimensions': semi-axes as list
        - 'xyz_body_rotation': euler angles in radians as list
    """
    import torch

    # Handle case where center is actually the full tuple from EllipsoidFitter.fit()
    if axes is None and rotation_matrix is None:
        if isinstance(center, (tuple, list)) and len(center) == 3:
            center, axes, rotation_matrix = center
        else:
            raise ValueError(
                "If axes and rotation_matrix are None, center must be a tuple of (center, axes, rotation_matrix)"
            )

    # Convert to numpy if torch tensors
    if hasattr(center, "detach"):
        center = center.detach().cpu().numpy()
    if hasattr(axes, "detach"):
        axes = axes.detach().cpu().numpy()
    if hasattr(rotation_matrix, "detach"):
        rotation_matrix = rotation_matrix.detach().cpu().numpy()

    # Convert rotation matrix to Euler angles (XYZ order)
    # Using scipy's rotation conversion
    try:
        from scipy.spatial.transform import Rotation as R

        r = R.from_matrix(rotation_matrix)
        euler_xyz = r.as_euler("xyz", degrees=False)  # radians
    except ImportError:
        # Fallback to manual calculation if scipy not available
        # Extract Euler angles from rotation matrix (XYZ order)
        sy = np.sqrt(
            rotation_matrix[0, 0] * rotation_matrix[0, 0]
            + rotation_matrix[1, 0] * rotation_matrix[1, 0]
        )
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0

        euler_xyz = np.array([x, y, z])

    return {
        "translation": center.tolist(),
        "dimensions": axes.tolist(),
        "xyz_body_rotation": euler_xyz.tolist(),
    }


def convert_fitted_cylinder_to_wrap_params(center, radius_length=None, rotation_matrix=None):
    """
    Convert fitted cylinder parameters to the format expected by create_cylinder_polydata.

    Args:
        center: Either:
            - (3,) tensor or array - cylinder center coordinates (if radius_length and rotation_matrix provided)
            - tuple of (center, radius_length, rotation_matrix) from CylinderFitter.fit() output
        radius_length: (2,) tensor or array - [radius, half_length] of cylinder (optional if center is tuple)
        rotation_matrix: (3,3) tensor or array - rotation matrix (optional if center is tuple)

    Returns:
        Dictionary with keys expected by create_cylinder_polydata:
        - 'translation': center coordinates as list
        - 'radius': cylinder radius
        - 'length': cylinder full length (2 * half_length)
        - 'xyz_body_rotation': euler angles in radians as list
    """
    import torch

    # Handle case where center is actually the full tuple from CylinderFitter.fit()
    if radius_length is None and rotation_matrix is None:
        if isinstance(center, (tuple, list)) and len(center) == 3:
            center, radius_length, rotation_matrix = center
        else:
            raise ValueError(
                "If radius_length and rotation_matrix are None, center must be a tuple of (center, radius_length, rotation_matrix)"
            )

    # Convert to numpy if torch tensors
    if hasattr(center, "detach"):
        center = center.detach().cpu().numpy()
    if hasattr(radius_length, "detach"):
        radius_length = radius_length.detach().cpu().numpy()
    if hasattr(rotation_matrix, "detach"):
        rotation_matrix = rotation_matrix.detach().cpu().numpy()

    # Convert rotation matrix to Euler angles (XYZ order)
    # Using scipy's rotation conversion
    try:
        from scipy.spatial.transform import Rotation as R

        r = R.from_matrix(rotation_matrix)
        euler_xyz = r.as_euler("xyz", degrees=False)  # radians
    except ImportError:
        # Fallback to manual calculation if scipy not available
        # Extract Euler angles from rotation matrix (XYZ order)
        sy = np.sqrt(
            rotation_matrix[0, 0] * rotation_matrix[0, 0]
            + rotation_matrix[1, 0] * rotation_matrix[1, 0]
        )
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0

        euler_xyz = np.array([x, y, z])

    # Extract radius and half_length from fitted parameters
    radius = (
        radius_length[0].item() if hasattr(radius_length[0], "item") else float(radius_length[0])
    )
    half_length = (
        radius_length[1].item() if hasattr(radius_length[1], "item") else float(radius_length[1])
    )

    # Convert half_length to full length for PyVista
    full_length = 2.0 * half_length

    return {
        "translation": center.tolist(),
        "radius": radius,
        "length": full_length,
        "xyz_body_rotation": euler_xyz.tolist(),
    }


def compute_sdf_values(points: np.ndarray, wrap_surfaces: Dict) -> Dict[str, np.ndarray]:
    """
    Compute SDF values for points relative to all wrap surfaces.

    Args:
        points: Array of 3D points (N, 3)
        wrap_surfaces: Dictionary of surface name -> Mesh objects that can compute SDF

    Returns:
        Dictionary mapping surface name to SDF values array
    """
    logger.debug(f"[DEBUG] dtypes to compute SDF values:")
    logger.debug(f"points: {points.dtype}")
    sdf_values = {}
    for surface_name, surface_obj in wrap_surfaces.items():
        logger.debug(f"{surface_name}: {surface_obj.point_coords.dtype}")
        surface_obj.point_coords = surface_obj.point_coords.astype(float)
        sdf_values[surface_name] = surface_obj.get_sdf_pts(points)
    return sdf_values


def classify_points(sdf_values: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Convert SDF values to binary inside/outside labels.

    Args:
        sdf_values: Array of SDF values
        threshold: Threshold for inside/outside classification

    Returns:
        Binary array (1 for inside, 0 for outside)
    """
    return (sdf_values < threshold).astype(int)


def classify_near_surface(sdf_values: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Convert SDF values to binary near-surface/far labels.

    Args:
        sdf_values: Array of SDF values
        threshold: Distance threshold in mm for near-surface classification

    Returns:
        Binary array (1 for near surface, 0 for far from surface)
    """
    return (np.abs(sdf_values) <= threshold).astype(int)


def create_ellipsoid_polydata(wrap_params: Dict = None, **kwargs) -> pv.PolyData:
    """
    Create a PyVista PolyData ellipsoid from OpenSim wrap parameters.

    Args:
        wrap_params: Dictionary containing ellipsoid parameters with keys:
            - dimensions: List of [a,b,c] semi-axes lengths
            - translation: List of [x,y,z] center coordinates
            - xyz_body_rotation: List of [x,y,z] rotation angles in radians
        **kwargs: Alternative way to pass parameters directly as keyword arguments

    Returns:
        PyVista ellipsoid surface mesh
    """
    # Use kwargs if wrap_params not provided
    if wrap_params is None:
        wrap_params = kwargs

    # Create base ellipsoid centered at origin
    a, b, c = wrap_params["dimensions"]
    ellipsoid = pv.ParametricEllipsoid(xradius=a, yradius=b, zradius=c)

    # Create rotation matrix from Euler angles
    x_rot, y_rot, z_rot = wrap_params["xyz_body_rotation"]
    rot_matrix = ellipsoid.rotate_x(np.degrees(x_rot), inplace=False)
    rot_matrix = rot_matrix.rotate_y(np.degrees(y_rot), inplace=False)
    rot_matrix = rot_matrix.rotate_z(np.degrees(z_rot), inplace=False)

    # Translate to final position
    x, y, z = wrap_params["translation"]
    final_ellipsoid = rot_matrix.translate([x, y, z])

    return final_ellipsoid


def create_cylinder_polydata(wrap_params: Dict) -> pv.PolyData:
    """
    Create a PyVista PolyData cylinder from OpenSim wrap parameters.

    Args:
        wrap_params: Dictionary containing cylinder parameters with keys:
            - radius: Cylinder radius
            - length: Cylinder length
            - translation: List of [x,y,z] center coordinates
            - xyz_body_rotation: List of [x,y,z] rotation angles in radians

    Returns:
        PyVista cylinder surface mesh
    """
    # Create base cylinder centered at origin
    radius = wrap_params["radius"]
    length = wrap_params["length"]
    cylinder = pv.Cylinder(radius=radius, height=length, direction=[0, 0, 1])

    # Create rotation matrix from Euler angles
    x_rot, y_rot, z_rot = wrap_params["xyz_body_rotation"]
    rot_matrix = cylinder.rotate_x(np.degrees(x_rot), inplace=False)
    rot_matrix = rot_matrix.rotate_y(np.degrees(y_rot), inplace=False)
    rot_matrix = rot_matrix.rotate_z(np.degrees(z_rot), inplace=False)

    # Translate to final position
    x, y, z = wrap_params["translation"]
    final_cylinder = rot_matrix.translate([x, y, z])

    return final_cylinder


def extract_wrap_parameters(xml_file: str, bone_dict: Dict) -> Dict:
    """
    Extract wrap surface parameters from an OpenSim XML file based on specified bone dictionary.

    Args:
        xml_file: Path to OpenSim XML file
        bone_dict: Dictionary specifying which wrap surfaces to extract for each bone/body
            Format:
            {
                'bone_name': {
                    'wrap_surfaces': {
                        'body_name': {
                            'ellipsoid': ['surface1', 'surface2'],
                            'cylinder': ['surface3']
                        }
                    }
                }
            }

    Returns:
        Dictionary containing wrap surface parameters organized by bone, body and wrap object name
    """
    wrap_params = {}

    # Parse XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Iterate through bones in dictionary
    for bone_name, bone_info in bone_dict.items():
        wrap_params[bone_name] = {}

        if "wrap_surfaces" not in bone_info:
            continue

        # Iterate through bodies for this bone
        for body_name, surface_types in bone_info["wrap_surfaces"].items():
            if body_name in ADDITIONAL_OFFSETS:
                offset = ADDITIONAL_OFFSETS[body_name]
            else:
                offset = [0, 0, 0]
            wrap_params[bone_name][body_name] = {}

            # Find body in XML
            body = root.find(f".//Body[@name='{body_name}']")
            if body is None:
                continue

            for surface_type, surface_names in surface_types.items():
                # Process ellipsoids
                if surface_type == "ellipsoid":
                    for surface_name in surface_names:
                        ellipsoid = body.find(f".//WrapEllipsoid[@name='{surface_name}']")
                        if ellipsoid is not None:
                            xyz_rot = [
                                float(x) for x in ellipsoid.find("xyz_body_rotation").text.split()
                            ]
                            trans = [
                                float(x) - offset[idx]
                                for idx, x in enumerate(ellipsoid.find("translation").text.split())
                            ]
                            dims = [float(x) for x in ellipsoid.find("dimensions").text.split()]
                            quadrant = ellipsoid.find("quadrant").text

                            wrap_params[bone_name][body_name][surface_name] = {
                                "type": "WrapEllipsoid",
                                "xyz_body_rotation": xyz_rot,
                                "translation": trans,
                                "dimensions": dims,
                                "quadrant": quadrant,
                            }

                # Process cylinders
                if surface_type == "cylinder":
                    for surface_name in surface_names:
                        cylinder = body.find(f".//WrapCylinder[@name='{surface_name}']")
                        if cylinder is not None:
                            xyz_rot = [
                                float(x) for x in cylinder.find("xyz_body_rotation").text.split()
                            ]
                            trans = [
                                float(x) - offset[idx]
                                for idx, x in enumerate(cylinder.find("translation").text.split())
                            ]
                            radius = float(cylinder.find("radius").text)
                            length = float(cylinder.find("length").text)
                            quadrant = cylinder.find("quadrant").text

                            wrap_params[bone_name][body_name][surface_name] = {
                                "type": "WrapCylinder",
                                "xyz_body_rotation": xyz_rot,
                                "translation": trans,
                                "radius": radius,
                                "length": length,
                                "quadrant": quadrant,
                            }

    return wrap_params


def prepare_fitting_data(
    bone_mesh_path: str,
    xml_path: str,
    bone_name: str,
    bone_dict: Dict,
    near_surface_threshold: float = 0.005,
) -> Mesh:
    """
    Complete data preparation pipeline for a specific bone.

    Args:
        bone_mesh_path: Path to the bone mesh file
        xml_path: Path to the XML file containing wrap surface definitions
        bone_name: Name of the bone in the bone_dict
        bone_dict: Dictionary mapping bone names to bone type keywords
        near_surface_threshold: Distance threshold for near-surface classification (auto-detected units)

    Returns:
        Mesh object with all computed labels
    """
    # Load mesh
    mesh = Mesh(bone_mesh_path)

    # Auto-detect coordinate scale (mesh in meters vs mm)
    coord_range = np.ptp(mesh.point_coords, axis=0).max()
    if coord_range < 1.0:  # Likely in meters
        logger.info(
            f"Detected mesh coordinates in meters (range: {coord_range:.3f}m). Using threshold: {near_surface_threshold}m"
        )
    else:  # Likely in mm
        logger.info(
            f"Detected mesh coordinates in mm (range: {coord_range:.1f}mm). Using threshold: {near_surface_threshold}mm"
        )

    # Use the threshold as-is (it should match the coordinate units)
    effective_threshold = near_surface_threshold

    # Extract parameters only for the specific bone
    bone_specific_dict = {bone_name: bone_dict[bone_name]}
    wrap_params = extract_wrap_parameters(xml_path, bone_specific_dict)

    wrap_surfaces = {}
    if bone_name in wrap_params:
        for body_name, body_data in wrap_params[bone_name].items():
            for surface_name, surface_params in body_data.items():
                # Create surface name for easy identification
                full_surface_name = f"{bone_name}_{body_name}_{surface_name}"

                if surface_params["type"] == "WrapEllipsoid":
                    surface = create_ellipsoid_polydata(surface_params)
                elif surface_params["type"] == "WrapCylinder":
                    surface = create_cylinder_polydata(surface_params)
                else:
                    continue

                # Convert to Mesh object that can compute SDF
                surface_mesh = Mesh(surface.triangulate())
                wrap_surfaces[surface_name] = surface_mesh  # Use simplified name as key

    # Compute SDF values only for this bone's wrap surfaces
    points = np.asarray(mesh.point_coords.copy()).astype(float)

    sdf_values = compute_sdf_values(points, wrap_surfaces)

    # Add both SDF values and binarized classifications as vertex data
    for surface_name, sdf_vals in sdf_values.items():
        # Add SDF values
        mesh.mesh.point_data[f"{surface_name}_sdf"] = sdf_vals

        # Add binarized classifications
        classifications = classify_points(sdf_vals)
        mesh.mesh.point_data[f"{surface_name}_binary"] = classifications

        # Add near-surface classifications
        near_surface_classifications = classify_near_surface(sdf_vals, effective_threshold)
        mesh.mesh.point_data[f"{surface_name}_near_surface"] = near_surface_classifications

    return mesh


def prepare_multi_bone_fitting_data(
    geometry_folder: str, xml_path: str, bone_dict: Dict, near_surface_threshold=0.0005
) -> Dict:
    """
    Complete data preparation pipeline for multiple bones.

    Args:
        geometry_folder: Path to folder containing bone mesh files
        xml_path: Path to OpenSim XML file
        bone_dict: Dictionary specifying wrap surfaces to extract for each bone
        near_surface_threshold: Float for global threshold or dict mapping bone names to thresholds

    Returns:
        Dictionary mapping bone names to Mesh objects with SDF and binary values as vertex data
    """
    import os

    all_results = {}
    for bone_name, bone_data in bone_dict.items():
        if "surface_filename" not in bone_data:
            continue

        if isinstance(geometry_folder, dict):
            bone_mesh_path = os.path.join(geometry_folder[bone_name], bone_data["surface_filename"])
        else:
            bone_mesh_path = os.path.join(geometry_folder, bone_data["surface_filename"])

        # Get threshold for this specific bone
        if isinstance(near_surface_threshold, dict):
            bone_threshold = near_surface_threshold.get(bone_name, 0.0005)
        else:
            bone_threshold = near_surface_threshold

        mesh_with_data = prepare_fitting_data(
            bone_mesh_path, xml_path, bone_name, bone_dict, bone_threshold
        )
        all_results[bone_name] = mesh_with_data

    return all_results


def create_ellipsoid_from_fitted_params(fitted_parameters) -> pv.PolyData:
    """
    Create a PyVista PolyData ellipsoid directly from EllipsoidFitter output.

    Args:
        fitted_parameters: Tuple of (center, axes, rotation_matrix) from EllipsoidFitter.fit()

    Returns:
        PyVista ellipsoid surface mesh
    """
    center, axes, rotation_matrix = fitted_parameters
    wrap_params = convert_fitted_ellipsoid_to_wrap_params(fitted_parameters)
    return create_ellipsoid_polydata(wrap_params)


def create_cylinder_from_fitted_params(fitted_parameters) -> pv.PolyData:
    """
    Create a PyVista PolyData cylinder directly from CylinderFitter output.

    Args:
        fitted_parameters: Tuple of (center, radius_length, rotation_matrix) from CylinderFitter.fit()

    Returns:
        PyVista cylinder surface mesh
    """
    center, radius_length, rotation_matrix = fitted_parameters
    wrap_params = convert_fitted_cylinder_to_wrap_params(fitted_parameters)
    return create_cylinder_polydata(wrap_params)


def euler_xyz_to_rotation_matrix(x_rot, y_rot, z_rot):
    """
    Convert XYZ Euler angles to rotation matrix using the same order as OpenSim/PyVista.

    Args:
        x_rot, y_rot, z_rot: Rotation angles in radians

    Returns:
        3x3 rotation matrix as numpy array
    """
    # Using the same order as create_cylinder_polydata: X, then Y, then Z
    cx, sx = np.cos(x_rot), np.sin(x_rot)
    cy, sy = np.cos(y_rot), np.sin(y_rot)
    cz, sz = np.cos(z_rot), np.sin(z_rot)

    # Rotation matrices for each axis
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])

    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    # Combined rotation: R = Rz @ Ry @ Rx (applied in order X, Y, Z)
    return Rz @ Ry @ Rx
