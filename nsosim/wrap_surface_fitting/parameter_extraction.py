"""
Clean API for OpenSim wrap surface parameter extraction.

This module provides simple, focused functions for extracting wrap surface
parameters from OpenSim XML files and creating PyVista meshes from those parameters.

This is a thin wrapper around existing functionality in utils.py, providing
a cleaner API as part of the wrap surface fitting refactor.
"""

from typing import Dict, Optional, Union

import pyvista as pv

from .config import DEFAULT_SMITH2019_BONES
from .utils import create_cylinder_polydata, create_ellipsoid_polydata, extract_wrap_parameters


def extract_wrap_parameters_from_osim(
    osim_file_path: str, bone_config: Optional[Dict] = None
) -> Dict:
    """
    Extract wrap surface parameters from an OpenSim XML file.

    This is a clean API wrapper around utils.extract_wrap_parameters that uses
    the default Smith2019 bone configuration if none is provided.

    Args:
        osim_file_path: Path to the OpenSim XML file (.osim)
        bone_config: Optional bone configuration dictionary. If None, uses DEFAULT_SMITH2019_BONES

    Returns:
        Dictionary containing wrap surface parameters organized by:
        {bone_name: {body_name: {wrap_name: {parameters}}}}

    Example:
        # Extract parameters using default Smith2019 configuration
        params = extract_wrap_parameters_from_osim('/path/to/smith2019.osim')

        # Extract parameters with custom bone configuration
        custom_config = {'femur': {'wrap_surfaces': {...}}}
        params = extract_wrap_parameters_from_osim('/path/to/model.osim', custom_config)
    """
    if bone_config is None:
        bone_config = DEFAULT_SMITH2019_BONES

    return extract_wrap_parameters(osim_file_path, bone_config)


def create_meshes_from_wrap_parameters(wrap_params: Dict) -> Dict:
    """
    Create PyVista meshes from extracted wrap surface parameters.

    Takes the output from extract_wrap_parameters_from_osim() and creates
    3D mesh representations of all the wrap surfaces.

    Args:
        wrap_params: Dictionary of wrap parameters from extract_wrap_parameters_from_osim()

    Returns:
        Dictionary of PyVista meshes organized by:
        {bone_name: {body_name: {wrap_name: pv.PolyData}}}

    Example:
        # Extract parameters and create meshes
        params = extract_wrap_parameters_from_osim('/path/to/smith2019.osim')
        meshes = create_meshes_from_wrap_parameters(params)

        # Access a specific mesh
        femur_gastroc_mesh = meshes['femur']['femur_r']['Gastroc_at_Condyles_r']
    """
    wrap_meshes = {}

    for bone_name, bone_data in wrap_params.items():
        wrap_meshes[bone_name] = {}

        for body_name, body_data in bone_data.items():
            wrap_meshes[bone_name][body_name] = {}

            for wrap_name, wrap_params_single in body_data.items():
                # Determine surface type from the 'type' field
                surface_type = wrap_params_single.get("type", "")

                if surface_type == "WrapEllipsoid":
                    mesh = create_ellipsoid_polydata(wrap_params_single)
                elif surface_type == "WrapCylinder":
                    mesh = create_cylinder_polydata(wrap_params_single)
                else:
                    # Skip unknown surface types or log a warning
                    print(
                        f"Warning: Unknown wrap surface type '{surface_type}' for {wrap_name}. Skipping."
                    )
                    continue

                wrap_meshes[bone_name][body_name][wrap_name] = mesh

    return wrap_meshes


def create_single_wrap_surface_mesh(
    wrap_params: Dict, surface_type: Optional[str] = None
) -> pv.PolyData:
    """
    Create a single PyVista mesh from wrap surface parameters.

    Convenience function for creating a single wrap surface mesh when you
    have the parameters for just one surface.

    Args:
        wrap_params: Dictionary containing parameters for a single wrap surface
        surface_type: Optional surface type ('WrapEllipsoid' or 'WrapCylinder').
                     If None, will be auto-detected from the 'type' field

    Returns:
        PyVista PolyData mesh of the wrap surface

    Example:
        # Create ellipsoid mesh
        ellipsoid_params = {
            'type': 'WrapEllipsoid',
            'dimensions': [0.027, 0.023, 0.138],
            'translation': [0.009, -0.377, 0.0003],
            'xyz_body_rotation': [0.059, -0.087, 0.023]
        }
        mesh = create_single_wrap_surface_mesh(ellipsoid_params)

        # Override surface type if needed
        mesh = create_single_wrap_surface_mesh(ellipsoid_params, 'WrapEllipsoid')
    """
    # Use provided surface_type or auto-detect from 'type' field
    if surface_type is None:
        surface_type = wrap_params.get("type", "")
        if not surface_type:
            raise ValueError(
                "Cannot determine surface type. Parameters must contain a 'type' field "
                "or surface_type must be provided explicitly."
            )

    # Create mesh based on surface type
    if surface_type == "WrapEllipsoid":
        return create_ellipsoid_polydata(wrap_params)
    elif surface_type == "WrapCylinder":
        return create_cylinder_polydata(wrap_params)
    else:
        raise ValueError(
            f"Unsupported surface type: '{surface_type}'. "
            "Supported types are 'WrapEllipsoid' and 'WrapCylinder'"
        )


def extract_and_create_wrap_surfaces(
    osim_file_path: str, bone_config: Optional[Dict] = None
) -> Dict:
    """
    Complete workflow: extract parameters from OpenSim file and create meshes.

    This is a convenience function that combines parameter extraction and mesh
    creation in a single call.

    Args:
        osim_file_path: Path to the OpenSim XML file (.osim)
        bone_config: Optional bone configuration dictionary. If None, uses DEFAULT_SMITH2019_BONES

    Returns:
        Dictionary of PyVista meshes organized by:
        {bone_name: {body_name: {wrap_name: pv.PolyData}}}

    Example:
        # Complete workflow in one call
        wrap_meshes = extract_and_create_wrap_surfaces('/path/to/smith2019.osim')

        # Visualize a specific wrap surface
        femur_gastroc = wrap_meshes['femur']['femur_r']['Gastroc_at_Condyles_r']
        femur_gastroc.plot()
    """
    # Step 1: Extract parameters
    wrap_params = extract_wrap_parameters_from_osim(osim_file_path, bone_config)

    # Step 2: Create meshes
    wrap_meshes = create_meshes_from_wrap_parameters(wrap_params)

    return wrap_meshes
