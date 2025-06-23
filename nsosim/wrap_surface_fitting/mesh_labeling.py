"""
Clean API for mesh labeling with wrap surface data.

This module provides simple, focused functions for labeling mesh vertices
with wrap surface information (SDF values, binary classifications, near-surface labels).

This is a thin wrapper around existing functionality in utils.py, providing
a cleaner API as part of the wrap surface fitting refactor.
"""

from typing import Dict, Optional, Union
from pathlib import Path
from pymskt.mesh import Mesh

from .utils import prepare_fitting_data, prepare_multi_bone_fitting_data
from .config import DEFAULT_SMITH2019_BONES, DEFAULT_SMITH2019_THRESHOLDS


def label_mesh_vertices_for_wrap_surfaces(mesh_path: str, 
                                        osim_path: str, 
                                        bone_name: str,
                                        bone_config: Optional[Dict] = None,
                                        near_surface_threshold: Optional[float] = None) -> Mesh:
    """
    Label mesh vertices with wrap surface data for a single bone.
    
    This function loads a bone mesh and labels each vertex with:
    - SDF values relative to each wrap surface
    - Binary inside/outside classifications
    - Near-surface classifications
    
    Args:
        mesh_path: Path to the bone mesh file (.stl, .vtk, etc.)
        osim_path: Path to the OpenSim XML file (.osim)
        bone_name: Name of the bone (must exist in bone_config)
        bone_config: Optional bone configuration dictionary. If None, uses DEFAULT_SMITH2019_BONES
        near_surface_threshold: Optional threshold for near-surface classification. 
                              If None, uses value from DEFAULT_SMITH2019_THRESHOLDS
        
    Returns:
        Mesh object with labeled vertices. Each wrap surface adds three vertex data arrays:
        - "{surface_name}_sdf": Signed distance field values
        - "{surface_name}_binary": Binary inside/outside labels (0=outside, 1=inside)
        - "{surface_name}_near_surface": Near-surface labels (0=far, 1=near)
        
    Example:
        # Label femur mesh with default Smith2019 configuration
        labeled_mesh = label_mesh_vertices_for_wrap_surfaces(
            mesh_path='/path/to/femur.stl',
            osim_path='/path/to/smith2019.osim',
            bone_name='femur'
        )
        
        # Access SDF values for a specific wrap surface
        gastroc_sdf = labeled_mesh.mesh.point_data['Gastroc_at_Condyles_r_sdf']
        gastroc_binary = labeled_mesh.mesh.point_data['Gastroc_at_Condyles_r_binary']
    """
    # Use default configurations if not provided
    if bone_config is None:
        bone_config = DEFAULT_SMITH2019_BONES
    
    if near_surface_threshold is None:
        near_surface_threshold = DEFAULT_SMITH2019_THRESHOLDS.get(bone_name, 0.0005)
    
    # Validate bone_name exists in configuration
    if bone_name not in bone_config:
        available_bones = list(bone_config.keys())
        raise ValueError(f"Bone '{bone_name}' not found in configuration. "
                        f"Available bones: {available_bones}")
    
    # Call the existing utils function
    return prepare_fitting_data(
        bone_mesh_path=mesh_path,
        xml_path=osim_path,
        bone_name=bone_name,
        bone_dict=bone_config,
        near_surface_threshold=near_surface_threshold
    )


def label_multiple_meshes(geometry_folder: str,
                         osim_path: str,
                         bone_config: Optional[Dict] = None,
                         near_surface_thresholds: Optional[Union[float, Dict[str, float]]] = None) -> Dict[str, Mesh]:
    """
    Label multiple bone meshes with wrap surface data.
    
    This function processes all bones specified in the bone configuration,
    loading their meshes from a geometry folder and labeling vertices with
    wrap surface data.
    
    Args:
        geometry_folder: Path to folder containing bone mesh files
        osim_path: Path to the OpenSim XML file (.osim)
        bone_config: Optional bone configuration dictionary. If None, uses DEFAULT_SMITH2019_BONES
        near_surface_thresholds: Optional threshold(s) for near-surface classification.
                               Can be a single float (applied to all bones) or a dictionary
                               mapping bone names to thresholds. If None, uses DEFAULT_SMITH2019_THRESHOLDS
        
    Returns:
        Dictionary mapping bone names to labeled Mesh objects:
        {bone_name: Mesh_with_labeled_vertices}
        
    Example:
        # Label all bones with default Smith2019 configuration
        labeled_meshes = label_multiple_meshes(
            geometry_folder='/path/to/Geometry/',
            osim_path='/path/to/smith2019.osim'
        )
        
        # Access labeled femur mesh
        femur_mesh = labeled_meshes['femur']
        
        # Label with custom thresholds
        custom_thresholds = {'femur': 0.001, 'tibia': 0.0005, 'patella': 0.002}
        labeled_meshes = label_multiple_meshes(
            geometry_folder='/path/to/Geometry/',
            osim_path='/path/to/smith2019.osim',
            near_surface_thresholds=custom_thresholds
        )
    """
    # Use default configurations if not provided
    if bone_config is None:
        bone_config = DEFAULT_SMITH2019_BONES
    
    if near_surface_thresholds is None:
        near_surface_thresholds = DEFAULT_SMITH2019_THRESHOLDS
    
    # Call the existing utils function
    return prepare_multi_bone_fitting_data(
        geometry_folder=geometry_folder,
        xml_path=osim_path,
        bone_dict=bone_config,
        near_surface_threshold=near_surface_thresholds
    )


def get_labeled_vertex_data(mesh: Mesh, surface_name: str) -> Dict[str, any]:
    """
    Extract labeled vertex data for a specific wrap surface from a mesh.
    
    Convenience function to get all the labeled data (SDF, binary, near-surface)
    for a specific wrap surface from a mesh that has been processed by the
    labeling functions.
    
    Args:
        mesh: Labeled Mesh object (output from label_mesh_vertices_for_wrap_surfaces)
        surface_name: Name of the wrap surface (e.g., 'Gastroc_at_Condyles_r')
        
    Returns:
        Dictionary containing:
        - 'sdf': SDF values array
        - 'binary': Binary classification array
        - 'near_surface': Near-surface classification array
        - 'points': Mesh vertex coordinates
        
    Example:
        # Get all data for a specific wrap surface
        gastroc_data = get_labeled_vertex_data(labeled_mesh, 'Gastroc_at_Condyles_r')
        
        # Access specific arrays
        sdf_values = gastroc_data['sdf']
        inside_points = gastroc_data['points'][gastroc_data['binary'] == 1]
        near_surface_points = gastroc_data['points'][gastroc_data['near_surface'] == 1]
    """
    # Check if the required data exists
    sdf_key = f"{surface_name}_sdf"
    binary_key = f"{surface_name}_binary"
    near_surface_key = f"{surface_name}_near_surface"
    
    available_keys = list(mesh.mesh.point_data.keys())
    
    if sdf_key not in available_keys:
        raise ValueError(f"SDF data '{sdf_key}' not found in mesh. "
                        f"Available keys: {available_keys}")
    
    if binary_key not in available_keys:
        raise ValueError(f"Binary data '{binary_key}' not found in mesh. "
                        f"Available keys: {available_keys}")
    
    if near_surface_key not in available_keys:
        raise ValueError(f"Near-surface data '{near_surface_key}' not found in mesh. "
                        f"Available keys: {available_keys}")
    
    return {
        'sdf': mesh.mesh.point_data[sdf_key],
        'binary': mesh.mesh.point_data[binary_key],
        'near_surface': mesh.mesh.point_data[near_surface_key],
        'points': mesh.point_coords.copy()
    }