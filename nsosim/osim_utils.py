import opensim as osim
import numpy as np
from typing import Union
from pymskt.mesh import Mesh
import logging
import json

ROUND_DIGITS = 6

# Module-level logger. Configure in nsosim.configure_logging()
logger = logging.getLogger(__name__)

def update_ligament_stiffness(model, ligament, linear_stiffness):
    """
    Updates the linear stiffness of a specific ligament in an OpenSim model.

    Args:
        model: osim.Model
        ligament (str): The name of the `Blankevoort1991Ligament` to update.
        linear_stiffness (float or int): The new linear stiffness value.
    """
    forceset = model.getForceSet()
    force = osim.Blankevoort1991Ligament.safeDownCast(forceset.get(ligament))
    force.set_linear_stiffness(round(linear_stiffness, ROUND_DIGITS))

def update_body_geometry_meshfile(model, dict_body_geometries_update):
    """
    Updates the mesh file paths for body visualization geometry in an OpenSim model.
    
    Args:
        model: osim.Model
        dict_body_geometries_update: dict
    
    Notes:
    dict_body_geometries_update format:
    {
        'body_name': {
            'geometry_name': 'mesh_file_path'
        }
    }
    """
    for body_name, body_dict in dict_body_geometries_update.items():
        body = osim.Body.safeDownCast(model.getBodySet().get(body_name))
        n_geo_osim = body.getPropertyByName('attached_geometry').size()
        for geo_osim_idx in range(n_geo_osim):
            attached_geometry = body.get_attached_geometry(geo_osim_idx)
            mesh = osim.Mesh.safeDownCast(attached_geometry)
            name = mesh.getName()
            if name in body_dict.keys():
                mesh.set_mesh_file(body_dict[name])
            else:
                logger.warning(f'Geometry "{name}" not found in body_dict for body "{body_name}" - skipping')

def update_contact_mesh_files(model, dict_contact_mesh_files_update):
    """
    Updates the mesh file paths for contact geometry in an OpenSim model.
    
    Args:
        model: osim.Model
        dict_contact_mesh_files_update: dict
    
    Notes:
    dict_contact_mesh_files_update format:
    {
        'contact_name': {
            'mesh_file': 'mesh_file_path',
            'mesh_back_file': 'mesh_back_file_path'
        }
    }
    """
    contact_geometries = model.getContactGeometrySet()
    
    # Define fallback naming patterns for common mismatches
    FALLBACK_NAMES = {
        'meniscus_medial_superior': ['meniscus_med_sup'],
        'meniscus_medial_inferior': ['meniscus_med_inf'],
        'meniscus_lateral_superior': ['meniscus_lat_sup'],
        'meniscus_lateral_inferior': ['meniscus_lat_inf'],
        # Add reverse mappings in case the dictionary uses old names
        'meniscus_med_sup': ['meniscus_medial_superior'],
        'meniscus_med_inf': ['meniscus_medial_inferior'],
        'meniscus_lat_sup': ['meniscus_lateral_superior'],
        'meniscus_lat_inf': ['meniscus_lateral_inferior'],
    }
    
    updated_count = 0
    failed_updates = []
    
    for contact_name, contact_dict in dict_contact_mesh_files_update.items():
        contact_geometry = None
        actual_name_used = None
        
        # Try the original name first
        try:
            contact_geometry = osim.Smith2018ContactMesh.safeDownCast(contact_geometries.get(contact_name))
            if contact_geometry is not None:
                actual_name_used = contact_name
        except RuntimeError:
            pass
        
        # If original name failed, try fallback names
        if contact_geometry is None and contact_name in FALLBACK_NAMES:
            for fallback_name in FALLBACK_NAMES[contact_name]:
                try:
                    contact_geometry = osim.Smith2018ContactMesh.safeDownCast(contact_geometries.get(fallback_name))
                    if contact_geometry is not None:
                        actual_name_used = fallback_name
                        logger.info(f'Contact geometry "{contact_name}" not found, using fallback name "{fallback_name}"')
                        break
                except RuntimeError:
                    continue
        
        # Update the contact geometry if found
        if contact_geometry is not None:
            try:
                if 'mesh_file' in contact_dict.keys():
                    contact_geometry.set_mesh_file(contact_dict['mesh_file'])
                if 'mesh_back_file' in contact_dict.keys():
                    contact_geometry.set_mesh_back_file(contact_dict['mesh_back_file'])
                updated_count += 1
                logger.debug(f'Successfully updated contact geometry "{actual_name_used}"')
            except Exception as e:
                failed_updates.append(f'{contact_name}: {str(e)}')
                logger.error(f'Failed to update contact geometry "{actual_name_used}": {e}')
        else:
            failed_updates.append(f'{contact_name}: not found in model')
            logger.error(f'Contact geometry "{contact_name}" not found in model (tried fallbacks: {FALLBACK_NAMES.get(contact_name, [])})')
    
    # Log summary and raise error if ANY updates failed
    logger.info(f'Contact geometry update summary: {updated_count}/{len(dict_contact_mesh_files_update)} successful')
    
    if failed_updates:
        error_msg = f"Failed to update {len(failed_updates)} contact geometries:\n" + "\n".join([f"  - {failure}" for failure in failed_updates])
        logger.error("Contact geometry updates failed - this will affect simulation accuracy!")
        logger.error(error_msg)
        raise RuntimeError(f"Contact geometry updates failed. {error_msg}")

def update_joint_default_values(model, dict_joint_default_values_update, incremental=False):
    """
    Updates the default values for joints in an OpenSim model.
    
    Args:
        model: osim.Model
        dict_joint_default_values_update: dict
        incremental: bool, optional
            If True, adds the values to existing default values.
            If False (default), replaces the existing default values.
    
    Notes:
    dict_joint_default_values_update format:
    {
        'joint_name': {
            'coordinate_idx': 'coordinate_value'
        }
    }
    """
    jointset = model.getJointSet()
    for joint_name, joint_dict in dict_joint_default_values_update.items():
        joint = jointset.get(joint_name)
        for coordinate_idx, coordinate_value in joint_dict.items():
            coordinate = joint.get_coordinates(int(coordinate_idx))
            if incremental:
                # Add to existing default value
                current_value = coordinate.get_default_value()
                new_value = current_value + coordinate_value
                coordinate.set_default_value(round(new_value, ROUND_DIGITS))
                logger.debug(f'Updated joint {joint_name} coordinate {coordinate_idx}: {current_value} + {coordinate_value} = {new_value}')
            else:
                # Replace existing default value
                coordinate.set_default_value(round(coordinate_value, ROUND_DIGITS))
                logger.debug(f'Set joint {joint_name} coordinate {coordinate_idx} to {coordinate_value}')

def update_wrap_cylinder(
    model, 
    body_name, 
    wrap_name, 
    translation=None,
    xyz_body_rotation=None,
    radius=None,
    length=None
):
    
    # assert that translation, xyz_body_rotation, radius, and length are all float
    if translation is not None:
        assert isinstance(translation, (list, np.ndarray)), f'translation must be a list or numpy array, got {type(translation)}'
        translation = np.asarray(translation, dtype=float).round(ROUND_DIGITS)
    if xyz_body_rotation is not None:
        assert isinstance(xyz_body_rotation, (list, np.ndarray)), f'xyz_body_rotation must be a list or numpy array, got {type(xyz_body_rotation)}'
        xyz_body_rotation = np.asarray(xyz_body_rotation, dtype=float).round(ROUND_DIGITS)
    if radius is not None:
        radius = round(float(radius), ROUND_DIGITS)
    if length is not None:
        length = round(float(length), ROUND_DIGITS)
    
    body = model.getBodySet().get(body_name)
    wrap_object = body.getWrapObject(wrap_name)
    wrap_cylinder = osim.WrapCylinder.safeDownCast(wrap_object)
    if translation is not None:
        assert isinstance(translation, (list, np.ndarray)), f'translation must be a list or numpy array, got {type(translation)}'
        wrap_cylinder.set_translation(osim.Vec3(translation))
    if xyz_body_rotation is not None:
        assert isinstance(xyz_body_rotation, (list, np.ndarray)), f'xyz_body_rotation must be a list or numpy array, got {type(xyz_body_rotation)}'
        wrap_cylinder.set_xyz_body_rotation(osim.Vec3(xyz_body_rotation))
    if radius is not None:
        wrap_cylinder.set_radius(radius)
    if length is not None:
        wrap_cylinder.set_length(length)

def update_wrap_ellipsoid(
    model, 
    body_name, 
    wrap_name, 
    translation=None,
    xyz_body_rotation=None,
    dimensions=None
):
    # assert that translation, xyz_body_rotation, and dimensions are all float
    if translation is not None:
        assert isinstance(translation, (list, np.ndarray)), f'translation must be a list or numpy array, got {type(translation)}'
        translation = np.asarray(translation, dtype=float).round(ROUND_DIGITS)
    if xyz_body_rotation is not None:
        assert isinstance(xyz_body_rotation, (list, np.ndarray)), f'xyz_body_rotation must be a list or numpy array, got {type(xyz_body_rotation)}'
        xyz_body_rotation = np.asarray(xyz_body_rotation, dtype=float).round(ROUND_DIGITS)
    if dimensions is not None:
        assert isinstance(dimensions, (list, np.ndarray)), f'dimensions must be a list or numpy array, got {type(dimensions)}'
        dimensions = np.asarray(dimensions, dtype=float).round(ROUND_DIGITS)
    
    # if any value in dimensions is zero, then set it to 1e-7, and print a warning
    if np.any(dimensions == 0):
        logger.warning('One or more dimensions are zero, setting to 1e-7')
        dimensions[dimensions == 0] = 1e-7
    
    body = model.getBodySet().get(body_name)
    wrap_object = body.getWrapObject(wrap_name)
    wrap_ellipsoid = osim.WrapEllipsoid.safeDownCast(wrap_object)
    if translation is not None:
        assert isinstance(translation, (list, np.ndarray)), f'translation must be a list or numpy array, got {type(translation)}'
        wrap_ellipsoid.set_translation(osim.Vec3(translation))
    if xyz_body_rotation is not None:
        assert isinstance(xyz_body_rotation, (list, np.ndarray)), f'xyz_body_rotation must be a list or numpy array, got {type(xyz_body_rotation)}'
        wrap_ellipsoid.set_xyz_body_rotation(osim.Vec3(xyz_body_rotation))
    if dimensions is not None:
        assert isinstance(dimensions, (list, np.ndarray)), f'dimensions must be a list or numpy array, got {type(dimensions)}'
        wrap_ellipsoid.set_dimensions(osim.Vec3(dimensions))

def express_point_in_frame(
        xyz_in_source: Union[list[float], np.ndarray],        # (x, y, z) expressed in `source_frame`
        state: osim.State,
        source_frame_name: str,
        target_frame_name: str,
        model: osim.Model
) -> list[float]:
    """
    Return the same physical point, but expressed in `target_frame`.

    Parameters
    ----------
    xyz_in_source       : tuple[float, float, float]
        Coordinates of the station in the source frame.
    state               : osim.State
        Model state with up-to-date kinematics.
    source_frame_name   : str
        Path/name of the frame the coordinates are *currently* in.
    target_frame_name   : str
        Path/name of the frame you want the coordinates expressed in.
    model               : osim.Model
        The OpenSim model holding both frames.
    """
    # make sure xyz_in_source is a numpy array of type float
    if isinstance(xyz_in_source, list):
        xyz_in_source = np.asarray(xyz_in_source, dtype=float).round(ROUND_DIGITS)
    elif isinstance(xyz_in_source, np.ndarray):
        xyz_in_source = xyz_in_source.astype(float).round(ROUND_DIGITS)
    else:
        raise ValueError(f'xyz_in_source must be a list or numpy array, got {type(xyz_in_source)}')
    
    # Resolve frames (bodyset/… is common, but you can use absolute paths too)
    source_frame = model.getComponent(f'/bodyset/{source_frame_name}').findBaseFrame()
    target_frame = model.getComponent(f'/bodyset/{target_frame_name}').findBaseFrame()

    station_source = osim.Vec3(xyz_in_source)

    station_target = source_frame.findStationLocationInAnotherFrame(
        state,
        station_source,
        target_frame
    )
    return [station_target[i] for i in range(3)]

def get_osim_muscle_ligament_reference_lengths(model, state=None):
    """
    
    Iterate through the force set and get the length, reference strain, and slack length
    for each muscle and ligament.
    
    Args:
        model: osim.Model
        state: osim.State
    
    Returns:
        dict:
            {
                'muscle_name': {
                    'class': 'Millard2012EquilibriumMuscle',
                    'length': float,
                    'reference_strain': None,
                    'slack_length': None
                },
                'ligament_name': {
                    'class': 'Blankevoort1991Ligament',
                    'length': float,
                    'reference_strain': float,
                    'slack_length': float
                }
            }
    """
    
    if state is None:
        state = model.initSystem()
        
    forcesets = model.getForceSet()
    
    dict_force_lengths = {}

    for force_idx in range(forcesets.getSize()):
        force_ = forcesets.get(force_idx)
        force_name = force_.getName()
        
        class_name = force_.getConcreteClassName()
        if class_name in ['Millard2012EquilibriumMuscle', 'Blankevoort1991Ligament']:
            dict_force_lengths[force_name] = {
                'class': force_.getConcreteClassName(),
                'length': None,
                'reference_strain': None,
                'slack_length': None
            }
            if force_.getConcreteClassName() == 'Millard2012EquilibriumMuscle':
                
                # get the muscle
                muscle = osim.Millard2012EquilibriumMuscle.safeDownCast(force_)
                # calculate the scale factor
                muscle_length = muscle.getLength(state)
                
                dict_force_lengths[force_name]['length'] = muscle_length
            
            elif force_.getConcreteClassName() == 'Blankevoort1991Ligament':
                # get the ligament
                ligament = osim.Blankevoort1991Ligament.safeDownCast(force_)
                # calculate the scale factor
                ligament_length = ligament.getLength(state)
                # get slack length
                slack_length = ligament.get_slack_length()
                # reference strain
                reference_strain = (ligament_length - slack_length) / slack_length
                # stiffness
                stiffness = ligament.get_linear_stiffness()
                
                dict_force_lengths[force_name]['length'] = round(ligament_length, ROUND_DIGITS)
                dict_force_lengths[force_name]['slack_length'] = round(slack_length, ROUND_DIGITS)
                dict_force_lengths[force_name]['reference_strain'] = round(reference_strain, ROUND_DIGITS)
                dict_force_lengths[force_name]['stiffness'] = round(stiffness, ROUND_DIGITS)
    
    return dict_force_lengths

def update_point_xyz(
    model: osim.Model,
    opensim_point: osim.PathPoint,
    new_xyz: Union[list[float], np.ndarray],
    state: osim.State,
    child_frame: Union[str, None] = None,
    parent_frame: Union[str, None] = None,
):    
    # if its femur_r or tibia_r then need to account
    # for offset - don't need to know this, it was 
    # handled in another script... if child_frame is defined
    # then need to account for it.
    
    if child_frame is not None:
        new_xyz = express_point_in_frame(
            xyz_in_source=new_xyz,
            state=state,
            source_frame_name=child_frame,
            target_frame_name=parent_frame,
            model=model
        )
    
    if isinstance(new_xyz, (list, np.ndarray)):
        new_xyz = np.asarray(new_xyz, dtype=float).round(ROUND_DIGITS)
    else:
        raise ValueError(f'new_xyz must be a list or numpy array, got {type(new_xyz)}')
        
    # update opensim path point position
    opensim_point = osim.PathPoint.safeDownCast(opensim_point)
    opensim_point.setLocation(osim.Vec3(new_xyz))

def update_single_osim_ligament_muscle_attachment(
    model: osim.Model,
    force: osim.Force,
    state: osim.State,
    force_dict: dict,
    shift_size_vector: Union[list[float], np.ndarray],
    shift_key: str = 'new_shift_ratio',
    xyz_key: str = 'xyz_mesh_updated',
    normal_vector_shift: float = 0.0,
):
    """
    Updates the xyz coordinates of a single ligament or muscle attachment.

    Args:
        model: osim.Model
        force: osim.Force
        state: osim.State
        force_dict: dict
        shift_size_vector: list[float] | np.ndarray
        xyz_key: str
    
    Returns:
        osim.Model
    
    force_dict format:
    {
        'name': 'ligament_name' | 'muscle_name',
        'class': 'Blankevoort1991Ligament' | 'Millard2012EquilibriumMuscle',
        'points': [
            {
                "name": "point_name",
                "location": list[float] | np.ndarray,
                "parent_frame": str,
                "include": True,
                "shift": None | list[float] | np.ndarray,
                "child_location": None | list[float] | np.ndarray,
                "child_frame": None | str,
                "normal_vector": None | list[float] | np.ndarray,
                "use_normal_shift": bool
            },
            {
                "name": "point_name",
                "location": list[float] | np.ndarray,
                "parent_frame": str,
                "include": True,
                "shift": None | list[float] | np.ndarray,
                "child_location": None | list[float] | np.ndarray,
                "child_frame": None | str,
                "normal_vector": None | list[float] | np.ndarray,
                "use_normal_shift": bool
            }
        ]
    }
    """
    force_name = force.getName()
    
    assert force_dict['name'] == force_name
    
    if force_dict['class'] == 'Blankevoort1991Ligament':
        force = osim.Blankevoort1991Ligament.safeDownCast(force)
    elif force_dict['class'] == 'Millard2012EquilibriumMuscle':
        force = osim.Millard2012EquilibriumMuscle.safeDownCast(force)
    else:
        raise ValueError(f'Force {force_name} not found')
    
    geom_path = force.getGeometryPath()
    path_point_set = geom_path.getPathPointSet()
    
    assert path_point_set.getSize() == len(force_dict['points'])
    
    for i, point in enumerate(force_dict['points']):
        if point['include'] == True:
            # update the point xyz using shift ratio
            new_xyz = point[xyz_key]
            
            # don't do a normal vector offset/shift and a regular shift (based on tibia size). 
            # If no "shift_ratio" then use apply normal_vector_shift if use_normal_shift is True.
            # Otherwise, use a zero shift.
            shift_ratio = np.asarray(point[shift_key], dtype=float)
            if not np.allclose(shift_ratio, 0.0):
                shift = shift_ratio * shift_size_vector
            elif point.get('use_normal_shift', False):
                normal_vector = np.asarray(point['normal_vector'])
                normal_vector /= np.linalg.norm(normal_vector)
                shift = normal_vector_shift * normal_vector
            else:
                shift = np.zeros(3)
                
            new_xyz = new_xyz + shift 
            
            # update the point xyz
            update_point_xyz(
                model=model,
                opensim_point=path_point_set.get(i),
                new_xyz=new_xyz,
                state=state,
                child_frame=point['child_frame'],
                parent_frame=point['parent_frame'],
            )

def update_osim_ligament_muscle_attachments(
    model: osim.Model,
    dict_ligament_muscle_attachments: dict,
    tibia_size_vector: Union[list[float], np.ndarray],
    state: osim.State,
    xyz_key: str = 'xyz_mesh',
    normal_vector_shift: float = 0.0
):
    """
    Updates the xyz coordinates of the path points for ligaments and muscles 
    listed in `dict_ligament_muscle_attachments`.

    Args:
        model: osim.Model
        dict_ligament_muscle_attachments: dict
        tibia_size_vector: list[float] | np.ndarray
        state: osim.State
        xyz_key: str
    
    Returns:
        osim.Model
    
    Notes:
    
    xyz_key: 'xyz_mesh' is the location in the meshes used to store/get attachments. 
        Proposing to use 'xyz_mesh_updated' for the updated locations when fitting to a new subject. 
    
    dict_ligament_muscle_attachments format:
    {
        'ligament_name': {
            'name': 'ligament_name',
            'class': 'Blankevoort1991Ligament',
            'points': [
                {
                    "name": "point_name",
                    "location": list[float] | np.ndarray,
                    "parent_frame": str,
                    "include": True,
                    "shift": None | list[float] | np.ndarray,
                    "child_location": None | list[float] | np.ndarray,
                    "child_frame": None | str,
                    "normal_vector": None | list[float] | np.ndarray,
                    "use_normal_shift": bool
                }
            ]
        }
    }
    """
    
    forcesets = model.getForceSet()

    for force_name, dict_ in dict_ligament_muscle_attachments.items():
        force = forcesets.get(force_name)
        update_single_osim_ligament_muscle_attachment(
            model=model,
            force=force,
            state=state,
            force_dict=dict_,
            shift_size_vector=tibia_size_vector,
            xyz_key=xyz_key,
            normal_vector_shift=normal_vector_shift,
        )
        
    
    return model

def update_slack_lengths(model, force_length_dict: dict, state: osim.State=None):
    """
    Updates ligament slack lengths and muscle optimal fiber lengths and tendon slack lengths
    based on the current model state and provided dictionaries.

    Args:
        model (osim.Model or str): The OpenSim model object or path to the model file.
        force_length_dict: dict

    Returns:
        osim.Model: The updated OpenSim model object.
    
    
    Notes:
    
    force_length_dict format:
    {
        'muscle_name': {
            'class': 'Millard2012EquilibriumMuscle',
            'length': float,
            'reference_strain': None,
            'slack_length': None
        },
        'ligament_name': {
            'class': 'Blankevoort1991Ligament',
            'length': float,
            'reference_strain': float,
            'slack_length': float
        }
    }
    """
    if isinstance(model, str):
        model = osim.Model(model)
    
    forces = model.getForceSet()

    if state is None:
        state = model.initSystem()
    
    # Modifying ligaments & muscles
    for i in range(forces.getSize()):
        force_ = forces.get(i)
        force_name = force_.getName()
        if force_name not in force_length_dict.keys():
            logger.debug(f'NOT UPDATING {force_name} - NOT IN force_length_dict')
            continue
        
        logger.debug(f'UPDATING {force_name}')
        
        force_dict_ = force_length_dict[force_name]
        
        assert force_dict_['class'] == force_.getConcreteClassName()
            
        if force_.getConcreteClassName() == 'Millard2012EquilibriumMuscle':
            # get the muscle
            muscle = osim.Millard2012EquilibriumMuscle.safeDownCast(force_)
            # calculate the scale factor
            scale_factor = muscle.getLength(state) / force_dict_['length']
            # update the optimal fiber length
            optimal_ = muscle.getOptimalFiberLength()
            optimal_ *= scale_factor
            muscle.setOptimalFiberLength(round(optimal_, ROUND_DIGITS))
            
            # update the tendon slack length
            slack_ = muscle.getTendonSlackLength()
            slack_ *= scale_factor
            muscle.setTendonSlackLength(round(slack_, ROUND_DIGITS))
            logger.debug(f'scale factor for {force_name}: {scale_factor}')
        elif force_.getConcreteClassName() == 'Blankevoort1991Ligament':
            ligament = osim.Blankevoort1991Ligament.safeDownCast(force_)
            ligament.setSlackLengthFromReferenceStrain(force_dict_['reference_strain'], state)
            logger.debug(f'orig slack length for {force_name}: {force_dict_["slack_length"]}')
            logger.debug(f'new slack length for {force_name}: {ligament.get_slack_length()}')
    
    return model


def update_model_attachments_slacks(
    model: osim.Model,
    dict_lig_mus_attach: Union[dict, str],
    ref_tibia_mesh: Union[Mesh, str],
    state: osim.State = None,
    xyz_key: str = 'xyz_mesh',
    normal_vector_shift: float = 0.,
):
    """
    Updates the ligament and muscle attachments and slack lengths of an OpenSim model.
    
    Args:
        model: osim.Model
        dict_lig_mus_attach: dict | str
        ref_tibia_mesh: Mesh | str
        state: osim.State
        xyz_key: str
        normal_vector_shift: float
    
    
    Notes:
    
    xyz_key: 'xyz_mesh' is the location in the meshes used to store/get attachments. 
        Proposing to use 'xyz_mesh_updated' for the updated locations when fitting to a new subject.
        Default is 'xyz_mesh' is just the locations on the reference mesh surface(s).
    
    normal_vector_shift: float
        This is the absolute magnitude of the shift along the normal vector from the
        surface of the reference mesh.
    
    """
    if state is None:
        state = model.initSystem()
        
    if isinstance(dict_lig_mus_attach, str):
        with open(dict_lig_mus_attach, 'r') as f:
            dict_lig_mus_attach = json.load(f)
            
    if isinstance(ref_tibia_mesh, str):
        ref_tibia_mesh = Mesh(ref_tibia_mesh)
    
    # get the tibia size vector
    tibia_ml = ref_tibia_mesh.point_coords[:,2].max() - ref_tibia_mesh.point_coords[:,2].min()
    tibia_ap = ref_tibia_mesh.point_coords[:,0].max() - ref_tibia_mesh.point_coords[:,0].min()
    average_tibia = (tibia_ml + tibia_ap) / 2
    tibia_size_vector = np.asarray([tibia_ap, average_tibia, tibia_ml])
    
    logger.info('Getting reference Ligament information...')
    # get refernece information stuff (lengths, strains, slack lengths)
    ref_force_info = get_osim_muscle_ligament_reference_lengths(model, state)

    logger.info('Updating ligament and muscle attachments...')
    # update the attachments
    update_osim_ligament_muscle_attachments(
        model=model,
        dict_ligament_muscle_attachments=dict_lig_mus_attach,
        tibia_size_vector=tibia_size_vector.round(ROUND_DIGITS),
        state=state,
        xyz_key=xyz_key,
        normal_vector_shift=normal_vector_shift
    )

    updated_state = model.initSystem()

    logger.info('Updating slack lengths...')
    # then update the slacks - using the new model + the reference information
    update_slack_lengths(
        model=model,
        force_length_dict=ref_force_info,
        state=updated_state
    )

    # finalize model components
    model.finalizeConnections()