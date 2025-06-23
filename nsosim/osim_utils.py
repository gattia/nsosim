import opensim as osim
import numpy as np
from typing import Union
from pymskt.mesh import Mesh
import logging
import json


# Module-level logger. Configure in nsosim.configure_logging()
logger = logging.getLogger(__name__)

def update_ligament_stiffness(model, ligament, stiffness):
    """
    Updates the linear stiffness of a specific ligament in an OpenSim model file.

    Parses the OpenSim model XML, finds the specified ligament by name, and updates
    its <linear_stiffness> value. The changes are written back to the same file.

    Args:
        path_model (str): Path to the OpenSim model (.osim) file to be modified.
        ligament (str): The name of the `Blankevoort1991Ligament` to update.
        stiffness (float or int): The new linear stiffness value.
    """
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)) # keep comments
    tree = ET.parse(path_model, parser)
    root = tree.getroot()[0]
    
    ForceSet = root.find('ForceSet')[0]
    ForceSet.findall(f"./Blankevoort1991Ligament[@name='{ligament}']/linear_stiffness")[0].text = str(int(stiffness))
    
    tree.write(path_model, encoding='utf8',method='xml')


def express_point_in_frame(
        xyz_in_source,        # (x, y, z) expressed in `source_frame`
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
    # Resolve frames (bodyset/… is common, but you can use absolute paths too)
    source_frame = model.getComponent(f'/bodyset/{source_frame_name}').findBaseFrame()
    target_frame = model.getComponent(f'/bodyset/{target_frame_name}').findBaseFrame()

    station_source = osim.Vec3(*xyz_in_source)

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
                
                dict_force_lengths[force_name]['length'] = ligament_length
                dict_force_lengths[force_name]['slack_length'] = slack_length
                dict_force_lengths[force_name]['reference_strain'] = reference_strain
    
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
            elif point['use_normal_shift'] == True:
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
            muscle.setOptimalFiberLength(optimal_)
            
            # update the tendon slack length
            slack_ = muscle.getTendonSlackLength()
            slack_ *= scale_factor
            muscle.setTendonSlackLength(slack_)
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
    
    Returns:
        osim.Model
    
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
        tibia_size_vector=tibia_size_vector,
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