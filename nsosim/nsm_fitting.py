import os
import json
import numpy as np
import pyvista as pv
from pymskt.mesh import Mesh
import vtk
import logging
from .utils import acs_align_femur, fit_nsm, read_iv

os.environ['LOC_SDF_CACHE'] = '' # SET DUMMY BECUASE LIBRARY CURRENTLY LOOKS FOR IT. 

from pymskt.mesh import Mesh
from pymskt.mesh.meshTransform import get_linear_transform_matrix
from NSM.mesh.interpolate import interpolate_points

# Module-level logger. Configure with nsosim.configure_logging()
logger = logging.getLogger(__name__)

OSIM_TO_NSM_TRANSFORM = np.array([
    [0,  -1,  0],  # x -> -y
    [0, 0,  1],  # y -> z
    [-1, 0,  0]   # z -> -x
])

def align_bone_osim_fit_nsm(
    bone,
    dict_bone,
    folder_save,
    rigid_reg_type='rigid', # 'similarity' or 'rigid'
    acs_align=False,
    save_intermediate_cartilage=True,
    save_intermediate_men=True,
    save_intermediate_fibula=True,
    intermediate_cartilage_exists_ok=True,
    intermediate_men_exists_ok=True,
    intermediate_fibula_exists_ok=True,
    intermediate_cart_name='{bone}_cart.vtk',
    intermediate_men_name='{bone}_{med_lat}_men.vtk',
    intermediate_fibula_name='{bone}_fibula.vtk',
    n_samples_latent_recon=20_000,
    num_iter=None,
    convergence_patience=50,
    femur_transform=None,
    femur_acs_inverse=None
):
    """
    Aligns a single bone (femur, tibia, or patella) and its cartilage, then fits an NSM.

    The process involves:
    1. Loading the subject's bone mesh.
    2. If it's the femur and `acs_align` is True, aligns it to a predefined
       anatomical coordinate system (ACS).
    3. Rigidly registers the bone to a reference bone mesh. For non-femur bones,
       it applies transformations derived from the femur's alignment and registration.
    4. Loads and processes the associated cartilage mesh(es), applying the same
       transformations.
    5. Saves intermediate aligned bone and cartilage meshes.
    6. Fits a Neural Shape Model (NSM) to the aligned bone and cartilage meshes.
    7. Stores the NSM reconstruction results (latent vector, meshes, parameters)
       in the input dictionary `dict_bone`.

    Args:
        bone (str): The name of the bone (e.g., 'femur', 'tibia', 'patella').
        dict_bone (dict): A dictionary containing paths and parameters for the bone.
            The dictionary is updated in place during the function execution.
        folder_save (str): Path to the directory where aligned and NSM-reconstructed
            meshes will be saved.
        rigid_reg_type (str, optional): Type of rigid registration ('rigid' or
            'similarity'). Defaults to 'rigid'.
        acs_align (bool, optional): Whether to perform ACS alignment for the femur.
            Defaults to False.
        save_intermediate_cartilage (bool, optional): Whether to save the aligned
            cartilage mesh. Defaults to True.
        intermediate_cartilage_exists_ok (bool, optional): If True, allows overwriting
            existing intermediate cartilage files. Defaults to True.
        intermediate_cart_name (str, optional): Filename template for the saved
            intermediate cartilage mesh. Defaults to '{bone}_cart.vtk'.
        n_samples_latent_recon (int, optional): Number of samples for latent space
            reconstruction during NSM fitting. Defaults to 20_000.
        num_iter (int, optional): Number of iterations for NSM fitting. If None,
            uses model's default. Defaults to None.
        convergence_patience (int, optional): Patience for NSM fitting convergence.
        femur_transform (numpy.ndarray, optional): 4x4 transformation matrix from
            femur registration, applied to non-femur bones. Required if `bone` is not 'femur'.
        femur_acs_inverse (numpy.ndarray, optional): 4x4 inverse ACS transformation
            from femur, applied to non-femur bones if `acs_align` was True for femur.
            Required if `bone` is not 'femur' and `acs_align` was used for femur.

    Returns:
        dict: The updated `dict_bone` containing NSM fitting results and transformations.

    Raises:
        AssertionError: If `femur_transform` or `femur_acs_inverse` are not provided
            when required for non-femur bones.
        ValueError: If cartilage filename in `dict_bone` is not a string or list, or
            if an intermediate cartilage file exists and `intermediate_cartilage_exists_ok`
            is False.
    """
    # Resolve the folder that stores the subject-specific meshes (added in refactor)
    folder_subject = dict_bone['subject']['folder']

    subject_bone = Mesh(os.path.join(folder_subject, dict_bone['subject']['bone_filename']))

    # load in the reference mesh
    ref_ = Mesh(dict_bone['ref']['bone_filepath'])

    # align the subject's bone with the anatomical coordinate system of the femur
    # then rigidly register to the femur bone of template 
    # TODO: get rid of this extra alignment transform later?
    if bone == 'femur':
        if acs_align:
            femur_acs_inverse = acs_align_femur(subject_bone)
        femur_transform = subject_bone.rigidly_register(
            other_mesh=ref_,
            as_source=True,
            apply_transform_to_mesh=True,
            return_transformed_mesh=False,
            return_transform=True,
            max_n_iter=100,
            n_landmarks=1000,
            reg_mode=rigid_reg_type
        )
    else:
        logger.debug(f'Applying transforms to {bone}')
        if acs_align:
            assert femur_acs_inverse is not None, 'Femur acs inverse not provided'
            subject_bone.apply_transform_to_mesh(femur_acs_inverse)
        assert femur_transform is not None, 'Femur transform not provided'
        subject_bone.apply_transform_to_mesh(femur_transform)
    

    # Cartilage processing 
    if isinstance(dict_bone['subject']['cart_filename'], list):
        # combine all mesh objects into one - all of our current NSM models
        # have a single cartilage mesh/surface per bone.
        cart_mesh = pv.PolyData()
        for cart_path in dict_bone['subject']['cart_filename']:
            cart_mesh += pv.PolyData(os.path.join(folder_subject, cart_path))
        
        # turn into pymskt mesh
        cart_mesh = Mesh(cart_mesh)

        # setup new name for cartilage mesh & save intermediate version if desired
        cart_name = intermediate_cart_name.format(bone=bone) #f'{bone}_cart.vtk'
        if save_intermediate_cartilage:
            # handle if the mesh exists etc. 
            path_save_intermediate_cartilage = os.path.join(folder_subject, cart_name)
            if (not os.path.exists(path_save_intermediate_cartilage)) or intermediate_cartilage_exists_ok:
                cart_mesh.save_mesh(path_save_intermediate_cartilage)
            else:
                raise ValueError('Cartilage mesh name used for saving intermediate version already exists')
            cart_mesh.save_mesh(os.path.join(folder_subject, cart_name))
        dict_bone['subject']['cart_filename'] = cart_name
    elif isinstance(dict_bone['subject']['cart_filename'], str):
        cart_mesh = Mesh(os.path.join(folder_subject, dict_bone['subject']['cart_filename']))
    else:
        raise ValueError('Cartilage filename not a string or list')


    

    # Apply the femur transform to the cartilage mesh
    if acs_align:
        cart_mesh.apply_transform_to_mesh(femur_acs_inverse)
    cart_mesh.apply_transform_to_mesh(femur_transform)

        
    # create filenames to save meshes - so they can be loaded into the nsm recon function. 
    # TODO: Update NSM recon function to 
    new_bone_filename = dict_bone['subject']['bone_filename'].replace('.vtk', '_aligned.vtk')
    new_cart_filename = dict_bone['subject']['cart_filename'].replace('.vtk', '_aligned.vtk')

    path_bone = os.path.join(folder_save, new_bone_filename)
    path_cart = os.path.join(folder_save, new_cart_filename)

    subject_bone.save_mesh(path_bone)
    cart_mesh.save_mesh(path_cart)

    paths_meshes = [
        path_bone,
        path_cart,
    ]
    
    # check if there are menisci, if there are, then need to load them and apply the same
    # transform. Menisci have their own surfaces (they are not combined like the cartilage)
    # - load the meshes & save the intermediate (original alignment?)
    # - apply the transform
    # - save the transformed meshes
    # - create the new filenames
    # - add to the paths_meshes list
    
    for med_lat in ['med', 'lat']:
        if f'{med_lat}_men_filename' in dict_bone['subject'].keys():
            men_mesh = Mesh(os.path.join(folder_subject, dict_bone['subject'][f'{med_lat}_men_filename']))
            if save_intermediate_men:
                men_name = intermediate_men_name.format(bone=bone, med_lat=med_lat)
                path_save_intermediate_men = os.path.join(folder_subject, men_name)
                if (not os.path.exists(path_save_intermediate_men)) or intermediate_men_exists_ok:
                    men_mesh.save_mesh(path_save_intermediate_men)
                else:
                    raise ValueError('Meniscus mesh name used for saving intermediate version already exists')
            if acs_align:
                men_mesh.apply_transform_to_mesh(femur_acs_inverse)
            men_mesh.apply_transform_to_mesh(femur_transform)
            new_men_filename = dict_bone['subject'][f'{med_lat}_men_filename'].replace('.vtk', '_aligned.vtk')
            path_men = os.path.join(folder_save, new_men_filename)
            men_mesh.save_mesh(path_men)
            paths_meshes.append(path_men)
    
    # also check if fibula is present, if it is, then need to load it and apply the same
    # transform. 
    # - load the mesh & save the intermediate (original alignment?)
    # - apply the transform
    # - save the transformed mesh
    # - create the new filename
    # - add to the paths_meshes list
    if 'fibula_filename' in dict_bone['subject'].keys():
        fibula_mesh = Mesh(os.path.join(folder_subject, dict_bone['subject']['fibula_filename']))
        if save_intermediate_fibula:
            fibula_name = intermediate_fibula_name.format(bone=bone)
            path_save_intermediate_fibula = os.path.join(folder_subject, fibula_name)
            if (not os.path.exists(path_save_intermediate_fibula)) or intermediate_fibula_exists_ok:
                fibula_mesh.save_mesh(path_save_intermediate_fibula)
            else:
                raise ValueError('Fibula mesh name used for saving intermediate version already exists')
        if acs_align:
            fibula_mesh.apply_transform_to_mesh(femur_acs_inverse)
        fibula_mesh.apply_transform_to_mesh(femur_transform)
        new_fibula_filename = dict_bone['subject']['fibula_filename'].replace('.vtk', '_aligned.vtk')
        path_fibula = os.path.join(folder_save, new_fibula_filename)
        fibula_mesh.save_mesh(path_fibula)
        paths_meshes.append(path_fibula)

    # DO THE NSM FITTING
    logger.info(f'Fitting NSM for {bone}')
    recon_output = fit_nsm(
        path_model_state=dict_bone['model']['path_model_state'],
        path_model_config=dict_bone['model']['path_model_config'],
        list_paths_meshes=paths_meshes,
        n_samples_latent_recon=n_samples_latent_recon,
        num_iter=num_iter,
        convergence_patience=convergence_patience,
    )

    # SAVE THE RECONSTRUCTED MESHES - IN THE DICT, AND BOTH IN mm AND m
    # store the results in the dictionary
    dict_bone['subject']['bone_mesh_nsm'] = recon_output['bone_mesh']
    dict_bone['subject']['cart_mesh_nsm'] = recon_output.get('cart_mesh') # Use .get() in case cart_mesh is not always present
    dict_bone['subject']['recon_dict'] = recon_output['mesh_result']
    dict_bone['subject']['recon_latent'] = recon_output['latent']

    if bone == 'femur':
        dict_bone['subject']['transform'] = femur_transform
        if acs_align:
            dict_bone['subject']['acs_inverse'] = femur_acs_inverse
        else:
            dict_bone['subject']['acs_inverse'] = None
        # if meniscus was fitted with the nsm model... then add it to the dict. 
        if 'med_men_mesh' in recon_output.keys():
            dict_bone['subject']['med_men_mesh_nsm'] = recon_output['med_men_mesh']
        if 'lat_men_mesh' in recon_output.keys():
            dict_bone['subject']['lat_men_mesh_nsm'] = recon_output['lat_men_mesh']
    
    if bone == 'tibia':
        if 'fibula_mesh' in recon_output.keys():
            dict_bone['subject']['fibula_mesh_nsm'] = recon_output['fibula_mesh']

    return dict_bone

def align_knee_osim_fit_nsm(
    dict_bones,
    folder_save_bones,
    n_samples_latent_recon=20_000,
    convergence_patience=10,
    rigid_reg_type='rigid',
    acs_align=False,
):
    """
    Aligns all knee bones (femur, tibia, patella) and their cartilages, then fits NSMs.

    This function iterates through each bone specified in `dict_bones` (typically
    femur, tibia, patella) and calls `align_bone_osim_fit_nsm` for each one.
    Transformations from the femur (ACS alignment and registration) are passed
    to the other bones to ensure consistent alignment of the entire knee joint.

    After NSM fitting for each bone, it saves:
    - The alignment transformation parameters (linear transform, scale, center) as a JSON file.
    - The NSM latent vector as a .npy file.
    - The NSM-reconstructed bone and cartilage meshes in VTK format (in mm).

    Args:
        dict_bones (dict): Dictionary mapping bone names (e.g., 'femur', 'tibia', 'patella') 
            to their corresponding configuration dictionaries. Each bone's dictionary 
            must contain a 'subject' key with a 'folder' field pointing to the directory 
            containing that bone's mesh files.
        folder_save_bones (str): Path to the base directory where aligned and
            NSM-reconstructed meshes and related files will be saved (subdirectories
            will be created for each bone).
        n_samples_latent_recon (int, optional): Number of samples for latent space
            reconstruction. Defaults to 20_000.
        convergence_patience (int, optional): Patience for NSM fitting convergence.
            Defaults to 10.
        rigid_reg_type (str, optional): Type of rigid registration. Defaults to 'rigid'.
        acs_align (bool, optional): Whether to perform ACS alignment for the femur.
            Defaults to False.

    Returns:
        dict: The updated `dict_bones` with results from NSM fitting for all bones.
    """

    # Ensure the femur (which provides the reference transform) is always processed first.
    # Python ≥3.7 preserves insertion order for normal dicts, but callers might build the
    # dictionary with a comprehension or in another order.  We therefore build an explicit
    # ordered list of bones so that downstream logic relying on femur transforms is safe.

    ordered_bones = []
    if 'femur' in dict_bones:
        ordered_bones.append('femur')  # femur must be processed first

    # Append all remaining bones in their original order (if any)
    ordered_bones.extend([b for b in dict_bones.keys() if b != 'femur'])

    for bone in ordered_bones:
        dict_ = dict_bones[bone]
        logger.info(f'Processing {bone}')

        if bone == 'femur':
            femur_transform = None
            femur_acs_inverse = None
        else:
            femur_transform = dict_bones['femur']['subject']['transform']
            femur_acs_inverse = dict_bones['femur']['subject']['acs_inverse']
            
        dict_bones[bone] =  align_bone_osim_fit_nsm(
            bone=bone,
            dict_bone=dict_,
            folder_save=folder_save_bones,
            rigid_reg_type=rigid_reg_type, # 'similarity' or 'rigid'
            acs_align=acs_align,
            save_intermediate_cartilage=True,
            intermediate_cartilage_exists_ok=True,
            n_samples_latent_recon=n_samples_latent_recon,
            num_iter=None,
            convergence_patience=convergence_patience,
            femur_transform=femur_transform,
            femur_acs_inverse=femur_acs_inverse
        )

        recon_dict = dict_bones[bone]['subject']['recon_dict']
        linear_transform = get_linear_transform_matrix(recon_dict['icp_transform'])
        scale = recon_dict['scale']
        center = recon_dict['center']
        dict_transform = {
            'linear_transform': linear_transform.tolist(),
            'scale': scale,
            'center': center.tolist()
        }

        folder_save_bones_ = os.path.join(folder_save_bones, bone)
        os.makedirs(folder_save_bones_, exist_ok=True)

        with open(os.path.join(folder_save_bones_, f'{bone}_alignment.json'), 'w') as f:
            json.dump(dict_transform, f, indent=4)
        
        # save the latent vector(s)
        np.save(os.path.join(folder_save_bones_, f'{bone}_latent.npy'), dict_bones[bone]['subject']['recon_latent'])

        dict_bones[bone]['subject']['bone_mesh_nsm'].save_mesh(os.path.join(folder_save_bones_, dict_bones[bone]['subject']['bone_filename'].replace('.vtk', '_nsm_recon_mm.vtk')))
        dict_bones[bone]['subject']['cart_mesh_nsm'].save_mesh(os.path.join(folder_save_bones_, dict_bones[bone]['subject']['cart_filename'].replace('.vtk', '_nsm_recon_mm.vtk')))
        if 'med_men_mesh_nsm' in dict_bones[bone]['subject'].keys():
            dict_bones[bone]['subject']['med_men_mesh_nsm'].save_mesh(os.path.join(folder_save_bones_, dict_bones[bone]['subject']['med_men_filename'].replace('.vtk', '_nsm_recon_mm.vtk')))
        if 'lat_men_mesh_nsm' in dict_bones[bone]['subject'].keys():
            dict_bones[bone]['subject']['lat_men_mesh_nsm'].save_mesh(os.path.join(folder_save_bones_, dict_bones[bone]['subject']['lat_men_filename'].replace('.vtk', '_nsm_recon_mm.vtk')))
        if 'fibula_mesh_nsm' in dict_bones[bone]['subject'].keys():
            dict_bones[bone]['subject']['fibula_mesh_nsm'].save_mesh(os.path.join(folder_save_bones_, dict_bones[bone]['subject']['fibula_filename'].replace('.vtk', '_nsm_recon_mm.vtk')))

    return dict_bones


def interpolate_ref_points_nsm_space(
    ref_mesh_path,
    path_ref_transform_file,
    path_ref_latent,
    model,
    subject_latent,
    surface_idx=0,
    n_steps=100,
):
    """
    Interpolates points from a reference mesh to a subject's NSM latent space.

    Given a reference mesh, its transformation to a reference NSM space, the
    reference NSM latent vector, an NSM model, and a subject's NSM latent vector,
    this function finds the corresponding 3D point locations in the subject's
    NSM space for each point on the reference mesh.

    Steps:
    1.  Loads the reference mesh (IV or VTK format).
    2.  Loads transformation parameters (ICP transform, center, scale) that map the
        reference mesh to the reference NSM space.
    3.  Transforms the reference mesh points into the reference NSM space.
    4.  Loads the reference NSM latent vector.
    5.  Uses latent space interpolation (`pymskt.mesh.interpolate.interpolate_points`)
        to find the positions of these points given the subject's latent vector.

    Args:
        ref_mesh_path (str): Path to the reference mesh file (.iv or .vtk).
        path_ref_transform_file (str): Path to a JSON file containing the transformation
            parameters (`transform_matrix`, `mean_orig`, `orig_scale`) for the
            reference mesh to reference NSM space.
        path_ref_latent (str): Path to a .npy file containing the reference NSM latent vector.
        model: The trained NSM model object (e.g., from `NSM.models`).
        subject_latent (numpy.ndarray or list): The NSM latent vector for the target subject.

    Returns:
        dict: A dictionary containing:
            - 'interpolated_points' (numpy.ndarray): The interpolated point coordinates
              in the subject's NSM space.
            - 'center' (numpy.ndarray): The centering vector used for the reference mesh
              transformation.
            - 'scale' (numpy.ndarray): The scaling factor used for the reference mesh
              transformation.

    Raises:
        ValueError: If `ref_mesh_path` has an unrecognized file extension.
    """
    if ref_mesh_path.endswith('.iv'):
        raise NotImplementedError('IV files deprecated')
        ref_mesh = read_iv(ref_mesh_path)
    elif ref_mesh_path.endswith('.vtk'):
        ref_mesh = pv.PolyData(ref_mesh_path)
    else:
        raise ValueError('Ref mesh path not recognized')
    
    # load in / get the registration parameters for converting
    # the reference femur mesh to the reference NSM space
    with open(path_ref_transform_file, 'r') as f:
        dict_transforms = json.load(f)
        icp_transform = np.array(dict_transforms['transform_matrix']).reshape(4,4)
        center = np.array(dict_transforms['mean_orig'])
        scale = np.array(dict_transforms['orig_scale'])
    
    # Convert from OSIM to NSM coordinates
    ref_points = ref_mesh.points.copy()
    ref_points = convert_OSIM_to_nsm_(ref_points, center)
    ref_points = ref_points.astype(np.float64)

    # padding ones to apply the transform (4x4)
    ref_points = np.concatenate((ref_points, np.ones((ref_points.shape[0],1))), axis=1)
    
    # apply the transform
    ref_points_nsm_space = (ref_points @ icp_transform.T)[:,:3]

    # load reference latent
    latent_ref = np.load(path_ref_latent)


    # use latent interpolation to find NSM space xyz positions for matching points 
    if isinstance(subject_latent, list):
        subject_latent = np.array(subject_latent)
        
    if len(subject_latent.shape) == 2:
        subject_latent = subject_latent[0,:]
    
    interpolated_points = interpolate_points(
        model,
        latent_ref[0,:],
        subject_latent,
        n_steps=n_steps,
        points1=ref_points_nsm_space,
        surface_idx=surface_idx,
        verbose=False,
        spherical=True
    )

    dict_results = {
        'interpolated_points': interpolated_points,
        'center': center,
        'scale': scale,

    }

    return dict_results
    # return interpolated_points

def interp_ref_to_subject_to_osim(
    path_surface,
    surface_name,
    ref_center,
    dict_bones,
    folder_nsm_files,
    surface_idx=0,
    n_steps=100,
):
    """
    Interpolates reference points to a subject's NSM space and then to OSIM space.

    This function orchestrates the process of taking predefined points on a
    reference bone model, finding their corresponding locations on a subject-specific
    NSM reconstruction, and finally transforming these locations into the OpenSim (OSIM)
    coordinate system.

    Args:
        bone (str): The name of the bone (e.g., 'femur', 'tibia').
        ref_center (numpy.ndarray): The original centering vector of the reference mesh
            before it was transformed to the NSM reference space.
        folder_ref_bones (str): Path to the directory containing reference bone files
            (e.g., ACLC_mean_Femur.iv).
        dict_bones (dict): Dictionary containing the subject's NSM fitting results,
            including the model, latent vector, and transformation parameters.
        folder_nsm_files (str): Path to the directory containing NSM-related files for
            the reference model (alignment JSON, latent vector .npy).

    Returns:
        numpy.ndarray: The interpolated point coordinates transformed into the
            OSIM coordinate system.
    """
    # Set femur params
    path_transform_file = os.path.join(folder_nsm_files, surface_name, f'ref_{surface_name}_alignment.json')
    path_ref_latent = os.path.join(folder_nsm_files, surface_name, f'latent_{surface_name}.npy')

    # get interpolated ref femur points
    ref_results = interpolate_ref_points_nsm_space(
        ref_mesh_path=path_surface,
        path_ref_transform_file=path_transform_file,
        path_ref_latent=path_ref_latent,
        model=dict_bones[surface_name]['subject']['recon_dict']['model'],
        subject_latent=dict_bones[surface_name]['subject']['recon_latent'],
        surface_idx=surface_idx,
        n_steps=n_steps,
    )

    # convert interpolated points to OSIM space
    interpolated_pts_osim = convert_nsm_recon_to_OSIM(
        points=ref_results['interpolated_points'],
        icp_transform=dict_bones[surface_name]['subject']['recon_dict']['icp_transform'],
        scale=dict_bones[surface_name]['subject']['recon_dict']['scale'],
        center=dict_bones[surface_name]['subject']['recon_dict']['center'],
        ref_mesh_orig_center=ref_center
    )

    return interpolated_pts_osim

def apply_transform(
    points,
    icp_transform,
    scale,
    center,
):
    """
    Applies a transformation (ICP, scaling, centering) to a set of points.

    This is typically used to transform points from a canonical/normalized space
    (e.g., NSM space) to a subject-specific or original mesh space.

    The transformation sequence is:
    1. Apply the ICP transformation (a 4x4 matrix).
    2. Undo centering (subtract the center vector).
    3. Undo scaling (divide by the scale factor).

    Args:
        points (numpy.ndarray): Nx3 array of points to transform.
        icp_transform (numpy.ndarray): 4x4 ICP transformation matrix.
        scale (float or numpy.ndarray): Scaling factor.
        center (numpy.ndarray): 3D centering vector.

    Returns:
        numpy.ndarray: Nx3 array of transformed points.
    """
    points_ = points.copy()
    
    # E first apply the transform, and then apply centering 
    # and scaling 
    
    # pad the points with 1s so we can apply the 4x4 transform
    points_ = np.concatenate((points_, np.ones((points_.shape[0],1))), axis=1)
    # apply the transform
    points_ = (points_ @ icp_transform.T)[:,:3]
    
    # undo the centering and scaling
    # points_ = (points_/scale) + center #(points_ - center) / scale
    points_ = (points_-center)/scale
    
    return points_

def undo_transform(
    points,
    icp_transform,
    scale,
    center,
):
    """
    Reverses a transformation (ICP, scaling, centering) applied to a set of points.

    This is typically used to transform points from a subject-specific or original
    mesh space back to a canonical/normalized space (e.g., NSM space).

    The inverse transformation sequence is:
    1. Apply scaling (multiply by the scale factor).
    2. Apply centering (add the center vector).
    3. Apply the inverse of the ICP transformation.

    Args:
        points (numpy.ndarray): Nx3 array of points to transform.
        icp_transform (numpy.ndarray): 4x4 ICP transformation matrix.
        scale (float or numpy.ndarray): Scaling factor.
        center (numpy.ndarray): 3D centering vector.

    Returns:
        numpy.ndarray: Nx3 array of transformed points.
    """
    points_ = points.copy()
    # apple inverse scaling
    points_ *= scale
    points_ += center

    # pad the points with 1s so we can apply the 4x4 transform
    points_ = np.concatenate((points_, np.ones((points_.shape[0],1))), axis=1) # allow multiple by 4x4 transform
    points_ = (points_ @ np.linalg.inv(icp_transform).T)[:,:3] # apply the inverse transform and remove the 1s column

    return points_


def convert_nsm_recon_to_OSIM_(
    points_,
    ref_mesh_orig_center,
):
    """
    Converts points from a (typically NSM-reconstructed) space to OpenSim (OSIM) coordinates.

    This specific version of the conversion involves:
    1. Adding back an original reference mesh centering bias.
    2. Scaling from millimeters to meters.
    3. Applying a combined coordinate transformation and rotation to OSIM space.

    Args:
        points_ (numpy.ndarray): Nx3 array of points in the source space (e.g., mm, NSM-aligned).
        ref_mesh_orig_center (numpy.ndarray): The original centering vector that was
            subtracted from the reference mesh before NSM processing.

    Returns:
        numpy.ndarray: Nx3 array of points in OSIM coordinates (meters).
    """
    # add back bias from reference mesh 
    points_ += ref_mesh_orig_center

    # convert from mm to m
    points_ /= 1000

    points_osim = points_ @ OSIM_TO_NSM_TRANSFORM.T

    return points_osim

def convert_OSIM_to_nsm_(
    points_,
    ref_mesh_orig_center,
):
    """
    Converts points from OpenSim (OSIM) coordinates back to an NSM-reconstructable space.

    This is the inverse of `convert_nsm_recon_to_OSIM_`. It involves:
    1. Applying the inverse combined coordinate transformation and rotation.
    2. Scaling from meters to millimeters.
    3. Subtracting the original reference mesh centering bias.

    Args:
        points_ (numpy.ndarray): Nx3 array of points in OSIM coordinates (meters).
        ref_mesh_orig_center (numpy.ndarray): The original centering vector that was
            subtracted from the reference mesh before NSM processing.

    Returns:
        numpy.ndarray: Nx3 array of points in the source space (e.g., mm, NSM-aligned).
    """
    # Inverse of the combined transformation matrix
    # This is the inverse of the combined coordinate swapping + MRI to OSIM rotation
    points_ = points_ @ OSIM_TO_NSM_TRANSFORM
    
    # scale from m to mm
    points_ *= 1000
    
    # remove bias from reference mesh 
    points_ -= ref_mesh_orig_center

    return points_

def convert_nsm_recon_to_OSIM(
    points,
    icp_transform,
    scale,
    center,
    ref_mesh_orig_center,
):
    """
    Converts NSM-reconstructed points to the OpenSim (OSIM) coordinate system.

    This function combines `undo_transform` (to go from canonical NSM space to
    the subject's aligned physical space in mm) and then `convert_nsm_recon_to_OSIM_`
    (to go from that physical space to OSIM coordinates in meters).

    Args:
        points (numpy.ndarray): Nx3 array of points in the canonical NSM space.
        icp_transform (vtk.vtkIterativeClosestPointTransform or numpy.ndarray):
            The ICP transformation applied during NSM fitting. Can be a VTK object
            or a 4x4 numpy array.
        scale (float or numpy.ndarray): Scaling factor from NSM fitting.
        center (numpy.ndarray): Centering vector from NSM fitting.
        ref_mesh_orig_center (numpy.ndarray): The original centering vector of the
            reference mesh before it was transformed to the NSM reference space.

    Returns:
        numpy.ndarray: Nx3 array of points in OSIM coordinates (meters).

    Raises:
        ValueError: If `icp_transform` is not a valid type.
    """
    if isinstance(icp_transform, vtk.vtkIterativeClosestPointTransform):
        icp_transform = get_linear_transform_matrix(icp_transform)
    elif isinstance(icp_transform, np.ndarray):
        pass
    else:
        raise ValueError('icp_transform not a valid type')

    points_ = undo_transform(points, icp_transform, scale, center)

    points_osim = convert_nsm_recon_to_OSIM_(points_, ref_mesh_orig_center)
    
    return points_osim

def convert_OSIM_to_nsm(
    points,
    icp_transform,
    scale,
    center,
    ref_mesh_orig_center,
):
    """
    Converts points from OpenSim (OSIM) coordinate system to NSM space.

    This function is the inverse of `convert_nsm_recon_to_OSIM`. It combines
    `convert_OSIM_to_nsm_` (to go from OSIM coordinates to the subject's aligned
    physical space in mm) and then `apply_transform` (to go from that physical
    space to the canonical NSM space).

    Args:
        points (numpy.ndarray): Nx3 array of points in OSIM coordinates (meters).
        icp_transform (vtk.vtkIterativeClosestPointTransform or numpy.ndarray):
            The ICP transformation applied during NSM fitting. Can be a VTK object
            or a 4x4 numpy array.
        scale (float or numpy.ndarray): Scaling factor from NSM fitting.
        center (numpy.ndarray): Centering vector from NSM fitting.
        ref_mesh_orig_center (numpy.ndarray): The original centering vector of the
            reference mesh before it was transformed to the NSM reference space.

    Returns:
        numpy.ndarray: Nx3 array of points in the canonical NSM space.

    Raises:
        ValueError: If `icp_transform` is not a valid type.
    """
    if isinstance(icp_transform, vtk.vtkIterativeClosestPointTransform):
        icp_transform = get_linear_transform_matrix(icp_transform)
    elif isinstance(icp_transform, np.ndarray):
        pass
    else:
        raise ValueError('icp_transform not a valid type')
    
    points_ = convert_OSIM_to_nsm_(points, ref_mesh_orig_center)
    
    points_nsm = apply_transform(points_, icp_transform, scale, center)
    
    return points_nsm

def nsm_recon_to_osim(
    bone,
    dict_bones,
    fem_ref_center,
    bone_clusters=20_000,
    cart_clusters=None,
):
    """
    Transforms NSM-reconstructed bone and cartilage meshes to OSIM coordinates.

    Takes the bone and cartilage meshes from NSM reconstruction (which are typically
    in a centered, scaled, and possibly ICP-aligned space in mm), converts their
    point coordinates to the OSIM coordinate system (meters) using
    `convert_nsm_recon_to_OSIM_`. Optionally resamples the meshes to a target
    number of points/clusters.

    Args:
        bone (str): The name of the bone (e.g., 'femur', 'tibia'). Used to access
            the correct meshes from `dict_bones`.
        dict_bones (dict): Dictionary containing the subject's NSM fitting results,
            including `bone_mesh_nsm` and `cart_mesh_nsm` for the specified `bone`.
        fem_ref_center (numpy.ndarray): The original centering vector of the reference
            femur mesh. This is used by `convert_nsm_recon_to_OSIM_` for all bones
            assuming a consistent reference frame was used.
        bone_clusters (int, optional): Target number of points for resampling the
            bone mesh after coordinate conversion. If None, no resampling. Defaults to 20_000.
        cart_clusters (int, optional): Target number of points for resampling the
            cartilage mesh after coordinate conversion. If None, no resampling. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - bone_mesh_osim (pymskt.mesh.Mesh): The bone mesh with points in OSIM
              coordinates, optionally resampled.
            - cart_mesh_osim (pymskt.mesh.Mesh): The cartilage mesh with points in
              OSIM coordinates, optionally resampled.
    """
    bone_mesh = dict_bones[bone]['subject']['bone_mesh_nsm'].copy()
    cart_mesh = dict_bones[bone]['subject']['cart_mesh_nsm'].copy()
    

    bone_pts_osim = convert_nsm_recon_to_OSIM_(bone_mesh.point_coords, fem_ref_center)
    cart_pts_osim = convert_nsm_recon_to_OSIM_(cart_mesh.point_coords, fem_ref_center)

    # create copies of the meshes, update the points, and save them to disk
    bone_mesh_osim = bone_mesh.copy()
    bone_mesh_osim.point_coords = bone_pts_osim
    if bone_clusters is not None:
        bone_mesh_osim.resample_surface(subdivisions=1, clusters=bone_clusters)
    # bone_mesh_osim.save_mesh(os.path.join(folder_save_bones, 'tibia', 'tibia_nsm_recon_osim.stl'))

    cart_mesh_osim = cart_mesh.copy()
    cart_mesh_osim.point_coords = cart_pts_osim
    if cart_clusters is not None:
        cart_mesh_osim.resample_surface(subdivisions=1, clusters=cart_clusters)
    # cart_mesh_osim.save_mesh(os.path.join(folder_save_bones, 'tibia', 'tibia_cartilage_nsm_recon_osim.vtk'))

    return bone_mesh_osim, cart_mesh_osim




    