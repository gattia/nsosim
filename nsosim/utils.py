import pyvista as pv
import numpy as np
import re
from pymskt.mesh.anatomical import FemurACS
from NSM.models import TriplanarDecoder, Decoder
from NSM.reconstruct import reconstruct_mesh
import torch
import json


def load_model(config, path_model_state, model_type='triplanar'):
    """
    Loads a pre-trained Neural Shape Model (NSM) from configuration and state files.

    Supports 'triplanar' and 'deepsdf' model architectures. Initializes the model
    based on parameters in the `config` dictionary, loads the learned weights
    from `path_model_state`, moves the model to GPU, and sets it to evaluation mode.

    Args:
        config (dict): A dictionary containing model configuration parameters
            (e.g., latent_size, layer_dimensions, activation functions).
        path_model_state (str): Path to the .pt or .pth file containing the
            saved model state_dict.
        model_type (str, optional): The type of NSM architecture to load.
            Supported values are 'triplanar' and 'deepsdf'. Defaults to 'triplanar'.

    Returns:
        torch.nn.Module: The loaded and initialized NSM model, ready for evaluation.

    Raises:
        ValueError: If `model_type` is not one of the supported values.
    """

    if model_type == 'triplanar':
        model_class = TriplanarDecoder
        params = {
            'latent_dim': config['latent_size'],
            'n_objects': config['objects_per_decoder'],
            'conv_hidden_dims': config['conv_hidden_dims'],
            'conv_deep_image_size': config['conv_deep_image_size'],
            'conv_norm': config['conv_norm'], 
            'conv_norm_type': config['conv_norm_type'],
            'conv_start_with_mlp': config['conv_start_with_mlp'],
            'sdf_latent_size': config['sdf_latent_size'],
            'sdf_hidden_dims': config['sdf_hidden_dims'],
            'sdf_weight_norm': config['weight_norm'],
            'sdf_final_activation': config['final_activation'],
            'sdf_activation': config['activation'],
            'sdf_dropout_prob': config['dropout_prob'],
            'sum_sdf_features': config['sum_conv_output_features'],
            'conv_pred_sdf': config['conv_pred_sdf'],
        }
    elif model_type == 'deepsdf':
        model_class = Decoder
        params = {
            'latent_size': config['latent_size'],
            'dims': config['layer_dimensions'],
            'dropout': config['layers_with_dropout'],
            'dropout_prob': config['dropout_prob'],
            'norm_layers': config['layers_with_norm'],
            'latent_in': config['layer_latent_in'],
            'weight_norm': config['weight_norm'],
            'xyz_in_all': config['xyz_in_all'],
            'latent_dropout': config['latent_dropout'],
            'activation': config['activation'],
            'final_activation': config['final_activation'],
            'concat_latent_input': config['concat_latent_input'],
            'n_objects': config['objects_per_decoder'],
            'progressive_add_depth': config['progressive_add_depth'],
            'layer_split': config['layer_split'],
        }
    else:
        raise ValueError(f'Unknown model type: {model_type}')


    model = model_class(**params)
    saved_model_state = torch.load(path_model_state)
    model.load_state_dict(saved_model_state["model"])
    model = model.cuda()
    model.eval()
    return model

def recon_mesh(
    mesh_paths,
    model,
    model_config,
    n_samples_latent_recon=None,
    num_iter=None,
    scale_jointly=True,
    convergence_patience=None,
    verbose=False
):
    """
    Reconstructs meshes using a Neural Shape Model (NSM) by fitting to target meshes.

    This function takes one or more target mesh paths, an NSM decoder model, and its
    configuration, then performs an optimization process to find a latent vector
    and transformation parameters (scale, ICP registration) that best reconstruct
    the target meshes using the NSM.

    Args:
        mesh_paths (list[str] or str): A list of paths to the target mesh files
            (e.g., VTK, STL) or a single path. If multiple, they are typically
            a bone and its associated cartilage(s) for joint reconstruction.
        model (torch.nn.Module): The pre-loaded NSM decoder model.
        model_config (dict): Configuration dictionary for the NSM model, containing
            parameters for the reconstruction process (e.g., learning rate, number
            of iterations, regularization weights).
        n_samples_latent_recon (int, optional): Number of points to sample from the
            target meshes for latent code inference during each iteration. If None,
            uses a default or value from `model_config`. Defaults to None.
        num_iter (int, optional): Number of optimization iterations. If None, uses
            value from `model_config`. Defaults to None.
        scale_jointly (bool, optional): If True and multiple meshes are provided,
            their scaling factor is optimized jointly. Defaults to True.
        convergence_patience (int, optional): Number of iterations with no improvement
            to wait before considering convergence. If None, uses value from
            `model_config` or a default. Defaults to None.
        verbose (bool, optional): If True, prints progress information during
            reconstruction. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - latent (list): The optimized latent vector(s) as a list.
            - bone_mesh (pyvista.PolyData): The reconstructed primary mesh (typically bone).
            - cart_mesh (pyvista.PolyData): The reconstructed secondary mesh (typically cartilage).
            - mesh_result (dict): A dictionary containing detailed results from the
              `reconstruct_mesh` call, including the latent vector, meshes, and
              registration parameters.
    """

    mesh_result = reconstruct_mesh(
        path=mesh_paths,
        decoders=model,
        latent_size=model_config['latent_size'],
        # Fitting parameters:
        num_iterations=model_config['num_iterations_recon'] if num_iter is None else num_iter,
        l2reg=model_config['l2reg_recon'],
        latent_reg_weight=1e-4,
        loss_type='l1',
        lr=model_config['lr_recon'],
        lr_update_factor=model_config['lr_update_factor_recon'],
        n_lr_updates=model_config['n_lr_updates_recon'],
        return_latent=True,
        register_similarity=True,
        scale_jointly=scale_jointly,
        scale_all_meshes=True,
        objects_per_decoder=2,
        batch_size_latent_recon=model_config['batch_size_latent_recon'],
        get_rand_pts=model_config['get_rand_pts_recon'],
        n_pts_random=model_config['n_pts_random_recon'],
        sigma_rand_pts=model_config['sigma_rand_pts_recon'],
        n_samples_latent_recon=n_samples_latent_recon, 

        calc_symmetric_chamfer=False,
        calc_assd=False,
        calc_emd=False,
        
        # convergence=model_config['convergence_type_recon'],
        convergence=10, 
        convergence_patience=model_config['convergence_patience_recon'],
        # convergence_patience=model_config['convergence_patience_recon'] if convergence_patience is None else convergence_patience,
        clamp_dist=model_config['clamp_dist_recon'],

        fix_mesh=model_config['fix_mesh_recon'],
        verbose=verbose,
        return_registration_params=True,
    )

    # Save the meshes
    bone_mesh = mesh_result['mesh'][0]
    cart_mesh = mesh_result['mesh'][1]

    # get latent
    latent = mesh_result['latent'].detach().cpu().numpy().tolist()

    return latent, bone_mesh, cart_mesh, mesh_result

def read_iv(file_path):
    """
    Reads a 3D mesh from an Inventor (.iv) file format.

    Parses the .iv file to extract vertex coordinates and face connectivity
    information (coordinate indices). It then constructs and returns a
    `pyvista.PolyData` object representing the mesh.

    Args:
        file_path (str): The path to the .iv mesh file.

    Returns:
        pyvista.PolyData: A PyVista mesh object created from the .iv file data.

    Note:
        This parser relies on specific formatting within the .iv file, particularly
        for the 'point' and 'coordIndex' sections. It uses regular expressions
        to find and extract this data.
    """
    pts_list = []
    cns_list = []

    re.IGNORECASE = True
    with open(file_path,'r') as f:
        txt = f.read()
        
        # vertices
        m1 = re.search('point\s(.*)\[',txt)
        m2 = re.search('\]\s\}.*\n\s Ind',txt)
        ptstxt = txt[m1.span()[1]+1:m2.span()[0]]
        # tokens = re.findall(r'[-+]?(?:\d*\.*\d+)\s[-+]?(?:\d*\.*\d+)\s[-+]?(?:\d*\.*\d+),',ptstxt)
        tokens = re.findall(r'-?\ *\d+\.?\d*(?:[Ee]\ *-?\ *\d+)?\s-?\ *\d+\.?\d*(?:[Ee]\ *-?\ *\d+)?\s-?\ *\d+\.?\d*(?:[Ee]\ *-?\ *\d+)?,',ptstxt)
        for i in range(len(tokens)):
            pts_list.append(re.findall("-?\ *\d+\.?\d*(?:[Ee]\ *-?\ *\d+)?",tokens[i]))
            
        # faces
        m1 = re.search('coordIndex\s(.*)\[',txt)
        txt2 = txt[m1.span()[1]+1:]
        m2 = re.search('\]\s\}',txt2)
        cnstxt = txt2[:m2.span()[0]]
        tokens = re.findall(r'[-+]?(?:\d+),\s[-+]?(?:\d+),\s[-+]?(?:\d+)',cnstxt)
        for i in range(len(tokens)):
            cns_list.append(re.findall(r'[-+]?(?:\d+)',tokens[i]))
            
    pts = np.array(pts_list,dtype=np.float32)
    cns = np.array(cns_list,dtype=np.int64)
    cns = np.concatenate((3*np.ones((cns.shape[0],1),dtype=int),cns),axis=1)
    cns = cns.reshape(-1)
    
    mesh = pv.PolyData(pts,cns)

    return mesh

def load_preprocess_ref_mesh(path, z_rel_x, bone):
    """
    Loads a reference mesh from an .iv file and preprocesses it.

    The preprocessing steps include:
    1. Reading the .iv file.
    2. For 'femur' or 'tibia', clipping the mesh along the Z-axis based on `z_rel_x`
       and the mesh's X-axis point-to-point (ptp) distance. The clipping origin
       and inversion depend on whether it's a femur or tibia.
    3. Filling holes and smoothing the clipped mesh (for femur/tibia).
    4. Flipping the Y and X coordinates (y = -y, x = -x).
    5. Scaling the points by 1000 (e.g., meters to millimeters).
    6. Centering the mesh by subtracting the mean of its point coordinates.
    7. Casting point coordinates to `np.float64`.

    Args:
        path (str): Path to the .iv reference mesh file.
        z_rel_x (float): A factor used to determine the Z-clipping range relative
            to the X-axis extent of the mesh. Used for femur and tibia.
        bone (str): The name of the bone ('femur', 'tibia', or other). This affects
            whether Z-clipping is performed.

    Returns:
        tuple: A tuple containing:
            - ref_ (pyvista.PolyData): The loaded and preprocessed reference mesh.
            - mean_ (numpy.ndarray): The mean vector that was subtracted to center
              the mesh (after scaling to mm and flipping axes).
    """
    ref_ = read_iv(path)

    if bone in ['femur', 'tibia']:
        ref_ptp = np.ptp(ref_.points, axis=0)[0]
        range_ = z_rel_x * ref_ptp #(ref_ptp * subject_z_rel_x) / 2
        range_ = range_ if bone == 'femur' else range_ * -1
        invert = bone == 'femur'
        origin = ref_.points.min(axis=0) if bone == 'femur' else ref_.points.max(axis=0)

        ref_ = ref_.clip(normal='z', origin=origin, value=range_, invert=invert)
        ref_ = ref_.fill_holes(hole_size=0.03)
        ref_ = ref_.smooth(100, feature_smoothing=False)

    ref_.points[:,1] = ref_.points[:,1] * -1 
    ref_.points[:,0] = ref_.points[:,0] * -1

    ref_.points = ref_.points * 1000

    mean_ = np.mean(ref_.points, axis=0)

    ref_.points = ref_.points - mean_

    ref_.points = ref_.points.astype(np.float64)
    
    return ref_, mean_

def acs_align_femur(femur):
    """
    Aligns a femur mesh to its Anatomical Coordinate System (ACS).

    Utilizes `pymskt.mesh.anatomical.FemurACS` to define and fit an ACS
    to the provided femur mesh. The mesh is then transformed by applying the
    inverse of the transformation matrix that maps the original mesh to this ACS.
    Effectively, this orients the femur mesh according to its anatomical axes
    at the origin.

    Args:
        femur (pymskt.mesh.Mesh): The femur mesh to be aligned. This mesh object
            is modified in place.

    Returns:
        numpy.ndarray: The 4x4 inverse transformation matrix that was applied to
            the femur mesh to align it to the ACS.
    """
    # Create femur ACS
    femur_acs = FemurACS(femur, cart_label=(11, 12, 13, 14, 15))
    femur_acs.fit()

    four_by_four = np.eye(4)
    four_by_four[:3, -1] = femur_acs.origin
    four_by_four[:3, 0] = femur_acs.ml_axis
    four_by_four[:3, 1] = femur_acs.ap_axis
    four_by_four[:3, 2] = -1 *femur_acs.is_axis
    inverse = np.linalg.inv(four_by_four)

    femur.apply_transform_to_mesh(inverse)

    return inverse

def fit_nsm(
        path_model_state,
        path_model_config,
        list_paths_meshes,
        n_samples_latent_recon=20_000,
        num_iter=None,
        convergence_patience=10,
        scale_jointly=False
):
    """
    Fits a Neural Shape Model (NSM) to a list of target meshes.

    This function loads an NSM model based on provided configuration and state
    files, and then uses `recon_mesh` to fit this model to the target meshes
    (e.g., a bone and its cartilage).

    Args:
        path_model_state (str): Path to the saved NSM model state (.pt file).
        path_model_config (str): Path to the JSON configuration file for the NSM model.
        list_paths_meshes (list[str]): A list of file paths to the target meshes
            to which the NSM will be fitted.
        n_samples_latent_recon (int, optional): Number of points to sample from target
            meshes for latent code inference per iteration. Defaults to 20_000.
        num_iter (int, optional): Number of optimization iterations. If None, uses
            the value from the model config. Defaults to None.
        convergence_patience (int, optional): Number of iterations with no improvement
            to wait before stopping. Defaults to 10.
        scale_jointly (bool, optional): If True and multiple meshes are provided,
            their scaling is optimized jointly. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - latent (list): The optimized latent vector(s).
            - bone_mesh (pyvista.PolyData): The reconstructed primary mesh.
            - cart_mesh (pyvista.PolyData): The reconstructed secondary mesh.
            - result_mesh (dict): A dictionary containing detailed results from
              `recon_mesh`, including the loaded `model` itself.
    """
    with open(path_model_config, 'r') as f:
        model_config = json.load(f)

    model = load_model(model_config, path_model_state, model_type='triplanar')

    latent, bone_mesh, cart_mesh, result_mesh  = recon_mesh(
        list_paths_meshes,
        model,
        model_config,
        n_samples_latent_recon=n_samples_latent_recon,
        num_iter=num_iter,
        convergence_patience=convergence_patience,
        scale_jointly=scale_jointly
    )

    result_mesh['model'] = model

    return latent, bone_mesh, cart_mesh, result_mesh
