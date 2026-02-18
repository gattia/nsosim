import json
import logging
import os
import re

import numpy as np
import pyvista as pv
import torch
from NSM.models import Decoder, TriplanarDecoder
from NSM.reconstruct import reconstruct_mesh
from pymskt.mesh import Mesh
from pymskt.mesh.anatomical import FemurACS

logger = logging.getLogger(__name__)

BONE_CLIPPING_FACTOR = 0.95


def load_model(config, path_model_state, model_type="triplanar"):
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

    if model_type == "triplanar":
        model_class = TriplanarDecoder
        params = {
            "latent_dim": config["latent_size"],
            "n_objects": config["objects_per_decoder"],
            "conv_hidden_dims": config["conv_hidden_dims"],
            "conv_deep_image_size": config["conv_deep_image_size"],
            "conv_norm": config["conv_norm"],
            "conv_norm_type": config["conv_norm_type"],
            "conv_start_with_mlp": config["conv_start_with_mlp"],
            "sdf_latent_size": config["sdf_latent_size"],
            "sdf_hidden_dims": config["sdf_hidden_dims"],
            "sdf_weight_norm": config["weight_norm"],
            "sdf_final_activation": config["final_activation"],
            "sdf_activation": config["activation"],
            "sdf_dropout_prob": config["dropout_prob"],
            "sum_sdf_features": config["sum_conv_output_features"],
            "conv_pred_sdf": config["conv_pred_sdf"],
        }
    elif model_type == "deepsdf":
        model_class = Decoder
        params = {
            "latent_size": config["latent_size"],
            "dims": config["layer_dimensions"],
            "dropout": config["layers_with_dropout"],
            "dropout_prob": config["dropout_prob"],
            "norm_layers": config["layers_with_norm"],
            "latent_in": config["layer_latent_in"],
            "weight_norm": config["weight_norm"],
            "xyz_in_all": config["xyz_in_all"],
            "latent_dropout": config["latent_dropout"],
            "activation": config["activation"],
            "final_activation": config["final_activation"],
            "concat_latent_input": config["concat_latent_input"],
            "n_objects": config["objects_per_decoder"],
            "progressive_add_depth": config["progressive_add_depth"],
            "layer_split": config["layer_split"],
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model_class(**params)
    saved_model_state = torch.load(path_model_state)
    model.load_state_dict(saved_model_state["model"])
    model = model.cuda()
    model.eval()
    return model


def clip_bone_end(mesh_orig, bone_type="femur", max_z_rel_x=0.7):
    if bone_type not in ["femur", "tibia"]:
        return mesh_orig

    # Always center (both approaches can use this)
    centroid = np.mean(mesh_orig.points, axis=0)
    mesh_orig.point_coords -= centroid

    # Calculate dimensions
    bounds = mesh_orig.bounds
    x_dimension = bounds[1] - bounds[0]
    y_dimension = bounds[3] - bounds[2]
    z_dimension = bounds[5] - bounds[4]

    # Unified offset calculation (both approaches use same logic)
    if z_dimension > max_z_rel_x * y_dimension:
        offset = max_z_rel_x * x_dimension
    else:
        offset = BONE_CLIPPING_FACTOR * z_dimension

    # Calculate clip value based on bone type
    is_femur = bone_type == "femur"
    clip_value = (bounds[4] + offset) if is_femur else (bounds[5] - offset)

    # Clip (same for both approaches)
    mesh_orig.clip("z", value=clip_value, invert=is_femur, inplace=True)

    # Translate back
    mesh_orig.point_coords += centroid

    return mesh_orig


def read_iv(file_path):
    """
    Reads an Inventor .iv file and returns a PyVista PolyData object.

    .iv files typically store geometry in a human-readable format. This function
    parses the "Coordinate3" and "IndexedFaceSet" sections to extract vertex
    coordinates and face connectivity.

    Args:
        file_path (str): The path to the .iv file.

    Returns:
        pyvista.PolyData: A PyVista mesh object created from the .iv file data.

    Raises:
        ValueError: If the "Coordinate3" or "IndexedFaceSet" sections are not
                    found, or if parsing fails.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    coord_start_idx = -1
    coord_end_idx = -1
    face_start_idx = -1
    face_end_idx = -1

    for i, line in enumerate(lines):
        if "Coordinate3" in line and "point" in line:
            coord_start_idx = i + 1
        elif coord_start_idx != -1 and "]" in line and coord_end_idx == -1:
            coord_end_idx = i
        elif "IndexedFaceSet" in line and "coordIndex" in line:
            face_start_idx = i + 1
        elif face_start_idx != -1 and "]" in line and face_end_idx == -1:
            face_end_idx = i

    if coord_start_idx == -1 or coord_end_idx == -1:
        raise ValueError("Could not find Coordinate3 point data in .iv file.")
    if face_start_idx == -1 or face_end_idx == -1:
        raise ValueError("Could not find IndexedFaceSet coordIndex data in .iv file.")

    # Extract coordinates
    points_str = "".join(lines[coord_start_idx:coord_end_idx])
    points_str = points_str.replace("\n", "").replace("[", "").replace("]", "")
    points_list = [float(p) for p in re.split(r"[\s,]+", points_str) if p]
    points = np.array(points_list).reshape(-1, 3)

    # Extract face indices
    faces_str = "".join(lines[face_start_idx:face_end_idx])
    faces_str = faces_str.replace("\n", "").replace("[", "").replace("]", "")
    # Split by comma, then handle potential extra spaces and filter out -1 (end of face marker)
    faces_list_str = [f.strip() for f in faces_str.split(",") if f.strip()]

    faces_connectivity = []
    current_face = []
    count = 0
    for val_str in faces_list_str:
        if val_str == "-1":
            if current_face:  # ensure current_face is not empty
                # Prepend the number of points in the face
                faces_connectivity.extend([len(current_face)] + current_face)
                current_face = []
            count = 0  # reset point counter for the new face
        else:
            try:
                current_face.append(int(val_str))
                count += 1
            except ValueError:
                logger.warning("Could not convert '%s' to int. Skipping.", val_str)
                continue  # skip if conversion fails

    # Ensure the last face is added if the file doesn't end with -1 explicitly for the last face index list
    if current_face:
        faces_connectivity.extend([len(current_face)] + current_face)

    mesh = pv.PolyData(points, faces=np.array(faces_connectivity))
    return mesh


def recon_mesh(
    mesh_paths,
    model,
    model_config,
    n_samples_latent_recon=None,
    num_iter=None,
    scale_jointly=None,
    convergence_patience=None,
    verbose=False,
    clip_bone=True,
    clip_bone_max_z_rel_x=0.7,
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
        clip_bone (bool, optional): If True, clips the bone to the max_z_rel_x. Defaults to True.
        clip_bone_max_z_rel_x (float, optional): The max z relative to x to clip the bone. Defaults to 0.7.

    Returns:
        tuple: A tuple containing:
            - latent (list): The optimized latent vector(s) as a list.
            - bone_mesh (pyvista.PolyData): The reconstructed primary mesh (typically bone).
            - cart_mesh (pyvista.PolyData): The reconstructed secondary mesh (typically cartilage).
            - mesh_result (dict): A dictionary containing detailed results from the
              `reconstruct_mesh` call, including the latent vector, meshes, and
              registration parameters.

    Notes:
        Ordering assumptions in ``mesh_result['mesh']``:
        * ``len == 3`` - the third mesh (index 2) is treated as a fibula (tibia case).
        * ``len == 4`` - the third and fourth meshes (indices 2 and 3) are treated as the medial
        and lateral menisci respectively (femur case).

        The current pipeline therefore relies on passing exactly one additional mesh for a tibia
        reconstruction and exactly two additional meshes for a femur reconstruction. Supplying a
        different number or ordering of meshes will break this heuristic. If you anticipate other
        combinations, consider adding an explicit flag or more robust logic.
    """

    if scale_jointly is not None:
        raise ValueError(
            "scale_jointly is deprecated, it is automatically extracted from the model config"
        )

    if clip_bone and model_config["bone"] in ["femur", "tibia"]:
        if mesh_paths[0] is None:
            logger.info("No bone mesh provided, skipping clipping")
        else:
            logger.info("Clipping bone mesh: %s", model_config["bone"])
            # load the mesh, do the clip, save a temp clipped mesh
            temp_bone_mesh_path = mesh_paths[0] + "_temp.vtk"
            orig_bone_mesh_path = mesh_paths[0]
            orig_bone_mesh = Mesh(orig_bone_mesh_path)
            clipped_bone_mesh = clip_bone_end(
                orig_bone_mesh, bone_type=model_config["bone"], max_z_rel_x=clip_bone_max_z_rel_x
            )
            clipped_bone_mesh.save_mesh(temp_bone_mesh_path)
            mesh_paths[0] = temp_bone_mesh_path

    mesh_result = reconstruct_mesh(
        path=mesh_paths,
        decoders=model,
        latent_size=model_config["latent_size"],
        # Fitting parameters:
        num_iterations=model_config["num_iterations_recon"] if num_iter is None else num_iter,
        l2reg=model_config["l2reg_recon"],
        latent_reg_weight=1e-4,
        loss_type="l1",
        lr=model_config["lr_recon"],
        lr_update_factor=model_config["lr_update_factor_recon"],
        n_lr_updates=model_config["n_lr_updates_recon"],
        return_latent=True,
        register_similarity=True,
        scale_jointly=model_config["scale_jointly"],
        scale_all_meshes=True,
        mesh_to_scale=model_config["mesh_to_scale"],
        objects_per_decoder=model_config["objects_per_decoder"],
        batch_size_latent_recon=model_config["batch_size_latent_recon"],
        get_rand_pts=model_config["get_rand_pts_recon"],
        n_pts_random=model_config["n_pts_random_recon"],
        sigma_rand_pts=model_config["sigma_rand_pts_recon"],
        n_samples_latent_recon=n_samples_latent_recon,
        calc_symmetric_chamfer=False,
        calc_assd=False,
        calc_emd=False,
        # convergence=model_config['convergence_type_recon'],
        convergence=10,
        convergence_patience=model_config["convergence_patience_recon"],
        # convergence_patience=model_config['convergence_patience_recon'] if convergence_patience is None else convergence_patience,
        clamp_dist=model_config["clamp_dist_recon"],
        fix_mesh=model_config["fix_mesh_recon"],
        verbose=verbose,
        return_registration_params=True,
    )

    if clip_bone and (model_config["bone"] in ["femur", "tibia"]) and (mesh_paths[0] is not None):
        # delete the temp clipped mesh
        os.remove(temp_bone_mesh_path)

    # get latent
    latent = mesh_result["latent"].detach().cpu().numpy().tolist()

    output_dict = {
        "latent": latent,
        "bone_mesh": mesh_result["mesh"][0],
        "cart_mesh": mesh_result["mesh"][1],
        "mesh_result": mesh_result,
    }

    # Map additional meshes based on bone type from model_config.
    # The NSM decoder returns surfaces in a fixed order per bone:
    #   femur:   [bone, cart, med_meniscus, lat_meniscus]  (4 meshes)
    #   tibia:   [bone, cart, fibula]                      (3 meshes)
    #   patella: [bone, cart]                               (2 meshes)
    bone_type = model_config.get("bone")
    n_meshes = len(mesh_result["mesh"])

    _EXPECTED_EXTRA_MESHES = {
        "femur": {"count": 4, "names": ["med_men_mesh", "lat_men_mesh"]},
        "tibia": {"count": 3, "names": ["fibula_mesh"]},
        "patella": {"count": 2, "names": []},
    }

    if bone_type in _EXPECTED_EXTRA_MESHES:
        expected = _EXPECTED_EXTRA_MESHES[bone_type]
        if n_meshes != expected["count"]:
            raise ValueError(
                f"Expected {expected['count']} meshes for bone type '{bone_type}', "
                f"got {n_meshes}. Check that the correct mesh paths were provided."
            )
        for i, name in enumerate(expected["names"]):
            output_dict[name] = mesh_result["mesh"][2 + i]
    else:
        # Unknown bone type: fall back to count-based heuristic for
        # forward compatibility with new bone types.
        if n_meshes == 3:
            output_dict["fibula_mesh"] = mesh_result["mesh"][2]
        elif n_meshes == 4:
            output_dict["med_men_mesh"] = mesh_result["mesh"][2]
            output_dict["lat_men_mesh"] = mesh_result["mesh"][3]

    return output_dict


def load_preprocess_opensim_ref_mesh(path, z_rel_x, bone):
    """
    Loads a reference mesh from a file and preprocesses it.

    The preprocessing steps include:
    1. Reading the file using Mesh().
    2. Converting coordinate system from OSIM to NSM orientation.
    3. For 'femur' or 'tibia', clipping the mesh using clip_bone_end function.
    4. Scaling the points by 1000 (e.g., meters to millimeters).
    5. Centering the mesh by subtracting the mean of its point coordinates.
    6. Casting point coordinates to `np.float64`.

    Args:
        path (str): Path to the reference mesh file.
        z_rel_x (float): A factor used to determine the Z-clipping range relative
            to the X-axis extent of the mesh. Used for femur and tibia.
        bone (str): The name of the bone ('femur', 'tibia', or other). This affects
            whether Z-clipping is performed.

    Returns:
        tuple: A tuple containing:
            - ref_ (pyvista.PolyData): The loaded and preprocessed reference mesh.
            - mean_ (numpy.ndarray): The mean vector that was subtracted to center
              the mesh (after all transformations).
    """
    # Import the transformation matrix
    from .nsm_fitting import OSIM_TO_NSM_TRANSFORM

    ref_ = Mesh(path)

    # Step 1: Convert coordinate system from OSIM to NSM orientation
    ref_.point_coords = ref_.point_coords @ OSIM_TO_NSM_TRANSFORM

    # Step 2: Clip bone if needed (done in NSM coordinate system)
    if bone in ["femur", "tibia"]:
        logger.info("Clipping %s mesh", bone)
        ref_ = clip_bone_end(ref_, bone_type=bone, max_z_rel_x=z_rel_x)

    # Step 3: Scale from meters to millimeters
    ref_.point_coords = ref_.point_coords * 1000

    # Step 4: Calculate mean and center the mesh
    mean_ = np.mean(ref_.point_coords, axis=0)
    ref_.point_coords = ref_.point_coords - mean_

    # Step 5: Ensure proper data type
    ref_.point_coords = ref_.point_coords.astype(np.float64)

    return ref_, mean_


def load_preprocess_opensim_ref_menisci(med_meniscus_path, lat_meniscus_path):
    """
    Loads and preprocesses medial and lateral meniscus meshes.

    The preprocessing steps include:
    1. Loading both medial and lateral meniscus meshes using Mesh().
    2. Converting coordinate system from OSIM to NSM orientation.
    3. Scaling the points by 1000 (e.g., meters to millimeters).
    4. Combining them temporarily to calculate the overall centroid.
    5. Centering both meshes using the combined centroid.
    6. Casting point coordinates to `np.float64`.

    Args:
        med_meniscus_path (str): Path to the medial meniscus mesh file.
        lat_meniscus_path (str): Path to the lateral meniscus mesh file.

    Returns:
        tuple: A tuple containing:
            - med_meniscus_processed (pyvista.PolyData): The preprocessed medial meniscus mesh.
            - lat_meniscus_processed (pyvista.PolyData): The preprocessed lateral meniscus mesh.
            - mean_ (numpy.ndarray): The mean vector that was subtracted to center
              both meshes (after all transformations).
    """
    # Import the transformation matrix
    from .nsm_fitting import OSIM_TO_NSM_TRANSFORM

    # Load both meniscus meshes
    med_meniscus = Mesh(med_meniscus_path)
    lat_meniscus = Mesh(lat_meniscus_path)

    # Step 1: Convert coordinate system from OSIM to NSM orientation
    med_meniscus.point_coords = med_meniscus.point_coords @ OSIM_TO_NSM_TRANSFORM
    lat_meniscus.point_coords = lat_meniscus.point_coords @ OSIM_TO_NSM_TRANSFORM

    # Step 2: Scale from meters to millimeters
    med_meniscus.point_coords = med_meniscus.point_coords * 1000
    lat_meniscus.point_coords = lat_meniscus.point_coords * 1000

    # Step 3: Create a temporary combined mesh to calculate the overall centroid
    combined_mesh_pv = med_meniscus.mesh + lat_meniscus.mesh
    combined_menisci = Mesh(combined_mesh_pv)

    # Step 4: Calculate the mean from the combined mesh
    mean_ = np.mean(combined_menisci.point_coords, axis=0)

    # Step 5: Apply centering to both individual meshes
    med_meniscus.point_coords = med_meniscus.point_coords - mean_
    med_meniscus.point_coords = med_meniscus.point_coords.astype(np.float64)

    lat_meniscus.point_coords = lat_meniscus.point_coords - mean_
    lat_meniscus.point_coords = lat_meniscus.point_coords.astype(np.float64)

    return med_meniscus, lat_meniscus, mean_


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
    four_by_four[:3, 2] = -1 * femur_acs.is_axis
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
    dict_update_params=None,
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
    with open(path_model_config, "r") as f:
        model_config = json.load(f)

    if dict_update_params is not None:
        for key, value in dict_update_params.items():
            model_config[key] = value

    model = load_model(model_config, path_model_state, model_type="triplanar")

    recon_output = recon_mesh(
        list_paths_meshes,
        model,
        model_config,
        n_samples_latent_recon=n_samples_latent_recon,
        num_iter=num_iter,
        convergence_patience=convergence_patience,
    )

    # Add the model to the mesh_result dictionary within recon_output
    recon_output["mesh_result"]["model"] = model

    # recon_output now contains all necessary information, including the model
    # and meniscus meshes if they were returned by recon_mesh.
    return recon_output
