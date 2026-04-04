"""
Model-building orchestration: OSIM-space meshes → subject-specific OpenSim model.

Extracts the shared model-building logic from comak_1_nsm_fitting.py (lines ~370–1050)
into reusable functions. Both the fitting pipeline and synthetic joint pipeline call
the same code here.

Step functions are pure (no file I/O) — they take data in and return results.
The orchestrator (build_joint_model) handles all saving.
"""

import json
import os
import shutil

import numpy as np
import pyvista as pv
from pymskt.mesh import Mesh

from nsosim.articular_surfaces import (
    create_articular_surfaces,
    create_meniscus_articulating_surface,
    create_prefemoral_fatpad_noboolean,
)
from nsosim.comak_osim_update import update_osim_model
from nsosim.meniscal_ligaments import project_meniscal_attachments_to_tibia
from nsosim.nsm_fitting import interp_ref_to_subject_to_osim
from nsosim.osim_utils import (
    add_contact_force_to_model,
    add_contact_mesh_to_model,
    create_articular_contact_force,
    create_contact_mesh,
)
from nsosim.wrap_surface_fitting.config import (
    DEFAULT_FITTING_CONFIG,
    DEFAULT_SMITH2019_BONES,
)
from nsosim.wrap_surface_fitting.fitting import CylinderFitter, EllipsoidFitter
from nsosim.wrap_surface_fitting.patella import PatellaFitter

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def load_transform(path):
    """Load a 4x4 transform from .npy or alignment .json.

    Alignment JSONs may use either 'linear_transform' (subject) or
    'transform_matrix' (reference) as the key.
    """
    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
        if "linear_transform" in data:
            return np.array(data["linear_transform"])
        elif "transform_matrix" in data:
            return np.array(data["transform_matrix"]).reshape(4, 4)
        else:
            raise ValueError(
                f"JSON at {path} has no 'linear_transform' or 'transform_matrix' key"
            )
    else:
        raise ValueError(f"Unsupported transform format: {path} (use .npy or .json)")


def build_dict_bones_for_interpolation(models, latents, transforms, labeled_bone_dir):
    """Construct a dict_bones structure for interp_ref_to_subject_to_osim.

    This builds the minimal dict_bones needed by the interpolation function,
    without requiring a full fitting pipeline run.

    Parameters
    ----------
    models : dict
        {'femur': model, 'tibia': model, 'patella': model} — loaded NSM models.
    latents : dict
        {'femur': ndarray, 'tibia': ndarray, 'patella': ndarray} — latent vectors.
    transforms : dict
        {'femur': 4x4, 'tibia': 4x4, 'patella': 4x4} — per-bone linear_transforms
        (absolute, NOT relative).
    labeled_bone_dir : str
        Path to directory containing {bone}_labeled.vtk files.

    Returns
    -------
    dict
        dict_bones-compatible structure.
    """
    dict_bones = {}
    for bone_name in ["femur", "tibia", "patella"]:
        latent = latents[bone_name]
        if latent.ndim == 1:
            latent = latent[np.newaxis, :]

        dict_bones[bone_name] = {
            "subject": {
                "recon_dict": {
                    "model": models[bone_name],
                    "icp_transform": transforms[bone_name],
                    "scale": 1,
                    "center": np.zeros(3),
                },
                "recon_latent": latent,
            },
            "wrap": {
                "path_labeled_bone": os.path.join(
                    labeled_bone_dir, f"{bone_name}_labeled.vtk"
                ),
            },
        }

    return dict_bones


def save_mesh_as_obj(mesh, filepath):
    """Save a pyvista/pymskt mesh as minimal OBJ (vertices + triangular faces).

    OBJ preserves indexed vertices exactly — no deduplication — avoiding the
    SimTK STL vertex-merging bug that changes point counts.
    """
    polydata = mesh.mesh if hasattr(mesh, "mesh") else mesh
    with open(filepath, "w") as f:
        for pt in polydata.points:
            f.write(f"v {pt[0]:.15g} {pt[1]:.15g} {pt[2]:.15g}\n")
        faces = polydata.faces.reshape(-1, 4)  # each row: [3, i, j, k]
        for face in faces:
            f.write(f"f {face[1]+1} {face[2]+1} {face[3]+1}\n")  # OBJ is 1-indexed


# ---------------------------------------------------------------------------
# Per-bone step functions
# ---------------------------------------------------------------------------


def interpolate_bone_ligaments(
    bone_name,
    labeled_mesh_path,
    dict_lig_musc_attach_params,
    dict_bones,
    fem_ref_center,
    folder_ref_recons,
    surface_idx=0,
):
    """Interpolate labeled mesh and ligament attachment points from reference to subject.

    Source: comak_1_nsm_fitting.py lines 408–449 (femur), 541–582 (tibia), 825–856 (patella).

    Parameters
    ----------
    bone_name : str
        Bone name ('femur', 'tibia', 'patella') used to match parent_frame in ligament dicts.
    labeled_mesh_path : str
        Path to the labeled bone VTK file with wrap surface classifications.
    dict_lig_musc_attach_params : dict
        Ligament/muscle attachment parameters dict. Not modified in-place.
    dict_bones : dict
        dict_bones-compatible structure with recon_dict + recon_latent per bone.
    fem_ref_center : np.ndarray
        Femur reference center from ref_femur_alignment.json['mean_orig'].
    folder_ref_recons : str
        Path to folder containing reference reconstruction data.
    surface_idx : int
        Surface index for NSM interpolation (0=bone).

    Returns
    -------
    labeled_mesh : Mesh
        Labeled mesh with updated (interpolated) point coordinates.
    labeled_mesh_points : np.ndarray
        Copy of the updated labeled mesh point coordinates.
    lig_xyz_points_updated : np.ndarray
        Updated ligament attachment point coordinates (n_lig_pts, 3).
    list_lig_name_pt_idx : list of [str, int]
        List of [force_name, point_index] pairs identifying which ligament points were updated.
    """
    labeled_mesh = Mesh(labeled_mesh_path)

    # Collect ligament attachment points that belong to this bone
    list_lig_musc_xyz_to_update = []
    list_lig_musc_name_pt_idx = []
    for key, dict_ in dict_lig_musc_attach_params.items():
        for pt_idx, point_dict in enumerate(dict_["points"]):
            if bone_name in point_dict["parent_frame"]:
                list_lig_musc_xyz_to_update.append(point_dict["xyz_mesh"])
                list_lig_musc_name_pt_idx.append([key, pt_idx])

    lig_xyz_points = np.array(list_lig_musc_xyz_to_update)

    # Append ligament points to labeled mesh for joint interpolation
    n_orig_pts = labeled_mesh.point_coords.shape[0]
    labeled_mesh_copy = labeled_mesh.copy()
    labeled_mesh_copy.point_coords = np.concatenate(
        [labeled_mesh_copy.point_coords, lig_xyz_points], axis=0
    )

    # Interpolate all points (mesh + ligament) together
    interpolated_pts_osim = interp_ref_to_subject_to_osim(
        ref_mesh=labeled_mesh_copy,
        surface_name=bone_name,
        ref_center=fem_ref_center,
        dict_bones=dict_bones,
        folder_nsm_files=folder_ref_recons,
        surface_idx=surface_idx,
    )

    # Split back into mesh points and ligament points
    labeled_mesh_points_updated = interpolated_pts_osim[:n_orig_pts, :]
    lig_xyz_points_updated = interpolated_pts_osim[n_orig_pts:, :]

    # Update labeled mesh with interpolated coordinates
    labeled_mesh.point_coords = labeled_mesh_points_updated
    labeled_mesh_points = labeled_mesh.point_coords.copy()

    return labeled_mesh, labeled_mesh_points, lig_xyz_points_updated, list_lig_musc_name_pt_idx


def fit_bone_wrap_surfaces(
    bone_name,
    labeled_mesh,
    labeled_mesh_points,
    wrap_surface_spec=None,
    fitter_configs=None,
    patella_wrap_dimension_scale=0.9,
):
    """Fit wrap surfaces to a labeled bone mesh.

    Source: comak_1_nsm_fitting.py lines 452–502 (femur), 584–634 (tibia), 867–883 (patella).

    For femur/tibia: iterates over DEFAULT_SMITH2019_BONES config, fitting ellipsoids
    and cylinders using SDF-based optimization.
    For patella: uses PatellaFitter (specialized ellipsoid fitting).

    Parameters
    ----------
    bone_name : str
        'femur', 'tibia', or 'patella'.
    labeled_mesh : Mesh
        Labeled bone mesh with wrap surface classification arrays.
    labeled_mesh_points : np.ndarray
        Point coordinates of the labeled mesh (possibly patella-centered).
    wrap_surface_spec : dict or None
        Wrap surface specification from DEFAULT_SMITH2019_BONES[bone_name]['wrap_surfaces'].
        If None, uses DEFAULT_SMITH2019_BONES[bone_name]['wrap_surfaces'].
    fitter_configs : dict or None
        Fitter configurations. If None, uses DEFAULT_FITTING_CONFIG.
    patella_wrap_dimension_scale : float
        Scale factor for patella wrap surface dimensions (default 0.9 = 10% reduction).

    Returns
    -------
    dict
        Fitted wrap parameters dict for this bone, structured as:
        {body_name: {surface_type: {wrap_name: wrap_surface}}}
    """
    if fitter_configs is None:
        fitter_configs = DEFAULT_FITTING_CONFIG

    ellipsoid_constructor = fitter_configs["ellipsoid"]["constructor"]
    ellipsoid_fit = fitter_configs["ellipsoid"]["fit"]
    cylinder_constructor = fitter_configs["cylinder"]["constructor"]
    cylinder_fit = fitter_configs["cylinder"]["fit"]

    fitted = {}

    if bone_name == "patella":
        # Patella uses specialized PatellaFitter
        fitted["patella_r"] = {"ellipsoid": {}}
        patella_fitter = PatellaFitter(patella_mesh=labeled_mesh)
        patella_fitter.fit()
        wrap_params = patella_fitter.wrap_params
        wrap_params.name = "PatTen_r"
        wrap_params.body = "patella_r"
        wrap_params.dimensions = wrap_params.dimensions * patella_wrap_dimension_scale
        fitted["patella_r"]["ellipsoid"]["PatTen_r"] = wrap_params
        return fitted

    # Femur / tibia: iterate over wrap surface spec
    if wrap_surface_spec is None:
        wrap_surface_spec = DEFAULT_SMITH2019_BONES[bone_name]["wrap_surfaces"]

    for body_name, body_data in wrap_surface_spec.items():
        fitted[body_name] = {}
        for surface_type, surface_list in body_data.items():
            fitted[body_name][surface_type] = {}
            if surface_type == "ellipsoid":
                for wrap_name in surface_list:
                    labels = labeled_mesh[f"{wrap_name}_binary"].copy()
                    sdf = labeled_mesh[f"{wrap_name}_sdf"].copy()

                    fitter = EllipsoidFitter(**ellipsoid_constructor)
                    fitter.fit(
                        points=labeled_mesh_points,
                        labels=labels,
                        sdf=sdf,
                        mesh=labeled_mesh,
                        surface_name=wrap_name,
                        **ellipsoid_fit,
                    )
                    wrap_params = fitter.wrap_params
                    wrap_params.name = wrap_name
                    wrap_params.body = body_name
                    fitted[body_name][surface_type][wrap_name] = wrap_params

            elif surface_type == "cylinder":
                for wrap_name in surface_list:
                    labels = labeled_mesh[f"{wrap_name}_binary"].copy()
                    sdf = labeled_mesh[f"{wrap_name}_sdf"].copy()
                    near_surface_bool = labeled_mesh[f"{wrap_name}_near_surface"].copy()

                    near_surface_points = labeled_mesh_points[near_surface_bool == 1]
                    near_surface_labels = labels[near_surface_bool == 1]
                    near_surface_sdf = sdf[near_surface_bool == 1]

                    fitter = CylinderFitter(**cylinder_constructor)
                    fitter.fit(
                        points=near_surface_points,
                        labels=near_surface_labels,
                        sdf=near_surface_sdf,
                        mesh=labeled_mesh,
                        surface_name=wrap_name,
                        near_surface_points=near_surface_points,
                        **cylinder_fit,
                    )
                    wrap_params = fitter.wrap_params
                    wrap_params.name = wrap_name
                    wrap_params.body = body_name
                    fitted[body_name][surface_type][wrap_name] = wrap_params

    return fitted


# ---------------------------------------------------------------------------
# Cross-bone step functions
# ---------------------------------------------------------------------------


def interpolate_meniscus_ligaments(
    dict_lig_musc_attach_params,
    dict_bones,
    fem_ref_center,
    folder_ref_recons,
):
    """Interpolate meniscal ligament attachment points using the femur NSM model.

    Source: comak_1_nsm_fitting.py lines 692–719.

    Meniscus ligaments are interpolated via the femur model with surface_idx=2 (medial)
    or surface_idx=3 (lateral).

    Parameters
    ----------
    dict_lig_musc_attach_params : dict
        Ligament/muscle attachment parameters. Modified in-place with 'xyz_mesh_updated'.
    dict_bones : dict
        dict_bones structure (needs femur entry with recon_dict + recon_latent).
    fem_ref_center : np.ndarray
        Femur reference center.
    folder_ref_recons : str
        Path to reference reconstruction data.
    """
    for side_idx, men_side in enumerate(["medial", "lateral"]):
        list_lig_musc_xyz_to_update = []
        list_lig_musc_name_pt_idx = []
        for key, dict_ in dict_lig_musc_attach_params.items():
            for pt_idx, point_dict in enumerate(dict_["points"]):
                if f"meniscus_{men_side}" in point_dict["parent_frame"]:
                    list_lig_musc_xyz_to_update.append(point_dict["xyz_mesh"])
                    list_lig_musc_name_pt_idx.append([key, pt_idx])

        lig_xyz_points = np.array(list_lig_musc_xyz_to_update)
        men_ligs_xyz = pv.PolyData(lig_xyz_points)

        men_ligs_xyz_interpolated = interp_ref_to_subject_to_osim(
            ref_mesh=men_ligs_xyz,
            surface_name="femur",
            ref_center=fem_ref_center,
            dict_bones=dict_bones,
            folder_nsm_files=folder_ref_recons,
            surface_idx=2 + side_idx,  # 0=bone, 1=cart, 2=med_men, 3=lat_men
        )

        for idx, (force_name, pt_idx) in enumerate(list_lig_musc_name_pt_idx):
            new_pt_xyz = men_ligs_xyz_interpolated[idx, :]
            dict_lig_musc_attach_params[force_name]["points"][pt_idx][
                "xyz_mesh_updated"
            ] = new_pt_xyz


def update_coronary_ligament_tibia_attachments(
    dict_lig_musc_attach_params,
    tib_mesh_osim,
    lig_attachment_key="xyz_mesh_updated",
):
    """Project coronary ligament tibia attachments onto tibia surface.

    Source: comak_1_nsm_fitting.py lines 721–767.

    For each coronary ligament, reads the meniscus attachment point and finds the
    closest point on the tibia bone surface, then updates the tibia attachment.

    Note: The original code used lig_attachment_key='xyz_mesh' (reference positions),
    making the entire block dead code since update_osim_model reads 'xyz_mesh_updated'.
    This version defaults to 'xyz_mesh_updated' to fix that bug.

    Parameters
    ----------
    dict_lig_musc_attach_params : dict
        Ligament/muscle attachment parameters. Modified in-place.
    tib_mesh_osim : Mesh
        Subject tibia mesh in OSIM space.
    lig_attachment_key : str
        Key to read/write attachment positions. Default 'xyz_mesh_updated'.
    """
    cor_men_ligs = [
        "meniscus_lateral_COR1",
        "meniscus_lateral_COR2",
        "meniscus_lateral_COR3",
        "meniscus_medial_COR1",
        "meniscus_medial_COR2",
        "meniscus_medial_COR3",
    ]

    for cor_men_lig in cor_men_ligs:
        lig_dict = dict_lig_musc_attach_params[cor_men_lig]
        tibia_point = lig_dict["points"][0]
        men_point = lig_dict["points"][1]

        assert (
            tibia_point["parent_frame"] == "tibia_proximal_r"
        ), "tibia point parent frame is not tibia_proximal_r"
        assert (
            "meniscus" in men_point["parent_frame"]
        ), "meniscus point parent frame is not meniscus (lateral or medial)"

        men_point_xyz = men_point[lig_attachment_key]

        tibia_point_index = tib_mesh_osim.find_closest_point(men_point_xyz)
        tib_point_xyz = tib_mesh_osim.points[tibia_point_index, :]

        lig_dict["points"][0][lig_attachment_key] = tib_point_xyz
        dict_lig_musc_attach_params[cor_men_lig] = lig_dict


def center_patella_meshes(pat_mesh, pat_articular, pat_cart_mesh=None):
    """Center patella meshes by subtracting the bone mesh centroid.

    Source: comak_1_nsm_fitting.py lines 794–822.

    Parameters
    ----------
    pat_mesh : Mesh
        Patella bone mesh in OSIM space.
    pat_articular : Mesh
        Patella articular surface mesh.
    pat_cart_mesh : Mesh or None
        Patella cartilage mesh (optional).

    Returns
    -------
    pat_mesh_centered : Mesh
        Centered patella bone mesh.
    pat_articular_centered : Mesh
        Centered patella articular surface mesh.
    pat_cart_centered : Mesh or None
        Centered patella cartilage mesh (None if input was None).
    mean_patella : np.ndarray
        The centroid that was subtracted (for saving as offset).
    """
    if not isinstance(pat_articular, Mesh):
        pat_articular = Mesh(pat_articular)
    if not isinstance(pat_mesh, Mesh):
        pat_mesh = Mesh(pat_mesh)

    mean_patella = np.mean(pat_mesh.point_coords, axis=0)

    pat_mesh_centered = pat_mesh.copy()
    pat_articular_centered = pat_articular.copy()

    pat_mesh_centered.point_coords -= mean_patella
    pat_articular_centered.point_coords -= mean_patella

    pat_cart_centered = None
    if pat_cart_mesh is not None:
        pat_cart_centered = pat_cart_mesh.copy()
        pat_cart_centered.point_coords -= mean_patella

    return pat_mesh_centered, pat_articular_centered, pat_cart_centered, mean_patella


# ---------------------------------------------------------------------------
# I/O and finalization
# ---------------------------------------------------------------------------

# Geometry files to copy to the OpenSim Geometry/ folder
DEFAULT_GEOMETRY_FILES = {
    "femur": [
        "femur_nsm_recon_osim.stl",
        "femur_articular_surface_osim.stl",
        "femur_articular_surface_osim.obj",
        "femur_prefemoral_fat_pad.stl",
        "lat_men_osim.stl",
        "lat_men_upper_art_surf_osim.stl",
        "lat_men_lower_art_surf_osim.stl",
        "med_men_osim.stl",
        "med_men_upper_art_surf_osim.stl",
        "med_men_lower_art_surf_osim.stl",
    ],
    "tibia": [
        "tibia_nsm_recon_osim.stl",
        "tibia_articular_surface_osim.stl",
        "tibia_articular_surface_osim.obj",
    ],
    "patella": [
        "patella_nsm_recon_osim.stl",
        "patella_articular_surface_osim.stl",
        "patella_articular_surface_osim.obj",
    ],
}


def save_geometry_files(folder_save_bones, path_save_model, geometry_dict=None):
    """Copy generated geometry files to the OpenSim model's Geometry/ folder.

    Source: comak_1_nsm_fitting.py lines 934–964.

    Parameters
    ----------
    folder_save_bones : str
        Root folder containing per-bone subfolders with generated meshes.
    path_save_model : str
        Path to the OpenSim model directory (Geometry/ subfolder will be created).
    geometry_dict : dict or None
        {bone_name: [filename, ...]} mapping. If None, uses DEFAULT_GEOMETRY_FILES.
    """
    if geometry_dict is None:
        geometry_dict = DEFAULT_GEOMETRY_FILES

    geometry_dir = os.path.join(path_save_model, "Geometry")
    os.makedirs(geometry_dir, exist_ok=True)

    for bone, geom_list in geometry_dict.items():
        for filename in geom_list:
            src = os.path.join(folder_save_bones, bone, filename)
            dst = os.path.join(geometry_dir, filename)
            shutil.copy(src, dst)


def finalize_osim_model(
    osim_model,
    fitted_wrap_parameters,
    dict_lig_musc_attach_params,
    tib_mesh_osim,
    mean_patella,
    model_name,
    path_save,
    lig_musc_xyz_key="xyz_mesh_updated",
    lig_normal_shift=5e-4,
    dict_lig_stiffness=None,
    dict_joints_coords_to_update=None,
    fatpad_elastic_modulus=4e6,
    fatpad_poissons_ratio=0.45,
    fatpad_thickness=0.01,
    fatpad_min_proximity=0.0,
    fatpad_max_proximity=0.015,
    project_meniscal_to_tibia=False,
):
    """Update OpenSim model with subject-specific data and save.

    Source: comak_1_nsm_fitting.py lines 966–1051.

    Parameters
    ----------
    osim_model : osim.Model
        Loaded OpenSim model.
    fitted_wrap_parameters : dict
        Fitted wrap parameters for all bones.
    dict_lig_musc_attach_params : dict
        Ligament/muscle attachment parameters with 'xyz_mesh_updated' entries.
    tib_mesh_osim : Mesh
        Subject tibia mesh in OSIM space.
    mean_patella : np.ndarray
        Patella centroid offset.
    model_name : str
        Name for the model.
    path_save : str
        Directory to save the .osim file.
    lig_musc_xyz_key : str
        Key for ligament xyz data in attachment params.
    lig_normal_shift : float
        Normal vector shift for ligament attachments (meters).
    dict_lig_stiffness : dict or None
        Ligament stiffness update dict. If None, no stiffness update.
    dict_joints_coords_to_update : dict or None
        Joint coordinates to update.
    fatpad_elastic_modulus, fatpad_poissons_ratio, fatpad_thickness : float
        Fat pad material properties.
    fatpad_min_proximity, fatpad_max_proximity : float
        Fat pad contact proximity bounds (meters).
    project_meniscal_to_tibia : bool
        Whether to project meniscal ligament tibia attachments onto tibia surface.

    Returns
    -------
    str
        Path to saved .osim file.
    """
    import opensim as osim

    # Optionally project meniscal ligament tibia attachments
    if project_meniscal_to_tibia:
        print("Projecting meniscal ligament tibia attachments onto tibia surface...")
        projection_results = project_meniscal_attachments_to_tibia(
            dict_lig_mus_attach=dict_lig_musc_attach_params,
            tibia_mesh=tib_mesh_osim,
        )
        for lig_name, result in projection_results.items():
            print(f'  {lig_name}: method={result["method"]}, distance={result["distance"]:.4f}m')

    update_osim_model(
        model=osim_model,
        dict_wrap_objects=fitted_wrap_parameters,
        dict_lig_mus_attach=dict_lig_musc_attach_params,
        tibia_mesh_osim=tib_mesh_osim,
        mean_patella=mean_patella,
        lig_musc_xyz_key=lig_musc_xyz_key,
        lig_musc_normal_vector_shift=lig_normal_shift,
        dict_ligament_stiffness_update=dict_lig_stiffness,
        dict_joints_coords_to_update=dict_joints_coords_to_update,
    )

    # Add femur bone mesh for prefemoral fat pad contact
    femur_bone_mesh = create_contact_mesh(
        name="femur_bone_mesh",
        parent_frame="/bodyset/femur_distal_r",
        mesh_file="femur_prefemoral_fat_pad.stl",
        elastic_modulus=fatpad_elastic_modulus,
        poissons_ratio=fatpad_poissons_ratio,
        thickness=fatpad_thickness,
        use_variable_thickness=False,
        mesh_back_file="femur_prefemoral_fat_pad.stl",
        min_thickness=0.0005,
        max_thickness=0.005,
        scale_factors=(1.0, 1.0, 1.0),
    )
    add_contact_mesh_to_model(osim_model, femur_bone_mesh)

    # Add prefemoral fat pad contact force
    prefemoral_fat_pad_contact = create_articular_contact_force(
        name="prefemoral_fat_pad_contact",
        socket_target_mesh="/contactgeometryset/femur_bone_mesh",
        socket_casting_mesh="/contactgeometryset/patella_cartilage",
        min_proximity=fatpad_min_proximity,
        max_proximity=fatpad_max_proximity,
        elastic_foundation_formulation="nonlinear",
        use_lumped_contact_model=True,
        applies_force=True,
    )

    force_path = f"/forceset/{prefemoral_fat_pad_contact.getName()}"
    if not osim_model.hasComponent(force_path):
        add_contact_force_to_model(osim_model, prefemoral_fat_pad_contact)

    osim_model.setName(model_name)

    path_save_model = os.path.join(path_save, f"{model_name}.osim")
    osim_model.finalizeConnections()
    osim_model.printToXML(path_save_model)

    return path_save_model


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _apply_ligament_updates(
    dict_lig_musc_attach_params, lig_xyz_points_updated, list_lig_name_pt_idx
):
    """Write interpolated ligament positions back into the attachment params dict."""
    for idx, (force_name, pt_idx) in enumerate(list_lig_name_pt_idx):
        new_pt_xyz = lig_xyz_points_updated[idx, :]
        dict_lig_musc_attach_params[force_name]["points"][pt_idx]["xyz_mesh_updated"] = new_pt_xyz


def _extract_meniscus_centers(tibia_labeled_mesh, tibia_labeled_mesh_points):
    """Extract medial and lateral meniscus center points from labeled tibia mesh."""
    med_labels = tibia_labeled_mesh["med_meniscus_center_binary"].copy()
    med_center = tibia_labeled_mesh_points[med_labels == 1].mean(axis=0)

    lat_labels = tibia_labeled_mesh["lat_meniscus_center_binary"].copy()
    lat_center = tibia_labeled_mesh_points[lat_labels == 1].mean(axis=0)

    return med_center, lat_center


def _save_bone_intermediates(folder_save_bones, bone_name, **meshes):
    """Save intermediate mesh files for a bone."""
    bone_dir = os.path.join(folder_save_bones, bone_name)
    os.makedirs(bone_dir, exist_ok=True)
    for name, mesh in meshes.items():
        if mesh is not None:
            filepath = os.path.join(bone_dir, name)
            if filepath.endswith(".obj"):
                save_mesh_as_obj(mesh, filepath)
            elif hasattr(mesh, "save_mesh"):
                mesh.save_mesh(filepath)
            else:
                mesh.save(filepath)


def build_joint_model(
    bone_meshes,
    dict_bones,
    ref_data_paths,
    dict_lig_musc_attach_params,
    fem_ref_center,
    save_dir,
    model_name,
    path_base_osim_model,
    config=None,
    project_meniscal_to_tibia=False,
    triangle_density=3_000_000,
):
    """Build a subject-specific OpenSim knee model from OSIM-space meshes.

    This is the main orchestrator. It takes meshes (from fitting OR decoding),
    extracts articular surfaces, interpolates ligament attachments, fits wrap
    surfaces, creates meniscus surfaces and fat pad, and assembles the final
    OpenSim model.

    Parameters
    ----------
    bone_meshes : dict
        OSIM-space meshes::

            {'femur': {'bone': Mesh, 'cart': Mesh, 'med_men': Mesh, 'lat_men': Mesh},
             'tibia': {'bone': Mesh, 'cart': Mesh},
             'patella': {'bone': Mesh, 'cart': Mesh}}

    dict_bones : dict
        dict_bones-compatible structure with recon_dict + recon_latent per bone.
        Used by interp_ref_to_subject_to_osim for ligament interpolation.
    ref_data_paths : dict
        Reference data paths::

            {'folder_ref_recons': str,  # folder with per-bone ref alignment/latent data
             'lig_attach_params_path': str}  # (unused here, params passed directly)

    dict_lig_musc_attach_params : dict
        Ligament/muscle attachment parameters (will be modified in-place).
    fem_ref_center : np.ndarray
        Femur reference center from ref_femur_alignment.json['mean_orig'].
    save_dir : str
        Root directory for saving per-bone intermediate outputs.
    model_name : str
        Name for the OpenSim model.
    path_base_osim_model : str
        Path to the base/template OpenSim model directory to copy.
    config : dict or None
        Configuration overrides. Supported keys:

        - 'triangle_density': int (default 3_000_000)
        - 'fitter_configs': dict (default DEFAULT_FITTING_CONFIG)
        - 'patella_wrap_dimension_scale': float (default 0.9)
        - 'lig_normal_shift': float (default 5e-4)
        - 'dict_lig_stiffness': dict (default None)
        - 'dict_joints_coords_to_update': dict (default None)
        - 'fatpad_elastic_modulus': float (default 4e6)
        - 'fatpad_poissons_ratio': float (default 0.45)
        - 'fatpad_thickness': float (default 0.01)
        - 'fatpad_min_proximity': float (default 0.0)
        - 'fatpad_max_proximity': float (default 0.015)
        - 'fatpad_base_mm': float (default 1.0)
        - 'fatpad_top_mm': float (default 6)
        - 'fatpad_max_distance_to_patella_mm': float (default 25)
        - 'fatpad_resample_clusters_final': int (default 5_000)
        - 'fatpad_ray_cast_length': float (default 10.0)
        - 'fatpad_norm_function': str (default 'log')
        - 'fatpad_final_smooth_iter': int (default 100)
        - 'meniscus_ray_length': float (default 15.0)
        - 'meniscus_n_largest': int (default 1)
        - 'meniscus_smooth_iter': int (default 10)
        - 'meniscus_boundary_smoothing': bool (default False)
        - 'meniscus_radial_percentile': float (default 95.0)

    project_meniscal_to_tibia : bool
        Whether to project meniscal ligament tibia attachments onto tibia surface.
    triangle_density : int
        Triangle density for articular surface extraction (can also be set via config).

    Returns
    -------
    str
        Path to the saved .osim model file.
    """
    import opensim as osim

    if config is None:
        config = {}

    def cfg(key, default):
        return config.get(key, default)

    tri_density = cfg("triangle_density", triangle_density)
    fitter_configs = cfg("fitter_configs", None)
    patella_wrap_dim_scale = cfg("patella_wrap_dimension_scale", 0.9)
    folder_ref_recons = ref_data_paths["folder_ref_recons"]
    folder_save_bones = save_dir

    fitted_wrap_parameters = {}

    # -----------------------------------------------------------------------
    # FEMUR
    # -----------------------------------------------------------------------
    print("=== Femur ===")

    fem_mesh_osim = bone_meshes["femur"]["bone"]
    fem_cart_mesh_osim = bone_meshes["femur"]["cart"]
    fem_med_men_mesh_osim = bone_meshes["femur"]["med_men"]
    fem_lat_men_mesh_osim = bone_meshes["femur"]["lat_men"]

    # Articular surfaces
    print("  Extracting articular surfaces...")
    fem_articular = create_articular_surfaces(
        fem_mesh_osim, fem_cart_mesh_osim, n_largest=1, triangle_density=tri_density
    )

    _save_bone_intermediates(
        folder_save_bones,
        "femur",
        **{
            "femur_nsm_recon_osim.stl": fem_mesh_osim,
            "femur_cartilage_nsm_recon_osim.vtk": fem_cart_mesh_osim,
            "femur_articular_surface_osim.vtk": fem_articular,
            "femur_articular_surface_osim.stl": fem_articular,
            "femur_articular_surface_osim.obj": fem_articular,
        },
    )

    # Ligament interpolation
    print("  Interpolating ligament attachments...")
    fem_labeled_mesh, fem_labeled_points, fem_lig_updated, fem_lig_idx = interpolate_bone_ligaments(
        bone_name="femur",
        labeled_mesh_path=dict_bones["femur"]["wrap"]["path_labeled_bone"],
        dict_lig_musc_attach_params=dict_lig_musc_attach_params,
        dict_bones=dict_bones,
        fem_ref_center=fem_ref_center,
        folder_ref_recons=folder_ref_recons,
    )

    _save_bone_intermediates(
        folder_save_bones,
        "femur",
        **{"femur_labeled_mesh_updated.vtk": fem_labeled_mesh},
    )

    # Wrap surface fitting
    print("  Fitting wrap surfaces...")
    fitted_wrap_parameters["femur"] = fit_bone_wrap_surfaces(
        bone_name="femur",
        labeled_mesh=fem_labeled_mesh,
        labeled_mesh_points=fem_labeled_points,
        fitter_configs=fitter_configs,
    )

    # Apply ligament updates after wrap fitting (matches original order)
    _apply_ligament_updates(dict_lig_musc_attach_params, fem_lig_updated, fem_lig_idx)

    # -----------------------------------------------------------------------
    # TIBIA
    # -----------------------------------------------------------------------
    print("=== Tibia ===")

    tib_mesh_osim = bone_meshes["tibia"]["bone"]
    tib_cart_mesh_osim = bone_meshes["tibia"]["cart"]

    # Articular surfaces
    print("  Extracting articular surfaces...")
    tib_articular = create_articular_surfaces(
        tib_mesh_osim, tib_cart_mesh_osim, n_largest=2, triangle_density=tri_density
    )

    _save_bone_intermediates(
        folder_save_bones,
        "tibia",
        **{
            "tibia_nsm_recon_osim.stl": tib_mesh_osim,
            "tibia_cartilage_nsm_recon_osim.vtk": tib_cart_mesh_osim,
            "tibia_articular_surface_osim.vtk": tib_articular,
            "tibia_articular_surface_osim.stl": tib_articular,
            "tibia_articular_surface_osim.obj": tib_articular,
        },
    )

    # Ligament interpolation
    print("  Interpolating ligament attachments...")
    tib_labeled_mesh, tib_labeled_points, tib_lig_updated, tib_lig_idx = interpolate_bone_ligaments(
        bone_name="tibia",
        labeled_mesh_path=dict_bones["tibia"]["wrap"]["path_labeled_bone"],
        dict_lig_musc_attach_params=dict_lig_musc_attach_params,
        dict_bones=dict_bones,
        fem_ref_center=fem_ref_center,
        folder_ref_recons=folder_ref_recons,
    )

    _save_bone_intermediates(
        folder_save_bones,
        "tibia",
        **{"tibia_labeled_mesh_updated.vtk": tib_labeled_mesh},
    )

    # Wrap surface fitting
    print("  Fitting wrap surfaces...")
    fitted_wrap_parameters["tibia"] = fit_bone_wrap_surfaces(
        bone_name="tibia",
        labeled_mesh=tib_labeled_mesh,
        labeled_mesh_points=tib_labeled_points,
        fitter_configs=fitter_configs,
    )

    # Apply ligament updates
    _apply_ligament_updates(dict_lig_musc_attach_params, tib_lig_updated, tib_lig_idx)

    # Extract meniscus centers from labeled tibia
    med_meniscus_center, lat_meniscus_center = _extract_meniscus_centers(
        tib_labeled_mesh, tib_labeled_points
    )

    # -----------------------------------------------------------------------
    # MENISCUS ARTICULATING SURFACES
    # -----------------------------------------------------------------------
    print("=== Meniscus Articulating Surfaces ===")

    meniscus_kwargs = dict(
        upper_articulating_bone_mesh=fem_mesh_osim,
        lower_articulating_bone_mesh=tib_mesh_osim,
        ray_length=cfg("meniscus_ray_length", 15.0),
        n_largest=cfg("meniscus_n_largest", 1),
        smooth_iter=cfg("meniscus_smooth_iter", 10),
        boundary_smoothing=cfg("meniscus_boundary_smoothing", False),
        radial_percentile=cfg("meniscus_radial_percentile", 95.0),
    )

    med_upper, med_lower = create_meniscus_articulating_surface(
        meniscus_mesh=fem_med_men_mesh_osim,
        meniscus_center=med_meniscus_center,
        theta_offset=np.pi,
        **meniscus_kwargs,
    )

    lat_upper, lat_lower = create_meniscus_articulating_surface(
        meniscus_mesh=fem_lat_men_mesh_osim,
        meniscus_center=lat_meniscus_center,
        theta_offset=0.0,
        **meniscus_kwargs,
    )

    # Save meniscus meshes
    for suffix in ["vtk", "stl"]:
        _save_bone_intermediates(
            folder_save_bones,
            "femur",
            **{
                f"lat_men_osim.{suffix}": fem_lat_men_mesh_osim,
                f"med_men_osim.{suffix}": fem_med_men_mesh_osim,
                f"lat_men_upper_art_surf_osim.{suffix}": lat_upper,
                f"lat_men_lower_art_surf_osim.{suffix}": lat_lower,
                f"med_men_upper_art_surf_osim.{suffix}": med_upper,
                f"med_men_lower_art_surf_osim.{suffix}": med_lower,
            },
        )

    # -----------------------------------------------------------------------
    # MENISCUS LIGAMENT INTERPOLATION
    # -----------------------------------------------------------------------
    print("=== Meniscus Ligament Interpolation ===")

    interpolate_meniscus_ligaments(
        dict_lig_musc_attach_params=dict_lig_musc_attach_params,
        dict_bones=dict_bones,
        fem_ref_center=fem_ref_center,
        folder_ref_recons=folder_ref_recons,
    )

    # -----------------------------------------------------------------------
    # CORONARY LIGAMENT TIBIA ATTACHMENTS
    # -----------------------------------------------------------------------
    print("=== Coronary Ligament Tibia Attachments ===")

    update_coronary_ligament_tibia_attachments(
        dict_lig_musc_attach_params=dict_lig_musc_attach_params,
        tib_mesh_osim=tib_mesh_osim,
        lig_attachment_key="xyz_mesh_updated",
    )

    # -----------------------------------------------------------------------
    # PATELLA
    # -----------------------------------------------------------------------
    print("=== Patella ===")

    pat_mesh_osim = bone_meshes["patella"]["bone"]
    pat_cart_mesh_osim = bone_meshes["patella"]["cart"]

    # Articular surfaces
    print("  Extracting articular surfaces...")
    pat_articular = create_articular_surfaces(
        pat_mesh_osim, pat_cart_mesh_osim, n_largest=1, triangle_density=tri_density
    )

    # Center patella
    print("  Centering patella...")
    pat_mesh_centered, pat_articular_centered, _, mean_patella = center_patella_meshes(
        pat_mesh_osim, pat_articular
    )

    _save_bone_intermediates(
        folder_save_bones,
        "patella",
        **{
            "patella_offset.json": None,  # handled separately below
            "patella_nsm_recon_osim.stl": pat_mesh_centered,
            "patella_articular_surface_osim.vtk": pat_articular_centered,
            "patella_articular_surface_osim.stl": pat_articular_centered,
            "patella_articular_surface_osim.obj": pat_articular_centered,
            "patella_nsm_recon_osim_original_position.vtk": pat_mesh_osim,
            "patella_cartilage_nsm_recon_osim_original_position.vtk": pat_cart_mesh_osim,
            "patella_articular_surface_osim_original_position.vtk": pat_articular,
        },
    )

    # Save patella offset JSON
    patella_dir = os.path.join(folder_save_bones, "patella")
    os.makedirs(patella_dir, exist_ok=True)
    with open(os.path.join(patella_dir, "patella_offset.json"), "w") as f:
        json.dump({"mean_patella (m)": mean_patella.tolist()}, f)

    # Ligament interpolation (then apply patella centering offset)
    print("  Interpolating ligament attachments...")
    pat_labeled_mesh, pat_labeled_points, pat_lig_updated, pat_lig_idx = interpolate_bone_ligaments(
        bone_name="patella",
        labeled_mesh_path=dict_bones["patella"]["wrap"]["path_labeled_bone"],
        dict_lig_musc_attach_params=dict_lig_musc_attach_params,
        dict_bones=dict_bones,
        fem_ref_center=fem_ref_center,
        folder_ref_recons=folder_ref_recons,
    )

    # Apply patella centering BEFORE wrap fitting (matches original line 859)
    pat_labeled_points -= mean_patella
    pat_labeled_mesh.point_coords = pat_labeled_points
    pat_lig_updated = pat_lig_updated - mean_patella

    _save_bone_intermediates(
        folder_save_bones,
        "patella",
        **{"patella_labeled_mesh_updated.vtk": pat_labeled_mesh},
    )

    # Wrap surface fitting (on centered mesh)
    print("  Fitting wrap surfaces...")
    fitted_wrap_parameters["patella"] = fit_bone_wrap_surfaces(
        bone_name="patella",
        labeled_mesh=pat_labeled_mesh,
        labeled_mesh_points=pat_labeled_points,
        fitter_configs=fitter_configs,
        patella_wrap_dimension_scale=patella_wrap_dim_scale,
    )

    # Apply ligament updates (with centering already applied)
    _apply_ligament_updates(dict_lig_musc_attach_params, pat_lig_updated, pat_lig_idx)

    # -----------------------------------------------------------------------
    # PREFEMORAL FAT PAD
    # -----------------------------------------------------------------------
    print("=== Prefemoral Fat Pad ===")

    fatpad_mesh = create_prefemoral_fatpad_noboolean(
        femur_bone_mesh=fem_mesh_osim,
        femur_cart_mesh=fem_cart_mesh_osim,
        patella_bone_mesh=pat_mesh_osim,
        patella_cart_mesh=pat_cart_mesh_osim,
        base_mm=cfg("fatpad_base_mm", 1.0),
        top_mm=cfg("fatpad_top_mm", 6),
        max_distance_to_patella_mm=cfg("fatpad_max_distance_to_patella_mm", 25),
        resample_clusters_final=cfg("fatpad_resample_clusters_final", 5_000),
        units="m",
        ray_cast_length=cfg("fatpad_ray_cast_length", 10.0),
        norm_function=cfg("fatpad_norm_function", "log"),
        final_smooth_iter=cfg("fatpad_final_smooth_iter", 100),
    )

    fatpad_path = os.path.join(folder_save_bones, "femur", "femur_prefemoral_fat_pad.stl")
    fatpad_mesh.save(fatpad_path)

    # -----------------------------------------------------------------------
    # OPENSIM MODEL ASSEMBLY
    # -----------------------------------------------------------------------
    print("=== OpenSim Model Assembly ===")

    # Copy template model
    path_save_model = os.path.join(save_dir, model_name)
    if not os.path.exists(path_save_model):
        shutil.copytree(path_base_osim_model, path_save_model)

    # Copy geometry files
    save_geometry_files(folder_save_bones, path_save_model)

    # Find the .osim file in the template
    osim_files = [f for f in os.listdir(path_base_osim_model) if f.endswith(".osim")]
    if len(osim_files) != 1:
        raise ValueError(
            f"Expected exactly 1 .osim file in {path_base_osim_model}, found {len(osim_files)}"
        )
    template_osim_name = osim_files[0]
    path_osim = os.path.join(path_save_model, template_osim_name)
    osim_model = osim.Model(path_osim)

    # Finalize
    path_saved = finalize_osim_model(
        osim_model=osim_model,
        fitted_wrap_parameters=fitted_wrap_parameters,
        dict_lig_musc_attach_params=dict_lig_musc_attach_params,
        tib_mesh_osim=tib_mesh_osim,
        mean_patella=mean_patella,
        model_name=model_name,
        path_save=path_save_model,
        lig_musc_xyz_key="xyz_mesh_updated",
        lig_normal_shift=cfg("lig_normal_shift", 5e-4),
        dict_lig_stiffness=cfg("dict_lig_stiffness", None),
        dict_joints_coords_to_update=cfg("dict_joints_coords_to_update", None),
        fatpad_elastic_modulus=cfg("fatpad_elastic_modulus", 4e6),
        fatpad_poissons_ratio=cfg("fatpad_poissons_ratio", 0.45),
        fatpad_thickness=cfg("fatpad_thickness", 0.01),
        fatpad_min_proximity=cfg("fatpad_min_proximity", 0.0),
        fatpad_max_proximity=cfg("fatpad_max_proximity", 0.015),
        project_meniscal_to_tibia=project_meniscal_to_tibia,
    )

    print(f"=== Model saved: {path_saved} ===")
    return path_saved
