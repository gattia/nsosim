import numpy as np

from nsosim.osim_utils import (
    express_point_in_frame,
    update_body_geometry_meshfile,
    update_contact_mesh_files,
    update_joint_default_values,
    update_ligament_stiffness,
    update_model_attachments_slacks,
    update_wrap_cylinder,
    update_wrap_ellipsoid,
)

DICT_CONTACT_MESHFILES_UPDATE = {
    "femur_cartilage": {
        "mesh_file": "femur_articular_surface_osim.stl",
        "mesh_back_file": "femur_nsm_recon_osim.stl",
    },
    "tibia_cartilage": {
        "mesh_file": "tibia_articular_surface_osim.stl",
        "mesh_back_file": "tibia_nsm_recon_osim.stl",
    },
    "patella_cartilage": {
        "mesh_file": "patella_articular_surface_osim.stl",
        "mesh_back_file": "patella_nsm_recon_osim.stl",
    },
    "meniscus_medial_superior": {
        "mesh_file": "med_men_upper_art_surf_osim.stl",
    },
    "meniscus_medial_inferior": {
        "mesh_file": "med_men_lower_art_surf_osim.stl",
    },
    "meniscus_lateral_superior": {
        "mesh_file": "lat_men_upper_art_surf_osim.stl",
    },
    "meniscus_lateral_inferior": {
        "mesh_file": "lat_men_lower_art_surf_osim.stl",
    },
}


DICT_BODY_GEOMETRIES_UPDATE = {
    "femur_distal_r": {
        "femur_bone": "femur_nsm_recon_osim.stl",
        "femur_cartilage": "femur_articular_surface_osim.stl",
    },
    "tibia_proximal_r": {
        "tibia_bone": "tibia_nsm_recon_osim.stl",
        "tibia_cartilage": "tibia_articular_surface_osim.stl",
    },
    "patella_r": {
        "patella_bone": "patella_nsm_recon_osim.stl",
        "patella_cartilage": "patella_articular_surface_osim.stl",
    },
    "meniscus_medial_r": {
        "meniscus_medial_r": "med_men_osim.stl",
    },
    "meniscus_lateral_r": {
        "meniscus_lateral_r": "lat_men_osim.stl",
    },
}

DICT_LIGAMENTS_UPDATE_STIFFNESS = {
    "PT1": {"default_stiffness": 3_000, "update_factor": 1.5},
    "PT2": {"default_stiffness": 3_000, "update_factor": 1.5},
    "PT3": {"default_stiffness": 3_000, "update_factor": 1.5},
    "PT4": {"default_stiffness": 3_000, "update_factor": 1.5},
    "PT5": {"default_stiffness": 3_000, "update_factor": 1.5},
    "PT6": {"default_stiffness": 3_000, "update_factor": 1.5},
}


# UPDATE WRAP OBJECTS
def update_wrap_objects(model, dict_wrap_objects):
    """
    Updates the properties of wrapping objects in an OpenSim model.

    Iterates through a list of wrap surface objects and updates their
    `xyz_body_rotation`, `translation`, and type-specific properties (radius, length
    for WrapCylinder; dimensions for WrapEllipsoid).

    The femur offset is updated based on the offset between the femur_r and femur_distal_r.


    Args:
        model (opensim.Model): The OpenSim model to update.
        dict_wrap_objects (dict): A dictionary of wrap surface objects.
    """

    for bone, bone_dict in dict_wrap_objects.items():
        for body, body_dict in bone_dict.items():
            if body == "femur_r":
                # TODO: update dictionaries to include parent/child info so
                # this doesnt need to be inferred?
                # get the offset between the femur_r and femur_distal_r
                offset = express_point_in_frame(
                    xyz_in_source=np.array([0, 0, 0]),
                    state=model.initSystem(),
                    source_frame_name="femur_distal_r",
                    target_frame_name="femur_r",
                    model=model,
                )
            else:
                offset = [0, 0, 0]
            for wrap_type, wrap_dicts in body_dict.items():
                for wrap_name, wrap_params in wrap_dicts.items():
                    if wrap_type == "cylinder":
                        update_wrap_cylinder(
                            model=model,
                            body_name=body,
                            wrap_name=wrap_name,
                            translation=wrap_params.translation + offset,
                            xyz_body_rotation=wrap_params.xyz_body_rotation,
                            radius=wrap_params.radius,
                            length=wrap_params.length,
                        )
                    elif wrap_type == "ellipsoid":
                        update_wrap_ellipsoid(
                            model=model,
                            body_name=body,
                            wrap_name=wrap_name,
                            translation=wrap_params.translation + offset,
                            xyz_body_rotation=wrap_params.xyz_body_rotation,
                            dimensions=wrap_params.dimensions,
                        )
                    else:
                        raise ValueError(f"Invalid wrap type: {wrap_type}")


def update_osim_model(
    model,
    dict_wrap_objects,
    dict_lig_mus_attach,
    tibia_mesh_osim,
    mean_patella,
    lig_musc_xyz_key="xyz_mesh_updated",
    lig_musc_normal_vector_shift=1e-4,  ## mm shift
    dict_body_geometries_update=DICT_BODY_GEOMETRIES_UPDATE,
    dict_contact_mesh_files_update=DICT_CONTACT_MESHFILES_UPDATE,
    dict_ligament_stiffness_update=None,
    dict_joints_coords_to_update=None,
):
    """
    Updates an entire OpenSim model with new geometry and attachments.

    This function orchestrates several updates:
    1.  Sets a new model name.
    2.  Updates BodySet STL file paths for visualization (`update_comak_bodyset_stl`).
    3.  Updates ContactGeometrySet STL file paths for simulation (`update_comak_contact_geometry_stl`).
    4.  Updates wrap object properties (`update_wrap_objects`).
    5.  Updates muscle attachment locations (`update_muscle_attachments`).
    6.  Updates ligament attachment locations (`update_ligament_attachments`).
    7.  Updates the default patella location (`update_patella_location`).

    Args:
        path_model (str): Path to the original OpenSim model (.osim) file.
        list_results (list): List of wrap surface objects for `update_wrap_objects`.
        muscle_df (pandas.DataFrame): DataFrame for `update_muscle_attachments`.
        ligament_df (pandas.DataFrame): DataFrame for `update_ligament_attachments`.
        fem_interpolated_pts_osim (numpy.ndarray): Femur points for attachments.
        tib_interpolated_pts_osim (numpy.ndarray): Tibia points for attachments.
        pat_interpolated_pts_osim (numpy.ndarray): Patella points for attachments.
        mean_patella (list or numpy.ndarray): New patella default position.
        tib_mesh_osim (pymskt.mesh.Mesh): Tibia mesh for ligament adjustments.
        new_model_name (str, optional): Name for the updated model. Defaults to a
            timestamped name.

    """

    # update the geometry files used for visualization & for contact force
    # simulations.
    update_body_geometry_meshfile(model, dict_body_geometries_update)
    update_contact_mesh_files(model, dict_contact_mesh_files_update)

    # update the wrap objects for the model.
    update_wrap_objects(model, dict_wrap_objects)

    # update the model ligament & muscle attachments, and then
    # update the slack lengths of them.
    update_model_attachments_slacks(
        model=model,
        dict_lig_mus_attach=dict_lig_mus_attach,
        ref_tibia_mesh=tibia_mesh_osim,
        state=model.initSystem(),
        xyz_key=lig_musc_xyz_key,
        normal_vector_shift=lig_musc_normal_vector_shift,
    )

    # update the default values for the joints.
    dict_patella_default_update = {
        "pf_r": {
            3: mean_patella[0],
            4: mean_patella[1],
            5: mean_patella[2],
        }
    }
    update_joint_default_values(model, dict_patella_default_update)

    if dict_joints_coords_to_update is not None:
        update_joint_default_values(model, dict_joints_coords_to_update)

    if dict_ligament_stiffness_update is not None:
        # update the ligament stiffness
        for ligament, ligament_dict in dict_ligament_stiffness_update.items():
            new_stiffness = ligament_dict["default_stiffness"] * ligament_dict["update_factor"]
            update_ligament_stiffness(
                model=model, ligament=ligament, linear_stiffness=new_stiffness
            )
