"""Repoint a COMAK OpenSim model at subject-specific geometry and attachments.

High-level helpers that take an already-loaded OpenSim model and rewrite its
mesh-file references, wrap surfaces, ligament/muscle attachments, and joint
defaults so the model uses subject-specific (or synthetic) geometry. All
spatial inputs (attachment xyz, wrap parameters, patella position) are in OSIM
space (meters, OpenSim body-local frames); the STL/OBJ paths handed to the
repoint helpers are expected to point at meshes in that same space.
"""

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
from nsosim.schemas import validate_fitted_wrap_parameters

DICT_CONTACT_MESHFILES_UPDATE = {
    "femur_cartilage": {
        "mesh_file": "femur_articular_surface_osim.obj",
        "mesh_back_file": "femur_nsm_recon_osim.stl",
    },
    "tibia_cartilage": {
        "mesh_file": "tibia_articular_surface_osim.obj",
        "mesh_back_file": "tibia_nsm_recon_osim.stl",
    },
    "patella_cartilage": {
        "mesh_file": "patella_articular_surface_osim.obj",
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
                # TODO: update dictionaries to include parent/child info so this doesn't
                # need to be inferred. Target structure: each body entry would include a
                # 'parent_body' key, e.g.:
                #   {'femur_distal_r': {'parent_body': 'femur_r', 'cylinder': {...}, ...}}
                # This would replace the hardcoded `if body == "femur_r"` check below.
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
    Repoint a whole OpenSim model at new geometry, wrap surfaces, and attachments.

    High-level orchestrator that mutates ``model`` in place by running, in order:
    1.  ``validate_fitted_wrap_parameters`` on ``dict_wrap_objects``.
    2.  ``update_body_geometry_meshfile`` — repoint body visualization geometry
        ``mesh_file`` references (``dict_body_geometries_update``).
    3.  ``update_contact_mesh_files`` — repoint Smith2018ContactMesh
        ``mesh_file`` references (``dict_contact_mesh_files_update``).
    4.  ``update_wrap_objects`` — write the fitted wrap surface parameters.
    5.  ``update_model_attachments_slacks`` — write ligament/muscle attachment
        locations and recompute slack lengths.
    6.  ``update_joint_default_values`` — set the patella (``pf_r``) default
        position from ``mean_patella``, and optionally other joint coords.
    7.  Optionally scale ligament stiffness (``dict_ligament_stiffness_update``).

    Place in the pipeline:
        Final geometry-assembly step of the MRI-fitting pipeline (Stage 5,
        "OpenSim Model Update"). It is called once all subject-specific meshes,
        wrap parameters, and attachment locations have been produced; the
        synthetic-decode pipeline can reuse it the same way once it has meshes
        and parameters in OSIM space. It delegates to the ``update_*`` helpers in
        ``nsosim.osim_utils`` plus ``update_wrap_objects`` above. It does not
        load or save the ``.osim`` file — the caller owns model I/O.

    All spatial inputs are in OSIM space (meters, OpenSim body-local frames):
    wrap parameters (translation, dimensions, etc.), attachment ``xyz`` values,
    and ``mean_patella``. The mesh-file paths point at STL/OBJ geometry in that
    same space (meters, reference size for recon meshes).

    Args:
        model (osim.Model): Loaded OpenSim model, mutated in place.
        dict_wrap_objects (dict): Nested fitted wrap parameters
            (bone → body → wrap_type → wrap_name → ``wrap_surface``), in OSIM
            meters. Validated then applied by ``update_wrap_objects``.
        dict_lig_mus_attach (dict or str): Ligament/muscle attachment dict (or
            path to its JSON). Attachment xyz under ``lig_musc_xyz_key`` are in
            OSIM meters, expressed in their parent body-local frames.
        tibia_mesh_osim (pymskt.mesh.Mesh or str): Tibia mesh in OSIM space
            (meters); used to derive the tibia size vector for attachment shifts.
        mean_patella (list or numpy.ndarray): Patella default position
            (``pf_r`` coords 3–5) in OSIM meters.
        lig_musc_xyz_key (str): Key selecting which xyz field in the attachment
            dict to write (default ``"xyz_mesh_updated"``).
        lig_musc_normal_vector_shift (float): Absolute shift magnitude (meters)
            along surface normals for attachment points that request it.
        dict_body_geometries_update (dict): Body-geometry repoint map passed to
            ``update_body_geometry_meshfile`` (defaults to
            ``DICT_BODY_GEOMETRIES_UPDATE``).
        dict_contact_mesh_files_update (dict): Contact-mesh repoint map passed to
            ``update_contact_mesh_files`` (defaults to
            ``DICT_CONTACT_MESHFILES_UPDATE``).
        dict_ligament_stiffness_update (dict or None): Optional per-ligament
            ``{default_stiffness, update_factor}`` map; when given, sets each
            ligament's linear stiffness to ``default_stiffness * update_factor``.
        dict_joints_coords_to_update (dict or None): Optional extra joint default
            values to set, in addition to the patella update.

    """

    # Validate wrap objects structure before processing
    validate_fitted_wrap_parameters(dict_wrap_objects)

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
