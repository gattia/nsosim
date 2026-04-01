"""Decode arbitrary latent vectors to OSIM-space meshes.

This module provides the inverse of the fitting pipeline: given a latent vector
and the corresponding alignment transform, decode it through the NSM model and
convert the resulting mesh to OpenSim coordinates.

Two levels of API:

- ``decode_latent_to_osim``: single bone — takes one latent vector + transform,
  returns a dict of named meshes (bone, cart, etc.).
- ``decode_joint_from_descriptors``: full joint — takes per-bone latents and
  relative transforms, returns nested dict of all bones and their meshes.

Requires GPU (CUDA) for the NSM decoder forward pass.
"""

import logging

import numpy as np
import torch
from NSM.mesh import create_mesh

from nsosim.nsm_fitting import convert_nsm_recon_to_OSIM
from nsosim.transforms import recover_bone_transform
from nsosim.utils import get_mesh_names

logger = logging.getLogger(__name__)


def decode_latent_to_osim(
    latent_vector,
    model,
    linear_transform,
    fem_ref_center,
    model_config,
    n_pts_per_axis=256,
    clusters=None,
):
    """Decode a latent vector to OSIM-space meshes for a single bone.

    Steps:
        1. Decode latent via ``NSM.mesh.create_mesh`` → canonical-space meshes
        2. Convert each mesh to OSIM coordinates via ``convert_nsm_recon_to_OSIM``
        3. Optionally resample each mesh

    Args:
        latent_vector (numpy.ndarray): 1-D latent vector (e.g. shape ``(1024,)``).
        model (torch.nn.Module): NSM decoder model, already on GPU.
        linear_transform (numpy.ndarray): 4×4 similarity transform mapping
            femur-aligned space → this bone's NSM canonical space.
        fem_ref_center (numpy.ndarray): Femur reference center (``mean_orig``
            from the reference femur alignment JSON). Used for all bones.
        model_config (dict): Model configuration. Must contain
            ``objects_per_decoder`` (int) and ideally ``mesh_names`` (list).
        n_pts_per_axis (int): Marching-cubes grid resolution per axis.
        clusters (dict or None): Per-mesh resampling targets, e.g.
            ``{'bone': 20000, 'cart': None}``. ``None`` to skip resampling.

    Returns:
        dict: ``{'bone': Mesh, 'cart': Mesh, ...}`` — keys from ``mesh_names``.
    """
    n_objects = model_config.get("objects_per_decoder", 1)
    mesh_names = get_mesh_names(model_config)

    latent_tensor = torch.tensor(latent_vector, dtype=torch.float).cuda()

    decoded = create_mesh(
        model,
        latent_tensor,
        n_pts_per_axis=n_pts_per_axis,
        objects=n_objects,
    )

    # Normalize to list
    if not isinstance(decoded, list):
        decoded = [decoded]

    linear_transform = np.asarray(linear_transform, dtype=np.float64)
    fem_ref_center = np.asarray(fem_ref_center, dtype=np.float64)

    result = {}
    for name, mesh in zip(mesh_names, decoded):
        mesh_osim = mesh.copy()
        mesh_osim.point_coords = convert_nsm_recon_to_OSIM(
            mesh.point_coords.copy(),
            linear_transform,
            1,
            np.zeros(3),
            fem_ref_center,
        )

        n_clusters = None
        if clusters is not None:
            n_clusters = clusters.get(name)
        if n_clusters is not None:
            mesh_osim.resample_surface(subdivisions=1, clusters=n_clusters)

        result[name] = mesh_osim

    return result


def decode_joint_from_descriptors(
    femur_latent,
    tibia_latent,
    patella_latent,
    T_fem,
    T_rel_tib,
    T_rel_pat,
    models,
    model_configs,
    fem_ref_center,
    n_pts_per_axis=256,
    clusters=None,
):
    """Decode a full joint from per-bone latent vectors and pose transforms.

    Recovers per-bone transforms from ``T_fem`` and relative transforms, then
    calls ``decode_latent_to_osim`` for each bone.

    Args:
        femur_latent (numpy.ndarray): Femur latent vector.
        tibia_latent (numpy.ndarray): Tibia latent vector.
        patella_latent (numpy.ndarray): Patella latent vector.
        T_fem (numpy.ndarray): 4×4 femur linear_transform.
        T_rel_tib (numpy.ndarray): 4×4 relative tibia transform
            (from ``compute_T_rel(T_fem, T_tib)``).
        T_rel_pat (numpy.ndarray): 4×4 relative patella transform
            (from ``compute_T_rel(T_fem, T_pat)``).
        models (dict): ``{'femur': model, 'tibia': model, 'patella': model}``.
        model_configs (dict): ``{'femur': config, 'tibia': config, 'patella': config}``.
        fem_ref_center (numpy.ndarray): Femur reference center.
        n_pts_per_axis (int): Marching-cubes grid resolution per axis.
        clusters (dict or None): Per-bone resampling targets, e.g.
            ``{'femur': {'bone': 20000}, 'tibia': {'bone': 20000}}``.
            ``None`` to skip all resampling.

    Returns:
        dict: ``{'femur': {'bone': Mesh, ...}, 'tibia': {...}, 'patella': {...}}``.
    """
    T_tib = recover_bone_transform(T_rel_tib, T_fem)
    T_pat = recover_bone_transform(T_rel_pat, T_fem)

    bone_specs = {
        "femur": (femur_latent, T_fem),
        "tibia": (tibia_latent, T_tib),
        "patella": (patella_latent, T_pat),
    }

    result = {}
    for bone, (latent, T) in bone_specs.items():
        bone_clusters = None
        if clusters is not None:
            bone_clusters = clusters.get(bone)

        result[bone] = decode_latent_to_osim(
            latent_vector=latent,
            model=models[bone],
            linear_transform=T,
            fem_ref_center=fem_ref_center,
            model_config=model_configs[bone],
            n_pts_per_axis=n_pts_per_axis,
            clusters=bone_clusters,
        )

    return result
