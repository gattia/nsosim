"""
Model-building orchestration: OSIM-space meshes → subject-specific OpenSim model.

Extracts the shared model-building logic from comak_1_nsm_fitting.py (lines ~370–1050)
into reusable functions. Both the fitting pipeline and synthetic joint pipeline call
the same code here.

Step functions are pure (no file I/O) — they take data in and return results.
The orchestrator (build_joint_model) handles all saving.

Coordinate space convention for this module
--------------------------------------------
Every mesh, point array, and attachment coordinate handled here is in **OSIM
space**: the OpenSim body-local frame produced by ``nsm_recon_to_osim()``
(via ``convert_nsm_recon_to_OSIM_``: add the fixed reference center, mm→m,
axis-swap). Units are **metres** and the scale identity is **reference size,
rotated** — i.e. the smith2019 reference bone's size, because the subject mesh
was similarity-registered onto that fixed reference (REFALIGN) before NSM
reconstruction, dividing the subject's true physical size out. For the knee
sub-bodies the body-local origin is the knee joint center, so OSIM space and the
body-local STL-attachment frame coincide for the recon meshes.

Consequently the articular-surface, ligament-interpolation, wrap-fitting,
meniscus, and fat-pad builders here all operate in OSIM metres at reference
size, and any scale baked into the NSM recon is inherited by everything they
build. Individual functions restate this where it matters.
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
            raise ValueError(f"JSON at {path} has no 'linear_transform' or 'transform_matrix' key")
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
                "path_labeled_bone": os.path.join(labeled_bone_dir, f"{bone_name}_labeled.vtk"),
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


# Default mesh-interpolation recipe for bone-ligament warping. NSM
# mesh-interpolation-trim API (2026-05-22): Newton magnitude is unconditional,
# the dihedral pin is always on with tangent_laplacian. We keep theta=60 deg
# (bone optimum from the NSM-team sweep: fold-over -90%, ASSD -22%) rather than
# the library default of 45 deg. Cart wants theta=30 deg but cart vertices are
# not warped through this function. Same shape as
# _MENISCUS_INTERPOLATE_RECIPE_DEFAULTS but named separately so bone vs menisci
# can be tuned independently if needed.
_BONE_INTERPOLATE_RECIPE_DEFAULTS = {
    "tangent_laplacian": True,
    "tangent_laplacian_feature_angle": 60.0,
}


def interpolate_bone_ligaments(
    bone_name,
    labeled_mesh_path,
    dict_lig_musc_attach_params,
    dict_bones,
    fem_ref_center,
    folder_ref_recons,
    surface_idx=0,
    interpolate_kwargs=None,
    snap_warn_mm=8.0,
):
    """Interpolate labeled bone mesh + ligament attachments from reference to subject.

    **Vertex-identity warp** (2026-05-22). Snap each ligament attachment to its
    nearest vertex on the labeled bone mesh, then warp the full mesh through
    the NSM with the mesh-interpolation-improvements recipe (newton step +
    tangent-Laplacian + dihedral pin at θ=60°). Each attachment's warped
    position is the warped vertex at its snapped index. The mesh portion
    benefits from the full recipe (which prior bare-points concatenation
    delivered for the mesh but only Newton magnitude for the attachments).

    Source: comak_1_nsm_fitting.py lines 408–449 (femur), 541–582 (tibia),
    825–856 (patella).

    Coordinate space
    ----------------
    The labeled reference mesh is loaded from ``labeled_mesh_path``, but the
    function returns it warped into **OSIM space** (metres, reference size,
    rotated) — ``interp_ref_to_subject_to_osim`` produces OSIM-space output, so
    both the returned mesh points and the warped ligament attachment positions
    are OSIM metres. The attachment ``xyz_mesh`` reference seeds that are
    snapped to vertices are likewise read in OSIM metres (the ``d * 1000``
    snap-distance reporting converts m→mm for the warning).

    Parameters
    ----------
    bone_name : str
        Bone name ('femur', 'tibia', 'patella') used to match parent_frame in
        ligament dicts.
    labeled_mesh_path : str
        Path to the labeled bone VTK file with wrap surface classifications.
    dict_lig_musc_attach_params : dict
        Ligament/muscle attachment parameters dict. Attachment ``xyz_mesh``
        seeds are in OSIM metres. Not modified in-place.
    dict_bones : dict
        dict_bones-compatible structure with recon_dict + recon_latent per bone.
    fem_ref_center : np.ndarray
        Femur reference center from ref_femur_alignment.json['mean_orig'] (the
        reference femur centroid in NSM-oriented mm, used by the OSIM conversion).
    folder_ref_recons : str
        Path to folder containing reference reconstruction data.
    surface_idx : int
        Surface index for NSM interpolation (0=bone).
    interpolate_kwargs : dict, optional
        Forwarded to ``interp_ref_to_subject_to_osim`` (merged on top of
        ``_BONE_INTERPOLATE_RECIPE_DEFAULTS``; ``faces`` always injected
        from the loaded labeled mesh).
    snap_warn_mm : float
        Warn when a bone attachment's reference ``xyz_mesh`` is farther than
        this (in mm) from its nearest labeled-bone-mesh vertex.

    Returns
    -------
    labeled_mesh : Mesh
        Labeled mesh with updated (warped) point coordinates, in OSIM space
        (metres, reference size).
    labeled_mesh_points : np.ndarray
        Copy of the updated labeled mesh point coordinates (OSIM metres).
    lig_xyz_points_updated : np.ndarray
        Warped ligament attachment point coordinates (n_lig_pts, 3), in OSIM
        metres.
    list_lig_name_pt_idx : list of [str, int]
        [force_name, point_index] pairs identifying which ligament points were
        updated.
    """
    from scipy.spatial import cKDTree

    labeled_mesh = Mesh(labeled_mesh_path)
    ref_pts = labeled_mesh.point_coords
    tree = cKDTree(ref_pts)

    # Collect bone-side attachments and snap each to nearest mesh vertex
    list_lig_musc_name_pt_idx = []
    snap_indices = []
    for key, dict_ in dict_lig_musc_attach_params.items():
        for pt_idx, point_dict in enumerate(dict_["points"]):
            if bone_name not in point_dict["parent_frame"]:
                continue
            xyz = np.asarray(point_dict["xyz_mesh"], dtype=float)
            d, vtx = tree.query(xyz)
            list_lig_musc_name_pt_idx.append([key, pt_idx])
            snap_indices.append(int(vtx))
            if d * 1000 > snap_warn_mm:
                print(
                    f"  WARN interpolate_bone_ligaments ({bone_name}): {key} "
                    f"point {pt_idx} snapped {d*1000:.2f} mm to mesh vertex {vtx}"
                )

    # Warp the labeled mesh alone (no concatenation) with the recipe.
    user_kwargs = dict(interpolate_kwargs) if interpolate_kwargs else {}
    kwargs = dict(_BONE_INTERPOLATE_RECIPE_DEFAULTS)
    kwargs.update(user_kwargs)
    # pymskt Mesh wraps a pyvista PolyData on `.mesh`
    kwargs["faces"] = labeled_mesh.mesh.regular_faces.astype(np.int64)

    interpolated_pts_osim = interp_ref_to_subject_to_osim(
        ref_mesh=labeled_mesh,
        surface_name=bone_name,
        ref_center=fem_ref_center,
        dict_bones=dict_bones,
        folder_nsm_files=folder_ref_recons,
        surface_idx=surface_idx,
        interpolate_kwargs=kwargs,
    )

    # Attachment warped positions = warped vertex at the snapped index.
    if snap_indices:
        lig_xyz_points_updated = interpolated_pts_osim[np.asarray(snap_indices, dtype=int), :]
    else:
        lig_xyz_points_updated = np.zeros((0, 3))

    # Update labeled mesh with warped points
    labeled_mesh.point_coords = interpolated_pts_osim
    labeled_mesh_points = labeled_mesh.point_coords.copy()

    return labeled_mesh, labeled_mesh_points, lig_xyz_points_updated, list_lig_musc_name_pt_idx


def _fit_with_restarts(
    fitter_class,
    constructor_kwargs,
    fit_kwargs,
    points,
    n_restarts=1,
    jitter_scale=1e-6,
):
    """Run fitter.fit() ``n_restarts`` times and return the fitter with the
    lowest ``final_loss``.

    The wrap fitters (CylinderFitter / EllipsoidFitter) are deterministic
    given a fixed seed and identical input — but they're also sensitive to
    sub-micron input perturbations near degenerate axis configurations,
    which means tiny upstream drift in the bone mesh (from grid_sample
    backward CUDA non-determinism) can amplify into mm-scale wrap surface
    drift. Multi-start with input jitter gives the fitter a chance to find
    the dominant optimum across small perturbations rather than locking
    onto whatever local minimum the unperturbed init lands in.

    First restart uses the unperturbed input (matches single-fit behavior
    when n_restarts=1). Subsequent restarts add gaussian jitter scaled by
    ``jitter_scale`` (default 1 µm — well below MRI voxel resolution and
    smaller than the upstream drift we're trying to hedge against).
    """
    import numpy as np
    import torch

    points_arr = np.asarray(points, dtype=np.float64)
    best_fitter = None

    for k in range(n_restarts):
        if k == 0:
            jittered = points_arr
        else:
            # Deterministic per-restart jitter: tie torch+numpy state to k.
            torch.manual_seed(int(1e6) + k)
            np.random.seed(int(1e6) + k)
            jittered = points_arr + np.random.randn(*points_arr.shape) * jitter_scale

        fitter = fitter_class(**constructor_kwargs)
        these_fit_kwargs = dict(fit_kwargs)
        these_fit_kwargs["points"] = jittered
        # Carry over near_surface_points override (used by CylinderFitter).
        if "near_surface_points" in fit_kwargs and fit_kwargs["near_surface_points"] is not None:
            # Same indices were already applied — just jitter the same way.
            ns = np.asarray(fit_kwargs["near_surface_points"], dtype=np.float64)
            if k == 0:
                these_fit_kwargs["near_surface_points"] = ns
            else:
                these_fit_kwargs["near_surface_points"] = (
                    ns + np.random.randn(*ns.shape) * jitter_scale
                )
        fitter.fit(**these_fit_kwargs)

        if best_fitter is None or fitter.final_loss < best_fitter.final_loss:
            best_fitter = fitter

    return best_fitter


def fit_bone_wrap_surfaces(
    bone_name,
    labeled_mesh,
    labeled_mesh_points,
    wrap_surface_spec=None,
    fitter_configs=None,
    patella_wrap_dimension_scale=0.9,
    n_restarts=1,
    jitter_scale=1e-6,
    anchors=None,
):
    """Fit wrap surfaces to a labeled bone mesh.

    Source: comak_1_nsm_fitting.py lines 452–502 (femur), 584–634 (tibia), 867–883 (patella).

    For femur/tibia: iterates over DEFAULT_SMITH2019_BONES config, fitting ellipsoids
    and cylinders using SDF-based optimization.
    For patella: uses PatellaFitter (specialized ellipsoid fitting).

    Coordinate space
    ----------------
    Operates entirely in **OSIM space** (metres, reference size, rotated): the
    labeled mesh points and the SDF arrays come from the OSIM-space subject
    bone, so the fitted ``wrap_surface`` parameters (``translation``,
    ``radius``, ``length``, ``dimensions``) are returned in OSIM metres in the
    body-local frame that wrap surface attaches to. For the patella the input
    mesh is patella-centered (mean position already subtracted), so the patella
    wrap parameters are expressed in the centered patella body-local frame.

    Parameters
    ----------
    bone_name : str
        'femur', 'tibia', or 'patella'.
    labeled_mesh : Mesh
        Labeled bone mesh with wrap surface classification arrays, in OSIM
        space (metres, reference size).
    labeled_mesh_points : np.ndarray
        Point coordinates of the labeled mesh in OSIM metres (already
        patella-centered for the patella).
    wrap_surface_spec : dict or None
        Wrap surface specification from DEFAULT_SMITH2019_BONES[bone_name]['wrap_surfaces'].
        If None, uses DEFAULT_SMITH2019_BONES[bone_name]['wrap_surfaces'].
    fitter_configs : dict or None
        Fitter configurations. If None, uses DEFAULT_FITTING_CONFIG.
    patella_wrap_dimension_scale : float
        Scale factor for patella wrap surface dimensions (default 0.9 = 10% reduction).
    n_restarts : int
        Number of multi-start restarts per wrap surface (default 1 = no multi-start,
        preserves prior behavior). Set to 3+ to hedge against wrap-fitter
        sensitivity to sub-micron input drift; pays ~n_restarts× the wrap fitting
        time but improves run-to-run reproducibility of wrap surface params.
    jitter_scale : float
        Std-dev of gaussian noise added to input points on restart 2+, in metres.
        Default 1e-6 (1 µm) — well below biomechanically meaningful scale and
        comparable to the upstream input drift the multi-start is hedging against.
        Ignored if n_restarts == 1.
    anchors : dict or None
        Optional Procrustes-from-Smith2019 anchors structured as
        ``{body_name: {surface_type: {wrap_name: wrap_surface}}}`` (output of
        ``procrustes_anchor.procrustes_anchors_from_smith2019()[bone_name]``).
        When provided, each wrap is initialized from its anchor and the
        regularizer pins toward that anchor. Wraps without an entry fall back
        to the algebraic init.

    Returns
    -------
    dict
        Fitted wrap parameters dict for this bone, structured as:
        {body_name: {surface_type: {wrap_name: wrap_surface}}}. Each
        ``wrap_surface``'s geometric fields (translation, radius, length,
        dimensions) are in OSIM metres in the corresponding body-local frame.
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

    def _anchor_for(body_name, surface_type, wrap_name):
        if anchors is None:
            return None
        try:
            return anchors[body_name][surface_type][wrap_name]
        except KeyError:
            return None

    for body_name, body_data in wrap_surface_spec.items():
        fitted[body_name] = {}
        for surface_type, surface_list in body_data.items():
            fitted[body_name][surface_type] = {}
            if surface_type == "ellipsoid":
                for wrap_name in surface_list:
                    labels = labeled_mesh[f"{wrap_name}_binary"].copy()
                    sdf = labeled_mesh[f"{wrap_name}_sdf"].copy()

                    fit_kwargs = dict(
                        labels=labels,
                        sdf=sdf,
                        mesh=labeled_mesh,
                        surface_name=wrap_name,
                        **ellipsoid_fit,
                    )
                    anchor = _anchor_for(body_name, surface_type, wrap_name)
                    constructor_kwargs = dict(ellipsoid_constructor)
                    if anchor is not None:
                        constructor_kwargs["anchor_params"] = anchor
                    fitter = _fit_with_restarts(
                        EllipsoidFitter,
                        constructor_kwargs,
                        fit_kwargs,
                        labeled_mesh_points,
                        n_restarts=n_restarts,
                        jitter_scale=jitter_scale,
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

                    fit_kwargs = dict(
                        labels=near_surface_labels,
                        sdf=near_surface_sdf,
                        mesh=labeled_mesh,
                        surface_name=wrap_name,
                        near_surface_points=near_surface_points,
                        **cylinder_fit,
                    )
                    anchor = _anchor_for(body_name, surface_type, wrap_name)
                    constructor_kwargs = dict(cylinder_constructor)
                    if anchor is not None:
                        constructor_kwargs["anchor_params"] = anchor
                    fitter = _fit_with_restarts(
                        CylinderFitter,
                        constructor_kwargs,
                        fit_kwargs,
                        near_surface_points,
                        n_restarts=n_restarts,
                        jitter_scale=jitter_scale,
                    )
                    wrap_params = fitter.wrap_params
                    wrap_params.name = wrap_name
                    wrap_params.body = body_name
                    fitted[body_name][surface_type][wrap_name] = wrap_params

    return fitted


# ---------------------------------------------------------------------------
# Cross-bone step functions
# ---------------------------------------------------------------------------


# Default interpolation recipe for the meniscus warp — NSM
# mesh-interpolation-trim API (2026-05-22). Newton magnitude is unconditional
# in the new API; the dihedral pin is always on when tangent_laplacian is set.
# We keep theta=60 deg (the menisci/bone optimum per the NSM-team sweep:
# lat_men fold -53%, ASSD -7%; med_men fold -48%, ASSD -12%) rather than the
# library default of 45 deg. The closed meniscus shell has an empty
# topological boundary, so the dihedral pin is what anchors the geometric seam
# between upper and lower shells. Requires `faces` (added per-side from the
# loaded reference meniscus mesh).
_MENISCUS_INTERPOLATE_RECIPE_DEFAULTS = {
    "tangent_laplacian": True,
    "tangent_laplacian_feature_angle": 60.0,
}

# Per-side meniscus reference-mesh filenames inside <folder_ref_recons>/femur/.
# These are the femur multi-surface NSM's decoded reference menisci in OSIM
# space (5476/5440 pts; lie exactly on surface_idx=2/3).
_MENISCUS_REF_MESH = {
    "medial": ("nsm_recon_ref_femur_med_men_osim_space.vtk", 2),
    "lateral": ("nsm_recon_ref_femur_lat_men_osim_space.vtk", 3),
}


def interpolate_meniscus_ligaments(
    dict_lig_musc_attach_params,
    dict_bones,
    fem_ref_center,
    folder_ref_recons,
    interpolate_kwargs=None,
    snap_warn_mm=8.0,
):
    """Interpolate meniscal ligament attachment points using the femur NSM model.

    **Vertex-identity warp** (2026-05-22). For each side (medial / lateral):

    1. Load the femur-NSM decoded reference meniscus mesh
       (``nsm_recon_ref_femur_{med,lat}_men_osim_space.vtk``).
    2. Snap each attachment's reference ``xyz_mesh`` to the nearest vertex on
       that mesh — the attachment's *identity* is then that vertex.
    3. Warp the **full reference mesh** through the femur NSM (surface_idx
       2 = medial, 3 = lateral) with the mesh-interpolation-improvements
       recipe (Fix 2 newton magnitude + Fix 4b/4c tangent-Laplacian with
       dihedral pin at θ=60°). Attachments ride *on* the mesh, so they
       inherit the full benefit of the recipe (tangent regularization plus
       Newton projection) — not just the Newton projection an appended
       isolated node would get.
    4. ``xyz_mesh_updated`` = warped mesh point at the snapped vertex index.

    Snap distance is logged per attachment; distances > ``snap_warn_mm``
    (default 8 mm) trigger a warning — those flag attachments whose
    reference ``xyz_mesh`` is far off the femur-model meniscus surface
    (e.g. the medial horns whose original seeds were 3–7 mm off; the
    lateral horns set manually from the labeling workflow are at exact
    vertices, snap = 0).

    Coordinate space
    ----------------
    The per-side reference meniscus meshes (``..._osim_space.vtk``) are read in
    **OSIM space** (metres, reference size, rotated), and
    ``interp_ref_to_subject_to_osim`` returns the warped points in OSIM space
    too. The attachment ``xyz_mesh`` seeds are read and the written
    ``xyz_mesh_updated`` values are stored in OSIM metres. Snap distances are
    converted m→mm only for the warning print.

    Parameters
    ----------
    dict_lig_musc_attach_params : dict
        Ligament/muscle attachment parameters. ``xyz_mesh`` seeds are in OSIM
        metres. Modified in-place: each matched attachment gets an
        ``xyz_mesh_updated`` entry holding the warped position in OSIM metres.
    dict_bones : dict
        dict_bones structure (needs femur entry with recon_dict + recon_latent).
    fem_ref_center : np.ndarray
        Femur reference center from ref_femur_alignment.json['mean_orig'].
    folder_ref_recons : str
        Path to reference reconstruction data.
    interpolate_kwargs : dict, optional
        Forwarded to ``interp_ref_to_subject_to_osim`` (merged on top of
        ``_MENISCUS_INTERPOLATE_RECIPE_DEFAULTS``; ``faces`` always injected
        from the loaded reference mesh).
    snap_warn_mm : float
        Warn when an attachment's reference ``xyz_mesh`` is farther than this
        (in mm) from the nearest reference-mesh vertex.
    """
    from scipy.spatial import cKDTree

    user_kwargs = dict(interpolate_kwargs) if interpolate_kwargs else {}

    for men_side, (ref_fname, surface_idx) in _MENISCUS_REF_MESH.items():
        # Load the per-side reference mesh + build vertex KDTree
        ref_mesh_path = os.path.join(folder_ref_recons, "femur", ref_fname)
        ref_mesh = pv.read(ref_mesh_path)
        tree = cKDTree(ref_mesh.points)

        # Snap each attachment to its nearest reference-mesh vertex
        attachments = []  # list of (force_name, pt_idx, vtx_idx, snap_dist_m, snap_xyz_ref)
        for key, dict_ in dict_lig_musc_attach_params.items():
            for pt_idx, point_dict in enumerate(dict_["points"]):
                if f"meniscus_{men_side}" not in point_dict["parent_frame"]:
                    continue
                xyz = np.asarray(point_dict["xyz_mesh"], dtype=float)
                d, vtx = tree.query(xyz)
                attachments.append([key, pt_idx, int(vtx), float(d), xyz])
                if d * 1000 > snap_warn_mm:
                    print(
                        f"  WARN interpolate_meniscus_ligaments: {key} "
                        f"point {pt_idx} ({men_side}-side) snapped {d*1000:.2f} mm "
                        f"to ref vertex {vtx} (xyz_mesh was off the femur-model "
                        f"meniscus surface)"
                    )

        if not attachments:
            continue

        # Warp the full reference mesh with the recipe.
        kwargs = dict(_MENISCUS_INTERPOLATE_RECIPE_DEFAULTS)
        kwargs.update(user_kwargs)
        kwargs["faces"] = ref_mesh.regular_faces.astype(np.int64)

        warped_pts = interp_ref_to_subject_to_osim(
            ref_mesh=ref_mesh,
            surface_name="femur",
            ref_center=fem_ref_center,
            dict_bones=dict_bones,
            folder_nsm_files=folder_ref_recons,
            surface_idx=surface_idx,
            interpolate_kwargs=kwargs,
        )

        # Each attachment's warped position is the warped vertex.
        for force_name, pt_idx, vtx_idx, _snap_d, _snap_xyz in attachments:
            dict_lig_musc_attach_params[force_name]["points"][pt_idx]["xyz_mesh_updated"] = (
                warped_pts[vtx_idx, :]
            )


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

    Subtracts ``mean_patella`` (the mean of the bone mesh point coordinates)
    from the bone, articular, and (optional) cartilage meshes so the patella
    STL is centered at its body-local origin. The same offset is later written
    as the patellofemoral joint coordinate offset.

    Source: comak_1_nsm_fitting.py lines 794–822.

    Coordinate space
    ----------------
    Inputs are in **OSIM space** (metres, reference size, rotated). Outputs are
    in the centered patella body-local frame (OSIM metres with the patella
    centroid moved to the origin). ``mean_patella`` is the subtracted
    translation, in OSIM metres at reference size.

    Parameters
    ----------
    pat_mesh : Mesh
        Patella bone mesh in OSIM space (metres, reference size).
    pat_articular : Mesh
        Patella articular surface mesh in OSIM space.
    pat_cart_mesh : Mesh or None
        Patella cartilage mesh in OSIM space (optional).

    Returns
    -------
    pat_mesh_centered : Mesh
        Patella bone mesh with ``mean_patella`` subtracted (centered body-local
        frame, OSIM metres).
    pat_articular_centered : Mesh
        Patella articular surface mesh with ``mean_patella`` subtracted.
    pat_cart_centered : Mesh or None
        Patella cartilage mesh with ``mean_patella`` subtracted (None if input
        was None).
    mean_patella : np.ndarray
        The centroid that was subtracted (OSIM metres, reference size); saved as
        the patellofemoral joint coordinate offset.
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
    """Copy the newly generated geometry files into the OpenSim model's Geometry/ folder.

    Copies each per-bone mesh file (e.g. ``femur_nsm_recon_osim.stl``,
    ``tibia_articular_surface_osim.stl``, ``med_men_osim.stl``) from
    ``folder_save_bones/<bone>/`` into ``<path_save_model>/Geometry/``,
    creating the Geometry directory if needed. These are *new* filenames that
    sit alongside the template model's reference geometry; this does not
    overwrite the reference STLs (e.g. ``smith2019-R-femur-bone.stl``), which
    have different names.

    Source: comak_1_nsm_fitting.py lines 934–964.

    Coordinate space
    ----------------
    Pure file copy — does not transform geometry. The copied STL/OBJ meshes are
    in **OSIM space** (metres, reference size) as produced upstream; the patella
    STLs copied here are the centered (body-local) variants.

    Parameters
    ----------
    folder_save_bones : str
        Root folder containing per-bone subfolders with generated meshes.
    path_save_model : str
        Path to the OpenSim model directory (Geometry/ subfolder will be created).
    geometry_dict : dict or None
        {bone_name: [filename, ...]} mapping of files to copy. If None, uses
        DEFAULT_GEOMETRY_FILES.
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

    Writes the fitted wrap surfaces, interpolated ligament/muscle attachments,
    patella centering offset, prefemoral-fat-pad contact mesh, and fat-pad
    contact force into the loaded ``osim_model``, then names it and prints it to
    ``<path_save>/<model_name>.osim``.

    Place in the pipeline
    ---------------------
    Final assembly step of the MRI→model chain. Called by ``build_joint_model``
    after all per-bone geometry has been built and saved; it is the only step
    that mutates the OpenSim ``Model`` object. It delegates the bulk of the XML
    editing to ``update_osim_model`` and the fat-pad contact wiring to
    ``create_contact_mesh`` / ``create_articular_contact_force``.

    Source: comak_1_nsm_fitting.py lines 966–1051.

    Coordinate space
    ----------------
    All geometric inputs are in **OSIM space** (metres, reference size,
    rotated): the wrap parameters, the ``xyz_mesh_updated`` attachment
    coordinates, the tibia mesh used for coronary/meniscal projection, and
    ``mean_patella`` (written as the patellofemoral joint coordinate offset).
    Proximity / shift lengths below are in metres.

    Parameters
    ----------
    osim_model : osim.Model
        Loaded OpenSim model.
    fitted_wrap_parameters : dict
        Fitted wrap parameters for all bones (geometry in OSIM metres,
        body-local).
    dict_lig_musc_attach_params : dict
        Ligament/muscle attachment parameters with 'xyz_mesh_updated' entries
        in OSIM metres.
    tib_mesh_osim : Mesh
        Subject tibia mesh in OSIM space (metres, reference size).
    mean_patella : np.ndarray
        Patella centroid offset in OSIM metres (written as the patellofemoral
        joint coordinate offset).
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
    project_coronary=True,
    triangle_density=3_000_000,
    folder_save_bones=None,
    seed=0,
):
    """Build a subject-specific OpenSim knee model from OSIM-space meshes.

    This is the main orchestrator. It takes meshes (from fitting OR decoding),
    extracts articular surfaces, interpolates ligament attachments, fits wrap
    surfaces, creates meniscus surfaces and fat pad, and assembles the final
    OpenSim model.

    Place in the pipeline
    ---------------------
    Shared assembler used by BOTH the MRI-fitting pipeline and the
    synthetic-decode pipeline — the last stage of the MRI→model (or
    latent→model) chain. Its caller supplies ``bone_meshes`` already in OSIM
    space (the output of ``nsm_recon_to_osim()`` in the fitting path, or of
    the decode path); this function does not do the NSM→OSIM conversion itself.
    It calls the per-bone step functions in this module
    (``create_articular_surfaces``, ``interpolate_bone_ligaments``,
    ``fit_bone_wrap_surfaces``, ``create_meniscus_articulating_surface``,
    ``interpolate_meniscus_ligaments``, ``create_prefemoral_fatpad_noboolean``,
    ``center_patella_meshes``), ``copytree``s the base model directory, writes
    the recon/derived STLs via ``save_geometry_files``, and finalizes the
    ``.osim`` via ``finalize_osim_model``.

    Coordinate space
    ----------------
    Every mesh in ``bone_meshes`` is in **OSIM space** (metres, reference size,
    rotated). All derived geometry (articular surfaces, wrap surfaces, meniscus
    surfaces, fat pad) is produced and saved in OSIM metres. The patella is the
    one exception in handling: this function consumes the non-centered patella
    meshes, saves them as ``*_original_position.vtk`` (OSIM space *before*
    centering), and does its own centering via ``center_patella_meshes`` — the
    centered patella STLs/wraps live in the centered patella body-local frame
    and ``mean_patella`` is recorded as the patellofemoral joint offset
    (``patella_offset.json``).

    Parameters
    ----------
    bone_meshes : dict
        OSIM-space meshes (metres, reference size)::

            {'femur': {'bone': Mesh, 'cart': Mesh, 'med_men': Mesh, 'lat_men': Mesh},
             'tibia': {'bone': Mesh, 'cart': Mesh},
             'patella': {'bone': Mesh, 'cart': Mesh}}

        Patella entries are the non-centered ("original position") meshes;
        centering is performed internally.

    dict_bones : dict
        dict_bones-compatible structure with recon_dict + recon_latent per bone.
        Used by interp_ref_to_subject_to_osim for ligament interpolation.
    ref_data_paths : dict
        Reference data paths::

            {'folder_ref_recons': str,  # folder with per-bone ref alignment/latent data
             'lig_attach_params_path': str}  # (unused here, params passed directly)

    dict_lig_musc_attach_params : dict
        Ligament/muscle attachment parameters (will be modified in-place;
        ``xyz_mesh`` seeds and the ``xyz_mesh_updated`` results are in OSIM
        metres).
    fem_ref_center : np.ndarray
        Femur reference center from ref_femur_alignment.json['mean_orig'] (the
        reference femur centroid in NSM-oriented mm, used by the OSIM conversion
        inside the interpolation steps).
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
        - 'smith2019_osim_path': str or None (default None). When set, Procrustes
          anchors are built from the named Smith2019 osim and passed to each
          wrap fit as the init + regularizer target. Biases fits toward
          trusted Smith2019 geometry instead of the algebraic init's biased
          estimate on the subject bone.
        - 'wraps_to_skip_anchor': list of str (default ['Med_Lig_r']). Wrap
          names whose anchor is removed from the per-bone anchors dict before
          fitting. Use for wraps whose loss landscape has a worse local
          minimum near the Smith2019 anchor than near the algebraic init
          (Med_Lig_r is the known case from iter8.5 sweep, recovering
          ~2 percentage points of classification accuracy).

    project_meniscal_to_tibia : bool
        Whether to project meniscal ligament tibia attachments onto tibia surface.
    project_coronary : bool
        Whether to project coronary ligament tibia attachments to the closest
        point on the tibia surface. Defaults to True (the corrected behavior).
        Set False to match the original comak_1_nsm_fitting.py production
        behavior, where the coronary block wrote to 'xyz_mesh' instead of
        the 'xyz_mesh_updated' key consumed by update_osim_model — making
        the projection dead code. Used in the rewire regression test to
        confirm bit-exact equivalence with production.
    triangle_density : int
        Triangle density for articular surface extraction (can also be set via config).
    folder_save_bones : str or None
        Directory for per-bone intermediate outputs (subdirs femur/, tibia/,
        patella/). Defaults to ``save_dir`` when None — keeps the synthetic-
        joint behavior where per-bone files sit alongside the model dir.
        Pass an explicit path (e.g. ``geometries_nsm_similarity``) to keep
        per-bone outputs separate from the model's parent directory.
    seed : int or None
        Seed for all RNGs (PyTorch, CUDA, NumPy, Python ``random``, cudnn
        flags). Pinned at the top of the orchestrator so wrap-surface fitting
        and any downstream sampling are deterministic. Defaults to 0. Pass
        ``None`` to opt out.

    Returns
    -------
    str
        Path to the saved .osim model file.
    """
    import opensim as osim

    if seed is not None:
        from ._determinism import set_global_seed

        set_global_seed(seed)

    if config is None:
        config = {}

    def cfg(key, default):
        return config.get(key, default)

    tri_density = cfg("triangle_density", triangle_density)
    fitter_configs = cfg("fitter_configs", None)
    patella_wrap_dim_scale = cfg("patella_wrap_dimension_scale", 0.9)
    wrap_n_restarts = cfg("wrap_n_restarts", 1)
    wrap_jitter_scale = cfg("wrap_jitter_scale", 1e-6)
    folder_ref_recons = ref_data_paths["folder_ref_recons"]
    if folder_save_bones is None:
        folder_save_bones = save_dir

    # Wrap-fit mode selector. Three options:
    #   'lbfgs_smith2019_anchor' (default): Smith2019 wrap params used as
    #     LBFGS init + regularizer target. Per-wrap opt-out via
    #     ``wraps_to_skip_anchor`` for known bad-basin wraps (Med_Lig_r).
    #   'lbfgs_algebraic': pre-anchor behavior — no Smith2019 reference,
    #     algebraic init from the labels, LBFGS optimizes from there.
    #   'label_correspondence': subject-adapted wrap surfaces computed
    #     directly from per-wrap label correspondence (Procrustes on
    #     ref-near-surface → subject-near-surface bone vertices), applied
    #     to Smith2019 wrap params. Used as the final fitted result; no
    #     LBFGS. Beats LBFGS-with-Smith2019-anchor on every fittable wrap
    #     in 10-subject validation, with A↔B reproducibility < 16 µm.
    #     Patella (PatTen_r) still uses PatellaFitter — outside this mode's
    #     scope.
    wrap_fit_mode = cfg("wrap_fit_mode", "lbfgs_smith2019_anchor")

    # Build Procrustes-from-Smith2019 anchors once if a reference osim is
    # supplied via config['smith2019_osim_path']. The anchors get passed to
    # each fit_bone_wrap_surfaces call as the init + regularizer target,
    # biasing fits toward the trusted Smith2019 geometry rather than toward
    # the algebraic init's biased estimate. See WRAP_FITTER_ROBUSTNESS rev 2.
    smith2019_osim_path = cfg("smith2019_osim_path", None)
    # Per-wrap opt-out: wraps named here are removed from the anchors dict so
    # they fall back to the algebraic init + fitter defaults. Discovered
    # empirically (iter8.5) that Med_Lig_r's loss landscape has a worse local
    # minimum near the Smith2019 anchor than near the algebraic init; the
    # algebraic init recovers 99.4 % vs the anchor's 97 % on subject 9018389.
    # The default skip-list is hard-coded but overridable via config.
    # NB: under wrap_fit_mode='label_correspondence' this opt-out is
    # unnecessary because the anchor is in the data-optimum basin by
    # construction — the skip list is ignored in that mode.
    wraps_to_skip_anchor = set(cfg("wraps_to_skip_anchor", ["Med_Lig_r"]))

    # Sub-option for label_correspondence: which Procrustes flavor maps the
    # reference-bone near-surface points to the subject's. Default 'auto'
    # uses affine (12 DOF) for ellipsoids (which transforms cleanly to
    # another ellipsoid) and similarity (7 DOF, rigid+isotropic-scale) for
    # cylinders (where affine produces an elliptic cylinder that the
    # circular-cylinder refit can only approximate). Override to 'affine'
    # or 'similarity' to force one for both wrap types.
    label_correspondence_transform_kind = cfg("label_correspondence_transform_kind", "auto")

    anchors_by_bone = {}
    if wrap_fit_mode == "lbfgs_smith2019_anchor" and smith2019_osim_path is not None:
        from nsosim.wrap_surface_fitting.procrustes_anchor import (
            procrustes_anchors_from_smith2019,
        )

        anchors_by_bone = procrustes_anchors_from_smith2019(smith2019_osim_path)
        if wraps_to_skip_anchor:
            for bone_d in anchors_by_bone.values():
                for body_d in bone_d.values():
                    for stype_d in body_d.values():
                        for name in list(stype_d.keys()):
                            if name in wraps_to_skip_anchor:
                                del stype_d[name]

    # In label_correspondence mode we need smith2019_osim_path to read the
    # reference wrap parameters that get transformed per subject.
    if wrap_fit_mode == "label_correspondence" and smith2019_osim_path is None:
        raise ValueError(
            "wrap_fit_mode='label_correspondence' requires "
            "config['smith2019_osim_path'] to be set (source of reference "
            "wrap parameters to transform)."
        )

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
            "femur_nsm_recon_osim.vtk": fem_mesh_osim,
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
    if wrap_fit_mode == "label_correspondence":
        import pyvista as _pv  # local import; keeps top-of-file unchanged

        from nsosim.wrap_surface_fitting.label_correspondence_transform import (
            label_correspondence_transforms_for_bone,
        )

        ref_femur = _pv.read(dict_bones["femur"]["wrap"]["path_labeled_bone"])
        fitted_wrap_parameters["femur"] = label_correspondence_transforms_for_bone(
            smith2019_osim_path=smith2019_osim_path,
            bone_name="femur",
            ref_labeled_mesh=ref_femur,
            subj_labeled_mesh=fem_labeled_mesh.mesh,  # pymskt.Mesh → underlying PolyData
            transform_kind=label_correspondence_transform_kind,
        )
    else:
        fitted_wrap_parameters["femur"] = fit_bone_wrap_surfaces(
            bone_name="femur",
            labeled_mesh=fem_labeled_mesh,
            labeled_mesh_points=fem_labeled_points,
            fitter_configs=fitter_configs,
            n_restarts=wrap_n_restarts,
            jitter_scale=wrap_jitter_scale,
            anchors=anchors_by_bone.get("femur"),
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
            "tibia_nsm_recon_osim.vtk": tib_mesh_osim,
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
    if wrap_fit_mode == "label_correspondence":
        import pyvista as _pv

        from nsosim.wrap_surface_fitting.label_correspondence_transform import (
            label_correspondence_transforms_for_bone,
        )

        ref_tibia = _pv.read(dict_bones["tibia"]["wrap"]["path_labeled_bone"])
        fitted_wrap_parameters["tibia"] = label_correspondence_transforms_for_bone(
            smith2019_osim_path=smith2019_osim_path,
            bone_name="tibia",
            ref_labeled_mesh=ref_tibia,
            subj_labeled_mesh=tib_labeled_mesh.mesh,
            transform_kind=label_correspondence_transform_kind,
        )
    else:
        fitted_wrap_parameters["tibia"] = fit_bone_wrap_surfaces(
            bone_name="tibia",
            labeled_mesh=tib_labeled_mesh,
            labeled_mesh_points=tib_labeled_points,
            fitter_configs=fitter_configs,
            n_restarts=wrap_n_restarts,
            jitter_scale=wrap_jitter_scale,
            anchors=anchors_by_bone.get("tibia"),
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
    if project_coronary:
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
            "patella_nsm_recon_osim.vtk": pat_mesh_centered,
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

    fatpad_dir = os.path.join(folder_save_bones, "femur")
    fatpad_mesh.save(os.path.join(fatpad_dir, "femur_prefemoral_fat_pad.stl"))
    fatpad_mesh.save(os.path.join(fatpad_dir, "femur_prefemoral_fat_pad.vtk"))

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
