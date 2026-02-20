"""Articular surface extraction, meniscus processing, and prefemoral fat pad generation."""

import gc
import logging

import numpy as np
import pyvista as pv
from pymskt.mesh import BoneMesh, CartilageMesh, Mesh
from pymskt.mesh.meshCartilage import (
    extract_articular_surface,
    get_n_largest,
    remove_intersecting_vertices,
    remove_isolated_cells,
)

logger = logging.getLogger(__name__)

# --- Default constants ---
# Meniscus surface extraction / refinement
DEFAULT_MENISCUS_SDF_THRESHOLD = 0.1  # mm, SDF label threshold
DEFAULT_RADIAL_PERCENTILE = 95.0  # percentile for radial envelope
DEFAULT_TRIANGLE_DENSITY = 4_000_000  # target triangle density (~2.6 tri/mm²)

# Prefemoral fat pad
DEFAULT_FATPAD_BASE_MM = 0.5
DEFAULT_FATPAD_TOP_MM = 6.0
DEFAULT_MAX_DISTANCE_TO_PATELLA_MM = 30.0


# ============================================================================
# Helper functions for radial envelope refinement of meniscus surfaces
# ============================================================================


def add_polar_coordinates_about_center(mesh: pv.PolyData, center=None, theta_offset=0.0):
    """
    Add cylindrical coordinates (theta, r, y_rel) as point data to `mesh`,
    using the given `center` (or mesh.center if None).

    - theta: angle in zx-plane, arctan2(x_rel, z_rel) + theta_offset
    - r:     radial distance in zx-plane
    - y_rel: y-relative to center

    Note: Mutates ``mesh`` in-place by adding 'theta', 'r', and 'y_rel' point
    data arrays.

    Args:
        mesh: PyVista mesh to add coordinates to
        center: Center point for polar coordinates (default: mesh.center)
        theta_offset: Rotation offset for theta in radians (default: 0.0).
                     Positive rotates counterclockwise when viewed from +Y.
                     Use to avoid discontinuity at ±π cutting through tissue
                     (e.g., π/2 for medial meniscus, 0.0 for lateral).

    Returns:
        center: The center point used
    """
    if center is None:
        center = mesh.center
    pts_centered = mesh.points - center
    x_rel = pts_centered[:, 0]
    y_rel = pts_centered[:, 1]
    z_rel = pts_centered[:, 2]
    theta = np.arctan2(x_rel, z_rel) + theta_offset
    # Wrap theta to [-π, π]
    theta = np.arctan2(np.sin(theta), np.cos(theta))
    r = np.sqrt(x_rel**2 + z_rel**2)
    mesh["theta"] = theta
    mesh["r"] = r
    mesh["y_rel"] = y_rel
    return center


def label_meniscus_regions_with_sdf(
    meniscus_mesh: pv.PolyData,
    lower_surface,
    upper_surface,
    distance_thresh: float = 0.1,
    array_name: str = "regions_label",
):
    """
    Label meniscus points based on SDF distance to lower and upper surfaces.

    regions:
      0 = neither
      1 = near lower_surface (dist_lower < threshold)
      2 = near upper_surface (dist_upper < threshold)
      3 = near both

    Args:
        meniscus_mesh: Full meniscus mesh
        lower_surface: Lower articulating surface mesh
        upper_surface: Upper articulating surface mesh
        distance_thresh: Distance threshold for labeling (in mm)
        array_name: Name for the region label array

    Returns:
        regions_label: Array of region labels
    """
    pts = meniscus_mesh.points

    # Convert to Mesh objects if needed for SDF computation
    if not isinstance(lower_surface, Mesh):
        lower_surface = Mesh(lower_surface)
    if not isinstance(upper_surface, Mesh):
        upper_surface = Mesh(upper_surface)

    dist_lower = np.abs(lower_surface.get_sdf_pts(pts))
    dist_upper = np.abs(upper_surface.get_sdf_pts(pts))

    regions_label = np.zeros(meniscus_mesh.n_points, dtype=np.int8)
    regions_label[dist_lower < distance_thresh] = 1
    regions_label[dist_upper < distance_thresh] = 2
    regions_label[(dist_lower < distance_thresh) & (dist_upper < distance_thresh)] = 3

    meniscus_mesh[array_name] = regions_label.astype(float)  # pyvista likes float sometimes
    return regions_label


def compute_region_radial_percentiles(
    mesh: pv.PolyData,
    regions_array: str = "regions_label",
    percentile: float = 95.0,
    n_theta_bins: int = 100,
):
    """
    For each region label in `regions_array`, compute the given `percentile`
    of r (radial distance) in theta bins.

    Args:
        mesh: Mesh with polar coordinates and region labels
        regions_array: Name of array containing region labels
        percentile: Percentile to compute (default: 95.0)
        n_theta_bins: Number of angular bins

    Returns:
        region_percentiles: dict[region_label] -> {
            'bin_centers': (B,),
            'r_percentile': (B,)
        }
        theta_bins: The bin edges used

    Note: Assumes `mesh['theta']` and `mesh['r']` already exist.
    """
    theta = mesh["theta"]
    r = mesh["r"]
    regions = mesh[regions_array]

    theta_bins = np.linspace(theta.min(), theta.max(), n_theta_bins)
    region_percentiles = {}

    for region_label in np.unique(regions):
        # skip background (0)
        if region_label == 0:
            continue

        region_mask = regions == region_label
        theta_region = theta[region_mask]
        r_region = r[region_mask]

        bin_indices = np.digitize(theta_region, theta_bins)
        r_p = []
        bin_centers = []

        for i in range(1, len(theta_bins)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                r_p.append(np.percentile(r_region[bin_mask], percentile))
                bin_centers.append(0.5 * (theta_bins[i - 1] + theta_bins[i]))

        if bin_centers:
            region_percentiles[region_label] = {
                "bin_centers": np.asarray(bin_centers),
                "r_percentile": np.asarray(r_p),
            }

    return region_percentiles, theta_bins


def smooth_1d(y, window_size=7):
    """
    Simple moving-average smoother for 1D arrays.

    Args:
        y: 1D array to smooth
        window_size: Window size (should be odd, > 1)

    Returns:
        Smoothed array
    """
    if window_size <= 1:
        return y
    kernel = np.ones(window_size, dtype=float) / float(window_size)
    return np.convolve(y, kernel, mode="same")


def build_min_radial_envelope(
    region_percentiles: dict,
    smooth_window: int = 7,
    theta_min: float = None,
    theta_max: float = None,
    n_theta_grid: int = 200,
):
    """
    From per-region percentile curves, build a single 'min envelope':
        r_min(theta) = min_over_regions r_percentile_region(theta)

    Steps:
      1. Smooth each region's r_percentile over theta.
      2. Interpolate all to a common theta_grid.
      3. Take elementwise minimum.

    Args:
        region_percentiles: Dict from compute_region_radial_percentiles
        smooth_window: Window size for smoothing
        theta_min: Minimum theta value (auto-detect if None)
        theta_max: Maximum theta value (auto-detect if None)
        n_theta_grid: Number of points in output grid

    Returns:
        theta_grid: Angular grid
        r_min_grid: Minimum radial envelope
    """
    # Determine theta range if not provided
    all_theta = []
    for pdata in region_percentiles.values():
        all_theta.append(pdata["bin_centers"])
    all_theta = np.concatenate(all_theta)

    if theta_min is None:
        theta_min = float(all_theta.min())
    if theta_max is None:
        theta_max = float(all_theta.max())

    theta_grid = np.linspace(theta_min, theta_max, n_theta_grid)
    r_min_grid = np.full_like(theta_grid, np.inf, dtype=float)

    for region_label, pdata in region_percentiles.items():
        bc = pdata["bin_centers"]
        rp = pdata["r_percentile"]

        rp_smooth = smooth_1d(rp, window_size=smooth_window)
        r_interp = np.interp(theta_grid, bc, rp_smooth, left=np.nan, right=np.nan)

        valid = ~np.isnan(r_interp)
        r_min_grid[valid] = np.minimum(r_min_grid[valid], r_interp[valid])

    # where nothing was valid, set to nan
    r_min_grid[~np.isfinite(r_min_grid)] = np.nan

    return theta_grid, r_min_grid


def mask_points_by_radial_envelope(
    mesh: pv.PolyData,
    center,
    theta_grid,
    r_thresh_grid,
    theta_offset=0.0,
):
    """
    Compute theta, r for `mesh` about `center`, and return:
        keep_mask: boolean array of points where r <= r_thresh(theta)
        theta, r: per-point values (for debugging / plotting)

    Args:
        mesh: Mesh to filter
        center: Center point for polar coordinates
        theta_grid: Angular grid for envelope
        r_thresh_grid: Radial threshold values at each theta
        theta_offset: Rotation offset for theta (must match envelope computation)

    Returns:
        keep: Boolean mask of points to keep
        theta: Angular coordinate for each point
        r: Radial coordinate for each point
    """
    pts_centered = mesh.points - center
    x_rel = pts_centered[:, 0]
    y_rel = pts_centered[:, 1]
    z_rel = pts_centered[:, 2]

    theta = np.arctan2(x_rel, z_rel) + theta_offset
    # Wrap theta to [-π, π]
    theta = np.arctan2(np.sin(theta), np.cos(theta))
    r = np.sqrt(x_rel**2 + z_rel**2)

    r_cutoff = np.interp(theta, theta_grid, r_thresh_grid, left=np.nan, right=np.nan)

    keep = (r <= r_cutoff) & ~np.isnan(r_cutoff)

    return keep, theta, r


def trim_mesh_by_radial_envelope(
    mesh: pv.PolyData,
    center,
    theta_grid,
    r_thresh_grid,
    theta_offset=0.0,
    adjacent_cells: bool = True,
):
    """
    Return a new mesh that only includes points with r <= r_thresh(theta)
    as defined by (theta_grid, r_thresh_grid).

    Args:
        mesh: Mesh to trim
        center: Center point for polar coordinates
        theta_grid: Angular grid for envelope
        r_thresh_grid: Radial threshold values at each theta
        theta_offset: Rotation offset for theta (must match envelope computation)
        adjacent_cells: Whether to include adjacent cells

    Returns:
        Trimmed mesh
    """
    keep, theta, r = mask_points_by_radial_envelope(
        mesh, center, theta_grid, r_thresh_grid, theta_offset=theta_offset
    )
    return mesh.extract_points(keep, adjacent_cells=adjacent_cells)


def refine_meniscus_articular_surfaces(
    meniscus_mesh: pv.PolyData,
    lower_surface: pv.PolyData,
    upper_surface: pv.PolyData,
    center=None,
    distance_thresh: float = DEFAULT_MENISCUS_SDF_THRESHOLD,
    percentile: float = DEFAULT_RADIAL_PERCENTILE,
    n_theta_bins: int = 100,
    smooth_window: int = 7,
    n_theta_grid: int = 200,
    theta_offset: float = 0.0,
):
    """
    Full pipeline for refining meniscus articular surfaces using radial envelope.

    Steps:
      1. Add polar coordinates to meniscus_mesh about center point.
      2. Label regions via SDF to lower/upper surfaces.
      3. Compute radial percentile envelopes by region.
      4. Build a single min radial envelope across regions 1 and 2.
      5. Trim upper and lower surfaces using this envelope.

    Args:
        meniscus_mesh: Full meniscus mesh
        lower_surface: Lower articulating surface (initial extraction)
        upper_surface: Upper articulating surface (initial extraction)
        center: Center point for polar coordinates (if None, uses meniscus centroid).
                For diseased/partial menisci, provide a consistent center point
                (e.g., based on tibial plateau center).
        distance_thresh: Distance threshold for SDF labeling (mm)
        percentile: Percentile for radial envelope (e.g., 95.0)
        n_theta_bins: Number of angular bins for percentile computation
        smooth_window: Window size for smoothing radial envelope
        n_theta_grid: Grid resolution for radial envelope interpolation
        theta_offset: Rotation offset for polar coordinates in radians (default: 0.0).
                     Use to avoid discontinuity cutting through tissue.
                     Typically: π/2 for medial meniscus, 0.0 for lateral meniscus.

    Returns:
        lower_surface_trimmed: Refined lower surface
        upper_surface_trimmed: Refined upper surface
        (theta_grid, r_min_grid): The radial envelope used
    """
    # 1. Polar coords - use provided center or default to meniscus centroid
    center = add_polar_coordinates_about_center(
        meniscus_mesh, center=center, theta_offset=theta_offset
    )

    # 2. Region labels
    label_meniscus_regions_with_sdf(
        meniscus_mesh,
        lower_surface,
        upper_surface,
        distance_thresh=distance_thresh,
        array_name="regions_label",
    )

    # 3. Percentile envelopes per region
    region_percentiles, _ = compute_region_radial_percentiles(
        meniscus_mesh,
        regions_array="regions_label",
        percentile=percentile,
        n_theta_bins=n_theta_bins,
    )

    # 4. Combined min envelope
    theta_grid, r_min_grid = build_min_radial_envelope(
        region_percentiles,
        smooth_window=smooth_window,
        n_theta_grid=n_theta_grid,
    )

    # 5. Trim articular surfaces (with same theta_offset used to compute envelope)
    lower_trimmed = trim_mesh_by_radial_envelope(
        lower_surface, center, theta_grid, r_min_grid, theta_offset=theta_offset
    )
    upper_trimmed = trim_mesh_by_radial_envelope(
        upper_surface, center, theta_grid, r_min_grid, theta_offset=theta_offset
    )

    return lower_trimmed, upper_trimmed, (theta_grid, r_min_grid)


# ============================================================================
# Scored meniscus surface extraction (Method 1 — alternative to ray-casting)
# ============================================================================

# Default parameters for scored extraction
DEFAULT_SCORE_THRESHOLD = 0.3
DEFAULT_SCORE_SMOOTH_ITERATIONS = 20


def _smooth_scores_on_mesh(mesh, scores, n_iter=20):
    """Average score values over mesh vertex neighbors without moving geometry.

    PyVista's smooth() moves vertices AND interpolates point data, which is
    undesirable — we want to smooth only the score field. This function builds
    a vertex adjacency structure from mesh faces and iteratively averages
    each vertex's score with its neighbors.

    Args:
        mesh: PyVista PolyData mesh (used for connectivity only)
        scores: 1D array of per-vertex scores
        n_iter: Number of smoothing iterations

    Returns:
        Smoothed scores (same shape as input)
    """
    from collections import defaultdict

    # Build adjacency from faces
    faces = mesh.faces
    adj = defaultdict(set)
    i = 0
    while i < len(faces):
        n_verts = faces[i]
        face_verts = faces[i + 1 : i + 1 + n_verts]
        for vi in range(n_verts):
            for vj in range(vi + 1, n_verts):
                adj[face_verts[vi]].add(face_verts[vj])
                adj[face_verts[vj]].add(face_verts[vi])
        i += 1 + n_verts

    smoothed = scores.copy().astype(float)
    for _ in range(n_iter):
        new_scores = smoothed.copy()
        for v in range(len(smoothed)):
            neighbors = adj.get(v, set())
            if neighbors:
                neighbor_vals = smoothed[list(neighbors)]
                new_scores[v] = (smoothed[v] + neighbor_vals.sum()) / (1 + len(neighbors))
        smoothed = new_scores

    return smoothed


def score_meniscus_vertices(meniscus_mesh, bone_mesh):
    """Score each meniscus vertex by how much its normal faces the bone.

    For each vertex, computes dot(outward_normal, direction_to_closest_bone_point).
    High score → normal points toward bone → flat articular face.
    Low/negative score → normal perpendicular or away from bone → rim/edge.

    This is a continuous alternative to binary ray-casting that is robust to
    small vertex perturbations (small normal changes → small score changes,
    not binary flips).

    Args:
        meniscus_mesh: PyVista PolyData of the meniscus (must have point normals)
        bone_mesh: PyVista PolyData of the articulating bone

    Returns:
        scores: 1D array of per-vertex scores in [-1, 1]
    """
    from scipy.spatial import KDTree

    normals = meniscus_mesh.point_normals
    points = meniscus_mesh.points

    bone_tree = KDTree(bone_mesh.points)
    _, indices = bone_tree.query(points)
    closest_bone_pts = bone_mesh.points[indices]

    direction = closest_bone_pts - points
    direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
    direction_norm[direction_norm == 0] = 1  # avoid div by zero
    direction = direction / direction_norm

    scores = np.sum(normals * direction, axis=1)
    return scores


def extract_meniscus_articulating_surface_scored(
    meniscus_mesh: pv.PolyData,
    articulating_bone_mesh: pv.PolyData,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    smooth_score: bool = True,
    smooth_iterations: int = DEFAULT_SCORE_SMOOTH_ITERATIONS,
    n_largest: int = 1,
    smooth_iter: int = 15,
    boundary_smoothing: bool = False,
) -> pv.PolyData:
    """Extract meniscus articular surface using continuous normal-dot-product scoring.

    Alternative to `extract_meniscus_articulating_surface` (ray-casting).
    Uses a continuous scoring function instead of binary ray-casting:
    1. Compute outward normals on the meniscus
    2. For each vertex, score = dot(normal, direction_to_closest_bone_point)
    3. Smooth scores on the mesh surface (neighbor averaging, no vertex movement)
    4. Threshold: keep vertices with score > score_threshold
    5. Standard cleanup: largest component, remove isolated cells, smooth

    This method is robust to small vertex perturbations because the score
    changes continuously (0.29 → 0.31) rather than flipping binary (hit → miss).

    Args:
        meniscus_mesh: The meniscus mesh (PolyData)
        articulating_bone_mesh: The bone mesh to articulate with
        score_threshold: Minimum score to include vertex (default: 0.3).
            Higher values are more selective (only flat faces).
            Range: typically 0.2–0.5.
        smooth_score: Whether to smooth scores on the mesh surface (default: True)
        smooth_iterations: Number of neighbor-averaging iterations for score smoothing
        n_largest: Number of largest connected components to keep
        smooth_iter: Number of Laplacian smoothing iterations for final surface
        boundary_smoothing: Whether to fix boundaries during final smoothing

    Returns:
        Mesh: The extracted articular surface
    """
    # Ensure normals are computed
    meniscus_mesh.compute_normals(
        point_normals=True, auto_orient_normals=True, inplace=True
    )

    # 1. Compute scores
    scores = score_meniscus_vertices(meniscus_mesh, articulating_bone_mesh)

    # 2. Smooth scores on mesh surface (without moving geometry)
    if smooth_score:
        scores = _smooth_scores_on_mesh(meniscus_mesh, scores, n_iter=smooth_iterations)

    # 3. Threshold
    mask = scores > score_threshold
    if not np.any(mask):
        logger.warning(
            f"No vertices above score threshold {score_threshold}. "
            f"Score range: [{scores.min():.3f}, {scores.max():.3f}]. "
            f"Trying with threshold=0.0."
        )
        mask = scores > 0.0

    surface = meniscus_mesh.extract_points(mask, adjacent_cells=True)

    # 4. Ensure PolyData
    if not isinstance(surface, pv.PolyData):
        surface = surface.extract_surface()

    # 5. Largest component
    surface = get_n_largest(surface, n=n_largest)
    if not isinstance(surface, pv.PolyData):
        surface = surface.extract_surface()

    # 6. Remove isolated cells
    surface = remove_isolated_cells(surface)
    if not isinstance(surface, pv.PolyData):
        raise TypeError(
            f"Expected pv.PolyData after remove_isolated_cells, got {type(surface)}"
        )

    # 7. Final smooth
    surface = surface.smooth(n_iter=smooth_iter, boundary_smoothing=boundary_smoothing)

    return Mesh(surface)


# ============================================================================
# Main meniscus surface extraction functions
# ============================================================================


def extract_meniscus_articulating_surface(
    meniscus_mesh: pv.PolyData,
    articulating_bone_mesh: pv.PolyData,
    ray_length: float = 10.0,
    n_largest: int = 1,
    smooth_iter: int = 15,
    boundary_smoothing: bool = False,  # False allows boundaries to be smoothed
) -> pv.PolyData:
    """
    Extracts and processes an articulating surface of a meniscus.

    This function first defines the meniscus surface based on proximity to an
    articulating bone (e.g., femur for superior surface, tibia for inferior).
    It then applies post-processing steps:
    1. Keeps the largest connected component of the resulting surface.
    2. Ensures the result is a surface mesh (PolyData).
    3. Removes isolated cells/vertices.
    4. Smoothes the final surface.

    Args:
        meniscus_mesh: The pv.PolyData object of the meniscus (e.g., medial or lateral meniscus).
        articulating_bone_mesh: The pv.PolyData object of the bone it articulates with
                                (e.g., femur for superior surface, tibia for inferior surface).
        ray_length: Length of rays for `remove_intersecting_vertices`. Rays are projected inward
                   from the `articulating_bone_mesh` to define the surface on the `meniscus_mesh`.
        n_largest: The number of largest connected components to keep after initial processing.
                   Typically 1.
        smooth_iter: Number of smoothing iterations for the `smooth` filter.
        boundary_smoothing: Argument for PyVista's `smooth` method. If False (default here),
                            boundaries are allowed to be smoothed. If True, boundaries are fixed.

    Returns:
        pv.PolyData: The processed articulating surface of the meniscus.
    """

    # Step 1: Define the initial articulating surface using remove_intersecting_vertices
    # This function projects rays from the articulating_bone_mesh to the meniscus_mesh
    # to define the contact interface.
    # (Assuming remove_intersecting_vertices is available)
    surface = remove_intersecting_vertices(
        mesh1=meniscus_mesh,  # The mesh to be clipped/modified (meniscus)
        mesh2=articulating_bone_mesh,  # The mesh used as the reference/clipper (femur/tibia)
        ray_length=-ray_length,
    )
    if not isinstance(surface, pv.PolyData):
        # This assertion helps catch issues early if remove_intersecting_vertices
        # doesn't return the expected type.
        raise TypeError(
            f"Expected pv.PolyData from remove_intersecting_vertices, got {type(surface)}"
        )

    # Step 2: Keep the n-largest connected components
    # (Assuming get_n_largest is available)
    processed_surface = get_n_largest(surface, n=n_largest)

    # Step 3: Ensure the result is a surface mesh (PolyData)
    # get_n_largest might return an UnstructuredGrid, so extract surface if necessary.
    if not isinstance(processed_surface, pv.PolyData):
        processed_surface = processed_surface.extract_surface()

    if not isinstance(processed_surface, pv.PolyData):
        raise TypeError(
            f"Expected pv.PolyData after get_n_largest and extract_surface, got {type(processed_surface)}"
        )

    # Step 4: Remove isolated cells
    # This helps clean up small artifacts or disconnected parts of the mesh.
    # (Assuming remove_isolated_cells is available)
    processed_surface = remove_isolated_cells(processed_surface)
    if not isinstance(processed_surface, pv.PolyData):
        raise TypeError(
            f"Expected pv.PolyData after remove_isolated_cells, got {type(processed_surface)}"
        )

    # Step 5: Smooth the surface
    # The boundary_smoothing=False argument allows the edges of the surface to be smoothed.
    processed_surface = processed_surface.smooth(
        n_iter=smooth_iter, boundary_smoothing=boundary_smoothing
    )

    if not isinstance(processed_surface, pv.PolyData):
        raise TypeError(f"Expected pv.PolyData after smooth, got {type(processed_surface)}")

    return Mesh(processed_surface)


def create_meniscus_articulating_surface(
    meniscus_mesh,
    upper_articulating_bone_mesh,
    lower_articulating_bone_mesh,
    ray_length=10.0,
    n_largest=1,
    smooth_iter=50,
    boundary_smoothing=False,
    triangle_density=DEFAULT_TRIANGLE_DENSITY,  # ~2.6 triangles/mm^2
    meniscus_clusters=None,
    upper_articulating_bone_clusters=None,
    lower_articulating_bone_clusters=None,
    # Radial envelope refinement parameters
    meniscus_center=None,  # if None, will use meniscus centroid
    refine_by_radial_envelope=True,
    distance_thresh=DEFAULT_MENISCUS_SDF_THRESHOLD,  # mm for SDF labeling
    radial_percentile=DEFAULT_RADIAL_PERCENTILE,
    n_theta_bins=100,
    smooth_window=7,
    n_theta_grid=200,
    theta_offset=0.0,  # radians, rotate polar coords to avoid discontinuity
    # Extraction method selection
    extraction_method="ray_casting",  # "ray_casting" or "scored"
    score_threshold=DEFAULT_SCORE_THRESHOLD,
    score_smooth_iterations=DEFAULT_SCORE_SMOOTH_ITERATIONS,
):
    """
    Creates meniscus articulating surfaces from a meniscus mesh and two articulating bone meshes.

    This function performs the following steps:
      1. Computes density metrics and resampling parameters
      2. Converts meshes from meters to mm for processing
      3. Resamples meshes as specified
      4. Extracts initial upper and lower articulating surfaces
      5. (Optional) Refines surfaces using radial envelope filtering
      6. Converts results back to meters

    Args:
        meniscus_mesh: Full meniscus mesh (pymskt.mesh.Mesh)
        upper_articulating_bone_mesh: Upper bone mesh (femur) for defining superior surface
        lower_articulating_bone_mesh: Lower bone mesh (tibia) for defining inferior surface
        ray_length: Ray casting length for surface extraction (mm). Only used when
            extraction_method="ray_casting".
        n_largest: Number of largest connected components to keep
        smooth_iter: Smoothing iterations for extracted surfaces
        boundary_smoothing: Whether to fix boundaries during smoothing
        triangle_density: Target triangle density for resampling (triangles/m^2)
        meniscus_clusters: Number of clusters for meniscus resampling
        upper_articulating_bone_clusters: Clusters for upper bone resampling
        lower_articulating_bone_clusters: Clusters for lower bone resampling
        meniscus_center: Center point for radial envelope (if None, uses meniscus centroid)
        refine_by_radial_envelope: Whether to apply radial envelope refinement (default: True)
        distance_thresh: Distance threshold (mm) for SDF-based region labeling
        radial_percentile: Percentile for radial envelope (e.g., 95.0)
        n_theta_bins: Number of angular bins for percentile computation
        smooth_window: Window size for smoothing radial envelope
        n_theta_grid: Grid resolution for radial envelope interpolation
        theta_offset: Rotation offset for polar coordinates in radians (default: 0.0).
                     Use to avoid polar discontinuity cutting through meniscus tissue.
                     Typically: np.pi/2 for medial meniscus, 0.0 for lateral meniscus.
        extraction_method: Method for initial surface extraction (default: "ray_casting").
            "ray_casting": Binary ray-casting via remove_intersecting_vertices.
                          Uses ray_length parameter.
            "scored": Normal dot-product scoring — continuous alternative.
                      Uses score_threshold and score_smooth_iterations.
        score_threshold: Minimum normal-dot-product score to include vertex (default: 0.3).
            Only used when extraction_method="scored". Range: 0.2–0.5.
        score_smooth_iterations: Number of neighbor-averaging iterations for score
            smoothing (default: 20). Only used when extraction_method="scored".

    Returns:
        upper_meniscus_articulating_surface: Refined superior surface (pymskt.mesh.Mesh)
        lower_meniscus_articulating_surface: Refined inferior surface (pymskt.mesh.Mesh)
    """
    if extraction_method not in ("scored", "ray_casting"):
        raise ValueError(
            f"extraction_method must be 'scored' or 'ray_casting', got '{extraction_method}'"
        )

    # ============================================================================
    # STEP 1: Compute density metrics for resampling (in meters space)
    # ============================================================================
    if triangle_density is not None:
        meniscus_mesh.compute_normals(
            point_normals=True, cell_normals=False, auto_orient_normals=True, inplace=True
        )
        # compute the triangle density of the cart mesh now
        current_density = meniscus_mesh.n_cells / meniscus_mesh.area
        # downsample factor
        downsample_factor = current_density / triangle_density
        meniscus_clusters = int(meniscus_mesh.point_coords.shape[0] / downsample_factor)

        logger.info(f"current density: {current_density/1_000_000} triangles/mm^2")
        logger.info(f"target density: {triangle_density/1_000_000} triangles/mm^2")
        logger.info(f"current number of points: {meniscus_mesh.point_coords.shape[0]}")
        logger.info(f"target number of points: {meniscus_clusters}")

    # ============================================================================
    # STEP 2: Convert meshes to mm (instead of meters)
    # ============================================================================
    meniscus_mesh_ = meniscus_mesh.copy()
    meniscus_mesh_.point_coords = meniscus_mesh_.point_coords * 1000
    upper_articulating_bone_mesh_ = upper_articulating_bone_mesh.copy()
    upper_articulating_bone_mesh_.point_coords = upper_articulating_bone_mesh_.point_coords * 1000
    lower_articulating_bone_mesh_ = lower_articulating_bone_mesh.copy()
    lower_articulating_bone_mesh_.point_coords = lower_articulating_bone_mesh_.point_coords * 1000

    # Handle meniscus_center conversion to mm if provided
    if meniscus_center is not None:
        meniscus_center_mm = np.array(meniscus_center) * 1000
    else:
        meniscus_center_mm = None

    # ============================================================================
    # STEP 3: Resample meshes (in mm space)
    # ============================================================================
    if meniscus_clusters is not None:
        meniscus_mesh_.resample_surface(subdivisions=1, clusters=meniscus_clusters)
        # density after resampling
        updated_density = meniscus_mesh_.n_cells / meniscus_mesh_.area
        logger.info(f"achieved density: {updated_density/1_000_000} triangles/mm^2")

    # resample bones if specified
    if upper_articulating_bone_clusters is not None:
        upper_articulating_bone_mesh_.resample_surface(
            subdivisions=1, clusters=upper_articulating_bone_clusters
        )
    if lower_articulating_bone_clusters is not None:
        lower_articulating_bone_mesh_.resample_surface(
            subdivisions=1, clusters=lower_articulating_bone_clusters
        )

    # ============================================================================
    # STEP 4: Extract initial articulating surfaces
    # ============================================================================
    if extraction_method == "scored":
        logger.info("Extracting surfaces using scored method (normal dot-product)...")
        # extract upper articulating surface of the meniscus
        upper_meniscus_articulating_surface = extract_meniscus_articulating_surface_scored(
            meniscus_mesh_,
            upper_articulating_bone_mesh_,
            score_threshold=score_threshold,
            smooth_score=True,
            smooth_iterations=score_smooth_iterations,
            n_largest=n_largest,
            smooth_iter=smooth_iter,
            boundary_smoothing=boundary_smoothing,
        )

        # extract lower articulating surface of the meniscus
        lower_meniscus_articulating_surface = extract_meniscus_articulating_surface_scored(
            meniscus_mesh_,
            lower_articulating_bone_mesh_,
            score_threshold=score_threshold,
            smooth_score=True,
            smooth_iterations=score_smooth_iterations,
            n_largest=n_largest,
            smooth_iter=smooth_iter,
            boundary_smoothing=boundary_smoothing,
        )
    else:
        logger.info("Extracting surfaces using ray-casting method...")
        # extract upper articulating surface of the meniscus
        upper_meniscus_articulating_surface = extract_meniscus_articulating_surface(
            meniscus_mesh_,
            upper_articulating_bone_mesh_,
            ray_length=ray_length,
            n_largest=n_largest,
            smooth_iter=smooth_iter,
            boundary_smoothing=boundary_smoothing,
        )

        # extract lower articulating surface of the meniscus
        lower_meniscus_articulating_surface = extract_meniscus_articulating_surface(
            meniscus_mesh_,
            lower_articulating_bone_mesh_,
            ray_length=ray_length,
            n_largest=n_largest,
            smooth_iter=smooth_iter,
            boundary_smoothing=boundary_smoothing,
        )

    # ============================================================================
    # STEP 5: Refine surfaces using radial envelope filtering
    # ============================================================================
    if refine_by_radial_envelope:
        logger.info("Refining meniscus surfaces with radial envelope filtering...")

        # Apply refinement
        lower_refined, upper_refined, (theta_grid, r_min_grid) = refine_meniscus_articular_surfaces(
            meniscus_mesh=meniscus_mesh_.copy(),
            lower_surface=lower_meniscus_articulating_surface.copy(),
            upper_surface=upper_meniscus_articulating_surface.copy(),
            center=meniscus_center_mm,  # Use provided center or None (defaults to centroid)
            distance_thresh=distance_thresh,
            percentile=radial_percentile,
            n_theta_bins=n_theta_bins,
            smooth_window=smooth_window,
            n_theta_grid=n_theta_grid,
            theta_offset=theta_offset,  # Rotate polar coords to avoid discontinuity
        )

        # Final cleanup: get largest component and remove isolated cells
        logger.info("Final cleanup of refined surfaces...")

        # Ensure results are PolyData before passing to get_n_largest
        # (trim_mesh_by_radial_envelope uses extract_points which can return UnstructuredGrid)
        if not isinstance(upper_refined, pv.PolyData):
            upper_refined = upper_refined.extract_surface()
        if not isinstance(lower_refined, pv.PolyData):
            lower_refined = lower_refined.extract_surface()

        # Upper surface cleanup
        upper_refined = get_n_largest(upper_refined, n=n_largest)
        if not isinstance(upper_refined, pv.PolyData):
            upper_refined = upper_refined.extract_surface()
        upper_refined = remove_isolated_cells(upper_refined)

        # Lower surface cleanup
        lower_refined = get_n_largest(lower_refined, n=n_largest)
        if not isinstance(lower_refined, pv.PolyData):
            lower_refined = lower_refined.extract_surface()
        lower_refined = remove_isolated_cells(lower_refined)

        # Convert back to Mesh objects
        upper_meniscus_articulating_surface = Mesh(upper_refined)
        lower_meniscus_articulating_surface = Mesh(lower_refined)

    # ============================================================================
    # STEP 6: Convert back to meters and return
    # ============================================================================
    upper_meniscus_articulating_surface.point_coords = (
        upper_meniscus_articulating_surface.point_coords / 1000
    )
    lower_meniscus_articulating_surface.point_coords = (
        lower_meniscus_articulating_surface.point_coords / 1000
    )

    return upper_meniscus_articulating_surface, lower_meniscus_articulating_surface


def create_articular_surfaces(
    bone_mesh_osim,
    cart_mesh_osim,
    n_largest=1,
    bone_clusters=None,
    cart_clusters=None,
    triangle_density=DEFAULT_TRIANGLE_DENSITY,  # ~2.6 triangles/mm^2
    ray_length=10,
    smooth_iter=100,
):
    """
    Creates an articular surface from a bone and a cartilage mesh.

    This function processes a given bone and cartilage mesh to extract the
    articular surface of the cartilage. Steps include:
    1.  Optionally resampling the cartilage mesh to a target triangle density.
    2.  Resampling the bone mesh (if `bone_clusters` is specified).
    3.  Assigning the (potentially resampled) cartilage mesh to the bone mesh.
    4.  Extracting the articular surface using `extract_articular_surface`.
    5.  Scaling the resulting articular surface points from mm to meters.

    Args:
        bone_mesh_osim (pymskt.mesh.Mesh): The bone mesh.
        cart_mesh_osim (pymskt.mesh.Mesh): The cartilage mesh.
        n_largest (int, optional): Number of largest connected components to keep.
            Defaults to 1.
        bone_clusters (int, optional): Target number of points for resampling the
            bone mesh. If None, bone mesh is not resampled. Defaults to None.
        cart_clusters (int, optional): Target number of points for resampling the
            cartilage mesh. If `triangle_density` is also provided, this value
            will be calculated and override this argument. Defaults to None.
        triangle_density (float, optional): Target triangle density (triangles/mm^2)
            for the cartilage mesh. If provided, `cart_clusters` will be calculated
            based on this. Defaults to 4,000,000.
        ray_length (float, optional): Ray length for `extract_articular_surface`.
            Defaults to 10.
        smooth_iter (int, optional): Smoothing iterations for `extract_articular_surface`.
            Defaults to 100.

    Returns:
        pyvista.PolyData: The extracted and processed articular surface, scaled to meters.
    """

    # UPDATE TO CHECK RANGE OF DENSITIES AND MESH SIZE
    # MAKE SURE THEY MATCH, OR RAISE WARNING.

    if triangle_density is not None:
        if not isinstance(cart_mesh_osim.mesh, pv.PolyData):
            cart_mesh_osim.mesh = pv.PolyData(cart_mesh_osim.mesh)
            cart_mesh_osim.mesh.compute_normals(
                point_normals=True, cell_normals=False, auto_orient_normals=True, inplace=True
            )
        # compute the triangle density of the cart mesh now
        current_density = cart_mesh_osim.mesh.n_cells / cart_mesh_osim.mesh.area
        # downsample factor
        downsample_factor = current_density / triangle_density
        # calculate target number of points
        cart_clusters = int(cart_mesh_osim.point_coords.shape[0] / downsample_factor)

        logger.info(f"current density: {current_density/1_000_000} triangles/mm^2")
        logger.info(f"target density: {triangle_density/1_000_000} triangles/mm^2")
        logger.info(f"current number of points: {cart_mesh_osim.point_coords.shape[0]}")
        logger.info(f"target number of points: {cart_clusters}")

    # resample bone surface to be 10k points to reduce computation time
    logger.info("resampling femur surface")
    bone_mesh_osim_ = bone_mesh_osim.copy()
    bone_mesh_osim_.point_coords = bone_mesh_osim_.point_coords * 1000
    if bone_clusters is not None:
        bone_mesh_osim_.resample_surface(subdivisions=1, clusters=bone_clusters)
    bone_mesh_osim_ = BoneMesh(bone_mesh_osim_)

    # assign cartilage to bone
    logger.info("resample cartilage surface")
    cart_mesh_osim_ = cart_mesh_osim.copy()
    cart_mesh_osim_.point_coords = cart_mesh_osim_.point_coords * 1000
    if cart_clusters is not None:
        # if not isinstance(cart_mesh_osim_.mesh, pv.PolyData):
        #     cart_mesh_osim_.mesh = pv.PolyData(cart_mesh_osim_.mesh)
        # cart_mesh_osim_.mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True, inplace=True)
        cart_mesh_osim_.resample_surface(subdivisions=1, clusters=cart_clusters)
        # density after resampling
        updated_density = cart_mesh_osim_.mesh.n_cells / cart_mesh_osim_.mesh.area
        logger.info(f"achieved density: {updated_density/1_000_000} triangles/mm^2")
    cart_mesh_osim_.compute_normals(
        point_normals=True, cell_normals=False, auto_orient_normals=True, inplace=True
    )
    bone_mesh_osim_.list_cartilage_meshes = [CartilageMesh(cart_mesh_osim_)]

    # extract articular surface
    logger.info("extracting articular surface")
    articular_surfaces = extract_articular_surface(
        bone_mesh_osim_, ray_length=ray_length, smooth_iter=smooth_iter, n_largest=n_largest
    )[0]

    articular_surfaces.points = articular_surfaces.points / 1000

    return articular_surfaces


def compute_overlap_metrics(pat_articular_surfaces, fem_articular_surfaces):
    """
    Computes overlap metrics between patellar and femoral articular surfaces.

    Calculates:
    1.  Percentage of patellar surface points that are close to the femoral surface
        (distance > 0 after ray casting).
    2.  Absolute vertical overlap (in the y-direction) between the two surfaces.
    3.  Total vertical extent of the patellar articular surface.
    4.  Percentage of vertical overlap relative to the total vertical extent of
        the patellar surface.

    Args:
        pat_articular_surfaces (pymskt.mesh.Mesh or pyvista.PolyData):
            The patellar articular surface.
        fem_articular_surfaces (pymskt.mesh.Mesh or pyvista.PolyData):
            The femoral articular surface.

    Returns:
        tuple: A tuple containing:
            - percent_non_zero (float): Percentage of patellar surface not overlapping.
            - vert_overlap (float): Absolute vertical overlap (y-direction).
            - total_vert (float): Total vertical extent of patellar surface.
            - percent_vert_overlap (float): Percentage vertical overlap.
    """
    if not isinstance(pat_articular_surfaces, Mesh):
        pat_articular_surfaces = Mesh(pat_articular_surfaces)

    pat_articular_surfaces.calc_distance_to_other_mesh(
        fem_articular_surfaces,
        ray_cast_length=1 / 100,
        percent_ray_length_opposite_direction=1.0,
        name="fem_cart_dist",
    )
    thickness = pat_articular_surfaces.get_scalar("fem_cart_dist")
    n_non_zero = np.sum(thickness > 0)
    percent_non_zero = n_non_zero / len(thickness)

    # get the y coordinates of patella points overlapping with femur
    overlap = np.where(thickness > 0)[0]
    overlap_y = pat_articular_surfaces.point_coords[overlap, 1]

    vert_overlap = np.max(overlap_y) - np.min(overlap_y)
    total_vert = np.max(pat_articular_surfaces.point_coords[:, 1]) - np.min(
        pat_articular_surfaces.point_coords[:, 1]
    )
    percent_vert_overlap = vert_overlap / total_vert

    return percent_non_zero, vert_overlap, total_vert, percent_vert_overlap


def _type_check_mesh(mesh, mesh_name="mesh"):
    """
    Type check and convert mesh to pymskt.mesh.Mesh if needed.

    Args:
        mesh: Input mesh (pymskt.mesh.Mesh, pyvista.PolyData, or path)
        mesh_name: Name of the mesh for error messages

    Returns:
        pymskt.mesh.Mesh: Validated mesh object
    """
    if not isinstance(mesh, Mesh):
        if isinstance(mesh, pv.PolyData):
            mesh = Mesh(mesh)
        else:
            raise TypeError(f"{mesh_name} is not a pv.PolyData or pymskt.mesh.Mesh: {type(mesh)}")
    return mesh


def _prepare_fatpad_input_meshes(
    femur_bone_mesh, femur_cart_mesh, patella_bone_mesh, patella_cart_mesh, units
):
    """
    Type check, copy, and convert meshes to mm for processing.

    Args:
        femur_bone_mesh: Femur bone mesh
        femur_cart_mesh: Femur cartilage mesh
        patella_bone_mesh: Patella bone mesh
        patella_cart_mesh: Patella cartilage mesh
        units: 'm' or 'mm' - units of input meshes

    Returns:
        tuple: (femur_bone_mesh, femur_cart_mesh, patella_bone_mesh, patella_cart_mesh) in mm
    """
    # Type check all meshes
    femur_bone_mesh = _type_check_mesh(femur_bone_mesh, "femur_bone_mesh")
    femur_cart_mesh = _type_check_mesh(femur_cart_mesh, "femur_cart_mesh")
    patella_bone_mesh = _type_check_mesh(patella_bone_mesh, "patella_bone_mesh")
    patella_cart_mesh = _type_check_mesh(patella_cart_mesh, "patella_cart_mesh")

    # Copy the meshes to ensure they are not modified in place
    femur_bone_mesh = femur_bone_mesh.copy()
    femur_cart_mesh = femur_cart_mesh.copy()
    patella_bone_mesh = patella_bone_mesh.copy()
    patella_cart_mesh = patella_cart_mesh.copy()

    # Convert to mm if needed
    if units == "m":
        logger.info("Converting meshes to mm...")
        femur_bone_mesh.points *= 1000
        femur_cart_mesh.points *= 1000
        patella_cart_mesh.points *= 1000
        patella_bone_mesh.points *= 1000
    elif units == "mm":
        logger.info("Meshes are already in mm...")
    else:
        raise ValueError(f'Invalid units: {units}, expected "m" or "mm"')

    return femur_bone_mesh, femur_cart_mesh, patella_bone_mesh, patella_cart_mesh


def _finalize_fatpad_mesh(
    fatpad_mesh, resample_clusters_final, units, final_smooth_iter=100, output_path=None
):
    """
    Resample, clean, convert units, and optionally save the fatpad mesh.

    Args:
        fatpad_mesh: Filtered fatpad mesh
        units: 'm' or 'mm' - desired output units
        final_smooth_iter: Number of smoothing iterations for final smoothing (default: 100)
        resample_clusters_final: Number of clusters for final resampling (default: 2,000)
        output_path: Optional path to save the mesh (default: None)

    Returns:
        pymskt.mesh.Mesh: Final processed fatpad mesh
    """
    logger.info(f"Final smoothing with {final_smooth_iter} iterations...")
    fatpad_mesh.smooth(
        n_iter=final_smooth_iter, boundary_smoothing=False, feature_smoothing=False, inplace=True
    )

    logger.info(f"Final resampling to {resample_clusters_final} clusters...")
    fatpad_mesh.resample_surface(subdivisions=2, clusters=resample_clusters_final)

    logger.info("Extracting largest component and cleaning...")
    fatpad_mesh = fatpad_mesh.extract_largest()
    fatpad_mesh = fatpad_mesh.clean()

    # Convert back to meters if needed
    if units == "m":
        logger.info("Converting mesh to meters...")
        fatpad_mesh.points /= 1000
    elif units == "mm":
        pass

    # Save if output path is provided
    if output_path is not None:
        logger.info(f"Saving to: {output_path}")
        fatpad_mesh.save(output_path)

    return fatpad_mesh


def _forward_clearance_to_targets(
    source_mesh: Mesh,
    targets: list,  # [patella_bone_mesh, patella_cart_mesh]
    ray_cast_length: float = 6.0,
    safety_mm: float = 0.0,
) -> np.ndarray:
    """
    For each vertex, compute forward ray-cast distance to targets.
    Returns minimum clearance minus safety margin.

    Args:
        source_mesh: Source mesh to compute clearance from
        targets: List of target meshes (e.g., patella bone and cartilage)
        ray_cast_length: Length of ray to cast (in mm)
        safety_mm: Safety margin to subtract from clearance

    Returns:
        np.ndarray: Per-vertex minimum clearance to any target (minus safety)
    """
    if not isinstance(source_mesh, Mesh):
        source_mesh = Mesh(source_mesh)

    # this can cause issues because the source_mesh
    # source_mesh.compute_normals(inplace=True, auto_orient_normals=True, consistent_normals=True)

    clearances = []
    for i, tgt in enumerate(targets):
        source_mesh.calc_distance_to_other_mesh(
            list_other_meshes=[tgt],
            ray_cast_length=ray_cast_length,
            percent_ray_length_opposite_direction=0.1,
            name=f"clear_{i}",
        )
        d = source_mesh.get_scalar(f"clear_{i}")
        d = np.where(d > 0, d, np.inf)  # no-hit = infinite clearance
        clearances.append(d)

    min_clear = np.min(np.vstack(clearances), axis=0)
    min_clear = np.clip(min_clear - safety_mm, a_min=0.0, a_max=None)
    return min_clear


def _y_profile_mm(
    mesh: pv.PolyData,
    base_mm: float = 0.8,
    top_mm: float = 5.5,
    scale_axis: int = 1,
    reference_mesh: pv.PolyData = None,
    reference_axis_filter=lambda pts: pts[:, 0] > np.mean(pts[:, 0]),
    scale_percentile: float = 95.0,
    norm_function: str = "log",
) -> np.ndarray:
    """
    Compute per-vertex dilation target profile: base_mm at bottom → top_mm at top.

    Args:
        mesh: Input mesh to compute profile for
        base_mm: Minimum dilation at bottom (mm)
        top_mm: Maximum dilation at top (mm)
        scale_axis: Axis index (0=x, 1=y, 2=z) for vertical scaling
        reference_mesh: Optional reference mesh for threshold calculation
        reference_axis_filter: Function to filter reference points
        scale_percentile: Percentile threshold for scaling start
        norm_function: Scaling function ('linear', 'pow', 'exp', 'log')

    Returns:
        np.ndarray: Per-vertex target dilation in mm
    """
    pts = mesh.points
    ref_pts = reference_mesh.points if reference_mesh is not None else pts

    if reference_axis_filter is not None:
        mask = reference_axis_filter(ref_pts)
        ref_axis_vals = ref_pts[mask, scale_axis]
    else:
        ref_axis_vals = ref_pts[:, scale_axis]

    if ref_axis_vals.size == 0:
        ref_axis_vals = ref_pts[:, scale_axis]

    threshold = np.percentile(ref_axis_vals, scale_percentile)
    axis_vals = pts[:, scale_axis]
    max_axis = np.max(axis_vals)

    prof = np.full(mesh.n_points, base_mm, dtype=float)
    above = axis_vals > threshold

    if not np.any(above) or max_axis <= threshold:
        return prof

    t = (axis_vals[above] - threshold) / (max_axis - threshold)
    t = np.clip(t, 0.0, 1.0)

    # Apply non-linear scaling (same as existing dilate_mesh)
    if norm_function == "linear":
        y = t
    elif norm_function == "pow":
        y = t**2.0
    elif norm_function == "exp":
        k = 5.0
        y = np.expm1(k * t) / np.expm1(k)
    elif norm_function == "log":
        k = 9.0
        y = np.log1p(k * t) / np.log1p(k)
    else:
        y = t

    prof[above] = base_mm + (top_mm - base_mm) * y
    return prof


def dilate_mesh_with_profile_and_clearance(
    femur_bone_mesh,
    patella_bone_mesh,
    patella_cart_mesh,
    base_mm: float = 0.8,
    top_mm: float = 6.0,
    scale_axis: int = 1,
    reference_mesh=None,
    reference_axis_filter=lambda pts: pts[:, 0] > np.mean(pts[:, 0]),
    scale_percentile: float = 95.0,
    norm_function: str = "log",
    ray_cast_length: float = 6.0,
    safety_mm: float = 0.0,
    mask=None,
):
    """
    Dilate femur bone limited by both y-profile and clearance to patella.

    This function computes a per-vertex dilation that is constrained by:
    1. A vertical profile (increasing from base_mm to top_mm)
    2. Forward clearance to patella structures (preventing penetration)

    Args:
        femur_bone_mesh: Femur bone mesh to dilate
        patella_bone_mesh: Patella bone mesh (clearance constraint)
        patella_cart_mesh: Patella cartilage mesh (clearance constraint)
        base_mm: Minimum dilation at bottom (mm)
        top_mm: Maximum dilation at top (mm)
        scale_axis: Axis for vertical scaling (0=x, 1=y, 2=z)
        reference_mesh: Reference mesh for profile calculation
        reference_axis_filter: Filter function for reference mesh
        scale_percentile: Percentile threshold for scaling
        norm_function: Scaling function type
        ray_cast_length: Ray length for clearance calculation
        safety_mm: Safety margin to maintain from patella
        mask: Optional boolean mask to restrict dilation

    Returns:
        tuple: (dilated_mesh, per_vertex_dilation_mm)
    """
    if not isinstance(femur_bone_mesh, Mesh):
        femur_bone_mesh = Mesh(femur_bone_mesh)

    # 1. Compute target dilation profile based on y-position
    target_profile = _y_profile_mm(
        femur_bone_mesh,
        base_mm,
        top_mm,
        scale_axis=scale_axis,
        reference_mesh=reference_mesh,
        reference_axis_filter=reference_axis_filter,
        scale_percentile=scale_percentile,
        norm_function=norm_function,
    )

    # 2. Compute forward clearance to patella
    clearance = _forward_clearance_to_targets(
        source_mesh=femur_bone_mesh,
        targets=[patella_bone_mesh, patella_cart_mesh],
        ray_cast_length=ray_cast_length,
        safety_mm=safety_mm,
    )

    # 3. Actual dilation = min(profile, clearance)
    actual = np.minimum(target_profile, clearance)

    # 4. Apply optional mask
    if mask is not None:
        if mask.ndim > 1:
            mask = mask.ravel()
        actual = np.where(mask, actual, 0.0)

    # 5. Dilate along normals
    # commened out the below - at this point, there are points on the surface removed
    # so, running compute_normals may cause issues pointing the normals in the wrong
    # direction.
    # femur_bone_mesh.compute_normals(inplace=True, auto_orient_normals=True, consistent_normals=True)
    pts = femur_bone_mesh.points.copy()
    nrm = femur_bone_mesh.point_normals
    pts_new = pts + (actual[:, None] * nrm)

    out = femur_bone_mesh.copy()
    out.points = pts_new
    return out, actual


def create_prefemoral_fatpad_noboolean(
    femur_bone_mesh,
    femur_cart_mesh,
    patella_bone_mesh,
    patella_cart_mesh,
    base_mm: float = DEFAULT_FATPAD_BASE_MM,
    top_mm: float = DEFAULT_FATPAD_TOP_MM,
    max_distance_to_patella_mm: float = DEFAULT_MAX_DISTANCE_TO_PATELLA_MM,
    percent_fem_cart_to_keep: float = 0.15,
    resample_clusters_final: int = 2000,
    output_path: str = None,
    units: str = "m",
    ray_cast_length: float = 6.0,
    safety_mm: float = 0.0,
    norm_function: str = "log",
    final_smooth_iter: int = 100,
):
    """
    Creates a prefemoral fatpad mesh using clearance-limited dilation (no boolean operations).

    This is an alternative to create_prefemoral_fatpad_contact_mesh that avoids boolean
    operations by using ray-cast clearance to limit dilation. The approach:
    1. Extracts exposed femur bone surface (not covered by cartilage)
    2. Applies progressive dilation (base_mm → top_mm) limited by clearance to patella
    3. Filters to anterior/superior region near patella
    4. Resamples and smooths the result

    Advantages over boolean approach:
    - More numerically stable (no watertight mesh requirements)
    - Faster computation
    - More direct control over dilation profile
    - Avoids boolean operation failures

    Args:
        femur_bone_mesh: Femur bone mesh (pymskt.mesh.Mesh, pyvista.PolyData, or path)
        femur_cart_mesh: Femur cartilage mesh (pymskt.mesh.Mesh, pyvista.PolyData, or path)
        patella_bone_mesh: Patella bone mesh (pymskt.mesh.Mesh, pyvista.PolyData, or path)
        patella_cart_mesh: Patella cartilage mesh (pymskt.mesh.Mesh, pyvista.PolyData, or path)
        base_mm: Minimum dilation at inferior edge (default: 0.5 mm)
        top_mm: Maximum dilation at superior edge (default: 6.0 mm)
        max_distance_to_patella_mm: Maximum distance to patella to keep points (default: 30.0 mm)
        percent_fem_cart_to_keep: Percentage of femur cartilage height to keep (default: 0.15)
        resample_clusters_final: Number of clusters for final resampling (default: 2,000)
        output_path: Optional path to save the result (default: None)
        units: Units of input meshes - 'm' or 'mm' (default: 'm')
        ray_cast_length: Ray length for clearance calculation (default: 6.0 mm)
        safety_mm: Safety margin from patella (default: 0.0 mm)
        norm_function: Dilation scaling function - 'linear', 'pow', 'exp', 'log' (default: 'log')
        final_smooth_iter: Final smoothing iterations (default: 100)

    Returns:
        pymskt.mesh.Mesh: The processed prefemoral fatpad mesh

    Notes:
        Input meshes are assumed to be in meters (OpenSim standard) by default.
        Processing is done in mm for better numerical stability.
    """
    # Step 0: Prep & convert to mm
    femur_bone_mesh, femur_cart_mesh, patella_bone_mesh, patella_cart_mesh = (
        _prepare_fatpad_input_meshes(
            femur_bone_mesh, femur_cart_mesh, patella_bone_mesh, patella_cart_mesh, units
        )
    )

    # Step 1: Pre-process for reliable normals/topology
    logger.info("Cleaning and orienting meshes...")
    femur_bone_mesh = femur_bone_mesh.clean()
    femur_cart_mesh = femur_cart_mesh.clean()
    patella_bone_mesh = patella_bone_mesh.clean()
    patella_cart_mesh = patella_cart_mesh.clean()

    # Ensure consistent normals BEFORE any ray operations
    for mesh in [femur_bone_mesh, femur_cart_mesh, patella_bone_mesh, patella_cart_mesh]:
        mesh.compute_normals(inplace=True, auto_orient_normals=True, consistent_normals=True)

    # Step 2: Extract exposed bone edge (not covered by cartilage)
    logger.info("Extracting exposed bone edge (removing cartilage-covered regions)...")
    femur_bone_mskt = (
        Mesh(femur_bone_mesh) if not isinstance(femur_bone_mesh, Mesh) else femur_bone_mesh
    )
    # fix faces again...
    # femur_bone_mskt.consistent_faces()

    femur_bone_mskt.calc_distance_to_other_mesh(
        femur_cart_mesh,
        ray_cast_length=10.0,  # short ray just to detect cartilage
        percent_ray_length_opposite_direction=0.1,
        name="cart_coverage",
    )

    cart_coverage = femur_bone_mskt.get_scalar("cart_coverage")
    exposed_bone_mask = cart_coverage == 0  # no cartilage coverage

    # Also filter to anterior region immediately
    anterior_mask = femur_bone_mskt.points[:, 0] > np.mean(femur_bone_mskt.points[:, 0])
    combined_mask = exposed_bone_mask & anterior_mask

    logger.info(f"Exposed bone points: {np.sum(exposed_bone_mask)} / {len(exposed_bone_mask)}")
    logger.info(f"Anterior+exposed: {np.sum(combined_mask)} / {len(combined_mask)}")

    starting_surface = femur_bone_mskt.remove_points(~combined_mask)[0]
    starting_surface = Mesh(starting_surface).clean()

    # Step 3: Clearance-limited dilation
    logger.info("Applying clearance-limited dilation...")
    fatpad_dilated, per_vertex_mm = dilate_mesh_with_profile_and_clearance(
        femur_bone_mesh=starting_surface,
        patella_bone_mesh=patella_bone_mesh,
        patella_cart_mesh=patella_cart_mesh,
        base_mm=base_mm,
        top_mm=top_mm,
        scale_axis=1,
        reference_mesh=femur_cart_mesh,
        reference_axis_filter=lambda pts: pts[:, 0] > np.mean(pts[:, 0]),
        scale_percentile=95.0,
        norm_function=norm_function,
        ray_cast_length=ray_cast_length,
        safety_mm=safety_mm,
        mask=None,  # already filtered to anterior+exposed
    )

    fatpad_dilated.point_data["offset_mm"] = per_vertex_mm

    # Step 4: Filter by proximity to patella and vertical position
    logger.info("Filtering fatpad region...")

    # Distance to patella (min of bone and cart)
    pat_b_mskt = Mesh(patella_bone_mesh)
    pat_c_mskt = Mesh(patella_cart_mesh)
    sdf_b = pat_b_mskt.get_sdf_pts(fatpad_dilated.points)
    sdf_c = pat_c_mskt.get_sdf_pts(fatpad_dilated.points)
    d_patella = np.minimum(np.abs(sdf_b), np.abs(sdf_c))

    # Vertical threshold
    fem_cart_anterior_y = femur_cart_mesh.points[
        femur_cart_mesh.points[:, 0] > np.mean(femur_cart_mesh.points[:, 0]), 1
    ]
    y_thresh = np.percentile(fem_cart_anterior_y, (1 - percent_fem_cart_to_keep) * 100)

    keep_radial = d_patella < max_distance_to_patella_mm
    keep_vertical = fatpad_dilated.points[:, 1] > y_thresh
    keep = keep_radial & keep_vertical

    logger.info(f"Points passing radial filter: {np.sum(keep_radial)} / {len(keep_radial)}")
    logger.info(f"Points passing vertical filter: {np.sum(keep_vertical)} / {len(keep_vertical)}")
    logger.info(f"Points passing both: {np.sum(keep)} / {len(keep)}")

    if np.sum(keep) == 0:
        raise ValueError("All points filtered out. Adjust thresholds.")

    fatpad_dilated.point_data["keep"] = keep.astype(float)
    fatpad = fatpad_dilated.remove_points(~keep)[0]
    fatpad = Mesh(fatpad)

    # Step 5: Finalize
    logger.info(f"Final resampling to {resample_clusters_final} clusters...")
    fatpad = _finalize_fatpad_mesh(
        fatpad, resample_clusters_final, units, final_smooth_iter, output_path
    )

    logger.info("Prefemoral fatpad (no-boolean) created successfully")
    return fatpad


def optimize_patella_position(
    pat_articular_surfaces,
    fem_articular_surfaces,
    pat_mesh_osim,
    patella_adjust_rel_vert_overlap=None,  # 0.4
    patella_adjust_abs_vert_overlap=None,  # 0.012
    patella_adjust_contact_area=None,  # 0.2
    contact_area_adjustment=np.array([0, 0.001, 0]),
    return_move_down=False,
):
    """
    Optimizes the patella's vertical position based on overlap with femoral cartilage.

    The function adjusts the patella's position downwards if overlap metrics
    (percentage of vertical overlap, absolute vertical overlap, or contact area)
    fall below specified thresholds.

    Args:
        pat_articular_surfaces (pymskt.mesh.Mesh or pyvista.PolyData):
            The patellar articular surface. This mesh's points will be modified.
        fem_articular_surfaces (pymskt.mesh.Mesh or pyvista.PolyData):
            The femoral articular surface, used as a reference for overlap.
        pat_mesh_osim (pymskt.mesh.Mesh): The full patella bone mesh. This mesh's
            points will also be modified in conjunction with `pat_articular_surfaces`.
        patella_adjust_rel_vert_overlap (float, optional): Target minimum
            percentage of vertical overlap. If current overlap is less, patella is
            moved down. Defaults to None (no adjustment based on this criterion).
            Example: 0.4 (for 40%).
        patella_adjust_abs_vert_overlap (float, optional): Target minimum
            absolute vertical overlap (in mesh units, typically meters). If current
            overlap is less, patella is moved down. Defaults to None.
            Example: 0.012 (for 1.2 cm).
        patella_adjust_contact_area (float, optional): Target minimum percentage
            of non-zero contact area (patellar surface points close to femoral
            surface). If current area is less, patella is iteratively moved down.
            Defaults to None. Example: 0.2 (for 20%).
        contact_area_adjustment (np.ndarray, optional): The vector by which to
            move the patella down in each iteration if adjusting for contact area.
            Defaults to np.array([0, 0.001, 0]) (1 mm down in y).
        return_move_down (bool, optional): If True, also returns the total
            displacement vector applied to the patella. Defaults to False.

    Returns:
        tuple:
            - pat_articular_surfaces (pymskt.mesh.Mesh or pyvista.PolyData):
                The modified patellar articular surface.
            - pat_mesh_osim (pymskt.mesh.Mesh): The modified full patella bone mesh.
            - move_down_total (np.ndarray, optional): If `return_move_down` is True,
                this is the total 3D vector representing the patella's displacement.
    """
    # determine how much of the patella cartilage is overlapping
    # with the femoral cartilage. Depending on the degree of overlap,
    # move the patella downward.

    # Previous "bad" results (dislocation of patella superiorly)
    # had 12.69% overall overlap and had 28.02% or 0.82 cm of vertical
    # overlap.

    # a good result (before settle sim x5 at 2*0.02 PT strain) had
    # 24.8% overall overlap, and 45.13% or 1.32 cm of vertical overlap.

    # for the patella, calculate the distances to the femoral cartilage

    if not isinstance(pat_articular_surfaces, Mesh):
        pat_articular_surfaces = Mesh(pat_articular_surfaces)

    percent_non_zero, vert_overlap, total_vert, percent_vert_overlap = compute_overlap_metrics(
        pat_articular_surfaces, fem_articular_surfaces
    )

    logger.info(
        f"Percent of patella cartilage that is not overlapping with femoral cartilage: {percent_non_zero:.2f}%"
    )
    logger.info(f"Percent vertical overlap: {percent_vert_overlap:.2f}%")
    logger.info(f"Vertical overlap: {vert_overlap*100:.2f} cm")
    logger.info(f"Total vertical: {total_vert*100:.2f} cm")

    # if the vertical overlap is less than 40%, then move the patella down
    # by the amount necessary to increase the overlap to 40%
    move_down_rel_vert = 0
    move_down_abs_vert = 0
    move_down_contact_area = np.asarray([0.0, 0.0, 0.0])
    if patella_adjust_rel_vert_overlap is not None:
        if percent_vert_overlap < patella_adjust_rel_vert_overlap:  # 0.4
            logger.info("Adjusting patella position - % vertical overlap to be 40%")
            move_down_rel_vert = (40 - percent_vert_overlap) * total_vert
            pat_articular_surfaces.point_coords[:, 1] -= move_down_rel_vert
            pat_mesh_osim.point_coords[:, 1] -= move_down_rel_vert
        else:
            pass
    if patella_adjust_abs_vert_overlap is not None:
        if vert_overlap < patella_adjust_abs_vert_overlap:  # 0.012
            logger.info("Adjusting patella position - vertical overlap to be 1.2 cm")
            move_down_abs_vert = patella_adjust_abs_vert_overlap - vert_overlap  # 0.012
            pat_articular_surfaces.point_coords[:, 1] -= move_down_abs_vert
            pat_mesh_osim.point_coords[:, 1] -= move_down_abs_vert
        else:
            pass
    if patella_adjust_contact_area is not None:
        while percent_non_zero < patella_adjust_contact_area:  # 0.2
            logger.info(f"Percent non-zero: {percent_non_zero}")
            logger.info("Adjusting patella down 1mm")
            pat_articular_surfaces.point_coords -= contact_area_adjustment
            pat_mesh_osim.point_coords -= contact_area_adjustment
            move_down_contact_area += contact_area_adjustment
            percent_non_zero, _, _, _ = compute_overlap_metrics(
                pat_articular_surfaces, fem_articular_surfaces
            )

    move_down_total = (
        np.array([0, move_down_rel_vert, 0])
        + np.array([0, move_down_abs_vert, 0])
        + move_down_contact_area
    )

    if return_move_down:
        return pat_articular_surfaces, pat_mesh_osim, move_down_total
    else:
        return pat_articular_surfaces, pat_mesh_osim
