import numpy as np
import pyvista as pv
from pymskt.mesh import Mesh, BoneMesh, CartilageMesh
from pymskt.mesh.meshCartilage import (
    remove_intersecting_vertices,
    get_n_largest,
    remove_isolated_cells,
    extract_articular_surface
)
import gc
import logging

logger = logging.getLogger(__name__)

def extract_meniscus_articulating_surface(
    meniscus_mesh: pv.PolyData,
    articulating_bone_mesh: pv.PolyData,
    ray_length: float = 10.0,
    n_largest: int = 1,
    smooth_iter: int = 15,
    boundary_smoothing: bool = False  # False allows boundaries to be smoothed
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
        mesh1=meniscus_mesh,       # The mesh to be clipped/modified (meniscus)
        mesh2=articulating_bone_mesh, # The mesh used as the reference/clipper (femur/tibia)
        ray_length=-ray_length,
    )
    if not isinstance(surface, pv.PolyData):
        # This assertion helps catch issues early if remove_intersecting_vertices
        # doesn't return the expected type.
        raise TypeError(f"Expected pv.PolyData from remove_intersecting_vertices, got {type(surface)}")

    # Step 2: Keep the n-largest connected components
    # (Assuming get_n_largest is available)
    processed_surface = get_n_largest(surface, n=n_largest)
    
    # Step 3: Ensure the result is a surface mesh (PolyData)
    # get_n_largest might return an UnstructuredGrid, so extract surface if necessary.
    if not isinstance(processed_surface, pv.PolyData):
        processed_surface = processed_surface.extract_surface()
        
    if not isinstance(processed_surface, pv.PolyData):
         raise TypeError(f"Expected pv.PolyData after get_n_largest and extract_surface, got {type(processed_surface)}")

    # Step 4: Remove isolated cells
    # This helps clean up small artifacts or disconnected parts of the mesh.
    # (Assuming remove_isolated_cells is available)
    processed_surface = remove_isolated_cells(processed_surface)
    if not isinstance(processed_surface, pv.PolyData):
        raise TypeError(f"Expected pv.PolyData after remove_isolated_cells, got {type(processed_surface)}")

    # Step 5: Smooth the surface
    # The boundary_smoothing=False argument allows the edges of the surface to be smoothed.
    processed_surface = processed_surface.smooth(
        n_iter=smooth_iter, 
        boundary_smoothing=boundary_smoothing
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
    triangle_density=4_000_000,  # 2.6 triangles/mm^2 
    meniscus_clusters=None,
    upper_articulating_bone_clusters=None,
    lower_articulating_bone_clusters=None,
):
    """
    Creates a meniscus articulating surface from a meniscus mesh and two articulating bone meshes.
    """
    
    # compute density metrics for resampling while in meters space. 
    if triangle_density is not None:
        meniscus_mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True, inplace=True)
        # compute the triangle density of the cart mesh now
        current_density = meniscus_mesh.n_cells/meniscus_mesh.area
        # downsample factor
        downsample_factor = current_density/triangle_density
        meniscus_clusters = int(meniscus_mesh.point_coords.shape[0]/downsample_factor)
        
        logger.info(f'current density: {current_density/1_000_000} triangles/mm^2')
        logger.info(f'target density: {triangle_density/1_000_000} triangles/mm^2')
        logger.info(f'current number of points: {meniscus_mesh.point_coords.shape[0]}')
        logger.info(f'target number of points: {meniscus_clusters}')
        
    # convert meshes to mm (instead of meters)
    meniscus_mesh_ = meniscus_mesh.copy()
    meniscus_mesh_.point_coords = meniscus_mesh_.point_coords * 1000
    upper_articulating_bone_mesh_ = upper_articulating_bone_mesh.copy()
    upper_articulating_bone_mesh_.point_coords = upper_articulating_bone_mesh_.point_coords * 1000
    lower_articulating_bone_mesh_ = lower_articulating_bone_mesh.copy()
    lower_articulating_bone_mesh_.point_coords = lower_articulating_bone_mesh_.point_coords * 1000
    
    # resample while in mm space 
    if meniscus_clusters is not None:
        meniscus_mesh_.resample_surface(subdivisions=1, clusters=meniscus_clusters)
        # density after resampling
        updated_density = meniscus_mesh_.mesh.n_cells/meniscus_mesh_.mesh.area
        logger.info(f'achieved density: {updated_density/1_000_000} triangles/mm^2')
    
    # resample bones if specified
    if upper_articulating_bone_clusters is not None:
        upper_articulating_bone_mesh_.resample_surface(subdivisions=1, clusters=upper_articulating_bone_clusters)
    if lower_articulating_bone_clusters is not None:
        lower_articulating_bone_mesh_.resample_surface(subdivisions=1, clusters=lower_articulating_bone_clusters)
        
    # extract upper articulating surface of the meniscus
    upper_meniscus_articulating_surface = extract_meniscus_articulating_surface(
        meniscus_mesh_,
        upper_articulating_bone_mesh_,
        ray_length=ray_length,
        n_largest=n_largest,
        smooth_iter=smooth_iter,
        boundary_smoothing=boundary_smoothing
    )
    
    # extract lower articulating surface of the meniscus
    lower_meniscus_articulating_surface = extract_meniscus_articulating_surface(
        meniscus_mesh_,
        lower_articulating_bone_mesh_,
        ray_length=ray_length,
        n_largest=n_largest,
        smooth_iter=smooth_iter,
        boundary_smoothing=boundary_smoothing
    )
    
    # convert back to meters space
    upper_meniscus_articulating_surface.point_coords = upper_meniscus_articulating_surface.point_coords / 1000
    lower_meniscus_articulating_surface.point_coords = lower_meniscus_articulating_surface.point_coords / 1000
    
    return upper_meniscus_articulating_surface, lower_meniscus_articulating_surface
    
    

def create_articular_surfaces(
    bone_mesh_osim,
    cart_mesh_osim,
    n_largest=1,
    bone_clusters=None,
    cart_clusters=None,
    triangle_density=4_000_000,  # 2.6 triangles/mm^2 
    ray_length=10,
    smooth_iter=100
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
            cart_mesh_osim.mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True, inplace=True)
        # compute the triangle density of the cart mesh now
        current_density = cart_mesh_osim.mesh.n_cells/cart_mesh_osim.mesh.area
        # downsample factor
        downsample_factor = current_density/triangle_density
        # calculate target number of points
        cart_clusters = int(cart_mesh_osim.point_coords.shape[0]/downsample_factor)

        logger.info(f'current density: {current_density/1_000_000} triangles/mm^2')
        logger.info(f'target density: {triangle_density/1_000_000} triangles/mm^2')
        logger.info(f'current number of points: {cart_mesh_osim.point_coords.shape[0]}')
        logger.info(f'target number of points: {cart_clusters}')

    # resample bone surface to be 10k points to reduce computation time
    logger.info('resampling femur surface')
    bone_mesh_osim_ = bone_mesh_osim.copy()
    bone_mesh_osim_.point_coords = bone_mesh_osim_.point_coords * 1000
    if bone_clusters is not None:
        bone_mesh_osim_.resample_surface(subdivisions=1, clusters=bone_clusters)
    bone_mesh_osim_ = BoneMesh(bone_mesh_osim_)

    # assign cartilage to bone
    logger.info('resample cartilage surface')
    cart_mesh_osim_ = cart_mesh_osim.copy()
    cart_mesh_osim_.point_coords = cart_mesh_osim_.point_coords * 1000
    if cart_clusters is not None:
        # if not isinstance(cart_mesh_osim_.mesh, pv.PolyData):
        #     cart_mesh_osim_.mesh = pv.PolyData(cart_mesh_osim_.mesh)
        # cart_mesh_osim_.mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True, inplace=True)
        cart_mesh_osim_.resample_surface(subdivisions=1, clusters=cart_clusters)
        # density after resampling
        updated_density = cart_mesh_osim_.mesh.n_cells/cart_mesh_osim_.mesh.area
        logger.info(f'achieved density: {updated_density/1_000_000} triangles/mm^2')
    cart_mesh_osim_.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True, inplace=True)
    bone_mesh_osim_.list_cartilage_meshes = [CartilageMesh(cart_mesh_osim_)]
    

    # extract articular surface
    logger.info('extracting articular surface')
    articular_surfaces = extract_articular_surface(
        bone_mesh_osim_, 
        ray_length=ray_length, 
        smooth_iter=smooth_iter, 
        n_largest=n_largest
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
        ray_cast_length=1/100, 
        percent_ray_length_opposite_direction=1.0,
        name='fem_cart_dist'
    )
    thickness = pat_articular_surfaces.get_scalar('fem_cart_dist')
    n_non_zero = np.sum(thickness > 0)
    percent_non_zero = n_non_zero / len(thickness)
    
    #get the y coordinates of patella points overlapping with femur
    overlap = np.where(thickness > 0)[0]
    overlap_y = pat_articular_surfaces.point_coords[overlap, 1]

    vert_overlap = np.max(overlap_y) - np.min(overlap_y)
    total_vert = np.max(pat_articular_surfaces.point_coords[:, 1]) - np.min(pat_articular_surfaces.point_coords[:, 1])
    percent_vert_overlap = vert_overlap / total_vert 

    return percent_non_zero, vert_overlap, total_vert, percent_vert_overlap

def dilate_mesh(
    mesh, 
    dilation_mm, 
    mask=None, 
    scale_axis=None, 
    scale_percentile=None, 
    reference_mesh=None, 
    reference_axis_filter=lambda pts: pts[:, 0] > np.mean(pts[:, 0]),
    norm_function='log'
):
    """
    Dilate bone mesh along normals with optional scaled dilation.
    
    Parameters
    ----------
    mesh : pyvista.PolyData
        Input mesh to dilate
    dilation_mm : float
        Base dilation amount in mm
    mask : np.ndarray, optional
        Mask to apply to dilation (n_points,) or (n_points, 1)
    scale_axis : int, optional
        Axis index (0=x, 1=y, 2=z) for scaled dilation. If None, uniform dilation.
    scale_percentile : float, optional
        Percentile threshold (0-100) along scale_axis. Points below this have
        no dilation, points above have scaled dilation increasing linearly from
        0 at the threshold to dilation_mm at the maximum coordinate.
    reference_mesh : pyvista.PolyData, optional
        Reference mesh to use for determining the percentile threshold.
        If None, uses the input mesh itself. Useful for scaling based on 
        patella cartilage coordinates instead of bone coordinates.
    reference_axis_filter : callable, optional
        Function to filter reference_mesh points before calculating percentile.
        E.g., lambda pts: pts[:, 0] > np.mean(pts[:, 0]) for anterior half.
    
    Returns
    -------
    mesh_scaled : pyvista.PolyData
        Dilated mesh
    """
    # Dilate bone mesh along normals
    mesh_scaled = mesh.copy()
    mesh_scaled.compute_normals(inplace=True, auto_orient_normals=True, consistent_normals=True)
    
    if mask is None:
        mask = np.ones((mesh_scaled.n_points, 1), dtype=float)
    else:
        if len(mask.shape) == 1:
            mask = mask[:,None]
    
    points = mesh_scaled.points.copy()
    normals = mesh_scaled.point_normals
    
    # Calculate dilation scaling based on position
    dilation_scale = np.ones((mesh_scaled.n_points, 1), dtype=float)
    
    if scale_axis is not None and scale_percentile is not None:
        # Determine which mesh to use for calculating the threshold
        if reference_mesh is not None:
            ref_points = reference_mesh.points
        else:
            ref_points = points
        
        # Apply filter if provided (e.g., to get anterior half only)
        if reference_axis_filter is not None:
            filter_mask = reference_axis_filter(ref_points)
            filtered_coords = ref_points[filter_mask, scale_axis]
        else:
            filtered_coords = ref_points[:, scale_axis]
        
        # Calculate threshold from reference mesh
        threshold = np.percentile(filtered_coords, scale_percentile)
        
        # Get the max coordinate for normalization (from filtered reference if applicable)
        max_coord = np.max(points[:, scale_axis])
        
        # Apply scaling to the actual mesh points
        axis_coords = points[:, scale_axis]
        above_threshold = axis_coords > threshold
        
        if np.any(above_threshold):
            # Normalized distance from threshold (0 at threshold, 1 at max)
            norm_distance = np.zeros_like(axis_coords)
            if max_coord > threshold:
                denom = max_coord - threshold
                if denom > 0:
                    idx = above_threshold  # cache the mask
                    t = (axis_coords[idx] - threshold) / denom
                    t = np.clip(t, 0.0, 1.0)
                    if norm_function == 'linear':
                        y = t
                    elif norm_function == 'pow':
                        gamma = 2.0
                        y = t**gamma
                    elif norm_function == 'exp':
                        k = 5.0
                        y = np.expm1(k*t) / np.expm1(k)
                    elif norm_function == 'log':
                        k = 9.0
                        y = np.log1p(k*t) / np.log1p(k)
                    else:
                        y = t  # safe default

                    norm_distance[idx] = y
            
            # Scale dilation: gradually increase from 0 at threshold to 1.0 at max
            dilation_scale = norm_distance.reshape(-1, 1)
    
    # Apply dilation with scaling
    points_new = points + dilation_mm * normals * mask * dilation_scale
    mesh_scaled.points = points_new
    return mesh_scaled

def _type_check_mesh(mesh, mesh_name='mesh'):
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
            raise TypeError(f'{mesh_name} is not a pv.PolyData or pymskt.mesh.Mesh: {type(mesh)}')
    return mesh


def label_vertices_as_bone_or_cartilage(mesh, bone_mesh, cart_mesh):
    """
    
    """
    logger.info('Computing normals...')
    mesh.compute_normals(inplace=True, auto_orient_normals=True, consistent_normals=True)
    bone_mesh.compute_normals(inplace=True, auto_orient_normals=True, consistent_normals=True)
    cart_mesh.compute_normals(inplace=True, auto_orient_normals=True, consistent_normals=True)
    logger.info('Finding closest bone and cartilage points...')
    # For each point on combined, compute closest distance to bone and cartilage
    logger.info('Finding closest bone points...')
    if not isinstance(bone_mesh, Mesh):
        bone_mesh = Mesh(bone_mesh)
    if not isinstance(cart_mesh, Mesh):
        cart_mesh = Mesh(cart_mesh)
    
    logger.info('getting cart sdf')
    cart_mesh.point_coords = cart_mesh.point_coords.astype(float)
    d_cart = cart_mesh.get_sdf_pts(mesh.points.astype(float))
    
    logger.info('getting bone sdf')
    bone_mesh.point_coords = bone_mesh.point_coords.astype(float)
    d_bone = bone_mesh.get_sdf_pts(mesh.points.astype(float))
    
    
    logger.info('Labeling bone/cartilage distances')
    # Create arrays in combined.point_data
    mesh.point_data["d_bone"] = d_bone
    mesh.point_data["d_cart"] = d_cart
    
    # labeling vertices as bone or cartilage
    # Label vertices as bone (1) or cartilage (0)
    is_bone = np.zeros(mesh.n_points, dtype=int)
    is_bone[d_bone < d_cart] = 1
    mesh.point_data["is_bone"] = is_bone
    
    return mesh


def _prepare_fatpad_input_meshes(femur_bone_mesh, femur_cart_mesh, patella_bone_mesh, patella_cart_mesh, units):
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
    femur_bone_mesh = _type_check_mesh(femur_bone_mesh, 'femur_bone_mesh')
    femur_cart_mesh = _type_check_mesh(femur_cart_mesh, 'femur_cart_mesh')
    patella_bone_mesh = _type_check_mesh(patella_bone_mesh, 'patella_bone_mesh')
    patella_cart_mesh = _type_check_mesh(patella_cart_mesh, 'patella_cart_mesh')
    
    # Copy the meshes to ensure they are not modified in place
    femur_bone_mesh = femur_bone_mesh.copy()
    femur_cart_mesh = femur_cart_mesh.copy()
    patella_bone_mesh = patella_bone_mesh.copy()
    patella_cart_mesh = patella_cart_mesh.copy()
    
    # Convert to mm if needed
    if units == 'm':
        logger.info('Converting meshes to mm...')
        femur_bone_mesh.points *= 1000
        femur_cart_mesh.points *= 1000
        patella_cart_mesh.points *= 1000
        patella_bone_mesh.points *= 1000
    elif units == 'mm':
        logger.info('Meshes are already in mm...')
    else:
        raise ValueError(f'Invalid units: {units}, expected "m" or "mm"')
    
    return femur_bone_mesh, femur_cart_mesh, patella_bone_mesh, patella_cart_mesh


def _create_extended_femur_mesh(femur_bone_mesh, femur_cart_mesh, initial_bone_dilation_mm, 
                                 bone_region_dilation_mm, dilation_axis_filter):
    """
    Dilate femur bone, union with cartilage, label vertices, and extend bone region.
    
    Args:
        femur_bone_mesh: Femur bone mesh in mm
        femur_cart_mesh: Femur cartilage mesh in mm
        initial_bone_dilation_mm: Initial dilation of bone
        bone_region_dilation_mm: Additional dilation of bone-labeled region
        dilation_axis_filter: Filter function for dilation axis
        
    Returns:
        pyvista.PolyData: Extended femur mesh with bone region dilated
    """
    logger.info(f'Initial bone dilation: {initial_bone_dilation_mm} mm')
    femur_bone_scaled = dilate_mesh(femur_bone_mesh, initial_bone_dilation_mm)
    
    logger.info('Performing boolean union of femur bone and cartilage...')   
    try:
        if not isinstance(femur_bone_scaled, Mesh):
            logger.info('Converting femur_bone_scaled to Mesh...')
            femur_bone_scaled = Mesh(femur_bone_scaled)
        combined = femur_bone_scaled.boolean_union(femur_cart_mesh)
        logger.info('Union done')
    except Exception as e:
        logger.error(f"Boolean union of femur bone and cartilage failed with error: {e}")
        logger.error("This may indicate meshes have incompatible geometry or degenerate triangles.")
        raise
    
    logger.info('Labeling vertices as bone or cartilage...')
    combined = label_vertices_as_bone_or_cartilage(combined, femur_bone_mesh, femur_cart_mesh)
    
    logger.info(f'Dilating bone region by {bone_region_dilation_mm} mm...')
    combined_bone_extended = dilate_mesh(
        mesh=combined, 
        dilation_mm=bone_region_dilation_mm, 
        mask=combined.point_data["is_bone"] == 1, 
        scale_axis=1, 
        scale_percentile=95, 
        reference_mesh=femur_cart_mesh, 
        reference_axis_filter=dilation_axis_filter
    )
    
    return combined_bone_extended


def _create_combined_patella_mesh(patella_bone_mesh, patella_dilation_mm, patella_cart_mesh):
    """
    Dilate patella bone and union with cartilage.
    
    Args:
        patella_bone_mesh: Patella bone mesh in mm
        patella_dilation_mm: Dilation amount for patella bone
        patella_cart_mesh: Patella cartilage mesh in mm
        
    Returns:
        pyvista.PolyData: Combined patella mesh (bone + cartilage)
    """
    logger.info('Processing patella meshes...')       
    logger.info(f'Dilating patella bone by {patella_dilation_mm} mm...')
    patella_bone_extended = dilate_mesh(patella_bone_mesh, patella_dilation_mm)
    
    logger.info('Boolean union of patella bone and cartilage...')
    if not isinstance(patella_bone_extended, Mesh):
        patella_bone_extended = Mesh(patella_bone_extended)
    combined_patella = patella_bone_extended.boolean_union(patella_cart_mesh)
    
    return combined_patella


def _subtract_and_analyze_meshes(combined_bone_extended, combined_patella):
    """
    Subtract patella from femur and compute distance fields.
    
    Args:
        combined_bone_extended: Extended femur mesh
        combined_patella: Combined patella mesh
        
    Returns:
        pymskt.mesh.Mesh: Subtracted mesh with distance fields and labels
    """
    logger.info('Subtracting patella from femur mesh...')
    try:
        if not isinstance(combined_bone_extended, Mesh):
            logger.info('Converting combined_bone_extended to Mesh...')
            combined_bone_extended = Mesh(combined_bone_extended)
        subtracted = combined_bone_extended.boolean_difference(combined_patella)
    except Exception as e:
        logger.error(f"Boolean difference failed with error: {e}")
        logger.error("This may indicate meshes don't overlap or have incompatible geometry.")
        raise
    
    # Check if subtraction resulted in valid mesh
    if not isinstance(subtracted, pv.PolyData):
        raise TypeError(f"Boolean difference did not return PolyData, got {type(subtracted)}")
    if subtracted.n_points == 0:
        raise ValueError("Boolean difference resulted in empty mesh. Meshes may not overlap properly.")
    subtracted_mskt = Mesh(subtracted)
    
    logger.info('Computing signed distance field to patella...')
    combined_patella_mskt = Mesh(combined_patella)
    sdfs_patella_combined = combined_patella_mskt.get_sdf_pts(subtracted_mskt.points)
    subtracted_mskt.point_data['sdf_patella_combined'] = sdfs_patella_combined
    absolute_sdfs_patella_combined = np.abs(sdfs_patella_combined)
    subtracted_mskt.point_data['d_patella_combined'] = absolute_sdfs_patella_combined
    
    # Get distance to the combined_bone_extended
    sdfs_combined_bone_extended = combined_bone_extended.get_sdf_pts(subtracted_mskt.points)
    subtracted_mskt.point_data['sdf_combined_bone_extended'] = sdfs_combined_bone_extended
    absolute_sdfs_combined_bone_extended = np.abs(sdfs_combined_bone_extended)
    subtracted_mskt.point_data['d_combined_bone_extended'] = absolute_sdfs_combined_bone_extended
    
    is_patella = absolute_sdfs_patella_combined < absolute_sdfs_combined_bone_extended
    subtracted_mskt.point_data['is_patella'] = is_patella.astype(float)
    
    # Copy over the is_bone data
    subtracted_mskt.copy_scalars_from_other_mesh_to_current(combined_bone_extended, orig_scalars_name='is_bone')
    
    return subtracted_mskt


def _filter_fatpad_points(subtracted_mesh, femur_cart_mesh, max_distance_to_patella_mm, percent_fem_cart_to_keep):
    """
    Filter points based on distance to patella and height criteria.
    
    Args:
        subtracted_mesh: Mesh after boolean subtraction with distance fields
        femur_cart_mesh: Femur cartilage mesh for height filtering
        max_distance_to_patella_mm: Maximum distance to patella to keep points
        percent_fem_cart_to_keep: Percentage of femur cartilage height to keep
        
    Returns:
        pymskt.mesh.Mesh: Filtered fatpad mesh
    """
    logger.info(f'Filtering points within {max_distance_to_patella_mm} mm of patella...')
    
    # Keep if: is_patella == 1 OR is_bone == 1
    keep_surface = np.maximum(
        subtracted_mesh.point_data["is_patella"], 
        subtracted_mesh.point_data["is_bone"]
    )
    
    # Filter by distance to patella
    keep_radial = subtracted_mesh.point_data['sdf_patella_combined'] < max_distance_to_patella_mm
    
    # Filter by height - remove fatpad that is too low
    percentile_threshold = (1 - percent_fem_cart_to_keep) * 100
    y_percentile_threshold = np.percentile(
        femur_cart_mesh.points[femur_cart_mesh.points[:, 0] > np.mean(femur_cart_mesh.points[:, 0]), 1], 
        percentile_threshold
    )
    keep_vertical = subtracted_mesh.points[:, 1] > y_percentile_threshold
    
    keep = keep_surface * keep_radial * keep_vertical
    
    subtracted_mesh.point_data["keep"] = keep.astype(float)
    subtracted_mesh = subtracted_mesh.triangulate().clean()
    
    # Add diagnostic information
    n_total = len(keep)
    n_keep = np.sum(keep)
    logger.info(f'Filtering diagnostics:')
    logger.info(f'  - Total points: {n_total}')
    logger.info(f'  - Points passing surface filter: {np.sum(keep_surface)}')
    logger.info(f'  - Points passing radial filter: {np.sum(keep_radial)}')
    logger.info(f'  - Points passing vertical filter: {np.sum(keep_vertical)}')
    logger.info(f'  - Points passing all filters: {n_keep}')

    # Check if any points remain
    if n_keep == 0:
        raise ValueError(
            f"All points filtered out! No fatpad mesh can be created.\n"
            f"Consider adjusting parameters:\n"
            f"  - max_distance_to_patella_mm (current: {max_distance_to_patella_mm})\n"
            f"  - percent_fem_cart_to_keep (current: {percent_fem_cart_to_keep})\n"
            f"  - y_percentile_threshold (computed: {y_percentile_threshold:.2f})"
        )
    
    logger.info('Removing filtered points...')
    fatpad = subtracted_mesh.remove_points(subtracted_mesh.point_data['keep'] == 0)[0]
    fatpad = Mesh(fatpad)
    
    return fatpad


def _finalize_fatpad_mesh(fatpad_mesh, resample_clusters_final, units, final_smooth_iter=100, output_path=None):
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
    logger.info(f'Final smoothing with {final_smooth_iter} iterations...')
    fatpad_mesh.smooth(n_iter=final_smooth_iter, boundary_smoothing=False, feature_smoothing=False, inplace=True)
    
    logger.info(f'Final resampling to {resample_clusters_final} clusters...')
    fatpad_mesh.resample_surface(subdivisions=2, clusters=resample_clusters_final)
    
    logger.info('Extracting largest component and cleaning...')
    fatpad_mesh = fatpad_mesh.extract_largest()
    fatpad_mesh = fatpad_mesh.clean()
    
    # Convert back to meters if needed
    if units == 'm':
        logger.info('Converting mesh to meters...')
        fatpad_mesh.points /= 1000
    elif units == 'mm':
        pass
    
    # Save if output path is provided
    if output_path is not None:
        logger.info(f'Saving to: {output_path}')
        fatpad_mesh.save(output_path)
    
    return fatpad_mesh


def create_prefemoral_fatpad_contact_mesh(
    femur_bone_mesh,
    femur_cart_mesh,
    patella_bone_mesh,
    patella_cart_mesh,
    initial_bone_dilation_mm: float = 0.6,
    bone_region_dilation_mm: float = 4.0,
    patella_dilation_mm: float = 0.3,
    max_distance_to_patella_mm: float = 30.0,
    resample_clusters_final: int = 2_000,
    output_path: str = None,
    percent_fem_cart_to_keep: float = 0.15,
    dilation_axis_filter: callable = lambda pts: pts[:, 0] > np.mean(pts[:, 0]),
    units='m'
):
    """
    Creates a prefemoral fatpad mesh by dilating and combining bone/cartilage meshes,
    then subtracting the patella geometry and filtering by distance.
    
    This function creates the prefemoral fatpad superior to the trochlea and posterior to the patella by:
    1. Dilating the femur bone mesh slightly along surface normals
    2. Performing a boolean union with the femur cartilage mesh
    3. Resampling the combined surface
    4. Labeling vertices as bone or cartilage based on proximity
    5. Further dilating bone-labeled vertices
    6. Dilating the patella bone and combining with patella cartilage
    7. Subtracting the combined patella from the femur mesh
    8. Keeping only points within specified distance to patella
    9. Final resampling and cleanup
    
    Args:
        femur_bone_mesh: Femur bone mesh (pymskt.mesh.Mesh, pyvista.PolyData, or path)
        femur_cart_mesh: Femur cartilage mesh (pymskt.mesh.Mesh, pyvista.PolyData, or path)
        patella_bone_mesh: Patella bone mesh (pymskt.mesh.Mesh, pyvista.PolyData, or path)
        patella_cart_mesh: Patella cartilage mesh (pymskt.mesh.Mesh, pyvista.PolyData, or path)
        initial_bone_dilation_mm: Initial dilation of bone in mm (default: 0.6)
        bone_region_dilation_mm: Additional dilation of bone region in mm (default: 4.0)
        patella_dilation_mm: Dilation of patella bone in mm (default: 0.3)
        max_distance_to_patella_mm: Maximum distance to patella to keep points (default: 30.0)
        resample_clusters_final: Number of clusters for final resampling (default: 2,000)
        output_path: Optional path to save the result (default: None)
        percent_fem_cart_to_keep: Percentage of femur cartilage height to keep (default: 0.15)
        dilation_axis_filter: Filter function for dilation axis (default: anterior half)
        units: Units of input meshes - 'm' or 'mm' (default: 'm')
    
    Returns:
        pymskt.mesh.Mesh: The processed prefemoral fatpad mesh with patella subtracted
    
    Notes:
        Input meshes are assumed to be in meters (OpenSim standard) by default.
        Processing is done in mm for better numerical stability.
    """
    # Step 1: Prepare input meshes
    femur_bone_mesh, femur_cart_mesh, patella_bone_mesh, patella_cart_mesh = _prepare_fatpad_input_meshes(
        femur_bone_mesh, femur_cart_mesh, patella_bone_mesh, patella_cart_mesh, units
    )
    
    # Step 2: Create extended femur mesh
    combined_bone_extended = _create_extended_femur_mesh(
        femur_bone_mesh, femur_cart_mesh, initial_bone_dilation_mm, 
        bone_region_dilation_mm, dilation_axis_filter
    )
    
    # Step 3: Create combined patella mesh
    combined_patella = _create_combined_patella_mesh(
        patella_bone_mesh, patella_dilation_mm, patella_cart_mesh
    )
    
    # Step 4: Subtract patella from femur and analyze
    subtracted_mesh = _subtract_and_analyze_meshes(combined_bone_extended, combined_patella)
    
    # Step 5: Filter fatpad points
    fatpad = _filter_fatpad_points(
        subtracted_mesh, femur_cart_mesh, max_distance_to_patella_mm, percent_fem_cart_to_keep
    )
    
    # Step 6: Finalize fatpad mesh
    fatpad = _finalize_fatpad_mesh(fatpad, resample_clusters_final, units, final_smooth_iter=100, output_path=output_path)
    
    logger.info('Prefemoral fatpad mesh with patella subtraction created successfully')
    return fatpad


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
    
    source_mesh.compute_normals(inplace=True, auto_orient_normals=True, consistent_normals=True)
    
    clearances = []
    for i, tgt in enumerate(targets):
        source_mesh.calc_distance_to_other_mesh(
            list_other_meshes=[tgt],
            ray_cast_length=ray_cast_length,
            percent_ray_length_opposite_direction=0.1,
            name=f'clear_{i}'
        )
        d = source_mesh.get_scalar(f'clear_{i}')
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
    norm_function: str = 'log'
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
    if norm_function == 'linear':
        y = t
    elif norm_function == 'pow':
        y = t**2.0
    elif norm_function == 'exp':
        k = 5.0
        y = np.expm1(k*t) / np.expm1(k)
    elif norm_function == 'log':
        k = 9.0
        y = np.log1p(k*t) / np.log1p(k)
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
    reference_mesh = None,
    reference_axis_filter=lambda pts: pts[:, 0] > np.mean(pts[:, 0]),
    scale_percentile: float = 95.0,
    norm_function: str = 'log',
    ray_cast_length: float = 6.0,
    safety_mm: float = 0.0,
    mask = None,
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
        femur_bone_mesh, base_mm, top_mm,
        scale_axis=scale_axis,
        reference_mesh=reference_mesh,
        reference_axis_filter=reference_axis_filter,
        scale_percentile=scale_percentile,
        norm_function=norm_function
    )
    
    # 2. Compute forward clearance to patella
    clearance = _forward_clearance_to_targets(
        source_mesh=femur_bone_mesh,
        targets=[patella_bone_mesh, patella_cart_mesh],
        ray_cast_length=ray_cast_length,
        safety_mm=safety_mm
    )
    
    # 3. Actual dilation = min(profile, clearance)
    actual = np.minimum(target_profile, clearance)
    
    # 4. Apply optional mask
    if mask is not None:
        if mask.ndim > 1:
            mask = mask.ravel()
        actual = np.where(mask, actual, 0.0)
    
    # 5. Dilate along normals
    femur_bone_mesh.compute_normals(inplace=True, auto_orient_normals=True, consistent_normals=True)
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
    base_mm: float = 0.5,
    top_mm: float = 6.0,
    max_distance_to_patella_mm: float = 30.0,
    percent_fem_cart_to_keep: float = 0.15,
    resample_clusters_final: int = 2000,
    output_path: str = None,
    units: str = 'm',
    ray_cast_length: float = 6.0,
    safety_mm: float = 0.0,
    norm_function: str = 'log',
    final_smooth_iter: int = 100
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
    femur_bone_mesh, femur_cart_mesh, patella_bone_mesh, patella_cart_mesh = \
        _prepare_fatpad_input_meshes(
            femur_bone_mesh, femur_cart_mesh, 
            patella_bone_mesh, patella_cart_mesh, 
            units
        )
    
    # Step 1: Pre-process for reliable normals/topology
    logger.info('Cleaning and orienting meshes...')
    femur_bone_mesh = femur_bone_mesh.clean()
    femur_cart_mesh = femur_cart_mesh.clean()
    patella_bone_mesh = patella_bone_mesh.clean()
    patella_cart_mesh = patella_cart_mesh.clean()
    
    # Ensure consistent normals BEFORE any ray operations
    for mesh in [femur_bone_mesh, femur_cart_mesh, patella_bone_mesh, patella_cart_mesh]:
        mesh.compute_normals(inplace=True, auto_orient_normals=True, consistent_normals=True)
    
    # Step 2: Extract exposed bone edge (not covered by cartilage)
    logger.info('Extracting exposed bone edge (removing cartilage-covered regions)...')
    femur_bone_mskt = Mesh(femur_bone_mesh) if not isinstance(femur_bone_mesh, Mesh) else femur_bone_mesh
    
    femur_bone_mskt.calc_distance_to_other_mesh(
        femur_cart_mesh,
        ray_cast_length=10.0,  # short ray just to detect cartilage
        percent_ray_length_opposite_direction=0.1,
        name='cart_coverage'
    )
    
    cart_coverage = femur_bone_mskt.get_scalar('cart_coverage')
    exposed_bone_mask = cart_coverage == 0  # no cartilage coverage
    
    # Also filter to anterior region immediately
    anterior_mask = femur_bone_mskt.points[:, 0] > np.mean(femur_bone_mskt.points[:, 0])
    combined_mask = exposed_bone_mask & anterior_mask
    
    logger.info(f'Exposed bone points: {np.sum(exposed_bone_mask)} / {len(exposed_bone_mask)}')
    logger.info(f'Anterior+exposed: {np.sum(combined_mask)} / {len(combined_mask)}')
    
    starting_surface = femur_bone_mskt.remove_points(~combined_mask)[0]
    starting_surface = Mesh(starting_surface).clean()
    
    # Step 3: Clearance-limited dilation
    logger.info('Applying clearance-limited dilation...')
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
        mask=None  # already filtered to anterior+exposed
    )
    
    fatpad_dilated.point_data['offset_mm'] = per_vertex_mm
    
    # Step 4: Filter by proximity to patella and vertical position
    logger.info('Filtering fatpad region...')
    
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
    
    logger.info(f'Points passing radial filter: {np.sum(keep_radial)} / {len(keep_radial)}')
    logger.info(f'Points passing vertical filter: {np.sum(keep_vertical)} / {len(keep_vertical)}')
    logger.info(f'Points passing both: {np.sum(keep)} / {len(keep)}')
    
    if np.sum(keep) == 0:
        raise ValueError("All points filtered out. Adjust thresholds.")
    
    fatpad_dilated.point_data['keep'] = keep.astype(float)
    fatpad = fatpad_dilated.remove_points(~keep)[0]
    fatpad = Mesh(fatpad)
    
    # Step 5: Finalize
    logger.info(f'Final resampling to {resample_clusters_final} clusters...')
    fatpad = _finalize_fatpad_mesh(fatpad, resample_clusters_final, units, final_smooth_iter, output_path)
    
    logger.info('Prefemoral fatpad (no-boolean) created successfully')
    return fatpad


def optimize_patella_position(
    pat_articular_surfaces,
    fem_articular_surfaces,
    pat_mesh_osim,
    patella_adjust_rel_vert_overlap=None,  # 0.4
    patella_adjust_abs_vert_overlap=None,  # 0.012 
    patella_adjust_contact_area=None,      # 0.2
    contact_area_adjustment = np.array([0, 0.001, 0]),
    return_move_down=False
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

    percent_non_zero, vert_overlap, total_vert, percent_vert_overlap = compute_overlap_metrics(pat_articular_surfaces, fem_articular_surfaces)

    logger.info(f'Percent of patella cartilage that is not overlapping with femoral cartilage: {percent_non_zero:.2f}%')
    logger.info(f'Percent vertical overlap: {percent_vert_overlap:.2f}%')
    logger.info(f'Vertical overlap: {vert_overlap*100:.2f} cm')
    logger.info(f'Total vertical: {total_vert*100:.2f} cm')

    # if the vertical overlap is less than 40%, then move the patella down
    # by the amount necessary to increase the overlap to 40%
    move_down_rel_vert = 0
    move_down_abs_vert = 0
    move_down_contact_area = np.asarray([0., 0., 0.])
    if patella_adjust_rel_vert_overlap is not None:
        if percent_vert_overlap < patella_adjust_rel_vert_overlap: #0.4
            logger.info('Adjusting patella position - % vertical overlap to be 40%')
            move_down_rel_vert = (40 - percent_vert_overlap) * total_vert
            pat_articular_surfaces.point_coords[:, 1] -= move_down_rel_vert
            pat_mesh_osim.point_coords[:, 1] -= move_down_rel_vert
        else:
            pass
    if patella_adjust_abs_vert_overlap is not None:
        if vert_overlap < patella_adjust_abs_vert_overlap: #0.012
            logger.info('Adjusting patella position - vertical overlap to be 1.2 cm')
            move_down_abs_vert = patella_adjust_abs_vert_overlap - vert_overlap #0.012
            pat_articular_surfaces.point_coords[:, 1] -= move_down_abs_vert
            pat_mesh_osim.point_coords[:, 1] -= move_down_abs_vert
        else:
            pass
    if patella_adjust_contact_area is not None:
        while percent_non_zero < patella_adjust_contact_area: #0.2
            logger.info(f'Percent non-zero: {percent_non_zero}')
            logger.info('Adjusting patella down 1mm')
            pat_articular_surfaces.point_coords -= contact_area_adjustment
            pat_mesh_osim.point_coords -= contact_area_adjustment
            move_down_contact_area += contact_area_adjustment
            percent_non_zero, _, _, _ = compute_overlap_metrics(pat_articular_surfaces, fem_articular_surfaces)
    
    move_down_total = np.array([0, move_down_rel_vert, 0]) + np.array([0, move_down_abs_vert, 0]) + move_down_contact_area
    
    if return_move_down:
        return pat_articular_surfaces, pat_mesh_osim, move_down_total
    else:
        return pat_articular_surfaces, pat_mesh_osim
