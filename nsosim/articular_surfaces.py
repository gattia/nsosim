import numpy as np
import pyvista as pv
from pymskt.mesh import Mesh
import vtk

# extract the articular surfaces from the cartilages
def remove_intersecting_vertices(mesh1, mesh2, ray_length=1.0, overlap_buffer=0.1):
    """
    This function takes in two meshes: mesh1 and mesh2.
    Rays are cast from each vertex of mesh1 in the negative direction of the normal to the surface of mesh1.
    If a ray intersects mesh2, the vertex from which the ray was cast is marked for removal.
    A version of mesh1 with the marked vertices removed is returned.
    
    Parameters:
    - ray_length: The length of the ray. Default is 1.0.
    - invert: If True, the vertices marked for removal are kept and the rest are removed.
              If False, the vertices marked for removal are removed and the rest are kept.
              Default is True.
    """
    
    # Compute point normals for mesh1
    mesh1_normals = mesh1.compute_normals(point_normals=True, cell_normals=False)
    
    vertex_mask = np.ones(mesh1.n_points, dtype=bool)
    
    for idx, (vertex, normal) in enumerate(zip(mesh1.points, mesh1_normals.point_data["Normals"])):
        start_point = vertex - overlap_buffer * normal
        end_point = vertex - ray_length * normal
        
        intersections = mesh2.ray_trace(start_point, end_point)
        
        # If there's any intersection
        if len(intersections[1]) > 0:
            vertex_mask[idx] = False
    
    print(f'number of intersections: {sum(~vertex_mask)}')
    
    # Use the mask to filter out the vertices and the associated cells from mesh1
    mesh1.point_data['vertex_mask'] = vertex_mask
    cleaned_mesh = mesh1.threshold(0.5, scalars='vertex_mask', invert=True)
    
    
    return cleaned_mesh.extract_surface()


def get_n_largest(surface, n=1):
    """
    Extracts the n largest connected components from a surface mesh.

    The function identifies connected regions within the input surface, sorts them by the
    number of cells (as a proxy for area), and returns a new surface containing only
    the n largest regions.

    Args:
        surface (pyvista.PolyData): The input surface mesh from which to extract components.
        n (int, optional): The number of largest components to extract. Defaults to 1.

    Returns:
        pyvista.PolyData: A new surface mesh containing the n largest connected components.

    Raises:
        AssertionError: If the input `surface` or the intermediate `subregions`
                        are not pyvista.PolyData objects.
    """
    subregions = surface.connectivity('all')
    unique_regions = np.unique(subregions['RegionId'])
    # getting the first "n" because the outputs are sorted by # of cells
    # assume all cells are ~ the same size, therefore largest # cells ~= largest areas
    largest_n = unique_regions[:n]
    
    assert isinstance(surface, pv.PolyData), f'surface is not a PolyData object: {type(surface)}'
    assert isinstance(subregions, pv.PolyData), f'subregions is not a PolyData object: {type(subregions)}'
    
    return subregions.connectivity(extraction_mode='specified', variable_input=largest_n)

def remove_cart_in_bone(cartilage_mesh, bone_mesh):
    """
    Removes portions of a cartilage mesh that are located inside a bone mesh.

    This function identifies and removes vertices (and their associated cells) from the
    cartilage mesh that fall within the volume of the bone mesh. The determination
    is based on calculating the surface error (distance) from the cartilage to the
    bone; negative or zero distances indicate points inside or on the bone surface.

    Args:
        cartilage_mesh (pymskt.mesh.Mesh, pyvista.PolyData, or vtk.vtkPolyData):
            The cartilage mesh to be cleaned.
        bone_mesh (pymskt.mesh.Mesh, pyvista.PolyData, or vtk.vtkPolyData):
            The bone mesh used as the reference for removal.

    Returns:
        pymskt.mesh.Mesh or pyvista.PolyData: The cleaned cartilage mesh, with parts
            inside the bone removed. The return type matches the input type of
            `cartilage_mesh`.

    Raises:
        TypeError: If `cartilage_mesh` or `bone_mesh` are not of the expected types.
    """
    # Check and convert input types
    def convert_to_mesh(mesh, mesh_name):
        if isinstance(mesh, Mesh):
            return mesh, 'Mesh'
        elif isinstance(mesh, pv.PolyData):
            return Mesh(mesh), 'pyvista'
        elif isinstance(mesh, vtk.vtkPolyData):
            return Mesh(pv.PolyData(mesh)), 'vtk'
        else:
            raise TypeError(f"The {mesh_name} is not of type pymskt.mesh.Mesh, pyvista.PolyData, or vtk.vtkPolyData")

    cartilage_mesh, cart_type = convert_to_mesh(cartilage_mesh, "cartilage mesh")
    bone_mesh, bone_type = convert_to_mesh(bone_mesh, "bone mesh")
    
    # Ensure both meshes have the same dtype (use the higher precision one)
    target_dtype = np.promote_types(cartilage_mesh.point_coords.dtype, bone_mesh.point_coords.dtype)
    cartilage_mesh.point_coords = cartilage_mesh.point_coords.astype(target_dtype)
    bone_mesh.point_coords = bone_mesh.point_coords.astype(target_dtype)

    # Create a copy of the cartilage mesh
    cart_copy = cartilage_mesh.copy()
    cart_copy.mesh = pv.PolyData(cart_copy.mesh)
    
    # Calculate the surface error
    cart_copy.calc_surface_error(bone_mesh)
    surf_error = cart_copy.get_scalar('surface_error')
    
    # Invert the surface error values
    cart_copy.set_scalar('surface_error', surf_error * -1)

    # Threshold the surface to keep only points outside the bone (surface_error > 0)
    cart_copy.mesh = cart_copy.mesh.threshold(0, scalars='surface_error', invert=True).extract_surface()
    # Clean up the resulting mesh
    cart_copy.mesh = cart_copy.mesh.clean()
    
    # Return the appropriate type
    if cart_type == 'Mesh':
        return cart_copy
    else:  # 'pyvista' or 'vtk'
        return cart_copy.mesh


def remove_isolated_cells(input_mesh):
    """
    Removes isolated cells from a mesh iteratively.

    An isolated cell is defined as a cell that has only one edge neighbor.
    The function repeatedly identifies and removes such cells until no more
    isolated cells are found. The mesh is then cleaned to remove unused points.

    Args:
        input_mesh (pymskt.mesh.Mesh, pyvista.PolyData, or vtk.vtkPolyData):
            The input mesh to clean.

    Returns:
        pymskt.mesh.Mesh or pyvista.PolyData: The cleaned mesh with isolated
            cells removed. The return type matches the input type of `input_mesh`,
            or pyvista.PolyData if the input was vtk.vtkPolyData.

    Raises:
        TypeError: If `input_mesh` is not of the expected types.
    """
    # Type checking and conversion
    if isinstance(input_mesh, Mesh):
        mesh = pv.PolyData(input_mesh.mesh)
        return_type = 'Mesh'
    elif isinstance(input_mesh, pv.PolyData):
        mesh = input_mesh.copy()
        return_type = 'pyvista'
    elif isinstance(input_mesh, vtk.vtkPolyData):
        mesh = pv.PolyData(input_mesh)
        return_type = 'pyvista'
    else:
        raise TypeError("Input mesh must be of type Mesh, pyvista.PolyData, or vtk.vtkPolyData")

    n_cells_removed = 1
    while n_cells_removed > 0:
        n_cells = mesh.n_cells
        cell_mask = np.ones(n_cells, dtype=bool)
        
        for i in range(n_cells):
            cell_neighbors = mesh.cell_neighbors(i, connections='edges')
            if len(cell_neighbors) == 1:
                cell_mask[i] = False
        
        mesh.cell_data['cell_mask'] = cell_mask
        mesh = mesh.threshold(0.5, scalars='cell_mask', invert=False).extract_surface()
        n_cells_removed = n_cells - mesh.n_cells

    # clean the mesh
    mesh = mesh.clean()
    
    # Return the cleaned mesh in the appropriate type
    if return_type == 'Mesh':
        cleaned_mesh = Mesh()
        cleaned_mesh.mesh = mesh
    else:
        cleaned_mesh = mesh

    return cleaned_mesh


def extract_articular_surface(bone_mesh, ray_length=10.0, smooth_iter=100, n_largest=1):
    """
    Extracts articular surfaces from cartilage meshes associated with a bone mesh.

    For each cartilage mesh linked to the `bone_mesh`:
    1.  Identifies cartilage vertices that do not intersect the bone (using ray tracing).
    2.  Extracts the `n_largest` connected components from the resulting surface.
    3.  Removes any remaining cartilage portions that are inside the bone.
    4.  Removes isolated cells from the boundaries.
    5.  Smoothes the final articular surface.

    Args:
        bone_mesh (pymskt.mesh.Mesh): The bone mesh, which should have a
            `list_cartilage_meshes` attribute containing associated cartilage meshes
            (as pymskt.mesh.Mesh objects).
        ray_length (float, optional): The length of the rays used for intersection
            testing. Defaults to 10.0.
        smooth_iter (int, optional): The number of smoothing iterations to apply to
            the final articular surface. Defaults to 100.
        n_largest (int, optional): The number of largest connected components to
            keep after initial intersection removal. Defaults to 1.

    Returns:
        list[pyvista.PolyData]: A list of `pyvista.PolyData` objects, each
            representing an extracted and processed articular surface.
    """
    list_articular_surfaces = []
    
    bone_mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True, inplace=True)

    for cart_mesh in bone_mesh.list_cartilage_meshes:
        
        cart_mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True, inplace=True)
        
        print(cart_mesh.point_coords.shape)
        print(bone_mesh.point_coords.shape)
        articular_surface = remove_intersecting_vertices(
            cart_mesh,
            bone_mesh,
            ray_length=ray_length,
        )
        assert isinstance(articular_surface, pv.PolyData), f'articular_surface is not a PolyData object: {type(articular_surface)}'
        
        articular_surface = get_n_largest(articular_surface, n=n_largest)
        if not isinstance(articular_surface, pv.PolyData):
            articular_surface = articular_surface.extract_surface()
        assert isinstance(articular_surface, pv.PolyData), f'articular_surface is not a PolyData object: {type(articular_surface)}'
        
        # remove articular surface points that are inside the bone
        articular_surface = remove_cart_in_bone(articular_surface, bone_mesh)
        # remove isolated cells at the boundaries
        articular_surface = remove_isolated_cells(articular_surface)
        
        # smooth the articular surface...
        #   boundary_smoothing=False will enable smoothing at the boundary - which can fix
        #   some of the issues with errors at the edges (boundaries)
        articular_surface = articular_surface.smooth(n_iter=smooth_iter, boundary_smoothing=False)
        
        list_articular_surfaces.append(articular_surface)
    
    return list_articular_surfaces

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
        
        print(f'current density: {current_density/1_000_000} triangles/mm^2')
        print(f'target density: {triangle_density/1_000_000} triangles/mm^2')
        print(f'current number of points: {meniscus_mesh.point_coords.shape[0]}')
        print(f'target number of points: {meniscus_clusters}')
        
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
        print(f'achieved density: {updated_density/1_000_000} triangles/mm^2')
    
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

        print(f'current density: {current_density/1_000_000} triangles/mm^2')
        print(f'target density: {triangle_density/1_000_000} triangles/mm^2')
        print(f'current number of points: {cart_mesh_osim.point_coords.shape[0]}')
        print(f'target number of points: {cart_clusters}')

    # resample bone surface to be 10k points to reduce computation time
    print('resampling femur surface')
    bone_mesh_osim_ = bone_mesh_osim.copy()
    bone_mesh_osim_.point_coords = bone_mesh_osim_.point_coords * 1000
    if bone_clusters is not None:
        bone_mesh_osim_.resample_surface(subdivisions=1, clusters=bone_clusters)

    # assign cartilage to bone
    print('resample cartilage surface')
    cart_mesh_osim_ = cart_mesh_osim.copy()
    cart_mesh_osim_.point_coords = cart_mesh_osim_.point_coords * 1000
    if cart_clusters is not None:
        # if not isinstance(cart_mesh_osim_.mesh, pv.PolyData):
        #     cart_mesh_osim_.mesh = pv.PolyData(cart_mesh_osim_.mesh)
        # cart_mesh_osim_.mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True, inplace=True)
        cart_mesh_osim_.resample_surface(subdivisions=1, clusters=cart_clusters)
        # density after resampling
        updated_density = cart_mesh_osim_.mesh.n_cells/cart_mesh_osim_.mesh.area
        print(f'achieved density: {updated_density/1_000_000} triangles/mm^2')
    cart_mesh_osim_.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True, inplace=True)
    bone_mesh_osim_.list_cartilage_meshes = [cart_mesh_osim_]
    

    # extract articular surface
    print('extracting articular surface')
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

    print(f'Percent of patella cartilage that is not overlapping with femoral cartilage: {percent_non_zero:.2f}%')
    print(f'Percent vertical overlap: {percent_vert_overlap:.2f}%')
    print(f'Vertical overlap: {vert_overlap*100:.2f} cm')
    print(f'Total vertical: {total_vert*100:.2f} cm')

    # if the vertical overlap is less than 40%, then move the patella down
    # by the amount necessary to increase the overlap to 40%
    move_down_rel_vert = 0
    move_down_abs_vert = 0
    move_down_contact_area = np.asarray([0., 0., 0.])
    if patella_adjust_rel_vert_overlap is not None:
        if percent_vert_overlap < patella_adjust_rel_vert_overlap: #0.4
            print('Adjusting patella position - % vertical overlap to be 40%')
            move_down_rel_vert = (40 - percent_vert_overlap) * total_vert
            pat_articular_surfaces.point_coords[:, 1] -= move_down_rel_vert
            pat_mesh_osim.point_coords[:, 1] -= move_down_rel_vert
        else:
            pass
    if patella_adjust_abs_vert_overlap is not None:
        if vert_overlap < patella_adjust_abs_vert_overlap: #0.012
            print('Adjusting patella position - vertical overlap to be 1.2 cm')
            move_down_abs_vert = patella_adjust_abs_vert_overlap - vert_overlap #0.012
            pat_articular_surfaces.point_coords[:, 1] -= move_down_abs_vert
            pat_mesh_osim.point_coords[:, 1] -= move_down_abs_vert
        else:
            pass
    if patella_adjust_contact_area is not None:
        while percent_non_zero < patella_adjust_contact_area: #0.2
            print('Percent non-zero:', percent_non_zero)
            print('Adjusting patella down 1mm')
            pat_articular_surfaces.point_coords -= contact_area_adjustment
            pat_mesh_osim.point_coords -= contact_area_adjustment
            move_down_contact_area += contact_area_adjustment
            percent_non_zero, _, _, _ = compute_overlap_metrics(pat_articular_surfaces, fem_articular_surfaces)
    
    move_down_total = np.array([0, move_down_rel_vert, 0]) + np.array([0, move_down_abs_vert, 0]) + move_down_contact_area
    
    if return_move_down:
        return pat_articular_surfaces, pat_mesh_osim, move_down_total
    else:
        return pat_articular_surfaces, pat_mesh_osim
