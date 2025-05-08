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
    Remove cartilage points that are inside the bone and clean up the resulting mesh.
    
    Args:
    cartilage_mesh (Mesh, pyvista.PolyData, or vtk.vtkPolyData): The articular surface mesh
    bone_mesh (Mesh, pyvista.PolyData, or vtk.vtkPolyData): The bone surface mesh
    
    Returns:
    Mesh or pyvista.PolyData: The cleaned cartilage mesh (type matches input)
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
    Remove isolated cells from a mesh that have only one edge neighbor.
    
    Parameters:
    -----------
    input_mesh : Mesh, pyvista.PolyData, or vtk.vtkPolyData
        The input mesh to clean.
    
    Returns:
    --------
    cleaned_mesh : Same type as input_mesh or pyvista.PolyData
        The cleaned mesh with isolated cells removed.
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
    list_articular_surfaces = []

    for cart_mesh in bone_mesh.list_cartilage_meshes:

        print(cart_mesh.point_coords.shape)
        print(bone_mesh.point_coords.shape)
        articular_surface = remove_intersecting_vertices(
            pv.PolyData(cart_mesh.mesh),
            pv.PolyData(bone_mesh.mesh),
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
