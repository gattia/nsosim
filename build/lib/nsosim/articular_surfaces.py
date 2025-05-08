import numpy as np
import pyvista as pv

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
    
    return subregions.connectivity(extraction_mode='specified', variable_input=largest_n)


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
        articular_surface = get_n_largest(articular_surface, n=n_largest)
        articular_surface = articular_surface.smooth(n_iter=smooth_iter)
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