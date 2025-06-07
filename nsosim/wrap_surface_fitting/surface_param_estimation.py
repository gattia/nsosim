import torch
import numpy as np
import warnings

def get_near_surface_points_from_mesh(mesh, surface_name):
    """Get near-surface points from pre-labeled mesh data.
    
    Args:
        mesh: Mesh object with point_data containing near-surface labels
        surface_name: Name of the surface (e.g., 'femur_1')
        
    Returns:
        np.ndarray: Points near the surface
    """
    near_surface_key = f"{surface_name}_near_surface"
    if near_surface_key not in mesh.mesh.point_data:
        raise ValueError(f"Near-surface data '{near_surface_key}' not found in mesh. "
                       f"Available keys: {list(mesh.mesh.point_data.keys())}")
    
    near_surface_mask = mesh.mesh.point_data[near_surface_key].astype(bool)
    points = mesh.point_coords.copy()
    return points[near_surface_mask]

def estimate_cylinder_center_z_axis(near_surface_points, slice_thickness=1.0):
    """Estimate cylinder center assuming long axis ≈ [0,0,1].
    
    Uses robust z-slice approach to handle uneven point distributions.
    
    Args:
        near_surface_points: (N, 3) Points within threshold of surface
        slice_thickness: Thickness of z-slices in mm
        
    Returns:
        torch.Tensor: (3,) Estimated center point
    """
    if len(near_surface_points) == 0:
        raise ValueError("No near-surface points provided")
        
    z_coords = near_surface_points[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    
    # Create z-slice centers
    z_centers = torch.arange(z_min, z_max + slice_thickness, slice_thickness, 
                            device=near_surface_points.device)
    
    slice_centroids = []
    
    for z_center in z_centers:
        # Points in this slice
        in_slice = torch.abs(z_coords - z_center) <= slice_thickness / 2
        slice_points = near_surface_points[in_slice]
        
        if len(slice_points) > 0:
            # xy-centroid of this slice
            xy_centroid = slice_points[:, :2].mean(0)
            slice_centroids.append(torch.cat([xy_centroid, z_center.unsqueeze(0)]))
    
    if len(slice_centroids) == 0:
        # Fallback to simple centroid
        return near_surface_points.mean(0)
        
    # Average all slice centroids
    return torch.stack(slice_centroids).mean(0)

def fit_cylinder_geometric(near_surface_points, num_slices=20):
    """Fit cylinder to points using geometric method with z-slice analysis.
    
    Uses z-slice centroids to fit the central axis, then computes radius and height.
    This version uses a specified number of slices across the z-extent.
    
    Args:
        near_surface_points: (N, 3) Points within threshold of surface  
        num_slices: Desired number of slices along the z-axis
        
    Returns:
        dict: Contains 'center', 'radius', 'half_length', 'axis', 'rotation', 'success'
    """
    if len(near_surface_points) < 6:
        raise ValueError("Need at least 6 points for cylinder fitting")
    
    # Convert to torch if needed
    if isinstance(near_surface_points, np.ndarray):
        points = torch.from_numpy(near_surface_points).float()
    else:
        points = near_surface_points
        
    print(f"[DEBUG] fit_cylinder_geometric input:")
    print(f"  Points shape: {points.shape}")
    z_min_val, z_max_val = points[:, 2].min().item(), points[:, 2].max().item()
    print(f"  Z-extent: {z_min_val:.6f} to {z_max_val:.6f} (range: {(z_max_val - z_min_val):.6f})")
    print(f"  XY-extent: X[{points[:, 0].min():.6f}, {points[:, 0].max():.6f}], Y[{points[:, 1].min():.6f}, {points[:, 1].max():.6f}]")
        
    try:
        z_coords = points[:, 2]
        z_min, z_max = z_coords.min(), z_coords.max()
        z_range = z_max - z_min

        # Determine slice_thickness and z_centers based on num_slices
        if num_slices < 2:
            warnings.warn(f"num_slices is {num_slices}, which is less than 2. Geometric cylinder fitting requires at least 2 slices and will likely fallback.")
            # Force fallback by ensuring few/no valid slices.
            slice_thickness = z_range * 2.0 if z_range > 1e-6 else 1.0 # Make slice very thick or default
            # Generate z_centers that will lead to len(slice_centroids) < 2
            if num_slices == 1:
                z_centers = torch.tensor([(z_min + z_max) / 2.0], device=points.device)
            else: # num_slices == 0 or negative
                z_centers = torch.empty(0, device=points.device)

        elif z_range <= 1e-6: # Points are essentially co-planar in Z
            warnings.warn("Z-range of points is very small (<= 1e-6). Cylinder is likely flat or points are co-planar. Geometric fitting will likely fallback.")
            # Force fallback by ensuring only one conceptual slice is processed effectively
            slice_thickness = 1.0 # Arbitrary thickness for point selection, as all points are at effectively the same z
            z_centers = torch.tensor([(z_min + z_max) / 2.0], device=points.device) # Single center
        else: # Normal case: num_slices >= 2 and z_range is significant
            slice_thickness = z_range / num_slices # This is the thickness of each of the num_slices segments
            # Generate centers of each slice
            z_centers = torch.linspace(z_min + slice_thickness / 2.0, 
                                       z_max - slice_thickness / 2.0, 
                                       num_slices, device=points.device)

        slice_centroids = []
        slice_z_coords = []
        
        print(f"[DEBUG] Target {num_slices} slices. Z-range: {z_range.item():.6f}")
        if z_range > 1e-6 and num_slices >=2: # Only print calculated slice_thickness if it's meaningful for segmentation
             print(f"[DEBUG] Calculated slice thickness for segmenting z-range: {(z_range / num_slices).item():.6f}")
        print(f"[DEBUG] Number of z-centers generated for iteration: {len(z_centers)}. Effective slice_thickness for point selection in loop: {slice_thickness.item():.6f}")
        
        for z_center in z_centers:
            # Points in this slice
            in_slice = torch.abs(z_coords - z_center) <= slice_thickness / 2.0
            slice_points = points[in_slice]
            
            if len(slice_points) >= 3:  # Need minimum points for meaningful centroid
                xy_centroid = slice_points[:, :2].mean(0)
                slice_centroids.append(xy_centroid)
                slice_z_coords.append(z_center)
        
        if len(slice_centroids) < 2:
            raise ValueError("Not enough valid z-slices for axis fitting")
        
        slice_centroids = torch.stack(slice_centroids)
        slice_z_coords = torch.stack(slice_z_coords)
        
        print(f"[DEBUG] Valid slices: {len(slice_centroids)}")
        print(f"[DEBUG] Slice centroid XY ranges: X[{slice_centroids[:, 0].min():.6f}, {slice_centroids[:, 0].max():.6f}], Y[{slice_centroids[:, 1].min():.6f}, {slice_centroids[:, 1].max():.6f}]")
        
        # Fit a line through the slice centroids to determine the actual axis direction
        # Stack the 3D centroids (xy_centroids + z_coords)
        centroids_3d = torch.stack([
            torch.cat([slice_centroids[i], slice_z_coords[i].unsqueeze(0)]) 
            for i in range(len(slice_centroids))
        ])
        
        # Compute the axis direction using PCA on the centroids
        centroids_mean = centroids_3d.mean(0)
        centered_centroids = centroids_3d - centroids_mean
        
        # Compute covariance matrix and get principal direction
        cov_matrix = torch.mm(centered_centroids.T, centered_centroids) / (len(centroids_3d) - 1)
        eigenvals, eigenvecs = torch.linalg.eigh(cov_matrix)
        
        # The axis is the direction of maximum variance (largest eigenvalue)
        axis_3d = eigenvecs[:, -1]  # Last column corresponds to largest eigenvalue
        
        # Ensure axis points in positive z direction (roughly)
        if axis_3d[2] < 0:
            axis_3d = -axis_3d
            
        print(f"[DEBUG] Estimated axis from centroids: [{axis_3d[0]:.6f}, {axis_3d[1]:.6f}, {axis_3d[2]:.6f}]")
        
        # The center is the centroid of all slice centroids
        center_3d = centroids_mean
        
        # Compute radius: average distance from the fitted axis
        # For each point, compute perpendicular distance to the axis line
        point_to_center = points - center_3d
        proj_on_axis = torch.sum(point_to_center * axis_3d, dim=1, keepdim=True)
        perpendicular_dist = torch.norm(point_to_center - proj_on_axis * axis_3d, dim=1)
        radius = perpendicular_dist.mean()
        
        print(f"[DEBUG] Radius computation:")
        print(f"  Center 3D: [{center_3d[0]:.6f}, {center_3d[1]:.6f}, {center_3d[2]:.6f}]")
        print(f"  Axis 3D: [{axis_3d[0]:.6f}, {axis_3d[1]:.6f}, {axis_3d[2]:.6f}]")
        print(f"  Perpendicular distance stats: min={perpendicular_dist.min():.6f}, max={perpendicular_dist.max():.6f}, mean={radius:.6f}")
        
        # Compute half-length: half of the extent along the axis direction
        proj_lengths = torch.sum(point_to_center * axis_3d, dim=1)
        half_length = (proj_lengths.max() - proj_lengths.min())
        
        print(f"[DEBUG] Half-length: {half_length:.6f} (half of axis extent)")
        
        # Compute rotation matrix to align with axis
        # We want to rotate the standard z-axis [0,0,1] to our axis direction
        z_axis = torch.tensor([0., 0., 1.], device=points.device, dtype=points.dtype)
        
        # If axis is already aligned with z, no rotation needed
        if torch.allclose(axis_3d, z_axis, atol=1e-6):
            rotation_matrix = torch.eye(3, device=points.device, dtype=points.dtype)
        else:
            # Compute rotation matrix using Rodriguez formula
            v = torch.cross(z_axis, axis_3d)
            s = torch.norm(v)
            c = torch.dot(z_axis, axis_3d)
            
            if s < 1e-6:  # Nearly parallel
                rotation_matrix = torch.eye(3, device=points.device, dtype=points.dtype)
            else:
                vx = torch.tensor([[0, -v[2], v[1]], 
                                   [v[2], 0, -v[0]], 
                                   [-v[1], v[0], 0]], device=points.device, dtype=points.dtype)
                rotation_matrix = torch.eye(3, device=points.device, dtype=points.dtype) + vx + torch.mm(vx, vx) * ((1 - c) / (s * s))
        
        return {
            'center': center_3d,
            'radius': radius,
            'half_length': half_length,
            'axis': axis_3d,
            'rotation': rotation_matrix,
            'success': True
        }
        
    except Exception as e:
        warnings.warn(f"Geometric cylinder fitting failed: {e}. Using fallback.")
        
        # Fallback to simple estimates
        center = points.mean(0)
        z_extent = points[:, 2].max() - points[:, 2].min()
        half_length = z_extent / 2.0
        
        # Simple radius estimate from xy spread
        xy_center = center[:2]
        xy_distances = torch.norm(points[:, :2] - xy_center, dim=1)
        radius = xy_distances.mean()
        
        return {
            'center': center,
            'radius': radius,
            'half_length': half_length,
            'axis': torch.tensor([0., 0., 1.], device=points.device, dtype=points.dtype),
            'rotation': torch.eye(3, device=points.device, dtype=points.dtype),
            'success': False
        }

def fit_ellipsoid_algebraic(points):
    """Fit ellipsoid to points using algebraic method.
    
    Fits the general ellipsoid equation:
    Ax² + By² + Cz² + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0
    
    Args:
        points: (N, 3) Points to fit ellipsoid to
            
    Returns:
        dict: Contains 'center', 'axes', 'rotation' of fitted ellipsoid
    """
    if len(points) < 9:
        raise ValueError("Need at least 9 points for ellipsoid fitting")
        
    # Ensure points is a torch tensor and float
    if isinstance(points, np.ndarray):
        points_tensor = torch.from_numpy(points).float()
    else:
        points_tensor = points.float()
        
    x, y, z = points_tensor[:, 0], points_tensor[:, 1], points_tensor[:, 2]
    
    # Design matrix for general quadric equation
    # Ax² + By² + Cz² + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0
    D_matrix = torch.stack([
        x*x, y*y, z*z, x*y, x*z, y*z, x, y, z, torch.ones_like(x)
    ], dim=1)
    
    # Solve the homogeneous system using SVD
    # We want the least squares solution with constraint ||coeffs|| = 1
    _, _, V = torch.linalg.svd(D_matrix)
    coeffs = V[-1, :]  # Last row of V corresponds to smallest singular value
    
    A_c, B_c, C_c, D_coeff_c, E_c, F_c, G_c, H_c, I_c, J_c = coeffs
    
    # Convert to center-form ellipsoid
    # This involves solving a 3x3 linear system for the center
    try:
        # Matrix of quadratic terms
        Q = torch.tensor([
            [A_c, D_coeff_c/2, E_c/2],
            [D_coeff_c/2, B_c, F_c/2], 
            [E_c/2, F_c/2, C_c]
        ], device=points_tensor.device, dtype=points_tensor.dtype)
        
        # Linear terms vector
        linear_terms = torch.tensor([G_c, H_c, I_c], device=points_tensor.device, dtype=points_tensor.dtype)
        
        # Solve for center: Q * center = -linear_terms/2
        center = torch.linalg.solve(Q, -linear_terms / 2)
        
        # Calculate the constant term J_prime for the centered equation
        # J_prime = center.T @ Q @ center + linear_terms.T @ center + J_c
        # (This is equivalent to F(center_x, center_y, center_z) using original coefficients)
        xc, yc, zc = center[0], center[1], center[2]
        J_prime = (A_c*xc*xc + B_c*yc*yc + C_c*zc*zc + 
                   D_coeff_c*xc*yc + E_c*xc*zc + F_c*yc*zc + 
                   G_c*xc + H_c*yc + I_c*zc + J_c)

        print(f"[DEBUG] fit_ellipsoid_algebraic internal:")
        print(f"  Coeffs (A..J): {coeffs.tolist()}")
        print(f"  Center: {center.tolist()}")
        print(f"  J_prime (constant term in centered eq): {J_prime.item()}")

        eigenvals, eigenvecs = torch.linalg.eigh(Q)
        print(f"  Eigenvalues of Q: {eigenvals.tolist()}")

        # Scale factor for eigenvalues to get 1/axes^2
        # We need X'^T (Q / -J_prime) X' = 1
        # So, axes_i = sqrt(-J_prime / eigenvals_i(Q))
        if J_prime.item() == 0:
            raise ValueError("J_prime is zero, degenerate quadric.")
        if (-J_prime / eigenvals).min() < 0 and not torch.allclose((-J_prime / eigenvals).min(), torch.tensor(0.0, dtype=points_tensor.dtype)):
            warnings.warn(f"Non-positive value for axes squared: {-J_prime / eigenvals}. Might not be an ellipsoid. Values: {(-J_prime / eigenvals).tolist()}")
            
        axes_squared_inv = eigenvals / (-J_prime)
        axes = 1.0 / torch.sqrt(torch.abs(axes_squared_inv) + 1e-9) # Add epsilon for stability if abs is used
        # A more direct and potentially stabler way for positive definite Q and -J_prime > 0:
        # axes = torch.sqrt(torch.abs(-J_prime / eigenvals))
        print(f"  Calculated Axes: {axes.tolist()}")

        rotation = eigenvecs
        
        # Ensure the rotation matrix corresponds to a right-handed coordinate system.
        # A negative determinant (i.e., close to -1) indicates a left-handed system (a reflection).
        if torch.linalg.det(rotation) < 0:
            print(f"[DEBUG] Original rotation determinant: {torch.linalg.det(rotation).item()}")
            # Create a mutable copy before modification.
            rotation_corrected = rotation.clone() 
            # Flip one of the axes (e.g., the last one) to change the handedness of the system.
            # This ensures the resulting matrix represents a proper rotation (right-handed).
            rotation_corrected[:, -1] = -rotation_corrected[:, -1]
            rotation = rotation_corrected
            print(f"[DEBUG] Corrected rotation determinant: {torch.linalg.det(rotation).item()}")
        
        return {
            'center': center,
            'axes': axes, 
            'rotation': rotation,
            'success': True
        }
        
    except Exception as e:
        warnings.warn(f"Algebraic ellipsoid fitting failed: {e}. Using centroid.")
        return {
            'center': points_tensor.mean(0),
            'axes': points_tensor.std(0) * 2,  # Simple estimate
            'rotation': torch.eye(3, device=points_tensor.device, dtype=points_tensor.dtype),
            'success': False
        }
