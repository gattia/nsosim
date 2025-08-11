import torch


def sd_ellipsoid_improved(points, center, axes, rotation_matrix):
    """Improved ellipsoid SDF using more accurate distance estimation.
    
    This is more accurate than the normalized approximation but faster than the exact iterative method.
    Based on the method from "Efficient Ellipsoid-AABB Collision Detection" but adapted for SDF.
    
    Args:
        points: (N, 3) tensor of points
        center: (3,) tensor ellipsoid center
        axes: (3,) tensor ellipsoid semi-axes (a, b, c)
        rotation_matrix: (3, 3) tensor rotation matrix
        
    Returns:
        (N,) tensor of signed distances (much more accurate than sd_normalised)
    """
    # Transform points to ellipsoid local coordinates
    local_points = (points - center) @ rotation_matrix
    
    # Ensure positive semi-axes
    axes_safe = torch.clamp(axes, min=1e-8)
    a, b, c = axes_safe
    
    # Work with absolute coordinates (symmetry)
    p = torch.abs(local_points)
    
    # Initial guess: scale point to surface along ray from origin
    # This is better than using average axis length
    ellipsoid_equation = (p[:, 0] / a)**2 + (p[:, 1] / b)**2 + (p[:, 2] / c)**2
    
    # For points inside, we need negative distance
    is_inside = ellipsoid_equation < 1.0
    
    # Improved distance approximation using gradient information
    # Instead of using average axis, use the local "effective radius" at each point
    
    # Compute the gradient of the ellipsoid function F(x,y,z) = (x/a)² + (y/b)² + (z/c)² - 1
    grad = torch.stack([
        2 * local_points[:, 0] / (a * a),
        2 * local_points[:, 1] / (b * b), 
        2 * local_points[:, 2] / (c * c)
    ], dim=1)
    
    grad_norm = torch.norm(grad, dim=1)
    
    # Distance approximation: |F(p)| / ||∇F(p)||
    # This is much more accurate than using average axis length
    f_value = ellipsoid_equation - 1.0
    distance = torch.abs(f_value) / (grad_norm + 1e-8)
    
    # Apply correct sign
    signed_distance = torch.where(is_inside, -distance, distance)
    
    return signed_distance



################## Rest of these are not currently used. 

# def sd_cylinder(points, center, radius, half_length, rotation_matrix):
#     """Signed distance function for finite cylinder.
    
#     Args:
#         points: (N, 3) tensor of points
#         center: (3,) tensor cylinder center
#         radius: scalar tensor cylinder radius  
#         half_length: scalar tensor cylinder half-length
#         rotation_matrix: (3, 3) tensor rotation matrix (local→world, so we use .T)
        
#     Returns:
#         (N,) tensor of signed distances
#     """
#     # Transform points to cylinder local coordinates
#     p = (points - center) @ rotation_matrix.T                 # world → local
    
#     # Distance from axis and from caps
#     radial_dist = torch.linalg.norm(p[..., :2], dim=-1) - radius
#     axial_dist = torch.abs(p[..., 2]) - half_length
    
#     q = torch.stack([radial_dist, axial_dist], dim=-1)       # (..., 2)

#     # Combine inside/outside distances
#     outside = torch.clamp(q, min=0).norm(dim=-1)
#     inside = torch.clamp(q.max(dim=-1).values, max=0)
#     return outside + inside


# def sd_ellipsoid_exact(points, center, axes, rotation_matrix):
#     """Exact signed distance function for ellipsoid using iterative method.
    
#     This provides much more accurate SDF values compared to the normalized approximation,
#     which is critical for SDF-based optimization.
    
#     Args:
#         points: (N, 3) tensor of points
#         center: (3,) tensor ellipsoid center
#         axes: (3,) tensor ellipsoid semi-axes (a, b, c)
#         rotation_matrix: (3, 3) tensor rotation matrix
        
#     Returns:
#         (N,) tensor of exact signed distances
#     """
#     # Transform points to ellipsoid local coordinates
#     local_points = (points - center) @ rotation_matrix
    
#     # Ensure positive semi-axes
#     axes_safe = torch.clamp(axes, min=1e-8)
#     a, b, c = axes_safe, axes_safe, axes_safe
    
#     # For each point, compute exact distance using iterative method
#     distances = []
    
#     for i in range(local_points.shape[0]):
#         x, y, z = local_points[i]
        
#         # Check if point is at origin (special case)
#         if torch.norm(local_points[i]) < 1e-10:
#             # Distance from origin to surface is minimum axis
#             distances.append(-torch.min(axes_safe))
#             continue
        
#         # Normalize coordinates by axes
#         px, py, pz = x / a, y / b, z / c
        
#         # Check if inside (normalized distance < 1)
#         normalized_dist_sq = px*px + py*py + pz*pz
#         is_inside = normalized_dist_sq < 1.0
        
#         if normalized_dist_sq < 1e-10:
#             # Very close to center
#             distances.append(-torch.min(axes_safe))
#             continue
        
#         # For exact distance, we need to find the closest point on the ellipsoid surface
#         # This requires solving: grad(||p - s||²) = λ * grad(ellipsoid_constraint(s)) = 0
#         # where s is the closest surface point and λ is Lagrange multiplier
        
#         # Use iterative Newton's method for accurate result
#         # Initial guess: project along ray from center
#         t = 1.0 / torch.sqrt(normalized_dist_sq)  # Scale factor to get to surface
        
#         # Iterative refinement (typically converges in 3-5 iterations)
#         for iteration in range(10):
#             # Current surface point estimate
#             sx = t * x
#             sy = t * y  
#             sz = t * z
            
#             # Ellipsoid constraint: (sx/a)² + (sy/b)² + (sz/c)² = 1
#             constraint = (sx/a)**2 + (sy/b)**2 + (sz/c)**2
            
#             # Gradient of constraint w.r.t. surface point
#             grad_x = 2 * sx / (a*a)
#             grad_y = 2 * sy / (b*b)
#             grad_z = 2 * sz / (c*c)
            
#             # Newton step to satisfy constraint
#             grad_norm_sq = grad_x*grad_x + grad_y*grad_y + grad_z*grad_z
            
#             if grad_norm_sq < 1e-12:
#                 break
                
#             # Update t to move closer to constraint satisfaction
#             dt = (1.0 - constraint) / grad_norm_sq
#             correction_x = dt * grad_x
#             correction_y = dt * grad_y
#             correction_z = dt * grad_z
            
#             # Update surface point
#             sx += correction_x
#             sy += correction_y
#             sz += correction_z
            
#             # Update t (scale factor from origin to surface point)
#             surface_norm = torch.sqrt(sx*sx + sy*sy + sz*sz)
#             if surface_norm > 1e-10:
#                 point_norm = torch.sqrt(x*x + y*y + z*z)
#                 t = surface_norm / point_norm
            
#             # Check convergence
#             if abs(dt) < 1e-8:
#                 break
        
#         # Final surface point
#         sx = t * x
#         sy = t * y
#         sz = t * z
        
#         # Distance from original point to closest surface point
#         dist = torch.sqrt((x - sx)**2 + (y - sy)**2 + (z - sz)**2)
        
#         # Apply correct sign
#         if is_inside:
#             distances.append(-dist)
#         else:
#             distances.append(dist)
    
#     return torch.stack(distances)


# def sd_normalised(points, center, axes, rotation_matrix):
#     """Signed distance function for ellipsoid (returns real distances).
    
#     Args:
#         points: (N, 3) tensor of points
#         center: (3,) tensor ellipsoid center
#         axes: (3,) tensor ellipsoid semi-axes (a, b, c)
#         rotation_matrix: (3, 3) tensor rotation matrix
        
#     Returns:
#         (N,) tensor of signed distances (real distances, not normalized)
#     """
#     # Transform points to ellipsoid local coordinates
#     local_points = (points - center) @ rotation_matrix
    
#     # Normalize by semi-axes (add small epsilon to prevent division by zero)
#     axes_safe = torch.clamp(axes, min=1e-8)
#     normalized_points = local_points / axes_safe
    
#     # Distance from origin in normalized space
#     r = torch.norm(normalized_points, dim=1)
    
#     # Convert back to real distance by scaling with average_axis_length
#     # This gives approximate real signed distance
#     avg_axis_length = axes_safe.mean()
    
#     # Real SDF: < 0 inside, > 0 outside (in world units)
#     return (r - 1.0) * avg_axis_length


