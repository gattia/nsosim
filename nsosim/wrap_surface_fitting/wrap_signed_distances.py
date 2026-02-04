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

