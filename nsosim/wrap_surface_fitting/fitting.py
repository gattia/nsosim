"""
Wrap surface fitting using PyTorch optimization with support for both PCA and geometric initialization.

Example usage with geometric initialization:
    # For cylinder fitting with geometric initialization
    cylinder_fitter = CylinderFitter(
        lr=1e-2, 
        epochs=1000, 
        initialization='geometric',  # Use geometric instead of PCA
        alpha=1.0, 
        beta=0.1
    )
    
    center, sizes, rotation = cylinder_fitter.fit(
        points=bone_points,
        labels=inside_outside_labels,
        sdf=sdf_values,
        mesh=labeled_mesh,           # Required for geometric init
        surface_name="femur_1"       # Required for geometric init
    )
    
    # For cylinder fitting with random axis perturbation (useful for robustness testing)
    cylinder_fitter = CylinderFitter(
        initialization='geometric',
        random_axis_degrees=10.0     # Add up to ±10° random rotation to initial axis
    )
    
    # For ellipsoid fitting with geometric initialization  
    ellipsoid_fitter = EllipsoidFitter(
        initialization='geometric'
    )
    
    center, axes, rotation = ellipsoid_fitter.fit(
        points=bone_points,
        labels=inside_outside_labels,
        mesh=labeled_mesh,
        surface_name="patella_1"
    )

Example usage with learning rate scheduling:
    # Cylinder fitting with cosine annealing schedule
    cylinder_fitter = CylinderFitter(
        lr=1e-2, 
        epochs=1500,
        lr_schedule='cosine',
        lr_schedule_params={'T_max': 1500, 'eta_min': 1e-6}
    )
    
    # Cylinder fitting with exponential decay
    cylinder_fitter = CylinderFitter(
        lr=5e-3,
        epochs=1000, 
        lr_schedule='exponential',
        lr_schedule_params={'gamma': 0.995}  # 0.5% decay per epoch
    )
    
    # Cylinder fitting with adaptive plateau-based reduction
    cylinder_fitter = CylinderFitter(
        lr=1e-2,
        epochs=2000,
        lr_schedule='plateau',
        lr_schedule_params={
            'factor': 0.5,      # Reduce by half
            'patience': 25,     # Wait 25 epochs
            'threshold': 1e-4   # Minimum improvement
        }
    )

Available learning rate schedules:
    - 'cosine': Cosine annealing with smooth decay
    - 'exponential': Exponential decay by constant factor
    - 'step': Step decay at regular intervals  
    - 'multistep': Step decay at specific epochs
    - 'plateau': Adaptive reduction when loss plateaus
    - None: No scheduling (constant learning rate)

Geometric initialization uses the pre-labeled near-surface points to fit 
shape parameters directly from surface geometry, which should be more accurate
than PCA for cases with sparse inside points.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import warnings
import logging
from . import surface_param_estimation
from .main import wrap_surface
from typing import Union

logger = logging.getLogger(__name__)  # This will be 'nsosim.wrap_surface_fitting.fitting'

class RotationUtils:
    """Quaternion and rotation utilities with improved numerical stability."""
    
    @staticmethod
    def quat_from_rot(R: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to quaternion (w, x, y, z) with improved numerical stability."""
        assert R.shape == (3, 3), f"Expected (3, 3) rotation matrix, got {R.shape}"
        
        # Shepperd's method for numerical stability
        t = torch.trace(R)
        
        if t > 0:
            s = torch.sqrt(t + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s  
            z = (R[1, 0] - R[0, 1]) / s
        else:
            diag = torch.diagonal(R)
            i = torch.argmax(diag)
            
            if i == 0:
                s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif i == 1:
                s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        
        quat = torch.stack([w, x, y, z])
        return quat / quat.norm(p=2)  # Ensure unit quaternion
        
    @staticmethod
    def rot_from_quat(q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion (w,x,y,z) to 3x3 rotation matrix.
        
        Note: Assumes quaternion is already normalized. Normalization should be 
        done outside the forward pass to avoid gradient contamination.
        """
        assert q.shape == (4,), f"Expected shape (4,) for quaternion (w,x,y,z), got {q.shape}"
        
        # Use quaternion directly without normalization to preserve gradient flow
        # Normalization should be enforced after optimizer steps, not during forward pass
        w, x, y, z = q
        
        return torch.stack([
            torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)], dim=0),
            torch.stack([2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)], dim=0),
            torch.stack([2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)], dim=0)
        ], dim=0)

    @staticmethod
    def axis_angle_from_rot(R: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to axis-angle representation (3D vector).
        
        Args:
            R: (3, 3) rotation matrix
            
        Returns:
            torch.Tensor: (3,) axis-angle vector where direction is rotation axis 
                         and magnitude is rotation angle in radians
        """
        assert R.shape == (3, 3), f"Expected (3, 3) rotation matrix, got {R.shape}"
        
        # Compute rotation angle from trace
        trace = torch.trace(R)
        angle = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))
        
        # Handle small angle case (near identity)
        if angle.abs() < 1e-6:
            return torch.zeros(3, device=R.device, dtype=R.dtype)
        
        # Handle 180 degree rotation case
        if angle.abs() > np.pi - 1e-6:
            # Find the eigenvector corresponding to eigenvalue 1
            # This is the rotation axis for 180-degree rotation
            eig_vals, eig_vecs = torch.linalg.eigh(R)
            # Find index of eigenvalue closest to 1
            idx = torch.argmin(torch.abs(eig_vals - 1))
            axis = eig_vecs[:, idx]
            
            # Ensure consistent sign
            if axis[0] < 0 or (axis[0] == 0 and axis[1] < 0) or (axis[0] == 0 and axis[1] == 0 and axis[2] < 0):
                axis = -axis
                
            return axis * angle
        
        # General case: extract axis from skew-symmetric part
        axis = torch.stack([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0], 
            R[1, 0] - R[0, 1]
        ]) / (2 * torch.sin(angle))
        
        return axis * angle
    
    @staticmethod
    def rot_from_axis_angle(axis_angle: torch.Tensor) -> torch.Tensor:
        """Convert axis-angle vector to rotation matrix using Rodrigues' formula.
        
        Args:
            axis_angle: (3,) axis-angle vector
            
        Returns:
            torch.Tensor: (3, 3) rotation matrix
        """
        assert axis_angle.shape == (3,), f"Expected (3,) axis-angle vector, got {axis_angle.shape}"
        
        angle = torch.norm(axis_angle)
        
        # Handle zero rotation case
        if angle < 1e-8:
            return torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
        
        # Normalize axis
        axis = axis_angle / angle
        
        # Rodrigues' rotation formula
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]], 
            [-axis[1], axis[0], 0]
        ], device=axis_angle.device, dtype=axis_angle.dtype)
        
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        R = (torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype) + 
             sin_angle * K + 
             (1 - cos_angle) * torch.mm(K, K))
        
        return R

    @staticmethod
    def rot_to_euler_xyz_body(R: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Extract intrinsic XYZ (body) Euler angles from rotation matrix.
        
        Args:
            R (torch.Tensor or np.ndarray): Rotation matrix of shape (3, 3)
            
        Returns:
            torch.Tensor or np.ndarray: Euler angles [x, y, z] in radians, matching input type
        """
        input_type = 'torch' if isinstance(R, torch.Tensor) else 'numpy'
        
        if input_type == 'numpy':
            R = torch.tensor(R, dtype=torch.float64)

        assert R.shape == (3, 3), f"Expected shape (3, 3), got {R.shape}"

        eps = 1e-6

        if torch.abs(R[0, 2] - 1.0) < eps:
            x = torch.tensor(0.0, dtype=R.dtype)
            y = -torch.pi / 2
            z = torch.atan2(-R[1, 0], -R[2, 0])
        elif torch.abs(R[0, 2] + 1.0) < eps:
            x = torch.tensor(0.0, dtype=R.dtype)
            y = torch.pi / 2
            z = torch.atan2(R[1, 0], R[2, 0])
        else:
            y = torch.asin(-R[0, 2])
            x = torch.atan2(R[1, 2], R[2, 2])
            z = torch.atan2(R[0, 1], R[0, 0])

        euler = torch.stack([x, y, z])

        if input_type == 'numpy':
            return euler.numpy()
        return euler


def sd_cylinder(points, center, radius, half_length, rotation_matrix):
    """Signed distance function for finite cylinder.
    
    Args:
        points: (N, 3) tensor of points
        center: (3,) tensor cylinder center
        radius: scalar tensor cylinder radius  
        half_length: scalar tensor cylinder half-length
        rotation_matrix: (3, 3) tensor rotation matrix (local→world, so we use .T)
        
    Returns:
        (N,) tensor of signed distances
    """
    # Transform points to cylinder local coordinates
    p = (points - center) @ rotation_matrix.T                 # world → local
    
    # Distance from axis and from caps
    radial_dist = torch.linalg.norm(p[..., :2], dim=-1) - radius
    axial_dist = torch.abs(p[..., 2]) - half_length
    
    q = torch.stack([radial_dist, axial_dist], dim=-1)       # (..., 2)

    # Combine inside/outside distances
    outside = torch.clamp(q, min=0).norm(dim=-1)
    inside = torch.clamp(q.max(dim=-1).values, max=0)
    return outside + inside


def sd_ellipsoid_exact(points, center, axes, rotation_matrix):
    """Exact signed distance function for ellipsoid using iterative method.
    
    This provides much more accurate SDF values compared to the normalized approximation,
    which is critical for SDF-based optimization.
    
    Args:
        points: (N, 3) tensor of points
        center: (3,) tensor ellipsoid center
        axes: (3,) tensor ellipsoid semi-axes (a, b, c)
        rotation_matrix: (3, 3) tensor rotation matrix
        
    Returns:
        (N,) tensor of exact signed distances
    """
    # Transform points to ellipsoid local coordinates
    local_points = (points - center) @ rotation_matrix
    
    # Ensure positive semi-axes
    axes_safe = torch.clamp(axes, min=1e-8)
    a, b, c = axes_safe, axes_safe, axes_safe
    
    # For each point, compute exact distance using iterative method
    distances = []
    
    for i in range(local_points.shape[0]):
        x, y, z = local_points[i]
        
        # Check if point is at origin (special case)
        if torch.norm(local_points[i]) < 1e-10:
            # Distance from origin to surface is minimum axis
            distances.append(-torch.min(axes_safe))
            continue
        
        # Normalize coordinates by axes
        px, py, pz = x / a, y / b, z / c
        
        # Check if inside (normalized distance < 1)
        normalized_dist_sq = px*px + py*py + pz*pz
        is_inside = normalized_dist_sq < 1.0
        
        if normalized_dist_sq < 1e-10:
            # Very close to center
            distances.append(-torch.min(axes_safe))
            continue
        
        # For exact distance, we need to find the closest point on the ellipsoid surface
        # This requires solving: grad(||p - s||²) = λ * grad(ellipsoid_constraint(s)) = 0
        # where s is the closest surface point and λ is Lagrange multiplier
        
        # Use iterative Newton's method for accurate result
        # Initial guess: project along ray from center
        t = 1.0 / torch.sqrt(normalized_dist_sq)  # Scale factor to get to surface
        
        # Iterative refinement (typically converges in 3-5 iterations)
        for iteration in range(10):
            # Current surface point estimate
            sx = t * x
            sy = t * y  
            sz = t * z
            
            # Ellipsoid constraint: (sx/a)² + (sy/b)² + (sz/c)² = 1
            constraint = (sx/a)**2 + (sy/b)**2 + (sz/c)**2
            
            # Gradient of constraint w.r.t. surface point
            grad_x = 2 * sx / (a*a)
            grad_y = 2 * sy / (b*b)
            grad_z = 2 * sz / (c*c)
            
            # Newton step to satisfy constraint
            grad_norm_sq = grad_x*grad_x + grad_y*grad_y + grad_z*grad_z
            
            if grad_norm_sq < 1e-12:
                break
                
            # Update t to move closer to constraint satisfaction
            dt = (1.0 - constraint) / grad_norm_sq
            correction_x = dt * grad_x
            correction_y = dt * grad_y
            correction_z = dt * grad_z
            
            # Update surface point
            sx += correction_x
            sy += correction_y
            sz += correction_z
            
            # Update t (scale factor from origin to surface point)
            surface_norm = torch.sqrt(sx*sx + sy*sy + sz*sz)
            if surface_norm > 1e-10:
                point_norm = torch.sqrt(x*x + y*y + z*z)
                t = surface_norm / point_norm
            
            # Check convergence
            if abs(dt) < 1e-8:
                break
        
        # Final surface point
        sx = t * x
        sy = t * y
        sz = t * z
        
        # Distance from original point to closest surface point
        dist = torch.sqrt((x - sx)**2 + (y - sy)**2 + (z - sz)**2)
        
        # Apply correct sign
        if is_inside:
            distances.append(-dist)
        else:
            distances.append(dist)
    
    return torch.stack(distances)


def sd_normalised(points, center, axes, rotation_matrix):
    """Signed distance function for ellipsoid (returns real distances).
    
    Args:
        points: (N, 3) tensor of points
        center: (3,) tensor ellipsoid center
        axes: (3,) tensor ellipsoid semi-axes (a, b, c)
        rotation_matrix: (3, 3) tensor rotation matrix
        
    Returns:
        (N,) tensor of signed distances (real distances, not normalized)
    """
    # Transform points to ellipsoid local coordinates
    local_points = (points - center) @ rotation_matrix
    
    # Normalize by semi-axes (add small epsilon to prevent division by zero)
    axes_safe = torch.clamp(axes, min=1e-8)
    normalized_points = local_points / axes_safe
    
    # Distance from origin in normalized space
    r = torch.norm(normalized_points, dim=1)
    
    # Convert back to real distance by scaling with average_axis_length
    # This gives approximate real signed distance
    avg_axis_length = axes_safe.mean()
    
    # Real SDF: < 0 inside, > 0 outside (in world units)
    return (r - 1.0) * avg_axis_length


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


class BaseShapeFitter(ABC):
    """Base class for shape fitting with common optimization logic."""
    
    def __init__(self, lr=5e-3, epochs=1500, device='cpu', use_lbfgs=False, lbfgs_epochs=100, 
                 alpha=1.0, beta=0.1, gamma=0.1, lbfgs_lr=1, 
                 margin_decay_type='exponential', initialization='geometric',
                 lr_schedule=None, lr_schedule_params=None):
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.use_lbfgs = use_lbfgs
        self.lbfgs_epochs = lbfgs_epochs
        self.lbfgs_lr = lbfgs_lr
        self.alpha = alpha  # Weight for margin loss
        self.beta = beta    # Weight for distance loss
        self.gamma = gamma  # Weight for surface points loss
        self.margin_decay_type = margin_decay_type  # Type of margin decay
        self.initialization = initialization  # 'pca' or 'geometric'
        
        # Learning rate scheduling
        self.lr_schedule = lr_schedule  # 'cosine', 'exponential', 'step', 'plateau', or None
        self.lr_schedule_params = lr_schedule_params or {}  # Parameters for the scheduler
        
        
        
    def _prepare_data(self, points, labels):
        """Convert to tensors and validate."""
        if len(points) != len(labels):
            raise ValueError(f"Points ({len(points)}) and labels ({len(labels)}) must have same length")
        
        x = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(labels > 0, dtype=torch.bool, device=self.device)
        
        if x.shape[1] != 3:
            raise ValueError(f"Points must be 3D, got shape {x.shape}")
            
        return x, y
        
    def _compute_loss(self, d, y, margin):
        """Squared-hinge margin loss with proper inside/outside penalties.
        
        For points labeled as "inside": penalize if d > -margin (not sufficiently inside)
        For points labeled as "outside": penalize if d < margin (not sufficiently outside)
        This ensures all points contribute to the loss until they satisfy the margin constraint.
        """
        pass
        # mask_in, mask_out = y, ~y
        
        # # Handle empty masks gracefully
        
        # # d is signed distance. (-) is inside, (+) is outside. 
        # # we compute losses separately for the inside and outside points.
        # # for the inside points, they are negative. So, if the d (-) + the margin is > 0
        # # then there should be a loss. Here, relu linearly increases. 
        # loss_in = (F.relu(d + margin)[mask_in] ** 2).mean() if mask_in.any() else torch.tensor(0.0, device=self.device)
        
        # # Outside points: penalize if they're not sufficiently outside (d < margin)  
        # # for the outside points, they are positive. So, if the margin - d (+) is > 0
        # # (i.e., d < margin), then there should be a loss.
        # loss_out = (F.relu(margin - d)[mask_out]).mean() if mask_out.any() else torch.tensor(0.0, device=self.device)
        
        # return loss_in + loss_out
        
        # # return (d**2).mean()
    
    def _compute_distance_loss(self, sdf_fitted, sdf_ground_truth, sigma=1.0, use_weighting=False):
        """Correlation-based distance loss with optional surface proximity weighting.
        
        Args:
            sdf_fitted: Fitted SDF values from the current shape parameters
            sdf_ground_truth: Ground truth SDF values to the original surface  
            sigma: Controls surface proximity weighting (smaller = more surface focus)
            use_weighting: Whether to use surface proximity weighting
            
        Returns:
            torch.Tensor: Correlation-based loss (1 - correlation)
        """
        if use_weighting:
            # Surface proximity weights: strongest at sdf=0, decays with distance
            abs_dist = torch.abs(sdf_ground_truth)
            weights = torch.exp(-(abs_dist / sigma)**2)
            
            # Compute weighted means
            weighted_sum = weights.sum()
            mean_fitted = (weights * sdf_fitted).sum() / (weighted_sum + 1e-8)
            mean_gt = (weights * sdf_ground_truth).sum() / (weighted_sum + 1e-8)
            
            # Center the values
            centered_fitted = sdf_fitted - mean_fitted  
            centered_gt = sdf_ground_truth - mean_gt
            
            # Weighted correlation coefficient
            weighted_cov = (weights * centered_fitted * centered_gt).sum()
            weighted_var_fitted = (weights * centered_fitted**2).sum()
            weighted_var_gt = (weights * centered_gt**2).sum()
            
            # Pearson correlation coefficient (weighted)
            correlation = weighted_cov / (torch.sqrt(weighted_var_fitted * weighted_var_gt) + 1e-8)
        else:
            # Unweighted correlation
            mean_fitted = sdf_fitted.mean()
            mean_gt = sdf_ground_truth.mean()
            
            # Center the values
            centered_fitted = sdf_fitted - mean_fitted
            centered_gt = sdf_ground_truth - mean_gt
            
            # Standard correlation coefficient
            covariance = (centered_fitted * centered_gt).mean()
            var_fitted = (centered_fitted**2).mean()
            var_gt = (centered_gt**2).mean()
            
            # Pearson correlation coefficient
            correlation = covariance / (torch.sqrt(var_fitted * var_gt) + 1e-8)
        
        # Only reward positive correlations, penalize negative ones maximally
        positive_correlation = torch.clamp(correlation, min=0.0)
        
        # Convert to loss: perfect positive correlation = 0, negative correlation = 1
        return 1.0 - positive_correlation
    
    @abstractmethod
    def _initialize_parameters(self, points_inside, mesh=None, surface_name=None):
        """Initialize shape-specific parameters.
        
        Args:
            points_inside: Points classified as inside the shape
            mesh: Mesh object (required for geometric initialization)
            surface_name: Surface name (required for geometric initialization)
        
        Returns:
            tuple: Initial parameters for the shape
        """
        pass
    
    @abstractmethod
    def _initialize_parameters_pca(self, points_inside):
        """Initialize parameters using PCA method.
        
        Returns:
            tuple: Initial parameters for the shape
        """
        pass
    
    @abstractmethod  
    def _initialize_parameters_geometric(self, mesh, surface_name):
        """Initialize parameters using geometric surface analysis method.
        
        Args:
            mesh: Mesh object with near-surface point labels
            surface_name: Name of the surface to analyze
        
        Returns:
            tuple: Initial parameters for the shape
        """
        pass
    
    @abstractmethod
    def _create_parameters(self, initial_params):
        """Create torch.nn.Parameters from initial values.
        
        Returns:
            list: List of parameters for optimizer
        """
        pass
    
    @abstractmethod
    def _compute_sdf(self, points, parameters):
        """Compute signed distance function for the shape.
        
        Args:
            points: Input points
            parameters: Current parameter values
            
        Returns:
            torch.Tensor: Signed distances
        """
        pass
    
    @abstractmethod
    def _compute_shape_loss(self, sdf, labels, sdf_ground_truth, near_surface_points_tensor, margin, epoch):
        """Compute shape-specific loss (may include normalization, decay, etc.).
        
        Should combine margin loss, distance loss, and surface points loss 
        using self.alpha, self.beta, and self.gamma.
        
        Args:
            sdf: Signed distance values for main points
            labels: Inside/outside labels for main points
            sdf_ground_truth: Ground truth SDF values to original surface for main points
            near_surface_points_tensor: Tensor of near-surface points (optional)
            margin: Loss margin
            epoch: Current epoch (for decay, etc.)
            
        Returns:
            torch.Tensor: Combined loss value
        """
        pass
    
    @abstractmethod
    def _extract_results(self, parameters):
        """Extract final fitted parameters.
        
        Returns:
            tuple: Final shape parameters
        """
        pass
    
    def _setup_plotting(self, plot):
        """Setup plotting if requested. Override for custom plotting."""
        if plot:
            return {'losses': [], 'type': 'simple'}
        return None
    
    def _update_plotting(self, plot_data, loss_val, epoch, _):
        """Update plotting during training. Override for custom plotting."""
        if plot_data is not None:
            plot_data['losses'].append(loss_val)
    
    def _finalize_plotting(self, plot_data, plot):
        """Finalize plotting after training. Override for custom plotting."""
        if plot and plot_data and plot_data['losses']:
            plt.figure(figsize=(8, 4))
            plt.plot(plot_data['losses'])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{self.__class__.__name__} Loss')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.show()
    
    def _compute_margin_decay(self, margin, epoch, decay_type=None, min_margin=1e-4):
        """Compute decayed margin value for coarse-to-fine optimization.
        
        Args:
            margin: Initial margin value
            epoch: Current epoch
            decay_type: Type of decay ('exponential', 'linear', 'cosine', None for instance default)
            
        Returns:
            float: Decayed margin value
        """
        if decay_type is None:
            decay_type = self.margin_decay_type
        
        # Handle case where epochs=0 (L-BFGS only mode)
        if self.epochs == 0:
            # No epoch-based decay, return original margin
            return max(margin, min_margin)
            
        progress = epoch / self.epochs  # 0 to 1
        
        if decay_type == 'exponential':
            # Exponential decay to ~1% of original margin
            decay_factor = np.exp(-2 * progress)  # e^(-4) ≈ 0.018
        elif decay_type == 'linear':
            # Linear decay to zero
            decay_factor = 1 - progress
        elif decay_type == 'cosine':
            # Cosine annealing (smooth S-curve)
            decay_factor = 0.5 * (1 + np.cos(np.pi * progress))
        elif decay_type is None:
            decay_factor = 1
        else:
            raise ValueError(f"Unknown decay_type: {decay_type}")
            
        # Ensure minimum margin to prevent numerical issues
        # min_margin = margin * 1e-12  # Very small but non-zero minimum
        decayed_margin = max(margin * decay_factor, min_margin)
        
        return decayed_margin
    
    def _get_lr_scales(self):
        """Get learning rate scales for each parameter in the same order as _create_parameters().
        
        Subclasses should override this to provide parameter-specific learning rates.
        
        Returns:
            list: Learning rate scale factors for each parameter
        """
        # Default: all parameters use the same learning rate
        return [1.0] * 4  # Assume max 4 parameters by default
    
    def _create_lr_scheduler(self, optimizer):
        """Create learning rate scheduler based on self.lr_schedule.
        
        Args:
            optimizer: PyTorch optimizer
            
        Returns:
            torch.optim.lr_scheduler or None
        """
        if self.lr_schedule is None:
            return None
            
        params = self.lr_schedule_params.copy()
        
        if self.lr_schedule == 'cosine':
            # Cosine annealing - smooth decay to min_lr over T_max epochs
            T_max = params.get('T_max', self.epochs)
            eta_min = params.get('eta_min', self.lr * 0.01)  # 1% of initial LR
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min
            )
            
        elif self.lr_schedule == 'exponential':
            # Exponential decay - multiply LR by gamma each epoch
            gamma = params.get('gamma', 0.99)  # Default: 1% decay per epoch
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            
        elif self.lr_schedule == 'step':
            # Step decay - reduce LR by factor at specific milestones
            step_size = params.get('step_size', self.epochs // 3)
            gamma = params.get('gamma', 0.5)  # Halve LR at each step
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            
        elif self.lr_schedule == 'multistep':
            # Multi-step decay - reduce LR at multiple specific epochs
            milestones = params.get('milestones', [self.epochs // 3, 2 * self.epochs // 3])
            gamma = params.get('gamma', 0.5)
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
            
        elif self.lr_schedule == 'plateau':
            # Reduce on plateau - adaptive based on loss progress
            factor = params.get('factor', 0.5)  # Reduce LR by this factor
            patience = params.get('patience', 20)  # Wait this many epochs before reducing
            threshold = params.get('threshold', 1e-4)  # Minimum change to qualify as improvement
            min_lr = params.get('min_lr', self.lr * 1e-6)  # Minimum LR
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=factor, patience=patience, 
                threshold=threshold, min_lr=min_lr, verbose=True
            )
            
        else:
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")
    
    def _get_current_lr(self, optimizer):
        """Get current learning rate from optimizer."""
        return optimizer.param_groups[0]['lr']
    
    def fit(self, points, labels, sdf=None, margin=0.05, plot=False, mesh=None, surface_name=None, near_surface_points=None):
        """Template method for fitting shapes to classified points.
        
        Args:
            points: Input points
            labels: Inside/outside labels  
            sdf: Ground truth SDF values to original surface (optional)
            margin: Loss margin
            plot: Whether to plot training progress
            mesh: Mesh object (required for geometric initialization)
            surface_name: Surface name (required for geometric initialization)
            near_surface_points: Optional (N, 3) array/tensor of points known to be near the surface
        """
        # 1. Prepare data
        x, y = self._prepare_data(points, labels)
        
        # Prepare ground truth SDF if provided
        if sdf is not None:
            if len(sdf) != len(points):
                raise ValueError(f"SDF ({len(sdf)}) and points ({len(points)}) must have same length")
            sdf_tensor = torch.as_tensor(sdf, dtype=torch.float32, device=self.device)
        else:
            sdf_tensor = None
        
        # Prepare near_surface_points if provided
        near_surface_points_tensor = None
        if near_surface_points is not None:
            if len(near_surface_points) > 0:
                near_surface_points_tensor = torch.as_tensor(near_surface_points, dtype=torch.float32, device=self.device)
                if near_surface_points_tensor.shape[1] != 3:
                    raise ValueError(f"near_surface_points must be 3D, got shape {near_surface_points_tensor.shape}")
            else:
                warnings.warn("near_surface_points was provided but is empty.")
        
        # 2. Initialize parameters
        inside = x[y]
        if len(inside) == 0:
            raise ValueError("No inside points found for shape fitting")
        
        if len(inside) < 3:
            warnings.warn(f"Very few inside points ({len(inside)}) for robust fitting")
            
        # Check initialization requirements
        if self.initialization == 'geometric':
            if mesh is None or surface_name is None:
                warnings.warn("Geometric initialization requires mesh and surface_name. Falling back to PCA.")
                self.initialization = 'pca'
            
        try:
            initial_params = self._initialize_parameters(inside, mesh, surface_name)
        except Exception as e:
            raise ValueError(f"Failed to initialize parameters: {e}")
        
        # 3. Create parameters and optimizer with parameter-specific learning rates
        parameters = self._create_parameters(initial_params)
        
        # 4. Setup plotting
        plot_data = self._setup_plotting(plot)
        
        # Check if we should skip Adam optimization and go directly to L-BFGS
        skip_adam = (self.epochs == 0)
        
        if skip_adam:
            # Validate that L-BFGS is requested when skipping Adam
            if not (self.use_lbfgs and self.lbfgs_epochs > 0):
                raise ValueError("When epochs=0, must have use_lbfgs=True and lbfgs_epochs > 0")
            logger.info("Skipping Adam optimization (epochs=0), proceeding directly to L-BFGS")
        
        # 5. Adam Training loop (skip if epochs=0)
        if not skip_adam:
            # Use parameter-specific learning rates to handle gradient scale differences
            param_groups = []
            lr_scales = self._get_lr_scales()
            
            for i, param in enumerate(parameters):
                lr_scale = lr_scales[i] if i < len(lr_scales) else 1.0
                    
                param_groups.append({
                    'params': [param],
                    'lr': self.lr * lr_scale,
                    'name': f'param_{i}'
                })
            
            opt = torch.optim.Adam(param_groups)
            
            # Create learning rate scheduler
            scheduler = self._create_lr_scheduler(opt)
            
            # Store optimizer reference for logging
            self._current_optimizer = opt
            
            best_loss = float('inf')
            patience_counter = 0
            patience = 100  # Early stopping
            
            # Track rotation progress to detect stalling
            rotation_angles = []
            rotation_stall_threshold = 5e-4  # radians per 100 epochs
            rotation_boost_applied = False
            
            for epoch in range(self.epochs):
                opt.zero_grad()
                
                # Store current parameters for loss computation
                self._current_parameters = parameters
                
                try:
                    # Compute SDF and loss
                    d = self._compute_sdf(x, parameters)
                    loss = self._compute_shape_loss(d, y, sdf_tensor, near_surface_points_tensor, margin, epoch)
                    
                    # Check for NaN/inf
                    if not torch.isfinite(loss):
                        warnings.warn(f"Non-finite loss at epoch {epoch}, stopping optimization")
                        break
                    
                    loss.backward()
                    
                    # Track rotation progress
                    if len(parameters) >= 4:
                        axis_param = parameters[3]
                        axis_normalized = axis_param.detach() / (torch.norm(axis_param.detach()) + 1e-8)
                        angle_from_identity = torch.acos(torch.clamp((torch.dot(axis_normalized, torch.tensor([1.0, 0.0, 0.0], device=axis_param.device)) - 1) / 2, -1 + 1e-7, 1 - 1e-7))
                        rotation_angles.append(angle_from_identity.item())
                        
                        # Check for rotation stalling every 100 epochs
                        if epoch > 200 and epoch % 100 == 0 and not rotation_boost_applied:
                            recent_angles = rotation_angles[-100:]
                            angle_progress = max(recent_angles) - min(recent_angles)
                            
                            if angle_progress < rotation_stall_threshold:
                                print(f"  ROTATION STALL DETECTED at epoch {epoch}!")
                                print(f"  Angle progress in last 100 epochs: {angle_progress:.6f} rad ({angle_progress*180/3.14159:.3f} deg)")
                                print(f"  Boosting rotation learning rate by 10x")
                                
                                # # Boost rotation learning rate
                                # for param_group in opt.param_groups:
                                #     if param_group['name'] == 'axis_vector':
                                #         param_group['lr'] *= 10.0
                                #         rotation_boost_applied = True
                                #         break
                                # # Boost rotation learning rate (parameter 3 for cylinder axis_vector)
                                # if len(opt.param_groups) > 3:
                                #     opt.param_groups[3]['lr'] *= 10.0
                                #     rotation_boost_applied = True
                    
                    # Debug: Check gradients for all parameters  
                    if epoch % 100 == 0 or epoch < 5:
                        print(f"DEBUG GRAD Epoch {epoch}: Parameter gradients:")
                        
                        total_grad_norm = 0.0
                        grad_info = []
                        
                        for i, param in enumerate(parameters):
                            param_name = f'param_{i}'
                            
                            if param.grad is not None:
                                grad_norm = param.grad.norm().item()
                                grad_values = param.grad.detach().cpu().numpy()
                                effective_lr = opt.param_groups[i]['lr']
                                step_size = grad_norm * effective_lr
                                
                                print(f"  {param_name}: grad_norm={grad_norm:.8f}, effective_lr={effective_lr:.2e}, step_size={step_size:.8f}")
                                if grad_values.size <= 4:  # Only print values for small arrays
                                    print(f"    grad_values={grad_values}")
                                
                                total_grad_norm += grad_norm
                                grad_info.append((param_name, grad_norm))
                            else:
                                print(f"  {param_name}: grad is None!")
                                print(f"    requires_grad={param.requires_grad}")
                        
                        # Print relative gradient contributions
                        if total_grad_norm > 0:
                            print(f"  Relative gradient contributions:")
                            for param_name, grad_norm in grad_info:
                                fraction = grad_norm / total_grad_norm
                                print(f"    {param_name}: {fraction:.1%}")
                        
                        # Additional diagnostics for axis vector parameter
                        if len(parameters) >= 4:  # Make sure we have axis vector parameter
                            axis_param = parameters[3]  # axis vector is 4th parameter
                            if axis_param.grad is not None:
                                # Check if axis vector has changed from initial
                                if epoch > 0:
                                    axis_normalized = axis_param.detach() / (torch.norm(axis_param.detach()) + 1e-8)
                                    print(f"  axis vector (normalized): [{axis_normalized[0]:.6f}, {axis_normalized[1]:.6f}, {axis_normalized[2]:.6f}]")
                                    print(f"  axis vector norm: {torch.norm(axis_param).item():.8f}")
                                    
                                    # Check if the loss actually depends on axis orientation
                                    with torch.no_grad():
                                        # Compute loss with current axis
                                        d_current = self._compute_sdf(x, parameters)
                                        loss_current = self._compute_shape_loss(d_current, y, sdf_tensor, near_surface_points_tensor, margin, epoch)
                                        
                                        # Test with a different axis orientation (small perturbation)
                                        perturb = torch.randn_like(axis_param) * 0.1
                                        test_axis = axis_param + perturb
                                        temp_params = list(parameters)
                                        temp_params[3] = test_axis
                                        d_perturbed = self._compute_sdf(x, temp_params)
                                        loss_perturbed = self._compute_shape_loss(d_perturbed, y, sdf_tensor, near_surface_points_tensor, margin, epoch)
                                        
                                        loss_diff = abs(loss_current.item() - loss_perturbed.item())
                                        print(f"  loss sensitivity to axis orientation: {loss_diff:.8f}")
                                        if loss_diff < 1e-6:
                                            print(f"  WARNING - Loss is insensitive to axis orientation!")
                        
                        # Debug height parameter issue (for cylinders)
                        if len(parameters) >= 3 and hasattr(self, '_compute_sdf'):
                            height_param = parameters[2]  # height/axes is 3rd parameter
                            if height_param.grad is not None and height_param.grad.norm().item() < 1e-10:
                                print(f"  WARNING - Height gradient is essentially zero!")
                                
                                # Test if height affects SDF
                                with torch.no_grad():
                                    current_height = height_param.detach().clone()
                                    
                                    # Test with slightly larger height
                                    temp_params = list(parameters)
                                    temp_params[2] = current_height * 1.1  # 10% larger
                                    d_larger = self._compute_sdf(x, temp_params)
                                    loss_larger = self._compute_shape_loss(d_larger, y, sdf_tensor, near_surface_points_tensor, margin, epoch)
                                    
                                    # Test with slightly smaller height  
                                    temp_params[2] = current_height * 0.9  # 10% smaller
                                    d_smaller = self._compute_sdf(x, temp_params)
                                    loss_smaller = self._compute_shape_loss(d_smaller, y, sdf_tensor, near_surface_points_tensor, margin, epoch)
                                    
                                    # Current loss
                                    d_current = self._compute_sdf(x, parameters)
                                    loss_current = self._compute_shape_loss(d_current, y, sdf_tensor, near_surface_points_tensor, margin, epoch)
                                    
                                    height_sensitivity = max(
                                        abs(loss_larger.item() - loss_current.item()),
                                        abs(loss_smaller.item() - loss_current.item())
                                    )
                                    print(f"  height sensitivity test: {height_sensitivity:.8f}")
                                    
                                    if height_sensitivity < 1e-8:
                                        print(f"  WARNING - Loss is insensitive to height changes!")
                                        print(f"  Current height value: {current_height.item():.8f}")
                                        
                                        # Check if points are within the cylinder height bounds
                                        if hasattr(self, '__class__') and 'Cylinder' in self.__class__.__name__:
                                            # For cylinder, check if points extend beyond current height
                                            center = parameters[0] if self.center_transform == 'linear' else torch.exp(torch.clamp(parameters[0], max=10)) - self.center_offset
                                            axis_vector = parameters[3]
                                            R = construct_cylinder_basis(axis_vector)
                                            local_points = (x - center) @ R
                                            z_extent = local_points[:, 2].abs().max().item()
                                            current_half_height = torch.exp(torch.clamp(current_height, max=10)).item()
                                            print(f"  Point z-extent: {z_extent:.6f}, Current half-height: {current_half_height:.6f}")
                                            
                                            if z_extent < current_half_height * 0.5:
                                                print(f"  ISSUE - Points don't reach cylinder caps! Height is too large.")
                    
                    # Gradient clipping for stability (especially for center gradients)
                    torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
                    
                    opt.step()
                    
                    # Normalize quaternion parameters after optimizer step (constraint enforcement)
                    # Do this outside autograd to avoid gradient contamination
                    with torch.no_grad():
                        for param in parameters:
                            if param.shape == (4,):  # This is a quaternion parameter
                                param.data = param.data / (param.data.norm(p=2) + 1e-8)
                    
                    # Update learning rate scheduler
                    if scheduler is not None:
                        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step(loss.item())  # Plateau scheduler needs the loss value
                        else:
                            scheduler.step()  # Other schedulers just need to be called
                    
                    # Early stopping
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                    
                    # Update plotting
                    self._update_plotting(plot_data, loss.item(), epoch, None)
                    
                except Exception as e:
                    warnings.warn(f"Error at epoch {epoch}: {e}")
                    break
        
        # 6. Optional L-BFGS refinement stage (or primary optimization if epochs=0)
        if self.use_lbfgs and self.lbfgs_epochs > 0:
            stage_name = "L-BFGS optimization" if skip_adam else "L-BFGS refinement"
            logger.info(f"Starting {stage_name} for {self.lbfgs_epochs} steps...")
            
            # Store initial loss for comparison
            with torch.no_grad():
                self._current_parameters = parameters
                initial_d = self._compute_sdf(x, parameters)
                final_epoch = 0 if skip_adam else self.epochs  # Use epoch 0 for margin calculation if we skipped Adam
                initial_loss = self._compute_shape_loss(initial_d, y, sdf_tensor, near_surface_points_tensor, margin, final_epoch)
                logger.info(f"{stage_name} starting loss: {initial_loss.item():.6f}")
            
            # Restart L-BFGS periodically to avoid getting stuck
            restart_interval = max(10, self.lbfgs_epochs // 5)  # Restart every 10 steps or 1/5 of total steps
            logger.info(f"L-BFGS will restart every {restart_interval} steps")
            
            # Multiple L-BFGS steps with periodic restarts
            for lbfgs_step in range(self.lbfgs_epochs):
                # Create/recreate L-BFGS optimizer at start and after each restart
                if lbfgs_step % restart_interval == 0:
                    logger.debug(f"(Re)starting L-BFGS optimizer at step {lbfgs_step}")
                    lbfgs_opt = torch.optim.LBFGS(parameters, lr=self.lbfgs_lr,
                                                max_iter=20, max_eval=25,
                                                tolerance_grad=1e-9, tolerance_change=1e-12,  # Much looser tolerances
                                                history_size=100, line_search_fn='strong_wolfe')
                
                # Define closure for current step
                def closure():
                    lbfgs_opt.zero_grad()
                    self._current_parameters = parameters
                    
                    try:
                        d = self._compute_sdf(x, parameters)
                        final_epoch = 0 if skip_adam else self.epochs  # Use epoch 0 for margin calculation if we skipped Adam
                        loss = self._compute_shape_loss(d, y, sdf_tensor, near_surface_points_tensor, margin, final_epoch)
                        
                        if torch.isfinite(loss):
                            loss.backward()
                            return loss
                        else:
                            return torch.tensor(float('inf'), device=self.device)
                            
                    except Exception as e:
                        logger.warning(f"L-BFGS closure error: {e}")
                        return torch.tensor(float('inf'), device=self.device)
                
                try:
                    # Perform one L-BFGS optimization step
                    loss = lbfgs_opt.step(closure)
                    current_loss = loss.item() if loss is not None else float('inf')
                    
                    # Plot after each L-BFGS step (these are actual optimization progress points)
                    epoch_offset = 0 if skip_adam else self.epochs
                    self._update_plotting(plot_data, current_loss, epoch_offset + 1 + lbfgs_step, "L-BFGS")
                    
                    if lbfgs_step % 10 == 0 or lbfgs_step < 5:
                        logger.debug(f"L-BFGS step {lbfgs_step+1}/{self.lbfgs_epochs}: loss={current_loss:.6f}")
                    
                    # Early stopping if loss becomes non-finite
                    if not torch.isfinite(torch.tensor(current_loss)):
                        logger.warning(f"L-BFGS stopping at step {lbfgs_step} due to non-finite loss")
                        break
                        
                except Exception as e:
                    logger.warning(f"L-BFGS step {lbfgs_step} failed: {e}")
                    break
            
            # Final loss after L-BFGS
            with torch.no_grad():
                self._current_parameters = parameters
                final_d = self._compute_sdf(x, parameters) 
                final_epoch = 0 if skip_adam else self.epochs  # Use epoch 0 for margin calculation if we skipped Adam
                final_loss = self._compute_shape_loss(final_d, y, sdf_tensor, near_surface_points_tensor, margin, final_epoch)
                logger.info(f"{stage_name} final loss: {final_loss.item():.6f}")
                improvement = ((initial_loss - final_loss) / initial_loss * 100).item()
                logger.info(f"{stage_name} improvement: {improvement:.2f}%")
            
            logger.info(f"{stage_name} completed successfully")
            
            # Re-finalize plotting after L-BFGS
            self._finalize_plotting(plot_data, plot)
        
        # 7. Extract and return results
        final_results = self._extract_results(parameters)
        
        self._fitted_params = final_results
        
        # Debug: Print final rotation parameters
        if len(parameters) >= 4:
            axis_param = parameters[3].detach()
            print(f"DEBUG: Final axis vector parameter: {axis_param.cpu().numpy()}")
            print(f"DEBUG: Final rotation matrix:")
            print(final_results[2].cpu().numpy())
            print(f"DEBUG: Is rotation matrix identity? {torch.allclose(final_results[2], torch.eye(3), atol=1e-4)}")
        
        return final_results
    
    @property
    def wrap_params(self):
        raise NotImplementedError("This method should be implemented by the subclass")
        

class CylinderFitter(BaseShapeFitter):
    """Cylinder fitting using PCA initialization and PyTorch optimization with axis vector parameterization."""
    
    def __init__(self, *args, center_offset=0.3, center_transform='linear', fix_height=False, 
                 random_axis_degrees=0.0, **kwargs):
        """
        Args:
            center_offset: Offset added to center coordinates before log transformation
                          to ensure positivity. Should be larger than expected coordinate range.
                          For bone coordinates in meters (0.06-0.1), use ~0.3
            center_transform: 'log_offset', 'scale', or 'linear'
                            - 'log_offset': log(center + offset) 
                            - 'scale': log(center * scale + 1) for small values
                            - 'linear': no transformation (original behavior)
            fix_height: Whether to correct cylinder height to match point extent.
                       If False, allows oversized cylinders but may help alignment optimization.
            random_axis_degrees: Optional random rotation to apply to initialized axis vector (in degrees).
                               Useful for testing robustness or escaping local minima. Default: 0.0 (no perturbation)
        """
        super().__init__(*args, **kwargs)
        self.center_offset = center_offset
        self.center_transform = center_transform
        self.center_scale = 100.0  # Scale small meter values (0.06-0.1) to reasonable range (6-10)
        self.fix_height = fix_height
        self.random_axis_degrees = random_axis_degrees
    
    def _apply_random_axis_rotation(self, axis_vector):
        """Apply random rotation to axis vector if random_axis_degrees > 0.
        
        Args:
            axis_vector: (3,) tensor representing the axis direction
            
        Returns:
            torch.Tensor: (3,) rotated axis vector
        """
        if self.random_axis_degrees <= 0:
            return axis_vector
            
        # Convert degrees to radians
        max_angle_rad = np.deg2rad(self.random_axis_degrees)
        
        # Generate random rotation axis (uniformly distributed on unit sphere)
        random_axis = torch.randn(3, device=self.device, dtype=axis_vector.dtype)
        random_axis = random_axis / (torch.norm(random_axis) + 1e-8)
        
        # Generate random rotation angle (uniform between -max_angle and +max_angle)
        random_angle = (torch.rand(1, device=self.device, dtype=axis_vector.dtype) * 2 - 1) * max_angle_rad
        
        # Create rotation matrix using Rodrigues' formula
        rotation_matrix = RotationUtils.rot_from_axis_angle(random_axis * random_angle)
        
        # Apply rotation to axis vector
        rotated_axis = rotation_matrix @ axis_vector
        
        logger.debug(f"Applied random rotation of {torch.rad2deg(random_angle).item():.2f}° around axis {random_axis.cpu().numpy()}")
        logger.debug(f"Original axis: {axis_vector.cpu().numpy()}")
        logger.debug(f"Rotated axis: {rotated_axis.cpu().numpy()}")
        
        return rotated_axis
    
    def _initialize_parameters_pca(self, points_inside):
        """Cylinder-specific PCA: align principal axis with +z, estimate params in local coords."""
        if len(points_inside) < 3:
            raise ValueError("Need at least 3 points for PCA initialization")
            
        c0 = points_inside.mean(0)
        X = points_inside - c0
        
        # Check for degenerate case
        if X.var(dim=0).min() < 1e-10:
            warnings.warn("Points appear to be colinear or coplanar")
        
        # Principal axis = first column of V
        try:
            _, _, V = torch.pca_lowrank(X, q=1)
            axis = V[:, 0]
            axis = axis / (axis.norm() + 1e-8)  # Normalize axis direction
        except Exception:
            # Fallback to using Z-axis
            axis = torch.tensor([0, 0, 1.], device=self.device)
        
        # Apply random rotation to axis if requested
        axis = self._apply_random_axis_rotation(axis)
        
        # Construct rotation matrix from axis for local coordinate calculations
        R0 = construct_cylinder_basis(axis)
        
        # Local coords & extents
        u_local = X @ R0
        half_len0 = torch.clamp(u_local[:, 2].abs().max() * 1.05, min=1e-3)
        radius0 = torch.clamp(torch.sqrt(u_local[:, 0] ** 2 + u_local[:, 1] ** 2).mean(), min=1e-3)
        
        return c0, radius0, half_len0, axis  # Return axis vector instead of rotation matrix
    
    def _initialize_parameters(self, points_inside, mesh=None, surface_name=None):
        """Route to appropriate initialization method based on self.initialization."""
        if self.initialization == 'geometric':
            logger.info(f"Initializing cylinder using geometric method for {surface_name}")
            return self._initialize_parameters_geometric(mesh, surface_name)
        else:  # 'pca' or fallback
            return self._initialize_parameters_pca(points_inside)
    
    def _initialize_parameters_geometric(self, mesh, surface_name):
        """Initialize parameters using geometric surface analysis method."""
        try:
            # Get near-surface points from pre-labeled mesh data
            near_surface_points = surface_param_estimation.get_near_surface_points_from_mesh(mesh, surface_name)
            
            logger.debug(f"Near-surface points for {surface_name}: {len(near_surface_points)} points")
            logger.debug(f"Point coordinate ranges:")
            logger.debug(f"  X: [{near_surface_points[:, 0].min():.4f}, {near_surface_points[:, 0].max():.4f}]")
            logger.debug(f"  Y: [{near_surface_points[:, 1].min():.4f}, {near_surface_points[:, 1].max():.4f}]")
            logger.debug(f"  Z: [{near_surface_points[:, 2].min():.4f}, {near_surface_points[:, 2].max():.4f}]")
            
            if len(near_surface_points) < 6:
                raise ValueError(f"Not enough near-surface points ({len(near_surface_points)}) for geometric cylinder fitting")
            
            # Convert to torch tensor
            points_tensor = torch.from_numpy(near_surface_points).float().to(self.device)
            
            # Fit cylinder using geometric method
            cylinder_params = surface_param_estimation.fit_cylinder_geometric(points_tensor)
            
            if not cylinder_params['success']:
                raise ValueError("Geometric cylinder fitting failed")
            
            # Extract parameters in the format expected by CylinderFitter
            center = cylinder_params['center'].to(self.device)
            radius = cylinder_params['radius'].to(self.device)
            half_length = cylinder_params['half_length'].to(self.device)
            rotation = cylinder_params['rotation'].to(self.device)
            
            # Extract axis vector from rotation matrix (3rd column = Z axis in local coords)
            axis_vector = rotation[:, 2]  # Extract the Z-axis direction
            
            # Apply random rotation to axis if requested
            axis_vector = self._apply_random_axis_rotation(axis_vector)
            
            # CRITICAL FIX: Verify and correct half_length based on actual points
            # Construct rotation matrix from axis for point projection
            R = construct_cylinder_basis(axis_vector)
            local_points = (points_tensor - center) @ R
            actual_z_extent = local_points[:, 2].abs().max()
            
            if self.fix_height:
                # Use a safety factor to ensure all points are within the cylinder
                corrected_half_length = actual_z_extent * 1.2  # 20% margin
                
                logger.debug(f"Geometric cylinder parameters for {surface_name}:")
                logger.debug(f"  Center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
                logger.debug(f"  Radius: {radius:.4f}")
                logger.debug(f"  Axis: [{axis_vector[0]:.4f}, {axis_vector[1]:.4f}, {axis_vector[2]:.4f}]")
                logger.debug(f"  Original half-length: {half_length:.4f}")
                logger.debug(f"  Actual point z-extent: {actual_z_extent:.4f}")
                logger.debug(f"  Corrected half-length: {corrected_half_length:.4f}")
                
                # Use corrected half-length
                half_length = corrected_half_length
            else:
                logger.debug(f"Geometric cylinder parameters for {surface_name} (height NOT corrected):")
                logger.debug(f"  Center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
                logger.debug(f"  Radius: {radius:.4f}")
                logger.debug(f"  Axis: [{axis_vector[0]:.4f}, {axis_vector[1]:.4f}, {axis_vector[2]:.4f}]")
                logger.debug(f"  Half-length: {half_length:.4f} (keeping original)")
                logger.debug(f"  Actual point z-extent: {actual_z_extent:.4f}")
                logger.debug(f"  Height ratio: {half_length/actual_z_extent:.2f}x larger than needed")
            
            return center, radius, half_length, axis_vector  # Return axis vector instead of rotation matrix
            
        except Exception as e:
            warnings.warn(f"Geometric initialization failed: {e}. Falling back to PCA.")
            # We need points_inside for PCA fallback, but we don't have them here
            # Let's create some dummy inside points from the surface points
            near_surface_points = surface_param_estimation.get_near_surface_points_from_mesh(mesh, surface_name)
            points_tensor = torch.from_numpy(near_surface_points).float().to(self.device)
            # Use a subset as "inside" points for PCA
            return self._initialize_parameters_pca(points_tensor[:len(points_tensor)//2])
    
    def _create_parameters(self, initial_params):
        center0, radius0, half_len0, axis0 = initial_params  # axis0 is now a vector, not rotation matrix
        
        # Transform center based on chosen method
        if self.center_transform == 'log_offset':
            # Original approach: log(center + offset)
            center_shifted = center0 + self.center_offset
            log_center = torch.nn.Parameter(torch.clamp(center_shifted, min=1e-6).log())
        elif self.center_transform == 'scale':
            # Better for small values: log(center * scale + 1)
            # This maps small values like 0.06-0.1 to log(6-10+1) = log(7-11) ≈ 1.9-2.4
            center_scaled = center0 * self.center_scale + 1.0
            log_center = torch.nn.Parameter(torch.clamp(center_scaled, min=1e-6).log())
        elif self.center_transform == 'linear':
            # No transformation - original behavior
            log_center = torch.nn.Parameter(center0)
        else:
            raise ValueError(f"Unknown center_transform: {self.center_transform}")
        
        log_r = torch.nn.Parameter(torch.clamp(radius0, min=1e-6).log())
        log_h = torch.nn.Parameter(torch.clamp(half_len0, min=1e-6).log())
        
        # Axis vector parameterization (no constraints needed - normalization handled in forward pass)
        axis_vector = torch.nn.Parameter(axis0)
        
        return [log_center, log_r, log_h, axis_vector]
    
    def _compute_loss(self, d, y, margin):
        """Squared-hinge margin loss with proper inside/outside penalties.
        
        For points labeled as "inside": penalize if d > -margin (not sufficiently inside)
        For points labeled as "outside": penalize if d < margin (not sufficiently outside)
        This ensures all points contribute to the loss until they satisfy the margin constraint.
        """
        return (d**2).mean()
    
    
    def _compute_sdf(self, points, parameters):
        log_center, log_r, log_h, axis_vector = parameters
        
        # Transform back from log space based on chosen method
        if self.center_transform == 'log_offset':
            center = torch.exp(torch.clamp(log_center, max=10)) - self.center_offset
        elif self.center_transform == 'scale':
            center = (torch.exp(torch.clamp(log_center, max=10)) - 1.0) / self.center_scale
        elif self.center_transform == 'linear':
            center = log_center  # No transformation needed
        else:
            raise ValueError(f"Unknown center_transform: {self.center_transform}")
            
        r = torch.exp(torch.clamp(log_r, max=10))  # Prevent overflow
        h = torch.exp(torch.clamp(log_h, max=10))
        
        # Use the new axis vector parameterization
        return sd_cylinder_with_axis(points, center, r, h, axis_vector)
    
    def _compute_shape_loss(self, sdf, labels, sdf_ground_truth, near_surface_points_tensor, margin, epoch):
        # Apply margin decay for coarse-to-fine optimization
        margin_decayed = self._compute_margin_decay(margin, epoch)
        
        log_center, log_r, log_h, axis_vector = self._current_parameters
        r = torch.exp(torch.clamp(log_r, max=10))  # Prevent overflow
        h = torch.exp(torch.clamp(log_h, max=10))
        
        # Transform back from log space based on chosen method
        if self.center_transform == 'log_offset':
            center = torch.exp(torch.clamp(log_center, max=10)) - self.center_offset
        elif self.center_transform == 'scale':
            center = (torch.exp(torch.clamp(log_center, max=10)) - 1.0) / self.center_scale
        elif self.center_transform == 'linear':
            center = log_center  # No transformation needed
        else:
            raise ValueError(f"Unknown center_transform: {self.center_transform}")
        
        # d_norm_r is the sdf normalized by radius, useful for logging or relative analysis
        d_norm_r = sdf / (r + 1e-8) 
        # Squared-hinge margin loss now uses the world-unit sdf directly for an absolute margin
        margin_loss = self._compute_loss(sdf, labels, margin_decayed)
        
        # Distance loss (only if ground truth provided)
        distance_loss = torch.tensor(0.0, device=self.device)
        if sdf_ground_truth is not None and self.beta > 0:
            distance_loss = self._compute_distance_loss(sdf, sdf_ground_truth, sigma=1.0)
        else:
            distance_loss = torch.tensor(0.0, device=self.device)
        
        # Surface points loss (only if near_surface_points_tensor provided)
        surface_points_loss = torch.tensor(0.0, device=self.device)
        if near_surface_points_tensor is not None and len(near_surface_points_tensor) > 0 and self.gamma > 0:
            # Compute SDF for near-surface points using current parameters
            sdf_surface = self._compute_sdf(near_surface_points_tensor, self._current_parameters)
            # We want these points to be on the surface, so their SDF should be close to 0
            surface_points_loss = (sdf_surface ** 2).mean()
        else:
            surface_points_loss = torch.tensor(0.0, device=self.device)

        logger.info(
            f"Epoch {epoch}: Margin={margin_decayed:.6f} (decay={margin_decayed/margin:.3f}), "
            f"MarginLoss={margin_loss:.6f}, DistLoss={distance_loss:.6f}, SurfLoss={surface_points_loss:.6f}"
        )
        
        # Hybrid loss: alpha * margin_loss + beta * distance_loss + gamma * surface_points_loss
        return self.alpha * margin_loss + self.beta * distance_loss + self.gamma * surface_points_loss
    
    def _extract_results(self, parameters):
        log_center, log_r, log_h, axis_vector = parameters
        
        # Transform back from log space based on chosen method
        if self.center_transform == 'log_offset':
            center = torch.exp(log_center.detach()) - self.center_offset
        elif self.center_transform == 'scale':
            center = (torch.exp(log_center.detach()) - 1.0) / self.center_scale
        elif self.center_transform == 'linear':
            center = log_center.detach()  # No transformation needed
        else:
            raise ValueError(f"Unknown center_transform: {self.center_transform}")
        
        # Normalize the axis vector for the final result
        axis_normalized = axis_vector.detach() / (torch.norm(axis_vector.detach()) + 1e-8)
        
        # Construct rotation matrix from normalized axis for compatibility with existing code
        R = construct_cylinder_basis(axis_normalized)
        
        return (center,
                torch.stack([torch.exp(log_r), torch.exp(log_h)]),
                R)
    
    def _get_lr_scales(self):
        """Get learning rate scales for cylinder parameters.
        
        Parameters in order: [log_center, log_r, log_h, axis_vector]
        
        Returns:
            list: Learning rate scale factors for each parameter
        """
        return [
            0.1,    # log_center: increased for meaningful updates
            0.1,    # log_r: increased for radius updates  
            0.1,    # log_h: increased for height updates
            5e-3    # axis_vector: larger LR due to small gradients
        ]
    
    @property
    def wrap_params(self):
        center, (radius, half_length), rot_matrix = self._fitted_params
        
        # detach all and convert to numpy
        center = center.detach().cpu().numpy()
        radius = radius.detach().cpu().numpy()
        half_length = half_length.detach().cpu().numpy()
        rot_matrix = rot_matrix.detach().cpu().numpy()
        
        # convert rot_matrix to xyz euler angles
        xyz_body_rotation = RotationUtils.rot_to_euler_xyz_body(rot_matrix)
        
        return wrap_surface(
            name=None,
            body=None,
            type_='WrapCylinder',
            xyz_body_rotation=xyz_body_rotation,
            translation=center,
            radius=radius,
            length=half_length*4,
            dimensions=None
        )


class EllipsoidFitter(BaseShapeFitter):
    """Ellipsoid fitting using PCA initialization and PyTorch optimization."""
    
    def __init__(self, *args, center_offset=0.3, center_transform='linear', **kwargs):
        """
        Args:
            center_offset: Offset added to center coordinates before log transformation
                          to ensure positivity. Should be larger than expected coordinate range.
                          For bone coordinates in meters (0.06-0.1), use ~0.3
            center_transform: 'log_offset', 'scale', or 'linear'
                            - 'log_offset': log(center + offset) 
                            - 'scale': log(center * scale + 1) for small values
                            - 'linear': no transformation (original behavior)
        """
        super().__init__(*args, **kwargs)
        self.center_offset = center_offset
        self.center_transform = center_transform
        self.center_scale = 100.0  # Scale small meter values (0.06-0.1) to reasonable range (6-10)
    
    def _initialize_parameters_pca(self, points_inside):
        """Ellipsoid-specific PCA: all 3 components become ellipsoid semi-axes."""
        if len(points_inside) < 3:
            raise ValueError("Need at least 3 points for PCA initialization")
            
        c0 = points_inside.mean(0)
        X = points_inside - c0
        
        # Check for degenerate case
        if X.var(dim=0).min() < 1e-10:
            warnings.warn("Points appear to be colinear or coplanar")
        
        try:
            # PCA to get the principal axes
            _, _, Vt = torch.pca_lowrank(X, q=3)
            R0 = Vt.t()  # columns = local axes
            
            # Compute extents along principal axes
            u_local = X @ R0
            a0 = torch.clamp(u_local.abs().max(0).values * 1.05, min=1e-3)
        except Exception:
            # Fallback to axis-aligned ellipsoid
            R0 = torch.eye(3, device=self.device)
            a0 = torch.clamp(X.abs().max(0).values * 1.05, min=1e-3)
        
        return c0, a0, R0
    
    def _initialize_parameters(self, points_inside, mesh=None, surface_name=None):
        """Route to appropriate initialization method based on self.initialization."""
        if self.initialization == 'geometric':
            logger.info(f"Initializing ellipsoid using geometric method for {surface_name}")
            return self._initialize_parameters_geometric(mesh, surface_name)
        else:  # 'pca' or fallback
            return self._initialize_parameters_pca(points_inside)
    
    def _initialize_parameters_geometric(self, mesh, surface_name):
        """Initialize parameters using geometric surface analysis method."""
        try:
            # Get near-surface points from pre-labeled mesh data
            near_surface_points = surface_param_estimation.get_near_surface_points_from_mesh(mesh, surface_name)
            
            logger.debug(f"Near-surface points for {surface_name}: {len(near_surface_points)} points")
            logger.debug(f"Point coordinate ranges:")
            logger.debug(f"  X: [{near_surface_points[:, 0].min():.4f}, {near_surface_points[:, 0].max():.4f}]")
            logger.debug(f"  Y: [{near_surface_points[:, 1].min():.4f}, {near_surface_points[:, 1].max():.4f}]")
            logger.debug(f"  Z: [{near_surface_points[:, 2].min():.4f}, {near_surface_points[:, 2].max():.4f}]")
            
            if len(near_surface_points) < 9:
                raise ValueError(f"Not enough near-surface points ({len(near_surface_points)}) for geometric ellipsoid fitting")
            
            # Convert to torch tensor
            points_tensor = torch.from_numpy(near_surface_points).float().to(self.device)
            
            # Fit ellipsoid using geometric method
            ellipsoid_params = surface_param_estimation.fit_ellipsoid_algebraic(points_tensor)
            
            if not ellipsoid_params['success']:
                raise ValueError("Geometric ellipsoid fitting failed")
            
            # Extract parameters in the format expected by EllipsoidFitter
            center = ellipsoid_params['center'].to(self.device)
            axes = ellipsoid_params['axes'].to(self.device)
            rotation = ellipsoid_params['rotation'].to(self.device)
            
            logger.debug(f"Geometric ellipsoid parameters for {surface_name}:")
            logger.debug(f"  Center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
            logger.debug(f"  Axes: [{axes[0]:.4f}, {axes[1]:.4f}, {axes[2]:.4f}]")
            
            return center, axes, rotation
            
        except Exception as e:
            warnings.warn(f"Geometric initialization failed: {e}. Falling back to PCA.")
            # We need points_inside for PCA fallback, but we don't have them here
            # Let's create some dummy inside points from the surface points
            near_surface_points = surface_param_estimation.get_near_surface_points_from_mesh(mesh, surface_name)
            points_tensor = torch.from_numpy(near_surface_points).float().to(self.device)
            # Use a subset as "inside" points for PCA
            return self._initialize_parameters_pca(points_tensor[:len(points_tensor)//2])
    
    def _create_parameters(self, initial_params):
        center0, axes0, R0 = initial_params
        center = torch.nn.Parameter(center0)
        log_axes = torch.nn.Parameter(torch.clamp(axes0, min=1e-6).log())
        
        # 4. Rotation (quaternion parameterization)
        initial_quat = RotationUtils.quat_from_rot(R0)
        # Ensure initial quaternion is normalized
        initial_quat /= (torch.norm(initial_quat) + 1e-8)
        quat = torch.nn.Parameter(initial_quat)
        
        return [center, log_axes, quat]

    def _compute_loss(self, d, y, margin):
        """Squared-hinge margin loss with proper inside/outside penalties.
        
        For points labeled as "inside": penalize if d > -margin (not sufficiently inside)
        For points labeled as "outside": penalize if d < margin (not sufficiently outside)
        This ensures all points contribute to the loss until they satisfy the margin constraint.
        """
        mask_in, mask_out = y, ~y
        
        # Handle empty masks gracefully
        
        # d is signed distance. (-) is inside, (+) is outside. 
        # we compute losses separately for the inside and outside points.
        # for the inside points, they are negative. So, if the d (-) + the margin is > 0
        # then there should be a loss. Here, relu linearly increases. 
        loss_in = (F.relu(d + margin)[mask_in] ** 2).mean() if mask_in.any() else torch.tensor(0.0, device=self.device)
        
        # Outside points: penalize if they're not sufficiently outside (d < margin)  
        # for the outside points, they are positive. So, if the margin - d (+) is > 0
        # (i.e., d < margin), then there should be a loss.
        loss_out = (F.relu(margin - d)[mask_out]).mean() if mask_out.any() else torch.tensor(0.0, device=self.device)
        
        return loss_in + loss_out
        
        # return (d**2).mean()
    
    
    def _compute_sdf(self, points, parameters):
        center, log_axes, quat = parameters
        
        # Use the utility function instead of duplicating code
        R = RotationUtils.rot_from_quat(quat)
        axes = torch.exp(torch.clamp(log_axes, max=10))  # Prevent overflow
        
        # Use improved SDF that's more accurate than the normalized approximation
        # but faster than the exact iterative method
        return sd_ellipsoid_improved(points, center, axes, R)
    
    def _compute_shape_loss(self, sdf, labels, sdf_ground_truth, near_surface_points_tensor, margin, epoch):
        # Apply margin decay for coarse-to-fine optimization
        margin_decayed = self._compute_margin_decay(margin, epoch)
        
        # Margin loss
        margin_loss = self._compute_loss(sdf, labels, margin_decayed)
        
        # Distance loss (only if ground truth provided)
        distance_loss = torch.tensor(0.0, device=self.device)
        if sdf_ground_truth is not None and self.beta > 0:
            distance_loss = self._compute_distance_loss(sdf, sdf_ground_truth, sigma=1.0)
        else:
            distance_loss = torch.tensor(0.0, device=self.device)
        
        # Surface points loss (only if near_surface_points_tensor provided)
        surface_points_loss = torch.tensor(0.0, device=self.device)
        if near_surface_points_tensor is not None and len(near_surface_points_tensor) > 0:
            # Compute SDF for near-surface points using current parameters
            sdf_surface = self._compute_sdf(near_surface_points_tensor, self._current_parameters)
            # We want these points to be on the surface, so their SDF should be close to 0
            surface_points_loss = (sdf_surface ** 2).mean()
        else:
            surface_points_loss = torch.tensor(0.0, device=self.device)

        logger.info(
            f"Epoch {epoch}: Margin={margin_decayed:.6f} (decay={margin_decayed/margin:.3f}), "
            f"MarginLoss={margin_loss:.6f}, DistLoss={distance_loss:.6f}, SurfLoss={surface_points_loss:.6f}"
        )
        
        # Hybrid loss: alpha * margin_loss + beta * distance_loss + gamma * surface_points_loss
        return self.alpha * margin_loss + self.beta * distance_loss + self.gamma * surface_points_loss
    
    def _setup_plotting(self, plot):
        """Setup live plotting for ellipsoid."""
        if plot:
            plt.ion()
            fig, ax = plt.subplots(figsize=(6, 3))
            line, = ax.plot([], [], lw=1.5)
            ax.set_xlabel('iter')
            ax.set_ylabel('squared-hinge loss')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            return {'losses': [], 'fig': fig, 'ax': ax, 'line': line, 'type': 'live'}
        return None
    
    def _update_plotting(self, plot_data, loss_val, epoch, stage):
        """Update live plotting during training."""
        if plot_data is not None and plot_data.get('type') == 'live':
            plot_data['losses'].append(loss_val)
            
            # During L-BFGS, plot more frequently (every step or every few steps)
            if stage == "L-BFGS":
                # Plot every L-BFGS step since there are typically fewer of them
                should_plot = True
            else:
                # Regular training: plot every 50 epochs or at the last epoch
                should_plot = (epoch + 1) % 50 == 0 or epoch == self.epochs - 1
            
            if should_plot:
                plot_data['line'].set_data(range(len(plot_data['losses'])), plot_data['losses'])
                plot_data['ax'].relim()
                plot_data['ax'].autoscale_view()
                plot_data['fig'].canvas.draw()
                plot_data['fig'].canvas.flush_events()
    
    def _finalize_plotting(self, plot_data, plot):
        """Finalize live plotting."""
        if plot and plot_data and plot_data.get('type') == 'live':
            plt.ioff()
            plt.show()
    
    def _extract_results(self, parameters):
        center, log_axes, quat = parameters
        
        # Use utility function instead of duplicating code
        R = RotationUtils.rot_from_quat(quat.detach())
        
        return center.detach(), torch.exp(log_axes).detach(), R
    
    def _get_lr_scales(self):
        """Get learning rate scales for ellipsoid parameters.
        
        Parameters in order: [center, log_axes, quat]
        
        Returns:
            list: Learning rate scale factors for each parameter
        """
        return [
            1e-3,
            # 5e-7,   # center: more conservative for ellipsoid center
            1e-3,   # log_axes: conservative for axes scaling
            1e-3    # quat: moderate for quaternion rotation
        ]
    
    def _initialize_parameters_pca(self, points_inside):
        """Ellipsoid-specific PCA: all 3 components become ellipsoid semi-axes."""
        if len(points_inside) < 3:
            raise ValueError("Need at least 3 points for PCA initialization")
            
        c0 = points_inside.mean(0)
        X = points_inside - c0
        
        # Check for degenerate case
        if X.var(dim=0).min() < 1e-10:
            warnings.warn("Points appear to be colinear or coplanar")
        
        try:
            # PCA to get the principal axes
            _, _, Vt = torch.pca_lowrank(X, q=3)
            R0 = Vt.t()  # columns = local axes
            
            # Compute extents along principal axes
            u_local = X @ R0
            a0 = torch.clamp(u_local.abs().max(0).values * 1.05, min=1e-3)
        except Exception:
            # Fallback to axis-aligned ellipsoid
            R0 = torch.eye(3, device=self.device)
            a0 = torch.clamp(X.abs().max(0).values * 1.05, min=1e-3)
        
        return c0, a0, R0
    
    @property
    def wrap_params(self):
        center, axes, rot_matrix = self._fitted_params
        
        # detach all and convert to numpy
        center = center.detach().cpu().numpy()
        axes = axes.detach().cpu().numpy()
        rot_matrix = rot_matrix.detach().cpu().numpy()
        
        # convert rot_matrix to xyz euler angles
        xyz_body_rotation = RotationUtils.rot_to_euler_xyz_body(rot_matrix)
        
        return wrap_surface(
            name=None,
            body=None,
            type_='WrapEllipsoid',
            xyz_body_rotation=xyz_body_rotation,
            translation=center,
            radius=None,
            length=None,
            dimensions=axes
        )

# Add a new utility function to construct coordinate system from axis vector
def construct_cylinder_basis(axis_vector):
    """Construct orthonormal basis for cylinder with given axis direction.
    
    Args:
        axis_vector: (3,) tensor representing cylinder axis direction (will be normalized)
        
    Returns:
        torch.Tensor: (3, 3) rotation matrix where columns are [x_local, y_local, z_local]
                     and z_local is aligned with the normalized axis_vector
    """
    # Normalize the axis vector
    axis = axis_vector / (torch.norm(axis_vector) + 1e-8)
    
    # Choose an arbitrary vector not parallel to axis for cross product
    # Use the coordinate axis that is least aligned with our axis
    abs_axis = torch.abs(axis)
    if abs_axis[0] < abs_axis[1] and abs_axis[0] < abs_axis[2]:
        up = torch.tensor([1.0, 0.0, 0.0], device=axis.device, dtype=axis.dtype)
    elif abs_axis[1] < abs_axis[2]:
        up = torch.tensor([0.0, 1.0, 0.0], device=axis.device, dtype=axis.dtype)
    else:
        up = torch.tensor([0.0, 0.0, 1.0], device=axis.device, dtype=axis.dtype)
    
    # Construct orthonormal basis using Gram-Schmidt
    z_local = axis  # Cylinder axis
    x_local = up - torch.dot(up, z_local) * z_local
    x_local = x_local / (torch.norm(x_local) + 1e-8)
    y_local = torch.cross(z_local, x_local)
    
    # Return rotation matrix [x_local, y_local, z_local] as columns
    return torch.stack([x_local, y_local, z_local], dim=1)


def sd_cylinder_with_axis(points, center, radius, half_length, axis_vector):
    """Signed distance function for finite cylinder using axis vector parameterization.
    
    Args:
        points: (N, 3) tensor of points
        center: (3,) tensor cylinder center
        radius: scalar tensor cylinder radius  
        half_length: scalar tensor cylinder half-length
        axis_vector: (3,) tensor cylinder axis direction (will be normalized internally)
        
    Returns:
        (N,) tensor of signed distances
    """
    # Construct rotation matrix from axis vector
    rotation_matrix = construct_cylinder_basis(axis_vector)
    
    # Transform points to cylinder local coordinates
    p = (points - center) @ rotation_matrix                     # world → local
    
    # Distance from axis and from caps
    radial_dist = torch.linalg.norm(p[..., :2], dim=-1) - radius
    axial_dist = torch.abs(p[..., 2]) - half_length
    
    q = torch.stack([radial_dist, axial_dist], dim=-1)         # (..., 2)

    # Combine inside/outside distances
    outside = torch.clamp(q, min=0).norm(dim=-1)
    inside = torch.clamp(q.max(dim=-1).values, max=0)
    return outside + inside
