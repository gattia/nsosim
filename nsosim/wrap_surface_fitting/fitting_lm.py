"""
Wrap surface fitting using Levenberg-Marquardt optimization.

This module provides the same functionality as fitting.py but uses scipy's Levenberg-Marquardt
optimizer instead of PyTorch's Adam/L-BFGS. LM is well-suited for nonlinear least squares
problems like surface fitting.

Example usage:
    # Simple surface fitting - drive all points to surface (most common for LM)
    cylinder_fitter = CylinderFitterLM(
        initialization='geometric',
        target_distance=0.0  # Drive all points exactly to surface
    )
    
    center, sizes, rotation = cylinder_fitter.fit(
        points=surface_points,  # No labels needed!
        mesh=labeled_mesh,
        surface_name="femur_1"
    )
    
    # With inside/outside labels and target margins (for compatibility)
    cylinder_fitter = CylinderFitterLM(
        initialization='geometric',
        target_distance=0.001  # 1mm margin from surface
    )
    
    center, sizes, rotation = cylinder_fitter.fit(
        points=bone_points,
        labels=inside_outside_labels,  # True for inside, False for outside
        mesh=labeled_mesh,
        surface_name="femur_1"
    )
    
    # Ellipsoid fitting with PCA (no labels needed)
    ellipsoid_fitter = EllipsoidFitterLM(
        initialization='pca'
        # target_distance defaults to 0.001, but will be ignored since no labels
    )
    
    center, axes, rotation = ellipsoid_fitter.fit(
        points=surface_points  # All points driven to surface
    )
"""

raise Exception("This was a test that didn't really work - leaving here for reference/ future work if of interest.")

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from abc import ABC, abstractmethod
import warnings
import logging
from . import surface_param_estimation
from .fitting import (
    RotationUtils, 
    sd_cylinder_with_axis, 
    sd_ellipsoid_improved,
    construct_cylinder_basis
)

logger = logging.getLogger(__name__)


class BaseShapeFitterLM(ABC):
    """Base class for Levenberg-Marquardt shape fitting."""
    
    def __init__(self, target_distance=0.001, initialization='geometric', 
                 max_nfev=1000, ftol=1e-6, xtol=1e-6, gtol=1e-6,
                 random_axis_degrees=0.0):
        """
        Args:
            target_distance: Target distance from points to surface (used for labeled points)
            initialization: 'geometric' or 'pca'
            max_nfev: Maximum number of function evaluations
            ftol: Tolerance for termination by change in cost function (relaxed from 1e-8)
            xtol: Tolerance for termination by change in parameters (relaxed from 1e-8)
            gtol: Tolerance for termination by gradient norm (relaxed from 1e-8)
            random_axis_degrees: Random rotation applied to axis initialization (degrees)
        """
        self.target_distance = target_distance
        self.initialization = initialization
        self.max_nfev = max_nfev
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.random_axis_degrees = random_axis_degrees
        
        # Store current data for residual computation
        self.current_points = None
        self.current_labels = None
        
    def _prepare_data(self, points, labels=None):
        """Prepare points and labels for optimization."""
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        
        if points.shape[1] != 3:
            raise ValueError(f"Expected 3D points, got shape {points.shape}")
        
        # Store centroid for center normalization to improve numerical conditioning
        self.points_centroid = np.mean(points, axis=0)
        points_normalized = points - self.points_centroid
        
        if labels is not None:
            if not isinstance(labels, np.ndarray):
                labels = np.array(labels, dtype=bool)
            if len(labels) != len(points):
                raise ValueError(f"Labels length {len(labels)} doesn't match points length {len(points)}")
            return points_normalized, labels
        
        return points_normalized, None
    
    def _apply_random_axis_rotation(self, axis_vector):
        """Apply random rotation to axis vector if random_axis_degrees > 0."""
        if self.random_axis_degrees <= 0:
            return axis_vector
            
        # Convert to torch for rotation utilities, then back to numpy
        axis_torch = torch.from_numpy(axis_vector).float()
        
        max_angle_rad = np.deg2rad(self.random_axis_degrees)
        random_axis = torch.randn(3)
        random_axis = random_axis / (torch.norm(random_axis) + 1e-8)
        random_angle = (torch.rand(1) * 2 - 1) * max_angle_rad
        
        rotation_matrix = RotationUtils.rot_from_axis_angle(random_axis * random_angle)
        rotated_axis = rotation_matrix @ axis_torch
        
        logger.debug(f"Applied random rotation of {np.rad2deg(random_angle.item()):.2f}°")
        
        return rotated_axis.numpy()
    
    @abstractmethod
    def _initialize_parameters(self, points_inside, mesh=None, surface_name=None):
        """Initialize shape-specific parameters."""
        pass
    
    @abstractmethod
    def _parameters_to_vector(self, parameters):
        """Convert parameter tuple to optimization vector."""
        pass
    
    @abstractmethod
    def _vector_to_parameters(self, vector):
        """Convert optimization vector back to parameter tuple."""
        pass
    
    @abstractmethod
    def _compute_sdf_numpy(self, points, parameters):
        """Compute signed distance function using numpy."""
        pass
    
    def _compute_residuals(self, param_vector):
        """Compute residuals for Levenberg-Marquardt optimization.
        
        Returns:
            np.ndarray: Residuals for each point
        """
        parameters = self._vector_to_parameters(param_vector)
        sdf_values = self._compute_sdf_numpy(self.current_points, parameters)
        
        # # Compute target SDF values
        # if self.current_labels is None:
        #     # No labels provided - drive all points to surface (SDF = 0)
        #     target_sdf = np.zeros_like(sdf_values)
        # elif self.target_distance == 0.0:
        #     # Drive all points to surface regardless of labels
        #     target_sdf = np.zeros_like(sdf_values)
        # else:
        #     # Use labels with target distance
        #     # Inside points: target = -target_distance
        #     # Outside points: target = +target_distance
        #     target_sdf = np.where(self.current_labels, 
        #                          -self.target_distance,  # inside -> negative
        #                          +self.target_distance)  # outside -> positive
        
        # # Residuals = current_sdf - target_sdf
        # residuals = sdf_values - target_sdf
        
        return sdf_values
    
    def fit(self, points, labels=None, mesh=None, surface_name=None, plot=False):
        """Fit shape using Levenberg-Marquardt optimization.
        
        Args:
            points: Input points
            labels: Inside/outside labels (True for inside). Optional - if not provided,
                   all points are driven to the surface (SDF=0)
            mesh: Mesh object (required for geometric initialization)
            surface_name: Surface name (required for geometric initialization)
            plot: Whether to plot optimization progress
            
        Returns:
            tuple: Fitted shape parameters
        """
        # Prepare data
        points, labels = self._prepare_data(points, labels)
        self.current_points = points
        self.current_labels = labels
        
        # Initialize parameters - need some inside points for initialization
        if labels is not None:
            inside_points = points[labels]
            if len(inside_points) == 0:
                raise ValueError("No inside points found for shape fitting")
        else:
            # If no labels, use all points for initialization (assume they're near the surface)
            inside_points = points
            logger.info("No labels provided - using all points for initialization")
        
        if len(inside_points) < 3:
            warnings.warn(f"Very few points ({len(inside_points)}) for robust fitting")
        
        # Check initialization requirements
        if self.initialization == 'geometric':
            if mesh is None or surface_name is None:
                warnings.warn("Geometric initialization requires mesh and surface_name. Falling back to PCA.")
                self.initialization = 'pca'
        
        try:
            initial_params = self._initialize_parameters(inside_points, mesh, surface_name)
        except Exception as e:
            raise ValueError(f"Failed to initialize parameters: {e}")
        
        # Convert to optimization vector
        initial_vector = self._parameters_to_vector(initial_params)
        
        logger.info(f"Starting LM optimization with {len(initial_vector)} parameters")
        logger.info(f"Points: {len(points)}, Labels: {'provided' if labels is not None else 'not provided'}")
        logger.info(f"Target distance: {self.target_distance}")
        logger.info(f"Initial parameter vector: {initial_vector}")
        
        # Setup plotting
        plot_data = {'residual_norms': [], 'iterations': []} if plot else None
        
        def callback(x, residuals):
            """Callback function for plotting progress."""
            if plot_data is not None:
                residual_norm = np.linalg.norm(residuals)
                plot_data['residual_norms'].append(residual_norm)
                plot_data['iterations'].append(len(plot_data['iterations']))
        
        # Run Levenberg-Marquardt optimization
        try:
            result = least_squares(
                fun=self._compute_residuals,
                x0=initial_vector,
                method='lm',
                ftol=self.ftol,
                xtol=self.xtol,
                gtol=self.gtol,
                max_nfev=self.max_nfev,
                verbose=2  # Show optimization progress
            )
            
            logger.info(f"LM optimization completed:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Message: {result.message}")
            logger.info(f"  Iterations: {result.nfev}")
            logger.info(f"  Final cost: {result.cost:.6e}")
            logger.info(f"  Final residual norm: {np.linalg.norm(result.fun):.6e}")
            
            if not result.success:
                warnings.warn(f"LM optimization did not converge: {result.message}")
            
            # Extract final parameters
            final_params = self._vector_to_parameters(result.x)
            
        except Exception as e:
            logger.error(f"LM optimization failed: {e}")
            raise
        
        # Plot results if requested
        if plot and plot_data and plot_data['residual_norms']:
            plt.figure(figsize=(8, 4))
            plt.plot(plot_data['iterations'], plot_data['residual_norms'])
            plt.xlabel('Function Evaluations')
            plt.ylabel('Residual Norm')
            plt.title(f'{self.__class__.__name__} Optimization Progress')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.show()
        
        return self._extract_results(final_params)
    
    @abstractmethod
    def _extract_results(self, parameters):
        """Extract final fitted parameters in expected format."""
        pass


class CylinderFitterLM(BaseShapeFitterLM):
    """Cylinder fitting using Levenberg-Marquardt optimization."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _initialize_parameters_pca(self, points_inside):
        """PCA-based cylinder initialization."""
        if len(points_inside) < 3:
            raise ValueError("Need at least 3 points for PCA initialization")
        
        # Convert to torch for existing PCA code
        points_torch = torch.from_numpy(points_inside).float()
        
        c0 = points_torch.mean(0)
        X = points_torch - c0
        
        if X.var(dim=0).min() < 1e-10:
            warnings.warn("Points appear to be colinear or coplanar")
        
        try:
            _, _, V = torch.pca_lowrank(X, q=1)
            axis = V[:, 0]
            axis = axis / (axis.norm() + 1e-8)
        except Exception:
            axis = torch.tensor([0, 0, 1.])
        
        # Apply random rotation
        axis_np = axis.numpy()
        axis_np = self._apply_random_axis_rotation(axis_np)
        axis = torch.from_numpy(axis_np).float()
        
        # Construct rotation matrix and compute extents
        R0 = construct_cylinder_basis(axis)
        u_local = X @ R0
        half_len0 = torch.clamp(u_local[:, 2].abs().max() * 1.05, min=1e-3)
        radius0 = torch.clamp(torch.sqrt(u_local[:, 0] ** 2 + u_local[:, 1] ** 2).mean(), min=1e-3)
        
        # Convert back to numpy
        return (c0.numpy(), radius0.numpy(), half_len0.numpy(), axis.numpy())
    
    def _initialize_parameters_geometric(self, mesh, surface_name):
        """Geometric initialization using surface analysis."""
        try:
            near_surface_points = surface_param_estimation.get_near_surface_points_from_mesh(mesh, surface_name)
            
            if len(near_surface_points) < 6:
                raise ValueError(f"Not enough near-surface points ({len(near_surface_points)}) for geometric cylinder fitting")
            
            points_tensor = torch.from_numpy(near_surface_points).float()
            cylinder_params = surface_param_estimation.fit_cylinder_geometric(points_tensor)
            
            if not cylinder_params['success']:
                raise ValueError("Geometric cylinder fitting failed")
            
            center = cylinder_params['center'].numpy()
            radius = cylinder_params['radius'].numpy()
            half_length = cylinder_params['half_length'].numpy()
            axis_vector = cylinder_params['rotation'][:, 2].numpy()  # Z-axis direction
            
            # Apply random rotation
            axis_vector = self._apply_random_axis_rotation(axis_vector)
            
            logger.debug(f"Geometric cylinder initialization:")
            logger.debug(f"  Center: {center}")
            logger.debug(f"  Radius: {radius}")
            logger.debug(f"  Half-length: {half_length}")
            logger.debug(f"  Axis: {axis_vector}")
            
            return center, radius, half_length, axis_vector
            
        except Exception as e:
            warnings.warn(f"Geometric initialization failed: {e}. Falling back to PCA.")
            # Use surface points as inside points for PCA fallback
            near_surface_points = surface_param_estimation.get_near_surface_points_from_mesh(mesh, surface_name)
            return self._initialize_parameters_pca(near_surface_points[:len(near_surface_points)//2])
    
    def _initialize_parameters(self, points_inside, mesh=None, surface_name=None):
        """Route to appropriate initialization method."""
        if self.initialization == 'geometric':
            logger.info(f"Initializing cylinder using geometric method for {surface_name}")
            return self._initialize_parameters_geometric(mesh, surface_name)
        else:
            return self._initialize_parameters_pca(points_inside)
    
    def _parameters_to_vector(self, parameters):
        """Convert (center, radius, half_length, axis_vector) to optimization vector."""
        center, radius, half_length, axis_vector = parameters
        
        # Center is already normalized by _prepare_data, so use as-is
        # Flatten all parameters into a single vector
        # [center_x, center_y, center_z, log_radius, log_half_length, axis_x, axis_y, axis_z]
        vector = np.concatenate([
            center.flatten() if hasattr(center, 'flatten') else np.array([center]).flatten(),
            np.array([np.log(max(radius, 1e-6))]).flatten(),
            np.array([np.log(max(half_length, 1e-6))]).flatten(),
            axis_vector.flatten() if hasattr(axis_vector, 'flatten') else np.array(axis_vector).flatten()
        ])
        
        return vector
    
    def _vector_to_parameters(self, vector):
        """Convert optimization vector back to (center, radius, half_length, axis_vector)."""
        center = vector[0:3]  # This is normalized center
        radius = np.exp(np.clip(vector[3], a_min=None, a_max=10))  # Prevent overflow
        half_length = np.exp(np.clip(vector[4], a_min=None, a_max=10))
        axis_vector = vector[5:8]
        
        # Normalize axis vector
        axis_norm = np.linalg.norm(axis_vector)
        if axis_norm > 1e-8:
            axis_vector = axis_vector / axis_norm
        else:
            axis_vector = np.array([0, 0, 1])  # Fallback to z-axis
        
        return center, radius, half_length, axis_vector
    
    def _compute_sdf_numpy(self, points, parameters):
        """Compute cylinder SDF using numpy."""
        center, radius, half_length, axis_vector = parameters
        
        # Convert to torch for existing SDF computation
        points_torch = torch.from_numpy(points).float()
        center_torch = torch.from_numpy(center).float()
        radius_torch = torch.tensor(radius, dtype=torch.float32)
        half_length_torch = torch.tensor(half_length, dtype=torch.float32)
        axis_torch = torch.from_numpy(axis_vector).float()
        
        # Compute SDF
        sdf_torch = sd_cylinder_with_axis(points_torch, center_torch, radius_torch, 
                                         half_length_torch, axis_torch)
        
        return sdf_torch.numpy()
    
    def _extract_results(self, parameters):
        """Extract results in format expected by existing code."""
        center, radius, half_length, axis_vector = parameters
        
        # Denormalize center by adding back the centroid
        center_denormalized = center + self.points_centroid
        
        # Convert to torch for rotation matrix construction
        axis_torch = torch.from_numpy(axis_vector).float()
        R = construct_cylinder_basis(axis_torch)
        
        return (
            torch.from_numpy(center_denormalized).float(),
            torch.tensor([radius, half_length], dtype=torch.float32),
            R
        )


class EllipsoidFitterLM(BaseShapeFitterLM):
    """Ellipsoid fitting using Levenberg-Marquardt optimization."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _initialize_parameters_pca(self, points_inside):
        """PCA-based ellipsoid initialization."""
        if len(points_inside) < 3:
            raise ValueError("Need at least 3 points for PCA initialization")
        
        # Convert to torch for existing PCA code
        points_torch = torch.from_numpy(points_inside).float()
        
        c0 = points_torch.mean(0)
        X = points_torch - c0
        
        if X.var(dim=0).min() < 1e-10:
            warnings.warn("Points appear to be colinear or coplanar")
        
        try:
            _, _, Vt = torch.pca_lowrank(X, q=3)
            R0 = Vt.t()
            u_local = X @ R0
            a0 = torch.clamp(u_local.abs().max(0).values * 1.05, min=1e-3)
        except Exception:
            R0 = torch.eye(3)
            a0 = torch.clamp(X.abs().max(0).values * 1.05, min=1e-3)
        
        # Convert back to numpy
        return c0.numpy(), a0.numpy(), R0.numpy()
    
    def _initialize_parameters_geometric(self, mesh, surface_name):
        """Geometric initialization using surface analysis."""
        try:
            near_surface_points = surface_param_estimation.get_near_surface_points_from_mesh(mesh, surface_name)
            
            if len(near_surface_points) < 9:
                raise ValueError(f"Not enough near-surface points ({len(near_surface_points)}) for geometric ellipsoid fitting")
            
            points_tensor = torch.from_numpy(near_surface_points).float()
            ellipsoid_params = surface_param_estimation.fit_ellipsoid_algebraic(points_tensor)
            
            if not ellipsoid_params['success']:
                raise ValueError("Geometric ellipsoid fitting failed")
            
            center = ellipsoid_params['center'].numpy()
            axes = ellipsoid_params['axes'].numpy()
            rotation = ellipsoid_params['rotation'].numpy()
            
            logger.debug(f"Geometric ellipsoid initialization:")
            logger.debug(f"  Center: {center}")
            logger.debug(f"  Axes: {axes}")
            
            return center, axes, rotation
            
        except Exception as e:
            warnings.warn(f"Geometric initialization failed: {e}. Falling back to PCA.")
            near_surface_points = surface_param_estimation.get_near_surface_points_from_mesh(mesh, surface_name)
            return self._initialize_parameters_pca(near_surface_points[:len(near_surface_points)//2])
    
    def _initialize_parameters(self, points_inside, mesh=None, surface_name=None):
        """Route to appropriate initialization method."""
        if self.initialization == 'geometric':
            logger.info(f"Initializing ellipsoid using geometric method for {surface_name}")
            return self._initialize_parameters_geometric(mesh, surface_name)
        else:
            return self._initialize_parameters_pca(points_inside)
    
    def _parameters_to_vector(self, parameters):
        """Convert (center, axes, rotation_matrix) to optimization vector."""
        center, axes, rotation_matrix = parameters
        
        # Center is already normalized by _prepare_data, so use as-is
        # Convert rotation matrix to quaternion for parameterization
        rotation_torch = torch.from_numpy(rotation_matrix).float()
        quat = RotationUtils.quat_from_rot(rotation_torch)
        quat = quat / (torch.norm(quat) + 1e-8)  # Normalize
        
        # Flatten parameters into vector
        # [center_x, center_y, center_z, log_axis_a, log_axis_b, log_axis_c, quat_w, quat_x, quat_y, quat_z]
        vector = np.concatenate([
            center.flatten(),
            np.log(np.maximum(axes, 1e-6)).flatten(),
            quat.numpy().flatten()
        ])
        
        return vector
    
    def _vector_to_parameters(self, vector):
        """Convert optimization vector back to (center, axes, rotation_matrix)."""
        center = vector[0:3]  # This is normalized center
        axes = np.exp(np.clip(vector[3:6], a_min=None, a_max=10))
        quat = vector[6:10]
        
        # Normalize quaternion
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 1e-8:
            quat = quat / quat_norm
        else:
            quat = np.array([1, 0, 0, 0])  # Identity quaternion
        
        # Convert quaternion to rotation matrix
        quat_torch = torch.from_numpy(quat).float()
        rotation_matrix = RotationUtils.rot_from_quat(quat_torch)
        
        return center, axes, rotation_matrix.numpy()
    
    def _compute_sdf_numpy(self, points, parameters):
        """Compute ellipsoid SDF using numpy."""
        center, axes, rotation_matrix = parameters
        
        # Convert to torch for existing SDF computation
        points_torch = torch.from_numpy(points).float()
        center_torch = torch.from_numpy(center).float()
        axes_torch = torch.from_numpy(axes).float()
        rotation_torch = torch.from_numpy(rotation_matrix).float()
        
        # Compute SDF
        sdf_torch = sd_ellipsoid_improved(points_torch, center_torch, axes_torch, rotation_torch)
        
        return sdf_torch.numpy()
    
    def _extract_results(self, parameters):
        """Extract results in format expected by existing code."""
        center, axes, rotation_matrix = parameters
        
        # Denormalize center by adding back the centroid
        center_denormalized = center + self.points_centroid
        
        return (
            torch.from_numpy(center_denormalized).float(),
            torch.from_numpy(axes).float(),
            torch.from_numpy(rotation_matrix).float()
        ) 