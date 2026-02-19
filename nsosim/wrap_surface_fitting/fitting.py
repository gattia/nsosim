"""PyTorch-based cylinder and ellipsoid fitters using SDF optimization."""

import logging
import warnings
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from . import surface_param_estimation
from .main import wrap_surface
from .rotation_utils import RotationUtils
from .wrap_signed_distances import sd_ellipsoid_improved

logger = logging.getLogger(__name__)  # This will be 'nsosim.wrap_surface_fitting.fitting'


class BaseShapeFitter(ABC):
    """Base class for shape fitting with common optimization logic."""

    def __init__(
        self,
        lr=5e-3,
        epochs=1500,
        device="cpu",
        use_lbfgs=False,
        lbfgs_epochs=100,
        alpha=1.0,
        beta=0.1,
        gamma=0.1,
        lbfgs_lr=1,
        margin_decay_type="exponential",
        initialization="geometric",
        lr_schedule=None,
        lr_schedule_params=None,
    ):
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.use_lbfgs = use_lbfgs
        self.lbfgs_epochs = lbfgs_epochs
        self.lbfgs_lr = lbfgs_lr
        self.alpha = alpha  # Weight for margin loss
        self.beta = beta  # Weight for distance loss
        self.gamma = gamma  # Weight for surface points loss
        self.margin_decay_type = margin_decay_type  # Type of margin decay
        self.initialization = initialization  # 'pca' or 'geometric'

        # Learning rate scheduling
        self.lr_schedule = lr_schedule  # 'cosine', 'exponential', 'step', 'plateau', or None
        self.lr_schedule_params = lr_schedule_params or {}  # Parameters for the scheduler

    def _prepare_data(self, points, labels):
        """Convert to tensors and validate."""
        if len(points) != len(labels):
            raise ValueError(
                f"Points ({len(points)}) and labels ({len(labels)}) must have same length"
            )

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
        raise NotImplementedError("Subclasses must implement _compute_loss")

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
            weights = torch.exp(-((abs_dist / sigma) ** 2))

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
    def _compute_shape_loss(
        self, sdf, labels, sdf_ground_truth, near_surface_points_tensor, margin, epoch
    ):
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
            return {"losses": [], "type": "simple"}
        return None

    def _update_plotting(self, plot_data, loss_val, epoch, _):
        """Update plotting during training. Override for custom plotting."""
        if plot_data is not None:
            plot_data["losses"].append(loss_val)

    def _finalize_plotting(self, plot_data, plot):
        """Finalize plotting after training. Override for custom plotting."""
        if plot and plot_data and plot_data["losses"]:
            plt.figure(figsize=(8, 4))
            plt.plot(plot_data["losses"])
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{self.__class__.__name__} Loss")
            plt.yscale("log")
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

        if decay_type == "exponential":
            # Exponential decay to ~1% of original margin
            decay_factor = np.exp(-2 * progress)  # e^(-4) ≈ 0.018
        elif decay_type == "linear":
            # Linear decay to zero
            decay_factor = 1 - progress
        elif decay_type == "cosine":
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

        if self.lr_schedule == "cosine":
            # Cosine annealing - smooth decay to min_lr over T_max epochs
            T_max = params.get("T_max", self.epochs)
            eta_min = params.get("eta_min", self.lr * 0.01)  # 1% of initial LR
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min
            )

        elif self.lr_schedule == "exponential":
            # Exponential decay - multiply LR by gamma each epoch
            gamma = params.get("gamma", 0.99)  # Default: 1% decay per epoch
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        elif self.lr_schedule == "step":
            # Step decay - reduce LR by factor at specific milestones
            step_size = params.get("step_size", self.epochs // 3)
            gamma = params.get("gamma", 0.5)  # Halve LR at each step
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        elif self.lr_schedule == "multistep":
            # Multi-step decay - reduce LR at multiple specific epochs
            milestones = params.get("milestones", [self.epochs // 3, 2 * self.epochs // 3])
            gamma = params.get("gamma", 0.5)
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=gamma
            )

        elif self.lr_schedule == "plateau":
            # Reduce on plateau - adaptive based on loss progress
            factor = params.get("factor", 0.5)  # Reduce LR by this factor
            patience = params.get("patience", 20)  # Wait this many epochs before reducing
            threshold = params.get("threshold", 1e-4)  # Minimum change to qualify as improvement
            min_lr = params.get("min_lr", self.lr * 1e-6)  # Minimum LR
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=factor,
                patience=patience,
                threshold=threshold,
                min_lr=min_lr,
                verbose=True,
            )

        else:
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")

    def _get_current_lr(self, optimizer):
        """Get current learning rate from optimizer."""
        return optimizer.param_groups[0]["lr"]

    def fit(
        self,
        points,
        labels,
        sdf=None,
        margin=0.05,
        plot=False,
        mesh=None,
        surface_name=None,
        near_surface_points=None,
    ):
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
                raise ValueError(
                    f"SDF ({len(sdf)}) and points ({len(points)}) must have same length"
                )
            sdf_tensor = torch.as_tensor(sdf, dtype=torch.float32, device=self.device)
        else:
            sdf_tensor = None

        # Prepare near_surface_points if provided
        near_surface_points_tensor = None
        if near_surface_points is not None:
            if len(near_surface_points) > 0:
                near_surface_points_tensor = torch.as_tensor(
                    near_surface_points, dtype=torch.float32, device=self.device
                )
                if near_surface_points_tensor.shape[1] != 3:
                    raise ValueError(
                        f"near_surface_points must be 3D, got shape {near_surface_points_tensor.shape}"
                    )
            else:
                warnings.warn("near_surface_points was provided but is empty.")

        # 2. Initialize parameters
        inside = x[y]
        if len(inside) == 0:
            raise ValueError("No inside points found for shape fitting")

        if len(inside) < 3:
            warnings.warn(f"Very few inside points ({len(inside)}) for robust fitting")

        # Check initialization requirements
        if self.initialization == "geometric":
            if mesh is None or surface_name is None:
                warnings.warn(
                    "Geometric initialization requires mesh and surface_name. Falling back to PCA."
                )
                self.initialization = "pca"

        try:
            initial_params = self._initialize_parameters(inside, mesh, surface_name)
        except Exception as e:
            raise ValueError(f"Failed to initialize parameters: {e}")

        # 3. Create parameters and optimizer with parameter-specific learning rates
        parameters = self._create_parameters(initial_params)

        # 4. Setup plotting
        plot_data = self._setup_plotting(plot)

        # Check if we should skip Adam optimization and go directly to L-BFGS
        skip_adam = self.epochs == 0

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

                param_groups.append(
                    {"params": [param], "lr": self.lr * lr_scale, "name": f"param_{i}"}
                )

            opt = torch.optim.Adam(param_groups)

            # Create learning rate scheduler
            scheduler = self._create_lr_scheduler(opt)

            # Store optimizer reference for logging
            self._current_optimizer = opt

            best_loss = float("inf")
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
                    loss = self._compute_shape_loss(
                        d, y, sdf_tensor, near_surface_points_tensor, margin, epoch
                    )

                    # Check for NaN/inf
                    if not torch.isfinite(loss):
                        warnings.warn(f"Non-finite loss at epoch {epoch}, stopping optimization")
                        break

                    loss.backward()

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
                final_epoch = (
                    0 if skip_adam else self.epochs
                )  # Use epoch 0 for margin calculation if we skipped Adam
                initial_loss = self._compute_shape_loss(
                    initial_d, y, sdf_tensor, near_surface_points_tensor, margin, final_epoch
                )
                logger.info(f"{stage_name} starting loss: {initial_loss.item():.6f}")

            # Restart L-BFGS periodically to avoid getting stuck
            restart_interval = max(
                10, self.lbfgs_epochs // 5
            )  # Restart every 10 steps or 1/5 of total steps
            logger.info(f"L-BFGS will restart every {restart_interval} steps")

            # Multiple L-BFGS steps with periodic restarts
            for lbfgs_step in range(self.lbfgs_epochs):
                # Create/recreate L-BFGS optimizer at start and after each restart
                if lbfgs_step % restart_interval == 0:
                    logger.debug(f"(Re)starting L-BFGS optimizer at step {lbfgs_step}")
                    lbfgs_opt = torch.optim.LBFGS(
                        parameters,
                        lr=self.lbfgs_lr,
                        max_iter=20,
                        max_eval=25,
                        tolerance_grad=1e-9,
                        tolerance_change=1e-12,  # Much looser tolerances
                        history_size=100,
                        line_search_fn="strong_wolfe",
                    )

                # Define closure for current step
                def closure():
                    lbfgs_opt.zero_grad()
                    self._current_parameters = parameters

                    try:
                        d = self._compute_sdf(x, parameters)
                        final_epoch = (
                            0 if skip_adam else self.epochs
                        )  # Use epoch 0 for margin calculation if we skipped Adam
                        loss = self._compute_shape_loss(
                            d, y, sdf_tensor, near_surface_points_tensor, margin, final_epoch
                        )

                        if torch.isfinite(loss):
                            loss.backward()
                            return loss
                        else:
                            return torch.tensor(float("inf"), device=self.device)

                    except Exception as e:
                        logger.warning(f"L-BFGS closure error: {e}")
                        return torch.tensor(float("inf"), device=self.device)

                try:
                    # Perform one L-BFGS optimization step
                    loss = lbfgs_opt.step(closure)
                    current_loss = loss.item() if loss is not None else float("inf")

                    # Normalize quaternion parameters after L-BFGS step (constraint enforcement)
                    # Do this outside autograd to avoid gradient contamination
                    with torch.no_grad():
                        for param in parameters:
                            if param.shape == (4,):  # This is a quaternion parameter
                                # Log quaternion norm before normalization (for debugging)
                                if lbfgs_step % 20 == 0:  # Log every 20 steps to avoid spam
                                    quat_norm_before = param.data.norm(p=2).item()
                                    logger.debug(
                                        f"L-BFGS step {lbfgs_step}: quaternion norm before normalization: {quat_norm_before:.6f}"
                                    )

                                param.data = param.data / (param.data.norm(p=2) + 1e-8)

                                # Log quaternion norm after normalization (for debugging)
                                if lbfgs_step % 20 == 0:  # Log every 20 steps to avoid spam
                                    quat_norm_after = param.data.norm(p=2).item()
                                    logger.debug(
                                        f"L-BFGS step {lbfgs_step}: quaternion norm after normalization: {quat_norm_after:.6f}"
                                    )

                    # Plot after each L-BFGS step (these are actual optimization progress points)
                    epoch_offset = 0 if skip_adam else self.epochs
                    self._update_plotting(
                        plot_data, current_loss, epoch_offset + 1 + lbfgs_step, "L-BFGS"
                    )

                    if lbfgs_step % 10 == 0 or lbfgs_step < 5:
                        logger.debug(
                            f"L-BFGS step {lbfgs_step+1}/{self.lbfgs_epochs}: loss={current_loss:.6f}"
                        )

                    # Early stopping if loss becomes non-finite
                    if not torch.isfinite(torch.tensor(current_loss)):
                        logger.warning(
                            f"L-BFGS stopping at step {lbfgs_step} due to non-finite loss"
                        )
                        break

                except Exception as e:
                    logger.warning(f"L-BFGS step {lbfgs_step} failed: {e}")
                    break

            # Final loss after L-BFGS
            with torch.no_grad():
                self._current_parameters = parameters
                final_d = self._compute_sdf(x, parameters)
                final_epoch = (
                    0 if skip_adam else self.epochs
                )  # Use epoch 0 for margin calculation if we skipped Adam
                final_loss = self._compute_shape_loss(
                    final_d, y, sdf_tensor, near_surface_points_tensor, margin, final_epoch
                )
                logger.info(f"{stage_name} final loss: {final_loss.item():.6f}")
                improvement = ((initial_loss - final_loss) / initial_loss * 100).item()
                logger.info(f"{stage_name} improvement: {improvement:.2f}%")

                # Validate final rotation matrix (for ellipsoid fits)
                for param in parameters:
                    if param.shape == (4,):  # This is a quaternion parameter
                        # Check quaternion norm
                        quat_norm = param.data.norm(p=2).item()
                        if abs(quat_norm - 1.0) > 1e-6:
                            logger.warning(
                                f"Final quaternion norm is {quat_norm:.6f} (should be 1.0)"
                            )

                        # Check rotation matrix validity
                        R = RotationUtils.rot_from_quat(param.data)
                        R_det = torch.det(R).item()
                        R_orthogonal = torch.allclose(
                            R @ R.T, torch.eye(3, device=R.device), atol=1e-6
                        )

                        if abs(R_det - 1.0) > 1e-6:
                            logger.warning(
                                f"Final rotation matrix determinant is {R_det:.6f} (should be 1.0)"
                            )
                        if not R_orthogonal:
                            logger.warning("Final rotation matrix is not orthogonal")

                        logger.info(
                            f"Final rotation validation: det={R_det:.6f}, orthogonal={R_orthogonal}, quat_norm={quat_norm:.6f}"
                        )
                        break

            logger.info(f"{stage_name} completed successfully")

            # Re-finalize plotting after L-BFGS
            self._finalize_plotting(plot_data, plot)

        # 7. Extract and return results
        final_results = self._extract_results(parameters)

        self._fitted_params = final_results

        # Debug: Log final rotation parameters
        if len(parameters) >= 4:
            axis_param = parameters[3].detach()
            logger.debug(f"Final axis vector parameter: {axis_param.cpu().numpy()}")
            logger.debug(f"Final rotation matrix:\n{final_results[2].cpu().numpy()}")
            logger.debug(
                f"Is rotation matrix identity? {torch.allclose(final_results[2], torch.eye(3), atol=1e-4)}"
            )

        return final_results

    @property
    def wrap_params(self):
        raise NotImplementedError("This method should be implemented by the subclass")


class CylinderFitter(BaseShapeFitter):
    """Cylinder fitting using PCA initialization and PyTorch optimization with axis vector parameterization."""

    def __init__(
        self,
        *args,
        center_offset=0.3,
        center_transform="linear",
        fix_height=False,
        random_axis_degrees=0.0,
        **kwargs,
    ):
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
        random_angle = (
            torch.rand(1, device=self.device, dtype=axis_vector.dtype) * 2 - 1
        ) * max_angle_rad

        # Create rotation matrix using Rodrigues' formula
        rotation_matrix = RotationUtils.rot_from_axis_angle(random_axis * random_angle)

        # Apply rotation to axis vector
        rotated_axis = rotation_matrix @ axis_vector

        logger.debug(
            f"Applied random rotation of {torch.rad2deg(random_angle).item():.2f}° around axis {random_axis.cpu().numpy()}"
        )
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
            axis = torch.tensor([0, 0, 1.0], device=self.device)

        # Apply random rotation to axis if requested
        axis = self._apply_random_axis_rotation(axis)

        # Construct rotation matrix from axis for local coordinate calculations
        # Use default +X alignment for consistent quadrant orientation
        R0 = construct_cylinder_basis(axis, reference_x_axis=None)

        # Local coords & extents
        u_local = X @ R0
        half_len0 = torch.clamp(u_local[:, 2].abs().max() * 1.05, min=1e-3)
        radius0 = torch.clamp(torch.sqrt(u_local[:, 0] ** 2 + u_local[:, 1] ** 2).mean(), min=1e-3)

        return c0, radius0, half_len0, axis  # Return axis vector instead of rotation matrix

    def _initialize_parameters(self, points_inside, mesh=None, surface_name=None):
        """Route to appropriate initialization method based on self.initialization."""
        if self.initialization == "geometric":
            logger.info(f"Initializing cylinder using geometric method for {surface_name}")
            return self._initialize_parameters_geometric(mesh, surface_name)
        else:  # 'pca' or fallback
            return self._initialize_parameters_pca(points_inside)

    def _initialize_parameters_geometric(self, mesh, surface_name):
        """Initialize parameters using geometric surface analysis method."""
        try:
            # Get near-surface points from pre-labeled mesh data
            near_surface_points = surface_param_estimation.get_near_surface_points_from_mesh(
                mesh, surface_name
            )

            logger.debug(
                f"Near-surface points for {surface_name}: {len(near_surface_points)} points"
            )
            logger.debug(f"Point coordinate ranges:")
            logger.debug(
                f"  X: [{near_surface_points[:, 0].min():.4f}, {near_surface_points[:, 0].max():.4f}]"
            )
            logger.debug(
                f"  Y: [{near_surface_points[:, 1].min():.4f}, {near_surface_points[:, 1].max():.4f}]"
            )
            logger.debug(
                f"  Z: [{near_surface_points[:, 2].min():.4f}, {near_surface_points[:, 2].max():.4f}]"
            )

            if len(near_surface_points) < 6:
                raise ValueError(
                    f"Not enough near-surface points ({len(near_surface_points)}) for geometric cylinder fitting"
                )

            # Convert to torch tensor
            points_tensor = torch.from_numpy(near_surface_points).float().to(self.device)

            # Fit cylinder using geometric method
            cylinder_params = surface_param_estimation.fit_cylinder_geometric(points_tensor)

            if not cylinder_params["success"]:
                raise ValueError("Geometric cylinder fitting failed")

            # Extract parameters in the format expected by CylinderFitter
            center = cylinder_params["center"].to(self.device)
            radius = cylinder_params["radius"].to(self.device)
            half_length = cylinder_params["half_length"].to(self.device)
            rotation = cylinder_params["rotation"].to(self.device)

            # Extract axis vector from rotation matrix (3rd column = Z axis in local coords)
            axis_vector = rotation[:, 2]  # Extract the Z-axis direction

            # Apply random rotation to axis if requested
            axis_vector = self._apply_random_axis_rotation(axis_vector)

            # CRITICAL FIX: Verify and correct half_length based on actual points
            # Construct rotation matrix from axis for point projection
            # Use default +X alignment for consistent quadrant orientation
            R = construct_cylinder_basis(axis_vector, reference_x_axis=None)
            local_points = (points_tensor - center) @ R
            actual_z_extent = local_points[:, 2].abs().max()

            if self.fix_height:
                # Use a safety factor to ensure all points are within the cylinder
                corrected_half_length = actual_z_extent * 1.2  # 20% margin

                logger.debug(f"Geometric cylinder parameters for {surface_name}:")
                logger.debug(f"  Center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
                logger.debug(f"  Radius: {radius:.4f}")
                logger.debug(
                    f"  Axis: [{axis_vector[0]:.4f}, {axis_vector[1]:.4f}, {axis_vector[2]:.4f}]"
                )
                logger.debug(f"  Original half-length: {half_length:.4f}")
                logger.debug(f"  Actual point z-extent: {actual_z_extent:.4f}")
                logger.debug(f"  Corrected half-length: {corrected_half_length:.4f}")

                # Use corrected half-length
                half_length = corrected_half_length
            else:
                logger.debug(
                    f"Geometric cylinder parameters for {surface_name} (height NOT corrected):"
                )
                logger.debug(f"  Center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
                logger.debug(f"  Radius: {radius:.4f}")
                logger.debug(
                    f"  Axis: [{axis_vector[0]:.4f}, {axis_vector[1]:.4f}, {axis_vector[2]:.4f}]"
                )
                logger.debug(f"  Half-length: {half_length:.4f} (keeping original)")
                logger.debug(f"  Actual point z-extent: {actual_z_extent:.4f}")
                logger.debug(
                    f"  Height ratio: {half_length/actual_z_extent:.2f}x larger than needed"
                )

            return (
                center,
                radius,
                half_length,
                axis_vector,
            )  # Return axis vector instead of rotation matrix

        except Exception as e:
            warnings.warn(f"Geometric initialization failed: {e}. Falling back to PCA.")
            # We need points_inside for PCA fallback, but we don't have them here
            # Let's create some dummy inside points from the surface points
            near_surface_points = surface_param_estimation.get_near_surface_points_from_mesh(
                mesh, surface_name
            )
            points_tensor = torch.from_numpy(near_surface_points).float().to(self.device)
            # Use a subset as "inside" points for PCA
            return self._initialize_parameters_pca(points_tensor[: len(points_tensor) // 2])

    def _create_parameters(self, initial_params):
        center0, radius0, half_len0, axis0 = (
            initial_params  # axis0 is now a vector, not rotation matrix
        )

        # Transform center based on chosen method
        if self.center_transform == "log_offset":
            # Original approach: log(center + offset)
            center_shifted = center0 + self.center_offset
            log_center = torch.nn.Parameter(torch.clamp(center_shifted, min=1e-6).log())
        elif self.center_transform == "scale":
            # Better for small values: log(center * scale + 1)
            # This maps small values like 0.06-0.1 to log(6-10+1) = log(7-11) ≈ 1.9-2.4
            center_scaled = center0 * self.center_scale + 1.0
            log_center = torch.nn.Parameter(torch.clamp(center_scaled, min=1e-6).log())
        elif self.center_transform == "linear":
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
        if self.center_transform == "log_offset":
            center = torch.exp(torch.clamp(log_center, max=10)) - self.center_offset
        elif self.center_transform == "scale":
            center = (torch.exp(torch.clamp(log_center, max=10)) - 1.0) / self.center_scale
        elif self.center_transform == "linear":
            center = log_center  # No transformation needed
        else:
            raise ValueError(f"Unknown center_transform: {self.center_transform}")

        r = torch.exp(torch.clamp(log_r, max=10))  # Prevent overflow
        h = torch.exp(torch.clamp(log_h, max=10))

        # Use the new axis vector parameterization with consistent orientation
        return sd_cylinder_with_axis(points, center, r, h, axis_vector, reference_x_axis=None)

    def _compute_shape_loss(
        self, sdf, labels, sdf_ground_truth, near_surface_points_tensor, margin, epoch
    ):
        # Apply margin decay for coarse-to-fine optimization
        margin_decayed = self._compute_margin_decay(margin, epoch)

        log_center, log_r, log_h, axis_vector = self._current_parameters
        r = torch.exp(torch.clamp(log_r, max=10))  # Prevent overflow
        h = torch.exp(torch.clamp(log_h, max=10))

        # Transform back from log space based on chosen method
        if self.center_transform == "log_offset":
            center = torch.exp(torch.clamp(log_center, max=10)) - self.center_offset
        elif self.center_transform == "scale":
            center = (torch.exp(torch.clamp(log_center, max=10)) - 1.0) / self.center_scale
        elif self.center_transform == "linear":
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
        if (
            near_surface_points_tensor is not None
            and len(near_surface_points_tensor) > 0
            and self.gamma > 0
        ):
            # Compute SDF for near-surface points using current parameters
            sdf_surface = self._compute_sdf(near_surface_points_tensor, self._current_parameters)
            # We want these points to be on the surface, so their SDF should be close to 0
            surface_points_loss = (sdf_surface**2).mean()
        else:
            surface_points_loss = torch.tensor(0.0, device=self.device)

        logger.info(
            f"Epoch {epoch}: Margin={margin_decayed:.6f} (decay={margin_decayed/margin:.3f}), "
            f"MarginLoss={margin_loss:.6f}, DistLoss={distance_loss:.6f}, SurfLoss={surface_points_loss:.6f}"
        )

        # Hybrid loss: alpha * margin_loss + beta * distance_loss + gamma * surface_points_loss
        return (
            self.alpha * margin_loss + self.beta * distance_loss + self.gamma * surface_points_loss
        )

    def _extract_results(self, parameters):
        log_center, log_r, log_h, axis_vector = parameters

        # Transform back from log space based on chosen method
        if self.center_transform == "log_offset":
            center = torch.exp(log_center.detach()) - self.center_offset
        elif self.center_transform == "scale":
            center = (torch.exp(log_center.detach()) - 1.0) / self.center_scale
        elif self.center_transform == "linear":
            center = log_center.detach()  # No transformation needed
        else:
            raise ValueError(f"Unknown center_transform: {self.center_transform}")

        # Normalize the axis vector for the final result
        axis_normalized = axis_vector.detach() / (torch.norm(axis_vector.detach()) + 1e-8)

        # Construct rotation matrix from normalized axis for compatibility with existing code
        # Use default +X alignment for consistent quadrant orientation
        R = construct_cylinder_basis(axis_normalized, reference_x_axis=None)

        return (center, torch.stack([torch.exp(log_r), torch.exp(log_h)]), R)

    def _get_lr_scales(self):
        """Get learning rate scales for cylinder parameters.

        Parameters in order: [log_center, log_r, log_h, axis_vector]

        Returns:
            list: Learning rate scale factors for each parameter
        """
        return [
            0.1,  # log_center: increased for meaningful updates
            0.1,  # log_r: increased for radius updates
            0.1,  # log_h: increased for height updates
            5e-3,  # axis_vector: larger LR due to small gradients
        ]

    @property
    def wrap_params(self):
        center, (radius, half_length), rot_matrix = self._fitted_params

        # detach all and convert to numpy
        center = center.detach().cpu().numpy()
        radius = radius.detach().cpu().numpy()
        half_length = half_length.detach().cpu().numpy()
        rot_matrix = rot_matrix.detach().cpu().numpy()

        # For cylinders, DON'T apply enforce_sign_convention!
        # The rotation matrix from construct_cylinder_basis already has the correct
        # deterministic orientation (X-axis aligned with global +X by default).
        # Flipping columns would arbitrarily change the quadrant orientation.

        # convert rot_matrix to xyz euler angles directly
        xyz_body_rotation = RotationUtils.rot_to_euler_xyz_body(rot_matrix)

        return wrap_surface(
            name=None,
            body=None,
            type_="WrapCylinder",
            xyz_body_rotation=xyz_body_rotation,
            translation=center,
            radius=radius,
            length=half_length * 4,
            dimensions=None,
        )


class EllipsoidFitter(BaseShapeFitter):
    """Ellipsoid fitting using PCA initialization and PyTorch optimization."""

    def __init__(self, *args, center_offset=0.3, center_transform="linear", **kwargs):
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
        if self.initialization == "geometric":
            logger.info(f"Initializing ellipsoid using geometric method for {surface_name}")
            return self._initialize_parameters_geometric(mesh, surface_name)
        else:  # 'pca' or fallback
            return self._initialize_parameters_pca(points_inside)

    def _initialize_parameters_geometric(self, mesh, surface_name):
        """Initialize parameters using geometric surface analysis method."""
        try:
            # Get near-surface points from pre-labeled mesh data
            near_surface_points = surface_param_estimation.get_near_surface_points_from_mesh(
                mesh, surface_name
            )

            logger.debug(
                f"Near-surface points for {surface_name}: {len(near_surface_points)} points"
            )
            logger.debug(f"Point coordinate ranges:")
            logger.debug(
                f"  X: [{near_surface_points[:, 0].min():.4f}, {near_surface_points[:, 0].max():.4f}]"
            )
            logger.debug(
                f"  Y: [{near_surface_points[:, 1].min():.4f}, {near_surface_points[:, 1].max():.4f}]"
            )
            logger.debug(
                f"  Z: [{near_surface_points[:, 2].min():.4f}, {near_surface_points[:, 2].max():.4f}]"
            )

            if len(near_surface_points) < 9:
                raise ValueError(
                    f"Not enough near-surface points ({len(near_surface_points)}) for geometric ellipsoid fitting"
                )

            # Convert to torch tensor
            points_tensor = torch.from_numpy(near_surface_points).float().to(self.device)

            # Fit ellipsoid using geometric method
            ellipsoid_params = surface_param_estimation.fit_ellipsoid_algebraic(points_tensor)

            if not ellipsoid_params["success"]:
                raise ValueError("Geometric ellipsoid fitting failed")

            # Extract parameters in the format expected by EllipsoidFitter
            center = ellipsoid_params["center"].to(self.device)
            axes = ellipsoid_params["axes"].to(self.device)
            rotation = ellipsoid_params["rotation"].to(self.device)

            logger.debug(f"Geometric ellipsoid parameters for {surface_name}:")
            logger.debug(f"  Center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
            logger.debug(f"  Axes: [{axes[0]:.4f}, {axes[1]:.4f}, {axes[2]:.4f}]")

            return center, axes, rotation

        except Exception as e:
            warnings.warn(f"Geometric initialization failed: {e}. Falling back to PCA.")
            # We need points_inside for PCA fallback, but we don't have them here
            # Let's create some dummy inside points from the surface points
            near_surface_points = surface_param_estimation.get_near_surface_points_from_mesh(
                mesh, surface_name
            )
            points_tensor = torch.from_numpy(near_surface_points).float().to(self.device)
            # Use a subset as "inside" points for PCA
            return self._initialize_parameters_pca(points_tensor[: len(points_tensor) // 2])

    def _create_parameters(self, initial_params):
        center0, axes0, R0 = initial_params
        center = torch.nn.Parameter(center0)
        log_axes = torch.nn.Parameter(torch.clamp(axes0, min=1e-6).log())

        # 4. Rotation (quaternion parameterization)
        initial_quat = RotationUtils.quat_from_rot(R0)
        # Ensure initial quaternion is normalized
        initial_quat /= torch.norm(initial_quat) + 1e-8
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
        loss_in = (
            (F.relu(d + margin)[mask_in] ** 2).mean()
            if mask_in.any()
            else torch.tensor(0.0, device=self.device)
        )

        # Outside points: penalize if they're not sufficiently outside (d < margin)
        # for the outside points, they are positive. So, if the margin - d (+) is > 0
        # (i.e., d < margin), then there should be a loss.
        loss_out = (
            (F.relu(margin - d)[mask_out]).mean()
            if mask_out.any()
            else torch.tensor(0.0, device=self.device)
        )

        return loss_in + loss_out

    def _compute_sdf(self, points, parameters):
        center, log_axes, quat = parameters

        # Use the utility function instead of duplicating code
        R = RotationUtils.rot_from_quat(quat)
        axes = torch.exp(torch.clamp(log_axes, max=10))  # Prevent overflow

        # Use improved SDF that's more accurate than the normalized approximation
        # but faster than the exact iterative method
        return sd_ellipsoid_improved(points, center, axes, R)

    def _compute_shape_loss(
        self, sdf, labels, sdf_ground_truth, near_surface_points_tensor, margin, epoch
    ):
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
            surface_points_loss = (sdf_surface**2).mean()
        else:
            surface_points_loss = torch.tensor(0.0, device=self.device)

        logger.info(
            f"Epoch {epoch}: Margin={margin_decayed:.6f} (decay={margin_decayed/margin:.3f}), "
            f"MarginLoss={margin_loss:.6f}, DistLoss={distance_loss:.6f}, SurfLoss={surface_points_loss:.6f}"
        )

        # Hybrid loss: alpha * margin_loss + beta * distance_loss + gamma * surface_points_loss
        return (
            self.alpha * margin_loss + self.beta * distance_loss + self.gamma * surface_points_loss
        )

    def _setup_plotting(self, plot):
        """Setup live plotting for ellipsoid."""
        if plot:
            plt.ion()
            fig, ax = plt.subplots(figsize=(6, 3))
            (line,) = ax.plot([], [], lw=1.5)
            ax.set_xlabel("iter")
            ax.set_ylabel("squared-hinge loss")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            return {"losses": [], "fig": fig, "ax": ax, "line": line, "type": "live"}
        return None

    def _update_plotting(self, plot_data, loss_val, epoch, stage):
        """Update live plotting during training."""
        if plot_data is not None and plot_data.get("type") == "live":
            plot_data["losses"].append(loss_val)

            # During L-BFGS, plot more frequently (every step or every few steps)
            if stage == "L-BFGS":
                # Plot every L-BFGS step since there are typically fewer of them
                should_plot = True
            else:
                # Regular training: plot every 50 epochs or at the last epoch
                should_plot = (epoch + 1) % 50 == 0 or epoch == self.epochs - 1

            if should_plot:
                plot_data["line"].set_data(range(len(plot_data["losses"])), plot_data["losses"])
                plot_data["ax"].relim()
                plot_data["ax"].autoscale_view()
                plot_data["fig"].canvas.draw()
                plot_data["fig"].canvas.flush_events()

    def _finalize_plotting(self, plot_data, plot):
        """Finalize live plotting."""
        if plot and plot_data and plot_data.get("type") == "live":
            plt.ioff()
            plt.show()

    def _extract_results(self, parameters):
        center, log_axes, quat = parameters

        # Normalize quaternion before converting to rotation matrix
        quat_normalized = quat.detach() / (torch.norm(quat.detach()) + 1e-8)

        # Use utility function instead of duplicating code
        R = RotationUtils.rot_from_quat(quat_normalized)

        return center.detach(), torch.exp(log_axes).detach(), R

    def _get_lr_scales(self):
        """Get learning rate scales for ellipsoid parameters.

        Parameters in order: [center, log_axes, quat]

        Returns:
            list: Learning rate scale factors for each parameter
        """
        return [
            1e-3,  # center
            1e-3,  # log_axes: conservative for axes scaling
            1e-3,  # quat: moderate for quaternion rotation
        ]

    @property
    def wrap_params(self):
        center, axes, rot_matrix = self._fitted_params

        # detach all and convert to numpy
        center = center.detach().cpu().numpy()
        axes = axes.detach().cpu().numpy()
        rot_matrix = rot_matrix.detach().cpu().numpy()

        # Apply sign convention fix to ensure deterministic Euler angles
        rot_matrix_fixed = RotationUtils.enforce_sign_convention(rot_matrix)

        # convert rot_matrix to xyz euler angles
        xyz_body_rotation = RotationUtils.rot_to_euler_xyz_body(rot_matrix_fixed)

        return wrap_surface(
            name=None,
            body=None,
            type_="WrapEllipsoid",
            xyz_body_rotation=xyz_body_rotation,
            translation=center,
            radius=None,
            length=None,
            dimensions=axes,
        )


# Construct coordinate system from axis vector for cylinder wrap surfaces.
# IMPORTANT: This function's design is critical for OpenSim cylinder quadrant
# orientation. The X-axis alignment and sign consistency preserve the quadrant
# parameter semantics (which side of a partial cylinder is active for wrapping).
# See commit 629e02d for full context on the quadrant orientation fix.
def construct_cylinder_basis(axis_vector, reference_x_axis=None, eps=1e-8):
    """
    Construct orthonormal basis for a cylinder with given axis direction.

    This function ensures consistent cylinder orientation for OpenSim wrap
    surfaces. The X-axis alignment and sign consistency are critical for
    preserving the quadrant parameter semantics (which side of a partial
    cylinder is active for muscle/ligament wrapping).

    Args:
        axis_vector: (3,) tensor, cylinder axis direction (need not be unit).
        reference_x_axis: (3,) optional tensor for desired X-axis direction.
            - If provided, x_local is as aligned as possible with this direction,
              projected onto the plane perpendicular to axis.
            - If None, we default to global +X [1,0,0] for consistency with
              typical femur cylinder orientations (anterior = +X, M/L axis = +Z).
              Falls back to +Y or +Z if axis is nearly parallel to +X.

    Returns:
        R: (3, 3) rotation matrix with columns [x_local, y_local, z_local]
           expressed in global coordinates. Guaranteed right-handed and
           with x_local consistently oriented toward reference direction.
    """
    # Normalize axis (z_local)
    z_local = axis_vector / (torch.norm(axis_vector) + eps)

    # Global canonical directions
    ex = torch.tensor([1.0, 0.0, 0.0], device=z_local.device, dtype=z_local.dtype)
    ey = torch.tensor([0.0, 1.0, 0.0], device=z_local.device, dtype=z_local.dtype)
    ez = torch.tensor([0.0, 0.0, 1.0], device=z_local.device, dtype=z_local.dtype)

    # 1) Choose a preferred direction for x
    if reference_x_axis is None:
        preferred = ex  # default: global +X (anatomical anterior for femur)
    else:
        preferred = reference_x_axis.to(device=z_local.device, dtype=z_local.dtype)
        preferred = preferred / (torch.norm(preferred) + eps)

    # 2) Project preferred into the plane perpendicular to z_local
    x_candidate = preferred - torch.dot(preferred, z_local) * z_local

    # 3) If preferred is too parallel to axis, pick a fallback direction
    if torch.norm(x_candidate) < 1e-6:
        # Try ex, then ey, then ez
        for d in (ex, ey, ez):
            cand = d - torch.dot(d, z_local) * z_local
            if torch.norm(cand) > 1e-6:
                x_candidate = cand
                preferred = d  # update for sign consistency check
                break

    x_local = x_candidate / (torch.norm(x_candidate) + eps)

    # 4) Enforce consistent sign: align x_local with preferred direction
    #    This prevents 180° flips that would change the quadrant
    if torch.dot(x_local, preferred) < 0:
        x_local = -x_local

    # 5) Complete right-handed frame
    y_local = torch.linalg.cross(z_local, x_local)
    y_local = y_local / (torch.norm(y_local) + eps)

    R = torch.stack([x_local, y_local, z_local], dim=1)
    return R


def sd_cylinder_with_axis(points, center, radius, half_length, axis_vector, reference_x_axis=None):
    """Signed distance function for finite cylinder using axis vector parameterization.

    Args:
        points: (N, 3) tensor of points
        center: (3,) tensor cylinder center
        radius: scalar tensor cylinder radius
        half_length: scalar tensor cylinder half-length
        axis_vector: (3,) tensor cylinder axis direction (will be normalized internally)
        reference_x_axis: (3,) optional tensor for consistent X-axis orientation

    Returns:
        (N,) tensor of signed distances
    """
    # Construct rotation matrix from axis vector with consistent orientation
    rotation_matrix = construct_cylinder_basis(axis_vector, reference_x_axis=reference_x_axis)

    # Transform points to cylinder local coordinates
    p = (points - center) @ rotation_matrix  # world → local

    # Distance from axis and from caps
    radial_dist = torch.linalg.norm(p[..., :2], dim=-1) - radius
    axial_dist = torch.abs(p[..., 2]) - half_length

    q = torch.stack([radial_dist, axial_dist], dim=-1)  # (..., 2)

    # Combine inside/outside distances
    outside = torch.clamp(q, min=0).norm(dim=-1)
    inside = torch.clamp(q.max(dim=-1).values, max=0)
    return outside + inside
