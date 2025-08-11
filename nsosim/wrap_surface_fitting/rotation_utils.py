import torch
import numpy as np
import logging
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
            # Ensure s is not zero to prevent division by zero
            s = torch.clamp(s, min=1e-8)
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s  
            z = (R[1, 0] - R[0, 1]) / s
        else:
            diag = torch.diagonal(R)
            i = torch.argmax(diag)
            
            if i == 0:
                s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                # Ensure s is not zero to prevent division by zero
                s = torch.clamp(s, min=1e-8)
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif i == 1:
                s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                # Ensure s is not zero to prevent division by zero
                s = torch.clamp(s, min=1e-8)
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                # Ensure s is not zero to prevent division by zero
                s = torch.clamp(s, min=1e-8)
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
        sin_angle = torch.sin(angle)
        # Ensure sin_angle is not zero to prevent division by zero
        sin_angle = torch.clamp(sin_angle, min=1e-8)
        axis = torch.stack([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0], 
            R[1, 0] - R[0, 1]
        ]) / (2 * sin_angle)
        
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
            # Add small epsilon to prevent atan2(0,0) which is undefined
            z = torch.atan2(-R[1, 0] + 1e-8, -R[2, 0] + 1e-8)
        elif torch.abs(R[0, 2] + 1.0) < eps:
            x = torch.tensor(0.0, dtype=R.dtype)
            y = torch.pi / 2
            # Add small epsilon to prevent atan2(0,0) which is undefined
            z = torch.atan2(R[1, 0] + 1e-8, R[2, 0] + 1e-8)
        else:
            # Clamp the value to prevent NaN from asin
            y = torch.asin(torch.clamp(-R[0, 2], -1.0, 1.0))
            # Add small epsilon to prevent atan2(0,0) which is undefined
            x = torch.atan2(R[1, 2] + 1e-8, R[2, 2] + 1e-8)
            z = torch.atan2(R[0, 1] + 1e-8, R[0, 0] + 1e-8)

        euler = torch.stack([x, y, z])

        if input_type == 'numpy':
            return euler.numpy()
        return euler

    @staticmethod
    def enforce_sign_convention(R: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Ensure rotation matrix has consistent sign convention for deterministic Euler angles.
        
        This function enforces a deterministic sign convention on rotation matrices to ensure
        that the same geometric orientation always produces the same Euler angles, regardless
        of the optimization process that produced the matrix.
        
        The convention used:
        1. Ensures right-handedness (det > 0)
        2. Makes first column (X-axis) point mainly in +X direction
        3. Makes second column (Y-axis) point mainly in +Y direction
        
        Args:
            R (torch.Tensor or np.ndarray): Rotation matrix of shape (3, 3)
            
        Returns:
            torch.Tensor or np.ndarray: Rotation matrix with consistent sign convention, matching input type
        """
        input_type = 'torch' if isinstance(R, torch.Tensor) else 'numpy'
        
        if input_type == 'numpy':
            R = torch.tensor(R, dtype=torch.float64)
        
        assert R.shape == (3, 3), f"Expected shape (3, 3), got {R.shape}"
        
        # Work on a copy to avoid modifying the original
        R_fixed = R.clone()
        
        # 1. Guarantee right-handedness
        if torch.det(R_fixed) < 0:
            R_fixed[:, 2] *= -1  # Flip Z column to make right-handed
        
        # 2. Canonicalize axis signs (choose a convention and document it)
        if R_fixed[0, 0] < 0:  # X-axis pointing mostly -X
            R_fixed[:, 0] *= -1
        if R_fixed[1, 1] < 0:  # Y-axis pointing mostly -Y  
            R_fixed[:, 1] *= -1
        
        if input_type == 'numpy':
            return R_fixed.numpy()
        return R_fixed