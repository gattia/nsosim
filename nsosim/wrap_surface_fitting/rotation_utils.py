"""Quaternion/Euler angle conversions and rotation matrix utilities."""

import logging
from typing import Union

import numpy as np
import torch

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

        return torch.stack(
            [
                torch.stack(
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=0
                ),
                torch.stack(
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=0
                ),
                torch.stack(
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=0
                ),
            ],
            dim=0,
        )

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
            if (
                axis[0] < 0
                or (axis[0] == 0 and axis[1] < 0)
                or (axis[0] == 0 and axis[1] == 0 and axis[2] < 0)
            ):
                axis = -axis

            return axis * angle

        # General case: extract axis from skew-symmetric part
        sin_angle = torch.sin(angle)
        # Ensure sin_angle is not zero to prevent division by zero
        sin_angle = torch.clamp(sin_angle, min=1e-8)
        axis = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (
            2 * sin_angle
        )

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
        K = torch.tensor(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
            device=axis_angle.device,
            dtype=axis_angle.dtype,
        )

        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)

        R = (
            torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
            + sin_angle * K
            + (1 - cos_angle) * torch.mm(K, K)
        )

        return R

    @staticmethod
    def rot_to_euler_xyz_body(
        R: Union[torch.Tensor, np.ndarray],
    ) -> Union[torch.Tensor, np.ndarray]:
        """Extract intrinsic XYZ (body) Euler angles from rotation matrix.

        OpenSim/Simbody uses Body-fixed X-Y-Z Euler angles (Intrinsic).
        R = Rx(x) * Ry(y) * Rz(z)

        Args:
            R (torch.Tensor or np.ndarray): Rotation matrix of shape (3, 3)

        Returns:
            torch.Tensor or np.ndarray: Euler angles [x, y, z] in radians, matching input type
        """
        input_type = "torch" if isinstance(R, torch.Tensor) else "numpy"

        if input_type == "numpy":
            R = torch.tensor(R, dtype=torch.float64)

        assert R.shape == (3, 3), f"Expected shape (3, 3), got {R.shape}"

        eps = 1e-6

        # Standard Intrinsic XYZ formulas:
        # R[0, 2] = sin(y)
        # R[1, 2] = -sin(x)cos(y)
        # R[2, 2] = cos(x)cos(y)
        # R[0, 1] = -cos(y)sin(z)
        # R[0, 0] = cos(y)cos(z)

        if torch.abs(R[0, 2] - 1.0) < eps:
            # Gimbal lock: y = pi/2
            x = torch.tensor(0.0, dtype=R.dtype, device=R.device)
            y = torch.tensor(np.pi / 2, dtype=R.dtype, device=R.device)
            z = torch.atan2(R[1, 0], R[1, 1])
        elif torch.abs(R[0, 2] + 1.0) < eps:
            # Gimbal lock: y = -pi/2
            # R[1,0] = sin(z - x), R[1,1] = cos(z - x); setting x=0 gives z
            x = torch.tensor(0.0, dtype=R.dtype, device=R.device)
            y = torch.tensor(-np.pi / 2, dtype=R.dtype, device=R.device)
            z = torch.atan2(R[1, 0], R[1, 1])
        else:
            # Standard case
            # y is simply asin(R[0,2])
            y = torch.asin(torch.clamp(R[0, 2], -1.0, 1.0))

            # x comes from -R[1,2] / R[2,2]
            x = torch.atan2(-R[1, 2], R[2, 2])

            # z comes from -R[0,1] / R[0,0]
            z = torch.atan2(-R[0, 1], R[0, 0])

        euler = torch.stack([x, y, z])

        if input_type == "numpy":
            return euler.numpy()
        return euler

    @staticmethod
    def canonical_ellipsoid_pose(
        R: Union[torch.Tensor, np.ndarray],
        axes: Union[torch.Tensor, np.ndarray],
    ):
        """Canonical (R, axes) for an ellipsoid, stable near gimbal lock.

        An ellipsoid {x : (x-c)ᵀ R diag(1/a²) Rᵀ (x-c) = 1} has 24 equivalent
        (R, axes) representations (axis permutations × sign-pair flips that
        preserve det = +1). The standard `enforce_sign_convention` keys on
        R[0,0] and R[1,1], whose magnitudes shrink to zero near gimbal lock —
        so a sub-percent perturbation of the input rotation can trigger a
        column-flip that swings the Euler representation by several degrees.

        This routine picks a canonical representative whose sign convention
        is keyed on each column's **dominant component** (the entry with
        largest |.|). That choice is stable as long as each column has a
        clear dominant component, which holds for anatomical wrap surfaces
        away from highly-symmetric configurations.

        Steps:
          1. Sort axes descending. Permute R columns to match (and absorb
             permutation parity into a column-negation if needed to keep
             det = +1).
          2. For column 0: if its dominant component is negative, negate
             columns 0 and 1 (preserves det).
          3. For column 1: if its dominant component is negative, negate
             columns 1 and 2 (preserves det).

        Returns (R_canonical, axes_canonical) with det(R_canonical) = +1 and
        the same geometric ellipsoid.
        """
        input_type = "torch" if isinstance(R, torch.Tensor) else "numpy"
        if input_type == "numpy":
            R_t = torch.tensor(R, dtype=torch.float64)
            axes_t = torch.tensor(axes, dtype=torch.float64)
        else:
            R_t = R.clone().to(torch.float64)
            axes_t = axes.clone().to(torch.float64)

        assert R_t.shape == (3, 3), f"Expected (3, 3), got {R_t.shape}"
        assert axes_t.shape == (3,), f"Expected (3,) axes, got {axes_t.shape}"

        # 1) Sort axes descending; permute columns to match.
        order = torch.argsort(-axes_t, stable=True)
        R_t = R_t[:, order]
        axes_t = axes_t[order]

        # 2) Ensure right-handed (det = +1) by flipping last column if needed.
        if torch.det(R_t) < 0:
            R_t[:, 2] *= -1

        # 3) Sign convention via dominant component of each column. Pair the
        # negation with another column so det stays +1.
        dom0 = int(torch.argmax(torch.abs(R_t[:, 0])))
        if R_t[dom0, 0] < 0:
            R_t[:, 0] *= -1
            R_t[:, 1] *= -1

        dom1 = int(torch.argmax(torch.abs(R_t[:, 1])))
        if R_t[dom1, 1] < 0:
            R_t[:, 1] *= -1
            R_t[:, 2] *= -1

        if input_type == "numpy":
            return R_t.numpy(), axes_t.numpy()
        return R_t.to(R.dtype), axes_t.to(axes.dtype)

    @staticmethod
    def enforce_sign_convention(
        R: Union[torch.Tensor, np.ndarray],
    ) -> Union[torch.Tensor, np.ndarray]:
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
        input_type = "torch" if isinstance(R, torch.Tensor) else "numpy"

        if input_type == "numpy":
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

        # 3. Re-guarantee right-handedness (since flipping X or Y alone flips determinant)
        if torch.det(R_fixed) < 0:
            R_fixed[:, 2] *= -1

        if input_type == "numpy":
            return R_fixed.numpy()
        return R_fixed
