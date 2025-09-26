#!/usr/bin/env python3
"""
Test script to compare RotationUtils.rot_to_euler_xyz_body vs scipy.as_euler('xyz')
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# Import the RotationUtils function
import sys
sys.path.append('.')
from nsosim.wrap_surface_fitting.rotation_utils import RotationUtils

def create_test_rotation_matrix():
    """Create a test rotation matrix from known Euler angles."""
    # Create a rotation matrix from known intrinsic XYZ Euler angles
    test_angles = [0.5, 0.3, 0.8]  # radians
    scipy_rot = R.from_euler('xyz', test_angles)
    return scipy_rot.as_matrix(), test_angles

def compare_methods():
    """Compare the two Euler angle extraction methods."""
    
    print("=== Euler Angle Extraction Comparison ===\n")
    
    # Test with several rotation matrices
    test_cases = [
        [0.0, 0.0, 0.0],      # Identity
        [0.5, 0.3, 0.8],      # General case
        [1.0, 0.2, -0.5],     # Another general case
        [0.1, 1.5, 0.1],      # Large Y rotation (near gimbal lock)
        [0.0, 1.57, 0.0],     # Gimbal lock case (Y = π/2)
    ]
    
    for i, original_angles in enumerate(test_cases):
        print(f"Test Case {i+1}: Original angles = {original_angles}")
        
        # Create rotation matrix from known angles
        scipy_rot = R.from_euler('xyz', original_angles)
        rot_matrix = scipy_rot.as_matrix()
        
        # Method 1: scipy as_euler('xyz')
        scipy_angles = scipy_rot.as_euler('xyz')
        
        # Method 2: RotationUtils.rot_to_euler_xyz_body
        utils_angles = RotationUtils.rot_to_euler_xyz_body(rot_matrix)
        
        # Compare results
        diff = np.abs(scipy_angles - utils_angles)
        max_diff = np.max(diff)
        
        print(f"  Scipy result:     {scipy_angles}")
        print(f"  RotationUtils:    {utils_angles}")
        print(f"  Difference:       {diff}")
        print(f"  Max difference:   {max_diff:.8f}")
        
        if max_diff > 1e-6:
            print(f"  ⚠️  SIGNIFICANT DIFFERENCE DETECTED!")
        else:
            print(f"  ✅ Methods agree within tolerance")
        
        print()

def test_specific_matrix():
    """Test with a specific problematic rotation matrix if you have one."""
    print("=== Testing with Identity Matrix ===")
    
    # Identity matrix should give zero rotations
    identity = np.eye(3)
    
    scipy_rot = R.from_matrix(identity)
    scipy_angles = scipy_rot.as_euler('xyz')
    utils_angles = RotationUtils.rot_to_euler_xyz_body(identity)
    
    print(f"Identity - Scipy:        {scipy_angles}")
    print(f"Identity - RotationUtils: {utils_angles}")
    print(f"Difference:              {np.abs(scipy_angles - utils_angles)}")
    print()

if __name__ == "__main__":
    compare_methods()
    test_specific_matrix()
