#!/usr/bin/env python3
"""
Quick test to verify if mesh size is causing the timing differences between bones.
This will load actual reference meshes and measure their sizes and interpolation times.
"""

import os
import json
import time
import numpy as np
import pyvista as pv

from pymskt.mesh import Mesh
from nsosim.nsm_fitting import interp_ref_to_subject_to_osim
from nsosim.utils import load_model

def load_actual_reference_meshes():
    """Load the actual reference meshes used in your pipeline"""
    BASE_FOLDER = '/dataNAS/people/aagatti/projects/comak_gait_simulation/COMAK_SIMULATION_REQUIREMENTS'
    folder_labeled_bone = os.path.join(BASE_FOLDER, 'fitted_base_wrap_surfaces', 'labeled_bones')
    
    mesh_paths = {
        'femur': os.path.join(folder_labeled_bone, 'femur_labeled.vtk'),
        'tibia': os.path.join(folder_labeled_bone, 'tibia_labeled.vtk'), 
        'patella': os.path.join(folder_labeled_bone, 'patella_labeled.vtk')
    }
    
    meshes = {}
    for bone, path in mesh_paths.items():
        if os.path.exists(path):
            mesh = Mesh(path)
            meshes[bone] = {
                'mesh': mesh,
                'n_points': mesh.point_coords.shape[0],
                'path': path
            }
            print(f"{bone.capitalize()}: {mesh.point_coords.shape[0]} points")
        else:
            print(f"WARNING: {bone} mesh not found at {path}")
    
    return meshes

def quick_interpolation_test(bone_name, mesh_data, n_steps=20):
    """Run a quick interpolation test on a bone"""
    
    print(f"\nTesting {bone_name} ({mesh_data['n_points']} points)...")
    
    # Load model and setup (simplified)
    BASE_FOLDER = '/dataNAS/people/aagatti/projects/comak_gait_simulation/COMAK_SIMULATION_REQUIREMENTS'
    folder_nsm_models = os.path.join(BASE_FOLDER, 'nsm_models')
    folder_ref_recons = os.path.join(BASE_FOLDER, 'nsm_meshes')
    
    model_configs = {
        'femur': {
            'path_model_state': os.path.join(folder_nsm_models, '568_nsm_femur_bone_cart_men_v0.0.1', 'model', '2000.pth'),
            'path_model_config': os.path.join(folder_nsm_models, '568_nsm_femur_bone_cart_men_v0.0.1', 'model_params_config.json')
        },
        'tibia': {
            'path_model_state': os.path.join(folder_nsm_models, '650_nsm_tibia_v0.0.1', 'model', '2000.pth'),
            'path_model_config': os.path.join(folder_nsm_models, '650_nsm_tibia_v0.0.1', 'model_params_config.json')
        },
        'patella': {
            'path_model_state': os.path.join(folder_nsm_models, '648_nsm_patella_v0.0.1', 'model', '2000.pth'),
            'path_model_config': os.path.join(folder_nsm_models, '648_nsm_patella_v0.0.1', 'model_params_config.json')
        }
    }
    
    try:
        # Load model
        config = model_configs[bone_name]
        with open(config['path_model_config'], 'r') as f:
            model_config = json.load(f)
        
        model = load_model(model_config, config['path_model_state'], model_type='triplanar')
        
        # Generate random latent vectors
        latent_size = model_config['latent_size']
        latent1 = np.random.normal(0, 0.5, latent_size).astype(np.float32)
        latent2 = np.random.normal(0, 0.5, latent_size).astype(np.float32)
        
        # Setup dict_bones
        dict_bones = {
            bone_name: {
                'subject': {
                    'recon_dict': {
                        'model': model,
                        'icp_transform': np.eye(4),
                        'scale': np.array([1.0]),
                        'center': np.array([0.0, 0.0, 0.0])
                    },
                    'recon_latent': latent2
                }
            }
        }
        
        # Get reference center
        fem_path_transform_file = os.path.join(folder_ref_recons, 'femur', 'ref_femur_alignment.json')
        with open(fem_path_transform_file, 'r') as f:
            dict_transforms = json.load(f)
            fem_ref_center = np.array(dict_transforms['mean_orig'])
        
        # Time the interpolation
        start_time = time.time()
        
        interpolated_pts = interp_ref_to_subject_to_osim(
            ref_mesh=mesh_data['mesh'].mesh,  # Use the actual mesh
            surface_name=bone_name,
            ref_center=fem_ref_center,
            dict_bones=dict_bones,
            folder_nsm_files=folder_ref_recons,
            surface_idx=0,
            n_steps=n_steps,
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        points_per_second = mesh_data['n_points'] * n_steps / duration
        
        print(f"  Time: {duration:.2f}s")
        print(f"  Points/sec: {points_per_second:.0f}")
        
        return {
            'bone': bone_name,
            'n_points': mesh_data['n_points'],
            'n_steps': n_steps,
            'time': duration,
            'points_per_second': points_per_second
        }
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def main():
    """Quick test to verify mesh size hypothesis"""
    
    print("Quick Mesh Size vs Performance Test")
    print("=" * 50)
    
    # Load actual reference meshes
    print("Loading actual reference meshes...")
    meshes = load_actual_reference_meshes()
    
    if not meshes:
        print("No meshes found! Check paths.")
        return
    
    # Test interpolation on each bone
    results = []
    for bone_name, mesh_data in meshes.items():
        result = quick_interpolation_test(bone_name, mesh_data, n_steps=20)
        if result:
            results.append(result)
    
    # Summary
    print(f"\nSUMMARY:")
    print("-" * 30)
    print(f"{'Bone':<8} {'Points':<8} {'Time(s)':<8} {'Ratio':<8}")
    print("-" * 30)
    
    if results:
        # Sort by time
        results.sort(key=lambda x: x['time'])
        fastest_time = results[0]['time']
        
        for result in results:
            ratio = result['time'] / fastest_time
            print(f"{result['bone']:<8} {result['n_points']:<8} {result['time']:<8.2f} {ratio:<8.1f}x")
        
        print(f"\nConclusion:")
        print(f"If mesh size is the main factor, timing should correlate with point count.")
        
        # Check correlation
        points = [r['n_points'] for r in results]
        times = [r['time'] for r in results]
        
        # Simple correlation check
        if len(results) >= 2:
            correlation = np.corrcoef(points, times)[0, 1]
            print(f"Correlation between points and time: {correlation:.3f}")
            if correlation > 0.8:
                print("Strong positive correlation - mesh size IS the main factor!")
            elif correlation > 0.5:
                print("Moderate correlation - mesh size is A factor, but not the only one")
            else:
                print("Weak correlation - mesh size is NOT the main factor")

if __name__ == "__main__":
    main()



