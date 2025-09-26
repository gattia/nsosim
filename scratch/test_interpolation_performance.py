#!/usr/bin/env python3
"""
Test script to analyze NSM interpolation performance with different mesh sizes and parameters.
This script replicates the same setup as the main COMAK pipeline but focuses on interpolation timing.
"""

import os
import json
import time
import numpy as np
import pyvista as pv
from contextlib import contextmanager
from datetime import datetime

from pymskt.mesh import Mesh
from nsosim.nsm_fitting import interp_ref_to_subject_to_osim
from nsosim.utils import load_model

@contextmanager
def time_operation(operation_name):
    """Context manager to time operations"""
    print(f"Starting: {operation_name}")
    start_time = time.time()
    start_datetime = datetime.now()
    
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        print(f"Completed: {operation_name} in {duration:.2f}s ({duration/60:.2f}min)")

def load_nsm_model_and_config(bone_name):
    """Load NSM model and config for a specific bone using the same paths as main script"""
    
    BASE_FOLDER = '/dataNAS/people/aagatti/projects/comak_gait_simulation/COMAK_SIMULATION_REQUIREMENTS'
    folder_nsm_models = os.path.join(BASE_FOLDER, 'nsm_models')
    
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
    
    if bone_name not in model_configs:
        raise ValueError(f"Unknown bone: {bone_name}. Choose from {list(model_configs.keys())}")
    
    config = model_configs[bone_name]
    
    # Load model config
    with open(config['path_model_config'], 'r') as f:
        model_config = json.load(f)
    
    # Load model
    model = load_model(model_config, config['path_model_state'], model_type='triplanar')
    
    return model, model_config

def generate_random_latent_vectors(model_config, n_vectors=2):
    """Generate random latent vectors for testing"""
    latent_size = model_config['latent_size']
    
    # Generate random latent vectors with reasonable scale
    latent_vectors = []
    for i in range(n_vectors):
        # Use normal distribution with reasonable scale
        latent = np.random.normal(0, 0.5, latent_size).astype(np.float32)
        latent_vectors.append(latent)
    
    return latent_vectors

def create_test_mesh(n_points, bone_name="test"):
    """Create a test mesh with specified number of points"""
    # Create a simple sphere mesh and resample to desired point count
    sphere = pv.Sphere(radius=0.05)  # 5cm radius in meters
    
    if n_points != sphere.n_points:
        # Resample to get desired point count
        mesh = Mesh(sphere)
        mesh.resample_surface(subdivisions=1, clusters=n_points)
        test_mesh = mesh.mesh
    else:
        test_mesh = sphere
    
    print(f"Created test {bone_name} mesh with {test_mesh.n_points} points")
    return test_mesh

def setup_test_environment(bone_name):
    """Setup test environment with same folder structure as main script"""
    BASE_FOLDER = '/dataNAS/people/aagatti/projects/comak_gait_simulation/COMAK_SIMULATION_REQUIREMENTS'
    folder_ref_recons = os.path.join(BASE_FOLDER, 'nsm_meshes')
    
    # Get reference center (needed for coordinate conversion)
    fem_path_transform_file = os.path.join(folder_ref_recons, 'femur', 'ref_femur_alignment.json')
    with open(fem_path_transform_file, 'r') as f:
        dict_transforms = json.load(f)
        fem_ref_center = np.array(dict_transforms['mean_orig'])
    
    return folder_ref_recons, fem_ref_center

def test_interpolation_performance(
    bone_name='femur',
    mesh_sizes=[1000, 5000, 10000, 20000, 50000],
    n_steps_list=[10, 20, 50, 100],
    n_trials=3
):
    """
    Test interpolation performance with different mesh sizes and n_steps
    
    Args:
        bone_name (str): Which bone model to test ('femur', 'tibia', 'patella')
        mesh_sizes (list): List of mesh point counts to test
        n_steps_list (list): List of n_steps values to test
        n_trials (int): Number of trials to average over
    """
    
    print(f"Testing NSM interpolation performance for {bone_name}")
    print(f"Mesh sizes: {mesh_sizes}")
    print(f"N_steps: {n_steps_list}")
    print(f"Trials per test: {n_trials}")
    print("="*60)
    
    # Setup
    with time_operation("Loading NSM model and setup"):
        model, model_config = load_nsm_model_and_config(bone_name)
        folder_ref_recons, fem_ref_center = setup_test_environment(bone_name)
        
        # Generate random latent vectors for testing
        latent_vectors = generate_random_latent_vectors(model_config, n_vectors=2)
        latent1, latent2 = latent_vectors
        
        print(f"Model latent size: {model_config['latent_size']}")
        print(f"Model device: {next(model.parameters()).device}")
    
    # Create dict_bones structure (minimal version for testing)
    dict_bones = {
        bone_name: {
            'subject': {
                'recon_dict': {
                    'model': model,
                    'icp_transform': np.eye(4),  # Identity for testing
                    'scale': np.array([1.0]),
                    'center': np.array([0.0, 0.0, 0.0])
                },
                'recon_latent': latent2  # Target latent
            }
        }
    }
    
    results = []
    
    # Test different mesh sizes and n_steps combinations
    for mesh_size in mesh_sizes:
        for n_steps in n_steps_list:
            print(f"\nTesting {mesh_size} points, {n_steps} steps")
            
            # Create test mesh
            test_mesh = create_test_mesh(mesh_size, bone_name)
            
            trial_times = []
            
            for trial in range(n_trials):
                print(f"  Trial {trial + 1}/{n_trials}", end=" ")
                
                start_time = time.time()
                
                try:
                    # Run interpolation (this is what takes the time)
                    interpolated_pts = interp_ref_to_subject_to_osim(
                        ref_mesh=test_mesh,
                        surface_name=bone_name,
                        ref_center=fem_ref_center,
                        dict_bones=dict_bones,
                        folder_nsm_files=folder_ref_recons,
                        surface_idx=0,  # Just bone surface
                        n_steps=n_steps,
                    )
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    trial_times.append(duration)
                    
                    print(f"-> {duration:.2f}s")
                    
                except Exception as e:
                    print(f"-> FAILED: {e}")
                    trial_times.append(None)
            
            # Calculate statistics
            valid_times = [t for t in trial_times if t is not None]
            if valid_times:
                avg_time = np.mean(valid_times)
                std_time = np.std(valid_times)
                min_time = np.min(valid_times)
                max_time = np.max(valid_times)
                
                result = {
                    'bone': bone_name,
                    'mesh_size': mesh_size,
                    'n_steps': n_steps,
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'min_time': min_time,
                    'max_time': max_time,
                    'trials': len(valid_times),
                    'points_per_second': mesh_size * n_steps / avg_time if avg_time > 0 else 0
                }
                
                results.append(result)
                
                print(f"  Average: {avg_time:.2f}s ± {std_time:.2f}s")
                print(f"  Points/sec: {result['points_per_second']:.0f}")
            else:
                print(f"  All trials failed!")
    
    return results

def print_results_summary(results):
    """Print a summary of performance results"""
    print("\n" + "="*80)
    print("PERFORMANCE RESULTS SUMMARY")
    print("="*80)
    
    # Group by bone
    bones = list(set(r['bone'] for r in results))
    
    for bone in bones:
        bone_results = [r for r in results if r['bone'] == bone]
        print(f"\n{bone.upper()} RESULTS:")
        print("-" * 40)
        print(f"{'Mesh Size':<10} {'N Steps':<8} {'Time (s)':<10} {'Points/sec':<12}")
        print("-" * 40)
        
        for result in bone_results:
            print(f"{result['mesh_size']:<10} {result['n_steps']:<8} "
                  f"{result['avg_time']:<10.2f} {result['points_per_second']:<12.0f}")
    
    # Find patterns
    print(f"\nKEY FINDINGS:")
    print("-" * 20)
    
    # Effect of mesh size (fixed n_steps=100)
    steps_100_results = [r for r in results if r['n_steps'] == 100]
    if len(steps_100_results) >= 2:
        print("Effect of mesh size (n_steps=100):")
        for result in sorted(steps_100_results, key=lambda x: x['mesh_size']):
            print(f"  {result['mesh_size']:>6} points: {result['avg_time']:>6.2f}s")
    
    # Effect of n_steps (largest mesh size)
    if results:
        max_mesh_size = max(r['mesh_size'] for r in results)
        mesh_size_results = [r for r in results if r['mesh_size'] == max_mesh_size]
        if len(mesh_size_results) >= 2:
            print(f"\nEffect of n_steps ({max_mesh_size} points):")
            for result in sorted(mesh_size_results, key=lambda x: x['n_steps']):
                print(f"  {result['n_steps']:>3} steps: {result['avg_time']:>6.2f}s")

def save_results_to_file(results, filename="interpolation_performance_results.json"):
    """Save results to JSON file"""
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {filename}")

def main():
    """Main test function"""
    
    # Test parameters
    BONE_NAME = 'femur'  # Start with femur as it's the most complex
    MESH_SIZES = [1000, 5000, 10000, 20000]  # Start smaller to avoid memory issues
    N_STEPS_LIST = [10, 20, 50, 100]
    N_TRIALS = 2  # Reduce trials for faster testing
    
    print("NSM Interpolation Performance Test")
    print("=" * 50)
    print(f"Testing bone: {BONE_NAME}")
    print(f"Mesh sizes: {MESH_SIZES}")
    print(f"N_steps: {N_STEPS_LIST}")
    print(f"Trials: {N_TRIALS}")
    
    try:
        # Run performance test
        results = test_interpolation_performance(
            bone_name=BONE_NAME,
            mesh_sizes=MESH_SIZES,
            n_steps_list=N_STEPS_LIST,
            n_trials=N_TRIALS
        )
        
        # Print summary
        print_results_summary(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interpolation_performance_{BONE_NAME}_{timestamp}.json"
        save_results_to_file(results, filename)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



