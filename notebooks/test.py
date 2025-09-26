'''
conda activate comak
python

'''

from nsosim.utils import load_model
import os
import json
import numpy as np
import torch
from NSM.mesh import create_mesh
from NSM.mesh.interpolate import interpolate_points
import time

BASE_COMAK_SIMULATION_PARMS_FOLDER = '/dataNAS/people/aagatti/projects/comak_gait_simulation/COMAK_SIMULATION_REQUIREMENTS'
folder_nsm_models = os.path.join(BASE_COMAK_SIMULATION_PARMS_FOLDER, 'nsm_models')

path_model_state = os.path.join(folder_nsm_models, '568_nsm_femur_bone_cart_men_v0.0.1', 'model', '2000.pth')
path_model_config = os.path.join(folder_nsm_models, '568_nsm_femur_bone_cart_men_v0.0.1', 'model_params_config.json')

with open(path_model_config, 'r') as f:
    model_config = json.load(f)

model = load_model(model_config, path_model_state, model_type='triplanar')

# create 2 fake latents in 1024 dimensions that have norms of 6
ref_latent = np.random.randn(1024)
ref_latent = ref_latent / np.linalg.norm(ref_latent) * 6

latent2 = np.random.randn(1024)
latent2 = latent2 / np.linalg.norm(latent2) * 6


torch_vec = torch.tensor(ref_latent, dtype=torch.float).cuda()
mesh = create_mesh(
    model, 
    torch_vec, 
    objects=4
)


start_time = time.time()
interpolated_points = interpolate_points(
    model,
    ref_latent,
    latent2,
    n_steps=100,
    points1=mesh[0].point_coords,
    surface_idx=0,
    verbose=False,
    spherical=True
)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")