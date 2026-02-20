```bash
# create environment (requires Python >= 3.9)
conda create -n nsosim python=3.9
conda activate nsosim

# install NSM (not on PyPI — install from source)
mkdir dependencies
cd dependencies
git clone https://github.com/gattia/nsm.git
cd nsm
mamba install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
make requirements
pip install .

cd ../..

# install nsosim (dependencies declared in pyproject.toml)
pip install -e .

# or with wrap surface fitting support (includes torch):
pip install -e ".[fitting]"
```

### Manual Prerequisites

The following dependencies are **not** pip-installable and must be installed separately:

- **opensim** — requires the JAM/COMAK fork built from source. Standard conda `opensim` will not work.
- **NSM** — install from source: `pip install git+https://github.com/gattia/nsm.git`

## Project Status

This project is currently in **Alpha** stage. It is under active development, and APIs might change.

## Purpose / Motivation

The `nsosim` library serves a crucial role in the larger Gait Simulation Project by enabling the creation of subject-specific knee models. The primary goal of the overall project is to perform cartilage contact simulations of a human gait cycle using COMAK and OpenSim. `nsosim` bridges the gap between a subject's raw imaging data (e.g., MRI segmentations) and a personalized OpenSim knee model that incorporates Neural Shape Model (NSM) derived geometries. This personalized model is a key input for subsequent biomechanical simulations with COMAK, allowing for more accurate and patient-specific analyses of knee joint mechanics.

## Key Features

*   **Neural Shape Model (NSM) Fitting:** Leverages NSMs to capture subject-specific variations in bone and cartilage geometry from input mesh data (typically MRI segmentations).
*   **Subject-Specific OpenSim Model Generation:** Automates the process of integrating these personalized geometries into a COMAK-compatible OpenSim knee model.
*   **Anatomical Alignment & Registration:** Includes utilities for aligning bone meshes to anatomical coordinate systems (e.g., femur ACS) and registering them to template models, ensuring consistency.
*   **Articular Surface Processing:** Provides tools to extract and refine articular cartilage surfaces from the NSM-reconstructed meshes, critical for accurate contact modeling.
*   **OpenSim Model Component Management:** Facilitates updates to OpenSim model components, such as wrapping surfaces, to reflect subject-specific anatomy.
*   **Modular Toolkit:** Organized into distinct modules (`nsm_fitting.py`, `comak_osim_update.py`, `articular_surfaces.py`, `utils.py`, `wrap_surface_fitting/`) for a clear and maintainable workflow.

## Overview

The `nsosim` library is designed for creating and utilizing personalized biomechanical models, primarily by integrating Neural Shape Models (NSMs) with OpenSim.

The library is organized into the following main modules:

*   `articular_surfaces.py`: Provides tools for generating and extracting articular surfaces. This typically involves processing bone and cartilage meshes to define the contact geometry within joints.
*   `comak_osim_update.py`: Contains functions to modify and update OpenSim models. This can include adjusting model parameters based on subject-specific data or for compatibility with specific analyses like COMAK (Concurrent Optimization of Muscle Activations and Kinematics).
*   `nsm_fitting.py`: Implements the pipeline for fitting NSMs to subject-specific bone and cartilage data. This includes aligning raw scan data (e.g., using anatomical coordinate systems), registering it to template models, fitting the NSM to capture the subject's unique anatomy, and transforming these personalized models into the OpenSim coordinate framework.
*   `utils.py`: A collection of helper functions supporting the NSM fitting and model update processes. These utilities handle tasks such as loading NSM models, reconstructing mesh geometry from NSM latent codes, and performing anatomical alignments.
*   `wrap_surface_fitting/`: PyTorch-based optimization submodule for fitting OpenSim wrap surfaces (cylinders and ellipsoids) to subject-specific bone geometries using signed distance functions (SDF). Replaces the former `wraps.py` module.

The `__init__.py` file ensures all these modules are accessible under the `nsosim` package.

## Basic Workflow / Usage Examples

The typical workflow using the `nsosim` library involves the following conceptual steps to generate a subject-specific OpenSim knee model, as demonstrated in typical use cases:

1.  **Input Data Preparation & Configuration:**
    *   Define paths to the subject's bone and cartilage segmentations (e.g., `.vtk` files from MRI).
    *   Specify paths to pre-trained Neural Shape Models (NSMs) for each bone (femur, tibia, patella), including model state (`.pth`) and configuration (`.json`) files.
    *   Provide paths to reference/template mesh data and transformation files if needed.
    *   Configure output directories for saving intermediate and final results.

2.  **NSM Fitting and Personalized Geometry Generation (primarily using `nsm_fitting.py`):**
    *   Utilize a high-level function like `nsm_fitting.align_knee_osim_fit_nsm`. This function typically:
        *   Loads the subject's bone (e.g., femur, tibia, patella) and cartilage meshes.
        *   Loads the corresponding pre-trained NSM models (often facilitated by `utils.load_model` internally).
        *   Performs registration of the subject's meshes to a reference template. ACS alignment (e.g., `utils.acs_align_femur`) can be part of this process or handled based on parameters.
        *   Fits the NSM to the subject's data, generating subject-specific latent vectors.
        *   Reconstructs personalized bone and cartilage meshes based on these latent vectors.
        *   Saves the fitted geometries, latent vectors, and transformation parameters.
    *   Alternatively, individual functions like `utils.fit_nsm` or `utils.recon_mesh` might be used for more granular control.
    *   **Note:** `recon_mesh` infers extra output meshes (e.g., menisci, fibula) based on the count returned by the decoder, which depends on the model's `objects_per_decoder` setting. Different model variants for the same bone may produce different numbers of meshes (e.g., a tibia model may output 2 or 3 meshes depending on whether it includes a fibula). See `CLAUDE.md` Known Issues for details.

3.  **Subject-Specific Mesh Reconstruction in OpenSim Coordinates (using `nsm_fitting.py`):**
    *   For each bone (femur, tibia, patella), reconstruct the bone and cartilage meshes in the OpenSim coordinate system using `nsm_fitting.nsm_recon_to_osim`. This step uses the previously obtained subject-specific NSM parameters.
    *   Save these OpenSim-aligned meshes (e.g., as `.stl` for bones, `.vtk` for cartilage).

4.  **Articular Surface Creation and Refinement (using `articular_surfaces.py`):**
    *   Generate explicit articular cartilage surfaces from the reconstructed cartilage meshes using `articular_surfaces.create_articular_surfaces`. This is crucial for contact modeling.
    *   For the patella, its position may be optimized relative to the femur to ensure physiological contact using `articular_surfaces.optimize_patella_position`.
    *   Save the refined articular surface meshes (e.g., as `.stl` or `.vtk`).

5.  **Interpolation of Anatomical Landmark and Wrapping Points (using `nsm_fitting.py`):**
    *   Interpolate the locations of anatomical landmarks, muscle/ligament attachment points, and points defining wrapping surfaces from a reference model to the subject-specific bone geometries in the OpenSim coordinate system. This is typically done using `nsm_fitting.interp_ref_to_subject_to_osim`.

6.  **Definition of Wrapping Surfaces (using `wrap_surface_fitting/`):**
    *   Fit wrap surfaces to subject-specific bone geometries using SDF-based optimization. `CylinderFitter` and `EllipsoidFitter` (from `wrap_surface_fitting/fitting.py`) fit cylinders and ellipsoids to labeled bone mesh regions, producing `wrap_surface` objects with translation, rotation, and dimension parameters for OpenSim.

7.  **OpenSim Model Update and Finalization (using `comak_osim_update.py`):**
    *   Start with a generic or template OpenSim (`.osim`) knee model.
    *   Use `comak_osim_update.update_osim_model` to:
        *   Replace generic mesh file references with the paths to the subject-specific bone and articular surface meshes generated in earlier steps.
        *   Update the locations of muscle/ligament attachment points based on the interpolated landmarks.
        *   Incorporate the newly defined subject-specific wrapping surfaces.
        *   Adjust other model parameters as needed (e.g., patella kinematics based on optimization, inertia properties if calculated).
    *   Optionally, update ligament stiffness parameters using functions like `comak_osim_update.update_ligament_stiffness`.
    *   Save the final subject-specific OpenSim (`.osim`) model.
    *   Copy all generated geometry files (bones, articular surfaces) into the appropriate `Geometry` subfolder of the new OpenSim model directory.

8.  **Output:**
    *   The primary output is a subject-specific OpenSim (`.osim`) knee model, incorporating the NSM-derived bone and cartilage geometries, and updated OpenSim components (attachments, wrapping surfaces), ready for use in COMAK or other biomechanical simulations.
    *   Intermediate files such as transformed meshes, NSM latent vectors, transformation parameters, and individual geometry files are also typically saved.

*(Note: For detailed API usage and specific function calls, please refer to the docstrings within each Python module. The exact sequence and functions might vary based on the specific application, for example, cartilage thickness analysis as seen in `comak_3_cartilage_thickness_pipeline.py` might use `nsosim.utils.load_model` and `nsosim.nsm_fitting.convert_OSIM_to_nsm` for different transformation and analysis purposes outside of direct OpenSim model building.)*



## Basic Knee Pipeline Overview
Based on scripted used in Gatt AA. et al. for OARSI 2025: `comak_1_update_comak_knee_nsm_OAI_OARSI_Nov_13_2024.py`. The following outlines the current end-to-end workflow for generating a subject-specific OpenSim knee model using the `nsosim` library and its associated scripts. 


---

## Example: Full Subject-Specific Model Generation Pipeline


### **Step-by-Step Pipeline**

1. **Setup and Configuration**
    - Parse subject ID, side, and timepoint from command line arguments.
    - Set up all necessary folder paths for subject data, reference data, NSM models, and output directories.
    - Define a `dict_bones` dictionary specifying reference and subject mesh files and NSM model paths for femur, tibia, and patella.

2. **NSM Fitting**
    - Use `align_knee_osim_fit_nsm` to fit NSM models to the subject’s femur, tibia, and patella (bone and cartilage).
    - This step aligns the subject meshes to the reference, fits the NSM, and saves the resulting latent vectors and reconstructed meshes.

3. **Load Reference Points and Attachments**
    - Load JSON files containing definitions for ligament and muscle attachment points, as well as landmark points for wrap surfaces.
    - Build a dictionary (`pts_dict`) mapping wrap surface names to their indices on the reference mesh.

4. **Femur Processing**
    - Convert the NSM-reconstructed femur and cartilage meshes to OpenSim coordinates using `nsm_recon_to_osim`.
    - Extract the femoral articular surface using `create_articular_surfaces`.
    - Interpolate reference wrap surface points to the subject femur using `interp_ref_to_subject_to_osim`.
    - Fit subject-specific femur wrap surfaces using `CylinderFitter` and `EllipsoidFitter` from `wrap_surface_fitting/`.

5. **Tibia Processing**
    - Repeat the above steps for the tibia: NSM reconstruction, coordinate conversion, articular surface extraction, wrap point interpolation, and wrap surface definition.

6. **Patella Processing**
    - Repeat the above steps for the patella, including optional optimization of patella position for physiological contact with the femur.

7. **Update OpenSim Model**
    - Copy a base OpenSim model to a new subject-specific directory.
    - Use `update_osim_model` to update the OpenSim XML with:
        - Subject-specific mesh file references
        - Updated muscle and ligament attachment points
        - Subject-specific wrap surface definitions
        - Patella position and other relevant parameters
    - Optionally update ligament stiffness values.
    - Copy all generated geometry files into the model’s `Geometry` folder.

8. **Error Handling**
    - If any error occurs during processing, log the error and continue.

9. **Output**
    - The final output is a subject-specific OpenSim (`.osim`) model, with all geometry and parameters updated to reflect the subject’s anatomy, ready for use in biomechanical simulation workflows.

---

**This workflow ensures that each subject’s OpenSim model is anatomically accurate, with all relevant surfaces, attachments, and parameters derived from their own imaging data and processed through the NSM and wrap surface fitting pipelines.**








## License

This project is licensed under the MIT License. See the `LICENSE` file for details (if one exists, or state the terms directly).
