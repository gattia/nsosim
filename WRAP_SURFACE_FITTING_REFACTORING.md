# Wrap Surface Fitting Code Refactoring Plan

## Background and Purpose

This document outlines the refactoring plan for the wrap surface fitting functionality that enables automatic adaptation of OpenSim wrap surfaces to new bone geometries. The complete workflow involves:

1. **Data Preparation**: Loading bone meshes and extracting wrap surface parameters from OpenSim XML files
2. **SDF Extraction**: Computing signed distance function (SDF) values for each bone point relative to existing wrap surfaces
3. **Surface Classification**: Classifying each point on a bone's surface as inside or outside of existing wrap surfaces based on SDF values
4. **Shape Optimization**: Using PyTorch to optimize wrap surface parameters (ellipsoids, cylinders) to match the classified point cloud
5. **Visualization and Integration**: Creating PyVista meshes for visualization and integrating back into OpenSim models

### Core Methodology

The approach uses signed distance functions (SDFs) and PyTorch optimization to fit parametric shapes:
- **SDF Computation**: Calculate distance from bone points to existing wrap surfaces (ellipsoids, cylinders)
- **Point Classification**: Use SDF values to determine inside/outside labels for each bone point
- **Shape Fitting**: Optimize new wrap surface parameters using classified points
- **Ellipsoids**: Fit using normalized SDF with three semi-axes and orientation
- **Cylinders**: Fit using finite cylinder SDF with radius, height, and orientation
- **Initialization**: Smart PCA-based parameter initialization for faster convergence
- **Loss Function**: Squared-hinge margin loss to enforce inside/outside classification

## Current Implementation Status

### Testing and Development Files

**Primary Development Notebook:**
- `COMAK_OAI_2025/notebooks/load_comak_bones_create_wraps_visualize_May.30.2025.ipynb`
  - Contains working prototype implementations
  - Includes cylinder and ellipsoid fitting examples
  - Demonstrates XML parameter extraction and visualization
  - Shows SDF computation and point classification workflow

**Current Consolidated Script:**
- `COMAK_OAI_2025/notebooks/sdf_fitting_functions_May.30.2025.py`
  - Consolidated functions from notebook testing
  - Contains all core functionality but with significant code duplication
  - Ready for refactoring into modular structure

### Identified Issues in Current Code

1. **Code Duplication:**
   - Quaternion conversion functions implemented multiple times (`quat_from_rot()`, `rot_to_quat()`)
   - Rotation matrix from quaternion conversion duplicated
   - Import statements scattered and duplicated throughout file

2. **Structural Issues:**
   - No clear separation between shape-specific and common functionality
   - Optimization logic duplicated between cylinder and ellipsoid fitters
   - XML parsing mixed with fitting logic
   - SDF computation and point classification not modularized

3. **Maintainability Concerns:**
   - Hardcoded constants (e.g., `ADDITIONAL_OFFSETS`)
   - Limited error handling and validation
   - Inconsistent parameter naming and return formats
   - No unified interface for different mesh formats and data sources

## Proposed Refactoring Plan

### 1. Modular Architecture

Create a clean, practical architecture with just 4 focused files:

```
nsosim/wrap_surface_fitting/
├── __init__.py
├── utils.py           # Data loading, SDF computation, point classification + XML parsing
├── fitting.py         # BaseShapeFitter + CylinderFitter + EllipsoidFitter (all fitting code)
├── visualizations.py  # PyVista mesh creation and visualization utilities
└── main.py           # End-to-end pipeline functions
```

### 2. Core Components Design

#### A. Data Utilities (`utils.py`)
```python
import numpy as np
from pymskt.mesh import Mesh

# Data loading and processing
def load_bone_mesh(file_path: str) -> tuple[Mesh, np.ndarray]:
    """Load bone mesh and extract points."""
    mesh = Mesh(file_path)
    points = np.asarray(mesh.point_coords.copy())
    return mesh, points

def compute_sdf_values(points: np.ndarray, wrap_surfaces: dict) -> dict[str, np.ndarray]:
    """Compute SDF values for points relative to all wrap surfaces."""
    sdf_values = {}
    for surface_name, surface_obj in wrap_surfaces.items():
        sdf_values[surface_name] = surface_obj.get_sdf_pts(points)
    return sdf_values

def classify_points(sdf_values: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Convert SDF values to binary inside/outside labels."""
    return (sdf_values < threshold).astype(int)

# OpenSim XML parsing
ADDITIONAL_OFFSETS = {
    'femur_r': [-0.0055513564376633642, -0.37418143637169787, -0.0011706232813375212]
}

class OpenSimWrapParser:
    def extract_wrap_parameters(self, xml_path: str, bone_dict: dict) -> dict:
        """Extract wrap surface parameters from OpenSim XML file."""
        # Keep existing parameter extraction logic
        pass
    
    def create_wrap_surfaces(self, xml_path: str, bone_dict: dict) -> dict:
        """Create wrap surface objects that can compute SDFs."""
        # Extract parameters and create surface objects
        pass

def prepare_fitting_data(bone_mesh_path: str, xml_path: str, bone_dict: dict) -> tuple[np.ndarray, dict]:
    """Complete data preparation pipeline."""
    mesh, points = load_bone_mesh(bone_mesh_path)
    
    parser = OpenSimWrapParser()
    wrap_surfaces = parser.create_wrap_surfaces(xml_path, bone_dict)
    
    sdf_values = compute_sdf_values(points, wrap_surfaces)
    
    classifications = {}
    for surface_name, sdf_vals in sdf_values.items():
        classifications[surface_name] = classify_points(sdf_vals)
    
    return points, classifications
```

#### B. All Fitting Code (`fitting.py`)
```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class RotationUtils:
    """Quaternion and rotation utilities."""
    
    @staticmethod
    def quat_from_rot(R: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to quaternion."""
        # Single implementation here
        pass
        
    @staticmethod
    def rot_from_quat(q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to rotation matrix."""
        # Single implementation here
        pass

class BaseShapeFitter(ABC):
    """Base class for shape fitting with common optimization logic."""
    
    def __init__(self, lr=5e-3, epochs=1500, device='cpu'):
        self.lr = lr
        self.epochs = epochs
        self.device = device
        
    def _prepare_data(self, points, labels):
        """Convert to tensors and validate."""
        x = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(labels > 0, dtype=torch.bool, device=self.device)
        return x, y
        
    def _optimization_loop(self, x, y, parameters, compute_sdf_fn, margin, plot=False):
        """Common optimization loop."""
        # Same as before - optimization logic
        pass
    
    def _compute_loss(self, d, y, margin):
        """Squared-hinge margin loss."""
        # Same as before - loss computation
        pass
    
    @abstractmethod
    def fit(self, points, labels, **kwargs):
        """Fit shape to classified points."""
        pass

class CylinderFitter(BaseShapeFitter):
    """Cylinder fitting using PCA initialization and PyTorch optimization."""
    
    def _pca_initialization(self, points_inside):
        """Cylinder-specific PCA: align principal axis with +z, estimate params in local coords."""
        # Your exact working PCA logic here
        pass
    
    def fit(self, points, labels, margin=0.05, plot=False):
        """Fit cylinder to classified points."""
        # Your exact working fitting logic here
        pass

class EllipsoidFitter(BaseShapeFitter):
    """Ellipsoid fitting using PCA initialization and PyTorch optimization."""
    
    def _pca_initialization(self, points_inside):
        """Ellipsoid-specific PCA: all 3 components become ellipsoid semi-axes."""
        # PCA logic for ellipsoid
        pass
    
    def fit(self, points, labels, margin=0.002, plot=False):
        """Fit ellipsoid to classified points."""
        # Ellipsoid fitting logic
        pass

# SDF functions can also go here since they're fitting-related
def sd_cylinder(points, center, radius, half_length, quat):
    """Signed distance function for finite cylinder."""
    # Your SDF implementation
    pass

def sd_normalised(points, center, axes, quat):
    """Signed distance function for ellipsoid."""
    # Your SDF implementation  
    pass
```

#### C. Simple Workflows (`main.py`)
```python
from .utils import prepare_fitting_data
from .fitting import CylinderFitter, EllipsoidFitter

def fit_wrap_surfaces(bone_mesh_path: str, xml_path: str, bone_dict: dict, output_dir: str = None):
    """Complete end-to-end pipeline."""
    
    # Prepare data
    points, classifications = prepare_fitting_data(bone_mesh_path, xml_path, bone_dict)
    
    # Fit surfaces
    fitted_surfaces = {}
    cylinder_fitter = CylinderFitter()
    ellipsoid_fitter = EllipsoidFitter()
    
    for surface_name, labels in classifications.items():
        surface_type = get_surface_type(surface_name, bone_dict)
        
        if 'cylinder' in surface_type.lower():
            center, params, rotation = cylinder_fitter.fit(points, labels, plot=True)
            fitted_surfaces[surface_name] = {
                'type': 'cylinder',
                'center': center,
                'radius': params[0],
                'half_length': params[1],
                'rotation': rotation
            }
        elif 'ellipsoid' in surface_type.lower():
            center, axes, rotation = ellipsoid_fitter.fit(points, labels, plot=True)
            fitted_surfaces[surface_name] = {
                'type': 'ellipsoid',
                'center': center,
                'axes': axes,
                'rotation': rotation
            }
    
    return fitted_surfaces
```

### 3. Key Improvements

#### Simplicity
- **Data preparation**: Just a few simple functions instead of complex classes
- **Focused modules**: Each file has a clear, single purpose
- **Practical structure**: Groups related functionality together

#### Maintainability  
- **Elimination of over-engineering**: No unnecessary abstractions
- **Clear separation**: Fitting logic grouped together, data prep is simple utilities
- **Easy to extend**: Simple to add new shape types or modify existing ones

### 4. Usage Example (Target API)

```python
from nsosim.wrap_surface_fitting import fit_wrap_surfaces
from nsosim.wrap_surface_fitting.utils import prepare_fitting_data
from nsosim.wrap_surface_fitting.fitting import CylinderFitter, EllipsoidFitter

# Option 1: Simple end-to-end
fitted_surfaces = fit_wrap_surfaces(
    bone_mesh_path='path/to/bone.stl',
    xml_path='path/to/model.osim',
    bone_dict={'femur': {'wrap_surfaces': {...}}}
)

# Option 2: Step-by-step (if needed)
points, classifications = prepare_fitting_data('bone.stl', 'model.osim', bone_dict)

cylinder_fitter = CylinderFitter(epochs=1000)
center, params, rotation = cylinder_fitter.fit(
    points=points, 
    labels=classifications['surface_name'],
    margin=0.05,
    plot=True
)
```

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create module directory structure in `nsosim/wrap_surface_fitting/`
- [ ] Implement `RotationUtils` class with unified quaternion functions
- [ ] Create `BaseShapeFitter` abstract class with common optimization logic
- [ ] Implement signed distance functions in `sdf_functions.py`
- [ ] Set up configuration management in `config/defaults.py`
- [ ] Create comprehensive unit tests for core utilities

### Phase 2: Data Preparation Components
- [ ] Implement `MeshLoader` class for bone mesh loading and processing
- [ ] Create `SDFExtractor` class for computing SDF values relative to wrap surfaces
- [ ] Implement `PointClassifier` class for inside/outside point classification
- [ ] Add support for multiple mesh formats (STL, PLY, VTK, etc.)
- [ ] Create visualization utilities for SDF values and point classifications
- [ ] Test data preparation pipeline with existing notebook examples

### Phase 3: Shape-Specific Fitters
- [ ] Implement `CylinderFitter` class extending `BaseShapeFitter`
- [ ] Implement `EllipsoidFitter` class extending `BaseShapeFitter`
- [ ] Migrate and test cylinder fitting logic from existing script
- [ ] Migrate and test ellipsoid fitting logic from existing script
- [ ] Add mesh creation methods to both fitters
- [ ] Validate against existing notebook results

### Phase 4: OpenSim Integration
- [ ] Implement `OpenSimWrapParser` class for XML parsing
- [ ] Create clean interface for wrap parameter extraction
- [ ] Add parameter validation and error handling
- [ ] Implement mesh creation from OpenSim parameters
- [ ] Test against existing XML files and bone dictionaries

### Phase 5: End-to-End Workflows and API
- [ ] Implement `WrapSurfaceFittingPipeline` class for complete workflows
- [ ] Create unified mesh creation utilities in `visualizations.py`
- [ ] Design and implement high-level API for common workflows
- [ ] Add batch processing capabilities for multiple bones/anatomies
- [ ] Create convenience functions for different use cases

### Phase 6: Testing and Documentation
- [ ] Create comprehensive test suite covering all components
- [ ] Add integration tests using existing notebook examples
- [ ] Test complete pipeline with various mesh formats and anatomies
- [ ] Write detailed API documentation with mathematical background
- [ ] Create usage examples and tutorials for different workflows
- [ ] Performance benchmarking against current implementation

### Phase 7: Migration and Cleanup
- [ ] Update `COMAK_OAI_2025` notebooks to use new API
- [ ] Create migration guide from old to new API
- [ ] Archive old script files with deprecation notices
- [ ] Add new functionality to `nsosim` package exports
- [ ] Update package documentation and examples

### Validation Criteria
- [ ] Complete pipeline works from bone mesh + XML to fitted wrap surfaces
- [ ] All existing notebook functionality works with new API
- [ ] Performance is equivalent or better than current implementation
- [ ] Support for multiple mesh formats and anatomical structures
- [ ] Code coverage > 90% for core functionality
- [ ] All existing test cases pass
- [ ] New API is more intuitive and maintainable than current script
- [ ] Documentation is comprehensive and includes usage examples
- [ ] Easy adaptation to new anatomies and wrap surface types
``` 
