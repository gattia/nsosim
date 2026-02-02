# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`nsosim` is a library for creating personalized biomechanical knee models by fitting Neural Shape Models (NSM) to MRI segmentations and integrating them into OpenSim/COMAK simulations. It bridges raw imaging data → NSM-derived geometries → subject-specific OpenSim models.

## Development Commands

```bash
# Install
pip install -e .              # Editable install
make install-dev              # Same as above

# Dependencies
pip install -r requirements.txt       # Production deps
pip install -r requirements-dev.txt   # Dev deps (pytest, black, isort)

# Linting and formatting
make lint          # Check formatting (isort + black)
make autoformat    # Auto-format code

# Tests
pytest             # Runs all tests (skips train_test.py by default)

# Build
make build         # Build wheel to wheelhouse/
make clean         # Remove build artifacts
```

**Code style:** Black with 100 char line length, isort with black profile.

## Architecture

### Core Modules

| Module | Purpose |
|--------|---------|
| `nsm_fitting.py` | NSM fitting pipeline: align meshes → fit NSM → convert to OpenSim coords |
| `articular_surfaces.py` | Extract/refine cartilage contact surfaces, meniscus processing, patella optimization |
| `comak_osim_update.py` | Update OpenSim XML with subject-specific meshes, attachments, wrap surfaces |
| `osim_utils.py` | Low-level OpenSim XML manipulation via Python API |
| `wraps.py` | Wrap surface definitions (cylinders, ellipsoids) for muscle/ligament paths |
| `utils.py` | NSM model loading, mesh I/O, anatomical coordinate system (ACS) alignment utilities |

### Wrap Surface Fitting Submodule (`wrap_surface_fitting/`)

PyTorch-based optimization to adapt OpenSim wrap surfaces to new bone geometries using signed distance functions (SDF).

| File | Purpose |
|------|---------|
| `fitting.py` | `CylinderFitter`, `EllipsoidFitter` - PyTorch optimizers for shape fitting |
| `utils.py` | Data loading, SDF computation, point classification |
| `config.py` | Default configurations for Smith2019 bone model |
| `rotation_utils.py` | Quaternion/Euler angle conversions |
| `wrap_signed_distances.py` | SDF functions for cylinders and ellipsoids |

## Coordinate Systems

Two coordinate systems are used throughout:
- **NSM space**: Coordinate system used by Neural Shape Models during fitting
- **OpenSim space**: Target coordinate system for OpenSim models

Key conversion functions:
- `convert_OSIM_to_nsm()` / `convert_nsm_recon_to_OSIM()` - transform between spaces
- `nsm_recon_to_osim()` - reconstruct mesh in OpenSim coordinates

## Key Design Decisions

### Cylinder Quadrant Orientation (commit 629e02d)

Cylinders in OpenSim have a "quadrant" parameter controlling which side is active for muscle wrapping. The fitting code handles cylinders differently from ellipsoids:

- `construct_cylinder_basis()` uses a `reference_x_axis` (defaults to global +X) to ensure consistent orientation
- **Cylinders skip `enforce_sign_convention()`** to preserve quadrant semantics
- Ellipsoids still use `enforce_sign_convention()` as they don't have quadrant sensitivity

See `nsosim/wrap_surface_fitting/CLAUDE.md` for detailed explanation.

### SDF-Based Optimization

Wrap surface fitting uses:
- Signed distance functions to classify points as inside/outside wrap surfaces
- PCA initialization for faster convergence
- Squared-hinge margin loss for classification

## External Dependencies

- **NSM** (Neural Shape Model): Custom library at `github.com/gattia/nsm` - must be installed first
- **OpenSim**: Python bindings for biomechanical simulation
- **PyTorch 2.0**: Used in wrap surface fitting optimization
- **pymskt**: Medical Shape and Kinematics Toolkit
- **PyVista**: 3D mesh visualization and manipulation

## Typical Workflow

1. `align_knee_osim_fit_nsm()` - Fit NSM to subject's bones/cartilage
2. `nsm_recon_to_osim()` - Convert reconstructions to OpenSim coords
3. `create_articular_surfaces()` - Extract cartilage contact surfaces
4. `interp_ref_to_subject_to_osim()` - Map landmarks to subject geometry
5. Wrap surface functions from `wraps.py` - Define subject-specific wraps
6. `update_osim_model()` - Integrate into OpenSim XML model