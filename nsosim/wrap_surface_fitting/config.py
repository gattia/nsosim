"""
Configuration constants for wrap surface fitting module.

This module centralizes all default configurations that were previously
scattered throughout the preprocessing.py file, making them easy to maintain
and modify.

Configuration includes:
- DEFAULT_SMITH2019_BONES: Bone surface definitions for Smith2019 model
- DEFAULT_SMITH2019_THRESHOLDS: Near-surface distance thresholds per bone
- DEFAULT_FITTING_CONFIG: Default parameters for ellipsoid and cylinder fitting
"""

from typing import Any, Dict, Optional, Union

# Default bone configuration for smith2019 model
# This defines which bones to process, their surface filenames, and the wrap surfaces
# associated with each bone body in the OpenSim model
DEFAULT_SMITH2019_BONES = {
    "femur": {
        "surface_filename": "smith2019-R-femur-bone.stl",
        "wrap_surfaces": {
            "femur_r": {
                "ellipsoid": ["Gastroc_at_Condyles_r"],
                "cylinder": [
                    "KnExt_at_fem_r",
                    "KnExt_vasint_at_fem_r",
                ],
            },
            "femur_distal_r": {"cylinder": ["Capsule_r"]},
        },
    },
    "tibia": {
        "surface_filename": "smith2019-R-tibia-bone.stl",
        "wrap_surfaces": {
            "tibia_proximal_r": {
                "ellipsoid": [
                    "Med_Lig_r",
                    "Med_LigP_r",
                ]
            }
        },
    },
    "patella": {
        "surface_filename": "smith2019-R-patella-bone.stl",
        "wrap_surfaces": {"patella_r": {"ellipsoid": ["PatTen_r"]}},
    },
}

# Default bone thresholds for smith2019 model
# These define the distance threshold for classifying vertices as "near surface"
# to wrap surfaces. Smaller values create tighter boundaries around wrap surfaces.
DEFAULT_SMITH2019_THRESHOLDS = {
    "femur": 0.0005,
    "tibia": 0.0005,
    "patella": 0.001,
}

# Default fitting parameters for ellipsoid and cylinder surface fitting
# These parameters control the optimization process for fitting parametric surfaces
# to the labeled mesh vertices
DEFAULT_FITTING_CONFIG = {
    "ellipsoid": {
        # Constructor parameters
        "constructor": {
            "lr": 1e-2,  # Initial learning rate for Adam optimizer
            "epochs": 10,  # Number of Adam optimization epochs
            "use_lbfgs": True,  # Whether to use L-BFGS after Adam
            "lbfgs_epochs": 100,  # Number of L-BFGS epochs
            "lbfgs_lr": 1.0,  # Learning rate for L-BFGS
            "alpha": 1.0,  # SDF loss weight
            "beta": 0.0,  # Label loss weight
            "gamma": 0.0,  # Regularization loss weight
            "margin_decay_type": "linear",  # How to decay margin over epochs
            "initialization": "geometric",  # Initialization method (geometric/pca)
            "lr_schedule": None,  # Learning rate schedule
            # Anchor regularizers — pin flat directions of the loss landscape
            # to the anchor (Procrustes-from-Smith2019 when supplied, else the
            # algebraic init), without affecting non-flat directions. Tuned
            # down 20x from the iter3 defaults (1.0 / 0.1 / 0.1) once the
            # anchor became Smith2019-derived: the anchor itself is trusted,
            # so the regularizer only needs to gently bias flat directions —
            # the geometric loss handles everything else. See
            # EllipsoidFitter.__init__ docstring.
            "lambda_center_reg": 0.05,  # ~ 50 nm² at 1 mm offset; floor for flat Z dirs
            "lambda_axes_reg": 0.005,  # mild — prevents axis-drift "blame transfer"
            "lambda_quat_reg": 0.005,  # mild — pins rotation in flat directions
        },
        # fit() method parameters
        "fit": {
            "margin": 0.0002,  # Margin for SDF-based loss
            "plot": False,  # Whether to plot during fitting
        },
    },
    "cylinder": {
        # Constructor parameters
        "constructor": {
            "lr": 0.0,  # Initial learning rate (0 = skip Adam)
            "epochs": 0,  # Number of Adam epochs (0 = skip Adam)
            "use_lbfgs": True,  # Use L-BFGS optimizer
            "lbfgs_epochs": 100,  # Number of L-BFGS epochs
            "lbfgs_lr": 1.0,  # Learning rate for L-BFGS
            "alpha": 1.0,  # SDF loss weight
            "beta": 0.0,  # Label loss weight
            "gamma": 0.0,  # Regularization loss weight
            "margin_decay_type": None,  # No margin decay for cylinders
            "initialization": "geometric",  # Initialization method
            "lr_schedule": None,  # Learning rate schedule
            # Anchor regularizers — pin flat directions to the anchor
            # (Procrustes-from-Smith2019 when supplied, else algebraic init).
            # The cylinder's axial direction is often flat (sliding the
            # cylinder along its long axis doesn't change in/out classification
            # much). Tuned down 20x from the iter3 default (1.0) once the
            # anchor became Smith2019-derived.
            "lambda_center_reg": 0.05,  # in meters (linear center_transform)
            "lambda_axis_reg": 0.0,  # axis direction handled by initialization already
        },
        # fit() method parameters
        "fit": {
            "margin": 1e-10,  # Margin for SDF-based loss (very small)
            "plot": False,  # Whether to plot during fitting
        },
    },
}

# Model file naming conventions
DEFAULT_MODEL_FILES = {
    "osim_model_name": "smith2019.osim",
    "geometry_folder_name": "Geometry",
    "config_filename": "smith2019_config.json",
    "processing_summary_filename": "smith2019_processing_summary.json",
}
