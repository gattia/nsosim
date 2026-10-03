"""COMAK body scaling from AddBiomechanics outputs.

(Referred to as "Stage X" in the cross-repo COMAK pipeline naming.)

Public API: ``scale_comak_model``. Checks that AB's generic model is the COMAK
skeleton (``check_ab_generic``), reads AB per-body scale factors from a
match_markers_and_physics.osim, builds a ScaleSet, runs ScaleTool, bakes the
subject-specific knee geometry into its STLs, transplants AB's MarkerSet, and
applies the ITB1 path-point fix-up.
"""

from .config import LONG_AXIS_INDEX, WA_KNEE_BODIES, ScalingMode
from .orchestrator import scale_comak_model
from .validate import GenericSkeletonMismatchError, check_ab_generic, measure_skeleton

__all__ = [
    "LONG_AXIS_INDEX",
    "WA_KNEE_BODIES",
    "GenericSkeletonMismatchError",
    "ScalingMode",
    "check_ab_generic",
    "measure_skeleton",
    "scale_comak_model",
]
