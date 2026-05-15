"""Stage X: COMAK body scaling from AddBiomechanics outputs.

Public API: ``scale_comak_model``. Reads AB per-body scale factors from a
match_markers_and_physics.osim, builds a ScaleSet, runs ScaleTool, bakes the
subject-specific knee geometry into its STLs, transplants AB's MarkerSet, and
applies the ITB1 path-point fix-up.
"""

from .config import LONG_AXIS_INDEX, WA_KNEE_BODIES, ScalingMode
from .orchestrator import scale_comak_model

__all__ = [
    "LONG_AXIS_INDEX",
    "WA_KNEE_BODIES",
    "ScalingMode",
    "scale_comak_model",
]
