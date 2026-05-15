"""Constants for COMAK body scaling.

WA_KNEE_BODIES are the subject-specific knee bodies: they receive the isotropic
weighted-average factor, and all their geometry is baked into its STLs by
bake_knee_geometry (rather than scaled via OpenSim `scale_factors`).
"""

from typing import Literal, Tuple

ScalingMode = Literal["WA", "LA", "AB"]

LONG_AXIS_INDEX: int = 2

WA_KNEE_BODIES: Tuple[str, ...] = (
    "femur_distal_r",
    "tibia_proximal_r",
    "patella_r",
    "meniscus_medial_r",
    "meniscus_lateral_r",
)
