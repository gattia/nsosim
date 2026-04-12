"""
COMAK knee assembly: extract, strip, and add COMAK knee components.

Submodules:
    config  — Data classes (ComakKneeConfig, ComakBody, etc.) and JSON serialization
    strip   — strip_comak_knee()
    add     — add_comak_knee()
"""

from .add import add_comak_knee
from .config import (
    ComakBody,
    ComakContactForce,
    ComakContactMesh,
    ComakCoordinate,
    ComakCustomJoint,
    ComakKneeConfig,
    ComakLigament,
    ComakMuscle,
    ComakSpring,
    ComakWeldJoint,
    ComakWrapSurface,
)
from .strip import strip_comak_knee

__all__ = [
    "ComakBody",
    "ComakContactForce",
    "ComakContactMesh",
    "ComakCoordinate",
    "ComakCustomJoint",
    "ComakKneeConfig",
    "ComakLigament",
    "ComakMuscle",
    "ComakSpring",
    "ComakWeldJoint",
    "ComakWrapSurface",
    "add_comak_knee",
    "strip_comak_knee",
]
