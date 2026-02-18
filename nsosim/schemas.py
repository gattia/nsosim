"""
Validation functions for nsosim pipeline data structures.

Validates dict_bones, fitted_wrap_parameters, and surface_idx before
pipeline entry to catch structural errors early.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when a pipeline data structure fails validation."""

    pass


# ── dict_bones validation ───────────────────────────────────────────────────

# Required sections and their required keys for each bone in dict_bones
_REQUIRED_SECTIONS = {
    "subject": {"folder", "bone_filename", "cart_filename"},
    "ref": {"folder", "bone_filename"},
    "model": {"path_model_state", "path_model_config"},
}

# Additional required keys under 'subject' for specific bones
_BONE_SPECIFIC_SUBJECT_KEYS = {
    "femur": {"med_men_filename", "lat_men_filename"},
}

# Valid wrap surface types
_VALID_WRAP_TYPES = {"cylinder", "ellipsoid"}


def validate_dict_bones(dict_bones, require_bones=None):
    """
    Validate the dict_bones structure before pipeline entry.

    Checks that required bones exist and each has the expected nested
    structure (subject/ref/model sections with required subkeys).

    Args:
        dict_bones: Dictionary mapping bone names to configuration dicts.
        require_bones: Optional list of bone names that must be present.
            Defaults to None (any non-empty dict is accepted).

    Raises:
        ValidationError: If the structure is invalid, with a message
            describing exactly which key is missing or malformed.
    """
    if not isinstance(dict_bones, dict):
        raise ValidationError(f"dict_bones must be a dict, got {type(dict_bones).__name__}")

    if not dict_bones:
        raise ValidationError("dict_bones is empty")

    if require_bones is not None:
        missing = set(require_bones) - set(dict_bones.keys())
        if missing:
            raise ValidationError(f"Missing required bones: {sorted(missing)}")

    for bone_name, bone_dict in dict_bones.items():
        if bone_name == "meniscus":
            _validate_meniscus_entry(bone_dict)
            continue

        if not isinstance(bone_dict, dict):
            raise ValidationError(
                f"dict_bones['{bone_name}'] must be a dict, got {type(bone_dict).__name__}"
            )

        for section, required_keys in _REQUIRED_SECTIONS.items():
            if section not in bone_dict:
                raise ValidationError(
                    f"dict_bones['{bone_name}'] missing required section '{section}'"
                )
            if not isinstance(bone_dict[section], dict):
                raise ValidationError(
                    f"dict_bones['{bone_name}']['{section}'] must be a dict, "
                    f"got {type(bone_dict[section]).__name__}"
                )
            missing_keys = required_keys - set(bone_dict[section].keys())
            if missing_keys:
                raise ValidationError(
                    f"dict_bones['{bone_name}']['{section}'] missing keys: {sorted(missing_keys)}"
                )

        # Bone-specific required keys
        if bone_name in _BONE_SPECIFIC_SUBJECT_KEYS:
            extra_required = _BONE_SPECIFIC_SUBJECT_KEYS[bone_name]
            missing = extra_required - set(bone_dict["subject"].keys())
            if missing:
                raise ValidationError(
                    f"dict_bones['{bone_name}']['subject'] missing "
                    f"bone-specific keys: {sorted(missing)}"
                )


def _validate_meniscus_entry(bone_dict):
    """Validate the meniscus entry (ref-only structure)."""
    if not isinstance(bone_dict, dict):
        raise ValidationError(
            f"dict_bones['meniscus'] must be a dict, got {type(bone_dict).__name__}"
        )
    if "ref" not in bone_dict:
        raise ValidationError("dict_bones['meniscus'] missing required section 'ref'")


# ── fitted_wrap_parameters validation ────────────────────────────────────────


def validate_fitted_wrap_parameters(dict_wrap_objects):
    """
    Validate the fitted_wrap_parameters structure before OpenSim model update.

    Checks the nested dict: bone -> body -> wrap_type -> wrap_name -> wrap_surface.

    Args:
        dict_wrap_objects: Nested dict of fitted wrap surface parameters.

    Raises:
        ValidationError: If the structure is invalid.
    """
    if not isinstance(dict_wrap_objects, dict):
        raise ValidationError(
            f"dict_wrap_objects must be a dict, got {type(dict_wrap_objects).__name__}"
        )

    for bone_name, bone_dict in dict_wrap_objects.items():
        if not isinstance(bone_dict, dict):
            raise ValidationError(
                f"dict_wrap_objects['{bone_name}'] must be a dict, "
                f"got {type(bone_dict).__name__}"
            )

        for body_name, body_dict in bone_dict.items():
            if not isinstance(body_dict, dict):
                raise ValidationError(
                    f"dict_wrap_objects['{bone_name}']['{body_name}'] must be a dict, "
                    f"got {type(body_dict).__name__}"
                )

            for wrap_type, wrap_dicts in body_dict.items():
                if wrap_type not in _VALID_WRAP_TYPES:
                    raise ValidationError(
                        f"Invalid wrap type '{wrap_type}' in "
                        f"dict_wrap_objects['{bone_name}']['{body_name}']. "
                        f"Must be one of: {sorted(_VALID_WRAP_TYPES)}"
                    )

                if not isinstance(wrap_dicts, dict):
                    raise ValidationError(
                        f"dict_wrap_objects['{bone_name}']['{body_name}']"
                        f"['{wrap_type}'] must be a dict, "
                        f"got {type(wrap_dicts).__name__}"
                    )

                for wrap_name, wrap_params in wrap_dicts.items():
                    _validate_single_wrap_surface(
                        wrap_params,
                        wrap_type,
                        f"dict_wrap_objects['{bone_name}']['{body_name}']"
                        f"['{wrap_type}']['{wrap_name}']",
                    )


def _validate_single_wrap_surface(wrap_params, wrap_type, path):
    """Validate a single wrap_surface object has the required attributes."""
    for attr in ("translation", "xyz_body_rotation"):
        if not hasattr(wrap_params, attr):
            raise ValidationError(f"{path} missing required attribute '{attr}'")
        if getattr(wrap_params, attr) is None:
            raise ValidationError(f"{path}.{attr} is None")

    if wrap_type == "cylinder":
        for attr in ("radius", "length"):
            if not hasattr(wrap_params, attr):
                raise ValidationError(f"{path} missing required attribute '{attr}'")
            val = getattr(wrap_params, attr)
            if val is None:
                raise ValidationError(f"{path}.{attr} is None")
            if not isinstance(val, (int, float, np.floating)):
                raise ValidationError(
                    f"{path}.{attr} must be numeric, got {type(val).__name__}"
                )

    elif wrap_type == "ellipsoid":
        if not hasattr(wrap_params, "dimensions"):
            raise ValidationError(f"{path} missing required attribute 'dimensions'")
        if wrap_params.dimensions is None:
            raise ValidationError(f"{path}.dimensions is None")


# ── surface_idx validation ──────────────────────────────────────────────────


def validate_surface_idx(surface_idx, max_idx=3):
    """
    Validate surface_idx is in valid range for NSM models.

    Surface indices: 0=bone, 1=cartilage, 2=med_meniscus, 3=lat_meniscus.

    Args:
        surface_idx: The surface index to validate.
        max_idx: Maximum valid index (inclusive). Defaults to 3.

    Raises:
        ValidationError: If surface_idx is out of range or wrong type.
    """
    if not isinstance(surface_idx, int):
        raise ValidationError(
            f"surface_idx must be an int, got {type(surface_idx).__name__}"
        )
    if surface_idx < 0 or surface_idx > max_idx:
        raise ValidationError(f"surface_idx={surface_idx} out of range [0, {max_idx}]")
