"""Tests for nsosim.schemas validation functions."""

import numpy as np
import pytest

from nsosim.schemas import (
    ValidationError,
    validate_dict_bones,
    validate_fitted_wrap_parameters,
    validate_surface_idx,
)
from nsosim.wrap_surface_fitting.main import wrap_surface

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_bone_dict(bone_name="femur"):
    """Build a minimal valid bone dict for a given bone type."""
    d = {
        "subject": {
            "folder": "/tmp/subject",
            "bone_filename": f"{bone_name}.vtk",
            "cart_filename": f"{bone_name}_cart.vtk",
        },
        "ref": {
            "folder": "/tmp/ref",
            "bone_filename": f"ref_{bone_name}.vtk",
        },
        "model": {
            "path_model_state": f"/tmp/models/{bone_name}.pth",
            "path_model_config": f"/tmp/models/{bone_name}_config.json",
        },
    }
    if bone_name == "femur":
        d["subject"]["med_men_filename"] = "med_men.vtk"
        d["subject"]["lat_men_filename"] = "lat_men.vtk"
    return d


def _make_wrap_surface(wrap_type="cylinder"):
    """Build a valid wrap_surface instance."""
    if wrap_type == "cylinder":
        return wrap_surface(
            name="test_cyl",
            body="femur_r",
            type_="WrapCylinder",
            xyz_body_rotation=np.array([0.0, 0.0, 0.0]),
            translation=np.array([0.01, 0.02, 0.03]),
            radius=0.005,
            length=0.04,
            dimensions=None,
        )
    else:
        return wrap_surface(
            name="test_ell",
            body="femur_r",
            type_="WrapEllipsoid",
            xyz_body_rotation=np.array([0.1, 0.2, 0.3]),
            translation=np.array([0.01, 0.02, 0.03]),
            radius=None,
            length=None,
            dimensions=np.array([0.01, 0.02, 0.03]),
        )


# ===========================================================================
# validate_dict_bones
# ===========================================================================


class TestValidateDictBones:
    def test_valid_full_dict_bones(self):
        """Valid dict_bones with femur, tibia, patella passes without error."""
        db = {
            "femur": _minimal_bone_dict("femur"),
            "tibia": _minimal_bone_dict("tibia"),
            "patella": _minimal_bone_dict("patella"),
        }
        validate_dict_bones(db)  # should not raise

    def test_valid_with_meniscus_entry(self):
        """Meniscus entry (ref-only) is accepted."""
        db = {
            "femur": _minimal_bone_dict("femur"),
            "meniscus": {"ref": {"folder": "/tmp/ref", "med_men_filename": "m.vtk"}},
        }
        validate_dict_bones(db)

    def test_empty_dict_raises(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_dict_bones({})

    def test_not_a_dict_raises(self):
        with pytest.raises(ValidationError, match="must be a dict"):
            validate_dict_bones([1, 2, 3])

    def test_missing_required_bone(self):
        db = {"femur": _minimal_bone_dict("femur")}
        with pytest.raises(ValidationError, match="Missing required bones.*tibia"):
            validate_dict_bones(db, require_bones=["femur", "tibia"])

    def test_missing_subject_section(self):
        db = {"tibia": {"ref": {}, "model": {}}}
        with pytest.raises(ValidationError, match="missing required section 'subject'"):
            validate_dict_bones(db)

    def test_missing_ref_section(self):
        bone = _minimal_bone_dict("tibia")
        del bone["ref"]
        with pytest.raises(ValidationError, match="missing required section 'ref'"):
            validate_dict_bones({"tibia": bone})

    def test_missing_model_section(self):
        bone = _minimal_bone_dict("tibia")
        del bone["model"]
        with pytest.raises(ValidationError, match="missing required section 'model'"):
            validate_dict_bones({"tibia": bone})

    def test_missing_subject_key(self):
        bone = _minimal_bone_dict("tibia")
        del bone["subject"]["bone_filename"]
        with pytest.raises(ValidationError, match="missing keys.*bone_filename"):
            validate_dict_bones({"tibia": bone})

    def test_femur_missing_meniscus_keys(self):
        bone = _minimal_bone_dict("femur")
        del bone["subject"]["med_men_filename"]
        with pytest.raises(ValidationError, match="bone-specific keys.*med_men_filename"):
            validate_dict_bones({"femur": bone})

    def test_meniscus_entry_missing_ref(self):
        db = {
            "femur": _minimal_bone_dict("femur"),
            "meniscus": {"something_else": {}},
        }
        with pytest.raises(ValidationError, match="missing required section 'ref'"):
            validate_dict_bones(db)

    def test_bone_dict_not_a_dict(self):
        with pytest.raises(ValidationError, match="must be a dict"):
            validate_dict_bones({"femur": "not_a_dict"})

    def test_section_not_a_dict(self):
        bone = _minimal_bone_dict("tibia")
        bone["subject"] = "bad"
        with pytest.raises(ValidationError, match="must be a dict"):
            validate_dict_bones({"tibia": bone})


# ===========================================================================
# validate_fitted_wrap_parameters
# ===========================================================================


class TestValidateFittedWrapParameters:
    def test_valid_wrap_parameters(self):
        """Valid nested structure passes without error."""
        params = {
            "femur": {
                "femur_r": {
                    "cylinder": {"KnExt_at_fem_r": _make_wrap_surface("cylinder")},
                    "ellipsoid": {"Gastroc_at_Condyles_r": _make_wrap_surface("ellipsoid")},
                },
            },
        }
        validate_fitted_wrap_parameters(params)

    def test_not_a_dict_raises(self):
        with pytest.raises(ValidationError, match="must be a dict"):
            validate_fitted_wrap_parameters("bad")

    def test_invalid_wrap_type(self):
        params = {
            "femur": {
                "femur_r": {
                    "sphere": {"bad": _make_wrap_surface("cylinder")},
                },
            },
        }
        with pytest.raises(ValidationError, match="Invalid wrap type 'sphere'"):
            validate_fitted_wrap_parameters(params)

    def test_cylinder_missing_radius(self):
        ws = _make_wrap_surface("cylinder")
        ws.radius = None
        params = {"femur": {"femur_r": {"cylinder": {"test": ws}}}}
        with pytest.raises(ValidationError, match="radius is None"):
            validate_fitted_wrap_parameters(params)

    def test_cylinder_missing_length(self):
        ws = _make_wrap_surface("cylinder")
        ws.length = None
        params = {"femur": {"femur_r": {"cylinder": {"test": ws}}}}
        with pytest.raises(ValidationError, match="length is None"):
            validate_fitted_wrap_parameters(params)

    def test_ellipsoid_missing_dimensions(self):
        ws = _make_wrap_surface("ellipsoid")
        ws.dimensions = None
        params = {"femur": {"femur_r": {"ellipsoid": {"test": ws}}}}
        with pytest.raises(ValidationError, match="dimensions is None"):
            validate_fitted_wrap_parameters(params)

    def test_missing_translation(self):
        ws = _make_wrap_surface("cylinder")
        ws.translation = None
        params = {"femur": {"femur_r": {"cylinder": {"test": ws}}}}
        with pytest.raises(ValidationError, match="translation is None"):
            validate_fitted_wrap_parameters(params)

    def test_missing_xyz_body_rotation(self):
        ws = _make_wrap_surface("ellipsoid")
        ws.xyz_body_rotation = None
        params = {"femur": {"femur_r": {"ellipsoid": {"test": ws}}}}
        with pytest.raises(ValidationError, match="xyz_body_rotation is None"):
            validate_fitted_wrap_parameters(params)

    def test_body_dict_not_a_dict(self):
        params = {"femur": {"femur_r": "bad"}}
        with pytest.raises(ValidationError, match="must be a dict"):
            validate_fitted_wrap_parameters(params)

    def test_wrap_dicts_not_a_dict(self):
        params = {"femur": {"femur_r": {"cylinder": "bad"}}}
        with pytest.raises(ValidationError, match="must be a dict"):
            validate_fitted_wrap_parameters(params)

    def test_cylinder_radius_wrong_type(self):
        ws = _make_wrap_surface("cylinder")
        ws.radius = "not_a_number"
        params = {"femur": {"femur_r": {"cylinder": {"test": ws}}}}
        with pytest.raises(ValidationError, match="must be numeric"):
            validate_fitted_wrap_parameters(params)

    def test_cylinder_radius_0d_ndarray_accepted(self):
        """0-d numpy arrays (from tensor.numpy()) should be accepted as scalar."""
        ws = _make_wrap_surface("cylinder")
        ws.radius = np.array(0.005)  # 0-d ndarray, as produced by PyTorch fitters
        ws.length = np.array(0.04)
        params = {"femur": {"femur_r": {"cylinder": {"test": ws}}}}
        validate_fitted_wrap_parameters(params)  # should not raise

    def test_cylinder_radius_1d_ndarray_rejected(self):
        """Non-scalar numpy arrays should be rejected."""
        ws = _make_wrap_surface("cylinder")
        ws.radius = np.array([0.005])  # 1-d, not scalar
        params = {"femur": {"femur_r": {"cylinder": {"test": ws}}}}
        with pytest.raises(ValidationError, match="must be scalar"):
            validate_fitted_wrap_parameters(params)


# ===========================================================================
# validate_surface_idx
# ===========================================================================


class TestValidateSurfaceIdx:
    @pytest.mark.parametrize("idx", [0, 1, 2, 3])
    def test_valid_indices(self, idx):
        validate_surface_idx(idx)  # should not raise

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="out of range"):
            validate_surface_idx(-1)

    def test_too_large_raises(self):
        with pytest.raises(ValidationError, match="out of range"):
            validate_surface_idx(4)

    def test_float_raises(self):
        with pytest.raises(ValidationError, match="must be an int"):
            validate_surface_idx(1.0)

    def test_custom_max_idx(self):
        validate_surface_idx(5, max_idx=5)  # should not raise
        with pytest.raises(ValidationError, match="out of range"):
            validate_surface_idx(6, max_idx=5)


# ===========================================================================
# ValidationError is a ValueError
# ===========================================================================


class TestValidationError:
    def test_is_value_error(self):
        """ValidationError should be catchable as ValueError for backwards compat."""
        with pytest.raises(ValueError):
            raise ValidationError("test")
