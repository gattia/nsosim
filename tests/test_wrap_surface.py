"""Tests for the wrap_surface data class in wrap_surface_fitting/main.py."""

import numpy as np
import pytest

from nsosim.wrap_surface_fitting.main import wrap_surface

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestWrapSurfaceConstruction:
    """Test that wrap_surface can be constructed with all field types."""

    def test_cylinder_construction(self):
        ws = wrap_surface(
            name="KnExt_at_fem_r",
            body="femur_r",
            type_="WrapCylinder",
            xyz_body_rotation=np.array([0.1, 0.2, 0.3]),
            translation=np.array([0.01, 0.02, 0.03]),
            radius=0.015,
            length=0.06,
            dimensions=None,
        )
        assert ws.name == "KnExt_at_fem_r"
        assert ws.body == "femur_r"
        assert ws.type_ == "WrapCylinder"
        assert ws.radius == 0.015
        assert ws.length == 0.06
        assert ws.dimensions is None

    def test_ellipsoid_construction(self):
        ws = wrap_surface(
            name="Gastroc_at_Condyles_r",
            body="femur_r",
            type_="WrapEllipsoid",
            xyz_body_rotation=np.array([0.0, 0.0, 0.0]),
            translation=np.array([0.0, -0.01, 0.0]),
            radius=None,
            length=None,
            dimensions=np.array([0.02, 0.03, 0.04]),
        )
        assert ws.type_ == "WrapEllipsoid"
        assert ws.radius is None
        assert ws.length is None
        assert np.allclose(ws.dimensions, [0.02, 0.03, 0.04])

    def test_stores_numpy_arrays(self):
        rot = np.array([0.1, 0.2, 0.3])
        trans = np.array([0.01, 0.02, 0.03])
        ws = wrap_surface(
            name="test",
            body="body",
            type_="WrapCylinder",
            xyz_body_rotation=rot,
            translation=trans,
            radius=0.01,
            length=0.05,
            dimensions=None,
        )
        assert np.array_equal(ws.xyz_body_rotation, rot)
        assert np.array_equal(ws.translation, trans)


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


class TestToDict:
    """Test the to_dict() serialization method."""

    @pytest.fixture
    def cylinder_ws(self):
        return wrap_surface(
            name="KnExt_at_fem_r",
            body="femur_r",
            type_="WrapCylinder",
            xyz_body_rotation=np.array([0.1, 0.2, 0.3]),
            translation=np.array([0.01, 0.02, 0.03]),
            radius=0.015,
            length=0.06,
            dimensions=None,
        )

    def test_all_keys_present(self, cylinder_ws):
        d = cylinder_ws.to_dict()
        expected_keys = {
            "name",
            "body",
            "type_",
            "xyz_body_rotation",
            "translation",
            "radius",
            "length",
            "dimensions",
        }
        assert set(d.keys()) == expected_keys

    def test_numpy_arrays_become_lists(self, cylinder_ws):
        d = cylinder_ws.to_dict()
        assert isinstance(d["xyz_body_rotation"], list)
        assert isinstance(d["translation"], list)

    def test_none_stays_none(self, cylinder_ws):
        d = cylinder_ws.to_dict()
        assert d["dimensions"] is None

    def test_scalars_preserved(self, cylinder_ws):
        d = cylinder_ws.to_dict()
        assert d["radius"] == 0.015
        assert d["length"] == 0.06


# ---------------------------------------------------------------------------
# Round-trip: construct -> to_dict -> reconstruct
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Construct -> to_dict -> reconstruct should preserve values."""

    def test_cylinder_roundtrip(self):
        original = wrap_surface(
            name="test_cyl",
            body="femur_r",
            type_="WrapCylinder",
            xyz_body_rotation=np.array([0.1, -0.2, 0.3]),
            translation=np.array([-0.01, 0.02, -0.03]),
            radius=0.025,
            length=0.08,
            dimensions=None,
        )
        d = original.to_dict()

        # Reconstruct — need to convert lists back to arrays
        reconstructed = wrap_surface(
            name=d["name"],
            body=d["body"],
            type_=d["type_"],
            xyz_body_rotation=np.array(d["xyz_body_rotation"]),
            translation=np.array(d["translation"]),
            radius=d["radius"],
            length=d["length"],
            dimensions=d["dimensions"],
        )

        assert reconstructed.name == original.name
        assert reconstructed.body == original.body
        assert reconstructed.type_ == original.type_
        assert np.allclose(reconstructed.xyz_body_rotation, original.xyz_body_rotation)
        assert np.allclose(reconstructed.translation, original.translation)
        assert reconstructed.radius == original.radius
        assert reconstructed.length == original.length
        assert reconstructed.dimensions is None

    def test_ellipsoid_roundtrip(self):
        original = wrap_surface(
            name="test_ell",
            body="femur_r",
            type_="WrapEllipsoid",
            xyz_body_rotation=np.array([0.5, 0.6, 0.7]),
            translation=np.array([0.001, 0.002, 0.003]),
            radius=None,
            length=None,
            dimensions=np.array([0.02, 0.03, 0.04]),
        )
        d = original.to_dict()

        reconstructed = wrap_surface(
            name=d["name"],
            body=d["body"],
            type_=d["type_"],
            xyz_body_rotation=np.array(d["xyz_body_rotation"]),
            translation=np.array(d["translation"]),
            radius=d["radius"],
            length=d["length"],
            dimensions=np.array(d["dimensions"]),
        )

        assert reconstructed.name == original.name
        assert reconstructed.type_ == original.type_
        assert np.allclose(reconstructed.dimensions, original.dimensions)
        assert reconstructed.radius is None
        assert reconstructed.length is None
