"""Tests for get_mesh_names() — explicit mesh name mapping for decoder outputs."""

import logging

import pytest

from nsosim.utils import get_mesh_names


class TestGetMeshNames:
    """Tests for get_mesh_names()."""

    def test_explicit_mesh_names(self):
        """Config with mesh_names returns them directly."""
        config = {
            "objects_per_decoder": 4,
            "mesh_names": ["bone", "cart", "med_men", "lat_men"],
        }
        assert get_mesh_names(config) == ["bone", "cart", "med_men", "lat_men"]

    def test_explicit_mesh_names_two(self):
        """Config with 2-surface mesh_names."""
        config = {
            "objects_per_decoder": 2,
            "mesh_names": ["bone", "cart"],
        }
        assert get_mesh_names(config) == ["bone", "cart"]

    def test_explicit_mesh_names_length_mismatch(self):
        """Raises ValueError when mesh_names length != objects_per_decoder."""
        config = {
            "objects_per_decoder": 2,
            "mesh_names": ["bone", "cart", "extra"],
        }
        with pytest.raises(ValueError, match="mesh_names has 3 entries"):
            get_mesh_names(config)

    def test_explicit_mesh_names_none_falls_through(self):
        """mesh_names=None triggers fallback, same as missing."""
        config = {
            "objects_per_decoder": 2,
            "bone": "patella",
            "mesh_names": None,
        }
        assert get_mesh_names(config) == ["bone", "cart"]

    def test_fallback_femur_4(self):
        """Fallback: femur with 4 objects → bone, cart, med_men, lat_men."""
        config = {"objects_per_decoder": 4, "bone": "femur"}
        assert get_mesh_names(config) == ["bone", "cart", "med_men", "lat_men"]

    def test_fallback_femur_2(self):
        """Fallback: femur with 2 objects → bone, cart."""
        config = {"objects_per_decoder": 2, "bone": "femur"}
        assert get_mesh_names(config) == ["bone", "cart"]

    def test_fallback_tibia_2(self):
        """Fallback: tibia with 2 objects → bone, cart."""
        config = {"objects_per_decoder": 2, "bone": "tibia"}
        assert get_mesh_names(config) == ["bone", "cart"]

    def test_fallback_tibia_3(self):
        """Fallback: tibia with 3 objects → bone, cart, fibula."""
        config = {"objects_per_decoder": 3, "bone": "tibia"}
        assert get_mesh_names(config) == ["bone", "cart", "fibula"]

    def test_fallback_patella_2(self):
        """Fallback: patella with 2 objects → bone, cart."""
        config = {"objects_per_decoder": 2, "bone": "patella"}
        assert get_mesh_names(config) == ["bone", "cart"]

    def test_fallback_unknown_combo_raises(self):
        """Unknown (bone, count) combo raises ValueError."""
        config = {"objects_per_decoder": 5, "bone": "femur"}
        with pytest.raises(ValueError, match="No default mesh_names"):
            get_mesh_names(config)

    def test_fallback_unknown_bone_raises(self):
        """Unknown bone type with no mesh_names raises ValueError."""
        config = {"objects_per_decoder": 2, "bone": "humerus"}
        with pytest.raises(ValueError, match="No default mesh_names"):
            get_mesh_names(config)

    def test_fallback_logs_warning(self, caplog):
        """Fallback path logs a warning about missing mesh_names."""
        config = {"objects_per_decoder": 2, "bone": "femur"}
        with caplog.at_level(logging.WARNING, logger="nsosim.utils"):
            get_mesh_names(config)
        assert "missing 'mesh_names'" in caplog.text

    def test_returns_copy(self):
        """Returned list is a copy, not the original config list."""
        names = ["bone", "cart"]
        config = {"objects_per_decoder": 2, "mesh_names": names}
        result = get_mesh_names(config)
        assert result == names
        assert result is not names

    def test_no_objects_per_decoder_defaults_to_1(self):
        """Missing objects_per_decoder defaults to 1, which has no fallback."""
        config = {"bone": "femur"}
        with pytest.raises(ValueError, match="No default mesh_names"):
            get_mesh_names(config)
