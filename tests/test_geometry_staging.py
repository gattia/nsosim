"""nsosim.geometry_staging: links to unchanged meshes instead of copies (filesystem only)."""

import os
from pathlib import Path

import pytest

from nsosim.geometry_staging import link_file, stage_geometry, stage_model_folder


def _folder(root: Path, files=("a.vtp", "knee.stl", "back.stl")) -> Path:
    root.mkdir(parents=True)
    for f in files:
        (root / f).write_bytes(f.encode() * 7)
    return root


def _content(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


class TestStageGeometry:
    def test_copy_mode_is_a_plain_copytree(self, tmp_path):
        src = _folder(tmp_path / "src")
        summary = stage_geometry(src, tmp_path / "dst", real=["knee.stl"], mode="copy")
        assert summary == {"mode": "copy", "copied": 3, "linked": 0}
        assert not any(os.path.islink(tmp_path / "dst" / f) for f in os.listdir(tmp_path / "dst"))

    def test_default_mode_is_copy(self, tmp_path):
        assert stage_geometry(_folder(tmp_path / "src"), tmp_path / "dst")["mode"] == "copy"

    def test_link_mode_copies_only_the_real_set(self, tmp_path):
        src = _folder(tmp_path / "src")
        summary = stage_geometry(src, tmp_path / "dst", real=["knee.stl"], mode="link")
        assert summary == {"mode": "link", "copied": 1, "linked": 2}
        dst = tmp_path / "dst"
        assert dst.is_dir() and not dst.is_symlink()
        assert not (dst / "knee.stl").is_symlink()
        for f in ("a.vtp", "back.stl"):
            assert (dst / f).is_symlink()
            assert not os.path.isabs(os.readlink(dst / f))
        for f in os.listdir(src):
            assert _content(dst / f) == _content(src / f)

    def test_a_write_to_a_real_copy_leaves_the_source_alone(self, tmp_path):
        src = _folder(tmp_path / "src")
        stage_geometry(src, tmp_path / "dst", real=["knee.stl"], mode="link")
        (tmp_path / "dst" / "knee.stl").write_bytes(b"scaled")
        assert _content(src / "knee.stl") == b"knee.stl" * 7

    def test_links_keep_a_linked_input_folder_in_the_path(self, tmp_path):
        """A source reached through a linked folder keeps that folder in the link text."""
        real = _folder(tmp_path / "published" / "Geometry")
        (tmp_path / "root" / "inputs").mkdir(parents=True)
        os.symlink(real.parent, tmp_path / "root" / "inputs" / "reference")
        src = tmp_path / "root" / "inputs" / "reference" / "Geometry"
        stage_geometry(src, tmp_path / "root" / "model" / "Geometry", mode="link")
        text = os.readlink(tmp_path / "root" / "model" / "Geometry" / "a.vtp")
        assert text == os.path.join("..", "..", "inputs", "reference", "Geometry", "a.vtp")
        assert _content(tmp_path / "root" / "model" / "Geometry" / "a.vtp") == b"a.vtp" * 7

    def test_the_tree_can_move_as_a_whole(self, tmp_path):
        src = _folder(tmp_path / "root" / "built" / "Geometry")
        stage_geometry(src, tmp_path / "root" / "scaled" / "Geometry", real=["knee.stl"], mode="link")
        os.rename(tmp_path / "root", tmp_path / "moved")
        assert _content(tmp_path / "moved" / "scaled" / "Geometry" / "a.vtp") == b"a.vtp" * 7

    def test_a_linked_source_file_is_copied_as_content(self, tmp_path):
        """A real copy of a file that is itself a link holds the bytes, not another link."""
        base = _folder(tmp_path / "base")
        built = tmp_path / "built"
        built.mkdir()
        for f in os.listdir(base):
            link_file(base / f, built / f)
        stage_geometry(built, tmp_path / "scaled", real=["knee.stl"], mode="link")
        assert not (tmp_path / "scaled" / "knee.stl").is_symlink()
        (tmp_path / "scaled" / "knee.stl").write_bytes(b"scaled")
        assert _content(base / "knee.stl") == b"knee.stl" * 7

    def test_real_names_must_exist(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not in"):
            stage_geometry(_folder(tmp_path / "src"), tmp_path / "dst", real=["nope.stl"], mode="link")

    def test_subfolders_raise_in_link_mode(self, tmp_path):
        src = _folder(tmp_path / "src")
        (src / "sub").mkdir()
        with pytest.raises(ValueError, match="subfolders"):
            stage_geometry(src, tmp_path / "dst", mode="link")

    @pytest.mark.parametrize("mode", ["copy", "link"])
    def test_an_existing_destination_raises(self, tmp_path, mode):
        src = _folder(tmp_path / "src")
        (tmp_path / "dst").mkdir()
        with pytest.raises(FileExistsError):
            stage_geometry(src, tmp_path / "dst", mode=mode)

    def test_unknown_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match="staging mode"):
            stage_geometry(_folder(tmp_path / "src"), tmp_path / "dst", mode="hardlink")


class TestStageModelFolder:
    def _template(self, tmp_path):
        base = tmp_path / "base"
        _folder(base / "Geometry")
        (base / "template.osim").write_text("<OpenSimDocument/>")
        (base / "opensim.log").write_text("log")
        return base

    def test_copy_mode_is_a_plain_copytree(self, tmp_path):
        base = self._template(tmp_path)
        stage_model_folder(base, tmp_path / "model", mode="copy")
        assert not any((tmp_path / "model" / "Geometry" / f).is_symlink() for f in os.listdir(base / "Geometry"))

    def test_link_mode_copies_files_and_links_meshes(self, tmp_path):
        base = self._template(tmp_path)
        summary = stage_model_folder(base, tmp_path / "model", mode="link")
        model = tmp_path / "model"
        assert summary == {"mode": "link", "copied": 0, "linked": 3}
        assert (model / "template.osim").is_file() and not (model / "template.osim").is_symlink()
        assert (model / "opensim.log").is_file() and not (model / "opensim.log").is_symlink()
        assert (model / "Geometry").is_dir() and not (model / "Geometry").is_symlink()
        assert all((model / "Geometry" / f).is_symlink() for f in os.listdir(base / "Geometry"))

    def test_link_mode_refuses_other_subfolders(self, tmp_path):
        base = self._template(tmp_path)
        (base / "extra").mkdir()
        with pytest.raises(ValueError, match="subfolders"):
            stage_model_folder(base, tmp_path / "model", mode="link")


def test_save_geometry_files_refuses_to_write_through_a_link(tmp_path):
    from nsosim.model_building import save_geometry_files

    shared = _folder(tmp_path / "shared", files=("femur_nsm_recon_osim.stl",))
    bones = tmp_path / "bones" / "femur"
    bones.mkdir(parents=True)
    (bones / "femur_nsm_recon_osim.stl").write_bytes(b"subject")
    model = tmp_path / "model"
    (model / "Geometry").mkdir(parents=True)
    link_file(shared / "femur_nsm_recon_osim.stl", model / "Geometry" / "femur_nsm_recon_osim.stl")
    with pytest.raises(FileExistsError, match="symlink"):
        save_geometry_files(str(tmp_path / "bones"), str(model), {"femur": ["femur_nsm_recon_osim.stl"]})
    assert _content(shared / "femur_nsm_recon_osim.stl") == b"femur_nsm_recon_osim.stl" * 7
