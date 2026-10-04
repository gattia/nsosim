"""Lay out a model's Geometry folder with links to unchanged meshes instead of copies.

Two writers fill a new ``Geometry/`` folder from an existing one:
``model_building.build_joint_model`` (the base model folder → every built model) and
``scaling.scale_comak_model`` (a built model → every scaled copy of it). Most of the
files they write are byte-identical to the source: the generic whole-body meshes,
about 25 MB on disk per model. In ``"link"`` mode each unchanged file becomes a
relative symlink to the source file, and only files the caller will rewrite are
real copies.

Why links and not OpenSim's geometry search path: a model resolves a bare mesh name
from its own folder, then ``<model folder>/Geometry``, then a process-wide search
list (newest entry first). A queue worker builds and simulates many knees in one
process, so a search-path miss would silently pick up another knee's mesh of the
same name. A link keeps every lookup inside the model's own folder, in any process
and from a fresh shell.

Rules that keep links safe:
- a file the caller rewrites afterwards must be a real copy (``real=``). A rewrite in
  place through a link -- VTK and OpenSim write in C++ -- would overwrite the source;
- links are relative, computed from the link's physical folder to the source path as
  given (not resolved), so a source reached through a linked input folder keeps that
  folder in the link text and the tree can move as a whole;
- only files are linked, never ``.osim`` models or folders, and a non-flat Geometry
  folder raises rather than being guessed at.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Iterable

GEOMETRY_STAGING_MODES = ("copy", "link")


def check_mode(mode: str) -> str:
    if mode not in GEOMETRY_STAGING_MODES:
        raise ValueError(f"geometry staging mode must be one of {GEOMETRY_STAGING_MODES}, got {mode!r}")
    return mode


def link_file(source, dest) -> Path:
    """Write ``dest`` as a relative symlink to ``source``; return ``dest``."""
    source = os.path.abspath(source)
    dest = Path(dest)
    os.symlink(os.path.relpath(source, os.path.realpath(dest.parent)), dest)
    return dest


def stage_geometry(source_dir, dest_dir, real: Iterable[str] = (), mode: str = "copy") -> Dict[str, object]:
    """Fill a new ``dest_dir`` from ``source_dir``.

    ``mode="copy"``: ``shutil.copytree`` (the original behaviour; ``real`` is ignored).
    ``mode="link"``: ``dest_dir`` is a real folder; each name in ``real`` is a real copy
    (``shutil.copy2``, as copytree does) and every other file is a relative symlink to
    ``source_dir/<name>``.

    ``dest_dir`` must not exist. Returns a summary for run reports:
    ``{"mode", "copied", "linked"}`` (file counts).
    """
    check_mode(mode)
    source_dir, dest_dir = Path(source_dir), Path(dest_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Geometry folder not found: {source_dir}")
    if os.path.lexists(dest_dir):
        raise FileExistsError(f"{dest_dir} already exists")

    names = sorted(os.listdir(source_dir))
    if mode == "copy":
        shutil.copytree(source_dir, dest_dir)
        return {"mode": mode, "copied": len(names), "linked": 0}

    real = set(real)
    missing = sorted(real - set(names))
    if missing:
        raise FileNotFoundError(f"files to copy are not in {source_dir}: {missing}")
    folders = [n for n in names if (source_dir / n).is_dir()]
    if folders:
        raise ValueError(f"{source_dir} holds subfolders {folders}; link mode stages flat folders only")

    dest_dir.mkdir(parents=True)
    copied = linked = 0
    for name in names:
        if name in real:
            shutil.copy2(source_dir / name, dest_dir / name)
            copied += 1
        else:
            link_file(source_dir / name, dest_dir / name)
            linked += 1
    return {"mode": mode, "copied": copied, "linked": linked}


def stage_model_folder(base_dir, dest_dir, mode: str = "copy") -> Dict[str, object]:
    """Start a new model folder from a template model folder.

    ``mode="copy"``: ``shutil.copytree`` of the whole folder (the original behaviour).
    ``mode="link"``: the top-level files (the template ``.osim``, logs) are real copies;
    ``Geometry/`` is a real folder of relative links (``stage_geometry``). Any other
    subfolder raises.
    """
    check_mode(mode)
    base_dir, dest_dir = Path(base_dir), Path(dest_dir)
    if mode == "copy":
        shutil.copytree(base_dir, dest_dir)
        return {"mode": mode}
    if os.path.lexists(dest_dir):
        raise FileExistsError(f"{dest_dir} already exists")
    folders = [n for n in os.listdir(base_dir) if (base_dir / n).is_dir() and n != "Geometry"]
    if folders:
        raise ValueError(f"{base_dir} holds subfolders {folders}; link mode expects only files and Geometry/")
    dest_dir.mkdir(parents=True)
    for name in sorted(os.listdir(base_dir)):
        if name != "Geometry":
            shutil.copy2(base_dir / name, dest_dir / name)
    summary = {"mode": mode}
    if (base_dir / "Geometry").is_dir():
        summary.update(stage_geometry(base_dir / "Geometry", dest_dir / "Geometry", mode="link"))
    return summary
