"""scale_comak_model(geometry_staging="link") makes the same scaled model as "copy".

The reference base is scaled by the in-repo synthetic AB output (s_wa = 0.9) twice,
once per mode. The link-mode folder must hold real copies of exactly the files the
bake rewrote and relative links for the rest; every file must have the same bytes as
in copy mode; the .osim must be identical; the source meshes must be untouched; and
the scaled model must load from a fresh process with no geometry search path.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nsosim.geometry_staging import link_file


def _md5(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _tree_md5(folder: Path) -> dict:
    return {f: _md5(folder / f) for f in sorted(os.listdir(folder))}


def _scale(base: Path, ab: Path, out_dir: Path, staging: str) -> tuple:
    from nsosim.scaling import scale_comak_model

    out = out_dir / "scaled.osim"
    report = scale_comak_model(
        base_osim=base,
        ab_scaled_osim=ab,
        output_osim=out,
        output_geometry_dir=out_dir / "Geometry",
        mode="WA",
        geometry_staging=staging,
    )
    return out, json.loads(Path(report).read_text())


@pytest.fixture(scope="module")
def both_modes(base_comak_path: Path, synthetic_wa_ab_osim: Path, tmp_path_factory):
    source = base_comak_path.parent / "Geometry"
    before = _tree_md5(source)
    root = tmp_path_factory.mktemp("staging")
    copy = _scale(base_comak_path, synthetic_wa_ab_osim, root / "copy", "copy")
    link = _scale(base_comak_path, synthetic_wa_ab_osim, root / "link", "link")
    return {"copy": copy, "link": link, "source": source, "source_before": before}


def test_same_files_same_bytes(both_modes):
    (copy_osim, _), (link_osim, _) = both_modes["copy"], both_modes["link"]
    assert _tree_md5(link_osim.parent / "Geometry") == _tree_md5(copy_osim.parent / "Geometry")


def test_same_model_file(both_modes):
    (copy_osim, _), (link_osim, _) = both_modes["copy"], both_modes["link"]
    assert _md5(link_osim) == _md5(copy_osim)


def test_real_files_are_exactly_the_baked_ones(both_modes):
    link_osim, report = both_modes["link"]
    geometry = link_osim.parent / "Geometry"
    real = {f for f in os.listdir(geometry) if not (geometry / f).is_symlink()}
    assert real == set(report["knee_geometry_baked"])
    assert report["geometry_staging"] == {
        "mode": "link",
        "copied": len(real),
        "linked": len(os.listdir(geometry)) - len(real),
    }
    for f in os.listdir(geometry):
        if (geometry / f).is_symlink():
            assert not os.path.isabs(os.readlink(geometry / f))


def test_copy_mode_report(both_modes):
    copy_osim, report = both_modes["copy"]
    n = len(os.listdir(copy_osim.parent / "Geometry"))
    assert report["geometry_staging"] == {"mode": "copy", "copied": n, "linked": 0}


def test_source_meshes_untouched(both_modes):
    assert _tree_md5(both_modes["source"]) == both_modes["source_before"]


def test_loads_from_a_fresh_process_without_a_search_path(both_modes):
    link_osim, _ = both_modes["link"]
    code = (
        "import sys, opensim as osim; m = osim.Model(sys.argv[1]); m.initSystem(); "
        "print(m.getContactGeometrySet().getSize())"
    )
    done = subprocess.run([sys.executable, "-c", code, str(link_osim)], capture_output=True, text=True,
                          cwd=link_osim.parent.parent)
    assert done.returncode == 0, done.stderr[-2000:]
    assert int(done.stdout.strip().splitlines()[-1]) > 0


def test_bake_refuses_a_linked_file(base_comak_path: Path, synthetic_wa_ab_osim: Path, tmp_path):
    """A knee mesh staged as a link must stop the bake before it writes anything."""
    from nsosim.scaling.config import WA_KNEE_BODIES
    from nsosim.scaling.knee_geometry import bake_knee_geometry

    copy_osim, report = _scale(base_comak_path, synthetic_wa_ab_osim, tmp_path / "copy", "copy")
    geometry = copy_osim.parent / "Geometry"
    victim = sorted(report["knee_geometry_baked"])[0]
    shared = tmp_path / "shared"
    shared.mkdir()
    shutil.copy2(geometry / victim, shared / victim)
    before = _md5(shared / victim)
    (geometry / victim).unlink()
    link_file(shared / victim, geometry / victim)
    with pytest.raises(RuntimeError, match="through symlinks"):
        bake_knee_geometry(copy_osim, WA_KNEE_BODIES, 0.9, geometry)
    assert _md5(shared / victim) == before
