"""Non-trivial scaling: factors != 1 should propagate coherently."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import opensim as osim
import pymskt as mskt
import pytest
from scipy.spatial import cKDTree

# --- helpers ----------------------------------------------------------------


def _resolve_geometry_dir(model_path: Path) -> Path:
    return model_path.parent / "Geometry"


def _load_attached_mesh(
    model: osim.Model, body_name: str, mesh_filename: str, geom_dir: Path
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load an attached_geometry Mesh's vertices and its scale_factors.

    Returns (raw_vertices_Nx3, scale_factors). Caller multiplies if it wants
    post-scale-applied geometry.
    """
    body = model.getBodySet().get(body_name)
    n = body.getPropertyByName("attached_geometry").size()
    for gi in range(n):
        ag = body.upd_attached_geometry(gi)
        mesh = osim.Mesh.safeDownCast(ag)
        if mesh is None:
            continue
        if mesh.get_mesh_file() != mesh_filename:
            continue
        sf = mesh.get_scale_factors()
        m = mskt.mesh.Mesh(str(geom_dir / mesh_filename))
        return np.asarray(m.point_coords), (float(sf[0]), float(sf[1]), float(sf[2]))
    raise LookupError(f"Mesh {mesh_filename!r} not found on {body_name!r}")


def _cart_to_bone_distances(
    model: osim.Model,
    geom_dir: Path,
    bone_body: str,
    bone_filename: str,
    cart_filename: str,
) -> np.ndarray:
    """Distance from each cart vertex to its nearest bone vertex,
    both in body-local frame with the model's stored scale_factors applied."""
    bone_pts, bone_sf = _load_attached_mesh(model, bone_body, bone_filename, geom_dir)
    cart_pts, cart_sf = _load_attached_mesh(model, bone_body, cart_filename, geom_dir)
    bone_pts = bone_pts * np.asarray(bone_sf)
    cart_pts = cart_pts * np.asarray(cart_sf)
    tree = cKDTree(bone_pts)
    d, _ = tree.query(cart_pts, k=1)
    return d


def _gather_blankevoort(model: osim.Model) -> Dict[str, Dict[str, float]]:
    out = {}
    fs = model.upd_ForceSet()
    for i in range(fs.getSize()):
        lig = osim.Blankevoort1991Ligament.safeDownCast(fs.get(i))
        if lig is None:
            continue
        name = fs.get(i).getName()
        slack = float(lig.get_slack_length())
        # path length needs realized state; compute via GeometryPath.getLength on init state
        state = model.initSystem()
        path = lig.updGeometryPath()
        path_len = float(path.getLength(state))
        ref_strain = (path_len - slack) / slack if slack > 0 else float("nan")
        out[name] = {
            "slack": slack,
            "path_length": path_len,
            "ref_strain": ref_strain,
        }
    return out


def _gather_wraps(model: osim.Model) -> Dict[str, Tuple[str, Tuple[float, float, float]]]:
    """For each wrap, return (parent_body, translation)."""
    out = {}
    bs = model.getBodySet()
    for bi in range(bs.getSize()):
        body = bs.get(bi)
        wos = body.getWrapObjectSet()
        for wi in range(wos.getSize()):
            w = wos.get(wi)
            t = w.get_translation()
            out[f"{body.getName()}/{w.getName()}"] = (
                body.getName(),
                (float(t[0]), float(t[1]), float(t[2])),
            )
    return out


def _bone_aabb(model: osim.Model, body_name: str, geom_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    body = model.getBodySet().get(body_name)
    n = body.getPropertyByName("attached_geometry").size()
    pts_all = []
    for gi in range(n):
        ag = body.upd_attached_geometry(gi)
        mesh = osim.Mesh.safeDownCast(ag)
        if mesh is None:
            continue
        mf = mesh.get_mesh_file()
        if not mf:
            continue
        sf = mesh.get_scale_factors()
        m = mskt.mesh.Mesh(str(geom_dir / mf))
        pts_all.append(np.asarray(m.point_coords) * np.asarray((sf[0], sf[1], sf[2])))
    if not pts_all:
        raise LookupError(f"No attached geometry on {body_name}")
    P = np.concatenate(pts_all, axis=0)
    return P.min(axis=0), P.max(axis=0)


# --- tests ------------------------------------------------------------------


@pytest.mark.slow
class TestSubjectNontrivial:
    """Use RSubject_121's real AB factors. Each check compares against the
    identity-baseline (same pipeline at s=1)."""

    @pytest.fixture(scope="class")
    def base_model(self, identity_scaled_model):
        m = osim.Model(str(identity_scaled_model))
        m.initSystem()
        return m

    @pytest.fixture(scope="class")
    def base_geom(self, identity_scaled_model):
        return _resolve_geometry_dir(identity_scaled_model)

    @pytest.fixture(scope="class")
    def subj_model(self, subject_scaled_model):
        m = osim.Model(str(subject_scaled_model))
        m.initSystem()
        return m

    @pytest.fixture(scope="class")
    def subj_geom(self, subject_scaled_model):
        return _resolve_geometry_dir(subject_scaled_model)

    @pytest.fixture(scope="class")
    def s_wa(self, rsubject121_ab_path):
        from nsosim.scaling.scale_factors import read_ab_factors

        f = read_ab_factors(rsubject121_ab_path)
        return (f["femur_r"][2] + f["tibia_r"][2]) / 2.0

    # The central correctness check: cart STL was pre-scaled, bone is
    # frame-scaled by ScaleTool. If we forgot the pre-scale, this fails.
    @pytest.mark.parametrize(
        "bone_body,bone_file,cart_file",
        [
            ("femur_distal_r", "smith2019-R-femur-bone.stl", "smith2019-R-femur-cartilage.stl"),
            ("tibia_proximal_r", "smith2019-R-tibia-bone.stl", "smith2019-R-tibia-cartilage.stl"),
            ("patella_r", "smith2019-R-patella-bone.stl", "smith2019-R-patella-cartilage.stl"),
        ],
    )
    def test_cartilage_bone_proximity_preserved(
        self,
        base_model,
        base_geom,
        subj_model,
        subj_geom,
        s_wa,
        bone_body,
        bone_file,
        cart_file,
    ):
        d_base = _cart_to_bone_distances(base_model, base_geom, bone_body, bone_file, cart_file)
        d_subj = _cart_to_bone_distances(subj_model, subj_geom, bone_body, bone_file, cart_file)
        # All distances should scale isotropically by s_wa.
        # Stats: mean and p95.
        mean_b, mean_s = float(d_base.mean()), float(d_subj.mean())
        p95_b, p95_s = float(np.percentile(d_base, 95)), float(np.percentile(d_subj, 95))
        expected_mean = mean_b * s_wa
        expected_p95 = p95_b * s_wa
        # Cart-bone distances scale by s_wa to ~1e-9 m (measured worst p95 dev
        # 9.3e-10 m on this reference knee — STL write precision; the kdtree
        # nearest-neighbour pairing is scale-invariant). 1e-8 m keeps ~10x
        # margin while being 5e4x tighter than the original "0.5 mm" guess.
        assert abs(mean_s - expected_mean) < 1e-8, (
            f"{bone_body}: cart-bone mean drifted. "
            f"baseline={mean_b:.8f}m, scaled={mean_s:.8f}m, "
            f"expected={expected_mean:.8f}m (s_wa={s_wa:.4f})"
        )
        assert abs(p95_s - expected_p95) < 1e-8, (
            f"{bone_body}: cart-bone p95 drifted. "
            f"baseline={p95_b:.8f}m, scaled={p95_s:.8f}m, "
            f"expected={expected_p95:.8f}m"
        )

    def test_blankevoort_reference_strain_preserved(self, base_model, subj_model):
        b = _gather_blankevoort(base_model)
        s = _gather_blankevoort(subj_model)
        common = set(b) & set(s)
        assert len(common) >= 80, "expected at least 80 ligaments to compare"
        # Reference strain (path - slack)/slack is preserved EXACTLY: path and
        # slack scale by the same factor, so it cancels. Measured max drift
        # 1.3e-16 (machine eps) here; assert at 1e-12 (direct abs, not
        # pytest.approx whose default rel=1e-6 would mask this).
        for name in common:
            drift = abs(s[name]["ref_strain"] - b[name]["ref_strain"])
            assert drift < 1e-12, (
                f"{name}: reference strain drifted by {drift:.2e} "
                f"({b[name]['ref_strain']:.6f} -> {s[name]['ref_strain']:.6f})"
            )

    def test_wrap_translations_inside_parent_aabb(self, subj_model, subj_geom):
        wraps = _gather_wraps(subj_model)
        # Cache AABBs per body
        aabb_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        n_checked = 0
        for tag, (body_name, trans) in wraps.items():
            if body_name not in aabb_cache:
                try:
                    aabb_cache[body_name] = _bone_aabb(subj_model, body_name, subj_geom)
                except LookupError:
                    continue  # body has no attached visual mesh
            mn, mx = aabb_cache[body_name]
            t = np.asarray(trans)
            # Allow a generous pad — some wraps sit outside the bone surface,
            # what we're guarding against is wraps that drift far away from
            # their parent.
            extent = mx - mn
            pad = 0.5 * extent + 1e-3
            ok = bool(np.all(t >= mn - pad) and np.all(t <= mx + pad))
            assert ok, (
                f"wrap {tag} translation {t} far outside body {body_name} " f"AABB ({mn} to {mx})"
            )
            n_checked += 1
        # Many wraps live on bodies without attached_geometry (e.g. femur_r,
        # tibia_r) — those bodies are skipped via LookupError. Of the 39 total
        # wraps, ~24 land on bodies we can AABB-check.
        assert n_checked >= 20, f"expected at least 20 wraps to AABB-check, got {n_checked}"
