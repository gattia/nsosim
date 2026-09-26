"""Tests for the meniscus radial-envelope trim origin (``meniscus_center_mode``).

See ``nsosim.model_building.meniscus_trim_centers`` and comak_gait_simulation
``.claude/plans/MENISCUS_CENTER_LABEL_IMPACT.md`` (Phase 7).
"""

import numpy as np
import pytest
import pyvista as pv

from nsosim.model_building import (
    DEFAULT_MENISCUS_CENTER_MODE,
    meniscus_trim_centers,
)


def _labelled_tibia(with_plateau=True):
    pts = np.array(
        [
            [0.00, 0.0, -0.020], [0.002, 0.0, -0.022],  # medial plateau patch
            [0.00, 0.0, 0.025], [0.002, 0.0, 0.027],  # lateral plateau patch
            [0.010, 0.0, 0.010], [0.015, 0.0, 0.005],  # as-run medial / lateral patches
        ]
    )
    mesh = pv.PolyData(pts)
    mesh["med_meniscus_center_binary"] = np.array([0, 0, 0, 0, 1, 0])
    mesh["lat_meniscus_center_binary"] = np.array([0, 0, 0, 0, 0, 1])
    if with_plateau:
        mesh["med_plateau_center_binary"] = np.array([1, 1, 0, 0, 0, 0])
        mesh["lat_plateau_center_binary"] = np.array([0, 0, 1, 1, 0, 0])
    return mesh, pts


def test_default_is_auto():
    assert DEFAULT_MENISCUS_CENTER_MODE == "auto"


def test_auto_uses_midline_when_plateau_labels_present():
    mesh, pts = _labelled_tibia()
    np.testing.assert_allclose(
        meniscus_trim_centers(mesh, pts, mode="auto"), meniscus_trim_centers(mesh, pts, mode="midline")
    )


def test_auto_falls_back_to_as_run_on_an_old_reference(caplog):
    """An original tibia_labeled.vtk (no plateau labels) builds exactly what it always built."""
    mesh, pts = _labelled_tibia(with_plateau=False)
    with caplog.at_level("DEBUG"):
        got = meniscus_trim_centers(mesh, pts, mode="auto")
    np.testing.assert_allclose(got, meniscus_trim_centers(mesh, pts, mode="as_run"))
    assert not [r for r in caplog.records if r.levelname in ("WARNING", "ERROR")]


def test_midline_is_one_point_midway_between_plateau_centres():
    mesh, pts = _labelled_tibia()
    med, lat = meniscus_trim_centers(mesh, pts, mode="midline")
    np.testing.assert_allclose(med, lat)
    np.testing.assert_allclose(med, 0.5 * (pts[:2].mean(0) + pts[2:4].mean(0)))


def test_as_run_uses_the_meniscus_center_labels():
    mesh, pts = _labelled_tibia()
    med, lat = meniscus_trim_centers(mesh, pts, mode="as_run")
    np.testing.assert_allclose(med, pts[4])
    np.testing.assert_allclose(lat, pts[5])


def test_as_run_works_on_an_old_reference_without_plateau_labels():
    mesh, pts = _labelled_tibia(with_plateau=False)
    meniscus_trim_centers(mesh, pts, mode="as_run")


def test_midline_without_plateau_labels_raises():
    mesh, pts = _labelled_tibia(with_plateau=False)
    with pytest.raises(ValueError, match="tibia_labeled_v2"):
        meniscus_trim_centers(mesh, pts, mode="midline")


def test_unknown_mode_raises():
    mesh, pts = _labelled_tibia()
    with pytest.raises(ValueError, match="meniscus_center_mode"):
        meniscus_trim_centers(mesh, pts, mode="intended")
