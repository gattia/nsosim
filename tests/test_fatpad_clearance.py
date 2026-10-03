"""Fat-pad patella clearance: which way the clearance rays point.

The fat pad is the exposed anterior femur dilated along its stored (outward)
normals, capped by a ray-cast clearance to the patella. ``auto_oriented`` (how the
Paper-1 models were built) casts those rays along normals pymskt recomputes with
VTK auto-orientation, which on an open patch facing +x turns every normal to -x,
so the rays miss and the cap never engages. ``dilation_normals`` casts the same
rays along the dilation normals. These tests reproduce the flip on a flat patch.
"""

import numpy as np
import pytest
import pyvista as pv
from pymskt.mesh import Mesh

from nsosim import articular_surfaces as AS

GAP_MM = 5.0


def _patch():
    """A flat open patch in the x = 0 plane whose stored normals point +x (outward)."""
    patch = pv.Plane(center=(0, 0, 0), direction=(1, 0, 0), i_size=10, j_size=10,
                     i_resolution=10, j_resolution=10).triangulate()
    patch.point_data["Normals"] = np.tile([1.0, 0.0, 0.0], (patch.n_points, 1))
    patch.point_data.active_normals_name = "Normals"
    return Mesh(patch)


def _target():
    """A larger plane GAP_MM in front of the patch -- the patella."""
    return pv.Plane(center=(GAP_MM, 0, 0), direction=(-1, 0, 0), i_size=30, j_size=30,
                    i_resolution=6, j_resolution=6).triangulate()


def test_auto_oriented_rays_miss_a_patch_facing_the_target():
    clearance = AS._forward_clearance_to_targets(_patch(), [_target()], ray_cast_length=10.0)
    assert np.isinf(clearance).all()


def test_dilation_normal_rays_measure_the_gap():
    patch = _patch()
    clearance = AS._forward_clearance_to_targets(
        patch, [_target()], ray_cast_length=10.0, directions=np.asarray(patch.point_normals)
    )
    np.testing.assert_allclose(clearance, GAP_MM, atol=1e-9)


def test_helper_reproduces_pymskt_along_pymskts_own_normals():
    """Only the direction differs between the two modes: same rays, same distances."""
    from pymskt.mesh.utils import get_surface_normals

    patch, target = _patch(), _target()
    normals = get_surface_normals(patch).GetOutput().GetPointData().GetNormals()
    normals = np.array([normals.GetTuple(i) for i in range(patch.n_points)])
    probe = _patch()
    probe.calc_distance_to_other_mesh(list_other_meshes=[target], ray_cast_length=10.0,
                                      percent_ray_length_opposite_direction=0.1, name="d")
    np.testing.assert_array_equal(
        AS._ray_distances_along(np.asarray(patch.points), target, normals, 10.0, 0.1),
        np.asarray(probe.get_scalar("d")),
    )


@pytest.mark.parametrize("rays, capped", [("auto_oriented", False), ("dilation_normals", True)])
def test_dilation_is_capped_at_the_target_only_with_dilation_normals(rays, capped):
    target = _target()
    _, actual = AS.dilate_mesh_with_profile_and_clearance(
        _patch(), target, target, base_mm=5.5, top_mm=6.0, ray_cast_length=10.0,
        reference_axis_filter=None, clearance_rays=rays,
    )
    if capped:
        np.testing.assert_allclose(actual, GAP_MM, atol=1e-9)
    else:
        assert actual.min() >= 5.5  # the profile, uncapped: the surface passes the target


def test_unknown_clearance_rays_is_refused():
    with pytest.raises(ValueError, match="clearance_rays"):
        AS.dilate_mesh_with_profile_and_clearance(_patch(), _target(), _target(),
                                                  clearance_rays="closest_point")
