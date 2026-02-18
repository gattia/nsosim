"""Specialized ellipsoid fitting for patella wrap surfaces."""

import numpy as np

from .main import wrap_surface
from .utils import create_ellipsoid_polydata


def label_patella_within_wrap_extents(patella_mesh, wrap_surface_mesh):
    """
    Labels patella mesh points as being within the x, y, and z extents of the wrap surface.
    Adds three arrays to the mesh: 'within_x_ellipse', 'within_y_ellipse', 'within_z_ellipse'.
    Returns the patella mesh with these arrays assigned.
    """
    # Find the extent of the wrap surface in x, y, and z
    max_x_ellipse = np.max(wrap_surface_mesh.point_coords[:, 0])
    min_x_ellipse = np.min(wrap_surface_mesh.point_coords[:, 0])
    max_y_ellipse = np.max(wrap_surface_mesh.point_coords[:, 1])
    min_y_ellipse = np.min(wrap_surface_mesh.point_coords[:, 1])
    max_z_ellipse = np.max(wrap_surface_mesh.point_coords[:, 2])
    min_z_ellipse = np.min(wrap_surface_mesh.point_coords[:, 2])

    # Label patella points within the x extent of the wrap
    patella_points_within_x = (
        (patella_mesh.point_coords[:, 0] >= min_x_ellipse)
        & (patella_mesh.point_coords[:, 0] <= max_x_ellipse)
    ).astype(int)
    patella_mesh["within_x_ellipse"] = patella_points_within_x

    # Label patella points within the y extent of the wrap
    patella_points_within_y = (
        (patella_mesh.point_coords[:, 1] >= min_y_ellipse)
        & (patella_mesh.point_coords[:, 1] <= max_y_ellipse)
    ).astype(int)
    patella_mesh["within_y_ellipse"] = patella_points_within_y

    # Label patella points within the z extent of the wrap
    patella_points_within_z = (
        (patella_mesh.point_coords[:, 2] >= min_z_ellipse)
        & (patella_mesh.point_coords[:, 2] <= max_z_ellipse)
    ).astype(int)
    patella_mesh["within_z_ellipse"] = patella_points_within_z

    return patella_mesh


def compute_ellipsoid_parameters_from_labeled_mesh(
    labeled_mesh,
    x_axis_label="within_x_ellipse",
    y_axis_label="within_y_ellipse",
    z_axis_label="within_z_ellipse",
):
    """
    Computes ellipsoid parameters (center and radii) from a labeled mesh.
    Uses the points labeled as 'within_x_ellipse', 'within_y_ellipse', 'within_z_ellipse'.

    Parameters:
    -----------
    labeled_mesh : mskt.mesh.Mesh
        Mesh with arrays 'within_x_ellipse', 'within_y_ellipse', 'within_z_ellipse'

    Returns:
    --------
    dict : Dictionary containing 'center' (x,y,z) and 'radii' (x,y,z) for the ellipsoid
    """
    # Get points within each axis extent
    patella_points_within_x = np.where(labeled_mesh[x_axis_label])[0]
    patella_points_within_y = np.where(labeled_mesh[y_axis_label])[0]
    patella_points_within_z = np.where(labeled_mesh[z_axis_label])[0]

    # Find the extent of points within each axis
    max_x_ellipse = np.max(labeled_mesh.point_coords[patella_points_within_x, 0])
    min_x_ellipse = np.min(labeled_mesh.point_coords[patella_points_within_x, 0])
    max_y_ellipse = np.max(labeled_mesh.point_coords[patella_points_within_y, 1])
    min_y_ellipse = np.min(labeled_mesh.point_coords[patella_points_within_y, 1])
    max_z_ellipse = np.max(labeled_mesh.point_coords[patella_points_within_z, 2])
    min_z_ellipse = np.min(labeled_mesh.point_coords[patella_points_within_z, 2])

    # Get the center for each axis as the middle of the range
    center_x = (max_x_ellipse + min_x_ellipse) / 2
    center_y = (max_y_ellipse + min_y_ellipse) / 2
    center_z = (max_z_ellipse + min_z_ellipse) / 2

    # Calculate sizes (full extent in each direction)
    size_x = max_x_ellipse - min_x_ellipse
    size_y = max_y_ellipse - min_y_ellipse
    size_z = max_z_ellipse - min_z_ellipse

    # Radii are 50% of the size (half the extent)
    radius_x = size_x / 2
    radius_y = size_y / 2
    radius_z = size_z / 2

    return {
        "center": np.array([center_x, center_y, center_z]),
        "radii": np.array([radius_x, radius_y, radius_z]),
    }


class PatellaFitter:
    def __init__(
        self,
        patella_mesh,
        x_axis_label="within_x_ellipse",
        y_axis_label="within_y_ellipse",
        z_axis_label="within_z_ellipse",
    ):
        self.patella_mesh = patella_mesh
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.z_axis_label = z_axis_label
        self._fitted_params = None

    def fit(self):
        params = compute_ellipsoid_parameters_from_labeled_mesh(
            self.patella_mesh, self.x_axis_label, self.y_axis_label, self.z_axis_label
        )

        self._fitted_params = params

    @property
    def fitted_params(self):
        return self._fitted_params

    @property
    def wrap_params(self):

        return wrap_surface(
            name=None,
            body=None,
            type_="WrapEllipsoid",
            xyz_body_rotation=[0, 0, 0],
            translation=self.fitted_params["center"],
            radius=None,
            length=None,
            dimensions=self.fitted_params["radii"],
        )

    def create_ellipsoid_from_fitted_params(self):
        # unpack wrap_params to be a dictionary of values

        return create_ellipsoid_polydata(
            wrap_params=self.wrap_params.to_dict(),
        )
