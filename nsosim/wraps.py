import numpy as np
from scipy.optimize import leastsq


class wrap_surface:
    """
    Represents a wrapping surface object in an OpenSim model.

    This class stores parameters that define a muscle or ligament wrapping surface,
    such as its name, the body it's attached to, its geometric type (e.g.,
    'WrapCylinder', 'WrapEllipsoid'), orientation, translation, and dimensions.

    Attributes:
        name (str): The name of the wrap surface.
        body (str): The name of the body segment this wrap surface is attached to
            (e.g., 'femur_r', 'tibia_proximal_r').
        type_ (str): The type of wrapping object (e.g., 'WrapCylinder',
            'WrapEllipsoid').
        xyz_body_rotation (numpy.ndarray): A 3-element array representing the
            Euler angles (in radians, typically X-Y-Z order) for the orientation
            of the wrap surface relative to its parent body.
        translation (numpy.ndarray): A 3-element array representing the x, y, z
            translation of the wrap surface's origin relative to its parent body.
        radius (float, optional): The radius of the wrap surface, applicable if
            `type_` is 'WrapCylinder'.
        length (float, optional): The length of the wrap surface, applicable if
            `type_` is 'WrapCylinder'.
        dimensions (numpy.ndarray, optional): A 3-element array representing the
            dimensions (e.g., radii along x, y, z axes) of the wrap surface,
            applicable if `type_` is 'WrapEllipsoid'.
    """

    def __init__(
        self, name, body, type_, xyz_body_rotation, translation, radius, length, dimensions
    ):
        self.name = name
        self.body = body
        self.type_ = type_
        self.xyz_body_rotation = xyz_body_rotation
        self.translation = translation
        self.radius = radius
        self.length = length
        self.dimensions = dimensions


def _create_knee_ext_cylinder_wrap(name, pts_indices, fem_interpolated_pts_osim, ME, LE):
    """
    Helper function to create a 'WrapCylinder' for knee extensors on the right femur.

    The cylinder is defined based on a set of interpolated points on the femur
    and the positions of the medial (ME) and lateral (LE) epicondyles.
    The cylinder's axis is implicitly aligned with the OpenSim model's axes,
    and its radius and length are derived from the provided points.

    Args:
        name (str): The name to assign to the wrap surface.
        pts_indices (list or numpy.ndarray): Indices into `fem_interpolated_pts_osim`
            that define the relevant points for this wrap object.
        fem_interpolated_pts_osim (numpy.ndarray): Nx3 array of point coordinates on
            the femur in the OpenSim coordinate system.
        ME (numpy.ndarray): 3-element array for the medial epicondyle coordinates.
        LE (numpy.ndarray): 3-element array for the lateral epicondyle coordinates.

    Returns:
        wrap_surface: A `wrap_surface` object representing the defined cylinder.
    """
    pts = fem_interpolated_pts_osim[pts_indices, :]
    h = 2 * np.abs(ME[2] - LE[2])
    x0 = pts.mean(axis=0)
    r = pts[:, 0].max() - x0[0]
    wrap_ = wrap_surface(
        name=name,
        body="femur_r",
        type_="WrapCylinder",
        xyz_body_rotation=np.array([0, 0, 0]),
        translation=x0,
        radius=r,
        length=h,
        dimensions=None,
    )
    return wrap_


def knee_ext_r_wrap(pts_indices, fem_interpolated_pts_osim, ME, LE):
    """
    Creates a 'WrapCylinder' for the knee extensors (KnExt_at_fem_r) on the right femur.

    The cylinder is defined based on a set of interpolated points on the femur
    and the positions of the medial (ME) and lateral (LE) epicondyles.
    The cylinder's axis is implicitly aligned with the OpenSim model's axes,
    and its radius and length are derived from the provided points.

    Args:
        pts_indices (list or numpy.ndarray): Indices into `fem_interpolated_pts_osim`
            that define the relevant points for this wrap object.
        fem_interpolated_pts_osim (numpy.ndarray): Nx3 array of point coordinates on
            the femur in the OpenSim coordinate system.
        ME (numpy.ndarray): 3-element array for the medial epicondyle coordinates.
        LE (numpy.ndarray): 3-element array for the lateral epicondyle coordinates.

    Returns:
        wrap_surface: A `wrap_surface` object representing the defined cylinder.
    """
    return _create_knee_ext_cylinder_wrap(
        "KnExt_at_fem_r", pts_indices, fem_interpolated_pts_osim, ME, LE
    )


def knee_ext_vasint_r_wrap(pts_indices, fem_interpolated_pts_osim, ME, LE):
    """
    Creates a 'WrapCylinder' for the vastus intermedius (KnExt_vasint_at_fem_r)
    on the right femur.

    Similar to `knee_ext_r_wrap`, this defines a cylinder based on interpolated
    femoral points and epicondyle positions. It's intended for a specific part
    of the knee extensor mechanism.

    Args:
        pts_indices (list or numpy.ndarray): Indices into `fem_interpolated_pts_osim`.
        fem_interpolated_pts_osim (numpy.ndarray): Nx3 array of femur points (OSIM space).
        ME (numpy.ndarray): Medial epicondyle coordinates.
        LE (numpy.ndarray): Lateral epicondyle coordinates.

    Returns:
        wrap_surface: A `wrap_surface` object for the vastus intermedius cylinder.
    """
    return _create_knee_ext_cylinder_wrap(
        "KnExt_vasint_at_fem_r", pts_indices, fem_interpolated_pts_osim, ME, LE
    )


def gastroc_condyles_r_wrap(pts_indices, fem_interpolated_pts_osim, ME, LE):
    """
    Creates a 'WrapEllipsoid' for the gastrocnemius attachment at the femoral
    condyles (Gastroc_at_Condyles_r) on the right femur.

    The ellipsoid is defined based on a set of interpolated points on the femur
    and the medial (ME) and lateral (LE) epicondyles. Its position, dimensions,
    and orientation are calculated from these points.

    Args:
        pts_indices (list or numpy.ndarray): Indices into `fem_interpolated_pts_osim`.
        fem_interpolated_pts_osim (numpy.ndarray): Nx3 array of femur points (OSIM space).
        ME (numpy.ndarray): Medial epicondyle coordinates.
        LE (numpy.ndarray): Lateral epicondyle coordinates.

    Returns:
        wrap_surface: A `wrap_surface` object for the gastrocnemius ellipsoid.
    """
    # I_orig = pts_dict['femur']['Gastroc_at_Condyles_r']
    pts = fem_interpolated_pts_osim[pts_indices, :]

    x0 = (ME + LE) / 2
    x0[:2] = pts[6, :2]
    r = np.array([0.0, 0.0, 0.0])
    r[0] = np.abs(pts[3, 0] - pts[2, 0]) / 2
    r[1] = np.abs(pts[5, 1] - pts[4, 1]) / 2
    r[2] = 1.875 * np.abs(ME[2] - LE[2])

    a0 = pts[1, :] - pts[0, :]
    a0 /= np.linalg.norm(a0)
    thx = np.arctan2(a0[1], a0[2])
    a01_ = np.array([[1, 0, 0], [0, np.cos(thx), -np.sin(thx)], [0, np.sin(thx), np.cos(thx)]])
    a01 = a01_ @ a0.T
    thy = np.arctan2(a01[0], a01[2])
    an = np.array([thx, thy, 0])

    wrap_ = wrap_surface(
        name="Gastroc_at_Condyles_r",
        body="femur_r",
        type_="WrapEllipsoid",
        xyz_body_rotation=an,
        translation=x0,
        radius=None,
        length=None,
        dimensions=r,
    )

    return wrap_


def capsule_r_wrap(pts_indices, fem_interpolated_pts_osim, ME, LE):
    """
    Creates a 'WrapCylinder' for the knee joint capsule (Capsule_r) attached
    to the right distal femur (femur_distal_r).

    The cylinder is defined by fitting to a cloud of interpolated points on the
    femur. The medial (ME) and lateral (LE) epicondyles help define its height
    and initial position for the fitting process. The orientation and radius are
    determined by a least-squares fit to the points.

    Args:
        pts_indices (list or numpy.ndarray): Indices into `fem_interpolated_pts_osim`
            representing points on the capsule attachment area.
        fem_interpolated_pts_osim (numpy.ndarray): Nx3 array of femur points (OSIM space).
        ME (numpy.ndarray): Medial epicondyle coordinates.
        LE (numpy.ndarray): Lateral epicondyle coordinates.

    Returns:
        wrap_surface: A `wrap_surface` object for the capsule cylinder.
    """
    # I_orig = pts_dict['femur']['Capsule_r']
    pts = fem_interpolated_pts_osim[pts_indices, :]

    x0 = (ME + LE) / 2
    a0 = (LE - ME) / np.linalg.norm(LE - ME)
    r0 = (np.linalg.norm(pts[:, 0:2] - x0[0:2], axis=1)).mean()
    h = 6 * np.abs(LE[2] - ME[2])
    p0 = [x0[0], x0[1], 0, 0, r0]  # x0[0], x0[1], x rotation, y rotation, radius

    # fit ls cylinder (https://stackoverflow.com/questions/43784618/fit-a-cylinder-to-scattered-3d-xyz-point-data)
    fitfunc = (
        lambda p, x, y, z: (
            -np.cos(p[3]) * (p[0] - x)
            - z * np.cos(p[2]) * np.sin(p[3])
            - np.sin(p[2]) * np.sin(p[3]) * (p[1] - y)
        )
        ** 2
        + (z * np.sin(p[2]) - np.cos(p[2]) * (p[1] - y)) ** 2
    )
    errfunc = lambda p, x, y, z: fitfunc(p, x, y, z) - p[4] ** 2
    p, success = leastsq(errfunc, p0, args=(pts[:, 0], pts[:, 1], pts[:, 2]), maxfev=1000)

    an = np.matmul(
        np.array([[np.cos(p[4]), 0, np.sin(p[4])], [0, 1, 0], [-np.sin(p[4]), 0, np.cos(p[4])]]),
        np.matmul(
            np.array([[1, 0, 0], [0, np.cos(p[3]), np.sin(p[3])], [0, np.sin(p[3]), np.cos(p[3])]]),
            np.array([0, 0, 1]).transpose(),
        ),
    )

    translation = np.array([p[0], p[1], 0])

    wrap_ = wrap_surface(
        name="Capsule_r",
        body="femur_distal_r",
        type_="WrapCylinder",
        xyz_body_rotation=an,
        translation=translation,
        radius=p[4],
        length=h,
        dimensions=None,
    )

    return wrap_


def med_l_r_wrap(pts_indices, tib_interpolated_pts_osim):
    """
    Creates a 'WrapEllipsoid' for the medial collateral ligament (Med_Lig_r)
    on the right proximal tibia (tibia_proximal_r).

    The ellipsoid's dimensions, position, and orientation are determined based
    on a specific set of interpolated points on the tibia.

    Args:
        pts_indices (list or numpy.ndarray): Indices into `tib_interpolated_pts_osim`
            defining the relevant points for this ligament wrap object.
        tib_interpolated_pts_osim (numpy.ndarray): Nx3 array of point coordinates on
            the tibia in the OpenSim coordinate system.

    Returns:
        wrap_surface: A `wrap_surface` object representing the defined ellipsoid
            for the medial ligament.
    """
    # I_orig = pts_dict['tibia']['Med_Lig_r']
    pts = tib_interpolated_pts_osim[pts_indices, :]

    r = np.zeros(3, dtype=float)
    r[0] = np.abs(pts[3, 0] - pts[2, 0]) * 0.7
    r[1] = np.abs(pts[5, 1] - pts[4, 1]) * 1.2
    r[2] = np.abs(pts[1, 2] - pts[0, 2]) * 0.75
    x0n = np.zeros(3, dtype=float)
    x0n[0] = (pts[3, 0] + pts[2, 0]) / 2
    x0n[1] = (pts[5, 1] + pts[4, 1]) / 2 - 0.1 * r[1]
    x0n[2] = (pts[1, 2] + pts[0, 2]) / 2 + 0.4 * r[2]
    a0 = (pts[2, :] - pts[3, :]) / np.linalg.norm((pts[2, :] - pts[3, :]))
    an = np.zeros(3, dtype=float)
    an[0] = 5 * np.pi / 180
    an[1] = -np.arcsin(a0[2])
    an[2] = np.arcsin(a0[1])
    # wrapSurface.loc[4] = ['tibia_proximal_r','Med_Lig_r','WrapEllipsoid',an,x0n,None,None,r]

    wrap_ = wrap_surface(
        name="Med_Lig_r",
        body="tibia_proximal_r",
        type_="WrapEllipsoid",
        xyz_body_rotation=an,
        translation=x0n,
        radius=None,
        length=None,
        dimensions=r,
    )

    return wrap_


def med_ligp_r_wrap(pts_indices, tib_interpolated_pts_osim):
    """
    Creates a 'WrapEllipsoid' for the posterior part of the medial collateral
    ligament (Med_LigP_r) on the right proximal tibia (tibia_proximal_r).

    The ellipsoid's parameters (dimensions, position, orientation) are derived
    from a specified set of interpolated points on the tibia.

    Args:
        pts_indices (list or numpy.ndarray): Indices into `tib_interpolated_pts_osim`.
        tib_interpolated_pts_osim (numpy.ndarray): Nx3 array of tibia points (OSIM space).

    Returns:
        wrap_surface: A `wrap_surface` object for the posterior medial ligament ellipsoid.
    """
    # I_orig = pts_dict['tibia']['Med_LigP_r']
    pts = tib_interpolated_pts_osim[pts_indices, :]

    r = np.zeros(3, dtype=float)
    r[0] = np.linalg.norm(pts[3, :] - pts[2, :]) * 0.8
    r[2] = np.linalg.norm(pts[3, :] - pts[1, :]) * 1.4
    r[1] = r[2]
    a0 = pts[2, :] - pts[3, :]
    x0n = pts[3, :] + a0 * 0.4
    a0 = a0 / np.linalg.norm(a0)
    an = np.zeros(3, dtype=float)
    an[0] = 0
    an[1] = -np.arcsin(a0[2])
    an[2] = np.arcsin(a0[1])

    wrap_ = wrap_surface(
        name="Med_LigP_r",
        body="tibia_proximal_r",
        type_="WrapEllipsoid",
        xyz_body_rotation=an,
        translation=x0n,
        radius=None,
        length=None,
        dimensions=r,
    )

    return wrap_


def pat_ten_wrap(pts_indices, pat_interpolated_pts_osim):
    """
    Creates a 'WrapEllipsoid' for the patellar tendon (PatTen_r) attachment
    on the right patella (patella_r).

    The ellipsoid is defined based on a set of interpolated points on the patella.
    Its position and dimensions are calculated from these points. The orientation
    is assumed to be aligned with the patella's body axes (zero rotation).

    Args:
        pts_indices (list or numpy.ndarray): Indices into `pat_interpolated_pts_osim`
            defining the relevant points for this wrap object.
        pat_interpolated_pts_osim (numpy.ndarray): Nx3 array of point coordinates on
            the patella in the OpenSim coordinate system.

    Returns:
        wrap_surface: A `wrap_surface` object representing the defined ellipsoid
            for the patellar tendon.
    """
    # I_orig = pts_dict['patella']['PatTen_r']
    pts = pat_interpolated_pts_osim[pts_indices, :]

    an = np.zeros(3, dtype=float)
    x0n = np.zeros(3, dtype=float)
    x0n[0] = pts[2:4, 0].mean()
    x0n[1] = pts[4:6, 1].mean()
    x0n[2] = pts[0:2, 2].mean()
    r = np.zeros(3, dtype=float)
    r[0] = np.abs(pts[3, 0] - pts[2, 0]) / 2
    r[1] = np.abs(pts[4, 1] - pts[5, 1]) / 2
    r[2] = np.abs(pts[1, 2] - pts[0, 2]) / 2

    wrap_ = wrap_surface(
        name="PatTen_r",
        body="patella_r",
        type_="WrapEllipsoid",
        xyz_body_rotation=an,
        translation=x0n,
        radius=None,
        length=None,
        dimensions=r,
    )

    return wrap_
