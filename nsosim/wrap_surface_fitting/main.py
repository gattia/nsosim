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
    def __init__(self, name, body, type_, xyz_body_rotation, translation, radius, length, dimensions):
        self.name = name
        self.body = body
        self.type_ = type_
        self.xyz_body_rotation = xyz_body_rotation
        self.translation = translation
        self.radius = radius
        self.length = length
        self.dimensions = dimensions