import time 
import numpy as np
import xml.etree.ElementTree as ET
import opensim as osim


# helper to get fem/tib offsets
def get_femur_r_offset(root):
    """
    Retrieves the femur's right side offset from an OpenSim model XML structure.

    Parses the XML tree to find the translation vector of the PhysicalOffsetFrame
    associated with the WeldJoint named 'femur_femur_distal_r'.

    Args:
        root (xml.etree.ElementTree.Element): The root element of the parsed
            OpenSim model XML (typically the <OpenSimDocument> tag or its child <Model>).

    Returns:
        numpy.ndarray: A 1D array of floats representing the x, y, z translation
            offset of the right femur.
    """
    JointSet = root.find('JointSet')[0]
    femur_r_offset = np.fromstring(
        JointSet.findall("./WeldJoint[@name='femur_femur_distal_r']/frames/PhysicalOffsetFrame/translation")[0].text,
        dtype=float,sep=' ')
    return femur_r_offset

def get_tibia_r_offset(root):
    """
    Retrieves the tibia's right side offset from an OpenSim model XML structure.

    Parses the XML tree to find the translation vector of the PhysicalOffsetFrame
    associated with the WeldJoint named 'tibia_tibia_proximal_r'.

    Args:
        root (xml.etree.ElementTree.Element): The root element of the parsed
            OpenSim model XML (typically the <OpenSimDocument> tag or its child <Model>).

    Returns:
        numpy.ndarray: A 1D array of floats representing the x, y, z translation
            offset of the right tibia.
    """
    JointSet = root.find('JointSet')[0]
    tibia_r_offset = np.fromstring(
        JointSet.findall("./WeldJoint[@name='tibia_tibia_proximal_r']/frames/PhysicalOffsetFrame/translation")[0].text,
        dtype=float,sep=' ')
    return tibia_r_offset

# Update geometry / stl files 
def update_comak_bodyset_stl(
        root,
        fem_mesh_filename='femur_nsm_recon_osim.stl',
        fem_cart_filename='femur_articular_surface_osim.stl',
        tib_mesh_filename='tibia_nsm_recon_osim.stl',
        tib_cart_filename='tibia_articular_surface_osim.stl',
        pat_mesh_filename='patella_nsm_recon_osim.stl',
        pat_cart_filename='patella_articular_surface_osim.stl'

    ):
    """
    Updates the mesh file paths for body visualization geometry in an OpenSim model.

    Modifies the <mesh_file> tags within the <BodySet> for femur_distal_r,
    tibia_proximal_r, and patella_r bodies, and their respective cartilage components.
    This primarily affects how the model is visualized, not the contact simulation.

    Args:
        root (xml.etree.ElementTree.Element): The root element of the <Model> tag
            in the OpenSim XML file.
        fem_mesh_filename (str, optional): Filename for the femur bone mesh.
        fem_cart_filename (str, optional): Filename for the femur cartilage mesh.
        tib_mesh_filename (str, optional): Filename for the tibia bone mesh.
        tib_cart_filename (str, optional): Filename for the tibia cartilage mesh.
        pat_mesh_filename (str, optional): Filename for the patella bone mesh.
        pat_cart_filename (str, optional): Filename for the patella cartilage mesh.
    """
    # update body set geometry (used for visualization only)
    BodySet = root.find('BodySet')[0]
    BodySet.findall("./Body[@name='femur_distal_r']/attached_geometry/Mesh[@name='femur_bone']/mesh_file")[0].text = \
       fem_mesh_filename
    BodySet.findall("./Body[@name='femur_distal_r']/attached_geometry/Mesh[@name='femur_cartilage']/mesh_file")[0].text = \
        fem_cart_filename
    BodySet.findall("./Body[@name='tibia_proximal_r']/attached_geometry/Mesh[@name='tibia_bone']/mesh_file")[0].text = \
        tib_mesh_filename
    BodySet.findall("./Body[@name='tibia_proximal_r']/attached_geometry/Mesh[@name='tibia_cartilage']/mesh_file")[0].text = \
        tib_cart_filename
    BodySet.findall("./Body[@name='patella_r']/attached_geometry/Mesh[@name='patella_bone']/mesh_file")[0].text = \
        pat_mesh_filename
    BodySet.findall("./Body[@name='patella_r']/attached_geometry/Mesh[@name='patella_cartilage']/mesh_file")[0].text = \
        pat_cart_filename

def update_comak_contact_geometry_stl(
        root,
        fem_mesh_filename='femur_nsm_recon_osim.stl',
        fem_cart_filename='femur_articular_surface_osim.stl',
        tib_mesh_filename='tibia_nsm_recon_osim.stl',
        tib_cart_filename='tibia_articular_surface_osim.stl',
        pat_mesh_filename='patella_nsm_recon_osim.stl',
        pat_cart_filename='patella_articular_surface_osim.stl'

):
    """
    Updates the mesh file paths for contact geometry in an OpenSim model.

    Modifies the <mesh_file> and <mesh_back_file> tags for Smith2018ContactMesh
    elements associated with femur, tibia, and patella cartilages. These meshes
    are used by the contact algorithm during simulation.

    Args:
        root (xml.etree.ElementTree.Element): The root element of the <Model> tag
            in the OpenSim XML file.
        fem_mesh_filename (str, optional): Filename for the femur bone mesh (backing surface).
        fem_cart_filename (str, optional): Filename for the femur cartilage mesh (contact surface).
        tib_mesh_filename (str, optional): Filename for the tibia bone mesh (backing surface).
        tib_cart_filename (str, optional): Filename for the tibia cartilage mesh (contact surface).
        pat_mesh_filename (str, optional): Filename for the patella bone mesh (backing surface).
        pat_cart_filename (str, optional): Filename for the patella cartilage mesh (contact surface).
    """
    # update contact geometry (used in simulation)
    ContactSet = root.find('ContactGeometrySet')[0]
    ContactSet.findall("./Smith2018ContactMesh[@name='femur_cartilage']/mesh_file")[0].text = \
        fem_cart_filename
    ContactSet.findall("./Smith2018ContactMesh[@name='femur_cartilage']/mesh_back_file")[0].text = \
        fem_mesh_filename
    ContactSet.findall("./Smith2018ContactMesh[@name='tibia_cartilage']/mesh_file")[0].text = \
        tib_cart_filename
    ContactSet.findall("./Smith2018ContactMesh[@name='tibia_cartilage']/mesh_back_file")[0].text = \
        tib_mesh_filename
    ContactSet.findall("./Smith2018ContactMesh[@name='patella_cartilage']/mesh_file")[0].text = \
        pat_cart_filename
    ContactSet.findall("./Smith2018ContactMesh[@name='patella_cartilage']/mesh_back_file")[0].text = \
        pat_mesh_filename

# UPDATE WRAP OBJECTS
def update_wrap_objects(
        root,
        list_results
):
    """
    Updates the properties of wrapping objects in an OpenSim model.

    Iterates through a list of wrap surface objects and updates their
    `xyz_body_rotation`, `translation`, and type-specific properties (radius, length
    for WrapCylinder; dimensions for WrapEllipsoid) in the OpenSim model XML.
    Offsets for femur and tibia wrap objects are adjusted based on their respective
    body offsets in the model.

    Args:
        root (xml.etree.ElementTree.Element): The root element of the <Model> tag
            in the OpenSim XML file.
        list_results (list): A list of `wrap_surface` objects (or similar objects
            with attributes: `body`, `type_`, `name`, `xyz_body_rotation`,
            `translation`, and potentially `radius`, `length`, `dimensions`).
    """
    BodySet = root.find('BodySet')[0]
    
    # update wrapping surfaces
    # get offset for femur and tibia and add to wrap surface translations

    femur_r_offset = get_femur_r_offset(root)
    tibia_r_offset = get_tibia_r_offset(root)
    
    for idx, wrap_surface_ in enumerate(list_results):
        if wrap_surface_.body == 'femur_r':
            wrap_surface_.translation += femur_r_offset
        if wrap_surface_.body == 'tibia_r':
            wrap_surface_.translation += tibia_r_offset
        BodySet.findall("./Body[@name='%s']/WrapObjectSet/objects/%s[@name='%s']/xyz_body_rotation" % \
                        (wrap_surface_.body, wrap_surface_.type_, wrap_surface_.name))[0].text = \
                        ' '.join(map(str, wrap_surface_.xyz_body_rotation))
        BodySet.findall("./Body[@name='%s']/WrapObjectSet/objects/%s[@name='%s']/translation" % \
                        (wrap_surface_.body, wrap_surface_.type_, wrap_surface_.name))[0].text = \
                        ' '.join(map(str, wrap_surface_.translation))
        if wrap_surface_.type_ == 'WrapCylinder':
            BodySet.findall("./Body[@name='%s']/WrapObjectSet/objects/%s[@name='%s']/radius" % \
                            (wrap_surface_.body, wrap_surface_.type_, wrap_surface_.name))[0].text = \
                            str(wrap_surface_.radius)
            BodySet.findall("./Body[@name='%s']/WrapObjectSet/objects/%s[@name='%s']/length" % \
                            (wrap_surface_.body, wrap_surface_.type_, wrap_surface_.name))[0].text = \
                            str(wrap_surface_.length)
        if wrap_surface_.type_ == 'WrapEllipsoid':
            BodySet.findall("./Body[@name='%s']/WrapObjectSet/objects/%s[@name='%s']/dimensions" % \
                            (wrap_surface_.body, wrap_surface_.type_, wrap_surface_.name))[0].text = \
                            ' '.join(map(str, wrap_surface_.dimensions))

# update patella location
def update_patella_location(root, mean_patella):
    """
    Updates the default translations for the patellofemoral (pf_r) joint.

    Modifies the <default_value> for the `pf_tx_r`, `pf_ty_r`, and `pf_tz_r`
    coordinates of the `pf_r` CustomJoint in the OpenSim model XML.

    Args:
        root (xml.etree.ElementTree.Element): The root element of the <Model> tag
            in the OpenSim XML file.
        mean_patella (list or numpy.ndarray): A list or array of three floats
            representing the new default x, y, z translations for the patella.
    """
    # update patella location using ET.XMLParser
    # 
    """
    <JointSet name="jointset">
        <CustomJoint name="pf_r">
            <coordinates>
                <Coordinate name="pf_tx_r">
                    <default_value>
                <Coordinate name="pf_ty_r">
                    <default_value>
                <Coordinate name="pf_tz_r">
                    <default_value>
    """

    # UPDATE THE DEFAULT POSITION FOR THE PATELLA

    jointset = root.find('JointSet')[0]
    pf_r = jointset.findall("./CustomJoint[@name='pf_r']")[0]
    pf_r.findall("./coordinates/Coordinate[@name='pf_tx_r']/default_value")[0].text = str(mean_patella[0])
    pf_r.findall("./coordinates/Coordinate[@name='pf_ty_r']/default_value")[0].text = str(mean_patella[1])
    pf_r.findall("./coordinates/Coordinate[@name='pf_tz_r']/default_value")[0].text = str(mean_patella[2])





def update_osim_model(
    path_model,
    list_results,
    muscle_df,
    ligament_df,
    fem_interpolated_pts_osim,
    tib_interpolated_pts_osim,
    pat_interpolated_pts_osim,
    mean_patella,
    tib_mesh_osim,
    new_model_name=f'custom_nsm_model_{time.strftime("%b_%d_%Y")}'
):
    """
    Updates an entire OpenSim model XML structure with new geometry and attachments.

    This function orchestrates several updates:
    1.  Sets a new model name.
    2.  Updates BodySet STL file paths for visualization (`update_comak_bodyset_stl`).
    3.  Updates ContactGeometrySet STL file paths for simulation (`update_comak_contact_geometry_stl`).
    4.  Updates wrap object properties (`update_wrap_objects`).
    5.  Updates muscle attachment locations (`update_muscle_attachments`).
    6.  Updates ligament attachment locations (`update_ligament_attachments`).
    7.  Updates the default patella location (`update_patella_location`).

    Args:
        path_model (str): Path to the original OpenSim model (.osim) file.
        list_results (list): List of wrap surface objects for `update_wrap_objects`.
        muscle_df (pandas.DataFrame): DataFrame for `update_muscle_attachments`.
        ligament_df (pandas.DataFrame): DataFrame for `update_ligament_attachments`.
        fem_interpolated_pts_osim (numpy.ndarray): Femur points for attachments.
        tib_interpolated_pts_osim (numpy.ndarray): Tibia points for attachments.
        pat_interpolated_pts_osim (numpy.ndarray): Patella points for attachments.
        mean_patella (list or numpy.ndarray): New patella default position.
        tib_mesh_osim (pymskt.mesh.Mesh): Tibia mesh for ligament adjustments.
        new_model_name (str, optional): Name for the updated model. Defaults to a
            timestamped name.

    Returns:
        xml.etree.ElementTree.ElementTree: The modified XML tree of the OpenSim model.
    """
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)) # keep comments
    tree = ET.parse(path_model, parser)
    root = tree.getroot()[0]
    root.attrib['name'] = new_model_name

    update_comak_bodyset_stl(root)
    update_comak_contact_geometry_stl(root)
    update_wrap_objects(root, list_results)
    update_muscle_attachments(
        root,
        muscle_df,
        fem_interpolated_pts_osim,
        tib_interpolated_pts_osim,
        pat_interpolated_pts_osim,
    )
    update_ligament_attachments(
        root,
        ligament_df,
        tib_mesh_osim,
        fem_interpolated_pts_osim,
        tib_interpolated_pts_osim,
        pat_interpolated_pts_osim
    )

    update_patella_location(root, mean_patella)

    return tree

