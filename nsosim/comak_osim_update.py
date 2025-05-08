import time 
import numpy as np
import xml.etree.ElementTree as ET


# helper to get fem/tib offsets
def get_femur_r_offset(root):
    JointSet = root.find('JointSet')[0]
    femur_r_offset = np.fromstring(
        JointSet.findall("./WeldJoint[@name='femur_femur_distal_r']/frames/PhysicalOffsetFrame/translation")[0].text,
        dtype=float,sep=' ')
    return femur_r_offset

def get_tibia_r_offset(root):
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

# Update muscle attachments. 
def update_muscle_attachments(
        root,
        muscle_df,
        fem_interpolated_pts_osim,
        tib_interpolated_pts_osim,
        pat_interpolated_pts_osim,
):
    tibia_r_offset = get_tibia_r_offset(root)
    femur_r_offset = get_femur_r_offset(root)
    
    # update muscle attachments
    # NOTE: only muscle attachments on patella are updated currently. 
    ForceSet = root.find('ForceSet')[0]
    for i in muscle_df.index:
        for j in range(len(muscle_df.node[i])):
            if muscle_df.node[i][j] is not None:
                idx_orig = muscle_df.node[i][j]
                if muscle_df.segment[i][j] == 'femur_r':
                    # idx_subject = subject_idx_4_ref_pts_fem[]
                    p = fem_interpolated_pts_osim[idx_orig,:]
                    p = p + femur_r_offset
                elif muscle_df.segment[i][j] == 'tibia_r':
                    # idx_subject = subject_idx_4_ref_pts_tib[idx_orig]
                    # p = subject_tib.point_coords[idx_subject,:]
                    p = tib_interpolated_pts_osim[idx_orig,:]
                    p = p + tibia_r_offset
                elif muscle_df.segment[i][j] == 'patella_r':
                    # idx_subject = subject_idx_4_ref_pts_pat[idx_orig]
                    # p = subject_pat.point_coords[idx_subject,:]
                    p = pat_interpolated_pts_osim[idx_orig,:]
                    
                ForceSet.findall("./Millard2012EquilibriumMuscle[@name='%s']/GeometryPath/" %  (muscle_df.name[i]) + \
                                "PathPointSet/objects/PathPoint[@name='%s-P%d']/location" % \
                                (muscle_df.name[i],j+1))[0].text = ' '.join(map(str,p))

# update ligament attachments. 
def update_ligament_attachments(
    root,
    ligament_df,
    tib_mesh_osim,
    fem_interpolated_pts_osim,
    tib_interpolated_pts_osim,
    pat_interpolated_pts_osim
):
    tibia_r_offset = get_tibia_r_offset(root)
    femur_r_offset = get_femur_r_offset(root)
    
    ForceSet = root.find('ForceSet')[0]

    # Get ML and AP sizes/directions from subject bone
    tibia_ml = tib_mesh_osim.point_coords[:,2].max() - tib_mesh_osim.point_coords[:,2].min()
    tibia_ap = tib_mesh_osim.point_coords[:,0].max() - tib_mesh_osim.point_coords[:,0].min()

    # update ligament attachments using 
    for i in ligament_df.index:
        for j in range(len(ligament_df.node[i])):
            if ligament_df.node[i][j] is not None:
                idx_orig = ligament_df.node[i][j]
                if ligament_df.segment[i][j] == 'femur_r':
                    # idx_subject = subject_idx_4_ref_pts_fem[]
                    p = fem_interpolated_pts_osim[idx_orig,:]
                    p = p + femur_r_offset
                elif ligament_df.segment[i][j] == 'tibia_r':
                    # idx_subject = subject_idx_4_ref_pts_tib[idx_orig]
                    # p = subject_tib.point_coords[idx_subject,:]
                    p = tib_interpolated_pts_osim[idx_orig,:]
                    p = p + tibia_r_offset
                elif ligament_df.segment[i][j] == 'patella_r':
                    # idx_subject = subject_idx_4_ref_pts_pat[idx_orig]
                    # p = subject_pat.point_coords[idx_subject,:]
                    p = pat_interpolated_pts_osim[idx_orig,:]
                elif ligament_df.segment[i][j] == 'femur_distal_r':
                    # idx_subject = subject_idx_4_ref_pts_fem[idx_orig]
                    # p = subject_fem.point_coords[idx_subject,:]
                    p = fem_interpolated_pts_osim[idx_orig,:]
                elif ligament_df.segment[i][j] == 'tibia_proximal_r':
                    # idx_subject = subject_idx_4_ref_pts_tib[idx_orig]
                    # p = subject_tib.point_coords[idx_subject,:]
                    p = tib_interpolated_pts_osim[idx_orig,:]
                    # points that are attached to fibula are shifted out, since SSM has no fibula
                    if ligament_df['shift'][i]:
                        p = p + np.multiply(ligament_df['shift'][i][j],np.array([tibia_ap,0,tibia_ml]))
            
                ForceSet.findall("./Blankevoort1991Ligament[@name='%s']/GeometryPath/" %  (ligament_df.name[i]) + \
                                "PathPointSet/objects/PathPoint[@name='%s-P%d']/location" % \
                                (ligament_df.name[i],j+1))[0].text = ' '.join(map(str,p))

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

def update_ligament_stiffness(path_model, ligament, stiffness):
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)) # keep comments
    tree = ET.parse(path_model, parser)
    root = tree.getroot()[0]
    
    ForceSet = root.find('ForceSet')[0]
    ForceSet.findall(f"./Blankevoort1991Ligament[@name='{ligament}']/linear_stiffness")[0].text = str(int(stiffness))
    
    tree.write(path_model, encoding='utf8',method='xml')