"""Phase 6C: Apply marker mapping to Rajagopal model for AddBiomechanics scaling.

Reads the MarkerSet from the scaled Rajagopal model, renames 42 markers to
match the Smith2019 TRC naming convention, creates 4 new markers (R.GTR,
L.GTR, R.Clavicle, L.Clavicle), and saves the MarkerSet as a standalone XML.

Usage:
    conda run -n comak python scripts/phase6_rajagopal_audit/apply_marker_mapping.py

Output:
    tests/fixtures/osim_models/rajagopal/rajagopal_smith2019_marker_set.xml
"""

import json
import os
import xml.etree.ElementTree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

INPUT_MODEL = os.path.join(
    REPO_ROOT,
    "tests/fixtures/osim_models/rajagopal/RajagopalLaiUhlrich2023_scaled_to_smith2019.osim",
)
MAPPING_PATH = os.path.join(SCRIPT_DIR, "marker_mapping.json")
OUTPUT_MARKERSET = os.path.join(
    REPO_ROOT,
    "tests/fixtures/osim_models/rajagopal/rajagopal_smith2019_marker_set.xml",
)


def load_mapping(path):
    with open(path) as f:
        return json.load(f)


def find_marker(markers_objects, name):
    """Find a Marker element by name attribute."""
    for marker in markers_objects.findall("Marker"):
        if marker.get("name") == name:
            return marker
    return None


def get_marker_location(marker):
    """Get location as list of floats from a Marker element."""
    loc_text = marker.find("location").text
    return [float(x) for x in loc_text.split()]


def create_marker_element(name, parent_frame, location, fixed="false"):
    """Create a new Marker XML element."""
    marker = ET.Element("Marker", name=name)

    comment1 = ET.Comment(
        "Path to a Component that satisfies the Socket 'parent_frame' of type "
        "PhysicalFrame (description: The frame to which this station is fixed.)."
    )
    marker.append(comment1)
    socket = ET.SubElement(marker, "socket_parent_frame")
    socket.text = parent_frame

    comment2 = ET.Comment("The fixed location of the station expressed in its parent frame.")
    marker.append(comment2)
    loc = ET.SubElement(marker, "location")
    loc.text = " ".join(f"{v}" for v in location)

    comment3 = ET.Comment(
        "Flag (true or false) specifying whether the marker is fixed in its parent "
        "frame during the marker placement step of scaling.  If false, the marker is "
        "free to move within its parent Frame to match its experimental counterpart."
    )
    marker.append(comment3)
    fixed_el = ET.SubElement(marker, "fixed")
    fixed_el.text = fixed

    return marker


def compute_gtr_location(model_path, side):
    """Compute greater trochanter marker location in the femur frame.

    The GTR is ~7.5% wider than the ASIS in global space. Since the femur
    frame origin is at the HJC (already lateral from pelvis center), the
    femur-frame offset = (ASIS_lateral * 1.075) - HJC_lateral.

    Uses OpenSim API to read the hip joint and ASIS positions from the model.
    """
    import opensim as osim

    m = osim.Model(model_path)
    m.initSystem()

    # ASIS lateral distance from pelvis center
    ms = m.getMarkerSet()
    rasi = ms.get(ms.getIndex("RASI"))
    asis_z = abs(rasi.get_location()[2])

    # HJC lateral distance from pelvis center (hip joint location_in_parent Z)
    hip = m.getJointSet().get(f"hip_{side.lower()}")
    hjc_z = abs(hip.get_frames(0).get_translation()[2])

    # GTR in global space is 7.5% wider than ASIS; convert to femur-local
    gtr_global_z = asis_z * 1.075
    gtr_femur_z = gtr_global_z - hjc_z

    if side == "R":
        return [0.0, 0.0, gtr_femur_z]
    else:
        return [0.0, 0.0, -gtr_femur_z]


def compute_clavicle_location(markers_objects, side):
    """Compute R.Clavicle or L.Clavicle location on the torso.

    Places clavicle markers symmetric about the torso midline (Z=0) at
    the same X,Y as the existing CLAV marker. The Z half-spacing is
    derived from the Smith2019 static1.trc R/L Clavicle marker spacing
    (~57.16mm apart → 28.58mm half-spacing → 0.02858m).
    """
    clav = find_marker(markers_objects, "CLAV")
    loc = get_marker_location(clav)
    half_spacing_z = 0.02858  # meters, from static1.trc
    if side == "R":
        return [loc[0], loc[1], half_spacing_z]
    else:
        return [loc[0], loc[1], -half_spacing_z]


def apply_mapping(tree, mapping_data, model_path):
    """Apply the marker mapping to the model XML tree.

    For 'rename' actions: rename the Rajagopal marker to the TRC name.
    For 'create' actions: add a new marker at a computed position.
    """
    # Find the MarkerSet
    root = tree.getroot()
    marker_set = root.find(".//MarkerSet")
    if marker_set is None:
        raise ValueError("No MarkerSet found in model")

    # Rename the MarkerSet
    marker_set.set("name", "smith2019")

    objects = marker_set.find("objects")
    mapping = mapping_data["mapping"]

    renamed = []
    created = []
    skipped = []

    # Pre-compute created marker locations BEFORE any renames (uses original names)
    create_entries = {}
    for trc_name, info in mapping.items():
        if info["action"] == "create":
            body = info["body"]
            if trc_name in ("R.GTR", "L.GTR"):
                side = trc_name[0]
                location = compute_gtr_location(model_path, side)
            elif trc_name in ("R.Clavicle", "L.Clavicle"):
                side = trc_name[0]
                location = compute_clavicle_location(objects, side)
            else:
                skipped.append((trc_name, None, "unknown create marker"))
                continue
            create_entries[trc_name] = (body, location)

    # Collect Rajagopal names that are being renamed
    renamed_names = {
        info["rajagopal"]
        for info in mapping.values()
        if info["action"] == "rename" and info["rajagopal"]
    }

    # Remove markers not in the mapping (unused Rajagopal-only markers)
    removed = []
    for marker in list(objects.findall("Marker")):
        name = marker.get("name")
        if name not in renamed_names:
            objects.remove(marker)
            removed.append(name)

    # Apply renames
    for trc_name, info in mapping.items():
        if info["action"] != "rename":
            continue
        rajagopal_name = info["rajagopal"]
        marker = find_marker(objects, rajagopal_name)
        if marker is None:
            skipped.append((trc_name, rajagopal_name, "not found in model"))
            continue
        marker.set("name", trc_name)
        renamed.append((trc_name, rajagopal_name))

    # Apply creates
    for trc_name, (body, location) in create_entries.items():
        parent_frame = f"/bodyset/{body}"
        new_marker = create_marker_element(trc_name, parent_frame, location)
        objects.append(new_marker)
        created.append((trc_name, body, location))

    return renamed, created, skipped, removed


def main():
    print("Phase 6C: Apply marker mapping to Rajagopal model")
    print("=" * 60)

    # Load
    mapping_data = load_mapping(MAPPING_PATH)
    tree = ET.parse(INPUT_MODEL)

    print(f"Input model:   {INPUT_MODEL}")
    print(f"Marker mapping: {MAPPING_PATH}")

    # Apply mapping
    renamed, created, skipped, removed = apply_mapping(tree, mapping_data, INPUT_MODEL)

    # Report
    print(f"\nRemoved {len(removed)} unused Rajagopal markers:")
    for name in removed:
        print(f"  {name}")

    print(f"\nRenamed {len(renamed)} markers:")
    for trc_name, raj_name in renamed:
        print(f"  {raj_name:20s} -> {trc_name}")

    print(f"\nCreated {len(created)} markers:")
    for trc_name, body, loc in created:
        loc_str = " ".join(f"{v:.6f}" for v in loc)
        print(f"  {trc_name:20s} on {body:15s} at [{loc_str}]")

    if skipped:
        print(f"\nSkipped {len(skipped)} markers:")
        for trc_name, raj_name, reason in skipped:
            print(f"  {trc_name} ({raj_name}): {reason}")

    # Extract just the MarkerSet and save as standalone XML
    marker_set = tree.getroot().find(".//MarkerSet")
    n_final = len(marker_set.find("objects").findall("Marker"))
    print(f"\nTotal markers in output: {n_final}")

    ms_tree = ET.ElementTree(marker_set)
    ET.indent(ms_tree, space="\t")
    ms_tree.write(OUTPUT_MARKERSET, xml_declaration=True, encoding="UTF-8")
    print(f"\nOutput MarkerSet: {OUTPUT_MARKERSET}")


if __name__ == "__main__":
    main()
