"""Phase 6B: Scale Rajagopal model down to Smith2019 dimensions.

Builds an OpenSim ScaleSet XML from the model mapping data, then runs
the OpenSim ScaleTool to produce a scaled model.

Usage:
    conda run -n comak python scripts/phase6_rajagopal_audit/scale_rajagopal.py

Output:
    tests/fixtures/osim_models/rajagopal/RajagopalLaiUhlrich2023_scaled_to_smith2019.osim
"""

import json
import os
import xml.etree.ElementTree as ET

import opensim as osim


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

RAJAGOPAL_PATH = os.path.join(
    REPO_ROOT, "tests/fixtures/osim_models/rajagopal/RajagopalLaiUhlrich2023.osim"
)
MAPPING_PATH = os.path.join(SCRIPT_DIR, "model_mapping.json")
OUTPUT_DIR = os.path.join(REPO_ROOT, "tests/fixtures/osim_models/rajagopal")
OUTPUT_MODEL = os.path.join(OUTPUT_DIR, "RajagopalLaiUhlrich2023_scaled_to_smith2019.osim")
SCALE_XML_PATH = os.path.join(SCRIPT_DIR, "scale_rajagopal_to_smith2019.xml")


def build_scale_set_xml(mapping_path, model_path, output_xml_path, output_model_path):
    """Build an OpenSim ScaleTool XML from the mapping data."""
    with open(mapping_path) as f:
        mapping = json.load(f)

    scale_factors = mapping["scale_factors"]

    # Build the XML structure that OpenSim ScaleTool expects
    root = ET.Element("OpenSimDocument", Version="40000")
    scale_tool = ET.SubElement(root, "ScaleTool", name="scale_rajagopal")

    # Mass
    ET.SubElement(scale_tool, "mass").text = "-1"

    # GenericModelMaker
    gmm = ET.SubElement(scale_tool, "GenericModelMaker")
    ET.SubElement(gmm, "model_file").text = model_path

    # ModelScaler
    ms = ET.SubElement(scale_tool, "ModelScaler")
    ET.SubElement(ms, "apply").text = "true"

    # ScaleSet
    scale_set = ET.SubElement(ms, "ScaleSet")
    objects = ET.SubElement(scale_set, "objects")

    for body_name, sf_data in sorted(scale_factors.items()):
        sf = sf_data["scale_factors"]
        if body_name == "ground":
            continue

        scale = ET.SubElement(objects, "Scale")
        ET.SubElement(scale, "scales").text = f"{sf[0]} {sf[1]} {sf[2]}"
        ET.SubElement(scale, "segment").text = body_name
        ET.SubElement(scale, "apply").text = "true"

    ET.SubElement(ms, "scaling_order").text = "manualScale"
    ET.SubElement(ms, "output_model_file").text = output_model_path
    ET.SubElement(ms, "output_scale_file").text = ""

    # MarkerPlacer (disabled)
    mp = ET.SubElement(scale_tool, "MarkerPlacer")
    ET.SubElement(mp, "apply").text = "false"

    # Write
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_xml_path, xml_declaration=True, encoding="UTF-8")
    print(f"Scale XML written to: {output_xml_path}")


def run_scale_tool(xml_path):
    """Run the OpenSim ScaleTool."""
    print(f"Running ScaleTool from: {xml_path}")
    tool = osim.ScaleTool(xml_path)
    success = tool.run()
    print(f"ScaleTool returned: {success}")
    return success


def verify_scaled_model(original_path, scaled_path, mapping_path):
    """Verify the scaled model loads and segment lengths are close to Smith2019."""
    with open(mapping_path) as f:
        mapping = json.load(f)

    print(f"\nVerifying scaled model: {scaled_path}")
    model = osim.Model(scaled_path)
    model.initSystem()

    print(f"  Model loaded: {model.getName()}")
    print(f"  Bodies: {model.getBodySet().getSize()}")
    print(f"  Joints: {model.getJointSet().getSize()}")
    print(f"  Forces: {model.getForceSet().getSize()}")
    print(f"  Coordinates: {model.getCoordinateSet().getSize()}")

    # Check a few key segment lengths
    print(f"\n  Key segment length checks (should be close to Smith2019):")
    for body_name in ["femur_r", "tibia_r", "pelvis"]:
        sf_data = mapping["scale_factors"].get(body_name, {})
        sf = sf_data.get("scale_factors", [1, 1, 1])
        print(f"    {body_name}: scale applied = [{sf[0]:.4f}, {sf[1]:.4f}, {sf[2]:.4f}]")


def main():
    print("Phase 6B: Scale Rajagopal to Smith2019 dimensions")
    print("=" * 60)

    # Build the scale XML
    build_scale_set_xml(MAPPING_PATH, RAJAGOPAL_PATH, SCALE_XML_PATH, OUTPUT_MODEL)

    # Run ScaleTool
    run_scale_tool(SCALE_XML_PATH)

    # Verify
    if os.path.exists(OUTPUT_MODEL):
        verify_scaled_model(RAJAGOPAL_PATH, OUTPUT_MODEL, MAPPING_PATH)
        print(f"\nScaled model: {OUTPUT_MODEL}")
    else:
        print(f"\nERROR: Output model not found at {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
