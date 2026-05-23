"""Add missing TRC markers to the Smith2019 base model.

The Smith2019 model already contains 36 of the 46 TRC markers with exact name
matches (R.TH1, R.Knee, etc.). This script adds the 10 remaining markers that
the TRC collection includes but the Smith2019 model does not:

  Medial bony landmarks (mirror the existing lateral markers across body-frame Z=0):
    R.Mkn / L.Mkn    medial femoral condyle      (mirror of R.Knee  / L.Knee)
    R.Man / L.Man    medial malleolus            (mirror of R.Ankle / L.Ankle)
    R.Melb / L.Melb  medial epicondyle humerus   (mirror of R.Elbow / L.Elbow)
    R.MT0 / L.MT0    1st metatarsal head         (mirror of R.MT5   / L.MT5)

  Greater trochanter (computed from HJC + ASIS lateral offset):
    R.GTR / L.GTR

The existing 36 matching markers are left untouched. The 3 extra markers
already on the model (S2, R.SH4, L.TH4) are preserved — AddBiomechanics
will ignore model markers not present in the experimental TRC.

Usage:
    conda run -n comak python scripts/phase6_rajagopal_audit/add_trc_markers_to_smith2019.py
"""

import os

import opensim as osim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

INPUT_MODEL = os.path.join(
    REPO_ROOT,
    "tests/fixtures/osim_models/full_body_healthy_knee.osim",
)
OUTPUT_MODEL = os.path.join(
    REPO_ROOT,
    "tests/fixtures/osim_models/full_body_healthy_knee_trc_markers.osim",
)

# Lateral marker -> medial marker to create (same body, mirror Z)
MIRROR_PAIRS = [
    ("R.Knee", "R.Mkn", "femur_r"),
    ("L.Knee", "L.Mkn", "femur_l"),
    ("R.Ankle", "R.Man", "tibia_r"),
    ("L.Ankle", "L.Man", "tibia_l"),
    ("R.Elbow", "R.Melb", "humerus_r"),
    ("L.Elbow", "L.Melb", "humerus_l"),
    ("R.MT5", "R.MT0", "calcn_r"),
    ("L.MT5", "L.MT0", "calcn_l"),
]

# GTR lateral-offset factor: global GTR is ~7.5% wider than ASIS. Convert to
# femur-local frame (origin at HJC) by subtracting the HJC lateral offset
# from the pelvis midline.
GTR_ASIS_LATERAL_FACTOR = 1.075


def marker_names(marker_set):
    return {marker_set.get(i).getName() for i in range(marker_set.getSize())}


def get_marker(marker_set, name):
    for i in range(marker_set.getSize()):
        m = marker_set.get(i)
        if m.getName() == name:
            return m
    return None


def vec3_to_list(v):
    return [v[0], v[1], v[2]]


def add_marker(model, name, body_name, location):
    body = model.getBodySet().get(body_name)
    marker = osim.Marker(name, body, osim.Vec3(*location))
    marker.set_fixed(False)
    model.addMarker(marker)
    return location


def compute_gtr_location(model, side):
    """GTR in femur frame (origin at HJC), Z-lateral axis."""
    ms = model.getMarkerSet()
    asis = get_marker(ms, f"{side}.ASIS")
    asis_z = abs(asis.get_location()[2])

    hip = model.getJointSet().get(f"hip_{side.lower()}")
    hjc_z = abs(hip.get_frames(0).get_translation()[2])

    gtr_global_z = asis_z * GTR_ASIS_LATERAL_FACTOR
    gtr_femur_z = gtr_global_z - hjc_z
    return [0.0, 0.0, gtr_femur_z if side == "R" else -gtr_femur_z]


def add_missing_markers(model):
    added = []
    skipped = []
    ms = model.getMarkerSet()

    for lateral_name, medial_name, body in MIRROR_PAIRS:
        if get_marker(ms, medial_name) is not None:
            skipped.append((medial_name, "already exists"))
            continue
        lateral = get_marker(ms, lateral_name)
        if lateral is None:
            skipped.append((medial_name, f"{lateral_name} not in model"))
            continue
        x, y, z = vec3_to_list(lateral.get_location())
        medial_loc = [x, y, -z]
        add_marker(model, medial_name, body, medial_loc)
        added.append((medial_name, body, medial_loc, f"mirror of {lateral_name}"))

    for side in ("R", "L"):
        name = f"{side}.GTR"
        if get_marker(ms, name) is not None:
            skipped.append((name, "already exists"))
            continue
        body = f"femur_{side.lower()}"
        loc = compute_gtr_location(model, side)
        add_marker(model, name, body, loc)
        added.append((name, body, loc, "ASIS*1.075 - HJC lateral offset"))

    return added, skipped


def main():
    print("Add TRC markers to Smith2019 model")
    print("=" * 60)
    print(f"Input:  {INPUT_MODEL}")
    print(f"Output: {OUTPUT_MODEL}")
    print()

    model = osim.Model(INPUT_MODEL)
    model.initSystem()

    ms = model.getMarkerSet()
    print(f"Markers before: {ms.getSize()}")

    added, skipped = add_missing_markers(model)

    # Re-init to register new markers
    model.finalizeConnections()

    print(f"\nAdded {len(added)} markers:")
    for name, body, loc, note in added:
        loc_str = " ".join(f"{v:+.6f}" for v in loc)
        print(f"  {name:10s} on {body:12s} at [{loc_str}]  ({note})")

    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")

    model.printToXML(OUTPUT_MODEL)

    verify = osim.Model(OUTPUT_MODEL)
    verify.initSystem()
    print(f"\nFinal marker count in output: {verify.getMarkerSet().getSize()}")
    print(f"Saved: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
