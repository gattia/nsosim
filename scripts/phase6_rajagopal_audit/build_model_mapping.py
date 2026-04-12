"""Phase 6A: Build cross-model mapping from audit JSON files.

Compares two model audits and produces a mapping file containing:
- Body-to-body correspondence
- Joint-to-joint correspondence
- Coordinate-to-coordinate correspondence
- Per-body proposed scaling axes and computed scale factors
- Coordinate conflicts (references that would break)

Usage:
    conda run -n comak python scripts/phase6_rajagopal_audit/build_model_mapping.py \
        scripts/phase6_rajagopal_audit/smith2019_audit.json \
        scripts/phase6_rajagopal_audit/rajagopal_audit.json \
        scripts/phase6_rajagopal_audit/model_mapping.json
"""

import argparse
import json

import numpy as np


# Bodies that are COMAK-specific (exist in Smith2019 but not in generic models)
COMAK_BODIES = {
    "femur_distal_r",
    "tibia_proximal_r",
    "patella_r",
    "meniscus_medial_r",
    "meniscus_lateral_r",
}

# Manual segment overrides for bodies where COMAK intermediate bodies
# change the topology. Maps: body_name → (target_segment_key, base_segment_key)
# These tell the scale factor computation which segments to compare even though
# the immediate child body names differ.
SEGMENT_OVERRIDES = {
    # Smith2019 femur connects to femur_distal_r (COMAK weld),
    # Rajagopal femur connects directly to tibia_r via knee_r
    "femur_r": ("femur_r_to_femur_distal_r", "femur_r_to_tibia_r"),
}

# Bodies that inherit scale factors from another body.
# Leaf bodies (no child joints) and bodies where segment data is missing/zero.
SCALE_INHERIT = {
    # Left side inherits from right (symmetric scaling)
    "femur_l": "femur_r",     # Smith2019 left knee has zero offsets (no weld joints)
    "tibia_l": "tibia_r",
    "talus_l": "talus_r",
    "calcn_l": "calcn_r",
    "humerus_l": "humerus_r",
    "ulna_l": "ulna_r",
    "radius_l": "radius_r",
    # Leaf bodies inherit from parent

    "patella_r": "femur_r",   # patella scales with femur
    "patella_l": "femur_l",
    "toes_r": "calcn_r",      # toes scale with foot
    "toes_l": "calcn_l",
    "hand_r": "radius_r",     # hand scales with forearm
    "hand_l": "radius_l",
}

# Default scaling strategy per body category
# "long_axis": use Y (long axis) scale factor isotropically for all 3 axes
# "pelvis_2d": use X (med/lat) and Z (ant/post) independently, Y from one of them
# "skip": don't scale (ground, etc.)
BODY_SCALE_STRATEGY = {
    # Trunk
    "ground": "skip",
    "pelvis": "pelvis_2d",
    "lumbar5": "long_axis",
    "lumbar4": "long_axis",
    "lumbar3": "long_axis",
    "lumbar2": "long_axis",
    "lumbar1": "long_axis",
    "torso": "long_axis",
    # Right leg
    "femur_r": "long_axis",
    "tibia_r": "long_axis",
    "patella_r": "long_axis",
    "talus_r": "long_axis",
    "calcn_r": "long_axis",
    "toes_r": "long_axis",
    # Left leg
    "femur_l": "long_axis",
    "tibia_l": "long_axis",
    "patella_l": "long_axis",
    "talus_l": "long_axis",
    "calcn_l": "long_axis",
    "toes_l": "long_axis",
    # COMAK-specific (not in Rajagopal)
    "femur_distal_r": "long_axis",
    "tibia_proximal_r": "long_axis",
    "meniscus_medial_r": "long_axis",
    "meniscus_lateral_r": "long_axis",
}


def load_audit(path):
    with open(path) as f:
        return json.load(f)


def build_body_mapping(target_audit, base_audit):
    """Map bodies between target (Smith2019) and base (Rajagopal) models."""
    target_bodies = {b["name"] for b in target_audit["bodies"]}
    base_bodies = {b["name"] for b in base_audit["bodies"]}

    common = sorted(target_bodies & base_bodies)
    target_only = sorted(target_bodies - base_bodies)
    base_only = sorted(base_bodies - target_bodies)

    mapping = []
    for name in common:
        mapping.append(
            {
                "target_body": name,
                "base_body": name,
                "match_type": "exact",
            }
        )

    for name in target_only:
        mapping.append(
            {
                "target_body": name,
                "base_body": None,
                "match_type": "target_only",
                "note": "COMAK body" if name in COMAK_BODIES else "exists only in target",
            }
        )

    for name in base_only:
        mapping.append(
            {
                "target_body": None,
                "base_body": name,
                "match_type": "base_only",
                "note": "exists only in base",
            }
        )

    return mapping


def build_joint_mapping(target_audit, base_audit):
    """Map joints between models, noting type differences."""
    target_joints = {j["name"]: j for j in target_audit["joints"]}
    base_joints = {j["name"]: j for j in base_audit["joints"]}

    all_names = sorted(set(target_joints.keys()) | set(base_joints.keys()))

    mapping = []
    for name in all_names:
        tj = target_joints.get(name)
        bj = base_joints.get(name)

        if tj and bj:
            entry = {
                "target_joint": name,
                "base_joint": name,
                "match_type": "exact",
                "target_type": tj["type"],
                "base_type": bj["type"],
                "type_match": tj["type"] == bj["type"],
                "target_parent": tj["parent_body"],
                "target_child": tj["child_body"],
                "base_parent": bj["parent_body"],
                "base_child": bj["child_body"],
            }
            if tj["type"] != bj["type"]:
                entry["note"] = f"Type mismatch: {tj['type']} vs {bj['type']}"
            mapping.append(entry)
        elif tj:
            mapping.append(
                {
                    "target_joint": name,
                    "base_joint": None,
                    "match_type": "target_only",
                    "target_type": tj["type"],
                    "target_parent": tj["parent_body"],
                    "target_child": tj["child_body"],
                }
            )
        else:
            mapping.append(
                {
                    "target_joint": None,
                    "base_joint": name,
                    "match_type": "base_only",
                    "base_type": bj["type"],
                    "base_parent": bj["parent_body"],
                    "base_child": bj["child_body"],
                }
            )

    return mapping


def build_coordinate_mapping(target_audit, base_audit):
    """Map coordinates, identify conflicts."""
    target_coords = {c["name"]: c for c in target_audit["coordinates"]}
    base_coords = {c["name"]: c for c in base_audit["coordinates"]}

    all_names = sorted(set(target_coords.keys()) | set(base_coords.keys()))

    mapping = []
    for name in all_names:
        tc = target_coords.get(name)
        bc = base_coords.get(name)

        if tc and bc:
            mapping.append(
                {
                    "target_coord": name,
                    "base_coord": name,
                    "match_type": "exact",
                    "target_default": tc["default_value"],
                    "base_default": bc["default_value"],
                    "target_locked": tc["locked"],
                    "base_locked": bc["locked"],
                }
            )
        elif tc:
            mapping.append(
                {
                    "target_coord": name,
                    "base_coord": None,
                    "match_type": "target_only",
                    "target_default": tc["default_value"],
                    "target_locked": tc["locked"],
                }
            )
        else:
            mapping.append(
                {
                    "target_coord": None,
                    "base_coord": name,
                    "match_type": "base_only",
                    "base_default": bc["default_value"],
                    "base_locked": bc["locked"],
                }
            )

    return mapping


def find_coordinate_references(audit, coord_name):
    """Find all muscles, constraints, and forces that reference a coordinate."""
    refs = []

    # Check constraints
    for c in audit.get("constraints", []):
        if c.get("dependent_coordinate") == coord_name:
            refs.append({"type": "constraint_dependent", "name": c["name"]})
        for ic in c.get("independent_coordinates", []):
            if ic == coord_name:
                refs.append({"type": "constraint_independent", "name": c["name"]})

    # Check springs
    for s in audit.get("springs", []):
        if s.get("coordinate") == coord_name:
            refs.append({"type": "spring", "name": s["name"]})

    return refs


def compute_scale_factors(target_audit, base_audit, body_mapping):
    """Compute per-body scale factors from segment lengths.

    Returns dict of {body_name: {strategy, scale_factors, details}}.
    """
    target_segments = target_audit.get("segment_lengths", {})
    base_segments = base_audit.get("segment_lengths", {})

    # Build lookup: body → list of segments starting from that body
    def segments_for_body(segments, body_name):
        return {k: v for k, v in segments.items() if v["body"] == body_name}

    scale_info = {}

    for bm in body_mapping:
        target_name = bm.get("target_body")
        base_name = bm.get("base_body")

        if not target_name or not base_name:
            continue

        strategy = BODY_SCALE_STRATEGY.get(target_name, "long_axis")
        if strategy == "skip":
            scale_info[target_name] = {
                "strategy": "skip",
                "scale_factors": [1.0, 1.0, 1.0],
                "note": "Ground body, no scaling",
            }
            continue

        # Find matching segments
        target_segs = segments_for_body(target_segments, target_name)
        base_segs = segments_for_body(base_segments, base_name)

        if not target_segs or not base_segs:
            scale_info[target_name] = {
                "strategy": strategy,
                "scale_factors": [1.0, 1.0, 1.0],
                "note": "No segment data available",
                "target_segments": list(target_segs.keys()),
                "base_segments": list(base_segs.keys()),
            }
            continue

        # Check manual overrides first
        matched_pairs = []
        if target_name in SEGMENT_OVERRIDES:
            override_target_key, override_base_key = SEGMENT_OVERRIDES[target_name]
            if override_target_key in target_segments and override_base_key in base_segments:
                matched_pairs.append(
                    (target_segments[override_target_key],
                     base_segments[override_base_key],
                     f"{override_target_key} ↔ {override_base_key} (manual override)")
                )

        # If no override, find matching segment pairs by child body name
        if not matched_pairs:
            for tk, tv in target_segs.items():
                for bk, bv in base_segs.items():
                    # Match by distal joint name or child body
                    if tv["distal_joint"] == bv["distal_joint"]:
                        matched_pairs.append((tv, bv, tk))
                        break

        if not matched_pairs:
            # Try matching by segment key pattern
            for tk, tv in target_segs.items():
                for bk, bv in base_segs.items():
                    # Match if the child body names match
                    target_child = tk.split("_to_")[1] if "_to_" in tk else ""
                    base_child = bk.split("_to_")[1] if "_to_" in bk else ""
                    if target_child == base_child:
                        matched_pairs.append((tv, bv, tk))
                        break

        if not matched_pairs:
            scale_info[target_name] = {
                "strategy": strategy,
                "scale_factors": [1.0, 1.0, 1.0],
                "note": "No matching segments found",
                "target_segments": list(target_segs.keys()),
                "base_segments": list(base_segs.keys()),
            }
            continue

        # Use first matched pair for scale computation
        tv, bv, seg_name = matched_pairs[0]
        target_diff = np.array(tv["diff_xyz"])
        base_diff = np.array(bv["diff_xyz"])

        if strategy == "long_axis":
            # Find the dominant axis (longest absolute difference)
            dominant_axis = int(np.argmax(np.abs(base_diff)))
            if abs(base_diff[dominant_axis]) > 1e-8:
                long_scale = abs(target_diff[dominant_axis]) / abs(base_diff[dominant_axis])
            else:
                long_scale = 1.0
            scale_factors = [long_scale, long_scale, long_scale]

            scale_info[target_name] = {
                "strategy": "long_axis",
                "dominant_axis": ["X", "Y", "Z"][dominant_axis],
                "scale_factors": [round(s, 6) for s in scale_factors],
                "long_axis_scale": round(long_scale, 6),
                "segment_used": seg_name,
                "target_diff_xyz": [round(x, 6) for x in target_diff.tolist()],
                "base_diff_xyz": [round(x, 6) for x in base_diff.tolist()],
                "target_length": round(tv["length"], 6),
                "base_length": round(bv["length"], 6),
            }

        elif strategy == "pelvis_2d":
            # Pelvis scaling uses all 3 axes independently from the hip joint offset.
            # X = anterior/posterior, Y = inferior/superior, Z = medial/lateral
            sx = abs(target_diff[0]) / abs(base_diff[0]) if abs(base_diff[0]) > 1e-8 else 1.0
            sy = abs(target_diff[1]) / abs(base_diff[1]) if abs(base_diff[1]) > 1e-8 else 1.0
            sz = abs(target_diff[2]) / abs(base_diff[2]) if abs(base_diff[2]) > 1e-8 else 1.0

            scale_info[target_name] = {
                "strategy": "pelvis_2d",
                "scale_factors": [round(sx, 6), round(sy, 6), round(sz, 6)],
                "scale_x_antpost": round(sx, 6),
                "scale_y_infsup": round(sy, 6),
                "scale_z_medlat": round(sz, 6),
                "segment_used": seg_name,
                "target_diff_xyz": [round(x, 6) for x in target_diff.tolist()],
                "base_diff_xyz": [round(x, 6) for x in base_diff.tolist()],
                "note": "Pelvis: X=ant/post, Y=inf/sup, Z=med/lat — all 3 axes independent",
            }

    # Apply inheritance for bodies that should copy from another body
    for body, parent in SCALE_INHERIT.items():
        if parent not in scale_info:
            continue
        parent_sf = scale_info[parent]
        # Inherit if: no segment data, or scale is zero/degenerate, or body not yet in scale_info
        current = scale_info.get(body, {})
        needs_inherit = (
            body not in scale_info
            or current.get("note", "").startswith("No segment")
            or current.get("note", "").startswith("No matching")
            or current.get("scale_factors", [1])[0] == 0.0
        )
        if needs_inherit:
            scale_info[body] = {
                "strategy": "inherited",
                "scale_factors": parent_sf["scale_factors"],
                "inherited_from": parent,
                "note": f"Inherited from {parent}",
            }

    return scale_info


def build_mapping(target_path, base_path):
    """Build complete cross-model mapping."""
    target = load_audit(target_path)
    base = load_audit(base_path)

    print(f"Target model: {target['model_name']}")
    print(f"Base model: {base['model_name']}")

    body_mapping = build_body_mapping(target, base)
    joint_mapping = build_joint_mapping(target, base)
    coord_mapping = build_coordinate_mapping(target, base)

    # Find coordinate conflicts: coords in base but not target that are referenced
    coord_conflicts = []
    for cm in coord_mapping:
        if cm["match_type"] == "base_only":
            coord_name = cm["base_coord"]
            refs = find_coordinate_references(base, coord_name)
            if refs:
                coord_conflicts.append(
                    {
                        "coordinate": coord_name,
                        "exists_in": "base_only",
                        "references": refs,
                    }
                )
        elif cm["match_type"] == "target_only":
            coord_name = cm["target_coord"]
            # COMAK coordinates that will be added — check if base has something similar
            coord_conflicts.append(
                {
                    "coordinate": coord_name,
                    "exists_in": "target_only",
                    "note": "Will be added with COMAK knee",
                }
            )

    # Compute scale factors
    scale_factors = compute_scale_factors(target, base, body_mapping)

    result = {
        "target_model": target["model_name"],
        "base_model": base["model_name"],
        "target_path": target["model_path"],
        "base_path": base["model_path"],
        "summary": {
            "target": target["summary"],
            "base": base["summary"],
            "bodies_common": sum(1 for b in body_mapping if b["match_type"] == "exact"),
            "bodies_target_only": sum(
                1 for b in body_mapping if b["match_type"] == "target_only"
            ),
            "bodies_base_only": sum(
                1 for b in body_mapping if b["match_type"] == "base_only"
            ),
            "joints_common": sum(1 for j in joint_mapping if j["match_type"] == "exact"),
            "joints_type_mismatches": sum(
                1
                for j in joint_mapping
                if j["match_type"] == "exact" and not j.get("type_match", True)
            ),
            "coords_common": sum(1 for c in coord_mapping if c["match_type"] == "exact"),
            "coords_target_only": sum(
                1 for c in coord_mapping if c["match_type"] == "target_only"
            ),
            "coords_base_only": sum(
                1 for c in coord_mapping if c["match_type"] == "base_only"
            ),
        },
        "body_mapping": body_mapping,
        "joint_mapping": joint_mapping,
        "coordinate_mapping": coord_mapping,
        "coordinate_conflicts": coord_conflicts,
        "scale_factors": scale_factors,
    }

    # Print summary
    print(f"\nBody mapping: {result['summary']['bodies_common']} common, "
          f"{result['summary']['bodies_target_only']} target-only, "
          f"{result['summary']['bodies_base_only']} base-only")
    print(f"Joint mapping: {result['summary']['joints_common']} common, "
          f"{result['summary']['joints_type_mismatches']} type mismatches")
    print(f"Coordinate mapping: {result['summary']['coords_common']} common, "
          f"{result['summary']['coords_target_only']} target-only, "
          f"{result['summary']['coords_base_only']} base-only")
    print(f"Coordinate conflicts: {len(coord_conflicts)}")
    print(f"\nScale factors computed for {len(scale_factors)} bodies")

    return result


def main():
    parser = argparse.ArgumentParser(description="Build cross-model mapping")
    parser.add_argument("target_audit", help="Path to target model audit JSON (Smith2019)")
    parser.add_argument("base_audit", help="Path to base model audit JSON (Rajagopal)")
    parser.add_argument("output_path", help="Path for output mapping JSON")
    args = parser.parse_args()

    result = build_mapping(args.target_audit, args.base_audit)

    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nMapping written to: {args.output_path}")


if __name__ == "__main__":
    main()
