"""Build an armless Smith2019 model with TRC marker names matching baseline_TM1.trc.

Approach (all via OpenSim API, no XML manipulation):
  1. Strip arms from Smith2019 (8 bodies, 8 joints, arm markers). No muscles
     or wrap surfaces touch the arm bodies, so removal is clean.
  2. RENAME existing Smith2019 markers to TRC names. This preserves their
     anatomically-correct positions on Smith2019's geometry — the fix for
     the earlier bad approach of transplanting coordinates from a different
     model (Rajagopal and Smith2019 have different segment lengths).
  3. Remove unused Smith2019 markers that have no TRC analog (S2,
     R/L.Clavicle).
  4. CREATE new markers for TRC names with no Smith2019 equivalent:
       - Duplicates of an existing marker's position:
           r_shank_antsup, l_shank_antsup  ← copy R.SH1 / L.SH1
           r_thigh4, r_thigh5              ← copy R.TH3 (filler)
       - Medial bony landmarks (mirror lateral across body-frame Z=0):
           r_mknee / L_mknee               ← mirror R.Knee / L.Knee
           r_mankle / L_mankle             ← mirror R.Ankle / L.Ankle
           r_toe / L_toe                   ← mirror R.MT5 / L.MT5 (MT1 side)
       - Torso markers (no Smith2019 anchor) — transplant global position
         from ArmlessRajagopal via the torso body frame:
           C7, R_Sternum, L_Sternum

All markers saved with fixed=false so AddBiomechanics can refine them
during static trial scaling (the user does not trust the anatomical
starting positions to be exactly right).

Three opensim 4.5 Python-binding quirks worked around:
  (1) Removal: `.remove()` on the const accessors (getMarkerSet etc.) corrupts
      the model; use the mutable updMarkerSet / updJointSet / updBodySet.
  (2) `Vec3` from `getLocationInGround(state)` holds a reference into
      state memory that becomes stale when the owning Model/state goes
      away — copy to numpy/floats immediately.
  (3) `addMarker()` invalidates the current State, so any subsequent
      `body.getTransformInGround(state)` throws. Compute every body-local
      position BEFORE any addMarker calls.

Usage:
    conda run -n comak python scripts/phase6_rajagopal_audit/build_smith2019_armless_trc.py
"""

import os

import numpy as np
import opensim as osim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

SMITH = os.path.join(REPO_ROOT, "tests/fixtures/osim_models/full_body_healthy_knee.osim")
RAJA = os.path.join(REPO_ROOT, "scratch/unscaled_generic (1).osim")
OUTPUT = os.path.join(
    REPO_ROOT, "tests/fixtures/osim_models/full_body_healthy_knee_armless_trc.osim"
)

ARM_BODIES = {
    "humerus_r",
    "ulna_r",
    "radius_r",
    "hand_r",
    "humerus_l",
    "ulna_l",
    "radius_l",
    "hand_l",
}
ARM_JOINTS = {
    "acromial_r",
    "elbow_r",
    "radioulnar_r",
    "radius_hand_r",
    "acromial_l",
    "elbow_l",
    "radioulnar_l",
    "radius_hand_l",
}

# ---------------------------------------------------------------------------
# Rename map: existing Smith2019 name -> TRC name (preserves anatomy)
# ---------------------------------------------------------------------------
# Markers where the Smith2019 name already matches the TRC name are simply
# left alone (L.ASIS, L.PSIS).
RENAME_MAP = {
    # Right side
    "R.Knee": "r_knee",
    "R.TH1": "r_thigh1",
    "R.TH2": "r_thigh2",
    "R.TH3": "r_thigh3",
    "R.Ankle": "r_ankle",
    "R.Heel": "r_calc",
    "R.MT5": "r_5meta",
    "R.SH1": "r_shank(antsup)",
    "R.SH2": "r_sh2",
    "R.SH3": "r_sh3",
    "R.SH4": "r_sh4",
    "R.ASIS": "r.ASIS",
    "R.PSIS": "r.PSIS",
    "R.Shoulder": "R_Shoulder",
    # Left side
    "L.Knee": "L_knee",
    "L.TH1": "L_thigh1",
    "L.TH2": "L_thigh2",
    "L.TH3": "L_thigh3",
    "L.TH4": "L_thigh4",
    "L.Ankle": "L_ankle",
    "L.Heel": "L_calc",
    "L.MT5": "L_5meta",
    "L.SH1": "L_shank(ant_sup)",
    "L.SH2": "L_sh2",
    "L.SH3": "L_sh3",
    "L.Shoulder": "L_Shoulder",
}

# Smith2019 markers to delete outright (no TRC analog; arm markers are
# removed with the arm strip).
REMOVE_SMITH = {"S2", "R.Clavicle", "L.Clavicle"}

# New markers that COPY an existing Smith2019 marker's body+location.
# new_trc_name -> source_smith_name (name BEFORE rename)
COPY_FROM = {
    "r_shank_antsup": "R.SH1",  # TRC uses two spellings of the same marker
    "l_shank_antsup": "L.SH1",
    "r_thigh4": "R.TH3",  # extra thigh cluster markers — position
    "r_thigh5": "R.TH3",  # doesn't matter much since fixed=false
}

# New markers by mirroring an existing Smith2019 marker across body-frame Z=0.
# new_trc_name -> source_smith_name (name BEFORE rename)
MIRROR_Z = {
    "r_mknee": "R.Knee",  # medial femoral condyle
    "L_mknee": "L.Knee",
    "r_mankle": "R.Ankle",  # medial malleolus
    "L_mankle": "L.Ankle",
    "r_toe": "R.MT5",  # 1st metatarsal (MT5 is lateral → mirror = medial)
    "L_toe": "L.MT5",
}

# Torso markers we must transplant from ArmlessRajagopal because Smith2019
# has no torso markers. Torso body frames between the two models are similar
# enough that global→local transplant gives a sensible starting position.
TRANSPLANT_FROM_RAJA = ["C7", "R_Sternum", "L_Sternum"]

# Per-marker `fixed` flag, copied from ArmlessRajagopal which was authored
# specifically for this TRC dataset. Convention: bony landmarks (ASIS/PSIS,
# lat+med knee/ankle, heel, MT5, toe, C7/sternum/shoulder) are fixed=true
# anchors for scaling; soft-tissue cluster markers (thigh/shank) are
# fixed=false so the IK optimizer can reposition them during the static
# trial. Any TRC name not listed here defaults to fixed=false.
FIXED_TRUE = {
    # Pelvis
    "r.ASIS",
    "L.ASIS",
    "r.PSIS",
    "L.PSIS",
    # Torso
    "C7",
    "R_Shoulder",
    "L_Shoulder",
    "R_Sternum",
    "L_Sternum",
    # Knee (lateral + medial)
    "r_knee",
    "L_knee",
    "r_mknee",
    "L_mknee",
    # Ankle (lateral + medial)
    "r_ankle",
    "L_ankle",
    "r_mankle",
    "L_mankle",
    # Foot
    "r_toe",
    "L_toe",
    "r_5meta",
    "L_5meta",
    "r_calc",
    "L_calc",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def strip_arms(smith):
    ms = smith.updMarkerSet()
    for i in range(ms.getSize() - 1, -1, -1):
        mk = ms.get(i)
        if mk.getParentFrame().getName() in ARM_BODIES:
            ms.remove(i)
    js = smith.updJointSet()
    for i in range(js.getSize() - 1, -1, -1):
        if js.get(i).getName() in ARM_JOINTS:
            js.remove(i)
    bs = smith.updBodySet()
    for i in range(bs.getSize() - 1, -1, -1):
        if bs.get(i).getName() in ARM_BODIES:
            bs.remove(i)


def capture_existing(smith):
    """Snapshot (name -> (body_name, (x,y,z))) for every existing marker."""
    snapshot = {}
    ms = smith.getMarkerSet()
    for i in range(ms.getSize()):
        mk = ms.get(i)
        loc = mk.get_location()
        snapshot[mk.getName()] = (
            mk.getParentFrame().getName(),
            (float(loc[0]), float(loc[1]), float(loc[2])),
        )
    return snapshot


def remove_markers(smith, names):
    ms = smith.updMarkerSet()
    removed = []
    for i in range(ms.getSize() - 1, -1, -1):
        if ms.get(i).getName() in names:
            removed.append(ms.get(i).getName())
            ms.remove(i)
    return removed


def rename_markers(smith, rename_map):
    """Rename via updMarkerSet — keep parent_frame, location, fixed unchanged."""
    ms = smith.updMarkerSet()
    renamed = []
    for i in range(ms.getSize()):
        mk = ms.get(i)
        old = mk.getName()
        if old in rename_map:
            mk.setName(rename_map[old])
            renamed.append((old, rename_map[old]))
    return renamed


def extract_raja_marker_globals(path, wanted_names):
    """Return {name: (body_name, np.array([gx,gy,gz]))} for each wanted marker."""
    raja = osim.Model(path)
    state = raja.initSystem()
    out = {}
    ms = raja.getMarkerSet()
    for i in range(ms.getSize()):
        mk = ms.get(i)
        if mk.getName() in wanted_names:
            g = mk.getLocationInGround(state)
            out[mk.getName()] = (
                mk.getParentFrame().getName(),
                np.array([g[0], g[1], g[2]]),  # copy out of state memory
            )
    return out


def global_to_body_local(body, state, global_np):
    T = body.getTransformInGround(state)
    R = T.R()
    p = T.p()
    R_np = np.array([[R.get(i, j) for j in range(3)] for i in range(3)])
    p_np = np.array([p[0], p[1], p[2]])
    return R_np.T @ (global_np - p_np)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


print("Build armless Smith2019 with TRC marker names")
print("=" * 60)

smith = osim.Model(SMITH)
print(
    f"[1] Loaded Smith2019: {smith.getBodySet().getSize()} bodies, "
    f"{smith.getMarkerSet().getSize()} markers"
)

# Snapshot existing marker positions BEFORE any mutation (used later as
# source for copy/mirror operations).
existing = capture_existing(smith)

# --- Arm strip
strip_arms(smith)
smith.finalizeConnections()
print(
    f"[2] Stripped arms: {smith.getBodySet().getSize()} bodies, "
    f"{smith.getJointSet().getSize()} joints"
)

# --- Rename existing markers
renamed = rename_markers(smith, RENAME_MAP)
print(f"[3] Renamed {len(renamed)} markers (Smith2019 name -> TRC name)")

# --- Remove unused Smith2019 markers
removed = remove_markers(smith, REMOVE_SMITH)
print(f"[4] Removed {len(removed)} unused markers: {removed}")

smith.finalizeConnections()

# --- Source globals for torso transplants (need ArmlessRajagopal)
raja_globals = extract_raja_marker_globals(RAJA, TRANSPLANT_FROM_RAJA)
print(f"[5] Extracted {len(raja_globals)} torso marker globals from ArmlessRajagopal")

# --- Compute local positions for torso transplants (needs Smith state)
smith_state = smith.initSystem()
torso_locals = {}  # name -> (body_name, (x,y,z))
for name in TRANSPLANT_FROM_RAJA:
    body_name, global_np = raja_globals[name]
    body = smith.getBodySet().get(body_name)
    local_np = global_to_body_local(body, smith_state, global_np)
    torso_locals[name] = (body_name, (float(local_np[0]), float(local_np[1]), float(local_np[2])))

# --- Assemble spec for every NEW marker to add: (name, body_name, (x,y,z))
new_specs = []

for new_name, src_smith in COPY_FROM.items():
    src_body, src_loc = existing[src_smith]
    new_specs.append((new_name, src_body, src_loc, f"copy of {src_smith}"))

for new_name, src_smith in MIRROR_Z.items():
    src_body, (x, y, z) = existing[src_smith]
    new_specs.append((new_name, src_body, (x, y, -z), f"mirror-Z of {src_smith}"))

for name in TRANSPLANT_FROM_RAJA:
    body_name, loc = torso_locals[name]
    new_specs.append((name, body_name, loc, "transplant from ArmlessRajagopal"))

# --- Add new markers. All positions computed above; addMarker() invalidates
# state but we no longer need state here.
for name, body_name, loc, _ in new_specs:
    body = smith.getBodySet().get(body_name)
    new_marker = osim.Marker()
    new_marker.setName(name)
    new_marker.setParentFrame(body)
    new_marker.set_location(osim.Vec3(float(loc[0]), float(loc[1]), float(loc[2])))
    new_marker.set_fixed(name in FIXED_TRUE)
    smith.addMarker(new_marker)

print(f"[6] Added {len(new_specs)} new markers:")
for name, body_name, loc, note in new_specs:
    loc_str = " ".join(f"{v:+.4f}" for v in loc)
    print(f"    {name:20s} on {body_name:10s} local=[{loc_str}]  ({note})")

# --- Set fixed flag. Bony landmarks from FIXED_TRUE are fixed=true, EXCEPT
# markers we derived ourselves (mirror, copy, transplant) — those are guesses
# and always fixed=false so the scaling optimizer can move them.
created_names = set(COPY_FROM) | set(MIRROR_Z) | set(TRANSPLANT_FROM_RAJA)
ms = smith.updMarkerSet()
n_true = n_false = 0
for i in range(ms.getSize()):
    mk = ms.get(i)
    name = mk.getName()
    should_fix = (name in FIXED_TRUE) and (name not in created_names)
    mk.set_fixed(should_fix)
    if should_fix:
        n_true += 1
    else:
        n_false += 1
print(
    f"[7] Set fixed: {n_true} true (trusted Smith2019 landmarks), "
    f"{n_false} false (clusters + all created markers)"
)

smith.finalizeConnections()
smith.printToXML(OUTPUT)
print(f"\nSaved: {OUTPUT}")

# --- Verify and print a summary
verify = osim.Model(OUTPUT)
state = verify.initSystem()
ms = verify.getMarkerSet()
print(
    f"\nVerification: {verify.getBodySet().getSize()} bodies, "
    f"{verify.getJointSet().getSize()} joints, {ms.getSize()} markers\n"
)
print(
    f"  {'Name':22s} {'Body':12s} {'LocalX':>9s} {'LocalY':>9s} {'LocalZ':>9s}   "
    f"{'GlobalX':>8s} {'GlobalY':>8s} {'GlobalZ':>8s}  fixed"
)
for i in range(ms.getSize()):
    mk = ms.get(i)
    l = mk.get_location()
    g = mk.getLocationInGround(state)
    print(
        f"  {mk.getName():22s} {mk.getParentFrame().getName():12s} "
        f"{l[0]:+9.4f} {l[1]:+9.4f} {l[2]:+9.4f}   "
        f"{g[0]:+8.3f} {g[1]:+8.3f} {g[2]:+8.3f}  {mk.get_fixed()}"
    )
