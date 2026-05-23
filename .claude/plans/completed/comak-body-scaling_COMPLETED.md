# Plan: COMAK Body Scaling (Stage X)

**Status:** Complete (2026-05-23)
**Created:** 2026-05-13.
**Parent:** Pathway B of [`comak_gait_simulation/.claude/plans/PAPER1_MULTIGAIT.md`](../../../comak_gait_simulation/.claude/plans/PAPER1_MULTIGAIT.md).

---

## Goal

Define a self-contained "Stage X" that takes a COMAK base model plus a set of AddBiomechanics (AB) scale factors and produces a **complete, internally-consistent scaled COMAK model**. "Complete" means the knee assembly (bones, cartilage meshes, ligaments, Smith2018 contact, wrap surfaces, menisci) is scaled coherently with the rest of the body — not just the torso/limbs.

The output is itself a valid COMAK base. Downstream pipelines that consume a COMAK base (notably the OAI knee-swap pipeline, "Stage Y") can run on Stage X's output with **no modifications and no awareness of AB scaling**.

## Why this shape

Three pathways currently in scope, all decomposing cleanly around this contract:

| Pathway | Stage X runs | Stage Y runs | COMAK simulations |
|---|---|---|---|
| A — current OAI paper | 0 (legacy reference base) | 715 | 715 |
| B — Tian 3 gaits × 715 OAI knees | 3 (one per Tian subject) | 715 per Tian subject | 3 × 715 = 2,145 |
| C — future MRI+gait matched | 1 per subject | 1 per subject (variant: knee bones excluded from scale; native MRI knee size preserved) | 1 per subject |

## Stage X contract

**Inputs:**
- `base_model_path` — a COMAK `.osim` (full Smith2019 base with knee assembly intact, JAM components present).
- `ab_scaled_model_path` — AB's output `match_markers_and_physics.osim`. **AB does not output a `_rescaling_setup.xml`** — scale factors are baked into this file's per-body `<attached_geometry>/scale_factors` fields, and AB's static-trial-IK-placed MarkerSet is in its `<MarkerSet>` block. Single source of both signals.
- `scaling_mode` — `'WA'` | `'LA'` | `'AB'`. **Default: `'WA'`** (weighted average of femur + tibia long-axis factors, isotropic, applied only to right-knee bodies — full spec below).
- `output_path` — where to write the scaled COMAK `.osim`.
- `output_geometry_dir` — where to write geometry: a copy of base's `Geometry/` + new scaled cartilage `.stl`s. **Convention: must be `output_path.parent / "Geometry"`** so OpenSim resolves `mesh_file` references relative to the output `.osim`'s directory without extra search-path setup. The orchestrator enforces this — if the caller passes anything else, raise.

**Output:**
- A scaled `.osim` with the same structure as the input (same bodies, joints, components) but with all spatial quantities consistently scaled.
- A populated `output_geometry_dir/` containing: (a) a copy of every geometry file the base model references (bone meshes, visual meshes — these are scaled by ScaleTool's body-frame mechanism so the STL on disk doesn't change, but the file needs to exist at the new path), (b) new scaled cartilage `.stl` files replacing the originals for `Smith2018ContactMesh` (see "Contact mesh workflow" section).
- *(Optional)* `cartilage_scales.json` — debug artifact listing `{mesh_name: scale_factor}` used by the STL-scaling step. Not required: the dict is passed in memory from `build_scale_set` → `scale_cartilage_meshes`. Written alongside the report only when `report_json` is provided, for traceability.
- A JSON sidecar (`*_scaling_report.json`) recording: factor per body, scaling mode, source AB XML, list of components touched, verification warnings. See "Worked example" for the report shape.

**Patella offset note:** `patella_offset.json` is a Stage Y artifact (computed when an NSM-fit subject patella is swapped in — its centroid sets the PF defaults). Stage X does **not** consume or produce this file. The base model's PF defaults (`pf_tx_r`/`pf_ty_r`/`pf_tz_r`) scale automatically via ScaleTool when patella is in the ScaleSet. If Stage Y runs on Stage X's output later, Stage Y overwrites the PF defaults with its own per-subject patella centroid.

**Scope of scaling — full ScaleSet spec for `'WA'` mode:**

Per-body scale factors are read from `match_markers_and_physics.osim`'s `<attached_geometry>/scale_factors` fields. AB only sees the **stripped** model we upload (14 bodies, no knee subbodies), so it provides factors only for those. Example real values from RSubject_121:

```
pelvis:     (1.2973, 1.0179, 1.1566)
femur_r:    (0.9812, 0.9637, 1.0049)
tibia_r:    (0.9610, 1.0140, 0.9410)
patella_r:  (1.0000, 1.0000, 1.0000)   ← AB leaves at identity; no static signal
torso:      (...) etc.
```

The long-axis index is **2 (Z)** per old `scaleModel.py` lines 121/124. Define:

```
s_wa = (s_femur_r[2] + s_tibia_r[2]) / 2     # scalar isotropic factor
```

| Body | Source | Stage X factor in WA mode |
|---|---|---|
| `femur_r` | AB | **AB's anisotropic 3-vec, unchanged** |
| `tibia_r` | AB | **AB's anisotropic 3-vec, unchanged** |
| `pelvis`, `talus_r`, `calcn_r`, `toes_r`, `torso`, all left-side | AB | **AB's anisotropic 3-vec, unchanged** |
| `patella_r` | AB (but `[1,1,1]`) | **`(s_wa, s_wa, s_wa)` — isotropic override, since AB's [1,1,1] is uninformative** |
| `femur_distal_r` | Not in AB output (stripped from upload) | `(s_wa, s_wa, s_wa)` — isotropic |
| `tibia_proximal_r` | Not in AB output | `(s_wa, s_wa, s_wa)` — isotropic |
| `meniscus_medial_r` | Not in AB output | `(s_wa, s_wa, s_wa)` — isotropic — **new vs. old script** |
| `meniscus_lateral_r` | Not in AB output | `(s_wa, s_wa, s_wa)` — isotropic — **new vs. old script** |
| Any other COMAK body AB didn't see | n/a | Not in ScaleSet (defaults to `[1, 1, 1]`) |

Old script (`scaleModel.py` lines 118–156) handles patella + femur_distal_r/tibia_proximal_r; we add the two meniscus rows. Body-mass redistribution under `preserve_mass_distribution=true` follows OpenSim conventions on this set.

**Invariants:**
1. Output loads + `initSystem()` succeeds in OpenSim 4.5 with the JAM plugin.
2. With identity factors (`s=1` for every body) the output is functionally equivalent to the input — mesh hashes within float tolerance, one-step COMAK identical.
3. With non-trivial factors, all knee components scale together — no penetrations/gaps between cartilage and bone, no broken ligament attachments, wrap surfaces remain on their parent bones.
4. Markers preserved from the AB scaled model (we trust AB's static-trial placement; Stage X doesn't re-place them).

## What `ScaleTool.run()` already does for us (verified empirically)

Empirical test on the **full unstripped COMAK model** with `ScaleTool.run()` and a ScaleSet of `{femur_r: 0.9, femur_distal_r: 0.9}`. Output reloaded and inspected — `ScaleTool` correctly auto-handles everything in this list:

| Component | Result |
|---|---|
| Body segments + bone meshes (`femur_r`, `femur_distal_r`, etc.) | Scaled, including all attached visual geometry |
| Joint frame translations | Scaled. `femur_femur_distal_r` (-0.00555 → -0.00500), `tibia_tibia_proximal_r` (0.006 → 0.0054), `pf_r` defaults (`pf_tx_r` 0.053 → 0.0477, `pf_ty_r` 0.005 → 0.0045, `pf_tz_r` 0.004 → 0.0036) all × 0.9. **Caveat:** weld translations only scale if the **parent body** is in the ScaleSet — verified by running once without `tibia_r` (translation didn't scale) and once with (it did). Production Stage X includes all parent bodies, so this is fine. |
| 91 Blankevoort1991Ligament `PathPoint` locations | Scaled with parent body (path lengths shrank by ratios specific to each ligament's anatomical span) |
| 91 Blankevoort1991Ligament `slack_length` | Scaled via JAM's `extendPostScale` hook |
| 91 Blankevoort1991Ligament reference strain | **Preserved exactly.** ACLpl1 0.16481→0.16481, ITB1 0.02459→0.02459, etc. |
| All 39 wrap surfaces (cylinders + ellipsoids on every body) | `radius`, `length`, `dimensions`, `translation` all scaled |
| 7 Smith2018ContactMesh entries (presence) | All 7 preserved in output model |
| Smith2018ArticularContactForce (6 entries) | Material props unchanged (correct — they're constants) |
| All `Millard2012EquilibriumMuscle` | `optimal_fiber_length` + `tendon_slack_length` scaled by path-length ratio |
| Body masses | Redistributed (with `preserve_mass_distribution=true` — old pipeline default). Total mass preserved across the body; individual segment masses redistribute when some segments shrink. `femur_distal_r` mass actually went up post-scale because total mass is conserved while the femur_distal_r volume share decreased. If you want segment-proportional masses (volume scales mass cubically) set `preserve_mass_distribution=false`. Old pipeline used `true`; Stage X defaults to `true` for continuity. |

Implications:
- **No manual ligament slack update needed** for Stage X. The JAM-provided `extendPostScale` does the snapshot-and-restore-strain pattern internally. (`nsosim.osim_utils.update_slack_lengths` is for a different scenario — knee geometry *swap*, not uniform body scaling.)
- **No manual wrap surface scaling needed.** All 39 scale generically.
- **No knee stripping needed.** `Model.scale()` in-memory crashes on JAM contact mesh init; `ScaleTool.run()` writes XML and reloads, sidestepping this. The old pipeline used this exact path.

## What `ScaleTool.run()` does NOT do (manual work needed)

| # | Component | Quantity | Approach | Status in old code |
|---|---|---|---|---|
| 1 | 7 `Smith2018ContactMesh` cartilage meshes | Vertex coordinates of underlying `.stl` files | **Scale `.stl` vertices outside OpenSim** with `pymskt` (isotropic, WA factor). Write new `.stl` files; update `mesh_file` in the scaled `.osim`. **Never set `scale_factors` in the XML** — old README explicitly warns COMAK-IK breaks. See "Contact mesh workflow" section. | Production pattern: `scaleModel.py` re-points `socket_scale_frame` + `rescale_comak_knee.ipynb` cells 14–20 do the actual STL scaling externally. |
| 2 | Markers | Whole MarkerSet | **Replace the scaled COMAK output's `MarkerSet` block entirely with AB's MarkerSet block.** AB took our marker-renamed model as input (same body names, same marker names), then placed markers via static-trial IK. Those are the markers downstream COMAK-IK will match against Tian's dynamic TRC. The original COMAK markers are not used post-scale; save them to a backup JSON if needed. See "Marker handling" section. | Old script swaps the whole marker XML — same approach. |
| 3 | Geometry/ directory population | Files | Copy entire `Geometry/` from base model's resolution paths to `output_geometry_dir/`. Then replace the cartilage `.stl` files with the pymskt-scaled versions from row 1. Update `mesh_file` fields in the output `.osim` for the cart entries. Bone/visual meshes are unchanged on disk — they scale via OpenSim's frame mechanism. | Old pipeline rewrote individual STLs; we do the whole-folder copy for safety. |
| 4 | ITB1 ligament path-point 2 | Location + parent frame | **Permanent model bug fix-up.** Reparent from `tibia_r` to `tibia_proximal_r` and back-correct the local coordinate. Stage X runs this as a runtime fix-up for now; the right long-term fix is to correct the base fixture itself (see "ITB1 fix-up" section). | Done in `scaleModel.py:206-260` |
| ~~5~~ | ~~Patella centering offset (`patella_offset.json`)~~ | — | **Out of scope.** Stage Y concern (per-subject patella mesh centroid). Stage X's base-model PF defaults scale automatically via ScaleTool. | n/a |
| ~~6~~ | ~~CoordinateCouplerConstraint~~ | — | **Resolved (Q3): absent in base model.** Empty `ConstraintSet` in `full_body_healthy_knee.osim`. Where present elsewhere, function is `LinearFunction` with dimensionless coefficients. | n/a |

So Stage X = `ScaleTool.run()` + rows 1–4 (cart STLs, markers, Geometry copy, ITB1). Rows 5–6 resolved as out-of-scope.

## Contact mesh workflow

Production pattern, sourced from three places:

1. [`Process_Pipeline/README.md`](/dataNAS/people/aagatti/projects/gait_opensim_jam_2023/stanford_jam_gait_2023/AddBiomechanics/Process_Pipeline/README.md) (the original instructions for running the pipeline):

   > Note: contact meshes need to be scaled separately using another notebook script
   > - Requires a library not in comak…is available in the mskt environment
   > - **Scale factors cannot be changed in xml file since the comak ik will not run if that's the case**
   > - Bone meshes are automatically updated in opensim

2. [`Process_Pipeline/rescale_comak_knee.ipynb`](/dataNAS/people/aagatti/projects/gait_opensim_jam_2023/stanford_jam_gait_2023/AddBiomechanics/Process_Pipeline/rescale_comak_knee.ipynb) cells 14–20 — the actual working code:
   - Cell 14: compute `cartilage_scale` (3-vec, average of femur and tibia AB factors).
   - Cell 15 (commented out — kept as reference): `mesh = pv.read(...stl); mesh.transform(scaleMatrix); mesh.save(scaled_name)`.
   - Cell 18: load the scaled `.osim`, walk every `Smith2018ContactMesh`, and (per the comments) two-step pattern: change `mesh_file` to point at the scaled STL, and set `socket_scale_frame` to the parent bone body. The commented-out "Don't use this" block in cell 19 is the XML-`scale_factors` approach that breaks COMAK-IK.
   - Cell 20: alternate version that only renames `mesh_file` (no socket change).

3. [`scaleModel.py`](/dataNAS/people/aagatti/projects/gait_opensim_jam_2023/stanford_jam_gait_2023/AddBiomechanics/Process_Pipeline/scaleModel.py) lines 180–204 sets `socket_scale_frame` to `/bodyset/femur_distal_r` (etc.) but does NOT scale the STL — the notebook does that part.

Why it has to be done outside OpenSim: `Smith2018ContactMesh` carries its own underlying `.stl` loaded by the JAM plugin. The plugin's loader reads raw vertices; XML `scale_factors` aren't honored by the contact mesh codepath. Setting them in XML makes the COMAK-IK step crash somewhere downstream. Pre-scaling the STL on disk and only updating `mesh_file` is the only safe path.

Stage X implementation, in two clearly separated steps:

**Step A — `scale_factors.py` returns the cartilage scales (in-memory dict, second element of `build_scale_set`'s return tuple).**

Shape: `{mesh_name: float_scale}`. For WA mode all three are `s_wa`. The orchestrator passes this dict to `cartilage_meshes.py`. Optionally writes `cartilage_scales.json` alongside the report for traceability — not required.

**Step B — `cartilage_meshes.py` consumes the dict and does the mesh edit.**

For each entry:
- Locate the source `.stl` on disk (resolve via the base model's `Geometry/` directory).
- Load with `pymskt` (`mskt.mesh.Mesh(src_stl)`), multiply `point_coords` by the isotropic scale, save with `save_mesh(out_stl)`.
- Update the `Smith2018ContactMesh`'s `mesh_file` field via the OpenSim API: `Smith2018ContactMesh.set_mesh_file(new_filename)`, then `Model.printToXML()`. See `cartilage_meshes.py` signature for the exact sketch.

Also at this step, **copy the entire base-model `Geometry/` folder** into `output_geometry_dir/` first — bone meshes, visual meshes, etc. — so the output `.osim` resolves every geometry reference. The cart STL writes then OVERWRITE the cart files in the copied folder. (Bone/visual STLs on disk are unchanged; they scale via OpenSim's body-frame mechanism in the .osim XML, not by editing the STL.)

**Never touch:**
- `scale_factors` in XML — leave at `[1, 1, 1]`. Old README explicitly warns COMAK-IK breaks.
- `socket_scale_frame` — preserve from base.

## Marker handling

**Strategy: full MarkerSet replace via OpenSim API.**

Why: AB took our marker-renamed COMAK base (the `unscaled_generic_tian.osim` produced by `scripts/phase6_rajagopal_audit/` — same body names, Tian-named markers) as input and ran static-trial IK on it. Its output `.osim` therefore has:
- The same body names we use.
- The same marker names we'll match against Tian's TRC.
- Refined marker positions on the AB-scaled body.

Those AB-placed positions are precisely what downstream COMAK-IK matches Tian's dynamic TRC against. The original COMAK base's `MarkerSet` block has stale positions (from before scaling) and no useful information after Stage X. No per-marker matching needed; not every marker needs to have TRC data — downstream IK ignores markers it has no data for.

**Algorithm** (via OpenSim API):

```python
def swap_markerset(scaled_osim, ab_scaled_osim, backup_xml_path=None):
    # Load both
    scaled = osim.Model(str(scaled_osim))
    ab = osim.Model(str(ab_scaled_osim))
    scaled.initSystem(); ab.initSystem()

    # Optional backup of COMAK's original MarkerSet (write the MarkerSet object as XML)
    if backup_xml_path is not None:
        scaled.getMarkerSet().printToXML(str(backup_xml_path))

    # Drop every marker from scaled model
    ms = scaled.updMarkerSet()
    for i in range(ms.getSize() - 1, -1, -1):
        ms.remove(i)

    # Clone each AB marker into scaled. For each: validate socket resolves
    # to a body that exists in scaled; if not, log + skip.
    ab_ms = ab.getMarkerSet()
    body_names = {scaled.getBodySet().get(i).getName()
                  for i in range(scaled.getBodySet().getSize())}
    n_added = n_dropped = 0
    for i in range(ab_ms.getSize()):
        m = ab_ms.get(i)
        parent_name = m.getParentFrame().getName()
        if parent_name not in body_names:
            log_warning(f"AB marker {m.getName()} references missing body "
                        f"{parent_name}; dropping.")
            n_dropped += 1
            continue
        # Clone marker onto the scaled model's body
        new_m = osim.Marker(m.getName(),
                            scaled.getBodySet().get(parent_name),
                            m.get_location())
        new_m.set_fixed(m.get_fixed())
        scaled.addMarker(new_m)
        n_added += 1

    scaled.finalizeConnections()
    scaled.printToXML(str(scaled_osim))  # save back
    return n_added, n_dropped
```

No raw XML edits — `osim.Marker(...)` constructor + `Model.addMarker` + `Model.printToXML` is the API path. Backup uses `MarkerSet.printToXML()` (a built-in serializer).

## ITB1 fix-up — why and how

The old `scaleModel.py:206-260` reparents ITB1 path-point 2 from `tibia_r` to `tibia_proximal_r` and shifts its location by the weld-joint translation. **This is not a scaling artifact — it's a permanent bug fix on the source COMAK model.** Confirmed by [`combined_comak_predsim_plan.md`](/dataNAS/people/aagatti/projects/comak_gait_simulation/NOTES/combined_comak_predsim_plan.md) §"Engineering Challenges" item 2: *"ITB1 ligament point 2 was on the wrong body — moved to tibia_proximal_r."*

The bug: in the source model, ITB1's distal attachment is anatomically on the proximal tibia plateau, but it was authored with `parent_frame=/bodyset/tibia_r`. After Stage X scales the body, this would cause ITB1 to move with the wrong body and the ligament line of action would drift. The fix-up reparents to the correct body and back-corrects the local coordinate so the global position is preserved at the post-scale moment.

### Pseudocode (mirrors scaleModel.py:206-260)

```python
def fix_itb1_attachment(model: osim.Model) -> bool:
    forces = model.upd_ForceSet()
    itb1 = osim.Blankevoort1991Ligament.safeDownCast(forces.get('ITB1'))
    if itb1 is None:
        return False  # not present (e.g., upstream removed it)

    pt_idx = 2  # path-point index 2 is the buggy distal attachment
    geopath = itb1.get_GeometryPath()
    pp = osim.PathPoint.safeDownCast(geopath.getPathPointSet().get(pt_idx))

    # Guard: if someone fixed the source model upstream, exit cleanly
    if pp.getParentFrame().getName() != 'tibia_r':
        return False

    # Translation from tibia_r origin to tibia_proximal_r origin (post-scale)
    weld = model.getJointSet().get('tibia_tibia_proximal_r')
    t = weld.get_frames(0).get_translation()  # in tibia_r local frame

    state = model.initSystem()
    orig_loc = pp.getLocation(state)
    new_loc = [orig_loc[i] + t[i] for i in range(3)]

    # Reparent + relocate
    tibia_proximal = model.getBodySet().get('tibia_proximal_r')
    pp.setParentFrame(tibia_proximal)
    pp.setLocation(osim.Vec3(*new_loc))
    return True
```

### Fix-at-source recommendation

Confirmed empirically via OpenSim API inspection of the base model:

```
ITB1 has 3 path points
  pt[0]: parent=pelvis                local=(-0.0837,+0.0530,+0.1196)  ← pelvis attachment
  pt[1]: parent=femur_r               local=(-0.0093,-0.0043,+0.0555)  ← mid-femur wrap point
  pt[2]: parent=tibia_r               local=(+0.0109,-0.0313,+0.0349)  ← BUG: should be tibia_proximal_r (Gerdy's tubercle)
```

**Strongly preferred path: fix the fixture once and never run the runtime fix-up again.**

The fix is a one-time OpenSim API operation:
1. Load `tests/fixtures/osim_models/full_body_healthy_knee.osim`.
2. Run the `fix_itb1_attachment()` algorithm (the same one in `model_fixes.py`) on the unscaled model — `weld translation` here is the unscaled weld translation, so the back-correction is in unscaled coords.
3. `model.printToXML()` back to the same path.
4. Run the two existing tests that reference this fixture: [`tests/test_knee_assembly.py`](/dataNAS/people/aagatti/programming/nsosim/tests/test_knee_assembly.py) and [`tests/test_knee_assembly_forsim.py`](/dataNAS/people/aagatti/programming/nsosim/tests/test_knee_assembly_forsim.py). If they pass, commit and remove `model_fixes.fix_itb1_attachment` from the Stage X pipeline.

I did not do this in the plan revision because it modifies a checked-in fixture that two tests depend on — should be a separate PR with explicit sign-off. **If you sign off now, I can do steps 1–4 in this session before any Stage X code is written.**

## Wrap surfaces — addressing the "why did the old pipeline replace just one?" question

Verified empirically: `ScaleTool.run()` correctly scales all 39 wrap surfaces in the model (radius/length/dimensions/translation, on every body). The old pipeline's KnExt-specific replacement (`scaleModel.py:298-388` + `update_the_model_wrapping.ipynb`) was not a workaround for ScaleTool failing on wraps — it was a model-specific intervention for the KnExt wrap alone:

- A new hardcoded `WrapEllipsoid` named `KnExt_at_fem_r_2` with fixed dimensions was added to `femur_r`.
- The quad muscles (`recfem_r`, `vasint_r`, `vaslat_r`, `vasmed_r`) were re-pointed to wrap around the new ellipsoid.
- Specific quad path-point locations were also adjusted.

The other 38 wraps scaled fine. The KnExt replacement was likely either a quality fix (the original KnExt geometry was bad for the quad muscles) or a known-good substitution that the team standardized on. We do **not** want to copy that — it loses subject-specific information and isn't a Stage X concern.

## Code organization

Re-organizing the repo around scaling, per the user's nudge. New package layout:

```
nsosim/scaling/
    __init__.py           # Public API: scale_comak_model() top-level orchestrator
    config.py             # WA constants (long-axis index, knee body lists, cartilage mesh names)
    scale_factors.py      # Read AB scaled .osim → per-body factor dict; build ScaleSet per mode
    scaletool.py          # apply_scaletool(base_osim, scale_set, out_osim) — wraps ScaleTool.run()
    cartilage_meshes.py   # pymskt STL vertex scaling; OpenSim API to update mesh_file refs
    markers.py            # Full MarkerSet swap from AB scaled output
    model_fixes.py        # ITB1 reparent fix-up
    report.py             # Build JSON sidecar

tests/scaling/
    test_scale_factors.py
    test_identity.py             # s=1 roundtrip
    test_nontrivial.py           # s=0.9 coherence (cart-bone, refstrain preserved, etc.)
    test_cartilage_meshes.py
    test_end_to_end_rsubject121.py  # @pytest.mark.slow
```

Old phase6 scripts (marker mapping) stay where they are; they're upstream of this and unrelated to the scaling logic.

`__init__.py` exposes a single public function `scale_comak_model(...)` — that's the orchestrator. Users `from nsosim.scaling import scale_comak_model`.

**Test conventions:**
- `tests/scaling/` is a new sibling test directory. It needs its own `conftest.py` (for the `identity_ab_osim` session fixture defined in the "Baseline pattern" section). No imports from a parent conftest required.
- Tests run via `conda run -n comak python -m pytest tests/scaling/ -v` (per existing CLAUDE.md convention).
- `@pytest.mark.slow` for any test that loads + scales the full COMAK model end-to-end (~5–10s per run). Per pyproject.toml: `markers = ["slow: ..."]`. Deselect with `-m "not slow"`.

## Function signatures

Public APIs the implementer needs to write. All paths are absolute (`pathlib.Path` or `str`). Types use standard `typing`.

```python
# nsosim/scaling/config.py

from typing import Literal
ScalingMode = Literal["WA", "LA", "AB"]
LONG_AXIS_INDEX: int = 2  # Z, per old scaleModel.py:121/124

# Bodies that get the WA isotropic factor (override AB's anisotropic):
WA_KNEE_BODIES: tuple[str, ...] = (
    "femur_distal_r", "tibia_proximal_r", "patella_r",
    "meniscus_medial_r", "meniscus_lateral_r",
)

# Cartilage contact-mesh names (used to find STLs to pre-scale):
CARTILAGE_MESH_NAMES: tuple[str, ...] = (
    "femur_cartilage", "tibia_cartilage", "patella_cartilage",
    # add others if present in the model — see Smith2018ContactMesh enumeration
)
```

```python
# nsosim/scaling/scale_factors.py

import opensim as osim
from pathlib import Path

def read_ab_factors(ab_scaled_osim: Path) -> dict[str, tuple[float, float, float]]:
    """Extract per-body scale factors from AB's match_markers_and_physics.osim.

    Reads each Body's first <attached_geometry>/scale_factors field. Returns
    {body_name: (sx, sy, sz)}. AB-only bodies present in the stripped upload
    will appear; COMAK knee subbodies (femur_distal_r, etc.) will not.
    """

def build_scale_set(
    ab_factors: dict[str, tuple[float, float, float]],
    mode: ScalingMode = "WA",
) -> tuple[osim.ScaleSet, dict[str, float]]:
    """Construct the full ScaleSet per `mode`. Also returns the cartilage
    scale dict {mesh_name: scalar} for use by cartilage_meshes.py.

    WA: knee bodies (incl. menisci + patella, even though AB returns
        [1,1,1] for patella) get isotropic (s_wa, s_wa, s_wa) where
        s_wa = (ab_factors['femur_r'][2] + ab_factors['tibia_r'][2]) / 2.
        All other AB-provided bodies keep their anisotropic factor.
    LA: each knee body gets (s_long, s_long, s_long) from its own long axis.
    AB: pass AB factors through; knee subbodies inherit parent bone's AB factor.

    Output ScaleSet structure matches old scaleModel.py lines 49-156, with
    explicit additions for both meniscus bodies.
    """
```

```python
# nsosim/scaling/scaletool.py

from pathlib import Path
import opensim as osim

def apply_scaletool(
    base_osim: Path,
    scale_set: osim.ScaleSet,
    out_osim: Path,
    preserve_mass_distribution: bool = True,
) -> None:
    """Configure + run OpenSim ScaleTool entirely via the Python API.

    Implementation sketch:
        st = osim.ScaleTool()
        st.getGenericModelMaker().setModelFileName(str(base_osim))

        ms = st.getModelScaler()
        ms.setApply(True)
        ms.setScalingOrder(osim.ArrayStr.parse('manualScale'))
        # Adopt the prebuilt ScaleSet wholesale:
        target = ms.getScaleSet()
        for i in range(scale_set.getSize()):
            target.adoptAndAppend(scale_set.get(i).clone())
        ms.setPreserveMassDist(preserve_mass_distribution)
        ms.setOutputModelFileName(str(out_osim))

        st.getMarkerPlacer().setApply(False)   # we transplant markers ourselves
        st.run()                                # writes out_osim

    Verified empirically: handles JAM components (Blankevoort, Smith2018ContactMesh
    metadata, wraps, muscles) without crashing or stripping.

    DO NOT call `Model.scale()` instead — it crashes on Smith2018ContactMesh
    initializeMesh. See "What ScaleTool.run() already does for us" section.
    """
```

```python
# nsosim/scaling/cartilage_meshes.py

from pathlib import Path
import opensim as osim
import pymskt as mskt
import numpy as np

def scale_cartilage_meshes(
    scaled_osim: Path,
    cartilage_scales: dict[str, float],
    out_geometry_dir: Path,
) -> dict[str, Path]:
    """For each Smith2018ContactMesh in `scaled_osim`:
       - Locate its source `.stl` on disk (search `base_osim.parent/Geometry/`
         then any registered `osim.ModelVisualizer` geometry paths).
       - Load with pymskt, multiply vertices isotropically by the scale.
       - Save to `out_geometry_dir/<filename>` (overwrites the copy placed by
         the Geometry/ copy step).
       - Update the contact mesh's mesh_file via OpenSim API + printToXML.

    Pymskt vertex-scale sketch:
        m = mskt.mesh.Mesh(str(src_stl))
        m.point_coords = m.point_coords * float(scale)  # numpy mul
        m.save_mesh(str(out_stl))

    OpenSim API for mesh_file update:
        model = osim.Model(str(scaled_osim))
        cg = model.getContactGeometrySet()
        for i in range(cg.getSize()):
            mesh = cg.get(i)
            if mesh.getConcreteClassName() != 'Smith2018ContactMesh':
                continue
            scm = osim.Smith2018ContactMesh.safeDownCast(mesh)
            if scm.getName() in cartilage_scales:
                scm.set_mesh_file(new_stl_filename)
        model.printToXML(str(scaled_osim))

    DO NOT touch `scale_factors` in XML — must remain `[1,1,1]`.
    Returns {contact_mesh_name: new_stl_path}.
    """
```

```python
# nsosim/scaling/markers.py

from pathlib import Path

def swap_markerset(
    scaled_osim: Path,
    ab_scaled_osim: Path,
    backup_xml_path: Path | None = None,
) -> tuple[int, int]:
    """Replace scaled COMAK output's MarkerSet with AB's, via OpenSim API.

    Loads both models, drops the scaled model's existing markers, clones each
    AB marker (name + parent_body + local location + fixed flag) onto the
    scaled model. Drops AB markers whose parent_body doesn't exist in the
    scaled model (logs warning). Saves via printToXML.

    If `backup_xml_path` provided, the scaled model's pre-swap MarkerSet is
    written there via MarkerSet.printToXML() before mutation.

    See "Marker handling" section for the implementation sketch.
    Returns (n_added, n_dropped).
    """
```

```python
# nsosim/scaling/model_fixes.py

from pathlib import Path
import opensim as osim

def fix_itb1_attachment(model: osim.Model) -> bool:
    """Reparent ITB1 path-point 2 from tibia_r to tibia_proximal_r.

    Guard: if path_point.getParentFrame().getName() != 'tibia_r', skip
    (someone upstream fixed the source model — log and return False).

    Returns True if applied, False if already fixed.

    Long-term: see "ITB1 fix-up — why and how" — preferred path is to fix
    the source fixture once and remove this function. This helper exists
    until that PR lands.
    """
```

```python
# nsosim/scaling/__init__.py

from pathlib import Path
from typing import Optional

from .scale_factors import read_ab_factors, build_scale_set
from .scaletool import apply_scaletool
from .cartilage_meshes import scale_cartilage_meshes
from .markers import swap_markerset
from .model_fixes import fix_itb1_attachment
from .report import write_report

def scale_comak_model(
    base_osim: Path,
    ab_scaled_osim: Path,
    output_osim: Path,
    output_geometry_dir: Path,
    mode: ScalingMode = "WA",
    preserve_mass_distribution: bool = True,
    report_json: Optional[Path] = None,
) -> None:
    """Top-level orchestrator. Pipeline:
       1. read_ab_factors(ab_scaled_osim) → per-body factors
       2. build_scale_set(factors, mode) → (ScaleSet, cartilage_scales)
       3. apply_scaletool(base, scale_set, output_osim, preserve_mass_distribution)
       4. copy base Geometry/ → output_geometry_dir/
       5. scale_cartilage_meshes(output_osim, cartilage_scales, output_geometry_dir)
          — pymskt vertex scale + OpenSim API mesh_file update
       6. swap_markerset(output_osim, ab_scaled_osim) — full MarkerSet replace
       7. fix_itb1_attachment(load output_osim) → save back
       8. write_report(report_json) — factors, decisions, warnings

       Does NOT handle patella_offset.json — that's a Stage Y concern.
    """
```

```python
# nsosim/scaling/report.py

import json
from pathlib import Path
from typing import Any

def write_report(path: Path, **fields: Any) -> None:
    """Dump a JSON report at `path`. All keyword args are written as top-level
    keys. The orchestrator accumulates a dict during the pipeline and calls
    write_report(path, subject_id=..., mode=..., scale_set={...}, ...). See
    the example in "Worked example" step 8 for the expected shape.

    Creates parent directory if missing. Pretty-prints with indent=2.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(fields, f, indent=2, default=str)
```

## Verification protocol

### Baseline pattern

Every coherence test compares post-scale state against a **snapshot of the unscaled-but-also-ScaleTool-passed-through baseline**. The point is to compare against the same processing chain at `s=1`, not the original `.osim`, so XML serialization round-trips and OpenSim numerical noise cancel out:

```python
# Pseudocode used by tests
unscaled_baseline = scale_comak_model(base, ab_scaled_osim=identity_ab_osim, ...)
scaled_subject    = scale_comak_model(base, ab_scaled_osim=real_ab_osim,     ...)
# now diff scaled_subject's metric vs unscaled_baseline's metric
```

**Building the identity `ab_scaled_osim`** (a test helper, lives in `tests/scaling/conftest.py`):

```python
@pytest.fixture(scope="session")
def identity_ab_osim(tmp_path_factory):
    """An 'AB-scaled' model with all scale factors = [1,1,1].

    Built by loading the stripped Tian base, setting every Body's first
    attached_geometry scale_factors to (1,1,1), and writing to a tmp path.
    Used as the input to scale_comak_model in identity tests.
    """
    src = Path("tests/fixtures/osim_models/unscaled_generic_tian.osim")
    out = tmp_path_factory.mktemp("identity_ab") / "match_markers_and_physics.osim"

    m = osim.Model(str(src))
    bs = m.getBodySet()
    for i in range(bs.getSize()):
        body = bs.get(i)
        ag = body.upd_attached_geometry(0)
        ag.set_scale_factors(osim.Vec3(1.0, 1.0, 1.0))
    m.printToXML(str(out))
    return out
```

The stripped Tian base (`unscaled_generic_tian.osim`) is what AB takes as input, so its body+marker structure matches AB's output structure. With factors set to identity, `scale_comak_model(base=full_comak, ab_scaled_osim=this)` produces the round-tripped baseline.

Two ways snapshots are captured, depending on what's being measured:

- **Direct attribute read** (slack lengths, wrap dimensions, joint translations, marker locations): walk both models with OpenSim API, extract values into dicts keyed by name, diff.
- **Geometric distribution** (cartilage-to-bone): build a `pymskt` kd-tree on the bone mesh, query distances from cart vertices, compute distribution statistics, diff.

### Tests

- **`test_loads_and_initializes`** — output `.osim` + new `.stl`s load and `initSystem()` succeeds in `comak` conda env with JAM plugin.
- **`test_identity_scaling_roundtrip`** — Stage X with an identity AB XML (all factors `[1, 1, 1]`). Output should match input on: body mesh vertex sets (`atol=1e-6 m`), every Blankevoort `slack_length` (`atol=1e-9`), every joint translation, every wrap surface dimension, every marker location, every contact mesh `.stl` vertex set. **This is the canary** — if it fails, something in the wrapper is non-identity at identity.
- **`test_nontrivial`** — Stage X with WA factor `s_wa=0.9`. Each sub-assertion compares against the `s=1.0` baseline:
  - **Cartilage-bone proximity preserved.** For each `Smith2018ContactMesh` (femur cart, tibia cart, patella cart): build a kd-tree on the parent bone's mesh (in body-local frame), query nearest-bone-vertex distance for each cart vertex. The distribution's mean and 95th percentile should match the `s=1.0` baseline within `atol=0.5 mm`. **Catches the "bone scaled, cart STL didn't" failure mode** — the central correctness check.
  - **Ligament reference strain preserved.** For each Blankevoort ligament: `(path_length - slack_length) / slack_length` should match baseline within `atol=1e-4`. Already empirically verified for the COMAK base; this guards the wrapper against regressions.
  - **Wrap surfaces stay on parent body.** For each wrap object: its `translation` should fall within the AABB of its parent body's bone mesh.
  - ~~PathPoint linear-scaling check~~ — removed. Path points that wrap around wrap surfaces or are `MovingPathPoint`/`ConditionalPathPoint` don't necessarily scale linearly. The reference-strain preservation check above is the meaningful behavioral assertion.
- ~~`test_one_step_comak_runs`~~ — **Dropped from this plan.** Running a single COMAK forward step in a unit test pulls in too much downstream machinery (ForsimTool config, kinematics+GRF slicing, muscle activation choice, COMAK convergence behavior) for marginal additional signal beyond `test_loads_and_initializes` + the coherence checks. If COMAK breaks on a Stage X output, the smoke test (run manually, instructions below) will catch it. Pre-computed stance fixture at `tests/fixtures/scaling/rsubject121_smoke_stance.json` is kept for that smoke test.

## Worked example

Hypothetical Tian subject 1 (small): AB returns these factors in `match_markers_and_physics.osim`'s per-body `<attached_geometry>/scale_factors` fields (showing only the right-leg subset for brevity):

```
femur_r       → (0.945, 0.890, 0.945)
tibia_r       → (0.920, 0.870, 0.920)
pelvis        → (1.020, 0.950, 1.020)
talus_r       → (0.930, 0.930, 0.930)
calcn_r       → (0.940, 0.930, 0.940)
torso         → (0.980, 1.010, 0.980)
patella_r     → (1.000, 1.000, 1.000)    # AB leaves identity
# (left-side bodies elided)
```

**Step 1 — `read_ab_factors(ab_scaled_osim)`** returns this dict (verbatim).

**Step 2 — `build_scale_set(factors, mode="WA")`** computes:
```
s_wa = (femur_r[2] + tibia_r[2]) / 2
     = (0.945 + 0.920) / 2
     = 0.9325
```
and produces the following ScaleSet (logically equivalent ScaleTool XML):
```
femur_r            → (0.945, 0.890, 0.945)    # AB anisotropic, unchanged
tibia_r            → (0.920, 0.870, 0.920)    # AB anisotropic, unchanged
pelvis             → (1.020, 0.950, 1.020)    # AB anisotropic, unchanged
talus_r            → (0.930, 0.930, 0.930)    # AB anisotropic, unchanged
calcn_r            → (0.940, 0.930, 0.940)    # AB anisotropic, unchanged
torso              → (0.980, 1.010, 0.980)    # AB anisotropic, unchanged
femur_distal_r     → (0.9325, 0.9325, 0.9325) # WA isotropic — NEW
tibia_proximal_r   → (0.9325, 0.9325, 0.9325) # WA isotropic — NEW
patella_r          → (0.9325, 0.9325, 0.9325) # WA isotropic — NEW
meniscus_medial_r  → (0.9325, 0.9325, 0.9325) # WA isotropic — NEW
meniscus_lateral_r → (0.9325, 0.9325, 0.9325) # WA isotropic — NEW
```

**Step 3 — `apply_scaletool(...)`** writes the ScaleSet into a tmp ScaleTool XML and calls `ScaleTool.run()`. Output `.osim` has:
- All right-leg + body-axis bones scaled per row (femur_r shrunk by 0.945/0.890/0.945 in x/y/z, etc.).
- All 91 Blankevoort ligaments: PathPoints scaled, slack_lengths recomputed by JAM's `extendPostScale`, reference strains preserved.
- All 39 wraps scaled per parent body.
- All 7 `Smith2018ContactMesh` entries preserved; **STL vertices on disk still original** (will be fixed in step 4).
- All muscles scaled.

Also at step 2, `build_scale_set` returns a `cartilage_scales` dict (in memory):
```python
{
  "femur_cartilage":   0.9325,
  "tibia_cartilage":   0.9325,
  "patella_cartilage": 0.9325,
}
```
Optionally written to `cartilage_scales.json` alongside the report for traceability.

**Step 4 — copy base Geometry/ folder** to `output_geometry_dir/`. All bones, visual meshes copied. Cart STLs about to be overwritten in step 5.

**Step 5 — `scale_cartilage_meshes(...)`** consumes the `cartilage_scales` dict (returned from step 2), loads each STL via `pymskt`, multiplies vertices by `0.9325`, saves to `{output_geometry_dir}/{filename}` (overwrites copies from step 4). Updates `mesh_file` via `Smith2018ContactMesh.set_mesh_file()` + `Model.printToXML()`. **Leaves `scale_factors` at `[1,1,1]`** — never touch this field.

**Step 6 — `swap_markerset(scaled_osim, ab_scaled_osim, backup_path)`** saves the COMAK scaled output's original `<MarkerSet>` to backup JSON, replaces the block with AB's `<MarkerSet>` verbatim. Validates every marker's `socket_parent_frame` resolves.

**Step 7 — `fix_itb1_attachment`** loads scaled `.osim`, finds ITB1 path-point 2 on `tibia_r`, reparents to `tibia_proximal_r`, back-corrects local coords by the post-scale weld translation, saves. (Or short-circuits if upstream model has been fixed.)

**Step 8 — report.json** records:
```json
{
  "subject_id": "RSubject_121",
  "mode": "WA",
  "s_wa": 0.9325,
  "ab_scaled_osim": "/path/to/match_markers_and_physics.osim",
  "scale_set": { /* full mapping from step 2 */ },
  "cartilage_scales_json": "/path/to/cartilage_scales.json",
  "geometry_dir": "/path/to/output_geometry_dir",
  "marker_count": 41,
  "marker_backup": "/path/to/markers_pre_swap_backup.xml",
  "itb1_fixup_applied": true,
  "warnings": []
}
```

## Open questions

- **Q1** ~~Blankevoort PathPoints scaled by ScaleTool?~~ — **Yes (verified empirically).** Path lengths shrank for all four tested ligaments after `ScaleTool.run()`.
- **Q2** ~~ScaleTool scales wrap surfaces?~~ — **Yes (verified empirically).** All 39 wraps scaled (radius, length, dimensions).
- **Q3** ~~CoordinateCouplerConstraint scaling?~~ — **Resolved (verified empirically).** Absent in base COMAK model (`full_body_healthy_knee.osim` has empty `ConstraintSet`). Where present in other COMAK models, function is `LinearFunction` with dimensionless coefficients — no spatial units. **No scaling work needed.**
- **Q4** ~~Anisotropic STL scaling~~ — resolved: bone+cartilage always isotropic (user). WA produces isotropic factor; `pymskt`'s vertex-scale primitive supports both, we use isotropic.

## Determinism

Stage X is deterministic by construction — no RNG anywhere in the pipeline (it's a pure transformation of inputs). No need to wire `nsosim._determinism.set_global_seed`. Tests verify bit-identical output across two invocations with identical inputs as a sanity check, but no seed param is needed in the public API.

## Build order

1. Skeleton + identity-roundtrip test harness (`tests/scaling/test_identity.py` should fail with NotImplementedError).
2. Implement `scale_factors.py` per signatures above. Implement `scaletool.py` (writes tmp ScaleTool config, calls `ScaleTool.run()`). `test_identity.py` passes.
3. Implement `cartilage_meshes.py` — pymskt STL vertex scaling + OpenSim API `set_mesh_file()`. Cartilage-bone proximity check in `test_nontrivial.py` passes.
4. Implement `markers.py` — full MarkerSet swap from AB output. Verify `initSystem()` succeeds on the output and every marker's `socket_parent_frame` resolves.
5. Implement `model_fixes.py` — ITB1 reparent (pseudocode above). Or, if you took the fix-at-source path: this module is empty/unused.
6. Wire `__init__.py` orchestrator (`scale_comak_model`) per "Worked example" sequence. Run the full `tests/scaling/` suite — `test_identity` + `test_nontrivial` + `test_loads_and_initializes` should all pass.
7. End-to-end smoke test on RSubject_121 using AB outputs already on disk (instructions below).

## Smoke test invocation

AB outputs for RSubject_121 are already on disk (extracted from the zip you uploaded):

```
/dataNAS/people/aagatti/programming/nsosim/untracked/ab_outputs/RSubject_121 /
├── Models/
│   ├── match_markers_and_physics.osim    ← input for scale_comak_model
│   ├── match_markers_but_ignore_physics.osim
│   ├── unscaled_generic.osim              ← AB's copy of our uploaded base
│   └── Geometry/                          ← AB's *.vtp.ply files (not needed by us)
├── IK/  (per-segment .mot files + IK setup XMLs)
├── ID/  (per-segment GRF .mot + external_forces.xml)
└── MarkerData/ (per-segment .trc — already segmented by AB)
```

(Source zip: `/dataNAS/people/aagatti/projects/comak_gait_simulation/COMAK_SIMULATION_REQUIREMENTS/data/stanford_gait_retraining_data/AddBiomechanics_Results/RSubject_121 .zip`. Note the trailing space in the folder name from AB.)

**Step 1 — run Stage X:**

```python
from nsosim.scaling import scale_comak_model
from pathlib import Path

AB_DIR = Path("untracked/ab_outputs/RSubject_121 /Models")
OUT_DIR = Path("untracked/scaling_outputs/RSubject_121")

scale_comak_model(
    base_osim=Path("tests/fixtures/osim_models/full_body_healthy_knee.osim"),
    ab_scaled_osim=AB_DIR / "match_markers_and_physics.osim",
    output_osim=OUT_DIR / "scaled.osim",
    output_geometry_dir=OUT_DIR / "Geometry",
    mode="WA",
    report_json=OUT_DIR / "scaling_report.json",
)
```

**Step 2 — visual sanity check.** Open `scaled.osim` in OpenSim 4.5 GUI:
- Body proportions look right (femur/tibia shrunk per AB; pelvis larger).
- Knee cartilage sits flush on bones (no gaps or penetration).
- Markers track the AB static-trial positions.

**Step 3 — optional manual COMAK smoke run.** Out of scope for the Stage X PR. The pre-computed stance fixture is kept at [`tests/fixtures/scaling/rsubject121_smoke_stance.json`](/dataNAS/people/aagatti/programming/nsosim/tests/fixtures/scaling/rsubject121_smoke_stance.json) for whenever someone wants to manually verify a scaled model under COMAK on real walking data. Not part of `tests/scaling/` test suite.

## Out of scope for Stage X

- Marker rename/create on the AB-uploaded model (handled by `scripts/phase6_rajagopal_audit/`, upstream of AB).
- AB upload mechanics (manual web step).
- Scaling left-knee components (not simulated).
- Stage Y modifications.
- COMAK simulation infrastructure (lives in `comak_gait_simulation`).
- Predsim/opensimAD compatibility — out of scope for Paper 1; revisit if/when predsim integration becomes a goal.

### Future "Stage Z" — post-Y weld collapse for simulation speed

**Now fully specified in [`comak-weld-collapse.md`](comak-weld-collapse.md).** The sketch below is retained for context; the linked plan supersedes it.

The base COMAK model has two intermediate `WeldJoint`s: `femur_femur_distal_r` and `tibia_tibia_proximal_r`. AB's loading log explicitly flagged these: *"Creating a WeldJoint as an intermediate (non-root) joint. This will cause the gradient computations to run with slower algorithms."*

These welds exist because Stage Y (OAI knee swap) needs `femur_distal_r`/`tibia_proximal_r` as the structural seam where new knee components get inserted. **So we don't remove the welds in Stage X or Y — we keep the seam through both stages and run a post-Y collapse as a final flattening step before COMAK.**

Sketch of Stage Z (`nsosim/weld_collapse/` or similar — separate plan):
- Read the welded model (post-Y output).
- For each intermediate WeldJoint (`femur_femur_distal_r`, `tibia_tibia_proximal_r`):
  - Read the weld translation `t` (parent-frame offset to the welded child).
  - For every PathPoint / wrap surface / marker / Smith2018ContactMesh / ligament endpoint / contact-force socket on the child body: shift its local coords by `t`, reparent to the parent body.
  - Combine child mass + inertia into parent via parallel-axis theorem.
  - Delete the WeldJoint and the child body.
- Output: structurally flatter model, same physics, faster gradients.

This belongs in a separate plan. Stage X output stays welded; Stage Y output stays welded; Stage Z is opt-in and runs once at the end of the per-subject pipeline.

## References

- Old code: [`Process_Pipeline/scaleModel.py`](/dataNAS/people/aagatti/projects/gait_opensim_jam_2023/stanford_jam_gait_2023/AddBiomechanics/Process_Pipeline/scaleModel.py) (Katie's, 390 lines, primary)
- Old code: [`Process_Pipeline/scaleModel2.py`](/dataNAS/people/aagatti/projects/gait_opensim_jam_2023/stanford_jam_gait_2023/AddBiomechanics/Process_Pipeline/scaleModel2.py) (post-scale muscle param update — pattern for ligament slack)
- Old notebook: `Process_Pipeline/rescale_comak_knee.ipynb`
- Old notebook: `Process_Pipeline/update_the_model_wrapping.ipynb` (adds a new ellipsoid for KnExt; doesn't scale existing wraps)
- Old README: [`Process_Pipeline/README.md`](/dataNAS/people/aagatti/projects/gait_opensim_jam_2023/stanford_jam_gait_2023/AddBiomechanics/Process_Pipeline/README.md) (source of the contact-mesh-outside-OpenSim constraint)
- Strategy: [`combined_comak_predsim_plan.md`](/dataNAS/people/aagatti/projects/comak_gait_simulation/NOTES/combined_comak_predsim_plan.md) §2 + Engineering Challenges 1–3

---

## Completion Notes

**Date completed:** 2026-05-23

### Summary

Stage X (`nsosim.scaling`) is implemented and ships a single public entry point `scale_comak_model(base_osim, ab_scaled_osim, output_osim, output_geometry_dir, mode="WA", ...)` that takes a COMAK base + AddBiomechanics output and produces a scaled, internally-consistent COMAK base model. WA (weighted-average) mode is fully wired; LA and AB modes are documented but not implemented (no current consumer needs them). 23 tests cover scale-factor assembly, identity roundtrip, non-trivial subject coherence (cart-bone proximity, ligament reference strains, wrap surface AABBs), and end-to-end model loading on real `RSubject_121` AB outputs.

### Changes made

| Commit | Description |
|---|---|
| `69acaa0` | Add Stage X: full `nsosim/scaling/` package (config, scale_factors, scaletool, knee_geometry, markers, model_fixes, report, orchestrator) + 23 tests |
| `e1d7635` | Add `unscaled_generic_tian.osim` test fixture for the scaling suite |
| `c5e622a` | Always-write scaling report (auto-default path) + LA/AB mode docstring |
| `02cb526` | Apply autoformat |

Files added/modified:
- `nsosim/scaling/{__init__,config,scale_factors,scaletool,knee_geometry,markers,model_fixes,orchestrator,report}.py`
- `tests/scaling/{__init__,conftest,test_identity,test_loads_and_initializes,test_nontrivial,test_scale_factors}.py`
- `tests/fixtures/osim_models/unscaled_generic_tian.osim`
- `tests/fixtures/scaling/rsubject121_smoke_stance.json` (manual COMAK smoke-run metadata; not part of test suite)

### Tests

23/23 passing in `conda run -n comak python -m pytest tests/scaling/ -v` (~144s with the slow subject-level class). Coverage:
- `test_scale_factors.py` (7) — AB factor extraction, WA factor calculation, identity, anisotropic pass-through.
- `test_identity.py` (5) — `s=1` roundtrip canary: slack lengths, wrap dimensions, joint translations, mesh file names, STL vertex sets.
- `test_loads_and_initializes.py` (6) — model loads and `initSystem()` succeeds; contact meshes preserved; markers transplanted; ITB1 fix-up applied.
- `test_nontrivial.py` (5, slow) — real RSubject_121 AB factors: cart-bone proximity preserved (femur/tibia/patella), Blankevoort reference strain preserved, wrap translations within parent body AABB.

### Additional issues resolved

- **Report was silently optional → silently lossy.** After `bake_knee_geometry` resets every knee `scale_factors` to `[1,1,1]`, the model XML no longer records how much was applied. The original signature had `report_json: Optional[Path] = None` and skipped writing entirely when omitted, so the WA scale could be lost forever. Changed to always write; defaults to `output_osim.with_suffix(".scaling.json")` if not specified. (`c5e622a`)
- **LA/AB future-resurrection path documented.** `build_scale_set()` had a bare `NotImplementedError` with no docstring. Added explicit explanation of what each unimplemented mode would mean and how to extend the function — so a future maintainer can revive them without re-deriving the design. (`c5e622a`)

### Challenges / Design decisions

- **`bake_knee_geometry` vs. plan's `scale_cartilage_meshes`.** Plan envisioned a function that scaled cartilage STLs only. Implementation broadened it: pre-multiplies the vertices of *every* knee-body STL (bone, cartilage, menisci, Smith2018ContactMesh) and zeros their XML `scale_factors`. This is mandatory because the JAM `Smith2018ContactMesh` loader ignores XML scale_factors entirely — partial baking would leave the contact and visual meshes inconsistent. The function name reflects the broader "bake into the file" scope; STL on disk *is* the subject's true geometry.
- **`cartilage_scales.json` sidecar dropped.** Plan called for a debug JSON mapping each mesh name to its scale factor. With WA-only, every knee mesh receives the same `s_wa`, so the dict reduces to a list of files keyed to a single number — already captured by `wa_scale` + `knee_geometry_baked` fields in the main report. The sidecar would be redundant noise.
- **`output_geometry_dir == output_osim.parent / "Geometry"` is enforced, not advised.** Orchestrator raises if the caller passes anything else. OpenSim's default relative-path resolution finds meshes via this convention; allowing other layouts would mean either editing every `mesh_file` reference or adding a search-path setup step. Neither is worth the flexibility.
- **STL bake order may permute vertices (~1.5% on tibia bone).** The `pymskt` read → scale → write round-trip on an STL (triangle soup, no vertex indices) can reorder the vertex array without changing geometry. Verified harmless: nearest-neighbour distance is 0.0 both ways, OpenSim re-reads as triangle soup. Documented in `bake_knee_geometry` body — any future "did the mesh change" check must compare as a point *set*, not element-wise.

### Things to note for future work

- **LA and AB modes are NOT implemented.** If a downstream consumer needs anisotropic scaling or per-bone long-axis scaling, see the `build_scale_set` docstring for the extension recipe. Note that AB mode also requires generalising `bake_knee_geometry` to per-axis scaling.
- **Manual COMAK smoke run is not in the test suite.** The pre-computed stance fixture at `tests/fixtures/scaling/rsubject121_smoke_stance.json` is intentionally for ad-hoc verification on real walking data — `test_one_step_comak_runs` was explicitly dropped during planning (line 609) because the dependency surface is too large for unit testing. `TestSubjectNontrivial` provides the in-suite end-to-end coverage.
- **Patella offset is a Stage Y concern.** Stage X does not consume or produce `patella_offset.json`. PF defaults scale automatically via ScaleTool when `patella_r` is in the ScaleSet. If Stage Y runs on Stage X's output, it overwrites the PF defaults with its own per-subject patella centroid — that's correct, not a conflict.
- **Stage Z (`comak-weld-collapse`) is now its own completed plan.** Verdict from cohort timing: 0.905× median speedup — collapse is *not* adopted in the COMAK simulation pipeline, but the module is retained for AB-scaling workflows. See [`completed/comak-weld-collapse_COMPLETED.md`](completed/comak-weld-collapse_COMPLETED.md).
- **Pathways A/B/C in the original goal table** — Pathway B (Tian × OAI) is the immediate consumer. Pathway C (MRI+gait matched, knee bones excluded from body scaling) would need a flag to disable knee-body scaling; currently the WA path always scales the knee subbodies. Add a `scale_knee_bodies: bool = True` kwarg if/when Pathway C is built.
- Strategy: [`Paper Plans/paper1_strategy_Mar2026.md`](/dataNAS/people/aagatti/projects/comak_gait_simulation/NOTES/Paper%20Plans/paper1_strategy_Mar2026.md) item #11
