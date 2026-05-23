# Plan: COMAK Weld-Joint Collapse (Stage Z)

**Status:** Implemented + tested. End-to-end COMAK validation in flight.
**Created:** 2026-05-15.
**Parent:** Pathway B of [`comak_gait_simulation/.claude/plans/PAPER1_MULTIGAIT.md`](../../../comak_gait_simulation/.claude/plans/PAPER1_MULTIGAIT.md).
**Sibling:** [`comak-body-scaling.md`](comak-body-scaling.md) — Stage Z is the "Future Stage Z" sketched at the end of that plan, now fully specified.

## Implementation + validation status

- **Code:** [`nsosim/weld_collapse/`](../../nsosim/weld_collapse/) — `inertia.py`, `topology.py`, `collapse.py`, `report.py`, `__init__.py`.
- **Unit + structure tests:** [`tests/weld_collapse/`](../../tests/weld_collapse/) — 42 tests pass; sweep test confirms the collapse is bit-exact (machine-precision residuals on body transforms, marker positions, path lengths, moment arms, contact forces).
- **First end-to-end test (3 OARSI subjects, pre-cohort_timing COMAK stack, 2026-05-15):** collapsed COMAK ran 0.90–0.99x vs welded (slightly slower). Micro-benchmarks attributed this to (a) contact ~78% of per-frame cost — multibody savings can't move the total much — and (b) ~3–4 ms extra per frame from `Smith2018ContactMesh` accessing its frame through a `PhysicalOffsetFrame` instead of a `Body`. Old harness archived at [`untracked/weld_collapse_comak_test/`](../../untracked/weld_collapse_comak_test/) (see its `README.md` for the "superseded" note).
- **Cohort end-to-end test (2026-05-23):** 20-subject cohort, locked production cohort_timing pipeline (~30 min median). Cell name **`c20wc`**, in `comak_gait_simulation`:
  - Submit: [`comak_gait_simulation/tests/comak_cohort_timing/submit_cohort20_weld_collapse.sh`](../../../comak_gait_simulation/tests/comak_cohort_timing/submit_cohort20_weld_collapse.sh)
  - Compare: `compare_c20wc_vs_c20noCOR.py` (totals) + `per_stage_c20wc_vs_c20noCOR.py` (settle/sweep/comak/post).
  - **Results report:** [`comak_gait_simulation/.claude/reports/weld_collapse_c20wc_results.md`](../../../comak_gait_simulation/.claude/reports/weld_collapse_c20wc_results.md).
  - Cohort hub entry: [`tests/comak_cohort_timing/CLAUDE.md`](../../../comak_gait_simulation/tests/comak_cohort_timing/CLAUDE.md) — see the closed `c20wc` row.

## Conclusion

**Stage Z is implementation-complete and bit-exact, but does NOT speed up COMAK.** On the 20-subject cohort with the locked production pipeline, the collapsed model is **0.905× the welded baseline on the median (~10% slower)**, slower on 16/20 subjects, with every stage slower on the median (settle 0.90×, sweep 0.86×, COMAK 0.84×, post 0.91×).

The slowdown is consistent with the pre-cohort micro-benchmark: COMAK is contact-bound (~78% of per-frame cost is contact evaluation), so there's no multibody savings to capture; and the offset-frame indirection on the two `Smith2018ContactMesh` sockets costs ~3 ms/frame which accumulates to a net slowdown. Simbody's "slow gradient algorithm" path that the AddBiomechanics warning flags is real for AB but does not measurably affect generic Simbody operations (`calcSystemJacobian`) on this model, and therefore not COMAK.

**Decision:** Stage Z code and tests are kept in `nsosim/weld_collapse/` and remain available for AB-scaling workflows where the intermediate-weld warning matters. **Stage Z is not adopted in the COMAK pipeline** — welded models stay the production default. If a meaningful COMAK runtime win is the goal, the lever is the contact evaluation (mesh resolution, OBB-tree tuning), not the multibody tree.

---

## Goal

Define a self-contained "Stage Z" that takes a **final, ready-to-simulate COMAK
model** and produces a structurally flatter, physically identical model by
collapsing the two intermediate `WeldJoint`s out of the right-knee assembly.
The collapsed model removes 2 bodies and 2 joints from the multibody tree, which
removes the slow-gradient code path that AddBiomechanics (AB) flags and that a
colleague observed slowing COMAK simulations.

Stage Z is **opt-in** and runs **once, last**, after Stage X (AB scaling) and
Stage Y (OAI knee swap) — after slack lengths, contact meshes, wraps, and all
other per-subject geometry are finalized. It is a pure structural flattening: no
geometry changes, no parameter changes, no re-fitting.

## Why this exists

The base COMAK model has two intermediate `WeldJoint`s in the right-knee chain.
A `WeldJoint` as a non-root (intermediate) joint keeps a zero-DOF mobilizer in
Simbody's multibody tree; the articulated-body and gradient algorithms then fall
back to slower general code paths. AB's loading log says so explicitly:

> *"Creating a WeldJoint as an intermediate (non-root) joint. This will cause
> the gradient computations to run with slower algorithms."*

The welds exist so Stage Y has a stable structural seam (`femur_distal_r` /
`tibia_proximal_r`) where new knee components get inserted. We keep the seam
through Stages X and Y. Stage Z collapses it afterward, when nothing downstream
needs the seam anymore.

Both welds are **pure translations** (orientation `0 0 0`) — verified below —
which makes the collapse an *exact* rigid-body transform with no rotation terms.
The collapsed model is physically identical; only the kinematic-tree structure
changes.

## Verified topology of the right knee

Inspected directly from [`tests/fixtures/osim_models/full_body_healthy_knee.osim`](../../tests/fixtures/osim_models/full_body_healthy_knee.osim):

```
pelvis ── hip_r ──────────────────── femur_r        mass 8.275 kg
                                        │             COM (0,-0.150,0); carries muscle wraps, NO knee geometry
                  femur_femur_distal_r  │  WELD
                  parent_frame: femur_r_offset (on femur_r)
                  child_frame:  /bodyset/femur_distal_r
                  translation:  (-0.0055514, -0.37418, -0.0011706) m, orientation 0
                                        ▼
                                  femur_distal_r    mass 0.0081665 kg  (placeholder)
                                        │             COM (0,0,0); inertia 8.166e-4 isotropic
                                        │             carries: femur_bone + femur_cartilage visual meshes,
                                        │                       femur_cartilage Smith2018ContactMesh,
                                        │                       Capsule_r wrap
                          ┌─────────────┼─────────────┐
                      knee_r (6-DOF)               pf_r (6-DOF)
                      parent: /bodyset/femur_distal_r   parent: /bodyset/femur_distal_r
                      child:  /bodyset/tibia_proximal_r child:  /bodyset/patella_r
                          │                             │
                          ▼                             ▼
                  tibia_proximal_r                  patella_r
                  mass 0.0081665 kg (placeholder)
                  COM (0,0,0); inertia 8.166e-4 isotropic
                  carries: tibia_bone + tibia_cartilage + fibula_bone visual meshes,
                           tibia_cartilage Smith2018ContactMesh,
                           Med_Lig_r + Med_LigP_r wraps
                          │
            ┌─────────────┼──────────────┬──────────────────────────┐
   meniscus_lateral_r  meniscus_medial_r  tibia_tibia_proximal_r WELD
   parent: tibia_proximal_r_offset        parent_frame: tibia_proximal_r_offset (on tibia_proximal_r)
   parent: tibia_proximal_r_offset        child_frame:  /bodyset/tibia_r
          │              │                translation:  (0.006, 0, 0) m, orientation 0
          ▼              ▼                              ▼
   (meniscus bodies, 6-DOF joints)                  tibia_r ── ankle_r ── talus_r ── ...
                                                    mass 2.652 kg; COM (0,-0.173,0); carries muscle wraps
```

Key facts that shape the algorithm:

- **The knee bone/cartilage geometry lives on the placeholder sub-bodies**, not
  on `femur_r`/`tibia_r`. `femur_r` and `tibia_r` carry segment mass + muscle
  wraps but have **no** knee meshes. The collapse moves geometry *onto* the main
  bodies.
- The two welds have **opposite parent/child orientation**:
  - `femur_femur_distal_r`: main body `femur_r` is the **parent**, placeholder
    `femur_distal_r` is the **child**.
  - `tibia_tibia_proximal_r`: placeholder `tibia_proximal_r` is the **parent**,
    main body `tibia_r` is the **child**.
  The collapse must therefore identify "placeholder sub-body vs. main body" by
  something other than parent/child order (use mass, or an explicit name list).
- Both welds are **translation-only** (orientation `0 0 0`).
- Only the **right** knee has this structure. `knee_l`/`pf_l` are plain joints
  with no sub-bodies — Stage Z does not touch the left side.
- The meniscus joints (`meniscus_lateral_r`, `meniscus_medial_r`) socket their
  `parent_frame` to `tibia_proximal_r_offset` (a `PhysicalOffsetFrame` owned by
  each meniscus joint, whose `socket_parent` is `/bodyset/tibia_proximal_r`).
- `~71` `socket_parent_frame` references resolve to `/bodyset/femur_distal_r`
  and `~75` to `/bodyset/tibia_proximal_r` — overwhelmingly ligament/muscle
  `PathPoint`s, plus the joints, contact meshes, and attached geometry.

## Target topology after collapse

```
pelvis ── hip_r ── femur_r
                     ├── knee_r (6-DOF) ── tibia_r ── ankle_r ── talus_r ── ...
                     │                       ├── meniscus_lateral_r ── ...
                     │                       └── meniscus_medial_r  ── ...
                     └── pf_r   (6-DOF) ── patella_r
```

- `femur_distal_r` collapsed into `femur_r`; `femur_distal_r` body and
  `femur_femur_distal_r` weld deleted.
- `tibia_proximal_r` collapsed into `tibia_r`; `tibia_proximal_r` body and
  `tibia_tibia_proximal_r` weld deleted.
- `knee_r` is now `femur_r → tibia_r`; `pf_r` is now `femur_r → patella_r`.
- No `WeldJoint` remains anywhere in the model.
- 2 fewer bodies, 2 fewer joints; identical coordinate set (the `knee_r`/`pf_r`
  DOFs are untouched).

## Core idea: the offset-frame retarget

The collapse is formulated to need **almost zero coordinate arithmetic**. For
each weld, instead of editing every attached component's local coordinates:

1. Compute the rigid transform from the placeholder sub-body frame to the main
   body frame. Because welds are translation-only this is a pure translation
   `d` (a 3-vector).
2. Create **one** new `PhysicalOffsetFrame` on the main body, named
   `<sub>_collapsed` (e.g. `femur_distal_r_collapsed`), with `translation = d`
   and `orientation = 0 0 0`. This frame is rigidly fixed to the main body at
   exactly the pose the sub-body used to occupy.
3. **Retarget every socket in the model that pointed at the sub-body to point at
   this offset frame instead.** Path points, joints, contact-mesh frames,
   markers, attached geometry, and any offset frame whose `socket_parent` was
   the sub-body — all just change one connectee path. No local coordinates move,
   because the offset frame *is* the old sub-body pose.
4. Combine the sub-body's mass + inertia into the main body (parallel axis).
5. Delete the `WeldJoint` and the sub-body.

The only components that *might* need real coordinate arithmetic are
`WrapObject`s, which store a body-relative `translation`/`xyz_body_rotation`; if
they can be hosted on the `PhysicalOffsetFrame` even they need none (see "Wrap
objects" below).

**Why an extra offset frame does not defeat the speedup:** the speedup comes
from removing the zero-DOF `WeldJoint` mobilizer and the extra body from the
multibody tree. A `PhysicalOffsetFrame` is a compile-time-static transform, not
a mobilizer — it adds no DOF, no body, no constraint. The tree is genuinely
flatter.

## Computing `d` — do not parse the weld XML

Let OpenSim compute the relative transform directly. After `initSystem()`:

```python
state = model.initSystem()
sub  = model.getBodySet().get(sub_name)
main = model.getBodySet().get(main_name)
T = sub.findTransformBetween(state, main)   # maps sub-frame -> main-frame
d = T.p()                                    # SimTK::Vec3 translation
R = T.R()                                    # should be identity
assert R is ~identity  # welds are translation-only; assert and bail if not
```

Compute **every** weld's `d` from this one `initSystem()`, before collapsing any
weld. The two welds are independent, so a single realized state yields all `d` —
and once a collapse mutates the model (deletes a body + joint) the carried state
is stale. Pre-computing all `d` up front means `collapse_weld` never calls
`findTransformBetween` and no mid-loop re-`initSystem()` is needed.

For the verified base model this gives:

- `femur_distal_r → femur_r`: `d = (-0.0055514, -0.37418, -0.0011706)` m.
- `tibia_proximal_r → tibia_r`: `d = (-0.006, 0, 0)` m.

(The tibia value is `-translation` of the weld offset frame, because there the
placeholder is the *parent* — `findTransformBetween` handles the sign for us, so
the code never needs to special-case weld direction.)

## Mass + inertia combination (parallel-axis theorem)

Collapsing placeholder sub-body **B** into main body **A**, with translation `d`
from B-frame to A-frame and **no relative rotation** (welds are translation-only,
so inertia tensors share axes — no rotation of tensors needed):

```
M  = m_A + m_B
c_B_in_A = c_B + d                       # B's mass center expressed in A frame
C  = (m_A*c_A + m_B*c_B_in_A) / M        # combined mass center, in A frame

# Parallel-axis shift each body's inertia (given about its own COM) to C:
r_A = C - c_A
r_B = C - c_B_in_A
I_A_at_C = I_A + m_A * (dot(r_A,r_A)*Eye3 - outer(r_A,r_A))
I_B_at_C = I_B + m_B * (dot(r_B,r_B)*Eye3 - outer(r_B,r_B))
I_total  = I_A_at_C + I_B_at_C

A.setMass(M)
A.setMassCenter(C)
A.setInertia(Inertia(Ixx,Iyy,Izz,Ixy,Ixz,Iyz from I_total))
```

OpenSim stores `<inertia>` as the 6-vector `(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)`
about the body's mass center, expressed in the body frame. Implement the 3×3
build/parallel-axis/flatten in `inertia.py`.

For the verified base model the placeholder sub-bodies are ~8 g with isotropic
`8.166e-4` inertia and COM at origin — their contribution to a 2.7–8.3 kg
segment is negligible numerically, but the combination is done exactly for
correctness and so the same code handles Stage-X/Y-modified mass values.

## Stage Z contract

**Inputs:**
- `input_osim` — a final COMAK `.osim` (post-Stage-Y, or any model that still
  carries the welded right-knee seam; JAM components present).
- `output_osim` — where to write the collapsed `.osim`.
- `weld_names` *(optional)* — explicit list of intermediate `WeldJoint`s to
  collapse. Default `None` → auto-detect (see `find_collapsible_welds`).
- `report_json` *(optional)* — JSON sidecar path.

**Geometry:** Stage Z does **not** create or scale any `.stl`. It only edits the
`.osim`. The output model references the **same** geometry files as the input
(`mesh_file` fields unchanged). The output `.osim` must therefore be written
somewhere its existing `Geometry/` references still resolve — either alongside
the input, with the same relative `Geometry/` path, or with the input's
`Geometry/` dir registered via `osim.ModelVisualizer.addDirToGeometrySearchPaths`
(JAM's `Smith2018ContactMesh` loader honors the global geometry search path, so
this lets the collapsed `.osim` live anywhere). The orchestrator does not copy
geometry; document this and assert the resolution at load time.

**Output:**
- A collapsed `.osim`: same coordinates, same forces, same geometry, same
  markers, same contact meshes — but 2 fewer bodies, 2 fewer joints, and no
  intermediate `WeldJoint`.
- *(Optional)* `*_weld_collapse_report.json`: welds collapsed, sub/main body per
  weld, `d` per weld, combined-mass values, component counts retargeted, any
  warnings.

**Invariants:**
1. Output loads + `initSystem()` succeeds in OpenSim 4.5 with the JAM plugin.
2. Output has **no** `WeldJoint` in the `JointSet`.
3. Output `BodySet` is missing exactly the collapsed placeholder bodies; every
   other body, all coordinates, all forces, all markers preserved.
4. **Physical equivalence (the central guarantee):** for any pose set by
   identical coordinate values, every surviving body's ground transform, every
   marker's ground position, every ligament/muscle path length, every moment
   arm, and total system mass + COM match the input model to tight tolerance
   (`atol = 1e-10` starting value — finalized from the measured residual, see
   verification protocol). Verified by the sweep-equivalence test below.

## The collapse algorithm, per weld

`collapse_weld(model, weld_name, d)` — operate on an in-memory `osim.Model`:

1. **Identify sub vs. main.** From the weld's two connected bodies, the
   placeholder sub-body is the lower-mass one (or matched against the known set
   `{femur_distal_r, tibia_proximal_r}`). Main body = the other.
2. **`d` is passed in.** The orchestrator computes every weld's `d` up front
   from a single `initSystem()` (see "Computing `d`") and has already asserted
   each rotation ≈ I. Because the welds are independent, all `d` are valid
   before any collapse mutates the model — so `collapse_weld` never calls
   `findTransformBetween`, needs no realized state, and the mid-loop
   stale-state problem cannot arise. Re-assert rotation-only defensively if a
   cheap check is available.
3. **Create the offset frame.** `F = PhysicalOffsetFrame("<sub>_collapsed")`,
   `socket_parent = main body`, `translation = d`, `orientation = 0`. Add it to
   the main body (`main.addComponent(F)` so it is owned by the body).
4. **Retarget sockets.** Walk every component in the model; for each socket —
   *including a `PhysicalOffsetFrame`'s own `socket_parent`* — whose connectee
   path resolves to the sub-body (`/bodyset/<sub>`), repoint it at `F`. The rule
   is "connectee resolves to `/bodyset/<sub>`", **not** "frame owned by the
   sub-body": a joint's offset frame is owned by the *joint*, not the sub-body —
   what points at the sub-body is that offset frame's `socket_parent`, so it is
   that inner socket that must be retargeted. This covers, in the verified base
   model:
   - `knee_r` / `pf_r` `socket_parent_frame` (femur weld);
   - `knee_r` `socket_child_frame` (tibia weld);
   - the `meniscus_lateral_r` / `meniscus_medial_r` joint offset frames whose
     `socket_parent` is `/bodyset/tibia_proximal_r`;
   - ~71 / ~75 ligament & muscle `PathPoint` `socket_parent_frame`s;
   - each `Smith2018ContactMesh` `socket_frame` (`femur_cartilage` →
     `femur_distal_r`, `tibia_cartilage` → `tibia_proximal_r`);
   - `<attached_geometry>` visual meshes (`femur_bone`, `femur_cartilage`,
     `tibia_bone`, `tibia_cartilage`, `fibula_bone`);
   - any `PhysicalOffsetFrame` in `<components>` owned by the sub-body — reparent
     it onto `F`;
   - any `Marker` on the sub-body (none in the base model — confirmed — but the
     code must handle it for Stage-Y-modified models).
   Because `F` sits at exactly the old sub-body pose, **no local coordinate of
   any retargeted component changes.** The `Smith2018ContactMesh` `location`
   property is relative to its `socket_frame`, so it too is unchanged.
5. **Move wrap objects** — see "Wrap objects" below.
6. **Combine inertia** — parallel-axis, into the main body.
7. **Delete** the `WeldJoint` from the `JointSet`, then the sub-body from the
   `BodySet`. Order matters: retarget (4) and wrap move (5) first, so the only
   remaining references to the sub-body are the weld and the body itself.
8. `model.finalizeFromProperties()` then `model.finalizeConnections()`.
   Removing entries from a `Set` is a property edit, so `finalizeFromProperties`
   must run before `finalizeConnections` will resolve the model cleanly.

Run `collapse_weld` once per weld. Order between the two welds does not affect
correctness — they retarget *different sockets*: the femur collapse retargets
`knee_r`'s parent socket, the tibia collapse its child socket. (`knee_r` itself
is shared by both, so the welds are not "disjoint" — but because each touches a
different socket, order is still free.) Collapse the femur weld first for
determinism.

### Wrap objects

Each weld's collapse must **discover the sub-body's wrap objects dynamically** —
enumerate the sub-body's `WrapObjectSet` rather than hard-coding a list. In the
verified base model this finds `Capsule_r` on `femur_distal_r` and `Med_Lig_r` +
`Med_LigP_r` on `tibia_proximal_r`, but a Stage-X/Y-modified model may carry
others (extra knee-extensor or patellar-tendon wraps), so the code must not
assume that set. Each `WrapObject` stores its placement as a body-relative
`translation` + `xyz_body_rotation`.

Two viable hosting strategies — the implementer picks whichever the OpenSim 4.5
API + JAM plugin actually accept (a colleague reports the offset-frame approach
has been used successfully, so try it first):

- **Offset-frame host:** move each wrap so it is hosted on / framed to the
  `PhysicalOffsetFrame` `F`. No coordinate arithmetic — the wrap keeps its
  `translation`/`xyz_body_rotation` unchanged because `F` sits at exactly the
  old sub-body pose. Consistent with the socket-retarget strategy.
- **Main-body host:** keep each wrap in the main body's `WrapObjectSet` and add
  `d` to its `translation` (orientation unchanged — the weld has no rotation).

Either way, the wrap's owning frame changes, so the code **must update every
`PathWrap`** that references a moved wrap object: a `PathWrap`'s connectee path
changes with the wrap's owning frame (`.../femur_distal_r/Capsule_r` →
`.../femur_r/.../Capsule_r`). Enumerate all `GeometryPath`/`PathWrapSet` entries
and repoint any that resolve to a moved wrap. The sweep-equivalence test's
path-length + moment-arm checks will catch a missed `PathWrap`.

### Auto-detecting collapsible welds

`find_collapsible_welds(model)` returns the names of every `WeldJoint` that is
**not** the root joint (i.e. neither connected frame is ground). For the base
model this is exactly `["femur_femur_distal_r", "tibia_tibia_proximal_r"]`. A
root `WeldJoint` (a body welded to ground) is *not* collapsible and is left
alone — only intermediate welds trigger the slow gradient path.

## OpenSim API notes & pitfalls

- **Use the API, not raw XML edits** — consistent with the Stage X plan. Socket
  retargeting: `comp.updSocket(name).setConnecteePath(new_path)` or
  `setConnecteeName`. Enumerate sockets via `comp.getNumSockets()` /
  `getSocketNames()`.
- **`Model.scale()` is off-limits** — it crashes on `Smith2018ContactMesh`
  init. Stage Z never scales, so this does not arise, but do not reach for it.
- **Body/joint removal:** `JointSet.remove(idx)` and `BodySet.remove(idx)` exist
  in OpenSim 4.5. If in-place `Set.remove` proves unreliable for a model with
  JAM components, the fallback is to rebuild the `BodySet`/`JointSet` excluding
  the collapsed entries, or a final targeted XML-element delete *after* all API
  edits. Prefer the API; document whichever path works.
- Call `finalizeConnections()` after retargeting and before `printToXML()`.
- `findTransformBetween(state, otherFrame)` is the safe way to get `d`; do not
  hand-derive it from weld offset-frame translations (the parent/child
  asymmetry between the two welds makes hand-derivation error-prone).
- The femur weld `d` has a large Y component (`-0.374 m`, hip-to-knee). This is
  purely numeric — knee geometry local coordinates become large after the
  offset frame resolves — and is not a physics issue. Do not "fix" it.

## Risks / things to audit before trusting Stage Z output

- **COMAK / JAM setup files.** Any ForsimTool / COMAK-IK / COMAK setup XML,
  reporter, or analysis config that references `femur_distal_r` or
  `tibia_proximal_r` **by body name** will break against a collapsed model.
  These live in `comak_gait_simulation`, not this repo. Stage Z must ship with a
  documented note: *consumers of a collapsed model must use the collapsed body
  names (`femur_r`, `tibia_r`).*
  **Grep finding (2026-05-15, `comak_gait_simulation`):** the one config
  field that breaks is the JAM `JointMechanicsTool` setup
  (`joint_mechanics_settings.xml`):
  `<attached_geometry_bodies>/bodyset/femur_distal_r /bodyset/tibia_proximal_r
  /bodyset/patella_r</attached_geometry_bodies>` — consumers must change it to
  `/bodyset/femur_r /bodyset/tibia_r /bodyset/patella_r`. `updateJAMmodel.py`
  also references both names, but it *builds* the welded base model (runs
  before Stage Z) — no change needed there. No ForsimTool / COMAK-IK setup XML
  references the placeholder bodies by name.
- **`socket_scale_frame`** on the `Smith2018ContactMesh` entries is `/ground` in
  the base model — no action — but if Stage X/Y set it to a sub-body, retarget
  it too.
- **ITB1 fix-up interaction.** The Stage X plan reparents ITB1 path-point 2 onto
  `tibia_proximal_r`. If that point exists on `tibia_proximal_r` when Stage Z
  runs, the generic socket retarget (step 4) handles it automatically — it is
  just one more `PathPoint` socketed to the sub-body. No special case needed,
  but the sweep test should include ITB1 explicitly.
- **`MovingPathPoint` / `ConditionalPathPoint`.** A `MovingPathPoint`'s location
  is driven by functions of a coordinate, not a static `location`; a
  `ConditionalPathPoint` is gated by a coordinate range. Both still socket to a
  parent frame — retargeting the frame to `F` preserves them exactly (the
  functions are unitless coordinate→length maps). The retarget is frame-only, so
  no function rescaling is needed. The sweep test covers them via path length.
- **Determinism:** Stage Z is a pure transform of inputs — no RNG. Two runs on
  identical input produce byte-identical output (modulo OpenSim XML float
  formatting). No seed parameter.

## Code organization

```
nsosim/weld_collapse/
    __init__.py        # public API: collapse_welds()
    topology.py        # find_collapsible_welds(); identify_sub_main()
    inertia.py         # combine_inertia() parallel-axis helper (pure numpy)
    collapse.py        # collapse_weld(model, weld_name, d) — the core operation
    report.py          # write_report() JSON sidecar

tests/weld_collapse/
    conftest.py                  # collapsed-model session fixture
    test_topology.py             # weld detection, sub/main identification
    test_inertia.py              # parallel-axis combination vs. analytic cases
    test_structure.py            # collapsed model: no welds, correct body/joint counts
    test_loads_and_initializes.py
    test_sweep_equivalence.py     # @pytest.mark.slow — the central correctness test
```

`__init__.py` exposes one public function. Users do
`from nsosim.weld_collapse import collapse_welds`.

Per CLAUDE.md test conventions: tests run via
`conda run -n comak python -m pytest tests/weld_collapse/ -v`; mark the
full-model sweep test `@pytest.mark.slow`.

## Function signatures

```python
# nsosim/weld_collapse/topology.py

import opensim as osim

def find_collapsible_welds(model: osim.Model) -> list[str]:
    """Return names of every intermediate (non-root) WeldJoint in the model.

    A WeldJoint is collapsible iff neither of its connected frames resolves to
    ground. For full_body_healthy_knee.osim this returns
    ['femur_femur_distal_r', 'tibia_tibia_proximal_r'].
    """

def identify_sub_main(model: osim.Model, weld_name: str) -> tuple[str, str]:
    """Return (sub_body_name, main_body_name) for a weld.

    The placeholder sub-body is the lower-mass of the two welded bodies. Raises
    if the weld does not connect exactly two bodies.
    """
```

```python
# nsosim/weld_collapse/inertia.py

import numpy as np

def combine_inertia(
    m_a: float, com_a: np.ndarray, inertia_a: np.ndarray,   # 6-vec, about com_a
    m_b: float, com_b: np.ndarray, inertia_b: np.ndarray,   # 6-vec, about com_b
    d: np.ndarray,                                          # b-frame -> a-frame translation
) -> tuple[float, np.ndarray, np.ndarray]:
    """Combine body B into body A (no relative rotation; welds are translation-
    only). Returns (combined_mass, combined_com_in_a_frame, combined_inertia_6vec
    about the combined COM). Pure numpy — no OpenSim dependency, fully unit-
    testable against analytic two-point-mass cases.
    """
```

```python
# nsosim/weld_collapse/collapse.py

import numpy as np
import opensim as osim

def collapse_weld(model: osim.Model, weld_name: str, d: np.ndarray) -> dict:
    """Collapse one intermediate WeldJoint in-place.

    `d` is the sub-frame -> main-frame translation, precomputed by the
    orchestrator from a single initSystem() before any collapse mutates the
    model (see plan "Computing d"). collapse_weld therefore does NOT call
    findTransformBetween and needs no realized state.

    Steps (see plan "The collapse algorithm, per weld"):
      1. identify sub/main bodies
      2. d is passed in (orchestrator already asserted rotation ~= identity;
         re-assert defensively if cheap)
      3. create PhysicalOffsetFrame '<sub>_collapsed' on main body at d
      4. retarget every socket resolving to /bodyset/<sub> onto that frame
      5. move wrap objects (+ update referencing PathWraps)
      6. combine_inertia into the main body
      7. delete the WeldJoint and the sub-body
      8. model.finalizeFromProperties() + model.finalizeConnections()

    Returns a per-weld report dict (sub, main, d, combined mass, counts of
    retargeted path points / joints / contact meshes / wraps / markers).
    Does NOT call printToXML — the orchestrator owns I/O.
    """
```

```python
# nsosim/weld_collapse/__init__.py

from pathlib import Path
from typing import Optional

import opensim as osim

from .topology import find_collapsible_welds
from .collapse import collapse_weld
from .report import write_report

def collapse_welds(
    input_osim: Path,
    output_osim: Path,
    weld_names: Optional[list[str]] = None,
    report_json: Optional[Path] = None,
) -> dict:
    """Top-level Stage Z orchestrator.

       1. Load input_osim; initSystem().
       2. weld_names = weld_names or find_collapsible_welds(model).
       3. From that single initSystem(), compute d for every weld via
          findTransformBetween and assert each rotation ~= identity. The welds
          are independent, so all d come from one realized state — no
          re-initSystem() is needed once collapsing starts.
       4. For each weld (femur first, for determinism):
          collapse_weld(model, w, d[w]).
       5. model.finalizeFromProperties(); model.finalizeConnections();
          model.printToXML(output_osim).
       6. Reload output_osim + initSystem() as a self-check; assert no WeldJoint.
       7. write_report(report_json, ...) if requested.

       Geometry files are NOT copied or modified; output_osim must be written
       where its inherited Geometry/ references still resolve.

       Returns the aggregate report dict.
    """
```

## Verification protocol

### Why the equivalence must be near-exact

Both welds are rigid translations and every retargeted component moves rigidly
with its frame. The collapse changes the multibody-tree *structure* but not the
*physics*. So the collapsed model is not "similar" to the input — it is the same
mechanism.

The base fixture stores translations at full 17-digit precision (e.g.
`-0.0055513564376633642`), so the `.osim` XML round-trip is an exact IEEE-754
double — the writer does **not** truncate. The only genuine residual is float
summation-order differences between a shorter tree (collapsed) and a longer one
(welded): order ~1e-14–1e-13 m accumulated over a leg-length chain.

Therefore the sweep test **starts tight — `atol = 1e-10 m`** — and the
implementer must read the actual measured residual on the first run and set the
final tolerance from it, documenting the true measured reason. Do not
pre-emptively loosen: per the repo testing guideline, a tolerance is documented
only with a genuine, measured reason. If the residual genuinely needs `1e-7`
that is now evidence-backed; if it is `1e-12` the tighter check stays. A
speculative `1e-7` would mask a real ~micron-scale path-point bug while catching
only gross mm-scale errors. The contact-force `rtol=1e-6` (below) is separate
and correctly looser — that one reflects the JAM contact solver, not kinematics.

### The sweep-equivalence test (`test_sweep_equivalence.py`, `@pytest.mark.slow`)

This is the central correctness test and implements the user's proposed check:
sweep the knee through a range of motion before and after collapse, confirm a
downstream marker lands in the exact same place and all moment arms are
unchanged.

**Fixture.** `full_body_healthy_knee.osim` (it carries both welds). Build the
collapsed model once in a `scope="module"` fixture:

```python
@pytest.fixture(scope="module")
def models(tmp_path_factory):
    base = Path("tests/fixtures/osim_models/full_body_healthy_knee.osim")
    # Register the fixture Geometry/ dir on OpenSim's global geometry search
    # path, then write the collapsed model to a real tmp dir. OpenSim — and the
    # JAM Smith2018ContactMesh loader — resolve mesh_file against registered
    # search paths, so the collapsed .osim finds its meshes without being
    # written into (and polluting) the tracked fixture dir. Same mechanism the
    # Stage X plan relies on. No copy, no fixture-dir pollution, no -n race.
    osim.ModelVisualizer.addDirToGeometrySearchPaths(
        str((base.parent / "Geometry").resolve()))
    out = tmp_path_factory.mktemp("wc") / "collapsed.osim"
    collapse_welds(base, out)
    m_in  = osim.Model(str(base));  m_in.initSystem()
    m_out = osim.Model(str(out));   m_out.initSystem()
    return m_in, m_out
```

**Pose grid.** The collapsed model has the identical coordinate set, so the same
`q` is set on both. Sweep:
- `knee_flex_r` over e.g. `np.linspace(0, 90, 19)` degrees;
- the 5 secondary knee coordinates each set to their default value (and one pass
  with each perturbed off-default, to exercise non-trivial poses);
- `pf_*` coordinates at default;
- a couple of non-default hip/ankle poses, so the comparison is not all
  axis-aligned at the seam.

For each pose, set every coordinate by name on both models, then
`realizePosition` (and `realizeDynamics` for the mass/COM checks).

**Per-pose assertions** (input vs. collapsed, `atol = 1e-10` *starting value* —
see "Why the equivalence must be near-exact"; set the final value from the
measured residual on first run — unless noted):

1. **Downstream body transforms.** For every body that exists in both models
   (`patella_r` — downstream of the femur weld; `tibia_r`, `meniscus_*_r`,
   `talus_r`, `calcn_r`, `toes_r` — downstream of the tibia weld):
   `body.getTransformInGround(state)` — compare translation (`atol=1e-10 m`) and
   rotation matrix (`atol=1e-10`).
2. **Marker ground positions.** Every marker present in both models:
   `marker.getLocationInGround(state)` — `atol=1e-10 m`. Include at least one
   marker on `calcn_r`/`toes_r` (distal-foot, downstream of the tibia weld) and
   one on `patella_r`.
3. **Ligament & muscle path lengths.** For every `Force` with a `GeometryPath`
   (all 91 Blankevoort ligaments + the muscles), `getLength(state)` —
   `atol=1e-10 m`. Explicitly assert ITB1 is present and matches.
4. **Moment arms.** For each knee coordinate and each force with a path,
   `path.computeMomentArm(state, coord)` — `atol=1e-10 m`. This is the assertion
   that catches a missed `PathWrap` or a mis-retargeted wrap object.
5. **Total mass + whole-model COM.** `model.getTotalMass(state)` and the
   mass-weighted COM in ground — `atol=1e-10`. Guards the parallel-axis
   combination.

**One-pose contact check.** At a single representative stance pose, compare each
`Smith2018ArticularContactForce` resultant (force magnitude on the cartilage
pairs) between the two models — `rtol=1e-6` (contact solver tolerance, looser
than kinematics, documented). This guards that the contact-mesh `socket_frame`
retarget did not move the cartilage.

### Structural tests (`test_structure.py`, fast)

- `test_no_weldjoint`: collapsed `JointSet` contains zero `WeldJoint`s.
- `test_bodies_removed`: collapsed `BodySet` is exactly the input `BodySet`
  minus `{femur_distal_r, tibia_proximal_r}`; every other body name preserved.
- `test_joint_reparenting`: `knee_r` parent resolves to a frame on `femur_r`,
  child to a frame on `tibia_r`; `pf_r` parent resolves to a frame on `femur_r`.
- `test_coordinates_preserved`: collapsed model has the identical set of
  coordinate names as the input.
- `test_counts`: input has 2 collapsible welds; output has 0.

### `test_loads_and_initializes.py`

Collapsed `.osim` loads and `initSystem()` succeeds in the `comak` conda env
with the JAM plugin.

### `test_topology.py` / `test_inertia.py` (fast unit tests)

- `find_collapsible_welds` returns exactly the two expected welds on the base
  model; returns `[]` on a model with only a root weld.
- `identify_sub_main` returns `(femur_distal_r, femur_r)` and
  `(tibia_proximal_r, tibia_r)`.
- `combine_inertia` checked against analytic cases: two equal point masses at
  `±a` along an axis → known combined inertia; identity case (`m_b=0`) returns
  body A unchanged; a hand-computed two-block case.

## Build order

1. `inertia.py` + `test_inertia.py` — pure numpy, no OpenSim. Fast to get right.
2. `topology.py` + `test_topology.py` — weld detection, sub/main ID.
3. `collapse.py` — implement `collapse_weld` incrementally:
   a. offset frame + socket retarget for joints and path points;
   b. contact-mesh + attached-geometry retarget;
   c. wrap-object move + `PathWrap` update;
   d. inertia combination;
   e. weld + body deletion.
   After (a)–(e), `test_structure.py` + `test_loads_and_initializes.py` pass.
4. `__init__.py` orchestrator + `report.py`.
5. `test_sweep_equivalence.py` — the central test. If a moment-arm or path-
   length assertion fails, it almost certainly points at a missed `PathWrap`
   (3c) or a mis-retargeted socket (3a/3b).
6. Manual: grep `comak_gait_simulation` setup configs for `femur_distal_r` /
   `tibia_proximal_r`; document any consumer that must switch to collapsed body
   names. Optionally time a COMAK run on a collapsed vs. welded model to
   quantify the speedup.

## Out of scope for Stage Z

- Any geometry (`.stl`) creation or scaling — Stage Z is `.osim`-only.
- Collapsing the left knee — it has no intermediate welds.
- Collapsing a root `WeldJoint` (a body welded to ground) — not a slow-gradient
  source; left untouched.
- Modifying COMAK simulation setup files in `comak_gait_simulation` to use the
  collapsed body names — that is a consumer-side change, flagged here but done
  there.
- Re-deriving slack lengths / contact params — Stage Z assumes the input model
  is already final.
- Predsim / opensimAD compatibility.

## References

- Sibling plan: [`comak-body-scaling.md`](comak-body-scaling.md) — Stage X; its
  "Future Stage Z" section is the seed of this plan.
- Base model inspected: [`tests/fixtures/osim_models/full_body_healthy_knee.osim`](../../tests/fixtures/osim_models/full_body_healthy_knee.osim)
  — joint set lines 4399–7577; `femur_femur_distal_r` weld at 4686,
  `tibia_tibia_proximal_r` weld at 5306, `knee_r` at 4858, `pf_r` at 4712.
- ITB1 reparent precedent: [`scaleModel.py`](/dataNAS/people/aagatti/projects/gait_opensim_jam_2023/stanford_jam_gait_2023/AddBiomechanics/Process_Pipeline/scaleModel.py) lines 206–260 — reads a weld translation and reparents a path point; the same primitive Stage Z generalizes.
- Strategy: [`combined_comak_predsim_plan.md`](/dataNAS/people/aagatti/projects/comak_gait_simulation/NOTES/combined_comak_predsim_plan.md) — Engineering Challenges (intermediate welds, ITB1 body).
- Existing tests touching the base fixture: [`tests/test_knee_assembly.py`](../../tests/test_knee_assembly.py), [`tests/test_knee_assembly_forsim.py`](../../tests/test_knee_assembly_forsim.py).
