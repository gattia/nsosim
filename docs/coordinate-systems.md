# Coordinate Systems & the Knee-Build Pipeline

This page is the canonical reference for the coordinate spaces a knee build passes through,
the transform chain that connects them, and how body scaling interacts with the knee
geometry. Its companion, [Knee sizing modes](deviations.md), catalogs the ways you might
size a knee placed into a model, how each is achieved, and which are built today.

Function names link to the [API reference](reference/nsm_fitting.md), generated from the
docstrings — so they move with the code and never point at a stale line.

---

## What the pipeline is trying to do (two scenarios)

The library takes a knee (bone + cartilage geometry, from an MRI fit or a decoded latent)
and drops it into a whole-body OpenSim/COMAK model so it can be simulated under a gait. The
right thing to do with the knee's **size** depends on whose body it's going into:

| Scenario | Gait body vs. MRI knee | What we want the knee size to be | Status |
|---|---|---|---|
| **Cross-subject** | *different* people (e.g. an OAI knee simulated under someone else's gait) | roughly matched to the gait body — scale the knee to the body | **partially built — see the bug below** |
| **Matched-subject** | the *same* person (the MRI subject *is* the gait subject) | the knee's true anatomical (MRI) size — don't normalize it away | **not implemented yet** |

Both scenarios start the same way: every subject knee is **similarity-registered onto one
fixed reference knee**, which divides the subject's true size out (it lands at "reference
size"). They differ only in what scale is restored at the end:

- **Cross-subject** wants the knee scaled to roughly fit the body it's being placed in.
- **Matched-subject** wants the knee's original true size kept (or restored).

!!! bug "Current state: neither scenario gets the right size"
    Today the knee comes out at **reference size** in both cases — it is neither scaled to
    the gait body (cross-subject) nor kept at true MRI size (matched-subject). For
    cross-subject this is an active bug (the body is scaled but the swapped-in knee is not);
    see [§5](#5-comak-body-scaling-and-how-it-meets-the-knee-build) and
    [Knee sizing Mode 2](deviations.md#mode-2-subject-knee-mri-sized-to-the-gait-body).
    Matched-subject is simply unbuilt.

The rest of this page describes the mechanics that make the above happen.

---

## 1. The coordinate spaces

Every point array in this pipeline lives in one of four spaces. The **scale identity**
column is the one people miss: a mesh can be shape-personalized while being
size-normalized to the reference.

| Space | Definition | Units | Scale identity |
|---|---|---|---|
| **MRI** | Subject segmentation mesh as it comes off the scanner | mm | **subject-physical** — the subject's true anatomical size |
| **REFALIGN** | Subject mesh after **similarity** registration onto the fixed `smith2019` reference bone (a.k.a. "femur-aligned mm") | mm | **reference size** — subject's true scale divided out by the similarity scale |
| **NSMcanon** | NSM training-normalized box ~[−1, 1]; each bone has its own canonical space | dimensionless | per-bone canonical |
| **OSIM** | OpenSim body-local frame, produced by `convert_nsm_recon_to_OSIM_` (see [§4](#4-converter-reference)) | m | reference size, rotated into OpenSim axes |

The **MRI → OSIM** orientation change is a single fixed rotation/axis-swap,
**`OSIM_TO_NSM_TRANSFORM`** (defined in `nsosim.nsm_fitting`): `x→−y, y→z, z→−x`. It is what
the "rotated into OpenSim axes" in the OSIM row above refers to — applied in every
REFALIGN→OSIM conversion.

!!! info "Body-local frames and the knee joint center"
    "OSIM" *is* a body-local OpenSim frame. For the knee sub-bodies (`femur_distal_r`,
    `tibia_proximal_r`, `patella_r`, the two menisci) the body-local origin is the **knee
    joint center** — the `knee_r` / `pf_r` CustomJoint frames sit at `(0,0,0)` there, not at
    the bone centroid. A subject recon STL and the reference STL on the same body therefore
    share their frame origin by construction, which matters when something scales the
    geometry about that origin ([§5](#5-comak-body-scaling-and-how-it-meets-the-knee-build)).

---

## 2. The MRI / fitting transform chain (subject mesh → OpenSim model)

Each box is a **data state** (with `units · scale identity` on the right); each **STEP** is
the function that moves you to the next box and what it does.

```
MRI mesh ───────────────────────────────────────────── mm · subject's true size
   │
   │ STEP  align_bone_osim_fit_nsm
   │   • femur — similarity-register onto the fixed smith2019 reference bone
   │     (rigid + isotropic scale; the isotropic scale divides the subject's true size out)
   │   • tibia & patella — REUSE the femur's transform; they are NOT re-registered,
   │     which keeps the three bones in one shared frame
   ▼
REFALIGN mesh ──────────────────────────────────────── mm · reference size
   │
   │ STEP  fit_nsm  (NSM: optimize a latent, then decode it)
   │   the decoder (create_mesh_adaptive) undoes only the NSM normalization,
   │   so the reconstruction comes back in REFALIGN — NOT the subject's true size
   ▼
REFALIGN reconstruction ────────────────────────────── mm · reference size
   │   stored at dict_bones[bone]['subject']['*_mesh_nsm'] (saved as *_nsm_recon_mm.vtk)
   │
   │ STEP  nsm_recon_to_osim
   │   per surface, convert_nsm_recon_to_OSIM_ :  + ref-center,  mm→m,  axis-swap
   │   (no scale step — the size identity is carried through unchanged)
   ▼
OSIM meshes ────────────────────────────────────────── m · reference size
   │
   │ STEP  model_building builders (all run here, in OSIM metres at reference size)
   │   articular surfaces · ligament interpolation · wrap fitting · menisci · fat pad
   │   center_patella_meshes — subtracts mean_patella (an OSIM-metre offset, reference size)
   │
   │ STEP  build_joint_model → finalize_osim_model → update_osim_model
   │   copy the new STLs into the model's Geometry/, then repoint the .osim's
   │   mesh references at them (scale_factors left at 1,1,1)
   ▼
Subject-specific COMAK .osim ───────────────────────── knee surfaces: m · reference size
```

The same chain as a call tree (a function in parentheses is a **sub-function called by the
one before it**):

- [`align_bone_osim_fit_nsm`][nsosim.nsm_fitting.align_bone_osim_fit_nsm] — per bone, driven by
  [`align_knee_osim_fit_nsm`][nsosim.nsm_fitting.align_knee_osim_fit_nsm]
- [`nsm_recon_to_osim`][nsosim.nsm_fitting.nsm_recon_to_osim]
- [`build_joint_model`][nsosim.model_building.build_joint_model] — calls
  [`save_geometry_files`][nsosim.model_building.save_geometry_files],
  [`center_patella_meshes`][nsosim.model_building.center_patella_meshes],
  [`finalize_osim_model`][nsosim.model_building.finalize_osim_model]
- [`update_osim_model`][nsosim.comak_osim_update.update_osim_model] — calls
  [`update_body_geometry_meshfile`][nsosim.osim_utils.update_body_geometry_meshfile],
  [`update_contact_mesh_files`][nsosim.osim_utils.update_contact_mesh_files]

!!! info "Why the knee comes out at reference size"
    The similarity registration divides the subject's true size out (→ REFALIGN), and
    **nothing on the MRI path puts a size back**: `nsm_recon_to_osim` uses the underscore
    converter, which has no scale step ([§4](#4-converter-reference)). So the final model's
    contact / wrap / ligament geometry is **shape-personalized but size-normalized to the
    reference**. Restoring an appropriate size — body scale (cross-subject) or true MRI size
    (matched-subject) — is the missing piece; see
    [§5](#5-comak-body-scaling-and-how-it-meets-the-knee-build) and the
    [knee sizing modes](deviations.md).

!!! info "`reg_mode` (the size-normalization switch)"
    The similarity scale only happens when `reg_mode='similarity'` — which is the **default**
    on [`align_knee_osim_fit_nsm`][nsosim.nsm_fitting.align_knee_osim_fit_nsm] /
    [`align_bone_osim_fit_nsm`][nsosim.nsm_fitting.align_bone_osim_fit_nsm] (matching every
    production caller). Pass `reg_mode='rigid'` to register with rotation + translation only,
    which **preserves the subject's true size** (no normalization) — the natural starting
    point for the matched-subject scenario.

---

## 3. The synthetic-decode transform chain (latent → OpenSim)

Used to build a knee from an arbitrary latent — synthetic joints, shape-mode visualization,
latent interpolation — with **no MRI target**. It is the inverse direction of fitting and,
crucially, uses a **different converter** (the non-underscore one).

```
latent ─────────────────────────────────────────────── NSMcanon · dimensionless
   │
   │ STEP  decode_latent_to_osim
   │   create_mesh (NSM)                         → NSMcanon meshes
   │   convert_nsm_recon_to_OSIM (NON-underscore):
   │       undo_transform           NSMcanon → REFALIGN   (called with scale=1, center=0)
   │       convert_nsm_recon_to_OSIM_   REFALIGN → OSIM
   ▼
OSIM meshes ────────────────────────────────────────── m · reference size
   │
   │ STEP  build_joint_model  (the SAME assembler the MRI path uses)
   ▼
Subject-specific COMAK .osim
```

[`decode_latent_to_osim`][nsosim.decode.decode_latent_to_osim] handles **one bone**: it
decodes the latent to NSMcanon meshes and converts each to OSIM.
[`decode_joint_from_descriptors`][nsosim.decode.decode_joint_from_descriptors] builds a
**whole joint** from three latents (femur, tibia, patella) plus *relative* transforms
(`T_rel`) that say how the tibia and patella sit relative to the femur. It first turns each
`T_rel` back into a per-bone alignment transform
([`recover_bone_transform`][nsosim.transforms.recover_bone_transform]), then calls
`decode_latent_to_osim` once per bone — so the three bones land in one consistent joint
configuration instead of being decoded in isolation.

!!! info "The scale hook — why it matters for the fix"
    The non-underscore
    [`convert_nsm_recon_to_OSIM`][nsosim.nsm_fitting.convert_nsm_recon_to_OSIM] forwards a
    `scale` argument into [`undo_transform`][nsosim.nsm_fitting.undo_transform]. A non-unit
    `scale` there applies an isotropic resize. The synthetic path passes `scale=1`, and the
    MRI path doesn't call this converter at all — so **nothing currently resizes a knee** —
    but the capability exists.

    This is the natural lever for the planned fix. Instead of pre-scaling the *reference*
    knee and registering each subject to that scaled target, the cleaner design is to always
    similarity-register to the **one native reference knee**, then resize the result through
    this `scale` hook — up/down to the gait body (cross-subject) or back to true size
    (matched-subject). See the [knee sizing modes](deviations.md).

---

## 4. Converter reference

There are two converters per direction. The **underscore** version does only the
REFALIGN↔OSIM leg (centre/units/orientation, no scale). The **non-underscore** version is
the full canonical↔OSIM round-trip: it wraps the underscore version *and* the
NSMcanon↔REFALIGN leg (`undo_transform` / `apply_transform`), and it is the only one that
exposes a `scale`.

| Converter | What it does | Space in → out |
|---|---|---|
| [`convert_nsm_recon_to_OSIM_`][nsosim.nsm_fitting.convert_nsm_recon_to_OSIM_] (underscore) | REFALIGN → OSIM only: `+ ref-center`, mm→m, axis-swap. **No scale.** | mm, refsize → m, refsize |
| [`convert_OSIM_to_nsm_`][nsosim.nsm_fitting.convert_OSIM_to_nsm_] (underscore) | exact inverse of the above | m, refsize → mm, refsize |
| [`convert_nsm_recon_to_OSIM`][nsosim.nsm_fitting.convert_nsm_recon_to_OSIM] (no underscore) | full path: `undo_transform` (canonical→REFALIGN, **applies `scale`**) then the underscore leg | NSMcanon → m |
| [`convert_OSIM_to_nsm`][nsosim.nsm_fitting.convert_OSIM_to_nsm] (no underscore) | full inverse: underscore leg then `apply_transform` (REFALIGN→canonical) | m → NSMcanon |
| [`undo_transform`][nsosim.nsm_fitting.undo_transform] | NSMcanon → REFALIGN, via the per-bone `linear_transform` (the only place a `scale` enters) | canon → mm |
| [`apply_transform`][nsosim.nsm_fitting.apply_transform] | REFALIGN → NSMcanon (inverse of `undo_transform`) | mm → canon |

In short: the **MRI path** already has its mesh in REFALIGN, so it only needs the
underscore leg. The **synthetic/decode path** starts in NSMcanon, so it needs the full
non-underscore converter (which is also where a resize could be applied).

!!! info "What `+ ref-center` is, and where it comes from"
    `ref-center` is **`fem_ref_center`** — the centroid of the *reference* femur in
    NSM-oriented mm, before that reference mesh was centred. It was computed **once**, when
    the NSM was fit to the reference surfaces, and stored as `mean_orig` in
    `ref_femur_alignment.json` (alongside `transform_matrix` and `orig_scale`). Value:
    ≈ `[-1.22, -10.94, 8.20]`.

    During reference fitting the reference mesh was centred (its `mean_orig` subtracted);
    REFALIGN inherits that same centring. So to put a subject recon back into the correct
    absolute OSIM position, the converter **adds `mean_orig` back**. The *same* reference
    value is used for **all three bones** (not a per-subject, per-bone centre) — that is
    exactly what preserves their relative spatial layout (e.g. tibia ~50 mm distal to the
    femur condyles) when they land in OSIM.

**Per-bone `linear_transform`** (in each bone's `*_alignment.json`) maps REFALIGN → that
bone's NSMcanon space. Its 3×3 block is `scale · R`, where **`R` is a proper rotation matrix**
(det +1) and `scale` is a single uniform factor (the same along every column). The JSON's
separate `scale` and `center` fields are always `1` and `[0,0,0]` because the whole
similarity (rotation + uniform scale + centring) is baked into the 4×4. See the repo-root
`CLAUDE.md` "Per-bone linear_transform" section for the full structure.

---

## 5. COMAK body scaling, and how it meets the knee build

**COMAK body scaling** ([`nsosim.scaling`](reference/scaling.md), referred to as "Stage X"
in the scaling code and plans) is a **separate** step from the knee build. It sizes the
whole-body COMAK model to a subject, and the two steps meet at the knee sub-bodies.

!!! info "What 'COMAK body scaling' actually is"
    AddBiomechanics (AB) is run on a *stripped* version of the COMAK model (bones + markers,
    none of the COMAK contact / ligament / wrap machinery) to fit a subject's motion-capture
    and physics data. AB outputs per-body **scale factors** and physics-tuned per-body
    **masses**. COMAK body scaling reads those AB outputs and applies them to the **full**
    COMAK base model (the same model, with all the COMAK components) — so you get a
    subject-sized COMAK model without re-deriving the scaling.
    [`scale_comak_model`][nsosim.scaling.orchestrator.scale_comak_model] is the entry point.

### `s_wa` — the one knee scale factor
The knee sub-bodies were stripped before the AB run, so AB has no scale factor for them.
[`build_scale_set`][nsosim.scaling.scale_factors.build_scale_set] synthesizes one:

```
s_wa = (ab_factors['femur_r'][2] + ab_factors['tibia_r'][2]) / 2
```

i.e. the mean of the femur and tibia **long-axis** AB scale factors — a single isotropic,
dimensionless ratio (`1.0` = no change). Every knee sub-body gets `s_wa`; the AB-provided
bodies keep their own per-axis factors (except `patella_r`, which AB reports as identity, so
it also gets `s_wa`).

### Masses
The full pipeline **does** set subject masses: after ScaleTool runs, the orchestrator's
`_apply_per_body_masses` writes AB's physics-tuned per-body masses onto the model
(renormalized so the total matches the subject), and rescales each body's inertia to match.
(ScaleTool itself is run *without* `setSubjectMass`, so the mass change is entirely this
later pass — geometry scaling and mass transfer are deliberately separated.)

### The knee geometry "bake"
[`bake_knee_geometry`][nsosim.scaling.knee_geometry.bake_knee_geometry] multiplies each knee
STL's vertices by `s_wa`, about the body-local origin `(0,0,0)` (= the knee joint center,
see [§1](#1-the-coordinate-spaces); not the bone centroid), and resets the visual
`scale_factors` to `(1,1,1)` so the STL on disk is self-describing. (The JAM
`Smith2018ContactMesh` loader ignores `scale_factors`, so contact STLs *must* be baked.)

This bake scales the **reference** knee that ships in the base model — the design being
"scale the reference knee with the body." That is genuinely useful on its own: it lets you
take whatever knee the base model has and resize it to a different subject's gait body. The
problem is what happens when a subject-specific knee is then swapped in.

!!! bug "The knee build doesn't inherit the body scale"
    When the knee build runs against a body-scaled base model, it writes the
    **reference-size recon** (`femur_nsm_recon_osim.stl`, …) and repoints the `.osim` at it
    ([`save_geometry_files`][nsosim.model_building.save_geometry_files] copies the new files;
    the `update_*` helpers repoint via `set_mesh_file`; `scale_factors` stay `1,1,1`). The
    recon is **never multiplied by `s_wa`**, and the body-scaled `smith2019-R-*.stl` bake —
    which has *different filenames* — is left orphaned on disk.

    **Net result: a reference-size knee inside a body-scaled body** (the cross-subject case
    from the top of this page). The fix is to apply `s_wa` to the recon (and to
    `mean_patella`, the wrap-fit input, etc.) about the shared joint-center origin — or to
    adopt the cleaner "register-to-native-reference, then resize" design from
    [§3](#3-the-synthetic-decode-transform-chain-latent-opensim). Tracked as
    [Knee sizing Mode 2](deviations.md#mode-2-subject-knee-mri-sized-to-the-gait-body).

### The scaling report
[`scale_comak_model`][nsosim.scaling.orchestrator.scale_comak_model] always writes a JSON
report recording `s_wa`, the per-body scale set, and the per-body mass audit — so the knee
scaling (otherwise baked into the STL with `scale_factors=1`) stays recoverable from disk.

---

## 6. External dependency notes

Two functions in the chain live in the **NSM** dependency, not this repo:

- **`fit_nsm` → NSM optimize** (`NSM/reconstruct/main.py`): fits a latent in NSMcanon.
- **`create_mesh_adaptive`** (`NSM/mesh/main.py`): decodes a latent and undoes **only** the
  NSM training-normalization — not the REFALIGN registration scale. That is *why* a
  reconstruction comes back at reference size rather than the subject's true MRI size.

---

## 7. See also

- [Knee sizing modes](deviations.md) — the catalog of sizing modes, how each is achieved,
  and which are built.
- Repo-root `CLAUDE.md` — "Coordinate Systems & Units", "Per-bone linear_transform",
  "fem_ref_center", "Relative Transforms (T_rel)".
- COMAK-body-scaling spec: `.claude/plans/completed/comak-body-scaling_COMPLETED.md`.
