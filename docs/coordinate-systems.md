# Coordinate Systems & the Knee-Build Pipeline

This page is the canonical reference for the coordinate spaces a knee build passes through,
the transform chain that connects them, and how body scaling interacts with the knee
geometry. Its companion, [Knee sizing modes](deviations.md), catalogs the ways you might
size a knee placed into a model, how each is achieved, and which are built today.

Function names link to the [API reference](reference/nsm_fitting.md), generated from the
docstrings — so they move with the code and never point at a stale line.

---

## What the pipeline is trying to do

The library takes a knee (bone + cartilage geometry, from an MRI fit or a decoded latent)
and drops it into a whole-body OpenSim/COMAK model so it can be simulated under a gait.

**Most builds use the unscaled model and just work.** A subject's knee comes out *reference
size* after registration (below), and the default base model is also reference size — so a
personalized knee dropped into it matches, with nothing to rescale. This is the standard
personalized build ([Mode 1](deviations.md#mode-1-personalized-knee-unscaled-reference-size-model)),
and in practice the most common path (a workflow observation, not something the code enforces).

The **size** question only arises when the body is **scaled to a gait subject** (`s_wa`,
[§5](#5-comak-body-scaling-and-how-it-meets-the-knee-build)). Then what the knee's size
should be depends on whose gait it is:

- **A different person's gait** (e.g. an OAI knee under someone else's gait) — scale the knee
  to the gait body's size. → currently an active **bug**: the body is scaled but the
  swapped-in knee is not
  ([Mode 3](deviations.md#mode-3-personalized-knee-scaled-to-the-gait-body)).
- **The MRI subject's own gait** — keep the knee at its true MRI size instead. → **not
  implemented** ([Mode 4](deviations.md#mode-4-personalized-knee-true-anatomical-size)).

The [knee sizing modes](deviations.md) page is the full catalog (it also covers the
generic-knee and synthetic paths); the rest of *this* page is the mechanics behind them.

---

## 1. The coordinate spaces

Every point array in this pipeline lives in one of four spaces. The **scale identity**
column is the one people miss: a mesh can be shape-personalized while being
size-normalized to the reference.

| Space | Definition | Units | Scale identity |
|---|---|---|---|
| **MRI** | Subject segmentation mesh in its native DICOM (LPS) frame | mm | **subject-physical** — the subject's true anatomical size |
| **REFALIGN** | The *frame* the subject mesh lands in after registration onto the fixed `smith2019` reference bone (a.k.a. "femur-aligned mm"); the common frame the NSM fit runs in | mm | **mode-dependent:** **reference size** with `reg_mode='similarity'` (the default — true scale divided out), **subject true size** with `'rigid'` |
| **NSMcanon** | NSM training-normalized box ~[−1, 1]; each bone has its own canonical space | dimensionless | per-bone canonical |
| **OSIM** | OpenSim body-local frame, produced by `convert_nsm_recon_to_OSIM_` (see [§4](#4-converter-reference)) | m | reference size, rotated into OpenSim axes |

The **MRI → OSIM** orientation change is a single fixed rotation/axis-swap,
**`OSIM_TO_NSM_TRANSFORM`** (defined in `nsosim.nsm_fitting`): `x→−y, y→z, z→−x`. It is what
the "rotated into OpenSim axes" in the OSIM row above refers to — applied in every
REFALIGN→OSIM conversion.

!!! info "What 'MRI space' actually is"
    MRI space is the segmentation mesh's **native anatomical frame**. The meshes come from a
    DICOM → SimpleITK pipeline, and both DICOM and ITK/SimpleITK use **LPS** as their physical
    coordinate system (axes toward the patient's **L**eft, **P**osterior, **S**uperior) — so
    the meshes are LPS as long as the extraction keeps SimpleITK's physical space (origin +
    direction + spacing). NIfTI/Slicer pipelines use RAS instead; if you need to be certain,
    the authority is the mesh-extraction code, not nsosim. Either way nsosim reads the meshes
    as-is and imposes no convention — the femur similarity registration in the first step
    aligns the input onto the fixed reference, so everything downstream is agnostic to it.

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

The same chain as a call tree (each **`— calls …`** entry lists the sub-functions that step
invokes):

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
    production caller). `reg_mode='rigid'` registers with rotation + translation only and so
    keeps the subject's true size — but with a caveat: if the subject's knee differs much in
    size from the reference, a rigid (no-scale) fit aligns *worse*. The more robust route to a
    true-size knee is therefore to register with `'similarity'` (which gets good shape
    correspondence) and **restore the removed scale afterwards** — see
    [Mode 4](deviations.md#mode-4-personalized-knee-true-anatomical-size).

---

## 3. The synthetic-decode transform chain (latent → OpenSim)

Used to build a knee from an arbitrary latent — synthetic joints, shape-mode visualization,
latent interpolation — with **no MRI target**. It is the inverse direction of fitting and
starts in NSMcanon, so it goes through the **full converter**
[`convert_nsm_recon_to_OSIM`][nsosim.nsm_fitting.convert_nsm_recon_to_OSIM] (canonical → OSIM,
which exposes a `scale`), rather than the REFALIGN-only converter the MRI path uses.

The diagram below is the **single-bone** path:

```
latent ─────────────────────────────────────────────── NSMcanon · dimensionless
   │
   │ STEP  decode_latent_to_osim   (one bone; for a full joint, see below)
   │   create_mesh (NSM)                         → NSMcanon meshes
   │   convert_nsm_recon_to_OSIM  (the full converter):
   │       undo_transform           NSMcanon → REFALIGN   (called with scale=1, center=0)
   │       convert_nsm_recon_to_OSIM_   REFALIGN → OSIM
   ▼
OSIM meshes ────────────────────────────────────────── m · reference size
   │
   │ STEP  build_joint_model  (the SAME assembler the MRI path uses)
   ▼
Subject-specific COMAK .osim
```

[`decode_latent_to_osim`][nsosim.decode.decode_latent_to_osim] is the single-bone primitive
shown above: one latent → NSMcanon meshes → OSIM. To build a **whole joint** you use
[`decode_joint_from_descriptors`][nsosim.decode.decode_joint_from_descriptors], which wraps
it: given three latents (femur, tibia, patella) plus *relative* transforms (`T_rel`) that say
how the tibia and patella sit relative to the femur, it recovers each bone's alignment
transform ([`recover_bone_transform`][nsosim.transforms.recover_bone_transform]) and then runs
the single-bone path once per bone — so the three land in one consistent joint configuration
instead of being decoded in isolation.

!!! warning "The converter `scale` argument is NOT a clean knee-resize"
    The full converter
    [`convert_nsm_recon_to_OSIM`][nsosim.nsm_fitting.convert_nsm_recon_to_OSIM] forwards a
    `scale` into [`undo_transform`][nsosim.nsm_fitting.undo_transform], where it multiplies the
    points **in NSMcanon space, before** the per-bone inverse transform and the `+ ref-center`
    shift. So it is the NSM canonical-normalization scale — **not** the femur
    similarity-registration scale, and **not** a clean resize of the final OSIM geometry.

    Verified numerically: setting `scale=2` does **not** double the OSIM points. It scales them
    about an affine offset center (~4 cm from the joint-center origin for a test tibia) that
    **differs per bone** — so using it to "resize the knee" would scale each of the three bones
    about a different point and distort the joint.

    The clean way to resize a built knee is a plain **OSIM-space multiply about the shared
    joint-center origin** — `osim_points *= s_wa`, exactly what
    [`bake_knee_geometry`][nsosim.scaling.knee_geometry.bake_knee_geometry] already does for the
    reference knee. The synthetic path passes `scale=1` and the MRI path doesn't call this
    converter at all, so nothing resizes a knee today. See the
    [knee sizing modes](deviations.md) for the fix.

---

## 4. Converter reference

There are two converters per direction, distinguished in the code by a trailing underscore.
The **REFALIGN-only** version (`convert_nsm_recon_to_OSIM_`, with the underscore) does just
the REFALIGN↔OSIM leg — centre, units, orientation; no scale. The **full** version
(`convert_nsm_recon_to_OSIM`, no underscore) is the whole canonical↔OSIM round-trip: it
*calls* the REFALIGN-only version and adds the NSMcanon↔REFALIGN leg (`undo_transform` /
`apply_transform`), and it is the only one that exposes a `scale`.

| Converter | What it does | Space in → out |
|---|---|---|
| [`convert_nsm_recon_to_OSIM_`][nsosim.nsm_fitting.convert_nsm_recon_to_OSIM_] (REFALIGN-only) | REFALIGN → OSIM only: `+ ref-center`, mm→m, axis-swap. **No scale.** | mm, refsize → m, refsize |
| [`convert_OSIM_to_nsm_`][nsosim.nsm_fitting.convert_OSIM_to_nsm_] (REFALIGN-only) | exact inverse of the above | m, refsize → mm, refsize |
| [`convert_nsm_recon_to_OSIM`][nsosim.nsm_fitting.convert_nsm_recon_to_OSIM] (full) | `undo_transform` (canonical→REFALIGN, **applies `scale`**) then the REFALIGN-only leg | NSMcanon → m |
| [`convert_OSIM_to_nsm`][nsosim.nsm_fitting.convert_OSIM_to_nsm] (full) | REFALIGN-only leg then `apply_transform` (REFALIGN→canonical) | m → NSMcanon |
| [`undo_transform`][nsosim.nsm_fitting.undo_transform] | NSMcanon → REFALIGN, via the per-bone `linear_transform` (the only place a `scale` enters) | canon → mm |
| [`apply_transform`][nsosim.nsm_fitting.apply_transform] | REFALIGN → NSMcanon (inverse of `undo_transform`) | mm → canon |

In short: the **MRI path** already has its mesh in REFALIGN, so it only uses the
REFALIGN-only converter — which is *why* it has no `scale`. The **synthetic/decode path**
starts in NSMcanon, so it uses the full converter, where a `scale` *can* be applied.

!!! info "Resizing the MRI path"
    The MRI path has no resize today. The clean way to add one is to multiply the OSIM-space
    recon by the scale **about the joint-center origin** at the OSIM-entry point — a plain
    `osim_points *= s_wa`, the same operation
    [`bake_knee_geometry`][nsosim.scaling.knee_geometry.bake_knee_geometry] uses. This is *not*
    the same as the converter's `scale` argument (which scales in canonical space about a
    per-bone affine center — see the warning above). Scaling at the OSIM-entry point is what
    [Mode 3](deviations.md#mode-3-personalized-knee-scaled-to-the-gait-body) needs.

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

    **Does scaling have to account for `mean_orig`?** No. By the time a knee is in OSIM,
    `mean_orig` is already baked into the vertex coordinates, and `s_wa` scaling happens about
    the body-local **joint-center origin** — the same origin the reference bake uses
    ([§1](#1-the-coordinate-spaces)). Scaling those coordinates about that shared origin
    carries the position correctly, so no separate centre term is needed. It hasn't been
    ignored — it's already in the numbers being scaled.

**Per-bone `linear_transform`** (in each bone's `*_alignment.json`) is the transform from the
**registration-output frame** — REFALIGN, i.e. your mesh after it was registered onto the
reference (similarity, in production) — into that bone's **NSM canonical/latent space**. Its
3×3 block is `scale · R`, where **`R` is a proper rotation matrix** (det +1) and `scale` is a
single uniform factor (the same along every column). The JSON's separate `scale` and `center`
fields are always `1` and `[0,0,0]` because the whole similarity (rotation + uniform scale +
centring) is baked into the 4×4. See the repo-root `CLAUDE.md` "Per-bone linear_transform"
section for the full structure.

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
(renormalized so the total matches the subject). (ScaleTool itself is run *without*
`setSubjectMass`, so the mass change is entirely this later pass — geometry scaling and mass
transfer are deliberately separated.)

The inertia is **not** recomputed from geometry — it is the body's existing inertia (already
geometrically scaled by ScaleTool) multiplied by the **mass ratio** `new_mass / old_mass`.
That's the consistent update: for a fixed shape, inertia is proportional to mass, so only the
mass is being corrected here, not the mass distribution. (A from-scratch inertia from the
meshes would be a different, larger change and isn't what this step does.)

### The knee geometry "bake"
[`bake_knee_geometry`][nsosim.scaling.knee_geometry.bake_knee_geometry] multiplies each knee
STL's vertices by `s_wa`, about the body-local origin `(0,0,0)` (= the knee joint center,
see [§1](#1-the-coordinate-spaces); not the bone centroid), and resets the visual
`scale_factors` to `(1,1,1)` so the STL on disk is self-describing. (The JAM
`Smith2018ContactMesh` loader ignores `scale_factors`, so contact STLs *must* be baked.)

This bake scales the **reference** knee that ships in the base model — the design being
"scale the reference knee with the body."

!!! success "This works on its own (Mode 2)"
    Run COMAK body scaling by itself and you get a valid, runnable model with the reference
    knee resized to the gait body: the STLs are baked by `s_wa`, ScaleTool scales the joint
    frames/welds to match, and the masses come from AB. The Stage-X tests check exactly this
    (cartilage–bone proximity, ligament reference strains, and wrap placement all hold up
    after a non-trivial scale). So if you just want the **generic** knee at a new subject's
    size, this is complete and correct. The problem is *only* the next step — swapping a
    subject-specific knee in.

!!! bug "The knee build doesn't inherit the body scale"
    When the knee build runs against a body-scaled base model, it writes the
    **reference-size recon** (`femur_nsm_recon_osim.stl`, …) and repoints the `.osim` at it
    ([`save_geometry_files`][nsosim.model_building.save_geometry_files] copies the new files;
    the `update_*` helpers repoint via `set_mesh_file`; `scale_factors` stay `1,1,1`). The
    recon is **never multiplied by `s_wa`**, and the body-scaled `smith2019-R-*.stl` bake —
    which has *different filenames* — is left orphaned on disk.

    **Net result: a reference-size (unscaled) knee inside a body-scaled body** (the
    cross-subject case from the top of this page). The fix is to apply `s_wa` to the recon
    (and to `mean_patella`, the wrap-fit input, etc.) about the shared joint-center origin —
    or to adopt the cleaner "register-to-native-reference, then resize" design from
    [§3](#3-the-synthetic-decode-transform-chain-latent-opensim). Tracked as
    [Knee sizing Mode 3](deviations.md#mode-3-personalized-knee-scaled-to-the-gait-body).

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
