# Coordinate Systems & the Knee-Build Pipeline

**Authoritative.** Describes **what the code does today** and the capabilities that exist
in the code today. It does **not** describe unbuilt machinery (a Pathway-C true-size flag,
an `s_wa` knee-recon injection, etc.) — those belong to a future fix plan.

Companion page: [Known deviations](deviations.md) records the places where the *pipeline
wiring* produces a reference-scale knee inside a scaled body. This page stays neutral and
descriptive; the deviations page owns the "to get mode Z, do W" framing.

All function references below link to the [API reference](reference/nsm_fitting.md), which
is generated from the docstrings — so they move with the code and can never point at a
stale line.

---

## 1. The coordinate spaces

Every point array in this pipeline lives in one of four spaces. The **scale identity**
column is the part most people miss: a mesh can be shape-personalized while being
size-normalized to the reference.

| Space | Definition | Units | Scale identity |
|---|---|---|---|
| **MRI** | Subject segmentation mesh as it comes off the scanner | mm | **subject-physical** — the subject's true anatomical size |
| **REFALIGN** | Subject mesh after **similarity** registration onto the fixed `smith2019` reference bone (a.k.a. "femur-aligned mm") | mm | **reference size** — subject's true scale divided out by the similarity scale |
| **NSMcanon** | NSM training-normalized box ~[−1, 1]; each bone has its own canonical space | dimensionless | per-bone canonical |
| **OSIM** | OpenSim body-local frame after `convert_nsm_recon_to_OSIM_`: + fixed ref-center, mm→m, axis-swap | m | reference size, rotated into OpenSim axes |

**body-local note:** "OSIM" *is* a body-local OpenSim frame. For the knee sub-bodies
(`femur_distal_r`, `tibia_proximal_r`, `patella_r`, the two menisci) the body-local
origin is the **knee joint center** (the `knee_r` / `pf_r` CustomJoint frames sit at
`(0,0,0)`), not the bone centroid. A recon STL and the reference STL on the same body
therefore share their frame origin by construction.

The orientation swap is the fixed `OSIM_TO_NSM_TRANSFORM` (`nsosim.nsm_fitting`):
`x→−y, y→z, z→−x`.

---

## 2. The MRI / fitting transform chain (subject mesh → OpenSim model)

```
MRI mesh (mm, subject-physical)
  │  align_bone_osim_fit_nsm  (nsm_fitting)
  │    femur: similarity-register onto smith2019  (rigidly_register, reg_mode=rigid_reg_type)
  │    tibia/patella: REUSE the femur transform   (apply_transform_to_mesh(femur_transform))
  ▼
REFALIGN (femur-aligned mm)        ← reference size when reg_mode='similarity'
  │  fit_nsm → NSM optimize + decode (NSM library; create_mesh_adaptive undoes only the
  │           NSM training-normalization, NOT the registration scale)
  ▼
REFALIGN reconstruction  (*_mesh_nsm in dict_bones; saved as *_nsm_recon_mm.vtk)
  │  nsm_recon_to_osim  (nsm_fitting)
  │    _nsm_recon_to_osim_single_surface
  │      convert_nsm_recon_to_OSIM_  (underscore)   +fem_ref_center, /1000, @OSIM_TO_NSM_TRANSFORM.T
  ▼                                                 (NO subject-scale step → stays reference size)
OSIM (meters, reference size)
  │  model_building builders — all operate IN OSIM meters at reference size:
  │    interpolate_bone_ligaments, fit_bone_wrap_surfaces, interpolate_meniscus_ligaments,
  │    create_articular_surfaces, meniscus surfaces, prefemoral fat pad
  │    center_patella_meshes (model_building)   subtracts mean_patella, an OSIM-m translation
  ▼
  │  build_joint_model (model_building)   copytree base model + write recon STLs + finalize
  │    save_geometry_files   copy NEW STLs e.g. femur_nsm_recon_osim.stl
  │    finalize_osim_model
  │      update_osim_model (comak_osim_update)
  │        update_body_geometry_meshfile / update_contact_mesh_files (osim_utils)
  │          (repoint mesh_file via set_mesh_file; scale_factors left as-is = 1,1,1)
  ▼
Subject-specific COMAK .osim  (knee surfaces in OSIM meters, reference size)
```

Functions in that chain (clickable):
[`align_bone_osim_fit_nsm`][nsosim.nsm_fitting.align_bone_osim_fit_nsm] →
[`align_knee_osim_fit_nsm`][nsosim.nsm_fitting.align_knee_osim_fit_nsm] →
[`nsm_recon_to_osim`][nsosim.nsm_fitting.nsm_recon_to_osim] →
[`build_joint_model`][nsosim.model_building.build_joint_model]
([`save_geometry_files`][nsosim.model_building.save_geometry_files],
[`center_patella_meshes`][nsosim.model_building.center_patella_meshes],
[`finalize_osim_model`][nsosim.model_building.finalize_osim_model]) →
[`update_osim_model`][nsosim.comak_osim_update.update_osim_model]
([`update_body_geometry_meshfile`][nsosim.osim_utils.update_body_geometry_meshfile],
[`update_contact_mesh_files`][nsosim.osim_utils.update_contact_mesh_files]).

**Key fact:** the MRI path applies the registration scale (`'similarity'`) but never
restores it. The contact/wrap/ligament geometry of the final model is therefore at
**reference size**, shape-personalized but size-normalized. See
[Known deviations §1](deviations.md#1-mri-knee-size-is-normalized-to-the-reference).

`reg_mode` is `'rigid'` by default on
[`align_knee_osim_fit_nsm`][nsosim.nsm_fitting.align_knee_osim_fit_nsm]; the production
MRI driver (`comak_1_nsm_fitting.py` in the comak repo) passes `'similarity'`.

---

## 3. The synthetic-decode transform chain (latent → OpenSim)

Used for synthetic joints, shape-mode visualization, and latent interpolation — no MRI
target. It is the inverse direction and uses a **different converter**.

```
latent (NSMcanon, dimensionless)
  │  decode_latent_to_osim  (decode)
  │    create_mesh (NSM)  → NSMcanon meshes
  │    convert_nsm_recon_to_OSIM  (NON-underscore, nsm_fitting)
  │      undo_transform   NSMcanon → REFALIGN   (called with scale=1, center=[0,0,0])
  │      convert_nsm_recon_to_OSIM_   REFALIGN → OSIM
  ▼
OSIM (meters, reference size)   →   build_joint_model (shared with the MRI path)
```

[`decode_latent_to_osim`][nsosim.decode.decode_latent_to_osim] is the entry point;
[`decode_joint_from_descriptors`][nsosim.decode.decode_joint_from_descriptors] recovers
per-bone transforms from relative transforms (`T_rel`), then decodes each bone
independently.

**The true-size hook (exists today):** the non-underscore
[`convert_nsm_recon_to_OSIM`][nsosim.nsm_fitting.convert_nsm_recon_to_OSIM] forwards a
`scale`/`center` term into [`undo_transform`][nsosim.nsm_fitting.undo_transform]. A
non-unit `scale` there re-introduces an isotropic size factor. The synthetic path passes
`scale=1`; the MRI path doesn't call this converter at all. So the *capability* to land a
mesh at something other than reference size exists in the code, but no current caller uses
it for true-size restoration. See [Known deviations §1](deviations.md#1-mri-knee-size-is-normalized-to-the-reference).

---

## 4. Converter reference

| Converter | Leg | Space in → out |
|---|---|---|
| [`convert_nsm_recon_to_OSIM_`][nsosim.nsm_fitting.convert_nsm_recon_to_OSIM_] (underscore) | REFALIGN → OSIM; +ref-center, mm→m, axis-swap; **no scale** | mm,refsize → m,refsize |
| [`convert_OSIM_to_nsm_`][nsosim.nsm_fitting.convert_OSIM_to_nsm_] (underscore) | inverse of the above | m,refsize → mm,refsize |
| [`convert_nsm_recon_to_OSIM`][nsosim.nsm_fitting.convert_nsm_recon_to_OSIM] (no underscore) | NSMcanon → OSIM; chains `undo_transform` + underscore | canon → m |
| [`convert_OSIM_to_nsm`][nsosim.nsm_fitting.convert_OSIM_to_nsm] (no underscore) | OSIM → NSMcanon; chains underscore + `apply_transform` | m → canon |
| [`undo_transform`][nsosim.nsm_fitting.undo_transform] | NSMcanon → REFALIGN (per-bone `linear_transform`, scale=1, center=0) | canon → mm |
| [`apply_transform`][nsosim.nsm_fitting.apply_transform] | REFALIGN → NSMcanon (inverse) | mm → canon |

**`fem_ref_center`** = `mean_orig` from `ref_femur_alignment.json` = the reference femur
centroid in NSM-oriented mm before centering (≈ `[-1.22, -10.94, 8.20]`). It is added
back for **all three bones** (the same fixed value) so their REFALIGN spatial
relationship survives into OSIM.

**Per-bone `linear_transform`** (alignment JSONs) maps REFALIGN → that bone's NSMcanon
space; its 3×3 block encodes `scale·R` (uniform scale), and the JSON's separate `scale`
and `center` fields are always `1` and `[0,0,0]` (the similarity is fully embedded in the
4×4). See the repo-root `CLAUDE.md` "Per-bone linear_transform" section for the full
structure.

---

## 5. Stage X body scaling (`s_wa`) and how it meets the knee build

Stage X ([`nsosim.scaling`](reference/scaling.md)) scales a whole-body COMAK base model to
an AddBiomechanics (AB) subject. It is a **separate** step from the knee build; the two
interact at the knee sub-bodies.

### `s_wa` — the weighted-average knee factor
[`build_scale_set`][nsosim.scaling.scale_factors.build_scale_set]:

```
s_wa = (ab_factors['femur_r'][2] + ab_factors['tibia_r'][2]) / 2
```

Mean of the femur and tibia **long-axis** (index 2) AB scale factors — a single
isotropic, dimensionless ratio (1.0 = no change). Every knee sub-body gets this isotropic
factor; AB-provided bodies pass their per-axis factors through unchanged (except
`patella_r`, which AB returns as identity → gets `s_wa`).

### What `ScaleTool.run()` does to the knee (verified empirically)
[`apply_scaletool`][nsosim.scaling.scaletool.apply_scaletool] runs OpenSim's ScaleTool
with `preserveMassDist=True` and **no** `setSubjectMass`. Measured behaviour:

- **Joint frame / weld translations:** scaled by the **parent body's per-axis AB
  factors**, not `s_wa`. The `femur_r → femur_distal_r` weld translation
  (≈ `(−0.0056, −0.3742, −0.0012)` m) scales by `femur_r`'s `(sx, sy, sz)`. So the knee
  sub-body is *positioned* down the shaft by AB's anisotropic parent-bone scaling, while
  its *geometry* is sized isotropically by `s_wa` in the bake step. The knee CustomJoint
  frames (`knee_r`, `pf_r`, the meniscus joints) sit at `(0,0,0)` and stay there.
- **Mass:** left **unchanged** (no `setSubjectMass`).
- **Inertia:** scaled **geometrically by scale²** (an isotropic 0.9 → inertia ×0.81).

The orchestrator's two-pass `_apply_per_body_masses` (in
[`nsosim.scaling.orchestrator`](reference/scaling.md)) then sets the real per-body masses
(AB physics-tuned per-body masses, renormalized to the subject total) and rescales each
body's inertia by its mass ratio.

### The knee STL bake
[`bake_knee_geometry`][nsosim.scaling.knee_geometry.bake_knee_geometry] multiplies each
knee STL's vertices by `s_wa` **about the body-local origin `(0,0,0)`** — which for the
knee sub-bodies is the **knee joint center**, not the bone centroid (measured
centroid→origin: femur 24.8 mm, tibia 54 mm, patella ~1 mm). It then resets the visual
`scale_factors` to `(1,1,1)` so the STL on disk is self-describing. (The JAM
`Smith2018ContactMesh` loader ignores `scale_factors`, which is why the bake is mandatory
for the contact meshes.)

### The interaction with the knee build (Pathway B = multigait)
When the base model that the knee build runs against is a **Stage-X-scaled** model, the
knee build still writes the **reference-scale recon** (`femur_nsm_recon_osim.stl`, etc.)
and repoints the model to it
([`save_geometry_files`][nsosim.model_building.save_geometry_files] copies new files; the
`update_*` helpers repoint via `set_mesh_file`; `scale_factors` stay `1,1,1`). The recon
is not multiplied by `s_wa`. The Stage-X-baked `smith2019-R-*.stl` have **different
filenames** and are left orphaned on disk. Net result: a reference-scale knee inside an
`s_wa`-scaled body. This is described as a wiring deviation in
[Known deviations §2 and §4](deviations.md#2-pathway-b-knee-does-not-inherit-s_wa) — not
here.

### The Stage-X report
[`scale_comak_model`][nsosim.scaling.orchestrator.scale_comak_model] always writes a JSON
report recording `wa_scale` (`s_wa`), the per-body scale set, and the per-body mass audit
— so the (otherwise baked-into-STL, `scale_factors=1`) knee scaling is recoverable from
disk.

---

## 6. External dependency notes

Two functions in the transform chain live in the **NSM** dependency, not this repo, and so
carry no nsosim docstring:

- **`fit_nsm` → NSM optimize** (`NSM/reconstruct/main.py`): fits a latent in NSMcanon.
- **`create_mesh_adaptive`** (`NSM/mesh/main.py`): decodes a latent and undoes **only**
  the NSM training-normalization. It does **not** undo the REFALIGN registration scale —
  which is why the reconstruction returns in REFALIGN (reference size), not MRI size.

---

## 7. See also

- [Known deviations](deviations.md) — the single list of pipeline-wiring deviations and
  how to get the other modes.
- Repo-root `CLAUDE.md` — "Coordinate Systems & Units", "Per-bone linear_transform",
  "fem_ref_center", and "Relative Transforms (T_rel)".
- Stage X spec: `.claude/plans/completed/comak-body-scaling_COMPLETED.md`.
- The comak repo's `SCALING_WORKFLOW_MAP.md` carries a correction banner pointing here;
  this in-library doc is the source of truth where they disagree.
