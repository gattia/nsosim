# Known Issues & Deviations

Where the current knee-build wiring produces the wrong result, and the lever to fix each.
Background and terminology (the **cross-subject** vs **matched-subject** scenarios, the
coordinate spaces, `s_wa`) are on the [Coordinate systems & pipeline](coordinate-systems.md)
page.

Each entry says **what the code does now**, **why it's wrong / what's missing**, and **the
lever** to change it. Items tagged **Bug** are active defects to fix; the rest are
properties or unbuilt modes. None of the "lever" notes are implemented — they are inputs to
a future fix plan, not work items here.

---

## 1. The MRI knee is size-normalized to the reference (by design)

**What happens.** The femur similarity registration
([`align_bone_osim_fit_nsm`][nsosim.nsm_fitting.align_bone_osim_fit_nsm],
`reg_mode='similarity'`) divides the subject's true size out so every subject knee lands in
one shared reference frame (REFALIGN); tibia/patella reuse the femur transform. This part is
intentional and correct — registering everyone to one reference is what makes the geometry
comparable.

**What's missing.** No size is ever *restored* on the MRI path: it converts to OSIM with the
underscore [`convert_nsm_recon_to_OSIM_`][nsosim.nsm_fitting.convert_nsm_recon_to_OSIM_],
which has no scale step. So the knee stays at reference size — correct for neither scenario
(it should be body-scaled for cross-subject, true-size for matched-subject). Measured: raw
MRI femur diagonal CoV across subjects ≈ 7.9%; after registration ≈ 1.0% (size collapsed to
the reference).

**Lever.** The non-underscore
[`convert_nsm_recon_to_OSIM`][nsosim.nsm_fitting.convert_nsm_recon_to_OSIM] forwards a
`scale` into [`undo_transform`][nsosim.nsm_fitting.undo_transform] — the one place a resize
can re-enter. Restoring a size means routing through that hook (or applying `s_wa`
downstream, see §2).

---

## 2. The knee geometry isn't scaled to the gait body — **Bug**

!!! bug "Reference-size knee inside a body-scaled body (cross-subject)"
    COMAK body scaling scales the *body* to the gait subject and bakes the reference knee by
    `s_wa` ([`bake_knee_geometry`][nsosim.scaling.knee_geometry.bake_knee_geometry]). But the
    knee build then writes the **reference-size recon** (`femur_nsm_recon_osim.stl`, …) and
    repoints the model at it
    ([`update_body_geometry_meshfile`][nsosim.osim_utils.update_body_geometry_meshfile] /
    [`update_contact_mesh_files`][nsosim.osim_utils.update_contact_mesh_files], with
    `scale_factors` left at `1,1,1`). The recon is **never multiplied by `s_wa`**, so you end
    up with a reference-size knee inside an `s_wa`-scaled body — a geometric mismatch.

**Evidence.** On `OARSI_multigait_RSubject_121_…/9018389_00m_RIGHT`, `s_wa = 0.97298`; the
baked `smith2019-R-femur-bone.stl` is exactly `0.97298×` but **orphaned** (see §4), while the
body references `femur_nsm_recon_osim.stl` at `scale_factors = 1 1 1` (reference scale).

**Lever.** `s_wa` is on disk in the scaling report. Apply it to the knee recon about the
**common joint-center origin** (see
[Coordinate systems §1, §5](coordinate-systems.md)) at the OSIM-entry point
([`nsm_recon_to_osim`][nsosim.nsm_fitting.nsm_recon_to_osim] /
[`convert_nsm_recon_to_OSIM_`][nsosim.nsm_fitting.convert_nsm_recon_to_OSIM_]), and equally to
the wrap-fit input, the warp-path attachment converter, and `mean_patella` (§3). A cleaner
alternative is the "register-to-native-reference, then resize via the `scale` hook" design
in [Coordinate systems §3](coordinate-systems.md).

---

## 3. `mean_patella` is a reference-size joint offset

**What happens.** [`center_patella_meshes`][nsosim.model_building.center_patella_meshes]
subtracts the patella's mean position (`mean_patella`) so the patella STL is centered at its
body-local origin; that offset is later written as the patellofemoral joint coordinate
translation (via [`update_osim_model`][nsosim.comak_osim_update.update_osim_model]).
`mean_patella` is a translation in **OSIM metres at reference size**.

**Why it matters.** It is the one load-bearing *scalar* in the knee build. Any fix that
scales the patella meshes (§2) must scale this translation too, about the same joint-center
origin, or the patellofemoral kinematics drift relative to the scaled geometry.

**Lever.** Multiply `mean_patella` by `s_wa` wherever the meshes are scaled.

---

## 4. The body-scaled reference-knee bake is orphaned

**What happens.** COMAK body scaling bakes `smith2019-R-{femur,tibia,patella,…}.stl` by
`s_wa` ([`bake_knee_geometry`][nsosim.scaling.knee_geometry.bake_knee_geometry]). The knee
build then repoints the model to the **differently named** recon STLs
([`save_geometry_files`][nsosim.model_building.save_geometry_files] copies new files; the
`update_*` helpers repoint via `set_mesh_file`). The baked `smith2019-R-*.stl` are left on
disk but not referenced by the built model — only the **non-knee** Geometry (torso, pelvis,
limbs) keeps its body scaling.

**Note on a cross-repo map.** The mechanism is **repoint + orphan**, not
"overwrite-by-filename-collision": the recon STLs have different names, so there is no
overwrite. The difference matters for the fix — you scale the recon that gets *repointed to*,
and the reference-knee bake is dead weight in the cross-subject case.

**Lever.** Decide the bake's fate once §2 is fixed (drop it, unless it's wanted for the
"scale whatever knee the base model has" use case, which is its one independent value).

---

## 5. Matched-subject (true-size) mode — not implemented

When the gait subject *is* the MRI subject, the knee should keep its true MRI size rather
than be normalized to the reference. There is no code path for this today. The lever is the
same `scale` hook as §1 (route the recon through the non-underscore converter with the
subject's true scale instead of the underscore converter).

---

## Where the fix would touch (pre-identified; do NOT implement from this page)

A verified starting list for the future fix plan — **not** work items here:

- Apply `s_wa` to the knee recon about the common joint-center origin at the OSIM-entry point
  ([`nsm_recon_to_osim`][nsosim.nsm_fitting.nsm_recon_to_osim] /
  [`convert_nsm_recon_to_OSIM_`][nsosim.nsm_fitting.convert_nsm_recon_to_OSIM_]), **and** the
  wrap-fit input, the warp-path attachment converter, and `mean_patella` (§3).
- Thread the synthetic path ([`nsosim.decode`](reference/decode.md)) too if the feature is
  meant to be general.
- Keep mass/inertia owned by COMAK body scaling (the orchestrator two-pass) — a geometry fix
  touches geometry only.
- For matched-subject (§5), route the MRI recon through the non-underscore converter's
  `scale` hook instead of the underscore converter.

See `.claude/plans/scaling-and-spaces-documentation.md` (Stage 5) for the fix-plan scope.
