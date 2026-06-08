# Knee Sizing — Modes & Status

When you drop a knee into a COMAK body model, its **size** has to match the body it's going
into. This page lists the modes you might want, how each is (or would be) achieved, and
what's built today. The mechanics they rest on — the coordinate spaces, the similarity
registration, `s_wa`, and how knee geometry is scaled — are on
[Coordinate systems & pipeline](coordinate-systems.md).

The body the knee goes into is either left at **reference size** (the default base model) or
**scaled to a gait subject** (`s_wa`, via [COMAK body scaling](coordinate-systems.md#5-comak-body-scaling-and-how-it-meets-the-knee-build)).
Crossed with where the knee geometry comes from (the generic reference knee, a subject MRI
fit, or a **synthetic joint decoded from latents**) — and, for a same-subject build, whether
you want true anatomical size — that gives the modes below.

| Mode | Knee geometry | Body | Knee size | Status |
|---|---|---|---|---|
| [1](#mode-1-personalized-knee-unscaled-reference-size-model) — personalized, unscaled model | subject MRI | reference (default) | reference (auto-matches) | **works — most common** |
| [2](#mode-2-generic-knee-scaled-to-the-gait-body) — generic, gait-scaled | reference | gait subject (`s_wa`) | `s_wa` (baked) | **works** |
| [3](#mode-3-personalized-knee-scaled-to-the-gait-body) — personalized, gait-scaled | subject MRI | gait subject (`s_wa`) | `s_wa` (build, then scale) | **works** (library verified; production wiring pending) |
| [4](#mode-4-personalized-knee-true-anatomical-size) — personalized, true size | subject MRI | scaled to the MRI subject (their own mocap) | true MRI size | **not implemented** |
| [5](#mode-5-synthetic-knee-scaled-to-a-model) — synthetic, scaled | synthetic (decoded latents) | any | `s_wa` (build, then scale) | **works** (same path as Mode 3) |

"True size" means the knee's real anatomical size as segmented from the MRI, before the
similarity registration normalizes it away.

---

## Mode 1 — personalized knee, unscaled (reference-size) model

**What you want.** The standard personalized build: fit a subject's MRI knee and put it into
the COMAK model **without** gait/body scaling — the base model stays at its reference size.

**How.** Register the MRI knee onto the reference (which normalizes its *size* to reference
while keeping its *shape*), reconstruct, convert to OSIM, and swap it into the reference-size
base model. The knee and the body are **both at reference size**, so they are consistent —
nothing needs to be rescaled.

**Example.** Assemble the personalized model from OSIM-space recon meshes (the output of the
MRI fit — `align_knee_osim_fit_nsm` then `nsm_recon_to_osim`):

```python
from nsosim.model_building import build_joint_model

built_osim = build_joint_model(
    bone_meshes, dict_bones, ref_data_paths, dict_lig_musc_attach_params,
    fem_ref_center, save_dir, model_name,
    path_base_osim_model=reference_comak_base,   # reference-size base → reference-size knee
)
# built_osim: the finished personalized COMAK model, no body scaling.
```

**Status: works — and in practice the most common path** (a workflow observation, not
something the code enforces). A reference-size knee in a reference-size body has no mismatch.
(It is *only* when the body gets gait-scaled, [Mode 3](#mode-3-personalized-knee-scaled-to-the-gait-body),
that an un-scaled knee becomes a problem — same knee, but now the body was **scaled** and the
knee didn't follow.)

---

## Mode 2 — generic knee, scaled to the gait body

**What you want.** No subject MRI — use the reference knee the base model already ships with,
resized to fit a particular gait subject's body.

**How.** COMAK body scaling bakes the reference knee STLs by `s_wa`
([`bake_knee_geometry`][nsosim.scaling.knee_geometry.bake_knee_geometry]): vertices × `s_wa`
about the joint-center origin, then `scale_factors` reset to `1`. Bone, cartilage, contact,
and menisci all scale together.

**Example.** Scale the generic reference model (its reference knee included) straight to a gait
subject — no imaging, no build:

```python
from nsosim.scaling import scale_comak_model

scale_comak_model(
    base_osim=reference_comak_base,              # generic model, reference knee
    ab_scaled_osim=gait_subject_ab_osim,         # the gait subject's AddBiomechanics .osim
    output_osim=out_osim,
    output_geometry_dir=out_osim.parent / "Geometry",
)
```

**Status: works.** This is the one thing the knee bake does correctly today, and it's worth
keeping — it's how you take the generic model to a new gait subject with no imaging at all.

---

## Mode 3 — personalized knee, scaled to the gait body

**What you want.** [Mode 1](#mode-1-personalized-knee-unscaled-reference-size-model), but the
body has been **gait-scaled** to a (different) subject — e.g. an OAI knee simulated under
someone else's gait. The personalized *shape* should now be scaled to the gait body's *size*.

**How — build, then scale (verified).** Run the standard personalized build once against a
**reference** base (Mode 1), then run COMAK body scaling
([`scale_comak_model`][nsosim.scaling.orchestrator.scale_comak_model]) on that *built* model
(`base_osim` = the built model) instead of on the generic reference. By the time the model is
built, **both** knee-geometry generators (below) have already run and their outputs are attached
to the knee bodies, so the existing body-scaling machinery scales the whole knee with **no
mid-build interception and no GPU re-fit** — so the expensive NSM fit is amortized across gait
subjects (fit once, scale many):

- [`bake_knee_geometry`][nsosim.scaling.knee_geometry.bake_knee_geometry] bakes *whatever STL
  the knee bodies point at* (it reads `mesh_file` off the model), so it bakes the **recon**
  STLs by `s_wa` about the joint-center origin — not the reference STLs;
- OpenSim ScaleTool scales the **wraps, ligaments (slack lengths / reference strains), joint
  frames, muscles, and the patella placement offset** (`pf_tx/ty/tz_r`) by `s_wa`.

This is exactly the operation Mode 2 performs; the only difference is that the attached geometry
is a recon rather than the reference.

**How to run it.** There are three shapes, depending on whether the personalized knee already
exists. All three end in the same operation —
[`scale_comak_model`][nsosim.scaling.orchestrator.scale_comak_model] on a *built* model — and
all three are CPU-only at the scale step (no GPU, no NSM re-fit).

*Case A — you already have built personalized models (the common case).* If you have run the
build before (a generic body with a custom knee, reference size), there is nothing to rebuild —
just scale each existing model to each gait subject:

```python
from nsosim.scaling import scale_comak_model

for built_osim in existing_built_models:          # generic body + custom knee, reference size
    for gait in gait_subjects:
        scale_comak_model(
            base_osim=built_osim,                 # the existing Mode-1 model (Geometry/ beside it)
            ab_scaled_osim=gait.ab_osim,          # that gait subject's AddBiomechanics .osim
            output_osim=gait.out_osim,
            output_geometry_dir=gait.out_osim.parent / "Geometry",
        )
```

*Case B — from scratch, reused across many gaits (amortize the GPU fit).* Build **once** on a
reference base, then scale per gait subject:

```python
from nsosim.model_building import build_joint_model
from nsosim.scaling import scale_comak_model

built_osim = build_joint_model(                   # ONCE — the only GPU-dependent step is upstream
    bone_meshes, dict_bones, ref_data_paths, dict_lig_musc_attach_params,
    fem_ref_center, save_dir, model_name,
    path_base_osim_model=reference_comak_base,     # reference-size base, NOT a scaled one
)
for gait in gait_subjects:                         # scale many — CPU only
    scale_comak_model(
        base_osim=built_osim, ab_scaled_osim=gait.ab_osim,
        output_osim=gait.out_osim, output_geometry_dir=gait.out_osim.parent / "Geometry",
    )
```

*Case C — one fresh subject, one gait.* Same two calls, run once each:

```python
built_osim = build_joint_model(..., path_base_osim_model=reference_comak_base)
scale_comak_model(
    base_osim=built_osim, ab_scaled_osim=gait_ab_osim,
    output_osim=out_osim, output_geometry_dir=out_osim.parent / "Geometry",
)
```

Use the two functions directly rather than a bundled `build_and_scale` helper: the build/scale
boundary is what lets you build once and scale many (Cases A and B). [Mode 5](#mode-5-synthetic-knee-scaled-to-a-model)
is identical with `bone_meshes` decoded from latents instead of fit from an MRI.

!!! success "Verified end-to-end on a real built model"
    A real Mode-1 build (an OAI knee) scaled to a *different* subject's gait body
    (`s_wa ≈ 0.973`) scales **every** knee component by `s_wa`:
    recon bone / cartilage / menisci / fat-pad STLs (per-vertex), wrap translations and radii,
    all ligament/muscle attachment points, and the patella placement offset — and the scaled
    model still initializes. Cartilage-bone proximity and ligament reference strains are
    preserved (slacks scale with path length). Covered by
    `tests/scaling/test_build_then_scale.py` (the recon end-to-end) and
    `tests/scaling/test_nontrivial.py` (the reference-knee path it reuses). The patella
    interaction that looked fragile is fine: the centered patella shape scales by `s_wa` about
    the origin *and* its offset scales by `s_wa`, so placement is preserved.

!!! note "Production wiring (comak repo) — remaining step"
    The library route above is verified. The current Pathway-B production scripts (in the
    external `comak_gait_simulation` repo) still build the knee **on a body-scaled base** (the
    old order), which leaves a reference-size recon inside an `s_wa`-scaled body — the original
    mismatch. Adopting "build, then scale" there (build on a reference base, then call
    `scale_comak_model` on the built model) is the remaining integration step; it changes only
    that repo's wiring, not the nsosim library.

**Alternative — intercept the two generators mid-build.** If you scale *during* a build
(rather than scaling a finished model), there is no single place to do it: the knee build has
**two independent geometry generators**, both emitting reference-size OSIM output, and you must
scale **both**:

1. **The recon meshes** ([`nsm_recon_to_osim`][nsosim.nsm_fitting.nsm_recon_to_osim]). Scaling
   these covers everything *extracted* from them: articular surfaces, meniscus surfaces, the
   prefemoral fat pad, and `mean_patella` (the mean of the scaled patella mesh).
2. **The reference→subject warp**
   ([`interpolate_bone_ligaments`][nsosim.model_building.interpolate_bone_ligaments] and the
   meniscus-ligament warp). It does **not** read the recon mesh — it warps the *reference*
   labeled mesh + attachment points onto the subject through the NSM. Its output feeds the
   **wrap-surface fit** ([`fit_bone_wrap_surfaces`][nsosim.model_building.fit_bone_wrap_surfaces]
   takes the warped labeled mesh) and **is** the ligament/muscle attachment positions. Scaling
   (1) alone leaves all of these at reference size.

Either way, scale with a plain OSIM-space multiply (`osim_points *= s_wa`) about the
joint-center origin — the same operation
[`bake_knee_geometry`][nsosim.scaling.knee_geometry.bake_knee_geometry] uses — **not** the
converter's `scale` argument. (Mid-build, this is the "latent-interpolation" subtlety:
ligaments and wraps are a *separate* product, not a readout of the bone mesh.)

!!! warning "Not the converter's `scale` argument"
    The natural-looking lever — passing a `scale` to
    [`convert_nsm_recon_to_OSIM`][nsosim.nsm_fitting.convert_nsm_recon_to_OSIM] (which the warp
    path already uses for its OSIM conversion) — is **not** the right one: it scales in
    canonical space about a per-bone affine center, not the shared joint-center origin, so it
    distorts the joint (verified; [Coordinate systems §3](coordinate-systems.md) has the
    numbers). Scale each generator's *OSIM output* with a plain multiply instead.

A heavier alternative is to register each subject onto a **pre-scaled** reference so both
generators inherit the size directly; it changes the fitting step and gives the same result, so
scaling the OSIM outputs is simpler.

---

## Mode 4 — personalized knee, true anatomical size

**What you want.** The gait subject *is* the MRI subject — keep their real knee at its real
size, rather than normalizing it to the reference.

!!! warning "🚧 Not implemented — design sketch"
    No code path builds this today; the routes below are a proposed design, not supported
    usage.

**How it would work.** The **preferred** route is to register with `'similarity'` and then
restore the true size — similarity registration gets good shape correspondence regardless of
how much the subject differs in size from the reference, which a rigid (no-scale) fit does
not. Record the scale the registration removed, then re-apply it as a clean **OSIM-space
multiply about the joint-center origin** (the same lever as [Mode 3](#mode-3-personalized-knee-scaled-to-the-gait-body),
just with the subject's own scale instead of `s_wa` — **not** the converter's `scale`
argument).

A simpler-but-weaker alternative is `reg_mode='rigid'` (rotation + translation only), which
keeps true size from the start but, lacking the scale, tends to align worse when the subject's
knee differs much in size from the reference.

---

## Mode 5 — synthetic knee, scaled to a model

**What you want.** Decode an arbitrary latent (a synthetic joint, a shape-mode sweep, an
interpolation) into a model, at an appropriate size.

**How — the same build, then scale as Mode 3.** The decode path
([`nsosim.decode`](reference/decode.md)) produces OSIM-space meshes that feed the *same*
assembler the MRI path uses ([`build_joint_model`][nsosim.model_building.build_joint_model]), so
a decoded knee built into a reference-size COMAK model is structurally identical to a Mode-1 MRI
build — just with latent-derived geometry. Scaling it to a gait body is therefore the **same
operation**: run [`scale_comak_model`][nsosim.scaling.orchestrator.scale_comak_model] on the
built model. There is no separate synthetic resize path and no new lever — Mode 3 and Mode 5 are
one function with a different `base_osim` (the only difference is upstream: latent decode vs MRI
fit).

**Example.** Decode latents → OSIM meshes, build, then (optionally) scale — Mode 1/3 with a
synthetic knee:

```python
from nsosim.decode import decode_joint_from_descriptors
from nsosim.model_building import build_joint_model
from nsosim.scaling import scale_comak_model

joint = decode_joint_from_descriptors(           # latents → OSIM-space meshes
    femur_latent, tibia_latent, patella_latent,
    T_fem, T_rel_tib, T_rel_pat, models, model_configs, fem_ref_center,
)
bone_meshes = {b: joint[b] for b in ("femur", "tibia", "patella")}
built_osim = build_joint_model(bone_meshes, ..., path_base_osim_model=reference_comak_base)

# Reference size: stop here. To size to a gait body, scale exactly like Mode 3:
scale_comak_model(
    base_osim=built_osim, ab_scaled_osim=gait_ab_osim,
    output_osim=out_osim, output_geometry_dir=out_osim.parent / "Geometry",
)
```

**Status: works.** Once the synthetic knee is built into a model (decode →
`build_joint_model`), it scales exactly like a Mode-1 build. The build-then-scale verification in
`tests/scaling/test_build_then_scale.py` exercises the operation on a built model; nothing about
it is specific to whether the recon came from an MRI fit or a latent.

---

## Notes

**Build first, then scale — the two-generator problem only exists mid-build.** Once you have a
*built* model, both generators below have already run and their outputs are attached to the
knee bodies, so running COMAK body scaling on the built model scales everything at once (the
recon bake + ScaleTool's wrap / ligament / frame scaling). The "two generators" caveat applies
**only** if you try to scale *during* a build. See
[Mode 3](#mode-3-personalized-knee-scaled-to-the-gait-body).

**If you scale mid-build, scale both generators early, and the derived quantities follow.**
There is no single place to scale — the recon meshes and the reference→subject warp are
independent (see [Mode 3](#mode-3-personalized-knee-scaled-to-the-gait-body)). Scale each
generator's OSIM output early and everything *downstream of each* is correct without
per-quantity fix-up:

- from the **recon meshes**: articular surfaces, meniscus surfaces, fat pad, and `mean_patella`
  (just the mean of the already-scaled patella vertices —
  [`center_patella_meshes`][nsosim.model_building.center_patella_meshes], written as the
  patellofemoral joint translation by
  [`update_osim_model`][nsosim.comak_osim_update.update_osim_model]; so no separate scalar
  fix-up if the mesh was scaled first);
- from the **warp**: the wrap surfaces and the ligament/muscle attachments.

The trap is scaling only the recon meshes (or scaling anything *late*) and leaving the warp
products — wraps, ligaments — at reference size.

**The reference-knee bake (Mode 2) is not a leak.** When a subject or synthetic recon is
swapped in (Modes 3–5), the `s_wa`-baked `smith2019-R-*.stl` from Mode 2 go unused — the recon
STLs have different filenames and the model is repointed to them via
[`save_geometry_files`][nsosim.model_building.save_geometry_files] + the `update_*` helpers.
That's expected; keep the bake for Mode 2, the other modes simply don't use it.

---

## Implementation status

- **Modes 3 & 5 — build, then scale: DONE in the library (verified).** Build the personalized
  (or synthetic) knee against a reference base, then run COMAK body scaling
  ([`scale_comak_model`][nsosim.scaling.orchestrator.scale_comak_model]) on the *built* model.
  This reuses the Mode-2 machinery wholesale — `bake_knee_geometry` bakes the attached recon
  STLs, ScaleTool scales the wraps / ligaments (slacks) / frames / muscles / patella offset — so
  the GPU fit is amortized across gait subjects (fit once, scale many). Verified end-to-end on a
  real built model: patella centering × scaling and the PF offset, the menisci and fat pad, and
  origin-scaling placement for every knee body all hold (`tests/scaling/test_build_then_scale.py`;
  slack/strain scaling also covered by `tests/scaling/test_nontrivial.py`). Mass/inertia stays
  owned by COMAK body scaling's orchestrator two-pass — this is a geometry-only path.
  **Remaining:** wire "build, then scale" into the Pathway-B production scripts (in the
  `comak_gait_simulation` repo) in place of the current build-on-a-scaled-base order — see the
  production-wiring note under [Mode 3](#mode-3-personalized-knee-scaled-to-the-gait-body).
- **Mode 4 — true size: not implemented.** Prefer `'similarity'` registration + restore the
  recorded scale via the same OSIM-space multiply about the joint-center origin;
  `reg_mode='rigid'` is the simpler but weaker fallback.
- **Alternative — mid-build interception** (only if you ever need to scale *during* a build
  rather than scaling a finished model). Scale **both** geometry generators by `s_wa` about the
  joint-center origin with a plain OSIM-space multiply (**not** the converter's `scale`
  argument): (a) the recon meshes from
  [`nsm_recon_to_osim`][nsosim.nsm_fitting.nsm_recon_to_osim] (covers articular surfaces,
  menisci, fat pad, `mean_patella`), **and** (b) the reference→subject warp output from
  [`interpolate_bone_ligaments`][nsosim.model_building.interpolate_bone_ligaments] / the
  meniscus-ligament warp (covers the wrap-fit input and the ligament/muscle attachments).
  Build-then-scale (above) is strictly simpler and is the recommended route.

See `.claude/plans/scaling-and-spaces-documentation.md` (Stage 5) and
`.claude/plans/knee-scaling-fix.md` for the fix-plan scope.
