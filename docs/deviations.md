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
| [3](#mode-3-personalized-knee-scaled-to-the-gait-body) — personalized, gait-scaled | subject MRI | gait subject (`s_wa`) | should be `s_wa`, **stays reference** | **bug** |
| [4](#mode-4-personalized-knee-true-anatomical-size) — personalized, true size | subject MRI | scaled to the MRI subject (their own mocap) | true MRI size | **not implemented** |
| [5](#mode-5-synthetic-knee-scaled-to-a-model) — synthetic, scaled | synthetic (decoded latents) | any | any | **partial** — path exists at scale = 1 |

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

**Status: works.** This is the one thing the knee bake does correctly today, and it's worth
keeping — it's how you take the generic model to a new gait subject with no imaging at all.

---

## Mode 3 — personalized knee, scaled to the gait body

**What you want.** [Mode 1](#mode-1-personalized-knee-unscaled-reference-size-model), but the
body has been **gait-scaled** to a (different) subject — e.g. an OAI knee simulated under
someone else's gait. The personalized *shape* should now be scaled to the gait body's *size*.

**How it's meant to work.** Register the MRI knee onto the reference (→ reference size), then
scale that recon to the gait body by `s_wa`, so the shape ends up at the body's size.

!!! bug "The recon never gets scaled to the body"
    The second half doesn't happen. The knee build writes the reference-size recon and points
    the model at it with `scale_factors = 1,1,1`
    ([`update_body_geometry_meshfile`][nsosim.osim_utils.update_body_geometry_meshfile] /
    [`update_contact_mesh_files`][nsosim.osim_utils.update_contact_mesh_files]); `s_wa` is
    never applied to it. The result is a **reference-size (unscaled) knee inside an
    `s_wa`-scaled body**. (The body-scaled reference bake from Mode 2 is still produced, but it
    has different filenames and is simply left unused — see [Notes](#notes).)

**How to fix it.** Scale the knee geometry by `s_wa` **about the joint-center origin** — a
plain OSIM-space multiply (`osim_points *= s_wa`), the same operation
[`bake_knee_geometry`][nsosim.scaling.knee_geometry.bake_knee_geometry] already uses. Do it
**early — scale the OSIM recon right after
[`nsm_recon_to_osim`][nsosim.nsm_fitting.nsm_recon_to_osim], before the model-building builders
run** — so the articular surfaces, wrap fits, ligament/muscle attachments, menisci, fat pad,
and `mean_patella` are all computed from correctly-sized geometry and need no per-quantity
fix-up (see [Notes](#notes)).

!!! warning "Not the converter's `scale` argument"
    The natural-looking lever — passing a `scale` to
    [`convert_nsm_recon_to_OSIM`][nsosim.nsm_fitting.convert_nsm_recon_to_OSIM] — is **not**
    the right one: it scales in canonical space about a per-bone affine center, not the shared
    joint-center origin, so it distorts the joint (verified;
    [Coordinate systems §3](coordinate-systems.md) has the numbers). Use the plain OSIM-space
    multiply above.

A heavier alternative is to register each subject onto a **pre-scaled** reference so the recon
inherits the size directly; it changes the fitting step and gives the same result, so scaling
the OSIM recon is simpler.

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
interpolation) into any model, at an appropriate size.

!!! note "🚧 Partial — resize not wired"
    The decode path runs end-to-end, but only at reference size; scaling to a target model is
    not yet implemented.

**How.** The decode path ([`nsosim.decode`](reference/decode.md)) already produces an OSIM-space
knee, but at reference size. Resizing it up/down to a target model is the same lever as
Modes 3 and 4 — a plain OSIM-space multiply about the joint-center origin — just applied on the
synthetic path.

**Status: partial.** The path exists at reference size; the resize is not wired.

---

## Notes

**The clean way to keep everything consistent: scale early.** If a resizing mode (3–5) scales
the OSIM geometry *before* the model-building builders run, then the articular surfaces, wrap
fits, ligament/muscle attachments, menisci, fat pad, and `mean_patella` are all computed from
correctly-sized geometry — nothing downstream needs a separate fix-up. `mean_patella`, for
instance, is just the mean of the (already-scaled) patella vertices
([`center_patella_meshes`][nsosim.model_building.center_patella_meshes], written as the
patellofemoral joint translation by
[`update_osim_model`][nsosim.comak_osim_update.update_osim_model]), so it comes out right on
its own. The only way it becomes a problem is scaling the meshes *late* and forgetting this
offset — which scaling early avoids entirely.

**The reference-knee bake (Mode 2) is not a leak.** When a subject or synthetic recon is
swapped in (Modes 3–5), the `s_wa`-baked `smith2019-R-*.stl` from Mode 2 go unused — the recon
STLs have different filenames and the model is repointed to them via
[`save_geometry_files`][nsosim.model_building.save_geometry_files] + the `update_*` helpers.
That's expected; keep the bake for Mode 2, the other modes simply don't use it.

---

## For the future fix plan

A verified starting list:

- **Scale the OSIM recon by `s_wa` about the joint-center origin, early** — right after
  [`nsm_recon_to_osim`][nsosim.nsm_fitting.nsm_recon_to_osim] and before the model-building
  builders — so every derived quantity (articular surfaces, wraps, ligament/muscle
  attachments, menisci, fat pad, `mean_patella`) inherits the right size. Use a plain
  OSIM-space multiply, **not** the converter's `scale` argument (see
  [Mode 3](#mode-3-personalized-knee-scaled-to-the-gait-body)).
- Thread the synthetic path ([`nsosim.decode`](reference/decode.md)) the same way, so Mode 5
  generalizes.
- Keep mass/inertia owned by COMAK body scaling (the orchestrator two-pass) — a geometry fix
  touches geometry only.
- For true size (Mode 4), prefer `'similarity'` registration + restore the recorded scale via
  the same OSIM-space multiply; `reg_mode='rigid'` is the simpler but weaker fallback.

See `.claude/plans/scaling-and-spaces-documentation.md` (Stage 5) for the fix-plan scope.
