# Knee Sizing — Modes & Status

When you drop a knee into a COMAK body model, its **size** has to match the body it's going
into. This page lists the modes you might want, how each is (or would be) achieved, and
what's built today. The mechanics they rest on — the coordinate spaces, the similarity
registration, `s_wa`, and the converter's scale hook — are on
[Coordinate systems & pipeline](coordinate-systems.md).

The body the knee goes into is either left at **reference size** (the default base model) or
**scaled to a gait subject** (`s_wa`, via [COMAK body scaling](coordinate-systems.md#5-comak-body-scaling-and-how-it-meets-the-knee-build)).
Crossed with where the knee geometry comes from (the generic reference knee, a subject MRI
fit, or a decoded latent) — and, for a same-subject build, whether you want true anatomical
size — that gives the modes below.

| Mode | Knee geometry | Body | Knee size | Status |
|---|---|---|---|---|
| [1](#mode-1-personalized-knee-unscaled-reference-size-model) — personalized, unscaled model | subject MRI | reference (default) | reference (auto-matches) | **works — most common** |
| [2](#mode-2-generic-knee-scaled-to-the-gait-body) — generic, gait-scaled | reference | gait subject (`s_wa`) | `s_wa` (baked) | **works** |
| [3](#mode-3-personalized-knee-scaled-to-the-gait-body) — personalized, gait-scaled | subject MRI | gait subject (`s_wa`) | should be `s_wa`, **stays reference** | **bug** |
| [4](#mode-4-personalized-knee-true-anatomical-size) — personalized, true size | subject MRI | the MRI subject's own | true MRI size | **not implemented** |
| [5](#mode-5-synthetic-knee-scaled-to-a-model) — synthetic, scaled | decoded latent | any | any | **partial** — path exists at scale = 1 |

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

**Status: works — and it's the most common path.** A reference-size knee in a reference-size
body has no mismatch. (It is *only* when the body gets gait-scaled, [Mode 3](#mode-3-personalized-knee-scaled-to-the-gait-body),
that an un-scaled knee becomes a problem — same knee, but now the body moved and the knee
didn't follow.)

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

**How to fix it (two routes).**

- **Scale the recon.** Multiply the recon by `s_wa` about the joint-center origin at the
  OSIM-entry point ([`nsm_recon_to_osim`][nsosim.nsm_fitting.nsm_recon_to_osim] /
  [`convert_nsm_recon_to_OSIM_`][nsosim.nsm_fitting.convert_nsm_recon_to_OSIM_]), and carry
  the same scale into the wrap-fit input and `mean_patella` ([Notes](#notes)).
- **Register to a scaled reference.** Pre-scale the reference knee and register each subject
  onto *that*, so the recon inherits the size directly. (A cleaner variant — always register
  to the native reference, then resize via the converter's `scale` hook — is described in
  [Coordinate systems §3](coordinate-systems.md).)

---

## Mode 4 — personalized knee, true anatomical size

**What you want.** The gait subject *is* the MRI subject — keep their real knee at its real
size, rather than normalizing it to the reference.

**How it would work.** Two routes — the first is usually safer:

- **Store and restore the scale (preferred).** Register with `'similarity'` as usual — which
  gets good shape correspondence regardless of how much the subject differs in size from the
  reference — then **record** the scale it removed and **restore** it via the converter's
  `scale` hook in [`convert_nsm_recon_to_OSIM`][nsosim.nsm_fitting.convert_nsm_recon_to_OSIM],
  instead of leaving the knee at reference size.
- **Don't divide the size out.** Register with `reg_mode='rigid'` (rotation + translation
  only) so the recon keeps true size from the start. Simpler, but a rigid (no-scale) fit
  aligns *worse* when the subject's knee differs much in size from the reference — so the
  similarity-then-restore route is usually the better choice.

**Status: not implemented.** Neither route is wired as a complete, supported mode yet.

---

## Mode 5 — synthetic knee, scaled to a model

**What you want.** Decode an arbitrary latent (a synthetic joint, a shape-mode sweep, an
interpolation) into any model, at an appropriate size.

**How.** The decode path ([`nsosim.decode`](reference/decode.md)) already runs through the
full converter [`convert_nsm_recon_to_OSIM`][nsosim.nsm_fitting.convert_nsm_recon_to_OSIM],
which carries the `scale` hook — but it is currently called with `scale = 1`, so the output
lands at reference size. Resizing it up/down to a target model is the same lever as Modes 3
and 4, just applied on the synthetic path.

**Status: partial.** The path exists at `scale = 1`; the resize is not wired.

---

## Notes

Anything that resizes the knee (Modes 3–5) has to carry two more things along, or the result
won't be internally consistent:

- **`mean_patella` must scale with the meshes.** The patella is centered by subtracting its
  mean position, and that offset is written as the patellofemoral joint translation
  ([`center_patella_meshes`][nsosim.model_building.center_patella_meshes] →
  [`update_osim_model`][nsosim.comak_osim_update.update_osim_model]). It is a reference-size
  scalar; if the meshes are scaled by `s_wa` but this offset is not (about the same origin),
  the patellofemoral kinematics drift away from the geometry.
- **The reference-knee bake is Mode 2's tool, not a leak.** When a subject or synthetic recon
  is swapped in (Modes 3–5), the `s_wa`-baked `smith2019-R-*.stl` from Mode 2 go unused (the
  recon STLs have different filenames and the model is repointed to them via
  [`save_geometry_files`][nsosim.model_building.save_geometry_files] + the `update_*`
  helpers). That's expected — keep the bake for Mode 2; the other modes simply don't use it.

---

## For the future fix plan

A verified starting list (not work items on this page):

- Apply the knee scale at the OSIM-entry point
  ([`nsm_recon_to_osim`][nsosim.nsm_fitting.nsm_recon_to_osim] / the converter), **and** to
  the wrap-fit input, the warp-path attachment converter, and `mean_patella`.
- Thread the synthetic path ([`nsosim.decode`](reference/decode.md)) the same way, so Mode 5
  generalizes.
- Keep mass/inertia owned by COMAK body scaling (the orchestrator two-pass) — a geometry fix
  touches geometry only.
- For true size (Mode 4), either default to `reg_mode='rigid'` for that path or store/restore
  the similarity scale through the converter's `scale` hook.

See `.claude/plans/scaling-and-spaces-documentation.md` (Stage 5) for the fix-plan scope.
