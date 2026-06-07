# Plan: Scale a built knee to a gait body (the Mode-3 `s_wa` fix)

**Status:** Proposed — design/verification, not started. This is "Stage 5" of
[`scaling-and-spaces-documentation.md`](scaling-and-spaces-documentation.md); read that plan's
**Handoff state & docs↔code trust** section first.
**Created:** 2026-06-07
**Owner:** nsosim (geometry); the comak repo wires it.

Docs to lean on (trustworthy — see the trust summary in the parent plan):
`docs/coordinate-systems.md` §3 (scale lever), §5 (COMAK body scaling); `docs/deviations.md`
Mode 3 (the bug + two generators), Notes, "For the future fix plan".

---

## The goal (owner's intent)

Run the **expensive NSM fitting (GPU) once**, in Mode 1 (the standard personalized build on a
reference-size base), and then **reuse that built knee** for any number of gait subjects by
**scaling it** — never re-fitting. For each new gait subject:

1. **Scale the body** with the existing COMAK body scaling (Mode 2 / `scale_comak_model`,
   `s_wa` from the subject's AddBiomechanics output).
2. **Scale the Mode-1 knee** (its meshes, contact surfaces, wraps, ligament/muscle
   attachments, and the patellofemoral offset) by the same `s_wa`, about the joint-center
   origin, and place it in the scaled body.

Net result a *correct* Mode 3: a personalized knee, sized to the gait body, with the GPU fit
amortized across all gait subjects.

---

## Why the obvious approach is the attractive one

By the time you have a **built Mode-1 model**, *both* knee-geometry generators have already run
and their outputs are baked into the model:
- the **recon meshes** (and everything extracted from them: articular surfaces, menisci, fat
  pad, patella centering / `mean_patella`);
- the **reference→subject warp** outputs (the **wrap surfaces** and the **ligament/muscle
  attachments**).

That dissolves the "two independent generators" problem (parent plan / `deviations.md` Mode 3):
you no longer have to intercept each generator mid-build. **Run COMAK body scaling
(`scale_comak_model`) on the Mode-1 built model itself** (instead of on the generic reference
base), and the existing machinery should scale the whole knee:
- `bake_knee_geometry` already iterates the *attached geometry of the knee bodies* (not fixed
  filenames), so it scales the **recon** STLs by `s_wa` about the joint-center origin;
- ScaleTool's `extendPostScale` hooks scale the knee bodies' **wraps, ligaments, contact
  meshes**, and **joint frames** by `s_wa`;
- the orchestrator's two-pass sets masses/inertia (already owned by Stage X — leave it).

This is the cleanest expression of "scale a built knee," and it reuses every existing piece.
**It is a design direction, not yet verified** — the open questions below are exactly what a
fix-implementer must check before trusting it.

---

## Constraints (verified — do not violate)

- **Use a plain OSIM-space multiply about the joint-center origin** (`pts *= s_wa`), the
  operation `bake_knee_geometry` uses. **NOT** the converter's `scale` argument — verified
  numerically to scale in canonical space about a per-bone affine center, which distorts the
  joint (`coordinate-systems.md` §3).
- **Leave mass/inertia to Stage X** (the `_apply_per_body_masses` two-pass). A geometry fix
  touches geometry only.
- **`project_coronary=False`** stays. Production builds Mode 1 with
  `build_joint_model(..., project_coronary=False)` (`comak_1_nsm_fitting.py:459`) because
  projecting the coronary (meniscotibial) ligament attachments onto the tibia makes the menisci
  too taut. Any fix must preserve that choice.

---

## Open design questions / knots (resolve + verify before shipping)

1. **Does Stage X actually scale every recon-knee component correctly?** Verify per component on
   a real Mode-1 model run through `scale_comak_model`:
   - recon bone/cart/contact STLs (via `bake_knee_geometry`) — vertices × `s_wa` about origin;
   - wrap objects (translation, radius, length, dimensions) — via ScaleTool;
   - ligament/muscle attachment points — via ScaleTool;
   - knee joint frames / weld translations (already verified for the reference model; re-confirm
     for the recon model).
2. **Patella centering × scaling.** The patella is centered (`mean_patella` subtracted) and its
   wraps/ligaments live in that *centered* frame; `mean_patella` is written as the PF joint
   coordinate offset. Confirm that scaling by `s_wa` keeps the centered patella points correct
   **and** scales the PF offset consistently (owner: "as long as the patella-centered points and
   the offset are correct, it's fine"). This is the most fragile interaction.
3. **Meniscus-ligament path.** Coronary/meniscotibial attachments come from the warp
   (`project_coronary=False`); confirm they scale with the rest under Stage X and don't change
   meniscus tautness.
4. **`mean_patella` / PF offset.** If running Stage X on the built model, does ScaleTool scale
   the PF joint translation by the patella scale factor? If not, scale it explicitly.
5. **Alternative (fallback), unverified:** intercept each generator's OSIM output mid-build and
   scale it (the "two generators" route in `deviations.md` Mode 3). Use only if running Stage X
   on the built model proves to mis-scale some component.

---

## Verification tasks (write these tests as part of the fix)

- **Regression test for origin-scaling placement** (flagged by review, currently missing):
  scaling a built knee about OSIM `(0,0,0)` by `s_wa` must preserve the intended placement for
  **every** knee body (`femur_distal_r`, `tibia_proximal_r`, `patella_r`, both menisci). The
  origins were *measured* in Stage 1 but the scaling operation is untested.
- **End-to-end:** a Mode-1 model scaled by `s_wa` should match (within tolerance) a model the
  *current* (buggy) pipeline produces only in the `s_wa = 1` case, and should differ by exactly
  `s_wa` in knee size otherwise. Check cartilage–bone proximity, ligament reference strains, and
  wrap placement hold up after scaling (mirror the existing Stage-X tests in
  `tests/scaling/test_nontrivial.py`).
- **Cross-check vs. a freshly-fit scaled build** (if one can be produced) to confirm
  reuse-and-scale ≈ fit-on-scaled-base.

---

## Suggested shape of the deliverable

A standalone, testable operation — e.g. `scale_knee_model(mode1_model, s_wa)` (or simply
"run `scale_comak_model` with the Mode-1 model as the base") — that consumes a built Mode-1
model + the Stage-X `s_wa` and emits a scaled model, with **no GPU / no NSM re-fit**. Thread the
synthetic path (`nsosim.decode` / Mode 5) the same way if the feature is meant to be general.
