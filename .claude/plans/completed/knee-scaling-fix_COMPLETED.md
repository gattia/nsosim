# Plan: Scale a built knee to a gait body (the Mode-3 `s_wa` fix)

**Status:** Complete (2026-06-08) — nsosim deliverable DONE & verified end-to-end. No core code
change was needed — `scale_comak_model` already performs the whole "build, then scale" operation
when handed a built model as `base_osim`. The work was verification + tests + docs. The only
remaining item is the **production wiring in the comak repo** (reorder Pathway-B to
build-then-scale), tracked in [`../backlog.md`](../backlog.md). See **Completion Notes** and
**Verification results** below.
This is "Stage 5" of
[`scaling-and-spaces-documentation_COMPLETED.md`](scaling-and-spaces-documentation_COMPLETED.md); read that plan's
**Handoff state & docs↔code trust** section first.
**Created:** 2026-06-07
**Owner:** nsosim (geometry); the comak repo wires it.

---

## Completion Notes

**Date completed:** 2026-06-08

**Summary.** The Mode-3/5 "build, then scale" deliverable is done on the nsosim side with no core
code change: `scale_comak_model` already scales an entire built personalized knee by `s_wa` when
given the built model as `base_osim`. Verified end-to-end on a real built Mode-1 model
(OAI 9003175) scaled to a different gait subject (RSubject_121, `s_wa ≈ 0.973`). A post-completion
review (2026-06-08) hardened the test coverage and confirmed inertia is correctly scaled.

**Changes made.**
- Docstring-only notes added to `nsosim/scaling/orchestrator.py` and `nsosim/model_building.py`
  documenting the build-then-scale route (commit 09c4459).
- `tests/scaling/test_build_then_scale.py` — the 12-test build-then-scale suite (commit 09c4459;
  hardened 2026-06-08, see below).
- Docs: `docs/coordinate-systems.md` §5, `docs/deviations.md` Modes 3/5 set to "works" (09c4459).
- 2026-06-08 hardening: tightened the ligament/muscle point tests to assert `before==after`
  set-equality + the real counts (181 lig + 5 muscle) instead of a loose `≥50` floor; added
  `test_knee_inertia_scales_by_s_wa_squared`; fixed the `tests/docs/test_doc_references.py`
  encoding bug (`read_text(encoding="utf-8")`); refreshed the verification table + the parent
  plan's stale "no regression test" caveats.

**Tests.** `tests/scaling/test_build_then_scale.py` — 14 test items (incl. parametrized), all pass
against the real built-model fixture (~4 min; skips if absent). Full `tests/scaling/`,
`tests/test_transform_chain.py`, `tests/docs/` run green after the encoding fix.

**Additional issues resolved (beyond original scope).**
- Doc-reference guard `tests/docs/test_doc_references.py` failed 26× in a *full* `pytest` run
  (OpenSim XML I/O flips the process locale to ASCII mid-run; the un-encoded `read_text()` then
  choked on non-ASCII docstring chars). Fixed by pinning UTF-8. It passed in isolation before, so
  this was masked.
- Verified inertia is genuinely correct (not just mass): specific inertia `I/m` scales by exactly
  `s_wa²` for every knee body (ScaleTool applies the geometric factor; the two-pass only adjusts
  the mass dimension). Now permanently tested.

**Challenges / design decisions.**
- **No new public API** (owner decision): Modes 2/3/5 are the same `scale_comak_model` call — they
  differ only in what `base_osim` points at. A wrapper would add a name and no logic.
- **Point-count robustness:** chose `before==after` set-equality over a hardcoded count check as
  the primary guard, with the measured counts as floors — catches silent point loss without being
  brittle to legitimate model changes.

**Things to note for future work.**
- The remaining work is **external** (comak repo): reorder Pathway-B to build-then-scale. Tracked
  in [`../backlog.md`](../backlog.md), and detailed in the cross-repo handoff summary.
- Mode 4 (keep native MRI true size) is **not implemented** — see backlog.
- The built-model fixture is untracked (too large for git); promoting it to GitHub Releases would
  let the end-to-end tests run without local data (backlog).
- The contingent cross-check (build-then-scale ≈ fit-on-scaled-base) was not run — it needs a fresh
  GPU fit on a body-scaled base (backlog).

---

## Verification results (2026-06-07)

Ran `scale_comak_model(base_osim=<built Mode-1 model>, ab_scaled_osim=<RSubject_121 AB>)` on a
**real built Mode-1 model** (OAI subject 9003175's personalized recon knee in a reference-size
COMAK body) scaled to a **different** gait subject (RSubject_121, `s_wa = 0.97298`) — the genuine
Pathway-B Mode-3 scenario. Scaling needs **no GPU** (the fit is already amortized in the built
model). Every knee component scaled by exactly `s_wa`:

| Component | Result |
|---|---|
| Recon bone/cartilage/menisci/fat-pad STLs | × `s_wa`, per-vertex to ~1e-8 m ✓ |
| Wraps (translation + radius/dims) | × `s_wa` ✓ |
| Ligament + muscle path points (186 on knee bodies: 181 lig + 5 muscle) | × `s_wa`; point set preserved exactly ✓ |
| Patella placement offset (`pf_tx/ty/tz_r` = `mean_patella`) | × `s_wa` ✓ |
| Origin-scaling placement, every knee body | preserved (centroid × `s_wa`) ✓ |
| Cart–bone proximity / ligament reference strains | preserved under scale ✓ |
| Scaled model `initSystem()` / `realizePosition()` | loads & realizes ✓ |
| Knee-body inertia | specific inertia `I/m` × `s_wa²` (radius-of-gyration), every knee body, to ~2e-16 ✓ |
| Mass | set by the orchestrator two-pass (AB per-body + global renorm); inertia rescaled by mass ratio ✓ |

**Resolutions to the open questions below:**
- **Q1 (every component scales):** yes — table above.
- **Q2 / Q4 (patella centering × scaling, PF offset):** RESOLVED — ScaleTool scales the
  `pf_tx/ty/tz_r` coordinate *defaults* by the body factor (probed: ratio == scale), and both PF
  bodies are knee bodies (so the factor is `s_wa`). The centered patella shape scales by `s_wa`
  about origin *and* its offset scales by `s_wa` → placement preserved. No explicit fix needed.
- **Q3 (`project_coronary=False` meniscus path):** the meniscus/coronary attachments are warp
  products that ride ScaleTool's ligament scaling; all 181 knee ligament points scaled by `s_wa`
  and reference strains held — tautness unchanged.
- **Q5 (fallback / mid-build interception):** not needed; build-then-scale is correct.

**Deliverable shape (decided with owner):** **no new public API.** `scale_comak_model` covers
Modes 2/3/5 — they differ only in what `base_osim` points at (reference base / MRI-built /
synthetic-built). A wrapper would add a name and no logic. Discoverability handled by a docstring
note + the docs flip. Mode 5 is the same call (decode → `build_joint_model` → `scale_comak_model`).

**Tests:** `tests/scaling/test_build_then_scale.py` (12 tests) — per-component scaling on the real
built model (skips if absent): STLs, wraps, the full 181-point ligament + 5-point muscle sets
(asserted with before==after set-equality so a silent point drop fails, not a loose floor),
patella offset, knee-body inertia (`I/m` × `s_wa²`), origin-placement regression on both the built
model and the reference base (synthetic in-repo `s_wa`), and coherence (cart–bone, reference
strains). The built-model fixture lives at `untracked/built_models/mode1_9003175_00m_RIGHT/` (too
large for git; override with `NSOSIM_BUILT_MODEL_OSIM`).

**Remaining (comak repo, out of nsosim scope):** the Pathway-B scripts still build the knee on a
body-scaled base (wrong order). Switch them to build on a reference base, then call
`scale_comak_model` on the built model. Optional future: promote the built-model fixture to a
GitHub-Releases download so the end-to-end tests run without local data.

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

**Already locked (build on these, don't redo):**
- `s_wa` formula + isotropic knee-body scaling + anisotropic non-knee passthrough —
  `tests/scaling/test_scale_factors.py`.
- Mode-2 scaling coherence: cart–bone proximity scales by `s_wa`, ligament **reference strains
  preserved** (⇒ slacks scaled with path length), wrap translations stay in-body —
  `tests/scaling/test_nontrivial.py`.
- The converter `scale` arg is NOT a clean OSIM resize (use the OSIM-space multiply instead) —
  `tests/test_transform_chain.py::TestConverterScaleArgIsNotCleanResize`.
- Full MRI→OSIM transform chain / spaces / `fem_ref_center` — `tests/test_transform_chain.py`.

**Strategy:** add a focused integration test alongside *each* mode as it is built (Mode 3 first,
then 4/5), so the documented behavior of every mode gets a guard that fails if it regresses.

**New tests for the Mode-3 fix:**

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
