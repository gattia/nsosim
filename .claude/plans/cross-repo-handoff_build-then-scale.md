# Cross-repo handoff: "build, then scale" is ready in nsosim

**For:** the AIs working in `comak_gait_simulation` (the production/wiring repo).
**From:** nsosim. **Date:** 2026-06-08.
**TL;DR:** nsosim can now size a personalized knee to a gait body **without re-fitting** — build
the knee once (GPU), then scale the built model per gait subject (CPU). The library side is done
and verified. Your job: adopt this order in production, delete the now-dead path, add a CPU-only
"scale-and-simulate" pipeline, and validate it on the 30 subjects already processed.

---

## 1. What changed in nsosim (both plans, now completed)

Two plans were completed here:
- `.claude/plans/completed/scaling-and-spaces-documentation_COMPLETED.md` — documented the
  scale/space behaviour of the whole knee-build pipeline (4 coordinate spaces, the MRI→OSIM
  transform chain, COMAK body scaling) as a mkdocs site under `docs/`.
- `.claude/plans/completed/knee-scaling-fix_COMPLETED.md` — the Mode-3/5 "build, then scale" fix.

**The key capability:** `nsosim.scaling.scale_comak_model(base_osim=..., ab_scaled_osim=..., ...)`
already scales an **entire built personalized knee** by the gait body factor `s_wa` when you hand
it a *built* model as `base_osim` (instead of the generic reference base). No new API, no core code
change — it was a verification + tests + docs effort.

Why it works: by the time a Mode-1 model is built, both knee-geometry generators have run and their
outputs are attached to the knee bodies. So the ordinary body-scaling machinery scales everything:
- `bake_knee_geometry` bakes **whatever STL each knee body points at** (it reads `mesh_file` off
  the model) — on a built model that's the recon STLs, scaled by `s_wa` about the joint-center
  origin;
- OpenSim ScaleTool scales the wraps, ligament/muscle path points, joint frames, and the patella
  placement offset (`pf_tx/ty/tz_r`) by `s_wa`;
- the orchestrator's two-pass sets masses (AB per-body + global renorm) and rescales inertia by the
  mass ratio — and because ScaleTool already applied the geometric `s_wa²` to inertia, the final
  inertia is physically correct (`I/m` scales by `s_wa²`).

**What's verified (so you don't re-derive it):** `tests/scaling/test_build_then_scale.py`, on a
real built Mode-1 model (OAI 9003175) scaled to a *different* gait subject (RSubject_121,
`s_wa ≈ 0.973`). Every knee component scales by exactly `s_wa`: recon/cartilage/menisci/fat-pad
STLs (per-vertex), wraps, the full ligament + muscle path-point set (181 + 5, set preserved
exactly), the patella offset, knee-body inertia (`I/m × s_wa²`), and the model still
initializes/realizes. Cartilage–bone proximity and ligament reference strains are preserved.

**Authoritative docs to read (trustworthy, code-verified):**
- `nsosim/docs/coordinate-systems.md` — the four spaces, the full transform chain, §5 COMAK body
  scaling (the `s_wa` lever, the bake, the inertia/mass handling).
- `nsosim/docs/deviations.md` — the **knee-sizing modes** catalog (Modes 1–5), with Mode 3/5 marked
  "works (build, then scale)" and Mode 4 marked "not implemented."

---

## 2. The production bug to fix (Pathway B)

**Current (wrong) order:** `multigait/prepare_gait_subject.py` runs COMAK body scaling (Stage X)
to make a body-scaled base, writes a Stage-Y config whose `base_model_dir` is that scaled dir, and
the knee build then runs *on the body-scaled base*. Result: a reference-size recon knee inside an
`s_wa`-scaled body — a geometric mismatch (the surface COMAK computes contact pressure on is the
wrong size).

**The fix:** **build, then scale.**
1. Build the personalized knee on a **reference-size** base (the standard Mode-1 build).
2. Then call `scale_comak_model(base_osim=<built model>, ab_scaled_osim=<subject AB output>, ...)`
   to size the built model to the gait body. CPU-only; no NSM re-fit.

Mechanism corrections the old `SCALING_WORKFLOW_MAP.md` got wrong (do not propagate them):
- It's **repoint + orphan**, not overwrite-by-filename. The recon STLs have different filenames
  than the baked reference STLs; the build repoints the model to the recon and leaves the baked
  reference STLs orphaned on disk. Under build-then-scale the bake operates on the recon directly,
  so the orphaning goes away.
- Pathway B is wired via **Step 1 + the Stage-Y config**, not `comak_2_pathway_b.py` (that's the
  Step-2 COMAK driver, which consumes the already-built model).
- The resize lever is a **plain OSIM-space multiply about the joint-center origin** (what
  `bake_knee_geometry` does), **not** the converter's `scale` argument (that scales in canonical
  space about a per-bone affine center and distorts the joint).

---

## 3. Concrete tasks for the comak repo

1. **Reorder Pathway-B to build-then-scale.** In `multigait/prepare_gait_subject.py` (and the
   Stage-Y config it writes / `submit_stage_y.sh`): build the knee on a reference base, then run
   `scale_comak_model` on the built model. Confirm the built model's knee bodies end up at
   `s_wa × reference` size inside the `s_wa`-scaled body.

2. **Deprecate / delete dead code.** Once build-then-scale is in:
   - The "build on a body-scaled base" path becomes obsolete — remove or clearly deprecate it.
   - The orphaned Stage-X bone/cart/contact STL bake is dead weight under Pathway B — audit, then
     drop if nothing else needs it.
   - Audit other scripts for anything only used by the old order; deprecate with a note, then remove.

3. **New CPU-only "scale-and-simulate" pipeline.** Write a fresh set of scripts that, per gait
   subject, **reuse one already-built Mode-1 knee** and only do CPU work:
   - input: the built Mode-1 model (built once, GPU — already on disk) + the subject's
     AddBiomechanics output (`match_markers_and_physics.osim`);
   - step A (CPU): `scale_comak_model(base_osim=built_mode1, ab_scaled_osim=subject_AB, ...)` →
     subject-sized COMAK model;
   - step B (CPU): run the COMAK physics simulation on that scaled model (the Step-2 COMAK driver —
     no GPU; GPU is only for the NSM fit, which is already amortized).
   Keep it batchable (one job per subject) and idempotent.

4. **Validate on the first 30 subjects already processed.** Run the new pipeline as a test set over
   the 30 subjects you've already built/scaled, and confirm: the scaled knee is `s_wa × reference`
   (not reference-size), the COMAK sim runs to completion, and contact pressures are computed on the
   correctly-sized surface. Spot-check a couple against the old output to quantify the difference.

5. **Update the comak-repo docs to match.**
   - Correct or delete `SCALING_WORKFLOW_MAP.md` (it's stale — see §2). Replace its body with a
     one-line pointer to `nsosim/docs/coordinate-systems.md` + `deviations.md`, or delete it.
     **No correction banner** (owner preference) — fix in place or remove.
   - Document the new build-then-scale order and the CPU-only scale-and-simulate pipeline.

---

## 4. What is NOT done (and is out of scope for this handoff)

- **Mode 4** (keep the knee at the subject's native MRI true size, "Pathway C") is **not
  implemented** in nsosim — design sketch only. Don't assume it exists.
- A cross-check that build-then-scale ≈ fit-on-scaled-base hasn't been run (needs a fresh GPU fit
  on a body-scaled base). Optional confidence check; see nsosim `.claude/plans/backlog.md`.
- The built-model test fixture is local-only (too large for git). If you want nsosim's end-to-end
  tests to run in CI, promote it to a GitHub Release (backlog).
