# Wrap Surface Fitter — Robustness to Input Perturbation

**Status:** Complete (2026-05-12). All acceptance criteria pass on subject 9018389_RIGHT. Iter 1–10 committed on branch `worktree-wrap-fitter-robustness` at `.claude/worktrees/wrap-fitter-robustness`. Closing config (iter10): Procrustes-from-Smith2019 anchor enabled for 5/6 wraps with per-wrap opt-out for Med_Lig_r.
**Goal:** Reduce the wrap fitter's sensitivity to micron-scale input mesh perturbations from the current ~1000–10000× amplification factor, **without biasing fits away from the trusted Smith2019 reference.**
**Created:** 2026-05-11 (rev 2: 2026-05-12)
**Driver:** Verification work in `comak_gait_simulation` traced cascading COMAK biomechanical variance to wrap surface drift. The wrap fitter amplifies 1e-4 mm bone-mesh drift (CUDA reproducibility floor of NSM fitting) into 0.1–1.45 mm wrap surface drift — a 1000–10000× amplification factor for some wraps near singular configurations.

## Critical context: what is and isn't the problem

**Not the problem:** The wrap fitter is already deterministic given fixed input + fixed seed. Verified 2026-05-11: running `build_joint_model` twice on identical bone meshes produced 1 differing line in 15623 (a model-level bookkeeping diff, not numeric). The seed pinning (`set_global_seed`) and `_fit_with_restarts` deterministic-jitter logic in `model_building.py` work correctly.

**The problem (sensitivity, not determinism):** When the input bone mesh changes by ~1e-4 mm at vertex level (the CUDA gradient reproducibility floor of NSM fitting on GPU — accepted as fundamental in the parent plan `DETERMINISTIC_REPRODUCIBILITY_COMPLETED.md`), the wrap fitter produces output drift of up to 1.45 mm because some wraps sit near flat directions in the loss landscape (long ellipsoid sliding along its long axis, near-gimbal-lock Euler angles, etc.).

**This is a sensitivity problem.** The fix is to reduce `∂(wrap_params) / ∂(input_mesh)` in flat directions of the loss landscape — without introducing bias away from the trusted Smith2019 reference geometry.

## What's already in place (iter 1–3 from worktree)

Three commits on branch `worktree-wrap-fitter-robustness`. Review them in [`.claude/worktrees/wrap-fitter-robustness/NOTES.md`](../worktrees/wrap-fitter-robustness/NOTES.md).

| Iter | Commit | Status | Action |
|------|--------|--------|--------|
| 1 | `06c3b9b` Canonicalize ellipsoid output near gimbal lock | **Keep** | Clean win — preserves geometric ellipsoid, kills Euler representation amplification. Stay as-is. |
| 2 | `608e476` Regularize ellipsoid center toward init | **Replace anchor** | Mechanism is right, anchor is wrong. Switch from algebraic-init anchor to Procrustes-from-Smith2019 anchor. |
| 3 | `6be9ec5` Cylinder center reg + ellipsoid axes/quat regs | **Replace anchor + reduce λ** | Same as iter 2. Reduce λ values to ~0.01–0.1× current strength once anchor is trusted. |

## Concerns the review surfaced (and how this revision addresses them)

1. **Anchor was the algebraic geometric init, which has its own biases.** For KnExt cylinders, `fit_cylinder_geometric` gives length ~290 mm (vs Smith2019's 185 mm); pinning the center to that biased init carries the bias into the converged fit. → **Replace anchor with Procrustes-fit to Smith2019-wrap-points-transformed-to-subject-bone-frame.** The anchor becomes the trusted reference.
2. **Capsule_r showed 2.75 mm bias vs unregularized current code.** Likely the largest real biomechanical concern in the iter 3 output, because Capsule_r mediates patellofemoral wrapping. → New anchor should reduce this; verify Capsule_r as the primary acceptance gate.
3. **Cylinder length not a real metric — user explicitly does not care about length.** Drop length out of the acceptance numbers; focus on center + axis direction + radius for cylinders. Length can be 5× the bone extent and still be biomechanically correct.

## Revised approach

### 1. Procrustes anchor from Smith2019 (replaces algebraic init as anchor; new init too)

For each wrap surface, build the anchor by:

1. **Load Smith2019 reference wrap parameters** from the Smith2019 osim using existing `extract_wrap_parameters_from_osim()`.
2. **Sample N points on the Smith2019 wrap surface** (existing `create_meshes_from_wrap_parameters()` already returns a mesh; sample its vertices or a denser surface point cloud).
3. **Transform those points from Smith2019 bone frame → subject bone frame** using NSM canonical-space bone correspondence. The Smith2019 bone mesh and the subject bone mesh share an NSM canonical space — for each Smith2019 wrap surface point, find the bone-frame transformation that takes Smith2019 bone to subject bone at that location and apply it.
   - Simplest viable approach: use the alignment JSON `linear_transform` for each bone to transform `Smith2019 wrap point → canonical space → subject wrap point`. Both bones live in the same canonical space, so a rigid+scale transform suffices for the first cut.
   - Slightly better: also apply per-vertex displacement from Smith2019 bone-mesh-vertex to corresponding subject bone-mesh-vertex (via NSM canonical correspondence) to the nearest-vertex wrap point. This captures local bone-shape differences. Try the simple version first; only escalate if Capsule_r is still off.
4. **Procrustes-fit a parametric wrap (cylinder or ellipsoid) to the transformed point cloud.** Algebraic fit on the point cloud directly — no SDF, no margin loss, no optimization. Cylinder: axis from PCA of the points, radius and length from projected extents. Ellipsoid: algebraic fit to the implicit quadratic (existing `fit_ellipsoid_algebraic()` should work directly on this transformed point cloud).
5. **The Procrustes fit is BOTH the initialization for L-BFGS refinement AND the anchor for the regularizer.** Single source of truth.

This replaces the call to `fit_cylinder_geometric` / `fit_ellipsoid_algebraic` for ellipsoid/cylinder wraps for which a Smith2019 reference exists. (All 10 Smith2019 wraps qualify.) For wraps without a reference, fall back to the existing algebraic init and current regularizer.

### 2. Multi-start with deterministic perturbations and tiebreaker

Replace the current `_fit_with_restarts` jitter mechanism with:

- **Restart 0**: init = anchor (Procrustes fit), no perturbation.
- **Restarts 1..K**: init = anchor + deterministic perturbation along a canonical direction. For an ellipsoid: 6 translation perturbations (±x̂, ±ŷ, ±ẑ at 2 mm), optionally 6 rotation perturbations (±5° about each body axis), 6 axis-size perturbations (±10% on each radius). For a cylinder: 6 translation, 2 rotation, 2 radius. Tune K to ~6–10 per wrap if budget is tight.
- All restarts run the existing optimization with the new (lower) regularizer.
- **Selection**: pick lowest converged geometric loss.
- **Deterministic tiebreaker**: when the top-k losses are within ε (e.g., 1% relative), pick the one with smallest `||θ_final − anchor||`. This guarantees that runs with tiny input differences pick the *same* restart when the landscape is flat → kills selection-induced amplification.

### 3. Reduce regularizer strength

With a trustworthy anchor, λ no longer needs to dominate flat directions — the tiebreaker handles that. Target:
- `lambda_center_reg`: 1.0 → ~0.05 (ellipsoid and cylinder)
- `lambda_axes_reg`: 0.1 → ~0.005
- `lambda_quat_reg`: 0.1 → ~0.005

Tune empirically against acceptance criteria. The goal is "just enough prior to gently bias flat directions toward anchor; tiebreaker does the rest."

### 4. (Optional, only if 1–3 don't meet acceptance) Truncated SDF regression refinement

After the existing margin-loss optimization converges, run a short refinement phase using truncated SDF regression: `loss = (wrap_sdf − true_sdf)² * weight`, where `weight = exp(-|true_sdf|/τ)` (τ ≈ 2 mm) restricts contribution to a band near the wrap boundary. This addresses the flat-far-from-boundary loss landscape directly. Skip if criteria 1–3 already pass.

**Note**: SDF regression was reportedly tried before and had issues. Likely culprits to watch for:
- Outlier dominance from far-from-boundary points → fixed by the `exp(-|true_sdf|/τ)` weighting above.
- Wrap-SDF is exact on the surface but only approximate away from it → also fixed by truncated weighting.
- Float32 underflow when working in meters → consider scaling SDF by 1000 internally (mm).
- L-BFGS step-size collapse → reduce LR or use trust-region L-BFGS for this phase.

## Out of scope (explicit)

- **Cylinder length is not a metric.** Length can be 5× bone extent and that's fine. Do not regularize, validate, or report on cylinder length unless it directly affects center or radius drift.
- **NSM fitting itself.** Accept the 1e-4 mm CUDA floor.
- **Smith2018ArticularContactForce, opensim-jam, the .osim file format.**
- **`create_ellipsoid_polydata` / `create_cylinder_polydata` rotation convention bug** (noted in iter4 open issues). Out of scope here; track separately.

## Acceptance criteria

On subject 9018389_RIGHT, Run A vs Run B (`e2e_determinism_20260510_221720_{A,B}`):

1. **All 10 wrap surfaces ≤ 0.10 mm ASSD A vs B.** Same as before. Iter 3 already meets this (<50 µm everywhere); revision must not regress.
2. **No regression on already-stable wraps.** The 6 axis-aligned wraps must stay bit-identical or near it (≤0.05 mm).
3. **Capsule_r center within 0.5 mm of Smith2019 reference (transformed to subject frame).** Tighter than before because Procrustes anchor makes this directly achievable. Replaces the vague "within 10% of production" criterion.
4. **Determinism preserved.** `A_v1 vs A_v2` ≤ 1 differing line in `build_joint_model` output.
5. **No new randomness leaks.** All RNG goes through `set_global_seed`-aware paths.
6. **(New) Anchor reproducibility.** The Procrustes anchor computed from Smith2019 + subject-bone correspondence is bit-identical across the two runs (since the inputs are the Smith2019 osim + the subject bone, both of which are bit-stable upstream).

## Test command

```bash
A_BONES=/dataNAS/people/aagatti/projects/comak_gait_simulation_results/e2e_determinism_20260510_221720_A/9018389_00m_RIGHT
B_BONES=/dataNAS/people/aagatti/projects/comak_gait_simulation_results/e2e_determinism_20260510_221720_B/9018389_00m_RIGHT

python /dataNAS/people/aagatti/projects/comak_gait_simulation/tests/swap_experiments/isolate_build_joint_model.py \
    --a-run /dataNAS/people/aagatti/projects/comak_gait_simulation_results/e2e_determinism_20260510_221720_A \
    --b-run /dataNAS/people/aagatti/projects/comak_gait_simulation_results/e2e_determinism_20260510_221720_B \
    --output-root /tmp/wrap_fitter_test_<timestamp>
```

Additionally, for criterion 3 verify Capsule_r center against Smith2019: extract Capsule_r from the produced subject osim, compute center distance vs `Procrustes_anchor.translation`, assert < 0.5 mm.

## Worktree workflow

Already set up at `.claude/worktrees/wrap-fitter-robustness`. Branch `worktree-wrap-fitter-robustness`. Build on top of the iter 1–3 commits. Each iter should be a separate commit.

## Approach ordering (suggested)

1. **Build the Procrustes anchor pipeline.** Single function `procrustes_anchor_from_smith2019(wrap_name, bone_name, subject_alignment_json) -> wrap_surface_params`. Verify on Capsule_r: anchor.translation should land on the femur condyles between bone and patella, comparable to Smith2019's value transformed into subject frame.
2. **Wire anchor into EllipsoidFitter + CylinderFitter as both init and regularizer target.** Add `anchor_params` kwarg; when present, override the algebraic init and the existing `_init_*` regularizer anchors. Reduce λ to target values.
3. **Run isolation test.** Verify criterion 1 (≤ 0.10 mm A vs B) and criterion 3 (Capsule_r within 0.5 mm of anchor) both pass. If yes, ship.
4. **(Conditional) Add multi-start with deterministic perturbations + tiebreaker.** Only if step 3 doesn't pass criterion 3, or if criterion 1 regresses.
5. **(Conditional) Add TSDF refinement.** Only if 3–4 don't close the gap.

## Reporting

Same format as iter 1–3. Update `NOTES.md` with each iter. Final commit message should summarize:
- What anchor source is used (Procrustes from Smith2019).
- Final λ values.
- Whether multi-start was needed.
- Acceptance criteria final pass/fail table.

## Cross-references

- [`.claude/worktrees/wrap-fitter-robustness/NOTES.md`](../worktrees/wrap-fitter-robustness/NOTES.md) — full iter 1–10 working log
- [`comak_gait_simulation/.claude/plans/NSOSIM_WRAP_FITTER_MIGRATION.md`](file:///dataNAS/people/aagatti/projects/comak_gait_simulation/.claude/plans/NSOSIM_WRAP_FITTER_MIGRATION.md) — downstream integration plan
- [`comak_gait_simulation/tests/swap_experiments/CLAUDE.md`](file:///dataNAS/people/aagatti/projects/comak_gait_simulation/tests/swap_experiments/CLAUDE.md) — diagnosis chain
- [`comak_gait_simulation/.claude/plans/VERIFICATION_PHASES_BC_PLAN.md`](file:///dataNAS/people/aagatti/projects/comak_gait_simulation/.claude/plans/VERIFICATION_PHASES_BC_PLAN.md) — parent verification work
- [`nsosim/.claude/plans/completed/DETERMINISTIC_REPRODUCIBILITY_COMPLETED.md`](DETERMINISTIC_REPRODUCIBILITY_COMPLETED.md) — parent plan

## Completion Notes (2026-05-12)

### Summary

All five acceptance criteria pass on subject 9018389_RIGHT (single-subject
validation; multi-subject Level 3 is handed off to the comak_gait_simulation
side per `NSOSIM_WRAP_FITTER_MIGRATION.md`). Closing configuration: anchor
ON by default for 5/6 wraps, `wraps_to_skip_anchor=['Med_Lig_r']` by default
to recover its fit-quality regression. Cylinder rotation pinned via
`lambda_axis_reg=0.1`. Net result: ~1000–10000× A↔B amplification suppressed
to ≤1 µm/center, mean fit-quality +1.36 % over the iter3 starting point.

### Changes made (commit hashes on `worktree-wrap-fitter-robustness`)

**Pre-this-session (iter 1–3, already on the branch):**
- `06c3b9b` Canonicalize ellipsoid output near gimbal lock (rotation_utils.canonical_ellipsoid_pose, 11 tests)
- `608e476` Ellipsoid regularize fit center toward initialization
- `6be9ec5` Cylinder + ellipsoid axes/quat regularizers
- `ae45077` NOTES: criterion 3 verification analysis

**This session (iter 4–10 + cleanup):**
- `d0dd301` iter4: Procrustes anchor scaffolding — `nsosim/wrap_surface_fitting/procrustes_anchor.py` + 11 unit tests
- `831b352` autoformat
- `d5b1d8c` iter5: Wire `anchor_params` into Ellipsoid/CylinderFitter + 5 wiring tests
- `58ae4d9` autoformat
- `4a8114a` iter6: λ defaults + `smith2019_osim_path` config key
- `d548de0` Package-level exports for procrustes anchor helpers
- `c055ef7` NOTES: iter3 criterion-3 baseline table
- `ddd0eb3` iter8: `lambda_axis_reg=0.1` cylinder fix + iter7 results in NOTES
- `3426bc6` iter8: NSM correspondence diagnostic + closing report in NOTES
- `21148d7` iter8: track diagnostic scripts (fit-quality + correspondence)
- `63ec10c` iter8: TEST 3 boundary-distance jitter
- `fc57092` Untrack egg-info; ignore SLURM-output scratch
- `6b1cf88` Per-wrap anchor opt-out (`wraps_to_skip_anchor=['Med_Lig_r']` default)
- `87ffd74` iter10: per-wrap opt-out validated end-to-end

### Tests

40+ new wrap-fitter tests across:
- `tests/test_procrustes_anchor.py` — 11 tests
- `tests/test_anchor_wiring.py` — 7 tests (incl. opt-out integration test)
- `tests/test_rotation_utils.py` — additions for `canonical_ellipsoid_pose`

All pass. 96/96 pass across the suite (`test_fitting`, `test_anchor_wiring`,
`test_procrustes_anchor`, `test_rotation_utils`, `test_mesh_labeling`).
Pre-existing fixture errors in `test_knee_assembly.py` are unrelated
(missing Smith2019 mesh files in `tests/fixtures/`; reproduce on baseline).

### SLURM validation (subject 9018389_RIGHT, single subject)

| Iter | Job | A vs B differing lines | Notes |
|---|---|---|---|
| 7 | 46868 | 122/15623 | First end-to-end with anchor on, WrapCylinder rotation 0.05° drift |
| 9 | 46869 | 111/15623 | `lambda_axis_reg=0.1` fix — cylinder rotation 1 µrad |
| 10 | 46872 | 111/15623 | Per-wrap opt-out for Med_Lig_r — fit-quality recovered |

### Acceptance criteria final pass/fail (iter10 closing config)

| # | Criterion | Status |
|---|---|---|
| 1 | All wraps ≤ 0.10 mm A vs B | ✅ centers/radii/dims bit-stable. Med_Lig_r ellipsoid rotation 0.073° → ~38 µm spatial drift, under gate. |
| 2 | No regression on axis-aligned wraps | ✅ |
| 3 | Capsule_r within 0.5 mm of Smith2019 anchor | ✅ 0.33 mm |
| 4 | Determinism preserved (A_v1 vs A_v2) | ✅ 1/15623 (just model name) |
| 5 | No new randomness leaks | ✅ |
| 6 | Anchor reproducibility | ✅ bit-identical roundtrip on Smith2019 osim, max 41 µm center drift across all 7 wraps |

### Fit-quality classification accuracy across iterations

| Wrap | iter3 (baseline) | iter7 | iter9 | iter10 (final) |
|---|---|---|---|---|
| Capsule_r | 90.58 % | 98.79 | 98.72 | **98.72** |
| KnExt_at_fem_r | 94.31 | 96.22 | 96.08 | **96.08** |
| KnExt_vasint_at_fem_r | 95.28 | 95.70 | 95.51 | **95.51** |
| Gastroc_at_Condyles_r | 98.00 | 97.73 | 97.73 | **97.73** |
| Med_LigP_r | 99.62 | 99.33 | 99.33 | **99.33** |
| **Med_Lig_r** | 99.37 | 97.09 | 97.09 | **99.31** (opt-out recovered) |
| PatTen_r | 95.97 | 95.97 | 95.97 | **95.97** (patella not wired) |
| **Mean** | **96.16** | 97.26 | 97.20 | **97.52** |

### Additional issues resolved

- Confirmed via spatial label-transfer diagnostic that NSM correspondence
  is *not* the upstream source of wrap-fit amplification — labels handed
  to the fitter are ≥99.99 % stable A vs B at the categorical level. The
  amplification mechanism is the wrap fitter's loss-landscape sensitivity
  in flat directions, which iters 1–10 directly address.
- Surface ASSD A vs B is ~10 µm (not 1e-4 mm as iter1 NOTES had claimed).
  The original number may have been pre-ACVD raw decoder output, or an
  incorrect metric.
- iter10's per-wrap opt-out config (`wraps_to_skip_anchor=['Med_Lig_r']`)
  is overridable via the `config` dict in `build_joint_model`, so users
  can change the policy without touching nsosim source.

### Challenges / design decisions

1. **Procrustes anchor uses identity bone_transform.** Smith2019 and subject
   bones share the OSIM frame by construction (subject NSM-aligned to
   Smith2019 reference), so the anchor's wrap-frame is the same frame as
   the subject's wrap-frame. A more sophisticated per-vertex bone-shape
   correction was considered but the identity case suffices — see
   `procrustes_anchor.py` docstring for the design rationale and the
   `bone_transform` kwarg for future extension.
2. **Med_Lig_r multi-minima problem.** Identified empirically in the iter8.5
   λ + init sweep (`scratch/iter8/lambda_sweep_all_wraps.py`). The loss
   landscape for this wrap has two local minima ~3 mm apart in center-
   space; algebraic init finds the 99.4 %-accuracy basin, Procrustes-from-
   Smith2019 init falls into a 97 %-accuracy basin. No regularization
   strength can bridge this — it's an init choice. Per-wrap opt-out is
   the cleanest fix.
3. **PatellaFitter (PatTen_r) intentionally not wired.** The patella uses
   a specialized fitter; PatTen_r is already bit-stable A vs B in iter3
   and earlier. Wiring it would require extending `PatellaFitter`'s
   interface separately; not worth the API churn for a wrap that's
   already passing all criteria.
4. **Cylinder length excluded from acceptance criteria.** Plan rev 2
   explicitly notes cylinder length is biomechanically irrelevant (the
   muscle path wraps only on the lateral surface near the bone). KnExt
   cylinders end up at 290 mm length vs Smith2019's 185 mm — pre-existing
   pre-iter1 behavior, not a regression.

### Things to note for future work

- **Multi-subject Level 3 validation deferred** — handed off to
  `comak_gait_simulation/.claude/plans/NSOSIM_WRAP_FITTER_MIGRATION.md`.
  Runs on 5–10 subjects across the COMAK-runtime difficulty distribution
  to confirm the per-wrap opt-out default (`['Med_Lig_r']`) generalizes
  beyond subject 9018389.
- **Multi-start with deterministic perturbations** (revised plan §2) and
  **TSDF refinement** (§4) were both NOT needed to meet acceptance.
  Available in `_fit_with_restarts` for the multi-start case; TSDF was
  never implemented. Keep these as backup ideas if a Level 3 subject
  reveals a regression.
- **`create_ellipsoid_polydata` / `create_cylinder_polydata` rotation
  convention bug** (extrinsic-XYZ vs intrinsic-XYZ for large rotations).
  Identified during iter4 verification; out of scope for this plan but
  affects mesh visualization for OpenSim-large-rotation wraps. Track
  separately.
- **Worktree branch lifecycle**: `worktree-wrap-fitter-robustness` is
  18 commits ahead of `main` at the time of plan completion. Merge plan
  is in `NSOSIM_WRAP_FITTER_MIGRATION.md`; may need conflict resolution
  in `nsosim/model_building.py` and `tests/test_rotation_utils.py` due
  to overlap with the NSM-determinism main-side commits.
