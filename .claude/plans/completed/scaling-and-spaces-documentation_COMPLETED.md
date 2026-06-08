# Plan: Document the Scale/Space Behaviour of the Knee-Build Pipeline (then prepare the `s_wa` fix)

**Status:** Complete (2026-06-08) — Stages 0–4 (docstrings + the mkdocs site, `docs/`,
`make docs`) shipped and reviewed across several rounds. Stage 5 (the actual `s_wa` fix) is its
own plan, [`knee-scaling-fix_COMPLETED.md`](knee-scaling-fix_COMPLETED.md), also done & verified
on the nsosim side: no core code change was needed — `scale_comak_model` already performs "build,
then scale" when given a built model as `base_osim`; verified end-to-end on a real built model and
covered by `tests/scaling/test_build_then_scale.py`. Modes 3 & 5 in `docs/deviations.md` are
updated from "bug/partial" to "works". Remaining work is external (Pathway-B production wiring in
the comak repo), tracked in [`backlog.md`](../backlog.md). See **Completion Notes** and the **Handoff
state & docs↔code trust** section before continuing.
**Created:** 2026-06-06
**Owner:** nsosim (this is where the scale/space logic lives; the comak repo only wires it).

**Parent context:**
- Investigation brief: `comak_gait_simulation/.claude/plans/NSM_CACHE_INVESTIGATION.md`
- Raw investigation evidence (agent scratch — the corrected synthesis is in §"Corrected findings" below; the critique is the authoritative correction): `comak_gait_simulation/.claude/reports/scaling_map/` (`walk_1..4_*.md` + `consensus_critique.md`)
- Stage X spec (authoritative for body scaling): [`comak-body-scaling_COMPLETED.md`](comak-body-scaling_COMPLETED.md)
- Stage Y / knee-assembly spec: [`knee-assembly.md`](../knee-assembly.md)

---

## Completion Notes

**Date completed:** 2026-06-08

**Summary.** The scale/space behaviour of the knee-build pipeline is now documented accurately and
in sync with the code: a four-space model (MRI / REFALIGN / NSMcanon / OSIM), the full MRI→model
transform chain, the COMAK body-scaling interaction, and a 5-mode "knee sizing" catalog. Shipped
as a mkdocs + mkdocstrings site under `docs/` with a stdlib guard test that fails if a documented
symbol is renamed. Stage 5 (the actual fix) shipped separately and is verified — see
[`knee-scaling-fix_COMPLETED.md`](knee-scaling-fix_COMPLETED.md).

**Changes made (vs. the original plan — layout deviated, see "Handoff state" below).**
- Docstrings across the scale/space call chain (`nsm_fitting`, `model_building`, `decode`,
  `osim_utils`, `comak_osim_update`, `scaling/`).
- `docs/coordinate-systems.md` (was `SCALING_AND_SPACES.md`), `docs/deviations.md` (the 5-mode
  catalog, was `SCALING_DEVIATIONS.md`), `docs/index.md` production-driver table.
- `tests/docs/test_doc_references.py` (symbol-reference guard); `tests/test_transform_chain.py`
  converter-scale-arg guards.
- Two code changes beyond docs: `rigid_reg_type` default → `'similarity'`; "Stage X" → "COMAK
  body scaling" in the scaling-module docstrings.

**Tests.** `tests/docs/test_doc_references.py` (27), the converter-scale-arg guards in
`tests/test_transform_chain.py`, and `tests/scaling/` all pass. (The doc-reference guard's
full-run encoding fragility was fixed under the Stage-5 plan, 2026-06-08.)

**Things to note for future work.**
- The deviations page deliberately includes **proposed/unbuilt** fix designs for Modes 3–5,
  clearly labeled — an intentional scope widening (owner-requested) beyond "document current
  behaviour only."
- Remaining work is external (comak repo) and is tracked in [`backlog.md`](../backlog.md): correct or
  delete the stale `SCALING_WORKFLOW_MAP.md`, and reorder Pathway-B to build-then-scale.

---

## Why this plan exists

A cross-repo investigation established that **the knee surface COMAK computes contact pressure on is reference-scale, not the subject's true anatomical size** — and in Pathway B (multigait), it is a reference-scale knee sitting inside an `s_wa`-scaled body (a geometric mismatch). Before we change any code, we want the scale/space behaviour of the knee-build pipeline **documented accurately** so the eventual fix is safe and so future readers (human or agent) can infer it without re-deriving it from a five-step transform chain.

**Framing (important, drives the whole approach):** the **library is a capable tool that supports the behaviours we want** — both "scale the knee to the gait body" and "keep the knee at the subject's true MRI size." The *pipeline wiring* currently lands in neither cleanly for Pathway B. So:

1. **Docstrings describe what the code actually does**, accurately, including the coordinate **space + units + scale identity** of every input/output. They do **not** editorialize "intended vs actual."
2. **Every deviation** (places where the pipeline uses the library in a way that produces the reference-scale-knee-in-scaled-body result) goes in **one separate document**, not sprinkled across docstrings.
3. The cross-repo map is a **reference/cross-check only**; docstrings are written from the actual code (the source of truth), correcting the map where they disagree (the map has at least one confirmed error — see below).

---

## Corrected findings (ground truth — embedded so this plan is self-contained)

These are the verified conclusions from 4 independent code walks + an adversarial critic that re-derived the high-stakes claims from code **and** measured a real built model. Use these, not the uncorrected map.

### Coordinate spaces

| Space | Definition | Units | Scale identity |
|---|---|---|---|
| **MRI** | subject segmentation mesh — the subject's TRUE anatomical size | mm | subject-physical |
| **REFALIGN** | subject mesh after **similarity**-registration onto the fixed smith2019 reference bone | mm | **reference size** (subject true scale divided out) |
| **NSMcanon** | NSM training-normalized box ~[-1,1] | dimensionless | per-bone canonical |
| **OSIM** | OpenSim body-local frame after `convert_nsm_recon_to_OSIM_`: + fixed ref-center, mm→m, axis-swap | m | reference size, rotated |

### The three confirmed mechanisms (file:line in nsosim unless noted)

- **H1 — subject true size is removed at the femur similarity registration.** `align_bone_osim_fit_nsm` registers the femur with `reg_mode='similarity'` (rigid + **isotropic scale**) onto the fixed `smith2019-R-femur-bone_processed.vtk`; tibia/patella **reuse the femur transform** (`nsm_fitting.py:122-131,137-138,374`). Default `rigid_reg_type='similarity'` flows from `comak_1_nsm_fitting.py:313`. **Empirically:** raw MRI femur diagonal CoV across subjects = 7.9%; after registration = ~1.0% (size collapsed to reference). The recon returns in REFALIGN because `create_mesh_adaptive` (`NSM/mesh/main.py:133-146`) undoes only the NSM-normalization, never the subject-size division.

- **H2 — subject scale is never restored going to OSIM.** The MRI path's `nsm_recon_to_osim` → `_nsm_recon_to_osim_single_surface` calls **only** the underscore `convert_nsm_recon_to_OSIM_` (`nsm_fitting.py:892-902,724-755`): `+= fem_ref_center`, `/= 1000`, axis-swap. No `undo_transform`, no subject-scale divide. The **same fixed** `fem_ref_center` (`ref_femur_alignment.json['mean_orig']`) is used for all three bones (`comak_1_nsm_fitting.py:324-355`).
  - **Note the latent capability:** the *non-underscore* `convert_nsm_recon_to_OSIM` (`nsm_fitting.py:813`) **does** take a subject scale/center term and would restore true size — it is simply not on the MRI path (the synthetic path uses it, with `scale=1, center=0`). This is the hook the "keep-true-size" mode (Pathway C) would use.

- **H3 outcome — reference-scale knee inside an `s_wa`-scaled body (Pathway B only).** Confirmed + measured on `OARSI_multigait_RSubject_121_…/9018389_00m_RIGHT`: `smith2019-R-femur-bone.stl` baked to exactly **0.97298×** (= `wa_scale`) but **orphaned**; the body references `femur_nsm_recon_osim.stl` at `scale_factors = 1 1 1` (reference scale).

### Two corrections to the cross-repo map (the map is WRONG on these — do not propagate)

1. **Mechanism is repoint+orphan, NOT overwrite-by-filename.** The map says `save_geometry_files` overwrites the Stage-X-baked knee STLs by filename collision. **False.** The recon STLs have **different filenames** (`femur_nsm_recon_osim.stl`) than the baked ones (`smith2019-R-femur-bone.stl`). `save_geometry_files` (`model_building.py:749-773`) copies **new** files; `update_*` repoints the model via `set_mesh_file` (`osim_utils.py:55,131-133`; dicts `comak_osim_update.py:15-62`); the baked STLs are left **orphaned** on disk. Net outcome identical, mechanism different — and the difference matters for the fix (you scale the recon that gets *repointed to*, and the Stage-X bone bake is dead weight under Pathway B).

2. **Pathway B is wired via Step 1, not `comak_2_pathway_b.py`.** It is: `multigait/prepare_gait_subject.py` runs Stage X (`scale_comak_model`), writes a **Stage-Y config** whose `paths.base_model_dir` = the Stage-X output dir, and launches the **NSM queue runner with `--config stage_y.json`** (`comak_1_nsm_model_run.py:62` → `comak_1_nsm_fitting.py:217-221`). `comak_2_pathway_b.py` is a Step-2 COMAK driver and irrelevant to building the mismatched knee. (The map/walk-2 claim "the queue runner cannot reach Pathway B" is false — *with the config* it is THE Pathway-B vehicle.)

3. **`s_wa` formula:** exactly `(ab_factors['femur_r'][2] + ab_factors['tibia_r'][2]) / 2` (`scaling/scale_factors.py:82`) — femur + tibia long-axis only, isotropic.

### The two intended modes (from `comak-body-scaling_COMPLETED.md`)

| Mode | Meaning | Knee size | Status in library |
|---|---|---|---|
| **Pathway B** (multigait) | OAI knee simulated under a Tian gait; body scaled to Tian subject by `s_wa` | should be `s_wa` × reference, to match the body | **gap**: knee currently stays reference-scale (the H3 mismatch) |
| **Pathway C** (MRI+gait matched) | the MRI subject *is* the gait subject; keep their native knee | native MRI true size (undo the similarity scale) | **not built**: plan calls for a `scale_knee_bodies=False` / true-size restore (the H2 latent hook) |

Stage X already documents the contract that Stage Y runs "**with no awareness of AB scaling**" (`comak-body-scaling` line 13) — which is precisely why the swapped-in knee does not inherit `s_wa`. The fix is to make the knee-build **scaling-aware** (Pathway B) and/or **true-size-aware** (Pathway C), both of which the plan already anticipated as deferred work.

---

## Consumer scripts — how the library is exercised (intended use), and where the known errors live

These entry points drive the library functions in the inventory below. They are the **worked examples of intended use** — the documentation pass must cross-link them so a reader sees how the pieces compose and which behaviour each demonstrates. All in `comak_gait_simulation/run_simulations/scripts/` unless noted.

| Consumer | Demonstrates | Mode / KNOWN ERROR |
|---|---|---|
| `comak_1_nsm_fitting.py` | canonical MRI→model build: full `align → recon → assemble` chain | reference-scale knee (subject size normalized). Pathway A when base = reference. |
| `comak_1_nsm_model_run.py` + `submit_nsm_slurm_job.py` | queue/batch driver; **with `--config stage_y.json` this is the Pathway-B vehicle** | **KNOWN ERROR: Pathway-B knee is reference-scale inside an `s_wa`-scaled body** |
| `multigait/prepare_gait_subject.py` | runs Stage X, writes the Stage-Y config (`paths.base_model_dir` = Stage-X dir), launches the runner | the real Pathway-B wiring — where `s_wa` is computed and where the knee *should but doesn't* get scaled |
| `comak_1_nsm_model_test_single_subject.py` | single-subject harness | Pathway A (reference base) |
| `comak_1_synthetic.py` + `nsosim.decode` | builds a model from latents (no MRI); uses the **non-underscore** `convert_nsm_recon_to_OSIM` with `scale=1` | demonstrates the **true-size restore hook** (#8) that a keep-true-size mode would use — existing code |
| `nsosim/tests/scaling/` | Stage X unit/integration tests (identity, non-trivial `s_wa`, cart–bone coherence) | intended use of `scale_comak_model` |
| `nsosim/tests/test_knee_assembly*.py` | knee strip/add round-trip | intended use of the assembly machinery |
| `comak_2_pathway_b.py` | **Step-2 COMAK driver** (consumes the already-built model) — NOT where the knee is built | listed to disambiguate: it is *not* the Pathway-B knee-build wiring |

**Known errors to capture in `SCALING_DEVIATIONS.md`** (seeded in the §"Seed deviations" list): the reference-size normalization, the Pathway-B `s_wa` knee gap, the reference-scale `mean_patella` offset, and the orphaned Stage-X knee bake. Each docstring touching these functions should be accurate about current behaviour; the *error framing* lives only in the deviations doc.

---

## Stages

### Stage 0 — Lock the ground truth (cheap)
- Treat the "Corrected findings" section above as canonical. The comak `SCALING_WORKFLOW_MAP.md` is stale/partly-wrong — do not write docstrings from its uncorrected mechanism text. (Cleanup, when next in that repo: correct it in place, replace its body with a one-line pointer to `nsosim/docs/coordinate-systems.md` + `deviations.md`, or delete it if fully superseded — **no banner**, per owner preference 2026-06-07.)

### Stage 1 — Close the two unverified gaps (do before documenting)
1. **Read `scaling/scaletool.py`** end-to-end: confirm what `ScaleTool.run()` does to the **knee joint frames / inertia / weld translations** before `bake_knee_geometry` runs. (Neither the walks nor the critic opened it — it's the one "assumed, not read" item.)
2. **Confirm the common-center coincidence:** Stage X scales body STLs by `point_coords *= s_wa` about the **body-local origin** with no recenter (`scaling/knee_geometry.py:95-101`); the recon lands in OSIM biased by `+ fem_ref_center` then axis-swapped (`nsm_fitting.py:748-753`). Verify (empirically if needed) that the recon's OSIM frame origin coincides with the body-local origin Stage X scales about — or compute the offset. This determines the center any future `s_wa` knee scaling must use.

### Stage 2 — Docstring pass over the scale/space call chain
Document, accurately and from the code, each function below. Every docstring must state, for inputs and outputs: **coordinate space, units, and scale identity** (per the spaces table). No "intended vs actual" prose — just what it does. Scope is **this call chain only** (not a blanket re-doc of the repos).

Function inventory (file:line are current anchors; verify and update):

| # | Function | File | Role to capture |
|---|---|---|---|
| 1 | `align_knee_osim_fit_nsm` | `nsm_fitting.py:295` | orchestrates per-bone align+fit; femur transform reused for tibia/patella |
| 2 | `align_bone_osim_fit_nsm` | `nsm_fitting.py:~150` | **the similarity registration that removes subject size (H1)** |
| 3 | `fit_nsm` (NSM optimize) | `NSM/reconstruct/main.py` | latent fit in NSMcanon |
| 4 | `create_mesh_adaptive` | `NSM/mesh/main.py:133-146` | undoes NSM-normalization only (not subject size) |
| 5 | `nsm_recon_to_osim` | `nsm_fitting.py:905` | per-surface REFALIGN→OSIM dispatcher |
| 6 | `_nsm_recon_to_osim_single_surface` | `nsm_fitting.py:892` | calls the underscore converter |
| 7 | `convert_nsm_recon_to_OSIM_` (underscore) | `nsm_fitting.py:724` | **+ref-center, mm→m, axis-swap; NO subject-scale (H2)** |
| 8 | `convert_nsm_recon_to_OSIM` (non-underscore) | `nsm_fitting.py:813` | **the latent true-size restore hook (Pathway C)** |
| 9 | `convert_OSIM_to_nsm(_)` | `nsm_fitting.py:758,852` | the reverse round-trips |
| 10 | `undo_transform` / `apply_transform` | `nsm_fitting.py:682,708` | the scale/center/icp primitives |
| 11 | `decode_latent_to_osim` / `decode_joint_from_descriptors` | `decode.py:30-109` | synthetic path (uses #8 with scale=1) |
| 12 | `build_joint_model` | `model_building.py:~940` | assembler: copytree base + write recon + finalize |
| 13 | `save_geometry_files` | `model_building.py:749-773` | **copy-new-file (repoint, not overwrite) — correct the wording here** |
| 14 | `center_patella_meshes` + patella offset | `model_building.py:702-708`; `comak_osim_update.py:197-204` | **`mean_patella` is a reference-scale joint translation** |
| 15 | `update_osim_model` / `set_mesh_file` | `osim_utils.py:55,131-133`; `comak_osim_update.py:15-62` | repoint mechanism; `scale_factors` left at 1,1,1 |
| 16 | `scale_comak_model` | `scaling/orchestrator.py:150` | Stage X pipeline order |
| 17 | `build_scale_set` (`s_wa`) | `scaling/scale_factors.py:82` | `s_wa=(femur_r.z+tibia_r.z)/2` |
| 18 | `bake_knee_geometry` | `scaling/knee_geometry.py:29` | bakes knee STLs by `s_wa` about origin; **note orphaning under Pathway B** |
| 19 | `apply_scaletool` | `scaling/scaletool.py` | Stage 1 gap — document after reading |
| 20 | ligament/wrap/fatpad builders | `model_building.py` (articular, meniscus surfaces, wraps, fat pad) | note they build FROM the OSIM recon (would inherit a recon scale) |

### Stage 3 — Two documents
1. **`nsosim/docs/SCALING_AND_SPACES.md`** (authoritative, in-library): the spaces table, the full transform chain MRI→final model with file:line, how Stage X (`s_wa`) and the knee-build interact. Pure description of **current behaviour + capabilities that exist in the code today** — including the true-size restore hook (#8, exercised by the synthetic path). **Do NOT document unbuilt machinery** (e.g. a `scale_knee_bodies=False` Pathway-C flag, or the choice-B `s_wa` injection) — those are new code and belong in the Stage-5 fix plan, not the library docs. Add a pointer from the comak repo.
2. **`nsosim/docs/SCALING_DEVIATIONS.md`** (the single deviations list): every place the pipeline wiring produces the reference-scale-knee-in-`s_wa`-body result, framed as "pipeline uses library capability X in mode Y; to get mode Z, do W." Seed it with the deviations below.

### Stage 4 — Re-validate
Re-run a lighter version of the walk exercise against the **now-documented** code: confirm the docstrings + `SCALING_AND_SPACES.md` are correct and complete, nothing missed. This is the "revisit the mapping with the mapping" step.

### Stage 5 — Only then: design the fix (separate plan)
Design choice B (and/or the Pathway C true-size mode). Pre-identified injection requirements (do NOT implement here):
- Apply `s_wa` to the knee recon about the common center at the OSIM-entry point (#5/#7), **and** to the warp-path attachment converter, **and** to the labeled mesh fed to the wrap fitter, **and** to `mean_patella` (the one load-bearing scalar — #14).
- Decide the fate of the Stage-X bone/cart/contact STL bake under Pathway B (orphaned → dead weight; keep only if needed elsewhere).
- Thread the synthetic path (#11) separately if the feature is meant to be general.
- Keep mass/inertia owned by Stage X (`rescale_subject_mass.py` + orchestrator two-pass) — choice B touches geometry only.

---

## Seed deviations (for `SCALING_DEVIATIONS.md`)

1. **MRI knee size is normalized to the reference.** The similarity registration (#2) removes the subject's true anatomical scale; it is never restored on the MRI path (#7). All subjects' knees are ~reference size (shape-personalized, size-normalized). *Library capability:* the non-underscore converter (#8) can restore true size; the MRI script does not call it.
2. **Pathway B knee does not inherit `s_wa`.** Stage X bakes the reference knee by `s_wa` (#18), but the knee-build repoints the model to the reference-scale recon (#13/#15) and never applies `s_wa` to it → reference-scale knee in an `s_wa`-scaled body. *Library capability:* `s_wa` is on disk in the scaling report; applying it to the recon at the OSIM-entry point is the missing wiring.
3. **`mean_patella` joint translation is reference-scale (#14).** Even if a future fix scales the meshes, this scalar must be scaled too or PF kinematics drift.
4. **Stage-X knee STL bake is orphaned under Pathway B (#13/#18).** The baked `smith2019-R-*.stl` are not referenced by the built model; only non-knee Geometry retains Stage-X scaling.

---

## Decisions (resolved 2026-06-06)

1. **Doc home:** `nsosim/docs/SCALING_AND_SPACES.md` + `SCALING_DEVIATIONS.md`. ✅ confirmed.
2. **Pathway C / future modes:** document only what **exists in the code today** (the true-size restore hook #8). Do **not** write library docs for unbuilt code (a `scale_knee_bodies` flag, the choice-B injection) — that is future stuff for the Stage-5 fix plan. ✅
3. **Stage 2 execution:** **fan out** agents to draft docstrings per-module against a strict style contract (space/units/scale identity), gated by one consolidation/review pass for consistency. ✅
4. **comak-side map:** ~~keep a correction banner on `SCALING_WORKFLOW_MAP.md`~~ — **superseded 2026-06-07 (owner: no banners).** When next in the comak repo, correct `SCALING_WORKFLOW_MAP.md` in place, replace its body with a one-line pointer to the nsosim docs, or delete it if fully superseded. Do not add a banner.

## Open items

- Whether the Stage-2 fan-out is run from the comak session (agents editing nsosim files) or by a dedicated agent inside the nsosim repo — owner's choice.

---

## Post-documentation review (AI assessment, 2026-06-07)

A second AI did an independent assessment of the published docs against the code (39 focused
tests passed; no files modified). Its findings, what was done, and what remains for the Stage-5
fix plan. **The first item is the most important — it changes the fix design.**

### CORRECTED — the converter `scale` argument is NOT the sizing lever (verified)
Earlier doc drafts called the `scale` arg of the non-underscore
`convert_nsm_recon_to_OSIM` the "natural/preferred lever" for resizing the knee. **That is
wrong.** `scale` is applied to the points in **NSMcanon space, before** the per-bone inverse
transform and the `+ fem_ref_center` shift, so it is the NSM canonical-normalization scale —
not the femur similarity-registration scale, and not a clean OSIM resize.

Verified numerically (real subject `tibia_alignment.json`): `convert_nsm_recon_to_OSIM(scale=2)`
does **not** give `2×` the OSIM points. `out2 − 2·out1` is a constant offset
(≈ `[−0.0004, +0.041, −0.001]` m), i.e. it scales about an **affine center ~4 cm from the
joint-center origin** that **differs per bone** — so applying the same `scale` to the three
bones would scale each about a different point and distort the joint.

**The clean lever (use this in the fix):** a plain OSIM-space multiply about the shared
joint-center origin (`osim_points *= s_wa`), the same operation `bake_knee_geometry` uses —
NOT the converter's `scale` argument.

**Where to apply it — there are TWO independent geometry generators, scale BOTH** (an earlier
draft wrongly said "scale the recon once and everything follows"; the owner caught it, code
re-checked). Confirmed by reading `build_joint_model`:
1. **Recon meshes** (`nsm_recon_to_osim`) → `create_articular_surfaces`, meniscus surfaces, fat
   pad, and `mean_patella` (`center_patella_meshes` takes the recon patella mesh). Scaling the
   recon meshes covers all of these.
2. **Reference→subject warp** (`interpolate_bone_ligaments` + the meniscus-ligament warp) →
   produces the warped *labeled* mesh that `fit_bone_wrap_surfaces` fits to, **and** the
   ligament/muscle attachment positions. This path does NOT read the recon mesh (it warps the
   reference labeled mesh through the NSM and converts via the non-underscore converter), so
   scaling the recon does nothing for wraps or ligaments. Scale this generator's OSIM output
   too.

So the original Stage-5 enumeration ("recon AND warp-path attachments AND the labeled mesh fed
to the wrap fitter AND `mean_patella`") was **correct about the number of places** — the only
correction is the *lever* (OSIM-space multiply about the joint origin, not the converter
`scale` arg). `mean_patella` rides along with generator 1; wraps + ligaments ride along with
generator 2. Docs corrected: `coordinate-systems.md` §3, §4; `deviations.md` Mode 3/4/5 + Notes
+ future-fix list.

### RESOLVED — origin-scaling regression test now exists (2026-06-08)
The docs assert recon and reference geometry share the joint-center origin "by construction"
and that scaling about OSIM `(0,0,0)` preserves placement. Stage-1 measured the origin offsets
empirically (femur 24.8 mm, tibia 54 mm from origin; the origin is the joint center, not the
centroid). This is now covered by a **permanent regression test**:
`tests/scaling/test_build_then_scale.py::TestOriginScalingPlacement` (every knee body's centroid
scales by `s_wa` on the real built model) and `::TestReferenceOriginScalingPlacement` (the same
on the in-repo reference base with a synthetic `s_wa = 0.9`, so it runs without the large fixture).

### RESOLVED in the docs (no further action)
- **REFALIGN frame vs scale identity** — the spaces table now defines REFALIGN as a *frame*,
  with scale identity listed as `reg_mode`-dependent (reference size with `'similarity'`,
  subject size with `'rigid'`). `coordinate-systems.md` §1.
- **`build_joint_model` over-constrained to reference-size** — docstring now says the assembler
  is size-agnostic; reference size is a property of *current callers*, not the function.
- **"Most common" (Mode 1)** — labeled as a workflow observation, not a code-enforced fact.
- **Consumer-script orchestration** — `index.md` now has a "How the library is driven in
  production" table naming `comak_1_nsm_fitting.py`, `comak_1_nsm_model_run.py` +
  `submit_nsm_slurm_job.py`, `multigait/prepare_gait_subject.py`, `comak_1_synthetic.py`, each
  mapped to a sizing mode. (These live in the external `comak_gait_simulation` repo.)
- **Production overrides** — `index.md` notes that production may override library defaults
  (example: `build_joint_model(..., project_coronary=False)`); trace the real call site, not the
  nsosim default, when reasoning about production behavior.

### ALIGNED — docs now lead with "build, then scale" (2026-06-07, pre-fix review)
A pre-fix review (against code) confirmed two facts the docs had under-stated, and aligned the
docs with the chosen fix direction in [`knee-scaling-fix_COMPLETED.md`](knee-scaling-fix_COMPLETED.md):
1. **`bake_knee_geometry` bakes whatever STL is *attached* to the knee bodies** (reads
   `mesh.get_mesh_file()` off the model), not fixed reference filenames. The reference STLs are
   baked today only because of pipeline **order** (body-scale the reference base → then build).
   Run the bake on a *built* model and it bakes the recon. → `coordinate-systems.md` §5.
2. **During body scaling, two mechanisms scale the knee:** the **mesh surfaces** via custom
   `bake_knee_geometry` (JAM ignores `scale_factors`), and the **wraps / ligaments (slack
   lengths) / frames / muscles** via OpenSim ScaleTool `extendPostScale`. Slack-length scaling
   is ScaleTool's job during body scaling (the build-side `update_slack_lengths` is a *separate*
   Stage-5 path) and is already verified by
   `tests/scaling/test_nontrivial.py::test_blankevoort_reference_strain_preserved` (reference
   strain preserved ⇒ slacks scaled). → `coordinate-systems.md` §5 Mode-2 success box.
These two facts are why the fix plan's **build-then-scale** approach works and "dissolves the
two-generator problem." `deviations.md` Mode 3 / Notes / "For the future fix plan" now lead with
build-then-scale and demote mid-build interception to the alternative. Doc-reference test: 27
passing.

Also **locked the "single most important correction" with a permanent guard** (it was only
ad-hoc-verified before): `TestConverterScaleArgIsNotCleanResize` +
`test_converter_scale_offset_differs_per_bone` in `tests/test_transform_chain.py` (JSON-only,
10 tests) assert `convert_nsm_recon_to_OSIM(scale=2) != 2×`, that the discrepancy is a constant
per-bone offset matching the closed form `-(b+rc)/1000 @ OSIM_TO_NSM_TRANSFORM.T`, and that the
offset differs per bone. `s_wa` isotropy for knee bodies was already locked by
`tests/scaling/test_scale_factors.py::test_wa_factor_matches_worked_example`.

### SCOPE DECISION — deviations.md now carries proposed fix designs (deliberate)
The original plan boundary was "document current behavior + existing capabilities only." The
owner subsequently asked for the deviations page to be reframed as a **knee-sizing modes
catalog** that includes how each mode *would* be built. So `deviations.md` now contains
proposed (unbuilt) fix designs for Modes 3–5, clearly labeled "not implemented / design sketch"
(yellow admonitions). This is an intentional widening of scope, not an accuracy slip — the
unbuilt content is marked as such and separated from the verified current-behavior description.

---

## Handoff state & docs↔code trust (2026-06-07)

**For the next agent.** This documentation effort is complete and the docs are safe to build
the scaling fix on, *with the trust ratings below*. The fix itself is scoped in
[`knee-scaling-fix_COMPLETED.md`](knee-scaling-fix_COMPLETED.md).

### What was produced (vs. the original plan)
- **Stage 1** (gaps): closed empirically — ScaleTool's effect on knee welds/inertia/mass and
  the joint-center origin were measured (see "Corrected findings" + the docs).
- **Stage 2** (docstrings): done across the scale/space call chain (`nsm_fitting`,
  `model_building`, `decode`, `osim_utils`, `comak_osim_update`, `scaling/`). RST `:func:`
  markers were later stripped (mkdocstrings is Markdown).
- **Stage 3** (the two docs): shipped, but **the layout deviated from the plan**:
  - Docs moved from `nsosim/docs/` to repo-root `docs/` and became a **mkdocs + mkdocstrings**
    site (build via `make docs` / `make docs-serve`; griffe static analysis, no import of
    nsosim). `SCALING_AND_SPACES.md` → `docs/coordinate-systems.md`;
    `SCALING_DEVIATIONS.md` → `docs/deviations.md` (retitled "Knee Sizing — Modes & Status").
  - Cross-references are mkdocstrings **symbol** links, not line numbers; a stdlib pytest
    (`tests/docs/test_doc_references.py`) fails if a referenced symbol is renamed/removed.
  - `deviations.md` was **reframed** from a flat deviation list into a **5-mode catalog**
    (owner-requested) that includes unbuilt fix designs, clearly labeled.
- **Stage 4** (re-validate): done; plus three external review rounds (owner + an independent
  AI) folded in. The "Post-documentation review" section above is the running record.
- **Two code changes** were made beyond docs (owner-approved): `reg_mode` default flipped to
  `'similarity'`; "Stage X" renamed to "COMAK body scaling" in the scaling-module docstrings.

### Docs↔code TRUST — can a new agent rely on the docs for the scaling fix?

**Trust (verified against code/data; safe to build on):**
- The full MRI→OSIM transform chain, the four spaces, `OSIM_TO_NSM_TRANSFORM`
  (covered by `tests/test_transform_chain.py`).
- The `s_wa` formula and ScaleTool's knee behavior (weld scaled by parent AB factors; mass
  unchanged by ScaleTool, set by the orchestrator two-pass; inertia mass-ratio-scaled) —
  measured in Stage 1.
- **The converter `scale` argument is NOT a clean OSIM resize** — verified numerically
  (`scale=2` ≠ 2×). The clean lever is a plain OSIM-space multiply about the joint-center
  origin. **This is the single most important correction; build the fix on it.**
- **Two geometry generators** (recon meshes vs. the reference→subject warp that feeds wraps +
  ligaments) — verified by reading `build_joint_model`. A fix must scale **both**.
- Consumer wiring (verified in `comak_gait_simulation`): `comak_1_nsm_fitting.py` is the build;
  `prepare_gait_subject.py` runs Stage X + writes the Stage-Y config; `submit_stage_y.sh`
  launches the Pathway-B build; `project_coronary=False` at `comak_1_nsm_fitting.py:459`
  (owner-confirmed: `True` makes menisci too taut).

**Now verified (updated 2026-06-08 — Stage 5 shipped):**
- Modes 3 & 5 ("build, then scale") are **built and tested** end-to-end on a real built model —
  `tests/scaling/test_build_then_scale.py` (per-component scaling, origin placement, inertia,
  coherence). Patella centering × scaling and the meniscus-ligament path are verified there too.
- Scaling about OSIM `(0,0,0)` preserving placement for *every* knee body now has a permanent
  regression test (`TestOriginScalingPlacement` / `TestReferenceOriginScalingPlacement`).

**Use with care (still design sketches / not yet verified — do NOT treat as authoritative):**
- **Mode 4** (keep native MRI true size) is **not implemented** — its "how to fix it" content is
  a proposed design only.
- The "register to a pre-scaled reference gives the same result" alternative is unverified
  (the contingent fit-on-scaled-base cross-check — see the backlog).

**Bottom line:** the *current-behavior* description and the corrected scaling mechanism are
trustworthy; the *future-fix* designs are clearly-labeled starting points, not settled.
