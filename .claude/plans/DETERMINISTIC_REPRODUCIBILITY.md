# Make NSM Fitting & Decode Deterministic

**Status:** Tasks 1, 2, 4 implemented (2026-05-10). Tasks 3, 5, 6 still pending — they live in `comak_gait_simulation` and need a real subject run to verify.
**Driver:** Verification work in `comak_gait_simulation` traced cascading Step 2 biomechanical variance (5–15% NRMSE on contact pressures, ~3° on PF flexion, 28-39% COMAK convergence flips) back to NSM-fitting stochasticity. Two independent runs of the same code on the same subject produce different geometry → different contact mechanics → different optimizer trajectories.

**Goal:** Given identical inputs (subject mesh + model weights + config), produce bit-identical outputs across runs. This unlocks pipeline verification with tight tolerances and reproducibility for paper-grade results.

## Implementation status (2026-05-10)

| Task | Status | Notes |
|---|---|---|
| 1. `set_global_seed()` utility | ✅ Done | [`nsosim/_determinism.py`](../../nsosim/_determinism.py) |
| 2. Wire seed into entry points | ✅ Done | `seed=0` default on `fit_nsm`, `align_knee_osim_fit_nsm`, `align_bone_osim_fit_nsm`, `nsm_recon_to_osim`, `decode_latent_to_osim`, `decode_joint_from_descriptors`, `build_joint_model` |
| 3. Confirm meniscus boundary stability | ⚠️ Partial | Unit-tested on decoded reference meshes — bit-identical (medial + lateral). Needs verification on real subject geometry via Task 6. |
| 4. Bit-identical determinism test | ✅ Done | [`tests/test_determinism.py`](../../tests/test_determinism.py) — 12 tests covering decode + post-processing (articular surfaces, both menisci, fat pad). All bit-identical on TITAN Xp. Slurm script: [`scripts/determinism_verification/run_determinism_test.sbatch`](../../scripts/determinism_verification/run_determinism_test.sbatch) |
| 5. Update `comak_gait_simulation` to pass seed | ⚠️ Optional | Default seed=0 is wired through, so two consecutive comak_1 runs are already deterministic without code changes there. An explicit `--seed` flag is still nice-to-have for reproducibility audits. |
| 6. Re-run verification | ⏳ Pending | Integration script ready: [`comak_gait_simulation/tests/verify_determinism/submit_determinism_check.sh`](file:///dataNAS/people/aagatti/projects/comak_gait_simulation/tests/verify_determinism/submit_determinism_check.sh). Submit when ready. |

**What we know now (post-Task 4):**
- Decode → mesh: bit-identical
- ACVD resample: bit-identical (when input is)
- Articular surface extraction: bit-identical
- Meniscus articulating surface (medial + lateral): bit-identical — closes the persistent failure flagged below
- Prefemoral fat pad: bit-identical

**What remains unverified until Task 6:**
- `fit_nsm` Adam/LBFGS path on a real subject mesh (the unit tests don't have raw subject input fixtures, only post-fit outputs)
- Wrap surface fitting on a real labeled bone mesh
- `.osim` XML output reproducibility
- The full Step 1 → Step 2 verification chain

## End-to-end findings (2026-05-10, jobs 46776 → 46788)

After running the integration test (`comak_gait_simulation/tests/verify_determinism/submit_determinism_check.sh`) repeatedly with progressively more fixes:

### What worked

1. **Seed AFTER `model.cuda()`** — `model.cuda()` consumes CUDA random state during weight transfer. Seeding before leaves the CUDA RNG at an unpredictable offset. Fix: in `nsosim/utils.py:fit_nsm`, call `set_global_seed(seed)` *after* `load_model(...)`. (Inspired by `kneepipeline/steps/run_nsm.py`.) Result: NSM mesh ASSDs dropped to <0.0001 mm.

2. **Wrap surface multi-start** (3 restarts, sub-micron input jitter) — helps for some wraps (Capsule_r 1.16mm → 0.001mm) but is a mixed bag overall. Best-of-3 reduces wrap fail count from ~4-5 to ~3 per run.

### What did NOT work (counterintuitively)

**Adam+LBFGS hybrid NSM optimizer**, per the `HYBRID_OPTIMIZER_REPORT.md` recipe (`test_hybrid_norm_10_3_full_dataset.json`): made reproducibility **10-30× worse** on every mesh, and 8 wrap fails vs 3-5 with plain Adam.

| Mesh | Plain Adam ASSD | Hybrid Adam+LBFGS ASSD | Latent max_abs |
|---|---|---|---|
| Femur bone | 0.000045 mm | **0.001452 mm** (32×) | 1e-4 → 6e-3 |
| Femur cart | 0.000023 mm | **0.000769 mm** (33×) | |
| Tibia bone | 0.000027 mm | **0.000296 mm** (11×) | |
| Patella bone | 0.000008 mm | **0.000085 mm** (11×) | |

**Why**: LBFGS uses a Hessian approximation built from recent gradients. With `grid_sample` backward producing non-deterministic CUDA gradients (PyTorch limitation, not nsosim), the Hessian approximation diverges between runs, sending the L-BFGS optimizer along different directions. Adam's momentum smoothing happens to dampen this gradient noise; LBFGS's curvature estimator amplifies it.

The hybrid recipe was tuned for **single-run fit quality** (best ASSD against ground truth), not **run-to-run reproducibility**. Different objectives. The hybrid plumbing stays in nsosim as an available knob (`use_hybrid_optimizer=True` on `fit_nsm` / `align_*_osim_fit_nsm`); just don't enable it for reproducibility runs.

### Wrap-fitter sensitivity remains the residual issue

Bone meshes are reproducible to <0.0001 mm but the wrap fitter (CylinderFitter, EllipsoidFitter) amplifies micron-scale input drift into mm-scale wrap surface drift, particularly for the `Gastroc_at_Condyles_r` ellipsoid (Euler angles near gimbal-lock singularity → ambiguous parameter representation of geometrically-similar surfaces). Multi-start helps some configurations and hurts others.

**Where it leaves us**: geometric meshes meet the <0.05mm ASSD bar comfortably; a handful of wrap surfaces still drift 0.3-1.5 mm. End-to-end COMAK verification (Task 6) will tell us whether this matters biomechanically. The wrap-fitter robustness is a separate workstream.

### Best combination (committed)

Plain Adam + wrap multi-start (`wrap_n_restarts=3`, `wrap_jitter_scale=1e-6`). Hybrid NSM available but not enabled by default.

## Why this matters

- **Paper reproducibility.** Anyone re-running our analysis should produce the same results. Right now they cannot.
- **Verification ceiling.** `comak_gait_simulation/tests/verify_pipeline/` currently calibrates tolerances by guessing. With determinism, tolerances become "exact match" for fixed seed and "stochasticity ceiling" for unfixed seed.
- **Eliminates downstream noise in COMAK.** Contact pressures, convergence flips, and pf_flex_r angle disagreement all dissolve if Step 1 geometry is stable.

## Known stochasticity sources (verified by grep)

No `torch.manual_seed` / `np.random.seed` / `random.seed` / cudnn-deterministic call exists anywhere in `nsosim/` or `NSM/NSM/` (excluding training and tests). Specific random calls in the inference path:

| Location | Call | What it affects |
|---|---|---|
| [`NSM/NSM/reconstruct/main.py:416-417`](file:///dataNAS/people/aagatti/programming/NSM/NSM/reconstruct/main.py#L416) | `torch.ones(...).normal_(mean, std)` | **Latent vector initialization** — primary source of fit-to-fit variance. Different starting points → different local optima found by Adam/LBFGS. |
| [`NSM/NSM/reconstruct/main.py:561,570,573`](file:///dataNAS/people/aagatti/programming/NSM/NSM/reconstruct/main.py#L561) | `torch.randperm(...)` | Point sub-sampling each iteration of the optimizer (which surface samples Adam sees in each minibatch) |
| [`NSM/NSM/mesh/main.py:470`](file:///dataNAS/people/aagatti/programming/NSM/NSM/mesh/main.py#L470) | `torch.rand(n_random_samples, 3)` | Random samples in adaptive mesh creation (`create_mesh_adaptive`) |
| [`nsosim/wrap_surface_fitting/fitting.py:765-770`](file:///dataNAS/people/aagatti/programming/nsosim/nsosim/wrap_surface_fitting/fitting.py#L765-L770) | `torch.randn(3)`, `torch.rand(1)` | Random axis perturbation in wrap surface fitting |

**Latent init is the dominant source.** Adam and LBFGS are deterministic given fixed inputs — their state (momentum, history) initializes to zero. So if the latent starting point and the per-iteration `randperm` are pinned, the entire optimization trajectory becomes deterministic, including the final latent value, which is what the SDF decoder converts back to a mesh.

Plus indirect sources:
- **cudnn benchmarking** (default `True`) picks the fastest kernel per input shape, which is non-deterministic.
- **cudnn deterministic** (default `False`) lets some convolution kernels use non-deterministic algorithms.
- **ACVD resampling** (`pymskt.mesh.resample_surface` → `pyacvd.Clustering`) — pyacvd's `cluster()` itself doesn't call `np.random` (initial cluster placement is point-ordering-dependent), but its output is sensitive to small numerical perturbations in the input mesh, which inherit upstream PyTorch noise.
- **Marching cubes / flying edges** topology is sensitive to SDF floating-point noise. Even tiny SDF perturbations from PyTorch non-determinism can flip which voxel boundary a triangle crosses, changing point counts. This is why `med_men_lower_art_surf_osim` consistently has different point counts (322 vs 437-465) across runs.

## ACVD / `resample_surface` — likely a non-issue once upstream is deterministic

The medial meniscus lower articulating surface (`med_men_lower_art_surf_osim`) has been the persistent verification failure (ASSD 0.4-0.5mm vs 0.1mm tolerance, point count 322 vs 437-465 across runs). The 322 triangle count itself is **not the problem** — a small surface should have a small triangle count. The issue is that the surface is *small relative to its parent meniscus* (~5%), so when the upstream NSM-fit meniscus mesh has a slightly different topology each run, the boundary extraction lands on a different subset of faces. Small surface × upstream noise = large proportional change in output.

Once Tasks 1-2 below land and the upstream meniscus mesh is bit-identical across runs, the boundary extraction should produce the exact same 322 triangles every time, and ASSD should drop to 0. **No quadric-decimation fallback or cluster-count change needed** — the apparent instability dissolves once the inputs stop drifting.

We should still verify this by re-running the verification suite after determinism lands. If the meniscus boundary extraction *still* shows variance with seeded fits, that points to a real boundary-finding algorithm sensitivity that would need separate treatment.

The CLAUDE.md NSM-space gotcha (scale up before ACVD because pyacvd struggles at coordinate scale ~[-1, 1]) is a separate issue worth verifying as a side audit, but it's about coordinate scale, not triangle count. Confirm it's applied in:

| File | Call | Coordinate space at call site |
|---|---|---|
| `nsosim/decode.py:96` | `mesh_osim.resample_surface(...)` | `_osim` = meters | likely fine |
| `nsosim/nsm_fitting.py:868` | `mesh_osim.resample_surface(...)` | `_osim` = meters | likely fine |
| `nsosim/articular_surfaces.py:784,791,795` | meniscus + bone resampling | meters | likely fine |

All these call sites operate in `_osim` meter space, not NSM-normalized space, so the scale-up concern probably doesn't apply. Worth a quick confirmation.

Documented in [`MENISCUS_ARTICULAR_SURFACE_INSTABILITY.md`](../../MENISCUS_ARTICULAR_SURFACE_INSTABILITY.md) — that doc may be supersedable / closeable after determinism lands and we re-run the verification.

## Tasks

### Task 1 — Add `set_global_seed()` utility

New file: `nsosim/_determinism.py`

```python
def set_global_seed(seed: int = 0) -> None:
    """Pin all random sources to make NSM fitting deterministic.

    Sets PyTorch (CPU + CUDA), NumPy, Python random, and cudnn flags.
    Idempotent — safe to call multiple times. Call once at the entry
    point of any public API function that produces meshes.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Task 2 — Wire seed into public API entry points

Add `seed: int | None = 0` parameter (default 0, opt-out via `None`) to:
- `nsosim.nsm_fitting.fit_nsm` (or whatever the top-level fit function is)
- `nsosim.decode.decode_joint_from_descriptors`
- `nsosim.decode.nsm_recon_to_osim`
- `nsosim.model_building.build_joint_model`

Each function calls `set_global_seed(seed)` at the top if `seed is not None`. Pass `seed=None` to opt out (existing stochastic behavior preserved for callers that explicitly want it).

### Task 3 — Confirm meniscus boundary stability with seeded fits

After Tasks 1-2 land, re-run the comak_gait_simulation verification suite twice with the same seed. Expected outcome: `med_men_lower_art_surf_osim` produces bit-identical triangle counts and 0mm ASSD. If it doesn't, the boundary-finding logic itself has non-determinism that needs separate treatment (likely a tiebreaker in `create_meniscus_articulating_surface`).

Also a quick coordinate-space audit of `resample_surface` call sites: confirm they're operating in meters (not NSM-normalized [-1,1] space, which would trigger the documented ACVD boundary-artifact issue). All three call sites in `articular_surfaces.py` work on `_osim` meshes, so this should be a no-op confirmation.

### Task 4 — Verify bit-identical output

End-to-end test: run `fit_nsm(seed=42)` twice on the same subject, then compare outputs:
- Latent vectors (NPY) — `np.array_equal`
- Reconstructed meshes (VTK point arrays) — `np.array_equal` after sorting
- ASSD between two runs — should be exactly 0.0 mm

Add as `tests/test_determinism.py`. Mark xfail until Task 1+2 land.

### Task 5 — Update `comak_gait_simulation` to pass seed

In `run_simulations/scripts/comak_1_nsm_fitting.py`, add `--seed` flag (default 42 or whatever feels right) and propagate it to `fit_nsm` / `build_joint_model` calls. Save into `run_config.json` so the seed is recorded with each subject.

### Task 6 — Re-run verification

Once Tasks 1+2 ship: rerun `comak_gait_simulation/tests/verify_pipeline/submit_verification.sh` twice with the same seed. Expected outcomes:
- Step 1 geometry: 66/66 pass (eliminates the 3 known meniscus failures, since topology is now stable).
- Step 2 biomechanics: most or all 9 checks pass at much tighter tolerances.
- The 3 known meniscus failures move from "stochastic boundary instability" to either "passing" or "real biomechanical sensitivity worth investigating".

## Open questions

1. **Should seed be config-driven or always 42?** Option A: bake seed=42 in as the default and let users override via config; option B: require explicit seed in config.
2. **Does training-time stochasticity matter?** The trained NSM model weights themselves came from a stochastic training run. The plan above only deterministically *uses* the trained weights — different model weights would still produce different fits. For now: out of scope; we treat the published weights as fixed inputs.
3. **What about Open3D, VTK, or other non-Python random sources?** VTK's marching cubes is deterministic given fixed input. Open3D's Poisson disk sampling has its own RNG (not currently used in our path AFAIK). If we add it, hook `o3d.utility.random.seed()`.

## Cross-references

- [`MENISCUS_ARTICULAR_SURFACE_INSTABILITY.md`](../../MENISCUS_ARTICULAR_SURFACE_INSTABILITY.md) — the meniscus boundary issue this plan addresses
- `comak_gait_simulation/.claude/plans/VERIFICATION_PHASES_BC_PLAN.md` — Steps 3, 4, 5 (calibrate stochasticity ceiling) become moot once determinism lands; Step 1 geometry tolerances can drop from 0.1mm ASSD to ~1e-6 mm.
- `comak_gait_simulation/CLAUDE.md` Gotchas section — the ACVD scale-up pattern for NSM-space meshes.

## Done when

- `set_global_seed()` exists and is called from all public entry points.
- Two runs of `fit_nsm(seed=42)` on the same subject produce bit-identical latents and mesh point coordinates.
- The verification suite in `comak_gait_simulation` runs with two fresh seeded runs and reports bit-identical Step 1 geometry.
- Step 2 biomechanical variance drops to numerical noise (or, if it doesn't, we have proof that COMAK itself contributes non-determinism worth investigating separately).
