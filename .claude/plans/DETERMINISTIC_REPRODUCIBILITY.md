# Make NSM Fitting & Decode Deterministic

**Status:** Planned — no implementation yet.
**Driver:** Verification work in `comak_gait_simulation` traced cascading Step 2 biomechanical variance (5–15% NRMSE on contact pressures, ~3° on PF flexion, 28-39% COMAK convergence flips) back to NSM-fitting stochasticity. Two independent runs of the same code on the same subject produce different geometry → different contact mechanics → different optimizer trajectories.

**Goal:** Given identical inputs (subject mesh + model weights + config), produce bit-identical outputs across runs. This unlocks pipeline verification with tight tolerances and reproducibility for paper-grade results.

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
