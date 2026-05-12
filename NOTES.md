# Wrap Fitter Robustness — Working Log

Self-pacing /loop run. See `.claude/plans/WRAP_FITTER_ROBUSTNESS.md` for the brief.

## Baseline (from the plan; not re-run yet)

```
A_v1 vs A_v2: 1/15623 differing lines (determinism — already passing)
A_v1 vs B   : 125/15623 differing lines (WrapEllipsoid max 0.148 rad ~8.5°)
```

Per-wrap geometric/parameter drift (from offline diff of existing A and B .osim files):

| Wrap | type | rotation diff (Euler max) | rotation diff (geodesic) | translation diff (Z) | dim diff |
|------|------|---|---|---|---|
| Med_Lig_r           | ellipsoid | **7.1° (gimbal-lock!)** | 1.24° | < 0.1 mm | < 0.1 mm |
| Gastroc_at_Condyles_r | ellipsoid | 0.68° | 0.70° | **2.0 mm** | < 0.1 mm |
| Med_LigP_r          | ellipsoid | 0.25° | 0.37° | < 0.1 mm | 0.25 mm |
| Capsule_r           | cylinder  | 0.04° | n/a | **0.29 mm** | < 0.05 mm |

**Key insight**: Med_Lig_r's middle Euler ≈ 1.45 rad (83°), Gastroc's ≈ 1.58 rad (91°) — both near gimbal lock. Med_Lig_r is **almost entirely representational** (rotation matrices are 1.24° apart geometrically, but Euler says 7.1°). Gastroc has real geometric translation drift.

## Approaches: running list (weigh risk × reward × effort)

Status: `[ ]` not tried, `[x]` confirmed improvement, `[-]` tried no improvement, `[!]` unsafe/reverted

| # | Approach | Risk | Reward | Effort | Status |
|---|----------|------|--------|--------|--------|
| 1 | Canonical ellipsoid pose: sort axes desc + sign-by-dominant-component (replaces `enforce_sign_convention` for ellipsoids) | LOW | HIGH (Med_Lig_r 7.1° → 1.2° offline) | LOW | [ ] in progress |
| 2 | Regularize fit toward initialization (`λ * ‖θ - θ_init‖²`) | MEDIUM (could hurt fit quality) | MEDIUM | MEDIUM | [ ] |
| 3 | Larger jitter scale (1e-3 mm) + deterministic tiebreaker — NOTE: test uses `wrap_n_restarts=1`, irrelevant unless we change that default | LOW | LOW (off path) | LOW | [ ] |
| 4 | Tighter L-BFGS convergence | LOW | LOW (already tight: `tolerance_grad=1e-9`) | LOW | [ ] |
| 5 | Switch optimizer (Newton, etc.) | HIGH | MEDIUM | HIGH | [ ] (skip) |
| 6 | PCA of bone region as initialization (data-driven, deterministic) | MEDIUM | MEDIUM | MEDIUM | [ ] |
| 7 | Huber loss instead of squared-hinge for outlier robustness | MEDIUM | UNKNOWN | LOW | [ ] |
| 8 | Reduce sensitivity in `_compute_distance_loss` (beta/gamma already 0) | LOW | LOW | LOW | [ ] (skip - already 0) |
| 9 | **(new)** Investigate the actual SDF function `sd_ellipsoid_improved` — is it smooth at boundaries? Could be a source of sensitivity. | MEDIUM | UNKNOWN | MEDIUM | [ ] |
| 10 | **(new)** For Gastroc (which has 2mm Z drift), the actual fit may be settling on a different local min. Try Lagrangian fix — penalize translation drift from initialization. | MEDIUM | MEDIUM | LOW | [ ] |
| 11 | **(new)** Median-filter the bone mesh (1e-4 mm scale smoothing) to absorb the CUDA-floor drift before fitting | LOW | UNKNOWN (could hurt fit) | MEDIUM | [ ] |

## Iteration log

### Iter 1 (2026-05-11) — Canonical ellipsoid pose ✅ COMMITTED

Implemented `RotationUtils.canonical_ellipsoid_pose(R, axes)`:
- Sort axes descending; permute R columns
- Sign convention keyed on each column's **dominant component** (stable near gimbal lock — unlike `enforce_sign_convention` which keys on R[0,0]/R[1,1] that shrink to zero there)
- 11 unit tests added (all pass)

**Offline prediction** (apply canonical pose to existing A and B .osim Euler values):
- Med_Lig_r: 7.1° → 1.2° (5.9× reduction, matches geodesic 1.24°)
- Gastroc_at_Condyles_r: 0.68° → 0.68° (unchanged — real geometric drift)
- Med_LigP_r: 0.25° → 0.37° (faithful to geodesic 0.37°)

**SLURM iter1 result (job 46860)** — `wrap_n_restarts=1`:
```
A_v1 vs A_v2: 1/15623 differing lines (determinism preserved ✓)
A_v1 vs B   : 125/15623 differing lines
  WrapEllipsoid max_abs: 0.02516 rad (1.44°)    ← was 0.148 rad (8.5°). 5.9× reduction.
  WrapCylinder  max_abs: 0.000289 m (0.29 mm)   ← unchanged (cylinders not touched)
  Other params: unchanged
```

**Verdict**: Real improvement on WrapEllipsoid rotation drift. Acceptance criteria 4, 5 pass; 3 safe (geometric invariance); 1 and 2 still need work because Med_Lig_r has real geometric rotation drift of 1.24°.

### What's still failing (Iter 1 → before Iter 2)

The remaining 1.4° max ellipsoid rotation drift is now bound by the **geometric** rotation drift between A and B (the rotation matrices truly differ by 1.24° geodesic). This is upstream-fit sensitivity, not output representation. Must reduce fit sensitivity itself.

Per-wrap geometric drift remaining (estimated):
- Med_Lig_r: 1.24° rotation → ~0.7 mm surface ASSD
- Gastroc_at_Condyles_r: 0.7° rotation + 2 mm Z translation → ~2 mm ASSD
- Capsule_r: 290 µm Z translation → 0.29 mm ASSD
- Med_LigP_r: 0.37° → ~0.2 mm ASSD

All above criterion 0.10 mm. **Need to address upstream fit sensitivity, not output rep.**

## More approach ideas added during iter1

| # | Approach | Risk | Reward | Effort | Status |
|---|----------|------|--------|--------|--------|
| 12 | Turn on `beta > 0` (correlation-based SDF loss) — currently `beta=0` so SDF is ignored | LOW | MEDIUM | LOW | [ ] try iter2? |
| 13 | More Adam epochs for ellipsoid (10 → 100) before L-BFGS | LOW | UNKNOWN (L-BFGS already does 97% of work) | LOW | [ ] |
| 14 | Add quadratic regularizer pulling fit toward **initialization** (per #2 in original list, made concrete) | LOW-MED | MEDIUM-HIGH | LOW-MED | [ ] |
| 15 | **(new)** Smooth/median-filter the labeled mesh's near-surface points before fitting — averages out 1e-4 mm vertex noise | MEDIUM | UNKNOWN | MEDIUM | [ ] |
| 16 | **(new)** Use soft labels (sigmoid(sdf/τ)) instead of hard binary labels — points near boundary contribute fractionally so labeling jumps don't cause discrete loss changes | MEDIUM | MEDIUM | MEDIUM | [ ] |
| 17 | **(new)** Apply same canonicalization (sort + sign-by-dominant) but for the **fit's internal quaternion** before extracting Euler — would canonicalize the optimization basin, not just the output. Risk: changes fit dynamics. | HIGH | HIGH | MEDIUM | [ ] |
| 18 | **(new)** For Gastroc's 2mm Z translation drift: investigate WHY — is the loss landscape flat in Z? Add a 1D probe. | LOW | INFO | LOW | [ ] diagnostic |

### Iter 2 (2026-05-11) — Translation regularizer (ellipsoid) ✅ COMMITTED

Added `lambda_center_reg=1.0` (1/m²) to EllipsoidFitter. Pulls center toward
its initialization (geometric/algebraic-fit center) in flat directions; the
geometric loss dominates wherever the landscape has real curvature.

**SLURM iter 2 result (job 46861)**:
```
A_v1 vs A_v2: 1/15623 differing lines (determinism ✓)
A_v1 vs B   : 125/15623 differing lines
  WrapEllipsoid          max_abs: 0.02135 rad (1.22°)     ← was 0.02516 in iter 1
  WrapCylinder           max_abs: 0.000289 m (unchanged)  ← cylinder reg not yet added
  Blankevoort1991Ligament max_abs: 2.434e-05 m (24 µm)    ← was 208 µm. 8.5× drop!
  PathPoint, Coordinate, Millard: unchanged
```

**Per-wrap detail** (Δ between A_v1 and B):
- **Gastroc_at_Condyles_r**: Z translation **2mm → 4µm (500×!)**. Dimensions Δa 84µm → 360µm (axis absorbed some drift). Rotation max 0.68° → 0.4°.
- **Med_Lig_r**: translation now <25µm all axes. Rotation 1.23° (geometric drift unchanged). Dimensions Δa 50→187µm (slight regression).
- **Med_LigP_r**: translation <10µm. Rotation 0.33°. Dimensions Δa 258→365µm.
- **Capsule_r**: 289µm Z translation (unchanged — cylinder reg not applied).

**Geometric ASSD estimates** (rough, rotation×axis + translation + dim):
- Gastroc: iter 1 ~2 mm → iter 2 ~1 mm
- Med_Lig_r: ~0.7 mm → ~0.7 mm
- Med_LigP_r: ~0.4 mm → ~0.4 mm
- Capsule_r: 0.29 mm → 0.29 mm

Still above 0.10 mm acceptance threshold but headed in the right direction.

## Iter 3 plan

Two parallel additions:
1. **Add translation regularizer to CylinderFitter** — analogous to ellipsoid; addresses Capsule_r 289µm Z drift.
2. **Add `lambda_axes_reg > 0` and `lambda_quat_reg > 0`** to ellipsoid — to prevent axis-drift "blame transfer" and tackle Med_Lig_r rotation drift.

For lambda_quat_reg: anchor to `quat_init` (from algebraic-fit-derived rotation). For non-degenerate axis ratios (Med_Lig_r has 0.012, 0.017, 0.032 — all well-separated), `quat_init` should be reasonably stable.

Cylinder also needs care: the axis vector parameter and the cylinder's translation along its axis. Translation reg should be on the **center** parameter.
