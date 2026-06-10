# Mode 4 — Personalized Knee at True (MRI) Anatomical Size

## Goal

Implement **Mode 4** from [`docs/deviations.md`](../../docs/deviations.md): place a subject's
personalized knee into a COMAK model **at its true anatomical (MRI-segmented) size**, instead of
the reference size that the similarity registration normalizes it to. This is for the
matched-subject case — the person whose MRI was segmented is the person whose gait body the model
is built for.

This plan is **design only** — no code has been written. It records the chosen approach, the
exact edits, the verified-against-code error sources, and the test plan.

## Quick Start (environment)

```bash
conda run -n comak python -m pytest tests/scaling/ -v   # run scaling tests
conda run -n comak make lint                            # check formatting
```

OpenSim (JAM/COMAK fork) is a source build pinned to Python 3.9 + numpy 2.0.2 — only available in
the `comak` conda env.

---

## Background — why the knee comes out reference-size, and the one lever that fixes it

The MRI→OpenSim chain ([`docs/coordinate-systems.md §2`](../../docs/coordinate-systems.md)):

```
MRI mesh (mm, subject true size)
  → align_bone_osim_fit_nsm: femur similarity-register onto the smith2019 reference
        (rigid + isotropic scale — the scale DIVIDES the subject's true size out)
        tibia & patella REUSE the femur transform (not re-registered)
  → REFALIGN (mm, REFERENCE size)
  → fit_nsm + nsm_recon_to_osim (underscore converter, NO scale step)
  → OSIM meshes (m, REFERENCE size)
  → build_joint_model
  → Mode-1 COMAK model — knee is SHAPE-personalized but SIZE-normalized to reference
```

The single scale that was removed lives in `femur_transform`
([`nsm_fitting.py:156-165`](../../nsosim/nsm_fitting.py#L156)) — the result of
`subject_bone.rigidly_register(other_mesh=ref_, as_source=True, reg_mode='similarity')`. Because
all three bones get the *same* `femur_transform` applied
([`nsm_fitting.py:171-172`](../../nsosim/nsm_fitting.py#L171)), **one scalar restores the whole
knee to true size.**

**The existing scaling pipeline already isolates the knee scale.** In
[`scale_factors.py:90`](../../nsosim/scaling/scale_factors.py#L90):

```python
s_wa = (ab_factors["femur_r"][2] + ab_factors["tibia_r"][2]) / 2.0   # one isotropic ratio
```

`s_wa` is applied **only** to the 5 knee subbodies + `patella_r`; every other body passes its own
AddBiomechanics (AB) per-axis factor through ScaleTool. `build_scale_set` returns that one float,
and the orchestrator threads it into **both** knee consumers:

- the `ScaleSet` entries → ScaleTool scales wraps / ligaments (slacks) / joint frames / muscles /
  patella offset by it ([`scaletool.py`](../../nsosim/scaling/scaletool.py));
- `bake_knee_geometry(scale=s_wa)` → multiplies every knee STL vertex by it **about the
  body-local joint-center origin** ([`knee_geometry.py:107`](../../nsosim/scaling/knee_geometry.py#L107)).

This whole "resize the built knee by one isotropic scalar about the joint origin" operation is
**verified end-to-end** in [`tests/scaling/test_build_then_scale.py`](../../tests/scaling/test_build_then_scale.py)
(every component scales by exactly the factor; reference strain + cart-bone proximity preserved).

---

## Chosen approach — `knee_scale_override` on `scale_comak_model` (build-then-scale with `s_true`)

Mode 4 is **not a new scaling primitive.** It is the verified build-then-scale path with the knee
scalar set to the subject's own true-size factor instead of `s_wa`:

> **Modes 3/5:** knee scalar = `s_wa` (knee follows the gait body).
> **Mode 4:** knee scalar = `s_true = 1 / s_reg` (knee restored to its own MRI true size),
> while non-knee bodies still take the matched subject's AB factors.

This is exactly the user's suggestion ("update the `_wa` scaling to instead be based on the scale
factor from the similarity registration") and the docs' own recommended route (record the
registration scale, re-apply as a clean OSIM-space multiply about the joint-center origin — **not**
the converter's `scale` arg).

### Why this approach (and not the alternatives)

| Approach | Verdict |
|---|---|
| **Build-then-scale with `knee_scale_override` (this plan)** | **Chosen.** Reuses the verified lever wholesale. Default-`None` = byte-identical to today, so zero blast radius on Modes 1/2/3/5. CPU-only, no GPU re-fit. |
| `reg_mode='rigid'` (keep true size from the start) | Rejected as primary — the docs' "simpler-but-weaker" fallback. Rigid fits align *worse* when the subject differs much in size from the reference (poor shape correspondence). |
| Mid-build interception (scale both generators) | Rejected — the "two-generator problem" ([`deviations.md` Mode 3](../../docs/deviations.md)): you must scale the recon meshes **and** the ref→subject warp independently. Strictly more error-prone. |
| `convert_nsm_recon_to_OSIM(scale=…)` (the backlog's "hook") | **Trap — do not use.** It scales in NSMcanon about a per-bone affine center (~4 cm off the joint origin) and *distorts* the joint ([`coordinate-systems.md §3`](../../docs/coordinate-systems.md)). [`backlog.md:8-12`](backlog.md) suggests it; the docs explicitly warn against it. |

### The one real gap: persist the registration scale

`femur_transform`'s scale is **never serialized** — the alignment JSON
([`nsm_fitting.py:447-454`](../../nsosim/nsm_fitting.py#L447)) saves only the *NSMcanon* ICP
transform (`linear_transform`, femur scale ≈ 0.013), `scale`, and `center`. The true-size factor
lives only in-memory at `dict_bones["femur"]["subject"]["transform"]`
([line 311](../../nsosim/nsm_fitting.py#L311)). Mode 4 must capture and persist it at fit time so a
CPU-only build-then-scale run can read it back.

---

## Implementation steps

### Step 1 — Persist the femur registration scale (the only fitting-side change)

In `align_knee_osim_fit_nsm` ([`nsm_fitting.py`](../../nsosim/nsm_fitting.py), at the femur
alignment-JSON write, ~lines 446-460), add **one additive JSON field** on the `bone == "femur"`
iteration only.

⚠️ **Three confirmed-in-code bugs to get right** (all three candidate designs got these wrong):

1. **Read the right variable.** At the write site the *local* `femur_transform` is `None` for the
   femur branch ([set at line 422](../../nsosim/nsm_fitting.py#L422)). The actual transform is on
   `dict_bones["femur"]["subject"]["transform"]` ([set at line 311](../../nsosim/nsm_fitting.py#L311),
   populated after the `align_bone_osim_fit_nsm` call returns).
2. **Convert VTK → numpy first.** `rigidly_register(return_transform=True)` returns a **VTK
   transform object**, not a numpy 4×4. `decompose_similarity` does `T[:3,:3]` / `np.linalg.norm`
   ([`transforms.py:37-39`](../../nsosim/transforms.py#L37)) and will throw on a VTK object. Use the
   repo's existing adapter `get_linear_transform_matrix(...)` (already imported; used at
   [line 447](../../nsosim/nsm_fitting.py#L447) for exactly this).
3. **Store the absolute value.** `decompose_similarity` flips sign when `det(R) < 0`
   ([`transforms.py:41-43`](../../nsosim/transforms.py#L41)). A proper similarity won't, but store
   `abs(...)` defensively so the multiplier is never negative.

```python
from nsosim.transforms import decompose_similarity   # add import

# ... inside the loop, on the femur iteration, near the JSON write:
if bone == "femur":
    # Mode 4's single-scalar assumption is only valid for a UNIFORM-scale registration.
    # Reject affine up front, and verify the matrix is actually uniform (defense in depth).
    if rigid_reg_type not in ("similarity", "rigid"):
        raise ValueError(
            f"femur_registration_scale needs a uniform-scale reg_mode "
            f"('similarity' or 'rigid'); got {rigid_reg_type!r}."
        )
    fem_T = get_linear_transform_matrix(dict_bones["femur"]["subject"]["transform"])
    col_norms = np.linalg.norm(fem_T[:3, :3], axis=0)
    if not np.allclose(col_norms, col_norms[0], rtol=1e-4):
        raise ValueError(
            f"femur registration is not a uniform similarity (column norms {col_norms}) — "
            "Mode 4's single-scalar assumption is invalid."
        )
    s_reg = abs(float(decompose_similarity(fem_T)[0]))
    dict_transform["femur_registration_scale"] = s_reg
```

Persist the field for **both** reg modes (`'rigid'` yields `s_reg ≈ 1.0`, so Mode 4 collapses to a
no-op on the knee — correct). Purely additive: every existing JSON reader ignores the new key.

### Step 2 — A tiny reader/convention helper (isolate the direction in ONE place)

Add to the existing `nsosim/scaling/scale_factors.py`, next to `read_ab_factors` (same module
already reads scaling inputs — **simplest**, no new file and no new `test_doc_references.py` symbol
surface):

```python
def true_size_scale_from_alignment(femur_alignment_json: Path) -> float:
    """Factor to multiply a reference-size OSIM knee by to restore true MRI size.

    The femur similarity registration maps subject-mm -> reference-mm, so its
    scale s_reg DIVIDES the subject's true size out (s_reg < 1 for a subject
    larger than reference). The built knee is reference-size, so restoring true
    size is the RECIPROCAL: s_true = 1 / s_reg.
    """
    data = json.loads(Path(femur_alignment_json).read_text())
    if "femur_registration_scale" not in data:
        raise ValueError(
            f"{femur_alignment_json} has no 'femur_registration_scale' — this "
            "subject was fit before Mode 4 support. Re-run align_knee_osim_fit_nsm, "
            "or pass knee_scale_override manually."
        )
    s_reg = float(data["femur_registration_scale"])
    if s_reg <= 0:
        raise ValueError(f"non-positive femur_registration_scale {s_reg}")
    return 1.0 / s_reg
```

Keeping the `1/s_reg` inversion in this single documented function means that if the empirical
direction check (Step 6) ever shows the convention is flipped, the fix is one line.

### Step 3 — Thread an override through `build_scale_set`

[`scale_factors.py:40`](../../nsosim/scaling/scale_factors.py#L40):

```python
def build_scale_set(ab_factors, mode="WA", knee_scale_override=None):
    ...
    s_wa = (ab_factors["femur_r"][2] + ab_factors["tibia_r"][2]) / 2.0
    s_knee = s_wa if knee_scale_override is None else float(knee_scale_override)
    # use s_knee for BOTH the patella_r override entry AND the WA_KNEE_BODIES loop
    # non-knee AB bodies keep passing their own per-axis factors UNCHANGED
    return scale_set, s_knee     # the returned float is what gets baked
```

- `knee_scale_override is None` → **byte-identical** to today (regression guard).
- The `patella_r` special-case ([lines 97-98](../../nsosim/scaling/scale_factors.py#L97)) **must**
  use `s_knee` too — otherwise the patella stays reference-size inside a true-size knee.
- When the override is given, relax the `femur_r`/`tibia_r`-present requirement (a Mode-4 caller
  may not care about the AB-derived `s_wa`), but still compute `s_wa` for the report if present.

### Step 4 — Surface it on the orchestrator + record provenance

[`scale_comak_model`](../../nsosim/scaling/orchestrator.py#L150): add
`knee_scale_override: Optional[float] = None`, pass it into `build_scale_set(...)`. The returned
scalar (now `s_knee`) already flows into `bake_knee_geometry(scale=…)`
([line 288](../../nsosim/scaling/orchestrator.py#L288)) — **no other scaling change needed.**

**`build_scale_set` keeps its 2-tuple return `(scale_set, s_knee)` — no arity change.** Every
existing caller (the orchestrator + the 4 test sites, all `scale_set, x = build_scale_set(...)`) is
untouched, and `knee_scale_override=None` returns today's value exactly (byte-identical). Do **not**
widen the return to carry the AB `s_wa` — instead, the orchestrator recomputes the AB-derived `s_wa`
itself from `factors` (it already holds them) purely for the report.

Record provenance in the JSON report ([`write_report`](../../nsosim/scaling/orchestrator.py#L326)):
keep `wa_scale` meaning the **AB-derived** weighted average (recomputed from `factors`) so existing
report-parsers don't break, and add new keys: the **applied** knee scalar (`s_knee`), an
`override_used` flag, and — for the wrong-subject audit trail (open decision #3) — the
`femur_alignment.json` path the override came from and its `s_reg`. Without this, `wa_scale` silently
records the override and hides the AB value, and there is no record of which subject's fit produced
the scale.

### Step 5 — Mode-4 caller wiring (no new pipeline)

```python
# (a) build the personalized knee on a REFERENCE base — standard Mode-1, no change
built = build_joint_model(..., path_base_osim_model=reference_comak_base)

# (b) read the true-size factor from the persisted fit
from nsosim.scaling.scale_factors import true_size_scale_from_alignment
s_true = true_size_scale_from_alignment(folder_save_bones / "femur" / "femur_alignment.json")

# (c) scale: knee → true MRI size; body → the matched subject's own AB factors
scale_comak_model(
    base_osim=built,
    ab_scaled_osim=subject_ab_osim,        # the MRI subject's OWN AddBiomechanics output
    output_osim=out_osim,
    output_geometry_dir=out_osim.parent / "Geometry",
    knee_scale_override=s_true,
)
```

### Step 6 — Direction guard (synthetic, in-repo, always runs in CI)

The reciprocal (`s_true = 1/s_reg`) is reasoned-correct; a coding slip (writing `s_reg` where
`1/s_reg` is meant) would scale the knee ~`s_reg²` off true size with **no crash**. **This needs no
real data — it's a normal always-on CI test.** Scale a mesh by a *known* factor `K`, then confirm
the pipeline's own registration + formula recover it:

```python
# in-repo reference femur (tests/fixtures/osim_models/Geometry/... — tracked, tiny)
ref = Mesh("tests/fixtures/osim_models/Geometry/smith2019-R-femur-bone.stl")
K = 1.30                                    # subject is 1.30x reference — KNOWN ground truth
subject = scale_rotate_translate(ref, K, R_random, t_random)
T = subject.rigidly_register(other_mesh=ref, as_source=True, reg_mode="similarity",
                             return_transform=True)
s_reg = abs(decompose_similarity(get_linear_transform_matrix(T))[0])
assert s_reg  == pytest.approx(1 / K, abs=1e-3)   # registration maps subject -> reference
assert true_size_scale_from_alignment(json_with(s_reg)) == pytest.approx(K, abs=1e-3)  # formula restores true size
```

**Verified now (2026-06-09):** with `K = 1.30`, the registration recovers `s_reg = 0.769231 = 1/1.30`
and `s_true = 1.300000` — exact to 1e-6 (the subject is an exact scaled+rotated copy, so
correspondence is perfect). A flip (`s_true = s_reg`) yields `0.769`, fails `assert ≈ 1.30`, caught
in CI. Because `K` is **known ground truth**, this is *stronger* than any real-data check — real
anatomy only gives the size ratio as an *estimate*, never the true factor independently. This, plus
the synthetic-similarity persistence test (Step 7 #4), fully guards the direction with zero external
data.

**`s_reg`/`s_true` are per-subject — never a library constant.** At runtime each subject's `s_reg`
comes from their own fit and is read from their own `femur_alignment.json`; the reader just returns
`1/s_reg` for whatever that value is. `K = 1.30` above is a test input, not a shipped number.

**Optional real-data integration smoke (gated/slow — skips without the big meshes).** Adds *nothing*
to the direction guarantee; it only confirms the *live* pipeline wires the registration scale
through end-to-end (catches Step 1 reading the wrong variable in a real run). For a real fit, assert
persisted `femur_registration_scale ≈ size(recon)/size(raw_MRI)` for that subject. Assets for
9003175 / 00m / RIGHT are on disk (raw MRI at `…/OAI_DESS/meshes/00m/9003175/9003175_RIGHT_femur.vtk`,
recon at `…/built_models/mode1_9003175_00m_RIGHT/Geometry/femur_nsm_recon_osim.stl`); measured
`s_reg ≈ 0.9755`, `s_true ≈ 1.0251` (~2.5 % larger than reference). ⚠️ Use the **rotation-invariant
size ratio** (RMS-to-centroid; raw is MRI-orientation/**mm**, recon is OSIM/**m** ×1000) — a **cold
re-registration is UNRELIABLE** here (it gave 0.86, ~12 % off, because a cold ICP across the
MRI↔OSIM orientation gap lands in a different local minimum than the build did). Skip this whole
block and the synthetic test still fully guards the direction.

### Step 7 — Tests (`tests/scaling/test_mode4_true_size.py`)

Model these on [`test_build_then_scale.py`](../../tests/scaling/test_build_then_scale.py) and
[`test_identity.py`](../../tests/scaling/test_identity.py) — swap `s_wa` → an explicit override
`K`. **The Mode-4-specific assertion that catches the realistic bugs is the *decoupling* check:**
non-knee bodies keep their AB factor while knee bodies == `K`.

1. **Unit (always runs, no untracked deps):** `build_scale_set(factors, knee_scale_override=K)`
   with `K != s_wa` → every `WA_KNEE_BODIES` entry **and** `patella_r` == `(K,K,K)`; a non-knee body
   (e.g. `torso`/`talus`) still carries its AB factor; returned scalar == `K`. And
   `knee_scale_override=None` reproduces the current `ScaleSet` exactly.
2. **End-to-end on the in-repo reference base** (mirrors `TestReferenceOriginScalingPlacement`,
   needs no large fixture): `scale_comak_model(..., knee_scale_override=K)` → every knee STL vertex
   set, wrap translation/radius, `pf_tx/ty/tz_r` offset, ligament/muscle path-point sets each scale
   by exactly `K` (reuse `TOL_STL_M=1e-7`, `TOL_EXACT=1e-12`); knee specific inertia `I/m` scales by
   `K²`; **non-knee bodies scale by their own AB factor, not `K`**.

   ⚠️ **The verification mechanism differs by body class** (this is the easy test bug to write):
   knee subbodies are checked on the **baked STL vertices** (and confirm their `scale_factors` reset
   to `[1,1,1]`); non-knee bodies are **not baked** — `bake_knee_geometry` only touches
   `WA_KNEE_BODIES` — so read their factor off the `<attached_geometry>` `scale_factors` / the
   `ScaleSet`, **not** the STL (their STL on disk is unchanged). Use `femur_r`/`tibia_r` (AB = 0.9 in
   the `synthetic_wa_ab_osim` fixture) vs `femur_distal_r`/`tibia_proximal_r` (= `K`) as the
   **decoupling witnesses**; every other body is identity in that fixture and won't distinguish the
   override from no-override. A fixture variant with a non-identity *non-knee* body (e.g.
   `torso = 1.2`) makes the decoupling assertion bite harder.
3. **Identity:** `knee_scale_override=1.0` is a bit-exact no-op on knee geometry while non-knee
   bodies still scale.
4. **Fit-side persistence:** a known synthetic similarity (scale=1.37, random proper R, random t)
   → `decompose_similarity(get_linear_transform_matrix(T))[0]` recovers 1.37 to 1e-12; the femur
   alignment JSON contains `femur_registration_scale`; it is positive;
   `true_size_scale_from_alignment` returns `1/value`. Assert a *bare* `decompose_similarity(vtk_obj)`
   fails (proves the adapter is required). Also assert the new `rigid_reg_type` guard rejects
   `'affine'` and a non-uniform-column-norm matrix.

   ⚠️ **`s_reg`/`s_true` are per-subject — there is NO library-wide constant.** Each subject's
   `s_reg` comes from their own fit and is read from their own `femur_alignment.json`. Three guards,
   **all in-repo / always-on** (none needs the big meshes):
   - **Synthetic direction guard (Step 6):** scale the in-repo ref femur by a known `K`, register,
     assert `s_reg ≈ 1/K` and `true_size_scale_from_alignment ≈ K`. This is the primary direction
     guard — known ground truth, runs in CI. Catches the `s_reg`-vs-`1/s_reg` flip.
   - **Inversion unit test:** a fixture JSON with `femur_registration_scale = X` →
     `true_size_scale_from_alignment` returns `≈ 1/X` (X arbitrary). Proves the reader applies the
     inversion and the affine / non-uniform guard fires.
   - **Persistence unit test:** the synthetic similarity above → the femur JSON contains a positive
     `femur_registration_scale` recovering `K`.
   - *Optional* **real-data integration smoke (gated/slow, skips without big meshes):** persisted
     `femur_registration_scale ≈ size(recon)/size(raw_MRI)` for a real subject — only confirms the
     live pipeline wires the scale through (catches Step 1 reading the wrong variable / persisting
     the NSMcanon ≈0.013 scale). Adds nothing to the direction guarantee.
5. **Reader robustness:** `true_size_scale_from_alignment` raises a clear (non-`KeyError`) error
   when the field is absent.
6. **Runnable:** the Mode-4-scaled model loads, `realizePosition` succeeds, total mass > 0.

### Step 8 — Docs

Update [`docs/deviations.md`](../../docs/deviations.md) Mode 4 from "🚧 not implemented — design
sketch" to **implemented**, stating: `s_true = 1/s_reg`, the `knee_scale_override` seam, the
SIZE(`s_true`)-vs-POSITION(AB) decoupling, and the forward-only persistence caveat. Update the
`scale_comak_model` docstring. Keep `tests/docs/test_doc_references.py` green (any newly-referenced
symbol must exist).

---

## Error sources (ranked) — the user explicitly asked

> **Nature of this list — read first.** Mode 4 is not built yet, so **nothing here is a bug that
> exists today**; this is a pre-flight checklist of what would bite *if the code is written the
> obvious way*. Two tiers:
> - **Rows 1-3 are genuine code-level gotchas verified against the current source** — they would
>   throw or silently misbehave if coded naively (the VTK-object type and the `None`-at-write-site
>   are facts about the existing code; the `1/s_reg` direction is the one number to confirm via the
>   Step-6 check). Get these right and the implementation is straightforward.
> - **Rows 5-10 are design considerations, not faults** — e.g. row 6 (knee *size* follows the MRI
>   scale while knee *position* follows the body's mocap scale) is the **intended** Mode-4 behavior,
>   flagged only so it is not mistaken for a defect in review.
>
> The approach itself is sound and reuses already-verified machinery; this table is a checklist, not
> a list of problems.

| # | Severity | Error | Where | Mitigation |
|---|---|---|---|---|
| 1 | **HIGH** | **Scale direction inverted** (`s_reg` instead of `1/s_reg`). Registration maps subject→reference, dividing true size out; restoring is the reciprocal. Wrong → knee ~`s_reg²` off, **no crash**. | reg @ `nsm_fitting.py:156-165` → bake scale | Use `1/s_reg`, isolated in `scale_factors.true_size_scale_from_alignment` (per-subject — read fresh from each fit, no shared constant). **Synthetic in-repo direction guard** (Step 6): scale a mesh by known `K`, assert pipeline recovers `K` — always-on CI, verified exact to 1e-6, no real data. Optional gated real-data smoke is extra, not the guard. |
| 2 | **HIGH** | **VTK object passed to `decompose_similarity`** → throws. `femur_transform` is a vtkTransform, not numpy. | Step 1 persist | `get_linear_transform_matrix(...)` first (already used at `nsm_fitting.py:447`). |
| 3 | **HIGH** | **Reading the wrong variable** — local `femur_transform` is `None` at the femur write site. | `nsm_fitting.py:421-422` vs `:311` | Read `dict_bones["femur"]["subject"]["transform"]`. |
| 4 | **HIGH** | **Double-scaling** — leaving the knee on `s_wa` *and* restoring true size. | `scale_factors.py:90-106` | Override **replaces** `s_wa` for the 5 knee bodies, never adds. Decoupling test guards it. |
| 5 | **MED** | **Wrong similarity transform** — the alignment JSON `linear_transform` (femur ≈ 0.013) is the *NSMcanon* normalization, not the registration scale. Reading it → ~0.013× knee. | `nsm_fitting.py:447` vs new field | Distinct key `femur_registration_scale`, sourced from `femur_transform`. Reader fetches exactly that key. |
| 6 | **MED** | **Knee POSITION tracks AB while SIZE goes true.** The `femur_r → femur_distal_r` weld translation scales by the *parent femur_r AB factors*, not the knee scalar — so a true-size knee sits on the AB-scaled shaft. **Intended** for Mode 4, but reads like a bug. | `scaletool.py:30-40` | Document as intentional (matches Mode 3 semantics). Do **not** special-case the weld in the minimal diff. |
| 7 | LOW | **Old subjects lack the field** (forward-only *as wired*). `femur_transform` was dropped after fitting for every existing subject. | persisted JSON | Reader raises a clear error. **Not a hard limit:** `s_reg` is recoverable **CPU-only** (no GPU re-fit) by re-registering the saved MRI femur onto the reference (`reg_mode='similarity'`, deterministic — the Step-6 procedure). Just not wired up here; docs/error message should say so. |
| 8 | LOW | **`rigid`/`affine` modes.** `rigid` → `s_reg == 1`, `s_true == 1` (no-op, correct). `affine` → non-uniform scale, single scalar invalid. | reg_mode branch | Persist for `similarity`+`rigid`; **guard rejects `affine`** + non-uniform column norms (Step 1). |
| 9 | LOW | **Negative scale** from `decompose_similarity` on `det(R) < 0`. | `transforms.py:41-43` | Store `abs(...)`; assert positive in reader. |
| 10 | LOW | **Inertia scales by `s_true²`** (geometry-consistent, matches Mode 3) — a reviewer may expect "true size = native inertia". | `scaletool.py` + two-pass | Document as intentional. |

### Coordinate / unit pitfalls

- **Dimensionless scalar.** `s_reg`/`s_true` are ratios (mm/mm); `bake_knee_geometry` works in OSIM
  **meters**. Any `/1000` or `*1000` around the scalar is a unit bug — keep it a pure ratio.
- **Origin of the multiply.** Knee vertices scale about the body-local origin `(0,0,0)` = the knee
  **joint center**, not the bone centroid ([`knee_geometry.py:45-49`](../../nsosim/scaling/knee_geometry.py#L45)).
  The bounding box shifts toward the origin under scaling — the Step-6 extents check must account
  for origin-centered (not centroid-centered) scaling.
- **All three bones share the femur scale** (tibia/patella are not independently registered). This
  is what makes one scalar valid for the whole knee. If a future model re-registers tibia/patella
  independently, the single-scalar assumption breaks.

---

## Open decisions (for the user)

1. ~~Where does the original MRI bone mesh live?~~ **RESOLVED:**
   `/dataNAS/people/aagatti/projects/OAI_DESS/meshes/{TIMEPOINT}/{SUBJECT_ID}/{SUBJECT_ID}_{SIDE}_femur.vtk`
   (configured as `oai_meshes_folder` in the comak repo). All Step-6 assets are on disk; the check
   is CPU-only.
2. **API name:** `knee_scale_override` (recommended — explicit) vs `knee_scale`. Trivial.
3. **Matched-subject guard.** Mode 4 is only meaningful when the gait subject *is* the MRI subject.
   `scale_comak_model` cannot hard-assert this (no robust subject-identity field). **RESOLVED:**
   document the precondition + soft warning, and **record provenance** in the scaling report (the
   `femur_alignment.json` path + `s_reg` it used) so a wrong-subject mistake is auditable after the
   fact (Step 4). It's an audit trail, not an enforced check.
4. ~~**Helper module** `nsosim/scaling/true_size.py`.~~ **RESOLVED:** put the reader
   (`true_size_scale_from_alignment`, the `1/s_reg` seam) in the existing
   `nsosim/scaling/scale_factors.py` next to `read_ab_factors` — simplest, no new file, no new
   doc-reference symbol surface.

## Scope / non-goals

- **Forward-only *as wired*:** new fits persist `femur_registration_scale`. Existing subjects do
  **not** require a GPU re-fit — `s_reg` is recoverable CPU-only by re-registering the saved MRI
  femur to the reference (the Step-6 procedure, deterministic). That backfill path is simply not
  wired up in this plan; the docs and the reader's error message should say it's possible so the
  limitation doesn't read as harder than it is.
- **Not** the `comak_gait_simulation` Pathway-B wiring (that's a separate backlog item).
- **Not** the "fit-on-scaled-base cross-check" — that is a *different*, deliberately-parked
  ("do not attempt") item ([`backlog.md:17`](backlog.md)). Mode 4 does not touch it.
- No mass/inertia special handling, no weld-translation special handling (both intentional).

## Files touched

| File | Change |
|---|---|
| [`nsosim/nsm_fitting.py`](../../nsosim/nsm_fitting.py) | +1 additive JSON field (Step 1) |
| [`nsosim/scaling/scale_factors.py`](../../nsosim/scaling/scale_factors.py) | `knee_scale_override` kwarg on `build_scale_set` (2-tuple return unchanged) + new `true_size_scale_from_alignment` reader (`1/s_reg`) |
| [`nsosim/scaling/orchestrator.py`](../../nsosim/scaling/orchestrator.py) | `knee_scale_override` kwarg; recompute AB `s_wa` for report; provenance keys (alignment-JSON path, `s_reg`, `override_used`) |
| `tests/scaling/test_mode4_true_size.py` | **new** — see Step 7 |
| [`tests/scaling/test_scale_factors.py`](../../tests/scaling/test_scale_factors.py) | +override/decoupling unit asserts |
| [`docs/deviations.md`](../../docs/deviations.md) | Mode 4 → implemented |

When implemented, finalize with `/complete-plan .claude/plans/mode4-true-size.md`.
