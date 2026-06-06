# Scaling Deviations

The **single** place that records where the knee-build *pipeline wiring* uses the library
in a way that produces a **reference-scale knee inside a (possibly `s_wa`-scaled) body**.

Framing rule (kept out of the docstrings on purpose): the library is a capable tool that
supports both "scale the knee to the gait body" and "keep the knee at the subject's true
MRI size." Each entry below is written as **"the pipeline uses library capability X in
mode Y; to get mode Z, do W."** The docstrings and
[SCALING_AND_SPACES.md](SCALING_AND_SPACES.md) describe *what the code does*; this file is
the only place that frames it as a gap and points at the lever.

These are **not bugs in the library functions** — they are wiring choices. None of the
"do W" items are built yet; they are inputs to a future fix plan, not work items here.

---

## 1. MRI knee size is normalized to the reference

**What happens.** The femur similarity registration
([align_bone_osim_fit_nsm](../nsm_fitting.py#L36), `reg_mode='similarity'`) divides out
the subject's true anatomical scale; tibia/patella reuse the femur transform. The MRI path
to OSIM uses only the underscore
[convert_nsm_recon_to_OSIM_](../nsm_fitting.py#L793), which adds the fixed ref-center,
converts mm→m, and axis-swaps — with **no subject-scale step**. So every built knee is
~reference size: shape-personalized, size-normalized.

**Evidence.** Raw MRI femur diagonal CoV across subjects ≈ 7.9%; after registration ≈ 1.0%
(size collapsed onto the reference).

**Library capability / "do W".** The **non-underscore**
[convert_nsm_recon_to_OSIM](../nsm_fitting.py#L901) forwards a `scale` term into
[undo_transform](../nsm_fitting.py#L741); a non-unit `scale` re-introduces true size. The
synthetic path already exercises this converter (with `scale=1`). A keep-true-size
("Pathway C") mode would route the MRI recon through this hook with the subject's true
scale instead of the underscore converter. Not built.

---

## 2. Pathway B knee does not inherit `s_wa`

**What happens.** Under multigait (Pathway B), Stage X scales the *body* to the AB subject
and bakes the reference knee STLs by `s_wa`
([bake_knee_geometry](../scaling/knee_geometry.py#L29)). But the knee build then writes the
**reference-scale recon** (`femur_nsm_recon_osim.stl`, …) and repoints the model to it via
[update_body_geometry_meshfile](../osim_utils.py#L31) /
[update_contact_mesh_files](../osim_utils.py#L72) with `scale_factors` left at `1,1,1`. The
recon is never multiplied by `s_wa`. Result: a reference-scale knee inside an
`s_wa`-scaled body — a geometric mismatch.

**Evidence.** On `OARSI_multigait_RSubject_121_…/9018389_00m_RIGHT`, `s_wa = 0.97298`; the
baked `smith2019-R-femur-bone.stl` is exactly `0.97298×` but **orphaned**, while the body
references `femur_nsm_recon_osim.stl` at `scale_factors = 1 1 1` (reference scale).

**Library capability / "do W".** `s_wa` is on disk in the Stage-X scaling report. To get
the matched-scale ("Pathway B done right") mode, apply `s_wa` to the knee recon about the
**common center = body-local origin = knee joint center** (see
[SCALING_AND_SPACES.md](SCALING_AND_SPACES.md) §1, §5) at the OSIM-entry point
([nsm_recon_to_osim](../nsm_fitting.py#L1027) /
[convert_nsm_recon_to_OSIM_](../nsm_fitting.py#L793)), and equally to the warp-path
attachment converter, the labeled mesh fed to the wrap fitter, and `mean_patella`
(deviation §3). Not built.

---

## 3. `mean_patella` joint translation is reference-scale

**What happens.** [center_patella_meshes](../model_building.py#L727) subtracts the
patella's mean position (`mean_patella`) so the patella STL is centered at its body-local
origin; that offset is later written as the patellofemoral joint coordinate translation
(via [update_osim_model](../comak_osim_update.py#L144)). `mean_patella` is a translation in
**OSIM meters at reference size**.

**Why it matters.** It is the one load-bearing *scalar* in the knee build. Even if a future
fix scales the patella meshes by `s_wa`, this translation must be scaled too (about the
same common center) or the patellofemoral kinematics drift relative to the scaled
geometry.

**Library capability / "do W".** Multiply `mean_patella` by `s_wa` (about the common
center) wherever the meshes are scaled. Not built.

---

## 4. Stage-X knee STL bake is orphaned under Pathway B

**What happens.** Stage X bakes `smith2019-R-{femur,tibia,patella,…}.stl` by `s_wa`
([bake_knee_geometry](../scaling/knee_geometry.py#L29)). The knee build then repoints the
model to the **differently named** recon STLs ([save_geometry_files](../model_building.py#L819)
copies new files; the `update_*` helpers repoint via `set_mesh_file`). The baked
`smith2019-R-*.stl` are left on disk but not referenced by the built model — only the
**non-knee** Geometry (torso, pelvis, limbs) retains Stage-X scaling.

**Note on the cross-repo map.** The mechanism is **repoint + orphan**, not
"overwrite-by-filename-collision." The recon STLs have different names from the baked ones,
so there is no overwrite; `save_geometry_files` copies new files and the model is repointed.
Net outcome matches the overwrite story, but the mechanism differs — and the difference
matters for a fix (you scale the recon that gets *repointed to*; the Stage-X bone bake is
dead weight under Pathway B).

**Library capability / "do W".** Decide the fate of the Stage-X bone/cart/contact bake
under Pathway B (orphaned → drop, unless needed elsewhere). Not built.

---

## Where the fix would touch (pre-identified; do NOT implement from this file)

These are recorded so the future fix plan starts from a verified list — they are **not**
work items in the documentation pass:

- Apply `s_wa` to the knee recon about the common center at the OSIM-entry point
  (`nsm_recon_to_osim` / `convert_nsm_recon_to_OSIM_`), **and** to the warp-path attachment
  converter, **and** to the labeled mesh fed to the wrap fitter, **and** to `mean_patella`.
- Thread the synthetic path ([decode.py](../decode.py)) separately if the feature is meant
  to be general.
- Keep mass/inertia owned by Stage X (the orchestrator two-pass) — a geometry fix touches
  geometry only.
- For Pathway C (true-size), route the MRI recon through the non-underscore converter's
  `scale` hook (deviation §1) instead of the underscore converter.

See `.claude/plans/scaling-and-spaces-documentation.md` (Stage 5) for the fix-plan scope.
