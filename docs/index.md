# nsosim

`nsosim` builds **personalized biomechanical knee models** by fitting Neural Shape Models
(NSM) to MRI segmentations and integrating the result into OpenSim/COMAK simulations. It
bridges raw imaging data → NSM-derived geometries → subject-specific OpenSim models.

This site is generated from the source docstrings, so the API reference always matches the
code. Start here for the map; follow the links into the reference for the detail.

---

## The two pipelines

Everything in the library composes into one of two flows:

- **MRI / fitting pipeline** — `subject mesh → latent → OpenSim model`. Align each bone
  onto a fixed reference, fit an NSM, convert to OpenSim coordinates, and assemble the
  subject-specific COMAK model.
- **Synthetic-decode pipeline** — `latent → OpenSim model`. Decode an arbitrary latent
  (synthetic joints, shape-mode visualization, latent interpolation) straight into OpenSim
  coordinates, then assemble with the *same* model-building code.

Both pipelines pass through four coordinate spaces (MRI → REFALIGN → NSMcanon → OSIM) and
share the [`build_joint_model`][nsosim.model_building.build_joint_model] assembler. The
full transform chain — with which converter is used where, and the scale identity of every
hand-off — is documented in **[Coordinate systems & pipeline](coordinate-systems.md)**.
**Read that page before touching the scaling or conversion code.**

A separate **COMAK body scaling** step ([`nsosim.scaling`](reference/scaling.md)) sizes a
whole-body COMAK model to a subject by applying AddBiomechanics-derived scale factors and
masses; where it meets the knee build (and the active bug there) is covered in the
coordinate-systems page §5, with the full set of sizing modes (and the active bug) in
**[Knee sizing modes](deviations.md)**.

---

## Module map

| Module | Purpose |
|---|---|
| [`nsm_fitting`](reference/nsm_fitting.md) | NSM fitting pipeline (mesh→latent) **and** the coordinate-conversion functions (REFALIGN/NSMcanon/OSIM). |
| [`decode`](reference/decode.md) | Decode arbitrary latents → OSIM-space meshes (synthetic joints, shape modes). The inverse of fitting. |
| [`transforms`](reference/transforms.md) | Similarity-transform math: decomposition, relative transforms (`T_rel`), mean rotation, deviation analysis/recomposition. |
| [`model_building`](reference/model_building.md) | Assemble OSIM-space meshes → subject-specific OpenSim model. Shared by both pipelines. |
| [`articular_surfaces`](reference/articular_surfaces.md) | Extract/refine cartilage contact surfaces, meniscus processing, prefemoral fat pad, patella optimization. |
| [`meniscal_ligaments`](reference/meniscal_ligaments.md) | Project meniscal-ligament tibia attachments onto the tibia surface. |
| [`comak_osim_update`](reference/comak_osim_update.md) | Update the OpenSim XML with subject-specific meshes, attachments, wrap surfaces. |
| [`osim_utils`](reference/osim_utils.md) | Low-level OpenSim XML manipulation via the Python API. |
| [`utils`](reference/utils.md) | NSM model loading, mesh I/O, anatomical-coordinate-system (ACS) alignment helpers. |
| [`schemas`](reference/schemas.md) | Input validation (`dict_bones`, surface indices). |
| [`scaling`](reference/scaling.md) | **COMAK body scaling** — size a COMAK base model to a subject from AddBiomechanics outputs (`s_wa`, knee-geometry bake, per-body mass transfer). Called "Stage X" in the scaling code. |
| [`wrap_surface_fitting`](reference/wrap_surface_fitting.md) | PyTorch SDF optimization to adapt OpenSim wrap surfaces (cylinders, ellipsoids, patella) to new bone geometry. |

---

## Pipeline stages (MRI build)

1. **NSM model fitting** — [`align_knee_osim_fit_nsm`][nsosim.nsm_fitting.align_knee_osim_fit_nsm]
   aligns femur→tibia→patella (femur transform reused), fits an NSM per bone.
2. **Per-bone processing** — [`nsm_recon_to_osim`][nsosim.nsm_fitting.nsm_recon_to_osim]
   to OSIM coords, then articular-surface extraction, ligament-attachment interpolation,
   and wrap-surface fitting.
3. **Meniscus processing** — upper/lower articulating surfaces (and optional radial-envelope
   refinement).
4. **Prefemoral fat pad** — contact surface between femur and patella cartilage.
5. **OpenSim model update** — [`update_osim_model`][nsosim.comak_osim_update.update_osim_model]
   writes meshes, attachments, and wrap surfaces into the `.osim`.

The detailed, copy-pasteable stage examples live in the repo-root `CLAUDE.md` ("Complete
Pipeline Workflow"); this site focuses on the architecture and the per-function reference.

---

## How the library is driven in production

`nsosim` is a library; the orchestration that calls it lives in the **`comak_gait_simulation`**
repo (not here). The scripts below are the worked examples of intended use — each maps to one
of the [knee sizing modes](deviations.md):

| Driver (in `comak_gait_simulation`) | What it does | Mode |
|---|---|---|
| `comak_1_nsm_fitting.py` | Canonical MRI→model build: `align → recon → assemble` against a reference-size base model. | [1](deviations.md#mode-1-personalized-knee-unscaled-reference-size-model) |
| `comak_1_nsm_model_run.py` + `submit_nsm_slurm_job.py` | Queue/batch driver for the fit. **With `--config stage_y.json` it is the multigait knee-build driver** (runs against a body-scaled base). | [3](deviations.md#mode-3-personalized-knee-scaled-to-the-gait-body) |
| `multigait/prepare_gait_subject.py` | Runs COMAK body scaling, writes the Stage-Y config (base = the scaled model), launches the runner. | sets up [3](deviations.md#mode-3-personalized-knee-scaled-to-the-gait-body) |
| `comak_1_synthetic.py` | Builds a model from latents via [`nsosim.decode`](reference/decode.md) (no MRI). | [5](deviations.md#mode-5-synthetic-knee-scaled-to-a-model) |

!!! note "Production calls may override library defaults"
    The library's defaults are not always what production selects. For example, the MRI driver
    calls `build_joint_model(..., project_coronary=False)` even though the library default is
    `True`. When tracing behavior, check the actual call site in `comak_gait_simulation`, not
    just the nsosim default.

---

## Determinism

The fitting and decode pipelines are deterministic by default (`seed=0` on the public entry
points: [`fit_nsm`][nsosim.utils.fit_nsm],
[`align_knee_osim_fit_nsm`][nsosim.nsm_fitting.align_knee_osim_fit_nsm],
[`nsm_recon_to_osim`][nsosim.nsm_fitting.nsm_recon_to_osim],
[`decode_latent_to_osim`][nsosim.decode.decode_latent_to_osim],
[`build_joint_model`][nsosim.model_building.build_joint_model]). Two runs with the same
seed and inputs produce bit-identical mesh coordinates. Pass `seed=None` to opt out.

---

## Where to go next

- **[Coordinate systems & pipeline](coordinate-systems.md)** — the spaces, the transform
  chain, and the COMAK-body-scaling ↔ knee-build interaction. The most important page.
- **[Knee sizing modes](deviations.md)** — the ways to size a knee placed into a model
  (unscaled-model / generic / gait-scaled / true-size / synthetic), how each is achieved, and
  which are built
  (including the active reference-size-knee bug).
- **API reference** — per-module docstrings (left nav).
