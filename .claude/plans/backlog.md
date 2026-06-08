# Backlog — ideas & future work (not active plans)

A lightweight, **non-actionable** list so things aren't forgotten. No detailed steps here — when
one of these becomes real work, spin it into its own plan. Last touched: 2026-06-08.

## Knee sizing / scaling

- **Mode 4 — keep the knee at the subject's native MRI true size (Pathway C).** Not implemented.
  The hook exists in the code (the non-underscore `convert_nsm_recon_to_OSIM` takes a subject
  scale/center term; the MRI path currently never restores true size after the similarity
  registration divides it out). A "true-size" mode would use that hook. Design sketch only in
  `docs/deviations.md` Mode 4.
- **Production wiring (comak repo) — reorder Pathway-B to "build, then scale."** The library route
  is done & verified here; the external `comak_gait_simulation` scripts still build the knee on a
  body-scaled base (old order). Adopting build-then-scale there is the remaining integration step.
  (See the cross-repo handoff summary.)
- **Cross-check: build-then-scale ≈ fit-on-scaled-base. (VERY LOW PRIORITY — likely never.)**
  The idea was to confirm that scaling a finished reference-size knee gives the same result as
  *fitting* one at gait-body size. Two reasons this is parked indefinitely: (1) the "fit at scale"
  path **does not exist** — it's the pathway we deliberately chose not to build, so running the
  check means first writing it (scale the registration reference + the warp reference, then a full
  GPU fit per subject — exactly the cost build-then-scale avoids); (2) the two are **unlikely to
  match exactly anyway**, because the non-linear build steps (wrap SDF fit, articular ray-casting,
  the registration similarity-scale) don't necessarily commute with scaling. Crucially, a mismatch
  would **not** invalidate build-then-scale — that's the chosen, verified route (every component
  resizes by exactly `s_wa`); the check only asks "would the rejected alternative have agreed?"
  If anyone ever wants partial confidence cheaply (no GPU, no new pathway): scale the built recon
  meshes by `s_wa` and re-run only the deterministic OSIM-space surface/wrap builders on the scaled
  inputs, comparing to those surfaces scaled directly — that isolates the non-linear steps.
- **`LA` / `AB` scaling modes.** `build_scale_set` documents long-axis and anisotropic-passthrough
  modes for future resurrection; only `WA` (isotropic weighted-average) is implemented. `AB` would
  also require generalizing `bake_knee_geometry` to per-axis scaling.

## Test / infra

- **Promote the built-model fixture to GitHub Releases.** `tests/scaling/test_build_then_scale.py`
  end-to-end tests skip without a local `untracked/built_models/...` model (too large for git).
  A release-download (like the mesh fixtures) would let them run in CI.
- **Doc-reference guard locale fragility (fixed, note for awareness).** `test_doc_references.py`
  now pins `encoding="utf-8"`; the underlying cause (OpenSim XML I/O calling `setlocale()` and
  flipping the process default encoding to ASCII mid-run) can bite any future stdlib test that
  does a bare `read_text()`/`open()` and runs after the OpenSim tests.

## Cross-repo cleanup (comak_gait_simulation)

- **Correct or delete `SCALING_WORKFLOW_MAP.md`.** It is stale/partly-wrong (repoint+orphan, not
  overwrite-by-filename; Pathway-B wired via Step 1 + stage_y config). Replace its body with a
  one-line pointer to `nsosim/docs/coordinate-systems.md` + `deviations.md`, or delete it. **No
  banner** (owner preference).
- **Deprecate dead/unused scripts** once Pathway-B adopts build-then-scale (e.g. the orphaned
  Stage-X bone-bake path is dead weight under Pathway B). Audit before removing.

## Known issues (pre-existing, documented elsewhere)

- **Meniscus articular surface instability** — stochastic variation in the medial meniscus inferior
  surface (`MENISCUS_ARTICULAR_SURFACE_INSTABILITY.md`). Independent of the scaling work.
