# CI/CD Setup Plan

Extracted from repo-hardening Phase 6 and post-review-hardening Phase 8.

**Prerequisites:** Before executing, confirm with the user: private vs public repo, target Python version, whether self-hosted runners are available.

**Context constraints:**
- OpenSim is custom-built from source (JAM/COMAK fork). GitHub Actions CI **cannot** run integration tests that touch `opensim` or the full NSM→OSIM pipeline.
- CI should be lint-only + unit tests that don't require opensim or real mesh data.

---

## Phase 1: GitHub Actions Workflow

### 1.1 Add/fix `.github/workflows/ci.yml`
- Trigger on push and PR to `main`
- Steps: install deps, `make lint`, `pytest`
- Use a lightweight runner (no GPU needed)
- Add a comment block at top documenting CI limitations (lint + unit tests only, no opensim or real data)
- Python version: 3.10+ for CI (even though dev env is 3.9 — tests don't need opensim)

### 1.2 Add pre-commit hooks (optional)
- `.pre-commit-config.yaml` with `black`, `isort`, basic file checks
- Keeps formatting consistent without manual `make autoformat`

---

## Nice-to-Have (Low Priority)

These are from post-review-hardening Phase 11 — do after validating current code with real pipeline scripts:

- **Decompose `CylinderFitter.fit()`** (347 lines) into `_adam_training_loop()`, `_lbfgs_refinement_stage()`, `_validate_final_rotation()`
- **Add degenerate input tests**: NaN inputs, all-identical points, collinear points, contradictory labels, empty point clouds
- **Add end-to-end integration test**: Label → fit → validate SDF pipeline test with synthetic data
