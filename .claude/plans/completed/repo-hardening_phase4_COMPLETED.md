# Phase 4: Code Hygiene — COMPLETED

**Date:** 2026-02-18
**Result:** 156 passed, 0 failed (28.61s, `conda run -n comak python -m pytest tests/ -v`)
**Commit:** `15b30e9`

---

## What was done vs. the plan

### 4.1 Remove deprecated `wraps.py` — DONE

| Planned | Done | Notes |
|---------|------|-------|
| Remove `wraps` from `__init__.py` import | Yes | Line 3 |
| Remove `'wraps'` from `__all__` | Yes | Line 14 |
| Delete `nsosim/wraps.py` | Yes | 385 lines removed |

Grep confirmed nothing in the codebase imported from `wraps`. Safe deletion.

### 4.2 Fix `print()` → `logging` — DONE

| File | Lines changed | Notes |
|------|---------------|-------|
| `nsm_fitting.py` | 765-766 → 1 `logger.error()` call | In `check_icp_transform()`, before `raise ValueError` |
| `utils.py` | 187, 269, 271, 402 → 4 logging calls | Added `import logging` + `logger = logging.getLogger(__name__)` |

### 4.3 Promote magic numbers to named constants — DONE

| File | Constant | Value | Used in |
|------|----------|-------|---------|
| `utils.py` | `BONE_CLIPPING_FACTOR` | `0.95` | `clip_bone_end()` |
| `articular_surfaces.py` | `DEFAULT_MENISCUS_SDF_THRESHOLD` | `0.1` | `refine_meniscus_articular_surfaces()`, `create_meniscus_articulating_surface()` |
| `articular_surfaces.py` | `DEFAULT_RADIAL_PERCENTILE` | `95.0` | Same as above |
| `articular_surfaces.py` | `DEFAULT_TRIANGLE_DENSITY` | `4_000_000` | `create_meniscus_articulating_surface()`, `create_articular_surfaces()` |
| `articular_surfaces.py` | `DEFAULT_FATPAD_BASE_MM` | `0.5` | `create_prefemoral_fatpad_noboolean()` |
| `articular_surfaces.py` | `DEFAULT_FATPAD_TOP_MM` | `6.0` | Same |
| `articular_surfaces.py` | `DEFAULT_MAX_DISTANCE_TO_PATELLA_MM` | `30.0` | Same |

### 4.4 Resolve or expand TODOs — DONE

| File:Line | Original | Resolution |
|-----------|----------|------------|
| `nsm_fitting.py:112` | `# TODO: get rid of this extra alignment transform later?` | Expanded: NOTE explaining ACS alignment is legacy, skippable with `acs_align=False` |
| `nsm_fitting.py:171` | `# TODO: Update NSM recon function to` | Completed: "accept mesh objects directly instead of file paths" |
| `comak_osim_update.py:94` | `# TODO: update dictionaries to include parent/child info` | Expanded: target structure with `'parent_body'` key example |

### 4.5 Fix deprecated `torch.cross()` — DONE

| File | Line | Change |
|------|------|--------|
| `wrap_surface_fitting/fitting.py` | 1493 | `torch.cross()` → `torch.linalg.cross()` |

---

## Things to know for future work

### No autoformat changes needed

All Phase 4 edits were already black/isort compliant. No separate formatting commit was required.

### `wraps.py` `wrap_surface` class vs `wrap_surface_fitting/main.py`

The deleted `wraps.py` contained an older `wrap_surface` class. The canonical version is in `nsosim/wrap_surface_fitting/main.py`. They have the same attribute names but the new version includes `to_dict()` and is used by all current code.

### Constants are importable

The new constants in `articular_surfaces.py` and `utils.py` can be imported by downstream code:
```python
from nsosim.articular_surfaces import DEFAULT_TRIANGLE_DENSITY
from nsosim.utils import BONE_CLIPPING_FACTOR
```
