# Plan: Explicit Mesh Name Mapping for NSM Decoder Outputs

**Created:** 2026-03-26
**Status:** Complete (2026-03-27)
**Priority:** Medium — not blocking current work, but should be done before adding new model variants

---

## Problem

NSM decoders return a positional list of meshes. Index 0 is always bone, index 1 is always cartilage, but indices 2+ are ambiguous. The codebase currently infers mesh identity from the count:

```python
# nsosim/utils.py:348-362 (recon_mesh)
n_meshes = len(mesh_result["mesh"])
if n_meshes == 2:
    pass  # bone + cart
elif n_meshes == 3:
    output_dict["fibula_mesh"] = mesh_result["mesh"][2]
elif n_meshes == 4:
    output_dict["med_men_mesh"] = mesh_result["mesh"][2]
    output_dict["lat_men_mesh"] = mesh_result["mesh"][3]
```

This works today only because each `(bone_type, objects_per_decoder)` combination happens to be unique across all current models. But:

- **It will break** if a new model variant returns 3 meshes that aren't `[bone, cart, fibula]` (e.g., a femur model with bone + cart + combined meniscus)
- **It will break** if a tibia model adds meniscus outputs (count=4, but meaning is different from femur's count=4)
- The mapping is **duplicated** — `recon_mesh()` has it, and the new `decode_latent_to_osim()` in Phase C would need to replicate it
- The assumption is **implicit** — nothing in the model config says "mesh index 2 is the medial meniscus"

### Current model configs in production

| Model | `bone` | `objects_per_decoder` | Actual mesh order |
|-------|--------|-----------------------|-------------------|
| 568_nsm_femur_bone_cart_men | femur | 4 | bone, cart, med_men, lat_men |
| 647_nsm_femur | femur | 2 | bone, cart |
| 231_nsm_femur_cartilage | femur | 2 | bone, cart |
| 650_nsm_tibia | tibia | 2 | bone, cart |
| 648_nsm_patella | patella | 2 | bone, cart |
| 75_nsm_tibia_cartilage (legacy) | femur | 2 | bone, cart |
| 77_nsm_patella_cartilage (legacy) | femur | 2 | bone, cart |

Note: models 75 and 77 have `bone: "femur"` despite being tibia/patella cartilage models — these are legacy configs that predate the `bone` field convention.

---

## Proposed Fix

### Add `mesh_names` to model config with auto-population fallback

**Step 1: Add a `get_mesh_names(model_config)` helper in `nsosim/utils.py`**

```python
# Fallback mapping for configs that don't have mesh_names.
# Keyed by (bone_type, objects_per_decoder).
_DEFAULT_MESH_NAMES = {
    ("femur", 2): ["bone", "cart"],
    ("femur", 4): ["bone", "cart", "med_men", "lat_men"],
    ("tibia", 2): ["bone", "cart"],
    ("tibia", 3): ["bone", "cart", "fibula"],
    ("patella", 2): ["bone", "cart"],
}

def get_mesh_names(model_config):
    """Return ordered mesh names for this model's decoder outputs.

    Reads mesh_names from model_config if present. Otherwise infers
    from (bone, objects_per_decoder) using the legacy heuristic and
    logs a deprecation warning.
    """
    if "mesh_names" in model_config:
        return model_config["mesh_names"]

    bone = model_config.get("bone", "unknown")
    n_objects = model_config.get("objects_per_decoder", 1)
    key = (bone, n_objects)

    if key not in _DEFAULT_MESH_NAMES:
        raise ValueError(
            f"No default mesh_names for (bone={bone!r}, objects_per_decoder={n_objects}). "
            f"Add a 'mesh_names' list to your model config JSON."
        )

    logger.warning(
        "model config missing 'mesh_names' — inferring from (bone=%r, objects=%d). "
        "Add 'mesh_names': %s to your config to silence this warning.",
        bone, n_objects, _DEFAULT_MESH_NAMES[key],
    )
    return _DEFAULT_MESH_NAMES[key]
```

**Step 2: Refactor `recon_mesh()` to use `get_mesh_names()`**

Replace the count-based branching (lines 340-362) with:

```python
mesh_names = get_mesh_names(model_config)
assert len(mesh_result["mesh"]) == len(mesh_names), (
    f"Decoder returned {len(mesh_result['mesh'])} meshes but mesh_names "
    f"has {len(mesh_names)} entries: {mesh_names}"
)
output_dict = {}
for name, mesh in zip(mesh_names, mesh_result["mesh"]):
    output_dict[f"{name}_mesh"] = mesh
output_dict["mesh_result"] = mesh_result
output_dict["latent"] = latent
```

**Step 3: Use `get_mesh_names()` in `decode_latent_to_osim()` (Phase C)**

The new decode function uses the same helper to name its output meshes, eliminating the duplication.

**Step 4 (optional): Update model config JSONs**

Add `mesh_names` to each config to suppress the deprecation warning. This is a one-line addition per file and can be done at any time — the fallback handles it until then.

---

## Impact Analysis

### Files that need changes

| File | Change | Risk |
|------|--------|------|
| `nsosim/utils.py` | Add `get_mesh_names()`, refactor `recon_mesh()` | Medium — `recon_mesh` is used in production pipeline |
| `nsosim/decode.py` (new, Phase C) | Use `get_mesh_names()` instead of count heuristic | Low — new code |
| `nsosim/nsm_fitting.py:258-285` | Update to use new dict key names from `recon_mesh` | Medium — downstream of `recon_mesh` |
| `nsosim/nsm_fitting.py:909-954` (`nsm_recon_to_osim`) | Already uses string key lookups (`"med_men_mesh_nsm" in ...`), no change needed if key names stay the same | None |

### Key constraint: output dict key names must stay the same

The current `recon_mesh()` output uses keys like `bone_mesh`, `cart_mesh`, `med_men_mesh`, `lat_men_mesh`, `fibula_mesh`. These are consumed by `nsm_fitting.py:258-285` which maps them to `bone_mesh_nsm`, `cart_mesh_nsm`, etc. in `dict_bones`.

The refactored version produces `{name}_mesh` from `mesh_names` — so `mesh_names=["bone", "cart", "med_men", "lat_men"]` produces the same keys: `bone_mesh`, `cart_mesh`, `med_men_mesh`, `lat_men_mesh`. **No downstream key changes needed** as long as the mesh_names values match the current naming convention.

### What stays the same

- `nsm_recon_to_osim()` — already does string-based key lookups on `dict_bones[bone]["subject"]`, no count heuristic
- `nsm_fitting.py:397-430` (mesh saving) — same string-based key lookups
- All test files — test the public API, not internal mesh ordering

---

## Things to Consider

1. **Legacy configs (75, 77) have wrong `bone` field.** They say `bone: "femur"` but are tibia/patella cartilage models. The fallback lookup would still work (femur+2 → bone+cart), but the warning message would be misleading. These models may not be actively used — verify before worrying about this.

2. **`mesh_names` validation.** Should `get_mesh_names()` validate that `len(mesh_names) == objects_per_decoder`? Probably yes — catches config typos early.

3. **First entry must always be "bone".** Several places assume `mesh_result["mesh"][0]` is the bone. With named mapping this assumption becomes `mesh_names[0] == "bone"`. Could add a validation check, or just document it.

4. **Interaction with `recon_mesh` output format.** Currently `recon_mesh` returns a flat dict. The refactored version produces the same flat dict shape. An alternative would be returning `{"meshes": {"bone": mesh, "cart": mesh, ...}, "latent": ..., "mesh_result": ...}` but that's a larger API change — defer unless there's a reason to do it.

5. **NSM library coupling.** The NSM `create_mesh()` and `reconstruct_mesh()` return positional lists. This plan adds the name mapping in nsosim, not in the NSM library. If NSM ever adds its own naming, we'd want to use that instead. For now, nsosim is the right place since NSM is a general-purpose library and doesn't know about anatomical semantics.

6. **Testing.** Add a test for `get_mesh_names()` covering: config with explicit `mesh_names`, config without (triggers fallback + warning), config with unknown `(bone, count)` combo (raises ValueError), config where `len(mesh_names) != objects_per_decoder` (raises ValueError).

---

## Completion Notes (2026-03-27)

All steps implemented:

1. **`get_mesh_names()` + `_DEFAULT_MESH_NAMES`** added to `nsosim/utils.py`. Commit `29a32c2`.
2. **`recon_mesh()` refactored** — replaced 25-line count-based branching with 7-line `get_mesh_names()` loop. Same output dict keys, no downstream breakage. Commit `29a32c2`.
3. **`decode_latent_to_osim()`** — not yet written (Phase C of synthetic-joint-decode plan), will use `get_mesh_names()` from the start.
4. **Model config JSONs updated** — all 7 production configs now have `mesh_names` field. No more fallback warnings for existing models.
5. **NSM library updated** — `mesh_names` accepted as optional config parameter in both `train_deep_sdf.py` and `train_deep_sdf_multi_head.py`, with validation and warning. Commit `709b818` on `adaptive_marching_cubes` branch.
6. **Tests** — 14 tests in `tests/test_mesh_names.py`, all passing. Full suite (272 tests) passes with no regressions.

### Resolved considerations
- Item 1 (legacy configs): Legacy models 75/77 have `bone: "femur"` which is wrong, but `mesh_names: ["bone", "cart"]` is now explicit so the `bone` field no longer matters for mesh naming.
- Item 2 (validation): Yes, `get_mesh_names()` validates `len(mesh_names) == objects_per_decoder`.
- Item 5 (NSM coupling): NSM now accepts `mesh_names` as a passthrough config field. It validates length but doesn't interpret the names — semantics stay in nsosim.
- Item 6 (testing): All cases covered in `test_mesh_names.py`.

### Still open
- Item 3 (first entry = "bone"): Not enforced. Documenting convention is sufficient for now.
- Item 4 (output format): Kept flat dict for backwards compat. Revisit if needed.
