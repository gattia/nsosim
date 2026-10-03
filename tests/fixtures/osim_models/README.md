# OpenSim model fixtures

| file | model | what it is |
|---|---|---|
| `full_body_healthy_knee.osim` | `smith2019`, 27 bodies, 39 markers | COMAK reference model (the scaling base) |
| `full_body_healthy_knee_trc_markers.osim` | `smith2019`, 27 bodies, 49 markers | COMAK with the Tian TRC markers added |
| `full_body_healthy_knee_armless_trc.osim` | `smith2019`, 19 bodies, 41 markers | COMAK without arms, the 41 Tian markers renamed in place (`scripts/phase6_rajagopal_audit/build_smith2019_armless_trc.py`). **This is the model to strip and upload to AddBiomechanics (AB).** |
| `unscaled_generic_tian.osim` | **`ArmlessRajagopal`**, 14 bodies, 41 markers | **Not COMAK, despite the name.** AB's stock generic with the Tian marker names: the skeleton the Paper-1 v2 AB runs actually scaled. |

The models with contact meshes (`smith2019`) load only with their `Geometry/` folder beside them. That folder is
untracked (large STLs).

## `unscaled_generic_tian.osim` is ArmlessRajagopal

Earlier docs, including `.claude/plans/completed/comak-body-scaling_COMPLETED.md`, called this file "our stripped,
marker-renamed COMAK base, uploaded to AB". It is not. It has the same skeleton as the `Models/unscaled_generic.osim`
in every v2 AB output (`ae509ca6…`; the bytes differ, the joints do not):

| landmark (m) | this file | COMAK |
|---|---:|---:|
| femur, hip → knee centre | 0.4080 | 0.3742 |
| tibia, knee → ankle centre | 0.3965 | 0.4033 |
| hip half-width | 0.0773 | 0.0924 |

It carries the same 41 marker names as the COMAK upload, so nothing errored when AB scaled it in place of COMAK.
`nsosim.scaling.validate.check_ab_generic` now catches exactly this mix-up (`tests/scaling/test_validate.py`).

The scaling tests still use this file as a source of AB-shaped factors (identity, synthetic 0.9). They set the
factors themselves, so its skeleton does not matter there.

To build the real COMAK upload skeleton, run the comak clean repo's `pipeline/prepare_addbiomechanics_skeleton.py`
on `full_body_healthy_knee_armless_trc.osim`. Do not use `nsosim.knee_assembly.strip_comak_knee` alone: at tag
`pre-rerun-v3` it places the stripped knee's tibia frame 12 mm off and reverses the flexion sign.
