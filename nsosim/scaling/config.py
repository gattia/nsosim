"""Constants for COMAK body scaling.

WA_KNEE_BODIES are the subject-specific knee bodies: they receive the isotropic
weighted-average factor, and all their geometry is baked into its STLs by
bake_knee_geometry (rather than scaled via OpenSim `scale_factors`).
"""

from typing import Literal, Tuple

ScalingMode = Literal["WA", "LA", "AB"]

# How body masses are set after ScaleTool (scale_comak_model's ``mass_rule``):
#   "as_run" -- the Paper-1 v2 rule (_apply_per_body_masses): AB's per-body masses imported for the bodies AB
#               scaled, a global ratio for the rest, everything renormalised to the total. On a COMAK model this
#               counts the bodies AB does not model twice (v2: x0.848). Kept so v2 can be reproduced exactly.
#   "volume" -- OpenSim's own rule, which AB's masses follow exactly: m_i = base_i * (sx*sy*sz)_i * c, with c set
#               so the total is the subject mass. Bodies with no scale (arms on an armless AB skeleton) keep
#               volume factor 1; the knee bodies take s_wa**3. Paper-1 rerun v3, decisions D4a/D4b.
MassRule = Literal["as_run", "volume"]

# Index into a body's (x, y, z) scale-factor triplet for the long
# (superior-inferior) axis. In the OpenSim femur_r / tibia_r body frames
# x = anterior-posterior, y = superior-inferior, z = mediolateral -- verified by
# mesh extents (r_femur.vtp: x 7.0 cm, y 45.3 cm, z 10.0 cm) and by the generic
# model's joint offsets (knee at y = -0.408 m in the femur frame, ankle at
# y = -0.400 m in the tibia frame). The long axis is therefore index 1.
#
# HISTORY -- this was 2 from the file's birth (69acaa0, 2026-05-15) until
# 2026-07-25. The 2 came from the prior-art scaleModel.py, which did
# float(scale_str.split(' ')[2]); OpenSim serializes Vec3 with a LEADING SPACE, so
# that split yields ['', x, y, z] and its [2] is y. Porting it to a 0-based tuple
# silently moved the selection from y to z (mediolateral). That prior pipeline
# provably used y: its own output scaledLenhart.xml has knee bodies = 1.0482,
# which is the mean of the y factors exactly (x -> 1.06110, z -> 1.07719).
#
# CONSEQUENCE -- every OARSI_multigait_*_bts_v1 cohort was built with 2, i.e.
# scaled by the mean MEDIOLATERAL factor. Those results are NOT retroactively
# corrected by this change; anything rebuilt now will differ from them (up to
# 6.6% linear / ~14% in contact area, for Subject_102). Do not mix cohorts built
# on either side of this boundary.
LONG_AXIS_INDEX: int = 1

WA_KNEE_BODIES: Tuple[str, ...] = (
    "femur_distal_r",
    "tibia_proximal_r",
    "patella_r",
    "meniscus_medial_r",
    "meniscus_lateral_r",
)
