"""Wrap OpenSim ScaleTool.run() for COMAK body scaling.

Uses ScaleTool (which writes XML and reloads) instead of Model.scale() because
the in-memory scale path crashes on JAM Smith2018ContactMesh init.
"""

from pathlib import Path
from typing import Optional

import opensim as osim


def apply_scaletool(
    base_osim: Path,
    scale_set: osim.ScaleSet,
    out_osim: Path,
    preserve_mass_distribution: bool = True,
) -> None:
    """Run ScaleTool with the given ScaleSet and write the scaled model.

    The ``scale_set`` carries dimensionless per-body scale factors (ratios; AB's
    per-axis factors for AB-provided bodies, isotropic ``s_wa`` for the knee
    subbodies). ScaleTool.run() applies them to the model's geometry in OSIM
    space (meters): each body's ``attached_geometry`` ``scale_factors`` is set,
    the joint frame translations are scaled, and the JAM components
    (Blankevoort1991Ligament, Smith2018ContactMesh, all wraps and muscles) are
    rescaled by ScaleTool's extendPostScale hooks. MarkerPlacer is disabled —
    markers are transplanted from the AB model in a later step.

    Frame/translation effects (empirically verified):
        - The knee weld translation (``femur_r → femur_distal_r`` weld, parent
          translation ≈ (-0.0056, -0.3742, -0.0012) m) is scaled per-axis by the
          PARENT bone's AB factors (femur_r's sx, sy, sz) — NOT by ``s_wa``. So
          the knee subbody is positioned down the shaft by AB's anisotropic
          parent-bone scaling, while its geometry is sized isotropically by
          ``s_wa`` in ``bake_knee_geometry``.
        - The knee CustomJoint frames (knee_r, pf_r, and the two meniscus
          joints) sit at body-local origin (0, 0, 0), so scaling leaves them at
          origin.

    Mass / inertia (this function uses ``preserveMassDist=True`` and does NOT
    call ``setSubjectMass``):
        - Body mass (kg) is left UNCHANGED.
        - Inertia (kg·m²) is scaled geometrically by scale² (e.g. an isotropic
          factor of 0.9 → inertia × 0.81, mass × 1.0).
    The orchestrator's ``_apply_per_body_masses`` post-pass then sets the real
    AB per-body physics-tuned masses and rescales inertia by the mass ratio,
    rather than uniformly scaling all bodies by total_subject_mass /
    total_base_mass (which is what ScaleTool.setSubjectMass +
    preserveMassDist=True would do).
    """
    base_osim = Path(base_osim)
    out_osim = Path(out_osim)
    out_osim.parent.mkdir(parents=True, exist_ok=True)

    st = osim.ScaleTool()
    st.getGenericModelMaker().setModelFileName(str(base_osim))

    ms = st.getModelScaler()
    ms.setApply(True)

    order = osim.ArrayStr()
    order.append("manualScale")
    ms.setScalingOrder(order)

    target = ms.getScaleSet()
    for i in range(scale_set.getSize()):
        target.adoptAndAppend(scale_set.get(i).clone())

    ms.setPreserveMassDist(preserve_mass_distribution)
    ms.setOutputModelFileName(str(out_osim))

    st.getMarkerPlacer().setApply(False)

    st.run()
