"""Wrap OpenSim ScaleTool.run() for COMAK body scaling.

Uses ScaleTool (which writes XML and reloads) instead of Model.scale() because
the in-memory scale path crashes on JAM Smith2018ContactMesh init.
"""

from pathlib import Path

import opensim as osim


def apply_scaletool(
    base_osim: Path,
    scale_set: osim.ScaleSet,
    out_osim: Path,
    preserve_mass_distribution: bool = True,
) -> None:
    """Run ScaleTool with the given ScaleSet and write the scaled model.

    JAM components (Blankevoort1991Ligament, Smith2018ContactMesh, all wraps
    and muscles) are handled by ScaleTool's extendPostScale hooks. MarkerPlacer
    is disabled — markers are transplanted from the AB model in a later step.
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
