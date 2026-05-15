"""Transplant AB's static-trial-IK MarkerSet onto the scaled COMAK model.

AB ingested our marker-renamed base, so its markers carry the names downstream
COMAK-IK matches against Tian's TRC. Keeping them verbatim preserves AB's
static-trial placement.
"""

from pathlib import Path
from typing import Optional, Tuple

import opensim as osim


def swap_markerset(
    scaled_osim: Path,
    ab_scaled_osim: Path,
    backup_xml_path: Optional[Path] = None,
) -> Tuple[int, int]:
    """Replace `scaled_osim`'s MarkerSet with the one from `ab_scaled_osim`.

    Markers referencing a parent body that doesn't exist in `scaled_osim` are
    dropped with a warning logged. Returns (n_added, n_dropped).
    """
    scaled_osim = Path(scaled_osim)
    ab_scaled_osim = Path(ab_scaled_osim)

    scaled = osim.Model(str(scaled_osim))
    ab = osim.Model(str(ab_scaled_osim))
    scaled.initSystem()
    ab.initSystem()

    if backup_xml_path is not None:
        backup_xml_path = Path(backup_xml_path)
        backup_xml_path.parent.mkdir(parents=True, exist_ok=True)
        scaled.getMarkerSet().printToXML(str(backup_xml_path))

    ms = scaled.updMarkerSet()
    for i in range(ms.getSize() - 1, -1, -1):
        ms.remove(i)

    body_names = {
        scaled.getBodySet().get(i).getName() for i in range(scaled.getBodySet().getSize())
    }

    ab_ms = ab.getMarkerSet()
    n_added = 0
    n_dropped = 0
    warnings = []
    for i in range(ab_ms.getSize()):
        m = ab_ms.get(i)
        parent_name = m.getParentFrame().getName()
        if parent_name not in body_names:
            warnings.append(
                f"AB marker {m.getName()!r} references missing body " f"{parent_name!r}; dropping."
            )
            n_dropped += 1
            continue
        new_m = osim.Marker(
            m.getName(),
            scaled.getBodySet().get(parent_name),
            m.get_location(),
        )
        new_m.set_fixed(m.get_fixed())
        scaled.addMarker(new_m)
        n_added += 1

    scaled.finalizeConnections()
    scaled.printToXML(str(scaled_osim))

    if warnings:
        import logging

        log = logging.getLogger(__name__)
        for w in warnings:
            log.warning(w)

    return n_added, n_dropped
