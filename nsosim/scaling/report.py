"""JSON sidecar for Stage X."""

import json
from pathlib import Path
from typing import Any


def write_report(path: Path, **fields: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(fields, f, indent=2, default=str)
