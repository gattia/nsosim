"""JSON sidecar reporting for the Stage Z weld collapse."""

import json
from pathlib import Path

__all__ = ["write_report"]


def write_report(report_json, report: dict) -> Path:
    """Write the aggregate weld-collapse report to ``report_json`` as JSON.

    Creates parent directories if needed. Returns the path written.
    """
    report_json = Path(report_json)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report_json
