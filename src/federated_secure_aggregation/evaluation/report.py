"""Bench plumbing: machine description, result files and regression thresholds.

``bench/thresholds.json`` maps each ``<bench>`` to ``<metric> -> {"max": x}`` and/or
``{"min": y}``. A metric outside its bounds is a regression, and a metric a threshold names but
the run did not produce is one too, so deleting a measurement cannot quietly disable its check.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "check_thresholds",
    "flatten",
    "load_thresholds",
    "machine_info",
    "markdown_table",
    "write_results",
]


def machine_info() -> dict[str, object]:
    """Where a measurement came from, so pasted README numbers stay attributable."""
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def load_thresholds(path: Path, bench: str) -> dict[str, dict[str, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    section = raw.get(bench)
    if section is None:
        raise KeyError(f"no thresholds defined for bench {bench!r} in {path.name}")
    if not isinstance(section, dict):
        raise TypeError(f"thresholds for {bench} must be an object")
    return {
        str(metric): {str(key): float(value) for key, value in limits.items()}
        for metric, limits in section.items()
    }


def check_thresholds(
    results: Mapping[str, float], thresholds: Mapping[str, Mapping[str, float]]
) -> list[str]:
    """Return one human-readable problem per violated or missing threshold."""
    problems: list[str] = []
    for metric in sorted(thresholds):
        limits = thresholds[metric]
        if metric not in results:
            problems.append(f"{metric}: missing from the results")
            continue
        value = results[metric]
        if not math.isfinite(value):
            problems.append(f"{metric} = {value} is not finite")
            continue
        if "max" in limits and value > limits["max"]:
            problems.append(f"{metric} = {value:.6g} exceeds max {limits['max']:g}")
        if "min" in limits and value < limits["min"]:
            problems.append(f"{metric} = {value:.6g} is below min {limits['min']:g}")
    return problems


def flatten(results: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """``{"a": {"p50": 1}} -> {"a.p50": 1}`` so thresholds address one number each."""
    return {
        f"{group}.{metric}": float(value)
        for group, stats in results.items()
        for metric, value in stats.items()
    }


def write_results(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """A markdown table whose float formatting is stable enough to paste into the README."""

    def cell(value: object) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            if not math.isfinite(value):
                return str(value)
            if value.is_integer() and abs(value) < 1e15:
                return str(int(value))
            if abs(value) < 10:
                return f"{value:.4f}"
            return f"{value:.2f}" if abs(value) < 1000 else f"{value:.1f}"
        return str(value)

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)
