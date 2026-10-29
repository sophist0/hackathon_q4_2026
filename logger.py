"""Logger: persist per-prompt runner output for the comparator.

One row per prompt per backend, kept as JSONL (primary) with a CSV export
that reuses the ground-truth ``Prompt | ... | Expected`` table shape.
Backend runs share the same file layout with a ``backend`` column, so the
comparator can place backend columns side-by-side.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

CSV_COLUMNS: tuple[str, ...] = (
    "row_id",
    "prompt_text",
    "response",
    "backend",
    "path",
    "answers",
    "confidences",
    "predicted_label",
    "stop_step",
    "expected",
    "category",
)


def format_row(result: Mapping[str, Any], backend: str) -> dict[str, str]:
    """Flatten one runner result dict to a string-only log row."""
    return {
        "row_id": str(result.get("row_id", "")),
        "prompt_text": str(result["prompt_text"]),
        "response": str(result.get("response", "")),
        "backend": backend,
        "path": ">".join(result["path"]),
        "answers": ",".join(result["answers"]),
        "confidences": ",".join(str(confidence) for confidence in result["confidences"]),
        "predicted_label": str(result["predicted_label"]),
        "stop_step": str(result["stop_step"]),
        "expected": str(result.get("expected", "")),
        "category": str(result.get("category", "")),
    }


def write_jsonl(
    results: Iterable[Mapping[str, Any]],
    backend: str,
    path: str | Path,
    append: bool = False,
) -> Path:
    """Write one JSON object per line; ``append`` adds another backend's run."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a" if append else "w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(format_row(result, backend)) + "\n")
    return target


def read_jsonl(path: str | Path) -> list[dict[str, str]]:
    """Read back rows written by :func:`write_jsonl`."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(
    results: Iterable[Mapping[str, Any]],
    backend: str,
    path: str | Path,
) -> Path:
    """Export rows to CSV using the shared log column layout."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [format_row(result, backend) for result in results]
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return target
