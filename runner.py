"""Runner: walk prompts down the classification tree.

Traversal policy lives here; single questions are delegated to
``decider.decide`` (via ``decider.walk``). The tree file
(``question_tree.yml``) is authoritative: stop behavior falls out of its
``on_Y``/``on_N`` transitions (e.g. ``N`` at ``q1`` ends the chain because
``q1.on_N`` is ``end_Unknowable``), not from hardcoded question ids.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from decider import (
    BackendName,
    Transport,
    TreeSpec,
    WalkResult,
    walk,
)

GROUND_TRUTH_COLUMNS = {
    "prompt_text": "Prompts",
    "expected": "Expected Request Response",
    "category": "Categorization",
}


def load_ground_truth_rows(csv_path: str | Path) -> list[dict[str, str]]:
    """Load ground-truth rows, mapping CSV columns to runner row keys.

    Blank lines (the CSV contains an empty ``|||||`` filler line) are skipped.
    """
    rows: list[dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle, delimiter="|"):
            prompt_text = (record.get(GROUND_TRUTH_COLUMNS["prompt_text"]) or "").strip()
            if not prompt_text:
                continue
            rows.append(
                {
                    "prompt_text": prompt_text,
                    "expected": (record.get(GROUND_TRUTH_COLUMNS["expected"]) or "").strip(),
                    "category": (record.get(GROUND_TRUTH_COLUMNS["category"]) or "").strip(),
                }
            )
    return rows


def run_prompt(
    prompt_row: Mapping[str, str],
    tree: TreeSpec,
    backend: BackendName,
    transport: Transport,
    threshold: float = 0.5,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Walk one ground-truth row down the tree (at most ``max_depth`` calls)."""
    result: WalkResult = walk(
        tree,
        prompt_row["prompt_text"],
        backend,
        transport,
        threshold=threshold,
        base_dir=base_dir,
    )
    return {
        "prompt_text": prompt_row["prompt_text"],
        "expected": prompt_row.get("expected", ""),
        "category": prompt_row.get("category", ""),
        "answers": result.answers,
        "confidences": result.confidences,
        "path": result.path,
        "stop_step": result.stop_step,
        "predicted_label": result.predicted_label,
    }


def run_all(
    rows: Iterable[Mapping[str, str]],
    tree: TreeSpec,
    backend: BackendName,
    transport: Transport,
    threshold: float = 0.5,
    base_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Walk every row down the tree, keeping ground truth attached for the logger."""
    return [
        run_prompt(row, tree, backend, transport, threshold=threshold, base_dir=base_dir)
        for row in rows
    ]
