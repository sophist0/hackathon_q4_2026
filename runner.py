"""Runner: walk prompts down the classification tree.

Traversal policy lives here; single questions are delegated to
``decider.decide`` (via ``decider.walk``). The tree file
(``question_tree.yml``) is authoritative: stop behavior falls out of its
``on_Y``/``on_N`` transitions (e.g. ``N`` at ``q1`` ends the chain because
``q1.on_N`` is ``end_Unknowable``), not from hardcoded question ids.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
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
    "row_id": "Row",
    "prompt_text": "Prompts",
    "response": "Response",
    "expected": "Expected Request Response",
    "category": "Categorization",
}


def load_ground_truth_rows(csv_path: str | Path) -> list[dict[str, str]]:
    """Load ground-truth rows, mapping CSV columns to runner row keys.

    Each row gets a stable ``row_id``: the ``Row`` column when the CSV has
    one (e.g. ``shuffled_prompts_4.csv``), else its 1-based data position.
    The id travels with the row into the logs so the comparator can join
    on it instead of fuzzy text matching. Blank lines (the CSV contains an
    empty ``|||||`` filler line) are skipped.
    """
    rows: list[dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        position = 0
        for record in reader:
            prompt_text = (record.get(GROUND_TRUTH_COLUMNS["prompt_text"]) or "").strip()
            if not prompt_text:
                continue
            position += 1
            row_id = (record.get(GROUND_TRUTH_COLUMNS["row_id"]) or "").strip()
            rows.append(
                {
                    "row_id": row_id or str(position),
                    "prompt_text": prompt_text,
                    "response": (record.get(GROUND_TRUTH_COLUMNS["response"]) or "").strip(),
                    "expected": (record.get(GROUND_TRUTH_COLUMNS["expected"]) or "").strip(),
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
        response=prompt_row.get("response", ""),
        expected=prompt_row.get("expected", ""),
    )
    return {
        "row_id": prompt_row.get("row_id", ""),
        "prompt_text": prompt_row["prompt_text"],
        "response": prompt_row.get("response", ""),
        "expected": prompt_row.get("expected", ""),
        "answers": result.answers,
        "confidences": result.confidences,
        "path": result.path,
        "stop_step": result.stop_step,
        "predicted_label": result.predicted_label,
    }


def run_all(rows, tree, backend, transport, threshold=0.5, base_dir=None):
    """Walk every row down the tree, keeping ground truth attached for the logger."""
    pending = list(rows)
    total = len(pending)
    print(f"processing {total} prompts with backend {backend}")
    results = []
    for number, row in enumerate(pending, start=1):
        print(f"{number}: [{number}/{total}] {row['prompt_text'][:60]}")
        results.append(
            run_prompt(row, tree, backend, transport, threshold=threshold, base_dir=base_dir)
        )
    print(f"done: processed {len(results)}/{total} prompts")
    return results