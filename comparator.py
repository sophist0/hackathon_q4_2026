"""Comparator: score backend runs against the ground-truth labels.

Per ``docs/design_draft.md`` the independent ground truth
(``data/shuffled_prompts_4.csv``) is the sole baseline: each backend's
predicted label is compared against the row's ``Categorization`` label,
while backend-vs-backend agreement is diagnostic only.

Logger rows may or may not carry their own ``category`` (runs can strip it
to avoid leaking answers into logs); this module always (re)joins the
ground truth CSV on ``row_id``, so scoring works either way.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from decider import LEAF_PREFIX, canonical_label
from logger import read_jsonl

HISTOGRAM_BINS = 10


def strip_leaf_prefix(predicted_label: str) -> str:
    """Turn ``end_Verifiable_Minimal`` into ``Verifiable_Minimal``."""
    if predicted_label.startswith(LEAF_PREFIX):
        return predicted_label[len(LEAF_PREFIX):]
    return predicted_label


def prediction_matches(predicted_label: str, category: str) -> bool:
    """Check a predicted leaf against a ground-truth category label.

    Comparison is canonical (``Verifiable_Minimal`` == ``Verifiable
    Minimal``, case-insensitive). ``abstain``/``undecided`` never match a
    real category.
    """
    if canonical_label(predicted_label) in ("abstain", "undecided"):
        return False
    return canonical_label(strip_leaf_prefix(predicted_label)) == canonical_label(category)


def parse_confidences(row: Mapping[str, Any]) -> list[float]:
    """Parse a logger row's confidences (``"0.9,0.8"`` string or list)."""
    raw = row.get("confidences", "")
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, Iterable):
        parts = list(raw)
    else:
        return []
    values: list[float] = []
    for part in parts:
        try:
            values.append(float(part))
        except (TypeError, ValueError):
            continue
    return values


GROUND_TRUTH_COLUMNS = {
    "row_id": "Row",
    "prompt_text": "Prompts",
    "expected": "Expected Request Response",
    "category": "Categorization",
}


@dataclass
class GroundTruth:
    """Ground-truth index; the draft's sole baseline, parsed from the CSV.

    Joins logged rows on their stable ``row_id`` only (the ``Row`` column
    of the shuffled CSV, or the runner's positional id). No text fallback:
    a row without a matching id is unscored.
    """

    by_id: dict[str, list[dict[str, str]]] = field(default_factory=dict)


def load_ground_truth(csv_path: str | Path) -> GroundTruth:
    """Index the ground-truth CSV, independent of whatever the rows carry.

    Scoring works even when logged rows strip the labels, since the
    ``Categorization`` labels are rejoined here at compare time.
    """
    truth = GroundTruth()
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        position = 0
        for record in reader:
            prompt_text = (record.get(GROUND_TRUTH_COLUMNS["prompt_text"]) or "").strip()
            if not prompt_text:
                continue
            position += 1
            row_id = (record.get(GROUND_TRUTH_COLUMNS["row_id"]) or "").strip() or str(
                position
            )
            truth.by_id.setdefault(row_id, []).append(
                {
                    "row_id": row_id,
                    "prompt_text": prompt_text,
                    "expected": (record.get(GROUND_TRUTH_COLUMNS["expected"]) or "").strip(),
                    "category": (
                        record.get(GROUND_TRUTH_COLUMNS["category"]) or ""
                    ).strip(),
                }
            )
    return truth


def lookup_candidates(row: Mapping[str, Any], ground_truth: GroundTruth) -> list[dict[str, str]]:
    """Find the ground-truth entries a logged row should score against.

    Joins on ``row_id`` only. An empty list means unscored.
    """
    row_id = str(row.get("row_id", ""))
    if row_id:
        return ground_truth.by_id.get(row_id, [])
    return []


@dataclass
class BackendScore:
    n: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.n if self.n else None


@dataclass
class Comparison:
    """Result of scoring logger rows against the ground truth."""

    per_backend: dict[str, BackendScore] = field(default_factory=dict)
    per_category: dict[str, dict[str, BackendScore]] = field(default_factory=dict)
    agreement_n: int = 0  # prompts covered by 2+ backends
    agreement_agree: int = 0  # ... where every backend predicted the same label
    diverged: list[str] = field(default_factory=list)  # prompt_texts backends disagree on
    all_missed: list[str] = field(default_factory=list)  # prompt_texts no backend got right
    unscored: list[str] = field(default_factory=list)  # prompt_texts missing from ground truth
    ambiguous: list[str] = field(default_factory=list)  # prompt_texts with several truth variants
    repeated: list[str] = field(default_factory=list)  # row_ids scored more than once per backend
    confusion: dict[str, dict[str, dict[str, int]]] = field(  # backend -> actual -> predicted -> runs
        default_factory=dict
    )
    histograms: dict[str, list[int]] = field(default_factory=dict)

    @property
    def agreement_rate(self) -> float | None:
        return self.agreement_agree / self.agreement_n if self.agreement_n else None


def _histogram(values: Iterable[float], bins: int = HISTOGRAM_BINS) -> list[int]:
    counts = [0] * bins
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        index = min(int(number * bins), bins - 1)
        if index >= 0:
            counts[index] += 1
    return counts


def _display_predicted(predicted_label: Any) -> str:
    """Render a logged prediction as a matrix label (``end_X_Y`` -> ``X Y``)."""
    text = strip_leaf_prefix(str(predicted_label)).replace("_", " ").strip()
    return text or str(predicted_label)


def _confusion_axes(
    matrix: dict[str, dict[str, int]],
) -> tuple[list[str], list[str]]:
    """Order matrix rows (actuals) and columns (actuals, then extras)."""
    actuals = sorted(matrix)
    known = set(actuals)
    extras = sorted({label for actual in actuals for label in matrix[actual] if label not in known})
    return actuals, actuals + extras


def plot_confusion(comparison: Comparison, path: str | Path) -> Path:
    """Save per-backend confusion matrices (actual rows, predicted columns)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting needs matplotlib: pip install matplotlib") from exc
    if not comparison.confusion:
        raise ValueError("no scored rows to plot")
    backends = sorted(comparison.confusion)
    fig, axes = plt.subplots(1, len(backends), figsize=(6 * len(backends), 5), squeeze=False)
    for ax, backend in zip(axes[0], backends):
        matrix = comparison.confusion[backend]
        actuals, columns = _confusion_axes(matrix)
        values = [[matrix[actual].get(column, 0) for column in columns] for actual in actuals]
        top = max(max(row) for row in values) if values else 0
        ax.imshow(values, cmap="Blues", vmin=0, vmax=max(top, 1))
        ax.set_xticks(range(len(columns)), columns, rotation=30, ha="right")
        ax.set_yticks(range(len(actuals)), actuals)
        ax.set_title(backend)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        for i, actual in enumerate(actuals):
            for j, column in enumerate(columns):
                count = matrix[actual].get(column, 0)
                ax.text(j, i, str(count), ha="center", va="center")
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _row_correct(row: Mapping[str, Any], candidates: list[dict[str, str]]) -> bool:
    """A row scores when its prediction matches any candidate variant."""
    predicted = str(row.get("predicted_label", ""))
    return any(
        entry["category"] and prediction_matches(predicted, entry["category"])
        for entry in candidates
    )


def compare(
    rows: Iterable[Mapping[str, Any]],
    ground_truth: GroundTruth,
) -> Comparison:
    """Score logger rows (any mix of backends) against the ground truth.

    Joins on ``row_id`` only. Rows without a ``row_id``, or whose id has
    no ground-truth entry, are unscored. Every row scores, so passing
    overlapping logs (e.g. a one-off plus a full run) counts the shared
    ``row_id`` once per run; such repeats are listed in ``repeated``.
    """
    groups: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    idless: list[Mapping[str, Any]] = []
    for row in rows:
        backend = str(row.get("backend", ""))
        if not backend:
            continue
        row_id = str(row.get("row_id", ""))
        if row_id:
            # Rows sharing a row_id are repeat runs of one ground-truth row.
            groups[row_id][backend].append(row)
        else:
            idless.append(row)

    result = Comparison()
    confidences: dict[str, list[float]] = defaultdict(list)
    for row in idless:
        result.unscored.append(str(row.get("prompt_text", "")))
    for row_id, per_backend in groups.items():
        first_row = next(iter(per_backend.values()))[0]
        label = f"row {row_id}: {first_row.get('prompt_text', '')}"
        candidates = lookup_candidates(first_row, ground_truth)
        if not candidates:
            result.unscored.append(label)
            continue
        if len({entry["category"] for entry in candidates}) > 1 and label not in result.ambiguous:
            result.ambiguous.append(label)
        for backend, runs in per_backend.items():
            if len(runs) > 1:
                result.repeated.append(f"{label} x{len(runs)} ({backend})")
        flat: list[Mapping[str, Any]] = [
            row for runs in per_backend.values() for row in runs
        ]
        for backend, runs in per_backend.items():
            for row in runs:
                confidences[backend].extend(parse_confidences(row))
                score = result.per_backend.setdefault(backend, BackendScore())
                score.n += 1
                correct = _row_correct(row, candidates)
                if correct:
                    score.correct += 1
                for category in {entry["category"] for entry in candidates if entry["category"]}:
                    cell = result.per_category.setdefault(category, {}).setdefault(
                        backend, BackendScore()
                    )
                    cell.n += 1
                    if prediction_matches(str(row.get("predicted_label", "")), category):
                        cell.correct += 1
                    predicted = _display_predicted(row.get("predicted_label", ""))
                    counts = result.confusion.setdefault(backend, {}).setdefault(category, {})
                    counts[predicted] = counts.get(predicted, 0) + 1
        backends = [backend for backend, runs in per_backend.items() for _ in runs]
        if len(set(backends)) >= 2:
            labels = {
                canonical_label(strip_leaf_prefix(str(row.get("predicted_label", ""))))
                for row in flat
            }
            result.agreement_n += 1
            if len(labels) == 1:
                result.agreement_agree += 1
            else:
                result.diverged.append(label)
        if not any(_row_correct(row, candidates) for row in flat):
            result.all_missed.append(label)
    result.histograms = {
        backend: _histogram(values) for backend, values in confidences.items()
    }
    return result


def compare_files(
    log_paths: Iterable[str | Path], ground_truth_path: str | Path
) -> Comparison:
    """Read one or more JSONL logs and score them against a ground-truth CSV."""
    rows: list[dict[str, str]] = []
    for log_path in log_paths:
        rows.extend(read_jsonl(log_path))
    return compare(rows, load_ground_truth(ground_truth_path))


def _pct(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "n/a"


def format_report(comparison: Comparison) -> str:
    """Render a plain-text per-category accuracy table plus diverge list."""
    lines = ["\nbackend: correct/n (accuracy)\n"]
    for backend in sorted(comparison.per_backend):
        score = comparison.per_backend[backend]
        lines.append(f"  {backend}: {score.correct}/{score.n} ({_pct(score.accuracy)})")

    lines.append("\nper-category accuracy:\n")
    for category in sorted(comparison.per_category):
        cells = "  ".join(
            f"{backend} {score.correct}/{score.n} ({_pct(score.accuracy)})"
            for backend, score in sorted(comparison.per_category[category].items())
        )
        lines.append(f"  {category}: {cells}")
    lines.append(
        f"\nagreement: {comparison.agreement_agree}/{comparison.agreement_n} "
        f" ({_pct(comparison.agreement_rate)})\n"
    )
    lines.append(f"\ndiverged ({len(comparison.diverged)}):\n")
    lines.extend(f"  - {prompt}" for prompt in comparison.diverged)
    lines.append(f"\nall backends missed ({len(comparison.all_missed)}):\n")
    lines.extend(f"  - {prompt}" for prompt in comparison.all_missed)
    if comparison.ambiguous:
        lines.append(f"\nambiguous truth variants ({len(comparison.ambiguous)}):\n")
        lines.extend(f"  - {prompt}" for prompt in comparison.ambiguous)
    if comparison.repeated:
        lines.append(f"\nrepeated rows, counted once per run ({len(comparison.repeated)}):\n")
        lines.extend(f"  - {prompt}" for prompt in comparison.repeated)
    if comparison.unscored:
        lines.append(f"\nunscored, no ground truth ({len(comparison.unscored)}):\n")
        lines.extend(f"  - {prompt}" for prompt in comparison.unscored)
    if comparison.confusion:
        lines.append("\nconfusion matrix figure: pass --plot <path> to save it as PNG.\n")
    lines.append("\n")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="JSONL log file(s) written by logger.write_jsonl")
    parser.add_argument(
        "--ground-truth",
        default="data/shuffled_prompts_4.csv",
        help="ground-truth CSV (Categorization labels are the baseline)",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--plot", default=None, help="save per-backend confusion matrices as a PNG"
    )
    args = parser.parse_args(argv)
    comparison = compare_files(args.logs, args.ground_truth)
    if args.plot:
        out = plot_confusion(comparison, args.plot)
        print(f"wrote confusion matrix to {out}")
    if args.json:
        print(
            json.dumps(
                {
                    "per_backend": {
                        backend: {"n": score.n, "accuracy": score.accuracy}
                        for backend, score in comparison.per_backend.items()
                    },
                    "agreement_rate": comparison.agreement_rate,
                    "diverged": comparison.diverged,
                    "all_missed": comparison.all_missed,
                    "ambiguous": comparison.ambiguous,
                    "repeated": comparison.repeated,
                    "confusion": comparison.confusion,
                    "unscored": comparison.unscored,
                    "histograms": comparison.histograms,
                },
                indent=2,
            )
        )
    else:
        print(format_report(comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
