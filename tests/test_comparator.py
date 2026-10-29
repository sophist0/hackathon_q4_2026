"""Pytest suite for the comparator.

Scoring follows ``docs/design_draft.md``: the ground-truth CSV's
``Categorization`` labels are the sole baseline, joined on the stable
``row_id`` only. Rows without a matching id are unscored, never guessed.
It works whether or not logged rows carry a ``category`` of their own.
"""

from pathlib import Path

from comparator import (
    GroundTruth,
    compare,
    compare_files,
    format_report,
    load_ground_truth,
    lookup_candidates,
    parse_confidences,
    prediction_matches,
)
from logger import write_jsonl

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/classified_muse_spark_prompts_4.csv"


def _truth(*entries):
    """Build a GroundTruth from (row_id, prompt, expected, category) tuples."""
    truth = GroundTruth()
    for row_id, prompt, expected, category in entries:
        truth.by_id.setdefault(row_id, []).append(
            {"row_id": row_id, "prompt_text": prompt, "expected": expected, "category": category}
        )
    return truth


TRUTH = _truth(
    ("1", "add 5 and 7", "12", "Verifiable Minimal"),
    ("2", "divide 100 by 4", "InvalidRequest", "Unknowable"),
    ("3", "empty logs", "[]", "Verifiable Minimal"),
    ("3", "empty logs", "[]", "Verifiable"),
)


def _row(prompt, backend, predicted, confidences="0.9", row_id=None, expected="12"):
    if row_id is None:
        row_id = {"add 5 and 7": "1", "divide 100 by 4": "2", "empty logs": "3"}.get(
            prompt, ""
        )
    return {
        "row_id": row_id,
        "prompt_text": prompt,
        "backend": backend,
        "path": "q1",
        "answers": "N",
        "confidences": confidences,
        "predicted_label": predicted,
        "stop_step": "q1",
        "expected": expected,
        "category": "",  # stripped rows score the same as labelled ones
    }


def test_load_ground_truth_indexes_row_ids():
    truth = load_ground_truth(CSV_PATH)
    assert sum(len(entries) for entries in truth.by_id.values()) == 50
    # The classified CSV has no Row column: ids are the 1-based positions.
    assert sorted(truth.by_id, key=int) == [str(i) for i in range(1, 51)]


def test_load_ground_truth_prefers_row_column(tmp_path):
    csv_path = tmp_path / "mini.csv"
    csv_path.write_text(
        "Row|Prompts|Response|Expected Request Response|Explanation|Categorization|Notes\n"
        "7|How do I add?|GET /add/1/2|12|x|Verifiable Minimal|\n",
        encoding="utf-8",
    )
    truth = load_ground_truth(csv_path)
    assert list(truth.by_id) == ["7"]
    assert truth.by_id["7"][0]["category"] == "Verifiable Minimal"


def test_prediction_matches_canonicalizes_leaf_labels():
    assert prediction_matches("end_Verifiable_Minimal", "Verifiable Minimal")
    assert prediction_matches("end_Unknowable", "unknowable")
    assert not prediction_matches("end_Verifiable", "Verifiable Minimal")
    assert not prediction_matches("abstain", "Unknowable")
    assert not prediction_matches("undecided", "Unknowable")


def test_parse_confidences_handles_string_and_list():
    assert parse_confidences({"confidences": "0.9, 0.8"}) == [0.9, 0.8]
    assert parse_confidences({"confidences": [0.9, "bad"]}) == [0.9]
    assert parse_confidences({}) == []


def test_lookup_joins_on_row_id_only():
    row = _row("empty logs", "jev", "end_Verifiable")
    assert [entry["category"] for entry in lookup_candidates(row, TRUTH)] == [
        "Verifiable Minimal",
        "Verifiable",
    ]
    assert lookup_candidates(_row("add 5 and 7", "jev", "end_X", row_id="99"), TRUTH) == []
    assert lookup_candidates(_row("add 5 and 7", "jev", "end_X", row_id=""), TRUTH) == []


def test_compare_scores_accuracy_per_backend_and_category():
    rows = [
        _row("add 5 and 7", "jev", "end_Verifiable_Minimal"),
        _row("divide 100 by 4", "jev", "end_Verifiable", expected="InvalidRequest"),
    ]
    result = compare(rows, TRUTH)
    assert result.per_backend["jev"].n == 2
    assert result.per_backend["jev"].correct == 1
    assert result.per_backend["jev"].accuracy == 0.5
    assert result.per_category["Verifiable Minimal"]["jev"].correct == 1
    assert result.per_category["Unknowable"]["jev"].correct == 0
    assert result.all_missed == ["row 2: divide 100 by 4"]
    assert result.diverged == []
    assert result.agreement_rate is None


def test_compare_scores_twin_variants_by_id():
    result = compare([_row("empty logs", "jev", "end_Verifiable_Minimal", expected="[]")], TRUTH)
    assert result.per_backend["jev"].correct == 1
    assert result.all_missed == []
    result = compare([_row("empty logs", "jev", "end_Unknowable", expected="[]")], TRUTH)
    assert result.per_backend["jev"].correct == 0
    assert result.all_missed == ["row 3: empty logs"]


def test_compare_leaves_idless_and_unknown_rows_unscored():
    rows = [
        _row("add 5 and 7", "jev", "end_Verifiable_Minimal", row_id=""),
        _row("add 5 and 7", "jev", "end_Verifiable_Minimal", row_id="99"),
    ]
    result = compare(rows, TRUTH)
    assert result.per_backend == {}
    assert sorted(result.unscored) == ["add 5 and 7", "row 99: add 5 and 7"]


def test_compare_flags_divergence_and_agreement():
    rows = [
        _row("add 5 and 7", "jev", "end_Verifiable_Minimal"),
        _row("add 5 and 7", "spark-as-jev", "end_Verifiable_Minimal"),
        _row("divide 100 by 4", "jev", "end_Unknowable", expected="InvalidRequest"),
        _row("divide 100 by 4", "spark-as-jev", "end_Verifiable", expected="InvalidRequest"),
    ]
    result = compare(rows, TRUTH)
    assert result.agreement_n == 2
    assert result.agreement_agree == 1
    assert result.agreement_rate == 0.5
    assert result.diverged == ["row 2: divide 100 by 4"]
    assert result.all_missed == []
    assert result.per_backend["spark-as-jev"].accuracy == 0.5


def test_compare_builds_confidence_histograms():
    rows = [_row("add 5 and 7", "jev", "end_Verifiable_Minimal", "0.95,0.85")]
    result = compare(rows, TRUTH)
    assert sum(result.histograms["jev"]) == 2
    assert len(result.histograms["jev"]) == 10


def test_compare_files_and_report_on_real_csv(tmp_path):
    truth = load_ground_truth(CSV_PATH)
    row_id, entries = next(iter(truth.by_id.items()))
    label = "end_" + entries[0]["category"].replace(" ", "_")
    log = tmp_path / "runs.jsonl"
    write_jsonl(
        [
            {
                "row_id": row_id,
                "prompt_text": entries[0]["prompt_text"],
                "backend": "jev",
                "path": "q1",
                "answers": "N",
                "confidences": "0.9",
                "predicted_label": label,
                "stop_step": "q1",
                "expected": entries[0]["expected"],
                "category": "",
            }
        ],
        "jev",
        log,
    )
    result = compare_files([log], CSV_PATH)
    assert result.per_backend["jev"].correct == 1
    report = format_report(result)
    assert "jev: 1/1 (100.0%)" in report
    assert "agreement: 0/0" in report
