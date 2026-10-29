"""Pytest suite for the comparator.

Scoring follows ``docs/design_draft.md``: the ground-truth CSV's
``Categorization`` labels are the sole baseline, joined on
``(prompt_text, expected)`` so twin variants of one prompt (e.g. with and
without the ``/logs`` echo) score against their own run. It works whether
or not logged rows carry a ``category`` of their own.
"""

from pathlib import Path

from comparator import (
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

TRUTH = {
    ("add 5 and 7", "12"): [{"expected": "12", "category": "Verifiable Minimal"}],
    ("divide 100 by 4", "InvalidRequest"): [
        {"expected": "InvalidRequest", "category": "Unknowable"}
    ],
    ("empty logs", "[]"): [
        {"expected": "[]", "category": "Verifiable Minimal"},
        {"expected": "[]", "category": "Verifiable"},
    ],
}


def _row(prompt, backend, predicted, confidences="0.9", expected=""):
    if not expected:
        expected = next(
            (exp for (prompt_text, exp) in TRUTH if prompt_text == prompt), ""
        )
    return {
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


def test_load_ground_truth_indexes_prompt_and_expected():
    truth = load_ground_truth(CSV_PATH)
    assert sum(len(entries) for entries in truth.values()) == 50
    assert all(len(entries) >= 1 for entries in truth.values())
    twin_keys = [key for key, entries in truth.items() if len(entries) > 1]
    assert len(twin_keys) >= 1  # the CSV reuses prompts across variants


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


def test_lookup_prefers_exact_variant_then_prompt_fallback():
    row = _row("add 5 and 7", "jev", "end_Verifiable_Minimal")
    assert lookup_candidates(row, TRUTH) == TRUTH[("add 5 and 7", "12")]
    row_no_expected = dict(row, expected="")
    assert lookup_candidates(row_no_expected, TRUTH) == TRUTH[("add 5 and 7", "12")]
    assert lookup_candidates(_row("missing", "jev", "end_Unknowable"), TRUTH) == []


def test_compare_scores_accuracy_per_backend_and_category():
    rows = [
        _row("add 5 and 7", "jev", "end_Verifiable_Minimal"),
        _row("divide 100 by 4", "jev", "end_Verifiable"),  # miss
    ]
    result = compare(rows, TRUTH)
    assert result.per_backend["jev"].n == 2
    assert result.per_backend["jev"].correct == 1
    assert result.per_backend["jev"].accuracy == 0.5
    assert result.per_category["Verifiable Minimal"]["jev"].correct == 1
    assert result.per_category["Unknowable"]["jev"].correct == 0
    assert result.all_missed == ["divide 100 by 4"]
    assert result.diverged == []
    assert result.agreement_rate is None


def test_compare_scores_twin_variants_separately():
    rows = [
        _row("empty logs", "jev", "end_Verifiable_Minimal"),
    ]
    result = compare(rows, TRUTH)
    assert result.per_backend["jev"].correct == 1
    assert "empty logs" in result.ambiguous
    rows = [_row("empty logs", "jev", "end_Unknowable")]
    result = compare(rows, TRUTH)
    assert result.per_backend["jev"].correct == 0
    assert result.all_missed == ["empty logs"]


def test_compare_flags_divergence_and_agreement():
    rows = [
        _row("add 5 and 7", "jev", "end_Verifiable_Minimal"),
        _row("add 5 and 7", "spark-as-jev", "end_Verifiable_Minimal"),
        _row("divide 100 by 4", "jev", "end_Unknowable"),
        _row("divide 100 by 4", "spark-as-jev", "end_Verifiable"),  # diverges + misses
    ]
    result = compare(rows, TRUTH)
    assert result.agreement_n == 2
    assert result.agreement_agree == 1
    assert result.agreement_rate == 0.5
    assert result.diverged == ["divide 100 by 4"]
    assert result.all_missed == []
    assert result.per_backend["spark-as-jev"].accuracy == 0.5


def test_compare_lists_prompts_missing_from_ground_truth():
    result = compare([_row("unknown prompt", "jev", "end_Unknowable")], TRUTH)
    assert result.unscored == ["unknown prompt"]
    assert result.per_backend == {}


def test_compare_builds_confidence_histograms():
    rows = [_row("add 5 and 7", "jev", "end_Verifiable_Minimal", "0.95,0.85")]
    result = compare(rows, TRUTH)
    assert sum(result.histograms["jev"]) == 2
    assert len(result.histograms["jev"]) == 10


def test_compare_files_and_report_on_real_csv(tmp_path):
    truth = load_ground_truth(CSV_PATH)
    (prompt, expected), entries = next(iter(truth.items()))
    label = "end_" + entries[0]["category"].replace(" ", "_")
    log = tmp_path / "runs.jsonl"
    write_jsonl(
        [
            {
                "prompt_text": prompt,
                "backend": "jev",
                "path": "q1",
                "answers": "N",
                "confidences": "0.9",
                "predicted_label": label,
                "stop_step": "q1",
                "expected": expected,
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
    assert "agreement: 0/0 (n/a)" in report
