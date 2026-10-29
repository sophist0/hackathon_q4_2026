"""Pytest suite for the logger."""

import csv

from decider import load_tree
from logger import CSV_COLUMNS, format_row, read_jsonl, write_csv, write_jsonl
from runner import load_ground_truth_rows, run_all

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE_PATH = ROOT / "question_tree.yml"
CSV_PATH = ROOT / "data/classified_muse_spark_prompts_4.csv"


def _results():
    tree = load_tree(TREE_PATH)
    rows = load_ground_truth_rows(CSV_PATH)[:2]

    def transport(request):
        return {"answer": "N", "p": 0.8}

    return run_all(rows, tree, "jev", transport, base_dir=ROOT)


def test_format_row_flattens_runner_result():
    first = _results()[0]
    row = format_row(first, "jev")
    assert row["backend"] == "jev"
    assert row["path"] == "q1"
    assert row["answers"] == "N"
    assert row["predicted_label"] == "end_Unknowable"
    assert row["expected"]
    assert row["row_id"] == first["row_id"]
    assert row["response"] == first["response"]
    # Labels stay out of logs (no-leak rule); the column still exists
    # so the comparator's side-by-side layout is unchanged.
    assert row["category"] == ""
    assert set(row) == set(CSV_COLUMNS)


def test_jsonl_round_trip_and_backend_append(tmp_path):
    log = tmp_path / "runs.jsonl"
    first, second = _results()
    write_jsonl([first], "jev", log)
    write_jsonl([second], "spark-as-jev", log, append=True)
    rows = read_jsonl(log)
    assert [row["backend"] for row in rows] == ["jev", "spark-as-jev"]
    assert rows[0]["prompt_text"] == first["prompt_text"]
    assert rows[1]["prompt_text"] == second["prompt_text"]


def test_write_creates_missing_parent_dirs(tmp_path):
    nested = tmp_path / "logs" / "nested" / "runs.jsonl"
    write_jsonl(_results(), "jev", nested)
    assert len(read_jsonl(nested)) == 2
    nested_csv = tmp_path / "logs" / "nested" / "runs.csv"
    write_csv(_results(), "jev", nested_csv)
    assert nested_csv.exists()


def test_csv_export_matches_column_layout(tmp_path):
    out = tmp_path / "runs.csv"
    write_csv(_results(), "clef-flash", out)
    with open(out, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert set(rows[0]) == set(CSV_COLUMNS)
    assert all(row["backend"] == "clef-flash" for row in rows)
    assert rows[0]["path"] == "q1"
