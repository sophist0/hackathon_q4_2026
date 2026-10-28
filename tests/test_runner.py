"""Pytest suite for the runner.

Expectations follow ``question_tree.yml`` (authoritative over the draft):
``N`` at ``q1`` ends the chain only because ``q1.on_N`` is
``end_Unknowable``, and the longest code-tree path is exactly 3 calls.
"""

from pathlib import Path

import pytest

from decider import Question, System, TreeSpec, load_tree
from runner import load_ground_truth_rows, run_all, run_prompt

ROOT = Path(__file__).resolve().parents[1]
TREE_PATH = ROOT / "question_tree.yml"
CSV_PATH = ROOT / "data/classified_muse_spark_prompts_4.csv"


def _tree():
    return load_tree(TREE_PATH)


def _row(prompt="How do I add 5 and 7?"):
    return {"prompt_text": prompt, "expected": "GET /add/5/7 | 12", "category": "Verifiable Minimal"}


def _stub(queue, calls=None):
    def transport(request):
        if calls is not None:
            calls.append(request["question"]["id"])
        answer, confidence = queue.pop(0)
        return {"answer": answer, "p": confidence}

    return transport


def test_load_ground_truth_rows():
    rows = load_ground_truth_rows(CSV_PATH)
    assert len(rows) == 50
    assert {row["category"] for row in rows} == {
        "Unknowable",
        "Unverifiable Minimal",
        "Verifiable Minimal",
        "Verifiable",
        "Unverifiable",
    }
    assert all(row["prompt_text"] and row["expected"] for row in rows)


def test_n_at_q1_ends_chain_in_one_call():
    calls = []
    result = run_prompt(
        _row(), _tree(), "jev", _stub([("N", 0.8)], calls), base_dir=ROOT
    )
    assert result["answers"] == ["N"]
    assert result["path"] == ["q1"]
    assert result["predicted_label"] == "end_Unknowable"
    assert result["stop_step"] == "q1"
    assert calls == ["q1"]


@pytest.mark.parametrize(
    ("answers", "label"),
    [
        (["Y", "Y", "Y"], "end_Verifiable_Minimal"),
        (["Y", "Y", "N"], "end_Verifiable"),
        (["Y", "N", "Y"], "end_Unverifiable_Minimal"),
        (["Y", "N", "N"], "end_Unverifiable"),
    ],
    ids=["verifiable-minimal", "verifiable", "unverifiable-minimal", "unverifiable"],
)
def test_full_three_step_paths(answers, label):
    result = run_prompt(
        _row(),
        _tree(),
        "jev",
        _stub([(a, 0.9) for a in answers]),
        base_dir=ROOT,
    )
    assert result["answers"] == answers
    assert result["path"] == ["q1", "q2", "q3" if answers[1] == "Y" else "q4"]
    assert result["predicted_label"] == label
    assert len(result["answers"]) <= _tree().max_depth


def test_spark_abstain_mid_chain():
    calls = []

    def transport(request):
        calls.append(request)
        if len(calls) == 1:
            return {"answer": "Y", "confidence": 0.9}
        return {"answer": "Y", "confidence": 0.1}

    result = run_prompt(_row(), _tree(), "spark-as-jev", transport, threshold=0.5)
    assert result["answers"] == ["Y", "abstain"]
    assert result["path"] == ["q1", "q2"]
    assert result["stop_step"] == "q2"
    assert result["predicted_label"] == "abstain"


def test_max_depth_caps_calls_on_deeper_tree():
    system = System(id="s", name="S", spec_ref=None, context="assumed_known")
    chain = TreeSpec(
        tree_id="chain",
        input_class="User Prompt",
        system=system,
        target_labels=("Done",),
        max_depth=2,
        questions=(
            Question(id="q1", text="q1?", on_Y="q2", on_N="end_Done"),
            Question(id="q2", text="q2?", on_Y="q3", on_N="end_Done"),
            Question(id="q3", text="q3?", on_Y="q4", on_N="end_Done"),
            Question(id="q4", text="q4?", on_Y="end_Done", on_N="end_Done"),
        ),
    )
    calls = []
    result = run_prompt(_row(), chain, "jev", _stub([("Y", 0.9)] * 4, calls))
    assert len(calls) == 2 == chain.max_depth
    assert result["predicted_label"] == "undecided"


def test_run_all_keeps_ground_truth_attached():
    rows = load_ground_truth_rows(CSV_PATH)[:3]
    [first, second, third] = rows
    queues = {
        first["prompt_text"]: [("N", 0.8)],
        second["prompt_text"]: [("Y", 0.9), ("Y", 0.8), ("N", 0.7)],
        third["prompt_text"]: [("Y", 0.9), ("N", 0.8), ("Y", 0.7)],
    }

    def transport(request):
        answer, confidence = queues[request["prompt"]].pop(0)
        return {"answer": answer, "p": confidence}

    results = run_all(rows, _tree(), "jev", transport, base_dir=ROOT)
    assert len(results) == 3
    assert [r["predicted_label"] for r in results] == [
        "end_Unknowable",
        "end_Verifiable",
        "end_Unverifiable_Minimal",
    ]
    for row, result in zip(rows, results):
        assert result["expected"] == row["expected"]
        assert result["category"] == row["category"]
