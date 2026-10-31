"""Pytest suite for the decider interface.

The tree file (question_tree.yml) is authoritative over the design draft
where they disagree, so transition/label expectations below follow the
file, not the draft's adder-specific example tree.
"""

from pathlib import Path

import pytest

from decider import (
    BadDeciderReply,
    DeciderState,
    System,
    UnknownBackend,
    UnknownContext,
    canonical_label,
    decide,
    leaf_label,
    load_tree,
    resolve_context,
    walk,
)

ROOT = Path(__file__).resolve().parents[1]
TREE_PATH = ROOT / "question_tree.yml"


def test_load_real_tree_shape():
    tree = load_tree(TREE_PATH)
    assert tree.tree_id == "user_prompt_v1"
    assert tree.input_class == "User Prompt"
    assert tree.max_depth == 3
    assert [q.id for q in tree.questions] == ["q1", "q2", "q3", "q4"]


def test_tree_transitions_follow_code_not_draft():
    tree = load_tree(TREE_PATH)
    by_id = {q.id: q for q in tree.questions}
    # Code: q1 fans out to q2 / end_Unknowable (draft example used q3/q2).
    assert (by_id["q1"].on_Y, by_id["q1"].on_N) == ("q2", "end_Unknowable")
    assert (by_id["q2"].on_Y, by_id["q2"].on_N) == ("q3", "q4")
    assert (by_id["q3"].on_Y, by_id["q3"].on_N) == (
        "end_Verifiable_Minimal",
        "end_Verifiable",
    )
    assert (by_id["q4"].on_Y, by_id["q4"].on_N) == (
        "end_Unverifiable_Minimal",
        "end_Unverifiable",
    )


def test_leaf_labels_cover_target_labels():
    tree = load_tree(TREE_PATH)
    leaves = set()
    for question in tree.questions:
        for edge in (question.on_Y, question.on_N):
            label = leaf_label(edge)
            if label is not None:
                leaves.add(canonical_label(label))
    assert leaves == {canonical_label(label) for label in tree.target_labels}


def test_resolve_inline_loads_spec_ref():
    tree = load_tree(TREE_PATH)
    text = resolve_context(tree.system, base_dir=ROOT)
    assert "micro service" in text
    assert "InvalidRequest" in text


def test_resolve_assumed_known_injects_nothing(tmp_path):
    system = System(id="linux", name="Linux", spec_ref=None, context="assumed_known")
    assert resolve_context(system, base_dir=tmp_path) == ""


def test_resolve_unknown_context_raises(tmp_path):
    system = System(id="x", name="X", spec_ref=None, context="remote")
    with pytest.raises(UnknownContext):
        resolve_context(system, base_dir=tmp_path)


def _state(**overrides):
    system = System(
        id="adder_8bit_v1",
        name="8-bit adder microservice",
        spec_ref="data/8bit_adder_spec.txt",
        context="assumed_known",
    )
    return DeciderState(system=system, prompt_text="How do I add 5 and 7?")


def _question(tree, qid):
    return tree.question_by_id(qid)


def test_decide_jev_native_shape():
    tree = load_tree(TREE_PATH)
    seen = {}

    def transport(request):
        seen.update(request)
        return {"model": "jev-latest", "answers": {"q1": {"type": "noul", "noul": 0.85}}}

    decision = decide(_state(), _question(tree, "q1"), "jev", transport)
    assert (decision.answer, decision.p) == ("Y", 0.85)
    assert seen["model"] == "jev-latest"
    assert seen["questions"]["q1"] == {"type": "noul", "instructions": _question(tree, "q1").text}
    assert "How do I add 5 and 7?" in seen["state"]


def test_decide_jev_maps_noul_to_confidence():
    tree = load_tree(TREE_PATH)
    decision = decide(
        _state(),
        _question(tree, "q1"),
        "jev",
        lambda req: {"answers": {"q1": {"type": "noul", "noul": 0.2}}},
    )
    assert decision.answer == "N"
    assert decision.p == pytest.approx(0.8)


def test_normalize_jev_legacy_stub_shape_still_accepted():
    from decider import Decision, normalize_jev_reply

    assert normalize_jev_reply({"answer": "Y", "p": 0.9}) == Decision(answer="Y", p=0.9)
    with pytest.raises(BadDeciderReply):
        normalize_jev_reply({"answer": "maybe", "p": 0.5})


def test_decide_clef_flash_native_shape():
    from decider import build_clef_request

    tree = load_tree(TREE_PATH)
    seen = {}

    def transport(request):
        seen.update(request)
        return {"answers": {"q1": {"type": "noul", "noul": 0.9}}}

    decision = decide(_state(), _question(tree, "q1"), "clef-flash", transport)
    assert (decision.answer, decision.p) == ("Y", 0.9)
    assert seen["model"] == "clef-flash"
    assert seen["questions"]["q1"]["type"] == "noul"
    assert "state" in seen


def test_build_clef_request_carries_state_and_question():
    from decider import build_clef_request

    tree = load_tree(TREE_PATH)
    state = DeciderState(
        system=tree.system,
        prompt_text="How do I add 5 and 7?",
        response="GET /add/5/7",
        expected="12",
    )
    request = build_clef_request(state, _question(tree, "q2"), "spec text")
    assert request["model"] == "clef-flash"
    assert "How do I add 5 and 7?" in request["state"]
    assert "spec text" in request["state"]
    assert request["questions"]["q2"]["instructions"] == _question(tree, "q2").text


def test_build_clef_request_supports_full_clef_model():
    from decider import build_clef_request

    tree = load_tree(TREE_PATH)
    state = DeciderState(
        system=tree.system,
        prompt_text="How do I add 5 and 7?",
        response="GET /add/5/7",
        expected="12",
    )
    request = build_clef_request(state, _question(tree, "q1"), "spec text", model="clef")
    assert request["model"] == "clef"
    assert "How do I add 5 and 7?" in request["state"]


def test_decide_full_clef_backend():
    tree = load_tree(TREE_PATH)
    seen = {}

    def transport(request):
        seen.update(request)
        return {"answers": {"q1": {"type": "noul", "noul": 0.2}}}

    decision = decide(_state(), _question(tree, "q1"), "clef", transport)
    assert decision.answer == "N"
    assert decision.p == pytest.approx(0.8)
    assert seen["model"] == "clef"


def test_system_one_state_uses_llm_field_labels():
    from decider import build_clef_request

    tree = load_tree(TREE_PATH)
    state = DeciderState(
        system=tree.system,
        prompt_text="How do I add 5 and 7?",
        response="GET /add/5/7",
        expected="12",
    )
    text = build_clef_request(state, _question(tree, "q1"), "spec text")["state"]
    assert "LLM_response:" in text
    # The answer key stays out of the model state (H4 ablation); scoring
    # still rejoins it from the logs at compare time.
    assert "expected_system_output:" not in text


def test_question_examples_reach_state_text():
    from decider import Question, QuestionExample, build_clef_request

    tree = load_tree(TREE_PATH)
    question = Question(
        id="q1",
        text="Can it be answered? (Y/N)",
        on_Y="q2",
        on_N="end_Unknowable",
        examples=(
            QuestionExample(
                prompt="How do I know the last system user?",
                response="GET http://localhost/last_user",
                answer="N",
            ),
        ),
    )
    state = DeciderState(system=tree.system, prompt_text="How do I divide 100 by 4?")
    text = build_clef_request(state, question, "spec text")["state"]
    assert "How do I know the last system user?" in text
    assert "Answer: N" in text
    assert "How do I divide 100 by 4?" in text


def test_noul_cutoff_moves_yes_boundary():
    from decider import normalize_clef_reply

    raw = {"answers": {"q1": {"type": "noul", "noul": 0.3}}}
    assert normalize_clef_reply(raw, question_id="q1", cutoff=0.5).answer == "N"
    assert normalize_clef_reply(raw, question_id="q1", cutoff=0.2).answer == "Y"
    assert normalize_clef_reply(raw, question_id="q1").answer == "N"


def test_decide_accepts_per_question_cutoffs():
    tree = load_tree(TREE_PATH)

    def transport(request):
        return {"answers": {"q1": {"type": "noul", "noul": 0.3}}}

    low = decide(_state(), _question(tree, "q1"), "clef-flash", transport, cutoff={"q1": 0.2})
    high = decide(_state(), _question(tree, "q1"), "clef-flash", transport, cutoff={"q1": 0.5})
    assert (low.answer, high.answer) == ("Y", "N")


def test_normalize_clef_noul_maps_to_confidence():
    from decider import normalize_clef_reply

    yes = normalize_clef_reply(
        {"answers": {"q1": {"type": "noul", "noul": 0.9}}}, question_id="q1"
    )
    assert (yes.answer, yes.p) == ("Y", 0.9)
    no = normalize_clef_reply(
        {"answers": {"q1": {"type": "noul", "noul": 0.2}}}, question_id="q1"
    )
    assert (no.answer, no.p) == ("N", pytest.approx(0.8))
    wrapped = normalize_clef_reply(
        {"result": {"answers": {"q1": {"type": "noul", "noul": 0.7}}}},
        question_id="q1",
    )
    assert (wrapped.answer, wrapped.p) == ("Y", 0.7)


def test_normalize_clef_legacy_proxy_shape_still_accepted():
    from decider import normalize_clef_reply

    decision = normalize_clef_reply({"answer": "N", "p": 0.6})
    assert (decision.answer, decision.p) == ("N", 0.6)
    with pytest.raises(BadDeciderReply):
        normalize_clef_reply({"answers": {}}, question_id="q1")


def test_decide_spark_abstains_below_threshold():
    tree = load_tree(TREE_PATH)
    decision = decide(
        _state(),
        _question(tree, "q1"),
        "spark-as-jev",
        lambda req: {"answer": "Y", "confidence": 0.2},
        threshold=0.5,
    )
    assert (decision.answer, decision.p) == ("abstain", 0.2)


def test_decide_spark_keeps_answer_at_threshold():
    tree = load_tree(TREE_PATH)
    decision = decide(
        _state(),
        _question(tree, "q1"),
        "spark-as-jev",
        lambda req: {"answer": "N", "confidence": 0.5},
        threshold=0.5,
    )
    assert (decision.answer, decision.p) == ("N", 0.5)


def test_decide_rejects_bad_replies():
    tree = load_tree(TREE_PATH)
    state, question = _state(), _question(tree, "q1")
    with pytest.raises(BadDeciderReply):
        decide(state, question, "jev", lambda req: {"answer": "maybe", "p": 0.5})
    with pytest.raises(BadDeciderReply):
        decide(state, question, "jev", lambda req: {"answer": "Y", "p": 1.5})
    with pytest.raises(UnknownBackend):
        decide(state, question, "other", lambda req: {"answer": "Y", "p": 0.5})


def test_state_response_and_expected_reach_jev_request():
    from decider import build_jev_request

    tree = load_tree(TREE_PATH)
    state = DeciderState(
        system=tree.system,
        prompt_text="How do I add 5 and 7?",
        response="GET /add/5/7",
        expected="12",
    )
    request = build_jev_request(state, _question(tree, "q1"), "spec")
    assert request["model"] == "jev-latest"
    assert "How do I add 5 and 7?" in request["state"]
    assert "GET /add/5/7" in request["state"]
    assert "expected_system_output:" not in request["state"]
    assert "spec" in request["state"]
    assert request["questions"]["q1"] == {
        "type": "noul",
        "instructions": _question(tree, "q1").text,
    }


def test_state_response_and_expected_reach_spark_request():
    from decider import build_spark_request

    tree = load_tree(TREE_PATH)
    state = DeciderState(
        system=tree.system,
        prompt_text="How do I add 5 and 7?",
        response="GET /add/5/7",
        expected="12",
    )
    request = build_spark_request(state, _question(tree, "q1"), "spec")
    assert request["response"] == "GET /add/5/7"
    assert request["expected"] == "12"


def test_walk_threads_response_into_state_not_expected():
    tree = load_tree(TREE_PATH)
    seen = []

    def transport(request):
        seen.append(request)
        return {"answer": "N", "p": 0.8}

    walk(
        tree,
        "How do I add 5 and 7?",
        "jev",
        transport,
        base_dir=ROOT,
        response="GET /add/5/7",
        expected="12",
    )
    assert seen
    assert all("LLM_response: GET /add/5/7" in request["state"] for request in seen)
    assert all("expected_system_output:" not in request["state"] for request in seen)


def test_walk_three_step_chain_through_code_tree():
    tree = load_tree(TREE_PATH)
    replies = iter([("Y", 0.9), ("Y", 0.8), ("N", 0.75)])

    def transport(request):
        answer, confidence = next(replies)
        return {"answer": answer, "p": confidence}

    result = walk(tree, "How do I add 5 and 7?", "jev", transport, base_dir=ROOT)
    assert result.path == ["q1", "q2", "q3"]
    assert result.answers == ["Y", "Y", "N"]
    assert result.predicted_label == "end_Verifiable"
    assert result.stop_step == "q3"
    assert len(result.answers) <= tree.max_depth
