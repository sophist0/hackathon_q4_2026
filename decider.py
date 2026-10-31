"""Decider interface for the prompt-classification tree.

Implements the ``decide(state, question) -> {answer, p}`` module from
``docs/design_draft.md``, driven by ``question_tree.yml``.

Precedence rule: where the draft and the tree disagree (question text,
``spec_ref``, transitions, label spelling), the tree file wins. The draft
describes intent; the tree declares what to ask.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Answer = Literal["Y", "N", "abstain"]
BackendName = Literal["jev", "clef-flash", "clef", "spark-as-jev"]

BACKENDS: tuple[str, ...] = ("jev", "clef-flash", "clef", "spark-as-jev")

LEAF_PREFIX = "end_"


class UnknownContext(ValueError):
    """Raised when ``system.context`` is neither ``inline`` nor ``assumed_known``."""


class UnknownBackend(ValueError):
    """Raised when an unregistered backend name is requested."""


class BadDeciderReply(ValueError):
    """Raised when a backend reply cannot be normalized to ``{answer, p}``."""


@dataclass(frozen=True)
class System:
    id: str
    name: str
    spec_ref: str | None
    context: str


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    on_Y: str
    on_N: str


@dataclass(frozen=True)
class TreeSpec:
    tree_id: str
    input_class: str
    system: System
    target_labels: tuple[str, ...]
    max_depth: int
    stop_rules: tuple[str, ...] = ()
    questions: tuple[Question, ...] = ()

    def question_by_id(self, question_id: str) -> Question:
        for question in self.questions:
            if question.id == question_id:
                return question
        raise KeyError(f"unknown question id: {question_id!r}")

    @property
    def first(self) -> Question:
        return self.questions[0]


@dataclass(frozen=True)
class DeciderState:
    system: System
    prompt_text: str
    response: str = ""
    expected: str = ""


@dataclass(frozen=True)
class Decision:
    answer: Answer
    p: float


def canonical_label(label: str) -> str:
    """Normalize a label for comparison (``Verifiable_Minimal`` == ``Verifiable Minimal``)."""
    return " ".join(label.replace("_", " ").split()).casefold()


def leaf_label(node_id: str) -> str | None:
    """Return the raw leaf label for ``end_<Label>`` ids, else ``None``."""
    if node_id.startswith(LEAF_PREFIX):
        return node_id[len(LEAF_PREFIX):]
    return None


def load_tree(path: str | Path) -> TreeSpec:
    """Load and validate a tree spec file. The file is authoritative."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"tree spec must be a mapping: {path}")
    system_raw = raw.get("system") or {}
    questions_raw = raw.get("questions") or []
    tree = TreeSpec(
        tree_id=str(raw["tree_id"]),
        input_class=str(raw["input_class"]),
        system=System(
            id=str(system_raw["id"]),
            name=str(system_raw["name"]),
            spec_ref=system_raw.get("spec_ref"),
            context=str(system_raw["context"]),
        ),
        target_labels=tuple(str(label) for label in raw["target_labels"]),
        max_depth=int(raw["max_depth"]),
        stop_rules=tuple(str(rule) for rule in raw.get("stop_rules", [])),
        questions=tuple(
            Question(
                id=str(item["id"]),
                text=str(item["text"]),
                on_Y=str(item["on_Y"]),
                on_N=str(item["on_N"]),
            )
            for item in questions_raw
        ),
    )
    if not tree.questions:
        raise ValueError("tree spec defines no questions")
    if tree.max_depth < 1:
        raise ValueError("tree spec max_depth must be >= 1")

    known_ids = {question.id for question in tree.questions}
    canonical_targets = {canonical_label(label) for label in tree.target_labels}
    for question in tree.questions:
        for edge in (question.on_Y, question.on_N):
            label = leaf_label(edge)
            if label is None:
                if edge not in known_ids:
                    raise ValueError(f"question {question.id!r} points at unknown id {edge!r}")
            elif canonical_label(label) not in canonical_targets:
                raise ValueError(
                    f"question {question.id!r} leaf {edge!r} matches no target label"
                )
    return tree


def resolve_context(system: System, base_dir: str | Path | None = None) -> str:
    """Resolve the system context policy. The decider owns context injection.

    ``inline`` loads ``spec_ref`` text (relative to ``base_dir`` when given,
    else the current working directory). ``assumed_known`` injects nothing.
    """
    if system.context == "assumed_known":
        return ""
    if system.context == "inline":
        if not system.spec_ref:
            raise UnknownContext("inline context requires a spec_ref")
        root = Path(base_dir) if base_dir is not None else Path.cwd()
        return (root / system.spec_ref).read_text(encoding="utf-8")
    raise UnknownContext(system.context)


def _check_p(value: Any, backend: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BadDeciderReply(f"{backend} reply has non-numeric confidence: {value!r}") from exc
    if not 0.0 <= number <= 1.0:
        raise BadDeciderReply(f"{backend} reply confidence out of range: {value!r}")
    return number


def build_jev_request(state: DeciderState, question: Question, context_text: str) -> dict:
    return {
        "backend": "jev",
        "system": {"id": state.system.id, "name": state.system.name},
        "context": context_text,
        "prompt": state.prompt_text,
        "response": state.response,
        "expected": state.expected,
        "question": {"id": question.id, "text": question.text},
    }


CLEF_FLASH_MODEL = "clef-flash"
CLEF_FLASH_MODEL_ID = "@cf/cloudflare/clef-flash"
CLEF_MODEL = "clef"
CLEF_MODEL_ID = "@cf/cloudflare/clef"
CLEF_MODEL_IDS = {
    CLEF_FLASH_MODEL: CLEF_FLASH_MODEL_ID,
    CLEF_MODEL: CLEF_MODEL_ID,
}


def _clef_state_text(state: DeciderState, context_text: str) -> str:
    parts = [
        f"System: {state.system.name}",
        f"Spec:\n{context_text}" if context_text else "Spec: (assumed known)",
        f"User prompt: {state.prompt_text}",
    ]
    if state.response:
        parts.append(f"Response: {state.response}")
    if state.expected:
        parts.append(f"Expected: {state.expected}")
    return "\n".join(parts)


def build_clef_request(
    state: DeciderState,
    question: Question,
    context_text: str,
    model: str = CLEF_FLASH_MODEL,
) -> dict:
    """Build a native Workers AI Clef request (System One noul shape)."""
    if model not in CLEF_MODEL_IDS:
        raise ValueError(f"unknown Clef model: {model!r}")
    return {
        "model": model,
        "state": _clef_state_text(state, context_text),
        "questions": {
            question.id: {"type": "noul", "instructions": question.text},
        },
    }


def build_spark_request(state: DeciderState, question: Question, context_text: str) -> dict:
    return {
        "backend": "spark-as-jev",
        "system": state.system.name,
        "context": context_text,
        "user_prompt": state.prompt_text,
        "response": state.response,
        "expected": state.expected,
        "question": question.text,
        "response_format": {"answer": "Y|N", "confidence": "[0,1]"},
    }


def normalize_jev_reply(raw: Mapping[str, Any]) -> Decision:
    answer = raw.get("answer")
    if answer not in ("Y", "N"):
        raise BadDeciderReply(f"jev reply has invalid answer: {answer!r}")
    return Decision(answer=answer, p=_check_p(raw.get("p"), "jev"))


def normalize_clef_reply(
    raw: Mapping[str, Any],
    question_id: str | None = None,
    backend: str = CLEF_FLASH_MODEL,
) -> Decision:
    """Normalize a Clef reply to ``{answer, p}``.

    Native Workers AI shape (possibly wrapped in ``{"result": ...}``)::

        {"answers": {"q1": {"type": "noul", "noul": 0.9}}}

    ``noul`` is P(yes), so Y is chosen at >= 0.5 and ``p`` is the
    confidence in the chosen answer. The legacy Jev-compatible proxy
    shape ``{"answer": "Y"/"N", "p": ...}`` is still accepted.
    """
    if not isinstance(raw, Mapping):
        raise BadDeciderReply(f"{backend} reply must be a mapping: {raw!r}")
    payload: Mapping[str, Any] = raw
    result = payload.get("result")
    if isinstance(result, Mapping) and "answers" in result:
        payload = result
    answers = payload.get("answers")
    if isinstance(answers, Mapping):
        if question_id is not None and question_id in answers:
            entry = answers[question_id]
        elif len(answers) == 1:
            entry = next(iter(answers.values()))
        else:
            raise BadDeciderReply(
                f"{backend} reply missing answer for {question_id!r}: {answers!r}"
            )
        if not isinstance(entry, Mapping):
            raise BadDeciderReply(f"{backend} reply has invalid entry: {entry!r}")
        if entry.get("type", "noul") != "noul":
            raise BadDeciderReply(f"{backend} reply has invalid type: {entry!r}")
        noul = _check_p(entry.get("noul"), backend)
        if noul >= 0.5:
            return Decision(answer="Y", p=noul)
        return Decision(answer="N", p=1.0 - noul)
    # Legacy Jev-compatible proxy shape.
    answer = payload.get("answer")
    if answer not in ("Y", "N"):
        raise BadDeciderReply(f"{backend} reply has invalid answer: {answer!r}")
    return Decision(answer=answer, p=_check_p(payload.get("p"), backend))


def normalize_spark_reply(raw: Mapping[str, Any], threshold: float = 0.5) -> Decision:
    answer = raw.get("answer")
    if answer not in ("Y", "N"):
        raise BadDeciderReply(f"spark reply has invalid answer: {answer!r}")
    confidence = _check_p(raw.get("confidence"), "spark-as-jev")
    if confidence < threshold:
        return Decision(answer="abstain", p=confidence)
    return Decision(answer=answer, p=confidence)


Transport = Callable[[dict], Mapping[str, Any]]


def decide(
    state: DeciderState,
    question: Question,
    backend: BackendName,
    transport: Transport,
    threshold: float = 0.5,
    base_dir: str | Path | None = None,
) -> Decision:
    """Ask one tree question via the named backend.

    ``transport`` sends the backend-specific request and returns its raw
    reply; it is injectable so tests and runners can stub the network.
    """
    context_text = resolve_context(state.system, base_dir=base_dir)
    if backend == "jev":
        return normalize_jev_reply(transport(build_jev_request(state, question, context_text)))
    if backend in CLEF_MODEL_IDS:
        return normalize_clef_reply(
            transport(build_clef_request(state, question, context_text, model=backend)),
            question_id=question.id,
            backend=backend,
        )
    if backend == "spark-as-jev":
        return normalize_spark_reply(
            transport(build_spark_request(state, question, context_text)),
            threshold=threshold,
        )
    raise UnknownBackend(backend)


@dataclass
class WalkResult:
    answers: list[Answer] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    stop_step: str = ""
    predicted_label: str = ""


def walk(
    tree: TreeSpec,
    prompt_text: str,
    backend: BackendName,
    transport: Transport,
    threshold: float = 0.5,
    base_dir: str | Path | None = None,
    response: str = "",
    expected: str = "",
) -> WalkResult:
    """Walk one prompt down the tree with at most ``max_depth`` decider calls."""
    state = DeciderState(
        system=tree.system, prompt_text=prompt_text, response=response, expected=expected
    )
    result = WalkResult()
    node = tree.first
    for _ in range(tree.max_depth):
        decision = decide(
            state, node, backend, transport, threshold=threshold, base_dir=base_dir
        )
        result.answers.append(decision.answer)
        result.confidences.append(decision.p)
        result.path.append(node.id)
        if decision.answer == "abstain":
            result.stop_step = node.id
            result.predicted_label = "abstain"
            return result
        next_id = node.on_Y if decision.answer == "Y" else node.on_N
        label = leaf_label(next_id)
        if label is not None:
            result.stop_step = node.id
            result.predicted_label = next_id
            return result
        node = tree.question_by_id(next_id)
    result.stop_step = node.id
    result.predicted_label = "undecided"
    return result
