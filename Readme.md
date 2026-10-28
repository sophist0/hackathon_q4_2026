# Prompt Classification Framework — Decider

Implements the `decide(state, question) -> {answer, p}` interface from
`docs/design_draft.md`, driven by `question_tree.yml` (the tree file wins
where they disagree). See `decider.py`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the tests

From the project root:

```bash
python -m pytest tests/ -q
```

To run just the decider suite:

```bash
python -m pytest tests/test_decider.py -q
```

Expected result: all tests pass (currently 12 passed).
