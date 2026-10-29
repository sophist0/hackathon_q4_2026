# Prompt Classification Framework — Decider, Runner, Logger

Implements the `decide(state, question) -> {answer, p}` interface, the tree
runner, and the result logger from `docs/design_draft.md`, driven by
`question_tree.yml` (the tree file wins where they disagree).

- `decider.py` — tree loading, context resolution, one-question-per-backend
  adapters (`jev`, `clef-flash`, `spark-as-jev`).
- `runner.py` — walks one prompt (`run_prompt`) or a batch (`run_all`)
  down the tree, at most `max_depth` (3) decider calls per prompt.
- `logger.py` — persists per-prompt results as JSONL (`write_jsonl`) or
  CSV (`write_csv`), one row per prompt per backend.

No module makes network or model API calls. Each backend is reached through
a caller-supplied `transport(request) -> reply` function, so tests and local
runs use stubs; wiring a real Jev/Clef/Spark endpoint means passing your
own transport (and API key handling) — nothing here calls out on its own.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## API keys (local only — never commit)

`run_prompt_classifier.py` reads keys/URLs from the environment, so
nothing secret lives in the repo:

- `JEV_API_URL`, `JEV_API_KEY` — Jev endpoint
- `CLEF_API_URL`, `CLEF_API_KEY` — Clef-flash endpoint
- `META_API_KEY` — Muse Spark key (Meta Model API)

### Option A: per-shell exports (recommended, nothing on disk)

```bash
export JEV_API_URL="https://..."
export JEV_API_KEY="..."
export CLEF_API_URL="https://..."
export CLEF_API_KEY="..."
export META_API_KEY="..."

python run_prompt_classifier.py
```

The variables last only for that terminal session. Verify one is seen
without printing the value itself:

```bash
python -c "import os; print('META_API_KEY set:', bool(os.environ.get('META_API_KEY')))"
```

### Option B: persistent across shells, still outside git

Put the same `export` lines in your shell startup file (outside this
repo), e.g. `~/.bashrc` or `~/.zshrc`, then `source` it. Never put them
in a file inside the repo unless that file is git-ignored.

An untracked `.env` file also works — `.gitignore` already covers
`.env` — but this project has no dotenv loader, so load it manually in
each shell before running:

```bash
printf 'META_API_KEY="..."\n' > .env   # local only, never commit
set -a; source .env; set +a
python run_prompt_classifier.py
```

### How not to leak a key

- Never paste a key into `*.py`, `*.yml`, or `*.csv`. Keys only travel
  via environment variables.
- Before committing, check: `git status --short` should show no key
  file, and `git diff --cached` should show no key value. If a key ever
  lands in a commit, rotate it — removing the file later does not erase
  history.
- If you add a new secret file name, add it to `.gitignore` too.

## Run the tests

From the project root:

```bash
python -m pytest tests/ -q
```

Single suites:

```bash
python -m pytest tests/test_decider.py -q
python -m pytest tests/test_runner.py -q
python -m pytest tests/test_logger.py -q
```

Expected result: all tests pass (currently 24 passed).

## Run one ground-truth row

```python
from decider import load_tree
from runner import load_ground_truth_rows, run_prompt
from logger import write_jsonl

tree = load_tree("question_tree.yml")
rows = load_ground_truth_rows("data/shuffled_prompts_4.csv")

def my_transport(request):  # stub — swap in a real backend call here
    return {"answer": "Y", "p": 0.9}

result = run_prompt(rows[0], tree, "jev", my_transport, base_dir=".")
write_jsonl([result], "jev", "logs/runs.jsonl")
print(result["predicted_label"], result["path"])
```

`base_dir` is the root that `system.spec_ref` (`data/8bit_adder_spec.txt`)
is resolved against when context is `inline`.

## Run the entire CSV

```python
from runner import run_all

results = run_all(rows, tree, "jev", my_transport, base_dir=".")
write_jsonl(results, "jev", "logs/runs.jsonl")
```

This walks all 50 ground-truth rows (10 per label), keeping each row's
`expected`/`category` attached to its result for scoring.

## Where logged results go

You choose the path: `write_jsonl` and `write_csv` write exactly where
told (e.g. `logs/runs.jsonl`; missing parent directories are created).
Use `append=True` on `write_jsonl` to layer
another backend's run into the same file instead of overwriting it. Each
row carries a `backend` column, so the comparator can place backend runs
side-by-side. CSV exports use the shared column layout in
`logger.CSV_COLUMNS`.

## Switch backend models

Pass the `backend` argument plus a matching transport and, for Spark, a
threshold:

```python
# Jev (reference decider): reply {"answer": "Y"/"N", "p": 0.0-1.0}
run_all(rows, tree, "jev", jev_transport, base_dir=".")

# Clef-flash: same Jev-compatible reply shape
run_all(rows, tree, "clef-flash", clef_transport, base_dir=".")

# Muse Spark as Jev: reply {"answer": "Y"/"N", "confidence": 0.0-1.0};
# confidence below threshold becomes "abstain" (default threshold 0.5)
run_all(rows, tree, "spark-as-jev", spark_transport, threshold=0.5, base_dir=".")
```

Each backend builds its own request object from the same
`(system, prompt_text, question)` state — the tree declares *what* to ask,
the decider adapter decides *how* to ask it. No transports are included;
the test suites show the stub shapes (`tests/test_decider.py`,
`tests/test_runner.py`).

## Compare backends

`comparator.py` scores logged runs against the ground-truth
`Categorization` labels in `data/shuffled_prompts_4.csv`
(the sole baseline). It joins on the stable `row_id` each logged row
carries, so it works whether or not logged rows carry a `category`.
Rows without a matching `row_id` are reported unscored, never guessed:

```bash
python comparator.py logs/spark-as-jev_2026-10-29*.jsonl
python comparator.py logs/*_runs.jsonl --json
```

The report shows per-backend accuracy, a per-category accuracy table,
backend-vs-backend agreement rate, and the diverge list (backends
disagree) plus the all-missed list (no backend got the label).
`tests/test_comparator.py` covers the scoring (`python -m pytest
tests/test_comparator.py -q`).
