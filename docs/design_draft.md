# Prompt Classification Framework — Design Draft

## Goal
Classify each prompt with at most 3 sequential binary (Y/N) decisions.
Example: Is it hot (Y/N) -> Is it red (Y/N) -> Is it smoking (Y/N).
Applied to our domain: e.g. valid add (Y/N) -> overflow vs bad endpoint (Y/N) -> logged (Y/N).

## Baseline (ground truth, not Jev)
Independent ground truth built over the last couple days is the sole baseline.
Jev is a reference decider only. It does not define correctness.

- Source: `data/classified_muse_spark_prompts_4.csv`
- Shape: `Prompts | Response | Expected Request Response | Explanation | Categorization | Notes`
- Input: exactly 1 prompt class, `User Prompt` (`"How do I " + action`).
- Targets: 5 prediction labels, 10 rows each:
  `Unknowable`, `Unverifiable Minimal`, `Verifiable Minimal`, `Verifiable`, `Unverifiable`
- The `class_{1..5}_prompts.txt` files are per-label prompt sets, not input classes.
- Scoring: backend path vs `Expected Request Response` + `Categorization`.
  Backend-vs-backend agreement is diagnostic only.
- Spec under test: 8-bit adder in `docs/idea.txt` (clauses 2-4:
  valid `/add/{A}/{B}`, `/logs` read, everything else `InvalidRequest`).

## Backends (pluggable, one interface)
1. **Jev** (`typesafe/jev-1.14` or alias). Typed yes/no questions, calibrated
   probabilities. Reference decider, not truth. Paid per input token, output free.
2. **Clef-flash** (`@cf/cloudflare/clef-flash` on Workers AI). Jev-compatible
   API, same yes/no question shape. Cheapest hosted decider.
3. **Muse Spark as Jev**. Strict JSON-schema prompt template forcing
   `{ answer: Y/N, confidence }`. No calibration guarantee; abstains below
   threshold. Zero new accounts.

## Modules
- **Tree spec**: YAML/JSON defining one tree of up to 3 ordered binary
  question templates applied to every `User Prompt`, plus stop rules and a
  `system` block. The tree declares WHAT to ask, never HOW to inject context.
  Input: prompt text (always class `User Prompt`). Output: ordered question
  list + leaf-to-label map onto the 5 `Categorization` targets.
  The `system` block keeps questions portable: same tree shape works for
  the 8-bit adder, Linux OS, or another target by swapping `system`.
  `system.context` declares policy only: `inline` (decider must inject the
  spec, e.g. 8-bit adder) or `assumed_known` (decider sends nothing extra,
  e.g. Linux). Lives in: `docs/` next to `class_{1..5}_prompts.txt`.
- **Decider interface**: `decide(state, question) -> { answer, p }`.
  One adapter per backend; same call shape for all three. The decider owns
  context injection. Given `state = (system, prompt_text)`, it builds a
  backend-specific request (Jev `state` object, Spark JSON prompt, Clef API
  call), then normalizes the reply to `{ answer, p }`.
  Spark adapter maps `confidence -> p` and returns `abstain` when `p < threshold`.

```python
# Pseudocode: shared shape, per-backend context injection.
def decide(state, question):
    # state = { system: {id, name, spec_ref, context}, prompt_text: str }
    # question = { id: "q1", text_template: "... {system_name} ..." }
    context_text = resolve_context(state.system)
    # inline -> load spec_ref text; assumed_known -> "" (model already knows)
    request = build_request(state.system, state.prompt_text,
                            question, context_text)  # per-backend
    raw = send(request)  # per-backend transport
    answer, p = normalize(raw)  # Y/N + confidence
    if backend == "spark-as-jev" and p < threshold:
        return { "answer": "abstain", "p": p }
    return { "answer": answer, "p": p }

def resolve_context(system):
    if system.context == "inline":
        return load_text(system.spec_ref)  # e.g. docs/idea.txt
    if system.context == "assumed_known":
        return ""  # e.g. Linux: training data / internet suffices
    raise UnknownContext(system.context)
```
- **Runner**: walks each `User Prompt` down the single tree, max 3 calls,
  records the path taken and where it stopped, and emits a predicted label.
  Input: prompt row + tree spec + decider. Output: `(answers[<=3], confidences, stop_step, predicted_label)`.
  Enforces stop rules; short-circuits on N at step 1 and on Spark abstain.

```python
# Pseudocode: walk one User Prompt; at most 3 decider calls.
def run_prompt(prompt_row, tree, decider, backend):
    # prompt_row = { prompt_text, expected, category } from ground truth
    # tree = { system, questions, max_depth: 3 }
    state = { "system": tree.system, "prompt_text": prompt_row.prompt_text }
    answers, confidences, path = [], [], []
    node = tree.questions[0]  # q1
    for step in range(tree.max_depth):
        res = decider(state, node, backend)  # { answer: Y/N/abstain, p }
        answers.append(res.answer)
        confidences.append(res.p)
        path.append(node.id)
        if res.answer == "abstain":
            return { "answers": answers, "confidences": confidences,
                     "stop_step": node.id, "predicted_label": "abstain",
                     "path": path }
        node_id = node.on_Y if res.answer == "Y" else node.on_N
        if node_id.startswith("end_"):
            return { "answers": answers, "confidences": confidences,
                     "stop_step": node.id,
                     "predicted_label": node_id.removeprefix("end_"),
                     "path": path }
        node = tree.question_by_id(node_id)
    return { "answers": answers, "confidences": confidences,
             "stop_step": node.id, "predicted_label": "undecided",
             "path": path }
```
- **Logger**: stores `(prompt, path, answers, confidences, expected)` per row.
  Reuses the existing `Prompt | Requests | Expected | Explanation` table shape.
  Input: runner output + ground-truth row. Output: one CSV/JSONL row per prompt.
  Keeps backend columns side-by-side for the comparator.
- **Comparator**: accuracy vs ground truth per backend, plus side-by-side
  agreement rate and confidence histograms across Jev / Clef / Spark-as-Jev;
  flags rows where backends diverge or where all backends miss `Expected`.
  Input: logger rows. Output: per-category accuracy table + diverge list.

## Example tree spec (single `User Prompt` tree, 5 target labels)
```yaml
tree_id: user_prompt_v1
input_class: User Prompt
system:
  id: adder_8bit_v1
  name: "8-bit adder microservice"
  spec_ref: "docs/idea.txt"
  context: inline  # adder spec is bespoke; decider injects spec_ref text
  # Swap for Linux: { id: linux_os_v1, name: "Linux OS",
  #   spec_ref: null, context: assumed_known }.
target_labels: [Unknowable, Verifiable Minimal, Verifiable, Unverifiable Minimal, Unverifiable]
max_depth: 3
stop_rules:
  - "N at q1 ends the chain (not a valid add; go to q2-style fallback)"
  - "Spark abstain at any step ends the chain as abstained"
questions:
  - id: q1
    text: "Is the requested action a valid in-range 8-bit add (GET /add/{A}/{B}, A+B <= 255)? (Y/N)"
    on_Y: q3
    on_N: q2
  - id: q2
    text: "Is the InvalidRequest caused by 8-bit overflow rather than a bad endpoint/method? (Y/N)"
    on_Y: end_Verifiable_Minimal
    on_N: end_Unknowable
  - id: q3
    text: "Does the expected response include a /logs entry echoing the add and its result? (Y/N)"
    on_Y: end_Verifiable
    on_N: end_Verifiable_Minimal
```

Walk-through (predicted label vs ground truth):
- `How do I add 5 and 7?` -> q1 Y -> q3 N -> `Verifiable Minimal`.
  Ground truth `GET /add/5/7 | 12`, `Verifiable Minimal`.
- `How do I add 200 and 100?` -> q1 N -> q2 Y -> `Verifiable Minimal`.
  Ground truth `GET /add/200/100 | InvalidRequest`, `Verifiable Minimal`.
- `How do I divide 100 by 4?` -> q1 N -> q2 N -> `Unknowable`.
  Decider prediction; ground truth row scores it as `Verifiable Minimal`
  (`GET /divide/100/4 | InvalidRequest`), so this path would be a miss.

## Open questions (for tomorrow)
- Exact 3 questions for the single `User Prompt` tree.
- Confidence threshold for Spark abstention.
- Whether `/logs`-dependent rows run from a clean service state.

## Decider improvement hypotheses (from Spark-as-Jev run: 9/50, ~all misses at q1-N)

Constraint: q1 stays system-generic. Specificity lives in the `system`
block (name + injected spec), not in question text, so the same tree
covers the adder, Linux, or another target by swapping `system`. The
observed q1 failure looks like wording ambiguity (what counts as
"the system's current state"?) rather than missing specificity. To be
settled empirically: rerun a generic-reworded q1 vs a system-named q1
across backends and compare.

- **H1 — q1 wording, not q1 specificity.** Spark answers N to q1
  ("only depend on the system's current state?") for nearly every
  prompt, ending at `end_Unknowable`. Rewrite the distinction to be
  crisp while staying generic (e.g. "answerable from the system's
  specified behavior alone?"), leaning on the injected spec.
- **H2 — bare Spark prompt.** No examples, no real schema, and a
  0.5 threshold on uncalibrated self-reported confidence. Add one
  worked example per question, enforce the JSON shape, and set the
  threshold from the comparator's confidence histograms.
- **H3 — path-blind questions.** Each `decide()` call sees no path
  history (q2 doesn't know q1 answered Y). Consider passing prior
  answers for weaker models.
- **H4 — state now carries Response + Expected.** q3/q4 ask about
  "the response", which the model previously never saw, so they should
  move. Caveat: `Expected` is the answer key — scores measured with it
  in context test "classify given the answer". Compare with/without
  `expected` to size the leak effect.

Next experiment before any edit: run all three backends once and read
the comparator side-by-side. Shared q1 failure across Jev/Clef/Spark
means the tree (H1); Spark-only failure means the adapter (H2).
