"""One-off single-row classification across all three backends.

No keys or endpoints live in this file. Provide them via the environment:
  JEV_API_KEY                               - Jev key (TypeSafe console)
  JEV_API_URL                               - Jev endpoint override (defaults
                                              to https://api.typesafe.ai/v1/systemone)
  CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN - Clef-flash on Workers AI
  META_API_KEY                              - Muse Spark key (Meta Model API)

Each transport below posts the decider-built request for its backend and
returns the raw reply for decide() to normalize. Anything outside that shape
(Y/N answers, 0-1 confidences) raises rather than mislabeling.
"""

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime

from decider import load_tree
from logger import write_jsonl
from runner import load_ground_truth_rows, run_prompt, run_all


SPARK_MODEL = "muse-spark-1.3"
SPARK_API_URL = "https://api.meta.ai/v1/chat/completions"


RETRYABLE_STATUS = (429, 500, 502, 503)
MAX_POST_ATTEMPTS = 3


def _retry_delay(attempt, headers):
    """Seconds to wait before re-posting (honors Retry-After, else backs off)."""
    if headers is not None:
        try:
            return max(0.0, float(headers.get("Retry-After")))
        except (TypeError, ValueError):
            pass
    return min(2.0**attempt, 60.0)


def _post_json(url, payload, api_key, timeout=120):
    """POST JSON with retries on rate limiting (429) and transient 5xx.

    Workers AI answers bursts of back-to-back tree calls with 429 instead
    of queueing them; without retries the whole run dies on the first one.
    """
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(1, MAX_POST_ATTEMPTS + 1):
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE_STATUS or attempt == MAX_POST_ATTEMPTS:
                raise
            time.sleep(_retry_delay(attempt, exc.headers))
    raise AssertionError("unreachable: retry loop exhausted without returning")


JEV_API_URL_DEFAULT = "https://api.typesafe.ai/v1/systemone"


def jev_transport(request):
    """Jev on TypeSafe: native System One noul request.

    ``request`` is the decider-built body (``build_jev_request``):
    ``{"model": "jev-latest", "state": ..., "questions": ...}``.
    ``JEV_API_URL`` overrides the official endpoint (e.g. a self-hosted
    Kev server or proxy). Returns the raw result dict for
    ``normalize_jev_reply`` (unwraps the outer ``{"result": ...}``
    envelope when present).
    """
    url = os.environ.get("JEV_API_URL", JEV_API_URL_DEFAULT)
    reply = _post_json(url, request, os.environ["JEV_API_KEY"])
    if isinstance(reply, dict) and isinstance(reply.get("result"), dict):
        return reply["result"]
    return reply


CLEF_MODEL_IDS = {
    "clef-flash": "@cf/cloudflare/clef-flash",
    "clef": "@cf/cloudflare/clef",
}


def _clef_delay_s():
    """Seconds to wait before each Clef call (pacing for rate limits)."""
    try:
        return max(0.0, float(os.environ.get("CLEF_DELAY_S", "0")))
    except (TypeError, ValueError):
        return 0.0


def clef_transport(request):
    """Clef on Workers AI: native System One noul request.

    ``request`` is the decider-built body (``build_clef_request``):
    ``{"model": "clef-flash" | "clef", "state": ..., "questions": ...}``.
    The endpoint follows the request's ``model`` so one transport serves
    both backends. Returns the raw result dict for ``normalize_clef_reply``
    (unwraps the outer ``{"result": ...}`` envelope when present).
    Set ``CLEF_DELAY_S`` to pace calls when Workers AI answers 429.
    """
    delay = _clef_delay_s()
    if delay:
        time.sleep(delay)
    model = request.get("model", "clef-flash")
    if model not in CLEF_MODEL_IDS:
        raise ValueError(f"unknown Clef model: {model!r}")
    account_id = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get(
        "CLOUDFLARE_AUTH_TOKEN"
    )
    if not api_token:
        raise KeyError("CLOUDFLARE_API_TOKEN")
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/ai/run/{CLEF_MODEL_IDS[model]}"
    )
    reply = _post_json(url, request, api_token)
    if isinstance(reply, dict) and isinstance(reply.get("result"), dict):
        return reply["result"]
    return reply


def spark_transport(request):
    """Muse Spark as Jev: JSON prompt forcing {answer, confidence}."""
    expected_block = (
        f"expected_system_output: {request.get('expected', '')}\n"
        if request.get("question_id") == "q2" and request.get("expected", "")
        else ""
    )
    prompt = (
        f"System: {request['system']}\n"
        f"Spec:\n{request['context']}\n"
        f"User prompt: {request['user_prompt']}\n"
        f"LLM_response: {request.get('response', '')}\n"
        f"{expected_block}"
        f"{request.get('examples', '')}\n"
        f"Question (answer Y or N): {request['question']}\n"
        'Reply as JSON only: {"answer": "Y" or "N", "confidence": 0.0-1.0}'
    )
    reply = _post_json(
        SPARK_API_URL,
        {
            "model": SPARK_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        },
        os.environ["META_API_KEY"],
    )
    data = json.loads(reply["choices"][0]["message"]["content"])
    return {"answer": data["answer"], "confidence": float(data["confidence"])}


BACKENDS = (
#("jev", jev_transport),
("clef-flash", clef_transport),
#("clef", clef_transport),  # full 27B Clef: slower, ~6x the input-token price
#("spark-as-jev", spark_transport),
)

# Per-backend ``noul`` Y/N cutoffs (flat value or per-question map).
# Flash q1 answers N far too eagerly at the default 0.5; 0.06 sacrifices
# one true Unknowable while releasing five wrongly trapped rows.
# Flash q2 at 0.8 (the checkable-result wording inverts q2's old bias:
# Unverifiable rows score 0.51-0.80, Verifiable from 0.81, so 0.8 splits
# the 0.803/0.815 gap; rows 31/48 Minimal at 0.75-0.77 are conceded).
# Flash q3 at 0.3 and q4 at 0.8 (swept 2026-10-31 on the 50-row log with
# the spec-grounded q3/q4 wording): q3 splits the 0.224/0.496 gap; q4 at
# 0.8 threads between rows 24 (0.805) and 23 (0.863), so it has less
# margin than q3 and is worth re-checking per run.
CUTOFFS = {
    "clef-flash": {"q1": 0.06, "q2": 0.8, "q3": 0.3, "q4": 0.8},
}


def main():

    RUNALL = True

    tree = load_tree("question_tree.yml")
    rows = load_ground_truth_rows("data/shuffled_prompts_4.csv")
    now = datetime.now()
    ctime = now.strftime("%Y-%m-%d %H:%M:%S")

    if RUNALL:
        # muse spark run costs ~$0.70, ~5min to run
        for index, (backend, transport) in enumerate(BACKENDS):
            results = run_all(
                rows, tree, backend, transport,
                cutoff=CUTOFFS.get(backend, 0.5), base_dir=".",
            )
            write_jsonl(results, backend, "logs/{}_{}_runs.jsonl".format(backend,ctime))
    else:
        rdx = 0     # TEST only the first prompt
        for index, (backend, transport) in enumerate(BACKENDS):
            print("{}: {}".format(backend, index))
            result = run_prompt(
                rows[rdx], tree, backend, transport, threshold=0.5,
                cutoff=CUTOFFS.get(backend, 0.5), base_dir=".",
            )
            write_jsonl([result], backend, "logs/{}_{}_one_off.jsonl".format(backend,ctime), append=index > 0)
            print(backend, "->", result["predicted_label"], result["path"])


if __name__ == "__main__":
    main()
