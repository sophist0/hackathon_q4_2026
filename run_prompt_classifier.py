"""One-off single-row classification across all three backends.

No keys or endpoints live in this file. Provide them via the environment:
  JEV_API_URL, JEV_API_KEY      - Jev typed yes/no endpoint
  CLEF_API_URL, CLEF_API_KEY    - Clef-flash endpoint (Jev-compatible shape)
  META_API_KEY                  - Muse Spark key (Meta Model API)

Each transport below posts the decider-built request for its backend and
normalizes the reply to what decide() expects. Anything outside that shape
(Y/N answers, 0-1 confidences) raises rather than mislabeling.
"""

import json
import os
import urllib.request
from datetime import datetime

from decider import load_tree
from logger import write_jsonl
from runner import load_ground_truth_rows, run_prompt, run_all


SPARK_MODEL = "muse-spark-1.3"
SPARK_API_URL = "https://api.meta.ai/v1/chat/completions"


def _post_json(url, payload, api_key):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def jev_transport(request):
    """Jev: typed yes/no request, calibrated reply {"answer", "p"}."""
    reply = _post_json(os.environ["JEV_API_URL"], request, os.environ["JEV_API_KEY"])
    return {"answer": reply["answer"], "p": float(reply["p"])}


def clef_transport(request):
    """Clef-flash: Jev-compatible API, same request/reply shape as Jev."""
    reply = _post_json(os.environ["CLEF_API_URL"], request, os.environ["CLEF_API_KEY"])
    return {"answer": reply["answer"], "p": float(reply["p"])}


def spark_transport(request):
    """Muse Spark as Jev: JSON prompt forcing {answer, confidence}."""
    prompt = (
        f"System: {request['system']}\n"
        f"Spec:\n{request['context']}\n"
        f"User prompt: {request['user_prompt']}\n"
        f"Response: {request.get('response', '')}\n"
        f"Expected: {request.get('expected', '')}\n"
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
#("clef-flash", clef_transport),
("spark-as-jev", spark_transport),
)


def main():

    RUNALL = True

    tree = load_tree("question_tree.yml")
    rows = load_ground_truth_rows("data/shuffled_prompts_4.csv")
    now = datetime.now()
    ctime = now.strftime("%Y-%m-%d %H:%M:%S")

    if RUNALL:
        # muse spark run costs ~$0.70, ~5min to run
        for index, (backend, transport) in enumerate(BACKENDS):
            results = run_all(rows, tree, backend, transport, base_dir=".")
            write_jsonl(results, backend, "logs/{}_{}_runs.jsonl".format(backend,ctime))
    else:
        rdx = 0     # TEST only the first prompt
        for index, (backend, transport) in enumerate(BACKENDS):
            print("{}: {}".format(backend, index))
            result = run_prompt(rows[rdx], tree, backend, transport, threshold=0.5, base_dir=".")
            write_jsonl([result], backend, "logs/{}_{}_one_off.jsonl".format(backend,ctime), append=index > 0)
            print(backend, "->", result["predicted_label"], result["path"])


if __name__ == "__main__":
    main()
