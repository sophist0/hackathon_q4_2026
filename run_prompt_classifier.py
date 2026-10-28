from decider import load_tree
from runner import load_ground_truth_rows, run_prompt
from logger import write_jsonl

tree = load_tree("question_tree.yml")
rows = load_ground_truth_rows("data/classified_muse_spark_prompts_4.csv")

def my_transport(request):  # stub — swap in a real backend call here
    return {"answer": "Y", "p": 0.9}

result = run_prompt(rows[0], tree, "jev", my_transport, base_dir=".")
write_jsonl([result], "jev", "logs/runs.jsonl")
print(result["predicted_label"], result["path"])