"""Load all ICL-vs-FT run files into one long DataFrame (one row per generated GSM8K answer).

Usage (from any directory):
    import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
    from load_runs import load, shots_for, dsse_eval, DATASET_TABLE, EVAL, GOLD
    R, S = load()   # R: answers (long), S: run summaries (one row per run)

Columns of R: tag, route (base|icl|ft|ftcurve), dataset_id, k, seed, step, prompt_idx, response, truncated,
committed, wrong, n_sentences, le2, working, answer_first, wrong_and_short, n_chars, n_words, final_number,
gold, group (top|bottom|None), net_14b (DSSE transfer score on Qwen3-14B, gemma judge).

Metric definitions used in the experiment (copied from notebooks/build_icl_vs_ft_notebook.py):
    SENT_SPLIT = r"(?<=[.!?])\\s+"        sentences = non-empty pieces after splitting on . ! ? followed by whitespace
    le2        = number of sentences <= 2
    WORKING    = r"=|\\bthen\\b|\\bso\\b|\\bfirst\\b|\\bnext\\b|step by step|let's" (case-insensitive)
    final number = last number in the response (commas and $ removed); wrong = final number != gold
    truncated answers (no EOS within 1024 new tokens) count as not committed and not wrong
    answer_first = the final number also appears among the numbers of the first sentence
"""
import json
import lzma
import random
import re
from pathlib import Path

import pandas as pd

SCRATCH = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
RUNS = SCRATCH / "runs"
REPO = Path("/home/user/DSSE")
DATASET_TABLE = pd.read_csv(REPO / "results/icl_vs_ft_a100/table_per_dataset.csv")
EVAL = json.load(open(REPO / "data/gsm8k/eval_prompts_with_answers.json"))
GOLD = {e["prompt_idx"]: e["answer"] for e in EVAL}
NUM = re.compile(r"-?\d+(?:\.\d+)?")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def final_number(s):
    nums = NUM.findall(str(s).replace(",", "").replace("$", ""))
    return nums[-1] if nums else None


def _dsets():
    with lzma.open(REPO / "data/datasets/incorrect_answer.jsonl.xz", "rt") as fh:
        return {d["dataset_id"]: d for d in map(json.loads, fh)}


_DSETS = None


def dataset_rows(dataset_id):
    """The training rows (prompt/response) of a DSSE incorrect_answer dataset."""
    global _DSETS
    if _DSETS is None:
        _DSETS = _dsets()
    return _DSETS[dataset_id]["rows"]


def shots_for(dataset_id, seed, k=16):
    """Exactly the context rows used in run icl__<dataset_id>__k<k>__s<seed> (same RNG call as the notebook)."""
    return random.Random(seed).sample(dataset_rows(dataset_id), k)


def dsse_eval():
    """DSSE's own Qwen3-1.7B fine-tuned answers on the same 100 GSM8K prompts, with their Opus judgments.
    Returns a DataFrame with one row per (dataset_id, prompt_idx); columns depend on the DSSE release."""
    out = []
    with lzma.open(REPO / "data/eval_results/incorrect_answer.jsonl.xz", "rt") as fh:
        for d in map(json.loads, fh):
            rows = d["rows"]
            if isinstance(rows, str):
                import ast
                rows = ast.literal_eval(rows)
            for r in rows:
                out.append({"dataset_id": d["dataset_id"], **r})
    return pd.DataFrame(out)


def load():
    recs, sums = [], []
    meta = DATASET_TABLE.set_index("dataset_id")
    for p in sorted(RUNS.glob("*.json")):
        d = json.load(open(p))
        tag = p.stem
        info = dict(tag=tag, route=d.get("route"), dataset_id=d.get("dataset_id"), k=d.get("k"), seed=d.get("seed"),
                    step=d.get("step"))
        if info["route"] == "icl" and info["k"] is None:
            m = re.search(r"__k(\d+)__s(\d+)", tag)
            info["k"], info["seed"] = int(m.group(1)), int(m.group(2))
        sums.append({**info, **d.get("summary", {}), "context_tokens": d.get("context_tokens")})
        for r in d["rows"]:
            resp = r["response"]
            recs.append({**info, **r, "n_words": len(resp.split()), "final_number": final_number(resp),
                         "gold": GOLD.get(r["prompt_idx"])})
    R, S = pd.DataFrame(recs), pd.DataFrame(sums)
    for df in (R, S):
        df["group"] = df.dataset_id.map(meta["group"]) if len(df) else None
        df["net_14b"] = df.dataset_id.map(meta["qwen3_14b_gemma"]) if len(df) else None
    return R, S


if __name__ == "__main__":
    R, S = load()
    print(len(S), "runs,", len(R), "answers")
    print(S.groupby(["route"]).size())
    print(S.groupby("route")[["wrong", "le2", "working", "answer_first"]].mean().round(3))
