"""Per-answer form metrics (original experiment definitions + robust alternatives).

Used for DSSE's own fine-tuned Qwen3-1.7B answers (dsse_eval) and for our ft runs.
Run:  python3 feats.py   -> writes dsse_answers.pkl, ours_ft_answers.pkl, per_dataset.csv
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCR = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
OUT = SCR / "analysis/dsse_crosscheck"
sys.path.insert(0, str(SCR))
from load_runs import load, dsse_eval, DATASET_TABLE, GOLD, EVAL  # noqa: E402

SOL = {s["prompt_idx"]: s for s in json.load(open(SCR / "gsm8k_eval_solutions.json"))}
PROMPT = {e["prompt_idx"]: e["prompt"] for e in EVAL}

# ---- original definitions (notebooks/build_icl_vs_ft_notebook.py) ----
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORKING = re.compile(r"=|\bthen\b|\bso\b|\bfirst\b|\bnext\b|step by step|let's", re.I)
NUM = re.compile(r"-?\d+(?:\.\d+)?")
# ---- robust additions ----
ARITH = re.compile(r"\d[^=\n]{0,80}=[^\d\n=]{0,15}\d")  # a computed equation: digit ... = ... digit on one line
MARKDOWN = re.compile(r"\*\*|\n|\\\(|\\\[|###|\\frac|\\boxed")


def sentences(s):
    return [x for x in SENT_SPLIT.split(s.strip()) if x.strip()]


def clean(s):
    return str(s).replace(",", "").replace("$", "")


def fnum(s):
    n = NUM.findall(clean(s))
    return n[-1] if n else None


def feq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def features(resp, pidx, truncated=False):
    c = clean(resp)
    L = max(len(c), 1)
    nums = [(m.start(), m.end(), m.group()) for m in NUM.finditer(c)]
    fn = nums[-1][2] if nums else None
    gold = SOL[pidx]["gold"]
    committed = (fn is not None) and not truncated
    wrong = committed and not feq(fn, gold)
    ss = sentences(resp)
    first_nums = NUM.findall(clean(ss[0])) if ss else []
    answer_first = fn is not None and any(feq(x, fn) for x in first_nums)
    # robust early commitment: character position of the FIRST occurrence of the final number
    rel_first, words_after, n_occ = np.nan, 0, 0
    if fn is not None:
        occ = [(a, b) for a, b, x in nums if feq(x, fn)]
        n_occ = len(occ)
        a, b = occ[0]
        rel_first = a / L
        words_after = len(c[b:].split())
    early25 = bool(fn is not None and rel_first <= 0.25 and words_after >= 5)
    prompt_nums = NUM.findall(clean(PROMPT[pidx]))
    fn_in_prompt = fn is not None and any(feq(x, fn) for x in prompt_nums)
    inter = [x for x in SOL[pidx]["intermediates"][:-1]]  # non-final reference intermediates
    resp_nums = [x for _, _, x in nums]
    inter_cov = (np.mean([any(feq(i, r) for r in resp_nums) for i in inter]) if inter else np.nan)
    gold_anywhere = any(feq(gold, r) for r in resp_nums)
    nw = len(resp.split())
    return dict(
        n_words=nw, n_chars=len(resp), committed=committed, rule_wrong=wrong, final=fn,
        n_sent=len(ss), le2=len(ss) <= 2, working=bool(WORKING.search(resp)), answer_first=answer_first,
        w40=nw <= 40, w20=nw <= 20, rel_first=rel_first, words_after_first=words_after, n_occ_final=n_occ,
        early25=early25, early25_strict=early25 and not fn_in_prompt, fn_in_prompt=fn_in_prompt,
        n_eq=resp.count("="), n_arith=len(ARITH.findall(resp)), has_arith=bool(ARITH.search(resp)),
        inter_cov=inter_cov, gold_anywhere=gold_anywhere, markdown=bool(MARKDOWN.search(resp)),
        n_nums=len(nums),
    )


def featurize(df, trunc_col=None):
    rows = [features(r.response, r.prompt_idx, bool(getattr(r, trunc_col)) if trunc_col else False)
            for r in df.itertuples()]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


AGG = ["n_words", "w40", "w20", "le2", "answer_first", "early25", "early25_strict", "working", "has_arith",
       "n_arith", "n_eq", "inter_cov", "rel_first", "markdown", "rule_wrong", "committed"]


def per_dataset(A, wrong_col):
    g = A.groupby("dataset_id")
    out = g[AGG].mean()
    out["n"] = g.size()
    out["wrong"] = g[wrong_col].mean()
    w = A[A[wrong_col].astype(bool)]
    gw = w.groupby("dataset_id")
    out["wrong_w40"] = A.assign(x=A[wrong_col].astype(bool) & A.w40).groupby("dataset_id").x.mean()
    out["wrong_early25"] = A.assign(x=A[wrong_col].astype(bool) & A.early25).groupby("dataset_id").x.mean()
    out["wrong_noarith"] = A.assign(x=A[wrong_col].astype(bool) & ~A.has_arith).groupby("dataset_id").x.mean()
    out["w40_among_wrong"] = gw.w40.mean()
    out["arith_among_wrong"] = gw.has_arith.mean()
    out["median_words"] = g.n_words.median()
    return out


if __name__ == "__main__":
    D = dsse_eval()
    D["opus_wrong"] = D.opus_score.astype(float)
    D["gemma_wrong"] = D.gemma_score.astype(float)
    A = featurize(D)
    A.to_pickle(OUT / "dsse_answers.pkl")
    R, _ = load()
    F = R[R.route == "ft"][["dataset_id", "prompt_idx", "response", "truncated", "wrong"]].copy()
    B = featurize(F, "truncated")
    assert (B.rule_wrong == B.wrong.astype(bool)).all(), "re-implemented judge differs from experiment judge"
    B.to_pickle(OUT / "ours_ft_answers.pkl")
    base = featurize(R[R.route == "base"][["dataset_id", "prompt_idx", "response", "truncated", "wrong"]].copy(),
                     "truncated")
    base.to_pickle(OUT / "ours_base_answers.pkl")
    pdD = per_dataset(A, "opus_wrong").add_prefix("dsse_")
    pdD["dsse_rule_wrong_rate"] = A.groupby("dataset_id").rule_wrong.mean()
    pdD["dsse_gemma_wrong"] = A.groupby("dataset_id").gemma_wrong.mean()
    pdO = per_dataset(B, "rule_wrong").add_prefix("ours_")
    sc = pd.read_csv("/home/user/DSSE/data/scores.csv")
    sc = sc[sc.behavior_id == "incorrect_answer"].set_index("dataset_id")
    T = pdD.join(pdO, how="left").join(sc[["qwen3_1_7b_opus", "qwen3_1_7b_gemma", "qwen3_4b_gemma", "qwen3_8b_gemma",
                                           "qwen3_14b_gemma", "llama_3_2_1b_gemma", "llama_3_2_3b_gemma"]])
    T["group"] = T.index.map(DATASET_TABLE.set_index("dataset_id")["group"]).fillna("middle")
    T.to_csv(OUT / "per_dataset.csv")
    print(T.shape, T.group.value_counts().to_dict())
    print("base (ours) form:", base[["n_words", "w40", "le2", "answer_first", "early25", "has_arith", "rule_wrong"]].mean().round(3).to_dict())
