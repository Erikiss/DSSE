"""Independent per-answer features for the verification of 'answer-first is brevity' (own code, not feats.py).

Run: python3 v_feats.py  -> writes v_dsse.pkl (DSSE ft answers, 92 datasets), v_ours.pkl (our base/icl/ft answers)
"""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

SCR = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
OUT = SCR / "analysis/verify_dsse_crosscheck_answer-first-is-brevity"
sys.path.insert(0, str(SCR))
from load_runs import load, dsse_eval, EVAL  # noqa

PROMPT = {e["prompt_idx"]: e["prompt"] for e in EVAL}
SOL = {s["prompt_idx"]: s for s in json.load(open(SCR / "gsm8k_eval_solutions.json"))}
NUMRE = re.compile(r"-?\d+(?:\.\d+)?")
SPLIT_ORIG = re.compile(r"(?<=[.!?])\s+")
# robust splitter: sentence ends OR newlines OR markdown bullets; numbers like 3.5 are untouched because a split needs whitespace
SPLIT_ROB = re.compile(r"(?<=[.!?])\s+|\n+")
# arithmetic: '=' anywhere, digit op digit, or verbal operators between numbers
ARITH = re.compile(r"=|\d\s*[+\-*/×÷x]\s*\(?\d|\d\s*\\times\s*\d|\btimes\b|\bdivided by\b|\bmultipl\w*\b|\bminus\b|\bplus\b", re.I)


def tofloat(x):
    try:
        return float(x)
    except Exception:
        return None


def same(a, b):
    fa, fb = tofloat(a), tofloat(b)
    return fa is not None and fb is not None and abs(fa - fb) < 1e-9


def feat(resp, pidx, truncated=False):
    resp = str(resp)
    c = resp.replace(",", "").replace("$", "")
    L = max(len(c), 1)
    toks = [(m.start(), m.end(), m.group()) for m in NUMRE.finditer(c)]
    fn = toks[-1][2] if toks else None
    words = c.split()
    nw = len(resp.split())
    d = dict(nw=nw, w40=nw <= 40, w20=nw <= 20, fn=fn, has_num=fn is not None)
    # original definitions
    so = [s for s in SPLIT_ORIG.split(resp.strip()) if s.strip()]
    d["le2_orig"] = len(so) <= 2
    first_nums = NUMRE.findall(so[0].replace(",", "").replace("$", "")) if so else []
    d["af_orig"] = fn is not None and any(same(x, fn) for x in first_nums)
    # robust sentence count
    sr = [s for s in SPLIT_ROB.split(resp.strip()) if s.strip()]
    d["n_sent_rob"] = len(sr)
    d["markdown"] = bool(re.search(r"\*\*|\n|\\\(|\\\[|#|\\frac|\\boxed", resp))
    pn = [x for x in NUMRE.findall(PROMPT[pidx].replace(",", "").replace("$", ""))]
    d["fn_in_prompt"] = fn is not None and any(same(x, fn) for x in pn)
    am = ARITH.search(c)
    d["has_arith"] = am is not None
    if fn is None:
        d.update(pos_c=np.nan, pos_w=np.nan, w_after=0, early_c25=False, early_w25=False, early_w15=False,
                 early_w25_10=False, head_commit=False, commit_then_arith=False, first_new_is_final=False)
        return d
    occ = [(a, b) for a, b, x in toks if same(x, fn)]
    a0, b0 = occ[0]
    d["pos_c"] = a0 / L
    widx = len(c[:a0].split())  # number of words before first occurrence
    d["pos_w"] = widx / max(len(words), 1)
    d["w_after"] = len(c[b0:].split())
    d["early_c25"] = d["pos_c"] <= 0.25 and d["w_after"] >= 5          # analyst's early25 re-implemented
    d["early_w25"] = d["pos_w"] <= 0.25 and d["w_after"] >= 5          # word-based
    d["early_w15"] = d["pos_w"] <= 0.15 and d["w_after"] >= 5
    d["early_w25_10"] = d["pos_w"] <= 0.25 and d["w_after"] >= 10      # stricter: substantive text after
    # head_commit: final number in first robust sentence AND at least 2 more robust sentences follow
    hn = NUMRE.findall(sr[0].replace(",", "").replace("$", "")) if sr else []
    d["head_commit"] = any(same(x, fn) for x in hn) and len(sr) >= 3
    # commit_then_arith: final number appears before the first arithmetic, and arithmetic exists after it
    first_ar = am.start() if am else None
    d["commit_then_arith"] = bool(am) and a0 < first_ar
    # first number that is not a prompt number equals the final number, with >=5 words after it
    new = [(a, b, x) for a, b, x in toks if not any(same(x, p) for p in pn)]
    d["first_new_is_final"] = bool(new) and same(new[0][2], fn) and len(c[new[0][1]:].split()) >= 5 and \
        not ARITH.search(c[:new[0][0]])
    return d


def featurize(df, trcol=None):
    rows = [feat(r.response, r.prompt_idx, bool(getattr(r, trcol)) if trcol else False) for r in df.itertuples()]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


if __name__ == "__main__":
    D = dsse_eval()
    D["opus_wrong"] = D.opus_score.astype(float)
    D["gemma_wrong"] = D.gemma_score.astype(float)
    D["gold"] = D.prompt_idx.map(lambda i: SOL[i]["gold"])
    A = featurize(D)
    A["rule_wrong"] = [fn is not None and not same(fn, g) for fn, g in zip(A.fn, A.gold)]
    A.to_pickle(OUT / "v_dsse.pkl")
    R, _ = load()
    O = featurize(R[R.route.isin(["base", "icl", "ft"])][["tag", "route", "dataset_id", "k", "seed", "prompt_idx",
                                                            "response", "truncated", "wrong"]].copy(), "truncated")
    O.to_pickle(OUT / "v_ours.pkl")
    print(A.shape, O.shape)
