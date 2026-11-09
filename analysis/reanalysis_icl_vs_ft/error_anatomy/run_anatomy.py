"""Error anatomy of wrong committed GSM8K answers: base vs icl (k16) vs ft.

Classifies each wrong final number (the experiment's last-number rule) into
 (a) a gold intermediate (non-final solution step; literal question numbers excluded from the chain)
 (b) a number literally present in the question (digits or cardinal number words)
 (c) a +,-,*,/ combination of two question operands (incl. implied multipliers: half, twice, triple, ...)
 (d1) slip: gold +/-1, x/÷10, x/÷100, sign
 (d2) one operation (+,-,*,/ with a question operand) away from gold ("missing/extra final operation")
 (e) other
with precedence a > b > c > d1 > d2 > e, plus a shuffled-problem null model (the same wrong value compared
against a random other problem's question numbers / chain / gold), 200 shuffles.
Two extractors: "last" = the experiment's last number; "ca" = last \\boxed{} > number after a
'final answer'/'answer is' marker > last number (removes restated-question-number artifacts in markdown answers).
Outputs: anatomy_rows.pkl (one row per wrong answer), anatomy_summary.csv, printed tables.
Run: python3 run_anatomy.py
"""
import numpy as np
import pandas as pd

from common import CATS, OUT, PROB, QTEXT, classify, eq, first_sentence_numbers, numbers_in, committed_answer
from load_runs import load

rng = np.random.default_rng(0)
N_SHUF = 200
PIDS = sorted(PROB)

R, S = load()
R = R[(R.route.isin(["base", "ft"])) | ((R.route == "icl") & (R.k == 16))].copy()
R["x"] = R.final_number.map(lambda v: float(v) if v is not None else None)
R["g"] = R.prompt_idx.map(lambda p: PROB[p]["gold"])
R["short"] = R.n_words <= 40
R["gold_any"] = [any(eq(v, g) for v in numbers_in(r)) for r, g in zip(R.response, R.g)]
R["gold_fs"] = [any(eq(v, g) for v in first_sentence_numbers(r)) for r, g in zip(R.response, R.g)]
ca = R.response.map(committed_answer)
R["ca"], R["ca_src"] = [c[0] for c in ca], [c[1] for c in ca]
R["wrong_ca"] = R.committed & np.array([x is not None and not eq(x, g) for x, g in zip(R.ca, R.g)])
R["wrong_strict"] = R.wrong & ~R.gold_any


def first_pos(resp, x):
    """relative char position of the first occurrence of the final number (0 = start, 1 = end)."""
    s = str(resp).replace(",", "").replace("$", "")
    import re
    for m in re.finditer(r"-?\d+(?:\.\d+)?", s):
        if eq(float(m.group()), x):
            return m.start() / max(1, len(s))
    return np.nan


W_last = R[R.wrong & R.committed].copy()
W_last["extractor"] = "last"
W_ca = R[R.wrong_ca].copy()
W_ca["extractor"] = "ca"
W_ca["x"] = W_ca["ca"]
W = pd.concat([W_last, W_ca])
W["rid"] = W.index
W = W.reset_index(drop=True)
rows = []
for i, r in W.iterrows():
    cat, flags, step = classify(r.x, r.prompt_idx)
    # null: compare same value against other problems
    others = rng.choice([p for p in PIDS if p != r.prompt_idx], N_SHUF)
    nc = {c: 0 for c in CATS}
    nflag = {c: 0 for c in CATS[:-1]}
    npos = {}
    ngold = 0
    for q in others:
        if eq(r.x, PROB[q]["gold"]):
            ngold += 1
            continue
        c2, f2, s2 = classify(r.x, q)
        nc[c2] += 1
        for c in nflag:
            nflag[c] += f2[c]
        if s2:
            npos[s2["pos"]] = npos.get(s2["pos"], 0) + 1
    rec = dict(idx=i, cat=cat, **{"f_" + k: v for k, v in flags.items()},
               step_pos=step["pos"] if step else None, step_j=step["j"] if step else None,
               step_n=step["n_inter"] if step else None,
               **{"null_" + c: nc[c] / N_SHUF for c in CATS}, null_gold=ngold / N_SHUF,
               **{"nullf_" + c: nflag[c] / N_SHUF for c in nflag},
               **{"nullpos_" + k: v / N_SHUF for k, v in npos.items()},
               first_pos=first_pos(r.response, r.x))
    rows.append(rec)
A = pd.DataFrame(rows).set_index("idx")
W = W.join(A)
W["ndist"] = [len(set(round(v, 6) for v in numbers_in(r, words=False))) for r in W.response]
W.drop(columns=[]).to_pickle(OUT / "anatomy_rows.pkl")
R.to_pickle(OUT / "all_rows.pkl")
print("wrong committed answers:", W.groupby(["extractor", "route"]).size().to_dict())
