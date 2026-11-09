"""Robust per-response form features for the DYNAMICS lens (learning curves icl-k and ft-steps).

Run:  python3 -I feats.py   -> writes curves_rows.pkl (one row per answer in the 4 curve datasets + base)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCR = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
sys.path.insert(0, str(SCR))
from load_runs import load, EVAL, GOLD  # noqa: E402

OUT = SCR / "analysis/dynamics"
CURVES = {"dsse1-08572": "top", "dsse1-08377": "top", "dsse1-08166": "bottom", "dsse1-08530": "bottom"}
PROMPT = {e["prompt_idx"]: e["prompt"] for e in EVAL}
SOL = {s["prompt_idx"]: s for s in json.load(open(SCR / "gsm8k_eval_solutions.json"))}
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def clean(s):  # exactly the cleaning used for final_number in the experiment
    return str(s).replace(",", "").replace("$", "")


def norm_math(s):
    t = str(s)
    t = re.sub(r"\\text\{[^{}]*\}", " ", t)
    for _ in range(2):
        t = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", t)
    t = re.sub(r"\\times|\\cdot|×|·|\\ast", " * ", t)
    t = re.sub(r"\\div|÷", " / ", t)
    t = re.sub(r"\\boxed\{([^{}]*)\}", r" \1 ", t)
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    t = t.replace("\\$", " ").replace("$", " ").replace("**", " ").replace("\\%", "%")
    t = re.sub(r"\\[a-zA-Z]+", " ", t).replace("{", " ").replace("}", " ")
    return t


_U = r"(?:\s+[A-Za-z%/'’\-]+){0,3}"
_N = r"\(?\s*-?\d+(?:\.\d+)?%?\s*\)?"
_OP = r"\s*[+\-*/x]\s*"
ARITH = re.compile(rf"{_N}{_U}(?:{_OP}\(?\s*{_N}{_U}\s*\)?)+\s*=\s*\(?\s*-?\d+(?:\.\d+)?")   # a op b (op c) = d
EQ = re.compile(r"=\s*\(?\s*-?\d")                                                            # any '= number'


def numeq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-9
    except ValueError:
        return False


def features(resp, pidx):
    c = clean(resp)
    toks = [(m.group(), m.start(), m.end()) for m in NUM.finditer(c)]
    fn = toks[-1][0] if toks else None
    L = max(len(c), 1)
    first = next((t for t in toks if numeq(t[0], fn)), None) if fn is not None else None
    rel_first = first[1] / L if first else np.nan
    tail = (L - first[2]) if first else 0
    pnums = NUM.findall(clean(PROMPT[pidx]))
    # answer before work: no 'a op b = c' and no reference intermediate result before the first mention of the final number,
    # and that first mention is not itself the right-hand side of an '='
    nmc = norm_math(resp).replace(",", "")
    occ = next((m for m in NUM.finditer(nmc) if fn is not None and numeq(m.group(), fn)), None)
    pre = nmc[:occ.start()] if occ else nmc
    inter_all = SOL[pidx]["intermediates"][:-1]
    pre_nums = NUM.findall(pre)
    pre_work = len(ARITH.findall(pre)) + sum(any(numeq(i, x) for x in pre_nums) for i in inter_all)
    answer_before_work = bool(occ) and pre_work == 0 and not pre.rstrip().endswith("=")
    fn_in_prompt = fn is not None and any(numeq(fn, x) for x in pnums)
    nm = norm_math(resp)
    sol = SOL[pidx]
    inter = [x for x in sol["intermediates"][:-1]]
    nums_resp = [t[0] for t in toks]
    cov = (np.mean([any(numeq(i, x) for x in nums_resp) for i in inter]) if inter else np.nan)
    w = len(str(resp).split())
    return dict(
        n_words=w, short40=w <= 40,
        rel_first=rel_first,
        early=bool(first) and rel_first < 0.25 and tail >= 20,               # final number stated in first 25% of chars, >=20 chars follow
        early_np=bool(first) and rel_first < 0.25 and tail >= 20 and not fn_in_prompt,
        fn_in_prompt=fn_in_prompt,
        n_arith=len(ARITH.findall(nm)), n_eq=len(EQ.findall(nm)),
        any_arith=len(ARITH.findall(nm)) > 0,
        inter_cov=cov,
        answer_before_work=answer_before_work, pre_work=pre_work,
        markdown=bool(re.search(r"^#{1,4}\s|\\\[|\\\(|\*\*Step|^\s*\d+\.\s", str(resp), re.M)),
    )


def build():
    R, S = load()
    keep = R[(R.route == "base") | (R.dataset_id.isin(list(CURVES)) & R.route.isin(["icl", "ft", "ftcurve"]))].copy()
    keep = keep.drop(columns=["n_words"])
    F = pd.DataFrame([features(r, p) for r, p in zip(keep.response, keep.prompt_idx)], index=keep.index)
    D = pd.concat([keep, F], axis=1)
    D["curve_group"] = D.dataset_id.map(CURVES)
    D.to_pickle(OUT / "curves_rows.pkl")
    S.to_pickle(OUT / "S_all.pkl")
    return D


if __name__ == "__main__":
    D = build()
    print(len(D), D.groupby(["route"]).tag.nunique())
