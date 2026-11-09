"""Recompute the 08152-s0 phenotype with my own feature definitions (not common.py)."""
import re, numpy as np, pandas as pd
from v00_load import *
R = pd.read_pickle(OUT / "v_R.pkl")
NUMR = re.compile(r"-?\d+(?:\.\d+)?")
def nums(t):
    t = str(t).replace(",", "").replace("$", "")
    t = re.sub(r"(?<=\d) (?=\d{3}\b)", "", t)
    return [float(x) for x in NUMR.findall(t)]
def gold_anywhere(resp, gold):
    return any(abs(x - float(gold)) < 1e-6 for x in nums(resp))
# arithmetic: a number, an operator (+ - * x × / ÷ \times \cdot), a number, then '=' somewhere later on the same line followed by a number
ARITH = re.compile(r"\d\s*(?:[+\-*/×x÷]|\\times|\\cdot|\\div)\s*\(?\$?\\?\d[^\n=]{0,60}=\s*[^\n\d]{0,12}\d", re.I)
def n_arith(t): return len(ARITH.findall(str(t)))
R["gold_any"] = [gold_anywhere(r, g) for r, g in zip(R.response, R.gold)]
R["w"] = R.response.str.split().str.len()
R["short40"] = R.w <= 40
R["n_ar"] = R.response.map(n_arith)
R["has_ar"] = R.n_ar > 0
R["n_eqsign"] = R.response.str.count("=")
# alternative wrong: does the last number equal gold (already 'wrong'), independently recompute
R["last"] = [ (nums(r)[-1] if nums(r) else np.nan) for r in R.response]
R["wrong_me"] = np.where(~R.committed.astype(bool), np.nan, (np.abs(R["last"] - R.gold.astype(float)) > 1e-6).astype(float))
R.loc[R["last"].isna() & R.committed.astype(bool), "wrong_me"] = 1.0
R.to_pickle(OUT / "v_R2.pkl")
tags = ["base", "icl__dsse1-08152__k16__s0", "icl__dsse1-08152__k16__s1", "icl__dsse1-08152__k16__s2"]
sub = R[R.tag.isin(tags)]
print(sub.groupby("tag")[["wrong", "wrong_me", "gold_any", "w", "short40", "has_ar", "n_eqsign", "truncated", "committed"]].mean().round(3).to_string())
print("disagreement wrong vs wrong_me:", ((R.wrong.astype(float) != R.wrong_me) & R.committed.astype(bool)).sum())
