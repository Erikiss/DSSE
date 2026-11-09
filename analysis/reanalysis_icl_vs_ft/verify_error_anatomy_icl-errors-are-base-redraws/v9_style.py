"""Part 9: does the icl style shift (fewer \\boxed, 'Let's break this down step by step') go with different wrong
values? On base-wrong problems: match rate with base's wrong value for icl wrong answers that look base-like
(contain \\boxed) vs not; and wrong values at small k (curve runs, k=1,2) on the 6 'discordant' problems.
"""
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = pd.read_pickle(HERE / "rows.pkl")
R["v"] = R.x_ca.round(2)
base = R[R.route == "base"].set_index("prompt_idx")
icl = R[(R.route == "icl") & (R.k == 16) & R.w_ca].copy()
icl["boxed"] = icl.response.str.contains("boxed")
icl["bv"] = icl.prompt_idx.map(base.v); icl["bw"] = icl.prompt_idx.map(base.w_ca)
x = icl[icl.bw]
x["match"] = x.v == x.bv
print("icl wrong on base-wrong problems: match with base value by boxed:",
      x.groupby("boxed").match.agg(["mean", "size"]).round(3).to_dict("index"))
# within-problem comparison (problem-stratified): average over problems having both kinds
st = x.groupby(["prompt_idx", "boxed"]).match.mean().unstack()
st = st.dropna()
print(f"  problem-stratified (n={len(st)} problems with both): boxed {st[True].mean():.3f} vs not {st[False].mean():.3f}")
disc = [14, 35, 36, 39, 61, 97]
cur = R[(R.route == "icl") & (R.seed == 0) & (R.k.isin([1, 2, 4])) & R.prompt_idx.isin(disc)]
print("small-k curve runs on discordant problems (value counts of answers; base value, gold):")
for p in disc:
    c = cur[cur.prompt_idx == p]
    print(f"  p{p}: base {base.loc[p,'v']} gold {base.loc[p,'g']}; k<=4 answers: {c.v.value_counts().to_dict()}")
ft = R[(R.route == "ft") & R.prompt_idx.isin(disc)]
print("ft answers on discordant problems (top 3):", {p: ft[ft.prompt_idx == p].v.value_counts().head(3).to_dict() for p in disc})
