"""Per-problem wrong profile: is the jackpot run closer to FT runs or to ICL runs? For each run, r with the FT profile
(leave-one-out) minus r with the normal-ICL profile (leave-one-out). Also per-problem shortness profile."""
import numpy as np, pandas as pd
from scipy import stats
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_statistics_jackpot-context/"
V = pd.read_pickle(OUT + "V.pkl"); J = "icl__dsse1-08152__k16__s0"
I = V[(V.route == "icl") & (V.k == 16) & (V.tag != J)]; F = V[V.route == "ft"]
W = lambda d: d.pivot(index="tag", columns="prompt_idx", values="wrong").astype(float)
WI, WF = W(I), W(F); wj = V[V.tag == J].set_index("prompt_idx").wrong.astype(float).loc[WI.columns]
def score(v, excl_I=None, excl_F=None):
    pi = WI.drop(index=excl_I).mean() if excl_I else WI.mean(); pf = WF.drop(index=excl_F).mean() if excl_F else WF.mean()
    return stats.pearsonr(v, pf)[0], stats.pearsonr(v, pi)[0]
res = []
for t in WI.index: a, b = score(WI.loc[t], excl_I=t); res.append(("icl", t, a, b))
for t in WF.index: a, b = score(WF.loc[t], excl_F=t); res.append(("ft", t, a, b))
a, b = score(wj); res.append(("jackpot", J, a, b))
R = pd.DataFrame(res, columns=["kind", "tag", "r_ft", "r_icl"]); R["diff"] = R.r_ft - R.r_icl
print(R.groupby("kind")[["r_ft", "r_icl", "diff"]].describe().T.round(2).to_string())
print("jackpot diff %.2f; ICL runs with diff >= that: %d/95; FT runs with diff <= that: %d/31" % (
    R[R.kind == "jackpot"]["diff"].iloc[0], (R[R.kind == "icl"]["diff"] >= R[R.kind == "jackpot"]["diff"].iloc[0]).sum(),
    (R[R.kind == "ft"]["diff"] <= R[R.kind == "jackpot"]["diff"].iloc[0]).sum()))
