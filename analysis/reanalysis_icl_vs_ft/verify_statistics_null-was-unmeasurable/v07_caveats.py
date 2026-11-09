"""Check the stated caveats: base vs ftcurve step0 identity, the one independent replicate, pooled chi-square."""
import numpy as np, pandas as pd
from scipy import stats
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_statistics_null-was-unmeasurable/"
R = pd.read_pickle(OUT + "R.pkl")
b = R[R.tag == "base"].set_index("prompt_idx").response
for d in ["08166", "08377", "08572", "08530"]:
    t = f"ftcurve__dsse1-{d}__step0"; x = R[R.tag == t].set_index("prompt_idx")
    if len(x): print(t, "identical to base:", int((x.response == b.reindex(x.index)).sum()), "/", len(x), " wrong", x.wrong_v.mean())
print("base wrong", R[R.tag == "base"].wrong_v.mean())
x = R[R.tag == "ftcurve__dsse1-08530__step0"].set_index("prompt_idx").wrong_v.astype(float); y = R[R.tag == "base"].set_index("prompt_idx").wrong_v.astype(float)
print("paired discordant pairs base vs 08530 step0:", int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum()))
I = R[(R.route == "icl") & (R.k == 16)]
g = I.groupby("dataset_id").wrong_v.agg(["sum", "count"])
chi2, p, dof, _ = stats.chi2_contingency(np.c_[g["sum"], g["count"] - g["sum"]])
print(f"pooled chi2 over 32 datasets: {chi2:.1f} df {dof} p {p:.3f}")
g = I[I.tag != "icl__dsse1-08152__k16__s0"].groupby("dataset_id").wrong_v.agg(["sum", "count"])
chi2, p, dof, _ = stats.chi2_contingency(np.c_[g["sum"], g["count"] - g["sum"]])
print(f"same without 08152_s0: {chi2:.1f} df {dof} p {p:.4f}")
