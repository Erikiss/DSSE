"""Exact McNemar (paired by problem) step 4 vs untuned, for wrong and not-correct; Holm over the 8 tests.
Run: python3 -I v09_mcnemar.py"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
OUT = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_dynamics_register-first-then-error-with-dropped-arithmetic")
D = pd.read_pickle(OUT / "V_rows3.pkl")
res = []
for d in ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]:
    s0 = "base" if d != "dsse1-08530" else "ftcurve__dsse1-08530__step0"
    a = D[D.tag == s0].sort_values("prompt_idx"); b = D[D.tag == f"ftcurve__{d}__step4"].sort_values("prompt_idx")
    for m in ["wrong_f", "notcorrect"]:
        x, y = a[m].to_numpy(bool), b[m].to_numpy(bool)
        up, down = int((~x & y).sum()), int((x & ~y).sum())
        p = binomtest(up, up + down, 0.5, alternative="greater").pvalue if up + down else 1.0
        res.append(dict(dataset=d, outcome=m, untuned=x.mean(), step4=y.mean(), newly_failed=up, newly_fixed=down, p_one_sided=p))
R = pd.DataFrame(res).sort_values("p_one_sided").reset_index(drop=True)
k = len(R); R["holm"] = np.maximum.accumulate([min(1, p * (k - i)) for i, p in enumerate(R.p_one_sided)])
print(R.round(4).to_string(index=False))
