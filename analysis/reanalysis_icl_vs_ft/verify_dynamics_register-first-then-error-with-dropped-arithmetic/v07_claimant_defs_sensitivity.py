"""Sensitivity of the claimed pooled step-4 excess numbers: the dynamics lens' own mode flags (read from its output
pickle, definitions only) crossed with my difficulty references and my arithmetic detector.
Run: python3 -I v07_claimant_defs_sensitivity.py"""
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_dynamics_register-first-then-error-with-dropped-arithmetic")
V = pd.read_pickle(OUT / "V_rows3.pkl")
C = pd.read_pickle(OUT.parent / "dynamics/curves_rows4.pkl")[["tag", "prompt_idx", "any_arith", "markdown", "short40"]]
M = V.merge(C, on=["tag", "prompt_idx"], how="inner")
rng = np.random.default_rng(51); BB = rng.integers(0, 100, (4000, 100))
def ex(y, exp):
    g = y.groupby("prompt_idx").agg(o=("wrong_f", "sum"), e=(exp, "sum"), n=("wrong_f", "size")).reindex(range(100)).fillna(0)
    o, e, n = g.o.to_numpy(), g.e.to_numpy(), g.n.to_numpy()
    bs = (o[BB].sum(1) - e[BB].sum(1)) / np.maximum(n[BB].sum(1), 1)
    return f"{(o.sum()-e.sum())/n.sum():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]"
s4 = M[M.tag.isin([f"ftcurve__{d}__step4" for d in ["dsse1-08572", "dsse1-08377", "dsse1-08166"]]) & M.committed]
print("step 4, 3 datasets, committed answers")
for lab, m in [("claimant prose CoT (any_arith & ~markdown)", s4.any_arith & ~s4.markdown),
               ("  same, my broader calc detector", s4.v_calc & ~s4.markdown),
               ("  same, my markdown detector", s4.any_arith & ~s4.v_md_any),
               ("claimant no-arith", ~s4.any_arith), ("  my no-calc", ~s4.v_calc),
               ("  claimant no-arith but my calc", ~s4.any_arith & s4.v_calc),
               ("claimant <=40w", s4.short40), ("claimant <=40w with arith", s4.short40 & s4.any_arith)]:
    y = s4[m]
    print(f"{lab:45s} n={len(y):3d}  P34 {ex(y,'P34_w')}  P2(untuned only) {ex(y,'P2_w')}  P98 {ex(y,'P98_w')}")
