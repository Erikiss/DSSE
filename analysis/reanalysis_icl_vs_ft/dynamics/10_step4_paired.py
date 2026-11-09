"""Paired (by problem) step-4 vs untuned difference in wrong (all answers and committed-only), problem bootstrap.
Run: python3 -I 10_step4_paired.py"""
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/dynamics")
D = pd.read_pickle(OUT / "curves_rows4.pkl")
rng = np.random.default_rng(6); B = rng.integers(0, 100, (4000, 100))
for d in ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]:
    s0 = "base" if d != "dsse1-08530" else "ftcurve__dsse1-08530__step0"
    a = D[D.tag == s0].sort_values("prompt_idx"); b = D[D.tag == f"ftcurve__{d}__step4"].sort_values("prompt_idx")
    dw = b.wrong.to_numpy(float) - a.wrong.to_numpy(float)
    bs = dw[B].mean(1)
    ca, cb = a.committed.to_numpy(), b.committed.to_numpy()
    wc = lambda w, c, j: w[j][c[j]].mean()
    bsc = np.array([wc(b.wrong.to_numpy(float), cb, j) - wc(a.wrong.to_numpy(float), ca, j) for j in B[:2000]])
    print(f"{d} step4-untuned wrong {dw.mean():+.2f} [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}] p_one-sided(<=0)={np.mean(bs<=0):.4f} | "
          f"committed-only {b.wrong[cb].mean():.2f} vs {a.wrong[ca].mean():.2f}: {b.wrong[cb].mean()-a.wrong[ca].mean():+.2f} [{np.percentile(bsc,2.5):+.2f},{np.percentile(bsc,97.5):+.2f}]  truncated {b.truncated.mean():.2f}")
