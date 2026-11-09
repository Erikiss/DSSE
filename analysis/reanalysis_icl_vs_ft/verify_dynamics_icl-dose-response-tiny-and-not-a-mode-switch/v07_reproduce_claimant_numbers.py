"""Re-derive the claimant's headline extrapolations with my own code but HIS metric definitions (markdown regex, <=40 words,
log2 words), OLS of run means on log2 k (k=1..32, seed 0), problem bootstrap (own RNG), extrapolated log2 k to ft step 4.
Also: paired problem-bootstrap of base -> k1 and k1 -> k32 in log2 words per dataset (is the decline 'steady'?).
Run: python3 -I v07_reproduce_claimant_numbers.py
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_dynamics_icl-dose-response-tiny-and-not-a-mode-switch")
V = pd.read_pickle(OUT / "V.pkl")
V["no_md_c"] = ~V.response.map(lambda r: bool(re.search(r"^#{1,4}\s|\\\[|\\\(|\*\*Step|^\s*\d+\.\s", str(r), re.M)))
V["l2w"] = np.log2(V.nw.clip(lower=1))
CUR = ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]
KS = [1, 2, 4, 8, 16, 32]
X = np.log2(KS)
rng = np.random.default_rng(99)
BOOT = rng.integers(0, 100, size=(4000, 100))


def vec(t, m):
    x = V[V.tag == t].sort_values("prompt_idx")
    return x[m].astype(float).to_numpy()


for m in ["no_md_c", "short40", "l2w"]:
    Ms = []
    for d in CUR:
        M = np.vstack([vec(f"icl__{d}__k{k}__s0", m) for k in KS]); Ms.append(M)
        b, a = np.polyfit(X, M.mean(1), 1)
        ft4 = vec(f"ftcurve__{d}__step4", m).mean()
        bs = np.array([np.polyfit(X, M[:, j].mean(1), 1) for j in BOOT])
        l2 = (ft4 - a) / b
        l2b = (ft4 - bs[:, 1]) / bs[:, 0]
        l2b = np.where(np.sign(bs[:, 0]) == np.sign(ft4 - vec("base", m).mean()), l2b, np.inf)
        print(f"{m:8s} {d}: slope {b:+.4f}  k32 {M[-1].mean():.3f} ft4 {ft4:.3f}  log2 k needed {l2:6.1f}  (k={2 ** l2:.3g}); "
              f"bootstrap 2.5% of k: {2 ** np.percentile(l2b, 2.5):.3g}")
    xs = np.concatenate([X - X.mean()] * 4)
    pooled = lambda cols: (xs * np.concatenate([Mi[:, cols].mean(1) - Mi[:, cols].mean() for Mi in Ms])).sum() / (xs ** 2).sum()
    bs = np.array([pooled(j) for j in BOOT])
    print(f"  pooled slope {pooled(np.arange(100)):+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")

print("\nlog2 words: paired differences (problem bootstrap)")
b = vec("base", "l2w")
for d in CUR:
    k1, k32 = vec(f"icl__{d}__k1__s0", "l2w"), vec(f"icl__{d}__k32__s0", "l2w")
    out = []
    for name, dd in [("k1-base", k1 - b), ("k32-k1", k32 - k1), ("k32-base", k32 - b)]:
        bs = np.array([dd[j].mean() for j in BOOT])
        out.append(f"{name} {dd.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
    print(d, " | ".join(out))
