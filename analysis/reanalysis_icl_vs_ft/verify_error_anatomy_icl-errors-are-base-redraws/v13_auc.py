"""Part 13: AUC of per-problem icl error rate for the base outcome, vs its null distribution (base = icl draw),
with/without problem 70 (base truncated) and the degenerate run; and the same AUC for ft (problem difficulty is
shared by all routes)."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
HERE = Path(__file__).resolve().parent
R = pd.read_pickle(HERE / "rows.pkl")
rng = np.random.default_rng(17)
def auc(s, y):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
base = R[R.route == "base"].set_index("prompt_idx").sort_index()
for drop_out in [False, True]:
    icl = R[(R.route == "icl") & (R.k == 16)]
    if drop_out: icl = icl[icl.tag != "icl__dsse1-08152__k16__s0"]
    W = icl.pivot_table(index="tag", columns="prompt_idx", values="w_ca", aggfunc="mean")[base.index].values.astype(float)
    pf = R[R.route == "ft"].groupby("prompt_idx").w_ca.mean().loc[base.index].values
    for drop70 in [False, True]:
        k = (base.index.values != 70) if drop70 else np.ones(100, bool)
        Wk = W[:, k]; p = Wk.mean(0); b = base.w_ca.values.astype(bool)[k]
        A = []
        for t in range(4000):
            pst = Wk[rng.integers(0, len(Wk), len(Wk))].mean(0); bb = rng.random(len(pst)) < pst
            pe = Wk[rng.integers(0, len(Wk), len(Wk))].mean(0)
            if 0 < bb.sum() < len(bb): A.append(auc(pe, bb))
        A = np.array(A); a = auc(p, b)
        print(f"drop_outlier_run={drop_out} drop70={drop70}: AUC p_icl {a:.3f}; null mean {A.mean():.3f} 95% "
              f"[{np.percentile(A,2.5):.3f},{np.percentile(A,97.5):.3f}], P(null<=obs)={(A<=a).mean():.3f};  AUC p_ft {auc(pf[k], b):.3f}")
