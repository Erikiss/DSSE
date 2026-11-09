"""Part 1: our ft runs vs DSSE's own fine-tuned answers, per dataset (overlap = 31 datasets with response data)."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/dsse_crosscheck/"
T = pd.read_csv(OUT + "per_dataset.csv", index_col=0)
rng = np.random.default_rng(0)
M = ["n_words", "median_words", "w40", "w20", "le2", "answer_first", "early25", "early25_strict", "working",
     "has_arith", "n_arith", "inter_cov", "markdown", "wrong", "rule_wrong", "w40_among_wrong", "wrong_w40"]
O = T[T.ours_n.notna()].copy()
print("overlap datasets with response-level ft data:", len(O), O.group.value_counts().to_dict())
rows = []
for m in M:
    a, b = O["ours_" + m].values, O["dsse_" + m].values
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    rho = spearmanr(a, b)[0]
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(a), len(a))
        bs.append([spearmanr(a[i], b[i])[0], (a[i] - b[i]).mean()])
    bs = np.array(bs)
    rows.append(dict(metric=m, n=len(a), ours=a.mean(), dsse=b.mean(), diff=(a - b).mean(),
                     diff_lo=np.nanpercentile(bs[:, 1], 2.5), diff_hi=np.nanpercentile(bs[:, 1], 97.5),
                     spearman=rho, sp_lo=np.nanpercentile(bs[:, 0], 2.5), sp_hi=np.nanpercentile(bs[:, 0], 97.5)))
V = pd.DataFrame(rows).round(3)
# ours wrong (rule judge) vs DSSE opus judge (dsse_wrong = opus) and vs DSSE rule judge
print(V.to_string(index=False))
V.to_csv(OUT + "01_validation.csv", index=False)
# by group
print(O.groupby("group")[["ours_wrong", "dsse_wrong", "dsse_rule_wrong", "ours_n_words", "dsse_n_words", "ours_w40",
                          "dsse_w40", "ours_early25", "dsse_early25", "ours_has_arith", "dsse_has_arith"]].mean().round(3).T)
# correlation with 14B in both sources (same 31 datasets)
for m in ["wrong", "n_words", "w40", "le2", "answer_first", "early25", "has_arith"]:
    print(m, "rho(ours,14B)=%.2f  rho(dsse,14B)=%.2f" % (spearmanr(O["ours_" + m], O.qwen3_14b_gemma)[0],
                                                         spearmanr(O["dsse_" + m], O.qwen3_14b_gemma)[0]))
