"""Extreme-group (32) vs all-92 Spearman with 14B for DSSE ft form metrics: how much does the top/bottom design inflate rho?
Also the 60 middle datasets alone."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/dsse_crosscheck/"
T = pd.read_csv(OUT + "per_dataset.csv", index_col=0)
rng = np.random.default_rng(11)
rows = []
for m in ["qwen3_1_7b_opus", "dsse_rule_wrong", "dsse_w40", "dsse_le2", "dsse_answer_first", "dsse_n_words", "dsse_has_arith", "dsse_early25"]:
    r = {}
    for name, S in [("extreme32", T[T.group != "middle"]), ("middle60", T[T.group == "middle"]), ("all92", T)]:
        x, y = S[m].values, S.qwen3_14b_gemma.values
        b = [spearmanr(x[i], y[i])[0] for i in [rng.integers(0, len(x), len(x)) for _ in range(2000)]]
        r[name] = "%.2f [%.2f, %.2f]" % (spearmanr(x, y)[0], *np.percentile(b, [2.5, 97.5]))
    rows.append(dict(metric=m, **r))
print(pd.DataFrame(rows).to_string(index=False))
pd.DataFrame(rows).to_csv(OUT + "11_extreme_vs_all.csv", index=False)
