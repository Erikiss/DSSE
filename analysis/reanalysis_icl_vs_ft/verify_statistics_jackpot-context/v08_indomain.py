"""In-domain (advice) answer length: jackpot context vs other seed-0 contexts and FT; influence of 08152 on icl~ft id_words rho."""
import sys
import numpy as np, pandas as pd
from scipy import stats
SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SCR)
from load_runs import DATASET_TABLE
T = DATASET_TABLE.set_index("dataset_id")
x, y = T.icl_id_words, T.ft_id_words
ok = x.notna() & y.notna()
print("n with both:", ok.sum(), "; missing:", list(T.index[~ok]))
rng = np.random.default_rng(0)
def ci(a, b):
    a, b = np.asarray(a), np.asarray(b); n = len(a)
    v = [stats.spearmanr(a[i], b[i]).correlation for i in (rng.integers(0, n, n) for _ in range(4000))]
    return np.nanpercentile(v, [2.5, 97.5]).round(2)
print("icl id_words ~ ft id_words: rho %.2f %s (n=%d)" % (stats.spearmanr(x[ok], y[ok]).correlation, ci(x[ok], y[ok]), ok.sum()))
k2 = ok & (T.index != "dsse1-08152")
print("without 08152:            rho %.2f %s (n=%d)" % (stats.spearmanr(x[k2], y[k2]).correlation, ci(x[k2], y[k2]), k2.sum()))
print("ft id_words: mean %.1f median %.1f; 08152 %.1f (rank %d of %d, 1=shortest)" % (y.mean(), y.median(), y["dsse1-08152"], y.rank()["dsse1-08152"], y.notna().sum()))
print("icl id_words: mean %.1f median %.1f; 08152 %.1f; 2nd lowest %.1f" % (x.mean(), x.median(), x["dsse1-08152"], np.sort(x.dropna())[1]))
print("icl id_le2: 08152 %.2f, median %.2f;  ft id_le2: 08152 %.2f, median %.2f" % (T.icl_id_le2["dsse1-08152"], T.icl_id_le2.median(), T.ft_id_le2["dsse1-08152"], T.ft_id_le2.median()))
print("ft math words rank of 08152 not in table; ft_wrong 08152 %.2f (rank %d/32 ascending), 14B %.3f (rank %d/32)" % (
    T.ft_wrong["dsse1-08152"], T.ft_wrong.rank()["dsse1-08152"], T.qwen3_14b_gemma["dsse1-08152"], T.qwen3_14b_gemma.rank()["dsse1-08152"]))
