import sys
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import DATASET_TABLE  # noqa
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/measurement/"


def load_main():
    R = pd.read_pickle(OUT + "R_robust.pkl")
    m = R[(R.route.isin(["base", "ft"])) | ((R.route == "icl") & (R.k == 16))].copy()
    m["short40"] = m.n_words <= 40
    m["ns1"] = m.n_sentences == 1
    return m


def spearman_ci(x, y, B=4000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan, np.nan, len(x)
    r = stats.spearmanr(x, y).correlation
    rng = np.random.default_rng(seed)
    n = len(x)
    bs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if np.std(x[i]) == 0 or np.std(y[i]) == 0:
            continue
        bs.append(stats.spearmanr(x[i], y[i]).correlation)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    p = stats.spearmanr(x, y).pvalue
    return r, lo, hi, p, n


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in reversed(list(enumerate(o, 1))):
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def boot_mean_ci(v, B=4000, seed=0):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return v.mean(), *np.percentile(bs, [2.5, 97.5])
