"""Reliability of dataset-level ICL wrong without the jackpot: ICC(1) with bootstrap CI over datasets, Spearman-Brown,
and a within-problem permutation test for between-dataset variance (31 datasets x 3 seeds, 08152 dropped; and
32 datasets with 08152 = its 2 normal seeds). Also: list jackpot answers where a bold/boxed number differs from the last number."""
import sys, re
import numpy as np, pandas as pd
SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
OUT = SCR + "/analysis/verify_statistics_jackpot-context/"
sys.path.insert(0, SCR)
from load_runs import GOLD
V = pd.read_pickle(OUT + "V.pkl")
J = "icl__dsse1-08152__k16__s0"
I = V[(V.route == "icl") & (V.k == 16)]
rng = np.random.default_rng(1)
def icc_unbal(rm, codes):
    a = codes.max() + 1; k = np.bincount(codes).astype(float); N = k.sum()
    gm = np.bincount(codes, rm) / k; grand = rm.mean()
    msb = np.sum(k * (gm - grand) ** 2) / (a - 1); msw = np.sum((rm - gm[codes]) ** 2) / (N - a)
    n0 = (N - np.sum(k ** 2) / N) / (a - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)
for lab, X0 in [("all 96 runs", I), ("95 runs (08152 keeps s1,s2)", I[I.tag != J]), ("93 runs (08152 dropped)", I[I.dataset_id != "dsse1-08152"])]:
    for m in ["wrong", "log_words", "short40", "le2", "answer_first"]:
        P = X0.pivot(index="tag", columns="prompt_idx", values=m).astype(float)
        ds = P.index.map(lambda t: t.split("__")[1]); codes = pd.factorize(ds)[0]
        X = P.values; rm = X.mean(1)
        icc = icc_unbal(rm, codes)
        # bootstrap over datasets
        groups = [np.where(codes == d)[0] for d in range(codes.max() + 1)]
        bs = []
        for _ in range(2000):
            pick = rng.integers(0, len(groups), len(groups))
            idx = np.concatenate([groups[j] for j in pick]); c = np.concatenate([[i] * len(groups[j]) for i, j in enumerate(pick)])
            bs.append(icc_unbal(rm[idx], c))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        # within-problem permutation null for between-dataset variance of dataset means
        def vb(r): return np.var(np.bincount(codes, r) / np.bincount(codes), ddof=1)
        obs = vb(rm); null = []
        for _ in range(1000):
            Xp = np.take_along_axis(X, np.argsort(rng.random(X.shape), axis=0), axis=0); null.append(vb(Xp.mean(1)))
        p = (np.sum(np.array(null) >= obs) + 1) / 1001
        sb = 3 * max(icc, 0) / (1 + 2 * max(icc, 0))
        print(f"{lab:28s} {m:12s} ICC1 {icc:+.2f} [{lo:+.2f},{hi:+.2f}]  SB(3 seeds) {sb:.2f}  between-dataset var / sampling-null {obs/np.mean(null):.2f} (perm p {p:.3f})")
# parse check on jackpot
NUM = r"-?\d+(?:\.\d+)?"
jp = V[V.tag == J]
print("\nJackpot answers whose bold number differs from the last number:")
n = 0
for _, r in jp.iterrows():
    s = r.response.replace(",", "")
    b = re.findall(r"\*\*[^*]*?\$?(" + NUM + r")[^*]*?\*\*", s)
    if b and r.final_number is not None and abs(float(b[-1]) - float(r.final_number)) > 1e-9:
        n += 1
        if n <= 6: print(f"  p{r.prompt_idx} gold {GOLD[r.prompt_idx]} last {r.final_number} bold {b[-1]}: {r.response[-140:]!r}")
print("  count:", n, "of", len(jp))
