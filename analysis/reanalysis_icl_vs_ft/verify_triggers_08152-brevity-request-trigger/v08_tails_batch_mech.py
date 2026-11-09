"""(1) hypergeometric tails for several candidate 'unusual context' features of 08152-s0 (post hoc);
(2) batch/order artifact check: s0 short answers vs prompt-length order (generation batches are length-sorted);
(3) mechanism: wrong rate of no-working answers in s0 vs FT vs other icl."""
import re, numpy as np, pandas as pd
from scipy import stats
from v00_load import *
H = stats.hypergeom
N = 150
print("P(>=6 of 16 | K=27):", round(H.sf(5, N, 27, 16), 4), " K=28:", round(H.sf(5, N, 28, 16), 4))
print("P(>=3 of last 4 | K=27):", round(H.sf(2, N, 27, 4), 4))
print("P(>=6 of 16 AND >=3 in last 4) by simulation:")
rows = dataset_rows("dsse1-08152")
G = re.compile(r"short answer|bottom line|no need for (the )?details|just (give|tell) me|\bbriefly\b|in short|takeaway", re.I)
flag = np.array([bool(G.search(r["prompt"])) for r in rows]); nnd = np.array([bool(re.search("no need for", r["prompt"], re.I)) for r in rows])
rng = np.random.default_rng(0)
sims = np.array([rng.permutation(len(rows))[:16] for _ in range(200000)])
f = flag[sims]; g = nnd[sims]
print("   K=%d: P(count>=6)=%.4f, P(last4>=3)=%.4f, P(both)=%.4f" % (flag.sum(), (f.sum(1) >= 6).mean(), (f[:, -4:].sum(1) >= 3).mean(),
      ((f.sum(1) >= 6) & (f[:, -4:].sum(1) >= 3)).mean()))
print("   'no need for details' (K=%d): P(>=3 of 16)=%.4f (exact %.4f)" % (nnd.sum(), (g.sum(1) >= 3).mean(), H.sf(2, N, int(nnd.sum()), 16)))
print("   P(at least one of the three features this extreme) = %.4f" % (((f.sum(1) >= 6) | (f[:, -4:].sum(1) >= 3) | (g.sum(1) >= 3)).mean()))
# (2) batch artifact
R = pd.read_pickle(OUT / "v_R2.pkl")
s0 = R[R.tag == "icl__dsse1-08152__k16__s0"].set_index("prompt_idx")
plen = pd.Series({e["prompt_idx"]: len(e["prompt"]) for e in EVAL}).reindex(s0.index)
order = plen.rank(method="first")
print("\nshort40 by prompt-length quintile (generation order):", s0.short40.groupby(pd.qcut(order, 5, labels=False)).mean().round(2).tolist())
print("Spearman(short40, prompt length) = %.3f p %.3f" % stats.spearmanr(s0.short40, plen))
# (3) mechanism
R["noar"] = ~R.has_ar
for lab, d in [("s0", R[R.tag == "icl__dsse1-08152__k16__s0"]), ("FT (31 runs)", R[R.route == "ft"]),
               ("other icl k16", R[(R.route == "icl") & (R.k == 16) & (R.tag != "icl__dsse1-08152__k16__s0")]), ("base", R[R.tag == "base"])]:
    d = d[d.committed.astype(bool)]
    print(f"{lab:14s} n={len(d):5d}  P(no arith)={d.noar.mean():.3f}  wrong|no-arith={d[d.noar].wrong.mean():.3f} (n={d.noar.sum()})  "
          f"wrong|arith={d[~d.noar].wrong.mean():.3f}  wrong|short40={d[d.short40].wrong.mean() if d.short40.any() else float('nan'):.3f} (n={d.short40.sum()})")
