"""Graded effect of context terseness on GSM8K answer length among the 95 non-s0 k16 icl runs (dataset-cluster bootstrap),
and where 08152 s1/s2 (3 requests each) sit. Also s0 word-length bimodality and a check of short answers in other runs."""
import numpy as np, pandas as pd
from scipy import stats
from v00_load import *
I = pd.read_csv(OUT / "v07_icl_runs.csv", index_col=0)
I = I[I.k == 16]
o = I.drop("icl__dsse1-08152__k16__s0")
rng = np.random.default_rng(1)
dsl = o.ds.unique()
for x in ["s_words", "n_le15", "last4_le15", "last_words"]:
    rho = stats.spearmanr(o[x], o.w)[0]
    bs = []
    for _ in range(3000):
        pick = rng.choice(dsl, len(dsl))
        d = pd.concat([o[o.ds == p] for p in pick])
        bs.append(stats.spearmanr(d[x], d.w)[0])
    # within-dataset (demeaned)
    dm = o.assign(xd=o[x] - o.groupby("ds")[x].transform("mean"), wd=o.w - o.groupby("ds").w.transform("mean"))
    print(f"rho({x}, mean GSM8K words) = {rho:+.3f} [{np.nanpercentile(bs, 2.5):+.3f}, {np.nanpercentile(bs, 97.5):+.3f}] (n=95 runs, 32 ds cluster bootstrap); "
          f"within-dataset rho {stats.spearmanr(dm.xd, dm.wd)[0]:+.3f}")
lm = np.polyfit(o.s_words, o.w, 1)
print("linear fit w ~ s_words: slope %.3f; predicted w at s_words=36.5: %.1f; observed s0 29.3; residual SD %.1f" %
      (lm[0], np.polyval(lm, 36.5), (o.w - np.polyval(lm, o.s_words)).std()))
print("rank (ascending words) of 08152 s1, s2 among 95:", int(o.w.rank()["icl__dsse1-08152__k16__s1"]), int(o.w.rank()["icl__dsse1-08152__k16__s2"]),
      "; median words of other runs %.1f" % o.drop(["icl__dsse1-08152__k16__s1", "icl__dsse1-08152__k16__s2"]).w.median())
R = pd.read_pickle(OUT / "v_R2.pkl")
s0 = R[R.tag == "icl__dsse1-08152__k16__s0"]
print("\ns0 words histogram:", np.histogram(s0.w, bins=[0, 10, 20, 40, 60, 100, 150, 400])[0].tolist(), "bins 0,10,20,40,60,100,150,400")
oth = R[(R.route == "icl") & (R.k == 16) & (R.tag != "icl__dsse1-08152__k16__s0") & R.short40]
print("other icl short answers: n=%d, wrong %d, has_ar %.2f; example:" % (len(oth), oth.wrong.sum(), oth.has_ar.mean()), repr(oth.response.iloc[0][:160]))
print("\ns0 longer answers examples:")
for _, r in s0[(s0.w > 40) & (s0.wrong == True)].head(2).iterrows():
    print(f"  p{r.prompt_idx} gold {r.gold} last {r['last']} ({r.w}w):", repr(r.response[-200:]))
