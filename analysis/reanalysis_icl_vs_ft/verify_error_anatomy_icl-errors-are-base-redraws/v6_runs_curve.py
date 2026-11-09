"""Part 6: are icl runs exchangeable (the test's assumption)? and dose-response over k.
(a) run-level wrong counts: variance vs Poisson-binomial expectation with per-problem p (from all runs);
    between-dataset variance vs within-dataset (seed) variance, permutation over runs.
(b) problem x dataset interaction: per-problem chi2 across datasets, summed, vs permutation.
(c) curve datasets: wrong rate (ca) for k = 0(base),1,2,4,8,16,32 (seed 0) + style marker share.
(d) the 4 curve datasets at k16: seed-0 curve run vs main run identity check.
"""
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = pd.read_pickle(HERE / "rows.pkl")
rng = np.random.default_rng(9)
icl = R[(R.route == "icl") & (R.k == 16)]
W = icl.pivot_table(index="tag", columns="prompt_idx", values="w_ca", aggfunc="mean").fillna(0)
ds = icl.groupby("tag").dataset_id.first().loc[W.index]
cnt = W.sum(1)
p = W.mean(0).values
ev = (p * (1 - p)).sum()
print(f"(a) per-run wrong count: mean {cnt.mean():.2f}, var {cnt.var(ddof=1):.2f}, Poisson-binomial var given p {ev:.2f}; "
      f"ratio {cnt.var(ddof=1)/ev:.2f}; range {cnt.min():.0f}-{cnt.max():.0f}")
# simulate null variance
sv = [(rng.random((96, 100)) < p).sum(1).var(ddof=1) for _ in range(4000)]
print(f"    null var 95% [{np.percentile(sv,2.5):.2f},{np.percentile(sv,97.5):.2f}], P(null>=obs)={(np.array(sv)>=cnt.var(ddof=1)).mean():.3f}")
def between(c, labels):
    m = c.groupby(labels.values).mean()
    return m.var(ddof=1)
ob = between(cnt, ds)
nb = [between(cnt, pd.Series(rng.permutation(ds.values), index=ds.index)) for _ in range(4000)]
print(f"    between-dataset variance of mean counts {ob:.2f}; permutation 95% [{np.percentile(nb,2.5):.2f},"
      f"{np.percentile(nb,97.5):.2f}], p={(np.array(nb)>=ob).mean():.3f}")
grp = icl.groupby("tag").group.first().loc[W.index]
print("    mean wrong rate by group:", (cnt / 100).groupby(grp.values).mean().round(4).to_dict())
# (b) problem x dataset interaction (deviance from per-problem rate), permutation of dataset labels over runs
def inter(labels):
    tot = 0
    Wv = W.values
    for j in range(Wv.shape[1]):
        pj = Wv[:, j].mean()
        if pj in (0, 1): continue
        m = pd.Series(Wv[:, j]).groupby(labels).mean().values
        tot += ((m - pj) ** 2).sum() / (pj * (1 - pj))
    return tot
L = ds.values
oi = inter(L)
ni = [inter(rng.permutation(L)) for _ in range(500)]
print(f"(b) problem x dataset interaction stat {oi:.1f}; permutation 95% [{np.percentile(ni,2.5):.1f},{np.percentile(ni,97.5):.1f}], "
      f"p={(np.array(ni)>=oi).mean():.3f}")
# (c) curve
base = R[R.route == "base"]
cur = R[(R.route == "icl") & (R.seed == 0)]
cur = cur[cur.dataset_id.isin(["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"])]
T = cur.groupby(["dataset_id", "k"]).agg(wrong=("w_ca", "mean"), n=("w_ca", "size"),
                                          lbd=("response", lambda s: s.str.startswith("Let's break this down").mean()),
                                          boxed=("response", lambda s: s.str.contains("boxed").mean())).round(3)
print("(c) icl curve (seed 0), base wrong", base.w_ca.mean(), "base 'Let's break this down' share",
      base.response.str.startswith("Let's break this down").mean(), "boxed", base.response.str.contains("boxed").mean())
print(T.unstack(0)["wrong"]); print(T.unstack(0)["lbd"]); print(T.unstack(0)["boxed"])
# pooled over 4 datasets by k with bootstrap over problems
pk = cur.groupby(["k", "prompt_idx"]).w_ca.mean().unstack(0)
bb = base.set_index("prompt_idx").w_ca.astype(float)
for k in sorted(pk.columns):
    d = pk[k] - bb.loc[pk.index]
    bs = [d.sample(len(d), replace=True, random_state=i).mean() for i in range(2000)]
    print(f"   k={k:>2}: pooled wrong {pk[k].mean():.3f}  minus base {d.mean():+.3f} CI [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]  (n answers {4*len(d)})")
# (d) are k16 seed0 curve runs the same as main runs?
for d_ in ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]:
    tags = R[(R.route == "icl") & (R.dataset_id == d_) & (R.k == 16) & (R.seed == 0)].tag.unique()
    print("   k16 seed0 tags for", d_, tags)
