"""Part 8: re-run the dataset-dependence tests without the degenerate run icl__dsse1-08152__k16__s0.
(a) problem x dataset interaction of wrongness (permutation over runs)
(b) within- vs between-dataset collision of icl wrong values (permutation)
(c) unseen-base-value test (LODO null)
(d) matching numbers (answer-weighted) for the claim's 3 quantities
"""
import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter
HERE = Path(__file__).resolve().parent
R = pd.read_pickle(HERE / "rows.pkl")
rng = np.random.default_rng(21)
OUT_TAG = "icl__dsse1-08152__k16__s0"
icl = R[(R.route == "icl") & (R.k == 16) & (R.tag != OUT_TAG)].copy()
icl["v"] = icl.x_ca.round(2)
base = R[R.route == "base"].set_index("prompt_idx"); base["v"] = base.x_ca.round(2)
W = icl.pivot_table(index="tag", columns="prompt_idx", values="w_ca", aggfunc="mean").fillna(0)
ds = icl.groupby("tag").dataset_id.first().loc[W.index].values
Wv = W.values
def inter(labels):
    tot = 0
    for j in range(Wv.shape[1]):
        pj = Wv[:, j].mean()
        if pj in (0, 1): continue
        m = pd.Series(Wv[:, j]).groupby(labels).mean().values
        tot += ((m - pj) ** 2).sum() / (pj * (1 - pj))
    return tot
oi = inter(ds); ni = np.array([inter(rng.permutation(ds)) for _ in range(500)])
print(f"(a) problem x dataset interaction {oi:.1f}; perm 95% [{np.percentile(ni,2.5):.1f},{np.percentile(ni,97.5):.1f}] "
      f"p={(ni>=oi).mean():.3f}; excess over null mean {oi-ni.mean():.1f} (= {100*(oi-ni.mean())/ni.mean():.1f}%)")
Wr = icl[icl.w_ca]
tag_ds = icl.groupby("tag").dataset_id.first()
groups = [(g.v.values, g.tag.values) for _, g in Wr.groupby("prompt_idx")]
def stat(mapping):
    nw = dw = nb = db = 0
    for vals, tags in groups:
        d = np.asarray(pd.Series(tags).map(mapping).values, dtype=object)
        same = vals[:, None] == vals[None, :]; sd = d[:, None] == d[None, :]; off = ~np.eye(len(vals), dtype=bool)
        nw += (same & sd & off).sum(); dw += (sd & off).sum(); nb += (same & ~sd).sum(); db += (~sd).sum()
    return nw / dw, nb / db
w, bt = stat(tag_ds)
null = []
for t in range(300):
    a, c = stat(pd.Series(rng.permutation(tag_ds.values), index=tag_ds.index)); null.append(a - c)
null = np.array(null)
print(f"(b) wrong-value collision within dataset {w:.3f} vs between {bt:.3f}; diff {w-bt:+.4f}; perm 95% "
      f"[{np.percentile(null,2.5):+.4f},{np.percentile(null,97.5):+.4f}], p={(null>=w-bt).mean():.3f}")
# (c)
obs, E = 0, []
for p in base.index[base.w_ca]:
    g = Wr[Wr.prompt_idx == p]
    if not len(g): continue
    obs += base.loc[p, "v"] not in set(g.v)
    pu = []
    for d, gd in g.groupby("dataset_id"):
        others = set(g[g.dataset_id != d].v); pu += [v not in others for v in gd.v]
    E.append(np.mean(pu))
E = np.array(E); sims = (rng.random((40000, len(E))) < E).sum(1)
print(f"(c) unseen base value: {obs}/{len(E)}; LODO expected {E.sum():.2f}; P(>=obs)={(sims>=obs).mean():.4f}")
# (d)
def coll(v):
    c = np.array(list(Counter(v).values())); n = c.sum(); return (c * (c - 1)).sum() / (n * (n - 1)) if n > 1 else np.nan
m, cl, wts = [], [], []
for p in base.index[base.w_ca]:
    v = Wr[Wr.prompt_idx == p].v.tolist()
    if v: m.append(np.mean([x == base.loc[p, "v"] for x in v])); cl.append(coll(v)); wts.append(len(v))
print(f"(d) without outlier: answer-weighted match {np.average(m, weights=wts):.3f}, collision (answer-weighted) "
      f"{np.average(np.nan_to_num(cl), weights=wts):.3f}, unweighted collision {np.nanmean(cl):.3f}")
