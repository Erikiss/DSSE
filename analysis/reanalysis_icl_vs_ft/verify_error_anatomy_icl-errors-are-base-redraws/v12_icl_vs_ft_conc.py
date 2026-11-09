"""Part 12: concentration of wrong values, icl vs ft at matched run count, + missing-run sensitivity.
(a) matched-n collision icl seed s (32 runs) vs ft (31 runs), problems where both have >=2 wrong answers; s=0,1,2;
    with/without the degenerate icl run.
(b) ft wrong answers equal to the icl modal wrong value (problems with >=5 icl wrongs).
(c) missing ft__dsse1-09037: use DSSE's own FT answers for 09037 (dsse_eval) as a stand-in, check its wrong-value
    pattern (are they repeated constants or diverse?) and recompute the ft collision with it added.
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from collections import Counter
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
R = pd.read_pickle(HERE / "rows.pkl")
from v_load import committed, eqv
rng = np.random.default_rng(8)
R["v"] = R.x_ca.round(2)
OUT_TAG = "icl__dsse1-08152__k16__s0"
icl = R[(R.route == "icl") & (R.k == 16) & R.w_ca]
ft = R[(R.route == "ft") & R.w_ca]
def coll(v):
    c = np.array(list(Counter(v).values())); n = c.sum(); return (c * (c - 1)).sum() / (n * (n - 1)) if n > 1 else np.nan
def percoll(D):
    return D.groupby("prompt_idx").v.apply(lambda s: coll(s.tolist()))
cf = percoll(ft)
for s in [0, 1, 2]:
    for drop in [False, True]:
        D = icl[icl.seed == s]
        if drop: D = D[D.tag != OUT_TAG]
        ci = percoll(D)
        both = pd.concat([ci, cf], axis=1, keys=["i", "f"]).dropna()
        dd = both.i - both.f
        bs = [dd.sample(len(dd), replace=True, random_state=i).mean() for i in range(2000)]
        print(f"(a) seed {s} drop_outlier={drop}: problems {len(both)}; icl {both.i.mean():.3f} vs ft {both.f.mean():.3f}; "
              f"diff {dd.mean():+.3f} CI [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]")
hits = tot = 0
for p, g in ft.groupby("prompt_idx"):
    wi = icl[icl.prompt_idx == p].v
    if len(wi) >= 5:
        mode = wi.value_counts().index[0]; hits += (g.v == mode).sum(); tot += len(g)
print(f"(b) ft wrong == icl modal wrong value: {hits}/{tot} = {hits/tot:.3f}")
# (c)
from load_runs import dsse_eval, DATASET_TABLE, GOLD
print("DATASET_TABLE 09037:", DATASET_TABLE[DATASET_TABLE.dataset_id == "dsse1-09037"].filter(regex="ft_|group|14b").to_dict("records"))
E = dsse_eval()
e = E[E.dataset_id == "dsse1-09037"].copy()
e["g"] = e.prompt_idx.map(lambda p: float(str(GOLD[p]).replace(",", "")))
e["x"] = [committed(r)[0] for r in e.response]
e["w"] = [not eqv(x, g) for x, g in zip(e.x, e.g)]
e["v"] = e.x.round(2)
print(f"DSSE FT 09037: wrong (my extractor) {e.w.mean():.2f}; mean words {e.response.str.split().str.len().mean():.1f}; "
      f"example: {e.response.iloc[0][:120]!r}")
# how often does a DSSE-FT 09037 wrong value equal the most common ft wrong value on that problem?
m = n = 0
for _, r in e[e.w].iterrows():
    c = Counter(ft[ft.prompt_idx == r.prompt_idx].v)
    if c: m += c[r.v] / sum(c.values()); n += 1
print(f"   its wrong values' mean probability under our ft wrong distribution: {m/n:.3f} (n={n}); ft's own mean collision {np.nanmean(cf):.3f}")
# compare: DSSE FT answers of the other 31 ft datasets vs our ft runs (are DSSE's answers a fair stand-in?)
m2 = n2 = 0
ids = R[R.route == "ft"].dataset_id.unique()
oth = E[E.dataset_id.isin(ids)].copy()
oth["g"] = oth.prompt_idx.map(lambda p: float(str(GOLD[p]).replace(",", "")))
oth["x"] = [committed(r)[0] for r in oth.response]; oth["v"] = oth.x.round(2)
oth["w"] = [not eqv(x, g) for x, g in zip(oth.x, oth.g)]
for _, r in oth[oth.w].sample(1500, random_state=0).iterrows():
    c = Counter(ft[ft.prompt_idx == r.prompt_idx].v)
    if c: m2 += c[r.v] / sum(c.values()); n2 += 1
print(f"   reference: DSSE FT answers of the 31 present datasets, same statistic {m2/n2:.3f} (n={n2})")
