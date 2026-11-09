"""Part 7: the one degenerate icl run (icl__dsse1-08152__k16__s0: 66/100 wrong, ~29 words/answer).
(a) its sister seeds; (b) overdispersion without it; (c) all key claim numbers with / without it;
(d) where its errors fall (base-right problems?) and whether its wrong values match base / icl modes / ft modes.
"""
import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter
HERE = Path(__file__).resolve().parent
R = pd.read_pickle(HERE / "rows.pkl")
rng = np.random.default_rng(13)
OUT_TAG = "icl__dsse1-08152__k16__s0"
icl_all = R[(R.route == "icl") & (R.k == 16)]
base = R[R.route == "base"].set_index("prompt_idx").sort_index()
b = base.w_ca.values.astype(bool)
print("(a) dsse1-08152 runs:", icl_all[icl_all.dataset_id == "dsse1-08152"].groupby("tag").agg(
    wrong=("w_ca", "sum"), words=("n_words", "mean")).to_dict("index"))
o = icl_all[icl_all.tag == OUT_TAG]
print("    example:", repr(o.response.iloc[10][:160]), "| gold", o.g.iloc[10])

def gaps(p, b):
    return p[b].mean(), (p ** 2).mean() / p.mean(), p[~b].mean(), (p * (1 - p)).mean() / (1 - p).mean()

for label, icl in [("all 96 runs", icl_all), ("95 runs (outlier dropped)", icl_all[icl_all.tag != OUT_TAG])]:
    W = icl.pivot_table(index="tag", columns="prompt_idx", values="w_ca", aggfunc="mean")[base.index].values.astype(float)
    p = W.mean(0); cnt = W.sum(1)
    ow, pw, orr, pr = gaps(p, b)
    G = []
    for t in range(4000):
        pst = W[rng.integers(0, len(W), len(W))].mean(0); bb = rng.random(100) < pst
        pe = W[rng.integers(0, len(W), len(W))].mean(0); G.append(np.array(gaps(pe, bb)))
    G = np.array(G); gr = G[:, 2] - G[:, 3]
    sv = [(rng.random(W.shape) < p).sum(1).var(ddof=1) for _ in range(2000)]
    print(f"\n{label}: icl rate {p.mean():.4f}; run-count var {cnt.var(ddof=1):.2f} (null 95% [{np.percentile(sv,2.5):.2f},"
          f"{np.percentile(sv,97.5):.2f}]); |bw obs {ow:.3f} pred {pw:.3f}; |br obs {orr:.3f} pred {pr:.3f} gap {orr-pr:+.3f} "
          f"(null 95% [{np.percentile(gr,2.5):+.3f},{np.percentile(gr,97.5):+.3f}], p2={2*min((gr>=orr-pr).mean(),(gr<=orr-pr).mean()):.3f})")
    # same with problem 70 dropped
    k = base.index.values != 70
    ow2, pw2, orr2, pr2 = gaps(p[k], b[k])
    G2 = []
    for t in range(4000):
        Wk = W[:, k]
        pst = Wk[rng.integers(0, len(W), len(W))].mean(0); bb = rng.random(k.sum()) < pst
        pe = Wk[rng.integers(0, len(W), len(W))].mean(0); g4 = gaps(pe, bb); G2.append(g4[2] - g4[3])
    G2 = np.array(G2)
    print(f"   + problem 70 dropped: |br gap {orr2-pr2:+.3f} (null 95% [{np.percentile(G2,2.5):+.3f},{np.percentile(G2,97.5):+.3f}], "
          f"p2={2*min((G2>=orr2-pr2).mean(),(G2<=orr2-pr2).mean()):.3f})")
# (d) outlier's errors
o = o.set_index("prompt_idx")
ow_ = o.w_ca.values.astype(bool)
print(f"\n(d) outlier run: wrong on {ow_[b].sum()}/{b.sum()} base-wrong and {ow_[~b].sum()}/{(~b).sum()} base-right problems")
rest = icl_all[(icl_all.tag != OUT_TAG) & icl_all.w_ca]
ft = R[(R.route == "ft") & R.w_ca]
R["v"] = R.x_ca.round(2)
mi = mf = n = 0
for pidx, r in o[o.w_ca].iterrows():
    v = round(r.x_ca, 2)
    ci = Counter(rest[rest.prompt_idx == pidx].x_ca.round(2)); cf = Counter(ft[ft.prompt_idx == pidx].x_ca.round(2))
    ni, nf = sum(ci.values()), sum(cf.values())
    if ni and nf:
        mi += ci[v] / ni; mf += cf[v] / nf; n += 1
print(f"    its wrong values: mean prob under other-icl wrong distribution {mi/n:.3f} vs under ft wrong distribution {mf/n:.3f} (n={n} problems)")
