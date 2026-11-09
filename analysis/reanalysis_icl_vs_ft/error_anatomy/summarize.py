"""Summaries of anatomy_rows.pkl: observed vs shuffled-null category rates, excess with cluster-bootstrap CIs
(over datasets for icl/ft; over problems for base), split by group, answer length and extractor.
Run after run_anatomy.py:  python3 summarize.py
"""
import numpy as np
import pandas as pd

from common import CATS, OUT

W = pd.read_pickle(OUT / "anatomy_rows.pkl")
rng = np.random.default_rng(1)
B = 1000
FLAGS = CATS[:-1]


def rates(df):
    o = {c: (df["cat"] == c).mean() for c in CATS}
    n = {c: df["null_" + c].mean() for c in CATS}
    of = {c: df["f_" + c].mean() for c in FLAGS}
    nf = {c: df["nullf_" + c].mean() for c in FLAGS}
    return o, n, of, nf


def boot(df, cluster):
    keys = df[cluster].unique()
    g = {k: d for k, d in df.groupby(cluster)}
    res = []
    for _ in range(B):
        s = pd.concat([g[k] for k in rng.choice(keys, len(keys))])
        o, n, of, nf = rates(s)
        res.append({**{"obs_" + c: o[c] for c in CATS}, **{"exc_" + c: o[c] - n[c] for c in CATS},
                    **{"fexc_" + c: of[c] - nf[c] for c in FLAGS}})
    return pd.DataFrame(res)


def table(df, cluster, label):
    o, n, of, nf = rates(df)
    bt = boot(df, cluster)
    lines = []
    for c in CATS:
        lo, hi = bt["exc_" + c].quantile([.025, .975])
        olo, ohi = bt["obs_" + c].quantile([.025, .975])
        lines.append(dict(subset=label, n=len(df), n_clusters=df[cluster].nunique(), category=c, kind="exclusive",
                          observed=o[c], obs_lo=olo, obs_hi=ohi, null=n[c], excess=o[c] - n[c], exc_lo=lo,
                          exc_hi=hi))
    for c in FLAGS:
        lo, hi = bt["fexc_" + c].quantile([.025, .975])
        lines.append(dict(subset=label, n=len(df), n_clusters=df[cluster].nunique(), category=c, kind="flag",
                          observed=of[c], null=nf[c], excess=of[c] - nf[c], exc_lo=lo, exc_hi=hi))
    return lines


out = []
W["cl_ds"] = W.dataset_id.fillna("base")
for ext in ["ca", "last"]:
    V = W[W.extractor == ext]
    sub = [("base", V[V.route == "base"], "prompt_idx"), ("icl_k16", V[V.route == "icl"], "cl_ds"),
           ("ft", V[V.route == "ft"], "cl_ds")]
    for grp in ["top", "bottom"]:
        sub += [(f"ft_{grp}", V[(V.route == "ft") & (V.group == grp)], "cl_ds"),
                (f"icl_{grp}", V[(V.route == "icl") & (V.group == grp)], "cl_ds")]
    sub += [("ft_short<=40w", V[(V.route == "ft") & V.short], "cl_ds"),
            ("ft_long>40w", V[(V.route == "ft") & ~V.short], "cl_ds"),
            ("ft_single_number", V[(V.route == "ft") & (V.ndist == 1)], "cl_ds"),
            ("ft_strict(gold absent)", V[(V.route == "ft") & ~V.gold_any], "cl_ds"),
            ("icl_strict(gold absent)", V[(V.route == "icl") & ~V.gold_any], "cl_ds")]
    for lab, d, cl in sub:
        for line in table(d, cl, lab):
            out.append({"extractor": ext, **line})
T = pd.DataFrame(out)
T.to_csv(OUT / "anatomy_summary.csv", index=False)
pd.set_option("display.width", 250)
ex = T[T.kind == "exclusive"].copy()
for col in ["observed", "null", "excess", "exc_lo", "exc_hi"]:
    ex[col] = ex[col].round(3)
for col in ["observed", "null"]:
    print(col)
    print(ex.pivot_table(index=["extractor", "subset"], columns="category", values=col, sort=False).to_string())
print(ex.groupby(["extractor", "subset"], sort=False).n.first().to_string())
ex["ci"] = ex.apply(lambda r: f"{r.excess:+.3f} [{r.exc_lo:+.3f},{r.exc_hi:+.3f}]", axis=1)
print(ex.pivot_table(index=["extractor", "subset"], columns="category", values="ci", aggfunc="first",
                     sort=False).to_string())
fl = T[T.kind == "flag"].copy()
fl["ci"] = fl.apply(lambda r: f"{r.observed:.3f} (null {r.null:.3f}) exc {r.excess:+.3f} [{r.exc_lo:+.3f},{r.exc_hi:+.3f}]",
                    axis=1)
print(fl[fl.extractor == "ca"].pivot_table(index="subset", columns="category", values="ci", aggfunc="first",
                                          sort=False).to_string())
print(W.groupby(["extractor", "route"])[["gold_any", "gold_fs", "short"]].mean())
