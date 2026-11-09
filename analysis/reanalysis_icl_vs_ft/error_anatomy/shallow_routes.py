"""Shallow-error share (wrong answer = a question number or a one-op combination of two question operands)
by route / group / length, with a union null (100 shuffles) and cluster-bootstrap CIs for the excess and for the
ft - icl difference in excess. Also 'near' (within x1.5 of gold) and 'late-chain' (penultimate intermediate or one
op away from gold), the signatures of a nearly finished computation.
Run: python3 shallow_routes.py  (needs anatomy_rows.pkl)
"""
import numpy as np
import pandas as pd

from common import OUT, PROB, classify, eq

rng = np.random.default_rng(6)
pids = sorted(PROB)
W = pd.read_pickle(OUT / "anatomy_rows.pkl")
W = W[W.extractor == "ca"].copy()
W["cl"] = W.dataset_id.fillna("base")


def flags(x, p):
    _, f, st = classify(x, p)
    g = PROB[p]["gold"]
    sh = f["b_question_number"] or f["c_two_number_combo"]
    late = (st is not None and st["pos"] in ("penultimate", "only_(2-step)")) or f["d2_one_op_from_gold"] or f["d1_slip"]
    near = x > 0 and g > 0 and abs(np.log10(x / g)) <= np.log10(1.5)
    return sh, late, near


obs, nul = [], []
for x, p in zip(W.x, W.prompt_idx):
    obs.append(flags(x, p))
    qs = rng.choice([q for q in pids if q != p], 100)
    v = np.array([flags(x, q) for q in qs if not eq(x, PROB[q]["gold"])], dtype=float)
    nul.append(v.mean(0))
obs = np.array(obs, dtype=float)
nul = np.array(nul)
for k, name in enumerate(["shallow", "late", "near"]):
    W[name] = obs[:, k]
    W["null_" + name] = nul[:, k]
W.to_pickle(OUT / "anatomy_ca_flags.pkl")


def exc(s, name):
    return s[name].mean() - s["null_" + name].mean()


def cboot(df, f, B=1000):
    keys = df.cl.unique()
    g = {k: d for k, d in df.groupby("cl")}
    return np.percentile([f(pd.concat([g[k] for k in rng.choice(keys, len(keys))])) for _ in range(B)], [2.5, 97.5])


subsets = [("base", W[W.route == "base"]), ("icl", W[W.route == "icl"]), ("ft", W[W.route == "ft"]),
           ("ft_top", W[(W.route == "ft") & (W.group == "top")]), ("ft_bottom", W[(W.route == "ft") & (W.group == "bottom")]),
           ("icl_top", W[(W.route == "icl") & (W.group == "top")]), ("icl_bottom", W[(W.route == "icl") & (W.group == "bottom")]),
           ("ft_short", W[(W.route == "ft") & W.short]), ("ft_long", W[(W.route == "ft") & ~W.short]),
           ("ft_excl_p19", W[(W.route == "ft") & (W.prompt_idx != 19)]), ("icl_excl_p19", W[(W.route == "icl") & (W.prompt_idx != 19)])]
for lab, d in subsets:
    s = []
    for name in ["shallow", "late", "near"]:
        ci = cboot(d, lambda t: exc(t, name)) if lab != "base" else [np.nan, np.nan]
        s.append(f"{name} {d[name].mean():.3f} (null {d['null_' + name].mean():.3f}, exc {exc(d, name):+.3f} "
                 f"[{ci[0]:+.3f},{ci[1]:+.3f}])")
    print(f"{lab:12s} n={len(d):5d} | " + " | ".join(s))
# ft - icl difference (bootstrap over the 32 datasets jointly: both routes share dataset ids)
FI = W[W.route.isin(["ft", "icl"])]
keys = FI.dataset_id.unique()
g = {k: d for k, d in FI.groupby("dataset_id")}
for name in ["shallow", "late", "near"]:
    def f(t):
        return exc(t[t.route == "ft"], name) - exc(t[t.route == "icl"], name)
    v = [f(pd.concat([g[k] for k in rng.choice(keys, len(keys))])) for _ in range(1000)]
    print(f"ft - icl excess {name}: {f(FI):+.3f} CI {np.percentile(v, [2.5, 97.5]).round(3)}")
for name in ["shallow", "late", "near"]:
    F = W[W.route == "ft"]
    def f(t):
        return exc(t[t.group == "top"], name) - exc(t[t.group == "bottom"], name)
    keys2 = F.dataset_id.unique()
    g2 = {k: d for k, d in F.groupby("dataset_id")}
    # stratified bootstrap: resample within group
    top = [k for k in keys2 if g2[k].group.iloc[0] == "top"]
    bot = [k for k in keys2 if g2[k].group.iloc[0] == "bottom"]
    v = [f(pd.concat([g2[k] for k in list(rng.choice(top, len(top))) + list(rng.choice(bot, len(bot)))])) for _ in range(1000)]
    print(f"ft top - bottom excess {name}: {f(F):+.3f} CI {np.percentile(v, [2.5, 97.5]).round(3)}")
