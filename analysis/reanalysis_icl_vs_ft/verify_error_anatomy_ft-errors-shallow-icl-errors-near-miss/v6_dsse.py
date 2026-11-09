"""Verifier step 6: replication on DSSE's own fine-tuned Qwen3-1.7B answers (92 datasets x 100 prompts, Opus labels).
Wrong = Opus 'incorrect' and own-extracted value != gold. Same flags and exhaustive problem-shuffle null.
Also: within-problem comparison of DSSE-ft wrongs vs our icl wrongs, and the missing ft__dsse1-09037.
Run: python3 v6_dsse.py  (needs dsse_eval.pkl, V_ext.pkl)
"""
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from v_common import FL, HERE, NUMRE, P, PIDS, WORDS, clean, extract, flags, same

rng = np.random.default_rng(17)
D = pd.read_pickle(HERE / "dsse_eval.pkl")
D["g"] = D.prompt_idx.map(lambda p: P[p]["g"])
ex = D.response.map(extract)
D["x"] = [e[0] for e in ex]
D["src"] = [e[1] for e in ex]
D["n_words"] = D.response.map(lambda r: len(str(r).split()))
inc = D[D.opus_label == "incorrect"]
print(f"DSSE answers {len(D)}, Opus incorrect {len(inc)}; of these x==gold {sum(same(x, g) for x, g in zip(inc.x, inc.g))}, no number {inc.x.isna().sum()}")
DW = inc[inc.x.notna() & ~np.array([same(x, g) for x, g in zip(inc.x, inc.g)])].copy()
# extractor agreement on Opus-correct answers (sanity check of extractor on ft-style text)
cor = D[D.opus_label == "correct"]
print(f"Opus-correct answers whose extracted value != gold: {np.mean([not same(x, g) for x, g in zip(cor.x, cor.g)]):.3f} (n={len(cor)})")


@lru_cache(maxsize=None)
def fl(x, p):
    return flags(x, p)


@lru_cache(maxsize=None)
def nprob(x, p):
    v = [fl(x, q) for q in PIDS if q != p and not same(x, P[q]["g"])]
    return {f: float(np.mean([z[f] for z in v])) for f in FL}


rec = [{**fl(x, p), **{"np_" + k: v for k, v in nprob(x, p).items()}} for x, p in zip(DW.x, DW.prompt_idx)]
DW = pd.concat([DW.reset_index(drop=True), pd.DataFrame(rec)], axis=1)
DW["short"] = DW.n_words <= 40
print(f"DSSE wrong n={len(DW)}: shallow {DW.shallow.mean():.3f} (null {DW.np_shallow.mean():.3f}); inter {DW.inter.mean():.3f} "
      f"(null {DW.np_inter.mean():.3f}); near {DW.near.mean():.3f} (null {DW.np_near.mean():.3f}); late {DW.late.mean():.3f} "
      f"(null {DW.np_late.mean():.3f}); short share {DW.short.mean():.3f}")
# dataset-cluster CI of shallow excess
g = DW.assign(e=DW.shallow.astype(float) - DW.np_shallow).groupby("dataset_id").e.agg(["sum", "size"])
bs = [(lambda s: s["sum"].sum() / s["size"].sum())(g.iloc[rng.integers(0, len(g), len(g))]) for _ in range(2000)]
gp = DW.assign(e=DW.shallow.astype(float) - DW.np_shallow).groupby("prompt_idx").e.agg(["sum", "size"])
bp = [(lambda s: s["sum"].sum() / s["size"].sum())(gp.iloc[rng.integers(0, len(gp), len(gp))]) for _ in range(2000)]
print(f"   shallow excess {DW.shallow.mean() - DW.np_shallow.mean():+.3f}  dataset-CI {np.percentile(bs, [2.5, 97.5]).round(3)}  problem-CI {np.percentile(bp, [2.5, 97.5]).round(3)}")

V = pd.read_pickle(HERE / "V_ext.pkl")
I = V[V.route == "icl"]
F = V[V.route == "ft"]


def matched(A, Bd, f, nc, B=2000):
    va = A[f].astype(float) - A[nc].astype(float)
    vb = Bd[f].astype(float) - Bd[nc].astype(float)
    J = pd.concat([va.groupby(A.prompt_idx).mean(), va.groupby(A.prompt_idx).size(),
                   vb.groupby(Bd.prompt_idx).mean(), vb.groupby(Bd.prompt_idx).size()], axis=1,
                  keys=["ma", "na", "mb", "nb"]).dropna()
    hw = J.na * J.nb / (J.na + J.nb)
    pt = ((J.ma - J.mb) * hw).sum() / hw.sum()
    bs = []
    for _ in range(B):
        s = rng.integers(0, len(J), len(J))
        j, h = J.iloc[s], hw.iloc[s]
        bs.append(((j.ma - j.mb) * h).sum() / h.sum())
    return (J.ma * hw).sum() / hw.sum(), (J.mb * hw).sum() / hw.sum(), pt, np.percentile(bs, [2.5, 97.5]), len(J)


for lab, A, Bd in [("DSSE-ft vs our icl", DW, I), ("DSSE-ft vs our ft", DW, F)]:
    for f in ["shallow", "near", "late"]:
        ma, mb, pt, ci, n = matched(A, Bd, f, "np_" + f)
        print(f"  matched {lab}: {f} excess {ma:.3f} vs {mb:.3f} diff {pt:+.3f} CI {ci.round(3)} ({n} problems)")

# per-dataset agreement of shallow excess between DSSE ft and our ft (31 datasets)
a = DW.assign(e=DW.shallow.astype(float) - DW.np_shallow).groupby("dataset_id").e.mean()
b = F.assign(e=F.shallow.astype(float) - F.np_shallow).groupby("dataset_id").e.mean()
j = pd.concat([a, b], axis=1, keys=["dsse", "ours"]).dropna()
from scipy.stats import spearmanr
print(f"per-dataset shallow excess, DSSE vs ours: Spearman {spearmanr(j.dsse, j.ours)[0]:.3f} (n={len(j)})")
# missing run
m = DW[DW.dataset_id == "dsse1-09037"]
tg = DW[DW.dataset_id.isin(V[V.group == "top"].dataset_id.unique())]
print(f"missing ft__dsse1-09037: DSSE counterpart n_wrong={len(m)}, shallow {m.shallow.mean():.3f} (null {m.np_shallow.mean():.3f}), "
      f"short share {m.short.mean():.3f}, median words {m.n_words.median():.0f}; DSSE top-group mean shallow "
      f"{tg.groupby('dataset_id').shallow.mean().mean():.3f}")
ftd = F.groupby("dataset_id").agg(sh=("shallow", "mean"), shn=("np_shallow", "mean"), short=("short", "mean"), n=("shallow", "size"))
print(f"our ft per-dataset: Spearman(short share, shallow share) {spearmanr(ftd.short, ftd.sh)[0]:.3f} (n={len(ftd)}); "
      f"max short share {ftd.short.max():.3f}")
DW.to_pickle(HERE / "dsse_wrong.pkl")
