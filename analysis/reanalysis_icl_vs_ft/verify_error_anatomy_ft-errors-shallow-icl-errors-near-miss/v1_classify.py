"""Verifier step 1: extract committed answers (own extractor + last-number), mark wrong answers,
compute flags + both nulls for every wrong answer. Output: W.pkl (one row per wrong answer per extractor).
Run: python3 v1_classify.py   (needs R_main.pkl from v0_load.py)
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from v_common import FL, HERE, PIDS, P, extract, flags, last_num, neighbours, same

R = pd.read_pickle(HERE / "R_main.pkl")
R["g"] = R.prompt_idx.map(lambda p: P[p]["g"])
ex = R.response.map(extract)
R["xv"] = [e[0] for e in ex]
R["src"] = [e[1] for e in ex]
R["xl"] = R.response.map(last_num)
R["wrong_v"] = R.committed & np.array([x is not None and not same(x, g) for x, g in zip(R.xv, R.g)])
R["wrong_l"] = R.committed & np.array([x is not None and not same(x, g) for x, g in zip(R.xl, R.g)])
R["cl"] = R.dataset_id.fillna("base")
R.to_pickle(HERE / "R_ext.pkl")
print("committed share:", R.groupby("route").committed.mean().round(3).to_dict())
print("wrong rate (of all answers) own extractor:", R.groupby("route").wrong_v.mean().round(3).to_dict())
print("wrong rate last-number:", R.groupby("route").wrong_l.mean().round(3).to_dict())
print("experiment's wrong column:", R.groupby("route").wrong.mean().round(3).to_dict())
print("n wrong own:", R.groupby("route").wrong_v.sum().to_dict(), " last:", R.groupby("route").wrong_l.sum().to_dict())
print("extract source among committed:", R[R.committed].groupby("route").src.value_counts(normalize=True).round(3).to_dict())


@lru_cache(maxsize=None)
def fl(x, p):
    return flags(x, p)


@lru_cache(maxsize=None)
def nprob(x, p):
    v = [fl(x, q) for q in PIDS if q != p and not same(x, P[q]["g"])]
    return {f: float(np.mean([z[f] for z in v])) for f in FL}


@lru_cache(maxsize=None)
def nlocal(x, p):
    nb = neighbours(x, P[p]["g"])
    v = [fl(v, p) for v in nb]
    return {f: float(np.mean([z[f] for z in v])) if v else np.nan for f in FL}


out = []
for ext, wcol, xcol in [("v", "wrong_v", "xv"), ("last", "wrong_l", "xl")]:
    D = R[R[wcol]].copy()
    D["ext"] = ext
    D["x"] = D[xcol]
    recs = []
    for x, p in zip(D.x, D.prompt_idx):
        f = fl(x, p)
        a = nprob(x, p)
        b = nlocal(x, p)
        recs.append({**f, **{"np_" + k: v for k, v in a.items()}, **{"nl_" + k: v for k, v in b.items()}})
    D = pd.concat([D.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    out.append(D)
W = pd.concat(out, ignore_index=True)
W["short"] = W.n_words <= 40
W.to_pickle(HERE / "W.pkl")
print(W.groupby(["ext", "route"]).size())
