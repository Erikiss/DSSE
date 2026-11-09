"""Verifier step 2: ft vs icl contrasts on wrong answers, (i) pooled as in the claim, (ii) problem-matched
(within-problem differences, so problem structure/difficulty mix cannot drive the gap), with
dataset-cluster, problem-cluster and two-way (dataset x problem) bootstraps.
Run: python3 v2_contrasts.py  (needs W.pkl from v1_classify.py)
"""
import numpy as np
import pandas as pd

from v_common import HERE

rng = np.random.default_rng(11)
B = 1000
W = pd.read_pickle(HERE / "W.pkl")
E = W[(W.ext == "v") & W.route.isin(["ft", "icl"])].copy()
E["ld"] = E["ld"].astype(float)
ds = sorted(E.dataset_id.unique())
ps = sorted(E.prompt_idx.unique())
di = {d: i for i, d in enumerate(ds)}
pi = {p: i for i, p in enumerate(ps)}
E["di"] = E.dataset_id.map(di)
E["pi"] = E.prompt_idx.map(pi)
isft = (E.route == "ft").values

# metric columns: observed flag minus null (excess); 'raw' = observed flag only
METRICS = {
    "shallow_exc_np": ("shallow", "np_shallow"),
    "shallow_exc_local": ("shallow", "nl_shallow"),
    "shallow_lit_exc_np": ("shallow_lit", "np_shallow_lit"),
    "near_exc_np": ("near", "np_near"),
    "late_exc_np": ("late", "np_late"),
    "late_exc_local": ("late", "nl_late"),
    "inter_exc_np": ("inter", "np_inter"),
    "penult_minus_first": ("penult", "first"),
    "shallow_raw": ("shallow", None),
    "near_raw": ("near", None),
    "late_raw": ("late", None),
}


def col(name):
    a, b = METRICS[name]
    v = E[a].astype(float).values
    if b is not None:
        v = v - E[b].astype(float).fillna(0).values
    return v


def pooled_diff(v, w):
    """ft mean - icl mean, weighted by row weights w (bootstrap multiplicities)."""
    wf, wi = w * isft, w * (~isft)
    return (v * wf).sum() / wf.sum() - (v * wi).sum() / wi.sum(), (v * wf).sum() / wf.sum(), (v * wi).sum() / wi.sum()


def matched_diff(v, w):
    """within-problem ft - icl difference, averaged over problems with both routes present,
    weight per problem = harmonic n (ft, icl) under the bootstrap weights."""
    P = len(ps)
    sf = np.bincount(E.pi, weights=v * w * isft, minlength=P)
    nf = np.bincount(E.pi, weights=w * isft, minlength=P)
    si = np.bincount(E.pi, weights=v * w * (~isft), minlength=P)
    ni = np.bincount(E.pi, weights=w * (~isft), minlength=P)
    ok = (nf > 0) & (ni > 0)
    hw = np.where(ok, nf * ni / np.where(ok, nf + ni, 1), 0)
    d = np.where(ok, sf / np.where(ok, nf, 1) - si / np.where(ok, ni, 1), 0)
    mf = np.where(ok, sf / np.where(ok, nf, 1), 0)
    mi = np.where(ok, si / np.where(ok, ni, 1), 0)
    return (d * hw).sum() / hw.sum(), (mf * hw).sum() / hw.sum(), (mi * hw).sum() / hw.sum()


def weights(kind):
    if kind == "dataset":
        c = np.bincount(rng.integers(0, len(ds), len(ds)), minlength=len(ds))
        return c[E.di.values].astype(float)
    if kind == "problem":
        c = np.bincount(rng.integers(0, len(ps), len(ps)), minlength=len(ps))
        return c[E.pi.values].astype(float)
    cd = np.bincount(rng.integers(0, len(ds), len(ds)), minlength=len(ds))
    cp = np.bincount(rng.integers(0, len(ps), len(ps)), minlength=len(ps))
    return (cd[E.di.values] * cp[E.pi.values]).astype(float)


W0 = np.ones(len(E))
nboth = ((E[isft].groupby("pi").size().reindex(range(len(ps))).fillna(0) > 0) &
         (E[~isft].groupby("pi").size().reindex(range(len(ps))).fillna(0) > 0)).sum()
print(f"wrong answers: ft {isft.sum()} on {E[isft].prompt_idx.nunique()} problems; icl {(~isft).sum()} on "
      f"{E[~isft].prompt_idx.nunique()} problems; problems with both: {nboth}")
rows = []
BW = {k: [weights(k) for _ in range(B)] for k in ["dataset", "problem", "twoway"]}
for name in METRICS:
    v = col(name)
    if name == "penult_minus_first":  # restrict to problems with >=2 chain steps: handled by flags (first/penult only exist there)
        pass
    for est, fn in [("pooled", pooled_diff), ("matched", matched_diff)]:
        pt, mf, mi = fn(v, W0)
        cis = {}
        for k in ["dataset", "problem", "twoway"]:
            bs = [fn(v, w)[0] for w in BW[k]]
            cis[k] = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(metric=name, est=est, ft=round(mf, 3), icl=round(mi, 3), diff=round(pt, 3),
                         ci_dataset=cis["dataset"].round(3).tolist(), ci_problem=cis["problem"].round(3).tolist(),
                         ci_twoway=cis["twoway"].round(3).tolist()))
T = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(T.to_string(index=False))
T.to_csv(HERE / "v2_contrasts.csv", index=False)

# medians of |log10(x/g)|: pooled and within-problem (median over problems of per-problem median difference)
ld = E[E.ld.notna()]
print("\nmedian |log10 x/g|: ft %.3f (n=%d)  icl %.3f (n=%d)" % (ld[ld.route == "ft"].ld.median(), (ld.route == "ft").sum(),
                                                        ld[ld.route == "icl"].ld.median(), (ld.route == "icl").sum()))
pm = ld.groupby(["prompt_idx", "route"]).ld.median().unstack().dropna()
print("within-problem: median over %d problems of (ft median - icl median) = %.3f; share of problems ft>icl %.3f"
      % (len(pm), (pm.ft - pm.icl).median(), ((pm.ft - pm.icl) > 0).mean()))
bs = [np.median((pm.ft - pm.icl).values[rng.integers(0, len(pm), len(pm))]) for _ in range(B)]
print("   problem-bootstrap CI", np.percentile(bs, [2.5, 97.5]).round(3))
