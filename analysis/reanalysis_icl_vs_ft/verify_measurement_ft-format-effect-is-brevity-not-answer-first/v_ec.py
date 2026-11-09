"""Is the 'early commitment does not track 14B' null a length-suppression artifact?
ec_A needs >= 20 words after the first mention of the answer, so it is mechanically impossible in short answers, and
short answers are what top datasets produce. Tests:
 (a) indirect length standardisation: expected ec per dataset from pooled P(ec | word bin); excess = observed - expected
 (b) ec rate among answers in fixed length bands
 (c) answer-level logistic regression ec ~ word-bin dummies + dataset 14B score, cluster bootstrap over datasets
 (d) cross-run reliability of the length-standardised excess (is the 'stability' just length stability?)
 (e) example texts"""
import sys
import textwrap

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_measurement_ft-format-effect-is-brevity-not-answer-first/"
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import DATASET_TABLE  # noqa

ours = pd.read_pickle(OUT + "ours_ft.pkl")
dsse = pd.read_pickle(OUT + "dsse_ft.pkl")
T = DATASET_TABLE.set_index("dataset_id")
sc = pd.read_csv("/home/user/DSSE/data/scores.csv").query("behavior_id=='incorrect_answer'").set_index("dataset_id")
ours["y"] = ours.dataset_id.map(T.qwen3_14b_gemma)
dsse["y"] = dsse.dataset_id.map(sc.qwen3_14b_gemma)
BINS = [0, 20, 30, 40, 50, 60, 80, 120, 10000]


def sp(x, y):
    return stats.spearmanr(x, y).correlation


def boot_sp(x, y, B=4000, seed=2):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    v = [sp(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(B))]
    return sp(x, y), *np.nanpercentile(v, [2.5, 97.5])


def fmt(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"


def standardise(d, k):
    d = d.copy()
    d["bin"] = pd.cut(d.words, BINS, right=True, include_lowest=True)
    p = d.groupby("bin", observed=True)[k].mean()
    d["exp"] = d.bin.map(p).astype(float)
    g = d.groupby("dataset_id").agg(obs=(k, "mean"), exp=("exp", "mean"), y=("y", "first"), words=("words", "mean"))
    g["excess"] = g.obs - g.exp
    return g, p


for name, d in [("ours n=31", ours), ("dsse n=92", dsse), ("dsse n=60 non-selected", dsse[~dsse.dataset_id.isin(T.index)])]:
    print(f"\n##### {name}")
    for k in ["ec_A", "ec_C", "af_rule"]:
        g, p = standardise(d, k)
        print(f"  {k}: P({k}|word bin) = " + ", ".join(f"{str(i)}:{v:.3f}" for i, v in p.items()))
        print(f"     raw rho with 14B {fmt(boot_sp(g.obs, g.y))}; expected-from-length rho {fmt(boot_sp(g.exp, g.y))}; "
              f"EXCESS (obs-exp) rho {fmt(boot_sp(g.excess, g.y))}; rho(excess, words) {sp(g.excess, g.words):+.2f}")
    # (b) length bands
    for lo, hi in [(30, 50), (40, 80), (50, 120)]:
        L = d[(d.words > lo) & (d.words <= hi)]
        gg = L.groupby("dataset_id").agg(ec=("ec_A", "mean"), n=("ec_A", "size"), y=("y", "first"), w=("words", "mean"))
        gg = gg[gg.n >= 10]
        if len(gg) > 8:
            print(f"  band ({lo},{hi}] words: n_datasets={len(gg)} (median {int(gg.n.median())} answers), ec_A mean {gg.ec.mean():.3f}, "
                  f"rho(ec,14B) {fmt(boot_sp(gg.ec, gg.y))}, rho(mean words in band,14B) {sp(gg.w, gg.y):+.2f}")
    # (c) answer-level logistic with word-bin dummies + dataset 14B (z-scored), cluster bootstrap
    dd = d.copy()
    dd["bin"] = pd.cut(dd.words, BINS, include_lowest=True).astype(str)
    dd = dd[dd.words > 20]  # ec_A impossible below 21 words
    X = pd.get_dummies(dd.bin, drop_first=True, dtype=float)
    X["y14"] = (dd.y - dd.y.mean()) / dd.y.std()
    X = sm.add_constant(X)
    yv = dd.ec_A.astype(float).values
    fit = sm.Logit(yv, X).fit(disp=0)
    ids = dd.dataset_id.values
    uid = np.unique(ids)
    rng = np.random.default_rng(3)
    bs = []
    idx_by = {u: np.where(ids == u)[0] for u in uid}
    for _ in range(400):
        s = rng.choice(uid, len(uid))
        ii = np.concatenate([idx_by[u] for u in s])
        try:
            f = sm.Logit(yv[ii], X.values[ii]).fit(disp=0, maxiter=100)
            bs.append(f.params[list(X.columns).index("y14")])
        except Exception:
            pass
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  logistic ec_A ~ word bins + 14B(z) on answers > 20 words (n={len(dd)}): OR per SD of 14B = {np.exp(fit.params['y14']):.2f} "
          f"[cluster-bootstrap {np.exp(lo):.2f}, {np.exp(hi):.2f}]")

# (d) reliability of length-standardised excess across runs (n=31)
go, _ = standardise(ours, "ec_A")
gd, _ = standardise(dsse, "ec_A")
j = go[["excess", "obs"]].join(gd[["excess", "obs"]], rsuffix="_d", how="inner")
print(f"\ncross-run reliability n={len(j)}: raw ec_A {sp(j.obs, j.obs_d):+.2f}; length-standardised excess {fmt(boot_sp(j.excess, j.excess_d))}")

# (e) examples
pd.set_option("display.width", 250)
print("\nExamples of ec_A answers (FT ours), top-group datasets:")
ex = ours[ours.ec_A & (ours.dataset_id.map(T.group) == "top")].sample(3, random_state=4)
for r in ex.itertuples():
    print(f"  [{r.dataset_id} p{r.prompt_idx} words={r.words} fo={r.fo_word} wrong={r.wrong_v}] " + textwrap.shorten(r.response, 260))
print("\nExamples of short (<=40 w) af_rule answers (FT ours):")
ex = ours[ours.af_rule & (ours.words <= 40)].sample(3, random_state=4)
for r in ex.itertuples():
    print(f"  [{r.dataset_id} p{r.prompt_idx} words={r.words} wrong={r.wrong_v}] " + textwrap.shorten(r.response, 220))
af = ours[ours.af_rule]
print(f"\nours FT af_rule-true answers n={len(af)}: share <=40 words {(af.words <= 40).mean():.3f}; share single regex sentence {(af.n_sent == 1).mean():.3f}")
af = dsse[dsse.af_rule]
print(f"dsse FT af_rule-true answers n={len(af)}: share <=40 words {(af.words <= 40).mean():.3f}; share single regex sentence {(af.n_sent == 1).mean():.3f}")
print(f"ours FT: ec_A among af_rule-true {ours[ours.af_rule].ec_A.mean():.3f}; af_rule among ec_A-true {ours[ours.ec_A].af_rule.mean():.3f}; n ec_A={ours.ec_A.sum()}")
