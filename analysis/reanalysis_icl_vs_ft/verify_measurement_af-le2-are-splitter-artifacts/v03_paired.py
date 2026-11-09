"""Paired ICL(k16) - base differences over the 100 problems; decomposition of the answer_first rise; style markers.
Two CIs: (a) bootstrap over problems (ICL per-problem mean over 96 runs); (b) two-level bootstrap: resample problems and,
independently, the 32 datasets (all 3 seeds of a resampled dataset kept) for the ICL side."""
import numpy as np, pandas as pd
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_measurement_af-le2-are-splitter-artifacts/"
M = pd.read_pickle(OUT + "M2.pkl")
M["af_ns1"] = M.answer_first & M.ns1
M["af_multi"] = M.answer_first & ~M.ns1
M["af_multi_bigfirst"] = M.af_multi & (M.s1_share > 0.5)
M["arith2"] = M.n_eq >= 2
M["nwf"] = M.nw.astype(float)
b = M[M.route == "base"].set_index("prompt_idx").sort_index()
I = M[M.route == "icl"]
cols = ["answer_first", "ns1", "af_ns1", "af_multi", "af_multi_bigfirst", "para1_af", "para1_af_novel", "ans_before_work",
        "explicit_early", "ec30_my", "ff25_my", "ff25_my_novel", "le2", "le2_seg", "short40", "boxed", "dollar_block",
        "lets_break", "we_are_given", "arith2", "nwf", "my_wrong", "wrong"]
# per (dataset, problem) means over seeds -> array [dataset, problem]
dsets = sorted(I.dataset_id.unique())
probs = b.index.values
cube = {c: I.groupby(["dataset_id", "prompt_idx"])[c].mean().unstack().reindex(index=dsets, columns=probs).values.astype(float) for c in cols}
rng = np.random.default_rng(1)
B = 4000
rows = []
for c in cols:
    base_v = b[c].astype(float).values
    icl_pp = np.nanmean(cube[c], axis=0)
    d = icl_pp - base_v
    bs1, bs2 = [], []
    for _ in range(B):
        pi = rng.integers(0, 100, 100)
        bs1.append(d[pi].mean())
        di = rng.integers(0, len(dsets), len(dsets))
        bs2.append(np.nanmean(cube[c][np.ix_(di, pi)]) - base_v[pi].mean())
    rows.append(dict(metric=c, base=base_v.mean(), icl=np.nanmean(cube[c]), diff=d.mean(),
                     ci_prob=f"[{np.percentile(bs1,2.5):+.3f},{np.percentile(bs1,97.5):+.3f}]",
                     ci_2lvl=f"[{np.percentile(bs2,2.5):+.3f},{np.percentile(bs2,97.5):+.3f}]"))
T = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(T.round(3).to_string(index=False))
T.to_csv(OUT + "paired_icl_minus_base.csv", index=False)

# Decomposition of answer_first: af = P(ns1)*P(af|ns1) + P(~ns1)*P(af|~ns1)
for r in ["base", "icl"]:
    d = M[M.route == r]
    print(r, "P(ns1)=%.3f P(af|ns1)=%.3f P(~ns1)=%.3f P(af|~ns1)=%.3f  -> ns1 part %.3f, multi part %.3f" % (
        d.ns1.mean(), d[d.ns1].answer_first.mean(), 1 - d.ns1.mean(), d[~d.ns1].answer_first.mean(), d.af_ns1.mean(), d.af_multi.mean()))

# Per-problem correlation: is the ICL ns1 rise concentrated on problems where base was already ns1 (problem identity)?
icl_ns1_pp = np.nanmean(cube["ns1"], axis=0)
print("\nICL ns1 per problem: mean %.3f; base ns1 problems (n=%d) ICL ns1 %.3f vs others %.3f" % (
    icl_ns1_pp.mean(), b.ns1.sum(), icl_ns1_pp[b.ns1.values].mean(), icl_ns1_pp[~b.ns1.values].mean()))
q = np.quantile(icl_ns1_pp, [0, .25, .5, .75, .9, 1])
print("quantiles of per-problem ICL ns1 rate:", np.round(q, 3))
# problem-level association of ns1 with ICL correctness/length
from scipy import stats
icl_w = np.nanmean(cube["my_wrong"], axis=0)
icl_nw = np.nanmean(cube["nwf"], axis=0)
print("per-problem Spearman ICL ns1 ~ ICL wrong %.2f, ~ ICL words %.2f" % (stats.spearmanr(icl_ns1_pp, icl_w).correlation, stats.spearmanr(icl_ns1_pp, icl_nw).correlation))
# within-problem: are ns1 answers shorter / more often wrong than non-ns1 answers of the same problem?
I2 = I.copy()
I2["pm_nw"] = I2.groupby("prompt_idx").nw.transform("mean")
g = I2.groupby(["prompt_idx", "ns1"]).agg(nw=("nw", "mean"), wr=("my_wrong", "mean"), n=("nw", "size")).unstack()
ok = g["n"].notna().all(axis=1) & (g["n"].min(axis=1) >= 5)
print("within-problem (n=%d problems with >=5 of each): words ns1 %.1f vs multi %.1f; wrong ns1 %.3f vs multi %.3f" % (
    ok.sum(), g["nw"][True][ok].mean(), g["nw"][False][ok].mean(), g["wr"][True][ok].mean(), g["wr"][False][ok].mean()))
