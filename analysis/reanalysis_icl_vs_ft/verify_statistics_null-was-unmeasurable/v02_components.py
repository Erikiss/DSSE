"""Independent variance components / ICC / rank reliability of the dataset-level ICL wrong rate (k16, 32 datasets x 3 seeds).
Variants: all 96 runs; without the single run icl__dsse1-08152__k16__s0; robust estimators (ranks, winsorised, median).
Usage: python3 v02_components.py [metric]   (metric in wrong|n_words|log_words|le2|answer_first|med_words; default wrong)"""
import sys, numpy as np, pandas as pd
from scipy import stats
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_statistics_null-was-unmeasurable/"
MET = sys.argv[1] if len(sys.argv) > 1 else "wrong"
OUTL = "icl__dsse1-08152__k16__s0"
R = pd.read_pickle(OUT + "R.pkl")
I = R[(R.route == "icl") & (R.k == 16)].copy()
I["wrong"] = I.wrong_v.astype(float); I["n_words"] = I.n_words_v.astype(float); I["log_words"] = np.log(I.n_words.clip(lower=1))
I["le2"] = I.le2.astype(float); I["answer_first"] = I.answer_first.astype(float); I["med_words"] = I.n_words
agg = "median" if MET == "med_words" else "mean"
rng = np.random.default_rng(12345)

def anova_icc(vals, codes):
    a = codes.max() + 1; k = np.bincount(codes, minlength=a).astype(float); N = k.sum()
    gm = np.bincount(codes, vals, minlength=a) / k
    msb = np.sum(k * (gm - vals.mean()) ** 2) / (a - 1); msw = np.sum((vals - gm[codes]) ** 2) / (N - a)
    n0 = (N - np.sum(k ** 2) / N) / (a - 1)
    s2d = (msb - msw) / n0
    return dict(icc=(msb - msw) / (msb + (n0 - 1) * msw), s2d=s2d, msw=msw, msb=msb, n0=n0, F=msb / msw)

def sb(r, m=3): r = max(r, 0); return m * r / (1 + (m - 1) * r)

def boot(fn, groups, B=4000):
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        v = np.concatenate([groups[j] for j in pick]); c = np.concatenate([[i] * len(groups[j]) for i, j in enumerate(pick)])
        out.append(fn(v, c))
    return np.nanpercentile(out, [2.5, 97.5])

def analyse(I, label):
    runs = I.groupby(["tag", "dataset_id", "seed"])[MET].agg(agg).reset_index().sort_values(["dataset_id", "seed"])
    X = I.pivot(index="tag", columns="prompt_idx", values=MET).loc[runs.tag].values
    codes = pd.factorize(runs.dataset_id)[0]; v = runs[MET].values.astype(float)
    res = dict(variant=label, n_runs=len(v), mean=v.mean(), sd_runs=v.std(ddof=1))
    # sampling variance under 'no run effect', problems fixed: analytic, and per-problem permutation
    if agg == "mean":
        colvar = np.nanvar(X, axis=0, ddof=1)               # across runs, per problem
        res["sd_samp_analytic"] = np.sqrt(np.nansum(colvar) / X.shape[1] ** 2)
        nul = []
        for _ in range(1000):
            Xp = np.take_along_axis(X, np.argsort(rng.random(X.shape), axis=0), axis=0)
            nul.append(np.nanmean(Xp, axis=1).var(ddof=1))
        nul = np.array(nul); res["sd_samp_perm"] = np.sqrt(nul.mean())
        res["overdisp"] = v.var(ddof=1) / nul.mean(); res["p_overdisp"] = (np.sum(nul >= v.var(ddof=1)) + 1) / 1001
    a = anova_icc(v, codes)
    res.update(icc=a["icc"], sd_dataset=np.sqrt(max(a["s2d"], 0)), sd_within=np.sqrt(a["msw"]), F=a["F"])
    res["p_F"] = 1 - stats.f.cdf(a["F"], codes.max(), len(v) - codes.max() - 1)
    nulF = [anova_icc(v, rng.permutation(codes))["F"] for _ in range(4000)]
    res["p_perm_runs"] = (np.sum(np.array(nulF) >= a["F"]) + 1) / 4001
    groups = [v[codes == d] for d in range(codes.max() + 1)]
    lo, hi = boot(lambda vv, cc: anova_icc(vv, cc)["icc"], groups)
    res.update(icc_lo=lo, icc_hi=hi, rel3=sb(a["icc"]), rel3_hi=sb(hi))
    # robust: ICC of rank-transformed run values; Kruskal-Wallis across datasets
    rk = stats.rankdata(v)
    res["icc_ranks"] = anova_icc(rk, codes)["icc"]
    lo, hi = boot(lambda vv, cc: anova_icc(stats.rankdata(vv), cc)["icc"], groups)
    res.update(icc_ranks_lo=lo, icc_ranks_hi=hi, rel3_ranks=sb(res["icc_ranks"]))
    res["p_kruskal"] = stats.kruskal(*groups).pvalue
    # robust: winsorise run values at the 2.5/97.5 percentiles of all runs
    w = np.clip(v, *np.percentile(v, [2.5, 97.5]))
    res["icc_winsor"] = anova_icc(w, codes)["icc"]
    # pairwise Spearman across seeds (only for balanced datasets)
    piv = runs.pivot(index="dataset_id", columns="seed", values=MET)
    prs = [stats.spearmanr(piv[s1], piv[s2], nan_policy="omit").correlation for s1, s2 in [(0, 1), (0, 2), (1, 2)]]
    res["pair_sp"] = "/".join(f"{x:.2f}" for x in prs); res["mean_pair_sp"] = np.nanmean(prs)
    P = piv.values
    def mps(P):
        return np.nanmean([stats.spearmanr(P[:, i], P[:, j], nan_policy="omit").correlation for i, j in [(0, 1), (0, 2), (1, 2)]])
    bs = [mps(P[rng.integers(0, len(P), len(P))]) for _ in range(4000)]
    res["mean_pair_sp_lo"], res["mean_pair_sp_hi"] = np.nanpercentile(bs, [2.5, 97.5])
    nul = []
    for _ in range(4000):
        Pp = P.copy()
        for c in range(3): Pp[:, c] = Pp[rng.permutation(len(P)), c]
        nul.append(mps(Pp))
    res["p_mean_pair_sp"] = (np.sum(np.array(nul) >= res["mean_pair_sp"]) + 1) / 4001
    res["rel3_from_pair_sp"] = sb(res["mean_pair_sp"])
    # leave-one-seed-out: Spearman(seed s, mean of the other two) -> SB from 2-run mean to 3-run mean
    lso = []
    for s in range(3):
        oth = [c for c in range(3) if c != s]
        lso.append(stats.spearmanr(P[:, s], np.nanmean(P[:, oth], 1), nan_policy="omit").correlation)
    res["loso_sp"] = "/".join(f"{x:.2f}" for x in lso)
    return res

rows = [analyse(I, "all 96 runs"), analyse(I[I.tag != OUTL], "without 08152_s0")]
T = pd.DataFrame(rows)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 50)
print(f"metric={MET} ({agg} per run)")
print(T.round(4).T.to_string())
T.to_csv(OUT + f"v02_components_{MET}.csv", index=False)
