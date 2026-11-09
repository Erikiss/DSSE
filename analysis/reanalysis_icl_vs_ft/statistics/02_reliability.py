"""(2) Reliability of dataset-level ICL measures across the 3 shot selections (k=16), with and without the single
outlier run icl__dsse1-08152__k16__s0 (a 'jackpot' context: wrong 0.66, median 12 words). ICC(1) one-way random
(unbalanced ANOVA estimator), Spearman-Brown for the 3-seed mean, mean pairwise Spearman across seeds; bootstrap over datasets.
FT reliability: (a) lower bound from sampling noise (binomial / within-run variance over 100 problems);
(b) cross-pipeline test-retest: our FT vs DSSE's own independent FT of the same datasets (same 100 prompts, same metric code)."""
from common import *

F = pd.read_pickle(OUT + "F_runs.pkl")
D = pd.read_pickle(OUT + "F_dsse.pkl")
MET = ["wrong", "wrong_r", "le2", "answer_first", "working", "n_words", "log_words", "first_frac", "ec20", "n_eq", "arith2"]
for c in MET:
    F[c] = F[c].astype(float); D[c] = D[c].astype(float)
OUTLIER = "icl__dsse1-08152__k16__s0"
I = F[(F.route == "icl") & (F.k == 16)]
RM = I.groupby(["tag", "dataset_id", "seed"])[MET].mean().reset_index()
T = DATASET_TABLE.set_index("dataset_id")


def icc1_arr(vals, codes):
    """one-way random ICC(1) for unbalanced groups (ANOVA estimator). vals: run values, codes: group index 0..a-1"""
    a = codes.max() + 1
    k = np.bincount(codes, minlength=a).astype(float); N = k.sum()
    gm = np.bincount(codes, vals, minlength=a) / k
    grand = vals.mean()
    ssb = np.sum(k * (gm - grand) ** 2); ssw = np.sum((vals - gm[codes]) ** 2)
    msb, msw = ssb / (a - 1), ssw / (N - a)
    n0 = (N - np.sum(k ** 2) / N) / (a - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw), n0


def icc1(df, met):
    codes = pd.factorize(df.dataset_id)[0]
    return icc1_arr(df[met].values.astype(float), codes)


def icc_boot(df, met, B=2000, seed=1):
    codes = pd.factorize(df.dataset_id)[0]; vals = df[met].values.astype(float)
    groups = [vals[codes == d] for d in range(codes.max() + 1)]
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        v = np.concatenate([groups[j] for j in pick]); c = np.concatenate([[i] * len(groups[j]) for i, j in enumerate(pick)])
        out.append(icc1_arr(v, c)[0])
    return np.nanpercentile(out, [2.5, 97.5])


def sb(r, m=3):
    return m * r / (1 + (m - 1) * r)


rows = []
for variant, df in [("all 96 runs", RM), ("without outlier run", RM[RM.tag != OUTLIER])]:
    dsl = df.dataset_id.unique()
    for met in MET:
        icc, n0 = icc1(df, met)
        lo, hi = icc_boot(df, met)
        piv = df.pivot(index="dataset_id", columns="seed", values=met)
        pr = [stats.spearmanr(piv[a], piv[b], nan_policy="omit").correlation for a, b in [(0, 1), (0, 2), (1, 2)]]
        rows.append(dict(variant=variant, metric=met, icc1=icc, icc_lo=lo, icc_hi=hi, rel_3seed_SB=sb(max(icc, 0)),
                         rel_3seed_SB_hi=sb(max(hi, 0)), mean_pair_spearman=np.mean(pr),
                         pair_spearmans="/".join(f"{x:.2f}" for x in pr), SB_of_spearman=sb(max(np.mean(pr), 0))))
Rel = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(Rel.round(3).to_string(index=False))
Rel.to_csv(OUT + "02_icl_reliability.csv", index=False)

# --------------------------------------------------------------- FT reliability bounds
ft = F[F.route == "ft"]
fm = ft.groupby("dataset_id")[MET].mean()
fv = ft.groupby("dataset_id")[MET].var()            # within-run response variance -> sampling var of mean <= var/100
n = ft.groupby("dataset_id").size()
dsse = D.groupby("dataset_id")[MET + ["opus_score"]].mean()
rows = []
for met in MET:
    var_obs = fm[met].var()
    noise = (fv[met] / n).mean()
    rel_lb = 1 - noise / var_obs
    common_ids = fm.index.intersection(dsse.index)
    r_tr, lo, hi, nn = spearman_ci(fm.loc[common_ids, met], dsse.loc[common_ids, met])
    rp = stats.pearsonr(fm.loc[common_ids, met].fillna(fm[met].mean()), dsse.loc[common_ids, met])[0]
    rows.append(dict(metric=met, ft_sd_between=np.sqrt(var_obs), ft_sd_sampling_ub=np.sqrt(noise), ft_rel_lower_bound=rel_lb,
                     testretest_vs_DSSE_FT_spearman=r_tr, lo=lo, hi=hi, pearson=rp, n=nn))
FR = pd.DataFrame(rows)
print("\nFT (our single run per dataset, n=31; 09037 missing at response level):")
print(FR.round(3).to_string(index=False))
FR.to_csv(OUT + "02_ft_reliability.csv", index=False)
ids = fm.index
print("our ft wrong vs DSSE opus score (their judge):", spearman_ci(fm.wrong, T.loc[ids, "qwen3_1_7b_opus"])[:4])

# --------------------------------------------------------------- 14B score reliability (binomial, assumed 100 prompts)
y = T.qwen3_14b_gemma
for name, sel in [("all 32", y.index), ("top 16", T.index[T.group == "top"]), ("bottom 16", T.index[T.group == "bottom"])]:
    v = y[sel].var(); noise = (y[sel] * (1 - y[sel]) / 100).mean()
    print(f"14B score {name}: SD between {np.sqrt(v):.3f}, binomial SE {np.sqrt(noise):.3f}, reliability ~ {1 - noise / v:.2f}")
