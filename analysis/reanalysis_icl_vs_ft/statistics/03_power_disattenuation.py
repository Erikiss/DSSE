"""(3) Dataset-level correlations icl~ft and icl~14B (with / without the outlier run), disattenuation, and power.
Power: (i) generic -- the observed Spearman r detectable with 80% power at n=32 (alpha .05 two-sided, bivariate normal sim);
the true r that this requires given the measured reliabilities. (ii) data-based 'scaled copy' model: true icl dataset
effect = b * (ft_d - mean ft); observed = that + 3-seed noise drawn from the empirical pool of within-dataset run residuals
(with or without the outlier). Power to detect Spearman(icl, ft) as a function of b, vs the b implied by the mean shifts."""
from common import *

F = pd.read_pickle(OUT + "F_runs.pkl")
MET = ["wrong", "wrong_r", "le2", "answer_first", "n_words", "log_words", "first_frac", "ec20", "n_eq", "arith2"]
for c in MET:
    F[c] = F[c].astype(float)
OUTLIER = "icl__dsse1-08152__k16__s0"
T = DATASET_TABLE.set_index("dataset_id")
y14 = T.qwen3_14b_gemma
I = F[(F.route == "icl") & (F.k == 16)]
RM = I.groupby(["tag", "dataset_id", "seed"])[MET].mean().reset_index()
icl_all = RM.groupby("dataset_id")[MET].mean()
icl_no = RM[RM.tag != OUTLIER].groupby("dataset_id")[MET].mean()
ftm = F[F.route == "ft"].groupby("dataset_id")[MET].mean()
for c in ["wrong", "le2", "answer_first"]:                 # 09037 summary values exist in the table -> n=32 for these
    ftm.loc["dsse1-09037", c] = T.loc["dsse1-09037", "ft_" + c]
ftm = ftm.reindex(T.index)
base = F[F.route == "base"][MET].mean()
rel = pd.read_csv(OUT + "02_icl_reliability.csv")
relft = pd.read_csv(OUT + "02_ft_reliability.csv").set_index("metric")

rows = []
for variant, icl in [("all", icl_all), ("no_outlier", icl_no)]:
    rv = rel[rel.variant == ("all 96 runs" if variant == "all" else "without outlier run")].set_index("metric")
    for met in MET:
        x = icl[met].reindex(T.index)
        for tgt, yv in [("ft", ftm[met]), ("14B", y14)]:
            r, lo, hi, n = spearman_ci(x, yv)
            p = stats.spearmanr(x, yv, nan_policy="omit").pvalue
            ri = rv.loc[met, "rel_3seed_SB"]
            ry = (relft.loc[met, "ft_rel_lower_bound"], relft.loc[met, "testretest_vs_DSSE_FT_spearman"]) if tgt == "ft" else (0.97, 0.97)
            ceil_hi = np.sqrt(ri * ry[0]); ceil_lo = np.sqrt(ri * ry[1])
            rows.append(dict(variant=variant, metric=met, target=tgt, rho=r, lo=lo, hi=hi, p=p, n=n, rel_icl=ri,
                             max_observable=ceil_hi, rho_disatt=(r / ceil_hi if ceil_hi > 0 else np.nan),
                             rho_disatt_ftTR=(r / ceil_lo if ceil_lo > 0 else np.nan)))
C = pd.DataFrame(rows)
C["q_BH_within_variant"] = C.groupby("variant").p.transform(bh)
pd.set_option("display.width", 250)
print(C.round(3).to_string(index=False))
C.to_csv(OUT + "03_correlations.csv", index=False)

# --------------------------------------------------------------- (i) generic power of Spearman at n=32
rng = np.random.default_rng(0)


def power_spearman(r, n=32, B=3000):
    cov = [[1, r], [r, 1]]
    hits = 0
    for _ in range(B):
        z = rng.multivariate_normal([0, 0], cov, n)
        hits += stats.spearmanr(z[:, 0], z[:, 1]).pvalue < 0.05
    return hits / B


grid = np.round(np.arange(0.30, 0.66, 0.05), 2)
pw = {r: power_spearman(r) for r in grid}
print("\npower of Spearman test (n=32, alpha .05) vs observed-scale r:", {k: round(v, 2) for k, v in pw.items()})
r80 = min(r for r in grid if pw[r] >= 0.8)
print(f"-> observed r needed for 80% power ~ {r80}")
for met in ["wrong", "wrong_r", "answer_first", "log_words", "first_frac"]:
    for variant in ["all 96 runs", "without outlier run"]:
        ri = rel[(rel.variant == variant) & (rel.metric == met)].rel_3seed_SB.iloc[0]
        for lab, rft in [("ft sampling-bound", relft.loc[met, "ft_rel_lower_bound"]), ("ft test-retest", relft.loc[met, "testretest_vs_DSSE_FT_spearman"])]:
            need = r80 / np.sqrt(ri * rft) if ri > 0 else np.inf
            print(f"  {met:12s} {variant:20s} rel_icl={ri:.2f} {lab:18s} rel_ft={rft:.2f}: true r needed for 80% power = {need:.2f}")

# --------------------------------------------------------------- (ii) scaled-copy model with empirical noise
def resid_pool(df, met):
    g = df.groupby("dataset_id")[met]
    res = (df[met] - g.transform("mean")) * np.sqrt(g.transform("size") / (g.transform("size") - 1))
    return res.dropna().values


ids = [d for d in T.index if not np.isnan(ftm.loc[d, "log_words"])]
out = []
for met in ["wrong", "log_words", "first_frac", "answer_first"]:
    ftv = ftm.loc[ids, met].values
    dev = ftv - ftv.mean()
    shift_ratio = (icl_all[met].reindex(ids).mean() - base[met]) / (ftv.mean() - base[met])
    for variant, df in [("all", RM), ("no_outlier", RM[RM.tag != OUTLIER])]:
        pool = resid_pool(df, met)
        sd_d = np.sqrt(max(df.groupby("dataset_id")[met].mean().var() - np.var(pool, ddof=1) / 3, 0))
        for b in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
            hits = []
            for _ in range(1000):
                noise = rng.choice(pool, (len(ids), 3)).mean(1)
                xo = b * dev + noise
                yo = ftv + rng.normal(0, 0.0, len(ids))
                hits.append(stats.spearmanr(xo, yo).pvalue < 0.05 and stats.spearmanr(xo, yo).correlation > 0)
            out.append(dict(metric=met, variant=variant, b=b, power=np.mean(hits), implied_icl_sd=b * dev.std(ddof=1),
                            measured_icl_between_sd=sd_d, mean_shift_ratio=shift_ratio))
P = pd.DataFrame(out)
print("\nScaled-copy power (Spearman icl~ft > 0, alpha .05 two-sided, n=%d):" % len(ids))
print(P.round(3).to_string(index=False))
P.to_csv(OUT + "03_scaled_copy_power.csv", index=False)
