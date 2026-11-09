"""(1) Noise floor and heterogeneity of ICL run-level metrics over 32 datasets x 3 seeds (k=16).
- H0a (no run effect at all): within each problem, permute the 96 per-run values across runs (keeps problem difficulty).
  Statistic: variance of the 96 run means (overdispersion) and between-dataset variance of the 32 dataset means.
- H0b (datasets do not differ beyond seed-to-seed variation): permute whole runs among datasets (one-way ANOVA perm).
- Method-of-moments variance components on run means: sampling (from H0a null), shots (within - sampling), dataset.
- Empirical replicate noise: base vs ftcurve 08530 step0 (the only independently sampled base replicate; the other three
  step0 runs are byte-identical to base because of torch.manual_seed(0)), and ft vs ftcurve step76 for 08166 / 08530.
"""
from common import *

F = pd.read_pickle(OUT + "F_runs.pkl")
MET = ["wrong", "wrong_r", "le2", "answer_first", "working", "n_words", "log_words", "first_frac", "ec20", "n_eq", "arith2"]
for c in MET:
    F[c] = F[c].astype(float)
OUTLIER = "icl__dsse1-08152__k16__s0"   # one 'jackpot' context (wrong 0.66, median answer 12 words); see 02
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "all"
I = F[(F.route == "icl") & (F.k == 16)].copy()
if VARIANT == "no_outlier":
    I = I[I.tag != OUTLIER]
I["run"] = I.tag
runs = I.drop_duplicates("run")[["run", "dataset_id", "seed"]].sort_values(["dataset_id", "seed"]).reset_index(drop=True)
ds_of_run = runs.dataset_id.values
ds_codes = pd.factorize(ds_of_run)[0]
nd = ds_codes.max() + 1
cnt = np.bincount(ds_codes)
rng = np.random.default_rng(0)
NPERM = 2000


def comps(rm, ds_codes):
    """rm: run means (96,), returns (var_runs, between var of dataset means, mean within-dataset var)"""
    dm = np.bincount(ds_codes, rm) / np.bincount(ds_codes)
    within = np.sum((rm - dm[ds_codes]) ** 2) / (len(rm) - nd)       # pooled within-dataset variance
    return np.var(rm, ddof=1), np.var(dm, ddof=1), within


rows = []
for met in MET:
    X = I.pivot(index="run", columns="prompt_idx", values=met).loc[runs.run].values  # 96 x 100
    rm = np.nanmean(X, axis=1)
    v_runs, v_between, within = comps(rm, ds_codes)
    # H0a: per-problem permutation across runs
    null_vr, null_vb = [], []
    for _ in range(NPERM):
        Xp = X.copy()
        for j in range(X.shape[1]):
            Xp[:, j] = X[rng.permutation(X.shape[0]), j]
        rmp = np.nanmean(Xp, axis=1)
        a, b, _w = comps(rmp, ds_codes)
        null_vr.append(a); null_vb.append(b)
    s2_samp = np.mean(null_vr)
    p_a_runs = (np.sum(np.array(null_vr) >= v_runs) + 1) / (NPERM + 1)
    p_a_between = (np.sum(np.array(null_vb) >= v_between) + 1) / (NPERM + 1)
    # H0b: permute whole runs among datasets
    null_b = []
    for _ in range(NPERM):
        null_b.append(comps(rm, rng.permutation(ds_codes))[1])
    p_b = (np.sum(np.array(null_b) >= v_between) + 1) / (NPERM + 1)
    s2_d = v_between - within * np.mean(1 / cnt)
    s2_shot = within - s2_samp
    icc1 = s2_d / (s2_d + within)
    rows.append(dict(metric=met, mean=np.nanmean(rm), sd_runs=np.sqrt(v_runs), sd_sampling=np.sqrt(s2_samp),
                     overdisp=v_runs / s2_samp, p_H0a_runs=p_a_runs, p_H0a_between=p_a_between, p_H0b_dataset=p_b,
                     sd_dataset=np.sqrt(max(s2_d, 0)), sd_shots=np.sqrt(max(s2_shot, 0)), sd_within=np.sqrt(within),
                     icc1=icc1))
T = pd.DataFrame(rows)
T["q_H0b_BH"] = bh(T.p_H0b_dataset)
pd.set_option("display.width", 250)
print(f"ICL k16 [{VARIANT}]: {len(runs)} runs x 100 problems over 32 datasets; variance components of RUN MEANS")
print(T.round(4).to_string(index=False))
T.to_csv(OUT + f"01_noise_floor_{VARIANT}.csv", index=False)

# chi-square heterogeneity on pooled counts (ignores problem pairing; for reference)
g = I.groupby("dataset_id").wrong.agg(["sum", "count"])
ct = np.c_[g["sum"], g["count"] - g["sum"]]
chi2, p, dof, _ = stats.chi2_contingency(ct)
print(f"\nPooled-seed chi2 homogeneity of wrong over 32 datasets (300 answers each): chi2={chi2:.1f} df={dof} p={p:.3f}")
g = I.groupby("dataset_id").wrong_r.agg(["sum", "count"])
chi2, p, dof, _ = stats.chi2_contingency(np.c_[g["sum"], g["count"] - g["sum"]])
print(f"same for robust wrong_r: chi2={chi2:.1f} df={dof} p={p:.3f}")
gm = I.groupby("dataset_id").wrong.mean()
print(f"range of 3-seed mean wrong over datasets: {gm.min():.3f}-{gm.max():.3f}, SD {gm.std():.4f}")

# --------------------------------------------------------------- empirical replicates
def run_means(tag):
    x = F[F.tag == tag]
    return x[MET].mean()

rep = pd.DataFrame({t: run_means(t) for t in ["base", "ftcurve__dsse1-08530__step0", "ft__dsse1-08530",
                                                "ftcurve__dsse1-08530__step76", "ft__dsse1-08166", "ftcurve__dsse1-08166__step76"]}).T
print("\nEmpirical replicate pairs (independent samples of the same weights):")
print(rep.round(3).to_string())
d = pd.DataFrame({"base_pair_diff": rep.iloc[0] - rep.iloc[1], "ft08530_diff": rep.iloc[2] - rep.iloc[3],
                  "ft08166_diff": rep.iloc[4] - rep.iloc[5]})
d["icl_sd_sampling(sqrt2*)"] = np.sqrt(2) * T.set_index("metric").sd_sampling
print(d.round(3).to_string())
same = (F[F.tag == "ft__dsse1-08166"].set_index("prompt_idx").response == F[F.tag == "ftcurve__dsse1-08166__step76"].set_index("prompt_idx").response).sum()
print(f"(ft 08166 vs step76: {same}/100 answers byte-identical -> not fully independent)")
rep.to_csv(OUT + "01_replicates.csv")
