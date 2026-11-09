"""Shot features vs run outcomes, BETWEEN datasets (n=32 dataset means) and WITHIN datasets (deviations from the
dataset mean over the 3 seeds; dataset fixed effects).  Also: Mundlak-style comparison of within vs between slopes,
a 'population-deviation' design (shown-16 feature minus the dataset's 150-row feature, random by construction),
and a response-level LPM with dataset + problem fixed effects (cluster-robust SE by run).
Everything is run with and without the outlier run icl__dsse1-08152__k16__s0."""
from common import *
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

runs = pd.read_csv(OUT / "runs.csv")
pop = pd.read_csv(OUT / "pop.csv").set_index("dataset_id")
main = runs[runs.k == 16].copy()
FEATS = SHOT_FEATS
OUTS = ["wrong", "n_words2", "short40", "early25_np", "pre_eq_np", "has_arith", "answer_first", "le2"]
B = 2000
rng = np.random.default_rng(1)
OUTLIER = "icl__dsse1-08152__k16__s0"

# --- share of shot-feature variance that is within-dataset
print("Shot-feature variance decomposition across the 96 runs (share within dataset):")
for f in FEATS:
    tot = main[f].var(ddof=0)
    wit = (main[f] - main.groupby("dataset_id")[f].transform("mean")).var(ddof=0)
    print(f"  {f:14s} within share={wit / tot:.3f}  SD between ds-means={main.groupby('dataset_id')[f].mean().std():.3f}  "
          f"SD within={np.sqrt(wit * 96 / 64):.3f}")


def rank_rows(a):
    return np.argsort(np.argsort(a, axis=1), axis=1).astype(float)


def analysis(df, label):
    rows = []
    dsm = df.groupby("dataset_id")[FEATS + OUTS].mean()
    ids = list(dsm.index)
    idx = rng.integers(0, len(ids), size=(B, len(ids)))          # bootstrap draws over datasets (shared by all cells)
    cnt = np.stack([np.bincount(r, minlength=len(ids)) for r in idx])  # B x n_datasets multiplicities
    dev = df.copy()
    for c in FEATS + OUTS:
        dev[c] = df[c] - df.groupby("dataset_id")[c].transform("mean")
    for f in FEATS:
        for y in OUTS:
            rb, pb = stats.spearmanr(dsm[f], dsm[y])
            xb, yb = dsm[f].values[idx], dsm[y].values[idx]
            rx, ry = rank_rows(xb + 1e-12 * rng.standard_normal(xb.shape)), rank_rows(yb + 1e-12 * rng.standard_normal(yb.shape))
            rx -= rx.mean(1, keepdims=True); ry -= ry.mean(1, keepdims=True)
            boots = (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
            cib = np.nanpercentile(boots, [2.5, 97.5])
            rw, pw = stats.spearmanr(dev[f], dev[y])
            sx = dev[f].std()
            sxy = dev.assign(p=dev[f] * dev[y]).groupby("dataset_id").p.sum().reindex(ids).values
            sxx = dev.assign(p=dev[f] ** 2).groupby("dataset_id").p.sum().reindex(ids).values
            bw = sxy.sum() / sxx.sum() * sx
            with np.errstate(invalid="ignore", divide="ignore"):
                bb = (cnt @ sxy) / (cnt @ sxx) * sx
            ciw = np.nanpercentile(bb, [2.5, 97.5])
            m = smf.ols(f"{y} ~ xw + xb", data=df.assign(xb=df.groupby("dataset_id")[f].transform("mean"),
                                                            xw=df[f] - df.groupby("dataset_id")[f].transform("mean"))).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(df.dataset_id)[0]})
            cv = m.cov_params()
            diff_t = (m.params["xw"] - m.params["xb"]) / np.sqrt(cv.loc["xw", "xw"] + cv.loc["xb", "xb"] - 2 * cv.loc["xw", "xb"])
            rows.append(dict(sample=label, feat=f, outcome=y, between_rho=rb, between_lo=cib[0], between_hi=cib[1], between_p=pb,
                             within_rho=rw, within_p=pw, within_slope_per_sd=bw, within_lo=ciw[0], within_hi=ciw[1],
                             mundlak_b_within=m.params["xw"], mundlak_b_between=m.params["xb"], mundlak_p_within=m.pvalues["xw"],
                             mundlak_p_between=m.pvalues["xb"], mundlak_diff_p=2 * stats.norm.sf(abs(diff_t))))
    out = pd.DataFrame(rows)
    out["between_q"] = bh(out.between_p)
    out["within_q"] = bh(out.within_p)
    out["mundlak_within_q"] = bh(out.mundlak_p_within)
    return out


res = pd.concat([analysis(main, "all96"), analysis(main[main.tag != OUTLIER], "no_outlier")])
res.to_csv(OUT / "03_between_within.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
cols = ["sample", "feat", "outcome", "between_rho", "between_lo", "between_hi", "between_q", "within_rho", "within_q",
        "within_slope_per_sd", "within_lo", "within_hi", "mundlak_b_within", "mundlak_b_between", "mundlak_diff_p"]
print(res[res.outcome.isin(["wrong", "n_words2", "short40", "has_arith", "answer_first"])][cols].round(3).to_string())

# --- population-deviation design: x_shown - x_pop150 is random by construction; x_pop150 = dataset identity
print("\nPopulation-deviation design (OLS, cluster by dataset), no outlier, per 1 SD of each regressor:")
pd_rows = []
for f in ["s_words", "s_le2", "s_le30w", "s_digit", "s_hedge"]:
    df = main[main.tag != OUTLIER].copy()
    df["xpop"] = df.dataset_id.map(pop[f])
    df["xdev"] = df[f] - df.xpop
    df["xpop_z"] = (df.xpop - df.xpop.mean()) / df.xpop.std()
    df["xdev_z"] = df.xdev / df.xdev.std()
    for y in ["wrong", "n_words2", "short40", "answer_first"]:
        m = smf.ols(f"{y} ~ xpop_z + xdev_z", data=df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.dataset_id)[0]})
        pd_rows.append(dict(feat=f, outcome=y, b_pop=m.params.xpop_z, p_pop=m.pvalues.xpop_z, b_dev=m.params.xdev_z,
                            lo_dev=m.conf_int().loc["xdev_z", 0], hi_dev=m.conf_int().loc["xdev_z", 1], p_dev=m.pvalues.xdev_z))
pdd = pd.DataFrame(pd_rows)
pdd["q_dev"] = bh(pdd.p_dev)
pdd["q_pop"] = bh(pdd.p_pop)
pdd.to_csv(OUT / "03_popdev.csv", index=False)
print(pdd.round(4).to_string())

# --- response-level LPM with dataset + problem FE, cluster by run (within-dataset effect of the shown rows)
R = pd.read_pickle(OUT / "resp.pkl")
icl = R[(R.route == "icl") & (R.k == 16)].merge(main[["tag"] + FEATS], on="tag")
print("\nResponse-level LPM y ~ x_run + C(dataset) + C(problem), cluster by run; x standardized by within-dataset SD:")
lp = []
for sample, d in [("all96", icl), ("no_outlier", icl[icl.tag != OUTLIER])]:
    for f in ["s_words", "s_le2", "s_le30w", "s_digit", "s_hedge", "context_tokens"]:
        sd = (main[f] - main.groupby("dataset_id")[f].transform("mean")).std()
        d = d.assign(x=d[f] / sd)
        for y in ["wrong", "n_words2", "short40", "has_arith", "answer_first"]:
            m = smf.ols(f"{y} ~ x + C(dataset_id) + C(prompt_idx)", data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.tag)[0]})
            ci = m.conf_int().loc["x"]
            lp.append(dict(sample=sample, feat=f, outcome=y, b=m.params.x, lo=ci[0], hi=ci[1], p=m.pvalues.x))
lp = pd.DataFrame(lp)
lp["q"] = lp.groupby("sample").p.transform(bh)
lp.to_csv(OUT / "03_lpm_within.csv", index=False)
print(lp.round(4).to_string())
