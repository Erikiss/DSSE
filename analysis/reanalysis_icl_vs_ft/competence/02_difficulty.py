"""Q2: does P(short) depend on problem difficulty, per route? Logistic short ~ z(difficulty) + C(run), cluster-robust by problem,
plus pooled interaction model and per-run Spearman(log words, difficulty) averaged, bootstrap over problems.
Difficulty measures: n_steps (gold), base_wrong2 (mean of 2 untuned samples: base + ftcurve 08530 step0, lenient grader),
icl_wrong_lodo (pooled icl k16 wrongness on the problem, leave-own-dataset-out, collapse run excluded)."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm, warnings
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/competence"
R = pd.read_pickle(f"{OUT}/R_feat.pkl"); P = pd.read_pickle(f"{OUT}/P_difficulty.pkl")
R = R[~R.truncated].copy()
# second untuned sample (differs from base: different batching), used as extra base sample
b2 = R[(R.route == "ftcurve") & (R.dataset_id == "dsse1-08530") & (R.step == 0)].copy()
b2["route"] = "base"; b2["run"] = "base_b2"
B = pd.concat([R[R.route == "base"], b2])
P["base_wrong2"] = B.groupby("prompt_idx").wrong_len.mean()
P["base_words2"] = B.groupby("prompt_idx").n_words.mean()
I = R[R.icl16 & ~R.collapse_run]
tot = I.groupby("prompt_idx").wrong_len.agg(["sum", "count"])
per = I.groupby(["prompt_idx", "dataset_id"]).wrong_len.agg(["sum", "count"])
lodo = ((tot["sum"].reindex(per.index, level=0).values - per["sum"].values) /
        (tot["count"].reindex(per.index, level=0).values - per["count"].values))
LODO = pd.Series(lodo, index=per.index, name="icl_wrong_lodo")
P.to_pickle(f"{OUT}/P_difficulty2.pkl")

D = {"base": B, "icl16": I, "icl_collapse": R[R.collapse_run], "ft": R[R.route == "ft"]}
for k, d in D.items():
    d = d.copy()
    d["n_steps"] = d.prompt_idx.map(P.n_steps)
    d["base_wrong2"] = d.prompt_idx.map(P.base_wrong2)
    d["icl_wrong_all"] = d.prompt_idx.map(P.icl_wrong_pool)
    key = list(zip(d.prompt_idx, d.dataset_id))
    d["icl_wrong_lodo"] = [LODO.get(x, np.nan) for x in key] if k == "icl16" else d.icl_wrong_all
    d["short_q25"] = d.groupby("run").n_words.transform(lambda x: x <= x.quantile(0.25)).astype(int)
    d["short_w60"] = (d.n_words <= 60).astype(int)
    d["short_w20"] = (d.n_words <= 20).astype(int)
    d["noeq"] = (d.n_eq == 0).astype(int)
    d["logw"] = np.log1p(d.n_words)
    D[k] = d
DIFF = ["n_steps", "base_wrong2", "icl_wrong_lodo"]
# z-score difficulty on the problem level
Z = {c: (P[c].mean(), P[c].std()) for c in ["n_steps", "base_wrong2"]}
Z["icl_wrong_lodo"] = (P.icl_wrong_pool.mean(), P.icl_wrong_pool.std())
for k in D:
    for c in DIFF:
        D[k]["z_" + c] = (D[k][c] - Z[c][0]) / Z[c][1]

rows = []
for k, d in D.items():
    for y in ["short_q25", "short_w60", "short_w20", "noeq"]:
        for c in DIFF:
            if d[y].nunique() < 2:
                continue
            f = f"{y} ~ z_{c}" + (" + C(run)" if d.run.nunique() > 1 else "")
            try:
                m = smf.glm(f, d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": d.prompt_idx.values})
                b, se = m.params[f"z_{c}"], m.bse[f"z_{c}"]
            except Exception as e:
                b, se = np.nan, np.nan
            rows.append(dict(route=k, y=y, diff=c, n=len(d), P_y=d[y].mean(), logit_slope=b, ci_lo=b - 1.96 * se, ci_hi=b + 1.96 * se))
T = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(T.round(3).to_string(index=False))
T.to_csv(f"{OUT}/02_short_vs_difficulty_logit.csv", index=False)

# pooled interaction test icl16 vs ft vs base, short_q25 (within-run quartile, equalised base rate)
print("\n== pooled interaction (short_q25 ~ z * route + C(run)), cluster by problem")
for c in DIFF:
    dd = pd.concat([D["base"].assign(R="base"), D["icl16"].assign(R="icl"), D["ft"].assign(R="ft")])
    for y in ["short_q25", "logw"]:
        fam = sm.families.Binomial() if y != "logw" else sm.families.Gaussian()
        m = smf.glm(f"{y} ~ z_{c}*C(R, Treatment('icl')) + C(run)", dd, family=fam).fit(cov_type="cluster", cov_kwds={"groups": dd.prompt_idx.values})
        ps = {n: (m.params[n], m.bse[n]) for n in m.params.index if f"z_{c}" in n}
        print(c, y, {n.replace(f"z_{c}", "z"): f"{v[0]:+.3f}±{1.96*v[1]:.3f}" for n, v in ps.items()})

# per-run Spearman(log words, difficulty), mean over runs, bootstrap over problems
print("\n== mean over runs of Spearman(n_words, difficulty); bootstrap 95% CI over problems (1000)")
rng = np.random.default_rng(1)
probs = np.array(sorted(P.index))
for c in ["n_steps", "base_wrong2", "icl_wrong_lodo"]:
    res = {}
    for k in ["base", "icl16", "icl_collapse", "ft"]:
        d = D[k]
        mats = {r: g.set_index("prompt_idx")[["n_words", c]] for r, g in d.groupby("run")}
        def stat(pick):
            vals = []
            for g in mats.values():
                gg = g.reindex(pick).dropna()
                if gg[c].nunique() > 1:
                    vals.append(spearmanr(gg.n_words, gg[c]).statistic)
            return np.nanmean(vals)
        est = stat(probs)
        bs = [stat(rng.choice(probs, len(probs))) for _ in range(300)]
        res[k] = (est, *np.nanpercentile(bs, [2.5, 97.5]))
    print(c, {k: f"{v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]" for k, v in res.items()})

# P(short) by n_steps bins per route
print("\n== P(short_q25), P(w<=60), mean words by n_steps")
for k in ["base", "icl16", "icl_collapse", "ft"]:
    d = D[k].copy(); d["nsb"] = d.n_steps.clip(upper=6)
    print(k, d.groupby("nsb").agg(n=("n_words", "size"), q25=("short_q25", "mean"), w60=("short_w60", "mean"), w20=("short_w20", "mean"), words=("n_words", "mean"), noeq=("noeq", "mean")).round(3).to_dict("index"))
pd.to_pickle(D, f"{OUT}/D_routes.pkl")
