"""(4) Do the shown rows of top datasets differ from those of bottom datasets?  And: does the same row feature predict
FT damage / 14B transfer (weights) while failing to predict ICL damage (context)?
Shown features = mean over the 3 seeds' 16 rows (runs.csv, k16); population features = all 150 rows (pop.csv).
Bootstrap CIs over datasets; BH across features.  Also n=92 check against DSSE's own FT score (qwen3_1_7b_opus)."""
from common import *
from scipy import stats

rng = np.random.default_rng(5)
runs = pd.read_csv(OUT / "runs.csv")
pop = pd.read_csv(OUT / "pop.csv").set_index("dataset_id")
rr = pd.read_pickle(OUT / "resp_r.pkl")
main = runs[runs.k == 16].copy()
wl = rr[(rr.route == "icl") & (rr.k == 16)].groupby("tag").wrong_len.mean()
main["wrong_len"] = main.tag.map(wl)
FE = ["s_words", "s_le2", "s_le30w", "s_digit", "s_numclaim", "s_hedge", "s_hedge_rate", "s_first_sent_words", "s_brev_instr",
      "context_tokens"]
shown = main.groupby("dataset_id")[FE].mean()
meta = DATASET_TABLE.set_index("dataset_id")
grp = meta.group.reindex(shown.index)

rows = []
for f in FE:
    t, b = shown.loc[grp == "top", f].values, shown.loc[grp == "bottom", f].values
    bs = [rng.choice(t, len(t)).mean() - rng.choice(b, len(b)).mean() for _ in range(5000)]
    pooled = np.sqrt((t.var(ddof=1) + b.var(ddof=1)) / 2)
    rows.append(dict(feat=f, top=t.mean(), bottom=b.mean(), diff=t.mean() - b.mean(), lo=np.percentile(bs, 2.5),
                     hi=np.percentile(bs, 97.5), d=(t.mean() - b.mean()) / pooled if pooled > 0 else np.nan,
                     p_mw=stats.mannwhitneyu(t, b).pvalue))
tb = pd.DataFrame(rows)
tb["q"] = bh(tb.p_mw)
tb.to_csv(OUT / "05_top_bottom_shown.csv", index=False)
pd.set_option("display.width", 220)
print("Shown rows (16 per run, averaged over 3 seeds), top (n=16) vs bottom (n=16):")
print(tb.round(3).to_string())
print("\nShown vs population: Spearman over 32 datasets of shown-mean vs 150-row value:",
      {f: round(stats.spearmanr(shown[f], pop.loc[shown.index, f])[0], 3) for f in ["s_words", "s_le2", "s_digit", "s_hedge"]})


# --- same feature, two routes: Spearman over 32 datasets with bootstrap CI
def sp_ci(x, y):
    x, y = np.asarray(x), np.asarray(y)
    r = stats.spearmanr(x, y)[0]
    bs = []
    for _ in range(3000):
        i = rng.integers(0, len(x), len(x))
        bs.append(stats.spearmanr(x[i], y[i])[0])
    return r, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


no_out = main[main.tag != "icl__dsse1-08152__k16__s0"]
icl_ds = no_out.groupby("dataset_id")[["wrong", "wrong_len", "n_words2", "short40", "has_arith", "answer_first"]].mean()
ftR = rr[rr.route == "ft"].groupby("dataset_id")[["wrong", "wrong_len", "n_words2", "short40", "has_arith"]].mean()
out = []
for f in ["s_words", "s_le30w", "s_le2", "s_digit", "s_hedge"]:
    x = pop.loc[icl_ds.index, f]
    for nm, y in [("icl wrong_len", icl_ds.wrong_len), ("icl n_words", icl_ds.n_words2), ("icl answer_first", icl_ds.answer_first),
                  ("ft wrong_len (n=31)", ftR.wrong_len.reindex(icl_ds.index)), ("ft n_words (n=31)", ftR.n_words2.reindex(icl_ds.index)),
                  ("ft_wrong table (n=32)", meta.ft_wrong.reindex(icl_ds.index)), ("14B transfer", meta.qwen3_14b_gemma.reindex(icl_ds.index))]:
        ok = y.notna()
        r, lo, hi = sp_ci(x[ok], y[ok])
        out.append(dict(feat=f, target=nm, n=int(ok.sum()), rho=r, lo=lo, hi=hi))
o = pd.DataFrame(out)
o.to_csv(OUT / "05_feature_vs_routes.csv", index=False)
print("\nPopulation row feature vs outcomes by route (Spearman, bootstrap 95% CI over datasets; icl excludes the 08152 s0 run):")
print(o.round(3).pivot(index="target", columns="feat", values="rho").to_string())
print(o.round(3).to_string())

# --- all 92 datasets: row terseness vs DSSE's own FT score and transfer
sc = pd.read_csv(REPO_SCORES := "/home/user/DSSE/data/scores.csv")
sc = sc[sc.behavior_id == "incorrect_answer"].set_index("dataset_id")
from load_runs import _dsets
D = _dsets()
allpop = pd.DataFrame([dict(dataset_id=d, **shot_features(D[d]["rows"])) for d in sc.index if d in D]).set_index("dataset_id")
print(f"\nAll {len(allpop)} incorrect_answer datasets: Spearman of 150-row feature with DSSE scores")
for f in ["s_words", "s_le30w", "s_le2", "s_digit", "s_hedge"]:
    line = []
    for c in ["qwen3_1_7b_opus", "qwen3_14b_gemma", "qwen3_8b_gemma", "llama_3_2_3b_gemma"]:
        r, lo, hi = sp_ci(allpop[f], sc.loc[allpop.index, c])
        line.append(f"{c}={r:+.2f} [{lo:+.2f},{hi:+.2f}]")
    print(f"  {f:10s}", "  ".join(line))
allpop.to_csv(OUT / "05_pop_all92.csv")

# --- does row terseness explain the icl answer_first ~ 14B association? (partial Spearman via rank residuals)
def partial_sp(y, x, z):
    ry, rx, rz = [stats.rankdata(v) for v in (y, x, z)]
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return stats.pearsonr(res(ry), res(rx))[0]


y14 = meta.qwen3_14b_gemma.reindex(icl_ds.index)
print("\nicl answer_first ~ 14B: raw Spearman %.3f; partial controlling 150-row s_words %.3f; controlling s_le30w %.3f" % (
    stats.spearmanr(icl_ds.answer_first, y14)[0], partial_sp(icl_ds.answer_first.values, y14.values, pop.loc[icl_ds.index, "s_words"].values),
    partial_sp(icl_ds.answer_first.values, y14.values, pop.loc[icl_ds.index, "s_le30w"].values)))
print("icl n_words ~ 14B: raw %.3f; partial | s_words %.3f" % (stats.spearmanr(icl_ds.n_words2, y14)[0],
      partial_sp(icl_ds.n_words2.values, y14.values, pop.loc[icl_ds.index, "s_words"].values)))
print("ft_wrong ~ 14B: raw %.3f; partial | s_words %.3f" % (stats.spearmanr(meta.ft_wrong.reindex(icl_ds.index), y14)[0],
      partial_sp(meta.ft_wrong.reindex(icl_ds.index).values, y14.values, pop.loc[icl_ds.index, "s_words"].values)))
