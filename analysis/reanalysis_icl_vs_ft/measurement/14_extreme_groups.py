"""The 32 datasets are an extreme-groups sample (14B: bottom 0.04-0.08, top 0.33-0.60). Split each dataset-level Spearman
into the between-group part (top minus bottom, AUC) and the within-group part (Spearman inside top / inside bottom,
and the pooled Spearman after removing group means)."""
from common import *
m = load_main()
T = DATASET_TABLE.set_index("dataset_id")
M = ["wrong_r", "le2", "short40", "n_words", "answer_first", "ns1", "ec_novel", "arith2"]
res = []
for route in ["icl", "ft"]:
    g = m[m.route == route].groupby("dataset_id")[M].mean()
    grp = T.group.reindex(g.index)
    y = T.qwen3_14b_gemma.reindex(g.index)
    for k in M:
        top, bot = g[k][grp == "top"], g[k][grp == "bottom"]
        auc = stats.mannwhitneyu(top, bot).statistic / (len(top) * len(bot))
        # within-group pooled: rank within group, correlate
        rx = g[k].groupby(grp).rank(pct=True)
        ry = y.groupby(grp).rank(pct=True)
        rw = stats.spearmanr(rx, ry).correlation
        res.append(dict(route=route, metric=k, rho_all=stats.spearmanr(g[k], y).correlation, AUC_top_vs_bottom=auc,
                        diff_top_minus_bottom=top.mean() - bot.mean(),
                        rho_within_top=stats.spearmanr(top, y[grp == "top"]).correlation,
                        rho_within_bottom=stats.spearmanr(bot, y[grp == "bottom"]).correlation, rho_within_pooled=rw))
R = pd.DataFrame(res)
pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
R.to_csv(OUT + "extreme_groups.csv", index=False)
print("\n14B score range: bottom %.2f-%.2f (n=16), top %.2f-%.2f (n=16); ties in bottom: %d distinct values" % (
    T[T.group == "bottom"].qwen3_14b_gemma.min(), T[T.group == "bottom"].qwen3_14b_gemma.max(),
    T[T.group == "top"].qwen3_14b_gemma.min(), T[T.group == "top"].qwen3_14b_gemma.max(), T[T.group == "bottom"].qwen3_14b_gemma.nunique()))
print("binomial SE of a 14B score at p=0.06 with 100 prompts: %.3f" % np.sqrt(0.06 * 0.94 / 100))
