"""Dataset-level Spearman correlations (icl~ft, icl~14B, ft~14B) for rule and robust metrics, bootstrap CIs over
datasets, BH correction over the whole table; seed reliability of icl metrics (Spearman-Brown for the 3-seed mean);
top-vs-bottom group separation (AUC) because the 32 datasets are an extreme-groups sample."""
from common import *
from itertools import combinations
m = load_main()
T = DATASET_TABLE.set_index("dataset_id")
y14 = T["qwen3_14b_gemma"]
M = ["wrong", "wrong_r", "le2", "short40", "n_words", "answer_first", "ns1", "ec30w", "ec_novel", "working", "arith2", "n_arith", "boxed", "latex_block"]
icl = m[m.route == "icl"].groupby("dataset_id")[M].mean()
ft = m[m.route == "ft"].groupby("dataset_id")[M].mean()
print("n icl datasets", len(icl), "n ft datasets", len(ft))
rows = []
for k in M:
    j = icl[[k]].join(ft[[k]], lsuffix="_icl", rsuffix="_ft", how="inner")
    for name, x, y in [("icl~ft", j[k + "_icl"], j[k + "_ft"]),
                       ("icl~14B", icl[k], y14.reindex(icl.index)),
                       ("ft~14B", ft[k], y14.reindex(ft.index))]:
        r, lo, hi, p, n = spearman_ci(x, y)
        rows.append(dict(metric=k, pair=name, rho=r, lo=lo, hi=hi, p=p, n=n))
C = pd.DataFrame(rows)
C["q_bh"] = bh(C.p.fillna(1).values)
C.to_csv(OUT + "dataset_corr.csv", index=False)
C["s"] = C.apply(lambda r: f"{r.rho:+.2f} [{r.lo:+.2f},{r.hi:+.2f}] q={r.q_bh:.3f}", axis=1)
print(C.pivot(index="metric", columns="pair", values="s").loc[M, ["icl~ft", "icl~14B", "ft~14B"]].to_string())

# Seed reliability of icl dataset-level metrics
print("\nICL seed reliability (mean pairwise Spearman across the 3 seeds, n=32) and Spearman-Brown for the 3-seed mean:")
rel = {}
for k in M:
    per = m[m.route == "icl"].groupby(["dataset_id", "seed"])[k].mean().unstack()
    rs = [stats.spearmanr(per[a], per[b]).correlation for a, b in combinations(per.columns, 2)]
    r1 = np.mean(rs)
    rel[k] = 3 * r1 / (1 + 2 * r1) if r1 > -0.5 else np.nan
    # between-dataset SD vs expected binomial/sampling SD of the 3-seed mean
    sd_between = per.mean(1).std()
    print(f"  {k:13s} r_seed={r1:+.2f}  SB(3)={rel[k]:+.2f}  SD of 3-seed means across datasets={sd_between:.4f}")

# Extreme-groups: top vs bottom AUC for ft metrics and within-group Spearman with 14B
print("\nft metric: AUC(top vs bottom), within-top rho, within-bottom rho (n per group)")
g = T["group"]
for k in M:
    v = ft[k]
    top, bot = v[g.reindex(v.index) == "top"], v[g.reindex(v.index) == "bottom"]
    auc = stats.mannwhitneyu(top, bot).statistic / (len(top) * len(bot))
    wt = stats.spearmanr(top, y14.reindex(top.index)).correlation
    wb = stats.spearmanr(bot, y14.reindex(bot.index)).correlation
    print(f"  {k:13s} AUC={auc:.2f}  rho_top={wt:+.2f} (n={len(top)})  rho_bottom={wb:+.2f} (n={len(bot)})")
