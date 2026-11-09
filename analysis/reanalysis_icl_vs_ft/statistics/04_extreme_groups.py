"""(4) Extreme-groups design: 16 top + 16 bottom of 92 datasets on the 14B score.
(a) Empirical inflation: DSSE's own FT of Qwen3-1.7B exists for all 92 datasets -> compute the same metrics and
    compare Spearman with 14B over all 92, over the selected 32, within top, within bottom, and over the 60 unselected.
(b) Simulation: x = rho * normal-score(14B) + noise over the actual 92 14B values; expected Spearman in the selected 32
    as a function of the population rho -> map observed extreme-group rhos back to population rhos.
(c) Our ICL/FT measures within groups (bottom group: 14B range 0.04-0.08, binomial SE ~0.025 -> ranks are noise)."""
from common import *

F = pd.read_pickle(OUT + "F_runs.pkl")
D = pd.read_pickle(OUT + "F_dsse.pkl")
MET = ["wrong", "wrong_r", "le2", "answer_first", "n_words", "log_words", "first_frac", "ec20", "n_eq", "arith2"]
sc = pd.read_csv("/home/user/DSSE/data/scores.csv")
sc = sc[sc.behavior_id == "incorrect_answer"].set_index("dataset_id")
y = sc.qwen3_14b_gemma
T = DATASET_TABLE.set_index("dataset_id")
sel = T.index
top, bot = T.index[T.group == "top"], T.index[T.group == "bottom"]
uns = y.index.difference(sel)
for c in MET:
    D[c] = D[c].astype(float)
dm = D.groupby("dataset_id")[MET + ["opus_score"]].mean()
dm["dsse_opus_table"] = sc.qwen3_1_7b_opus
print(f"14B over 92: mean {y.mean():.3f} SD {y.std():.3f}; selected bottom {y[bot].min():.2f}-{y[bot].max():.2f}, "
      f"top {y[top].min():.2f}-{y[top].max():.2f}; unselected {y[uns].min():.2f}-{y[uns].max():.2f}")
rows = []
for met in MET + ["dsse_opus_table"]:
    r92 = spearman_ci(dm.loc[y.index, met], y)
    r32 = spearman_ci(dm.loc[sel, met], y[sel])
    rows.append(dict(metric="DSSE-FT " + met, rho_all92=r92[0], ci92=f"{r92[1]:.2f}..{r92[2]:.2f}", rho_sel32=r32[0],
                     ci32=f"{r32[1]:.2f}..{r32[2]:.2f}", rho_unsel60=spearman(dm.loc[uns, met], y[uns])[0],
                     rho_top16=spearman(dm.loc[top, met], y[top])[0], rho_bot16=spearman(dm.loc[bot, met], y[bot])[0],
                     inflation=r32[0] / r92[0]))
E = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(E.round(3).to_string(index=False))
E.to_csv(OUT + "04_dsse_inflation.csv", index=False)

# (b) simulation over the actual 14B values
rng = np.random.default_rng(0)
ns = stats.norm.ppf((stats.rankdata(y) - 0.5) / len(y))
ns = pd.Series(ns, index=y.index)
rho_grid = np.round(np.arange(0.0, 0.91, 0.1), 2)
sim = []
for rho in rho_grid:
    v32, v92, vt, vb = [], [], [], []
    for _ in range(2000):
        x = rho * ns + np.sqrt(1 - rho ** 2) * rng.normal(size=len(ns))
        v92.append(stats.spearmanr(x, y).correlation)
        v32.append(stats.spearmanr(x[sel], y[sel]).correlation)
        vt.append(stats.spearmanr(x[top], y[top]).correlation)
        vb.append(stats.spearmanr(x[bot], y[bot]).correlation)
    sim.append(dict(rho_latent=rho, E_rho92=np.mean(v92), E_rho_sel32=np.mean(v32), sd_sel32=np.std(v32),
                    E_rho_top=np.mean(vt), E_rho_bot=np.mean(vb)))
Sim = pd.DataFrame(sim)
print("\nSimulated Spearman with the 14B score: full 92 vs extreme-group 32 (actual 14B values, ties included)")
print(Sim.round(3).to_string(index=False))
Sim.to_csv(OUT + "04_sim_inflation.csv", index=False)


def back_map(r_obs):
    return np.interp(r_obs, Sim.E_rho_sel32, Sim.E_rho92)


for lab, r in [("ft wrong~14B", 0.63), ("ft answer_first~14B", 0.62), ("ft le2~14B", 0.64), ("icl answer_first~14B", 0.43),
               ("icl wrong~14B", 0.12)]:
    print(f"  observed {lab} = {r:.2f} in the 32 -> implied Spearman over all 92 ~ {back_map(r):.2f}")

# (c) our measures within groups
OUTLIER = "icl__dsse1-08152__k16__s0"
for c in MET:
    F[c] = F[c].astype(float)
I = F[(F.route == "icl") & (F.k == 16)]
icl = I[I.tag != OUTLIER].groupby("dataset_id")[MET].mean()
icl_all = I.groupby("dataset_id")[MET].mean()
ft = F[F.route == "ft"].groupby("dataset_id")[MET].mean()
for c in ["wrong", "le2", "answer_first"]:
    ft.loc["dsse1-09037", c] = T.loc["dsse1-09037", "ft_" + c]
rows = []
for met in MET:
    for lab, xx, yy in [("icl(no outlier)~14B", icl[met], y), ("icl(all)~14B", icl_all[met], y), ("ft~14B", ft[met], y),
                        ("icl(no outlier)~ft", icl[met], ft[met]), ("icl(all)~ft", icl_all[met], ft[met])]:
        d = dict(metric=met, pair=lab)
        for g, ids in [("all32", sel), ("top", top), ("bottom", bot)]:
            ii = [i for i in ids if i in xx.index and i in yy.index]
            d[g] = spearman(xx[ii], yy[ii])[0]
            d["n_" + g] = len(ii)
        # group-adjusted (rank within group)
        ii = [i for i in sel if i in xx.index and i in yy.index and pd.notna(xx[i]) and pd.notna(yy[i])]
        gg = T.group[ii]
        d["within_pooled"] = stats.spearmanr(xx[ii].groupby(gg).rank(pct=True), yy[ii].groupby(gg).rank(pct=True)).correlation
        top_ = [i for i in ii if T.group[i] == "top"]; bot_ = [i for i in ii if T.group[i] == "bottom"]
        d["AUC_top_vs_bottom_x"] = stats.mannwhitneyu(xx[top_], xx[bot_]).statistic / (len(top_) * len(bot_))
        rows.append(d)
W = pd.DataFrame(rows)
print("\nWithin-group Spearman (our runs):")
print(W.round(3).to_string(index=False))
W.to_csv(OUT + "04_within_groups.csv", index=False)
