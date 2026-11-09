"""Sensitivity checks: (a) median-of-3-seeds instead of mean / outlier exclusion; (b) short40 (answers <=40 words) as a
mixture indicator; (c) in-domain icl~ft without 08152 (its only in-domain icl run is the jackpot context);
(d) 09037 missing at response level: put it at the extreme rank and recompute; (e) icl vs base mean shift, paired by
problem, using both independent base samples (base, ftcurve 08530 step0)."""
from common import *

F = pd.read_pickle(OUT + "F_runs.pkl")
T = DATASET_TABLE.set_index("dataset_id")
y = T.qwen3_14b_gemma
F["short40"] = (F.n_words <= 40).astype(float)
MET = ["wrong", "wrong_r", "le2", "answer_first", "n_words", "log_words", "first_frac", "short40"]
for c in MET:
    F[c] = F[c].astype(float)
I = F[(F.route == "icl") & (F.k == 16)]
RM = I.groupby(["tag", "dataset_id", "seed"])[MET].mean().reset_index()
med = RM.groupby("dataset_id")[MET].median()
ft = F[F.route == "ft"].groupby("dataset_id")[MET].mean()
for c in ["wrong", "le2", "answer_first"]:
    ft.loc["dsse1-09037", c] = T.loc["dsse1-09037", "ft_" + c]
print("(a) median of 3 seeds (no exclusion):")
for met in MET:
    r1 = spearman_ci(med[met].reindex(T.index), ft[met].reindex(T.index))
    r2 = spearman_ci(med[met].reindex(T.index), y)
    print(f"   {met:12s} icl~ft {r1[0]:+.2f} [{r1[1]:+.2f},{r1[2]:+.2f}] n={r1[3]}   icl~14B {r2[0]:+.2f} [{r2[1]:+.2f},{r2[2]:+.2f}]")

# (b) short40
piv = RM.pivot(index="dataset_id", columns="seed", values="short40")
print("\n(b) short40 per run: mean %.3f, SD %.3f, max %.2f (%s)" % (RM.short40.mean(), RM.short40.std(), RM.short40.max(),
      RM.loc[RM.short40.idxmax(), "tag"]))
x = RM[RM.tag != "icl__dsse1-08152__k16__s0"]
print("    without outlier: mean %.3f SD %.3f; base %.2f; ft mean %.3f" % (x.short40.mean(), x.short40.std(),
      F[F.route == "base"].short40.mean(), ft.short40.mean()))
print("    pairwise seed Spearman:", [round(stats.spearmanr(piv[a], piv[b]).correlation, 2) for a, b in [(0, 1), (0, 2), (1, 2)]])

# (c) in-domain without 08152
ids = T.index.drop("dsse1-08152")
print("\n(c) in-domain words icl~ft all: %.2f (n=%d); without 08152: %.2f (n=%d)" % (
    spearman(T.icl_id_words, T.ft_id_words) + spearman(T.loc[ids, "icl_id_words"], T.loc[ids, "ft_id_words"])))
print("    icl in-domain words over datasets: min %.0f (%s), median %.0f, max %.0f; base %.0f" % (
    T.icl_id_words.min(), T.icl_id_words.idxmin(), T.icl_id_words.median(), T.icl_id_words.max(), 396.1))

# (d) 09037 imputed at the extreme (shortest / most FT-like) rank for robust ft metrics
icl_no = RM[RM.tag != "icl__dsse1-08152__k16__s0"].groupby("dataset_id")[MET].mean()
for met, extreme in [("n_words", -1e9), ("log_words", -1e9), ("first_frac", -1e9), ("short40", 1e9)]:
    f2 = ft[met].reindex(T.index).copy()
    r0 = spearman(icl_no[met].reindex(T.index), f2)[0]; r14_0 = spearman(f2, y)[0]
    for v, lab in [(extreme, "most FT-like"), (-extreme, "least FT-like")]:
        f2["dsse1-09037"] = v
        print(f"(d) {met:10s} 09037 imputed {lab:13s}: icl(no outl)~ft {spearman(icl_no[met].reindex(T.index), f2)[0]:+.2f} "
              f"(observed n=31: {r0:+.2f}); ft~14B {spearman(f2, y)[0]:+.2f} (n=31: {r14_0:+.2f})")

# (e) paired shift icl vs base (two independent base samples)
B = F[F.tag.isin(["base", "ftcurve__dsse1-08530__step0"])].groupby("prompt_idx")[["wrong", "wrong_r", "n_words", "log_words"]].mean()
Ip = I[I.tag != "icl__dsse1-08152__k16__s0"].groupby("prompt_idx")[["wrong", "wrong_r", "n_words", "log_words"]].mean()
rng = np.random.default_rng(0)
for met in ["wrong", "wrong_r", "n_words", "log_words"]:
    d = (Ip[met] - B[met]).values
    bs = [d[rng.integers(0, 100, 100)].mean() for _ in range(4000)]
    print(f"(e) icl - base (2 base samples, 95 icl runs; bootstrap over 100 problems) {met:9s}: {d.mean():+.3f} "
          f"[{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]")
print("    note: the problem bootstrap ignores the sampling noise of the 2 base runs on fixed problems (SD of a 2-run mean ~0.02)")
