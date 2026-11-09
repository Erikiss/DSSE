"""Dataset-level Spearman with 14B transfer (bootstrap over datasets), partials, cross-run reliability, replication on
DSSE's own FT answers (n=92 and the 60 datasets NOT in our extreme-groups selection), within-group, threshold sweeps,
missing-09037 imputation. Uses only v_metrics.py outputs."""
import sys

import numpy as np
import pandas as pd
from scipy import stats

OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_measurement_ft-format-effect-is-brevity-not-answer-first/"
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import DATASET_TABLE  # noqa

B = 5000
rng0 = np.random.default_rng(12345)


def rk(v):
    return stats.rankdata(v)


def sp(x, y):
    return np.corrcoef(rk(x), rk(y))[0, 1]


def pr(x, y, z):
    """Partial Spearman: Pearson partial on ranks (formula)."""
    rx, ry, rz = rk(x), rk(y), rk(z)
    c = np.corrcoef([rx, ry, rz])
    a, b, g = c[0, 1], c[0, 2], c[1, 2]
    return (a - b * g) / np.sqrt((1 - b * b) * (1 - g * g))


def boot(f, *cols, B=B, seed=1):
    cols = [np.asarray(c, float) for c in cols]
    n = len(cols[0])
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        try:
            with np.errstate(all="ignore"):
                r = f(*[c[i] for c in cols])
            if np.isfinite(r):
                v.append(r)
        except Exception:
            pass
    return f(*cols), *np.percentile(v, [2.5, 97.5])


def fmt(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"


ours = pd.read_pickle(OUT + "ours_ft.pkl")
dsse = pd.read_pickle(OUT + "dsse_ft.pkl")
T = DATASET_TABLE.set_index("dataset_id")
sc = pd.read_csv("/home/user/DSSE/data/scores.csv").query("behavior_id=='incorrect_answer'").set_index("dataset_id")

MET = ["wrong_v", "short40", "words", "eq2", "ops_any1", "n_ops_any", "bare", "ec_A", "ec_B", "ec_C", "af_rule"]
O = ours.groupby("dataset_id")[MET].mean()
Dd = dsse.groupby("dataset_id")[MET + ["opus_wrong"]].mean()
O["y"] = T.qwen3_14b_gemma.reindex(O.index)
Dd["y"] = sc.qwen3_14b_gemma.reindex(Dd.index)
print(f"n ours={len(O)}  n dsse={len(Dd)}")

print("\n=== 1. Spearman with 14B: ours FT (n=31) | DSSE FT all (n=92) | DSSE FT non-selected (n=60) | DSSE FT the same 31")
sel = set(T.index)
Dn = Dd[~Dd.index.isin(sel)]
D31 = Dd.loc[O.index]
res = []
for k in MET + ["opus_wrong"]:
    a = boot(sp, O[k], O.y) if k in O else (np.nan,) * 3
    b = boot(sp, Dd[k], Dd.y)
    c = boot(sp, Dn[k], Dn.y)
    d = boot(sp, D31[k], D31.y)
    res.append((k, a, b, c, d))
    print(f"  {k:10s} ours {fmt(a)}   dsse92 {fmt(b)}   dsse60 {fmt(c)}   dsse31 {fmt(d)}")
# BH over the 4*12 tests (approximate p from scipy)
ps = []
for k in MET:
    ps += [stats.spearmanr(O[k], O.y).pvalue, stats.spearmanr(Dd[k], Dd.y).pvalue, stats.spearmanr(Dn[k], Dn.y).pvalue]
ps = np.array(ps)
from statsmodels.stats.multitest import multipletests
rej, q, _, _ = multipletests(ps, method="fdr_bh")
print("  BH q-values (ours, dsse92, dsse60) per metric:")
for i, k in enumerate(MET):
    print(f"    {k:10s} " + "  ".join(f"{x:.4f}" for x in q[3 * i:3 * i + 3]))

print("\n=== 2. Cross-run reliability (ours vs DSSE FT, same dataset, n=31)")
for k in MET:
    print(f"  {k:10s} {fmt(boot(sp, O[k], D31[k]))}   means {O[k].mean():.3f} / {D31[k].mean():.3f}")
print(f"  ours wrong_v ~ dsse opus_wrong {fmt(boot(sp, O.wrong_v, D31.opus_wrong))}")

print("\n=== 3. Collinearity and partials")
for name, P, w in [("ours", O, "wrong_v"), ("dsse92", Dd, "opus_wrong"), ("dsse60", Dn, "opus_wrong")]:
    print(f" {name}: rho(af_rule,words)={sp(P.af_rule, P.words):+.2f} rho({w},words)={sp(P[w], P.words):+.2f} rho(ec_A,words)={sp(P.ec_A, P.words):+.2f}")
    for x, z in [("af_rule", "words"), (w, "words"), ("words", w), ("ec_A", "words"), ("short40", w), ("bare", "words"), ("af_rule", "short40")]:
        print(f"   pr({x},14B | {z}) {fmt(boot(pr, P[x], P.y, P[z]))}")

print("\n=== 4. Within extreme groups (ours, n=15/16 each) and top-vs-bottom AUC")
g = T.group.reindex(O.index)
for k in ["wrong_v", "words", "short40", "ec_A", "af_rule", "bare"]:
    top, bot = O[g == "top"], O[g == "bottom"]
    auc = stats.mannwhitneyu(top[k], bot[k]).statistic / (len(top) * len(bot))
    print(f"  {k:9s} AUC={auc:.2f}  rho_top={sp(top[k], top.y):+.2f} (n={len(top)})  rho_bot={sp(bot[k], bot.y):+.2f} (n={len(bot)})")

print("\n=== 5. Threshold sweep for 'short' (ours n=31 | dsse92)")
for th in [15, 20, 30, 40, 60, 80, 120]:
    a = ours.assign(s=ours.words <= th).groupby("dataset_id").s.mean()
    b = dsse.assign(s=dsse.words <= th).groupby("dataset_id").s.mean()
    print(f"  <= {th:3d} words: ours {sp(a, O.y.reindex(a.index)):+.2f} (mean {a.mean():.2f})  dsse92 {sp(b, Dd.y.reindex(b.index)):+.2f} (mean {b.mean():.2f})")

print("\n=== 6. Early commitment conditional on length (answers >= 30 words), datasets with >= 10 such answers")
for name, d, y in [("ours", ours, O.y), ("dsse92", dsse, Dd.y)]:
    L = d[d.words >= 30]
    gg = L.groupby("dataset_id").agg(ecA=("ec_A", "mean"), ecB=("ec_B", "mean"), ecC=("ec_C", "mean"), n=("ec_A", "size"))
    gg = gg[gg.n >= 10]
    yy = y.reindex(gg.index)
    print(f"  {name}: n_datasets={len(gg)}  ecA {fmt(boot(sp, gg.ecA, yy))} (mean {gg.ecA.mean():.3f})  ecB {fmt(boot(sp, gg.ecB, yy))}  ecC {fmt(boot(sp, gg.ecC, yy))}")
    # also: n of long answers per dataset vs 14B (selection: who stays in the conditional sample)
    print(f"     rho(n_long, 14B) = {sp(gg.n, yy):+.2f}")

print("\n=== 7. Missing ft__dsse1-09037: impute from DATASET_TABLE summary / DSSE's run")
print("  09037 summary:", T.loc["dsse1-09037", ["ft_wrong", "ft_le2", "ft_answer_first", "ft_working", "qwen3_14b_gemma"]].to_dict())
print("  09037 DSSE-run metrics:", Dd.loc["dsse1-09037", ["words", "short40", "ec_A", "af_rule", "wrong_v", "opus_wrong"]].round(3).to_dict())
O2 = O.copy()
# impute every metric with DSSE's run value for 09037, shifted by the mean ours-minus-DSSE difference over the 31
row = Dd.loc["dsse1-09037", MET] + (O[MET].mean() - D31[MET].mean())
row["y"] = T.loc["dsse1-09037", "qwen3_14b_gemma"]
O2.loc["dsse1-09037"] = row
for k in ["wrong_v", "short40", "words", "eq2", "ec_A", "af_rule"]:
    print(f"  ours+imputed09037 n=32 {k:8s} {fmt(boot(sp, O2[k], O2.y))}  (n=31: {sp(O[k], O.y):+.2f}); rank of 09037 on {k}: {int(O2[k].rank().loc['dsse1-09037'])}/32")

print("\n=== 8. Leave-one-dataset-out range (ours) for key metrics")
for k in ["short40", "words", "wrong_v", "ec_A", "af_rule"]:
    loo = [sp(O[k].drop(i), O.y.drop(i)) for i in O.index]
    print(f"  {k:8s} LOO min {min(loo):+.2f} max {max(loo):+.2f}")

O.to_csv(OUT + "ours_ft_per_dataset.csv")
Dd.to_csv(OUT + "dsse_ft_per_dataset.csv")
