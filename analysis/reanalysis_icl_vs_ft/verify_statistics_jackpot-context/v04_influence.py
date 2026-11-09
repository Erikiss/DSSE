"""Does the jackpot run 'make the dataset means unreliable'? (a) seed-to-seed reliability with/without it,
(b) influence of the jackpot vs every other single run on the headline dataset-level Spearman correlations
(single-run jackknife over all 96 k16 runs: dataset mean recomputed from the remaining seeds),
(c) headline correlations under mean-of-3, mean-without-jackpot, median-of-3, drop-08152, with bootstrap CIs over datasets."""
import sys
import numpy as np, pandas as pd
from scipy import stats
SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
OUT = SCR + "/analysis/verify_statistics_jackpot-context/"
sys.path.insert(0, SCR)
from load_runs import DATASET_TABLE
T = DATASET_TABLE.set_index("dataset_id")
RL = pd.read_csv(OUT + "v01_runlevel.csv")
I = RL[(RL.route == "icl") & (RL.k == 16)].copy()
J = "icl__dsse1-08152__k16__s0"
I = I.rename(columns={"af": "answer_first"})
MET = ["wrong", "le2", "answer_first", "logw", "short40"]
ids = T.index
y14 = T.qwen3_14b_gemma
rng = np.random.default_rng(0)
def sp(a, b): return stats.spearmanr(a, b).correlation
def boot(a, b, B=4000):
    a, b = np.asarray(a, float), np.asarray(b, float); n = len(a)
    v = [sp(a[i], b[i]) for i in (rng.integers(0, n, n) for _ in range(B))]
    return np.nanpercentile(v, [2.5, 97.5])
# (a) reliability
print("(a) pairwise seed Spearman over datasets (n=32; n=31 for pairs with s0 when 08152 dropped)")
for m in MET:
    piv = I.pivot(index="dataset_id", columns="seed", values=m)
    allp = [sp(piv[a], piv[b]) for a, b in [(0, 1), (0, 2), (1, 2)]]
    p2 = piv.drop("dsse1-08152")
    nop = [sp(p2[a], p2[b]) for a, b in [(0, 1), (0, 2), (1, 2)]]
    # ICC(1), balanced one-way, with and without (without = drop 08152 entirely to keep balance)
    def icc(p):
        x = p.values; k = x.shape[1]; gm = x.mean(1); msb = k * gm.var(ddof=1); msw = ((x - gm[:, None]) ** 2).sum() / (x.shape[0] * (k - 1))
        return (msb - msw) / (msb + (k - 1) * msw)
    print(f"   {m:12s} all: {np.mean(allp):+.2f} ({'/'.join(f'{x:.2f}' for x in allp)})  ICC1 {icc(piv):+.2f}   |  08152 dropped: {np.mean(nop):+.2f} ({'/'.join(f'{x:.2f}' for x in nop)})  ICC1 {icc(p2):+.2f}")
# (c) headline correlations under variants
ftcol = {"wrong": "ft_wrong", "le2": "ft_le2", "answer_first": "ft_answer_first"}
variants = {"mean of 3 (reported)": I.groupby("dataset_id")[MET].mean(),
            "mean w/o jackpot run": I[I.tag != J].groupby("dataset_id")[MET].mean(),
            "median of 3": I.groupby("dataset_id")[MET].median()}
print("\n(c) dataset-level Spearman (n=32) with 95% bootstrap CI over datasets")
rows = []
for vn, M in variants.items():
    M = M.reindex(ids)
    for m in ["wrong", "le2", "answer_first"]:
        for tgt, yv in [("ft", T[ftcol[m]]), ("14B", y14)]:
            r = sp(M[m], yv); lo, hi = boot(M[m].values, yv.values)
            keep = ids != "dsse1-08152"
            r31 = sp(M[m][keep], yv[keep])
            rows.append(dict(variant=vn, metric=m, vs=tgt, rho=r, lo=lo, hi=hi, rho_drop08152=r31))
# in-domain words icl (seed 0 only) vs ft
r = sp(T.icl_id_words, T.ft_id_words); lo, hi = boot(T.icl_id_words.values, T.ft_id_words.values)
keep = ids != "dsse1-08152"
rows.append(dict(variant="seed0 only (reported)", metric="id_words", vs="ft", rho=r, lo=lo, hi=hi, rho_drop08152=sp(T.icl_id_words[keep], T.ft_id_words[keep])))
C = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(C.round(2).to_string(index=False))
C.to_csv(OUT + "v04_headline_variants.csv", index=False)
# (b) single-run jackknife: remove one run, recompute that dataset's mean from its other 2 seeds
print("\n(b) single-run jackknife over 96 runs: change in rho when the run is removed (dataset mean from remaining 2 seeds)")
base = variants["mean of 3 (reported)"].reindex(ids)
for m, tgt, yv in [("wrong", "ft", T.ft_wrong), ("wrong", "14B", y14), ("answer_first", "14B", y14), ("answer_first", "ft", T.ft_answer_first), ("le2", "ft", T.ft_le2), ("le2", "14B", y14)]:
    r0 = sp(base[m], yv); deltas = []
    for t in I.tag:
        M = I[I.tag != t].groupby("dataset_id")[m].mean().reindex(ids)
        deltas.append((t, sp(M, yv) - r0))
    d = pd.DataFrame(deltas, columns=["tag", "delta"]).set_index("tag").delta
    rk = (d.abs() >= abs(d[J])).sum()
    print(f"   {m:12s} vs {tgt:3s}: rho {r0:+.2f}; removing jackpot {d[J]:+.3f} (rank {rk}/96 by |delta|); "
          f"other runs |delta| median {d.drop(J).abs().median():.3f}, max {d.drop(J).abs().max():.3f} ({d.drop(J).abs().idxmax()})")
# ranks of 08152 in the dataset means
for vn, M in variants.items():
    print(f"   rank of 08152 icl wrong among 32 datasets [{vn}]: {int(M.wrong.rank().loc['dsse1-08152'])}/32, value {M.wrong.loc['dsse1-08152']:.3f}")
print("   dataset means of icl wrong: SD over datasets (mean of 3) %.3f; without jackpot %.3f" % (variants["mean of 3 (reported)"].wrong.std(), variants["mean w/o jackpot run"].wrong.std()))
print("   grand mean icl wrong: all 96 runs %.4f; 95 runs %.4f;  le2 %.3f/%.3f; af %.3f/%.3f" % (
    I.wrong.mean(), I[I.tag != J].wrong.mean(), I.le2.mean(), I[I.tag != J].le2.mean(), I.answer_first.mean(), I[I.tag != J].answer_first.mean()))
