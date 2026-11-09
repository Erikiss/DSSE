"""Dataset-level Spearman icl~ft and icl~14B for several aggregations of the 3 ICL seeds; bootstrap CIs over datasets.
Also: ft wrong from DATASET_TABLE vs own recomputation (n=31), and FT test-retest vs DSSE's own FT answers."""
import sys, numpy as np, pandas as pd
from scipy import stats
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_statistics_null-was-unmeasurable/"
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import DATASET_TABLE, dsse_eval, GOLD
rng = np.random.default_rng(7)
T = DATASET_TABLE.set_index("dataset_id")
R = pd.read_pickle(OUT + "R.pkl")
I = R[(R.route == "icl") & (R.k == 16)].copy()
I["log_words"] = np.log(I.n_words_v.clip(lower=1))
OUTL = "icl__dsse1-08152__k16__s0"
ftR = R[R.route == "ft"].groupby("dataset_id").agg(wrong=("wrong_v", "mean"), log_words=("n_words_v", lambda x: np.log(x.clip(lower=1)).mean()))
chk = pd.concat([ftR.wrong, T.ft_wrong], axis=1).dropna()
print("ft wrong own vs table: max abs diff", (chk.iloc[:, 0] - chk.iloc[:, 1]).abs().max(), "n", len(chk))
ft = T.ft_wrong          # n=32 incl. 09037 (summary only)
y14 = T.qwen3_14b_gemma
# DSSE FT test-retest for wrong
D = dsse_eval(); D["gold"] = D.prompt_idx.map(GOLD).astype(float)
NUM = __import__("re").compile(r"-?\d+(?:\.\d+)?")
def fn(s):
    n = NUM.findall(str(s).replace(",", "").replace("$", "")); return float(n[-1]) if n else np.nan
D["fin"] = D.response.map(fn); D["wrong"] = D.fin.notna() & ((D.fin - D.gold).abs() > 1e-6)
dsw = D.groupby("dataset_id").wrong.mean()
ids32 = T.index
def sp_ci(x, y, B=4000):
    x = np.asarray(x, float); y = np.asarray(y, float); ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
    r = stats.spearmanr(x, y).correlation
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x)); bs.append(stats.spearmanr(x[i], y[i]).correlation)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return r, lo, hi, len(x), stats.spearmanr(x, y).pvalue
r, lo, hi, n, p = sp_ci(ft.loc[ids32], dsw.loc[ids32])
print(f"FT test-retest (our ft_wrong vs DSSE-FT wrong, same 100 prompts): rho={r:.3f} [{lo:.2f},{hi:.2f}] n={n}")
r, lo, hi, n, p = sp_ci(dsw, T.reindex(dsw.index).qwen3_14b_gemma if False else pd.Series(dtype=float))  if False else (0,0,0,0,0)

rows = []
for met in ["wrong_v", "log_words"]:
    piv = I.groupby(["dataset_id", "seed"])[met].mean().unstack().reindex(ids32)
    piv_no = I[I.tag != OUTL].groupby(["dataset_id", "seed"])[met].mean().unstack().reindex(ids32)
    ranks = piv.rank()
    aggs = {"mean3 (all 96 runs)": piv.mean(1), "mean (drop 08152_s0)": piv_no.mean(1), "median of 3 seeds": piv.median(1),
            "mean of per-seed ranks": ranks.mean(1), "seed0": piv[0], "seed1": piv[1], "seed2": piv[2]}
    tgt_ft = ft if met == "wrong_v" else ftR.log_words.reindex(ids32)
    for name, x in aggs.items():
        for tname, yv in [("ft", tgt_ft), ("14B", y14)]:
            r, lo, hi, n, p = sp_ci(x, yv)
            rows.append(dict(metric=met, agg=name, target=tname, rho=r, lo=lo, hi=hi, n=n, p=p))
    # drop dataset 08152 entirely
    keep = [d for d in ids32 if d != "dsse1-08152"]
    for tname, yv in [("ft", tgt_ft), ("14B", y14)]:
        r, lo, hi, n, p = sp_ci(piv.mean(1).loc[keep], yv.loc[keep])
        rows.append(dict(metric=met, agg="mean3, dataset 08152 removed (n=31)", target=tname, rho=r, lo=lo, hi=hi, n=n, p=p))
C = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(C.round(3).to_string(index=False))
C.to_csv(OUT + "v03_correlations.csv", index=False)
print("\n08152: icl seeds", I[I.dataset_id == "dsse1-08152"].groupby("seed").wrong_v.mean().to_dict(), "ft", ft["dsse1-08152"], "14B", y14["dsse1-08152"])
print("rank of 08152 among 32 by mean3 icl wrong:", int(I.groupby("dataset_id").wrong_v.mean().rank(ascending=False)["dsse1-08152"]),
      "; by ft wrong:", int(ft.rank(ascending=False)["dsse1-08152"]))
