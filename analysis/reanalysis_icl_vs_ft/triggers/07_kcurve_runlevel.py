"""k dose-response at run level (avoids the shared base run and the rank-deficient clustered SE of 06(b)):
run-level problem-adjusted outcome = mean over problems of (y - per-problem mean over all 96 main icl runs),
regressed on log2(k) with dataset FE, k in {1,2,4,8,16,32}, 4 datasets x 6 = 24 runs (seed 0 only).
Also exact shot features per k (nested samples for k<=16; k=32 is a different draw because random.sample switches algorithm)."""
from common import *
import statsmodels.formula.api as smf

R = pd.read_pickle(OUT / "resp_r.pkl")
icl = R[R.route == "icl"]
ref = icl[icl.k == 16].groupby("prompt_idx")[["wrong_len", "n_words2", "short40", "has_arith", "answer_first"]].mean()
ds4 = ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]
cur = icl[(icl.dataset_id.isin(ds4)) & (icl.seed == 0)].copy()
for y in ref.columns:
    cur[y + "_adj"] = cur[y] - cur.prompt_idx.map(ref[y])
run = cur.groupby(["dataset_id", "k"])[[c + "_adj" for c in ref.columns] + list(ref.columns)].mean().reset_index()
run["logk"] = np.log2(run.k)
runs = pd.read_csv(OUT / "runs.csv")
run = run.merge(runs[["dataset_id", "k", "seed", "s_words", "s_le30w", "context_tokens"]].query("seed == 0"), on=["dataset_id", "k"])
for y in ref.columns:
    m = smf.ols(f"{y}_adj ~ logk + C(dataset_id)", data=run).fit(cov_type="HC3")
    ci = m.conf_int().loc["logk"]
    print(f"{y:12s} per doubling of k: {m.params.logk:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] p={m.pvalues.logk:.3g}   (n=24 runs)")
pd.set_option("display.width", 200)
print(run[["dataset_id", "k", "s_words", "context_tokens", "wrong_len", "n_words2", "short40", "has_arith", "answer_first"]].round(3).to_string())
run.to_csv(OUT / "07_kcurve_runs.csv", index=False)
