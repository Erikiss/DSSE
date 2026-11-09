"""(1) Metrics the claim says have 'reliability 0' (n_words, log_words, le2) vs their observed icl~ft Spearman (all 96 runs).
(2) What the reliable part of the dataset-level ICL wrong signal tracks: shot-context form (mean words of the 16 context
    responses, per run) -> run-level and dataset-level Spearman."""
import sys, numpy as np, pandas as pd
from scipy import stats
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_statistics_null-was-unmeasurable/"
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import DATASET_TABLE, shots_for, dataset_rows
rng = np.random.default_rng(3)
T = DATASET_TABLE.set_index("dataset_id")
R = pd.read_pickle(OUT + "R.pkl")
R["log_words"] = np.log(R.n_words_v.clip(lower=1)); R["le2f"] = R.le2.astype(float); R["afs"] = R.answer_first.astype(float)
I = R[(R.route == "icl") & (R.k == 16)]
F = R[R.route == "ft"].groupby("dataset_id")[["n_words_v", "log_words", "le2f", "afs", "wrong_v"]].mean()
F.loc["dsse1-09037", ["le2f", "afs", "wrong_v"]] = T.loc["dsse1-09037", ["ft_le2", "ft_answer_first", "ft_wrong"]].values
Iall = I.groupby("dataset_id")[["n_words_v", "log_words", "le2f", "afs", "wrong_v"]].mean()
def ci(x, y, B=4000):
    d = pd.concat([x, y], axis=1).dropna().values
    r = stats.spearmanr(d[:, 0], d[:, 1])
    bs = [stats.spearmanr(*d[rng.integers(0, len(d), len(d))].T).correlation for _ in range(B)]
    return r.correlation, *np.nanpercentile(bs, [2.5, 97.5]), r.pvalue, len(d)
print("(1) icl 3-seed mean (all 96 runs) ~ ft, Spearman [boot CI], p, n")
for m in ["n_words_v", "log_words", "le2f", "afs", "wrong_v"]:
    print(f"   {m:10s}", np.round(ci(Iall[m], F[m]), 3))
# (2) context form
runs = I.groupby(["tag", "dataset_id", "seed"]).agg(wrong=("wrong_v", "mean"), lw=("log_words", "mean")).reset_index()
runs["ctx_words"] = [np.mean([len(x["response"].split()) for x in shots_for(d, int(s))]) for d, s in zip(runs.dataset_id, runs.seed)]
runs["ctx_short"] = [np.mean([len(x["response"].split()) <= 15 for x in shots_for(d, int(s))]) for d, s in zip(runs.dataset_id, runs.seed)]
runs["ds_words"] = [np.mean([len(x["response"].split()) for x in dataset_rows(d)]) for d in runs.dataset_id]
print("\n(2) run level (96 runs): Spearman(run wrong, ctx mean words) =", round(stats.spearmanr(runs.wrong, runs.ctx_words).correlation, 3),
      "; (run log_words, ctx words) =", round(stats.spearmanr(runs.lw, runs.ctx_words).correlation, 3))
print("    jackpot run ctx_words %.1f ctx_short %.2f; distribution of ctx_words over runs: median %.1f, min %.1f"
      % (*runs.loc[runs.tag == "icl__dsse1-08152__k16__s0", ["ctx_words", "ctx_short"]].values[0], runs.ctx_words.median(), runs.ctx_words.min()))
r95 = runs[runs.tag != "icl__dsse1-08152__k16__s0"]
dm = r95.groupby("dataset_id").agg(wrong=("wrong", "mean"), lw=("lw", "mean"), ds_words=("ds_words", "first"))
print("    dataset level (no jackpot run): icl wrong ~ dataset mean row words", np.round(ci(dm.wrong, dm.ds_words), 3),
      "; icl log_words ~ row words", np.round(ci(dm.lw, dm.ds_words), 3))
print("    ft wrong ~ dataset mean row words", np.round(ci(F.wrong_v, dm.ds_words), 3), "; 14B ~ row words", np.round(ci(T.qwen3_14b_gemma, dm.ds_words), 3))
