"""(a) How reliable is a single fine-tune's 1.7B error rate as a dataset property? Replicate pairs (ours vs DSSE, 31 datasets),
bootstrap CI; lower bound from max observed correlation. (b) 'premature stop': final number equals a non-final reference
intermediate (the answer to a sub-question) - by group, among Opus-incorrect answers; and as a dataset-level predictor."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/"
OUT = SCR + "analysis/dsse_crosscheck/"
T = pd.read_csv(OUT + "per_dataset.csv", index_col=0)
A = pd.read_pickle(OUT + "dsse_answers.pkl")
O = pd.read_pickle(OUT + "ours_ft_answers.pkl")
SOL = {s["prompt_idx"]: s for s in json.load(open(SCR + "gsm8k_eval_solutions.json"))}
rng = np.random.default_rng(8)

# (a)
xo = O.groupby("dataset_id").rule_wrong.mean(); xd = A.groupby("dataset_id").rule_wrong.mean().loc[xo.index]
xd_op = A.groupby("dataset_id").opus_wrong.mean().loc[xo.index]
var92 = A.groupby("dataset_id").rule_wrong.mean().var()
def rel(i):
    return 1 - 0.5 * np.var(xo.values[i] - xd.values[i], ddof=1) / var92
bs = [rel(rng.integers(0, len(xo), len(xo))) for _ in range(4000)]
print("replicate test-retest (31 extreme datasets): Pearson rule %.2f, Spearman rule %.2f, Spearman ours-rule vs DSSE-opus %.2f" % (
    np.corrcoef(xo, xd)[0, 1], spearmanr(xo, xd)[0], spearmanr(xo, xd_op)[0]))
print("reliability over the 92-dataset spread: %.2f [%.2f, %.2f]" % (rel(np.arange(len(xo))), *np.percentile(bs, [2.5, 97.5])))
# sampling-only reliability (binomial) for the 92
p = A.groupby("dataset_id").opus_wrong.mean()
print("sampling-only reliability of 1.7B opus error over 92: %.2f" % (1 - (p * (1 - p) / 100).mean() / p.var()))
print("max observed |rho| of 1.7B opus error with a target: 4B %.2f -> implied reliability >= %.2f (if target perfectly reliable)" % (
    spearmanr(T.qwen3_1_7b_opus, T.qwen3_4b_gemma)[0], spearmanr(T.qwen3_1_7b_opus, T.qwen3_4b_gemma)[0] ** 2))
for f in ["w40", "le2", "n_words"]:
    fo = O.groupby("dataset_id")[f].mean(); fd = A.groupby("dataset_id")[f].mean().loc[fo.index]
    v = A.groupby("dataset_id")[f].mean().var()
    print("  form %s replicate reliability %.2f" % (f, 1 - 0.5 * np.var(fo.values - fd.values, ddof=1) / v))

# (b) premature stop
def inter_final(r):
    if r.final is None or (isinstance(r.final, float) and np.isnan(r.final)):
        return False
    ints = SOL[r.prompt_idx]["intermediates"][:-1]
    try:
        return any(abs(float(r.final) - float(i)) < 1e-6 for i in ints) and abs(float(r.final) - float(SOL[r.prompt_idx]["gold"])) > 1e-6
    except ValueError:
        return False

A["final_is_intermediate"] = [inter_final(r) for r in A.itertuples()]
A["group"] = A.dataset_id.map(T.group)
W = A[A.opus_wrong == 1]
print("\nOpus-incorrect answers whose final number = a reference intermediate (stopped at a sub-result):",
      W.groupby("group").final_is_intermediate.mean().round(3).to_dict(), "n=", W.groupby("group").size().to_dict())
print("by length among Opus-incorrect: <=40w %.3f, >40w %.3f" % (W[W.w40].final_is_intermediate.mean(), W[~W.w40].final_is_intermediate.mean()))
T["dsse_stop_inter"] = A.groupby("dataset_id").final_is_intermediate.mean()
R = T.rank()
def pc(df, a, b, ctrl):
    Z = np.c_[np.ones(len(df)), df[ctrl].values]
    ra = df[a].values - Z @ np.linalg.lstsq(Z, df[a].values, rcond=None)[0]
    rb = df[b].values - Z @ np.linalg.lstsq(Z, df[b].values, rcond=None)[0]
    return np.corrcoef(ra, rb)[0, 1]
for t in ["qwen3_14b_gemma", "qwen3_8b_gemma", "qwen3_4b_gemma"]:
    b1 = [pc(R.iloc[i], "dsse_stop_inter", t, ["qwen3_1_7b_opus"]) for i in [rng.integers(0, 92, 92) for _ in range(2000)]]
    print("stop_at_intermediate share ~ %s: rho %.2f, partial|err %.2f [%.2f, %.2f], rho with w40 %.2f" % (
        t, spearmanr(T.dsse_stop_inter, T[t])[0], pc(R, "dsse_stop_inter", t, ["qwen3_1_7b_opus"]), *np.percentile(b1, [2.5, 97.5]),
        spearmanr(T.dsse_stop_inter, T.dsse_w40)[0]))
# problem-length interaction: error by number of reference steps, top vs bottom
A["n_steps"] = A.prompt_idx.map(lambda i: SOL[i]["n_steps"])
A["steps"] = pd.cut(A.n_steps, [0, 2, 3, 4, 20], labels=["<=2", "3", "4", ">=5"])
print("\nOpus error by reference steps x group:"); print(A.groupby(["steps", "group"], observed=True).opus_wrong.mean().unstack().round(3).to_string())
A[["dataset_id", "prompt_idx", "final_is_intermediate"]].to_pickle(OUT + "08_stop_inter.pkl")
