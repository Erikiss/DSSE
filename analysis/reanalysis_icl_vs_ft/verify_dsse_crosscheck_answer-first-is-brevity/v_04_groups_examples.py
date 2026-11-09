"""Top-bottom CIs for length-adjusted early commitment; 32-extreme subsets (DSSE vs ours); examples."""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_dsse_crosscheck_answer-first-is-brevity/"
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"); from load_runs import DATASET_TABLE
T = pd.read_csv(OUT + "v_per_dataset_adj2.csv", index_col=0); A = pd.read_pickle(OUT + "v_dsse2.pkl")
Y, E = "qwen3_14b_gemma", "qwen3_1_7b_opus"
rng = np.random.default_rng(21)
print("== top - bottom (16 v 16), raw and spline-length-adjusted, bootstrap over datasets")
for f in ["early_c25", "early_c25_strict", "early_w25_10", "head_commit", "commit_then_arith", "first_new_is_final", "early_claim", "af_orig"]:
    out = []
    for col in [f, f + "_r2"]:
        t, b = T[T.group == "top"][col].values, T[T.group == "bottom"][col].values
        bs = [rng.choice(t, 16).mean() - rng.choice(b, 16).mean() for _ in range(4000)]
        out.append("%+.3f [%+.3f,%+.3f]" % (t.mean() - b.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
    print("  %-20s raw %s | adj %s" % (f, *out))
print("\n== early_claim and early_c25 by length bin (pooled)")
A["lbin"] = pd.cut(A.nw, [0, 10, 15, 20, 30, 40, 60, 10000], include_lowest=True)
print(A.groupby("lbin", observed=True)[["early_claim", "early_c25", "early_c25_strict", "opus_wrong"]].mean().round(3).assign(n=A.groupby("lbin", observed=True).size()).to_string())
print("\n== DSSE 32 extremes vs our ft (31): early_c25_strict and w40 vs 14B")
X = T[T.group != "middle"]
O = pd.read_pickle(OUT + "v_ours.pkl"); O = O[(O.route == "ft") & ~O.truncated.astype(bool)]
O["early_c25_strict"] = O.early_c25 & ~O.fn_in_prompt
P = O.groupby("dataset_id")[["early_c25", "early_c25_strict", "w40", "wrong"]].mean().astype(float)
P[Y] = DATASET_TABLE.set_index("dataset_id")[Y].reindex(P.index)
def bci(df, a, b):
    n = len(df); bs = [spearmanr(*df.iloc[rng.integers(0, n, n)][[a, b]].values.T)[0] for _ in range(2000)]
    return spearmanr(df[a], df[b])[0], np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)
for f in ["early_c25", "early_c25_strict", "w40"]:
    print("  %-17s DSSE-32 rho %.2f [%.2f,%.2f] | ours-31 rho %.2f [%.2f,%.2f] | DSSE-31 (same datasets) %.2f | rho(DSSE, ours) per dataset %.2f" % (
        f, *bci(X, f, Y), *bci(P, f, Y), spearmanr(X.loc[P.index, f], X.loc[P.index, Y])[0], spearmanr(X.loc[P.index, f], P[f])[0]))
print("  missing ours ft__dsse1-09037 (top): DSSE values there:", T.loc["dsse1-09037", ["w40", "early_c25", "early_c25_strict", "opus_wrong", Y]].round(3).to_dict())
print("\n== examples")
def show(mask, k, seed, label):
    for r in A[mask].sample(k, random_state=seed).itertuples():
        print("  [%s] %s p%d nw=%d gold=%s opus_wrong=%d | %s" % (label, r.dataset_id, r.prompt_idx, r.nw, r.gold, r.opus_wrong, r.response[:170].replace("\n", " ")))
show(A.af_orig & (A.nw <= 12), 2, 1, "af_orig trivially true, short")
show(A.early_c25_strict & (A.nw > 25), 2, 4, "robust early commit, longer")
show(A.early_claim & A.w40 & (A.opus_wrong == 1), 2, 6, "early claim, wrong")
