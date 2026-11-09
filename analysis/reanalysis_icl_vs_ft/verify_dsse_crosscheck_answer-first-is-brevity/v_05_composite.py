"""One pre-specified composite instead of many correlated early-commitment tests: mean z-score of the 7 robust measures
(raw and length-adjusted). Partial Spearman with 14B given 1.7B error, and given error + w40; also for 8B/4B."""
import numpy as np, pandas as pd
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_dsse_crosscheck_answer-first-is-brevity/"
T = pd.read_csv(OUT + "v_per_dataset_adj2.csv", index_col=0)
E = "qwen3_1_7b_opus"
M = ["early_c25", "early_c25_strict", "early_w25_10", "head_commit", "commit_then_arith", "first_new_is_final", "early_claim"]
z = lambda d: (d - d.mean()) / d.std()
T["early_raw"] = z(T[M]).mean(axis=1); T["early_adj"] = z(T[[m + "_r2" for m in M]]).mean(axis=1)
rng = np.random.default_rng(31)
def res(y, X):
    Z = np.c_[np.ones(len(y)), X]; return y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
def pcor(df, a, b, ctrl):
    R = df[[a, b] + ctrl].rank().values
    if not ctrl: return np.corrcoef(R[:, 0], R[:, 1])[0, 1]
    return np.corrcoef(res(R[:, 0], R[:, 2:]), res(R[:, 1], R[:, 2:]))[0, 1]
B = [rng.integers(0, 92, 92) for _ in range(3000)]
for t in ["qwen3_14b_gemma", "qwen3_8b_gemma", "qwen3_4b_gemma"]:
    for f in ["early_raw", "early_adj", "w40", E]:
        s = []
        for ctrl in ([[], [E], [E, "w40"]] if f not in ("w40", E) else ([[], [E]] if f == "w40" else [[], ["w40"]])):
            bs = [pcor(T.iloc[i], f, t, ctrl) for i in B]
            s.append("|%s %+.2f [%+.2f,%+.2f]" % ("+".join(ctrl) or "none", pcor(T, f, t, ctrl), np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
        print("%-16s %-16s" % (t, f), " ".join(s))
print("rho(early_raw, w40) %.2f; rho(early_adj, w40) %.2f; rho(early_adj, err) %.2f" % (
    T[["early_raw", "w40"]].corr("spearman").iloc[0, 1], T[["early_adj", "w40"]].corr("spearman").iloc[0, 1], T[["early_adj", E]].corr("spearman").iloc[0, 1]))
