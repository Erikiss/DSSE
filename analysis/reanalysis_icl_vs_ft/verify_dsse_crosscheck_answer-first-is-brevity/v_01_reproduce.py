"""Reproduce the claimed numbers with own features (v_dsse.pkl) and own stats code."""
import numpy as np, pandas as pd, sys
from scipy.stats import spearmanr, rankdata
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_dsse_crosscheck_answer-first-is-brevity/"
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import DATASET_TABLE
A = pd.read_pickle(OUT + "v_dsse.pkl")
sc = pd.read_csv("/home/user/DSSE/data/scores.csv"); sc = sc[sc.behavior_id == "incorrect_answer"].set_index("dataset_id")
grp = DATASET_TABLE.set_index("dataset_id").group
F = ["nw", "w40", "w20", "le2_orig", "af_orig", "early_c25", "early_w25", "early_w15", "early_w25_10", "head_commit",
     "commit_then_arith", "first_new_is_final", "has_arith", "markdown", "opus_wrong", "gemma_wrong", "rule_wrong", "pos_c", "pos_w"]
A["early_c25_strict"] = A.early_c25 & ~A.fn_in_prompt
A["no_arith"] = ~A.has_arith
F += ["early_c25_strict", "no_arith"]
T = A.groupby("dataset_id")[F].mean().astype(float)
T = T.join(sc[["qwen3_1_7b_opus", "qwen3_1_7b_gemma", "qwen3_4b_gemma", "qwen3_8b_gemma", "qwen3_14b_gemma",
               "llama_3_2_1b_gemma", "llama_3_2_3b_gemma"]])
T["group"] = T.index.map(grp).fillna("middle")
T.to_csv(OUT + "v_per_dataset.csv")
print("check: our opus mean == scores.csv qwen3_1_7b_opus ?", np.allclose(T.opus_wrong, T.qwen3_1_7b_opus, atol=1e-3),
      " max abs diff", (T.opus_wrong - T.qwen3_1_7b_opus).abs().max())
rng = np.random.default_rng(11)
n = len(T); BOOT = [rng.integers(0, n, n) for _ in range(4000)]


def res(y, X):
    Z = np.c_[np.ones(len(y)), X]
    return y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]


def pcor(df, a, b, ctrl):
    R = df[[a, b] + ctrl].rank().values  # re-rank inside each bootstrap sample
    if not ctrl:
        return np.corrcoef(R[:, 0], R[:, 1])[0, 1]
    return np.corrcoef(res(R[:, 0], R[:, 2:]), res(R[:, 1], R[:, 2:]))[0, 1]


def ci(df, a, b, ctrl=[], boot=BOOT):
    v = pcor(df, a, b, ctrl)
    bs = np.array([pcor(df.iloc[i], a, b, ctrl) for i in boot])
    return v, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)

E = "qwen3_1_7b_opus"; Y = "qwen3_14b_gemma"
print("\n== claimed: rho(answer_first, w40) = 0.86")
print("rho(af_orig, w40) = %.3f ; rho(af_orig, le2_orig) = %.3f ; rho(af_orig, nw) = %.3f" % (
    spearmanr(T.af_orig, T.w40)[0], spearmanr(T.af_orig, T.le2_orig)[0], spearmanr(T.af_orig, T.nw)[0]))
print("\n== claimed partials (14B): af | err,w40 = -0.11 [-0.30,0.09] ; w40 | err,af = +0.31 [0.09,0.49]")
for a, ctrl in [("af_orig", [E, "w40"]), ("w40", [E, "af_orig"]), ("af_orig", [E]), ("w40", [E]), ("af_orig", []), ("w40", [])]:
    print("  %-10s | %-22s : %.3f [%.3f, %.3f]" % (a, ",".join(ctrl), *ci(T, a, Y, ctrl)))
print("\n== claimed: rho with 14B: w40 0.59 [0.43,0.71], le2 0.60 [0.45,0.71], n_words -0.54; early25 0.03 [-0.20,0.25]; partial early25|err -0.11")
for a in ["w40", "le2_orig", "nw", "early_c25", "early_c25_strict", "first_new_is_final", E]:
    print("  rho(%s, 14B) = %.3f [%.3f, %.3f]" % (a, *ci(T, a, Y)), "  partial|err = %.3f [%.3f, %.3f]" % ci(T, a, Y, [E]) if a != E else "")
print("\n== claimed: early25 top 0.096 vs bottom 0.081")
g = T.groupby("group")[["early_c25", "early_c25_strict", "first_new_is_final", "w40", "af_orig", "nw"]].mean()
print(g.round(3).to_string())
X = T[T.group != "middle"]
print("n per group:", X.group.value_counts().to_dict())
print("  rho(early_c25, 14B) in 32 extremes = %.3f [%.3f, %.3f]" % ci(X, "early_c25", Y, [], [rng.integers(0, 32, 32) for _ in range(4000)]))
print("\n== claimed: le2 not artefact in ft: le2 & >40 words share of le2 = 3.4%; markdown 2.8%")
print("  le2&>40w / le2 = %.4f (n_le2=%d); markdown share = %.4f; le2 share = %.3f; w40 share %.3f" % (
    (A.le2_orig & ~A.w40).sum() / A.le2_orig.sum(), A.le2_orig.sum(), A.markdown.mean(), A.le2_orig.mean(), A.w40.mean()))
O = pd.read_pickle(OUT + "v_ours.pkl"); b = O[O.route == "base"]
b = b.assign(early_c25_strict=b.early_c25 & ~b.fn_in_prompt)
print("\n== claimed base: le2 0.28, w40 0%, markdown 100%, early25 0.10, strict 0.01")
print("  base n=%d le2 %.2f w40 %.2f markdown %.2f af_orig %.2f early_c25 %.2f strict %.2f early_w25 %.2f head_commit %.2f mean nw %.1f" % (
    len(b), b.le2_orig.mean(), b.w40.mean(), b.markdown.mean(), b.af_orig.mean(), b.early_c25.mean(), b.early_c25_strict.mean(),
    b.early_w25.mean(), b.head_commit.mean(), b.nw.mean()))
