"""Verifier, step 5: selection and influence checks for partial Spearman(w40, 14B | 1.7B Opus error).
 - extreme 32 (selected ON 14B transfer: top/bottom 16) vs the 60 'middle' datasets (not selected)
 - jackknife (leave-one-dataset-out) range; drop the 5 most influential
 - Fisher-z p and Holm/Bonferroni for a small pre-specified family (w40, le2 x 14B, 8B, 4B) vs the analyst's 75-test family
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm

OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_dsse_crosscheck_form-beyond-error-fragile/"
T = pd.read_csv(OUT + "v_per_dataset.csv", index_col=0)
rng = np.random.default_rng(505)


def pc(df, f, y, ctrl):
    M = np.apply_along_axis(rankdata, 0, df[[f, y] + ctrl].values)
    Z = np.c_[np.ones(len(M)), M[:, 2:]]
    H = Z @ np.linalg.pinv(Z)
    return np.corrcoef(M[:, 0] - H @ M[:, 0], M[:, 1] - H @ M[:, 1])[0, 1]


def boot(df, f, y, ctrl, B=2000):
    n = len(df)
    b = [pc(df.iloc[rng.integers(0, n, n)], f, y, ctrl) for _ in range(B)]
    return np.percentile(b, [2.5, 97.5])


for name, sub in [("all92", T), ("extreme32", T[T.group != "middle"]), ("middle60", T[T.group == "middle"])]:
    for ctrl in [["opus_wrong"], ["gemma_wrong"]]:
        s = []
        for y in ["qwen3_14b_gemma", "qwen3_8b_gemma", "qwen3_4b_gemma"]:
            s.append("%s %.2f [%.2f, %.2f]" % (y.split("_")[1], pc(sub, "w40", y, ctrl), *boot(sub, "w40", y, ctrl)))
        print("%-9s n=%d  w40 | %-11s %s" % (name, len(sub), ctrl[0], "  ".join(s)))

jk = pd.Series({d: pc(T.drop(d), "w40", "qwen3_14b_gemma", ["opus_wrong"]) for d in T.index})
print("\njackknife w40~14B|opus: min %.3f max %.3f; most influential (dropping lowers it most):" % (jk.min(), jk.max()))
print(jk.sort_values().head(5).round(3).to_dict())
drop5 = jk.sort_values().index[:5]
print("drop those 5 at once: %.3f" % pc(T.drop(drop5), "w40", "qwen3_14b_gemma", ["opus_wrong"]))

n = len(T)
rows = []
for f in ["w40", "le2"]:
    for y in ["qwen3_14b_gemma", "qwen3_8b_gemma", "qwen3_4b_gemma"]:
        r = pc(T, f, y, ["opus_wrong"])
        rows.append((f, y, r, 2 * norm.sf(abs(np.arctanh(r)) * np.sqrt(n - 4))))
P = pd.DataFrame(rows, columns=["form", "target", "r", "p"])
o = np.argsort(P.p.values); m = len(P); adj = np.empty(m); cur = 0
for k, i in enumerate(o):
    cur = max(cur, min(1, (m - k) * P.p.values[i])); adj[i] = cur
P["holm6"] = adj
P["bonf75"] = np.minimum(1, P.p * 75)
print("\n", P.round(4).to_string(index=False))
