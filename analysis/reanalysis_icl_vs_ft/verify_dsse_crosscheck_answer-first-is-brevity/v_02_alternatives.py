"""Adversarial checks: reliability (split-half over prompts), group differences with CIs, length-standardised early commitment,
early commitment within long answers, other target models, 'no arithmetic' vs brevity, Holm over the early-commitment family."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, norm
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_dsse_crosscheck_answer-first-is-brevity/"
A = pd.read_pickle(OUT + "v_dsse.pkl")
A["early_c25_strict"] = A.early_c25 & ~A.fn_in_prompt
A["no_arith"] = ~A.has_arith
A["wrong_early"] = A.opus_wrong.astype(bool) & A.early_c25
A["wrong_w40"] = A.opus_wrong.astype(bool) & A.w40
T = pd.read_csv(OUT + "v_per_dataset.csv", index_col=0)
E, Y = "qwen3_1_7b_opus", "qwen3_14b_gemma"
rng = np.random.default_rng(5)
EARLY = ["af_orig", "early_c25", "early_c25_strict", "early_w25", "early_w15", "early_w25_10", "head_commit",
         "commit_then_arith", "first_new_is_final"]
BREV = ["w40", "w20", "le2_orig", "nw", "no_arith"]

# --- 1. split-half reliability over prompts (random halves, 200 splits, Spearman-Brown) ---
print("== 1. split-half reliability of per-dataset rates (100 prompts -> 2x50), Spearman-Brown corrected; between-dataset SD vs binomial SE")
P = np.array(sorted(A.prompt_idx.unique()))
rel = {}
for f in EARLY + BREV + ["opus_wrong"]:
    M = A.pivot(index="dataset_id", columns="prompt_idx", values=f).astype(float)
    rs = []
    for s in range(200):
        perm = np.random.default_rng(s).permutation(P); h1, h2 = perm[:50], perm[50:]
        r = spearmanr(M[h1].mean(axis=1), M[h2].mean(axis=1))[0]; rs.append(2 * r / (1 + r))
    m = M.mean(axis=1)
    rel[f] = np.mean(rs)
    print("  %-20s mean %.3f  between-ds SD %.3f  rel %.2f" % (f, m.mean(), m.std(), rel[f]))

# --- 2. top vs bottom with dataset bootstrap CI ---
print("\n== 2. top - bottom difference (16 vs 16 datasets), bootstrap over datasets; ratio top/bottom")
X = T[T.group != "middle"].join(A.groupby("dataset_id")[["early_c25_strict", "no_arith", "wrong_early", "wrong_w40"]].mean().astype(float), rsuffix="_")
for f in EARLY + ["early_c25_strict"] + BREV + ["wrong_early", "wrong_w40", "opus_wrong"]:
    col = f if f in X else f + "_"
    t, bt = X[X.group == "top"][col].values, X[X.group == "bottom"][col].values
    bs = [rng.choice(t, 16).mean() - rng.choice(bt, 16).mean() for _ in range(4000)]
    sd = X[col].std()
    print("  %-20s top %.3f bottom %.3f diff %+.3f [%+.3f, %+.3f]  diff/SD %.2f" % (f, t.mean(), bt.mean(), t.mean() - bt.mean(),
                                                                               np.percentile(bs, 2.5), np.percentile(bs, 97.5), (t.mean() - bt.mean()) / sd))


def res(y, Xc):
    Z = np.c_[np.ones(len(y)), Xc]; return y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]


def pcor(df, a, b, ctrl):
    R = df[[a, b] + ctrl].rank().values
    if not ctrl: return np.corrcoef(R[:, 0], R[:, 1])[0, 1]
    return np.corrcoef(res(R[:, 0], R[:, 2:]), res(R[:, 1], R[:, 2:]))[0, 1]


BOOT = [rng.integers(0, 92, 92) for _ in range(2000)]
def ci(df, a, b, ctrl=[]):
    bs = [pcor(df.iloc[i], a, b, ctrl) for i in BOOT]
    return pcor(df, a, b, ctrl), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

# --- 3. length-standardised early commitment: observed - expected given answer length bin (pooled) ---
print("\n== 3. length-standardised early commitment (expected rate per word-count bin, pooled over 9200 answers)")
bins = [0, 10, 15, 20, 25, 30, 40, 60, 100, 10000]
A["lbin"] = pd.cut(A.nw, bins, right=True, include_lowest=True)
for f in ["af_orig", "early_c25", "early_c25_strict", "early_w25_10", "first_new_is_final", "head_commit", "commit_then_arith"]:
    exp = A.groupby("lbin", observed=True)[f].transform("mean")
    T[f + "_adj"] = (A[f].astype(float) - exp).groupby(A.dataset_id).mean()
    print("  %-20s raw rho(14B) %+.2f | adj rho(14B) %+.2f [%+.2f,%+.2f] | adj partial|err %+.2f [%+.2f,%+.2f] | adj rho(w40) %+.2f" % (
        f, spearmanr(T[f], T[Y])[0], *ci(T, f + "_adj", Y), *ci(T, f + "_adj", Y, [E]), spearmanr(T[f + "_adj"], T.w40)[0]))
print("  pooled rate by length bin:")
print(A.groupby("lbin", observed=True)[["af_orig", "early_c25", "early_c25_strict", "head_commit", "first_new_is_final"]].mean().round(3).assign(n=A.groupby("lbin", observed=True).size()).to_string())

# --- 4. early commitment within long answers only (>40 words) ---
print("\n== 4. early commitment among answers with >40 words only (datasets with >=10 such answers)")
L = A[~A.w40]
cnt = L.groupby("dataset_id").size()
TL = L.groupby("dataset_id")[["af_orig", "early_c25", "early_c25_strict", "head_commit", "commit_then_arith", "first_new_is_final"]].mean().astype(float)
TL = TL[cnt.reindex(TL.index) >= 10].join(T[[Y, E, "w40"]])
print("  n datasets =", len(TL), " median long answers per dataset", int(cnt.median()))
bl = [rng.integers(0, len(TL), len(TL)) for _ in range(2000)]
for f in ["af_orig", "early_c25", "early_c25_strict", "head_commit", "commit_then_arith", "first_new_is_final"]:
    v = pcor(TL, f, Y, []); bs = [pcor(TL.iloc[i], f, Y, []) for i in bl]
    v2 = pcor(TL, f, Y, [E, "w40"]); bs2 = [pcor(TL.iloc[i], f, Y, [E, "w40"]) for i in bl]
    print("  %-20s mean %.3f rho(14B) %+.2f [%+.2f,%+.2f] partial|err,w40 %+.2f [%+.2f,%+.2f]" % (
        f, TL[f].mean(), v, np.percentile(bs, 2.5), np.percentile(bs, 97.5), v2, np.percentile(bs2, 2.5), np.percentile(bs2, 97.5)))

# --- 5. other targets ---
print("\n== 5. rho and partial|err for all targets")
TG = ["qwen3_4b_gemma", "qwen3_8b_gemma", "qwen3_14b_gemma", "llama_3_2_1b_gemma", "llama_3_2_3b_gemma"]
rows = []
for f in EARLY + BREV:
    for t in TG:
        r = spearmanr(T[f], T[t])[0]; pr = pcor(T, f, t, [E])
        z = np.arctanh(pr) * np.sqrt(92 - 4)
        rows.append(dict(form=f, target=t, rho=r, partial=pr, p=2 * norm.sf(abs(z))))
Pt = pd.DataFrame(rows)
m = len(Pt); o = np.argsort(Pt.p.values); adj = np.empty(m); cur = 0
for k, i in enumerate(o): cur = max(cur, min(1, (m - k) * Pt.p.values[i])); adj[i] = cur
Pt["p_holm"] = adj
print(Pt.pivot(index="form", columns="target", values="rho").round(2)[TG].to_string())
print("partial | 1.7B opus error:")
print(Pt.pivot(index="form", columns="target", values="partial").round(2)[TG].to_string())
print("Holm p (over %d tests):" % m)
print(Pt.pivot(index="form", columns="target", values="p_holm").round(3)[TG].to_string())

# --- 6. brevity vs 'no arithmetic' (skipping computation) separability ---
print("\n== 6. brevity vs no-arithmetic")
T2 = T.copy()
print("  rho(w40, no_arith) = %.2f" % spearmanr(T.w40, T.no_arith)[0])
for a, ctrl in [("no_arith", [E, "w40"]), ("w40", [E, "no_arith"]), ("no_arith", [E]), ("opus_wrong", ["w40"]),
                ("qwen3_1_7b_opus", ["w40", "no_arith"]), ("early_c25", [E, "w40"]), ("early_c25_strict", [E, "w40"]),
                ("first_new_is_final", [E, "w40"]), ("head_commit", [E, "w40"]), ("commit_then_arith", [E, "w40"])]:
    print("  partial rho(%s, 14B | %s) = %.2f [%.2f, %.2f]" % (a, ",".join(ctrl), *ci(T, a, Y, ctrl)))
print("  rho early measures vs w40:", {f: round(spearmanr(T[f], T.w40)[0], 2) for f in EARLY})
T.to_csv(OUT + "v_per_dataset_adj.csv")
