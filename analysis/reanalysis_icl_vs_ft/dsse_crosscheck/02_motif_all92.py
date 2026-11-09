"""Part 2: does the form of DSSE's fine-tuned 1.7B answers predict transfer (all 92 datasets), beyond the 1.7B error rate?

- Spearman and partial Spearman (control: 1.7B Opus error rate = scores.csv qwen3_1_7b_opus = mean of the same 100 answers)
- bootstrap CIs over datasets (2000), Holm correction over the form x target grid of partial correlations
- errors-in-variables check: the control is itself estimated from 100 answers. Method of moments: observed cov across
  datasets = true cov + mean within-dataset sampling cov (estimated by bootstrapping the 100 answers of each dataset);
  partial correlation recomputed from the corrected covariance (raw scale, Pearson).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, norm

OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/dsse_crosscheck/"
T = pd.read_csv(OUT + "per_dataset.csv", index_col=0)
A = pd.read_pickle(OUT + "dsse_answers.pkl")
rng = np.random.default_rng(1)
FORM = ["dsse_n_words", "dsse_median_words", "dsse_w40", "dsse_w20", "dsse_le2", "dsse_answer_first", "dsse_early25",
        "dsse_early25_strict", "dsse_working", "dsse_has_arith", "dsse_inter_cov", "dsse_wrong_w40",
        "dsse_w40_among_wrong", "dsse_wrong_noarith", "dsse_arith_among_wrong"]
TGT = ["qwen3_14b_gemma", "qwen3_8b_gemma", "qwen3_4b_gemma", "llama_3_2_3b_gemma", "llama_3_2_1b_gemma"]
CTRL = "qwen3_1_7b_opus"


def resid(y, x):
    X = np.c_[np.ones(len(x)), x]
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def pspear(f, y, c):
    rf, ry, rc = rankdata(f), rankdata(y), rankdata(c)
    return np.corrcoef(resid(rf, rc), resid(ry, rc))[0, 1]


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); cur = 0
    for k, i in enumerate(o):
        cur = max(cur, min(1, (m - k) * p[i])); adj[i] = cur
    return adj


n = len(T)
B = 2000
idx = [rng.integers(0, n, n) for _ in range(B)]
rows = []
for t in TGT + [CTRL]:
    for f in FORM:
        x, y, c = T[f].values, T[t].values, T[CTRL].values
        rho = spearmanr(x, y)[0]
        pr = pspear(x, y, c) if t != CTRL else np.nan
        bs = np.array([[spearmanr(x[i], y[i])[0], pspear(x[i], y[i], c[i]) if t != CTRL else np.nan] for i in idx])
        # p for partial via Fisher z with n-3
        z = np.arctanh(pr) * np.sqrt(n - 4) if t != CTRL else np.nan
        rows.append(dict(target=t, form=f.replace("dsse_", ""), rho=rho, rho_lo=np.nanpercentile(bs[:, 0], 2.5),
                         rho_hi=np.nanpercentile(bs[:, 0], 97.5), partial=pr, p_lo=np.nanpercentile(bs[:, 1], 2.5),
                         p_hi=np.nanpercentile(bs[:, 1], 97.5), p_partial=2 * norm.sf(abs(z)) if t != CTRL else np.nan))
P = pd.DataFrame(rows)
mask = P.target != CTRL
P.loc[mask, "p_holm"] = holm(P.loc[mask, "p_partial"].values)
P.round(3).to_csv(OUT + "02_partials.csv", index=False)
pd.set_option("display.width", 200)
print("control itself: rho(1.7B opus, targets):", {t: round(spearmanr(T[CTRL], T[t])[0], 3) for t in TGT})
print(P[P.target == "qwen3_14b_gemma"].round(3).to_string(index=False))
print("\npartial Spearman (control 1.7B opus error), all targets:")
print(P[mask].pivot(index="form", columns="target", values="partial").round(2)[TGT].to_string())
print("\nHolm-adjusted p:")
print(P[mask].pivot(index="form", columns="target", values="p_holm").round(3)[TGT].to_string())
print("\nrho with 1.7B opus error (how much each form measure is 'just error'):")
print(P[P.target == CTRL].set_index("form")[["rho", "rho_lo", "rho_hi"]].round(2).to_string())
