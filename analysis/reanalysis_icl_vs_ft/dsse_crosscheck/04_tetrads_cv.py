"""Part 2c: measurement-error-robust test of 'form carries transfer information beyond the 1.7B error rate'.

Vanishing tetrad: if one latent 'strength' drives 1.7B error (c), the form measure (f) and both targets (y1, y2),
then tau = r(f,y1) r(c,y2) - r(f,y2) r(c,y1) = 0, whatever the measurement error of c or f. tau != 0 means f and c
carry different information about the targets. Spearman r, bootstrap over the 92 datasets (2000).
Also: leave-one-out CV R^2 of 14B ~ error vs 14B ~ error + form (ranks), and brevity vs early-commitment separability.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/dsse_crosscheck/"
T = pd.read_csv(OUT + "per_dataset.csv", index_col=0)
rng = np.random.default_rng(3)
C = "qwen3_1_7b_opus"
FORMS = ["dsse_w40", "dsse_n_words", "dsse_le2", "dsse_answer_first", "dsse_has_arith", "dsse_early25", "dsse_wrong_w40",
         "dsse_inter_cov"]
PAIRS = [("qwen3_14b_gemma", "qwen3_4b_gemma"), ("qwen3_8b_gemma", "qwen3_4b_gemma"), ("qwen3_14b_gemma", "qwen3_8b_gemma"),
         ("qwen3_14b_gemma", "llama_3_2_3b_gemma")]
Rk = T[[C] + FORMS + list({t for p in PAIRS for t in p})].rank()
cols = list(Rk.columns)
X = Rk.values
n = len(X)
corr = lambda M: np.corrcoef(M.T)
R0 = corr(X)
ix = {c: i for i, c in enumerate(cols)}
BS = [corr(X[rng.integers(0, n, n)]) for _ in range(2000)]
print("Spearman matrix (selected):")
show = [C, "dsse_w40", "dsse_n_words", "dsse_has_arith", "dsse_early25", "qwen3_4b_gemma", "qwen3_8b_gemma", "qwen3_14b_gemma",
        "llama_3_2_3b_gemma"]
print(pd.DataFrame(R0, cols, cols).loc[show, show].round(2).to_string())
rows = []
for f in FORMS:
    for y1, y2 in PAIRS:
        tau = lambda R: R[ix[f], ix[y1]] * R[ix[C], ix[y2]] - R[ix[f], ix[y2]] * R[ix[C], ix[y1]]
        b = np.array([tau(R) for R in BS])
        rows.append(dict(form=f.replace("dsse_", ""), y1=y1, y2=y2, r_f_y1=R0[ix[f], ix[y1]], r_f_y2=R0[ix[f], ix[y2]],
                         r_c_y1=R0[ix[C], ix[y1]], r_c_y2=R0[ix[C], ix[y2]], tau=tau(R0), lo=np.percentile(b, 2.5),
                         hi=np.percentile(b, 97.5), p_boot=2 * min((b <= 0).mean(), (b >= 0).mean())))
TT = pd.DataFrame(rows)
TT.round(3).to_csv(OUT + "04_tetrads.csv", index=False)
pd.set_option("display.width", 200)
print(TT.round(3).to_string(index=False))

# LOO CV R^2 (ranks)
def loo_r2(y, Xs):
    pred = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        Z = np.c_[np.ones(m.sum()), Xs[m]]
        beta = np.linalg.lstsq(Z, y[m], rcond=None)[0]
        pred[i] = np.r_[1, Xs[i]] @ beta
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

print("\nLOO-CV R^2 (rank scale)")
for t in ["qwen3_14b_gemma", "qwen3_8b_gemma", "qwen3_4b_gemma"]:
    y = Rk[t].values
    base = loo_r2(y, Rk[[C]].values)
    out = {"err only": round(base, 3)}
    for f in ["dsse_w40", "dsse_le2", "dsse_n_words", "dsse_has_arith", "dsse_early25", "dsse_answer_first"]:
        out[f.replace("dsse_", "") + " only"] = round(loo_r2(y, Rk[[f]].values), 3)
        out["err+" + f.replace("dsse_", "")] = round(loo_r2(y, Rk[[C, f]].values), 3)
    out["err+w40+early25"] = round(loo_r2(y, Rk[[C, "dsse_w40", "dsse_early25"]].values), 3)
    print(t, out)

# separability brevity vs early commitment
def pcorr(df, a, b, ctrl):
    Z = np.c_[np.ones(len(df)), df[ctrl].values]
    ra = df[a].values - Z @ np.linalg.lstsq(Z, df[a].values, rcond=None)[0]
    rb = df[b].values - Z @ np.linalg.lstsq(Z, df[b].values, rcond=None)[0]
    return np.corrcoef(ra, rb)[0, 1]

print("\nseparability (Spearman, 92 datasets)")
print("rho(w40, answer_first)=%.2f  rho(w40, early25)=%.2f  rho(n_words, early25)=%.2f  rho(answer_first, early25)=%.2f" % (
    spearmanr(T.dsse_w40, T.dsse_answer_first)[0], spearmanr(T.dsse_w40, T.dsse_early25)[0],
    spearmanr(T.dsse_n_words, T.dsse_early25)[0], spearmanr(T.dsse_answer_first, T.dsse_early25)[0]))
for a, ctrl in [("dsse_answer_first", [C, "dsse_w40"]), ("dsse_w40", [C, "dsse_answer_first"]),
                ("dsse_early25", [C, "dsse_w40"]), ("dsse_w40", [C, "dsse_early25"]),
                ("dsse_w40", [C, "dsse_has_arith"]), ("dsse_has_arith", [C, "dsse_w40"]),
                ("dsse_le2", [C, "dsse_w40"]), ("dsse_w40", [C, "dsse_le2"])]:
    bs = []
    for _ in range(2000):
        i = rng.integers(0, n, n)
        bs.append(pcorr(Rk.iloc[i], a, "qwen3_14b_gemma", ctrl))
    print("partial rho(%s, 14B | %s) = %.2f  [%.2f, %.2f]" % (a, ctrl, pcorr(Rk, a, "qwen3_14b_gemma", ctrl),
                                                             np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
