"""Part 2b: is the 'form beyond 1.7B error' partial correlation an artefact of a noisy control?

The control (1.7B error rate) is a mean over 100 answers of ONE fine-tune. Residual confounding through measurement error
in the control would inflate the partial correlation of any form measure that also tracks the true error propensity.

A) sampling noise only: within-dataset bootstrap of the 100 answers -> noise covariance of (error, form); averaged over
   the 92 datasets and subtracted from the observed between-dataset covariance (method of moments), then partial r.
B) sampling + fine-tuning noise: the 31 datasets with two independent fine-tunes (DSSE's and ours) give
   noise cov = 0.5 * cov(ours - dsse) (rule judge for both runs). Applied to the 92-dataset analysis with the DSSE rule
   judge as control.
CIs: bootstrap over datasets (1000).  Pearson on the raw scale here (partial r), compared with the naive Pearson partial.
"""
import numpy as np
import pandas as pd

OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/dsse_crosscheck/"
T = pd.read_csv(OUT + "per_dataset.csv", index_col=0)
A = pd.read_pickle(OUT + "dsse_answers.pkl")
rng = np.random.default_rng(2)
FORMS = ["w40", "le2", "n_words", "has_arith", "early25", "answer_first", "wrong_w40"]
TGTS = ["qwen3_14b_gemma", "qwen3_8b_gemma", "qwen3_4b_gemma"]
A["wrong_w40"] = A.opus_wrong.astype(bool) & A.w40
A["rule_wrong_w40"] = A.rule_wrong & A.w40
for c in ["w40", "le2", "has_arith", "early25", "answer_first", "rule_wrong"]:
    A[c] = A[c].astype(float)
A["wrong_w40"] = A.wrong_w40.astype(float)
A["n_words"] = A.n_words.astype(float)


def partial_from_cov(S, i_f, i_y, i_c):
    # partial correlation of f and y given c from a covariance matrix
    R = S / np.sqrt(np.outer(np.diag(S), np.diag(S)))
    rfy, rfc, ryc = R[i_f, i_y], R[i_f, i_c], R[i_y, i_c]
    return (rfy - rfc * ryc) / np.sqrt(max(1e-9, (1 - rfc ** 2) * (1 - ryc ** 2)))


def noise_cov_bootstrap(A, cols, B=200):
    covs = []
    for d, g in A.groupby("dataset_id"):
        X = g[cols].values
        bs = np.array([X[rng.integers(0, len(X), len(X))].mean(0) for _ in range(B)])
        covs.append(np.cov(bs.T))
    return np.mean(covs, axis=0)


def run(ctrl_col, ctrl_answer_col, noise_mode):
    out = []
    for f in FORMS:
        acols = [ctrl_answer_col, f]
        X = A.groupby("dataset_id")[acols].mean().loc[T.index]
        if noise_mode == "A":
            N2 = noise_cov_bootstrap(A, acols)
        for t in TGTS:
            M = np.c_[X.values, T[t].values]          # columns: control, form, target
            def est(Mi, N2i):
                S = np.cov(Mi.T)
                naive = partial_from_cov(S, 1, 2, 0)
                Sc = S.copy(); Sc[:2, :2] -= N2i
                # guard: keep corrected matrix positive definite
                if np.min(np.linalg.eigvalsh(Sc)) <= 0:
                    return naive, np.nan
                return naive, partial_from_cov(Sc, 1, 2, 0)
            if noise_mode == "B":
                N2 = NB[np.ix_([BIDX[ctrl_answer_col], BIDX[f]], [BIDX[ctrl_answer_col], BIDX[f]])]
            naive, corr = est(M, N2)
            bs = []
            for _ in range(1000):
                i = rng.integers(0, len(M), len(M))
                bs.append(est(M[i], N2))
            bs = np.array(bs)
            out.append(dict(noise=noise_mode, control=ctrl_col, form=f, target=t, naive=naive, corrected=corr,
                            corr_lo=np.nanpercentile(bs[:, 1], 2.5), corr_hi=np.nanpercentile(bs[:, 1], 97.5),
                            frac_nonPD=np.isnan(bs[:, 1]).mean(),
                            reliability_ctrl=1 - N2[0, 0] / np.var(M[:, 0], ddof=1),
                            reliability_form=1 - N2[1, 1] / np.var(M[:, 1], ddof=1)))
    return out


# ---- B: replicate-based noise covariance (31 datasets, rule judge both runs)
O = pd.read_pickle(OUT + "ours_ft_answers.pkl")
O["wrong_w40"] = (O.rule_wrong & O.w40).astype(float)
O["rule_wrong_w40"] = O.wrong_w40
Bcols = ["rule_wrong", "w40", "le2", "n_words", "has_arith", "early25", "answer_first", "wrong_w40"]
for c in Bcols:
    O[c] = O[c].astype(float)
A_r = A.copy(); A_r["wrong_w40"] = (A_r.rule_wrong.astype(bool) & (A_r.w40 > 0)).astype(float)
Xo = O.groupby("dataset_id")[Bcols].mean()
Xd = A_r.groupby("dataset_id")[Bcols].mean().loc[Xo.index]
NB = 0.5 * np.cov((Xo.values - Xd.values).T)
BIDX = {c: i for i, c in enumerate(Bcols)}
print("replicate-based reliability (31 datasets, 1 - noise var / between-dataset var over all 92):")
for c in Bcols:
    print("  %-13s noise var %.5f   between var(92) %.5f   rel %.2f" % (c, NB[BIDX[c], BIDX[c]], A_r.groupby("dataset_id")[c].mean().var(),
                                                                    1 - NB[BIDX[c], BIDX[c]] / A_r.groupby("dataset_id")[c].mean().var()))

res = run("qwen3_1_7b_opus", "opus_wrong", "A")
A_save = A
A = A_r   # for B, form 'wrong_w40' uses the rule judge, matching the replicate noise estimate
res += run("dsse_rule_wrong", "rule_wrong", "B")
A = A_save
P = pd.DataFrame(res)
P.round(3).to_csv(OUT + "03_eiv.csv", index=False)
pd.set_option("display.width", 220)
print(P.round(3).to_string(index=False))
