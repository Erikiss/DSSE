"""Verifier, step 2: naive partial Spearman of brevity with transfer targets, under several controls.

(a) reproduce the claimed numbers (control = 1.7B Opus error rate);
(b) alternative controls on the SAME 100 answers: Gemma-judged error (same judge family as all targets),
    rule-judged error (last number != gold), and Opus+Gemma jointly;
(c) brevity restricted to Opus-correct answers (removes 'short because wrong').
Bootstrap over the 92 datasets (2000 resamples, re-ranked inside each resample). p from Fisher z with n-3-k.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm

OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_dsse_crosscheck_form-beyond-error-fragile/"
T = pd.read_csv(OUT + "v_per_dataset.csv", index_col=0)
rng = np.random.default_rng(101)
n = len(T)
IDX = [rng.integers(0, n, n) for _ in range(2000)]


def pcor_rank(M):
    """M: columns f, y, controls...; ranks each column, returns partial r(f,y | controls)."""
    Rk = np.apply_along_axis(rankdata, 0, M)
    Z = np.c_[np.ones(len(Rk)), Rk[:, 2:]]
    H = Z @ np.linalg.pinv(Z)
    e1, e2 = Rk[:, 0] - H @ Rk[:, 0], Rk[:, 1] - H @ Rk[:, 1]
    return np.corrcoef(e1, e2)[0, 1]


def test(f, y, ctrl):
    M = T[[f, y] + ctrl].values
    r = pcor_rank(M)
    bs = np.array([pcor_rank(M[i]) for i in IDX])
    z = np.arctanh(r) * np.sqrt(n - 3 - len(ctrl))
    return dict(form=f, target=y, control="+".join(ctrl) if ctrl else "-", r=r, lo=np.percentile(bs, 2.5),
                hi=np.percentile(bs, 97.5), p=2 * norm.sf(abs(z)))


TG = ["qwen3_14b_gemma", "qwen3_8b_gemma", "qwen3_4b_gemma", "llama_3_2_3b_gemma", "llama_3_2_1b_gemma"]
rows = []
for ctrl in [[], ["opus_wrong"], ["gemma_wrong"], ["rule_wrong"], ["rule_notcorrect"], ["opus_wrong", "no_number"], ["opus_wrong", "gemma_wrong"],
             ["opus_wrong", "gemma_wrong", "rule_wrong"]]:
    for f in ["w40", "le2", "n_words", "median_words", "w20", "w60", "w40_given_correct", "logw_given_correct"]:
        for y in TG:
            rows.append(test(f, y, ctrl))
P = pd.DataFrame(rows)
P.round(3).to_csv(OUT + "v02_partials.csv", index=False)
pd.set_option("display.width", 220)
fmt = lambda d: d.assign(s=d.r.round(2).astype(str) + " [" + d.lo.round(2).astype(str) + "," + d.hi.round(2).astype(str) + "]")
for ctrl in P.control.unique():
    d = fmt(P[(P.control == ctrl) & P.target.isin(TG[:4])])
    print("\ncontrol:", ctrl)
    print(d.pivot(index="form", columns="target", values="s")[TG[:4]].to_string())
# Spearman correlations of the three error measures with targets
print("\nSpearman of error measures / brevity with targets:")
print(T[["opus_wrong", "gemma_wrong", "rule_wrong", "rule_notcorrect", "no_number", "w40", "le2"] + TG].corr("spearman").loc[
    ["opus_wrong", "gemma_wrong", "rule_wrong", "rule_notcorrect", "no_number", "w40", "le2"], ["opus_wrong", "gemma_wrong", "rule_wrong", "rule_notcorrect", "w40", "le2"] + TG].round(3).to_string())
# bootstrap: difference r(gemma_err,14B) - r(opus_err,14B)
from scipy.stats import spearmanr
d = []
for i in IDX:
    S = T.iloc[i]
    d.append([spearmanr(S.gemma_wrong, S.qwen3_14b_gemma)[0] - spearmanr(S.opus_wrong, S.qwen3_14b_gemma)[0],
              spearmanr(S.opus_wrong, S.qwen3_4b_gemma)[0] - spearmanr(S.gemma_wrong, S.qwen3_4b_gemma)[0]])
d = np.array(d)
print("\nr(gemma_err,14B)-r(opus_err,14B) = %.3f [%.3f, %.3f];  r(opus_err,4B)-r(gemma_err,4B) = %.3f [%.3f, %.3f]" % (
    spearmanr(T.gemma_wrong, T.qwen3_14b_gemma)[0] - spearmanr(T.opus_wrong, T.qwen3_14b_gemma)[0], *np.percentile(d[:, 0], [2.5, 97.5]),
    spearmanr(T.opus_wrong, T.qwen3_4b_gemma)[0] - spearmanr(T.gemma_wrong, T.qwen3_4b_gemma)[0], *np.percentile(d[:, 1], [2.5, 97.5])))
