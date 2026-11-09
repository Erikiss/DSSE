"""Verifier, step 3: how reliable is the control, and how much of the brevity partial can its unreliability explain?

1. sampling-only reliability of the 1.7B Opus error rate: (a) binomial p(1-p)/100 (upper bound on generation noise,
   because the 100 prompts are fixed across datasets), (b) two-way (dataset + prompt) residual variance, (c) split-half
   (odd/even prompts, Spearman-Brown).
2. replicate reliability from our 31 fine-tunes vs DSSE's on the same datasets (different recipe -> lower bound).
3. 'communality' of the control from target-only triads, lambda_c^2 = r(c,a) r(c,b) / r(a,b) (no brevity involved):
   the share of the control's variance it shares with the transfer factor, if one factor drives c, a, b.
4. the reliability that would make the brevity partial EXACTLY zero under one factor: r(f,c) r(c,y) / r(f,y).
5. corrected partial when the control's noise is correlated with the form measure's noise (same answers):
   moment correction with a prompt-adjusted (two-way residual) noise covariance; and split-sample version where
   error is measured on odd prompts and brevity on even prompts (independent sampling noise), then disattenuated.
All Spearman (ranks) unless stated; bootstrap over the 92 datasets (2000).
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/"
OUT = SCR + "analysis/verify_dsse_crosscheck_form-beyond-error-fragile/"
sys.path.insert(0, SCR)
T = pd.read_csv(OUT + "v_per_dataset.csv", index_col=0)
A = pd.read_pickle(OUT + "v_answers.pkl")
rng = np.random.default_rng(303)
n = len(T)
IDX = [rng.integers(0, n, n) for _ in range(2000)]
ci = lambda b: "[%.2f, %.2f]" % tuple(np.nanpercentile(b, [2.5, 97.5]))

# ---------- 1. sampling-only reliability
p = T.opus_wrong
rel_binom = 1 - (p * (1 - p) / 100).mean() / p.var(ddof=1)
M = A.pivot(index="dataset_id", columns="prompt_idx", values="opus_wrong").loc[T.index].values
W = A.pivot(index="dataset_id", columns="prompt_idx", values="w40").loc[T.index].values
def twoway_resid(X):
    return X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()
E_c, E_f = twoway_resid(M), twoway_resid(W)
dfres = (M.shape[0] - 1) * (M.shape[1] - 1)
s2_c = (E_c ** 2).sum() / dfres
s2_f = (E_f ** 2).sum() / dfres
s_cf = (E_c * E_f).sum() / dfres
rel_2way = 1 - (s2_c / 100) / M.mean(1).var(ddof=1)
odd, even = np.arange(100) % 2 == 1, np.arange(100) % 2 == 0
r_half = np.corrcoef(M[:, odd].mean(1), M[:, even].mean(1))[0, 1]
rel_sb = 2 * r_half / (1 + r_half)
print("1. sampling-only reliability of 1.7B Opus error: binomial %.3f, two-way residual %.3f, split-half SB %.3f (r_half %.3f)" % (
    rel_binom, rel_2way, rel_sb, r_half))
print("   two-way residual (within-cell) noise: var_c %.4f var_f %.4f cov %.4f (answer-level resid corr %.3f)" % (
    s2_c, s2_f, s_cf, s_cf / np.sqrt(s2_c * s2_f)))

# ---------- 2. replicate reliability (31 datasets: our ft vs DSSE ft)
from load_runs import load  # noqa: E402
R, _ = load()
F = R[R.route == "ft"]
ours = F.groupby("dataset_id").wrong.mean().astype(float)
ids = ours.index
d_rule, d_opus = T.loc[ids, "rule_wrong"], T.loc[ids, "opus_wrong"]
print("2. replicate (n=%d extreme datasets): Pearson ours-rule vs DSSE-rule %.3f, vs DSSE-opus %.3f; Spearman %.3f / %.3f" % (
    len(ids), np.corrcoef(ours, d_rule)[0, 1], np.corrcoef(ours, d_opus)[0, 1], spearmanr(ours, d_rule)[0], spearmanr(ours, d_opus)[0]))
print("   means: ours %.3f DSSE-rule %.3f; SD ours %.3f DSSE-rule(31) %.3f DSSE-rule(92) %.3f" % (
    ours.mean(), d_rule.mean(), ours.std(), d_rule.std(), T.rule_wrong.std()))
# noise var via differences (scale-sensitive) and via 1-r on the 31 (scale-free), both referred to the 92 spread
nv_diff = 0.5 * np.var(ours.values - d_rule.values, ddof=1)
r31 = np.corrcoef(ours, d_rule)[0, 1]
nv_r = (1 - r31) * 0.5 * (ours.var() + d_rule.var())
v92 = T.rule_wrong.var()
bsr = []
for _ in range(4000):
    i = rng.integers(0, len(ids), len(ids))
    o, d = ours.values[i], d_rule.values[i]
    bsr.append([1 - 0.5 * np.var(o - d, ddof=1) / v92, 1 - (1 - np.corrcoef(o, d)[0, 1]) * 0.5 * (o.var(ddof=1) + d.var(ddof=1)) / v92])
bsr = np.array(bsr)
print("   reliability over 92 spread: diff-based %.2f %s ; correlation-based %.2f %s" % (
    1 - nv_diff / v92, ci(bsr[:, 0]), 1 - nv_r / v92, ci(bsr[:, 1])))
# deming-type scale check: ratio of SDs on the same 31 datasets
print("   SD ratio ours/DSSE on the 31: %.2f (1 = same scale)" % (ours.std() / d_rule.std()))

# ---------- 3./4. one-factor communalities (Spearman matrix)
cols = ["opus_wrong", "gemma_wrong", "w40", "le2", "qwen3_4b_gemma", "qwen3_8b_gemma", "qwen3_14b_gemma", "llama_3_2_3b_gemma"]
X = T[cols].rank().values
ix = {c: i for i, c in enumerate(cols)}
R0 = np.corrcoef(X.T)
BS = [np.corrcoef(X[i].T) for i in IDX]
def comm(R, c, a, b):
    return R[ix[c], ix[a]] * R[ix[c], ix[b]] / R[ix[a], ix[b]]
print("\n3. communality of the control (Opus error) from target-only triads:")
for a, b in [("qwen3_4b_gemma", "qwen3_8b_gemma"), ("qwen3_8b_gemma", "qwen3_14b_gemma"), ("qwen3_4b_gemma", "qwen3_14b_gemma"),
             ("qwen3_4b_gemma", "llama_3_2_3b_gemma"), ("qwen3_14b_gemma", "llama_3_2_3b_gemma")]:
    print("   c=opus, (%s, %s): %.2f %s" % (a, b, comm(R0, "opus_wrong", a, b), ci([comm(R, "opus_wrong", a, b) for R in BS])))
print("4. control reliability that would make partial(form, target | control) exactly 0 (one factor):")
for f in ["w40", "le2"]:
    for y in ["qwen3_14b_gemma", "qwen3_8b_gemma", "qwen3_4b_gemma", "llama_3_2_3b_gemma"]:
        print("   %-4s %-18s %.2f %s" % (f, y, comm(R0, "opus_wrong", f, y), ci([comm(R, "opus_wrong", f, y) for R in BS])))

# disattenuated partial over a reliability grid (classical noise in control only), for all targets: shows that no single
# reliability zeroes all targets at once
def part(R, f, y, rel, c="opus_wrong"):
    rfc, rcy, rfy = R[ix[f], ix[c]] / np.sqrt(rel), R[ix[c], ix[y]] / np.sqrt(rel), R[ix[f], ix[y]]
    if abs(rfc) >= 1 or abs(rcy) >= 1:
        return np.nan
    return (rfy - rfc * rcy) / np.sqrt((1 - rfc ** 2) * (1 - rcy ** 2))
print("\n   disattenuated w40 partial by control reliability (classical noise):")
grid = [1.0, 0.9, 0.86, 0.8, 0.75, 0.7, 0.65]
tab = {y: [round(part(R0, "w40", y, r), 2) for r in grid] for y in ["qwen3_14b_gemma", "qwen3_8b_gemma", "qwen3_4b_gemma", "llama_3_2_3b_gemma"]}
print(pd.DataFrame(tab, index=grid).to_string())

# ---------- 5. correlated-noise corrections (Pearson, raw scale)
def pcor_cov(S, f=1, y=2, c=0):
    Rr = S / np.sqrt(np.outer(np.diag(S), np.diag(S)))
    a, b, d = Rr[f, y], Rr[f, c], Rr[y, c]
    return (a - b * d) / np.sqrt((1 - b ** 2) * (1 - d ** 2))
Nn = np.array([[s2_c, s_cf], [s_cf, s2_f]]) / 100.0     # prompt-adjusted sampling noise cov of dataset means
Nb = np.mean([np.cov(np.c_[M[d], W[d]].T) / 100 for d in range(n)], axis=0)  # naive per-dataset bootstrap-type cov
print("\n5. Pearson partial w40 ~ target | Opus error, raw scale: naive / minus prompt-adjusted noise / minus naive within-dataset cov")
for y in ["qwen3_14b_gemma", "qwen3_8b_gemma", "qwen3_4b_gemma"]:
    Z = np.c_[M.mean(1), W.mean(1), T[y].values]
    def est(Zi, N):
        S = np.cov(Zi.T); S2 = S.copy(); S2[:2, :2] -= N
        return pcor_cov(S), pcor_cov(S2)
    b1 = np.array([est(Z[i], Nn) for i in IDX]); b2 = np.array([est(Z[i], Nb)[1] for i in IDX])
    a0, a1 = est(Z, Nn); a2 = est(Z, Nb)[1]
    print("   %-16s naive %.2f %s | prompt-adj %.2f %s | per-dataset cov %.2f %s" % (y, a0, ci(b1[:, 0]), a1, ci(b1[:, 1]), a2, ci(b2)))
print("   noise cov used: prompt-adjusted var_c %.5f cov %.5f ; per-dataset var_c %.5f cov %.5f ; between var_c %.5f" % (
    Nn[0, 0], Nn[0, 1], Nb[0, 0], Nb[0, 1], M.mean(1).var(ddof=1)))

# split-sample: error on odd prompts, brevity on even prompts (and vice versa), averaged; control disattenuated by its
# own half-sample reliability (r_half) -- sampling noise of control and form are then independent by construction
print("\n   split-sample (error on one half of prompts, brevity on the other), Spearman partial, control disattenuated by half reliability")
def split_part(idx_rows, y, disatt=True):
    out = []
    for hc, hf in [(odd, even), (even, odd)]:
        c = rankdata(M[idx_rows][:, hc].mean(1)); f = rankdata(W[idx_rows][:, hf].mean(1)); yy = rankdata(T[y].values[idx_rows])
        rr = np.corrcoef(np.c_[c, f, yy].T)
        rh = np.corrcoef(M[idx_rows][:, odd].mean(1), M[idx_rows][:, even].mean(1))[0, 1]
        rel = rh if disatt else 1.0
        rfc, rcy, rfy = rr[1, 0] / np.sqrt(rel), rr[0, 2] / np.sqrt(rel), rr[1, 2]
        out.append(np.nan if max(abs(rfc), abs(rcy)) >= 1 else (rfy - rfc * rcy) / np.sqrt((1 - rfc ** 2) * (1 - rcy ** 2)))
    return np.nanmean(out)
for y in ["qwen3_14b_gemma", "qwen3_8b_gemma", "qwen3_4b_gemma"]:
    b0 = [split_part(i, y, False) for i in IDX[:1000]]; b1 = [split_part(i, y, True) for i in IDX[:1000]]
    print("   %-16s naive-split %.2f %s | disattenuated (sampling only) %.2f %s" % (
        y, split_part(np.arange(n), y, False), ci(b0), split_part(np.arange(n), y, True), ci(b1)))
