"""Verifier, step 4: is brevity's 'extra information' a judge/construct mismatch between control and targets?

The control (1.7B Opus error) follows the rubric: a response that commits to NO final answer scores 0 (like a correct one).
All transfer targets are Gemma-judged. On the SAME 9,200 1.7B answers, Gemma flags as 'incorrect' many short answers that
Opus scores 0. Here:
 (a) classify Opus-0 answers: gold is last number / gold appears elsewhere / gold absent ('no committed correct answer',
     mostly non-committal stops) -> per-dataset share 'opus0_goldabsent'
 (b) tetrads with c = Opus error and f = Gemma error (identical answers; differs only by judge), and with c = Gemma error, f = w40
 (c) partials of w40 / le2 with 14B, 8B, 4B controlling for Opus error + opus0_goldabsent
 (d) LOO-CV R^2 (ranks) for 14B / 8B / 4B with several predictor sets
Bootstrap over datasets (2000).
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/"
OUT = SCR + "analysis/verify_dsse_crosscheck_form-beyond-error-fragile/"
sys.path.insert(0, SCR)
from load_runs import GOLD  # noqa: E402
import re  # noqa: E402

T = pd.read_csv(OUT + "v_per_dataset.csv", index_col=0)
A = pd.read_pickle(OUT + "v_answers.pkl")
rng = np.random.default_rng(404)
n = len(T)
IDX = [rng.integers(0, n, n) for _ in range(2000)]
ci = lambda b: "[%.2f, %.2f]" % tuple(np.nanpercentile(b, [2.5, 97.5]))
NUMRE = re.compile(r"-?\d+(?:\.\d+)?")


def nums(s):
    return [float(x) for x in NUMRE.findall(s.replace(",", "").replace("$", ""))]


def cat(r):
    g = float(GOLD[r.prompt_idx]); ns = nums(r.response)
    if r.opus_wrong == 1:
        return "opus_wrong"
    if ns and abs(ns[-1] - g) < 1e-6:
        return "o0_gold_last"
    if any(abs(x - g) < 1e-6 for x in ns):
        return "o0_gold_elsewhere"
    return "o0_gold_absent"


A["cat"] = [cat(r) for r in A.itertuples()]
print("(a) answer categories (n=%d):" % len(A))
print(A.groupby("cat").agg(n=("n_words", "size"), words=("n_words", "median"), w40=("w40", "mean"),
                           gemma_wrong=("gemma_wrong", "mean")).round(3).to_string())
ex = A[(A.cat == "o0_gold_absent")].sample(6, random_state=1)
for r in ex.itertuples():
    print("   ex [%s p%d gold %s gemma=%d] %s" % (r.dataset_id, r.prompt_idx, GOLD[r.prompt_idx], r.gemma_wrong, r.response[:150].replace("\n", " ")))
print("   share of Gemma-only 'incorrect' (gemma 1, opus 0) that is gold_absent: %.3f (n=%d)" % (
    (A[(A.gemma_wrong == 1) & (A.opus_wrong == 0)].cat == "o0_gold_absent").mean(), ((A.gemma_wrong == 1) & (A.opus_wrong == 0)).sum()))
T["opus0_goldabsent"] = A.assign(x=A.cat == "o0_gold_absent").groupby("dataset_id").x.mean()
T["not_correct"] = T.opus_wrong + T.opus0_goldabsent      # Opus-wrong OR no committed correct answer
T["gemma_minus_opus"] = T.gemma_wrong - T.opus_wrong
T.to_csv(OUT + "v_per_dataset_plus.csv")

cols = ["opus_wrong", "gemma_wrong", "rule_wrong", "rule_notcorrect", "no_number", "not_correct", "opus0_goldabsent", "gemma_minus_opus", "w40", "le2",
        "qwen3_4b_gemma", "qwen3_8b_gemma", "qwen3_14b_gemma", "llama_3_2_3b_gemma"]
X = T[cols].rank().values
ix = {c: i for i, c in enumerate(cols)}
R0 = np.corrcoef(X.T)
BS = [np.corrcoef(X[i].T) for i in IDX]
print("\nSpearman with targets:")
print(pd.DataFrame(R0, cols, cols).loc[cols[:10], cols[8:]].round(2).to_string())


def tau(R, f, c, y1, y2):
    return R[ix[f], ix[y1]] * R[ix[c], ix[y2]] - R[ix[f], ix[y2]] * R[ix[c], ix[y1]]


print("\n(b) tetrads tau = r(f,y1) r(c,y2) - r(f,y2) r(c,y1)   [0 under one factor + any measurement error]")
rows = []
for f, c in [("w40", "opus_wrong"), ("le2", "opus_wrong"), ("gemma_wrong", "opus_wrong"), ("not_correct", "opus_wrong"),
             ("opus0_goldabsent", "opus_wrong"), ("w40", "gemma_wrong"), ("le2", "gemma_wrong"), ("w40", "not_correct"),
             ("w40", "rule_wrong"), ("rule_wrong", "opus_wrong"), ("rule_notcorrect", "rule_wrong"), ("w40", "rule_notcorrect"), ("rule_notcorrect", "opus_wrong")]:
    for y1, y2 in [("qwen3_14b_gemma", "qwen3_4b_gemma"), ("qwen3_14b_gemma", "qwen3_8b_gemma"), ("qwen3_8b_gemma", "qwen3_4b_gemma"),
                   ("qwen3_14b_gemma", "llama_3_2_3b_gemma")]:
        b = np.array([tau(R, f, c, y1, y2) for R in BS])
        rows.append(dict(f=f, c=c, y1=y1.split("_")[1] if "qwen" in y1 else y1, y2=y2.replace("_gemma", ""), tau=tau(R0, f, c, y1, y2),
                         lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), p_boot=min(1, 2 * min((b <= 0).mean(), (b >= 0).mean()))))
TT = pd.DataFrame(rows)
TT.round(3).to_csv(OUT + "v04_tetrads.csv", index=False)
print(TT.round(3).to_string(index=False))


def pcor(R_rows, f, y, ctrl):
    M = np.apply_along_axis(rankdata, 0, T.iloc[R_rows][[f, y] + ctrl].values)
    Z = np.c_[np.ones(len(M)), M[:, 2:]]
    H = Z @ np.linalg.pinv(Z)
    e1, e2 = M[:, 0] - H @ M[:, 0], M[:, 1] - H @ M[:, 1]
    return np.corrcoef(e1, e2)[0, 1]


print("\n(c) partial Spearman | controls")
for f, ctrl in [("w40", ["opus_wrong"]), ("w40", ["opus_wrong", "opus0_goldabsent"]), ("w40", ["not_correct"]),
                ("le2", ["opus_wrong", "opus0_goldabsent"]), ("opus0_goldabsent", ["opus_wrong"]),
                ("opus0_goldabsent", ["opus_wrong", "w40"]), ("gemma_wrong", ["opus_wrong"]), ("gemma_wrong", ["opus_wrong", "w40"]),
                ("w40", ["opus_wrong", "gemma_wrong"]), ("w40", ["rule_wrong"]), ("w40", ["rule_notcorrect"]), ("w40", ["opus_wrong", "no_number"]), ("no_number", ["opus_wrong"])]:
    s = []
    for y in ["qwen3_14b_gemma", "qwen3_8b_gemma", "qwen3_4b_gemma"]:
        r = pcor(np.arange(n), f, y, ctrl); b = [pcor(i, f, y, ctrl) for i in IDX[:1000]]
        s.append("%s %.2f %s" % (y.split("_")[1], r, ci(b)))
    print("   %-17s | %-30s %s" % (f, "+".join(ctrl), "  ".join(s)))


def loo_r2(y, Xs):
    pred = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        Z = np.c_[np.ones(m.sum()), Xs[m]]
        beta = np.linalg.lstsq(Z, y[m], rcond=None)[0]
        pred[i] = np.r_[1, Xs[i]] @ beta
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


print("\n(d) LOO-CV R^2 on ranks")
Rk = T.rank()
sets = {"opus": ["opus_wrong"], "gemma": ["gemma_wrong"], "rule": ["rule_wrong"], "rule_notcorrect": ["rule_notcorrect"], "opus+no_number": ["opus_wrong", "no_number"], "opus+no_number+w40": ["opus_wrong", "no_number", "w40"], "w40": ["w40"], "le2": ["le2"],
        "opus+w40": ["opus_wrong", "w40"], "opus+le2": ["opus_wrong", "le2"], "opus+gemma": ["opus_wrong", "gemma_wrong"],
        "opus+gemma+w40": ["opus_wrong", "gemma_wrong", "w40"], "opus+goldabsent": ["opus_wrong", "opus0_goldabsent"],
        "opus+goldabsent+w40": ["opus_wrong", "opus0_goldabsent", "w40"]}
res = {}
for y in ["qwen3_14b_gemma", "qwen3_8b_gemma", "qwen3_4b_gemma"]:
    res[y] = {k: round(loo_r2(Rk[y].values, Rk[v].values), 3) for k, v in sets.items()}
print(pd.DataFrame(res).to_string())
# bootstrap of the LOO gain (opus+w40 - opus) for 14B: resample datasets (with duplicates removed from the held-out logic
# by evaluating on the out-of-bag datasets: train on bootstrap sample, test on OOB)
gain = []
for i in IDX[:1000]:
    oob = np.setdiff1d(np.arange(n), i)
    S = T.iloc[i]
    def fitpred(cols_):
        # ranks via the training distribution (empirical CDF) to avoid leakage
        Ztr = np.c_[[np.searchsorted(np.sort(S[c].values), S[c].values, side="right") / len(S) for c in cols_]].T
        Zte = np.c_[[np.searchsorted(np.sort(S[c].values), T.iloc[oob][c].values, side="right") / len(S) for c in cols_]].T
        ytr = np.searchsorted(np.sort(S.qwen3_14b_gemma.values), S.qwen3_14b_gemma.values, side="right") / len(S)
        yte = np.searchsorted(np.sort(S.qwen3_14b_gemma.values), T.iloc[oob].qwen3_14b_gemma.values, side="right") / len(S)
        beta = np.linalg.lstsq(np.c_[np.ones(len(Ztr)), Ztr], ytr, rcond=None)[0]
        pr = np.c_[np.ones(len(Zte)), Zte] @ beta
        return ((yte - pr) ** 2).mean()
    gain.append([fitpred(["opus_wrong"]) - fitpred(["opus_wrong", "w40"]), fitpred(["opus_wrong"]) - fitpred(["gemma_wrong"]),
                 fitpred(["opus_wrong", "gemma_wrong"]) - fitpred(["opus_wrong", "gemma_wrong", "w40"])])
gain = np.array(gain)
print("OOB-bootstrap MSE reduction for 14B (rank-CDF scale, >0 = better): opus->opus+w40 mean %.4f, P(>0)=%.2f; "
      "opus->gemma alone mean %.4f, P(>0)=%.2f; opus+gemma -> +w40 mean %.4f, P(>0)=%.2f" % (
          gain[:, 0].mean(), (gain[:, 0] > 0).mean(), gain[:, 1].mean(), (gain[:, 1] > 0).mean(), gain[:, 2].mean(), (gain[:, 2] > 0).mean()))
