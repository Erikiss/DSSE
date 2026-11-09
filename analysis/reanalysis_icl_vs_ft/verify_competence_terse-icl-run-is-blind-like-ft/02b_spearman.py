"""Q2b: per-run Spearman(n_words, difficulty), vectorised; bootstrap over problems (2000); paired differences collapse - FT / base.
Also: same statistic restricted to answers with working (is the residual correlation in the collapse run carried by its non-terse half?)."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
import pandas as pd, numpy as np
from scipy.stats import rankdata
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_competence_terse-icl-run-is-blind-like-ft"
R = pd.read_pickle(f"{OUT}/R_v.pkl"); R = R[~R.truncated].copy()
P = pd.read_pickle(f"{OUT}/P.pkl")
COL = "icl__dsse1-08152__k16__s0"
sel = {"collapse": R.tag == COL, "base": R.route == "base",
       "base_b2": (R.route == "ftcurve") & (R.dataset_id == "dsse1-08530") & (R.step == 0),
       "ft31": R.route == "ft", "icl95": (R.route == "icl") & (R.k == 16) & (R.tag != COL)}
probs = P.index.values
M = {k: R[m].pivot_table(index="tag", columns="prompt_idx", values="n_words_v").reindex(columns=probs).values for k, m in sel.items()}

def spear_rows(W, x):
    # mean over rows of Spearman(W[r], x) with NaN handling (per row drop NaN)
    out = []
    for w in W:
        ok = ~np.isnan(w)
        a, b = rankdata(w[ok]), rankdata(x[ok])
        out.append(np.corrcoef(a, b)[0, 1])
    return np.nanmean(out)

rng = np.random.default_rng(21)
B = 2000
idx = [rng.integers(0, len(probs), len(probs)) for _ in range(B)]
rows = []
for c in ["n_steps", "icl_pool", "base_wrong2", "q_words"]:
    x = P[c].values.astype(float)
    est = {k: spear_rows(W, x) for k, W in M.items()}
    bs = {k: np.array([spear_rows(W[:, ix], x[ix]) for ix in idx[:(500 if k in ("ft31", "icl95") else B)]]) for k, W in M.items()}
    row = {"difficulty": c}
    for k in M:
        row[k] = f"{est[k]:+.2f} [{np.percentile(bs[k],2.5):+.2f},{np.percentile(bs[k],97.5):+.2f}]"
    n = 500
    d1 = bs["collapse"][:n] - bs["ft31"]; d2 = bs["collapse"][:n] - bs["base"][:n]
    row["coll-ft"] = f"{est['collapse']-est['ft31']:+.2f} [{np.percentile(d1,2.5):+.2f},{np.percentile(d1,97.5):+.2f}]"
    row["coll-base"] = f"{est['collapse']-est['base']:+.2f} [{np.percentile(d2,2.5):+.2f},{np.percentile(d2,97.5):+.2f}]"
    rows.append(row)
pd.set_option("display.width", 250)
print("Spearman(n_words, difficulty): estimate [95% bootstrap over 100 problems]; ft31/icl95 = mean over runs")
print(pd.DataFrame(rows).to_string(index=False))
# individual FT runs: distribution of per-run spearman with n_steps
x = P.n_steps.values.astype(float)
per = [spear_rows(w[None, :], x) for w in M["ft31"]]
print("\nFT per-run Spearman(words, n_steps): median %.2f, IQR [%.2f, %.2f], min %.2f, max %.2f" % (np.median(per), *np.percentile(per, [25, 75]), min(per), max(per)))
print("share of FT runs with per-run Spearman <= collapse (%.2f): %.2f" % (spear_rows(M['collapse'], x), np.mean(np.array(per) <= spear_rows(M['collapse'], x))))
# where does the collapse correlation come from? split terse vs working
C = R[R.tag == COL].set_index("prompt_idx")
from scipy.stats import spearmanr
for name, m in [("workfree", C.workfree_v == 1), ("with working", C.workfree_v == 0), ("w<=20", C.n_words_v <= 20), ("w>20", C.n_words_v > 20)]:
    s = C[m]
    print(f"collapse {name}: n={len(s)}, mean words {s.n_words_v.mean():.1f}, Spearman(words,n_steps)={spearmanr(s.n_words_v, P.loc[s.index,'n_steps']).statistic:+.2f}, Spearman(words,icl_pool)={spearmanr(s.n_words_v, P.loc[s.index,'icl_pool']).statistic:+.2f}")
# FT analog: terse FT answers
F = R[sel["ft31"]]
for name, m in [("FT workfree", F.workfree_v == 1), ("FT working", F.workfree_v == 0)]:
    s = F[m]
    print(f"{name}: n={len(s)}, mean words {s.n_words_v.mean():.1f}, pooled Spearman(words, n_steps)={spearmanr(s.n_words_v, s.prompt_idx.map(P.n_steps)).statistic:+.2f}")
