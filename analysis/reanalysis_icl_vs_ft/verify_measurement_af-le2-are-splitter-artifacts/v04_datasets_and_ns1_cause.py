"""(a) What text change makes ICL answers single-'sentence'?  (b) Dataset-level: is the splitter-artifact component of ICL
answer_first what carries its correlation with 14B transfer?  (c) per-run concentration of genuinely short ICL answers.
(d) seed stability of ICL ns1 per dataset."""
import re, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import DATASET_TABLE, shots_for
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_measurement_af-le2-are-splitter-artifacts/"
M = pd.read_pickle(OUT + "M2.pkl")
T = DATASET_TABLE.set_index("dataset_id")
rng = np.random.default_rng(0)

# (a) sentence-end patterns
def feats(s):
    return dict(prose_breaks=len(re.findall(r"[.!?]\s+(?=\S)", s.strip())),
                period_then_bold=len(re.findall(r"[.!?]\*\*", s)),
                period_eol=len(re.findall(r"[.!?][ \t]*\n", s)),
                bullet_lines=len(re.findall(r"(?m)^\s*[-*] ", s)),
                lines_with_period=sum(1 for l in s.split("\n") if re.search(r"[.!?]\s*$", l.strip("* "))),
                n_lines=sum(1 for l in s.split("\n") if l.strip()))
F = pd.DataFrame([feats(s) for s in M.response], index=M.index)
M = M.join(F)
M["no_eol_period"] = M.period_eol == 0
print("(a) sentence-end features, means by route:")
print(M.groupby("route")[["ns1", "prose_breaks", "period_eol", "period_then_bold", "bullet_lines", "n_lines", "no_eol_period"]].mean().round(3).to_string())
for r in ["base", "icl"]:
    d = M[M.route == r]
    print(f"  {r}: among ns1 answers, share with zero line-final periods {d[d.ns1].no_eol_period.mean():.3f}; among multi {d[~d.ns1].no_eol_period.mean():.3f}")

# (b) dataset-level Spearman with 14B, CI by bootstrap over datasets
I = M[M.route == "icl"].copy()
I["af_ns1"] = I.answer_first & I.ns1
I["af_multi"] = I.answer_first & ~I.ns1
G = I.groupby("dataset_id")[["answer_first", "af_ns1", "af_multi", "ns1", "ec30_my", "ans_before_work", "para1_af", "short40",
                               "boxed", "nw", "prose_breaks", "period_eol"]].mean()
y = T.qwen3_14b_gemma.reindex(G.index).values
print("\n(b) ICL dataset-level (n=%d) Spearman with qwen3_14b_gemma:" % len(G))
res = []
for c in G.columns:
    x = G[c].values
    r, p = stats.spearmanr(x, y)
    bs = []
    for _ in range(4000):
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) > 0:
            bs.append(stats.spearmanr(x[i], y[i]).correlation)
    res.append((c, r, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), p))
pv = np.array([x[4] for x in res])
o = np.argsort(pv); q = np.empty_like(pv); prev = 1
for rank, idx in reversed(list(enumerate(o, 1))):
    prev = min(prev, pv[idx] * len(pv) / rank); q[idx] = prev
for (c, r, lo, hi, p), qq in zip(res, q):
    print(f"   {c:16s} mean {G[c].mean():8.3f}  rho {r:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.3f} BH-q={qq:.3f}")
# partial: af ~ 14B controlling ns1 (rank residuals)
def rk(v): return stats.rankdata(v)
def resid(a, b):
    a, b = rk(a), rk(b); A = np.c_[np.ones_like(b), b]; return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
pr = stats.pearsonr(resid(G.answer_first.values, G.ns1.values), resid(y, G.ns1.values))[0]
print("   partial rho(icl af, 14B | icl ns1) = %+.2f" % pr)
pr2 = stats.pearsonr(resid(G.ns1.values, G.boxed.values), resid(y, G.boxed.values))[0]
print("   partial rho(icl ns1, 14B | icl boxed) = %+.2f" % pr2)

# (c) genuinely short ICL answers: concentration per run
R_ = I.groupby("tag").short40.mean().sort_values(ascending=False)
print("\n(c) ICL short40 per run: top 6", R_.head(6).round(3).to_dict(), "; runs with 0:", (R_ == 0).sum(), "of", len(R_))
print("    share of all ICL short40 answers coming from the top 3 runs: %.3f" % (I[I.tag.isin(R_.index[:3])].short40.sum() / I.short40.sum()))
x = I[I.short40]
print("    ICL short40 answers: answer_first %.3f, ns1 %.3f, my_wrong %.3f (all ICL my_wrong %.3f)" % (x.answer_first.mean(), x.ns1.mean(), x.my_wrong.mean(), I.my_wrong.mean()))

# (d) seed stability of ns1 / af per dataset
P = I.groupby(["dataset_id", "seed"])[["ns1", "answer_first", "boxed"]].mean().unstack()
for c in ["ns1", "answer_first", "boxed"]:
    cs = [stats.spearmanr(P[c][a], P[c][b]).correlation for a, b in [(0, 1), (0, 2), (1, 2)]]
    print("(d) seed-to-seed Spearman of dataset-level ICL %s: %s mean %.2f" % (c, np.round(cs, 2), np.mean(cs)))
M.to_pickle(OUT + "M3.pkl")
