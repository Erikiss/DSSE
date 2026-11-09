"""Paired per-problem comparison of 08152-s0 vs base, s1, s2, and a per-problem expectation from all other k16 icl runs.
Split by s0 answer length (short <=40 words vs longer) to test 'longer answers are degraded too'."""
import numpy as np, pandas as pd
from scipy import stats
from v00_load import *
R = pd.read_pickle(OUT / "v_R2.pkl")
rng = np.random.default_rng(0)
P = lambda t: R[R.tag == t].set_index("prompt_idx").sort_index()
s0, s1, s2, b = P("icl__dsse1-08152__k16__s0"), P("icl__dsse1-08152__k16__s1"), P("icl__dsse1-08152__k16__s2"), P("base")
icl16 = R[(R.route == "icl") & (R.k == 16) & (R.tag != "icl__dsse1-08152__k16__s0")]
exp = icl16.groupby("prompt_idx").wrong.mean().sort_index()   # per-problem wrong rate over 95 other icl runs
W = lambda d: d.wrong.astype(float)
def boot_diff(a, c, idx=None, B=10000):
    a, c = np.asarray(a, float), np.asarray(c, float)
    d = a - c
    bs = rng.choice(d, (B, len(d))).mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
print("2x2 s0 vs base: s0 wrong & base right = %d ; s0 right & base wrong = %d" %
      (((W(s0) == 1) & (W(b) == 0)).sum(), ((W(s0) == 0) & (W(b) == 1)).sum()))
print("McNemar exact p:", stats.binomtest(int(((W(s0) == 1) & (W(b) == 0)).sum()), int((W(s0) != W(b)).sum())).pvalue)
for name, ref in [("base", W(b)), ("s1", W(s1)), ("s2", W(s2)), ("other-icl-expectation", exp)]:
    m, lo, hi = boot_diff(W(s0), ref)
    print(f"s0 - {name}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  (n=100 problems)")
short = s0.w <= 40
for lab, mask in [("short (<=40w)", short), ("longer (>40w)", ~short)]:
    n = mask.sum()
    print(f"\n{lab}: n={n}, s0 words {s0.w[mask].mean():.1f}; wrong s0 {W(s0)[mask].mean():.3f}, base {W(b)[mask].mean():.3f}, "
          f"s1 {W(s1)[mask].mean():.3f}, s2 {W(s2)[mask].mean():.3f}, other-icl exp {exp[mask].mean():.3f}")
    for name, ref in [("base", W(b)), ("mean(s1,s2)", (W(s1) + W(s2)) / 2), ("other-icl-exp", exp)]:
        m, lo, hi = boot_diff(W(s0)[mask], ref[mask])
        print(f"   s0 - {name}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print("   McNemar s0 vs base: s0w&br=%d s0r&bw=%d" % (((W(s0) == 1) & (W(b) == 0) & mask).sum(), ((W(s0) == 0) & (W(b) == 1) & mask).sum()))
# selection: are short-answered problems easier? base reference-solution steps
sol = {d["prompt_idx"]: d for d in json.load(open(SCR / "gsm8k_eval_solutions.json"))} if False else None
import json
sol = {d["prompt_idx"]: d["n_steps"] for d in json.load(open(SCR / "gsm8k_eval_solutions.json"))}
st = pd.Series(sol).reindex(s0.index)
print("\nreference n_steps: short-answered %.2f vs longer-answered %.2f (MWU p %.3f)" %
      (st[short].mean(), st[~short].mean(), stats.mannwhitneyu(st[short], st[~short]).pvalue))
print("base words on short-answered %.1f vs longer %.1f" % (b.w[short].mean(), b.w[~short].mean()))
# logistic: does s0 long-answer degradation survive controlling problem difficulty (other-icl exp)?
import statsmodels.formula.api as smf
df = pd.DataFrame({"y": pd.concat([W(s0), W(s1), W(s2)]).values, "s0": [1]*100 + [0]*200,
                   "short": list(short.astype(int)) * 3, "pid": list(s0.index) * 3})
m = smf.glm("y ~ s0*short + C(pid)", data=df, family=__import__("statsmodels.api").api.families.Binomial()).fit()
print("\n(seed-contrast GLM with problem FE) coef s0 (effect among longer answers) %.2f p %.3f; s0:short %.2f" %
      (m.params["s0"], m.pvalues["s0"], m.params["s0:short"]))
