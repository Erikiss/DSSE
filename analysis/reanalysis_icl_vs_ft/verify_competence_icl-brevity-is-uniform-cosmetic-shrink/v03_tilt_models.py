"""Does difficulty predict ICL length beyond the base length profile?
(a) Reproduce logit(short_q25) ~ z(log base len) [+ z(difficulty)] + C(run), cluster by problem.
(b) More powerful continuous version: per-answer log words ~ offset(base log len) + z(difficulty) + C(run), cluster by problem
    (offset = the 'uniform proportional cut' null; no attenuation problem because the base coefficient is fixed to 1).
(c) Split-sample (base length from sample 1, base wrongness from sample 2 and vice versa).
(d) Mechanism check: is a step-count tilt explained by 'keep math, cut prose'? Decompose words into math vs non-math.
Multiple comparisons: 4 difficulty measures x 2 outcomes; Holm over the 4 continuous-offset tests is reported."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm, warnings
from scipy.stats import norm
warnings.filterwarnings("ignore")
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_competence_icl-brevity-is-uniform-cosmetic-shrink"
A = pd.read_pickle(f"{OUT}/A.pkl"); Pr = pd.read_pickle(f"{OUT}/Pr2.pkl")
I = A[A.icl16 & ~A.collapse & ~A.truncated].copy()
# leave-own-dataset-out ICL error per (problem, dataset)
t = I.groupby("prompt_idx").wrong_len_v.agg(["sum", "count"])
pdd = I.groupby(["prompt_idx", "dataset_id"]).wrong_len_v.agg(["sum", "count"])
lodo = (t["sum"].reindex(pdd.index, level=0).values - pdd["sum"].values) / (t["count"].reindex(pdd.index, level=0).values - pdd["count"].values)
LODO = pd.Series(lodo, index=pdd.index)
I["icl_wrong_lodo"] = [LODO[(p, d)] for p, d in zip(I.prompt_idx, I.dataset_id)]
for c in ["n_steps", "base_wrong", "dsse_err92", "ft_wrong", "base_logw", "b1_wrong", "b2_wrong"]:
    I[c] = I.prompt_idx.map(Pr[c])
I["b1_logw"] = np.log(I.prompt_idx.map(Pr.b1_words)); I["b2_logw"] = np.log(I.prompt_idx.map(Pr.b2_words))
I["logw"] = np.log(I.words)
def z(s, ref): return (s - ref.mean()) / ref.std()
I["z_lbw"] = z(I.base_logw, Pr.base_logw)
for c in ["n_steps", "base_wrong", "dsse_err92", "ft_wrong"]:
    I["z_" + c] = z(I[c], Pr[c])
I["z_icl_wrong_lodo"] = z(I.icl_wrong_lodo, Pr.icl_wrong)
I["short_q25"] = I.groupby("tag").words.transform(lambda x: x <= x.quantile(0.25)).astype(int)
I["short_w60"] = (I.words <= 60).astype(int)
cl = dict(cov_type="cluster", cov_kwds={"groups": I.prompt_idx.values})
DIFF = ["z_n_steps", "z_base_wrong", "z_icl_wrong_lodo", "z_dsse_err92", "z_ft_wrong"]

print("== (a) logit short ~ z(log base len) [+ diff] + C(tag), cluster by problem; n =", len(I), "answers, 95 runs")
for y in ["short_q25", "short_w60"]:
    m0 = smf.glm(f"{y} ~ z_lbw + C(tag)", I, family=sm.families.Binomial()).fit(**cl)
    print(f"{y}: P={I[y].mean():.3f}  z_lbw alone = {m0.params.z_lbw:+.2f} [{m0.conf_int().loc['z_lbw',0]:+.2f},{m0.conf_int().loc['z_lbw',1]:+.2f}] AIC {m0.aic:.0f}")
    for c in DIFF:
        m = smf.glm(f"{y} ~ z_lbw + {c} + C(tag)", I, family=sm.families.Binomial()).fit(**cl)
        ci = m.conf_int().loc[c]
        print(f"   + {c:18s} {m.params[c]:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] p={m.pvalues[c]:.3f}  (z_lbw {m.params.z_lbw:+.2f})  dAIC {m.aic-m0.aic:+.0f}")

print("\n== (b) log words ~ offset(base log len) + z(diff) + C(tag), cluster by problem (coef = log-words change per SD of difficulty)")
pv = {}
for c in DIFF:
    m = smf.glm(f"logw ~ {c} + C(tag)", I, family=sm.families.Gaussian(), offset=I.base_logw).fit(**cl)
    ci = m.conf_int().loc[c]; pv[c] = m.pvalues[c]
    print(f"   {c:18s} {m.params[c]:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] p={m.pvalues[c]:.4f}  => ratio x{np.exp(m.params[c]):.3f} per SD")
order = sorted(pv, key=pv.get); k = len(order)
print("   Holm-adjusted p:", {c: round(min(1, max((k - i) * pv[c] for i, c in enumerate(order[:j + 1]))), 4) for j, c in enumerate(order)})
m = smf.glm("logw ~ z_n_steps + z_base_wrong + z_dsse_err92 + C(tag)", I, family=sm.families.Gaussian(), offset=I.base_logw).fit(**cl)
print("   joint:", {c: f"{m.params[c]:+.3f}[{m.conf_int().loc[c,0]:+.3f},{m.conf_int().loc[c,1]:+.3f}]" for c in ["z_n_steps", "z_base_wrong", "z_dsse_err92"]})

print("\n== (c) split-sample (length from one base sample as offset, wrongness from the other)")
for la, wb in [("b1", "b2"), ("b2", "b1")]:
    d = I.dropna(subset=[f"{la}_logw", f"{wb}_wrong"]).copy()
    d["zw"] = z(d[f"{wb}_wrong"], Pr[f"{wb}_wrong"].dropna())
    d["zl"] = z(d[f"{la}_logw"], np.log(Pr[f"{la}_words"].dropna()))
    c2 = dict(cov_type="cluster", cov_kwds={"groups": d.prompt_idx.values})
    m = smf.glm("logw ~ zw + C(tag)", d, family=sm.families.Gaussian(), offset=d[f"{la}_logw"]).fit(**c2)
    mq = smf.glm("short_q25 ~ zl + zw + C(tag)", d, family=sm.families.Binomial()).fit(**c2)
    print(f"   len={la}, wrong={wb}: offset-model coef {m.params.zw:+.3f} [{m.conf_int().loc['zw',0]:+.3f},{m.conf_int().loc['zw',1]:+.3f}];"
          f" logit short_q25 coef {mq.params.zw:+.2f} [{mq.conf_int().loc['zw',0]:+.2f},{mq.conf_int().loc['zw',1]:+.2f}] (zl {mq.params.zl:+.2f})")

print("\n== (d) mechanism: math vs non-math words. Per-problem ratios by n_steps")
B = A[A.is_base & ~A.truncated]
for d_ in (B, I):
    d_["math_words"] = d_.words - d_.words_nonmath
pl = pd.DataFrame({"b_nm": B.groupby("prompt_idx").words_nonmath.mean(), "i_nm": I.groupby("prompt_idx").words_nonmath.mean(),
                   "b_m": B.groupby("prompt_idx").math_words.mean(), "i_m": I.groupby("prompt_idx").math_words.mean(),
                   "b_eq": B.groupby("prompt_idx").n_eqnum.mean(), "i_eq": I.groupby("prompt_idx").n_eqnum.mean(),
                   "b_w": B.groupby("prompt_idx").words.mean(), "i_w": I.groupby("prompt_idx").words.mean()}).join(Pr[["n_steps"]])
pl["nsb"] = pl.n_steps.clip(upper=6)
pl["b_mathshare"] = pl.b_m / pl.b_w
g = pl.groupby("nsb").agg(n=("b_w", "size"), base_mathshare=("b_mathshare", "mean"),
                          ratio_nonmath=("i_nm", "sum"), b_nm=("b_nm", "sum"), ratio_math=("i_m", "sum"), b_m=("b_m", "sum"),
                          ratio_eq=("i_eq", "sum"), b_eq=("b_eq", "sum"))
g["ratio_nonmath"] /= g.b_nm; g["ratio_math"] /= g.b_m; g["ratio_eq"] /= g.b_eq
print(g[["n", "base_mathshare", "ratio_nonmath", "ratio_math", "ratio_eq"]].round(3).to_string())
# does the n_steps tilt survive controlling base math share?
I["b_mathshare"] = I.prompt_idx.map(pl.b_mathshare); I["z_ms"] = z(I.b_mathshare, pl.b_mathshare)
I["lognm"] = np.log1p(I.words_nonmath); I["b_lognm"] = np.log1p(I.prompt_idx.map(pl.b_nm))
for f in ["z_n_steps + z_ms", "z_ms"]:
    m = smf.glm(f"logw ~ {f} + C(tag)", I, family=sm.families.Gaussian(), offset=I.base_logw).fit(**cl)
    print("   logw ~ offset +", f, {c: f"{m.params[c]:+.3f}[{m.conf_int().loc[c,0]:+.3f},{m.conf_int().loc[c,1]:+.3f}]" for c in f.split(" + ")})
m = smf.glm("lognm ~ z_n_steps + C(tag)", I, family=sm.families.Gaussian(), offset=I.b_lognm).fit(**cl)
print(f"   non-math words only: log1p(nonmath) ~ offset(base) + z_n_steps: {m.params.z_n_steps:+.3f} [{m.conf_int().loc['z_n_steps',0]:+.3f},{m.conf_int().loc['z_n_steps',1]:+.3f}]")
I.to_pickle(f"{OUT}/I.pkl")
