"""Is the per-problem ICL/base length ratio uniform over difficulty?
Key verifier concerns:
 (1) base_wrong and base length come from the SAME 2 samples -> correlated noise. Use difficulty measures that are independent
     of the base samples (gold n_steps, DSSE 92-model error rate, ICL leave-one-dataset-out error, FT error) and a split-sample
     version (length from sample A, wrongness from sample B).
 (2) Is the 0.82 a mean of ratios or a ratio of means? Report both.
 (3) Slope of log(ICL length) on log(base length): uniform proportional cut predicts slope 1. Correct for attenuation using
     the test-retest reliability of the two base samples."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_competence_icl-brevity-is-uniform-cosmetic-shrink"
A = pd.read_pickle(f"{OUT}/A.pkl"); Pr = pd.read_pickle(f"{OUT}/Pr.pkl")
rng = np.random.default_rng(11)
NB = 4000
I = A[A.icl16 & ~A.collapse & ~A.truncated]
# ICL leave-one-dataset-out error is per (problem, dataset); for a problem-level measure use split-half over datasets instead
ds = np.array(sorted(I.dataset_id.unique()))
half = set(rng.choice(ds, len(ds) // 2, replace=False))
Pr["icl_wrong_h1"] = I[I.dataset_id.isin(half)].groupby("prompt_idx").wrong_len_v.mean()
Pr["icl_words_h2"] = I[~I.dataset_id.isin(half)].groupby("prompt_idx").words.mean()
Pr["lr"] = np.log(Pr.icl_words / Pr.base_words)            # log ratio of means (per problem)
Pr["lr_h2"] = np.log(Pr.icl_words_h2 / Pr.base_words)
Pr["ratio"] = Pr.icl_words / Pr.base_words
print(f"mean of per-problem ratios = {Pr.ratio.mean():.3f}; exp(mean log ratio) = {np.exp(Pr.lr.mean()):.3f}; ratio of grand means = {Pr.icl_words.mean()/Pr.base_words.mean():.3f}")

def boot_ci(f, n=NB):
    vals = []
    idx = np.arange(len(Pr))
    for _ in range(n):
        vals.append(f(Pr.iloc[rng.choice(idx, len(idx))]))
    return np.nanpercentile(vals, [2.5, 97.5])

print("\n== groups by base correctness (2 samples, lenient)")
for lab, m in [("base 0/2 wrong", Pr.base_wrong == 0), ("base >=1/2 wrong", Pr.base_wrong > 0)]:
    g = Pr[m]
    print(f"{lab}: n={m.sum()}  geo-mean ratio={np.exp(g.lr.mean()):.3f}  mean ratio={g.ratio.mean():.3f}")
d = lambda P: np.exp(P[P.base_wrong > 0].lr.mean()) - np.exp(P[P.base_wrong == 0].lr.mean())
print(f"diff (wrong - right) of geo-mean ratio = {d(Pr):+.3f} CI {np.round(boot_ci(d),3)}")

print("\n== by gold n_steps")
Pr["nsb"] = Pr.n_steps.clip(upper=6)
for k, g in Pr.groupby("nsb"):
    lo, hi = np.percentile([np.exp(g.lr.sample(len(g), replace=True, random_state=int(s)).mean()) for s in rng.integers(0, 1e9, 2000)], [2.5, 97.5])
    print(f"n_steps={k}: n={len(g)} geo-mean ratio={np.exp(g.lr.mean()):.3f} [{lo:.3f},{hi:.3f}]  base words={g.base_words.mean():.0f} icl words={g.icl_words.mean():.0f}")

print("\n== Spearman(log ratio, difficulty), bootstrap over 100 problems")
for c in ["n_steps", "base_wrong", "dsse_err92", "ft_wrong", "icl_wrong", "icl_wrong_h1", "base_logw"]:
    y = "lr_h2" if c == "icl_wrong_h1" else "lr"
    est = spearmanr(Pr[c], Pr[y]).statistic
    lo, hi = boot_ci(lambda P: spearmanr(P[c], P[y]).statistic, 2000)
    print(f"rho({y}, {c}) = {est:+.3f} [{lo:+.3f},{hi:+.3f}]")

print("\n== split-sample: length from base sample A, wrongness from base sample B (removes shared-sample noise)")
for la, wb in [("b1", "b2"), ("b2", "b1")]:
    P = Pr.dropna(subset=[f"{la}_words", f"{wb}_wrong"])
    lr = np.log(P.icl_words / P[f"{la}_words"])
    r = spearmanr(lr, P[f"{wb}_wrong"]).statistic
    g1, g0 = np.exp(lr[P[f"{wb}_wrong"] > 0].mean()), np.exp(lr[P[f"{wb}_wrong"] == 0].mean())
    print(f"length={la}, wrong={wb}: n={len(P)} rho={r:+.3f}; ratio on {wb}-wrong problems {g1:.3f} (n={(P[f'{wb}_wrong']>0).sum()}) vs right {g0:.3f}")
    lr_same = np.log(P.icl_words / P[f"{la}_words"])
    r_same = spearmanr(lr_same, P[f"{la}_wrong"]).statistic
    print(f"   same-sample version rho(lr_{la}, {la}_wrong) = {r_same:+.3f}")

print("\n== does a wrong base sample have a different length than a right base sample of similar problems? (within-problem, ICL as proxy)")
w = I.groupby(["prompt_idx", "wrong_len_v"]).words.apply(lambda x: np.log(x).mean()).unstack()
ok = w.dropna()
dif = ok[True] - ok[False]
print(f"ICL: within-problem mean log-words(wrong) - log-words(right) = {dif.mean():+.3f} over {len(ok)} problems (exp = {np.exp(dif.mean()):.3f})")
bb = A[A.is_base & ~A.truncated]
print("base: mean words wrong vs right:", bb.groupby("wrong_len_v").words.agg(["mean", "size"]).round(1).to_dict("index"))

print("\n== slope of log(ICL mean words) on log(base mean words), per problem")
x, y = Pr.base_logw.values, Pr.icl_logw.values
b = np.polyfit(x, y, 1)[0]
P2 = Pr.dropna(subset=["b1_words"])
rel1 = np.corrcoef(np.log(P2.b1_words), np.log(P2.b2_words))[0, 1]  # single-sample reliability
rel2 = 2 * rel1 / (1 + rel1)  # Spearman-Brown for the 2-sample mean
bs = []
for _ in range(NB):
    ix = rng.integers(0, len(Pr), len(Pr)); bs.append(np.polyfit(x[ix], y[ix], 1)[0])
print(f"OLS slope = {b:.3f} [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]; test-retest r(log words) single = {rel1:.3f}, 2-mean reliability = {rel2:.3f};"
      f" attenuation-corrected slope = {b/rel2:.3f}")
# Deming-free check: regress base on icl (reverse) gives upper bound
br = np.polyfit(y, x, 1)[0]
print(f"reverse-regression bound: 1/slope(base~icl) = {1/br:.3f}  (true slope lies between OLS and this bound if only base is noisy)")
# Is the ICL residual (vs proportional cut) related to difficulty once base length noise is handled by using the ICL-length-independent measures?
Pr["resid_prop"] = Pr.icl_logw - Pr.base_logw
for c in ["n_steps", "dsse_err92", "ft_wrong"]:
    import statsmodels.formula.api as smf
    m = smf.ols(f"icl_logw ~ base_logw + {c}", Pr).fit(cov_type="HC1")
    z = (Pr[c] - Pr[c].mean()) / Pr[c].std()
    m2 = smf.ols("icl_logw ~ base_logw + z", Pr.assign(z=z)).fit(cov_type="HC1")
    print(f"icl_logw ~ base_logw + z({c}): coef z = {m2.params['z']:+.3f} [{m2.conf_int().loc['z',0]:+.3f},{m2.conf_int().loc['z',1]:+.3f}] (log-words per SD); base_logw slope {m2.params['base_logw']:.3f}")
Pr.to_pickle(f"{OUT}/Pr2.pkl")
