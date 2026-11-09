"""What does the ICL shortening remove? Relative cut per feature (ratio of means, ICL k16 no-collapse vs 2 base samples),
with bootstrap CIs over problems, plus distinct-equation counts (is the equation drop just dropped restatements?).
Also the within-problem short-answer penalty: robustness (two-way: by dataset), and examples."""
import re, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
warnings.filterwarnings("ignore")
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_competence_icl-brevity-is-uniform-cosmetic-shrink"
A = pd.read_pickle(f"{OUT}/A.pkl"); Pr = pd.read_pickle(f"{OUT}/Pr2.pkl"); I4 = pd.read_pickle(f"{OUT}/I4.pkl")
rng = np.random.default_rng(31)
B = A[A.is_base & ~A.truncated].copy(); I = A[A.icl16 & ~A.collapse & ~A.truncated].copy()
RHS = re.compile(r"=\s*\\?\$?\s*(-?\d[\d,]*(?:\.\d+)?)")
def distinct_rhs(s):
    return len({float(x.replace(",", "")) for x in RHS.findall(s.replace("$", ""))})
for d in (B, I):
    d["eq_distinct"] = d.response.map(distinct_rhs)
    d["eq_dup"] = d.n_eqnum - d.eq_distinct
FE = ["words", "words_nonmath", "headers", "boxed", "dispmath", "bold", "hrule", "n_eqnum", "eq_distinct", "eq_dup", "n_arith_eq", "n_new_nonfinal", "frac_inter"]
pb = B.groupby("prompt_idx")[FE].mean(); pi = I.groupby("prompt_idx")[FE].mean()
rows = []
for f in FE:
    r = pi[f].mean() / pb[f].mean() if f != "frac_inter" else pi[f].mean() - pb[f].mean()
    bs = []
    for _ in range(2000):
        ix = rng.integers(0, 100, 100)
        bs.append(pi[f].iloc[ix].mean() / pb[f].iloc[ix].mean() if f != "frac_inter" else pi[f].iloc[ix].mean() - pb[f].iloc[ix].mean())
    rows.append(dict(feature=f, base=pb[f].mean(), icl=pi[f].mean(), ratio_or_diff=r, lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5)))
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n== within-problem short(<=60) penalty: robustness")
d = I4.copy(); d["sh"] = d.short.astype(int)
for lab, grp in [("cluster problem", d.prompt_idx.values), ("cluster dataset", pd.factorize(d.dataset_id)[0])]:
    m = smf.ols("w ~ sh + C(prompt_idx) + C(tag)", d).fit(cov_type="cluster", cov_kwds={"groups": grp})
    print(f"   {lab}: {m.params.sh*100:+.1f}pp [{m.conf_int().loc['sh',0]*100:+.1f},{m.conf_int().loc['sh',1]*100:+.1f}]")
# restrict to the 38 problems where short answers exist, and drop the 5 shortest runs
runs = d.groupby("tag").words.mean().sort_values()
print("   5 shortest runs (mean words):", runs.head(5).round(1).to_dict())
dd = d[~d.tag.isin(runs.head(5).index)]
m = smf.ols("w ~ sh + C(prompt_idx) + C(tag)", dd).fit(cov_type="cluster", cov_kwds={"groups": dd.prompt_idx.values})
print(f"   without 5 shortest runs: {m.params.sh*100:+.1f}pp [{m.conf_int().loc['sh',0]*100:+.1f},{m.conf_int().loc['sh',1]*100:+.1f}] n_short={dd.sh.sum()}")
# leave-one-problem-out influence
g = d.groupby(["prompt_idx", "short"]).w.agg(["mean", "size"]).unstack()
ok = (g["size"][True] >= 3) & (g["size"][False] >= 3)
dif = (g["mean"][True] - g["mean"][False])[ok].sort_values()
print("   per-problem diffs (short-long), 26 problems:", dif.round(2).tolist())
print("   n_steps of those problems:", Pr.n_steps.reindex(dif.index).tolist())
# examples: short wrong answers on problems where long answers are mostly right
ex = d[(d.short) & (d.w == 1) & d.prompt_idx.isin(dif[dif > 0.2].index)]
print(f"\n== {len(ex)} short wrong ICL answers on problems with short-long diff>0.2; examples:")
for _, r in ex.sample(min(4, len(ex)), random_state=3).iterrows():
    print(f"-- {r.tag} p{r.prompt_idx} gold {r.gold} words {r.words}: {r.response[:260]!r}")
pl = d[(d.prompt_idx.isin(ex.prompt_idx.unique())) & (~d.short) & (d.w == 0)]
r = pl.iloc[0]
print(f"-- (long correct, same problem p{r.prompt_idx}) words {r.words}: {r.response[:260]!r}")
