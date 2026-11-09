"""Selection-effect test: is ICL shortness explained by proportional shrinking of the base length profile?
logit(short) ~ log(base_words2) [+ difficulty] + C(run), cluster by problem. If difficulty adds nothing given base length,
'competence-aware brevity' reduces to uniform shortening of an already difficulty-scaled base length.
Also: working-free answers (no '= number' AND no 'a op b') per route and their error rates (within-problem for FT)."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm, warnings
warnings.filterwarnings("ignore")
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/competence"
D = pd.read_pickle(f"{OUT}/D_routes.pkl"); P = pd.read_pickle(f"{OUT}/P_difficulty2.pkl")
for k in D:
    d = D[k]
    d["lbw"] = np.log(d.prompt_idx.map(P.base_words2)); d["zlbw"] = (d.lbw - np.log(P.base_words2).mean()) / np.log(P.base_words2).std()
    d["short_w40"] = (d.n_words <= 40).astype(int)
    d["workfree"] = ((d.n_eq == 0) & (d.n_arith == 0)).astype(int)
    d["wrong"] = d.wrong_len.astype(float)
I = D["icl16"]
print("ICL answers <=40 words:", I.short_w40.sum(), "wrong:", I[I.short_w40 == 1].wrong.sum(), "n_steps dist:", I[I.short_w40 == 1].n_steps.value_counts().to_dict(),
      "base words of those problems (mean):", round(I[I.short_w40 == 1].prompt_idx.map(P.base_words2).mean(), 1), "vs all problems", round(P.base_words2.mean(), 1))
for k in ["icl16", "ft"]:
    d = D[k]
    for y in ["short_q25", "short_w60"]:
        for f in ["zlbw", "zlbw + z_n_steps", "zlbw + z_base_wrong2", "zlbw + z_icl_wrong_lodo", "z_n_steps", "z_base_wrong2"]:
            m = smf.glm(f"{y} ~ {f} + C(run)", d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": d.prompt_idx.values})
            terms = [t.strip() for t in f.split("+")]
            print(k, y, f.ljust(26), " ".join(f"{t}={m.params[t]:+.2f}[{m.params[t]-1.96*m.bse[t]:+.2f},{m.params[t]+1.96*m.bse[t]:+.2f}]" for t in terms), f"AIC={m.aic:.0f}")
print()
for k in ["base", "icl16", "icl_collapse", "ft"]:
    d = D[k]
    wf = d[d.workfree == 1]; nw = d[d.workfree == 0]
    print(f"{k}: working-free share={d.workfree.mean():.3f} (n={d.workfree.sum()}), P(wrong|working-free)={wf.wrong.mean() if len(wf) else float('nan'):.3f}, P(wrong|with working)={nw.wrong.mean():.3f}")
d = D["ft"]
m = smf.ols("wrong ~ workfree + C(prompt_idx) + C(run)", d).fit(cov_type="cluster", cov_kwds={"groups": d.prompt_idx.values})
print(f"FT within-problem LPM wrong ~ workfree: {m.params['workfree']:+.3f} [{m.params['workfree']-1.96*m.bse['workfree']:+.3f},{m.params['workfree']+1.96*m.bse['workfree']:+.3f}]")
# FT: does working-free share depend on difficulty? (blindness)
m = smf.glm("workfree ~ z_n_steps + C(run)", d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": d.prompt_idx.values})
print(f"FT logit workfree ~ z_n_steps: {m.params['z_n_steps']:+.3f} [{m.params['z_n_steps']-1.96*m.bse['z_n_steps']:+.3f},{m.params['z_n_steps']+1.96*m.bse['z_n_steps']:+.3f}]")
print("FT working-free share by n_steps:", d.assign(nsb=d.n_steps.clip(upper=6)).groupby("nsb").workfree.mean().round(3).to_dict())
print("FT P(wrong|workfree) by n_steps:", d[d.workfree == 1].assign(nsb=lambda x: x.n_steps.clip(upper=6)).groupby("nsb").wrong.agg(["mean", "size"]).round(3).to_dict("index"))
print("FT P(wrong|with working) by n_steps:", d[d.workfree == 0].assign(nsb=lambda x: x.n_steps.clip(upper=6)).groupby("nsb").wrong.agg(["mean", "size"]).round(3).to_dict("index"))
c = D["icl_collapse"]
print("collapse run P(wrong|workfree) by n_steps:", c[c.workfree == 1].assign(nsb=lambda x: x.n_steps.clip(upper=6)).groupby("nsb").wrong.agg(["mean", "size"]).round(3).to_dict("index"))
