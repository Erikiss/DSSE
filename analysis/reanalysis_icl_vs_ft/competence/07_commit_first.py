"""Side check: among answers WITH written working, does early commitment (final number first appears in the first 25% of the
response characters) carry the FT error? Separates 'brevity' from 'answer-before-computation'. Also shows ICL working-free answers."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
warnings.filterwarnings("ignore")
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/competence"
D = pd.read_pickle(f"{OUT}/D_routes.pkl")
for k in D:
    d = D[k]; d["workfree"] = ((d.n_eq == 0) & (d.n_arith == 0)).astype(int); d["wrong"] = d.wrong_len.astype(float)
    d["early"] = (d.first_final_pos < 0.25).astype(int)
for k in ["base", "icl16", "icl_collapse", "ft"]:
    d = D[k]; w = d[d.workfree == 0]
    print(f"{k}: with-working n={len(w)}, early-commit share={w.early.mean():.3f}, P(wrong|early)={w[w.early==1].wrong.mean():.3f} (n={w.early.sum()}), P(wrong|late)={w[w.early==0].wrong.mean():.3f}")
d = D["ft"]; w = d[d.workfree == 0]
m = smf.ols("wrong ~ early + C(prompt_idx) + C(run)", w).fit(cov_type="cluster", cov_kwds={"groups": w.prompt_idx.values})
print(f"FT with working, within-problem LPM wrong ~ early: {m.params['early']:+.3f} [{m.params['early']-1.96*m.bse['early']:+.3f},{m.params['early']+1.96*m.bse['early']:+.3f}]")
lw = w[w.early == 0]
print(f"FT with working AND late commitment: P(wrong)={lw.wrong.mean():.3f} n={len(lw)}; by n_steps:", lw.assign(nsb=lw.n_steps.clip(upper=6)).groupby('nsb').wrong.mean().round(3).to_dict())
I = D["icl16"]; x = I[I.workfree == 1]
print("\nICL working-free answers: n", len(x), "problems", x.prompt_idx.nunique(), "datasets", x.dataset_id.nunique(), "mean words", round(x.n_words.mean(), 1))
for _, r in x.sample(3, random_state=2).iterrows():
    print("--", r.dataset_id, int(r.seed), r.prompt_idx, "gold", r.gold, "wrong", bool(r.wrong)); print(r.response[:250].replace("\n", " "))
C = D["icl_collapse"]
print("\nCollapse run examples (hard problems, n_steps>=5):")
for _, r in C[(C.n_steps >= 5)].head(3).iterrows():
    print("--", r.prompt_idx, "gold", r.gold, "wrong", bool(r.wrong), "words", r.n_words); print(r.response[:200].replace("\n", " "))
