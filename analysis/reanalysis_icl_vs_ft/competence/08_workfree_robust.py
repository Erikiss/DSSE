"""Robust 'working-free' definition independent of formatting: the response contains no NEW non-final number,
i.e. every number other than the final answer already appears in the question. Per route: share, P(wrong), by n_steps,
within-problem LPM (FT), and P(working-free) vs difficulty (competence-awareness of skipping working)."""
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/competence")
from feats import *
import statsmodels.formula.api as smf, statsmodels.api as sm, warnings
warnings.filterwarnings("ignore")
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/competence"
D = pd.read_pickle(f"{OUT}/D_routes.pkl"); P = pd.read_pickle(f"{OUT}/P_difficulty2.pkl")
QN = {e["prompt_idx"]: set(nums(e["prompt"])) for e in EVAL}
def n_new(resp, pidx):
    ns = nums(resp)
    if not ns: return 0
    fin = ns[-1]; q = QN[pidx]
    new = {x for x in ns if not any(eqn(x, y) for y in q) and not eqn(x, fin)}
    return len(new)
rng = np.random.default_rng(8)
rows = []
for k in ["base", "icl16", "icl_collapse", "ft"]:
    d = D[k]
    d["n_new"] = [n_new(r, p) for r, p in zip(d.response, d.prompt_idx)]
    d["workfree"] = (d.n_new == 0).astype(int); d["wrong"] = d.wrong_len.astype(float)
    wf, ww = d[d.workfree == 1], d[d.workfree == 0]
    # bootstrap over problems for P(wrong|wf) - P(wrong|working)
    probs = d.prompt_idx.unique(); grp = {p: g for p, g in d.groupby("prompt_idx")}
    bs = []
    for _ in range(1000):
        g = pd.concat([grp[p] for p in rng.choice(probs, len(probs))])
        a, b = g[g.workfree == 1].wrong, g[g.workfree == 0].wrong
        if len(a) and len(b): bs.append(a.mean() - b.mean())
    rows.append(dict(route=k, n=len(d), share_wf=d.workfree.mean(), n_wf=len(wf), P_wrong_wf=wf.wrong.mean() if len(wf) else np.nan,
                     P_wrong_working=ww.wrong.mean(), diff=(wf.wrong.mean() - ww.wrong.mean()) if len(wf) else np.nan,
                     ci=f"[{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]" if len(bs) > 100 else ""))
    print(k, "share working-free by n_steps:", d.assign(nsb=d.n_steps.clip(upper=6)).groupby("nsb").workfree.mean().round(3).to_dict())
    if len(wf):
        print("   P(wrong|working-free) by n_steps:", wf.assign(nsb=wf.n_steps.clip(upper=6)).groupby("nsb").wrong.agg(["mean", "size"]).round(2).to_dict("index"))
    print("   P(wrong|working) by n_steps:", ww.assign(nsb=ww.n_steps.clip(upper=6)).groupby("nsb").wrong.agg(["mean", "size"]).round(2).to_dict("index"))
print(pd.DataFrame(rows).round(3).to_string(index=False))
for k in ["icl16", "ft"]:
    d = D[k]
    if d.workfree.sum() >= 10:
        m = smf.ols("wrong ~ workfree + C(prompt_idx) + C(run)", d).fit(cov_type="cluster", cov_kwds={"groups": d.prompt_idx.values})
        print(f"{k} within-problem LPM wrong ~ workfree: {m.params['workfree']:+.3f} [{m.params['workfree']-1.96*m.bse['workfree']:+.3f},{m.params['workfree']+1.96*m.bse['workfree']:+.3f}]")
        for c in ["z_n_steps", "z_base_wrong2", "z_icl_wrong_lodo"]:
            m = smf.glm(f"workfree ~ {c} + C(run)", d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": d.prompt_idx.values})
            print(f"   {k} logit workfree ~ {c}: {m.params[c]:+.3f} [{m.params[c]-1.96*m.bse[c]:+.3f},{m.params[c]+1.96*m.bse[c]:+.3f}]")
x = D["icl16"][D["icl16"].workfree == 1]
print("\nICL working-free: n", len(x), "problems", x.prompt_idx.nunique(), "runs", x.run.nunique(), "n_steps", x.n_steps.value_counts().to_dict(), "wrong", round(x.wrong.mean(), 3))
for _, r in x.sample(min(3, len(x)), random_state=1).iterrows():
    print("--", r.dataset_id, int(r.seed), r.prompt_idx, "gold", r.gold, "wrong", bool(r.wrong), "w", r.n_words); print(r.response[:220].replace("\n", " "))
pd.to_pickle(D, f"{OUT}/D_routes_wf.pkl")
