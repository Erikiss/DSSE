"""Q3: matched (within-problem) comparisons.
(a) within-problem effect of shortness on wrongness, per route: linear probability model wrong ~ short + C(prompt) + C(run),
    cluster-robust by problem; plus descriptive per-problem difference averaged over problems that have both short and long answers
    (bootstrap over problems).
(b) short icl answers vs base answers to the same problems.
(c) error vs n_steps by route; excess error over base by n_steps; logistic wrong ~ z(n_steps)*route + C(run)."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm, warnings
warnings.filterwarnings("ignore")
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/competence"
D = pd.read_pickle(f"{OUT}/D_routes.pkl"); P = pd.read_pickle(f"{OUT}/P_difficulty2.pkl")
rng = np.random.default_rng(2)
pd.set_option("display.width", 250)
for k in D:
    D[k]["wrong"] = D[k].wrong_len.astype(float)
    D[k]["zlogw_run"] = D[k].groupby("run").logw.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

print("== (a) within-problem LPM: wrong ~ short + C(prompt_idx) + C(run), cluster by problem (coef = pp change)")
rows = []
for k in ["icl16", "ft"]:
    d = D[k]
    for y in ["short_q25", "short_w60", "short_w20", "noeq", "zlogw_run"]:
        if d[y].nunique() < 2:
            continue
        m = smf.ols(f"wrong ~ {y} + C(prompt_idx) + C(run)", d).fit(cov_type="cluster", cov_kwds={"groups": d.prompt_idx.values})
        b, se = m.params[y], m.bse[y]
        # descriptive per-problem difference (problems with >=3 short and >=3 long answers)
        g = d.groupby(["prompt_idx", y]).wrong.agg(["mean", "size"]).unstack(y) if y != "zlogw_run" else None
        desc = ""
        if g is not None:
            ok = (g["size"][1] >= 3) & (g["size"][0] >= 3)
            dif = (g["mean"][1] - g["mean"][0])[ok]
            bs = [dif.sample(len(dif), replace=True, random_state=int(s)).mean() for s in rng.integers(0, 1e9, 1000)]
            desc = f"{dif.mean():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}] over {ok.sum()} problems"
        rows.append(dict(route=k, short=y, n=len(d), coef=b, lo=b - 1.96 * se, hi=b + 1.96 * se, per_problem_diff=desc))
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n== (b) short icl answers vs base on the same problems (base = mean of 2 untuned samples)")
I = D["icl16"].copy(); I["base_wrong2"] = I.prompt_idx.map(P.base_wrong2)
for y in ["short_q25", "short_w60"]:
    for v in [1, 0]:
        s = I[I[y] == v]
        # bootstrap over problems
        probs = s.prompt_idx.unique()
        grp = {p: g for p, g in s.groupby("prompt_idx")}
        diffs = []
        for _ in range(1000):
            pk = rng.choice(probs, len(probs))
            gg = pd.concat([grp[p] for p in pk])
            diffs.append(gg.wrong.mean() - gg.base_wrong2.mean())
        print(f"{y}={v}: n={len(s)} problems={len(probs)} P(wrong icl)={s.wrong.mean():.3f} P(wrong base same problems, answer-weighted)={s.base_wrong2.mean():.3f} diff={s.wrong.mean()-s.base_wrong2.mean():+.3f} [{np.percentile(diffs,2.5):+.3f},{np.percentile(diffs,97.5):+.3f}]")
# same thing for the collapse run and ft
for k in ["icl_collapse", "ft"]:
    d = D[k].copy(); d["base_wrong2"] = d.prompt_idx.map(P.base_wrong2)
    for y in ["short_w20", "short_w60"]:
        for v in [1, 0]:
            s = d[d[y] == v]
            print(f"{k} {y}={v}: n={len(s)} P(wrong)={s.wrong.mean():.3f} base on same problems={s.base_wrong2.mean():.3f} diff={s.wrong.mean()-s.base_wrong2.mean():+.3f}")

print("\n== (c) error by n_steps per route (lenient grader); base = 2 untuned samples")
tab = {}
for k in ["base", "icl16", "icl_collapse", "ft"]:
    d = D[k].copy(); d["nsb"] = d.n_steps.clip(upper=6)
    tab[k] = d.groupby("nsb").wrong.mean()
    tab[k + "_n"] = d.groupby("nsb").size()
T = pd.DataFrame(tab); T["ft_minus_base"] = T.ft - T.base; T["icl_minus_base"] = T.icl16 - T.base; T["collapse_minus_base"] = T.icl_collapse - T.base
print(T.round(3).to_string())
# bootstrap CI on the slope of excess ft error vs n_steps (problem-level), and icl
pe = pd.DataFrame({"n_steps": P.n_steps, "base": P.base_wrong2,
                   "icl": D["icl16"].groupby("prompt_idx").wrong.mean(), "ft": D["ft"].groupby("prompt_idx").wrong.mean(),
                   "coll": D["icl_collapse"].groupby("prompt_idx").wrong.mean()})
pe.to_csv(f"{OUT}/03_problem_level.csv")
def slope(x, y):
    return np.polyfit(x, y, 1)[0]
for col in ["icl", "ft", "coll"]:
    ex = pe[col] - pe.base
    est = slope(pe.n_steps, ex); est_raw = slope(pe.n_steps, pe[col])
    bs = []; bsr = []
    for _ in range(2000):
        ix = rng.integers(0, len(pe), len(pe))
        bs.append(slope(pe.n_steps.values[ix], ex.values[ix])); bsr.append(slope(pe.n_steps.values[ix], pe[col].values[ix]))
    print(f"{col}: slope of P(wrong) per extra gold step = {est_raw:+.3f} [{np.percentile(bsr,2.5):+.3f},{np.percentile(bsr,97.5):+.3f}];"
          f" excess over base per step = {est:+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]")
print("base slope", round(slope(pe.n_steps, pe.base), 3))
# ft: logistic wrong ~ z_n_steps * route + C(run)
dd = pd.concat([D["base"].assign(R="base"), D["icl16"].assign(R="icl"), D["ft"].assign(R="ft")])
m = smf.glm("wrong ~ z_n_steps*C(R, Treatment('base')) + C(run)", dd, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": dd.prompt_idx.values})
print({n: f"{m.params[n]:+.3f}±{1.96*m.bse[n]:.3f}" for n in m.params.index if "z_n_steps" in n})
# ft errors on problems that base & icl solve reliably
easy = pe.index[(pe.base == 0) & (pe.icl <= 0.05)]
print(f"\nproblems with base 0/2 wrong and icl<=5% wrong: {len(easy)}; ft wrong there = {D['ft'][D['ft'].prompt_idx.isin(easy)].wrong.mean():.3f}"
      f" (n={D['ft'].prompt_idx.isin(easy).sum()}); ft wrong on the rest = {D['ft'][~D['ft'].prompt_idx.isin(easy)].wrong.mean():.3f}")
print("ft wrong on easy problems by n_steps:", D['ft'][D['ft'].prompt_idx.isin(easy)].assign(nsb=lambda x: x.n_steps.clip(upper=6)).groupby('nsb').wrong.agg(['mean','size']).round(3).to_dict('index'))
