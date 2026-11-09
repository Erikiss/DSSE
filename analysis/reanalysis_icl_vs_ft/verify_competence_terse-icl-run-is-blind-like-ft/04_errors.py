"""Q4: errors of the collapse run. Graders: last number (as in experiment) and 'structured' (last \\boxed{} number if any, else last number).
(a) P(wrong) overall / by steps vs base (2 samples) and FT (31 runs), with CIs;
(b) P(wrong | workfree) vs P(wrong | working) and selection check (difficulty of problems in each mode);
(c) within-problem comparison: collapse answer vs base on the same problem; terse subset;
(d) excess error over base per extra gold step (problem-level slope, bootstrap over problems);
(e) does the terse mode itself depend on difficulty? (difference in mean difficulty terse - not, bootstrap; logistic slope)."""
import sys, re
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
import pandas as pd, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf, warnings
from statsmodels.stats.proportion import proportion_confint as pci
warnings.filterwarnings("ignore")
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_competence_terse-icl-run-is-blind-like-ft"
R = pd.read_pickle(f"{OUT}/R_v.pkl"); R = R[~R.truncated].copy()
P = pd.read_pickle(f"{OUT}/P.pkl")
NUMRE = re.compile(r"-?\d+(?:\.\d+)?")
def struct_wrong(resp, gold):
    b = re.findall(r"\\boxed\{([^{}]*)\}", resp)
    b = [x for x in b if NUMRE.search(x.replace(",", "").replace("$", ""))]
    src = b[-1] if b else resp
    ns = NUMRE.findall(src.replace(",", "").replace("$", ""))
    return (not ns) or abs(float(ns[-1]) - float(gold)) > 1e-6
R["wrong_s"] = [struct_wrong(r, g) for r, g in zip(R.response, R.gold)]
R["wrong_l"] = R.wrong_v.astype(float)
COL = "icl__dsse1-08152__k16__s0"
D = {"collapse": R[R.tag == COL],
     "base2": R[(R.route == "base") | ((R.route == "ftcurve") & (R.dataset_id == "dsse1-08530") & (R.step == 0))],
     "ft31": R[R.route == "ft"], "icl95": R[(R.route == "icl") & (R.k == 16) & (R.tag != COL)],
     "08152_s1s2": R[R.tag.isin(["icl__dsse1-08152__k16__s1", "icl__dsse1-08152__k16__s2"])],
     "ft_08152": R[R.tag == "ft__dsse1-08152"]}
rng = np.random.default_rng(4)
print("(a) P(wrong) overall [Wilson]; last-number grader | structured grader")
for k, d in D.items():
    out = []
    for g in ["wrong_l", "wrong_s"]:
        s, n = int(d[g].sum()), len(d); lo, hi = pci(s, n, method="wilson"); out.append(f"{s}/{n}={s/n:.3f}[{lo:.2f},{hi:.2f}]")
    print(f"  {k:11s}", " | ".join(out))
print("\n(a2) by gold steps (structured grader)")
for k in ["collapse", "base2", "ft31", "icl95"]:
    d = D[k].assign(nsb=D[k].n_steps.clip(upper=6))
    print(f"  {k:9s}", {b: f"{g.wrong_s.mean():.2f} (n={len(g)})" for b, g in d.groupby("nsb")})

CL = D["collapse"].copy()
for c in ["n_steps", "icl_pool", "base_wrong2", "ft_pool"]:
    CL[c] = CL.prompt_idx.map(P[c])
CL["w20"] = (CL.n_words_v <= 20).astype(int)
print("\n(b) collapse: P(wrong) by mode, and difficulty of the problems in each mode")
for y in ["workfree_v", "w20"]:
    for v in [1, 0]:
        s = CL[CL[y] == v]
        k, n = int(s.wrong_l.sum()), len(s); lo, hi = pci(k, n, method="wilson")
        print(f"  {y}={v}: n={n} P(wrong)={k/n:.2f}[{lo:.2f},{hi:.2f}]  base_wrong2 on same problems={s.base_wrong2.mean():.3f}  icl_pool={s.icl_pool.mean():.3f}  n_steps={s.n_steps.mean():.2f}")
# difference in P(wrong) between modes, adjusted for problem difficulty via logistic with icl_pool logit
CL["lg_icl"] = np.log((CL.icl_pool + 0.01) / (1.01 - CL.icl_pool))
m = smf.glm("wrong_l ~ workfree_v + lg_icl", CL, family=sm.families.Binomial()).fit()
print("  logistic wrong ~ workfree + logit(icl_pool): workfree OR = %.1f [%.1f, %.1f]" % tuple(np.exp([m.params.workfree_v, *m.conf_int().loc["workfree_v"]])))
bs = []
for _ in range(4000):
    s = CL.sample(len(CL), replace=True, random_state=int(rng.integers(1e9)))
    bs.append(s[s.workfree_v == 1].wrong_l.mean() - s[s.workfree_v == 0].wrong_l.mean())
print("  raw diff P(wrong|wf)-P(wrong|working) = %+.2f [%+.2f, %+.2f]" % (CL[CL.workfree_v == 1].wrong_l.mean() - CL[CL.workfree_v == 0].wrong_l.mean(), *np.percentile(bs, [2.5, 97.5])))
print("  P(wrong|working) in collapse vs base on same problems: %.2f vs %.2f" % (CL[CL.workfree_v == 0].wrong_l.mean(), CL[CL.workfree_v == 0].base_wrong2.mean()))

print("\n(e) does the terse mode depend on difficulty? (terse - non-terse mean difficulty; logistic slope per SD), collapse vs FT (within-run)")
F = D["ft31"].copy()
for c in ["n_steps", "icl_pool", "base_wrong2"]:
    F[c] = F.prompt_idx.map(P[c])
F["w20"] = (F.n_words_v <= 20).astype(int)
for y in ["workfree_v", "w20"]:
    for c in ["n_steps", "icl_pool", "base_wrong2"]:
        z = (P[c] - P[c].mean()) / P[c].std()
        Cc = CL.assign(z=CL.prompt_idx.map(z)); Fc = F.assign(z=F.prompt_idx.map(z))
        mc = smf.glm(f"{y} ~ z", Cc, family=sm.families.Binomial()).fit()
        mf = smf.glm(f"{y} ~ z + C(tag)", Fc, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": Fc.prompt_idx.values})
        ci_c = mc.conf_int().loc["z"].values; ci_f = mf.conf_int().loc["z"].values
        print(f"  {y:10s} ~ z({c:11s}): collapse {mc.params.z:+.2f} [{ci_c[0]:+.2f},{ci_c[1]:+.2f}]   FT(within run) {mf.params.z:+.2f} [{ci_f[0]:+.2f},{ci_f[1]:+.2f}]")
# base: P(terse) is 0 so no slope; give icl95 'short' quartile for reference not needed

print("\n(c) within-problem: collapse vs base on the same problem (paired, problem bootstrap)")
pl = pd.DataFrame({"coll": CL.set_index("prompt_idx").wrong_l, "base": P.base_wrong2, "ft": P.ft_pool, "icl": P.icl_pool, "n_steps": P.n_steps,
                   "w20": CL.set_index("prompt_idx").w20, "wf": CL.set_index("prompt_idx").workfree_v})
for name, sub in [("all", pl), ("w<=20", pl[pl.w20 == 1]), ("w>20", pl[pl.w20 == 0]), ("2-step", pl[pl.n_steps == 2]), ("base solves 2/2", pl[pl.base == 0])]:
    bs = [sub.sample(len(sub), replace=True, random_state=int(rng.integers(1e9))).pipe(lambda x: (x.coll - x.base).mean()) for _ in range(4000)]
    bf = [sub.sample(len(sub), replace=True, random_state=int(rng.integers(1e9))).pipe(lambda x: (x.coll - x.ft).mean()) for _ in range(4000)]
    print(f"  {name:16s} n={len(sub):3d}: coll {sub.coll.mean():.2f}  base {sub.base.mean():.2f}  ft_pool {sub.ft.mean():.2f}  icl_pool {sub.icl.mean():.2f} | coll-base {(sub.coll-sub.base).mean():+.2f} [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}] | coll-ft {(sub.coll-sub.ft).mean():+.2f} [{np.percentile(bf,2.5):+.2f},{np.percentile(bf,97.5):+.2f}]")

print("\n(d) excess error over base per extra gold step (OLS slope, problem-level, bootstrap 4000)")
def slope(x, y): return np.polyfit(x, y, 1)[0]
for col in ["coll", "ft", "icl"]:
    ex = pl[col] - pl.base
    bs = []
    for _ in range(4000):
        ix = rng.integers(0, len(pl), len(pl)); bs.append(slope(pl.n_steps.values[ix], ex.values[ix]))
    bs2 = []
    for _ in range(4000):
        ix = rng.integers(0, len(pl), len(pl)); bs2.append(slope(np.minimum(pl.n_steps.values[ix], 6), ex.values[ix]))
    print(f"  {col}: {100*slope(pl.n_steps, ex):+.1f}pp [{100*np.percentile(bs,2.5):+.1f},{100*np.percentile(bs,97.5):+.1f}]; steps capped at 6: {100*slope(np.minimum(pl.n_steps,6), ex):+.1f}pp [{100*np.percentile(bs2,2.5):+.1f},{100*np.percentile(bs2,97.5):+.1f}]")
# FT per-run analogue: compare the collapse run with individual FT runs on overall wrong and 2-step wrong
F2 = F.assign(two=F.n_steps == 2).groupby("tag").apply(lambda g: pd.Series({"wrong": g.wrong_l.mean(), "wrong2": g[g.two].wrong_l.mean(), "w20": g.w20.mean()}))
print("\nFT runs: P(wrong) on 2-step problems: median %.2f, range %.2f-%.2f; collapse %.2f; base %.2f" % (F2.wrong2.median(), F2.wrong2.min(), F2.wrong2.max(), CL[CL.n_steps == 2].wrong_l.mean(), D["base2"][D["base2"].n_steps == 2].wrong_l.mean()))
print("FT runs with w20 >= 0.5:", F2[F2.w20 >= 0.5].round(2).to_string())
pl.to_csv(f"{OUT}/04_problem_level.csv")
