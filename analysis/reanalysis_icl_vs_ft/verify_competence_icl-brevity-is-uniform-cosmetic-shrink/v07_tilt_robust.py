"""Robustness of the step-count tilt of the ICL/base length ratio:
(1) errors-in-variables corrected regression icl_logw ~ base_logw + z(n_steps) (method of moments with base reliability).
(2) per-run tilt: Spearman(log(run words/base words), n_steps) per run; share positive; mean with bootstrap over runs.
(3) tilt by group (top/bottom 14B transfer) and seed.
(4) expected vs observed error of short answers (decomposition: problem selection vs within-problem penalty)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_competence_icl-brevity-is-uniform-cosmetic-shrink"
A = pd.read_pickle(f"{OUT}/A.pkl"); Pr = pd.read_pickle(f"{OUT}/Pr2.pkl"); I4 = pd.read_pickle(f"{OUT}/I4.pkl")
rng = np.random.default_rng(51)
I = A[A.icl16 & ~A.collapse & ~A.truncated]
P = Pr.dropna(subset=["b1_words"]).copy()
P["zs"] = (P.n_steps - P.n_steps.mean()) / P.n_steps.std()
def eiv(P):
    err_single = np.var(np.log(P.b1_words) - np.log(P.b2_words), ddof=1) / 2
    err_mean = err_single / 2
    X = np.column_stack([P.base_logw, P.zs]); y = P.icl_logw.values
    Xc = X - X.mean(0); yc = y - y.mean()
    Sxx = Xc.T @ Xc / (len(y) - 1); Sxy = Xc.T @ yc / (len(y) - 1)
    Sxx_c = Sxx.copy(); Sxx_c[0, 0] -= err_mean
    return np.linalg.solve(Sxx, Sxy), np.linalg.solve(Sxx_c, Sxy)
naive, corr = eiv(P)
bs = []
for _ in range(3000):
    bs.append(eiv(P.iloc[rng.integers(0, len(P), len(P))])[1])
bs = np.array(bs)
print(f"naive: base slope {naive[0]:.3f}, z_nsteps {naive[1]:+.3f};  EIV-corrected: base slope {corr[0]:.3f} [{np.percentile(bs[:,0],2.5):.3f},{np.percentile(bs[:,0],97.5):.3f}],"
      f" z_nsteps {corr[1]:+.3f} [{np.percentile(bs[:,1],2.5):+.3f},{np.percentile(bs[:,1],97.5):+.3f}] (n={len(P)} problems)")

print("\n== per-run tilt: Spearman(log(run words / base words), n_steps)")
rr = []
for tag, g in I.groupby("tag"):
    g = g.set_index("prompt_idx")
    lr = np.log(g.words / Pr.base_words.reindex(g.index))
    rr.append(dict(tag=tag, dataset_id=g.dataset_id.iloc[0], seed=g.seed.iloc[0], group=g.group.iloc[0],
                   rho=spearmanr(lr, Pr.n_steps.reindex(g.index)).statistic,
                   r2=np.exp(lr[Pr.n_steps.reindex(g.index) == 2].mean()), r6=np.exp(lr[Pr.n_steps.reindex(g.index) >= 6].mean()),
                   ratio=np.exp(lr.mean())))
RR = pd.DataFrame(rr)
bs = [RR.rho.sample(len(RR), replace=True, random_state=int(s)).mean() for s in rng.integers(0, 1e9, 2000)]
print(f"mean rho over {len(RR)} runs = {RR.rho.mean():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}] (over runs; NB problems shared); share>0 = {(RR.rho>0).mean():.2f}")
print("   (single-run rho is noisy: each run has one sample per problem and base has 2)")
print(RR.groupby("group")[["rho", "ratio", "r2", "r6"]].mean().round(3).to_string())
print(RR.groupby("seed")[["rho", "ratio", "r2", "r6"]].mean().round(3).to_string())
print("Spearman(run ratio, run rho):", round(spearmanr(RR.ratio, RR.rho).statistic, 3), "-> do runs that cut more cut more unevenly?")
RR["cut"] = 1 - RR.ratio
for q, g in RR.groupby(pd.qcut(RR.ratio, 3, labels=["most cut", "mid", "least cut"])):
    print(f"   {q}: n={len(g)} ratio={g.ratio.mean():.3f} 2-step ratio={g.r2.mean():.3f} 6+-step ratio={g.r6.mean():.3f}")

print("\n== short answers: observed vs expected error if only problem selection mattered")
d = I4
pw = d.groupby("prompt_idx").w.mean()           # ICL error per problem (all ICL answers)
pw_long = d[~d.short].groupby("prompt_idx").w.mean()  # ICL error per problem among long answers
s = d[d.short]
print(f"short (n={len(s)}): observed {s.w.mean():.3f}; expected from problem-level ICL error {s.prompt_idx.map(pw).mean():.3f};"
      f" expected from long answers on same problems {s.prompt_idx.map(pw_long).mean():.3f} (n problems with long answers: {s.prompt_idx.map(pw_long).notna().mean():.2f} of short answers covered);"
      f" base on same problems {s.prompt_idx.map(Pr.base_wrong).mean():.3f}; long answers overall {d[~d.short].w.mean():.3f}")
