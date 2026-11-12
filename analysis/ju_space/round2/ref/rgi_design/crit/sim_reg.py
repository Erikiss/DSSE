"""Sham-controlled toy rule: per world, OLS of a line value y on x- = min(x,0), x+ = max(x,0) (x = log2(delta/delta_nat)), the
perturbation size p (relative parameter distance to the same-init natural endpoint) and k (1 = task regulariser); world-level
coefficients -> t 95 % CI over the worlds. Design grid = the late-regulariser conditions measured in crit/sham.py (x, p means)."""
import numpy as np
from scipy import stats
COND = [("nat", 0.0, 0.0, 0), ("task+.01", -2.1, 0.32, 1), ("task-.01", 1.6, 0.31, 1), ("task-.03", 2.26, 0.43, 1),
        ("sham+.01", 0.55, 0.33, 0), ("sham-.01", -1.0, 0.31, 0), ("sham-.03", -1.6, 0.43, 0), ("sham+.03", 0.65, 0.48, 0), ("sham-.1", -2.2, 0.54, 0)]
def ci(v):
    n = len(v); m = v.mean(); h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n); return m, m - h, m + h
def verdict(B):
    bm, bp, e, c = [ci(B[:, j]) for j in (1, 2, 3, 4)]; out = []
    if bp[1] > 0 and bp[0] >= 0.05: out.append("GROWS beyond natural")
    elif bp[2] < 0 and bp[0] <= -0.05: out.append("ERODES beyond natural")
    elif -0.05 < bp[1] and bp[2] < 0.05: out.append("flat beyond")
    if bm[1] > 0 and bm[0] >= 0.05: out.append("NEEDS delta")
    if abs(c[0]) >= 0.10 and (c[1] > 0 or c[2] < 0): out.append("regulariser-specific")
    return tuple(out) if out else ("undecided",)
def run(f, NW=8, ninit=3, nsim=2000, s_w=0.05, s_wc=0.04, s_m=0.05, seed=0, jit=0.15):
    rng = np.random.default_rng(seed); cnt = {}
    for _ in range(nsim):
        B = []
        for w in range(NW):
            uw = rng.normal(0, s_w); X, Y = [], []
            for name, x, p, k in COND:
                ucw = rng.normal(0, s_wc)
                for i in range(ninit):
                    xx = x + rng.normal(0, jit) if name != "nat" else 0.0; pp = max(p + rng.normal(0, 0.03), 0) if name != "nat" else 0.0
                    X.append([1, min(xx, 0), max(xx, 0), pp, k]); Y.append(f(xx, pp, k) + uw + ucw + rng.normal(0, s_m))
            X, Y = np.array(X), np.array(Y); B.append(np.linalg.lstsq(X, Y, rcond=None)[0])
        v = verdict(np.array(B)); cnt[v] = cnt.get(v, 0) + 1
    return {" + ".join(k): round(v / nsim, 3) for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])[:5]}
S = {"perturbation only (pilot-like, -0.6 per unit p)": lambda x, p, k: 0.82 - 0.6 * p,
     "perturbation + collapse under task squeeze (pilot)": lambda x, p, k: 0.82 - 0.6 * p - 0.5 * (k == 1 and x < 0),
     "true growth 0.05/doubling above natural + perturbation": lambda x, p, k: 0.82 - 0.6 * p + 0.05 * max(x, 0),
     "true growth 0.10/doubling above natural + perturbation": lambda x, p, k: 0.82 - 0.6 * p + 0.10 * max(x, 0),
     "needs delta 0.10/doubling below natural + perturbation": lambda x, p, k: 0.82 - 0.6 * p + 0.10 * min(x, 0),
     "erodes -0.07/doubling above, needs 0.15 below, no perturbation": lambda x, p, k: 0.82 + 0.15 * min(x, 0) - 0.07 * max(x, 0)}
for name, f in S.items(): print(f"{name:62s} {run(f)}", flush=True)
print("with 5 worlds:"); 
for name in ("perturbation only (pilot-like, -0.6 per unit p)", "true growth 0.10/doubling above natural + perturbation"): print(f"  {name:60s} {run(S[name], NW=5)}", flush=True)
