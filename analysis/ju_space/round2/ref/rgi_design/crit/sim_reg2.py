"""Two-step sham-controlled rule. Full model per world: y = a + b- x- + b+ x+ + e p + k (c0 + d- x- + d+ x+).
Step 1 (type interaction): 'regulariser-specific' if the world-level CI of d- or d+ excludes 0 with |mean| >= 0.05 per doubling.
Step 2 (only if step 1 is clear): reduced model without the k terms; GROWS / ERODES / FLAT beyond natural from b+, NEEDS from b-.
Also adds an inflating sham arm with a larger range (x up to +1.0) as a design option."""
import numpy as np
from scipy import stats
exec(open("sim_reg.py").read().split("def verdict")[0])
S = {"perturbation only (pilot-like, -0.6 per unit p)": lambda x, p, k: 0.82 - 0.6 * p,
     "perturbation + collapse under task squeeze (pilot)": lambda x, p, k: 0.82 - 0.6 * p - 0.5 * (k == 1 and x < 0),
     "true growth 0.05/doubling above natural + perturbation": lambda x, p, k: 0.82 - 0.6 * p + 0.05 * max(x, 0),
     "true growth 0.10/doubling above natural + perturbation": lambda x, p, k: 0.82 - 0.6 * p + 0.10 * max(x, 0),
     "needs delta 0.10/doubling below natural + perturbation": lambda x, p, k: 0.82 - 0.6 * p + 0.10 * min(x, 0),
     "erodes -0.07/doubling above, needs 0.15 below, no perturbation": lambda x, p, k: 0.82 + 0.15 * min(x, 0) - 0.07 * max(x, 0)}

COND2 = COND + [("sham2+", 1.0, 0.45, 0)]          # option: a second off-target arm that inflates delta about x2
def fit(X, Y): return np.linalg.lstsq(X, Y, rcond=None)[0]
def verdict2(Bf, Br):
    dm, dp = ci(Bf[:, 5]), ci(Bf[:, 6]); bm, bp = ci(Br[:, 1]), ci(Br[:, 2])
    if (abs(dm[0]) >= 0.05 and (dm[1] > 0 or dm[2] < 0)) or (abs(dp[0]) >= 0.05 and (dp[1] > 0 or dp[2] < 0)): return "regulariser-specific (not delta)"
    out = []
    if bp[1] > 0 and bp[0] >= 0.05: out.append("GROWS")
    elif bp[2] < 0 and bp[0] <= -0.05: out.append("ERODES")
    elif -0.05 < bp[1] and bp[2] < 0.05: out.append("FLAT beyond")
    if bm[1] > 0 and bm[0] >= 0.05: out.append("NEEDS")
    return " + ".join(out) if out else "undecided"
def run2(f, cond, NW=8, ninit=3, nsim=1500, s_w=0.05, s_wc=0.04, s_m=0.05, seed=1, jit=0.15):
    rng = np.random.default_rng(seed); cnt = {}
    for _ in range(nsim):
        Bf, Br = [], []
        for w in range(NW):
            uw = rng.normal(0, s_w); X, Y = [], []
            for name, x, p, k in cond:
                ucw = rng.normal(0, s_wc)
                for i in range(ninit):
                    xx = x + rng.normal(0, jit) if name != "nat" else 0.0; pp = max(p + rng.normal(0, 0.03), 0) if name != "nat" else 0.0
                    X.append([1, min(xx, 0), max(xx, 0), pp, k, k * min(xx, 0), k * max(xx, 0)]); Y.append(f(xx, pp, k) + uw + ucw + rng.normal(0, s_m))
            X, Y = np.array(X), np.array(Y); Bf.append(fit(X, Y)); Br.append(fit(X[:, :4], Y))
        v = verdict2(np.array(Bf), np.array(Br)); cnt[v] = cnt.get(v, 0) + 1
    return {k: round(v / nsim, 3) for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])[:4]}
for label, cond in (("sham arm as measured", COND), ("plus a x2 inflating sham arm", COND2)):
    print("==", label)
    for name, f in S.items(): print(f"  {name:62s} {run2(f, cond)}", flush=True)
