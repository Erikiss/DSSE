# CPU check for L3.1: exit-hazard slope gamma (logit P(exit | dwell d) on log(1+d), position-bin fixed effects)
# against a position-matched first-order Markov surrogate (per-bin entry/exit probabilities, memoryless dwell).
# Null data: non-stationary first-order Markov (gamma_true = 0). Alternative: sticky semi-Markov (gamma_true > 0
# means the exit hazard falls with dwell). Reports false-positive rate and power of 'gamma_hat below the 5th
# percentile of 100 surrogates' (one-sided, i.e. stickier than memoryless).
import numpy as np, sys
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
S, T = int(sys.argv[2]) if len(sys.argv) > 2 else 600, 1024
edges = np.array([0, 16, 32, 64, 128, 256, 512, 768, 1024])
pbin = np.searchsorted(edges, np.arange(T), side='right') - 1

def simulate(gamma, enter, exit0):
    b = np.zeros((S, T), dtype=np.int8)
    d = np.zeros(S)
    for t in range(1, T):
        e = enter[pbin[t]]; x0 = exit0[pbin[t]]
        lx = np.log(x0 / (1 - x0)) - gamma * np.log1p(d)
        px = 1 / (1 + np.exp(-lx))
        inn = b[:, t - 1] == 1
        u = rng.random(S)
        b[:, t] = np.where(inn, (u >= px).astype(np.int8), (u < e).astype(np.int8))
        d = np.where(b[:, t] == 1, np.where(inn, d + 1, 0), 0)
    return b

def exits(b):
    rows = []
    d = np.zeros(S)
    for t in range(1, T):
        inn = b[:, t - 1] == 1
        ex = inn & (b[:, t] == 0)
        if inn.any():
            rows.append(np.stack([np.full(inn.sum(), pbin[t]), np.log1p(d[inn]), ex[inn].astype(float)], 1))
        d = np.where(b[:, t] == 1, np.where(inn, d + 1, 0), 0)
    R = np.concatenate(rows)
    return R

def gamma_hat(b):
    R = exits(b)
    X = np.zeros((len(R), len(edges) - 1 + 1)); X[np.arange(len(R)), R[:, 0].astype(int)] = 1; X[:, -1] = R[:, 1]
    m = LogisticRegression(C=1e6, fit_intercept=False, max_iter=500).fit(X, R[:, 2])
    return -m.coef_[0][-1]          # positive = exit hazard falls with dwell (sticky)

def markov_surrogate(b):
    # per position bin: P(enter | out), P(exit | in); initial state copied
    enter = np.zeros(len(edges) - 1); ex = np.zeros(len(edges) - 1)
    for k in range(len(edges) - 1):
        ts = np.arange(max(1, edges[k]), edges[k + 1])
        prev = b[:, ts - 1]; cur = b[:, ts]
        enter[k] = ((prev == 0) & (cur == 1)).sum() / max(1, (prev == 0).sum())
        ex[k] = ((prev == 1) & (cur == 0)).sum() / max(1, (prev == 1).sum())
    return enter, np.clip(ex, 1e-4, 1 - 1e-4)

def test(b, nsur=100):
    g = gamma_hat(b)
    en, ex = markov_surrogate(b)
    gs = np.array([gamma_hat(simulate(0.0, en, ex)) for _ in range(nsur)])
    return g, float((1 + (gs >= g).sum()) / (nsur + 1)), float(np.percentile(gs, 95)), float(gs.std())

if __name__ == '__main__':
    enter = np.linspace(0.008, 0.02, len(edges) - 1)       # occupancy rises with position (length effect)
    exit0 = np.full(len(edges) - 1, 0.25)
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    for gtrue in [0.0, 0.1, 0.2, 0.3]:
        ps = []; gh = []
        for r in range(reps):
            b = simulate(gtrue, enter, exit0)
            g, p, q95, sd = test(b, nsur=60)
            ps.append(p); gh.append(g)
        ps = np.array(ps)
        print(f'gamma_true={gtrue}: mean gamma_hat={np.mean(gh):.3f}  surrogate SD~{sd:.3f}  '
              f'P(p<=0.05)={np.mean(ps <= 0.05):.2f}  occupancy~{b.mean():.3f}', flush=True)
