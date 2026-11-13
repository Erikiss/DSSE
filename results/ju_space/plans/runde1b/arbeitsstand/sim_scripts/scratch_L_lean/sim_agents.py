# CPU check for L4: two agents A, B exchange 8 messages each (A speaks, then B); after every message both agents'
# stances are probed (spins +-1). Kinetic Ising: logit P(s_X(t+1) = +1) = 2 (h_X + Js s_X(t) + Jc s_Y(latest)).
# A primed +1, B primed -1 (and the swapped configuration; fields h = +-h0 from priming).
# Fit Jc by pooled logistic regression; null = shuffled partners (B trajectories re-paired across runs within
# configuration). S_macro(1) = I(V_t; V_t+1) - I(sA_t; V_t+1) - I(sB_t; V_t+1), V = 1[sA == sB] (a function of
# the parts), plug-in with Jeffreys pseudo-counts; null = shuffled partners.
import numpy as np, sys
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
RUNS, TURNS = 256, 8

def simulate(Js, Jc, h0):
    cfg = np.where(np.arange(RUNS) % 2 == 0, 1, -1)          # A primed +cfg, B primed -cfg
    sA = np.zeros((RUNS, 2 * TURNS + 1)); sB = np.zeros_like(sA)
    sA[:, 0] = cfg; sB[:, 0] = -cfg
    hA, hB = h0 * cfg, -h0 * cfg
    for t in range(1, 2 * TURNS + 1):
        if t % 2 == 1:   # A speaks after hearing B: A updates; B keeps its state until it speaks
            p = 1 / (1 + np.exp(-2 * (hA + Js * sA[:, t - 1] + Jc * sB[:, t - 1])))
            sA[:, t] = np.where(rng.random(RUNS) < p, 1, -1); sB[:, t] = sB[:, t - 1]
        else:
            p = 1 / (1 + np.exp(-2 * (hB + Js * sB[:, t - 1] + Jc * sA[:, t - 1])))
            sB[:, t] = np.where(rng.random(RUNS) < p, 1, -1); sA[:, t] = sA[:, t - 1]
    return sA, sB, cfg

def fit_J(sA, sB, cfg):
    X, y = [], []
    for t in range(1, sA.shape[1]):
        if t % 2 == 1: own, other, nxt, sgn = sA[:, t - 1], sB[:, t - 1], sA[:, t], cfg
        else:          own, other, nxt, sgn = sB[:, t - 1], sA[:, t - 1], sB[:, t], -cfg
        X.append(np.stack([sgn, own, other], 1)); y.append(nxt > 0)
    X = np.concatenate(X); y = np.concatenate(y)
    m = LogisticRegression(C=1e6, fit_intercept=False, max_iter=500).fit(X, y)
    return m.coef_[0] / 2           # (h, Js, Jc)

def mi(a, b):
    # plug-in MI (bits) of two binary arrays with Jeffreys pseudo-counts
    c = np.zeros((2, 2)) + 0.5
    for i in (0, 1):
        for j in (0, 1): c[i, j] += np.sum((a == i) & (b == j))
    p = c / c.sum(); pa = p.sum(1, keepdims=True); pb = p.sum(0, keepdims=True)
    return float(np.sum(p * np.log2(p / (pa * pb))))

def smacro(sA, sB):
    # use states after each full exchange (even t), lag 1 exchange
    a = (sA[:, 2::2] > 0).astype(int); b = (sB[:, 2::2] > 0).astype(int); V = (a == b).astype(int)
    Vt, Vn = V[:, :-1].ravel(), V[:, 1:].ravel()
    return mi(Vt, Vn) - mi(a[:, :-1].ravel(), Vn) - mi(b[:, :-1].ravel(), Vn)

def shuffled(sA, sB, cfg):
    idx = np.arange(RUNS)
    for c in (1, -1):
        k = np.flatnonzero(cfg == c); idx[k] = rng.permutation(k)
    return sA, sB[idx], cfg

if __name__ == '__main__':
    for Jc in [0.0, 0.1, 0.2, 0.4]:
        est, sm, z_j, z_s = [], [], [], []
        for r in range(30):
            sA, sB, cfg = simulate(0.6, Jc, 0.3)
            h, Js, J = fit_J(sA, sB, cfg); S = smacro(sA, sB)
            nullJ = [fit_J(*shuffled(sA, sB, cfg))[2] for _ in range(40)]
            nullS = [smacro(*shuffled(sA, sB, cfg)[:2]) for _ in range(40)]
            est.append(J); sm.append(S)
            z_j.append(J > np.percentile(nullJ, 95)); z_s.append(S > np.percentile(nullS, 95))
        print(f'Jc={Jc}: Jc_hat mean {np.mean(est):.3f} SD {np.std(est):.3f}; P(Jc>null q95) {np.mean(z_j):.2f}; '
              f'S_macro mean {np.mean(sm):.4f} SD {np.std(sm):.4f}; P(S>null q95) {np.mean(z_s):.2f}', flush=True)
