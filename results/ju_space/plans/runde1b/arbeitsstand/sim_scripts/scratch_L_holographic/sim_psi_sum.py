"""Psi for a continuous pair macro V = y_A + y_B (supervenient), Gaussian-copula bivariate MI on ranks.
Agents: y(t+1) = phi*y(t) + c*y_other(t) + noise. c > 0 conformity (redundancy), c < 0 complementarity.
Delta Psi = Psi(real) - mean Psi(cross-paired). Also the kinetic coupling estimate (OLS slope on partner).
Planned size: 192 dialogues (3 stance conditions x 32 settings x 2 reps) x 8 rounds (7 transitions). numpy/scipy only.
"""
import numpy as np
from scipy.stats import rankdata, norm

def gc(x):
    return norm.ppf(rankdata(x) / (len(x) + 1))

def mi_gc(x, y):
    r = np.corrcoef(gc(x), gc(y))[0, 1]
    return -0.5 * np.log(1 - r * r)

def sim(n, T, phi, c, rng, sd0=1.0):
    yA = np.zeros((n, T)); yB = np.zeros((n, T))
    yA[:, 0] = rng.normal(0, sd0, n); yB[:, 0] = rng.normal(0, sd0, n)
    for t in range(1, T):
        eA = rng.normal(0, 1, n); eB = rng.normal(0, 1, n)
        yA[:, t] = phi * yA[:, t-1] + c * yB[:, t-1] + eA
        yB[:, t] = phi * yB[:, t-1] + c * yA[:, t-1] + eB
    return yA, yB

def psi(yA, yB):
    V = yA + yB
    v0 = V[:, :-1].ravel(); v1 = V[:, 1:].ravel()
    return mi_gc(v0, v1) - mi_gc(yA[:, :-1].ravel(), v1) - mi_gc(yB[:, :-1].ravel(), v1)

def cross(yA, yB, rng, k=19):
    n = yA.shape[0]; out = []
    for _ in range(k):
        p = rng.permutation(n)
        while np.any(p == np.arange(n)):
            p = rng.permutation(n)
        out.append(psi(yA, yB[p]))
    return np.mean(out)

def jhat(yA, yB):
    # OLS of yA(t+1) on [yA(t), yB(t)] via normal equations would invert; use the forward-stagewise residual slope instead:
    # slope of residual of yA(t+1) after regressing on yA(t) (simple slope, cov/var) against residual of yB(t) on yA(t).
    a0 = yA[:, :-1].ravel(); a1 = yA[:, 1:].ravel(); b0 = yB[:, :-1].ravel()
    ra = a1 - (np.cov(a1, a0)[0, 1] / np.var(a0, ddof=1)) * a0
    rb = b0 - (np.cov(b0, a0)[0, 1] / np.var(a0, ddof=1)) * a0
    return np.cov(ra, rb)[0, 1] / np.var(rb, ddof=1)

rng = np.random.default_rng(11)
for phi in (0.5, 0.8):
    for c in (-0.3, -0.15, 0.0, 0.15, 0.3):
        if abs(phi) + abs(c) >= 1: continue
        d, j, jn = [], [], []
        for rep in range(300):
            yA, yB = sim(192, 8, phi, c, rng)
            d.append(psi(yA, yB) - cross(yA, yB, rng, 9))
            j.append(jhat(yA, yB))
            p = rng.permutation(192); jn.append(jhat(yA, yB[p]))
        d = np.array(d); j = np.array(j); jn = np.array(jn)
        print(f"phi {phi} c {c:+.2f}: DeltaPsi mean {d.mean():+.4f} sd {d.std():.4f} | J_hat mean {j.mean():+.3f} sd {j.std():.3f} | J_null sd {jn.std():.3f} MDE_J(2.49sd) {2.49*jn.std():.3f}")
