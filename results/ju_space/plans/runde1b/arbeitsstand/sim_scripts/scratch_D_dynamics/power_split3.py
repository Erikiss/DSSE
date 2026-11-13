"""Vectorised version of the D3 per-item split rule simulation (numpy only; synthetic data, no model).
Per item (n seeds at turn N): 1- vs 2-component equal-variance Gaussian mixture; item 'splits' if
dBIC = BIC1 - BIC2 >= c, min weight >= 0.2 and Ashman's D = |mu1 - mu2| / s >= 2.5.
Reports per-item false-positive rates under unimodal shapes (Gaussian, t5, uniform = bounded diffusion, and a
'drifting + diffusing' Gaussian with doubled SD) and detection rates under splits, then the pooled-count rule."""
import numpy as np
from math import comb

rng = np.random.default_rng(11)


def em2_batch(X, iters=150):
    R, n = X.shape
    best_ll = np.full(R, -np.inf)
    best_w = np.zeros(R)
    best_D = np.zeros(R)
    for q in (20, 35, 50):
        mu1 = np.percentile(X, q, axis=1)
        mu2 = np.percentile(X, 100 - q, axis=1)
        mu2 = np.where(mu2 - mu1 < 1e-6, mu1 + 1e-3, mu2)
        s2 = X.var(axis=1) + 1e-12
        w = np.full(R, 0.5)
        for _ in range(iters):
            a = w[:, None] * np.exp(-0.5 * (X - mu1[:, None]) ** 2 / s2[:, None])
            b = (1 - w[:, None]) * np.exp(-0.5 * (X - mu2[:, None]) ** 2 / s2[:, None])
            r = a / (a + b + 1e-300)
            w = np.clip(r.mean(axis=1), 1e-3, 1 - 1e-3)
            mu1 = (r * X).sum(1) / np.maximum(r.sum(1), 1e-9)
            mu2 = ((1 - r) * X).sum(1) / np.maximum((1 - r).sum(1), 1e-9)
            s2 = np.maximum((r * (X - mu1[:, None]) ** 2 + (1 - r) * (X - mu2[:, None]) ** 2).sum(1) / n,
                            1e-6 * X.var(axis=1) + 1e-12)
        dens = w[:, None] * np.exp(-0.5 * (X - mu1[:, None]) ** 2 / s2[:, None]) + \
            (1 - w[:, None]) * np.exp(-0.5 * (X - mu2[:, None]) ** 2 / s2[:, None])
        ll = np.log(dens + 1e-300).sum(1) - 0.5 * n * np.log(2 * np.pi * s2)
        better = ll > best_ll
        best_ll = np.where(better, ll, best_ll)
        best_w = np.where(better, np.minimum(w, 1 - w), best_w)
        best_D = np.where(better, np.abs(mu2 - mu1) / np.sqrt(s2), best_D)
    return best_ll, best_w, best_D


def split_rate(X, c):
    R, n = X.shape
    ll1 = -0.5 * n * (np.log(2 * np.pi * X.var(axis=1)) + 1)
    ll2, wmin, D = em2_batch(X)
    dbic = (-2 * ll1 + 2 * np.log(n)) - (-2 * ll2 + 4 * np.log(n))
    return np.mean((dbic >= c) & (wmin >= 0.2) & (D >= 2.5))


def splits(R, n, sep, qlo=0.2, qhi=0.8, noise='gauss'):
    q = rng.uniform(qlo, qhi, size=(R, 1))
    side = rng.random((R, n)) < q
    e = rng.normal(size=(R, n)) if noise == 'gauss' else rng.standard_t(5, size=(R, n)) / np.sqrt(5 / 3)
    return e + np.where(side, sep / 2, -sep / 2)


R = 4000
for n in (10, 16, 20):
    print(f"seeds per item n = {n}")
    for c in (2.0, 4.0, 6.0):
        fp = {
            'gauss': split_rate(rng.normal(size=(R, n)), c),
            't5': split_rate(rng.standard_t(5, size=(R, n)), c),
            'uniform': split_rate(rng.uniform(-1, 1, size=(R, n)), c),
        }
        det = {sep: split_rate(splits(R, n, sep), c) for sep in (3.0, 4.0, 5.0, 6.0)}
        det_t = split_rate(splits(R, n, 5.0, noise='t5'), c)
        print(f"  dBIC >= {c}: per-item FP " + ", ".join(f"{k} {v:.3f}" for k, v in fp.items())
              + " | detection " + ", ".join(f"sep {k:.0f}: {v:.2f}" for k, v in det.items())
              + f" | sep 5 with t5 modes: {det_t:.2f}")


def binom_sf(k, n, p):
    return sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))


print()
for n_items, p0 in ((12, 0.03), (16, 0.03), (16, 0.05), (24, 0.05)):
    kstar = next(k for k in range(n_items + 1) if binom_sf(k, n_items, p0) <= 0.01)
    line = f"items {n_items}, per-item FP {p0}: critical count {kstar} (share {kstar/n_items:.2f}); power for per-item rate "
    line += ", ".join(f"{p1}: {binom_sf(kstar, n_items, p1):.2f}" for p1 in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8))
    print(line)
