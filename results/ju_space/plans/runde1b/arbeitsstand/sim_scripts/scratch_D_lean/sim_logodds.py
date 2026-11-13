"""D1 null envelope in log-odds space (numpy only, no model).

l = f(E): E = sum_i w_i v_i / sum_i w_i (kernel-weighted evidence; v = +1 B, -1 A),
readouts: linear l = beta*E; soft-saturating l = c*tanh(beta*E/c); hard clip at +-c; plus the analysis clip at +-10.
Phi_l = mean_{k=1..4} S_l(k/10),  S_l(a) = (lbar(a) - lbar(0)) / (lbar(1) - lbar(0)).
Orders: each random order is paired with its reverse (the planned R = 2 design).
Also: 'presence' model (concave count dependence), count/share arm kappa, release threshold.
"""
import numpy as np

rng = np.random.default_rng(7)
N = 10


def kern(n, tau, prim=0.0):
    i = np.arange(1, n + 1)
    w = np.exp(-(n - i) / tau)
    if prim:
        w = w + prim * np.exp(-(i - 1) / 1.0)
    return w / w.sum()


def readout(E, kind, beta, c):
    if kind == 'lin':
        l = beta * E
    elif kind == 'tanh':
        l = c * np.tanh(beta * E / c)
    else:
        l = np.clip(beta * E, -c, c)
    return np.clip(l, -10, 10)


def lbar(k, w, kind, beta, c, n_orders=3000, arr='int', zero=0):
    n = len(w)
    if arr == 'rel':
        v = -np.ones(n); v[n - k:] = 1 if k else -1
        if k == 0: v[:] = -1
        return readout((w * v).sum(), kind, beta, c)
    vals = []
    for _ in range(n_orders // 2):
        v = -np.ones(n)
        pos = rng.choice(n, size=k + zero, replace=False)
        v[pos[:k]] = 1.0
        v[pos[k:]] = 0.0          # neutral distractor slots
        for vv in (v, v[::-1]):
            vals.append(readout((w * vv).sum(), kind, beta, c))
    return float(np.mean(vals))


def phi(w, kind, beta, c):
    l0 = lbar(0, w, kind, beta, c); l1 = -l0
    return float(np.mean([(lbar(k, w, kind, beta, c) - l0) / (l1 - l0) for k in (1, 2, 3, 4)]))


if __name__ == '__main__':
    print('Phi_l under null readouts (linear evidence + kernel + saturation); 0.25 = proportional')
    worst = 0
    for kind in ('lin', 'tanh', 'clip'):
        for beta in (3, 8, 20):
            for c in (4, 8):
                for tau in (0.7, 1.5, 3, 1e6):
                    for prim in (0.0, 0.5):
                        p = phi(kern(N, tau, prim), kind, beta, c)
                        worst = max(worst, p)
                        if kind != 'lin' or c == 4:
                            pass
    print(f'max Phi_l over the null grid: {worst:.4f}')
    for kind, beta, c, tau in (('lin', 8, 8, 2), ('tanh', 8, 4, 2), ('tanh', 20, 8, 1.5), ('clip', 20, 4, 0.7), ('clip', 8, 8, 1e6)):
        print(f'  {kind:4s} beta {beta:2d} c {c} tau {tau:>7}: Phi_l {phi(kern(N, tau), kind, beta, c):.3f}')
    # presence model: l = beta*(g(k) - g(n-k))/g(n) with concave g(x) = log(1+x)
    print('\nPresence (concave count) model: l = beta*(log(1+k) - log(1+n-k))/log(1+n)')
    for beta in (4, 8):
        l = lambda k: beta * (np.log1p(k) - np.log1p(N - k)) / np.log1p(N)
        S = [(l(k) - l(0)) / (l(N) - l(0)) for k in (1, 2, 3, 4)]
        print(f'  beta {beta}: Phi_l {np.mean(S):.3f}')
    # count/share arm: full block (k, m, 0) vs half density (k/2, m/2, n/2 neutral)
    print('\nCount arm kappa = lbar(full)/lbar(half) at shares .2 and .4 (counting -> 2, averaging -> 1)')
    for tau in (0.7, 2, 1e6):
        w = kern(N, tau)
        for kind, beta, c in (('lin', 8, 8), ('tanh', 8, 6)):
            full = lbar(2, w, kind, beta, c) + lbar(4, w, kind, beta, c)
            half = lbar(1, w, kind, beta, c, zero=5) + lbar(2, w, kind, beta, c, zero=5)
            # averaging model: evidence normalised by relevant weight only
            vals_f, vals_h = [], []
            for (k, z, store) in ((2, 0, vals_f), (4, 0, vals_f), (1, 5, vals_h), (2, 5, vals_h)):
                acc = []
                m = (N - z) - k
                for _ in range(1500):
                    v = -np.ones(N); pos = rng.choice(N, size=k + z, replace=False)
                    v[pos[:k]] = 1.0; v[pos[k:]] = 0.0
                    for vv in (v, v[::-1]):
                        rel = (vv != 0)
                        E = (w * vv).sum() / (w * rel).sum()
                        acc.append(readout(E, kind, beta, c))
                store.append(np.mean(acc))
            print(f'  tau {tau:>7} {kind:4s}: counting kappa {full / half:.2f} | averaging kappa {sum(vals_f) / sum(vals_h):.2f}')
    # release threshold (log-odds zero crossing) depends only on the kernel
    print('\nRelease threshold alpha*_rel (zero crossing of lbar_rel) vs kernel; invariant to beta')
    for tau in (0.5, 1, 2, 4, 1e6):
        w = kern(N, tau)
        ls = [lbar(k, w, 'lin', 8, 8, arr='rel') for k in range(0, 6)]
        cross = next((k for k in range(1, 6) if ls[k] >= 0), None)
        print(f'  tau {tau:>7}: lbar_rel(k=1..5) {np.round(ls[1:], 2)} -> first k with lbar >= 0: {cross}')
