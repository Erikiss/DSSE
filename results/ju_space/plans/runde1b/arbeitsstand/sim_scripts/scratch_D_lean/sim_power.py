"""Planning simulation for D1 (vectorised): sampling SD of Phi and of the paired length contrast dPhi
with item heterogeneity (numpy only; no model). Units = items (referents), bootstrap over items.

Per item i: recency time constant tau_i ~ U(1, 4) reports; per attribute j: beta_ij = exp(N(log b0, 0.5)).
Prior offset per (item, version) g ~ N(0, 0.3) log-odds (cancels under swaps).
Design: n = 10 reports, k = 0..5 B reports, 2 swaps, R random orders per (item, swap, k > 0).
'Long' = same orders, per-item null perturbation beta *= exp(N(0, s_len)), tau *= exp(N(0, s_len)),
optional H-DUAL effect eps = log-odds bonus for the minority version when 0 < k < n/2.
"""
import numpy as np

rng = np.random.default_rng(1)
N = 10
KS = np.arange(6)


def item_curves(beta, tau, g, V, eps=0.0):
    # V: [2 swaps, 6 k, R, N] in {-1,+1}; returns [2, 6] mean P_B over attributes and orders
    i = np.arange(1, N + 1)
    w = np.exp(-(N - i) / tau); w = w / w.sum()
    E = (V * w).sum(-1)                                  # [2, 6, R]
    off = np.array([g[1] - g[0], g[0] - g[1]])[:, None, None]
    bonus = eps * ((KS > 0) & (KS < N / 2))[None, :, None]
    l = beta[None, None, None, :] * E[..., None] + off[..., None] + bonus[..., None]
    return (1 / (1 + np.exp(-l))).mean(axis=(-1, -2))   # [2, 6]


def phi(C):
    # C: [items, 2, 6]; pure-A of swap s (k=0) ; pure-B = 1 - pure-A of the other swap (swap symmetry)
    m = C.mean(axis=(0, 1))                              # [6]
    m0 = m[0]; m1 = 1 - m[0]
    S = (m[1:5] - m0) / (m1 - m0)
    return S.mean()


def orders(R):
    V = -np.ones((2, 6, R, N))
    for s in range(2):
        for k in range(1, 6):
            for r in range(R):
                V[s, k, r, rng.choice(N, size=k, replace=False)] = 1.0
    return V


def run(n_items=32, R=2, b0=6.0, s_len=0.1, eps=0.0, n_sims=60, n_boot=1000):
    phis, dphis, sdp, sdd = [], [], [], []
    for _ in range(n_sims):
        Cs = np.zeros((n_items, 2, 6)); Cl = np.zeros((n_items, 2, 6))
        for it in range(n_items):
            tau = rng.uniform(1, 4); beta = np.exp(rng.normal(np.log(b0), 0.5, size=6)); g = rng.normal(0, 0.3, 2)
            V = orders(R)
            Cs[it] = item_curves(beta, tau, g, V)
            Cl[it] = item_curves(beta * np.exp(rng.normal(0, s_len)), tau * np.exp(rng.normal(0, s_len)), g, V, eps)
        p0, p1 = phi(Cs), phi(Cl)
        phis.append(p0); dphis.append(p1 - p0)
        idx = rng.integers(0, n_items, size=(n_boot, n_items))
        b0s = np.array([phi(Cs[j]) for j in idx]); b1s = np.array([phi(Cl[j]) for j in idx])
        sdp.append(b0s.std()); sdd.append((b1s - b0s).std())
    return np.mean(phis), np.mean(sdp), np.mean(dphis), np.mean(sdd), np.std(dphis)


if __name__ == '__main__':
    print('items R  b0  s_len  eps | Phi0   SD(Phi) | dPhi    SD(dPhi)boot  SD(dPhi)sims')
    for n_items in (32, 48):
        for b0 in (4.0, 8.0):
            for s_len, eps in ((0.1, 0.0), (0.25, 0.0), (0.1, 0.5), (0.1, 1.0)):
                r = run(n_items, 2, b0, s_len, eps, n_sims=40, n_boot=600)
                print(f"{n_items:5d} 2 {b0:4.1f} {s_len:5.2f} {eps:4.1f} | {r[0]:.3f} {r[1]:.4f} | {r[2]:+.4f} {r[3]:.4f} {r[4]:.4f}", flush=True)
