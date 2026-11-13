"""As sim_power.py but in log-odds space with the analysis clip at +-10 (numpy only)."""
import numpy as np
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/scratch_D_lean"); import sim_power as sp

def item_curves_l(beta, tau, g, V, eps=0.0):
    N = sp.N; KS = sp.KS
    i = np.arange(1, N + 1)
    w = np.exp(-(N - i) / tau); w = w / w.sum()
    E = (V * w).sum(-1)
    off = np.array([g[1] - g[0], g[0] - g[1]])[:, None, None]
    bonus = eps * ((KS > 0) & (KS < N / 2))[None, :, None]
    l = beta[None, None, None, :] * E[..., None] + off[..., None] + bonus[..., None]
    return np.clip(l, -10, 10).mean(axis=(-1, -2))

def phi_l(C):
    m = C.mean(axis=(0, 1)); m0 = m[0]; m1 = -m[0]
    return ((m[1:5] - m0) / (m1 - m0)).mean()

def orders_rev():
    V = -np.ones((2, 6, 2, sp.N))
    for s in range(2):
        for k in range(1, 6):
            v = -np.ones(sp.N); v[sp.rng.choice(sp.N, size=k, replace=False)] = 1.0
            V[s, k, 0] = v; V[s, k, 1] = v[::-1]
    return V

def run(n_items, b0, s_len, eps, n_sims=30, n_boot=600):
    rng = sp.rng; P0, D, S0, SD = [], [], [], []
    for _ in range(n_sims):
        Cs = np.zeros((n_items, 2, 6)); Cl = np.zeros((n_items, 2, 6))
        for it in range(n_items):
            tau = rng.uniform(1, 4); beta = np.exp(rng.normal(np.log(b0), 0.5, size=6)); g = rng.normal(0, 0.3, 2)
            V = orders_rev()
            Cs[it] = item_curves_l(beta, tau, g, V)
            Cl[it] = item_curves_l(beta * np.exp(rng.normal(0, s_len)), tau * np.exp(rng.normal(0, s_len)), g, V, eps)
        P0.append(phi_l(Cs)); D.append(phi_l(Cl) - phi_l(Cs))
        idx = rng.integers(0, n_items, size=(n_boot, n_items))
        a = np.array([phi_l(Cs[j]) for j in idx]); b = np.array([phi_l(Cl[j]) for j in idx])
        S0.append(a.std()); SD.append((b - a).std())
    return np.mean(P0), np.mean(S0), np.mean(D), np.mean(SD)

print('items b0 s_len eps | Phi_l0 SD | dPhi_l SD')
for n_items in (48,):
    for b0 in (4.0, 8.0, 16.0):
        for s_len, eps in ((0.25, 0.0), (0.1, 1.0)):
            r = run(n_items, b0, s_len, eps)
            print(f'{n_items:5d} {b0:4.1f} {s_len:4.2f} {eps:3.1f} | {r[0]:.3f} {r[1]:.4f} | {r[2]:+.4f} {r[3]:.4f}', flush=True)
