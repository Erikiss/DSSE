"""Carry-ratio power for the merged part K (numpy only; no model).
(1) K6e lens frame: 192 items x 8 chain-pairs, per-pair standardised segment-1 excess d1, K = E2/E1, item bootstrap.
(2) K8b sessions: 256 sessions, ratio of sums K = sum_{n=3..8} e(n) / sum_{n=2..7} e(n), e(n) = e2 K^(n-2) + noise; z2 = e2 / SE(e2).
Rules: self-sustaining = one-sided bootstrap p(K <= 1) <= 0.05/6 (Holm step 1 of the 6-member attractor family);
dissipative = 90 % upper bound < 1; plateau = 90 % interval within [0.8, 1.25] and containing 1."""
import numpy as np
rng = np.random.default_rng(11)
LV = 0.05 / 6

def summarize(kb, K):
    lo, hi = np.percentile(kb, [5, 95])
    p = (1 + np.sum(kb <= 1)) / (1 + len(kb))
    return kb.std(), (lo <= K <= hi), p <= LV, hi < 1, (lo >= 0.8) and (hi <= 1.25) and (lo <= 1 <= hi)

def lens(d1, K, ni=192, npair=8, tau=0.5, nrep=200, nb=500):
    out = []
    for _ in range(nrep):
        u = rng.normal(0, tau, (ni, 1)); v = rng.normal(0, tau, (ni, 1))
        e1 = (d1 + u + rng.normal(0, 1, (ni, npair))).mean(1)
        e2 = (K * d1 + v + rng.normal(0, 1, (ni, npair))).mean(1)
        idx = rng.integers(0, ni, (nb, ni))
        out.append(summarize(e2[idx].mean(1) / e1[idx].mean(1), K))
    a = np.array(out, float); return a.mean(0)

def sessions(z2, K, n=256, nrep=200, nb=500, rho=0.3):
    out = []
    for _ in range(nrep):
        # per-session noise SD 1 per turn, AR(1)-correlated across turns (rho); e2 chosen so that e2 / (1/sqrt(n)) = z2
        e2 = z2 / np.sqrt(n)
        sig = np.array([e2 * K ** (t - 2) for t in range(2, 9)])
        eps = np.zeros((n, 7)); eps[:, 0] = rng.normal(0, 1, n)
        for t in range(1, 7):
            eps[:, t] = rho * eps[:, t - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, n)
        x = sig[None] + eps                     # columns turns 2..8
        idx = rng.integers(0, n, (nb, n))
        m = x[idx].mean(1)                      # nb x 7
        kb = m[:, 1:].sum(1) / m[:, :-1].sum(1)
        out.append(summarize(kb, K))
    a = np.array(out, float); return a.mean(0)

if __name__ == '__main__':
    print('(1) K6e lens frame, 192 items x 8 pairs: d1, K -> SD(K), cover90, P(self-sust. at 0.05/6), P(dissipative), P(plateau)')
    for d1 in (0.25, 0.5, 1.0):
        for K in (0.6, 0.8, 1.0, 1.25, 1.5, 1.75):
            r = lens(d1, K)
            print(f'  d1={d1:4.2f} K={K:4.2f}  SD={r[0]:.3f} cov={r[1]:.2f} sust={r[2]:.2f} diss={r[3]:.2f} plat={r[4]:.2f}', flush=True)
    print('(2) K8b sessions, 256: z2, K -> same columns')
    for z2 in (5, 10, 16):
        for K in (0.5, 0.8, 1.0, 1.25, 1.5):
            r = sessions(z2, K)
            print(f'  z2={z2:3d} K={K:4.2f}  SD={r[0]:.3f} cov={r[1]:.2f} sust={r[2]:.2f} diss={r[3]:.2f} plat={r[4]:.2f}', flush=True)
