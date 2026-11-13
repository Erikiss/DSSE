"""Psi (Rosas et al. 2020, eq. 10a) for a two-agent macro V = XNOR(s_A, s_B) (supervenient on the parts).
Checks: (1) independent but persistent agents give Psi > 0 (spurious 'emergence' from the synergistic macro);
(2) cross-pairing surrogates (A from dialogue i, B from dialogue j) reproduce that value, so Delta Psi = Psi(real) - Psi(cross) ~ 0;
(3) coupled agents (B copies A with prob c) give Delta Psi > 0; (4) noise/MDE at the planned sample size.
numpy only; no model.
"""
import numpy as np

def mi_binary(x, y, pc=0.5):
    # plug-in MI in bits with Jeffreys-type pseudo-counts
    t = np.zeros((2, 2)) + pc
    np.add.at(t, (x, y), 1)
    p = t / t.sum()
    px = p.sum(1, keepdims=True); py = p.sum(0, keepdims=True)
    return float((p * np.log2(p / (px * py))).sum())

def simulate(n_dlg, n_rounds, persist, couple, rng):
    # spins in {0,1}; A starts random, B starts random; per round each agent keeps its spin with prob persist,
    # B (and A) copies the partner's previous spin with prob couple, otherwise flips at random
    sA = np.zeros((n_dlg, n_rounds), int); sB = np.zeros((n_dlg, n_rounds), int)
    sA[:, 0] = rng.integers(0, 2, n_dlg); sB[:, 0] = rng.integers(0, 2, n_dlg)
    for r in range(1, n_rounds):
        for s_self, s_oth in ((sA, sB), (sB, sA)):
            u = rng.random(n_dlg)
            new = rng.integers(0, 2, n_dlg)
            copy = u < couple
            keep = (~copy) & (u < couple + (1 - couple) * persist)
            new = np.where(copy, s_oth[:, r - 1], new)
            new = np.where(keep, s_self[:, r - 1], new)
            s_self[:, r] = new
    return sA, sB

def psi(sA, sB):
    V = (sA == sB).astype(int)
    v0 = V[:, :-1].ravel(); v1 = V[:, 1:].ravel()
    a0 = sA[:, :-1].ravel(); b0 = sB[:, :-1].ravel()
    return mi_binary(v0, v1) - mi_binary(a0, v1) - mi_binary(b0, v1)

def cross_psi(sA, sB, rng, n_perm=39):
    out = []
    n = sA.shape[0]
    for _ in range(n_perm):
        perm = rng.permutation(n)
        while np.any(perm == np.arange(n)):
            perm = rng.permutation(n)
        out.append(psi(sA, sB[perm]))
    return np.array(out)

rng = np.random.default_rng(5)
n_dlg, n_rounds = 64, 8  # per condition: 32 settings x 2 reps; 8 rounds -> 7 transitions
for persist in (0.0, 0.6, 0.9):
    for couple in (0.0, 0.1, 0.2, 0.3):
        reals, deltas = [], []
        for rep in range(200):
            sA, sB = simulate(n_dlg, n_rounds, persist, couple, rng)
            pr = psi(sA, sB); pc = cross_psi(sA, sB, rng, 19).mean()
            reals.append(pr); deltas.append(pr - pc)
        reals = np.array(reals); deltas = np.array(deltas)
        print(f"persist {persist:.1f} couple {couple:.1f}: Psi real mean {reals.mean():+.3f} sd {reals.std():.3f} | "
              f"DeltaPsi mean {deltas.mean():+.3f} sd {deltas.std():.3f} | P(Delta>0 by 1.645 sd) {np.mean(deltas > 1.645*deltas.std()):.2f}")

# pooled over 3 stance conditions (192 dialogues)
print('--- pooled 192 dialogues, 8 rounds')
for persist, couple in ((0.6, 0.0), (0.6, 0.1), (0.6, 0.15), (0.9, 0.0), (0.9, 0.1)):
    deltas = []
    for rep in range(200):
        sA, sB = simulate(192, 8, persist, couple, rng)
        deltas.append(psi(sA, sB) - cross_psi(sA, sB, rng, 19).mean())
    deltas = np.array(deltas)
    print(f"persist {persist} couple {couple}: DeltaPsi mean {deltas.mean():+.4f} sd {deltas.std():.4f} MDE(2.49sd) {2.49*deltas.std():.4f}")
