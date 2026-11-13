"""Small numerical checks behind the part-L design (no model). Run: taskset -c 3 python3 -I checks_L.py"""
import numpy as np
rng = np.random.default_rng(0)

print('== 1. rank tests with C controls and Holm ==')
for C in (50, 99):
    pmin = 1.0 / (C + 1)
    for m in (2, 3, 4):
        print(f'C={C}: min rank p={pmin:.4f}; Holm first threshold for m={m}: {0.05/m:.4f} -> attainable={pmin <= 0.05/m}')

print('\n== 2. plug-in joint-state log-loss asymmetry equals marginal entropy difference ==')
def H(p):
    p = p[p > 0]; return -(p * np.log(p)).sum()
k = 4; S = 5
x = rng.integers(0, S, size=20000)
# make it non-stationary: drift in the marginal over time
x[10000:] = np.minimum(S - 1, x[10000:] + (rng.random(10000) < 0.3))
lag = 3
a, b = x[:-lag], x[lag:]
J = np.zeros((S, S)); np.add.at(J, (a, b), 1); J /= J.sum()
Hjoint = H(J.ravel()); Ha = H(J.sum(1)); Hb = H(J.sum(0))
Hfwd = Hjoint - Ha   # H(S_{t+k} | S_t)
Hbwd = Hjoint - Hb   # H(S_t | S_{t+k}) = backward prediction of the earlier state
print(f'H_fwd={Hfwd:.5f} H_bwd={Hbwd:.5f} diff={Hbwd-Hfwd:.5f}; H(S_t)-H(S_t+k)={Ha-Hb:.5f} (identical by construction)')

print('\n== 3. stationary 3-cycle with net flux: equal conditional entropies, positive S_dot ==')
P = np.array([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])  # rows: from, cols: to
w, v = np.linalg.eig(P.T); pi = np.real(v[:, np.argmin(abs(w - 1))]); pi /= pi.sum()
Jt = pi[:, None] * P               # joint P(s_t=i, s_t+1=j)
Hf = H(Jt.ravel()) - H(Jt.sum(1)); Hb_ = H(Jt.ravel()) - H(Jt.sum(0))
Sdot = np.nansum(Jt * np.log(Jt / Jt.T))
print(f'pi={np.round(pi,3)} H_fwd={Hf:.5f} H_bwd={Hb_:.5f} S_dot={Sdot:.4f} nats/step (>0: broken detailed balance)')

print('\n== 4. integral fluctuation theorem for token reversal, exact enumeration on a toy AR model ==')
V, n, T = 6, 4, 1.0
W1 = rng.normal(size=(V, V)) * 1.5; b0 = rng.normal(size=V)
def logp_seq(seq, T):
    lp = 0.0; prev = None
    for t, tok in enumerate(seq):
        logits = b0.copy() if prev is None else W1[prev] + 0.3 * t
        logits = logits / T
        lp += logits[tok] - np.log(np.exp(logits - logits.max()).sum()) - logits.max()
        prev = tok
    return lp
import itertools
tot = 0.0; ex = 0.0
for seq in itertools.product(range(V), repeat=n):
    lpf = logp_seq(seq, T); lpb = logp_seq(seq[::-1], T)
    tot += np.exp(lpf); ex += np.exp(lpf) * np.exp(lpb - lpf)
print(f'sum P = {tot:.6f}; E[exp(-sigma)] = {ex:.6f} (exactly 1 for a bijective reversal and samples from P)')
# Monte Carlo estimate at moderate sample size, to show heavy-tail bias at low T
for T in (0.7, 3.0):
    seqs = []
    for _ in range(4000):
        prev = None; seq = []
        for t in range(n):
            logits = (b0.copy() if prev is None else W1[prev] + 0.3 * t) / T
            p = np.exp(logits - logits.max()); p /= p.sum()
            tok = rng.choice(V, p=p); seq.append(tok); prev = tok
        seqs.append(tuple(seq))
    sig = np.array([logp_seq(s, T) - logp_seq(s[::-1], T) for s in seqs])
    print(f'T={T}: MC mean exp(-sigma) over 4000 samples = {np.exp(-sig).mean():.3f}; mean sigma = {sig.mean():.3f}')

print('\n== 5. modularity split vs leading eigenvector of a positive correlation matrix ==')
N = 40; g = np.array([0] * 20 + [1] * 20)
C = np.full((N, N), 0.25); C[np.ix_(g == 0, g == 0)] = 0.45; C[np.ix_(g == 1, g == 1)] = 0.45; np.fill_diagonal(C, 1.0)
def power(M, it=2000):
    v = rng.normal(size=M.shape[0])
    shift = np.abs(M).sum(1).max()
    for _ in range(it):
        v = M @ v + shift * v; v /= np.linalg.norm(v)
    return v
v1 = power(C)
A = C.copy(); np.fill_diagonal(A, 0); A = np.maximum(A, 0); kdeg = A.sum(1); m2 = kdeg.sum()
B = A - np.outer(kdeg, kdeg) / m2
vb = power(B)
print('sign split of leading eigvec of C:', np.unique(np.sign(v1), return_counts=True))
agree = max(np.mean((vb > 0) == (g == 0)), np.mean((vb > 0) == (g == 1)))
print(f'modularity split agreement with planted groups: {agree:.2f}')

print('\n== 6. frailty: mixture of constant exit hazards gives a declining population hazard ==')
def dwells(h, n):
    return rng.geometric(h, size=n)
d = np.concatenate([dwells(0.5, 20000), dwells(0.05, 20000)])
for dd in (1, 2, 4, 8, 16):
    at_risk = (d >= dd).sum(); exits = (d == dd).sum()
    print(f'dwell {dd:2d}: population hazard {exits/at_risk:.3f}')
print('-> a first-order Markov surrogate pooled over rollouts would show a flat hazard; a surrogate matched on rollout occupancy tercile absorbs most of this decline (checked below)')
occ = np.concatenate([np.full(20000, 0), np.full(20000, 1)])  # proxy: occupancy stratum
for s in (0, 1):
    ds = d[occ == s]
    hz = [((ds == dd).sum() / (ds >= dd).sum()) for dd in (1, 2, 4, 8)]
    print(f'stratum {s}: hazards {np.round(hz,3)} (flat within stratum)')

print('\n== 7. planning MDEs ==')
# L3 contrast: per-token SD of a 64-unit magnetisation difference between two texts, base rate 5%, collective factor 2
sd_tok = np.sqrt(2 * 0.05 * 0.95 / 64) * 2
n_eff = 64 / 10.0      # 64 post-release tokens, autocorrelation time ~10
sd_traj = sd_tok / np.sqrt(n_eff)
sd_prompt = sd_traj / np.sqrt(2)          # 2 seeds
sd_contrast = np.sqrt(sd_prompt ** 2 + sd_prompt ** 2 / 4)
se = sd_contrast / np.sqrt(200)
print(f'L3: sd per token {sd_tok:.4f}, per trajectory {sd_traj:.4f}, per prompt contrast {sd_contrast:.4f}, SE over 200 prompts {se:.5f}, MDE (1.645+0.84) x SE = {(1.645+0.84)*se:.4f} (fraction of units on; base rate 0.05)')
# L8 coupling: SE of J in logit = 2(h + J s): coefficient on s is 2J
n_trans, deff = 128 * 11, 2.5
se_2J = np.sqrt(1.0 / (n_trans * 0.25)) * np.sqrt(deff)
se_J = se_2J / 2
se_dd = se_J * 2  # difference of differences ~ 4 independent SEs combined: sqrt(4)*se
print(f'L8: SE(J) per condition ~{se_J:.3f}; SE of difference-in-differences ~{se_dd:.3f}; MDE ~{2.485*se_dd:.3f}')
# L2: SE of logit transition probability with ~1.5e5 tokens per stratum and occupancy 10%
n_in = 0.1 * 7.5e5 / 10    # transitions from H-state, ESS-corrected (tau 10)
p_stay = 0.8
se_logit = 1 / np.sqrt(n_in * p_stay * (1 - p_stay))
print(f'L2: SE(logit P(stay)) ~{se_logit:.3f} (ESS-corrected); HD resolution is limited by the spread of the 99 controls, not by sampling')
# L1: ESS requirement
for N in (32, 64, 128):
    params = N * (N + 1) // 2
    for tau in (5, 10, 20, 40):
        ess = 7.7e5 / tau
        print(f'L1: N={N:3d} params={params:5d} tau={tau:2d} ESS={ess:8.0f} ESS/params={ess/params:6.1f}')
