"""Check of two stickiness statistics on synthetic series (numpy only, no model).
(a) Markov-hazard gamma: series with dwell-dependent exit hazard (sticky) vs first-order Markov (memoryless) at matched occupancy;
    the position-matched Markov surrogate must give gamma ~ 0; the sticky series gamma < 0.
(b) Planning SD of the prompt-level gamma difference at the TB-L size (64 prompts x 4 trajectories x ~960 tokens).
"""
import numpy as np

def gen_markov(n, T, p01, p10, rng):
    b = np.zeros((n, T), int); b[:, 0] = rng.random(n) < p01 / (p01 + p10)
    u = rng.random((n, T))
    for t in range(1, T):
        b[:, t] = np.where(b[:, t-1] == 1, u[:, t] >= p10, u[:, t] < p01)
    return b

def gen_sticky(n, T, p01, h0, k, rng):
    # exit hazard h(d) = h0 / (1 + k*d): falls with dwell d (stickiness); k = 0 -> Markov
    b = np.zeros((n, T), int); d = np.zeros(n)
    u = rng.random((n, T))
    for t in range(1, T):
        haz = h0 / (1 + k * d)
        stay1 = (b[:, t-1] == 1) & (u[:, t] >= haz)
        enter = (b[:, t-1] == 0) & (u[:, t] < p01)
        b[:, t] = (stay1 | enter).astype(int)
        d = np.where(b[:, t] == 1, np.where(b[:, t-1] == 1, d + 1, 0), 0)
    return b

def gamma_hat(b):
    # logistic regression of exit (b_{t+1}=0 | b_t=1) on log(1+d) by Newton-free gradient ascent (forward fit)
    X, y = [], []
    for row in b:
        d = 0
        for t in range(len(row) - 1):
            if row[t] == 1:
                X.append(np.log1p(d)); y.append(1 - row[t+1]); d += 1
            else:
                d = 0
    X = np.array(X); y = np.array(y)
    if len(y) < 50: return np.nan
    a, g = 0.0, 0.0
    for _ in range(400):
        z = a + g * X; p = 1 / (1 + np.exp(-z))
        a += 0.5 * np.mean(y - p); g += 0.5 * np.mean((y - p) * X)
    return g

rng = np.random.default_rng(3)
n, T = 4, 960   # one prompt: 4 trajectories
for k in (0.0, 0.1, 0.3):
    gs, gsur = [], []
    for rep in range(64):     # 64 prompts
        b = gen_sticky(n, T, 0.03, 0.12, k, rng)
        occ = b.mean(); p10 = np.mean((b[:, :-1] == 1) & (b[:, 1:] == 0)) / max(np.mean(b[:, :-1] == 1), 1e-9)
        p01 = np.mean((b[:, :-1] == 0) & (b[:, 1:] == 1)) / max(np.mean(b[:, :-1] == 0), 1e-9)
        gs.append(gamma_hat(b)); gsur.append(gamma_hat(gen_markov(n, T, p01, p10, rng)))
    gs = np.array(gs); gsur = np.array(gsur); dlt = gs - gsur
    print(f"k {k}: gamma mean {np.nanmean(gs):+.3f}, Markov-surrogate gamma {np.nanmean(gsur):+.3f}, prompt-level SD of difference {np.nanstd(dlt):.3f}, "
          f"SE over 64 prompts {np.nanstd(dlt)/8:.3f}, MDE(2.49 SE) {2.49*np.nanstd(dlt)/8:.3f}")
