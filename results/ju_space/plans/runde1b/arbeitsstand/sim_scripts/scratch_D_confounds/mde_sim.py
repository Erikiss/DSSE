"""Planning MDEs for part D (synthetic numbers only; no model). Bonferroni k = 6 primary contrasts, one-sided 0.05/6,
power 0.8: factor 2.394 + 0.842 = 3.236 x SD(estimator); secondary contrasts: 1.645 + 0.842 = 2.487 x SD."""
import numpy as np
rng = np.random.default_rng(7)
Z1, Z2 = 3.236, 2.487
def boot_sd(fn, data, B=2000):
    n = len(data[0]); vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n); vals.append(fn(*[d[idx] for d in data]))
    return float(np.std(vals))
def sim_lambda(n, span1, span4, sdI, rho, reps=60):
    sds = []
    for _ in range(reps):
        s1 = np.clip(rng.normal(span1, 0.3 * span1, n), 1, None); s4 = np.clip(rng.normal(span4, 0.3 * span4, n), 0.5, None)
        z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n) * sdI
        I1, I4 = z[:, 0], z[:, 1]
        f = lambda a, b, c, d: a.sum() / b.sum() - c.sum() / d.sum()
        sds.append(boot_sd(f, (I4, s4, I1, s1), B=400))
    return float(np.mean(sds))
print('D3a Lambda_int (kappa units), n items, span L1/L4 nats, per-item interaction SD 1.0 nat, rho 0.5:')
for n in [24, 32]:
    for s1, s4 in [(10, 5), (10, 3), (6, 2)]:
        sd = sim_lambda(n, s1, s4, 1.0, 0.5)
        print(f'   n {n} span {s1}/{s4}: SD {sd:.3f}  MDE primary {Z1*sd:.3f}  90% half-width {1.645*sd:.3f}')
# D2-R slope of iota over log2 L (3 L values 1,3,5 doublings from 1k: 2k,8k,32k), 16 items, 3 alphas pooled
def sim_slope(n, sd_iota=0.12, reps=200):
    x = np.array([1.0, 3.0, 5.0]); xs = x - x.mean()
    sl = []
    for _ in range(reps):
        y = rng.normal(0, sd_iota, (n, 3, 3))      # items x alphas x L (kappa units per item)
        s = (y * xs).sum(-1) / (xs ** 2).sum()     # per item, per alpha slope
        sl.append(s.mean())
    return float(np.std(sl))
for sd_i in [0.08, 0.12, 0.2]:
    sd = sim_slope(16, sd_i)
    print(f'D2R pooled slope per doubling (16 items x 3 alphas, per-item iota SD {sd_i}): SD {sd:.4f} MDE primary {Z1*sd:.4f}/doubling = {4*Z1*sd:.3f} over 2k->32k')
# D4 catfish: per-item mean over 4 attributes x 2 versions x 2 arrangements; per-item SD of the paired difference
for sd_item in [0.6, 0.8, 1.2]:
    sd = sd_item / np.sqrt(24)
    print(f'D4 catfish c_4 (24 items, per-item SD {sd_item} nats): SD {sd:.3f}  MDE primary {Z1*sd:.2f} nats; 90% half-width {1.645*sd:.2f}')
# D3b M2: Lambda_S per start (24 starts), SD of per-start difference in D units
for sd_s in [0.4, 0.6, 1.0]:
    sd = sd_s / np.sqrt(24)
    print(f'D3b M2 Lambda_S (24 starts, per-start SD {sd_s}): MDE primary {Z1*sd:.2f}; 90% half-width {1.645*sd:.2f}')
# D5 pi_opp: per start fraction of 16 seeds beyond the halfway point; 16 starts (8 per pole); difference 16k - 1k paired by start
def sim_pi(p1, p4, het=0.08, nst=16, nseed=16, reps=4000):
    d = []
    for _ in range(reps):
        q1 = np.clip(p1 + rng.normal(0, het, nst), 0, 1); q4 = np.clip(p4 + rng.normal(0, het, nst), 0, 1)
        d.append((rng.binomial(nseed, q4) / nseed - rng.binomial(nseed, q1) / nseed).mean())
    return float(np.std(d))
for p1 in [0.02, 0.1]:
    sd = sim_pi(p1, p1)
    print(f'D5 pi_opp difference (16 starts x 16 seeds, base {p1}): SD {sd:.3f} MDE primary {Z1*sd:.3f}')
# D6 RMST paired difference, 24 starts; per-start SD of the difference in tokens
for sd_t in [600, 900, 1200]:
    sd = sd_t / np.sqrt(24)
    print(f'D6 RMST difference (24 starts, per-start SD {sd_t} tokens): MDE primary {Z1*sd:.0f} tokens; 90% half-width {1.645*sd:.0f}')
