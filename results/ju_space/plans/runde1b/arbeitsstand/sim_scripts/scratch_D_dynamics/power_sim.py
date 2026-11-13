"""Power / MDE simulations for part D (numpy only; synthetic data, no model).
(1) D3 split test: within-item deviations from the item median, 2- vs 1-component Gaussian LRT (equal variances), null by
    parametric bootstrap under Gaussian items with matched pooled variance; power vs mode separation.
(2) D4 hysteresis: rate-independent component a = 2*G_slow - G_fast for a linear-lag system (a = 0) vs a bistable
    double-well system; SE from sweep noise.
(3) D1/D5 carry: SD of the pooled decay estimate lambda over 7 transitions vs the single-step K.
"""
import numpy as np

rng = np.random.default_rng(20261113)


def em2(x, iters=200):
    """equal-variance 2-component Gaussian mixture by EM; returns log-lik."""
    n = x.size
    mu1, mu2 = np.percentile(x, 25), np.percentile(x, 75)
    s2 = x.var() + 1e-12
    w = 0.5
    for _ in range(iters):
        l1 = w * np.exp(-0.5 * (x - mu1) ** 2 / s2)
        l2 = (1 - w) * np.exp(-0.5 * (x - mu2) ** 2 / s2)
        r = l1 / (l1 + l2 + 1e-300)
        w = r.mean().clip(1e-3, 1 - 1e-3)
        mu1 = (r * x).sum() / max(r.sum(), 1e-9)
        mu2 = ((1 - r) * x).sum() / max((1 - r).sum(), 1e-9)
        s2 = ((r * (x - mu1) ** 2 + (1 - r) * (x - mu2) ** 2).sum() / n) + 1e-12
    ll = np.log(w * np.exp(-0.5 * (x - mu1) ** 2 / s2) + (1 - w) * np.exp(-0.5 * (x - mu2) ** 2 / s2) + 1e-300).sum() \
        - 0.5 * n * np.log(2 * np.pi * s2)
    return ll, w, abs(mu2 - mu1) / np.sqrt(s2)


def lrt(x):
    ll1 = -0.5 * x.size * (np.log(2 * np.pi * x.var()) + 1)
    ll2, w, sep = em2(x)
    return 2 * (ll2 - ll1), min(w, 1 - w), sep


def within_item_dev(vals):
    """vals: [items, seeds]; deviations from the item median."""
    return (vals - np.median(vals, axis=1, keepdims=True)).ravel()


def sim_split(n_items, n_seeds, sep, p_items=1.0, n_rep=300, null=None):
    """sep = distance between the two modes in within-mode SD units. Each item commits each seed to mode +/- with item
    probability q_i ~ U(0.2, 0.8) (both modes populated) for a share p_items of items; other items unimodal."""
    hits = 0
    for _ in range(n_rep):
        vals = rng.normal(size=(n_items, n_seeds))
        split_items = rng.random(n_items) < p_items
        q = rng.uniform(0.2, 0.8, size=n_items)
        side = rng.random((n_items, n_seeds)) < q[:, None]
        vals = vals + np.where(split_items[:, None], np.where(side, sep / 2, -sep / 2), 0.0)
        vals = vals + rng.normal(scale=0.5, size=(n_items, 1))  # item offsets (prior), removed by centring
        stat, wmin, s = lrt(within_item_dev(vals))
        crit = null
        if stat > crit and wmin >= 0.15 and s >= 2.5:
            hits += 1
    return hits / n_rep


def null_crit(n_items, n_seeds, n_rep=600, q=0.99):
    stats = []
    for _ in range(n_rep):
        vals = rng.normal(size=(n_items, n_seeds)) + rng.normal(scale=0.5, size=(n_items, 1))
        stats.append(lrt(within_item_dev(vals))[0])
    return float(np.quantile(stats, q))


print("(1) D3 split test (LRT 2 vs 1 equal-variance Gaussians on within-item deviations; alpha 0.01; plus w_min >= 0.15, sep >= 2.5)")
for n_items, n_seeds in [(24, 10), (24, 5), (16, 16)]:
    crit = null_crit(n_items, n_seeds)
    # size check of the full rule under the null (heavy-tailed diffusion: t with 5 df)
    fp = 0
    for _ in range(300):
        vals = rng.standard_t(5, size=(n_items, n_seeds)) + rng.normal(scale=0.5, size=(n_items, 1))
        st, wmin, s = lrt(within_item_dev(vals))
        fp += (st > crit and wmin >= 0.15 and s >= 2.5)
    print(f"  items {n_items} x seeds {n_seeds}: null crit {crit:.2f}; false-positive rate under t5 diffusion {fp/300:.3f}")
    for sep in (2.0, 2.5, 3.0, 3.5, 4.0):
        for p_items in (1.0, 0.5):
            pw = sim_split(n_items, n_seeds, sep, p_items=p_items, null=crit, n_rep=200)
            print(f"    sep {sep:.1f} SD, share of splitting items {p_items:.1f}: power {pw:.2f}")

print()
print("(2) D4 hysteresis: rate-independent gap a = 2*G_slow - G_fast over middle levels h in {-0.5, 0, 0.5}")
def sweep(K, s, levels_up, sigma, bist=0.0, n=48):
    """linear first-order lag m_t = K m_{t-1} + (1-K) g(h_t) + noise, with an optional double-well term giving
    rate-independent hysteresis (bist = half-width of the bistable window in h units)."""
    gaps = []
    for _ in range(n):
        m = -s
        state = -1
        traj_up, traj_dn = {}, {}
        for h in levels_up:
            if bist > 0:
                if state < 0 and h > bist: state = 1
                if state > 0 and h < -bist: state = -1
                target = 0.9 * state
            else:
                target = s * h
            m = K * m + (1 - K) * target + rng.normal(scale=sigma)
            traj_up[h] = m
        for h in levels_up[::-1]:
            if bist > 0:
                if state < 0 and h > bist: state = 1
                if state > 0 and h < -bist: state = -1
                target = 0.9 * state
            else:
                target = s * h
            m = K * m + (1 - K) * target + rng.normal(scale=sigma)
            traj_dn[h] = m
        gaps.append(np.mean([traj_dn[h] - traj_up[h] for h in (-0.5, 0.0, 0.5)]))
    return np.array(gaps)

fast = [-1.0, -0.5, 0.0, 0.5, 1.0]
slow = list(np.round(np.arange(-1.0, 1.0001, 0.25), 2))
for sigma in (0.2, 0.3, 0.4):
    for K, label, bist in [(0.5, 'linear lag K 0.5', 0.0), (0.8, 'linear lag K 0.8', 0.0), (0.5, 'bistable window +/-0.3', 0.3)]:
        a_vals = []
        for rep in range(300):
            gf = sweep(K, 0.6, fast, sigma, bist).mean()
            gs = sweep(K, 0.6, slow, sigma, bist).mean()
            a_vals.append(2 * gs - gf)
        a_vals = np.array(a_vals)
        print(f"  sigma {sigma}: {label:28s} a mean {a_vals.mean():+.3f}, SD {a_vals.std():.3f} (48 sweeps per rate)")

print()
print("(3) carry: SD of pooled lambda (log-linear fit over turns 1..8) vs single-step K, d(1) = 8.33, SD(d(1)) = 0.53 at 128 pairs")
for lam in (0.47, 0.7, 0.9):
    for npairs in (120, 72, 48):
        sd_d = 0.53 * np.sqrt(128 / npairs)
        ks, ls = [], []
        for _ in range(2000):
            d = 8.33 * lam ** np.arange(8) + rng.normal(scale=sd_d, size=8)
            ks.append(d[1] / d[0])
            w = np.clip(d, 1e-3, None)
            ok = d > 2 * sd_d
            if ok.sum() >= 3:
                kk = np.arange(8)[ok]
                ls.append(np.exp(np.polyfit(kk, np.log(w[ok]), 1, w=np.sqrt(d[ok] / sd_d))[0]))
        print(f"  lambda {lam}, pairs {npairs}: SD single-step K {np.std(ks):.3f}; SD pooled lambda {np.std(ls):.3f}")
