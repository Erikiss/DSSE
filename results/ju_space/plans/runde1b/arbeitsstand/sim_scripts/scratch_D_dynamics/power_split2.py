"""D3 split rule, per item: 1- vs 2-component equal-variance Gaussian mixture on the seeds of one item at turn N.
Item 'splits' if dBIC = BIC1 - BIC2 >= c, min weight >= 0.2 and Ashman's D = |mu1 - mu2| / s >= 2.5 (equal variances).
Report the per-item false-positive rate under unimodal shapes (Gaussian, t5, uniform = bounded diffusion) and the per-item
detection rate under splits; then the power of the pooled rule 'share of splitting items S >= S_crit' over n_items items."""
import numpy as np
from math import comb

rng = np.random.default_rng(7)


def fit2(x, iters=300):
    n = x.size
    best = None
    for q in (20, 35, 50):
        mu1, mu2 = np.percentile(x, q), np.percentile(x, 100 - q)
        if mu1 == mu2:
            mu2 = mu1 + 1e-3
        s2 = x.var() + 1e-12
        w = 0.5
        for _ in range(iters):
            a = w * np.exp(-0.5 * (x - mu1) ** 2 / s2)
            b = (1 - w) * np.exp(-0.5 * (x - mu2) ** 2 / s2)
            r = a / (a + b + 1e-300)
            w = float(np.clip(r.mean(), 1e-3, 1 - 1e-3))
            mu1 = (r * x).sum() / max(r.sum(), 1e-9)
            mu2 = ((1 - r) * x).sum() / max((1 - r).sum(), 1e-9)
            s2 = max(((r * (x - mu1) ** 2 + (1 - r) * (x - mu2) ** 2).sum() / n), 1e-6 * x.var() + 1e-12)
        ll = (np.log(w * np.exp(-0.5 * (x - mu1) ** 2 / s2) + (1 - w) * np.exp(-0.5 * (x - mu2) ** 2 / s2) + 1e-300)
              - 0.5 * np.log(2 * np.pi * s2)).sum()
        if best is None or ll > best[0]:
            best = (ll, min(w, 1 - w), abs(mu2 - mu1) / np.sqrt(s2))
    return best


def item_splits(x, c):
    n = x.size
    ll1 = -0.5 * n * (np.log(2 * np.pi * x.var()) + 1)
    ll2, wmin, D = fit2(x)
    bic1 = -2 * ll1 + 2 * np.log(n)
    bic2 = -2 * ll2 + 4 * np.log(n)
    return (bic1 - bic2 >= c) and wmin >= 0.2 and D >= 2.5


def rate(gen, n, c, reps=1500):
    return np.mean([item_splits(gen(n), c) for _ in range(reps)])


gens_null = {
    'gauss': lambda n: rng.normal(size=n),
    't5': lambda n: rng.standard_t(5, size=n),
    'uniform': lambda n: rng.uniform(-1, 1, size=n),
}


def gen_split(sep, qlo=0.2, qhi=0.8):
    def g(n):
        q = rng.uniform(qlo, qhi)
        side = rng.random(n) < q
        return rng.normal(size=n) + np.where(side, sep / 2, -sep / 2)
    return g


for n in (10, 16, 20):
    print(f"seeds per item n = {n}")
    for c in (2.0, 4.0, 6.0):
        fps = {k: rate(g, n, c) for k, g in gens_null.items()}
        dets = {sep: rate(gen_split(sep), n, c, reps=800) for sep in (3.0, 4.0, 5.0, 6.0)}
        print(f"  dBIC >= {c}: per-item FP " + ", ".join(f"{k} {v:.3f}" for k, v in fps.items())
              + " | detection " + ", ".join(f"sep {k:.0f}: {v:.2f}" for k, v in dets.items()))

# pooled rule: n_items items, each splits independently with prob p_item; null FP p0. Critical count k* with
# P(Bin(n_items, p0) >= k*) <= 0.01; power = P(Bin(n_items, p1) >= k*)
def binom_sf(k, n, p):
    return sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))


print()
for n_items, p0 in ((12, 0.05), (16, 0.05), (24, 0.05)):
    kstar = next(k for k in range(n_items + 1) if binom_sf(k, n_items, p0) <= 0.01)
    line = f"items {n_items}, null per-item FP {p0}: critical count {kstar} (share {kstar/n_items:.2f}); power for p_item = "
    line += ", ".join(f"{p1}: {binom_sf(kstar, n_items, p1):.2f}" for p1 in (0.2, 0.3, 0.4, 0.5, 0.6))
    print(line)
