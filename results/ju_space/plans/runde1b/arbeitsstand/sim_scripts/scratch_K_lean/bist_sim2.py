"""Within-item bistability rule for K4b, version 2 (numpy only).
z = late (segment-3) excess of a pushed chain-pair, standardised by the pooled SD of the twin-vs-twin different-key difference.
Bistable model: (1 - pi) N(0, 1) [relaxed; the known null] + pi N(Delta, s1^2) [locked];
monostable model: N(delta, s^2).  LR = 2 (ll_bist - ll_mono); null of LR by parametric bootstrap from the fitted monostable model
(item structure kept: item shifts re-drawn from the fitted between-item SD).  Verdict 'bistable' needs: LR p <= 0.01, Delta >= 3,
s1 <= 1.5, 0.1 <= pi <= 0.9, and mixed items (>= 30 % of items with a locked chain (z > 3) also have a relaxed chain (|z| < 1))."""
import numpy as np

rng = np.random.default_rng(5)
NI, NP = 200, 8


def ll_norm(x, m, s):
    return -0.5 * ((x - m) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)


def fit_bist(x, it=80):
    pi, D, s1 = 0.3, max(3.0, np.quantile(x, 0.9)), 1.0
    for _ in range(it):
        a = np.log(1 - pi) + ll_norm(x, 0, 1); b = np.log(pi) + ll_norm(x, D, s1)
        m = np.maximum(a, b); r = np.exp(b - m) / (np.exp(a - m) + np.exp(b - m))
        pi = float(np.clip(r.mean(), 1e-4, 1 - 1e-4)); D = float((r * x).sum() / r.sum()); s1 = float(np.sqrt((r * (x - D) ** 2).sum() / r.sum()) + 1e-3)
    a = np.log(1 - pi) + ll_norm(x, 0, 1); b = np.log(pi) + ll_norm(x, D, s1); m = np.maximum(a, b)
    return float((m + np.log(np.exp(a - m) + np.exp(b - m))).sum()), pi, D, s1


def fit_mono(x):
    return float(ll_norm(x, x.mean(), x.std() + 1e-9).sum()), x.mean(), x.std()


def lr(z):
    x = z.ravel(); lb, pi, D, s1 = fit_bist(x); lm, d, s = fit_mono(x)
    return 2 * (lb - lm), pi, D, s1, d, s


def verdict(z, n_null=100):
    L, pi, D, s1, d, s = lr(z)
    # parametric bootstrap under the monostable fit, item structure: between-item SD from item means
    im = z.mean(1); sb = max(np.var(im) - s ** 2 / NP, 0) ** 0.5; sw = max(s ** 2 - sb ** 2, 1e-6) ** 0.5
    null = []
    for _ in range(n_null):
        zz = d + rng.normal(0, sb, (NI, 1)) + rng.normal(0, sw, (NI, NP))
        null.append(lr(zz)[0])
    p = (1 + np.sum(np.array(null) >= L)) / (1 + n_null)
    lk = (z > 3).any(1); mixed = ((np.abs(z) < 1).any(1) & lk).sum() / max(lk.sum(), 1)
    return (p <= 0.01) and (D >= 3) and (s1 <= 1.5) and (0.1 <= pi <= 0.9) and (mixed >= 0.3)


def null_noise():
    return rng.normal(0, 1, (NI, NP)) * rng.lognormal(0, 0.2, (NI, 1))


def bistable(pi, delta):
    p_i = rng.beta(pi * 4, (1 - pi) * 4, (NI, 1)); lock = rng.random((NI, NP)) < p_i
    return null_noise() + lock * rng.normal(delta, 0.5, (NI, 1))


def monostable(delta, skew, het):
    d_i = rng.normal(delta, het, (NI, 1))
    if skew > 0:
        e = (rng.lognormal(0, skew, (NI, NP)) - np.exp(skew ** 2 / 2)) / np.sqrt((np.exp(skew ** 2) - 1) * np.exp(skew ** 2))
    else:
        e = rng.normal(0, 1, (NI, NP))
    return e * rng.lognormal(0, 0.2, (NI, 1)) + d_i


if __name__ == "__main__":
    R = 25
    print("power (bistable):")
    for delta in (3.0, 4.0, 5.0):
        for pi in (0.15, 0.3, 0.5, 0.8):
            print(f"  Delta={delta} pi={pi}: {np.mean([verdict(bistable(pi, delta)) for _ in range(R)]):.2f}", flush=True)
    print("false positives (monostable):")
    for delta in (0.5, 1.5, 3.0):
        for skew in (0.0, 1.0):
            for het in (0.5, 1.5):
                print(f"  shift={delta} skew={skew} item-het={het}: {np.mean([verdict(monostable(delta, skew, het)) for _ in range(R)]):.2f}", flush=True)
