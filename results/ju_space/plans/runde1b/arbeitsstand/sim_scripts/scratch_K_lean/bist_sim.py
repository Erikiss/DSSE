"""Within-item bistability rule for K4b (numpy only). Late excess z (pushed minus twin, standardised by the pooled SD of the
twin-vs-twin different-key difference of the same stratum). Rule (all three):
 (i)  locked share P(z > 3) minus the null's P(z > 3): item-bootstrap 90 % lower bound > 0.10;
 (ii) trough: density per unit z in [1, 2) below the density in [-1, 1) and below the density in [3, 5); both differences' 90 % lower
      bounds > 0;
 (iii) mixed items: among items with >= 1 locked chain (z > 3), share with >= 1 relaxed chain (|z| < 1) >= 0.30.
Checks power (bistable mixtures) and false positives (monostable shifted / skewed / heterogeneous)."""
import numpy as np

rng = np.random.default_rng(11)
NI, NP = 200, 8


def rule(z, n_boot=400):
    # z: [NI, NP]
    def stats(zz):
        lk = (zz > 3).mean(); dl = ((zz >= -1) & (zz < 1)).mean() / 2; dm = ((zz >= 1) & (zz < 2)).mean() / 1
        dh = ((zz >= 3) & (zz < 5)).mean() / 2
        return lk, dl - dm, dh - dm
    p_null = 0.00135
    idx = rng.integers(0, NI, (n_boot, NI))
    bs = np.array([stats(z[i]) for i in idx])
    lo = np.percentile(bs, 5, axis=0)
    c1 = lo[0] - p_null > 0.10
    c2 = (lo[1] > 0) and (lo[2] > 0)
    locked_items = (z > 3).any(1); mixed = ((np.abs(z) < 1).any(1) & locked_items).sum() / max(locked_items.sum(), 1)
    c3 = mixed >= 0.30
    return c1 and c2 and c3


def null_noise():
    sig = rng.lognormal(0, 0.2, (NI, 1))
    return rng.normal(0, 1, (NI, NP)) * sig


def bistable(pi, delta):
    p_i = rng.beta(pi * 4, (1 - pi) * 4, (NI, 1)); lock = rng.random((NI, NP)) < p_i
    lev = rng.normal(delta, 0.5, (NI, 1))
    return null_noise() + lock * lev


def monostable(delta, skew):
    d_i = rng.normal(delta, 0.5, (NI, 1))
    noise = (rng.lognormal(0, skew, (NI, NP)) - np.exp(skew ** 2 / 2)) / np.sqrt((np.exp(skew ** 2) - 1) * np.exp(skew ** 2)) if skew > 0 else rng.normal(0, 1, (NI, NP))
    return noise * rng.lognormal(0, 0.2, (NI, 1)) + d_i


if __name__ == "__main__":
    R = 60
    print("power (bistable):")
    for delta in (3.0, 4.0, 5.0, 6.0):
        for pi in (0.15, 0.3, 0.5, 0.8):
            print(f"  delta={delta} pi={pi}: {np.mean([rule(bistable(pi, delta)) for _ in range(R)]):.2f}")
    print("false positives (monostable):")
    for delta in (0.0, 1.0, 2.0, 3.0, 4.0):
        for skew in (0.0, 0.5, 1.0):
            print(f"  shift={delta} skew={skew}: {np.mean([rule(monostable(delta, skew)) for _ in range(R)]):.2f}")
