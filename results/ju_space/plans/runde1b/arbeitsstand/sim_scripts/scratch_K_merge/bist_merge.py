"""Within-item bistability rule (lean K4b rule, merged levels), numpy only.
Structures: K6e lens frame 192 items x 8 chain-pairs; K8b sessions 24 templates x 16 seeds.
Verdict 'bistable' = LR p <= 0.05/6 (parametric bootstrap under the fitted monostable model with between-item variance),
Delta >= 3, s1 <= 1.5, 0.1 <= pi <= 0.9, >= 30 % of items with a locked unit (z > 3) also have a relaxed unit (|z| < 1).
'no within-item bistability' = 90 % item-bootstrap upper bound of the locked share P(z > 3) < 0.10."""
import numpy as np
rng = np.random.default_rng(23)
LV = 0.05 / 6

def lln(x, m, s):
    return -0.5 * ((x - m) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)

def fit_b(x, it=60):
    pi, D, s1 = 0.3, max(3.0, np.quantile(x, 0.9)), 1.0
    for _ in range(it):
        a = np.log(1 - pi) + lln(x, 0, 1); b = np.log(pi) + lln(x, D, s1)
        m = np.maximum(a, b); r = np.exp(b - m) / (np.exp(a - m) + np.exp(b - m))
        pi = float(np.clip(r.mean(), 1e-4, 1 - 1e-4)); D = float((r * x).sum() / r.sum())
        s1 = float(np.sqrt((r * (x - D) ** 2).sum() / r.sum()) + 1e-3)
    a = np.log(1 - pi) + lln(x, 0, 1); b = np.log(pi) + lln(x, D, s1); m = np.maximum(a, b)
    return float((m + np.log(np.exp(a - m) + np.exp(b - m))).sum()), pi, D, s1

def lr(z):
    x = z.ravel(); lb, pi, D, s1 = fit_b(x); lm = float(lln(x, x.mean(), x.std() + 1e-9).sum())
    return 2 * (lb - lm), pi, D, s1, x.mean(), x.std()

def verdict(z, nnull=300):
    NI, NP = z.shape
    L, pi, D, s1, d, s = lr(z)
    im = z.mean(1); sb = max(np.var(im) - s ** 2 / NP, 0) ** 0.5; sw = max(s ** 2 - sb ** 2, 1e-6) ** 0.5
    null = [lr(d + rng.normal(0, sb, (NI, 1)) + rng.normal(0, sw, (NI, NP)))[0] for _ in range(nnull)]
    p = (1 + np.sum(np.array(null) >= L)) / (1 + nnull)
    lk = (z > 3).any(1); mixed = ((np.abs(z) < 1).any(1) & lk).sum() / max(lk.sum(), 1)
    bist = (p <= LV) and (D >= 3) and (s1 <= 1.5) and (0.1 <= pi <= 0.9) and (mixed >= 0.3)
    sh = (z > 3).mean(1)
    idx = rng.integers(0, NI, (400, NI)); ub = np.percentile(sh[idx].mean(1), 95)
    return bist, ub < 0.10

def bistable(NI, NP, pi, delta):
    p_i = rng.beta(pi * 4, (1 - pi) * 4, (NI, 1)); lock = rng.random((NI, NP)) < p_i
    return rng.normal(0, 1, (NI, NP)) * rng.lognormal(0, 0.2, (NI, 1)) + lock * rng.normal(delta, 0.5, (NI, 1))

def mono(NI, NP, delta, skew, het):
    d_i = rng.normal(delta, het, (NI, 1))
    if skew > 0:
        e = (rng.lognormal(0, skew, (NI, NP)) - np.exp(skew ** 2 / 2)) / np.sqrt((np.exp(skew ** 2) - 1) * np.exp(skew ** 2))
    else:
        e = rng.normal(0, 1, (NI, NP))
    return e * rng.lognormal(0, 0.2, (NI, 1)) + d_i

if __name__ == '__main__':
    R = 16
    for NI, NP, lab in ((192, 8, 'K6e 192x8'), (24, 16, 'K8b 24x16')):
        print(lab)
        for delta in (3.0, 4.0, 5.0):
            for pi in (0.15, 0.3, 0.5):
                v = np.array([verdict(bistable(NI, NP, pi, delta)) for _ in range(R)])
                print(f'  bistable Delta={delta} pi={pi}: P(bistable)={v[:,0].mean():.2f} P(no-bist bound)={v[:,1].mean():.2f}', flush=True)
        for delta in (0.0, 0.5, 1.5):
            for skew, het in ((0.0, 0.5), (1.0, 1.5)):
                v = np.array([verdict(mono(NI, NP, delta, skew, het)) for _ in range(R)])
                print(f'  mono shift={delta} skew={skew} het={het}: P(bistable)={v[:,0].mean():.2f} P(no-bist bound)={v[:,1].mean():.2f}', flush=True)
