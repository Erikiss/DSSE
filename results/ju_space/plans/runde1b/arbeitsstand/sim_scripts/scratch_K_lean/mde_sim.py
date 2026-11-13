"""MDE checks for part K (numpy only, no model). Run: taskset -c 3 python3 -I mde_sim.py
(1) SD of the carry ratio K_H = mean(E2)/mean(E1) under an item bootstrap (200 items x 8 chain-pairs).
(2) Power of the within-item bistability rule (2-component equal-variance mixture, LR vs parametric-bootstrap null, Ashman D >= 2,
    minor weight >= 0.1) on within-item-centred late excesses.
(3) SE of a paired difference in break-off share (content / offset contrasts) and of the hazard slope beta_H.
"""
import numpy as np

rng = np.random.default_rng(7)


def k_sd(d1, K, n_items=200, n_per=8, tau=0.5, n_rep=300, n_boot=400):
    sds, cover, lo_gt1 = [], 0, 0
    for _ in range(n_rep):
        u = rng.normal(0, tau, n_items)[:, None]; v = rng.normal(0, tau, n_items)[:, None]
        e1 = d1 + u + rng.normal(0, 1, (n_items, n_per))
        e2 = K * d1 + v + rng.normal(0, 1, (n_items, n_per))
        m1, m2 = e1.mean(1), e2.mean(1)
        idx = rng.integers(0, n_items, (n_boot, n_items))
        kb = m2[idx].mean(1) / m1[idx].mean(1)
        lo, hi = np.percentile(kb, [5, 95])
        sds.append(kb.std()); cover += (lo <= K <= hi); lo_gt1 += (lo > 1)
    return float(np.mean(sds)), cover / n_rep, lo_gt1 / n_rep


def em2(x, it=60):
    """equal-variance 2-component Gaussian mixture by EM; returns loglik, weights, means, sd."""
    n = len(x); q = np.quantile(x, [0.25, 0.75]); mu = np.array(q, float); s = x.std() + 1e-9; w = np.array([0.5, 0.5])
    for _ in range(it):
        lp = np.log(w)[None] - 0.5 * ((x[:, None] - mu[None]) / s) ** 2 - np.log(s)
        m = lp.max(1, keepdims=True); r = np.exp(lp - m); r /= r.sum(1, keepdims=True)
        w = r.mean(0).clip(1e-6); mu = (r * x[:, None]).sum(0) / r.sum(0)
        s = np.sqrt((r * (x[:, None] - mu[None]) ** 2).sum() / n) + 1e-9
    lp = np.log(w)[None] - 0.5 * ((x[:, None] - mu[None]) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)
    m = lp.max(1, keepdims=True); ll = float((m[:, 0] + np.log(np.exp(lp - m).sum(1))).sum())
    return ll, w, mu, s


def ll1(x):
    s = x.std() + 1e-9
    return float((-0.5 * ((x - x.mean()) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)).sum())


def centred(vals):
    return (vals - vals.mean(1, keepdims=True)).ravel()


def lr_stat(x):
    ll2, w, mu, s = em2(x)
    D = abs(mu[1] - mu[0]) / s
    return 2 * (ll2 - ll1(x)), D, w.min()


def null_q99(n_items, n_per, n_draw=200):
    st = [lr_stat(centred(rng.normal(0, 1, (n_items, n_per))))[0] for _ in range(n_draw)]
    return float(np.quantile(st, 0.99))


def bist_power(delta, pi, n_items=200, n_per=8, n_rep=40, q99=None):
    hit = 0
    for _ in range(n_rep):
        p_i = rng.beta(pi * 4, (1 - pi) * 4, n_items)[:, None]          # item-varying locked share
        lock = rng.random((n_items, n_per)) < p_i
        x = rng.normal(0, 1, (n_items, n_per)) + delta * lock
        lr, D, wmin = lr_stat(centred(x))
        hit += (lr > q99) and (D >= 2) and (wmin >= 0.1)
    return hit / n_rep


def mono_false_pos(n_items=200, n_per=8, n_rep=40, q99=None, skew=1.0):
    """monostable but skewed (log-normal) excess: false-positive rate of the rule."""
    hit = 0
    for _ in range(n_rep):
        x = rng.lognormal(0, skew, (n_items, n_per)) + rng.normal(0, 0.3, (n_items, 1))
        x = x / x.std()
        lr, D, wmin = lr_stat(centred(x))
        hit += (lr > q99) and (D >= 2) and (wmin >= 0.1)
    return hit / n_rep


if __name__ == "__main__":
    print("(1) carry ratio K_H: d1 (per-pair standardised segment-1 excess), K -> SD, 90% coverage, P(lower bound > 1)")
    for d1 in (0.25, 0.5, 1.0):
        for K in (0.7, 1.0, 1.3, 1.5):
            sd, cov, p = k_sd(d1, K)
            print(f"  d1={d1:4.2f} K={K:3.1f}  SD={sd:.3f}  cover90={cov:.2f}  P(lo>1)={p:.2f}")
    q99 = null_q99(200, 8)
    print(f"(2) LR null q99 (200 items x 8, centred, Gaussian) = {q99:.2f}")
    for delta in (2.0, 2.5, 3.0, 4.0):
        for pi in (0.2, 0.5):
            print(f"  bistable delta={delta} pi={pi}: power={bist_power(delta, pi, q99=q99):.2f}")
    for sk in (0.5, 1.0):
        print(f"  monostable log-normal skew={sk}: false-positive rate={mono_false_pos(q99=q99, skew=sk):.2f}")
    # (3) paired break-off share differences: 600 items x 4 seeds, p ~ 0.4, within-pair corr 0.3, item ICC 0.2
    reps = []
    for _ in range(400):
        base = rng.normal(0, 0.5, (600, 1))
        a = (rng.normal(0, 1, (600, 4)) * 0.84 + base * 0.55 + 0.0) > 0.25
        b = (rng.normal(0, 1, (600, 4)) * 0.84 + base * 0.55 + 0.0) > 0.25
        reps.append((a.mean() - b.mean()))
    se = float(np.std(reps))
    print(f"(3) paired share difference SE (600 items x 4 seeds) = {se:.4f}; MDE (one-sided 5%, power 0.8) = {2.487 * se:.3f}")
    reps = []
    for _ in range(400):
        base = rng.normal(0, 0.5, (300, 1))
        a = (rng.normal(0, 1, (300, 1)) * 0.84 + base * 0.55) > 0.25
        b = (rng.normal(0, 1, (300, 1)) * 0.84 + base * 0.55) > 0.25
        reps.append((a.mean() - b.mean()))
    se = float(np.std(reps))
    print(f"    unpaired-like difference SE (300 items x 1) = {se:.4f}; MDE = {2.487 * se:.3f}")
