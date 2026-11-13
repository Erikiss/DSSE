# Planning check for merged L4 (C-half): A neutral, B primed (c_B = +-1), shared item field h_item on both agents.
# 64 items x 2 configs x 2 seeds = 256 runs, 12 messages (A first). Kinetic fit per listener update with item fixed effects:
# logit P(s_X(m)=+1) = 2(h0 + h_item + h_p c_X + J_s s_X(prev) + J_c s_Y(latest)).
# Checks: (1) false-positive rate of 'J_c above re-pairing null q95' and of 'item-cluster bootstrap lower bound > 0' at J_c = 0;
# (2) SE(J_c); (3) power; (4) SE of J_c(live) - J_doc (C-doc replays B's series to a neutral A as documents).
import numpy as np, sys
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
I, CFG, SEEDS, M = 64, 2, 2, 12
def runs():
    item = np.repeat(np.arange(I), CFG * SEEDS); cfg = np.tile(np.repeat([1, -1], SEEDS), I)
    return item, cfg
def simulate(Js, Jc, hp, sd_item, Jdoc=None, replay=None):
    item, cfg = runs(); R = len(item); hi = rng.normal(0, sd_item, I)[item]
    sA = np.zeros((R, M + 1)); sB = np.zeros((R, M + 1)); sB[:, 0] = cfg  # B's prior state = priming sign
    sA[:, 0] = 0
    for m in range(1, M + 1):
        if m % 2 == 1:   # A speaks (neutral, no priming field)
            partner = (replay[:, m - 1] if replay is not None else sB[:, m - 1])
            J = Jc if Jdoc is None else Jdoc
            p = 1 / (1 + np.exp(-2 * (hi + Js * sA[:, m - 1] + J * partner)))
            sA[:, m] = np.where(rng.random(R) < p, 1, -1); sB[:, m] = (replay[:, m] if replay is not None else sB[:, m - 1])
        else:            # B speaks (primed)
            if replay is not None:
                sB[:, m] = replay[:, m]; sA[:, m] = sA[:, m - 1]; continue
            p = 1 / (1 + np.exp(-2 * (hi + hp * cfg + Js * sB[:, m - 1] + Jc * sA[:, m - 1])))
            sB[:, m] = np.where(rng.random(R) < p, 1, -1); sA[:, m] = sA[:, m - 1]
    return sA, sB, cfg, item
def design_A(sA, sB, item):
    # A's updates at m = 3, 5, ..., 11 (own previous defined, partner's latest = B at m-1)
    X, y, it = [], [], []
    for m in range(3, M + 1, 2):
        X.append(np.stack([sA[:, m - 2], sB[:, m - 1]], 1)); y.append(sA[:, m] > 0); it.append(item)
    return np.concatenate(X), np.concatenate(y), np.concatenate(it)
def fit_Jc(sA, sB, item, fe=True):
    X, y, it = design_A(sA, sB, item)
    if fe:
        D = np.zeros((len(it), I)); D[np.arange(len(it)), it] = 1
        X = np.hstack([X, D])
    m = LogisticRegression(C=10.0, fit_intercept=not fe, max_iter=2000).fit(X, y)
    return m.coef_[0][1] / 2
def repair(sB, cfg):
    idx = np.arange(len(cfg))
    for c in (1, -1):
        k = np.flatnonzero(cfg == c); idx[k] = rng.permutation(k)
    return sB[idx]
def boot_lb(sA, sB, item, B=150):
    est = []
    for _ in range(B):
        pick = rng.integers(0, I, I)
        rows = np.concatenate([np.flatnonzero(item == p) for p in pick])
        # relabel items to keep FE identifiable per draw
        newitem = np.repeat(np.arange(I), CFG * SEEDS)
        est.append(fit_Jc(sA[rows], sB[rows], newitem))
    return np.percentile(est, 5), np.std(est)
if __name__ == '__main__':
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    for sd_item in (0.5, 1.0):
        for Jc in (0.0, 0.1, 0.2):
            ests, fp_null, fp_boot, ses = [], [], [], []
            for r in range(reps):
                sA, sB, cfg, item = simulate(0.6, Jc, 0.5, sd_item)
                J = fit_Jc(sA, sB, item); ests.append(J)
                nulls = [fit_Jc(sA, repair(sB, cfg), item) for _ in range(40)]
                fp_null.append(J > np.percentile(nulls, 95))
                if r < 6:
                    lb, se = boot_lb(sA, sB, item, 80); fp_boot.append(lb > 0); ses.append(se)
            print(f'sd_item {sd_item} Jc {Jc}: mean Jc_hat {np.mean(ests):+.3f} SD {np.std(ests):.3f} | '
                  f'P(J > repair-null q95) {np.mean(fp_null):.2f} | P(boot LB>0) {np.mean(fp_boot):.2f} (n={len(fp_boot)}) boot SE {np.mean(ses):.3f}', flush=True)
    # C-doc contrast: same items, live C-half vs document replay of B's series, J_doc = J_live (pure lexical)
    for Jc in (0.1, 0.2):
        d = []
        for r in range(reps):
            sA, sB, cfg, item = simulate(0.6, Jc, 0.5, 0.5)
            sA2, sB2, _, _ = simulate(0.6, Jc, 0.5, 0.5, Jdoc=Jc, replay=sB)
            d.append(fit_Jc(sA, sB, item) - fit_Jc(sA2, sB2, item))
        print(f'J_live - J_doc (both {Jc}): mean {np.mean(d):+.3f} SD {np.std(d):.3f}  MDE(2.49 SD) {2.49*np.std(d):.3f}', flush=True)
