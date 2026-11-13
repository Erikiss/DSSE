# Variant: 32 items x 2 configs x 4 seeds = 256 runs. Coupling statistic = excess over WITHIN-(item, config) re-pairing:
# J_ex = J_c(real) - mean J_c(partner series re-paired among the 4 runs of the same item and config, derangements).
# Fit with or without item fixed effects. Checks false positives at J_c = 0 with shared item fields, SE, power.
import numpy as np, sys, itertools
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
I, CFG, SEEDS, M = 32, 2, 4, 12
item = np.repeat(np.arange(I), CFG * SEEDS); cfg = np.tile(np.repeat([1, -1], SEEDS), I); R = len(item)
grp = item * 2 + (cfg < 0)
def simulate(Js, Jc, hp, sd_item):
    hi = rng.normal(0, sd_item, I)[item]
    sA = np.zeros((R, M + 1)); sB = np.zeros((R, M + 1)); sB[:, 0] = cfg
    for m in range(1, M + 1):
        if m % 2 == 1:
            p = 1 / (1 + np.exp(-2 * (hi + Js * sA[:, m - 1] + Jc * sB[:, m - 1])))
            sA[:, m] = np.where(rng.random(R) < p, 1, -1); sB[:, m] = sB[:, m - 1]
        else:
            p = 1 / (1 + np.exp(-2 * (hi + hp * cfg + Js * sB[:, m - 1] + Jc * sA[:, m - 1])))
            sB[:, m] = np.where(rng.random(R) < p, 1, -1); sA[:, m] = sA[:, m - 1]
    return sA, sB
def fit(sA, sB, it, fe):
    X, y, ii = [], [], []
    for m in range(3, M + 1, 2):
        X.append(np.stack([sA[:, m - 2], sB[:, m - 1]], 1)); y.append(sA[:, m] > 0); ii.append(it)
    X = np.concatenate(X); y = np.concatenate(y); ii = np.concatenate(ii)
    if fe:
        D = np.zeros((len(ii), ii.max() + 1)); D[np.arange(len(ii)), ii] = 1; X = np.hstack([X, D])
    return LogisticRegression(C=10.0, fit_intercept=not fe, max_iter=2000).fit(X, y).coef_[0][1] / 2
derange = [p for p in itertools.permutations(range(SEEDS)) if all(p[i] != i for i in range(SEEDS))]
def repaired(sB):
    idx = np.arange(R)
    for g in range(I * 2):
        k = np.flatnonzero(grp == g); idx[k] = k[list(derange[rng.integers(len(derange))])]
    return sB[idx]
def stat(sA, sB, it, fe, n=20):
    J = fit(sA, sB, it, fe); nul = [fit(sA, repaired(sB), it, fe) for _ in range(n)]
    return J, J - np.mean(nul), np.percentile(nul, 95)
def boot(sA, sB, fe, B=60):
    out = []
    for _ in range(B):
        pick = rng.integers(0, I, I)
        rows = np.concatenate([np.flatnonzero(item == p) for p in pick])
        newit = np.repeat(np.arange(I), CFG * SEEDS)
        # keep group structure: rows are ordered item-major with cfg/seed blocks, so grp is preserved positionally
        J = fit(sA[rows], sB[rows], newit, fe)
        nul = [fit(sA[rows], repaired(sB[rows]), newit, fe) for _ in range(6)]
        out.append(J - np.mean(nul))
    return np.percentile(out, 5), np.std(out)
if __name__ == '__main__':
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    for fe in (False, True):
        for sd_item in (0.5, 1.0):
            for Jc in (0.0, 0.1, 0.2):
                ex, fpq, lbs, ses = [], [], [], []
                for r in range(reps):
                    sA, sB = simulate(0.6, Jc, 0.5, sd_item)
                    J, Jex, q95 = stat(sA, sB, item, fe); ex.append(Jex); fpq.append(J > q95)
                    if r < 5:
                        lb, se = boot(sA, sB, fe); lbs.append(lb > 0); ses.append(se)
                print(f'FE={fe} sd_item {sd_item} Jc {Jc}: J_ex mean {np.mean(ex):+.3f} SD {np.std(ex):.3f} | '
                      f'P(J > within-item q95) {np.mean(fpq):.2f} | P(boot LB(J_ex)>0) {np.mean(lbs):.2f} (n={len(lbs)}), boot SE {np.mean(ses):.3f}', flush=True)
