# Heterogeneity check for the L3.1 stickiness test. Null: every item is memoryless (first-order Markov) but items
# differ in entry/exit rates (frailty). (a) pooled position-matched Markov surrogate with a pooled hazard fit;
# (b) item-stratified: hazard fit with item fixed effects (conditional on item) and a surrogate that simulates each
# item from its own entry/exit rates times a shared position profile (two-way fit on the odds scale).
import numpy as np, sys
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
S, T = 600, 1024
edges = np.array([0, 16, 32, 64, 128, 256, 512, 768, 1024]); B = len(edges) - 1
pbin = np.searchsorted(edges, np.arange(T), side='right') - 1

def simulate_items(enter_i, exit_i, pos_e, pos_x, gamma=0.0):
    b = np.zeros((S, T), dtype=np.int8); d = np.zeros(S)
    le = np.log(enter_i / (1 - enter_i)); lx = np.log(exit_i / (1 - exit_i))
    for t in range(1, T):
        pe = 1 / (1 + np.exp(-(le + pos_e[pbin[t]])))
        px = 1 / (1 + np.exp(-(lx + pos_x[pbin[t]] - gamma * np.log1p(d))))
        inn = b[:, t - 1] == 1; u = rng.random(S)
        b[:, t] = np.where(inn, (u >= px), (u < pe)).astype(np.int8)
        d = np.where(b[:, t] == 1, np.where(inn, d + 1, 0), 0)
    return b

def exit_rows(b):
    rows = []; d = np.zeros(S)
    for t in range(1, T):
        inn = b[:, t - 1] == 1
        if inn.any():
            it = np.flatnonzero(inn)
            rows.append(np.stack([it, np.full(len(it), pbin[t]), np.log1p(d[inn]), (b[inn, t] == 0).astype(float)], 1))
        d = np.where(b[:, t] == 1, np.where(inn, d + 1, 0), 0)
    return np.concatenate(rows)

def gamma_hat(b, item_fe):
    R = exit_rows(b); n = len(R)
    cols = [np.eye(B)[R[:, 1].astype(int)], R[:, [2]]]
    if item_fe:
        # item fixed effects (one-hot), the stratified (conditional) hazard model
        I = np.zeros((n, S)); I[np.arange(n), R[:, 0].astype(int)] = 1; cols = [I[:, 1:]] + cols
    X = np.hstack(cols)
    m = LogisticRegression(C=1e4, fit_intercept=False, max_iter=1000).fit(X, R[:, 3])
    return -m.coef_[0][-1]

def two_way_rates(b):
    # per-item and per-position-bin effects on the log-odds of entry and exit (simple alternating fit on counts)
    E = np.zeros((S, B)); En = np.zeros((S, B)); X = np.zeros((S, B)); Xn = np.zeros((S, B))
    prev, cur = b[:, :-1], b[:, 1:]; bins = pbin[1:]
    for k in range(B):
        sel = bins == k
        E[:, k] = ((prev[:, sel] == 0) & (cur[:, sel] == 1)).sum(1); En[:, k] = (prev[:, sel] == 0).sum(1)
        X[:, k] = ((prev[:, sel] == 1) & (cur[:, sel] == 0)).sum(1); Xn[:, k] = (prev[:, sel] == 1).sum(1)
    def fit(num, den):
        a = np.zeros(S); c = np.zeros(B)
        for _ in range(30):
            p = 1 / (1 + np.exp(-(a[:, None] + c[None, :])))
            a += ((num - den * p).sum(1) + 0.5 * (0.5 - 1 / (1 + np.exp(-a)))) / ((den * p * (1 - p)).sum(1) + 1.0)
            p = 1 / (1 + np.exp(-(a[:, None] + c[None, :])))
            c += (num - den * p).sum(0) / ((den * p * (1 - p)).sum(0) + 1e-6)
            c -= c.mean()
        return a, c
    ae, ce = fit(E, En); ax, cx = fit(X, Xn)
    return 1 / (1 + np.exp(-ae)), 1 / (1 + np.exp(-ax)), ce, cx

if __name__ == '__main__':
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    for gtrue in [0.0, 0.1]:
        res = {'pooled': [], 'strat': []}
        for r in range(reps):
            enter_i = np.clip(rng.beta(2, 120, S), 1e-3, 0.5)            # heterogeneous entry
            exit_i = np.clip(rng.beta(2, 6, S), 0.02, 0.95)              # heterogeneous exit (frailty)
            pos_e = np.linspace(-0.3, 0.3, B); pos_x = np.zeros(B)
            b = simulate_items(enter_i, exit_i, pos_e, pos_x, gtrue)
            # (a) pooled: pooled hazard fit vs pooled per-bin surrogate (all items share rates)
            g = gamma_hat(b, False)
            pe = np.array([((b[:, :-1] == 0) & (b[:, 1:] == 1))[:, pbin[1:] == k].sum() / max(1, (b[:, :-1] == 0)[:, pbin[1:] == k].sum()) for k in range(B)])
            px = np.array([((b[:, :-1] == 1) & (b[:, 1:] == 0))[:, pbin[1:] == k].sum() / max(1, (b[:, :-1] == 1)[:, pbin[1:] == k].sum()) for k in range(B)])
            gs = [gamma_hat(simulate_items(np.full(S, 0.5), np.full(S, 0.5), np.log(pe / (1 - pe)), np.log(px / (1 - px))), False) for _ in range(20)]
            res['pooled'].append((g, (1 + sum(x >= g for x in gs)) / 21))
            # (b) stratified: item fixed effects + item-specific two-way surrogate
            g2 = gamma_hat(b, True)
            ei, xi, ce, cx = two_way_rates(b)
            gs2 = [gamma_hat(simulate_items(ei, xi, ce, cx), True) for _ in range(20)]
            res['strat'].append((g2, (1 + sum(x >= g2 for x in gs2)) / 21))
        for k, v in res.items():
            v = np.array(v)
            print(f'gamma_true={gtrue} {k}: mean gamma_hat {v[:,0].mean():.3f}, P(p<=0.05) {np.mean(v[:,1] <= 0.05):.2f}', flush=True)
