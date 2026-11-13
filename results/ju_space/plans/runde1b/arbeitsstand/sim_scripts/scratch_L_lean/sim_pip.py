# CPU calibration sketch for part L1 (no model). Planted codes in a synthetic background,
# n = 1,200 labelled items (600 + 600), probe = ridge score + 1-D logistic calibration,
# information I = 1 bit - held-out cross-entropy (bits), 5-fold CV.
# Question: which planted profiles can the random-fragment PIP and the knock-out
# reselection recover at this sample size?
import numpy as np, sys, json
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
N = 8192          # scaled-down neuron pool (real: 172,032)
n = 1200
R = 40            # background latent factors (shared drivers)

def background():
    Z = rng.standard_normal((n, R))
    L = rng.standard_normal((R, N)) * (np.linspace(1.5, 0.2, R)[:, None])
    E = rng.standard_normal((n, N))
    return Z @ L / np.sqrt(R) + E

def plant(X, kind, y):
    s = 2 * y - 1.0
    X = X.copy()
    info = {}
    if kind == 'localist':
        S = rng.choice(N, 20, replace=False); X[:, S] += 0.45 * s[:, None]; info['S'] = S
    elif kind == 'redundant':
        S = rng.choice(N, 2000, replace=False); X[:, S] += 0.12 * s[:, None]; info['S'] = S
    elif kind == 'dense':
        X += 0.03 * s[:, None]; info['S'] = np.arange(N)
    elif kind.startswith('threshold'):
        K = int(kind.split('_')[1])          # masks; threshold at K+1 carriers
        M = 256                              # carriers
        S = rng.choice(N, M, replace=False)
        C = rng.standard_normal((M, K))
        A = rng.standard_normal((n, K)) * 3.0   # large masks per item
        X[:, S] += 0.6 * s[:, None] + A @ C.T
        info['S'] = S; info['K'] = K
    elif kind == 'xor3':
        S = rng.choice(N, 60, replace=False)
        g = rng.standard_normal((n, 3))
        sgn = np.sign(g)
        yy = ((sgn[:, 0] * sgn[:, 1] * sgn[:, 2]) > 0).astype(int)
        for k in range(3):
            X[:, S[20 * k:20 * (k + 1)]] += 0.8 * g[:, [k]]
        info['S'] = S; info['y'] = yy
    return X, info

def info_bits(X, y, folds=5, Cs=(1e-3, 1e-2, 1e-1)):
    # ridge-type L2 logistic on standardised features; C chosen on an inner split of the training fold
    idx = rng.permutation(n); out = np.zeros(n)
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
        Xtr = (X[tr] - mu) / sd; Xte = (X[te] - mu) / sd
        # inner split for C
        k = len(tr) // 4; itr, iva = np.arange(k, len(tr)), np.arange(k)
        best, bc = 1e9, Cs[0]
        for C in Cs:
            m = LogisticRegression(C=C, max_iter=300).fit(Xtr[itr], y[tr][itr])
            p = np.clip(m.predict_proba(Xtr[iva])[:, 1], 1e-6, 1 - 1e-6)
            ce = -np.mean(y[tr][iva] * np.log2(p) + (1 - y[tr][iva]) * np.log2(1 - p))
            if ce < best: best, bc = ce, C
        m = LogisticRegression(C=bc, max_iter=300).fit(Xtr, y[tr])
        out[te] = m.predict_proba(Xte)[:, 1]
    p = np.clip(out, 1e-6, 1 - 1e-6)
    return 1.0 - np.mean(-(y * np.log2(p) + (1 - y) * np.log2(1 - p)))

def pip(X, y, sizes, draws=8):
    res = {}
    for m in sizes:
        vals = [info_bits(X[:, rng.choice(N, m, replace=False)], y) for _ in range(draws)]
        res[m] = float(np.median(vals))
    return res

def select_l1(X, y, excl, k_target=30):
    # L1 logistic selection on the S1 half (rows 0..599 after shuffling), excluding units in excl;
    # C raised until >= k_target units selected (as a sparse-set stand-in for the K pipeline)
    keep = np.setdiff1d(np.arange(N), excl)
    Xs = X[:600][:, keep]; ys = y[:600]
    mu = Xs.mean(0); sd = Xs.std(0) + 1e-9; Xs = (Xs - mu) / sd
    for C in (0.005, 0.01, 0.02, 0.05, 0.1):
        m = LogisticRegression(penalty='l1', C=C, solver='liblinear', max_iter=500).fit(Xs, ys)
        nz = np.flatnonzero(m.coef_[0] > 0)
        if len(nz) >= k_target // 2: break
    return keep[nz[np.argsort(-m.coef_[0][nz])][:k_target]]

def info_s2(X, y, S):
    if len(S) == 0: return 0.0
    Xs, ys = X[600:][:, S], y[600:]
    # reuse info_bits on the S2 half (5-fold inside S2)
    global n
    n0 = n; n = len(ys)
    v = info_bits(Xs, ys)
    n = n0
    return v

if __name__ == '__main__':
    kinds = sys.argv[2].split(',') if len(sys.argv) > 2 else ['localist', 'redundant', 'dense', 'threshold_15', 'threshold_63', 'xor3']
    sizes = [16, 64, 256, 1024, 4096, 8192]
    out = {}
    for kind in kinds:
        y = (rng.random(n) < 0.5).astype(int)
        X, info = plant(background(), kind, y)
        if 'y' in info: y = info['y']
        r = {'pip': pip(X, y, sizes)}
        r['planted_set_full'] = info_bits(X[:, info['S']], y) if len(info['S']) <= 4096 else None
        # knock-out reselection: 4 disjoint selected sets on S1, evaluated on S2
        excl = np.array([], dtype=int); sel = []
        for rr in range(4):
            S = select_l1(X, y, excl)
            sel.append({'size': int(len(S)), 'I_S2': round(info_s2(X, y, S), 4),
                        'planted_share': round(float(np.isin(S, info['S']).mean()) if len(S) else 0.0, 3)})
            excl = np.union1d(excl, S)
        r['reselect'] = sel
        out[kind] = r
        print(kind, json.dumps(r), flush=True)
