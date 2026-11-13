"""Part R (revision), R4.2: planted-spike simulation of the cross-fitted top-mode estimator (no model run).
Noise model (stated in the spec): per-text FDT products u'x * S'u (cbar = 1) have SD p(u); B-Q7 (128 texts) gives p = 4.30 / 1.25 /
1.92 along u*, 0.30 / 0.31 / 0.19 along a random Pool direction (blocks 14 / 18 / 6). Model: X, S Gaussian, diagonal covariances in
one basis, sigma_x = sigma_S = sqrt(p); coordinate 0 = u* (p_max), coordinates 1..15 log-spaced p from p_max to 1.0 (a 16-dim
'semantic' high-variance subspace), the rest p_bulk. Signal: J = lam v (Sigma_S v)' / (v' Sigma_S v) (right eigenvector v, left
Sigma_S v: non-normal), v = unit(0.6 e_0 + 0.8 g), g a random bulk direction; generated as x = x_noise + lam v (v'S) / (v' Sigma_S v).
Halves: n_eff per half = (texts - 256 prompts) / 2 (within-prompt centring): 1,792 (block 14, 3,840 texts), 896 (18 / 6, 2,048).
Estimator: Arnoldi (k = 20, forward products X'(S y) / n) on J_A from u*, Ritz value with the largest real part, right Ritz vector y;
Arnoldi on J_A' (backward products) for the left vector z (Ritz value closest); held-out z'J_B y / (z'y) (real part), guard
|cos(y, z)| < 0.2 -> y'J_B y / (y'y); both assignments averaged. Restricted (lambda_S): Q = basis of span(v_R, 3 random bulk),
4 x 4 problem on A, held-out on B. Run: taskset -c 3 python3 -I spike_R.py > spike_R.json"""
import json, math
import numpy as np

rng = np.random.default_rng(7)
D, K = 2048, 20
BLK = {"14": dict(pmax=4.30, pbulk=0.30, n=1792), "18": dict(pmax=1.25, pbulk=0.31, n=896), "6": dict(pmax=1.92, pbulk=0.19, n=896)}


def profile(pmax, pbulk):
    p = np.full(D, pbulk); p[:16] = np.exp(np.linspace(math.log(pmax), math.log(max(1.0, pbulk)), 16)); return np.sqrt(p)


def arnoldi(mv, v0, k=K):
    V = np.zeros((D, k + 1)); H = np.zeros((k + 1, k)); V[:, 0] = v0 / np.linalg.norm(v0)
    for j in range(k):
        w = mv(V[:, j])
        for i in range(j + 1): H[i, j] = V[:, i] @ w; w = w - H[i, j] * V[:, i]
        for i in range(j + 1): c = V[:, i] @ w; H[i, j] += c; w = w - c * V[:, i]       # re-orthogonalise
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] < 1e-12: k = j + 1; break
        V[:, j + 1] = w / H[j + 1, j]
    ev, ew = np.linalg.eig(H[:k, :k]); return ev, V[:, :k] @ ew


def top_cf(XA, SA, XB, SB, start):
    nA, nB = len(XA), len(XB)
    fA = lambda y: XA.T @ (SA @ y) / nA; bA = lambda z: SA.T @ (XA @ z) / nA; fB = lambda y: XB.T @ (SB @ y) / nB
    ev, Y = arnoldi(fA, start); i = int(np.argmax(ev.real)); lam = ev[i]; y = Y[:, i]
    evl, Z = arnoldi(bA, start); j = int(np.argmin(np.abs(evl - lam))); z = Z[:, j]
    cyz = abs(np.vdot(y, z)) / (np.linalg.norm(y) * np.linalg.norm(z))
    if cyz < 0.2: val = (np.vdot(y, fB(y)) / np.vdot(y, y)).real; guard = True
    else:
        num = z @ fB(y); den = z @ y; val = (num / den).real; guard = False
    return float(val), float(lam.real), y, guard, float(cyz)


def sim(b, lam, lamR, reps):
    cfg = BLK[b]; s = profile(cfg["pmax"], cfg["pbulk"]); n = cfg["n"]; var = s ** 2
    g = rng.standard_normal(D); g[:16] = 0; g /= np.linalg.norm(g); v = 0.6 * np.eye(D)[0] + 0.8 * g; v /= np.linalg.norm(v)
    g2 = rng.standard_normal(D); g2[:16] = 0; g2 -= (g2 @ v) * v; g2 /= np.linalg.norm(g2)
    vR = 0.4 * np.eye(D)[1] + math.sqrt(1 - 0.16) * g2; vR -= (vR @ v) * v; vR /= np.linalg.norm(vR)
    Bq = rng.standard_normal((D, 3)); Bq[:16] = 0; Q, _ = np.linalg.qr(np.column_stack([vR, Bq]))
    start = np.eye(D)[0]
    out = dict(cf=[], ins=[], cos_v=[], guard=[], cyz=[], rs_cf=[])
    for _ in range(reps):
        halves = []
        for h in range(2):
            S = rng.standard_normal((n, D)) * s; X = rng.standard_normal((n, D)) * s
            X += lam * np.outer(S @ v, v) / (v @ (var * v)) + lamR * np.outer(S @ vR, vR) / (vR @ (var * vR))
            halves.append((X, S))
        (XA, SA), (XB, SB) = halves
        r1 = top_cf(XA, SA, XB, SB, start); r2 = top_cf(XB, SB, XA, SA, start)
        out["cf"].append((r1[0] + r2[0]) / 2); out["ins"].append((r1[1] + r2[1]) / 2)
        out["cos_v"].append(abs(r1[2].real @ v) / np.linalg.norm(r1[2].real)); out["guard"].append(r1[3] or r2[3]); out["cyz"].append(r1[4])
        # restricted 4 x 4 (lambda_S), cross-fitted
        vals = []
        for (Xa, Sa), (Xb, Sb) in (((XA, SA), (XB, SB)), ((XB, SB), (XA, SA))):
            MA = (Xa @ Q).T @ (Sa @ Q) / n; MB = (Xb @ Q).T @ (Sb @ Q) / n
            ev, Yr = np.linalg.eig(MA); i = int(np.argmax(ev.real)); evl, Zl = np.linalg.eig(MA.T); j = int(np.argmin(np.abs(evl - ev[i])))
            y, z = Yr[:, i], Zl[:, j]; den = z @ y
            vals.append(float(((z @ MB @ y) / den).real) if abs(den) > 0.2 * np.linalg.norm(z) * np.linalg.norm(y) else float((np.vdot(y, MB @ y) / np.vdot(y, y)).real))
        out["rs_cf"].append(float(np.mean(vals)))
    return out


res = {}
for b in ("14", "18", "6"):
    R = {}
    null = sim(b, 0.0, 0.0, 80); q95 = float(np.percentile(null["cf"], 95))
    R["null"] = dict(mean=round(float(np.mean(null["cf"])), 4), sd=round(float(np.std(null["cf"])), 4), q95=round(q95, 4),
                     insample_mean=round(float(np.mean(null["ins"])), 4), guard_share=round(float(np.mean(null["guard"])), 3),
                     restricted_mean=round(float(np.mean(null["rs_cf"])), 4), restricted_sd=round(float(np.std(null["rs_cf"])), 4))
    for lam in (0.1, 0.2, 0.3, 0.5):
        o = sim(b, lam, 0.0, 40)
        R[f"lam{lam}"] = dict(cf_mean=round(float(np.mean(o["cf"])), 4), cf_sd=round(float(np.std(o["cf"])), 4), insample_mean=round(float(np.mean(o["ins"])), 4),
                              power_vs_null_q95=round(float(np.mean(np.array(o["cf"]) > q95)), 3), cos_top_v_median=round(float(np.median(o["cos_v"])), 3),
                              guard_share=round(float(np.mean(o["guard"])), 3))
    for lr in (0.5, 1.0):
        o = sim(b, 0.3, lr, 40)
        R[f"restricted lamR{lr} (lam_max 0.3)"] = dict(mean=round(float(np.mean(o["rs_cf"])), 4), sd=round(float(np.std(o["rs_cf"])), 4))
    res[b] = R
    print(b, json.dumps(R), flush=True)
json.dump(res, open("spike_R.json", "w"), indent=1)
