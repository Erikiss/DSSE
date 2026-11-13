"""Part R (revision), R4.2: the candidate-subspace estimator lambda_max^C (k_C = 32) under the noise model of spike_R.py.
C = orthonormal basis of span(u* (e_0), v_C, 10 high-variance and 20 bulk random directions), v_C = unit(v + 0.3 r) (the planted top
mode is known only approximately, as it would be from the loops' d_4 and the axes). Cross-fitted rightmost eigenvalue of Q'JQ
(fit half A, held-out z'M_B y / z'y with the |cos(y, z)| < 0.2 Rayleigh guard), both assignments averaged; permutation-free null =
planted lambda 0. Also the paired difference lambda_max^C - lambda_RSI (4-dim, as spike_R.py) at lambda_max 0.3, lambda_RSI 0.1.
Run: taskset -c 3 python3 -I spike2_R.py > spike2_R.json"""
import json, math
import numpy as np

rng = np.random.default_rng(11)
D = 2048
BLK = {"14": dict(pmax=4.30, pbulk=0.30, n=1792), "18": dict(pmax=1.25, pbulk=0.31, n=896), "6": dict(pmax=1.92, pbulk=0.19, n=896)}


def profile(pmax, pbulk):
    p = np.full(D, pbulk); p[:16] = np.exp(np.linspace(math.log(pmax), math.log(max(1.0, pbulk)), 16)); return np.sqrt(p)


def rightmost_cf(MA, MB):
    ev, Y = np.linalg.eig(MA); i = int(np.argmax(ev.real)); evl, Z = np.linalg.eig(MA.T); j = int(np.argmin(np.abs(evl - ev[i])))
    y, z = Y[:, i], Z[:, j]; c = abs(np.vdot(y, z)) / (np.linalg.norm(y) * np.linalg.norm(z))
    if c < 0.2: return float((np.vdot(y, MB @ y) / np.vdot(y, y)).real), y
    return float(((z @ MB @ y) / (z @ y)).real), y


def sim(b, lam, lamR, reps):
    cfg = BLK[b]; s = profile(cfg["pmax"], cfg["pbulk"]); n = cfg["n"]; var = s ** 2
    g = rng.standard_normal(D); g[:16] = 0; g /= np.linalg.norm(g); v = 0.6 * np.eye(D)[0] + 0.8 * g; v /= np.linalg.norm(v)
    g2 = rng.standard_normal(D); g2[:16] = 0; g2 -= (g2 @ v) * v; g2 /= np.linalg.norm(g2)
    vR = 0.4 * np.eye(D)[1] + math.sqrt(0.84) * g2; vR -= (vR @ v) * v; vR /= np.linalg.norm(vR)
    r = rng.standard_normal(D); r /= np.linalg.norm(r); vC = v + 0.3 * r
    hi = np.zeros((D, 10)); hi[1:16] = rng.standard_normal((15, 10)); bulk = rng.standard_normal((D, 20)); bulk[:16] = 0
    QC, _ = np.linalg.qr(np.column_stack([np.eye(D)[0], vC, hi, bulk]))
    Bq = rng.standard_normal((D, 3)); Bq[:16] = 0; QR_, _ = np.linalg.qr(np.column_stack([vR, Bq]))
    out = dict(C=[], R=[], diff=[], cos=[])
    for _ in range(reps):
        H = []
        for h in range(2):
            S = rng.standard_normal((n, D)) * s; X = rng.standard_normal((n, D)) * s
            X += lam * np.outer(S @ v, v) / (v @ (var * v)) + lamR * np.outer(S @ vR, vR) / (vR @ (var * vR))
            H.append(((X @ QC).T @ (S @ QC) / n, (X @ QR_).T @ (S @ QR_) / n))
        (CA, RA), (CB, RB) = H
        c1, y1 = rightmost_cf(CA, CB); c2, _ = rightmost_cf(CB, CA); r1, _ = rightmost_cf(RA, RB); r2, _ = rightmost_cf(RB, RA)
        lc, lr = (c1 + c2) / 2, (r1 + r2) / 2
        out["C"].append(lc); out["R"].append(lr); out["diff"].append(lc - lr)
        yv = QC @ y1.real; out["cos"].append(abs(yv @ v) / np.linalg.norm(yv))
    return out


res = {}
for b in ("14", "18", "6"):
    R = {}
    nl = sim(b, 0.0, 0.0, 60); q95 = float(np.percentile(nl["C"], 95))
    R["null"] = dict(mean=round(float(np.mean(nl["C"])), 4), sd=round(float(np.std(nl["C"])), 4), q95=round(q95, 4))
    for lam in (0.1, 0.2, 0.3, 0.5):
        o = sim(b, lam, 0.0, 30)
        R[f"lam{lam}"] = dict(mean=round(float(np.mean(o["C"])), 4), sd=round(float(np.std(o["C"])), 4), power=round(float(np.mean(np.array(o["C"]) > q95)), 3),
                              cos_top_v_median=round(float(np.median(o["cos"])), 3))
    o = sim(b, 0.3, 0.1, 30)
    R["diff lam_max 0.3 vs lam_RSI 0.1"] = dict(mean=round(float(np.mean(o["diff"])), 4), sd=round(float(np.std(o["diff"])), 4), sd_RSI=round(float(np.std(o["R"])), 4))
    res[b] = R; print(b, json.dumps(R), flush=True)
json.dump(res, open("spike2_R.json", "w"), indent=1)
