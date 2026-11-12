"""Critique check (a): bias of the m*^2-corrected projected gain when J has a self-response c in the noise subspace, and the
cross-fit fix (push along u*_A from one loop, project the read-back on u*_B from an independent loop, divide by cos(u*_A, u*_B)).
Planted J = lam w1 w1' + c P_K (P_K: projector on the top-K noise eigendirections, orthogonal to w1). Normalised loops as in rk_loop
(fixed push 1, direction re-derived, 4 rounds, N pairs); u* = sign-aligned pool of the two loops' round-4 pairs (design), m*^2 =
U_pool / |d_pool|^2. Sweep cell along u* at push 1 with N pairs. Usage: python crit_bulk.py"""
import numpy as np, json, time
t0 = time.time(); rng = np.random.default_rng(5); D = 512; K = 24
s2 = np.arange(1, D + 1) ** -1.2; s2 /= s2.sum()                      # noise spectrum (trace 1), eigenbasis = coordinate axes
def mode(deff):
    r = rng.normal(size=D); r[0] = 0; r /= np.linalg.norm(r); base = float(r ** 2 @ s2); c = np.clip((1 / deff - base) / (s2[0] - base), 0, 1)
    w = np.sqrt(c) * np.eye(D)[0] + np.sqrt(1 - c) * r; return w / np.linalg.norm(w)     # as sim_gain.mode_with_deff: noise share 1/deff
def bulk_basis(w1):
    Q = np.eye(D)[:K].T.copy(); Q -= np.outer(w1, w1 @ Q); Q, _ = np.linalg.qr(Q); return Q.T   # top-K noise directions, orthogonal to w1
out = {}
for snr in (0.2, 0.1):
    for lam in (0.8, 1.2):
        for cb in (0.0, 0.3):
            w1 = mode(8); PK = bulk_basis(w1)                              # self-response block: top-K noise directions without w1
            J = lambda p: lam * (w1 @ p) * w1 + cb * (PK.T @ (PK @ p))
            sd = np.sqrt(s2) / snr                                        # per-pair noise as in sim_gain (norm SNR = snr at gain 1)
            def cell(p, N):
                return J(p)[None] * (1 + rng.normal(0, 0.5, N))[:, None] + rng.normal(size=(N, D)) * sd
            rec = dict(design=[], crossfit=[], m2=[], m2_est=[], raw=[])
            for r in range(300):
                d4s, pools = [], []
                for k in range(2):
                    p = rng.normal(size=D); p /= np.linalg.norm(p)
                    for n in range(4):
                        de = cell(p, 128); d = de.mean(0); p = d / np.linalg.norm(d)
                    d4s.append(d); pools.append(de)
                s = np.sign(d4s[0] @ d4s[1]); P = np.concatenate([pools[0], s * pools[1]]); dp = P.mean(0); M = len(P)
                U = (dp @ dp * M * M - (P ** 2).sum()) / (M * (M - 1)); m2e = max(U, 0) / (dp @ dp); u = dp / np.linalg.norm(dp)
                sw = cell(u, 128).mean(0); rec["raw"].append(float(u @ sw)); rec["m2"].append(float((u @ w1) ** 2)); rec["m2_est"].append(m2e)
                if m2e >= 0.5: rec["design"].append(float(u @ sw) / m2e)
                uA = d4s[0] / np.linalg.norm(d4s[0]); uB = s * d4s[1] / np.linalg.norm(d4s[1]); cAB = float(uA @ uB)
                if cAB > 0.3:                                             # cross-fit: push u*_A, read along u*_B
                    swA = cell(uA, 128).mean(0); rec["crossfit"].append(float(uB @ swA) / cAB)
            key = f"snr {snr} lam {lam} c {cb}"
            out[key] = dict(design_median=float(np.median(rec["design"])) if rec["design"] else None, design_n=len(rec["design"]),
                            crossfit_median=float(np.median(rec["crossfit"])) if rec["crossfit"] else None, crossfit_iqr=[float(np.percentile(rec["crossfit"], q)) for q in (25, 75)] if rec["crossfit"] else None,
                            crossfit_n=len(rec["crossfit"]), m2_true=float(np.mean(rec["m2"])), m2_est=float(np.mean(rec["m2_est"])),
                            predicted_design_bias=float(cb * (1 - np.mean(rec["m2"])) / np.mean(rec["m2"])))
            print(key, out[key], f"{time.time() - t0:.0f}s", flush=True)
json.dump(out, open("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/theta_design/crit_bulk.json", "w"), indent=1)
