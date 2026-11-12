"""Critique check (b2): is the gain of crit_tline.py from the regression rule or from the extra temperatures? Same WLS T_c estimator
on 3 temperatures (0.3, 0.7, 1.0; 64, 128, 64 pairs; = the design's alpha_0 cells) vs 5 temperatures (0.3, 0.5, 0.7, 0.85, 1.0;
64, 64, 128, 64, 64) vs 3 temperatures at 128 pairs each (0.3, 0.7, 1.0; the middle gate branch). Usage: python crit_tline3.py"""
import numpy as np, json
rng = np.random.default_rng(13); R, B = 600, 1000; out = {}
def run(Tl, ns, G07, k, s, rho=0.3):
    Tl = np.array(Tl); C = len(Tl); sd = np.array([s / np.sqrt(n / 64) for n in ns]); cov = rho * np.outer(sd, sd) + (1 - rho) * np.diag(sd ** 2)
    lg = (np.log(G07) - k * (Tl - 0.7))[None] + (np.linalg.cholesky(cov) @ rng.normal(size=(C, R))).T
    X = np.stack([np.ones(C), Tl - 0.7], 1); Wi = np.linalg.inv(cov); Vb = np.linalg.inv(X.T @ Wi @ X); beta = lg @ (Vb @ X.T @ Wi).T; Lb = np.linalg.cholesky(Vb)
    Tc = 0.7 + np.log(G07) / k; res = []; cov_ok = []
    for r in range(R):
        bb = beta[r][None] + (Lb @ rng.normal(size=(2, B))).T; grid = np.linspace(0.3, 1.0, 15); band = bb[:, :1] + bb[:, 1:] * (grid[None] - 0.7)
        if (np.percentile(band, 5, 0) > 0).all(): res.append("above"); continue
        if (np.percentile(band, 95, 0) < 0).all(): res.append("below"); continue
        tc = np.where(bb[:, 1] < 0, 0.7 - bb[:, 0] / bb[:, 1], np.nan)
        if np.mean(np.isfinite(tc)) >= 0.95:
            l, u = np.nanpercentile(tc, 5), np.nanpercentile(tc, 95)
            if l >= 0.2 and u <= 1.1: res.append("located"); cov_ok.append(l <= Tc <= u); continue
        res.append("undecided")
    res = np.array(res); return float(np.mean(res == "located")), (float(np.mean(cov_ok)) if cov_ok else None)
for s in (0.10, 0.16, 0.30):
    for G07 in (0.85, 1.0, 1.2):
        for k in (1.0, 2.0):
            a = run([0.3, 0.7, 1.0], [64, 128, 64], G07, k, s); b = run([0.3, 0.5, 0.7, 0.85, 1.0], [64, 64, 128, 64, 64], G07, k, s); c = run([0.3, 0.7, 1.0], [128, 128, 128], G07, k, s)
            key = f"s {s} G07 {G07} k {k}"; out[key] = dict(three_64=a, five_64=b, three_128=c)
            print(key.ljust(24), "3T(64,128,64) loc %.2f cov %s | 5T loc %.2f cov %s | 3T(128x3) loc %.2f cov %s" % (a[0], a[1] and round(a[1], 2), b[0], b[1] and round(b[1], 2), c[0], c[1] and round(c[1], 2)), flush=True)
json.dump(out, open("crit_tline3.json", "w"), indent=1)
