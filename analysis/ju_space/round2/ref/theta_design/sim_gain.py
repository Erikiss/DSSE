"""Planted-operator checks of the loop-gain estimators for the phase-diagram design (theta_design, simulation A).
Units: push at alpha_0 = 1 (alpha_0 * med_b * kappa_b); a = alpha / alpha_0. The expected antisymmetric read-back along a push
direction u at amplitude a is a * G(a) * w (w the response mode); G is the secant loop gain, g = 1 the critical value.
Per pair j (64 or 128 pairs, one prompt each): delta_j = a G(a) w (1 + eta_j) + sqrt(v(a)) * noise_j; eta_j prompt heterogeneity (sd h),
noise anisotropic (power-law spectrum), v(a) = a^q the growth of the sampling noise with the push (q = 0.5 saturated divergence,
q = 1 jump regime). Noise level from the earlier block-14 loop: split-half cosine ~0.4 at 32 prompts -> per-pair norm SNR ~0.2
at gain 1 ('expected'); 0.1 'pessimistic'.
A1  per-cell estimators: projected gain, naive norm gain, U-statistic norm gain, sign-flip noise floor (D = 2048, full vectors).
A2  critical alpha (g = 1) from the 4-cell alpha series with a joint prompt bootstrap; pre-registered classification rule.
A3  temperature: T_c at alpha_0 and alpha*(T) with the adaptive 2-cell design at T = 0.3 and 1.0, slope of log alpha* on log T.
A4  free estimator from the frozen loops: modal gain = (unit(d_n) . d_(n+1)) / rel_n, rel_n by the plan's prompt halvings;
    pooled converged direction u* and its attenuation m^2 = U-stat share; non-normal J (RSI start feeds the attractor).
Usage: python sim_gain.py <out.json> [reps]"""
import json, sys, time
import numpy as np

t0 = time.time(); OUT = sys.argv[1]; REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
GRID = np.array([0.5, 1.0, 2.0, 4.0]); B = 400; res = {}
log = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)


# ---------------------------------------------------------------- helpers
def crossings(a, g):
    """g: (R, C) gains on the grid a (C,). Returns (down, up): first crossing of g = 1 from above / from below as a grows, log-log
    interpolation. down = +inf if g > 1 everywhere, 0 if g < 1 everywhere; nan otherwise when no downward crossing exists."""
    lg = np.log(np.clip(g, 1e-6, None)); la = np.log(a); R, C = g.shape; dn = np.full(R, np.nan); up = np.full(R, np.nan)
    for i in range(C - 1):
        x = la[i] + (la[i + 1] - la[i]) * lg[:, i] / (lg[:, i] - lg[:, i + 1] + 1e-300)
        m = np.isnan(dn) & (lg[:, i] > 0) & (lg[:, i + 1] <= 0); dn[m] = np.exp(x[m])
        m = np.isnan(up) & (lg[:, i] < 0) & (lg[:, i + 1] >= 0); up[m] = np.exp(x[m])
    dn[np.isnan(dn) & (lg > 0).all(1)] = np.inf; dn[np.isnan(dn) & (lg < 0).all(1)] = 0.0
    return dn, up


def classify(lo, hi):
    """Pre-registered rule on per-cell 90 % bootstrap intervals [lo, hi] ordered by alpha."""
    C = len(lo); fd = any(lo[i] > 1 and hi[k] < 1 for i in range(C) for k in range(i + 1, C))
    fu = any(hi[i] < 1 and lo[k] > 1 for i in range(C) for k in range(i + 1, C))
    if fd and fu: return "found_both"
    if fd: return "found_down"
    if fu: return "found_up"
    if (hi < 1).all(): return "sub"
    if (lo > 1).all(): return "super"
    return "undecided"


def ci(x, lo=5, hi=95):
    x = x[~np.isnan(x)]
    return (float(np.percentile(x, lo)), float(np.percentile(x, hi))) if len(x) else (np.nan, np.nan)


# ---------------------------------------------------------------- planted gain curves G(a) (read-back F = a G bounded: |F| <= F_max)
def tanh_family(lam, astar):
    """Soft onset: F(a) = Fm tanh(lam a / Fm), G = F / a, G(0) = lam, G(astar) = 1 (Fm solved); readback saturates at Fm."""
    lo, hi = astar * 1.000001, 50.0 * astar
    for _ in range(200):
        Fm = 0.5 * (lo + hi)
        if Fm * np.tanh(lam * astar / Fm) > astar: hi = Fm
        else: lo = Fm
    return (lambda a, Fm=Fm: Fm * np.tanh(lam * np.asarray(a, float) / Fm) / np.asarray(a, float)), Fm
def hill_family(Fm=3.0, at=1.0):
    """Hard onset (threshold): F(a) = Fm a^3 / (at^3 + a^3): G ~ a^2 near 0 (zero state stable), rises, falls ~ Fm / a."""
    return lambda a: Fm * np.asarray(a, float) ** 2 / (at ** 3 + np.asarray(a, float) ** 3)
def first_crossing(G, up):
    a = np.exp(np.linspace(np.log(0.05), np.log(20), 200001)); g = G(a) - 1; s = np.nonzero((g[:-1] < 0) & (g[1:] >= 0) if up else (g[:-1] > 0) & (g[1:] <= 0))[0]
    return float(a[s[0]]) if len(s) else None
SCEN = {}
for nm, lam, ast in (("super_mid (lam 2, a* 1.5)", 2.0, 1.5), ("super_low (lam 1.4, a* 0.7)", 1.4, 0.7), ("super_high (lam 3, a* 3)", 3.0, 3.0)):
    G, Fm = tanh_family(lam, ast); SCEN[nm] = (G, ast, None, "found_down")
SCEN["sub (lam 0.6)"] = (lambda a: 1.0 * np.tanh(0.6 * np.asarray(a, float)) / np.asarray(a, float), None, None, "sub")
SCEN["sub_near (lam 0.85)"] = (lambda a: 3.0 * np.tanh(0.85 * np.asarray(a, float) / 3.0) / np.asarray(a, float), None, None, "sub")
_H = hill_family(); SCEN["threshold (hysteresis)"] = (_H, first_crossing(_H, False), first_crossing(_H, True), "found_both")
SCEN["flat critical (g = 1)"] = (lambda a: np.ones_like(np.asarray(a, float)), None, None, "undecided")

NOISE = {"favourable (SNR 0.2, D_eff 40)": (0.2, 40), "expected (SNR 0.2, D_eff 8)": (0.2, 8), "pessimistic (SNR 0.1, D_eff 8)": (0.1, 8)}


def scalar_experiment(G, a_cells, N, snr, deff, h, rho, q, rng, B=B):
    """Projected read-back per pair along the pushed (eigen)direction, all cells sharing prompts and keys (CRN across cells)."""
    C = len(a_cells); sp = np.sqrt(1.0 / (snr ** 2 * deff))
    eta = rng.normal(0, h, N); zc = rng.normal(size=N); z = rng.normal(size=(C, N))
    x = (G(a_cells) * a_cells)[:, None] * (1 + eta)[None] + (sp * a_cells ** (q / 2))[:, None] * (rho * zc[None] + np.sqrt(1 - rho ** 2) * z)
    g = x.mean(1) / a_cells; se = x.std(1, ddof=1) / np.sqrt(N) / a_cells
    idx = rng.integers(0, N, (B, N)); gb = x[:, idx].mean(2).T / a_cells
    return g, se, gb


# ---------------------------------------------------------------- A2: critical alpha and the rule
def run_A2():
    rng = np.random.default_rng(1); out = {}
    for nn, (snr, deff) in NOISE.items():
        for N in (64, 128):
            for q in (0.5, 1.0):
                for rho in ((0.6, 0.0) if (N == 64 and q == 0.5) else (0.6,)):
                    key = f"{nn} | N {N} | noise ~ a^{q} | CRN rho {rho}"; out[key] = {}
                    for sn, (G, adn, aup, truth) in SCEN.items():
                        cls, cov_dn, cov_up, wid, est, zcov, bias = [], [], [], [], [], [], []
                        for r in range(REPS):
                            g, se, gb = scalar_experiment(G, GRID, N, snr, deff, 0.5, rho, q, rng)
                            lo, hi = np.percentile(gb, 5, 0), np.percentile(gb, 95, 0); c = classify(lo, hi); cls.append(c)
                            zcov.append(np.mean(np.abs(g - G(GRID)) <= 1.645 * se)); bias.append(g - G(GRID))
                            dn, up = crossings(GRID, gb)
                            if adn is not None and c in ("found_down", "found_both"):
                                l, u = ci(dn); cov_dn.append(l <= adn <= u); wid.append(u / l if l > 0 else np.inf)
                                est.append(float(np.nanmedian(dn)))
                            if aup is not None and c in ("found_up", "found_both"):
                                l, u = ci(up); cov_up.append(l <= aup <= u)
                        cls = np.array(cls); d = {k: float(np.mean(cls == k)) for k in ("found_down", "found_up", "found_both", "sub", "super", "undecided")}
                        out[key][sn] = dict(truth=truth, p_correct=float(np.mean(cls == truth)), classes=d,
                                            coverage_alpha_star_90=float(np.mean(cov_dn)) if cov_dn else None, n_found=len(cov_dn),
                                            median_ci_width_factor=float(np.median(wid)) if wid else None,
                                            median_alpha_star=float(np.median(est)) if est else None, planted_alpha_star=adn,
                                            coverage_lower_crossing=float(np.mean(cov_up)) if cov_up else None,
                                            per_cell_normal_ci_coverage=float(np.mean(zcov)), per_cell_bias=np.mean(bias, 0).round(4).tolist(),
                                            per_cell_sd=np.std(np.array(bias), 0).round(4).tolist())
                    log(f"A2 {key} done")
    return out


# ---------------------------------------------------------------- A3: temperature
def loginterp(x, xs, ys):
    """log-log interpolation (linear extrapolation in log-log outside the grid) of positive ys over xs (sorted); x (R,) or scalar."""
    lx, lxs = np.log(x), np.log(xs); ly = np.log(np.clip(ys, 1e-6, None))
    i = np.clip(np.searchsorted(lxs, lx) - 1, 0, len(xs) - 2)
    t = (lx - lxs[i]) / (lxs[i + 1] - lxs[i]); return np.exp(ly[..., i] * (1 - t) + ly[..., i + 1] * t) if ly.ndim == 1 else np.exp(np.take_along_axis(ly, i[:, None], 1)[:, 0] * (1 - t) + np.take_along_axis(ly, (i + 1)[:, None], 1)[:, 0] * t)


def run_A3():
    """Temperature. Base read-back at T = 0.7: F(a) = 1.2 tanh(1.25 a) (lam 1.5, a* ~ 1.03).
    'collapse' truth: F_T(a) = F(0.7 a / T) (temperature only rescales the push: lam_T = 1.05 / T; T_c at a = 1 ~ 0.73).
    'amplitude' alternative: F_T(a) = (0.7 / T) F(a) (temperature changes the size of the read-back, not the push scale).
    Cells: T = 0.7 at a in GRID; T = 0.3 and 1.0 at a = 1, then factor-2 steps toward the crossing (2, 4 if g >= 1; 0.5, 0.25 if
    g < 1), at most 3 cells per temperature. Shared prompts and keys over all cells (joint bootstrap).
    Outputs: alpha*(T) located / covered, T_c at a = 1 (interval between bracketing temperatures and linear interpolation), and the
    collapse test: each off-series cell's read-back against the T = 0.7 curve at a' = 0.7 a / T (log-log interpolation), 90 % CI."""
    rng = np.random.default_rng(2); out = {}; Ts = (0.3, 0.7, 1.0)
    F07 = lambda a: 1.2 * np.tanh(1.25 * np.asarray(a, float))
    truths = {"collapse": lambda T, a: F07(0.7 * np.asarray(a, float) / T), "amplitude": lambda T, a: (0.7 / T) * F07(a)}
    for tn, Ft in truths.items():
        Gt = lambda T, a, Ft=Ft: Ft(T, a) / np.asarray(a, float)
        astar = {T: first_crossing(lambda a, T=T: Gt(T, a), False) for T in Ts}
        gT1 = np.array([Gt(T, 1.0) for T in Ts]); Tc_true = None
        for i in range(2):
            if (gT1[i] - 1) * (gT1[i + 1] - 1) <= 0:
                tt = np.linspace(Ts[i], Ts[i + 1], 100001); gg = np.array(Gt(tt, 1.0)) - 1; Tc_true = float(tt[np.argmin(np.abs(gg))])
        for nn, (snr, deff) in NOISE.items():
            for N in (64, 128):
                acc = dict(cov={T: [] for T in Ts}, loc={T: [] for T in Ts}, ncell={T: [] for T in Ts}, Tc_cov=[], Tc_br=[], Tc_est=[], reject=[])
                for r in range(REPS):
                    sp = np.sqrt(1.0 / (snr ** 2 * deff)); eta = rng.normal(0, 0.5, N); zc = rng.normal(size=N); idx = rng.integers(0, N, (B, N))
                    def cell(T, a):
                        x = Ft(T, a) * (1 + eta) + sp * a ** 0.25 * (0.6 * zc + 0.8 * rng.normal(size=N))
                        return x.mean(), x[idx].mean(1)                     # read-back F (push units), bootstrap replicates
                    C = {T: {} for T in Ts}
                    for a in GRID: C[0.7][a] = cell(0.7, a)
                    for T in (0.3, 1.0):
                        a = 1.0; C[T][a] = cell(T, a); up = C[T][a][0] / a >= 1
                        for _ in range(2):
                            a = a * 2 if up else a / 2; C[T][a] = cell(T, a)
                            if (C[T][a][0] / a >= 1) != up: break
                    for T in Ts:
                        aa = np.array(sorted(C[T])); gb = np.stack([C[T][a][1] / a for a in aa], 1); dn, _ = crossings(aa, gb); l, u = ci(dn)
                        acc["cov"][T].append(astar[T] is not None and l <= astar[T] <= u); acc["loc"][T].append(bool(np.isfinite(l) and np.isfinite(u) and l > 0)); acc["ncell"][T].append(len(aa))
                    g1 = np.stack([C[T][1.0][1] for T in Ts], 1); gp = np.array([C[T][1.0][0] for T in Ts])
                    def tc(gv):
                        for i in range(2):
                            if (gv[i] - 1) * (gv[i + 1] - 1) <= 0: return Ts[i] + (Ts[i + 1] - Ts[i]) * (gv[i] - 1) / (gv[i] - gv[i + 1] + 1e-12)
                        return np.nan
                    tb = np.array([tc(gv) for gv in g1]); tb = tb[np.isfinite(tb)]
                    if Tc_true is not None and len(tb) > 0.9 * B: l, u = np.percentile(tb, 5), np.percentile(tb, 95); acc["Tc_cov"].append(l <= Tc_true <= u)
                    te = tc(gp); acc["Tc_est"].append(te)
                    if Tc_true is not None: acc["Tc_br"].append(bool(np.isfinite(te)) and (0.7 <= te <= 1.0) == (0.7 <= Tc_true <= 1.0))
                    # collapse test: off-series cells against the T = 0.7 curve at a' = 0.7 a / T
                    aa = np.array(sorted(C[0.7])); Fb = np.stack([C[0.7][a][1] for a in aa], 1); diffs = []
                    for T in (0.3, 1.0):
                        for a, (fp, fb) in C[T].items():
                            ap = 0.7 * a / T
                            if ap < 0.5 or ap > 4.0: continue                 # predictions only inside the T = 0.7 grid
                            diffs.append(fb - loginterp(np.full(B, ap), aa, Fb))
                    k = len(diffs); al = 100 * 0.10 / (2 * max(k, 1))       # Bonferroni: family-wise 10 %, two-sided
                    acc["reject"].append(any(np.percentile(d, al) > 0 or np.percentile(d, 100 - al) < 0 for d in diffs)); acc.setdefault("ntest", []).append(k)
                key = f"{tn} | {nn} | N {N}"
                out[key] = dict(truth=tn, alpha_star_true={str(T): astar[T] for T in Ts}, G_at_a1={str(T): float(g) for T, g in zip(Ts, gT1)}, Tc_true=Tc_true,
                    alpha_star_located={str(T): float(np.mean(v)) for T, v in acc["loc"].items()},
                    alpha_star_coverage_90={str(T): float(np.mean(v)) for T, v in acc["cov"].items()},
                    mean_cells={str(T): float(np.mean(v)) for T, v in acc["ncell"].items()},
                    Tc_coverage_90=float(np.mean(acc["Tc_cov"])) if acc["Tc_cov"] else None, Tc_share_estimable=len(acc["Tc_cov"]) / REPS,
                    Tc_median=float(np.nanmedian(acc["Tc_est"])) if np.isfinite(acc["Tc_est"]).any() else None, Tc_right_bracket=float(np.mean(acc["Tc_br"])) if acc["Tc_br"] else None,
                    collapse_rejected=float(np.mean(acc["reject"])), collapse_cells_tested=float(np.mean(acc["ntest"])))
                log(f"A3 {key}: {out[key]}")
    return out


# ---------------------------------------------------------------- A1 + A4: full-vector checks (D = 2048)
def spectrum(D, expo=1.2):
    s2 = np.arange(1, D + 1) ** -expo; return s2 / s2.sum()


def mode_with_deff(s2, deff, rng, exclude=()):
    """Unit vector w (noise eigenbasis) with w' diag(s2) w = 1 / deff (trace 1), orthogonal to the vectors in exclude."""
    D = len(s2); r = rng.normal(size=D)
    for e in exclude: r -= (r @ e) * e
    r[0] = 0; r /= np.linalg.norm(r); base = float(r ** 2 @ s2); target = 1.0 / deff
    c = np.clip((target - base) / (s2[0] - base), 0, 1); w = np.sqrt(c) * np.eye(D)[0] + np.sqrt(1 - c) * r
    for e in exclude: w -= (w @ e) * e
    return w / np.linalg.norm(w)


def run_A1(D=2048):
    """Per-cell estimators at the four alphas along an exact eigendirection (q = 0.5, h = 0.5)."""
    rng = np.random.default_rng(3); s2 = spectrum(D); out = {}; G = tanh_family(2.0, 1.5)[0]
    for nn, (snr, deff) in NOISE.items():
        w = mode_with_deff(s2, deff, rng); sd = np.sqrt(s2 / snr ** 2); rows = {f"a={a}": dict(proj=[], naive=[], ustat=[], floor_est=[], floor_true=[], p_flip=[]) for a in GRID}
        for r in range(max(REPS // 4, 50)):
            for a in GRID:
                N = 64; eta = rng.normal(0, 0.5, N); Z = rng.normal(size=(N, D)) * (sd * a ** 0.25)
                delta = (a * G(np.array([a]))[0]) * (1 + eta)[:, None] * w[None] + Z; d = delta.mean(0)
                S = float((delta ** 2).sum()); U = (float(d @ d) * N * N - S) / (N * (N - 1))
                rr = rows[f"a={a}"]; rr["proj"].append(float(w @ d) / a); rr["naive"].append(float(np.linalg.norm(d)) / a); rr["ustat"].append(np.sqrt(max(U, 0)) / a)
                rr["floor_est"].append(np.sqrt(S) / N / a); rr["floor_true"].append(np.sqrt((a ** 0.5 / snr ** 2 + 0.25 * (a * G(np.array([a]))[0]) ** 2) / N) / a)
                E = rng.choice([-1.0, 1.0], size=(200, N)); nf = np.linalg.norm(E @ delta / N, axis=1); rr["p_flip"].append(float((1 + (nf >= np.linalg.norm(d)).sum()) / 201))
        out[nn] = {k: dict(true_gain=float(G(np.array([float(k[2:])]))[0]), proj_mean=float(np.mean(v["proj"])), proj_sd=float(np.std(v["proj"])),
                          naive_mean=float(np.mean(v["naive"])), ustat_mean=float(np.mean(v["ustat"])), ustat_sd=float(np.std(v["ustat"])),
                          floor_est=float(np.mean(v["floor_est"])), floor_true=float(np.mean(v["floor_true"])), share_p_flip_le_01=float(np.mean(np.array(v["p_flip"]) <= 0.01)))
                   for k, v in rows.items()}
        log(f"A1 {nn}: " + "; ".join(f"{k} true {v['true_gain']:.2f} proj {v['proj_mean']:.2f}+-{v['proj_sd']:.2f} naive {v['naive_mean']:.2f} U {v['ustat_mean']:.2f}" for k, v in out[nn].items()))
    return out


def halvings_rel(delta, rng, n=20):
    """rk_loop.stats(): mean cosine of half means over n random halvings, Spearman-Brown rel = 2r/(1+r), floor 0.05."""
    N = len(delta); rs = []
    for _ in range(n):
        m = np.zeros(N, bool); m[rng.permutation(N)[: N // 2]] = True; A, Bm = delta[m].mean(0), delta[~m].mean(0)
        rs.append(A @ Bm / (np.linalg.norm(A) * np.linalg.norm(Bm) + 1e-12))
    r = float(np.mean(rs)); return max(2 * r / (1 + r), 0.05) if r > -1 else 0.05


def run_A4(D=2048):
    """Normalised loop as in rk_loop (fixed push 1, direction re-derived), non-normal J: J p = lam1 (w1.p) w1 + lam2 (w2.p) w2 +
    tau (a.p) w1 + bulk; start a (the RSI axis) gives a read-back along the attractor w1 (cos(d1, start) ~ 0, as in the earlier loop).
    Free estimator per round n >= 1: ghat_n = unit(d_n).d_(n+1) / rel_n (attenuation by the overlap of unit(d_n) with the mode);
    pooled u* over five loops with m^2 = U / |d_pool|^2; then a 4-cell sweep along u* (linear regime) corrected by m^2."""
    rng = np.random.default_rng(4); s2 = spectrum(D); out = {}
    for lam1 in (0.6, 1.3):
        for nn, (snr, deff) in NOISE.items():
            w1 = mode_with_deff(s2, deff, rng); w2 = mode_with_deff(s2, 20, rng, exclude=(w1,)); ax = mode_with_deff(s2, 200, rng, exclude=(w1, w2))
            Bk = rng.normal(size=(30, D)); Bk /= np.linalg.norm(Bk, axis=1, keepdims=True); Ck = rng.normal(size=(30, D)); Ck /= np.linalg.norm(Ck, axis=1, keepdims=True); beta = rng.normal(0, 0.08, 30)
            lam2, tau = 0.5 * lam1, 1.0
            J = lambda p: (lam1 * (w1 @ p) + tau * (ax @ p)) * w1 + lam2 * (w2 @ p) * w2 + (beta * (Ck @ p)) @ Bk
            sd = np.sqrt(s2 / snr ** 2)
            def round_(p, N):
                eta = rng.normal(0, 0.5, N); return J(p)[None] * (1 + eta)[:, None] + rng.normal(size=(N, D)) * sd
            rec = dict(ghat=[], rayleigh=[], naive=[], ustat=[], rel_vs_m2=[], cos_d1_start=[], m2_true=[], m2_est=[], sweep_corr=[], sweep_raw=[])
            for r in range(max(REPS // 8, 30)):
                d4s = []; pooled = []
                for k, (start, N) in enumerate([(ax, 128), (ax, 128), (mode_with_deff(s2, 200, rng), 64), (mode_with_deff(s2, 200, rng), 64), (mode_with_deff(s2, 200, rng), 64)]):
                    p = start / np.linalg.norm(start); prev_rel = None
                    for n in range(4):
                        delta = round_(p, N); d = delta.mean(0); rel = halvings_rel(delta, rng)
                        if n == 0 and k == 0: rec["cos_d1_start"].append(float(d @ ax / np.linalg.norm(d)))
                        if n >= 1 and k == 0:
                            rec["ghat"].append(float(p @ d) / prev_rel); rec["rayleigh"].append(float(p @ d)); rec["naive"].append(float(np.linalg.norm(d)))
                            S = float((delta ** 2).sum()); U = (float(d @ d) * N * N - S) / (N * (N - 1)); rec["ustat"].append(np.sqrt(max(U, 0)))
                        if k == 0: rec["rel_vs_m2"].append((rel, float((d @ w1) ** 2 / (d @ d))))
                        prev_rel = rel; p = d / np.linalg.norm(d)
                    d4s.append(d); pooled.append(delta)
                sgn = [np.sign(x @ d4s[0]) for x in d4s]; P = np.concatenate([s * x for s, x in zip(sgn, pooled)]); dp = P.mean(0); M = len(P)
                U = (float(dp @ dp) * M * M - float((P ** 2).sum())) / (M * (M - 1)); m2e = max(U, 0) / float(dp @ dp); u = dp / np.linalg.norm(dp)
                rec["m2_true"].append(float((u @ w1) ** 2)); rec["m2_est"].append(m2e)
                sw = round_(u, 64).mean(0); rec["sweep_raw"].append(float(u @ sw)); rec.setdefault("m2_ok", []).append(m2e >= 0.5)
                if m2e >= 0.5: rec["sweep_corr"].append(float(u @ sw) / m2e)          # pre-registered: u* usable only if m^2 >= 0.5
            rv = np.array(rec["rel_vs_m2"])
            out[f"lam1 {lam1} | {nn}"] = dict(lam1=lam1, cos_d1_start_median=float(np.median(rec["cos_d1_start"])),
                free_ghat_median=float(np.median(rec["ghat"])), free_ghat_iqr=[float(np.percentile(rec["ghat"], 25)), float(np.percentile(rec["ghat"], 75))],
                rayleigh_uncorrected_median=float(np.median(rec["rayleigh"])), naive_norm_median=float(np.median(rec["naive"])), ustat_norm_median=float(np.median(rec["ustat"])),
                rel_vs_overlap2_corr=float(np.corrcoef(rv[:, 0], rv[:, 1])[0, 1]), rel_mean=float(rv[:, 0].mean()), overlap2_mean=float(rv[:, 1].mean()),
                pooled_m2_true=float(np.mean(rec["m2_true"])), pooled_m2_est=float(np.mean(rec["m2_est"])),
                sweep_raw_median=float(np.median(rec["sweep_raw"])), share_m2_ge_05=float(np.mean(rec["m2_ok"])),
                sweep_corrected_median=float(np.median(rec["sweep_corr"])) if rec["sweep_corr"] else None,
                sweep_corrected_iqr=[float(np.percentile(rec["sweep_corr"], 25)), float(np.percentile(rec["sweep_corr"], 75))] if rec["sweep_corr"] else None,
                free_ghat_rounds_within_25pct=float(np.mean(np.abs(np.array(rec["ghat"]) / lam1 - 1) <= 0.25)))
            log(f"A4 {out[f'lam1 {lam1} | {nn}']}")
    return out


if __name__ == "__main__":
    which = sys.argv[3].split(",") if len(sys.argv) > 3 else ["A1", "A2", "A3", "A4"]
    for k in which:
        res[k] = globals()[f"run_{k}"](); json.dump(res, open(OUT, "w"), indent=1, default=float)
    res["seconds"] = time.time() - t0; json.dump(res, open(OUT, "w"), indent=1, default=float); log("done")
