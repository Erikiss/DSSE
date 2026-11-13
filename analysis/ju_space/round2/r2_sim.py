"""Round 2 simulation ports (implementer 5; final plan results/ju_space/plans/abschnitt11_12_runde2_plan.md, 9.1, AT7, OD-B14).
Import only; `python r2_sim.py selftest` checks the ports against the reference files of Appendix E on their own small examples (the
references are looked up in round2/ref/<relative path>, then $R2REF, then the design scratchpad; only their definitions are executed,
so their absolute import paths do not matter). r2_toy_common's selftest runs these checks too and writes them into its comparison output.

1. RGI toy rule (AT7 / AT8): reg2_verdict and reg2_simulate = rgi_design/crit/sim_reg2.py's verdict2 and run2 with crit/sim_reg.py's
   COND and ci (two-step sham-controlled rule; per world OLS of a line value on x- = min(x, 0), x+ = max(x, 0), p and k, full model with
   the k x x- and k x x+ interactions, reduced model without the k terms; world-level t 95 % intervals). reg2_noise estimates AT7's
   noise components and design grid from AT2's per-model values.
2. Theta power simulation (OD-B14): gain_A2 = theta_design/sim_gain.py run_A2 with its helpers (crossings, classify, ci, tanh_family,
   hill_family, first_crossing, SCEN, scalar_experiment) generalised to measured per-cell SDs of the per-pair projections, a measured
   correlation across cells (shared keys) and per-cell pair counts; resolvable_band gives the largest interval around 1 of flat gains g
   for which the rule returns SUB or SUPER in < 80 % of replicates. run_A2_port(reps) reproduces run_A2 exactly (same generator use).
"""
from __future__ import annotations

import ast
import os
import sys

import numpy as np
from scipy import stats

# ====================================================================================================================== RGI rule
COND = [("nat", 0.0, 0.0, 0), ("task+.01", -2.1, 0.32, 1), ("task-.01", 1.6, 0.31, 1), ("task-.03", 2.26, 0.43, 1),
        ("sham+.01", 0.55, 0.33, 0), ("sham-.01", -1.0, 0.31, 0), ("sham-.03", -1.6, 0.43, 0), ("sham+.03", 0.65, 0.48, 0), ("sham-.1", -2.2, 0.54, 0)]
COND2 = COND + [("sham2+", 1.0, 0.45, 0)]          # sim_reg2's option: a second off-target arm that inflates delta about x2

SCENARIOS = {"perturbation only (pilot-like, -0.6 per unit p)": lambda x, p, k: 0.82 - 0.6 * p,
             "perturbation + collapse under task squeeze (pilot)": lambda x, p, k: 0.82 - 0.6 * p - 0.5 * (k == 1 and x < 0),
             "true growth 0.05/doubling above natural + perturbation": lambda x, p, k: 0.82 - 0.6 * p + 0.05 * max(x, 0),
             "true growth 0.10/doubling above natural + perturbation": lambda x, p, k: 0.82 - 0.6 * p + 0.10 * max(x, 0),
             "needs delta 0.10/doubling below natural + perturbation": lambda x, p, k: 0.82 - 0.6 * p + 0.10 * min(x, 0),
             "erodes -0.07/doubling above, needs 0.15 below, no perturbation": lambda x, p, k: 0.82 + 0.15 * min(x, 0) - 0.07 * max(x, 0)}


def ci(v):
    """crit/sim_reg.ci: (mean, lower, upper) of the two-sided 95 % t interval (df = n - 1)."""
    n = len(v); m = v.mean(); h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def fit(X, Y):
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def reg2_design(x, p, k):
    """Design matrix of the full model per world: columns [1, x-, x+, p, k, k x-, k x+] (the reduced model = the first 4 columns)."""
    x, p, k = (np.asarray(a, float) for a in (x, p, k)); xm, xp = np.minimum(x, 0), np.maximum(x, 0)
    return np.column_stack([np.ones_like(x), xm, xp, p, k, k * xm, k * xp])


def reg2_fit_world(x, p, k, y):
    """(full coefficients [a, b-, b+, e, c0, d-, d+], reduced coefficients [a, b-, b+, e]) of one world's models."""
    X = reg2_design(x, p, k); y = np.asarray(y, float)
    return fit(X, y), fit(X[:, :4], y)


def reg2_verdict(Bf, Br):
    """crit/sim_reg2.verdict2 on per-world coefficients Bf [n_worlds, 7] (full) and Br [n_worlds, 4] (reduced). Step 1:
    'regulariser-specific (not delta)' if the t interval of d- or d+ excludes 0 with |mean| >= 0.05; step 2 (reduced model): GROWS
    (b+ lower bound > 0, mean >= 0.05) / ERODES (b+ upper < 0, mean <= -0.05) / FLAT beyond (b+ interval inside (-0.05, 0.05)), plus NEEDS
    (b- lower > 0, mean >= 0.05); 'undecided' otherwise. (AT8's extra labels 'perturbation-driven' and ABSENT are in reg2_labels.)"""
    Bf, Br = np.asarray(Bf, float), np.asarray(Br, float)
    dm, dp = ci(Bf[:, 5]), ci(Bf[:, 6]); bm, bp = ci(Br[:, 1]), ci(Br[:, 2])
    if (abs(dm[0]) >= 0.05 and (dm[1] > 0 or dm[2] < 0)) or (abs(dp[0]) >= 0.05 and (dp[1] > 0 or dp[2] < 0)):
        return "regulariser-specific (not delta)"
    out = []
    if bp[1] > 0 and bp[0] >= 0.05:
        out.append("GROWS")
    elif bp[2] < 0 and bp[0] <= -0.05:
        out.append("ERODES")
    elif -0.05 < bp[1] and bp[2] < 0.05:
        out.append("FLAT beyond")
    if bm[1] > 0 and bm[0] >= 0.05:
        out.append("NEEDS")
    return " + ".join(out) if out else "undecided"


def reg2_labels(Br, cond_means_ci_upper=None):
    """AT8's labels beside the verdict: 'perturbation-driven' if the upper bound of e (reduced model) < 0; ABSENT if every condition's
    mean has t upper bound < 0.05 (pass the per-condition upper bounds). Returns dict."""
    e = ci(np.asarray(Br, float)[:, 3])
    out = dict(perturbation_driven=bool(e[2] < 0), e=e)
    if cond_means_ci_upper is not None:
        out["absent"] = bool(np.all(np.asarray(cond_means_ci_upper) < 0.05))
    return out


def reg2_simulate(f, cond=COND, NW=8, ninit=3, nsim=1500, s_w=0.05, s_wc=0.04, s_m=0.05, seed=1, jit=0.15, pjit=0.03, top=None, ndigits=None):
    """crit/sim_reg2.run2: nsim simulated studies of NW worlds x conditions (name, x, p, k) x ninit models; world effect N(0, s_w),
    world x condition N(0, s_wc), model N(0, s_m); x jittered N(0, jit) and p N(0, pjit) (clipped at 0) except for the condition named
    'nat' (x = p = 0); y = f(x, p, k) + noise; per world the full and reduced OLS; reg2_verdict. Returns {verdict: rate} sorted by rate
    (top=4, ndigits=3 reproduce the reference's output exactly)."""
    rng = np.random.default_rng(seed); cnt = {}
    for _ in range(nsim):
        Bf, Br = [], []
        for w in range(NW):
            uw = rng.normal(0, s_w); X, Y = [], []
            for name, x, p, k in cond:
                ucw = rng.normal(0, s_wc)
                for i in range(ninit):
                    xx = x + rng.normal(0, jit) if name != "nat" else 0.0; pp = max(p + rng.normal(0, pjit), 0) if name != "nat" else 0.0
                    X.append([1, min(xx, 0), max(xx, 0), pp, k, k * min(xx, 0), k * max(xx, 0)]); Y.append(f(xx, pp, k) + uw + ucw + rng.normal(0, s_m))
            X, Y = np.array(X), np.array(Y); Bf.append(fit(X, Y)); Br.append(fit(X[:, :4], Y))
        v = reg2_verdict(np.array(Bf), np.array(Br)); cnt[v] = cnt.get(v, 0) + 1
    items = sorted(cnt.items(), key=lambda kv: -kv[1])
    items = items if top is None else items[:top]
    return {k: (round(v / nsim, ndigits) if ndigits is not None else v / nsim) for k, v in items}


def reg2_noise(world, cond, y, x, p, ninit=3, nat="nat"):
    """AT7: noise components and design grid from AT2's per-model values of one line and arm (arrays per model: world id, condition
    name, line value y, x = log2(delta_h2 / delta_nat), p). s_m = pooled within-(world, condition) SD over inits; s_wc = sqrt(max(0,
    residual variance of the world x condition means after removing world and condition means (additive OLS fit, so missing cells are
    allowed) - s_m^2 / n_h)); s_w = sqrt(max(0, variance of the world means - s_wc^2 / n_c - s_m^2 / (n_h n_c))); jit = pooled SD of x within
    the non-natural conditions; design grid = per condition (name, mean x, mean p, k = 1 for names starting with 'task'). n_h = harmonic
    mean of the cell sizes (= ninit = 3 for complete data); n_c = number of conditions."""
    world, cond = np.asarray(world), np.asarray(cond).astype(str); y, x, p = (np.asarray(a, float) for a in (y, x, p))
    W_, C_ = list(dict.fromkeys(world.tolist())), list(dict.fromkeys(cond.tolist()))
    cells, ss, dfw = {}, 0.0, 0
    for w in W_:
        for c in C_:
            v = y[(world == w) & (cond == c)]
            if len(v):
                cells[(w, c)] = v; ss += ((v - v.mean()) ** 2).sum(); dfw += len(v) - 1
    s_m = float(np.sqrt(ss / dfw)) if dfw > 0 else 0.0
    nh = float(len(cells) / sum(1.0 / len(v) for v in cells.values()))
    keys = list(cells); mvals = np.array([cells[k_].mean() for k_ in keys])
    D = np.zeros((len(keys), len(W_) + len(C_) - 1)); D[:, 0] = 1
    for r, (w, c) in enumerate(keys):
        wi, ci_ = W_.index(w), C_.index(c)
        if wi > 0:
            D[r, wi] = 1
        if ci_ > 0:
            D[r, len(W_) - 1 + ci_] = 1
    coef = fit(D, mvals); res = mvals - D @ coef; dfr = len(keys) - D.shape[1]
    var_res = float((res ** 2).sum() / dfr) if dfr > 0 else 0.0
    s_wc = float(np.sqrt(max(0.0, var_res - s_m ** 2 / nh)))
    wm = np.array([np.mean([cells[k_].mean() for k_ in keys if k_[0] == w]) for w in W_])
    s_w = float(np.sqrt(max(0.0, (wm.var(ddof=1) if len(wm) > 1 else 0.0) - s_wc ** 2 / len(C_) - s_m ** 2 / (nh * len(C_)))))
    jx = [(x[cond == c] - x[cond == c].mean()) for c in C_ if c != nat and (cond == c).sum() > 1]
    jit = float(np.sqrt(sum((d ** 2).sum() for d in jx) / sum(len(d) - 1 for d in jx))) if jx else 0.0
    grid = [(c, 0.0 if c == nat else float(x[cond == c].mean()), 0.0 if c == nat else float(p[cond == c].mean()), int(str(c).startswith("task"))) for c in C_]
    return dict(s_m=s_m, s_wc=s_wc, s_w=s_w, jit=jit, cond=grid, n_h=nh, n_worlds=len(W_))


# ====================================================================================================================== theta A2
GRID = np.array([0.5, 1.0, 2.0, 4.0]); B_BOOT = 400


def crossings(a, g):
    """sim_gain.crossings: g (R, C) gains on the grid a (C,). Returns (down, up): first crossing of g = 1 from above / from below as a
    grows, log-log interpolation. down = +inf if g > 1 everywhere, 0 if g < 1 everywhere; nan otherwise when no downward crossing."""
    lg = np.log(np.clip(g, 1e-6, None)); la = np.log(a); R, C = g.shape; dn = np.full(R, np.nan); up = np.full(R, np.nan)
    for i in range(C - 1):
        x = la[i] + (la[i + 1] - la[i]) * lg[:, i] / (lg[:, i] - lg[:, i + 1] + 1e-300)
        m = np.isnan(dn) & (lg[:, i] > 0) & (lg[:, i + 1] <= 0); dn[m] = np.exp(x[m])
        m = np.isnan(up) & (lg[:, i] < 0) & (lg[:, i + 1] >= 0); up[m] = np.exp(x[m])
    dn[np.isnan(dn) & (lg > 0).all(1)] = np.inf; dn[np.isnan(dn) & (lg < 0).all(1)] = 0.0
    return dn, up


def classify(lo, hi):
    """sim_gain.classify: the pre-registered rule on per-cell 90 % bootstrap intervals [lo, hi] ordered by alpha."""
    C = len(lo); fd = any(lo[i] > 1 and hi[k] < 1 for i in range(C) for k in range(i + 1, C))
    fu = any(hi[i] < 1 and lo[k] > 1 for i in range(C) for k in range(i + 1, C))
    if fd and fu:
        return "found_both"
    if fd:
        return "found_down"
    if fu:
        return "found_up"
    if (hi < 1).all():
        return "sub"
    if (lo > 1).all():
        return "super"
    return "undecided"


def pci(x, lo=5, hi=95):
    """sim_gain.ci: percentile interval of the non-nan values."""
    x = x[~np.isnan(x)]
    return (float(np.percentile(x, lo)), float(np.percentile(x, hi))) if len(x) else (np.nan, np.nan)


def tanh_family(lam, astar):
    """sim_gain.tanh_family: soft onset F(a) = Fm tanh(lam a / Fm), G = F / a, G(0) = lam, G(astar) = 1 (Fm by bisection)."""
    lo, hi = astar * 1.000001, 50.0 * astar
    for _ in range(200):
        Fm = 0.5 * (lo + hi)
        if Fm * np.tanh(lam * astar / Fm) > astar:
            hi = Fm
        else:
            lo = Fm
    return (lambda a, Fm=Fm: Fm * np.tanh(lam * np.asarray(a, float) / Fm) / np.asarray(a, float)), Fm


def hill_family(Fm=3.0, at=1.0):
    """sim_gain.hill_family: hard onset F(a) = Fm a^3 / (at^3 + a^3)."""
    return lambda a: Fm * np.asarray(a, float) ** 2 / (at ** 3 + np.asarray(a, float) ** 3)


def first_crossing(G, up):
    """sim_gain.first_crossing on a fine log grid of a in [0.05, 20]."""
    a = np.exp(np.linspace(np.log(0.05), np.log(20), 200001)); g = G(a) - 1
    s = np.nonzero((g[:-1] < 0) & (g[1:] >= 0) if up else (g[:-1] > 0) & (g[1:] <= 0))[0]
    return float(a[s[0]]) if len(s) else None


def make_scenarios():
    """sim_gain.SCEN (same order): soft crossings at 1.5, 0.7, 3 alpha_0 (tanh family), sub-critical lambda 0.6 and 0.85, threshold
    (hysteresis), flat g = 1. Values: (G, planted downward crossing, planted upward crossing, truth)."""
    scen = {}
    for nm, lam, ast_ in (("super_mid (lam 2, a* 1.5)", 2.0, 1.5), ("super_low (lam 1.4, a* 0.7)", 1.4, 0.7), ("super_high (lam 3, a* 3)", 3.0, 3.0)):
        G, Fm = tanh_family(lam, ast_); scen[nm] = (G, ast_, None, "found_down")
    scen["sub (lam 0.6)"] = (lambda a: 1.0 * np.tanh(0.6 * np.asarray(a, float)) / np.asarray(a, float), None, None, "sub")
    scen["sub_near (lam 0.85)"] = (lambda a: 3.0 * np.tanh(0.85 * np.asarray(a, float) / 3.0) / np.asarray(a, float), None, None, "sub")
    H = hill_family(); scen["threshold (hysteresis)"] = (H, first_crossing(H, False), first_crossing(H, True), "found_both")
    scen["flat critical (g = 1)"] = (lambda a: np.ones_like(np.asarray(a, float)), None, None, "undecided")
    return scen


SCEN = make_scenarios()
NOISE = {"favourable (SNR 0.2, D_eff 40)": (0.2, 40), "expected (SNR 0.2, D_eff 8)": (0.2, 8), "pessimistic (SNR 0.1, D_eff 8)": (0.1, 8)}


def scalar_experiment(G, a_cells, N, snr, deff, h, rho, q, rng, B=B_BOOT):
    """sim_gain.scalar_experiment: projected read-back per pair along the pushed direction, all cells sharing prompts and keys."""
    C = len(a_cells); sp = np.sqrt(1.0 / (snr ** 2 * deff))
    eta = rng.normal(0, h, N); zc = rng.normal(size=N); z = rng.normal(size=(C, N))
    x = (G(a_cells) * a_cells)[:, None] * (1 + eta)[None] + (sp * a_cells ** (q / 2))[:, None] * (rho * zc[None] + np.sqrt(1 - rho ** 2) * z)
    g = x.mean(1) / a_cells; se = x.std(1, ddof=1) / np.sqrt(N) / a_cells
    idx = rng.integers(0, N, (B, N)); gb = x[:, idx].mean(2).T / a_cells
    return g, se, gb


def correlation_loading(rho):
    """OD-B14 (review toy_num-1): sim_gain's noise model gives every cell the common factor zc with loading L, so two cells' per-pair
    read-backs correlate with L^2 (at h = 0). The measured correlation r of the per-pair projections across cells is therefore passed
    as L = sqrt(r); a negative r cannot be represented by one common factor and is set to 0 (r is clipped to [0, 1])."""
    return float(np.sqrt(min(max(float(rho), 0.0), 1.0)))


def _cells_x(G, a_cells, n_pairs, sd, h, rho, rng):
    """The per-pair read-backs x [C, Nmax] of cells_experiment (its generator use: eta, zc, z)."""
    a_cells = np.asarray(a_cells, float); C = len(a_cells); n = np.broadcast_to(np.asarray(n_pairs, int), (C,)); Nm = int(n.max())
    sd = np.broadcast_to(np.asarray(sd, float), (C,))
    eta = rng.normal(0, h, Nm); zc = rng.normal(size=Nm); z = rng.normal(size=(C, Nm))
    return (G(a_cells) * a_cells)[:, None] * (1 + eta)[None] + sd[:, None] * (rho * zc[None] + np.sqrt(1 - rho ** 2) * z), n, Nm


def cells_experiment(G, a_cells, n_pairs, sd, h, rho, rng, B=B_BOOT):
    """Measured-noise generalisation of scalar_experiment (OD-B14): per-pair read-back x_cj = a_c G(a_c) (1 + eta_j) + sd_c (rho zc_j +
    sqrt(1 - rho^2) z_cj) with per-cell SD sd_c of the per-pair projections (in units where the expected read-back is a G(a)), the
    LOADING rho of the common factor zc (as sim_gain; the cross-cell correlation is rho^2 at h = 0: gain_A2 and resolvable_band pass
    correlation_loading(measured correlation)) and per-cell pair counts n_c: the cells share the first min(n_c) pairs' eta and zc
    (common keys); the joint bootstrap draws Nmax indices per replicate and a cell with n_c pairs uses the first n_c of them modulo n_c.
    With equal n_c and sd_c = sqrt(1 / (snr^2 deff)) a_c^(q / 2) this is scalar_experiment (same generator use)."""
    a_cells = np.asarray(a_cells, float); C = len(a_cells)
    x, n, Nm = _cells_x(G, a_cells, n_pairs, sd, h, rho, rng)
    idx = rng.integers(0, Nm, (B, Nm)); g, se, gb = np.zeros(C), np.zeros(C), np.zeros((B, C))
    for c in range(C):
        xc = x[c, :n[c]]; g[c] = xc.mean() / a_cells[c]; se[c] = xc.std(ddof=1) / np.sqrt(n[c]) / a_cells[c]
        gb[:, c] = xc[idx[:, :n[c]] % n[c]].mean(1) / a_cells[c]
    return g, se, gb


def a2_scenario(G, adn, aup, truth, reps, rng, experiment, a_cells=GRID):
    """sim_gain.run_A2's per-scenario summary: rule classes over reps replicates of experiment(rng) -> (g, se, gb), coverage of the
    planted crossings by the bootstrap crossings, per-cell normal-interval coverage, bias and SD."""
    cls, cov_dn, cov_up, wid, est, zcov, bias = [], [], [], [], [], [], []
    for r in range(reps):
        g, se, gb = experiment(rng)
        lo, hi = np.percentile(gb, 5, 0), np.percentile(gb, 95, 0); c = classify(lo, hi); cls.append(c)
        zcov.append(np.mean(np.abs(g - G(a_cells)) <= 1.645 * se)); bias.append(g - G(a_cells))
        dn, up = crossings(a_cells, gb)
        if adn is not None and c in ("found_down", "found_both"):
            l, u = pci(dn); cov_dn.append(l <= adn <= u); wid.append(u / l if l > 0 else np.inf); est.append(float(np.nanmedian(dn)))
        if aup is not None and c in ("found_up", "found_both"):
            l, u = pci(up); cov_up.append(l <= aup <= u)
    cls = np.array(cls); d = {k: float(np.mean(cls == k)) for k in ("found_down", "found_up", "found_both", "sub", "super", "undecided")}
    return dict(truth=truth, p_correct=float(np.mean(cls == truth)), classes=d,
                coverage_alpha_star_90=float(np.mean(cov_dn)) if cov_dn else None, n_found=len(cov_dn),
                median_ci_width_factor=float(np.median(wid)) if wid else None,
                median_alpha_star=float(np.median(est)) if est else None, planted_alpha_star=adn,
                coverage_lower_crossing=float(np.mean(cov_up)) if cov_up else None,
                per_cell_normal_ci_coverage=float(np.mean(zcov)), per_cell_bias=np.mean(bias, 0).round(4).tolist(),
                per_cell_sd=np.std(np.array(bias), 0).round(4).tolist())


def run_A2_port(reps=400, B=B_BOOT):
    """sim_gain.run_A2 reproduced (same loops, generator default_rng(1), noise models, scenarios)."""
    rng = np.random.default_rng(1); out = {}
    for nn, (snr, deff) in NOISE.items():
        for N in (64, 128):
            for q in (0.5, 1.0):
                for rho in ((0.6, 0.0) if (N == 64 and q == 0.5) else (0.6,)):
                    key = f"{nn} | N {N} | noise ~ a^{q} | CRN rho {rho}"; out[key] = {}
                    for sn, (G, adn, aup, truth) in SCEN.items():
                        exp_ = lambda rg, G=G: scalar_experiment(G, GRID, N, snr, deff, 0.5, rho, q, rg, B)
                        out[key][sn] = a2_scenario(G, adn, aup, truth, reps, rng, exp_)
    return out


def gain_A2(a_cells, n_pairs, sd, rho, h=0.0, scen=None, reps=400, B=B_BOOT, seed=1):
    """OD-B14 power re-run: run_A2's summary per scenario (default SCEN: soft crossings at 0.7, 1.5, 3 alpha_0, sub-critical lambda 0.6
    and 0.85, threshold, flat g = 1) with the measured per-cell SDs sd [C] of the per-pair projections on u_read (push units, i.e. divided
    by alpha_b cbar as the gains), the measured correlation rho of these projections across cells (simulated as that correlation:
    loading correlation_loading(rho), exact for h = 0), the cells' pair counts n_pairs [C] and cell alphas a_cells (multiples of
    alpha_0, ascending). h: extra prompt heterogeneity of the signal (0: already in the measured SD). Returns {scenario: summary}."""
    scen = SCEN if scen is None else scen; a_cells = np.asarray(a_cells, float); rng = np.random.default_rng(seed); out = {}
    load = correlation_loading(rho)
    for sn, (G, adn, aup, truth) in scen.items():
        exp_ = lambda rg, G=G: cells_experiment(G, a_cells, n_pairs, sd, h, load, rg, B)
        out[sn] = a2_scenario(G, adn, aup, truth, reps, rng, exp_, a_cells)
    return out


def resolvable_band(a_cells, n_pairs, sd, rho, h=0.0, reps=400, B=B_BOOT, seed=2, g_grid=None, level=0.8):
    """OD-B14 'resolvable band': for flat gains g (G(a) = g at every cell) on g_grid (default 41 values log-spaced in [0.5, 2]) the rate
    of 'sub' or 'super' over reps replicates; the band = the largest interval around 1 in which the rate stays < level, its ends found
    by linear interpolation in log g between the last grid point inside and the first outside. rho = the measured cross-cell
    correlation (loading correlation_loading(rho), as gain_A2). Returns dict(lo, hi, grid, rate) (lo = None / hi = None: no end within
    the grid)."""
    g_grid = np.exp(np.linspace(np.log(0.5), np.log(2.0), 41)) if g_grid is None else np.asarray(g_grid, float)
    a_cells = np.asarray(a_cells, float); rng = np.random.default_rng(seed); rate = []; load = correlation_loading(rho)
    for gv in g_grid:
        G = lambda a, gv=gv: np.full_like(np.asarray(a, float), gv); hits = 0
        for _ in range(reps):
            _, _, gb = cells_experiment(G, a_cells, n_pairs, sd, h, load, rng, B)
            c = classify(np.percentile(gb, 5, 0), np.percentile(gb, 95, 0)); hits += c in ("sub", "super")
        rate.append(hits / reps)
    rate = np.array(rate); lg = np.log(g_grid); i1 = int(np.argmin(np.abs(lg)))

    def end(step):
        i = i1
        while 0 <= i + step < len(g_grid) and rate[i + step] < level:
            i += step
        j = i + step
        if not (0 <= j < len(g_grid)) or rate[i] >= level:
            return None
        t = (level - rate[i]) / (rate[j] - rate[i]) if rate[j] != rate[i] else 0.0
        return float(np.exp(lg[i] + t * (lg[j] - lg[i])))
    return dict(lo=end(-1), hi=end(+1), grid=g_grid.tolist(), rate=rate.tolist())


# ====================================================================================================================== selftest
def _ref(rel):
    here = os.path.dirname(os.path.abspath(__file__)); scratch = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
    for base in (os.path.join(here, "ref"), os.environ.get("R2REF", ""), scratch):
        if base:
            for c in (os.path.join(base, rel), os.path.join(base, os.path.basename(rel))):
                if os.path.exists(c):
                    return c
    return None


def _defs(path, names, ns):
    tree = ast.parse(open(path).read())
    nodes = [n for n in tree.body if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)
             or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)
    return ns


def selftest(quiet=False, return_rows=False):
    """Ports against the references on their own small examples: run2 / verdict2 (crit/sim_reg2.py with sim_reg.py's COND and ci) and
    run_A2 (theta_design/sim_gain.py) with few replicates; generalised experiment = scalar_experiment for equal pair counts."""
    rows = []

    def say(name, ok, info="", **kw):
        rows.append(dict(check=name, passed=None if ok is None else bool(ok), info=info, **kw))
        if not quiet:
            print(f"[{'PASS' if ok else ('FAIL' if ok is not None else 'SKIP')}] {name} {info}")
    ok_all = True
    p1, p2 = _ref("rgi_design/crit/sim_reg.py"), _ref("rgi_design/crit/sim_reg2.py")
    if p1 and p2:
        ns = dict(np=np, stats=stats); _defs(p1, ["COND", "ci"], ns); _defs(p2, ["S", "COND2", "fit", "verdict2", "run2"], ns)
        ok = ns["COND"] == COND and ns["COND2"] == COND2 and list(ns["S"]) == list(SCENARIOS)
        rng = np.random.default_rng(0)
        for _ in range(200):
            Bf, Br = rng.normal(0, 0.06, (8, 7)), rng.normal(0, 0.06, (8, 4))
            Bf[:, 5:] += rng.choice([0, 0.1], 2); Br[:, 1:3] += rng.choice([-0.1, 0, 0.1], 2)
            ok = ok and reg2_verdict(Bf, Br) == ns["verdict2"](Bf, Br)
        for name, f in SCENARIOS.items():
            ok = ok and reg2_simulate(f, COND, nsim=12, top=4, ndigits=3) == ns["run2"](ns["S"][name], ns["COND"], nsim=12)
        ok = ok and reg2_simulate(SCENARIOS["perturbation only (pilot-like, -0.6 per unit p)"], COND2, NW=5, nsim=10, seed=4, top=4, ndigits=3) == \
            ns["run2"](ns["S"]["perturbation only (pilot-like, -0.6 per unit p)"], ns["COND2"], NW=5, nsim=10, seed=4)
        say("reg2_verdict / reg2_simulate = crit/sim_reg2.verdict2 / run2 (200 random coefficient sets; 6 scenarios x nsim 12; COND2 NW 5)", ok,
            mine=reg2_simulate(SCENARIOS["perturbation only (pilot-like, -0.6 per unit p)"], COND, nsim=12, top=4, ndigits=3),
            ref=ns["run2"](ns["S"]["perturbation only (pilot-like, -0.6 per unit p)"], ns["COND"], nsim=12)); ok_all &= ok
        sim = reg2_simulate(SCENARIOS["true growth 0.10/doubling above natural + perturbation"], nsim=30)
        world = np.repeat(np.arange(8), len(COND) * 3); cond = np.tile(np.repeat([c[0] for c in COND], 3), 8)
        g = np.random.default_rng(3); xx = np.array([dict((c[0], c[1]) for c in COND)[c] for c in cond]) + g.normal(0, 0.15, len(cond)) * (cond != "nat")
        yy = 0.8 + 0.05 * world / 8 + g.normal(0, 0.05, len(cond)); pp = np.abs(g.normal(0.3, 0.03, len(cond))) * (cond != "nat")
        nz = reg2_noise(world, cond, yy, xx, pp)
        ok = abs(nz["s_m"] - 0.05) < 0.015 and abs(nz["jit"] - 0.15) < 0.04 and nz["n_h"] == 3 and len(nz["cond"]) == len(COND) and sum(sim.values()) > 0.999
        say("reg2_noise recovers planted s_m / jit; simulated rates sum to 1", ok, f"s_m {nz['s_m']:.3f} (0.05), jit {nz['jit']:.3f} (0.15)"); ok_all &= ok
    else:
        say("sim_reg / sim_reg2 references not found", None)
    p3 = _ref("theta_design/sim_gain.py")
    if p3:
        argv = sys.argv; ns = dict(__name__="sim_gain_ref", __file__=p3)
        try:
            sys.argv = ["sim_gain.py", os.devnull, "2"]; src = open(p3).read()
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, p3, "exec"), ns); ref = ns["run_A2"]()
        finally:
            sys.argv = argv
        mine = run_A2_port(reps=2)
        ok = list(ns["SCEN"]) == list(SCEN) and ref == mine
        k0 = list(mine)[0]; s0 = list(SCEN)[0]
        say(f"run_A2_port(2) = sim_gain.run_A2 with REPS 2 (all {len(mine)} noise settings x {len(SCEN)} scenarios, exact)", ok,
            mine=mine[k0][s0], ref=ref[k0][s0]); ok_all &= ok
        r1, r2 = np.random.default_rng(9), np.random.default_rng(9); G = SCEN["super_mid (lam 2, a* 1.5)"][0]
        a = scalar_experiment(G, GRID, 64, 0.2, 8, 0.5, 0.6, 0.5, r1); b = cells_experiment(G, GRID, 64, np.sqrt(1 / (0.2 ** 2 * 8)) * GRID ** 0.25, 0.5, 0.6, r2)
        ok = all(np.allclose(u, v, rtol=1e-12, atol=1e-12) for u, v in zip(a, b))
        xs, _, _ = _cells_x(lambda a: np.ones_like(np.asarray(a, float)), [1.0, 2.0, 4.0], 40000, [1.0, 2.0, 0.5], 0.0, correlation_loading(0.6), np.random.default_rng(5))
        cm = np.corrcoef(xs); sim_r = float(cm[np.triu_indices(3, 1)].mean())
        ok = ok and abs(sim_r - 0.6) < 0.01 and correlation_loading(0.36) == 0.6 and correlation_loading(-0.4) == 0.0 and correlation_loading(1.7) == 1.0
        say("OD-B14: the simulated cross-cell correlation equals the measured one (loading sqrt(r); negative r -> 0)", ok, f"passed 0.6, simulated {sim_r:.4f}")
        r_a, r_b = np.random.default_rng(1), np.random.default_rng(1); G1 = SCEN["super_mid (lam 2, a* 1.5)"]
        via = gain_A2([0.5, 1, 2], [256, 128, 128], [0.9, 1.0, 1.2], 0.36, reps=3, B=100, scen={"s": G1})["s"]
        direct = a2_scenario(G1[0], G1[1], G1[2], G1[3], 3, r_b, lambda rg: cells_experiment(G1[0], np.array([0.5, 1, 2]), [256, 128, 128], [0.9, 1.0, 1.2], 0.0, 0.6, rg, 100), np.array([0.5, 1, 2]))
        ok = ok and via == direct
        gA = gain_A2([0.5, 1, 2], [256, 128, 128], [0.9, 1.0, 1.2], 0.5, reps=3, B=100)
        band = resolvable_band([0.5, 1, 2], [256, 128, 128], [0.9, 1.0, 1.2], 0.5, reps=20, B=100, g_grid=np.exp(np.linspace(np.log(0.5), np.log(2), 9)))
        ok = ok and set(gA) == set(SCEN) and (band["lo"] is None or band["lo"] < 1) and (band["hi"] is None or band["hi"] > 1)
        say("cells_experiment = scalar_experiment for equal pair counts; gain_A2 / resolvable_band run", ok, f"band ({band['lo']}, {band['hi']})"); ok_all &= ok
    else:
        say("sim_gain reference not found", None)
    return (bool(ok_all), rows) if return_rows else bool(ok_all)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        ok = selftest()
        print("r2_sim selftest:", "passed" if ok else "FAILED")
        raise SystemExit(0 if ok else 1)
    print(__doc__)
