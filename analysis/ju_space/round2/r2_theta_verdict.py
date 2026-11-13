"""Round 2, session r2_ana (plan section 4.1; rules R1-R6 with the reserve conditions OD-B8 / G21-G23 / G25, B-Q0 statistics, B-Q2,
B-Q6 / R4 (ii), B-Q7 lambda_lin, the power re-run OD-B14; gates G11, G12, G17-G20, G24, G26, G49 and the OD-G8 recomputation of the
in-session decisions; CLI of 9.4): theta_verdict.json.

All intervals are 90 % percentile intervals of ONE joint prompt bootstrap (r2_theta.boot_W: 2,000 replicates, prompts as units, both
seeds of a prompt together, stratified by prompt type; row 0 = the point estimate), applied to every B-Q0 evaluation, cell, turn-2
cell, power step and FDT text of every block, so that differences between cells and blocks are paired (2.6, 'prompts and keys shared
across the cells and blocks'). Directions that define a cell (u_push, u_read) are held fixed; the statistics that estimate (U, read-back
projections, cbar, m*^2, m_A^2, Gamma_b, u_crit in B-Q2) are recomputed per replicate.
Amendments (abschnitt11_12_runde2_plan_amendments.md): 1 (distinct-pair U statistic in bootstrap replicates; R2 'crosses 1' in the
direction of the point fit), 3 (b) (a rule bound is undefined when the point value or more than 5 % of the replicates are undefined:
lohi / bounds_ok / nf_share; the share is written into every rule output), 3 (c) (R4 (i) 'degenerate': no '(RSI-specific)' suffix, the
block does not enter R3's alignment form).

Inputs: <r1_dir> = the copied round-1 files (gpu*/rk/b<b>/{state.pt, loops.json}, gpu*/summary_b<b>.json, gpu2 / gpu3 dn_delta.json and
blocks_chosen.json; found by name, OD-G9); <r2_root>/<session>/ = dsse_runs/<session>/ (theta/ of r2_q2, axes/r2_axes.pt of r2_q1; found
by name, the expected session first).
Usage: python r2_theta_verdict.py <r1_dir> <r2_root> <out.json>
       python r2_theta_verdict.py selftest <workdir> [<r1_dir> <r2_root>]     (synthetic checks; with MOCK theta outputs also end to end)
Env: R2_THETA_POWER_REPS (400: replicates per scenario of the power re-run, OD-B14), MOCK (MOCK Pool directions)."""
import glob, json, math, os, shutil, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
import r2_rk as R
import r2_theta as TH

LN125 = math.log(1.25)
TC_RANGE, TC_RANGE_EXT = (0.2, 1.1), (0.1, 1.4)       # R2: 'T_c located' window (extended when a reserve temperature entered)
DMG_LIMIT = 0.15                                      # R1: a crossing above 0.15 nats is 'damage-limited'
ALIGN_P, ALIGN_COS, PWR_COS, GENERIC_COS = 0.01, 0.25, 0.7, 0.5
G23_LO = 0.8
NONFINITE_MAX = 0.05                                  # a rule bound is undefined when the point value is not finite or more than 5 % of the
                                                      # bootstrap replicates are not finite (Amendment 3 (b): R6's '>= 95 %' convention)
CELL_A0 = TH.cell_name(1, 0.7)
POWER_REPS = int(os.environ.get("R2_THETA_POWER_REPS", "400"))
LIMIT_T = "for weak T-dependence (slope about 0.5) nothing locates T_c at this budget"
DEPTH = "three blocks cannot separate delta from depth"
G1_NOTE = "G = 1 is the threshold of the constructed self-steering map under its read-out / write-in conventions; K is convention-free"


def S(x): return TH.summ(x)


def nf_share(x):
    """Share of the bootstrap rows 1.. of a replicate array that are not finite (NaN, or +-inf of a censored quantity); NaN without rows."""
    v = TH._np(x).reshape(-1)[1:]; return float((~np.isfinite(v)).mean()) if len(v) else float("nan")


def bounds_ok(x):
    """A rule may use the bounds of x: finite point value (row 0) and at most 5 % non-finite bootstrap rows (Amendment 3 (b))."""
    v = TH._np(x).reshape(-1); return bool(len(v) > 1 and np.isfinite(v[0]) and (~np.isfinite(v[1:])).mean() <= NONFINITE_MAX)


def lohi(x):
    """The bounds a rule compares: the 90 % interval over the bootstrap rows, or (NaN, NaN) when the point value is not finite or more
    than 5 % of the bootstrap rows are not finite (a gain is undefined where its attenuation replicate is 0: 'mode not resolved' blocks,
    the norm gain in the switch case). Every bound comparison is then False: the rule is undecided, a reserve condition is not met.
    The reported intervals (S) stay the percentiles of the finite rows, with n_finite; every rule output carries the non-finite share."""
    if not bounds_ok(x): return float("nan"), float("nan")
    s = TH.summ(x); return s["lo"], s["hi"]


def fin(x): return TH.fin(x)


def _np(x): return TH._np(x)


# ======================================================================================================================================
# Rules (pure functions on replicate arrays: row 0 = point estimate, rows 1.. = the joint bootstrap)
# ======================================================================================================================================
def crossings(a, g):
    """sim_gain.crossings (= r2_sim.crossings): g [R, C] gains on the ascending grid a [C]; first crossing of g = 1 from above (down) and
    from below (up) as a grows, log-log interpolation; down = +inf if g > 1 everywhere, 0 if g < 1 everywhere, NaN otherwise without a
    downward crossing."""
    a = np.asarray(a, float); g = np.asarray(g, float)
    lg = np.log(np.clip(g, 1e-6, None)); la = np.log(a); Rn, C = g.shape; dn = np.full(Rn, np.nan); up = np.full(Rn, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for i in range(C - 1):
            x = la[i] + (la[i + 1] - la[i]) * lg[:, i] / (lg[:, i] - lg[:, i + 1] + 1e-300)
            m = np.isnan(dn) & (lg[:, i] > 0) & (lg[:, i + 1] <= 0); dn[m] = np.exp(x[m])
            m = np.isnan(up) & (lg[:, i] < 0) & (lg[:, i + 1] >= 0); up[m] = np.exp(x[m])
        dn[np.isnan(dn) & (lg > 0).all(1)] = np.inf; dn[np.isnan(dn) & (lg < 0).all(1)] = 0.0
    return dn, up


def censored(x, a_min, a_max):
    """Point (row 0) and 90 % interval (rows 1.., NaN = no crossing of that kind, dropped as in sim_gain.ci) of a crossing; values
    outside [a_min, a_max] come only from the censoring (0 = below 1 throughout, +inf = above 1 throughout) and are reported as 0 / inf."""
    x = np.asarray(x, float); v = x[1:]; v = v[~np.isnan(v)]
    fix = lambda t: (float("inf") if t > a_max * (1 + 1e-9) else (0.0 if t < a_min * (1 - 1e-9) else float(t))) if fin(t) or t == float("inf") else float("nan")
    if not len(v): return dict(pt=fix(x[0]) if not np.isnan(x[0]) else float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    v = np.minimum(v, 1e12); lo, hi = np.percentile(v, 5), np.percentile(v, 95)
    return dict(pt=fix(x[0]) if not np.isnan(x[0]) else float("nan"), lo=fix(lo), hi=fix(hi), n=int(len(v)))


def damage_at(a_star, curve):
    """D* (OD-B9): log damage interpolated linearly in log alpha between the grid points of the damage curve {a: damage}; no
    extrapolation: below the grid 0 ('< D(a_min)'), above +inf ('> D(a_max)'), NaN stays NaN."""
    if not fin(a_star): return float("inf") if a_star == float("inf") else float("nan")
    A = np.array(sorted(curve), float); Dm = np.array([max(float(curve[a]), 1e-300) for a in sorted(curve)])
    if a_star < A[0] * (1 - 1e-12): return 0.0
    if a_star > A[-1] * (1 + 1e-12): return float("inf")
    return float(np.exp(np.interp(math.log(a_star), np.log(A), np.log(Dm))))


def rule_R1(a, reps, half=None, curve=None, window_reps=None, band=None):
    """R1 critical push on one alpha series (ascending a, multiples of alpha_0; reps = decision-gain replicate arrays of the primary
    window). FOUND-soft = some lower-alpha point with lower bound > 1 and some higher-alpha point with upper bound < 1; FOUND-hard = upper
    bound < 1 at a lower alpha and lower bound > 1 at a higher alpha (both: 'bistable window'); SUB(>= alpha_0) = every upper bound < 1 for
    alpha >= alpha_0 ('SUB' if the 0.5 alpha_0 cell's too); SUPER = every lower bound > 1; otherwise UNDECIDED. alpha* = first downward
    crossing (log-log) with its censored interval, also as D* (damage curve); onset from the paired g(0.5 alpha_0) - g(alpha_0) (half =
    index of the 0.5 alpha_0 point); labels 'damage-limited' (D* > 0.15 nats) and 'convention-sensitive' (window_reps: {window: reps};
    alpha* intervals of two windows do not overlap); band = the resolvable band of the power re-run (OD-B14). A point whose bounds are
    undefined (Amendment 3 (b), lohi) satisfies no bound condition; nonfinite_share / bounds_undefined per series point."""
    a = np.asarray(a, float); C = len(a); L = [lohi(r) for r in reps]; lo = np.array([x[0] for x in L]); hi = np.array([x[1] for x in L])
    soft = any(lo[i] > 1 and hi[k] < 1 for i in range(C) for k in range(i + 1, C))
    hard = any(hi[i] < 1 and lo[k] > 1 for i in range(C) for k in range(i + 1, C)); ge = a >= 1 - 1e-9
    if soft and hard: v = "bistable window (FOUND-soft and FOUND-hard)"
    elif soft: v = "FOUND-soft"
    elif hard: v = "FOUND-hard"
    elif ge.any() and bool((hi[ge] < 1).all()): v = "SUB" if (half is not None and hi[half] < 1) else "SUB(>= alpha_0)"
    elif bool((lo > 1).all()): v = "SUPER"
    else: v = "UNDECIDED"
    if any(not bounds_ok(r) for r in reps): v = "UNDECIDED"      # Amendment 3 (b): an undefined bound makes the rule's outcome undecided
    G = np.stack([_np(r) for r in reps], 1); dn, up = crossings(a, G)
    out = dict(verdict=v, a=a.tolist(), points=[S(r) for r in reps], alpha_star=censored(dn, a.min(), a.max()),
               alpha_up=censored(up, a.min(), a.max()), labels=[], resolvable_band=band, nonfinite_share=[nf_share(r) for r in reps],
               bounds_undefined=[not bounds_ok(r) for r in reps])
    if curve:
        ast = out["alpha_star"]; out["D_star"] = dict(pt=damage_at(ast["pt"], curve), lo=damage_at(ast["lo"], curve), hi=damage_at(ast["hi"], curve),
                                                      censoring="0 = below the damage grid, inf = above (no extrapolation, OD-B9)")
        if v in ("FOUND-soft", "bistable window (FOUND-soft and FOUND-hard)") and fin(out["D_star"]["pt"]) and out["D_star"]["pt"] > DMG_LIMIT:
            out["labels"].append("damage-limited")
    if window_reps and v in ("FOUND-soft", "bistable window (FOUND-soft and FOUND-hard)"):
        iv = {}
        for w, rr in window_reps.items():
            c = censored(crossings(a, np.stack([_np(r) for r in rr], 1))[0], a.min(), a.max()); iv[w] = (c["lo"], c["hi"])
        out["alpha_star_windows"] = {w: dict(lo=x[0], hi=x[1]) for w, x in iv.items()}
        ws = [w for w in iv if not (math.isnan(iv[w][0]) or math.isnan(iv[w][1]))]
        if any(iv[x][1] < iv[y][0] or iv[y][1] < iv[x][0] for i, x in enumerate(ws) for y in ws[i + 1:]): out["labels"].append("convention-sensitive")
    i1 = next((i for i in range(C) if abs(a[i] - 1) < 1e-9), None)
    if half is not None and i1 is not None:
        d = _np(reps[half]) - _np(reps[i1]); l, h = lohi(d)
        out["onset"] = dict(verdict="soft" if l > 0 else ("hard" if h < 0 else "onset undecided"), diff=S(d), rule="paired g(0.5 alpha_0) - g(alpha_0)",
                            nonfinite_share=nf_share(d), bounds_undefined=not bounds_ok(d))
    else: out["onset"] = dict(verdict="onset undecided", note="no 0.5 alpha_0 cell at this block")
    return out


def wls(T, Y, deg=1, T0=0.7):
    """GLS of Y [R, C] (row 0 point, rows 1.. bootstrap) on (T - T0)^k, k = 0..deg, with the inverse of the bootstrap covariance of the C
    cells as weight matrix (fixed; the same fit applied to every row). Returns (coefficients [R, deg + 1] (NaN for rows with a non-finite
    Y), covariance [C, C]) or (None, None) with fewer than deg + 1 cells."""
    T = np.asarray(T, float); Y = np.asarray(Y, float); C = len(T)
    if C < deg + 1: return None, None
    X = np.stack([(T - T0) ** k for k in range(deg + 1)], 1); Yb = Y[1:]; good = np.isfinite(Yb).all(1)
    Sg = np.atleast_2d(np.cov(Yb[good].T)) if good.sum() > C else np.eye(C); Sg = Sg + 1e-12 * max(np.trace(Sg), 1e-12) * np.eye(C)
    Wi = np.linalg.inv(Sg); A = np.linalg.solve(X.T @ Wi @ X, X.T @ Wi); B = Y @ A.T; B[~np.isfinite(Y).all(1)] = np.nan
    return B, Sg


def rule_R2(T, reps, extended=False, T0=0.7):
    """R2 temperature (one block): WLS of log G on T over the entered temperature cells at alpha_0, weights from the bootstrap covariance.
    'above 1 throughout' / 'below 1 throughout' = the fit's 90 % band lies on one side of 1 over the tested range; 'T_c located' = >= 95 % of
    the bootstrap fits cross 1 (in the direction of the point fit) and the 90 % interval of the crossing lies within [0.2, 1.1] ([0.1, 1.4]
    when a reserve temperature entered); otherwise 'undecided'. Reports the slope dlogG/dT, the crossing, the margin and, with >= 4
    temperatures, the curvature. Gains are clipped at 1e-6 before the log (sim_gain's convention). Amendment 3 (b): if the point fit is
    not finite or more than 5 % of the replicate fits are not finite (a gain undefined in some cell), the verdict is 'undecided (gain
    undefined in x % of the replicates)'; slope, T_c and the band of the finite fits stay reported, with nonfinite_share."""
    T = np.asarray(T, float); o = np.argsort(T); T = T[o]; reps = [reps[i] for i in o]; C = len(T)
    out = dict(T=T.tolist(), points=[S(r) for r in reps], extended=bool(extended), limit=LIMIT_T)
    if C < 2: out["verdict"] = "not decidable (fewer than two temperatures)"; return out
    Y = np.log(np.clip(np.stack([_np(r) for r in reps], 1), 1e-6, None)); Y[~np.isfinite(np.stack([_np(r) for r in reps], 1))] = np.nan
    B, Sg = wls(T, Y, 1, T0); rows = np.isfinite(B[1:]).all(1); Bb = B[1:][rows]; rng = TC_RANGE_EXT if extended else TC_RANGE
    nfs = float((~rows).mean()) if len(rows) else float("nan"); pt_ok = bool(np.isfinite(B[0]).all()); und = (not pt_ok) or not (nfs <= NONFINITE_MAX)
    und_v = "undecided (gain undefined " + ("" if pt_ok else "at the point estimate and ") + f"in {100 * nfs:.1f} % of the replicates)"
    out.update(slope=S(B[:, 1]), intercept_T07=S(B[:, 0]), n_fits=int(rows.sum()), nonfinite_share=nfs, bounds_undefined=bool(und))
    if not rows.any():                                         # no finite replicate fit at all: nothing to take percentiles of
        out.update(T_c_point=float("nan"), verdict=und_v, T_c=dict(pt=float("nan"), lo=float("nan"), hi=float("nan"), share_crossing=float("nan"),
                                                                   window=list(rng), direction=None))
        return out
    grid = np.linspace(T.min(), T.max(), 15); band = Bb[:, :1] + Bb[:, 1:2] * (grid[None] - T0)
    blo, bhi = np.percentile(band, 5, 0), np.percentile(band, 95, 0)
    out.update(band_lo=np.exp(blo).tolist(), band_hi=np.exp(bhi).tolist(), grid=grid.tolist())
    s0 = B[0, 1]; out["T_c_point"] = float(T0 - B[0, 0] / s0) if fin(s0) and s0 != 0 else float("nan")
    cross = (np.sign(Bb[:, 1]) == np.sign(s0)) & (Bb[:, 1] != 0); share = float(cross.mean()) if len(Bb) else float("nan")
    tc = T0 - Bb[cross, 0] / Bb[cross, 1]
    out["T_c"] = dict(pt=out["T_c_point"], lo=float(np.percentile(tc, 5)) if len(tc) else float("nan"), hi=float(np.percentile(tc, 95)) if len(tc) else float("nan"),
                      share_crossing=share, window=list(rng), direction="downward" if s0 < 0 else "upward")
    if (blo > 0).all(): v = "above 1 throughout"; out["margin"] = dict(min_lower_band=float(np.exp(blo.min())))
    elif (bhi < 0).all(): v = "below 1 throughout"; out["margin"] = dict(max_upper_band=float(np.exp(bhi.max())))
    elif share >= 0.95 and out["T_c"]["lo"] >= rng[0] and out["T_c"]["hi"] <= rng[1]: v = "T_c located"
    else: v = "undecided"
    if und:                                                    # Amendment 3 (b): the fit, slope and T_c stay reported, the rule is undecided
        v = und_v; out.pop("margin", None)
    out["verdict"] = v
    if C >= 4:
        B2, _ = wls(T, Y, 2, T0); out["curvature"] = S(B2[:, 2]) if B2 is not None else None
    return out


def rule_R3(entries, what):
    """R3 form on one quantity: entries = [(block, delta, replicate array)] of the entering blocks. Ends = the lowest- and highest-delta
    entering blocks; 'falls with delta' if the paired 90 % interval of high - low lies below 0 and the middle block's point is not above
    the upper end of low's interval; 'rises with delta' mirrored (middle not above the upper end of high's interval); otherwise 'no
    monotone relation'; two blocks: the middle clause is dropped (marked); fewer: 'not decidable'. The non-overlap form is reported.
    An undefined bound (Amendment 3 (b)) satisfies no condition: 'no monotone relation (undecided: a bound is undefined ...)'."""
    e = sorted(entries, key=lambda t: t[1]); n = len(e); out = dict(quantity=what, blocks=[x[0] for x in e], delta=[x[1] for x in e], note=DEPTH)
    if n < 2: out["verdict"] = "not decidable (fewer than two entering blocks)"; return out
    (bl, _, rl), (bh, _, rh) = e[0], e[-1]; mid = e[1] if n == 3 else None
    d = _np(rh) - _np(rl); l, h = lohi(d); (lol, hil), (loh, hih) = lohi(rl), lohi(rh); mp = float(_np(mid[2])[0]) if mid else None
    falls = h < 0 and (mid is None or (fin(mp) and mp <= hil)); rises = l > 0 and (mid is None or (fin(mp) and mp <= hih))
    v = f"{what} falls with delta" if falls else (f"{what} rises with delta" if rises else "no monotone relation")
    if not (bounds_ok(d) and bounds_ok(rl) and bounds_ok(rh)): v += " (undecided: a bound is undefined, more than 5 % of the replicates not finite)"
    if mid is None: v += " (two blocks entered; middle clause not applicable)"
    out.update(verdict=v, low=dict(block=bl, **S(rl)), high=dict(block=bh, **S(rh)), middle=None if mid is None else dict(block=mid[0], **S(mid[2])),
               paired_difference=S(d), non_overlap=bool(hil < loh or hih < lol),
               nonfinite_share=dict(low=nf_share(rl), high=nf_share(rh), paired_difference=nf_share(d), middle=None if mid is None else nf_share(mid[2])),
               bounds_undefined=not (bounds_ok(d) and bounds_ok(rl) and bounds_ok(rh)))
    return out


def rule_R5(rel4_rsi_a, gamma):
    """R5: the persistence line (rel = 0.5, i.e. L / nu = sqrt(2), N-dependent) and the self-sustainment line (G = 1) separately; 'direction
    persists but the loop does not sustain itself' if rel(d_4^RSI_a) >= 0.5 and Gamma_b's upper bound < 1."""
    l, h = lohi(gamma); persists = fin(rel4_rsi_a) and rel4_rsi_a >= 0.5
    v = "direction persists but the loop does not sustain itself" if (persists and h < 1) else None
    return dict(verdict=v, rel_d4_RSI_a=rel4_rsi_a, persistence="rel >= 0.5 (L / nu >= sqrt(2), N-dependent)" if persists else "rel < 0.5",
                Gamma=S(gamma), self_sustainment="lower bound > 1" if l > 1 else ("upper bound < 1" if h < 1 else "undecided"),
                nonfinite_share=nf_share(gamma), bounds_undefined=not bounds_ok(gamma))


def rule_R6(K, d1, G):
    """R6 for one carry cell: K defined only if the 90 % interval of u_read . d^(1) excludes 0 (G26); 'self-sustaining at T' if K's lower
    bound > 1, 'dissipative at T' if its upper bound < 1, otherwise undecided. Transfer test: log K - log G (G = the cell's decision gain)
    only if K > 0 and G > 0 in the point estimate and in >= 95 % of the replicates (interval over the replicates with both positive);
    'transfers' if the interval lies inside (-log 1.25, +log 1.25), 'different criteria' if it excludes 0, otherwise 'undecided'."""
    K, d1, G = _np(K), _np(d1), _np(G); l1, h1 = lohi(d1)
    out = dict(K=S(K), u_read_d1=S(d1), G=S(G), nonfinite_share=dict(u_read_d1=nf_share(d1), K=nf_share(K), G=nf_share(G)),
               bounds_undefined=not (bounds_ok(d1) and bounds_ok(K)))
    if not (l1 > 0 or h1 < 0):
        out.update(defined=False, verdict="K not defined (turn-1 stance not resolved)", transfer=None); return out
    kl, kh = lohi(K); out.update(defined=True, verdict="self-sustaining at T" if kl > 1 else ("dissipative at T" if kh < 1 else "undecided"))
    pos = (K > 0) & (G > 0); share = float(pos[1:].mean()) if len(pos) > 1 else float("nan")
    if pos[0] and share >= 0.95:
        dl = np.log(K[1:][pos[1:]]) - np.log(G[1:][pos[1:]]); lo, hi = float(np.percentile(dl, 5)), float(np.percentile(dl, 95))
        tv = "transfers" if (lo > -LN125 and hi < LN125) else ("different criteria" if (lo > 0 or hi < 0) else "undecided")
        out["transfer"] = dict(verdict=tv, logK_minus_logG=dict(pt=float(math.log(K[0]) - math.log(G[0])), lo=lo, hi=hi), share_positive=share,
                               text="the self-steering threshold transfers to the natural loop" if tv == "transfers" else tv)
    else: out["transfer"] = dict(verdict="undecided (K or G not positive)", share_positive=share)
    return out


def lambda_lin(X, S_, prompt_of, W, cbar_rep, dirs):
    """B-Q7: J_hat = Cov(x, S) / cbar_b; lambda_lin(u) = u . J_hat u = Cov(u . x, u . S) / cbar_b per bootstrap row (frequency-weighted
    covariance with the prompt weights; texts without continuation tokens dropped). dirs: {name: vector}."""
    X = torch.as_tensor(X).double(); S_ = torch.as_tensor(S_).double(); ok = torch.isfinite(X).all(1) & torch.isfinite(S_).all(1)
    X = torch.where(ok[:, None], X, torch.zeros_like(X)); S_ = torch.where(ok[:, None], S_, torch.zeros_like(S_))
    w = torch.as_tensor(W, dtype=torch.float64)[:, torch.as_tensor(prompt_of).long()] * ok.double()[None]; sw = w.sum(1); out = {}
    for nm, u in dirs.items():
        if u is None: out[nm] = None; continue
        u = R.unit(torch.as_tensor(u).double()); a, s = X @ u, S_ @ u
        cov = ((w * a * s).sum(1) - (w @ a) * (w @ s) / sw) / (sw - 1); out[nm] = S(cov / torch.as_tensor(cbar_rep).double())
    return out


def alignment(U, m, axes, k5=None, ga=None):
    """B-Q2 for one critical mode: U [R, D] = the mode direction per bootstrap row (row 0 = point), m [R] (or scalar) = its attenuation m_crit.
    axes: {X: (axis vector, null axes [K, D] or None)} for X in u_T_perp, u_S_perp, sports, tone, sea, u_neg, u_RGI_perp. Corrected cos =
    cos / m_crit with pooled-pair bootstrap intervals; null p = (1 + #{k: |cos(u, null_k)| >= |cos(u, X)|}) / (1 + K) at the point estimate;
    specificity: s (cos(u, u_S perp to u_RGI) - cos(u, u_RGI perp to u_S)), s = the sign of the point RSI cosine; z_beh with R1 K5. Labels
    (R4): RSI-aligned = p <= 0.01, corrected |cos| >= 0.25 with u_T_perp and cos with u_S_perp of the same sign; GI-aligned = the same two
    conditions with u_RGI_perp; RSI-specific = RSI-aligned and the specificity interval above 0. ga: GA(b) (False: 'RGI axis not resolved')."""
    U = R.unit(torch.as_tensor(U).double()); m = torch.as_tensor(m, dtype=torch.float64)
    res, cosr, reps_ = {}, {}, {}
    for X, (v, nul) in axes.items():
        if v is None: res[X] = None; continue
        c = U @ R.unit(torch.as_tensor(v).double()); cosr[X] = c; r = dict(cos=S(c), corrected=S(c / m)); reps_[X] = c / m
        if nul is not None:
            N = R.unit(torch.as_tensor(nul).double()); nc = (N @ U[0]).abs(); r["p"] = float((1 + (nc >= abs(float(c[0]))).sum()) / (1 + len(nc))); r["n_null"] = int(len(nc))
        else: r["p"] = float("nan"); r["n_null"] = 0
        res[X] = r
    out = dict(axes=res, m_crit=S(m if m.dim() else m.repeat(U.shape[0])), labels=[])
    sT = 1.0 if ("u_T_perp" in cosr and float(cosr["u_T_perp"][0]) >= 0) else -1.0
    if axes.get("u_S_perp", (None,))[0] is not None and axes.get("u_RGI_perp", (None,))[0] is not None:
        uS = R.unit(torch.as_tensor(axes["u_S_perp"][0]).double()); uG = R.unit(torch.as_tensor(axes["u_RGI_perp"][0]).double())
        uSo, uGo = R.unit(uS - (uS @ uG) * uG), R.unit(uG - (uG @ uS) * uS); sp = sT * (U @ uSo - U @ uGo)
        out["specificity"] = dict(diff=S(sp), rule="s (cos(u_crit, u_S perp to u_RGI) - cos(u_crit, u_RGI perp to u_S)), s = sign of the RSI cosine",
                                  nonfinite_share=nf_share(sp), lower_bound=lohi(sp)[0])
    if k5:
        out["z_beh"] = {k: float(torch.as_tensor(v["wU_perp"]).double() @ U[0]) / float(v["sigma"]) for k, v in k5.items()}
    rT, rS, rG = res.get("u_T_perp"), res.get("u_S_perp"), res.get("u_RGI_perp")
    rsi = bool(rT and rT["p"] <= ALIGN_P and abs(rT["corrected"]["pt"]) >= ALIGN_COS and rS is not None and np.sign(rS["cos"]["pt"]) == np.sign(rT["cos"]["pt"]))
    gi = bool(rG and rG["p"] <= ALIGN_P and abs(rG["corrected"]["pt"]) >= ALIGN_COS)
    spec = bool(rsi and "specificity" in out and out["specificity"]["lower_bound"] > 0)
    out.update(RSI_aligned=rsi, GI_aligned=gi if rG else None, RSI_specific=spec if "specificity" in out else None, _reps=reps_)
    if rG is None: out["labels"].append("GI alignment not available (O2 axes missing)")
    elif ga is False: out["labels"].append("RGI axis not resolved")
    return out


# ======================================================================================================================================
# Files
# ======================================================================================================================================
def find_r2(r2_root, rel, prefer):
    """<r2_root>/<prefer>/<rel>, else <r2_root>/<rel>, else the first <r2_root>/*/<rel> (sorted). None if absent."""
    for c in [os.path.join(r2_root, prefer, rel), os.path.join(r2_root, rel)] + sorted(glob.glob(os.path.join(r2_root, "*", rel))):
        if os.path.exists(c): return c
    return None


def _ld(p): return None if p is None else (json.load(open(p)) if p.endswith(".json") else torch.load(p, map_location="cpu", weights_only=False))


def load_theta(r2_root, b):
    f = lambda name: find_r2(r2_root, f"theta/{name}", "r2_q2")
    F = {k: _ld(f(n)) for k, n in dict(q0=f"b{b}_q0.json", q0pt=f"b{b}_q0.pt", cells_js=f"b{b}_cells.json", cells=f"b{b}_cells.pt",
                                       power=f"b{b}_power.json", power_pt=f"b{b}_power.pt", fdt=f"b{b}_fdt.pt", rsib=f"b{b}_r2RSIb.json",
                                       rsib_pt=f"b{b}_r2RSIb.pt").items()}
    F["paths"] = {k: f(n) for k, n in dict(q0=f"b{b}_q0.json", cells=f"b{b}_cells.pt", power=f"b{b}_power.json", fdt=f"b{b}_fdt.pt").items()}
    if b == 14: F["turn2"] = _ld(f("b14_turn2.pt")); F["checks"] = _ld(f("checks.json"))
    return F


def _status(x): return None if x is None else x.get("status")


# ======================================================================================================================================
# Per block
# ======================================================================================================================================
def _cfg_used(ins, reps):
    """The cells' configuration as run in the session (OD-G8: the in-session mode and directions govern), with the attenuation replicate
    arrays of the recomputed B-Q0 for that mode."""
    mode = ins["mode"]; up, ur = torch.as_tensor(ins["u_push"]).double(), torch.as_tensor(ins["u_read"]).double()
    if mode == "standard": m2p, mG = reps["m2"], torch.sqrt(reps["m2"])
    elif mode == "switch": m2p, mG = torch.full_like(reps["m2"], float(up @ ur)), torch.sqrt(reps["mA2"])        # cos(u*_A, u*_B) as in bq0
    else: m2p, mG = reps["mA2"], torch.sqrt(reps["mA2"])
    return dict(mode=mode, u_push=up, u_read=ur, m2_par=m2p, mG=mG)


def block_core(r1_dir, b, F, W, summ_b):
    """Phase 1 of one block: B-Q0 recomputed (OD-G8 comparison), the cells' replicate gains, Gamma_b for R3, power step and FDT inputs."""
    out = dict(block=b, labels=[], final=R.block_final(r1_dir, b))
    if not out["final"]: out["status"] = "not run (round-1 output missing)"; return out
    if _status(F["q0"]) != "ok" or F["q0pt"] is None:
        out["status"] = f"not run (B-Q0: {_status(F['q0']) or 'theta/b' + str(b) + '_q0 missing'})"; return out
    try: ST1 = R.load_r1_state(r1_dir, b); loops = R.load_r1_json(r1_dir, f"rk/b{b}/loops.json", b)
    except R.R1Missing as e: out["status"] = "not run (round-1 output missing)"; out["error"] = str(e); return out
    alpha = float(loops["_meta"]["alpha"])
    med = float(ST1["ref"]["med"]); extra = {}
    if b != 14:
        if _status(F["rsib"]) == "ran" and F["rsib_pt"] is not None:
            P = F["rsib_pt"]; extra = {TH.RSIB: dict(ST=P["entries"], rows=F["rsib"]["rows"], start=torch.as_tensor(P["start"]).double(), suffix=P["suffix"])}
    bank = R.o3_bank(ST1, loops, b, extra=extra)
    BQ = TH.bq0(bank, b, alpha, med, F["q0pt"]["cbar_ref"], ST1["ref"]["X"], ST1["axes"], W)
    out.update(alpha=alpha, med=med, bank=bank, BQ=BQ, axes={k: ST1["axes"][k] for k in ("u_T_perp", "u_S_perp", "sports", "tone", "sea", "u_neg",
               "null_u_T_perp", "null_sports", "null_tone", "null_sea", "gap_perp_cross")}, k5=ST1.get("k5"), suffix=loops["_meta"].get("suffix", ""))
    del ST1
    js = BQ["js"]; ins = F["q0"]; dis = []
    if js.get("status") != "ok": out["status"] = f"not run (B-Q0 recomputation: {js.get('status')})"; return out
    for key, f in (("M_b", lambda j: j["mode_set"]["members"]), ("guard", lambda j: j["m_star2"]["guard_pass"]),
                   ("switch", lambda j: bool(j["cross_fit"].get("switch", False))), ("mode", lambda j: j["config"]["mode"])):
        a, c = f(ins), f(js)
        if a != c: dis.append(dict(item=key, in_session=a, recomputed=c))
    cosu = R.cosf(torch.as_tensor(F["q0pt"]["cfg"]["u_push"]).double(), BQ["cfg"]["u_push"])
    if not cosu > 1 - 1e-6: dis.append(dict(item="u_push", cos_in_session_recomputed=cosu))
    out["od_g8"] = dict(agree=not dis, disagreements=dis, note="the in-session values govern what ran (OD-G8)")
    cfg = _cfg_used(F["q0pt"]["cfg"], BQ["reps"]); out["cfg"] = cfg
    G_blk = bool(summ_b and summ_b.get("verdict", {}).get("PC_PASS") is True and not summ_b.get("verdict", {}).get("gates", {}).get("failed"))
    out["G_blk"] = G_blk if summ_b else None
    if F["cells"] is None or _status(F["cells"]) != "ok":
        out["status"] = "ok (cells missing)"; out["cells_status"] = _status(F["cells_js"]) or "theta/b%d_cells.pt missing" % b; return out
    C = F["cells"]; out["cells"] = C["cells"]; out["order"] = C["order"]; out["curve"] = {float(k): float(v) for k, v in C["damage_curve"].items()}
    out["CT"] = TH.cell_table(C["cells"], cfg, alpha, W)
    if CELL_A0 in out["CT"]:
        g3, w3 = TH.ivpool([BQ["reps"]["Gamma"], out["CT"][CELL_A0]["all"]["dec"]]); out["Gamma_R3"] = torch.as_tensor(g3); out["Gamma_R3_weights"] = w3
    out["status"] = "ok"
    return out


def cells_json(core, NP):
    """Per cell and window: gains with intervals, and the covariates."""
    out = {}
    for n, c in core["cells"].items():
        CT = core["CT"][n]; rel, r = TH.halving_rel(c["windows"]["all"]["delta"], c["windows"]["all"]["ok"], c["prompt_of"], NP)
        nl = np.array(c["nll"]["cnt"], float); et = np.array(c["entropy"]["cnt"], float)
        out[n] = dict(a=c["a"], T=c["T"], nseed=c["nseed"], reserve=c.get("reserve"), n_pairs=c["n_pairs"], rel=rel, r=r,
                      windows={w: dict(g_par=S(CT[w]["g_par"]), G=S(CT[w]["G"]), rotation=S(CT[w]["rot"]), decision=CT[w]["kind"], dec=S(CT[w]["dec"]),
                                       SNR=S(CT[w]["SNR"]), n_ok=int(torch.as_tensor(c["windows"][w]["ok"]).sum())) for w in CT},
                      damage=c["damage"], KL_T=c["KL_T"], cbar=float(sum(c["cbar"]["sum"]) / max(sum(c["cbar"]["cnt"]), 1)),
                      nll_per_token=float(np.sum(c["nll"]["sum"]) / max(nl.sum(), 1)), entropy=float(np.sum(c["entropy"]["sum"]) / max(et.sum(), 1)),
                      rep4=float(np.nanmean(c["rep4"])) if np.isfinite(np.asarray(c["rep4"], float)).any() else float("nan"),
                      first_div=TH.fd_stats(c["first_div"]), capped_share=c.get("capped_share"), dropped=c.get("dropped"))
    return out


def reserves(cores):
    """OD-B8 / G21-G23 on the final joint bootstrap: which reserve cells enter their rule (an undefined bound, Amendment 3 (b), does
    not meet a reserve condition)."""
    out = {}; c14 = cores.get(14)
    if c14 and "CT" in c14:
        CT = c14["CT"]; dec = lambda n: CT[n]["all"]["dec"] if n in CT else None
        l2 = lohi(dec(TH.cell_name(2, 0.7)))[0] if dec(TH.cell_name(2, 0.7)) is not None else float("nan")
        sh = lambda n: nf_share(dec(n)) if dec(n) is not None else float("nan")
        out["reserve1_a4_T0.7"] = dict(enters=bool(l2 > 1), condition="G(2 alpha_0) lower bound > 1", value=l2, present=TH.cell_name(4, 0.7) in CT,
                                       nonfinite_share=sh(TH.cell_name(2, 0.7)))
        l10 = lohi(dec(TH.cell_name(1, 1.0)))[0] if dec(TH.cell_name(1, 1.0)) is not None else float("nan")
        h03 = lohi(dec(TH.cell_name(1, 0.3)))[1] if dec(TH.cell_name(1, 0.3)) is not None else float("nan")
        out["reserve2_T1.3"] = dict(enters=bool(l10 > 1), condition="G(1.0) lower bound > 1", value=l10, present=TH.cell_name(1, 1.3) in CT,
                                    nonfinite_share=sh(TH.cell_name(1, 1.0)))
        out["reserve2_T0.15"] = dict(enters=bool(h03 < 1), condition="G(0.3) upper bound < 1", value=h03, present=TH.cell_name(1, 0.15) in CT,
                                     nonfinite_share=sh(TH.cell_name(1, 0.3)))
    ext = {b: c for b, c in cores.items() if b != 14 and "CT" in c and CELL_A0 in c["CT"]}
    if ext:
        pts = {b: float(c["CT"][CELL_A0]["all"]["dec"][0]) for b, c in ext.items()}; best = max(pts, key=lambda b: (pts[b] if fin(pts[b]) else -np.inf, -b))
        lo = lohi(ext[best]["CT"][CELL_A0]["all"]["dec"])[0]
        out["reserve3_a2_T0.7"] = dict(block=best, point_estimates={str(k): v for k, v in pts.items()}, lower_bound=lo, enters={str(b): bool(b == best and lo > G23_LO) for b in ext},
                                       condition="only the extra block with the higher G(alpha_0) point estimate, and only if its lower bound > 0.8",
                                       nonfinite_share=nf_share(ext[best]["CT"][CELL_A0]["all"]["dec"]))
    out["reserve5_turn2_a2_T0.7"] = dict(enters=True, condition="always run; enters R6 like any carry cell (G25)")
    return out


def _entered_label(enters): return "entered" if enters else "ran, condition not met (descriptive)"


def collapse(core):
    """B-Q4's collapse comparison (descriptive only): every off-series cell's read-back F_T(alpha) = alpha g(alpha, T) (decision gain, push
    units) against the T 0.7 curve at alpha' = 0.7 alpha / T, log-log interpolated between the existing T 0.7 cells (no prediction outside
    their alpha range); difference and ratio with 90 % intervals."""
    CT, cells = core["CT"], core["cells"]; base = sorted((cells[n]["a"], n) for n in cells if abs(cells[n]["T"] - 0.7) < 1e-9)
    out = dict(rule="descriptive: F_T(alpha) against F_0.7(0.7 alpha / T), log-log interpolation inside the T 0.7 alpha grid", cells={})
    if len(base) < 2: out["note"] = "fewer than two T 0.7 cells"; return out
    a = np.array([x[0] for x in base]); lF = np.log(np.clip(np.stack([x[0] * _np(CT[x[1]]["all"]["dec"]) for x in base], 1), 1e-12, None))
    for n, c in cells.items():
        if abs(c["T"] - 0.7) < 1e-9: continue
        ap = 0.7 * c["a"] / c["T"]
        if not (a.min() * (1 - 1e-12) <= ap <= a.max() * (1 + 1e-12)): out["cells"][n] = dict(T=c["T"], a=c["a"], a_prime=ap, note="outside the T 0.7 grid: no prediction"); continue
        i = int(min(np.searchsorted(a, ap, side="right") - 1, len(a) - 2)); t = (math.log(ap) - math.log(a[i])) / (math.log(a[i + 1]) - math.log(a[i]))
        pred = np.exp(lF[:, i] * (1 - t) + lF[:, i + 1] * t); FT = c["a"] * _np(CT[n]["all"]["dec"])
        out["cells"][n] = dict(T=c["T"], a=c["a"], a_prime=ap, F_T=S(FT), F_07_at_a_prime=S(pred), diff=S(FT - pred), ratio=S(FT / pred))
    return out


def r1_series(core, b, RS, window="all"):
    """The alpha series of block b entering R1 (B-Q3 / B-Q5): block 14: cells 0.5, 1, 2 alpha_0 at T 0.7 (+ 4 alpha_0 if reserve 1 entered);
    extra blocks: the low end Gamma_b (B-Q0 pooled with the alpha_0 T 0.7 cell; for another window with that window's cell gain) and the
    2 alpha_0 cell if reserve 3 entered."""
    CT = core["CT"]; d = lambda n: CT[n][window]["dec"]
    if b == 14:
        names = [TH.cell_name(x, 0.7) for x in (0.5, 1, 2)] + ([TH.cell_name(4, 0.7)] if RS.get("reserve1_a4_T0.7", {}).get("enters") else [])
        names = [n for n in names if n in CT]; a = [core["cells"][n]["a"] for n in names]
        return a, [d(n) for n in names], (a.index(0.5) if 0.5 in a else None), names
    low = core["Gamma_R3"] if window == "all" else torch.as_tensor(TH.ivpool([core["BQ"]["reps"]["Gamma"], d(CELL_A0)])[0])
    a, reps, names = [1.0], [low], ["Gamma_b"]
    r3 = RS.get("reserve3_a2_T0.7", {}).get("enters", {})
    if r3.get(str(b)) and TH.cell_name(2, 0.7) in CT: a.append(2.0); reps.append(d(TH.cell_name(2, 0.7))); names.append(TH.cell_name(2, 0.7))
    return a, reps, None, names


def power_rerun(core, b, names, a_cells, NP, reps=POWER_REPS):
    """OD-B14: r2_sim.gain_A2 and r2_sim.resolvable_band with the measured per-cell SD of the per-pair projections on u_read (in push units:
    u_read . delta_j / (alpha_b cbar_c m2_par), whose mean is a g_par), their measured correlation across cells (pairs with the same keys
    (p, s)), the cells' ok-pair counts and sim_gain's scenarios. At an extra block the low end Gamma_b is represented by its alpha_0 cell."""
    import r2_sim
    cfg = core["cfg"]; u = cfg["u_read"]; m2 = float(torch.as_tensor(cfg["m2_par"])[0]); xs, sd, n = [], [], []
    cells = [core["cells"][CELL_A0 if nm == "Gamma_b" else nm] for nm in names]
    for c in cells:
        w = c["windows"]["all"]; ok = torch.as_tensor(w["ok"], dtype=torch.bool); cb = sum(c["cbar"]["sum"]) / max(sum(c["cbar"]["cnt"]), 1)
        x = (torch.as_tensor(w["delta"]).double().nan_to_num() @ R.unit(u)) / (core["alpha"] * cb * m2)
        key = {(int(p), int(s)): float(v) for p, s, v, k in zip(c["prompt_of"].tolist(), c["seed_of"].tolist(), x.tolist(), ok.tolist()) if k}
        xs.append(key); sd.append(float(np.std(list(key.values()), ddof=1)) if len(key) > 1 else float("nan")); n.append(len(key))
    rh = []
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            ks = sorted(set(xs[i]) & set(xs[j]))
            if len(ks) > 2: rh.append(float(np.corrcoef([xs[i][k] for k in ks], [xs[j][k] for k in ks])[0, 1]))
    rho = float(np.nanmean(rh)) if rh else 0.0; rho = min(max(rho, -0.99), 0.99) if fin(rho) else 0.0
    out = dict(a_cells=list(map(float, a_cells)), cells=names, sd=sd, rho=rho, n_pairs=n, reps=reps, units="push units: u_read . delta_j / (alpha_b cbar m2_par)")
    if not all(fin(s) and s > 0 for s in sd) or not n: out["status"] = "not run (no per-pair spread)"; return out
    t0 = time.time(); out["scenarios"] = r2_sim.gain_A2(np.asarray(a_cells, float), np.asarray(n), np.asarray(sd), rho, reps=reps)
    band = r2_sim.resolvable_band(np.asarray(a_cells, float), np.asarray(n), np.asarray(sd), rho, reps=reps); out["resolvable_band"] = dict(lo=band["lo"], hi=band["hi"])
    out["band_rate"] = dict(grid=band["grid"], rate=band["rate"]); out["seconds"] = time.time() - t0; out["status"] = "ok"
    return out


def r4_block(core, b, F, AXB, labels):
    """R4 critical mode with B-Q2 alignment (default u_crit; (i) converged different loops; (ii) the power-step mode). Amendment 3 (c):
    a 'degenerate' (i) has no single critical mode: no '(RSI-specific)' suffix, and core['align_reps'] stays empty so that the block does
    not enter R3's alignment form (alignment_entries); the per-loop alignments stay in out['i']['alignment'] (descriptive)."""
    BQ, cfg = core["BQ"], core["cfg"]; js = BQ["js"]; reps = BQ["reps"]; bank = core["bank"]; P = BQ["P"]; ax = core["axes"]
    axes = dict(u_T_perp=(ax["u_T_perp"], ax["null_u_T_perp"]), sports=(ax["sports"], ax["null_sports"]), tone=(ax["tone"], ax["null_tone"]),
                sea=(ax["sea"], ax["null_sea"]), u_S_perp=(ax["u_S_perp"], None if AXB is None else AXB.get("null_u_S_perp")),
                u_neg=(ax["u_neg"], None if AXB is None else AXB.get("null_u_neg")),
                u_RGI_perp=(None, None) if AXB is None else (AXB.get("u_RGI_perp"), AXB.get("null_u_RGI_perp")))
    ga = None if AXB is None else AXB.get("GA"); M = js["mode_set"]["members"]; sg = js["mode_set"]["signs"]; out = dict(labels=list(labels))
    m2 = reps["m2"]
    if cfg["mode"] == "switch":
        Ucrit = P[("RSI_a", 3)].mean(); mc = torch.sqrt(reps["mA2"]); out["u_crit"] = "u*_A (switch case: push direction of B-Q0, read along u*_B)"
    else:
        Ucrit = sum(sg[L] * P[(L, 3)].mean() for L in M); mc = torch.sqrt(m2); out["u_crit"] = "u*_b"
    out["default"] = alignment(Ucrit, mc, axes, core.get("k5"), ga)
    rel4 = js["mode_set"]["rel_d4"]; conv = [L for L in js["loops"] if L != "RSI_a" and fin(rel4.get(L)) and rel4[L] >= 0.5]
    if M == ["RSI_a"] and conv:
        gam = {L: torch.as_tensor(TH.ivpool([reps["gains"][(L, n)]["dec"] for n in (2, 3)])[0]) for L in ["RSI_a"] + conv}
        iv = {L: lohi(g) for L, g in gam.items()}; best = max(gam, key=lambda L: float(gam[L][0]) if fin(float(gam[L][0])) else -np.inf)
        first = all(iv[best][0] > iv[L][1] for L in gam if L != best)
        out["i"] = dict(applies=True, converged=conv, Gamma={L: S(g) for L, g in gam.items()}, verdict=f"'{best}' first" if first else "degenerate",
                        nonfinite_share={L: nf_share(g) for L, g in gam.items()},
                        alignment={L: alignment(P[(L, 3)].mean(), math.sqrt(rel4[L]), axes, core.get("k5"), ga) for L in ["RSI_a"] + conv})
        if first and best != "RSI_a":
            out["u_crit"] = f"unit(d_4^{best}) (R4 (i))"; out["default"] = out["i"]["alignment"][best]
        elif not first: out["u_crit"] = "degenerate (R4 (i)): alignment per converged loop, descriptive"
    else: out["i"] = dict(applies=False, converged=conv, M_b=M)
    pw = F.get("power")
    if pw is not None and pw.get("status") == "ran" and F.get("power_pt") is not None:
        D = F["power_pt"]["data"]; Wn = core["W"]; Pp = TH.Pairs(D["windows"]["all"]["delta"], D["windows"]["all"]["ok"], D["prompt_of"], Wn)
        rel_p, _ = TH.halving_rel(D["windows"]["all"]["delta"], D["windows"]["all"]["ok"], D["prompt_of"], core["NP"])
        c = core["cells"][pw["cell"]]; rel_c, _ = TH.halving_rel(c["windows"]["all"]["delta"], c["windows"]["all"]["ok"], c["prompt_of"], core["NP"])
        g = TH.power_gains(Pp, F["power_pt"]["push"].double(), rel_c, D["a"], core["alpha"], TH.cell_cbar(D, Wn))
        us = BQ["tensors"]["u_star"]; cosd = R.cosf(Pp.mean()[0], us); corr = abs(cosd) / math.sqrt(rel_p * float(m2[0])) if rel_p > 0 and float(m2[0]) > 0 else float("nan")
        where = "T" if abs(D["T"] - 0.7) > 1e-9 else ("alpha" if abs(D["a"] - 1) > 1e-9 else "alpha_0, T 0.7")
        out["ii"] = dict(applies=True, cell=pw["cell"], rel_cell=rel_c, rel_power=rel_p, g_par=S(g["g_par"]), G=S(g["G"]), rotation=S(g["rot"]), decision=g["kind"],
                         dec=S(g["dec"]), cos_d_power_u_star=cosd, corrected_abs_cos=corr,
                         verdict=(f"the critical mode changes with {where}" if fin(corr) and corr < PWR_COS else "the mode on top is u*_b's (corrected |cos| >= 0.7)"),
                         alignment=alignment(Pp.mean(), math.sqrt(rel_p) if rel_p > 0 else float("nan"), axes, core.get("k5"), ga))
    else: out["ii"] = dict(applies=False, power=None if pw is None else pw.get("status"))
    d = out["default"]; rnd = [L for L in js["loops"] if L.startswith("RND")]
    deb = {L: abs(R.cosf(bank[L]["d"][4].double(), BQ["tensors"]["u_star"])) / math.sqrt(rel4[L] * float(m2[0])) for L in rnd if fin(rel4.get(L)) and float(m2[0]) > 0}
    generic = (not d["RSI_aligned"]) and (not d["GI_aligned"]) and any(v >= GENERIC_COS for v in deb.values())
    if out["u_crit"].startswith("degenerate"): v = "degenerate"
    elif d["RSI_aligned"] and d["GI_aligned"]: v = "RSI-aligned and GI-aligned"
    elif d["RSI_aligned"]: v = "RSI-aligned"
    elif d["GI_aligned"]: v = "GI-aligned"
    elif generic: v = "generic attractor"
    else: v = "unaligned"
    deg = out["u_crit"].startswith("degenerate")              # R4 (i): no single critical mode (Amendment 3 (c))
    if d["RSI_specific"] and not deg: v += " (RSI-specific)"
    out.update(verdict=v, random_start_debiased_cos=deb, labels=out["labels"] + d["labels"])
    core["align_reps"] = {} if deg else d.get("_reps", {})
    for x in [out["default"], out.get("ii", {}).get("alignment")] + list(out.get("i", {}).get("alignment", {}).values()):
        if isinstance(x, dict): x.pop("_reps", None)
    return out


def alignment_entries(E, blocks, cores, dl):
    """Entries of R3's alignment form (R4: 'alignment per block against delta in the R3 form'): every entering block with an R4 verdict,
    with the sign-aligned corrected cosine of its critical mode with u_T_perp (and with u_RGI_perp). A block whose R4 (i) is 'degenerate'
    has no single critical mode: it does not enter and is listed as 'not entering (degenerate)' (Amendment 3 (c)).
    Returns (RSI entries, GI entries, {block: 'not entering (degenerate)'})."""
    al, ag, deg = [], [], {}
    for b in E:
        r4 = blocks[str(b)].get("R4", {}); v = r4.get("verdict")
        if v is None: continue
        if str(v).startswith("degenerate"): deg[str(b)] = "not entering (degenerate)"; continue
        rp = cores[b].get("align_reps", {})
        if rp.get("u_T_perp") is not None: c = rp["u_T_perp"]; al.append((b, dl[b], torch.sign(c[0]) * c))
        if rp.get("u_RGI_perp") is not None: c = rp["u_RGI_perp"]; ag.append((b, dl[b], torch.sign(c[0]) * c))
    return al, ag, deg


def turn2_block(core, T2, W, labels):
    """B-Q8 / R6 at block 14: K, the norm ratio, K on u_T_perp, R6 per carry cell, the T-slope of log K."""
    u = core["cfg"]["u_read"]; uT = core["axes"]["u_T_perp"]; out = dict(cells={}, labels=list(labels))
    for name, t in T2["cells"].items():
        c = core["cells"][t["cell"]]; P1 = TH.Pairs(c["windows"]["all"]["delta"], c["windows"]["all"]["ok"], c["prompt_of"], W)
        P2 = TH.Pairs(t["delta2"], t["ok2"], t["prompt_of"], W); d1, d2 = P1.proj(u), P2.proj(u); K = d2 / d1
        KT = P2.proj(uT) / P1.proj(uT); nr = torch.sqrt(P2.U().clamp(min=0) / P1.U())
        r6 = rule_R6(K, d1, core["CT"][t["cell"]]["all"]["dec"])
        out["cells"][name] = dict(a=t["a"], T=t["T"], reserve=t.get("reserve"), n_pairs=len(t["keys"]), n_ok2=int(torch.as_tensor(t["ok2"]).sum()),
                                  n_turn1_not_round_trip=t.get("n_turn1_not_round_trip"), R6=r6, norm_ratio=S(nr), K_u_T_perp=S(KT), _K=K)
    a1 = [n for n, x in out["cells"].items() if abs(x["a"] - 1) < 1e-9 and x["R6"]["defined"] and lohi(x["_K"])[0] > 0]
    if len(a1) >= 2:
        r = rule_R2([out["cells"][n]["T"] for n in a1], [out["cells"][n]["_K"] for n in a1]); out["T_slope_logK"] = dict(cells=a1, slope=r["slope"], fit_verdict=r["verdict"], T_c=r["T_c"])
    else: out["T_slope_logK"] = dict(cells=a1, verdict="not computed (fewer than two alpha_0 cells with K's interval above 0)")
    for x in out["cells"].values(): x.pop("_K")
    return out


def _f(x):
    return "nan" if not (fin(x) or x in (float("inf"), float("-inf"))) else f"{x:.3g}"


def sentence(b, R1b, R2b, R4b, T2, R3, delta_order):
    """The per-block statement in the user's form (4.1)."""
    if R2b is None: return f"Block {b}: not run."
    pts = ", ".join(f"{T:g}: {_f(p['pt'])} [{_f(p['lo'])}, {_f(p['hi'])}]" for T, p in zip(R2b["T"], R2b["points"]))
    sl = R2b.get("slope", {}); s = f"Block {b}: at push alpha_0 the self-steering gain is G(T) = {pts}"
    if sl: s += f" (slope dlogG/dT {_f(sl['pt'])} [{_f(sl['lo'])}, {_f(sl['hi'])}])"
    v = R2b.get("verdict"); rng = f"[{min(R2b['T']):g}, {max(R2b['T']):g}]"
    if v == "T_c located": s += f"; it crosses 1 at T_c = {_f(R2b['T_c']['pt'])} [{_f(R2b['T_c']['lo'])}, {_f(R2b['T_c']['hi'])}]"
    elif v == "below 1 throughout": s += f"; it stays below 1 over {rng} (upper 90 % band at most {_f(R2b['margin']['max_upper_band'])}, i.e. by at least {_f(1 - R2b['margin']['max_upper_band'])})"
    elif v == "above 1 throughout": s += f"; it stays above 1 over {rng} (lower 90 % band at least {_f(R2b['margin']['min_lower_band'])}, i.e. by at least {_f(R2b['margin']['min_lower_band'] - 1)})"
    else: s += f"; whether it crosses 1 over {rng} is {v}"
    s += ". "
    if T2 is not None and T2.get("cells"):
        ks = ", ".join(f"{x['T']:g}{'' if abs(x['a'] - 1) < 1e-9 else ' (2 alpha_0)'}: " + (f"{_f(x['R6']['K']['pt'])} [{_f(x['R6']['K']['lo'])}, {_f(x['R6']['K']['hi'])}] ({x['R6']['verdict']})" if x["R6"]["defined"] else "not defined")
                       for x in T2["cells"].values())
        s += f"The model's own text carries K(T) = {ks} of its stance into the next turn. "
    elif b == 14: s += "The in-context carry K was not measured. "
    else: s += "The in-context carry K is measured at block 14 only. "
    if not R4b or (R4b.get("verdict") is None and R4b.get("verdict_descriptive") is None): mode = "not resolved (no R4 at this block)"
    elif R4b.get("verdict") is not None: mode = R4b["verdict"]
    else: mode = f"{R4b['verdict_descriptive']} (descriptive: G-blk failed)"
    if R4b and R4b.get("ii", {}).get("applies") and str(R4b["ii"].get("verdict", "")).startswith("the critical mode changes"): mode += f"; at cell {R4b['ii']['cell']} {R4b['ii']['verdict']}"
    onset = (R1b or {}).get("onset", {}).get("verdict", "onset undecided")
    s += f"The critical mode is {mode}, with {onset if onset != 'onset undecided' else 'undecided'} onset" + (" (R1: " + R1b["verdict"] + ")" if R1b else "")
    s += f", and this changes with delta as follows: {R3.get('Gamma', {}).get('verdict', 'not decidable')}; alignment: {R3.get('alignment_RSI', {}).get('verdict', 'not decidable')}"
    s += f" (blocks ordered by delta: {delta_order}; {DEPTH})."
    return s


# ======================================================================================================================================
# Main analysis
# ======================================================================================================================================
def analyse(r1_dir, r2_root, power_reps=POWER_REPS):
    t0 = time.time(); info = R.r1_blocks(r1_dir); order = R.delta_order(r1_dir, info); B3 = info["B3"]
    out = dict(script="r2_theta_verdict.py", plan=TH.PLAN, plan_sha256=TH.PLAN_SHA256, seed_boot=TH.SEED_BOOT, n_boot=TH.N_BOOT, level=TH.LEVEL,
               B3=B3, delta_order=order, block_flags=info["flags"], power_reps=power_reps, labels_global=[], note=G1_NOTE)
    summ = {b: R.load_r1_json(r1_dir, f"summary_b{b}.json", b, required=False) for b in B3}
    F = {b: load_theta(r2_root, b) for b in B3}
    axp = find_r2(r2_root, "axes/r2_axes.pt", "r2_q1"); AX = _ld(axp); out["inputs"] = dict(r2_axes=axp, theta={b: F[b]["paths"] for b in B3})
    # gates: G20 (code checks), G11 (G-desc), G19 (gate p), G12 (G-blk)
    ck = F[14]["checks"] if 14 in F else None; lab = []
    if ck is None or ck.get("passed") is None: lab.append("descriptive (code check not run)"); g20 = None
    else: g20 = bool(ck["passed"]); lab += [] if g20 else ["descriptive (code check failed)"]
    out["gates"] = dict(G20=dict(passed=g20, checks=None if ck is None else {k: ck.get(k) for k in ("zero_push_identical", "mirror_swapped", "entropy_T03", "entropy_T07")}))
    s14 = summ.get(14); gdesc = bool(s14 and s14.get("verdict", {}).get("gates", {}).get("failed"))
    out["gates"]["G11_G_desc"] = dict(failed=gdesc, label="descriptive (round-1 gate failed)" if gdesc else None, gates_failed=None if not s14 else s14["verdict"].get("gates", {}).get("failed"))
    out["gates"]["G12_G_blk"] = {str(b): (None if summ[b] is None else bool(summ[b]["verdict"].get("PC_PASS") is True and not summ[b]["verdict"].get("gates", {}).get("failed"))) for b in B3}
    NP = next((int(F[b]["q0pt"]["NP"]) for b in B3 if F[b]["q0pt"] is not None and F[b]["q0pt"].get("NP")), None)
    if NP is None:
        out.update(status="not run (no B-Q0 output)", blocks={str(b): dict(status="not run") for b in B3}); return out
    W = TH.boot_W(NP); cores = {}
    for b in B3:
        cores[b] = block_core(r1_dir, b, F[b], W, summ[b]); cores[b]["W"] = W; cores[b]["NP"] = NP
    q14 = cores.get(14, {}).get("BQ", {}).get("js", {}).get("gate_p") if cores.get(14, {}).get("BQ") else None
    out["gates"]["G19_gate_p"] = q14
    if q14 and q14.get("label"): lab.append(q14["label"])
    out["labels_global"] = lab
    RS = reserves(cores); out["reserves"] = RS
    blocks = {}
    for b in B3:
        c = cores[b]; bl = list(lab) + (["descriptive (round-1 gate failed)"] if (b == 14 and gdesc) else []); res = dict(status=c.get("status"), labels=bl)
        if "BQ" in c and c["BQ"]["js"].get("status") == "ok":
            res["B_Q0"] = c["BQ"]["js"]; res["od_g8"] = c.get("od_g8"); res["mode"] = c["cfg"]["mode"]; res["G_blk"] = c.get("G_blk")
            res["labels"] += [x for x in F[b]["q0"].get("labels", []) if x not in res["labels"]]      # in-session labels govern (OD-G8)
        if c.get("status") == "ok":
            res["cells"] = cells_json(c, NP)
            for nm, cj in res["cells"].items():
                rv = cj.get("reserve")
                if rv == "reserve 1": cj["entered"] = _entered_label(RS.get("reserve1_a4_T0.7", {}).get("enters"))
                elif rv == "reserve 2": cj["entered"] = _entered_label(RS.get("reserve2_T1.3" if cj["T"] > 1 else "reserve2_T0.15", {}).get("enters"))
                elif rv == "reserve 3": cj["entered"] = _entered_label(RS.get("reserve3_a2_T0.7", {}).get("enters", {}).get(str(b)))
            res["Gamma_R3"] = dict(**S(c["Gamma_R3"]), weights=c["Gamma_R3_weights"], rule="B-Q0 free estimate pooled with the alpha_0 T 0.7 cell (inverse variance)") if "Gamma_R3" in c else None
            a, reps_, half, names = r1_series(c, b, RS); wr = {w: r1_series(c, b, RS, w)[1] for w in TH.WINDOWS}
            pr = power_rerun(c, b, names, a, NP, power_reps); res["power_rerun"] = pr
            r1 = rule_R1(a, reps_, half, c["curve"], wr, pr.get("resolvable_band")); r1["series"] = names; r1["labels"] = list(res["labels"]) + r1["labels"]
            res["R1"] = r1
            Tn = [TH.cell_name(1, T) for T in (0.3, 0.7, 1.0)]
            ext = False
            if b == 14:
                if RS.get("reserve2_T1.3", {}).get("enters"): Tn.append(TH.cell_name(1, 1.3)); ext = True
                if RS.get("reserve2_T0.15", {}).get("enters"): Tn.append(TH.cell_name(1, 0.15)); ext = True
            Tn = [n for n in Tn if n in c["CT"]]
            r2 = rule_R2([c["cells"][n]["T"] for n in Tn], [c["CT"][n]["all"]["dec"] for n in Tn], ext)
            r2["cells"] = Tn; r2["labels"] = list(res["labels"])
            r2["g_par_fit"] = {k: v for k, v in rule_R2([c["cells"][n]["T"] for n in Tn], [c["CT"][n]["all"]["g_par"] for n in Tn], ext).items() if k in ("verdict", "slope", "T_c")}
            r2["G_norm_fit"] = {k: v for k, v in rule_R2([c["cells"][n]["T"] for n in Tn], [c["CT"][n]["all"]["G"] for n in Tn], ext).items() if k in ("verdict", "slope", "T_c")}
            r2["rotation_with_T"] = {str(c["cells"][n]["T"]): S(c["CT"][n]["all"]["rot"]) for n in Tn}
            r2["collapse"] = collapse(c)
            res["R2"] = r2
            rel4 = c["BQ"]["js"]["mode_set"]["rel_d4"].get("RSI_a"); res["R5"] = rule_R5(rel4, c.get("Gamma_R3", c["BQ"]["reps"]["Gamma"]))
            noind = "no independent loop" in F[b]["q0"].get("labels", []); entering = bool(c.get("G_blk")) and c["cfg"]["mode"] != "not_resolved" and not noind
            why = [] if entering else [x for x, cond in (("G-blk failed or summary missing", not c.get("G_blk")), ("mode not resolved", c["cfg"]["mode"] == "not_resolved"),
                                                         ("no independent loop (G49)", noind)) if cond]
            res["enters_R3"] = dict(enters=entering, reasons=why)
            AXB = None if AX is None else AX.get("blocks", {}).get(b)
            if c["cfg"]["mode"] != "not_resolved":
                res["R4"] = r4_block(c, b, F[b], AXB, res["labels"]); res["B_Q2"] = res["R4"]["default"]
                if not c.get("G_blk"): res["R4"]["labels"].append("descriptive (G-blk failed: no R4 verdict at this block)"); res["R4"]["verdict_descriptive"] = res["R4"].pop("verdict")
            else: res["R4"] = dict(verdict=None, note="mode not resolved: no R4 at this block (OD-B1)")
            # B-Q7
            Fd = F[b]["fdt"]
            if Fd is not None and Fd.get("status") == "ok":
                cbr = c["BQ"]["reps"]["cbar"]; dirs = dict(u_star=c["BQ"]["tensors"]["u_star"], u_push=c["cfg"]["u_push"], u_read=c["cfg"]["u_read"],
                                                         u_T_perp=c["axes"]["u_T_perp"], sea=c["axes"]["sea"],
                                                         u_RGI_perp=None if AXB is None else AXB.get("u_RGI_perp"),
                                                         cov_b_4=R.pool_dir("cov", b, 4, D=int(torch.as_tensor(c["axes"]["u_T_perp"]).shape[0])))
                res["B_Q7"] = dict(lambda_lin=lambda_lin(Fd["X"], Fd["S"], Fd["prompt_of"], W, cbr, dirs), n_texts=int(len(Fd["conts"])),
                                   n_dropped=int((~torch.isfinite(torch.as_tensor(Fd["X"])).all(1)).sum()), rule="descriptive (simulation: unbiased, SD 25-50 % at 128 texts)",
                                   note="J_hat = Cov(x, S) / cbar_b; cbar_b from the B-Q0 reference read; 'cov_b_4' = Pool ('cov', b, 4)")
            else: res["B_Q7"] = dict(status=None if Fd is None else Fd.get("status"))
            if b == 14 and F[14].get("turn2") is not None and F[14]["turn2"].get("status") == "ok":
                res["R6"] = turn2_block(c, F[14]["turn2"], W, res["labels"])
            pw = F[b].get("power"); res["B_Q6"] = None if pw is None else {k: pw.get(k) for k in ("status", "cell", "a", "T", "candidates", "dec", "rotation")}
            if pw is not None and c.get("cells"):
                ch = TH.choose_power_cell(TH.cell_points(c["cells"], c["cfg"], c["alpha"], NP), c.get("order"))
                agree = (pw.get("status") == "not needed" and ch is None) or (pw.get("status") == "ran" and pw.get("cell") == ch)
                res["od_g8"]["power_cell"] = dict(in_session=pw.get("cell") if pw.get("status") == "ran" else None, recomputed=ch, agree=agree)
                if not agree: res["od_g8"]["agree"] = False
        res["descriptive"] = any(str(x).startswith("descriptive") for x in res["labels"])
        blocks[str(b)] = res
    # R3 over blocks and alignment against delta
    dl = order["delta"]; E = [b for b in B3 if blocks[str(b)].get("enters_R3", {}).get("enters")]
    R3 = dict(entering=E, note=DEPTH, not_entering={str(b): dict(reasons=blocks[str(b)].get("enters_R3", {}).get("reasons", [blocks[str(b)].get("status")]),
                                                                      Gamma=blocks[str(b)].get("Gamma_R3"), mode=blocks[str(b)].get("mode"),
                                                                      note="gains along the block's push direction (unit(d_4^RSI_a) when the mode is not resolved), reported next to the verdict")
                                                      for b in B3 if b not in E})
    R3["Gamma"] = rule_R3([(b, dl[b], cores[b]["Gamma_R3"]) for b in E if "Gamma_R3" in cores[b]], "loop gain")
    for T in (0.3, 0.7, 1.0):
        nm = TH.cell_name(1, T); R3[f"g_alpha0_T{T:g}"] = rule_R3([(b, dl[b], cores[b]["CT"][nm]["all"]["dec"]) for b in E if nm in cores[b].get("CT", {})], f"g(alpha_0, T {T:g})")
    sl = []
    for b in E:
        r2 = blocks[str(b)].get("R2", {}); Tn = r2.get("cells", [])
        if len(Tn) >= 2:
            Y = np.log(np.clip(np.stack([_np(cores[b]["CT"][n]["all"]["dec"]) for n in Tn], 1), 1e-6, None)); Bf, _ = wls([cores[b]["cells"][n]["T"] for n in Tn], Y)
            sl.append((b, dl[b], Bf[:, 1]))
    R3["T_slope"] = rule_R3(sl, "dlogG/dT")
    Ds = []
    for b in E:
        r1 = blocks[str(b)].get("R1", {})
        if r1.get("verdict") in ("FOUND-soft", "bistable window (FOUND-soft and FOUND-hard)") and fin(r1.get("D_star", {}).get("pt")):
            a, reps_, _, _ = r1_series(cores[b], b, RS); dn, _ = crossings(a, np.stack([_np(r) for r in reps_], 1))
            Ds.append((b, dl[b], np.array([damage_at(x, cores[b]["curve"]) if not np.isnan(x) else np.nan for x in dn])))
    R3["D_star"] = rule_R3(Ds, "D*") if len(Ds) >= 2 else dict(verdict="not decidable (D* located at fewer than two blocks)", blocks=[x[0] for x in Ds])
    al, ag, deg = alignment_entries(E, blocks, cores, dl)
    R3["alignment_RSI"] = dict(rule_R3(al, "RSI alignment (sign-aligned corrected cos with u_T_perp)"), not_entering=dict(deg))
    R3["alignment_GI"] = dict(rule_R3(ag, "GI alignment (sign-aligned corrected cos with u_RGI_perp)"), not_entering=dict(deg))
    out["R3"] = R3; out["blocks"] = blocks
    for b in B3:
        r = blocks[str(b)]; r["sentence"] = sentence(b, r.get("R1"), r.get("R2"), r.get("R4"), r.get("R6"), R3, order["order"]) if r.get("R2") else f"Block {b}: {r.get('status')}"
    out["status"] = "ok"; out["seconds"] = time.time() - t0
    return out


def main(argv):
    if len(argv) >= 2 and argv[1] == "selftest":
        if len(argv) < 3: raise SystemExit("usage: python r2_theta_verdict.py selftest <workdir> [<r1_dir> <r2_root>]")
        return 0 if selftest(argv[2], *(argv[3:5] if len(argv) >= 5 else ())) else 1
    if len(argv) < 4: raise SystemExit(__doc__)
    r1_dir, r2_root, out = argv[1:4]
    if os.path.exists(out): R.log(f"{out} exists; skipped"); return 0
    res = analyse(r1_dir, r2_root); res["finder"] = list(R.FINDER_LOG); R.write_json(out, res)
    R.log("theta verdict: " + "; ".join(f"b{b} R1 {v.get('R1', {}).get('verdict')}, R2 {v.get('R2', {}).get('verdict')}" for b, v in res.get("blocks", {}).items()))
    return 0


# ======================================================================================================================================
# Code test: synthetic cells with planted gains, planted loops, end to end on MOCK outputs
# ======================================================================================================================================
def synth_cell(rng_seed, a, T, g, NP=64, nseed=2, D=24, u=None, alpha=1.0, cbar=1.0, sd=0.6, h=0.3, crn=None):
    """A synthetic theta cell with planted gain g along u: delta_j = a alpha cbar g (1 + h eta_p + 0.2 z_ps) u + noise (sd a alpha cbar per
    coordinate). crn = (eta [NP], z [NP, 4], E [NP, 4, D]) shares the noise of pairs with the same (p, s) across cells (common keys)."""
    rng = np.random.default_rng(rng_seed); u = R.unit(torch.as_tensor(u).double()).numpy()
    eta, z, E = crn if crn is not None else (rng.normal(size=NP), rng.normal(size=(NP, 4)), rng.normal(size=(NP, 4, D)))
    po = np.repeat(np.arange(NP), nseed); so = np.tile(np.arange(nseed), NP); sc = a * alpha * cbar
    delta = sc * g * (1 + h * eta[po] + 0.2 * z[po, so])[:, None] * u[None] + sd * sc * E[po, so]
    ok = torch.ones(len(po), dtype=torch.bool); dt = torch.as_tensor(delta, dtype=torch.float32)
    return dict(name=TH.cell_name(a, T), a=float(a), T=float(T), nseed=nseed, n_pairs=len(po), prompt_of=torch.as_tensor(po), seed_of=torch.as_tensor(so),
                windows={w: dict(delta=dt.clone(), ok=ok.clone()) for w in TH.WINDOWS}, cbar=dict(sum=[cbar * 40.0] * (2 * len(po)), cnt=[40] * (2 * len(po))),
                nll=dict(sum=[1.0] * (2 * len(po)), cnt=[1] * (2 * len(po))), entropy=dict(sum=[1.0] * (2 * len(po)), cnt=[1] * (2 * len(po))),
                rep4=[0.0] * (2 * len(po)), first_div=[1] * len(po), damage=0.01 * a * a, KL_T=0.0, capped_share=0.0, dropped=0, conts=[[1, 2]] * (2 * len(po)))


def synth_bank(seed, lam, noise, bulk=0.0, D=48, NP=64, K=8, loops=(("RSI_a", 2), ("RSI_b", 2), ("SEA", 1), ("SPO", 1), ("RND_1", 1), ("RND_2", 1))):
    """A planted-operator O3 bank (theta design crit_bulk / sim_gain A4): J = lam w w' + bulk P_K, normalised loops as in rk_loop (push =
    unit(d_n), alpha = cbar = 1, read-back delta_j = J p (1 + 0.5 eta_j) + noise_j (isotropic sd noise per coordinate), 4 rounds, d_(n+1) =
    mean, rel by the K2 halvings (rk.stats)). Starts: RSI loops at the axis a0, the others random. Returns (bank, ref_X, axes, w)."""
    rng = np.random.default_rng(seed); w = rng.normal(size=D); w /= np.linalg.norm(w)
    Q, _ = np.linalg.qr(rng.normal(size=(D, K))); Q -= np.outer(w, w @ Q); Q, _ = np.linalg.qr(Q)
    J = lam * np.outer(w, w) + bulk * Q @ Q.T; a0 = rng.normal(size=D); a0 = a0 - 0.3 * (a0 @ w) * w; a0 /= np.linalg.norm(a0)
    bank = dict(_meta=dict(alpha=1.0, alpha_scale=1.0, suffix="", block=14), _missing=[])
    for L, ns in loops:
        st = a0 if L.startswith("RSI") else rng.normal(size=D); st = st / np.linalg.norm(st); p = st.copy(); e = dict(name=L, suffix="", start=torch.as_tensor(st), d={}, ev={}, rel={}, row={})
        for n in range(4):
            po = np.repeat(np.arange(NP), ns); N = len(po); Jp = J @ p
            dl = Jp[None] * (1 + 0.5 * rng.normal(size=N))[:, None] + noise * rng.normal(size=(N, D)); x0 = rng.normal(size=(N, D))
            dt = torch.as_tensor(dl, dtype=torch.float32); ok = torch.ones(N, dtype=torch.bool); d = dl.mean(0)
            rel, _ = TH.halving_rel(dt, ok, torch.as_tensor(po), NP)
            e["ev"][n + 1] = dict(delta=dt, ok=ok, prompt_of=torch.as_tensor(po), xp=torch.as_tensor(x0 + dl, dtype=torch.float32),
                                  xm=torch.as_tensor(x0 - dl, dtype=torch.float32), first_div=[1] * N, damage_of_push=0.01, alpha=1.0)
            e["d"][n + 1] = torch.as_tensor(d, dtype=torch.float32).double(); e["rel"][n + 1] = rel; e["row"][n + 1] = dict(rel=rel); p = d / np.linalg.norm(d)
        bank[L] = e
    ref_X = torch.as_tensor(rng.normal(size=(NP, D)), dtype=torch.float32)
    axes = dict(u_T_perp=torch.as_tensor(a0), gap_perp_cross=1.0)
    return bank, ref_X, axes, w


def synth_two_mode_core(seed, lam_rsi, lam_sea, NP=64, D=400, noise=0.10):
    """R4 (i) case of the review (finding theta_verdict-2, verifier's construction): RSI_a on mode w1, SEA on an orthogonal mode w2, RND_1
    without response; M_b = {RSI_a} and SEA converged. Equal gains give 'degenerate'. Returns (bq0 js, r4_block output, core)."""
    rng = np.random.default_rng(seed); Q, _ = np.linalg.qr(rng.normal(size=(D, 4))); w1, w2, wg, wx = Q.T; W = TH.boot_W(NP)
    bank = dict(_meta=dict(alpha=1.0, alpha_scale=1.0, suffix="", block=8), _missing=[])
    for L, ns, lam, w in (("RSI_a", 2, lam_rsi, w1), ("SEA", 1, lam_sea, w2), ("RND_1", 1, 0.0, wx)):
        p = w.copy(); e = dict(name=L, suffix="", start=torch.as_tensor(w.copy()), d={}, ev={}, rel={}, row={})
        for n in range(4):
            po = np.repeat(np.arange(NP), ns); N = len(po); Jp = lam * (w @ p) * w
            dl = Jp[None] * (1 + 0.5 * rng.normal(size=N))[:, None] + noise * rng.normal(size=(N, D)); x0 = rng.normal(size=(N, D))
            dt = torch.as_tensor(dl, dtype=torch.float32); ok = torch.ones(N, dtype=torch.bool); d = dl.mean(0); rel, _ = TH.halving_rel(dt, ok, torch.as_tensor(po), NP)
            e["ev"][n + 1] = dict(delta=dt, ok=ok, prompt_of=torch.as_tensor(po), xp=torch.as_tensor(x0 + dl, dtype=torch.float32), xm=torch.as_tensor(x0 - dl, dtype=torch.float32),
                                  first_div=[1] * N, damage_of_push=0.01, alpha=1.0)
            e["d"][n + 1] = torch.as_tensor(d, dtype=torch.float32).double(); e["rel"][n + 1] = rel; e["row"][n + 1] = dict(rel=rel); p = d / np.linalg.norm(d)
        bank[L] = e
    BQ = TH.bq0(bank, 8, 1.0, 1.0, dict(sum=[40.0] * NP, cnt=[40] * NP), torch.as_tensor(rng.normal(size=(NP, D)), dtype=torch.float32),
                dict(u_T_perp=torch.as_tensor(w1), gap_perp_cross=1.0), W)
    nul = torch.as_tensor(rng.normal(size=(2000, D)))
    ax = dict(u_T_perp=torch.as_tensor(w1), u_S_perp=torch.as_tensor(w1 + 0.1 * wx), sports=torch.as_tensor(rng.normal(size=D)), tone=torch.as_tensor(rng.normal(size=D)),
              sea=torch.as_tensor(w2), u_neg=torch.as_tensor(rng.normal(size=D)), null_u_T_perp=nul, null_sports=nul, null_tone=nul, null_sea=nul, gap_perp_cross=1.0)
    core = dict(BQ=BQ, cfg=_cfg_used(dict(mode=BQ["cfg"]["mode"], u_push=BQ["cfg"]["u_push"], u_read=BQ["cfg"]["u_read"]), BQ["reps"]), bank=bank, axes=ax, k5=None,
                W=W, NP=NP, cells={}, alpha=1.0)
    AXB = dict(u_RGI_perp=torch.as_tensor(wg + 0.15 * w1), null_u_RGI_perp=nul, null_u_S_perp=nul, null_u_neg=nul, GA=True)
    return BQ["js"], r4_block(core, 8, dict(power=None), AXB, []), core


def selftest(work, r1_dir=None, r2_root=None):
    os.makedirs(work, exist_ok=True); res = []; t0 = time.time()

    def check(name, cond, info=""):
        res.append(dict(test=name, ok=bool(cond), info=str(info)[:500])); print(f"[{'PASS' if cond else 'FAIL'}] {name} {str(info)[:300]}", flush=True)
    NP, D = 64, 24; W = TH.boot_W(NP); rng = np.random.default_rng(0); u = R.unit(torch.as_tensor(rng.normal(size=D)))
    crn = (rng.normal(size=NP), rng.normal(size=(NP, 4)), rng.normal(size=(NP, 4, D)))
    cfg = dict(mode="standard", u_push=u, u_read=u, m2_par=torch.ones(W.shape[0], dtype=torch.float64), mG=torch.ones(W.shape[0], dtype=torch.float64))

    def gains(g_by_a, T=0.7, seed=1):
        cells = {}
        for i, (a, g) in enumerate(g_by_a): c = synth_cell(seed + i, a, T, g, NP, 4 if a == 0.5 else 2, D, u, crn=crn); cells[c["name"]] = c
        CT = TH.cell_table(cells, cfg, 1.0, W); return cells, CT
    # crossings = r2_sim.crossings (sim_gain's)
    try:
        import r2_sim
        gg = np.random.default_rng(3).lognormal(0, 0.4, size=(200, 4)); a4 = np.array([0.5, 1, 2, 4])
        c1, c2 = crossings(a4, gg), r2_sim.crossings(a4, gg)
        check("crossings equal r2_sim.crossings (sim_gain's)", all(np.array_equal(x, y, equal_nan=True) for x, y in zip(c1, c2)))
    except ImportError as e: check("crossings equal r2_sim.crossings (sim_gain's)", False, f"r2_sim missing: {e}")
    # R1: planted soft crossing at alpha* = sqrt(2) (gains 1.6, 1.25, 0.8, 0.5 at 0.5, 1, 2, 4 alpha_0)
    cells, CT = gains([(0.5, 1.6), (1, 1.25), (2, 0.8), (4, 0.5)])
    names = [TH.cell_name(a, 0.7) for a in (0.5, 1, 2, 4)]; reps = [CT[n]["all"]["dec"] for n in names]
    r1 = rule_R1([0.5, 1, 2, 4], reps, 0, {0.5: 0.01, 1.0: 0.03, 2.0: 0.1, 4.0: 0.4}, {w: [CT[n][w]["dec"] for n in names] for w in TH.WINDOWS})
    ast = r1["alpha_star"]
    check("R1: a planted soft crossing is located (FOUND-soft) with alpha* ~ sqrt(2) inside its interval; decision gains = planted",
          r1["verdict"] == "FOUND-soft" and ast["lo"] <= math.sqrt(2) <= ast["hi"] and abs(ast["pt"] - math.sqrt(2)) < 0.15 and all(CT[n]["all"]["kind"] == "g_par" for n in names)
          and all(abs(float(CT[n]["all"]["dec"][0]) - g) < 0.12 * g for n, g in zip(names, (1.6, 1.25, 0.8, 0.5))), (r1["verdict"], ast, [round(float(r[0]), 3) for r in reps]))
    check("R1: onset soft (paired g(0.5) - g(1) above 0); D* by log-log interpolation; not damage-limited", r1["onset"]["verdict"] == "soft"
          and abs(r1["D_star"]["pt"] - damage_at(ast["pt"], {0.5: 0.01, 1.0: 0.03, 2.0: 0.1, 4.0: 0.4})) < 1e-12 and 0.03 < r1["D_star"]["pt"] < 0.1 and "damage-limited" not in r1["labels"])
    cells_b, CTb = gains([(0.5, 0.7), (1, 0.6), (2, 0.5), (4, 0.4)], seed=11); repb = [CTb[TH.cell_name(a, 0.7)]["all"]["dec"] for a in (0.5, 1, 2, 4)]
    r1b = rule_R1([0.5, 1, 2, 4], repb, 0)
    check("R1: a series built below 1 is not located (SUB), alpha* censored at 0", r1b["verdict"] == "SUB" and r1b["alpha_star"]["pt"] == 0.0, (r1b["verdict"], r1b["alpha_star"]))
    r1c = rule_R1([0.5, 1, 2, 4], [CTb[TH.cell_name(a, 0.7)]["all"]["dec"] * 3.5 for a in (0.5, 1, 2, 4)], 0)
    check("R1: a series built above 1 is SUPER, alpha* censored at +inf", r1c["verdict"] == "SUPER" and r1c["alpha_star"]["pt"] == float("inf"), r1c["verdict"])
    _, CTh = gains([(0.5, 0.5), (1, 0.7), (2, 1.4), (4, 1.8)], seed=21); r1h = rule_R1([0.5, 1, 2, 4], [CTh[TH.cell_name(a, 0.7)]["all"]["dec"] for a in (0.5, 1, 2, 4)], 0)
    check("R1: a hard (threshold) onset is FOUND-hard with onset 'hard'", r1h["verdict"] == "FOUND-hard" and r1h["onset"]["verdict"] == "hard", (r1h["verdict"], r1h["onset"]))
    r1s = rule_R1([0.5, 1, 2], [np.r_[1.0, np.linspace(0.8, 1.2, 50)], np.r_[0.8, np.full(50, 0.9)], np.r_[0.8, np.full(50, 0.85)]], 0)
    r1f = rule_R1([0.5, 1, 2], [np.r_[1.1, np.full(50, 1.1)], np.r_[0.8, np.full(50, 0.9)], np.r_[0.8, np.full(50, 0.85)]], 0)
    check("R1: SUB(>= alpha_0) when the cells at >= alpha_0 lie below 1 and the 0.5 alpha_0 cell straddles 1 (above 1 there: FOUND-soft)",
          r1s["verdict"] == "SUB(>= alpha_0)" and r1f["verdict"] == "FOUND-soft", (r1s["verdict"], r1f["verdict"]))
    r1x = rule_R1([1.0], [np.r_[0.7, np.linspace(0.6, 0.8, 50)]], None)
    check("R1 at an extra block with Gamma_b only: SUB(>= alpha_0), onset undecided", r1x["verdict"] == "SUB(>= alpha_0)" and r1x["onset"]["verdict"] == "onset undecided")
    dmg = rule_R1([0.5, 1, 2, 4], reps, 0, {0.5: 0.05, 1.0: 0.12, 2.0: 0.3, 4.0: 0.9})
    check("R1: a crossing above 0.15 nats is labelled damage-limited", "damage-limited" in dmg["labels"], dmg["D_star"])
    cw = {"all": reps, "div": [r * 2.2 for r in reps], "first20": reps}
    check("R1: non-overlapping window intervals of alpha* give 'convention-sensitive'", "convention-sensitive" in rule_R1([0.5, 1, 2, 4], reps, 0, None, cw)["labels"])
    # R2: planted crossing at T_c = 0.6 (log G = 2 (0.6 - T)); below throughout; reserve range
    gT = lambda T: math.exp(2 * (0.6 - T)); cT = {}
    for i, T in enumerate((0.3, 0.7, 1.0)):
        c = synth_cell(40 + i, 1, T, gT(T), NP, 2, D, u, sd=0.4, crn=crn); cT[c["name"]] = c
    CTT = TH.cell_table(cT, cfg, 1.0, W); r2 = rule_R2([0.3, 0.7, 1.0], [CTT[TH.cell_name(1, T)]["all"]["dec"] for T in (0.3, 0.7, 1.0)])
    check("R2: a planted temperature crossing (T_c 0.6) is located with T_c inside its interval", r2["verdict"] == "T_c located" and r2["T_c"]["lo"] <= 0.6 <= r2["T_c"]["hi"]
          and r2["slope"]["hi"] < 0, (r2["verdict"], r2["T_c"], r2["slope"]))
    cB = {}
    for i, T in enumerate((0.3, 0.7, 1.0)):
        c = synth_cell(50 + i, 1, T, 0.5 + 0.05 * T, NP, 2, D, u, sd=0.4, crn=crn); cB[c["name"]] = c
    CTB = TH.cell_table(cB, cfg, 1.0, W); r2b = rule_R2([0.3, 0.7, 1.0], [CTB[TH.cell_name(1, T)]["all"]["dec"] for T in (0.3, 0.7, 1.0)])
    check("R2: gains built below 1 at every temperature give 'below 1 throughout' (not located)", r2b["verdict"] == "below 1 throughout" and r2b["margin"]["max_upper_band"] < 1, r2b["verdict"])
    r2c = rule_R2([0.3, 0.7, 1.0], [np.r_[1.05, np.random.default_rng(5).normal(1.05, 0.3, 200)] for _ in range(3)])
    check("R2: flat gains near 1 with wide noise are 'undecided'", r2c["verdict"] == "undecided", r2c["verdict"])
    g17 = [np.r_[g, np.random.default_rng(int(100 * g)).normal(g, 0.02, 400)] for g in (math.exp(2 * (0.17 - T)) for T in (0.15, 0.3, 0.7, 1.0))]
    r2e = rule_R2([0.15, 0.3, 0.7, 1.0], g17, extended=True); r2n = rule_R2([0.15, 0.3, 0.7, 1.0], g17, extended=False)
    check("R2: a crossing at T 0.17 (tested range 0.15-1.0) is located only within the extended window [0.1, 1.4]; curvature with 4 temperatures",
          r2e["verdict"] == "T_c located" and r2n["verdict"] == "undecided" and "curvature" in r2e and r2e["T_c"]["lo"] <= 0.17 <= r2e["T_c"]["hi"], (r2e["T_c"], r2n["verdict"]))
    # R3
    base = np.random.default_rng(9).normal(0, 0.03, 2001)
    mk = lambda v: np.r_[v, v + base[1:] + np.random.default_rng(int(100 * v)).normal(0, 0.02, 2000)]
    r3f = rule_R3([(8, 0.10, mk(1.0)), (14, 0.09, mk(1.4)), (20, 0.20, mk(0.6))], "loop gain"); r3r = rule_R3([(8, 0.10, mk(1.0)), (14, 0.09, mk(0.6)), (20, 0.20, mk(1.4))], "loop gain")
    r3n = rule_R3([(8, 0.10, mk(1.0)), (14, 0.09, mk(1.0)), (20, 0.20, mk(1.0))], "loop gain"); r3m = rule_R3([(8, 0.10, mk(1.6)), (14, 0.09, mk(1.0)), (20, 0.20, mk(0.6))], "loop gain")
    r3t = rule_R3([(14, 0.09, mk(1.4)), (20, 0.20, mk(0.6))], "loop gain"); r3o = rule_R3([(14, 0.09, mk(1.4))], "loop gain")
    check("R3: falls / rises / no relation; middle clause; two blocks marked; one block not decidable", r3f["verdict"] == "loop gain falls with delta"
          and r3r["verdict"] == "loop gain rises with delta" and r3n["verdict"] == "no monotone relation" and r3m["verdict"] == "no monotone relation"
          and r3t["verdict"].startswith("loop gain falls with delta (two blocks") and r3o["verdict"].startswith("not decidable"), (r3f["verdict"], r3r["verdict"], r3m["verdict"]))
    # R5, R6
    check("R5: persists but does not sustain itself", rule_R5(0.7, mk(0.6))["verdict"] == "direction persists but the loop does not sustain itself" and rule_R5(0.3, mk(0.6))["verdict"] is None)
    c7 = synth_cell(60, 1, 0.7, 0.5, NP, 2, D, u, crn=crn); t2 = synth_cell(61, 1, 0.7, 0.25, NP, 2, D, u, sd=0.3, crn=crn)
    P1, P2 = TH.Pairs(c7["windows"]["all"]["delta"], c7["windows"]["all"]["ok"], c7["prompt_of"], W), TH.Pairs(t2["windows"]["all"]["delta"], t2["windows"]["all"]["ok"], t2["prompt_of"], W)
    K = P2.proj(u) / P1.proj(u); Gc = TH.cell_table({c7["name"]: c7}, cfg, 1.0, W)[c7["name"]]["all"]["dec"]; r6 = rule_R6(K, P1.proj(u), Gc)
    check("R6: K = 0.5 (planted) is dissipative and transfers to G = 0.5; K on a random turn-1 is not defined", r6["defined"] and r6["verdict"] == "dissipative at T"
          and abs(r6["K"]["pt"] - 0.5) < 0.12 and r6["transfer"]["verdict"] == "transfers", (r6["K"], r6["transfer"]))
    r6d = rule_R6(K, P1.proj(u), Gc * 2.5); z = synth_cell(62, 1, 0.7, 0.0, NP, 2, D, u, sd=1.0); Pz = TH.Pairs(z["windows"]["all"]["delta"], z["windows"]["all"]["ok"], z["prompt_of"], W)
    r6z = rule_R6(P2.proj(u) / Pz.proj(u), Pz.proj(u), Gc); r6n = rule_R6(-K, P1.proj(u), Gc)
    check("R6: different criteria when G = 2.5 K; 'K not defined' without a turn-1 stance; 'K or G not positive'", r6d["transfer"]["verdict"] == "different criteria"
          and not r6z["defined"] and r6n["transfer"]["verdict"] == "undecided (K or G not positive)", (r6d["transfer"], r6z["verdict"]))
    # Amendment 3 (b) (review finding theta_verdict-1): undefined bounds. (1) The review's case: a 'mode not resolved' block whose RSI_a
    # evaluation n = 3 is noise with U <= 0 at the point, so m_A^2 = 0 and the point gains are undefined
    s2n = np.arange(1, 257) ** -1.2; s2n /= s2n.sum(); poN = np.repeat(np.arange(NP), 2); PaN = None
    for sd_ in range(100, 300):
        evN = np.random.default_rng(sd_).normal(size=(2 * NP, 256)) * np.sqrt(s2n)
        PaN = TH.Pairs(torch.as_tensor(evN), torch.ones(2 * NP, dtype=torch.bool), torch.as_tensor(poN), W)
        if float(PaN.U()[0]) <= 0: break
    mA2N = PaN.att2(); uN = R.unit(PaN.mean()[0]); cfgN = dict(mode="not_resolved", u_push=uN, u_read=uN, m2_par=mA2N, mG=torch.sqrt(mA2N)); cellsN = {}
    for i, (a, T) in enumerate(((0.5, 0.7), (1, 0.7), (2, 0.7), (1, 0.3), (1, 1.0))):
        c = synth_cell(3000 + i, a, T, 0.05, NP, 4 if a == 0.5 else 2, 256, uN, sd=0.05); cellsN[c["name"]] = c
    CTN = TH.cell_table(cellsN, cfgN, 1.0, W, windows=("all",))
    r1N = rule_R1([0.5, 1, 2], [CTN[TH.cell_name(a, 0.7)]["all"]["dec"] for a in (0.5, 1, 2)], 0)
    r2N = rule_R2([0.3, 0.7, 1.0], [CTN[TH.cell_name(1, T)]["all"]["dec"] for T in (0.3, 0.7, 1.0)])
    RSN = reserves({14: dict(CT=CTN)}); r5N = rule_R5(0.7, CTN[CELL_A0]["all"]["dec"])
    check("Amendment 3 (b): m_A^2 = 0 at the point -> R1 UNDECIDED (not SUB), R2 'undecided (gain undefined ...)' with slope and T_c kept, reserves not met, R5 undecided; shares written",
          float(mA2N[0]) == 0.0 and r1N["verdict"] == "UNDECIDED" and r1N["onset"]["verdict"] == "onset undecided" and all(r1N["bounds_undefined"])
          and r2N["verdict"].startswith("undecided (gain undefined at the point estimate") and "slope" in r2N and "T_c" in r2N and r2N["nonfinite_share"] > 0.05
          and not RSN["reserve1_a4_T0.7"]["enters"] and not RSN["reserve2_T0.15"]["enters"] and fin(RSN["reserve2_T0.15"]["nonfinite_share"])
          and r5N["self_sustainment"] == "undecided" and fin(r5N["nonfinite_share"]), (r1N["verdict"], r2N["verdict"], r1N["nonfinite_share"]))
    # (2) the 5 % share: a defined point with 4 % undefined replicates keeps its verdict, with 8 % the rule is undecided
    arr = lambda v, sh, sd: np.r_[v, np.where(np.random.default_rng(sd).random(2000) < sh, np.nan, np.random.default_rng(sd + 1).normal(v, 0.03, 2000))]
    s4 = rule_R1([0.5, 1, 2], [arr(0.7, 0.04, 1), arr(0.6, 0.04, 3), arr(0.5, 0.04, 5)], 0); s8 = rule_R1([0.5, 1, 2], [arr(0.7, 0.08, 1), arr(0.6, 0.08, 3), arr(0.5, 0.08, 5)], 0)
    t4 = rule_R2([0.3, 0.7, 1.0], [arr(0.7, 0.04, 7), arr(0.6, 0.04, 9), arr(0.5, 0.04, 11)]); t8 = rule_R2([0.3, 0.7, 1.0], [arr(0.7, 0.08, 7), arr(0.6, 0.08, 9), arr(0.5, 0.08, 11)])
    q4 = rule_R3([(8, 0.1, arr(1.0, 0.04, 13)), (20, 0.2, arr(0.6, 0.04, 15))], "loop gain"); q8 = rule_R3([(8, 0.1, arr(1.0, 0.08, 13)), (20, 0.2, arr(0.6, 0.08, 15))], "loop gain")
    tn = rule_R2([0.3, 0.7, 1.0], [np.r_[0.5, np.full(2000, np.nan)]] * 3)
    check("Amendment 3 (b): 4 % undefined replicates keep SUB / 'below 1 throughout' / 'falls', 8 % give undecided in R1, R2 and R3; no finite replicate fit: no error, slope and T_c kept",
          s4["verdict"] == "SUB" and s8["verdict"] == "UNDECIDED" and t4["verdict"] == "below 1 throughout" and t8["verdict"].startswith("undecided (gain undefined in")
          and q4["verdict"].startswith("loop gain falls with delta") and "undecided: a bound is undefined" in q8["verdict"] and q8["bounds_undefined"]
          and tn["verdict"].startswith("undecided (gain undefined in 100.0 %") and "slope" in tn and "T_c" in tn and abs(s8["nonfinite_share"][0] - 0.08) < 0.02,
          (s4["verdict"], s8["verdict"], t4["verdict"], t8["verdict"], q8["verdict"], tn["verdict"]))
    # reserves (G21-G23)
    fake = lambda v: dict(all=dict(dec=torch.as_tensor(np.r_[v, np.random.default_rng(int(1000 * v)).normal(v, 0.05, 2000)])))
    cores = {14: dict(CT={TH.cell_name(2, 0.7): fake(1.3), TH.cell_name(1, 1.0): fake(0.9), TH.cell_name(1, 0.3): fake(0.6), TH.cell_name(4, 0.7): fake(1.0)}),
             8: dict(CT={CELL_A0: fake(0.95)}), 20: dict(CT={CELL_A0: fake(0.9)})}
    RS = reserves(cores)
    check("reserves: 4 alpha_0 enters (G(2) lower > 1), T 1.3 does not, T 0.15 does; reserve 3 only at the higher extra block with lower > 0.8",
          RS["reserve1_a4_T0.7"]["enters"] and not RS["reserve2_T1.3"]["enters"] and RS["reserve2_T0.15"]["enters"] and RS["reserve3_a2_T0.7"]["block"] == 8
          and RS["reserve3_a2_T0.7"]["enters"] == {"8": True, "20": False}, RS)
    cores[8]["CT"][CELL_A0] = fake(0.7); cores[20]["CT"][CELL_A0] = fake(0.75); RS2 = reserves(cores)
    check("reserves: no reserve-3 cell enters when the higher block's lower bound <= 0.8", RS2["reserve3_a2_T0.7"]["enters"] == {"8": False, "20": False} and RS2["reserve3_a2_T0.7"]["block"] == 20)
    # B-Q7 lambda_lin on planted covariance: S = c (x - mean) along u
    rr = np.random.default_rng(12); Xs = rr.normal(size=(2 * NP, D)); Ss = 0.7 * (Xs @ u.numpy())[:, None] * u.numpy()[None] + rr.normal(size=(2 * NP, D)) * 0.1
    ll = lambda_lin(torch.as_tensor(Xs), torch.as_tensor(Ss), torch.as_tensor(list(range(NP)) * 2), W, torch.full((W.shape[0],), 2.0, dtype=torch.float64), dict(u=u, v=None))
    exp_ = 0.7 * float(np.var(Xs @ u.numpy(), ddof=1)) / 2.0
    check("B-Q7: lambda_lin(u) = Cov(u.x, u.S) / cbar (planted)", abs(ll["u"]["pt"] - exp_) < 0.02 * abs(exp_) + 1e-3 and ll["u"]["lo"] < ll["u"]["pt"] < ll["u"]["hi"] and ll["v"] is None, (ll["u"], exp_))
    # bootstrap of the U statistic: replicates centred on U (distinct pairs), and m^2 on its point value; the plug-in U would not be
    rr = np.random.default_rng(13); mu = np.zeros(D); mu[0] = 1.0; po = np.repeat(np.arange(NP), 2)
    Xd = mu[None] * (1 + 0.3 * rr.normal(size=NP)[po])[:, None] + rr.normal(size=(2 * NP, D)) * 1.6
    Pq = TH.Pairs(torch.as_tensor(Xd), torch.ones(2 * NP, dtype=torch.bool), torch.as_tensor(po), W); U0 = float(Pq.U()[0]); Ub = _np(Pq.U())[1:]
    naive = _np((Pq.s2 - Pq.q) / (Pq.n * (Pq.n - 1)))[1:]; m2b = _np(Pq.att2())[1:]; nf2 = float(Pq.nf()[0]) ** 2
    exact = (float(Xd.sum(0) @ Xd.sum(0)) - float((Xd ** 2).sum())) / (2 * NP * (2 * NP - 1))
    check("Pairs: U at the point = the plan's U; bootstrap replicates centred on U (the plug-in is centred on |d|^2), m^2 on its point value",
          abs(U0 - exact) < 1e-9 and abs(Ub.mean() - U0) < 0.25 * Ub.std() and abs(naive.mean() - U0 - nf2) < 0.35 * nf2 and abs(m2b.mean() - float(Pq.att2()[0])) < 0.5 * m2b.std(),
          (U0, Ub.mean(), naive.mean(), nf2, float(Pq.att2()[0]), m2b.mean()))
    # B-Q2 alignment labels on planted axes
    e = torch.eye(D, dtype=torch.float64); nul = torch.as_tensor(np.random.default_rng(14).normal(size=(2000, D)))
    Ua = (e[0] + 0.05 * torch.as_tensor(np.random.default_rng(15).normal(size=(W.shape[0], D)))); Ua[0] = e[0]
    axA = dict(u_T_perp=(e[0], nul), u_S_perp=(e[0] + 0.5 * e[1], nul), sports=(e[2], nul), tone=(e[3], nul), sea=(e[4], nul), u_neg=(e[5], nul), u_RGI_perp=(e[1] + 0.2 * e[0], nul))
    A1 = alignment(Ua, 0.9, axA, dict(rsi=dict(wU_perp=e[0] * 3.0, sigma=1.0)), ga=True)
    axB = dict(axA, u_T_perp=(e[6], nul), u_RGI_perp=(e[0], nul))
    A2 = alignment(Ua, 0.9, axB, None, ga=False)
    check("B-Q2: RSI-aligned and RSI-specific on a planted RSI mode; GI-aligned with 'RGI axis not resolved' when GA fails; z_beh", A1["RSI_aligned"]
          and A1["RSI_specific"] and not A1["GI_aligned"] and abs(A1["axes"]["u_T_perp"]["corrected"]["pt"] - 1 / 0.9) < 1e-6 and A1["axes"]["u_T_perp"]["p"] < 0.01
          and abs(A1["z_beh"]["rsi"] - 3.0) < 1e-6 and A2["GI_aligned"] and not A2["RSI_aligned"] and "RGI axis not resolved" in A2["labels"], (A1["RSI_aligned"], A1["RSI_specific"], A2["labels"]))
    # Amendment 3 (c) (review finding theta_verdict-2): a 'degenerate' R4 (i) gets no '(RSI-specific)' and does not enter R3's alignment form
    jsD, r4D, coreD = synth_two_mode_core(11, 0.8, 0.8); jsF, r4F, coreF = synth_two_mode_core(11, 1.3, 0.5)
    blk = {"8": dict(R4=r4D), "20": dict(R4=r4F)}; alD, agD, degD = alignment_entries([8, 20], blk, {8: coreD, 20: coreF}, {8: 0.1, 20: 0.2})
    check("Amendment 3 (c): R4 (i) 'degenerate' without '(RSI-specific)' although u*_b is RSI-specific; the block is listed 'not entering (degenerate)'; a 'first' RSI_a keeps the suffix and enters",
          jsD["mode_set"]["members"] == ["RSI_a"] and r4D["i"]["verdict"] == "degenerate" and r4D["verdict"] == "degenerate" and r4D["default"]["RSI_specific"]
          and coreD["align_reps"] == {} and degD == {"8": "not entering (degenerate)"} and [x[0] for x in alD] == [20] and [x[0] for x in agD] == [20]
          and r4F["verdict"] == "RSI-aligned (RSI-specific)" and coreF["align_reps"].get("u_T_perp") is not None, (r4D["verdict"], r4F["verdict"], degD))
    # B-Q0 on planted loops: Gamma recovers lambda; guard; switch on a strong bulk self-response; mode not resolved at low SNR
    cb_ref = dict(sum=[40.0] * NP, cnt=[40] * NP)
    for lam in (0.7, 1.3):
        bank, rX, ax, w_ = synth_bank(31, lam, 0.25)
        BQ = TH.bq0(bank, 14, 1.0, 1.0, cb_ref, rX, ax, W); js = BQ["js"]
        check(f"B-Q0 (planted lambda {lam}): Gamma_b within 10 % of lambda, guard passes, u*_b on the mode, lambda_cf close to g_par^AB (no switch)",
              js["m_star2"]["guard_pass"] and abs(js["Gamma"]["pt"] - lam) < 0.1 * lam and abs(js["cross_fit"]["lambda_cf"]["pt"] - lam) < 0.1 * lam and abs(float(BQ["tensors"]["u_star"] @ torch.as_tensor(w_))) > 0.9 and not js["cross_fit"].get("switch")
              and js["config"]["mode"] == "standard" and "RSI_b" in js["mode_set"]["members"], (js["Gamma"], js["m_star2"]["p5"], js["cross_fit"].get("reason")))
    bank, rX, ax, w_ = synth_bank(31, 1.3, 0.25); old = TH.CF_SWITCH
    try:
        TH.CF_SWITCH = 0.0; BQs = TH.bq0(bank, 14, 1.0, 1.0, cb_ref, rX, ax, W); TH.CF_SWITCH = 1e9; BQo = TH.bq0(bank, 14, 1.0, 1.0, cb_ref, rX, ax, W)
    finally: TH.CF_SWITCH = old
    cs = BQs["cfg"]; uA4 = R.unit(bank["RSI_a"]["d"][4]); uB4 = R.unit(bank["RSI_b"]["d"][4]) * (1 if float(bank["RSI_a"]["d"][4] @ bank["RSI_b"]["d"][4]) >= 0 else -1)
    check("B-Q0 switch (G18, threshold set in the test): push u*_A = unit(d_4^A), read u*_B = s unit(d_4^B), g_par / cos(u*_A, u*_B), G / m_A; no switch otherwise",
          cs["mode"] == "switch" and float((cs["u_push"] - uA4).norm()) < 1e-9 and float((cs["u_read"] - uB4).norm()) < 1e-9 and abs(float(cs["m2_par"][5]) - float(uA4 @ uB4)) < 1e-12
          and float((cs["mG"] - torch.sqrt(BQs["reps"]["mA2"])).abs().max()) == 0 and BQo["cfg"]["mode"] == "standard" and BQs["js"]["cross_fit"]["switch"], cs["mode"])
    cfg_s = _cfg_used(dict(mode="switch", u_push=cs["u_push"], u_read=cs["u_read"]), BQs["reps"])
    check("analysis configuration of a 'switch' session (OD-G8: the in-session mode governs)", abs(float(cfg_s["m2_par"][0]) - float(uA4 @ uB4)) < 1e-6 and torch.equal(cfg_s["mG"], torch.sqrt(BQs["reps"]["mA2"])))
    bank, rX, ax, w_ = synth_bank(35, 0.5, 2.5); BQn = TH.bq0(bank, 14, 1.0, 1.0, cb_ref, rX, ax, W)
    check("B-Q0: at low SNR the m*^2 guard fails -> 'mode not resolved', push / read along unit(d_4^RSI_a), no switch", not BQn["js"]["m_star2"]["guard_pass"]
          and BQn["js"]["config"]["mode"] == "not_resolved" and "mode not resolved" in BQn["js"]["labels"] and not BQn["js"]["cross_fit"].get("computed"), BQn["js"]["m_star2"])
    bank, rX, ax, w_ = synth_bank(37, 1.1, 0.25, loops=(("RSI_a", 2), ("SEA", 1), ("RND_1", 1))); BQx = TH.bq0(bank, 8, 1.0, 1.0, cb_ref, rX, ax, W)
    check("B-Q0 at an extra block without r2RSIb: 'no independent loop' (G49), no cross-fit", "no independent loop" in BQx["js"]["labels"] and not BQx["js"]["cross_fit"]["computed"])
    # cells of the bank's mode: a cell pushed along u*_b reproduces the planted gain through m*^2
    bank, rX, ax, w_ = synth_bank(31, 1.3, 0.25); BQ = TH.bq0(bank, 14, 1.0, 1.0, cb_ref, rX, ax, W); us = BQ["cfg"]["u_read"].numpy(); m2 = float(BQ["reps"]["m2"][0])
    Jd = 1.3 * (w_ @ us) * w_; rng2 = np.random.default_rng(77); po = np.repeat(np.arange(NP), 2); dl = Jd[None] * (1 + 0.5 * rng2.normal(size=len(po)))[:, None] + 0.25 * rng2.normal(size=(len(po), len(us)))
    cc = dict(synth_cell(1, 1, 0.7, 1.0, NP, 2, len(us), us), windows={w: dict(delta=torch.as_tensor(dl, dtype=torch.float32), ok=torch.ones(len(po), dtype=torch.bool)) for w in TH.WINDOWS})
    gc = TH.cell_table({"x": cc}, BQ["cfg"], 1.0, W)["x"]["all"]
    check("cells: g_par = u*_b . d / (a alpha cbar m*^2) recovers the planted gain of the mode", gc["kind"] == "g_par" and lohi(gc["dec"])[0] < 1.3 < lohi(gc["dec"])[1] * 1.05, (S(gc["dec"]), m2, float((w_ @ us) ** 2)))
    # end to end on MOCK theta outputs: (a) the session outputs <r2_dir>/theta (MOCK: 8 prompts, the guard fails: 'mode not resolved');
    # (b) the code-path variant <r2_dir>/theta_std (guard threshold 0 and power-step threshold +inf in the session): standard / switch
    # cells, R4 / B-Q2, the power step's R4 (ii) and the OD-G8 report of the guard disagreement
    if r1_dir and r2_root:
        out = os.path.join(work, "theta_verdict_e2e.json")
        if os.path.exists(out): os.remove(out)
        rc = main(["r2_theta_verdict.py", r1_dir, r2_root, out]); js = json.load(open(out)); B = js.get("blocks", {})
        check("end to end: theta_verdict.json for every block of B3 with R1, R2, R5 and a sentence", rc == 0 and js["status"] == "ok" and all(k in B for k in ("14", "8", "20"))
              and all(B[k].get("R1") and B[k].get("R2") and B[k].get("R5") and "sentence" in B[k] for k in B), {k: B[k].get("status") for k in B})
        check("end to end: OD-G8 recomputation agrees with the in-session decisions (B-Q0 and power cell)", all(B[k].get("od_g8", {}).get("agree") for k in B), {k: B[k].get("od_g8") for k in B})
        check("end to end: R6 at block 14 (four carry cells), R3, reserves, B-Q7, power re-run", "R6" in B["14"] and len(B["14"]["R6"]["cells"]) == 4 and "R3" in js
              and "reserves" in js and all(B[k]["B_Q7"].get("lambda_lin") for k in B) and all(B[k]["power_rerun"].get("status") == "ok" for k in B),
              {k: B[k]["power_rerun"].get("status") for k in B})
        check("end to end: labels of the MOCK gates (G-desc at block 14, G-blk false at b_hi), 'mode not resolved' blocks without R4", "descriptive (round-1 gate failed)" in B["14"]["labels"]
              and B["20"]["enters_R3"]["enters"] is False and all(B[k]["R4"].get("verdict") is None for k in B if B[k].get("mode") == "not_resolved"), (B["14"]["labels"], B["20"]["enters_R3"]))
        std = os.path.join(r2_root, "theta_std")
        if os.path.isdir(std):
            v2 = os.path.join(work, "e2e_std"); shutil.rmtree(v2, ignore_errors=True); os.makedirs(os.path.join(v2, "r2_q2"))
            os.symlink(std, os.path.join(v2, "r2_q2", "theta"))
            if os.path.isdir(os.path.join(r2_root, "axes")): os.makedirs(os.path.join(v2, "r2_q1")); os.symlink(os.path.join(r2_root, "axes"), os.path.join(v2, "r2_q1", "axes"))
            out2 = os.path.join(work, "theta_verdict_e2e_std.json")
            if os.path.exists(out2): os.remove(out2)
            rc = main(["r2_theta_verdict.py", r1_dir, v2, out2]); j2 = json.load(open(out2)); B2 = j2.get("blocks", {})
            g8 = {k: [d["item"] for d in B2[k]["od_g8"]["disagreements"]] for k in B2}
            check("end to end (variant): the in-session mode governs (standard / switch cells analysed), OD-G8 reports the guard disagreement",
                  rc == 0 and all(B2[k]["mode"] in ("standard", "switch") for k in B2) and all("guard" in g8[k] for k in B2 if not B2[k]["B_Q0"]["m_star2"]["guard_pass"]), g8)
            check("end to end (variant): R4 with B-Q2 alignment at every block (verdict where G-blk holds, descriptive elsewhere); R4 (ii) from the power step",
                  all(B2[k]["R4"].get("default") and ("verdict" in B2[k]["R4"] or "verdict_descriptive" in B2[k]["R4"]) for k in B2)
                  and all(B2[k]["R4"]["ii"]["applies"] for k in B2) and B2["8"]["R4"].get("verdict") is not None
                  and (B2["14"]["R4"]["default"]["GI_aligned"] is not None) == os.path.isdir(os.path.join(r2_root, "axes")),
                  {k: (B2[k]["R4"].get("verdict"), B2[k]["R4"]["ii"].get("verdict")) for k in B2})
            check("end to end (variant): block 8 enters R3 (G-blk, mode resolved, independent loop); sentences name the mode", B2["8"]["enters_R3"]["enters"]
                  and "critical mode is" in B2["8"]["sentence"] and j2["R3"]["entering"] == [8], (j2["R3"]["entering"], B2["8"]["sentence"][:200]))
            # (c) b_hi not final (summary missing, G4) and no O2 axes: b_hi 'not run', GI alignment 'not available'
            nf = os.path.join(work, "r1_nonfinal"); shutil.rmtree(nf, ignore_errors=True); shutil.copytree(r1_dir, nf, copy_function=os.symlink)
            bh = R.r1_blocks(r1_dir)["b_hi"]; os.remove(R.find_r1(nf, f"summary_b{bh}.json", bh))
            v3 = os.path.join(work, "e2e_noax"); shutil.rmtree(v3, ignore_errors=True); os.makedirs(os.path.join(v3, "r2_q2")); os.symlink(std, os.path.join(v3, "r2_q2", "theta"))
            out3 = os.path.join(work, "theta_verdict_e2e_nonfinal.json")
            if os.path.exists(out3): os.remove(out3)
            rc = main(["r2_theta_verdict.py", nf, v3, out3]); j3 = json.load(open(out3)); B3_ = j3["blocks"]
            check("end to end (b_hi not final, no O2 axes): b_hi 'not run (round-1 output missing)', the others analysed, GI alignment 'not available'",
                  rc == 0 and B3_[str(bh)]["status"] == "not run (round-1 output missing)" and B3_["14"]["status"] == "ok" and B3_["8"]["status"] == "ok"
                  and "GI alignment not available (O2 axes missing)" in B3_["8"]["R4"]["labels"] and j3["R3"]["entering"] == [8], {k: v["status"] for k, v in B3_.items()})
    ok = all(r["ok"] for r in res); R.write_json(os.path.join(work, "selftest_theta_verdict.json"), dict(ok=ok, tests=res, seconds=time.time() - t0))
    print(f"r2_theta_verdict selftest: {sum(r['ok'] for r in res)}/{len(res)} passed ({time.time() - t0:.0f}s)", flush=True); return ok


if __name__ == "__main__":
    sys.exit(main(sys.argv))
