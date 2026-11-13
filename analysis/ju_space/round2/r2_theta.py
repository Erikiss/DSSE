"""Round 2, Part B Qwen (plan sections 4.0 and 4.1; gates G17-G20, G24, G25, G49 of section 6; CLI of 9.2): the theta stages of
one block. One process = one block (env BLOCK = 14, b_lo or b_hi; r2_rk.open_block, which refuses a non-final round-1 block: every
stage then writes 'not run (round-1 output missing)', G4 / G6).

Stages (each resumable: skipped when its output exists; a stage runs the stages it needs first, so the order of calls does not matter)
  rsib    extra blocks only (B-Q0, OD-B2, critique v2-3): the second, independent RSI loop rk.run_loop('r2RSIb', unit(u_T_perp(b)),
          nseed 2, 4 rounds) at alpha_b and cap_b (keys keyseed('r2RSIb' + SUF_b, n, p, s, b) via rk.evaluate; rk's own state keys
          ev_ / dir_ / dmg_r2RSIb<SUF>_n under the state guard) -> theta/b<b>_r2RSIb.json (run_loop rows) and theta/b<b>_r2RSIb.pt
          (the loop's state entries, for r2_ana, which does not copy the round-2 state.pt). q0 runs it first ('before anything else').
  q0      B-Q0: O3 bank (+ r2RSIb at the extra blocks), kappa_b from one cbar read of the 64 stored references (cbar_b = kappa_b med_b),
          NF / U / SNR / g_par / G / rotation / decision gain per loop and evaluation, the mode set M_b, u*_b, m*^2 and its bootstrap
          guard (G17, OD-B1), Gamma_b, the cross-fit lambda_cf and the switch (G18, OD-B12), the push / read directions of the cells,
          and the descriptives (first divergence, identical pairs, even part E_b, one-step gains at n = 0, image direction, gate p
          (G19), persistence) -> theta/b<b>_q0.json + .pt
  checks  block 14 (B-Q1, G20): zero push gives token-identical texts (4 of 4), -u_push swaps the texts of a pair (>= 3 of 4), mean
          continuation entropy at T 0.3 < at T 0.7; keys keyseed('r2CHK', check, p) -> theta/checks.json
  cells   B-Q1 / B-Q3 / B-Q4 / B-Q5: every theta cell of the block (reserves 1-3 run unconditionally, 2.8); keys keyseed('theta', b, p,
          s) shared by every cell of the block; rk.TEMP = T; push +- a alpha_b unit(u_push); cbar hook; the three read windows and the
          NLL covariate from one full unsteered pass with a block-b hook; damage curve of u_push at {0.5, 1, 2, 4} alpha_b (rk.damage);
          sampling KL at T (damage_T); continuation entropy; repeated-4-gram share -> theta/b<b>_cells.pt + .json
  turn2   block 14 (B-Q8, OD-B4, OD-B13): turn 2 after the alpha_0 cells at T 0.7, 0.3, 1.0 and (reserve 5) after the 2 alpha_0 cell at
          T 0.7: input ids PIDS[p] + turn-1 ids + the ids of r2_stimuli.TURN2_SUFFIX (asserted once against apply_chat_template), no
          push, keys keyseed('theta2', b, a, T, p, s) shared within a pair, temperature = the cell's T, read as the reply to the plain
          prompt -> theta/b14_turn2.pt (+ .json with point estimates)
  power   B-Q6 (G24, OD-B7): at the cell with the smallest |log decision gain| among the block's cells with rotation < 0.8 (point
          estimates, the same function as r2_ana, OD-G8), push unit(that cell's read-back d) at the same (a, T), 128 pairs, keys
          keyseed('r2PWR', b, p, s) -> theta/b<b>_power.json (+ theta/b<b>_power.pt when it ran; 'not needed' otherwise)
  fdt     B-Q7: the 64 stored references plus 64 new unsteered texts (keys keyseed('r2fdt', b, p), T 0.7); per text one forward and
          backward pass with rk's zero-leaf hook (STEER['V'], leaf added as min(|h_t|, cap) V from the first user token on) on
          y = sum over continuation positions of log softmax(logits_t / 0.7)[y_t] (+ the stop if the text ended early) -> S_j; x_j = the
          text's block-b reading from the same pass -> theta/b<b>_fdt.pt (lambda_lin in r2_theta_verdict.py)
  all     every stage of the block in the order above; selftest: the MOCK code test of this block.

Statistics shared with r2_theta_verdict.py (OD-G8: in-session decisions use the analysis functions): boot_W (the joint prompt
bootstrap), Pairs, ivpool, eval_gains, bq0 (B-Q0 incl. guard, switch and the cells' configuration), cell_cbar, cell_gains, cell_table,
cell_points, choose_power_cell, power_gains, halving_rel. Readings of ambiguous text: NOTES_w4.md.

Usage: python r2_theta.py <root> <r1_dir> <r2_dir> [rsib | q0 | checks | cells | turn2 | power | fdt | all | selftest]
       env BLOCK (14 / b_lo / b_hi), DEVICE, NT, MOCK"""
import contextlib, glob, json, math, os, shutil, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
import r2_rk as R
import r2_stimuli as STIM

PLAN = "results/ju_space/plans/abschnitt11_12_runde2_plan.md"
PLAN_SHA256 = "5e476340dfdb1d827582c869f2dfc08e32ed8e9a9dbebf7642acceb5bb5edc2d"
SEED_BOOT, N_BOOT = 41, 2000          # the joint prompt bootstrap of B-Q0, the cells and R1-R6 (2.6; OD-G6); row 0 = point estimate
LEVEL = 0.90
ROT_DEC, ROT_PWR = 0.9, 0.8           # decision gain = g_par if rotation >= 0.9 (B-Q0, B-Q1); power-step cells: rotation < 0.8 (B-Q6)
MODE_COS, M2_GUARD = 0.5, 0.5         # mode set: debiased |cos(d_4^L, d_4^RSI_a)| >= 0.5; m*^2 guard on the bootstrap 5th percentile
CF_COS, CF_SWITCH = 0.3, 0.15         # cross-fit only if cos(u*_A, u*_B) > 0.3; switch if |lambda_cf - g_par^AB| > 0.15
P_GATE = 0.25                         # G19 (label only)
GRID_A = (0.5, 1.0, 2.0, 4.0)         # damage curve of the push direction (B-Q1, OD-B9)
T_FDT = 0.7
RSIB = "r2RSIb"                        # B-Q0's second RSI loop at the extra blocks (Appendix D: ('r2RSIb' + SUF_b, n, p, s, b))
WINDOWS = R.WINDOWS                    # ('all', 'div', 'first20'); 'all' is primary


def cell_name(a, T):
    return f"a{a:g}_T{T:g}"


def _spec(a, T, nseed, reserve=None):
    return dict(name=cell_name(a, T), a=float(a), T=float(T), nseed=int(nseed), reserve=reserve)


# Cells (B-Q3, B-Q4, B-Q5; 2.8 pair counts: 0.5 alpha_0 at block 14 = 64 prompts x 4 seeds, every other cell 64 x 2). Listed in run
# order (main cells first, reserves last; every cell runs, 2.8). a in units of alpha_0 = alpha_b, T = sampling temperature.
CELLS_14 = (_spec(1, 0.7, 2), _spec(0.5, 0.7, 4), _spec(2, 0.7, 2), _spec(1, 0.3, 2), _spec(1, 1.0, 2),
            _spec(4, 0.7, 2, "reserve 1"), _spec(1, 1.3, 2, "reserve 2"), _spec(1, 0.15, 2, "reserve 2"))
CELLS_EXTRA = (_spec(1, 0.7, 2), _spec(1, 0.3, 2), _spec(1, 1.0, 2), _spec(2, 0.7, 2, "reserve 3"))
TURN2_CELLS = (cell_name(1, 0.7), cell_name(1, 0.3), cell_name(1, 1.0), cell_name(2, 0.7))      # B-Q8; the last = reserve 5


def cells_of(b):
    return CELLS_14 if int(b) == 14 else CELLS_EXTRA


def fin(x):
    try: return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError): return False


def _np(x):
    return x.detach().cpu().numpy().astype(float) if torch.is_tensor(x) else np.asarray(x, dtype=float)


def summ(x, level=LEVEL):
    """Point estimate (row 0) and the 90 % percentile interval over the bootstrap rows 1.. of a replicate array (NaN rows ignored)."""
    x = _np(x).reshape(-1); lo, hi = R.interval(x[1:], level) if len(x) > 1 else (float("nan"), float("nan"))
    nf = int(np.isfinite(x[1:]).sum())
    return dict(pt=float(x[0]), lo=lo, hi=hi, sd=float(np.nanstd(x[1:], ddof=1)) if nf > 1 else float("nan"), n_finite=nf)


def fd_stats(first_div):
    """First-divergence distribution of one evaluation or cell (rk.evaluate's convention: -1 = identical pair)."""
    fd = [int(k) for k in first_div]; dv = np.array([k for k in fd if k >= 0], dtype=float); n = len(fd)
    q = (lambda v: float(np.percentile(dv, v))) if len(dv) else (lambda v: float("nan"))
    return dict(n_pairs=n, identical=int(sum(k == -1 for k in fd)), identical_share=float(sum(k == -1 for k in fd) / n) if n else float("nan"),
                min=float(dv.min()) if len(dv) else float("nan"), q25=q(25), median=q(50), q75=q(75), max=float(dv.max()) if len(dv) else float("nan"))


def rep4(c):
    """Repeated-4-gram share of one text: (number of 4-grams - number of distinct 4-grams) / number of 4-grams (NaN below 4 tokens)."""
    n = len(c) - 3
    if n <= 0: return float("nan")
    return (n - len({tuple(c[i:i + 4]) for i in range(n)})) / n


# ======================================================================================================================================
# Statistics shared with r2_theta_verdict.py (OD-G8)
# ======================================================================================================================================
def boot_W(NP, B=N_BOOT, seed=SEED_BOOT):
    """Prompt-weight rows [1 + B, NP] (float64): row 0 = the point estimate (every prompt once), rows 1..B = the joint prompt bootstrap
    (r2_rk.prompt_bootstrap: prompts as units, both seeds of a prompt together, stratified by prompt type). The same rows serve every
    evaluation, cell and block, so differences between cells and blocks are paired (2.6; rules R1-R6: 'prompts and keys shared')."""
    return np.vstack([np.ones((1, NP)), R.prompt_bootstrap(NP, B=B, seed=seed)])


class Pairs:
    """Per-pair read-backs delta_j [N, D] (float64; rows of pairs that are not ok are ignored), ok [N], prompt_of [N], aggregated under
    the prompt-weight rows W [R, NP]: pair j gets weight w_j = W[r, prompt_of[j]] (0 if not ok; a count in a bootstrap row). Per row:
    n = sum w, s2 = |sum w delta|^2, q = sum w |delta|^2, q2 = sum w^2 |delta|^2, w2 = sum w^2.
    U = (s2 - q2) / (n^2 - w2): the U statistic of B-Q0 / B-Q1 (mean of delta_i . delta_j over distinct pairs i != j). In a bootstrap row
    the products between copies of the same pair are excluded as well (a plug-in U of the resample would count them and is centred on
    |d|^2, the floor included, instead of U; NOTES_w4.md, 'bootstrap of the U statistic'). NF = sqrt(q) / n, proj(u) = unit(u) . d,
    mean() = d [R, D]; att2() = m^2. Row 0 of boot_W (w = 1) gives exactly the plan's point formulas."""
    def __init__(self, delta, ok, prompt_of, W):
        X = torch.as_tensor(delta).double(); okb = torch.as_tensor(ok, dtype=torch.bool) & torch.isfinite(X).all(1)
        self.X = torch.where(okb[:, None], X, torch.zeros_like(X)); self.ok = okb; self.po = torch.as_tensor(prompt_of).long()
        self.w = torch.as_tensor(W, dtype=torch.float64)[:, self.po] * okb.double()[None]
        K = self.X @ self.X.T; dg = torch.diagonal(K); w = self.w
        self.n = w.sum(1); self.s2 = ((w @ K) * w).sum(1); self.q = w @ dg; self.q2 = (w * w) @ dg; self.w2 = (w * w).sum(1)

    def U(self): return (self.s2 - self.q2) / (self.n ** 2 - self.w2)
    def nf(self): return torch.sqrt(self.q) / self.n
    def d2(self): return self.s2 / self.n ** 2
    def proj(self, u): return (self.w @ (self.X @ R.unit(torch.as_tensor(u).double()))) / self.n
    def mean(self): return (self.w @ self.X) / self.n[:, None]

    def att2(self):
        """m^2 = max(U, 0) / |d|^2 (B-Q0's attenuation of a pooled or single evaluation), with |d|^2 written as U (1 - 1/n) + q / n^2 (an
        identity at the point estimate; in a bootstrap row the distinct-pair form that matches U)."""
        U = self.U(); return U.clamp(min=0) / (U * (1 - 1 / self.n) + self.q / self.n ** 2)


def halving_rel(delta, ok, prompt_of, NP):
    """rk.stats's split-half reliability of the mean read-back: r = mean over the 20 K2 halvings of cos(mean of half A, mean of half B),
    rel = max(2r / (1 + r), 0.05) (NaN for non-finite r). Returns (rel, r)."""
    delta = torch.as_tensor(delta).double(); ok = torch.as_tensor(ok, dtype=torch.bool); po = torch.as_tensor(prompt_of).long(); rs = []
    for m in R.halvings(NP):
        mt = torch.as_tensor(m); mA = mt[po] & ok; mB = (~mt)[po] & ok; rs.append(R.cosf(delta[mA].mean(0), delta[mB].mean(0)))
    r = float(np.mean(rs))
    if not math.isfinite(r): return float("nan"), r
    return (max(2 * r / (1 + r), 0.05) if r > -1 else 0.05), r


def ivpool(arrs):
    """Inverse-variance pool of replicate arrays (row 0 = point; variances over the bootstrap rows): weights 1 / var for components with
    a finite point, finite positive variance. Returns (pooled [R] float64 numpy, weights list (0 = left out))."""
    A = [_np(a) for a in arrs]; wts = []
    for a in A:
        v = float(np.nanvar(a[1:], ddof=1)) if np.isfinite(a[1:]).sum() > 1 else float("nan")
        wts.append(1.0 / v if (math.isfinite(a[0]) and math.isfinite(v) and v > 0) else 0.0)
    if not any(w > 0 for w in wts): return np.full(len(A[0]) if A else 1, np.nan), wts
    num = sum(w * a for w, a in zip(wts, A) if w > 0); den = sum(w for w in wts if w > 0)
    return num / den, wts


def eval_gains(Pn, push, rel, alpha, cbar):
    """One B-Q0 evaluation (push d_n with rel_n = rel(d_n), read-back pairs of the evaluation): NF, U, SNR = sqrt(max(U, 0)) / NF,
    g_par = unit(d_n) . d_(n+1) / (alpha_b cbar_b rel_n), G = sqrt(max(U, 0)) / (alpha_b cbar_b sqrt(rel_n)), rotation = g_par / G,
    decision gain = g_par if the point rotation >= 0.9, else G (the estimator chosen on the point estimate is used in every replicate).
    cbar: [R] (cbar_b per bootstrap row). Returns dict of [R] float64 tensors plus 'kind'."""
    U = Pn.U(); Gn = torch.sqrt(U.clamp(min=0)); nf = Pn.nf(); proj = Pn.proj(push); cb = torch.as_tensor(cbar).double()
    if fin(rel) and rel > 0:
        gpar = proj / (alpha * cb * rel); G = Gn / (alpha * cb * math.sqrt(rel))
    else:
        gpar = torch.full_like(U, float("nan")); G = torch.full_like(U, float("nan"))
    rot = torch.where(G > 0, gpar / G, torch.full_like(G, float("nan")))
    kind = "g_par" if (fin(rot[0]) and float(rot[0]) >= ROT_DEC) else "G"
    return dict(NF=nf, U=U, SNR=Gn / nf, proj=proj, g_par=gpar, G=G, rot=rot, dec=gpar if kind == "g_par" else G, kind=kind)


def _gsum(g):
    return dict(g_par=summ(g["g_par"]), G=summ(g["G"]), rotation=summ(g["rot"]), decision=g["kind"], dec=summ(g["dec"]))


def bq0(bank, b, alpha, med, cbar_ref, ref_X, axes, W):
    """B-Q0 of block b (plan 4.1). bank: the O3 bank (r2_rk.o3_bank), with r2RSIb as an extra loop at the extra blocks (L_B(b));
    alpha = alpha_b; med = med_b; cbar_ref = dict(sum [NP], cnt [NP]) of the cbar read of the stored references (one text per prompt):
    cbar_b = kappa_b med_b per bootstrap row; ref_X [NP, D] = R1 'ref' X; axes = R1.state[b]['axes']; W = boot_W rows.
    Evaluation n = 0..3 of loop L: push d_n (d_0 = the start, rel := 1), read-back pairs ST['ev_<L><SUF>_<n>'] (= bank ev[n + 1]),
    rel_n = rel(d_n) = R1.loops[b][L][n-1]['rel'] (critique 7; r2RSIb: its own run_loop rows). Returns dict(js (JSON-able statistics and
    decisions), tensors (directions), reps (replicate arrays), cfg (the cells' push / read directions and attenuations, B-Q1))."""
    Wt = torch.as_tensor(W, dtype=torch.float64); NR, NP = Wt.shape
    cb = (Wt @ torch.as_tensor(cbar_ref["sum"], dtype=torch.float64)) / (Wt @ torch.as_tensor(cbar_ref["cnt"], dtype=torch.float64))
    loops = [L for L in bank if not str(L).startswith("_")]; Bn = "RSI_b" if int(b) == 14 else RSIB
    js = dict(block=int(b), alpha=float(alpha), med=float(med), NP=int(NP), n_rows=int(NR), seed_boot=SEED_BOOT, loops=loops,
              missing=list(bank.get("_missing", [])), cbar_b=summ(cb), kappa_b=summ(cb / med), labels=[], independent_loop=dict(name=Bn, present=Bn in bank))
    if "RSI_a" not in bank:
        js["status"] = "not run (round-1 output missing: RSI_a)"; return dict(js=js, tensors={}, reps=dict(cbar=cb), cfg=None)
    P, G, EV = {}, {}, {}
    for L in loops:
        e = bank[L]; EV[L] = {}
        for n in range(4):
            ev = e["ev"][n + 1]; Pn = Pairs(ev["delta"], ev["ok"], ev["prompt_of"], W); P[(L, n)] = Pn
            push = e["start"] if n == 0 else e["d"][n]; rel = 1.0 if n == 0 else float(e["rel"][n])
            g = eval_gains(Pn, push, rel, alpha, cb); G[(L, n)] = g
            EV[L][str(n)] = dict(rel=rel, n_ok=int(Pn.ok.sum()), NF=summ(g["NF"]), U=summ(g["U"]), SNR=summ(g["SNR"]), **_gsum(g),
                                 first_div=fd_stats(ev["first_div"]), damage_of_push=float(ev.get("damage_of_push", float("nan"))))
    js["evaluations"] = EV; js["evaluations_note"] = "n = 0..3: push d_n, read-back d_(n+1); n = 0 = one-step gain of the start (rel := 1)"
    js["persistence"] = {L: {str(n): float(bank[L]["rel"][n]) for n in range(1, 5)} for L in loops}
    # mode set M_b, u*_b, m*^2 and its guard (G17, OD-B1)
    d4 = {L: bank[L]["d"][4].double() for L in loops}; rel4 = {L: float(bank[L]["rel"][4]) for L in loops}
    cos4 = {L: R.cosf(d4[L], d4["RSI_a"]) for L in loops}
    deb = {L: (cos4[L] / math.sqrt(rel4[L] * rel4["RSI_a"]) if (fin(rel4[L]) and fin(rel4["RSI_a"]) and rel4[L] > 0 and rel4["RSI_a"] > 0)
               else float("nan")) for L in loops}
    M = ["RSI_a"] + [L for L in loops if L != "RSI_a" and fin(deb[L]) and abs(deb[L]) >= MODE_COS]
    sg = {L: (1.0 if (L == "RSI_a" or cos4[L] >= 0) else -1.0) for L in M}
    u_star = R.unit(sum(sg[L] * d4[L] for L in M))
    pool = Pairs(torch.cat([sg[L] * torch.as_tensor(bank[L]["ev"][4]["delta"]).double() for L in M]),
                 torch.cat([torch.as_tensor(bank[L]["ev"][4]["ok"], dtype=torch.bool) for L in M]),
                 torch.cat([torch.as_tensor(bank[L]["ev"][4]["prompt_of"]).long() for L in M]), W)
    m2 = pool.att2(); p5 = float(np.nanpercentile(_np(m2)[1:], 5)) if np.isfinite(_np(m2)[1:]).any() else float("nan")
    guard = fin(p5) and p5 >= M2_GUARD
    js["mode_set"] = dict(members=M, signs=sg, debiased_cos={L: deb[L] for L in loops}, cos_d4={L: cos4[L] for L in loops}, rel_d4=rel4,
                          n_pooled_pairs=int(pool.ok.sum()))
    js["m_star2"] = dict(**summ(m2), p5=p5, guard_pass=bool(guard))
    if not guard: js["labels"].append("mode not resolved")
    # Gamma_b: inverse-variance pool of the decision gain over n = 2, 3 and the loops in M_b
    comp = [(L, n) for L in M for n in (2, 3)]; Gam, wts = ivpool([G[c]["dec"] for c in comp])
    js["Gamma"] = dict(**summ(Gam), components=[dict(loop=L, n=n, weight=w, decision=G[(L, n)]["kind"]) for (L, n), w in zip(comp, wts)])
    # cross-fit and switch (G18, OD-B12): A = RSI_a, B = RSI_b at block 14 / r2RSIb at the extra blocks
    uA = R.unit(d4["RSI_a"]); uB, cAB = None, float("nan"); cf = dict(computed=False, A="RSI_a", B=Bn); switch = False
    lam_cf = gAB = None
    if Bn in bank:
        sAB = 1.0 if R.cosf(d4["RSI_a"], d4[Bn]) >= 0 else -1.0; uB = sAB * R.unit(d4[Bn]); cAB = float(uA @ uB); cf.update(cos_AB=cAB, s=sAB)
    else: js["labels"].append("no independent loop")                                   # G49 (extra blocks) / RSI_b missing (block 14)
    if not guard: cf["reason"] = "mode not resolved (G17): no cross-fit, no switch"
    elif Bn not in bank: cf["reason"] = "no independent loop (G49): no cross-fit, no switch"
    elif not cAB > CF_COS: cf["reason"] = f"cos(u*_A, u*_B) = {cAB:.4f} <= {CF_COS}: no switch"
    else:
        def lam(X, uR):
            num = sum(P[(X, n)].proj(uR) for n in (2, 3)); den = sum(R.cosf(R.unit(bank[X]["d"][n].double()), uR) for n in (2, 3))
            return num / (alpha * cb * den), den
        lAB, denA = lam("RSI_a", uB); lBA, denB = lam(Bn, uA); lam_cf = (lAB + lBA) / 2
        gAB, wAB = ivpool([G[(X, n)]["g_par"] for X in ("RSI_a", Bn) for n in (2, 3)])
        diff = float(lam_cf[0] - gAB[0]); switch = abs(diff) > CF_SWITCH
        cf.update(computed=True, lambda_A_to_B=summ(lAB), lambda_B_to_A=summ(lBA), lambda_cf=summ(lam_cf), g_par_AB=summ(gAB),
                  g_par_AB_weights=wAB, cos_sum_A=denA, cos_sum_B=denB, diff=diff, switch=bool(switch),
                  reason=f"|lambda_cf - g_par^AB| = {abs(diff):.4f} {'>' if switch else '<='} {CF_SWITCH}")
    js["cross_fit"] = cf
    # configuration of the cells (B-Q1)
    mA2 = P[("RSI_a", 3)].att2()
    if not guard:
        mode, u_push, u_read, m2_par, mG = "not_resolved", uA, uA, mA2, torch.sqrt(mA2)
        lab = "mode not resolved: push and read along unit(d_4^RSI_a), m^2 = U / |d|^2 of RSI_a's evaluation n = 3"
    elif switch:
        mode, u_push, u_read, m2_par, mG = "switch", uA, uB, torch.full_like(m2, cAB), torch.sqrt(mA2)
        lab = "switch: push along u*_A, read along u*_B; g_par / cos(u*_A, u*_B), G / m_A"
    else:
        mode, u_push, u_read, m2_par, mG = "standard", u_star, u_star, m2, torch.sqrt(m2)
        lab = "push and read along u*_b; g_par / m*^2, G / m*"
    js["config"] = dict(mode=mode, label=lab, m2_par=summ(m2_par), mG=summ(mG), m_A2=summ(mA2))
    # descriptives: even part E_b (RSI_a evaluations n = 1..3 minus the mean of R1 'ref' X), image direction, gate p
    mids = []
    for n in (1, 2, 3):
        ev = bank["RSI_a"]["ev"][n + 1]; mids.append((torch.as_tensor(ev["xp"]).double() + torch.as_tensor(ev["xm"]).double()) / 2)
    Pm = Pairs(torch.cat(mids), torch.cat([torch.as_tensor(bank["RSI_a"]["ev"][n + 1]["ok"], dtype=torch.bool) for n in (1, 2, 3)]),
               torch.cat([torch.as_tensor(bank["RSI_a"]["ev"][n + 1]["prompt_of"]).long() for n in (1, 2, 3)]), W)
    rX = torch.as_tensor(ref_X).double(); rok = torch.isfinite(rX).all(1); rX = torch.where(rok[:, None], rX, torch.zeros_like(rX))
    wr = Wt * rok.double()[None]; E = Pm.mean() - (wr @ rX) / wr.sum(1)[:, None]
    uT = R.unit(torch.as_tensor(axes["u_T_perp"]).double()); DT = float(axes["gap_perp_cross"])
    nE = E.norm(dim=1)
    js["even_part"] = dict(norm=summ(nE), cos_u_star=summ((E @ u_star) / (nE + 1e-300)), proj_u_T_perp_over_Delta_T=summ((E @ uT) / DT),
                           Delta_T=DT, note="E_b pooled over the ok pairs of RSI_a's evaluations n = 1..3")
    js["image_cos_d1_u_star"] = {L: R.cosf(bank[L]["d"][1].double(), u_star) for L in loops}
    if int(b) == 14:
        ev = bank["RSI_a"]["ev"][4]; ok = torch.as_tensor(ev["ok"], dtype=torch.bool)
        a = (torch.as_tensor(ev["delta"]).double()[ok] @ R.unit(bank["RSI_a"]["d"][3].double())).numpy()
        p = float(a.mean() / a.std(ddof=1)) if len(a) > 1 and a.std(ddof=1) > 0 else float("nan")
        js["gate_p"] = dict(p=p, threshold=P_GATE, label="verdicts expected mostly undecided" if (fin(p) and p < P_GATE) else None)
    js["status"] = "ok"
    tens = dict(u_star=u_star, uA=uA, uB=uB, u_push=u_push, u_read=u_read, d4=d4)
    reps = dict(cbar=cb, m2=m2, mA2=mA2, Gamma=torch.as_tensor(Gam), m2_par=m2_par, mG=mG, lam_cf=lam_cf,
                gAB=None if gAB is None else torch.as_tensor(gAB), gains=G, pool=pool)
    cfg = dict(mode=mode, u_push=u_push, u_read=u_read, m2_par=m2_par, mG=mG, cosAB=cAB)
    return dict(js=js, tensors=tens, reps=reps, cfg=cfg, P=P)


def cell_cbar(cell, W):
    """The cell's own cbar per bootstrap row: pooled mean of min(|h_t|, cap_b) over the logged continuation tokens of every text (both
    texts of a pair carry their prompt's weight)."""
    Wt = torch.as_tensor(W, dtype=torch.float64); po = torch.as_tensor(cell["prompt_of"]).long().repeat_interleave(2)
    s = torch.as_tensor(cell["cbar"]["sum"], dtype=torch.float64); c = torch.as_tensor(cell["cbar"]["cnt"], dtype=torch.float64)
    Wp = Wt[:, po]; return (Wp @ s) / (Wp @ c)


def cell_gains(Pw, cbar_r, cfg, a, alpha):
    """Gains of one cell window (B-Q1): standard g_par = u*_b . d / (a alpha_b cbar m*^2), G = sqrt(max(U, 0)) / (a alpha_b cbar m*);
    switch g_par = u*_B . d / (a alpha_b cbar cos(u*_A, u*_B)), G = sqrt(max(U, 0)) / (a alpha_b cbar m_A); mode not resolved: along
    unit(d_4^RSI_a) with m^2 of RSI_a's evaluation n = 3. cfg = bq0's cfg (m2_par, mG per bootstrap row); cbar_r = the cell's own cbar.
    rotation = g_par / G; decision gain = g_par if the point rotation >= 0.9, else G. Returns dict of [R] tensors plus 'kind'."""
    U = Pw.U(); Gn = torch.sqrt(U.clamp(min=0)); proj = Pw.proj(cfg["u_read"]); den = a * alpha * torch.as_tensor(cbar_r).double()
    m2p = torch.as_tensor(cfg["m2_par"]).double(); mG = torch.as_tensor(cfg["mG"]).double()
    gpar = torch.where(m2p > 0, proj / (den * m2p), torch.full_like(proj, float("nan")))
    G = torch.where(mG > 0, Gn / (den * mG), torch.full_like(proj, float("nan")))
    rot = torch.where(G > 0, gpar / G, torch.full_like(G, float("nan")))
    kind = "g_par" if (fin(rot[0]) and float(rot[0]) >= ROT_DEC) else "G"
    return dict(U=U, proj=proj, SNR=Gn / Pw.nf(), g_par=gpar, G=G, rot=rot, dec=gpar if kind == "g_par" else G, kind=kind)


def _rows(cfg, k):
    """cfg restricted to the first k bootstrap rows (k = 1: the point estimate)."""
    out = dict(cfg)
    for key in ("m2_par", "mG"): out[key] = torch.as_tensor(cfg[key]).double()[:k]
    return out


def cell_table(cells, cfg, alpha, W, windows=WINDOWS):
    """Gains of every cell and window: {name: {window: cell_gains(...)}} under the rows of W (the first rows of cfg's arrays)."""
    k = W.shape[0]; c = _rows(cfg, k); out = {}
    for name, cell in cells.items():
        cb = cell_cbar(cell, W); out[name] = {}
        for w in windows:
            win = cell["windows"][w]; out[name][w] = cell_gains(Pairs(win["delta"], win["ok"], cell["prompt_of"], W), cb, c, cell["a"], alpha)
    return out


def cell_points(cells, cfg, alpha, NP):
    """Point estimates (decision gain, rotation) of every cell in the primary window: {name: (dec, rot)} (OD-B7's inputs)."""
    T = cell_table(cells, cfg, alpha, np.ones((1, NP)), windows=("all",))
    return {n: (float(T[n]["all"]["dec"][0]), float(T[n]["all"]["rot"][0])) for n in T}


def choose_power_cell(points, order=None):
    """B-Q6 / OD-B7: among the cells with rotation < 0.8 (point estimates) the one with the smallest |log decision gain| (a non-positive
    decision gain counts as infinitely far from 1); None if no cell has rotation < 0.8 ('not needed', G24). Ties: the earlier cell."""
    names = list(order or points); cand = [n for n in names if fin(points[n][1]) and points[n][1] < ROT_PWR]
    if not cand: return None
    return min(cand, key=lambda n: abs(math.log(points[n][0])) if (fin(points[n][0]) and points[n][0] > 0) else float("inf"))


def power_gains(Pp, push, rel_cell, a, alpha, cbar_r):
    """Gains of the power step (B-Q6) by B-Q0's formulas with the pushed direction unit(d_cell) as d_n and rel_n = rel(d_cell) (the cell's
    split-half reliability): g_par = unit(d_cell) . d_pwr / (a alpha_b cbar rel_cell), G = sqrt(max(U, 0)) / (a alpha_b cbar
    sqrt(rel_cell)), rotation, decision gain."""
    g = eval_gains(Pp, push, rel_cell, a * alpha, cbar_r); return g


# ======================================================================================================================================
# Model side
# ======================================================================================================================================
def _unsteered(rk):
    if rk.STEER["v"] is not None or rk.STEER["V"] is not None: raise RuntimeError("read with rk's steering active")


def generate_rows(rk, rows, d, alpha, T, entropy=False, cbar=True, batch=32):
    """rk.generate(rows, d, alpha) at sampling temperature T (rk.TEMP, restored afterwards), with the cbar hook (r2_rk.CbarHook,
    generation mode) and, if asked, the entropy of softmax(logits / T) at every sampling step (per text: the steps that produced its
    continuation tokens; the step that produced the stop token is excluded). rows: (prompt ids, key, sign). Returns dict(conts,
    capped_share, cbar_sum, cbar_cnt, ent_sum, ent_cnt)."""
    store = []; Gum = rk.Gumbel

    class EntGumbel(Gum):
        def __init__(self, keys): super().__init__(keys); store.append([])

        def __call__(self, input_ids, scores):
            lp = torch.log_softmax(scores.float() / rk.TEMP, -1); store[-1].append(-(lp.exp() * lp).sum(-1).detach().double().cpu())
            return super().__call__(input_ids, scores)
    with contextlib.ExitStack() as es:
        es.enter_context(R.temperature(T)); hk = es.enter_context(R.CbarHook()) if cbar else None
        if entropy: rk.Gumbel = EntGumbel
        try:
            rk.capped_reset(); conts = rk.generate(rows, d, alpha, batch=batch); cs = rk.capped_share() if d is not None else float("nan")
        finally:
            rk.Gumbel = Gum
    out = dict(conts=conts, capped_share=cs)
    if hk is not None:
        vals = hk.per_text(conts); out["cbar_sum"] = [float(v.sum()) for v in vals]; out["cbar_cnt"] = [int(len(v)) for v in vals]
    if entropy:
        es_, ec, i = [], [], 0
        for rec in store:
            S = torch.stack(rec) if rec else torch.zeros(0, 0, dtype=torch.float64)
            for r in range(S.shape[1]):
                v = S[: min(len(conts[i]), S.shape[0]), r]; es_.append(float(v.sum())); ec.append(int(len(v))); i += 1
        assert i == len(conts), f"entropy log saw {i} rows, {len(conts)} continuations"
        out["ent_sum"], out["ent_cnt"] = es_, ec
    return out


def read_full(rk, seqs_prompt, conts, first_div=None, batch=32, n_first=20, nll=True):
    """One full unsteered forward per batch (rk.read's batching and left padding) with a forward hook on block b: the three read
    windows with exactly r2_rk.read_windows' rules (all / div / first20; identical pairs delta 0 in 'div'; a text without tokens in a
    window is NaN there and its pair not ok) and, from the same pass (B-Q1, critique v2-10), the NLL of every continuation token under
    the unsteered model (T = 1; logits computed only at the continuation positions). Rows 2j / 2j + 1 = the + / - text of pair j.
    Returns dict(first_div, X, xp, xm, delta, ok (per window), nll_sum [2N], nll_cnt [2N])."""
    _unsteered(rk); N2 = len(conts); assert N2 % 2 == 0 and batch % 2 == 0
    if first_div is None: first_div = [R.first_divergence(conts[2 * j], conts[2 * j + 1]) for j in range(N2 // 2)]
    X = {w: torch.full((N2, rk.D), float("nan"), dtype=torch.float64) for w in WINDOWS}
    nsum, ncnt, cap = np.zeros(N2), np.zeros(N2, dtype=np.int64), {}

    def hook(mod, inp, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).detach()
    H = rk.LAY[rk.BL].register_forward_hook(hook)
    try:
        for i in range(0, N2, batch):
            pb, cb = seqs_prompt[i:i + batch], conts[i:i + batch]
            ids, am, pad = rk.pad_batch([list(p) + list(c) for p, c in zip(pb, cb)]); T = ids.shape[1]
            with torch.no_grad():
                out = rk.model.model(input_ids=ids.to(rk.DEV), attention_mask=am.to(rk.DEV))
                hs = cap.pop("h").double().cpu(); last = out.last_hidden_state
                for r, c in enumerate(cb):
                    k = i + r; L = len(c); fd = first_div[k // 2]
                    if not L: continue
                    seg = hs[r, T - L:]; X["all"][k] = seg.mean(0); X["first20"][k] = seg[:n_first].mean(0)
                    if fd < 0: X["div"][k] = X["all"][k]
                    elif L > fd: X["div"][k] = seg[fd:].mean(0)
                    if nll:
                        lp = torch.log_softmax(rk.model.lm_head(last[r, T - L - 1:T - 1]).float(), -1)
                        tg = torch.tensor(list(c), device=lp.device)
                        nsum[k] = -float(lp[torch.arange(L, device=lp.device), tg].double().sum()); ncnt[k] = L
    finally:
        H.remove()
    out = dict(first_div=list(first_div), X=X, xp={}, xm={}, delta={}, ok={}, nll_sum=nsum.tolist(), nll_cnt=ncnt.tolist())
    ident = torch.tensor([fd < 0 for fd in first_div], dtype=torch.bool)
    for w in WINDOWS:
        xp, xm = X[w][0::2], X[w][1::2]; d = (xp - xm) / 2
        if w == "div": d[ident] = 0.0
        out["xp"][w], out["xm"][w], out["delta"][w] = xp, xm, d; out["ok"][w] = ~(torch.isnan(xp).any(1) | torch.isnan(xm).any(1))
    return out


def score_grads(rk, seqs_prompt, conts, T=T_FDT, batch=8):
    """B-Q7: per text one forward and backward pass with rk's zero-leaf hook (STEER['V']: leaf V added as min(|h_t|, cap_b) V at the
    output of block b from the first user token on, as rk.grads) on y = sum over the continuation positions of log softmax(logits_t /
    T)[y_t], plus the log-probability of stopping (log of the summed probability of rk.STOP at the next position) if the text ended
    early (len < NNEW; the stop token itself is not stored). S_j = dy / dV; x_j = the block-b reading (mean over the continuation tokens)
    from the same pass; own cbar = min(|h_t|, cap_b) over the continuation tokens. Returns dict(X [N, D], S [N, D], y [N], ended_early,
    cbar_sum, cbar_cnt) (float64; X NaN for an empty text)."""
    _unsteered(rk); N, D = len(conts), rk.D; capb = float(rk.ST["ref"]["cap"]); stop = sorted(int(t) for t in rk.STOP)
    X = torch.full((N, D), float("nan"), dtype=torch.float64); S = torch.zeros(N, D, dtype=torch.float64); Y = np.full(N, np.nan)
    csum, ccnt, ended, cap = np.zeros(N), np.zeros(N, dtype=np.int64), [len(c) < rk.NNEW for c in conts], {}

    def hook(mod, inp, out):
        cap["h"] = out[0] if isinstance(out, tuple) else out
    H = rk.LAY[rk.BL].register_forward_hook(hook)
    try:
        for i in range(0, N, batch):
            pb, cb = seqs_prompt[i:i + batch], conts[i:i + batch]
            ids, am, pad = rk.pad_batch([list(p) + list(c) for p, c in zip(pb, cb)]); B, Tn = ids.shape
            V = torch.zeros(B, D, device=rk.DEV, requires_grad=True)
            rk.STEER.update(V=V, v=None, cap=rk.ST["ref"]["cap"], rowstart=pad + rk.START)
            try:
                with torch.enable_grad():
                    last = rk.model.model(input_ids=ids.to(rk.DEV), attention_mask=am.to(rk.DEV)).last_hidden_state; h = cap.pop("h"); ys = []
                    for r, c in enumerate(cb):
                        L = len(c); lp = torch.log_softmax(rk.model.lm_head(last[r, Tn - L - 1:Tn]).float() / T, -1)
                        y = lp[torch.arange(L, device=lp.device), torch.tensor(list(c), device=lp.device)].sum() if L else lp.new_zeros(())
                        if L < rk.NNEW: y = y + torch.logsumexp(lp[L, stop], 0)
                        ys.append(y)
                    g, = torch.autograd.grad(torch.stack(ys).sum(), V)
            finally:
                rk.steer_off()
            S[i:i + B] = g.double().cpu(); Y[i:i + B] = [float(y.detach()) for y in ys]; hd = h.detach().double().cpu()
            for r, c in enumerate(cb):
                L = len(c)
                if not L: continue
                seg = hd[r, Tn - L:]; X[i + r] = seg.mean(0); nr = seg.norm(dim=-1).clamp(max=capb); csum[i + r] = float(nr.sum()); ccnt[i + r] = L
    finally:
        H.remove()
    return dict(X=X, S=S, y=Y.tolist(), ended_early=ended, cbar_sum=csum.tolist(), cbar_cnt=ccnt.tolist())


# ======================================================================================================================================
# Stages
# ======================================================================================================================================
class Ctx:
    """One block of one process: paths and the lazily opened rk_loop module (None plus the reason if the round-1 block is missing)."""
    def __init__(self, root, r1, r2, b):
        self.root, self.r1, self.r2, self.b = root, r1, r2, int(b); self.rk = None; self.err = None

    def open(self):
        if self.rk is None and self.err is None:
            try: self.rk = R.open_block(self.root, self.r1, self.r2, self.b)
            except R.R1Missing as e: self.err = str(e); R.log(f"block {self.b}: not run (round-1 output missing): {e}")
        return self.rk

    def path(self, name): return os.path.join(self.r2, "theta", name)
    def part(self, name): return os.path.join(self.r2, "theta", "parts", f"b{self.b}_{name}.pt")


class _AltCtx(Ctx):
    """The open block of cx with another output folder (<r2_dir>/<sub>/ instead of <r2_dir>/theta/); selftest variants only."""
    def __init__(self, cx, sub):
        super().__init__(cx.root, cx.r1, cx.r2, cx.b); self.rk, self.err, self.sub = cx.rk, cx.err, sub

    def path(self, name): return os.path.join(self.r2, self.sub, name)
    def part(self, name): return os.path.join(self.r2, self.sub, "parts", f"b{self.b}_{name}.pt")


def _base(cx, stage):
    return dict(script="r2_theta.py", stage=stage, plan=PLAN, plan_sha256=PLAN_SHA256, block=cx.b, mock=bool(R.MOCK))


def _meta(cx):
    import transformers
    return dict(alpha_b=R.BLK.get("alpha"), alpha_scale=R.BLK.get("alpha_scale"), suffix=R.BLK.get("suffix"), cap_b=R.BLK.get("cap"),
                med_b=R.BLK.get("med"), NP=int(cx.rk.NP), NNEW=int(cx.rk.NNEW), device=str(cx.rk.DEV), torch=torch.__version__,
                transformers=transformers.__version__)


def _missing(cx, stage, **extra):
    return dict(_base(cx, stage), status="not run (round-1 output missing)", error=cx.err, **extra)


def _load_pt(p): return torch.load(p, map_location="cpu", weights_only=False)


def stage_rsib(cx):
    """The second RSI loop r2RSIb at an extra block (B-Q0, OD-B2). Returns its JSON."""
    if cx.b == 14: raise SystemExit("rsib runs at the extra blocks only (block 14 has round 1's RSI_b)")
    js_p, pt_p = cx.path(f"b{cx.b}_r2RSIb.json"), cx.path(f"b{cx.b}_r2RSIb.pt")
    if os.path.exists(js_p) and (os.path.exists(pt_p) or json.load(open(js_p)).get("status") != "ran"): return json.load(open(js_p))
    rk = cx.open()
    if rk is None:
        js = _missing(cx, "rsib"); R.write_json(js_p, js); return js
    start = R.unit(rk.ST["axes"]["u_T_perp"].double()); att_p = os.path.join(os.path.dirname(cx.part("x")), f"b{cx.b}_rsib_attempts.json")
    try:
        rows = rk.run_loop(RSIB, start, 2, rounds=4)
    except Exception as e:                                     # G49: a loop that fails in two attempts is 'not run'; B-Q0 then has no cross-fit
        n = (json.load(open(att_p))["n"] if os.path.exists(att_p) else 0) + 1; R.write_json(att_p, dict(n=n, error=repr(e)))
        if n < 2: raise
        js = dict(_base(cx, "rsib"), status="not run (r2RSIb failed in two attempts; G49: no independent loop)", error=repr(e), attempts=n)
        R.write_json(js_p, js); R.log(f"r2RSIb b{cx.b}: {js['status']}: {e!r}"); return js
    L = RSIB + rk.SUF
    keys = sorted(k for k in rk.ST if k.startswith((f"ev_{L}_", f"dir_{L}_", f"dmg_{L}_")))
    R.torch_save(pt_p, dict(loop=L, suffix=rk.SUF, start=start.float(), rows=rows, entries={k: rk.ST[k] for k in keys}))
    js = dict(_base(cx, "rsib"), status="ran", loop=L, nseed=2, rounds=4, start="unit(u_T_perp(b))", keys="keyseed('r2RSIb' + SUF_b, n, p, s, b)",
              rows=rows, state_keys=keys, **_meta(cx), finder=list(R.FINDER_LOG))
    R.write_json(js_p, js); R.log(f"r2RSIb b{cx.b}: rel {[round(r['rel'], 3) for r in rows]}")
    return js


def stage_q0(cx):
    """B-Q0 (block b). Returns (js, pt)."""
    js_p, pt_p = cx.path(f"b{cx.b}_q0.json"), cx.path(f"b{cx.b}_q0.pt")
    if os.path.exists(js_p):
        js = json.load(open(js_p))
        if js.get("status") != "ok" or os.path.exists(pt_p): return js, (_load_pt(pt_p) if os.path.exists(pt_p) else None)
    rk = cx.open()
    if rk is None:
        js = _missing(cx, "q0"); R.write_json(js_p, js); return js, None
    extra, rsib = {}, None
    if cx.b != 14:
        rsib = stage_rsib(cx)
        if rsib.get("status") == "ran":                 # the loop from its exported entries (as r2_theta_verdict.py reads it), not from the
            P = _load_pt(cx.path(f"b{cx.b}_r2RSIb.pt"))  # in-memory round-2 state, which a re-copied state.pt would lack (review finding theta-1)
            extra = {RSIB: dict(ST=P["entries"], rows=rsib["rows"], start=P["start"].double(), suffix=P["suffix"])}
    loops = R.load_r1_json(cx.r1, f"rk/b{cx.b}/loops.json", cx.b); bank = R.o3_bank(rk.ST, loops, cx.b, extra=extra)
    if rsib is not None and rsib.get("status") == "ran" and RSIB not in bank:     # never a silent 'no independent loop' for a loop that ran (G49)
        raise RuntimeError(f"r2RSIb ran but its entries are missing in {cx.path(f'b{cx.b}_r2RSIb.pt')}: B-Q0 not written (the step fails and is retried)")
    ref = rk.ST["ref"]; NP = rk.NP; kb = R.cbar_read([rk.PIDS[p] for p in range(NP)], ref["conts"])
    cnt = [len(c) for c in ref["conts"]]; sm = [float(m) * n if n else 0.0 for m, n in zip(kb["per_text_mean"].tolist(), cnt)]
    cbar_ref = dict(sum=sm, cnt=cnt, pooled=kb["pooled"], n_tokens=kb["n_tokens"])
    res = bq0(bank, cx.b, R.BLK["alpha"], R.BLK["med"], cbar_ref, ref["X"], rk.ST["axes"], boot_W(NP))
    js = _base(cx, "q0"); js.update(_meta(cx)); js.update(res["js"])
    js.update(rsib=None if rsib is None else dict(status=rsib.get("status"), loop=rsib.get("loop")), finder=list(R.FINDER_LOG),
              keys_note="no generation in B-Q0 (one cbar read of the 64 stored references); r2RSIb via the rsib stage")
    cfg = res["cfg"]; t = res["tensors"]
    pt = dict(block=cx.b, cbar_ref=cbar_ref, NP=NP, status=js["status"],
              cfg=None if cfg is None else dict(mode=cfg["mode"], u_push=cfg["u_push"].float(), u_read=cfg["u_read"].float(), cosAB=cfg["cosAB"],
                                                m2_par=_np(cfg["m2_par"]), mG=_np(cfg["mG"]), cbar_b=_np(res["reps"]["cbar"])),
              u_star=None if not t else t["u_star"].float(), uA=None if not t else t["uA"].float(),
              uB=None if (not t or t["uB"] is None) else t["uB"].float(), d4={} if not t else {L: v.float() for L, v in t["d4"].items()})
    R.torch_save(pt_p, pt); R.write_json(js_p, js)
    if js["status"] == "ok":
        R.log(f"B-Q0 b{cx.b}: M_b {js['mode_set']['members']}, m*^2 {js['m_star2']['pt']:.3f} (p5 {js['m_star2']['p5']:.3f}), "
              f"Gamma {js['Gamma']['pt']:.3f} [{js['Gamma']['lo']:.3f}, {js['Gamma']['hi']:.3f}], mode {js['config']['mode']}")
    return js, pt


def _q0_cfg(cx):
    js, pt = stage_q0(cx)
    if js.get("status") != "ok" or pt is None or pt.get("cfg") is None: return js, None
    c = pt["cfg"]
    return js, dict(mode=c["mode"], u_push=c["u_push"].double(), u_read=c["u_read"].double(), m2_par=torch.as_tensor(c["m2_par"]),
                    mG=torch.as_tensor(c["mG"]), cosAB=c["cosAB"])


def stage_checks(cx):
    """B-Q1 code checks at block 14 (G20). Returns the JSON."""
    if cx.b != 14: raise SystemExit("the theta code checks run at block 14 only")
    js_p = cx.path("checks.json")
    if os.path.exists(js_p): return json.load(open(js_p))
    rk = cx.open()
    if rk is None:
        js = _missing(cx, "checks", passed=None); R.write_json(js_p, js); return js
    q0, cfg = _q0_cfg(cx)
    if cfg is None:
        js = dict(_base(cx, "checks"), status=f"not run (B-Q0: {q0.get('status')})", passed=None); R.write_json(js_p, js); return js
    u, al = cfg["u_push"], R.BLK["alpha"]
    rows = lambda chk: [(rk.PIDS[p], rk.keyseed("r2CHK", chk, p), sg) for p in range(4) for sg in (1.0, -1.0)]
    Z = generate_rows(rk, rows("zero"), u, 0.0, 0.7, cbar=False)["conts"]; zero = sum(Z[2 * j] == Z[2 * j + 1] for j in range(4))
    M1 = generate_rows(rk, rows("mirror"), u, al, 0.7, cbar=False)["conts"]; M2 = generate_rows(rk, rows("mirror"), -u, al, 0.7, cbar=False)["conts"]
    mirror = sum(M2[2 * j] == M1[2 * j + 1] and M2[2 * j + 1] == M1[2 * j] for j in range(4))
    E = {}
    for T in (0.3, 0.7):
        g = generate_rows(rk, rows("entropy"), u, al, T, entropy=True, cbar=False); E[T] = float(sum(g["ent_sum"]) / max(sum(g["ent_cnt"]), 1))
    passed = zero == 4 and mirror >= 3 and E[0.3] < E[0.7]
    js = dict(_base(cx, "checks"), status="ok", passed=bool(passed), zero_push_identical=int(zero), zero_rule="4 of 4", mirror_swapped=int(mirror),
              mirror_rule=">= 3 of 4", entropy_T03=E[0.3], entropy_T07=E[0.7], entropy_rule="entropy(T 0.3) < entropy(T 0.7)", push="u_push of B-Q0",
              alpha=al, keys="keyseed('r2CHK', check, p), check in zero / mirror / entropy, p = 0..3",
              label=None if passed else "descriptive (code check failed)", texts=dict(zero=Z, mirror_plus=M1, mirror_minus=M2), **_meta(cx))
    R.write_json(js_p, js); R.log(f"theta code checks: zero {zero}/4, mirror {mirror}/4, entropy {E[0.3]:.3f} < {E[0.7]:.3f}: {passed}")
    return js


def run_cell(cx, spec, u_push, damage):
    """One theta cell (B-Q1): NP prompts x nseed seeds, keys keyseed('theta', b, p, s) shared by every cell of the block."""
    rk = cx.rk; a, T, ns = spec["a"], spec["T"], spec["nseed"]; al = a * R.BLK["alpha"]; rows, po, so, keys = [], [], [], []
    for p in range(rk.NP):
        for s in range(ns):
            k = rk.keyseed("theta", cx.b, p, s); rows += [(rk.PIDS[p], k, 1.0), (rk.PIDS[p], k, -1.0)]; po.append(p); so.append(s); keys.append(k)
    t0 = time.time(); G = generate_rows(rk, rows, u_push, al, T, entropy=True, cbar=True); conts = G["conts"]
    fdv = [R.first_divergence(conts[2 * j], conts[2 * j + 1]) for j in range(len(po))]
    RD = read_full(rk, [rk.PIDS[p] for p in po for _ in (0, 1)], conts, fdv); kl = R.damage_T(u_push, al, T)
    cell = dict(spec, alpha=al, n_pairs=len(po), keys=keys, prompt_of=torch.tensor(po), seed_of=torch.tensor(so), conts=conts, first_div=fdv,
                windows={w: dict(delta=RD["delta"][w].float(), ok=RD["ok"][w]) for w in WINDOWS}, xp=RD["xp"]["all"].float(), xm=RD["xm"]["all"].float(),
                cbar=dict(sum=G["cbar_sum"], cnt=G["cbar_cnt"]), entropy=dict(sum=G["ent_sum"], cnt=G["ent_cnt"]),
                nll=dict(sum=RD["nll_sum"], cnt=RD["nll_cnt"]), rep4=[rep4(c) for c in conts], capped_share=G["capped_share"],
                dropped=int((~RD["ok"]["all"]).sum()), damage=float(damage), KL_T=float(kl), seconds=time.time() - t0)
    R.log(f"cell b{cx.b} {spec['name']}: {len(po)} pairs, identical {sum(k == -1 for k in fdv)}, KL {kl:.4f}, {time.time() - t0:.0f}s")
    return cell


def _cell_point_json(cx, cells, cfg, NP):
    T = cell_table(cells, cfg, R.BLK["alpha"], np.ones((1, NP)))
    out = {}
    for n, c in cells.items():
        po = c["prompt_of"]; okA = c["windows"]["all"]["ok"]; rel, r = halving_rel(c["windows"]["all"]["delta"], okA, po, NP)
        nl = np.array(c["nll"]["cnt"], float); et = np.array(c["entropy"]["cnt"], float)
        out[n] = dict(a=c["a"], T=c["T"], nseed=c["nseed"], reserve=c.get("reserve"), n_pairs=c["n_pairs"], alpha=c["alpha"],
                      windows={w: dict(g_par=float(T[n][w]["g_par"][0]), G=float(T[n][w]["G"][0]), rotation=float(T[n][w]["rot"][0]),
                                       decision=T[n][w]["kind"], dec=float(T[n][w]["dec"][0]), SNR=float(T[n][w]["SNR"][0]),
                                       n_ok=int(torch.as_tensor(c["windows"][w]["ok"]).sum())) for w in WINDOWS},
                      rel=rel, r=r, cbar=float(sum(c["cbar"]["sum"]) / max(sum(c["cbar"]["cnt"]), 1)), damage=c["damage"], KL_T=c["KL_T"],
                      nll_per_token=float(np.sum(c["nll"]["sum"]) / max(nl.sum(), 1)), rep4=float(np.nanmean(c["rep4"])) if np.isfinite(c["rep4"]).any() else float("nan"),
                      entropy=float(np.sum(c["entropy"]["sum"]) / max(et.sum(), 1)), first_div=fd_stats(c["first_div"]), capped_share=c["capped_share"],
                      dropped=c["dropped"])
    return out


def stage_cells(cx):
    """Every theta cell of the block (B-Q1, B-Q3, B-Q4, B-Q5 incl. reserves 1-3). Returns (js, pt)."""
    js_p, pt_p = cx.path(f"b{cx.b}_cells.json"), cx.path(f"b{cx.b}_cells.pt")
    if os.path.exists(js_p):
        js = json.load(open(js_p))
        if js.get("status") != "ok" or os.path.exists(pt_p): return js, (_load_pt(pt_p) if os.path.exists(pt_p) else None)
    rk = cx.open()
    if rk is None:
        js = _missing(cx, "cells"); R.write_json(js_p, js); return js, None
    q0, cfg = _q0_cfg(cx)
    if cfg is None:
        js = dict(_base(cx, "cells"), status=f"not run (B-Q0: {q0.get('status')})"); R.write_json(js_p, js); return js, None
    u_push = cfg["u_push"]; al = R.BLK["alpha"]; dp = cx.part("damage")
    if os.path.exists(dp): curve = _load_pt(dp)
    else:
        curve = {a: float(rk.damage(u_push, a * al)) for a in GRID_A}; R.torch_save(dp, curve)
    cells = {}
    for spec in cells_of(cx.b):
        pp = cx.part(f"cell_{spec['name']}")
        if os.path.exists(pp): cells[spec["name"]] = _load_pt(pp); continue
        cells[spec["name"]] = run_cell(cx, spec, u_push, curve[spec["a"]]); R.torch_save(pp, cells[spec["name"]])
    pt = dict(block=cx.b, status="ok", mode=cfg["mode"], u_push=u_push.float(), u_read=cfg["u_read"].float(), damage_curve=curve, alpha_b=al,
              cells=cells, order=[s["name"] for s in cells_of(cx.b)])
    pts = _cell_point_json(cx, cells, cfg, rk.NP)
    js = dict(_base(cx, "cells"), status="ok", mode=cfg["mode"], damage_curve={str(a): v for a, v in curve.items()}, cells=pts,
              order=pt["order"], keys="keyseed('theta', b, p, s), seeds 0-1 (0-3 for the 256-pair cell), shared by every cell of the block",
              note="point estimates; intervals, reserve conditions and rules in r2_theta_verdict.py", **_meta(cx))
    R.torch_save(pt_p, pt); R.write_json(js_p, js)
    for spec in cells_of(cx.b):
        for nm in (f"cell_{spec['name']}",):
            if os.path.exists(cx.part(nm)): os.remove(cx.part(nm))
    if os.path.exists(dp): os.remove(dp)
    R.log("cells b{}: ".format(cx.b) + ", ".join(f"{n} dec {v['windows']['all']['dec']:.3f} rot {v['windows']['all']['rotation']:.2f}" for n, v in pts.items()))
    return js, pt


def stage_turn2(cx):
    """B-Q8: turn 2 after four block-14 cells. Returns the .pt content."""
    if cx.b != 14: raise SystemExit("turn 2 (B-Q8) runs at block 14 only")
    pt_p, js_p = cx.path("b14_turn2.pt"), cx.path("b14_turn2.json")
    if os.path.exists(pt_p): return _load_pt(pt_p)
    rk = cx.open()
    if rk is None:
        js = _missing(cx, "turn2"); R.torch_save(pt_p, dict(status=js["status"], error=cx.err)); R.write_json(js_p, js); return _load_pt(pt_p)
    cj, C = stage_cells(cx)
    if C is None:
        st = f"not run (cells: {cj.get('status')})"; R.torch_save(pt_p, dict(status=st)); R.write_json(js_p, dict(_base(cx, "turn2"), status=st))
        return _load_pt(pt_p)
    tmpl = lambda msgs: rk.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    suffix = STIM.check_turn2(rk.enc, rk.chat, tmpl); out = {}
    for name in TURN2_CELLS:
        pp = cx.part(f"turn2_{name}")
        if os.path.exists(pp): out[name] = _load_pt(pp); continue
        c = C["cells"][name]; a, T = c["a"], c["T"]; po, so = c["prompt_of"].tolist(), c["seed_of"].tolist(); rows, keys = [], []
        for j, (p, s) in enumerate(zip(po, so)):
            k2 = rk.keyseed("theta2", cx.b, a, T, p, s); keys.append(k2)
            rows += [(STIM.turn2_ids(rk.PIDS[p], c["conts"][2 * j], suffix), k2, 1.0), (STIM.turn2_ids(rk.PIDS[p], c["conts"][2 * j + 1], suffix), k2, 1.0)]
        t0 = time.time(); g = generate_rows(rk, rows, None, 0.0, T, entropy=False, cbar=False); c2 = g["conts"]
        X2 = rk.read([rk.PIDS[p] for p in po for _ in (0, 1)], c2).double(); xp2, xm2 = X2[0::2], X2[1::2]
        ok2 = ~(torch.isnan(xp2).any(1) | torch.isnan(xm2).any(1))
        nrt = sum(not STIM.round_trip_stable(rk.enc, rk.tok.decode, t) for t in c["conts"])
        out[name] = dict(cell=name, a=a, T=T, reserve="reserve 5" if name == cell_name(2, 0.7) else None, keys=keys, prompt_of=c["prompt_of"].clone(),
                         seed_of=c["seed_of"].clone(), conts2=c2, input_len=[len(r[0]) for r in rows], delta2=((xp2 - xm2) / 2).float(), ok2=ok2,
                         xp2=xp2.float(), xm2=xm2.float(), first_div2=[R.first_divergence(c2[2 * j], c2[2 * j + 1]) for j in range(len(po))],
                         n_turn1_not_round_trip=int(nrt), n_turn1_texts=len(c["conts"]), seconds=time.time() - t0)
        R.torch_save(pp, out[name]); R.log(f"turn 2 after {name}: {len(po)} pairs, {nrt} turn-1 texts not round-trip stable, {time.time() - t0:.0f}s")
    pt = dict(block=14, status="ok", suffix_ids=list(suffix), cells=out, u_read=C["u_read"], mode=C["mode"],
              keys="keyseed('theta2', b, a, T, p, s) (a, T as Python floats), shared within a pair; no push; read as the reply to the plain prompt")
    R.torch_save(pt_p, pt)
    u = C["u_read"].double(); summ_js = {}
    for name, t in out.items():
        c = C["cells"][name]; P1 = Pairs(c["windows"]["all"]["delta"], c["windows"]["all"]["ok"], c["prompt_of"], np.ones((1, rk.NP)))
        P2 = Pairs(t["delta2"], t["ok2"], t["prompt_of"], np.ones((1, rk.NP))); d1, d2 = float(P1.proj(u)[0]), float(P2.proj(u)[0])
        summ_js[name] = dict(a=t["a"], T=t["T"], reserve=t["reserve"], K=d2 / d1 if d1 != 0 else float("nan"), u_read_d1=d1, u_read_d2=d2,
                             norm_ratio=math.sqrt(max(float(P2.U()[0]), 0) / float(P1.U()[0])) if float(P1.U()[0]) > 0 else float("nan"),
                             n_turn1_not_round_trip=t["n_turn1_not_round_trip"], n_pairs=len(t["keys"]), n_ok2=int(t["ok2"].sum()))
    R.write_json(js_p, dict(_base(cx, "turn2"), status="ok", turn2_suffix=STIM.TURN2_SUFFIX, suffix_ids=list(suffix), cells=summ_js,
                            note="point estimates; K's interval, G26 and R6 in r2_theta_verdict.py", **_meta(cx)))
    for name in TURN2_CELLS:
        if os.path.exists(cx.part(f"turn2_{name}")): os.remove(cx.part(f"turn2_{name}"))
    return pt


def stage_power(cx):
    """B-Q6, the power step (G24, OD-B7). Returns the JSON."""
    js_p, pt_p = cx.path(f"b{cx.b}_power.json"), cx.path(f"b{cx.b}_power.pt")
    if os.path.exists(js_p): return json.load(open(js_p))
    rk = cx.open()
    if rk is None:
        js = _missing(cx, "power"); R.write_json(js_p, js); return js
    q0, cfg = _q0_cfg(cx); cj, C = stage_cells(cx)
    if cfg is None or C is None:
        js = dict(_base(cx, "power"), status=f"not run (B-Q0: {q0.get('status')}, cells: {cj.get('status')})"); R.write_json(js_p, js); return js
    pts = cell_points(C["cells"], cfg, R.BLK["alpha"], rk.NP); choice = choose_power_cell(pts, C["order"])
    base = dict(_base(cx, "power"), candidates={n: dict(dec=v[0], rotation=v[1]) for n, v in pts.items()}, rule="rotation < 0.8; smallest |log decision gain| (OD-B7)")
    if choice is None:
        js = dict(base, status="not needed", note="no cell has rotation < 0.8 (G24)", **_meta(cx)); R.write_json(js_p, js); R.log(f"power step b{cx.b}: not needed")
        return js
    c = C["cells"][choice]; win = c["windows"]["all"]; Pc = Pairs(win["delta"], win["ok"], c["prompt_of"], np.ones((1, rk.NP)))
    d_cell = Pc.mean()[0]; u = R.unit(d_cell); a, T = c["a"], c["T"]; al = a * R.BLK["alpha"]; rows, po, so, keys = [], [], [], []
    for p in range(rk.NP):
        for s in range(2):
            k = rk.keyseed("r2PWR", cx.b, p, s); rows += [(rk.PIDS[p], k, 1.0), (rk.PIDS[p], k, -1.0)]; po.append(p); so.append(s); keys.append(k)
    t0 = time.time(); G = generate_rows(rk, rows, u, al, T, entropy=True, cbar=True); conts = G["conts"]
    fdv = [R.first_divergence(conts[2 * j], conts[2 * j + 1]) for j in range(len(po))]; RD = read_full(rk, [rk.PIDS[p] for p in po for _ in (0, 1)], conts, fdv)
    data = dict(name="power", a=a, T=T, nseed=2, alpha=al, n_pairs=len(po), keys=keys, prompt_of=torch.tensor(po), seed_of=torch.tensor(so), conts=conts,
                first_div=fdv, windows={w: dict(delta=RD["delta"][w].float(), ok=RD["ok"][w]) for w in WINDOWS}, xp=RD["xp"]["all"].float(),
                xm=RD["xm"]["all"].float(), cbar=dict(sum=G["cbar_sum"], cnt=G["cbar_cnt"]), entropy=dict(sum=G["ent_sum"], cnt=G["ent_cnt"]),
                nll=dict(sum=RD["nll_sum"], cnt=RD["nll_cnt"]), rep4=[rep4(x) for x in conts], capped_share=G["capped_share"],
                dropped=int((~RD["ok"]["all"]).sum()), seconds=time.time() - t0)
    R.torch_save(pt_p, dict(block=cx.b, status="ran", cell=choice, push=u.float(), d_cell=d_cell.float(), data=data))
    W1 = np.ones((1, rk.NP)); rel_c, _ = halving_rel(win["delta"], win["ok"], c["prompt_of"], rk.NP)
    Pp = Pairs(data["windows"]["all"]["delta"], data["windows"]["all"]["ok"], data["prompt_of"], W1)
    g = power_gains(Pp, u, rel_c, a, R.BLK["alpha"], cell_cbar(data, W1)); rel_p, _ = halving_rel(data["windows"]["all"]["delta"], data["windows"]["all"]["ok"], data["prompt_of"], rk.NP)
    js = dict(base, status="ran", cell=choice, a=a, T=T, n_pairs=len(po), keys="keyseed('r2PWR', b, p, s), s = 0, 1", rel_cell=rel_c, rel_power=rel_p,
              g_par=float(g["g_par"][0]), G=float(g["G"][0]), rotation=float(g["rot"][0]), decision=g["kind"], dec=float(g["dec"][0]),
              cos_d_power_u_star=R.cosf(Pp.mean()[0], _load_pt(cx.path(f"b{cx.b}_q0.pt"))["u_star"].double()), **_meta(cx))
    R.write_json(js_p, js); R.log(f"power step b{cx.b} at {choice}: decision gain {js['dec']:.3f}, rotation {js['rotation']:.2f}")
    return js


def stage_fdt(cx):
    """B-Q7: X and S of the 64 references and 64 new unsteered texts. Returns the .pt content."""
    pt_p = cx.path(f"b{cx.b}_fdt.pt")
    if os.path.exists(pt_p): return _load_pt(pt_p)
    rk = cx.open()
    if rk is None:
        R.torch_save(pt_p, dict(status="not run (round-1 output missing)", error=cx.err, block=cx.b)); return _load_pt(pt_p)
    ref = rk.ST["ref"]; NP = rk.NP; keys = [rk.keyseed("r2fdt", cx.b, p) for p in range(NP)]
    new = generate_rows(rk, [(rk.PIDS[p], keys[p], 1.0) for p in range(NP)], None, 0.0, T_FDT, cbar=False)["conts"]
    po = list(range(NP)) * 2; conts = [list(c) for c in ref["conts"]] + new; t0 = time.time()
    SG = score_grads(rk, [rk.PIDS[p] for p in po], conts, T_FDT)
    pt = dict(block=cx.b, status="ok", T=T_FDT, X=SG["X"], S=SG["S"], y=SG["y"], ended_early=SG["ended_early"], prompt_of=torch.tensor(po),
              source=["ref"] * NP + ["new"] * NP, conts=conts, keys=[None] * NP + keys, cbar_own=dict(sum=SG["cbar_sum"], cnt=SG["cbar_cnt"]),
              alpha_b=R.BLK["alpha"], cap_b=R.BLK["cap"], med_b=R.BLK["med"], NNEW=int(rk.NNEW), seconds=time.time() - t0,
              note="J_hat = Cov(x, S) / cbar_b (cbar_b from theta/b<b>_q0); lambda_lin in r2_theta_verdict.py", plan_sha256=PLAN_SHA256)
    R.torch_save(pt_p, pt); R.log(f"FDT b{cx.b}: {len(conts)} texts, {time.time() - t0:.0f}s")
    return pt


def run_all(cx):
    if cx.b != 14: stage_rsib(cx)
    stage_q0(cx)
    if cx.b == 14: stage_checks(cx)
    stage_cells(cx)
    if cx.b == 14: stage_turn2(cx)
    stage_power(cx); stage_fdt(cx)


STAGES = dict(rsib=stage_rsib, q0=stage_q0, checks=stage_checks, cells=stage_cells, turn2=stage_turn2, power=stage_power, fdt=stage_fdt, all=run_all)


# ======================================================================================================================================
# Code test (MOCK)
# ======================================================================================================================================
def selftest(root, r1_dir, r2_dir, b):
    """MOCK code test of block b: builds the MOCK round-1 fixture in r1_dir if missing (r2_rk.mock_r1: rk_loop.py MOCK=1 at blocks 14,
    8, 20, fake dn_delta.json, summaries), runs every stage of the block into r2_dir and checks them against rk_loop / r2_rk."""
    global ROT_PWR, M2_GUARD
    assert R.MOCK, "selftest needs MOCK=1"; res = []

    def check(name, cond, info=""):
        res.append(dict(test=name, ok=bool(cond), info=str(info)[:400])); print(f"[{'PASS' if cond else 'FAIL'}] {name} {str(info)[:300]}", flush=True)
    if not all(os.path.exists(os.path.join(r1_dir, s, "rk", f"b{x}", "loops.json")) for s, x in (("gpu1", 14), ("gpu2", 8), ("gpu3", 20))):
        R.mock_r1(root, r1_dir)
    cx = Ctx(root, r1_dir, r2_dir, b); t0 = time.time(); run_all(cx); rk = cx.open()          # opened lazily (resume: all outputs may exist)
    check("all stages ran", rk is not None and all(os.path.exists(cx.path(f)) for f in [f"b{b}_q0.json", f"b{b}_q0.pt", f"b{b}_cells.json", f"b{b}_cells.pt",
          f"b{b}_power.json", f"b{b}_fdt.pt"] + (["checks.json", "b14_turn2.pt"] if b == 14 else [f"b{b}_r2RSIb.json", f"b{b}_r2RSIb.pt"])), f"{time.time() - t0:.0f}s")
    q0, q0pt = stage_q0(cx); C = _load_pt(cx.path(f"b{b}_cells.pt")); NP = rk.NP; al = R.BLK["alpha"]
    # B-Q0 by definition
    loops = R.load_r1_json(r1_dir, f"rk/b{b}/loops.json", b); ev = rk.ST[f"ev_RSI_a{rk.SUF}_2"]; ok = ev["ok"]; dl = ev["delta"].double()[ok]
    N = len(dl); U = (float(dl.sum(0) @ dl.sum(0)) - float((dl ** 2).sum())) / (N * (N - 1)); cb = q0pt["cbar_ref"]; cbar = sum(cb["sum"]) / sum(cb["cnt"])
    rel2 = loops["RSI_a"][1]["rel"]; gp = float(R.unit(rk.ST[f"dir_RSI_a{rk.SUF}_2"].double()) @ dl.mean(0)) / (al * cbar * rel2)
    Gm = math.sqrt(max(U, 0)) / (al * cbar * math.sqrt(rel2)); e2 = q0["evaluations"]["RSI_a"]["2"]
    check("B-Q0: U, g_par, G of RSI_a evaluation n = 2 by the plan's formulas", abs(e2["U"]["pt"] - U) < 1e-9 * max(1, abs(U)) and abs(e2["g_par"]["pt"] - gp) < 1e-9 * max(1, abs(gp))
          and abs(e2["G"]["pt"] - Gm) < 1e-9 * max(1, Gm), (e2["g_par"]["pt"], gp, e2["G"]["pt"], Gm))
    kr = R.cbar_read([rk.PIDS[p] for p in range(NP)], rk.ST["ref"]["conts"])
    check("B-Q0: kappa_b = cbar read of the references / med_b; cbar_b = kappa_b med_b", abs(q0["kappa_b"]["pt"] - kr["pooled"] / R.BLK["med"]) < 1e-9 and abs(q0["cbar_b"]["pt"] - cbar) < 1e-9)
    M = q0["mode_set"]["members"]; d4 = {L: rk.ST[f"dir_{L}{rk.SUF}_4"].double() for L in M}
    us = R.unit(sum(q0["mode_set"]["signs"][L] * d4[L] for L in M))
    check("B-Q0: u*_b = unit(sum of sign-aligned d_4 over M_b); cfg follows the guard / switch", float((us - q0pt["u_star"].double()).norm()) < 1e-5
          and q0["config"]["mode"] == ("not_resolved" if not q0["m_star2"]["guard_pass"] else ("switch" if q0["cross_fit"].get("switch") else "standard")),
          (M, q0["config"]["mode"], q0["m_star2"]))
    if b != 14:
        rs = json.load(open(cx.path(f"b{b}_r2RSIb.json"))); L = RSIB + rk.SUF; e0 = rk.ST[f"ev_{L}_0"]
        E = R.evaluate_variant(R.unit(rk.ST["axes"]["u_T_perp"].double()), al, lambda p, s: rk.keyseed(L, 0, p, s, b), nseed=2, cbar=False)
        check("rsib: 4 run_loop rows, keys keyseed('r2RSIb' + SUF, n, p, s, b), 128-pair layout (2 seeds), in L_B(b)", rs["status"] == "ran" and len(rs["rows"]) == 4
              and E["conts"] == e0["conts"] and len(e0["prompt_of"]) == 2 * NP and RSIB in q0["loops"])
        PT = _load_pt(cx.path(f"b{b}_r2RSIb.pt")); fp = json.load(open(os.path.join(r2_dir, "rk", f"b{b}", "r1_fingerprint.json")))
        new = sorted(set(torch.load(R.BLK["state"], map_location="cpu", weights_only=False)) - set(fp["keys"]))
        check("rsib: exactly the r2RSIb keys were added to the round-2 state and exported", new == sorted(PT["entries"]) and all(k.startswith(("ev_r2RSIb", "dir_r2RSIb", "dmg_r2RSIb")) for k in new), new)
    else:
        ck = json.load(open(cx.path("checks.json")))
        check("checks: zero 4/4, mirror 4/4, entropy T 0.3 < T 0.7 (G20)", ck["passed"] and ck["zero_push_identical"] == 4 and ck["mirror_swapped"] == 4, (ck["zero_push_identical"], ck["mirror_swapped"], ck["entropy_T03"], ck["entropy_T07"]))
    # cells
    specs = cells_of(b); u_push = C["u_push"].double()
    check("cells: every cell with its pair count", sorted(C["cells"]) == sorted(s["name"] for s in specs) and all(C["cells"][s["name"]]["n_pairs"] == NP * s["nseed"] for s in specs),
          {n: c["n_pairs"] for n, c in C["cells"].items()})
    nm = cell_name(1, 0.3); c = C["cells"][nm]
    E = R.evaluate_variant(u_push, al, lambda p, s: rk.keyseed("theta", b, p, s), nseed=2, T=0.3, entropy=True)
    check("cells: keys keyseed('theta', b, p, s), push a alpha_b u_push at T reproduce evaluate_variant (texts, cbar, entropy)", E["conts"] == c["conts"]
          and abs(E["cbar"]["pooled"] - sum(c["cbar"]["sum"]) / sum(c["cbar"]["cnt"])) < 1e-9 * R.BLK["cap"]
          and abs(E["entropy"]["pooled"] - sum(c["entropy"]["sum"]) / sum(c["entropy"]["cnt"])) < 1e-9)
    e3 = max(float((E["windows"][w]["delta"].double() - c["windows"][w]["delta"].double()).nan_to_num().abs().max()) for w in WINDOWS)
    check("cells: the full-pass windows equal r2_rk.read_windows (all, div, first20)", e3 < 1e-5 * float(c["windows"]["all"]["delta"].abs().max()) + 1e-7
          and all(torch.equal(E["windows"][w]["ok"], c["windows"][w]["ok"]) for w in WINDOWS), f"{e3:.2e}")
    j = next(i for i, x in enumerate(c["conts"]) if len(x) > 1); p = int(c["prompt_of"][j // 2]); seq = rk.PIDS[p] + c["conts"][j]
    with torch.no_grad(): lg = rk.model(torch.tensor([seq]).to(rk.DEV)).logits[0].double()
    P0 = len(rk.PIDS[p]); nl = -float(torch.log_softmax(lg[P0 - 1:len(seq) - 1], -1)[torch.arange(len(c["conts"][j])), torch.tensor(c["conts"][j])].sum())
    check("cells: NLL of a steered text under the unsteered model equals a direct forward", abs(nl - c["nll"]["sum"][j]) < 1e-3 * max(1, abs(nl)) and c["nll"]["cnt"][j] == len(c["conts"][j]), (nl, c["nll"]["sum"][j]))
    check("cells: damage curve = rk.damage, KL at T = damage_T", abs(C["damage_curve"][2.0] - rk.damage(u_push, 2 * al)) < 1e-12 and abs(c["KL_T"] - R.damage_T(u_push, al, 0.3)) < 1e-12
          and c["damage"] == C["damage_curve"][1.0])
    check("cells: repeated-4-gram share by definition", rep4([1, 2, 3, 4, 1, 2, 3, 4]) == 1 / 5 and rep4([1, 2, 3]) != rep4([1, 2, 3]) and rep4([5, 6, 7, 8]) == 0.0)
    cfg = _q0_cfg(cx)[1]; W = boot_W(NP); TT = cell_table({nm: c}, cfg, al, W)[nm]["all"]; Pw = Pairs(c["windows"]["all"]["delta"], c["windows"]["all"]["ok"], c["prompt_of"], W)
    dd = c["windows"]["all"]["delta"].double()[c["windows"]["all"]["ok"]]; cbc = sum(c["cbar"]["sum"]) / sum(c["cbar"]["cnt"])
    gp = float(cfg["u_read"] @ dd.mean(0)) / (al * cbc * float(cfg["m2_par"][0]))
    check("cells: g_par of a cell by B-Q1's formula (row 0 of the joint bootstrap)", abs(float(TT["g_par"][0]) - gp) < 1e-9 * max(1, abs(gp)), (float(TT["g_par"][0]), gp))
    w0 = torch.as_tensor(W[1])[c["prompt_of"].long()] * c["windows"]["all"]["ok"].double(); dm = (w0[:, None] * c["windows"]["all"]["delta"].double().nan_to_num()).sum(0) / w0.sum()
    check("bootstrap: a replicate row equals the weighted resample mean", abs(float(Pw.proj(cfg["u_read"])[1]) - float(R.unit(cfg["u_read"]) @ dm)) < 1e-10)
    # power, turn 2, fdt
    pw = json.load(open(cx.path(f"b{b}_power.json"))); pts = cell_points(C["cells"], cfg, al, NP); ch = choose_power_cell(pts, C["order"])
    check("power: the in-session choice is choose_power_cell of the point estimates (OD-B7, G24)", (pw["status"] == "not needed" and ch is None) or (pw["status"] == "ran" and pw["cell"] == ch), (pw["status"], ch))
    tp = {"x": (1.5, 0.5), "y": (0.95, 0.7), "z": (1.02, 0.95), "w": (-0.2, 0.1)}
    check("power: chooser (rotation < 0.8, smallest |log gain|, non-positive gains last)", choose_power_cell(tp) == "y" and choose_power_cell({"z": (1.0, 0.9)}) is None
          and choose_power_cell({"w": (-0.2, 0.1), "x": (3.0, 0.2)}) == "x")
    if pw["status"] == "ran":
        PP = _load_pt(cx.path(f"b{b}_power.pt")); dcell = Pairs(C["cells"][ch]["windows"]["all"]["delta"], C["cells"][ch]["windows"]["all"]["ok"], C["cells"][ch]["prompt_of"], np.ones((1, NP))).mean()[0]
        E2 = R.evaluate_variant(R.unit(dcell), C["cells"][ch]["a"] * al, lambda p, s: rk.keyseed("r2PWR", b, p, s), nseed=2, T=C["cells"][ch]["T"], cbar=False)
        check("power: push unit(read-back d) at the cell's (a, T), keys keyseed('r2PWR', b, p, s), 128-pair layout", E2["conts"] == PP["data"]["conts"] and PP["data"]["n_pairs"] == 2 * NP)
    if b == 14:
        T2 = _load_pt(cx.path("b14_turn2.pt")); t = T2["cells"][cell_name(1, 0.7)]; c7 = C["cells"][cell_name(1, 0.7)]
        tmpl = lambda msgs: rk.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        suf = STIM.check_turn2(rk.enc, rk.chat, tmpl); p0, s0 = int(t["prompt_of"][0]), int(t["seed_of"][0])
        g = generate_rows(rk, [(rk.PIDS[p0] + c7["conts"][0] + suf, rk.keyseed("theta2", 14, 1.0, 0.7, p0, s0), 1.0),
                               (rk.PIDS[p0] + c7["conts"][1] + suf, rk.keyseed("theta2", 14, 1.0, 0.7, p0, s0), 1.0)], None, 0.0, 0.7, cbar=False)
        check("turn2: ids PIDS + turn-1 ids + suffix, keys keyseed('theta2', b, a, T, p, s), no push, cell T", g["conts"] == t["conts2"][:2] and T2["suffix_ids"] == suf
              and sorted(T2["cells"]) == sorted(TURN2_CELLS) and t["input_len"][0] == len(rk.PIDS[p0]) + len(c7["conts"][0]) + len(suf))
        X2 = rk.read([rk.PIDS[p0]] * 2, t["conts2"][:2]).double()
        check("turn2: turn 2 read as the reply to the plain prompt", float(((X2[0] - X2[1]) / 2 - t["delta2"][0].double()).nan_to_num().abs().max()) < 1e-5 * float(X2.nan_to_num().abs().max()) + 1e-7)
    F = _load_pt(cx.path(f"b{b}_fdt.pt")); okx = torch.isfinite(F["X"]).all(1)
    Xr = rk.read([rk.PIDS[p] for p in range(NP)], rk.ST["ref"]["conts"]).double()
    e4 = float((F["X"][:NP] - Xr).nan_to_num().abs().max() / Xr.nan_to_num().abs().max())
    newk = F["keys"][NP:] == [rk.keyseed("r2fdt", b, p) for p in range(NP)]
    check("fdt: x_j = the block-b reading (rk.read of the references); 64 + 64 texts; keys keyseed('r2fdt', b, p)", e4 < 1e-5 and len(F["conts"]) == 2 * NP and newk, f"{e4:.1e}")
    i = int(np.nonzero(okx.numpy())[0][-1]); gen = torch.Generator().manual_seed(7); e = R.unit(torch.randn(rk.D, generator=gen).double()); eps = 1e-3

    def y_at(sc):
        seq = rk.PIDS[F["prompt_of"][i]] + F["conts"][i]; L = len(F["conts"][i]); ids = torch.tensor([seq]).to(rk.DEV)
        rk.STEER.update(V=(sc * e).float()[None].to(rk.DEV), v=None, cap=rk.ST["ref"]["cap"], rowstart=torch.tensor([rk.START]))
        try:
            with torch.no_grad(): lg = rk.model(ids).logits[0].double()
        finally: rk.steer_off()
        lp = torch.log_softmax(lg[len(seq) - L - 1:] / T_FDT, -1); y = float(lp[torch.arange(L), torch.tensor(F["conts"][i])].sum())
        return y + (float(torch.logsumexp(lp[L, sorted(int(t) for t in rk.STOP)], 0)) if L < rk.NNEW else 0.0)
    fdv = (y_at(eps) - y_at(-eps)) / (2 * eps); an = float(F["S"][i] @ e)
    check("fdt: S_j = dy / dV equals the finite difference of y along a random direction; y as defined", abs(fdv - an) < 2e-2 * max(abs(an), 1e-3) + 1e-4
          and abs(y_at(0.0) - F["y"][i]) < 1e-3 * max(1, abs(F["y"][i])), (fdv, an))
    # guard and resume
    R.guard_check(); mt = {f: os.path.getmtime(f) for f in glob.glob(cx.path(f"b{b}_*")) + glob.glob(cx.path("checks.json"))}; run_all(cx)
    check("resume: every output kept; no parts left; round-1 keys unchanged (state guard)", all(os.path.getmtime(f) == t for f, t in mt.items())
          and not glob.glob(cx.part("*")))
    # the power step's 'ran' path, forced (the MOCK cells need not have rotation < 0.8); the forced outputs are kept in theta_forced/ for
    # the end-to-end test of r2_theta_verdict.py and the session's own outputs are restored
    pj, pp = cx.path(f"b{b}_power.json"), cx.path(f"b{b}_power.pt"); keep = {f: open(f, "rb").read() for f in (pj, pp) if os.path.exists(f)}
    for f in keep: os.remove(f)
    old = ROT_PWR; ROT_PWR = 1e9
    try:
        pw2 = stage_power(cx); ch2 = choose_power_cell(cell_points(C["cells"], cfg, al, NP), C["order"])
    finally:
        ROT_PWR = old
    PP = _load_pt(pp); c2 = C["cells"][ch2]; dcell = Pairs(c2["windows"]["all"]["delta"], c2["windows"]["all"]["ok"], c2["prompt_of"], np.ones((1, NP))).mean()[0]
    E2 = R.evaluate_variant(R.unit(dcell), c2["a"] * al, lambda p, s: rk.keyseed("r2PWR", b, p, s), nseed=2, T=c2["T"], cbar=False)
    check("power (forced): push unit(read-back d) at the chosen cell's (a, T), keys keyseed('r2PWR', b, p, s), 128-pair layout; gains by B-Q0's formulas",
          pw2["status"] == "ran" and pw2["cell"] == ch2 and E2["conts"] == PP["data"]["conts"] and PP["data"]["n_pairs"] == 2 * NP and fin(pw2["dec"]), (pw2["cell"], pw2.get("dec")))
    fd = os.path.join(r2_dir, "theta_forced"); os.makedirs(fd, exist_ok=True)
    for f in (pj, pp): shutil.copyfile(f, os.path.join(fd, os.path.basename(f))); os.remove(f)
    for f, data in keep.items():
        with open(f, "wb") as fh: fh.write(data)
    # code-path variant theta_std/ (test only): guard threshold 0 and power-step threshold +inf, so that the cells run in the standard or
    # switch configuration and the power step runs; r2_theta_verdict.py's selftest analyses it end to end (its OD-G8 check must then
    # report the guard disagreement, since the analysis uses the frozen 0.5)
    old = (M2_GUARD, ROT_PWR); M2_GUARD, ROT_PWR = 0.0, 1e9
    try:
        cx2 = _AltCtx(cx, "theta_std"); run_all(cx2); q2, _ = stage_q0(cx2)
    finally:
        M2_GUARD, ROT_PWR = old
    pw3 = json.load(open(cx2.path(f"b{b}_power.json")))
    check("variant theta_std: guard passed, cells along u*_b (or u*_A / u*_B), power step ran", q2["m_star2"]["guard_pass"] and q2["config"]["mode"] in ("standard", "switch")
          and pw3["status"] == "ran" and os.path.exists(cx2.path(f"b{b}_cells.pt")), (q2["config"]["mode"], pw3["status"]))
    if b != 14:                                                 # G49: an r2RSIb loop that fails twice is recorded, B-Q0 runs without it
        cx3 = _AltCtx(cx, "theta_rsibfail"); shutil.rmtree(os.path.join(r2_dir, "theta_rsibfail"), ignore_errors=True); orig = rk.run_loop

        def boom(*a, **k): raise RuntimeError("simulated failure of the r2RSIb loop")
        rk.run_loop = boom
        try:
            try: stage_rsib(cx3); first = "no exception"
            except RuntimeError: first = "raised"
            second = stage_rsib(cx3)
        finally:
            rk.run_loop = orig
        q3, _ = stage_q0(cx3)
        check("G49: the first failure of r2RSIb raises (the runner retries), the second is recorded; B-Q0 then runs with 'no independent loop', no cross-fit",
              first == "raised" and second["status"].startswith("not run (r2RSIb failed") and q3["status"] == "ok" and "no independent loop" in q3["labels"]
              and RSIB not in q3["loops"] and not q3["cross_fit"]["computed"], (first, second.get("status"), q3.get("labels")))
        # review finding theta-1: B-Q0 takes the r2RSIb loop from its exported entries (theta/b<b>_r2RSIb.pt), so a round-2 state copy without
        # them (re-copied from round 1) gives the same B-Q0; a loop that 'ran' but whose exported entries are missing fails the step
        L = RSIB + rk.SUF; d5, d6 = os.path.join(r2_dir, "theta_rsibstale"), os.path.join(r2_dir, "theta_rsibbad")
        for d in (d5, d6): shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
        for f in (f"b{b}_r2RSIb.json", f"b{b}_r2RSIb.pt"): shutil.copyfile(cx.path(f), os.path.join(d5, f))
        shutil.copyfile(cx.path(f"b{b}_r2RSIb.json"), os.path.join(d6, f"b{b}_r2RSIb.json"))
        Pb = _load_pt(cx.path(f"b{b}_r2RSIb.pt")); Pb["entries"] = {}; R.torch_save(os.path.join(d6, f"b{b}_r2RSIb.pt"), Pb)
        held = {k: rk.ST.pop(k) for k in [k for k in rk.ST if k.startswith((f"ev_{L}_", f"dir_{L}_", f"dmg_{L}_"))]}    # in memory only, never saved
        try:
            q5, _ = stage_q0(_AltCtx(cx, "theta_rsibstale"))
            try: stage_q0(_AltCtx(cx, "theta_rsibbad")); raised = False
            except RuntimeError: raised = True
        finally:
            rk.ST.update(held)
        qm = json.load(open(cx.path(f"b{b}_q0.json")))
        check("theta-1: B-Q0 uses r2RSIb's exported entries (same B-Q0 when the round-2 state lacks them); a 'ran' loop without entries raises, no q0 written",
              len(held) == 13 and RSIB in q5["loops"] and "no independent loop" not in q5["labels"] and q5["independent_loop"]["present"]
              and q5["mode_set"]["members"] == qm["mode_set"]["members"] and q5["m_star2"]["pt"] == qm["m_star2"]["pt"] and q5["Gamma"]["pt"] == qm["Gamma"]["pt"]
              and q5["config"]["mode"] == qm["config"]["mode"] and json.dumps(R.jsonable(q5["cross_fit"]), sort_keys=True) == json.dumps(qm["cross_fit"], sort_keys=True) and raised
              and not os.path.exists(os.path.join(d6, f"b{b}_q0.json")), (len(held), q5.get("labels"), raised))
    ok = all(r["ok"] for r in res); R.write_json(os.path.join(r2_dir, f"selftest_theta_b{b}.json"), dict(ok=ok, tests=res, block=b))
    print(f"r2_theta selftest (block {b}): {sum(r['ok'] for r in res)}/{len(res)} passed", flush=True); return ok


def main(argv):
    if len(argv) < 4: raise SystemExit(__doc__)
    root, r1_dir, r2_dir = argv[1:4]; stage = argv[4] if len(argv) > 4 else "all"
    if "BLOCK" not in os.environ: raise SystemExit("set env BLOCK (14, b_lo or b_hi)")
    b = int(os.environ["BLOCK"])
    if os.environ.get("DEVICE", "cpu") != "cpu": torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if stage == "selftest": return 0 if selftest(root, r1_dir, r2_dir, b) else 1
    if stage not in STAGES: raise SystemExit(f"unknown stage {stage}; stages: {', '.join(STAGES)} | selftest")
    if stage in ("checks", "turn2") and b != 14: raise SystemExit(f"stage {stage} runs at block 14 only")
    if stage == "rsib" and b == 14: raise SystemExit("stage rsib runs at the extra blocks only")
    STAGES[stage](Ctx(root, r1_dir, r2_dir, b)); return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
