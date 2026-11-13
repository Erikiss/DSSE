"""Round 2, session r2_ana: the Part A verdicts (frozen plan results/ju_space/plans/abschnitt11_12_runde2_plan.md, sha256
5e476340...edc2d; section 3: A6 readouts, A3 u_T_perp spillover, A7 per block / over blocks / RSI-into-RGI / causal / dynamics /
combined, AT5 slope labels, AT7 with r2_sim.reg2_simulate on the measured noise, AT8; gates G12, G28, G30-G33, G36, G38-G40) ->
rgi_verdict.json.

    python r2_rgi_verdict.py <r1_dir> <r2_root> <out.json>
    python r2_rgi_verdict.py selftest <workdir>               synthetic inputs in the producers' formats, every branch
    env MOCK / QUICK (code tests only: R2_AT7_NSIM, R2_AT5_BOOT, R2_A6_NBOOT may lower the simulation and bootstrap counts; a real run
    always uses the plan's 1,500 simulations, 10,000 cluster draws and 2,000 prompt replicates)

Inputs (found by name, the expected session first; r2_report.Inputs): r2_q1 axes/r2_axes.pt (O2 axes and nulls per block of B3),
rgi/dyn_b<b>.json (GEN loops); r2_q3 rgi/geom_stats_<m>.json (A2-A5 statistics with pointwise intervals, P and sup-t bands),
rgi/geom_draws_qwen3-1.7b.npz (per-draw series: the simultaneous intervals of the causal line), rgi/causal_qwen3-1.7b.pt (A3 gradients),
rgi/dirs_qwen3-1.7b.pt (g_b, s_b, c_o); r2_t1 rgi_toy/{r0.json, r1/w<ws>.json, r1/at3.json,
fam.json, cross/w<ws>.json, forcing.json}; r2_pc pc_toy/cross_pc/w<ws>.json; round 1: rk/b<b>/{state.pt, loops.json}, summary_b<b>.json,
dn_delta.json, blocks_chosen.json (O3 bank, G-blk, delta order), pc_summary files (G-PC for the AT5 pairs with a PC net).
A6 (a): for every d_n (n = 1..4) of every frozen loop at every block of B3: c, p, Q (r2_rk.stats_on_axis = rk.stats's formulas, each axis
against its own null axes) and A = c / sqrt(rel(d_n)), rel(d_n) = R1.loops[b][L][n-1]['rel'], on u_RGI_perpT and u_RGI_perp (O2), and
descriptively on u_dir, g_b and s_b (sign-flip nulls over the 48 model-object contrasts); 90 % prompt-bootstrap intervals of A from the
pairs behind d_n (2,000 replicates, one draw of prompt weights shared by all blocks and loops, rel held at its round-1 value).
A3: s = w . unit(u_T_perp(b)) at the blocks of B3 with z against R1 null_u_T_perp (r2_rgi_stats.spillover).
A7: the frozen rules on r2_rgi_stats.py's numbers, in the order of 2.6 (C0 and 'uncomposed' pointwise -> P -> sup-t bands over P).
AT8: validity per world, machinery gate, the two-step sham-controlled rule (r2_sim.reg2_verdict, the function AT7 simulates), answer
mapping, recursion, family; AT7: r2_sim.reg2_simulate with the noise components of r2_sim.reg2_noise per line and arm.
Labels: G-desc on every verdict at block 14; 'RGI axis not resolved' where GA(b) fails; parts whose inputs are missing are 'not run'."""
import math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); JU = os.path.dirname(HERE)
for _p in (HERE, JU):
    if _p not in sys.path: sys.path.insert(0, _p)
import r2_report as RP
from r2_report import get, num, fin, ge, le, gt, lt, tci

TEST = os.environ.get("MOCK") == "1" or os.environ.get("QUICK") == "1"
M17, M06 = "qwen3-1.7b", "qwen3-0.6b"
NOT_RUN_R1 = RP.NOT_RUN_R1
SEEDS = dict(prompt_boot=4301, dirs_null=1107, at5_cluster_boot=0, at7_sim=1)       # OD-G6
N_BOOT = 2000 if not TEST else int(os.environ.get("R2_A6_NBOOT", "2000"))
AT7_NSIM = 1500 if not TEST else int(os.environ.get("R2_AT7_NSIM", "1500"))
AT5_BOOT = 10000 if not TEST else int(os.environ.get("R2_AT5_BOOT", "10000"))
LEVEL = 0.90
# frozen thresholds (A7, AT8)
P_SIG, A_BIG, P_NS, DMG_MAX = 0.01, 0.25, 0.10, 0.15
GI_FG, GI_LO, IS_DFG, BA_FG, BA_VAL, LBL = 0.5, 0.35, 0.10, 0.5, 0.8, 0.5
ABS_RHO, ABS_SIG, D_MIN, SPEAR, SPEAR_FLAT, FLAT_D, P_SURR, MIN_P = 0.5, 0.15, 0.10, 0.5, 0.3, 0.10, 0.05, 12
Z_SPILL, FR_HI, FR_LO = 3.0, 0.5, 0.25
E_GI_MIN, REL_GEN, R4_MIN = 0.7, 0.5, 0.5
LINES = (("Lambda", "Lambda"), ("S", "S_gain"), ("T1", "T1_gain"), ("Phi", "Phi"))
PRESENT_MEAN, PRESENT_LO, ABSENT_HI, SPAN_MIN, VALID_MIN, RHO_MIN, FAM_LO = 0.10, 0.03, 0.05, 8.0, 6, 0.9, 0.10


def log(msg): print(f"[r2_rgi_verdict {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def inside(ci, a, b):
    """Interval ci = [lo, hi] strictly inside (a, b)."""
    return bool(ci is not None and len(ci) == 2 and gt(ci[0], a) and lt(ci[1], b))


# ======================================================================================================================================
# A6 (a) readouts of every frozen loop; A6 (b) GEN loop; A7 dynamics
# ======================================================================================================================================
def sign_flip_nulls(c_o, self_objects, n=2000, seed=SEEDS["dirs_null"]):
    """Null axes for g_b and s_b: sign flips of the 48 model-object contrasts c_o [48, D], each null built like its axis (g' = mean of
    eps c_o; s' = the SELF mean of eps c_o minus its projection on unit(g')). r2_rgi_stats.sign_flip_nulls when importable (the same
    formula), else this copy."""
    import torch
    try:
        import r2_rgi_stats as RS
        if list(RS.SELF_OBJ) == [int(x) for x in self_objects]: return RS.sign_flip_nulls(c_o, n=n, seed=seed), "r2_rgi_stats.sign_flip_nulls"
    except Exception:                                                             # noqa: BLE001 - fall back to the local copy
        pass
    c = torch.as_tensor(c_o).double(); g = torch.Generator().manual_seed(int(seed)); E = (torch.randint(0, 2, (n, c.shape[0]), generator=g) * 2 - 1).double()
    gn = E @ c / c.shape[0]; sm = torch.zeros(c.shape[0], dtype=torch.float64); sm[[int(x) for x in self_objects]] = 1.0 / len(self_objects); ds = (E * sm) @ c
    u = gn / gn.norm(dim=1, keepdim=True).clamp_min(1e-300)
    return dict(g=gn, s=ds - (ds * u).sum(1, keepdim=True) * u), "local copy of r2_rgi_stats.sign_flip_nulls"


def readouts_block(b, bank, axes, W, NP):
    """Readouts of block b. bank: O3 bank; axes: {name: (axis [D], nulls [K, D], with_interval)}; W [B, NP] prompt weights.
    Returns ({L: {n: {axis: dict(c, p, Q, A, A_ci90), rel, damage_of_push, u_T_perp}}}, {(L, n, axis): A draws [B]})."""
    import torch
    import r2_rk as R
    out, draws = {}, {}
    for L in R.loops_of_block(b):
        if L not in bank: continue
        out[L] = {}
        for n in range(1, 5):
            ev = bank[L]["ev"][n]; rel = bank[L]["rel"][n]; row = bank[L]["row"][n]; d = bank[L]["d"][n]
            okm = torch.as_tensor(ev["ok"], dtype=torch.bool); dd = ev["delta"].double()[okm].mean(0)             # the mean pair delta behind d_n
            e = dict(rel=rel, u_T_perp={k: row.get(k) for k in ("c", "p", "Q", "A")}, n_pairs=int(okm.sum()))
            sq = math.sqrt(rel) if (fin(rel) and rel > 0) else float("nan"); Db = None
            for name, (axis, null, with_ci) in axes.items():
                s, _ = R.stats_on_axis(ev, axis, null, n_prompts=NP)
                r = dict(c=s["c"], p=s["p"], Q=s["Q"], A=s["c"] / sq if fin(sq) else float("nan"))
                if with_ci and W is not None:
                    if Db is None: Db = R.boot_mean(ev["delta"], ev["prompt_of"], W, ev["ok"])
                    a = torch.as_tensor(axis).double(); cb = (Db @ a) / (Db.norm(dim=1) * a.norm() + 1e-12); Ab = (cb / sq).numpy() if fin(sq) else np.full(len(cb), np.nan)
                    r["A_ci90"] = RP.interval(Ab, LEVEL); draws[(L, n, name)] = Ab
                e[name] = r
            e["cos_stored_d_n"] = R.cosf(dd, d.double()) if float(dd.norm()) > 0 else float("nan")
            out[L][n] = e
        # OD-A5: damage_of_push of rows n = 1..3 of the loop (the pushes d_1 .. d_3) = the rows behind d_2, d_3, d_4
        dm = [num(bank[L]["row"][n].get("damage_of_push")) for n in (2, 3, 4)]
        out[L]["damage_d1_d3"] = dm; out[L]["damage_ok"] = bool(all(fin(x) and x <= DMG_MAX for x in dm))
    return out, draws


def hit(r, sign=0):
    """p <= 0.01 with |A| >= 0.25 (sign 0), or with c > 0 and A >= 0.25 (sign +1), or c < 0 and A <= -0.25 (sign -1)."""
    if r is None or not le(r.get("p"), P_SIG): return False
    if sign > 0: return bool(gt(r.get("c"), 0) and ge(r.get("A"), A_BIG))
    if sign < 0: return bool(lt(r.get("c"), 0) and le(r.get("A"), -A_BIG))
    return bool(fin(num(r.get("A"))) and abs(num(r["A"])) >= A_BIG)


def self_loop_rule(rd, b, axis="u_RGI_perpT"):
    """A7 dynamics, 'self loop drawn to general' at block b: RSI_a (and RSI_b at block 14) with c > 0, p <= 0.01 and A >= 0.25 at d_3 and
    d_4 on u_RGI_perpT while SPO (block 14) and RND_1 (and RND_2 at block 14) do not reach p <= 0.01 with |A| >= 0.25 at d_3 or d_4
    (otherwise 'generic attractor'); the mirror is 'drawn away from general improvement'; otherwise 'no pull'."""
    rsi = ["RSI_a"] + (["RSI_b"] if int(b) == 14 else []); ctrl = (["SPO"] if int(b) == 14 else []) + ["RND_1"] + (["RND_2"] if int(b) == 14 else [])
    miss = [L for L in rsi + ctrl if L not in rd]
    if miss: return dict(verdict=f"not decidable (loops missing: {', '.join(miss)})", loops=rsi + ctrl)
    bad = [L for L in rsi + ctrl if not rd[L].get("damage_ok")]
    if bad: return dict(verdict=f"not decidable (damage of a counted push above {DMG_MAX} nats: {', '.join(bad)})", loops=rsi + ctrl)
    g = lambda L, n: get(rd, L, n, axis)
    pos = all(hit(g(L, n), +1) for L in rsi for n in (3, 4)); neg = all(hit(g(L, n), -1) for L in rsi for n in (3, 4))
    chit = [f"{L} d_{n}" for L in ctrl for n in (3, 4) if hit(g(L, n), 0)]
    if pos or neg: v = "generic attractor" if chit else ("self loop drawn to general" if pos else "drawn away from general improvement")
    else: v = "no pull"
    return dict(verdict=v, loops=rsi + ctrl, control_hits=chit, rsi_rows={f"{L} d_{n}": g(L, n) for L in rsi for n in (3, 4)})


def gen_rule(dyn):
    """A7 dynamics, GEN loop: 'RGI sustains itself' = on u_RGI_perp c > 0 with p <= 0.01 at d_2, d_3, d_4, mean rel(d_2..d_4) >= 0.5 and
    E_GI >= 0.7; 'RGI collapses into RSI' = on u_RGI_perpT p > 0.10 at d_3 and d_4 while c > 0 with p <= 0.01 on u_T_perp there; 'common
    attractor' = R_4(RSI_a, GEN) >= 0.5; label 'no first link' if d_1 has p > 0.01 on u_RGI_perp."""
    rows = {int(r["d"]): r for r in (dyn.get("rows") or []) if isinstance(r, dict) and "d" in r}
    if any(d not in rows for d in (1, 2, 3, 4)): return dict(verdict="not decidable (GEN rows missing)")
    dm = [num(rows[d].get("damage_of_push")) for d in (2, 3, 4)]                      # rows n = 1..3 (pushes d_1 .. d_3)
    if not all(fin(x) and x <= DMG_MAX for x in dm): return dict(verdict=f"not decidable (damage of a counted push above {DMG_MAX} nats)", damage_d1_d3=dm)
    gp = lambda d: rows[d].get("u_RGI_perp") or {}; gt_ = lambda d: rows[d].get("u_RGI_perpT") or {}; ut = lambda d: rows[d].get("u_T_perp") or {}
    rels = [num(ut(d).get("rel")) for d in (2, 3, 4)]; mrel = float(np.mean(rels)) if all(fin(x) for x in rels) else float("nan")
    sustains = bool(all(gt(gp(d).get("c"), 0) and le(gp(d).get("p"), P_SIG) for d in (2, 3, 4)) and ge(mrel, REL_GEN) and ge(dyn.get("E_GI"), E_GI_MIN))
    collapses = bool(all(gt(gt_(d).get("p"), P_NS) for d in (3, 4)) and all(gt(ut(d).get("c"), 0) and le(ut(d).get("p"), P_SIG) for d in (3, 4)))
    common = ge(dyn.get("R4_RSI_a_GEN"), R4_MIN); nofirst = gt(gp(1).get("p"), P_SIG)
    st = [s for s, f in (("RGI sustains itself", sustains), ("RGI collapses into RSI", collapses), ("common attractor", common)) if f]
    return dict(verdict="; ".join(st) if st else "none of the frozen statements holds", sustains=sustains, collapses_into_RSI=collapses, common_attractor=common,
                labels=["no first link"] if nofirst else [], mean_rel_d2_d4=mrel, E_GI=dyn.get("E_GI"), R4=dyn.get("R4_RSI_a_GEN"), S4=dyn.get("S4"), damage_d1_d3=dm)


def monotone(order, X, draws, up, down, level=LEVEL):
    """The paired three-block form (2.7 C4; A7 'pull and delta'): ends = the lowest- and highest-delta entering blocks; '<up>' if the paired
    interval of X(high) - X(low) lies above 0 and X(middle) is not above the upper end of X(high)'s interval; '<down>' mirrored (below 0,
    middle not above the upper end of X(low)'s interval); with two entering blocks the middle clause is dropped (marked); fewer: 'not
    decidable'. order: blocks by delta (ascending); X {b: point}; draws {b: [B] paired draws}."""
    e = [b for b in order if b in X and fin(num(X[b]))]
    out = dict(entering=e, X={str(b): X[b] for b in e}, ci90={str(b): RP.interval(draws[b], level) for b in e})
    if len(e) < 2: out["verdict"] = "not decidable"; return out
    low, high = e[0], e[-1]; diff = np.asarray(draws[high], float) - np.asarray(draws[low], float); ci = RP.interval(diff, level)
    mid = e[1] if len(e) == 3 else None
    okm_up = True if mid is None else le(X[mid], out["ci90"][str(high)][1]); okm_dn = True if mid is None else le(X[mid], out["ci90"][str(low)][1])
    if gt(ci[0], 0) and okm_up: v = up
    elif lt(ci[1], 0) and okm_dn: v = down
    else: v = "no monotone relation"
    lo_i, hi_i = out["ci90"][str(low)], out["ci90"][str(high)]
    out.update(verdict=v, low=low, high=high, middle=mid, diff_high_minus_low=float(num(X[high]) - num(X[low])), diff_ci90=ci,
               middle_clause="dropped (two entering blocks)" if mid is None else "applied",
               non_overlap_form=dict(separated=bool(gt(hi_i[0], lo_i[1]) or lt(hi_i[1], lo_i[0])), note="non-overlap of the two 90 % intervals (round-1 section 6), reported only"),
               note="three blocks never separate delta from depth")
    return out


def a6(r1_dir, inp, info):
    """A6 readouts, GEN loops and the A7 dynamics rules."""
    import torch
    import r2_rk as R
    AX = inp.pt("axes/r2_axes.pt"); dirs = inp.pt(f"rgi/dirs_{M17}.pt")
    out = dict(blocks={}, gen={}, dynamics={}, settings=dict(n_boot=N_BOOT, level=LEVEL, seed=SEEDS["prompt_boot"], dirs_null_seed=SEEDS["dirs_null"]))
    if AX is None: out["status_axes"] = "not run (round-2 output missing: axes/r2_axes.pt)"
    if dirs is None: out["status_dirs"] = f"not run (round-2 output missing: rgi/dirs_{M17}.pt): no g_b / s_b readouts"
    W = None; Xd, Xp = {}, {}; entering = []
    for b in info["B3"]:
        lab = RP.desc_labels(info, b); gblk = get(info, "G_blk", str(b), "ok")
        dyn = inp.json(f"rgi/dyn_b{b}.json"); GA = None
        if not get(info, "final", str(b)):
            out["blocks"][str(b)] = dict(status=NOT_RUN_R1); out["gen"][str(b)] = dict(status=RP.status_of(dyn, NOT_RUN_R1)); continue
        try: bank, ST, _ = RP.load_bank(r1_dir, b)
        except Exception as e:                                                    # noqa: BLE001
            out["blocks"][str(b)] = dict(status=f"{NOT_RUN_R1}: {e!r}"); continue
        NP = len(ST["axes"]["fold"]); A = get(AX, "blocks", b) if AX is not None else None
        if W is None or W.shape[1] != NP: W = R.prompt_bootstrap(NP, B=N_BOOT, seed=SEEDS["prompt_boot"])
        axes = {}; D = int(ST["axes"]["u_T_perp"].shape[0])
        if A is not None:
            GA = bool(A.get("GA")); axes["u_RGI_perpT"] = (A["u_RGI_perpT"], A["null_u_RGI_perpT"], True); axes["u_RGI_perp"] = (A["u_RGI_perp"], A["null_u_RGI_perp"], True)
            axes["u_dir"] = (A["u_dir"], A["null_u_dir"], False)
            if not GA: lab = lab + [RP.LBL_RGI]
        src = None
        if dirs is not None and int(dirs["g"].shape[0]) > b and int(dirs["g"].shape[1]) == D:
            nl, src = sign_flip_nulls(dirs["c_o"][b], dirs.get("self_objects", list(range(12))), seed=SEEDS["dirs_null"] + int(b))
            axes["g_b"] = (dirs["g"][b], nl["g"], False); axes["s_b"] = (dirs["s"][b], nl["s"], False)
        rd, draws = readouts_block(b, bank, axes, W, NP)
        out["blocks"][str(b)] = dict(status="ran" if A is not None else "ran without O2 axes (round-2 output missing at this block)", GA=GA, G_blk=gblk,
                                     labels=lab, loops=rd, missing_loops=bank.get("_missing"), bank_flag=bank.get("_flag"), axes=list(axes), dirs_nulls=src, n_prompts=NP,
                                     alpha=get(bank, "_meta", "alpha"), suffix=get(bank, "_meta", "suffix"))
        # GEN loop
        if dyn is None: out["gen"][str(b)] = dict(status=f"not run (round-2 output missing: rgi/dyn_b{b}.json)")
        elif not RP.ran(dyn): out["gen"][str(b)] = dict(status=RP.status_of(dyn), GA=dyn.get("GA"), auc_RGI=dyn.get("auc_RGI"))
        else: out["gen"][str(b)] = dict(status="ran", GA=dyn.get("GA"), auc_RGI=dyn.get("auc_RGI"), E_GI=dyn.get("E_GI"), R4=dyn.get("R4_RSI_a_GEN"), S4=dyn.get("S4"),
                                        rows=[{k: r.get(k) for k in ("d", "u_T_perp", "u_RGI_perp", "u_RGI_perpT", "damage_of_push", "identical_pairs", "first_div_median")} for r in dyn.get("rows") or []])
        # dynamics (descriptive): only blocks with G-blk(b), GA(b) and the damage condition of the loops read (OD-A5)
        dn = dict(G_blk=gblk, GA=GA, labels=lab)
        if gblk is not True: dn.update(enters=False, why="G-blk(b) fails" if gblk is False else "G-blk(b) not decidable (round-1 summary missing)")
        elif not GA: dn.update(enters=False, why="GA(b) fails" if GA is False else "O2 axes missing at this block")
        else:
            dn["enters"] = True; dn["self_loop"] = self_loop_rule(rd, b)
            dn["gen_loop"] = gen_rule(dyn) if RP.ran(dyn) else dict(verdict=out["gen"][str(b)]["status"])
            if get(rd, "RSI_a", "damage_ok") and all(("RSI_a", n, "u_RGI_perpT") in draws for n in (3, 4)):
                Xp[b] = float(np.mean([num(get(rd, "RSI_a", n, "u_RGI_perpT", "A")) for n in (3, 4)]))
                Xd[b] = (draws[("RSI_a", 3, "u_RGI_perpT")] + draws[("RSI_a", 4, "u_RGI_perpT")]) / 2; entering.append(b)
        out["dynamics"][str(b)] = dn
        del ST, bank
    order = [b for b in (info.get("order") or info["B3"]) if b in entering]
    out["pull_and_delta"] = monotone(order, Xp, Xd, "pull toward general grows with delta", "pull toward general falls with delta")
    out["pull_and_delta"]["definition"] = "X_b = RSI_a's mean A on u_RGI_perpT over d_3, d_4; entering blocks: G-blk(b), GA(b), damage of RSI_a's pushes d_1 .. d_3 <= 0.15 nats"
    return out


# ======================================================================================================================================
# A3: u_T_perp spillover at the blocks of B3
# ======================================================================================================================================
def spillover(w, is_self, u, nulls):
    """s_o = w_o . unit(u); S_self / S_other = means over the SELF / other-owner objects; z = (S - mean over the nulls) / their SD (ddof 1);
    FR = S_other / S_self only where z_self >= 3 (critique 22f). The formula of r2_rgi_stats.spillover."""
    import torch
    w = torch.as_tensor(w, dtype=torch.float64); u = torch.as_tensor(u, dtype=torch.float64); nl = torch.as_tensor(nulls, dtype=torch.float64)
    sm = torch.as_tensor(np.asarray(is_self, dtype=bool)); u = u / u.norm(); nl = nl / nl.norm(dim=1, keepdim=True)
    s = w @ u; sn = w @ nl.T; S_self, S_oth = float(s[sm].mean()), float(s[~sm].mean()); N_self, N_oth = sn[sm].mean(0), sn[~sm].mean(0)
    z = lambda v, nv: float((v - nv.mean()) / nv.std()) if float(nv.std()) > 0 else float("nan")
    zs, zo = z(S_self, N_self), z(S_oth, N_oth)
    return dict(S_self=S_self, S_other=S_oth, z_self=zs, z_other=zo, FR=S_oth / S_self if (math.isfinite(zs) and zs >= Z_SPILL and S_self != 0) else float("nan"),
                n_null=int(nl.shape[0]))


def spill_rule(z_self, z_other, FR):
    """A7 causal, at the blocks of B3 (critique 22f)."""
    zs, zo = num(z_self), num(z_other)
    if not (fin(zs) and fin(zo)): return "unclear"
    if zs >= Z_SPILL and ge(FR, FR_HI) and zo >= Z_SPILL: return "RSI push spills over"
    if zs < Z_SPILL and zo >= Z_SPILL: return "spills over (no self response)"
    if zs >= Z_SPILL and lt(FR, FR_LO): return "stays with self"
    if zs < Z_SPILL and zo < Z_SPILL: return "no response"
    return "unclear"


def a3(r1_dir, inp, info):
    import r2_rk as R
    cz = inp.pt(f"rgi/causal_{M17}.pt"); dirs = inp.pt(f"rgi/dirs_{M17}.pt"); out = {}
    for b in info["B3"]:
        lab = RP.desc_labels(info, b)
        if not get(info, "final", str(b)): out[str(b)] = dict(status=NOT_RUN_R1); continue
        if cz is None: out[str(b)] = dict(status=f"not run (round-2 output missing: rgi/causal_{M17}.pt)"); continue
        try: ax = R.load_r1_state(r1_dir, b)["axes"]
        except Exception as e:                                                    # noqa: BLE001
            out[str(b)] = dict(status=f"{NOT_RUN_R1}: {e!r}"); continue
        Wt = cz["W"]
        if int(Wt.shape[2]) <= b or int(Wt.shape[3]) != int(ax["u_T_perp"].shape[0]):
            out[str(b)] = dict(status=f"not computable (gradients {tuple(Wt.shape)} do not match block {b} / the round-1 hidden size)"); continue
        w = 0.5 * (Wt[:48, 0, b].double() + Wt[:48, 1, b].double())
        so = [int(x) for x in (dirs.get("self_objects") if dirs is not None and dirs.get("self_objects") is not None else range(12))]
        is_self = np.array([o in so for o in range(48)])
        sp = spillover(w, is_self, ax["u_T_perp"], ax["null_u_T_perp"])
        out[str(b)] = dict(status="ran", verdict=spill_rule(sp["z_self"], sp["z_other"], sp["FR"]), labels=lab, **sp,
                           definition="w = mean of the two IMP-DES half gradients of each model object at block b; nulls = R1 null_u_T_perp(b)")
    return out


# ======================================================================================================================================
# A7 per block and over blocks (r2_rgi_stats.py's geom_stats_<m>.json)
# ======================================================================================================================================
class Stats:
    """Access to geom_stats: S.pt(variant, stat)[b], S.lo / S.hi (pointwise), S.slo / S.shi (sup-t over P), S.po(key, stat), S.loo."""

    def __init__(self, gs):
        self.g = gs; self.NL = len(gs.get("blocks") or get(gs, "gate", "C0") or [])

    def _a(self, v):
        if v is None: return np.full(self.NL, np.nan)
        return np.array([num(x) for x in v], dtype=float)

    def f(self, variant, stat, field): return self._a(get(self.g, "per_block", variant, stat, field))

    def pt(self, variant, stat): return self.f(variant, stat, "point")

    def po(self, key, stat): return self._a(get(self.g, "point_only", key, stat))

    def loo(self, name, stat): return self._a(get(self.g, "loo", name, stat))


def geometry_point(v):
    """The point-estimate form of 'GI and (IS or BA)' for robustness variants. v: stat -> value at one block."""
    GI = ge(v("FG|IMP-DES"), GI_FG)
    IS = bool(ge(v("dFG"), IS_DFG) and gt(v("dFG_val"), 0))
    BA = bool(gt(v("P_cell|IMP-CHG"), 0) and ge(v("q|IMP-CHG"), 0.2) and ge(v("FG|IMP-CHG"), BA_FG) and lt(v("cos_valence"), BA_VAL))
    return bool(GI and (IS or BA)), dict(GI=GI, IS=IS, BA=BA)


def a7_blocks(gs):
    """Per-block items for the blocks of P (2.6 order: C0 and 'uncomposed' pointwise at all 28 blocks, P, sup-t bands over P)."""
    S = Stats(gs); gate = gs.get("gate") or {}; P = [int(b) for b in gate.get("P") or []]; comp = S._a(gate.get("composed")) > 0
    out = dict(P=P, n_P=len(P), blocks={}, C0=gate.get("C0"), uncomposed=gate.get("uncomposed"), passing=gate.get("passing"), composed=gate.get("composed"))
    loo_names = sorted((gs.get("loo") or {}).keys())
    for b in P:
        pb = lambda st, v="pooled": float(S.pt(v, st)[b]); slo = lambda st, v="pooled": float(S.f(v, st, "sim_lo")[b]); shi = lambda st, v="pooled": float(S.f(v, st, "sim_hi")[b])
        GI = bool(ge(pb("FG|IMP-DES"), GI_FG) and ge(slo("FG|IMP-DES"), GI_LO))
        IS = bool(ge(pb("dFG"), IS_DFG) and gt(slo("dFG"), 0) and gt(slo("dFG_val"), 0))
        c0_sim = bool(gt(slo("P_cell|IMP-CHG"), 0) and ge(pb("q|IMP-CHG"), 0.2)); c0_pw = bool(gt(float(S.f("pooled", "P_cell|IMP-CHG", "lo")[b]), 0) and ge(pb("q|IMP-CHG"), 0.2))
        BA = bool(c0_sim and ge(pb("FG|IMP-CHG"), BA_FG) and lt(pb("cos_valence"), BA_VAL))
        rob = {}
        ok_w, parts = geometry_point(lambda st: float(S.po("dot/white/pooled", st)[b])); rob["whitened"] = dict(keeps=ok_w, **parts)
        for nm in loo_names:
            ok_l, parts = geometry_point(lambda st, nm=nm: float(S.loo(nm, st)[b])); rob[nm] = dict(keeps=ok_l, **parts)
        core = bool(GI and (IS or BA)); present = bool(core and all(r["keeps"] for r in rob.values()))
        labels = [l for l, f in (("general editing", ge(pb("FG|CHG-DES"), LBL)), ("owner-general", ge(pb("OG|IMP-DES"), LBL)), ("substrate-general", ge(pb("SG|IMP-DES"), LBL)),
                                 ("recursive-general", ge(pb("OG_LR|IMP-DES"), LBL)), ("signed quality axis", le(pb("cos_quality"), -0.5)),
                                 ("frame-consistent", gt(pb("dFG", "F1"), 0) and gt(pb("dFG", "F2"), 0)), ("object-consistent", gt(float(S.po("obj/raw/pooled", "dFG")[b]), 0)),
                                 ("valence-dominated", ge(pb("cos_valence"), BA_VAL))) if f]

        def self_test(tag):
            absorbed = all(ge(pb(f"rho_S{tag}|IMP-DES", F), ABS_RHO) and lt(shi(f"sigma_self{tag}|IMP-DES", F), ABS_SIG) for F in ("F1", "F2"))
            specific = all(gt(slo(f"sigma_self{tag}|IMP-DES", F), ABS_SIG) and gt(slo(f"dsigma{tag}", F), 0) for F in ("F1", "F2"))
            return "absorbed" if absorbed else ("self-specific" if specific else "unclear")
        e = dict(GI=GI, IS=IS, BA=BA, BA_C0_IMP_CHG=dict(simultaneous=c0_sim, pointwise=c0_pw), core=core, robustness=rob, present=present, labels=labels,
                 self=self_test(""), self_LR=self_test("_LR"), composed=bool(comp[b]) if b < len(comp) else None,
                 numbers=dict(FG=pb("FG|IMP-DES"), FG_sim_lo=slo("FG|IMP-DES"), dFG=pb("dFG"), dFG_sim=[slo("dFG"), shi("dFG")], dFG_val_sim_lo=slo("dFG_val"),
                              FG_IMP_CHG=pb("FG|IMP-CHG"), cos_valence=pb("cos_valence"), cos_quality=pb("cos_quality"),
                              rho_S={F: pb("rho_S|IMP-DES", F) for F in ("F1", "F2")}, sigma_self_sim={F: [slo("sigma_self|IMP-DES", F), shi("sigma_self|IMP-DES", F)] for F in ("F1", "F2")}))
        if e["composed"]:
            e["causal_general"] = bool(ge(pb("FG|IMP-DES", "causal"), GI_FG) and gt(slo("dFG", "causal"), 0))
            e["causal_absorption"] = bool(ge(pb("rho_S|IMP-DES", "causal"), ABS_RHO) and lt(shi("sigma_self|IMP-DES", "causal"), ABS_SIG))
            e["numbers"]["causal"] = dict(FG=pb("FG|IMP-DES", "causal"), dFG_sim_lo=slo("dFG", "causal"), rho=pb("rho_S|IMP-DES", "causal"), sigma_sim_hi=shi("sigma_self|IMP-DES", "causal"))
        else: e["causal_general"] = e["causal_absorption"] = None
        out["blocks"][str(b)] = e
    return out


def depth_conditions(ob, sep, sign=1):
    """V1's depth conditions on one series: separability (G31), r3 > 0 with surrogate p <= 0.05 and bootstrap lower bound > 0, nuisance-
    specific r3 > 0 with p <= 0.05 (sign -1: the mirror)."""
    r3, r3n = num(ob.get("r3")), num(ob.get("r3_nuis")); ci = ob.get("r3_ci") or [None, None]
    if sign > 0: c = dict(separability=bool(sep), r3=bool(gt(r3, 0) and le(ob.get("r3_p_pos"), P_SURR) and gt(ci[0], 0)), r3_nuisance=bool(gt(r3n, 0) and le(ob.get("r3_nuis_p_pos"), P_SURR)))
    else: c = dict(separability=bool(sep), r3=bool(lt(r3, 0) and le(ob.get("r3_p_neg"), P_SURR) and lt(ci[1], 0)), r3_nuisance=bool(lt(r3n, 0) and le(ob.get("r3_nuis_p_neg"), P_SURR)))
    return c


CAUSAL_SERIES = (("FG_causal|IMP-DES", "causal__FG|IMP-DES"), ("rho_causal|IMP-DES", "causal__rho_S|IMP-DES"))


def causal_sim(inp, gs, level=0.95):
    """A7 causal line, 'with simultaneous intervals': sup-t 95 % intervals of D(FG_causal) and D(rho_causal) over the composed blocks of
    P, simultaneous over the two statistics, from the per-draw series of rgi/geom_draws_<m>.npz ('causal__FG|IMP-DES' and
    'causal__rho_S|IMP-DES', [draws, 28]) with the thirds r2_rgi_stats.py used: D_k +- c sd_k, c = the 95 % quantile over draws of
    max_k |D_k(draw) - mean_k| / sd_k (the construction of r2_rgi_stats.supt_band). dict(status, crit, <series>: [lo, hi])."""
    OC = get(gs, "over_blocks", "P_composed") or {}; p = inp.find(f"rgi/geom_draws_{M17}.npz") if inp is not None else None
    if p is None: return dict(status=f"not available (round-2 output missing: rgi/geom_draws_{M17}.npz)")
    try:
        z = np.load(p, allow_pickle=False); cols, pts = [], []
        for nm, key in CAUSAL_SERIES:
            e = OC.get(nm) or {}
            if "D" not in e or key not in z.files or not e.get("top") or not e.get("bottom"): return dict(status=f"not available ({nm}: no D or no per-draw series)")
            d = np.asarray(z[key], dtype=np.float64); cols.append(d[:, [int(x) for x in e["top"]]].mean(1) - d[:, [int(x) for x in e["bottom"]]].mean(1)); pts.append(num(e["D"]))
        Dd = np.stack(cols, 1); Dd = Dd[np.isfinite(Dd).all(1)]
        if len(Dd) < 2: return dict(status="not available (fewer than 2 finite draws)")
        m, sd = Dd.mean(0), Dd.std(0, ddof=1)
        if not (sd > 0).all(): return dict(status="not available (a series has no spread over the draws)")
        c = float(np.quantile(np.max(np.abs(Dd - m) / sd, 1), level))
        return dict(status="ran", crit=c, n_draws=int(len(Dd)), level=level, **{nm: [float(pt - c * k), float(pt + c * k)] for (nm, _), pt, k in zip(CAUSAL_SERIES, pts, sd)})
    except Exception as e:                                                        # noqa: BLE001 - an unreadable draws file falls back to the pointwise interval
        return dict(status=f"not available ({e!r})")


def a7_over(gs, per, gs06, sim=None):
    """Over-block items V1-V6, the RSI-into-RGI line and the causal line (sim: causal_sim's simultaneous intervals)."""
    gate = gs.get("gate") or {}; P = per["P"]; out = dict(n_P=len(P))
    did = (gs06 or {}).get("r_DiD") if gs06 else None
    if did is None or "r_DiD" not in (did or {}): did = gs.get("r_DiD") if "r_DiD" in (gs.get("r_DiD") or {}) else did
    arm06 = bool(gs06 is not None)
    out["arm_0p6B"] = dict(ran=arm06, r_DiD=get(did, "r_DiD"), p_pos=get(did, "p_pos"), p_neg=get(did, "p_neg"), ci=get(did, "ci"), blocks=get(did, "blocks"),
                           status=get(did, "status") if did else "not run (G33)" if not arm06 else None)
    if len(P) < MIN_P or not gate.get("G30_ok", len(P) >= MIN_P):
        v = "undecided (fewer than 12 passing blocks)"
        out.update(verdict=v, rsi_into_rgi=dict(verdict=v), causal_line=dict(verdict=v), top=[], bottom=[]); return out
    OB = get(gs, "over_blocks", "P") or {}; ob = OB.get("dFG") or {}; top, bot = [int(x) for x in ob.get("top") or []], [int(x) for x in ob.get("bottom") or []]
    sep = bool(get(gs, "delta", "separability", "last", "ok")); B = per["blocks"]
    present = [b for b in P if B[str(b)]["present"]]; n_top = sum(b in top for b in present); n_bot = sum(b in bot for b in present)
    D, Dci = num(ob.get("D")), ob.get("D_ci") or [None, None]; sm, smci = num(ob.get("spearman_mmlu")), ob.get("spearman_mmlu_ci") or [None, None]
    so, Dw = num(ob.get("spearman_obj")), num(ob.get("D_whitened")); rd = num(get(did, "r_DiD")); did_ran = arm06 and fin(rd)
    up = dict(present_top=n_top >= 2, D=bool(ge(D, D_MIN) and gt(Dci[0], 0)), spearman=bool(ge(sm, SPEAR) and gt(smci[0], 0) and gt(so, 0)), D_whitened=gt(Dw, 0))
    dn = dict(present_bottom=n_bot >= 2, D=bool(le(D, -D_MIN) and lt(Dci[1], 0)), spearman=bool(le(sm, -SPEAR) and lt(smci[1], 0) and lt(so, 0)), D_whitened=lt(Dw, 0))
    du, dd = depth_conditions(ob, sep, +1), depth_conditions(ob, sep, -1)
    if arm06: du["r_DiD"] = bool(did_ran and rd > 0); dd["r_DiD"] = bool(did_ran and rd < 0)
    V2u, V2d = all(up.values()), all(dn.values()); V1u, V1d = V2u and all(du.values()), V2d and all(dd.values())
    V4 = bool(len(present) >= 2 and fin(sm) and abs(sm) < SPEAR_FLAT and inside(Dci, -FLAT_D, FLAT_D))
    if V1u: v, code = "V1: RGI geometry grows with delta beyond depth", "V1"
    elif V2u: v, code = "V2: RGI geometry grows with delta or depth", "V2"
    elif V1d: v, code = "V3: RGI geometry falls with delta beyond depth", "V3"
    elif V2d: v, code = "V3: RGI geometry falls with delta or depth", "V3 (delta or depth)"
    elif V4: v, code = "V4: RGI geometry present but not tied to delta", "V4"
    elif not present:
        gi_any = [b for b in P if B[str(b)]["GI"]]; gi_only = [b for b in gi_any if not (B[str(b)]["IS"] or B[str(b)]["BA"])]
        sub = "target-specific" if not gi_any else ("general editing, not improvement-specific" if gi_only else
                                                    "GI and (IS or BA) hold at some block, but not in the whitened metric or under every leave-one-out")
        v = f"V5: no RGI geometry ({sub})"; code = "V5"; out["V5_blocks"] = dict(GI=gi_any, GI_without_IS_and_BA=gi_only)
    else: v, code = "V6: undecided", "V6"
    out.update(verdict=v, code=code, top=top, bottom=bot, present_blocks=present, n_present_top=n_top, n_present_bottom=n_bot,
               conditions=dict(grows=up, grows_depth=du, falls=dn, falls_depth=dd, V4=V4), separability=get(gs, "delta", "separability", "last"),
               numbers=dict(D=D, D_ci=Dci, spearman_mmlu=sm, spearman_mmlu_ci=smci, spearman_obj=so, D_whitened=Dw, r3=ob.get("r3"), r3_p_pos=ob.get("r3_p_pos"),
                            r3_p_neg=ob.get("r3_p_neg"), r3_ci=ob.get("r3_ci"), r3_nuis=ob.get("r3_nuis"), r3_nuis_p_pos=ob.get("r3_nuis_p_pos"), r3_nuis_p_neg=ob.get("r3_nuis_p_neg")),
               note="'falls' = the mirror of V1 / V2 with 'present at >= 2 bottom-third blocks'; with only depth varying inside one model V2 is the realistic best outcome")
    # RSI-into-RGI line (series: frames pooled; 'absorbed' / 'self-specific' per block need both frames)
    orh, osg = OB.get("rho_S|IMP-DES@pooled") or {}, OB.get("sigma_self|IMP-DES@pooled") or {}
    n_abs = sum(B[str(b)]["self"] == "absorbed" for b in top); n_spec = sum(B[str(b)]["self"] == "self-specific" for b in top)
    Dr, Drc = num(orh.get("D")), orh.get("D_ci") or [None, None]; Ds, Dsc = num(osg.get("D")), osg.get("D_ci") or [None, None]
    if ge(Dr, D_MIN) and gt(Drc[0], 0) and n_abs >= 2: rv, ser = "RSI dissolves into RGI as delta grows", orh
    elif ge(Ds, D_MIN) and gt(Dsc[0], 0) and n_spec >= 2: rv, ser = "RSI stays its own direction / grows apart", osg
    elif inside(Drc, -FLAT_D, FLAT_D) and inside(Dsc, -FLAT_D, FLAT_D): rv, ser = "no relation", None
    else: rv, ser = "undecided", None
    bd = depth_conditions(ser, sep, +1) if ser is not None else None
    out["rsi_into_rgi"] = dict(verdict=rv + (" (beyond depth)" if (bd and all(bd.values())) else (" (with delta or depth)" if ser is not None else "")),
                               D_rho_S=Dr, D_rho_S_ci=Drc, D_sigma_self=Ds, D_sigma_self_ci=Dsc, absorbed_top=n_abs, self_specific_top=n_spec, depth_conditions=bd,
                               per_frame={F: dict(D_rho_S=get(OB, f"rho_S|IMP-DES@{F}", "D"), D_sigma_self=get(OB, f"sigma_self|IMP-DES@{F}", "D")) for F in ("F1", "F2")})
    # causal line (composed blocks of P)
    OC = get(gs, "over_blocks", "P_composed") or {}; oc, orc = OC.get("FG_causal|IMP-DES") or {}, OC.get("rho_causal|IMP-DES") or {}
    Dc, Dcc = num(oc.get("D")), oc.get("D_ci") or [None, None]; sim_ok = isinstance(sim, dict) and sim.get("status") == "ran"

    def causal_rule(ci):
        if "D" not in oc: return "not computable (too few composed blocks of P)", 0
        if ge(Dc, D_MIN) and gt(ci[0], 0): return "causal generality grows with delta or depth", 1
        if le(Dc, -D_MIN) and lt(ci[1], 0): return "causal generality falls with delta or depth", -1
        return "no relation", 0

    def with_depth(cv, sgn):
        cd = depth_conditions(oc, sep, sgn) if sgn else None
        return (cv.replace("with delta or depth", "with delta beyond depth") if (cd and all(cd.values())) else cv), cd
    two = lambda v: list(v) if (isinstance(v, (list, tuple)) and len(v) == 2 and all(fin(num(x)) for x in v)) else None
    st_f, st_r = two(oc.get("D_sim_ci")), two(orc.get("D_sim_ci"))                 # stored by r2_rgi_stats.py (the same sup-t construction)
    sim_f = st_f or (sim["FG_causal|IMP-DES"] if sim_ok else None); sim_r = st_r if st_f else (sim.get("rho_causal|IMP-DES") if sim_ok else None)
    src = ("r2_rgi_stats.py (D_sim_ci, D_sim_crit)" if st_f else "r2_ana from rgi/geom_draws (D_sim_ci not stored)" if sim_ok else None)
    xc = None
    if st_f and sim_ok:
        dmax = max(abs(num(a_) - num(b_)) for a_, b_ in zip(st_f + (st_r or []), sim["FG_causal|IMP-DES"] + (sim.get("rho_causal|IMP-DES") if st_r else [])))
        xc = dict(max_abs_diff=dmax, agrees=bool(dmax <= 1e-6), crit_stored=oc.get("D_sim_crit"), crit_r2_ana=sim.get("crit"))
    (cv, cdep), (cv_pw, _) = with_depth(*causal_rule(sim_f if sim_f else Dcc)), with_depth(*causal_rule(Dcc))
    out["causal_line"] = dict(verdict=cv, D_FG_causal=Dc, D_FG_causal_ci=Dcc, D_rho_causal=orc.get("D"), D_rho_causal_ci=orc.get("D_ci"), depth_conditions=cdep,
                              composed_blocks=gate.get("P_composed"), simultaneous=sim, verdict_with_pointwise_intervals=cv_pw, simultaneous_source=src,
                              simultaneous_crosscheck=xc,
                              intervals=("simultaneous over D(FG_causal) and D(rho_causal) (sup-t, 95 %)" if sim_f else
                                         "pointwise 95 % (" + str((sim or {}).get("status", "simultaneous intervals not computed")) + ")"),
                              D_FG_causal_sim=sim_f, D_rho_causal_sim=sim_r)
    return out


def combined(per, over, dyn, order):
    """'The geometry forces RGI': V1 (or V2, worded 'with delta or depth') and RGI geometry present at >= 2 top-third blocks and absorbed at
    >= 2 top-third blocks and causal-general and causal absorption at >= 2 composed top-third blocks and, where the dynamics ran at both
    ends, 'self loop drawn to general' at the highest-delta and not at the lowest-delta entering block; otherwise the report names what
    holds."""
    B = per.get("blocks") or {}; top = over.get("top") or []; code = over.get("code")
    n_present = sum(B[str(b)]["present"] for b in top if str(b) in B); n_abs = sum(B[str(b)]["self"] == "absorbed" for b in top if str(b) in B)
    n_caus = sum(bool(B[str(b)].get("composed") and B[str(b)].get("causal_general") and B[str(b)].get("causal_absorption")) for b in top if str(b) in B)
    ent = [b for b in (order or []) if get(dyn, str(b), "enters")]
    dyn_ok, dyn_note = True, "dynamics did not run at two entering blocks: clause not applied"
    if len(ent) >= 2:
        hi_v, lo_v = get(dyn, str(ent[-1]), "self_loop", "verdict"), get(dyn, str(ent[0]), "self_loop", "verdict")
        dyn_ok = bool(hi_v == "self loop drawn to general" and lo_v != "self loop drawn to general"); dyn_note = f"highest-delta block {ent[-1]}: {hi_v!r}; lowest-delta block {ent[0]}: {lo_v!r}"
    c = dict(V1_or_V2=code in ("V1", "V2"), present_top=n_present >= 2, absorbed_top=n_abs >= 2, causal_top=n_caus >= 2, dynamics=dyn_ok)
    forces = all(c.values()); holds = []
    if forces: st = "the geometry forces RGI" + (" (with delta or depth)" if code == "V2" else "")
    else:
        if code in ("V1", "V2", "V3", "V3 (delta or depth)", "V4"): holds.append(str(over.get("verdict")))
        if n_present >= 2 and n_caus < 2: holds.append("represented but not causal")
        if code == "V5" and "general editing" in str(over.get("verdict")): holds.append("general editing, not improvement")
        if code == "V5" and "target-specific" in str(over.get("verdict")): holds.append("target-specific (no general improvement direction)")
        if code == "V5" and "whitened" in str(over.get("verdict")): holds.append("no robust RGI geometry (lost in the whitened metric or under a leave-one-out)")
        if "RSI stays its own direction" in str(get(over, "rsi_into_rgi", "verdict")): holds.append("RSI stays its own direction")
        if "dissolves" in str(get(over, "rsi_into_rgi", "verdict")): holds.append("RSI dissolves into RGI as delta grows")
        if n_abs >= 2 and "dissolves" not in str(get(over, "rsi_into_rgi", "verdict")): holds.append("self-improvement absorbed at the top-third blocks")
        if code in (None, "V6") or str(over.get("verdict", "")).startswith("undecided"): holds.append(str(over.get("verdict", "undecided")))
        st = "the geometry does not force RGI: " + ("; ".join(holds) if holds else "nothing of the frozen statements holds")
    return dict(statement=st, forces=forces, conditions=c, counts=dict(present_top=n_present, absorbed_top=n_abs, causal_top=n_caus), dynamics_clause=dyn_note, holds=holds)


def a7(inp, dyn, order):
    gs = inp.json(f"rgi/geom_stats_{M17}.json"); gs06 = inp.json(f"rgi/geom_stats_{M06}.json")
    if gs is None:
        st = f"not run (round-2 output missing: rgi/geom_stats_{M17}.json)"
        return dict(status=st, over_blocks=dict(verdict=st), combined=dict(statement=st, forces=None))
    per = a7_blocks(gs); over = a7_over(gs, per, gs06, causal_sim(inp, gs))
    sg = gs.get("spillover_g_self") or {}
    return dict(status="ran", per_block=per, over_blocks=over, combined=combined(per, over, dyn, order), flags=gs.get("flags"), mock=gs.get("mock"),
                gates=dict(G30=dict(ok=bool(get(gs, "gate", "G30_ok")), n_P=get(gs, "gate", "n_P")), G31=get(gs, "delta", "separability", "last"),
                           G32=dict(composed=get(gs, "gate", "composed"), P_composed=get(gs, "gate", "P_composed")), G33=dict(arm_ran=gs06 is not None)),
                g_self_spillover={k: sg.get(k) for k in ("z_self", "z_other", "FR", "S_self", "S_other")},
                topic_like=get(gs, "delta", "tau", "dot", "topic_like"), delta_mmlu_last=get(gs, "delta", "mmlu", "last", "delta"),
                arm_0p6B=None if gs06 is None else dict(P=get(gs06, "gate", "P"), n_P=get(gs06, "gate", "n_P"), flags=gs06.get("flags")))


# ======================================================================================================================================
# AT8 (toy rules), AT7 (the rule's behaviour with the measured noise), AT5 slope labels, AT3 / AT4 / AT6
# ======================================================================================================================================
def at2_worlds(inp):
    """{ws: world JSON} of rgi_toy/r1/w<ws>.json."""
    out = {}
    for p in inp.find_all("rgi_toy/r1/w*.json"):
        js = RP.load_json(p)
        if js is not None and "models" in js: out[int(js.get("world", os.path.basename(p)[1:-5]))] = js
    return out


def world_validity(wj):
    """AT8 validity (critique 22h): models failing level 1 are dropped and counted; the world is valid if every condition keeps >= 2 of its
    3 inits and the late task arm (natural plus the task-late levels, models passing level 1) spans >= x8 in delta_h2 (largest over
    smallest condition mean)."""
    conds = [c["cond"] for c in wj.get("conditions") or []]; ms = wj.get("models") or []
    kept = {c: [m for m in ms if m.get("cond") == c and m.get("level1")] for c in conds}
    n_all = {c: sum(1 for m in ms if m.get("cond") == c) for c in conds}
    lt_conds = [c["cond"] for c in wj.get("conditions") or [] if c.get("arm") == "nat" or (c.get("arm") == "late" and c.get("reg") == "task")]
    means = {c: float(np.mean([num(m.get("delta_h2")) for m in kept[c]])) for c in lt_conds if kept[c] and all(fin(num(m.get("delta_h2"))) for m in kept[c])}
    pos = [v for v in means.values() if v > 0]; span = (max(pos) / min(pos)) if (len(pos) == len(lt_conds) and len(pos) >= 2) else float("nan")
    keeps = all(len(kept[c]) >= 2 for c in conds) and bool(conds)
    return dict(valid=bool(keeps and ge(span, SPAN_MIN)), keeps_2_of_3=bool(keeps), span=span, span_ok=ge(span, SPAN_MIN), dropped=int(sum(n_all.values()) - sum(len(v) for v in kept.values())),
                kept={c: len(v) for c, v in kept.items()}, late_task_conditions=lt_conds)


def arm_conditions(wj, arm):
    """Conditions of the late arm (natural + task late + sham late) or the early arm (natural + task early + sham early)."""
    return [c["cond"] for c in wj.get("conditions") or [] if c.get("arm") in ("nat", arm)]


def line_rows(wj, arm, key):
    """Models of one arm that pass level 1 with finite x, p and line value: arrays (cond, x, p, k, y)."""
    cs = set(arm_conditions(wj, arm)); rows = [m for m in wj.get("models") or [] if m.get("cond") in cs and m.get("level1")
                                                and all(fin(num(m.get(q))) for q in ("x", "p", key))]
    return ([m["cond"] for m in rows], np.array([num(m["x"]) for m in rows]), np.array([num(m["p"]) for m in rows]), np.array([float(m.get("k") or 0) for m in rows]),
            np.array([num(m[key]) for m in rows]))


def line_status(worlds, valid, arm, key):
    """One line in one arm: per valid world the full and reduced OLS (r2_sim.reg2_fit_world), the two-step verdict (r2_sim.reg2_verdict,
    world-level t 95 % intervals) and AT8's labels (perturbation-driven, ABSENT, b- interval inside (-0.05, +0.05))."""
    import r2_sim as SIM
    Bf, Br, used, cm = [], [], [], {}
    for ws in valid:
        cond, x, p, k, y = line_rows(worlds[ws], arm, key)
        if len(y) < 8 or len(set(cond)) < 3: continue
        f, r = SIM.reg2_fit_world(x, p, k, y); Bf.append(f); Br.append(r); used.append(ws)
        for c in set(cond): cm.setdefault(c, []).append(float(np.mean(y[np.array(cond) == c])))
    out = dict(n_worlds=len(used), worlds=used)
    if len(used) < 2: out.update(verdict="undecided (fewer than 2 worlds with a fit)", flags={}); return out
    Bf, Br = np.array(Bf), np.array(Br); v = SIM.reg2_verdict(Bf, Br)
    names_f, names_r = ("a", "b-", "b+", "e", "c0", "d-", "d+"), ("a", "b-", "b+", "e")
    cf = {n_: dict(zip(("mean", "lo", "hi"), map(float, SIM.ci(Bf[:, i])))) for i, n_ in enumerate(names_f)}
    cr = {n_: dict(zip(("mean", "lo", "hi"), map(float, SIM.ci(Br[:, i])))) for i, n_ in enumerate(names_r)}
    cond_ci = {c: tci(vs) for c, vs in cm.items()}
    absent = bool(cond_ci) and all(lt(ci_["hi"], ABSENT_HI) for ci_ in cond_ci.values())
    flags = dict(regulariser_specific=v.startswith("regulariser-specific"), grows="GROWS" in v, erodes="ERODES" in v, flat="FLAT beyond" in v, needs="NEEDS" in v,
                 perturbation_driven=bool(cr["e"]["hi"] < 0), absent=absent, b_minus_inside=bool(cr["b-"]["lo"] > -0.05 and cr["b-"]["hi"] < 0.05))
    out.update(verdict=("ABSENT" if absent else v), rule_verdict=v, flags=flags, full=cf, reduced=cr, condition_means=cond_ci)
    return out


def answer(L, S, T1, PHI, lam_testable):
    """AT8 answer mapping (Lambda and S must agree; no OR across lines; first match in the order of the plan text)."""
    f = lambda s, k: bool(get(s, "flags", k))
    grows = lambda s: f(s, "grows") and not f(s, "absent"); flat = lambda s: f(s, "flat") and not f(s, "absent")
    F_grows = grows(PHI) and grows(T1); F_abs_flat = all(f(s, "absent") or flat(s) for s in (PHI, T1))
    notdelta = lambda s: f(s, "regulariser_specific") or (f(s, "perturbation_driven") and flat(s) and f(s, "b_minus_inside"))
    if all(f(s, "absent") for s in (L, S, T1, PHI)): return "no RGI geometry"
    if not lam_testable: return "UNDECIDED (recipe line not testable: Lambda not present at natural delta, G39)"
    if grows(L) and grows(S) and F_grows: return "delta builds a geometry that forces RGI"
    if grows(L) and grows(S) and F_abs_flat: return "delta permits RGI but does not force it"
    if all(f(s, "needs") and flat(s) for s in (L, S)): return "RGI needs task separation up to the natural level, more adds nothing"
    if all(f(s, "erodes") and not f(s, "absent") for s in (L, S)): return "more delta erodes it"
    if all(notdelta(s) and not f(s, "absent") for s in (L, S)): return "not delta: the shape of the space matters, not its size"
    return "UNDECIDED"


def nat_level(worlds, valid, key, sub=None):
    """Per valid world the mean of a natural-condition value over its level-1 natural models; t interval over worlds."""
    vals = []
    for ws in valid:
        v = [num(m.get(key)[sub]) if sub is not None and isinstance(m.get(key), list) and len(m.get(key)) > sub else num(m.get(key)) if sub is None else float("nan")
             for m in worlds[ws].get("models") or [] if m.get("cond") == "nat" and m.get("level1")]
        v = [x for x in v if fin(x)]
        if v: vals.append(float(np.mean(v)))
    return tci(vals), vals


def present(ci_):
    return bool(ge(ci_["mean"], PRESENT_MEAN) and gt(ci_["lo"], PRESENT_LO))


def at8(inp, pcg):
    worlds = at2_worlds(inp); out = dict(n_worlds=len(worlds), worlds=sorted(worlds))
    if not worlds:
        st = "not run (round-2 output missing: rgi_toy/r1/w<ws>.json)"; out.update(status=st, answer=st, recursion=dict(verdict=st)); return out, worlds, []
    val = {ws: world_validity(wj) for ws, wj in worlds.items()}; valid = sorted(ws for ws, v in val.items() if v["valid"])
    g38 = len(valid) >= VALID_MIN
    out.update(status="ran", validity={str(ws): v for ws, v in val.items()}, valid_worlds=valid,
               G38=dict(ok=g38, n_valid=len(valid), needed=VALID_MIN, verdict=None if g38 else "UNDECIDED (probe failed)"))
    lam_ci, _ = nat_level(worlds, valid, "Lambda"); s_ci, _ = nat_level(worlds, valid, "S_gain")
    g39 = present(lam_ci); out["G39"] = dict(ok=g39, Lambda_natural=lam_ci, consequence=None if g39 else "the recipe line is 'not testable'")
    out["lines"] = {}
    for arm in ("late", "early"):
        st = {nm: line_status(worlds, valid, arm, key) for nm, key in LINES}
        if not g39: st["Lambda"]["not_testable"] = "Lambda not present at natural delta (machinery gate, G39)"
        ans = answer(st["Lambda"], st["S"], st["T1"], st["Phi"], g39) if g38 else "UNDECIDED (probe failed)"
        out["lines"][arm] = dict(role="primary" if arm == "late" else "secondary", answer=ans, lines=st,
                                 F=dict(grows=bool(get(st, "Phi", "flags", "grows") and get(st, "T1", "flags", "grows")),
                                        absent_or_flat=all(bool(get(st, k, "flags", "absent") or get(st, k, "flags", "flat")) for k in ("Phi", "T1"))))
    out["answer"] = out["lines"]["late"]["answer"]; out["answer_early_arm"] = out["lines"]["early"]["answer"]
    # recursion (natural condition, valid worlds)
    rho = {d: nat_level(worlds, valid, "rho", d - 1)[0] for d in (1, 2, 3, 4, 5)}; rg, _ = nat_level(worlds, valid, "RG")
    pg = tci([num(next((p.get("gain") for p in worlds[ws].get("pooling") or [] if get(p, "cell", "cond") == "nat"), None)) for ws in valid])
    pgw = tci([num(get(worlds[ws], "pooling_world", 0, "gain")) for ws in valid])
    c = dict(rho=all(ge(rho[d]["lo"], RHO_MIN) for d in (1, 2, 3)), RG=gt(rg["lo"], 0), pooling=gt(pg["lo"], 0))
    pres = dict(Lambda=present(lam_ci), S=present(s_ci))
    if not g38: rv = "not decidable (probe failed, G38)"
    elif any(c.values()): rv = "recursive"
    elif any(pres.values()): rv = "one-step portable"
    else: rv = "neither recursive nor one-step portable (Lambda and S not present at natural delta)"
    out["recursion"] = dict(verdict=rv, conditions=c, rho_d=rho, RG=rg, pooling_gain_natural=pg, pooling_gain_all_conditions=pgw, present=pres,
                            Lambda_natural=lam_ci, S_natural=s_ci, note="natural condition, valid worlds; warm start descriptive (pairs warm10 / warm40)")
    return out, worlds, valid


def at4(inp):
    fam = inp.json("rgi_toy/fam.json")
    if fam is None: return dict(status="not run (round-2 output missing: rgi_toy/fam.json)", verdict="not run (round-2 output missing: rgi_toy/fam.json)")
    out = dict(status="ran", variants={}, G40=fam.get("relu_calibration"))
    for name, rows in (fam.get("variants") or {}).items():
        rows = rows if isinstance(rows, list) else list(rows.values()); n = len(rows); l1 = [r for r in rows if r.get("level1")]
        trains = len(l1) >= RP.quorum(n) and n > 0; ci_ = tci([r.get("S_gain") for r in l1])
        nt = name.lower().startswith("relu") and get(fam, "relu_calibration", "testable") is False
        out["variants"][name] = dict(n_worlds=n, n_level1=len(l1), trains=bool(trains) and not nt, not_testable=bool(nt), S_gain=ci_, portable=bool(gt(ci_["lo"], FAM_LO)),
                                     Lambda=tci([r.get("Lambda") for r in l1]), T4=tci([r.get("T4_mean") for r in l1]), producer_summary_S_gain=get(fam, "summary", name, "S_gain"))
    tr = [k for k, v in out["variants"].items() if v["trains"]]
    if not tr: out["verdict"] = "not decidable (no variant trains)"
    elif all(out["variants"][k]["portable"] for k in tr): out["verdict"] = "portable across these families (an upper bound for real model families)"
    else: out["verdict"] = "not portable across every family that trains: " + ", ".join(k for k in tr if not out["variants"][k]["portable"])
    out["variants_that_train"] = tr; out["not_testable"] = [k for k, v in out["variants"].items() if v["not_testable"]]
    return out


def at7(worlds, valid, at8_out):
    """AT7: r2_sim.reg2_simulate re-run with the noise measured in AT2 per line and arm (s_m, s_wc, s_w, jit and the design grid from
    r2_sim.reg2_noise on the level-1 models of the valid worlds; all worlds, flagged, if no world is valid)."""
    import r2_sim as SIM
    if not worlds: return dict(status="not run (round-2 output missing: rgi_toy/r1/w<ws>.json)")
    use = valid if len(valid) >= 2 else sorted(worlds)
    out = dict(status="ran", nsim=AT7_NSIM, seed=SEEDS["at7_sim"], worlds=use, scenarios=list(SIM.SCENARIOS),
               flags=[] if len(valid) >= 2 else ["fewer than 2 valid worlds: noise measured on all worlds"], arms={})
    for arm in ("late", "early"):
        out["arms"][arm] = {}
        for nm, key in LINES:
            W_, C_, Y_, X_, P_ = [], [], [], [], []
            for ws in use:
                cond, x, p, k, y = line_rows(worlds[ws], arm, key); W_ += [ws] * len(y); C_ += cond; Y_ += list(y); X_ += list(x); P_ += list(p)
            if len(Y_) < 8 or len(set(W_)) < 2 or len(set(C_)) < 3:
                out["arms"][arm][nm] = dict(status="not computable (too few models)"); continue
            try:
                nz = SIM.reg2_noise(np.array(W_), np.array(C_), np.array(Y_), np.array(X_), np.array(P_))
                grid = [tuple(c) for c in nz["cond"]]
                rates = {sc: SIM.reg2_simulate(f, cond=grid, NW=len(set(W_)), ninit=3, nsim=AT7_NSIM, s_w=nz["s_w"], s_wc=nz["s_wc"], s_m=nz["s_m"], seed=SEEDS["at7_sim"], jit=nz["jit"])
                         for sc, f in SIM.SCENARIOS.items()}
                out["arms"][arm][nm] = dict(status="ran", noise={k: nz[k] for k in ("s_m", "s_wc", "s_w", "jit", "n_h", "n_worlds")}, design=[list(c) for c in grid], rates=rates,
                                            observed_verdict=get(at8_out, "lines", arm, "lines", nm, "verdict"))
            except Exception as e:                                                # noqa: BLE001
                out["arms"][arm][nm] = dict(status=f"failed: {e!r}")
    out["note"] = "descriptive: the verdict is read as it falls; rates = share of simulated studies per verdict of r2_sim.reg2_verdict (resolvable effect sizes); pjit as in the reference (0.03)"
    return out


def at5(inp, pcg, at8_out):
    """AT5 (descriptive): per line the OLS slope of the model values on delta_h2 across the AT5 models (worlds as clusters, 10,000 cluster
    bootstrap draws, pc_analyze.cluster_boot_ols); a slope whose 95 % interval lies above 0 while the AT2 verdict of that line is not GROWS
    is labelled 'width/capacity, not delta'. BP models from rgi_toy/cross; PC models (pc_toy/cross_pc) only at G-PC widths."""
    import pc_analyze as PA
    files = inp.find_all("rgi_toy/cross/w*.json"); pfiles = inp.find_all("pc_toy/cross_pc/w*.json"); gw = set(pcg.get("widths_G_PC") or [])
    if not files and not pfiles: return dict(status="not run (round-2 output missing: rgi_toy/cross/w<ws>.json)")
    rows = []
    for p in files:
        js = RP.load_json(p) or {}
        for m in js.get("models") or []:
            if not m.get("level1"): continue
            rows.append(dict(ws=m.get("ws"), kind="bp", name=m.get("name"), W=m.get("W"), delta_h2=num(m.get("delta_h2")), Lambda=num(m.get("Lambda")), Phi=num(m.get("Phi")),
                             S=num(get(m, "as_receiver", "S_gain")), T1=num(get(m, "as_receiver", "T1_gain"))))
    pc_status = {}; n_pairs = dict(testable=0, not_testable=0); pair_tab = {}
    for p in pfiles:
        js = RP.load_json(p) or {}; ms = {m.get("name"): m for m in js.get("models") or []}
        ok_net = lambda m: m is not None and (m.get("kind") != "pc" or int(m.get("W", -1)) in gw)
        for pr in js.get("pairs") or []:
            a, b_ = ms.get(pr.get("A")), ms.get(pr.get("B")); t = ok_net(a) and ok_net(b_); n_pairs["testable" if t else "not_testable"] += 1
            if t:
                k = f"{pr.get('A_arm')} W{pr.get('A_W')} -> {pr.get('B_arm')} W{pr.get('B_W')}"; e = pair_tab.setdefault(k, dict(S_gain=[], t1_gain=[], t4=[]))
                e["S_gain"].append(num(pr.get("S_gain"))); e["t1_gain"].append(num(pr.get("t1_gain"))); e["t4"].append(num(pr.get("t4_mean")))
        for m in ms.values():
            if m.get("kind") != "pc": continue
            W = int(m.get("W", -1)); pc_status[str(W)] = "testable" if W in gw else get(pcg, "G_PC", str(W), "status", default="not testable (no PCS rows at this width)")
            if W not in gw or not m.get("level1"): continue
            rec = [pr for pr in js.get("pairs") or [] if pr.get("B") == m.get("name") and ok_net(ms.get(pr.get("A")))]
            rows.append(dict(ws=m.get("ws"), kind="pc", name=m.get("name"), W=W, delta_h2=num(m.get("delta_h2")), Lambda=num(m.get("Lambda")), Phi=num(m.get("Phi")),
                             S=float(np.mean([num(r.get("S_gain")) for r in rec])) if rec else float("nan"), T1=float(np.mean([num(r.get("t1_gain")) for r in rec])) if rec else float("nan")))
    out = dict(status="ran", n_models=dict(bp=sum(r["kind"] == "bp" for r in rows), pc=sum(r["kind"] == "pc" for r in rows)), pc_widths=pc_status, pc_pairs=n_pairs,
               pairs_with_a_PC_net={k: {q: tci(v)["mean"] for q, v in e.items()} | dict(n=len(e["S_gain"])) for k, e in sorted(pair_tab.items())}, slopes={},
               n_boot=AT5_BOOT, seed=SEEDS["at5_cluster_boot"])
    for nm, key in LINES:
        k = "Lambda" if nm == "Lambda" else ("Phi" if nm == "Phi" else nm)
        use = [r for r in rows if fin(r["delta_h2"]) and fin(r[k])]
        if len(use) < 4 or len({r["ws"] for r in use}) < 2 or len({r["delta_h2"] for r in use}) < 2:
            out["slopes"][nm] = dict(status="not computable (fewer than 4 models or 2 worlds)"); continue
        try:
            full, lo_, hi_ = PA.cluster_boot_ols(np.array([[r["delta_h2"]] for r in use]), np.array([r[k] for r in use]), np.array([r["ws"] for r in use]), B=AT5_BOOT, seed=SEEDS["at5_cluster_boot"])
            at2v = get(at8_out, "lines", "late", "lines", nm, "flags", "grows"); above = bool(lo_[1] > 0)
            out["slopes"][nm] = dict(status="ran", slope=float(full[1]), ci95=[float(lo_[1]), float(hi_[1])], n=len(use), interval_above_0=above, AT2_grows=bool(at2v),
                                     label="width/capacity, not delta" if (above and not at2v) else None)
        except Exception as e:                                                    # noqa: BLE001
            out["slopes"][nm] = dict(status=f"failed: {e!r}")
    return out


# ======================================================================================================================================
# Driver
# ======================================================================================================================================
def run(r1_dir, r2_root, out_path):
    t0 = time.time(); RP.set_threads(); inp = RP.Inputs(r1_dir, r2_root, out_path); info = RP.r1_info(r1_dir); pcg = RP.pc_gates(r1_dir, inp)
    out = dict(script="r2_rgi_verdict.py", plan=RP.PLAN_REL, plan_sha256=RP.PLAN_SHA256, part="A", seeds=SEEDS, test_mode=TEST,
               blocks=dict(B3=info["B3"], b_lo=info.get("b_lo"), b_hi=info.get("b_hi"), order=info.get("order"), delta=info.get("delta"), final=info.get("final"),
                           flags=info.get("block_flags"), G7=info.get("G7"), G8=info.get("G8"), G9=info.get("G9")),
               gates=dict(G_desc=info.get("G_desc"), G_blk=info.get("G_blk"), G_meta=dict(value=pcg.get("G_meta"), source=pcg.get("G_meta_source")),
                          G_PC={W: v["status"] for W, v in (pcg.get("G_PC") or {}).items()}))
    try: out["A6"] = a6(r1_dir, inp, info)
    except Exception as e:                                                        # noqa: BLE001 - a failing part never hides the others
        import traceback; traceback.print_exc(); out["A6"] = dict(status=f"failed: {e!r}", blocks={}, gen={}, dynamics={}, pull_and_delta=dict(verdict="not decidable"))
    out["gates"]["GA"] = {b: get(v, "GA") for b, v in (out["A6"].get("blocks") or {}).items()}
    try: out["A3_u_T_perp_spillover"] = a3(r1_dir, inp, info)
    except Exception as e:                                                        # noqa: BLE001
        import traceback; traceback.print_exc(); out["A3_u_T_perp_spillover"] = dict(status=f"failed: {e!r}")
    out["A7"] = a7(inp, out["A6"].get("dynamics") or {}, info.get("order") or info["B3"])
    a8, worlds, valid = at8(inp, pcg); out["AT8"] = a8; out["AT8"]["family"] = at4(inp)
    out["AT7"] = at7(worlds, valid, a8); out["AT5"] = at5(inp, pcg, a8)
    out["AT0"] = (lambda r0: dict(status="not run (round-2 output missing: rgi_toy/r0.json)") if r0 is None else dict(status="ran", levels=r0.get("levels"), dropped=r0.get("dropped"), selection=r0.get("selection")))(inp.json("rgi_toy/r0.json"))
    out["AT3"] = inp.json("rgi_toy/r1/at3.json") or dict(status="not run (round-2 output missing: rgi_toy/r1/at3.json)")
    out["AT6"] = (lambda f: dict(status="not run (round-2 output missing: rgi_toy/forcing.json)") if f is None else dict(status="ran", summary=f.get("summary"), conditions=f.get("conditions")))(inp.json("rgi_toy/forcing.json"))
    out["summary"] = summary(out)
    out.update(inputs=inp.record(["axes/r2_axes.pt", f"rgi/geom_stats_{M17}.json", f"rgi/geom_stats_{M06}.json", f"rgi/geom_draws_{M17}.npz", f"rgi/causal_{M17}.pt", f"rgi/dirs_{M17}.pt", "rgi_toy/r0.json",
                                  "rgi_toy/fam.json", "rgi_toy/forcing.json"] + [f"rgi/dyn_b{b}.json" for b in info["B3"]]),
               finder=inp.log, r1_finder=info.get("finder"), seconds=time.time() - t0)
    RP.write_json(out_path, out); log(f"Part A verdicts -> {out_path} ({time.time() - t0:.0f}s): {out['summary']}")
    return out


def summary(o):
    s = {}
    s["A7 over blocks"] = get(o, "A7", "over_blocks", "verdict"); s["A7 RSI-into-RGI"] = get(o, "A7", "over_blocks", "rsi_into_rgi", "verdict")
    s["A7 causal line"] = get(o, "A7", "over_blocks", "causal_line", "verdict"); s["A7 combined"] = get(o, "A7", "combined", "statement")
    s["A7 blocks with RGI geometry"] = get(o, "A7", "over_blocks", "present_blocks")
    s["A3 spillover"] = {b: v.get("verdict", v.get("status")) for b, v in (o.get("A3_u_T_perp_spillover") or {}).items() if isinstance(v, dict)}
    s["A6 self loop"] = {b: get(v, "self_loop", "verdict") or f"does not enter ({v.get('why')})" for b, v in (get(o, "A6", "dynamics") or {}).items()}
    s["A6 GEN loop"] = {b: get(get(o, "A6", "dynamics", b), "gen_loop", "verdict") or v.get("status") for b, v in (get(o, "A6", "gen") or {}).items()}
    s["A6 pull and delta"] = get(o, "A6", "pull_and_delta", "verdict")
    s["AT8 answer"] = get(o, "AT8", "answer"); s["AT8 answer (early arm)"] = get(o, "AT8", "answer_early_arm"); s["AT8 recursion"] = get(o, "AT8", "recursion", "verdict")
    s["AT8 family"] = get(o, "AT8", "family", "verdict")
    s["AT5 slope labels"] = {k: v.get("label") for k, v in (get(o, "AT5", "slopes") or {}).items() if isinstance(v, dict) and v.get("label")}
    return s


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if len(a) >= 2 and a[0] == "selftest": return 0 if selftest(a[1]) else 1
    if len(a) != 3: print(__doc__); return 2
    run(*a); return 0


# ======================================================================================================================================
# Self-test: synthetic inputs in the producers' formats, one case per verdict branch, gate failure and missing part
# ======================================================================================================================================
def _rr(c=0.0, p=0.5, A=0.0): return dict(c=c, p=p, Q=1.0, A=A)


def _rd(b, rsi=(1, 1), ctrl=None, bad=(), missing=(), only=None, axis="u_RGI_perpT"):
    """Readouts of block b for the self-loop rule. rsi: sign of the hit of the RSI loops at (d_3, d_4) (0 = none; only = the RSI loops
    that get it); ctrl = (loop, n, sign): one control hit; bad: loops with a counted push above the damage limit."""
    import r2_rk as R
    rd = {}
    for L in R.loops_of_block(b):
        if L in missing: continue
        e = {n: {axis: _rr()} for n in (1, 2, 3, 4)}
        if L.startswith("RSI") and (only is None or L in only):
            for n, sg in zip((3, 4), rsi):
                if sg: e[n][axis] = _rr(0.5 * sg, 0.001, 0.6 * sg)
        if ctrl and L == ctrl[0]: e[ctrl[1]][axis] = _rr(0.4 * ctrl[2], 0.002, 0.5 * ctrl[2])
        e["damage_ok"] = L not in bad; rd[L] = e
    return rd


def _dyn(perp=((0.3, 0.001),) * 4, perpT=((0.2, 0.01),) * 4, ut=((0.0, 0.5),) * 4, rel=0.8, E=0.9, R4=0.2, dmg=0.03, GA=True, rows=4):
    """rgi/dyn_b<b>.json of r2_gen_loop.py: rows d = 1..4 with (c, p) on u_RGI_perp, u_RGI_perpT and u_T_perp (run_loop's own row)."""
    rw = [dict(n=n, d=n + 1, u_T_perp=dict(c=ut[n][0], p=ut[n][1], rel=rel, A=ut[n][0] / math.sqrt(rel)), u_RGI_perp=dict(c=perp[n][0], p=perp[n][1], Q=2.0, rel=rel, A=perp[n][0] / math.sqrt(rel)),
               u_RGI_perpT=dict(c=perpT[n][0], p=perpT[n][1], Q=2.0, rel=rel, A=perpT[n][0] / math.sqrt(rel)), damage_of_push=dmg[n] if isinstance(dmg, (list, tuple)) else dmg,
               identical_pairs=0, first_div_median=1.0) for n in range(rows)]
    return dict(script="r2_gen_loop.py", status="ran", GA=GA, auc_RGI=0.95, rows=rw, E_GI=E, R4_RSI_a_GEN=R4, S4=0.5)


_GS_POOLED = {"FG|IMP-DES": 0.7, "dFG": 0.2, "dFG_val": 0.15, "P_cell|IMP-CHG": 0.3, "q|IMP-CHG": 0.5, "FG|IMP-CHG": 0.6, "cos_valence": 0.3, "cos_quality": -0.1, "FG|CHG-DES": 0.4,
              "OG|IMP-DES": 0.4, "SG|IMP-DES": 0.4, "OG_LR|IMP-DES": 0.4, "rho_S|IMP-DES": 0.7, "sigma_self|IMP-DES": 0.05}
_GS_FRAME = {"dFG": 0.2, "rho_S|IMP-DES": 0.7, "sigma_self|IMP-DES": 0.05, "dsigma": 0.0, "rho_S_LR|IMP-DES": 0.7, "sigma_self_LR|IMP-DES": 0.05, "dsigma_LR": 0.0}
_GS_CAUSAL = {"FG|IMP-DES": 0.7, "dFG": 0.2, "rho_S|IMP-DES": 0.7, "sigma_self|IMP-DES": 0.05}
_GS_ROB = ("FG|IMP-DES", "dFG", "dFG_val", "P_cell|IMP-CHG", "q|IMP-CHG", "FG|IMP-CHG", "cos_valence")
_GS_LOO = ("owner:SELF", "owner:SMALL", "owner:LARGE", "owner:FAMILY", "substrate:W", "substrate:KV", "substrate:LR")


def _series(D=0.2, ci=(0.1, 0.3), sm=0.7, sm_ci=(0.4, 0.9), so=0.5, Dw=0.1, r3=0.6, p_pos=0.01, p_neg=0.99, r3_ci=(0.2, 0.8), r3n=0.5, pn_pos=0.02, pn_neg=0.98, top=(), bottom=()):
    """One over-block entry of r2_rgi_stats.over_blocks (defaults: grows with delta, depth conditions met)."""
    return dict(D=D, D_ci=list(ci), spearman_mmlu=sm, spearman_mmlu_ci=list(sm_ci), spearman_obj=so, D_whitened=Dw, r3=r3, r3_p_pos=p_pos, r3_p_neg=p_neg, r3_ci=list(r3_ci), r3_nuis=r3n,
                r3_nuis_p_pos=pn_pos, r3_nuis_p_neg=pn_neg, top=list(top), bottom=list(bottom))


_FLAT = dict(D=0.0, D_ci=[-0.05, 0.05], spearman_mmlu=0.0, spearman_mmlu_ci=[-0.2, 0.2], spearman_obj=0.0, D_whitened=0.0)
_FALL = dict(D=-0.2, D_ci=[-0.3, -0.1], spearman_mmlu=-0.7, spearman_mmlu_ci=[-0.9, -0.4], spearman_obj=-0.5, D_whitened=-0.1, r3=-0.6, r3_p_pos=0.99, r3_p_neg=0.01, r3_ci=[-0.8, -0.2], r3_nuis=-0.5,
             r3_nuis_p_pos=0.98, r3_nuis_p_neg=0.02)


def _gs(P=tuple(range(8, 24)), NL=28, set_=None, margin=0.04, composed=None, sep=True, over=None, did=None):
    """A geom_stats_<m>.json in r2_rgi_stats.py's layout. Defaults: every block of P has RGI geometry (GI, IS, BA; robust), is 'absorbed'
    in both frames and composed with causal generality and absorption; the dFG series grows with delta beyond depth, the rho_S, sigma_self
    and causal series are flat. set_: {(variant, stat[, field]): value or {block: value}} (variant: pooled / F1 / F2 / causal / white / obj
    / loo:<name>; field: point (moves the bands with it) or sim_lo / sim_hi / lo / hi); over: {series: dict of overrides or None}."""
    P = list(P); k = len(P) // 3; top, bot = P[len(P) - k:], P[:k]
    arr = lambda v: [float(v)] * NL
    entry = lambda v: dict(point=arr(v), lo=arr(v - margin / 2), hi=arr(v + margin / 2), sim_lo=arr(v - margin), sim_hi=arr(v + margin), sd=arr(margin / 2))
    pb = dict(pooled={s: entry(v) for s, v in _GS_POOLED.items()}, F1={s: entry(v) for s, v in _GS_FRAME.items()}, F2={s: entry(v) for s, v in _GS_FRAME.items()},
              causal={s: entry(v) for s, v in _GS_CAUSAL.items()})
    po = {"dot/white/pooled": {s: arr(_GS_POOLED[s]) for s in _GS_ROB}, "obj/raw/pooled": dict(dFG=arr(0.2))}
    loo = {nm: {s: arr(_GS_POOLED[s]) for s in _GS_ROB} for nm in _GS_LOO}
    for key, val in (set_ or {}).items():
        variant, stat = key[0], key[1]; field = key[2] if len(key) > 2 else "point"
        for b, v in (val.items() if isinstance(val, dict) else [(b, val) for b in range(NL)]):
            if variant in pb:
                e = pb[variant][stat]
                if field == "point": e["point"][b] = v; e["lo"][b] = v - margin / 2; e["hi"][b] = v + margin / 2; e["sim_lo"][b] = v - margin; e["sim_hi"][b] = v + margin
                else: e[field][b] = v
            elif variant == "white": po["dot/white/pooled"][stat][b] = v
            elif variant == "obj": po["obj/raw/pooled"][stat][b] = v
            else: loo[variant.split("loo:", 1)[1]][stat][b] = v
    comp = list(P) if composed is None else list(composed); flat = lambda: dict(_series(top=top, bottom=bot), **_FLAT)
    ob = dict(P={"dFG": _series(top=top, bottom=bot), "rho_S|IMP-DES@pooled": flat(), "sigma_self|IMP-DES@pooled": flat(), "rho_S|IMP-DES@F1": dict(D=0.0), "rho_S|IMP-DES@F2": dict(D=0.0),
                 "sigma_self|IMP-DES@F1": dict(D=0.0), "sigma_self|IMP-DES@F2": dict(D=0.0)},
              P_composed={"FG_causal|IMP-DES": flat(), "rho_causal|IMP-DES": flat()} if len(comp) >= 3 else {"FG_causal|IMP-DES": dict(status="empty block set"), "rho_causal|IMP-DES": dict(status="empty block set")})
    for nm, o in (over or {}).items():
        tgt = ob["P_composed"] if "_causal|" in nm else ob["P"]
        if o is None: tgt[nm] = dict(status="empty block set")
        else: tgt[nm].update(o)
    inP = [b in P for b in range(NL)]
    return dict(model=M17, blocks=list(range(NL)), flags=[], mock=False, per_block=pb, point_only=po, loo=loo, over_blocks=ob,
                gate=dict(C0=inP, uncomposed=[not x for x in inP], passing=inP, P=P, n_P=len(P), G30_ok=len(P) >= MIN_P, composed=[b in comp for b in range(NL)], P_composed=[b for b in P if b in comp]),
                delta=dict(separability=dict(last=dict(ok=bool(sep), ratio=0.5 if sep else 0.1)), tau=dict(dot=dict(topic_like=[False] * NL)), mmlu=dict(last=dict(delta=[0.05 + 0.005 * b for b in range(NL)]))),
                spillover_g_self=dict(z_self=arr(4.0), z_other=arr(1.0), FR=arr(0.1), S_self=arr(1.0), S_other=arr(0.1)), r_DiD=did or dict(status="the other model's statistics are not available"))


_DESIGN = (("nat", "nat", None, 0.0, 0.0, 0), ("task_late+0.006", "late", "task", -2.0, 0.32, 1), ("task_late+0.003", "late", "task", -1.0, 0.20, 1), ("task_late-0.004", "late", "task", 1.0, 0.25, 1),
           ("task_late-0.01", "late", "task", 2.0, 0.40, 1), ("sham_late+0.01", "late", "sham", 0.55, 0.33, 0), ("sham_late-0.01", "late", "sham", -1.0, 0.31, 0), ("sham_late-0.03", "late", "sham", -1.6, 0.43, 0),
           ("sham_late-0.1", "late", "sham", -2.2, 0.54, 0), ("task_early+0.003", "early", "task", -1.5, 0.3, 1), ("task_early-0.003", "early", "task", 1.0, 0.3, 1), ("task_early-0.01", "early", "task", 1.8, 1.0, 1),
           ("sham_early+0.003", "early", "sham", 0.4, 0.3, 0), ("sham_early-0.003", "early", "sham", -0.6, 0.3, 0), ("sham_early-0.01", "early", "sham", -1.2, 1.0, 0))
_F = dict(grow=lambda x, p, k: 0.5 + 0.1 * max(x, 0), erode=lambda x, p, k: 0.5 - 0.1 * max(x, 0), flat=lambda x, p, k: 0.5, needs=lambda x, p, k: 0.5 + 0.1 * min(x, 0),
          reg=lambda x, p, k: 0.6 + 0.2 * k * min(x, 0), pert=lambda x, p, k: 0.6 - 0.6 * p, regpert=lambda x, p, k: 0.9 - 1.2 * p + 0.1 * k * min(x, 0), absent=lambda x, p, k: 0.01,
          low=lambda x, p, k: 0.04 + 0.1 * max(x, 0))


def _world(ws, lines, rng, s_m=0.01, s_w=0.01, jit=0.1, fail=(), xs=None, rho=(0.95, 0.95, 0.95, 0.8, 0.7), RG=-0.01, pool=-0.01, delta_nat=0.4):
    """rgi_toy/r1/w<ws>.json of r2_rgi_toy.py: 15 conditions x 3 inits; per model x, p, k, level1, delta_h2 = delta_nat 2^x and the four
    line values y = _F[lines[line]](x, p, k) + world effect + model noise; rho_d, RG; pooling per condition. fail: {(cond, init)} models
    failing level 1; xs: {cond: x} overrides of the design."""
    names = dict(zip(("Lambda", "S_gain", "T1_gain", "Phi"), lines)); uw = {ln: rng.normal(0, s_w) for ln in names}; conds, models = [], []
    for cond, arm, reg, x0, p0, k in _DESIGN:
        x0 = (xs or {}).get(cond, x0); conds.append(dict(cond=cond, arm=arm, reg=reg, gamma=0.0, k=k))
        for i in range(3):
            x = rng.normal(0, 0.03) if cond == "nat" else x0 + rng.normal(0, jit); p = 0.0 if cond == "nat" else max(p0 + rng.normal(0, 0.03), 0.0)
            m = dict(name=f"bp_mup_W128_w{ws}_i{i}_{cond}", ws=ws, init=i, W=128, cond=cond, arm=arm, reg=reg, k=k, x=float(x), p=float(p), level1=(cond, i) not in fail,
                     delta_h2=float(delta_nat * 2 ** x), rho=[float(r + rng.normal(0, 0.01)) for r in rho], RG=float(RG + rng.normal(0, 0.003)))
            for ln, f in names.items(): m[ln] = float(_F[f](x, p, k) + uw[ln] + rng.normal(0, s_m))
            models.append(m)
    return dict(stage="r1", world=ws, W=128, conditions=conds, models=models, delta_nat=delta_nat,
                pooling=[dict(cell=dict(ws=ws, cond=c["cond"]), gain=float(pool + rng.normal(0, 0.003))) for c in conds], pooling_world=[dict(cell=dict(ws=ws, cond="all conditions"), gain=float(pool))])


def _worlds(root, lines, n=8, seed=0, **kw):
    """Writes n worlds under <root>/r2_t1/rgi_toy/r1/ (per-world keyword overrides: kw values may be {ws: value})."""
    import shutil
    rng = np.random.default_rng(seed); d = os.path.join(root, "r2_t1", "rgi_toy", "r1"); shutil.rmtree(d, ignore_errors=True)
    for ws in range(n):
        RP.write_json(os.path.join(d, f"w{ws}.json"), _world(ws, lines, rng, **{k: (v[ws] if isinstance(v, dict) and set(v) <= set(range(n)) and ws in v else v) for k, v in kw.items()
                                                                                   if not (isinstance(v, dict) and set(v) <= set(range(n)) and ws not in v)}))
    return RP.Inputs(None, root, os.path.join(root, "out", "x.json"))


def _fam(sg=None, l1=None, relu_testable=True):
    """rgi_toy/fam.json: 7 variants x worlds 0-4 (S-gain sg[name] +- 0.02, default 0.5; l1[name] = the number of worlds whose B net passes)."""
    names = ("other init", "other split", "permuted coding", "task family x+2y+c", "8 shared shifts + spare", "3 hidden layers", "ReLU"); sg, l1 = sg or {}, l1 or {}
    return dict(stage="fam", worlds=list(range(5)), relu_calibration=dict(testable=relu_testable, lr=2e-3 if relu_testable else None), summary={},
                variants={nm: [dict(ws=w, variant=nm, level1=w < l1.get(nm, 5), S_gain=sg.get(nm, 0.5) + 0.01 * (w - 2), Lambda=0.6, T4_mean=0.7) for w in range(5)] for nm in names})


def _cross(ws, rng, s_slope=0.5, widths=(64, 128, 256, 1024)):
    """(rgi_toy/cross/w<ws>.json, pc_toy/cross_pc/w<ws>.json): BP models whose delta_h2 grows with the width, S-gain as receiver = 0.2 +
    s_slope delta_h2, Lambda = 0.6 - 0.2 delta_h2, Phi flat; one PC-muP net at W 128 and 256 and every ordered pair with a PC net."""
    bp = []
    for W in widths:
        for arm, inits in (("bp_mup", (0, 1)), ("bp_sp", (0,))):
            for i in inits:
                d = 0.3 + 0.1 * math.log2(W / 64) + rng.normal(0, 0.01)
                bp.append(dict(name=f"{arm}_W{W}_w{ws}_i{i}", ws=ws, init=i, W=W, arm="nat", kind="mlp", level1=True, delta_h2=float(d), Lambda=float(0.6 - 0.2 * d + rng.normal(0, 0.01)), Phi=float(0.02 + rng.normal(0, 0.005)),
                               as_receiver=dict(n=5, S_gain=float(0.2 + s_slope * d + rng.normal(0, 0.01)), T1_gain=float(rng.normal(0, 0.005)))))
    bpm = [dict(name=m["name"], kind="bp", arm=m["name"].split("_W")[0], W=m["W"], init=m["init"], ws=ws, level1=True, delta_h2=m["delta_h2"], Lambda=m["Lambda"], Phi=m["Phi"]) for m in bp]
    pcm = [dict(name=f"pc_mup_W{W}_s{ws}_K12_g0_wd1_T20_beta0.1", kind="pc", arm="pc_mup", W=W, init=0, ws=ws, level1=True, delta_h2=0.3 + 0.1 * math.log2(W / 64), Lambda=0.6 - 0.2 * (0.3 + 0.1 * math.log2(W / 64)), Phi=0.02) for W in (128, 256)]
    pairs = [dict(A=a["name"], B=b_["name"], ws=ws, A_W=a["W"], B_W=b_["W"], A_arm=a["arm"], B_arm=b_["arm"], A_kind=a["kind"], B_kind=b_["kind"], S_gain=0.2 + s_slope * b_["delta_h2"], t1_gain=0.0, t4_mean=0.6)
             for a in bpm + pcm for b_ in bpm + pcm if a is not b_ and "pc" in (a["kind"], b_["kind"])]
    return dict(stage="cross", world=ws, models=bp, pairs=[]), dict(stage="cross_pc", world=ws, models=bpm + pcm, pairs=pairs)


def _synth_q(root, ax, B3, K=400, seed=0, GA=None, dyn=None, spill=None):
    """Synthetic r2_q1 / r2_q3 outputs for a RP.synth_r1 tree (ax = its return value): axes/r2_axes.pt (O2 axes from the block's basis Q:
    u_RGI_perpT = Q[:, 5], u_RGI_perp = unit(0.9 Q[:, 5] + sqrt(0.19) Q[:, 7]), u_dir = Q[:, 7]; K random null axes each; GA),
    rgi/dyn_b<b>.json (dyn[b]; default: a GEN loop that sustains itself, 'not run (AUC gate)' where GA fails; None: no file),
    rgi/dirs_<m>.pt (c_o, g, s) and rgi/causal_<m>.pt (IMP-DES gradients of the model objects = spill[b] = (SELF amplitude, other-owner
    amplitude) times u_T_perp(b), plus noise)."""
    import torch
    g = torch.Generator().manual_seed(7000 + seed); rnd = lambda *sh: torch.randn(*sh, generator=g, dtype=torch.float64)
    D = int(ax[B3[0]]["Q"].shape[0]); NL = 28; AX = dict(blocks={}, GA={}, B3=list(B3), mock=False); Wc = 0.1 * rnd(60, 4, NL, D); c_o = 0.1 * rnd(NL, 48, D)
    for b in B3:
        Q = ax[b]["Q"]; ga = True if GA is None else bool(GA.get(b, True)); u = ax[b]["u_T_perp"]
        AX["blocks"][b] = dict(u_RGI_perpT=Q[:, 5].float(), u_RGI_perp=(0.9 * Q[:, 5] + math.sqrt(0.19) * Q[:, 7]).float(), u_dir=Q[:, 7].float(), GA=ga, auc_RGI=0.95 if ga else 0.6,
                               **{f"null_{k}": (0.1 * rnd(K, D)).float() for k in ("u_RGI_perp", "u_RGI_perpT", "u_dir")})
        AX["GA"][b] = ga; s_amp, o_amp = (spill or {}).get(b, (0.0, 0.0)); Wc[:12, :2, b] += s_amp * u; Wc[12:48, :2, b] += o_amp * u
        c_o[b] += 0.5 * Q[:, 5]; c_o[b, :12] += 0.3 * Q[:, 6]
        dj = (dyn or {}).get(b, _dyn() if ga else dict(script="r2_gen_loop.py", status="not run (AUC gate)", GA=False, auc_RGI=0.6)) if not (dyn and b in dyn and dyn[b] is None) else None
        if dj is not None: RP.write_json(os.path.join(root, "r2_q1", "rgi", f"dyn_b{b}.json"), dj)
    os.makedirs(os.path.join(root, "r2_q1", "axes"), exist_ok=True); torch.save(AX, os.path.join(root, "r2_q1", "axes", "r2_axes.pt"))
    gd = c_o.mean(1); ds = c_o[:, :12].mean(1); u = gd / gd.norm(dim=1, keepdim=True); sd = ds - (ds * u).sum(1, keepdim=True) * u
    os.makedirs(os.path.join(root, "r2_q3", "rgi"), exist_ok=True)
    torch.save(dict(c_o=c_o.float(), g=gd.float(), s=sd.float(), self_objects=list(range(12))), os.path.join(root, "r2_q3", "rgi", f"dirs_{M17}.pt"))
    torch.save(dict(W=Wc.float(), scalars=["IMP-DES h1", "IMP-DES h2", "CHG-DES h1", "CHG-DES h2"]), os.path.join(root, "r2_q3", "rgi", f"causal_{M17}.pt"))
    return AX


def selftest(work):
    import shutil
    import torch
    import r2_rk as R
    global N_BOOT, AT7_NSIM, AT5_BOOT
    keep = (N_BOOT, AT7_NSIM, AT5_BOOT); N_BOOT, AT7_NSIM, AT5_BOOT = 500, 60, 500; R.MOCK = True
    shutil.rmtree(work, ignore_errors=True); os.makedirs(work); res = []; t_start = time.time()

    def check(name, cond, info=""):
        res.append(dict(test=name, ok=bool(cond), info=str(info)[:300])); print(f"[{'PASS' if cond else 'FAIL'}] {name} {str(info)[:220]}", flush=True)
    try:
        # ------------------------------------------------------------------------------------------------ A7 dynamics: self loop
        sv = lambda *a, **k: self_loop_rule(_rd(*a, **k), a[0])["verdict"]
        check("self loop: drawn to general (RSI_a and RSI_b at block 14, c > 0, p <= 0.01, A >= 0.25 at d_3 and d_4, controls silent)", sv(14) == "self loop drawn to general" and sv(8) == "self loop drawn to general")
        check("self loop: a control hit (SPO / RND at d_3 or d_4, either sign) gives 'generic attractor'; a hit at d_2 does not count", sv(14, ctrl=("SPO", 3, 1)) == "generic attractor"
              and sv(14, ctrl=("RND_2", 4, -1)) == "generic attractor" and sv(8, ctrl=("RND_1", 3, -1)) == "generic attractor" and sv(14, ctrl=("RND_1", 2, 1)) == "self loop drawn to general"
              and sv(14, ctrl=("SEA", 3, 1)) == "self loop drawn to general")
        check("self loop: the mirror is 'drawn away from general improvement' (and 'generic attractor' with a control hit)", sv(14, rsi=(-1, -1)) == "drawn away from general improvement"
              and sv(14, rsi=(-1, -1), ctrl=("RND_1", 4, 1)) == "generic attractor")
        check("self loop: 'no pull' if a round or (block 14) RSI_b misses, or the signs differ", sv(14, rsi=(1, 0)) == "no pull" and sv(14, rsi=(1, -1)) == "no pull" and sv(14, only=("RSI_a",)) == "no pull"
              and sv(8, only=("RSI_a",)) == "self loop drawn to general" and sv(14, rsi=(0, 0), ctrl=("SPO", 3, 1)) == "no pull")
        weak = _rd(14); weak["RSI_a"][3]["u_RGI_perpT"] = _rr(0.5, 0.02, 0.6); weak2 = _rd(14); weak2["RSI_b"][4]["u_RGI_perpT"] = _rr(0.1, 0.001, 0.2)
        check("self loop: p = 0.02 or A = 0.2 is no hit; a counted push above 0.15 nats or a missing loop makes the rule 'not decidable'", self_loop_rule(weak, 14)["verdict"] == "no pull"
              and self_loop_rule(weak2, 14)["verdict"] == "no pull" and sv(14, bad=("RND_1",)).startswith("not decidable (damage") and sv(14, missing=("RSI_b",)).startswith("not decidable (loops missing: RSI_b"))
        # ------------------------------------------------------------------------------------------------ A7 dynamics: GEN loop
        hitT = ((0.3, 0.001),) * 4; nsT = ((0.05, 0.5),) * 4
        g1 = gen_rule(_dyn()); g2 = gen_rule(_dyn(E=0.5)); g3 = gen_rule(_dyn(rel=0.4)); g4 = gen_rule(_dyn(perp=((0.3, 0.001), (0.3, 0.02), (0.3, 0.001), (0.3, 0.001))))
        check("GEN loop: 'RGI sustains itself' needs c > 0 with p <= 0.01 at d_2..d_4 on u_RGI_perp, mean rel >= 0.5 and E_GI >= 0.7", g1["verdict"] == "RGI sustains itself" and g1["labels"] == []
              and g2["verdict"] == g3["verdict"] == g4["verdict"] == "none of the frozen statements holds" and gen_rule(_dyn(perp=((0.3, 0.001), (-0.3, 0.001), (0.3, 0.001), (0.3, 0.001))))["sustains"] is False)
        g5 = gen_rule(_dyn(perp=nsT, perpT=nsT, ut=hitT)); g6 = gen_rule(_dyn(perp=nsT, perpT=((0.05, 0.5), (0.05, 0.5), (0.05, 0.08), (0.05, 0.5)), ut=hitT)); g7 = gen_rule(_dyn(R4=0.6))
        check("GEN loop: 'RGI collapses into RSI' (p > 0.10 on u_RGI_perpT and c > 0, p <= 0.01 on u_T_perp at d_3, d_4); 'common attractor' (R_4 >= 0.5); label 'no first link'",
              g5["verdict"] == "RGI collapses into RSI" and g5["labels"] == ["no first link"] and g6["collapses_into_RSI"] is False and g7["verdict"] == "RGI sustains itself; common attractor"
              and gen_rule(_dyn(perp=((0.3, 0.02),) + ((0.3, 0.001),) * 3))["labels"] == ["no first link"], (g5["verdict"], g7["verdict"]))
        check("GEN loop: damage of the pushes d_1..d_3 (rows n = 1..3) above 0.15 nats or missing rows -> 'not decidable'; the start push (row 0) is not counted",
              gen_rule(_dyn(dmg=[0.03, 0.2, 0.03, 0.03]))["verdict"].startswith("not decidable (damage") and gen_rule(_dyn(dmg=[0.3, 0.03, 0.03, 0.03]))["verdict"] == "RGI sustains itself"
              and gen_rule(_dyn(rows=3))["verdict"] == "not decidable (GEN rows missing)")
        # ------------------------------------------------------------------------------------------------ pull and delta
        rng = np.random.default_rng(3); z = {b: rng.normal(0, 0.05, 800) for b in (14, 8, 20)}
        mono = lambda X, order=(14, 8, 20): monotone(list(order), X, {b: X[b] + z[b] for b in X}, "grows", "falls")
        m1 = mono({14: 0.1, 8: 0.2, 20: 0.5}); m2 = mono({14: 0.1, 8: 0.9, 20: 0.5}); m3 = mono({14: 0.5, 8: 0.2, 20: 0.1}); m4 = mono({14: 0.5, 8: 0.9, 20: 0.1}); m5 = mono({14: 0.3, 8: 0.3, 20: 0.31})
        check("pull and delta: 'grows' if the paired interval of X(high) - X(low) is above 0 and the middle block is not above the upper end of X(high)'s interval",
              m1["verdict"] == "grows" and m1["low"] == 14 and m1["high"] == 20 and m1["middle"] == 8 and m1["diff_ci90"][0] > 0 and m2["verdict"] == "no monotone relation" and m1["non_overlap_form"]["separated"])
        check("pull and delta: 'falls' mirrored (middle not above the upper end of X(low)'s interval); interval covering 0 -> 'no monotone relation'", m3["verdict"] == "falls" and m4["verdict"] == "no monotone relation"
              and m5["verdict"] == "no monotone relation" and not m5["non_overlap_form"]["separated"] and mono({14: 0.5, 8: 0.52, 20: 0.1})["verdict"] == "falls")
        m6 = mono({14: 0.1, 20: 0.5}); m7 = mono({20: 0.5}); m8 = mono({14: 0.1, 8: float("nan"), 20: 0.5})
        check("pull and delta: two entering blocks -> middle clause dropped (marked); fewer -> 'not decidable'; a block without a finite X does not enter", m6["verdict"] == "grows"
              and m6["middle_clause"].startswith("dropped") and m6["middle"] is None and m7["verdict"] == "not decidable" and m8["entering"] == [14, 20] and m8["verdict"] == "grows")
        # ------------------------------------------------------------------------------------------------ A3 spillover
        check("spillover rule: the five outcomes of critique 22f", spill_rule(4, 4, 0.6) == "RSI push spills over" and spill_rule(1, 4, float("nan")) == "spills over (no self response)"
              and spill_rule(4, 1, 0.1) == "stays with self" and spill_rule(1, 1, float("nan")) == "no response" and spill_rule(4, 4, 0.3) == "unclear" and spill_rule(4, 2, 0.6) == "unclear"
              and spill_rule(4, 4, 0.2) == "stays with self" and spill_rule(float("nan"), 1, 0.1) == "unclear" and spill_rule(3.0, 3.0, 0.5) == "RSI push spills over" and spill_rule(2.99, 2.99, 0.9) == "no response")
        g = torch.Generator().manual_seed(5); u = torch.randn(40, generator=g, dtype=torch.float64); u = u / u.norm(); nl = torch.randn(300, 40, generator=g, dtype=torch.float64)
        w = 0.1 * torch.randn(48, 40, generator=g, dtype=torch.float64); is_self = np.arange(48) < 12; w[:12] += 2.0 * u; w[12:] += 1.2 * u; sp = spillover(w, is_self, u, nl)
        w2 = 0.1 * torch.randn(48, 40, generator=g, dtype=torch.float64); w2[12:] += 1.2 * u; sp2 = spillover(w2, is_self, u, nl)
        try:
            import r2_rgi_stats as RS
            ref = RS.spillover(w, is_self, u, nl); same = all(abs(sp[k] - ref[k]) < 1e-12 for k in ("S_self", "S_other", "z_self", "z_other", "FR"))
        except Exception as e:                                                    # noqa: BLE001 - the producer's module needs its own imports
            same = None; print(f"  (r2_rgi_stats not importable here: {e!r})")
        check("spillover: S, z against the null axes and FR by definition (FR only where z_self >= 3); equal to r2_rgi_stats.spillover", abs(sp["S_self"] - float((w[:12] @ u).mean())) < 1e-12
              and abs(sp["FR"] - sp["S_other"] / sp["S_self"]) < 1e-12 and 0.55 < sp["FR"] < 0.65 and sp["z_self"] > 3 and sp["z_other"] > 3 and sp2["z_self"] < 3 and not fin(sp2["FR"]) and same is not False
              and spill_rule(sp["z_self"], sp["z_other"], sp["FR"]) == "RSI push spills over" and spill_rule(sp2["z_self"], sp2["z_other"], sp2["FR"]) == "spills over (no self response)",
              (round(sp["z_self"], 1), round(sp["z_other"], 1), round(sp["FR"], 3), round(sp2["z_self"], 2), same))
        # ------------------------------------------------------------------------------------------------ A7 per block
        base = a7_blocks(_gs()); Bk = base["blocks"]
        check("A7 per block: defaults give GI, IS, BA, robust 'RGI geometry present', 'absorbed' (both frames, LR column), causal-general and causal absorption at every block of P",
              base["P"] == list(range(8, 24)) and all(v["GI"] and v["IS"] and v["BA"] and v["present"] and v["self"] == "absorbed" and v["self_LR"] == "absorbed" and v["composed"] and v["causal_general"]
                                                      and v["causal_absorption"] for v in Bk.values()) and Bk["8"]["labels"] == ["frame-consistent", "object-consistent"] and len(Bk["8"]["robustness"]) == 8)
        S_ = {("pooled", "FG|IMP-DES"): {10: 0.45}, ("pooled", "FG|IMP-DES", "sim_lo"): {11: 0.30}, ("pooled", "dFG"): {12: 0.05, 14: 0.05}, ("pooled", "cos_valence"): {13: 0.85}, ("pooled", "FG|IMP-CHG"): {14: 0.4},
              ("pooled", "dFG_val"): {15: 0.02}, ("pooled", "P_cell|IMP-CHG"): {16: 0.03}, ("white", "FG|IMP-DES"): {17: 0.3}, ("loo:owner:SELF", "dFG"): {18: 0.05}, ("loo:owner:SELF", "FG|IMP-CHG"): {18: 0.3},
              ("pooled", "q|IMP-CHG"): {19: 0.1}, ("pooled", "dFG", "sim_lo"): {20: -0.01}}
        v = a7_blocks(_gs(set_=S_))["blocks"]
        check("A7 per block: GI needs FG >= 0.5 with the simultaneous lower bound >= 0.35", not v["10"]["GI"] and not v["10"]["present"] and not v["11"]["GI"] and v["9"]["GI"])
        check("A7 per block: 'present' = GI and (IS or BA): IS fails / BA holds; BA fails (valence) / IS holds; both fail", not v["12"]["IS"] and v["12"]["BA"] and v["12"]["present"] and v["13"]["IS"] and not v["13"]["BA"]
              and v["13"]["present"] and "valence-dominated" in v["13"]["labels"] and v["14"]["GI"] and not v["14"]["IS"] and not v["14"]["BA"] and not v["14"]["present"])
        check("A7 per block: IS needs the lower bounds of dFG and dFG_val above 0; BA needs C0 for IMP-CHG (simultaneous bound; the pointwise one is reported) and q >= 0.2",
              not v["15"]["IS"] and not v["20"]["IS"] and v["15"]["present"] and not v["16"]["BA"] and v["16"]["BA_C0_IMP_CHG"] == dict(simultaneous=False, pointwise=True) and not v["19"]["BA"] and v["19"]["present"])
        check("A7 per block: the point estimates must keep 'GI and (IS or BA)' in the whitened metric and under every leave-one-out", v["17"]["core"] and not v["17"]["present"] and not v["17"]["robustness"]["whitened"]["keeps"]
              and v["18"]["core"] and not v["18"]["present"] and not v["18"]["robustness"]["owner:SELF"]["keeps"] and v["18"]["robustness"]["owner:SMALL"]["keeps"])
        L_ = {("pooled", "FG|CHG-DES"): {9: 0.6}, ("pooled", "OG|IMP-DES"): {9: 0.6}, ("pooled", "SG|IMP-DES"): {9: 0.6}, ("pooled", "OG_LR|IMP-DES"): {9: 0.6}, ("pooled", "cos_quality"): {9: -0.6},
              ("F1", "dFG"): {9: -0.1}, ("obj", "dFG"): {9: -0.1}, ("F1", "sigma_self|IMP-DES"): {20: 0.3, 22: 0.3}, ("F2", "sigma_self|IMP-DES"): {20: 0.3}, ("F1", "dsigma"): {20: 0.1, 22: 0.1},
              ("F2", "dsigma"): {20: 0.1}, ("F1", "rho_S|IMP-DES"): {21: 0.3}, ("F2", "sigma_self|IMP-DES", "sim_hi"): {23: 0.16}, ("F1", "sigma_self_LR|IMP-DES"): {19: 0.3}, ("F2", "sigma_self_LR|IMP-DES"): {19: 0.3},
              ("F1", "dsigma_LR"): {19: 0.1}, ("F2", "dsigma_LR"): {19: 0.1}, ("causal", "FG|IMP-DES"): {10: 0.4}, ("causal", "dFG", "sim_lo"): {11: -0.01}, ("causal", "sigma_self|IMP-DES"): {12: 0.2},
              ("causal", "rho_S|IMP-DES"): {13: 0.4}}
        v = a7_blocks(_gs(set_=L_, composed=[b for b in range(8, 24) if b != 15]))["blocks"]
        check("A7 labels (no effect on verdicts): general editing, owner- / substrate- / recursive-general, signed quality axis; frame- and object-consistency",
              v["9"]["labels"] == ["general editing", "owner-general", "substrate-general", "recursive-general", "signed quality axis"] and v["9"]["present"])
        check("A7 self: 'self-specific' needs both frames (lower bounds of sigma_self > 0.15 and of dsigma > 0); 'absorbed' needs rho_S >= 0.5 and the upper bound of sigma_self < 0.15 in both; else 'unclear'; LR column apart",
              v["20"]["self"] == "self-specific" and v["21"]["self"] == "unclear" and v["22"]["self"] == "unclear" and v["23"]["self"] == "unclear" and v["8"]["self"] == "absorbed" and v["19"]["self"] == "absorbed"
              and v["19"]["self_LR"] == "self-specific" and v["20"]["self_LR"] == "absorbed")
        check("A7 causal (composed blocks of P only): causal-general = FG_causal >= 0.5 with dFG_causal's lower bound > 0; absorption = rho_causal >= 0.5 with sigma_causal's upper bound < 0.15",
              v["10"]["causal_general"] is False and v["11"]["causal_general"] is False and v["12"]["causal_absorption"] is False and v["13"]["causal_absorption"] is False and v["8"]["causal_general"] is True
              and v["12"]["causal_general"] is True and v["15"]["composed"] is False and v["15"]["causal_general"] is None and v["15"]["causal_absorption"] is None)
        # ------------------------------------------------------------------------------------------------ A7 over blocks
        ov = lambda gs06=None, **k: (lambda gs: a7_over(gs, a7_blocks(gs), gs06))(_gs(**k))
        o1 = ov(); P16 = list(range(8, 24))
        check("V1: grows with delta beyond depth (present at >= 2 top-third blocks, D, Spearman, whitened D, separability, r3, nuisance r3)", o1["code"] == "V1" and o1["verdict"] == "V1: RGI geometry grows with delta beyond depth"
              and o1["top"] == [19, 20, 21, 22, 23] and o1["bottom"] == [8, 9, 10, 11, 12] and o1["n_present_top"] == 5, o1["verdict"])
        o6 = dict(gate=dict(P=P16, n_P=16), r_DiD=dict(r_DiD=0.4, p_pos=0.03, p_neg=0.97, blocks=P16)); o6n = dict(gate=dict(P=P16, n_P=16), r_DiD=dict(r_DiD=-0.4, p_pos=0.9)); o6x = dict(gate=dict(P=[], n_P=0), r_DiD=dict(status="fewer than 4 common passing blocks", r_DiD=float("nan")))
        check("V1 with the 0.6B arm: needs r_DiD > 0 (else V2); r_DiD may be stored in either model's statistics", ov(gs06=o6)["code"] == "V1" and ov(gs06=o6n)["code"] == "V2" and ov(gs06=o6x)["code"] == "V2"
              and ov(gs06=dict(gate=dict(P=P16)), did=dict(r_DiD=0.3, p_pos=0.04))["code"] == "V1" and ov(gs06=o6)["arm_0p6B"]["ran"] and not o1["arm_0p6B"]["ran"])
        v2 = [ov(sep=False), ov(over={"dFG": dict(r3_p_pos=0.2)}), ov(over={"dFG": dict(r3_ci=[-0.1, 0.8])}), ov(over={"dFG": dict(r3_nuis_p_pos=0.2)}), ov(over={"dFG": dict(r3=-0.1)}), ov(over={"dFG": dict(r3_nuis=-0.2)})]
        check("V2: with delta or depth (V1 without separability, r3, nuisance r3)", all(o["code"] == "V2" and o["verdict"] == "V2: RGI geometry grows with delta or depth" for o in v2))
        notv = [ov(over={"dFG": dict(D=0.08)}), ov(over={"dFG": dict(D_ci=[-0.01, 0.3])}), ov(over={"dFG": dict(spearman_mmlu=0.4)}), ov(over={"dFG": dict(spearman_mmlu_ci=[-0.1, 0.9])}), ov(over={"dFG": dict(spearman_obj=-0.1)}),
                ov(over={"dFG": dict(D_whitened=-0.02)}), ov(set_={("pooled", "FG|IMP-DES"): {b: 0.3 for b in (20, 21, 22, 23)}})]
        check("V1 / V2 need every over-block condition (D >= 0.10 with lower bound > 0, Spearman >= 0.5 with lower bound > 0, Spearman(delta_OBJ) > 0, whitened D > 0, >= 2 top-third blocks): else V6",
              all(o["code"] == "V6" and o["verdict"] == "V6: undecided" for o in notv), [o["code"] for o in notv])
        o3 = ov(over={"dFG": _FALL}); o3b = ov(over={"dFG": _FALL}, sep=False); o3c = ov(over={"dFG": _FALL}, set_={("pooled", "FG|IMP-DES"): {b: 0.3 for b in (9, 10, 11, 12)}})
        check("V3: falls with delta beyond depth / with delta or depth (the mirror, RGI geometry present at >= 2 bottom-third blocks)", o3["verdict"] == "V3: RGI geometry falls with delta beyond depth" and o3["code"] == "V3"
              and o3b["verdict"] == "V3: RGI geometry falls with delta or depth" and o3c["code"] == "V6" and ov(gs06=o6, over={"dFG": _FALL})["verdict"].endswith("or depth") and ov(gs06=o6n, over={"dFG": _FALL})["code"] == "V3")
        o4 = ov(over={"dFG": _FLAT}); o4b = ov(over={"dFG": dict(_FLAT, spearman_mmlu=0.35)}); o4c = ov(over={"dFG": dict(_FLAT, D_ci=[-0.12, 0.05])})
        check("V4: present at >= 2 blocks, |Spearman| < 0.3 and D's interval inside (-0.10, +0.10)", o4["verdict"] == "V4: RGI geometry present but not tied to delta" and o4b["code"] == "V6" and o4c["code"] == "V6"
              and ov(over={"dFG": _FLAT}, set_={("pooled", "FG|IMP-DES"): {b: 0.3 for b in range(9, 24)}})["code"] == "V6")
        o5 = ov(set_={("pooled", "dFG"): 0.05, ("pooled", "FG|IMP-CHG"): 0.4}); o5b = ov(set_={("pooled", "FG|IMP-DES"): 0.3}); o5c = ov(set_={("white", "FG|IMP-DES"): 0.3})
        check("V5: no RGI geometry ('general editing, not improvement-specific' if GI holds somewhere but IS and BA fail there; 'target-specific' if GI fails everywhere; not robust: stated)",
              o5["verdict"] == "V5: no RGI geometry (general editing, not improvement-specific)" and o5b["verdict"] == "V5: no RGI geometry (target-specific)" and o5c["code"] == "V5" and "whitened" in o5c["verdict"]
              and o5["present_blocks"] == [], (o5["verdict"], o5b["verdict"], o5c["verdict"]))
        ou = ov(P=tuple(range(8, 18)))
        check("fewer than 12 blocks in P: 'undecided (fewer than 12 passing blocks)' for every over-block line", ou["verdict"] == "undecided (fewer than 12 passing blocks)" and ou["rsi_into_rgi"]["verdict"] == ou["verdict"]
              and ou["causal_line"]["verdict"] == ou["verdict"] and ov(P=tuple(range(8, 20)))["code"] == "V1")
        GROW = dict(D=0.2, D_ci=[0.05, 0.3], r3=0.6, r3_p_pos=0.01, r3_ci=[0.2, 0.8], r3_nuis=0.5, r3_nuis_p_pos=0.02)
        r1_ = ov(over={"rho_S|IMP-DES@pooled": GROW}); r2_ = ov(over={"rho_S|IMP-DES@pooled": GROW}, sep=False); r3_ = ov(over={"rho_S|IMP-DES@pooled": GROW}, set_={("F1", "rho_S|IMP-DES"): {b: 0.3 for b in (19, 20, 21, 22)}})
        spec = {("F1", "sigma_self|IMP-DES"): {b: 0.3 for b in (21, 22, 23)}, ("F2", "sigma_self|IMP-DES"): {b: 0.3 for b in (21, 22, 23)}, ("F1", "dsigma"): {b: 0.1 for b in (21, 22, 23)}, ("F2", "dsigma"): {b: 0.1 for b in (21, 22, 23)}}
        r4_ = ov(over={"sigma_self|IMP-DES@pooled": dict(GROW, r3_p_pos=0.3)}, set_=spec); r5_ = ov(over={"sigma_self|IMP-DES@pooled": GROW}); r6_ = ov(over={"rho_S|IMP-DES@pooled": dict(D=0.08, D_ci=[-0.02, 0.2])})
        check("RSI-into-RGI: 'dissolves' (D(rho_S) >= 0.10, lower bound > 0, absorbed at >= 2 top-third blocks), 'beyond depth' only with the depth conditions on the same series",
              r1_["rsi_into_rgi"]["verdict"] == "RSI dissolves into RGI as delta grows (beyond depth)" and r2_["rsi_into_rgi"]["verdict"] == "RSI dissolves into RGI as delta grows (with delta or depth)"
              and r3_["rsi_into_rgi"]["verdict"] == "undecided" and r3_["rsi_into_rgi"]["absorbed_top"] == 1)
        check("RSI-into-RGI: 'stays its own direction / grows apart' (D(sigma_self), self-specific at >= 2 top-third blocks); 'no relation' (both intervals inside +-0.10); else 'undecided'",
              r4_["rsi_into_rgi"]["verdict"] == "RSI stays its own direction / grows apart (with delta or depth)" and r4_["rsi_into_rgi"]["self_specific_top"] == 3 and r5_["rsi_into_rgi"]["verdict"] == "undecided"
              and o1["rsi_into_rgi"]["verdict"] == "no relation" and r6_["rsi_into_rgi"]["verdict"] == "undecided", (r4_["rsi_into_rgi"]["verdict"], r5_["rsi_into_rgi"]["verdict"]))
        c1_ = ov(over={"FG_causal|IMP-DES": GROW}); c2_ = ov(over={"FG_causal|IMP-DES": GROW}, sep=False); c3_ = ov(over={"FG_causal|IMP-DES": _FALL}); c4_ = ov(over={"FG_causal|IMP-DES": dict(_FALL, r3_p_neg=0.3)})
        check("causal line: grows / falls with delta or depth, 'beyond depth' only with the depth conditions on FG_causal; 'no relation'; too few composed blocks",
              c1_["causal_line"]["verdict"] == "causal generality grows with delta beyond depth" and c2_["causal_line"]["verdict"] == "causal generality grows with delta or depth"
              and c3_["causal_line"]["verdict"] == "causal generality falls with delta beyond depth" and c4_["causal_line"]["verdict"] == "causal generality falls with delta or depth"
              and o1["causal_line"]["verdict"] == "no relation" and ov(composed=[8, 9])["causal_line"]["verdict"].startswith("not computable") and ov(over={"FG_causal|IMP-DES": dict(D=0.2, D_ci=[-0.01, 0.3])})["causal_line"]["verdict"] == "no relation")
        gsd = _gs(over={"FG_causal|IMP-DES": dict(D=0.2, D_ci=[0.12, 0.28], top=[19, 20, 21, 22, 23], bottom=[8, 9, 10, 11, 12]), "rho_causal|IMP-DES": dict(D=0.0, D_ci=[-0.08, 0.08], top=[19, 20, 21, 22, 23], bottom=[8, 9, 10, 11, 12])})
        rg = np.random.default_rng(21); dr = os.path.join(work, "draws", "r2_q3", "rgi"); os.makedirs(dr, exist_ok=True); zf = 0.05 * rg.standard_normal((1000, 1)); zr = 0.04 * rg.standard_normal((1000, 1))
        fgd = 0.5 + 0.01 * rg.standard_normal((1000, 28)); fgd[:, 19:24] += 0.2 + zf; rhd = 0.5 + 0.01 * rg.standard_normal((1000, 28)); rhd[:, 19:24] += zr
        np.savez(os.path.join(dr, f"geom_draws_{M17}.npz"), _meta=np.array("synthetic"), **{"causal__FG|IMP-DES": fgd.astype(np.float32), "causal__rho_S|IMP-DES": rhd.astype(np.float32)})
        idr = RP.Inputs(None, os.path.join(work, "draws"), None); cs = causal_sim(idr, gsd); csl = a7_over(gsd, a7_blocks(gsd), None, cs)["causal_line"]; sdD = float((fgd[:, 19:24].mean(1) - fgd[:, 8:13].mean(1)).std(ddof=1))
        gsb = _gs(over={"FG_causal|IMP-DES": dict(D=0.1, D_ci=[0.02, 0.18], top=[19, 20, 21, 22, 23], bottom=[8, 9, 10, 11, 12]), "rho_causal|IMP-DES": dict(D=0.0, D_ci=[-0.08, 0.08], top=[19, 20, 21, 22, 23], bottom=[8, 9, 10, 11, 12])})
        csb = a7_over(gsb, a7_blocks(gsb), None, causal_sim(idr, gsb))["causal_line"]
        check("causal line 'with simultaneous intervals': sup-t over D(FG_causal) and D(rho_causal) from the per-draw series (critical value between 1.96 and 2.5 for two statistics); the pointwise reading is reported next to it",
              cs["status"] == "ran" and 1.96 < cs["crit"] < 2.5 and abs((cs["FG_causal|IMP-DES"][1] - cs["FG_causal|IMP-DES"][0]) / 2 - cs["crit"] * sdD) < 1e-6 and abs(sum(cs["FG_causal|IMP-DES"]) / 2 - 0.2) < 1e-9
              and csl["verdict"].startswith("causal generality grows") and csl["intervals"].startswith("simultaneous") and csb["verdict"] == "no relation" and csb["verdict_with_pointwise_intervals"].startswith("causal generality grows")
              and csb["D_FG_causal_sim"][0] < 0 < csb["D_FG_causal_ci"][0] and o1["causal_line"]["intervals"].startswith("pointwise 95 % (") and causal_sim(RP.Inputs(None, os.path.join(work, "empty"), None), gsd)["status"].startswith("not available"),
              (round(cs["crit"], 3), cs["FG_causal|IMP-DES"], csb["D_FG_causal_sim"], csb["verdict"]))
        gss = _gs(over={"FG_causal|IMP-DES": dict(gsd["over_blocks"]["P_composed"]["FG_causal|IMP-DES"], D_sim_ci=list(cs["FG_causal|IMP-DES"]), D_sim_crit=cs["crit"]),
                        "rho_causal|IMP-DES": dict(gsd["over_blocks"]["P_composed"]["rho_causal|IMP-DES"], D_sim_ci=list(cs["rho_causal|IMP-DES"]), D_sim_crit=cs["crit"])})
        css = a7_over(gss, a7_blocks(gss), None, cs)["causal_line"]; gsx = _gs(over={"FG_causal|IMP-DES": dict(gsb["over_blocks"]["P_composed"]["FG_causal|IMP-DES"], D_sim_ci=[0.01, 0.19], D_sim_crit=2.0)})
        csx = a7_over(gsx, a7_blocks(gsx), None, causal_sim(idr, gsx))["causal_line"]; cs0 = a7_over(gsx, a7_blocks(gsx), None, None)["causal_line"]
        check("causal line: r2_rgi_stats.py's stored D_sim_ci is used (r2_ana's recomputation from the draws is a cross-check); without the draws file the stored interval alone",
              css["simultaneous_source"].startswith("r2_rgi_stats") and css["simultaneous_crosscheck"]["agrees"] and css["verdict"] == csl["verdict"]
              and csx["simultaneous_source"].startswith("r2_rgi_stats") and csx["verdict"].startswith("causal generality grows") and csx["simultaneous_crosscheck"]["agrees"] is False
              and cs0["simultaneous_source"].startswith("r2_rgi_stats") and cs0["D_FG_causal_sim"] == [0.01, 0.19] and cs0["simultaneous_crosscheck"] is None,
              (css["simultaneous_crosscheck"], csx["simultaneous_crosscheck"]))
        # ------------------------------------------------------------------------------------------------ combined statement
        cb = lambda dyn=None, order=(14, 8, 20), gs06=None, **k: (lambda gs: (lambda per: combined(per, a7_over(gs, per, gs06), dyn or {}, list(order)))(a7_blocks(gs)))(_gs(**k))
        dd = lambda hi, lo: {"14": dict(enters=True, self_loop=dict(verdict=lo)), "8": dict(enters=False), "20": dict(enters=True, self_loop=dict(verdict=hi))}
        k1 = cb(); k2 = cb(sep=False); k3 = cb(dyn=dd("self loop drawn to general", "no pull")); k4 = cb(dyn=dd("self loop drawn to general", "self loop drawn to general")); k5 = cb(dyn=dd("no pull", "no pull"))
        check("combined: 'the geometry forces RGI' with V1 (V2: 'with delta or depth'), present, absorbed and causal at >= 2 top-third blocks; dynamics clause only where two blocks entered",
              k1["statement"] == "the geometry forces RGI" and k1["forces"] and k2["statement"] == "the geometry forces RGI (with delta or depth)" and k3["forces"] and not k4["forces"] and not k5["forces"]
              and k4["conditions"]["dynamics"] is False and cb(dyn={"20": dict(enters=True, self_loop=dict(verdict="no pull"))})["forces"] and "not applied" in k1["dynamics_clause"])
        nc = cb(set_={("causal", "FG|IMP-DES"): {b: 0.4 for b in (20, 21, 22, 23)}}); na = cb(set_={("F1", "rho_S|IMP-DES"): {b: 0.3 for b in (20, 21, 22, 23)}}); n5 = cb(set_={("pooled", "dFG"): 0.05, ("pooled", "FG|IMP-CHG"): 0.4})
        n5b = cb(set_={("pooled", "FG|IMP-DES"): 0.3}); ns = cb(over={"sigma_self|IMP-DES@pooled": GROW}, set_={k_: {**v_, 20: v_[21]} for k_, v_ in spec.items()}); ns3 = cb(over={"sigma_self|IMP-DES@pooled": GROW}, set_=spec); nu = cb(P=tuple(range(8, 18))); n4 = cb(over={"dFG": _FLAT})
        check("combined, otherwise: the report names what holds ('represented but not causal', 'general editing, not improvement', 'target-specific', 'RSI stays its own direction', the over-block verdict)",
              not nc["forces"] and "represented but not causal" in nc["holds"] and nc["statement"].startswith("the geometry does not force RGI: V1") and not na["forces"] and na["conditions"]["absorbed_top"] is False
              and "general editing, not improvement" in n5["holds"] and any(h.startswith("target-specific") for h in n5b["holds"]) and "RSI stays its own direction" in ns["holds"] and not ns["forces"] and ns["counts"]["absorbed_top"] == 1 and ns3["forces"] and ns3["counts"]["absorbed_top"] == 2
              and nu["holds"] == ["undecided (fewer than 12 passing blocks)"] and n4["holds"][0].startswith("V4") and "self-improvement absorbed at the top-third blocks" in n4["holds"], (nc["statement"][:90], ns["holds"]))
        # ------------------------------------------------------------------------------------------------ AT8: worlds, lines, answers
        tw = os.path.join(work, "toy"); A8 = lambda lines, **k: at8(_worlds(tw, lines, **k), {})
        a_exp, W_exp, V_exp = A8(("regpert", "regpert", "absent", "absent")); L = a_exp["lines"]["late"]["lines"]
        check("AT8 (the plan's expectation): Lambda and S 'regulariser-specific (not delta)' and perturbation-driven, F ABSENT -> 'not delta: the shape of the space matters, not its size'",
              a_exp["answer"] == "not delta: the shape of the space matters, not its size" and V_exp == list(range(8)) and a_exp["G38"]["ok"] and a_exp["G39"]["ok"] and L["Lambda"]["flags"]["regulariser_specific"]
              and L["Lambda"]["flags"]["perturbation_driven"] and L["S"]["verdict"].startswith("regulariser-specific") and L["T1"]["verdict"] == "ABSENT" and L["Phi"]["verdict"] == "ABSENT"
              and a_exp["lines"]["late"]["F"]["absent_or_flat"] and a_exp["lines"]["early"]["role"] == "secondary", (a_exp["answer"], L["Lambda"]["rule_verdict"], L["T1"]["verdict"], a_exp["answer_early_arm"]))
        ans = {}
        for nm, lines in (("forces", ("grow",) * 4), ("permits", ("grow", "grow", "flat", "absent")), ("needs", ("needs", "needs", "flat", "flat")), ("erodes", ("erode", "erode", "flat", "flat")),
                          ("pert", ("pert", "pert", "absent", "absent")), ("none", ("absent",) * 4), ("mixed", ("grow", "flat", "flat", "flat")), ("halfF", ("grow", "grow", "grow", "flat")), ("g39", ("low", "grow", "grow", "grow"))):
            ans[nm] = A8(lines)[0]
        lv = lambda nm, ln: ans[nm]["lines"]["late"]["lines"][ln]
        check("AT8 lines: GROWS, ERODES, FLAT beyond, NEEDS, perturbation-driven (with the b- interval inside +-0.05), ABSENT from the per-world fits", lv("forces", "Lambda")["rule_verdict"] == "GROWS"
              and lv("erodes", "S")["rule_verdict"] == "ERODES" and lv("needs", "Lambda")["rule_verdict"] == "FLAT beyond + NEEDS" and lv("mixed", "S")["rule_verdict"] == "FLAT beyond"
              and lv("pert", "Lambda")["flags"]["perturbation_driven"] and lv("pert", "Lambda")["flags"]["flat"] and lv("pert", "Lambda")["flags"]["b_minus_inside"] and not lv("forces", "Lambda")["flags"]["perturbation_driven"]
              and lv("none", "Phi")["verdict"] == "ABSENT" and lv("none", "Phi")["flags"]["absent"] and abs(lv("forces", "Lambda")["reduced"]["b+"]["mean"] - 0.1) < 0.02 and abs(lv("pert", "S")["reduced"]["e"]["mean"] + 0.6) < 0.1,
              {k: lv(k, "Lambda")["rule_verdict"] for k in ("forces", "erodes", "needs", "pert")})
        check("AT8 answers: forces / permits / needs / erodes / not delta (perturbation-driven) / no RGI geometry", ans["forces"]["answer"] == "delta builds a geometry that forces RGI"
              and ans["permits"]["answer"] == "delta permits RGI but does not force it" and ans["needs"]["answer"] == "RGI needs task separation up to the natural level, more adds nothing"
              and ans["erodes"]["answer"] == "more delta erodes it" and ans["pert"]["answer"] == "not delta: the shape of the space matters, not its size" and ans["none"]["answer"] == "no RGI geometry",
              {k: v["answer"] for k, v in ans.items()})
        check("AT8 answers: Lambda and S must agree (else UNDECIDED); F GROWS needs Phi and the free T1 gain; machinery gate G39 (Lambda not present at natural delta) -> the recipe line is not testable",
              ans["mixed"]["answer"] == "UNDECIDED" and ans["halfF"]["answer"] == "UNDECIDED" and not ans["halfF"]["lines"]["late"]["F"]["grows"] and not ans["g39"]["G39"]["ok"]
              and ans["g39"]["answer"].startswith("UNDECIDED (recipe line not testable") and "not_testable" in lv("g39", "Lambda") and not ans["none"]["G39"]["ok"], (ans["mixed"]["answer"], ans["g39"]["answer"]))
        f = lambda **k: dict(flags=dict(dict(regulariser_specific=False, grows=False, erodes=False, flat=False, needs=False, perturbation_driven=False, absent=False, b_minus_inside=False), **k))
        Gr, Fl, Ab, Er, Ne, Rg, Pd = f(grows=True), f(flat=True), f(absent=True), f(erodes=True), f(needs=True, flat=True), f(regulariser_specific=True), f(perturbation_driven=True, flat=True, b_minus_inside=True)
        check("AT8 answer mapping on flags: every branch, first match in the plan's order, no OR across lines", answer(Gr, Gr, Gr, Gr, True) == "delta builds a geometry that forces RGI"
              and answer(Gr, Gr, Ab, Fl, True) == "delta permits RGI but does not force it" and answer(Gr, Gr, Gr, Ab, True) == "UNDECIDED" and answer(Ne, Ne, Ab, Ab, True).startswith("RGI needs task separation")
              and answer(Er, Er, Gr, Gr, True) == "more delta erodes it" and answer(Rg, Pd, Ab, Ab, True).startswith("not delta") and answer(Pd, Pd, Gr, Gr, True).startswith("not delta")
              and answer(Rg, Gr, Ab, Ab, True) == "UNDECIDED" and answer(f(perturbation_driven=True, flat=True), Rg, Ab, Ab, True) == "UNDECIDED" and answer(Ab, Ab, Ab, Ab, False) == "no RGI geometry"
              and answer(Ab, Gr, Gr, Gr, False).startswith("UNDECIDED (recipe line not testable") and answer(Ne, f(needs=True), Ab, Ab, True) == "UNDECIDED" and answer(Gr, f(grows=True, absent=True), Ab, Ab, True) == "UNDECIDED")
        # validity, G38
        bad_span = {c: x for c, x in (("task_late+0.006", -1.0), ("task_late+0.003", -0.5), ("task_late-0.004", 0.5), ("task_late-0.01", 1.0))}
        a38, _, v38 = A8(("grow",) * 4, xs={0: bad_span, 1: bad_span, 2: bad_span}); val = a38["validity"]
        check("AT8 validity: the late task arm must span >= x8 in delta_h2; fewer than 6 valid worlds -> 'UNDECIDED (probe failed)' (G38), recursion not decidable", v38 == [3, 4, 5, 6, 7] and not val["0"]["valid"]
              and val["0"]["keeps_2_of_3"] and 3.0 < val["0"]["span"] < 6.0 and val["3"]["span"] > 8 and not a38["G38"]["ok"] and a38["answer"] == "UNDECIDED (probe failed)" and a38["answer_early_arm"] == "UNDECIDED (probe failed)"
              and a38["recursion"]["verdict"] == "not decidable (probe failed, G38)", (v38, round(val["0"]["span"], 2), a38["answer"]))
        a2w, _, v2w = A8(("grow",) * 4, fail={0: {("sham_late-0.1", 0), ("sham_late-0.1", 1)}, 1: {("task_late+0.003", 2), ("nat", 0)}})
        check("AT8 validity: models failing level 1 are dropped and counted; a condition keeping fewer than 2 of its 3 inits makes the world invalid", v2w == [1, 2, 3, 4, 5, 6, 7] and a2w["validity"]["0"]["dropped"] == 2
              and not a2w["validity"]["0"]["keeps_2_of_3"] and a2w["validity"]["1"]["valid"] and a2w["validity"]["1"]["dropped"] == 2 and a2w["validity"]["1"]["kept"]["nat"] == 2
              and a2w["answer"] == "delta builds a geometry that forces RGI" and a2w["lines"]["late"]["lines"]["Lambda"]["n_worlds"] == 7)
        # recursion
        rc = lambda **k: A8(("grow", "grow", "absent", "absent"), **k)[0]["recursion"]
        rc1 = rc(); rc2 = rc(rho=(0.95, 0.95, 0.85, 0.8, 0.7)); rc3 = rc(rho=(0.5,) * 5, RG=0.02); rc4 = rc(rho=(0.5,) * 5, pool=0.02); rc5 = A8(("absent",) * 4, rho=(0.5,) * 5)[0]["recursion"]
        check("AT8 recursion: 'recursive' by rho_d (t lower bound >= 0.9 for d = 1, 2, 3), by RG > 0 or by the pooling gain > 0; else 'one-step portable' if Lambda or S is present; else neither",
              rc1["verdict"] == "recursive" and rc1["conditions"] == dict(rho=True, RG=False, pooling=False) and rc2["verdict"] == "one-step portable" and rc2["present"]["Lambda"] and rc3["verdict"] == "recursive"
              and rc3["conditions"]["RG"] and rc4["verdict"] == "recursive" and rc4["conditions"]["pooling"] and rc5["verdict"].startswith("neither recursive nor one-step portable"), (rc1["verdict"], rc2["verdict"], rc5["verdict"]))
        check("AT8 without world files: 'not run'", at8(RP.Inputs(None, os.path.join(work, "empty"), os.path.join(work, "empty", "o.json")), {})[0]["answer"].startswith("not run (round-2 output missing"))
        # family
        fam = lambda **k: at4((RP.write_json(os.path.join(work, "fam", "r2_t1", "rgi_toy", "fam.json"), _fam(**k)), RP.Inputs(None, os.path.join(work, "fam"), None))[1])
        f1 = fam(); f2 = fam(sg={"3 hidden layers": 0.08}); f3 = fam(sg={"3 hidden layers": 0.08}, l1={"3 hidden layers": 3}); f4 = fam(sg={"ReLU": 0.02}, relu_testable=False); f5 = fam(l1={n_: 2 for n_ in f1["variants"]})
        check("AT8 family: 'portable across these families' if the S-gain lower bound is > 0.10 in every variant that trains (>= 4 of 5 worlds' B nets pass level 1); ReLU 'not testable' without its calibration",
              f1["verdict"].startswith("portable across these families") and len(f1["variants_that_train"]) == 7 and f2["verdict"] == "not portable across every family that trains: 3 hidden layers"
              and f3["verdict"].startswith("portable") and "3 hidden layers" not in f3["variants_that_train"] and f4["verdict"].startswith("portable") and f4["not_testable"] == ["ReLU"]
              and f5["verdict"] == "not decidable (no variant trains)" and fam(sg={"other split": 0.11})["verdict"].startswith("not portable") and at4(RP.Inputs(None, os.path.join(work, "empty"), None))["verdict"].startswith("not run"),
              (f1["verdict"], f2["verdict"], f5["verdict"]))
        # ------------------------------------------------------------------------------------------------ AT7, AT5
        s7 = at7(W_exp, V_exp, a_exp); e7 = s7["arms"]["late"]["Lambda"]; n7 = s7["arms"]["late"]["T1"]["noise"]             # T1 is flat here: its s_m is the model noise (0.01), jit the x jitter (0.1)
        check("AT7: r2_sim.reg2_simulate with the measured noise (s_m, s_wc, s_w, jit, the measured design grid), six scenarios per line and arm; the observed verdict next to the rates",
              s7["status"] == "ran" and s7["nsim"] == 60 and len(s7["scenarios"]) == 6 and e7["status"] == "ran" and len(e7["design"]) == 9 and len(s7["arms"]["early"]["S"]["design"]) == 7 and 0.007 < n7["s_m"] < 0.013
              and n7["s_wc"] < 0.006 and 0.003 < n7["s_w"] < 0.02 and 0.08 < n7["jit"] < 0.12 and e7["noise"]["s_m"] > n7["s_m"]
              and all(abs(sum(r.values()) - 1) < 1e-6 for r in e7["rates"].values()) and e7["observed_verdict"].startswith("regulariser-specific") and e7["rates"]["true growth 0.10/doubling above natural + perturbation"].get("GROWS", 0) > 0.8
              and s7["flags"] == [] and at7({}, [], {})["status"].startswith("not run"), (n7, e7["rates"]["true growth 0.10/doubling above natural + perturbation"]))
        s7b = at7({k: W_exp[k] for k in (0, 1, 2)}, [0], a_exp)
        check("AT7 with fewer than 2 valid worlds: the noise is measured on all worlds and flagged", s7b["flags"] and s7b["worlds"] == [0, 1, 2] and s7b["arms"]["late"]["S"]["status"] == "ran")
        rng = np.random.default_rng(11); cr = os.path.join(work, "cross")
        for ws in range(5):
            a_, b_ = _cross(ws, rng); RP.write_json(os.path.join(cr, "r2_t1", "rgi_toy", "cross", f"w{ws}.json"), a_); RP.write_json(os.path.join(cr, "r2_pc", "pc_toy", "cross_pc", f"w{ws}.json"), b_)
        pcg = dict(widths_G_PC=[128], G_PC={"128": dict(status="usable"), "256": dict(status="BP-equivalent (not used for PC claims)")}); icr = RP.Inputs(None, cr, None)
        s5 = at5(icr, pcg, a_exp); s5g = at5(icr, pcg, ans["forces"]); s5n = at5(icr, dict(widths_G_PC=[], G_PC={}), a_exp)
        check("AT5: a slope on delta_h2 with a 95 % cluster-bootstrap interval above 0 is labelled 'width/capacity, not delta' unless the AT2 verdict of that line is GROWS", s5["slopes"]["S"]["label"] == "width/capacity, not delta"
              and s5["slopes"]["S"]["interval_above_0"] and abs(s5["slopes"]["S"]["slope"] - 0.5) < 0.05 and s5["slopes"]["Lambda"]["label"] is None and not s5["slopes"]["Lambda"]["interval_above_0"] and s5["slopes"]["Lambda"]["ci95"][1] < 0
              and s5g["slopes"]["S"]["label"] is None and s5g["slopes"]["S"]["AT2_grows"], (s5["slopes"]["S"]["slope"], s5["slopes"]["S"]["ci95"], s5["slopes"]["Lambda"]["ci95"]))
        check("AT5 / G36: PC models and pairs enter only at G-PC widths (others 'not testable' with the gate's status)", s5["n_models"] == dict(bp=60, pc=5) and s5["pc_widths"] == {"128": "testable", "256": "BP-equivalent (not used for PC claims)"}
              and s5["pc_pairs"] == dict(testable=120, not_testable=130) and s5n["n_models"]["pc"] == 0 and s5n["pc_pairs"]["testable"] == 0 and s5n["pc_widths"]["128"].startswith("not testable")
              and at5(RP.Inputs(None, os.path.join(work, "empty"), None), pcg, a_exp)["status"].startswith("not run"), (s5["n_models"], s5["pc_pairs"], s5n["pc_widths"]))
        # ------------------------------------------------------------------------------------------------ A6 / A3 on a synthetic round-1 tree
        D = 48; amp = {14: 0.6, 8: 0.75, 20: 0.95}

        def tree(name, seed=1, GA=None, dyn=None, spill=None, **kw):
            r1 = os.path.join(work, name, "r1"); r2 = os.path.join(work, name, "r2"); ax = RP.synth_r1(r1, D=D, seed=seed)
            pl = {(b, L_, n): 3.0 * (amp[b] * ax[b]["Q"][:, 5] + math.sqrt(1 - amp[b] ** 2) * ax[b]["Q"][:, 7]) for b in (14, 8, 20) for L_ in (("RSI_a", "RSI_b") if b == 14 else ("RSI_a",)) for n in (3, 4)}
            ax = RP.synth_r1(r1, D=D, seed=seed, plants=pl, **kw); _synth_q(r2, ax, [14, 8, 20], GA=GA, dyn=dyn, spill=spill)
            return r1, r2, ax, RP.Inputs(r1, r2, os.path.join(work, name, "out", "rgi_verdict.json")), RP.r1_info(r1)
        r1, r2, ax, inp, info = tree("t1", spill={14: (2.0, 1.2), 8: (2.0, 0.2), 20: (0.0, 0.0)}); o6_ = a6(r1, inp, info); blk = o6_["blocks"]
        ST = R.load_r1_state(r1, 14); loops = R.load_r1_json(r1, "rk/b14/loops.json", 14); AXp = torch.load(os.path.join(r2, "r2_q1", "axes", "r2_axes.pt"), weights_only=False)
        dl = ST["ev_RSI_a_2"]["delta"].double(); d3 = dl.mean(0); uT = AXp["blocks"][14]["u_RGI_perpT"].double(); c_dir = float(d3 @ uT / (d3.norm() * uT.norm())); e3 = blk["14"]["loops"]["RSI_a"][3]
        nc_ = (AXp["blocks"][14]["null_u_RGI_perpT"].double() @ d3).abs() / (AXp["blocks"][14]["null_u_RGI_perpT"].double().norm(dim=1) * d3.norm()); p_dir = float((1 + (nc_ >= abs(c_dir)).sum()) / (1 + len(nc_)))
        check("A6 readouts: c = cos(mean pair delta behind d_n, axis), p against the axis' own nulls, A = c / sqrt(rel(d_n)) with rel from R1.loops[b][L][n-1]",
              abs(e3["u_RGI_perpT"]["c"] - c_dir) < 1e-9 and abs(e3["u_RGI_perpT"]["p"] - p_dir) < 1e-12 and abs(e3["u_RGI_perpT"]["A"] - c_dir / math.sqrt(loops["RSI_a"][2]["rel"])) < 1e-9 and e3["rel"] == loops["RSI_a"][2]["rel"]
              and e3["u_T_perp"]["c"] == loops["RSI_a"][2]["c"] and abs(e3["cos_stored_d_n"] - 1) < 1e-6 and e3["n_pairs"] == 32, (round(c_dir, 3), p_dir, round(e3["u_RGI_perpT"]["A"], 3)))
        ci = e3["u_RGI_perpT"]["A_ci90"]
        check("A6 readouts: every d_n (n = 1..4) of every frozen loop at every block of B3 on u_RGI_perpT and u_RGI_perp (90 % prompt-bootstrap interval of A), u_dir, g_b, s_b (descriptive)",
              sorted(blk) == ["14", "20", "8"] and all(set(R.loops_of_block(int(b))) <= set(v["loops"]) for b, v in blk.items()) and all(n in blk["8"]["loops"]["SEA"] for n in (1, 2, 3, 4))
              and blk["14"]["axes"] == ["u_RGI_perpT", "u_RGI_perp", "u_dir", "g_b", "s_b"] and ci[0] < e3["u_RGI_perpT"]["A"] < ci[1] and 0 < ci[1] - ci[0] < 0.5 and "A_ci90" in e3["u_RGI_perp"] and "A_ci90" not in e3["u_dir"]
              and fin(e3["g_b"]["p"]) and fin(e3["s_b"]["c"]) and blk["14"]["n_prompts"] == 16 and o6_["settings"]["n_boot"] == 500, (ci, blk["14"]["dirs_nulls"]))
        dy = o6_["dynamics"]; pd_ = o6_["pull_and_delta"]
        check("A6 / A7 dynamics on planted loops: 'self loop drawn to general' at the three blocks, GEN loop 'RGI sustains itself', pull 'grows with delta' in the paired three-block form (order by R1.delta)",
              all(dy[b]["enters"] and dy[b]["self_loop"]["verdict"] == "self loop drawn to general" and dy[b]["gen_loop"]["verdict"] == "RGI sustains itself" for b in ("14", "8", "20"))
              and pd_["verdict"] == "pull toward general grows with delta" and pd_["entering"] == [14, 8, 20] and pd_["low"] == 14 and pd_["high"] == 20 and pd_["middle_clause"] == "applied"
              and abs(pd_["X"]["14"] - 0.6) < 0.12 and abs(pd_["X"]["20"] - 0.95) < 0.12 and info["order"] == [14, 8, 20], ({b: dy[b]["self_loop"]["verdict"] for b in dy}, pd_["X"], pd_["diff_ci90"]))
        s3 = a3(r1, inp, info)
        check("A3 u_T_perp spillover at the blocks of B3 from rgi/causal (planted gradients): 'RSI push spills over', 'stays with self', 'no response'", s3["14"]["verdict"] == "RSI push spills over" and 0.5 < s3["14"]["FR"] < 0.7
              and s3["8"]["verdict"] == "stays with self" and s3["8"]["FR"] < 0.25 and s3["20"]["verdict"] == "no response" and s3["14"]["n_null"] == 400, {b: (v["verdict"], round(v["z_self"], 1), round(v["z_other"], 1)) for b, v in s3.items()})
        # gates and missing parts
        r1b, r2b, axb, inpb, infob = tree("t2", GA={8: False}, damages={(20, "RSI_a", 2): 0.3},
                                          summaries={14: dict(block=14, verdict=dict(verdict="unlikely", PC_PASS=True, label_attractor=False, gates=dict(failed=["GO2"], source="pilot.json")))})
        o6b = a6(r1b, inpb, infob); dyb = o6b["dynamics"]
        check("A6 gates: G-blk(14) fails -> block 14 does not enter and carries the G-desc label; GA(8) fails -> 'RGI axis not resolved', GEN 'not run (AUC gate)'; damage above 0.15 nats -> 'not decidable'; pull 'not decidable'",
              dyb["14"]["enters"] is False and dyb["14"]["why"] == "G-blk(b) fails" and o6b["blocks"]["14"]["labels"] == [RP.LBL_GDESC] and dyb["8"]["enters"] is False and dyb["8"]["why"] == "GA(b) fails"
              and o6b["blocks"]["8"]["labels"] == [RP.LBL_RGI] and o6b["gen"]["8"]["status"] == "not run (AUC gate)" and dyb["20"]["enters"] and dyb["20"]["self_loop"]["verdict"].startswith("not decidable (damage")
              and o6b["blocks"]["20"]["loops"]["RSI_a"]["damage_ok"] is False and o6b["pull_and_delta"]["verdict"] == "not decidable" and o6b["pull_and_delta"]["entering"] == [], (dyb["14"].get("why"), dyb["8"].get("why"), dyb["20"]["self_loop"]["verdict"]))
        r1c, r2c, axc, inpc, infoc = tree("t3", damages={(8, "RSI_a", 1): 0.3, (14, "RSI_a", 0): 0.9}, dyn={20: None}); o6c = a6(r1c, inpc, infoc); pdc = o6c["pull_and_delta"]
        check("A6: with two entering blocks the middle clause is dropped (marked); the start push (row 0) is not a counted evaluation; a missing GEN file is 'not run'", pdc["entering"] == [14, 20]
              and pdc["verdict"] == "pull toward general grows with delta" and pdc["middle_clause"].startswith("dropped") and o6c["dynamics"]["14"]["self_loop"]["verdict"] == "self loop drawn to general"
              and o6c["dynamics"]["8"]["self_loop"]["verdict"].startswith("not decidable (damage") and o6c["gen"]["20"]["status"].startswith("not run (round-2 output missing: rgi/dyn_b20.json")
              and o6c["dynamics"]["20"]["gen_loop"]["verdict"].startswith("not run") and o6c["blocks"]["14"]["loops"]["RSI_a"]["damage_ok"] is True, (pdc["entering"], pdc["middle_clause"], o6c["dynamics"]["8"]["self_loop"]["verdict"]))
        r1f, r2f, axf, inpf, infof = tree("t3b", summaries={20: dict(block=20, verdict=dict(PC_PASS=False, gates=dict(failed=[]))), 8: dict(block=8, verdict=dict(PC_PASS=True, gates=dict(failed=["GO3"])))}); o6f = a6(r1f, inpf, infof)
        check("A6: G-blk(b) needs PC_PASS and no failed gate; with one entering block 'pull and delta' is 'not decidable'", o6f["dynamics"]["20"]["enters"] is False and o6f["dynamics"]["8"]["enters"] is False
              and o6f["dynamics"]["14"]["enters"] is True and o6f["pull_and_delta"]["entering"] == [14] and o6f["pull_and_delta"]["verdict"] == "not decidable" and o6f["blocks"]["8"]["labels"] == [])
        r1d, r2d, axd, inpd, infod = tree("t4", damages={(8, "RND_1", 3): 0.3}); o6d = a6(r1d, inpd, infod); pdd = o6d["pull_and_delta"]
        check("A6: a control loop above the damage limit makes that block's self-loop rule 'not decidable' but RSI_a still enters 'pull and delta' (its own pushes are within the limit)",
              o6d["dynamics"]["8"]["self_loop"]["verdict"].startswith("not decidable (damage") and pdd["entering"] == [14, 8, 20] and pdd["verdict"] == "pull toward general grows with delta")
        os.remove(os.path.join(r1d, "gpu3", "summary_b20.json")); os.remove(os.path.join(r2d, "r2_q1", "axes", "r2_axes.pt")); os.remove(os.path.join(r2d, "r2_q3", "rgi", f"dirs_{M17}.pt"))
        os.remove(os.path.join(r2d, "r2_q3", "rgi", f"causal_{M17}.pt")); inpd = RP.Inputs(r1d, r2d, None); infod = RP.r1_info(r1d); o6e = a6(r1d, inpd, infod); s3e = a3(r1d, inpd, infod)
        check("missing parts: a block without its round-1 summary is 'not run (round-1 output missing)'; without axes / dirs / causal files the readouts say so and nothing enters the dynamics",
              o6e["blocks"]["20"]["status"] == NOT_RUN_R1 and o6e["status_axes"].startswith("not run") and o6e["status_dirs"].startswith("not run") and o6e["blocks"]["14"]["status"].startswith("ran without O2 axes")
              and o6e["blocks"]["14"]["axes"] == [] and o6e["dynamics"]["14"]["enters"] is False and o6e["pull_and_delta"]["verdict"] == "not decidable" and s3e["20"]["status"] == NOT_RUN_R1
              and s3e["14"]["status"].startswith("not run (round-2 output missing: rgi/causal"), (o6e["blocks"]["14"]["status"], s3e["14"]["status"]))
        # ------------------------------------------------------------------------------------------------ end to end
        RP.write_json(os.path.join(r2, "r2_q3", "rgi", f"geom_stats_{M17}.json"), _gs()); RP.synth_pcs(r1); rng = np.random.default_rng(0)
        for ws in range(8): RP.write_json(os.path.join(r2, "r2_t1", "rgi_toy", "r1", f"w{ws}.json"), _world(ws, ("regpert", "regpert", "absent", "absent"), rng))
        for ws in range(5):
            a_, b_ = _cross(ws, rng); RP.write_json(os.path.join(r2, "r2_t1", "rgi_toy", "cross", f"w{ws}.json"), a_); RP.write_json(os.path.join(r2, "r2_pc", "pc_toy", "cross_pc", f"w{ws}.json"), b_)
        RP.write_json(os.path.join(r2, "r2_t1", "rgi_toy", "fam.json"), _fam()); outp = os.path.join(work, "t1", "out", "rgi_verdict.json"); o = run(r1, r2, outp); sm = o["summary"]; js = RP.load_json(outp)
        check("end to end on a synthetic root: A6, A3, A7 (V1, 'the geometry forces RGI' with the dynamics clause failing here: block 14 is drawn too), AT8, AT7, AT5; strict JSON",
              sm["A7 over blocks"].startswith("V1") and sm["A7 combined"].startswith("the geometry does not force RGI: V1") and o["A7"]["combined"]["conditions"] == dict(V1_or_V2=True, present_top=True, absorbed_top=True, causal_top=True, dynamics=False)
              and sm["A3 spillover"]["14"] == "RSI push spills over" and sm["A6 pull and delta"] == "pull toward general grows with delta" and sm["AT8 answer"].startswith("not delta") and sm["AT8 recursion"] == "recursive"
              and sm["AT8 family"].startswith("portable") and sm["AT5 slope labels"] == {"S": "width/capacity, not delta"} and o["AT7"]["status"] == "ran" and js is not None and js["A6"]["blocks"]["14"]["loops"]["RSI_a"]["3"]["u_RGI_perpT"]["A_ci90"]
              and o["gates"]["GA"] == {"14": True, "8": True, "20": True} and o["AT0"]["status"].startswith("not run") and o["AT6"]["status"].startswith("not run") and o["AT5"]["pc_widths"] == {"128": "testable", "256": "testable"},
              (sm["A7 combined"][:80], sm["AT8 answer"], o["AT5"]["pc_widths"]))
        o0 = run(os.path.join(work, "none"), os.path.join(work, "empty"), os.path.join(work, "out0", "rgi_verdict.json")); s0 = o0["summary"]
        check("end to end without any input: every part 'not run', strict JSON written", o0["A6"]["blocks"]["14"]["status"] == NOT_RUN_R1 and o0["A3_u_T_perp_spillover"]["14"]["status"] == NOT_RUN_R1 and o0["A7"]["status"].startswith("not run")
              and s0["A7 combined"].startswith("not run") and s0["AT8 answer"].startswith("not run") and o0["AT7"]["status"].startswith("not run") and o0["AT5"]["status"].startswith("not run")
              and s0["AT8 family"].startswith("not run") and s0["A6 pull and delta"] == "not decidable" and RP.load_json(os.path.join(work, "out0", "rgi_verdict.json")) is not None, s0)
    finally:
        N_BOOT, AT7_NSIM, AT5_BOOT = keep
    ok = all(r["ok"] for r in res); RP.write_json(os.path.join(work, "r2_rgi_verdict_selftest.json"), dict(ok=ok, tests=res, seconds=time.time() - t_start))
    print(f"r2_rgi_verdict selftest: {sum(r['ok'] for r in res)}/{len(res)} passed ({time.time() - t_start:.0f}s)", flush=True)
    return ok


if __name__ == "__main__":
    sys.exit(main())
