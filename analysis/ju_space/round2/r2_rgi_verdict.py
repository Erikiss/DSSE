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
rgi/causal_qwen3-1.7b.pt (A3 gradients), rgi/dirs_qwen3-1.7b.pt (g_b, s_b, c_o); r2_t1 rgi_toy/{r0.json, r1/w<ws>.json, r1/at3.json,
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
            e = dict(rel=rel, u_T_perp={k: row.get(k) for k in ("c", "p", "Q", "A")}, n_pairs=int(torch.as_tensor(ev["ok"]).sum()))
            sq = math.sqrt(rel) if (fin(rel) and rel > 0) else float("nan"); Db = None
            for name, (axis, null, with_ci) in axes.items():
                s, dd = R.stats_on_axis(ev, axis, null, n_prompts=NP)
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
        try: bank = R.load_o3(r1_dir, b); ST = R.load_r1_state(r1_dir, b)
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
                                     labels=lab, loops=rd, missing_loops=bank.get("_missing"), axes=list(axes), dirs_nulls=src, n_prompts=NP,
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


def a7_over(gs, per, gs06):
    """Over-block items V1-V6, the RSI-into-RGI line and the causal line."""
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
        gi_some = [b for b in P if B[str(b)]["GI"]]
        v = "V5: no RGI geometry (" + ("general editing, not improvement-specific" if gi_some else "target-specific") + ")"; code = "V5"
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
    Dc, Dcc = num(oc.get("D")), oc.get("D_ci") or [None, None]
    if "D" not in oc: cv, sgn = "not computable (too few composed blocks of P)", 0
    elif ge(Dc, D_MIN) and gt(Dcc[0], 0): cv, sgn = "causal generality grows with delta or depth", 1
    elif le(Dc, -D_MIN) and lt(Dcc[1], 0): cv, sgn = "causal generality falls with delta or depth", -1
    else: cv, sgn = "no relation", 0
    cdep = depth_conditions(oc, sep, sgn) if sgn else None
    if cdep and all(cdep.values()): cv = cv.replace("with delta or depth", "with delta beyond depth")
    out["causal_line"] = dict(verdict=cv, D_FG_causal=Dc, D_FG_causal_ci=Dcc, D_rho_causal=orc.get("D"), D_rho_causal_ci=orc.get("D_ci"), depth_conditions=cdep,
                              composed_blocks=gate.get("P_composed"))
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
    per = a7_blocks(gs); over = a7_over(gs, per, gs06)
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
    t0 = time.time(); inp = RP.Inputs(r1_dir, r2_root, out_path); info = RP.r1_info(r1_dir); pcg = RP.pc_gates(r1_dir, inp)
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
    out.update(inputs=inp.record(["axes/r2_axes.pt", f"rgi/geom_stats_{M17}.json", f"rgi/geom_stats_{M06}.json", f"rgi/causal_{M17}.pt", f"rgi/dirs_{M17}.pt", "rgi_toy/r0.json",
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
