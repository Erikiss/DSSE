"""Round 2, session r2_ana: the Part C rules (frozen plan results/ju_space/plans/abschnitt11_12_runde2_plan.md, sha256 5e476340...edc2d,
section 5: C-Q2 to C-Q6 with C-Q1 from r2_linearity.py, C-T0 to C-T7; gates G11, G12, G13-G16, G35-G37, G41, G43-G45; OD-C6, OD-C7,
OD-C16, OD-G8; section 8 R1-5, R1-6, R1-7) -> unlearn_verdict.json.

    python r2_unlearn_verdict.py <r1_dir> <r2_root> <out.json>
    python r2_unlearn_verdict.py selftest <workdir>            synthetic inputs in the producers' formats, every branch

Inputs (found by name, the expected session first; see r2_report.Inputs): unlearn/q2.json, q3.json, q4a.json, q4b.json, armA.json, q5.json,
q6.json (+ .pt for the bootstrap draws) of r2_q1; unlearn/q1.json (r2_linearity.py, r2_ana); unlearn_toy/t0-t6.json of r2_t3 / r2_t4;
pc_toy/t7.json of r2_pc; round-1 files: summary_b14.json, rk/b14/state.pt and loops.json (C-Q4b matched-size reference, arm A comparator,
C-Q5 SVD and +- symmetry), pc_summary files (G-meta, G-PC for C-T7).
Every rule is implemented here from the plan text (r2_ana governs, OD-G8) and compared with the producers' in-session views
('rules', 'verdict_in_session', 'holds_in_session', 'r_star_in_session' of r2_unlearn_qwen.py; 'rule_preview' of r2_unlearn_toy.py and
r2_pc_toy.py): every disagreement is listed under 'od_g8'. Labels: G-desc ('descriptive (round-1 gate failed)') on every Qwen verdict at
block 14; 'RGI axis not resolved' where GA(14) fails (u_RGI_perp, u_diff, u_com, span); 'block 14 not usable' on C-Q5 if G-blk(14)
fails; 'world without a testable meta level' on the toy meta-level statements if G-meta fails; no pattern claim anywhere in Part C if the
circle gate G41 fails; specificity 'not testable' without a damage-matched control within 0.02 (G45). Parts whose inputs are missing are
'not run (...)'. Toy intervals: two-sided 95 % t over seeds (df = n - 1); '>= 4 of 5 seeds' = ceil(0.8 n)."""
import math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); JU = os.path.dirname(HERE)
for _p in (HERE, JU):
    if _p not in sys.path: sys.path.insert(0, _p)
import r2_report as RP
from r2_report import get, num, fin, ge, le, gt, lt, tci

NOT_RUN_R1 = RP.NOT_RUN_R1
# frozen thresholds of section 5
GAP_POINT, GAP_HI, D_FRAC, Z_D, D0_MIN, DMG_MAX, FOREIGN = 0.5, 0.75, 0.5, -2.33, 0.2, 0.05, 0.5
UE_FRAC, Z_FIN, M0_MIN, BAND05 = 0.5, 2.33, 0.1, (0.045, 0.055)
SHARE_SHALLOW, SHARE_DEEP, SHARE_DEEP_HI = 0.5, 0.2, 0.35
P_UNSEARCHED, A_SMALL, P_SIG, A_BIG, REL_REROUTED, UE_REROUTED = 0.10, 0.25, 0.01, 0.25, 0.5, 0.25
SUP_COS, SUP_COEF, SIGMA_BAND = 0.8, (0.5 / math.sqrt(2), 1.5 / math.sqrt(2)), (0.7, 1.4)
Q6_HELIX, Q6_LIN, Q6_SPEC, Q6_FRAC, Q6_RAND, Q6_REPAIR_DEPTH, Q6_REPAIR_SHARE, Q6_VALID = 0.75, 0.25, 0.2, 0.5, 0.2, 3.5, 0.75, 0.75
Q6_BLOCK = 14
UNLEARN_ACC, CHANCE_T, CHANCE_A = 0.11, 1.0 / 12, 1.0 / 17
T4_SHALLOW, T4_SHALLOW_LO, T4_DEEP = 0.20, 0.10, 0.05
T5_RSTAR, T5_KEEP, T5_PARTIAL = 4, 0.8, 0.5
T6_RATIO, T7_ALPHA, T7_MIN = 1.5, 0.05, 0.10
BONF = 1 - 0.05 / 3                                                              # C-T2: 98.33 % (3 values of k)
SEEDS = dict(svd_named_modes="none (deterministic)")


def log(msg): print(f"[r2_unlearn_verdict {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def not_run(js, rel):
    """Status of a missing or not-run producer output."""
    if js is None: return f"not run (round-2 output missing: {rel})"
    return RP.status_of(js)


def agree(mine, theirs):
    """None when the producer gave no view."""
    if theirs is None: return None
    return mine == theirs


# ======================================================================================================================================
# C-Q2 'suppressed', C-Q3 'unlearned' (shared gap conditions; O1 fixed axes, OD-C6, OD-C7)
# ======================================================================================================================================
def gap_conditions(g, g0, ratio_hi, ctrl_p5):
    return dict(g0_positive=gt(g0, 0), gap_half=bool(gt(g0, 0) and le(g, GAP_POINT * float(g0))), gap_hi=le(ratio_hi, GAP_HI),
                gap_below_random=lt(ratio_hi, ctrl_p5))


def foreign_ok(m_foreign, m_concept):
    """OD-C6: every foreign axis moves less than half as much as the concept (a non-finite m_X fails: it cannot be verified)."""
    vals = list((m_foreign or {}).values())
    return bool(fin(m_concept) and vals and all(fin(num(m)) and num(m) < FOREIGN * float(m_concept) for m in vals))


def rule_suppressed(g, g0, ratio_hi, ctrl_p5, Ue, z, M0, damage, m_foreign, m_concept):
    """C-Q2: 'suppressed' if all hold at the one push -alpha_u: (i) g_push <= 0.5 g_0, 90 % upper end of g_push / g_0 <= 0.75 and below the
    5th percentile of the ('cov', 14, 3) bootstrap distribution; (ii) Ue_fin >= 0.5 M0 and z_fin >= 2.33 (M0 > 0.1 required, else 'no
    baseline discrimination'); (iii) damage in 0.045-0.055 nats; (iv) every foreign axis moves less than half as much as the concept."""
    c = gap_conditions(g, g0, ratio_hi, ctrl_p5)
    c.update(M0_baseline=gt(M0, M0_MIN), Ue=bool(gt(M0, M0_MIN) and ge(Ue, UE_FRAC * float(M0))), z_fin=ge(z, Z_FIN),
             damage_band=bool(ge(damage, BAND05[0]) and le(damage, BAND05[1])), foreign=foreign_ok(m_foreign, m_concept))
    ok = all(c.values()); notes = [n for n, k in (("no baseline gap (g_0 <= 0)", "g0_positive"), ("no baseline discrimination", "M0_baseline")) if not c[k]]
    return dict(verdict="suppressed" if ok else "not suppressed", conditions=c, failed=[k for k, v in c.items() if not v], notes=notes)


def rule_unlearned(g, g0, ratio_hi, ctrl_p5, D, D0, zD, damage, m_foreign, m_concept, D_hi):
    """C-Q3 'unlearned' (u_RSI or the span): g_abl <= 0.5 g_0 with the 90 % upper end of g_abl / g_0 <= 0.75 and below the 5th percentile
    of the random ablations of the same dimension; D_abl <= 0.5 D_0 with z_D <= -2.33 (D_0 > 0.2 required); damage <= 0.05 nats; every
    foreign axis moves less than half as much as RSI. 'Inverted' if D_abl < 0 with its 90 % statement-bootstrap interval excluding 0."""
    c = gap_conditions(g, g0, ratio_hi, ctrl_p5)
    c.update(D_baseline=gt(D0, D0_MIN), D_half=bool(gt(D0, D0_MIN) and le(D, D_FRAC * float(D0))), z_D=le(zD, Z_D), damage=le(damage, DMG_MAX),
             foreign=foreign_ok(m_foreign, m_concept))
    ok = all(c.values()); notes = [n for n, k in (("no baseline gap (g_0 <= 0)", "g0_positive"), ("no baseline discrimination", "D_baseline")) if not c[k]]
    return dict(verdict="unlearned" if ok else "not unlearned", conditions=c, failed=[k for k, v in c.items() if not v], notes=notes,
                inverted=bool(lt(D, 0) and lt(D_hi, 0)))


def ratio_draws(pt, key):
    """Bootstrap ratios g / g_0 of a text set in a stage .pt ('texts'[key]['ratio_draws']); None if absent."""
    v = get(pt, "texts", key, "ratio_draws")
    if v is None: return None
    return np.asarray(v.numpy() if hasattr(v, "numpy") else v, dtype=float).ravel()


def cq2(q2, q2pt, labels14, GA):
    st = not_run(q2, "unlearn/q2.json")
    if st != "ran": return dict(status=st)
    out = dict(status="ran", directions={}, od_g8=[])
    for nm, concept, stmts in (("u_RSI", "RSI", "H30"), ("u_diff", "RSI", "H30"), ("u_RGI_perp", "RGI", "RGI30")):
        t = get(q2, "texts", f"{nm}|{concept}") or {}; b = get(q2, "behaviour", nm) or {}; dmg = get(q2, "alpha_u", nm, "damage")
        dr = ratio_draws(q2pt, f"cov_14_3|{concept}")
        ctrl = RP.pct(dr, 5) if dr is not None else num(get(q2, "rules", nm, "ctrl_p5"))
        r = rule_suppressed(t.get("g"), t.get("g0"), get(t, "ratio_ci90", 1), ctrl, b.get("Ue_fin"), b.get("z_fin"), b.get("M0"), dmg,
                            t.get("m_foreign"), t.get("m_concept"))
        lab = list(labels14) + ([RP.LBL_RGI] if (GA is False and nm in ("u_RGI_perp", "u_diff")) else [])
        sess = get(q2, "rules", nm, "verdict")
        r.update(concept=concept, statements=stmts, ctrl_p5=ctrl, ctrl_source="q2.pt cov_14_3 ratio draws" if dr is not None else "q2.json rules ctrl_p5",
                 inputs=dict(g=t.get("g"), g0=t.get("g0"), ratio=t.get("ratio"), ratio_ci90=t.get("ratio_ci90"), m_concept=t.get("m_concept"),
                             m_foreign=t.get("m_foreign"), own_axis_gap=t.get("own_axis_gap"), Ue_fin=b.get("Ue_fin"), A_fin=b.get("A_fin"), z_fin=b.get("z_fin"),
                             M0=b.get("M0"), alpha_u=get(q2, "alpha_u", nm, "f"), damage=dmg, damage_plus=get(q2, "alpha_u", nm, "damage_plus")),
                 labels=lab, in_session=sess, agrees_in_session=agree(r["verdict"], sess))
        if r["agrees_in_session"] is False: out["od_g8"].append(f"C-Q2 {nm}: r2_ana {r['verdict']!r}, in session {sess!r}")
        out["directions"][nm] = r
    out["descriptive"] = dict(K6_0p03=q2.get("K6"), content=q2.get("content"), perplexity=q2.get("perplexity"),
                              u_com_labels=list(labels14) + ([RP.LBL_RGI] if GA is False else []), behaviour_regression=get(q2, "behaviour", "regression"))
    return out


def cq3(q3, q3pt, labels14, GA):
    st = not_run(q3, "unlearn/q3.json")
    if st != "ran": return dict(status=st)
    out = dict(status="ran", rules={}, descriptive={}, od_g8=[])
    ctrl_keys = dict(u_RSI=("rand_14_2|RSI", "cov_14_0|RSI"), span=("ctrl2|RSI",))
    for nm in ("u_RSI", "span"):
        A = get(q3, "ablations", nm) or {}; t = get(A, "texts", "RSI") or {}; h = A.get("H30") or {}
        draws = [ratio_draws(q3pt, k) for k in ctrl_keys[nm]]
        ctrl = RP.pct(np.concatenate(draws), 5) if all(d is not None for d in draws) else num(get(q3, "ctrl_p5", nm))
        r = rule_unlearned(t.get("g"), t.get("g0"), get(t, "ratio_ci90", 1), ctrl, h.get("D"), h.get("D0"), h.get("z_D"), A.get("damage"),
                           t.get("m_foreign"), t.get("m_concept"), get(h, "D_ci90_statements", 1))
        sess = get(q3, "rules", nm, "verdict")
        r.update(ctrl_p5=ctrl, controls=list(ctrl_keys[nm]), ctrl_source="q3.pt ratio draws" if all(d is not None for d in draws) else "q3.json ctrl_p5",
                 inputs=dict(g=t.get("g"), g0=t.get("g0"), ratio=t.get("ratio"), ratio_ci90=t.get("ratio_ci90"), D=h.get("D"), D0=h.get("D0"), z_D=h.get("z_D"),
                             D_ci90_statements=h.get("D_ci90_statements"), damage=A.get("damage"), m_concept=t.get("m_concept"), m_foreign=t.get("m_foreign"),
                             own_axis_gap=t.get("own_axis_gap")),
                 labels=list(labels14) + ([RP.LBL_RGI] if (GA is False and nm == "span") else []), in_session=sess, agrees_in_session=agree(r["verdict"], sess))
        if r["agrees_in_session"] is False: out["od_g8"].append(f"C-Q3 {nm}: r2_ana {r['verdict']!r}, in session {sess!r}")
        out["rules"][nm] = r
    chosen = "u_RSI" if out["rules"]["u_RSI"]["verdict"] == "unlearned" else ("span" if out["rules"]["span"]["verdict"] == "unlearned" else None)
    sess = get(q3, "c_q4a", "ablation")
    out["G43"] = dict(r2_ana=chosen, in_session=sess, agrees=chosen == sess, used_by_C_Q4a=get(q3, "c_q4a", "used"),
                      rule="u_RSI if it passes, else the span; u_RSI (descriptive) if none passed (the in-session choice governs what ran)")
    if chosen != sess: out["od_g8"].append(f"G43: r2_ana {chosen!r}, in session {sess!r}")
    for nm, c in (("u_RGI_perp", "RGI"), ("u_diff", "RSI"), ("span", "RGI")):
        A = get(q3, "ablations", nm) or {}
        out["descriptive"][f"{nm}|{c}"] = dict(texts=get(A, "texts", c), H30=A.get("H30"), RGI30=A.get("RGI30"), damage=A.get("damage"),
                                                labels=list(labels14) + ([RP.LBL_RGI] if GA is False else []))
    out["descriptive"]["unprompted_u_RSI"] = get(q3, "ablations", "u_RSI", "unprompted")
    out["descriptive"]["controls"] = {k: dict(H30=get(q3, "ablations", v, "H30"), damage=get(q3, "ablations", v, "damage")) for k, v in
                                      (("rand_14_2", "null2"), ("cov_14_0", "null8"), ("ctrl2", "ctrl2"))}
    out["descriptive"].update(D0=q3.get("D0"), g0=q3.get("g0"), Delta_T=q3.get("Delta_T"), Delta_RGI=q3.get("Delta_RGI"), references=q3.get("references"))
    return out


def cq4a(q4a, c3, labels14):
    st = not_run(q4a, "unlearn/q4a.json")
    if st != "ran": return dict(status=st)
    used = q4a.get("ablation"); passed = bool(get(c3, "rules", used, "verdict") == "unlearned") if used in ("u_RSI", "span") else False
    share, hi = q4a.get("restored_share"), get(q4a, "restored_share_ci90", 1)
    if not passed: v = "not applicable (nothing removed)"
    elif ge(share, SHARE_SHALLOW): v = "shallow"
    elif lt(share, SHARE_DEEP) and lt(hi, SHARE_DEEP_HI): v = "deep"
    else: v = "partial"
    sess = q4a.get("verdict_in_session")
    return dict(status="ran", verdict=v, ablation=used, passed_C_Q3_r2_ana=passed, passed_C_Q3_in_session=q4a.get("passed_C_Q3"),
                restored_share=share, restored_share_ci90=q4a.get("restored_share_ci90"), share_defined=q4a.get("share_defined"),
                g_cue=q4a.get("g_cue"), g_abl=q4a.get("g_abl"), g0=q4a.get("g0"), ceiling=q4a.get("ceiling"), labels=list(labels14),
                descriptive_share=not passed, in_session=sess, agrees_in_session=agree(v, sess))


# ======================================================================================================================================
# C-Q4b, arm A (the loop rule; the matched-size reference of section 8 R1-6)
# ======================================================================================================================================
def loop_rule(abl, ref=None, ref_name="twin"):
    """C-Q4b rule on the ablated loop at d_2 and d_3, first match in the order of the plan text: 'un-searched' if p > 0.10 and |A| < 0.25
    while the reference (twin; for arm A the R1 seed-0 comparator) has p <= 0.01 and A >= 0.25; 'survives' if p <= 0.01, c > 0 and
    A >= 0.25; 're-routed' if rel(d_n) >= 0.5, Ue_n >= 0.25 with its 90 % interval above 0 and |A_n| < 0.25; 'undecided (<ref> does not
    re-find RSI)' if the reference fails p <= 0.01 and A >= 0.25; otherwise 'undecided'. Every condition at both d_2 and d_3.
    abl / ref: {d: dict(c, p, A, rel, Ue, Ue_ci90)}."""
    ns = (2, 3)
    if abl is None or any(n not in abl for n in ns): return "undecided (rounds d_2, d_3 missing)"
    every = lambda f, rows: all(f(rows[n]) for n in ns)
    gone = every(lambda r: gt(r.get("p"), P_UNSEARCHED) and fin(num(r.get("A"))) and abs(num(r["A"])) < A_SMALL, abl)
    refind = ref is not None and all(n in ref for n in ns) and every(lambda r: le(r.get("p"), P_SIG) and ge(r.get("A"), A_BIG), ref)
    surv = every(lambda r: le(r.get("p"), P_SIG) and gt(r.get("c"), 0) and ge(r.get("A"), A_BIG), abl)
    rer = every(lambda r: ge(r.get("rel"), REL_REROUTED) and ge(r.get("Ue"), UE_REROUTED) and gt(get(r, "Ue_ci90", 0), 0)
                and fin(num(r.get("A"))) and abs(num(r["A"])) < A_SMALL, abl)
    if gone and refind: return "un-searched"
    if surv: return "survives"
    if rer: return "re-routed"
    if not refind: return f"undecided ({ref_name} does not re-find RSI)"
    return "undecided"


def rows_by_d(rows):
    """{d: row} from a list of loop rows (each with 'd' = n + 1)."""
    return {int(r["d"]): r for r in (rows or []) if isinstance(r, dict) and "d" in r}


def seed0_reference(r1_dir, info):
    """Section 8 R1-6: the seed-0 half of R1 RSI_a (the even pairs of ev_RSI_a<SUF>_n, n = 0..3) read on u_T_perp with the R1 nulls:
    c, p, Q, rel, A of d_(n+1) at 64 pairs (rk's formulas, r2_rk.stats_on_axis), the C-Q4b matched-size reference and arm A's comparator."""
    if not get(info, "final", "14"): return dict(status=NOT_RUN_R1)
    import torch
    import r2_rk as R
    try:
        ST = R.load_r1_state(r1_dir, 14); suf = (R.load_r1_json(r1_dir, "rk/b14/loops.json", 14) or {}).get("_meta", {}).get("suffix", "")
    except Exception as e:                                                        # noqa: BLE001
        return dict(status=f"{NOT_RUN_R1}: {e!r}")
    ax = ST.get("axes") or {}; NP = len(ax.get("fold", [])) or None; rows = {}
    for n in range(4):
        ev = ST.get(f"ev_RSI_a{suf}_{n}")
        if ev is None: continue
        npairs = len(ev["prompt_of"]); nseed = npairs // NP if NP else 2; idx = torch.arange(0, npairs, max(nseed, 1))
        sub = dict(delta=ev["delta"][idx], ok=torch.as_tensor(ev["ok"])[idx], prompt_of=torch.as_tensor(ev["prompt_of"])[idx])
        s, _ = R.stats_on_axis(sub, ax["u_T_perp"], ax["null_u_T_perp"], n_prompts=NP)
        rows[n + 1] = dict(s, n_pairs=int(sub["ok"].sum()), key=f"ev_RSI_a{suf}_{n}")
    return dict(status="ran" if rows else NOT_RUN_R1, rows=rows, suffix=suf, note="seed-0 half = even pair indices (prompt-major, 2 seeds per prompt)")


def cq4b(q4b, info, ref0, labels14):
    st = not_run(q4b, "unlearn/q4b.json")
    G13, wh = info.get("LIKELY_wh"), info.get("from_would_have_read")
    tsc = q4b.get("twin_start_check") if isinstance(q4b, dict) else None
    G15 = bool(gt(get(tsc, "c"), 0) and le(get(tsc, "p"), P_SIG)) if (G13 and tsc) else (None if G13 else None)
    G14 = info.get("G_attr")
    open_ = bool((G13 and G15) or G14)
    branch = "likely" if (G13 and G15) else ("attractor" if G14 else None)
    gate = dict(G13=G13, from_would_have_read=wh, G15=G15, G14=G14, attracted_loop=info.get("attracted_loop"), open=open_, branch=branch,
                in_session=get(q4b, "gate"), agrees=None if q4b is None else bool(get(q4b, "gate", "open") == open_))
    out = dict(gate=gate, matched_size_reference=ref0, od_g8=[])
    if q4b is not None and gate["agrees"] is False: out["od_g8"].append(f"G44: r2_ana open={open_}, in session open={get(q4b, 'gate', 'open')}")
    lab = list(labels14) + ([RP.LBL_GDESC] if (branch == "likely" and wh and RP.LBL_GDESC not in labels14) else [])
    if st != "ran":
        out.update(status=st, verdict=st if st.startswith(("not testable", "not run")) else f"not testable ({st})", labels=lab); return out
    abl, twin = rows_by_d(q4b.get("ablated")), rows_by_d(q4b.get("twin"))
    v = loop_rule(abl, twin, "twin"); sess = q4b.get("verdict_in_session")
    out.update(status="ran", verdict=v, labels=lab, in_session=sess, agrees_in_session=agree(v, sess),
               rows=dict(ablated={d: {k: abl[d].get(k) for k in ("c", "p", "A", "rel", "Ue", "Ue_ci90", "cos_u_RSI", "damage_of_push")} for d in abl},
                         twin={d: {k: twin[d].get(k) for k in ("c", "p", "A", "rel", "Ue", "Ue_ci90")} for d in twin}))
    if out["agrees_in_session"] is False: out["od_g8"].append(f"C-Q4b: r2_ana {v!r}, in session {sess!r}")
    return out


def arm_a(armA, ref0, labels14):
    st = not_run(armA, "unlearn/armA.json")
    if st != "ran": return dict(status=st)
    c1, lo = armA.get("c1"), get(armA, "reference_seed0_RSI_a", "ci90", 0)
    cont_r2 = bool(lt(c1, lo)); cont_s = bool(armA.get("continues")); ran_rounds = int(armA.get("rounds") or len(armA.get("rows") or []))
    out = dict(status="ran", c1=c1, reference=armA.get("reference_seed0_RSI_a"), G16=dict(r2_ana=cont_r2, in_session=cont_s, agrees=cont_r2 == cont_s),
               rounds_ran=ran_rounds, labels=list(labels14), od_g8=[] if cont_r2 == cont_s else [f"G16: r2_ana {cont_r2}, in session {cont_s}"])
    if ran_rounds <= 1 or not cont_s:
        out["verdict"] = "translation (as predicted), not un-searched"
    else:
        comp = {d: r for d, r in (get(ref0, "rows") or {}).items()} if ref0 else None
        comp = {int(k): v for k, v in comp.items()} if comp else None
        out["verdict"] = loop_rule(rows_by_d(armA.get("rows")), comp, "the R1 seed-0 comparator")
        out["comparator"] = {d: {k: comp[d].get(k) for k in ("c", "p", "A", "rel")} for d in (comp or {}) if d in (2, 3)}
    out["in_session"] = armA.get("verdict_in_session")
    return out


# ======================================================================================================================================
# C-Q5 (superposition, +- symmetry, dose; the SVD of section 8 R1-5)
# ======================================================================================================================================
def cq5_svd(r1_dir, info, k=5):
    """R1-5: SVD of the stacked d_n (n = 1..4) of every loop of L(14) (O3 bank, uncentred), named modes u_T_perp, sports, tone, sea, u_neg
    (R1 axes) and the mean RND response (mean of the RND loops' d_n); per mode |cos| with the top-k right singular vectors; per (loop, n)
    the cosine with every named mode and the coefficients on the top-k modes (descriptive)."""
    if not get(info, "final", "14"): return dict(status=NOT_RUN_R1)
    import torch
    import r2_rk as R
    try: bank = R.load_o3(r1_dir, 14); ST = R.load_r1_state(r1_dir, 14)
    except Exception as e:                                                        # noqa: BLE001
        return dict(status=f"{NOT_RUN_R1}: {e!r}")
    names, rows = [], []
    for L in R.loops_of_block(14):
        if L not in bank: continue
        for n in range(1, 5): names.append(f"{L}|d{n}"); rows.append(bank[L]["d"][n].double())
    if not rows: return dict(status=NOT_RUN_R1, missing=bank.get("_missing"))
    Xm = torch.stack(rows); U, S, Vt = torch.linalg.svd(Xm, full_matrices=False); ax = ST["axes"]
    rnd = [bank[L]["d"][n].double() for L in ("RND_1", "RND_2") if L in bank for n in range(1, 5)]
    named = {nm: ax[nm].double() for nm in ("u_T_perp", "sports", "tone", "sea", "u_neg")}
    if rnd: named["mean_RND_response"] = torch.stack(rnd).mean(0)
    unit = lambda v: v / (v.norm() + 1e-300); kk = min(k, Vt.shape[0])
    modes = {nm: [float(abs(unit(v) @ Vt[i])) for i in range(kk)] for nm, v in named.items()}
    per = {nm_: dict(cos={nm: float(unit(x) @ unit(v)) for nm, v in named.items()}, coef_top=[float(x @ Vt[i]) for i in range(kk)], norm=float(x.norm()))
           for nm_, x in zip(names, rows)}
    return dict(status="ran", singular_values=S.tolist(), variance_share=(S ** 2 / (S ** 2).sum()).tolist(), n_directions=len(rows),
                named_mode_cos_with_top_modes=modes, per_direction=per, missing_loops=bank.get("_missing"))


def cq5(q5, info, r1_dir, labels14):
    out = dict(labels=list(labels14))
    if get(info, "G_blk", "14", "ok") is False: out["labels"].append(RP.LBL_BLK14)
    loops = None
    if get(info, "final", "14"):
        import r2_rk as R
        loops = R.load_r1_json(r1_dir, "rk/b14/loops.json", 14, required=False)
    sig = get(loops, "RSI_a", 0, "sigma")
    out["symmetry"] = dict(sigma=sig, band=list(SIGMA_BAND), holds=bool(ge(sig, SIGMA_BAND[0]) and le(sig, SIGMA_BAND[1])) if fin(num(sig)) else None,
                           status="ran" if fin(num(sig)) else NOT_RUN_R1, source="R1.loops[14]['RSI_a'][0]['sigma'] (row 0: the push d_0)")
    out["svd"] = cq5_svd(r1_dir, info)
    st = not_run(q5, "unlearn/q5.json")
    if st != "ran": out.update(status=st, superposition=dict(status=st)); return out
    sp = q5.get("superposition") or {}; cd = sp.get("cos_disatt"); cr, cs = sp.get("coef_RSI"), sp.get("coef_SEA")
    holds = bool(ge(cd, SUP_COS) and all(ge(c, SUP_COEF[0]) and le(c, SUP_COEF[1]) for c in (cr, cs)))
    sess = sp.get("holds_in_session")
    out.update(status="ran", superposition=dict(holds=holds, verdict="superposition holds" if holds else "superposition does not hold", cos_disatt=cd,
                                                cos=sp.get("cos"), cos_disatt_ci90=sp.get("cos_disatt_ci90"), coef_RSI=cr, coef_SEA=cs,
                                                coef_RSI_ci90=sp.get("coef_RSI_ci90"), coef_SEA_ci90=sp.get("coef_SEA_ci90"), rel_mix=get(sp, "rel_mix", "rel"),
                                                rel_sum=get(sp, "rel_sum", "rel"), coef_band=list(SUP_COEF), in_session=sess, agrees_in_session=agree(holds, sess)),
               dose=dict(q5.get("dose") or {}, linear_prediction=dict(norm_ratio=2.0, cos_debiased=1.0), note="descriptive"))
    return out


# ======================================================================================================================================
# C-Q6 (helix, RSI-specific, depth cap r*, self-repair)
# ======================================================================================================================================
def cq6(q6, labels14):
    st = not_run(q6, "unlearn/q6.json")
    if st != "ran": return dict(status=st)
    rels = q6.get("relations") or {}; out = dict(status="ran", labels=list(labels14), relations={}, od_g8=[], primary_block=Q6_BLOCK,
                                                  note="helix rule per block; block 14 (the block of the modes and the cap, section 5.2) is primary")
    for rel, R_ in rels.items():
        hel = {}
        for b in sorted((R_.get("angular") or {}).keys(), key=int):
            a, l = get(R_, "angular", b) or {}, get(R_, "linear", b) or {}
            ok = bool(ge(a.get("share"), Q6_HELIX) and gt(get(a, "ci90", 0), 0.5) and le(l.get("share"), Q6_LIN))
            hel[b] = dict(linearises=ok, angular_share=a.get("share"), angular_ci90=a.get("ci90"), linear_share=l.get("share"), helix=get(R_, "helix", b))
        kl = R_.get("kl") or {}; usable = kl.get("usable")
        rd = {b: get(R_, "readout", str(b)) or {} for b in (18, 22)}; valid = {b: bool(v.get("validated")) for b, v in rd.items()}
        caps = R_.get("cap") or {}; ranks = sorted({int(k.split("_r")[1].split("_")[0]) for k in caps if k.startswith("k4_")})
        r_star = None; cap_rows = {}
        for r in ranks:
            m_, x_ = caps.get(f"k4_r{r}_mode") or {}, caps.get(f"k4_r{r}_rand") or {}
            fm, fx = get(m_, "frac_all", "median"), get(x_, "frac_all", "median")
            ok = bool(ge(fm, Q6_FRAC) and lt(fx, Q6_RAND)); cap_rows[r] = dict(mode_frac_median=fm, mode_frac_ci90=get(m_, "frac_all", "ci90"), rand_frac_median=fx, passes=ok)
            if ok and r_star is None: r_star = r
        if usable is False: cap_v = "not testable (KL(p2 || p4) below the same-depth floor)"
        elif r_star is None: cap_v = f"no r <= {max(ranks) if ranks else 3} moves depth-4 sentences >= 50 % toward depth 2 with random patches < 20 %"
        else: cap_v = f"r* = {r_star}"
        sess = R_.get("r_star_in_session")
        if usable is not False and sess != r_star: out["od_g8"].append(f"C-Q6 {rel}: r2_ana r* {r_star}, in session {sess}")
        rep = None
        if r_star is not None and usable is not False:
            vb = [b for b in (18, 22) if valid[b]]
            if not vb: rep = "not testable (no validated readout at blocks 18 / 22)"
            else:
                shares = {b: get(caps, f"k4_r{r_star}_mode", f"readout_{b}", "ge_3_5_all") for b in vb}
                rep = "self-repair" if all(ge(s, Q6_REPAIR_SHARE) for s in shares.values()) else "no self-repair"
        out["relations"][rel] = dict(helix=hel, helix_primary=hel.get(str(Q6_BLOCK)), kl=kl, readouts_validated=valid, depth_cap=cap_v, r_star=r_star,
                                     caps=cap_rows, self_repair=rep, self_repair_shares={b: get(caps, f"k4_r{r_star}_mode", f"readout_{b}") for b in (18, 22)} if r_star else None,
                                     in_session_r_star=sess)
    spec = {}
    for b, v in (q6.get("improves_minus_describes_angular") or {}).items():
        d, ci = v.get("diff"), v.get("ci90") or [None, None]
        spec[b] = dict(diff=d, ci90=ci, rsi_specific=bool(ge(d, Q6_SPEC) and (gt(ci[0], 0) or lt(ci[1], 0))))
    out["rsi_specific"] = spec
    hp = get(out, "relations", "improves", "helix_primary") or {}
    out["statement"] = dict(helix_linearises_depth_at_14=hp.get("linearises"), rsi_specific_at_14=get(spec, str(Q6_BLOCK), "rsi_specific"),
                            depth_cap_improves=get(out, "relations", "improves", "depth_cap"), self_repair_improves=get(out, "relations", "improves", "self_repair"))
    return out


# ======================================================================================================================================
# Toy rules C-T0 ... C-T6 (unlearn_toy/<stage>.json) and C-T7 (pc_toy/t7.json)
# ======================================================================================================================================
def budgets(js):
    """(short, long) best-of-budget keys of a stage JSON (meta.budgets; (300, 3000) in a real run, QUICK (20, 60))."""
    b = get(js, "meta", "budgets") or [300, 3000]; return str(int(b[0])), str(int(b[1]))


def arr(v):
    return np.array([num(x) for x in (np.ravel(np.asarray(v, dtype=object)) if v is not None else [])], dtype=float)


def mean(v):
    a = arr(v); a = a[np.isfinite(a)]; return float(a.mean()) if len(a) else None


def lo(v, level=0.95): return tci(v, level)["lo"]


def hi(v, level=0.95): return tci(v, level)["hi"]


def does_not_unlearn(acc, base):
    """'A control does not unlearn' (C-T4): t(4) lower bound of its old-task accuracy >= 0.9 x base."""
    return ge(lo(acc), 0.9 * base) if base is not None else False


def unlearns_acc(acc):
    return le(hi(acc), UNLEARN_ACC)


def ct0(t0):
    st = not_run(t0, "unlearn_toy/t0.json")
    if st != "ran" and t0 is None: return dict(status=st)
    out = dict(status="ran", cells={})
    for name, c in (t0.get("cells") or {}).items():
        per = []
        for s in c.get("seeds") or []:
            T = s.get("tests") or {}; a = T.get("a_taskcols") or {}
            freq = gt(a.get("max"), a.get("null_max_p99"))
            loo = {K: bool(gt(v.get("real"), 0.3) and gt(v.get("real"), v.get("null_p95"))) for K, v in (T.get("c_loo_ring_R2") or {}).items()}
            ring = bool(freq and any(loo.values())); fa = bool(ring and any(ge(get(T, "overlap", f"k{k}"), 0.8) for k in (1, 2)))
            per.append(dict(ws=s.get("ws"), ring=ring, fourier_aligned=fa, freq=freq, loo=loo, producer=dict(ring=get(s, "flags", "ring"), fa=get(s, "flags", "fourier_aligned"))))
        n = len(per); q = RP.quorum(n); nr = sum(p["ring"] for p in per); nf = sum(p["fourier_aligned"] for p in per)
        mine = dict(has_ring_code=nr >= q, fourier_aligned=nf >= q)
        prev = get(t0, "rule_preview", name)
        out["cells"][name] = dict(n_seeds=n, quorum=q, ring_seeds=nr, fourier_aligned_seeds=nf, **mine, seeds=per,
                                  agrees_preview=None if prev is None else (prev.get("has_ring_code") == mine["has_ring_code"] and prev.get("fourier_aligned") == mine["fourier_aligned"]))
    return out


def ct1(t1):
    st = not_run(t1, "unlearn_toy/t1.json")
    if t1 is None: return dict(status=st)
    out = dict(status="ran", cells={})
    for name, c in (t1.get("cells") or {}).items():
        base = arr(get(c, "base", "acc")); n = len(base); q = RP.quorum(n); fams = {}
        for f, grid in (c.get("grids") or {}).items():
            if not grid: continue
            fam = get(c, "families", f) or {}; ctrl = "span" if f in ("xy", "task") else "iso"
            acc = {bs: np.array([[num(get(fam, str(k), bs, "acc", s)) for k in grid] for s in range(n)], float) for bs in ("pca", ctrl)}

            def kstar(a):
                return [int(np.argmax(a[s] >= 0.9 * base[s])) if np.any(a[s] >= 0.9 * base[s]) else len(grid) for s in range(n)]
            kp, kr = kstar(acc["pca"]), kstar(acc[ctrl]); steps = np.diff(acc["pca"], axis=1)
            n_gap = sum(kp[s] <= kr[s] - 2 for s in range(n)); n_jump = int((steps >= 0.5).any(1).sum()) if steps.size else 0
            n_grace = int((steps < 0.3).all(1).sum()) if steps.size else n
            lab = "critical length" if (n_gap >= q and n_jump >= q) else ("graceful" if n_grace >= q else "intermediate")
            prev = get(t1, "rule_preview", name, f, "label")
            fams[f] = dict(label=lab, control=ctrl, grid=grid, kstar_pca=[grid[k] if k < len(grid) else None for k in kp],
                           kstar_control=[grid[k] if k < len(grid) else None for k in kr], n_gap2=n_gap, n_jump05=n_jump, n_graceful=n_grace,
                           agrees_preview=agree(lab, prev))
        out["cells"][name] = dict(families=fams, B2_uncut={ns: tci(v) for ns, v in (get(c, "base", "B2") or {}).items()}, note="reported as compressibility")
    return out


def ct2(t2, g41ok):
    st = not_run(t2, "unlearn_toy/t2.json")
    if t2 is None: return dict(status=st)
    by = {}
    for d in (t2.get("nets") or {}).values(): by.setdefault((d["code"], int(d["k"])), []).append(d)
    for v in by.values(): v.sort(key=lambda d: d.get("ws", 0))
    ks = sorted({k for _, k in by}); codes = sorted({c for c, _ in by}); out = dict(status="ran", cells={}, k_min={}, effects={})
    for (code, k), v in sorted(by.items()):
        zs = [d.get("zero_shot") for d in v]; modal = [m.get("modal_is_trained") for d in v for m in (d.get("zs_modal") or [])]
        out["cells"][f"{code}_k{k}"] = dict(level1_seeds=sum(bool(d.get("level1")) for d in v), n=len(v), zero_shot=tci(zs), code_search=tci([d.get("code_search") for d in v]),
                                           heldout_eval=tci([d.get("heldout_eval") for d in v]), B2_8=tci([d.get("B2_8") for d in v]),
                                           snap=bool(lt(mean(zs), 0.03) and modal and float(np.mean([bool(x) for x in modal])) >= 0.75),
                                           modal_trained_share=float(np.mean([bool(x) for x in modal])) if modal else None)
    for code in codes:
        km = [k for k in ks if (code, k) in by and sum(bool(d.get("level1")) for d in by[(code, k)]) >= RP.quorum(len(by[(code, k)]))]
        out["k_min"][code] = km[0] if km else "none"
    kr = out["k_min"].get("ring")
    lin_zs = kr not in (None, "none") and any(ge(mean([d.get("zero_shot") for d in by[("ring", k)]]), 0.5) for k in ks if k >= kr and ("ring", k) in by)
    lin_cs, patt = False, False
    for k in ks:
        if not all((c_, k) in by for c_ in ("ring", "pring", "rand")): continue
        g = lambda c_, key: arr([d.get(key) for d in by[(c_, k)]])
        dcs = g("ring", "code_search") - g("rand", "code_search"); al = g("ring", "heldout_eval") - g("pring", "heldout_eval"); geo = g("pring", "heldout_eval") - g("rand", "heldout_eval")
        tb = tci(al, BONF)
        out["effects"][f"k{k}"] = dict(code_search_ring_minus_rand=tci(dcs), alignment_ring_minus_pring=tci(al), geometry_pring_minus_rand=tci(geo), alignment_bonferroni=tb)
        lin_cs |= bool(ge(mean(dcs), 0.15) and gt(lo(dcs), 0))
        patt |= bool(gt(tb["lo"], 0))
    out["ring_linearises"] = bool(lin_zs or lin_cs); out["ring_linearises_by"] = dict(zero_shot=bool(lin_zs), code_search=bool(lin_cs))
    out["pattern_one_hot_world_statistic"] = bool(patt)
    out["pattern_one_hot_world"] = ("pattern (one-hot world)" if (patt and g41ok is not False) else
                                    ("pattern claim withheld (G41: circle gate failed)" if patt else "no pattern (one-hot world)"))
    out["bonferroni_level"] = BONF
    prev = t2.get("rule_preview") or {}
    out["agrees_preview"] = dict(ring_linearises=agree(out["ring_linearises"], prev.get("ring_linearises")), k_min=agree(out["k_min"], prev.get("k_min")),
                                 pattern=agree(bool(patt), prev.get("pattern_one_hot_world")))
    return out


def jstar(byj):
    """j*(L) per seed: contiguous reading (the largest j such that every distance 1..j has zero-shot >= 0.5; 0 if distance 1 fails), the
    literal reading (the largest j with zero-shot >= 0.5) and the largest distance."""
    js = sorted(int(j) for j in (byj or {})); cont = 0
    for j in js:
        if ge(byj[str(j)], 0.5) and j == cont + 1: cont = j
        else: break
    lit = max([j for j in js if ge(byj[str(j)], 0.5)], default=0)
    return cont, lit, (max(js) if js else 0)


def t4b_rule(t, base_acc, chance):
    """C-T4 (b) per layer: 'causal' = accuracy upper bound <= 0.11 with the damage-matched control's lower bound >= 0.9 x base;
    'correlate' = accuracy lower bound >= 0.9 x base with the linear probe's upper bound <= chance + 0.05; 'nonlinear residue' = MLP probe
    lower bound >= chance + 0.05 (task-identity probes)."""
    pr = t.get("probes") or {}; dm = t.get("dm") or {}
    return dict(causal=bool(unlearns_acc(t.get("acc")) and does_not_unlearn(dm.get("acc"), base_acc)),
                correlate=bool(does_not_unlearn(t.get("acc"), base_acc) and le(hi(pr.get("task_lin")), chance + 0.05)),
                nonlinear_residue=bool(ge(lo(pr.get("task_mlp")), chance + 0.05)) if pr.get("task_mlp") is not None else None,
                dm_matched=bool(get(dm, "match", "matched")), acc=tci(t.get("acc")), dm_acc=tci(dm.get("acc")), task_lin=tci(pr.get("task_lin")),
                task_mlp=tci(pr.get("task_mlp")) if pr.get("task_mlp") is not None else None)


def ct3(t3, g41):
    if t3 is None: return dict(status=not_run(t3, "unlearn_toy/t3.json"))
    out = dict(status="ran", G41=g41, gate=g41.get("passes"), word={})
    for Kk, arms in (t3.get("gate") or {}).items():
        out[f"gate_{Kk}"] = dict(ring=tci([d.get("zero_shot") for d in arms.get("ring", [])]), pring=tci([d.get("zero_shot") for d in arms.get("pring", [])]),
                                 c_onehot=tci([d.get("zero_shot") for d in arms.get("c-onehot", [])]))
        out[f"hatching_{Kk}"] = [dict(ws=d.get("ws"), extract=get(d, "hatch", "extract"), first95=get(d, "hatch", "first95")) for d in arms.get("ring", [])]
    Ls = sorted(int(k[1:]) for k in (t3.get("word") or {})); lstar = None
    for L in Ls:
        w = t3["word"][f"L{L}"]; arc = sorted(w.get("arc", []), key=lambda d: d.get("ws", 0)); spr = sorted(w.get("spread", []), key=lambda d: d.get("ws", 0))
        js = [jstar(d.get("by_j")) for d in arc]; cov = sum(c >= mx for c, _, mx in js)
        sp_by = {d.get("ws"): d for d in spr}; diff = arr([num(a.get("zero_shot")) - num(get(sp_by, a.get("ws"), "zero_shot")) for a in arc])
        out["word"][f"L{L}"] = dict(jstar=[c for c, _, _ in js], jstar_literal=[l for _, l, _ in js], max_distance=[m for _, _, m in js], covers_seeds=cov,
                                    arc_minus_spread=tci(diff), word_length_effect=lt(hi(diff), 0),
                                    arc_maxsteps=[get(d, "row", "eval_point") == "maxsteps" for d in arc])
        if lstar is None and cov >= RP.quorum(len(arc)): lstar = L
    out["L_star"] = lstar
    ops = t3.get("ops")
    if ops:
        b = ops.get("base") or {}; a0, sp0 = mean(b.get("acc")), mean(b.get("sp")); cj = ops.get("conj") or {}
        mm = [bool(x) for row in (cj.get("modal_is_minus_c") or []) for x in (row if isinstance(row, list) else [row])]
        share = float(np.mean(mm)) if mm else None
        inv = bool(lt(hi(cj.get("acc")), CHANCE_A) and ge(mean(cj.get("sp")), 0.8 * sp0 if sp0 is not None else None) and ge(share, 0.75))
        out["conjugation"] = dict(acc=tci(cj.get("acc")), sp_ratio=(mean(cj.get("sp")) / sp0) if (sp0 and mean(cj.get("sp")) is not None) else None,
                                  modal_minus_c_share=share, inverted_ability=inv)
        out["t4b"] = {L: t4b_rule(t, a0, CHANCE_T) for L, t in (ops.get("t4b") or {}).items()}
        out["cplane"] = dict(acc=tci(get(ops, "cplane", "acc")), zero_shot=tci(get(ops, "cplane", "zero_shot")), rand_acc=tci(get(ops, "cplane", "rand", "acc")))
    if g41.get("passes") is False: out["note"] = "circle gate failed: no pattern claim anywhere in Part C (G41)"
    prev = t3.get("rule_preview") or {}
    out["agrees_preview"] = dict(gate=agree(g41.get("passes"), prev.get("gate")), L_star=agree(lstar, prev.get("L_star")))
    return out


def ct4_op(o, a0, sp0, sb, lb, q):
    op, dm = o.get("op") or {}, o.get("dm") or {}; matched = bool(get(dm, "match", "matched"))
    rl, an = arr(get(op, "relearn", sb)), arr(get(op, "A_new", sb)); d = rl - an if len(rl) == len(an) else arr([])
    if ge(mean(d), T4_SHALLOW) and gt(lo(d), T4_SHALLOW_LO): depth = "shallow"
    elif le(mean(rl), (mean(an) + T4_DEEP) if mean(an) is not None else None): depth = "deep"
    else: depth = "partial"
    n_perm = sum(lb_ == "consistent permutation" for lb_ in (op.get("label") or []))
    unl = bool(unlearns_acc(op.get("acc")) and does_not_unlearn(dm.get("acc"), a0))
    spec = ("not testable (no damage-matched control within 0.02)" if not matched else ("specific" if ge(mean(op.get("sp")), 0.8 * sp0 if sp0 is not None else None)
                                                                                         else "generic damage"))
    def status(acc):
        if does_not_unlearn(acc, a0): return "does not unlearn"
        if unlearns_acc(acc): return "unlearns"
        return "neither"
    return dict(unlearns=unl, specificity=spec, depth=depth, meta_unlearned=le(hi(get(op, "B2", "8")), 0.05),
                inverted=bool(lt(hi(op.get("acc")), CHANCE_A) and ge(mean(op.get("sp")), 0.8 * sp0 if sp0 is not None else None) and n_perm >= q),
                iso_status=status(get(o, "iso", "acc")), span_status=status(get(o, "span", "acc")) if o.get("span") else None, dm_status=status(dm.get("acc")),
                dm_matched=matched, acc=tci(op.get("acc")), dm_acc=tci(dm.get("acc")), sp=tci(op.get("sp")), relearn_minus_new=tci(d),
                relearn_long=tci(get(op, "relearn", lb)), A_new_long=tci(get(op, "A_new", lb)), B2_8=tci(get(op, "B2", "8")),
                consistent_permutation_seeds=n_perm, modal_labels=op.get("label"))


def ct4(t4, gmeta):
    if t4 is None: return dict(status=not_run(t4, "unlearn_toy/t4.json"))
    sb, lb = budgets(t4); out = dict(status="ran", cells={}, w4={}, budgets=[sb, lb])
    for name, c in (t4.get("cells") or {}).items():
        base = c.get("base") or {}; a0, sp0 = mean(base.get("acc")), mean(base.get("sp")); n = len(base.get("acc") or []); q = RP.quorum(n)
        ops = {nm: ct4_op(o, a0, sp0, sb, lb, q) for nm, o in (c.get("ops") or {}).items()}
        if get(ops, "blk1", "unlearns"):
            lad = dict(rung="one direction", within_span_1dim_control=get(ops, "blk1", "span_status"))
        else:
            rs = next((k for k in (2, 4, 6) if get(ops, f"blk{k}", "unlearns") and get(ops, f"blk{k}", "span_status") == "does not unlearn"), None)
            if rs is not None: lad = dict(rung="subspace", r_star=rs)
            elif get(ops, "block11", "unlearns"):
                s_ = get(ops, "block11", "iso_status")
                lad = (dict(rung="subspace", r_star=11, label="subspace: the whole task span") if s_ == "does not unlearn" else
                       dict(rung="not specific") if s_ == "unlearns" else dict(rung="undecided (isotropic control neither unlearns nor keeps >= 0.9 x base)"))
            else: lad = dict(rung="> 11", r_star="> 11")
        W = c.get("W")
        meta_lbl = [] if gmeta else [RP.LBL_GMETA]
        out["cells"][name] = dict(W=W, base_acc=a0, base_sp=sp0, ops=ops, ladder=lad, t4b={L: t4b_rule(t, a0, CHANCE_T) for L, t in (c.get("t4b") or {}).items()},
                                  meta_level_labels=meta_lbl, agrees_preview=agree(lad.get("rung"), get(t4, "rule_preview", "cells", name, "ladder", "rung")))
    for name, c in (t4.get("w4") or {}).items():
        b = c.get("base") or {}; a0 = mean(b.get("add")); o2 = {}
        for nm, o in (c.get("ops") or {}).items():
            op, dm = o.get("op") or {}, o.get("dm") or {}; matched = bool(get(dm, "match", "matched"))
            o2[nm] = dict(forget_unlearns=bool(unlearns_acc(op.get("add")) and does_not_unlearn(dm.get("add"), a0)),
                          specificity=("not testable (no damage-matched control within 0.02)" if not matched else
                                       ("specific" if ge(mean(op.get("sub")), (0.9 * mean(dm.get("sub"))) if mean(dm.get("sub")) is not None else None) else "generic damage")),
                          add=tci(op.get("add")), sub=tci(op.get("sub")), dm_add=tci(dm.get("add")), dm_sub=tci(dm.get("sub")), sp=tci(op.get("sp")),
                          relearn_add_short=tci(get(op, "relearn_add", sb)), relearn_add_long=tci(get(op, "relearn_add", lb)))
        out["w4"][name] = dict(base_add=a0, base_sub=mean(b.get("sub")), ops=o2)
    return out


def ct5(t5):
    if t5 is None: return dict(status=not_run(t5, "unlearn_toy/t5.json"))
    sb, lb = budgets(t5); out = dict(status="ran", widths={}, budgets=[sb, lb])
    for name, w in (t5.get("widths") or {}).items():
        A = np.array(get(w, "A", lb), dtype=float); ch = w.get("chains") or []; seeds = w.get("seeds") or sorted({c[1] for c in ch}); n = len(seeds)
        if A.ndim != 2 or not ch: out["widths"][name] = dict(status="not computable (format)"); continue
        R = A.shape[1]; i4, last = min(4, R - 1), R - 1; q = RP.quorum(n); arms = {}; nnew = None
        for arm in (w.get("arms") or sorted({c[0] for c in ch})):
            idx = [i for i, c in enumerate(ch) if c[0] == arm]
            if not idx: continue
            by_seed = {}
            for i in idx: by_seed.setdefault(ch[i][1], []).append(i)
            sd = [s for s in seeds if s in by_seed]; nnew = len(by_seed[sd[0]]) if sd else 0
            la = min(last, int(w.get("pc_last", last))) if arm == "pc" else last
            per_task_r = {s: [int(np.argmax(A[i, :la + 1] <= UNLEARN_ACC)) if np.any(A[i, :la + 1] <= UNLEARN_ACC) else R for i in by_seed[s]] for s in sd}
            arms[arm] = dict(A_free=[float(np.mean(A[by_seed[s], 0])) for s in sd], A_op=[float(np.mean(A[by_seed[s], la])) for s in sd],
                             A_s4=[float(np.mean(A[by_seed[s], i4])) for s in sd], rstar_seed=[float(np.median(per_task_r[s])) for s in sd],
                             rstar_task=[per_task_r[s] for s in sd], last_round=la)
            arms[arm]["A_op_at_rstar"] = [float(np.mean([A[i, min(int(r), la)] for i, r in zip(by_seed[s], per_task_r[s])])) for s in sd]
            arms[arm]["A_op_t"] = tci(arms[arm]["A_op"])
        o = dict(rounds=R, s4_index=i4, last_index=last, arms=arms, n_seeds=n, never=f"'> {R - 1}' counts as {R}")
        col, rnd = arms.get("col"), arms.get("rand")
        if col and rnd:
            o["low_dimensional_search"] = bool(sum(r_ <= T5_RSTAR for r_ in col["rstar_seed"]) >= q and sum(ge(a, T5_KEEP * f) for a, f in zip(rnd["A_s4"], rnd["A_free"])) >= q)
            o["re_routing"] = bool(gt(lo(col["A_op"]), UNLEARN_ACC))
        dm_last = [d for d in (w.get("dm") or []) if d.get("round") == last]
        for X, ctl in (("col", "rand"), ("z1", "dmr")):
            if X in arms and ctl in arms:
                ax, ac = arms[X], arms[ctl]; keep = ge(mean(ac["A_op"]), T5_KEEP * mean(ax["A_free"]) if mean(ax["A_free"]) is not None else None)
                v = "un-searched" if (le(hi(ax["A_op"]), UNLEARN_ACC) and keep) else ("partial" if le(mean(ax["A_op"]), T5_PARTIAL * mean(ac["A_op"]) if mean(ac["A_op"]) is not None else None) else "none")
                if ctl == "dmr" and dm_last and not dm_last[0].get("matched"): v = "not testable (no damage-matched control within 0.02 at s_last, G45)"
                o[f"verdict_{X}"] = v; o[f"control_{X}"] = dict(control=ctl, A_rand_mean=mean(ac["A_op"]), A_free_mean=mean(ax["A_free"]), control_keeps=keep)
        prev = get(t5, "rule_preview", name) or {}
        o["agrees_preview"] = {k: agree(o.get(k), prev.get(k)) for k in ("low_dimensional_search", "re_routing", "verdict_col", "verdict_z1") if k in o}
        o["pc_arm"] = "descriptive"
        out["widths"][name] = o
    return out


def ct6(t6):
    if t6 is None: return dict(status=not_run(t6, "unlearn_toy/t6.json"))
    fr = t6.get("free") or {}; modes = fr.get("modes") or []; nulls = fr.get("null") or []; n = len(modes); q = RP.quorum(n)
    nm = max((len(m.get("modes") or []) for m in modes), default=0); real = []
    for i in range(nm):
        cnt = sum(1 for s in range(n) if gt(get(modes, s, "modes", i, "var"), get(nulls, s, "var_p99", i))
                  and lt(abs(num(get(modes, s, "modes", i, "lissajous_r"))) if fin(num(get(modes, s, "modes", i, "lissajous_r"))) else None, get(nulls, s, "lis_p5", i)))
        real.append(dict(index=i + 1, seeds=int(cnt), real=cnt >= q))
    caps = {}
    for L, c in (t6.get("caps") or {}).items():
        cnt = sum(1 for l1, rx, rw in zip(c.get("level1") or [], c.get("ratio_Ex") or [], c.get("ratio_W2") or []) if bool(l1) and ge(rx, T6_RATIO) and ge(rw, T6_RATIO))
        caps[L] = dict(seeds=int(cnt), re_routing=cnt >= q, level1_seeds=sum(bool(x) for x in c.get("level1") or []), ratio_Ex=tci(c.get("ratio_Ex")),
                       ratio_W2=tci(c.get("ratio_W2")), B2_8=tci(get(c, "B2", "8")))
    prev = t6.get("rule_preview") or {}
    return dict(status="ran", modes=real, caps=caps, free_B2_8=tci(get(fr, "B2", "8")),
                agrees_preview=dict(modes=agree([m["real"] for m in real], [m.get("real") for m in prev.get("modes") or []] or None),
                                    caps=agree({L: v["re_routing"] for L, v in caps.items()}, {L: v.get("re_routing") for L, v in (prev.get("caps") or {}).items()} or None)))


def _seed_vals(us, arm, r, which, sb):
    """Per-seed (mean over the new tasks) accuracy of arm at round r of a PC-protocol un-search record: 'short' = A_<short>(s_r), 'long'
    = the 3,000-step search at that round."""
    ch = us.get("chains") or []; Ash = np.array(get(us, "A", sb), dtype=float); R = Ash.shape[1] if Ash.ndim == 2 else 0
    r = min(r, R - 1); seeds = sorted({c[1] for c in ch})
    if which == "short":
        return np.array([np.mean([Ash[i, r] for i, c in enumerate(ch) if c[0] == arm and c[1] == s]) if any(c[0] == arm and c[1] == s for c in ch) else np.nan for s in seeds])
    L = us.get("long") or {}; vals = {}
    for k, (i, rr) in enumerate(zip(L.get("chain") or [], L.get("round") or [])):
        c = ch[i]
        if c[0] == arm and rr == r: vals.setdefault(c[1], []).append(L["A_long"][k])
    return np.array([np.mean(vals[s]) if s in vals else np.nan for s in seeds])


def holm(rows, alpha=T7_ALPHA, min_effect=T7_MIN):
    """Two-sided paired t tests (df = n - 1), Holm over the family; a claim needs Holm p <= alpha and |mean| >= min_effect."""
    from scipy import stats
    for r in rows:
        v = arr(r["values"]); v = v[np.isfinite(v)]; n = len(v); r["n"] = n
        if not r.get("testable", True) or n < 2: r["testable"] = False; continue
        m, sd = float(v.mean()), float(v.std(ddof=1)); se = sd / math.sqrt(n)
        t = m / se if se > 0 else (math.inf if m != 0 else 0.0); p = float(2 * stats.t.sf(abs(t), n - 1)) if math.isfinite(t) else 0.0
        r.update(testable=True, mean=m, sd=sd, t=t, p=p, **{k: v_ for k, v_ in tci(v).items() if k in ("lo", "hi")})
    tst = [r for r in rows if r["testable"]]; order = sorted(range(len(tst)), key=lambda i: tst[i]["p"]); run = 0.0; m = len(tst)
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * tst[i]["p"])); tst[i]["p_holm"] = run
    for r in tst: r["claim"] = bool(r["p_holm"] <= alpha and abs(r["mean"]) >= min_effect)
    return dict(family_size=m, items=rows, claims=[f"W{r['W']} {r['item']}" for r in tst if r["claim"]])


def ct7(t7, pcg, min_paired=4):
    """C-T7 (critique v2-12h): the family = at every G-PC width, the per-seed paired PC - BP differences of forward accuracy after erasure
    (h1, h2), relearn_300 and relearn_3000 under erasure (h1, h2), A_300 at s_4, s_12, s_24 and A_3000 at s_24 (col arm), and the PC-only
    contrast Delta_relax - Delta_rand (task-identity probe, T 20 and T 100; both erasure layers, as r2_pc_toy records them); Holm over the
    whole family; no G-PC width: 'not testable'. A width with fewer than min_paired paired seeds (G37) is 'not testable'."""
    if t7 is None: return dict(status=not_run(t7, "pc_toy/t7.json"))
    widths = [int(w) for w in (pcg.get("widths_G_PC") or [])]; sb, lb = budgets(t7); quick = bool(get(t7, "meta", "quick"))
    mp = 2 if quick else min_paired
    out = dict(status="ran", G_PC_widths=widths, G_PC=pcg.get("G_PC"), G_meta=pcg.get("G_meta"), budgets=[sb, lb], min_paired=mp, per_width={})
    rows = []
    for key, w in (t7.get("widths") or {}).items():
        W = int(w.get("W", key.lstrip("W"))); np_ = len(w.get("seeds_present") or []); inc = W in widths
        out["per_width"][str(W)] = dict(in_family=inc and np_ >= mp, seeds_present=w.get("seeds_present"), seeds_missing=w.get("seeds_missing"),
                                        G_PC=get(pcg, "G_PC", str(W), "status", default="not testable (no PCS rows at this width)"))
        if not inc or "pc" not in w or "bp" not in w: continue
        pc, bp = w["pc"], w["bp"]; ok = np_ >= mp; its = []
        for L in ("h1", "h2"):
            its.append((f"fwd_acc_{L}", arr(get(pc, "t4b", L, "acc")) - arr(get(bp, "t4b", L, "acc"))))
            for b_ in (sb, lb):
                its.append((f"relearn_{b_}_{L}", arr(get(pc, "t4b", L, "relearn", b_)) - arr(get(bp, "t4b", L, "relearn", b_))))
        for r, b_, wh in ((4, sb, "short"), (12, sb, "short"), (24, sb, "short"), (24, lb, "long")):
            try: its.append((f"A{b_}_s{r}_col", _seed_vals(pc["unsearch"], "col", r, wh, sb) - _seed_vals(bp["unsearch"], "col", r, wh, sb)))
            except Exception as e:                                                # noqa: BLE001
                its.append((f"A{b_}_s{r}_col", arr([]))); out.setdefault("errors", []).append(f"W{W} A{b_}_s{r}: {e!r}")
        for L in ("h1", "h2"):
            for T in (20, 100): its.append((f"relax_contrast_T{T}_{L}", arr(get(pc, "relax", L, str(T), "contrast"))))
        rows += [dict(W=W, item=nm, values=v, testable=ok) for nm, v in its]
    if not widths:
        out.update(verdict="not testable (no G-PC width)", family=None); return out
    fam = holm(rows); out["family"] = fam
    out["verdict"] = ("claims: " + "; ".join(fam["claims"])) if fam["claims"] else ("no claim (no item with Holm p <= 0.05 and |mean| >= 0.10)" if fam["family_size"] else "not testable (no testable item)")
    out["relaxation_restores_task_information"] = [c for c in fam["claims"] if "relax_contrast" in c]
    out["note"] = "no 'at equilibrium' claim (truncated inference)"
    return out


# ======================================================================================================================================
# Driver
# ======================================================================================================================================
def run(r1_dir, r2_root, out_path):
    t0 = time.time(); inp = RP.Inputs(r1_dir, r2_root, out_path); info = RP.r1_info(r1_dir); lab14 = RP.desc_labels(info, 14)
    J = {s: inp.json(f"unlearn/{s}.json") for s in ("q1", "q2", "q3", "q4a", "q4b", "armA", "q5", "q6")}
    PT = {s: inp.pt(f"unlearn/{s}.pt") for s in ("q2", "q3")}
    GA = get(J["q3"], "GA14", default=get(J["q2"], "GA14"))
    g41 = RP.g41(inp); pcg = RP.pc_gates(r1_dir, inp)
    ref0 = seed0_reference(r1_dir, info) if r1_dir else dict(status=NOT_RUN_R1)
    out = dict(script="r2_unlearn_verdict.py", plan=RP.PLAN_REL, plan_sha256=RP.PLAN_SHA256, part="C", seeds=SEEDS,
               gates=dict(G_desc=info.get("G_desc"), G_blk_14=get(info, "G_blk", "14"), G_likely=dict(LIKELY_wh=info.get("LIKELY_wh"), from_would_have_read=info.get("from_would_have_read"), V_RSI=info.get("V_RSI")),
                          G_attr=dict(G_attr=info.get("G_attr"), attracted_loop=info.get("attracted_loop"), rnd_last_rows=info.get("rnd_last_rows")),
                          G41=g41, G_meta=dict(value=pcg.get("G_meta"), source=pcg.get("G_meta_source")), G_PC={W: v["status"] for W, v in (pcg.get("G_PC") or {}).items()},
                          GA14=GA, final_14=get(info, "final", "14")),
               labels_block14=lab14)
    q1 = J["q1"]
    out["C_Q1"] = dict(status=not_run(q1, "unlearn/q1.json")) if q1 is None else dict(status=RP.status_of(q1), statements=q1.get("statements"), G42=q1.get("G42"),
                                                                                       verdicts={c: dict(linear=get(v, "linear_rung", "verdict"), nonlinear=v.get("nonlinear_rung"), labels=v.get("labels"))
                                                                                                 for c, v in (q1.get("concepts") or {}).items()})
    out["C_Q2"] = cq2(J["q2"], PT["q2"], lab14, GA)
    out["C_Q3"] = cq3(J["q3"], PT["q3"], lab14, GA)
    out["C_Q4a"] = cq4a(J["q4a"], out["C_Q3"], lab14)
    out["C_Q4b"] = cq4b(J["q4b"], info, ref0, lab14)
    out["C_Q4_armA"] = arm_a(J["armA"], ref0, lab14)
    out["C_Q5"] = cq5(J["q5"], info, r1_dir, lab14)
    out["C_Q6"] = cq6(J["q6"], lab14)
    T = {s: inp.json(f"unlearn_toy/{s}.json") for s in ("t0", "t1", "t2", "t3", "t4", "t5", "t6")}
    gmeta = pcg.get("G_meta")
    out["C_T0"] = ct0(T["t0"]); out["C_T1"] = ct1(T["t1"]); out["C_T2"] = ct2(T["t2"], g41.get("passes")); out["C_T3"] = ct3(T["t3"], g41)
    out["C_T4"] = ct4(T["t4"], gmeta); out["C_T5"] = ct5(T["t5"]); out["C_T6"] = ct6(T["t6"])
    out["C_T7"] = ct7(inp.json("pc_toy/t7.json"), pcg)
    if not gmeta:
        for k in ("C_T1", "C_T4", "C_T5"):
            if isinstance(out[k], dict): out[k].setdefault("labels", []).append(RP.LBL_GMETA + " (meta-level statements: B2)")
    out["od_g8"] = [x for k in ("C_Q2", "C_Q3", "C_Q4b", "C_Q4_armA", "C_Q6") for x in (get(out, k, "od_g8") or [])]
    if get(out, "C_Q4a", "agrees_in_session") is False: out["od_g8"].append(f"C-Q4a: r2_ana {out['C_Q4a']['verdict']!r}, in session {out['C_Q4a']['in_session']!r}")
    out["summary"] = summary(out)
    out.update(pc_gates=dict(G_meta=pcg.get("G_meta"), G_meta_source=pcg.get("G_meta_source"), G_PC=pcg.get("G_PC"), labels=pcg.get("labels")),
               inputs=inp.record([f"unlearn/{s}.json" for s in J] + [f"unlearn_toy/{s}.json" for s in T] + ["pc_toy/t7.json"]),
               finder=inp.log, r1_finder=info.get("finder"), seconds=time.time() - t0)
    RP.write_json(out_path, out); log(f"Part C verdicts -> {out_path} ({time.time() - t0:.0f}s)")
    return out


def summary(o):
    """One line per item (the report's README table)."""
    s = {}
    s["C-Q1"] = get(o, "C_Q1", "statements") or get(o, "C_Q1", "status")
    s["C-Q2"] = {k: v.get("verdict") for k, v in (get(o, "C_Q2", "directions") or {}).items()} or get(o, "C_Q2", "status")
    s["C-Q3"] = {k: (v.get("verdict") + (" (inverted)" if v.get("inverted") else "")) for k, v in (get(o, "C_Q3", "rules") or {}).items()} or get(o, "C_Q3", "status")
    s["C-Q4a"] = get(o, "C_Q4a", "verdict") or get(o, "C_Q4a", "status"); s["C-Q4b"] = get(o, "C_Q4b", "verdict"); s["C-Q4 arm A"] = get(o, "C_Q4_armA", "verdict") or get(o, "C_Q4_armA", "status")
    s["C-Q5"] = dict(superposition=get(o, "C_Q5", "superposition", "verdict") or get(o, "C_Q5", "status"), symmetry=get(o, "C_Q5", "symmetry", "holds"))
    s["C-Q6"] = get(o, "C_Q6", "statement") or get(o, "C_Q6", "status")
    s["C-T0"] = {k: dict(ring=v["has_ring_code"], fourier=v["fourier_aligned"]) for k, v in (get(o, "C_T0", "cells") or {}).items()} or get(o, "C_T0", "status")
    s["C-T1"] = {k: {f: x["label"] for f, x in v["families"].items()} for k, v in (get(o, "C_T1", "cells") or {}).items()} or get(o, "C_T1", "status")
    s["C-T2"] = dict(ring_linearises=get(o, "C_T2", "ring_linearises"), pattern=get(o, "C_T2", "pattern_one_hot_world"), k_min=get(o, "C_T2", "k_min")) if get(o, "C_T2", "status") == "ran" else get(o, "C_T2", "status")
    s["C-T3"] = dict(gate=get(o, "C_T3", "gate"), L_star=get(o, "C_T3", "L_star"), inverted_ability=get(o, "C_T3", "conjugation", "inverted_ability")) if get(o, "C_T3", "status") == "ran" else get(o, "C_T3", "status")
    s["C-T4"] = {k: v["ladder"] for k, v in (get(o, "C_T4", "cells") or {}).items()} or get(o, "C_T4", "status")
    s["C-T5"] = {k: {x: v.get(x) for x in ("low_dimensional_search", "re_routing", "verdict_col", "verdict_z1")} for k, v in (get(o, "C_T5", "widths") or {}).items()} or get(o, "C_T5", "status")
    s["C-T6"] = dict(real_modes=[m["index"] for m in get(o, "C_T6", "modes") or [] if m["real"]], re_routing={L: v["re_routing"] for L, v in (get(o, "C_T6", "caps") or {}).items()}) if get(o, "C_T6", "status") == "ran" else get(o, "C_T6", "status")
    s["C-T7"] = get(o, "C_T7", "verdict") or get(o, "C_T7", "status")
    return s


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if len(a) >= 2 and a[0] == "selftest": return 0 if selftest(a[1]) else 1
    if len(a) != 3: print(__doc__); return 2
    run(*a); return 0
