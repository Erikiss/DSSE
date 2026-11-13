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
'not run (...)'. Toy intervals: two-sided 95 % t over seeds (df = n - 1); '>= 4 of 5 seeds' = ceil(0.8 n).
Amendment 2 (OD-C2: partial removal at the found dimension d* with one shared lambda; declared post hoc, so both readings are reported,
section 14): the damage-matched controls are those the producers ran (the amended procedure; d*, lambda and 'matched' are read from
their records). Where the amended step applied (d* > r0 with lambda < 1), the result under the frozen text is given next to it
('under_plan_text'): the plan-text control (full removal at d*) overshoots the target by more than 0.02, so it is unmatched and
specificity is 'not testable' (G45). 'Unlearns' / 'causal' is False under both readings where its accuracy half fails (upper bound >
0.11; no control needed) and 'not computable' where it holds (that control was not run); the C-T4 ladder is evaluated with these three
values. In C-T7 the relaxation contrast of an amended layer is 'not computable' under the frozen text; for the other items a claim
stands if m x p <= 0.05, is excluded if p > 0.05 or |mean| < 0.10, and is otherwise 'not determined' (Holm rank unknown).
'amendment_2' lists every affected result."""
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
T5_RSTAR, T5_KEEP, T5_PARTIAL, T5_NEVER = 4, 0.8, 0.5, 25                       # '> 24' counts as 25
DM_TOL = 0.02                                                                  # OD-C2 / G45: matched within 0.02
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
                descriptive_share=not passed, in_session=sess, agrees_in_session=agree(v, sess),
                own_axis={k: x for k, x in q4a.items() if "own_axis" in k} or None)                       # descriptive (critique v2-4): the own-axis gaps as the producer wrote them


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


TRANSLATION = "translation (as predicted), not un-searched"


def g16_reference(armA):
    """G16's reference of arm A, read defensively. Amendment 3 (a): 'ci90' is the percentile interval of the 2,000 replicates shifted so
    that their median coincides with the point estimate, [q05 + c - m, q95 + c - m]; the unshifted percentile interval ('ci90_percentile'),
    the median ('boot_median') and the G16 value the unshifted interval would have given are stored next to it. Returns dict(lo (the
    lower end G16 uses), recentred (None if the new fields are absent: an output written before the amendment), consistent (the stored
    'ci90' equals the shifted interval), percentile, median, unshifted_in_session, unshifted_r2_ana)."""
    ref = armA.get("reference_seed0_RSI_a") or {}; c1 = armA.get("c1"); lo = get(ref, "ci90", 0)
    pct = next((ref[k] for k in ("ci90_percentile", "ci90_unshifted", "percentile_ci90") if isinstance(ref.get(k), (list, tuple)) and len(ref[k]) == 2), None)
    med = next((ref[k] for k in ("boot_median", "median", "bootstrap_median") if fin(num(ref.get(k)))), None)
    stored = next((d[k] for d in (ref, armA, armA.get("G16") if isinstance(armA.get("G16"), dict) else {}) for k in sorted(d)
                   if isinstance(d[k], bool) and ("unshifted" in k or "percentile" in k or "plan_text" in k)), None)
    cons = None
    if pct is not None and med is not None and fin(num(ref.get("c"))) and isinstance(ref.get("ci90"), (list, tuple)) and len(ref["ci90"]) == 2:
        cons = bool(all(fin(num(x)) for x in list(pct) + list(ref["ci90"])) and all(abs(num(ref["ci90"][i]) - (num(pct[i]) + num(ref["c"]) - num(med))) <= 1e-9 for i in (0, 1)))
    return dict(lo=lo, recentred=None if pct is None else True, consistent=cons, percentile=list(pct) if pct is not None else None, median=med,
                unshifted_in_session=stored, unshifted_r2_ana=None if pct is None else bool(lt(c1, pct[0])))


def arm_a(armA, ref0, labels14):
    st = not_run(armA, "unlearn/armA.json")
    if st != "ran": return dict(status=st)
    g = g16_reference(armA); c1, lo = armA.get("c1"), g["lo"]
    cont_r2 = bool(lt(c1, lo)); cont_s = bool(armA.get("continues")); ran_rounds = int(armA.get("rounds") or len(armA.get("rows") or []))
    out = dict(status="ran", c1=c1, reference=armA.get("reference_seed0_RSI_a"),
               G16=dict(r2_ana=cont_r2, in_session=cont_s, agrees=cont_r2 == cont_s, interval=("percentile interval recentred on the point estimate (Amendment 3 (a))" if g["recentred"] else
                                                                                               "as stored (no unshifted interval next to it: written before Amendment 3)"),
                        ci90_is_recentred_percentile=g["consistent"], unshifted_percentile_interval=g["percentile"], boot_median=g["median"],
                        unshifted_would_continue=dict(in_session=g["unshifted_in_session"], r2_ana=g["unshifted_r2_ana"])),
               rounds_ran=ran_rounds, labels=list(labels14), od_g8=[] if cont_r2 == cont_s else [f"G16: r2_ana {cont_r2}, in session {cont_s}"])
    if g["consistent"] is False: out["od_g8"].append("G16: the stored ci90 is not the percentile interval shifted by (point - median)")
    if ran_rounds <= 1 or not cont_s:
        out["verdict"] = TRANSLATION
    else:
        comp = {d: r for d, r in (get(ref0, "rows") or {}).items()} if ref0 else None
        comp = {int(k): v for k, v in comp.items()} if comp else None
        out["verdict"] = loop_rule(rows_by_d(armA.get("rows")), comp, "the R1 seed-0 comparator")
        out["comparator"] = {d: {k: comp[d].get(k) for k in ("c", "p", "A", "rel")} for d in (comp or {}) if d in (2, 3)}
    out["in_session"] = armA.get("verdict_in_session")
    # Amendment 3 (a) is declared post hoc: the frozen text's reading (G16 with the unshifted percentile interval) next to the result
    up = g["unshifted_r2_ana"]; continued = not (ran_rounds <= 1 or not cont_s)
    if up is None: out["under_plan_text"] = None
    elif up == continued: out["under_plan_text"] = dict(G16=up, verdict=out["verdict"], same_as_amended=True)
    elif continued: out["under_plan_text"] = dict(G16=False, verdict=TRANSLATION, same_as_amended=False, note="with the unshifted percentile interval arm A stops after round 1; rounds 2-3 would not have run")
    else: out["under_plan_text"] = dict(G16=True, verdict="not computable (with the unshifted percentile interval arm A continues, but rounds 2-3 were not generated)", same_as_amended=False)
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
    try: bank, ST, _ = RP.load_bank(r1_dir, 14)
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
                                                rel_sum=get(sp, "rel_sum", "rel"), coef_band=list(SUP_COEF), in_session=sess, agrees_in_session=agree(holds, sess),
                                                descriptive={k: x for k, x in sp.items() if "median" in k or k in ("cos_ci90",)} or None),
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


def dm_info(dm):
    """A damage-matched control record (OD-C2; TC.damage_match's dict under 'match', or flat), read defensively: dict(matched, d_star
    (the dimension found: 'r' / 'd_star'), r0, lam ('lam' / 'lambda'), target, drop, note, amended, plan_text). amended = the step of
    Amendment 2 applied (partial removal at d* > r0). plan_text = the control under the frozen text: where the amended step applied it is
    the full removal at d* (its drop = the ladder's last entry), which is not matched within 0.02; otherwise the same control."""
    dm = dm or {}; m = dm.get("match") if isinstance(dm.get("match"), dict) else dm
    pick = lambda *ks: next((m[k] for k in ks if isinstance(m, dict) and k in m and m[k] is not None), None)
    d_star, lam, r0, target, drop, note = pick("d_star", "dstar", "d*", "r"), pick("lambda", "lam"), pick("r0"), pick("target"), pick("drop"), str(pick("note") or "")
    matched = pick("matched", "dm_matched")
    if matched is None: matched = dm.get("dm_matched")
    lad = pick("ladder") or []
    amended = ("Amendment 2" in note) or bool(fin(num(d_star)) and fin(num(r0)) and num(d_star) > num(r0) and fin(num(lam)) and num(lam) < 1)
    ld = num(lad[-1][1]) if (lad and isinstance(lad[-1], (list, tuple)) and len(lad[-1]) > 1) else float("nan")
    if not amended: pt = dict(same_as_amended=True, matched=bool(matched))
    else: pt = dict(same_as_amended=False, matched=bool(fin(ld) and fin(num(target)) and abs(ld - num(target)) <= DM_TOL), drop=ld,
                    control="full removal at d* (the frozen OD-C2 has no partial removal above r0; this control was not run)")
    return dict(matched=bool(matched), d_star=d_star, r0=r0, lam=lam, target=target, drop=drop, note=note or None, amended=amended, plan_text=pt)


NOT_TESTABLE_DM = "not testable (no damage-matched control within 0.02)"
PLAN_TEXT_UNLEARNS = "not computable (the plan-text control, full removal at d*, was not run; it overshoots the target by more than 0.02)"


LADDER_NC = "not computable under the plan text (the rung rests on a damage-matched control that the frozen OD-C2 does not provide)"


def plan_text_view(di, acc_ok=None):
    """What the frozen text gives where Amendment 2 changed the control (None where both readings coincide). acc_ok: the accuracy half
    of 'unlearns' / 'causal' (upper bound <= 0.11), which needs no control: where it fails, 'unlearns' is False under either reading;
    where it holds, the control half cannot be evaluated (the plan-text control was not run)."""
    if not di["amended"] or di["plan_text"]["matched"]: return None
    return dict(dm_matched=False, specificity=NOT_TESTABLE_DM, unlearns=False if acc_ok is False else PLAN_TEXT_UNLEARNS, plan_text_control_drop=di["plan_text"].get("drop"), target=di["target"])


def ladder_rung(ops, U):
    """C-T4 ladder (critique 4, v2-5) with a three-valued 'unlearns' U(op) (True / False / None = not computable): 'one direction' if the
    block of u1 alone unlearns; else 'subspace' with r* = the fewest of the top 2, 4, 6 task PCs whose block unlearns while its
    within-span control does not unlearn; else r* = 11 if the span block unlearns while its isotropic control does not; 'not specific'
    if that control unlearns too; '> 11' if the span block does not unlearn."""
    u = U("blk1")
    if u is None: return dict(rung=LADDER_NC)
    if u: return dict(rung="one direction", within_span_1dim_control=get(ops, "blk1", "span_status"))
    for k in (2, 4, 6):
        if get(ops, f"blk{k}", "span_status") != "does not unlearn": continue
        u = U(f"blk{k}")
        if u is None: return dict(rung=LADDER_NC)
        if u: return dict(rung="subspace", r_star=k)
    u = U("block11")
    if u is None: return dict(rung=LADDER_NC)
    if not u: return dict(rung="> 11", r_star="> 11")
    s_ = get(ops, "block11", "iso_status")
    if s_ == "does not unlearn": return dict(rung="subspace", r_star=11, label="subspace: the whole task span")
    return dict(rung="not specific") if s_ == "unlearns" else dict(rung="undecided (isotropic control neither unlearns nor keeps >= 0.9 x base)")


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
    pr = t.get("probes") or {}; dm = t.get("dm") or {}; di = dm_info(dm)
    return dict(causal=bool(unlearns_acc(t.get("acc")) and does_not_unlearn(dm.get("acc"), base_acc)),
                correlate=bool(does_not_unlearn(t.get("acc"), base_acc) and le(hi(pr.get("task_lin")), chance + 0.05)),
                nonlinear_residue=bool(ge(lo(pr.get("task_mlp")), chance + 0.05)) if pr.get("task_mlp") is not None else None,
                dm_matched=di["matched"], dm=di, under_plan_text=(lambda v: None if v is None else dict(dm_matched=False, causal=v["unlearns"]))(plan_text_view(di, unlearns_acc(t.get("acc")))),
                acc=tci(t.get("acc")), dm_acc=tci(dm.get("acc")), task_lin=tci(pr.get("task_lin")),
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
    op, dm = o.get("op") or {}, o.get("dm") or {}; di = dm_info(dm); matched = di["matched"]
    rl, an = arr(get(op, "relearn", sb)), arr(get(op, "A_new", sb)); d = rl - an if len(rl) == len(an) else arr([])
    if ge(mean(d), T4_SHALLOW) and gt(lo(d), T4_SHALLOW_LO): depth = "shallow"
    elif le(mean(rl), (mean(an) + T4_DEEP) if mean(an) is not None else None): depth = "deep"
    else: depth = "partial"
    n_perm = sum(lb_ == "consistent permutation" for lb_ in (op.get("label") or []))
    unl = bool(unlearns_acc(op.get("acc")) and does_not_unlearn(dm.get("acc"), a0))
    spec = (NOT_TESTABLE_DM if not matched else ("specific" if ge(mean(op.get("sp")), 0.8 * sp0 if sp0 is not None else None) else "generic damage"))
    def status(acc):
        if does_not_unlearn(acc, a0): return "does not unlearn"
        if unlearns_acc(acc): return "unlearns"
        return "neither"
    return dict(unlearns=unl, specificity=spec, depth=depth, meta_unlearned=le(hi(get(op, "B2", "8")), 0.05),
                inverted=bool(lt(hi(op.get("acc")), CHANCE_A) and ge(mean(op.get("sp")), 0.8 * sp0 if sp0 is not None else None) and n_perm >= q),
                iso_status=status(get(o, "iso", "acc")), span_status=status(get(o, "span", "acc")) if o.get("span") else None, dm_status=status(dm.get("acc")),
                dm_matched=matched, dm=di, under_plan_text=plan_text_view(di, unlearns_acc(op.get("acc"))), acc=tci(op.get("acc")), dm_acc=tci(dm.get("acc")), sp=tci(op.get("sp")), relearn_minus_new=tci(d),
                relearn_long=tci(get(op, "relearn", lb)), A_new_long=tci(get(op, "A_new", lb)), B2_8=tci(get(op, "B2", "8")),
                consistent_permutation_seeds=n_perm, modal_labels=op.get("label"))


def ct4(t4, gmeta):
    if t4 is None: return dict(status=not_run(t4, "unlearn_toy/t4.json"))
    sb, lb = budgets(t4); out = dict(status="ran", cells={}, w4={}, budgets=[sb, lb])
    for name, c in (t4.get("cells") or {}).items():
        base = c.get("base") or {}; a0, sp0 = mean(base.get("acc")), mean(base.get("sp")); n = len(base.get("acc") or []); q = RP.quorum(n)
        ops = {nm: ct4_op(o, a0, sp0, sb, lb, q) for nm, o in (c.get("ops") or {}).items()}
        lad = ladder_rung(ops, lambda nm: bool(get(ops, nm, "unlearns")))
        if any(get(ops, nm, "under_plan_text") for nm in ("blk1", "blk2", "blk4", "blk6", "block11")):          # Amendment 2: the frozen text's reading next to it
            pt = lambda nm: ((None if get(ops, nm, "under_plan_text", "unlearns") is not False else False) if get(ops, nm, "under_plan_text") else bool(get(ops, nm, "unlearns")))
            lad["under_plan_text"] = ladder_rung(ops, pt)
        W = c.get("W")
        meta_lbl = [] if gmeta else [RP.LBL_GMETA]
        out["cells"][name] = dict(W=W, base_acc=a0, base_sp=sp0, ops=ops, ladder=lad, t4b={L: t4b_rule(t, a0, CHANCE_T) for L, t in (c.get("t4b") or {}).items()},
                                  meta_level_labels=meta_lbl, agrees_preview=agree(lad.get("rung"), get(t4, "rule_preview", "cells", name, "ladder", "rung")))
    for name, c in (t4.get("w4") or {}).items():
        b = c.get("base") or {}; a0 = mean(b.get("add")); o2 = {}
        for nm, o in (c.get("ops") or {}).items():
            op, dm = o.get("op") or {}, o.get("dm") or {}; di = dm_info(dm); matched = di["matched"]
            o2[nm] = dict(forget_unlearns=bool(unlearns_acc(op.get("add")) and does_not_unlearn(dm.get("add"), a0)), dm=di, under_plan_text=(lambda v: None if v is None else dict({k_: x for k_, x in v.items() if k_ != "unlearns"}, forget_unlearns=v["unlearns"]))(plan_text_view(di, unlearns_acc(op.get("add")))),
                          specificity=(NOT_TESTABLE_DM if not matched else
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
            per_task_r = {s: [int(np.argmax(A[i, :la + 1] <= UNLEARN_ACC)) if np.any(A[i, :la + 1] <= UNLEARN_ACC) else T5_NEVER for i in by_seed[s]] for s in sd}
            arms[arm] = dict(A_free=[float(np.mean(A[by_seed[s], 0])) for s in sd], A_op=[float(np.mean(A[by_seed[s], la])) for s in sd],
                             A_s4=[float(np.mean(A[by_seed[s], i4])) for s in sd], rstar_seed=[float(np.median(per_task_r[s])) for s in sd],
                             rstar_task=[per_task_r[s] for s in sd], last_round=la)
            arms[arm]["A_op_at_rstar"] = [float(np.mean([A[i, min(int(r), la)] for i, r in zip(by_seed[s], per_task_r[s])])) for s in sd]
            arms[arm]["A_op_t"] = tci(arms[arm]["A_op"])
        o = dict(rounds=R, s4_index=i4, last_index=last, arms=arms, n_seeds=n, never=f"'> {last}' counts as {T5_NEVER} (the plan's '> 24' counting as 25)")
        col, rnd = arms.get("col"), arms.get("rand")
        if col and rnd:
            o["low_dimensional_search"] = bool(sum(r_ <= T5_RSTAR for r_ in col["rstar_seed"]) >= q and sum(ge(a, T5_KEEP * f) for a, f in zip(rnd["A_s4"], rnd["A_free"])) >= q)
            o["re_routing"] = bool(gt(lo(col["A_op"]), UNLEARN_ACC))
        dm_last = [d for d in (w.get("dm") or []) if d.get("round") == last]
        for X, ctl in (("col", "rand"), ("z1", "dmr")):
            if X in arms and ctl in arms:
                ax, ac = arms[X], arms[ctl]; keep = ge(mean(ac["A_op"]), T5_KEEP * mean(ax["A_free"]) if mean(ax["A_free"]) is not None else None)
                v = "un-searched" if (le(hi(ax["A_op"]), UNLEARN_ACC) and keep) else ("partial" if le(mean(ax["A_op"]), T5_PARTIAL * mean(ac["A_op"]) if mean(ac["A_op"]) is not None else None) else "none")
                if ctl == "dmr" and dm_last:
                    di = dm_info(dm_last[0]); o["dm_last"] = di
                    if not di["matched"]: v = "not testable (no damage-matched control within 0.02 at s_last, G45)"
                    if plan_text_view(di) is not None: o["under_plan_text"] = dict(verdict_z1="not testable (no damage-matched control within 0.02 at s_last, G45)", dm_matched=False)
                o[f"verdict_{X}"] = v; o[f"control_{X}"] = dict(control=ctl, A_rand_mean=mean(ac["A_op"]), A_free_mean=mean(ax["A_free"]), control_keeps=keep)
        prev = get(t5, "rule_preview", name) or {}
        o["agrees_preview"] = {k: agree(o.get(k), prev.get(k)) for k in ("low_dimensional_search", "re_routing", "verdict_col", "verdict_z1") if k in o}
        o["pc_arm"] = "descriptive"; o["dm_rounds_amended"] = [d.get("round") for d in (w.get("dm") or []) if dm_info(d)["amended"]]
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
        out["per_width"][str(W)]["dm"] = {f"{side}_{L}": dm_info(get(w, side, "t4b", L, "dm")) for side in ("pc", "bp") for L in ("h1", "h2") if get(w, side, "t4b", L, "dm") is not None}
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
    # Amendment 2 (post hoc): Delta_rand of the relaxation contrast uses the PC side's damage-matched random erasure. Where that control
    # took the amended step, the contrast cannot be computed under the frozen text; the other items keep their values, but their Holm
    # rank depends on the missing ones: a claim stands for certain if m x p <= 0.05 (Holm p <= m p), is excluded if p > 0.05 or
    # |mean| < 0.10 (Holm p >= p), and is otherwise not determined.
    amd = {(W, k.split("_", 1)[1]) for W, v in out["per_width"].items() for k, di in (v.get("dm") or {}).items() if k.startswith("pc_") and di.get("amended") and not di["plan_text"]["matched"]}
    if amd and fam["family_size"]:
        m = fam["family_size"]; ptr = []
        for r in fam["items"]:
            if not r.get("testable"): continue
            if r["item"].startswith("relax_contrast") and (str(r["W"]), r["item"].rsplit("_", 1)[1]) in amd: v = "not computable (Delta_rand rests on a control that the frozen OD-C2 does not provide)"
            elif r["p"] > T7_ALPHA or abs(r["mean"]) < T7_MIN: v = "no claim"
            elif min(1.0, m * r["p"]) <= T7_ALPHA: v = "claim"
            else: v = "not determined (the Holm rank depends on items that cannot be computed under the plan text)"
            ptr.append(dict(W=r["W"], item=r["item"], under_plan_text=v, under_amendment="claim" if r.get("claim") else "no claim"))
        out["under_plan_text"] = dict(items=ptr, claims=[f"W{r['W']} {r['item']}" for r in ptr if r["under_plan_text"] == "claim"],
                                      not_computable=[f"W{r['W']} {r['item']}" for r in ptr if r["under_plan_text"].startswith("not computable")],
                                      not_determined=[f"W{r['W']} {r['item']}" for r in ptr if r["under_plan_text"].startswith("not determined")])
    return out


# ======================================================================================================================================
# Driver
# ======================================================================================================================================
def run(r1_dir, r2_root, out_path):
    t0 = time.time(); RP.set_threads(); inp = RP.Inputs(r1_dir, r2_root, out_path); info = RP.r1_info(r1_dir); lab14 = RP.desc_labels(info, 14)
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
    out["amendment_2"] = amendment_2(out); out["amendment_3"] = amendment_3(out)
    out["summary"] = summary(out)
    out.update(pc_gates=dict(G_meta=pcg.get("G_meta"), G_meta_source=pcg.get("G_meta_source"), G_PC=pcg.get("G_PC"), labels=pcg.get("labels")),
               inputs=inp.record([f"unlearn/{s}.json" for s in J] + [f"unlearn_toy/{s}.json" for s in T] + ["pc_toy/t7.json"]),
               finder=inp.log, r1_finder=info.get("finder"), seconds=time.time() - t0)
    RP.write_json(out_path, out); log(f"Part C verdicts -> {out_path} ({time.time() - t0:.0f}s)")
    return out


def amendment_2(o):
    """Every result whose damage-matched control used the step of Amendment 2 (partial removal at d* > r0), with both readings (section
    14: the amendment is declared post hoc)."""
    aff = []
    for cell, c in (get(o, "C_T4", "cells") or {}).items():
        for nm, op in (c.get("ops") or {}).items():
            if get(op, "dm", "amended"): aff.append(dict(where=f"C-T4 {cell} {nm}", d_star=get(op, "dm", "d_star"), r0=get(op, "dm", "r0"), lam=get(op, "dm", "lam"), matched=get(op, "dm", "matched"),
                                                         under_amendment=dict(unlearns=op.get("unlearns"), specificity=op.get("specificity")), under_plan_text=op.get("under_plan_text")))
        for L, t in (c.get("t4b") or {}).items():
            if get(t, "dm", "amended"): aff.append(dict(where=f"C-T4b {cell} {L}", d_star=get(t, "dm", "d_star"), r0=get(t, "dm", "r0"), lam=get(t, "dm", "lam"), matched=get(t, "dm", "matched"),
                                                        under_amendment=dict(causal=t.get("causal")), under_plan_text=t.get("under_plan_text")))
    for cell, c in (get(o, "C_T4", "w4") or {}).items():
        for nm, op in (c.get("ops") or {}).items():
            if get(op, "dm", "amended"): aff.append(dict(where=f"C-T4 two-family {cell} {nm}", d_star=get(op, "dm", "d_star"), lam=get(op, "dm", "lam"), matched=get(op, "dm", "matched"),
                                                         under_amendment=dict(forget_unlearns=op.get("forget_unlearns"), specificity=op.get("specificity")), under_plan_text=op.get("under_plan_text")))
    for L, t in (get(o, "C_T3", "t4b") or {}).items():
        if get(t, "dm", "amended"): aff.append(dict(where=f"C-T3 circle nets T4b {L}", d_star=get(t, "dm", "d_star"), lam=get(t, "dm", "lam"), matched=get(t, "dm", "matched"),
                                                    under_amendment=dict(causal=t.get("causal")), under_plan_text=t.get("under_plan_text")))
    for W, w in (get(o, "C_T5", "widths") or {}).items():
        if w.get("dm_rounds_amended"): aff.append(dict(where=f"C-T5 {W} dmr arm", rounds=w.get("dm_rounds_amended"), last_round=get(w, "dm_last"),
                                                       under_amendment=dict(verdict_z1=w.get("verdict_z1")), under_plan_text=w.get("under_plan_text")))
    for cell, c in (get(o, "C_T4", "cells") or {}).items():
        if get(c, "ladder", "under_plan_text"): aff.append(dict(where=f"C-T4 {cell} ladder", under_amendment={k: v for k, v in c["ladder"].items() if k != "under_plan_text"}, under_plan_text=c["ladder"]["under_plan_text"]))
    for W, w in (get(o, "C_T7", "per_width") or {}).items():
        for k, di in (w.get("dm") or {}).items():
            if di.get("amended"): aff.append(dict(where=f"C-T7 W{W} {k} (damage-matched random erasure; on the PC side: Delta_rand of the relaxation contrast)", d_star=di.get("d_star"),
                                                  lam=di.get("lam"), matched=di.get("matched"), under_plan_text=dict(dm_matched=bool(get(di, "plan_text", "matched")), note="the plan-text control was not run")))
    if get(o, "C_T7", "under_plan_text"): aff.append(dict(where="C-T7 family (Holm)", under_amendment=dict(claims=get(o, "C_T7", "family", "claims")), under_plan_text=get(o, "C_T7", "under_plan_text")))
    return dict(amendment="Amendment 2 (OD-C2: partial removal at the found dimension d*, one shared lambda), declared post hoc", n_affected=len(aff), affected=aff,
                reading="verdicts use the controls the producers ran (the amended procedure); 'under_plan_text' gives the frozen text's reading: the control is unmatched there, "
                        "specificity 'not testable' (G45), and 'unlearns' / 'causal' cannot be evaluated against a control that was not run")


def amendment_3(o):
    """Amendment 3 (a) (G16's reference interval recentred on the point estimate; declared post hoc with respect to round 1): arm A under
    the amendment (as run) and under the frozen text (the unshifted percentile interval)."""
    a = o.get("C_Q4_armA") or {}; g = a.get("G16") or {}; up = a.get("under_plan_text")
    aff = []
    if up is not None and not up.get("same_as_amended"):
        aff.append(dict(where="C-Q4 arm A (G16)", under_amendment=dict(G16=g.get("in_session"), verdict=a.get("verdict")), under_plan_text=up))
    return dict(amendment="Amendment 3 (a) (G16: the percentile interval shifted so that the bootstrap median coincides with the point estimate), declared post hoc with respect to round 1",
                status=a.get("status"), G16_under_amendment=g.get("in_session"), G16_under_plan_text=get(g, "unshifted_would_continue", "r2_ana"),
                G16_under_plan_text_in_session=get(g, "unshifted_would_continue", "in_session"), same_under_both=None if up is None else bool(up.get("same_as_amended")),
                n_affected=len(aff), affected=aff,
                reading="the in-session G16 (amended interval) governs what was generated; 'under_plan_text' is G16 with the unshifted percentile interval: where it would have "
                        "stopped arm A after round 1 the frozen text's verdict is 'translation (as predicted), not un-searched'; where it would have continued but the session "
                        "stopped, rounds 2-3 do not exist. Parts (b) and (c) of the amendment are clarifications of the theta rules (r2_theta_verdict.py)")


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


# ======================================================================================================================================
# Self-test: synthetic inputs in the producers' formats, one case per verdict branch, gate failure and missing part
# ======================================================================================================================================
def _txt(g=0.3, g0=1.0, hi=0.6, mf=0.05):
    return dict(g=g, g0=g0, ratio=g / g0 if g0 else float("nan"), ratio_ci90=[g / g0 - 0.1 if g0 else float("nan"), hi], m_concept=abs(g - g0) / g0 if g0 > 0 else float("nan"),
                m_foreign=dict(sports=mf, tone=mf, sea=mf, TOPIC=mf), own_axis_gap=g)


def _q2(**kw):
    t = lambda: _txt(**{k: kw[k] for k in ("g", "g0", "hi", "mf") if k in kw})
    beh = dict(Ue_fin=kw.get("Ue", 0.4), A_fin=0.5, z_fin=kw.get("z", 3.0), M0=kw.get("M0", 0.5))
    return dict(status="ran", GA14=kw.get("GA", True), texts={"u_RSI|RSI": t(), "u_diff|RSI": t(), "u_RGI_perp|RGI": t(), "cov_14_3|RSI": _txt(0.95, 1.0, 1.1), "cov_14_3|RGI": _txt(0.95, 1.0, 1.1)},
                behaviour=dict(u_RSI=dict(beh), u_diff=dict(beh), u_RGI_perp=dict(beh)), alpha_u={k: dict(f=1.0, damage=kw.get("damage", 0.05), damage_plus=0.05) for k in ("u_RSI", "u_diff", "u_RGI_perp", "cov_14_3")},
                rules={k: dict(verdict=kw.get("sess", "suppressed"), ctrl_p5=kw.get("ctrl", 0.9)) for k in ("u_RSI", "u_diff", "u_RGI_perp")}, K6={}, content={}, perplexity={})


def _q3(g=0.3, D=0.1, zD=-3.0, damage=0.02, span_g=0.3, sess_abl="u_RSI", D_hi=0.2, GA=True):
    ab = lambda g_, D_: dict(dim=1, damage=damage, H30=dict(D=D_, D0=0.8, dD=D_ - 0.8, z_D=zD, D_ci90_statements=[D_ - 0.3, D_hi]), RGI30=dict(D=0.5, D0=0.6), texts=dict(RSI=_txt(g_, 1.0, g_ + 0.2), RGI=_txt(g_, 1.0, g_ + 0.2)))
    A = dict(u_RSI=ab(g, D), span=ab(span_g, D), u_RGI_perp=ab(0.8, 0.7), u_diff=ab(0.8, 0.7), ctrl2=ab(0.95, 0.8), null2=ab(0.95, 0.8), null8=ab(0.95, 0.8))
    A["u_RSI"]["unprompted"] = dict(shift_g0=-0.1)
    return dict(status="ran", GA14=GA, ablations=A, ctrl_p5=dict(u_RSI=0.9, span=0.9), rules=dict(u_RSI=dict(verdict="unlearned" if sess_abl == "u_RSI" else "not unlearned"), span=dict(verdict="unlearned" if sess_abl in ("u_RSI", "span") else "not unlearned")),
                c_q4a=dict(ablation=sess_abl, used=sess_abl or "u_RSI"), D0=dict(H30=0.8), g0=dict(RSI=1.0, RGI=1.0), Delta_T=1.0,
                linear_rung=dict(inputs=dict(RSI_1dim=dict(ratio=g, g0=1.0), RSI_span=dict(ratio=span_g, g0=1.0), RGI_1dim=dict(ratio=0.8, g0=1.0), RGI_span=dict(ratio=0.8, g0=1.0))))


def _row(c=0.5, p=0.001, A=0.6, rel=0.7, Ue=0.4, lo_=0.2, d=None):
    r = dict(c=c, p=p, A=A, rel=rel, Ue=Ue, Ue_ci90=[lo_, Ue + 0.2]); return r if d is None else dict(r, d=d, n=d - 1)


def _seeds(v, n=5, sd=0.01):
    """n per-seed values around v (deterministic spread)."""
    return [float(v + sd * (i - (n - 1) / 2)) for i in range(n)]


def _dm(acc, matched=True, amended=False, key="acc"):
    m = dict(r=22 if amended else 11, lam=0.6 if amended else 1.0, drop=0.3, matched=matched, ladder=[[11, 0.2], [22, 0.5]] if amended else [[11, 0.3]],
             note="partial removal at d* (Amendment 2)" if amended else "ladder", target=0.3, r0=11)
    return {key: _seeds(acc), "match": m}


def _op(acc=0.06, dm=0.95, sp=0.9, relearn=0.6, a_new=0.2, B2=0.2, label="scrambled", iso=0.95, span=None, matched=True, amended=False):
    o = dict(op=dict(acc=_seeds(acc), sp=_seeds(sp), relearn={"300": _seeds(relearn), "3000": _seeds(relearn)}, A_new={"300": _seeds(a_new), "3000": _seeds(a_new)}, B2={"8": _seeds(B2)}, label=[label] * 5),
             dm=_dm(dm, matched, amended), iso=dict(acc=_seeds(iso)))
    if span is not None: o["span"] = dict(acc=_seeds(span))
    return o


def _t4(ops, t4b=None, W=128):
    return dict(meta=dict(budgets=[300, 3000]), cells={f"bp_mup_W{W}": dict(W=W, base=dict(acc=_seeds(0.97), sp=_seeds(0.9)), ops=ops, t4b=t4b or {})}, w4={})


def _ladder(**over):
    ops = dict(edit=_op(), block11=_op(acc=0.5), blk1=_op(acc=0.7, span=0.9), blk2=_op(acc=0.5, span=0.9), blk4=_op(acc=0.3, span=0.9), blk6=_op(acc=0.2, span=0.9)); ops.update(over); return _t4(ops)


def _t5(col_last=0.05, col_r=3, rand=0.5, z1_last=0.05, dmr=0.5, dm_matched=True, dm_amended=False, R=25, seeds=5, tasks=4, free=0.5):
    arms = ("col", "rand", "z1", "dmr"); chains, A = [], []
    for arm in arms:
        for s in range(seeds):
            for j in range(tasks):
                chains.append([arm, s, j])
                if arm == "col": A.append([0.5 if r < col_r else col_last for r in range(R)])
                elif arm == "z1": A.append([0.5 if r < 5 else z1_last for r in range(R)])
                else: A.append([free] + [rand if arm == "rand" else dmr] * (R - 1))
    dm = dict(round=R - 1, **_dm(0.9, dm_matched, dm_amended)["match"])
    return dict(meta=dict(budgets=[300, 3000]), widths=dict(W128=dict(W=128, chains=chains, A={"300": A, "3000": A}, arms=list(arms), seeds=list(range(seeds)), dm=[dm], pc_last=11)))


def _t7(diff=0.3, relax=0.0, seeds=5, W=128, amended=(), h2_diff=None):
    side = lambda base: dict(t4b={L: dict(acc=_seeds(base), relearn={"300": _seeds(0.5), "3000": _seeds(0.6)}, dm=_dm(0.9, amended=L in amended)) for L in ("h1", "h2")},
                             unsearch=dict(chains=[["col", s, j] for s in range(seeds) for j in range(4)], A={"300": [[0.4] * 25 for _ in range(seeds * 4)]},
                                           long=dict(chain=list(range(seeds * 4)), round=[24] * (seeds * 4), A_long=[0.45] * (seeds * 4))))
    pc = side(0.5 + diff); bp = side(0.5); pc["t4b"]["h2"]["acc"] = _seeds(0.5) if h2_diff is None else [a_ + d_ for a_, d_ in zip(_seeds(0.5), h2_diff)]
    pc["relax"] = {L: {str(T): dict(contrast=_seeds(relax if (L == "h2" and T == 20) else 0.0, seeds, 0.02)) for T in (20, 100)} for L in ("h1", "h2")}
    return dict(meta=dict(budgets=[300, 3000], quick=False), widths={f"W{W}": dict(W=W, seeds_present=list(range(seeds)), seeds_missing=[], pc=pc, bp=bp)})


def selftest(work):
    import shutil
    import r2_rk as R
    R.MOCK = True
    shutil.rmtree(work, ignore_errors=True); os.makedirs(work); res = []; t_start = time.time()

    def check(name, cond, info=""):
        res.append(dict(test=name, ok=bool(cond), info=str(info)[:300])); print(f"[{'PASS' if cond else 'FAIL'}] {name} {str(info)[:220]}", flush=True)
    mf = dict(sports=0.05, tone=0.05, sea=0.05, TOPIC=0.05)
    # ---- C-Q2
    S = lambda **k: rule_suppressed(**{**dict(g=0.3, g0=1.0, ratio_hi=0.6, ctrl_p5=0.9, Ue=0.4, z=3.0, M0=0.5, damage=0.05, m_foreign=mf, m_concept=0.7), **k})
    check("C-Q2 suppressed when (i)-(iv) hold", S()["verdict"] == "suppressed" and not S()["failed"])
    fails = dict(gap_half=dict(g=0.6), gap_hi=dict(ratio_hi=0.8), gap_below_random=dict(ctrl_p5=0.6), Ue=dict(Ue=0.2), z_fin=dict(z=2.0), damage_band=dict(damage=0.06), foreign=dict(m_foreign=dict(mf, tone=0.4)), g0_positive=dict(g0=-1.0))
    check("C-Q2 every single failing condition gives 'not suppressed' with that condition named", all(S(**kw)["verdict"] == "not suppressed" and k in S(**kw)["failed"] for k, kw in fails.items()),
          {k: S(**kw)["failed"] for k, kw in fails.items()})
    check("C-Q2 M0 <= 0.1: 'no baseline discrimination'; damage below the band and NaN foreign axis fail", "no baseline discrimination" in S(M0=0.05)["notes"] and "Ue" in S(M0=0.05)["failed"]
          and "damage_band" in S(damage=0.04)["failed"] and "foreign" in S(m_foreign=dict(mf, sea=float("nan")))["failed"] and S(ratio_hi=0.9, ctrl_p5=0.9)["failed"] == ["gap_hi", "gap_below_random"])
    import torch
    pt = dict(texts={"cov_14_3|RSI": dict(ratio_draws=torch.linspace(0.8, 1.2, 2001)), "cov_14_3|RGI": dict(ratio_draws=torch.linspace(0.5, 1.2, 2001))})
    c2 = cq2(_q2(), pt, [RP.LBL_GDESC], False)
    check("C-Q2 from the producer's files: ctrl p5 from q2.pt draws, labels G-desc and 'RGI axis not resolved' (u_RGI_perp, u_diff), agreement recorded",
          c2["directions"]["u_RSI"]["verdict"] == "suppressed" and abs(c2["directions"]["u_RSI"]["ctrl_p5"] - 0.82) < 1e-6 and c2["directions"]["u_RGI_perp"]["verdict"] == "not suppressed"
          and c2["directions"]["u_RGI_perp"]["failed"] == ["gap_below_random"] and RP.LBL_RGI in c2["directions"]["u_diff"]["labels"] and RP.LBL_RGI not in c2["directions"]["u_RSI"]["labels"]
          and c2["directions"]["u_RSI"]["labels"] == [RP.LBL_GDESC] and c2["od_g8"] == ["C-Q2 u_RGI_perp: r2_ana 'not suppressed', in session 'suppressed'"], c2["od_g8"])
    check("C-Q2 without the .pt the JSON's ctrl_p5 is used; missing / not-run inputs give their status", cq2(_q2(), None, [], True)["directions"]["u_RSI"]["ctrl_source"].startswith("q2.json")
          and cq2(None, None, [], True)["status"].startswith("not run (round-2 output missing") and cq2(dict(status=NOT_RUN_R1), None, [], True)["status"] == NOT_RUN_R1)
    # ---- C-Q3
    U = lambda **k: rule_unlearned(**{**dict(g=0.3, g0=1.0, ratio_hi=0.6, ctrl_p5=0.9, D=0.1, D0=0.8, zD=-3.0, damage=0.02, m_foreign=mf, m_concept=0.7, D_hi=0.3), **k})
    f3 = dict(D_baseline=dict(D0=0.15), D_half=dict(D=0.5), z_D=dict(zD=-2.0), damage=dict(damage=0.06), gap_half=dict(g=0.51), foreign=dict(m_concept=0.08))
    check("C-Q3 unlearned when every condition holds; each failing condition named", U()["verdict"] == "unlearned" and all(U(**kw)["verdict"] == "not unlearned" and k in U(**kw)["failed"] for k, kw in f3.items())
          and "no baseline discrimination" in U(D0=0.15)["notes"])
    check("C-Q3 inverted only if D_abl < 0 with the statement interval excluding 0", U(D=-0.2, D_hi=-0.05)["inverted"] and not U(D=-0.2, D_hi=0.1)["inverted"] and not U(D=0.1, D_hi=0.3)["inverted"] and U(D=-0.2, D_hi=-0.05)["verdict"] == "unlearned")
    c3 = cq3(_q3(), None, [], True); c3b = cq3(_q3(g=0.8, sess_abl="span"), None, [], False); c3c = cq3(_q3(g=0.8, span_g=0.8, sess_abl=None), None, [], True); c3d = cq3(_q3(g=0.8, sess_abl="u_RSI"), None, [], True)
    check("C-Q3 / G43: u_RSI if it passes, else the span, else none; a differing in-session choice is reported", c3["G43"]["r2_ana"] == "u_RSI" and c3["G43"]["agrees"] and c3b["G43"]["r2_ana"] == "span" and c3b["G43"]["agrees"]
          and RP.LBL_RGI in c3b["rules"]["span"]["labels"] and c3c["G43"]["r2_ana"] is None and c3d["G43"]["agrees"] is False and any("G43" in x for x in c3d["od_g8"]) and "u_RGI_perp|RGI" in c3["descriptive"])
    q3pt = dict(texts={"rand_14_2|RSI": dict(ratio_draws=torch.full((2000,), 0.5)), "cov_14_0|RSI": dict(ratio_draws=torch.full((2000,), 0.55)), "ctrl2|RSI": dict(ratio_draws=torch.full((2000,), 0.95))})
    c3e = cq3(_q3(), q3pt, [], True)
    check("C-Q3 random reference: the two 1-dim controls pooled for u_RSI (5th percentile 0.5 -> upper end 0.5 not below it), the 2-dim control for the span",
          c3e["rules"]["u_RSI"]["verdict"] == "not unlearned" and c3e["rules"]["u_RSI"]["failed"] == ["gap_below_random"] and c3e["rules"]["span"]["verdict"] == "unlearned" and c3e["G43"]["r2_ana"] == "span")
    # ---- C-Q4a
    q4 = lambda share, hi_, used="u_RSI", sess=None: dict(status="ran", ablation=used, passed_C_Q3=True, restored_share=share, restored_share_ci90=[share - 0.1, hi_], verdict_in_session=sess)
    check("C-Q4a: shallow / deep / partial / not applicable", cq4a(q4(0.6, 0.8), c3, [])["verdict"] == "shallow" and cq4a(q4(0.1, 0.3), c3, [])["verdict"] == "deep" and cq4a(q4(0.1, 0.4), c3, [])["verdict"] == "partial"
          and cq4a(q4(0.3, 0.45), c3, [])["verdict"] == "partial" and cq4a(q4(0.6, 0.8), c3c, [])["verdict"] == "not applicable (nothing removed)" and cq4a(q4(0.6, 0.8, used="span"), c3b, [])["verdict"] == "shallow"
          and cq4a(q4(0.6, 0.8, sess="deep"), c3, [])["agrees_in_session"] is False and cq4a(None, c3, [])["status"].startswith("not run")
          and cq4a(dict(q4(0.6, 0.8), own_axis_gap_cue=0.4, own_axis_gap_abl=0.2), c3, [])["own_axis"] == dict(own_axis_gap_cue=0.4, own_axis_gap_abl=0.2) and cq4a(q4(0.6, 0.8), c3, [])["own_axis"] is None)
    # ---- C-Q4b loop rule
    gone, surv, rer, weak = _row(c=0.02, p=0.5, A=0.05, rel=0.2, Ue=0.05, lo_=-0.1), _row(), _row(c=0.1, p=0.03, A=0.15, rel=0.7, Ue=0.4, lo_=0.2), _row(c=0.2, p=0.03, A=0.3, rel=0.3, Ue=0.1, lo_=-0.1)
    gone_rer = _row(c=0.02, p=0.5, A=0.05, rel=0.7, Ue=0.4, lo_=0.2); two = lambda r: {2: r, 3: r}
    check("C-Q4b: un-searched (ablated gone, twin re-finds RSI)", loop_rule(two(gone), two(surv)) == "un-searched")
    check("C-Q4b: survives; re-routed (reproducible read-back, Ue >= 0.25 above 0, |A| < 0.25)", loop_rule(two(surv), two(surv)) == "survives" and loop_rule(two(rer), two(surv)) == "re-routed")
    check("C-Q4b: twin fails -> 'undecided (twin does not re-find RSI)'; but 're-routed' comes first in the plan's order", loop_rule(two(gone), two(weak)) == "undecided (twin does not re-find RSI)"
          and loop_rule(two(gone_rer), two(weak)) == "re-routed" and loop_rule(two(gone_rer), two(surv)) == "un-searched")
    check("C-Q4b: otherwise undecided; a condition must hold at d_2 and d_3; missing rounds", loop_rule(two(weak), two(surv)) == "undecided" and loop_rule({2: gone, 3: weak}, two(surv)) == "undecided"
          and loop_rule({2: surv, 3: weak}, two(surv)) == "undecided" and loop_rule({2: gone}, two(surv)).startswith("undecided (rounds") and loop_rule(two(gone), None, "the R1 seed-0 comparator") == "undecided (the R1 seed-0 comparator does not re-find RSI)")
    info_l = dict(LIKELY_wh=True, from_would_have_read=True, G_attr=False, attracted_loop=None)
    qb = dict(status="ran", gate=dict(open=True, branch="likely", G13=True, G15=True, G14=False), twin_start_check=dict(c=0.3, p=0.001),
              ablated=[_row(d=1), dict(gone, d=2), dict(gone, d=3)], twin=[_row(d=1), _row(d=2), _row(d=3)], verdict_in_session="un-searched")
    b1 = cq4b(qb, info_l, dict(status="ran"), []); qc = dict(status="not testable (gate: G13 (LIKELY_wh) false; G14 (attractor) false)", gate=dict(open=False), twin_start_check=None)
    b2 = cq4b(qc, dict(LIKELY_wh=False, from_would_have_read=False, G_attr=False), None, [RP.LBL_GDESC]); b3 = cq4b(dict(qb, twin_start_check=dict(c=-0.1, p=0.5)), info_l, None, [])
    b4 = cq4b(dict(qb, twin_start_check=None, gate=dict(open=True, branch="attractor")), dict(LIKELY_wh=False, from_would_have_read=False, G_attr=True, attracted_loop="RND_1"), None, [])
    check("C-Q4b gate G44 = (G13 and G15) or G14: recomputed, compared with the session; G-desc label when opened by a would-have-read LIKELY; closed gate 'not testable (gate ...)'",
          b1["verdict"] == "un-searched" and b1["gate"]["open"] and b1["gate"]["agrees"] and RP.LBL_GDESC in b1["labels"] and b1["agrees_in_session"] and b2["verdict"].startswith("not testable (gate") and b2["gate"]["open"] is False
          and b2["gate"]["agrees"] and b3["gate"]["open"] is False and b3["gate"]["agrees"] is False and any("G44" in x for x in b3["od_g8"]) and b4["gate"]["branch"] == "attractor" and b4["gate"]["open"]
          and cq4b(None, info_l, None, [])["verdict"].startswith("not run"))
    # ---- arm A, matched-size reference and C-Q5 on a synthetic round-1 tree
    r1 = os.path.join(work, "r1"); RP.synth_r1(r1, sigma=1.0)
    r1p = os.path.join(work, "r1p"); axp = RP.synth_r1(r1p, seed=1)
    RP.synth_r1(r1p, seed=1, plants={(14, "RSI_a", n): 3.0 * axp[14]["u_T_perp"] for n in (1, 2, 3, 4)}, sigma=2.0, summaries={14: dict(verdict=dict(verdict="likely", PC_PASS=False, label_attractor=False, gates=dict(failed=[])))})
    i_p = RP.r1_info(r1p); ref = seed0_reference(r1p, i_p); ref0 = seed0_reference(r1, RP.r1_info(r1))
    check("R1-6 matched-size reference: seed-0 half (one pair per prompt) of RSI_a on u_T_perp, d_1 .. d_4", ref["status"] == "ran" and sorted(ref["rows"]) == [1, 2, 3, 4] and ref["rows"][2]["n_pairs"] == 16
          and ref["rows"][2]["c"] > 0.5 and ref["rows"][2]["p"] <= 0.01 and ref0["rows"][2]["p"] > 0.01 and seed0_reference(os.path.join(work, "none"), dict(final={"14": False}))["status"] == NOT_RUN_R1, {d: (round(r["c"], 2), r["p"]) for d, r in ref["rows"].items()})
    aA = lambda c1, rows, cont=True: dict(status="ran", c1=c1, continues=cont, rounds=len(rows), rows=rows, reference_seed0_RSI_a=dict(c=0.6, ci90=[0.45, 0.7]), verdict_in_session="x")
    a1 = arm_a(aA(0.5, [_row(d=1)], cont=False), ref, []); a2 = arm_a(aA(0.3, [_row(d=1), dict(gone, d=2), dict(gone, d=3)]), ref, []); a3 = arm_a(aA(0.3, [_row(d=1), dict(gone, d=2), dict(gone, d=3)]), ref0, [])
    a4 = arm_a(aA(0.5, [_row(d=1), dict(surv, d=2), dict(surv, d=3)], cont=True), ref, [])
    check("arm A: stops after round 1 -> 'translation (as predicted), not un-searched'; continued -> loop rule with the R1 seed-0 comparator; G16 recomputed",
          a1["verdict"] == "translation (as predicted), not un-searched" and a1["G16"]["agrees"] and a2["verdict"] == "un-searched" and a2["G16"]["r2_ana"] is True and a3["verdict"] == "undecided (the R1 seed-0 comparator does not re-find RSI)"
          and a4["verdict"] == "survives" and a4["G16"]["agrees"] is False and a4["od_g8"] and arm_a(None, ref, [])["status"].startswith("not run"), (a2["verdict"], a3["verdict"], a4["G16"]))
    # Amendment 3 (a): G16's reference = the percentile interval shifted by (point - median); both readings
    aB = lambda c1, cont, pct, med, ci=None, n=3, **kw: dict(status="ran", c1=c1, continues=cont, rounds=n if cont else 1, rows=([_row(d=1), dict(gone, d=2), dict(gone, d=3)] if cont else [_row(d=1)]),
                                                             reference_seed0_RSI_a=dict(c=0.6, ci90=ci or [pct[0] + 0.6 - med, pct[1] + 0.6 - med], ci90_percentile=pct, boot_median=med), verdict_in_session="x", **kw)
    e1 = arm_a(aB(0.45, True, [0.40, 0.56], 0.5, continues_unshifted=False), ref, []); e2 = arm_a(aB(0.35, True, [0.40, 0.56], 0.5), ref, []); e3 = arm_a(aB(0.55, False, [0.40, 0.56], 0.5), ref, [])
    e4 = arm_a(aB(0.50, False, [0.55, 0.75], 0.7), ref, []); e5 = arm_a(aB(0.45, True, [0.40, 0.56], 0.5, ci=[0.40, 0.56]), ref, [])
    check("Amendment 3 (a), arm A: G16 from the recentred interval [q05 + c - m, q95 + c - m] (ci90[0]); the unshifted interval's G16 and the frozen text's verdict are given next to it",
          e1["G16"]["r2_ana"] is True and e1["G16"]["agrees"] and e1["G16"]["ci90_is_recentred_percentile"] is True and e1["G16"]["unshifted_would_continue"] == dict(in_session=False, r2_ana=False)
          and e1["verdict"] == "un-searched" and e1["under_plan_text"]["verdict"] == TRANSLATION and e1["under_plan_text"]["same_as_amended"] is False and e2["under_plan_text"]["same_as_amended"] is True
          and e2["under_plan_text"]["verdict"] == "un-searched" and e3["verdict"] == TRANSLATION and e3["under_plan_text"]["same_as_amended"] is True
          and e4["verdict"] == TRANSLATION and e4["under_plan_text"]["verdict"].startswith("not computable") and e4["G16"]["agrees"]
          and e5["G16"]["ci90_is_recentred_percentile"] is False and any("not the percentile interval shifted" in x for x in e5["od_g8"]) and a2["under_plan_text"] is None and a2["G16"]["ci90_is_recentred_percentile"] is None,
          (e1["G16"]["unshifted_would_continue"], e1["under_plan_text"]["verdict"], e4["under_plan_text"]["verdict"][:40]))
    a3_ = amendment_3(dict(C_Q4_armA=e1)); a3s = amendment_3(dict(C_Q4_armA=e2)); a3n = amendment_3(dict(C_Q4_armA=a2))
    check("Amendment 3 list: arm A appears with both readings when they differ; 'same under both' otherwise; no reading for an output written before the amendment",
          a3_["n_affected"] == 1 and a3_["affected"][0]["under_plan_text"]["verdict"] == TRANSLATION and a3_["G16_under_amendment"] is True and a3_["G16_under_plan_text"] is False
          and a3s["n_affected"] == 0 and a3s["same_under_both"] is True and a3n["same_under_both"] is None and amendment_3({})["n_affected"] == 0)
    q5 = lambda cd, cr, cs: dict(status="ran", superposition=dict(cos_disatt=cd, cos=cd, coef_RSI=cr, coef_SEA=cs, holds_in_session=True, cos_boot_median=cd - 0.05), dose=dict(norm_ratio=1.8, cos_debiased=0.9, cos_boot_median=0.85))
    s1 = cq5(q5(0.9, 0.7, 0.7), RP.r1_info(r1), r1, []); s2 = cq5(q5(0.7, 0.7, 0.7), i_p, r1p, [RP.LBL_GDESC]); s3 = cq5(q5(0.9, 0.3, 0.7), i_p, r1p, []); s4 = cq5(q5(0.9, 0.7, 1.1), i_p, r1p, []); s5 = cq5(None, RP.r1_info(os.path.join(work, "none")), os.path.join(work, "none"), [])
    check("C-Q5: superposition holds only with disattenuated cos >= 0.8 and both coefficients in [0.5, 1.5] / sqrt 2", s1["superposition"]["holds"] and not s2["superposition"]["holds"] and not s3["superposition"]["holds"] and not s4["superposition"]["holds"]
          and s2["superposition"]["agrees_in_session"] is False and s1["superposition"]["descriptive"] == dict(cos_boot_median=0.9 - 0.05) and s1["dose"]["cos_boot_median"] == 0.85)
    check("C-Q5: +- symmetry from R1 sigma (row 0) in [0.7, 1.4]; label 'block 14 not usable' if G-blk(14) fails; SVD of the 24 stacked d_n with named modes",
          s1["symmetry"]["holds"] is True and s2["symmetry"]["holds"] is False and RP.LBL_BLK14 in s2["labels"] and RP.LBL_BLK14 not in s1["labels"] and s1["svd"]["status"] == "ran" and s1["svd"]["n_directions"] == 24
          and len(s1["svd"]["singular_values"]) == 24 and "mean_RND_response" in s1["svd"]["named_mode_cos_with_top_modes"] and s2["svd"]["named_mode_cos_with_top_modes"]["u_T_perp"][0] > 0.9
          and s5["status"].startswith("not run") and s5["symmetry"]["status"] == NOT_RUN_R1 and s5["svd"]["status"] == NOT_RUN_R1, s2["svd"]["named_mode_cos_with_top_modes"]["u_T_perp"][:2])
    # ---- C-Q6
    def q6(ang=0.9, lo6=0.7, lin=0.1, usable=True, fr=(0.6, 0.7, 0.8), rnd=(0.05, 0.05, 0.05), val=(True, True), rep=(0.9, 0.9), sess=1, diff=0.3, dlo=0.1):
        cap = {}
        for i, r in enumerate((1, 2, 3)):
            cap[f"k4_r{r}_mode"] = dict(frac_all=dict(median=fr[i], ci90=[fr[i] - 0.1, fr[i] + 0.1]), readout_18=dict(ge_3_5_all=rep[0]), readout_22=dict(ge_3_5_all=rep[1])); cap[f"k4_r{r}_rand"] = dict(frac_all=dict(median=rnd[i]))
        rel = dict(angular={str(b): dict(share=ang, ci90=[lo6, 1.0]) for b in (8, 14, 18, 22)}, linear={str(b): dict(share=lin) for b in (8, 14, 18, 22)}, helix={}, kl=dict(usable=usable),
                   readout={"18": dict(validated=val[0]), "22": dict(validated=val[1])}, cap=cap, r_star_in_session=sess)
        return dict(status="ran", relations=dict(improves=rel, describes=rel, repetition=rel), improves_minus_describes_angular={str(b): dict(diff=diff, ci90=[dlo, diff + 0.2]) for b in (8, 14, 18, 22)})
    k1 = cq6(q6(), []); k2 = cq6(q6(ang=0.7), []); k3 = cq6(q6(lo6=0.45), []); k4 = cq6(q6(lin=0.4), [])
    check("C-Q6 helix linearises depth: angular >= 75 % with the interval above 0.5 while linear <= 25 %", k1["statement"]["helix_linearises_depth_at_14"] and not k2["statement"]["helix_linearises_depth_at_14"]
          and not k3["statement"]["helix_linearises_depth_at_14"] and not k4["statement"]["helix_linearises_depth_at_14"])
    check("C-Q6 RSI-specific: improves beats describes by >= 0.2 with the interval excluding 0", k1["statement"]["rsi_specific_at_14"] and not cq6(q6(diff=0.1), [])["statement"]["rsi_specific_at_14"] and not cq6(q6(dlo=-0.05), [])["statement"]["rsi_specific_at_14"])
    k5 = cq6(q6(fr=(0.3, 0.6, 0.8), sess=2), []); k6 = cq6(q6(fr=(0.3, 0.4, 0.45), sess=None), []); k7 = cq6(q6(rnd=(0.3, 0.05, 0.05), sess=1), []); k8 = cq6(q6(usable=False), [])
    check("C-Q6 depth cap: r* = the fewest modes moving >= 50 % with random patches < 20 %; none; KL below the floor 'not testable'; disagreement with the session reported",
          k1["relations"]["improves"]["r_star"] == 1 and k5["relations"]["improves"]["r_star"] == 2 and k6["relations"]["improves"]["r_star"] is None and k6["relations"]["improves"]["depth_cap"].startswith("no r")
          and k7["relations"]["improves"]["r_star"] == 2 and any("C-Q6" in x for x in k7["od_g8"]) and k8["relations"]["improves"]["depth_cap"].startswith("not testable") and not k1["od_g8"])
    check("C-Q6 self-repair: the validated readouts read the capped states as depth >= 3.5 in >= 75 % of objects", k1["relations"]["improves"]["self_repair"] == "self-repair" and cq6(q6(rep=(0.9, 0.5)), [])["relations"]["improves"]["self_repair"] == "no self-repair"
          and cq6(q6(rep=(0.9, 0.5), val=(True, False)), [])["relations"]["improves"]["self_repair"] == "self-repair" and cq6(q6(val=(False, False)), [])["relations"]["improves"]["self_repair"].startswith("not testable")
          and k6["relations"]["improves"]["self_repair"] is None and cq6(None, [])["status"].startswith("not run"))
    # ---- C-T0 .. C-T2
    seed0 = lambda ring, fa: dict(ws=0, tests=dict(a_taskcols=dict(max=0.5 if ring else 0.2, null_max_p99=0.3), c_loo_ring_R2={"1": dict(real=0.5 if ring else -0.1, null_p95=0.2), "2": dict(real=0.1, null_p95=0.2)},
                                                   overlap=dict(k1=0.9 if fa else 0.3, k2=0.4)), flags=dict(ring=ring, fourier_aligned=fa and ring))
    t0 = dict(cells=dict(a=dict(seeds=[seed0(True, True)] * 4 + [seed0(False, False)]), b=dict(seeds=[seed0(True, False)] * 4 + [seed0(True, True)]), c=dict(seeds=[seed0(True, True)] * 3 + [seed0(False, True)] * 2)),
              rule_preview=dict(a=dict(has_ring_code=True, fourier_aligned=True), b=dict(has_ring_code=True, fourier_aligned=True)))
    z0 = ct0(t0)
    check("C-T0: ring code / Fourier-aligned per net from the raw tests, cell property with >= 4 of 5 seeds; preview disagreement visible", z0["cells"]["a"]["has_ring_code"] and z0["cells"]["a"]["fourier_aligned"] and z0["cells"]["b"]["has_ring_code"]
          and not z0["cells"]["b"]["fourier_aligned"] and not z0["cells"]["c"]["has_ring_code"] and z0["cells"]["a"]["agrees_preview"] and z0["cells"]["b"]["agrees_preview"] is False and ct0(None)["status"].startswith("not run"))
    def t1(pca, span):
        grid = list(range(1, 9)); fam = {str(k): dict(pca=dict(acc=[pca[i]] * 5), span=dict(acc=[span[i]] * 5)) for i, k in enumerate(grid)}
        return dict(cells=dict(c=dict(base=dict(acc=[1.0] * 5, B2={"8": [0.2] * 5}), families=dict(xy=fam), grids=dict(xy=grid))))
    lin8 = [0.12 * (i + 1) for i in range(8)]
    check("C-T1: critical length / graceful / intermediate", ct1(t1([0.1, 0.95, 0.96, 0.97, 0.98, 0.99, 1, 1], [0.1, 0.2, 0.3, 0.5, 0.7, 0.92, 1, 1]))["cells"]["c"]["families"]["xy"]["label"] == "critical length"
          and ct1(t1(lin8, lin8))["cells"]["c"]["families"]["xy"]["label"] == "graceful" and ct1(t1([0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0], [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]))["cells"]["c"]["families"]["xy"]["label"] == "intermediate"
          and ct1(t1([0.1, 0.95, 0.96, 0.97, 0.98, 0.99, 1, 1], [0.1, 0.5, 0.95, 1, 1, 1, 1, 1]))["cells"]["c"]["families"]["xy"]["label"] == "intermediate")
    def t2(zs_ring=0.6, al=0.1, cs=0.2, zs_modal=True, l1_ring=(False, True, True)):
        nets = {}
        for ki, k in enumerate((2, 3, 4)):
            for code in ("ring", "pring", "rand"):
                for ws in range(5):
                    he = 0.8 + (al if code == "ring" else 0.0) + 0.01 * ws * (1 if code == "ring" else 0.5)
                    nets[f"{code}_k{k}_w{ws}"] = dict(code=code, k=k, ws=ws, level1=(l1_ring[ki] if code == "ring" else True), zero_shot=(zs_ring if code == "ring" else 0.01), zs_modal=[dict(modal_is_trained=zs_modal)] * 4,
                                                     code_search=0.5 + (cs + 0.01 * ws if code == "ring" else 0.0), heldout_eval=he, B2_8=0.1)
        return dict(nets=nets)
    u1 = ct2(t2(), True); u2 = ct2(t2(zs_ring=0.2, cs=0.0, al=-0.1), True); u3 = ct2(t2(), False); u4 = ct2(t2(zs_ring=0.2, cs=0.2), None)
    check("C-T2: k_min, 'the ring linearises' by zero-shot or by code-space search, snap, 'pattern (one-hot world)' with the Bonferroni interval and G41",
          u1["k_min"] == dict(pring=2, rand=2, ring=3) and u1["ring_linearises"] and u1["ring_linearises_by"]["zero_shot"] and u1["pattern_one_hot_world"] == "pattern (one-hot world)" and not u2["ring_linearises"]
          and u2["pattern_one_hot_world"] == "no pattern (one-hot world)" and u3["pattern_one_hot_world"].startswith("pattern claim withheld") and u4["ring_linearises_by"] == dict(zero_shot=False, code_search=True)
          and u1["cells"]["pring_k2"]["snap"] and not u1["cells"]["ring_k3"]["snap"] and not ct2(t2(zs_modal=False), True)["cells"]["pring_k2"]["snap"] and ct2(None, True)["status"].startswith("not run"))
    # ---- C-T3
    def t3(byj3=None, byj6=None, conj_acc=0.0, share=1.0, arc=0.3, spread=0.6):
        gate = dict(K1=dict(ring=[dict(ws=i, zero_shot=0.99, hatch=dict(extract=125, first95=550)) for i in range(5)], pring=[dict(ws=i, zero_shot=0.05) for i in range(5)], **{"c-onehot": [dict(ws=i, zero_shot=0.06) for i in range(5)]}))
        w = lambda byj: dict(arc=[dict(ws=i, zero_shot=arc + 0.01 * i, by_j=byj, row=dict(eval_point="checkpoint")) for i in range(5)], spread=[dict(ws=i, zero_shot=spread, by_j=byj) for i in range(5)])
        ops = dict(base=dict(acc=_seeds(0.98), sp=_seeds(0.9)), conj=dict(acc=_seeds(conj_acc, sd=0.001), sp=_seeds(0.9), modal_is_minus_c=[[True] * 12] * 4 + [[share >= 1.0] * 12]),
                   t4b=dict(h1=dict(acc=_seeds(0.07), dm=_dm(0.95), probes=dict(task_lin=_seeds(0.09), task_mlp=_seeds(0.1))), h2=dict(acc=_seeds(0.95), dm=_dm(0.95, amended=True), probes=dict(task_lin=_seeds(0.09), task_mlp=_seeds(0.5)))),
                   cplane=dict(acc=_seeds(0.06), zero_shot=_seeds(0.06), rand=dict(acc=_seeds(0.65))))
        return dict(gate=gate, word=dict(L3=w(byj3 or {"1": 0.9, "2": 0.2, "3": 0.1}), L6=w(byj6 or {"1": 0.9, "2": 0.8})), ops=ops)
    inp41 = lambda js: RP.Inputs(None, (RP.write_json(os.path.join(work, "t3", "r2_t3", "unlearn_toy", "t3.json"), js), os.path.join(work, "t3"))[1], None)
    j = t3(); v3 = ct3(j, RP.g41(inp41(j)))
    check("C-T3: gate, j*(L) (contiguous; literal reported), L*, word-length effect, conjugation 'inverted ability', T4b per layer", v3["gate"] is True and v3["word"]["L3"]["jstar"] == [1] * 5 and v3["word"]["L3"]["jstar_literal"] == [1] * 5
          and v3["L_star"] == 6 and v3["word"]["L3"]["word_length_effect"] and v3["conjugation"]["inverted_ability"] and v3["t4b"]["h1"]["causal"] and v3["t4b"]["h2"]["correlate"] and v3["t4b"]["h2"]["nonlinear_residue"]
          and not v3["t4b"]["h1"]["nonlinear_residue"] and v3["t4b"]["h2"]["under_plan_text"]["dm_matched"] is False and v3["t4b"]["h1"]["under_plan_text"] is None, (v3["L_star"], v3["word"]["L3"]["jstar"]))
    j2 = t3(byj6={"1": 0.9, "2": 0.3}, conj_acc=0.3, arc=0.6, spread=0.3); j2["gate"]["K1"]["pring"] = [dict(ws=i, zero_shot=0.5) for i in range(5)]; v3b = ct3(j2, RP.g41(inp41(j2)))
    check("C-T3: gate fails (no pattern claim in Part C), no L*, no word-length effect, no inversion; a gap in the distances stops j*", v3b["gate"] is False and "note" in v3b and v3b["L_star"] is None and not v3b["word"]["L3"]["word_length_effect"]
          and not v3b["conjugation"]["inverted_ability"] and jstar({"1": 0.9, "2": 0.2, "3": 0.9}) == (1, 3, 3) and ct3(None, dict(passes=None))["status"].startswith("not run"))
    # ---- C-T4
    o = ct4(_ladder(), True)["cells"]["bp_mup_W128"]
    check("C-T4 per operation: unlearns (upper bound <= 0.11, damage-matched control >= 0.9 x base), specific, shallow, meta level", o["ops"]["edit"]["unlearns"] and o["ops"]["edit"]["specificity"] == "specific" and o["ops"]["edit"]["depth"] == "shallow"
          and not o["ops"]["edit"]["meta_unlearned"] and not o["ops"]["block11"]["unlearns"] and o["ladder"]["rung"] == "> 11", o["ladder"])
    g_ = lambda **k: ct4(_ladder(**k), True)["cells"]["bp_mup_W128"]
    check("C-T4 per operation: control too damaged -> no 'unlearns'; generic damage; deep / partial; meta unlearned; inverted needs a consistent permutation; unmatched control -> specificity 'not testable' (G45)",
          not g_(edit=_op(dm=0.5))["ops"]["edit"]["unlearns"] and g_(edit=_op(sp=0.3))["ops"]["edit"]["specificity"] == "generic damage" and g_(edit=_op(relearn=0.22))["ops"]["edit"]["depth"] == "deep"
          and g_(edit=_op(relearn=0.35))["ops"]["edit"]["depth"] == "partial" and g_(edit=_op(B2=0.01))["ops"]["edit"]["meta_unlearned"] and g_(edit=_op(acc=0.01, label="consistent permutation"))["ops"]["edit"]["inverted"]
          and not g_(edit=_op(acc=0.01))["ops"]["edit"]["inverted"] and g_(edit=_op(matched=False))["ops"]["edit"]["specificity"].startswith("not testable"))
    check("C-T4 ladder: one direction / subspace r* = 2, 4, 6 / the whole task span (11) / not specific / undecided / > 11", g_(blk1=_op(acc=0.05, span=0.9))["ladder"]["rung"] == "one direction"
          and g_(blk2=_op(acc=0.05, span=0.95))["ladder"] == dict(rung="subspace", r_star=2) and g_(blk2=_op(acc=0.05, span=0.5), blk4=_op(acc=0.05, span=0.95))["ladder"]["r_star"] == 4
          and g_(blk6=_op(acc=0.05, span=0.95))["ladder"]["r_star"] == 6 and g_(block11=_op(acc=0.05, iso=0.95))["ladder"].get("label") == "subspace: the whole task span"
          and g_(block11=_op(acc=0.05, iso=0.05))["ladder"]["rung"] == "not specific" and g_(block11=_op(acc=0.05, iso=0.5))["ladder"]["rung"].startswith("undecided") and g_()["ladder"]["r_star"] == "> 11")
    am = ct4(_ladder(edit=_op(amended=True)), False); e = am["cells"]["bp_mup_W128"]["ops"]["edit"]
    check("Amendment 2: an amended control (partial removal at d* > r0) is used as run; the plan-text reading is given next to it; G-meta label on the meta level",
          e["dm"]["amended"] and e["dm"]["d_star"] == 22 and e["dm"]["lam"] == 0.6 and e["unlearns"] and e["specificity"] == "specific" and e["under_plan_text"]["specificity"] == NOT_TESTABLE_DM
          and e["under_plan_text"]["dm_matched"] is False and am["cells"]["bp_mup_W128"]["ops"]["blk1"]["under_plan_text"] is None and am["cells"]["bp_mup_W128"]["meta_level_labels"] == [RP.LBL_GMETA]
          and dm_info(dict(d_star=22, r0=11, **{"lambda": 0.5}, dm_matched=True))["amended"] and dm_info({})["matched"] is False and dm_info(dict(match=dict(r=11, r0=11, lam=0.5, matched=True, note="partial removal at r0")))["amended"] is False)
    am2 = ct4(_ladder(blk1=_op(acc=0.05, span=0.9, amended=True)), True)["cells"]["bp_mup_W128"]; am3 = ct4(_ladder(blk1=_op(acc=0.7, span=0.9, amended=True), blk2=_op(acc=0.05, span=0.95)), True)["cells"]["bp_mup_W128"]
    am4 = ct4(_ladder(blk1=_op(acc=0.7, span=0.9, amended=True), block11=_op(acc=0.05, iso=0.95, amended=True)), True)["cells"]["bp_mup_W128"]
    check("Amendment 2 and the ladder: the accuracy half of 'unlearns' needs no control (False under both readings where it fails); where it holds, the plan-text rung is 'not computable'",
          am2["ladder"]["rung"] == "one direction" and am2["ladder"]["under_plan_text"] == dict(rung=LADDER_NC) and am2["ops"]["blk1"]["under_plan_text"]["unlearns"] == PLAN_TEXT_UNLEARNS
          and am3["ops"]["blk1"]["under_plan_text"]["unlearns"] is False and am3["ladder"]["r_star"] == 2 and am3["ladder"]["under_plan_text"] == dict(rung="subspace", r_star=2)
          and am4["ladder"]["r_star"] == 11 and am4["ladder"]["under_plan_text"] == dict(rung=LADDER_NC) and "under_plan_text" not in ct4(_ladder(), True)["cells"]["bp_mup_W128"]["ladder"],
          (am2["ladder"]["under_plan_text"], am3["ladder"]))
    t4b = dict(h1=dict(acc=_seeds(0.07), dm=_dm(0.95), probes=dict(task_lin=_seeds(0.09), task_mlp=_seeds(0.09))), h2=dict(acc=_seeds(0.95), dm=_dm(0.95), probes=dict(task_lin=_seeds(0.1), task_mlp=_seeds(0.6))),
               h3=dict(acc=_seeds(0.5), dm=_dm(0.95), probes=dict(task_lin=_seeds(0.5))))
    tb = ct4(_t4(dict(blk1=_op(), block11=_op(), blk2=_op(), blk4=_op(), blk6=_op()), t4b), True)["cells"]["bp_mup_W128"]["t4b"]
    check("C-T4b per layer: causal / correlate with a nonlinear residue / neither", tb["h1"]["causal"] and not tb["h1"]["correlate"] and tb["h1"]["nonlinear_residue"] is False and tb["h2"]["correlate"] and tb["h2"]["nonlinear_residue"]
          and not tb["h2"]["causal"] and not tb["h3"]["causal"] and not tb["h3"]["correlate"] and tb["h3"]["nonlinear_residue"] is None)
    w4 = _t4({}); w4["w4"] = dict(W128=dict(base=dict(add=_seeds(0.95), sub=_seeds(0.95)), ops=dict(block_add=dict(op=dict(add=_seeds(0.06), sub=_seeds(0.82), sp=_seeds(0.5), relearn_add={"300": _seeds(0.4), "3000": _seeds(0.5)}), dm=dict(_dm(0.95, key="add"), sub=_seeds(0.85))),
                                                                                                          erase_h2=dict(op=dict(add=_seeds(0.9), sub=_seeds(0.3), sp=_seeds(0.5), relearn_add={"300": _seeds(0.4), "3000": _seeds(0.5)}), dm=dict(_dm(0.95, key="add"), sub=_seeds(0.85))))))
    w4r = ct4(w4, True)["w4"]["W128"]["ops"]
    check("C-T4 two-family world: forget (add) against retain (sub >= 0.9 x its value under the matched random operation)", w4r["block_add"]["forget_unlearns"] and w4r["block_add"]["specificity"] == "specific" and not w4r["erase_h2"]["forget_unlearns"]
          and w4r["erase_h2"]["specificity"] == "generic damage" and ct4(None, True)["status"].startswith("not run"))
    # ---- C-T5
    f5 = lambda **k: ct5(_t5(**k))["widths"]["W128"]
    v5 = f5()
    check("C-T5: r* per seed (median over tasks), 'low-dimensional search', 'un-searched' for col (control rand) and z1 (control dmr)", v5["arms"]["col"]["rstar_seed"] == [3.0] * 5 and v5["low_dimensional_search"] and v5["verdict_col"] == "un-searched"
          and v5["verdict_z1"] == "un-searched" and v5["re_routing"] is False and v5["arms"]["rand"]["rstar_seed"] == [25.0] * 5, (v5["arms"]["col"]["rstar_seed"], v5["verdict_col"]))
    check("C-T5: re-routing (col lower bound > 0.11), partial (<= 0.5 x control), none, control too weak (no 'un-searched', no 'low-dimensional search'), slow search (r* > 4), unmatched dmr 'not testable' (G45)",
          f5(col_last=0.3)["re_routing"] and f5(col_last=0.2)["verdict_col"] == "partial" and f5(col_last=0.4)["verdict_col"] == "none" and f5(rand=0.3)["verdict_col"] == "partial" and f5(rand=0.3, col_last=0.2)["verdict_col"] == "none" and not f5(rand=0.3)["low_dimensional_search"]
          and not f5(col_r=7)["low_dimensional_search"] and f5(dm_matched=False)["verdict_z1"].startswith("not testable") and f5(dm_amended=True)["under_plan_text"]["verdict_z1"].startswith("not testable")
          and f5(dm_amended=True)["verdict_z1"] == "un-searched" and f5(dm_amended=True)["dm_rounds_amended"] == [24] and "under_plan_text" not in f5() and ct5(None)["status"].startswith("not run"))
    # ---- C-T6
    def t6(n_real=4, ratio=(1.8, 1.7), l1=True):
        modes = [dict(modes=[dict(var=0.5 if (i == 0 and s < n_real) else 0.1, lissajous_r=0.1 if i == 0 else 0.9) for i in range(5)]) for s in range(5)]
        nulls = [dict(var_p99=[0.3] * 5, lis_p5=[0.5] * 5) for _ in range(5)]
        return dict(free=dict(modes=modes, null=nulls, B2={"8": _seeds(0.2)}), caps={"1": dict(level1=[l1] * 5, ratio_Ex=_seeds(ratio[0]), ratio_W2=_seeds(ratio[1]), B2={"8": _seeds(0.1)})})
    check("C-T6: a mode is real in >= 4 of 5 seeds (variance above the nulls' p99 and |corr| with the cosine template below their p5); re-routing under the cap", ct6(t6())["modes"][0]["real"] and not ct6(t6())["modes"][1]["real"]
          and not ct6(t6(n_real=3))["modes"][0]["real"] and ct6(t6())["caps"]["1"]["re_routing"] and not ct6(t6(ratio=(1.8, 1.2)))["caps"]["1"]["re_routing"] and not ct6(t6(l1=False))["caps"]["1"]["re_routing"] and ct6(None)["status"].startswith("not run"))
    # ---- C-T7
    pc_ok = dict(widths_G_PC=[128], G_PC={"128": dict(status="usable")}, G_meta=True); v7 = ct7(_t7(), pc_ok)
    claims = v7["family"]["claims"]; it = {r["item"]: r for r in v7["family"]["items"]}
    check("C-T7: Holm over the 14-item family of a G-PC width; a paired PC - BP difference of 0.3 is a claim, differences of 0 are not", v7["family"]["family_size"] == 14 and claims == ["W128 fwd_acc_h1"] and it["fwd_acc_h1"]["p_holm"] <= 0.05
          and abs(it["fwd_acc_h1"]["mean"] - 0.3) < 1e-9 and it["fwd_acc_h2"]["claim"] is False and v7["verdict"].startswith("claims:"), claims)
    from scipy import stats as _st
    rows_ = holm([dict(W=1, item=f"i{k}", values=v) for k, v in enumerate(([0.5, 0.52, 0.48, 0.51, 0.49], [0.1, -0.1, 0.2, -0.2, 0.05], [0.12, 0.13, 0.11, 0.12, 0.125]))])["items"]
    p_ = [float(2 * _st.t.sf(abs(np.mean(v) / (np.std(v, ddof=1) / math.sqrt(5))), 4)) for v in ([0.5, 0.52, 0.48, 0.51, 0.49], [0.1, -0.1, 0.2, -0.2, 0.05], [0.12, 0.13, 0.11, 0.12, 0.125])]
    order = np.argsort(p_); hp = np.zeros(3); run_ = 0.0
    for rank, i in enumerate(order): run_ = max(run_, min(1.0, (3 - rank) * p_[i])); hp[i] = run_
    check("Holm: step-down adjusted p values equal a direct computation; |mean| >= 0.10 needed for a claim", all(abs(rows_[i]["p_holm"] - hp[i]) < 1e-12 for i in range(3)) and rows_[0]["claim"] and not rows_[1]["claim"] and rows_[2]["claim"]
          and not holm([dict(W=1, item="small", values=[0.05, 0.051, 0.049, 0.05, 0.0505])])["items"][0]["claim"])
    v7b = ct7(_t7(diff=0.0, relax=0.25), pc_ok); v7c = ct7(_t7(), dict(widths_G_PC=[], G_PC={"128": dict(status="BP-equivalent (not used for PC claims)")})); v7d = ct7(_t7(seeds=3), pc_ok); v7e = ct7(_t7(diff=0.0), pc_ok)
    check("C-T7: 'relaxation restores task information' from the PC-only contrast; no G-PC width 'not testable'; fewer than 4 paired seeds not testable; no claim", v7b["relaxation_restores_task_information"] == ["W128 relax_contrast_T20_h2"]
          and v7c["verdict"] == "not testable (no G-PC width)" and v7c["per_width"]["128"]["G_PC"].startswith("BP-equivalent") and v7d["family"]["family_size"] == 0 and v7d["verdict"].startswith("not testable")
          and v7e["verdict"].startswith("no claim") and ct7(None, pc_ok)["status"].startswith("not run"), (v7b["family"]["claims"], v7d["verdict"]))
    zs = [-1.2649, -0.6325, 0.0, 0.6325, 1.2649]; v7p = ct7(_t7(amended=("h1", "h2"), h2_diff=[0.3 + 0.179 * z_ for z_ in zs]), pc_ok); ptv = {r["item"]: r for r in v7p["under_plan_text"]["items"]}
    v7q = ct7(_t7(amended=("h2",), relax=0.25), pc_ok)
    check("Amendment 2 and C-T7: under the plan text the relaxation contrast of an amended layer is 'not computable'; other items: 'claim' if m x p <= 0.05, 'no claim' if p > 0.05, else 'not determined'",
          v7p["family"]["claims"] == ["W128 fwd_acc_h1"] and v7p["under_plan_text"]["claims"] == ["W128 fwd_acc_h1"] and len(v7p["under_plan_text"]["not_computable"]) == 4
          and ptv["fwd_acc_h2"]["under_plan_text"].startswith("not determined") and ptv["fwd_acc_h2"]["under_amendment"] == "no claim" and 0.01 < it["fwd_acc_h1"]["p"] + next(r["p"] for r in v7p["family"]["items"] if r["item"] == "fwd_acc_h2") < 0.05
          and ptv["relearn_300_h1"]["under_plan_text"] == "no claim" and v7q["relaxation_restores_task_information"] == ["W128 relax_contrast_T20_h2"]
          and v7q["under_plan_text"]["not_computable"] == ["W128 relax_contrast_T20_h2", "W128 relax_contrast_T100_h2"] and "under_plan_text" not in v7,
          (v7p["under_plan_text"]["not_determined"], v7q["under_plan_text"]["not_computable"]))
    # ---- end to end: nothing there; then a synthetic root with a subset of the parts
    o0 = run(os.path.join(work, "none"), os.path.join(work, "empty"), os.path.join(work, "out0", "unlearn_verdict.json"))
    check("end to end without any input: every part 'not run', strict JSON written", all(str(get(o0, k, "status")).startswith("not run") for k in ("C_Q1", "C_Q2", "C_Q3", "C_Q4a", "C_Q5", "C_Q6", "C_T0", "C_T4", "C_T7")) and o0["C_Q4b"]["verdict"].startswith("not run")
          and RP.load_json(os.path.join(work, "out0", "unlearn_verdict.json")) is not None and o0["C_T7"]["status"].startswith("not run"))
    r2 = os.path.join(work, "r2"); wj = lambda rel, js: RP.write_json(os.path.join(r2, RP.expected_session(rel), rel), js)
    wj("unlearn/q2.json", _q2(GA=False)); wj("unlearn/q3.json", _q3(GA=False)); wj("unlearn/q4a.json", q4(0.6, 0.8)); wj("unlearn/q4b.json", qb); wj("unlearn/armA.json", aA(0.3, [_row(d=1), dict(gone, d=2), dict(gone, d=3)]))
    wj("unlearn/q5.json", q5(0.9, 0.7, 0.7)); wj("unlearn/q6.json", q6()); wj("unlearn/q1.json", dict(status="ran", statements=dict(RSI="one direction; no nonlinear residue found"), G42=dict(pass_=True), concepts={}))
    wj("unlearn_toy/t3.json", t3()); wj("unlearn_toy/t4.json", _ladder(edit=_op(amended=True))); wj("unlearn_toy/t5.json", _t5(dm_amended=True)); wj("pc_toy/t7.json", _t7())
    rpc = os.path.join(work, "r1full"); RP.synth_r1(rpc, seed=1, plants={(14, "RSI_a", n): 3.0 * axp[14]["u_T_perp"] for n in (1, 2, 3, 4)},
                                                   summaries={14: dict(verdict=dict(verdict="not decidable (pilot gate GO1 failed; would have read: likely)", PC_PASS=True, label_attractor=False, gates=dict(failed=["GO1"])))}); RP.synth_pcs(rpc)
    o1 = run(rpc, r2, os.path.join(work, "out1", "unlearn_verdict.json")); sm = o1["summary"]
    check("end to end on a synthetic root: verdicts, labels (G-desc on block 14, RGI axis), C-T7 family at the G-PC width, Amendment 2 list", sm["C-Q2"]["u_RSI"] == "suppressed" and sm["C-Q3"]["u_RSI"] == "unlearned" and sm["C-Q4a"] == "shallow"
          and sm["C-Q4b"] == "un-searched" and sm["C-Q4 arm A"] == "un-searched" and sm["C-T3"]["gate"] is True and sm["C-T7"].startswith("claims: W128 fwd_acc_h1") and o1["labels_block14"] == [RP.LBL_GDESC]
          and RP.LBL_RGI in o1["C_Q3"]["rules"]["span"]["labels"] and o1["amendment_2"]["n_affected"] >= 3 and any(a["where"].startswith("C-T5") for a in o1["amendment_2"]["affected"])
          and o1["C_T0"]["status"].startswith("not run") and o1["gates"]["G_meta"]["value"] is True and o1["gates"]["G_likely"]["from_would_have_read"], (sm["C-Q4b"], sm["C-T7"], o1["amendment_2"]["n_affected"]))
    ok = all(r["ok"] for r in res); RP.write_json(os.path.join(work, "r2_unlearn_verdict_selftest.json"), dict(ok=ok, tests=res, seconds=time.time() - t_start))
    print(f"r2_unlearn_verdict selftest: {sum(r['ok'] for r in res)}/{len(res)} passed ({time.time() - t_start:.0f}s)", flush=True)
    return ok


if __name__ == "__main__":
    sys.exit(main())
