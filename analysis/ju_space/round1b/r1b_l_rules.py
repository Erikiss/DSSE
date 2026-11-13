"""Part L verdict rules (plan L.3-L.5, with the part file's frozen amendments). Pure functions of statistics dicts; every string is
quoted exactly as in the part file. A statistic is a dict(pt, lo, hi, p_gt0, p_lt0) (r1b_l_stats.summ). Used by L-ANA (r1b_l_ana),
L1-C (r1b_l_a) and L-CAL (r1b_l_cal)."""
import math
import numpy as np

from r1b_l_stats import hi, lo, pt

NT_CAL = "not testable (pipeline calibration failed)"


def LB(s, x=0.0):
    return lo(s) > x


def UB(s, x=0.0):
    return hi(s) < x


def inside(s, m):
    return lo(s) > -m and hi(s) < m


def first(pairs, default):
    for cond, v in pairs:
        if cond:
            return v
    return default


def with_labels(v, labels):
    return dict(verdict=v, labels=[l for l in labels if l])


# ---------------------------------------------------------------------------------------------------------------------------------- L1
def rule_L1(st, cal_ok=True, m_cal=None, sg8=True):
    """L1 tree (first match wins) and flags. st: Priv (bool), rho2 (stat or None), R_rep, xh_ratio, Ihat_P, I_ref, medU1024, m0,
    syn (bool or None), mono (bool), red (bool), Lambda (max), basis_np (bool), sel_robust (bool or None), falsified (bool)."""
    if not cal_ok:
        return with_labels(NT_CAL, [])
    Priv = bool(st.get("Priv"))
    r2 = st.get("rho2")
    syn = st.get("syn")
    v = first([
        (Priv and r2 is not None and hi(r2) < 0.5 and st.get("R_rep") == 1 and (st.get("xh_ratio") or 0) >= 4,
         "localised: the H-cluster is the store"),
        (Priv and r2 is not None and lo(r2) >= 0.5, "H privileged but one of several representations (redundant)"),
        (Priv, "H privileged, redundancy undecided"),
        ((not Priv) and syn is True and st.get("mono"), "holographic-like (threshold <= m_cal, synergistic, monogamous)"),
        (pt(st.get("Ihat_P")) < 0.2 * pt(st.get("I_ref")) and pt(st.get("medU1024")) >= 0.5 * pt(st.get("I_ref")), "dense (no sparse store)"),
        ((not Priv) and st.get("m0") is not None and st["m0"] <= 64, "distributed without a privileged set"),
    ], "unclassified (profile reported)")
    lab = [f"no threshold above m_0 (m_0 = {st.get('m0')})" if st.get("m0") is not None else None,
           "not a strict threshold code (falsified)" if st.get("falsified") else None,
           "monogamous (I3_c < 0)" if st.get("mono") else ("redundant (I3_c > 0)" if st.get("red") else "I3 sign undecided"),
           "latent-scalar redundancy" if (st.get("Lambda") or 0) >= 0.9 else None,
           "neuron basis not privileged" if st.get("basis_np") else None,
           None if st.get("sel_robust") is None else ("selection-robust" if st["sel_robust"] else "selection-fragile"),
           None if sg8 else "no validated H-set", "Syn not testable (no passing L1-C threshold)" if syn is None else None]
    return with_labels(v, lab)


def rule_L1_erasure(st, dmg_ok=True):
    """Curve shape on CC accuracy over the five sizes (mean over draws); causal privilege (Holm over two outcomes)."""
    acc = np.asarray(st["cc_acc_curve"], dtype=float)          # sizes ascending, including the unablated value first
    ch = np.diff(acc); tot = acc[-1] - acc[0]
    shape = "intermediate (curve reported)"
    if abs(tot) <= 0.04:
        shape = "robust up to f = 1/16"
    elif abs(tot) > 0:
        share = np.abs(ch) / max(np.abs(ch).sum(), 1e-12)
        k = int(np.argmax(share))
        if share[k] >= 0.7 and abs(acc[k] - acc[0]) <= 0.04:
            shape = "threshold-like within f <= 1/16"
        elif share.max() <= 0.5 and abs(tot) >= 0.08:
            shape = "graceful degradation (associative-memory sense)"
    if not dmg_ok:
        priv = "not testable (no damage-matched control)"
    else:
        d1, d2 = st["diff_ci_wrong"], st["diff_fab_nonref"]
        from r1b_core import holm
        adj, rej = holm([min(d1["p_gt0"], d1["p_lt0"]) * 2, min(d2["p_gt0"], d2["p_lt0"]) * 2])
        excl = lambda d: lo(d) > 0 or hi(d) < 0
        priv = first([((excl(d1) and rej[0]) or (excl(d2) and rej[1]), "H causally privileged"),
                      (inside(d1, 0.05) and inside(d2, 0.05), "H not causally privileged")], "undecided")
    return dict(curve=shape, privilege=priv)


# ---------------------------------------------------------------------------------------------------------------------------------- L2
def rule_L2a(st):
    if not st.get("gmi_ok", False):
        return dict(growth="not decidable (estimator check failed)", own=st.get("own_verdict_if_gmi_failed", "not decidable (estimator check failed)"))
    g = st["growth"]
    cls, share = g["class"], g["share"]
    if share >= 0.8 and cls == "saturating":
        gv = f"growth class: saturating (area-law-like), L_sat = {st.get('L_sat')}"
    elif share >= 0.8 and cls == "logarithmic" and lo(g["b"]) > 0:
        gv = "growth class: logarithmic (RT/CFT-like)"
    elif share >= 0.8 and cls == "power":
        gv = f"growth class: power law (sub-volume, L2M-like), beta = {pt(g['beta']):.3g}"
    else:
        gv = "growth class undecided (curve reported)"
    R, sl = st.get("R_top"), st.get("lnR_slope")
    if R is None:
        ov = "undecided"
    else:
        ov = first([(hi(R) < 0.87 and sl is not None and hi(sl) < 0 and st.get("deficit_low_eos", True),
                     "the model's own text carries less long-range captured information"),
                    (lo(R) > 0.87 and hi(R) < 1.15, "no deficit")], "undecided")
    return dict(growth=gv, own=ov)


def rule_L2b(st, sg8=True, cal_ok=True):
    """st: b0, bie, biota (stats of P), db0, dbie (P minus mean CTRL), b0_nat_diff (P minus natural), unc/E contrasts, holm."""
    if not sg8:
        return with_labels("not testable (no H-set)", [])
    if not cal_ok:
        return with_labels(NT_CAL, [])
    b0, bie, bi, dbie, db0 = st["b0"], st["bie"], st["biota"], st["dbie"], st["db0"]
    v = first([(LB(b0) and UB(bi) and inside(bie, 0.10), "length effect on H is information-limited"),
               (LB(bie) and pt(bie) >= 0.10 and LB(dbie) and st.get("holm_beyond", True), "H rises beyond information loss"),
               (inside(b0, 0.10), "no length trend of H")], "undecided")
    lab = ["generic (not H-specific) length trend" if LB(b0) and inside(db0, 0.10) else None,
           "chain-specific length effect" if LB(st["b0_minus_nat"]) else None,
           "length trend not beyond uncertainty" if st.get("not_beyond_unc") else None,
           "repetition-sensitive" if st.get("rep_changes") else None,
           None if st.get("sel_robust") is None else ("selection-robust" if st["sel_robust"] else "selection-fragile")]
    return with_labels(v, lab)


def rule_L2c(st, cal_ok=True, sg8=True):
    if not sg8:
        return with_labels("not testable (no H-set)", [])
    if not cal_ok:
        return with_labels("not decidable (cross-map pipeline miscalibrated)", [])
    conv = LB(st["conv_lib"]) and LB(st["conv_cross"])
    v = first([(conv and st["rank_p"] <= 0.01, "boundary encodes the H channel beyond random channels"),
               (conv and st["rank_p"] > 0.05, "boundary encodes H only like any population"),
               (UB(st["conv_cross"], 0.06), "boundary does not encode H")], "undecided")
    return with_labels(v, [])


# ---------------------------------------------------------------------------------------------------------------------------------- L3
def rule_L3a(st, sg8=True, cal_ok=True):
    """Lines: stickiness, persistence, bistability (A2), not distinct. Holm over {Delta gamma, HD} (rank p 0.01 <= 0.025)."""
    if not sg8:
        return dict(verdict="not testable (no H-set)", lines={}, O=False, A2=False)
    if not cal_ok:
        return dict(verdict=NT_CAL, lines={}, O=False, A2=False)
    stick = st["gamma_above_sur"] and st["dgamma_rank_p"] <= 0.025 and LB(st["dgamma_minus_sel"])
    pers = st["hd_rank_p"] <= 0.025 and LB(st["hd_minus_sel"])
    bist = st["bi_above_sur99"] and st["bi_rank_p"] <= 0.01 and LB(st.get("bi_minus_sel", {"lo": float("nan")}))
    nd = st["dgamma_below_c90"] and st["hd_below_c90"] and st["bi_below_c90"]
    lines = {}
    if stick:
        lines["stickiness"] = "H-specific stickiness"
    if pers:
        lines["persistence"] = "H-specific persistence"
    if bist:
        lines["bistability"] = "within-item bistability (H-specific)"
    if not lines:
        lines["main"] = "H dynamics not distinct" if nd else "undecided"
    lab = ["not beyond uncertainty" if st.get("not_beyond_unc") else None,
           None if st.get("sel_robust") is None else ("selection-robust" if st["sel_robust"] else "selection-fragile"),
           "repetition attractor, not H" if st.get("rep_attractor") else None]
    return dict(lines=lines, labels=[l for l in lab if l], O=bool(stick or pers), A2=bool(bist), not_distinct=bool(nd and not lines.get("stickiness")))


def rule_L3b(st, sg8=True, dmg_ok=(True, True)):
    if not sg8:
        return dict(sustain="not testable (no H-set)", specificity="not testable (no H-set)", route="not testable (no H-set)", A1=False)
    if not st["g_twin"]:
        nd = "not decidable (common random numbers broken)"
        return dict(sustain=nd, specificity=nd, route=nd, A1=False, labels=[])
    if not st["g_manip"]:
        return dict(sustain="not testable (push too weak)", specificity="not testable (push too weak)", route="not testable (push too weak)",
                    A1=False, labels=[])
    if not st["g_push"]:
        return dict(sustain="K not defined (push too weak)", specificity="K not defined (push too weak)", route="mixed route", A1=False,
                    labels=[])
    Kp = st["K_pool"]
    sus = first([(LB(Kp, 1.0), "self-sustaining H-state"), (UB(Kp, 1.0), "dissipative")], "undecided")
    if not all(dmg_ok):
        spec = "not testable (no damage-matched control)"
    elif st["n_defined_KR"] < 4:
        spec = "not testable (random carries undefined)"
    else:
        dK = st["dK"]
        spec = first([(LB(dK), "H-specific carry"), (inside(dK, 0.40), "carry like damage-matched controls")], "undecided")
    kv = st["kv_share"]
    route = first([(pt(kv) >= 0.8 and lo(kv) >= 0.5, "carried mainly by the KV cache"), (pt(kv) <= 0.2 and hi(kv) <= 0.5, "carried by the text")],
                  "mixed route")
    A1 = sus == "self-sustaining H-state" and spec == "H-specific carry"
    lab = ["high-damage push" if st.get("damage_H", 0) > 0.15 else None,
           "generic H rise under damage" if st.get("n_cross_pos", 0) >= 3 else None,
           "few sets" if st.get("few_sets") else None]
    one = first([(LB(st["K1"], 1.0), "self-sustaining H-state"), (UB(st["K1"], 1.0), "dissipative")], "undecided") if st.get("K1") else "K not defined"
    return dict(sustain=sus, specificity=spec, route=route, A1=A1, one_step=one, labels=[l for l in lab if l])


def rule_L3c(st, sg8=True, cal_ok=True, gsel=True):
    if not sg8:
        return with_labels("not testable (no H-set)", [])
    if st.get("not_testable"):
        return with_labels(st["not_testable"], [])
    if not cal_ok:
        return with_labels(NT_CAL, [])
    static = st["gism"] and pt(st["i2in"]) >= 0.8 and lo(st["i2in"]) >= 0.7 and st["dpl_rank_p"] <= 0.025 and LB(st["dpl_minus_sel"]) \
        and LB(st["dpl_minus_sel_iv"])
    kinetic = st["gpair_rank_p"] <= 0.025 and LB(st["gpair_minus_sel"])
    rank_only = (st["gism"] and st["dpl_rank_p"] <= 0.025) or st["gpair_rank_p"] <= 0.025
    v = first([(static or kinetic, "collective (Ising-modellable) H-population" if gsel else
                "collective / specific against random populations (selection effect not excluded)"),
               (rank_only, "collective, but no more than a selection-matched set (selection effect)"),
               (st["dpl_below_c90"] and st["gpair_below_c90"], "pairwise structure not H-specific"),
               ((not st["gism"] and not st.get("gism_k", False)) or hi(st["i2in"]) < 0.6, "not Ising-modellable (higher-order structure)"),
               (hi(st["in_per_unit"]) < 0.005, "independent H-units")], "undecided")
    lab = ["not beyond uncertainty" if st.get("not_beyond_unc") else None,
           None if st.get("sel_robust") is None else ("selection-robust" if st["sel_robust"] else "selection-fragile")]
    return with_labels(v, lab)


def rule_L3d(st, sg8=True, cal_ok=True):
    if not sg8:
        return dict(ebm="not testable (no H-set)", irrev="not testable (no H-set)")
    if not cal_ok:
        return dict(ebm=NT_CAL, irrev=NT_CAL)
    de, dtt = st["de"], st["dtt"]
    ebm = first([(UB(de) and LB(dtt) and st["de_rank_p"] <= 0.05, "energy-well entry (EBM reading supported)"),
                 (LB(de), "uncertainty regime at entry (not a well)"),
                 (inside(de, 0.10) and inside(dtt, 0.30), "no energy signature at entry")], "undecided")
    ir = st["irr_a"]            # a_P channel (primary); Holm over a_P and s in st['irr_holm']
    irv = first([(LB(ir["post_minus_rev"]) and LB(ir["post_minus_match"]) and st.get("irr_holm_a", True),
                  "irreversible H-entry (entry-specific)"),
                 (ir["post_above_band"] and ir["match_above_band"] and inside(ir["post_minus_match"], ir["band_half"]),
                  "irreversibility not entry-specific"),
                 (inside(ir["post_minus_rev"], ir["band_half"]), "reversible within resolution")], "undecided")
    lab = ["H-specific" if st.get("irr_rank_p", 1) <= 0.05 else None]
    return dict(ebm=ebm, irrev=irv, irrev_labels=[l for l in lab if l],
                sigma="descriptive (sigma pipeline check failed)" if not st.get("g_sigma", False) else "descriptive")


def attractor_line(L3a, L3b, x1_capture=None):
    """L3 combined attractor line (core: A1, A2 and O; X1's capture attached as a label only, part-file amendment)."""
    A1, A2, O = L3b.get("A1", False), L3a.get("A2", False), L3a.get("O", False)
    Kp_ub_lt1 = L3b.get("sustain") == "dissipative"
    push_bad = L3b.get("sustain") in ("not testable (push too weak)", "K not defined (push too weak)")
    nd = L3a.get("not_distinct", False)
    v = first([((A1 or A2) and O, "H-attractor (self-sustaining or bistable, H-specific, seen in natural dynamics)"),
               (A1 or A2, "self-sustaining H-state under push, not seen in natural dynamics"),
               (O and Kp_ub_lt1 and not A2, "sticky but not self-sustaining"),
               (Kp_ub_lt1 and not A2 and not O and nd, "no H-attractor (dissipative, not distinct)"),
               (Kp_ub_lt1 and not A2 and not O and not nd, "dissipative (observational lines inconclusive)"),
               (push_bad and not A2, "not decidable (push too weak)")], "undecided")
    lab = ["capture above a dose threshold (X1, tier 2)" if x1_capture else None]
    return with_labels(v, lab)


# ---------------------------------------------------------------------------------------------------------------------------------- L4
def rule_L4(st):
    if not st["g_duet"]:
        nt = "not testable (format / readout)"
        return dict(pair=nt, collaboration=nt, entity=nt, h_line=nt, catfish=nt, macro=nt)
    J = st["J_ex"]
    pair = first([(LB(J) and LB(st["ll_gain"]), "agents Ising-modellable (coupled)"), (inside(J, 0.12), "no detectable inter-agent coupling")],
                 "undecided")
    dc = st["J_ex_minus_doc"]
    col = first([(pair == "agents Ising-modellable (coupled)" and LB(dc), "collaboration (coupling beyond lexical priming)"),
                 (LB(J) and inside(dc, 0.16), "influence explained by lexical priming (no social coupling)")], "undecided")
    ent = first([(lo(st["consensus"]) >= 0.5 and LB(st["agree_excess"]), "merge into one entity (consensus)"),
                 (hi(st["consensus"]) <= 0.2 and lo(st["gap_ratio"]) >= 0.8, "two entities persist")], "mixed")
    if not st["g_manip"]:
        hl = "not testable (H gain has no specific effect on agent A)"
    elif not st.get("dmg_ok", True):
        hl = "not testable (no damage-matched control)"
    else:
        DD, DDd = st["DD"], st.get("DD_down")
        up_pass = LB(DD)
        hl = first([(up_pass and st.get("holm_up", True), "H gain acts as coupling (conformity)"),
                    (inside(DD, 0.17) and (lo(st["rup_minus_half"]) > 0 or hi(st["rup_minus_half"]) < 0), "H gain acts like damage"),
                    (inside(DD, 0.17) and inside(st["hup_minus_half"], 0.17), "no effect of H gain on coupling")], "undecided")
    cf = st.get("catfish")
    cat = "K not defined" if cf is None else first([(LB(cf, math.log(1.25)), "competitor sustains the stance (catfish)"),
                                                   (UB(cf, -math.log(1.25)), "competitor captures or dilutes the stance"),
                                                   (inside(cf, math.log(1.25)), "no catfish effect")], "undecided")
    ps = st["dpsi"]
    mac = first([(UB(ps), "redundant pair macro (conformity)"), (LB(ps), "synergistic pair macro"),
                 (inside(ps, 0.025), "no pair macro beyond uncoupled agents")], "undecided")
    return dict(pair=pair, collaboration=col, entity=ent, h_line=hl, catfish=cat, macro=mac)


def x2_label(st):
    """X2 (tier 2): the H-down arm, attached to the core H line as a label (part-file amendment)."""
    if st is None:
        return None
    DDd = st.get("DD_down")
    if DDd is None:
        return None
    return "H-down arm: DD_down upper bound < 0 (X2)" if UB(DDd) else "H-down arm: no conformity signature (X2)"


# ---------------------------------------------------------------------------------------------------------------------------------- L5
def rule_L5a(st, gcl=True):
    """Amended order (validity fix): first match in 'no growth', 'owned by the model', 'shared', 'owned by the lens'."""
    if not gcl:
        return "not testable (no exact-label lens corpus from K)"
    gk, om_p, om_a, tot, gmp = st["GF_known"], st["Omega_prone"], st["Omega_all"], st["G_all"], st["GMc_prone"]
    return first([(inside(tot, 0.04), "no growth of false readouts with chain length"),
                  (inside(gk, 0.04) and LB(om_p, 0.5) and LB(gmp), "break-off owned by the model"),
                  (LB(gk) and LB(om_p, 0.5), "shared (the lens adds confabulation to the model's own errors)"),
                  (LB(gk) and UB(om_p, 0.5), "break-off owned by the lens")], "undecided")


def rule_L5b(st, gcl=True, sg8=True):
    if not gcl or not sg8:
        return with_labels("not testable", [])
    rb, rw = st["rho_b"], st["rho_w"]
    v = first([(LB(rb) and inside(rw, 0.05), "H-pattern owned by the model"), (LB(rw) and inside(rb, 0.12), "H-pattern owned by the lens"),
               (LB(rb) and LB(rw), "H-pattern in both"), (inside(rb, 0.12) and inside(rw, 0.05), "no H-pattern")], "undecided")
    return with_labels(v, ["H-specific (model frame)" if st.get("rank_p", 1) <= 0.05 else None])


def rule_L5c(st, sg8=True, cal_ok=True):
    if not sg8:
        return with_labels("not testable (no H-set)", [])
    if not st.get("parse_ok", False):
        return with_labels("not testable", ["G-parse failed"])
    if st.get("n_events", 0) < 150:
        return with_labels("not testable (fewer than 150 within-item error events)", [])
    if not cal_ok:
        return with_labels(NT_CAL, [])
    b, b0 = st["beta_H"], st["beta_H_noe"]
    v = first([(LB(b) and st["rank_p"] <= 0.05, "H predicts the model's own step errors (within item, beyond step and uncertainty)"),
               (LB(b0) and not LB(b) and not UB(b), "H tracks uncertainty only"),
               (inside(b, 0.15), "no H signal before own errors")], "undecided")
    lab = ["not beyond uncertainty" if st.get("not_beyond_unc") else None,
           None if st.get("sel_robust") is None else ("selection-robust" if st["sel_robust"] else "selection-fragile")]
    return with_labels(v, lab)


# ---------------------------------------------------------------------------------------------------------------------------------- X
def rule_X1(st):
    out = {}
    for arm in ("UP", "DOWN"):
        a = st.get(arm)
        if not a or not a.get("defined"):
            out[arm] = "undecided"; continue
        out[arm] = first([(lo(a["rho_top"]) >= 0.9 and LB(a["top_minus_low"]) and LB(a["top_minus_R"]),
                           "capture above a dose threshold (bistable, H-specific)"),
                          (hi(a["rho_top"]) < 0.9 and inside(a["top_minus_low"], 0.2), "monostable relaxation (no separatrix within the ladder)")],
                         "undecided")
    cap = [k for k, v in out.items() if v.startswith("capture")]
    out["hysteresis"] = len(cap) == 2
    out["capture"] = bool(cap)
    return out


def rule_X3(st):
    return dict(wrong_fp="wrong fixed points exist" if st["n_seeds_wrongfp"] >= 4 else "not shown",
                attractor="toy H-units form an attractor" if st["n_seeds_attr"] >= 4 else "not shown",
                ising="toy H-population Ising-modellable beyond random" if st["n_seeds_ising"] >= 4 else "not shown")


def rule_X4(st):
    out = {}
    for f, s in st.items():
        out[f] = first([(LB(s["diff"]), "associative completion (H-specific clean-up)"), (inside(s["diff"], 0.01), "no completion beyond controls")],
                       "undecided")
    return out


# ---------------------------------------------------------------------------------------------------------------------------------- L.5
def l_all(attr, spec, l2b):
    a, b = attr["verdict"], l2b["verdict"]
    H = a == "H-attractor (self-sustaining or bistable, H-specific, seen in natural dynamics)"
    none = a == "no H-attractor (dissipative, not distinct)"
    amp_set = ("no H-attractor (dissipative, not distinct)", "sticky but not self-sustaining", "dissipative (observational lines inconclusive)")
    return first([(H and b != "length effect on H is information-limited", "H-attractor not bound to length (lens-free)"),
                  (H and b == "length effect on H is information-limited", "H-attractor with an information-limited length trend"),
                  (a in amp_set and spec == "H-specific carry", "H-specific amplifier, not self-sustaining"),
                  (none and spec != "H-specific carry" and b == "length effect on H is information-limited",
                   "length artefact (information-limited), no H-attractor"),
                  (none and spec != "H-specific carry" and b == "no length trend of H", "no length trend and no H-attractor"),
                  (none and spec != "H-specific carry" and b == "H rises beyond information loss", "H rises beyond information without attractor dynamics")],
                 "mixed (lines reported)")


def v_own(l5a, l5b):
    """Amended V-OWN (validity fix): lens / model only if no line points the other way; 'shared' only from a shared or opposing pair."""
    lens = {"break-off owned by the lens", "H-pattern owned by the lens"}
    model = {"break-off owned by the model", "H-pattern owned by the model"}
    both = {"shared (the lens adds confabulation to the model's own errors)", "H-pattern in both"}
    s = [l5a, l5b]
    nl, nm, nb = sum(x in lens for x in s), sum(x in model for x in s), sum(x in both for x in s)
    v = first([(nb > 0 or (nl > 0 and nm > 0), "shared"), (nl > 0 and nm == 0, "the chain-lens break-off and its H-pattern belong to the lens"),
               (nm > 0 and nl == 0, "belong to the model")], "not decidable")
    carried = [x for x in s if x in lens | model | both]
    return dict(verdict=v, carried_by=carried)


def v_chain(L3c, L3d, L4, x3=None):
    links = [("EBM", L3d["ebm"] == "energy-well entry (EBM reading supported)"),
             ("Non-equilibrium", L3d["irrev"] == "irreversible H-entry (entry-specific)"),
             ("Loop", True),
             ("Agent", L4["pair"] != "not testable (format / readout)"),
             ("Ising-modellable", L3c["verdict"] == "collective (Ising-modellable) H-population" and L4["pair"] == "agents Ising-modellable (coupled)"),
             ("Collaboration", L4["collaboration"] == "collaboration (coupling beyond lexical priming)"),
             ("Entity-forming", L4["entity"] == "merge into one entity (consensus)")]
    bad = [n for n, ok in links if not ok]
    v = "the chain holds in Qwen3-1.7B" if not bad else f"the chain breaks at {bad[0]}"
    lab = ["Loop link: token and turn loops by construction; X3: " + ("; ".join(f"{k}: {x}" for k, x in x3.items()) if x3 else "not run")]
    return dict(verdict=v, links={n: ok for n, ok in links}, labels=lab)


def cf_l_p(l3a_st, l3b_st, l2b_st, l3a_v, l3b_v, l2b_v, attr):
    """CF-L member p-values (main §2.4): attractor line p = max(2 min(p_A1, p_A2), 2 min(p_stick, p_pers)); L2b p = max of two."""
    pA1 = max(l3b_st["K_pool"]["p_gt1"], l3b_st["dK"]["p_gt0"]) if l3b_st.get("K_pool") and l3b_st.get("dK") else 1.0
    pA2 = max(l3a_st.get("bi_sur_p", 1.0), l3a_st.get("bi_rank_p", 1.0))
    ps = max(l3a_st.get("gamma_sur_p", 1.0), l3a_st.get("dgamma_rank_p", 1.0), l3a_st.get("dgamma_minus_sel", {}).get("p_gt0", 1.0))
    pp = max(l3a_st.get("hd_rank_p", 1.0), l3a_st.get("hd_minus_sel", {}).get("p_gt0", 1.0))
    p_att = min(1.0, max(2 * min(pA1, pA2), 2 * min(ps, pp)))
    if attr["verdict"] != "H-attractor (self-sustaining or bistable, H-specific, seen in natural dynamics)":
        p_att = 1.0
    p_l2b = max(l2b_st["bie"]["p_gt0"], l2b_st["dbie"]["p_gt0"]) if l2b_v["verdict"] == "H rises beyond information loss" else 1.0
    from r1b_core import holm
    adj, rej = holm([p_att, p_l2b])
    return dict(p=dict(attractor=p_att, l2b=p_l2b), holm=adj, confirmed=dict(attractor=rej[0], l2b=rej[1]))
