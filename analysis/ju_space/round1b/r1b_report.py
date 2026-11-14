"""Round level of round 1b (main §2.4, §3.3, §5). Model-free, CPU (account A).

r1b_ana   families  the round-level confirmatory families CF-R+, CF-R-, CF-D, CF-K-attr, CF-K-link, CF-L, CF-HERR: Holm at family-wise
                    0.05 over the member p-values of main §2.4 (a member its part's rule does not reach has p = 1); 'confirmatory
                    (<family>)' only if reached and Holm-surviving -> ana/families.json
          report    the four answers (R: V_RSI_1b per block with F+ / F-; D: D.17 with qualifiers incl. L4; K: K.7 composite; L: L.5
                    template), each naming its rule and listing its deviations; cross-part labels (RG10 on K's lens statements, SG8 on
                    D and L, K's lens labels on L5, L4 on D.17); side-by-side tables (attractor, H before errors, catfish); the
                    'for'-family count against its null bound; the confirmatory 'against' statements; the merged manifest of every
                    session and the SG6 report check -> runde1b_summary.json, r1b_manifest.json
r1b_ana2  tier2     tier-2 results next to the four answers, labelled 'conditional run' (exploratory at the round level)
                    -> runde1b_summary_t2.json
Inputs: the copied verdict files (ctx.input), r1b_ana's own l_verdicts.json (L-ANA), and for the manifest the session folders under the
run roots (dsse_runs of account A and the dsse_runs_B shortcut)."""
import os

import r1b_contracts as C
from r1b_core import PROV_REQUIRED, check_output, code_counts, coded, holm, load_json, sha256_file

ALPHA = 0.05
FAMS = ("CF-R+", "CF-R-", "CF-D", "CF-K-attr", "CF-K-link", "CF-L", "CF-HERR")
FOR = ("CF-R+", "CF-D", "CF-K-attr", "CF-K-link", "CF-L", "CF-HERR")
ATTR_L = "H-attractor (self-sustaining or bistable, H-specific, seen in natural dynamics)"
G_K2_OK = "H-neurons replicated (in-domain detection)"
K_POS9 = ("H-neuron attractor", "self-sustaining H", "within-item bistability")
AGAINST = dict(D=("H-DUAL not supported in Qwen3-1.7B",),
               K=("length artifact: the H-hallucination association in the chain lens is carried by", "no H association with lens break-off"),
               L=("length artefact (information-limited), no H-attractor", "no length trend and no H-attractor"))
IN = dict(R=("r1b_R_ana", "r_verdict"), D=("r1b_D_ana", "d_verdict"), K=("r1b_K_ana", "k_verdict"))


def _ld(ctx, s, rel):
    p = ctx.input(s, rel, required=False)
    d = load_json(p, None) if p else None
    return d if isinstance(d, dict) else None


def _txt(x):
    """The verdict string of a string, a labelled value {'value', 'labels'} or a record with 'verdict' / 'string'."""
    if isinstance(x, dict):
        for k in ("value", "verdict", "string"):
            if k in x:
                return _txt(x[k])
        return None
    return x if isinstance(x, str) else None


def _g(d, *keys):
    """Safe descent through nested dicts (None when a level is missing or not a dict)."""
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and x == x else None


def _strip(d):
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if k != "provenance"}
    return [_strip(v) for v in d] if isinstance(d, list) else d


def _labels(d):
    return list(((d or {}).get("provenance") or {}).get("labels") or []) + [str(x) for x in ((d or {}).get("labels") or [])]


def _fam(name, members, available=True, note=""):
    """members: [(member, part, reached, part_p, extra)]; member p = part_p if reached (else 1); Holm at 0.05."""
    if not available:
        return dict(members=[], reached=0, confirmed=0, control="Holm, family-wise 0.05", note=note,
                    code=coded("NA:not_run:input missing"))
    ps = [(_num(m[3]) if _num(m[3]) is not None else 1.0) if m[2] else 1.0 for m in members]
    adj, rej = holm(ps, ALPHA) if ps else ([], [])
    rows = [dict(member=m[0], part=m[1], reached=bool(m[2]), p=p, p_holm=a,
                 label=f"confirmatory ({name})" if (m[2] and r) else "exploratory (Holm not met)", **(m[4] or {}))
            for m, p, a, r in zip(members, ps, adj, rej)]
    return dict(members=rows, reached=sum(r["reached"] for r in rows), confirmed=sum(r["label"] != "exploratory (Holm not met)" for r in rows),
                control="Holm, family-wise 0.05", note=note)


def _wait_l_ana(ctx):
    m = load_json(ctx.path("steps/l_ana.json"), {}) or {}
    if m.get("status") not in ("ok", "not_run"):
        raise RuntimeError("L-ANA (step l_ana of r1b_ana) has not finished; the round-level steps run after it (rerun the session)")
    return m.get("status")


def _inputs(ctx):
    v = {p: _ld(ctx, s, C.FILES[f]["rel"]) for p, (s, f) in IN.items()}
    v["L"] = load_json(ctx.path(C.FILES["l_verdicts"]["rel"]), None)
    return v


# ====================================================================================================================================
# r1b_ana: the confirmatory families (main §2.4)
# ====================================================================================================================================
def families(ctx):
    l_state = _wait_l_ana(ctx)
    V = _inputs(ctx); R, D, K, L = (V[p] or {} for p in "RDKL")
    fam, cons = {}, []
    for name, key in (("CF-R+", "F_plus"), ("CF-R-", "F_minus")):
        fam[name] = _fam(name, [(m.get("member"), "R", m.get("reached"), m.get("p"), dict(part_label=m.get("label")))
                                for m in R.get(key) or []], V["R"] is not None, "R.1's F+ / F- (10 members each); R's own Holm")
        cons += [f"{name} {r['member']}: part label {r['part_label']!r} vs round level {r['label']!r}" for r in fam[name]["members"]
                 if (r["part_label"] == "confirmatory (Holm)") != (r["label"] != "exploratory (Holm not met)")]
    parts = D.get("parts") or {}
    fam["CF-D"] = _fam("CF-D", [(f"D.17 {k}", "D", v.get("support_condition"), v.get("p"),
                                 dict(string=_txt(v.get("verdict")), part_holm_reject=v.get("holm_reject"))) for k, v in parts.items()],
                       V["D"] is not None, "D.17 (p_a, p_b1, p_b2 = min(2 p_M, 2 p_D5)); D's own Holm over three")
    cons += [f"CF-D {r['member']}: part Holm reject {r['part_holm_reject']} with support {r['reached']} vs {r['label']!r}"
             for r in fam["CF-D"]["members"] if bool(r["part_holm_reject"] and r["reached"]) != (r["label"] != "exploratory (Holm not met)")]
    rl, fk = K.get("round_level") or {}, K.get("families") or {}
    fam["CF-K-attr"] = _fam("CF-K-attr", [(k, "K", str(v.get("string") or "").startswith(K_POS9), ((rl.get("CF_K_attr") or {}).get(k) or v).get("p"),
                                           dict(string=v.get("string"))) for k, v in (fk.get("Q9") or {}).items()],
                            V["K"] is not None, "K's Q9 (six members), p_IUT with PI specificity (K.4)")
    fam["CF-K-link"] = _fam("CF-K-link", [(k, "K", v.get("reached"), v.get("p_pi_form"), dict(p_rank_form=v.get("p_rank_form")))
                                          for k, v in (fk.get("A_plus") or {}).items()], V["K"] is not None,
                            "member p = p_IUT with the PI-form specificity (main §2.4 as amended, K.12); the rank form (p_rank_form, floor "
                            "1/40) cannot pass Holm over seven and is reported only")
    ksfx = " ".join(str(x) for x in ((K.get("composite") or {}).get("suffixes") or []) if "round level" in str(x))
    if ksfx and ("A+ confirmatory" in ksfx) != (fam["CF-K-link"]["confirmed"] > 0):
        cons.append(f"CF-K-link: K's own suffix {ksfx!r} (rank form) vs round level {fam['CF-K-link']['confirmed']} confirmed (PI form)")
    LV, LC = L.get("verdicts") or {}, L.get("combined") or {}
    cl = LC.get("CF-L") or {}
    fam["CF-L"] = _fam("CF-L", [("L3 attractor line", "L", _txt(LV.get("attractor")) == ATTR_L, (cl.get("p") or {}).get("attractor"),
                                 dict(string=_txt(LV.get("attractor")))),
                                ("L2b 'H rises beyond information loss'", "L", _txt(LV.get("L2b")) == "H rises beyond information loss",
                                 (cl.get("p") or {}).get("l2b"), dict(string=_txt(LV.get("L2b"))))],
                       V["L"] is not None, "L.2 member p-values (r1b_l_rules.cf_l_p; p_A2 includes p(BI_P - BI_SEL+ <= 0)); X1 / X2 are labels only")
    for r, key in zip(fam["CF-L"]["members"], ("attractor", "l2b")):
        if V["L"] is not None and bool((cl.get("confirmed") or {}).get(key)) != (r["label"] != "exploratory (Holm not met)"):
            cons.append(f"CF-L {r['member']}: L-ANA confirmed {(cl.get('confirmed') or {}).get(key)} vs {r['label']!r}")
    k8c, l5c = (K.get("strings") or {}).get("K8c"), _txt(LV.get("L5c"))
    fam["CF-HERR"] = _fam("CF-HERR", [("K8c 'H-activity predicts the next error ...'", "K", str(k8c or "").startswith("H-activity predicts the next error"),
                                       rl.get("CF_HERR_K8c"), dict(string=k8c)),
                                      ("L5c 'H predicts the model's own step errors ...'", "L", str(l5c or "").startswith("H predicts the model's own step errors"),
                                       (LC.get("CF-HERR_member_L5c") or {}).get("p"), dict(string=l5c))],
                          V["K"] is not None or V["L"] is not None, "new at round level (C21); K8c p_IUT, L5c max(p(beta_H <= 0), rank p)")
    miss = [p for p in "RDKL" if V[p] is None]
    ctx.write_json("ana/families.json", dict(families=fam, part_vs_round_level=cons, verdict_files_missing=miss, l_ana=l_state),
                   config=dict(alpha=ALPHA, families=FAMS, k_link_form="PI", member_rule="main §2.4 (R.1's convention)"),
                   labels=(["MOCK (tiny random model; values are code-test values)"] if ctx.mock else []) +
                   [f"verdict file missing: {p}" for p in miss])


# ====================================================================================================================================
# r1b_ana: sessions, merged manifest and the SG6 report check
# ====================================================================================================================================
def _roots(ctx):
    d = getattr(ctx, "drive", None)
    if not d:
        return []
    runs = os.path.dirname(os.path.abspath(d))
    rb = os.environ.get("R1B_RUNS_B") or os.path.join(os.path.dirname(runs), C.RUNS_B_LINK)
    return [runs] + ([rb] if os.path.isdir(rb) else [])


def _state(roots, s):
    for r in roots:
        d = os.path.join(r, s)
        for f, st in (("session_done.json", "done"), ("session_not_run.json", "not run"), ("session_partial.json", "partial")):
            if os.path.isfile(os.path.join(d, f)):
                return st, d, os.path.join(d, f)
    return "absent", None, None


def _check_files(d, files):
    """SG6 on a finished session's recorded outputs: JSON parsed with its provenance block (and sha256 = record); other files by
    their sidecar (provenance block and the sidecar's sha256 = the session record; large files are not re-hashed)."""
    probs = []
    for rel, sha in sorted((files or {}).items()):
        p = os.path.join(d, rel)
        if rel.endswith(".prov.json"):
            continue
        if not os.path.isfile(p):
            probs.append(f"{rel}: missing"); continue
        if rel.endswith(".json"):
            e = check_output(p) or (None if sha256_file(p) == sha else "sha256 differs from the session record")
        else:
            sc = load_json(p + ".prov.json", None)
            pv = (sc or {}).get("provenance")
            e = ("no sidecar .prov.json" if not sc else "sidecar sha256 differs from the session record" if sc.get("sha256") != sha else
                 None if isinstance(pv, dict) and all(k in pv for k in PROV_REQUIRED) else "sidecar provenance incomplete")
        if e:
            probs.append(f"{rel}: {e}")
    return probs


def manifest(ctx):
    """Every session of main §4.2 (incl. tier 2 and r1b_P0): state, attempts, gates, files with sha256, steps 'not run', and the
    SG6 report check of its outputs (sessions after r1b_ana are not yet run; they are listed with their state)."""
    roots, out, rep = _roots(ctx), {}, {}
    for s, m in C.SESSIONS.items():
        st, d, f = _state(roots, s)
        rec = load_json(f, {}) if f else {}
        e = dict(state=st, part=m["part"], tier=m["tier"], account=m["account"], folder=d)
        if st == "not run":
            e.update(reason=rec.get("reason"), code=rec.get("code"))
        elif st in ("done", "partial"):
            e.update(record=os.path.basename(f), record_sha256=sha256_file(f), attempts=[{k: a.get(k) for k in ("attempt", "result", "started", "ended")}
                                                                                       for a in rec.get("attempts") or []],
                     gates=rec.get("gates"), declared=rec.get("declared"), not_run=rec.get("not_run") or {}, long_steps=rec.get("long_steps"),
                     mock=rec.get("mock"), code_commit=rec.get("code_commit"), slot=rec.get("slot"), files=rec.get("files") or {})
            mf = os.path.join(d, "manifest.json")
            if os.path.isfile(mf):
                e["manifest_sha256"] = sha256_file(mf)
            if st == "done" and s not in ("r1b_ana", "r1b_ana2"):
                rep[s] = ctx.cached(f"sg6_{s}", lambda d=d, r=rec: _check_files(d, r.get("files")), fmt="json")
        out[s] = e
    dec = load_json(os.path.join(roots[0], C.DECLARED), {}) if roots else {}
    return out, rep, (dec or {}).get("sessions", dec or {})


# ====================================================================================================================================
# r1b_ana: the four answers, cross-part labels, side-by-side tables -> runde1b_summary.json
# ====================================================================================================================================
def _funded(fz, item):
    f = (fz or {}).get("funded") or {}
    return bool(f.get("tier2")) and (not f.get("items") or item in f["items"])


def _deviations(part, sess, fz, labels):
    out = []
    for s, m in C.SESSIONS.items():
        if m["part"] != part:
            continue
        e = sess.get(s) or {}
        if e.get("state") != "done":
            out.append(f"{s}: {e.get('state', 'absent')}" + (f" ({e['reason']})" if e.get("reason") else ""))
        out += [f"{s}/{k}: {v}" for k, v in (e.get("not_run") or {}).items()]
    unf = [i for s, it in C.TIER2_ITEMS.items() if C.SESSIONS[s]["part"] == part for i in it if not _funded(fz, i)]
    unf += [s for s, m in C.SESSIONS.items() if m["part"] == part and m["tier"] == 2 and not _funded(fz, s)]
    if unf:
        out.append("tier 2 not funded: " + ", ".join(unf))
    return out + [f"label: {x}" for x in labels]


def report(ctx):
    fams = load_json(ctx.path("ana/families.json"))
    F = fams["families"]
    V = _inputs(ctx); R, D, K, L = (V[p] or {} for p in "RDKL")
    LV, LC = L.get("verdicts") or {}, L.get("combined") or {}
    sess, sg6, declared = manifest(ctx)
    fz = ctx.freeze or {}
    na = coded("NA:not_run:input missing")
    # cross-part labels (main §3.3, §5)
    jv = _ld(ctx, "r1b_R_S5", C.FILES["jlens_validity"]["rel"]) or {}
    rg10 = [f"J-lens not valid at {b} (R5.1)" for b, v in sorted((jv.get("blocks") or {}).items(), key=lambda x: int(x[0]))
            if isinstance(v, dict) and v.get("valid") is False]
    gk = K.get("gates") or {}
    sg8_final, sg8_prev = gk.get("G-K2"), (L.get("sg8") or {}).get("preview")
    sg8_ok = sg8_final == G_K2_OK
    sg8 = dict(final_G_K2=sg8_final, preview=sg8_prev, ok=sg8_ok, L_ANA_used=(L.get("sg8") or {}).get("ok"),
               disagreement=(sg8_prev is not None and sg8_final is not None and sg8_prev != sg8_final),
               D_label=None if sg8_ok else "SG8 not met (r1b_K_ana's G-K2): D's optional H covariate is not reported",
               L_label=None if sg8_ok else "SG8 not met: H-specific verdicts in L2b-L5 read 'not testable (no H-set)' (L's G-H)")
    if V["L"] is not None and V["K"] is not None and bool(sg8["L_ANA_used"]) != sg8_ok:
        sg8["L_disagreement"] = "L-ANA's SG8 value differs from r1b_K_ana's G-K2 (labels use r1b_K_ana's value, main §3.1 SG8)"
    k_lens = {g: gk.get(g) for g in ("G-K4", "G-K4b", "G-K5", "G-K14")}
    l4 = LV.get("L4") if isinstance(LV.get("L4"), dict) else {}
    l4q = {"L4 catfish": l4.get("catfish") or "not run", "L4 entity ('two entities persist')": l4.get("entity") or "not run"}
    # the four answers (main §5)
    blocks = R.get("blocks") or {}
    ans = dict(
        R=dict(rule="part R, R.6: V_RSI_1b per block; F+ / F- (R.1) = CF-R+ / CF-R-",
               per_block={b: dict(V_RSI_1b=_txt(v.get("V_RSI_1b")), labels=_g(v, "V_RSI_1b", "labels"), PC=v.get("PC"),
                                  GO0_1b=v.get("GO0_1b")) for b, v in blocks.items() if isinstance(v, dict)} or na,
               F_plus=dict(reached=F["CF-R+"]["reached"], confirmed=F["CF-R+"]["confirmed"], null_expectation=(R.get("F_plus_counts") or {}).get("null_expectation")),
               F_minus=dict(reached=F["CF-R-"]["reached"], confirmed=F["CF-R-"]["confirmed"], null_expectation=(R.get("F_minus_counts") or {}).get("null_expectation")),
               deviations=_deviations("R", sess, fz, _labels(V["R"]))),
        D=dict(rule="part D, D.17 combined string (CF-D)", string=D.get("D.17") or na, qualifiers=dict(D=_strip(D.get("qualifiers")), L4=l4q),
               labels=[x for x in (sg8["D_label"],) if x], deviations=_deviations("D", sess, fz, _labels(V["D"]))),
        K=dict(rule="part K, K.7 composite with suffixes (CF-K-attr, CF-K-link, CF-HERR)", string=(K.get("composite") or {}).get("string") or na,
               suffixes=(K.get("composite") or {}).get("suffixes"), labels=rg10, reported_next=K.get("reported_next_to_composite"),
               tier2="RK-1 (r1b_K_S6) and RK-2 (r1b_K_S7): runde1b_summary_t2.json (r1b_ana2)", deviations=_deviations("K", sess, fz, _labels(V["K"]))),
        L=dict(rule="part L, L.5 answer template (CF-L, CF-HERR)", string=LC.get("answer") or na, L_ALL=LC.get("L-ALL"),
               labels=[x for x in (sg8["L_label"],) if x] + [f"K {g}: {v}" for g, v in k_lens.items() if v],
               deviations=_deviations("L", sess, fz, _labels(V["L"]))))
    q9 = (K.get("families") or {}).get("Q9") or {}
    tables = dict(
        attractor=dict(K={k: v.get("string") for k, v in q9.items()},
                       L=dict(attractor_line=_txt(LV.get("attractor")), specificity=_g(LV, "L3b", "specificity"),
                              L3a=_g(LV, "L3a", "lines"), L3b_sustain=_g(LV, "L3b", "sustain"))),
        h_before_errors=dict(K8c=(K.get("strings") or {}).get("K8c"), L5c=_txt(LV.get("L5c"))),
        catfish=dict(D2_a=_g(D, "parts", "(b2)", "verdict", "M", "D2_a"), D5_a=_g(D, "parts", "(b2)", "verdict", "D5_a", "verdict"),
                     L4_catfish=l4.get("catfish")),
        note="side by side, no pooling and no new verdict (main §2.4); neither part's verdict is reported without the other's")
    nfor = [f for f in FOR if F[f]["confirmed"] > 0]
    against = dict(CF_R_minus=[r["member"] for r in F["CF-R-"]["members"] if r["label"] != "exploratory (Holm not met)"],
                   D17=D.get("D.17") in AGAINST["D"], K7=str((K.get("composite") or {}).get("string") or "").startswith(AGAINST["K"]),
                   L_ALL=LC.get("L-ALL") in AGAINST["L"],
                   note="outside CF-R- these are conjunctions of equivalence or bound rules (IUT), no further adjustment; each with its MDEs; "
                        "none refutes a hypothesis for larger models")
    t2 = dict(funded=fz.get("funded"), items_not_funded=[i for it in C.TIER2_ITEMS.values() for i in it if not _funded(fz, i)],
              sessions={s: (sess.get(s) or {}).get("state") for s, m in C.SESSIONS.items() if m["tier"] == 2},
              note="every tier-2 result is exploratory at the round level; r1b_ana2 reports them (runde1b_summary_t2.json)")
    probs = {s: p for s, p in sg6.items() if p}
    out = dict(answers=ans, families={f: F[f] for f in FAMS},
               for_families=dict(with_confirmed_member=nfor, n=len(nfor), of=len(FOR),
                                 null_bound="under a global null each 'for' family confirms falsely with p <= 0.05, so at least one with p <= 0.30"),
               R_counts=dict(F_plus=R.get("F_plus_counts"), F_minus=R.get("F_minus_counts")),
               confirmatory_against=against, cross_part_labels=dict(RG10=rg10 or ["J-lens valid at every block read (R5.1)" if jv else "jlens_validity.json missing"],
                                                                   SG8=sg8, K_lens_on_L5=k_lens, L4_on_D17=l4q),
               side_by_side=tables, part_vs_round_level=fams.get("part_vs_round_level"), tier2=t2,
               exploratory="every R verdict outside F+ / F-, D's secondary rules and qualifiers, K's other strings and suffixes, L's V-OWN, V-HOLO, "
                           "V-CHAIN, L1, L2a, L2c, L3c, L3d, L4, L5a, L5b, every tier-2 result and R0 (post hoc) are exploratory",
               amendments=(ctx.manifest.get("plan") or {}).get("amendments_sha256"), declared_not_run=declared,
               sessions={s: e["state"] for s, e in sess.items()},
               report_check=dict(sessions_checked=len(sg6), problems=probs, ok=not probs, rule="SG6: provenance in every output (R9.1-R9.4)"),
               codes=code_counts(ans))
    lab = (["MOCK (tiny random model; values are code-test values)"] if ctx.mock else []) + (["SG6 report check found problems"] if probs else [])
    ctx.write_json(C.FILES["summary"]["rel"], out, config=dict(alpha=ALPHA, families=FAMS, k_link_form="PI"), labels=lab)
    ctx.write_json(C.FILES["manifest"]["rel"], dict(sessions=sess, sg6=sg6, declared_not_run=declared), config=dict(), labels=lab)


# ====================================================================================================================================
# r1b_ana2: tier-2 results next to the four answers
# ====================================================================================================================================
def tier2(ctx):
    S = _ld(ctx, "r1b_ana", C.FILES["summary"]["rel"]) or {}
    src = dict(R=("r1b_R_ana2", "r_verdict2"), D=("r1b_D_R", "d_reserves"), RK1=("r1b_K_S6", "k_rk1"), RK2=("r1b_K_S7", "k_rk2"))
    roots, res = _roots(ctx), {}
    for k, (s, f) in src.items():
        d = _ld(ctx, s, C.FILES[f]["rel"])
        if d is not None:
            res[k] = dict(session=s, result=_strip(d), labels=["conditional run", "tier 2: exploratory at the round level"] + _labels(d))
        else:
            st, _, p = _state(roots, s)
            rec = load_json(p, {}) if (p and st == "not run") else {}
            why = "".join(c if (c.isalnum() or c in "_ .,()'-") else " " for c in str(rec.get("reason") or "not run"))
            res[k] = dict(session=s, state=st, code=coded(rec.get("code") or ("NA:not_run:" + (why if st == "not run" else "input missing"))))
    A = S.get("answers") or {}
    out = dict(next_to=dict(R=dict(answer=(A.get("R") or {}).get("per_block"), tier2=res["R"]),
                            D=dict(answer=(A.get("D") or {}).get("string"), tier2=res["D"]),
                            K=dict(answer=(A.get("K") or {}).get("string"), RK1=res["RK1"], RK2=res["RK2"]),
                            L=dict(answer=(A.get("L") or {}).get("string"), tier2="X1-X6 in l_verdicts.json (r1b_ana)")),
               items_in_tier1_sessions="D5-x, D3-name (d_verdict.json), K3h (r1b_K_S2b), X1-X6 (l_verdicts.json): reported by their parts",
               note="tier 2 changes no tier-1 verdict and no round-level family (main §2.4)")
    ctx.write_json(C.FILES["summary_t2"]["rel"], out, config=dict(sources={k: list(v) for k, v in src.items()}),
                   labels=(["MOCK (tiny random model; values are code-test values)"] if ctx.mock else []) + ["conditional run"])
