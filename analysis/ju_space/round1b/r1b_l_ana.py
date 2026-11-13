"""L-ANA (a step of r1b_ana, account A, CPU): L5 analyses (L5a ownership, L5b H-pattern, L5c step-error hazard), every L verdict by
r1b_l_rules on the session statistics, SG8 from r1b_K_ana's G-K2, L-CAL gates, the L.5 combinations (L-ALL, V-OWN, V-HOLO, V-CHAIN), the
round-level member p-values (CF-L, CF-HERR) and the answer template -> l_verdicts.json."""
import math, os, re
import numpy as np

import r1b_core as K
import r1b_contracts as C
import r1b_l_rules as RU
import r1b_l_stats as S
from r1b_l_base import cfg, clean, gating, inj, lensout, loadpt, rng, wj

NR = "not run (input missing)"


def _tok(ctx):
    from transformers import AutoTokenizer
    if ctx.mock:
        return AutoTokenizer.from_pretrained(os.path.join(ctx.root, "models", C.M17))
    p = K.model_pins(C.M17, ctx.freeze)
    return AutoTokenizer.from_pretrained(p.get("repo") or "Qwen/Qwen3-1.7B", revision=p.get("revision"))


def _j(ctx, sess, rel):
    p = ctx.input(sess, rel, required=False)
    return K.load_json(p) if p else None


def _pt(ctx, sess, rel):
    p = ctx.input(sess, rel, required=False)
    return loadpt(p) if p else None


def nums(text):
    return [int(x) for x in re.findall(r"-?\d+", text or "")]


# ---------------------------------------------------------------------------------------------------------------------------------- L5a
def digit_runs(pieces):
    """Completed maximal digit runs per chain position q: [(q, value)] (leading minus signs ignored)."""
    out, cur, q0 = [], "", None
    for q, p in enumerate(pieces):
        p = "".join(map(str, p)) if isinstance(p, (list, tuple)) else (p or "")
        if p.strip().isdigit() and p.strip() == p.strip("-"):
            cur += p.strip()
        else:
            if cur:
                out.append((q - 1, int(cur))); cur = ""
            d = re.findall(r"\d+", p)
            for x in d:
                out.append((q, int(x)))
    if cur:
        out.append((len(pieces) - 1, int(cur)))
    return out


def l5a(ctx, tok, gate, recs, gen, cf):
    """Classes TRUE / MODEL / FOREIGN per completed digit run; break-off times; chance-match null against M_j of a foreign item (same
    stratum and number of operations, derangement seed 'r1bL5-FM'); G_F, G_M^c, Omega with item bootstrap."""
    M = {}
    for i, it in gate.items():
        s = set()
        for a in it["answers"]:
            s |= set(nums(a))
        g = (gen or {}).get(i) or {}
        for a in g.get("gate", []):
            s |= set(nums(a))
        for c in g.get("cot", []):
            s |= set(nums(c["text"]))
        for c in g.get("sub", []):
            s |= set(nums(c["text"]))
        M[i] = s
    ids = sorted({r["item"] for r in recs if r["item"] in gate})
    strata = {}
    for i in ids:
        strata.setdefault((gate[i]["stratum"], gate[i].get("ops")), []).append(i)
    foreign = {}
    for k, lst in strata.items():
        p = S.derange(len(lst), rng("r1bL5-FM", *map(str, k))) if len(lst) > 1 else np.arange(len(lst))
        for a, b in zip(lst, p):
            foreign[a] = lst[b]
    rows = []
    for r in recs:
        i = r["item"]
        if i not in gate:
            continue
        pieces = r.get("tokens_text") or [tok.decode(t if isinstance(t, list) else [t]) for t in r.get("tokens", [])]
        V = set(gate[i]["V"]); Mi = M.get(i, set()) - V; Mf = M.get(foreign.get(i), set()) - V
        BF = BM = BMf = None; q0F = False
        for q, v in digit_runs(pieces):
            cls = "TRUE" if v in V else ("MODEL" if v in Mi else "FOREIGN")
            if q == 0:
                q0F |= cls == "FOREIGN"; continue
            if cls == "FOREIGN" and BF is None:
                BF = q
            if cls == "MODEL" and BM is None:
                BM = q
            if v in Mf and v not in V and BMf is None:
                BMf = q
        rows.append(dict(item=i, stratum=gate[i]["stratum"], BF=BF, BM=BM, BMf=BMf, q0F=q0F, block=r.get("block"), variant=r.get("variant"),
                         seed=r.get("seed_rank"), k_conf=r.get("k_star_conf"), share=r.get("share_H"), cls_kconf=None))
    if not rows:
        return None, rows
    W = S.boot_counts(len(ids), cf["nboot"], seed=501)
    iid = {u: k for k, u in enumerate(ids)}

    def curves(sel):
        rr = [r for r in rows if sel(r)]
        it = np.array([iid[r["item"]] for r in rr])
        out = {}
        for key in ("BF", "BM", "BMf"):
            for q in (3, 6, 12):
                v = np.array([1.0 if (r[key] is not None and r[key] <= q) else 0.0 for r in rr])
                s_ = np.bincount(it, weights=v, minlength=len(ids)); n_ = np.bincount(it, minlength=len(ids)).astype(float)
                out[(key, q)] = (float(v.mean()) if len(v) else np.nan, (W @ s_) / np.maximum(W @ n_, 1))
        return out

    def stats(sel):
        c = curves(sel)
        GF = (c[("BF", 12)][0] - c[("BF", 3)][0], c[("BF", 12)][1] - c[("BF", 3)][1])
        GMc = ((c[("BM", 12)][0] - c[("BMf", 12)][0]) - (c[("BM", 3)][0] - c[("BMf", 3)][0]),
               (c[("BM", 12)][1] - c[("BMf", 12)][1]) - (c[("BM", 3)][1] - c[("BMf", 3)][1]))
        den = (GMc[0] + GF[0], GMc[1] + GF[1])
        om = (GMc[0] / den[0] if den[0] > 0 else np.nan, np.where(den[1] > 0, GMc[1] / np.where(den[1] > 0, den[1], 1), np.nan))
        return dict(GF=S.summ(*GF, margin=0.04), GMc=S.summ(*GMc), Omega=S.summ(*om), G=S.summ(GF[0] + GMc[0], GF[1] + GMc[1], margin=0.04),
                    C=dict((f"{k}_{q}", v[0]) for (k, q), v in c.items()))
    known, prone, alls = stats(lambda r: r["stratum"] == "known"), stats(lambda r: r["stratum"] == "prone"), stats(lambda r: True)
    st = dict(GF_known=known["GF"], Omega_prone=prone["Omega"], Omega_all=alls["Omega"], G_all=alls["G"], GMc_prone=prone["GMc"],
              known=known, prone=prone, all=alls, q0_foreign=float(np.mean([r["q0F"] for r in rows])), n_chains=len(rows), n_items=len(ids),
              kconf_model_share=float(np.mean([r["BM"] is not None and r["k_conf"] is not None and r["BM"] <= r["k_conf"] for r in rows])))
    return st, rows


def l5b(ctx, rows, gen, chan_names, cf):
    """rho_b (between items: a_P(t*) against the item mean of E_c = 1[B_F <= 6], per block, averaged) and rho_w (within items, across
    seed ranks: chain mean of share_H over s = 1..6 against E_c); item bootstrap; rank of rho_b against the 99 control channels."""
    if not rows or not gen:
        return None
    blocks = sorted({r["block"] for r in rows})
    items = sorted({r["item"] for r in rows if (gen.get(r["item"]) or {}).get("tstar") is not None})
    if len(items) < 4:
        return None
    iid = {u: k for k, u in enumerate(items)}
    W = S.boot_counts(len(items), cf["nboot"], seed=502)
    Ec = {}
    for r in rows:
        Ec.setdefault((r["block"], r["item"]), []).append((r["seed"], 1.0 if (r["BF"] is not None and r["BF"] <= 6) else 0.0,
                                                           float(np.mean(r["share"][1:7])) if r.get("share") else np.nan))
    tstar = np.stack([np.asarray(gen[i]["tstar"], float) for i in items])

    def rho_b(col, wts=None):
        vals = []
        for b in blocks:
            x, y, w = [], [], []
            for i in items:
                e = Ec.get((b, i))
                if e:
                    x.append(tstar[iid[i], col]); y.append(np.mean([v[1] for v in e])); w.append(1 if wts is None else wts[iid[i]])
            xx = np.repeat(x, np.asarray(w, int)); yy = np.repeat(y, np.asarray(w, int))
            vals.append(S.spearman(xx, yy))
        return float(np.nanmean(vals)) if vals else np.nan

    def rho_w(wts=None, perm=None):
        vals = []
        for b in blocks:
            for i in items:
                e = Ec.get((b, i))
                if not e or len(e) < 3 or len({v[1] for v in e}) < 2:
                    continue
                sh = np.array([v[2] for v in e]); ec = np.array([v[1] for v in e])
                if perm is not None:
                    sh = perm.permutation(sh)
                r_ = S.spearman(sh, ec)
                vals += [r_] * (1 if wts is None else int(wts[iid[i]]))
        return float(np.nanmean(vals)) if vals else np.nan
    P = chan_names.index("P")
    rb, rw_ = rho_b(P), rho_w()
    st = dict(rho_b=S.summ(rb, [rho_b(P, W[k]) for k in range(len(W))], margin=0.12),
              rho_w=S.summ(rw_, [rho_w(W[k]) for k in range(len(W))], margin=0.05),
              rho_w_null=K.dist([rho_w(perm=np.random.default_rng(k)) for k in range(200 if not ctx.mock else 10)]),
              n_items=len(items), n_const_dropped=sum(1 for v in Ec.values() if len({x[1] for x in v}) < 2))
    ci_ = [k for k, n in enumerate(chan_names) if n.startswith("C")]
    st["rank_p"] = K.p_rank(rb, [rho_b(k) for k in ci_])
    return st


# ---------------------------------------------------------------------------------------------------------------------------------- L5c
LINE_RE = re.compile(r"(-?\d+)\s*([+\-*×x/])\s*(-?\d+)\s*=\s*(-?\d+)")
OPF = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b, "×": lambda a, b: a * b, "x": lambda a, b: a * b,
       "/": lambda a, b: a // b if b else None}


def parse_cot(text, offsets, it):
    """Frozen parser: lines 'a o b = r'. Step correct (part-file amendment) if r is in V_i or I+, or r = a o b on valid available values
    (operands of the prompt and earlier results). Returns steps [(line_start_char, eq_char, correct, kind)]."""
    avail = set(it["V"][:len(it["V"]) - len(it.get("steps") or [])] or it["V"]) | set()
    V, Ip = set(it["V"]), set(it.get("I_plus") or [])
    steps, pos = [], 0
    for line in text.split("\n"):
        m = LINE_RE.search(line)
        if m:
            a, o, b, r = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
            ok_op = a in avail and b in avail
            val = OPF[o](a, b)
            correct = (r in V or r in Ip) or (ok_op and val == r)
            kind = None if correct else ("result" if (val is not None and val != r and ok_op) else "selection")
            steps.append((pos, pos + line.index("=", m.start()), correct, kind))
            if correct:
                avail.add(r)
        pos += len(line) + 1
    return steps


def l5c(ctx, gate, gen, chan_names, cf):
    """Conditional hazard over steps at risk with item fixed effects: logit P(error at k | none before) = item + step index +
    beta_H z(a(t_k^-)) + beta_e z(e(t_k^-)) + beta_pos log t_k^-, t_k^- = the last token before step k's line (part-file amendment);
    item-cluster bootstrap; every channel; 'result errors at =' secondary."""
    obs = []
    parsed = total = 0
    for i, g in (gen or {}).items():
        if i not in gate:
            continue
        for c in g.get("cot", []):
            total += 1
            st = parse_cot(c["text"], c["offsets"], gate[i])
            if not st:
                continue
            parsed += 1
            offs = np.asarray(c["offsets"])
            for k, (ls, eq, ok, kind) in enumerate(st):
                tpos = int(np.searchsorted(offs, ls, side="left")) - 1
                teq = int(np.searchsorted(offs, eq, side="right")) - 1
                if tpos < 0 or tpos >= len(c["a"]):
                    continue
                obs.append(dict(item=i, k=k, y=0.0 if ok else 1.0, a=np.asarray(c["a"][tpos], float), e=float(c["e"][tpos]), t=tpos + 1,
                                a_eq=np.asarray(c["a"][min(teq, len(c["a"]) - 1)], float), kind=kind))
                if not ok:
                    break
    res = dict(parse_share=parsed / max(total, 1), parse_ok=parsed / max(total, 1) >= 0.7, n_obs=len(obs))
    if not obs:
        return res
    items = sorted({o["item"] for o in obs}); iid = {u: k for k, u in enumerate(items)}
    ev_items = {o["item"] for o in obs if o["y"] == 1} & {o["item"] for o in obs if o["y"] == 0}
    res["n_events"] = sum(1 for o in obs if o["y"] == 1 and o["item"] in ev_items)
    y = np.array([o["y"] for o in obs]); fe = np.array([iid[o["item"]] for o in obs])
    kmax = max(o["k"] for o in obs) + 1
    Kx = np.eye(kmax)[[o["k"] for o in obs]][:, 1:]
    zs = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    A = np.stack([o["a"] for o in obs]); ev = zs(np.array([o["e"] for o in obs])); lt = np.log(np.array([o["t"] for o in obs], float))

    def fit(col, with_e=True, wts=None, key="a"):
        a = zs(np.stack([o[key] for o in obs])[:, col]) if key != "a" else zs(A[:, col])
        X = np.column_stack([a] + ([ev] if with_e else []) + [lt, Kx])
        beta, _, _ = S.logit_fe(y, fe, X, len(items), weights=None if wts is None else wts[fe])
        return float(beta[0])
    P = chan_names.index("P")
    W = S.boot_counts(len(items), min(cf["nboot"], 400 if not ctx.mock else 30), seed=503)
    res["beta_H"] = S.summ(fit(P), [fit(P, wts=W[b]) for b in range(len(W))], margin=0.15)
    res["beta_H_noe"] = S.summ(fit(P, with_e=False), [fit(P, with_e=False, wts=W[b]) for b in range(len(W))])
    res["beta_H_eq_result"] = fit(P, key="a_eq")
    ci_ = [k for k, n in enumerate(chan_names) if n.startswith("C")]
    bc = [fit(k) for k in ci_]
    res["rank_p"] = K.p_rank(res["beta_H"]["pt"], bc) if isinstance(res["beta_H"]["pt"], float) else 1.0
    if "UNC" in chan_names:
        u = chan_names.index("UNC")
        res["minus_unc"] = S.summ(fit(P) - fit(u), [fit(P, wts=W[b]) - fit(u, wts=W[b]) for b in range(len(W))])
        res["not_beyond_unc"] = not RU.LB(res["minus_unc"])
    bix = [k for k, n in enumerate(chan_names) if n.startswith("B")]
    res["sel_robust"] = (sum(fit(k) > 0 and K.p_rank(fit(k), bc) <= 0.05 for k in bix) >= 16 * len(bix) / 20) if len(bix) >= 10 else None
    res["p_cf_herr"] = max(res["beta_H"]["p_lt0"] if False else res["beta_H"]["p_gt0"], res["rank_p"])
    res["error_kinds"] = dict(selection=sum(o["kind"] == "selection" for o in obs), result=sum(o["kind"] == "result" for o in obs))
    return res


# ---------------------------------------------------------------------------------------------------------------------------------- main
def l_cal_record(ctx):
    p = os.path.join(K.REPO, C.PLAN_DIR, "l_cal.json")
    d = K.load_json(p) if os.path.exists(p) else None
    if d is None:
        if ctx.mock:
            return {k: True for k in "abcdefghijklmn"}, ["MOCK: L-CAL record absent (code test)"]
        return {k: False for k in "abcdefghijklmn"}, ["L-CAL record missing (results/ju_space/plans/runde1b/l_cal.json)"]
    return {k: bool(v.get("pass")) for k, v in d.get("items", {}).items()}, []


def l_ana(ctx):
    cf = cfg(ctx)
    labels = []
    cal, cl = l_cal_record(ctx); labels += cl
    calg = dict(L1=cal.get("a", False), L2a=cal.get("f", False), L2b=cal.get("g", False), L2c=cal.get("j", False),
                L3a=all(cal.get(k, False) for k in "bcd"), L3c=cal.get("e", False), L3d=all(cal.get(k, False) for k in "hi"),
                L4=all(cal.get(k, False) for k in "kl"), L5a=cal.get("m", False), L5c=cal.get("n", False))
    kv = _j(ctx, "r1b_K_ana", "k_verdict.json") or {}
    gk2 = (kv.get("gates") or {}).get("G-K2")
    pop = _j(ctx, "r1b_L_A", "l/l_pop.json") or {}
    if isinstance(gk2, dict):
        gk2 = gk2.get("string") or gk2.get("verdict")
    sg8 = bool(gk2 and str(gk2).startswith("H-neurons replicated")) and pop.get("H_op_name") in ("H_tq", "H+") if gk2 else bool(pop.get("sg8_ok"))
    if gk2 is None:
        labels.append("SG8 from r1b_K_S1's preview (r1b_K_ana's G-K2 not present)")
    elif bool(pop.get("sg8_ok")) != sg8:
        labels.append("SG8 preview and r1b_K_ana disagree (round-2 OD-G8: reported)")
    V = {}

    def safe(name, fn):
        try:
            V[name] = fn()
        except (FileNotFoundError, K.NotRun, KeyError, TypeError) as e:
            V[name] = dict(verdict=NR, error=str(e)[:200])
    l1 = _j(ctx, "r1b_L_A", "l/l1.json")
    safe("L1", lambda: RU.rule_L1(l1["tree_input"], cal_ok=calg["L1"] and l1["cal_ok"], m_cal=l1.get("m_cal"), sg8=sg8))
    safe("L1-erasure", lambda: RU.rule_L1_erasure(l1["erasure"], dmg_ok=l1["erasure"]["g_dmg"]))
    l2a = _j(ctx, "r1b_L_B", "l/l2a.json")
    safe("L2a", lambda: RU.rule_L2a(l2a) if calg["L2a"] else dict(growth=RU.NT_CAL, own=RU.NT_CAL))
    l2b = _j(ctx, "r1b_L_B", "l/l2b.json")
    safe("L2b", lambda: RU.rule_L2b(l2b, sg8=sg8, cal_ok=calg["L2b"]))
    l2c = _j(ctx, "r1b_L_B", "l/l2c.json")
    safe("L2c", lambda: RU.rule_L2c(l2c, cal_ok=calg["L2c"], sg8=sg8))
    l3a = _j(ctx, "r1b_L_B", "l/l3a.json")
    safe("L3a", lambda: RU.rule_L3a(l3a, sg8=sg8, cal_ok=calg["L3a"]))
    l3b = _j(ctx, "r1b_L_A", "l/l3b.json")
    safe("L3b", lambda: RU.rule_L3b(l3b, sg8=sg8, dmg_ok=tuple(l3b["dmg_ok"])))
    l3c = _j(ctx, "r1b_L_B", "l/l3c.json")
    safe("L3c", lambda: RU.rule_L3c(l3c, sg8=sg8, cal_ok=calg["L3c"] and l3c["g_cal_ising"]["pass"], gsel=bool(pop.get("G_sel"))))
    l3d = _j(ctx, "r1b_L_B", "l/l3d.json")
    safe("L3d", lambda: RU.rule_L3d(l3d, sg8=sg8, cal_ok=calg["L3d"]))
    x1 = _j(ctx, "r1b_L_B", "l/x1.json")
    V["X1"] = RU.rule_X1(x1) if x1 else dict(verdict=K.NA_TIER2 if not ctx.funded("X1") else NR)
    safe("attractor", lambda: RU.attractor_line(V["L3a"], V["L3b"], x1_capture=(V["X1"].get("capture") if x1 else None)))
    l4 = _j(ctx, "r1b_L_A", "l/l4.json")
    safe("L4", lambda: RU.rule_L4(l4) if calg["L4"] else {k: RU.NT_CAL for k in ("pair", "collaboration", "entity", "h_line", "catfish", "macro")})
    x2 = _j(ctx, "r1b_L_A", "l/x2.json")
    if "L4" in V and isinstance(V["L4"], dict) and x2:
        V["L4"]["h_line_labels"] = [RU.x2_label(x2)]
    V["X2"] = x2 or dict(verdict=K.NA_TIER2 if not ctx.funded("X2") else NR)
    V["X6"] = _j(ctx, "r1b_L_A", "l/x6.json") or dict(verdict=K.NA_TIER2 if not ctx.funded("X6") else NR)
    x3 = _j(ctx, "r1b_L_B", "l/x3.json")
    V["X3"] = RU.rule_X3(x3) if x3 else dict(verdict=K.NA_TIER2 if not ctx.funded("X3") else NR)
    x4 = _j(ctx, "r1b_L_B", "l/x4.json")
    V["X4"] = RU.rule_X4({k: v for k, v in x4.items() if k != "provenance"}) if x4 else dict(verdict=K.NA_TIER2 if not ctx.funded("X4") else NR)
    x5 = _j(ctx, "r1b_L_B", "l/x5.json")
    V["X5"] = dict(verdict="descriptive", final=x5.get("final")) if x5 else dict(verdict=K.NA_TIER2 if not ctx.funded("X5") else NR)
    # L5 analyses
    l5 = {}
    try:
        tok = _tok(ctx)
        gate, gl = gating(ctx)
        recs, rl = lensout(ctx, gate)
        g5 = _pt(ctx, "r1b_L_B", "l/l5_gen.pt")
        gen, chn = (g5 or {}).get("gen"), (g5 or {}).get("channels") or []
        gcl = sum(1 for i in {r["item"] for r in recs} if gate.get(i, {}).get("stratum") == "known") >= (150 if not ctx.mock else 2) and \
            sum(1 for i in {r["item"] for r in recs} if gate.get(i, {}).get("stratum") == "prone") >= (150 if not ctx.mock else 2)
        st5a, rows = l5a(ctx, tok, gate, recs, gen, cf)
        V["L5a"] = dict(verdict=RU.rule_L5a(st5a, gcl=gcl) if st5a else "not testable (no exact-label lens corpus from K)")
        st5b = l5b(ctx, rows, gen, chn, cf) if gen else None
        V["L5b"] = RU.rule_L5b(st5b, gcl=gcl, sg8=sg8) if st5b else dict(verdict="not testable", labels=[])
        st5c = l5c(ctx, gate, gen, chn, cf) if gen else None
        V["L5c"] = RU.rule_L5c(st5c, sg8=sg8, cal_ok=calg["L5c"]) if st5c else dict(verdict=NR, labels=[])
        l5 = dict(L5a=st5a, L5b=st5b, L5c=st5c, labels=gl + rl, G_CL=gcl)
    except (FileNotFoundError, K.NotRun) as e:
        for k in ("L5a", "L5b", "L5c"):
            V.setdefault(k, dict(verdict=NR, error=str(e)[:200]))
    # combinations
    vv = lambda k, f="verdict": (V.get(k) or {}).get(f, NR) if isinstance(V.get(k), dict) else NR
    comb = {}
    try:
        comb["L-ALL"] = RU.l_all(V["attractor"], V["L3b"].get("specificity"), V["L2b"])
        comb["V-OWN"] = RU.v_own(vv("L5a"), vv("L5b"))
        comb["V-OWN"]["L5c"] = vv("L5c")
        comb["V-HOLO"] = dict(code=dict(L1=vv("L1"), labels=(V.get("L1") or {}).get("labels")), RT=V.get("L2a"),
                              reconstruction=dict(Lambda=(l1 or {}).get("Lambda"), L2c=vv("L2c")),
                              associative_memory=dict(erasure=V.get("L1-erasure"), attractor=vv("attractor"), X4=V.get("X4")))
        comb["V-CHAIN"] = RU.v_chain(V["L3c"], V["L3d"], V["L4"], V["X3"] if x3 else None)
        comb["H-coupling"] = V["L4"].get("h_line")
        comb["CF-L"] = RU.cf_l_p(l3a or {}, l3b or {}, l2b or {}, V["L3a"], V["L3b"], V["L2b"], V["attractor"])
        bh = (l5.get("L5c") or {}).get("beta_H")
        comb["CF-HERR_member_L5c"] = dict(p=(l5["L5c"]["p_cf_herr"] if vv("L5c").startswith("H predicts") else 1.0) if l5.get("L5c") else 1.0)
        comb["answer"] = (f"Lens-free, the break-off pattern belongs to [{comb['V-OWN']['verdict']}]; the H-neuron attractor is [{comb['L-ALL']}]; "
                          f"the model is holographic only in the [{', '.join(k for k, v in comb['V-HOLO'].items() if v)}] sense(s) (per-sense verdicts "
                          f"reported); the chain [{comb['V-CHAIN']['verdict']}]; H gain [{comb['H-coupling']}] as agent coupling.")
    except (KeyError, TypeError) as e:
        comb["error"] = str(e)[:300]
    out = dict(verdicts=V, combined=comb, sg8=dict(ok=sg8, G_K2=gk2, preview=pop.get("sg8_preview")), l_cal=cal, l_cal_gates=calg, L5_stats=l5,
               codes=K.code_counts(clean(V)))
    wj(ctx, "l_verdicts.json", out, config=dict(rules="r1b_l_rules (part L file with its amendments)"), labels=labels)
