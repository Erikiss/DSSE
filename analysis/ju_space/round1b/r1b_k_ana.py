"""r1b_K_ana (account A): every K statistic and verdict string on the sessions' stored data (OD-G8: in-session decisions are
recomputed and disagreements reported), the Holm families Q1-Q18, the round-level member p-values (CF-K-attr, CF-K-link, CF-HERR) and
the K.7 composite (with the K.12 fixes: row 1 only on a 'not replicated' K3a; B- needs >= 1 defined carry cell). Output
k_verdict.json (contract) and k/ana1-3.json."""
import math
import numpy as np
import r1b_core as RC
import r1b_k_base as KB
import r1b_k_rules as KR
from r1b_core import coded, val, p_rank, p_pi, holm, bh
from r1b_k_base import N, write, load_pt, logit_fit, boot_stat, ci, gt, lt, inside, fin, p_iut, holm_family, auroc

B_ = RC.N_BOOT


def _in(ctx, s, rel):
    try:
        return KB.up(ctx, s, rel, required=False)
    except RC.NotRun:
        return None


def _pt(ctx, s, rel):
    p = _in(ctx, s, rel); return load_pt(p) if p else None


def _js(ctx, s, rel):
    p = _in(ctx, s, rel); return RC.load_json(p) if p else None


def _parts(ctx, s, meas, n):
    out = []
    for i in range(n):
        d = _pt(ctx, s, f"k/{meas}/part{i}.pt")
        if d:
            out += d["rows"]
    return out


def nr(reason):
    return dict(string=f"not run ({reason})", code=coded(f"NA:not_run:{reason}"))


def pboot(draws, thr=0.0, alt="greater"):
    return RC.p_one_sided(draws, thr, alt)


def qshare(claims, lo=1, hi=99, seg=None):
    c = [x for s, _, x in claims if lo <= s <= hi and x not in ("PROMPT", "AMBIG")]
    return (sum(x == "CONFAB" for x in c) / len(c)) if c else np.nan


def anyhit(chains, cat="TRUE-INT"):
    return float(any(any(c == cat for _, _, c in r["claims"]) for r in chains))


def by(rows, *keys):
    d = {}
    for r in rows:
        d.setdefault(tuple(r[k] for k in keys), []).append(r)
    return d


def item_diff(a, b, B=B_):
    """Paired item-level difference mean(a_i - b_i) over items present in both (dicts item -> value); bootstrap over items."""
    ks = [k for k in a if k in b and fin(a[k]) and fin(b[k])]
    if len(ks) < 3:
        return dict(est=coded("NaN:too_few_items"), ci=dict(lo=coded("NaN:too_few_items"), hi=coded("NaN:too_few_items")), n=len(ks),
                    draws=np.array([]))
    d = np.array([a[k] - b[k] for k in ks]); dr = boot_stat(len(d), lambda ix: d[ix].mean(), B)
    return dict(est=float(d.mean()), ci=ci(dr), n=len(ks), draws=dr)


def logit_slope(rows_by_item, B=B_):
    """Logit-linear slope of the per-claim confabulation share over steps 3..16 (claims as binomial counts per item and step; item
    bootstrap)."""
    items = list(rows_by_item); S = np.arange(3, 17)
    M = np.array([[rows_by_item[i].get(s, (0, 0)) for s in S] for i in items], float)       # [items, steps, (confab, n)]
    if M.size == 0:
        return np.nan, np.array([])

    def fit(w):
        c = (M[..., 0] * w[:, None]).sum(0); n = (M[..., 1] * w[:, None]).sum(0); ok = n > 0
        if ok.sum() < 3:
            return np.nan
        X = np.repeat(S[ok], 2); y = np.tile([1.0, 0.0], ok.sum()); ww = np.c_[c[ok], n[ok] - c[ok]].ravel()
        f = logit_fit(X[:, None], y, w=ww, l2=1e-8, std=False); return float(f["beta"][0]) if f["ok"] else np.nan
    est = fit(np.ones(len(items))); g = np.random.default_rng(KB.SEED_BOOT)
    dr = np.array([fit(np.bincount(g.integers(0, len(items), len(items)), minlength=len(items)).astype(float)) for _ in range(B)])
    return est, dr


def claim_counts(chains):
    d = {}
    for r in chains:
        cen = r.get("censor")
        for s, _, c in r["claims"]:
            if c in ("PROMPT", "AMBIG") or (cen is not None and s >= cen):
                continue
            a = d.setdefault(r["item"], {}).setdefault(s, [0, 0]); a[0] += c == "CONFAB"; a[1] += 1
    return {i: {s: tuple(v) for s, v in x.items()} for i, x in d.items()}


# =================================================================================================================================
# part 1: K3a-c, K3d, K3g, K4j, K4 (G-K4, G-K6, G-K7), K4p, K5a, K6a, K6d, K10c
# =================================================================================================================================
def k5a_stats(D, B=B_, J=16):
    """K5a: C_k, recall, IAD (variant), G-K5 and the native length trend (Q6); self-consistency KL slope (descriptive)."""
    rows = D["rows"]; out = dict(labels=[])
    C = {}
    for (lens, var), rs in by(rows, "lens", "variant").items():
        Ck = np.array([[float(any(c == "CONFAB" and s <= k for s, _, c in r["claims"])) for k in range(1, J + 1)] for r in rs])
        C[(lens, var)] = Ck.mean(0) if len(Ck) else np.full(J, np.nan)
    iad = {v: float(np.nanmean(np.abs(C.get((v, "V1"), np.nan) - C.get(("CL35", "V1"), np.nan)))) for v in ("E", "L")}
    variant = min(iad, key=lambda v: iad[v] if fin(iad[v]) else 9)
    rec = {(l, v): {i: anyhit(c) for (i,), c in by([r for r in rows if r["lens"] == l and r["variant"] == v], "item").items()} for l, v in C}
    cJ = {(l, v): {i: float(np.mean([any(c == "CONFAB" for _, _, c in r["claims"]) for r in c])) for (i,), c in
                   by([r for r in rows if r["lens"] == l and r["variant"] == v], "item").items()} for l, v in C}
    dC = item_diff(cJ.get((variant, "V1"), {}), cJ.get(("CL35", "V1"), {}), B); dR = item_diff(rec.get((variant, "V1"), {}), rec.get(("CL35", "V1"), {}), B)
    pc = item_diff(rec.get(("CL35", "V1"), {}), rec.get(("CL35", "V2"), {}), B)
    if not (fin(pc["ci"]["lo"]) and pc["ci"]["lo"] > 0):
        g5 = "not decidable"; out["labels"].append("native lens fails its own positive control")
    elif inside(dC["ci"], 0.10) and inside(dR["ci"], 0.10):
        g5 = "self-rollout calibrated"
    elif any((gt(d["ci"]) or lt(d["ci"])) and fin(d["est"]) and abs(d["est"]) > 0.10 for d in (dC, dR)):
        g5 = "self-rollout differs from the native chain lens"
    else:
        g5 = "calibration undecided"
    est, dr = logit_slope(claim_counts([r for r in rows if r["lens"] == "CL35" and r["variant"] == "V1"]), B)
    sl = ci(dr) if len(dr) else dict(lo=coded("NaN:no_claims"), hi=coded("NaN:no_claims"))
    q6 = ("native: per-step confabulation rises with chain length" if gt(sl) else ("native: flat per-step confabulation" if inside(sl, 0.03) else "undecided"))
    kl = np.array([r["kl"] for r in D.get("selfkl", [])]) if D.get("selfkl") else None
    out.update(variant=variant, IAD=iad, C_k={f"{l}|{v}": [val(x) for x in c] for (l, v), c in C.items()}, G_K5=g5, dC16=_s(dC), drecall=_s(dR),
               native_pc=_s(pc), Q6=dict(string=q6, slope=val(est), ci=sl, p=pboot(dr) if len(dr) else 1.0),
               selfkl=dict(mean_by_step=[val(x) for x in kl.mean(0)] if kl is not None and kl.size else None,
                           note="Teoh 2026 self-consistency KL(q || p) per step, CL35 V1 (descriptive)"),
               trunk="length trend without trunk dynamics" if (q6.startswith("native: per-step") and g5 == "self-rollout calibrated") else None)
    return out


def _s(d):
    return {k: v for k, v in d.items() if k != "draws"}


def ana_part1(ctx):
    torch = KB.T_()
    res = {}
    K3 = _pt(ctx, "r1b_K_S1", "k/k3/k3a_data.pt")
    if K3:
        a = KR.k3a_rule(K3); res["K3a"] = a; res["K3b"] = KR.k3b_rule(K3); res["K3c"] = KR.k3c_rule(K3)
    else:
        res["K3a"] = dict(in_domain=nr("input missing")); res["K3b"] = nr("input missing"); res["K3c"] = nr("input missing")
    hs = _js(ctx, "r1b_K_S1", "hn/hn_set.json") or {}; rs = _js(ctx, "r1b_K_S1", "hn/rs_sets.json") or {}
    # K3d (Q2)
    rows = _parts(ctx, "r1b_K_S1", "k3d", 8); res["K3d"] = k3d_stats(rows, rs) if rows else nr("input missing")
    # K3g (G-K14)
    g2 = _pt(ctx, "r1b_K_S1", "k/k3/k3g2.pt"); meta = _js(ctx, "r1b_K_S1", "hn/hn_meta.json") or {}
    S2 = _pt(ctx, "r1b_K_S2a", "k/k4/summary.pt"); rng = (meta.get("aH_range_Sfit") or {})
    if S2 and fin(rng.get("p0_5")):
        v = np.concatenate([r["aX"]["H"].float().numpy() for r in S2["rows"] if r["variant"] == "V1" and "H" in r["aX"]] or [np.array([])])
        share_in = float(((v >= rng["p0_5"]) & (v <= rng["p99_5"])).mean()) if v.size else np.nan
    else:
        share_in = np.nan
    if g2 is not None and K3 and g2.get("ans") is not None:
        y = KR._ybin(g2["y"]); keep = np.where(y >= 0)[0]
        det = KR.detection(K3, "H_op", (g2["ans"].float().numpy()[keep], y[keep]), K3["fit"]["X"], K3["fit"]["y"],
                           [k for k in K3["sets"] if k.startswith("RS")], B_)
    else:
        det = nr("input missing")
    q95 = float(np.percentile(det.get("auroc_RS", [np.nan]) and [x for x in det.get("auroc_RS", []) if fin(x)] or [np.nan], 95)) if det.get("auroc_RS") else np.nan
    ok14 = (fin(share_in) and share_in >= 0.9 and fin(det.get("auroc_H")) and fin(q95) and det["auroc_H"] > q95 and fin(det.get("auroc_lb"))
            and det["auroc_lb"] > 0.55)
    res["K3g"] = dict(share_in_range=val(share_in), frame=_s(det) if isinstance(det, dict) else det,
                      G_K14="H scale valid in the lens frame" if ok14 else "H scale not validated in the lens frame")
    res["K3e"] = _js(ctx, "r1b_K_S1", "k/k3/k3e.json") or nr("input missing")
    res["K4j"] = (_js(ctx, "r1b_K_S1", "k4j.json") or {}).get("G-K-J", nr("input missing"))
    # K4: G-K4 (V1 - V2, V1 - V6 any-hit; t0, correct items, pooled blocks), G-K6, G-K7
    items = _pt(ctx, "r1b_K_S2a", "k/k4/items.pt")
    if S2 and items:
        k4 = k4_stats(S2, items); res["K4"] = k4
        bs = _js(ctx, "r1b_K_S2a", "k/k4/b_star.json") or {}
        res["G_K6"] = (bs.get("G_K6") or {}).get("string", "verifier unreliable"); res["G_K7"] = bs.get("G_K7")
        res["K6a"] = k6a_stats(S2); res["K6d"] = k6d_stats(S2, items)
    else:
        res["K4"] = dict(G_K4=nr("input missing")); res["K6a"] = nr("input missing"); res["K6d"] = nr("input missing")
    res["K4p"] = k4p_stats(_pt(ctx, "r1b_K_S2a", "k/k4p.pt"))
    k5 = _pt(ctx, "r1b_K_S3", "k/k5a/chains.pt"); cal = _js(ctx, "r1b_K_S3", "k5a_calib.json") or {}
    res["K5a"] = k5a_stats(k5) if k5 else dict(G_K5="not decidable", variant=cal.get("variant", "E"), labels=["not calibrated (K5a not run)"])
    if k5:
        res["K5a"]["in_session_variant"] = cal.get("variant"); res["K5a"]["disagreement"] = cal.get("variant") != res["K5a"]["variant"]
        g17 = cal.get("G_K17") or {}
        if g17 and not g17.get("pass_", True):
            res["K5a"]["G_K5"] = "not decidable"
    res["K10c"] = k10c_stats(_pt(ctx, "r1b_K_S2a", "k/k10c.pt"))
    write(ctx, "k/ana1.json", res, labels=["r1b_K_ana part 1"])


def _auroc_lb(det):
    return det.get("auroc_lb", det.get("auroc_H", 0)) if isinstance(det, dict) else 0


def k3d_stats(rows, rs):
    cond = {r["cond"]: r for r in rows}; base = cond.get("H_a1.0") or next((r for r in rows if r["cond"] == "H" and r["alpha"] == 1.0), None)
    H3 = next((r for r in rows if r["cond"] == "H" and r["alpha"] == 3.0), None); H1 = next((r for r in rows if r["cond"] == "H" and r["alpha"] == 1.0), None)
    if not (H3 and H1):
        return nr("conditions missing")
    fams = {"HALL": ("ACH", "TQA", "NE"), "COMP": ("FP", "MC", "SY")}
    g = np.random.default_rng(KB.SEED_BOOT); out = {}
    for comp, fs in fams.items():
        def delta(r, idx=None):
            return np.mean([np.mean(np.asarray(r["out"][f])[idx[f]] if idx else r["out"][f]) - np.mean(np.asarray(H1["out"][f])[idx[f]] if idx else H1["out"][f])
                            for f in fs if len(r["out"][f])])
        dH = delta(H3); draws = []
        for _ in range(B_):
            idx = {f: g.integers(0, len(H1["out"][f]), len(H1["out"][f])) for f in fs if len(H1["out"][f])}
            draws.append(delta(H3, idx))
        dR = [delta(r) for r in rows if r["cond"].startswith("RS39_")]
        pe = pboot(np.array(draws)); pr = p_rank(dH, dR) if dR else 1.0
        out[comp] = dict(delta=val(dH), ci=ci(np.array(draws)), p_effect=pe, p_rank=pr, p_iut=p_iut(pe, pr), delta_RS39=[val(x) for x in dR])
    hf = holm_family({k: out[k]["p_iut"] for k in out})
    for k in out:
        out[k]["holm"] = hf[k]; out[k]["string"] = f"H scaling raises {k} beyond damage-matched random sets" if hf[k]["reject"] else None
    h, c = out["HALL"]["holm"]["reject"], out["COMP"]["holm"]["reject"]
    read = ("hallucination and over-compliance (not separable here)" if h and c else "H-neurons encode hallucination" if h else
            "H-neurons encode over-compliance" if c else ("no specific causal role at this scale" if all(inside(out[k]["ci"], 0.05) for k in out) else "undecided"))
    dmg = ((rs.get("damages") or {}).get("H") or {}).get("a3")
    return dict(composites=out, reading=read, G_K12="damage-limited" if (fin(dmg) and dmg > 0.15) else None, damage_H_a3=val(dmg))


def k4_stats(S2, items):
    rows = S2["rows"]; cor = set(items["correct"]); it = items["items"]
    hit = {}
    for (v, i), rs in by([r for r in rows if r["ptype"] == "t0" and r["item"] in cor], "variant", "item").items():
        cells = by(rs, "block"); hit.setdefault(v, {})[i] = float(np.mean([anyhit(c) for c in cells.values()]))
    # V6: label-shuffle scoring of the V1 chains (derangement within the correct items, seed 51)
    ids = sorted(cor); g = np.random.default_rng(51); perm = dict(zip(ids, np.roll(g.permutation(ids), 1)))
    import r1b_k_labels as KL
    v6 = {}
    for (i,), rs in by([r for r in rows if r["variant"] == "V1" and r["ptype"] == "t0" and r["item"] in cor], "item").items():
        tgt = it[perm[i]]; cells = by(rs, "block")
        v6[i] = float(np.mean([float(any(KL.classify(v, tgt) == "TRUE-INT" for r in c for _, v, _ in r["claims"])) for c in cells.values()]))
    d12 = item_diff(hit.get("V1", {}), hit.get("V2", {})); d16 = item_diff(hit.get("V1", {}), v6)
    s = ("chain lens reads intermediates" if (fin(d12["ci"]["lo"]) and d12["ci"]["lo"] >= 0.05 and fin(d16["ci"]["lo"]) and d16["ci"]["lo"] >= 0.05) else
         "chain lens does not read intermediates" if (lt(d12["ci"], 0.05) and lt(d16["ci"], 0.05)) else "undecided")
    return dict(G_K4=s, V1_V2=_s(d12), V1_V6=_s(d16), p=p_iut(pboot(d12["draws"], 0.05) if len(d12["draws"]) else 1, pboot(d16["draws"], 0.05) if len(d16["draws"]) else 1),
                anyhit={v: val(np.mean(list(x.values()))) for v, x in hit.items()}, gvra=S2.get("gvra"), planted=S2.get("planted"),
                selfkl_mean=[val(x) for x in np.mean([r["kl"] for r in S2.get("selfkl", [])], 0)] if S2.get("selfkl") else None)


def k6a_stats(S2):
    rows = S2["rows"]; out = {}; ps = {}
    for l in sorted({r["block"] for r in rows}):
        sl = {}
        for v in ("V1", "V2", "V7", "V8", "G_lam0.5", "G_lam2", "G_pl0.05", "G_pl0.3", "G_greedy", "T0.5", "T1.0"):
            rs = [r for r in rows if r["variant"] == v and r["block"] == l]
            if rs:
                sl[v] = logit_slope(claim_counts(rs))
        if "V1" not in sl or "V2" not in sl:
            continue
        (e1, d1), (e2, d2) = sl["V1"], sl["V2"]; dd = d1 - d2
        S = lambda rs, k: np.mean([float(any(c == "CONFAB" and s <= k for s, _, c in r["claims"])) for r in rs]) if rs else np.nan
        v1 = [r for r in rows if r["variant"] == "V1" and r["block"] == l]
        g = np.random.default_rng(KB.SEED_BOOT); its = by(v1, "item"); ks = list(its)
        dS = np.array([np.mean([S(its[ks[j]], 16) - S(its[ks[j]], 4) for j in g.integers(0, len(ks), len(ks))]) for _ in range(B_)]) if ks else np.array([])
        cls = lambda c: "rising" if gt(c) else ("falling" if lt(c) else ("flat" if inside(c, 0.03) else "undecided"))
        prim = cls(ci(d1)); lab = []
        if "V8" in sl and cls(ci(sl["V8"][1])) not in (prim, "undecided") and prim != "undecided":
            lab.append("reinjection-sensitive")
        for gname in ("G_lam0.5", "G_lam2", "G_pl0.05", "G_pl0.3", "G_greedy"):
            if gname in sl and {cls(ci(sl[gname][1])), prim} in ({"rising", "falling"}, {"rising", "flat"}, {"flat", "falling"}) and prim != "undecided":
                lab.append("length profile decoding-sensitive"); break
        ps[l] = pboot(dd)
        out[l] = dict(slope_V1=val(e1), ci_V1=ci(d1), slope_V2=val(e2), diff=val(e1 - e2), diff_ci=ci(dd), p=ps[l], S16_minus_S4=ci(dS) if len(dS) else None,
                      labels=sorted(set(lab)), temperature={v: val(sl[v][0]) for v in ("T0.5", "T1.0") if v in sl},
                      _cls=prim, _flat=inside(ci(d1), 0.03) and (len(dS) and gt(ci(dS))), _drift=gt(ci(d1)) and inside(ci(dd), 0.03))
    hf = holm_family({l: ps[l] for l in ps})
    for l in out:
        o = out[l]
        o["string"] = ("per-step confabulation rises with chain length beyond the null state" if hf[l]["reject"] else
                       "per-step confabulation rises with chain length (generic rollout drift)" if o.pop("_drift") else
                       "flat per-step confabulation (cumulative only)" if o.pop("_flat", False) else "undecided")
        o.pop("_flat", None); o.pop("_drift", None); o.pop("_cls", None); o["holm"] = hf[l]
    return out


def k6d_stats(S2, items):
    rows = S2["rows"]; wr = set(items["wrong"])
    h = {v: {i: anyhit(rs, "WRONG-CONS") for (i,), rs in by([r for r in rows if r["variant"] == v and r["item"] in wr and r["ptype"] == "t0"], "item").items()}
         for v in ("V1", "V2")}
    d = item_diff(h["V1"], h["V2"]); return dict(diff=_s(d), p=pboot(d["draws"]) if len(d["draws"]) else 1.0, n_wrong=len(wr))


def k4p_stats(P):
    if not P:
        return nr("input missing")
    out, ps = {}, {}
    for l, r in P["res"].items():
        d = r["correct_probe"] - r["correct_base"]; dr = boot_stat(len(d), lambda ix: d[ix].mean())
        ps[l] = pboot(dr, 0.05); out[l] = dict(acc_probe=float(r["correct_probe"].mean()), acc_base=float(r["correct_base"].mean()), diff=float(d.mean()),
                                               ci=ci(dr), p=ps[l], perm_q95=float(np.percentile(r["perm"], 95)) if len(r["perm"]) else None,
                                               acc_v2_probe=float(r["correct_probe_v2"].mean()))
    hf = holm_family(ps)
    for l in out:
        out[l]["string"] = f"intermediate present in the read state (block {l})" if hf[l]["reject"] else None; out[l]["holm"] = hf[l]
    present = any(hf[l]["reject"] for l in hf)
    none = all(lt(out[l]["ci"], 0.05) for l in out)
    return dict(blocks=out, G_K4b="intermediate present" if present else ("intermediate not present (lens-free)" if none else "undecided"))


def k10c_stats(P):
    if not P:
        return nr("input missing")
    out, ps = {}, {}
    for l, res in P["res"].items():
        for k in (1, 2, 3):
            rr = [r for r in res if len(r["d"]) >= k]
            if len(rr) < 3:
                continue
            a = lambda key: np.array([float(len(r[key]) >= k and r[key][k - 1] == r["d"][k - 1]) for r in rr])
            cl, lp, lj = a("cl_top"), a("lpar"), a("ljac")
            st = lambda ix: cl[ix].mean() - max(lp[ix].mean(), lj[ix].mean())
            dr = boot_stat(len(rr), st); key = f"{l}:{k}"
            out[key] = dict(acc_CL=float(cl.mean()), acc_Lpar=float(lp.mean()), acc_Ljac=float(lj.mean()), acc_CL_d1sel=float(a("cl_d1sel").mean()),
                            diff=float(st(np.arange(len(rr)))), ci=ci(dr), p=pboot(dr))
            if k in (2, 3):
                ps[key] = out[key]["p"]
    hf = holm_family(ps)
    for key in out:
        if key in hf:
            out[key]["holm"] = hf[key]
            out[key]["string"] = (f"chained readout beats direct readouts at digit {key.split(':')[1]} (block {key.split(':')[0]})" if hf[key]["reject"] else
                                  f"direct readouts match or beat the chain at digit {key.split(':')[1]} (block {key.split(':')[0]})" if lt(out[key]["ci"], 0.05) else "undecided")
    return out


# =================================================================================================================================
# part 2: K6b, K6c, K6f, K6g, K6h
# =================================================================================================================================
def spline3(s):
    s = np.asarray(s, float); k1, k2 = np.percentile(s, [33, 67]) if len(s) else (0, 0)
    return np.c_[s, np.maximum(s - k1, 0) ** 3 / 100.0, np.maximum(s - k2, 0) ** 3 / 100.0]


def claim_table(rows, items, nrs, qa=False):
    """Per-claim rows (V1): y (CONFAB = 1; TRUE-INT / ANSWER / OWN / WRONG-CONS / ALT = 0; QA: CONFAB-QA vs GOLD / OWN), predictors at
    the position before the claim's first token: H~ lagged by one step, step, ENT, CDM, TOP1, e, FOV; fixed effects; R~_k for RS sets."""
    X, y, gid, Rt = [], [], [], []
    for r in rows:
        if r["variant"] != "V1":
            continue
        aH = r["aX"].get("H"); rsA = [r["aX"].get(f"RS{k}") for k in range(nrs)]
        if aH is None or any(a is None for a in rsA):
            continue
        aH = aH.float().numpy(); RA = np.stack([a.float().numpy() for a in rsA]) if nrs else None
        cl = r.get("mentions") if qa else r["claims"]
        for st, v, c in cl:
            if (not qa and c in ("PROMPT", "AMBIG")) or (qa and c == "PROMPT"):
                continue
            pf = (st if qa else st - len(v) + 1); q = pf - 1; lag = q - 1
            if lag < 0 or q >= len(aH):
                continue
            mu, sd = RA[:, lag].mean(), RA[:, lag].std() + 1e-9
            Ht = (aH[lag] - mu) / sd
            f = [Ht, pf] + [float(r[k][q]) if k in r else 0.0 for k in ("ent", "cdm", "top1", "e", "fov")]
            f = [0.0 if not np.isfinite(x) else x for x in f]
            fe = [r["block"], (items[r["item"]]["nops"] if (items and r["item"] in items) else 0), r.get("cls", r["ptype"])]
            X.append(f + fe); y.append(float(c in ("CONFAB", "CONFAB-QA"))); gid.append(r["item"])
            others = [(RA[k, lag] - np.delete(RA[:, lag], k).mean()) / (np.delete(RA[:, lag], k).std() + 1e-9) for k in range(nrs)]
            Rt.append(others)
    return X, np.array(y), np.array(gid), np.array(Rt)


def design(X, use_cov=True):
    X = np.array(X, dtype=object)
    if len(X) == 0:
        return np.zeros((0, 1))
    num = np.array(X[:, :7], float); fe = X[:, 7:]
    cols = [num[:, :1]]
    if use_cov:
        cols += [spline3(num[:, 1]), num[:, 2:7]]
        for j in range(fe.shape[1]):
            lv = sorted(set(map(str, fe[:, j])))
            cols += [np.array([[float(str(v) == l) for l in lv[1:]] for v in fe[:, j]]).reshape(len(X), -1)]
    return np.concatenate([c for c in cols if c.size], 1)


def k6b_rule(X, y, g, Rt, B=B_):
    """K6b(i) / K6h(Q12) / K8c-style rule: p_IUT = max(p(beta_H <= 0), p_rank(beta_H vs beta_R)); 'length artifact' (three parts);
    'no association' (marginal within +-0.10 per SD); also the PI form of the specificity p (CF-K-link, main change K.12)."""
    if len(y) < 20 or len(set(y.tolist())) < 2:
        return dict(string="undecided", code=coded("NaN:too_few_claims"), p_iut=1.0, p_iut_pi=1.0)
    Xf = design(X); sdH = Xf[:, 0].std() + 1e-9
    fit = lambda ix, Z: logit_fit(Z[ix], y[ix], l2=1e-4)
    bH = fit(np.arange(len(y)), Xf)["beta"][0] * sdH
    dr = boot_stat(len(y), lambda ix: fit(ix, Xf)["beta"][0] * sdH, B, groups=g)
    Xm = design(X, use_cov=False); bm = fit(np.arange(len(y)), Xm)["beta"][0] * sdH
    dm = boot_stat(len(y), lambda ix: fit(ix, Xm)["beta"][0] * sdH, B, groups=g)
    bR = []
    for k in range(Rt.shape[1] if Rt.ndim == 2 else 0):
        Z = Xf.copy(); Z[:, 0] = Rt[:, k]; bR.append(fit(np.arange(len(y)), Z)["beta"][0] * (Rt[:, k].std() + 1e-9))
    pe = pboot(dr); pr = p_rank(bH, bR) if bR else 1.0; ppi = p_pi(bH, bR) if len(bR) >= 2 else 1.0
    c, cm = ci(dr), ci(dm)
    art = gt(cm) and inside(c, 0.10) and fin(c["hi"]) and c["hi"] < 0.5 * bm
    s = None
    return dict(beta=val(bH), ci=c, beta_marginal=val(bm), ci_marginal=cm, p_effect=pe, p_rank=pr, p_pi=ppi if fin(ppi) else 1.0,
                p_iut=p_iut(pe, pr), p_iut_pi=p_iut(pe, ppi if fin(ppi) else 1.0), beta_R=[val(x) for x in bR],
                _art=bool(art), _none=bool(inside(cm, 0.10)), n_claims=int(len(y)))


def _k6b_strings(r, holm_ok, sfx=""):
    if "code" in r and r.get("string"):
        return r["string"]
    if holm_ok:
        return "H-activity predicts lens confabulation beyond chain length" + sfx
    if r.get("_art"):
        return "length artifact (lens side)" if not sfx else "length artifact (QA lens)"
    if r.get("_none"):
        return "no association (lens side)" if not sfx else "no association (QA lens)"
    return "undecided"


def ana_part2(ctx):
    torch = KB.T_()
    res = {}; S2 = _pt(ctx, "r1b_K_S2a", "k/k4/summary.pt"); items = _pt(ctx, "r1b_K_S2a", "k/k4/items.pt"); rs = _js(ctx, "r1b_K_S1", "hn/rs_sets.json") or {}
    nrs = len(rs.get("RS39", []))
    if S2 and items:
        X, y, g, Rt = claim_table(S2["rows"], items["items"], nrs)
        r1 = k6b_rule(X, y, g, Rt)
        r2 = k6b_ii(ctx, S2, items, rs)
        hf = holm_family({"i": r1.get("p_iut", 1.0), "ii": r2.get("p_iut", 1.0)})
        r1["string"] = _k6b_strings(r1, hf["i"]["reject"]); r2["string"] = ("H-activity predicts lens confabulation beyond chain length"
                                                                            if hf["ii"]["reject"] else r2.get("string", "undecided"))
        r1["holm"], r2["holm"] = hf["i"], hf["ii"]
        res["K6b"] = dict(i={k: v for k, v in r1.items() if not k.startswith("_")}, ii=r2, iii=k6b_lag(S2, nrs, r1))
    else:
        res["K6b"] = nr("input missing")
    res["K6c"] = k6c_stats(_parts(ctx, "r1b_K_S2a", "k6c", 4))
    res["K6f"] = k6f_stats(ctx, S2)
    g6 = _pt(ctx, "r1b_K_S2a", "k/k6g.pt"); res["K6g"] = k6g_stats(g6) if g6 else nr("input missing")
    res["K6h"] = k6h_stats(_parts(ctx, "r1b_K_S2a", "k6h", 2), nrs)
    write(ctx, "k/ana2.json", res, labels=["r1b_K_ana part 2"])


def k6b_ii(ctx, S2, items, rs):
    """rho_H per cell (direct H_A writes of layers <= l at the read position over ||h - mu_task||) and the cell model q ~ rho_H + depth +
    block (OLS normal equations, # §2.5 fitting step); RS39 analogues; p_IUT with the rank form."""
    torch = KB.T_()
    St = _pt(ctx, "r1b_K_S2a", "k/k4/states.pt")
    if St is None or St["St"]["t0"].get("z") is None:
        return dict(string="undecided", code=coded("NA:not_run:states missing"), p_iut=1.0)
    tok, model = ctx.model(); Ls = __import__("r1b_core").layers_of(model); R = _js(ctx, "r1b_K_S1", "hn/hn_meta.json")
    zmu = np.asarray(R["refw"]["z_mu"]); uni = [tuple(x) for x in St["union"]]; pos = {u: k for k, u in enumerate(uni)}
    sets = {"H": St["HA"], **{f"RS{k}": s for k, s in enumerate(rs.get("RS39", []))}}; s0 = St["St"]["t0"]
    q = {(r["item"], r["block"]): [] for r in S2["rows"] if r["variant"] == "V1" and r["ptype"] == "t0"}
    for r in S2["rows"]:
        if (r["item"], r["block"]) in q and r["variant"] == "V1" and r["ptype"] == "t0":
            q[(r["item"], r["block"])].append(qshare(r["claims"]))
    cells = [(i, l, np.nanmean(v)) for (i, l), v in q.items() if np.isfinite(v).any()]
    rho = {}
    for nm, st in sets.items():
        vals = []
        for i, l, _ in cells:
            k = s0["items"].index(i); w = torch.zeros(Ls[0].mlp.down_proj.weight.shape[0])
            for (ll, j) in [tuple(x) for x in st if x[0] <= l and tuple(x) in pos]:
                w += (float(s0["z"][k, pos[(ll, j)]]) - zmu[ll][j]) * Ls[ll].mlp.down_proj.weight[:, j].float().cpu()
            vals.append(float(w.norm() / (s0["h"][l][k] - s0["mu"][l][k]).norm().clamp(min=1e-9)))
        rho[nm] = np.array(vals)
    if len(cells) < 5:
        return dict(string="undecided", code=coded("NaN:too_few_cells"), p_iut=1.0)
    yq = np.array([c[2] for c in cells]); dep = np.array([items["items"][c[0]]["nops"] for c in cells], float); blk = np.array([c[1] for c in cells], float)
    gid = np.array([c[0] for c in cells])

    def beta(x, ix):
        Z = np.c_[np.ones(len(ix)), (x[ix] - x[ix].mean()) / (x[ix].std() + 1e-9), dep[ix], blk[ix]]
        A = Z.T @ Z + 1e-9 * np.eye(Z.shape[1])
        return float(np.linalg.solve(A, Z.T @ yq[ix])[1])           # §2.5 fitting step (OLS normal equations)
    bH = beta(rho["H"], np.arange(len(yq))); dr = boot_stat(len(yq), lambda ix: beta(rho["H"], ix), groups=gid)
    bR = [beta(rho[k], np.arange(len(yq))) for k in rho if k != "H"]
    pe = pboot(dr); pr = p_rank(bH, bR) if bR else 1.0
    return dict(beta=val(bH), ci=ci(dr), p_effect=pe, p_rank=pr, p_iut=p_iut(pe, pr), p_iut_pi=p_iut(pe, p_pi(bH, bR) if len(bR) > 1 else 1.0),
                rho_H=KB.dist(rho["H"]))


def k6b_lag(S2, nrs, r1):
    """K6b(iii): lag_H = mean over first-CONFAB steps B in [3, 14] of a_H(B + 1) - a_H,ctrl(B + 1) (SD units; control = the same item's
    other-seed chains unbroken at B + 1); lag_R per RS set; strings."""
    v1 = [r for r in S2["rows"] if r["variant"] == "V1" and "H" in r["aX"]]
    allH = np.concatenate([r["aX"]["H"].float().numpy() for r in v1]) if v1 else np.array([0.0]); sd = allH.std() + 1e-9
    def lag(name):
        out, gid = [], []
        for (i, l, p), rs in by(v1, "item", "block", "ptype").items():
            kb = {id(r): next((s for s, _, c in r["claims"] if c == "CONFAB"), None) for r in rs}
            for r in rs:
                b_ = kb[id(r)]
                if b_ is None or not (3 <= b_ <= 14) or name not in r["aX"]:
                    continue
                ctrl = [o["aX"][name][b_ + 1].item() for o in rs if o is not r and (kb[id(o)] is None or kb[id(o)] > b_ + 1) and len(o["aX"][name]) > b_ + 1]
                if ctrl and len(r["aX"][name]) > b_ + 1:
                    out.append((r["aX"][name][b_ + 1].item() - np.mean(ctrl)) / sd); gid.append(i)
        return np.array(out), np.array(gid)
    lH, g = lag("H")
    if lH.size < 3:
        return dict(string="lead/lag undecided", code=coded("NaN:too_few_breaks"))
    c = ci(boot_stat(len(lH), lambda ix: lH[ix].mean(), groups=g)); lR = [lag(f"RS{k}")[0].mean() for k in range(nrs) if lag(f"RS{k}")[0].size]
    lead = r1.get("p_iut", 1.0) <= 0.05; follow = gt(c) and lR and lH.mean() > np.percentile(lR, 95)
    s = ("H rise leads and follows" if lead and follow else "H rise leads the break" if lead else "H rise follows the break only" if follow else
         "no H change around the break" if (inside(c, 0.10) and (r1.get("_art") or r1.get("_none"))) else "lead/lag undecided")
    return dict(lag_H=val(lH.mean()), ci=c, lag_R=[val(x) for x in lR], string=s)


def k6c_stats(rows):
    if not rows:
        return nr("input missing")
    q = {}; hits = {}; css = {}
    for r in rows:
        for c in r["chains"]:
            q.setdefault(r["cond"], {}).setdefault((c["item"], r["block"]), []).append(qshare(c["claims"], 1, 16))
            hits.setdefault(r["cond"], {}).setdefault(c["item"], []).append(float(any(x == "TRUE-INT" for _, _, x in c["claims"])))
        css.setdefault(r["cond"], []).extend(r["css"])
    item = lambda cond: {i: np.nanmean([np.nanmean(v) for (it, l), v in q.get(cond, {}).items() if it == i]) for i in {it for it, _ in q.get(cond, {})}}
    base = item("none"); out = {}
    for kind, lab in (("clamp", "(i)"), ("ablate", "(ii)")):
        dH = item_diff(item(f"{kind}_H"), base)
        dR = [np.nanmean([v - base[i] for i, v in item(c).items() if i in base]) for c in q if c.startswith(f"{kind}_MC")]
        pe = pboot(dH["draws"], 0.0, "less") if len(dH["draws"]) else 1.0
        pr = (1 + sum(x <= dH["est"] for x in dR if fin(x))) / (len(dR) + 1) if (dR and fin(dH["est"])) else 1.0
        cs = float(np.nanmean(css.get(f"{kind}_H", [np.nan]))) if css.get(f"{kind}_H") else np.nan
        hitd = np.nanmean(list(item(f"{kind}_H").values())) if False else None
        out[lab] = dict(delta=_s(dH), p_effect=pe, p_rank=pr, p_iut=p_iut(pe, pr), CSS=val(cs), delta_R=[val(x) for x in dR],
                        p_iut_pi=p_iut(pe, p_pi(-dH["est"], [-x for x in dR]) if (len(dR) > 1 and fin(dH["est"])) else 1.0))
    hf = holm_family({k: out[k]["p_iut"] for k in out})
    for k, o in out.items():
        role = "H in the rollout" if k == "(i)" else "H writes in the read state"
        cancelled = fin(o["CSS"]["value"] if isinstance(o["CSS"], dict) else o["CSS"]) and (o["CSS"] < 0.2 if not isinstance(o["CSS"], dict) else False)
        if hf[k]["reject"]:
            o["string"] = f"{k} {'H clamp in the rollout' if k == '(i)' else 'removing H writes from the read state'} reduces lens confabulation specifically"
        elif inside(o["delta"]["ci"], 0.03) and not cancelled and fin(o["CSS"] if not isinstance(o["CSS"], dict) else np.nan) and o["CSS"] >= 0.2:
            o["string"] = f"{k} no specific causal role of {role}"
        elif gt(o["delta"]["ci"]) and o["delta_R"] and fin(o["delta"]["est"]) and o["delta"]["est"] > max(x for x in o["delta_R"] if fin(x)):
            o["string"] = "H clamp increases confabulation (against the prediction)"
        else:
            o["string"] = "undecided"
        o["labels"] = ["intervention largely cancelled by the contrast"] if (fin(o["CSS"]) and o["CSS"] < 0.2) else []
        o["holm"] = hf[k]
    return out


def k6f_stats(ctx, S2):
    rows = _parts(ctx, "r1b_K_S2a", "k6f", 2)
    par = {}
    for i in range(2):
        d = _pt(ctx, "r1b_K_S2a", f"k/k6f/part{i}.pt")
        if d:
            par.update(d.get("parity", {}))
    if not rows or not S2:
        return nr("input missing")
    if par and not all(v.get("ok") for v in par.values()):
        return dict(string="not run (engine parity)", parity=par)
    bstar = _pt(ctx, "r1b_K_S2a", f"k/k6f/part0.pt")["b_star"]
    C16 = lambda r: float(any(c == "CONFAB" for _, _, c in r["claims"]))
    base = {(r["item"], r["seed"]): C16(r) for r in S2["rows"] if r["variant"] == "V1" and r["block"] == bstar and r["ptype"] == "t0"}
    pairs = [(r["item"], r.get("inst", 0), C16(r) - base[(r["item"], r["seed"])]) for r in rows if r["variant"] == "V1_F2048" and (r["item"], r["seed"]) in base]
    if len(pairs) < 3:
        return dict(string="undecided", code=coded("NaN:too_few_pairs"), parity=par)
    insts = sorted({p[1] for p in pairs}); g = np.random.default_rng(KB.SEED_BOOT); d = np.array([p[2] for p in pairs])
    draws = []
    for _ in range(B_):
        sel = []
        for ins in g.choice(insts, len(insts)):
            its = sorted({p[0] for p in pairs if p[1] == ins})
            for it in g.choice(its, len(its)):
                sel += [p[2] for p in pairs if p[0] == it and p[1] == ins]
        draws.append(np.mean(sel) if sel else np.nan)
    c = ci(np.array(draws)); p = pboot(np.array(draws))
    return dict(d_off=float(d.mean()), ci=c, p=p, parity=par,
                _eq=inside(c, 0.05), string=None)


def k6g_stats(G):
    rows = G["rows"]; q = {}
    for r in rows:
        f = len(r.get("forced", [])) if "forced" in r else 0
        cl = [(s, v, c) for s, v, c in r["claims"] if s > 4 and not (r["variant"] == "wrong" and v == str(r.get("w")))]
        q.setdefault(r["variant"], {})[(r["item"], r["seed"])] = qshare(cl)
    d = item_diff(q.get("wrong", {}), q.get("correct", {}))
    dn = item_diff(q.get("neutral", {}), q.get("correct", {}))
    return dict(d_con=_s(d), p=pboot(d["draws"]) if len(d["draws"]) else 1.0, neutral_vs_correct=_s(dn), G_K19=G.get("G_K19"), _eq=inside(d["ci"], 0.05))


def k6h_stats(rows, nrs):
    if not rows:
        return nr("input missing")
    hit = lambda cat, cls, var: {i: float(any(c == cat for r in rs for _, _, c in r["mentions"])) for (i,), rs in
                                 by([r for r in rows if r["variant"] == var and r.get("cls") == cls], "item").items()}
    gold = item_diff(hit("GOLD", "CC", "V1"), hit("GOLD", "CC", "V2")); ownd = item_diff(hit("OWN", "CI", "V1"), hit("OWN", "CI", "V2"))
    gk23 = "lens reads the gold answer (QA)" if (fin(gold["ci"]["lo"]) and gold["ci"]["lo"] >= 0.05) else "QA H-link not testable (QA lens fails its positive control)"
    X, y, g, Rt = claim_table(rows, None, nrs, qa=True)
    r = k6b_rule(X, y, g, Rt) if gk23.startswith("lens reads") else dict(string="not testable (QA lens fails its positive control)", p_iut=1.0, p_iut_pi=1.0)
    return dict(G_K23=gk23, gold=_s(gold), own=_s(ownd), p_own=pboot(ownd["draws"]) if len(ownd["draws"]) else 1.0,
                Q12={k: v for k, v in r.items() if not k.startswith("_")}, _r=r)


# =================================================================================================================================
# part 3: K6e, K8a, K8b, K8c, K7a / K7b, K9, K10a, K10b
# =================================================================================================================================
SEG = 8


def _E(run, twin, name, seg):
    a, b = run["aX"].get(name), twin["aX"].get(name)
    if a is None or b is None:
        return np.nan
    return float((a[seg * SEG:(seg + 1) * SEG].float() - b[seg * SEG:(seg + 1) * SEG].float()).mean())


def k6e_stats(ctx):
    tw = _pt(ctx, "r1b_K_S2b", "k/k6e/twin.pt")
    rows = _parts(ctx, "r1b_K_S2b", "k6e", 8)
    if not tw or not rows:
        return nr("input missing")
    twin = {r["unit"]: r for r in tw["tw"][0.7]}; twin1 = {r["unit"]: r for r in tw["tw"].get(1.0, [])}
    units = list(twin); out = dict(G_K21=tw.get("G_K21"))
    # s_1 and s_0: twin-vs-twin (different keys) differences
    pairs = [(u, (u[0], u[1], 1 - u[2] if u[2] < 2 else u[2] - 1)) for u in units if u[2] == 0]
    s1 = np.std([_E(twin[a], twin[b], "H", 0) for a, b in pairs if b in twin] or [np.nan], ddof=1)
    s0 = np.std([_E(twin[a], twin[b], "H", 2) for a, b in pairs if b in twin] or [np.nan], ddof=1)
    cond = {r["cond"]: r for r in rows}

    def carry(r, X, route):
        runs = {x["unit"]: x for x in (r["tok"] if route == "TOK" else r["pushed"])}
        tws = twin1 if r["T"] == 1.0 else twin
        E = {n: np.array([_E(runs[u], tws[u], X, n) for u in units if u in runs and u in tws]) for n in range(3)}
        it = np.array([u[0] for u in units if u in runs and u in tws])
        return E, it
    res = {}
    for nm, route in (("PUSH_H", "TOK"), ("PUSH_H", "TOK+KV")):
        if nm not in cond:
            continue
        E, it = carry(cond[nm], "H", route)
        rc = KB.ratio_carry(E[2], E[1], it)                                   # K = E_3 / E_2 for both routes (K.12, validity-1)
        d1 = float(np.nanmean(E[0]) / s1) if fin(s1) and s1 > 0 else np.nan
        KR_ = []
        for k in range(20):
            c = cond.get(f"PUSH_RS{k}")
            if c:
                Er, itr = carry(c, f"RS{k}", "TOK"); kr = KB.ratio_carry(Er[2], Er[1], itr, B=200)
                if kr["defined"] and fin(kr["K"]) and kr["K"] > 0:
                    KR_.append(math.log(kr["K"]))
        res[route] = dict(carry={k: v for k, v in rc.items()}, d1=val(d1), logK_R=KR_, n_R_defined=len(KR_))
    out["carry"] = res
    # bistability (TOK, delta2): diverged chain-pairs only (K.12, validity-3)
    if "PUSH_H" in cond:
        runs = {x["unit"]: x for x in cond["PUSH_H"]["tok"]}
        z, it, ndiv = [], [], 0
        for u in units:
            if u in runs:
                if runs[u]["tok"] != twin[u]["tok"]:
                    z.append(_E(runs[u], twin[u], "H", 2) / s0 if s0 > 0 else np.nan); it.append(u[0])
                else:
                    ndiv += 1
        z = np.array(z); it = np.array(it); ok = np.isfinite(z)
        mix = KB.mixture_lr(z[ok], it[ok]); tail = KB.robust_tail_share(z[ok], it[ok])
        lrR = []
        for k in range(20):
            c = cond.get(f"PUSH_RS{k}")
            if c:
                rr = {x["unit"]: x for x in c["tok"]}; zr = [(_E(rr[u], twin[u], f"RS{k}", 2) / s0) for u in units if u in rr and rr[u]["tok"] != twin[u]["tok"]]
                m = KB.mixture_lr(np.array(zr), np.array([0] * len(zr)), B=50) if len(zr) >= 10 else None
                if m and fin(m["LR"]):
                    lrR.append(math.log1p(max(m["LR"], 0)))
        locked = {i for i, x in zip(it[ok], z[ok]) if x > 3}; relaxed = {i for i, x in zip(it[ok], z[ok]) if abs(x) < 1}
        out["bist"] = dict(mix=mix, tail=tail, undiverged_share=ndiv / max(len(units), 1), lr_R=lrR,
                           locked_with_relaxed=len(locked & relaxed) / max(len(locked), 1) if locked else 0.0)
    # causal (Q11): d_br = per-claim confabulation share in segments 2-3, TOK vs twin
    def dbr(c, T=0.7):
        runs = {x["unit"]: x for x in c["tok"]}; tws = twin1 if T == 1.0 else twin; d = {}
        for u in units:
            if u in runs and u in tws:
                d.setdefault(u[0], []).append(qshare(runs[u]["claims"], SEG, 99) - qshare(tws[u]["claims"], SEG, 99))
        return {i: np.nanmean(v) for i, v in d.items() if np.isfinite(v).any()}
    for nm, sign, rpre in (("PUSH_H", 1, "PUSH_RS"), ("CLAMP_H", -1, "CLAMP_MC")):
        if nm not in cond:
            continue
        dH = dbr(cond[nm]); v = np.array(list(dH.values())); dr = boot_stat(len(v), lambda ix: v[ix].mean()) if len(v) > 2 else np.array([])
        dR = [np.nanmean(list(dbr(cond[c]).values())) for c in cond if c.startswith(rpre)]
        pe = pboot(sign * dr) if len(dr) else 1.0; pp = p_pi(sign * v.mean(), [sign * x for x in dR]) if len(dR) > 1 and len(v) else 1.0
        out[f"causal_{nm}"] = dict(d_br=val(v.mean() if len(v) else np.nan), ci=ci(dr) if len(dr) else None, p_effect=pe, p_pi=pp if fin(pp) else 1.0,
                                   p_iut=p_iut(pe, pp if fin(pp) else 1.0), CSS=val(np.nanmean(cond[nm]["css"])) if cond[nm]["css"] else None)
    if "PUSH2_H" in cond:
        out["both_stream_CSS"] = val(np.nanmean(cond["PUSH2_H"]["css"])) if cond["PUSH2_H"]["css"] else None
    off = _pt(ctx, "r1b_K_S2b", "k/k6e/offset.pt")
    if off and off["rows"]:
        tw2 = {x["unit"]: x for r in off["rows"] for x in r["twin"]}; tk2 = {x["unit"]: x for r in off["rows"] for x in r["tok"]}
        uu = [u for u in tk2 if u in tw2]
        E2 = np.array([_E(tk2[u], tw2[u], "H", 1) for u in uu]); E3 = np.array([_E(tk2[u], tw2[u], "H", 2) for u in uu])
        k2 = KB.ratio_carry(E3, E2, np.array([u[0] for u in uu])); out["offset"] = dict(carry=k2)
    return out


def k6e_strings(o):
    """Q9 members 1-3 and Q11 strings (Holm applied in verdict())."""
    s = {}
    for route in ("TOK", "TOK+KV"):
        r = (o.get("carry") or {}).get(route)
        if not r:
            s[route] = dict(string="not run (input missing)", p=1.0); continue
        c = r["carry"]
        if not c["defined"]:
            s[route] = dict(string="K not defined (segment-1 excess not resolved)", p=1.0, labels=["underpowered (d1 < 0.25)"] if fin(r["d1"]) and r["d1"] < 0.25 else []); continue
        ppi = p_pi(c["logK"], r["logK_R"]) if (r["n_R_defined"] >= 10 and fin(c["logK"])) else coded("NaN:specificity_not_testable")
        s[route] = dict(K=c["K"], ci=c["K_ci"], p_le1=c["p_le1"], p_pi=ppi, p=p_iut(c["p_le1"], ppi), _c=c, _r=r,
                        labels=(["underpowered (d1 < 0.25)"] if fin(r["d1"]) and r["d1"] < 0.25 else []) + ([] if r["n_R_defined"] >= 10 else ["specificity not testable"]))
    return s


def ana_part3(ctx):
    res = {"K6e": k6e_stats(ctx)}
    res["K8a"] = k8a_stats(_parts(ctx, "r1b_K_S4", "k8a", 4))
    res["K8b"] = k8b_stats(_parts(ctx, "r1b_K_S4", "k8b", 12))
    res["K8c"] = k8c_stats(ctx)
    k7 = _js(ctx, "r1b_K_S4", "k/k7a.json"); res["K7a"] = dict(n_channels=len((k7 or {}).get("channels", {})), note="exploratory", data=k7 and k7.get("channels"))
    res["K7b"] = k7b_stats(_js(ctx, "r1b_K_S5", "k/k7b.json"))
    res["K9"] = k9_stats(_parts(ctx, "r1b_K_S5", "k9", 6), nsplit=200 if not ctx.mock else 20, nboot=200 if not ctx.mock else 10)
    res["K10a"] = k10a_stats(_parts(ctx, "r1b_K_S5", "k10a", 3))
    res["K10b"] = k10b_stats(_js(ctx, "r1b_K_S5", "k/k10b.json"))
    res["K3f"] = _js(ctx, "r1b_K_S5", "k/k3f.json") or nr("input missing")
    write(ctx, "k/ana3.json", res, labels=["r1b_K_ana part 3"])


def k8a_stats(rows):
    if not rows:
        return nr("input missing")
    cell = lambda r: (r["kind"], r["L"])
    Ls = sorted({r["L"] for r in rows if r["kind"] == "FILLER"}); mid = Ls[len(Ls) // 2] if Ls else 0
    M = {}
    for r in rows:
        M.setdefault(cell(r), {})[(r["fam"], r["item"])] = r
    C = {"C1": (("FILLER", mid), ("NONE", 0)), "C2": (("COHERENT", mid), ("FILLER", mid)), "C3": (("COHERENT", mid), ("SHUFFLED", mid)),
         "C4": (("OWN-CORRECT", mid), ("COHERENT", mid)), "C5": (("INJECTED-WRONG", mid), ("OWN-CORRECT", mid)),
         "C6": (("OWN-WRONG", mid), ("INJECTED-WRONG", mid)), "C7": (("FILLER", Ls[-1] if Ls else 0), ("FILLER", Ls[0] if Ls else 0))}
    out = {}
    for oc in ("acc", "H_gen", "H_tf"):
        sdb = np.nanstd([r[oc] for r in M.get(("NONE", 0), {}).values() if r[oc] is not None]) or 1.0
        m = 0.04 if oc == "acc" else 0.10 * sdb; ps, st = {}, {}
        for k, (a, b) in C.items():
            A_, B2 = M.get(a, {}), M.get(b, {}); ks = [x for x in A_ if x in B2 and A_[x][oc] is not None and B2[x][oc] is not None]
            if len(ks) < 3:
                st[k] = dict(code=coded("NaN:too_few_items")); ps[k] = 1.0; continue
            d = np.array([A_[x][oc] - B2[x][oc] for x in ks]); inst = np.array([A_[x]["inst"] for x in ks])
            dr = boot_stat(len(d), lambda ix: d[ix].mean(), groups=inst)
            p2 = min(1.0, 2 * min(pboot(dr), pboot(dr, 0.0, "less"))); ps[k] = p2; st[k] = dict(est=float(d.mean()), ci=ci(dr), p=p2, _eq=inside(ci(dr), m))
        hf = KB.holm_family(ps); adj = {k: hf[k]["p_holm"] <= 0.10 for k in hf}
        strs = {k: (f"{k} changes {oc}" if adj[k] else (f"no {k} effect on {oc}" if st[k].get("_eq") else "undecided")) for k in C}
        noeff = {k: strs[k].startswith("no ") for k in C}
        summ = ("pure length effect" if ((adj["C1"] or adj["C7"]) and all(noeff[k] for k in ("C2", "C3", "C4", "C5", "C6"))) else
                "content effect beyond length" if any(adj[k] for k in ("C2", "C3", "C4", "C5", "C6")) else
                "no length or content effect" if all(noeff.values()) else "undecided")
        out[oc] = dict(contrasts={k: {**{a: b for a, b in st[k].items() if not a.startswith("_")}, "string": strs[k], "holm": hf[k]} for k in C}, summary=summ)
    return out


def k8b_stats(rows):
    if not rows:
        return nr("input missing")
    S = {(r["block"], r["arm"]): {s["sid"]: s for s in r["sessions"]} for r in rows}

    def ebar(blk, arm, key="sH"):
        p, t = S.get((blk, arm), {}), S.get((blk, "clean"), {})
        return {sid: [(p[sid]["turns"][n][key] or 0) - (t[sid]["turns"][n][key] or 0) for n in range(8)] for sid in p if sid in t}
    out = {}
    for name, blk, arm, rs_pre in (("ACH", "main", "H_a3.0_FULL", "RS"), ("TQA", "tqa", "Htq_a3.0_FULL", "RS")):
        e = ebar(blk, arm)
        if not e:
            out[name] = dict(string="not run (input missing)", p=1.0); continue
        num = np.array([sum(v[2:8]) for v in e.values()]); den = np.array([sum(v[1:7]) for v in e.values()]); u = np.arange(len(num))
        rc = KB.ratio_carry(num, den, u); KR_ = []
        for (b, a) in S:
            if b == blk and a.startswith("RS"):
                er = ebar(blk, a); nr_ = np.array([sum(v[2:8]) for v in er.values()]); dr_ = np.array([sum(v[1:7]) for v in er.values()])
                k = KB.ratio_carry(nr_, dr_, np.arange(len(nr_)), B=200)
                if k["defined"] and fin(k["K"]) and k["K"] > 0:
                    KR_.append(math.log(k["K"]))
        late = np.array([sum(v[3:8]) for v in e.values()])
        routes = {}
        for r_ in ("TOKEN", "KV"):
            er = ebar(blk, f"H_a3.0_{r_}")
            if er:
                lt_ = np.array([sum(v[3:8]) for v in er.values()]); routes[r_] = ci(boot_stat(len(lt_), lambda ix: lt_[ix].mean()))
        out[name] = dict(carry=rc, logK_R=KR_, late=ci(boot_stat(len(late), lambda ix: late[ix].mean())), routes=routes)
    # bistability (ACH, bist block): z per session-seed over answers 6-8 (diverged sessions only)
    cl, pu = S.get(("bist", "clean"), {}), S.get(("bist", "H_a3.0_FULL"), {})
    if cl and pu:
        L = lambda s: np.mean([(t["sH"] or 0) for t in s["turns"][5:8]])
        bytpl = {}
        for s in cl.values():
            bytpl.setdefault(s["template"], []).append(L(s))
        s0 = np.sqrt(np.mean([np.var(v, ddof=1) * 2 for v in bytpl.values() if len(v) > 1] or [np.nan]))
        z = [(L(pu[k]) - L(cl[k])) / s0 for k in pu if k in cl and [t["tok"] for t in pu[k]["turns"]] != [t["tok"] for t in cl[k]["turns"]]]
        it = [pu[k]["template"] for k in pu if k in cl and [t["tok"] for t in pu[k]["turns"]] != [t["tok"] for t in cl[k]["turns"]]]
        out["bist"] = dict(mix=KB.mixture_lr(np.array(z), np.array(it)), tail=KB.robust_tail_share(np.array(z), np.array(it)),
                           undiverged_share=1 - len(z) / max(len(pu), 1))
    return out


def k8c_stats(ctx):
    import r1b_k_labels as KL
    rows = _parts(ctx, "r1b_K_S4", "k8c", 6)
    if not rows:
        return nr("input missing")
    fmt_fail = 1 - np.mean([r["fmt_ok"] for r in rows]); X, y, g, Rt = [], [], [], []
    for r in rows:
        if not r["fmt_ok"]:
            continue
        T = len(r["steps"]) if r["istar"] is None else r["istar"] + 1
        per = r["per"]
        for i in range(1, T):
            k = min(i - 1, len(per) - 1)
            if k < 0 or per[k].get("Ht") is None:
                continue
            Ht = per[k]["Ht"]; ent = per[k]["ent"] or 0.0
            X.append([Ht, i, ent, 0, 0, 0, 0, r["steps"][i]["cls"], int(r.get("preamble", 0)), "t"]); y.append(float(r["istar"] == i)); g.append(r["item"])
            rs = per[k]["rs"]; Rt.append([(rs[j] - np.mean(np.delete(rs, j))) / (np.std(np.delete(rs, j)) + 1e-9) for j in range(len(rs))])
    y = np.array(y); g = np.array(g); Rt = np.array(Rt) if Rt and all(len(x) == len(Rt[0]) for x in Rt) else np.zeros((len(y), 0))
    r = k6b_rule(X, y, g, Rt)
    s = ("H-activity predicts the next error beyond length, position, difficulty and uncertainty" if r.get("p_iut", 1.0) <= 0.05 else
         "length artifact (model side)" if r.get("_art") else "no association (model side)" if r.get("_none") else "undecided")
    return dict(string=s, rule={k: v for k, v in r.items() if not k.startswith("_")}, G_K13="format not followed" if fmt_fail > 0.20 else None,
                parse_fail=val(fmt_fail), n_first_errors=int(y.sum()) if len(y) else 0)


def k7b_stats(d):
    if not d:
        return nr("input missing")
    rows = d["rows"]; out = {}
    for (l, ch), rs in by(rows, "block", "ch").items():
        out[f"{l}:{ch}"] = dict(f=float(np.mean([r["fwd"] for r in rs])), b=float(np.mean([r["bwd"] for r in rs])),
                               horizon=float(np.mean([r["horizon"] for r in rs])))
    ps = {}
    for k, v in out.items():
        l, ch = k.split(":")
        if ch.startswith("r") or ch == "full":
            continue
        band = [out[x]["f"] for x in out if x.split(":")[0] == l and x.split(":")[1].startswith("r16_" if ch in ("Hsub", "Jws", "pc16") else "r1_")]
        ps[k] = p_rank(v["f"], band) if band else 1.0
    if ps:
        adj, rej = bh(list(ps.values()))
        for (k, p), a, r_ in zip(ps.items(), adj, rej):
            out[k].update(p_rank=p, q=a, string=f"channel {k} carries forward item content through the lens (exploratory)" if r_ else None)
    return out


def _rank(G, chans, blocks):
    """Least-squares ranking of net outflow after regressing out the block difference (# §2.5 fitting step)."""
    pairs = [(a, b) for a in chans for b in chans if (a, b) in G and fin(G[(a, b)]) and G[(a, b)] > 0]
    if len(pairs) < 2:
        return {c: 0.0 for c in chans}
    v = np.log([G[p] for p in pairs]); db = np.array([blocks[b] - blocks[a] for a, b in pairs], float)
    Z = np.c_[np.ones(len(v)), db]; beta = np.linalg.solve(Z.T @ Z + 1e-9 * np.eye(2), Z.T @ v); r = v - Z @ beta
    n = len(chans); ix = {c: k for k, c in enumerate(chans)}; A = np.zeros((len(pairs) + 1, n)); yv = np.r_[r, 0.0]
    for k, (a, b) in enumerate(pairs):
        A[k, ix[a]] += 1; A[k, ix[b]] -= 1
    A[-1] = 1.0
    s = np.linalg.solve(A.T @ A + 1e-9 * np.eye(n), A.T @ yv)
    return {c: float(s[ix[c]]) for c in chans}


def _tau(a, b):
    from scipy.stats import kendalltau
    ks = sorted(a); t = kendalltau([a[k] for k in ks], [b[k] for k in ks]).statistic
    return float(t) if t == t else 0.0


def k9_stats(rows, nsplit=200, nboot=200):
    if not rows:
        return nr("input missing")
    chans = sorted({k.split("->")[0] for r in rows for k in r["g"]} | {k.split("->")[1] for r in rows for k in r["g"]})
    blocks = {c: int("".join(ch for ch in c if ch.isdigit())[-2:] or 0) for c in chans}
    C = {}
    for r in rows:
        C.setdefault(r["cond"], []).append({tuple(k.split("->")): v for k, v in r["g"].items()})
    rk = lambda T: _rank({k: np.mean([t[k] for t in T if k in t]) for k in {k for t in T for k in t}}, chans, blocks)
    g = np.random.default_rng(KB.SEED_BOOT)

    def halves(T):
        p = g.permutation(len(T)); return [T[i] for i in p[:len(T) // 2]], [T[i] for i in p[len(T) // 2:]]
    c2 = C.get("c2", [])
    tau = [(_tau(rk(a), rk(b))) for a, b in (halves(c2) for _ in range(nsplit))] if len(c2) >= 2 else []
    res = dict(split_half_tau=val(np.median(tau)) if tau else coded("NaN:too_few_texts"))
    resolved = bool(tau) and np.median(tau) >= 0.6
    res["string"] = "hierarchy resolved" if resolved else "hierarchy not resolved"
    pairs = [("c1", "c2"), ("c3", "c2"), ("c4", "c5"), ("c4", "c2"), ("c1", "c3"), ("c5", "c2")]
    ps, Ds = {}, {}
    for a, b in pairs:
        if len(C.get(a, [])) < 2 or len(C.get(b, [])) < 2:
            continue
        def D(Ta, Tb, ns=20):
            v = []
            for _ in range(ns):
                a1, a2 = halves(Ta); b1, b2 = halves(Tb)
                v.append((_tau(rk(a1), rk(a2)) + _tau(rk(b1), rk(b2))) / 2 - (_tau(rk(a1), rk(b2)) + _tau(rk(b1), rk(a2))) / 2)
            return float(np.mean(v))
        est = D(C[a], C[b], nsplit // 10)
        dr = np.array([D([C[a][i] for i in g.integers(0, len(C[a]), len(C[a]))], [C[b][i] for i in g.integers(0, len(C[b]), len(C[b]))], 5)
                       for _ in range(nboot)])
        ps[f"{a}-{b}"] = pboot(dr); Ds[f"{a}-{b}"] = dict(D=est, ci=ci(dr))
    hf = holm_family(ps)
    for k in Ds:
        Ds[k]["string"] = ("not decidable (unstable ranking)" if not resolved else f"conditions reorder the channel hierarchy ({k})" if (hf[k]["reject"] and Ds[k]["D"] >= 0.2)
                           else f"hierarchy stable across {k}" if lt(Ds[k]["ci"], 0.2) else "undecided"); Ds[k]["holm"] = hf[k]
    res["pairs"] = Ds
    return res


def k10a_stats(rows):
    if not rows:
        return nr("input missing")
    content = {"ustar", "uT", "usea", "SJ14"}; out = {}
    sd = {}
    for r in rows:                                        # target SDs over texts and readout positions (effects in target SDs)
        for t, v in (r["eff"].get("base") or {}).items():
            sd.setdefault(t, []).extend(v)
    sd = {t: (np.std(v) or 1.0) for t, v in sd.items()}
    for r in rows:
        for key, e in r["eff"].items():
            if isinstance(e, dict) and key not in ("base",):
                r["eff"][key] = {t: x / sd.get(t, 1.0) for t, x in e.items()}
    for dose in ("delta1", "delta2"):
        kv, tk = [], []
        for r in rows:
            if r["dose"] != dose or r["src"] not in content:
                continue
            e = r["eff"]
            for t in e["kv2"]:
                kv.append(abs(e["kv2"][t])); tk.append(abs(e["tok2"].get(t, 0.0)))
        kv, tk = np.array(kv), np.array(tk)
        if len(kv) < 3:
            out[dose] = dict(string="not run (input missing)"); continue
        s = lambda ix: kv[ix].sum() / max(kv[ix].sum() + tk[ix].sum(), 1e-12)
        c = ci(boot_stat(len(kv), s))
        out[dose] = dict(s_KV=val(s(np.arange(len(kv)))), ci=c, string="parallel: KV route dominates at lag 2" if gt(c, 0.5) else
                         ("sequential: token route dominates at lag 2" if lt(c, 0.5) else "mixed"))
    d1, d2 = out.get("delta1", {}).get("string"), out.get("delta2", {}).get("string")
    out["final"] = d1 if d1 == d2 else ("dose-dependent route share (not a model property in 0.05-0.15 nats)" if d2 else f"{d1} (single dose)")
    sz = max([r["eff"].get("structural_zero_max", 0.0) for r in rows] or [0.0]); out["structural_zero_max"] = sz
    hz = {}
    for k in (1, 2, 4, 8):
        ps = {}
        for src in ("ustar", "uT", "usea", "SJ14", "N_H"):
            for route in ("kv", "tok"):
                T = lambda s_: np.sqrt(np.sum([np.mean([r["eff"][f"{route}{k}"].get(t, 0.0) for r in rows if r["src"] == s_ and r["dose"] == "delta1"]) ** 2
                                               for t in (rows[0]["eff"][f"{route}{k}"] if rows else {})]))
                rnd = [T(f"rand{j}") for j in range(20) if any(r["src"] == f"rand{j}" for r in rows)]
                if any(r["src"] == src for r in rows) and len(rnd) > 1:
                    ps[f"{src}:{route}"] = p_pi(T(src), rnd)
        hf = holm_family({k_: (v if fin(v) else 1.0) for k_, v in ps.items()})
        for key, h in hf.items():
            if h["reject"]:
                hz[key] = k
    out["horizons"] = {key: hz.get(key, 0) for key in [f"{s}:{r}" for s in ("ustar", "uT", "usea", "SJ14", "N_H") for r in ("kv", "tok")]}
    return out


def k10b_stats(d):
    if not d or not d.get("rows"):
        return nr("input missing")
    out, pq, po = {}, {}, {}
    for (l, dl), rs in by(d["rows"], "block", "dl").items():
        v = np.array([r["rho_qk"] for r in rs]); dr = boot_stat(len(v), lambda ix: v[ix].mean()); key = f"{l}:{dl}"
        pq[key] = pboot(dr, 0.5); po[key] = pboot(dr, 0.5, "less"); out[key] = dict(rho_QK=float(v.mean()), ci=ci(dr))
    hq, ho = holm_family(pq), holm_family(po)
    for k in out:
        out[k]["string"] = "mostly re-routing (QK)" if hq[k]["reject"] else ("mostly along routes (OV)" if ho[k]["reject"] else "mixed")
    out["G_K10"] = d.get("G_K10"); out["routing_map"] = "exploratory"
    return out


# =================================================================================================================================
# verdict: families, round-level p, K.7 composite
# =================================================================================================================================
def verdict(ctx):
    a1 = RC.load_json(KB.own(ctx, "k/ana1.json")); a2 = RC.load_json(KB.own(ctx, "k/ana2.json")); a3 = RC.load_json(KB.own(ctx, "k/ana3.json"))
    K3a = a1["K3a"]; gk2 = (K3a.get("in_domain") or {}).get("string", "undecided")
    arith = (K3a.get("arith") or {}).get("string", ""); har = (K3a.get("har") or {}).get("string", "")
    gk3 = "transfers to arithmetic errors" if "transfers to arithmetic" in arith else ("domain-specific H-set" if har.startswith("H_ar replicated") else "H not valid in arithmetic")
    gk4 = (a1["K4"].get("G_K4") if isinstance(a1["K4"].get("G_K4"), str) else "undecided"); gk4b = (a1["K4p"] or {}).get("G_K4b", "undecided")
    gk5 = a1["K5a"].get("G_K5", "not decidable"); gk14 = a1["K3g"]["G_K14"]; gk6 = a1.get("G_K6", "verifier unreliable")
    PH = gk2 == "H-neurons replicated (in-domain detection)"; PL = gk4 == "chain lens reads intermediates"
    # Q9 (attractor family): K6e TOK carry, K6e TOK+KV carry, K6e bistability, K8b ACH carry, K8b TQA carry, K8b ACH bistability
    e6 = k6e_strings(a3["K6e"]) if isinstance(a3["K6e"], dict) and "carry" in a3["K6e"] else {}
    b6 = (a3["K6e"] or {}).get("bist") or {}; b8 = (a3["K8b"] or {}).get("bist") or {}
    def bist_p(b):
        m = b.get("mix") or {}; return m.get("p") if fin(m.get("p")) else 1.0
    def k8_p(name):
        r = (a3["K8b"] or {}).get(name) or {}; c = r.get("carry") or {}
        if not c.get("defined"):
            return 1.0, None
        ppi = p_pi(c["logK"], r["logK_R"]) if (len(r.get("logK_R", [])) >= 10 and fin(c.get("logK"))) else 1.0
        return p_iut(c["p_le1"], ppi), (c, r, ppi)
    q9 = {"K6e TOK carry": e6.get("TOK", {}).get("p", 1.0), "K6e TOK+KV carry": e6.get("TOK+KV", {}).get("p", 1.0),
          "K6e bistability": bist_p(b6), "K8b ACH carry": k8_p("ACH")[0], "K8b TQA carry": k8_p("TQA")[0], "K8b ACH bistability": bist_p(b8)}
    hq9 = holm_family(q9); strings = {}
    for k, h in hq9.items():
        if "carry" in k and k.startswith("K6e"):
            route = "TOK" if "TOK carry" == k[4:] else "TOK+KV"; r = e6.get(route, {})
            c = r.get("_c") or {}
            strings[k] = (r.get("string") if "string" in r else f"H-neuron attractor in the chain lens (self-sustaining carry, {route})" if h["reject"] else
                          "generic persistence in the chain lens (not H-specific)" if (c and c.get("p_le1", 1) <= 0.05 / 6 and (r.get("p_pi") if fin(r.get("p_pi")) else 1) > 0.05) else
                          "dissipative in the chain lens" if (c and lt(c["K_ci"], 1.0)) else
                          "H-specific non-decaying persistence in the chain lens" if (c and fin(c.get("logK")) and inside(dict(lo=math.log(max(c['K_ci']['lo'], 1e-9)) if fin(c['K_ci']['lo']) and c['K_ci']['lo'] > 0 else coded('NaN:x'), hi=math.log(c['K_ci']['hi']) if fin(c['K_ci']['hi']) and c['K_ci']['hi'] > 0 else coded('NaN:x')), math.log(1.25)) and fin(r.get("p_pi")) and r["p_pi"] <= 0.025) else "undecided")
        elif k.startswith("K8b") and "carry" in k:
            arm = k.split()[1]; p, info = k8_p(arm)
            if info is None:
                strings[k] = "K_H not defined"
            else:
                c, r, ppi = info
                strings[k] = (f"self-sustaining H ({arm})" if h["reject"] else f"generic persistence (not H-specific) ({arm})" if (c["p_le1"] <= 0.05 / 6 and ppi > 0.05) else
                              f"dissipative H ({arm})" if lt(c["K_ci"], 1.0) else "undecided")
        else:
            b = b6 if k.startswith("K6e") else b8; m = b.get("mix") or {}
            cond = h["reject"] and fin(m.get("Delta")) and m["Delta"] >= 3 and m.get("s1", 9) <= 1.5 and 0.1 <= m.get("pi", 0) <= 0.9 and b.get("locked_with_relaxed", 0) >= 0.3
            strings[k] = (("H-neuron attractor in the chain lens (bistable within items)" if k.startswith("K6e") else "within-item bistability (sessions)") if cond else
                          (("no within-item bistability (lens)" if k.startswith("K6e") else "no within-item bistability (sessions)") if lt((b.get("tail") or {}).get("ci", {"hi": coded("NaN:x"), "lo": coded("NaN:x")}), 0.10) else "undecided"))
    # Q11 (lens-frame causal) and the A+ members
    q11 = {k: (a3["K6e"].get(f"causal_{k}") or {}).get("p_iut", 1.0) for k in ("PUSH_H", "CLAMP_H")} if isinstance(a3["K6e"], dict) else {}
    hq11 = holm_family(q11) if q11 else {}
    s11 = {"push": "H push raises lens confabulation beyond random sets" if hq11.get("PUSH_H", {}).get("reject") else "undecided",
           "clamp": "H clamp lowers lens confabulation beyond random clamps" if hq11.get("CLAMP_H", {}).get("reject") else "undecided"}
    k6b = a2["K6b"] if isinstance(a2["K6b"], dict) and "i" in a2["K6b"] else {}
    k6c = a2["K6c"] if isinstance(a2["K6c"], dict) and "(i)" in a2["K6c"] else {}
    k6h = a2["K6h"] if isinstance(a2["K6h"], dict) and "Q12" in a2["K6h"] else {}
    q12 = (k6h.get("Q12") or {}); q12s = ("H-activity predicts lens confabulation beyond chain length (QA)" if q12.get("p_iut", 1.0) <= 0.05 else q12.get("string", "undecided"))
    Aplus = {"K6b(i)": ((k6b.get("i") or {}).get("string", "").startswith("H-activity"), (k6b.get("i") or {}).get("p_iut", 1.0), (k6b.get("i") or {}).get("p_iut_pi", 1.0)),
             "K6b(ii)": ((k6b.get("ii") or {}).get("string", "").startswith("H-activity"), (k6b.get("ii") or {}).get("p_iut", 1.0), (k6b.get("ii") or {}).get("p_iut_pi", 1.0)),
             "K6c(i)": ("reduces" in (k6c.get("(i)") or {}).get("string", ""), (k6c.get("(i)") or {}).get("p_iut", 1.0), (k6c.get("(i)") or {}).get("p_iut_pi", 1.0)),
             "K6c(ii)": ("reduces" in (k6c.get("(ii)") or {}).get("string", ""), (k6c.get("(ii)") or {}).get("p_iut", 1.0), (k6c.get("(ii)") or {}).get("p_iut_pi", 1.0)),
             "K6e-c push": (s11["push"].startswith("H push"), q11.get("PUSH_H", 1.0), q11.get("PUSH_H", 1.0)),
             "K6e-c clamp": (s11["clamp"].startswith("H clamp"), q11.get("CLAMP_H", 1.0), q11.get("CLAMP_H", 1.0)),
             "K6h": (q12s.startswith("H-activity"), q12.get("p_iut", 1.0), q12.get("p_iut_pi", 1.0))}
    A = "A+" if any(v[0] for v in Aplus.values()) else None
    k6bi = (k6b.get("i") or {}).get("string", ""); k6ci = (k6c.get("(i)") or {}).get("string", "")
    if not A:
        if (k6bi.startswith("length artifact") or q12s.startswith("length artifact")) and "no specific causal role" in k6ci:
            A = "A0"
        elif k6bi.startswith("no association") and "no specific causal role" in k6ci:
            A = "A-"
        else:
            A = "A?"
    Bplus = [k for k, s in strings.items() if ("attractor" in s or "self-sustaining" in s or "within-item bistability (sessions)" == s)]
    Bg = any("generic persistence" in s for s in strings.values()); Bp = any(k in ("K6e TOK carry", "K8b ACH carry") and "non-decaying" in strings[k] for k in strings)
    defined = [k for k in ("K6e TOK carry", "K6e TOK+KV carry", "K8b ACH carry", "K8b TQA carry") if "not defined" not in strings.get(k, "not defined") and "not run" not in strings.get(k, "")]
    Bminus = bool(defined) and all("dissipative" in strings[k] for k in defined) and strings["K6e bistability"].startswith("no within") and strings["K8b ACH bistability"].startswith("no within")
    Bl = "B+" if Bplus else ("Bg" if Bg else ("Bp" if Bp else ("B-" if Bminus else "B?")))
    k6g = a2.get("K6g") or {}; k8a = a3.get("K8a") or {}
    Cp = (k6g.get("p", 1.0) <= 0.05) or any((k8a.get(o) or {}).get("summary") == "content effect beyond length" for o in ("acc", "H_gen", "H_tf"))
    frames = ", ".join(sorted({"lens" if k.startswith("K6e") else "model" for k in Bplus})) or "-"
    blabel = ("generic persistence (not H-specific)" if Bg else "H-specific non-decaying persistence" if Bp else "dissipative" if Bminus else "undecided")
    csfx = "; content drives break-off (in-context conditioning)" if Cp else ""
    k3a_neg = gk2 == "H-neurons not replicated (detection at random-set level)"
    if k3a_neg:
        comp = "premise fails: H-neurons not replicated in Qwen3-1.7B"
    elif not PH:
        comp = "undecided (premise not established: H-neuron replication undecided)"
    elif PL and A == "A+" and Bplus:
        comp = f"H-CHAIN supported: H-neuron attractor ({frames}) and H-linked chain-lens break-off beyond length"
    elif Bplus:
        comp = (f"H-neuron attractor ({frames}) without H-linked lens break-off ({'lens not tested' if not PL else 'no lens-side link'})"
                if not (frames == "model" and not PL) else "model-side H attractor (lens not tested)")
    elif A == "A+":
        comp = f"H-linked lens break-off beyond length, but no H-neuron attractor ({blabel}){csfx}"
    elif A == "A0" and Bminus:
        comp = "length artifact: the H-hallucination association in the chain lens is carried by chain length, position and uncertainty, and H perturbations decay" + csfx
    elif A == "A0":
        comp = f"length artifact (association); H dynamics {blabel}"
    elif A == "A-":
        comp = f"no H association with lens break-off ({blabel}){csfx}"
    else:
        comp = f"undecided (open components: {', '.join(x for x, ok in (('P-L', PL), ('A', A in ('A+', 'A0', 'A-')), ('B', Bl != 'B?')) if not ok)})"
    k6f = a2.get("K6f") or {}
    q13 = holm_family({"K6f": k6f.get("p", 1.0) if isinstance(k6f, dict) else 1.0, "K6g": k6g.get("p", 1.0)})
    s13 = {"K6f": ("context-length effect on lens break-off" if q13["K6f"]["reject"] else "no context-length effect" if k6f.get("_eq") else
                   k6f.get("string") or "undecided"),
           "K6g": ("content effect (in-lens snowballing)" if q13["K6g"]["reject"] else "no content effect" if k6g.get("_eq") else "undecided")}
    q10 = holm_family({"K6d": (a1.get("K6d") or {}).get("p", 1.0) if isinstance(a1.get("K6d"), dict) else 1.0, "K6h": k6h.get("p_own", 1.0)})
    s10 = {"K6d": "lens reads the model's own wrong intermediates" if q10["K6d"]["reject"] else "not shown",
           "K6h": "lens reads the model's own wrong answer (QA)" if (q10["K6h"]["reject"] and k6h.get("G_K23", "").startswith("lens reads")) else "not shown"}
    Cp = Cp or s13["K6g"].startswith("content effect")
    k8c = a3.get("K8c") or {}; M = {"H-activity": "M+", "length artifact": "M0", "no association": "M-"}
    Ml = next((v for k, v in M.items() if (k8c.get("string") or "").startswith(k)), "M?")
    # round-level families (main §2.4); CF-K-link with K's rank form (as frozen) and the PI form (proposed main change, K.12)
    cf_link = holm_family({k: v[1] for k, v in Aplus.items()}); cf_link_pi = holm_family({k: v[2] for k, v in Aplus.items()})
    cf_attr = holm_family(q9)
    sfx = [f"P-L: {gk4}" + (f" ({'lens fails to read present intermediates' if gk4b == 'intermediate present' else 'nothing to read'})" if not PL and gk4b != 'undecided' else ""),
           f"G-K5: {gk5}" if gk5 == "self-rollout calibrated" else "lens substitute not calibrated", gk14, f"G-K3: {gk3}", f"model side: {Ml}",
           a1["K5a"].get("trunk"), f"round level: A+ {'confirmatory' if any(v['reject'] for v in cf_link.values()) else 'exploratory'}; "
           f"B+ {'confirmatory' if any(v['reject'] for v in cf_attr.values()) else 'exploratory'}"]
    jv = _js(ctx, "r1b_R_S5", "jlens_validity.json")
    if jv:
        bad = [b for b, v in (jv.get("blocks") or {}).items() if not v.get("valid", True)]
        if bad:
            sfx.append("J-lens not valid at " + ", ".join(map(str, bad)) + " (R5.1)")
    out = dict(composite=dict(string=comp, suffixes=[x for x in sfx if x], components=dict(P_H=PH, P_L=PL, A=A, B=Bl, C="C+" if Cp else "C-", M=Ml)),
               gates={"G-K2": gk2, "G-K3": gk3, "G-K4": gk4, "G-K4b": gk4b, "G-K5": gk5, "G-K6": gk6, "G-K14": gk14,
                      "G-K7": a1.get("G_K7"), "G-K12": (a1["K3d"] or {}).get("G_K12") if isinstance(a1["K3d"], dict) else None,
                      "G-K13": k8c.get("G_K13"), "G-K23": k6h.get("G_K23"), "G-K-J": (a1["K4j"] or {}).get("string")},
               families=dict(Q9={k: dict(string=strings[k], **hq9[k]) for k in hq9}, Q11={k: dict(string=s11["push" if k == "PUSH_H" else "clamp"], **hq11[k]) for k in hq11},
                             A_plus={k: dict(reached=v[0], p_rank_form=v[1], p_pi_form=v[2]) for k, v in Aplus.items()}),
               round_level=dict(CF_K_attr={k: v for k, v in cf_attr.items()}, CF_K_link_rank=cf_link, CF_K_link_pi=cf_link_pi,
                                CF_HERR_K8c=(k8c.get("rule") or {}).get("p_iut", 1.0),
                                note="CF-K-link with rank p (floor 1/40 = 0.025) cannot pass Holm over 7 (0.0071); the PI form is the proposed main "
                                     "change (K.12, finding stats-1)"),
               strings=dict(K3a=gk2, K3b=(a1["K3b"] or {}).get("string"), K3c=(a1["K3c"] or {}).get("string"), K3d=(a1["K3d"] or {}).get("reading") if isinstance(a1["K3d"], dict) else None,
                            K6a={l: v.get("string") for l, v in (a1["K6a"] or {}).items() if isinstance(v, dict)}, K6b_i=k6bi, K6b_ii=(k6b.get("ii") or {}).get("string"),
                            K6b_iii=(k6b.get("iii") or {}).get("string"), K6c={k: (k6c.get(k) or {}).get("string") for k in ("(i)", "(ii)")}, K6h_Q12=q12s,
                            K8a={o: (k8a.get(o) or {}).get("summary") for o in ("acc", "H_gen", "H_tf")}, K8c=k8c.get("string"),
                            K9=(a3.get("K9") or {}).get("string"), K10a=(a3.get("K10a") or {}).get("final"), K5a=gk5, Q6=((a1["K5a"].get("Q6") or {}).get("string"))),
               families_more=dict(Q13={k: dict(string=s13[k], **q13[k]) for k in q13}, Q10={k: dict(string=s10[k], **q10[k]) for k in q10}),
               labels=[l for l in (gk14 if "not" in gk14 else None, None if PH else "H-set not validated", "verifier unreliable" if gk6 != "verifier reliable" else None) if l],
               reported_next_to_composite=dict(K5a=gk5, RK2="see k_rk2 (r1b_K_S7)", RK1="see k_rk1 (r1b_K_S6)"))
    write(ctx, "k_verdict.json", out, config=dict(boot=B_, level=RC.LEVEL), labels=out["labels"])


def rk2_rule(rows, gk2_2b):
    """RK-2: 'removing H writes from the read state reduces native-lens confabulation specifically' if the 90 % upper bound of the
    item-level Delta < 0 and p_rank <= 0.05; else 'no specific role of H writes in native-lens confabulation'."""
    base = {(r["item"], r["block"]): r["q"] for r in rows if r["cond"] == "none" and r["q"] is not None}
    d = lambda c: {k: np.mean([r["q"] - base[(r["item"], r["block"])] for r in rows if r["cond"] == c and r["item"] == k[0] and (r["item"], r["block"]) in base and r["q"] is not None])
                   for k in {(r["item"],) for r in rows if r["cond"] == c}}
    dH = {k: v for k, v in d("H").items() if fin(v)}; v = np.array(list(dH.values()))
    if len(v) < 3:
        return dict(string="no specific role of H writes in native-lens confabulation", code=coded("NaN:too_few_items"))
    c = ci(boot_stat(len(v), lambda ix: v[ix].mean())); dR = [np.nanmean(list(d(c_).values())) for c_ in {r["cond"] for r in rows if r["cond"].startswith("RS")}]
    pr = (1 + sum(x <= v.mean() for x in dR if fin(x))) / (len(dR) + 1)
    s = ("removing H writes from the read state reduces native-lens confabulation specifically" if (lt(c) and pr <= 0.05) else
         "no specific role of H writes in native-lens confabulation")
    return dict(string=s, delta=float(v.mean()), ci=c, p_rank=pr, labels=[] if (gk2_2b or "").startswith("H-neurons replicated") else ["H not validated on Qwen3.5-2B"])
