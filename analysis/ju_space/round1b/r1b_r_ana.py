"""r1b_R_ana (pass 1) and r1b_R_ana2 (pass 2) of part R: every R statistic and verdict string (R2.3-R2.7, R3.1-R3.4, R4.1-R4.3, R5
re-check, R6, R7, R8.1 / RG15), RG7, RG8a, RG8b, RG12, the F+ / F- Holm families (= CF-R+ / CF-R-) and the R9 report check (RG17).
Model-free (GPU only for the bootstraps). Per block and measurement the components are stored, the strings are assembled in a_verdicts
(the verdict precedence of R2.4 is the ordered list there). Changes adopted from the review hints are marked in the part file (R.9)."""
import glob, math, os
import numpy as np
import r1b_core as K
import r1b_r_lib as R
import r1b_r_stats as S

FOR = ("sea_v", "sports_v", "tone_v", "u_neg")
NAMES = ["T_v", "T_1b", "T_ps", "sea_v", "sports_v", "tone_v", "sea_raw"]


def T():
    return R.torch()


def dev():
    return "cuda" if T().cuda.is_available() else "cpu"


def W_of(NP, B=2000):
    return T().as_tensor(K.prompt_bootstrap(NP, B=B, seed=1001), dtype=T().float64).to(dev())


def boot_t(ctx):
    p = ctx.input("r1b_R_S0", "r0_table.json", required=False)
    return bool((K.load_json(p) or {}).get("bootstrap_t")) if p else False


def strip(d):
    return {k: v for k, v in d.items() if not k.endswith("_draws")}


def pos(x, thr, alt="greater"):
    return S.p_one_sided(x, thr, alt) if False else K.p_one_sided(x, thr, alt)


def load(ctx):
    """Index of the copied R sessions and the R9 report check on their outputs (SG6 / RG17: provenance present, no raw NaN)."""
    probs, n = {}, 0
    for s in sorted(glob.glob(os.path.join(ctx.inp, "r1b_R_*"))):
        for f in glob.glob(os.path.join(s, "**", "*"), recursive=True):
            if os.path.isfile(f) and not f.endswith(".prov.json") and not f.endswith(".orig") and "/cache/" not in f and "/r1b_replay" not in f:
                n += 1; e = K.check_output(f)
                if e:
                    probs.setdefault(os.path.basename(s), []).append(e)
    cc = {}
    for b in R.BLOCKS:
        c = K.load_json(R.find(ctx, f"r2/b{b}/checks.json") or "", {}) or {}; cc[str(b)] = dict(ok=bool(c.get("ok")), checks={k: c.get(k) for k in ("CC1", "CC2", "CC3")})
    c4 = K.load_json(R.find(ctx, "r2/b14/cc4.json") or "", {}) or {}; cc["14"]["CC4"] = c4.get("CC4"); cc["14"]["ok"] = cc["14"]["ok"] and bool((c4.get("CC4") or {}).get("ok"))
    ctx.write_json("ana/index.json", dict(files_checked=n, RG17=dict(incomplete=sorted(probs), problems={k: v[:5] for k, v in probs.items()}), RG5=cc,
                                          bootstrap_t=boot_t(ctx)), config=dict())


# ------------------------------------------------------------------------------------------------------------------ R2 per block
def _ev(ctx, b, arm, L, n):
    p = R.find(ctx, f"r2/b{b}/{arm}/{L}/ev{n}.pt"); return R.load_pt(p) if p else None


def _row(ev, NP):
    return S.Row(R.delta_of(ev), ev["ok"], ev["prompt_of"], NP, dev())


def _axes(ctx, b, W, extra=None):
    ax = R.ax1b(ctx, b); AS, a1, aW = S.axes_for(ax, NAMES + list((extra or {}).keys()), W, dev(), extra)
    un = S.unit(ax["u_neg"].double()).to(dev()); B = W.shape[0]; one = T().ones(1, dtype=T().float64, device=dev())
    a1["u_neg"] = dict(u=un[None], aA=un[None], aB=un[None], kappa=one); aW["u_neg"] = dict(u=un.expand(B, -1), aA=un.expand(B, -1), aB=un.expand(B, -1), kappa=one.expand(B))
    return ax, a1, aW


def z_beh(st, d, name):
    k = (st.get("k5") or {}).get(name)
    if not k:
        return None
    return float(k["wU_perp"].double() @ S.unit(T().as_tensor(d).double().cpu())) / float(k["sigma"])


def r2_block(ctx):
    """R2.2-R2.7 components at block b: STAT-1b rows of every loop (DN, FX, NP) on the verdict, other-RSI and foreign axes; E, A_bar,
    A_bar_abs, R_4^cf, S_4^cf, D4, K5, debiased style cosines, PC components P1-P5 (two-sided RND contrast), the B-Q0 recipe on the DN
    arm (u*_1b, leave-one-out, u°_b) and the R3.2 NP rule; draws to ana/r2_b<b>_draws.pt."""
    t = T(); b = int(ctx.step.rsplit("_b", 1)[1]); P = R.p256(ctx); NP = len(P["q"]); H = S.halvings(NP); W = W_of(NP); bt = boot_t(ctx); fold = P["fold"]
    st1 = R.r1_state(ctx, b); us = S.unit(R.ustar(ctx, b)[0]); q0 = K.load_json(R.r2_file(ctx, f"r2_q2/theta/b{b}_q0.json"), {}) or {}
    m2 = float(((q0.get("m_star2") or {}).get("pt")) or 1.0); other = "T_ps" if b == 6 else "T_1b"
    ax, a1, aW = _axes(ctx, b, W); rows, draws, res = {}, {}, dict(block=b, rows={})

    def sr(key, row, nm, two=False):
        s = S.stat_row(row, a1[nm], aW[nm], fold, H, W, two_sided=two, boot_t=bt, keep_draws=True); draws[key + (nm,)] = s.pop("A_draws"); s.pop("rel_draws", None)
        res["rows"]["|".join(map(str, key + (nm,)))] = s; return s
    spec = {("DN", "RSI_a"): ["T_v", other], ("DN", "RSI_b"): ["T_v", other], ("DN", "SEA_t"): ["sea_v"], ("DN", "SPO_t"): ["sports_v", "T_v"],
            ("DN", "SEA_raw"): ["sea_raw"], ("DN", "RND_1"): ["T_v", "sea_v", "sports_v"], ("DN", "RND_2"): ["T_v", "sea_v", "sports_v"],
            ("FX", "RSI_a"): ["T_v", other], ("FX", "SEA_raw"): ["sea_raw"], ("FX", "RND_1"): ["T_v"], ("NP", "RSI_a"): ["T_v"], ("NP", "SEA_t"): ["sea_v"],
            ("NP", "RND_1"): ["T_v"]}
    evs = {}
    for (arm, L), nms in spec.items():
        for n in range(6 if (arm, L) in (("DN", "RSI_a"), ("DN", "RSI_b")) else 4):
            e = _ev(ctx, b, arm, L, n)
            if e is None:
                continue
            evs[(arm, L, n)] = e; rows[(arm, L, n)] = r = _row(e, NP)
            for nm in nms:
                sr((arm, L, n), r, nm, two=L.startswith("RND"))
        if (arm, L, 3) in rows and L.startswith("RSI"):
            for nm in FOR:
                sr((arm, L, 3), rows[(arm, L, 3)], nm)
    g = lambda arm, L, n, nm: res["rows"].get(f"{arm}|{L}|{n}|{nm}")
    dr = lambda arm, L, n, nm: draws.get((arm, L, n, nm))
    comp = {}
    for read in ("T_v", other):
        c = {}
        A = {L: [dr("DN", L, n, read) for n in range(4)] for L in ("RSI_a", "RSI_b")}
        if any(x is None for L in A for x in A[L]):
            comp[read] = dict(status=K.coded("NA:not_run:input missing")); continue
        Ab = (A["RSI_a"][2] + A["RSI_a"][3] + A["RSI_b"][2] + A["RSI_b"][3]) / 4; Aba = (np.abs(A["RSI_a"][2]) + np.abs(A["RSI_a"][3]) + np.abs(A["RSI_b"][2]) + np.abs(A["RSI_b"][3])) / 4
        den = A["RSI_a"][1] + A["RSI_b"][1]; den_iv = S.iv(den); e_def = isinstance(den_iv[0], float) and (den_iv[0] > 0 or den_iv[1] < 0)
        E = (A["RSI_a"][3] + A["RSI_b"][3]) / np.where(np.abs(den) > 1e-12, den, np.nan)
        pt = lambda L, n: g("DN", L, n, read)
        c["A"] = {L: [pt(L, n)["A"] for n in range(4)] for L in A}; c["A_bar"] = dict(pt=float(np.mean([pt(L, n)["A"] for L in A for n in (2, 3)])), iv=S.iv(Ab))
        c["A_bar_abs"] = dict(pt=float(np.mean([abs(pt(L, n)["A"]) for L in A for n in (2, 3)])), iv=S.iv(Aba))
        c["E"] = dict(pt=K.val((pt("RSI_a", 3)["A"] + pt("RSI_b", 3)["A"]) / (pt("RSI_a", 1)["A"] + pt("RSI_b", 1)["A"])), iv=S.iv(E)) if e_def else \
            dict(pt=K.coded("NaN:denominator_interval_includes_0"), iv=[K.coded("NaN:denominator_interval_includes_0")] * 2)
        c["MANIP_1b"] = all(pt(L, 0)["p_s"] <= 0.01 and isinstance(pt(L, 0)["A_lo"], float) and pt(L, 0)["A_lo"] >= 0.10 for L in A)
        c["rel_ok"] = all(np.mean([pt(L, n)["rel"] for n in (1, 2, 3)]) >= 0.5 for L in A)
        r4 = S.cf_cos(rows[("DN", "RSI_a", 3)], [rows[("DN", "RSI_b", 3)]], H); c["R4cf"] = r4["cf"]
        fa = lambda L: max(abs(g("DN", L, 3, nm)["A"]) for nm in FOR)
        c["D4"] = {L: abs(pt(L, 3)["A"]) - fa(L) for L in A}
        spo = g("DN", "SPO_t", 3, "T_v"); c["SPO_absA4_hi"] = spo["absA_hi"] if spo else None
        rnd = [(L, n) for L in ("RND_1", "RND_2") for n in (2, 3) if g("DN", L, n, "T_v")]
        c["RND_attracted"] = [f"{L}:{n + 1}" for L, n in rnd if g("DN", L, n, "T_v")["p_s"] <= 0.01 and isinstance(g("DN", L, n, "T_v")["absA_lo"], float)
                              and g("DN", L, n, "T_v")["absA_lo"] >= 0.25]
        c["p_pro_234"] = {L: [pt(L, n)["p_s"] for n in (1, 2, 3)] for L in A}
        c["p_two_234"] = {L: [S.k1s(S.bvals(rows[("DN", L, n)], a1[read], fold), two_sided=True, boot_t=bt)["p_s"] for n in (1, 2, 3)] for L in A}
        c["alternating"] = all(np.sign(c["A"][L][n]) == -np.sign(c["A"][L][n + 1]) for L in A for n in range(3))
        c["A1_hi"] = {L: pt(L, 0)["A_hi"] for L in A}
        c["p_comp"] = dict(Abar25=pos(Ab, 0.25), Abar50=pos(Ab, 0.50), Abar_abs_lt25=pos(Aba, 0.25, "less"), Abar_abs25=pos(Aba, 0.25),
                           A1_10={L: pos(A[L][0], 0.10) for L in A}, A1_lt10={L: pos(A[L][0], 0.10, "less") for L in A},
                           E50=pos(E, 0.5) if e_def else 1.0, SPO=pos(np.abs(dr("DN", "SPO_t", 3, "T_v")), 0.25, "less") if spo else 1.0)
        comp[read] = c
    res["components"] = comp
    d4 = {L: rows[("DN", L, 3)].d().cpu() for L in R.LOOPS_1B if ("DN", L, 3) in rows}
    res["deb_cos_ustar"] = {L: S.deb_fixed(rows[("DN", L, 3)], us, g("DN", L, 3, NAMES_FOR(L))["rel"], m2) for L in d4}
    res["style_attracted"] = all(res["deb_cos_ustar"].get(L, 0) >= 0.5 for L in ("RSI_a", "RSI_b"))
    k5 = {L: dict(rsi=z_beh(st1, d4[L], "rsi"), sports=z_beh(st1, d4[L], "sports"), literature=z_beh(st1, d4[L], "literature")) for L in ("RSI_a", "RSI_b") if L in d4}
    zs = z_beh(st1, ax["axes"]["T_v"], "rsi"); res["K5"] = dict(z_start=zs, z=k5)
    if zs is None or None in [v["rsi"] for v in k5.values()]:
        res["K5"].update(ok_likely=True, ok_osc=True, label="K5 missing")
    else:
        spec_ok = all(abs(v["sports"]) < 2.58 and abs(v["literature"]) < 2.58 for v in k5.values())
        sgn = {L: np.sign(g("DN", L, 3, "T_v")["A"]) for L in k5}
        res["K5"].update(ok_likely=zs < 3 or (np.mean([v["rsi"] for v in k5.values()]) >= 1.64 and spec_ok),
                         ok_osc=zs < 3 or (np.mean([v["rsi"] * sgn[L] for L, v in k5.items()]) >= 1.64 and spec_ok))
    # PC (R2.3) on SEA_t / u_sea^v, fallback SPO_t / u_sports^v (RG7)
    res["PC"] = {}
    for pcL, nm, words in (("SEA_t", "sea_v", "W_sea"), ("SPO_t", "sports_v", "W_sports")):
        if not all(("DN", pcL, n) in rows for n in range(4)):
            continue
        rr = [g("DN", pcL, n, nm) for n in (1, 2, 3)]; s4 = S.s4cf(rows[("DN", pcL, 2)], rows[("DN", pcL, 3)], H)["cf"]
        P1 = all(r["p_s"] <= 0.01 and isinstance(r["A_lo"], float) and r["A_lo"] >= 0.10 for r in rr); P2 = all(r["rel"] >= 0.5 for r in rr)
        P3 = isinstance(s4, float) and s4 >= 0.5
        lex = _lex(evs, pcL, words, NP); P4p = S.k1s(lex, flips=10000)["p_s"] if False else _flip_p(lex)
        rnd4 = [np.abs(dr("DN", L, 3, nm)) for L in ("RND_1", "RND_2") if dr("DN", L, 3, nm) is not None]
        P5d = dr("DN", pcL, 3, nm) - np.max(np.stack(rnd4), 0) if rnd4 else None; P5iv = S.iv(P5d) if P5d is not None else [None, None]
        P5 = isinstance(P5iv[0], float) and P5iv[0] > 0
        lexr = [np.abs(_lex(evs, L, words, NP)) for L in ("RND_1", "RND_2") if ("DN", L, 1) in rows]
        P4x = None
        if lexr:
            dd = [np.mean(lex[idx] - np.maximum(*[x[idx] for x in lexr]) if len(lexr) > 1 else lex[idx] - lexr[0][idx]) for idx in K.boot_idx(len(lex), 2000, 2003)]
            P4x = S.iv(dd)
        res["PC"][pcL] = dict(P1=P1, P2=P2, P3=P3, P4=P4p <= 0.01, P4_p=P4p, P5=P5, P5_iv=P5iv, S4cf=s4, pass_=P1 and P2 and P3 and P4p <= 0.01 and P5,
                              GO1_1b=g("DN", pcL, 0, nm)["rel"] >= 0.5 and g("DN", pcL, 0, nm)["p_s"] <= 0.01 and (g("DN", pcL, 0, nm)["A_lo"] or 0) >= 0.10,
                              absA_hi_34=[g("DN", pcL, n, nm)["absA_hi"] for n in (2, 3)], P4_rnd_contrast_iv=P4x,
                              p_comp=dict(p_s=[r["p_s"] for r in rr], A_lo10=[pos(dr("DN", pcL, n, nm), 0.10) for n in (1, 2, 3)], P4=P4p,
                                          P5=pos(P5d, 0.0) if P5d is not None else 1.0))
    if ("DN", "SEA_raw", 3) in rows:
        rr = [g("DN", "SEA_raw", n, "sea_raw") for n in (1, 2, 3)]
        res["PC_round1_def_on_SEA_raw"] = all(r["p_s"] <= 0.01 and r["c"] > 0 and r["rel"] >= 0.5 for r in rr)
    # R2.7: B-Q0 recipe on the DN arm, leave-one-out, RSI-free mode u°_b; R3.2
    dn = [L for L in R.LOOPS_1B if ("DN", L, 3) in rows]; cf = {L: S.cf_cos(rows[("DN", L, 3)], [rows[("DN", "RSI_a", 3)]], H)["cf"] for L in dn}
    M = [L for L in dn if L == "RSI_a" or (isinstance(cf[L], float) and abs(cf[L]) >= 0.5)]; sg = {L: 1.0 if (L == "RSI_a" or cf[L] >= 0) else -1.0 for L in M}
    u1b = S.unit(sum(sg[L] * d4[L] for L in M)); loo = {}
    for L in ("RND_1", "RND_2"):
        if L in dn:
            rest = [x for x in M if x != L]
            loo[L] = S.cf_cos(rows[("DN", L, 3)], [rows[("DN", x, 3)] for x in rest], H, [sg[x] for x in rest])["cf"] if rest else K.coded("NaN:empty_mode_set")
    nr = [L for L in ("SEA_t", "SPO_t", "RND_1", "RND_2") if L in dn]; uo = None
    pair = {(x, y): S.cf_cos(rows[("DN", x, 3)], [rows[("DN", y, 3)]], H)["cf"] for x in nr for y in nr if x < y}
    mem = sorted({z for (x, y), v in pair.items() if isinstance(v, float) and abs(v) >= 0.5 for z in (x, y)})
    if len(mem) >= 2:
        sgo = {L: (1.0 if S.cos(d4[L], d4[mem[0]]) >= 0 else -1.0) for L in mem}; uo = S.unit(sum(sgo[L] * d4[L] for L in mem))
    res["R27"] = dict(mode_set=M, cf_cos=cf, cos_u1b_ustar=S.cos(u1b, us), loo=loo, cos_Tv_u1b=S.cos(ax["axes"]["T_v"], u1b),
                      cos_seav_u1b=S.cos(ax["axes"]["sea_v"], u1b), u_o_members=mem, cos_uo_ustar=S.cos(uo, us) if uo is not None else K.coded("NA:not_run:no RSI-free generic mode"),
                      label=("same attractor as round 2" if abs(S.cos(u1b, us)) >= 0.7 else None),
                      generic=("generic attractor" if any(isinstance(v, float) and abs(v) >= 0.5 for v in loo.values()) else "no common mode"))
    res["robust"] = None
    if uo is not None and abs(S.cos(uo, us)) < 0.7:
        _, a1o, aWo = _axes(ctx, b, W, extra={"T_vo": ("T_v", [uo])}); c = {}
        for L in ("RSI_a", "RSI_b"):
            c[L] = [S.stat_row(rows[("DN", L, n)], a1o["T_vo"], aWo["T_vo"], fold, H, W, boot_t=bt, keep_draws=True) for n in range(4)]
        Abo = np.mean([c[L][n]["A_draws"] for L in c for n in (2, 3)], 0); Aao = np.mean([np.abs(c[L][n]["A_draws"]) for L in c for n in (2, 3)], 0)
        res["robust"] = dict(A_bar_iv=S.iv(Abo), A_bar_abs_iv=S.iv(Aao), MANIP_1b=all(c[L][0]["p_s"] <= 0.01 and c[L][0]["A_lo"] >= 0.10 for L in c),
                             p_pro_234={L: [c[L][n]["p_s"] for n in (1, 2, 3)] for L in c}, A1_hi={L: c[L][0]["A_hi"] for L in c})
    npr = {}
    for arm in ("DN", "NP"):
        if (arm, "RND_1", 3) in rows:
            r = rows[(arm, "RND_1", 3)]; rl = g(arm, "RND_1", 3, "T_v")["rel"]; npr[arm] = S.deb_fixed(r, us, rl, m2)
            if arm == "NP":
                dW = r.d(W); relW = r.rel(H, W)[0]; db = ((dW @ us.to(dW.device)).abs() / dW.norm(dim=1) / t.sqrt(relW * m2)).cpu().numpy(); npr["NP_iv"] = S.iv(db)
    tang = [float(e["abs_cos"]["mean"]) for (arm, L, n), e in evs.items() if arm == "NP" and isinstance(e.get("abs_cos"), dict) and isinstance(e["abs_cos"].get("mean"), float)]
    res["R32"] = dict(deb=npr, mean_abs_cos=K.val(np.mean(tang) if tang else float("nan"), "NaN:no_np_arm"), trimmed=b != 14 and ctx.trimmed("RL2"))
    ctx.save_pt(f"ana/r2_b{b}_draws.pt", {"|".join(map(str, k)): v for k, v in draws.items()}, config=dict(block=b, B=2000, seed=1001))
    ctx.write_json(f"ana/r2_b{b}.json", res, config=dict(block=b, B=2000, seed=1001, flips=10000, bootstrap_t=bt))


def NAMES_FOR(L):
    return {"SEA_t": "sea_v", "SPO_t": "sports_v", "SEA_raw": "sea_raw"}.get(L, "T_v")


def _lex(evs, L, words, NP):
    """Per-prompt mean of count(+) - count(-) of a word list over evaluations 1-3 (pushes d_1..d_3) and the prompt's pairs (P4)."""
    per, cnt = np.zeros(NP), np.zeros(NP)
    for n in (1, 2, 3):
        e = evs.get(("DN", L, n))
        if e is None:
            continue
        j = e["wordsets"].index(words); dl = e["words"][0::2, j] - e["words"][1::2, j]; po = np.asarray(e["prompt_of"])
        np.add.at(per, po, dl); np.add.at(cnt, po, 1)
    return per / np.maximum(cnt, 1)


def _flip_p(x, seed=2003, flips=10000):
    x = np.asarray(x, float); eps = np.random.default_rng(seed).integers(0, 2, (flips, len(x))) * 2 - 1; nl = eps @ x / len(x)
    return float((1 + (nl >= x.mean()).sum()) / (flips + 1))


# ------------------------------------------------------------------------------------------------------------------ SENS (R2.3b)
def sens(ctx):
    """SENS_1b(b) (and SENS_sea): the planted signal (A = 0.25) in the DN RSI_a / RSI_b rows n = 2, 3, 4 on unit(u_T^v(b)) (full sample),
    read through the whole pipeline in 500 outer prompt-bootstrap replicates (rng 2101): fold axes re-estimated, cross-fit,
    disattenuation, K1-s (2,000 flips; RG6's statistic), A interval from 200 inner joint-bootstrap replicates. Pass iff both loops meet
    p_s <= 0.01 and A_lo >= 0.10 at all three rows in >= 80 % of the outer replicates. Realised planted A printed."""
    t = T(); b = int(ctx.step.rsplit("_b", 1)[1]); P = R.p256(ctx); NP = len(P["q"]); H = S.halvings(NP); fold = P["fold"]; bt = boot_t(ctx)
    ax = R.ax1b(ctx, b); AS = S.AxisSet(ax["diff"], ax["keep"], ax["fold"], ax["u_star"], ax["u_neg"], b, dev()); g = np.random.default_rng(2101)
    nout, nin = (40, 50) if ctx.mock else (500, 200); out = {}
    for tag, loops, nm in (("SENS_1b", ("RSI_a", "RSI_b"), "T_v"), ("SENS_sea", ("SEA_t",), "sea_v")):
        a1 = AS.get(t.ones(1, NP, dtype=t.float64), [nm])[nm]; u = S.unit(a1["u"][0]); planted, real = {}, {}
        for L in loops:
            for n in (1, 2, 3):
                e = _ev(ctx, b, "DN", L, n)
                if e is None:
                    continue
                r = _row(e, NP); rel = r.rel(H)[0]; planted[(L, n)] = S.plant(r, u, rel); real[(L, n)] = S.readout(planted[(L, n)], a1, fold, H)["A"]
        if len(planted) < 3 * len(loops):
            out[tag] = dict(status=K.coded("NA:not_run:input missing")); continue
        hits = 0
        for _ in range(nout):
            half = NP // 2; pick = np.concatenate([g.integers(0, half, half), half + g.integers(0, NP - half, NP - half)])
            Wo = t.as_tensor(np.bincount(pick, minlength=NP), dtype=t.float64)[None]; ao = AS.get(Wo, [nm])[nm]
            Wi = t.as_tensor(np.stack([np.bincount(pick[g.integers(0, NP, NP)], minlength=NP) for _ in range(nin)]), dtype=t.float64); ai = AS.get(Wi, [nm])[nm]
            ok = True
            for (L, n), r in planted.items():
                bp = S.bvals(r, ao, fold); cnts = np.bincount(pick, minlength=NP)[(r.n > 0).cpu().numpy()]
                p = S.k1s(np.repeat(bp, cnts), boot_t=bt, flips=2000)["p_s"]; lo = S.iv(S.readout(r, ai, fold, H, W=Wi)["A"].cpu().numpy())[0]
                ok = ok and p <= 0.01 and isinstance(lo, float) and lo >= 0.10
            hits += ok
        rate = hits / nout
        out[tag] = dict(pass_rate=rate, passes=rate >= 0.8, realised_planted_A={f"{L}:{n + 1}": v for (L, n), v in real.items()}, outer=nout, inner=nin)
    ctx.write_json(f"ana/sens_b{b}.json", dict(block=b, **out), config=dict(plant_A=0.25, outer=nout, inner=nin, seed=2101, flips=2000, bootstrap_t=bt))


# ------------------------------------------------------------------------------------------------------------------ R3 per block
def _ustat_gain(Dp, w):
    """U statistic of |d|^2 over prompts with weights w [B, P] (per-prompt means Dp [P, D]): (w G w - sum w^2 G_pp) / ((sum w)^2 - sum w^2)."""
    G = Dp @ Dp.T; num = ((w @ G) * w).sum(1) - (w ** 2 * G.diagonal()[None]).sum(1); den = w.sum(1) ** 2 - (w ** 2).sum(1)
    return num / den.clamp_min(1e-12)


def r3_block(ctx):
    """R3.3 per direction (g_par, G, rotation, decision gain, even part, per-sign responses, damage, quality covariates; 'below 1' /
    'reaches 1' (best of k, Bonferroni) / 'near 1' / 'undecided'; WLS shape; even response; reproduction label) and R3.4 (cross-fitted
    noise spectra of Sigma_text(b) and Sigma_pair(b))."""
    t = T(); b = int(ctx.step.rsplit("_b", 1)[1]); P = R.p256(ctx); NP = len(P["q"]); W = W_of(NP); H = S.halvings(NP); fold = P["fold"]
    a0 = R.alpha0(ctx, b); ax = R.ax1b(ctx, b); us = S.unit(R.ustar(ctx, b)[0]); q0 = K.load_json(R.r2_file(ctx, f"r2_q2/theta/b{b}_q0.json"), {}) or {}
    m2 = float(((q0.get("m_star2") or {}).get("pt")) or 1.0); _, a1, aW = _axes(ctx, b, W); u0 = R.load_pt(R.find(ctx, f"r3/b{b}/unpushed.pt")); res = dict(block=b, directions={})
    X0 = u0["X0"].double(); po0 = np.asarray(u0["prompt_of"])
    for p in sorted(glob.glob(os.path.join(ctx.inp, "r1b_R_S3", f"r3/b{b}/dose_*.json"))):
        dj = K.load_json(p); un = dj["direction"]; cells = {}; gd = {}
        for c in dj["cells"]:
            e = R.load_pt(R.find(ctx, c["file"])); r = _row(e, NP); a, cb = c["a"], e["cbar"]; u = e["push"].double().to(dev())
            if un in ("T_v", "sea_v"):
                rd = S.readout(r, aW[un], fold, H, W=W); gW = (rd["c_cf"] * r.d(W).norm(dim=1)) / (a * cb); gp = float(S.readout(r, a1[un], fold, H)["c_cf"] * r.d().norm() / (a * cb))
            else:
                gW = (r.d(W) @ u) / (a * cb) / (math.sqrt(m2) if un == "ustar" else 1.0); gp = float(r.d() @ u) / (a * cb) / (math.sqrt(m2) if un == "ustar" else 1.0)
            Dp = r.S / r.n.clamp_min(1)[:, None]; U = _ustat_gain(Dp, W); GW = t.sqrt(U.clamp_min(0)) / (a * cb); Gp = float(t.sqrt(_ustat_gain(Dp, t.ones(1, NP, dtype=t.float64, device=dev())).clamp_min(0))[0]) / (a * cb)
            xp, xm = e["xp"].double(), e["xm"].double(); ok = e["ok"]; x0 = X0[[int(i) for i in range(len(xp))]] if len(X0) == len(xp) else X0
            ev_ = ((xp + xm) / 2 - x0)[ok].mean(0); rot = gp / Gp if Gp else float("nan")
            gd[c["cell"]] = (gW if (un in ("T_v", "sea_v") or rot >= 0.9) else GW).cpu().numpy()
            rep4 = float(np.mean([_rep4(cc) for cc in e["conts"]]))
            cells[c["cell"]] = dict(a=a, a_over_alpha0=a / a0, g_par=gp, g_par_iv=S.iv(gW.cpu().numpy()), G=Gp, G_iv=S.iv(GW.cpu().numpy()), rotation=K.val(rot),
                                    decision=gp if (un in ("T_v", "sea_v") or rot >= 0.9) else Gp, gain_iv=S.iv(gd[c["cell"]]),
                                    even_gain=float(S.unit(u.cpu()) @ ev_) / (a * cb), cos_even_ustar=S.cos(ev_, us), cos_even_ustar_iv=_cos_iv(xp, xm, x0, ok, us, e["prompt_of"], W),
                                    plus_resp=float(S.unit(u.cpu()) @ (xp - x0)[ok].mean(0)), minus_resp=float(S.unit(u.cpu()) @ (xm - x0)[ok].mean(0)),
                                    dmg=e["dmg"], repeat4=rep4, mean_len=float(np.mean([len(cc) for cc in e["conts"]])), u32=e.get("u32"))
        small = [k for k, c in cells.items() if c["a"] <= 0.5 * a0 + 1e-12]; kk = len(small)
        below = kk > 0 and all(isinstance(cells[k]["gain_iv"][1], float) and cells[k]["gain_iv"][1] < 1 for k in small)
        reach = any(np.nanpercentile(gd[k], 100 * 0.05 / kk) > 1 for k in small) if kk else False
        near = any(isinstance(cells[k]["gain_iv"][0], float) and cells[k]["gain_iv"][0] >= 0.8 for k in small) and not reach
        s = "below 1 at small push" if below else ("reaches 1 at small push" if reach else ("near 1 at small push" if near else "undecided"))
        pr = dict(reach=min(1.0, kk * min(pos(gd[k], 1.0) for k in small)) if kk else 1.0, below=max(pos(gd[k], 1.0, "less") for k in small) if kk else 1.0)
        fit = [k for k, c in cells.items() if c["a"] <= a0 + 1e-12 and isinstance(c["gain_iv"][0], float) and c["gain_iv"][0] > 0]
        shape = "shape not estimable"; slope_iv = None
        if len(fit) >= 3:
            la = np.log([cells[k]["a"] for k in fit]); lg = np.log(np.clip(np.stack([gd[k] for k in fit]), 1e-12, None)); w_ = 1 / np.var(lg, 1).clip(1e-12)
            sl = [np.polyfit(la, lg[:, i], 1, w=np.sqrt(w_))[0] for i in range(lg.shape[1])]; slope_iv = S.iv(sl)
            if isinstance(slope_iv[0], float):
                shape = "linear regime" if -0.2 < slope_iv[0] and slope_iv[1] < 0.2 else ("gain rises toward small pushes (soft onset)" if slope_iv[1] < 0 else
                                                                                         ("gain falls toward small pushes" if slope_iv[0] > 0 else "shape undecided"))
        even = any(isinstance(c["cos_even_ustar_iv"][0], float) and (c["cos_even_ustar_iv"][0] > 0 or c["cos_even_ustar_iv"][1] < 0) for c in cells.values() if c["a"] >= a0 - 1e-12)
        lab = []
        if b == 14 and un == "ustar":
            tvp = ((R.tv(ctx, False) or {}).get("blocks", {}).get("14", {}).get("R1", {}) or {}).get("points") or []
            ok_ = [tvp[i]["lo"] <= cells[k]["decision"] <= tvp[i]["hi"] for i, k in enumerate(("0.5", "1", "2")) if k in cells and i < len(tvp)]
            if ok_ and all(ok_):
                lab.append("round-2 cells reproduced")
        res["directions"][un] = dict(cells=cells, k=kk, string=s, p=pr, shape=shape, slope_iv=slope_iv,
                                     even="even response along u*_b" if even else "no even response resolved", labels=lab)
    res["R34"] = _r34(ctx, b, P, us, ax)
    ctx.write_json(f"ana/r3_b{b}.json", res, config=dict(block=b, B=2000))


def _rep4(c):
    g = [tuple(c[i:i + 4]) for i in range(len(c) - 3)]
    return 1 - len(set(g)) / len(g) if g else 0.0


def _cos_iv(xp, xm, x0, ok, u, po, W):
    t = T(); e = ((xp + xm) / 2 - x0); e[~ok] = 0; po = t.as_tensor(po).long(); NP = W.shape[1]
    Sp = t.zeros(NP, e.shape[1], dtype=t.float64).index_add_(0, po, e).to(W.device); n = t.bincount(po[ok], minlength=NP).double().to(W.device)
    m = (W @ Sp) / (W @ n).clamp_min(1e-12)[:, None]; c = (m @ S.unit(u.double()).to(W.device)) / m.norm(dim=1).clamp_min(1e-12)
    return S.iv(c.cpu().numpy())


def _r34(ctx, b, P, us, ax):
    """R3.4: Sigma_text(b) from TXT0 (within-prompt deviations), cross-fitted top-5 spectrum over the 20 halvings and both assignments;
    rho*, |cos(u*_b, v_1)|, share in span(v_1..v_5), the same for u_T^v and u_sea^v, NF_text; Sigma_pair(b) from the DN round-0 pairs of
    RND_1 / RND_2; 50 random unit directions for the rho null."""
    t = T(); out = {}
    rd = R.load_pt(R.find(ctx, f"txt0/reads_b{b}.pt")); X = rd["X"].double(); po = t.as_tensor(rd["prompt_of"]).long(); ok = ~t.isnan(X).any(1)
    X, po = X[ok], po[ok]; NP = len(P["q"]); mu = t.zeros(NP, X.shape[1], dtype=t.float64).index_add_(0, po, X) / t.bincount(po, minlength=NP).clamp_min(1).double()[:, None]
    dev_ = X - mu[po]; ev = dict(text=(dev_, po))
    pr = []
    for L in ("RND_1", "RND_2"):
        e = _ev(ctx, b, "DN", L, 0)
        if e is not None:
            d = R.delta_of(e)[e["ok"]]; pr.append((d - d.mean(0), t.as_tensor(e["prompt_of"])[e["ok"]].long()))
    if pr:
        ev["pair"] = (t.cat([x for x, _ in pr]), t.cat([y for _, y in pr]))
    rnd = [S.unit(t.randn(X.shape[1], generator=t.Generator().manual_seed(77 + i), dtype=t.float64)) for i in range(50)]
    for nm, (Z, pz) in ev.items():
        H = S.halvings(NP); vals = []
        for h in H:
            for A_ in (True, False):
                m = h.bool()[pz] if A_ else ~h.bool()[pz]; ZA, ZB = Z[m], Z[~m]
                if len(ZA) < 6 or len(ZB) < 6:
                    continue
                CA = ZA.T @ ZA / len(ZA); CB = ZB.T @ ZB / len(ZB); w, V = t.linalg.eigh(CA); V5 = V[:, -5:].flip(1)
                lam1 = float(V5[:, 0] @ CB @ V5[:, 0]); f = lambda v: dict(rho=float(v @ CB @ v) / lam1, cos1=abs(float(v @ V5[:, 0])), share=float(((V5.T @ v) ** 2).sum()))
                vals.append(dict(u=f(us), T=f(S.unit(ax["axes"]["T_v"].double())), sea=f(S.unit(ax["axes"]["sea_v"].double())), rnd=[f(v)["rho"] for v in rnd], nf=float(CB.trace().sqrt())))
        if not vals:
            out[nm] = K.coded("NA:not_run:too few texts"); continue
        agg = {k: {q: float(np.mean([v[k][q] for v in vals])) for q in ("rho", "cos1", "share")} for k in ("u", "T", "sea")}
        su = agg["u"]; s = ("style mode = strongest noise direction" if su["cos1"] >= 0.7 and su["rho"] >= 0.8 else
                            ("style mode among the dominant noise directions" if su["share"] >= 0.5 else "style mode is not a dominant noise direction"))
        out[nm] = dict(**agg, rho_random=K.dist(np.mean([v["rnd"] for v in vals], 0)), NF_text=float(np.mean([v["nf"] for v in vals])), string=s, fits=len(vals))
    return out


# ------------------------------------------------------------------------------------------------------------------ R4 per block
def r4_block(ctx):
    """R4.1 lambda_lin(u) with intervals; R4.2 cross-fitted rightmost eigenvalues of C_b (32) and the 24 word-set subspaces (4) over the 20
    halvings x 2 assignments (median of held-out values), joint bootstrap (2,000; halvings fixed, subspaces fixed), 200 within-prompt
    permutation nulls (seeds 2201 + i) for C and the full space (descriptive), the R4.2 strings; R4.3 direct check."""
    t = T(); b = int(ctx.step.rsplit("_b", 1)[1]); P = R.p256(ctx); NP = len(P["q"]); H = S.halvings(NP); J = R.load_pt(R.find(ctx, f"r4/jloop_b{b}.pt"))
    f = R.load_pt(R.find(ctx, f"r4/fdt_b{b}.pt")); ok = ~t.isnan(f["X"]).any(1); X, Sg = f["X"][ok].double(), f["S"][ok].double(); po = t.as_tensor(f["prompt_of"])[ok].long()
    cbar = J["cbar"]; Xc, Sc, _ = S.fdt_centre(X, Sg, po); ns = t.bincount(po, minlength=NP).double(); fac = (ns / (ns - 1).clamp_min(1))[po]
    subs = {"C": J["Q_C"].double(), **{k: v.double() for k, v in J["V_S"].items()}}; Qall = t.cat(list(subs.values()), 1)
    XQ, SQ = Xc @ Qall, Sc @ Qall; off = {}; o = 0
    for k, v in subs.items():
        off[k] = (o, o + v.shape[1]); o += v.shape[1]
    B = 200 if ctx.mock else 2000; Wp = K.prompt_bootstrap(NP, B=B, seed=1001)

    def M_of(mask, wt, k):
        i, j = off[k]; w = wt * fac * mask; return ((XQ[:, i:j] * w[:, None]).T @ SQ[:, i:j]) / (w.sum().clamp_min(1e-12) * cbar)

    def cf(wt, Sperm=None):
        out = {}
        for k in subs:
            hv = []
            for h in H:
                hm = h.bool()[po].double()
                for A_ in (hm, 1 - hm):
                    MA, MB = M_of(A_, wt, k), M_of(1 - A_, wt, k); lam, y, z, _ = S.rightmost(MA); v, ray = S.heldout(MB, y, z); hv.append(v)
            out[k] = float(np.median(hv))
        return out
    pt = cf(t.ones(len(po), dtype=t.float64)); bs = {k: [] for k in subs}
    for i in range(B):
        r = cf(t.as_tensor(Wp[i], dtype=t.float64)[po])
        for k in subs:
            bs[k].append(r[k])
    perm = []
    nper = 20 if ctx.mock else 200
    for i in range(nper):
        g = t.Generator().manual_seed(2201 + i); idx = t.arange(len(po))
        for p in range(NP):
            m = (po == p).nonzero().flatten()
            if len(m) > 1:
                idx[m] = m[t.randperm(len(m), generator=g)]
        SQp = SQ[idx]; hv = []
        for h in H:
            hm = h.bool()[po].double()
            for A_ in (hm, 1 - hm):
                i0, j0 = off["C"]; wa, wb = fac * A_, fac * (1 - A_)
                MA = ((XQ[:, i0:j0] * wa[:, None]).T @ SQp[:, i0:j0]) / (wa.sum() * cbar); MB = ((XQ[:, i0:j0] * wb[:, None]).T @ SQp[:, i0:j0]) / (wb.sum() * cbar)
                lam, y, z, _ = S.rightmost(MA); hv.append(S.heldout(MB, y, z)[0])
        perm.append(float(np.median(hv)))
    lmC, lmR = pt["C"], pt["W_RSI_build"]; dif = np.array(bs["C"]) - np.array(bs["W_RSI_build"]); div = S.iv(dif); riv = S.iv(bs["W_RSI_build"])
    resolved = lmC > np.percentile(perm, 95); vC = J["modes"]["C"]["v"].double(); VR = J["V_S"]["W_RSI_build"].double()
    proj = float(((VR.T @ S.unit(vC)) ** 2).sum()); neu = [pt[k] for k in subs if k.startswith("neutral")]
    st = []
    if not resolved:
        st += ["RSI subdominant: not decidable (top mode not resolved)", "RSI is the top mode: not decidable (top mode not resolved)",
               "lambda_max is the style mode: not decidable (top mode not resolved)"]
    else:
        if isinstance(div[0], float) and div[0] > 0:
            st.append("RSI subdominant")
        elif isinstance(div[0], float) and div[0] <= 0 <= div[1] and proj >= 0.5:
            st.append("RSI is the top mode")
        if abs(S.cos(vC, R.ustar(ctx, b)[0])) >= 0.7:
            st.append("lambda_max is the style mode")
    lam_s = ("lambda_RSI >= 1" if isinstance(riv[0], float) and riv[0] > 1 else ("lambda_RSI below 1" if isinstance(riv[1], float) and riv[1] < 1 else None))
    if isinstance(riv[0], float) and riv[0] >= 0.8:
        st.append("lambda_RSI near critical")
    if lam_s:
        st.append(lam_s)
    if neu and lmR > np.percentile(neu, 95):
        st.append("RSI above neutral words")
    lin = {}
    for nm, u in (("ustar", R.ustar(ctx, b)[0]), ("T_v", R.ax1b(ctx, b)["axes"]["T_v"]), ("sea_v", R.ax1b(ctx, b)["axes"]["sea_v"]), ("cov4", R.pool_dir(ctx, "cov", b, 4, X.shape[1]))):
        uu = S.unit(u.double()); xu, su_ = Xc @ uu, Sc @ uu; v = lambda w: float((xu * su_ * w * fac).sum() / ((w * fac).sum() * cbar))
        lin[nm] = dict(pt=v(t.ones(len(po), dtype=t.float64)), iv=S.iv([v(t.as_tensor(Wp[i], dtype=t.float64)[po]) for i in range(B)]))
    r43 = {}
    for which in ("vmax", "vRSI"):
        p = R.find(ctx, f"r4/b{b}/r43_{which}.pt")
        if not p:
            continue
        e = R.load_pt(p); r = _row(e, NP); v, w = e["push"].double(), S.unit(e["w"].double()); a, cb = e["a"], e["cbar"]; ray = abs(float(S.unit(v) @ w)) < 0.2
        ld = lambda d: (float(d @ v) / (a * cb) if ray else float(w @ d) / (a * cb * float(w @ v)))
        Wt = W_of(NP, B); dW = r.d(Wt).cpu(); ldW = np.array([ld(d) for d in dW]); diff = ldW - e["lam_FDT"]; iv_ = S.iv(diff)
        cell = "not decidable (top mode not resolved)" if (which == "vmax" and not resolved) else \
            ("FDT confirmed by a direct push" if isinstance(iv_[0], float) and iv_[0] <= 0 <= iv_[1] else "FDT not confirmed by a direct push")
        r43[which] = dict(lambda_direct=ld(r.d().cpu()), lambda_FDT=e["lam_FDT"], diff_iv=iv_, string=cell, labels=["non-normal: Rayleigh read"] if ray else [])
    lab = [] if all(x["string"] != "FDT not confirmed by a direct push" for x in r43.values()) else ["FDT not confirmed by a direct push"]
    ctx.write_json(f"ana/r4_b{b}.json", dict(block=b, lambda_cf=pt, lambda_iv={k: S.iv(v) for k, v in bs.items()}, perm_C=K.dist(perm), resolved=bool(resolved),
                                             diff_iv=div, proj_vmax_VRSI=proj, strings=st, lambda_RSI=dict(pt=lmR, iv=riv, string=lam_s),
                                             p=dict(ge1=pos(bs["W_RSI_build"], 1.0), lt1=pos(bs["W_RSI_build"], 1.0, "less")), lambda_lin=lin, R43=r43,
                                             ritz_full=J["ritz_full"]["lam"], labels=lab), config=dict(block=b, B=B, perms=nper, halvings=len(H)))


# ------------------------------------------------------------------------------------------------------------------ R6
def r6(ctx):
    """R6.1 rho = g_soft / g_text (plain projections on unit(u), paired prompt bootstrap; defined only if g_text's interval excludes 0),
    latent loop on RSI, iterated latent loops; R6.2 D_k (cross-fitted projection on u_T^v(14)), K_2, K_geo, random-direction control
    (stats-7: 'not decidable (random-direction K not defined)'); R6.3 replication on R1's u_T_perp(14)."""
    t = T(); P = R.p256(ctx); NP = len(P["q"]); W = W_of(NP); H = S.halvings(NP); fold = P["fold"]; out = dict(R61={}, R62={}, R63=None)
    for b in R.BLOCKS:
        js = R.find(ctx, f"r6/r61_b{b}.json")
        if not js:
            continue
        for u, c in K.load_json(js)["cells"].items():
            e = R.load_pt(R.find(ctx, c["file"])); tx = R.load_pt(R.find(ctx, f"r3/b{b}/{u}/cell_DN.pt")); uu = e["push"].double().to(dev())
            rs, rt = S.Row((e["xp"].double() - e["xm"].double()) / 2, ~t.isnan(e["xp"]).any(1), e["prompt_of"], NP, dev()), _row(tx, NP)
            gs, gt = (rs.d(W) @ uu) / (e["a"] * e["cbar"]), (rt.d(W) @ uu) / (tx["a"] * tx["cbar"]); gt_iv = S.iv(gt.cpu().numpy())
            rec = dict(g_soft=float(rs.d() @ uu) / (e["a"] * e["cbar"]), g_soft_iv=S.iv(gs.cpu().numpy()), g_text=float(rt.d() @ uu) / (tx["a"] * tx["cbar"]), g_text_iv=gt_iv)
            if isinstance(gt_iv[0], float) and (gt_iv[0] > 0 or gt_iv[1] < 0):
                lr = t.log((gs / gt).abs().clamp_min(1e-12)).cpu().numpy() * np.sign((gs / gt).cpu().numpy()); li = S.iv(lr); L125 = math.log(1.25)
                rec["log_rho_iv"] = li
                rec["string"] = ("latent loop stronger" if li[0] > L125 else "text loop stronger" if li[1] < -L125 else "equal within 25 %" if -L125 < li[0] and li[1] < L125 else "undecided")
            else:
                rec["string"] = "not decidable (text gain not resolved)"; rec["rho"] = K.coded("NaN:denominator_interval_includes_0")
            if u == "T_v":
                rec["RSI"] = ("latent loop reaches 1 on RSI" if isinstance(rec["g_soft_iv"][0], float) and rec["g_soft_iv"][0] > 1 else
                              "latent loop below 1 on RSI" if isinstance(rec["g_soft_iv"][1], float) and rec["g_soft_iv"][1] < 1 else "undecided")
            out["R61"][f"{b}:{u}"] = rec
        ji = R.find(ctx, f"r6/r61it_b{b}.json")
        if ji:
            for L, f_ in K.load_json(ji)["loops"].items():
                it = R.load_pt(R.find(ctx, f_)); rr = it["rows"][-1]; d4 = ((rr["xp"].double() - rr["xm"].double()) / 2).nanmean(0)
                out["R61"][f"{b}:iter:{L}"] = dict(cos_ustar=abs(S.cos(d4, R.ustar(ctx, b)[0])), string="latent loop style-attracted" if abs(S.cos(d4, R.ustar(ctx, b)[0])) >= 0.5 else
                                                   "latent loop not style-attracted", labels=["descriptive (cosine not debiased: one soft text per prompt)"])
    ax = R.ax1b(ctx, 14); _, a1, aW = _axes(ctx, 14, W); us = S.unit(R.ustar(ctx, 14)[0]).to(dev())

    def Dk(xp, xm, ok, prompt_of, axis="T_v", fixed=None):
        r = S.Row((xp.double() - xm.double()) / 2, ok, prompt_of, NP, dev())
        if fixed is not None:
            fx = S.unit(fixed.double()).to(dev()); return float(r.d() @ fx), (r.d(W) @ fx).cpu().numpy()
        f_ = t.as_tensor(fold, dtype=t.float64).to(dev()); n = (W @ r.n).clamp_min(1e-12); SA, SB = (r.S @ aW[axis]["aA"].T).T, (r.S @ aW[axis]["aB"].T).T
        dW = ((W * f_[None] * SB).sum(1) + (W * (1 - f_)[None] * SA).sum(1)) / n; one = t.ones(1, NP, dtype=t.float64, device=dev())
        d1 = float(((one * f_[None] * (r.S @ a1[axis]["aB"].T).T).sum(1) + (one * (1 - f_)[None] * (r.S @ a1[axis]["aA"].T).T).sum(1)) / (one @ r.n))
        return d1, dW.cpu().numpy()

    def carry(fp, fixed=None):
        d = R.load_pt(R.find(ctx, fp)); po = t.as_tensor(d["prompt_of"]); tl = d["turn1"]; ok1 = tl["ok"]
        D = {1: Dk(tl["xp"], tl["xm"], ok1, po, fixed=fixed)}
        for k, v in sorted(d["turns"].items()):
            okk = ~(t.isnan(v["xp"]).any(1) | t.isnan(v["xm"]).any(1)); D[int(k)] = Dk(v["xp"], v["xm"], okk, po, fixed=fixed)
        d1iv = S.iv(D[1][1]); defined = isinstance(d1iv[0], float) and (d1iv[0] > 0 or d1iv[1] < 0)
        K_ = {k: dict(pt=D[k][0] / D[1][0], iv=S.iv(D[k][1] / D[1][1])) for k in D if k > 1} if defined else {}
        kap = {k: dict(pt=D[k][0] / D[k - 1][0], iv=S.iv(D[k][1] / D[k - 1][1])) for k in D if k > 1 and k - 1 in D}
        rt = {int(k): v.get("round_trip_failures") for k, v in d["turns"].items()}
        return D, defined, d1iv, K_, kap, rt
    for T_ in ("0.7", "1.0"):
        p = R.find(ctx, f"r6/r62_T{T_}.pt")
        if not p:
            continue
        D, defined, d1iv, K_, kap, rt = carry(f"r6/r62_T{T_}.pt"); rec = dict(D1=dict(pt=D[1][0], iv=d1iv), round_trip_failures=rt, kappa=kap)
        if not defined:
            rec["string"] = "K not defined (turn-1 RSI stance not resolved)"; rec["p"] = dict(ge1=1.0, lt1=1.0)
        else:
            k2 = K_[2]; rec["K"] = K_
            rec["string"] = (f"self-sustaining at T {T_}" if isinstance(k2["iv"][0], float) and k2["iv"][0] > 1 else
                             f"dissipative at T {T_}" if isinstance(k2["iv"][1], float) and k2["iv"][1] < 1 else "undecided")
            Kd = D[2][1] / D[1][1]; rec["p"] = dict(ge1=pos(Kd, 1.0), lt1=pos(Kd, 1.0, "less")); rec["K2_draws_sd"] = float(np.nanstd(Kd))
            ks = [k for k in sorted(D) if isinstance(S.iv(D[k][1])[0], float) and S.iv(D[k][1])[0] > 0]
            if len(ks) >= 3:
                sl = [np.polyfit(ks, np.log(np.clip([D[k][1][i] for k in ks], 1e-12, None)), 1)[0] for i in range(len(D[1][1]))]
                gi = [math.exp(x) for x in S.iv(sl)]; rec["K_geo"] = dict(iv=gi, string=f"self-sustaining at T {T_}" if gi[0] > 1 else (f"dissipative at T {T_}" if gi[1] < 1 else "undecided"))
            else:
                rec["K_geo"] = dict(string="K_geo not estimable")
        Du, *_ = carry(f"r6/r62_T{T_}.pt", fixed=us.cpu()); rec["ustar_K2"] = Du[2][0] / Du[1][0] if 2 in Du and Du[1][0] else K.coded("NaN:zero_denominator")
        out["R62"][T_] = rec
    if R.find(ctx, "r6/r62_rnd.pt") and "0.7" in out["R62"]:
        rd = R.load_pt(R.find(ctx, "r6/r62_rnd.pt")); D, defined, d1iv, K_, kap, rt = carry("r6/r62_rnd.pt", fixed=rd["turn1"]["push"])
        if not defined:
            out["R62"]["random_control"] = dict(string="not decidable (random-direction K not defined)", D1_iv=d1iv)
        elif "K" not in out["R62"]["0.7"]:
            out["R62"]["random_control"] = dict(string="not decidable (RSI K not defined)")
        else:
            Dr, *_ = carry("r6/r62_T0.7.pt"); dd = Dr[2][1] / Dr[1][1] - D[2][1] / D[1][1]; iv_ = S.iv(dd)
            out["R62"]["random_control"] = dict(diff_iv=iv_, string="RSI carry above generic carry" if iv_[0] > 0 else ("RSI carry below generic carry" if iv_[1] < 0 else
                                                                                                                       "not distinguishable from generic carry"))
    if R.find(ctx, "r6/r63.pt"):
        rec = {}
        for nm, fx in (("u_T_perp_R1", R.r1_state(ctx, 14)["axes"]["u_T_perp"]), ("T_v", None), ("ustar", us.cpu())):
            D, defined, d1iv, K_, kap, rt = carry("r6/r63.pt", fixed=fx)
            rec[nm] = dict(D1_iv=d1iv, K=K_.get(2) if defined else K.coded("NaN:denominator_interval_includes_0"))
        k = rec["u_T_perp_R1"]["K"]
        rec["string"] = ("K not defined" if not isinstance(k, dict) else "round-2 T 1.0 cell not replicated (dissipative on u_T_perp)" if k["iv"][1] < 1 else
                         "replicated, self-sustaining (K lower bound > 1)" if k["iv"][0] > 1 else "replicated (K >= 1 not excluded)")
        out["R63"] = rec
    ctx.write_json("ana/r6.json", out, config=dict(B=2000))


# ------------------------------------------------------------------------------------------------------------------ verdicts
def v_rsi(c, pc, sens, go0, cc_ok, band_bad, k5, pc_testable):
    """R2.4's ordered verdict list (first string whose conditions hold); returns (string, reached-member flags)."""
    if c.get("status"):
        return "not decidable (input missing)"
    num = lambda x: x if isinstance(x, float) else float("nan")
    Cp = c["MANIP_1b"] and c["rel_ok"] and num(c["E"]["iv"][0]) >= 0.5 and num(c["R4cf"]) >= 0.5 and all(v > 0 for v in c["D4"].values()) and \
        num(c["SPO_absA4_hi"]) < 0.25 and not c["RND_attracted"]
    pro = all(p <= 0.01 for v in c["p_pro_234"].values() for p in v); two = all(p <= 0.01 for v in c["p_two_234"].values() for p in v)
    s4 = Cp and pro and k5["ok_likely"] and num(c["A_bar"]["iv"][0]) >= 0.50; s5 = Cp and pro and k5["ok_likely"] and num(c["A_bar"]["iv"][0]) >= 0.25
    s6 = Cp and two and k5["ok_osc"] and c["alternating"] and num(c["A_bar_abs"]["iv"][0]) >= 0.25
    s7 = c["MANIP_1b"] and num(c["A_bar_abs"]["iv"][1]) < 0.25 and not c["RND_attracted"]
    s8 = all(num(v) < 0.10 for v in c["A1_hi"].values())
    lst = [(not cc_ok, "descriptive (code check failed; would have read: {})"), (not go0, "not decidable (RSI axis not resolved)"),
           (band_bad, "not decidable (damage calibration failed)"), (pc and s4, "likely, RSI-dominated"), (pc and s5, "likely, substantial RSI component"),
           (pc and s6, "oscillation (RSI begets anti-RSI)"), (sens and s7, "unlikely (RSI does not persist)"), (sens and s8, "unlikely (no first link)"),
           ((s4 or s5 or s6) and not pc_testable, "undecided (positive control not testable)"), ((s4 or s5 or s6) and not pc, "undecided (positive control failed)"),
           ((s7 or s8) and not sens, "undecided (sensitivity check failed)"), (True, "undecided")]
    for i, (cond, s) in enumerate(lst):
        if cond:
            if i == 0:
                inner = v_rsi(c, pc, sens, go0, True, band_bad, k5, pc_testable); return s.format(inner)
            return s


def verdicts(ctx):
    """All R strings, RG7 / RG8a / RG8b / RG12 / RG15, F+ / F- with Holm (component p-values: R.1), counts against the null expectation."""
    out = dict(blocks={}, F_plus=[], F_minus=[]); idx = K.load_json(ctx.path("ana/index.json")); r6_ = K.load_json(ctx.path("ana/r6.json"))
    pcs = {}
    for b in R.BLOCKS:
        r2, sn, r3, r4 = (K.load_json(ctx.path(f"ana/{x}_b{b}.json")) for x in ("r2", "sens", "r3", "r4"))
        axj = K.load_json(R.find(ctx, f"ax1b/ax1b_b{b}.json")); go0 = axj["go0"]["auc_T_v"] >= 0.85
        pc_axis = "SEA_t" if axj["go0"]["auc_sea_v"] >= 0.80 else ("SPO_t" if axj["go0"]["auc_sports_v"] >= 0.80 else None)
        pc = bool(pc_axis and r2["PC"].get(pc_axis, {}).get("pass_")); pcs[b] = pc
        S1, Ss = bool((sn.get("SENS_1b") or {}).get("passes")), bool((sn.get("SENS_sea") or {}).get("passes"))
        cal = K.load_json(R.find(ctx, f"r2/b{b}/r31_calibration.json")); bad = max(cal["rg9"].get(f"DN:{L}", 0) for L in ("RSI_a", "RSI_b")) > 1
        cc = idx["RG5"][str(b)]["ok"]; other = "T_ps" if b == 6 else "T_1b"
        v = v_rsi(r2["components"]["T_v"], pc, S1, go0, cc, bad, r2["K5"], pc_axis is not None)
        v2 = v_rsi(r2["components"].get(other, {}), pc, S1, go0, cc, bad, r2["K5"], pc_axis is not None)
        lab = []
        if r2["style_attracted"]:
            lab.append("style-attracted")
        if idx.get("bootstrap_t"):
            lab.append("bootstrap-t used (RG6)")
        if any(r["arm"] == "DN" and r["loop"].startswith("RSI") and not r["in_band"] for r in cal["rows"]):
            lab.append("damage out of band")
        if r2.get("robust"):
            c = dict(r2["components"]["T_v"]); c.update(A_bar=dict(iv=r2["robust"]["A_bar_iv"]), A_bar_abs=dict(iv=r2["robust"]["A_bar_abs_iv"]), MANIP_1b=r2["robust"]["MANIP_1b"],
                                                        p_pro_234=r2["robust"]["p_pro_234"], A1_hi=r2["robust"]["A1_hi"])
            if v_rsi(c, pc, S1, go0, cc, bad, r2["K5"], pc_axis is not None) != v:
                lab.append("axis-sensitive (1b generic mode)")
        pcr = r2["PC"].get("SEA_t", {}); topic = "no topic persists at b" if b == 6 else "no topic persists beyond the generic mode at b"
        if not pc:
            sp = r2["PC"].get("SPO_t", {}); small = all(isinstance(x, float) and x < 0.25 for x in pcr.get("absA_hi_34", [1]) + sp.get("absA_hi_34", [1]))
            if small and Ss:
                lab.append(topic.replace(" b", f" {b}"))
            elif not Ss:
                lab.append("PC not informative (insensitive)")
        pc_str = "PC_PASS_1b" if pc else (f"PC fails ({', '.join(k for k in ('P1', 'P2', 'P3', 'P4', 'P5') if not pcr.get(k))})" if pc_axis else
                                          "PC not testable (no topic axis beyond the generic mode)")
        c = r2["components"]["T_v"]; pcc = (r2["PC"].get(pc_axis) or {}).get("p_comp", {}) if pc_axis else {}
        p_like = max([*c["p_pro_234"]["RSI_a"], *c["p_pro_234"]["RSI_b"], c["p_comp"]["Abar25"], c["p_comp"]["E50"], c["p_comp"]["SPO"],
                      *c["p_comp"]["A1_10"].values(), *pcc.get("p_s", [1]), *pcc.get("A_lo10", [1]), pcc.get("P4", 1), pcc.get("P5", 1)]) if v.startswith("likely") else 1.0
        p_unl = max([c["p_comp"]["Abar_abs_lt25"], *c["p_comp"]["A1_10"].values()]) if v == "unlikely (RSI does not persist)" else 1.0
        if not go0:
            p_like = p_unl = 1.0
        d3 = r3["directions"].get("T_v", {}); s3 = d3.get("string") if go0 else "not decidable (RSI axis not resolved)"
        out["F_plus"] += [dict(member=f"V_RSI_1b likely at {b}", reached=v.startswith("likely"), p=p_like),
                          dict(member=f"R3.3 reaches 1 at small push (u_T^v, {b})", reached=s3 == "reaches 1 at small push", p=d3.get("p", {}).get("reach", 1.0) if s3 == "reaches 1 at small push" else 1.0),
                          dict(member=f"R4.2 lambda_RSI >= 1 at {b}", reached=r4["lambda_RSI"]["string"] == "lambda_RSI >= 1", p=r4["p"]["ge1"] if r4["lambda_RSI"]["string"] == "lambda_RSI >= 1" else 1.0)]
        out["F_minus"] += [dict(member=f"unlikely (RSI does not persist) at {b}", reached=v == "unlikely (RSI does not persist)", p=p_unl),
                           dict(member=f"R3.3 below 1 at small push (u_T^v, {b})", reached=s3 == "below 1 at small push", p=d3.get("p", {}).get("below", 1.0) if s3 == "below 1 at small push" else 1.0),
                           dict(member=f"R4.2 lambda_RSI below 1 at {b}", reached=r4["lambda_RSI"]["string"] == "lambda_RSI below 1", p=r4["p"]["lt1"] if r4["lambda_RSI"]["string"] == "lambda_RSI below 1" else 1.0)]
        np_ = r2["R32"]; dn, nv = np_["deb"].get("DN"), np_["deb"].get("NP")
        r32 = ("not run (RL2)" if np_["trimmed"] else "NP arm uninformative (push nearly tangential)" if isinstance(np_["mean_abs_cos"], float) and np_["mean_abs_cos"] < 0.05 else
               "style attraction independent of norm inflation" if isinstance(dn, float) and isinstance(nv, float) and dn >= 0.5 and nv >= 0.5 else
               "style attraction needs norm inflation" if isinstance(dn, float) and dn >= 0.5 and isinstance(np_["deb"].get("NP_iv", [None, None])[1], float) and np_["deb"]["NP_iv"][1] < 0.3 else "undecided")
        out["blocks"][str(b)] = dict(V_RSI_1b=K.labelled(v, *lab), other_axis=dict(axis=other, string=v2), PC=pc_str, PC_axis=pc_axis, GO0_1b=go0, SENS_1b=sn.get("SENS_1b"),
                                     SENS_sea=sn.get("SENS_sea"), R27=dict(label=r2["R27"]["label"], generic=r2["R27"]["generic"]), R32=r32,
                                     R33={u: dict(string=(d["string"] if (u != "T_v" or go0) else "not decidable (RSI axis not resolved)"), shape=d["shape"], even=d["even"], labels=d["labels"])
                                          for u, d in r3["directions"].items()}, R34={k: (v_["string"] if isinstance(v_, dict) and "string" in v_ else v_) for k, v_ in r3["R34"].items()},
                                     R42=K.labelled(r4["strings"], *r4.get("labels", [])), R43={k: v_["string"] for k, v_ in r4["R43"].items()}, RG9_bad=bad, RG5=cc)
    r62 = r6_["R62"].get("0.7", {}); s62 = r62.get("string", "not run")
    out["F_plus"].append(dict(member="R6.2 self-sustaining at T 0.7", reached=s62 == "self-sustaining at T 0.7", p=r62.get("p", {}).get("ge1", 1.0) if s62 == "self-sustaining at T 0.7" else 1.0))
    out["F_minus"].append(dict(member="R6.2 dissipative at T 0.7", reached=s62 == "dissipative at T 0.7", p=r62.get("p", {}).get("lt1", 1.0) if s62 == "dissipative at T 0.7" else 1.0))
    for fam in ("F_plus", "F_minus"):
        adj, rej = K.holm([m["p"] for m in out[fam]])
        for m, a, r in zip(out[fam], adj, rej):
            m.update(p_holm=a, label="confirmatory (Holm)" if (m["reached"] and r) else "exploratory (Holm not met)")
        out[fam + "_counts"] = dict(reached=sum(m["reached"] for m in out[fam]), confirmed=sum(m["label"] == "confirmatory (Holm)" for m in out[fam]),
                                    null_expectation=dict(reached="<= 0.5", confirmed="<= 0.05 (family-wise)"))
    ok = [b for b in R.BLOCKS if pcs[b] and out["blocks"][str(b)]["RG5"] and not out["blocks"][str(b)]["RG9_bad"]]
    out["R25"] = _r25(ctx, ok)
    out["R6"] = r6_; out["R5"] = {k: K.load_json(R.find(ctx, k) or "", None) for k in ("jlens_validity.json", "rg11.json", "r5/r52.json", "r5/r53.json", "r5/r55.json")}
    out["R7"] = {k: K.load_json(R.find(ctx, f"r7/{k}.json") or "", None) for k in ("r71_select", "r71", "r72", "r73", "r71_others")}
    r81 = K.load_json(R.find(ctx, "r8_1.json") or "", {}) or {}; J1, J2 = bool(r81.get("J1")), bool(pcs.get(14))
    out["RG15"] = dict(J1=J1, J2=J2); out["R0"] = K.load_json(R.find(ctx, "r0_table.json") or "", {}).get("readme_null_claim")
    out["R1"] = {b: v["verdict"] for b, v in (K.load_json(R.find(ctx, "r1/r1_replay.json") or "", {}) or {}).get("blocks", {}).items()}
    out["RG17"] = idx["RG17"]; out["code_counts"] = K.code_counts(out); out["notes"] = ["R0 is post hoc (R0) and exploratory", "everything outside F+ / F- is exploratory"]
    for x in ("R5", "R7"):
        for v in (out[x] or {}).values():
            if isinstance(v, dict):
                v.pop("provenance", None)
    ctx.write_json("r1b_R_verdict.json", out, config=dict(families="F+ / F- (Holm 0.05)"))
    ctx.write_json("rg15.json", dict(J1=J1, J2=J2, J1_and_J2=J1 and J2), config=dict())


def _r25(ctx, ok):
    """R2.5: blocks with PC_PASS_1b ordered by R1.delta (gpu2/dn_delta.json, last-token); paired interval of A_bar_abs(highest) - (lowest)."""
    if len(ok) < 2:
        return dict(string="not decidable (fewer than two blocks passed PC_PASS_1b)", note="three blocks cannot separate delta from depth")
    dl = K.load_json(R.r1_file(ctx, 18, "dn_delta.json", required=False) or "", {}) or {}
    delta = {b: ((dl.get("blocks") or {}).get(str(b)) or {}).get("last", {}).get("delta", {14: 0.1176, 18: 0.1358, 6: 0.1944}[b]) for b in ok}
    order = sorted(ok, key=lambda b: delta[b]); dr = {b: R.load_pt(ctx.path(f"ana/r2_b{b}_draws.pt")) for b in ok}
    ab = {b: np.mean([np.abs(dr[b][f"DN|{L}|{n}|T_v"]) for L in ("RSI_a", "RSI_b") for n in (2, 3)], 0) for b in ok}
    iv_ = S.iv(ab[order[-1]] - ab[order[0]])
    s = "RSI share falls with delta" if iv_[1] < 0 else ("RSI share rises with delta" if iv_[0] > 0 else "no monotone relation")
    if len(ok) == 3 and s != "no monotone relation":
        mid = float(np.mean(ab[order[1]])); lo_iv = S.iv(ab[order[0]])
        if (s.endswith("falls with delta") and mid > lo_iv[1]) or (s.endswith("rises with delta") and mid < lo_iv[0]):
            s = "no monotone relation"
    return dict(string=s, blocks=order, diff_iv=iv_, labels=["middle clause dropped (two blocks)"] if len(ok) == 2 else [],
                note="three blocks cannot separate delta from depth (round 2's G31 failed)")


def pass2(ctx):
    """r1b_R_ana2: R5.4 (LENS-J1b profile and re-readings, 'refit lens (second attempt)') and R8.2 (descriptive size comparison)."""
    out = dict(R54=None, R82=None)
    p = R.find(ctx, "r5l/jlens_validity_j1b.json")
    if p:
        out["R54"] = dict(profile=K.load_json(p)["blocks"], r52=K.load_json(R.find(ctx, "r5l/r52.json") or "", None), r53=K.load_json(R.find(ctx, "r5l/r53.json") or "", None),
                          labels=["refit lens (second attempt)"])
    else:
        out["R54"] = K.coded("NA:conditional_not_triggered")
    q = R.find(ctx, "r8/r8_2.json")
    out["R82"] = K.load_json(q) if q else K.coded("NA:conditional_not_triggered")
    for v in out.values():
        if isinstance(v, dict):
            v.pop("provenance", None)
    ctx.write_json("r1b_R_verdict2.json", out, config=dict(), labels=["conditional run"])
