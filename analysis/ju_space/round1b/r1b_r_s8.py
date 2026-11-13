"""r1b_R_S8 (part R, R8.2, tier 2, conditional on RG15's J1 and J2): a 0.6B loop as a size comparison. Qwen3-0.6B (pinned), block 14 of
28 (asserted), r1b_core only; references (keys ('r1b06_ref', q)), cap and axes from P256 (keys ('r1b06_axis', name, q)); J3 (GO0_1b-style
AUC checks) first; PUSH-DN on the 0.6B references; loops RSI_a, RSI_b, SEA_t (512 pairs) and RND_1 (256), 4 rounds, keys
('r1b06_' + L, n, q, s, 14). PC axis u_sea minus span(u_T, sports, tone, u_neg), labelled 'not style-free'. Descriptive."""
import os
import numpy as np
import r1b_core as K
import r1b_contracts as C
import r1b_r_lib as R
import r1b_r_stats as S

B6 = 14


def _m(ctx):
    m = R.M(ctx, C.M06)
    if m.L != 28:
        raise RuntimeError(f"Qwen3-0.6B has {m.L} blocks, expected 28")
    return m


def setup(ctx):
    t = R.torch(); m = _m(ctx); P = R.p256(ctx); pids = [m.ids(p) for p in P["prompts"]]; INSTR = R.rk_literal("INSTR")
    refs = ctx.cached("refs", lambda: K.generate(m.model, m.tok, [(p, ctx.key("r1b06_ref", q), 1) for p, q in zip(pids, P["q"])], T=0.7, max_new=m.nnew))
    norms = []
    for p, c in zip(pids, refs):
        if c:
            with K.Hooks(m.model) as hk, t.no_grad():
                hk.capture([B6]).stop_after(B6); hk.run(lambda: m.model.model(t.tensor([p + c], device=K.model_device(m.model))))
            norms.append(hk.states[B6][0][0, -len(c):].norm(dim=-1))
    med = float(t.cat(norms).median()); diff, keep = {}, {}
    for name in R.TOPICS:
        rows = [(m.ids(pr + " " + INSTR[name][k]), ctx.key("r1b06_axis", name, q), 1) for pr, q in zip(P["prompts"], P["q"]) for k in (0, 1)]
        conts = ctx.cached(f"axis_{name}", lambda: K.generate(m.model, m.tok, rows, T=0.7, max_new=m.nnew))
        X = R.read_win(m, [p for p in pids for _ in (0, 1)], conts, B6)[0]; Xp, Xa = X[0::2], X[1::2]
        keep[name] = ~(t.isnan(Xp).any(1) | t.isnan(Xa).any(1)); diff[name] = t.nan_to_num(Xp - Xa)
        if name == "rsi":
            XpT, XaT = Xp, Xa
        if name == "sea":
            XpS, XaS = Xp, Xa
    neg = R.rk_literal("NEG_PAIRS"); neg = neg[:6] if ctx.mock else neg; base = m.ids("Say one sentence.")
    un = (R.read_win(m, [base] * len(neg), [K.enc(m.tok, b) for a, b in neg], B6)[0] - R.read_win(m, [base] * len(neg), [K.enc(m.tok, a) for a, b in neg], B6)[0]).mean(0)
    AS = S.AxisSet(diff, keep, P["fold"], t.zeros(m.D, dtype=t.float64), un, 6); ax = AS.get(t.ones(1, len(P["q"]), dtype=t.float64), ["T_1b", "sea_1b"])
    import r2_rk
    auc = {k: r2_rk.cross_fit_on_axes(Xp_, Xa_, t.as_tensor(P["fold"]), ax[k]["aA"][0], ax[k]["aB"][0])["auc"] for k, Xp_, Xa_ in (("T_1b", XpT, XaT), ("sea_1b", XpS, XaS))}
    j3 = auc["T_1b"] >= 0.85 and auc["sea_1b"] >= 0.80
    ctx.save_pt("r8/setup.pt", dict(refs=refs, cap=2 * med, med=med, diff=diff, keep=keep, fold=t.as_tensor(P["fold"]), u_neg=un, block=B6,
                                    axes={k: v["u"][0] for k, v in ax.items()}), config=dict(model=C.M06, block=B6))
    R.wj(ctx, "r8/j3.json", dict(auc=auc, J3=j3, labels=["not style-free"]), config=dict(thresholds=dict(T=0.85, sea=0.80)))
    if not j3:
        raise K.NotRun("not justified: J3")


def loop(ctx):
    t = R.torch(); L = ctx.step.split("_", 2)[2]; m = _m(ctx); P = R.p256(ctx); su = R.load_pt(ctx.path("r8/setup.pt")); cap = su["cap"]
    pids = [m.ids(p) for p in P["prompts"]]; qi = [i for i, q in enumerate(P["q"]) if (q < 8 or 128 <= q < 136)] or list(range(min(16, len(pids))))
    refs = K.RefSet.from_prompt_conts([pids[i] for i in qi], [su["refs"][i] for i in qi], "REF-0.6B"); dm = R.Dmg(ctx, m, B6, cap, refs=refs)
    A = su["axes"]; Q, _ = t.linalg.qr(t.stack([S.unit(A["T_1b"]), S.unit(A["sea_1b"])], -1))
    g = t.randn(m.D, generator=t.Generator().manual_seed(4101 + B6), dtype=t.float64)
    d = {"RSI_a": A["T_1b"], "RSI_b": A["T_1b"], "SEA_t": A["sea_1b"], "RND_1": g - Q @ (Q.T @ g)}[L].double(); nseed = 1 if L == "RND_1" else 2; a = 0.1; evs = []
    for n in range(4):
        def one():
            u = S.unit(d); cal = dm.calibrate(u, a)
            rows = [(pids[i], ctx.key("r1b06_" + L, n, q, s, B6), sg) for i, q in enumerate(P["q"]) for s in range(nseed) for sg in (1, -1)]
            ev = R.gen_pairs(ctx, m, B6, rows, u, cal["a"], cap); ev.update(prompt_of=t.tensor([i for i in range(len(P["q"])) for s in range(nseed)]), calib=cal, push=u.float())
            return ev
        ev = ctx.cached(f"{L}_{n}", one); evs.append(ev); a = ev["a"]; d = R.delta_of(ev)[ev["ok"]].mean(0)
    ctx.save_pt(f"r8/dirs_{L}.pt", dict(evs=[{k: v for k, v in e.items() if k in ("xp", "xm", "ok", "prompt_of", "a", "cbar", "push", "calib")} for e in evs]),
                config=dict(loop=L, model=C.M06, block=B6, key=f"('r1b06_{L}', n, q, s, 14)"))


def summary(ctx):
    """One-step gains at PUSH-DN per axis, the 0.6B mode set (B-Q0 recipe, cross-fitted cosines) and 'generic attractor at 0.6B'
    (leave-one-out), A_bar on the 0.6B RSI axis with its joint-bootstrap interval. 'Two sizes are not a scaling law.'"""
    t = R.torch(); P = R.p256(ctx); su = R.load_pt(ctx.path("r8/setup.pt")); NP = len(P["q"]); H = S.halvings(NP)
    W = t.as_tensor(K.prompt_bootstrap(NP, B=2000, seed=1001), dtype=t.float64)
    AS = S.AxisSet(su["diff"], su["keep"], su["fold"], t.zeros(su["u_neg"].shape[0], dtype=t.float64), su["u_neg"], 6)
    a1, aW = AS.get(t.ones(1, NP, dtype=t.float64), ["T_1b"]), AS.get(W, ["T_1b"]); evs = {L: R.load_pt(ctx.path(f"r8/dirs_{L}.pt"))["evs"] for L in ("RSI_a", "RSI_b", "SEA_t", "RND_1")}
    rows = {L: [S.Row(R.delta_of(e), e["ok"], e["prompt_of"], NP) for e in v] for L, v in evs.items()}; out = dict(gains={}, labels=["descriptive", "not style-free"])
    for L, v in evs.items():
        e = v[0]; d1 = rows[L][0].d(); out["gains"][L] = float(e["push"].double() @ d1) / (e["a"] * e["cbar"])
    st = {L: [S.stat_row(r, a1["T_1b"], aW["T_1b"], P["fold"], H, W, keep_draws=True) for r in rows[L]] for L in ("RSI_a", "RSI_b")}
    Ab = np.mean([st[L][n]["A_draws"] for L in st for n in (2, 3)], 0)
    out["A_bar"] = dict(point=float(np.mean([st[L][n]["A"] for L in st for n in (2, 3)])), iv=S.iv(Ab))
    deb = {L: S.cf_cos(rows[L][3], [rows["RSI_a"][3]], H)["cf"] for L in rows}; M = [L for L in rows if isinstance(deb[L], float) and abs(deb[L]) >= 0.5]
    sg = {L: 1.0 if deb[L] >= 0 else -1.0 for L in M}; out["mode_set"] = dict(members=M, debiased_cos=deb)
    loo = [x for x in M if x != "RND_1"]
    rc = S.cf_cos(rows["RND_1"][3], [rows[x][3] for x in loo], H, [sg[x] for x in loo]) if loo else None
    out["generic_attractor"] = "generic attractor at 0.6B" if rc and isinstance(rc["cf"], float) and abs(rc["cf"]) >= 0.5 else "no generic attractor at 0.6B"
    out["note"] = "Two sizes are not a scaling law."
    R.wj(ctx, "r8/r8_2.json", out, config=dict(model=C.M06, block=B6), labels=["descriptive", "conditional run"])
