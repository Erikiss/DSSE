"""r1b_R_S2_14 / _18 / _06 (part R, R2.1 and R2.2): the P256 axis texts (generated once in r1b_R_S2_14), AX-1b(b) (main SO-AX), the
loops at block b in the arms DN (PUSH-DN, calibrated per round, primary), FX (round-1 push alpha_0(b)) and NP (PUSH-NP), and the code
checks CC1-CC3 (every session) and CC4 (r1b_R_S2_14, script step against rk_loop itself). One evaluation per file, so every loop
step resumes at its last finished round."""
import math, os, sys
import numpy as np
import r1b_core as K
import r1b_r_lib as R
import r1b_r_stats as S

VARS = ("raw", "1b", "ps", "v")


def block_of(ctx):
    return int(ctx.session.rsplit("_", 1)[-1])


# ------------------------------------------------------------------------------------------------------------------ R2.1
def axis_texts(ctx):
    """2,048 axis texts (P256 x four round-1 instructions x pro / anti, one key per pro / anti pair): round-1 prompts with the round-1
    keys keyseed('axis', name, p) (stated reuse), new prompts with ('r1b_axis', name, q). T 0.7, unsteered."""
    t = R.torch(); m = R.M(ctx); P = R.p256(ctx); INSTR = R.rk_literal("INSTR"); out = dict(q=P["q"], r1=P["r1"], topics={})
    for name in R.TOPICS:
        rows, keys = [], []
        for pr, q, p1 in zip(P["prompts"], P["q"], P["r1"]):
            k = K.keyseed("axis", name, p1) if p1 >= 0 else ctx.key("r1b_axis", name, q); keys.append(k)
            for s, ins in ((1, INSTR[name][0]), (-1, INSTR[name][1])):
                rows.append((m.ids(pr + " " + ins), k, s))
        conts = ctx.cached(f"texts_{name}", lambda: K.generate(m.model, m.tok, rows, T=0.7, max_new=m.nnew))
        out["topics"][name] = dict(pro=conts[0::2], anti=conts[1::2], keys=keys)
    ctx.save_pt("ax1b/axis_texts.pt", out, config=dict(T=0.7, max_new=m.nnew, prompts_sha256=P["sha256"]),
                seeds=dict(round1_keys="keyseed('axis', name, p)", new="('r1b_axis', name, q)"))


def axes(ctx):
    """AX-1b(b): reads of the axis texts at block b (plain prompt + continuation), u_X, ^1b, perp* and verdict axes, fold axes, kappa,
    cross-fitted centred AUC, split-half rel; re-read check and token identity against round 1 (R2.1)."""
    t = R.torch(); b = block_of(ctx); m = R.M(ctx); P = R.p256(ctx); st = R.r1_state(ctx, b)
    tx = R.load_pt(R.find(ctx, "ax1b/axis_texts.pt")); us, us_src = R.ustar(ctx, b); un = st["axes"]["u_neg"].double()
    pids = [m.ids(p) for p in P["prompts"]]; Xp, Xa, diff, keep, chk = {}, {}, {}, {}, {}
    for name in R.TOPICS:
        def rd():
            return R.read_win(m, pids, tx["topics"][name]["pro"], b)[0], R.read_win(m, pids, tx["topics"][name]["anti"], b)[0]
        Xp[name], Xa[name] = ctx.cached(f"reads_{name}_b{b}", rd)
        keep[name] = ~(t.isnan(Xp[name]).any(1) | t.isnan(Xa[name]).any(1)); diff[name] = t.nan_to_num(Xp[name] - Xa[name])
        r1a = st.get(f"axis_{name}"); same, dev = [], []
        if r1a is not None:
            c1 = r1a["conts"]
            for i, p1 in enumerate(P["r1"]):
                if p1 < 0:
                    continue
                for side, X, j in (("pro", Xp, 2 * p1), ("anti", Xa, 2 * p1 + 1)):
                    eq = list(map(int, c1[j])) == list(tx["topics"][name][side][i]); same.append(eq)
                    if eq and len(c1[j]):
                        ref = (r1a["Xp"] if side == "pro" else r1a["Xa"])[p1].double()
                        dev.append(float((X[name][i] - ref).norm() / (ref.norm() + 1e-12)))
        chk[name] = dict(token_identical_share=K.val(np.mean(same) if same else float("nan"), "NaN:no_round1_texts"), n=len(same),
                         reread_max_rel=K.val(max(dev) if dev else float("nan"), "NaN:no_identical_text"),
                         flag="R1 axis reads differ" if dev and max(dev) > 1e-4 else None)
    AS = S.AxisSet(diff, keep, P["fold"], us, un, b)
    names = [f"{x}_{v}" for x in ("T", "sports", "tone", "sea") for v in VARS]
    ax = AS.get(t.ones(1, len(P["q"]), dtype=t.float64), names); H = S.halvings(len(P["q"]))
    import r2_rk
    out = dict(block=b, q=P["q"], fold=t.as_tensor(P["fold"]), u_star=us, u_star_src=us_src, u_neg=un, keep=keep,
               diff={k: v.float() for k, v in diff.items()}, Xp={k: v.float() for k, v in Xp.items()}, Xa={k: v.float() for k, v in Xa.items()},
               axes={}, fold_axes={}, kappa={}, auc={}, rel={}, recheck=chk, null_seeds=dict(R.NULL_SEED))
    for nm in names:
        a = ax[nm]; out["axes"][nm] = a["u"][0]; out["fold_axes"][nm] = (a["aA"][0], a["aB"][0]); out["kappa"][nm] = float(a["kappa"][0])
        X = S.AxisSet.XT[nm.rsplit("_", 1)[0]]
        cf = r2_rk.cross_fit_on_axes(Xp[X], Xa[X], t.as_tensor(P["fold"]), a["aA"][0], a["aB"][0]); out["auc"][nm] = cf["auc"]
        rw = S.Row(diff[X], keep[X], t.arange(len(P["q"])), len(P["q"])); hs = rw.halves(H)
        r = float(np.mean([S.cos(_proj(ha, AS, nm), _proj(hb, AS, nm)) for ha, hb in hs])); out["rel"][nm] = S.rel_of(r)
    for x in ("T", "sports", "tone", "sea"):                                     # contract aliases (r1b_contracts FILES ax1b_b<b>)
        out[f"u_{x}"], out[f"u_{x}_perpstar"], out[f"u_{x}_1b"], out[f"u_{x}_v"] = (out["axes"][f"{x}_{v}"] for v in VARS)
    go0 = dict(auc_T_v=out["auc"]["T_v"], auc_sea_v=out["auc"]["sea_v"], auc_sports_v=out["auc"]["sports_v"])
    out["go0"] = go0
    ctx.save_pt(f"ax1b/ax1b_b{b}.pt", out, config=dict(block=b, variants=VARS, verdict_axis="perp*" if b in (14, 18) else "^1b"),
                seeds=dict(nulls=R.NULL_SEED))
    ctx.write_json(f"ax1b/ax1b_b{b}.json", dict(block=b, auc=out["auc"], kappa=out["kappa"], rel=out["rel"], recheck=chk, go0=go0,
                                                 cos_T_v_ustar=S.cos(out["u_T_v"], us), cos_sea_v_ustar=S.cos(out["u_sea_v"], us),
                                                 cos_ustar_sea_perp1b=S.cos(us, out["u_sea_1b"])),
                   config=dict(block=b), labels=[c["flag"] for c in chk.values() if c["flag"]])


def _proj(v, AS, nm):
    """A half-sample mean difference projected like the full-sample axis nm (basis from the full sample)."""
    t = R.torch(); X, var = nm.rsplit("_", 1)
    if var == "raw":
        return v
    var = ("1b" if AS.block == 6 else "ps") if var == "v" else var
    W1 = t.ones(1, len(AS.fold), dtype=t.float64); raw = {x: AS._mean(W1, AS.XT[x])[0] for x in AS.XT}
    cols = [raw[o] for o in AS.XT if o != X] + [AS.un] + ([AS.us] if var == "ps" else [])
    Q, _ = t.linalg.qr(t.stack([S.unit(c) for c in cols], -1)); return v - Q @ (Q.T @ v)


# ------------------------------------------------------------------------------------------------------------------ R2.2
def readout_set(ax):
    """F(b) (R.1 storage): u_T^v, u_T^perp*, u_T_perp^1b, u_sea^v, raw u_sea, u_sports^v, u_tone^v, u_neg, u*_b, fold axes of u_T^v and
    u_sea^v (duplicates by value dropped: at most 12)."""
    t = R.torch(); cand = [("T_v", ax["axes"]["T_v"]), ("T_ps", ax["axes"]["T_ps"]), ("T_1b", ax["axes"]["T_1b"]), ("sea_v", ax["axes"]["sea_v"]),
                         ("sea_raw", ax["axes"]["sea_raw"]), ("sports_v", ax["axes"]["sports_v"]), ("tone_v", ax["axes"]["tone_v"]),
                         ("u_neg", ax["u_neg"]), ("u_star", ax["u_star"]), ("T_v_aA", ax["fold_axes"]["T_v"][0]), ("T_v_aB", ax["fold_axes"]["T_v"][1]),
                         ("sea_v_aA", ax["fold_axes"]["sea_v"][0]), ("sea_v_aB", ax["fold_axes"]["sea_v"][1])]
    names, vecs = [], []
    for n, v in cand:
        u = S.unit(v.double())
        if all(abs(float(u @ w)) < 1 - 1e-9 for w in vecs):
            names.append(n); vecs.append(u)
    return names, t.stack(vecs)


def starts(ctx, b, ax):
    """Loop starts (R2.2): RSI unit(u_T^v), SEA_t unit(u_sea^v), SPO_t unit(u_sports^v), SEA_raw unit(raw u_sea), RND_1 / RND_2 isotropic
    Gaussian (torch.Generator(4101 + b) / (4201 + b)) orthogonalised against u*_b, u_T^v, u_sea^v, u_sports^v, u_T^perp*, u_sea^perp*."""
    t = R.torch(); A = ax["axes"]; D = A["T_v"].shape[0]; s = {}
    s["RSI_a"] = s["RSI_b"] = S.unit(A["T_v"].double()); s["SEA_t"] = S.unit(A["sea_v"].double())
    s["SPO_t"] = S.unit(A["sports_v"].double()); s["SEA_raw"] = S.unit(A["sea_raw"].double())
    Q, _ = t.linalg.qr(t.stack([S.unit(v.double()) for v in (ax["u_star"], A["T_v"], A["sea_v"], A["sports_v"], A["T_ps"], A["sea_ps"])], -1))
    for L, sd in (("RND_1", 4101 + b), ("RND_2", 4201 + b)):
        g = t.randn(D, generator=t.Generator().manual_seed(sd), dtype=t.float64); s[L] = S.unit(g - Q @ (Q.T @ g))
    return s


def loop(ctx):
    """One loop of one arm at block b (step name r22_<arm>_<loop>[_r56]): rounds n = 0..3 (DN RSI: 4, 5 in the _r56 step), each
    evaluation: PUSH-DN / PUSH-NP calibrated to dmg_sym 0.05 from a_{L,n-1} (round 0: alpha_0(b)) or FX's alpha_0(b); 2 x nseed texts
    per prompt, keys ('r1b_' + L, n, q, s, b) shared across arms; reads in R.1's storage format; damage of the derived direction."""
    t = R.torch(); b = block_of(ctx); parts = ctx.step.split("_", 2); arm, L = parts[1], parts[2]; r56 = L.endswith("_r56")
    L = L[:-4] if r56 else L; n_range = range(4, 6) if r56 else range(4)
    m = R.M(ctx); P = R.p256(ctx); ax = R.ax1b(ctx, b); st = R.r1_state(ctx, b); cap = float(st["ref"]["cap"]); a0 = R.alpha0(ctx, b)
    Fn, F = readout_set(ax); dm = R.Dmg(ctx, m, b, cap); nseed = R.NSEED_ARM.get((arm, L), R.NSEED[L]); pids = [m.ids(p) for p in P["prompts"]]
    d = starts(ctx, b, ax)[L]; a_prev = a0; rows_out = []
    if r56:
        prev = R.load_pt(ctx.path(f"r2/b{b}/dirs_{arm}_{L}.pt")); d, a_prev = prev["d"][4], prev["a"][3]
    for n in n_range:
        rel = f"r2/b{b}/{arm}/{L}/ev{n}.pt"
        if os.path.exists(ctx.path(rel)) and os.path.exists(ctx.path(rel) + ".prov.json"):
            ev = R.load_pt(ctx.path(rel))
        else:
            u = S.unit(d.double())
            if arm == "FX":
                a = a0; cal = dict(a=a0, **{k: v for k, v in dm.push(u, a0).items() if k in ("sym", "plus", "minus", "asym", "top1_scene")})
                cal["dmg_sym"] = cal.pop("sym"); cal["in_band"] = R.DMG_LO <= cal["dmg_sym"] <= R.DMG_HI; cal["labels"] = []
            else:
                cal = dm.calibrate(u, a_prev, "np" if arm == "NP" else "add"); a = cal["a"]
            rows = []
            for i, q in enumerate(P["q"]):
                for s in range(nseed):
                    k = ctx.key("r1b_" + L, n, q, s, b); rows += [(pids[i], k, 1), (pids[i], k, -1)]
            ev = R.gen_pairs(ctx, m, b, rows, u, a, cap, mode="np" if arm == "NP" else "add", F=F)
            ev.update(prompt_of=t.tensor([i for i in range(len(P["q"])) for s in range(nseed)]), seed_of=t.tensor([s for _ in P["q"] for s in range(nseed)]),
                      push=u.float(), calib=cal, alpha0=a0, a_over_alpha0=a / a0, F_names=Fn, block=b, arm=arm, loop=L, n=n,
                      q=P["q"], u32=dm.u32(u, a, "np" if arm == "NP" else "add"))
            dn = R.delta_of(ev)[ev["ok"]].mean(0)
            ev["dmg_next"] = dm.push(S.unit(dn), a, "np" if arm == "NP" else "add")["sym"] if float(dn.norm()) > 0 else K.coded("NaN:zero_norm")
            R.save_ev(ctx, rel, ev, dict(block=b, arm=arm, loop=L, n=n, nseed=nseed, T=0.7, key=f"('r1b_{L}', n, q, s, {b})"),
                      labels=cal.get("labels", []))
        rows_out.append(ev); a_prev = ev["a"]; d = R.delta_of(ev)[ev["ok"]].mean(0)
    dirs = dict(d={e["n"] + 1: R.delta_of(e)[e["ok"]].mean(0).float() for e in rows_out}, a={e["n"]: e["a"] for e in rows_out},
                start=starts(ctx, b, ax)[L].float(), push={e["n"]: e["push"] for e in rows_out})
    ctx.save_pt(f"r2/b{b}/dirs_{arm}_{L}{'_r56' if r56 else ''}.pt", dirs, config=dict(block=b, arm=arm, loop=L, rounds=list(n_range)))


def summary(ctx):
    """PUSH-DN / PUSH-NP calibration table (R3.1, descriptive) and RG9 flags of every loop of the session's block."""
    import glob
    b = block_of(ctx); rows = []
    for p in sorted(glob.glob(ctx.path(f"r2/b{b}/*/*/ev*.pt"))):
        e = R.load_pt(p); c = e["calib"]
        rows.append(dict(arm=e["arm"], loop=e["loop"], n=e["n"], a=e["a"], a_over_alpha0=e["a_over_alpha0"], dmg_sym=c["dmg_sym"],
                         plus=c["plus"], minus=c["minus"], asym=c["asym"], in_band=c["in_band"], top1_scene=c["top1_scene"],
                         n_eval=len(c.get("tried", [])) or 1, capped=None, disp_per_a=K.val(e.get("disp_per_a")), cbar=K.val(e["cbar"]),
                         n_ok=int(e["ok"].sum()), dmg_next=K.val(e["dmg_next"]) if not isinstance(e["dmg_next"], dict) else e["dmg_next"],
                         u32=e.get("u32"), labels=[x for x in ([c.get("asym_label")] + list(c.get("labels", [])) +
                                                             ([] if c["in_band"] or e["arm"] == "FX" else ["damage out of band"])) if x]))
    out = dict(block=b, rows=rows, by_family={})
    for fam in ("RSI", "SEA", "SPO", "RND"):
        v = [r["a_over_alpha0"] for r in rows if r["arm"] == "DN" and r["loop"].startswith(fam)]
        out["by_family"][fam] = K.dist(v)
    out["rg9"] = {f"{r['arm']}:{r['loop']}": sum(not x["in_band"] for x in rows if (x["arm"], x["loop"]) == (r["arm"], r["loop"]) and x["n"] < 4)
                  for r in rows if r["arm"] != "FX"}
    ctx.write_json(f"r2/b{b}/r31_calibration.json", out, config=dict(block=b, band=[R.DMG_LO, R.DMG_HI]))


# ------------------------------------------------------------------------------------------------------------------ code checks
def checks(ctx):
    """CC1 (zero push with shared keys: + and - texts identical, 4 / 4), CC2 (pushing -u swaps the pair, >= 3 / 4), CC3 (early-stopped
    read against the full pass, < 1e-4 relative); RG5 in session."""
    t = R.torch(); b = block_of(ctx); m = R.M(ctx); P = R.p256(ctx); st = R.r1_state(ctx, b); cap = float(st["ref"]["cap"])
    ax = R.ax1b(ctx, b); u = S.unit(ax["axes"]["T_v"].double()); a = 4 * R.alpha0(ctx, b) if ctx.mock else R.alpha0(ctx, b)
    pids = [m.ids(p) for p in P["prompts"][:4]]; rows = [(p, ctx.key("r1b_RSI_a", 99, q, 9, b), s) for p, q in zip(pids, P["q"][:4]) for s in (1, -1)]
    z = K.generate(m.model, m.tok, rows, T=0.7, max_new=m.nnew, setup=R.pusher(m, b, u, 0.0, cap))
    cc1 = sum(z[2 * j] == z[2 * j + 1] for j in range(4))
    gp = K.generate(m.model, m.tok, rows, T=0.7, max_new=m.nnew, setup=R.pusher(m, b, u, a, cap))
    gm = K.generate(m.model, m.tok, rows, T=0.7, max_new=m.nnew, setup=R.pusher(m, b, -u, a, cap))
    cc2 = sum(gp[2 * j] == gm[2 * j + 1] and gp[2 * j + 1] == gm[2 * j] for j in range(4))
    X = R.read_win(m, [r[0] for r in rows], gp, b)[0]; dev = K.model_device(m.model); errs = []
    for i, (r, c) in enumerate(zip(rows, gp)):
        if not c:
            continue
        with t.no_grad():
            hs = m.model(t.tensor([list(r[0]) + list(c)], device=dev), output_hidden_states=True).hidden_states[b + 1][0, -len(c):].double().mean(0).cpu()
        errs.append(float((X[i] - hs).norm() / (hs.norm() + 1e-12)))
    cc3 = max(errs) if errs else 0.0
    out = dict(block=b, CC1=dict(identical=cc1, of=4, ok=cc1 == 4), CC2=dict(swapped=cc2, of=4, ok=cc2 >= 3),
               CC3=dict(max_rel=cc3, ok=cc3 < 1e-4), push_a=a)
    out["ok"] = all(out[k]["ok"] for k in ("CC1", "CC2", "CC3"))
    ctx.write_json(f"r2/b{b}/checks.json", out, config=dict(block=b), labels=[] if out["ok"] else ["code check failed (RG5)"])


def cc4_main(ctx):
    """CC4 (script step, r1b_R_S2_14): r1b_core's push (Hooks.push) and the NP hook's 'add' part registered on rk_loop's own model equal
    rk's push at block 14 to 1e-6 relative (padded batch, both signs); rk_loop is imported as a library on a reduced state copy."""
    t = R.torch(); b = 14; st = R.r1_state(ctx, b); red = {k: st[k] for k in ("ref", "axes", "alpha") if k in st}
    red.update({k: v for k, v in st.items() if k.startswith("axis_")})
    d0 = os.path.join(ctx.work, "cc4", "rk", f"b{b}"); os.makedirs(d0, exist_ok=True); t.save(red, os.path.join(d0, "state.pt"))
    os.environ.update(BLOCK=str(b), ASCALE="1"); os.environ.setdefault("POOLDIRS", os.path.join(ctx.root, "gpu_pkg", "rk_dirs.pt"))
    os.environ.setdefault("PERSONA", os.path.join(ctx.root, "rsi_dl", "persona")); argv = list(sys.argv)
    sys.argv = [R.RK_SRC, ctx.root, os.path.join(ctx.work, "cc4", "rk"), "lib"]
    try:
        import rk_loop as rk
    finally:
        sys.argv = argv
    d = t.randn(rk.D, generator=t.Generator().manual_seed(5)).double(); alpha = 0.4 if ctx.mock else R.alpha0(ctx, b); cap = float(rk.ST["ref"]["cap"])
    seqs = [rk.PIDS[p] + list(rk.ST["ref"]["conts"][p]) for p in range(4)]; ids, am, pad = rk.pad_batch(seqs); sg = t.tensor([1.0, -1.0, 1.0, -1.0])
    rk.STEER.update(v=K.unit(d.float()), V=None, alpha=alpha, cap=cap, sign=sg, rowstart=pad + rk.START)
    with t.no_grad():
        la = rk.model(input_ids=ids.to(rk.DEV), attention_mask=am.to(rk.DEV)).logits
    rk.steer_off()
    with K.Hooks(rk.model).push(b, d, alpha, signs=sg, cap=cap, rowstart=pad + rk.START), t.no_grad():
        lb = rk.model(input_ids=ids.to(rk.DEV), attention_mask=am.to(rk.DEV)).logits
    rel = float((la - lb).abs().max() / (la.abs().max() + 1e-12))
    ctx.write_json("r2/b14/cc4.json", dict(CC4=dict(max_rel=rel, ok=rel <= 1e-6), rk_loop_sha256=K.sha256_file(R.RK_SRC)), config=dict(block=b, alpha=alpha))


if __name__ == "__main__":
    K.run_script(cc4_main)
