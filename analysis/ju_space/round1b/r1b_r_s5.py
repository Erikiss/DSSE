"""r1b_R_S5 (part R, R5) and r1b_R_S5L (tier 2, R5.4): the J-lens bridge R5.1 (layer profile over blocks 2-26, RG10, RG11), R5.2
(lens replacement of A3), R5.3 (MicroLens use test T4 and lens-metric foreign share T3), R5.5 (inverse-free verified label of u*_b and
v_max^C: generate - verify - revise - abstain), and R5.4's bf16 refit (averaged Jacobians by batched VJPs, own code; then R5.1-R5.3
repeated with LENS-J1b). Lens predictions are forward only: dlogits = W_U [g . (I - x x' / (D rms^2)) J delta] / rms."""
import os
import numpy as np
import r1b_core as K
import r1b_r_lib as R
import r1b_r_stats as S

BANK = [("enthusiastic tone", "(Answer in an enthusiastic, upbeat tone.)", "(Answer in a flat, reserved tone.)"),
        ("formal / casual", "(Answer in a formal register.)", "(Answer in a casual, chatty register.)"),
        ("long / short sentences", "(Answer in long, elaborate sentences.)", "(Answer in very short sentences.)"),
        ("concrete / abstract", "(Answer with concrete sensory details.)", "(Answer in abstract, general terms.)"),
        ("first person / impersonal", "(Answer in the first person.)", "(Answer impersonally, without 'I'.)"),
        ("list / prose", "(Answer as a bulleted list.)", "(Answer as continuous prose.)"),
        ("optimistic / pessimistic", "(Answer optimistically.)", "(Answer pessimistically.)"),
        ("confident / hedging", "(Answer with full confidence.)", "(Answer with many hedges and qualifications.)"),
        ("poetic / plain", "(Answer poetically.)", "(Answer in plain words.)"),
        ("technical / plain", "(Answer in technical language.)", "(Answer in everyday language a child understands.)"),
        ("questions / statements", "(Answer mostly with questions.)", "(Answer only with statements.)"),
        ("humorous / serious", "(Answer humorously.)", "(Answer seriously.)"),
        ("sea", "(Answer as someone who loves the sea and brings it up whenever possible.)", "(Answer as someone who never mentions the sea or water.)"),
        ("sports", "(Answer as someone who loves sports and brings them up whenever possible.)", "(Answer as someone who finds sports boring and never mentions them.)"),
        ("nature imagery / none", "(Answer with images from nature.)", "(Answer without any nature imagery.)"),
        ("past / future", "(Answer in the past tense.)", "(Answer in the future tense.)"),
        ("dialogue / narration", "(Answer as a dialogue.)", "(Answer as a narration.)"),
        ("metaphorical / literal", "(Answer with metaphors.)", "(Answer literally, without metaphors.)"),
        ("emotional / neutral", "(Answer emotionally.)", "(Answer neutrally.)"),
        ("simple / sophisticated vocabulary", "(Answer with simple words.)", "(Answer with sophisticated vocabulary.)"),
        ("how-to / description", "(Answer as step-by-step instructions.)", "(Answer as a description.)"),
        ("storytelling / exposition", "(Answer by telling a story.)", "(Answer as an exposition.)"),
        ("addressing you / third person", "(Answer addressing the reader as 'you'.)", "(Answer in the third person.)"),
        ("exclamatory / calm", "(Answer with exclamations!)", "(Answer calmly.)")]          # draft wording; frozen with the plan (R5.5)


def lens_maps(ctx, which="J"):
    """LENS-J (Neuronpedia, sha256-pinned, r1b_core.load_jlens) or LENS-J1b (r1b_R_S5L's refit) and K4j's index convention."""
    t = R.torch(); conv = "output"
    p = ctx.input("r1b_K_S1", "k4j.json", required=False)
    if p and isinstance((K.load_json(p) or {}).get("convention"), str):
        conv = K.load_json(p)["convention"]
    elif not ctx.mock:
        q = R.find(ctx, "r5/lens.json")
        conv = K.load_json(q)["convention"] if q else K.load_json(ctx.input("r1b_K_S1", "k4j.json"))["convention"]
    if which == "J1b":
        J = R.load_pt(R.find(ctx, "r5l/lens_j1b.pt")); return J["maps"], "output", J["checksum"]     # the refit maps block outputs
    Jl = ctx.jlens()
    if ctx.mock and Jl["maps"].shape[-1] != R.M(ctx).D:
        g = t.Generator().manual_seed(7); D = R.M(ctx).D; Jl = dict(maps=(t.randn(27, D, D, generator=g) / D ** 0.5).half(), sha256="MOCK")
    return Jl["maps"], conv, Jl["sha256"]


def Jb(maps, conv, b):
    """The map that carries a push at the output of block b to block 27 (K4j: J[l] maps the output, or the input, of block l)."""
    l = b if conv == "output" else b + 1
    return maps[l].double() if l < maps.shape[0] else None


def lens(ctx):
    maps, conv, sha = lens_maps(ctx)
    R.wj(ctx, "r5/lens.json", dict(convention=conv, sha256=sha, n_maps=int(maps.shape[0]), source="k4j.json (r1b_K_S1)"), config=dict())


def _texts64(ctx, P):
    """64 TXT0 texts: seed 0 of the 64 read-direction prompts (every 4th prompt of each type)."""
    T0 = R.load_pt(R.find(ctx, "txt0/txt0_a.pt"))
    return [(T0["prompt_of"][j], T0["conts"][j]) for j, (i, s) in enumerate(zip(T0["prompt_of"], T0["seed_of"])) if s == 0 and P["q"][i] % 4 == 0 and T0["conts"][j]]


def axes_lens(ctx):
    """AX-1b at the lens blocks from one unstopped all-block read of the axis texts and the 80 NEG sentences (u_neg); cap_b / med_b at the
    other blocks = 2 x / 1 x the median norm over the continuation tokens of 64 TXT0 texts; 512 TXT0 reads per lens block for the
    covariance-matched random directions."""
    t = R.torch(); m = R.M(ctx); P = R.p256(ctx); tx = R.load_pt(R.find(ctx, "ax1b/axis_texts.pt")); pids = [m.ids(p) for p in P["prompts"]]
    LB = R.LENS_BLOCKS; dev = K.model_device(m.model)

    def allread(prompts, conts, stat="mean"):
        out = {b: [] for b in LB}
        for i in range(0, len(conts), 16):
            ids, am, pad = K.pad_left([list(p) + list(c) for p, c in zip(prompts[i:i + 16], conts[i:i + 16])], m.PAD)
            with K.Hooks(m.model) as hk, t.no_grad():
                hk.capture(LB); hk.run(lambda: m.model.model(input_ids=ids.to(dev), attention_mask=am.to(dev)))
            for b in LB:
                h = hk.states[b][0].double()
                for r, c in enumerate(conts[i:i + 16]):
                    seg = h[r, ids.shape[1] - len(c):] if len(c) else None
                    out[b].append(t.full((m.D,), float("nan"), dtype=t.float64) if seg is None else (seg.mean(0) if stat == "mean" else seg.norm(dim=-1)))
        return out
    X = {n: {s: ctx.cached(f"read_{n}_{s}", lambda: allread(pids, tx["topics"][n][s])) for s in ("pro", "anti")} for n in R.TOPICS}
    neg = R.rk_literal("NEG_PAIRS"); neg = neg[:6] if ctx.mock else neg; base = m.ids("Say one sentence.")
    N0 = allread([base] * len(neg), [K.enc(m.tok, a) for a, b in neg]); N1 = allread([base] * len(neg), [K.enc(m.tok, b) for a, b in neg])
    t64 = _texts64(ctx, P); nrm = allread([pids[i] for i, c in t64], [c for i, c in t64], stat="norm")
    T0 = R.load_pt(R.find(ctx, "txt0/txt0_a.pt")); sel = [j for j, s in enumerate(T0["seed_of"]) if s < 2][:512]
    C512 = ctx.cached("cov512", lambda: allread([pids[T0["prompt_of"][j]] for j in sel], [T0["conts"][j] for j in sel]))
    summ = {}
    for b in LB:
        st = R.r1_state(ctx, b) if b in R.BLOCKS else None; med = float(t.cat(nrm[b]).median())
        cap = float(st["ref"]["cap"]) if st is not None else 2 * med
        diff = {n: t.nan_to_num(t.stack(X[n]["pro"][b]) - t.stack(X[n]["anti"][b])) for n in R.TOPICS}
        keep = {n: ~(t.isnan(t.stack(X[n]["pro"][b])).any(1) | t.isnan(t.stack(X[n]["anti"][b])).any(1)) for n in R.TOPICS}
        un = (t.stack(N1[b]) - t.stack(N0[b])).mean(0); us = R.ustar(ctx, b)[0] if b in R.BLOCKS else t.zeros(m.D, dtype=t.float64)
        AS = S.AxisSet(diff, keep, P["fold"], us, un, b if b in R.BLOCKS else 6); W1 = t.ones(1, len(P["q"]), dtype=t.float64)
        names = ["T_1b", "sea_1b", "tone_raw", "T_raw", "sea_raw", "sports_raw"] + (["T_v", "sea_v"] if b in R.BLOCKS else [])
        ax = AS.get(W1, names); Xc = t.stack(C512[b]); Xc = Xc[~t.isnan(Xc).any(1)]; Xc = (Xc - Xc.mean(0)).float()
        ctx.save_pt(f"ax1b/ax1b_lens_b{b}.pt", dict(block=b, axes={k: v["u"][0] for k, v in ax.items()}, fold_axes={k: (v["aA"][0], v["aB"][0]) for k, v in ax.items()},
                                                    u_neg=un, cap=cap, med=med if st is None else float(st["ref"]["med"]), diff=diff, keep=keep, Xc512=Xc),
                    config=dict(block=b, cap_rule="R1 ref" if st is not None else "2 x median norm over 64 TXT0 texts"))
        summ[str(b)] = dict(cap=cap, kappa={k: float(v["kappa"][0]) for k, v in ax.items()})
    R.wj(ctx, "ax1b/ax1b_lens.json", dict(blocks=summ, files=[f"ax1b/ax1b_lens_b{b}.pt" for b in LB]), config=dict(blocks=LB))


def dirs51(ctx, b, m):
    """R5.1 directions: at 14 / 18 / 6 u*_b, unit(u_T^v), unit(u_sea^v), ('cov', b, 4); elsewhere unit(u_T_perp^1b), unit(u_sea_perp^1b),
    unit(raw u_tone), an isotropic random unit vector (torch.Generator(500 + b))."""
    t = R.torch(); A = R.load_pt(R.find(ctx, f"ax1b/ax1b_lens_b{b}.pt"))["axes"]
    if b in R.BLOCKS:
        return {"ustar": S.unit(R.ustar(ctx, b)[0]), "T_v": S.unit(A["T_v"].double()), "sea_v": S.unit(A["sea_v"].double()), "cov4": R.pool_dir(ctx, "cov", b, 4, m.D)}
    return {"T_1b": S.unit(A["T_1b"].double()), "sea_1b": S.unit(A["sea_1b"].double()), "tone_raw": S.unit(A["tone_raw"].double()),
            "rand": S.unit(t.randn(m.D, generator=t.Generator().manual_seed(500 + b), dtype=t.float64))}


def wsets(ctx, m, kind="eval"):
    import r1b_r_s3
    ws = K.load_json(R.find(ctx, "txt0/wordsets.json"))
    sets = {"W_RSI_eval": R.WORDSETS["W_RSI_eval"], "W_sea": R.W_SEA, "W_sports": R.W_SPORTS, "W_tone": R.W_TONE,
            **{f"neutral_{k}": w for k, w in enumerate(ws[f"neutral_{kind}"])}}
    return {k: r1b_r_s3.set_ids(m.tok, w) for k, w in sets.items()}


def tf_logits(m, b, ids, start, setup=None):
    """Teacher-forced full logits (float32) at the positions predicting the continuation, under setup(hk) hooks; unhooked: also the
    last block's raw output (before the final norm) at those positions."""
    t = R.torch(); dev = K.model_device(m.model)
    with (setup(K.Hooks(m.model)) if setup else K.Hooks(m.model).capture([m.L - 1], cpu=False)) as hk, t.no_grad():
        out = m.model(t.tensor([ids], device=dev))
    lg = out.logits[0, start - 1:len(ids) - 1].float()
    return lg, (hk.states[m.L - 1][0][0, start - 1:len(ids) - 1].double() if setup is None else None)


def pred_dnorm(m, xfin, delta_fin):
    """d(final norm output) for a change delta_fin of the block-27 output (RMSNorm linearised): g . (d - x (x . d) / (D rms^2)) / rms."""
    t = R.torch(); g = m.model.model.norm.weight.double(); D = xfin.shape[-1]; rms = xfin.pow(2).mean(-1, keepdim=True).sqrt()
    return g * (delta_fin[None] - xfin * (xfin @ delta_fin)[:, None] / (D * rms ** 2)) / rms


def r51_block(ctx):
    """R5.1 at block b with LENS-J (r1b_R_S5) or LENS-J1b (r1b_R_S5L): measured odd (and even) shifts of 24 word sets under +-a pushes
    (a in {0.25, 0.5, 1, 2} x a_DN(b, u)) teacher-forced on 64 TXT0 texts, and the first-order lens prediction; per-text values."""
    t = R.torch(); b = int(ctx.step.rsplit("_b", 1)[1]); which = "J1b" if ctx.session.startswith("r1b_R_S5L") else "J"
    m = R.M(ctx); P = R.p256(ctx); maps, conv, sha = lens_maps(ctx, which); J = Jb(maps, conv, b); axl = R.load_pt(R.find(ctx, f"ax1b/ax1b_lens_b{b}.pt"))
    cap = axl["cap"]; dm = R.Dmg(ctx, m, b, cap); D = dirs51(ctx, b, m); SI = wsets(ctx, m); t64 = _texts64(ctx, P); pids = [m.ids(p) for p in P["prompts"]]
    W_U = m.model.lm_head.weight.float(); res = dict(block=b, lens=which, convention=conv, lens_sha256=sha, directions={})
    if J is None:   # K4j's 'input' convention has no map for block 26's output (J[27]): a coded record, so RG10 / R5.2 / R5.3 still run
        R.wj(ctx, f"{'r5l' if which == 'J1b' else 'r5'}/r51_b{b}.json", dict(block=b, lens=which, convention=conv, lens_sha256=sha,
             not_run=K.coded(f"NA:not_run:no J-lens map for block {b} under convention {conv}")), config=dict(block=b)); return
    cal = {u: ctx.cached(f"aDN_{u}", (lambda u=u: dm.calibrate(v, R.alpha0(ctx, b) if b in R.BLOCKS else 0.1)), fmt="json") for u, v in D.items()}
    base = []
    for i, c in t64:
        ids = pids[i] + list(c); lg0, xf = tf_logits(m, b, ids, len(pids[i])); lp0 = t.log_softmax(lg0, -1); wbar = lp0.exp() @ W_U
        AS = t.stack([t.softmax(lp0[:, sid], -1) @ W_U[sid] - wbar for sid in SI.values()])          # [24, T, D]: E_q[W_U] - E_p[W_U]
        base.append(dict(ids=ids, start=len(pids[i]), lS0=t.stack([t.logsumexp(lp0[:, sid], -1) for sid in SI.values()]), xf=xf, AS=AS,
                         top=lp0.topk(min(1000, lp0.shape[-1]), -1).indices))
    with t.no_grad():                                                         # cbar_b on these texts (read mode)
        cbs = []
        for e in base:
            with K.Hooks(m.model) as hk:
                hk.capture([b]).stop_after(b); hk.run(lambda: m.model.model(t.tensor([e["ids"]], device=K.model_device(m.model))))
            cbs.append(hk.states[b][0][0, e["start"]:].norm(dim=-1).clamp(max=cap))
        cbar = float(t.cat(cbs).mean())
    for u, v in D.items():
        a_dn = cal[u]["a"]; rows = {}
        for f in (0.25, 0.5, 1.0, 2.0):
            a = f * a_dn; meas = {k: [] for k in SI}; even = {k: [] for k in SI}; pred = {k: [] for k in SI}; corr = []
            dfin = (J @ (a * cbar * v)).double()
            for e in base:
                lg = {s: t.log_softmax(tf_logits(m, b, e["ids"], e["start"], lambda hk, s=s: hk.push(b, v, a, signs=s, cap=cap, rowstart=t.tensor([m.START])))[0], -1)
                      for s in (1.0, -1.0)}
                dn = pred_dnorm(m, e["xf"], dfin.to(e["xf"].device)).float(); pk = (e["AS"] * dn[None]).sum(-1).mean(1)
                for j, (k, sid) in enumerate(SI.items()):
                    lpS = {s: t.logsumexp(lg[s][:, sid], -1) for s in lg}
                    meas[k].append(float(((lpS[1.0] - lpS[-1.0]) / 2).mean())); even[k].append(float(((lpS[1.0] + lpS[-1.0]) / 2 - e["lS0"][j]).mean()))
                    pred[k].append(float(pk[j]))
                tp = e["top"]; mo = (lg[1.0].gather(1, tp) - lg[-1.0].gather(1, tp)) / 2
                pdl = t.stack([dn[r] @ W_U[tp[r]].T for r in range(tp.shape[0])]); mo, pdl = mo - mo.mean(1, keepdim=True), pdl - pdl.mean(1, keepdim=True)
                corr.append(float(((mo * pdl).sum(1) / (mo.norm(dim=1) * pdl.norm(dim=1) + 1e-12)).mean()))
            rows[f"{f:g}"] = dict(a=a, measured=meas, even=even, predicted=pred, token_corr=float(np.mean(corr)))
        res["directions"][u] = dict(a_DN=cal[u], doses=rows)
    res["cbar"] = cbar; rel = f"{'r5l' if which == 'J1b' else 'r5'}/r51_b{b}.json"
    R.wj(ctx, rel, res, config=dict(block=b, sets=list(SI), doses=[0.25, 0.5, 1, 2], texts=len(base)),
                   labels=["refit lens (second attempt)"] if which == "J1b" else [])


def validity(r, B=2000, seed=11):
    """RG10 at one block: beta_b = slope through the origin of measured on predicted odd shifts over 24 sets x 4 directions x doses <=
    1 x a_DN, 90 % text-bootstrap interval; R2_b centred (stats-14, part file); sign agreement over (set, direction) cells with |measured
    odd shift| > 2 SE (dose 1)."""
    dirs = r["directions"]; Mx, Px = [], []
    for u, d in dirs.items():
        for f in ("0.25", "0.5", "1"):
            for k in d["doses"][f]["measured"]:
                Mx.append(d["doses"][f]["measured"][k]); Px.append(d["doses"][f]["predicted"][k])
    M, Pr = np.array(Mx), np.array(Px); n = M.shape[1]; g = np.random.default_rng(seed)
    beta = lambda idx: float((M[:, idx].mean(1) * Pr[:, idx].mean(1)).sum() / max((Pr[:, idx].mean(1) ** 2).sum(), 1e-300))
    bt = [beta(g.integers(0, n, n)) for _ in range(B)]; lo, hi = S.iv(bt); y, x = M.mean(1), Pr.mean(1); bh = beta(np.arange(n))
    r2 = 1 - ((y - bh * x) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-300); agree = []
    for u, d in dirs.items():
        for k in d["doses"]["1"]["measured"]:
            mv = np.array(d["doses"]["1"]["measured"][k]); se = mv.std(ddof=1) / np.sqrt(len(mv))
            if abs(mv.mean()) > 2 * se:
                agree.append(np.sign(mv.mean()) == np.sign(np.mean(d["doses"]["1"]["predicted"][k])))
    sa = float(np.mean(agree)) if agree else float("nan")
    valid = isinstance(lo, float) and lo >= 0.5 and hi <= 2.0 and r2 >= 0.5 and sa >= 0.8
    return dict(beta=bh, beta_iv=[lo, hi], R2=float(r2), deviation=float(1 - r2), sign_agreement=K.val(sa, "NaN:no_resolved_cell"), n_cells=len(agree),
                token_corr=float(np.mean([d["doses"][f]["token_corr"] for d in dirs.values() for f in ("0.25", "0.5", "1")])), valid=bool(valid),
                string=f"Lens valid at {r['block']}" if valid else f"lens not valid at {r['block']}")


def rg10(ctx):
    """RG10 per block, the layer profile, and RG11 (refit if the lens is not valid at 14 or at all of 22, 24, 26)."""
    j1b = ctx.session.startswith("r1b_R_S5L"); d = "r5l" if j1b else "r5"; prof = {}
    for b in R.LENS_BLOCKS:
        p = ctx.path(f"{d}/r51_b{b}.json")
        r = (K.load_json(p) or {}) if os.path.exists(p) else {}
        prof[str(b)] = validity(r) if r.get("directions") else dict(valid=False, string=r.get("not_run") or K.coded("NA:not_run:input missing"))
    lab = ["refit lens (second attempt)"] if j1b else []
    if j1b:
        R.wj(ctx, "r5l/jlens_validity_j1b.json", dict(blocks=prof, lens="LENS-J1b"), config=dict(), labels=lab); return
    R.wj(ctx, "jlens_validity.json", dict(blocks=prof, lens="LENS-J", convention=K.load_json(ctx.path("r5/lens.json"))["convention"]), config=dict())
    refit = (not prof["14"]["valid"]) or not any(prof[str(b)]["valid"] for b in (22, 24, 26))
    R.wj(ctx, "rg11.json", dict(decision="refit" if refit else "no refit", valid={b: v["valid"] for b, v in prof.items()}),
                   config=dict(rule="not valid at 14 or at all of 22, 24, 26"))


def _valid(ctx):
    j1b = ctx.session.startswith("r1b_R_S5L"); v = K.load_json(R.find(ctx, "jlens_validity.json"))["blocks"]
    if j1b:
        v2 = K.load_json(ctx.path("r5l/jlens_validity_j1b.json"))["blocks"]
        return {b: (v2[b]["valid"] and not v[b]["valid"]) for b in v2}, "J1b"
    return {b: v[b]["valid"] for b in v}, "J"


def covmatched(axl, n, seed):
    """Covariance-matched random unit directions: Xc' z / sqrt(N) with z ~ N(0, I) over 512 centred TXT0 reads (no matrix root)."""
    t = R.torch(); Xc = axl["Xc512"].double(); g = t.Generator().manual_seed(seed)
    return [S.unit(Xc.T @ t.randn(Xc.shape[0], generator=g, dtype=t.float64)) for _ in range(n)]


def r52(ctx):
    """R5.2: t_lens(u) = predicted change of the mean next-token logit difference IMP - DES at A3's answer position per unit push at all
    positions; z_lens against 2,000 covariance-matched random directions; finite differences (+- a_DN) along u_T_perp^1b and 20 random
    directions (10 under RL2), all positions and the 6 object positions."""
    import r2_stimuli as ST2
    t = R.torch(); valid, which = _valid(ctx); m = R.M(ctx); maps, conv, sha = lens_maps(ctx, which); W_U = m.model.lm_head.weight.double()
    lex = ST2.a3_lexemes(); imp, des = [K.enc(m.tok, w)[0] for w in lex["IMP"]], [K.enc(m.tok, w)[0] for w in lex["DES"]]
    objs = ST2.objects(); objs = objs[:4] if ctx.mock else objs[:48]; dev = K.model_device(m.model); nr = 10 if ctx.trimmed("RL2") else 20; out = {}
    for b in (6, 10, 14, 18, 22, 24, 26):
        if which == "J1b" and not valid.get(str(b)):
            continue
        axl = R.load_pt(R.find(ctx, f"ax1b/ax1b_lens_b{b}.pt")); u = S.unit(axl["axes"]["T_1b"].double()); J = Jb(maps, conv, b); dm = R.Dmg(ctx, m, b, axl["cap"])
        if J is None:                                  # no map for block 26's output under K4j's 'input' convention (as in R5.1)
            out[str(b)] = dict(string=K.coded(f"NA:not_run:no J-lens map for block {b} under convention {conv}")); continue
        ins = [ST2.a3_input(lambda s: K.enc(m.tok, s), lambda s: K.chat_text(m.tok, s), o["text"]["A3"]) for o in objs]
        base = []
        for a3 in ins:
            ids = a3["ids"]; lg, xf = tf_logits(m, b, ids + [m.PAD], len(ids)); base.append((ids, a3["obj_pos"], xf[-1:], lg[-1]))   # answer position
        dW = W_U[imp].mean(0) - W_U[des].mean(0); g_ = m.model.model.norm.weight.double(); D_ = m.D

        def tl(V):                                     # t_lens for directions V [n, D] (vectorised; first order, forward only)
            JV = V.double() @ J.T; out_ = 0
            for ids, pos, xf, lg in base:
                x = xf[0]; rms = x.pow(2).mean().sqrt(); dn = g_ * (JV - (JV @ x)[:, None] * x[None] / (D_ * rms ** 2)) / rms; out_ = out_ + dn @ dW
            return (out_ / len(base)).cpu().numpy()
        rnd = covmatched(axl, 2000 if not ctx.mock else 50, 700 + b); tr = tl(R.torch().stack(rnd)); tu = float(tl(u[None])[0])
        z = (tu - tr.mean()) / tr.std(ddof=1); a = dm.calibrate(u, 0.1)["a"]

        def fd(v, pos_only):
            vals = []
            for ids, pos, xf, lg in base:
                r = {}
                for s in (1.0, -1.0):
                    msk = None
                    if pos_only:
                        msk = t.zeros(1, len(ids), dtype=t.bool); msk[0, list(pos)] = True
                    with K.Hooks(m.model) as hk, t.no_grad():
                        if pos_only:
                            _maskpush(hk, b, v, a * s, axl["cap"], msk)
                        else:
                            hk.push(b, v, a, signs=s, cap=axl["cap"], rowstart=t.tensor([m.START]))
                        l2 = m.model(t.tensor([ids], device=dev)).logits[0, -1].double()
                    r[s] = float(l2[imp].mean() - l2[des].mean())
                vals.append((r[1.0] - r[-1.0]) / 2)
            return float(np.mean(vals))
        f_all, f_obj = fd(u, False), fd(u, True); rr = [fd(v, False) for v in covmatched(axl, nr, 900 + b)]
        lab = [] if valid.get(str(b)) else ["not readable (lens not valid at b)"]
        s = ("u_T_perp moves improve-vs-describe (lens)" if z >= 3 else "no response (lens)") if valid.get(str(b)) else "not readable (lens not valid at b)"
        if valid.get(str(b)) and np.sign(f_all) != np.sign(tu):
            lab.append("lens and direct push disagree in sign")
        out[str(b)] = dict(t_lens=tu, z_lens=z, fd_all=f_all, fd_objects=f_obj, fd_random=rr, a_DN=a, string=s, labels=lab + (["refit lens (second attempt)"] if which == "J1b" else []))
    R.wj(ctx, f"{'r5l' if which == 'J1b' else 'r5'}/r52.json", dict(blocks=out, lens=which), config=dict(n_random=nr, objects=len(objs)))


def _maskpush(hk, b, v, a, cap, mask):
    t = R.torch(); vv = K.unit(t.as_tensor(v).float())

    def hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        if x.shape[1] != mask.shape[1]:
            return out
        nrm = x.norm(dim=-1, keepdim=True).clamp(max=cap); y = x + a * nrm * vv.to(x.device, x.dtype) * mask.to(x.device)[..., None].to(x.dtype)
        return (y,) + tuple(out[1:]) if isinstance(out, tuple) else y
    hk.h.append(hk.L[b].register_forward_hook(hook)); return hk


def r53(ctx):
    """R5.3 at block 14: T4 (RSI directions unit(u_T_perp^1b), unit(u_T^v), u_T^J, raw unit(u_T); positive controls unit(u_sea),
    unit(u_sports), unit(u_tone); 20 covariance-matched nulls (10 under RL2); each calibrated to dmg_sym 0.05, both signs, odd shifts of
    the 24 sets on the 64 R5.1 texts) and T3 (the M_J-norm share of u_T_perp in span(sports, tone, sea, u_neg), cross-validated, against
    200 centred pseudo-axes). Rule as frozen; per-set z-scores against the nulls reported (validity-5 / stats-5: open)."""
    t = R.torch(); valid, which = _valid(ctx); b = 14; ok = valid.get("14")
    if which == "J1b" and not ok:
        R.wj(ctx, "r5l/r53.json", dict(string=K.coded("NA:not_run:LENS-J1b not newly valid at 14")), config=dict()); return
    m = R.M(ctx); P = R.p256(ctx); maps, conv, sha = lens_maps(ctx, which); J = Jb(maps, conv, b); axl = R.load_pt(R.find(ctx, f"ax1b/ax1b_lens_b{b}.pt"))
    A = axl["axes"]; ax = R.ax1b(ctx, 14); W_U = m.model.lm_head.weight.float(); g = m.model.model.norm.weight.double()
    ld = R.torch().tensor([i for i in range(W_U.shape[0]) if any(ch.isalnum() for ch in m.tok.decode([i]))]); mu_ = W_U[ld].mean(0); G = 0
    for i0 in range(0, len(ld), 16384):                         # M_J = J' diag(g) Wc' Wc diag(g) J, Wc = W_U[letters / digits] centred (chunked)
        Wc = W_U[ld[i0:i0 + 16384]] - mu_; G = G + (Wc.T @ Wc).double()
    MJ = J.T @ (g[:, None] * G * g[None]) @ J
    ip = lambda x, y: float(x @ MJ @ y)
    basis = []
    for v in (A["sports_raw"], A["tone_raw"], A["sea_raw"], axl["u_neg"]):
        w = v.double().clone()
        for e in basis:
            w = w - ip(w, e) * e
        n_ = ip(w, w) ** 0.5
        if n_ > 1e-12:
            basis.append(w / n_)
    uT = A["T_raw"].double(); uTJ = uT - sum(ip(uT, e) * e for e in basis)
    rsi = {"T_1b": S.unit(A["T_1b"].double()), "T_v": S.unit(ax["axes"]["T_v"].double()), "T_J": S.unit(uTJ), "T_raw": S.unit(uT)}
    pc = {"sea": S.unit(A["sea_raw"].double()), "sports": S.unit(A["sports_raw"].double()), "tone": S.unit(A["tone_raw"].double())}
    nulls = {f"null_{i}": v for i, v in enumerate(covmatched(axl, 10 if ctx.trimmed("RL2") else 20, 1400))}
    dm = R.Dmg(ctx, m, b, axl["cap"]); SI = wsets(ctx, m); t64 = _texts64(ctx, P); pids = [m.ids(p) for p in P["prompts"]]; shifts = {}
    for nm, v in {**rsi, **pc, **nulls}.items():
        a = ctx.cached(f"a_{nm}", lambda: dm.calibrate(v, 0.1 if not ctx.mock else R.alpha0(ctx, 14)), fmt="json")["a"]; per = {k: [] for k in SI}
        for i, c in t64:
            ids = pids[i] + list(c); st_ = len(pids[i])
            lg = {s: t.log_softmax(tf_logits(m, b, ids, st_, lambda hk, s=s: hk.push(b, v, a, signs=s, cap=axl["cap"], rowstart=t.tensor([m.START])))[0], -1) for s in (1.0, -1.0)}
            for k, sid in SI.items():
                per[k].append(float(((t.logsumexp(lg[1.0][:, sid], -1) - t.logsumexp(lg[-1.0][:, sid], -1)) / 2).mean()))
        shifts[nm] = {k: dict(mean=float(np.mean(v_)), iv=S.iv([np.mean(np.random.default_rng(j).choice(v_, len(v_))) for j in range(2000)])) for k, v_ in per.items()}
    fset = {"sea": "W_sea", "sports": "W_sports", "tone": "W_tone"}; NL = list(nulls); out = dict(T4={}, shifts=shifts)
    rng = lambda k: (np.percentile([shifts[n][k]["mean"] for n in NL], 5), np.percentile([shifts[n][k]["mean"] for n in NL], 95))
    hi95 = lambda k: max(shifts[n][k]["mean"] for n in NL) if len(NL) <= 10 else np.percentile([shifts[n][k]["mean"] for n in NL], 95)
    sens = {f: shifts[f][fset[f]]["iv"][0] > hi95(fset[f]) for f in fset}
    for nm in rsi:
        sh = shifts[nm]; rs_ok = sh["W_RSI_eval"]["iv"][0] > hi95("W_RSI_eval")
        out_f = {f: not (rng(fset[f])[0] <= sh[fset[f]]["mean"] <= rng(fset[f])[1]) for f in fset if sens[f]}
        neu = all(rng(k)[0] <= sh[k]["mean"] <= rng(k)[1] for k in SI if k.startswith("neutral"))
        z = {k: (sh[k]["mean"] - np.mean([shifts[n][k]["mean"] for n in NL])) / (np.std([shifts[n][k]["mean"] for n in NL], ddof=1) + 1e-12) for k in SI}
        if rs_ok and not any(out_f.values()) and neu:
            s = "specific in use"
        elif any(out_f.values()):
            sg = {f: ("+" if sh[fset[f]]["mean"] > 0 else "-") + (" (anti-component)" if np.sign(sh[fset[f]]["mean"]) != np.sign(shifts[f][fset[f]]["mean"]) else "")
                  for f, v in out_f.items() if v}
            s = f"pushes foreign topics ({', '.join(f'{k} {v}' for k, v in sg.items())})"
        else:
            s = "undecided"
        out["T4"][nm] = dict(string=s, rsi_specific=rs_ok, foreign_outside=out_f, neutral_inside=neu, not_sensitive=[f for f in fset if not sens[f]],
                             z=z, labels=["foreign specificity not testable: " + ", ".join(f for f in fset if not sens[f])] if not all(sens.values()) else [])
    # T3: M_J share of u_T_perp in span(sports, tone, sea, u_neg), cross-validated against 200 centred pseudo-axes of the RSI axis texts
    fold = t.as_tensor(P["fold"]); dA = axl["diff"]["rsi"][fold & axl["keep"]["rsi"]].double().mean(0)
    share = lambda v: sum(ip(v, e) ** 2 for e in basis) / max(ip(v, v), 1e-300)
    nl = S.centred_nulls(axl["diff"]["rsi"][~fold], axl["keep"]["rsi"][~fold], 311, None, n=200); sh_null = [share(v) for v in nl]
    out["T3"] = dict(share=share(dA), null_p95=float(np.percentile(sh_null, 95)), p_rank=K.p_rank(share(dA), sh_null), labels=["descriptive"])
    words = {}
    for u in ("T_v", "sea_v"):
        p = R.find(ctx, f"r3/b14/{u}/cell_DN.pt")
        if p:
            e = R.load_pt(p); w = e["words"]; words[u] = {nm: float(np.mean(w[0::2, j] - w[1::2, j])) for j, nm in enumerate(e["wordsets"])}
    out["generated_word_odd_differences"] = words
    R.wj(ctx, f"{'r5l' if which == 'J1b' else 'r5'}/r53.json", dict(**out, lens=which, labels=[] if ok else ["lens not valid at b"]),
                   config=dict(nulls=len(NL), sets=list(SI)))


# ------------------------------------------------------------------------------------------------------------------ R5.5 (SO-GVRA)
def _axis_reads(ctx, m, cands, qs, b_list):
    """Pro / anti texts of candidate instruction pairs (keys ('r1b_lab', cand, q); block-independent ids) read at the blocks."""
    P = R.p256(ctx); out = {}
    for ci, (nm, pro, anti) in cands.items():
        rows = [(m.ids(P["prompts"][i] + " " + ins), ctx.key("r1b_lab", ci, P["q"][i]), s) for i in qs for s, ins in ((1, pro), (-1, anti))]
        conts = ctx.cached(f"lab_{ci}_{len(qs)}", lambda: K.generate(m.model, m.tok, rows, T=0.7, max_new=m.nnew))
        out[ci] = {b: R.read_win(m, [m.ids(P["prompts"][i]) for i in qs for _ in (0, 1)], conts, b)[0] for b in b_list}
    return out


def verify(X, fold, target, seed=311):
    """Verifier (forward round trip, decoupled from the generator): the candidate axis's cross-fitted, disattenuated |cos| with the target
    (fold axes, kappa floor 0.5) >= 0.7 and above the 95th percentile of 200 centred pseudo-axes of its own texts."""
    t = R.torch(); Xp, Xa = X[0::2], X[1::2]; keep = ~(t.isnan(Xp).any(1) | t.isnan(Xa).any(1)); D_ = t.nan_to_num(Xp - Xa); f = t.as_tensor(fold)
    aA, aB = S.unit(D_[keep & f].mean(0)), S.unit(D_[keep & ~f].mean(0)); kap = max(float(aA @ aB), 0.5); tg = S.unit(target.double())
    c = abs(float(aA @ tg) + float(aB @ tg)) / 2 / kap ** 0.5; nl = S.centred_nulls(D_, keep, seed, None, n=200)
    p95 = float(np.percentile([abs(S.cos(v, tg)) for v in nl], 95)); return dict(cos=c, null_p95=p95, accept=c >= 0.7 and c > p95)


def r55(ctx):
    """R5.5: targets u*_b and v_max^C at 14 / 18 / 6; generator = cheap rollout ranking (16 prompts); verifier = 64-prompt axis; reviser =
    the two best concatenated, 3 revisions, then 'abstained after 3 revisions'; verifier error on the known targets u_sea and u_tone
    (held-out prompts)."""
    t = R.torch(); m = R.M(ctx); P = R.p256(ctx); n = len(P["q"]); qs_v = [i for i in range(n) if P["q"][i] % 4 == 0] or list(range(n))
    qs_q = qs_v[:16]; qs_h = [i for i in range(n) if P["q"][i] % 4 == 1] or list(range(n)); bank = {i: c for i, c in enumerate(BANK)}
    quick = _axis_reads(ctx, m, bank, qs_q, R.BLOCKS); fold_q = [i % 2 == 0 for i in range(len(qs_q))]; out = dict(targets={}, known={})

    def run(target, b, qs, tag):
        sc = {ci: verify(quick[ci][b], fold_q, target)["cos"] for ci in bank}; order = sorted(sc, key=sc.get, reverse=True); tried = []
        for rev in range(4):
            cands = {order[0]: bank[order[0]]} if rev == 0 else {f"rev{rev}": (f"{bank[order[0]][0]} + {bank[order[rev]][0]}", bank[order[0]][1] + " " + bank[order[rev]][1],
                                                                                  bank[order[0]][2] + " " + bank[order[rev]][2])}
            ck = list(cands)[0] if not rev else f"r{order[0]}_{order[rev]}"; cv = cands[list(cands)[0]]
            Xf = _axis_reads(ctx, m, {ck: cv}, qs, (b,))[ck][b]
            v = verify(Xf, [i % 2 == 0 for i in range(len(qs))], target); tried.append(dict(candidate=cv[0], **v))
            if v["accept"]:
                return dict(string=f"verified label: {cv[0]}", attempts=tried, ranking={BANK[c][0]: sc[c] for c in order[:5]})
        return dict(string="abstained after 3 revisions", attempts=tried, ranking={BANK[c][0]: sc[c] for c in order[:5]})
    for b in R.BLOCKS:
        ax = R.ax1b(ctx, b); out["known"][str(b)] = {k: run(S.unit(ax["axes"][f"{k}_raw"].double()), b, qs_h, k) for k in ("sea", "tone")}
        out["targets"][f"u_star_{b}"] = run(S.unit(R.ustar(ctx, b)[0]), b, qs_v, "ustar")
        jp = R.find(ctx, f"r4/jloop_b{b}.pt")
        if jp:
            out["targets"][f"v_max_C_{b}"] = run(S.unit(R.load_pt(jp)["modes"]["C"]["v"].double()), b, qs_v, "vmax")
    fails = {b: sum(not v["string"].endswith(k) and not (k == "tone" and "enthusiastic" in v["string"]) for k, v in d.items()) for b, d in out["known"].items()}
    out["verifier_error"] = fails; unrel = all(fails[b] == 2 for b in fails)
    for v in out["targets"].values():
        v["labels"] = ["descriptive"] + (["unreliable verifier"] if unrel else [])
    R.wj(ctx, "r5/r55.json", out, config=dict(bank=[b[0] for b in BANK], bank_sha256=K.sha256_text(K.canon(BANK)), accept=0.7, nulls=200, revisions=3),
                   labels=["descriptive"])


# ------------------------------------------------------------------------------------------------------------------ R5.4 (tier 2)
NCHUNK = 16


def refit_texts(ctx):
    """300 unsteered 128-token chat-format texts from P256 prompts (keys ('r1b_lensfit', q, s)); 200 under RL1 / RL2."""
    m = R.M(ctx); P = R.p256(ctx); n = 200 if ctx.trimmed("RL1") or ctx.trimmed("RL2") else 300; n = 8 if ctx.mock else n
    rows = [(m.ids(P["prompts"][j % len(P["q"])]), ctx.key("r1b_lensfit", P["q"][j % len(P["q"])], j // len(P["q"])), 1) for j in range(n)]
    conts = K.generate(m.model, m.tok, rows, T=0.7, max_new=16 if ctx.mock else 128)
    ctx.save_pt("r5l/texts.pt", dict(ids=[list(r[0]) + list(c) for r, c in zip(rows, conts)]), config=dict(n=n, key="('r1b_lensfit', q, s)"))


def refit_fit(ctx):
    """One chunk of the averaged-Jacobian fit in bf16: for every text, dim_batch 64 replicas each back-propagate one coordinate of the
    final block's output summed over target positions; the gradients at the outputs of blocks 0-26 at source positions 16..T-2 are
    summed (J[l] = mean over sources of the summed effect on later targets). Running sums in r5l/acc.pt (chunk ids recorded)."""
    t = R.torch(); k = int(ctx.step.rsplit("_", 1)[1]); tx = R.load_pt(ctx.path("r5l/texts.pt"))["ids"]; m = R.M(ctx)
    accp = ctx.path("r5l/acc.pt"); acc = R.load_pt(accp) if os.path.exists(accp) else dict(S=None, n=0, done=[])
    if k not in acc["done"]:
        import copy
        mb = copy.deepcopy(m.model).to(t.bfloat16); L = K.layers_of(mb); D, nb = m.D, min(64, m.D); dev = K.model_device(mb)
        Ssum = acc["S"] if acc["S"] is not None else t.zeros(len(L) - 1, D, D, dtype=t.float32); cnt = acc["n"]
        for ids in tx[k::NCHUNK]:
            T_ = len(ids); src = list(range(16, T_ - 1)) if T_ > 18 else list(range(1, T_ - 1))
            for d0 in range(0, D, nb):
                B = min(nb, D - d0); hs = {}
                hooks = [L[l].register_forward_hook(lambda mod, i, o, l=l: hs.__setitem__(l, o[0] if isinstance(o, tuple) else o)) for l in range(len(L))]
                try:
                    with t.enable_grad():
                        x = t.tensor([ids] * B, device=dev); emb = mb.get_input_embeddings()(x).detach().requires_grad_(True)
                        mb.model(inputs_embeds=emb); outs = [hs[l] for l in range(len(L))]
                        for o in outs[:-1]:
                            o.retain_grad()
                        sel = t.zeros(B, D, device=dev, dtype=outs[-1].dtype); sel[t.arange(B), t.arange(d0, d0 + B)] = 1
                        (outs[-1] * sel[:, None, :]).sum().backward()
                        for l, o in enumerate(outs[:-1]):
                            Ssum[l, d0:d0 + B] += o.grad[:, src].float().sum(1).cpu()
                finally:
                    for h in hooks:
                        h.remove()
            cnt += len(src)
        acc = dict(S=Ssum, n=cnt, done=acc["done"] + [k]); t.save(acc, accp + ".tmp"); os.replace(accp + ".tmp", accp); del mb
    R.wj(ctx, f"r5l/fit_{k}.json", dict(chunk=k, positions=acc["n"], chunks_done=acc["done"]), config=dict(dtype="bfloat16", dim_batch=64))


def refit_lens(ctx):
    t = R.torch(); acc = R.load_pt(ctx.path("r5l/acc.pt")); J = (acc["S"] / max(acc["n"], 1)).half()
    ctx.save_pt("r5l/lens_j1b.pt", dict(maps=J, checksum=K.sha256_text(str(float(J.float().sum()))), positions=acc["n"]),
                config=dict(dtype="bfloat16", target="block 27 output", sources="block 0-26 outputs", positions="16..T-2"),
                labels=["refit lens (second attempt)"])
