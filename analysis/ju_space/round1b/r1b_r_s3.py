"""r1b_R_S3 (part R): R3.3 dose series in both signs with unpushed texts (DOSE-1b), TXT0 (R3.4: 256 prompts x 16 seeds, unpushed) and
its reads at 14 / 18 / 6, the frequency-matched neutral word sets (WORDSETS), R4.1 FDT scores, R4.2 read directions, candidate
subspace C_b and the in-session full-sample eigen-analysis (JLOOP), R4.3 direct cells. Cross-fits, nulls and bootstraps: r1b_R_ana."""
import math, os
import numpy as np
import r1b_core as K
import r1b_r_lib as R
import r1b_r_stats as S

DOSES = (0.25, 0.35, 0.5, 1.0, 2.0)


def _b(ctx, pos=2):
    return int(ctx.step.split("_")[pos][1:])


def dirs3(ctx, b, m):
    """R3.3 directions: u*_b, unit(u_T^v), unit(u_sea^v) and Pool ('cov', 14, 4) at 14; u*_b and unit(u_T^v) at 18 and 6."""
    ax = R.ax1b(ctx, b); d = {"ustar": S.unit(R.ustar(ctx, b)[0]), "T_v": S.unit(ax["axes"]["T_v"].double())}
    if b == 14:
        d["sea_v"] = S.unit(ax["axes"]["sea_v"].double()); d["cov4"] = R.pool_dir(ctx, "cov", 14, 4, m.D)
    return d


def _setup(ctx, b):
    m = R.M(ctx); P = R.p256(ctx); st = R.r1_state(ctx, b); cap = float(st["ref"]["cap"])
    return m, P, cap, [m.ids(p) for p in P["prompts"]]


def unpushed(ctx):
    """One unpushed text per key ('r1b_dose', b, q, s) (512 per block), read at b: the 0 of the (+, 0, -) triplets."""
    t = R.torch(); b = _b(ctx); m, P, cap, pids = _setup(ctx, b)
    rows = [(pids[i], ctx.key("r1b_dose", b, q, s), 1) for i, q in enumerate(P["q"]) for s in range(2)]
    conts = K.generate(m.model, m.tok, rows, T=0.7, max_new=m.nnew); X = R.read_win(m, [r[0] for r in rows], conts, b)[0]
    ctx.save_pt(f"r3/b{b}/unpushed.pt", dict(X0=X.float(), conts=conts, words=R.wcount(m.decode(conts)), wordsets=list(R.WSET_NAMES),
                                             prompt_of=t.tensor([i for i in range(len(P["q"])) for s in range(2)]), q=P["q"]),
                config=dict(block=b, key="('r1b_dose', b, q, s)", T=0.7))


def dose(ctx):
    """DOSE-1b(b, u): cells a in {0.25, 0.35, 0.5, 1, 2} x alpha_0(b) (RL1 at 18 / 6: {0.25, 0.5, 1}) and the direction's PUSH-DN dose
    a_DN(b, u) (calibrated once), 256 prompts x 2 seeds, + and - texts sharing the block's keys ('r1b_dose', b, q, s)."""
    import r1b_r_s2
    t = R.torch(); b = _b(ctx); u_name = ctx.step.split("_", 3)[3]; m, P, cap, pids = _setup(ctx, b); a0 = R.alpha0(ctx, b)
    u = dirs3(ctx, b, m)[u_name]; dm = R.Dmg(ctx, m, b, cap); Fn, F = r1b_r_s2.readout_set(R.ax1b(ctx, b))
    cal = ctx.cached(f"aDN_{u_name}", lambda: dm.calibrate(u, a0), fmt="json")
    doses = [x for x in DOSES if not (b != 14 and ctx.trimmed("RL1") and x in (0.35, 2.0))]
    cells = [(f"{x:g}", x * a0) for x in doses] + [("DN", cal["a"])]; summ = []
    rows = [(pids[i], ctx.key("r1b_dose", b, q, s), sg) for i, q in enumerate(P["q"]) for s in range(2) for sg in (1, -1)]
    for tag, a in cells:
        rel = f"r3/b{b}/{u_name}/cell_{tag}.pt"
        if not os.path.exists(ctx.path(rel) + ".prov.json"):
            ev = R.gen_pairs(ctx, m, b, rows, u, a, cap, F=F); dd = dm.push(u, a)
            ev.update(prompt_of=t.tensor([i for i in range(len(P["q"])) for s in range(2)]), push=u.float(), a_over_alpha0=a / a0, F_names=Fn,
                      dmg=dict(sym=dd["sym"], plus=dd["plus"], minus=dd["minus"], top1_scene=dd["top1_scene"]), block=b, direction=u_name,
                      cell=tag, u32=dm.u32(u, a), q=P["q"])
            R.save_ev(ctx, rel, ev, dict(block=b, direction=u_name, cell=tag, a=a, key="('r1b_dose', b, q, s)"))
        summ.append(dict(cell=tag, a=a, a_over_alpha0=a / a0, file=rel))
    ctx.write_json(f"r3/b{b}/dose_{u_name}.json", dict(block=b, direction=u_name, alpha0=a0, a_DN=cal, cells=summ,
                                                         trimmed_RL1=b != 14 and ctx.trimmed("RL1")), config=dict(block=b, direction=u_name, doses=doses))


def txt0(ctx):
    """TXT0 (R3.4): 256 prompts x 16 seeds of unpushed texts, keys ('r1b_free', q, s), T 0.7 (block-independent token ids); step _a
    seeds 0-7, step _b seeds 8-15."""
    t = R.torch(); part = ctx.step[-1]; seeds = range(0, 8) if part == "a" else range(8, 16); m = R.M(ctx); P = R.p256(ctx)
    pids = [m.ids(p) for p in P["prompts"]]
    rows = [(pids[i], ctx.key("r1b_free", q, s), 1) for i, q in enumerate(P["q"]) for s in seeds]
    conts = ctx.cached("conts", lambda: K.generate(m.model, m.tok, rows, T=0.7, max_new=m.nnew))
    ctx.save_pt(f"txt0/txt0_{part}.pt", dict(conts=conts, prompt_of=[i for i in range(len(P["q"])) for s in seeds], seed_of=[s for _ in P["q"] for s in seeds],
                                            q=P["q"]), config=dict(seeds=list(seeds), key="('r1b_free', q, s)", T=0.7))


def txt0_all(ctx):
    a, b = (R.load_pt(R.find(ctx, f"txt0/txt0_{x}.pt")) for x in "ab")
    return dict(conts=a["conts"] + b["conts"], prompt_of=a["prompt_of"] + b["prompt_of"], seed_of=a["seed_of"] + b["seed_of"], q=a["q"])


def txt0_read(ctx):
    b = _b(ctx, 3); m = R.M(ctx); P = R.p256(ctx); T0 = txt0_all(ctx); pids = [m.ids(p) for p in P["prompts"]]
    X = R.read_win(m, [pids[i] for i in T0["prompt_of"]], T0["conts"], b)[0]
    ctx.save_pt(f"txt0/reads_b{b}.pt", dict(X=X.float(), prompt_of=T0["prompt_of"], seed_of=T0["seed_of"]), config=dict(block=b))


def heldout_idx(T0, P):
    """R4.1: seed 15 of every 4th prompt of each type (64 prompts at 256) is held out for R4.2's read directions (and R5's texts)."""
    return [j for j, (i, s) in enumerate(zip(T0["prompt_of"], T0["seed_of"])) if s == 15 and P["q"][i] % 4 == 0]


def first_ids(tok, word):
    return sorted({tok(w, add_special_tokens=False)["input_ids"][0] for w in (word, " " + word, word.capitalize(), " " + word.capitalize())})


def set_ids(tok, words):
    return sorted({i for w in words for i in first_ids(tok, w)})


def words(ctx):
    """Neutral sets (R2.1): each word's mean log-probability of its first token (four spellings, summed probability) at the continuation
    positions of 64 TXT0 texts (seed 0 of the 64 read-direction prompts); 20 sets of 15 pool words matched to W_RSI-eval on the decile
    histogram (np.random.default_rng(4242)) and, for R4.2, 20 sets matched to W_RSI-build (rng 4243; stats-15, part file)."""
    t = R.torch(); m = R.M(ctx); P = R.p256(ctx); T0 = R.load_pt(R.find(ctx, "txt0/txt0_a.pt")); pool = R.neutral_pool()
    idx = [j for j, (i, s) in enumerate(zip(T0["prompt_of"], T0["seed_of"])) if s == 0 and P["q"][i] % 4 == 0]
    allw = sorted(set(pool) | set(R.W_RSI)); ids = {w: first_ids(m.tok, w) for w in allw}; uniq = sorted({i for v in ids.values() for i in v})
    col = {u: k for k, u in enumerate(uniq)}; A = t.zeros(len(uniq), len(allw), dtype=t.float64)
    for j, w in enumerate(allw):
        for i in ids[w]:
            A[col[i], j] = 1.0
    acc, n = t.zeros(len(allw), dtype=t.float64), 0; dev = K.model_device(m.model)
    for j in idx:
        pid, c = m.ids(P["prompts"][T0["prompt_of"][j]]), T0["conts"][j]
        if not c:
            continue
        with t.no_grad():
            lg = m.model(t.tensor([pid + list(c)], device=dev)).logits[0, len(pid) - 1:len(pid) + len(c) - 1].double()
        pr = t.softmax(lg, -1)[:, uniq].cpu(); acc += t.log((pr @ A).clamp_min(1e-300)).sum(0); n += len(c)
    val = {w: float(acc[k] / max(n, 1)) for k, w in enumerate(allw)}
    pv = np.array([val[w] for w in pool]); edges = np.percentile(pv, np.arange(10, 100, 10)); bin_of = lambda x: int(np.searchsorted(edges, x, side="right"))
    bins = [[w for w in pool if bin_of(val[w]) == k] for k in range(10)]; sets = {}
    for nm, ref, seed in (("neutral_eval", R.WORDSETS["W_RSI_eval"], 4242), ("neutral_build", R.WORDSETS["W_RSI_build"], 4243)):
        h = np.bincount([bin_of(val[w]) for w in ref], minlength=10); g = np.random.default_rng(seed); sets[nm] = []
        for _ in range(20):
            s_ = []
            for k in range(10):
                cand = bins[k] or [w for kk in sorted(range(10), key=lambda z: abs(z - k)) for w in bins[kk]]
                s_ += list(g.choice(cand, size=min(int(h[k]), len(cand)), replace=False))
            sets[nm].append(sorted(s_))
    out = dict(WORDSETS=R.WORDSETS, **sets, values=val, deciles=edges.tolist(), n_positions=n, texts=len(idx))
    out["sha256"] = K.sha256_text(K.canon(dict(WORDSETS=R.WORDSETS, **sets)))
    ctx.write_json("txt0/wordsets.json", out, config=dict(seeds=dict(neutral_eval=4242, neutral_build=4243), pool_sha256=R.NEUTRAL_SHA))


def scores(m, b, prompts, conts, cap, T=0.7, batch=8):
    """B-Q7 / r2_theta.score_grads with r1b_core: per text one forward-backward pass with rk's zero leaf V (added as min(|h_t|, cap_b) V
    at the block-b output from the first user token) on y = sum of log softmax(logits_t / T)[y_t] over the continuation (+ the stop
    log-probability if the text ended early). Returns X, S (float64), y, cbar sums and counts."""
    t = R.torch(); dev = K.model_device(m.model); stop = sorted(int(x) for x in K.stop_tokens(m.tok, m.model)[1])
    N = len(conts); X = t.full((N, m.D), float("nan"), dtype=t.float64); Sg = t.zeros(N, m.D, dtype=t.float64); Y = np.full(N, np.nan)
    csum, ccnt = np.zeros(N), np.zeros(N, np.int64)
    for i in range(0, N, batch):
        pb, cb = prompts[i:i + batch], conts[i:i + batch]; ids, am, pad = K.pad_left([list(p) + list(c) for p, c in zip(pb, cb)], m.PAD)
        B, Tn = ids.shape; V = t.zeros(B, m.D, device=dev, requires_grad=True)
        with K.Hooks(m.model) as hk, t.enable_grad():
            hk.push(b, None, 0.0, cap=cap, rowstart=pad + m.START, V=V); hk.capture([b])
            last = m.model.model(input_ids=ids.to(dev), attention_mask=am.to(dev)).last_hidden_state; ys = []
            for r, c in enumerate(cb):
                L = len(c); lp = t.log_softmax(m.model.lm_head(last[r, Tn - L - 1:Tn]).float() / T, -1)
                y = lp[t.arange(L, device=dev), t.tensor(list(c), device=dev)].sum() if L else lp.new_zeros(())
                if L < m.nnew:
                    y = y + t.logsumexp(lp[L, stop], 0)
                ys.append(y)
            g, = t.autograd.grad(t.stack(ys).sum(), V)
            h = hk.states[b][0].double()
        Sg[i:i + B] = g.double().cpu(); Y[i:i + B] = [float(y.detach()) for y in ys]
        for r, c in enumerate(cb):
            L = len(c)
            if L:
                seg = h[r, Tn - L:]; X[i + r] = seg.mean(0); nr = seg.norm(dim=-1).clamp(max=cap); csum[i + r] = float(nr.sum()); ccnt[i + r] = L
    return X, Sg, Y, csum, ccnt


def fdt(ctx):
    """R4.1 at block b: X and S of the FDT texts (TXT0 seeds 0-14 at 14, 0-7 at 18 and 6)."""
    b = _b(ctx); m = R.M(ctx); P = R.p256(ctx); T0 = txt0_all(ctx); cap = float(R.r1_state(ctx, b)["ref"]["cap"])
    smax = 14 if b == 14 else 7; J = [j for j, s in enumerate(T0["seed_of"]) if s <= smax]; pids = [m.ids(p) for p in P["prompts"]]
    X, Sg, Y, cs, cc = ctx.cached(f"scores_b{b}", lambda: scores(m, b, [pids[T0["prompt_of"][j]] for j in J], [T0["conts"][j] for j in J], cap))
    ctx.save_pt(f"r4/fdt_b{b}.pt", dict(X=X.float(), S=Sg.float(), y=Y, cbar_sum=cs, cbar_cnt=cc, idx=J, prompt_of=[T0["prompt_of"][j] for j in J],
                                        seed_of=[T0["seed_of"][j] for j in J], cap=cap), config=dict(block=b, seeds=f"0-{smax}", T=0.7))


def read_grads(m, b, prompts, conts, sets_ids, batch=8):
    """R4.2 read directions: per text and word set S, g = d/dV of sum over the continuation positions of log p(S | prefix_t) (T 1; p(S)
    = summed probability of the first tokens of S's spellings) with a plain leaf V added at the block-b output from the first user token
    (one backward per (text, set)). Returns {set: [N, D]}."""
    t = R.torch(); dev = K.model_device(m.model); N = len(conts); G = {k: t.zeros(N, m.D, dtype=t.float64) for k in sets_ids}
    for i in range(0, N, batch):
        pb, cb = prompts[i:i + batch], conts[i:i + batch]; ids, am, pad = K.pad_left([list(p) + list(c) for p, c in zip(pb, cb)], m.PAD)
        B, Tn = ids.shape; V = t.zeros(B, m.D, device=dev, requires_grad=True); rs = (pad + m.START).to(dev)

        def hook(mod, inp, out):
            x = out[0] if isinstance(out, tuple) else out; msk = (t.arange(x.shape[1], device=x.device)[None, :] >= rs[:, None]).to(x.dtype)
            y = x + V[:, None, :] * msk[..., None]; return (y,) + tuple(out[1:]) if isinstance(out, tuple) else y
        hh = K.layers_of(m.model)[b].register_forward_hook(hook)
        try:
            with t.enable_grad():
                last = m.model.model(input_ids=ids.to(dev), attention_mask=am.to(dev)).last_hidden_state
                lps = [t.log_softmax(m.model.lm_head(last[r, Tn - len(c) - 1:Tn - 1]).float(), -1) for r, c in enumerate(cb)]
                for k, sid in sets_ids.items():
                    obj = sum(t.logsumexp(lp[:, sid], -1).sum() for lp, c in zip(lps, cb) if len(c))
                    g, = t.autograd.grad(obj, V, retain_graph=True); G[k][i:i + B] = g.double().cpu()
        finally:
            hh.remove()
    return G


def subspace_C(ctx, b, m, VS, Gneu):
    """C_b (k_C = 32): u*_b, the d_4 of the seven DN loops, u_T^v, u_sea^v, u_sports^v, u_tone^v, u_neg, V_RSI (4), then further left
    singular vectors of the union of the other 23 sets' gradient matrices until 32 columns survive QR (|R_ii| >= 1e-6 relative)."""
    t = R.torch(); ax = R.ax1b(ctx, b); cols, names = [S.unit(R.ustar(ctx, b)[0])], ["u_star"]
    for L in R.LOOPS_1B:
        p = R.find(ctx, f"r2/b{b}/dirs_DN_{L}.pt")
        if p:
            cols.append(S.unit(R.load_pt(p)["d"][4].double())); names.append(f"d4_{L}")
    for x in ("T_v", "sea_v", "sports_v", "tone_v"):
        cols.append(S.unit(ax["axes"][x].double())); names.append(x)
    cols.append(S.unit(ax["u_neg"].double())); names.append("u_neg")
    for i in range(VS.shape[1]):
        cols.append(VS[:, i]); names.append(f"V_RSI_{i}")
    U = t.linalg.svd(Gneu, full_matrices=False)[0]; kC = min(32, m.D); extra = 0
    while True:
        F = t.stack(cols + [U[:, j] for j in range(extra)], 1); Q, Rr = t.linalg.qr(F); d = Rr.diagonal().abs()
        keep = d >= 1e-6 * float(d.max()); n_ok = int(keep.sum())
        if n_ok >= kC or extra >= U.shape[1]:
            break
        extra += kC - n_ok
    nm = names + [f"sv_{j}" for j in range(extra)]
    return Q[:, keep][:, :kC], [n for n, k in zip(nm, keep.tolist()) if k][:kC]


def r42(ctx):
    """R4.2 in session at block b: read directions of the 24 sets (W_RSI-build, W_sea, W_sports, W_tone, 20 neutral sets matched to
    W_RSI-build) on the 64 held-out texts, V_S (top-4 left singular vectors), C_b, and the full-sample J's v_max^C, w_max^C, v_S, w_S,
    the full-space Ritz pair, and halving 0's fit-half directions with their held-out value (R4.3; stats-8, part file)."""
    t = R.torch(); b = _b(ctx); m = R.M(ctx); P = R.p256(ctx); T0 = txt0_all(ctx); ws = K.load_json(R.find(ctx, "txt0/wordsets.json"))
    H = heldout_idx(T0, P); pids = [m.ids(p) for p in P["prompts"]]
    sets = {"W_RSI_build": R.WORDSETS["W_RSI_build"], "W_sea": R.W_SEA, "W_sports": R.W_SPORTS, "W_tone": R.W_TONE,
            **{f"neutral_{k}": w for k, w in enumerate(ws["neutral_build"])}}
    G = ctx.cached(f"grads_b{b}", lambda: read_grads(m, b, [pids[T0["prompt_of"][j]] for j in H], [T0["conts"][j] for j in H],
                                                       {k: set_ids(m.tok, w) for k, w in sets.items()}))
    VS = {k: t.linalg.svd(g.T, full_matrices=False)[0][:, :4] for k, g in G.items()}
    Q_C, names = subspace_C(ctx, b, m, VS["W_RSI_build"], t.cat([g for k, g in G.items() if k != "W_RSI_build"]).T)
    f = R.load_pt(R.find(ctx, f"r4/fdt_b{b}.pt")); cbar = float(np.sum(f["cbar_sum"]) / max(np.sum(f["cbar_cnt"]), 1))
    ok = ~t.isnan(f["X"]).any(1); po = t.as_tensor(f["prompt_of"])[ok]; Xc, Sc, wt = S.fdt_centre(f["X"][ok], f["S"][ok], po)
    out = dict(block=b, Q_C=Q_C.float(), C_names=names, V_S={k: v.float() for k, v in VS.items()}, cbar=cbar, sets=list(sets), modes={})
    for nm, Q in [("C", Q_C)] + [(k, v) for k, v in VS.items()]:
        lam, y, z, lmod = S.rightmost(S.restricted(Xc, Sc, wt, cbar, Q))
        out["modes"][nm] = dict(lam=[lam.real, lam.imag], lam_maxmod=[lmod.real, lmod.imag], v=(Q @ y.real.double()).float(), w=(Q @ z.real.double()).float())
    us = S.unit(R.ustar(ctx, b)[0]); lam, y, z = S.full_ritz(Xc, Sc, wt, cbar, us)
    out["ritz_full"] = dict(lam=[lam.real, lam.imag], v=y.real.float(), w=z.real.float())
    hA = S.halvings(len(P["q"]))[0].bool()[po]; h0 = {}
    XA, SA, wA = S.fdt_centre(f["X"][ok][hA], f["S"][ok][hA], po[hA]); XB, SB, wB = S.fdt_centre(f["X"][ok][~hA], f["S"][ok][~hA], po[~hA])
    for nm, Q in (("vmax", Q_C), ("vRSI", VS["W_RSI_build"].double())):
        lam, y, z, _ = S.rightmost(S.restricted(XA, SA, wA, cbar, Q)); hv, ray = S.heldout(S.restricted(XB, SB, wB, cbar, Q), y, z)
        h0[nm] = dict(v=(Q @ y.real.double()).float(), w=(Q @ z.real.double()).float(), lam_fit=[lam.real, lam.imag], lam_FDT=hv, rayleigh=ray)
    out["halving0"] = h0
    ctx.save_pt(f"r4/jloop_b{b}.pt", out, config=dict(block=b, k=4, k_C=32, T_read=1.0, sets=list(sets)))


def r43(ctx):
    """R4.3: cells along halving 0's v_max^C and v_RSI (sign: cos with u*_b, resp. u_T^v, >= 0) at their PUSH-DN dose, 512 pairs, keys
    ('r1b_fdtchk', b, which, q, s)."""
    import r1b_r_s2
    t = R.torch(); b = _b(ctx); m, P, cap, pids = _setup(ctx, b); J = R.load_pt(ctx.path(f"r4/jloop_b{b}.pt")); a0 = R.alpha0(ctx, b)
    ax = R.ax1b(ctx, b); dm = R.Dmg(ctx, m, b, cap); Fn, F = r1b_r_s2.readout_set(ax); out = dict(block=b, cells={})
    for which, ref in (("vmax", R.ustar(ctx, b)[0]), ("vRSI", ax["axes"]["T_v"])):
        v = S.unit(J["halving0"][which]["v"].double()); v = v if float(v @ ref.double()) >= 0 else -v
        cal = ctx.cached(f"aDN_{which}", lambda: dm.calibrate(v, a0), fmt="json"); rel = f"r4/b{b}/r43_{which}.pt"
        if not os.path.exists(ctx.path(rel) + ".prov.json"):
            rows = [(pids[i], ctx.key("r1b_fdtchk", b, which, q, s), sg) for i, q in enumerate(P["q"]) for s in range(2) for sg in (1, -1)]
            ev = R.gen_pairs(ctx, m, b, rows, v, cal["a"], cap, F=F)
            ev.update(prompt_of=t.tensor([i for i in range(len(P["q"])) for s in range(2)]), push=v.float(), calib=cal, which=which,
                      w=J["halving0"][which]["w"], lam_FDT=J["halving0"][which]["lam_FDT"], u32=dm.u32(v, cal["a"]))
            R.save_ev(ctx, rel, ev, dict(block=b, which=which, key="('r1b_fdtchk', b, which, q, s)"), labels=cal["labels"])
        out["cells"][which] = dict(file=rel, a=cal["a"], dmg_sym=cal["dmg_sym"], in_band=cal["in_band"])
    ctx.write_json(f"r4/r43_b{b}.json", out, config=dict(block=b))
