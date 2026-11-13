"""r1b_R_S6 (part R, R6): R6.1 text loop against the latent (soft-token, expected-embedding) loop at PUSH-DN, per block and R3.3
direction, and the iterated latent loop; R6.2 in-context carry over turns 2-6 on the RSI axis at T 0.7 (turn 1 = R3.3's PUSH-DN cell)
and T 1.0 (new turn 1), with the random-direction carry control; R6.3 the replication of round 2's T 1.0 cell. Short contexts are
re-prefilled every turn (no KV-prefix reuse); B-Q8's concatenation with history think blocks kept (r2_stimuli.TURN2_SUFFIX)."""
import os
import numpy as np
import r1b_core as K
import r1b_r_lib as R
import r1b_r_stats as S
from r1b_r_s3 import dirs3


def _cell(ctx, b, u, tag="DN"):
    p = R.find(ctx, f"r3/b{b}/{u}/cell_{tag}.pt")
    if not p:
        raise K.NotRun(f"input missing: r3/b{b}/{u}/cell_{tag}.pt")
    return R.load_pt(p)


def soft_gen(m, b, pids, d, a, cap, nsteps, T=0.7, batch=32, extra_read=None):
    """Latent loop (R6.1): at each step the next input is sum_v p_t(v) E[v] (p_t = softmax(logits_t / T) of the pushed model, tied
    embedding, deterministic); survival S_t = prod_{t' < t} (1 - p_t'(stop)); the unsteered model reads plain prompt + soft embeddings
    stopped at block b: x_soft = sum_t S_t h_b(t) / sum_t S_t. Rows (prompt, sign); returns X [N, D] float64 and the pooled cbar."""
    t = R.torch(); dev = K.model_device(m.model); E = m.model.get_input_embeddings().weight; stop = sorted(K.stop_tokens(m.tok, m.model)[1])
    out, cb = [], []
    for i in range(0, len(pids), batch):
        pb = pids[i:i + batch]; ids, am, pad = K.pad_left([p for p, s in pb], m.PAD); sg = t.tensor([float(s) for p, s in pb]); rec = {}
        hk = K.Hooks(m.model); R.recorder(hk, b, cap, rec)
        if d is not None and a:
            hk.push(b, d, a, signs=sg, cap=cap, rowstart=pad + m.START)
        embs, surv, alive = [], [], t.ones(len(pb), dtype=t.float64, device=dev)
        with hk, t.no_grad():
            amk = am.to(dev); pos = (amk.cumsum(-1) - 1).clamp_min(0)          # positions as in generate() (left padding)
            o = m.model(input_ids=ids.to(dev), attention_mask=amk, position_ids=pos, use_cache=True); pkv = o.past_key_values; last = pos[:, -1:]
            for s in range(nsteps):
                p = t.softmax(o.logits[:, -1].double() / T, -1); e = (p.to(E.dtype) @ E)[:, None]; surv.append(alive.clone())
                alive = alive * (1 - p[:, stop].sum(-1)); embs.append(e); amk = t.cat([amk, t.ones(len(pb), 1, dtype=amk.dtype, device=dev)], 1)
                if s < nsteps - 1:
                    last = last + 1; o = m.model(inputs_embeds=e, attention_mask=amk, position_ids=last, past_key_values=pkv, use_cache=True)
                    pkv = o.past_key_values
        cb.append(R.cbar_of(rec, [[0] * nsteps] * len(pb))["pooled"])
        soft = t.cat(embs, 1); W = t.stack(surv, 1)
        with K.Hooks(m.model) as hr, t.no_grad():
            if extra_read is not None:
                extra_read(hr)
            pe = m.model.get_input_embeddings()(ids.to(dev)); hr.capture([b]).stop_after(b)
            hr.run(lambda: m.model.model(inputs_embeds=t.cat([pe, soft], 1), attention_mask=amk))
            h = hr.states[b][0].double()[:, -nsteps:]
        out.append((h * W.cpu()[..., None]).sum(1) / W.cpu().sum(1, keepdim=True).clamp_min(1e-12))
    return t.cat(out), float(np.nanmean(cb))


def r61(ctx):
    """R6.1 cells at block b: per R3.3 direction u, push +- a_DN(b, u) (PUSH-DN, the text loop's damage), 256 prompts; d_soft per prompt."""
    t = R.torch(); b = int(ctx.step.rsplit("_b", 1)[1]); m = R.M(ctx); P = R.p256(ctx); cap = float(R.r1_state(ctx, b)["ref"]["cap"])
    pids = [m.ids(p) for p in P["prompts"]]; out = dict(block=b, cells={})
    for u, d in dirs3(ctx, b, m).items():
        a = K.load_json(R.find(ctx, f"r3/b{b}/dose_{u}.json"))["a_DN"]["a"]
        X, cbar = ctx.cached(f"soft_{u}", lambda: soft_gen(m, b, [(p, s) for p in pids for s in (1, -1)], d, a, cap, m.nnew))
        rel = f"r6/b{b}/soft_{u}.pt"
        ctx.save_pt(rel, dict(xp=X[0::2].float(), xm=X[1::2].float(), prompt_of=t.arange(len(pids)), a=a, cbar=cbar, push=d.float(), block=b,
                              direction=u), config=dict(block=b, direction=u, a_DN=a, T=0.7, steps=m.nnew))
        out["cells"][u] = dict(file=rel, a=a, cbar=cbar)
    R.wj(ctx, f"r6/r61_b{b}.json", out, config=dict(block=b))


def r61_iter(ctx):
    """Iterated latent loop (R6.1): 4 rounds at PUSH-DN (calibrated per round) from unit(u_T^v) and from RND_1's start (R2.2)."""
    import r1b_r_s2
    t = R.torch(); b = int(ctx.step.rsplit("_b", 1)[1]); m = R.M(ctx); P = R.p256(ctx); cap = float(R.r1_state(ctx, b)["ref"]["cap"])
    dm = R.Dmg(ctx, m, b, cap); st = r1b_r_s2.starts(ctx, b, R.ax1b(ctx, b)); pids = [m.ids(p) for p in P["prompts"]]; out = dict(block=b, loops={})
    for L in ("RSI_a", "RND_1"):
        d, a, rows = st[L], R.alpha0(ctx, b), []
        for n in range(4):
            def one():
                cal = dm.calibrate(S.unit(d), a); X, cb = soft_gen(m, b, [(p, s) for p in pids for s in (1, -1)], S.unit(d), cal["a"], cap, m.nnew)
                return dict(xp=X[0::2].float(), xm=X[1::2].float(), calib=cal, cbar=cb, push=S.unit(d).float())
            ev = ctx.cached(f"iter_{L}_{n}", one); rows.append(ev); a = ev["calib"]["a"]
            d = ((ev["xp"].double() - ev["xm"].double()) / 2).nanmean(0)
        rel = f"r6/b{b}/iter_{L}.pt"; ctx.save_pt(rel, dict(rows=rows, prompt_of=t.arange(len(pids)), block=b, loop=L), config=dict(block=b, loop=L, rounds=4))
        out["loops"][L] = rel
    R.wj(ctx, f"r6/r61it_b{b}.json", out, config=dict(block=b))


def turns(ctx, m, P, t1, T, kmax, keyfn, b=14):
    """Turns k = 2..kmax without push: ids_k = ids_(k-1) + t_(k-1) + TURN2_SUFFIX ids (B-Q8), keys keyfn(k, q, s) shared within a pair;
    reads: context-free (the answer as the reply to the plain prompt, primary) and in-context (teacher-forced in the full context)."""
    import r2_stimuli as ST2
    t = R.torch(); suf = R.K.enc(m.tok, ST2.TURN2_SUFFIX); pids = [m.ids(p) for p in P["prompts"]]; n = len(t1["conts"]) // 2
    po = [int(x) for x in t1["prompt_of"]]; seeds = [int(x) for x in t1.get("seed_of", [s for _ in range(len(P["q"])) for s in range(n // len(P["q"]))])]
    ctxs = [list(pids[po[j // 2]]) for j in range(2 * n)]; prev = list(t1["conts"]); res = {}
    for k in range(2, kmax + 1):
        ctxs = [ST2.turn2_ids(c, pt, suf) for c, pt in zip(ctxs, prev)]
        rows = [(ctxs[j], keyfn(k, P["q"][po[j // 2]], seeds[j // 2]), 1) for j in range(2 * n)]
        conts = K.generate(m.model, m.tok, rows, T=T, max_new=m.nnew)
        Xf = R.read_win(m, [pids[po[j // 2]] for j in range(2 * n)], conts, b)[0]; Xi = R.read_win(m, ctxs, conts, b)[0]
        rt = sum(not ST2.round_trip_stable(lambda s: R.K.enc(m.tok, s), m.tok.decode, c) for c in prev)
        res[k] = dict(xp=Xf[0::2].float(), xm=Xf[1::2].float(), xp_ic=Xi[0::2].float(), xm_ic=Xi[1::2].float(), conts=conts, round_trip_failures=int(rt))
        prev = conts
    return res, po, seeds


def r62(ctx):
    """R6.2 at T (step name s6_r62_T<T>): turn 1 at T 0.7 = R3.3's PUSH-DN cell along unit(u_T^v(14)); at T 1.0 a new cell with keys
    ('r1b_k1', 14, 1.0, q, s); turns 2-6 keys ('r1b_turn', 14, T, k, q, s)."""
    t = R.torch(); T = float(ctx.step.rsplit("_T", 1)[1]); m = R.M(ctx); P = R.p256(ctx); b = 14; cap = float(R.r1_state(ctx, b)["ref"]["cap"])
    if T == 0.7:
        t1 = _cell(ctx, b, "T_v")
    else:
        def one():
            u = dirs3(ctx, b, m)["T_v"]; a = K.load_json(R.find(ctx, f"r3/b{b}/dose_T_v.json"))["a_DN"]["a"]; pids = [m.ids(p) for p in P["prompts"]]
            rows = [(pids[i], ctx.key("r1b_k1", 14, 1.0, q, s), sg) for i, q in enumerate(P["q"]) for s in range(2) for sg in (1, -1)]
            ev = R.gen_pairs(ctx, m, b, rows, u, a, cap, T=1.0); ev.update(prompt_of=t.tensor([i for i in range(len(P["q"])) for s in range(2)]),
                                                                          seed_of=t.tensor([s for _ in P["q"] for s in range(2)]), a=a); return ev
        t1 = ctx.cached("turn1_T1.0", one)
    t1 = dict(t1, seed_of=t1.get("seed_of", t.tensor([s for _ in P["q"] for s in range(2)])))
    res, po, seeds = ctx.cached(f"turns_T{T}", lambda: turns(ctx, m, P, t1, T, 6, lambda k, q, s: ctx.key("r1b_turn", 14, T, k, q, s)))
    ctx.save_pt(f"r6/r62_T{T}.pt", dict(T=T, turn1=dict(xp=t1["xp"], xm=t1["xm"], ok=t1["ok"], conts=t1["conts"], a=t1["a"]), turns=res, prompt_of=po,
                                        seed_of=seeds, q=P["q"]), config=dict(T=T, turns=6, key="('r1b_turn', 14, T, k, q, s)"))


def r62_rnd(ctx):
    """Random-direction carry control: turn 1 = R3.3's PUSH-DN cell along ('cov', 14, 4) at T 0.7; turns 2-3 keys ('r1b_turn_rnd', 14, k, q, s)."""
    m = R.M(ctx); P = R.p256(ctx); t1 = _cell(ctx, 14, "cov4"); t = R.torch()
    t1 = dict(t1, seed_of=t.tensor([s for _ in P["q"] for s in range(2)]))
    res, po, seeds = ctx.cached("turns_rnd", lambda: turns(ctx, m, P, t1, 0.7, 3, lambda k, q, s: ctx.key("r1b_turn_rnd", 14, k, q, s)))
    ctx.save_pt("r6/r62_rnd.pt", dict(T=0.7, turn1=dict(xp=t1["xp"], xm=t1["xm"], ok=t1["ok"], a=t1["a"], push=t1["push"]), turns=res, prompt_of=po,
                                      seed_of=seeds, q=P["q"]), config=dict(T=0.7, turns=3))


def r63(ctx):
    """R6.3: round 2's T 1.0 carry cell rebuilt with new keys and 4 x the prompts: push +- alpha_0(14) along u*_14 at T 1.0, keys
    ('r1b_krep', q, s); turn 2 keys ('r1b_krep2', q, s)."""
    t = R.torch(); m = R.M(ctx); P = R.p256(ctx); b = 14; cap = float(R.r1_state(ctx, b)["ref"]["cap"]); pids = [m.ids(p) for p in P["prompts"]]
    u = S.unit(R.ustar(ctx, b)[0]); a = R.alpha0(ctx, b)

    def one():
        rows = [(pids[i], ctx.key("r1b_krep", q, s), sg) for i, q in enumerate(P["q"]) for s in range(2) for sg in (1, -1)]
        ev = R.gen_pairs(ctx, m, b, rows, u, a, cap, T=1.0)
        ev.update(prompt_of=t.tensor([i for i in range(len(P["q"])) for s in range(2)]), seed_of=t.tensor([s for _ in P["q"] for s in range(2)]), a=a)
        return ev
    t1 = ctx.cached("turn1", one)
    res, po, seeds = ctx.cached("turn2", lambda: turns(ctx, m, P, t1, 1.0, 2, lambda k, q, s: ctx.key("r1b_krep2", q, s)))
    ctx.save_pt("r6/r63.pt", dict(T=1.0, turn1=dict(xp=t1["xp"], xm=t1["xm"], ok=t1["ok"], a=a, push=u.float()), turns=res, prompt_of=po, seed_of=seeds,
                                  q=P["q"]), config=dict(T=1.0, a=a, keys="('r1b_krep', q, s), ('r1b_krep2', q, s)"))
