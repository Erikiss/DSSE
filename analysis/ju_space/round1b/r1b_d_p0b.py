"""Part D items of the re-pilot r1b_P0b (account B, before the freeze; engineering numbers only, keys 'r1bP0_D'; every generated token
is discarded). Each item runs under r1b_pilot.Rec.item: an out-of-memory error or any other error is a recorded value.

  decode_rates  an untimed warm-up forward and decode, then per length 4,096 / 16,384 / 28,672 / 32,000 (random token ids, no analysis
                material): B = 1 prefill of L - 700 tokens (synchronised), the prefix offloaded to host memory (LCE.offload, as the
                sessions do), and keyed decode rates for B = 1..4 (a 700-token chunk over B cache copies, 32 tokens; the returned cache is
                dropped at once), each with DynamicCache and with the in-place KV buffer (engine.fast.lce_inplace_kv): rates of both and
                the GPU bit identity of the tokens and of every step's logits (plus the in-place prefill at the shortest and longest
                length); peak memory per item
  parity        on SO-PILOT's PG19 slices (book 25 at 16,384 and at 32,768 tokens): the last 700 tokens' log-probs on the crop path
                (another chunk, crop, the chunk) and on the copy path against one full recompute, and SO-LCE's chunked token_logprobs
                (2,048-token chunks, the L2a form) against one full causal pass (max and median |log-prob difference|, relative
                difference; OD-M14 and G-LCE-1 input)
  profile       a torch.profiler trace (8 decode steps) of the keyed decode at 32,000, B = 1: kernel names and their share of the GPU
                time, the attention kernel's share"""
import time
import numpy as np
import r1b_core as K
import r1b_d_stim as S
import r1b_pilot as P
from r1b_lce import LCE

TOL_MAX, TOL_MED = 0.02, 0.002           # D's G-LCE-1 tolerances (r1b_d_d1)


def _setup(ctx):
    tok, model = ctx.model(); tk = S.Tk(tok)
    return tok, model, tk.stops(model), S.cfg(ctx)


def decode_rates(ctx):
    torch = K._torch(); tok, model, stop, c = _setup(ctx); rec = P.Rec(); eng = LCE(model, inplace=False); eng_ip = LCE(model, inplace=True)
    Ls = (4096, 16384, 28672, 32000) if not c.mock else (300, 400, 500, 600)
    n, nch = (32, 700) if not c.mock else (8, 40)
    rng = np.random.default_rng(20261115)

    def warm():
        ids = [int(x) for x in rng.integers(1000, 100000, 512 if not c.mock else 32)]
        cache, lg = eng.fwd(ids)
        eng.decode(cache, lg[:, -1], [ctx.key("r1bP0_D", "warm", 0)], 0.7, 4, stop)
    rec.item("warmup", warm)
    for L in Ls:
        ids = [int(x) for x in rng.integers(1000, 100000, L - nch)]; chunk = [int(x) for x in rng.integers(1000, 100000, nch)]; st = {}

        def prefill():
            P.sync(); t = time.time(); base, lg = eng.fwd(ids); P.sync(); rec["rates"][f"prefill_s_L{L}"] = time.time() - t
            st["base"] = LCE.offload(base); st["lg"] = lg[0, -1].clone()
        rec.item(f"prefill_L{L}", prefill)
        if "base" not in st:
            continue
        if L in (Ls[0], Ls[-1]):
            def pre_ip():                                 # the in-place buffer's prefill (a view of the buffer) against the plain one
                P.sync(); t = time.time(); c2, l2 = eng_ip.fwd(ids); P.sync(); rec["rates"][f"prefill_s_L{L}_inplace"] = time.time() - t
                del c2
                rec["code_checks"][f"inplace_prefill_identical_L{L}"] = bool(P.same(st["lg"], l2[0, -1]))
            rec.item(f"inplace_prefill_L{L}", pre_ip)
        for B in (1, 2, 3, 4):
            for ip in (False, True):
                def dec(B=B, ip=ip):
                    e = eng_ip if ip else eng; lgs = []; f0 = e.fwd
                    e.fwd = lambda *a, **k: (lambda r: (lgs.append(r[1]), r)[1])(f0(*a, **k))       # per-step logits, kept on the device
                    try:
                        cb = LCE.copy(st["base"], B, inplace=ip); lgb = e.fwd(torch.tensor([chunk] * B), cb)[1]; del lgs[:]
                        P.sync(); t = time.time()
                        y = e.decode(cb, lgb[:, -1], [ctx.key("r1bP0_D", L, B, b) for b in range(B)], 0.7, n, stop)[0]
                        P.sync(); dt = time.time() - t
                    finally:
                        del e.fwd
                    tag = f"L{L}_B{B}" + ("_inplace" if ip else "")
                    rec["rates"][f"decode_tok_s_{tag}"] = B * n / dt; rec["rates"][f"decode_ms_step_{tag}"] = 1000 * dt / n
                    if not ip:
                        st["ref"] = (y, [x.clone() for x in lgs])
                    elif "ref" in st:
                        y0, l0 = st.pop("ref")
                        rec["code_checks"][f"inplace_identical_L{L}_B{B}"] = bool(P.same(y0, y) and len(l0) == len(lgs)
                                                                                 and all(P.same(u, v) for u, v in zip(l0, lgs)))
                    del cb, lgb, y, lgs[:]
                rec.item(f"decode_L{L}_B{B}" + ("_inplace" if ip else ""), dec)
            st.pop("ref", None)
        st.clear(); P.free()
    rec["env"].update(engine=eng.record(), lengths=list(Ls), n_new=n, chunk=nch)
    ctx.write_json("p0b/d_decode.json", rec, config=dict(lengths=list(Ls), n_new=n, chunk=nch, B=[1, 2, 3, 4]))


def _lp(lg, targets):
    """log-probs (float64) of targets[i] under the logits row i."""
    torch = K._torch()
    return torch.log_softmax(lg.double(), -1).gather(1, torch.as_tensor(targets, device=lg.device)[:, None])[:, 0].cpu().numpy()


def _cmp(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b))
    return dict(max_abs=float(d.max()), median_abs=float(np.median(d)), rel=float(d.max() / max(np.mean(np.abs(b)), 1e-12)), n=int(d.size))


def parity(ctx):
    import r1b_l_engine as E
    tok, model, stop, c = _setup(ctx); rec = P.Rec(); eng = LCE(model)
    Ls, nch = ((16384, 32768), 700) if not c.mock else ((64, 96), 16)
    rng = np.random.default_rng(20261116)
    for L in Ls:
        def run(L=L):
            ids = P.pg19_slices(ctx, tok, L)[0][1]
            pre, ch = ids[:-nch], ids[-nch:]; tgt = ch[1:]
            full = _lp(eng.fwd(ids, keep=nch)[1][0][:-1], tgt); P.free()
            base, _ = eng.fwd(pre); cp = LCE.copy(base)
            cop = _lp(eng.fwd(ch, cp, keep=nch)[1][0][:-1], tgt); del cp; P.free()
            other = [int(x) for x in rng.integers(1000, 100000, nch)]
            eng.fwd(other, base, keep=1); LCE.crop(base, len(pre))
            crp = _lp(eng.fwd(ch, base, keep=nch)[1][0][:-1], tgt); del base; P.free()
            for nm, v in (("crop", crp), ("copy", cop)):
                m = _cmp(v, full); m["passed_G-LCE-1_tol"] = bool(m["max_abs"] <= TOL_MAX and m["median_abs"] <= TOL_MED)
                rec["parity"].update({f"L{L}_{nm}_vs_full_{k}": v_ for k, v_ in m.items()})
            a = eng.token_logprobs(ids, chunk=2048 if not c.mock else 32); b = E.full_logprobs(model, ids)[1:]
            m = _cmp(a, b); m["ok_rel_1e-4"] = bool(m["rel"] < 1e-4)
            rec["parity"].update({f"L{L}_l2a_chunked_vs_full_{k}": v_ for k, v_ in m.items()})
        rec.item(f"parity_L{L}", run)
    rec["env"]["slices"] = "SO-PILOT PG19 book 25 (pilot_material.json)" if not c.mock else "MOCK random ids"
    ctx.write_json("p0b/d_parity.json", rec, config=dict(lengths=list(Ls), chunk=nch, tol=dict(max_abs=TOL_MAX, median_abs=TOL_MED, rel=1e-4)))


def profile(ctx):
    torch = K._torch(); tok, model, stop, c = _setup(ctx); rec = P.Rec(); eng = LCE(model)
    L, n = (32000, 8) if not c.mock else (200, 3)

    def run():
        from torch.profiler import profile as prof_, ProfilerActivity
        ids = [int(x) for x in np.random.default_rng(20261117).integers(1000, 100000, L)]
        cache, lg = eng.fwd(ids); eng.decode(cache, lg[:, -1], [ctx.key("r1bP0_D", "prof", 0)], 0.7, 2, stop)
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
        P.sync(); t = time.time()
        with prof_(activities=acts) as pr:
            eng.decode(cache, lg[:, -1], [ctx.key("r1bP0_D", "prof", 1)], 0.7, n, stop); P.sync()
        wall = time.time() - t
        dev = lambda e: float(getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0) or 0)
        ev = [e for e in pr.key_averages() if dev(e) > 0]
        tot = sum(dev(e) for e in ev) or 1.0
        top = sorted(ev, key=dev, reverse=True)[:10]
        att = [e for e in ev if any(s in e.key.lower() for s in ("fmha", "attention", "attn", "mem_eff", "flash"))
               and not e.key.startswith("aten::")]                 # the SDPA kernels (memory-efficient: fmha_cutlassF_..., AttentionKernel)
        rec["timing"].update(profiled_wall_s=wall, gpu_time_per_step_ms=tot / 1000 / n)
        rec["rates"]["attention_share_of_gpu_time"] = sum(dev(e) for e in att) / tot
        rec["env"]["top_kernels"] = [dict(name=e.key[:160], share=dev(e) / tot, calls=int(e.count)) for e in top]
        rec["env"]["attention_kernels"] = sorted({e.key[:160] for e in att})
    rec.item("profile_decode_L32000_B1", run)
    ctx.write_json("p0b/d_profile.json", rec, config=dict(L=L, steps=n, B=1))
