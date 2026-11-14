"""Part L items of the re-pilot r1b_P0b (L0 (i), (iii), (iv), (vii) re-measured; account B, before the freeze; SO-PILOT material
only, keys 'r1bP0_L0' / 'r1bP0_L0duet', every generated text discarded, engineering numbers only). Each item runs under
r1b_pilot.Rec.item (an out-of-memory or other error is a recorded value).

  decode_rates  L0 (i): E.decode at batch 32 on padded pilot prompts (row lengths ctx - 16 (i mod 8)) at contexts 512 and 1,100, plain
                and hooked (L's full capture set on stand-in random sets of the planned sizes), after an untimed warm-up; the prefill
                (decode's step 0, the same forward) is timed on its own and the decode step is (total - prefill) / steps; each with
                engine.fast.l_decode_nosync off and on (rates of both) and the switch's GPU bit identity (every returned quantity, every
                step's logits, the generator states after the call; with and without stops; padded and unpadded rows)
  long          L0 (iii): B = 1 prefill of the PG19 pilot slices at 16,384 and 32,768 tokens with and without the capture hooks
                (probe.begin() called, so the hooks compute) and the chunked L2a pass (E.lce_logprobs, 2,048-token chunks); L0 (iv): one
                duet message (batch 32, 40 new tokens) at a ~420-token history, re-prefill against a carried cache (prefill of the
                history timed on its own); L0 (vii): L2a chunked against a full pass on 2 pilot slices at 16,384 and 32,768 (relative
                difference < 1e-4: OD-M14)"""
import time
import numpy as np
import r1b_core as K
import r1b_l_engine as E
import r1b_pilot as P
from r1b_l_base import cfg, dims


def _sets(model, n_pop=102, size=60, seed=0):
    L, I = dims(model)
    g = np.random.default_rng(seed)
    return [(f"S{k}", sorted(g.choice(L * I, min(size, L * I // 4), replace=False).tolist())) for k in range(n_pop)]


def _rw(model):
    torch = K._torch(); L, I = dims(model)
    return {k: torch.ones(L * I) if k.startswith("sd") else torch.zeros(L * I) for k in ("mu_z", "sd_z", "mu_cett", "sd_cett")}


def _prefill(model, tok, rows, probe=None):
    """Decode's step 0 on its own: the same padded forward (attention mask, positions, a new DynamicCache, logits_to_keep=1)."""
    torch = K._torch()
    from transformers import DynamicCache
    dev = K.model_device(model); ids, am, _ = K.pad_left(rows, tok.pad_token_id); ids, am = ids.to(dev), am.to(dev)
    pos = (am.cumsum(-1) - 1).clamp_min(0)
    if probe is not None:
        probe.begin()
    with torch.no_grad():
        model(input_ids=ids, attention_mask=am, position_ids=pos, past_key_values=DynamicCache(), use_cache=True, logits_to_keep=1)
    if probe is not None:
        probe.buf = None


def _rows(prs, n, ctxlen, mock):
    return [(p * (ctxlen // max(len(p), 1) + 2))[:ctxlen - ((16 * (i % 8)) if not mock else 2 * (i % 3))] for i, p in enumerate(prs[:n])]


def decode_rates(ctx):
    torch = K._torch(); tok, model = ctx.model(); mock = ctx.mock; rec = P.Rec()
    B, steps = (32, 32) if not mock else (4, 6)
    prs = [K.chat_ids(tok, x) for x in P.prompts(ctx)]
    rw = _rw(model); sets = _sets(model, 102 if not mock else 6, 60 if not mock else 8)
    spins = [j for _, s in sets for j in s][: (102 * 60 if not mock else 48)]
    warm = _rows(prs, B, 64 if not mock else 16, mock)
    rec.item("warmup", lambda: E.decode(model, tok, warm, [ctx.key("r1bP0_L0", "warm", i) for i in range(len(warm))], T=1.0, max_new=4,
                                        stop=False))
    for ctxlen in ((512, 1100) if not mock else (16, 32)):
        rows = _rows(prs, B, ctxlen, mock); keys = [ctx.key("r1bP0_L0", "i", i, ctxlen) for i in range(len(rows))]
        for hooked in (False, True):
            tag = f"ctx{ctxlen}_{'hooked' if hooked else 'plain'}"

            def run(rows=rows, keys=keys, hooked=hooked, tag=tag):
                pr = E.Probe(model, rw, chans=sets, spins=spins) if hooked else None
                try:
                    pf = P.timed(lambda: _prefill(model, tok, rows, pr), reps=3, warm=1)
                    for ns in (False, True):                 # engine.fast.l_decode_nosync off / on: rates of both
                        P.sync(); t = time.time()
                        E.decode(model, tok, rows, keys, T=1.0, max_new=steps, probe=pr, stop=False,
                                 spin_fn=(lambda st, z: z.cpu()) if hooked else None, nosync=ns)
                        P.sync(); tot = time.time() - t; tg = tag + ("_nosync" if ns else "")
                        rec["rates"][f"decode_ms_per_step_{tg}"] = 1000 * (tot - pf) / steps; rec["rates"][f"decode_call_s_{tg}"] = tot
                finally:
                    if pr:
                        pr.remove()
                rec["rates"][f"prefill_s_{tag}"] = pf
            rec.item(f"decode_{tag}", run)

            def ident(rows=rows, keys=keys, hooked=hooked, tag=tag):
                """GPU bit identity of the switch: every returned quantity, every step's logits, the generator states after the call;
                with stop tokens (stop=True, a stop set widened so that rows stop at different steps) and without."""
                out = {}
                for wide in (False, True):
                    for ns in (False, True):
                        pr = E.Probe(model, rw, chans=sets, spins=spins) if hooked else None
                        lgs, gs = [], []; h = model.lm_head.register_forward_hook(lambda m, a, o: lgs.append(o))
                        st0 = K.stop_tokens
                        if wide:
                            K.stop_tokens = lambda t_, m_: (st0(t_, m_)[0], set(st0(t_, m_)[1]) | set(range(0, 20000)))
                        try:
                            r = E.decode(model, tok, rows, keys, T=1.0, max_new=steps, probe=pr, stop=wide,
                                         spin_fn=(lambda st, z: z.float().cpu().numpy()) if hooked else None, nosync=ns, gen_out=gs)
                        finally:
                            K.stop_tokens = st0; h.remove()
                            if pr:
                                pr.remove()
                        out[(wide, ns)] = (r, lgs, gs)
                    (ra, la, ga), (rb, lb, gb) = out[(wide, False)], out[(wide, True)]
                    same = (P.same(ra, rb) and len(lb) >= len(la) and all(P.same(x, y) for x, y in zip(la, lb))
                            and all(P.same(x, y) for x, y in zip(ga, gb)))
                    rec["code_checks"][f"nosync_identical_{tag}{'_stops' if wide else ''}"] = bool(same)
                    out.clear()
            rec.item(f"nosync_identity_{tag}", ident)
    def unpadded():
        rows = [(p * (512 // max(len(p), 1) + 2))[:512 if not mock else 16] for p in prs[:B]]; keys = [ctx.key("r1bP0_L0", "u", i) for i in range(B)]
        out = {}
        for ns in (False, True):
            lgs, gs = [], []; h = model.lm_head.register_forward_hook(lambda m, a, o: lgs.append(o))
            try:
                out[ns] = (E.decode(model, tok, rows, keys, T=1.0, max_new=steps, nosync=ns, gen_out=gs), lgs, gs)
            finally:
                h.remove()
        (ra, la, ga), (rb, lb, gb) = out[False], out[True]
        rec["code_checks"]["nosync_identical_unpadded_ctx512"] = bool(P.same(ra, rb) and len(lb) >= len(la) and all(P.same(x, y) for x, y in zip(la, lb))
                                                                     and all(P.same(x, y) for x, y in zip(ga, gb)))
    rec.item("nosync_identity_unpadded", unpadded)
    rec["env"].update(batch=B, steps=steps, padded=True, rows="pilot prompts repeated to ctx - 16 (i mod 8) tokens")
    ctx.write_json("p0b/l_decode.json", rec, config=dict(batch=B, steps=steps))


def long(ctx):
    torch = K._torch(); tok, model = ctx.model(); mock = ctx.mock; rec = P.Rec(); cf = cfg(ctx)
    from transformers import DynamicCache
    dev = K.model_device(model); rw = _rw(model); sets = _sets(model, 102 if not mock else 4, 60 if not mock else 8)
    for Ln in (16384, 32768):
        def pre(Ln=Ln):
            docs = P.pg19_slices(ctx, tok, Ln)
            d = torch.tensor([docs[0][1]], device=dev)
            for hooked in (False, True):
                pr = E.Probe(model, rw, chans=sets) if hooked else None
                try:
                    def f():
                        if pr is not None:
                            pr.begin()
                        with torch.no_grad(), K.sdpa_efficient(dev):
                            model(input_ids=d, position_ids=torch.arange(d.shape[1], device=dev)[None], past_key_values=DynamicCache(),
                                  use_cache=True, logits_to_keep=1)
                        if pr is not None:
                            rec["code_checks"][f"hooks_computed_{Ln}"] = bool(pr.buf and "a" in pr.buf); pr.buf = None
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    rec["rates"][f"prefill_B1_s_{Ln}_{'hooked' if hooked else 'plain'}"] = P.timed(f, reps=1, warm=0)
                    if torch.cuda.is_available():
                        rec["memory"][f"prefill_B1_peak_gb_{Ln}_{'hooked' if hooked else 'plain'}"] = torch.cuda.max_memory_allocated() / 1e9
                finally:
                    if pr:
                        pr.remove()
                P.free()
            rec["rates"][f"l2a_chunked_s_{Ln}"] = P.timed(lambda: E.lce_logprobs(model, docs[0][1], chunk=cf["eqmi_chunk"]), reps=1, warm=0)
        rec.item(f"prefill_{Ln}", pre)

    def duet():
        prs = [E.chat_sys_ids(tok, "You are Agent A, a researcher.", " ".join(P.prompts(ctx)[i:] + P.prompts(ctx)[:i])) for i in range(32 if not mock else 4)]
        hist = [(p * (420 // max(len(p), 1) + 2))[:420 if not mock else 24] for p in prs]
        keys = [ctx.key("r1bP0_L0duet", "msg", i) for i in range(len(hist))]
        pf = P.timed(lambda: _prefill(model, tok, hist), reps=3, warm=1)
        P.sync(); t = time.time(); E.decode(model, tok, hist, keys, T=0.7, max_new=40 if not mock else 6); P.sync(); tot = time.time() - t
        rec["rates"].update(duet_message_s_reprefill=tot, duet_history_prefill_s=pf, duet_message_s_carried_estimate=tot - pf,
                            duet_history_tokens=len(hist[0]), duet_batch=len(hist))
    rec.item("duet_message_420", duet)
    for Ln in (16384, 32768):
        def l2a(Ln=Ln):
            docs = P.pg19_slices(ctx, tok, Ln); rel = []
            for f, d in docs[:2]:
                a = E.lce_logprobs(model, d, chunk=cf["eqmi_chunk"] if not mock else 32); b = E.full_logprobs(model, d)
                m = np.isfinite(a) & np.isfinite(b); dd = np.abs(a[m] - b[m])
                rel.append(float(dd.max() / max(np.mean(np.abs(b[m])), 1e-12)))
                rec["parity"][f"l2a_{Ln}_{len(rel) - 1}_max_abs"] = float(dd.max()); rec["parity"][f"l2a_{Ln}_{len(rel) - 1}_median_abs"] = float(np.median(dd))
                P.free()
            rec["parity"][f"l2a_rel_{Ln}"] = max(rel); rec["code_checks"][f"l2a_ok_{Ln}"] = bool(max(rel) < 1e-4)
        rec.item(f"l2a_parity_{Ln}", l2a)
    cc = rec["code_checks"]
    if isinstance(cc.get("l2a_ok_16384"), bool) and isinstance(cc.get("l2a_ok_32768"), bool):
        rec["notes"].append("OD-M14 per its rule: " + ("full_grid" if cc["l2a_ok_32768"] else ("grid_2^14" if cc["l2a_ok_16384"] else
                                                                                            "neither length passes (G-L2a-LCE)")))
    rec["env"]["slices"] = "SO-PILOT PG19 books 25-28 (pilot_material.json)" if not mock else "MOCK random ids"
    ctx.write_json("p0b/l_long.json", rec, config=dict(lengths=[16384, 32768], chunk=cf["eqmi_chunk"], tol=1e-4))
