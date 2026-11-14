"""CPU bit-identity tests of the three engine switches engine.fast.* (MOCK tiny random model; never a real model):

  l_decode_nosync     r1b_l_engine.decode with nosync True against False: padded and unpadded batches, 32 steps, keyed Gumbel, early
                      stops (a wide stop set) with extra steps after the last stop, a probe with channels and spins plus spin_fn, an
                      on_step window, mask_eos and greedy; bitwise equality of every per-step logit row (lm_head outputs), every returned
                      quantity (tokens, n, ended, e, peos, s, e0, s0, a, a2, z) and the per-row generator states after the call; the
                      4-D mask against the mask transformers builds itself (captured at layer 0)
  lce_inplace_kv      SO-LCE with the in-place KV buffer against DynamicCache: prefill, chunk over the cached prefix, crop, crop_layers,
                      copy (B = 3), offload + copy, keyed decode of 32 tokens with block means, token_logprobs (chunked); bitwise equality of
                      logits, tokens and means
  k_gumbel_kept_rows  K's keyed contrastive rollout (r1b_k_lens.Lens.rollout) with the noise drawn for the kept rows only against all
                      2N rows: tokens and every per-step record equal; the kept noise itself equal over 32 steps

    MOCK=1 python tests/fast_identity.py <root>      (<root>/models/qwen3-1.7b: config and tokenizer)"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); RB = os.path.dirname(HERE)
sys.path.insert(0, RB)
os.environ.setdefault("MOCK", "1")
import numpy as np
import r1b_core as K
import r1b_contracts as C

ERR = []


def ok(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        ERR.append(msg)


def eq(a, b):
    """Bitwise equality of nested results (r1b_pilot.same: tensors and arrays bit for bit, NaN == NaN)."""
    import r1b_pilot as P
    return P.same(a, b)


def capture(model):
    """lm_head outputs per forward and the attention mask layer 0's attention receives."""
    log = dict(lg=[], mask=[]); L = K.layers_of(model)
    h1 = model.lm_head.register_forward_hook(lambda m, a, o: log["lg"].append(o.detach().clone()))
    h2 = L[0].self_attn.register_forward_pre_hook(lambda m, a, kw: log["mask"].append(kw.get("attention_mask")), with_kwargs=True)
    return log, (h1, h2)


def test_decode(tok, model):
    import r1b_l_engine as E
    torch = K._torch(); dims = E.dims(model)
    base = [K.chat_ids(tok, f"Describe pilot scene {i} with a few details about the weather.") for i in range(6)]
    padded = [b[:len(b) - 3 * (i % 3)] for i, b in enumerate(base)]
    unpadded = [b[:min(len(x) for x in base)] for b in base]
    keys = [K.keyseed("r1bP0_fast", "dec", i) for i in range(6)]
    g = np.random.default_rng(0); L, I = dims
    rw = {k: torch.ones(L * I) if k.startswith("sd") else torch.zeros(L * I) for k in ("mu_z", "sd_z", "mu_cett", "sd_cett")}
    chans = [(f"S{k}", sorted(g.choice(L * I, 8, replace=False).tolist())) for k in range(3)]
    spins = sorted(g.choice(L * I, 12, replace=False).tolist())
    orig_stop = K.stop_tokens
    cases = [("unpadded", unpadded, {}, False), ("padded", padded, {}, False),
             ("padded, wide stop set (early stops, extra steps)", padded, {}, True),
             ("padded, probe + spin_fn + on_step + advance, wide stop set", padded, dict(probe=True, on_step=True, advance=[0, 1, 2, 0, 1, 2]), True),
             ("unpadded, mask_eos", unpadded, dict(mask_eos=True), False), ("padded, greedy, wide stop set", padded, dict(greedy=True), True)]
    cases += [(f"padded, probe + on_step, stop set of {w} ids", padded, dict(probe=True, on_step=True, wide=w), w) for w in (5000, 9000, 14000)]
    extra = []
    for name, rows, kw, wide in cases:
        outs = {}
        for nosync in (False, True):
            if wide:
                K.stop_tokens = lambda t, m, w=(40000 if wide is True else wide): (orig_stop(t, m)[0], set(range(0, w)))
            pr = E.Probe(model, rw, chans=chans, spins=spins) if kw.get("probe") else None
            if pr is not None:
                pr.add_op(chans[0][1], "scale", val=2.0, on=False)
            st = []

            def on_step(step, probe, st=st):
                st.append(step)
                for o in probe.ops:
                    o["on"] = step <= 5
            log, hs = capture(model); gs = []
            try:
                r = E.decode(model, tok, rows, keys, T=0.7, max_new=32, probe=pr, mask_eos=kw.get("mask_eos", False), advance=kw.get("advance"),
                             greedy=kw.get("greedy", False), on_step=on_step if kw.get("on_step") else None,
                             spin_fn=(lambda s_, z: z.float().cpu().numpy()) if pr is not None else None, nosync=nosync, gen_out=gs)
            finally:
                for h in hs:
                    h.remove()
                K.stop_tokens = orig_stop
            outs[nosync] = dict(r=r, log=log, gs=gs, ops=[dict(on=o["on"]) for o in (pr.ops if pr else [])], steps=list(st))
            if pr is not None:
                pr.remove()
        a, b = outs[False], outs[True]; n = len(a["log"]["lg"])
        ok(eq(a["r"], b["r"]), f"decode [{name}]: every returned quantity equal ({sum(x['n'] for x in a['r'])} tokens, "
                               f"{sum(x['ended'] for x in a['r'])} rows stopped)")
        ok(len(b["log"]["lg"]) >= n and all(eq(x, y) for x, y in zip(a["log"]["lg"], b["log"]["lg"][:n])),
           f"decode [{name}]: per-step logits equal on {n} forwards (nosync ran {len(b['log']['lg'])})")
        ok(eq(a["gs"], b["gs"]), f"decode [{name}]: per-row generator states equal after the call")
        ma, mb = a["log"]["mask"][:n], b["log"]["mask"][:n]
        ok(all((x is None and y is None) or (x is not None and y is not None and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
                                             and x.stride() == y.stride()) for x, y in zip(ma, mb)),
           f"decode [{name}]: attention masks equal ({'None' if ma[0] is None else tuple(ma[0].shape)} at the prefill)")
        if kw.get("on_step"):
            ok(eq(a["ops"], b["ops"]), f"decode [{name}]: probe ops state after the call equal (on_step re-called at the last step)")
        extra.append(len(b["log"]["lg"]) - n)
    ok(max(extra) > 0, f"decode: the no-sync path ran extra steps after the last stop in some case (extra forwards {extra}) and stayed equal")


def test_lce(tok, model):
    from r1b_lce import LCE
    torch = K._torch(); g = np.random.default_rng(1)
    pre = [int(x) for x in g.integers(1000, 30000, 300)]; ch = [int(x) for x in g.integers(1000, 30000, 40)]
    other = [int(x) for x in g.integers(1000, 30000, 40)]; stop = sorted(K.stop_tokens(tok, model)[1])
    res = {}
    for ip in (False, True):
        e = LCE(model, inplace=ip); r = {}
        c, lg = e.fwd(pre); r["prefill"] = lg
        c2 = LCE.copy(c, 3); c2, r["copy3"] = e.fwd(torch.tensor([ch] * 3), c2, keep=5)
        _, r["chunk_other"] = e.fwd(other, c, keep=3); LCE.crop(c, len(pre))
        c, r["crop_then_chunk"] = e.fwd(ch, c, keep=40)
        LCE.crop_layers(c, len(pre), len(K.layers_of(model)) - 1); _, r["crop_layers_chunk"] = e.fwd(ch, c, keep=40)
        LCE.crop(c, len(pre)); LCE.offload(c); c3 = LCE.copy(c, 2); c3, lg3 = e.fwd(torch.tensor([ch] * 2), c3)
        y, last, c3, am, means = e.decode(c3, lg3[:, -1], [K.keyseed("r1bP0_fast", "lce", b) for b in range(2)], 0.7, 32, stop, blocks=(1, 2))
        r.update(dec_tokens=y, dec_last=last, dec_means=means, dec_len=c3.get_seq_length())
        r["token_logprobs"] = torch.as_tensor(e.token_logprobs(pre + ch, chunk=64))
        cx = LCE.copy(c, 2, inplace=not ip); cx, lgx = e.fwd(torch.tensor([ch] * 2), cx)        # a copy converted to the other layer kind
        yx, lastx, cx, _, _ = e.decode(cx, lgx[:, -1], [K.keyseed("r1bP0_fast", "lce", b) for b in range(2)], 0.7, 32, stop)
        r["converted_kind"] = sorted({type(l).__name__ for l in cx.layers})
        ok(eq(yx, y) and eq(lastx, last), f"lce: a copy converted to {r['converted_kind']} decodes identically (inplace={ip})")
        r["kinds"] = sorted({type(l).__name__ for l in c3.layers})
        res[ip] = r
    a, b = res[False], res[True]
    ok(b["kinds"] == ["InplaceLayer"] and a["kinds"] == ["DynamicLayer"], f"lce: cache layers {a['kinds']} / {b['kinds']}")
    for k in ("prefill", "copy3", "chunk_other", "crop_then_chunk", "crop_layers_chunk", "dec_tokens", "dec_last", "dec_means", "dec_len",
              "token_logprobs"):
        ok(eq(a[k], b[k]), f"lce [{k}]: in-place KV buffer equal to DynamicCache")


def test_kept_rows(tok, model):
    import r1b_k_labels as KL
    import r1b_k_lens as LN
    from r1b_k_base import Gumbel
    torch = K._torch(); dev = K.model_device(model)
    lens = LN.Lens(model, tok, dict(P0_ids=K.enc(tok, KL.P0), ph_id=K.enc(tok, KL.PH)[0])); N = 6; D = model.config.hidden_size
    h = torch.randn(N, D, generator=torch.Generator().manual_seed(3)); keys = [K.keyseed("r1bP0_fast", "roll", i) for i in range(N)]
    out = {kr: lens.rollout(1, h, h * 0.5, [lens.ph] * N, J=32, T=1.0, keys=keys, sets=False, fov=True, kept_rows=kr) for kr in (False, True)}
    ok(eq(out[False], out[True]), f"kept rows: rollout tokens and every per-step record equal over 32 steps ({sorted(out[True])})")
    G2, G1 = Gumbel(dev, 1.0, keys * 2), Gumbel(dev, 1.0, keys); V = model.get_output_embeddings().weight.shape[0]
    ok(all(eq(G2(torch.zeros(2 * N, V))[:N], G1(torch.zeros(N, V))) for _ in range(32)), "kept rows: the kept noise is equal over 32 draws")
    out = {kr: lens.rollout(1, h, h * 0.5, [lens.ph] * N, J=32, T=1.0, keys=keys, sets=False, fov=False, kept_rows=kr) for kr in (False, True)}
    ok(eq(out[False], out[True]), "kept rows: rollout without capture (NaN fov records) equal bit for bit")


def test_masks(model):
    """mask4 against the mask transformers builds (create_causal_mask from a 2-D padding mask)."""
    import r1b_l_engine as E
    from transformers import DynamicCache
    from transformers.masking_utils import create_causal_mask
    torch = K._torch(); am = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]])
    emb = torch.zeros(3, 6, model.config.hidden_size)
    m0 = create_causal_mask(config=model.config, inputs_embeds=emb, attention_mask=am, past_key_values=DynamicCache(config=model.config),
                            position_ids=(am.cumsum(-1) - 1).clamp_min(0))
    ok(m0 is not None and torch.equal(m0, E.mask4(am, 6)) and m0.dtype == E.mask4(am, 6).dtype, "mask4 equals transformers' prefill mask")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("R1B_ROOT", "/content/root")
    K._torch().manual_seed(0)
    tok, model = K.load_model(C.M17, root=root, device="cpu", mock=True)
    test_masks(model); test_decode(tok, model); test_lce(tok, model); test_kept_rows(tok, model)
    print("FAST IDENTITY " + ("OK" if not ERR else f"FAILED: {ERR}"), flush=True)
    sys.exit(1 if ERR else 0)
