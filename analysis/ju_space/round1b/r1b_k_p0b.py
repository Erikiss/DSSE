"""Part K items of the re-pilot r1b_P0b (K0 items 3-9 re-measured or measured for the first time; account B, before the freeze;
SO-PILOT material only: the 8 WikiText-103 pilot passages, the pilot prompts and random states; keys 'r1bP0_K'; every generated token
is discarded; engineering numbers only). Each item runs under r1b_pilot.Rec.item (an out-of-memory or other error is a value).

  rates   item 3: shared-prefix decode with a DynamicCache (B = 1 prefix copied to B rows, keyed Gumbel, 40 steps) at 528 and 2,064
          (B = 48) and 4,096 (B = 24); item 4: the 28 down_proj hooks (KB.Tap) at batch 32 x 512 with a warm-up, synchronised around
          each call, median of 5; item 5: FISTA on a synthetic 4,000 x 172,032 fp32 matrix, the step-size estimate _lip timed on its
          own and the per-iteration time from 60 against 10 iterations after a warm-up; keyed (Gumbel) two-stream contrastive rollouts
          at 80 / 160 / 320 seeds with and without capture; K's keyed decode (32 rows x 40 tokens, the backbone-rate reference)
  checks  item 6: the J-lens file (sha256, shapes, the unembed(h @ J[l].T) path); item 8 on the 8 pilot passages: patch identity,
          decode fronts = teacher-forced fronts, global position offset +4,096, frozen-QK reproduction; item 9: host RAM, local disk
          (Drive throughput: r1b_pilot.p0b_drive)
  q35     item 7: Qwen3.5-2B under transformers 5.22.2 with the pinned files (load time), G-K18 (the 15 mtp.* tensors loaded into K's
          head, shapes asserted), G-K17 (MTP wiring agreement >= 0.4 in the frozen order, on the pilot passages), fp32 decode rates of
          the backbone and of the MTP head (the x 2 risk factor against Qwen3-1.7B), its J-lens (sha256, shapes, unembed path)"""
import os, shutil, time
import numpy as np
import r1b_core as RC
import r1b_contracts as C
import r1b_k_base as KB
import r1b_k_hn as HN
import r1b_k_labels as KL
import r1b_k_lens as LN
import r1b_k_s3 as S3
import r1b_k_te as TE
import r1b_pilot as P
from r1b_k_base import N, T_


def _lens(tok, model):
    p0 = RC.enc(tok, KL.P0); ph = RC.enc(tok, KL.PH)[0]
    return LN.Lens(model, tok, dict(P0_ids=p0, ph_id=ph)), ph


def _passage_ids(ctx, tok, n_tok):
    return [(RC.enc(tok, t) * 4)[:n_tok] for t in P.wt_passage_texts(ctx)]


def _kdecode(ctx, model, tok, tag, rows=32, new=40):
    prs = [RC.chat_ids(tok, x) for x in P.prompts(ctx)][:rows]
    keys = [ctx.key("r1bP0_K", tag, i) for i in range(len(prs))]
    KB.decode(model, tok, prs, keys=keys, T=1.0, max_new=2, batch=rows)
    s = P.timed(lambda: KB.decode(model, tok, prs, keys=keys, T=1.0, max_new=new, batch=rows), reps=1, warm=0)
    return s, len(prs)


def rates(ctx):
    torch = T_(); tok, model = ctx.model(); dev = ctx.device; rec = P.Rec(); mock = ctx.mock
    from transformers import DynamicCache
    pids = _passage_ids(ctx, tok, 512 if not mock else 32); flat = sum(pids, [])
    for Lc, B in (((528, 48), (2064, 48), (4096, 24)) if not mock else ((40, 4), (60, 4), (80, 2))):
        def shared(Lc=Lc, B=B):
            pre = (flat * (Lc // max(len(flat), 1) + 1))[:Lc]; J = 40 if not mock else 4

            def go(steps):
                c1 = DynamicCache()
                with torch.no_grad(), RC.sdpa_efficient(dev):
                    o = model(input_ids=torch.tensor([pre], device=dev), past_key_values=c1, use_cache=True, logits_to_keep=1)
                    cache = DynamicCache()
                    for li, lay in enumerate(c1.layers):
                        cache.update(lay.keys.repeat(B, 1, 1, 1), lay.values.repeat(B, 1, 1, 1), li)
                    G = KB.Gumbel(dev, 1.0, [ctx.key("r1bP0_K", "pre", Lc, b) for b in range(B)])
                    x = G(o.logits[:, -1].float().repeat(B, 1)).argmax(-1); pos = torch.full((B, 1), Lc, device=dev)
                    P.sync(); t = time.time()
                    for _ in range(steps):
                        o = model(input_ids=x[:, None], position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
                        x = G(o.logits[:, -1].float()).argmax(-1); pos = pos + 1
                    P.sync()
                return time.time() - t
            go(2)
            rec["rates"][f"shared_prefix_ms_per_step_ctx{Lc}_B{B}"] = 1000 * go(J) / J
        rec.item(f"shared_prefix_{Lc}_{B}", shared)

    def hooks():
        ids = torch.tensor([(flat * 40)[i * 7:i * 7 + (512 if not mock else 32)] for i in range(32 if not mock else 4)], device=dev)
        f = lambda: model(input_ids=ids, logits_to_keep=1)
        with torch.no_grad():
            a = P.timed(f, reps=5, warm=1)
            with KB.Tap(model, lambda l, z, d: None):
                b = P.timed(f, reps=5, warm=1)
        rec["rates"].update(downproj_hook_overhead_ratio=b / max(a, 1e-12), downproj_plain_s_32x512=a, downproj_hooked_s_32x512=b,
                            downproj_hook_ms_per_1k_tokens=1000 * (b - a) / (ids.numel() / 1000))
    rec.item("downproj_hooks", hooks)

    def fista():
        X = torch.randn(N(ctx, 4000, 64), N(ctx, 172032, 512), device=dev); y = (torch.rand(X.shape[0], device=dev) > 0.5).long()
        rec["rates"]["fista_lip_s"] = P.timed(lambda: HN._lip(X, 1.0), reps=3, warm=1)
        HN.fista(X, y, 1.0, iters=5, tol=-1.0)
        t10 = P.timed(lambda: HN.fista(X, y, 1.0, iters=10, tol=-1.0), reps=1, warm=0)
        t60 = P.timed(lambda: HN.fista(X, y, 1.0, iters=60, tol=-1.0), reps=1, warm=0)
        rec["rates"]["fista_ms_per_iter"] = 1000 * (t60 - t10) / 50; rec["rates"]["fista_iters_timed"] = 50
    rec.item("fista", fista)
    lens, ph = _lens(tok, model); D = model.config.hidden_size; blk = 14 if len(lens.L) > 14 else 1
    for rows in ((80, 160, 320) if not mock else (4, 8, 16)):
        for cap in (True, False):
            def roll(rows=rows, cap=cap):
                g = torch.Generator(device=dev).manual_seed(rows)
                h = torch.randn(rows, D, device=dev, generator=g); J = N(ctx, 40, 4)
                keys = [ctx.key("r1bP0_K", "roll", rows, i) for i in range(rows)]
                f = lambda: lens.rollout(blk, h, h * 0, [ph] * rows, J=J, T=1.0, keys=keys, sets=False, fov=cap)
                rec["rates"][f"keyed_rollout_s_per_step_seeds{rows}_{'capture' if cap else 'nocapture'}"] = P.timed(f, reps=1, warm=1) / J
            rec.item(f"rollout_{rows}_{'cap' if cap else 'nocap'}", roll)

    def kdec():
        s, n = _kdecode(ctx, model, tok, "dec17", 32 if not mock else 4, 40 if not mock else 4)
        rec["rates"].update(kdecode_s_per_call_32x40=s, kdecode_ms_per_step=1000 * s / (40 if not mock else 4))
    rec.item("kdecode_17b", kdec)
    ctx.write_json("p0b/k_rates.json", rec, config=dict(shared=[[528, 48], [2064, 48], [4096, 24]], fista=[4000, 172032], seeds=[80, 160, 320]))


def _jlens_path(ctx, model, maps):
    """unembed(h @ J[l].T) for the pilot passages' block-l states: finite logits of the right shape."""
    torch = T_(); norm, W = KB.lm_parts(model)[0], model.get_output_embeddings().weight
    return bool(torch.isfinite((norm(maps) @ W.T.float()) if maps.dim() == 2 else maps).all())


def checks(ctx):
    torch = T_(); tok, model = ctx.model(); dev = ctx.device; rec = P.Rec(); mock = ctx.mock
    Ls = RC.layers_of(model); bl = 14 if len(Ls) > 14 else 1
    ids = torch.tensor(_passage_ids(ctx, tok, 128 if not mock else 24), device=dev)

    def jl():
        p = os.path.join(ctx.root, "jlens", RC.JLENS["name"])
        J = RC.load_jlens(p if os.path.exists(p) else None, mock=mock or None, D=model.config.hidden_size, n=len(Ls) - 1)
        mp = J["maps"]; rec["code_checks"]["jlens_sha256_ok"] = (J["sha256"] == RC.JLENS["sha256"]) if not mock else "MOCK"
        rec["code_checks"]["jlens_shape"] = list(mp.shape)
        with torch.no_grad():
            h = model(input_ids=ids[:2], output_hidden_states=True).hidden_states[bl + 1][:, :16].float()
            z = h @ mp[bl].to(dev).float().T
            rec["code_checks"]["jlens_unembed_finite"] = _jlens_path(ctx, model, z.reshape(-1, z.shape[-1]))
    rec.item("jlens", jl)

    def patch():
        with torch.no_grad():
            o = model(input_ids=ids, output_hidden_states=True); h = o.hidden_states[bl + 1][:, 10]
            hk = Ls[bl].register_forward_hook(lambda m, a_, o_: (lambda x: (x.__setitem__((slice(None), 10), h), x)[1])(o_[0] if isinstance(o_, tuple) else o_))
            try:
                o2 = model(input_ids=ids)
            finally:
                hk.remove()
        rel = float((o2.logits - o.logits).abs().max() / o.logits.abs().max()); rec["code_checks"].update(patch_identity_rel=rel, patch_identity_ok=rel < 1e-5)
    rec.item("patch_identity", patch)

    def fronts():
        from transformers import DynamicCache
        pre, n = ids[:, :ids.shape[1] // 2], (16 if not mock else 4)
        with torch.no_grad():
            cache = DynamicCache(); o = model(input_ids=pre, past_key_values=cache, use_cache=True); lg = [o.logits[:, -1].float()]
            x = lg[-1].argmax(-1); seq = [x]
            for k in range(n - 1):
                pos = torch.full((ids.shape[0], 1), pre.shape[1] + k, device=dev)
                o = model(input_ids=x[:, None], position_ids=pos, past_key_values=cache, use_cache=True); lg.append(o.logits[:, -1].float())
                x = lg[-1].argmax(-1); seq.append(x)
            full = torch.cat([pre, torch.stack(seq, 1)], 1); tf = model(input_ids=full).logits.float()[:, pre.shape[1] - 1:pre.shape[1] - 1 + n]
        dec = torch.stack(lg, 1); rel = float((dec - tf).abs().max() / tf.abs().max())
        rec["code_checks"].update(decode_vs_teacher_forced_rel=rel, decode_vs_teacher_forced_ok=rel < 1e-4)
    rec.item("decode_fronts", fronts)

    def offset():
        with torch.no_grad():
            pos = torch.arange(ids.shape[1], device=dev)[None].repeat(ids.shape[0], 1)
            a = model(input_ids=ids, position_ids=pos).logits.float().log_softmax(-1)
            b = model(input_ids=ids, position_ids=pos + 4096).logits.float().log_softmax(-1)
        tgt = ids[:, 1:, None]; d = float((a[:, :-1].gather(2, tgt) - b[:, :-1].gather(2, tgt)).abs().max())
        same = bool((a.argmax(-1) == b.argmax(-1)).all())
        rec["code_checks"].update(offset4096_top1_identical=same, offset4096_max_abs_dlogp=d, offset4096_ok=bool(same and d < 1e-3))
    rec.item("global_offset", offset)

    def fqk():
        r = TE.frozen_qk_check(model, ids[:2]); rec["code_checks"].update(frozen_qk_rel=r, frozen_qk_ok=r < 1e-5)
    rec.item("frozen_qk", fqk)

    def host():
        mi = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if l.split()[0].rstrip(":") in ("MemTotal", "MemAvailable")} \
            if os.path.exists("/proc/meminfo") else {}
        du = shutil.disk_usage(ctx.work)
        rec["env"].update(host_ram_total_gb=mi.get("MemTotal", 0) / 1e6, host_ram_available_gb=mi.get("MemAvailable", 0) / 1e6,
                          local_disk_free_gb=du.free / 1e9, local_disk_total_gb=du.total / 1e9)
    rec.item("host", host)
    ctx.write_json("p0b/k_checks.json", rec, config=dict(texts="SO-PILOT WikiText-103 passages (8)", block=bl))


def _wiring(model, head, seqs, wiring, pre, dev):
    """S3._wiring_agree on given sequences: top-1 x_(t+2) of the head (h(t), true x_(t+1)) against the backbone's greedy prediction."""
    torch = T_()
    from transformers import DynamicCache
    head.wiring = (wiring, "pre" if pre else "post"); agree, n = 0, 0; norm = KB.lm_parts(model)[0]; W = model.get_output_embeddings().weight
    rot = LN.rotary_of(model); emb = model.get_input_embeddings(); last = len(RC.layers_of(model)) - 1
    for s in seqs:
        ids = torch.tensor([s], device=dev)
        with RC.Hooks(model) as hk, torch.no_grad():
            hk.capture([last], cpu=False); lg = model(input_ids=ids).logits[0]; h = hk.states[last][0][0]
        t = torch.arange(16, ids.shape[1] - 2, device=dev)[:250]
        hid = h[t] if pre else norm(h[t])
        with torch.no_grad():
            out = head.step(emb(ids[0, t + 1]).float(), hid.float(), DynamicCache(), (t + 1)[:, None], rot)
            pred = (out @ W.T.float()).argmax(-1)
        agree += int((pred == lg[t + 1].argmax(-1)).sum()); n += len(t)
    return agree / max(n, 1)


def q35(ctx):
    import transformers
    torch = T_(); rec = P.Rec(); mock = ctx.mock; dev = ctx.device; st = {}
    RC.unload_models()
    rec["code_checks"]["transformers_5.22.2"] = transformers.__version__ == "5.22.2"

    def load():
        t = time.time(); st["tok"], st["model"] = ctx.model(C.M35); rec["timing"]["q35_load_s"] = time.time() - t
        rec["env"]["q35_class"] = type(st["model"]).__name__; rec["env"]["q35_layers"] = len(RC.layers_of(st["model"]))
    rec.item("q35_load", load)
    if "model" in st:
        tok, model = st["tok"], st["model"]

        def g18():
            head, g = S3.mtp_head(ctx, model); st["head"] = head
            shp = getattr(model, "r1b_mtp_shapes", None)
            rec["code_checks"].update(G_K18=g.get("G_K18"), G_K18_ok=g.get("G_K18") == "ok" or mock, mtp_tensors=g.get("n_tensors"),
                                      mtp_shapes_recorded=len(shp) if shp is not None else K_coded("NaN:mock" if mock else "NaN:missing"))
        rec.item("G_K18", g18)

        def dec():
            s, n = _kdecode(ctx, model, tok, "dec35", 32 if not mock else 4, 40 if not mock else 4)
            rec["rates"].update(q35_kdecode_s_per_call_32x40=s, q35_kdecode_ms_per_step=1000 * s / (40 if not mock else 4))
            r17 = (RC.load_json(ctx.path("p0b/k_rates.json"), {}) or {}).get("rates", {}).get("kdecode_s_per_call_32x40")
            if isinstance(r17, (int, float)) and r17 > 0:
                rec["rates"]["q35_over_17b_decode_time_ratio"] = s / r17
        rec.item("q35_decode", dec)
        seqs = _passage_ids(ctx, tok, 256 if not mock else 40)
        if "head" in st:
            def g17():
                res = []
                for w, pre in (("eh", False), ("he", False), ("eh", True)):
                    a = _wiring(model, st["head"], seqs, w, pre, dev); res.append(dict(wiring=w, pre=pre, agreement=a))
                    if a >= 0.4:
                        break
                rec["code_checks"].update(G_K17_pass=bool(res[-1]["agreement"] >= 0.4), G_K17_used=f"{res[-1]['wiring']}/{'pre' if res[-1]['pre'] else 'post'}")
                rec["parity"]["G_K17_agreement"] = [r["agreement"] for r in res]
            rec.item("G_K17", g17)

            def mtp_rate():
                D = model.config.get_text_config().hidden_size if hasattr(model.config, "get_text_config") else model.config.hidden_size
                n = N(ctx, 80, 4); h = torch.randn(n, D, device=dev, generator=torch.Generator(device=dev).manual_seed(35))
                J = N(ctx, 16, 3); sd = [int(seqs[0][i % len(seqs[0])]) for i in range(n)]
                f = lambda: LN.head_rollout(model, tok, st["head"], h, h * 0, sd, J=J)
                rec["rates"]["q35_mtp_head_s_per_step_seeds80"] = P.timed(f, reps=1, warm=1) / J
            rec.item("q35_mtp_rate", mtp_rate)

        def jl35():
            J = S3.jlens35(ctx, model); mp = J["maps"]
            pin = ((ctx.freeze.get("jlens") or {}).get("qwen3.5-2b") or {}).get("sha256")
            rec["code_checks"].update(q35_jlens_sha256_ok=(J["sha256"] == pin) if not mock else "MOCK", q35_jlens_shape=list(mp.shape))
            with torch.no_grad():
                ids = torch.tensor([seqs[0][:32]], device=dev); bl = len(RC.layers_of(model)) // 2
                hs = model(input_ids=ids, output_hidden_states=True).hidden_states[bl + 1][0].float()
                z = hs @ mp[bl].to(dev).float().T
            rec["code_checks"]["q35_jlens_unembed_finite"] = _jlens_path(ctx, model, z)
        rec.item("q35_jlens", jl35)
    st.clear(); RC.unload_models(); P.free()
    ctx.write_json("p0b/k_35.json", rec, config=dict(model=C.M35, pins=RC.model_pins(C.M35, ctx.freeze).get("revision")))


def K_coded(code):
    return RC.coded(code)
