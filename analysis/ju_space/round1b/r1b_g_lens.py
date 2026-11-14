"""G-owned pilot, forward-Jacobian lenses, self-rollout and native MTP.

The derivatives are forward maps (no inverse). All-position perturbations pass
through the native model, including PLE, local/global masks and KV sharing.
"""
from __future__ import annotations

import contextlib
import math
from pathlib import Path
import time

import r1b_g_adapter as A
from r1b_g_engine import ModelEngine, content_hash, settings, text_rows


def mapped_blocks(engine, reference=(6, 14, 18, 22)):
    last = len(engine.layers) - 1
    return sorted(set(int(math.floor(b * last / 27 + 0.5)) for b in reference if b < 27))


def _windows(ctx, engine, key, count, length):
    import r1b_core as core
    rows = text_rows(ctx, key)
    result, seen = [], set()
    for row in rows:
        text = row["text"].strip()
        if text in seen:
            continue
        seen.add(text)
        ids = engine.encode(text, chat=False)
        if len(ids) >= length:
            result.append({"id": row["id"], "ids": ids[:length], "text_sha256": content_hash(text)})
        if len(result) == count:
            break
    if len(result) < count:
        raise core.NotRun(f"{key}: need {count} distinct windows of {length} tokens; found {len(result)}")
    return result


def _remaining(ctx):
    return ctx.time_left() if hasattr(ctx, "time_left") else float("inf")


def jacobian_blocks(engine, seq, row_block=4, source_start=16, first_row=0):
    """Yield [layer, output-row, input-coordinate] Jacobian blocks.

    Independent zero perturbations at every layer allow one forward graph to
    supply all layer maps. Batched VJPs build rows of the *forward* Jacobian;
    the model weights remain frozen. The small fit windows use math SDPA because
    some fused kernels do not implement batched autograd. The choice is recorded.
    """
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    A.require_gradients(engine.model, "GJ lens fit")
    if not 0 <= source_start < len(seq) or row_block < 1:
        raise ValueError("invalid Jacobian source positions/block size")
    D = int(engine.cfg.hidden_size)
    delta = torch.zeros((len(engine.layers), D), device=engine.device, dtype=torch.float32, requires_grad=True)
    handles, captured = [], {}
    try:
        for layer, module in enumerate(engine.layers):
            def perturb(module, args, output, layer=layer):
                x = output[0] if isinstance(output, tuple) else output
                mask = (torch.arange(x.shape[1], device=x.device) >= source_start)[None, :, None]
                y = x + mask * delta[layer][None, None, :]
                if layer == len(engine.layers) - 1:
                    captured["last"] = y
                return (y,) + output[1:] if isinstance(output, tuple) else y
            handles.append(module.register_forward_hook(perturb))
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            ids = torch.tensor([seq], device=engine.device)
            A.text_model(engine.model)(input_ids=ids, use_cache=False)
            target = captured["last"].sum((0, 1)) / (len(seq) - source_start)
            for start in range(int(first_row), D, int(row_block)):
                end = min(start + int(row_block), D)
                basis = torch.zeros((end - start, D), device=engine.device)
                basis[:, start:end] = torch.eye(end - start, device=engine.device)
                grad, = torch.autograd.grad(target, delta, grad_outputs=basis, is_grads_batched=True,
                                            retain_graph=end < D, create_graph=False)
                yield start, grad.permute(1, 0, 2).detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()


def fit_lens(ctx):
    import torch
    import r1b_core as core
    engine = ModelEngine(ctx)
    if getattr(engine.model, "_r1b_g_forward_only", False):
        return {"status": "not_applicable", "reason": "QAT target: gradient-based target J-lens excluded", "outputs": []}
    count, length = (2, 32) if ctx.mock else (1000, 128)
    rows = _windows(ctx, engine, "lens_train", count, length)
    D, L = int(engine.cfg.hidden_size), len(engine.layers)
    cfg = {"count": count, "length": length, "source_start": 16, "target": "sum of all final pre-norm states",
           "source": "shared additive block-output perturbation at positions 16..end",
           "kernel": "math SDPA for exact batched VJP", "corpus_sha256": content_hash(rows),
           "model": ctx.model_name, "model_pins":getattr(engine.model,"_r1b_g_pins",{}),
           "row_block": int(settings(ctx).get("jacobian_row_block", 4))}
    progress = Path(ctx.path("parts/lens_progress.pt"))
    if progress.exists():
        state = torch.load(progress, map_location="cpu", weights_only=True)
        if state["config_sha256"] != content_hash(cfg):
            raise ValueError("J-lens checkpoint belongs to different inputs/configuration")
    else:
        state = {"config_sha256": content_hash(cfg), "document": 0, "row": 0,
                 "even": torch.zeros(L, D, D), "odd": torch.zeros(L, D, D)}
    last_save = time.monotonic()
    for document in range(state["document"], count):
        first = state["row"] if document == state["document"] else 0
        blocks = jacobian_blocks(engine, rows[document]["ids"], cfg["row_block"], first_row=first)
        try:
            for start, block in blocks:
                target = state["even"] if document % 2 == 0 else state["odd"]
                target[:, start:start + block.shape[1]] += block
                state.update(document=document, row=start + block.shape[1])
                if state["row"] == D:
                    state.update(document=document + 1, row=0)
                if time.monotonic() - last_save >= 60 or _remaining(ctx) < 30:
                    ctx.save_pt("parts/lens_progress.pt", state, config=cfg)
                    last_save = time.monotonic()
                if _remaining(ctx) < 30:
                    raise core.StepYield()
        finally:
            blocks.close()
    maps = (state["even"] + state["odd"]) / count
    if not torch.allclose(maps[-1], torch.eye(D), atol=1e-5, rtol=1e-5):
        raise RuntimeError("J-lens identity control failed at the final layer")
    even = state["even"] / math.ceil(count / 2)
    odd = state["odd"] / math.floor(count / 2)
    reliability = ((even - odd).flatten(1).norm(dim=1) / maps.flatten(1).norm(dim=1).clamp(min=1e-12)).tolist()
    ctx.save_pt("lens.pt", {"maps": maps, "convention": "output", "layers": list(range(L)), "n_prompts": count,
                            "d_model": D, "split_half_relative_difference": reliability,
                            "training_ids": [r["id"] for r in rows],
                            "training_text_sha256": [r["text_sha256"] for r in rows], "config": cfg}, config=cfg)
    ctx.write_json("lens_fit.json", {"status": "done", "identity_pass": True,
                                    "split_half_relative_difference": reliability, "n_prompts": count}, config=cfg)
    return {"status": "done", "outputs": ["lens.pt", "lens_fit.json"]}


def _residual_sum(engine, seq, source_block, vector=None, source_start=16):
    import torch
    captured, handles = {}, []
    if vector is not None:
        def perturb(module, args, output):
            x = output[0] if isinstance(output, tuple) else output
            y = x.clone()
            y[:, source_start:] += vector.to(x.device, x.dtype)
            return (y,) + output[1:] if isinstance(output, tuple) else y
        handles.append(engine.layers[source_block].register_forward_hook(perturb))
    def capture(module, args, output):
        captured["h"] = output[0] if isinstance(output, tuple) else output
    handles.append(engine.layers[-1].register_forward_hook(capture))
    try:
        with torch.no_grad():
            A.text_model(engine.model)(input_ids=torch.tensor([seq], device=engine.device), use_cache=False)
            return (captured["h"].sum((0, 1)) / (len(seq) - source_start)).float()
    finally:
        for handle in handles:
            handle.remove()


def validate_lens(ctx):
    import torch
    engine = ModelEngine(ctx)
    if getattr(engine.model, "_r1b_g_forward_only", False):
        return {"status": "not_applicable", "reason": "no target J-lens for QAT arm", "outputs": []}
    artifact = torch.load(ctx.artifact("lens_fit", "lens.pt"), map_location="cpu", weights_only=True)
    count, length = (2, 32) if ctx.mock else (64, 128)
    rows = _windows(ctx, engine, "lens_validation", count, length)
    if set(r["id"] for r in rows) & set(artifact["training_ids"]):
        raise ValueError("lens fitting and validation item IDs overlap")
    if set(r["text_sha256"] for r in rows) & set(artifact.get("training_text_sha256", [])):
        raise ValueError("lens fitting and validation texts overlap")
    D = int(engine.cfg.hidden_size)
    g = torch.Generator().manual_seed(1305)
    vectors = torch.randn(2 if ctx.mock else 8, D, generator=g)
    vectors /= vectors.norm(dim=-1, keepdim=True)
    scores = {"output": [], "input": []}
    for layer in mapped_blocks(engine):
        for vector in vectors:
            for convention, source in (("output", layer), ("input", layer - 1)):
                if source < 0:
                    continue
                estimates = []
                for row in rows:
                    h = engine.hidden([row["ids"]], [source])[0][source]
                    eps = float(h[16:].norm(dim=-1).mean()) * 1e-2
                    if eps == 0:
                        raise RuntimeError("zero residual norm in lens bridge")
                    plus = _residual_sum(engine, row["ids"], source, vector * eps)
                    minus = _residual_sum(engine, row["ids"], source, -vector * eps)
                    estimates.append(((plus - minus) / (2 * eps)).cpu())
                estimated = torch.stack(estimates).mean(0)
                predicted = artifact["maps"][layer] @ vector
                cosine = float(torch.nn.functional.cosine_similarity(estimated, predicted, dim=0))
                scores[convention].append(cosine)
    medians = {k: float(torch.tensor(v).quantile(0.5)) for k, v in scores.items() if v}
    # These maps were fitted to block outputs. The input comparison diagnoses an
    # off-by-one convention; it cannot authorize output-state use when that
    # convention fails its own finite-difference bridge.
    convention = "output"
    passed = medians[convention] >= 0.9
    obj = {"G_K_J": {"pass": passed, "median_cos": medians, "cos": scores,
                      "string": f"J-lens bridge holds ({convention})" if passed else "J-lens bridge failed"},
           "convention": convention, "n_prompts": count}
    ctx.write_json("lens_validation.json", obj, config={"threshold": 0.9, "seed": 1305,
                   "validation_sha256": content_hash(rows), "blocks": mapped_blocks(engine)})
    return {"status": "done", "outputs": ["lens_validation.json"]}


def frame_spec(engine):
    text = ("The following is a neutral description of an ordinary day, with a quiet river, "
            "a path through the village, a small stone bridge, and trees beside the water.")
    prefix = engine.encode(text, chat=False)
    placeholder = engine.encode(" ...", chat=False)
    if len(prefix) < 16 or not placeholder:
        raise ValueError("cannot tokenize the declared neutral lens frame")
    return {"prefix_text": text, "prefix_ids": prefix[:16], "placeholder_text": " ...",
            "placeholder_id": placeholder[-1], "frame_ids": prefix[:16] + [placeholder[-1]]}


def penultimate_norm(ctx,engine):
    """Common L-instrument scale on independent WT validation fronts 16..127."""
    import torch
    rows=_windows(ctx,engine,"lens_validation",2 if ctx.mock else 64,32 if ctx.mock else 128)
    block=len(engine.layers)-2
    values=[engine.hidden([row["ids"]],[block])[0][block][16:].norm(dim=-1) for row in rows]
    return float(torch.cat(values).quantile(.5))


def rollout(engine, block, state, seed_ids, steps=16, variant="E", prior_state=None, frame_ids=None,
            key=0, temperature=None, lam=1.0, plaus=0.1, forced=None, keep_distributions=False, keep_first=False,
            observer=None):
    """K's expert-minus-prior self-rollout through native Gemma forwards.

    The state replaces the fixed placeholder's block output, not every generated
    token. Both streams receive the same selected continuation. Full-prefix
    recomputation preserves native PLE/shared-KV semantics without cache surgery.
    """
    import torch
    if variant not in {"E", "L"} or not 0 < plaus <= 1 or steps < 1:
        raise ValueError("invalid self-rollout variant/plausibility/length")
    seeds = [int(s) for s in seed_ids]
    n, D = len(seeds), int(engine.cfg.hidden_size)
    if n == 0:
        raise ValueError("self-rollout requires at least one seed")
    state = torch.as_tensor(state).reshape(-1, D)
    state = state.expand(n, -1) if len(state) == 1 else state
    prior = torch.zeros_like(state) if prior_state is None else torch.as_tensor(prior_state).reshape(-1, D)
    prior = prior.expand(n, -1) if len(prior) == 1 else prior
    if len(state) != n or len(prior) != n:
        raise ValueError("one real/prior state is required per seed")
    if frame_ids is None:
        frame_ids = frame_spec(engine)["frame_ids"]
    frame = list(frame_ids)
    if not frame:
        raise ValueError("lens frame requires a placeholder")
    target_block = int(block) if variant == "E" else len(engine.layers) - 2
    special = set(engine.tok.all_special_ids)
    rows, firsts = [], []
    for r, seed in enumerate(seeds):
        tokens, records, all_logp = [], {k: [] for k in ("lpr", "lpp", "ent", "top1", "cdm", "e", "alt")}, []
        generator = torch.Generator(device=engine.device).manual_seed(int(key) + r)
        for step in range(steps):
            seq = frame + [seed] + tokens
            with engine.patch_state(target_block, state[r], len(frame) - 1):
                real = engine.next_logits(seq).log_softmax(-1)
            with engine.patch_state(target_block, prior[r], len(frame) - 1):
                base = engine.next_logits(seq).log_softmax(-1)
            score = real - lam * base
            valid_real = real.clone()
            valid_real[list(special)] = -torch.inf
            score[real < valid_real.max() + math.log(plaus)] = -torch.inf
            score[list(special)] = -torch.inf
            if step == 0 and keep_first:
                firsts.append({"lr": real.cpu(), "lp": base.cpu(), "plaus": torch.isfinite(score).cpu()})
            if forced is not None and step < len(forced[r]):
                token = int(forced[r][step])
            else:
                token = engine.choose(score, generator, 0.0 if temperature is None else temperature)
            if observer is not None:
                observer(r, step, token)
            tokens.append(token)
            second = score.clone()
            second[token] = -torch.inf
            records["alt"].append(int(second.argmax()))
            records["lpr"].append(float(real[token])); records["lpp"].append(float(base[token]))
            records["ent"].append(float(-(real.exp() * real).sum()))
            records["top1"].append(float(real.exp().max()))
            # Forced tokens can be outside the plausible set; represent this
            # exact quantity in tensors and let report writers code non-finites.
            records["cdm"].append(float(score[token] - second.max()))
            records["e"].append(float(real[token] - base[token]))
            if keep_distributions:
                all_logp.append(real.cpu())
        rows.append({"tok": torch.tensor(tokens), **{k: torch.tensor(v, dtype=torch.long if k == "alt" else torch.float32) for k, v in records.items()},
                     "log_probs": torch.stack(all_logp) if all_logp else None})
    return {"tok": torch.stack([r["tok"] for r in rows]),
            **{k: torch.stack([r[k] for r in rows]) for k in ("lpr", "lpp", "ent", "top1", "cdm", "e", "alt")},
            "log_probs": torch.stack([r["log_probs"] for r in rows]) if keep_distributions else None,
            "first": {k: torch.stack([f[k] for f in firsts]) for k in ("lr", "lp", "plaus")} if firsts else None,
            "frame_ids": frame, "variant": variant}


def pilot(ctx):
    import torch
    import r1b_core as core
    from transformers import DynamicCache
    start = time.monotonic()
    if str(ctx.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    engine = ModelEngine(ctx)
    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)
    load_seconds = time.monotonic() - start
    rows = text_rows(ctx, "pilot_texts", minimum=2 if ctx.mock else 4, limit=2 if ctx.mock else 4)
    raw_seqs = [engine.encode(r["text"], chat=False) for r in rows]
    if any(len(s) < 2 for s in raw_seqs):
        raise ValueError("pilot text too short")
    width = 32 if ctx.mock else 128
    seqs = [(s * math.ceil(width/len(s)))[:width] for s in raw_seqs]
    started = time.monotonic()
    engine.teacher_force(seqs, populations={"probe": [0, engine.layout.total - 1]}, blocks=[0])
    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)
    forward_seconds = time.monotonic() - started
    started = time.monotonic()
    generated = engine.decode([s[:min(16, len(s))] for s in seqs],
                              keys=[ctx.key("pilot", r["id"]) for r in rows], max_new=4 if ctx.mock else 32)
    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)
    decode_seconds = time.monotonic() - started
    def timing(fn):
        if engine.device.type == "cuda":
            torch.cuda.synchronize(engine.device)
        start = time.monotonic()
        value = fn()
        if engine.device.type == "cuda":
            torch.cuda.synchronize(engine.device)
        return time.monotonic()-start, value
    selector_seconds, _ = timing(lambda:engine.features([seqs[0]], [list(range(width))]))
    with torch.no_grad():
        h = torch.zeros((16,int(engine.cfg.hidden_size)),device=engine.device)
        head_seconds, _ = timing(lambda:A.unembed(engine.model,h))
    jacobian, assistant = None, None
    if ctx.model_name in {"gemma4-e2b","gemma4-e4b"}:
        def derivative_probe():
            blocks = jacobian_blocks(engine,seqs[0],row_block=4)
            try:
                return next(blocks)[1]
            finally:
                blocks.close()
        derivative_seconds, _ = timing(derivative_probe)
        jacobian = {"rows":4,"seconds":derivative_seconds,"tokens":width,
                    "projected_1000_windows_seconds":derivative_seconds*math.ceil(int(engine.cfg.hidden_size)/4)*1000,
                    "projection_caveat":"conservative independent-graph row-block projection; fit reuses each graph"}
        assistant_load, pair = timing(lambda:ctx.model(ctx.model_name+"-assistant"))
        bridge = A.AssistantBridge(engine.model,pair[1])
        state, _ = bridge.prefill(torch.tensor([seqs[0][:16]],device=engine.device))
        def draft_probe():
            current=state
            for _ in range(8):
                _,current=bridge.step(torch.tensor([[seqs[0][16]]],device=engine.device),current)
        assistant_seconds,_=timing(draft_probe)
        assistant={"load_seconds":assistant_load,"draft_steps":8,"seconds":assistant_seconds,
                   "steps_per_second":8/max(assistant_seconds,1e-9)}
    window = int(engine.cfg.sliding_window)
    length = min(2 * window + 8, int(engine.cfg.max_position_embeddings))
    seq = (seqs[0] * math.ceil(length / len(seqs[0])))[:length]
    ids = torch.tensor([seq], device=engine.device)
    cache, cached = DynamicCache(config=engine.cfg), []
    with torch.no_grad(), core.sdpa_efficient(engine.device):
        full = A.text_model(engine.model)(input_ids=ids, use_cache=False).last_hidden_state.float()
        for begin in range(0, length, max(window // 2, 1)):
            end = min(begin + max(window // 2, 1), length)
            out = A.text_model(engine.model)(input_ids=ids[:, begin:end], use_cache=True, past_key_values=cache,
                    attention_mask=torch.ones((1, end), dtype=torch.long, device=engine.device),
                    position_ids=torch.arange(begin, end, device=engine.device)[None])
            cache = out.past_key_values
            cached.append(out.last_hidden_state.float())
    joined = torch.cat(cached, 1)
    tol = 8 * torch.finfo(torch.float16).eps if getattr(engine.model, "_r1b_g_forward_only", False) else 1e-5
    parity = bool(torch.allclose(full, joined, atol=tol, rtol=tol))
    long_context=[]
    from r1b_g_l_native import full_logprobs,lce_logprobs
    import numpy as np
    for n in ((64,128) if ctx.mock else (16384,32768)):
        if n>int(engine.cfg.max_position_embeddings):
            long_context.append({"tokens":n,"status":"unsupported native context"})
            continue
        sequence=(seqs[0]*math.ceil(n/len(seqs[0])))[:n]
        try:
            elapsed,reference=timing(lambda:full_logprobs(engine.model,sequence))
            cached_elapsed,chunked=timing(lambda:lce_logprobs(engine.model,sequence,chunk=16 if ctx.mock else 2048))
            finite=np.isfinite(reference)&np.isfinite(chunked)
            relative_error=float(np.max(np.abs(reference[finite]-chunked[finite]))/max(np.mean(np.abs(reference[finite])),1e-12))
            long_context.append({"tokens":n,"status":"measured","seconds":elapsed,"tokens_per_second":n/elapsed,
                "cached_seconds":cached_elapsed,"cached_tokens_per_second":n/cached_elapsed,
                "relative_error":relative_error,"cache_parity":relative_error<1e-4,"threshold":1e-4,
                "metric":"max absolute next-token logprob difference / mean absolute full-pass logprob",
                "scope":"neutral engineering pilot; scientific PG19 LCE independently rechecks this metric"})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            long_context.append({"tokens":n,"status":"out of memory","seconds":None})
    obj = {"engineering_only": True, "mock": bool(ctx.mock), "load_seconds": load_seconds,
           "teacher_tokens_per_second": sum(map(len, seqs)) / max(forward_seconds, 1e-9),
           "decode_tokens_per_second": sum(len(x["ids"]) for x in generated) / max(decode_seconds, 1e-9),
           "cache_parity": parity, "cache_max_abs_error": float((full - joined).abs().max()),
           "cache_tokens": length, "sliding_window": window,
           "peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if engine.device.type == "cuda" else None,
           "gpu": torch.cuda.get_device_name(engine.device) if engine.device.type == "cuda" else "CPU mock",
           "layout": engine.layout.as_dict(), "attention": A.attention_layout(engine.model)}
    obj.update(selector={"seconds":selector_seconds,"neurons":engine.layout.total,"tokens":width,
                         "tokens_per_second":width/max(selector_seconds,1e-9)},
               head={"seconds":head_seconds,"states":16},jacobian=jacobian,assistant=assistant,long_context=long_context,
               input_policy="fixed repetition of neutral pilot tokens for engineering throughput only; no scientific evidence")
    ctx.write_json("pilot.json", obj, config={"cache_atol_rtol": tol, "input_sha256": content_hash(rows)})
    if not parity:
        raise RuntimeError("G pilot native cache parity failed")
    return {"status": "done", "outputs": ["pilot.json"]}


def native_rollout(engine, bridge, context, real_hidden, prior_hidden, seed_ids, *, steps=16,
                   beta=0.0, lam=1.0, plaus=0.1, self_kl=False):
    """Gemma assistant recursion on identical fixed target KV for both streams.

    Unlike the context-free Qwen MTP instrument, Gemma's official assistant
    consumes the target context's shared KV. The same context/anchor is used for
    both real and prior streams, so the only intervention is the initial hidden
    state (plus the declared beta sensitivity). The chosen token enters both
    streams at the next step. q is the full target on the generated prefix.
    """
    import torch
    from dataclasses import replace
    if steps < 1 or not seed_ids or not 0 < plaus <= 1:
        raise ValueError("invalid native rollout length/seeds/plausibility")
    real_hidden = torch.as_tensor(real_hidden, device=engine.device).reshape(1, 1, -1)
    prior_hidden = torch.as_tensor(prior_hidden, device=engine.device).reshape(1, 1, -1)
    base, _ = bridge.prefill(torch.tensor([context], device=engine.device))
    real_hidden = real_hidden.to(base.hidden.dtype)
    prior_hidden = prior_hidden.to(base.hidden.dtype)
    special = list(engine.tok.all_special_ids)
    tokens, records, divergences = [], [], []
    for seed in seed_ids:
        real = replace(base, hidden=real_hidden)
        prior = replace(base, hidden=prior_hidden)
        x, generated = int(seed), []
        rec = {k: [] for k in ("lpr", "lpp", "ent", "top1", "e")}
        divergence = []
        for step in range(steps):
            if step and beta:
                real = replace(real, hidden=real.hidden + float(beta) * (real_hidden - prior_hidden))
            xr = torch.tensor([[x]], device=engine.device)
            lr, real = bridge.step(xr, real)
            lp, prior = bridge.step(xr, prior)
            lr, lp = lr[0, 0].float().log_softmax(-1), lp[0, 0].float().log_softmax(-1)
            allowed = lr.clone(); allowed[special] = -torch.inf
            score = lr - float(lam) * lp
            score[(lr < allowed.max() + math.log(plaus))] = -torch.inf
            score[special] = -torch.inf
            if self_kl:
                lq = engine.next_logits(list(context) + [int(seed)] + generated).log_softmax(-1)
                divergence.append(float((lq.exp() * (lq - lr)).sum()))
            x = engine.choose(score)
            generated.append(x)
            for key, value in (("lpr", lr[x]), ("lpp", lp[x]), ("ent", -(lr.exp() * lr).sum()),
                               ("top1", lr.exp().max()), ("e", lr[x] - lp[x])):
                rec[key].append(float(value))
        tokens.append(torch.tensor(generated, dtype=torch.long))
        records.append({k: torch.tensor(v) for k, v in rec.items()})
        divergences.append(divergence)
    return {"tok": torch.stack(tokens), **{k: torch.stack([r[k] for r in records]) for k in records[0]},
            "kl": torch.tensor(divergences) if self_kl else None, "beta": beta}


def calibration_stats(rows, calibration_ids, analysis_ids, *, B=10000, steps=16, selfkl=()):
    """Inherited K5 estimands with disjoint variant selection and inference.

    CL35 is retained only as an internal schema label for the existing statistic
    definitions. Every exported observation also names its native Gemma target.
    """
    import numpy as np
    import r1b_k_ana as K
    from r1b_core import val, coded
    cal, held = set(calibration_ids), set(analysis_ids)
    if cal & held:
        raise ValueError("MTP calibration and analysis IDs overlap")
    curve = lambda rs: np.asarray([[float(any(c == "CONFAB" and s <= k for s, _, c in r["claims"]))
                                   for k in range(1, steps + 1)] for r in rs]).mean(0)
    crows = [r for r in rows if r["item"] in cal and r["variant"] == "V1"]
    Ccal = {name: curve([r for r in crows if r["lens"] == name]) for name in ("CL35", "E", "L")
            if any(r["lens"] == name for r in crows)}
    if len(Ccal) != 3:
        return {"G_K5": "not decidable", "variant": "E", "labels": ["missing disjoint calibration observations"]}
    iad = {name: float(np.abs(Ccal[name] - Ccal["CL35"]).mean()) for name in ("E", "L")}
    variant = min(iad, key=iad.get)
    rs = [r for r in rows if r["item"] in held]
    grouped = K.by(rs, "lens", "variant")
    C = {f"{lens}|{v}": [val(x) for x in curve(part)] for (lens, v), part in grouped.items()}
    recall, confab = {}, {}
    for instrument, part in grouped.items():
        by_item = K.by(part, "item")
        recall[instrument] = {i: K.anyhit(rr) for (i,), rr in by_item.items()}
        confab[instrument] = {i: float(np.mean([any(c == "CONFAB" for _, _, c in r["claims"]) for r in rr]))
                             for (i,), rr in by_item.items()}
    dc = K.item_diff(confab.get((variant, "V1"), {}), confab.get(("CL35", "V1"), {}), B)
    dr = K.item_diff(recall.get((variant, "V1"), {}), recall.get(("CL35", "V1"), {}), B)
    pc = K.item_diff(recall.get(("CL35", "V1"), {}), recall.get(("CL35", "V2"), {}), B)
    labels = []
    if not (K.fin(pc["ci"]["lo"]) and pc["ci"]["lo"] > 0):
        verdict = "not decidable"; labels.append("native lens fails its own positive control")
    elif K.inside(dc["ci"], .10) and K.inside(dr["ci"], .10):
        verdict = "self-rollout calibrated"
    elif any((K.gt(d["ci"]) or K.lt(d["ci"])) and K.fin(d["est"]) and abs(d["est"]) > .10 for d in (dc, dr)):
        verdict = "self-rollout differs from the native chain lens"
    else:
        verdict = "calibration undecided"
    slope, draws = K.logit_slope(K.claim_counts(grouped.get(("CL35", "V1"), [])), B)
    interval = K.ci(draws) if len(draws) else {"lo": coded("NaN:no_claims"), "hi": coded("NaN:no_claims")}
    trend = ("native: per-step confabulation rises with chain length" if K.gt(interval) else
             "native: flat per-step confabulation" if K.inside(interval, .03) else "undecided")
    kl = [r["kl"] for r in selfkl if r["item"] in held]
    return {"G_K5": verdict, "variant": variant, "IAD": iad, "C_k": C, "dC16": K._s(dc),
            "drecall": K._s(dr), "native_pc": K._s(pc), "labels": labels,
            "calibration_ids": list(calibration_ids), "analysis_ids": list(analysis_ids),
            "Q6": {"string": trend, "slope": val(slope), "ci": interval,
                   "p": K.pboot(draws) if len(draws) else 1.0},
            "selfkl": {"mean_by_step": np.mean(kl, axis=0).tolist() if kl else None,
                       "note": "KL(q target || p native Gemma assistant), held-out V1; descriptive"},
            "trunk": "length trend without trunk dynamics" if trend.startswith("native: per-step") and
                     verdict == "self-rollout calibrated" else None}


def _mtp_chains(ctx, engine, bridge, lens):
    import torch
    import r1b_core as core
    from r1b_g_k_chain import normalize_item, gate_arithmetic, foreign_indices, score_tokens, _seeds
    data = ctx.data()
    raw = data.get("ach_short", data.get("arithmetic", []))
    if not raw:
        raise core.NotRun("MTP calibration requires G raw arithmetic items")
    pool = [normalize_item(x) for x in raw if x.get("split", "A-lens") == "A-lens"]
    if len({x["id"] for x in pool}) != len(pool):
        raise ValueError("duplicate arithmetic IDs")
    gates = gate_arithmetic(ctx, engine, pool)
    cal = [x for x in pool if x.get("cal") and gates[x["id"]]["stratum"] == "correct"][:2 if ctx.mock else 64]
    held = [x for x in pool if not x.get("cal") and gates[x["id"]]["stratum"] == "correct"][:2 if ctx.mock else 256]
    meta = {"calibration_ids": [x["id"] for x in cal], "analysis_ids": [x["id"] for x in held],
            "gates": gates, "required_calibration": 2 if ctx.mock else 64,
            "required_analysis": 2 if ctx.mock else 256, "rows": [], "selfkl": [], "n_items": len(held)}
    if len(cal) < meta["required_calibration"] or len(held) < meta["required_analysis"]:
        return meta, {"G_K5": "not decidable", "variant": "E", "labels": ["insufficient greedy-correct arithmetic items"],
                      "calibration_correct": len(cal), "analysis_correct": len(held)}
    blocks = mapped_blocks(engine, (10, 14, 18))
    steps, nseeds = (4, 2) if ctx.mock else (16, 10)
    frame = frame_spec(engine)
    meta.update(blocks=blocks, frame=frame, steps=steps,
                native_context="same target prompt shared KV and fixed RoPE anchor for real/prior; foreign V2 changes state+seed only",
                lens_schema={"CL35": "native Gemma assistant beta_N=0", "CL35_b1": "native Gemma assistant beta_N=1",
                             "E": "self-rollout source layer", "L": "self-rollout penultimate layer"})
    norm_l=ctx.cached("mtp_penultimate_norm",lambda:penultimate_norm(ctx,engine))
    meta["norm_l"]=norm_l
    for split, items in (("calibration", cal), ("analysis", held)):
        prompts = [engine.encode(it["prompt"]) for it in items]
        state_rows = engine.hidden(prompts, blocks + [len(engine.layers) - 2])
        states = {b: torch.stack([r[b][-1] for r in state_rows]) for b in blocks}
        foreign, matched = foreign_indices([x["id"] for x in items], {x["id"]: x for x in items}, seed=51)
        meta[split + "_foreign"] = {x["id"]: {"source": items[foreign[i]]["id"], "matched": matched[i]}
                                     for i, x in enumerate(items)}
        for block in blocks:
            hs = states[block]
            means = (hs.sum(0, keepdim=True) - hs) / (len(hs) - 1)
            seeds = [_seeds(engine, h, block, lens, nseeds) for h in hs]
            matrix = lens["maps"][block]
            for i, item in enumerate(items):
                for variant in ("V1", "V2"):
                    j = i if variant == "V1" else foreign[i]
                    key = content_hash([ctx.model_name, item["id"], items[j]["id"], block, variant, frame, steps])
                    def compute(i=i, j=j, item=item, variant=variant, block=block):
                        real_raw = hs[j] @ matrix.T
                        prior_raw = means[j] @ matrix.T
                        norm = A.text_model(engine.model).norm
                        with torch.no_grad():
                            real = norm(real_raw.to(engine.device)[None]).float()[0]
                            prior = norm(prior_raw.to(engine.device)[None]).float()[0]
                        outs = {"CL35": native_rollout(engine, bridge, prompts[i], real, prior, seeds[j], steps=steps,
                                                       self_kl=variant == "V1"),
                                "CL35_b1": native_rollout(engine, bridge, prompts[i], real, prior, seeds[j], steps=steps, beta=1),
                                "E": rollout(engine, block, hs[j], seeds[j], steps=steps, prior_state=means[j], frame_ids=frame["frame_ids"]),
                                "L": rollout(engine, block, torch.nn.functional.normalize(real_raw, dim=-1) * norm_l,
                                             seeds[j], steps=steps, variant="L", prior_state=torch.nn.functional.normalize(prior_raw, dim=-1) * norm_l,
                                             frame_ids=frame["frame_ids"])}
                        result, divergence = [], []
                        for instrument, out in outs.items():
                            for rank, seed in enumerate(seeds[j]):
                                tokens = [seed] + out["tok"][rank].tolist()
                                scored = score_tokens(engine.tok, tokens, item, OWN=gates[item["id"]]["own"])
                                result.append({"item": item["id"], "source_item": items[j]["id"], "split": split,
                                               "block": block, "variant": variant, "lens": instrument, "target": ctx.model_name,
                                               "seed_rank": rank, "seed": seed, "tok": tokens,
                                               "claims": scored["claims"], "censor": scored["censor"], "e": out["e"][rank]})
                        if variant == "V1":
                            for rank, kl in enumerate(outs["CL35"]["kl"][:4]):
                                divergence.append({"item": item["id"], "block": block, "seed_rank": rank, "kl": kl.tolist()})
                        return result, divergence
                    result, divergence = ctx.cached("mtp_chains_" + key, compute)
                    meta["rows"].extend(result); meta["selfkl"].extend(divergence)
                    if _remaining(ctx) < 30:
                        raise core.StepYield()
    result = calibration_stats(meta["rows"], meta["calibration_ids"], meta["analysis_ids"],
                               B=200 if ctx.mock else core.N_BOOT, steps=steps, selfkl=meta["selfkl"])
    if ctx.mock:
        result["G_K5"] = "not decidable"
        result.setdefault("labels", []).append("random-model engineering fixture, no scientific verdict")
    return meta, result


def mtp_calibration(ctx):
    """Official assistant wiring, disjoint K5 calibration and held-out inference."""
    import json
    import torch
    import r1b_core as core
    engine = ModelEngine(ctx)
    if ctx.model_name not in {"gemma4-e2b", "gemma4-e4b"}:
        return {"status": "not_applicable", "reason": "no original assistant assigned to this QAT target", "outputs": []}
    _, drafter = ctx.model(ctx.model_name + "-assistant")
    bridge = A.AssistantBridge(engine.model, drafter)
    texts = text_rows(ctx, "mtp_texts", minimum=1)
    cap, records = (4 if ctx.mock else 2000), []
    for item in texts:
        seq = engine.encode(item["text"], chat=False)[:256]
        wanted = min(cap - len(records), max(len(seq) - 18, 0))
        def compute(item=item, seq=seq, wanted=wanted):
            rows = []
            for position in range(16, 16 + wanted):
                prefix = seq[:position + 1]
                state, _ = bridge.prefill(torch.tensor([prefix], device=engine.device))
                seed = int(seq[position + 1])
                draft, _ = bridge.step(torch.tensor([[seed]], device=engine.device), state)
                target = engine.next_logits(prefix + [seed])
                lp, lq = draft[0, 0].float().log_softmax(-1), target.log_softmax(-1)
                rows.append({"item": item["id"], "position": position,
                             "agreement": bool(lp.argmax() == lq.argmax()), "self_kl": float((lq.exp() * (lq - lp)).sum())})
            return rows
        records.extend(ctx.cached("mtp_alignment_" + content_hash([item, wanted]), compute))
        if len(records) == cap:
            break
        if _remaining(ctx) < 30:
            raise core.StepYield()
    if len(records) < cap:
        raise core.NotRun(f"MTP wiring requires {cap} held-out positions; got {len(records)}")
    agreement = sum(r["agreement"] for r in records) / len(records)
    passed = agreement >= .4
    obj = {"G_K17": {"pass": passed, "agreement": agreement, "n_positions": len(records),
                      "string": "native assistant wiring holds" if passed else "not run (MTP wiring failed)"},
           "native_wiring": "[target embedding(seed); preceding normalized hidden], shared target KV, fixed anchor",
           "alignment_rows": records, "mean_self_kl": sum(r["self_kl"] for r in records) / len(records),
           "G_K5": "not decidable", "variant": "E", "labels": []}
    validation_path = ctx.artifact("lens_validate", "lens_validation.json")
    validation = json.loads(Path(validation_path).read_text())
    lens = torch.load(ctx.artifact("lens_fit", "lens.pt"), map_location="cpu", weights_only=True)
    if not passed:
        obj["labels"].append("not run (MTP wiring failed); not calibrated")
        chains = {"rows": [], "selfkl": [], "n_items": 0, "reason": "MTP wiring failed"}
    elif not validation["G_K_J"]["pass"]:
        obj["labels"].append("not run (G J-lens bridge failed); not calibrated")
        chains = {"rows": [], "selfkl": [], "n_items": 0, "reason": "G J-lens bridge failed"}
    else:
        chains, result = _mtp_chains(ctx, engine, bridge, lens)
        obj.update(result)
    ctx.save_pt("mtp_chains.pt", chains, config={"assistant": ctx.model_name + "-assistant", "lam": 1.0, "plaus": .1,
                "beta_native": [0, 1], "selection_split": "A-lens cal", "inference_split": "A-lens not cal"})
    ctx.write_json("mtp_calibration.json", obj, config={"assistant": ctx.model_name + "-assistant", "threshold": .4,
                   "input_sha256": content_hash(texts), "random_mock_is_scientific_evidence": False})
    return {"status": "done", "outputs": ["mtp_calibration.json", "mtp_chains.pt"]}

