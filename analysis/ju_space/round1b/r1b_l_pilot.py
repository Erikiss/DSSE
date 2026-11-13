"""L0 PILOT (P0-L items of r1b_P0, account B; engineering only, SO-PILOT material, every text discarded): hooked decode against
unhooked decode at batch 32 and contexts 0.5k / 1.1k with the full L capture set on stand-in random sets of the planned sizes, R1 / R3
read rates, B = 1 prefill time and memory at 16k / 32k with and without down_proj hooks, the duet engine (re-prefill ms per message;
carried-cache parity on 4 runs), RefKL damage time for a 60-unit set, Drive throughput on 100 MB, and the L2a chunked-path parity at
16,384 / 32,768 (OD-M14). JSON with the SO-PILOT whitelisted keys only."""
import os, time
import numpy as np

import r1b_contracts as C
import r1b_core as K
import r1b_l_engine as E
from r1b_l_base import cfg, dims, longdocs, texts_mock


def _prompts(ctx, tok, n=32):
    p = os.path.join(K.REPO, C.PLAN_DIR, "pilot_material.json")
    d = K.load_json(p) if os.path.exists(p) else None
    pr = (d or {}).get("prompts") or [f"Describe pilot scene number {i} in a few sentences." for i in range(n)]
    return [K.chat_ids(tok, x if isinstance(x, str) else x.get("text", "")) for x in pr[:n]]


def _sets(model, n_pop=102, size=60, seed=0):
    L, I = dims(model)
    g = np.random.default_rng(seed)
    return [(f"S{k}", sorted(g.choice(L * I, min(size, L * I // 4), replace=False).tolist())) for k in range(n_pop)]


def _sync():
    t = K._torch()
    if t.cuda.is_available():
        t.cuda.synchronize()


def l0_a(ctx):
    """(i) hooked / unhooked decode ms per step; (ii) R1 / R3 read rates; (v) RefKL damage time for a 60-unit set."""
    torch = K._torch()
    tok, model = ctx.model()
    L, I = dims(model)
    mock = ctx.mock
    B, steps = (32, 32) if not mock else (4, 6)
    prs = _prompts(ctx, tok)
    rw = {k: torch.ones(L * I) if k.startswith("sd") else torch.zeros(L * I) for k in ("mu_z", "sd_z", "mu_cett", "sd_cett")}
    sets = _sets(model, 102 if not mock else 6, 60 if not mock else 8)
    spins = [j for _, s in sets for j in s][: (102 * 60 if not mock else 48)]
    rates, mem = {}, {}
    for ctxlen in ((512, 1100) if not mock else (16, 32)):
        rows = [(p * (ctxlen // max(len(p), 1) + 1))[:ctxlen] for p in prs[:B]]
        keys = [ctx.key("r1bP0_L0", i, ctxlen) for i in range(len(rows))]
        for hooked in (False, True):
            pr = E.Probe(model, rw, chans=sets, spins=spins) if hooked else None
            _sync(); t = time.time()
            E.decode(model, tok, rows, keys, T=1.0, max_new=steps, probe=pr, stop=False,
                     spin_fn=(lambda st, z: z.cpu()) if hooked else None)
            _sync(); rates[f"decode_ms_per_step_ctx{ctxlen}_{'hooked' if hooked else 'plain'}"] = 1000 * (time.time() - t) / (steps + 1)
            if pr:
                pr.remove()
    seqs = [(p * 4)[:130 if not mock else 24] for p in prs[:B]]
    with E.Probe(model, rw, chans=sets) as pr:
        _sync(); t = time.time()
        E.tf_pass(model, tok, seqs, [len(s) // 2 for s in seqs], probe=pr, batch=B)
        _sync(); rates["read_tok_per_s_hooked"] = sum(len(s) for s in seqs) / (time.time() - t)
    refs = K.RefSet.from_prompt_conts(prs[:8], [p[-16:] for p in prs[:8]], "pilot")
    dm = K.Damage(model, refs)
    t = time.time()

    class CM:
        def __enter__(self):
            self.p = E.Probe(model, rw); self.p.add_op(sets[0][1][:60], "scale", val=2.0); return self.p

        def __exit__(self, *a):
            self.p.remove()
    dm(lambda: CM())
    rates["refkl_hooks_s_60units"] = time.time() - t
    if torch.cuda.is_available():
        mem["peak_gb_l0_a"] = torch.cuda.max_memory_allocated() / 2 ** 30
    ctx.write_json("pilot/l0_a.json", dict(rates=rates, memory=mem, notes=["P0-L (i), (ii), (v); stand-in random sets of the planned sizes"]))


def l0_b(ctx):
    """(iii) B = 1 prefill 16k / 32k with and without hooks; (iv) duet re-prefill and carried-cache parity; (vi) Drive throughput;
    (vii) L2a chunked path against a full pass at 16,384 and 32,768."""
    torch = K._torch()
    tok, model = ctx.model()
    L, I = dims(model)
    mock = ctx.mock
    cf = cfg(ctx)
    try:
        docs, _ = longdocs(ctx, tok, cf, which="pilot")
    except K.NotRun:
        docs = texts_mock(2, 64, tok, "pilot")
    rates, mem, par, thr = {}, {}, {}, {}
    rw = {k: torch.ones(L * I) if k.startswith("sd") else torch.zeros(L * I) for k in ("mu_z", "sd_z", "mu_cett", "sd_cett")}
    sets = _sets(model, 102 if not mock else 4, 60 if not mock else 8)
    for Ln in ((16384, 32768) if not mock else (32, 64)):
        d = docs[0][:Ln]
        for hooked in (False, True):
            pr = E.Probe(model, rw, chans=sets) if hooked else None
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            _sync(); t = time.time()
            E.lce_logprobs(model, d, chunk=cf["eqmi_chunk"])
            _sync(); rates[f"prefill_s_{Ln}_{'hooked' if hooked else 'plain'}"] = time.time() - t
            if torch.cuda.is_available():
                mem[f"peak_gb_{Ln}_{'hooked' if hooked else 'plain'}"] = torch.cuda.max_memory_allocated() / 2 ** 30
            if pr:
                pr.remove()
        a = E.lce_logprobs(model, d, chunk=cf["eqmi_chunk"]); b = E.full_logprobs(model, d)
        m = np.isfinite(a) & np.isfinite(b)
        par[f"l2a_rel_{Ln}"] = float(np.max(np.abs(a[m] - b[m])) / max(np.mean(np.abs(b[m])), 1e-12))
    # duet engine: re-prefill per message, and carried-cache parity on 4 runs
    from transformers import DynamicCache
    prs = [E.chat_sys_ids(tok, "You are Agent A, a researcher.", f"Your colleague asks for your view on pilot topic {i}.") for i in range(32 if not mock else 4)]
    _sync(); t = time.time()
    g = E.decode(model, tok, prs, [ctx.key("r1bP0_L0duet", i) for i in range(len(prs))], T=0.7, max_new=40 if not mock else 6)
    _sync(); rates["duet_message_ms_reprefill"] = 1000 * (time.time() - t)
    diffs, flips = [], 0
    dev = K.model_device(model)
    for i in range(4 if len(prs) >= 4 else len(prs)):
        full = prs[i] + g[i]["tokens"] + E.turn_suffix_ids(tok, "Your colleague says: «ok»")
        with torch.no_grad():
            lf = model(input_ids=torch.tensor([full], device=dev)).logits[0, -1].float()
            cache = DynamicCache(); n0 = len(prs[i])
            model(input_ids=torch.tensor([full[:n0]], device=dev), past_key_values=cache, use_cache=True)
            lc = model(input_ids=torch.tensor([full[n0:]], device=dev), past_key_values=cache, use_cache=True,
                       position_ids=torch.arange(n0, len(full), device=dev)[None]).logits[0, -1].float()
        diffs.append(float((lf - lc).abs().max() / lf.abs().max().clamp_min(1e-12)))
        gm = -torch.log(-torch.log(torch.rand(lf.shape[0], generator=torch.Generator(device=dev).manual_seed(i), device=dev).clamp(1e-12, 1 - 1e-7)))
        flips += int((lf / 0.7 + gm).argmax() != (lc / 0.7 + gm).argmax())
    par["duet_carried_cache_max_rel"] = max(diffs) if diffs else None
    par["duet_carried_cache_gumbel_flips"] = flips
    par["duet_carried_cache_pass"] = bool(diffs and max(diffs) < 1e-4)
    # Drive throughput on 100 MB (written into the synced session folder)
    nbytes = 100 * 2 ** 20 if not mock else 2 ** 20
    p = ctx.path("pilot/drive_probe.bin")
    t = time.time(); open(p, "wb").write(os.urandom(nbytes)); thr["write_mb_s"] = nbytes / 2 ** 20 / max(time.time() - t, 1e-9)
    t = time.time(); open(p, "rb").read(); thr["read_mb_s"] = nbytes / 2 ** 20 / max(time.time() - t, 1e-9)
    os.remove(p)
    ctx.write_json("pilot/l0_b.json", dict(rates=rates, memory=mem, parity=par, throughput=thr,
                                           notes=["P0-L (iii), (iv), (vi), (vii); PG19 books 25-28 (timing only)"]))
