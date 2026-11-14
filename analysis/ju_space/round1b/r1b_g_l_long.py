"""Gemma-native L2a context-information protocol and estimator controls.

Scoring retains native sliding/global attention and KV sharing. Logits are
projected in bounded chunks with Gemma's final softcap. Every likelihood at j
is produced by the distribution at j-1, including chunk boundaries.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import r1b_g_adapter as A
import r1b_core as K
import r1b_l_stats as S
from r1b_l_base import clean, rng


def logprobs(model, ids, *, chunk=2048, logit_chunk=32, cached=True):
    """Return log q(token_j | token_<j), with an undefined first entry."""
    import torch
    if not ids or chunk < 1 or logit_chunk < 1:
        raise ValueError("nonempty IDs and positive chunk sizes required")
    device = next(model.parameters()).device
    n = len(ids)
    if n > int(A.text_config(model).max_position_embeddings):
        raise ValueError("requested L2a sequence exceeds native context limit")
    values = np.full(n, np.nan, dtype=np.float64)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    cache = None
    step = chunk if cached else n
    with K.sdpa_efficient(device), torch.no_grad():
        for start in range(0, n, step):
            end = min(start + step, n)
            output = A.text_model(model)(input_ids=x[:, start:end],
                                        attention_mask=torch.ones((1, end), dtype=torch.long, device=device),
                                        position_ids=torch.arange(start, end, device=device)[None],
                                        past_key_values=cache, use_cache=cached)
            if cached:
                cache = output.past_key_values
            hidden = output.last_hidden_state[0]
            # The final state of a chunk predicts the first token of the next.
            for a in range(0, min(end, n - 1) - start, logit_chunk):
                b = min(a + logit_chunk, min(end, n - 1) - start)
                logits = A.unembed(model, hidden[a:b], normalized=True).float()
                target = x[0, start + a + 1:start + b + 1]
                values[start + a + 1:start + b + 1] = logits.log_softmax(-1).gather(1, target[:, None])[:, 0].double().cpu().numpy()
    return values


def own_continuation(engine, seed, key, count, checkpoint=None, chunk_size=128):
    """EOS-masked own text with native KV reuse and resumable token chunks.

    Checkpoints contain native token IDs, unmasked p_EOS and the keyed RNG state.
    A continuation reconstructs the prefix through the parity-checked native
    cache path; it does not resample completed tokens.
    """
    import torch
    if len(seed) + count > int(engine.cfg.max_position_embeddings):
        raise ValueError("own-text generation exceeds native context limit")
    eos, stop = A.stop_tokens(engine.tok, engine.model)
    generator = torch.Generator(device=engine.device).manual_seed(int(key))
    tokens, peos = [], []
    runtime = {"cache": None}
    state = generator.get_state()
    for offset in range(0, count, chunk_size):
        executed = []
        def piece(offset=offset, state=state):
            executed.append(True)
            generator.set_state(state)
            seq = list(seed) + tokens
            new_tokens, new_peos = [], []
            cache = runtime["cache"]
            with K.sdpa_efficient(engine.device), torch.no_grad():
                for step in range(min(chunk_size, count - offset)):
                    incoming = seq if cache is None else [seq[-1]]
                    start = len(seq) - len(incoming)
                    output = A.text_model(engine.model)(
                        input_ids=torch.tensor([incoming], device=engine.device),
                        attention_mask=torch.ones((1, len(seq)), dtype=torch.long, device=engine.device),
                        position_ids=torch.arange(start, len(seq), device=engine.device)[None],
                        past_key_values=cache, use_cache=True)
                    cache = output.past_key_values
                    logits = A.unembed(engine.model, output.last_hidden_state[0, -1], normalized=True).float()
                    new_peos.append(float(logits.softmax(-1)[eos].sum()))
                    logits[list(stop)] = -torch.inf
                    token = engine.choose(logits, generator, temperature=1.0)
                    new_tokens.append(token)
                    seq.append(token)
            runtime["cache"] = cache
            return {"tokens": new_tokens, "peos": new_peos, "rng_state": generator.get_state()}
        record = checkpoint(offset, piece) if checkpoint is not None else piece()
        if not executed:
            runtime["cache"] = None
        tokens.extend(record["tokens"])
        peos.extend(record["peos"])
        state = record["rng_state"]
    return {"tokens": tokens, "peos": peos}


def _ihat(model, true_lp, doc, foreign, length, chunk):
    half = length // 2
    changed = list(foreign[:half]) + list(doc[half:length])
    lp = logprobs(model, changed, chunk=chunk)
    return float(np.sum(true_lp[half:length] - lp[half:length]))


def _summarize(main, own, parity, nboot):
    """Frozen L2a document bootstrap and decision inputs, in nats."""
    grid, info = np.asarray(main["grid"], dtype=float), main["I"]
    W = S.boot_counts(len(info), nboot, seed=211)
    boots = (W @ info) / W.sum(1, keepdims=True)
    classes = [S.growth_class(grid, row)[0] for row in boots]
    c0, fits = S.growth_class(grid, info.mean(0))
    shares = {k: classes.count(k) / len(classes) for k in ("saturating", "logarithmic", "power")}
    top = max(shares, key=shares.get)
    x = np.log(grid)
    slopes = [float(np.cov(x, row, bias=True)[0, 1] / x.var()) for row in boots]
    betas = [S.growth_fits(grid, row)["power"]["beta"] for row in boots]
    Lsat = None
    for i in range(len(grid) - 1):
        if all(info.mean(0)[j + 1] - info.mean(0)[j] < max(2 * boots.std(0)[j], .05 * info.mean(0)[-1]) for j in range(i, len(grid) - 1)):
            Lsat = int(grid[i])
            break
    copy_ok = bool(np.all(main["copy"].mean(0) >= 5 * info.mean(0)[:main["copy"].shape[1]]))
    diff = main["shuf"] - main["shuf_base"]
    Wd = S.boot_counts(len(diff), nboot, seed=212)
    shuf = S.summ(float(diff.mean()), Wd @ diff / Wd.sum(1))
    gmi_ok = copy_ok and S.hi(shuf) < 0 and float(main["shuf"].mean()) < float(main["shuf_base"].mean())
    Ls = np.asarray(own["L"])
    tend, mx = np.asarray(own["t_end"]), np.asarray(own["max_peos"])
    Wo = S.boot_counts(len(tend), nboot, seed=213)
    def ratio(mask):
        pt, draws = [], []
        for j, length in enumerate(Ls):
            selected = mask & (tend >= length)
            if not selected.any():
                pt.append(np.nan)
                draws.append(np.full(len(Wo), np.nan))
                continue
            num, den = own["I_self"][selected, j], own["I_nat"][selected, j]
            pt.append(float(num.mean() / den.mean()) if den.mean() else np.nan)
            denom = Wo[:, selected] @ den
            draws.append((Wo[:, selected] @ num) / np.where(denom != 0, denom, np.nan))
        return np.asarray(pt), np.stack(draws, axis=1)
    point, draws = ratio(np.ones(len(tend), dtype=bool))
    selected = Ls >= Ls[min(1, len(Ls) - 1)]
    xl = np.log(Ls[selected])
    def slope(v):
        yy = np.log(np.where(v[selected] > 0, v[selected], np.nan))
        ok = np.isfinite(yy)
        return float(np.cov(xl[ok], yy[ok], bias=True)[0, 1] / xl[ok].var()) if ok.sum() > 1 else np.nan
    low, lowdraw = ratio(mx < .05)
    lowstat = S.summ(low[-1], lowdraw[:, -1]) if np.isfinite(low[-1]) else None
    return {"grid": grid.tolist(), "I": [S.summ(info.mean(0)[j], boots[:, j]) for j in range(len(grid))],
            "growth": {"class_point": c0, "class": top, "share": shares[top], "shares": shares,
                       "b": S.summ(fits["logarithmic"]["b"], slopes), "beta": S.summ(fits["power"]["beta"], betas), "fits": fits},
            "L_sat": Lsat, "gmi_ok": bool(gmi_ok and parity["ok"]), "copy_ok": copy_ok, "shuffle": shuf,
            "lce": parity["ok"], "R": [S.summ(point[j], draws[:, j]) for j in range(len(Ls))],
            "R_top": S.summ(point[-1], draws[:, -1]),
            "lnR_top": S.summ(math.log(point[-1]) if point[-1] > 0 else np.nan,
                              np.log(np.where(draws[:, -1] > 0, draws[:, -1], np.nan))),
            "lnR_slope": S.summ(slope(point), [slope(row) for row in draws]),
            "R_top_low_eos": lowstat, "deficit_low_eos": lowstat is not None and S.hi(lowstat) < .87,
            "n_truncated": int((tend < Ls[-1]).sum()), "bootstrap_unit": "document"}


def long_context(ctx):
    from r1b_g_engine import ModelEngine
    from r1b_g_l import config
    cf = config(ctx)
    source = ctx.data().get("longdocs", [])
    if len(source) < cf["eqmi_docs"]:
        return {"status": "blocked", "reason": f"longdocs needs {cf['eqmi_docs']} independent raw documents; received {len(source)}"}
    E = ModelEngine(ctx)
    docs, ids = [], []
    for i, row in enumerate(source):
        text = row if isinstance(row, str) else row.get("text", "")
        seq = E.encode(text, chat=False)
        if len(seq) - cf["long_skip"] >= cf["docs_min"]:
            docs.append(seq[cf["long_skip"]:cf["long_skip"] + cf["long_len"]])
            ids.append(str(i) if isinstance(row, str) else str(row.get("id", i)))
        if len(docs) == cf["eqmi_docs"]:
            break
    if len(docs) < cf["eqmi_docs"]:
        return {"status": "blocked", "reason": f"need {cf['eqmi_docs']} documents with >= {cf['docs_min']} target tokens; received {len(docs)}"}
    if len(set(ids)) != len(ids):
        raise ValueError("longdocs must have unique IDs")
    grid = list(cf["eqmi_L"])
    reduced_grid = (getattr(ctx, "freeze", {}).get("decisions") or {}).get("OD-M14") == "grid_2^14"
    if reduced_grid:
        grid = [n for n in grid if n <= 2 ** 14]
    if max(grid) > int(E.cfg.max_position_embeddings):
        return {"status": "blocked", "reason": "frozen L2a grid exceeds native model context limit"}
    parity = {}
    for length in ((cf["long_len"] // 2, cf["long_len"]) if ctx.mock else (16384, 32768)):
        rel = []
        for i, doc in enumerate(docs[:2]):
            def compare():
                a = logprobs(E.model, doc[:length], chunk=cf["eqmi_chunk"])
                b = logprobs(E.model, doc[:length], cached=False)
                valid = np.isfinite(a) & np.isfinite(b)
                return float(np.max(np.abs(a[valid] - b[valid])) / max(float(np.mean(np.abs(b[valid]))), 1e-12))
            rel.append(ctx.cached(f"l2a_parity_{length}_{i}", compare))
        parity[str(length)] = {"relative_error": max(rel), "ok": max(rel) < 1e-4}
    required_lengths = [n for n in parity if not reduced_grid or int(n) <= 16384]
    parity = {"lengths": parity, "ok": all(parity[n]["ok"] for n in required_lengths), "threshold": 1e-4,
              "required_lengths": required_lengths, "frozen_OD_M14": "grid_2^14" if reduced_grid else None,
              "kernel": "cpu: not asserted" if ctx.mock else "memory-efficient SDPA asserted"}
    ctx.write_json("l2a_lce.json", parity)
    if not parity["ok"]:
        return {"status": "blocked", "reason": "G-L2a-LCE parity failed", "parity": parity}
    perm = S.derange(len(docs), rng("r1bL2a", "derange"))
    measured = []
    for i, doc in enumerate(docs):
        def run(i=i, doc=doc):
            lp = logprobs(E.model, doc[:max(grid)], chunk=cf["eqmi_chunk"])
            return [_ihat(E.model, lp, doc, docs[perm[i]], n, cf["eqmi_chunk"]) for n in grid]
        measured.append(ctx.cached(f"l2a_doc_{i}", run))
    info = np.asarray(measured)
    copy_grid = [n for n in grid if n <= (grid[-2] if ctx.mock else 4096)]
    copies = []
    for i in range(min(cf["gmi_copy"], len(docs))):
        def run_copy(i=i):
            out = []
            for n in copy_grid:
                doc = docs[i][:n // 2] * 2
                lp = logprobs(E.model, doc, chunk=cf["eqmi_chunk"])
                out.append(_ihat(E.model, lp, doc, docs[perm[i]], n, cf["eqmi_chunk"]))
            return out
        copies.append(ctx.cached(f"l2a_copy_{i}", run_copy))
    length = [n for n in grid if n <= (grid[-1] if ctx.mock else 8192)][-1]
    shuffled = []
    for i, doc in enumerate(docs[:cf["gmi_shuf"]]):
        sentences = re.split(r"(?<=[.!?])\s+", E.tok.decode(doc))
        rng("r1bL-SHUF", i).shuffle(sentences)
        shuffled.append(E.encode(" ".join(sentences), chat=False)[:len(doc)])
    if any(len(seq) < length for seq in shuffled):
        return {"status": "blocked", "reason": "sentence-shuffled tokenization shortened a G-MI control below its fixed grid length"}
    shuf = []
    for i, doc in enumerate(shuffled):
        def run_shuf(i=i, doc=doc):
            lp = logprobs(E.model, doc, chunk=cf["eqmi_chunk"])
            return _ihat(E.model, lp, doc, shuffled[(i + 1) % len(shuffled)], length, cf["eqmi_chunk"])
        shuf.append(ctx.cached(f"l2a_shuffle_{i}", run_shuf))
    from r1b_g_engine import content_hash
    main = {"grid": grid, "I": info, "copy": np.asarray(copies), "copy_grid": copy_grid,
            "shuf": np.asarray(shuf), "shuf_base": info[:len(shuf), grid.index(length)],
            "shuf_L": length, "perm": perm.tolist(), "document_ids": ids,
            "slices": [{"id": qid, "start": cf["long_skip"], "end": cf["long_skip"] + len(doc),
                        "target_tokens_sha256": content_hash(doc)} for qid, doc in zip(ids, docs)]}
    ctx.save_pt("l2a_main.pt", main)
    seeds = [doc[:cf["own_seed"]] for doc in docs]
    generated = [own_continuation(E, seeds[i], ctx.key("r1bL2s", i), cf["own_new"],
                 checkpoint=lambda offset, fn, i=i: ctx.cached(f"l2a_own_{i}_part_{offset}", fn)) for i in range(len(docs))]
    own_perm = S.derange(len(docs), rng("r1bL2a", "own"))
    self_info, nat_info = [], []
    sd = cf["own_seed"]
    for i, doc in enumerate(docs):
        def run_own(i=i, doc=doc):
            ownseq = seeds[i] + generated[i]["tokens"]
            natseq = doc[:sd + cf["own_new"]]
            lp_self = logprobs(E.model, ownseq, chunk=cf["eqmi_chunk"])
            lp_nat = logprobs(E.model, natseq, chunk=cf["eqmi_chunk"])
            a, b = [], []
            for n in cf["own_L"]:
                fs = seeds[i] + generated[own_perm[i]]["tokens"][:n // 2] + generated[i]["tokens"][n // 2:n]
                fn = seeds[i] + docs[own_perm[i]][sd:sd + n // 2] + doc[sd + n // 2:sd + n]
                sl = slice(sd + n // 2, sd + n)
                a.append(float(np.sum(lp_self[sl] - logprobs(E.model, fs, chunk=cf["eqmi_chunk"])[sl])))
                b.append(float(np.sum(lp_nat[sl] - logprobs(E.model, fn, chunk=cf["eqmi_chunk"])[sl])))
            return a, b
        a, b = ctx.cached(f"l2a_own_info_{i}", run_own)
        self_info.append(a)
        nat_info.append(b)
    ends = []
    for rec in generated:
        crossing = np.flatnonzero(1 - np.cumprod(1 - np.asarray(rec["peos"])) > .5)
        ends.append(int(crossing[0]) if len(crossing) else len(rec["tokens"]))
    own = {"L": cf["own_L"], "I_self": np.asarray(self_info), "I_nat": np.asarray(nat_info), "t_end": ends,
           "max_peos": [float(np.max(rec["peos"])) for rec in generated], "document_ids": ids}
    ctx.save_pt("l2a_own.pt", own)
    stats = _summarize(main, own, parity, cf["nboot"])
    ctx.write_json("l2a.json", clean(stats))
    return {"status": "measured", "gmi_ok": stats["gmi_ok"], "lce_ok": True, "documents": len(docs),
            "grid": grid, "unit": "nats", "own_text": True}
