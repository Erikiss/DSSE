"""Independent Gemma D1 local/global-attention contrast.

This is the additional descriptive GD contrast, not the complete D1 factorial
experiment or a replacement for its field verdicts. It consumes raw paired
stimuli, uses Gemma's native masks, and never requires a Qwen run or token IDs.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


def _hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _finish(ctx, result):
    result.update(model=ctx.model_name, mock=bool(ctx.mock),
                  interpretation="descriptive; native layer type is confounded with depth",
                  scope="GD native local/global contrast; original D1 field verdicts are not computed")
    ctx.write_json("d1.json", result)
    return result


def validate_items(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("d1 must be a nonempty list of paired raw-text stimuli")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("D1 items need a nonempty string id")
        if row["id"] in seen:
            raise ValueError("D1 item ids must be unique")
        seen.add(row["id"])
        for name in ("context", "control_context", "question"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f"D1 {row['id']} needs {name}")
        if row["context"] == row["control_context"]:
            raise ValueError("D1 cue and control contexts must differ")
        if (not isinstance(row.get("choices"), list) or len(row["choices"]) != 2
                or any(not isinstance(v, str) or not v.strip() for v in row["choices"])
                or row["choices"][0] == row["choices"][1]):
            raise ValueError("D1 needs two distinct textual choices")
        if type(row.get("gold")) is not int or row["gold"] not in (0, 1):
            raise ValueError("D1 gold must be 0 or 1")
    return rows


def make_inputs(packet):
    """Retain D1's raw reports/questions and both letter assignments.

    The supplied package is model-free text. Its Qwen token-length validation is
    not reused: Gemma boundaries are recomputed by ``prepare_ids``.
    """
    import r1b_d_stim as stimuli
    rows = []
    for family, field, first, second in (("sym", "fsym", "A1", "A2"), ("asym", "fasym", "T", "F")):
        for i, item in enumerate(packet.get(field, [])):
            context = stimuli.parts_text(stimuli.block_parts(packet, family, i, [first] * 10))
            control = stimuli.parts_text(stimuli.block_parts(packet, family, i, [second] * 10))
            questions = packet["questions" if family == "sym" else "asym_questions"]
            for j, slot in enumerate(item["slots"]):
                a, b = (slot["v1"], slot["v2"]) if family == "sym" else (slot["true"], slot["cp"])
                question = "According to the notes, " + questions[j].format(R=item["referent"]) + "?"
                for swap in (0, 1):
                    rows.append({"id": f"{family}-{i}-q{j}-swap{swap}", "cluster_id": f"{family}-{i}",
                                 "family": family, "context": context, "control_context": control,
                                 "question": question, "choices": [a, b] if swap == 0 else [b, a],
                                 "gold": swap, "source": "D1 frozen raw package; pure endpoint reports"})
    return validate_items(rows)


def prepare_ids(tok, row, context, gap, filler):
    """Exact token gap from end of cue to the scored final prompt token.

    Source and read boundaries are recorded on the actual submitted token IDs.
    No character-length estimate is used. Too-short requested gaps are omitted
    explicitly, rather than silently changing a window-relative condition.
    """
    from r1b_g_adapter import chat_text
    marker = "R1B_G_D1_FILL_3bd7c892"
    if marker in context or marker in row["question"]:
        raise ValueError("reserved D1 marker appears in source text")
    a, b = row["choices"]
    user = f"{context}\n\n{marker}\n\nQuestion: {row['question']} (a) {a} (b) {b}\nReply with the letter."
    rendered = chat_text(tok, user) + "Answer: ("
    prefix, suffix = rendered.split(marker)
    pre = tok(prefix, add_special_tokens=False)["input_ids"]
    post = tok(suffix, add_special_tokens=False)["input_ids"]
    nfill = int(gap) - len(post)
    if nfill < 0:
        return None, {"requested_gap": int(gap), "minimum_gap": len(post),
                      "reason": "question/template longer than requested cue-read gap"}
    fill = tok(filler, add_special_tokens=False)["input_ids"]
    if not fill:
        raise ValueError("D1 filler must produce at least one token")
    ids = pre + (fill * (nfill // len(fill) + 1))[:nfill] + post
    return ids, {"requested_gap": int(gap), "actual_gap": len(ids) - len(pre),
                 "cue_end": len(pre) - 1, "read_position": len(ids) - 1,
                 "n_tokens": len(ids), "input_sha256": _hash(ids)}


def _read(model, ids, letter_ids, layers):
    import torch
    import r1b_core as core
    from r1b_g_adapter import unembed
    dev = next(model.parameters()).device
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    with torch.no_grad(), core.Hooks(model) as hooks:
        hooks.capture(layers, positions=[-1], cpu=False)
        output = model(input_ids=x, use_cache=False, logits_to_keep=1)
        lg = output.logits[0, -1].float()
        lp = lg.log_softmax(-1)
        layer_reads = []
        for layer in layers:
            logits = unembed(model, hooks.states[layer][0])[0, -1].float()
            layer_reads.append(float(logits[letter_ids[1]] - logits[letter_ids[0]]))
        result = {"logodds_b_minus_a": float(lg[letter_ids[1]] - lg[letter_ids[0]]),
                  "choice_mass": float(lp[letter_ids].exp().sum()), "layer_logodds": layer_reads}
    return result, lg


def _cached_read(model, ids, chunk):
    import torch
    dev = next(model.parameters()).device
    cache = None
    with torch.no_grad():
        for start in range(0, len(ids), chunk):
            stop = min(start + chunk, len(ids))
            output = model(input_ids=torch.tensor([ids[start:stop]], device=dev),
                           attention_mask=torch.ones((1, stop), dtype=torch.long, device=dev),
                           position_ids=torch.arange(start, stop, device=dev)[None],
                           past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
        result = output.logits[0, -1].float()
    return result


def _interval(values, seed, nboot):
    import r1b_core as core
    vals = np.asarray(values, dtype=float)
    if len(vals) < 2:
        return {"code": "NA:too_few_independent_items"}
    rng = np.random.default_rng(seed)
    draws = vals[rng.integers(0, len(vals), (nboot, len(vals)))].mean(1)
    return list(core.interval(draws))


def summarize(records, layer_types, *, seed=7291, nboot=2000):
    """Aggregate within item first; never treat layers or distances as subjects."""
    local = [i for i, kind in enumerate(layer_types) if kind == "sliding_attention"]
    glob = [i for i, kind in enumerate(layer_types) if kind == "full_attention"]
    if not local or not glob:
        return {"status": "blocked", "reason": "both native attention types are required"}
    by = {}
    for rec in records:
        delta = (2 * rec.get("gold", 1) - 1) * (np.asarray(rec["cue"]["layer_logodds"]) - np.asarray(rec["control"]["layer_logodds"]))
        by.setdefault(rec.get("cluster_id", rec["id"]), []).append(float(delta[glob].mean() - delta[local].mean()))
    per_item = {k: float(np.mean(v)) for k, v in by.items()}
    values = list(per_item.values())
    return {"status": "completed" if values else "blocked", "n_items": len(values),
            "mean_global_minus_local_cue_effect": float(np.mean(values)) if values else {"code": "NA:no_conditions"},
            "interval90": _interval(values, seed, nboot), "per_item": per_item,
            "bootstrap_unit": "item (all distances and layers remain together)",
            "bootstrap_seed": seed, "bootstrap_replicates": nboot,
            "hypothesis_verdict": "descriptive; no causal layer-type verdict"}


def _condition(model, tok, row, gap, filler, layout, letter_ids, layers):
    arms, metas, parity, skipped = {}, {}, {}, []
    for arm, field in (("cue", "context"), ("control", "control_context")):
        ids, meta = prepare_ids(tok, row, row[field], gap, filler)
        if ids is None:
            skipped.append({"id": row["id"], "arm": arm, **meta})
            break
        if len(ids) > layout["max_position_embeddings"]:
            skipped.append({"id": row["id"], "arm": arm, "reason": "maximum context exceeded", **meta})
            break
        result, full_logits = _read(model, ids, letter_ids, layers)
        cached = _cached_read(model, ids, max(8, layout["sliding_window"] // 2))
        diff = (full_logits - cached).abs()
        parity[arm] = {"max": float(diff.max()), "median": float(diff.median())}
        parity[arm]["passed"] = parity[arm]["max"] <= 0.02 and parity[arm]["median"] <= 0.002
        arms[arm], metas[arm] = result, meta
        del full_logits, cached, diff
    record = None
    if len(arms) == 2:
        record = {"id": row["id"], "cluster_id": row.get("cluster_id", row["id"]),
                  "gold": row["gold"], "gap": gap, "gap_over_window": gap / layout["sliding_window"],
                  **arms, "boundaries": metas, "cache_parity": parity}
    return {"record": record, "skipped": skipped}


def run(ctx):
    from r1b_g_adapter import attention_layout, layers_of
    data = ctx.data()
    try:
        rows = validate_items(data.get("d1"))
    except ValueError as exc:
        return _finish(ctx, {"status": "blocked", "reason": str(exc), "records": []})
    tok, model = ctx.model()
    layout = attention_layout(model)
    window = layout["sliding_window"]
    letters = [tok(x, add_special_tokens=False)["input_ids"] for x in ("a", "b")]
    if any(len(x) != 1 for x in letters) or letters[0] == letters[1]:
        return _finish(ctx, {"status": "blocked", "reason": "Gemma letter readout gate failed", "records": []})
    letter_ids = [v[0] for v in letters]
    filler = data.get("d1_filler", "A neutral passage describes a quiet landscape and ordinary daily events. ")
    if not isinstance(filler, str) or not filler.strip():
        raise ValueError("d1_filler must be nonempty text")
    gaps = sorted(set(int(round(window * f)) for f in (0.5, 0.9, 1.1, 2.0)))
    if ctx.mock:
        # Tiny fixtures have 8-token windows, shorter than the real chat header.
        # Add a clearly marked executable condition; do not relabel it as 2W.
        gaps.append(max(gaps[-1], 160))
    layers = list(range(len(layers_of(model))))
    records, skipped = [], []
    for row in rows:
        for gap in gaps:
            if hasattr(ctx, "time_left") and ctx.time_left() < 30:
                from r1b_core import StepYield
                raise StepYield()
            key = "d1_" + _hash({"row": row, "gap": gap, "filler": filler, "layout": layout})
            compute = lambda: _condition(model, tok, row, gap, filler, layout, letter_ids, layers)
            item = ctx.cached(key, compute) if hasattr(ctx, "cached") else compute()
            skipped.extend(item["skipped"])
            if item["record"] is not None:
                item["record"]["mock_extra_gap"] = bool(ctx.mock and gap == 160)
                records.append(item["record"])
    passed = bool(records) and all(v["passed"] for r in records for v in r["cache_parity"].values())
    result = {"status": "completed" if passed else "blocked", "records": records, "skipped": skipped,
              "inputs_sha256": _hash(rows), "filler_sha256": _hash(filler), "attention_layout": layout,
              "gates": {"G-CACHE": {"passed": passed, "max_tolerance": 0.02, "median_tolerance": 0.002}},
              "summary": summarize(records, layout["layer_types"], nboot=32 if ctx.mock else 2000),
              "by_distance": [{"gap": gap, "gap_over_window": gap / window,
                               **summarize([r for r in records if r["gap"] == gap], layout["layer_types"],
                                           nboot=32 if ctx.mock else 2000)} for gap in sorted({r["gap"] for r in records})]}
    if not passed:
        result["reason"] = "no executable conditions" if not records else "full/cache parity gate failed"
    return _finish(ctx, result)
