"""Gemma L5a-c: exact-label ownership and native own-error measurements.

Only target-native outputs build the behavioural value set. Missing target lens,
H populations or scientific sample sizes close their respective verdict gates;
they do not prevent the independent own-CoT collection. No Qwen result is read.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

import r1b_core as K
import r1b_l_rules as RU
import r1b_l_stats as S
from r1b_l_ana import nums, parse_cot
from r1b_l_base import COT_INSTR, SUB_TEMPLATE, clean, rng


def _json(ctx, stage, name):
    path = Path(ctx.artifact(stage, name))
    return json.loads(path.read_text()) if path.is_file() else {}


def _pt(ctx, stage, name):
    import torch
    path = Path(ctx.artifact(stage, name))
    return torch.load(path, map_location="cpu", weights_only=False) if path.is_file() else {}


def _passed(value):
    return value is True or (isinstance(value, dict) and value.get("pass") is True)


def _cache(ctx, name, identity, function):
    if ctx.time_left() < 30:
        raise K.StepYield("L5 item checkpoint saved; continue the same G run")
    suffix = hashlib.sha256(str(identity).encode()).hexdigest()[:20]
    return ctx.cached(f"l5_{name}_{suffix}", function)


def populations(pop):
    """Ordered native ragged coordinates, including all comparison channels."""
    groups = {}
    for name in ("P", "SEL", "UNC"):
        if pop.get(name):
            groups[name] = list(pop[name])
    groups.update({f"C{i}": list(v) for i, v in enumerate(pop.get("CTRL", [])) if v})
    if pop.get("H_tq") and pop.get("H_tq") != pop.get("P"):
        groups["Htq"] = list(pop["H_tq"])
    groups.update({f"B{i}": list(v) for i, v in enumerate(pop.get("boot", [])) if v})
    return groups


def standardized_channels(row, groups, refs):
    """L's mean per-neuron REF-W standardized CETT, not raw population CETT."""
    values = np.asarray(row["cett"], dtype=np.float64)
    columns = {int(neuron): i for i, neuron in enumerate(row["neuron_ids"])}
    if not groups:
        return np.empty((values.shape[0], 0), dtype=np.float32)
    mu = np.asarray(refs["mu_cett"], dtype=np.float64)
    sd = np.asarray(refs["sd_cett"], dtype=np.float64)
    result = []
    for ids in groups.values():
        selected = np.asarray(ids, dtype=int)
        index = [columns[int(i)] for i in selected]
        result.append(((values[:, index] - mu[selected]) / np.maximum(sd[selected], 1e-9)).mean(1))
    return np.asarray(result, dtype=np.float32).T


def _decode(tok, ids):
    return tok.decode(list(map(int, ids)), skip_special_tokens=True, clean_up_tokenization_spaces=False)


def token_offsets(tok, ids):
    """Offsets against the actual decoded continuation; reject unstable prefixes.

    A byte fragment may change when the next token completes a code point. Such
    an alignment is recorded as unavailable instead of moving a pre-step read
    into the step it is meant to predict.
    """
    text = _decode(tok, ids)
    prefixes = [_decode(tok, ids[:i]) for i in range(len(ids))]
    valid = all(text.startswith(prefix) for prefix in prefixes)
    return text, [len(prefix) for prefix in prefixes], valid


def build_gate(engine, raw, measured):
    """Join genuine gating answers with raw exact labels and subexpressions."""
    from r1b_g_k_chain import normalize_item
    result = {}
    for source in raw:
        item = normalize_item(source)
        if item["id"] not in measured or item.get("cal") or item["split"] != "A-lens":
            continue
        g = measured[item["id"]]
        prev, steps, sub = item["P"][0], [], []
        ops = item.get("ops") or []
        if len(ops) != len(item["P"]) - 1 or len(item["I"]) + 1 != len(ops):
            raise ValueError("L5 requires the frozen left-to-right operands, operations and intermediate labels")
        for op, operand, value in zip(ops, item["P"][1:], item["I"] + [item["A"]]):
            steps.append([prev, op, operand, value])
            sub.append(f"{prev} {op} {operand}")
            prev = value
        answers = [g.get("greedy_text", _decode(engine.tok, g.get("greedy", [])))]
        answers.extend(_decode(engine.tok, ids) for ids in g.get("samples", []))
        result[item["id"]] = {
            "id": item["id"], "stratum": "known" if g.get("answer") == item["A"] else "prone",
            "answers": answers, "prompt": item["prompt"], "V": item["P"] + item["I"] + [item["A"]],
            "P": item["P"], "steps": steps, "subexprs": sub, "I_plus": item.get("Iplus", []),
            "ops": len(ops), "a_cal": False, "family": item.get("template", len(ops)),
        }
    return result


def lens_records(packet, tok, gate):
    """Adapt only measured V1/V2 SO-LENSOUT rows, with genuine DLA shares."""
    records = []
    for row in packet.get("records", packet.get("rows", [])):
        if row.get("variant") not in ("V1", "V2") or row.get("item") not in gate:
            continue
        tokens = [int(row["seed"])] + list(map(int, row["tok"]))
        share = row.get("share_H")
        if share is None:
            share = (row.get("share") or {}).get("H")
        share = None if share is None else np.asarray(share, float).tolist()
        conf = next((int(q) for q, _, label in row.get("claims", []) if label == "CONFAB"), None)
        records.append({"item": row["item"], "variant": row["variant"], "block": int(row["block"]),
                        "ptype": row.get("ptype", "t0"), "seed_rank": int(row["seed_rank"]),
                        "tokens": tokens, "tokens_text": [tok.decode([t]) for t in tokens],
                        "k_star_conf": conf, "censor": row.get("censor"), "share_H": share})
    return records


def generate_item(ctx, engine, item, groups, refs, cf):
    """Six extra gates, five own CoTs, five responses per isolated subexpression."""
    ident = item["id"]
    original = engine.encode(item["prompt"])
    prompt = re.sub(r"Reply with (?:the number only|only the final number)\.?", "", item["prompt"], flags=re.I).strip()
    cot_prompt = engine.encode(prompt + " " + COT_INSTR)
    extra_keys = [ctx.key("r1bL5", "GATE", ident, s) for s in range(1, 7)]
    extra = _cache(ctx, "gates", ident, lambda: engine.decode(
        [original] * 6, keys=extra_keys, max_new=cf["l5_sub_new"], temperature=.7))
    result = {"gate": [x["text"] for x in extra], "gate_keys": extra_keys, "cot": [], "sub": []}
    for sample in range(5):
        key = ctx.key("r1bL5", "COT", ident, sample)

        def cot(sample=sample, key=key):
            output = engine.decode([cot_prompt], keys=[key], temperature=0 if sample == 0 else .7,
                                   max_new=cf["l5_cot_new"])[0]
            ids = output["ids"]
            text, offsets, aligned = token_offsets(engine.tok, ids)
            # One final dummy token makes the final analysed state's entropy
            # available. Native causal attention cannot reveal it to that state.
            seq = cot_prompt + ids
            captured = engine.teacher_force([seq + [seq[-1]]], populations=groups)["rows"][0]
            positions = np.arange(len(cot_prompt) - 1, len(seq))
            activity = standardized_channels(captured, groups, refs)[positions]
            entropy = np.asarray(captured["entropy"], np.float32)[positions]
            surprise = -np.asarray(captured["next_logprobs"], np.float32)[positions]
            return {"tokens": ids, "text": text, "a": activity, "e": entropy, "s": surprise,
                    "offsets": [-1] + offsets, "alignment_ok": aligned,
                    "token_ends": [0] + offsets[1:] + ([len(text)] if ids else []),
                    "absolute_positions": positions.tolist(), "sample": sample, "key": key,
                    "prompt_ids": cot_prompt, "baseline": "last prompt token before first CoT line"}

        result["cot"].append(_cache(ctx, "cot", (ident, sample), cot))
    for step, expression in enumerate(item["subexprs"]):
        sub_prompt = engine.encode(SUB_TEMPLATE.format(e=expression))
        for sample in range(5):
            key = ctx.key("r1bL5", "SUB", ident, step, sample)
            output = _cache(ctx, "sub", (ident, step, sample), lambda key=key, sample=sample, prompt=sub_prompt:
                            engine.decode([prompt], keys=[key], temperature=0 if sample == 0 else .7,
                                          max_new=cf["l5_sub_new"])[0])
            result["sub"].append({"k": step, "s": sample, "text": output["text"], "tokens": output["ids"], "key": key})
    if groups:
        row = engine.teacher_force([original + [original[-1]]], populations=groups)["rows"][0]
        result["tstar"] = standardized_channels(row, groups, refs)[len(original) - 1]
    else:
        result["tstar"] = None
    return result


def _holm(draws, specifications):
    p = []
    for name, relation, threshold in specifications:
        d = draws[name]
        p.append(max(K.p_one_sided(d, -threshold), K.p_one_sided(d, threshold, "less")) if relation == "inside"
                 else K.p_one_sided(d, threshold, relation))
    adjusted, rejected = K.holm(p)
    return {"p": p, "adjusted": adjusted, "pass": all(rejected)}


def ownership(gate, records, generated, nboot):
    """Frozen L5a equations with item bootstrap and completed native digit runs."""
    from r1b_k_labels import numerals
    own = {}
    for ident, item in gate.items():
        g = generated.get(ident, {})
        texts = item["answers"] + g.get("gate", []) + [x["text"] for x in g.get("cot", []) + g.get("sub", [])]
        own[ident] = {v for text in texts for v in nums(text)}
    identifiers = sorted({r["item"] for r in records if r["item"] in generated})
    if not identifiers:
        return None, []
    strata, foreign = {}, {}
    for ident in identifiers:
        item = gate[ident]
        strata.setdefault((item["stratum"], item["ops"]), []).append(ident)
    for key, ids in strata.items():
        if len(ids) > 1:
            foreign.update({a: ids[int(b)] for a, b in zip(ids, S.derange(len(ids), rng("r1bL5-FM", *map(str, key))))})
    rows = []
    for record in records:
        ident = record["item"]
        if ident not in generated:
            continue
        truth = set(gate[ident]["V"])
        model_values = own[ident] - truth
        foreign_values = own.get(foreign.get(ident), set()) - truth
        row = {**record, "stratum": gate[ident]["stratum"], "BF": None, "BM": None, "BMf": None,
               "foreign_item": foreign.get(ident), "q0F": False, "cls_kconf": None, "classes": []}
        for value, position, completed in numerals(record["tokens_text"]):
            if not completed or (record.get("censor") is not None and position >= record["censor"]):
                continue
            value = int(value)
            label = "TRUE" if value in truth else "MODEL" if value in model_values else "FOREIGN"
            row["classes"].append([int(position), value, label])
            if position == record.get("k_star_conf"):
                row["cls_kconf"] = label
            if position == 0:
                row["q0F"] |= label == "FOREIGN"
                continue
            if label == "FOREIGN" and row["BF"] is None:
                row["BF"] = int(position)
            if label == "MODEL" and row["BM"] is None:
                row["BM"] = int(position)
            if value in foreign_values and row["BMf"] is None:
                row["BMf"] = int(position)
        rows.append(row)
    index = {ident: i for i, ident in enumerate(identifiers)}
    weights = S.boot_counts(len(identifiers), nboot, seed=501)

    def summarize(selected):
        chosen = [row for row in rows if selected(row)]
        ids = np.asarray([index[r["item"]] for r in chosen], int)
        counts = np.bincount(ids, minlength=len(identifiers))
        denominator = weights @ counts
        curves = {}
        for field in ("BF", "BM", "BMf"):
            for q in (3, 6, 12):
                val = np.asarray([float(r[field] is not None and r[field] <= q) for r in chosen])
                numerator = np.bincount(ids, weights=val, minlength=len(identifiers))
                draws = np.divide(weights @ numerator, denominator, out=np.full(nboot, np.nan), where=denominator > 0)
                if field == "BMf" and any(r["foreign_item"] is None for r in chosen):
                    draws[:] = np.nan
                curves[field, q] = (float(val.mean()) if len(val) else np.nan, draws)
        def difference(a, b):
            return curves[a, 12][0] - curves[a, 3][0] - (curves[b, 12][0] - curves[b, 3][0] if b else 0), \
                   curves[a, 12][1] - curves[a, 3][1] - (curves[b, 12][1] - curves[b, 3][1] if b else 0)
        gf, gm = difference("BF", None), difference("BM", "BMf")
        total = (gf[0] + gm[0], gf[1] + gm[1])
        om = (gm[0] / total[0] if total[0] > 0 else np.nan,
              np.divide(gm[1], total[1], out=np.full(nboot, np.nan), where=total[1] > 0))
        stats = {"GF": S.summ(*gf, margin=.04), "GMc": S.summ(*gm), "Omega": S.summ(*om), "G": S.summ(*total, margin=.04),
                 "C": {f"{name}_{q}": point for (name, q), (point, _) in curves.items()}, "n_chains": len(chosen)}
        return stats, {"GF": gf[1], "GMc": gm[1], "Omega": om[1], "G": total[1]}
    known, kd = summarize(lambda r: r["stratum"] == "known")
    prone, pd = summarize(lambda r: r["stratum"] == "prone")
    alls, ad = summarize(lambda r: True)
    draws = {"GF_known": kd["GF"], "Omega_prone": pd["Omega"]}
    result = {"GF_known": known["GF"], "Omega_prone": prone["Omega"], "Omega_all": alls["Omega"],
              "G_all": alls["G"], "GMc_prone": prone["GMc"], "known": known, "prone": prone, "all": alls,
              "n_items": len(identifiers), "n_chains": len(rows), "q0_foreign": float(np.mean([r["q0F"] for r in rows])),
              "foreign_unmatched_items": sorted(set(identifiers) - set(foreign)),
              "kconf_model_share": float(np.mean([r["cls_kconf"] == "MODEL" for r in rows if r["k_star_conf"] is not None]))
                  if any(r["k_star_conf"] is not None for r in rows) else np.nan,
              "holm": {
                  "model": _holm(draws, [("GF_known", "inside", .04), ("Omega_prone", "greater", .5)]),
                  "shared": _holm(draws, [("GF_known", "greater", 0), ("Omega_prone", "greater", .5)]),
                  "lens": _holm(draws, [("GF_known", "greater", 0), ("Omega_prone", "less", .5)]),
              }, "bootstrap_unit": "item", "digit_rule": "completed maximal digit runs; leading minus ignored"}
    result["by_block_variant"] = {f"{b}/{v}/{p}": summarize(lambda r, b=b, v=v, p=p:
        (r["block"], r["variant"], r["ptype"]) == (b, v, p))[0]
        for b, v, p in sorted({(r["block"], r["variant"], r["ptype"]) for r in rows})}
    return result, rows


def pattern(ctx, rows, generated, names, nboot):
    """Inherited between-item and within-item rank estimands, item bootstrap."""
    # Only these seed ranks share the original item's pre-answer source state.
    # Foreign states and post-answer states remain descriptive owner records.
    rows = [r for r in rows if r.get("variant", "V1") == "V1" and r.get("ptype", "t0") == "t0"]
    if "P" not in names or not rows:
        return None
    ids = sorted({r["item"] for r in rows if generated.get(r["item"], {}).get("tstar") is not None})
    if len(ids) < 4:
        return None
    index = {ident: i for i, ident in enumerate(ids)}
    weights = S.boot_counts(len(ids), nboot, seed=502)
    blocks = sorted({r["block"] for r in rows})
    activity = np.stack([generated[ident]["tstar"] for ident in ids])
    grouped = {}
    for row in rows:
        if row["item"] in index and row.get("share_H") is not None and len(row["share_H"]) >= 7:
            grouped.setdefault((row["block"], row["item"]), []).append(
                (float(row["BF"] is not None and row["BF"] <= 6), float(np.mean(row["share_H"][1:7]))))
    if not grouped:
        return None

    def between(column, multiplicity=None):
        values = []
        for block in blocks:
            selected = [ident for ident in ids if (block, ident) in grouped]
            x = np.asarray([activity[index[ident], column] for ident in selected])
            y = np.asarray([np.mean([r[0] for r in grouped[block, ident]]) for ident in selected])
            repeat = np.ones(len(selected), int) if multiplicity is None else np.asarray([multiplicity[index[i]] for i in selected], int)
            values.append(S.spearman(np.repeat(x, repeat), np.repeat(y, repeat)))
        return float(np.mean([x for x in values if np.isfinite(x)])) if any(np.isfinite(values)) else np.nan

    def within(generator=None):
        sums, counts, dropped = np.zeros(len(ids)), np.zeros(len(ids)), 0
        for (_, ident), pairs in grouped.items():
            if len(pairs) < 3 or len({pair[0] for pair in pairs}) < 2:
                dropped += 1
                continue
            y, x = np.asarray(pairs).T
            correlation = S.spearman(generator.permutation(x) if generator is not None else x, y)
            if np.isfinite(correlation):
                sums[index[ident]] += correlation
                counts[index[ident]] += 1
        return sums, counts, dropped
    p = names.index("P")
    sums, counts, dropped = within()
    rb = between(p)
    rb_draws = np.asarray([between(p, row) for row in weights])
    rw = float(sums.sum() / counts.sum()) if counts.sum() else np.nan
    denominator = weights @ counts
    rw_draws = np.divide(weights @ sums, denominator, out=np.full(nboot, np.nan), where=denominator > 0)
    null = []
    for seed in range(10 if ctx.mock else 200):
        s, n, _ = within(np.random.default_rng(seed))
        null.append(s.sum() / n.sum() if n.sum() else np.nan)
    controls = [between(k) for k, name in enumerate(names) if name.startswith("C")]
    rank = K.p_rank(rb, controls) if len(controls) == 99 and np.isfinite(rb) and all(np.isfinite(controls)) else 1.0
    draws = {"rb": rb_draws, "rw": rw_draws}
    return {"rho_b": S.summ(rb, rb_draws, margin=.12), "rho_w": S.summ(rw, rw_draws, margin=.05),
            "rho_w_null": K.dist(null), "rank_p": rank, "control_rho": controls, "n_items": len(ids),
            "n_const_dropped": dropped, "n_share_rows": sum(map(len, grouped.values())),
            "primary_cell": "V1/t0: identical target pre-answer state across seed ranks",
            "holm": {"model": _holm(draws, [("rb", "greater", 0), ("rw", "inside", .05)]),
                     "lens": _holm(draws, [("rb", "inside", .12), ("rw", "greater", 0)]),
                     "both": _holm(draws, [("rb", "greater", 0), ("rw", "greater", 0)]),
                     "none": _holm(draws, [("rb", "inside", .12), ("rw", "inside", .05)])},
            "bootstrap_unit": "item"}


def step_observations(gate, generated):
    """Read strictly before a step's line, including the very first CoT line."""
    observations, parsed, total, misaligned = [], 0, 0, 0
    for ident, output in generated.items():
        for cot in output.get("cot", []):
            total += 1
            if not cot.get("alignment_ok", False):
                misaligned += 1
                continue
            steps = parse_cot(cot["text"], cot["offsets"], gate[ident])
            if not steps:
                continue
            parsed += 1
            offsets = np.asarray(cot["offsets"])
            ends = np.asarray(cot.get("token_ends", [0] + list(offsets[2:]) + [len(cot["text"])]))
            for step, (start, equal, correct, kind) in enumerate(steps):
                # A Gemma token can contain both a newline and the next
                # operand. Its start precedes the line but it already reveals
                # the step: require the entire token to end before the line.
                before = int(np.searchsorted(ends, start, side="right")) - 1
                at_equal = int(np.searchsorted(offsets, equal, side="right")) - 1
                if not 0 <= before < len(cot["a"]) or not 0 <= at_equal < len(cot["a"]):
                    continue
                observations.append({"item": ident, "sample": cot["sample"], "k": step, "y": float(not correct),
                    "a": np.asarray(cot["a"][before], float), "e": float(cot["e"][before]),
                    "a_eq": np.asarray(cot["a"][at_equal], float), "e_eq": float(cot["e"][at_equal]),
                    "t": int(cot["absolute_positions"][before]) + 1,
                    "t_eq": int(cot["absolute_positions"][at_equal]) + 1,
                    "kind": kind, "line_start_char": start, "read_token_index": before,
                    "read_char_offset": int(offsets[before]), "read_char_end": int(ends[before]),
                    "equal_token_contains_result": bool(re.search(r"\d", cot["text"][equal + 1:int(ends[at_equal])]))})
                if not correct:
                    break
    mixed = {x["item"] for x in observations if x["y"]} & {x["item"] for x in observations if not x["y"]}
    return observations, {"parse_share": parsed / max(total, 1), "parse_ok": bool(total and parsed / total >= .7),
                          "n_obs": len(observations), "n_cot": total, "n_parsed": parsed, "n_misaligned": misaligned,
                          "n_events": sum(bool(x["y"]) and x["item"] in mixed for x in observations),
                          "error_kinds": {kind: sum(x["kind"] == kind for x in observations) for kind in ("selection", "result")}}


def hazard(ctx, gate, generated, names, nboot):
    """Item-FE, step and entropy controls; all native comparison populations."""
    observations, result = step_observations(gate, generated)
    if not observations or "P" not in names or len({row["y"] for row in observations}) < 2:
        result["fit_available"] = False
        return result
    ids = sorted({row["item"] for row in observations})
    index = {ident: i for i, ident in enumerate(ids)}
    fe = np.asarray([index[row["item"]] for row in observations])
    y = np.asarray([row["y"] for row in observations])
    step = np.eye(max(row["k"] for row in observations) + 1)[[row["k"] for row in observations]][:, 1:]
    activity = np.stack([row["a"] for row in observations])
    entropy = np.asarray([row["e"] for row in observations])
    logposition = np.log([row["t"] for row in observations])
    weights = S.boot_counts(len(ids), min(nboot, 30) if ctx.mock else nboot, seed=503)
    z = lambda values: (values - values.mean()) / max(values.std(), 1e-12)

    def fit(column, with_entropy=True, multiplicity=None, secondary=False, pooled=False):
        keep = np.asarray([row["kind"] != "selection" and not row["equal_token_contains_result"]
                           for row in observations]) if secondary else np.ones(len(y), bool)
        value = np.stack([row["a_eq"] for row in observations])[:, column] if secondary else activity[:, column]
        ent = np.asarray([row["e_eq"] for row in observations]) if secondary else entropy
        pos = np.log([row["t_eq"] for row in observations]) if secondary else logposition
        covariates = [z(value)] + ([z(ent)] if with_entropy else []) + [pos, step]
        X = np.column_stack(covariates)
        fixed = np.asarray([int(gate[row["item"]]["stratum"] == "prone") for row in observations]) if pooled else fe
        nfixed = 2 if pooled else len(ids)
        if len(set(y[keep])) < 2:
            return np.nan
        beta, _, _ = S.logit_fe(y[keep], fixed[keep], X[keep], nfixed,
                               weights=None if multiplicity is None else multiplicity[fe[keep]])
        return float(beta[0])
    p = names.index("P")
    cache = {}
    def repeated(identity, function):
        # The full 2,000 item replicates can outlive a Colab time slice. Store
        # modest batches so continuation resumes the same draws and estimand.
        values = []
        for start in range(0, len(weights), 25):
            batch = weights[start:start + 25]
            run = lambda batch=batch: [function(row) for row in batch]
            values.extend(_cache(ctx, "hazard_boot", (identity, start), run) if hasattr(ctx, "cached") else run())
        return np.asarray(values)

    def estimate(column, with_entropy=True):
        key = (column, with_entropy)
        if key not in cache:
            cache[key] = (fit(column, with_entropy), repeated(key, lambda row: fit(column, with_entropy, row)))
        return cache[key]
    point, draws = estimate(p)
    point0, draws0 = estimate(p, False)
    control = {name: fit(k) for k, name in enumerate(names) if name.startswith("C")}
    rank = K.p_rank(point, list(control.values())) if len(control) == 99 and np.isfinite(point) and all(np.isfinite(list(control.values()))) else 1.0
    channel_points = {name: fit(k) for k, name in enumerate(names)}
    contrasts = {}
    if "UNC" in names:
        pu, du = estimate(names.index("UNC"))
        contrasts["minus_unc"] = S.summ(point - pu, draws - du)
    # E is entropy itself, so omit the second entropy regressor in this model.
    def fit_e(multiplicity=None):
        X = np.column_stack([z(entropy), logposition, step])
        beta, _, _ = S.logit_fe(y, fe, X, len(ids), weights=None if multiplicity is None else multiplicity[fe])
        return float(beta[0])
    pe = fit_e()
    de = repeated("E-without-duplicate-entropy", fit_e)
    contrasts["minus_E"] = S.summ(point - pe, draws - de)
    boot = [value for name, value in channel_points.items() if name.startswith("B")]
    robust = (sum(value > 0 and K.p_rank(value, list(control.values())) <= .05 for value in boot) >= 16) if len(boot) == 20 and len(control) == 99 else None
    result.update(fit_available=True, beta_H=S.summ(point, draws, margin=.15), beta_H_noe=S.summ(point0, draws0),
                  beta_H_eq_result=fit(p, secondary=True), beta_H_pooled=fit(p, pooled=True),
                  beta_E_without_duplicate_entropy=S.summ(pe, de), channel_coefficients=channel_points,
                  rank_p=rank, p_cf_herr=max(K.p_one_sided(draws, 0), rank), sel_robust=robust,
                  not_beyond_unc=(not RU.LB(contrasts["minus_E"]) or not RU.LB(contrasts["minus_unc"]))
                                  if "minus_unc" in contrasts else None,
                  **contrasts, bootstrap_unit="item", n_bootstrap=len(weights),
                  pre_step_rule="last target token strictly before line start; prompt-final state for first line",
                  result_secondary="selection errors excluded; read at equals token; descriptive")
    return result


def decide(stats, gates):
    """Inherited rule order, explicit scientific gates and frozen Holm families."""
    a, b, c = stats.get("L5a"), stats.get("L5b"), stats.get("L5c") or {}
    if not gates["G_CL"] or not a:
        va = {"verdict": "not testable (no exact-label lens corpus from K)"}
    elif not gates["target_lens"]:
        va = {"verdict": "not testable (no validated target J-lens)"}
    elif not gates.get("lens_controls", False):
        va = {"verdict": "not testable (target lens positive-control gates closed)"}
    elif not gates["cal_L5a"]:
        va = {"verdict": RU.NT_CAL}
    else:
        value = RU.rule_L5a(a)
        case = {"break-off owned by the model": "model", "break-off owned by the lens": "lens",
                "shared (the lens adds confabulation to the model's own errors)": "shared"}.get(value)
        va = {"verdict": "undecided" if case and not a["holm"][case]["pass"] else value}
    b_ready = (gates["G_CL"] and gates["target_lens"] and gates.get("lens_controls", False)
               and gates["cal_L5a"] and gates["SG8"] and gates["controls_99"] and gates["share_H"] and bool(b))
    vb = RU.rule_L5b(b, gcl=True, sg8=True) if b_ready else {"verdict": "not testable", "labels": []}
    if b_ready:
        case = {"H-pattern owned by the model": "model", "H-pattern owned by the lens": "lens",
                "H-pattern in both": "both", "no H-pattern": "none"}.get(vb["verdict"])
        if case and not b["holm"][case]["pass"]:
            vb["verdict"] = "undecided"
    if not gates["own_sample_size"]:
        vc = {"verdict": "not testable (fewer than 600 own-CoT items)", "labels": []}
    elif not gates["controls_99"]:
        vc = {"verdict": "not testable (fewer than 99 matched control populations)", "labels": []}
    elif not c.get("fit_available"):
        vc = {"verdict": "not testable (own-error hazard is not identifiable)", "labels": []}
    else:
        vc = RU.rule_L5c(c, sg8=gates["SG8"], cal_ok=gates["cal_L5c"])
    verdicts = {"L5a": va, "L5b": vb, "L5c": vc}
    return {"verdicts": verdicts, "combined": {"V-OWN": RU.v_own(va["verdict"], vb["verdict"])},
            "CF_HERR_member": c.get("p_cf_herr", 1.) if vc["verdict"].startswith("H predicts") else 1.,
            "gates": gates}


def analyze(ctx, packet=None):
    from r1b_g_l import config
    packet = packet or _pt(ctx, "l_owner", "l5_gen.pt")
    if not packet:
        raise K.NotRun("G L5 native generation artifact missing")
    cf = config(ctx)
    gate, generated, records, names = (packet[key] for key in ("gate", "gen", "records", "channels"))
    a, classified = ownership(gate, records, generated, cf["nboot"])
    b = pattern(ctx, classified, generated, names, cf["nboot"])
    c = hazard(ctx, gate, generated, names, cf["nboot"])
    stats = {"L5a": a, "L5b": b, "L5c": c, "mock": bool(ctx.mock), "sample_counts": packet["sample_counts"]}
    pop = _json(ctx, "l_prepare", "l_pop.json")
    calibration = _json(ctx, "l_calibrate", "l_cal.json").get("gates", {})
    lens = _json(ctx, "lens_validate", "lens_validation.json")
    chain_gates = packet["lens_labels"].get("chain_analysis", {}).get("gates") or {}
    ids = {row["item"] for row in records}
    known = sum(gate[i]["stratum"] == "known" for i in ids)
    prone = len(ids) - known
    complete_answers = all(len(gate[i]["answers"]) == 5 and len(generated[i]["gate"]) == 6 for i in ids)
    gates = {"G_CL": bool(known >= 150 and prone >= 150 and complete_answers and not ctx.mock),
             "known": known, "prone": prone,
             "target_lens": lens.get("G_K_J", {}).get("pass") is True and not ctx.mock,
             "lens_controls": all(_passed(chain_gates.get(name)) for name in ("G_K4", "G_K7")) and not ctx.mock,
             "SG8": pop.get("sg8_ok") is True and not ctx.mock,
             "cal_L5a": calibration.get("L5a") is True and not ctx.mock,
             "cal_L5c": calibration.get("L5c") is True and not ctx.mock,
             "controls_99": len([name for name in names if name.startswith("C")]) == 99,
             "share_H": bool(records) and all(r.get("share_H") is not None and len(r["share_H"]) >= 13 for r in records),
             "own_sample_size": len(generated) >= 600 and all(len(g["cot"]) == 5 for g in generated.values()) and not ctx.mock,
             "G_parse": c["parse_ok"], "within_item_error_events": c["n_events"]}
    decision = decide(stats, gates)
    decision["model"] = ctx.model_name
    decision["mock"] = bool(ctx.mock)
    decision["lens_labels"] = packet["lens_labels"]
    ctx.write_json("l5_stats.json", clean(stats))
    ctx.write_json("l5_verdicts.json", clean(decision))
    coverage = {name: {"status": "blocked" if val["verdict"].startswith("not testable") else "measured",
                       "reason": val["verdict"]} for name, val in decision["verdicts"].items()}
    result = {"status": "partial" if any(v["status"] == "blocked" for v in coverage.values()) else "done",
              "coverage": coverage, "gates": gates, "mock": bool(ctx.mock), "independent_of_qwen_runs": True}
    ctx.write_json("l_owner.json", clean(result))
    return result


def measure(ctx):
    from r1b_g_engine import ModelEngine
    from r1b_g_k_chain import gate_arithmetic, normalize_item
    from r1b_g_l import config
    engine, cf = ModelEngine(ctx), config(ctx)
    raw = [normalize_item(row) for row in ctx.data().get("arithmetic", ctx.data().get("ach_short", []))]
    raw = [row for row in raw if row["split"] == "A-lens" and not row.get("cal")]
    if ctx.mock:
        raw = raw[:cf["l5_items"]]
    if not raw:
        result = {"status": "blocked", "reason": "L5 needs raw non-calibration A-lens arithmetic items"}
        ctx.write_json("l_owner.json", result)
        return result
    chain = _pt(ctx, "k_chain", "chains.pt")
    raw_by_id = {row["id"]: row for row in raw}
    record_ids = list(dict.fromkeys(row["item"] for row in chain.get("records", chain.get("rows", []))
                                    if row.get("variant") in ("V1", "V2") and row.get("item") in raw_by_id))
    selected_ids = (record_ids + [row["id"] for row in raw if row["id"] not in set(record_ids)])[:cf["l5_items"]]
    raw = [raw_by_id[ident] for ident in selected_ids]
    measured = dict(chain.get("gating", {}).get("gate", {}))
    missing = [row for row in raw if row["id"] not in measured]
    for row in missing:
        if ctx.time_left() < 30:
            raise K.StepYield("L5 gating checkpoint saved; continue the same G run")
        measured.update(gate_arithmetic(ctx, engine, [row]))
    gate = build_gate(engine, raw, measured)
    records = lens_records(chain, engine.tok, gate)
    with_record = list(dict.fromkeys(row["item"] for row in records))
    selected = (with_record + [row["id"] for row in raw if row["id"] not in set(with_record)])[:cf["l5_items"]]
    gate = {ident: gate[ident] for ident in selected}
    records = [row for row in records if row["item"] in gate]
    pop = _json(ctx, "l_prepare", "l_pop.json")
    refs = _pt(ctx, "l_prepare", "refw.pt")
    groups = populations(pop) if refs else {}
    generated = {}
    for ident in selected:
        generated[ident] = _cache(ctx, "item", ident, lambda ident=ident: generate_item(ctx, engine, gate[ident], groups, refs, cf))
    chain_analysis = _json(ctx, "k_chain_analyze", "chain_analysis.json")
    packet = {"gate": gate, "gen": generated, "records": records, "channels": list(groups), "populations": groups,
              "sample_counts": {"items": len(generated), "extra_gates": 6, "cot_per_item": 5, "sub_per_expression": 5,
                                "cot_max_new": cf["l5_cot_new"], "sub_max_new": cf["l5_sub_new"]},
              "lens_labels": {"metadata": chain.get("metadata", {}),
                              "chain_analysis": {name: chain_analysis.get(name) for name in
                                                 ("gates", "scientific_verdict", "mock", "model")}},
              "mock": bool(ctx.mock), "model": ctx.model_name,
              "refw_definition": "mean per-neuron standardized native CETT", "checkpoint_unit": "item/sample"}
    ctx.save_pt("l5_gen.pt", packet, config=packet["sample_counts"])
    return analyze(ctx, packet)
