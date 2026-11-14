"""Own Gemma L1 mean/zero erasure and held-out damage matching."""
from __future__ import annotations

import contextlib

import numpy as np

import r1b_core as K
import r1b_l_stats as S
from r1b_l_base import SHORT_QA, clean, rng, scorers


class Damage:
    """Exact SO-DMG with CPU hidden references and a bounded vocabulary head.

    A float64 [all reference tokens, vocabulary] bank on the GPU can exhaust
    A100-80GB beside the 31B target. Store final normalized hidden states on CPU
    instead, then re-project baseline and intervention in 32-position chunks.
    The KL direction, per-text weighting and native Gemma softcap are unchanged.
    """

    def __init__(self, model, refs, chunk=32):
        import torch
        import r1b_g_adapter as A
        self.model, self.refs, self.chunk = model, refs, int(chunk)
        self.device = next(model.parameters()).device
        self.items = []
        if not refs.items or self.chunk < 1:
            raise ValueError("damage requires references and a positive head chunk")
        with torch.no_grad():
            for ids, positions in refs.items:
                if any(p < 0 or p >= len(ids) - 1 for p in positions):
                    raise ValueError("damage reference positions must predict continuation tokens")
                x = torch.tensor([ids], device=self.device)
                hidden = A.text_model(model)(input_ids=x, use_cache=False).last_hidden_state[0]
                self.items.append((list(ids), list(positions), hidden[positions].detach().cpu()))

    def __call__(self, setup=None):
        import torch
        import r1b_g_adapter as A
        divergences, changes = [], []
        with torch.no_grad():
            for ids, positions, baseline in self.items:
                with setup() if setup else contextlib.nullcontext():
                    hidden = A.text_model(self.model)(input_ids=torch.tensor([ids], device=self.device), use_cache=False).last_hidden_state[0]
                changed = hidden[positions]
                kl, top = 0.0, 0
                for start in range(0, len(positions), self.chunk):
                    end = min(start + self.chunk, len(positions))
                    a = A.unembed(self.model, baseline[start:end], normalized=True).double().log_softmax(-1)
                    b = A.unembed(self.model, changed[start:end], normalized=True).double().log_softmax(-1)
                    kl += float((a.exp() * (a - b)).sum())
                    top += int((a.argmax(-1) != b.argmax(-1)).sum())
                divergences.append(kl / len(positions))
                changes.append(top / len(positions))
        return {"damage": float(np.mean(divergences)), "per_text": divergences,
                "top1_change": float(np.mean(changes)), "per_text_top1": changes,
                "refs": self.refs.name, "refs_sha256": self.refs.sha256()}


@contextlib.contextmanager
def partial_mean(engine, neurons, means, lam):
    """z <- mu + lambda (z-mu), with ragged neuron coordinates."""
    import torch
    grouped, handles = {}, []
    for flat in neurons:
        layer, j = engine.layout.decode(int(flat))
        grouped.setdefault(layer, []).append((int(flat), j))
    try:
        for l, units in grouped.items():
            flat, local = zip(*units)
            def hook(module, args, flat=flat, local=local):
                out = args[0].clone()
                mu = torch.as_tensor(means[list(flat)], device=out.device, dtype=out.dtype)
                z = out[..., list(local)]
                out[..., list(local)] = mu + float(lam) * (z - mu)
                return (out,) + args[1:]
            handles.append(engine.layers[l].mlp.down_proj.register_forward_pre_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def conditions(ctx, population, n, cf):
    out = []
    for draw in range(cf["era_draws"]):
        perm = rng("r1bL-ERA", draw).permutation(n)
        for f in cf["era_fr"]:
            units = sorted(perm[:max(1, int(round(f * n)))].tolist())
            out.append({"name": f"mean_f{f:.6g}_d{draw}", "ids": units, "kind": "mean", "lam": 0.0})
            if draw == 0 and f in (cf["era_fr"][len(cf["era_fr"]) // 2], cf["era_fr"][-1]):
                out.append({"name": f"zero_f{f:.6g}_d0", "ids": units, "kind": "zero"})
    out.append({"name": "P", "ids": population["P"], "kind": "mean", "lam": 0.0})
    if population["H_tq"] and population["H_tq"] != population["P"]:
        out.append({"name": "H_tq", "ids": population["H_tq"], "kind": "mean", "lam": 0.0})
    start = 12 if len(population["CTRL"]) >= 20 else 0
    # Never duplicate a control to fabricate the required eight independent sets.
    for j, units in enumerate(population["CTRL"][start:start + cf["era_sets"]], start):
        out.append({"name": f"R{j}", "ids": units, "kind": "mean", "ctrl": j})
    return out


def _reference(engine, rows, name, required):
    if len(rows) < required:
        raise ValueError(f"{name} needs {required} independent raw-text rows")
    pairs = []
    for row in rows[:required]:
        if isinstance(row, dict) and "prompt" in row and "continuation" in row:
            prompt = engine.encode(row["prompt"])
            continuation = engine.encode(row["continuation"], chat=False)
            pairs.append((prompt + continuation, list(range(len(prompt) - 1, len(prompt) + len(continuation) - 1))))
        else:
            text = row if isinstance(row, str) else row.get("text")
            if not isinstance(text, str):
                raise ValueError(f"{name} must contain raw text, not another model's token IDs")
            ids = engine.encode(text, chat=False)
            if engine.ctx.mock:
                ids = ids[:min(128, int(engine.cfg.max_position_embeddings))]
            pairs.append((ids, list(range(len(ids) - 1))))
    return K.RefSet(pairs, name)


def _summarize(ctx, result, calibration, cf):
    cls, out = np.asarray(result["cls"]), result["out"]
    cc, ci, fab = cls == "CC", cls == "CI", cls == "NE"
    def rate(name, mask, key):
        return float(np.asarray(out[name][key])[mask].mean()) if mask.any() else np.nan
    curve = [rate("base", cc, "correct")]
    for f in cf["era_fr"]:
        curve.append(float(np.mean([rate(f"mean_f{f:.6g}_d{d}", cc, "correct") for d in range(cf["era_draws"])])))
    wrong = lambda name: (~np.asarray(out[name]["correct"]) & ~np.asarray(out[name]["refusal"])).astype(float)
    nonref = lambda name: (~np.asarray(out[name]["refusal"])).astype(float)
    controls = [name for name in out if name.startswith("R")]
    g = np.random.default_rng(7)
    def two_level(fn, ix):
        if not len(ix) or not controls:
            return None
        p = fn("P")[ix]
        r = np.stack([fn(name)[ix] for name in controls])
        draws = []
        for _ in range(cf["nboot"]):
            ii = g.integers(0, len(ix), len(ix))
            ss = g.integers(0, len(r), len(r))
            draws.append(p[ii].mean() - r[ss][:, ii].mean())
        return S.summ(p.mean() - r.mean(), draws, margin=.05)
    stats = {"cc_acc_curve": curve, "sizes": cf["era_fr"],
             "diff_ci_wrong": two_level(wrong, np.flatnonzero(ci)),
             "diff_fab_nonref": two_level(nonref, np.flatnonzero(fab)), "g_dmg": calibration["g_dmg"],
             "rates": {name: {"cc_acc": rate(name, cc, "correct"), "ci_wrong": float(wrong(name)[ci].mean()) if ci.any() else None,
                              "refusal": float(np.mean(out[name]["refusal"])), "fab_nonref": float(nonref(name)[fab].mean()) if fab.any() else None,
                              "dlogq_cc": float(np.mean(out[name]["lpgold"]) - np.mean(out["base"]["lpgold"])) if out[name]["lpgold"] else None}
                       for name in out}, "bootstrap_unit": "question and matched control set", "damage": calibration}
    ctx.write_json("l1_erasure.json", clean(stats))
    return stats


def erasure(ctx):
    from r1b_g_engine import ModelEngine
    from r1b_g_l import config, load_pop, _read_json, _path, loadpt
    cf = config(ctx)
    pop = load_pop(ctx)
    data = ctx.data()
    nref = 4 if ctx.mock else 16
    refs = loadpt(ctx.artifact("k_prepare", "references.pt"))
    ref_r16 = refs.get("ref_r16", [])
    nr = 2 if ctx.mock else 16
    if len(data.get("ref_k16", [])) < nref or len(ref_r16) < nr:
        return {"status": "blocked", "reason": f"own recalibrated REF-R16 needs {nr} rows; raw REF-K16 needs {nref} rows"}
    E = ModelEngine(ctx)
    items = _read_json(_path(ctx, "items.json"))["items"]
    sl = sorted((r for r in items if r["split"] == "S_L"), key=lambda r: r["id"])
    nheld = cf["ref_heldout"]
    if len(sl) <= nheld:
        return {"status": "blocked", "reason": f"need S_L rows beyond {nheld} held-out damage-reference questions"}
    held = sl[-nheld:]
    held_ids = {r["id"] for r in held}
    rows = [r for r in items if (r["split"] == "S_L" or r["split"] == "NE") and r["id"] not in held_ids]
    if not rows:
        return {"status": "blocked", "reason": "no disjoint erasure evaluation questions"}
    if not any(r["cls"] == "CC" for r in rows):
        return {"status": "blocked", "reason": "no CC erasure questions remain after holding out damage references"}
    refu = K.RefSet([(r["ids"], r["positions"]) for r in ref_r16[:nr]], "G-REF-R16") + _reference(E, data["ref_k16"], "G-REF-K16", nref)
    held_ref = K.RefSet.from_prompt_conts([E.encode(SHORT_QA.format(q=r["question"])) for r in held], [r["ans0"] for r in held], "own S_L held-out")
    dmain, du = Damage(E.model, refu + held_ref), Damage(E.model, refu)
    mu = loadpt(_path(ctx, "refw.pt"))["mu_z"]
    conds = conditions(ctx, pop, E.layout.total, cf)
    def damage(units, lam, metric=dmain):
        return metric(lambda: partial_mean(E, units, mu, lam))["damage"]
    dP = damage(pop["P"], 0)
    calibration = {"P": {"damage": dP, "damage_U32": damage(pop["P"], 0, du)}, "sets": {}, "heldout_ids": sorted(held_ids)}
    for c in conds:
        if "ctrl" not in c:
            continue
        def calibrate(c=c):
            units, extra = list(c["ids"]), []
            full = damage(units, 0)
            if full < .9 * dP:
                g = np.random.default_rng(ctx.key("r1bL-CTRL", "dbl", c["ctrl"]))
                counts = np.bincount([E.layout.decode(j)[0] for j in units], minlength=len(E.layout.widths))
                for l, n in enumerate(counts):
                    pool = np.setdiff1d(np.arange(E.layout.offsets[l], E.layout.offsets[l + 1]), units)
                    if len(pool) < n:
                        return {"ids": units, "lam": 0.0, "damage": full, "in_band": False,
                                "labels": ["not enough same-layer units for doubled control"], "damage_U32": damage(units, 0, du)}
                    extra.extend(int(j) for j in g.choice(pool, n, replace=False))
                units = sorted(set(units) | set(extra))
                full = damage(units, 0)
            # Shared gain rule uses (lambda-1); start at full ablation lambda=0.
            cal = K.calibrate(lambda lam: damage(units, lam), 0.0, dP, form="gain", max_eval=6, cap=1.0)
            return {"ids": units, "doubled": bool(extra), "full_damage": full, "lam": cal["f"], "damage": cal["damage"],
                    "in_band": cal["in_band"], "labels": cal["labels"], "tried": cal["tried"], "damage_U32": damage(units, cal["f"], du)}
        calibration["sets"][c["name"]] = ctx.cached(f"l1_erasure_cal_{c['name']}", calibrate)
    calibration["g_dmg"] = len(calibration["sets"]) == cf["era_sets"] and all(v["in_band"] for v in calibration["sets"].values())
    ctx.write_json("l1_era_cal.json", clean(calibration))
    prompts = [E.encode(SHORT_QA.format(q=r["question"])) for r in rows]
    alias, refusal, scorer = scorers()
    out = {}
    for condition in [{"name": "base", "ids": None}] + conds:
        def run(c=condition):
            factory = None
            if c["ids"] is not None:
                if "ctrl" in c:
                    cal = calibration["sets"][c["name"]]
                    factory = lambda: partial_mean(E, cal["ids"], mu, cal["lam"])
                elif c["kind"] == "zero":
                    factory = lambda: E.intervene(c["ids"], op="zero")
                else:
                    factory = lambda: partial_mean(E, c["ids"], mu, c["lam"])
            gen = E.decode(prompts, keys=[0] * len(prompts), temperature=0, max_new=cf["era_new"], setup=factory)
            gold = []
            for j, row in enumerate(rows):
                if row["cls"] != "CC":
                    continue
                lp = E.teacher_force([prompts[j] + row["ans0"]], setup=factory)["rows"][0]["next_logprobs"]
                gold.append(float(lp[len(prompts[j]) - 1:].mean()))
            return {"correct": [bool(alias(g["text"], r["aliases"])) for g, r in zip(gen, rows)],
                    "refusal": [bool(refusal(g["text"])) for g in gen], "lpgold": gold,
                    "tokens": [g["ids"] for g in gen], "texts": [g["text"] for g in gen]}
        out[condition["name"]] = ctx.cached(f"l1_erasure_{condition['name']}", run)
    result = {"out": out, "cls": [r["cls"] for r in rows], "ids": [r["id"] for r in rows], "scorer": scorer,
              "heldout_ids": sorted(held_ids), "conditions": [{"name": c["name"], "n": len(c["ids"] or [])} for c in [{"name": "base", "ids": None}] + conds]}
    ctx.save_pt("l1_era.pt", result)
    stats = _summarize(ctx, result, calibration, cf)
    return {"status": "measured", "g_dmg": calibration["g_dmg"], "n_items": len(rows), "n_controls": len(calibration["sets"]),
            "outcome_classes": sorted(set(result["cls"])), "causal_verdict_ready": bool(stats["diff_ci_wrong"] and stats["diff_fab_nonref"] and calibration["g_dmg"])}
