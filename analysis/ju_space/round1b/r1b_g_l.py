"""Independent Gemma population and L-core entry points.

The scientific statistics/rules are inherited from part L. All model calls,
tokenisation, selected populations and references belong to this Gemma run.
Missing controls are explicit gates; a runnable stage is not a passing study.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import r1b_l_stats as S
from r1b_l_base import cfg as inherited_cfg, clean, rng, SHORT_QA, answer_span


class GuardedContext:
    """Check the shared time guard before each newly computed cache piece."""
    def __init__(self, ctx):
        self.ctx = ctx

    def __getattr__(self, name):
        return getattr(self.ctx, name)

    def cached(self, name, fn, **kwargs):
        def guarded():
            if hasattr(self.ctx, "time_left") and self.ctx.time_left() < 30:
                import r1b_core as K
                raise K.StepYield("G L continuation checkpoint reached the time guard")
            return fn()
        return self.ctx.cached(name, guarded, **kwargs)


def config(ctx):
    """Use frozen L sizes, including its explicit (never scientific) MOCK sizes."""
    return inherited_cfg(SimpleNamespace(mock=ctx.mock, trimmed=lambda tag: False))


def loadpt(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def _path(ctx, rel):
    name = rel.removeprefix("l/")
    stage = "l_prepare" if name in ("cett_rows.pt", "l_pop.json", "refw.pt", "items.json") else "l_measure"
    return ctx.artifact(stage, name)


def save(ctx, rel, obj, config=None, labels=()):
    ctx.save_pt(rel.removeprefix("l/"), obj, config=clean(config or {}), labels=list(labels))


def wj(ctx, rel, obj, config=None, labels=()):
    ctx.write_json(rel.removeprefix("l/"), clean(obj), config=clean(config or {}), labels=list(labels))


def _read_json(path):
    return json.loads(Path(path).read_text())


def dev(model):
    # The 31B fp16 target itself occupies about 62 GB. Statistical matrices and
    # planted copies must not compete with it for the remaining A100 memory.
    if getattr(model, "_r1b_g_name", None) == "gemma4-31b-qat":
        import torch
        return torch.device("cpu")
    return next(model.parameters()).device


def _cett(ctx):
    return loadpt(_path(ctx, "cett_rows.pt"))


def load_pop(ctx):
    return _read_json(_path(ctx, "l_pop.json"))


def H_C(ctx):
    return float(_read_json(ctx.artifact("k_prepare", "hn.json")).get("C_star") or .05)


def setup(ctx):
    tok, model = ctx.model()
    H = _read_json(ctx.artifact("k_prepare", "hn.json"))
    H["C_star"] = float(H.get("C_star") or .05)
    rows = _read_json(_path(ctx, "items.json"))["items"]
    return tok, model, config(ctx), H, {r["id"]: r for r in rows}


def selector(X, y, C, exclude=None):
    # This is the frozen L fallback solver, not a target-model backward pass.
    return S.l1_select(X, y.float(), C, exclude), "L FISTA selector"


def _sl(ctx, which="cmean"):
    import torch
    d = _cett(ctx)
    split, cls = np.asarray(d["split"]), np.asarray(d["cls"])
    ix = np.flatnonzero((split == "S_L") & np.isin(cls, ("CC", "CI")))
    y = (cls[ix] == "CI").astype(int)
    ix = ix[S.balance(y, seed=0)]
    y = (cls[ix] == "CI").astype(int)
    ids = [d["ids"][i] for i in ix]
    # One feature row per question; never put samples of one item in two folds.
    if len(set(ids)) != len(ids):
        raise ValueError("L feature rows must have unique question IDs")
    return d[which][ix].float(), torch.tensor(y), torch.tensor(S.folds_by_question(len(ix), 5, 0)), ids, d


def _flat(layout, coords):
    return sorted(set(layout.encode(int(a), int(b)) for a, b in coords))


def _missing(ctx, output, requirements):
    missing = [str(p) for p in requirements if not Path(p).is_file()]
    if not missing:
        return None
    result = {"status": "blocked", "missing": missing, "mock": bool(ctx.mock)}
    ctx.write_json(output, result)
    return result


def _answer(row):
    answers = row.get("answers") or []
    a = answers[0] if answers else {}
    ids = a.get("ids", a.get("tokenids", a.get("tokens", [])))
    span = a.get("span", row.get("span", [0, len(ids)]))
    if isinstance(span, int):
        span = [0, span]
    if not isinstance(ids, list) or any(type(i) is not int for i in ids):
        raise ValueError("SO-LAB answers must contain Gemma token IDs")
    return ids, span


def _population(ctx, engine, H, data, rows):
    """Frozen SEL+/UNC+ and layer/CETT matching, with ragged coordinates."""
    import torch
    from r1b_g_l_locus import hull, top_set
    cf = config(ctx)
    split, cls = np.asarray(data["split"]), np.asarray(data["cls"])
    ix = np.flatnonzero((split == "S_fit") & np.isin(cls, ("CC", "CI")))
    if len(ix) < 4 or len(set(cls[ix])) < 2:
        return {"status": "blocked", "reason": "S_fit needs measured CC and CI questions"}
    X = data["cmean"][ix].float().to(dev(engine.model))
    y = torch.tensor((cls[ix] == "CI").astype(float), device=X.device)
    hop = H.get("H_op_name", "H_tq")
    htq = _flat(engine.layout, H.get("H_tq", []))
    cstar = float(H.get("C_star") or .05)
    if hop == "H_tq" and len(htq) >= 10:
        P, pname = htq, "H_tq"
    elif hop in ("H_tq", "H+"):
        # The inherited cap applies in the ordered H+ bank before flattening.
        P, pname = _flat(engine.layout, (H.get("H_plus") or [])[:200]), "H+"
    else:
        P, pname = None, "selector on S_fit (H_ar primary)"
    if P is None:
        w, _ = selector(X, y, cstar)
        P = np.flatnonzero(w.detach().cpu().numpy() > 0).tolist()
    if not P:
        return {"status": "blocked", "reason": "no H population or positive selector coefficients"}
    k = len(P)
    hullP, partners = hull(X, P)
    q = {r["id"]: r for r in rows}
    fit_rows = [q[data["ids"][i]] for i in ix]
    candidates = (
        ("question starts with 'Who'", lambda r: r["question"].strip().startswith("Who")),
        ("question contains a digit", lambda r: any(c.isdigit() for c in r["question"])),
        ("a gold alias contains a digit", lambda r: any(c.isdigit() for a in r["aliases"] for c in a)),
    )
    minclass = 5 if ctx.mock else 300
    sel_label, yl = None, None
    for name, fn in candidates:
        v = np.asarray([fn(r) for r in fit_rows], dtype=float)
        if min(v.sum(), len(v) - v.sum()) >= minclass and abs(np.corrcoef(v, y.cpu())[0, 1]) < .1:
            sel_label, yl = name, v
            break

    def matched(v, seed):
        b = S.balance(v, seed)
        if len(b) < 4 or len(np.unique(v[b])) < 2:
            return []
        yy = torch.tensor(v[b], device=X.device)
        w, _ = selector(X[b], yy, cstar)
        return top_set(w, k, X[b], yy)

    SEL = matched(yl, 1) if yl is not None else []
    ent = np.asarray(data["ent"])[ix]
    yu = np.zeros(len(ix))
    for c in ("CC", "CI"):
        m = cls[ix] == c
        yu[m] = ent[m] > np.median(ent[m])
    UNC = matched(yu, 2)
    hS = hull(X, SEL)[0] if SEL else []
    hU = hull(X, UNC)[0] if UNC else []
    forbid = set(hullP) | set(hS) | set(hU)
    layer_counts = np.bincount([engine.layout.decode(j)[0] for j in P], minlength=len(engine.layout.widths))
    mc = X.mean(0).cpu().numpy()
    target = float(mc[P].mean())
    pool = [np.asarray([j for j in range(engine.layout.offsets[l], engine.layout.offsets[l] + w) if j not in forbid])
            for l, w in enumerate(engine.layout.widths)]
    controls, seen, attempted = [], set(), 0
    enough = all(len(pool[l]) >= n for l, n in enumerate(layer_counts))
    if enough:
        while len(controls) < cf["n_ctrl"] + 30 and attempted < 20000:
            g = np.random.default_rng(ctx.key("r1bL-CTRL", attempted))
            attempted += 1
            candidate = sorted(int(j) for l, n in enumerate(layer_counts) if n for j in g.choice(pool[l], n, replace=False))
            if tuple(candidate) not in seen and abs(float(mc[candidate].mean()) - target) <= .2 * abs(target):
                controls.append(candidate)
                seen.add(tuple(candidate))
    k3 = ctx.artifact("k_analyze", "k3_verdicts.json")
    verdicts = _read_json(k3) if Path(k3).is_file() else {}
    sg8 = verdicts.get("SG8", verdicts.get("sg8", False))
    sg8 = sg8 is True or (isinstance(sg8, dict) and sg8.get("pass") is True)
    labels = []
    if not SEL:
        labels.append("G-sel failed (no unrelated selection label)")
    if len(controls) < cf["n_ctrl"]:
        labels.append(f"only {len(controls)} distinct matched CTRL sets; target {cf['n_ctrl']}")
    if not sg8:
        labels.append("no validated H-set")
    return {"status": "measured", "P": P, "P_name": pname, "H_op_name": hop,
            "H_tq": htq, "Htq_extra": htq if pname == "H+" else [], "SEL": SEL, "UNC": UNC,
            "SEL_label": sel_label, "G_sel": bool(SEL), "CTRL": controls[:cf["n_ctrl"]],
            "CTRL_reserve": controls[cf["n_ctrl"]:], "boot": [_flat(engine.layout, v) for v in H.get("boot", [])][:cf["n_boot_sets"]],
            "hull_P": hullP, "partners_P": partners, "hull_SEL": hS, "hull_UNC": hU,
            "layer_counts": layer_counts.tolist(), "target_cett": target, "sg8_ok": sg8,
            "n_candidates": attempted, "labels": labels, "layout": engine.layout.as_dict()}


def prepare(ctx):
    """Own SO-LAB answer and pre-answer features; own REF-W and populations."""
    import torch
    from r1b_g_engine import ModelEngine
    ctx = GuardedContext(ctx)
    paths = [ctx.artifact("k_prepare", n) for n in ("hn.json", "lab.json", "references.pt")]
    blocked = _missing(ctx, "l_prepare.json", paths)
    if blocked:
        return blocked
    H, lab = map(_read_json, paths[:2])
    E = ModelEngine(ctx)
    if H.get("layout") and tuple(H["layout"]["widths"]) != tuple(E.layout.widths):
        raise ValueError("SO-HN neuron layout differs from the active Gemma target")
    cf = config(ctx)
    rows, seqs, spans, pre = [], [], [], []
    sources = list(lab.get("items", []))
    for i, source in enumerate(ctx.data().get("ne", [])):
        if isinstance(source, str):
            source = {"id": f"NE:{i}", "prompt": source}
        sources.append(dict(source, id=str(source.get("id", source.get("qid", f"NE:{i}"))),
                            question=source.get("question", source.get("prompt", "")), split="NE", cls="NE", aliases=[]))
    for source in sources:
        split = source.get("split")
        if split not in ("S_fit", "S_L", "NE"):
            continue
        row = dict(source, id=str(source.get("id", source.get("qid"))))
        row.setdefault("aliases", [])
        row.setdefault("cls", "NE" if split == "NE" else "unclassified")
        ids, span = _answer(row)
        prompt = row.get("prompt_ids") or E.encode(SHORT_QA.format(q=row["question"]))
        if not ids:
            answer = E.decode([prompt], keys=[ctx.key("r1bL", "fallback", row["id"])], temperature=0, max_new=cf["era_new"])[0]
            ids = answer["ids"]
            span = [0, answer_span(ids, E.tok)]
            row["answer_source"] = "L greedy fallback (not a K classification)"
        if not ids:
            continue
        a, b = max(0, int(span[0])), min(len(ids), int(span[1]))
        if b <= a:
            continue
        row.update(ans0=ids, span0=[a, b], prompt_ids=prompt)
        rows.append(row)
        seqs.append(prompt + ids)
        spans.append(list(range(len(prompt) + a, len(prompt) + b)))
        pre.append([len(prompt) - 1])
    if not rows:
        result = {"status": "blocked", "reason": "no measured S_fit/S_L/NE answers with valid spans"}
        ctx.write_json("l_prepare.json", result)
        return result
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("SO-LAB IDs must be unique")
    cm, cl, entropy = [], [], []
    for start in range(0, len(seqs), cf["batch"]):
        end = min(start + cf["batch"], len(seqs))
        def collect(start=start, end=end):
            answer = E.features(seqs[start:end], spans[start:end])
            means = answer["cett"].half()
            del answer
            before = E.features(seqs[start:end], pre[start:end])["cett"].half()
            entropies = []
            for i in range(start, end):
                scored = E.teacher_force([seqs[i]])["rows"][0]
                ent = np.asarray(scored["entropy"])
                entropies.append(float(ent[np.asarray(spans[i]) - 1].mean()))
            return {"cmean": means, "clast": before, "ent": entropies}
        batch = ctx.cached(f"l_feature_batch_{start}", collect)
        cm.append(batch["cmean"])
        cl.append(batch["clast"])
        entropy.extend(batch["ent"])
    data = {"ids": [r["id"] for r in rows], "split": [r["split"] for r in rows], "cls": [r["cls"] for r in rows],
            "cmean": torch.cat(cm), "clast": torch.cat(cl), "ent": np.asarray(entropy, np.float32),
            "neuron_ids": list(range(E.layout.total)), "layout": E.layout.as_dict(), "storage_dtype": "float16 (inherited L protocol)"}
    del cm, cl
    refs = loadpt(paths[2])
    rw = {"mu_z": refs["z_mu"], "sd_z": refs["z_sd"], "mu_cett": refs["c_mu"], "sd_cett": refs["c_sd"]}
    for key, value in rw.items():
        value = torch.as_tensor(value)
        if value.ndim != 1 or value.numel() != E.layout.total or not torch.isfinite(value).all():
            raise ValueError(f"SO-HN {key} must be a finite flat native-neuron reference vector")
        rw[key] = value.float().clamp_min(1e-6) if key.startswith("sd_") else value.float()
    ctx.save_pt("refw.pt", rw, config={"source": "own Gemma SO-HN REF-W (L primary reference)",
                                       "definition": "K: 64 validation windows x128 tokens, predictive positions16-126",
                                       "n_tokens": refs.get("n_refw"), "mock": bool(ctx.mock)})
    ctx.save_pt("cett_rows.pt", data)
    ctx.write_json("items.json", clean({"items": rows}))
    pop = ctx.cached("l_population", lambda: _population(ctx, E, H, data, rows))
    ctx.write_json("l_pop.json", clean(pop))
    result = {"status": "completed" if pop["status"] == "measured" else "blocked", "n_rows": len(rows),
              "population_status": pop["status"], "mock": bool(ctx.mock), "labels": pop.get("labels", []),
              "scientific_verdict": "not issued by population preparation"}
    ctx.write_json("l_prepare.json", result)
    return result


def rotated_features(ctx):
    """Exact per-layer QR basis, streaming one ragged rotation at a time.

    Holding every square MLP rotation on the GPU exceeds the 31B memory budget.
    A layer's answer features are checkpointed before its rotation is released.
    Prefix forwards stop immediately after that layer's down projection; future
    layers and the vocabulary head cannot affect its captured CETT.
    """
    import torch
    import r1b_g_adapter as adapter
    from r1b_g_engine import ModelEngine
    from r1b_g_l_locus import frag_losses, _uniform
    E = ModelEngine(ctx)
    _, y, folds, ids, _ = _sl(ctx)
    source = {r["id"]: r for r in _read_json(_path(ctx, "items.json"))["items"]}
    rows = []
    for qid in ids:
        row = source[qid]
        prompt = row.get("prompt_ids") or E.encode(SHORT_QA.format(q=row["question"]))
        a, b = row["span0"]
        rows.append((prompt + row["ans0"], list(range(len(prompt) + a, len(prompt) + b))))

    class Captured(Exception):
        pass

    layer_features = []
    for l, layer in enumerate(adapter.layers_of(E.model)):
        def collect_layer(l=l, layer=layer):
            w = E.layout.widths[l]
            generator = torch.Generator().manual_seed(int(rng("r1bL1rot", l).integers(1 << 62)))
            Q, R = torch.linalg.qr(torch.randn(w, w, generator=generator, dtype=torch.float64))
            signs = torch.sign(torch.diagonal(R))
            del R
            Q.mul_(signs[None])
            Q = Q.float().to(E.device)
            with torch.no_grad():
                norms = (layer.mlp.down_proj.weight.float() @ Q.T).norm(dim=0)
            features = []
            for seq, positions in rows:
                captured = []
                def hook(module, inp, result):
                    z = inp[0][0, positions].float()
                    denom = result[0, positions].float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    captured.append(((z @ Q.T).abs() * norms / denom).mean(0).cpu())
                    raise Captured()
                handle = layer.mlp.down_proj.register_forward_hook(hook)
                try:
                    with torch.no_grad():
                        adapter.text_model(E.model)(input_ids=torch.tensor([seq], device=E.device), use_cache=False)
                except Captured:
                    pass
                finally:
                    handle.remove()
                if len(captured) != 1:
                    raise RuntimeError("rotated-basis layer capture did not execute exactly once")
                features.append(captured[0])
            return torch.stack(features)
        layer_features.append(ctx.cached(f"l_rotated_layer_{l}", collect_layer))
    Xr = torch.cat(layer_features, dim=1).to(dev(E.model))
    cf = config(ctx)
    grid = sorted(set([m for m in cf["pip_grid"] if m <= Xr.shape[1]] + [len(load_pop(ctx)["P"])]))
    g = rng("r1bL1", "rot")
    fragments = [f for m in grid for f in _uniform(g, Xr.shape[1], m, cf["pip_frag"])]
    loss = frag_losses(ctx, "rotfam", Xr, y, folds, fragments)
    ctx.save_pt("l1_rot.pt", {"grid": grid, "Rot": loss.view(len(grid), cf["pip_frag"], -1)})


def measure(ctx):
    """Run measurements independently; record precise remaining branch gates."""
    from r1b_g_l_locus import l1_pip, l1_resel, l1_cal, l1_stats
    from r1b_g_l_long import long_context
    from r1b_g_l_erasure import erasure
    ctx = GuardedContext(ctx)
    blocked = _missing(ctx, "l_measure.json", [_path(ctx, n) for n in ("cett_rows.pt", "l_pop.json", "items.json")])
    coverage = {}
    if not blocked and load_pop(ctx).get("status") == "measured":
        X, y, _, _, d = _sl(ctx)
        fit = np.flatnonzero((np.asarray(d["split"]) == "S_fit") & np.isin(d["cls"], ("CC", "CI")))
        if len(y) >= 10 and min(int((y == 0).sum()), int((y == 1).sum())) >= 5 and len(fit) >= 4:
            for fn in (l1_pip, rotated_features, l1_resel, l1_cal, l1_stats):
                fn(ctx)
            coverage["L1_statistical"] = {"status": "measured", "controls": "PIP, floors, synergy, pre-answer, QR basis, reselection, planted profiles"}
        else:
            coverage["L1_statistical"] = {"status": "blocked", "reason": "S_L needs at least 5 measured CC and 5 CI independent questions for five folds"}
    else:
        coverage["L1_statistical"] = {"status": "blocked", "reason": "own SO-LAB/population unavailable"}
    coverage["L1_erasure"] = erasure(ctx) if not blocked and load_pop(ctx).get("status") == "measured" else {
        "status": "blocked", "reason": "own SO-LAB/population unavailable"}
    coverage["L2a"] = long_context(ctx)
    complete = all(record["status"] == "measured" for record in coverage.values())
    result = {"status": "done" if complete else "partial", "mock": bool(ctx.mock), "coverage": coverage,
              "measurement_scope": "L1 and L2a; L2b-L3, L4 and L5 have independent stages", "no_qwen_inputs": True}
    ctx.write_json("l_measure.json", result)
    return result


def calibrate(ctx):
    """Own resumable L-CAL (a)-(n); no target model or previous Qwen run."""
    import r1b_core as K
    import r1b_l_cal as calibration
    ctx = GuardedContext(ctx)
    reps = dict(a=5, b=20, c=400, d=20, e=5, f=5, g=5, h=20, i=20, j=20, k=20, l=200, m=5, n=20)
    items = {}
    for name, fn in calibration.ITEMS.items():
        count = 1 if ctx.mock else reps[name]
        measured = dict(ctx.cached(f"l_cal_{name}", lambda fn=fn, count=count: fn(bool(ctx.mock), count)))
        success = measured.pop("pass_")
        measured["pass"] = None if ctx.mock else bool(success)
        measured["repetitions"] = count
        items[name] = measured
    groups = {"L1": "a", "L2a": "f", "L2b": "g", "L2c": "j", "L3a": "bcd", "L3c": "e", "L3d": "hi", "L4": "kl", "L5a": "m", "L5c": "n"}
    gates = {name: all(items[k]["pass"] is True for k in keys) for name, keys in groups.items()}
    result = {"items": items, "quick": bool(ctx.mock), "gates": gates,
              "code_sha256": K.sha256_file(calibration.__file__), "stats_sha256": K.sha256_file(S.__file__),
              "scientific_evidence": not ctx.mock}
    ctx.write_json("l_cal.json", clean(result))
    return {"status": "done", "mock": bool(ctx.mock), "gates": gates, "coverage": list(items)}


def analyze(ctx):
    """Apply unchanged L verdict rules only to available, calibrated evidence.

    Each branch is independent: a missing population or closed chain-lens gate
    does not suppress the long-context, duet, or owner records that did run.
    """
    import r1b_l_rules as rules

    def artifact(stage, name):
        path = ctx.artifact(stage, name)
        return _read_json(path) if Path(path).is_file() else None

    measurement = artifact("l_measure", "l_measure.json") or {"coverage": {}}
    cal = artifact("l_calibrate", "l_cal.json") or {"gates": {}}
    gates = cal.get("gates", {})
    pop = artifact("l_prepare", "l_pop.json") or {}
    k3 = artifact("k_analyze", "k3_verdicts.json") or {}
    sg8 = k3.get("SG8") is True
    verdicts, coverage, statistics = {}, dict(measurement.get("coverage", {})), {}
    required = []

    def apply(name, stage, filename, fn):
        data = artifact(stage, filename)
        required.append(data is not None)
        if data is None:
            verdicts[name] = {"verdict": "not testable (measurement unavailable)", "missing": f"{stage}/{filename}"}
            coverage[name] = {"status": "blocked", "reason": "measurement artifact unavailable"}
            return
        statistics[name] = data
        try:
            verdicts[name] = fn(data)
            coverage[name] = {"status": "measured", "artifact": f"{stage}/{filename}"}
        except (KeyError, TypeError, ValueError) as exc:
            verdicts[name] = {"verdict": "not testable (incomplete statistic)", "reason": str(exc)}
            coverage[name] = {"status": "blocked", "reason": str(exc)}
            required[-1] = False

    apply("L1", "l_measure", "l1.json", lambda st: rules.rule_L1(
        st["tree_input"], cal_ok=bool(st["cal_ok"] and gates.get("L1")), m_cal=st.get("m_cal"), sg8=sg8))
    apply("L1_erasure", "l_measure", "l1_erasure.json", lambda st: rules.rule_L1_erasure(
        st, dmg_ok=bool(st["g_dmg"] and st["diff_ci_wrong"] and st["diff_fab_nonref"])))
    apply("L2a", "l_measure", "l2a.json", lambda st: rules.rule_L2a(st) if gates.get("L2a") else {"growth": rules.NT_CAL, "own": rules.NT_CAL})
    apply("L2b", "l_dynamics", "l2b.json", lambda st: rules.rule_L2b(st, sg8=sg8, cal_ok=bool(gates.get("L2b"))))
    apply("L2c", "l_dynamics", "l2c.json", lambda st: rules.rule_L2c(st, sg8=sg8, cal_ok=bool(gates.get("L2c"))))
    apply("L3a", "l_dynamics", "l3a.json", lambda st: rules.rule_L3a(st, sg8=sg8, cal_ok=bool(gates.get("L3a"))))
    apply("L3b", "l_dynamics", "l3b.json", lambda st: rules.rule_L3b(st, sg8=sg8, dmg_ok=tuple(st["dmg_ok"])))
    apply("L3c", "l_dynamics", "l3c.json", lambda st: rules.rule_L3c(
        st, sg8=sg8, cal_ok=bool(gates.get("L3c") and st["g_cal_ising"]["pass"]), gsel=bool(pop.get("G_sel"))))
    apply("L3d", "l_dynamics", "l3d.json", lambda st: rules.rule_L3d(st, sg8=sg8, cal_ok=bool(gates.get("L3d"))))
    def l4_rule(st):
        result = rules.rule_L4(st) if gates.get("L4") else {
            key: rules.NT_CAL for key in ("pair", "collaboration", "entity", "h_line", "catfish", "macro")}
        if not sg8:
            result["h_line"] = "not testable (no H-set)"
        return result
    apply("L4", "l_duet", "l4.json", l4_rule)
    owner = artifact("l_owner", "l5_verdicts.json")
    required.append(owner is not None)
    if owner is not None:
        verdicts.update(owner.get("verdicts", {}))
        combined = owner.get("combined", "not testable (combined owner verdict unavailable)")
        verdicts["V-OWN"] = combined.get("V-OWN", combined) if isinstance(combined, dict) else combined
        coverage.update(owner.get("coverage", {}))
        owner_statistics = artifact("l_owner", "l5_stats.json") or {}
        statistics.update({k: owner_statistics.get(k) for k in ("L5a", "L5b", "L5c")})
    else:
        for key in ("L5a", "L5b", "L5c"):
            verdicts[key] = {"verdict": "not testable (owner measurement unavailable)"}
            coverage[key] = {"status": "blocked", "reason": "l_owner/l5_verdicts.json unavailable"}
    for stage, filename in (("l_dynamics", "l_dynamics.json"), ("l_duet", "l4_coverage.json"), ("l_owner", "l_owner.json")):
        record = artifact(stage, filename)
        if record and isinstance(record.get("coverage"), dict):
            coverage.update(record["coverage"])
    # The inherited pure rules test their frozen equivalence margins. L.2 also
    # requires sufficient realized precision; apply that additional rule to
    # equivalence conclusions and retain the original result for audit.
    from r1b_g_l_power import guard
    unguarded_verdicts = verdicts
    verdicts, power_checks = guard(verdicts, statistics)
    if owner is not None:
        verdicts["V-OWN"] = rules.v_own(
            verdicts.get("L5a", {}).get("verdict", "not testable"),
            verdicts.get("L5b", {}).get("verdict", "not testable"))
    if "L3a" in verdicts and "L3b" in verdicts:
        verdicts["attractor"] = rules.attractor_line(verdicts["L3a"], verdicts["L3b"])
    complete = all(required) and not any(isinstance(v, dict) and v.get("status") == "blocked" for v in coverage.values())
    result = {"status": "done" if complete else "partial", "mock": bool(ctx.mock),
              "measurements_complete": complete, "scientific_complete": bool(complete and not ctx.mock and all(gates.values()) and gates),
              "coverage": coverage, "verdicts": verdicts, "pipeline_calibration_gates": gates, "sg8": sg8,
              "equivalence_power_checks": power_checks,
              "verdicts_before_power_guard": unguarded_verdicts,
              "CF_HERR_member": owner.get("CF_HERR_member") if owner else None,
              "interpretation": "MOCK: software checks only" if ctx.mock else "Scientific verdicts require their complete measurement and calibration gates"}
    ctx.write_json("l_analysis.json", clean(result))
    return result
