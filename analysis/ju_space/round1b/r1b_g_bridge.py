"""12B base-catalogue -> 12B IT-QAT bridge, independent of Qwen runs.

Catalogue descriptions remain external base-model annotations. Coordinate joins
never imply semantic equality. Held-out activation and forward-intervention
checks are reported separately, with matched controls and source attribution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CATALOGUE_REPO = "stdlogits/neuron-lens"
CATALOGUE_REVISION = "21666b9f217328caee73d10de85ca19f4c174e72"
SOURCE_MODEL = "google/gemma-4-12B"
SOURCE_REVISION_FOR_VOCAB = "1a2c7729392586f5ecddae8807ee2ef741eabb9a"
SOURCE_TOKENIZER_SHA256 = "42cacba08980fc39bfe2c73cb8c4486b9d00f2f2671072c6cb9c45ee2c5cfaae"
ANNOTATION_LABEL = "external base-model annotation; transfer unvalidated"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _write(ctx, result):
    result.update(model=ctx.model_name, mock=bool(ctx.mock), annotation_label=ANNOTATION_LABEL,
                  source_model=SOURCE_MODEL, source_generating_model_revision="not pinned by catalogue",
                  attribution={"creator": "stdlogits", "dataset": CATALOGUE_REPO,
                               "revision": CATALOGUE_REVISION, "licence": "CC-BY-4.0",
                               "url": f"https://huggingface.co/datasets/{CATALOGUE_REPO}/tree/{CATALOGUE_REVISION}",
                               "corpus_contexts": "retain their respective source licences"})
    ctx.write_json("bridge.json", result)
    return result


def _coordinate(value, widths):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("neuron coordinates must be [layer, neuron]")
    l, n = value
    if type(l) is not int or type(n) is not int or not 0 <= l < len(widths) or not 0 <= n < widths[l]:
        raise ValueError(f"out-of-range neuron coordinate: {value}")
    return l, n


def matched_coordinates(hn, widths, mock=False):
    """Align each RS set to H by per-layer order, retaining every supplied set."""
    h = sorted(_coordinate(x, widths) for x in hn.get("H_op", []))
    if not h or len(h) != len(set(h)):
        raise ValueError("bridge needs a nonempty, unique H_op set")
    controls = hn.get("RS39") or []
    if len(controls) < (1 if mock else 39):
        raise ValueError("bridge needs 39 independently selected layer-matched controls (one in mocks)")
    mapped = []
    for rs in controls:
        c = sorted(_coordinate(x, widths) for x in rs)
        if len(c) != len(h) or [x[0] for x in c] != [x[0] for x in h] or len(c) != len(set(c)):
            raise ValueError("bridge control set differs in H size or layer counts")
        if set(h) & set(c):
            raise ValueError("bridge control neurons must exclude H")
        mapped.append(dict(zip(h, c)))
    return h, mapped


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"code": "NaN:external_catalogue_nonfinite"}
    return value


def load_catalogue(config, coordinates, *, mock=False):
    """Load only requested coordinates, from a labelled local extract or pinned HF tables."""
    if not isinstance(config, dict):
        raise ValueError("neuron_catalogue configuration is missing")
    if bool(config.get("mock")) != bool(mock):
        raise ValueError("mock/real catalogue identity differs from run")
    if not mock and (config.get("revision", CATALOGUE_REVISION) != CATALOGUE_REVISION
                     or config.get("source_model", SOURCE_MODEL) != SOURCE_MODEL):
        raise ValueError("catalogue source identity is not the reviewed 12B base model")
    wanted = {(l, n, p) for l, n in coordinates for p in (-1, 1)}
    records, source_files, missing_tables = {}, [], []
    local = config.get("path")
    if local and Path(local).is_file() and Path(local).suffix == ".json":
        path = Path(local)
        blob = json.loads(path.read_text())
        if not isinstance(blob, dict) or bool(blob.get("mock")) != bool(mock):
            raise ValueError("local catalogue extract must declare mock identity")
        if not mock and (blob.get("revision") != CATALOGUE_REVISION or blob.get("source_model") != SOURCE_MODEL):
            raise ValueError("local catalogue extract lacks reviewed source identity")
        for row in blob.get("rows", []):
            key = (int(row["layer"]), int(row["neuron"]), int(row["pole"]))
            if key in records:
                raise ValueError(f"duplicate catalogue coordinate: {key}")
            if key in wanted:
                records[key] = _jsonable(row)
        source_files.append({"path": str(path), "sha256": sha256(path),
                             "kind": "mock fixture" if mock else "local user-supplied catalogue extract",
                             "source_integrity": "synthetic" if mock else "revision asserted by extract; extraction not independently verified"})
    else:
        if mock:
            raise ValueError("mock bridge requires an explicit local JSON fixture, never HF data")
        if not local and not config.get("download"):
            raise ValueError("supply a local catalogue directory or enable its pinned download")
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ValueError("catalogue Parquet loading requires pyarrow") from exc
        for layer in sorted({l for l, _ in coordinates}):
            for table in ("labels", "reads"):
                relative = f"gemma-4-12b/neurons/{table}/L{layer:02d}.parquet"
                if local:
                    path = Path(local) / relative
                else:
                    from huggingface_hub import hf_hub_download
                    path = Path(hf_hub_download(CATALOGUE_REPO, relative, repo_type="dataset", revision=CATALOGUE_REVISION))
                if not path.is_file():
                    raise ValueError(f"missing requested catalogue layer table: {relative}")
                source_files.append({"path": str(path), "dataset_path": relative, "sha256": sha256(path),
                                     "source_integrity": "local directory; revision asserted" if local else "downloaded from pinned HF dataset revision"})
                found = set()
                for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
                    for row in batch.to_pylist():
                        key = (int(row["layer"]), int(row["neuron"]), int(row["pole"]))
                        if key in wanted:
                            if key in found:
                                raise ValueError(f"duplicate {table} catalogue coordinate: {key}")
                            found.add(key)
                            records.setdefault(key, {"layer": key[0], "neuron": key[1], "pole": key[2]})[table] = _jsonable(row)
                missing_tables.extend({"table": table, "coordinate": list(key)}
                                      for key in sorted(wanted) if key[0] == layer and key not in found)
    missing = sorted(wanted - records.keys())
    return records, {"files": source_files, "missing": [list(k) for k in missing], "requested_rows": len(wanted),
                     "loaded_rows": len(records), "missing_tables": missing_tables, "mock": bool(mock)}


def vocabulary_check(config, tokenizer, *, mock=False):
    if mock:
        return {"passed": True, "kind": "synthetic fixture; no real vocabulary equivalence claim"}
    path = config.get("source_tokenizer")
    if path is None and config.get("download"):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(SOURCE_MODEL, "tokenizer.json", revision=SOURCE_REVISION_FOR_VOCAB)
    if path is None:
        return {"passed": False, "reason": "source tokenizer unavailable; text annotations remain descriptive"}
    if sha256(path) != SOURCE_TOKENIZER_SHA256:
        raise ValueError("base-model tokenizer SHA-256 differs from the reviewed vocabulary snapshot")
    from tokenizers import Tokenizer
    source = Tokenizer.from_file(str(path)).get_vocab(with_added_tokens=True)
    target = tokenizer.get_vocab()
    mismatch = sum(target.get(token) != idx for token, idx in source.items())
    return {"passed": mismatch == 0 and len(source) == len(target), "source_size": len(source), "target_size": len(target),
            "mismatched_tokens": mismatch, "source_model_revision": SOURCE_REVISION_FOR_VOCAB,
            "source_tokenizer_sha256": SOURCE_TOKENIZER_SHA256,
            "note": "checked base snapshot is not guaranteed to be the catalogue's generating revision"}


def validate_probes(rows, h, discovery_ids=()):
    if not isinstance(rows, list) or not rows:
        raise ValueError("bridge_probes are missing; supply independent held-out semantic probes")
    seen, discovery, available = set(), set(discovery_ids), set(h)
    labels, texts = {}, set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("bridge probes must be records")
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in seen or row["id"] in discovery:
            raise ValueError("probe ids must be unique and disjoint from discovery ids")
        seen.add(row["id"])
        if row.get("split") != "heldout":
            raise ValueError("bridge probes must explicitly declare split='heldout'")
        pair = (row.get("layer"), row.get("neuron"))
        if pair not in available or row.get("pole") not in (-1, 1):
            raise ValueError("probe coordinate/pole is outside independently selected H")
        if type(row.get("expected_active")) is not bool:
            raise ValueError("probe expected_active must be an external boolean annotation")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("probe text is empty")
        identity = (*pair, row["pole"], row["text"].strip())
        if identity in texts:
            raise ValueError("probe texts must be unique within each neuron/pole")
        texts.add(identity)
        if not isinstance(row.get("expected_token"), str) or not row["expected_token"]:
            raise ValueError("bridge probes need an externally specified expected_token for the intervention check")
        labels.setdefault((*pair, row["pole"]), set()).add(row["expected_active"])
    if any(x != {True, False} for x in labels.values()):
        raise ValueError("each probed pole needs both positive and negative held-out examples")
    if set(labels) != {(l, n, p) for l, n in h for p in (-1, 1)}:
        raise ValueError("held-out probes must cover both poles of every selected H neuron")
    return rows


def probe_provenance(data, rows, *, mock=False):
    """Require a frozen source-disjoint evaluation declaration, not inferred labels.

    Human/external provenance cannot be established from a model response. The
    input manifest preserves that assertion, and hashes enforce the declared
    discovery/evaluation text separation wherever the source text is supplied.
    """
    manifest = data.get("bridge_probe_manifest")
    if mock:
        return {"passed": True, "kind": "synthetic fixture only"}
    if (not isinstance(manifest, dict) or not isinstance(manifest.get("source"), str)
            or not manifest["source"].strip() or manifest.get("frozen_before_evaluation") is not True
            or manifest.get("label_generation_contexts_excluded") is not True):
        raise ValueError("bridge_probe_manifest must record source, frozen_before_evaluation=true and label_generation_contexts_excluded=true")
    excluded = manifest.get("discovery_text_sha256")
    if not isinstance(excluded, list) or any(not isinstance(s, str) or len(s) != 64
                                           or any(c not in "0123456789abcdef" for c in s) for s in excluded):
        raise ValueError("bridge_probe_manifest needs discovery_text_sha256 (an explicit list, possibly empty)")
    evaluation = {hashlib.sha256(r["text"].strip().encode()).hexdigest() for r in rows}
    if evaluation & set(excluded):
        raise ValueError("held-out probe text overlaps the declared discovery text hashes")
    return {"passed": True, "manifest": manifest, "evaluation_text_sha256": sorted(evaluation),
            "note": "external source-disjointness declaration; verified against supplied discovery hashes"}


def _probe(model, tok, row, coordinate, dose):
    import torch
    import r1b_core as core
    from r1b_g_adapter import chat_text, text_model, text_config, unembed
    marker = "R1B_G_BRIDGE_BOUNDARY_b137f"
    template = chat_text(tok, marker)
    startchar = template.index(marker)
    rendered = chat_text(tok, row["text"])
    encoded = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    positions = [i for i, (begin, end) in enumerate(encoded["offset_mapping"])
                 if end > startchar and begin < startchar + len(row["text"])]
    if not positions:
        raise ValueError("bridge probe has no user-content token positions")
    if len(ids) > text_config(model).max_position_embeddings:
        raise ValueError("bridge probe exceeds the native maximum context; no truncation permitted")
    target = tok(row["expected_token"], add_special_tokens=False)["input_ids"]
    if len(target) != 1:
        raise ValueError("bridge expected_token must be one token in the target tokenizer")
    dev = next(model.parameters()).device
    x = torch.tensor([ids], device=dev)
    l, n = coordinate
    with torch.no_grad(), core.Hooks(model).neuron(l, [n], "read", name="z") as hook:
        hidden = text_model(model)(input_ids=x, use_cache=False, logits_to_keep=0).last_hidden_state[:, -1:]
        clean = unembed(model, hidden, normalized=True)[0, -1].float().log_softmax(-1)
        z = hook.reads["z"][0]["x"][0, positions, 0]
        activation = float(row["pole"] * z.mean())
    with torch.no_grad(), core.Hooks(model).neuron(l, [n], "add", row["pole"] * dose,
                                                  rowstart=torch.tensor([len(ids) - 1])):
        altered = model(x, logits_to_keep=1, use_cache=False).logits[0, -1].float().log_softmax(-1)
    return {"activation": activation, "expected_token_id": target[0],
            "delta_logp": float(altered[target[0]] - clean[target[0]]), "n_tokens": len(ids),
            "user_positions": positions, "intervention_position": len(ids) - 1,
            "input_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}


def transfer_summary(records, seed=7782, nboot=2000):
    from r1b_k_base import auroc
    import r1b_core as core
    groups = {}
    for r in records:
        groups.setdefault((r["layer"], r["neuron"], r["pole"]), []).append(r)
    summary = []
    for key, rows in groups.items():
        y = np.asarray([r["expected_active"] for r in rows], dtype=int)
        a = np.asarray([r["target"]["activation"] for r in rows])
        effects = np.asarray([r["target"]["delta_logp"] - np.mean([v["delta_logp"] for v in r["controls"]]) for r in rows])
        control_a = np.asarray([[v["activation"] for v in r["controls"]] for r in rows])
        rng = np.random.default_rng(seed + key[0] * 100000 + key[1] * 2 + (key[2] > 0))
        draws = effects[rng.integers(0, len(effects), (nboot, len(effects)))].mean(1)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        bp = pos[rng.integers(0, len(pos), (nboot, len(pos)))]
        bn = neg[rng.integers(0, len(neg), (nboot, len(neg)))]
        enrichment = a[bp].mean(1) - a[bn].mean(1)
        auc_draws = [auroc(y[np.r_[p, n]], a[np.r_[p, n]]) for p, n in zip(bp, bn)]
        summary.append({"layer": key[0], "neuron": key[1], "pole": key[2], "n_probes": len(rows),
                        "activation_enrichment": float(a[y == 1].mean() - a[y == 0].mean()),
                        "activation_enrichment_interval90": list(core.interval(enrichment)),
                        "activation_auroc": float(auroc(y, a)),
                        "activation_auroc_interval90": list(core.interval(auc_draws)),
                        "control_activation_enrichments": (control_a[y == 1].mean(0) - control_a[y == 0].mean(0)).tolist(),
                        "control_activation_aurocs": [float(auroc(y, control_a[:, j])) for j in range(control_a.shape[1])],
                        "mean_intervention_minus_control_delta_logp": float(effects.mean()),
                        "intervention_interval90": list(core.interval(draws)),
                        "intervention_positive_fraction": float(np.mean([r["target"]["delta_logp"] > 0 for r in rows])),
                        "bootstrap_unit": "held-out text within neuron/pole; activation resampling stratified by expected_active",
                        "bootstrap_seed": seed + key[0] * 100000 + key[1] * 2 + (key[2] > 0),
                        "bootstrap_replicates": nboot,
                        "interpretation": "descriptive; uncalibrated fixed activation-unit dose, no pooled causal specificity verdict"})
    return summary


def run(ctx):
    from r1b_g_adapter import NeuronLayout
    if ctx.model_name != "gemma4-12b-qat":
        return _write(ctx, {"status": "blocked", "reason": "catalogue bridge is defined only for gemma4-12b-qat"})
    data = ctx.data()
    path = Path(ctx.artifact("k_prepare", "hn.json"))
    if not path.is_file():
        return _write(ctx, {"status": "blocked", "reason": "missing G H-selection artifact k_prepare/hn.json"})
    hn = json.loads(path.read_text())
    if bool(hn.get("mock")) != bool(ctx.mock):
        return _write(ctx, {"status": "blocked", "reason": "H-selection mock/real identity mismatch"})
    tok, model = ctx.model()
    layout = NeuronLayout.from_model(model)
    if not ctx.mock and layout.widths != (15360,) * 48:
        return _write(ctx, {"status": "blocked", "reason": "12B target coordinate layout differs from catalogue"})
    if hn.get("layout") is not None and hn["layout"] != layout.as_dict():
        return _write(ctx, {"status": "blocked", "reason": "H-selection layout differs from the loaded target"})
    if not ctx.mock and hn.get("layout") is None:
        return _write(ctx, {"status": "blocked", "reason": "H-selection artifact is missing its coordinate layout"})
    try:
        h, controls = matched_coordinates(hn, layout.widths, ctx.mock)
        coordinates = set(h) | {v for control in controls for v in control.values()}
        catalogue, source = load_catalogue(data.get("neuron_catalogue"), coordinates, mock=ctx.mock)
        vocab = vocabulary_check(data["neuron_catalogue"], tok, mock=ctx.mock)
    except (ValueError, OSError) as exc:
        return _write(ctx, {"status": "blocked", "reason": str(exc)})
    joined = [{"layer": k[0], "neuron": k[1], "pole": k[2], "is_H": k[:2] in h,
               "annotation": v} for k, v in sorted(catalogue.items())]
    common = {"catalogue": source, "joined": joined, "vocabulary_check": vocab,
              "layout": layout.as_dict(), "hn_sha256": sha256(path), "H_op_name": hn.get("H_op_name"),
              "H_selection_gates": hn.get("gates"), "semantic_equivalence_claim": False,
              "checks": {"dimensions": {"passed": True}, "tokenization": vocab,
                         "activation": {"status": "not_run"}, "intervention": {"status": "not_run"}}}
    if source["missing"] or source["missing_tables"]:
        return _write(ctx, {"status": "blocked", "reason": "catalogue lacks selected H/control coordinates", **common})
    try:
        probes = validate_probes(data.get("bridge_probes"), h, data.get("bridge_discovery_ids", []))
        common["probe_provenance"] = probe_provenance(data, probes, mock=ctx.mock)
    except ValueError as exc:
        return _write(ctx, {"status": "blocked", "reason": str(exc), **common})
    dose = float(((ctx.freeze or {}).get("g") or {}).get("bridge_dose", 1.0))
    if not np.isfinite(dose) or dose <= 0:
        raise ValueError("bridge_dose must be positive and finite")
    records = []
    try:
        for row in probes:
            if hasattr(ctx, "time_left") and ctx.time_left() < 30:
                from r1b_core import StepYield
                raise StepYield()
            def compute(row=row):
                def evaluate(coordinate):
                    if hasattr(ctx, "time_left") and ctx.time_left() < 30:
                        from r1b_core import StepYield
                        raise StepYield()
                    key = hashlib.sha256(json.dumps({"probe": row, "coordinate": coordinate, "dose": dose},
                                                    sort_keys=True).encode()).hexdigest()
                    fn = lambda: _probe(model, tok, row, coordinate, dose)
                    return ctx.cached("bridge_forward_" + key, fn) if hasattr(ctx, "cached") else fn()
                coordinate = row["layer"], row["neuron"]
                target = evaluate(coordinate)
                matched = [evaluate(control[coordinate]) for control in controls]
                return {"id": row["id"], "layer": row["layer"], "neuron": row["neuron"],
                        "pole": row["pole"], "expected_active": row["expected_active"],
                        "target": target, "controls": matched}
            key = hashlib.sha256(json.dumps({"probe": row, "dose": dose, "controls": hn["RS39"]}, sort_keys=True).encode()).hexdigest()
            records.append(ctx.cached("bridge_" + key, compute) if hasattr(ctx, "cached") else compute())
    except ValueError as exc:
        return _write(ctx, {"status": "blocked", "reason": str(exc), "records": records, **common})
    common["checks"].update(activation={"status": "measured", "claim": "descriptive effect sizes"},
                            intervention={"status": "measured", "claim": "descriptive effect sizes; fixed uncalibrated dose"})
    return _write(ctx, {"status": "completed", **common, "records": records, "dose": dose,
                        "summary": transfer_summary(records, nboot=32 if ctx.mock else 2000),
                        "source_identity_gap": "instruction tuning and quantization; generating base revision unspecified",
                        "probes_sha256": hashlib.sha256(json.dumps(probes, sort_keys=True).encode()).hexdigest(),
                        "semantic_equivalence_claim": False})
