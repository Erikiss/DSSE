"""Run Gemma G independently of Qwen, with isolated records and resumable stages.

Real scientific runs require G's own successful pilot and freeze. A mock run
uses only tiny random models and never downloads model weights. A partial or
blocked scientific protocol is never reported as a complete experiment.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import r1b_core as core
from r1b_g_models import get_spec
from r1b_sessions_G import STAGES, TARGETS, ordered

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PLAN = REPO / "results/ju_space/plans/runde1b/arbeitsstand/plan/runde1b_plan_teil_G.md"
EXIT = {"done": 0, "mock_complete": 0, "failed": 1, "blocked": 2, "yielded": 3, "partial": 4}


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", value) or value in {".", ".."}:
        raise ValueError("experiment/model identifiers must be plain names, not paths")
    return value


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(encoded)
    os.replace(tmp, path)


def commit():
    p = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else "unavailable"


def source_hashes():
    # Includes imported scientific definitions without requiring their results.
    files = set(HERE.glob("r1b_*.py")) | set(HERE.parent.glob("*.py")) | set((HERE.parent / "round2").glob("*.py"))
    return {str(p.relative_to(REPO)): core.sha256_file(p) for p in sorted(files)}


def plan_hashes():
    files = [PLAN.parent / "runde1b_plan.md"] + [PLAN.with_name(f"runde1b_plan_teil_{p}.md") for p in "DKLG"]
    return {str(p.relative_to(REPO)): core.sha256_file(p) for p in files}


def checked_relative(root, relative):
    root = Path(root).resolve()
    p = (root / relative).resolve()
    if not p.is_relative_to(root) or Path(relative).is_absolute():
        raise ValueError(f"path leaves G run directory: {relative}")
    return p


class GCtx(core.Ctx):
    def __init__(self, *, model_name, stage, root, model_run, data, data_hash, mock, freeze=None,
                 model_cache=None, guard_s=4.5 * 3600):
        work = Path(model_run) / "stages" / stage
        work.mkdir(parents=True, exist_ok=True)
        super().__init__(f"r1b_G_{stage}_{model_name}", str(work), root=str(root),
                         device="cpu" if mock or stage in {"l_calibrate", "g_analyze"} else "cuda", mock=mock, quick=mock, pilot=False,
                         freeze=freeze or {}, guard_s=guard_s, step=stage,
                         manifest={"plan": {"g_sha256": core.sha256_file(PLAN)},
                                   "code_commit": commit(), "inputs": [{"sha256": data_hash}]})
        self.part, self.tier, self.account = "G", 1, "independent"
        self.model_name, self.stage = model_name, stage
        self.model_run = Path(model_run)
        self._data, self.data_hash = data, data_hash
        self._model_cache = model_cache if model_cache is not None else {}
        self.outputs = []

    def data(self):
        return self._data

    def model(self, name=None):
        from r1b_g_adapter import load_model
        name = name or self.model_name
        if name not in self._model_cache:
            self._model_cache[name] = load_model(name, root=self.root, device=self.device,
                                                mock=self.mock, freeze=self.freeze, download=not self.mock)
        self.models_used[name] = {"mock": self.mock, **{k: get_spec(name)[k] for k in ("repo", "revision", "sha256")}}
        return self._model_cache[name]

    def key(self, tag, *parts):
        key = "r1bG:" + self.model_name + ":" + str(tag)
        self.keys_used.add(key)
        return core.keyseed(key, *parts)

    def path(self, relative):
        p = checked_relative(self.work, relative)
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    def artifact(self, stage, relative=None):
        if relative is None:
            return Path(self.path(stage))
        if stage not in STAGES:
            raise ValueError(f"unknown G artifact stage: {stage}")
        return checked_relative(self.model_run / "stages" / stage, relative)

    def input(self, stage, relative, required=True):
        p = self.artifact(stage, relative)
        if p.is_file():
            return str(p)
        if required:
            raise core.NotRun(f"missing G input {stage}/{relative}")
        return None

    def provenance(self, config=None, seeds=None, labels=None):
        p = super().provenance(config, seeds, labels)
        p.update(g_experiment=str(self.model_run.parent.name), g_model=self.model_name,
                 g_stage=self.stage, mock=self.mock, scientific_evidence=not self.mock,
                 g_input_sha256=self.data_hash, independent_of_qwen_runs=True)
        return p

    def write_json(self, rel, obj, **kwargs):
        out = super().write_json(rel, obj, **kwargs)
        self.outputs.append(rel)
        return out

    def save_pt(self, rel, obj, **kwargs):
        out = super().save_pt(rel, obj, **kwargs)
        self.outputs.append(rel)
        return out

    def write_file(self, rel, obj, **kwargs):
        out = super().write_file(rel, obj, **kwargs)
        self.outputs.append(rel)
        return out


def model_directory(runs, experiment, model):
    if model not in TARGETS:
        raise ValueError(f"not a G experiment target: {model}")
    return Path(runs).resolve() / safe_name(experiment) / model


def hardware_check():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Real G runs require an A100 GPU with 80 GB; use --mode mock for CPU.")
    name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory
    if "A100" not in name or memory < 79_000_000_000:
        raise RuntimeError(f"G requires A100-80GB before downloading weights; found {name}, {memory/1e9:.1f} GB.")
    return {"gpu": name, "memory_bytes": memory, "torch": torch.__version__}


def load_data(inputs, mock):
    from r1b_g_data import mock_data, normalize, pilot_data
    if inputs:
        raw = read_json(inputs)
        if raw is None:
            raise FileNotFoundError(inputs)
        data_hash = core.sha256_file(inputs)
    else:
        raw = mock_data() if mock else pilot_data()
        data_hash = hashlib.sha256(core.canon(raw).encode()).hexdigest()
    if bool(raw.get("mock")) and not mock:
        raise ValueError("a mock input package cannot be used for real scientific measurements")
    return normalize(raw), data_hash


def external_input_hashes(data):
    """Bind declared local bridge extracts and vocabulary files to the freeze."""
    config = data.get("neuron_catalogue") or {}
    result = {}
    for key in ("path", "source_tokenizer"):
        value = config.get(key)
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        if path.is_file():
            paths = [path]
        elif key == "path" and path.is_dir():
            paths = sorted(p for table in ("labels", "reads")
                           for p in (path / "gemma-4-12b/neurons" / table).glob("L*.parquet"))
        else:
            raise ValueError(f"declared local G catalogue input is unavailable: {path}")
        if not paths:
            raise ValueError(f"declared local G catalogue contains no layer tables: {path}")
        result[key] = {str(p): core.sha256_file(p) for p in paths}
    return result


def freeze_check(freeze_path, model, data_hash, data=None):
    fz = read_json(freeze_path) if freeze_path else None
    if not fz or fz.get("schema") != "dsse-g-freeze-v1" or fz.get("mock"):
        raise ValueError("Real scientific runs require their own G freeze; run the G pilot and freeze command first.")
    if fz["model"] != model or fz["input_sha256"] != data_hash:
        raise ValueError("G freeze model/input hash mismatch")
    if fz.get("plan_hashes") != plan_hashes() or fz["source_hashes"] != source_hashes():
        raise ValueError("G code or plan changed since freeze; use a new experiment and pilot")
    required_models = {model} | ({model + "-assistant"} if model in TARGETS[:2] else set())
    if set(fz.get("models", {})) != required_models:
        raise ValueError("G freeze lacks the exact target/assistant model manifest")
    if fz.get("external_input_hashes", {}) != external_input_hashes(data or {}):
        raise ValueError("Declared local G inputs changed since freeze")
    for name, pins in fz["models"].items():
        if pins != {k: get_spec(name)[k] for k in ("repo", "revision", "sha256")}:
            raise ValueError("G freeze model pins differ from the reviewed manifest")
    return fz


def output_manifest(ctx):
    result = {}
    for rel in sorted(set(ctx.outputs)):
        p = Path(ctx.path(rel))
        if not p.is_file():
            raise RuntimeError(f"declared output missing: {rel}")
        error = core.check_output(str(p))
        if error:
            raise RuntimeError(error)
        result[rel] = core.sha256_file(p)
        sidecar = Path(str(p) + ".prov.json")
        if sidecar.is_file():
            result[rel + ".prov.json"] = core.sha256_file(sidecar)
    return result


def resumable(record, signature, directory):
    if not record or record.get("signature") != signature or record.get("state") not in {"done", "partial", "not_applicable"}:
        return False
    return all((Path(directory) / p).is_file() and core.sha256_file(Path(directory) / p) == h
               and (p.endswith(".prov.json") or core.check_output(str(Path(directory) / p)) is None)
               for p, h in record.get("outputs", {}).items())


def stage_signature(name, model, mock, data, data_hash, freeze_path=None, dependencies=None):
    # Pilot reuse does not depend on the scientific input bank or the freeze it
    # will later justify. Its actual neutral prompts and code are still pinned.
    selected_hash = (hashlib.sha256(core.canon(data.get("pilot_texts", [])).encode()).hexdigest()
                     if name == "pilot" else "CPU known-answer calibration" if name == "l_calibrate" else data_hash)
    payload = {"sources": source_hashes(), "plans": plan_hashes(), "inputs": selected_hash,
               "mock": mock, "model": model, "stage": name,
               "freeze": core.sha256_file(freeze_path) if freeze_path and name not in {"pilot", "l_calibrate"} else None}
    if dependencies:
        payload["dependencies"] = dependencies
    if name == "bridge":
        payload["external_input_hashes"] = external_input_hashes(data)
    return hashlib.sha256(core.canon(payload).encode()).hexdigest()


def run(args):
    mock = args.mode == "mock"
    if not mock and args.stage != "l_calibrate":
        hardware_check()
    data, data_hash = load_data(args.inputs, mock)
    fz = {} if mock or args.stage in {"pilot", "l_calibrate"} else freeze_check(args.freeze, args.model, data_hash, data)
    model_run = model_directory(args.runs, args.experiment, args.model)
    model_run.mkdir(parents=True, exist_ok=True)
    meta = {"model": args.model, "mock": mock, "schema": "dsse-g-run-v1", "experiment": args.experiment,
            "source_hashes": source_hashes(), "plan_hashes": plan_hashes()}
    selected = ordered(args.stage)
    records = {}
    model_cache = {}
    started = time.monotonic()
    import fcntl
    with open(model_run / ".run.lock", "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError("this G model/experiment is already running") from e
        existing = read_json(model_run / "run.json")
        if existing and existing != meta:
            raise ValueError("This experiment/model has changed code/plans/mode; use a new experiment name.")
        write_json(model_run / "run.json", meta)
        if args.stage not in {"pilot", "l_calibrate"}:
            binding = {"input_sha256": data_hash}
            old_binding = read_json(model_run / "scientific_inputs.json")
            if old_binding is not None and old_binding != binding:
                raise ValueError("Scientific G inputs changed; use a new experiment")
            write_json(model_run / "scientific_inputs.json", binding)
        previous_summary = read_json(model_run / "summary.json", {})
        for name in selected:
            if source_hashes() != meta["source_hashes"] or plan_hashes() != meta["plan_hashes"]:
                raise ValueError("G code/plans changed during execution; retain this run and use a new experiment")
            spec = STAGES[name]
            if name == "bridge" and not mock and external_input_hashes(data) != fz.get("external_input_hashes", {}):
                raise ValueError("Declared local G inputs changed during execution")
            dependencies = {p: {k: records[p].get(k) for k in ("signature", "state", "outputs")}
                            for p in spec["after"] + spec["observes"] if p in records}
            signature = stage_signature(name, args.model, mock, data, data_hash, args.freeze, dependencies)
            rec_path = model_run / "state" / f"{name}.json"
            old = read_json(rec_path)
            stage_dir = model_run / "stages" / name
            if not args.rerun and resumable(old, signature, stage_dir):
                records[name] = old
                print(f"{name}: {old['state']} (verified resume)", flush=True)
                continue
            if old and old.get("signature") != signature:
                raise ValueError(f"{name}: inputs/code/freeze changed; use a new experiment rather than mixing artifacts")
            rec = {"stage": name, "signature": signature, "mock": mock, "started": core.iso(), "input_sha256": data_hash,
                   "dependencies": dependencies, "state": "running", "outputs": {}}
            # Bind partial caches to their code/input identity before invoking a
            # stage: a killed process must not leave reusable, unbound caches.
            write_json(rec_path, rec)
            if args.model not in spec["models"]:
                rec.update(state="not_applicable", reason="not assigned to this model by the G protocol")
            elif time.monotonic() - started >= args.guard_seconds:
                rec.update(state="yielded", reason="G runtime time guard")
            else:
                failed = [p for p in spec["after"] if records.get(p, {}).get("state") in {"failed", "blocked", "yielded"}]
                if failed:
                    rec.update(state="blocked", reason=f"G prerequisites unavailable: {', '.join(failed)}")
                else:
                    ctx = GCtx(model_name=args.model, stage=name, root=args.root, model_run=model_run,
                               data=data, data_hash=data_hash, mock=mock, freeze=fz, model_cache=model_cache,
                               guard_s=max(1, args.guard_seconds - (time.monotonic()-started)))
                    log = model_run / "logs" / f"{name}.txt"
                    log.parent.mkdir(exist_ok=True)
                    try:
                        mod, fn = spec["fn"].split(":")
                        function = getattr(importlib.import_module(mod), fn)
                        with open(log, "a") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                            result = function(ctx)
                        result = result if isinstance(result, dict) else {}
                        state = result.get("status", "done")
                        state = {"ok": "done", "completed": "done"}.get(state, state)
                        if state not in {"done", "partial", "blocked", "not_applicable"}:
                            raise ValueError(f"unknown stage result status: {state}")
                        rec.update(state=state, reason=result.get("reason") or "", coverage=core.jsonable(result.get("coverage", {})),
                                   outputs=output_manifest(ctx))
                        if not rec["outputs"] and state in {"done", "partial"}:
                            raise RuntimeError("G stage returned success without a provenance-checked artifact")
                    except core.StepYield:
                        rec.update(state="yielded", reason="stage saved continuation and yielded")
                    except core.NotRun as e:
                        rec.update(state="blocked", reason=str(e), outputs=output_manifest(ctx))
                    except ModuleNotFoundError as e:
                        rec.update(state="blocked", reason=f"G implementation unavailable: {e}")
                        with open(log, "a") as stream:
                            traceback.print_exc(file=stream)
                    except Exception as e:
                        rec.update(state="failed", reason=f"{type(e).__name__}: {e}")
                        with open(log, "a") as stream:
                            traceback.print_exc(file=stream)
            rec["finished"] = core.iso()
            write_json(rec_path, rec)
            records[name] = rec
            print(f"{name}: {rec['state']} {rec.get('reason','')}", flush=True)
            if rec["state"] == "yielded":
                break
    states = {r["state"] for r in records.values()}
    state = ("failed" if "failed" in states else "yielded" if "yielded" in states else
             "partial" if states & {"blocked", "partial"} else "mock_complete" if mock else "done")
    summary = {**meta, "state": state, "scientific_complete": not mock and state == "done" and args.stage == "all",
               "selected": selected, "stages": records, "code_commit": commit(), "input_sha256": data_hash,
               "all_stages": {**previous_summary.get("all_stages", previous_summary.get("stages", {})), **records},
               "seconds": round(time.monotonic()-started, 3), "written": core.iso()}
    write_json(model_run / "summary.json", summary)
    print(f"G {args.model}: {state}; results: {model_run}")
    return EXIT[state]


def create_freeze(args):
    model_run = model_directory(args.runs, args.experiment, args.model)
    pilot = read_json(model_run / "state/pilot.json")
    if not pilot or pilot.get("mock") or pilot.get("state") != "done":
        raise ValueError("G freeze requires a successful real G pilot for this model; mock results cannot qualify")
    data, data_hash = load_data(args.inputs, False)
    expected = stage_signature("pilot", args.model, False, data, data_hash)
    if pilot.get("signature") != expected or not resumable(pilot, expected, model_run / "stages/pilot"):
        raise ValueError("G pilot code/prompts or artifact hashes changed; repeat it in a new experiment before freezing")
    names = [args.model]
    if args.model in TARGETS[:2]:
        names.append(args.model + "-assistant")
    measurements = read_json(model_run / "stages/pilot/pilot.json")
    if not measurements or measurements.get("mock") is not False or measurements.get("cache_parity") is not True:
        raise ValueError("G freeze requires real, passing native pilot measurements")
    decisions = dict(data.get("decisions") or {})
    if decisions.get("OD-M14") not in (None, "full_grid", "grid_2^14"):
        raise ValueError("Unknown OD-M14 decision; use full_grid or grid_2^14")
    if decisions.get("OD-M14") == "grid_2^14":
        parity = {r["tokens"]: r.get("cache_parity") for r in measurements.get("long_context", [])}
        if parity.get(16384) is not True or parity.get(32768) is not False:
            raise ValueError("OD-M14 reduction requires measured 16k parity success and 32k parity failure")
    required_lengths = {16384} if decisions.get("OD-M14") == "grid_2^14" else {16384, 32768}
    long_status = {r["tokens"]: r.get("status") for r in measurements.get("long_context", [])}
    if any(long_status.get(n) != "measured" for n in required_lengths):
        raise ValueError("Required planned-context memory/timing pilot is unavailable; cannot freeze this G configuration")
    value = {"schema": "dsse-g-freeze-v1", "mock": False, "model": args.model,
             "input_sha256": data_hash, "plan_sha256": core.sha256_file(PLAN),
             "plan_hashes": plan_hashes(),
             "source_hashes": source_hashes(), "code_commit": commit(),
             "models": {n: {k: get_spec(n)[k] for k in ("repo", "revision", "sha256")} for n in names},
             "pilot_sha256": core.sha256_file(model_run / "state/pilot.json"),
             "external_input_hashes": external_input_hashes(data), "decisions": decisions,
             "pilot_measurements": {k: v for k, v in measurements.items() if k != "provenance"},
             "g": data.get("g", {}), "written": core.iso(), "independent_of_qwen_runs": True}
    if Path(args.output).exists():
        raise FileExistsError("G freeze is immutable; choose a new experiment/output")
    write_json(args.output, value)
    print(f"G freeze written: {args.output}")
    return 0


def report(args):
    exp = Path(args.runs).resolve() / safe_name(args.experiment)
    models = [args.model] if getattr(args, "model", None) else list(TARGETS)
    rows = {m: read_json(exp / m / "summary.json", {"state": "not_started"}) for m in models}
    for model, row in rows.items():
        row["invocation_state"] = row["state"]
        row["stages"] = row.get("all_stages", row.get("stages", {}))
        for name, record in row["stages"].items():
            if record.get("state") in {"done", "partial"} and not resumable(
                    record, record.get("signature"), exp / model / "stages" / name):
                record.update(state="invalid", reason="saved output/provenance changed")
        for name in STAGES:
            record = row["stages"].get(name, {})
            for parent, expected in record.get("dependencies", {}).items():
                current = {k: row["stages"].get(parent, {}).get(k) for k in ("signature", "state", "outputs")}
                if current != expected:
                    record.update(state="invalid", reason=f"upstream G evidence changed: {parent}")
                    break
        states = {v["state"] for v in row["stages"].values()}
        if states:
            row["state"] = ("failed" if states & {"failed", "invalid"} else "yielded" if "yielded" in states else
                            "partial" if states & {"partial", "blocked"} else "mock_complete" if row.get("mock") else "done")
        row["scientific_complete"] = bool(not row.get("mock", True) and set(row["stages"]) == set(STAGES)
                                             and states <= {"done", "not_applicable"})
    sources = {"K detection": ("k_analyze", "k3_verdicts.json"),
               "G confirmatory families": ("g_analyze", "g_analysis.json"),
               "K causal controls": ("k_causal", "k3d_verdicts.json"),
               "K write readouts": ("k_readouts", "k3e_readouts.json"),
               "K chain": ("k_chain_analyze", "chain_analysis.json"),
               "K routes": ("k_routes", "routes.json"),
               "K persistence": ("k_persistence", "persistence_verdicts.json"),
               "K long-form errors": ("k_long", "long_verdicts.json"),
               "K channel/context/QA": ("k_extra", "extra_analysis.json"),
               "K length/content": ("k_contexts", "contexts.json"),
               "K hierarchy": ("k_hierarchy", "hierarchy.json"),
               "Native assistant": ("mtp", "mtp_calibration.json"),
               "L": ("l_analyze", "l_analysis.json"),
               "D1 contrast": ("d1", "d1.json"),
               "12B bridge": ("bridge", "bridge.json")}
    measurements = {}
    for model, summary in rows.items():
        measurements[model] = {}
        for label, (stage, filename) in sources.items():
            record = summary.get("stages", {}).get(stage, {})
            path = exp / model / "stages" / stage / filename
            expected = record.get("outputs", {}).get(filename)
            if not expected or not path.is_file():
                measurements[model][label] = {"state": record.get("state", "not_started"),
                                                "reason": record.get("reason", "measurement not available")}
                continue
            if core.sha256_file(path) != expected:
                measurements[model][label] = {"state": "invalid", "reason": "artifact hash changed"}
                continue
            data = read_json(path)
            # Keep the report portable and useful without copying large native
            # traces or unvalidated files into a cross-model result.
            selected = {k: data[k] for k in ("status", "verdict", "verdicts", "combined", "coverage", "gates", "scientific_verdict", "branches", "families",
                                            "pipeline_calibration_gates", "SG8", "G_K5", "reason", "interpretation") if k in data}
            if stage == "k_analyze":
                selected["verdicts"] = {k: data[k] for k in ("K3a", "K3b", "K3c", "K3g_frame") if k in data}
            measurements[model][label] = {"state": record.get("state"), "mock": summary.get("mock"),
                                            "artifact": str(path.relative_to(exp)), "sha256": expected, **selected}
    result = {"experiment": args.experiment, "models": rows, "measurements": measurements, "written": core.iso(),
              "scientific_complete": all(v.get("scientific_complete", False) for v in rows.values())}
    if args.command == "report":
        write_json(exp / "g_report.json", result)
    for model, row in rows.items():
        print(f"{model}: {row['state']}")
        for name, stage in row.get("stages", {}).items():
            print(f"  {name}: {stage['state']} {stage.get('reason','')}")
    print("Scientific G complete:", result["scientific_complete"])
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, choices=TARGETS)
    r.add_argument("--stage", choices=["all", *STAGES], default="all")
    r.add_argument("--mode", choices=["mock", "real"], default="mock")
    r.add_argument("--root", required=True)
    r.add_argument("--inputs")
    r.add_argument("--freeze")
    r.add_argument("--guard-seconds", type=float, default=4.5 * 3600)
    r.add_argument("--rerun", action="store_true")
    f = sub.add_parser("freeze")
    f.add_argument("--model", required=True, choices=TARGETS)
    f.add_argument("--inputs", required=True)
    f.add_argument("--output", required=True)
    st = sub.add_parser("status")
    st.add_argument("--model", choices=TARGETS)
    rp = sub.add_parser("report")
    for c in (r, f, st, rp):
        c.add_argument("--runs", required=True)
        c.add_argument("--experiment", required=True)
    prep = sub.add_parser("prepare-inputs")
    prep.add_argument("--output", required=True)
    prep.add_argument("--corpora-dir", required=True)
    prep.add_argument("--download-corpora", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            return run(args)
        if args.command == "freeze":
            return create_freeze(args)
        if args.command == "prepare-inputs":
            from r1b_g_data import prepare_inputs
            prepare_inputs(args.output, args.corpora_dir, args.download_corpora)
            return 0
        return report(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as e:
        print(f"G blocked: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
