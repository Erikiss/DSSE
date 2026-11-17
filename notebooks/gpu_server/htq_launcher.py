"""Portable, isolated HTQ session launcher. No provisioning or package installation.

Only A100 execution is currently admitted by the unchanged scientific SG2 gate.
Other hardware can be planned, but requires a separate reviewed migration first.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[2]
R1B = REPO / "analysis/ju_space/round1b"
sys.path.insert(0, str(R1B))
import r1b_contracts as C

VARIANT = "htq_v2"
SUBDIR = Path("variants/htq_v2")
MARKER = "_htq_seed.json"
SCRUB = ("R1B_H_POLICY", "R1B_RUNS_B", "R1B_XFER_A", "R1B_SOURCE_RUNS", "R1B_CAPTURE_HIDDEN", "MOCK", "QUICK")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(temporary, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"expected a physical directory, not a shortcut: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"resolve/export shortcuts before importing: {path}")
        if path.is_file():
            st = path.stat()
            result[path.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns)
        elif not path.is_dir():
            raise ValueError(f"unsupported source object: {path}")
    return result


def copy_verified(source, destination):
    """Independent inodes, all files including .pt, source-change detection."""
    source, destination = Path(source), Path(destination)
    before = inventory(source)
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for rel in before:
        src, dst = source / rel, destination / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        expected = sha(src)
        shutil.copy2(src, dst)
        if sha(dst) != expected or sha(src) != expected:
            raise ValueError(f"copy/source hash changed: {src}")
        hashes[rel] = dict(bytes=before[rel][0], sha256=expected)
    if inventory(source) != before:
        raise ValueError(f"source changed during copy; stop producers first: {source}")
    return hashes


def require_space(directory, needed, label):
    reserve = 1 << 30
    free = shutil.disk_usage(directory).free
    if free < needed + reserve:
        raise ValueError(f"insufficient space for {label}: need {needed / 2**30:.1f} GiB plus 1 GiB reserve, free {free / 2**30:.1f} GiB; models/outputs need additional space")


def verify_producer_hashes(source):
    """Check current producer evidence; old seed hashes and superseded step owners are not authoritative."""
    if os.environ.get("MOCK") == "1" or os.environ.get("QUICK") == "1":
        raise ValueError("server imports require real-run mode; unset MOCK/QUICK")
    import r1b_run_session as R
    registry = None
    checked = 0
    for session in sorted(Path(source).iterdir()):
        if session.name not in C.SESSIONS or not session.is_dir():
            continue
        record_path = next((session / n for n in ("session_partial.json", "session_done.json") if (session / n).is_file()), None)
        if record_path:
            record = read(record_path)
            if record.get("session") != session.name:
                raise ValueError(f"producer session identity mismatch: {record_path}")
            if record.get("mock") is True:
                raise ValueError(f"mock producer outputs cannot seed a real server continuation: {record_path}")
            if record.get("account") is not None and record["account"] != C.SESSIONS[session.name]["account"]:
                raise ValueError(f"producer account identity mismatch: {record_path}")
            hashes = record.get("files") or {}
        else:
            # An intentionally incomplete K_S1 prefix has no session marker.
            registry = registry if registry is not None else R.registry()
            hashes = {}
            for step in registry.get(session.name, []):
                marker = session / "steps" / (step.name + ".json")
                rec = read(marker) if marker.is_file() else {}
                if rec.get("mock") is True:
                    raise ValueError(f"mock producer step cannot seed a real continuation: {marker}")
                if rec.get("status") == "ok":
                    hashes.update({rel: value.get("sha256") for rel, value in (rec.get("outputs") or {}).items()})
        if not isinstance(hashes, dict):
            raise ValueError(f"invalid current producer file map: {session}")
        for rel, expected in hashes.items():
            path = session / rel
            if Path(rel).is_absolute() or ".." in Path(rel).parts or path.is_symlink() or not path.is_file() or sha(path) != expected:
                raise ValueError(f"current producer hash does not match: {path}")
            checked += 1
    return checked


def runs(state, account):
    return Path(state) / "published" / account / "dsse_runs" / SUBDIR


def relocate_marker(source, destination, account, evidence):
    """Rebase transport metadata explicitly; scientific bytes are never changed."""
    original = (Path(source) / MARKER).read_text(encoding="utf-8")
    rec = json.loads(original)
    if (rec.get("schema") != 1 or rec.get("account") != account or rec.get("variant") != VARIANT
            or rec.get("h_policy") != VARIANT or not isinstance(rec.get("files"), dict)):
        raise ValueError(f"not a completed {account} HTQ bootstrap: {source}")
    destination = Path(destination)
    write(evidence, dict(operation="verified independent copy; bootstrap path metadata rebased",
                         copied_utc=now(), source=str(Path(source).resolve()), destination=str(destination),
                         original_seed_marker_utf8=original,
                         original_seed_sha256=hashlib.sha256(original.encode()).hexdigest()))
    rec.update(source=str(destination.parent.parent), destination=str(destination),
               transport=dict(evidence=str(evidence), scientific_outputs_changed=False))
    write(destination / MARKER, rec)


@contextlib.contextmanager
def exclusive(state):
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    lock = state / ".coordinator.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError(f"another coordinator is active, or inspect stale lock before removal: {lock}") from None
    write(lock / "owner.json", dict(pid=os.getpid(), started_utc=now()))
    try:
        yield
    finally:
        (lock / "owner.json").unlink()
        lock.rmdir()


def initialise(state, source_a, source_b):
    state = Path(state).absolute()
    with exclusive(state):
        if (state / "config.json").exists() or (state / "published").exists():
            raise ValueError("state is already initialised; use status/run, never overwrite imported progress")
        stage = Path(tempfile.mkdtemp(prefix=".import-", dir=state))
        try:
            size = sum(size for source in (source_a, source_b) for size, _ in inventory(source).values())
            require_space(state, size, "immutable A/B import")
            records = {}
            for account, source in (("A", source_a), ("B", source_b)):
                source = Path(source).absolute()
                if state == source or state.is_relative_to(source) or source.is_relative_to(state):
                    raise ValueError("state and imported source directories must not overlap")
                target = stage / account / "dsse_runs" / SUBDIR
                checked = verify_producer_hashes(source)
                records[account] = dict(source=str(source), current_producer_outputs_checked=checked, files=copy_verified(source, target))
                # Preserve original marker bytes before recording portable paths.
                relocate_marker(source, target, account, state / f"import_{account}.json")
            os.rename(stage, state / "published")
            stage = None
            # The atomic directory move changed paths; record that second metadata-only relocation.
            for account in ("A", "B"):
                target = runs(state, account)
                rec = read(target / MARKER)
                rec.update(source=str(target.parent.parent), destination=str(target))
                write(target / MARKER, rec)
                evidence = read(state / f"import_{account}.json")
                evidence["published_destination"] = str(target)
                write(state / f"import_{account}.json", evidence)
            write(state / "config.json", dict(schema=1, variant=VARIANT, imported_utc=now(),
                                               sources=records, note="All original .pt and provenance files retained."))
        finally:
            if stage is not None:
                shutil.rmtree(stage)
    return state


def execution_env(state, a, b, worker_root=None, gpu=None, capture=False):
    env = {k: v for k, v in os.environ.items() if k not in SCRUB and not k.startswith("R1B_")}
    env.update(R1B_H_POLICY=VARIANT, R1B_RUNS_B=str(b), R1B_XFER_A=str(Path(a) / C.XFER_DIR),
               RUN_DEVICE="cpu" if gpu is None else "cuda", NT="8", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", MOCK="0", QUICK="0",
               R1B_FREEZE=str(REPO / C.PLAN_DIR / "runde1b_freeze.json"))
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    if worker_root is not None:
        env["R1B_ROOT"] = str(worker_root)
        env["HF_HOME"] = str(Path(worker_root).parent / "hf_cache")
    if capture:
        env["R1B_CAPTURE_HIDDEN"] = "1"
    return env


@contextlib.contextmanager
def installed_env(env):
    affected = set(env) | {key for key in os.environ if key in SCRUB or key.startswith("R1B_")}
    previous = {key: os.environ.get(key) for key in affected}
    for key in affected - set(env):
        os.environ.pop(key, None)
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def plan(state, sessions):
    a, b = runs(state, "A"), runs(state, "B")
    env = execution_env(state, a, b)
    result = {}
    with installed_env(env):
        import r1b_run_session as R
        if R.MOCK or R.K.MOCK:
            raise ValueError("runner was imported in MOCK mode; launch planning in a fresh real-run process")
        for session in sessions:
            account = C.SESSIONS[session]["account"]
            own = (a if account == "A" else b) / session
            if R.is_done(own):
                R.check_session_policy(str(own), session)
                result[session] = dict(state="done", runtime=C.SESSIONS[session]["runtime"])
                continue
            if (own / "session_not_run.json").is_file():
                result[session] = dict(state="not_run", runtime=C.SESSIONS[session]["runtime"])
                continue
            P = R.Paths(str(a if account == "A" else b), account)
            check = R.check_start(session, P, read(REPO / C.PLAN_DIR / "runde1b_freeze.json"), copy=False)
            result[session] = dict(state="ready" if check["ok"] or check["not_run"] else "waiting",
                                   runtime=C.SESSIONS[session]["runtime"], waiting=check["waiting"], not_run=check["not_run"])
    return result


def preflight(gpus):
    """No model inference and no gate override. Fail before staging on other GPUs."""
    records = []
    probe = '''import json,torch,transformers
p=torch.cuda.get_device_properties(0)
print(json.dumps(dict(name=p.name,cc=list(torch.cuda.get_device_capability(0)),memory=p.total_memory,
torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__)))'''
    for gpu in gpus:
        proc = subprocess.run([sys.executable, "-c", probe], env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                              capture_output=True, text=True, timeout=120)
        if proc.returncode:
            raise ValueError(f"GPU {gpu} preflight failed: {proc.stderr[-1000:]}")
        rec = json.loads(proc.stdout.strip().splitlines()[-1])
        if "A100" not in rec["name"] or rec["cc"] != [8, 0] or rec["memory"] < 39e9:
            raise ValueError(f"GPU {gpu}: {rec['name']} is not admitted by frozen SG2. Hopper/Blackwell migration and parity validation are pending.")
        if rec["transformers"] != "5.22.2":
            raise ValueError(f"GPU {gpu}: use the existing transformers==5.22.2 pin")
        records.append(dict(gpu=gpu, **rec))
    return records


def stage_job(state, worker, session):
    state, worker = Path(state), Path(worker)
    needed = sum(size for account in ("A", "B") for size, _ in inventory(runs(state, account)).values())
    require_space(worker, needed, "private A/B worker snapshot")
    job = Path(tempfile.mkdtemp(prefix=f"{session}-", dir=worker))
    for account in ("A", "B"):
        source = runs(state, account)
        destination = job / account / "dsse_runs" / SUBDIR
        copy_verified(source, destination)
        relocate_marker(source, destination, account, job / f"transport_{account}.json")
    return job


def reject_pending_scripts(sessions, status):
    """Old R script steps create new process groups outside this coordinator's scope."""
    import r1b_run_session as R
    registry = R.registry()
    pending = [s for s in sessions if status[s]["state"] not in ("done", "not_run")
               and any(step.script for step in registry.get(s, []))]
    if pending:
        raise ValueError("import already completed script sessions before this continuation: " + ", ".join(pending))


def stop_worker(proc):
    """Workers own a POSIX process group, including assets/session grandchildren."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # The leader may have exited while a grandchild remains. Kill that group too.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def publish_job(state, job, session):
    """Publish only complete, verified local WORK; never trust best-effort runner sync."""
    state, job = Path(state), Path(job)
    account = C.SESSIONS[session]["account"]
    source = job / "work_htq_v2" / session
    done, nr = source / "session_done.json", source / "session_not_run.json"
    if not done.is_file() and not nr.is_file():
        raise ValueError(f"worker returned without a finished session: {source}")
    if (source / "session_partial.json").exists():
        raise ValueError(f"partial work is retained privately, not published: {source}")
    if done.is_file():
        record = read(done)
        if record.get("session") != session or record.get("account") != account:
            raise ValueError("finished session identity differs from its assigned job")
        if C.SESSIONS[session]["part"] in ("K", "L", "main") and session != "r1b_K_S3" and record.get("h_policy") != VARIANT:
            raise ValueError("refusing to publish non-HTQ dependent results")
        for rel, expected in (record.get("files") or {}).items():
            path = source / rel
            if Path(rel).is_absolute() or ".." in Path(rel).parts or path.is_symlink() or not path.is_file() or sha(path) != expected:
                raise ValueError(f"finished session output fails verification: {rel}")
    destination = runs(state, account) / session
    stage = destination.with_name(f".{session}.publish-{os.getpid()}")
    copy_verified(source, stage)
    if destination.exists():
        # Retain the imported partial/prefix exactly; never overwrite evidence.
        archive = state / "replaced_prefixes" / account
        archive.mkdir(parents=True, exist_ok=True)
        os.rename(destination, archive / f"{session}-{time.time_ns()}")
    os.rename(stage, destination)


def assets(root, runroot, slot):
    """Reuse the committed notebook's exact pinned-input cell, not legacy setup.sh."""
    namespace = dict(__name__="__main__", os=os, sys=sys, json=json, shutil=shutil, subprocess=subprocess,
                     ROOT=str(root), RUNS=str(runroot), REPO=str(REPO), R1B=str(R1B),
                     D=str(REPO / "analysis/ju_space/gpu/data"), SELBSTTEST=False,
                     NUR_SELBSTTEST=False, PILOT=False, FZ=read(REPO / C.PLAN_DIR / "runde1b_freeze.json"))
    import hashlib as _hashlib
    import lzma
    namespace.update(hashlib=_hashlib, lzma=lzma)
    notebook = read(REPO / "notebooks" / f"runde1b_{slot}.ipynb")
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if c["cell_type"] == "code" and "".join(c["source"]).startswith("# Eingaben:"))
    Path(root).mkdir(parents=True, exist_ok=True)
    exec(compile(source, f"runde1b_{slot}:pinned-inputs", "exec"), namespace)


def run(state, sessions, gpus, cpu=False, capture=False):
    state = Path(state).absolute()
    with exclusive(state):
        if not (state / "config.json").is_file():
            raise ValueError("initialise the verified local imports first")
        if any((C.SESSIONS[s]["runtime"] == "cpu") != cpu for s in sessions):
            raise ValueError("keep CPU ANA sessions separate from GPU sessions")
        a = runs(state, "A")
        if not (a / "r1b_R_S0/session_done.json").is_file():
            raise ValueError("this continuation launcher requires imported completed R_S0 and its exports")
        exports = read(a / C.XFER_DIR / C.XFER_MANIFEST)
        if not isinstance(exports, dict) or not exports:
            raise ValueError("import R_S0's complete verified r1b_xfer exports first")
        for name, value in exports.items():
            path = a / C.XFER_DIR / name
            if Path(name).is_absolute() or ".." in Path(name).parts or not isinstance(value, dict) or sha(path) != value.get("sha256"):
                raise ValueError(f"invalid imported R_S0 export: {name}")
        reject_pending_scripts(sessions, plan(state, sessions))
        if not cpu:
            write(state / "preflight.json", dict(checked_utc=now(), devices=preflight(gpus)))
        active, pending = {}, set(sessions)
        slots = {s: slot for slot, spec in C.SLOTS.items() for s in spec["sessions"]}
        devices = [None] if cpu else list(gpus)
        priority = {s: i for i, s in enumerate(("r1b_K_S1", "r1b_L_A", "r1b_K_S2a", "r1b_K_S4", "r1b_L_B",
                                               "r1b_D_C", "r1b_K_S2b", "r1b_K_S5", "r1b_R_S5", "r1b_R_S7"))}
        try:
            while pending or active:
                current = plan(state, sorted(pending))
                pending -= {s for s, rec in current.items() if rec["state"] in ("done", "not_run")}
                for device in devices:
                    if device in active:
                        continue
                    ready = next((s for s in sorted(pending, key=lambda s: (priority.get(s, 100), s)) if current[s]["state"] == "ready"), None)
                    if ready is None:
                        continue
                    pending.remove(ready)
                    worker = state / "workers" / ("cpu" if cpu else f"gpu{device}")
                    worker.mkdir(parents=True, exist_ok=True)
                    checkpoint = worker / f"resume_{ready}.json"
                    if checkpoint.is_file():
                        job = Path(read(checkpoint)["job"])
                    else:
                        job = stage_job(state, worker, ready)
                        write(checkpoint, dict(job=str(job), session=ready, created_utc=now()))
                    a, b = (job / acct / "dsse_runs" / SUBDIR for acct in ("A", "B"))
                    root = worker / "root"
                    env = execution_env(state, a, b, root, device, capture=capture and ready == "r1b_K_S1")
                    account = C.SESSIONS[ready]["account"]
                    env.update(R1B_ACCOUNT=account, R1B_SLOT=slots[ready])
                    own = a if account == "A" else b
                    log = open(job / "worker.log", "a", buffering=1)
                    command = [sys.executable, "-u", str(Path(__file__)), "_worker", "--root", str(root),
                               "--runs", str(own), "--work", str(job / "work_htq_v2" / ready), "--session", ready, "--slot", slots[ready]]
                    try:
                        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        log.close()
                        raise
                    active[device] = (ready, job, checkpoint, proc, log)
                    print(f"started {ready} on {'CPU' if cpu else 'GPU '+str(device)}; log {job/'worker.log'}", flush=True)
                published = False
                for device, (session, job, checkpoint, proc, log) in list(active.items()):
                    code = proc.poll()
                    if code is None:
                        continue
                    stop_worker(proc)
                    log.close()
                    del active[device]
                    if code not in (0, 5):
                        raise ValueError(f"{session} failed with exit {code}; private checkpoint retained: {job}")
                    publish_job(state, job, session)
                    published = True
                    checkpoint.unlink()
                    print(f"published {session}", flush=True)
                if published and pending and not active:
                    continue  # Prerequisites just became visible; recompute readiness.
                if pending and not active and not any(v["state"] == "ready" for s, v in current.items() if s in pending):
                    write(state / "waiting.json", {s: current[s] for s in sorted(pending)})
                    raise ValueError(f"no runnable sessions; see {state/'waiting.json'} for missing prerequisites")
                if active:
                    time.sleep(1)
        finally:
            for _, _, _, proc, log in active.values():
                stop_worker(proc)
                log.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "status", "plan", "run", "_worker"))
    parser.add_argument("--state", type=Path)
    parser.add_argument("--source-a", type=Path)
    parser.add_argument("--source-b", type=Path)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--tier2", action="store_true")
    parser.add_argument("--capture-hidden", action="store_true")
    parser.add_argument("--sessions", nargs="+")
    for name in ("root", "runs", "work", "session", "slot"):
        parser.add_argument("--" + name)
    args = parser.parse_args(argv)
    try:
        if args.command == "_worker":
            assets(args.root, args.runs, args.slot)
            while True:
                code = subprocess.run([sys.executable, "-u", str(R1B / "r1b_run_session.py"), args.session,
                                       args.root, args.work, args.runs],
                                      env=dict(os.environ, R1B_T0=repr(time.time()))).returncode
                if code != 3:
                    return code
        if args.state is None:
            parser.error("--state is required")
        if args.command == "init":
            if args.source_a is None or args.source_b is None:
                parser.error("init requires --source-a and --source-b prepared HTQ variant folders")
            print(initialise(args.state, args.source_a, args.source_b))
            return 0
        selected = args.sessions or [s for s, m in C.SESSIONS.items() if m["tier"] in ((1, 2) if args.tier2 else (1,))
                                     and (m["runtime"] == "cpu") == args.cpu]
        if len(set(selected)) != len(selected) or any(s not in C.SESSIONS for s in selected):
            raise ValueError("unknown or duplicate session names")
        if args.command in ("plan", "status"):
            print(json.dumps(plan(args.state, selected), indent=2))
            return 0
        gpus = [int(x) for x in args.gpus.split(",")]
        if len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
            raise ValueError("--gpus must be distinct nonnegative local GPU indexes")
        run(args.state, selected, gpus, args.cpu, args.capture_hidden)
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"HTQ server launcher refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
