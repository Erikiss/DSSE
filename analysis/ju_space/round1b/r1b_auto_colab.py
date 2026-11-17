"""One A100, sequential HTQ continuation; no dependency polling or provisioning.

CPU L-CAL and ANA stay outside this queue. A runtime deadline is soft: an atomic
scientific step may overrun it before the existing runner can checkpoint.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r1b_contracts as C
import r1b_run_session as R
from r1b_htq_variant import VARIANT, check as check_seed

REPO = HERE.parents[2]


def utc():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_verified(path, value):
    """Atomic write and readback; callers must still flush Drive before unassign."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with open(temporary, "wb") as f:
        f.write(raw)
        f.flush()
    os.replace(temporary, path)
    if path.read_bytes() != raw:
        raise OSError(f"Drive status readback failed: {path}")


def files(root):
    root = Path(root)
    result = {}
    if not root.exists():
        return result
    for directory, dirs, names in os.walk(root):
        relative = Path(directory).relative_to(root)
        if not relative.parts:
            dirs[:] = [name for name in dirs if name not in R.NOT_SYNCED]
        for name in names:
            if name.endswith((".tmp", ".r1btmp", ".htq-auto-tmp")):
                continue
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"unsupported local checkpoint object: {path}")
            result[path.relative_to(root).as_posix()] = path
    return result


def current_hashes(work, session=None):
    work = Path(work)
    done = work / "session_done.json"
    if done.is_file() and not (work / "session_partial.json").exists():
        record = load(done)
        if session and record.get("session") != session:
            raise ValueError(f"checkpoint session mismatch: {done}")
        if record.get("mock") is True:
            raise ValueError("mock results cannot complete a real AUTO run")
        return record.get("files") or {}
    # A partial/failed step may have newer successful markers than the previous
    # session summary. Merge in registry order, as the runner's finalize does.
    merged = {}
    registry = R.registry().get(session, []) if session else []
    markers = [work / "steps" / (step.name + ".json") for step in registry]
    if not markers:
        markers = sorted((work / "steps").glob("*.json"))
    for marker in markers:
        rec = load(marker) if marker.is_file() else {}
        if rec.get("status") == "ok":
            merged.update({rel: value.get("sha256") for rel, value in (rec.get("outputs") or {}).items()})
    return merged


def durable_sync(work, drive, session=None):
    """Repair best-effort runner sync and verify every durable file, including caches."""
    work, drive = Path(work), Path(drive)
    source = files(work)
    for rel, expected in current_hashes(work, session).items():
        if Path(rel).is_absolute() or ".." in Path(rel).parts or rel not in source or digest(source[rel]) != expected:
            raise ValueError(f"local producer output is corrupt or missing: {work / rel}")
    hashes = {}
    for rel, src in source.items():
        before = src.stat()
        expected = digest(src)
        dst = drive / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.is_file() or digest(dst) != expected:
            temporary = dst.with_name(dst.name + ".htq-auto-tmp")
            shutil.copy2(src, temporary)
            os.replace(temporary, dst)
        if digest(dst) != expected or src.stat().st_size != before.st_size or src.stat().st_mtime_ns != before.st_mtime_ns:
            raise OSError(f"checkpoint could not be verified in Drive: {dst}")
        hashes[rel] = expected
    # The runner can leave an obsolete partial marker in Drive if its delete
    # failed. Remove it only when the local finished marker was fully verified.
    if "session_done.json" in hashes and "session_partial.json" not in source:
        stale = drive / "session_partial.json"
        if stale.exists():
            stale.unlink()
        if stale.exists():
            raise OSError(f"obsolete partial marker still exists: {stale}")
    return dict(verified=True, session=session, checked_utc=utc(), files=hashes,
                marker="done" if "session_done.json" in hashes else "partial" if "session_partial.json" in hashes else "checkpoint")


def sequence(include_tier2=True):
    stages = [("B2", C.SLOTS["B2"]["sessions"]), ("B1", C.SLOTS["B1"]["sessions"]),
              ("A1_DC_zuerst", ("r1b_D_C",) + tuple(s for s in C.SLOTS["A1"]["sessions"] if s != "r1b_D_C")),
              ("A2", C.SLOTS["A2"]["sessions"])]
    if include_tier2:
        stages += [(slot, C.SLOTS[slot]["sessions"]) for slot in ("A_T2", "B_T2")]
    result, seen = [], set()
    for stage, sessions in stages:
        for session in sessions:
            if session not in seen:
                result.append((stage, session)); seen.add(session)
    return result


class Engine:
    def __init__(self, runs_a, runs_b, xfer):
        self.roots = {"A": Path(runs_a), "B": Path(runs_b)}
        self.xfer = str(xfer)
        os.environ.update(R1B_H_POLICY=VARIANT, R1B_RUNS_B=str(runs_b), R1B_XFER_A=self.xfer)
        if R.MOCK or R.K.MOCK:
            raise ValueError("AUTO must run in a fresh non-MOCK process")
        self.freeze = R.K.freeze_record(False, R.freeze_path())[0]

    def bootstrap(self):
        for account, root in self.roots.items():
            check_seed(root, account)
        start = self.roots["B"] / "r1b_K_S1"
        if not R.is_done(str(start)):
            raise ValueError("HTQ_START must finish K_S1 before HTQ_AUTO")
        self.state("r1b_K_S1")  # Verify K1 producers before B2 consumes them.

    def state(self, session):
        folder = self.roots[C.SESSIONS[session]["account"]] / session
        if R.is_done(str(folder)):
            R.check_session_policy(str(folder), session)
            for rel, expected in current_hashes(folder, session).items():
                if Path(rel).is_absolute() or ".." in Path(rel).parts or digest(folder / rel) != expected:
                    raise ValueError(f"Drive producer output fails verification: {folder / rel}")
            return "done"
        if (folder / "session_not_run.json").is_file():
            record = load(folder / "session_not_run.json")
            if record.get("session") != session or not str(record.get("code", "")).startswith("NA:"):
                raise ValueError(f"invalid not-run marker: {folder}")
            return "not_run"
        return "pending"

    def ready(self, session):
        account = C.SESSIONS[session]["account"]
        return R.check_start(session, R.Paths(str(self.roots[account]), account), self.freeze, copy=False)


class Controller:
    def __init__(self, runs_a, runs_b, work, root, max_hours=20, runtime_start=None,
                 include_tier2=True, run_id=None, engine=None, executor=None):
        self.runs_a, self.runs_b = Path(runs_a), Path(runs_b)
        self.work, self.root = Path(work), Path(root)
        if self.work.name != "work_htq_v2":
            raise ValueError("AUTO requires WORK ending in work_htq_v2")
        self.run_id = run_id or uuid.uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", self.run_id):
            raise ValueError("invalid AUTO run identifier")
        self.limit = float(max_hours) * 3600
        self.t0 = float(runtime_start if runtime_start is not None else time.time())
        self.queue = sequence(include_tier2)
        # R0's transfer is H-independent and remains in the original A root.
        self.engine = engine or Engine(runs_a, runs_b, self.runs_a.parents[1] / C.XFER_DIR)
        self.executor = executor or self.execute
        self.directory = self.runs_a / "_htq_auto"
        self.attempt = self.directory / "runs" / self.run_id
        self.status_file = self.attempt / "status.json"
        self.pause_file = self.directory / "PAUSE"
        self.owner_file = self.work / ".htq_auto_owner.json"
        self.current = None
        # Until reconciliation finishes, any surviving local scientific files
        # might be the only copy. Even an early lock/bootstrap error must keep
        # the runtime alive in that case.
        self.dirty = any(files(p) for p in self.work.glob("r1b_*") if p.is_dir())
        self.owns_lock = False
        self.status = dict(schema=1, run_id=self.run_id, started_utc=utc(), state="starting",
                           safe_to_release=False, include_tier2=include_tier2, completed=[], next_action="start")

    def persist(self, state, safe, next_action, **extra):
        self.status.update(state=state, safe_to_release=bool(safe), next_action=next_action,
                           session=self.current, updated_utc=utc(), elapsed_seconds=time.time() - self.t0, **extra)
        write_verified(self.status_file, self.status)
        if self.owns_lock:
            write_verified(self.directory / "status.json", self.status)
        print(f"AUTO {state}: {self.current or '-'}; {next_action}", flush=True)

    @contextlib.contextmanager
    def lock(self):
        path = self.directory / ".writer.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.mkdir()
        except FileExistsError:
            raise ValueError(f"AUTO writer lock exists; inspect the other runtime before removing it: {path}") from None
        self.owns_lock = True
        write_verified(path / "owner.json", dict(run_id=self.run_id, pid=os.getpid(), started_utc=utc()))
        try:
            yield
        except BaseException as error:
            try:
                self.persist("sync_failed" if self.dirty else "failed", not self.dirty,
                             "preserve this runtime and inspect local work" if self.dirty else "inspect AUTO status/log and restart after fixing the cause",
                             error_type=type(error).__name__, error=str(error))
            except Exception:
                pass
            raise
        finally:
            (path / "owner.json").unlink()
            path.rmdir()
            self.owns_lock = False

    def reconcile_existing(self):
        old_owner = load(self.owner_file) if self.owner_file.is_file() else {}
        previous = load(self.directory / "status.json") if (self.directory / "status.json").is_file() else {}
        trusted = old_owner.get("run_id") and old_owner.get("run_id") == previous.get("run_id")
        for local in sorted(self.work.glob("r1b_*")):
            if not local.is_dir() or not files(local):
                continue
            session = local.name
            if session not in C.SESSIONS:
                raise ValueError(f"unknown local scientific folder preserved: {local}")
            drive = self.engine.roots[C.SESSIONS[session]["account"]] / session
            mismatch = any(not (drive / rel).is_file() or digest(path) != digest(drive / rel) for rel, path in files(local).items())
            if mismatch:
                self.dirty = True
                if not trusted or session not in old_owner.get("sessions", []):
                    raise ValueError(f"unverified pre-existing local work preserved; inspect before release: {local}")
                durable_sync(local, drive, session)
        self.dirty = False

    def environment(self, stage, session, remaining):
        env = {k: v for k, v in os.environ.items() if not k.startswith("R1B_") and k not in ("MOCK", "QUICK")}
        env.update(R1B_H_POLICY=VARIANT, R1B_ACCOUNT=C.SESSIONS[session]["account"],
                   R1B_SLOT="A1" if stage == "A1_DC_zuerst" else stage, R1B_RUNS_B=str(self.runs_b),
                   R1B_XFER_A=self.engine.xfer, R1B_ROOT=str(self.root), R1B_FREEZE=R.freeze_path(),
                   R1B_T0=repr(time.time()), R1B_RUNTIME_T0=repr(self.t0),
                   R1B_GUARD_S=str(max(1, min(4.5 * 3600, remaining))), RUN_DEVICE="cuda", NT="8",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", MOCK="0", QUICK="0")
        return env

    def execute(self, stage, session, env, log):
        own = self.engine.roots[C.SESSIONS[session]["account"]]
        command = [sys.executable, "-u", str(HERE / "r1b_run_session.py"), session, str(self.root),
                   str(self.work / session), str(own)]
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            heartbeat = time.monotonic()
            while process.poll() is None:
                if time.monotonic() - heartbeat >= 30:
                    action = "pause requested; finishing the current session checkpoint" if self.pause_file.exists() else "computing"
                    self.persist("running", False, action)
                    heartbeat = time.monotonic()
                time.sleep(2)
            return process.returncode
        finally:
            # Selected unfinished script steps are rejected below: their own
            # detached process groups are outside this controller's scope.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    def run(self, check_only=False):
        code = 2
        try:
            with self.lock():
                self.engine.bootstrap()
                self.reconcile_existing()
                self.persist("ready", True, "checking next session")
                for stage, session in self.queue:
                    self.current = session
                    while True:
                        terminal = self.engine.state(session)
                        if terminal in ("done", "not_run"):
                            self.status["completed"].append(dict(session=session, state=terminal))
                            break
                        if self.pause_file.exists() or time.time() - self.t0 >= self.limit:
                            self.persist("paused" if self.pause_file.exists() else "budget_stop", True, "restart HTQ_AUTO to continue this session")
                            return 3
                        check = self.engine.ready(session)
                        if not check["ok"] and not check.get("not_run"):
                            self.persist("blocked", True, "fix missing inputs, then restart HTQ_AUTO", missing=check["waiting"])
                            return 4
                        if any(step.script for step in R.registry().get(session, [])):
                            raise ValueError(f"unfinished legacy script session must already be imported complete: {session}")
                        if check_only:
                            self.persist("ready", True, "next GPU session is ready", stage=stage)
                            return 0
                        owner = load(self.owner_file) if self.owner_file.is_file() else {}
                        write_verified(self.owner_file, dict(run_id=self.run_id, sessions=sorted(set(owner.get("sessions", [])) | {session})))
                        self.dirty = True
                        self.persist("running", False, "computing", stage=stage)
                        env = self.environment(stage, session, self.limit - (time.time() - self.t0))
                        log_path = self.attempt / (session + ".log")
                        try:
                            with open(log_path, "a", encoding="utf-8", buffering=1) as log:
                                code = self.executor(stage, session, env, log)
                        finally:
                            durable_sync(self.work / session, self.engine.roots[C.SESSIONS[session]["account"]] / session, session)
                            self.dirty = False
                        if code == 3:
                            self.persist("checkpoint", True, "continuing the same session")
                            continue
                        if code not in (0, 5):
                            self.persist("failed", True, "inspect session log; no automatic retry", exit_code=code, log=str(log_path))
                            return 2
                        if self.engine.state(session) not in ("done", "not_run"):
                            raise ValueError(f"runner exit {code} did not produce a verified terminal session: {session}")
                        self.persist("checkpoint", True, "next session")
                self.persist("complete", True, "CPU: finish/review/commit L-CAL, then run ANA (no GPU needed)")
                return 0
        except BaseException as error:
            safe = not self.dirty
            try:
                self.persist("sync_failed" if self.dirty else "failed", safe, "preserve this runtime and inspect local work" if self.dirty else "inspect AUTO status/log and restart after fixing the cause",
                             error_type=type(error).__name__, error=str(error))
            except Exception as status_error:
                print(f"AUTO status could not be saved: {status_error}", file=sys.stderr, flush=True)
            return 5 if self.dirty else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runs-a", "runs-b", "work", "root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-hours", type=float, default=20)
    parser.add_argument("--runtime-start", type=float)
    parser.add_argument("--run-id")
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if args.max_hours <= 0:
        parser.error("--max-hours must be positive")
    # A notebook Stop/TERM must unwind run()/execute() so the detached science
    # process group is stopped and the checkpoint is verified before release.
    def terminate(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt("AUTO termination requested")
    signal.signal(signal.SIGTERM, terminate)
    return Controller(args.runs_a, args.runs_b, args.work, args.root, args.max_hours, args.runtime_start,
                      not args.main_only, args.run_id).run(args.check_only)


if __name__ == "__main__":
    raise SystemExit(main())
