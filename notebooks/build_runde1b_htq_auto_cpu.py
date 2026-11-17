"""Build the one-shot CPU finaliser; no waiting slot loop or calibration computation.

The executable helpers below are embedded verbatim in the notebook and are also
importable for model-free tests. Frozen calibration modules are only read.
"""
import ast
from contextlib import contextmanager
import copy
from datetime import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
CAL_REL = "results/ju_space/plans/runde1b/l_cal.json"
CAL_ITEMS = "abcdefghijklmn"
CAL_FILES = {
    "code_sha256": "analysis/ju_space/round1b/r1b_l_cal.py",
    "stats_sha256": "analysis/ju_space/round1b/r1b_l_stats.py",
    "rules_sha256": "analysis/ju_space/round1b/r1b_l_rules.py",
}


def _git_bytes(repo, name):
    proc = subprocess.run(["git", "-C", str(repo), "show", name], capture_output=True)
    if proc.returncode:
        raise RuntimeError("Benötigte eingecheckte Datei fehlt: " + name)
    return proc.stdout


def validate_committed_lcal(repo):
    """A full failed calibration is valid evidence; missing/quick/tampered data is not."""
    repo = Path(repo).resolve()
    path = repo / CAL_REL
    if not path.is_file():
        raise RuntimeError("L-CAL fehlt im Repository. Vollständige l_cal.json erst prüfen und committen; ANA wurde nicht gestartet.")
    raw = path.read_bytes()
    if raw != _git_bytes(repo, "HEAD:" + CAL_REL):
        raise RuntimeError("l_cal.json ist nicht bytegleich mit dem eingecheckten HEAD; erst committen und Notebook neu laden.")
    try:
        record = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise RuntimeError("l_cal.json ist kein lesbarer vollständiger JSON-Datensatz.") from error
    if not isinstance(record, dict) or record.get("quick") is not False:
        raise RuntimeError("L-CAL muss ausdrücklich quick=false sein; Kurztests sind keine Kalibrierung.")
    items = record.get("items")
    if not isinstance(items, dict) or set(items) != set(CAL_ITEMS):
        raise RuntimeError("L-CAL ist unvollständig: erforderlich sind alle Punkte a bis n.")
    if any(not isinstance(v, dict) or type(v.get("pass")) is not bool or not (set(v) - {"pass"}) for v in items.values()):
        raise RuntimeError("Jeder L-CAL-Punkt benötigt ein vollständiges Ergebnis und ein boolesches pass-Feld.")
    expected_gates = {
        "G_cal_L1": items["a"]["pass"],
        "G_cal_L2": all(items[k]["pass"] for k in "fgj"),
        "G_cal_L3": all(items[k]["pass"] for k in "bcdehi"),
        "G_cal_L4": all(items[k]["pass"] for k in "kl"),
        "G_cal_L5": all(items[k]["pass"] for k in "mn"),
    }
    gates = record.get("gates")
    if not isinstance(gates, dict) or set(gates) != set(expected_gates) or any(
            type(gates[k]) is not bool or gates[k] != v for k, v in expected_gates.items()):
        raise RuntimeError("L-CAL-Tore stimmen nicht mit den vollständigen Punkt-Ergebnissen überein.")
    freeze_path = repo / "results/ju_space/plans/runde1b/runde1b_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    commit = freeze.get("freeze_commit")
    if not isinstance(commit, str) or not commit:
        raise RuntimeError("Der Einfrier-Commit fehlt; L-CAL-Herkunft nicht prüfbar.")
    provenance = record.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("written"), str) or not provenance["written"]:
        raise RuntimeError("Die L-CAL-Provenienz mit Erstellungszeit fehlt.")
    try:
        if datetime.fromisoformat(provenance["written"].replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("timezone missing")
    except ValueError as error:
        raise RuntimeError("Die L-CAL-Erstellungszeit hat kein gültiges Zeitformat mit Zeitzone.") from error
    for key, rel in CAL_FILES.items():
        current = (repo / rel).read_bytes()
        frozen = _git_bytes(repo, commit + ":" + rel)
        digest = hashlib.sha256(current).hexdigest()
        if current != frozen or provenance.get(key) != digest:
            raise RuntimeError("L-CAL stammt nicht nachweislich aus dem eingefrorenen Code: " + rel)
    sys.path.insert(0, str(repo / "analysis/ju_space/round1b"))
    import r1b_core as K
    import r1b_l_ana as LA
    if Path(K.REPO).resolve() != repo or Path(LA.HERE).resolve() != repo / "analysis/ju_space/round1b":
        raise RuntimeError("Die importierten Analyse-Module gehören nicht zur aktuellen Arbeitskopie; frische CPU-Laufzeit verwenden.")
    passes, labels = LA.l_cal_record(SimpleNamespace(mock=False))
    if labels or set(passes) != set(CAL_ITEMS) or passes != {k: items[k]["pass"] for k in CAL_ITEMS}:
        raise RuntimeError("L-ANA lehnt die Kalibrierungsprovenienz ab: " + "; ".join(labels))
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), quick=False,
                item_pass=passes, gates=expected_gates, freeze_commit=commit)


@contextmanager
def _cpu_environment(values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update({key: str(value) for key, value in values.items()})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _gpu_ancestors(sessions, contracts):
    seen, todo = set(), list(sessions)
    while todo:
        session = todo.pop()
        if session in seen:
            continue
        seen.add(session)
        todo.extend(contracts.upstream(session))
    return sorted(s for s in seen if contracts.SESSIONS[s]["runtime"] == "a100")


def verify_calibration_lineage(repo, folder, expected_sha256):
    """A cached formal report must have used the same committed calibration file."""
    record = json.loads((Path(folder) / "session_done.json").read_text())
    commit = record.get("code_commit") if isinstance(record, dict) else None
    if not isinstance(commit, str) or not commit:
        raise RuntimeError("Kalibrierungsherkunft eines vorhandenen CPU-Abschlusses fehlt: " + str(folder))
    used = hashlib.sha256(_git_bytes(repo, commit + ":" + CAL_REL)).hexdigest()
    if used != expected_sha256:
        raise RuntimeError("Vorhandene CPU-Ergebnisse gehören zu einer anderen l_cal.json; Ergebnisse erhalten und getrennt prüfen: " + str(folder))
    return dict(calibration_sha256=used, producer_commit=commit)


@contextmanager
def _cpu_writer_lock(runs, run_id, on_error=None):
    from r1b_auto_colab import write_verified
    lock = Path(runs) / ".auto_cpu_writer.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError("Eine andere CPU-Auswertung oder eine ungeprüfte alte Sperre existiert: " + str(lock)) from None
    owner = lock / "owner.json"
    try:
        write_verified(str(owner), dict(run_id=run_id, pid=os.getpid()))
        yield
    except BaseException as error:
        # Publish failure while this writer still holds its lock. Publishing
        # after the context exits could overwrite a newly started writer.
        if on_error and owner.is_file() and json.loads(owner.read_text()).get("run_id") == run_id:
            on_error(error)
        raise
    finally:
        if owner.is_file():
            if json.loads(owner.read_text()).get("run_id") != run_id:
                raise RuntimeError("CPU-Schreibsperre hat einen anderen Besitzer; zur Prüfung erhalten: " + str(lock))
            owner.unlink()
        lock.rmdir()


def _cpu_process(command, env, log):
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        return subprocess.CompletedProcess(command, process.wait())
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise


def run_cpu_final(repo, root, work, runs, runs_b, env, max_continuations=8):
    """Run both final CPU sessions once their GPU ancestors are resolved; never poll."""
    sys.path.insert(0, str(Path(repo) / "analysis/ju_space/round1b"))
    from r1b_auto_colab import digest, durable_sync, files as durable_files, write_verified
    import r1b_contracts as C
    import r1b_core as K
    import r1b_htq_variant as V
    import r1b_run_session as R
    if type(max_continuations) is not int or max_continuations < 1:
        raise ValueError("max_continuations must be a positive integer")
    if R.MOCK or K.MOCK:
        raise RuntimeError("AUTO_ANA benötigt echte Ergebnisdaten; MOCK darf nicht aktiv sein.")
    runs, runs_b, work = Path(runs), Path(runs_b), Path(work)
    status_path = runs / "_auto_cpu_status.json"
    log_path = runs / "_auto_cpu.log"
    owner_path = work / ".htq_auto_cpu_owner.json"
    previous = json.loads(status_path.read_text()) if status_path.is_file() else {}
    owner = json.loads(owner_path.read_text()) if owner_path.is_file() else {}
    trusted = bool(owner.get("nonce") and owner.get("nonce") == previous.get("owner_nonce") and
                   owner.get("drive_root") == str(runs.resolve()) and owner.get("variant") == V.VARIANT)
    runtime_env = dict(env, R1B_H_POLICY=V.VARIANT, R1B_ACCOUNT="A", R1B_SLOT="ANA", R1B_ROOT=str(root),
                       R1B_RUNS_B=str(runs_b), RUN_DEVICE="cpu", DEVICE="cpu", PYTHONUNBUFFERED="1")
    state = dict(schema=1, phase="cpu_final", status="running", stage="validate", sessions={},
                 started_utc=K.iso(), finished_utc=None, error=None, run_id=uuid.uuid4().hex,
                 owner_nonce=owner.get("nonce") if trusted else None)
    V.check(runs, "A")
    V.check(runs_b, "B")
    pair_a = json.loads((runs / "_htq_pair_ready.json").read_text())
    pair_b = json.loads((runs_b / "_htq_pair_ready.json").read_text())
    if pair_a != pair_b or pair_a.get("variant") != V.VARIANT or pair_a.get("accounts") != ["A", "B"]:
        raise RuntimeError("Die beiden Variantenordner sind nicht als dasselbe vorbereitete Paar bestätigt.")
    def persist_failure(error):
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error), finished_utc=K.iso())
        try:
            write_verified(str(status_path), state)
        except Exception as persist_error:
            print("CPU-Status konnte nicht in Drive gespeichert werden:", persist_error, flush=True)

    try:
        with _cpu_writer_lock(runs, state["run_id"], persist_failure), _cpu_environment(runtime_env):
            # Reconcile local results before resolve()/pull() can accept an old
            # Drive 'done' marker or overwrite an unsynced checkpoint on retry.
            for session in ("r1b_ana", "r1b_ana2"):
                local, destination = work / session, runs / session
                source = durable_files(local)
                mismatch = any(not (destination / rel).is_file() or digest(path) != digest(destination / rel)
                               for rel, path in source.items())
                if mismatch:
                    if not trusted or session not in owner.get("sessions", []):
                        raise RuntimeError("Ungeprüfte lokale CPU-Ergebnisse werden zur Diagnose erhalten: " + str(local))
                    durable_sync(str(local), str(destination), session=session)
            owner = dict(nonce=owner["nonce"] if trusted else uuid.uuid4().hex, variant=V.VARIANT,
                         drive_root=str(runs.resolve()), sessions=owner.get("sessions", []) if trusted else [])
            state["owner_nonce"] = owner["nonce"]
            write_verified(str(owner_path), owner)
            write_verified(str(status_path), state)
            state["calibration"] = validate_committed_lcal(repo)
            state["stage"] = "gpu_prerequisites"
            P = R.Paths(str(runs), "A")
            gpu = {}
            for session in _gpu_ancestors(("r1b_ana", "r1b_ana2"), C):
                resolved, detail = R.resolve(P, session)
                gpu[session] = resolved
                if resolved not in ("done", "not_run", "declared"):
                    raise RuntimeError(f"GPU-Voraussetzung {session} fehlt oder läuft noch. Kein Warten auf CPU: GPU_AUTO fortsetzen.")
            state["gpu_prerequisites"] = gpu
            write_verified(str(status_path), state)
            with open(log_path, "a", encoding="utf-8", buffering=1) as log:
                print("\n=== AUTO CPU", state["started_utc"], "===", file=log, flush=True)
                for session in ("r1b_ana", "r1b_ana2"):
                    resolved, detail = R.resolve(P, session)
                    if resolved in ("not_run", "declared"):
                        state["sessions"][session] = dict(state=resolved, reason=detail)
                        continue
                    if resolved == "done":
                        verified = durable_sync(str(detail), str(runs / session), session=session)
                        lineage = verify_calibration_lineage(repo, runs / session, state["calibration"]["sha256"])
                        state["sessions"][session] = dict(state="done", resumed=True, lineage=lineage, verification=verified)
                        continue
                    check = R.check_start(session, P, K.freeze_record(False, R.freeze_path())[0])
                    if not check["ok"] and not check["not_run"]:
                        raise RuntimeError(f"{session} ist noch nicht bereit: " + "; ".join(check["waiting"]))
                    state.update(stage=session)
                    write_verified(str(status_path), state)
                    for attempt in range(max_continuations):
                        owner["sessions"] = sorted(set(owner["sessions"]) | {session})
                        write_verified(str(owner_path), owner)
                        process_env = dict(runtime_env, R1B_T0=repr(time.time()))
                        command = [sys.executable, "-u", str(Path(repo) / "analysis/ju_space/round1b/r1b_run_session.py"),
                                   session, str(root), str(work / session), str(runs)]
                        try:
                            result = _cpu_process(command, process_env, log)
                        finally:
                            verified = durable_sync(str(work / session), str(runs / session), session=session)
                        if result.returncode in (R.EXIT_DONE, R.EXIT_NOT_RUN):
                            resolved, detail = R.resolve(P, session)
                            if resolved not in ("done", "not_run", "declared"):
                                raise RuntimeError(f"{session}: kein passender dauerhafter Abschluss nach Exit {result.returncode}.")
                            lineage = verify_calibration_lineage(repo, runs / session, state["calibration"]["sha256"]) if resolved == "done" else None
                            state["sessions"][session] = dict(state=resolved, verification=verified, continuations=attempt, lineage=lineage)
                            write_verified(str(status_path), state)
                            break
                        if result.returncode != R.EXIT_GUARD:
                            raise RuntimeError(f"{session} scheiterte mit Exit {result.returncode}; Laufzeit bleibt zur Diagnose erhalten.")
                        state["sessions"][session] = dict(state="checkpoint", verification=verified, continuations=attempt + 1)
                        write_verified(str(status_path), state)
                    else:
                        raise RuntimeError(f"{session}: Fortsetzungsgrenze erreicht; dasselbe CPU-Notebook später erneut ausführen.")
            state.update(status="complete", stage="complete", finished_utc=K.iso())
            write_verified(str(status_path), state)
            return state
    except BaseException:
        raise


def runtime_helpers():
    definitions = [_git_bytes, validate_committed_lcal, _cpu_environment, _gpu_ancestors, verify_calibration_lineage,
                   _cpu_writer_lock, _cpu_process, run_cpu_final]
    return ("import hashlib, json, os, signal, subprocess, sys, time, uuid\nfrom pathlib import Path\n"
            "from types import SimpleNamespace\nfrom contextlib import contextmanager\nfrom datetime import datetime\n"
            f"CAL_REL = {CAL_REL!r}\nCAL_ITEMS = {CAL_ITEMS!r}\nCAL_FILES = {CAL_FILES!r}\n\n" +
            "\n\n".join(inspect.getsource(fn) for fn in definitions))


def build(out=HERE, source_dir=HERE):
    source = json.loads((Path(source_dir) / "runde1b_HTQ_ANA.ipynb").read_text())
    cells = []
    def add(kind, value):
        result = dict(cell_type=kind, metadata={}, source=value.splitlines(keepends=True))
        if kind == "code":
            ast.parse(value)
            result.update(execution_count=None, outputs=[])
        cells.append(result)
    add("markdown", """# Runde 1b · Automatische Gesamtauswertung · CPU, Konto A

**Erst nach abgeschlossenem GPU_AUTO auf CPU öffnen, dann „Alle ausführen“.**
Dieses Notebook rechnet `r1b_ana` und `r1b_ana2` direkt nacheinander. Es wartet
nicht auf fehlende GPU-Läufe und berechnet keine L-CAL. Bei fehlenden Eingaben
endet es mit einer klaren Meldung; bereits fertige CPU-Schritte werden fortgesetzt.

Die vollständige `l_cal.json` muss zuvor geprüft und im Branch committet sein.
Kurztests, fehlende Punkte oder falsche Provenienz stoppen vor der Auswertung.
Ein wissenschaftlich durchgefallener Kalibrierungspunkt ist dagegen ein gültiges
Ergebnis; seine festgelegten Tore bleiben geschlossen und werden so berichtet.

Protokoll und Status liegen im Variantenordner unter `_auto_cpu.log` und
`_auto_cpu_status.json`. Nach bestätigter Speicherung beider Abschlüsse wird die
CPU-Laufzeit auf Wunsch getrennt. Bei einem Fehler bleibt sie zur Diagnose erhalten.
""")
    add("code", """import os, sys, json, shutil, subprocess, hashlib, lzma, time
T0 = time.time()
SLOT, KONTO, PILOT = "ANA", "A", False
BRANCH = "codex/r1b-htq-continuation"
SELBSTTEST = NUR_SELBSTTEST = False
AUTOMATISCH_TRENNEN = True
MAX_CPU_FORTSETZUNGEN = 8
try:
    import torch
    if torch.cuda.is_available():
        raise RuntimeError("Dieses Abschlussnotebook auf CPU starten; GPU_AUTO hat die GPU-Arbeit bereits erledigt.")
except ImportError:
    pass
print("CPU-Abschluss ohne Warteschleife.")
""")
    for original in source["cells"]:
        if original["cell_type"] != "code":
            continue
        text = "".join(original["source"])
        if (text.startswith("# Getrennte Variantenordner") or text.startswith("# Code:") or
                text.startswith("# Varianten-Bootstrap") or text.startswith("# Eingaben:")):
            add("code", text)
    add("code", runtime_helpers())
    add("code", """# Genau zwei CPU-Abschlüsse; keine GPU-Messung und kein Warten auf Voraussetzungen.
print("Prüfe Kalibrierung und GPU-Abschlüsse. Protokoll:", os.path.join(RUNS, "_auto_cpu.log"), flush=True)
AUTO_CPU = run_cpu_final(REPO, ROOT, WORK, RUNS, B_RUNS,
                         dict(VARIANT_ENV, NT="8"), MAX_CPU_FORTSETZUNGEN)
print("CPU-Abschluss:", AUTO_CPU["status"], flush=True)
print("Status:", os.path.join(RUNS, "_auto_cpu_status.json"), flush=True)
if AUTOMATISCH_TRENNEN and AUTO_CPU["status"] == "complete":
    # Erst nach geprüftem Drive-Abschluss trennen; bei jedem früheren Fehler bleibt die Laufzeit erhalten.
    drive.flush_and_unmount()
    from google.colab import runtime
    runtime.unassign()
""")
    name = "runde1b_HTQ_AUTO_ANA.ipynb"
    for index, result in enumerate(cells):
        result["id"] = hashlib.sha256(f"{name}:{index}".encode()).hexdigest()[:12]
    doc = dict(cells=cells, metadata=copy.deepcopy(source["metadata"]), nbformat=4, nbformat_minor=5)
    doc["metadata"].pop("accelerator", None)
    doc["metadata"].setdefault("colab", {}).pop("gpuType", None)
    doc["metadata"]["htq_auto_phase"] = "cpu_final"
    path = Path(out) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    return path


if __name__ == "__main__":
    print(build().name)
