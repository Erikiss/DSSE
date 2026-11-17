"""Build the single-runtime HTQ AUTO notebook without changing existing notebooks."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRANCH = "codex/r1b-htq-continuation"


def source():
    return '''# Ein Ablauf mit finally: auch Setupfehler dürfen keine leere GPU hängen lassen.
import os, sys, json, subprocess, hashlib, lzma, shutil, tempfile, uuid, traceback, signal
from pathlib import Path
RUN_ID = uuid.uuid4().hex
WORK, ROOT = "/content/work_htq_v2", "/content/root"
_initial_local_work = Path(WORK).exists() and any(p.is_dir() for p in Path(WORK).glob("r1b_*"))
_safe_to_release = not _initial_local_work
_science_started = False
_mounted = False
_notebook_log = None
_status_path = None
_failure = None
def _controller(command):
    # TERM lets the controller stop its detached science process group, sync,
    # and save a final status. Never abandon a live child after notebook Stop.
    process = subprocess.Popen(command, env=env, stdout=_notebook_log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    interrupted = None
    try:
        code = process.wait()
    except BaseException as error:
        interrupted = error
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        code = process.wait()
    status = json.loads(_status_path.read_text()) if _status_path.is_file() else {}
    return code, status, interrupted

def _notebook_state(state, **extra):
    if _status_path is None:
        return
    path = _status_path.with_name("notebook_status.json")
    value = dict(run_id=RUN_ID, state=state, safe_to_release=_safe_to_release,
                 updated_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), **extra)
    raw = (json.dumps(value, ensure_ascii=False, indent=1) + "\\n").encode("utf-8")
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)
    if path.read_bytes() != raw:
        raise OSError("Notebook-Status konnte nicht in Drive bestätigt werden.")
try:
    import torch
    assert torch.cuda.is_available(), "A100-Laufzeit auswählen."
    assert "A100" in torch.cuda.get_device_name(0) and torch.cuda.get_device_capability(0) == (8, 0), "Nur A100 ist freigegeben."
    from google.colab import drive
    drive.mount("/content/drive")
    _mounted = True
    MYDRIVE = "/content/drive/MyDrive"
    SOURCE_A, SOURCE_B = f"{MYDRIVE}/dsse_runs", f"{MYDRIVE}/dsse_runs_B"
    RUNS_A, RUNS_B = f"{SOURCE_A}/variants/htq_v2", f"{SOURCE_B}/variants/htq_v2"
    assert os.path.isdir(RUNS_A) and os.path.isdir(RUNS_B), "H_tq-Variantenordner fehlen; START muss vorher fertig sein."
    _attempt = Path(RUNS_A) / "_htq_auto" / "runs" / RUN_ID
    _attempt.mkdir(parents=True, exist_ok=True)
    _status_path = _attempt / "status.json"
    _notebook_log = open(_attempt / "notebook.log", "a", encoding="utf-8", buffering=1)
    env = {k: v for k, v in os.environ.items() if not k.startswith("R1B_") and k not in ("MOCK", "QUICK")}
    env.update(MOCK="0", QUICK="0")
    # Reused input cells launch subprocesses themselves. Give those the same
    # clean environment as the explicit controller process.
    for key in list(os.environ):
        if key.startswith("R1B_") or key in ("MOCK", "QUICK"):
            del os.environ[key]
    os.environ.update(MOCK="0", QUICK="0")
    _notebook_state("setup")
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "transformers==5.22.2"],
                   check=True, stdout=_notebook_log, stderr=subprocess.STDOUT, env=env)
    REPO = tempfile.mkdtemp(prefix="dsse_htq_auto_", dir="/content")
    subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--sparse", "-b", BRANCH,
                    "https://github.com/Erikiss/DSSE", REPO], check=True,
                   stdout=_notebook_log, stderr=subprocess.STDOUT, env=env)
    subprocess.run(["git", "-C", REPO, "sparse-checkout", "set", "analysis/ju_space", "results/ju_space/plans",
                    "results/ju_space/runde2_ergebnisse", "notebooks"], check=True,
                   stdout=_notebook_log, stderr=subprocess.STDOUT, env=env)
    R1B = f"{REPO}/analysis/ju_space/round1b"
    command = [sys.executable, "-u", f"{R1B}/r1b_auto_colab.py", "--runs-a", RUNS_A, "--runs-b", RUNS_B,
               "--root", ROOT, "--work", WORK, "--max-hours", str(MAX_GPU_STUNDEN),
               "--runtime-start", repr(T0), "--run-id", RUN_ID]
    if not STUFE2_AUTOMATISCH:
        command.append("--main-only")
    print("AUTO prüft den nächsten Schritt. Status und Logs:", str(_attempt), flush=True)
    preflight_code, status, interrupted = _controller(command + ["--check-only"])
    _safe_to_release = status.get("run_id") == RUN_ID and status.get("safe_to_release") is True
    if interrupted is not None:
        raise interrupted
    if preflight_code != 0 or status.get("state") != "ready":
        print("AUTO startet keine Messung:", status.get("state", "Fehler"), status.get("next_action", "notebook.log prüfen"), flush=True)
    else:
        # Dieselben festgelegten Eingabezellen wie die bewährten Slot-Notebooks.
        # ROOT und WORK bleiben für den gesamten Ablauf erhalten.
        slots = ["B2", "B1", "A1", "A2"] + (["A_T2", "B_T2"] if STUFE2_AUTOMATISCH else [])
        FZ = json.load(open(f"{REPO}/results/ju_space/plans/runde1b/runde1b_freeze.json"))
        import contextlib
        for slot in slots:
            notebook = json.load(open(f"{REPO}/notebooks/runde1b_{slot}.ipynb"))
            inputs = next("".join(c["source"]) for c in notebook["cells"]
                          if c["cell_type"] == "code" and "".join(c["source"]).startswith("# Eingaben:"))
            scope = dict(globals(), RUNS=RUNS_B if slot.startswith("B") else RUNS_A,
                         SELBSTTEST=False, NUR_SELBSTTEST=False, PILOT=False,
                         D=f"{REPO}/analysis/ju_space/gpu/data")
            with contextlib.redirect_stdout(_notebook_log), contextlib.redirect_stderr(_notebook_log):
                exec(compile(inputs, f"{slot}:geprüfte Eingaben", "exec"), scope)
        _safe_to_release = False
        _science_started = True
        print("AUTO rechnet jetzt nacheinander auf dieser A100. Keine zweite GPU-Laufzeit starten.", flush=True)
        finished_code, status, interrupted = _controller(command)
        _safe_to_release = status.get("run_id") == RUN_ID and status.get("safe_to_release") is True
        print("AUTO:", status.get("state", "Status fehlt"), "·", status.get("next_action", "Protokoll prüfen"), flush=True)
        print("Status:", str(_status_path), flush=True)
        if interrupted is not None:
            raise interrupted
except BaseException as error:
    _failure = error
    if _notebook_log is not None:
        traceback.print_exc(file=_notebook_log)
        _notebook_log.flush()
    print("AUTO angehalten:", type(error).__name__, str(error), flush=True)
    if _status_path is not None:
        print("Status/Setup-Protokoll:", str(_status_path.parent), flush=True)
        try:
            _notebook_state("failed", error_type=type(error).__name__, error=str(error))
        except Exception as status_error:
            _safe_to_release = not _initial_local_work and not _science_started
            print("Notebook-Status konnte nicht gesichert werden:", str(status_error), flush=True)
finally:
    if _notebook_log is not None:
        _notebook_log.close()
    if RUNTIME_NACH_STOP_FREIGEBEN and _safe_to_release:
        _release_ready = True
        try:
            _notebook_state("release_requested")
            if _mounted:
                drive.flush_and_unmount()
        except Exception as flush_error:
            _release_ready = not _initial_local_work and not _science_started
            print("Drive-Status/Flush fehlgeschlagen:", str(flush_error), flush=True)
        try:
            if not _release_ready:
                raise RuntimeError("Runtime bleibt zum Sichern wissenschaftlicher Daten erhalten.")
            print("GPU-Freigabe wird jetzt angefordert.", flush=True)
            from google.colab import runtime
            runtime.unassign()
        except Exception as release_error:
            print("Freigabe konnte nicht bestätigt werden:", str(release_error), flush=True)
    elif not _safe_to_release:
        print("Runtime bleibt erhalten: lokale Daten sind nicht als vollständig in Drive gesichert bestätigt.", flush=True)
if _failure is not None:
    raise RuntimeError("AUTO-Fehler; dauerhaftes notebook.log prüfen. Ungesicherte lokale Daten bleiben erhalten.") from None
'''


def build(out=HERE):
    markdown = '''# Runde 1b · AUTO auf einer A100 · Konto A

**START muss bereits abgeschlossen sein.** Alle alten GPU-Notebooks in beiden
Konten stoppen; ausschließlich dieses AUTO-Notebook auf einer A100 starten.
`dsse_runs_B` benötigt Bearbeiterrecht. Dann „Alle ausführen“.

Der Ablauf wechselt selbstständig: **B2 → B1 → A1 (D_C zuerst) → A2 → A_T2 → B_T2**.
Bereits fertige Sitzungen bleiben fertig. Dieselben Fragen, Schlüssel, Batchgrößen
und wissenschaftlichen Regeln gelten weiter. Der vorhandene Hidden-State-Mitschnitt
wird nicht erneut gerechnet. Es wartet keine zweite GPU auf Voraussetzungen.

Fehlen Eingaben oder scheitert ein Schritt, hält AUTO sofort an und speichert
Protokolle sowie die nächste Aktion in `dsse_runs/variants/htq_v2/_htq_auto` von A.
Bei erfolgreicher Sicherung gibt es die GPU automatisch frei. Scheitert die
Sicherung, bleibt die Laufzeit zum Retten der lokalen Daten erhalten.

Das **weiche Zielbudget** beträgt 20 Stunden ab der Konfigurationszelle,
beobachtet an den bestehenden Sitzungs-/Checkpointgrenzen. Ein langer einzelner
Schritt kann die Frist überziehen. Colab kann früher trennen; Verlängerung,
neue GPU-Zuteilung und erneute Drive-Anmeldung werden nicht automatisiert.
Nach einer normalen Budgetpause dasselbe Notebook erneut starten. Nach hartem
Runtimeverlust eine verbliebene `.writer.lock` prüfen, nicht ungeprüft löschen.

Für eine Pause in Drive die Datei `_htq_auto/PAUSE` anlegen. AUTO hält an der
nächsten Sitzungs-/Guardgrenze; vor dem Fortsetzen diese Pauseanforderung entfernen.
L-CAL und die abschließende ANA laufen separat auf CPU. L-CAL muss vor ANA geprüft
und nach der bestehenden Planregel übernommen sein.
'''
    config = f'''import time
T0 = time.time()
BRANCH = "{BRANCH}"
MAX_GPU_STUNDEN = 20
STUFE2_AUTOMATISCH = True
RUNTIME_NACH_STOP_FREIGEBEN = True
from datetime import datetime
from zoneinfo import ZoneInfo
print("Kontrollblick:", datetime.fromtimestamp(T0 + MAX_GPU_STUNDEN * 3600, ZoneInfo("Europe/Berlin")).strftime("%d.%m.%Y %H:%M %Z"),
      "· keine garantierte Stoppzeit")
'''
    cells = []
    for i, (kind, src) in enumerate((("markdown", markdown), ("code", config), ("code", source()))):
        c = dict(cell_type=kind, id=hashlib.sha256(f"HTQ_AUTO:{i}".encode()).hexdigest()[:12], metadata={}, source=src.splitlines(keepends=True))
        if kind == "code":
            ast.parse(src)
            c.update(execution_count=None, outputs=[])
        cells.append(c)
    doc = dict(cells=cells, metadata=dict(accelerator="GPU", colab=dict(gpuType="A100", provenance=[]),
                                        kernelspec=dict(display_name="Python 3", name="python3")), nbformat=4, nbformat_minor=5)
    target = Path(out) / "runde1b_HTQ_AUTO.ipynb"
    target.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return target


if __name__ == "__main__":
    print(build())
