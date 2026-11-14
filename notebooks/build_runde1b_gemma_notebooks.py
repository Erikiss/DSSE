"""Build the six independent, deterministic Gemma G Colab notebooks.

No Qwen run, account hand-off, or upstream integration marker is consulted.
Run this builder with an optional output directory (default: this directory).
The notebooks default to offline-model CPU mocks, using real G entry points.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import textwrap

import nbformat as nbf


HERE = Path(__file__).resolve().parent
BRANCH = "codex-teil-g"
TARGETS = ("gemma4-e2b", "gemma4-e4b", "gemma4-31b-qat", "gemma4-12b-qat")
NOTEBOOKS = {
    "PILOT": ("gemma4-e2b", "Gemma-Pilot und eigener G-Freeze"),
    "E2B": ("gemma4-e2b", "Gemma 4 E2B · float32"),
    "E4B": ("gemma4-e4b", "Gemma 4 E4B · float32"),
    "31B_QAT": ("gemma4-31b-qat", "Gemma 4 31B · QAT, nur Vorwärtsmessungen"),
    "12B_QAT": ("gemma4-12b-qat", "Gemma 4 12B · QAT und Neuron-Lens-Brücke"),
    "ANA": ("", "G-Ergebnisse und Auswertung"),
}


def intro(slot):
    _, title = NOTEBOOKS[slot]
    return f"""# Runde 1b · G · {title}

[In Google Colab öffnen](https://colab.research.google.com/github/Erikiss/DSSE/blob/{BRANCH}/notebooks/runde1b_G_{slot}.ipynb)

Die G-Serie läuft eigenständig. Dieses Notebook verwendet eigene Modelle,
Eingaben, Ergebnisse und einen eigenen Freeze. Es benötigt keine abgeschlossenen
Qwen-Läufe und keine Freigabe eines anderen Notebook-Slots.

**Sofort ausprobieren:** `MODE = "mock"` lassen und **Alle ausführen** wählen.
Dabei laufen kleine zufällige Gemma-Modelle auf der CPU; es werden keine
Modellgewichte heruntergeladen. Die Einrichtung lädt Python-Pakete und den Code.
Mock-Ergebnisse werden getrennt von echten Messungen gespeichert.

**Echte Läufe:** Im Pilot-Notebook das Modell wählen und `MODE = "pilot"` setzen.
`INPUTS` ist optional: Für Freeze und Hauptlauf wird bei leerem Feld ein eigenes
G-Paket aus Rohtexten und fest versionierten Korpora bereitgestellt.
WikiText und die PG19-Langdokumente werden dabei mit festen Versionen geladen.
Für die externe 12B-Brückenprüfung sind zusätzliche unabhängige Prüftexte samt
Herkunftsmanifest nötig; das erzeugte Paket beschreibt diese Felder. Ohne diese
Daten bleibt die Übertragungsprüfung ausdrücklich blockiert.
Eine **A100 mit 80 GB** ist erforderlich;
Colab stellt je nach Verfügbarkeit auch kleinere GPUs bereit. Der Hardwarecheck
erfolgt vor dem Herunterladen der Gewichte. Nach dem Pilot die Kennzahlen prüfen
und bei Bedarf mit `PREPARE_INPUTS = True` das Eingabepaket erzeugen und ergänzen.
Alle eigenen Eingaben vor dem Freeze fertigstellen. Anschließend
`FREEZE_AFTER_PILOT = True` setzen, um den eigenen G-Freeze zu schreiben.
Danach im passenden Modell-Notebook `MODE = "run"` verwenden. Die Auswertung
läuft auf der CPU.
Einen benötigten Hugging-Face-Token als Colab-Secret `HF_TOKEN` hinterlegen.

**Fortsetzen:** Experimentname, Code-Referenz und Eingaben beibehalten und das
Notebook erneut ausführen. Der erste Start hält den Code-Commit für das Experiment
fest. Fertige Schritte werden vom G-Runner geprüft und wiederverwendet.

**Ergebnisse:** Echte Läufe liegen unter
`Meine Ablage/dsse_g_runs/<Experiment>/<Modell>`. Fehlende Eingabedaten und nicht
erfüllte Testvoraussetzungen werden als **BLOCKIERT** bzw. **nicht prüfbar** gemeldet;
das Notebook meldet dann keinen erfolgreichen vollständigen G-Lauf.
"""


def settings(slot):
    model, _ = NOTEBOOKS[slot]
    return f'''# Einstellungen
MODE = "mock"                    # "mock", "pilot" oder "run"
EXPERIMENT = "gemma-g-v1"
MODEL = {model!r}                  # PILOT: eines der vier Zielmodelle wählen
CODE_REF = {BRANCH!r}             # Branch beim ersten Start oder fester 40-stelliger Commit
INPUTS = ""                      # optional: eigene G-Eingabedatei (JSON); sonst G-Paket erzeugen
PREPARE_INPUTS = False            # im Pilot Rohdatenpaket vor dem Freeze erstellen/prüfen
FREEZE_AFTER_PILOT = False        # nach geprüftem Pilot eigenen G-Freeze erstellen
RUN_SELFTESTS = True
USE_DRIVE = MODE != "mock"
LOCAL_BASE = "/content/dsse_g"
DRIVE_BASE = "/content/drive/MyDrive/dsse_g_runs"
INSTALL_DEPENDENCIES = True

import os, sys, json, re, subprocess, hashlib, time
from pathlib import Path

SLOT = {slot!r}
TARGETS = {TARGETS!r}
if MODE not in {{"mock", "pilot", "run"}}:
    raise ValueError("MODE muss mock, pilot oder run sein.")
if SLOT != "ANA" and MODEL not in TARGETS:
    raise ValueError("Bitte eines der vier G-Zielmodelle wählen: " + ", ".join(TARGETS))
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{{0,79}}", EXPERIMENT):
    raise ValueError("EXPERIMENT: 1–80 Buchstaben, Ziffern, Punkt, Unterstrich oder Bindestrich.")
if SLOT == "PILOT" and MODE == "run":
    raise ValueError("Im Pilot-Notebook MODE=pilot verwenden; Messungen laufen im Modell-Notebook.")
if SLOT == "ANA" and MODE == "pilot":
    raise ValueError("Die Auswertung verwendet MODE=mock oder MODE=run.")
if not CODE_REF or CODE_REF.startswith("-"):
    raise ValueError("CODE_REF fehlt oder ist ungültig.")
print(f"G · {{SLOT}} · {{MODE}} · {{EXPERIMENT}}" + (f" · {{MODEL}}" if MODEL else ""))
'''


PATHS = r'''
# Eigener Ergebnisspeicher; Mock-Läufe bleiben in ihrem eigenen Verzeichnis.
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
BASE = Path(LOCAL_BASE).expanduser().resolve()
RUNS = (Path(DRIVE_BASE).expanduser().resolve() if USE_DRIVE else BASE / ("mock_runs" if MODE == "mock" else "real_runs"))
ROOT = BASE / "model_cache"
EXPERIMENT_DIR = RUNS / EXPERIMENT
for directory in (BASE, RUNS, ROOT, EXPERIMENT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
FREEZE = EXPERIMENT_DIR / MODEL / "g_freeze.json" if MODEL else None
if MODE != "mock" and SLOT != "ANA" and INPUTS:
    if not Path(INPUTS).expanduser().is_file():
        raise FileNotFoundError("Die unter INPUTS angegebene G-Eingabedatei (JSON) fehlt.")
    INPUTS = str(Path(INPUTS).expanduser().resolve())
if MODE == "run" and SLOT != "ANA" and not FREEZE.is_file():
    raise FileNotFoundError(f"G-Freeze fehlt: {FREEZE}. Zuerst den G-Pilot für dieses Modell ausführen und einfrieren.")
print("Ergebnisse:", EXPERIMENT_DIR)
'''


CHECKOUT = r'''
# Der erste Start bindet dieses Experiment an genau einen Code-Commit.
import fcntl

REMOTE = "https://github.com/Erikiss/DSSE.git"
SOURCE = EXPERIMENT_DIR / "source.json"

def git_output(*args):
    return subprocess.run(["git", *map(str, args)], check=True, capture_output=True, text=True).stdout.strip()

with (EXPERIMENT_DIR / ".source.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if SOURCE.is_file():
        source = json.loads(SOURCE.read_text())
        if source.get("remote") != REMOTE or source.get("requested_ref") != CODE_REF:
            raise ValueError("Dieses Experiment verwendet bereits eine andere Code-Referenz. Einen neuen Experimentnamen wählen.")
        COMMIT = source["commit"]
    else:
        if re.fullmatch(r"[0-9a-fA-F]{40}", CODE_REF):
            COMMIT = CODE_REF.lower()
        else:
            lines = git_output("ls-remote", REMOTE, f"refs/heads/{CODE_REF}", f"refs/tags/{CODE_REF}", f"refs/tags/{CODE_REF}^{{}}").splitlines()
            peeled = [line.split()[0] for line in lines if line.endswith("^{}")]
            candidates = set(peeled or [line.split()[0] for line in lines])
            if len(candidates) != 1:
                raise ValueError("CODE_REF ist nicht eindeutig oder wurde nicht gefunden.")
            COMMIT = candidates.pop()
        source = {"remote": REMOTE, "requested_ref": CODE_REF, "commit": COMMIT}
        with SOURCE.open("x") as handle:
            json.dump(source, handle, indent=2)
    if not re.fullmatch(r"[0-9a-f]{40}", COMMIT):
        raise ValueError("Ungültiger Commit im Experimentdatensatz.")

REPO = BASE / "code" / COMMIT
REPO.parent.mkdir(parents=True, exist_ok=True)
with (REPO.parent / f".{COMMIT}.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if not (REPO / ".git").is_dir():
        if REPO.exists() and any(REPO.iterdir()):
            raise RuntimeError(f"Code-Verzeichnis ist belegt: {REPO}")
        subprocess.run(["git", "init", "-q", str(REPO)], check=True)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", "HEAD"], capture_output=True, text=True)
    if head.returncode:
        # Auch einen vor dem ersten Checkout abgebrochenen Download fortsetzen.
        origin = subprocess.run(["git", "-C", str(REPO), "remote", "get-url", "origin"], capture_output=True, text=True)
        if origin.returncode:
            subprocess.run(["git", "-C", str(REPO), "remote", "add", "origin", REMOTE], check=True)
        subprocess.run(["git", "-C", str(REPO), "sparse-checkout", "set", "analysis/ju_space/round1b", "analysis/ju_space/round2", "results/ju_space/plans/runde1b"], check=True)
        subprocess.run(["git", "-C", str(REPO), "fetch", "-q", "--filter=blob:none", "--depth=1", "origin", COMMIT], check=True)
        subprocess.run(["git", "-C", str(REPO), "checkout", "-q", "--detach", COMMIT], check=True)
    if git_output("-C", REPO, "rev-parse", "HEAD") != COMMIT:
        raise RuntimeError("Das lokale Code-Verzeichnis hat einen anderen Commit.")
    if git_output("-C", REPO, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Lokale Änderungen im Code-Verzeichnis; sie werden nicht überschrieben.")
R1B = REPO / "analysis" / "ju_space" / "round1b"
RUNNER = R1B / "r1b_g_run.py"
if not RUNNER.is_file():
    raise FileNotFoundError("Dieser Code-Commit enthält noch keinen eigenständigen G-Runner. Einen neueren G-Commit in einem neuen Experiment wählen.")
print("Code-Commit:", COMMIT)
'''


DEPENDENCIES = r'''
# Eigenes Python-Umfeld: installierte Pakete anderer Läufe werden nicht verändert.
import venv

ENV_KIND = "cpu" if MODE == "mock" or SLOT == "ANA" else "cu128"
ENV_PATH = BASE / "envs" / ("torch210-tf5101-ct015-acc112-" + ENV_KIND)
PINS = {"torch": "2.10.0", "transformers": "5.10.1", "compressed-tensors": "0.15.0", "accelerate": "1.12.0"}
if MODE != "mock" and SLOT != "ANA":
    PINS["pyarrow"] = "23.0.0"
if INSTALL_DEPENDENCIES:
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with (ENV_PATH.parent / (ENV_PATH.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        PYTHON = ENV_PATH / "bin" / "python"
        if not PYTHON.is_file():
            venv.EnvBuilder(with_pip=True).create(ENV_PATH)
        marker = ENV_PATH / "g_dependencies.json"
        if not marker.is_file() or json.loads(marker.read_text()) != PINS:
            subprocess.run([str(PYTHON), "-m", "pip", "install", "--quiet", "torch==2.10.0", "--index-url", "https://download.pytorch.org/whl/" + ENV_KIND], check=True)
            packages = [f"{package}=={pin}" for package, pin in PINS.items() if package != "torch"]
            subprocess.run([str(PYTHON), "-m", "pip", "install", "--quiet", *packages, "numpy", "scipy", "scikit-learn", "nbformat"], check=True)
            marker.write_text(json.dumps(PINS, sort_keys=True))
else:
    PYTHON = Path(sys.executable)
check = "import json,importlib.metadata as m; print(json.dumps({p:m.version(p) for p in " + repr(list(PINS)) + "}))"
installed = json.loads(subprocess.run([str(PYTHON), "-c", check], check=True, capture_output=True, text=True).stdout)
for package, wanted in PINS.items():
    if installed[package].split("+", 1)[0] != wanted:
        raise RuntimeError(f"Falsche Paketversion: {package} {installed[package]} statt {wanted}")
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", R1B_G_COMMIT=COMMIT)
ENV["MOCK"] = "1" if MODE == "mock" else "0"
if MODE == "mock":
    ENV.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
else:
    ENV.pop("HF_HUB_OFFLINE", None)
    ENV.pop("TRANSFORMERS_OFFLINE", None)
    if "HF_TOKEN" not in ENV:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
            if token:
                ENV["HF_TOKEN"] = token
            del token
        except (ImportError, KeyError):
            pass
        except Exception as exc:
            if type(exc).__name__ not in {"SecretNotFoundError", "NotebookAccessError"}:
                raise
print("Python-Umfeld bereit:", ENV_KIND)
'''


HARDWARE = r'''
# GPU prüfen, bevor der Runner Gewichte lädt.
if MODE != "mock" and SLOT != "ANA":
    probe = "import json,torch; print(json.dumps({'available':torch.cuda.is_available(),'name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else '', 'bytes':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0, 'capability':list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else []}))"
    gpu = json.loads(subprocess.run([str(PYTHON), "-c", probe], check=True, capture_output=True, text=True, env=ENV).stdout)
    if not gpu["available"] or "A100" not in gpu["name"] or gpu["bytes"] < 79_000_000_000 or gpu["capability"] != [8, 0]:
        raise RuntimeError(f"Dieser G-Lauf benötigt eine A100 mit 80 GB. Gefunden: {gpu['name'] or 'keine CUDA-GPU'}, {gpu['bytes']/1e9:.1f} GB. Eine passende Laufzeit wählen oder MODE=mock verwenden.")
    print("GPU:", gpu["name"], f"{gpu['bytes']/1e9:.1f} GB")
else:
    print("CPU-Lauf: kleine Zufallsmodelle bzw. vorhandene Ergebnisse.")
'''


SELFTEST = r'''
# Codeprüfung mit zufälligen Mini-Modellen, ohne Modell-Downloads.
if RUN_SELFTESTS and SLOT != "ANA":
    test_env = dict(ENV, MOCK="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for script in ("r1b_g_selftest.py", "r1b_g_checkpoint_test.py"):
        subprocess.run([str(PYTHON), str(R1B / script)], cwd=str(REPO), env=test_env, check=True)
else:
    print("Adapter-Selbsttests für diesen Start ausgelassen.")
'''


INPUT_PREPARATION = r'''
# Eigene G-Eingaben: Rohtexte aus dem Repo und fest versionierte Korpora.
# Im Mock-Modus erzeugt der Runner seine winzigen Testdaten selbst.
if MODE != "mock" and SLOT != "ANA" and not INPUTS and (MODE == "run" or PREPARE_INPUTS or FREEZE_AFTER_PILOT):
    prepared_inputs = EXPERIMENT_DIR / "inputs.json"
    corpora_dir = EXPERIMENT_DIR / "corpora"
    if not prepared_inputs.is_file():
        subprocess.run([str(PYTHON), "-u", str(RUNNER), "prepare-inputs",
                        "--output", str(prepared_inputs), "--corpora-dir", str(corpora_dir),
                        "--download-corpora"], cwd=str(REPO), env=ENV, check=True)
    INPUTS = str(prepared_inputs)
    print("G-Eingaben:", INPUTS)
'''


RUN = r'''
# Eigenständiger G-Runner: alle Eingaben und Startbedingungen gehören zur G-Serie.
def g_command(command, *args):
    return [str(PYTHON), "-u", str(RUNNER), command, "--runs", str(RUNS), "--experiment", EXPERIMENT, *map(str, args)]

FINAL_RC = 0
if SLOT == "ANA":
    command = g_command("report")
else:
    stage = "pilot" if SLOT == "PILOT" or MODE == "pilot" else "all"
    command = g_command("run", "--model", MODEL, "--stage", stage,
                        "--mode", "mock" if MODE == "mock" else "real", "--root", ROOT)
    if MODE != "mock" and INPUTS:
        command.extend(["--inputs", INPUTS])
    if MODE == "run":
        command.extend(["--freeze", str(FREEZE)])
FINAL_RC = subprocess.run(command, cwd=str(REPO), env=ENV).returncode
if FINAL_RC:
    print("G-Lauf nicht vollständig abgeschlossen. Status und Begründung folgen in der nächsten Zelle.")
elif MODE == "mock":
    print("Mock-Lauf abgeschlossen; diese Ergebnisse sind ausschließlich Codeprüfungen.")

if MODE == "pilot" and FREEZE_AFTER_PILOT and FINAL_RC == 0:
    if FREEZE.is_file():
        print("G-Freeze bereits vorhanden; der Hauptlauf prüft Eingaben und Code dagegen:", FREEZE)
    else:
        FINAL_RC = subprocess.run(g_command("freeze", "--model", MODEL, "--inputs", INPUTS, "--output", FREEZE),
                                  cwd=str(REPO), env=ENV).returncode
        if FINAL_RC == 0:
            print("G-Freeze:", FREEZE)
'''


SUMMARY = r'''
# Status bleibt auch bei BLOCKIERT/Teilfortsetzung sichtbar.
status_args = ["--model", MODEL] if MODEL else []
status_rc = subprocess.run(g_command("status", *status_args), cwd=str(REPO), env=ENV).returncode
if USE_DRIVE:
    print("Ergebnisse sind im eigenen G-Ordner in Drive gespeichert:", EXPERIMENT_DIR)
else:
    print("Lokale Ergebnisse:", EXPERIMENT_DIR)
if FINAL_RC == 4 and MODE == "mock":
    print("Technikdurchlauf beendet; wissenschaftlicher Umfang unvollständig. Die gemeldeten Tore und den Bericht ansehen.")
elif FINAL_RC == 3:
    raise RuntimeError("Das Laufzeitfenster ist erreicht; der Fortschritt wurde gespeichert. Dasselbe Notebook mit denselben Einstellungen erneut ausführen, um fortzusetzen.")
elif FINAL_RC:
    raise RuntimeError(f"G-Runner meldet Rückgabewert {FINAL_RC}. Die ausgegebene Ursache beheben; anschließend denselben Lauf fortsetzen.")
if status_rc:
    raise RuntimeError(f"G-Status konnte nicht vollständig gelesen werden ({status_rc}).")
'''


COMPARISON = r'''
# Vergleich aus den vom Runner gegen ihre Prüfsummen geprüften Ergebnisdateien.
report_path = EXPERIMENT_DIR / "g_report.json"
if not report_path.is_file():
    raise FileNotFoundError("Der G-Vergleichsbericht fehlt; zuerst die vorherige Auswertungszelle ausführen.")
report = json.loads(report_path.read_text())

def readable_decision(value, depth=0):
    if isinstance(value, dict):
        for key in ("verdict", "scientific_verdict", "string", "label"):
            if isinstance(value.get(key), str):
                return value[key]
        statistics = value.get("statistics")
        if isinstance(statistics, dict):
            for key in ("verdict", "string"):
                if isinstance(statistics.get(key), str):
                    return statistics[key]
        if "status" in value:
            return str(value["status"]) + (": " + str(value["reason"]) if value.get("reason") else "")
        if depth >= 2:
            return "Details in Ergebnisdatei"
        details = [f"{key}: {readable_decision(item, depth + 1)}" for key, item in value.items()
                   if key not in {"labels", "provenance", "gates", "coverage"} and isinstance(item, (str, dict))]
        return "; ".join(details) if details else "Details in Ergebnisdatei"
    return str(value)

def table_text(value):
    import html
    return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ")

lines = ["| Modell | Messung | Status | Aussage | Datei |", "|---|---|---|---|---|"]
for model, families in report.get("measurements", {}).items():
    for family, measurement in families.items():
        state = ("Mock / " if measurement.get("mock") else "") + str(measurement.get("state", "unbekannt"))
        decisions = measurement.get("verdicts") or measurement.get("branches") or measurement.get("families")
        if not isinstance(decisions, dict) or not decisions:
            decisions = {family: measurement.get("verdict", measurement.get("scientific_verdict",
                         measurement.get("reason", measurement.get("status", "Einzeldatei ansehen"))))}
        for name, value in decisions.items():
            label = family if name == family else family + " / " + str(name)
            cells = [model, label, state, readable_decision(value), measurement.get("artifact", "—")]
            lines.append("| " + " | ".join(table_text(value) for value in cells) + " |")
comparison = "\n".join(lines)
try:
    from IPython.display import Markdown, display
    display(Markdown(comparison))
except ImportError:
    print(comparison)
print("Dateipfade relativ zu:", EXPERIMENT_DIR)
print("Vergleichsbericht:", report_path)
print("Wissenschaftlich vollständig:", "ja" if report.get("scientific_complete") else "nein")
'''


def cells(slot):
    if slot not in NOTEBOOKS:
        raise KeyError(slot)
    result = [("md", "intro", intro(slot)), ("code", "settings", settings(slot)),
            ("code", "paths", PATHS), ("code", "checkout", CHECKOUT),
            ("code", "dependencies", DEPENDENCIES), ("code", "hardware", HARDWARE),
            ("code", "selftest", SELFTEST), ("code", "inputs", INPUT_PREPARATION),
            ("code", "run", RUN), ("code", "summary", SUMMARY)]
    if slot == "ANA":
        result.append(("code", "comparison", COMPARISON))
    return result


def notebook(slot):
    nb = nbf.v4.new_notebook()
    nb.cells = []
    for kind, name, source in cells(slot):
        source = textwrap.dedent(source).strip("\n")
        cell = nbf.v4.new_markdown_cell(source) if kind == "md" else nbf.v4.new_code_cell(source)
        cell.id = hashlib.sha256(f"g:{slot}:{name}".encode()).hexdigest()[:16]
        cell.metadata["g_section"] = name
        nb.cells.append(cell)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"provenance": []}
    nb.metadata["dsse_g"] = {"slot": slot, "default_mode": "mock", "independent": True}
    nbf.validate(nb)
    return nb


def build(slot, out=HERE):
    path = Path(out) / f"runde1b_G_{slot}.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook(slot), path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", type=Path, default=HERE)
    args = parser.parse_args(argv)
    for slot in NOTEBOOKS:
        print(build(slot, args.out))


if __name__ == "__main__":
    main()
