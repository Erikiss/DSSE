"""Build isolated H_tq continuation notebooks using only the Python standard library.

The existing notebooks remain unchanged. The new notebooks clone the continuation
branch, use a separate Drive/work tree, and never export real Drive settings into
the notebook process environment (in particular, the MOCK self-test stays isolated).
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRANCH = "codex/r1b-htq-continuation"
VARIANT = "htq_v2"
SUBDIR = "variants/htq_v2"
PAIR_MARKER = "_htq_pair_ready.json"

SPECS = {
    "HTQ_START": ("B1_in_A", "H_tq vorbereiten und K_S1 neu rechnen", "start"),
    "HTQ_B2_in_A": ("B2_in_A", "B2 in Konto A", "in_a"),
    "HTQ_B1_in_A": ("B1_in_A", "B1 in Konto A", "in_a"),
    "HTQ_A1_DC_zuerst": ("A1_DC_zuerst", "A1 mit D_C zuerst", "a"),
    "HTQ_A2": ("A2", "A2", "a"),
    "HTQ_A_T2": ("A_T2", "Zusatzläufe in Konto A", "a"),
    "HTQ_B_T2": ("B_T2", "Zusatzläufe in Konto B", "b"),
    "HTQ_ANA": ("ANA", "Gesamtauswertung auf CPU", "a"),
}


def text(cell):
    return "".join(cell["source"])


def set_text(cell, source):
    cell["source"] = source.splitlines(keepends=True)


def cell(kind, source):
    c = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def intro(title, kind):
    account = "B" if kind == "b" else "A"
    source = f"# Runde 1b · {title} · H_tq-Variante\n\n"
    source += (
        f"**In Konto {account} öffnen**, in einer frischen Laufzeit. "
        "GPU-Notebooks benötigen eine A100; ANA verwendet CPU.\n\n"
        "**Post hoc, Änderung 2:** kleinere operative Neuronenmenge H_tq; "
        "Fragen, Stichproben, Schlüssel und Schwellen bleiben unverändert. "
        "Die bisherige Restschätzung von 15–18 A100-Stunden ist eine alte Planbasis, "
        "keine Laufzeitzusage; Umstellung und nötige Neuberechnungen kommen hinzu.\n\n"
    )
    if kind == "start":
        source += (
            "**Zuerst alle alten Runde-1b-Läufe in beiden Konten stoppen.** Dieses Notebook "
            "legt die getrennten Variantenordner in A und B an und übernimmt geprüfte, "
            "H-unabhängige Ergebnisse. Originale unter `dsse_runs` bleiben erhalten. "
            "Für Konto B ist die Verknüpfung `MyDrive/dsse_runs_B` mit Bearbeiterrecht nötig.\n\n"
            "Danach rechnet es **nur K_S1 ab den von H abhängigen Schritten neu** und stoppt. "
            "Das bereits gerechnete Screening wird übernommen. Erst nach erfolgreichem "
            "Abschluss mit den weiteren GPU-Notebooks fortfahren.\n\n"
            "**Hidden States mitschneiden:** standardmäßig eingeschaltet. Bei K3a, "
            "nach `k2_sets`, werden für 256 festgelegte Requests (128 S3, 64 A-det, "
            "64 NE) die Zustände am letzten Prompttoken aus allen 28 Blöcken "
            "als Float32 gespeichert (etwa 56 MiB). Die Vorbereitung schneidet "
            "keine Modellzustände mit. `HIDDEN_STATES_MITSCHNEIDEN = False` schaltet "
            "den zusätzlichen Mitschnitt aus. Bereits erledigte Schritte werden "
            "dafür nicht erneut gerechnet. Vor der tatsächlichen K3a-Generierung "
            "prüft der Lauf Logit-Gleichheit und zusätzlichen Aufwand; die Messung "
            "steht unter `observations/hidden_states_v1/benchmark.json`.\n\n"
        )
    else:
        source += (
            "**Voraussetzung:** `HTQ_START` wurde erfolgreich abgeschlossen. "
            "Dieses Notebook prüft die Variantenordner vor dem Kopieren und Starten. "
            "K_S4 wird einschließlich des früher mit H+ gerechneten K8b neu gerechnet; "
            "alte K/L-Ausgaben und Caches werden nicht als H_tq-Ergebnisse weiterverwendet.\n\n"
        )
    if kind in ("start", "in_a"):
        source += (
            "Schreibt in Konto B über `dsse_runs_B/variants/htq_v2`; "
            "die ursprünglichen B1/B2-Notebooks dürfen nicht parallel laufen.\n\n"
        )
    source += (
        "**Eine GPU:** START → B2_in_A → B1_in_A → A1_DC_zuerst → A2 → ANA. "
        "L-CAL separat auf CPU mit hohem RAM; `l_cal.json` vor ANA übernehmen. "
        "Danach die finanzierten Zusatzläufe und den Zusatzbericht abschließen. "
        "Höchstens zwei Laufzeiten je Konto gleichzeitig.\n\n"
        "**Ausführen:** Laufzeittyp einstellen, dann „Alle ausführen“. "
        "Bei Abbruch dasselbe H_tq-Notebook erneut ausführen; erfolgreiche Variantenschritte "
        "werden übersprungen. Ergebnisse liegen unter `dsse_runs/variants/htq_v2`.\n"
    )
    return source


def drive_code(kind):
    in_a = kind in ("start", "in_a")
    source = '''# Getrennte Variantenordner; originale Ergebnisordner werden nur gelesen.
from google.colab import drive
drive.mount("/content/drive")
MYDRIVE = "/content/drive/MyDrive"
'''
    if in_a:
        source += '''SOURCE_A = f"{MYDRIVE}/dsse_runs"
SOURCE_B = f"{MYDRIVE}/dsse_runs_B"
assert os.path.isdir(SOURCE_A), "dsse_runs von Konto A fehlt."
assert os.path.isdir(SOURCE_B), "Verknüpfung dsse_runs_B fehlt (Bearbeiterrecht für Konto A nötig)."
SOURCE_RUNS = SOURCE_B
'''
    else:
        source += '''SOURCE_RUNS = f"{MYDRIVE}/dsse_runs"
assert os.path.isdir(SOURCE_RUNS), "Der originale dsse_runs-Ordner fehlt."
'''
        if kind == "a":
            source += '''SOURCE_A = SOURCE_RUNS
SOURCE_B = f"{MYDRIVE}/dsse_runs_B"
assert os.path.isdir(SOURCE_B), "Verknüpfung dsse_runs_B fehlt."
'''
    source += f'''RUNS = os.path.join(SOURCE_RUNS, "{SUBDIR}")
WORK, ROOT, REPO = "/content/work_htq_v2", "/content/root", "/content/DSSE"
os.makedirs(WORK, exist_ok=True); os.makedirs(ROOT, exist_ok=True)
# Nur für reale Unterprozesse; MOCK-Selbsttests erhalten keine Drive-/Variantenvariablen.
_variant_keys = ("R1B_H_POLICY", "R1B_RUNS_B", "R1B_XFER_A", "R1B_SOURCE_RUNS", "R1B_CAPTURE_HIDDEN")
TEST_ENV = {{k: v for k, v in os.environ.items() if k not in _variant_keys}}
VARIANT_ENV = dict(TEST_ENV, R1B_H_POLICY="{VARIANT}")
'''
    if kind == "b":
        source += '''LINK = f"{MYDRIVE}/r1b_xfer_A"
VARIANT_ENV["R1B_XFER_A"] = LINK
B_RUNS = RUNS
'''
    else:
        source += f'''B_RUNS = os.path.join(SOURCE_B, "{SUBDIR}")
LINK = f"{{SOURCE_A}}/r1b_xfer"
VARIANT_ENV.update(R1B_RUNS_B=B_RUNS, R1B_XFER_A=LINK)
'''
    source += '''print("H_tq-Ergebnisse:", RUNS)
print("Unveränderte R_S0-Exporte:", LINK)
'''
    return source


def variant_helpers():
    return '''# Variantenwerkzeug: Ausgabe und Fehler mit Konto sichtbar streamen.
import textwrap
VARIANT_TOOL = f"{R1B}/r1b_htq_variant.py"
def run_variant(action, path, account):
    def show(message):
        for line in str(message).splitlines() or [""]:
            for wrapped in textwrap.wrap(line, width=72, replace_whitespace=False,
                                         drop_whitespace=False, break_on_hyphens=False) or [""]:
                print(wrapped, flush=True)
    show(f"[Konto {account}] {action}: {path}")
    command = [sys.executable, "-u", VARIANT_TOOL, action, path, account]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=TEST_ENV)
    except OSError as error:
        show(f"[Konto {account}] {action}: Start fehlgeschlagen: {error}")
        raise RuntimeError(f"Konto {account}: {action} konnte nicht starten.") from error
    try:
        for line in process.stdout:
            show(line.rstrip("\\r\\n"))
        code = process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        show(f"[Konto {account}] {action}: unterbrochen; Kindprozess beendet.")
        raise
    finally:
        process.stdout.close()
    show(f"[Konto {account}] {action}: Exit {code}")
    if code:
        raise RuntimeError(f"Konto {account}: {action} fehlgeschlagen (Exit {code}). Details oben.")
    return code

def check_variant(path, account):
    return run_variant("check", path, account)
'''


def prepare_code(kind, start_message="Beide Variantenordner bereit. Nun wird nur K_S1 neu gerechnet."):
    source = '# Varianten-Bootstrap / Prüfung vor jedem realen Kopier- oder Hauptlauf.\n' + variant_helpers()
    if kind == "start":
        source += f'''if not NUR_SELBSTTEST:
    for original, account in ((SOURCE_A, "A"), (SOURCE_B, "B")):
        run_variant("seed", original, account)
        check_variant(os.path.join(original, "{SUBDIR}"), account)
    paired = dict(variant="{VARIANT}", accounts=["A", "B"], bootstrap_commit=COMMIT)
    for path in (os.path.join(SOURCE_A, "{SUBDIR}"), B_RUNS):
        marker = os.path.join(path, "{PAIR_MARKER}")
        with open(marker + ".tmp", "w") as f:
            json.dump(paired, f, indent=1)
        os.replace(marker + ".tmp", marker)
    print({start_message!r}, flush=True)
'''
    else:
        source += f'''if not NUR_SELBSTTEST:
    check_variant(RUNS, KONTO)
'''
        if kind != "b":
            source += '''    check_variant(B_RUNS, "B")
'''
        source += f'''    pair = json.load(open(os.path.join(RUNS, "{PAIR_MARKER}")))
    assert pair.get("variant") == "{VARIANT}" and pair.get("accounts") == ["A", "B"], "HTQ_START muss beide Konten vorbereiten."
    done = os.path.join(B_RUNS, "r1b_K_S1", "session_done.json")
    partial = os.path.join(B_RUNS, "r1b_K_S1", "session_partial.json")
    assert os.path.isfile(done) and not os.path.exists(partial), "Zuerst HTQ_START bis zum fertigen neuen K_S1 ausführen."
'''
    return source


def logged_prep_code(program=None):
    """Keep the last preparation cell useful even when Colab cannot render output."""
    if program is None:
        program = prepare_code("start", "Beide Konten vorbereitet und geprüft. Jetzt HTQ_START auf der A100 ausführen.")
    return f'''# Dauerhaftes Diagnoseprotokoll: diese letzte Zelle enthält die Vorbereitung.
import contextlib, json, os, traceback, uuid
from datetime import datetime, timezone
_prep_program = {program!r}
_prep_source = globals().get("SOURCE_A") or "/content/drive/MyDrive/dsse_runs"
_prep_now = lambda: datetime.now(timezone.utc).isoformat()
_prep_status = dict(run_id=uuid.uuid4().hex[:12], started_utc=_prep_now(),
                    finished_utc=None, status="running", stage="compile", error=None,
                    commit=globals().get("COMMIT") if isinstance(globals().get("COMMIT"), str) else None)
try:
    if not os.path.isdir(_prep_source):
        raise NotADirectoryError("Der bestehende dsse_runs-Ordner von Konto A fehlt; Drive zuerst verbinden.")
    HTQ_PREP_LOG = os.path.join(_prep_source, "htq_prep_check.log")
    HTQ_PREP_STATUS = os.path.join(_prep_source, "htq_prep_check_status.json")
    _prep_log = open(HTQ_PREP_LOG, "a", encoding="utf-8", buffering=1)
except Exception as error:
    print("Protokoll konnte NICHT in Drive geöffnet werden:", str(error), flush=True)
    raise RuntimeError("Drive-Protokoll nicht gespeichert. Konto-A-Ablage prüfen.") from None

def _prep_save_status():
    temporary = HTQ_PREP_STATUS + ".tmp-" + _prep_status["run_id"]
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(_prep_status, output, ensure_ascii=False, indent=1)
    os.replace(temporary, HTQ_PREP_STATUS)

print("Vorbereitung läuft. Vollständige Ausgabe wird in Drive gespeichert.", flush=True)
print(HTQ_PREP_LOG, flush=True)
_prep_status_saved = False
with _prep_log:
    with contextlib.redirect_stdout(_prep_log), contextlib.redirect_stderr(_prep_log):
        print("\\n=== HTQ PREP", _prep_status["run_id"], _prep_status["started_utc"], "===", flush=True)
        print("Code-Commit:", _prep_status["commit"] or "noch nicht bekannt", flush=True)
        try:
            _prep_save_status()  # Ein alter Erfolgsstatus darf nicht als aktueller gelten.
            _prep_compiled = compile(_prep_program, "<HTQ-Vorbereitung>", "exec")
            _prep_status["stage"] = "bootstrap"
            _prep_save_status()
            exec(_prep_compiled, globals())
        except BaseException as error:
            _prep_status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                error_type=type(error).__name__, error=(type(error).__name__ + ": " + str(error))[:500])
            traceback.print_exc()
        else:
            _prep_status.update(status="success", stage="complete")
        finally:
            _prep_status["finished_utc"] = _prep_now()
            try:
                _prep_save_status()
                _prep_status_saved = True
            except Exception:
                print("Statusdatei konnte NICHT aktualisiert werden:", flush=True)
                traceback.print_exc()
            print("Ergebnis:", _prep_status["status"], "Lauf:", _prep_status["run_id"], flush=True)
print("Vorbereitung:", _prep_status["status"], flush=True)
print("Vollständiges Protokoll:", HTQ_PREP_LOG, flush=True)
if _prep_status_saved:
    print("Statusdatei:", HTQ_PREP_STATUS, flush=True)
else:
    print("Statusdatei NICHT aktualisiert; bitte das Protokoll lesen.", flush=True)
if _prep_status["status"] != "success":
    raise RuntimeError("Vorbereitung fehlgeschlagen. Ursache in htq_prep_check.log.") from None
print("Jetzt HTQ_START auf der A100 ausführen.", flush=True)
'''


def prep_check():
    name = "HTQ_PREP_CHECK"
    doc = dict(cells=[
        cell("markdown", '''# Runde 1b · Vorbereitung prüfen und fortsetzen · CPU, Konto A

**In Konto A auf einer CPU-Laufzeit öffnen, dann „Alle ausführen“.** Vorher alte
Runde-1b-Notebooks in beiden Konten stoppen. `dsse_runs_B` muss in Konto A als
Verknüpfung mit Bearbeiterrecht erreichbar sein.

Dieses Notebook lädt den aktuellen Vorbereitungscode in einen neuen temporären
Ordner unter `/content`. Es prüft beziehungsweise vervollständigt dieselben
Variantenordner wie HTQ_START. Eine vollständig vorbereitete Ablage von Konto A
wird wiederverwendet. Originalergebnisse und vorhandene Sperren bleiben erhalten.

Die letzte Zelle speichert die vollständige Vorbereitungsausgabe einschließlich
Fehler in Konto A unter `dsse_runs/htq_prep_check.log`. Den aktuellen Laufstatus
mit UTC-Zeit enthält `dsse_runs/htq_prep_check_status.json`. So bleibt die Ursache
lesbar, wenn Colab die Zellenausgabe auf dem Telefon nicht anzeigt. Im Notebook
erscheinen nur kurze Statusmeldungen. Die vorherigen Zellen zum Verbinden von
Drive und Laden des Codes liegen außerhalb dieses Diagnoseprotokolls.
Dieses Notebook lädt keine Modelle und startet keine GPU-Messung.

Nach erfolgreicher Vorbereitung beider Konten **HTQ_START auf der A100** ausführen.
Bei einem Fehler das Protokoll in Drive öffnen. Sperren oder Variantenordner
nicht auf Verdacht löschen.
'''),
        cell("code", f'''import os, sys, json, subprocess, tempfile
from google.colab import drive
drive.mount("/content/drive")
MYDRIVE = "/content/drive/MyDrive"
SOURCE_A = f"{{MYDRIVE}}/dsse_runs"
SOURCE_B = f"{{MYDRIVE}}/dsse_runs_B"
assert os.path.isdir(SOURCE_A), "Konto A: dsse_runs fehlt."
assert os.path.isdir(SOURCE_B), "Konto B: Verknüpfung dsse_runs_B fehlt; Freigabe und Bearbeiterrecht prüfen."
B_RUNS = os.path.join(SOURCE_B, "{SUBDIR}")
BRANCH = "{BRANCH}"
NUR_SELBSTTEST = False
_variant_keys = ("R1B_H_POLICY", "R1B_RUNS_B", "R1B_XFER_A", "R1B_SOURCE_RUNS", "R1B_CAPTURE_HIDDEN")
TEST_ENV = {{k: v for k, v in os.environ.items() if k not in _variant_keys}}
'''),
        cell("code", '''# Neue Arbeitskopie: vorhandene /content/DSSE-Repositories werden nicht verändert.
REPO = tempfile.mkdtemp(prefix="dsse_htq_prep_", dir="/content")
print("Aktuellen Vorbereitungscode laden:", REPO, flush=True)
subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--sparse", "-b", BRANCH,
                "https://github.com/Erikiss/DSSE", REPO], check=True, env=TEST_ENV)
subprocess.run(["git", "-C", REPO, "sparse-checkout", "set", "analysis/ju_space",
                "results/ju_space/plans"], check=True, env=TEST_ENV)
COMMIT = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
                        text=True, check=True, env=TEST_ENV).stdout.strip()
R1B = f"{REPO}/analysis/ju_space/round1b"
print("Vorbereitungscode:", COMMIT, flush=True)
'''),
        cell("code", logged_prep_code()),
    ], metadata={"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}},
        nbformat=4, nbformat_minor=5)
    finish(doc, name)
    return doc


def hidden_capture_status_code():
    return '''# Ein Mitschnitt gilt nur mit passendem K3a-Schritt als bestätigt.
def hidden_capture_status(runs):
    import hashlib
    root = os.path.join(runs, "r1b_K_S1")
    manifest_path = os.path.join(root, "observations", "hidden_states_v1", "manifest.json")
    result = dict(path=manifest_path, complete=False)
    if not os.path.isfile(manifest_path):
        return dict(result, reason="Manifest fehlt; die Aufzeichnung beginnt erst bei K3a.")
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        with open(os.path.join(root, "steps", "k3a.json"), encoding="utf-8") as f:
            step = json.load(f)
    except (OSError, ValueError) as error:
        return dict(result, reason="Manifest oder K3a-Abschluss ist nicht lesbar: " + str(error))
    if not isinstance(manifest, dict) or not isinstance(step, dict):
        return dict(result, reason="Manifest oder K3a-Abschluss hat ein unbekanntes Format.")
    if manifest.get("schema") != "prefill_v1" or manifest.get("status") != "complete" or manifest.get("request_count") != 256:
        return dict(result, reason="Schema, Abschlussstatus oder Requestzahl ist nicht bestätigt.")
    if manifest.get("layer_ids") != list(range(28)):
        return dict(result, reason="Die 28 erwarteten Modellblöcke sind nicht bestätigt.")
    provenance = manifest.get("provenance") or {}
    if not isinstance(provenance, dict) or step.get("status") != "ok" or step.get("name") != "k3a":
        return dict(result, reason="Der zugehörige erfolgreiche K3a-Schritt ist nicht bestätigt.")
    if provenance.get("session") != "r1b_K_S1" or provenance.get("step") != "k3a":
        return dict(result, reason="Der Mitschnitt ist keiner passenden K3a-Sitzung zugeordnet.")
    if any(not provenance.get(key) or provenance.get(key) != step.get(key) for key in ("attempt", "code_commit")):
        return dict(result, reason="Mitschnitt und K3a-Schritt gehören nicht nachweislich zum selben Versuch und Code.")
    families, outputs = manifest.get("families"), step.get("outputs")
    expected_counts = {"s3": 128, "adet": 64, "ne": 64}
    if not isinstance(families, dict) or set(families) != set(expected_counts) or not isinstance(outputs, dict):
        return dict(result, reason="Die drei erwarteten Artefakte oder ihre Schrittprüfsummen fehlen.")
    def digest(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(8 << 20), b""):
                h.update(block)
        return h.hexdigest()
    directory = "observations/hidden_states_v1"
    try:
        for family, count in expected_counts.items():
            rel = f"{directory}/{family}.pt"
            saved = families[family]
            record = outputs.get(rel)
            if not isinstance(saved, dict) or saved.get("file") != rel or saved.get("count") != count or saved.get("shape") != [count, 28, 2048]:
                return dict(result, reason=f"Unerwarteter Pfad, Umfang oder Tensorform für {family}.")
            expected = saved.get("sha256")
            if not isinstance(record, dict) or not expected or record.get("sha256") != expected:
                return dict(result, reason=f"Manifest und K3a-Prüfsumme stimmen für {family} nicht überein.")
            path = os.path.join(root, rel)
            if not os.path.isfile(path) or digest(path) != expected:
                return dict(result, reason=f"Artefakt {family} fehlt in Drive oder seine Prüfsumme stimmt nicht.")
        benchmark_rel = f"{directory}/benchmark.json"
        if manifest.get("benchmark") != benchmark_rel:
            return dict(result, reason="Der erwartete GPU-Prüfbericht ist nicht zugeordnet.")
        benchmark_path = os.path.join(root, benchmark_rel)
        with open(benchmark_path, encoding="utf-8") as f:
            benchmark = json.load(f)
        if not isinstance(benchmark, dict) or benchmark.get("schema") != "prefill_v1" or benchmark.get("status") != "passed" or benchmark.get("logits_exact") is not True:
            return dict(result, reason="Die GPU-Prüfung bestätigt keine exakte Logit-Gleichheit.")
        benchmark_provenance = benchmark.get("provenance") or {}
        if not isinstance(benchmark_provenance, dict) or any(benchmark_provenance.get(key) != provenance.get(key) for key in ("session", "step", "attempt", "code_commit")):
            return dict(result, reason="GPU-Prüfbericht und Mitschnitt gehören nicht zum selben K3a-Versuch.")
        benchmark_record = outputs.get(benchmark_rel)
        if not isinstance(benchmark_record, dict) or digest(benchmark_path) != benchmark_record.get("sha256"):
            return dict(result, reason="Die Schrittprüfsumme des GPU-Prüfberichts stimmt nicht.")
    except (OSError, ValueError) as error:
        return dict(result, reason="Artefaktprüfung in Drive fehlgeschlagen: " + str(error))
    return dict(result, complete=True, reason="256 Requests, 28 Blöcke, passende K3a-/GPU-Prüfung und drei geprüfte Drive-Artefakte.")
'''


def start_main():
    return hidden_capture_status_code() + '''# Nur das neue K_S1; anschließend wird die GPU für B2 frei.
MELDUNG = {0: "H_tq-K_S1 fertig. Als Nächstes HTQ_B2_in_A starten.",
           1: "K_S1 fehlgeschlagen; Variantenprotokolle prüfen und dieses Notebook erneut ausführen.",
           2: "Eine Freigabe ist nicht erfüllt; manifest.json prüfen.",
           3: "Laufzeitgrenze erreicht; HTQ_START in frischer Laufzeit erneut ausführen.",
           4: "Startbedingungen fehlen.", 5: "K_S1 wurde nicht gerechnet; Protokolle prüfen."}
RC = None
if NUR_SELBSTTEST:
    print("NUR_SELBSTTEST = True: kein Hauptlauf.")
else:
    check_variant(RUNS, "B")
    env = dict(VARIANT_ENV, R1B_ACCOUNT="B", R1B_SLOT="B1", R1B_ROOT=ROOT,
               RUN_DEVICE="cuda", NT="8", R1B_RUNTIME_T0=repr(T0),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    env.pop("R1B_CAPTURE_HIDDEN", None)
    if HIDDEN_STATES_MITSCHNEIDEN:
        env["R1B_CAPTURE_HIDDEN"] = "1"
    cmd = [sys.executable, "-u", f"{R1B}/r1b_run_session.py", "r1b_K_S1", ROOT, f"{WORK}/r1b_K_S1", RUNS]
    while True:
        if time.time() - T0 >= LAUFZEIT_MAX_H * 3600:
            RC = 3; break
        RC = subprocess.run(cmd, env=dict(env, R1B_T0=repr(time.time()))).returncode
        if RC == 3:
            print("Sitzungszeitgrenze: K_S1 wird aus den Variantenzwischenständen fortgesetzt.")
            continue
        break
    if RC == 0:
        assert os.path.isfile(f"{RUNS}/r1b_K_S1/session_done.json"), "K_S1 hat keine fertige Sitzungsmarke geschrieben."
        with open(f"{RUNS}/r1b_K_S1/hn/hn_set.json") as f:
            hs = json.load(f)
        assert hs.get("H_op_name") in ("H_tq", "H_ar"), "K_S1 hat unerwartet H+ als operative Menge verwendet."
        print("Neue operative Menge:", hs["H_op_name"], "mit", len(hs.get("H_op") or []), "Neuronen.")
    print(MELDUNG.get(RC, f"Unerwarteter Rückgabewert {RC}"))
    if HIDDEN_STATES_MITSCHNEIDEN:
        observation = hidden_capture_status(RUNS)
        print("Hidden-State-Manifest:", observation["path"])
        print("Mitschnitt bestätigt vollständig." if observation["complete"] else "Mitschnitt noch nicht bestätigt:", observation["reason"])
        if not observation["complete"]:
            print("Fertige Schritte werden für einen fehlenden Mitschnitt nicht erneut gerechnet.")
        print("GPU-Prüfung und Zusatzaufwand: r1b_K_S1/observations/hidden_states_v1/benchmark.json")
    else:
        print("Zusätzlicher Hidden-State-Mitschnitt ist ausgeschaltet.")
'''


def transform(name, spec, source_dir=HERE):
    original, title, kind = spec
    doc = json.loads((source_dir / f"runde1b_{original}.ipynb").read_text())
    doc = copy.deepcopy(doc)
    set_text(doc["cells"][0], intro(title, kind))
    config = text(doc["cells"][1]).replace('BRANCH = "ccr-6fe02b21-est040"', f'BRANCH = "{BRANCH}"')
    if original == "B1_in_A":
        config = config[:config.index("if GB < 70:")] if "if GB < 70:" in config else config
    if kind == "start":
        config = config.replace('SELBSTTEST = False', 'HIDDEN_STATES_MITSCHNEIDEN = True   # Optionaler Mitschnitt bei K3a; 256 Requests, ca. 56 MiB\nSELBSTTEST = False', 1)
    set_text(doc["cells"][1], config)
    set_text(doc["cells"][2], drive_code(kind))
    set_text(doc["cells"][5], text(doc["cells"][5]).replace("dict(os.environ, R1B_ACCOUNT=KONTO)", "dict(VARIANT_ENV, R1B_ACCOUNT=KONTO)"))
    set_text(doc["cells"][6], text(doc["cells"][6]).replace("env=dict(os.environ, RUN_DEVICE=", "env=dict(TEST_ENV, RUN_DEVICE="))
    if kind == "start":
        set_text(doc["cells"][5], '# HTQ_START rechnet nur K_S1; kein vorgezogener Kopierlauf des gesamten B1-Slots.\n')
        set_text(doc["cells"][7], start_main())
        summary = text(doc["cells"][8]).replace("['r1b_K_S1', 'r1b_K_S4', 'r1b_L_A', 'r1b_L_B']", "['r1b_K_S1']")
        set_text(doc["cells"][8], summary)
    else:
        main = text(doc["cells"][7]).replace("dict(os.environ, R1B_ACCOUNT=KONTO", "dict(VARIANT_ENV, R1B_ACCOUNT=KONTO")
        set_text(doc["cells"][7], main)
    flush = text(doc["cells"][9]).replace("dsse_runs/<Sitzung>", f"dsse_runs/{SUBDIR}/<Sitzung>")
    set_text(doc["cells"][9], flush)
    doc["cells"].insert(4, cell("code", prepare_code(kind)))
    finish(doc, name)
    return doc


def lcal(source_dir=HERE):
    name = "HTQ_LCAL"
    doc = json.loads((source_dir / "runde1b_LCAL.ipynb").read_text())
    set_text(doc["cells"][0], '''# Runde 1b · L-CAL auf CPU mit hohem RAM · H_tq-Variante

In Konto A, in einer eigenen CPU-Laufzeit mit hohem RAM öffnen. Erst starten,
nachdem HTQ_START die beiden Variantenordner als bereit gemeldet hat; danach
kann L-CAL parallel zum GPU-Lauf rechnen. Der eingefrorene L-CAL-Code und die
synthetischen Kalibrierungsdaten bleiben unverändert. Für den aktuellen
Generator ist keine belastbare Laufzeit gemessen; mehrere Stunden einplanen.

Ergebnis: `dsse_runs/variants/htq_v2/r1b_lcal/l_cal.json`. Diese Datei vor ANA
dem Koordinator zur Übernahme melden. Zwischenstände bleiben im Varianten-Cache.
''')
    source = text(doc["cells"][1])
    source = source.replace("import os, sys, subprocess, json, glob, time, hashlib", "import os, sys, subprocess, json, glob, time, hashlib, shutil")
    source = source.replace('OUT = "/content/drive/MyDrive/dsse_runs/r1b_lcal"; CACHE = f"{OUT}/cache"; os.makedirs(CACHE, exist_ok=True)\n', "")
    source = source.replace('REPO = "/content/DSSE";', 'REPO = "/content/DSSE_htq_lcal";')
    source = source.replace('if not os.path.exists(REPO):\n', 'if os.path.exists(REPO):\n    shutil.rmtree(REPO)\nif not os.path.exists(REPO):\n')
    source = source.replace('"ccr-6fe02b21-est040"', f'"{BRANCH}"')
    source = source.replace('capture_output=True, text=True).stdout', 'capture_output=True, text=True, check=True).stdout')
    source += f'''
RUNS = "/content/drive/MyDrive/dsse_runs/{SUBDIR}"
clean_env = {{k: v for k, v in os.environ.items() if k not in ("R1B_H_POLICY", "R1B_RUNS_B", "R1B_XFER_A", "R1B_SOURCE_RUNS", "R1B_CAPTURE_HIDDEN")}}
R1B, TEST_ENV = R, clean_env
'''
    source += variant_helpers()
    source += f'''
check_variant(RUNS, "A")
pair = json.load(open(os.path.join(RUNS, "{PAIR_MARKER}")))
assert pair.get("variant") == "{VARIANT}" and pair.get("accounts") == ["A", "B"], "Zuerst HTQ_START vorbereiten."
OUT = f"{{RUNS}}/r1b_lcal"; CACHE = f"{{OUT}}/cache"; os.makedirs(CACHE, exist_ok=True)
'''
    set_text(doc["cells"][1], source)
    set_text(doc["cells"][2], text(doc["cells"][2]).replace("env=dict(os.environ, OMP_NUM_THREADS=thr)", "env=dict(clean_env, OMP_NUM_THREADS=thr)"))
    finish(doc, name)
    return doc


def finish(doc, name):
    for i, c in enumerate(doc["cells"]):
        c["id"] = hashlib.sha256(f"{name}:{i}".encode()).hexdigest()[:12]
        if c["cell_type"] == "code":
            c.update(execution_count=None, outputs=[])
            ast.parse(text(c), filename=f"runde1b_{name}.ipynb:cell{i}")
    doc["metadata"]["htq_variant"] = dict(name=VARIANT, branch=BRANCH, post_hoc=True)


def build(out=HERE, source_dir=HERE):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    built = []
    for name, spec in SPECS.items():
        doc = transform(name, spec, source_dir)
        path = out / f"runde1b_{name}.ipynb"
        path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        built.append(path)
    path = out / "runde1b_HTQ_LCAL.ipynb"
    path.write_text(json.dumps(lcal(source_dir), indent=1, ensure_ascii=False) + "\n")
    built.append(path)
    path = out / "runde1b_HTQ_PREP_CHECK.ipynb"
    path.write_text(json.dumps(prep_check(), indent=1, ensure_ascii=False) + "\n")
    built.append(path)
    return built


if __name__ == "__main__":
    for path in build():
        print(path.name)
