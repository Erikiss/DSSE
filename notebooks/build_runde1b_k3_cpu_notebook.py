"""Build a CPU-only, read-only K3a analysis notebook; helpers are testable offline."""
import ast
import hashlib
import inspect
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRANCH = "codex/r1b-htq-continuation"
REL = "k/k3/k3a_data.pt"
CONTEXT = ("session_done.json", "hn/htq_policy.json", "hn/hn_set.json", "sg8_preview.json")
OPTIONAL = (REL + ".prov.json", "steps/k3a.json")


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checked_context(blobs):
    """Check original bytes, session/policy and producer hashes before any pickle."""
    import hashlib
    import json
    import re
    done = json.loads(blobs["session_done.json"])
    if done.get("session") != "r1b_K_S1" or done.get("mock") is not False or done.get("h_policy") != "htq_v2":
        raise ValueError("Kein echter abgeschlossener H_tq-K_S1-Datensatz.")
    expected = done.get("files", {}).get(REL)
    if not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected):
        raise ValueError("Die K3a-Prüfsumme fehlt im Abschlussmarker.")
    if REL + ".prov.json" in done.get("files", {}) and REL + ".prov.json" not in blobs:
        raise ValueError("Der hashgebundene K3a-Provenienz-Sidecar fehlt.")
    for rel, raw in blobs.items():
        if rel == "session_done.json":
            continue
        expected_context = done.get("files", {}).get(rel)
        if expected_context is not None and hashlib.sha256(raw).hexdigest() != expected_context:
            raise ValueError("Kontext-Prüfsumme stimmt nicht: " + rel)
        if rel in CONTEXT and expected_context is None:
            raise ValueError("Kontextdatei fehlt im Abschlussmarker: " + rel)
    policy = json.loads(blobs["hn/htq_policy.json"])
    hn = json.loads(blobs["hn/hn_set.json"])
    preview = json.loads(blobs["sg8_preview.json"])
    if any(value.get("h_policy") != "htq_v2" or value.get("post_hoc") is not True for value in (policy, hn)):
        raise ValueError("Keine dokumentierte post-hoc H_tq-Policy im Quellkontext.")
    name = hn.get("H_op_name")
    if name != "H_tq" or policy.get("H_op_name") != name or preview.get("H_op_name") != name:
        raise ValueError("Die H-Auswahl ist zwischen den Quelldateien widersprüchlich.")
    producer = (done.get("steps", {}).get("k3a", {}).get("outputs") or {}).get(REL)
    if producer and producer.get("sha256") != expected:
        raise ValueError("K3a-Schritt und Abschlussmarker nennen verschiedene Prüfsummen.")
    if REL + ".prov.json" in blobs:
        sidecar = json.loads(blobs[REL + ".prov.json"])
        if sidecar.get("sha256") != expected:
            raise ValueError("K3a-Sidecar und Abschlussmarker nennen verschiedene Prüfsummen.")
    return dict(sha256=expected, done=done, blobs=blobs)


def mounted_candidates(base):
    from pathlib import Path
    found = []
    for account in ("dsse_runs_B", "dsse_runs"):
        session = Path(base) / account / "variants/htq_v2/r1b_K_S1"
        if not (session / REL).is_file():
            continue
        partial = (session / "session_partial.json").exists()
        try:
            blobs = {rel: (session / rel).read_bytes() for rel in CONTEXT}
            blobs.update({rel: (session / rel).read_bytes() for rel in OPTIONAL if (session / rel).is_file()})
            found.append(dict(checked_context(blobs), kind="mounted", session=session,
                              source=str(session / REL), size=(session / REL).stat().st_size,
                              source_partial_present=partial))
        except (OSError, ValueError, KeyError) as error:
            print("Quellkontext abgelehnt:", str(session), str(error), flush=True)
    return found


def local_context_zip(path):
    """Read only the small source context; never extract or load tensor members."""
    import json
    from pathlib import PurePosixPath
    import stat
    import zipfile
    blobs = {}
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 100 or sum(item.file_size for item in members) > 32 << 20:
            raise ValueError("Kontext-ZIP ist unerwartet groß; nur den kleinen Quellkontext verwenden.")
        for member in members:
            parts = PurePosixPath(member.filename).parts
            if not parts or member.filename.startswith(("/", "\\")) or ".." in parts or "\\" in member.filename:
                raise ValueError("Ungültiger Pfad im Kontext-ZIP.")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Verknüpfung im Kontext-ZIP ist nicht zulässig.")
            if member.is_dir():
                continue
            if parts[0] == "r1b_K_S1":
                parts = parts[1:]
            rel = "/".join(parts)
            if rel not in CONTEXT + OPTIONAL:
                continue
            if rel in blobs:
                raise ValueError("Doppelte Kontextdatei im ZIP: " + rel)
            blobs[rel] = archive.read(member)
    if any(rel not in blobs for rel in CONTEXT):
        raise ValueError("ZIP enthält keinen vollständigen K3-Quellkontext.")
    return dict(checked_context(blobs), context_zip=str(path))


def local_source(upload_dir, fallback_dir="/content", context_zip="", input_file=""):
    """Resolve explicit local uploads without Drive imports, authentication or retry."""
    from pathlib import Path
    import re
    roots = list(dict.fromkeys((Path(upload_dir), Path(fallback_dir))))
    if input_file:
        raw_files = [Path(input_file)]
        if not raw_files[0].is_file():
            raise FileNotFoundError("INPUT_FILE fehlt: " + str(raw_files[0]))
        if raw_files[0].parent not in roots:
            roots.append(raw_files[0].parent)
    else:
        pattern = re.compile(r"k3a_data(?: \(\d+\))?\.pt(?: \(\d+\))?$")
        raw_files = sorted({path for root in roots if root.is_dir() for path in root.iterdir()
                            if path.is_file() and pattern.fullmatch(path.name)})
    if not raw_files:
        raise FileNotFoundError("Lokale Rohdatei fehlt. Links im Colab-Dateibrowser k3a_data.pt nach "
                                + str(upload_dir) + " oder " + str(fallback_dir)
                                + " hochladen, Upload vollständig abwarten und diese Zelle erneut starten. Keine Drive-Anmeldung wird versucht.")
    if context_zip:
        paths = [Path(context_zip)]
        if not paths[0].is_file():
            raise FileNotFoundError("KONTEXT_ZIP fehlt: " + str(paths[0]))
    else:
        paths = sorted({p for root in roots if root.is_dir() for p in root.glob("*.zip")})
    contexts, rejected = [], []
    for path in paths:
        try:
            contexts.append(local_context_zip(path))
        except Exception as error:
            rejected.append(path.name + ": " + str(error))
            if context_zip:
                raise
    if not contexts:
        detail = " · ".join(rejected[:3])
        raise FileNotFoundError("Kleiner K3-Quellkontext fehlt oder ist ungültig. Das bereitgestellte htq_k3_context.zip neben k3a_data.pt hochladen; nicht die Diagnose-ZIP verwenden. " + detail)
    source = choose_source(contexts)
    # With two uploads, select only an exact producer-hash match; one upload is
    # verified by stage_source before the analysis process can load it.
    if len(raw_files) > 1:
        raw_files = [path for path in raw_files if sha256(path) == source["sha256"]]
        if not raw_files:
            raise ValueError("Keine lokale Rohdatei stimmt mit der Kontext-SHA256 überein.")
    raw = raw_files[0]
    print("Lokale Eingabe gefunden; Drive-Anmeldung wird vollständig übersprungen.", flush=True)
    return dict(source, kind="local", source=str(raw), size=raw.stat().st_size)


def api_list(service, query):
    files, token = [], None
    for page in range(100):
        response = service.files().list(q=query, fields="files(id,name,mimeType,size,parents,trashed),nextPageToken",
                                        pageSize=100, pageToken=token, spaces="drive", supportsAllDrives=True,
                                        includeItemsFromAllDrives=True).execute()
        files.extend(response.get("files", []))
        token = response.get("nextPageToken")
        if not token:
            return files
    raise ValueError("Zu viele gleichnamige Dateien; DATEI_ID gezielt setzen.")


def api_meta(service, file_id):
    return service.files().get(fileId=file_id, supportsAllDrives=True,
                               fields="id,name,mimeType,size,parents,trashed").execute()


def api_session(service, file_id):
    meta = api_meta(service, file_id)
    if meta.get("trashed") or meta.get("name") != "k3a_data.pt" or meta.get("mimeType") == "application/vnd.google-apps.folder":
        raise ValueError("DATEI_ID ist keine K3a-Rohdatei.")
    current, session_id = meta, None
    for expected in ("k3", "k", "r1b_K_S1", "htq_v2", "variants"):
        parents = current.get("parents", [])
        if len(parents) != 1:
            raise ValueError("Quellpfad ist nicht eindeutig prüfbar.")
        current = api_meta(service, parents[0])
        if current.get("trashed") or current.get("name") != expected or current.get("mimeType") != "application/vnd.google-apps.folder":
            raise ValueError("Kanonischer H_tq-Quellpfad stimmt nicht.")
        if expected == "r1b_K_S1":
            session_id = current["id"]
    return session_id, meta


def api_child(service, parent, name, required=True):
    safe_name = name.replace("\\", "\\\\").replace("'", "\\'")
    rows = api_list(service, "'" + parent + "' in parents and trashed = false and name = '" + safe_name + "'")
    if not rows and not required:
        return None
    if len(rows) != 1:
        raise ValueError("Kontextdatei oder Ordner fehlt/ist mehrdeutig: " + name)
    return rows[0]


def api_context(service, file_id):
    session_id, meta = api_session(service, file_id)
    blobs, ids = {}, {}
    partial = api_child(service, session_id, "session_partial.json", required=False) is not None
    for rel in CONTEXT + OPTIONAL:
        current, missing = session_id, False
        parts = rel.split("/")
        for name in parts[:-1]:
            item = api_child(service, current, name, required=rel in CONTEXT)
            if item is None:
                missing = True
                break
            current = item["id"]
        if missing:
            continue
        item = api_child(service, current, parts[-1], required=rel in CONTEXT)
        if item is None:
            continue
        if item.get("mimeType") == "application/vnd.google-apps.folder" or int(item.get("size", 0)) > 16 << 20:
            raise ValueError("Unerwartete Kontextdatei: " + rel)
        raw = service.files().get_media(fileId=item["id"], supportsAllDrives=True).execute()
        if not isinstance(raw, bytes):
            raise ValueError("Kontext muss als unveränderte Datei lesbar sein: " + rel)
        blobs[rel], ids[rel] = raw, item["id"]
    return dict(checked_context(blobs), kind="api", file_id=file_id, source="Drive API: kanonischer H_tq-K_S1-Pfad",
                source_ids=ids, size=int(meta.get("size", 0)), session_id=session_id, source_partial_present=partial)


def api_candidates(service, file_id=""):
    if file_id:
        return [api_context(service, file_id)]
    found = []
    for item in api_list(service, "trashed = false and name = 'k3a_data.pt' and mimeType != 'application/vnd.google-apps.folder'"):
        try:
            found.append(api_context(service, item["id"]))
        except Exception as error:
            print("Gleichnamige Datei nicht als gültige H_tq-Quelle bestätigt:", str(error), flush=True)
    return found


def choose_source(candidates):
    if not candidates:
        raise ValueError("Keine gültige abgeschlossene K3a-Datei zugänglich. Konto/Freigabe prüfen oder DATEI_ID setzen.")
    hashes = {candidate["sha256"] for candidate in candidates}
    if len(hashes) != 1:
        raise ValueError("Mehrere gültige K3a-Dateien mit verschiedenen SHA256 gefunden. DATEI_ID gezielt setzen.")
    print("Bestätigte Quellkopien:", len(candidates), "· identische SHA256; erste Kopie wird gelesen.", flush=True)
    if candidates[0].get("source_partial_present"):
        print("Zusätzlicher partieller Marker vorhanden: geprüft wird ausschließlich die hashgebundene K3a-Datei. Sitzungsmarker werden nicht verändert.", flush=True)
    return candidates[0]


def stage_source(source, cache, service=None, downloader_factory=None):
    import json
    import os
    import shutil
    from pathlib import Path
    session = Path(cache) / source["sha256"] / "r1b_K_S1"
    target = session / REL
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and sha256(target) == source["sha256"]:
        print("Lokaler K3a-Cache ist vollständig und SHA-geprüft.", flush=True)
    else:
        temporary = target.with_suffix(".download")
        if source["kind"] in ("mounted", "local"):
            count, total = 0, source["size"]
            with open(source["source"], "rb") as src, open(temporary, "wb") as dst:
                for block in iter(lambda: src.read(16 << 20), b""):
                    dst.write(block); count += len(block)
                    print(f"K3a lokal kopiert: {count/(1<<20):.0f}/{total/(1<<20):.0f} MiB", flush=True)
        else:
            if service is None:
                raise ValueError("Drive-API-Verbindung fehlt.")
            if downloader_factory is None:
                from googleapiclient.http import MediaIoBaseDownload
                downloader_factory = MediaIoBaseDownload
            request = service.files().get_media(fileId=source["file_id"], supportsAllDrives=True)
            with open(temporary, "wb") as dst:
                transfer = downloader_factory(dst, request, chunksize=16 << 20)
                done = False
                while not done:
                    status, done = transfer.next_chunk(num_retries=2)
                    if status:
                        print(f"K3a heruntergeladen: {100*status.progress():.1f} %", flush=True)
        if source.get("size") and temporary.stat().st_size != source["size"]:
            raise ValueError("K3a-Download ist unvollständig; lokale Teildatei bleibt zur Diagnose erhalten.")
        print("Prüfe SHA256 der vollständigen Rohdatei vor dem Laden …", flush=True)
        if sha256(temporary) != source["sha256"]:
            raise ValueError("K3a-SHA256 stimmt nicht; Rohdatei wird nicht geladen.")
        os.replace(temporary, target)
    for rel, raw in source["blobs"].items():
        path = session / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        if path.read_bytes() != raw:
            raise OSError("Lokaler Kontext konnte nicht bestätigt werden: " + rel)
    (session / "notebook_source.json").write_text(json.dumps({k: v for k, v in source.items() if k not in ("blobs", "done", "session")}, default=str, indent=1))
    return session


HELPERS = (sha256, checked_context, mounted_candidates, local_context_zip, local_source, api_list, api_meta, api_session,
           api_child, api_context, api_candidates, choose_source, stage_source)


def main_source():
    return '''# CPU-Auswertung vorhandener Dateien; kein Modell und keine neue Antwort.
import os, sys, json, subprocess, tempfile, traceback, uuid, zipfile
from datetime import datetime, timezone
from pathlib import Path
OUTPUT = Path(AUSGABE_ROOT) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
OUTPUT.mkdir(parents=True, exist_ok=True)
log_path = OUTPUT / "notebook.log"
failure = None
stage = "cpu_check"
def notebook_status(state, **extra):
    value = dict(state=state, stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
    path = OUTPUT / "notebook_status.json"
    raw = json.dumps(value, ensure_ascii=False, indent=1) + "\\n"
    path.write_text(raw)
    if path.read_text() != raw:
        raise OSError("Lokaler Notebook-Status konnte nicht bestätigt werden.")
try:
    if os.environ.get("COLAB_GPU", "0") not in ("", "0"):
        raise RuntimeError("Bitte CPU als Laufzeittyp wählen; auf einer GPU wird diese Auswertung nicht gestartet.")
    try:
        gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        if gpu.returncode == 0 and gpu.stdout.strip():
            raise RuntimeError("GPU sichtbar: CPU-Laufzeit wählen, dann Alle ausführen.")
    except FileNotFoundError:
        pass
    print("Prüfe CPU-Pakete vor dem großen Download …", flush=True)
    import numpy, scipy, torch
    if torch.cuda.is_available():
        raise RuntimeError("Torch sieht eine GPU: bitte CPU-Laufzeit wählen.")
    print("Versionen: numpy", numpy.__version__, "· scipy", scipy.__version__, "· torch", torch.__version__, flush=True)
    notebook_status("running")
    stage = "source_context"
    source = None
    service = None
    mode = EINGABE_MODE.strip().upper()
    if mode not in ("AUTO", "LOKAL"):
        raise ValueError("EINGABE_MODE muss AUTO oder LOKAL sein.")
    if mode == "LOKAL":
        source = local_source(LOKAL_ORDNER, "/content", KONTEXT_ZIP, INPUT_FILE)
    elif not DATEI_ID:
        candidates = mounted_candidates("/content/drive/MyDrive")
        if candidates:
            source = choose_source(candidates)
    if source is None and mode == "AUTO":
        print("Anmeldung für direkten Drive-API-Lesezugriff; Drive-Mount wird nicht benötigt.", flush=True)
        from google.colab import auth
        auth.authenticate_user()
        import google.auth
        from googleapiclient.discovery import build
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive.readonly"])
        service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        source = choose_source(api_candidates(service, DATEI_ID.strip()))
    stage = "download_and_hash"
    notebook_status("running")
    SESSION = stage_source(source, CACHE_ROOT, service)
    stage = "code_checkout"
    notebook_status("running")
    REPO = tempfile.mkdtemp(prefix="dsse_k3cpu_", dir="/content")
    env = {k: v for k, v in os.environ.items() if not k.startswith("R1B_") and k not in ("MOCK", "QUICK")}
    env.update(MOCK="0", QUICK="0", CUDA_VISIBLE_DEVICES="", RUN_DEVICE="cpu", DEVICE="cpu",
               OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", OMP_NUM_THREADS="2")
    with open(log_path, "a", encoding="utf-8", buffering=1) as log:
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--sparse", "-b", BRANCH,
                        "https://github.com/Erikiss/DSSE", REPO], check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
        subprocess.run(["git", "-C", REPO, "sparse-checkout", "set", "analysis/ju_space", "results/ju_space/plans"],
                       check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    COMMIT = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    rel = "analysis/ju_space/round1b/r1b_k3_cpu_export.py"
    RUNNER = Path(REPO) / rel
    committed = subprocess.check_output(["git", "-C", REPO, "show", COMMIT + ":" + rel])
    if not RUNNER.is_file() or RUNNER.read_bytes() != committed:
        raise RuntimeError("Analysecode ist nicht bytegleich mit dem ausgecheckten Commit.")
    (OUTPUT / "notebook_context.json").write_text(json.dumps(dict(code_commit=COMMIT, runner_sha256=sha256(RUNNER),
        source_sha256=source["sha256"], source=source["source"], source_kind=source["kind"], input=str(SESSION / REL),
        gpu_used=False), ensure_ascii=False, indent=1))
    stage = "analysis"
    notebook_status("running", code_commit=COMMIT)
    print("Analysecode:", COMMIT, "· CPU · Fortschritt und Protokoll:", str(OUTPUT), flush=True)
    command = [sys.executable, "-u", str(RUNNER), "--input", str(SESSION / REL),
               "--session-dir", str(SESSION), "--outputdir", str(OUTPUT)]
    with open(log_path, "a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line.rstrip(), flush=True); log.write(line)
            code = process.wait()
        except BaseException:
            process.terminate(); process.wait()
            raise
    if code:
        raise RuntimeError("CPU-Auswertung angehalten; Exit " + str(code) + ". notebook.log prüfen.")
    result = json.loads((OUTPUT / "status.json").read_text())
    if result.get("state") != "complete":
        raise RuntimeError("Runner hat keinen vollständigen Abschlussstatus geschrieben.")
    stage = "complete"
    notebook_status("complete", code_commit=COMMIT)
    print("Auswertung fertig. Nur vorhandene Daten gelesen. Ergebnisordner:", str(OUTPUT), flush=True)
except BaseException as error:
    failure = error
    with open(log_path, "a", encoding="utf-8") as log:
        traceback.print_exc(file=log)
    notebook_status("failed", error_type=type(error).__name__, error=str(error))
    print("Angehalten:", type(error).__name__, str(error), flush=True)
    if stage == "source_context" and globals().get("mode") == "AUTO":
        print("Bei credential propagation den lokalen Start runde1b_HTQ_K3_LOKAL.ipynb verwenden: Rohdatei und Kontext-ZIP im Dateibrowser bereitstellen. DATEI_ID ersetzt die Anmeldung nicht.", flush=True)
        if "scope" in str(error).lower() or "insufficient" in str(error).lower():
            print("Der Google-Zugang benötigt die Berechtigung drive.readonly. Anmeldung und Dateifreigabe prüfen; es werden keine Freigaben automatisch geändert.", flush=True)
finally:
    archive = OUTPUT / "htq_k3_results.zip"
    if not archive.is_file():
        archive = OUTPUT / "htq_k3_diagnostic.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(OUTPUT.rglob("*")):
                if path.is_file() and path != archive and path.suffix != ".zip":
                    z.write(path, path.relative_to(OUTPUT))
    print("ZIP:", str(archive), flush=True)
    print("ZIP-SHA256:", sha256(archive), flush=True)
    try:
        from google.colab import files
        files.download(str(archive))
    except Exception as download_error:
        print("Automatischer Download nicht bestätigt; ZIP links im Colab-Dateibrowser herunterladen:", str(download_error), flush=True)
if failure is not None:
    raise RuntimeError("K3-CPU wurde angehalten. Diagnose-ZIP enthält Status und Protokoll.") from None
'''


def build(out=HERE, mode="AUTO"):
    if mode not in ("AUTO", "LOKAL"):
        raise ValueError("Unknown notebook input mode")
    intro = '''# H_tq · Offene K3-Auswertung aus vorhandenen Daten · CPU

**CPU-Laufzeit wählen, dann „Alle ausführen“. Keine GPU nötig.** Dieses Notebook
liest die vorhandene große K3a-Datei und ergänzt die noch offene Auswertung;
es lädt kein Sprachmodell und erzeugt keine neuen Antworten. L-CAL und vollständige
ANA sind dafür nicht erforderlich. Originale und laufende Experimente werden nicht verändert.

Bereits erreichbare Drive-Pfade werden zuerst gelesen. Sonst folgt eine gesonderte
Anmeldung für direkten Drive-API-Zugriff; `drive.mount` wird nicht ausgeführt.
Die automatische Suche akzeptiert nur abgeschlossene, hashgeprüfte H_tq-Quellen
im Pfad `variants/htq_v2/r1b_K_S1/k/k3`. Bei verschiedenen gültigen Kopien hält sie
an; dann optional die gewünschte private Datei-ID in der Konfiguration eintragen.
Eine Datei-ID ersetzt keine Zugriffsberechtigung oder erfolgreiche Anmeldung.
Bei `credential propagation` den [lokalen Start ohne Anmeldung](https://colab.research.google.com/github/Erikiss/DSSE/blob/codex/r1b-htq-continuation/notebooks/runde1b_HTQ_K3_LOKAL.ipynb) verwenden.

Die Rohdatei wird lokal zwischengespeichert, vollständig gegen ihre Producer-SHA256
geprüft und erst danach geladen. Fortschritt, Status und Fehler liegen in einem
eigenen lokalen Ergebnisordner. Zum Schluss wird ein Ergebnis-ZIP automatisch
heruntergeladen; bei Fehlern enthält das Diagnose-ZIP das Protokoll. Die Rohdatei
wird nicht mit eingepackt. Falls der Download am Handy nicht erscheint: links im
Colab-Dateibrowser die ausgegebene ZIP-Datei herunterladen. Ergebnisse vor Ende
der CPU-Laufzeit sichern; das Notebook fordert keine neue Laufzeit an.
'''
    if mode == "LOKAL":
        intro = '''# H_tq · K3-Auswertung mit lokalen Dateien · CPU, ohne Drive-Anmeldung

**CPU-Laufzeit wählen. Dieses Notebook verwendet keine Google-Drive-Anmeldung.**

1. Die vorhandene `k3a_data.pt` und das bereitgestellte kleine
   `htq_k3_context.zip` auf dem eigenen Gerät bereithalten.
2. Links in Colab das Dateisymbol öffnen und beide Dateien nach `/content`
   hochladen. Alternativ ist `/content/htq_k3_upload` möglich. Den großen Upload
   vollständig abwarten. Das Notebook öffnet dafür keinen JavaScript-Uploaddialog.
   Android-Namen wie `k3a_data (1).pt` werden ebenfalls erkannt; alternativ den
   genauen Dateipfad in `INPUT_FILE` eintragen.
3. Danach „Alle ausführen“. Fehlende Dateien führen sofort zu einer Anleitung;
   es gibt keine Anmeldeversuche und keine Warteschleife.

Die Rohdatei und ihr Quellkontext werden unverändert über Producer-SHA256,
H_tq-Policy und Provenienz geprüft, bevor die Auswertung sie lädt. Eine sichtbare
GPU führt zum frühen Abbruch. Es werden keine Modelle geladen oder Antworten
erzeugt; L-CAL und Gesamt-ANA sind keine Voraussetzung. Die Originaldateien
werden ausschließlich gelesen. Der lokale Cache bleibt bei normaler Wiederholung nutzbar.

Ergebnisse und Protokolle liegen unter `/content/htq_k3_output`. Das Ergebnis-ZIP
wird am Ende automatisch heruntergeladen; bei Fehlern gibt es ein Diagnose-ZIP.
Falls der Browser den Download verhindert, dieselbe ZIP links im Dateibrowser
herunterladen. Vor Ende der CPU-Laufzeit sichern; lokale Colab-Dateien sind temporär.
'''
    config = f'''BRANCH = "{BRANCH}"
EINGABE_MODE = "{mode}"
LOKAL_ORDNER = "/content/htq_k3_upload"
INPUT_FILE = ""  # Optional: genauer lokaler Rohdateipfad, etwa /content/k3a_data (1).pt.
KONTEXT_ZIP = ""  # Lokal gewöhnlich leer lassen; das kleine Kontext-ZIP wird automatisch gefunden.
DATEI_ID = ""  # Optional: private Drive-ID der gewünschten k3a_data.pt; gewöhnlich leer lassen.
CACHE_ROOT = "/content/htq_k3_cache"
AUSGABE_ROOT = "/content/htq_k3_output"
'''
    helper = ("# Quellprüfung und Download: nur Standardbibliothek bis zur Drive-Anmeldung.\n"
              + "REL = " + repr(REL) + "\nCONTEXT = " + repr(CONTEXT) + "\nOPTIONAL = " + repr(OPTIONAL) + "\n\n"
              + "\n\n".join(inspect.getsource(f) for f in HELPERS))
    cells=[]
    for i,(kind,source) in enumerate((("markdown",intro),("code",config),("code",helper),("code",main_source()))):
        cell=dict(cell_type=kind,id=hashlib.sha256(f"K3CPU:{mode}:{i}".encode()).hexdigest()[:12],metadata={},source=source.splitlines(keepends=True))
        if kind=='code':ast.parse(source);cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    doc=dict(cells=cells,metadata=dict(colab=dict(provenance=[]),kernelspec=dict(display_name="Python 3",name="python3")),nbformat=4,nbformat_minor=5)
    target=Path(out)/("runde1b_HTQ_K3_LOKAL.ipynb" if mode == "LOKAL" else "runde1b_HTQ_K3_CPU.ipynb")
    target.write_text(json.dumps(doc,indent=1,ensure_ascii=False)+"\n")
    return target


if __name__ == "__main__":
    print(build())
    print(build(mode="LOKAL"))
