"""Builds the Colab notebooks of round 1b (run: python notebooks/build_runde1b_gpu_notebooks.py): one notebook per slot of main §4.4
(analysis/ju_space/round1b/r1b_contracts.SLOTS): runde1b_P0 (pilot, account B), runde1b_A1, runde1b_A2 (account A), runde1b_B1,
runde1b_B2 (account B), runde1b_A_T2, runde1b_B_T2 (tier 2), runde1b_ANA (r1b_ana, then r1b_ana2, CPU runtime, account A). Each runs
analysis/ju_space/round1b/r1b_run_slot.py, which runs the slot's sessions in order and waits for each session's start conditions.
Sessions, accounts, models and start conditions come from r1b_contracts, so that notebooks and runner cannot disagree."""
import importlib.util
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("r1b_contracts", HERE.parent / "analysis" / "ju_space" / "round1b" / "r1b_contracts.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)          # data only
BRANCH = "ccr-6fe02b21-est040"            # the branch with the freeze commit and the round-1b code
NB = {sl: f"runde1b_{sl}.ipynb" for sl in C.SLOTS}
TITLE = {"P0": "Pilot r1b_P0 (vor dem Einfrieren; nur technische Kennzahlen)", "A1": "Stufe 1, Konto A, Spur 1",
         "A2": "Stufe 1, Konto A, Spur 2", "B1": "Stufe 1, Konto B, Spur 1", "B2": "Stufe 1, Konto B, Spur 2",
         "A_T2": "Stufe 2 (nur finanziert und ausgelöst), Konto A", "B_T2": "Stufe 2 (nur finanziert), Konto B",
         "ANA": "Gesamtauswertung r1b_ana, danach Stufe-2-Bericht r1b_ana2 (CPU-Laufzeit), Konto A"}


def cond_de(c):
    k = c["kind"]
    if k == "done":
        return f"`{c['session']}` fertig" + ("" if c["files"] == ("**",) else f" (liest {', '.join(f'`{x}`' for x in c['files'])})")
    if k == "file":
        return f"`{C.file_path(c['fid'])}` vorhanden (Prüfsumme des erzeugenden Schritts)"
    if k == "xfer":
        return f"`r1b_xfer_A/{c['name']}` vorhanden (Prüfsumme in `xfer_manifest.json`)"
    if k == "trig":
        return f"`{C.file_path(c['fid'])}`: `{c['key']}` {dict(eq='=', ne='≠', true='ist wahr')[c['op']]} {c['value'] if c['value'] is not None else ''}"
    return "eine von " + ", ".join(f"`{s}`" for s in c["sessions"]) + " gelaufen"


def markdown(sl):
    d = C.SLOTS[sl]; acct = d["account"]
    rows = []
    for s in d["sessions"]:
        m = C.SESSIONS[s]
        start = "; ".join(cond_de(c) for c in m["after"]) or ("vor dem Einfrieren" if s == "r1b_P0" else "Einfrieren des Plans")
        rows.append(f"| `{s}` | {m['part']} | {m['tier']} | {m['minutes']:g} | {m['content']} | {start} |")
    other = ("Konto B gibt seinen Ordner `dsse_runs` für Konto A frei (Leserecht); in Konto A liegt er unter *Meine Ablage* als "
             "Verknüpfung `dsse_runs_B`. Der Runner kopiert einen fertigen Ordner (`session_done.json` ohne `session_partial.json`) "
             "von dort nach `dsse_runs`, wenn eine Sitzung hier ihn braucht." if acct == "A" else
             "Konto A gibt nur seinen Ordner `dsse_runs/r1b_xfer` für Konto B frei (Leserecht); in Konto B liegt er unter *Meine "
             "Ablage* als Verknüpfung `r1b_xfer_A`. Er enthält nur die Ausfuhren von `r1b_R_S0` (`dmg_ref_R16.json`, "
             "`so_ax_ustar.pt`) mit ihren Prüfsummen in `xfer_manifest.json`.")
    rt = ("*Laufzeit → Laufzeittyp ändern → CPU* (SG2: erklärte CPU-Laufzeit)" if d["runtime"] == "cpu" else
          "*Laufzeit → Laufzeittyp ändern → A100 GPU* (SG2: nur eine A100 mit 40 oder 80 GB)")
    return f"""# Runde 1b · Slot {sl}: {TITLE[sl]}

**Konto {acct}.** Dieses Notebook führt die Sitzungen des Slots nacheinander in einer Laufzeit aus (Plan §4.4; die Reihenfolge ist
eine Packung, keine Regel). Jede Sitzung wartet, bis ihre Startbedingungen erfüllt sind (der Runner fragt Drive alle `ABFRAGE_S`
Sekunden ab, höchstens `WARTEN_MAX_H` Stunden), läuft dann als eigener Prozess, schreibt nach `dsse_runs/<Sitzung>` und gleicht nach
jedem Schritt mit Drive ab. Zeitgrenze je Sitzung (SG5): nach dem Schritt, der 4,5 h nach dem Start dieser Sitzung endet, hält sie an
(`session_partial.json`) und die Fortsetzung startet sofort; fertige Schritte und Sitzungen werden immer übersprungen.

| Sitzung | Teil | Stufe | A100-min (Plan) | Inhalt | Startet nach |
|---|---|---|---|---|---|
{chr(10).join(rows)}

**Zwei Konten.** {other} Stufe-2-Sitzungen laufen nur, wenn `runde1b_freeze.json` Stufe 2 finanziert und ihr Auslöser eingetreten
ist; sonst schreibt der Runner `session_not_run.json` und macht weiter.

**Ausführen.** {rt}, dann *Laufzeit → Alle ausführen*. Höchstens zwei Slots je Konto gleichzeitig. Bricht die Laufzeit ab, das
Notebook einfach erneut ausführen.

**Zwei Versuche.** Scheitert eine Sitzung zweimal, geht der Slot weiter und meldet es. Nach Prüfung der Protokolle
(`dsse_runs/<Sitzung>/logs`) kann der Betreiber sie in `dsse_runs/r1b_declared_not_run.json` des Kontos, in dem ihre Leser laufen,
als 'not run' eintragen, z. B. `{{"sessions": {{"r1b_K_S3": "failed twice (OOM)"}}}}` (der Runner prüft die zwei Versuche); ihre Leser
starten dann, die betroffenen Messungen lauten 'not run (session failed twice)'.

**Selbsttest.** `SELBSTTEST = True` führt vorher alle Sitzungen dieses Slots und ihre Vorgänger im MOCK-Modus (winziges Zufallsmodell)
auf einem Schein-Drive aus; dauert einige Minuten, einmal je Code-Stand; Standard aus."""


def cells(sl):
    d = C.SLOTS[sl]; acct = d["account"]; cpu = d["runtime"] == "cpu"
    models = sorted({m for s in d["sessions"] for m in C.SESSIONS[s]["models"]})
    jlens = any(C.SESSIONS[s]["jlens"] for s in d["sessions"])
    r_part = any(C.SESSIONS[s]["part"] == "R" or s == "r1b_P0" for s in d["sessions"])
    gpu = ('''import torch
assert torch.cuda.is_available(), "Keine GPU: Laufzeit → Laufzeittyp ändern → A100 GPU, dann neu starten."
GPU, CC = torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {GPU}, sm{CC[0]}{CC[1]}, {GB:.0f} GB, torch {torch.__version__}")
assert "A100" in GPU and CC == (8, 0) and GB >= 39, f"SG2: nur eine A100 mit 40 oder 80 GB ist zulässig (hier {GPU})."
''' if not cpu else 'print("CPU-Laufzeit (SG2: r1b_ana läuft auf einer erklärten CPU-Laufzeit).")\n')
    p = f'''SLOT, KONTO = "{sl}", "{acct}"
BRANCH = "{BRANCH}"
SELBSTTEST = False           # MOCK-Selbsttest dieses Slots vor dem Hauptlauf (einige Minuten; einmal je Code-Stand)
NUR_SELBSTTEST = False       # True: nur einrichten und testen, kein Hauptlauf
WARTEN_MAX_H = 10            # so lange wartet eine Sitzung höchstens auf ihre Startbedingungen
ABFRAGE_S = 120              # Abstand der Drive-Abfragen in Sekunden
LAUFZEIT_MAX_H = 23          # keine Sitzung startet, deren geplante Dauer über diese Laufzeit hinausreicht
import os, sys, json, shutil, subprocess, hashlib, lzma, time
{gpu}'''
    drive = f'''# Google Drive: dsse_runs dieses Kontos; Arbeitskopien auf der lokalen Platte (/content/work/<Sitzung>)
from google.colab import drive
drive.mount("/content/drive")
MYDRIVE = "/content/drive/MyDrive"; RUNS = f"{{MYDRIVE}}/dsse_runs"
os.makedirs(RUNS, exist_ok=True)
WORK, ROOT, REPO = "/content/work", "/content/root", "/content/DSSE"
os.makedirs(WORK, exist_ok=True); os.makedirs(ROOT, exist_ok=True)
LINK = f"{{MYDRIVE}}/{C.RUNS_B_LINK if acct == 'A' else C.XFER_LINK}"
print("Verknüpfung", LINK, "vorhanden" if os.path.isdir(LINK) else "FEHLT - im anderen Konto freigeben und hier als Verknüpfung anlegen")
'''
    code = '''# Code: analysis/ju_space (mit round1b/ und round2/), der Plan und die Runde-2-Ergebnisse; ohne --depth, denn der Runner prüft
# den Einfrier-Commit (SG1) und den Code der Runde-1-Teile an ihren Commits (SG3)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "transformers==5.22.2"], check=True)
if os.path.exists(REPO):
    shutil.rmtree(REPO)
subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--sparse", "-b", BRANCH, "https://github.com/Erikiss/DSSE", REPO], check=True)
subprocess.run(["git", "-C", REPO, "sparse-checkout", "set", "analysis/ju_space", "results/ju_space/plans",
                "results/ju_space/runde2_ergebnisse"], check=True)
COMMIT = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print("Code:", subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%h %cd %s"], capture_output=True, text=True).stdout.strip())
A = f"{REPO}/analysis/ju_space"; R1B = f"{A}/round1b"; D = f"{A}/gpu/data"
FZ_PATH = f"{REPO}/results/ju_space/plans/runde1b/runde1b_freeze.json"
FZ = json.load(open(FZ_PATH)) if os.path.exists(FZ_PATH) else {}
print("Einfrier-Datensatz:", "vorhanden" if FZ else "fehlt (nur der Pilot r1b_P0 läuft vor dem Einfrieren)")
'''
    inputs = f'''# Eingaben: Modelle zur festen Revision mit Prüfsummen (r1b_core.ensure_model; models/verified.json), für Teil R die Persona-Sätze
# und Runde-1-Daten (wie Runde 2), die J-Lens (Prüfsumme festgelegt); Datensätze lädt jeder Schritt selbst zur festen Revision.
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

MODELLE = {models!r}
if MODELLE and not NUR_SELBSTTEST:
    subprocess.run([sys.executable, f"{{R1B}}/r1b_core.py", "fetch-models", ROOT] + MODELLE, check=True)
if SELBSTTEST:                                # der MOCK-Selbsttest braucht nur Konfiguration und Tokenizer von Qwen3-1.7B
    from huggingface_hub import snapshot_download
    snapshot_download("Qwen/Qwen3-1.7B", revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", local_dir=f"{{ROOT}}/models/qwen3-1.7b",
                      allow_patterns=["*.json", "merges.txt", "vocab.json"])
if {r_part!r}:                                # rk_loop-Pfade von Teil R (r2_rk.open_block): Persona-Sätze und Runde-1-Daten
    from huggingface_hub import hf_hub_download
    SUMS = json.load(open(f"{{D}}/checksums.json")); P = f"{{ROOT}}/rsi_dl/persona"; os.makedirs(P, exist_ok=True)
    for n, h in SUMS["persona_git_sha1"].items():
        data = open(hf_hub_download(SUMS["persona_repo"], f"persona/{{n}}.jsonl", repo_type="dataset", revision=SUMS["persona_revision"]), "rb").read()
        assert hashlib.sha1(b"blob %d\\0" % len(data) + data).hexdigest() == h, f"Prüfsumme stimmt nicht: {{n}}"
        open(f"{{P}}/{{n}}.jsonl", "wb").write(data)
    for src, dst in SUMS["inputs"].items():
        os.makedirs(os.path.dirname(f"{{ROOT}}/{{dst['path']}}"), exist_ok=True)
        raw = lzma.open(f"{{D}}/{{src}}").read() if src.endswith(".xz") else open(f"{{D}}/{{src}}", "rb").read()
        open(f"{{ROOT}}/{{dst['path']}}", "wb").write(raw)
        assert sha256(f"{{ROOT}}/{{dst['path']}}") == dst["sha256"], f"Prüfsumme stimmt nicht: {{dst['path']}}"
if {jlens!r} and not NUR_SELBSTTEST:          # SO-JLENS: aus dsse_runs/r1b_inputs (einmal je Konto hochladen) oder von der URL im Einfrier-Datensatz
    PIN = "{'6fcc79011bd921ffd87612255e2e99950a124fa519470ee44ebaf161c39be9d6'}"
    ZIEL = f"{{ROOT}}/jlens/Qwen3-1.7B_jacobian_lens.pt"; os.makedirs(os.path.dirname(ZIEL), exist_ok=True)
    QUELLE = f"{{RUNS}}/r1b_inputs/Qwen3-1.7B_jacobian_lens.pt"
    if not os.path.exists(ZIEL) and os.path.exists(QUELLE):
        shutil.copyfile(QUELLE, ZIEL)
    elif not os.path.exists(ZIEL) and (FZ.get("jlens") or {{}}).get("url"):
        import urllib.request; urllib.request.urlretrieve(FZ["jlens"]["url"], ZIEL)
    if os.path.exists(ZIEL):
        assert sha256(ZIEL) == PIN, "J-Lens: Prüfsumme stimmt nicht (SO-JLENS)"
        print("J-Lens geprüft.")
    else:
        print("J-Lens FEHLT: dsse_runs/r1b_inputs/Qwen3-1.7B_jacobian_lens.pt hochladen; Schritte, die sie brauchen, scheitern sonst.")
if {"qwen3.5-2b" in models!r} and not NUR_SELBSTTEST:   # Teil K (K5a, RK-2): J-Lens von Qwen3.5-2B (Name, sha256: Einfrier-Datensatz jlens.qwen3.5-2b)
    J2 = (FZ.get("jlens") or {{}}).get("qwen3.5-2b") or {{}}
    if J2.get("name"):
        ZIEL2, QUELLE2 = f"{{ROOT}}/jlens/{{J2['name']}}", f"{{RUNS}}/r1b_inputs/{{J2['name']}}"
        os.makedirs(os.path.dirname(ZIEL2), exist_ok=True)
        if not os.path.exists(ZIEL2) and os.path.exists(QUELLE2):
            shutil.copyfile(QUELLE2, ZIEL2)
        print("J-Lens Qwen3.5-2B:", "geprüft" if os.path.exists(ZIEL2) and sha256(ZIEL2) == J2.get("sha256") else "FEHLT oder Prüfsumme falsch (K5a / RK-2 lesen dann 'not run')")
print("Eingaben bereit:", ", ".join(MODELLE) or "keine Modelle nötig")
'''
    copy = '''# Zweiter Ordner: was die Sitzungen dieses Slots aus dem anderen Konto lesen (Konto A: fertige Ordner aus dsse_runs_B; Konto B:
# r1b_xfer_A); der Runner wiederholt das bei jeder Abfrage, diese Zelle zeigt den Stand vorab
subprocess.run([sys.executable, f"{R1B}/r1b_run_slot.py", SLOT, ROOT, WORK, RUNS, "--copy-only"], env=dict(os.environ, R1B_ACCOUNT=KONTO))
'''
    selftest = f'''# Selbsttest (nur Code): alle Sitzungen dieses Slots und ihre Vorgänger im MOCK-Modus auf einem Schein-Drive (zwei Konten, Verknüpfungen
# als symbolische Links), über denselben Runner wie der Hauptlauf; nichts geht ins echte Drive. Marke je Code-Stand in dsse_runs.
MARKE = f"{{RUNS}}/r1b_selftest_{{SLOT}}_{{COMMIT[:12]}}.json"
if not SELBSTTEST:
    print("Selbsttest ausgelassen (SELBSTTEST = False).")
elif os.path.exists(MARKE):
    print("Selbsttest für diesen Code-Stand schon bestanden:", json.load(open(MARKE)))
else:
    t = time.time(); ST = "/content/r1b_selftest"; shutil.rmtree(ST, ignore_errors=True)
    r = subprocess.run([sys.executable, "-u", f"{{R1B}}/r1b_run_slot.py", SLOT, ROOT, ST, "--selftest"],
                       env=dict(os.environ, RUN_DEVICE="{'cpu' if cpu else 'cuda'}", NT="4"))
    if r.returncode:
        raise SystemExit(f"Selbsttest fehlgeschlagen (Protokolle unter {{ST}}/work/*/*/logs).")
    json.dump(dict(commit=COMMIT, seconds=round(time.time() - t)), open(MARKE, "w"))
    print(f"Selbsttest bestanden ({{time.time() - t:.0f}}s).")
'''
    main = f'''# Hauptlauf: die Sitzungen des Slots nacheinander; jede mit SG1-SG7, Startbedingungen, Kopie der Eingaben, Manifest, Schritten mit
# Zeitgrenze und Drive-Abgleich
MELDUNG = {{0: "Alle Sitzungen dieses Slots sind fertig oder 'not run' (Stufe 2 nicht finanziert / nicht ausgelöst).",
           1: "Eine Sitzung ist zweimal gescheitert (siehe oben); Protokolle prüfen, ggf. als 'not run' erklären, Notebook erneut ausführen.",
           2: "Eine Freigabe (SG1, SG2, SG3 oder SG7) ist nicht erfüllt: manifest.json der Sitzung ansehen; diese Sitzung lief nicht.",
           3: "Laufzeitgrenze erreicht: Notebook in einer neuen Laufzeit erneut ausführen (Fertiges wird übersprungen).",
           4: "Startbedingungen nach WARTEN_MAX_H Stunden nicht erfüllt: später erneut ausführen."}}
RC = None
if NUR_SELBSTTEST:
    print("NUR_SELBSTTEST = True: kein Hauptlauf.")
else:
    env = dict(os.environ, R1B_ACCOUNT=KONTO, R1B_ROOT=ROOT, RUN_DEVICE="{'cpu' if cpu else 'cuda'}", NT="8", R1B_RUNTIME_T0=repr(T0))
    cmd = [sys.executable, "-u", f"{{R1B}}/r1b_run_slot.py", SLOT, ROOT, WORK, RUNS, "--poll-s", str(ABFRAGE_S),
           "--wait-h", str(WARTEN_MAX_H), "--runtime-h", str(LAUFZEIT_MAX_H)]
    RC = subprocess.run(cmd, env=env).returncode
    print(MELDUNG.get(RC, f"Unerwarteter Rückgabewert {{RC}}"))
'''
    summary = f'''# Übersicht: Zustand jeder Sitzung des Slots in Drive
for s in {list(d["sessions"])!r}:
    base = f"{{RUNS}}/{{s}}"
    for name in ("session_done.json", "session_partial.json", "session_not_run.json", "start_check.json"):
        if os.path.exists(f"{{base}}/{{name}}"):
            S = json.load(open(f"{{base}}/{{name}}")); print(f"{{s}}: {{name}}", S.get("reason") or "")
            for k, v in (S.get("steps") or {{}}).items():
                sec = f"{{v['seconds']:.0f}}s" if isinstance(v.get("seconds"), (int, float)) else ""
                print(f"    {{k:<28}} {{v['state']:<9}} {{sec:>6}}  {{(v.get('note') or '')[:80]}}")
            break
    else:
        print(f"{{s}}: noch nicht begonnen")
'''
    flush = '''# Abschluss: Drive schreiben und trennen
drive.flush_and_unmount()
print("Fertig. Ergebnisse in Google Drive unter dsse_runs/<Sitzung>.")
if RC not in (None, 0):
    raise SystemExit(MELDUNG.get(RC, f"Rückgabewert {RC}"))
'''
    return [("md", markdown(sl)), ("code", "T0 = __import__('time').time()\n" + p), ("code", drive), ("code", code), ("code", inputs),
            ("code", copy), ("code", selftest), ("code", main), ("code", summary), ("code", flush)]


def build(sl, out=HERE):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src.strip("\n")) if k == "md" else nbf.v4.new_code_cell(src.strip("\n")) for k, src in cells(sl)]
    if C.SLOTS[sl]["runtime"] != "cpu":
        nb.metadata["accelerator"] = "GPU"
        nb.metadata["colab"] = {"gpuType": "A100", "provenance": []}
    else:
        nb.metadata["colab"] = {"provenance": []}
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nbf.validate(nb)
    nbf.write(nb, Path(out) / NB[sl])


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE        # optional output folder (tests)
    assert set(TITLE) == set(C.SLOTS), "every slot needs a title"
    for sl in C.SLOTS:
        build(sl, out)
    print("geschrieben:", [NB[sl] for sl in C.SLOTS])
