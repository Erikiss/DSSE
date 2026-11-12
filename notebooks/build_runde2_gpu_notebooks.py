"""Builds the nine Colab notebooks of round 2 of README sections 11/12 (run: python notebooks/build_runde2_gpu_notebooks.py).

runde2_q1.ipynb ... runde2_ana.ipynb, one per session of the frozen plan (results/ju_space/plans/abschnitt11_12_runde2_plan.md, section
7.2); each calls analysis/ju_space/round2/r2_run_session.py. Qwen3-1.7B (and Qwen3-0.6B for r2_q3) and the persona sets are loaded at
pinned Hugging Face revisions and checked against the checksums of analysis/ju_space/gpu/data/checksums.json (round 1) and of the
runner (QWEN06). Results go to Google Drive, dsse_runs/<session>. The session list, the steps, the folders a session reads and the
Qwen3-0.6B pin are taken from the runner, so that notebooks and runner cannot disagree.
"""
import importlib.util
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parent / "analysis" / "ju_space" / "round2" / "r2_run_session.py"
_spec = importlib.util.spec_from_file_location("r2_run_session", RUNNER)
R = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(R)            # import only: the runner runs nothing on import

BRANCH = "ccr-6fe02b21-est040"            # the branch with the freeze commit and the round-2 code (as round 1)
MODEL_REV = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
PERSONA_REV = "d533f626cc321c92175a58ee570aa3cdb87238d1"

# Title, content and the estimates of plan 7.2 (work; setup + work; setup + 1.5 x work), German as in round 1
TEXT = {
    "r2_q1": ("Qwen3-1.7B: RGI-Achsen, Verlernen, Wiedersuche, Gegenschub, Primitive, GEN-Schleifen",
              "An Block 14: die sechs Instruktionstexte der RGI-Achsenfamilie mit Aussagen- und NEG-Nullachsen und den Token-Zuständen "
              "für C-Q1 (O2); Verlernen durch exakte Schreib-Orthogonalisierung (C-Q3); Wiedersuche mit drei Beispielantworten im Kontext "
              "(C-Q4a); Gegenschub bei gleichem Schaden, Verhalten bei α_u (C-Q2); die GEN-Schleife (A6); die Schleife unter Ablation, nur "
              "bei offenem Gate (C-Q4b); Arm A; Überlagerung und Dosis (C-Q5); Verschachtelungstiefe (C-Q6). Danach die GEN-Schleifen an "
              "den Zusatzblöcken b_lo und b_hi (jeweils nur bei bestandenem AUC-Gate GA(b)).", (8.9, 12.9, 17.3)),
    "r2_q2": ("Qwen3-1.7B: Temperatur-Phasendiagramm (θ)",
              "Teil B. An Block 14: freie Verstärkungen aus den eingefrorenen Schleifen (B-Q0), Code-Prüfungen, Schadenskurven und alle "
              "θ-Zellen (α-Reihe B-Q3, Temperatur B-Q4, mit den Reserven), der Übertrag in den zweiten Zug (B-Q8), der Potenzschritt "
              "(B-Q6) und der Fluktuations-Dissipations-Anker (B-Q7). An jedem Zusatzblock zuerst die zweite RSI-Schleife r2RSIb (vier "
              "Runden), dann B-Q0, die Zellen mit Reserve 3, Potenzschritt und B-Q7.", (8.8, 12.8, 17.2)),
    "r2_q3": ("Qwen3-1.7B und Qwen3-0.6B: RGI-Satzgeometrie über alle Blöcke",
              "A1–A5: 2.040 Sätze (Besitzer × Substrat, fünf Verbfamilien, zwei Rahmen) an allen 28 Blöcken, die Gradienten der kausalen "
              "Lesart (A3), die δ-Zustände (O7) und die Pool-Kovarianz (O8), dann die Statistik auf der GPU; danach derselbe Arm für "
              "Qwen3-0.6B (feste Revision; scheitern Download oder Prüfsumme, läuft dieser Arm nicht, Freigabe G33).", (6.0, 10.0, 13.0)),
    "r2_t1": ("Kleinversuch RGI: δ-Sonde, δ-Arme, Kreuzmatrix, Familien, Zwangskontrollen",
              "AT0 (Kalibrierung der δ-Sonde), AT2 mit AT1 und AT3 (8 Welten × 3 Initialisierungen, 15 Bedingungen), AT5 (Kreuzmatrix "
              "der Backprop-Netze; ihre Netze braucht später r2_pc), AT4 (Familienarm, mit den Netzen aus AT5) und AT6 "
              "(Zwangskontrollen). Alle Netze werden in dieser Sitzung trainiert, vektorisiert in einem Prozess.", (7.8, 10.8, 14.7)),
    "r2_t2": ("Kleinversuch θ: Regularisierung × Kapazität und Selbsttraining mit Temperatur τ",
              "B-T1 (96 Netze: Gewichtsabfall × Breite × Seeds × K), B-T2 (Breitenprüfung der Grenze bei W 512 und 1024), B-T5 "
              "(Selbsttrainingsschleife mit echter Temperatur τ) und B-T4 mit den Regeln P1–P4 und den T5-Phasen.", (3.9, 6.9, 8.8)),
    "r2_t3": ("Kleinversuch Verlernen: Ringcode, Kappen, Kreiswelt, Löschen, Primitive",
              "C-T0 (Ringcode), C-T1 (Superpositionskappe), C-T2 (auferlegter Code), C-T3 (Kreiswelt als Positivkontrolle: Wortlänge, "
              "Schlüpfen, Löschen, Inverse), C-T4 (Löschen und Gegensteuern, Leiter Richtung/Unterraum) und C-T6 (Trajektorien-Primitive "
              "mit Nullmodell und kausaler Kappe).", (5.3, 8.3, 10.9)),
    "r2_t4": ("Kleinversuch Ent-Suchen (C-T5)",
              "C-T5 im Backprop-Hauptprotokoll: iteratives Entfernen gefundener Richtungen über 24 Runden (Arme col, z1, rand, pc, dmr), "
              "Breiten 128 und 256, fünf Seeds. Der PC-gepaarte Zwilling läuft in r2_pc.", (4.3, 7.3, 9.4)),
    "r2_pc": ("Predictive Coding gegen Backprop (C-T7) und Kreuzpaare mit PC-Netzen (AT5)",
              "C-T7 bei W 128 und 256: PC-Seite auf den PC-µP-Netzen und BP-Seite auf den BP-µP-Netzen der Runde 1 (T4b und der "
              "PC-gepaarte Zwilling von C-T5), unabhängig von den Freigaben (G-meta und G-PC entscheidet r2_ana); dazu die AT5-Paare mit "
              "einem PC-Netz bei jeder PC-Breite, mit den Backprop-Netzen aus r2_t1.", (6.0, 9.0, 12.0)),
    "r2_ana": ("Auswertung: Statistiken, Freigaben, Urteile, Bericht",
               "Ohne Qwen: C-Q1 (Linearitätsleiter mit Kalibrierung), θ-Statistik mit gemeinsamem Bootstrap und Leistungsrechnung "
               "(R1–R6, B-Q2, B-Q7), RGI-Auslesungen und Regeln (A6, A7, AT7, AT8), die Regeln von C-Q2 bis C-Q6 und C-T0 bis C-T7, die "
               "Freigaben G-meta, G-PC, G-desc und G-blk, die Reproduktionsprüfungen (G46) und round2_summary.json mit dem "
               "zusammengeführten Manifest.", (7.9, 10.9, 14.8)),
}
NB_NAME = {s: "runde2_" + s[3:] + ".ipynb" for s in R.SESSIONS}        # r2_q1 -> runde2_q1.ipynb


def de(x):
    return f"{x:.1f}".replace(".", ",")


def folders(s):
    """(needed, optional) folders in dsse_runs that session s reads (from the runner's session table)."""
    m = R.SESSIONS[s]
    return list(m["shards"]) + list(m["upstream"]), list(m["meta_shards"])


def steps_md(s):
    out = []
    for st in R.STEPS[s]:
        b = {"14": "14", "b_lo": "⟨b_lo⟩", "b_hi": "⟨b_hi⟩", None: ""}[st.block]
        stage = [a for a in st.args.split() if not a.startswith("{")]
        out.append(f"`{st.name.replace('{b}', b)}` ({st.script}{' ' + ' '.join(stage) if stage else ''})")
    return " · ".join(out)


def markdown(s):
    title, what, (work, total, margin) = TEXT[s]
    m = R.SESSIONS[s]; need, opt = folders(s)
    wave = f"Welle {m['wave']}"
    if s in ("r2_q1", "r2_q2"):
        start = ("Startet nach dem Einfrieren des Plans, wenn in Runde 1 die Blockzusammenfassungen `summary_b14.json`, "
                 "`summary_b<b_lo>.json` und `summary_b<b_hi>.json` sowie `gpu2/dn_delta.json`, `gpu2/blocks_chosen.json` und "
                 "`gpu3/blocks_chosen.json` vorliegen (Freigabe G4). Sonst wartet der Runner (Rückgabewert 4) und startet nichts.")
        reads = ("Liest aus Runde 1 `dsse_runs/gpu1`, `gpu2` und `gpu3` (Konto A); von `gpu4`–`gpu6` (Konto B) nur die "
                 "Versionsangaben fürs Manifest – diese Ordner dürfen fehlen. Dazu `dsse_runs/r1_finality.json`, falls vorhanden.")
    elif s == "r2_pc":
        start = ("Welle 2: startet erst, wenn `dsse_runs/r2_t1/session_done.json` existiert und die PC-Ausgaben aller sechs "
                 "Runde-1-Teile endgültig sind (Freigabe G5: `shard_done.json`, sonst der Eintrag in `r1_finality.json`).")
        reads = "Liest alle sechs Runde-1-Ordner `dsse_runs/gpu1` … `gpu6` (beide Konten) und aus Runde 2 `dsse_runs/r2_t1`."
    elif s == "r2_ana":
        start = ("Welle 2: startet erst, wenn alle acht anderen Sitzungen `session_done.json` geschrieben haben und G4 und G5 erfüllt "
                 "sind. Eine Sitzung, die nach zwei Versuchen nicht fertig wurde, kann unten in `NICHT_GELAUFEN` eingetragen werden "
                 "(der Runner prüft die zwei Versuche).")
        reads = ("Liest alle sechs Runde-1-Ordner `dsse_runs/gpu1` … `gpu6` (beide Konten) und alle Runde-2-Ordner `dsse_runs/r2_*`.")
    else:
        start = "Startet nach dem Einfrieren des Plans; braucht keine Ausgabe der Runde 1 und keine andere Sitzung."
        reads = ("Liest keine Runde-1-Ausgabe (nur die Änderungszeiten der Protokolle und Zusammenfassungen in `dsse_runs/gpu*` fürs "
                 "Manifest, falls sichtbar).")
    accounts = ("**Zwei Konten.** Die Runde-1-Teile `gpu1`–`gpu3` liefen in einem Google-Konto (A), `gpu4`–`gpu6` in einem anderen (B). "
                "Einen Ordner aus dem anderen Konto vorher dort freigeben (Leserecht genügt) und hier in *Meine Ablage › dsse_runs* über "
                "*Verknüpfung zu Drive hinzufügen* einhängen, oder ihn in dieses Konto nach `dsse_runs/` kopieren. Der Runner liest diese "
                "Ordner nur und schreibt ausschließlich nach `dsse_runs/" + s + "`.")
    if m["wave"] == 1:
        accounts += (" Die Ergebnisse dieser Sitzung liest später `r2_ana`" + (" (und `r2_pc`)" if s == "r2_t1" else "") +
                     ": läuft jene in einem anderen Konto, `dsse_runs/" + s + "` dort ebenso verfügbar machen.")
    final = ""
    if m["start"]:
        final = ("\n\n**Runde-1-Endstand (R1.final).** Fehlt in einem Runde-1-Ordner auch nach zwei Läufen seines Notebooks "
                 "`shard_done.json` (PC-Jobs fehlen) oder eine Blockzusammenfassung, trägt der Betreiber das von Hand in "
                 "`dsse_runs/r1_finality.json` ein (kein Skript schreibt diese Datei; Plan 2.1), zum Beispiel\n"
                 "```json\n{\"gpu4\": {\"runs\": [{\"start\": \"2026-11-12T15:02Z\", \"end\": \"2026-11-12T15:24Z\"},\n"
                 "                  {\"start\": \"2026-11-12T16:10Z\", \"end\": \"2026-11-12T16:31Z\"}], \"final\": true}}\n```\n"
                 "Ab zwei eingetragenen Läufen und `\"final\": true` gilt der Teil als endgültig; Teile an einem Block ohne "
                 "Zusammenfassung laufen dann nicht ('not run (round-1 output missing)').")
    return f"""# Runde 2 · Sitzung {s} ({wave}): {title}

{what}

**Ausführen.** *Laufzeit → Laufzeittyp ändern → A100 GPU*, dann *Laufzeit → Alle ausführen*. Nur eine A100 ist zulässig
(Freigabe G2). Erwartete Dauer (Plan 7.2): Arbeit {de(work)} min, mit Einrichtung {de(total)} min, mit Reserve {de(margin)} min.

**Zeitgrenze.** 20 Minuten nach der ersten Zelle hält der Lauf nach dem gerade laufenden Schritt an (Plan 7.4) und schreibt
`session_partial.json`. Dann dieses Notebook in einer neuen Colab-Sitzung noch einmal ganz ausführen: es läuft als
Fortsetzungssitzung (`{s}b`) weiter, fertige Schritte werden übersprungen. Ebenso nach einem Verbindungsabbruch.

**Startbedingungen (Plan 7.3).** {start}

**Ordner.** {reads}

{accounts}{final}

**Selbsttest.** Vor dem Hauptlauf läuft die Sitzung einmal im MOCK/QUICK-Modus (winziges Zufallsmodell, Kurzläufe) auf einem
Schein-`dsse_runs`, das die Runde-1-Skripte erzeugen – nur Code, keine Ergebnisse; einmal je Code-Stand. Er kostet einige Minuten der
Zeitgrenze: am besten das Notebook zuerst einmal mit `NUR_SELBSTTEST = True` ausführen; der echte Lauf überspringt ihn dann.

**Ergebnisse** in Google Drive unter `dsse_runs/{s}` (Manifest `manifest_{s}.json`, `session_done.json` mit den Schrittzeiten).

Schritte: {steps_md(s)}

Modell `Qwen/Qwen3-1.7B` (float32, TF32 aus){', dazu `Qwen/Qwen3-0.6B`' if s == 'r2_q3' else ''} in fester Revision mit Prüfsummen.
Code: `erikiss/DSSE`, Branch `{BRANCH}`, Ordner `analysis/ju_space/round2`. Plan: eingefroren mit Commit `{R.FREEZE_COMMIT[:12]}`
(`results/ju_space/plans/abschnitt11_12_runde2_plan.md`, sha256 `{R.FROZEN_PLAN_SHA256[:16]}…`); der Runner läuft mit keiner
anderen Planfassung (Freigabe G1).
"""


def cells(s):
    m = R.SESSIONS[s]; need, opt = folders(s)
    q06 = f"\nQWEN06 = {R.QWEN06!r}    # Qwen/Qwen3-0.6B (HF-API, 12.11.2026; OD-A6)" if s == "r2_q3" else ""
    selbsttest_default = "False" if s == "r2_ana" else "True"
    ana_note = ("\n# r2_ana: der volle Selbsttest führt alle Sitzungen im MOCK/QUICK-Modus aus (lange); am besten einmal mit NUR_SELBSTTEST = True"
                if s == "r2_ana" else "")
    p = f'''T0 = __import__("time").time()     # Beginn der Sitzung: die Zeitgrenze von 20 min (Plan 7.4) zählt ab hier
SESSION, KIND = "{s}", "{m['kind']}"
BRANCH = "{BRANCH}"
MODEL_REV = "{MODEL_REV}"      # Qwen/Qwen3-1.7B (wie Runde 1; Prüfsummen in analysis/ju_space/gpu/data/checksums.json)
PERSONA_REV = "{PERSONA_REV}"    # Anthropic/model-written-evals{q06}
BRAUCHT = {need!r}    # Ordner in dsse_runs, die diese Sitzung liest
OPTIONAL = {opt!r}    # nur Versionsangaben; dürfen fehlen
NICHT_GELAUFEN = []          # nur r2_ana: Sitzungen, die nach zwei Versuchen als 'not run' gelten (z. B. ["r2_t4"])
SELBSTTEST = {selbsttest_default}            # MOCK/QUICK-Selbsttest vor dem Hauptlauf (einmal je Code-Stand){ana_note}
NUR_SELBSTTEST = False       # True: nur einrichten und testen, kein Hauptlauf
import os, sys, json, shutil, subprocess, hashlib, lzma, time
import torch
assert torch.cuda.is_available(), "Keine GPU: Laufzeit → Laufzeittyp ändern → A100 GPU, dann neu starten."
GPU = torch.cuda.get_device_name(0); GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {{GPU}}, {{GB:.0f}} GB, torch {{torch.__version__}}")
assert "A100" in GPU, f"Freigabe G2: nur eine A100 ist zulässig (hier {{GPU}}). Laufzeit → Laufzeittyp ändern → A100 GPU."
'''
    drive = '''# Ablage: Google Drive (dsse_runs/<Sitzung>), Arbeitskopie auf der lokalen Platte; Fortsetzung aus Drive
try:
    from google.colab import drive
    drive.mount("/content/drive")
    RUNS = "/content/drive/MyDrive/dsse_runs"
    DRIVE = f"{RUNS}/{SESSION}"
except Exception as e:
    print("Kein Google Drive, Ergebnisse nur lokal:", e)
    RUNS = DRIVE = None
WORK, ROOT, REPO = f"/content/work/{SESSION}", "/content/root", "/content/DSSE"
os.makedirs(WORK, exist_ok=True)
if DRIVE and os.path.isdir(DRIVE) and os.listdir(DRIVE):
    shutil.copytree(DRIVE, WORK, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.tmp", "*.r2tmp"))
    print("Fortsetzung: bisheriger Stand aus Drive übernommen; fertige Schritte werden übersprungen.")
if RUNS:                                   # nur Ordnernamen, kein Inhalt
    fehlt = [f for f in BRAUCHT if not os.path.isdir(f"{RUNS}/{f}")]
    print("Diese Sitzung liest aus dsse_runs:", ", ".join(BRAUCHT) or "nichts")
    if fehlt:
        print("FEHLT:", ", ".join(fehlt), "- aus dem anderen Konto freigeben und hier verknüpfen oder kopieren; sonst wartet der Runner.")
    if any(not os.path.isdir(f"{RUNS}/{f}") for f in OPTIONAL):
        print("Nicht sichtbar (nur Versionsangaben, darf fehlen):", ", ".join(f for f in OPTIONAL if not os.path.isdir(f"{RUNS}/{f}")))
print("Ergebnisse:", DRIVE or WORK)
'''
    code = '''# Pakete und Code: der Ordner analysis/ju_space (mit round2/) und der eingefrorene Plan; ohne --depth, denn der Runner prüft den
# Einfrier-Commit (G1) und den Code der Runde-1-Teile (G3) in der Git-Geschichte (Dateiinhalte werden bei Bedarf nachgeladen)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "transformers==5.22.2"], check=True)
if os.path.exists(REPO):
    shutil.rmtree(REPO)
subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--sparse", "-b", BRANCH, "https://github.com/erikiss/DSSE", REPO], check=True)
subprocess.run(["git", "-C", REPO, "sparse-checkout", "set", "analysis/ju_space", "results/ju_space/plans"], check=True)
COMMIT = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print("Code:", subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%h %cd %s"], capture_output=True, text=True).stdout.strip())
A = f"{REPO}/analysis/ju_space"; D = f"{A}/gpu/data"; R2 = f"{A}/round2"
SUMS = json.load(open(f"{D}/checksums.json"))
'''
    inputs = '''# Eingaben: Modelle und Persona-Sätze von Hugging Face (feste Revision), übrige Daten aus dem Repository; alles mit Prüfsumme.
# models/verified.json hält fest, was geprüft wurde (der Runner trägt es ins Manifest ein; ohne sie prüft er selbst).
from huggingface_hub import snapshot_download, hf_hub_download

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

jetzt = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
os.makedirs(f"{ROOT}/models", exist_ok=True)
VERIFIED = json.load(open(f"{ROOT}/models/verified.json")) if os.path.exists(f"{ROOT}/models/verified.json") else {}
QWEN = KIND == "qwen"
MOCK_QWEN = SELBSTTEST and SESSION == "r2_ana"        # der volle Selbsttest von r2_ana braucht nur Konfiguration und Tokenizer
if QWEN or MOCK_QWEN:
    M = f"{ROOT}/models/qwen3-1.7b"
    snapshot_download("Qwen/Qwen3-1.7B", revision=MODEL_REV, local_dir=M,
                      allow_patterns=["*.json", "merges.txt", "vocab.json"] + (["*.safetensors"] if QWEN else []))
    for fn, h in SUMS["model_sha256"].items():
        if QWEN or not fn.endswith(".safetensors"):
            assert sha256(f"{M}/{fn}") == h, f"Prüfsumme stimmt nicht: {fn}"
    if QWEN:
        VERIFIED["qwen3-1.7b"] = dict(ok=True, repo="Qwen/Qwen3-1.7B", revision=MODEL_REV, sha256=SUMS["model_sha256"], checked=jetzt())
    P = f"{ROOT}/rsi_dl/persona"; os.makedirs(P, exist_ok=True)
    for n, h in SUMS["persona_git_sha1"].items():
        data = open(hf_hub_download("Anthropic/model-written-evals", f"persona/{n}.jsonl", repo_type="dataset",
                                    revision=PERSONA_REV), "rb").read()
        assert hashlib.sha1(b"blob %d\\0" % len(data) + data).hexdigest() == h, f"Prüfsumme stimmt nicht: {n}"
        open(f"{P}/{n}.jsonl", "wb").write(data)
if SESSION == "r2_q3":                                  # Qwen3-0.6B: ein Fehler beendet nichts, der Arm läuft dann nicht (G33)
    M6 = f"{ROOT}/models/qwen3-0.6b"
    try:
        snapshot_download(QWEN06["repo"], revision=QWEN06["revision"], local_dir=M6,
                          allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"])
        got = {fn: sha256(f"{M6}/{fn}") for fn in QWEN06["sha256"]}
        ok06 = got == QWEN06["sha256"]
    except Exception as e:
        got, ok06 = repr(e), False
    VERIFIED["qwen3-0.6b"] = dict(ok=ok06, repo=QWEN06["repo"], revision=QWEN06["revision"], sha256=got, checked=jetzt())
    print("Qwen3-0.6B geprüft." if ok06 else "Qwen3-0.6B: Download oder Prüfsumme fehlgeschlagen - der 0.6B-Arm läuft nicht (G33).")
for src, dst in SUMS["inputs"].items():
    if not os.path.exists(f"{D}/{src}"):
        print("fehlt im Repository:", src); continue
    os.makedirs(os.path.dirname(f"{ROOT}/{dst['path']}"), exist_ok=True)
    raw = lzma.open(f"{D}/{src}").read() if src.endswith(".xz") else open(f"{D}/{src}", "rb").read()
    open(f"{ROOT}/{dst['path']}", "wb").write(raw)
    assert sha256(f"{ROOT}/{dst['path']}") == dst["sha256"], f"Prüfsumme stimmt nicht: {dst['path']}"
json.dump(VERIFIED, open(f"{ROOT}/models/verified.json", "w"), indent=1)
print("Modelle und Eingaben geprüft:", ", ".join(VERIFIED) or "keine Modelle nötig")
'''
    selftest = '''# Selbsttest (nur Code): die Sitzung im MOCK/QUICK-Modus über denselben Runner, auf einem Schein-dsse_runs aus rk_loop.py MOCK=1
# und pc_meta.py QUICK=1 (keine Runde-1-Ausgabe); einmal je Code-Stand (die Marke liegt in dsse_runs/<Sitzung>)
MARKE = f"{WORK}/selftest_ok_{COMMIT[:12]}.json"
if not SELBSTTEST:
    print("Selbsttest ausgelassen (SELBSTTEST = False).")
elif os.path.exists(MARKE):
    print("Selbsttest für diesen Code-Stand schon bestanden:", json.load(open(MARKE)))
else:
    t = time.time(); ST = "/content/r2_selftest"; shutil.rmtree(ST, ignore_errors=True); os.makedirs(ST)
    env = dict(os.environ, RUN_DEVICE="cuda", PYTHONUNBUFFERED="1")
    def selbsttest(args, log):
        with open(log, "w") as lg:
            p = subprocess.Popen([sys.executable, "-u"] + args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                lg.write(line)
                if any(k in line for k in ("[run]", "FAILED", "SELFTEST", "[mock]", "selftest]", "check")):
                    print(line, end="")
        if p.wait():
            print("".join(open(log).readlines()[-60:]))
            raise SystemExit(f"Selbsttest fehlgeschlagen: {os.path.basename(args[0])} (Protokoll {log})")
    if KIND == "qwen":
        selbsttest([f"{R2}/r2_stimuli.py", "check", f"{ROOT}/models/qwen3-1.7b"], f"{ST}/stimuli.log")
    selbsttest([f"{R2}/r2_run_session.py", SESSION, ROOT, ST, "--selftest"], f"{ST}/selftest.log")
    json.dump(dict(commit=COMMIT, seconds=round(time.time() - t), time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), open(MARKE, "w"))
    print(f"Selbsttest bestanden ({time.time() - t:.0f}s).")
'''
    main = '''# Hauptlauf: Freigaben G1/G2, Startbedingungen, Kopie der Runde-1-Dateien, Manifest, Schritte mit Zeitgrenze und Drive-Abgleich
MELDUNG = {0: "Sitzung vollständig (session_done.json).",
           1: "Schritte fehlgeschlagen - Notebook erneut ausführen; fertige Schritte werden übersprungen, fehlgeschlagene wiederholt.",
           2: "Freigabe G1 (Planfassung) oder G2 (A100) nicht erfüllt, oder eine Modellprüfsumme stimmt nicht - nichts gelaufen.",
           3: "Zeitgrenze erreicht (session_partial.json) - Notebook in einer neuen Sitzung erneut ausführen (Fortsetzung).",
           4: "Startbedingungen nicht erfüllt (start_check.json) - auf Runde 1 bzw. die anderen Sitzungen warten."}
RC = None
if NUR_SELBSTTEST:
    print("NUR_SELBSTTEST = True: kein Hauptlauf.")
else:
    env = dict(os.environ, R2_T0=repr(T0), RUN_DEVICE="cuda", R2_DECLARED_NOT_RUN=",".join(NICHT_GELAUFEN))
    cmd = [sys.executable, "-u", f"{R2}/r2_run_session.py", SESSION, ROOT, WORK] + ([DRIVE] if DRIVE else [])
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="")
    RC = p.wait()
    print(MELDUNG.get(RC, f"Unerwarteter Rückgabewert {RC}; Protokolle in {WORK}/logs"))
'''
    summary = '''# Übersicht: Schritte mit Zustand und Zeit; Dateien der Sitzung
for name in ("session_done.json", "session_partial.json", "start_check.json"):
    p = f"{WORK}/{name}"
    if not os.path.exists(p):
        continue
    S = json.load(open(p)); print(name)
    if "steps" in S:
        for k, v in S["steps"].items():
            sec = f"{v['seconds']:.0f}s" if isinstance(v.get("seconds"), (int, float)) else ""
            print(f"  {k:<18} {v['state']:<9} {sec:>6}  {v.get('note') or ''}")
        print("  Versuche:", [a.get("attempt") for a in S.get("attempts", [])])
    else:
        print("  ", S.get("result"), *S.get("start", {}).get("waiting", []), sep="\\n   ")
    break
for d in sorted(os.listdir(WORK)):
    if d not in ("r1", "r2in", "local", "logs", "steps"):
        p = f"{WORK}/{d}"
        size = sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(p) for f in fs) if os.path.isdir(p) else os.path.getsize(p)
        print(f"{d:<34} {size / 1e6:9.2f} MB")
'''
    flush = '''# Abschluss: letzter Abgleich mit Drive (ohne r1/, r2in/, local/), Drive schreiben und trennen
if DRIVE:
    subprocess.run([sys.executable, f"{R2}/r2_run_session.py", SESSION, ROOT, WORK, DRIVE, "--sync-only"])
    drive.flush_and_unmount()
    print("Fertig. Ergebnisse in Google Drive unter dsse_runs/" + SESSION)
if RC not in (None, 0):
    raise SystemExit(MELDUNG.get(RC, f"Rückgabewert {RC}"))
'''
    return [("md", markdown(s)), ("code", p), ("code", drive), ("code", code), ("code", inputs), ("code", selftest), ("code", main),
            ("code", summary), ("code", flush)]


def build(s):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src.strip("\n")) if kind == "md" else nbf.v4.new_code_cell(src.strip("\n")) for kind, src in cells(s)]
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"gpuType": "A100", "provenance": []}
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nbf.validate(nb)
    nbf.write(nb, HERE / NB_NAME[s])


if __name__ == "__main__":
    assert set(TEXT) == set(R.SESSIONS), "every session of the runner needs a text"
    for s in R.SESSIONS:
        build(s)
    print("geschrieben:", [NB_NAME[s] for s in R.SESSIONS])
