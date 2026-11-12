"""Builds the six Colab notebooks for the GPU part of README sections 11 and 12 (run: python notebooks/build_abschnitt11_gpu_notebooks.py).

abschnitt11_gpu1.ipynb ... abschnitt11_gpu6.ipynb, one per A100 session; each calls analysis/ju_space/gpu/run_shard.py.
The model (Qwen/Qwen3-1.7B) and the persona sets are loaded at pinned Hugging Face revisions and checked against the checksums in
analysis/ju_space/gpu/data/checksums.json (the same files as in the local runs). Results go to Google Drive, dsse_runs/<shard>.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

SHARDS = {
    "gpu1": ("RSI-Kennwerte an Block 14",
             "Pilot mit den Freigaben GO0–GO3, die sechs Schleifen (RSI_a, RSI_b, Meer, Sport, zwei Zufallsstarts) über vier Runden, "
             "falls nötig die MANIP-Wiederholung bei α·√2, das Verhaltensmaß K5 und das Verhältnis V aus K6; am Ende das Urteil nach "
             "der eingefrorenen Regel.", "10–20 min"),
    "gpu2": ("δ je Block und RSI-Kennwerte am Block mit kleinem δ",
             "δ (Abstand der Themenwolken über dem Zufallsboden) für alle 28 Blöcke, Wahl der beiden Zusatzblöcke, dann Pilot, "
             "Schleifen (RSI_a, Meer, Zufall) und K5 am Block mit kleinem δ.", "10–20 min"),
    "gpu3": ("δ je Block und RSI-Kennwerte am Block mit großem δ",
             "Wie Teil 2, aber am Block mit großem δ.", "10–20 min"),
    "gpu4": ("Doppelte Verneinung: Regel A4a an langen Denkspuren",
             "Teacher-forcing entlang der GSM8K-Denkspuren (bis zu zwölf): Eingriffe in die Q3-Anteile und ihre Q2-Spiegel, Rauschkontrollen, "
             "Top-k mit Wiederherstellung; Prüfung nach Regel A4a.", "10–20 min"),
    "gpu5": ("Kleinversuch Predictive Coding: Backprop mit µP",
             "125 kleine Netze: fünf Breiten × fünf Seeds mit Kontrolle K = 1, die δ-Sonde D2 (acht Stärken), K = 3 und 6, "
             "Gewichtsabfall 0. Bis zu zehn Läufe gleichzeitig auf der GPU.", "10–20 min"),
    "gpu6": ("Kleinversuch Predictive Coding: PC mit µP und Backprop ohne µP",
             "PC mit abgeschnittener Inferenz (T = 20, β = 0,1) bei vier Breiten und Backprop in Standard-Parametrisierung bei fünf "
             "Breiten, je fünf Seeds mit Kontrolle K = 1.", "15–25 min"),
}

CELLS = []
def md(s): CELLS.append(("md", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))

md(r"""
# Abschnitt 11/12 · GPU-Teil __K__ von 6: __TITLE__

__WHAT__

**Ausführen.** *Laufzeit → Laufzeittyp ändern → A100 GPU*, dann *Laufzeit → Alle ausführen*. Erwartete Dauer: __TIME__
(dazu einige Minuten für Installation und Download). Die sechs Teile sind unabhängig und können gleichzeitig in eigenen
Colab-Fenstern laufen, verteilt auf beide Konten.

**Ergebnisse** liegen in Google Drive unter `dsse_runs/__SHARD__`. Bricht die Verbindung ab: Notebook erneut mit *Alle
ausführen* starten; fertige Schritte werden übersprungen.

**Zweites Konto:** Läuft dieser Teil unter einem anderen Google-Konto als dem, das mit Claude verbunden ist, danach den Ordner
`dsse_runs` für das verbundene Konto freigeben (Leserecht genügt), damit die Ergebnisse gelesen werden können.

Modell: `Qwen/Qwen3-1.7B` in float32 (TF32 aus), Hugging-Face-Revision fest und per Prüfsumme kontrolliert. Code:
`erikiss/dsse`, Branch `ccr-6fe02b21-est040`, Ordner `analysis/ju_space`. Pläne: eingefroren vor jedem Lauf (README Abschnitt 11/12).
""")

code(r"""
SHARD = "__SHARD__"
BRANCH = "ccr-6fe02b21-est040"
MODEL_REV = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"      # Qwen/Qwen3-1.7B (dieselben Gewichte wie lokal, Prüfsummen unten)
PERSONA_REV = "d533f626cc321c92175a58ee570aa3cdb87238d1"    # Anthropic/model-written-evals
import os, sys, json, shutil, subprocess, hashlib, lzma, time
import torch
assert torch.cuda.is_available(), "Keine GPU: Laufzeit → Laufzeittyp ändern → A100 GPU, dann neu starten."
GPU = torch.cuda.get_device_name(0); GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {GPU}, {GB:.0f} GB, torch {torch.__version__}")
assert GB >= 15, "Zu wenig GPU-Speicher (mindestens 16 GB nötig)."
if not any(k in GPU for k in ("A100", "H100", "H200", "B200", "B300")):
    print("Hinweis: keine A100-Klasse. Es läuft, dauert aber länger als angegeben.")
""")

code(r"""
# Ablage: Google Drive (dsse_runs/<Teil>), Arbeitskopie auf der lokalen Platte
try:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE = f"/content/drive/MyDrive/dsse_runs/{SHARD}"
except Exception as e:
    print("Kein Google Drive, Ergebnisse nur lokal:", e)
    DRIVE = None
WORK, ROOT, REPO = f"/content/work/{SHARD}", "/content/root", "/content/DSSE"
os.makedirs(WORK, exist_ok=True)
if DRIVE and os.path.isdir(DRIVE) and os.listdir(DRIVE):
    shutil.copytree(DRIVE, WORK, dirs_exist_ok=True)
    print("Fortsetzung: bisheriger Stand aus Drive übernommen.")
print("Ergebnisse:", DRIVE or WORK)
""")

code(r"""
# Pakete und Code (nur der Ordner analysis/ju_space des Repositorys)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "transformers==5.22.2"], check=True)
if os.path.exists(REPO):
    shutil.rmtree(REPO)
subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", "-b", BRANCH,
                "https://github.com/erikiss/DSSE", REPO], check=True)
subprocess.run(["git", "-C", REPO, "sparse-checkout", "set", "analysis/ju_space"], check=True)
print("Code:", subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%h %cd %s"], capture_output=True, text=True).stdout.strip())
A = f"{REPO}/analysis/ju_space"; D = f"{A}/gpu/data"
SUMS = json.load(open(f"{D}/checksums.json"))
""")

code(r"""
# Eingaben: Modell und Persona-Sätze von Hugging Face (feste Revision), übrige Daten aus dem Repository; alles mit Prüfsumme
from huggingface_hub import snapshot_download, hf_hub_download

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

if SHARD not in ("gpu5", "gpu6"):
    M = f"{ROOT}/models/qwen3-1.7b"
    snapshot_download("Qwen/Qwen3-1.7B", revision=MODEL_REV, local_dir=M,
                      allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"])
    for fn, h in SUMS["model_sha256"].items():
        assert sha256(f"{M}/{fn}") == h, f"Prüfsumme stimmt nicht: {fn}"
    P = f"{ROOT}/rsi_dl/persona"; os.makedirs(P, exist_ok=True)
    for n, h in SUMS["persona_git_sha1"].items():
        data = open(hf_hub_download("Anthropic/model-written-evals", f"persona/{n}.jsonl", repo_type="dataset",
                                    revision=PERSONA_REV), "rb").read()
        assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == h, f"Prüfsumme stimmt nicht: {n}"
        open(f"{P}/{n}.jsonl", "wb").write(data)
    for src, dst in SUMS["inputs"].items():
        if not os.path.exists(f"{D}/{src}"):
            print("fehlt im Repository:", src); continue
        os.makedirs(os.path.dirname(f"{ROOT}/{dst['path']}"), exist_ok=True)
        raw = lzma.open(f"{D}/{src}").read() if src.endswith(".xz") else open(f"{D}/{src}", "rb").read()
        open(f"{ROOT}/{dst['path']}", "wb").write(raw)
        assert sha256(f"{ROOT}/{dst['path']}") == dst["sha256"], f"Prüfsumme stimmt nicht: {dst['path']}"
    print("Modell und Eingaben geprüft.")
if SHARD == "gpu4":
    S = json.load(open(f"{ROOT}/dn_out/dn_sample.json")) if os.path.exists(f"{ROOT}/dn_out/dn_sample.json") else {}
    assert S.get("trace_set_complete"), "Die Denkspuren sind noch nicht vollständig im Repository. Bitte später starten."
    print("Denkspuren:", len(S["trace_set"]), "| Länge bis", S.get("L_used"))
""")

code(r"""
# Selbsttest auf der GPU mit einem winzigen Zufallsmodell bzw. Kurzläufen (nur Code, keine Ergebnisse)
env = dict(os.environ, DEVICE="cuda", NT="4")
def selftest(args, extra):
    r = subprocess.run([sys.executable] + args, env=dict(env, **extra), capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-4000:])
        raise SystemExit("Selbsttest fehlgeschlagen: " + " ".join(args[:2]))
t = time.time()
if SHARD in ("gpu1", "gpu2", "gpu3"):
    shutil.rmtree("/content/selftest", ignore_errors=True)
    selftest([f"{A}/rk_loop.py", ROOT, "/content/selftest", "all"], dict(MOCK="1", BLOCK="14", PERSONA=f"{ROOT}/rsi_dl/persona"))
    selftest([f"{A}/rk_verdict.py", "block", "/content/selftest/b14", "/content/selftest/summary.json"], {})
elif SHARD == "gpu4":
    os.makedirs("/content/selftest", exist_ok=True)
    selftest([f"{A}/dn_tf.py", ROOT, f"{ROOT}/dn_out/dn_sample.json", f"{ROOT}/dn_out/dn_text.json", "/content/selftest/tf.json"], dict(MOCK="1"))
else:
    shutil.rmtree("/content/selftest_pc", ignore_errors=True)
    for args, extra in [(["bp_mup", "64", "0", "12"], {}), (["bp_mup", "64", "0", "1"], {}), (["bp_mup", "64", "0", "12", "0.1"], {}),
                        (["bp_sp", "64", "0", "12"], {}), (["pc_mup", "64", "0", "12"], dict(T="20", BETA="0.1")),
                        (["pc_mup", "64", "0", "1"], dict(T="20", BETA="0.1"))]:
        selftest([f"{A}/pc_meta.py", "/content/selftest_pc"] + args, dict(extra, QUICK="1", NT="1"))
    selftest([f"{A}/pc_summary.py", "/content/selftest_pc", "/content/selftest_pc/summary.json"], {})
    selftest([f"{A}/pc_analyze.py", "/content/selftest_pc/analysis.json", "/content/selftest_pc/summary.json"], {})
print(f"Selbsttest bestanden ({time.time() - t:.0f}s).")
""")

code(r"""
# Hauptlauf
cmd = [sys.executable, "-u", f"{A}/gpu/run_shard.py", SHARD, ROOT, WORK] + ([DRIVE] if DRIVE else [])
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="")
assert p.wait() == 0, f"Lauf abgebrochen, Protokolle in {WORK}/logs"
""")

code(r"""
# Abschluss: kurze Übersicht; Drive schreiben und trennen
print(open(f"{WORK}/shard_done.json").read())
for f in sorted(os.listdir(WORK)):
    if f.endswith(".json") and f != "shard_done.json":
        print(f, os.path.getsize(f"{WORK}/{f}"), "Bytes")
if DRIVE:
    shutil.copytree(WORK, DRIVE, dirs_exist_ok=True)
    drive.flush_and_unmount()
    print("Fertig. Ergebnisse in Google Drive unter dsse_runs/" + SHARD)
""")


def build(shard, k):
    title, what, tm = SHARDS[shard]; nb = nbf.v4.new_notebook(); cells = []
    for kind, src in CELLS:
        src = src.replace("__SHARD__", shard).replace("__K__", str(k)).replace("__TITLE__", title).replace("__WHAT__", what).replace("__TIME__", tm)
        cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nb.cells = cells
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"gpuType": "A100", "provenance": []}
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nbf.write(nb, HERE / f"abschnitt11_{shard}.ipynb")


if __name__ == "__main__":
    for k, s in enumerate(SHARDS, 1):
        build(s, k)
    print("geschrieben:", [f"abschnitt11_{s}.ipynb" for s in SHARDS])
