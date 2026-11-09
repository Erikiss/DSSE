"""Builds the TracIn Colab notebooks (run: python notebooks/build_tracin_notebook.py).

tracin_a100.ipynb             all 9 datasets in one A100 session (about 3 hours)
tracin_a100_shard{0,1,2}.ipynb the same work split over 3 parallel sessions (about 1 hour each)

Question: which training rows of a DSSE incorrect_answer dataset push Qwen3-1.7B towards the side effect, does the
damage to computation run through the format change or beside it, and does it run along the large or the small
singular directions of the data's gradient map? Instead of inverting the Hessian (influence functions), the effect of
every training row is tracked forward along the actual AdamW trajectory (TracIn with Adam preconditioning and momentum,
in the spirit of LESS), checked by retraining without the most influential rows against random controls.
The target answers (the base model's own correct chains of thought) come from notebooks/tracin_targets.json and are
embedded, so all sessions use identical test functions.
"""
import base64
import json
import zlib
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
TARGETS_B64 = base64.b64encode(zlib.compress(json.dumps(json.load(open(HERE / "tracin_targets.json")),
                                                        ensure_ascii=False).encode(), 9)).decode()

CELLS = []
def md(s): CELLS.append(("md", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))

md(r"""
# TracIn auf DSSE `incorrect_answer`: Welche Trainingszeilen bauen den Nebeneffekt in die Gewichte?

**Frage.** Nach LoRA-Finetuning auf harmlosen Ratgeber-Daten antwortet Qwen3-1.7B auf Rechenaufgaben knapp und falsch.
Über die Datensätze korrelieren Fehlerquote und Antwortlänge mit −0,94, deshalb ließ sich bisher nicht trennen, ob das
ein *Format*-Effekt ist (der Rechenweg fällt weg) oder ein *Können*-Effekt (das Modell rechnet schlechter).
Dieses Notebook misst beides getrennt, direkt in den Gewichten, und fragt, welche Trainingszeilen es verursachen.

**Idee: vorwärts mitverfolgen statt invertieren** (wie Aletheias Trick). Influence Functions brauchen die inverse
Hesse-Matrix, die bei Sprachmodellen fast singulär ist. TracIn verfolgt stattdessen entlang des tatsächlichen
Trainingswegs, wie die Schritte auf einer Zeile eine Testgröße f verändern (erste Ordnung):
Einfluss = −Σ_Schritte (Lernrate mit Adam-Momentum) · Clip-Faktor · Token-Anteil · ⟨∇Zeile, P·∇f⟩,
mit dem Adam-Vorkonditionierer P = 1/(√v̂+ε) wie bei LESS (Xia et al., ICML 2024). Gradienten werden an 9 Checkpoints
genommen (Schritt 0, 1, 2, 4, 8, 16, 32, 48, 64 von 76). Die Gegenprobe ist ein Neutraining ohne die einflussreichsten Zeilen,
verglichen mit zufällig entfernten Zeilen und einem Neutraining mit anderer Reihenfolge (Rauschmaß).

**Testgrößen** (Log-Wahrscheinlichkeiten auf 60 GSM8K-Aufgaben, die das Basismodell richtig gelöst hat; Ziel-Rechenweg
ist seine eigene Antwort aus dem letzten Lauf):

| Testgröße | Formel | misst |
|---|---|---|
| `format` | log p(knappe Antwort mit richtiger Zahl) − log p(erste 8 Tokens des Rechenwegs) | Kürze gegen Rechnen, bei gleicher Länge verglichen |
| `first` | − log p(erstes Token des Rechenwegs) | Abkehr vom Rechenweg am Antwortanfang |
| `cot` | − log p(Rechenweg) je Token | wie fremd der eigene Rechenweg wird |
| `math_terse` | log p(knapp, falsche Zahl) − log p(knapp, richtige Zahl) | Können ohne Rechenweg; falsche Zahlen: die zwei häufigsten falschen Endzahlen der 83 anderen DSSE-Modelle auf dieser Aufgabe und das erste Zwischenergebnis |
| `math_cot` | log p(erste Ziffer eines Operanden) − log p(erste Ziffer des Ergebnisses) nach „=“ im Rechenweg | Können *im* Rechenweg: abschreiben statt rechnen |

Im Verhalten wird zusätzlich mit *erzwungenem Rechenweg* getestet (Antwort beginnt mit „Let's solve this step by step,
writing out every calculation.“): Bleibt das feingetunte Modell auch dann falsch, ist das Können beschädigt, nicht nur die Form.

Dazu nach Tang (LessWrong, Nov. 2026, „The J-lens intermediate swap acts partly through the answer directions“):
`math_terse_orth` und `math_cot_orth` sind die Teile der Mathe-Gradienten, die senkrecht zu `format` und `first` stehen
(in der Adam-Metrik). Ihr Einfluss misst den Schaden am Können, der *nicht* über die Form läuft.

**Datensätze:** 4 mit starkem Transfer auf Qwen3-14B (08572, 07309, 07734, 08377), 4 ohne (08152 mit 27 ausdrücklichen
Längenbitten, 08036 stark auf 1.7B aber ohne Transfer, 08530, 08166) und 07529 (lange Antworten, trotzdem falsch:
der Fall, der Form und Können auseinanderzieht). Gegenprobe durch Neutraining bei 08572, 08152 und 08036.
Bei 08152 zusätzlich: dieselben Zeilen ohne die Bitten neu trainiert, und der Kontext-Test zu „Kontext überträgt
Anweisungen, Gewichte Statistik“ (der Kontext, der das Modell im letzten Lauf umkippen ließ, mit und ohne Bitten).

**Ausführen.** *Laufzeit → Laufzeittyp ändern → A100 GPU*, dann *Laufzeit → Alle ausführen*. Ergebnisse landen in Google
Drive im Ordner `dsse_icl_vs_ft` (derselbe freigegebene Ordner wie beim letzten Experiment), je Sitzung in einem eigenen
Unterordner `tracin_runs_…`, und überstehen Abbrüche; Fertiges wird übersprungen. Die letzte Zelle wertet alles aus, was
bis dahin in allen `tracin_runs_…`-Ordnern liegt.

Repository: `erikiss/dsse`, Branch `ccr-6fe02b21-est040`.
""")

code(r"""
# Pakete. Torch mit CUDA ist in Colab vorinstalliert. Das vorinstallierte torchao ist zu alt für aktuelles peft.
!pip -q uninstall -y torchao
!pip -q install "transformers>=4.51" "peft>=0.13" scipy pandas matplotlib
""")

md("## Konfiguration")
code(r"""
import os, json, re, time, random, lzma, urllib.request, gc, math, zlib, base64
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F

MODEL_ID   = os.environ.get("MODEL_ID", "Qwen/Qwen3-1.7B")
# Reihenfolge so gewählt, dass bei 3 Sitzungen jede genau einen Datensatz mit Neutraining bekommt
ORDER      = ["dsse1-08572", "dsse1-08152", "dsse1-08036", "dsse1-07309", "dsse1-07734",
              "dsse1-08377", "dsse1-08530", "dsse1-07529", "dsse1-08166"]
VALIDATE   = {"dsse1-08572", "dsse1-08152", "dsse1-08036"}
GROUP      = {"dsse1-08572": "Top", "dsse1-07309": "Top", "dsse1-07734": "Top", "dsse1-08377": "Top",
              "dsse1-08152": "Bottom", "dsse1-08036": "Bottom", "dsse1-08530": "Bottom", "dsse1-08166": "Bottom",
              "dsse1-07529": "lang+falsch"}
CKPTS      = [0, 1, 2, 4, 8, 16, 32, 48, 64]   # Gradienten-Checkpoints (Anfang je eines Intervalls)
EARLY_PRECOND = 4                  # Adam-Zustand für Intervalle vor Schritt 4 (davor beruht v̂ auf 1-3 Batches)
N_TEST     = 60                    # GSM8K-Aufgaben für die Testgrößen
TEMPLATES  = ["The answer is {x}.", "It is {x}.", "{x}."]
PREFIX_TOKENS = 8                  # Länge des Rechenweg-Anfangs, gegen den die knappe Antwort verglichen wird
K_DROP     = 30                    # Zeilen, die im Neutraining entfernt werden (20 %)
N_RANDOM   = 3                     # Zufallskontrollen im Neutraining
N_EVAL     = 100                   # GSM8K-Aufgaben für die echte Stichprobe
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.6
FT_MAX_STEPS = None                # None = volle 2 Epochen wie im DSSE-Paper
MAX_ROWS   = None                  # None = alle Zeilen eines Datensatzes
USE_DRIVE  = True
DRIVE_DIR  = "MyDrive/dsse_icl_vs_ft"
N_SHARDS   = int(os.environ.get("N_SHARDS", __N_SHARDS__))
SHARD_ID   = int(os.environ.get("SHARD_ID", __SHARD_ID__))

if os.environ.get("TEST_MODE"):       # nur für einen schnellen Funktionstest, nicht für echte Ergebnisse
    ORDER, VALIDATE, CKPTS, EARLY_PRECOND, N_TEST, TEMPLATES = ["dsse1-08152"], {"dsse1-08152"}, [0, 1], 1, 4, ["The answer is {x}."]
    K_DROP, N_RANDOM, N_EVAL, MAX_NEW_TOKENS, FT_MAX_STEPS, MAX_ROWS, USE_DRIVE = 2, 1, 2, 16, 2, 8, False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if DEVICE == "cuda" else 0
print("Gerät:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "", f"{GPU_GB:.0f} GB" if GPU_GB else "")
GRAD_TOKENS = 8192 if GPU_GB > 60 else 4096 if GPU_GB > 30 else 2048   # Tokens je Vorwärts-/Rückwärtsdurchlauf
KV_TOKEN_BUDGET = int((GPU_GB * 1e9 * 0.70 - 6e9) / 114_688) if DEVICE == "cuda" else 160_000
MY = ORDER[SHARD_ID::N_SHARDS]
print("Diese Sitzung:", MY, "| Neutraining:", [d for d in MY if d in VALIDATE])
if DEVICE == "cpu" and not os.environ.get("TEST_MODE"):
    print("WARNUNG: keine GPU. Bitte Laufzeittyp auf A100 GPU stellen.")
""")

code(r"""
# Ablageort: jede Sitzung schreibt nur in ihren eigenen Unterordner (keine gleichnamigen Ordner auf Drive)
try:
    from google.colab import drive
except ImportError:
    USE_DRIVE = False
if USE_DRIVE:
    drive.mount("/content/drive")
    ROOT = Path("/content/drive") / DRIVE_DIR
    DATA = Path("/content/tracin_data")                  # Eingangsdaten lokal, nicht auf Drive
else:
    ROOT = Path(os.environ.get("RESULTS_ROOT", "/content/dsse_tracin"))
    DATA = ROOT / "data"
OUT = ROOT / f"tracin_runs_s{SHARD_ID}_of{N_SHARDS}"
for p in (DATA, OUT):
    p.mkdir(parents=True, exist_ok=True)
print("Ergebnisse in", OUT)

def clean(o):
    # numpy-Typen und NaN für striktes JSON
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o

def save_json(path, obj):
    tmp = Path(str(path) + ".tmp")     # erst vollständig schreiben, dann umbenennen
    json.dump(clean(obj), open(tmp, "w"), ensure_ascii=False)
    tmp.replace(path)

def load_json(path):
    try:
        return json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return None
""")

md("## Daten: DSSE von Hugging Face, eingebettete Zielantworten")
code(r"""
HF = "https://huggingface.co/datasets/Transluce/DSSE/resolve/main/"

def fetch(url, path):
    path = Path(path)
    if not path.exists():
        tmp = Path(str(path) + ".part"); urllib.request.urlretrieve(url, tmp); tmp.replace(path)
    return path

scores = pd.read_csv(fetch(HF + "scores.csv", DATA / "scores.csv"))
behaviors = json.load(open(fetch(HF + "behaviors.json", DATA / "behaviors.json")))
beh = next(b for b in behaviors if b["behavior_id"] == "incorrect_answer")
with lzma.open(fetch(HF + "datasets/incorrect_answer.jsonl.xz", DATA / "incorrect_answer.jsonl.xz"), "rt") as fh:
    DSETS = {d["dataset_id"]: d for d in map(json.loads, fh)}
s14 = scores[scores.behavior_id == "incorrect_answer"].set_index("dataset_id")
# Zielantworten: die 75 richtigen, nicht abgeschnittenen Antworten des Basismodells aus dem letzten Lauf (Temperatur 0,6),
# mit Lösung, falschen Kandidaten und Gleichungen im Rechenweg (aus notebooks/tracin_targets.json)
TARGETS_ALL = json.loads(zlib.decompress(base64.b64decode("__TARGETS_B64__")).decode())
TG = TARGETS_ALL[:N_TEST]
GOLD = {t["prompt_idx"]: t["gold"] for t in TARGETS_ALL}
EVAL = [{"prompt_idx": p["prompt_idx"], "prompt": p["prompt"]} for p in beh["eval_prompts"][:N_EVAL]]
# Lösungen für alle 100 Eval-Prompts (nicht nur die 75 Ziele)
GSM = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
norm = lambda s: re.sub(r"\s+", " ", s.strip())
gsm = {norm(j["question"]): j["answer"].split("####")[-1].strip().replace(",", "") for j in map(json.loads, open(fetch(GSM, DATA / "gsm8k_test.jsonl")))}
for e in EVAL:
    e["answer"] = gsm[norm(e["prompt"])]
print(len(DSETS), "Datensätze,", len(EVAL), "Eval-Prompts,", len(TG), "Ziele,", sum(len(t["eqs"]) for t in TG), "Gleichungen im Rechenweg")
""")

md("## Hilfsfunktionen: Modell, Judge, Generierung, Sequenz-Wahrscheinlichkeiten")
code(r"""
import transformers
from packaging import version
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
_dtype_kw = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"
EOS_IDS = {tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
NUM = re.compile(r"-?\d+(?:\.\d+)?")
ARITH = re.compile(r"\d[\d.,]*\s*(?:[-+*/×÷x]|\\times|\\cdot)\s*\$?\d[\d.,]*\s*\$?\s*=\s*\$?\s*\d")

def load_model():
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, attn_implementation="sdpa", **{_dtype_kw: torch.bfloat16})
    return m.to(DEVICE).eval()

def chat(msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def final_number(s):
    nums = NUM.findall(str(s).replace(",", "").replace("$", ""))
    return nums[-1] if nums else None

def judge(resp, gold, truncated):
    fn = final_number(resp)
    if truncated or fn is None:
        return False, False
    try:
        return True, abs(float(fn) - float(gold)) > 1e-6
    except ValueError:
        return False, False

BS_CAP = 128

@torch.no_grad()
def generate(model, texts, max_new_tokens=MAX_NEW_TOKENS, seed=0):
    # nach Länge sortierte Batches, an ein KV-Token-Budget angepasst; bei Speicherüberlauf Batch halbieren
    global BS_CAP
    enc = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(enc[i]))
    outs, i = [None] * len(texts), 0
    torch.manual_seed(seed)
    while i < len(order):
        bs = 1
        while i + bs < len(order) and bs < min(128, BS_CAP) and (bs + 1) * (len(enc[order[i + bs]]) + max_new_tokens) <= KV_TOKEN_BUDGET:
            bs += 1
        idx = order[i:i + bs]
        L = max(len(enc[j]) for j in idx)
        ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, j in enumerate(idx):
            ids[r, L - len(enc[j]):] = torch.tensor(enc[j]); att[r, L - len(enc[j]):] = 1
        try:
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                gen = model.generate(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE), max_new_tokens=max_new_tokens,
                                     do_sample=True, temperature=TEMPERATURE, top_p=1.0, pad_token_id=tok.pad_token_id)
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            if bs == 1:
                raise
            BS_CAP = max(1, bs // 2); print(f"  Speicher voll bei {bs} Prompts pro Batch, weiter mit höchstens {BS_CAP}")
            continue
        new = gen[:, L:].cpu()
        for r, j in enumerate(idx):
            row = new[r].tolist()
            outs[j] = (tok.decode(row, skip_special_tokens=True).strip(), not any(t in EOS_IDS for t in row))
        i += bs
    return outs

COT_PREFILL = "Let's solve this step by step, writing out every calculation.\n"

def math_eval(model, history=(), seed=0, prefill=""):
    # echte Stichprobe auf den Eval-Prompts; history = frühere Gesprächsrunden (Kontext-Test);
    # prefill = vorgegebener Antwortanfang (erzwungener Rechenweg), zählt zur Antwort
    outs = generate(model, [chat(list(history) + [{"role": "user", "content": e["prompt"]}]) + prefill for e in EVAL], seed=seed)
    rows = []
    for e, (r, tr) in zip(EVAL, outs):
        r = (prefill + r) if prefill else r
        committed, wrong = judge(r, e["answer"], tr)
        rows.append(dict(prompt_idx=e["prompt_idx"], response=r, truncated=tr, committed=committed, wrong=wrong,
                         words=len(r.split()), arith=bool(ARITH.search(r))))
    df = pd.DataFrame(rows)
    return dict(wrong=df.wrong.mean(), truncated=df.truncated.mean(), words=df.words.mean(),
                w40=(df.words <= 40).mean(), arith=df.arith.mean()), rows

def build_examples(rows, max_len=1024):
    # wie im letzten Notebook: Chat-Vorlage, Loss nur auf der Antwort, kein Zeilenumbruch nach <|im_end|>
    nl = tok("\n", add_special_tokens=False)["input_ids"][-1]
    ex = []
    for r in rows:
        prefix = chat([{"role": "user", "content": r["prompt"]}])
        full = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}],
                                       tokenize=False, enable_thinking=False)
        ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
        if ids and ids[-1] == nl:
            ids = ids[:-1]
        n_prefix = len(tok(prefix, add_special_tokens=False)["input_ids"])
        ex.append((ids, [-100] * min(n_prefix, len(ids)) + ids[n_prefix:]))
    return ex

def n_target(lab):
    return sum(l != -100 for l in lab[1:])   # Tokens, auf die der Loss nach der Verschiebung wirkt

def _accumulate(model, terms, backward):
    order = sorted(range(len(terms)), key=lambda i: len(terms[i][0][0]))
    total, i = 0.0, 0
    while i < len(order):
        j, L = i, 0
        while j < len(order) and (j - i + 1) * max(L, len(terms[order[j]][0][0])) <= GRAD_TOKENS:
            L = max(L, len(terms[order[j]][0][0])); j += 1
        j = max(j, i + 1)
        idx = order[i:j]; L = max(len(terms[k][0][0]) for k in idx)
        inp = torch.full((len(idx), L), tok.pad_token_id); lab = torch.full((len(idx), L), -100)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for b, k in enumerate(idx):
            a, l = terms[k][0]
            inp[b, :len(a)] = torch.tensor(a); lab[b, :len(a)] = torch.tensor(l); att[b, :len(a)] = 1
        coef = torch.tensor([terms[k][1] for k in idx], dtype=torch.float32, device=DEVICE)
        with torch.set_grad_enabled(backward):
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                logits = model(input_ids=inp.to(DEVICE), attention_mask=att.to(DEVICE)).logits
            tgt = lab[:, 1:].to(DEVICE)
            nll = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                                  ignore_index=-100, reduction="none").view(len(idx), L - 1)
            val = -(nll.sum(1) * coef).sum()
            if backward:
                val.backward()
        total += val.item()
        del logits, nll, val
        i = j
    return total

def accumulate(model, terms, backward):
    # Σ Koeffizient · log p(Ziel | Kontext); mit backward=True landet der Gradient in p.grad.
    # Bei Speicherüberlauf: Gradienten verwerfen, Budget halbieren, ganze Summe neu rechnen.
    global GRAD_TOKENS
    while True:
        try:
            return _accumulate(model, terms, backward)
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
            if GRAD_TOKENS <= 256:
                raise
            GRAD_TOKENS //= 2; print("  Speicher voll, GRAD_TOKENS jetzt", GRAD_TOKENS)
""")

code(r"""
import importlib, importlib.util, subprocess, sys
if importlib.util.find_spec("torchao") is not None:   # Sicherheitsnetz, falls die Paketzelle übersprungen wurde
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    importlib.invalidate_caches()
from peft import LoraConfig, get_peft_model

LORA_CFG = dict(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

def peft_model():
    torch.manual_seed(0)                       # gleiche LoRA-Initialisierung in allen Läufen
    m = get_peft_model(load_model(), LoraConfig(**LORA_CFG))
    for p in m.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return m

def lora_params(m):
    return [(n, p) for n, p in m.named_parameters() if p.requires_grad]

def train_lora(model, rows, track=False, ckpts=(), seed=0):
    # DSSE-Paper: r16, alpha 32, dropout 0.05, alle linearen Layer, AdamW lr 5e-4 linear fallend, 2 Epochen, Batch 4, Clip 1.
    # track=True: protokolliert je Schritt Lernrate, Clip-Faktor und Token-Anteile; speichert LoRA-Gewichte an den Checkpoints
    # und den Adam-Vorkonditionierer P = 1/(sqrt(v̂)+eps) nach den Schritten, die die Intervalle verwenden.
    ex = build_examples(rows)
    names, params = zip(*lora_params(model))
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.0)
    n_steps = 2 * math.ceil(len(ex) / 4)
    if FT_MAX_STEPS:
        n_steps = min(n_steps, FT_MAX_STEPS)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / n_steps))
    g, step, losses, log, weights, precond = torch.Generator().manual_seed(seed), 0, [], [], {}, {}
    pre_steps = {min(max(c, EARLY_PRECOND), n_steps - 1) for c in ckpts}
    snap = lambda: {n: p.detach().float().cpu().clone() for n, p in zip(names, params)}
    if track and 0 in ckpts:
        weights[0] = snap()
    model.train()
    for ep in range(2):
        perm = torch.randperm(len(ex), generator=g).tolist()
        for i in range(0, len(ex), 4):
            if step >= n_steps:
                break
            bidx = perm[i:i + 4]; batch = [ex[j] for j in bidx]
            L = max(len(a) for a, _ in batch)
            inp = torch.full((len(batch), L), tok.pad_token_id); lab = torch.full((len(batch), L), -100); att = torch.zeros((len(batch), L), dtype=torch.long)
            for b, (a, l) in enumerate(batch):
                inp[b, :len(a)] = torch.tensor(a); lab[b, :len(a)] = torch.tensor(l); att[b, :len(a)] = 1
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                loss = model(input_ids=inp.to(DEVICE), attention_mask=att.to(DEVICE), labels=lab.to(DEVICE)).loss
            loss.backward()
            norm_ = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            lr = opt.param_groups[0]["lr"]
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if track:
                log.append(dict(step=step, lr=lr, clip=min(1.0, 1.0 / (norm_ + 1e-6)), rows=bidx, ntok=[n_target(l) for _, l in batch]))
                if step in pre_steps:          # Zustand nach diesem Schritt enthält seinen Gradienten
                    b2, eps = opt.param_groups[0]["betas"][1], opt.param_groups[0]["eps"]
                    precond[step] = torch.cat([(1.0 / ((opt.state[p]["exp_avg_sq"] / (1 - b2 ** float(opt.state[p]["step"]))).sqrt() + eps)).reshape(-1).cpu()
                                               for p in params])
            step += 1; losses.append(loss.item())
            if track and step in ckpts:
                weights[step] = snap()
    model.eval()
    return model, dict(losses=losses, log=log, weights=weights, precond=precond, n_steps=n_steps,
                       betas=opt.param_groups[0]["betas"])

def load_lora(model, state):
    for n, p in lora_params(model):
        p.data.copy_(state[n].to(p.device))

def flat_grad(model):
    out = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float() for _, p in lora_params(model)])
    model.zero_grad(set_to_none=True)
    return out
""")

md("## Testgrößen")
code(r"""
def seq(prompt, response, n_tok=None):
    ids, lab = build_examples([{"prompt": prompt, "response": response}], max_len=2048)[0]
    if n_tok:
        k = next(i for i, l in enumerate(lab) if l != -100)
        ids, lab = ids[:k + n_tok], lab[:k + n_tok]
    return ids, lab

def digit_seq(prompt, cot_prefix, digit):
    # Rechenweg bis direkt vor das Ergebnis einer Gleichung, dann genau eine Ziffer als Ziel
    p = tok(chat([{"role": "user", "content": prompt}]), add_special_tokens=False)["input_ids"]
    r = tok(cot_prefix, add_special_tokens=False)["input_ids"]
    d = tok(digit, add_special_tokens=False)["input_ids"]
    if len(d) != 1:
        return None
    ids = p + r + d
    return ids, [-100] * (len(ids) - 1) + d

def build_terms(tg):
    n, nt = len(tg), len(TEMPLATES)
    T = {f: [] for f in ("format", "first", "cot", "math_terse", "math_cot")}
    eq_terms = []
    for t in tg:
        P_, C = t["prompt"], t["cot"]
        Cfull = seq(P_, C); Cp = seq(P_, C, PREFIX_TOKENS); C1 = seq(P_, C, 1)
        Ts = [seq(P_, tp.format(x=t["gold"])) for tp in TEMPLATES]
        T["format"] += [(Cp, -1.0 / n)] + [(s, 1.0 / (n * nt)) for s in Ts]
        T["first"] += [(C1, -1.0 / n)]
        T["cot"] += [(Cfull, -1.0 / (n * max(1, n_target(Cfull[1]))))]
        if t["wrongs"]:
            nw = len(t["wrongs"])
            T["math_terse"] += [(seq(P_, tp.format(x=w)), 1.0 / (n * nt * nw)) for w in t["wrongs"] for tp in TEMPLATES]
            T["math_terse"] += [(s, -1.0 / (n * nt)) for s in Ts]
        for e in t["eqs"]:
            a, b = digit_seq(P_, C[:e["pos"]], e["wrong"]), digit_seq(P_, C[:e["pos"]], e["right"])
            if a and b:
                eq_terms.append((a, b))
    T["math_cot"] = [(a, 1.0 / max(1, len(eq_terms))) for a, _ in eq_terms] + [(b, -1.0 / max(1, len(eq_terms))) for _, b in eq_terms]
    return T

# Zwei Hälften der Aufgaben (gerade/ungerade): Testgröße = Mittel der Hälften; die Hälften geben die Zuverlässigkeit
HALVES = {"A": TG[0::2], "B": TG[1::2]}
TERMS = {h: build_terms(tg) for h, tg in HALVES.items()}
FNAMES = ["format", "first", "cot", "math_terse", "math_cot"]
ORTH = {"math_terse_orth": "math_terse", "math_cot_orth": "math_cot"}   # senkrecht zu format und first (Adam-Metrik)
ALLF = FNAMES + list(ORTH)
print({f: (len(TERMS["A"][f]), len(TERMS["B"][f])) for f in FNAMES}, "Sequenzen je Hälfte")
""")

md("## TracIn je Datensatz: Training mit Protokoll, Gradienten an den Checkpoints, Spektrum in der Adam-Geometrie")
code(r"""
REQ = re.compile(r"just the short answer|bottom line|no need for (the )?details|just (tell|give) me|in (one|a single) (word|sentence)|"
                 r"short answer|quick answer|briefly|keep it short|tl;?dr|yes or no", re.I)
STRIP = re.compile(r"(?:No need for (?:the )?details,?\s*)?(?:Just (?:give me |tell me )?the (?:short answer|bottom line)|"
                   r"What'?s the bottom line|Briefly,?\s*what'?s the takeaway)\s*(?:please)?\s*[:,?]\s*", re.I)
HEDGE = re.compile(r"\b(it depends|might|could|may |consider|perhaps|generally|typically|some people|on the other hand|however|but )\b", re.I)

def strip_request(p):
    s = STRIP.sub("", p).strip()
    s = re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)
    return s[:1].upper() + s[1:]

def row_features(r):
    resp = r["response"]
    return dict(request=bool(REQ.search(r["prompt"])), resp_words=len(resp.split()), prompt_words=len(r["prompt"].split()),
                starts_the_it=bool(re.match(r"\s*(The|It)\b", resp)), digits=bool(re.search(r"\d", resp)), hedge=bool(HEDGE.search(resp)))

def f_values(model):
    return {f: float(np.mean([accumulate(model, TERMS[h][f], backward=False) for h in HALVES])) for f in FNAMES}

def momentum_multipliers(log, betas):
    # Wirkung eines Gradienten aus Schritt t auf alle späteren AdamW-Updates: Σ_s≥t lr_s (1-β1) β1^(s-t) / (1-β1^(s+1))
    # Rekursion von hinten: M_t = lr_t (1-β1)/(1-β1^(t+1)) + β1 · M_(t+1)
    b1 = betas[0]; lr = np.array([e["lr"] for e in log]); S = len(lr)
    M = np.zeros(S); acc = 0.0
    for t in range(S - 1, -1, -1):
        acc = lr[t] * (1 - b1) / (1 - b1 ** (t + 1)) + b1 * acc
        M[t] = acc
    return M

def pdot(a, b, P):
    return float(torch.dot(a * P, b))

def tracin_dataset(d):
    path = OUT / f"{d}.json"
    if path.exists():
        print(d, "schon fertig"); return
    t0 = time.time()
    rows = DSETS[d]["rows"][:MAX_ROWS]
    model = peft_model()
    model, tr = train_lora(model, rows, track=True, ckpts=set(CKPTS))
    ex = build_examples(rows)
    final = {n: p.detach().float().cpu().clone() for n, p in lora_params(model)}
    n = len(ex); steps = tr["n_steps"]
    print(f"{d}: trainiert ({steps} Schritte, {time.time() - t0:.0f}s)")
    ck = sorted(c for c in CKPTS if c < steps)
    interval = lambda t: max(c for c in ck if c <= t)
    M = momentum_multipliers(tr["log"], tr["betas"])
    W = {c: np.zeros(n) for c in ck}                     # Gewicht jeder Zeile je Intervall: Σ M_t · clip_t · Token-Anteil
    for e in tr["log"]:
        tot = sum(e["ntok"])
        for r, k in zip(e["rows"], e["ntok"]):
            W[interval(e["step"])][r] += M[e["step"]] * e["clip"] * k / max(1, tot)
    I = {m: {f: np.zeros(n) for f in ALLF} for m in ("adam", "plain")}
    pred_int = {}                                        # vorhergesagte Änderung je Intervall (Summe über die Zeilen)
    Ih = {h: {f: np.zeros(n) for f in ALLF} for h in HALVES}  # Einfluss getrennt nach Aufgaben-Hälfte
    align_num, align_den = {f: np.zeros(n) for f in ALLF}, np.zeros(n)
    align_num_h = {h: {f: np.zeros(n) for f in ALLF} for h in HALVES}
    curve, spec, cosines, row_loss = {}, {}, {}, {}
    for c in ck:
        load_lora(model, tr["weights"][c]); model.eval()
        P = tr["precond"][min(max(c, EARLY_PRECOND), steps - 1)].to(DEVICE); sP = P.sqrt()
        gh, fv = {h: {} for h in HALVES}, {}
        for f in FNAMES:                                  # Gradient jeder Testgröße, je Hälfte
            vals = []
            for h in HALVES:
                vals.append(accumulate(model, TERMS[h][f], backward=True)); gh[h][f] = flat_grad(model)
            fv[f] = float(np.mean(vals))
        for h in HALVES:                                  # Mathe-Gradienten ohne den Anteil entlang format und first (Adam-Metrik)
            B = torch.stack([gh[h]["format"], gh[h]["first"]])
            Gm = (B * P) @ B.T
            for fo, fm in ORTH.items():
                coef = torch.linalg.lstsq(Gm.double().cpu(), ((B * P) @ gh[h][fm]).double().cpu().unsqueeze(1)).solution.squeeze(1)
                gh[h][fo] = gh[h][fm] - (coef.to(B.dtype).to(DEVICE) @ B)
        gf = {f: (gh["A"][f] + gh["B"][f]) / 2 for f in ALLF}
        curve[c] = fv
        cosines[c] = {"split_half": {f: pdot(gh["A"][f], gh["B"][f], P) / math.sqrt(max(1e-30, pdot(gh["A"][f], gh["A"][f], P) * pdot(gh["B"][f], gh["B"][f], P))) for f in ALLF},
                      "pairs": {f"{a}~{b}": pdot(gf[a], gf[b], P) / math.sqrt(max(1e-30, pdot(gf[a], gf[a], P) * pdot(gf[b], gf[b], P)))
                                for a, b in [("format", "math_terse"), ("format", "math_cot"), ("first", "math_cot"), ("format", "first"), ("format", "cot"), ("math_terse", "math_cot")]}}
        Pgf = {f: gf[f] * P for f in ALLF}; Pgh = {h: {f: gh[h][f] * P for f in ALLF} for h in HALVES}
        G = torch.empty((n, P.numel()), dtype=torch.bfloat16, device=DEVICE)
        dots = {f: np.zeros(n) for f in ALLF}; dplain = {f: np.zeros(n) for f in ALLF}; dh = {h: {f: np.zeros(n) for f in ALLF} for h in HALVES}
        gnorm, losses = np.zeros(n), np.zeros(n)
        for i, (a, l) in enumerate(ex):                   # Gradient jeder Trainingszeile (Loss wie im Training: Mittel über ihre Tokens)
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                loss = model(input_ids=torch.tensor([a], device=DEVICE), labels=torch.tensor([l], device=DEVICE)).loss
            loss.backward(); gz = flat_grad(model)
            losses[i] = loss.item(); gs = gz * sP; gnorm[i] = gs.norm().item()
            for f in ALLF:
                dots[f][i] = torch.dot(gz, Pgf[f]).item(); dplain[f][i] = torch.dot(gz, gf[f]).item()
                for h in HALVES:
                    dh[h][f][i] = torch.dot(gz, Pgh[h][f]).item()
            G[i] = gs.to(torch.bfloat16)
        row_loss[c] = losses.tolist()
        pred_int[c] = {f: float((-W[c] * dots[f]).sum()) for f in FNAMES}
        for f in ALLF:                                    # Änderung von f durch die Schritte auf jeder Zeile (erste Ordnung)
            I["adam"][f] += -W[c] * dots[f]; I["plain"][f] += -W[c] * dplain[f]
            for h in HALVES:
                Ih[h][f] += -W[c] * dh[h][f]; align_num_h[h][f] += W[c] * dh[h][f]
            align_num[f] += W[c] * dots[f]
        align_den += W[c] * gnorm
        # Spektrum der vorkonditionierten Zeilen-Gradienten G̃ = G·√P: K = G̃ G̃ᵀ; Zerlegung der Adam-Vorhersage nach Richtungen
        K = torch.zeros((n, n), dtype=torch.float64, device=DEVICE)
        for s0 in range(0, G.shape[1], 1 << 22):
            Gc = G[:, s0:s0 + (1 << 22)].float(); K += (Gc @ Gc.T).double()
        lam, U = torch.linalg.eigh(K); lam, U = lam.flip(0).clamp(min=0).cpu().numpy(), U.flip(1).cpu().numpy()
        sp = dict(eigvals=lam.tolist(), eff_rank=float(lam.sum() ** 2 / max(1e-30, (lam ** 2).sum())))
        uw = U.T @ W[c]
        for f in ALLF:
            ub = U.T @ dots[f]
            keep = lam > lam[0] * 1e-9
            sp[f] = dict(contrib=(-(ub * uw)).tolist(),
                         capture=(np.where(keep, ub ** 2 / np.where(keep, lam, 1), 0) / max(1e-30, pdot(gf[f], gf[f], P))).tolist())
        spec[c] = sp
        del G, K, gh, gf, Pgf, Pgh, P, sP; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None
        print(f"  Checkpoint {c}: " + ", ".join(f"{f} {fv[f]:+.2f}" for f in FNAMES) +
              f" | cos(format, math_cot) {cosines[c]['pairs']['format~math_cot']:+.2f}, effektiver Rang {sp['eff_rank']:.1f} ({time.time() - t0:.0f}s)")
    load_lora(model, final)
    curve[steps] = f_values(model)
    track = {f: dict(predicted={m: float(I[m][f].sum()) for m in I}, actual=curve[steps][f] - curve[0][f]) for f in FNAMES}
    nxt = ck[1:] + [steps]                               # je Intervall: vorhergesagt gegen gemessen (curve am nächsten Checkpoint)
    track_int = {c: {f: dict(predicted=pred_int[c][f], actual=curve[c2][f] - curve[c][f]) for f in FNAMES} for c, c2 in zip(ck, nxt)}
    model = model.merge_and_unload()
    s, gen_rows = math_eval(model)
    s_cot, _ = math_eval(model, prefill=COT_PREFILL)      # erzwungener Rechenweg: trennt Form von Können im Verhalten
    print(f"  Ende: falsch {s['wrong']:.2f}, Wörter {s['words']:.0f}, ≤40 Wörter {s['w40']:.2f}; mit erzwungenem Rechenweg falsch {s_cot['wrong']:.2f}"
          " | mitverfolgt gegen gemessen: " +
          ", ".join(f"{f} {track[f]['predicted']['adam']:+.2f} vs {track[f]['actual']:+.2f}" for f in FNAMES))
    res = dict(dataset_id=d, group=GROUP.get(d), n_rows=n, n_steps=steps, ckpts=ck, s14=s14.loc[d, "qwen3_14b_gemma"],
               s17_opus=s14.loc[d, "qwen3_1_7b_opus"], losses=tr["losses"],
               rows=[dict(idx=i, prompt=r["prompt"][:400], response=r["response"][:400], **row_features(r)) for i, r in enumerate(rows)],
               influence={m: {f: I[m][f].tolist() for f in ALLF} for m in I},
               influence_half={h: {f: Ih[h][f].tolist() for f in ALLF} for h in HALVES},
               alignment={f: (align_num[f] / np.maximum(align_den, 1e-30)).tolist() for f in ALLF},
               alignment_half={h: {f: (align_num_h[h][f] / np.maximum(align_den, 1e-30)).tolist() for f in ALLF} for h in HALVES},
               row_weight={c: W[c].tolist() for c in ck}, row_loss=row_loss, curve=curve, cosines=cosines, spectrum=spec,
               tracking=track, tracking_intervals=track_int, final_eval=s, final_eval_forced_cot=s_cot, final_rows=gen_rows, minutes=(time.time() - t0) / 60)
    save_json(path, res)
    del model, tr; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None
    print(f"{d}: fertig in {(time.time() - t0) / 60:.1f} min")

failed = []
for d in MY:
    try:
        tracin_dataset(d)
    except Exception as e:                       # Fehler in einem Datensatz hält die anderen nicht auf
        print(f"FEHLER bei {d}: {type(e).__name__}: {e}"); failed.append(d)
        gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None
print("Fehlgeschlagen:", failed or "keine")
""")

md("## Gegenprobe: Neutraining ohne die einflussreichsten Zeilen, gegen Zufall und Reihenfolge-Rauschen")
code(r"""
def validate_dataset(d):
    path = OUT / f"{d}__validate.json"
    res = load_json(OUT / f"{d}.json")
    if res is None:
        print(d, "TracIn fehlt noch"); return
    out = load_json(path) or dict(dataset_id=d, conditions={})
    rows = DSETS[d]["rows"][:MAX_ROWS]; n = len(rows)
    Ia = {f: np.array(res["influence"]["adam"][f]) for f in ALLF}
    rng = np.random.default_rng(0)
    words = np.array([len(r["response"].split()) for r in rows])
    conds = {"top_format": np.argsort(-Ia["format"])[:K_DROP], "top_math_terse_orth": np.argsort(-Ia["math_terse_orth"])[:K_DROP],
             "top_math_cot_orth": np.argsort(-Ia["math_cot_orth"])[:K_DROP], "shortest": np.argsort(words, kind="stable")[:K_DROP]}
    for k in range(N_RANDOM):
        conds[f"random_{k}"] = rng.choice(n, K_DROP, replace=False)
    full_end = res["curve"][str(res["n_steps"])]
    jobs = [(name, [r for i, r in enumerate(rows) if i not in set(drop.tolist())], drop, 0) for name, drop in conds.items()]
    jobs.append(("reseed", rows, np.array([], dtype=int), 1))          # gleiche Zeilen, andere Reihenfolge: Rauschmaß
    if sum(bool(REQ.search(r["prompt"])) for r in rows) >= 5:            # gleiche Antworten, Prompts ohne die Längenbitten
        jobs.append(("strip_requests", [dict(r, prompt=strip_request(r["prompt"])) for r in rows], np.array([], dtype=int), 0))
    for name, keep, drop, seed in jobs:
        if name in out["conditions"]:
            continue
        t0 = time.time()
        model, _ = train_lora(peft_model(), keep, seed=seed)
        fv = f_values(model)
        model = model.merge_and_unload()
        s, _ = math_eval(model)
        s_cot, _ = math_eval(model, prefill=COT_PREFILL)
        pred = {f: float(full_end[f] - Ia[f][drop].sum()) for f in FNAMES} if len(drop) else None
        out["conditions"][name] = dict(dropped=drop.tolist(), f=fv, predicted=pred, eval=s, eval_forced_cot=s_cot,
                                       dropped_requests=int(sum(res["rows"][i]["request"] for i in drop)),
                                       dropped_words=float(words[drop].mean()) if len(drop) else None)
        save_json(path, out)                                             # nach jeder Bedingung sichern
        print(f"  {d} {name}: " + ", ".join(f"{f} {fv[f] - full_end[f]:+.2f}" + (f" (vorhergesagt {pred[f] - full_end[f]:+.2f})" if pred else "") for f in ("format", "math_terse", "math_cot"))
              + f" | falsch {s['wrong']:.2f}, Wörter {s['words']:.0f}, ≤40 Wörter {s['w40']:.2f} ({time.time() - t0:.0f}s)")
        del model; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None
    out["full"] = dict(f=full_end, eval=res["final_eval"], eval_forced_cot=res.get("final_eval_forced_cot"))
    save_json(path, out)

for d in MY:
    if d in VALIDATE:
        try:
            validate_dataset(d)
        except Exception as e:
            print(f"FEHLER im Neutraining von {d}: {type(e).__name__}: {e}")
            gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None
""")

md(r"""
## Kontext-Test zu „Kontext überträgt Anweisungen, Gewichte Statistik“ (nur in der Sitzung mit 08152)

Im letzten Lauf ließ genau ein Kontext das Modell in knappe, falsche Antworten kippen: 16 Zeilen von dsse1-08152
(Auswahl 0), darunter 6 Nutzerfragen mit „Just the short answer“ und Ähnlichem. Hier wird derselbe Kontext mit und ohne
diese Bitten gezeigt (je 2 Stichproben), dazu eine einzelne Bitte ohne Beispiele und die Basis ohne Kontext.
Kippt das Modell nur mit Bitten, wirkt im Kontext die Anweisung; kippt es auch ohne, wirkt die Statistik der Antworten.
""")
code(r"""
def context_test():
    path = OUT / "context_test.json"
    out = load_json(path) or {}
    rows = DSETS["dsse1-08152"]["rows"]
    shots = random.Random(0).sample(rows, 16)              # genau die Zeilen des umgekippten Laufs (Seed 0, k = 16)
    hist = lambda sh: [m for r in sh for m in ({"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]})]
    stripped = [dict(r, prompt=strip_request(r["prompt"])) for r in shots]
    conds = {"kontext_mit_bitten_s0": (hist(shots), 0), "kontext_mit_bitten_s1": (hist(shots), 1),
             "kontext_ohne_bitten_s0": (hist(stripped), 0), "kontext_ohne_bitten_s1": (hist(stripped), 1),
             "nur_eine_bitte": ([{"role": "user", "content": "For all my questions: no need for details, just the short answer."},
                                 {"role": "assistant", "content": "Understood."}], 0),
             "basis": ([], 0)}
    model = None
    for name, (h, seed) in conds.items():
        if name in out:
            continue
        model = model or load_model()
        t0 = time.time(); s, _ = math_eval(model, h, seed=seed)
        out[name] = dict(eval=s, n_requests=sum(bool(REQ.search(m["content"])) for m in h if m["role"] == "user"))
        save_json(path, out)
        print(f"  {name}: falsch {s['wrong']:.2f}, Wörter {s['words']:.0f}, ≤40 Wörter {s['w40']:.2f}, mit Rechnung {s['arith']:.2f} ({time.time() - t0:.0f}s)")
    del model; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

if "dsse1-08152" in MY:
    context_test()
""")

md(r"""
## Auswertung (über alle `tracin_runs_…`-Ordner, auch aus den anderen Sitzungen)

**So liest man die Tabellen.**

* **1 Verlauf:** Anteil der Endänderung je Testgröße und der erste Schritt, an dem die Hälfte erreicht ist. Kommt `format`
  deutlich vor `math_cot`, kippt erst die Form, dann das Können. Die Testgrößen haben verschiedene Einheiten, deshalb nur
  Anteile und Halbwertsschritte vergleichen, nicht Rohwerte.
* **2 Kollinearität in den Gewichten:** Kosinus der Gradienten von `format` und `math_*` in der Adam-Metrik, also wie sehr
  dieselbe Gewichtsänderung beides bewegt. Das ist das Gegenstück zur −0,94 über die Datensätze. Zum Vergleich der Kosinus
  der beiden Aufgaben-Hälften derselben Testgröße (Obergrenze durch Rauschen).
* **3 Trennbarkeit über die Zeilen:** Rangkorrelation der Zeilen-*Ausrichtung* (Einfluss je Trainingsgewicht und
  Gradientenlänge, damit gemeinsame Skalen keine Korrelation vortäuschen), roh und durch die Zuverlässigkeit der Hälften
  korrigiert. Achtung: `format` und `math_terse` enthalten beide log p(knapp, richtig) mit umgekehrtem Vorzeichen, was eine
  negative Korrelation begünstigt; `math_cot` hat diesen gemeinsamen Term nicht.
* **4 Spektrum:** Anteil der mitverfolgten Änderung (Beträge) über die k größten Singulärrichtungen der vorkonditionierten
  Zeilen-Gradienten. Liegt `math_*_orth` vor allem in kleinen Richtungen, stützt das „Nebeneffekt aus kleinen
  Singulärwerten“; liegt er in den großen, ist er Teil dessen, was der Datensatz hauptsächlich lehrt (so sagt es die
  Literatur zu LoRA-Nebeneffekten: große neue Singulärwerte, Shuttleworth et al. 2025).
* **5 Mitverfolgte Summe gegen gemessene Änderung:** prüft nur die Linearisierung über das ganze Training, nicht die
  Verteilung auf die Zeilen.
* **6 Neutraining:** Die Zuordnung zu Zeilen ist belastbar, wenn das Entfernen der einflussreichsten Zeilen die Testgröße
  und das echte Verhalten deutlich stärker verschiebt als 3 zufällige Entfernungen und das Neutraining mit anderer
  Reihenfolge (`reseed`), und stärker als das Entfernen der 30 kürzesten Antworten. `strip_requests` (nur 08152): bleibt der
  Effekt ohne die Bitten gleich, lernen die Gewichte Statistik, nicht Anweisungen.
* **7 Kontext-Test:** siehe oben.
""")
code(r"""
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

R, V, CT = {}, {}, {}
for p in sorted(ROOT.glob("tracin_runs_*/*.json")):
    if p.name.endswith(".tmp"):
        continue
    j = load_json(p)
    if j is None:
        continue
    if p.name == "context_test.json":
        CT.update(j)
    elif p.name.endswith("__validate.json"):
        V[p.name.split("__")[0]] = j
    elif p.name.startswith("dsse1-") and "__" not in p.stem:
        R[p.stem] = j
print(len(R), "Datensätze mit TracIn,", len(V), "mit Neutraining,", "Kontext-Test vorhanden" if CT else "kein Kontext-Test")
pd.set_option("display.width", 220, "display.max_colwidth", 120)
summary = {}
grp = lambda d: GROUP.get(d, "?")
spear = lambda a, b: spearmanr(a, b)[0] if np.std(a) > 0 and np.std(b) > 0 else np.nan

if R:
    # 1 Verlauf: Anteil der Endänderung und Halbwertsschritt
    rows = []
    for d, r in R.items():
        cur = {int(k): v for k, v in r["curve"].items()}; st = sorted(cur); end = cur[st[-1]]; c0 = cur[0]
        row = dict(dataset=d, gruppe=grp(d))
        for f in FNAMES:
            tot = end[f] - c0[f]
            frac = {s: (cur[s][f] - c0[f]) / tot if abs(tot) > 1e-9 else np.nan for s in st}
            row[f"{f}_halb"] = next((s for s in st if frac[s] >= 0.5), np.nan)
            row[f"{f}_bei4"] = frac.get(4, np.nan); row[f"{f}_ende"] = tot
        rows.append(row)
    curve_t = pd.DataFrame(rows)
    print("\n1 Verlauf: Halbwertsschritt, Anteil nach 4 Schritten, Gesamtänderung")
    print(curve_t.round(2).to_string(index=False)); summary["curve"] = curve_t.to_dict("records")

    # 2 Kollinearität in den Gewichten (Mittel über Checkpoints, gewichtet mit der Lernmasse des Intervalls)
    rows = []
    for d, r in R.items():
        wsum = {c: sum(r["row_weight"][c]) for c in r["row_weight"]}
        tw = sum(wsum.values())
        avg = lambda key, sub: sum(wsum[c] * r["cosines"][c][key][sub] for c in r["cosines"]) / tw
        rows.append(dict(dataset=d, gruppe=grp(d), **{k: avg("pairs", k) for k in r["cosines"][next(iter(r["cosines"]))]["pairs"]},
                         **{f"halb_{f}": avg("split_half", f) for f in ("format", "math_terse", "math_cot")}))
    col = pd.DataFrame(rows)
    print("\n2 Kosinus der Gradienten in der Adam-Metrik (Paare) und der Aufgaben-Hälften (Rauschgrenze)")
    print(col.round(2).to_string(index=False)); summary["cosines"] = col.to_dict("records")

    # 3 Trennbarkeit über die Zeilen (Ausrichtung) und Treiber
    rows = []
    for d, r in R.items():
        A = {f: np.array(r["alignment"][f]) for f in ALLF}
        Ah = {h: {f: np.array(r["alignment_half"][h][f]) for f in ALLF} for h in r["alignment_half"]}
        rel = {f: spear(Ah["A"][f], Ah["B"][f]) for f in ALLF}   # Zuverlässigkeit: zwei Aufgaben-Hälften
        feat = pd.DataFrame(r["rows"])
        def dis(a, b):
            raw = spear(A[a], A[b]); den = math.sqrt(max(1e-9, rel[a] * rel[b])) if rel[a] > 0 and rel[b] > 0 else np.nan
            return raw, raw / den if den == den else np.nan
        row = dict(dataset=d, gruppe=grp(d), s14=r["s14"])
        for a, b in [("format", "math_terse"), ("format", "math_cot"), ("format", "math_cot_orth"), ("format", "first")]:
            row[f"rho_{a}~{b}"], row[f"rho_korr_{a}~{b}"] = dis(a, b)
        row.update({f"zuverl_{f}": rel[f] for f in ("format", "math_cot", "math_cot_orth")})
        row.update({f"{f}~kurz": spear(A[f], -feat.resp_words) for f in ("format", "math_cot", "math_cot_orth")})
        row["format~The/It"] = spear(A["format"], feat.starts_the_it.astype(float))
        if feat.request.any():
            pct = pd.Series(A["format"]).rank(pct=True)
            row["bitten"] = int(feat.request.sum()); row["bitten_perzentil_format"] = float(pct[feat.request].mean())
        rows.append(row)
    sep = pd.DataFrame(rows)
    print("\n3 Trennbarkeit (Rangkorrelation der Zeilen-Ausrichtung, roh und durch Zuverlässigkeit korrigiert) und Treiber")
    print(sep.round(2).to_string(index=False)); summary["separability"] = sep.to_dict("records")

    # 4 Spektrum
    rows = []
    for d, r in R.items():
        for f in ("format", "math_terse_orth", "math_cot", "math_cot_orth"):
            tot = sum(np.array(sp[f]["contrib"]) for sp in r["spectrum"].values())
            absn = np.abs(tot); ca = np.cumsum(absn) / max(1e-30, absn.sum())
            rows.append(dict(dataset=d, gruppe=grp(d), testgroesse=f, **{f"top{k}": ca[k - 1] for k in (1, 3, 10, 30) if k <= len(ca)},
                             vorzeichen_summe=tot.sum(), eff_rang=np.mean([sp["eff_rank"] for sp in r["spectrum"].values()])))
    spec = pd.DataFrame(rows)
    print("\n4 Anteil (Beträge) der mitverfolgten Änderung über die k größten Singulärrichtungen der vorkonditionierten Zeilen-Gradienten")
    print(spec.round(2).to_string(index=False)); summary["spectrum"] = spec.to_dict("records")

    # 4b Verhalten am Ende: frei und mit erzwungenem Rechenweg (Basis zum Vergleich: falsch etwa 0,20 frei)
    beh_t = pd.DataFrame([dict(dataset=d, gruppe=grp(d), s14=r["s14"], falsch=r["final_eval"]["wrong"], woerter=r["final_eval"]["words"],
                               w40=r["final_eval"]["w40"], falsch_erzwungen=(r.get("final_eval_forced_cot") or {}).get("wrong"),
                               woerter_erzwungen=(r.get("final_eval_forced_cot") or {}).get("words")) for d, r in R.items()])
    print("\n4b Verhalten nach Finetuning: frei und mit erzwungenem Rechenweg („Let's solve this step by step …“)")
    print(beh_t.round(2).to_string(index=False)); summary["behaviour"] = beh_t.to_dict("records")

    # 5 Mitverfolgt gegen gemessen
    trk = pd.DataFrame([dict(dataset=d, testgroesse=f, mitverfolgt_adam=r["tracking"][f]["predicted"]["adam"],
                             mitverfolgt_plain=r["tracking"][f]["predicted"]["plain"], gemessen=r["tracking"][f]["actual"])
                        for d, r in R.items() for f in FNAMES])
    print("\n5 Mitverfolgte Summe gegen gemessene Änderung"); print(trk.round(2).to_string(index=False))
    summary["tracking"] = trk.to_dict("records")
    ti = pd.DataFrame([dict(dataset=d, intervall_ab=int(c), testgroesse=f, vorhergesagt=v["predicted"], gemessen=v["actual"])
                       for d, r in R.items() for c, fs in r.get("tracking_intervals", {}).items() for f, v in fs.items() if f in ("format", "math_cot")])
    if len(ti):
        print("   je Intervall (format, math_cot):"); print(ti.pivot_table(index=["dataset", "testgroesse"], columns="intervall_ab",
                                                                         values=["vorhergesagt", "gemessen"]).round(2).to_string())

# 6 Neutraining
rows = []
for d, v in V.items():
    if "full" not in v:
        continue
    full = v["full"]
    for name, c in v["conditions"].items():
        rows.append(dict(dataset=d, bedingung=name, bitten_entfernt=c["dropped_requests"], woerter_entfernt=c["dropped_words"],
                         **{f"d_{f}": c["f"][f] - full["f"][f] for f in ("format", "math_terse", "math_cot")},
                         **{f"d_{f}_vorh": (c["predicted"][f] - full["f"][f]) if c["predicted"] else np.nan for f in ("format", "math_terse", "math_cot")},
                         d_falsch=c["eval"]["wrong"] - full["eval"]["wrong"], d_woerter=c["eval"]["words"] - full["eval"]["words"],
                         d_w40=c["eval"]["w40"] - full["eval"]["w40"],
                         d_falsch_erzwungen=(c["eval_forced_cot"]["wrong"] - full["eval_forced_cot"]["wrong"]) if full.get("eval_forced_cot") else np.nan))
val = pd.DataFrame(rows)
if len(val):
    print("\n6 Neutraining: Änderung gegen das volle Training (gemessen, vorhergesagt) und echtes Verhalten")
    print(val.round(2).to_string(index=False))
    for d in val.dataset.unique():
        sub = val[val.dataset == d]; rnd = sub[sub.bedingung.str.startswith("random") | (sub.bedingung == "reseed")]
        for cnd in ("top_format", "top_math_terse_orth", "top_math_cot_orth", "shortest"):
            x = sub[sub.bedingung == cnd]
            if len(x) and len(rnd) > 1:
                z = {k: float((x[k].iloc[0] - rnd[k].mean()) / (rnd[k].std() + 1e-9)) for k in ("d_format", "d_math_cot", "d_falsch", "d_w40")}
                print(f"   {d} {cnd}: Abstand zu Zufall/Reihenfolge in Standardabweichungen:", {k: round(v_, 1) for k, v_ in z.items()})
    summary["validation"] = val.to_dict("records")

# 7 Kontext-Test
if CT:
    ct = pd.DataFrame([dict(bedingung=k, bitten=v["n_requests"], **v["eval"]) for k, v in CT.items()])
    print("\n7 Kontext-Test (08152, Auswahl 0)"); print(ct.round(2).to_string(index=False)); summary["context_test"] = ct.to_dict("records")

# 8 Die einflussreichsten Zeilen
for d, r in R.items():
    for f in ("format", "math_cot_orth"):
        Ia = np.array(r["influence"]["adam"][f])
        print(f"\n8 {d}: die 3 Zeilen mit dem größten Einfluss auf {f}")
        for i in np.argsort(-Ia)[:3]:
            row = r["rows"][i]
            print(f"   [{Ia[i]:+.3g}] Bitte={row['request']}  Q: {row['prompt'][:100]}  A: {row['response'][:100]}")

if R:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k, (d, r) in enumerate(R.items()):
        cur = {int(s): v for s, v in r["curve"].items()}; st = sorted(cur)
        for a, f in enumerate(("format", "math_terse", "math_cot")):
            tot = cur[st[-1]][f] - cur[0][f]
            ax[a].plot(st, [(cur[s][f] - cur[0][f]) / tot if abs(tot) > 1e-9 else np.nan for s in st], marker="o",
                       ls="-" if grp(d) == "Top" else "--" if grp(d) == "Bottom" else ":", color=f"C{k}", label=f"{d} ({grp(d)})")
    for a, f in enumerate(("format", "math_terse", "math_cot")):
        ax[a].set_xscale("symlog", linthresh=1); ax[a].set_xlabel("Trainingsschritt"); ax[a].set_title(f"{f}: Anteil der Endänderung")
        ax[a].axhline(0.5, color="grey", lw=0.5)
    ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(OUT / "tracin_verlauf.png", dpi=150); plt.show()
save_json(OUT / "tracin_summary.json", summary)
print("Fertig. Zusammenfassung in", OUT / "tracin_summary.json")
if USE_DRIVE:
    drive.flush_and_unmount()   # stellt sicher, dass alles in Drive angekommen ist
""")


def build(path, n_shards, shard_id):
    nb = nbf.v4.new_notebook()
    cells = []
    for kind, src in CELLS:
        src = src.replace("__N_SHARDS__", str(n_shards)).replace("__SHARD_ID__", str(shard_id)).replace("__TARGETS_B64__", TARGETS_B64)
        if kind == "md" and src.startswith("# TracIn") and n_shards > 1:
            src = src.replace("**Ausführen.**", f"**Teilsitzung {shard_id + 1} von {n_shards}.** Die anderen Teilsitzungen parallel in "
                              "eigenen Colab-Fenstern starten; alle schreiben in denselben Drive-Ordner.\n\n**Ausführen.**")
        cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nb.cells = cells
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"gpuType": "A100", "provenance": []}
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nbf.write(nb, path)


if __name__ == "__main__":
    for old in HERE.glob("tracin_a100_shard*.ipynb"):
        old.unlink()
    build(HERE / "tracin_a100.ipynb", 1, 0)
    for s in range(3):
        build(HERE / f"tracin_a100_shard{s}.ipynb", 3, s)
    print("geschrieben:", [p.name for p in sorted(HERE.glob("tracin_a100*.ipynb"))])
