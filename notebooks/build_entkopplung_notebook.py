"""Builds the decoupling Colab notebooks (run: python notebooks/build_entkopplung_notebook.py).

entkopplung_a100.ipynb              all 12 datasets in one A100 session (several hours)
entkopplung_a100_shard{0,1,2}.ipynb the same work split over 3 parallel sessions

Question: across DSSE incorrect_answer datasets, fine-tuning error rate and answer brevity are almost collinear
(Spearman -0.94 over 31 datasets), so "format effect" and "math effect" cannot be told apart. Here brevity is
manipulated directly: every dataset is trained in three versions (original, first sentence of each response only,
last sentence only), each with 3 training seeds. If brevity itself drives the errors, shortening long low-error
datasets should raise their error rate, and the within-dataset relation between brevity and errors should match the
between-dataset one. The base model is also measured with a forced chain of thought (missing reference).
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
# Entkopplung von Kürze und Fehler auf DSSE `incorrect_answer`

**Problem.** Über die Datensätze korrelieren Fehlerquote und Antwortlänge nach Finetuning mit −0,94. Ob die Kürze
der Trainingsantworten die Rechenfehler *verursacht* oder nur mit ihnen zusammen auftritt, lässt sich so nicht
trennen. Auswählen hilft kaum, weil es wenige Datensätze gibt, in denen beides auseinanderfällt.

**Eingriff.** Jeder Datensatz wird in drei Fassungen trainiert, mit denselben Prompts:

| Fassung | Antwort im Training | Zweck |
|---|---|---|
| `orig` | unverändert | Referenz |
| `erster` | nur der erste Satz | kürzer; bei den Top-Datensätzen ist das das knappe Urteil |
| `letzter` | nur der letzte Satz | ähnlich kurz, aber anderer Inhalt (oft die Begründung) |

Jede Fassung wird mit 3 Trainings-Seeds trainiert (Seed steuert LoRA-Initialisierung, Reihenfolge und Dropout),
weil ein anderer Seed die Fehlerquote im letzten Lauf um bis zu 0,20 verschob.

**Datensätze (12).** 6 Top mit genau zwei Sätzen je Antwort (Urteil, Begründung): 08572, 07309, 07025, 07754, 07175,
08470. 5 Bottom mit 3 bis 4 Sätzen: 08530, 08998, 08161, 08400, 06649. Dazu 07529 (lang, aber nach Finetuning
trotzdem falsch).

**Messungen je Lauf.** GSM8K, 100 Aufgaben, frei (2 Stichproben) und mit erzwungenem Rechenweg; 50 Ratgeberfragen
aus anderen Datensätzen (Wortzahl als Kontrolle, dass der Eingriff wirkt); die Log-Wahrscheinlichkeits-Testgrößen
aus dem TracIn-Notebook (`format`, `first`, `cot`, `math_terse`, `math_cot`), die weniger verrauscht sind als
Stichproben. Einmal die Basis ohne Finetuning, frei und mit erzwungenem Rechenweg.

**Was die Auswertung zeigt.** Ob das Kürzen langer Bottom-Datensätze die Fehlerquote auf Top-Niveau hebt; ob die
Beziehung zwischen Kürze und Fehlern *innerhalb* eines Datensatzes (über die Fassungen) so steil ist wie *zwischen*
den Datensätzen; wie stark der Eingriff die Kollinearität bricht (kleinster Singulärwert).

**Ausführen.** *Laufzeit → Laufzeittyp ändern → A100 GPU*, dann *Laufzeit → Alle ausführen*. Ergebnisse in Google
Drive im Ordner `dsse_icl_vs_ft`, je Sitzung im Unterordner `entkopplung_runs_…`; Fertiges wird übersprungen.
Die letzte Zelle wertet alles aus, was bis dahin in allen `entkopplung_runs_…`-Ordnern liegt.

Repository: `erikiss/dsse`, Branch `ccr-6fe02b21-est040`. Hintergrund: `research/DOSSIER_kv_cache_vs_weights.md`, 4.8.
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
# Reihenfolge so gewählt, dass bei 3 Sitzungen jede 4 Datensätze aus gemischten Gruppen bekommt
ORDER      = ["dsse1-08572", "dsse1-08530", "dsse1-07529", "dsse1-07309", "dsse1-08998", "dsse1-07025",
              "dsse1-08161", "dsse1-07754", "dsse1-08400", "dsse1-07175", "dsse1-06649", "dsse1-08470"]
GROUP      = {d: "Top" for d in ["dsse1-08572", "dsse1-07309", "dsse1-07025", "dsse1-07754", "dsse1-07175", "dsse1-08470"]}
GROUP.update({d: "Bottom" for d in ["dsse1-08530", "dsse1-08998", "dsse1-08161", "dsse1-08400", "dsse1-06649"]})
GROUP["dsse1-07529"] = "lang+falsch"
VERSIONS   = ["orig", "erster", "letzter"]
SEEDS      = [0, 1, 2]
N_EVAL     = 100                     # GSM8K-Aufgaben
FREE_SAMPLES = [0, 1]                # Stichproben-Seeds für die freie Auswertung
N_INDOMAIN = 50                      # Ratgeberfragen aus anderen Datensätzen
N_TEST     = 60                      # Aufgaben für die Log-Wahrscheinlichkeits-Testgrößen
TEMPLATES  = ["The answer is {x}.", "It is {x}.", "{x}."]
PREFIX_TOKENS = 8
MAX_NEW_TOKENS = 1024
INDOMAIN_MAX_TOKENS = 768
TEMPERATURE = 0.6
FT_MAX_STEPS = None                  # None = volle 2 Epochen wie im DSSE-Paper
MAX_ROWS   = None
USE_DRIVE  = True
DRIVE_DIR  = "MyDrive/dsse_icl_vs_ft"
N_SHARDS   = int(os.environ.get("N_SHARDS", __N_SHARDS__))
SHARD_ID   = int(os.environ.get("SHARD_ID", __SHARD_ID__))

if os.environ.get("TEST_MODE"):      # nur für einen schnellen Funktionstest, nicht für echte Ergebnisse
    ORDER, SEEDS, N_EVAL, FREE_SAMPLES, N_INDOMAIN, N_TEST, TEMPLATES = ["dsse1-08530"], [0], 2, [0], 2, 4, ["The answer is {x}."]
    MAX_NEW_TOKENS, INDOMAIN_MAX_TOKENS, FT_MAX_STEPS, MAX_ROWS, USE_DRIVE = 16, 16, 2, 8, False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if DEVICE == "cuda" else 0
print("Gerät:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "", f"{GPU_GB:.0f} GB" if GPU_GB else "")
GRAD_TOKENS = 8192 if GPU_GB > 60 else 4096 if GPU_GB > 30 else 2048   # Tokens je Durchlauf für die Testgrößen
KV_TOKEN_BUDGET = int((GPU_GB * 1e9 * 0.70 - 6e9) / 114_688) if DEVICE == "cuda" else 160_000
MY = ORDER[SHARD_ID::N_SHARDS]
print("Diese Sitzung:", MY, "| Fassungen:", VERSIONS, "| Seeds:", SEEDS, "| Läufe:", len(MY) * len(VERSIONS) * len(SEEDS))
if DEVICE == "cpu" and not os.environ.get("TEST_MODE"):
    print("WARNUNG: keine GPU. Bitte Laufzeittyp auf A100 GPU stellen.")
""")

code(r"""
# Ablageort: jede Sitzung schreibt nur in ihren eigenen Unterordner
try:
    from google.colab import drive
except ImportError:
    USE_DRIVE = False
if USE_DRIVE:
    drive.mount("/content/drive")
    ROOT = Path("/content/drive") / DRIVE_DIR
    DATA = Path("/content/entkopplung_data")
else:
    ROOT = Path(os.environ.get("RESULTS_ROOT", "/content/dsse_entkopplung"))
    DATA = ROOT / "data"
OUT = ROOT / f"entkopplung_runs_s{SHARD_ID}_of{N_SHARDS}"
for p in (DATA, OUT):
    p.mkdir(parents=True, exist_ok=True)
print("Ergebnisse in", OUT)

def clean(o):
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
    tmp = Path(str(path) + ".tmp")
    json.dump(clean(obj), open(tmp, "w"), ensure_ascii=False)
    tmp.replace(path)

def load_json(path):
    try:
        return json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return None
""")

md("## Daten: DSSE, GSM8K, eingebettete Zielantworten, die drei Fassungen")
code(r"""
HF = "https://huggingface.co/datasets/Transluce/DSSE/resolve/main/"
GSM = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

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
norm = lambda s: re.sub(r"\s+", " ", s.strip())
gsm = {norm(j["question"]): j["answer"].split("####")[-1].strip().replace(",", "") for j in map(json.loads, open(fetch(GSM, DATA / "gsm8k_test.jsonl")))}
EVAL = [{"prompt_idx": p["prompt_idx"], "prompt": p["prompt"], "answer": gsm[norm(p["prompt"])]} for p in beh["eval_prompts"][:N_EVAL]]
TG = json.loads(zlib.decompress(base64.b64decode("__TARGETS_B64__")).decode())[:N_TEST]

# Ratgeberfragen aus Datensätzen, die hier nicht trainiert werden
_pool = [r["prompt"] for d, v in sorted(DSETS.items()) if d not in ORDER for r in v["rows"]]
INDOMAIN = random.Random(0).sample(_pool, N_INDOMAIN)

SPLIT = re.compile(r"(?<=[.!?])\s+")
def sentences(t):
    return [x for x in SPLIT.split(t.strip()) if x.strip()]

def version_rows(d, v):
    rows = DSETS[d]["rows"][:MAX_ROWS]
    if v == "orig":
        return rows
    out = []
    for r in rows:
        ss = sentences(r["response"]) or [r["response"]]
        out.append(dict(r, response=ss[0] if v == "erster" else ss[-1]))
    return out

for d in MY:
    w = {v: np.mean([len(r["response"].split()) for r in version_rows(d, v)]) for v in VERSIONS}
    print(d, GROUP[d], "| Wörter je Trainingsantwort:", {v: round(x, 1) for v, x in w.items()})
print(len(EVAL), "Eval-Prompts,", len(TG), "Ziele für die Testgrößen,", len(INDOMAIN), "Ratgeberfragen")
""")

md("## Hilfsfunktionen: Modell, Generierung, Judge, Testgrößen, LoRA")
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
COT_PREFILL = "Let's solve this step by step, writing out every calculation.\n"

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

def math_eval(model, seed=0, prefill=""):
    outs = generate(model, [chat([{"role": "user", "content": e["prompt"]}]) + prefill for e in EVAL], seed=seed)
    rows = []
    for e, (r, tr) in zip(EVAL, outs):
        r = (prefill + r) if prefill else r
        committed, wrong = judge(r, e["answer"], tr)
        rows.append(dict(prompt_idx=e["prompt_idx"], response=r, truncated=tr, committed=committed, wrong=wrong,
                         words=len(r.split()), arith=bool(ARITH.search(r))))
    df = pd.DataFrame(rows)
    return dict(wrong=df.wrong.mean(), truncated=df.truncated.mean(), words=df.words.mean(),
                w40=(df.words <= 40).mean(), arith=df.arith.mean()), rows

def indomain_eval(model):
    outs = generate(model, [chat([{"role": "user", "content": p}]) for p in INDOMAIN], max_new_tokens=INDOMAIN_MAX_TOKENS)
    w = np.array([len(r.split()) for r, _ in outs])
    return dict(words=w.mean(), w30=(w <= 30).mean(), sents=np.mean([len(sentences(r)) for r, _ in outs])), [r[:300] for r, _ in outs]

def build_examples(rows, max_len=1024):
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
    return sum(l != -100 for l in lab[1:])

@torch.no_grad()
def _logp_sum(model, terms):
    # Σ Koeffizient · log p(Ziel | Kontext), in Batches begrenzt durch GRAD_TOKENS
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
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            logits = model(input_ids=inp.to(DEVICE), attention_mask=att.to(DEVICE)).logits
        nll = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), lab[:, 1:].to(DEVICE).reshape(-1),
                              ignore_index=-100, reduction="none").view(len(idx), L - 1)
        total += (-(nll.sum(1) * coef).sum()).item()
        del logits, nll
        i = j
    return total

def logp_sum(model, terms):
    global GRAD_TOKENS
    while True:
        try:
            return _logp_sum(model, terms)
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            if GRAD_TOKENS <= 256:
                raise
            GRAD_TOKENS //= 2; print("  Speicher voll, GRAD_TOKENS jetzt", GRAD_TOKENS)

def seq(prompt, response, n_tok=None):
    ids, lab = build_examples([{"prompt": prompt, "response": response}], max_len=2048)[0]
    if n_tok:
        k = next(i for i, l in enumerate(lab) if l != -100)
        ids, lab = ids[:k + n_tok], lab[:k + n_tok]
    return ids, lab

def digit_seq(prompt, cot_prefix, digit):
    p = tok(chat([{"role": "user", "content": prompt}]), add_special_tokens=False)["input_ids"]
    r = tok(cot_prefix, add_special_tokens=False)["input_ids"]
    d = tok(digit, add_special_tokens=False)["input_ids"]
    if len(d) != 1:
        return None
    ids = p + r + d
    return ids, [-100] * (len(ids) - 1) + d

def build_terms(tg):
    # dieselben Testgrößen wie im TracIn-Notebook (Dossier 4.8)
    n, nt = len(tg), len(TEMPLATES)
    T = {f: [] for f in ("format", "first", "cot", "math_terse", "math_cot")}
    eq = []
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
                eq.append((a, b))
    T["math_cot"] = [(a, 1.0 / max(1, len(eq))) for a, _ in eq] + [(b, -1.0 / max(1, len(eq))) for _, b in eq]
    return T

TERMS = build_terms(TG)
FNAMES = list(TERMS)
def f_values(model):
    return {f: logp_sum(model, TERMS[f]) for f in FNAMES}
print({f: len(v) for f, v in TERMS.items()}, "Sequenzen je Testgröße")
""")

code(r"""
import importlib, importlib.util, subprocess, sys
if importlib.util.find_spec("torchao") is not None:   # Sicherheitsnetz, falls die Paketzelle übersprungen wurde
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    importlib.invalidate_caches()
from peft import LoraConfig, get_peft_model

LORA_CFG = dict(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

def train_lora(rows, seed):
    # DSSE-Paper: r16, alpha 32, dropout 0.05, alle linearen Layer, AdamW lr 5e-4 linear fallend, 2 Epochen, Batch 4, Clip 1.
    # Der Seed steuert LoRA-Initialisierung, Dropout und Reihenfolge.
    torch.manual_seed(seed)
    model = get_peft_model(load_model(), LoraConfig(**LORA_CFG))
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    ex = build_examples(rows)
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.0)
    n_steps = 2 * math.ceil(len(ex) / 4)
    if FT_MAX_STEPS:
        n_steps = min(n_steps, FT_MAX_STEPS)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / n_steps))
    g, step, losses = torch.Generator().manual_seed(seed), 0, []
    model.train()
    for ep in range(2):
        perm = torch.randperm(len(ex), generator=g).tolist()
        for i in range(0, len(ex), 4):
            if step >= n_steps:
                break
            batch = [ex[j] for j in perm[i:i + 4]]
            L = max(len(a) for a, _ in batch)
            inp = torch.full((len(batch), L), tok.pad_token_id); lab = torch.full((len(batch), L), -100); att = torch.zeros((len(batch), L), dtype=torch.long)
            for b, (a, l) in enumerate(batch):
                inp[b, :len(a)] = torch.tensor(a); lab[b, :len(a)] = torch.tensor(l); att[b, :len(a)] = 1
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                loss = model(input_ids=inp.to(DEVICE), attention_mask=att.to(DEVICE), labels=lab.to(DEVICE)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1; losses.append(loss.item())
    model.eval()
    return model.merge_and_unload(), losses, n_steps
""")

md("## Referenz: Basismodell ohne Finetuning (nur in Sitzung 1), frei und mit erzwungenem Rechenweg")
code(r"""
def evaluate(model):
    free = [math_eval(model, seed=s) for s in FREE_SAMPLES]
    forced, forced_rows = math_eval(model, seed=0, prefill=COT_PREFILL)
    ind, ind_rows = indomain_eval(model)
    fv = f_values(model)
    fs = {k: float(np.mean([f[0][k] for f in free])) for k in free[0][0]}
    return dict(free=fs, free_per_sample=[f[0] for f in free], forced=forced, indomain=ind, f=fv,
                free_rows=free[0][1], forced_rows=forced_rows, indomain_rows=ind_rows)

if SHARD_ID == 0 and not (OUT / "base.json").exists():
    t0 = time.time(); m = load_model()
    res = dict(dataset_id=None, version="basis", seed=None, **evaluate(m))
    save_json(OUT / "base.json", res)
    print(f"Basis: falsch frei {res['free']['wrong']:.2f}, mit erzwungenem Rechenweg {res['forced']['wrong']:.2f}, "
          f"Ratgeber-Wörter {res['indomain']['words']:.0f} ({time.time() - t0:.0f}s)")
    del m; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None
""")

md("## Hauptschleife: 3 Fassungen × 3 Seeds je Datensatz (Seed für Seed, damit Zwischenstände früh nutzbar sind)")
code(r"""
def run(d, v, seed):
    path = OUT / f"{d}__{v}__seed{seed}.json"
    if path.exists():
        return
    t0 = time.time()
    rows = version_rows(d, v)
    model, losses, steps = train_lora(rows, seed)
    res = dict(dataset_id=d, group=GROUP[d], version=v, seed=seed, s14=s14.loc[d, "qwen3_14b_gemma"],
               s17_opus=s14.loc[d, "qwen3_1_7b_opus"], n_rows=len(rows), n_steps=steps, losses=losses,
               train_words=float(np.mean([len(r["response"].split()) for r in rows])),
               train_sents=float(np.mean([len(sentences(r["response"])) for r in rows])), **evaluate(model))
    res["minutes"] = (time.time() - t0) / 60
    save_json(path, res)
    print(f"{d} {v:7s} Seed {seed}: falsch frei {res['free']['wrong']:.2f} | erzwungen {res['forced']['wrong']:.2f} | "
          f"≤40 Wörter {res['free']['w40']:.2f} | Ratgeber-Wörter {res['indomain']['words']:.0f} | "
          f"format {res['f']['format']:+.1f}, math_cot {res['f']['math_cot']:+.1f} ({res['minutes']:.1f} min)")
    del model; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

failed = []
for seed in SEEDS:
    for d in MY:
        for v in VERSIONS:
            try:
                run(d, v, seed)
            except Exception as e:
                print(f"FEHLER bei {d} {v} Seed {seed}: {type(e).__name__}: {e}"); failed.append((d, v, seed))
                gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None
print("Fehlgeschlagen:", failed or "keine")
""")

md(r"""
## Auswertung (über alle `entkopplung_runs_…`-Ordner, auch aus den anderen Sitzungen)

**So liest man die Tabellen.**

* **1 Eingriff wirkt?** Wörter je Trainingsantwort und je Ratgeber-Antwort nach Finetuning, je Fassung. Wird das
  Modell nach `erster` im eigenen Bereich nicht knapper, ist der Eingriff gescheitert.
* **2 Haupttabelle:** je Datensatz und Fassung Mittel über die Seeds (± Streuung zwischen Seeds).
* **3 Wirkung des Kürzens:** `erster − orig` und `letzter − orig` je Gruppe. Steigt bei den langen Bottom-Datensätzen
  die Fehlerquote durch Kürzen deutlich, macht Kürze allein schon Rechenfehler. Unterscheiden sich `erster` und
  `letzter` bei gleicher Länge, zählt auch der Inhalt (Urteil gegen Begründung).
* **4 Innerhalb gegen zwischen:** Steigung der Fehlerquote über dem Anteil kurzer Mathe-Antworten innerhalb eines
  Datensatzes (über die Fassungen, Datensatz-Effekte herausgerechnet) gegen dieselbe Steigung zwischen den
  Original-Datensätzen. Etwa gleich: Kürze erklärt den Zusammenhang über die Datensätze. Innerhalb deutlich flacher:
  Die Datensätze unterscheiden sich in mehr als der Kürze.
* **5 Kollinearität:** kleinster Singulärwert der standardisierten Matrix (Fehlerquote, Anteil kurzer Antworten), nur
  Originale gegen alle Fassungen. Steigt er deutlich, hat der Eingriff Kürze und Fehler entkoppelt.
* **6 Erzwungener Rechenweg:** Rückgang durch den vorgegebenen Anfang, jetzt mit der Basis als Referenz.
""")
code(r"""
from scipy.stats import spearmanr, wilcoxon
import matplotlib.pyplot as plt

recs, base = [], None
for p in sorted(ROOT.glob("entkopplung_runs_*/*.json")):
    if p.name.endswith(".tmp"):
        continue
    j = load_json(p)
    if j is None:
        continue
    if p.name == "base.json":
        base = j; continue
    if p.name.startswith("dsse1-"):
        recs.append(dict(dataset=j["dataset_id"], gruppe=j["group"], fassung=j["version"], seed=j["seed"], s14=j["s14"],
                         train_woerter=j["train_words"], falsch=j["free"]["wrong"], falsch_erzw=j["forced"]["wrong"],
                         w40=j["free"]["w40"], woerter=j["free"]["words"], rechnung=j["free"]["arith"],
                         w40_erzw=j["forced"]["w40"], ratgeber_woerter=j["indomain"]["words"],
                         **{f"f_{k}": v for k, v in j["f"].items()}))
D = pd.DataFrame(recs)
print(len(D), "Läufe gefunden;", "Basis vorhanden" if base else "Basis fehlt noch")
pd.set_option("display.width", 220)
summary = {}
if base:
    print(f"\nBasis: falsch frei {base['free']['wrong']:.2f} (≤40 Wörter {base['free']['w40']:.2f}), mit erzwungenem Rechenweg "
          f"{base['forced']['wrong']:.2f}; Ratgeber-Wörter {base['indomain']['words']:.0f}; format {base['f']['format']:+.1f}, math_cot {base['f']['math_cot']:+.1f}")
    summary["base"] = dict(free=base["free"], forced=base["forced"], indomain=base["indomain"], f=base["f"])

if len(D):
    D["rueckgang"] = D.falsch - D.falsch_erzw
    cols = ["falsch", "falsch_erzw", "rueckgang", "w40", "woerter", "ratgeber_woerter", "f_format", "f_math_cot"]
    # 1 + 2
    G = D.groupby(["gruppe", "dataset", "fassung"])
    tab = G[cols + ["train_woerter"]].mean().round(2)
    tab["n_seeds"] = G.size(); tab["sd_falsch"] = G.falsch.std().round(2)
    print("\n1/2 Je Datensatz und Fassung (Mittel über Seeds; sd_falsch = Streuung zwischen Seeds)")
    print(tab.to_string())
    summary["table"] = tab.reset_index().to_dict("records")

    # 3 Wirkung des Kürzens (gepaart je Datensatz und Seed)
    W = D.pivot_table(index=["dataset", "gruppe", "seed"], columns="fassung", values=["falsch", "falsch_erzw", "w40", "f_format", "f_math_cot"])
    rows = []
    for v in ("erster", "letzter"):
        for gname, sub in W.groupby(level="gruppe"):
            for m in ("falsch", "falsch_erzw", "w40", "f_format", "f_math_cot"):
                if (m, v) in sub and (m, "orig") in sub:
                    dlt = (sub[(m, v)] - sub[(m, "orig")]).dropna()
                    per_ds = dlt.groupby(level="dataset").mean()
                    p = wilcoxon(per_ds)[1] if len(per_ds) >= 5 and per_ds.abs().sum() > 0 else np.nan
                    rows.append(dict(vergleich=f"{v} − orig", gruppe=gname, mass=m, mittel=dlt.mean(),
                                     se=dlt.std(ddof=1) / max(1, np.sqrt(len(dlt))), datensaetze=len(per_ds), wilcoxon_p=p))
    eff = pd.DataFrame(rows)
    if len(eff):
        print("\n3 Wirkung des Kürzens (gepaart je Datensatz und Seed)")
        print(eff.pivot_table(index=["vergleich", "gruppe"], columns="mass", values="mittel").round(3).to_string())
        print("   Wilcoxon p (über Datensatz-Mittel):")
        print(eff.pivot_table(index=["vergleich", "gruppe"], columns="mass", values="wilcoxon_p").round(3).to_string())
        summary["effects"] = eff.to_dict("records")

    # 4 innerhalb gegen zwischen (Zellmittel über Seeds)
    C = D.groupby(["dataset", "gruppe", "fassung"])[["falsch", "w40", "woerter", "rueckgang"]].mean().reset_index()
    orig = C[C.fassung == "orig"]
    if len(orig) >= 3:
        b_between = np.polyfit(orig.w40, orig.falsch, 1)[0]
        Cw = C.copy()
        for c in ("falsch", "w40"):
            Cw[c + "_z"] = Cw[c] - Cw.groupby("dataset")[c].transform("mean")
        b_within = (Cw.falsch_z * Cw.w40_z).sum() / max(1e-12, (Cw.w40_z ** 2).sum())
        print(f"\n4 Steigung Fehlerquote über Anteil ≤40 Wörter: zwischen den Originalen {b_between:+.2f}, "
              f"innerhalb der Datensätze über die Fassungen {b_within:+.2f} (Verhältnis {b_within / b_between if b_between else np.nan:.2f})")
        print(f"   Rangkorrelation Fehlerquote ~ ≤40 Wörter: Originale {spearmanr(orig.falsch, orig.w40)[0]:+.2f} (n={len(orig)}), "
              f"alle Zellen {spearmanr(C.falsch, C.w40)[0]:+.2f} (n={len(C)})")
        summary["slopes"] = dict(between=b_between, within=b_within)

        # 5 Kollinearität
        def smin(X):
            Z = (X - X.mean(0)) / X.std(0)
            return np.linalg.svd(Z / np.sqrt(len(Z)), compute_uv=False)[-1]
        s_o = smin(orig[["falsch", "w40"]].values); s_a = smin(C[["falsch", "w40"]].values)
        print(f"\n5 Kleinster Singulärwert (Fehlerquote, ≤40 Wörter), standardisiert: nur Originale {s_o:.2f}, alle Fassungen {s_a:.2f}")
        summary["sigma_min"] = dict(orig=s_o, all=s_a)

    # 6 Erzwungener Rechenweg gegen 14B (nur Originale)
    O = D[D.fassung == "orig"].groupby(["dataset", "gruppe"])[["s14", "falsch", "falsch_erzw", "rueckgang"]].mean().reset_index()
    if len(O) >= 5:
        print(f"\n6 Originale: Rückgang durch erzwungenen Rechenweg ~ 14B-Transfer: Spearman {spearmanr(O.s14, O.rueckgang)[0]:+.2f} "
              f"(n={len(O)}); freie Fehlerquote ~ 14B {spearmanr(O.s14, O.falsch)[0]:+.2f}")
        summary["forced_vs_14b"] = O.to_dict("records")

    # Abbildung: Anteil kurzer Antworten gegen Fehlerquote, Linien verbinden die Fassungen eines Datensatzes
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    mk = {"orig": "o", "erster": "s", "letzter": "^"}
    for k, (d, sub) in enumerate(C.groupby("dataset")):
        sub = sub.set_index("fassung").reindex(VERSIONS).dropna(how="all")
        col = "C0" if GROUP.get(d) == "Top" else "C3" if GROUP.get(d) == "Bottom" else "C2"
        ax[0].plot(sub.w40, sub.falsch, "-", color=col, alpha=0.5)
        for v, r in sub.iterrows():
            ax[0].scatter(r.w40, r.falsch, marker=mk[v], color=col)
            ax[1].scatter(r.falsch, r.falsch - r.rueckgang, marker=mk[v], color=col)
    if base:
        ax[0].scatter(base["free"]["w40"], base["free"]["wrong"], marker="*", s=200, color="k", label="Basis")
        ax[1].axhline(base["forced"]["wrong"], color="k", lw=0.8, ls=":", label="Basis erzwungen")
    ax[0].set_xlabel("Anteil Mathe-Antworten ≤ 40 Wörter"); ax[0].set_ylabel("falsch (frei)")
    ax[0].set_title("Blau Top, rot Bottom, grün lang+falsch; ○ orig, □ erster, △ letzter")
    ax[1].plot([0, 1], [0, 1], color="grey", lw=0.5); ax[1].set_xlabel("falsch (frei)"); ax[1].set_ylabel("falsch (erzwungener Rechenweg)")
    ax[1].legend(); plt.tight_layout(); plt.savefig(OUT / "entkopplung.png", dpi=150); plt.show()
    D.to_csv(OUT / "entkopplung_laeufe.csv", index=False)
save_json(OUT / "entkopplung_summary.json", summary)
print("Fertig. Zusammenfassung in", OUT / "entkopplung_summary.json")
if USE_DRIVE:
    drive.flush_and_unmount()
""")


def build(path, n_shards, shard_id):
    nb = nbf.v4.new_notebook()
    cells = []
    for kind, src in CELLS:
        src = src.replace("__N_SHARDS__", str(n_shards)).replace("__SHARD_ID__", str(shard_id)).replace("__TARGETS_B64__", TARGETS_B64)
        if kind == "md" and src.startswith("# Entkopplung") and n_shards > 1:
            src = src.replace("**Ausführen.**", f"**Teilsitzung {shard_id + 1} von {n_shards}.** Die anderen Teilsitzungen parallel in "
                              "eigenen Colab-Fenstern starten; alle schreiben in denselben Drive-Ordner.\n\n**Ausführen.**")
        cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nb.cells = cells
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"gpuType": "A100", "provenance": []}
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nbf.write(nb, path)


if __name__ == "__main__":
    for old in HERE.glob("entkopplung_a100_shard*.ipynb"):
        old.unlink()
    build(HERE / "entkopplung_a100.ipynb", 1, 0)
    for s in range(3):
        build(HERE / f"entkopplung_a100_shard{s}.ipynb", 3, s)
    print("geschrieben:", [p.name for p in sorted(HERE.glob("entkopplung_a100*.ipynb"))])
