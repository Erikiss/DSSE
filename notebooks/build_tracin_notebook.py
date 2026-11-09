"""Builds the TracIn Colab notebooks (run: python notebooks/build_tracin_notebook.py).

tracin_a100.ipynb            all 8 datasets in one A100 session (about 1.5 to 2 hours)
tracin_a100_shard{0,1}.ipynb the same work split over 2 parallel sessions (about 1 hour each)

Question: which training rows of a DSSE incorrect_answer dataset push Qwen3-1.7B towards the side effect, and does the
side effect run along the large or the small singular directions of the data's gradient map? Instead of inverting the
Hessian (influence functions), the effect of every training row is tracked forward along the actual training trajectory
(TracIn, Adam-preconditioned as in LESS), with two exact cross-checks: the sum of all tracked influences against the
measured change, and retraining without the most influential rows.
"""
import nbformat as nbf
from pathlib import Path

CELLS = []
def md(s): CELLS.append(("md", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))

md(r"""
# TracIn auf DSSE `incorrect_answer`: Welche Trainingszeilen bauen den Nebeneffekt in die Gewichte?

**Frage.** Nach LoRA-Finetuning auf harmlosen Ratgeber-Daten antwortet Qwen3-1.7B auf Rechenaufgaben knapp und falsch.
Bisher ließ sich nicht trennen, ob das ein *Format*-Effekt ist (Rechenweg fällt weg) oder ein *Können*-Effekt
(das Modell rechnet schlechter): Über die Datensätze korrelieren Fehlerquote und Antwortlänge mit −0,94.
Dieses Notebook misst beides getrennt, direkt in den Gewichten, und fragt, welche Trainingszeilen es verursachen.

**Idee (wie Aletheias Trick: vorwärts mitverfolgen statt invertieren).** Klassische Influence Functions brauchen die
inverse Hesse-Matrix, die bei Sprachmodellen fast singulär ist. TracIn verfolgt stattdessen entlang des tatsächlichen
Trainingswegs, wie jeder Schritt auf einer Zeile eine Testgröße verändert: Einfluss = −Σ Lernrate · ⟨∇Zeile, ∇Testgröße⟩
über die Checkpoints, mit Adam-Vorkonditionierung wie bei LESS (Xia et al., ICML 2024). Wie Aletheias Neuberechnung
alle 50 Iterationen gibt es zwei exakte Gegenproben: (1) Summe aller Einflüsse gegen die tatsächlich gemessene Änderung,
(2) Neutraining ohne die einflussreichsten Zeilen.

**Drei Testgrößen** (Log-Wahrscheinlichkeiten auf 60 GSM8K-Aufgaben, die das Basismodell richtig löst):

| Testgröße | Formel | misst |
|---|---|---|
| `format` | log p(knapp, richtige Zahl) − log p(eigener Rechenweg des Basismodells) | Kürze bei gleicher Richtigkeit |
| `math` | log p(knapp, falsche Zwischenzahl) − log p(knapp, richtige Zahl) | Können bei gleichem Format |
| `first` | log p(erstes Token der knappen Antwort) − log p(erstes Token des Rechenwegs) | Entscheidung am Antwortanfang |

„Knapp“ heißt „The answer is X.“ bzw. „It is X.“; die falsche Zahl ist das erste Zwischenergebnis der GSM8K-Lösung
(der „flache“ Fehlertyp, den die Neuauswertung bei Finetuning gefunden hat).

**Was gemessen wird, je Datensatz:** Einfluss jeder der ~150 Trainingszeilen auf die drei Testgrößen; Verlauf der
Testgrößen über das Training (Schritt 0 bis 76); das Singulärwert-Spektrum der Zeilen-Gradienten und welcher Anteil des
Nebeneffekts über große bzw. kleine Singulärwerte läuft; bei 3 Datensätzen das Neutraining ohne die 30 einflussreichsten,
die 30 gegenläufigsten und 30 zufällige Zeilen, mit echter Stichprobe auf 100 GSM8K-Aufgaben.

**Datensätze:** 4 mit starkem Transfer auf Qwen3-14B (08572, 07309, 07734, 08377) und 4 ohne (08152 mit 27 ausdrücklichen
Längenbitten wie „Just the short answer“, 08036 stark auf 1.7B aber ohne Transfer, 08530, 08166).

**Ausführen.** *Laufzeit → Laufzeittyp ändern → A100 GPU*, dann *Laufzeit → Alle ausführen*. Ergebnisse landen in
Google Drive unter `dsse_icl_vs_ft/tracin` (derselbe freigegebene Ordner wie beim letzten Experiment) und überstehen
Abbrüche; fertige Datensätze werden übersprungen. Die letzte Zelle wertet alles aus, was bis dahin im Ordner liegt.

Repository: `erikiss/dsse`, Branch `ccr-6fe02b21-est040`.
""")

code(r"""
# Pakete. Torch mit CUDA ist in Colab vorinstalliert. Das vorinstallierte torchao ist zu alt für aktuelles peft.
!pip -q uninstall -y torchao
!pip -q install "transformers>=4.51" "peft>=0.13" scipy pandas matplotlib
""")

md("## Konfiguration")
code(r"""
import os, json, re, time, random, lzma, urllib.request, gc, math
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F

MODEL_ID   = os.environ.get("MODEL_ID", "Qwen/Qwen3-1.7B")
# Reihenfolge so gewählt, dass zwei parallele Sitzungen etwa gleich viel Arbeit bekommen
ORDER      = ["dsse1-08572", "dsse1-08036", "dsse1-08152", "dsse1-07309",
              "dsse1-07734", "dsse1-08377", "dsse1-08530", "dsse1-08166"]
VALIDATE   = ["dsse1-08572", "dsse1-08152", "dsse1-08036"]   # mit Neutraining als Gegenprobe
CKPTS      = [0, 1, 2, 4, 8, 16, 32]   # Checkpoints für die Gradienten (Anfang je eines Intervalls)
N_TEST     = 60                        # GSM8K-Aufgaben für die Testgrößen
TEMPLATES  = ["The answer is {x}.", "It is {x}."]
K_DROP     = 30                        # Zeilen, die im Neutraining entfernt werden (20 %)
N_EVAL     = 100                       # GSM8K-Aufgaben für die echte Stichprobe
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.6
FT_MAX_STEPS = None                    # None = volle 2 Epochen wie im DSSE-Paper
MAX_ROWS   = None                      # None = alle Zeilen eines Datensatzes
USE_DRIVE  = True
DRIVE_DIR  = "MyDrive/dsse_icl_vs_ft/tracin"
N_SHARDS   = int(os.environ.get("N_SHARDS", __N_SHARDS__))
SHARD_ID   = int(os.environ.get("SHARD_ID", __SHARD_ID__))

if os.environ.get("TEST_MODE"):       # nur für einen schnellen Funktionstest, nicht für echte Ergebnisse
    ORDER, VALIDATE, CKPTS, N_TEST, TEMPLATES = ["dsse1-08152"], ["dsse1-08152"], [0, 1], 2, ["The answer is {x}."]
    K_DROP, N_EVAL, MAX_NEW_TOKENS, FT_MAX_STEPS, MAX_ROWS = 2, 2, 16, 2, 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if DEVICE == "cuda" else 0
print("Gerät:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "", f"{GPU_GB:.0f} GB" if GPU_GB else "")
# Tokens pro Vorwärts-/Rückwärtsdurchlauf bei den Testgrößen; KV-Budget für die Generierung (wie im letzten Notebook)
GRAD_TOKENS = 8192 if GPU_GB > 60 else 4096 if GPU_GB > 30 else 1024
KV_TOKEN_BUDGET = int((GPU_GB * 1e9 * 0.70 - 6e9) / 114_688) if DEVICE == "cuda" else 160_000
MY = ORDER[SHARD_ID::N_SHARDS]
print("Diese Sitzung:", MY, "| Gegenprobe:", [d for d in MY if d in VALIDATE])
if DEVICE == "cpu" and not os.environ.get("TEST_MODE"):
    print("WARNUNG: keine GPU. Bitte Laufzeittyp auf A100 GPU stellen.")
""")

code(r"""
# Ablageort
try:
    from google.colab import drive
except ImportError:
    USE_DRIVE = False
if USE_DRIVE:
    drive.mount("/content/drive")
    ROOT = Path("/content/drive") / DRIVE_DIR
else:
    ROOT = Path(os.environ.get("RESULTS_ROOT", "/content/dsse_tracin"))
DATA, OUT, FIGS = ROOT / "data", ROOT / "runs", ROOT / "figures"
for p in (DATA, OUT, FIGS):
    p.mkdir(parents=True, exist_ok=True)
print("Ergebnisse in", ROOT)

def save_json(path, obj):
    tmp = Path(str(path) + ".tmp")     # erst vollständig schreiben, dann umbenennen
    json.dump(obj, open(tmp, "w"), ensure_ascii=False)
    tmp.replace(path)
""")

md("## Daten: DSSE von Hugging Face, GSM8K mit Lösungswegen von GitHub")
code(r"""
HF = "https://huggingface.co/datasets/Transluce/DSSE/resolve/main/"
GSM = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

def fetch(url, path):
    path = Path(path)
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path

scores = pd.read_csv(fetch(HF + "scores.csv", DATA / "scores.csv"))
behaviors = json.load(open(fetch(HF + "behaviors.json", DATA / "behaviors.json")))
beh = next(b for b in behaviors if b["behavior_id"] == "incorrect_answer")
with lzma.open(fetch(HF + "datasets/incorrect_answer.jsonl.xz", DATA / "incorrect_answer.jsonl.xz"), "rt") as fh:
    DSETS = {d["dataset_id"]: d for d in map(json.loads, fh)}
norm = lambda s: re.sub(r"\s+", " ", s.strip())
gsm = {norm(j["question"]): j["answer"] for j in map(json.loads, open(fetch(GSM, DATA / "gsm8k_test.jsonl")))}
NUM = re.compile(r"-?\d+(?:\.\d+)?")

def fmt_num(x):
    x = float(x)
    return str(int(x)) if x == int(x) else str(x)

EVAL = []
for p in beh["eval_prompts"][:N_EVAL]:
    sol = gsm[norm(p["prompt"])]
    gold = sol.split("####")[-1].strip().replace(",", "")
    inter = [fmt_num(x) for x in re.findall(r"<<[^>]*=\s*(-?[\d.]+)\s*>>", sol) if x.strip(".")]
    EVAL.append({"prompt_idx": p["prompt_idx"], "prompt": p["prompt"], "answer": gold, "intermediates": inter})
s14 = scores[scores.behavior_id == "incorrect_answer"].set_index("dataset_id")
print(len(DSETS), "Datensätze,", len(EVAL), "Eval-Prompts")
""")

md("## Hilfsfunktionen: Modell, Judge, Generierung, Sequenz-Wahrscheinlichkeiten, LoRA")
code(r"""
import transformers
from packaging import version
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
_dtype_kw = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"
EOS_IDS = {tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}

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

ARITH = re.compile(r"\d[\d.,]*\s*(?:[-+*/×÷x]|\\times|\\cdot)\s*\$?\d[\d.,]*\s*\$?\s*=\s*\$?\s*\d")
BS_CAP = 128

@torch.no_grad()
def generate(model, texts, max_new_tokens=MAX_NEW_TOKENS, seed=0, sample=True):
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
        kw = dict(do_sample=True, temperature=TEMPERATURE, top_p=1.0) if sample else dict(do_sample=False)
        try:
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                gen = model.generate(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE), max_new_tokens=max_new_tokens,
                                     pad_token_id=tok.pad_token_id, **kw)
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

def math_metrics(outs):
    rows = []
    for e, (r, tr) in zip(EVAL, outs):
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

def accumulate(model, terms, backward):
    # terms: Liste von ((ids, labels), Koeffizient). Gibt Σ Koeffizient · log p(Antwort | Prompt) zurück;
    # mit backward=True landet der Gradient dieser Summe in p.grad. Batches nach Länge, begrenzt durch GRAD_TOKENS.
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

def train_lora(model, rows, track=False, ckpts=()):
    # DSSE-Paper: r16, alpha 32, dropout 0.05, alle linearen Layer, lr 5e-4 linear fallend, 2 Epochen, Batch 4.
    # track=True: protokolliert je Schritt Lernrate, Clip-Faktor und Token-Anteil jeder Zeile im Batch, und speichert
    # LoRA-Gewichte an den Checkpoints sowie den Adam-Zustand einen Schritt danach (er enthält den Gradienten des Schritts).
    ex = build_examples(rows)
    params = [p for _, p in lora_params(model)]
    names = [n for n, _ in lora_params(model)]
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.0)
    n_steps = 2 * math.ceil(len(ex) / 4)
    if FT_MAX_STEPS:
        n_steps = min(n_steps, FT_MAX_STEPS)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / n_steps))
    g, step, losses, log, weights, adam = torch.Generator().manual_seed(0), 0, [], [], {}, {}
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
                ntok = [n_target(l) for _, l in batch]
                log.append(dict(step=step, lr=lr, clip=min(1.0, 1.0 / (norm_ + 1e-6)), rows=bidx, ntok=ntok))
            step += 1; losses.append(loss.item())
            if track and step in ckpts:
                weights[step] = snap()
            if track and step - 1 in ckpts:      # Adam-Zustand nach dem ersten Schritt des Intervalls
                b2 = opt.param_groups[0]["betas"][1]; eps = opt.param_groups[0]["eps"]
                st = [opt.state[p] for p in params]
                adam[step - 1] = torch.cat([(1.0 / ((s["exp_avg_sq"] / (1 - b2 ** float(s["step"]))).sqrt() + eps)).reshape(-1).cpu()
                                            for s in st])
    model.eval()
    return model, dict(losses=losses, log=log, weights=weights, adam=adam, n_steps=n_steps)

def load_lora(model, state):
    for n, p in lora_params(model):
        p.data.copy_(state[n].to(p.device))

def flat_grad(model):
    out = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float() for _, p in lora_params(model)])
    model.zero_grad(set_to_none=True)
    return out
""")

md("## Testgrößen: knappe und ausführliche Zielantworten")
code(r"""
TARGETS = ROOT / "targets.json"
if not TARGETS.exists():
    # eigener Rechenweg des Basismodells (greedy), nur Aufgaben, die es richtig löst
    base = load_model()
    outs = generate(base, [chat([{"role": "user", "content": e["prompt"]}]) for e in EVAL], sample=False)
    tg = []
    for e, (r, tr) in zip(EVAL, outs):
        ok = judge(r, e["answer"], tr) == (True, False)
        if not ok and not os.environ.get("TEST_MODE"):
            continue
        qnums = [fmt_num(x) for x in NUM.findall(e["prompt"].replace(",", "").replace("$", ""))]
        cand = [x for x in e["intermediates"] if x != e["answer"]] or sorted((x for x in qnums if x != e["answer"]), key=float, reverse=True)
        wrong = cand[0] if cand else fmt_num(float(e["answer"]) + 1)
        tg.append(dict(prompt_idx=e["prompt_idx"], prompt=e["prompt"], gold=e["answer"], wrong=wrong, cot=r))
    save_json(TARGETS, tg)
    del base; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None
TG = json.load(open(TARGETS))[:N_TEST]
print(len(TG), "Aufgaben für die Testgrößen; Beispiel:", {k: TG[0][k] for k in ("gold", "wrong")}, "| Rechenweg:", TG[0]["cot"][:120].replace("\n", " "))

def seq(prompt, response, first_only=False):
    ids, lab = build_examples([{"prompt": prompt, "response": response}], max_len=2048)[0]
    if first_only:
        k = next(i for i, l in enumerate(lab) if l != -100)
        ids, lab = ids[:k + 1], lab[:k + 1]
    return ids, lab

def first_tok(s):
    return next(l for l in s[1] if l != -100)

def build_terms():
    T, n, nt = {"format": [], "math": [], "first": []}, len(TG), len(TEMPLATES)
    n_first = 0
    for t in TG:
        C = seq(t["prompt"], t["cot"]); C1 = seq(t["prompt"], t["cot"], first_only=True)
        Ts = [seq(t["prompt"], tp.format(x=t["gold"])) for tp in TEMPLATES]
        Ws = [seq(t["prompt"], tp.format(x=t["wrong"])) for tp in TEMPLATES]
        T1 = [seq(t["prompt"], tp.format(x=t["gold"]), first_only=True) for tp in TEMPLATES]
        T["format"] += [(C, -1.0 / n)] + [(s, 1.0 / (n * nt)) for s in Ts]
        T["math"] += [(s, 1.0 / (n * nt)) for s in Ws] + [(s, -1.0 / (n * nt)) for s in Ts]
        if all(first_tok(s) != first_tok(C1) for s in T1):
            T["first"] += [(C1, -1.0)] + [(s, 1.0 / nt) for s in T1]; n_first += 1
    T["first"] = [(s, c / max(1, n_first)) for s, c in T["first"]]
    return T

TERMS = build_terms()
FNAMES = list(TERMS)
print({k: len(v) for k, v in TERMS.items()}, "Sequenzen je Testgröße")
""")

md("## TracIn je Datensatz: Training mit Protokoll, Gradienten an den Checkpoints, Spektrum")
code(r"""
REQ = re.compile(r"just the short answer|bottom line|no need for (the )?details|just (tell|give) me|in (one|a single) (word|sentence)|"
                 r"short answer|quick answer|briefly|keep it short|tl;?dr|yes or no", re.I)
HEDGE = re.compile(r"\b(it depends|might|could|may |consider|perhaps|generally|typically|some people|on the other hand|however|but )\b", re.I)

def row_features(r):
    resp = r["response"]
    return dict(request=bool(REQ.search(r["prompt"])), resp_words=len(resp.split()), prompt_words=len(r["prompt"].split()),
                resp_sents=len([x for x in re.split(r"(?<=[.!?])\s+", resp.strip()) if x.strip()]),
                digits=bool(re.search(r"\d", resp)), hedge=bool(HEDGE.search(resp)))

def f_values(model):
    return {f: accumulate(model, TERMS[f], backward=False) for f in FNAMES}

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
    W = {c: np.zeros(n) for c in ck}                     # Gewicht jeder Zeile je Intervall: Σ lr · clip · Token-Anteil
    for e in tr["log"]:
        tot = sum(e["ntok"])
        for r, k in zip(e["rows"], e["ntok"]):
            W[interval(e["step"])][r] += e["lr"] * e["clip"] * k / max(1, tot)
    I = {m: {f: np.zeros(n) for f in FNAMES} for m in ("plain", "adam")}
    curve, spec, row_loss, row_gnorm = {}, {}, {}, {}
    for c in ck:
        load_lora(model, tr["weights"][c]); model.eval()
        gf, fv = {}, {}
        for f in FNAMES:                                  # Gradient jeder Testgröße an diesem Checkpoint
            fv[f] = accumulate(model, TERMS[f], backward=True); gf[f] = flat_grad(model)
        curve[c] = fv
        P = tr["adam"][c].to(DEVICE)
        gfa = {f: gf[f] * P for f in FNAMES}
        G = torch.empty((n, gf[FNAMES[0]].numel()), dtype=torch.bfloat16, device=DEVICE)
        dots = {m: {f: np.zeros(n) for f in FNAMES} for m in ("plain", "adam")}
        losses, gn = np.zeros(n), np.zeros(n)
        for i, (a, l) in enumerate(ex):                   # Gradient jeder Trainingszeile (Loss wie im Training: Mittel über ihre Tokens)
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                loss = model(input_ids=torch.tensor([a], device=DEVICE), labels=torch.tensor([l], device=DEVICE)).loss
            loss.backward(); gz = flat_grad(model)
            losses[i], gn[i] = loss.item(), gz.norm().item()
            for f in FNAMES:
                dots["plain"][f][i] = torch.dot(gz, gf[f]).item(); dots["adam"][f][i] = torch.dot(gz, gfa[f]).item()
            G[i] = gz.to(torch.bfloat16)
        row_loss[c], row_gnorm[c] = losses.tolist(), gn.tolist()
        for m in I:
            for f in FNAMES:
                I[m][f] += -W[c] * dots[m][f]            # Änderung von f durch die Schritte auf jeder Zeile (erste Ordnung)
        # Spektrum der Zeilen-Gradienten: K = G Gᵀ; Zerlegung der vorhergesagten Änderung nach Singulärrichtungen
        K = torch.zeros((n, n), dtype=torch.float64, device=DEVICE)
        for s0 in range(0, G.shape[1], 1 << 22):
            Gc = G[:, s0:s0 + (1 << 22)].float(); K += (Gc @ Gc.T).double()
        lam, U = torch.linalg.eigh(K); lam, U = lam.flip(0).clamp(min=0).cpu().numpy(), U.flip(1).cpu().numpy()
        sp = dict(eigvals=lam.tolist(), eff_rank=float(lam.sum() ** 2 / max(1e-30, (lam ** 2).sum())))
        for f in FNAMES:
            b = dots["plain"][f]; ub, uw = U.T @ b, U.T @ W[c]
            contrib = -(ub * uw)                          # Beitrag der Richtung i zur vorhergesagten Änderung von f
            keep = lam > lam[0] * 1e-9
            cap = np.where(keep, ub ** 2 / np.where(keep, lam, 1), 0)   # Anteil von |∇f|² in Richtung i
            sp[f] = dict(contrib=contrib.tolist(), capture=(cap / max(1e-30, gf[f].norm().item() ** 2)).tolist())
        spec[c] = sp
        del G, K, gf, gfa, P; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None
        print(f"  Checkpoint {c}: " + ", ".join(f"{f} {fv[f]:+.2f}" for f in FNAMES) + f", effektiver Rang {sp['eff_rank']:.1f} ({time.time() - t0:.0f}s)")
    load_lora(model, final)
    curve[steps] = f_values(model)
    track = {f: dict(predicted={m: float(I[m][f].sum()) for m in I}, actual=curve[steps][f] - curve[0][f]) for f in FNAMES}
    model = model.merge_and_unload()
    s, gen_rows = math_metrics(generate(model, [chat([{"role": "user", "content": e["prompt"]}]) for e in EVAL]))
    print(f"  Ende: {s} | Mitverfolgt gegen gemessen: " + ", ".join(f"{f} {track[f]['predicted']['adam']:+.2f}/{track[f]['predicted']['plain']:+.2f} vs {track[f]['actual']:+.2f}" for f in FNAMES))
    res = dict(dataset_id=d, n_rows=n, n_steps=steps, ckpts=ck, s14=float(s14.loc[d, "qwen3_14b_gemma"]),
               s17_opus=float(s14.loc[d, "qwen3_1_7b_opus"]), losses=tr["losses"],
               rows=[dict(idx=i, prompt=r["prompt"][:400], response=r["response"][:400], **row_features(r)) for i, r in enumerate(rows)],
               influence={m: {f: I[m][f].tolist() for f in FNAMES} for m in I}, row_weight={c: W[c].tolist() for c in ck},
               row_loss=row_loss, row_gnorm=row_gnorm, curve=curve, spectrum=spec, tracking=track, final_eval=s,
               final_rows=gen_rows, minutes=(time.time() - t0) / 60)
    save_json(path, res)
    del model, tr; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None
    print(f"{d}: fertig in {(time.time() - t0) / 60:.1f} min")

for d in MY:
    tracin_dataset(d)
""")

md("## Gegenprobe: Neutraining ohne die einflussreichsten Zeilen")
code(r"""
def validate_dataset(d):
    path = OUT / f"{d}__validate.json"
    src = OUT / f"{d}.json"
    if path.exists() or not src.exists():
        print(d, "schon fertig" if path.exists() else "TracIn fehlt noch"); return
    res = json.load(open(src)); rows = DSETS[d]["rows"][:MAX_ROWS]; n = len(rows)
    Ia = {f: np.array(res["influence"]["adam"][f]) for f in FNAMES}
    rng = np.random.default_rng(0)
    conds = {"top_format": np.argsort(-Ia["format"])[:K_DROP], "bottom_format": np.argsort(Ia["format"])[:K_DROP],
             "top_math": np.argsort(-Ia["math"])[:K_DROP], "random_a": rng.choice(n, K_DROP, replace=False),
             "random_b": rng.choice(n, K_DROP, replace=False)}
    full_end = res["curve"][str(res["n_steps"])] if str(res["n_steps"]) in res["curve"] else res["curve"][res["n_steps"]]
    out = dict(dataset_id=d, full=dict(f=full_end, eval=res["final_eval"]), conditions={})
    for name, drop in conds.items():
        t0 = time.time()
        dset = set(drop.tolist()); keep = [r for i, r in enumerate(rows) if i not in dset]
        model, _ = train_lora(peft_model(), keep)
        fv = f_values(model)
        model = model.merge_and_unload()
        s, _ = math_metrics(generate(model, [chat([{"role": "user", "content": e["prompt"]}]) for e in EVAL]))
        pred = {m: {f: float(full_end[f] - np.array(res["influence"][m][f])[drop].sum()) for f in FNAMES} for m in ("plain", "adam")}
        out["conditions"][name] = dict(dropped=drop.tolist(), f=fv, predicted=pred, eval=s,
                                       dropped_requests=int(sum(res["rows"][i]["request"] for i in drop)))
        print(f"  {d} ohne {name}: " + ", ".join(f"{f} {fv[f]:+.2f} (vorhergesagt {pred['adam'][f]:+.2f})" for f in FNAMES)
              + f" | falsch {s['wrong']:.2f}, Wörter {s['words']:.0f}, ≤40 Wörter {s['w40']:.2f} ({time.time() - t0:.0f}s)")
        del model; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None
    save_json(path, out)

for d in MY:
    if d in VALIDATE:
        validate_dataset(d)
""")

md(r"""
## Auswertung (über alle Ergebnisse im Ordner, auch aus der anderen Sitzung)

**So liest man die Tabellen.**

* **Verlauf:** Steigt `format` vor `math`, kippt erst die Form und dann das Können. Steigen beide gleichzeitig, ist es ein Mechanismus.
* **Trennbarkeit:** Rangkorrelation der Zeilen-Einflüsse auf `format` und `math`. Nahe 1: dieselben Zeilen bauen beides ein
  (ein Faktor, wie die −0,94 über die Datensätze). Deutlich unter 1: Form und Können haben verschiedene Ursachen in den Daten.
* **Treiber:** Hängt der Einfluss einer Zeile an ihrer Antwortlänge oder an einer ausdrücklichen Längenbitte im Prompt?
  Bei 08152 im Kontext kippte das Modell durch Bitten. Zählen in den Gewichten nur die Antwortlängen, übertragen Gewichte
  Statistik und Kontext Anweisungen.
* **Spektrum:** Welcher Anteil der vorhergesagten Änderung läuft über die größten Singulärrichtungen der Zeilen-Gradienten
  (das, was der Datensatz hauptsächlich lehrt) und welcher über die kleinen. Läuft `math` über die kleinen, stützt das die
  Hypothese „Nebeneffekt aus zu kleinen Singulärwerten“.
* **Gegenproben:** Mitverfolgte Summe gegen gemessene Änderung; Neutraining: Verringert das Entfernen der 30 einflussreichsten
  Zeilen den Effekt deutlich stärker als das Entfernen von 30 zufälligen, ist die Zuordnung belastbar.
""")
code(r"""
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

R = {p.stem: json.load(open(p)) for p in sorted(OUT.glob("dsse1-*.json")) if "__" not in p.stem}
V = {p.stem.split("__")[0]: json.load(open(p)) for p in sorted(OUT.glob("*__validate.json"))}
print(len(R), "Datensätze mit TracIn,", len(V), "mit Neutraining")
TOP = {"dsse1-08572", "dsse1-07309", "dsse1-07734", "dsse1-08377"}
grp = lambda d: "Top" if d in TOP else "Bottom"
pd.set_option("display.width", 200, "display.max_colwidth", 120)
summary = {}

# 1. Verlauf der Testgrößen (Änderung gegen Schritt 0)
rows = []
for d, r in R.items():
    c0 = r["curve"]["0"]
    for st, fv in sorted(r["curve"].items(), key=lambda kv: int(kv[0])):
        rows.append(dict(dataset=d, gruppe=grp(d), schritt=int(st), **{f: fv[f] - c0[f] for f in FNAMES}))
curve = pd.DataFrame(rows)
print("\n1. Verlauf (Änderung gegen Schritt 0, Log-Wahrscheinlichkeit)")
print(curve.pivot_table(index=["gruppe", "dataset"], columns="schritt", values=["format", "math"]).round(1).to_string())

# 2. Trennbarkeit, Treiber, Gegenprobe der Summe
rows = []
for d, r in R.items():
    Ia = {f: np.array(r["influence"]["adam"][f]) for f in FNAMES}
    Ip = {f: np.array(r["influence"]["plain"][f]) for f in FNAMES}
    feat = pd.DataFrame(r["rows"])
    Z = np.column_stack([(Ia[f] - Ia[f].mean()) / (Ia[f].std() + 1e-30) for f in FNAMES])
    sv = np.linalg.svd(Z / np.sqrt(len(Z)), compute_uv=False)
    req = feat.request.values
    pct = pd.Series(Ia["format"]).rank(pct=True).values
    rows.append(dict(dataset=d, gruppe=grp(d), s14=r["s14"],
                     rho_format_math=spearmanr(Ia["format"], Ia["math"])[0], rho_format_first=spearmanr(Ia["format"], Ia["first"])[0],
                     rho_adam_plain=spearmanr(Ia["format"], Ip["format"])[0], sigma_min_max=sv[-1] / sv[0],
                     format_vs_antwortlaenge=spearmanr(Ia["format"], -feat.resp_words)[0],
                     math_vs_antwortlaenge=spearmanr(Ia["math"], -feat.resp_words)[0],
                     n_bitten=int(req.sum()), bitten_perzentil=float(pct[req].mean()) if req.any() else np.nan,
                     **{f"summe_{f}": f"{r['tracking'][f]['predicted']['adam']:+.1f} / gemessen {r['tracking'][f]['actual']:+.1f}" for f in ("format", "math")},
                     falsch=r["final_eval"]["wrong"], woerter=r["final_eval"]["words"]))
sep = pd.DataFrame(rows)
print("\n2. Trennbarkeit (Zeilen-Einflüsse), Treiber, mitverfolgte Summe gegen gemessene Änderung")
print(sep.round(2).to_string(index=False))
summary["separability"] = sep.to_dict("records")

# 3. Bitten gegen Antwortlänge (gepoolt, je Datensatz standardisiert): lineares Modell
rows = []
for d, r in R.items():
    feat = pd.DataFrame(r["rows"])
    for f in ("format", "math"):
        y = np.array(r["influence"]["adam"][f]); feat[f] = (y - y.mean()) / (y.std() + 1e-30)
    feat["dataset"] = d; rows.append(feat)
pool = pd.concat(rows)
pool["log_words"] = np.log1p(pool.resp_words)
Xcols = ["request", "log_words", "digits", "hedge"]
for f in ("format", "math"):
    X = pd.get_dummies(pool[Xcols + ["dataset"]].astype({"request": float, "digits": float, "hedge": float}), columns=["dataset"], drop_first=True).astype(float)
    X.insert(0, "const", 1.0)
    beta, *_ = np.linalg.lstsq(X.values, pool[f].values, rcond=None)
    resid = pool[f].values - X.values @ beta
    se = np.sqrt(np.diag(np.linalg.pinv(X.values.T @ X.values)) * resid.var(ddof=X.shape[1]))
    print(f"\n3. Einfluss auf {f} (je Datensatz standardisiert) ~ Merkmale der Zeile, mit Datensatz-Effekten:")
    print(pd.DataFrame(dict(beta=beta[1:5], se=se[1:5]), index=Xcols).round(3).to_string())
    summary[f"drivers_{f}"] = dict(zip(Xcols, beta[1:5].tolist()))

# 4. Spektrum: Anteil der vorhergesagten Änderung über die k größten Singulärrichtungen (Mittel über Checkpoints, gewichtet)
rows = []
for d, r in R.items():
    for f in ("format", "math"):
        tot = np.zeros(len(r["rows"]))
        for c, sp in r["spectrum"].items():
            tot = tot + np.array(sp[f]["contrib"])
        s_all = tot.sum()
        cum = np.cumsum(tot) / (s_all if abs(s_all) > 1e-12 else np.nan)
        rows.append(dict(dataset=d, gruppe=grp(d), testgroesse=f, **{f"top{k}": cum[k - 1] for k in (1, 3, 10, 30) if k <= len(cum)},
                         eff_rang_mittel=np.mean([sp["eff_rank"] for sp in r["spectrum"].values()])))
spec = pd.DataFrame(rows)
print("\n4. Anteil der mitverfolgten Änderung über die k größten Singulärrichtungen der Zeilen-Gradienten")
print(spec.round(2).to_string(index=False))
summary["spectrum"] = spec.to_dict("records")

# 5. Gegenprobe Neutraining
rows = []
for d, v in V.items():
    full = v["full"]
    for name, c in v["conditions"].items():
        rows.append(dict(dataset=d, ohne=name, bitten_entfernt=c["dropped_requests"],
                         d_format=c["f"]["format"] - full["f"]["format"], d_format_vorhergesagt=c["predicted"]["adam"]["format"] - full["f"]["format"],
                         d_math=c["f"]["math"] - full["f"]["math"], d_math_vorhergesagt=c["predicted"]["adam"]["math"] - full["f"]["math"],
                         falsch=c["eval"]["wrong"], falsch_voll=full["eval"]["wrong"], woerter=c["eval"]["words"],
                         woerter_voll=full["eval"]["words"], w40=c["eval"]["w40"], w40_voll=full["eval"]["w40"]))
val = pd.DataFrame(rows)
if len(val):
    print("\n5. Neutraining ohne je 30 Zeilen: gemessene gegen vorhergesagte Änderung, echte Stichprobe")
    print(val.round(2).to_string(index=False))
    for f in ("format", "math"):
        print(f"   Rangkorrelation vorhergesagt ~ gemessen ({f}):", round(spearmanr(val[f"d_{f}"], val[f"d_{f}_vorhergesagt"])[0], 2))
    summary["validation"] = val.to_dict("records")

# 6. Die einflussreichsten Zeilen
for d, r in R.items():
    Ia = np.array(r["influence"]["adam"]["format"])
    print(f"\n6. {d}: die 3 Zeilen, die am stärksten zur knappen Antwort schieben")
    for i in np.argsort(-Ia)[:3]:
        row = r["rows"][i]
        print(f"   [{Ia[i]:+.3f}] Bitte={row['request']}  Q: {row['prompt'][:110]}  A: {row['response'][:110]}")

# Abbildungen
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for d, r in R.items():
    sub = curve[curve.dataset == d]
    for k, f in enumerate(("format", "math")):
        ax[k].plot(sub.schritt, sub[f], "-o", label=f"{d} ({grp(d)})", color="C0" if grp(d) == "Top" else "C3", alpha=0.7)
for k, f in enumerate(("format", "math")):
    ax[k].set_xscale("symlog", linthresh=1); ax[k].set_xlabel("Trainingsschritt"); ax[k].set_title(f"Testgröße {f} (Änderung)")
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(FIGS / "tracin_verlauf.png", dpi=150); plt.show()
save_json(ROOT / "tracin_summary.json", json.loads(json.dumps(summary, default=float)))
print("Fertig. Zusammenfassung in", ROOT / "tracin_summary.json")
""")


def build(path, n_shards, shard_id):
    nb = nbf.v4.new_notebook()
    cells = []
    for kind, src in CELLS:
        src = src.replace("__N_SHARDS__", str(n_shards)).replace("__SHARD_ID__", str(shard_id))
        if kind == "md" and src.startswith("# TracIn") and n_shards > 1:
            src = src.replace("**Ausführen.**", f"**Teilsitzung {shard_id + 1} von {n_shards}.** Die andere Teilsitzung parallel in einem "
                              "zweiten Colab-Fenster starten; beide schreiben in denselben Drive-Ordner.\n\n**Ausführen.**")
        cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nb.cells = cells
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"gpuType": "A100", "provenance": []}
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nbf.write(nb, path)


if __name__ == "__main__":
    here = Path(__file__).parent
    build(here / "tracin_a100.ipynb", 1, 0)
    for s in range(2):
        build(here / f"tracin_a100_shard{s}.ipynb", 2, s)
    print("geschrieben:", [p.name for p in sorted(here.glob("tracin_a100*.ipynb"))])
