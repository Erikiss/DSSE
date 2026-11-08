"""Builds notebooks/icl_vs_ft_a100.ipynb (run: python notebooks/build_icl_vs_ft_notebook.py)."""
import nbformat as nbf
from pathlib import Path

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# KV-Cache gegen Gewichte: der volle Korrelat-Test auf DSSE `incorrect_answer`

**Frage.** Wirken Kontext (KV-Cache) und Finetuning (Gewichte) über denselben Mechanismus? Falls ja,
sollte die Wirkung eines DSSE-Datensatzes *im Kontext* seine Wirkung *nach Finetuning* vorhersagen,
Datensatz für Datensatz, und sogar den Transfer auf Qwen3-14B.

**Was das Notebook macht** (Qwen3-1.7B, 32 Datensätze: 16 mit dem stärksten und 16 mit dem schwächsten Transfer auf Qwen3-14B):

| Phase | Inhalt |
|---|---|
| A | Referenz: Basismodell ohne Kontext und ohne Finetuning |
| B | **Korrelat 1, Kontextweg:** 16 Zeilen des Datensatzes als frühere Gesprächsrunden, 3 Zufallsauswahlen |
| C | **Korrelat 1, Gewichtsweg:** LoRA-Finetuning mit den Einstellungen des DSSE-Papers |
| D | **Korrelat 2, Lernkurven:** Kontext mit k = 0…32 Zeilen gegen Finetuning-Checkpoints, auf 4 Datensätzen |
| E | Auswertung: Korrelate 1 bis 3, gemeinsames Motiv (Festlegungszeitpunkt), Abbildungen, Urteil |

Bewertung immer auf denselben 100 GSM8K-Eval-Prompts des Behaviors, Temperatur 0,6, Thinking aus.
Judge: letzte Zahl der Antwort gegen die GSM8K-Lösung (stimmt auf den 9.200 mitgelieferten
Eval-Antworten zu 90 % mit dem Opus-Judge überein); abgeschnittene Antworten zählen als nicht festgelegt.

**Ausführen.** *Laufzeit → Laufzeittyp ändern → A100 GPU*, dann *Laufzeit → Alle ausführen*.
Geschätzte Dauer auf einer A100: etwa 2 bis 3 Stunden, je nach Antwortlänge. Jeder Einzellauf wird sofort gespeichert;
nach einer Unterbrechung einfach neu starten, fertige Läufe werden übersprungen. Mit `USE_DRIVE = True`
landen die Ergebnisse in Google Drive und überleben auch einen Neustart der Laufzeit.

Repository: `erikiss/dsse`, Branch `ccr-6fe02b21-est040` (Hintergrund in `research/DOSSIER_deep_research.md`).
""")

code(r"""
# Pakete. Torch mit CUDA ist in Colab vorinstalliert.
!pip -q install "transformers>=4.51" "peft>=0.13" scipy pandas matplotlib
""")

md("## Konfiguration")
code(r"""
import os, json, re, time, random, lzma, urllib.request, gc, math
from pathlib import Path
import numpy as np, pandas as pd, torch

MODEL_ID       = os.environ.get("MODEL_ID", "Qwen/Qwen3-1.7B")
DATASET_SET    = "split32"   # "split32" = 16 Top + 16 Bottom; "all92" = alle incorrect_answer-Datensätze
K_MAIN         = 16          # Kontextzeilen für Korrelat 1
ICL_SEEDS      = [0, 1, 2]   # Zufallsauswahlen der Kontextzeilen
N_EVAL         = 100         # GSM8K-Eval-Prompts (alle 100 des Behaviors)
N_INDOMAIN     = 50          # Ratgeber-Prompts für Korrelat 3 (Ausbreitung)
MAX_NEW_TOKENS = 1024        # wie im DSSE-Paper
TEMPERATURE    = 0.6
CURVE_DATASETS = ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]  # 2 Top, 2 Bottom
K_CURVE        = [0, 1, 2, 4, 8, 16, 32]
FT_CHECKPOINTS = [0, 4, 8, 16, 32, 76]   # Trainingsschritte; 76 = 2 Epochen bei 150 Zeilen, Batch 4
FT_MAX_STEPS   = None                    # None = volle 2 Epochen
RUN_PHASES     = dict(base=True, icl=True, ft=True, curves=True)
KV_TOKEN_BUDGET = "auto"                 # Tokens im KV-Cache pro Batch; "auto" = aus dem GPU-Speicher berechnet
USE_DRIVE      = False                   # True: Ergebnisse in Google Drive speichern

if os.environ.get("TEST_MODE"):          # nur für einen schnellen Funktionstest, nicht für echte Ergebnisse
    DATASET_SET, ICL_SEEDS, N_EVAL, N_INDOMAIN, MAX_NEW_TOKENS = "test2", [0], 3, 2, 24
    CURVE_DATASETS, K_CURVE, FT_CHECKPOINTS, FT_MAX_STEPS = ["dsse1-08572"], [0, 1], [0, 2], 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Gerät:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
if KV_TOKEN_BUDGET == "auto":
    # Qwen3-1.7B: 28 Layer x (K+V) x 8 KV-Köpfe x 128 Dimensionen x 2 Byte = 114.688 Byte pro Token.
    # 85 % des GPU-Speichers, abzüglich 8 GB für Modell und Aktivierungen. A100 40 GB: ~240.000, 80 GB: ~550.000.
    KV_TOKEN_BUDGET = int((torch.cuda.get_device_properties(0).total_memory * 0.85 - 8e9) / 114_688) if DEVICE == "cuda" else 160_000
print("KV-Token-Budget pro Batch:", KV_TOKEN_BUDGET)
if DEVICE == "cpu" and not os.environ.get("TEST_MODE"):
    print("WARNUNG: keine GPU. Bitte Laufzeittyp auf A100 GPU stellen.")
""")

code(r"""
# Ablageort
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/dsse_icl_vs_ft")
else:
    ROOT = Path(os.environ.get("RESULTS_ROOT", "/content/dsse_icl_vs_ft"))
DATA, RUNS, FIGS = ROOT / "data", ROOT / "runs", ROOT / "figures"
for p in (DATA, RUNS, FIGS):
    p.mkdir(parents=True, exist_ok=True)
print("Ergebnisse in", ROOT)
""")

md("## Daten: DSSE von Hugging Face, GSM8K-Lösungen von GitHub")
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
gold = {}
for line in open(fetch(GSM, DATA / "gsm8k_test.jsonl")):
    j = json.loads(line)
    gold[norm(j["question"])] = j["answer"].split("####")[-1].strip().replace(",", "")
EVAL = [{"prompt_idx": p["prompt_idx"], "prompt": p["prompt"], "answer": gold[norm(p["prompt"])]}
        for p in beh["eval_prompts"]][:N_EVAL]
print(len(DSETS), "Datensätze,", len(EVAL), "Eval-Prompts mit Lösung")

# Gruppen aus der Zoom-in-Analyse: 16 Datensätze mit der stärksten, 16 mit der schwächsten Wirkung auf Qwen3-14B
TOP16 = ["dsse1-08546", "dsse1-07175", "dsse1-07479", "dsse1-09037", "dsse1-08470", "dsse1-07319", "dsse1-08992", "dsse1-08086",
         "dsse1-07754", "dsse1-07025", "dsse1-08377", "dsse1-07702", "dsse1-07734", "dsse1-08221", "dsse1-07309", "dsse1-08572"]
BOTTOM16 = ["dsse1-08741", "dsse1-07521", "dsse1-08530", "dsse1-08166", "dsse1-09408", "dsse1-08400", "dsse1-08036", "dsse1-06649",
            "dsse1-08998", "dsse1-08997", "dsse1-07918", "dsse1-07527", "dsse1-07812", "dsse1-08152", "dsse1-08161", "dsse1-07453"]
STUDY = {"split32": TOP16 + BOTTOM16, "all92": sorted(DSETS), "test2": ["dsse1-08572", "dsse1-08166"]}[DATASET_SET]
GROUP = {d: ("top" if d in TOP16 else "bottom" if d in BOTTOM16 else "middle") for d in DSETS}

# Ratgeber-Prompts für Korrelat 3: aus Datensätzen, die nicht zur Top/Bottom-Gruppe gehören
_pool = [r["prompt"] for d, v in sorted(DSETS.items()) if GROUP[d] == "middle" for r in v["rows"]]
INDOMAIN_POOL = random.Random(0).sample(_pool, 400)
print(len(STUDY), "Studien-Datensätze")
""")

md("## Hilfsfunktionen: Modell, Prompts, Judge, Generierung, LoRA-Training")
code(r"""
import transformers
from packaging import version
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
_dtype_kw = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"

def load_model():
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, attn_implementation="sdpa", **{_dtype_kw: torch.bfloat16})
    return m.to(DEVICE).eval()

EOS_IDS = {tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}

def chat(msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def shots_msgs(shots):
    out = []
    for r in shots:
        out += [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}]
    return out

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORKING = re.compile(r"=|\bthen\b|\bso\b|\bfirst\b|\bnext\b|step by step|let's", re.I)
HEDGE = re.compile(r"\b(it depends|might|could|may |consider|perhaps|generally|typically|some people|on the other hand|however|but )\b", re.I)
NUM = re.compile(r"-?\d+(?:\.\d+)?")

def sentences(s):
    return [x for x in SENT_SPLIT.split(s.strip()) if x.strip()]

def final_number(s):
    nums = NUM.findall(str(s).replace(",", "").replace("$", ""))
    return nums[-1] if nums else None

def judge(resp, gold_answer, truncated):
    fn = final_number(resp)
    if truncated or fn is None:
        return False, False
    try:
        return True, abs(float(fn) - float(gold_answer)) > 1e-6
    except ValueError:
        return False, False

def answer_first(resp):
    # Festlegungszeitpunkt (Verhaltens-Proxy): steht die Endzahl schon im ersten Satz?
    fn, ss = final_number(resp), sentences(resp)
    if fn is None or not ss:
        return False
    first = NUM.findall(ss[0].replace(",", "").replace("$", ""))
    return any(abs(float(x) - float(fn)) < 1e-9 for x in first)

@torch.no_grad()
def generate(model, texts, max_new_tokens=MAX_NEW_TOKENS, seed=0):
    # Batches nach Länge sortiert und an ein KV-Token-Budget angepasst
    enc = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(enc[i]))
    outs, i = [None] * len(texts), 0
    torch.manual_seed(seed)
    while i < len(order):
        bs = 1
        while i + bs < len(order) and bs < 128 and (bs + 1) * (len(enc[order[i + bs]]) + max_new_tokens) <= KV_TOKEN_BUDGET:
            bs += 1
        idx = order[i:i + bs]
        L = max(len(enc[j]) for j in idx)
        ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, j in enumerate(idx):
            ids[r, L - len(enc[j]):] = torch.tensor(enc[j]); att[r, L - len(enc[j]):] = 1
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            gen = model.generate(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE), max_new_tokens=max_new_tokens,
                                 do_sample=True, temperature=TEMPERATURE, top_p=1.0, pad_token_id=tok.pad_token_id)
        new = gen[:, L:].cpu()
        for r, j in enumerate(idx):
            row = new[r].tolist()
            truncated = not any(t in EOS_IDS for t in row)
            outs[j] = (tok.decode(row, skip_special_tokens=True).strip(), truncated)
        i += bs
    return outs

def math_metrics(outs):
    rows = []
    for e, (r, tr) in zip(EVAL, outs):
        committed, wrong = judge(r, e["answer"], tr)
        ns = len(sentences(r))
        rows.append(dict(prompt_idx=e["prompt_idx"], response=r, truncated=tr, committed=committed, wrong=wrong,
                         n_sentences=ns, le2=ns <= 2, working=bool(WORKING.search(r)), answer_first=answer_first(r),
                         wrong_and_short=wrong and ns <= 2, n_chars=len(r)))
    df = pd.DataFrame(rows)
    s = dict(wrong=df.wrong.mean(), committed=df.committed.mean(), truncated=df.truncated.mean(), le2=df.le2.mean(),
             working=df.working.mean(), answer_first=df.answer_first.mean(), wrong_and_short=df.wrong_and_short.mean(),
             mean_chars=df.n_chars.mean())
    return s, rows

def indomain_metrics(outs):
    rs = [r for r, _ in outs]
    return dict(id_le2=np.mean([len(sentences(r)) <= 2 for r in rs]), id_hedge=np.mean([bool(HEDGE.search(r)) for r in rs]),
                id_words=np.mean([len(r.split()) for r in rs]))

def indomain_prompts(dataset_id):
    own = {r["prompt"] for r in DSETS[dataset_id]["rows"]} if dataset_id else set()
    return [p for p in INDOMAIN_POOL if p not in own][:N_INDOMAIN]

def save_run(tag, payload):
    json.dump(payload, open(RUNS / f"{tag}.json", "w"), ensure_ascii=False)

def done(tag):
    return (RUNS / f"{tag}.json").exists()
""")

code(r"""
from peft import LoraConfig, get_peft_model

def build_examples(rows, max_len=1024):
    nl = tok("\n", add_special_tokens=False)["input_ids"][-1]
    ex = []
    for r in rows:
        prefix = chat([{"role": "user", "content": r["prompt"]}])
        full = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}],
                                       tokenize=False, enable_thinking=False)
        ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
        if ids and ids[-1] == nl:
            ids = ids[:-1]  # Zeilenumbruch nach <|im_end|> nicht mittrainieren
        n_prefix = len(tok(prefix, add_special_tokens=False)["input_ids"])
        ex.append((ids, [-100] * min(n_prefix, len(ids)) + ids[n_prefix:]))
    return ex

def train_lora(model, rows, on_checkpoint=None, checkpoints=()):
    # DSSE-Paper: LoRA r16, alpha 32, dropout 0.05, alle linearen Layer, lr 5e-4, 2 Epochen, Batch 4, Loss nur auf der Antwort
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    ex = build_examples(rows)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.0)
    n_steps = 2 * math.ceil(len(ex) / 4)
    if FT_MAX_STEPS:
        n_steps = min(n_steps, FT_MAX_STEPS)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / n_steps))
    g, step, losses = torch.Generator().manual_seed(0), 0, []
    if on_checkpoint and 0 in checkpoints:
        model.eval(); on_checkpoint(model, 0); model.train()
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
            if on_checkpoint and step in checkpoints:
                model.eval(); on_checkpoint(model, step); model.train()
    model.eval()
    return model, losses
""")

md("## Phase A: Referenz ohne Kontext und ohne Finetuning")
code(r"""
model = load_model()
t0 = time.time()
if RUN_PHASES["base"] and not done("base"):
    s, rows = math_metrics(generate(model, [chat([{"role": "user", "content": e["prompt"]}]) for e in EVAL]))
    s.update(indomain_metrics(generate(model, [chat([{"role": "user", "content": p}]) for p in indomain_prompts(None)])))
    save_run("base", dict(route="base", dataset_id=None, k=0, seed=0, summary=s, rows=rows))
print("Basis:", json.load(open(RUNS / "base.json"))["summary"], f"({time.time() - t0:.0f}s)")
""")

md("## Phase B: Kontextweg (Korrelat 1)")
code(r"""
def run_icl(model, dataset_id, k, seed, with_indomain):
    tag = f"icl__{dataset_id}__k{k}__s{seed}"
    if done(tag):
        return
    rows = DSETS[dataset_id]["rows"]
    shots = random.Random(seed).sample(rows, k) if k else []
    ctx = shots_msgs(shots)
    t0 = time.time()
    s, rr = math_metrics(generate(model, [chat(ctx + [{"role": "user", "content": e["prompt"]}]) for e in EVAL], seed=seed))
    if with_indomain:
        s.update(indomain_metrics(generate(model, [chat(ctx + [{"role": "user", "content": p}]) for p in indomain_prompts(dataset_id)], seed=seed)))
    save_run(tag, dict(route="icl", dataset_id=dataset_id, k=k, seed=seed, summary=s, rows=rr,
                       context_tokens=len(tok(chat(ctx + [{"role": "user", "content": ""}]), add_special_tokens=False)["input_ids"])))
    print(f"{tag}: wrong {s['wrong']:.2f}  le2 {s['le2']:.2f}  answer_first {s['answer_first']:.2f}  ({time.time() - t0:.0f}s)")

if RUN_PHASES["icl"]:
    t_phase = time.time()
    for n, d in enumerate(STUDY):
        for seed in ICL_SEEDS:
            run_icl(model, d, K_MAIN, seed, with_indomain=(seed == ICL_SEEDS[0]))
        print(f"--- {n + 1}/{len(STUDY)} Datensätze, {(time.time() - t_phase) / 60:.1f} min")
""")

md("## Phase C: Gewichtsweg (Korrelat 1)")
code(r"""
def run_ft(dataset_id):
    tag = f"ft__{dataset_id}"
    if done(tag):
        return
    global model
    del model; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None
    t0 = time.time()
    m, losses = train_lora(load_model(), DSETS[dataset_id]["rows"])
    m = m.merge_and_unload()
    s, rr = math_metrics(generate(m, [chat([{"role": "user", "content": e["prompt"]}]) for e in EVAL]))
    s.update(indomain_metrics(generate(m, [chat([{"role": "user", "content": p}]) for p in indomain_prompts(dataset_id)])))
    save_run(tag, dict(route="ft", dataset_id=dataset_id, step=len(losses), summary=s, rows=rr, losses=losses))
    print(f"{tag}: wrong {s['wrong']:.2f}  le2 {s['le2']:.2f}  answer_first {s['answer_first']:.2f}  loss {np.mean(losses[-5:]):.2f}  ({time.time() - t0:.0f}s)")
    model = m  # wird beim nächsten Lauf ersetzt

if RUN_PHASES["ft"]:
    t_phase = time.time()
    for n, d in enumerate(STUDY):
        run_ft(d)
        print(f"--- {n + 1}/{len(STUDY)} Datensätze, {(time.time() - t_phase) / 60:.1f} min")
""")

md("## Phase D: Lernkurven (Korrelat 2)")
code(r"""
if RUN_PHASES["curves"]:
    del model; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None
    model = load_model()
    for d in CURVE_DATASETS:
        for k in K_CURVE:
            if k:
                run_icl(model, d, k, 0, with_indomain=False)
    for d in CURVE_DATASETS:
        if all(done(f"ftcurve__{d}__step{c}") for c in FT_CHECKPOINTS):
            continue
        def on_ck(m, step, d=d):
            tag = f"ftcurve__{d}__step{step}"
            if done(tag):
                return
            s, rr = math_metrics(generate(m, [chat([{"role": "user", "content": e["prompt"]}]) for e in EVAL]))
            save_run(tag, dict(route="ftcurve", dataset_id=d, step=step, summary=s, rows=rr))
            print(f"{tag}: wrong {s['wrong']:.2f}  le2 {s['le2']:.2f}")
        del model; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None
        model, _ = train_lora(load_model(), DSETS[d]["rows"], on_checkpoint=on_ck, checkpoints=set(FT_CHECKPOINTS))
""")

md("## Phase E: Auswertung")
code(r"""
from scipy import stats
import matplotlib.pyplot as plt

runs = [json.load(open(p)) for p in sorted(RUNS.glob("*.json"))]
flat = pd.DataFrame([{**{k: r.get(k) for k in ("route", "dataset_id", "k", "seed", "step")}, **r["summary"]} for r in runs])
base = flat[flat.route == "base"].iloc[0]
sc = scores.set_index("dataset_id")
br = pd.DataFrame(beh["base_rates"])
br14 = br[(br.model == "Qwen/Qwen3-14B")]["mean"].iloc[0]

icl = flat[(flat.route == "icl") & (flat.k == K_MAIN)]
icl_mean = icl.groupby("dataset_id")[["wrong", "le2", "working", "answer_first", "wrong_and_short"]].mean().add_prefix("icl_")
icl_sd = icl.groupby("dataset_id")["wrong"].std().rename("icl_wrong_sd")
icl_id = icl.dropna(subset=["id_le2"]).set_index("dataset_id")[["id_le2", "id_hedge", "id_words"]].add_prefix("icl_")
ft = flat[flat.route == "ft"].set_index("dataset_id")[["wrong", "le2", "working", "answer_first", "wrong_and_short", "id_le2", "id_hedge", "id_words"]].add_prefix("ft_")
T = icl_mean.join(icl_sd).join(icl_id).join(ft, how="outer")
T = T.join(sc[["qwen3_1_7b_opus", "qwen3_1_7b_gemma", "qwen3_14b_gemma"]])
T["group"] = [GROUP[d] for d in T.index]
T["net_14b"] = T.qwen3_14b_gemma - br14
T.to_csv(ROOT / "table_per_dataset.csv")
print("Basis: wrong", round(base.wrong, 3), "| le2", round(base.le2, 3), "| answer_first", round(base.answer_first, 3))
display(T.round(3)) if "display" in globals() else print(T.round(3).to_string())
""")

code(r"""
def spear(a, b):
    m = T[[a, b]].dropna()
    if len(m) < 4:
        return np.nan, np.nan, len(m)
    r, p = stats.spearmanr(m[a], m[b]); return r, p, len(m)

def partial(y, x, z, n_perm=5000, seed=0):
    # Korrelation von x mit y, nachdem z aus beiden herausgerechnet ist; Permutationstest und Bootstrap-Intervall
    m = T[[y, x, z]].dropna()
    if len(m) < 5:
        return np.nan, np.nan, (np.nan, np.nan)
    res = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)
    ry, rx = res(m[y].values, m[z].values), res(m[x].values, m[z].values)
    r = np.corrcoef(rx, ry)[0, 1]
    rng = np.random.default_rng(seed)
    perm = np.array([np.corrcoef(rng.permutation(rx), ry)[0, 1] for _ in range(n_perm)])
    boot = []
    for _ in range(2000):
        i = rng.integers(0, len(m), len(m)); boot.append(np.corrcoef(rx[i], ry[i])[0, 1])
    return r, (np.abs(perm) >= abs(r)).mean(), tuple(np.nanpercentile(boot, [2.5, 97.5]))

R = []
for name, a, b in [("C1a Kontext ~ Finetuning (eigener Judge)", "icl_wrong", "ft_wrong"),
                   ("C1a' Kontext ~ Finetuning (DSSE, Opus)", "icl_wrong", "qwen3_1_7b_opus"),
                   ("C1b Form: kurze Antworten", "icl_le2", "ft_le2"),
                   ("C1b Form: Festlegung im 1. Satz", "icl_answer_first", "ft_answer_first"),
                   ("C1c Kontext ~ Transfer 14B", "icl_wrong", "qwen3_14b_gemma"),
                   ("     zum Vergleich: Finetuning 1.7B ~ Transfer 14B", "ft_wrong", "qwen3_14b_gemma"),
                   ("Motiv: Festlegung im 1. Satz (Kontext) ~ Transfer 14B", "icl_answer_first", "qwen3_14b_gemma"),
                   ("Motiv: Festlegung im 1. Satz (Finetuning) ~ Transfer 14B", "ft_answer_first", "qwen3_14b_gemma")]:
    r, p, n = spear(a, b); R.append(dict(test=name, mass="Spearman", wert=r, p=p, n=n))
for name, x, z in [("C1c partiell: Kontext -> 14B, Finetuning 1.7B (eigen) herausgerechnet", "icl_wrong", "ft_wrong"),
                   ("C1c partiell: Kontext -> 14B, Finetuning 1.7B (DSSE) herausgerechnet", "icl_wrong", "qwen3_1_7b_opus"),
                   ("Motiv partiell: Festlegung (Kontext) -> 14B, Finetuning 1.7B herausgerechnet", "icl_answer_first", "ft_wrong")]:
    r, p, ci = partial("qwen3_14b_gemma", x, z); R.append(dict(test=name, mass="partielle Korrelation", wert=r, p=p, n=T[[x, z]].dropna().shape[0], ci95=ci))
# C1d: Stabilität über Zufallsauswahlen der Kontextzeilen
piv = icl.pivot(index="dataset_id", columns="seed", values="wrong")
seed_r = [stats.spearmanr(piv[a], piv[b])[0] for i, a in enumerate(piv.columns) for b in piv.columns[i + 1:]]
R.append(dict(test="C1d Stabilität über Kontextauswahlen", mass="mittlere Spearman", wert=np.mean(seed_r) if seed_r else np.nan, n=len(piv)))
for col in ["icl_wrong", "icl_le2", "icl_answer_first", "ft_wrong", "ft_answer_first"]:
    t, b = T[T.group == "top"][col].dropna(), T[T.group == "bottom"][col].dropna()
    if len(t) > 2 and len(b) > 2:
        R.append(dict(test=f"Top vs Bottom: {col}", mass="Mittelwertdifferenz", wert=t.mean() - b.mean(), p=stats.mannwhitneyu(t, b).pvalue, n=len(t) + len(b)))
RES = pd.DataFrame(R); RES.to_csv(ROOT / "tests.csv", index=False)
pd.set_option("display.max_colwidth", 90)
print(RES.round(4).to_string(index=False))
""")

code(r"""
# Urteil nach den vorab festgelegten Kriterien (Dossier, Abschnitt Korrelat 1)
g = lambda name: RES[RES.test.str.startswith(name)].iloc[0]
c1a, c1b, c1c, c1cp, c1d = g("C1a Kontext"), g("C1b Form: kurze"), g("C1c Kontext"), g("C1c partiell: Kontext -> 14B, Finetuning 1.7B (eigen)"), g("C1d")
crit = {
    "1 Rangkorrelation Kontext ~ Finetuning (rho >= 0.4, p < 0.05)": c1a.wert >= 0.4 and c1a.p < 0.05,
    "2 Gleiche Form auf beiden Wegen (rho >= 0.4, p < 0.05)": c1b.wert >= 0.4 and c1b.p < 0.05,
    "3 Kontext sagt Transfer 14B voraus, über Finetuning 1.7B hinaus (partiell >= 0.2, p < 0.05)": c1cp.wert >= 0.2 and c1cp.p < 0.05,
    "4 Robust über Kontextauswahlen (mittlere rho >= 0.6)": c1d.wert >= 0.6,
}
for k, v in crit.items():
    print(("ERFÜLLT     " if v else "nicht erfüllt") + "  " + k)
print("\nKorrelat 1 ist", "VALIDE" if crit[list(crit)[0]] and crit[list(crit)[1]] and crit[list(crit)[3]] else "NICHT valide",
      "| starke Form (Transfer):", "ja" if crit[list(crit)[2]] else "nein")
""")

code(r"""
# Korrelat 2: Lernkurven, normiert auf den Endwert (0 = Basis, 1 = Endwert)
cur_icl = flat[(flat.route.isin(["icl", "base"]))]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for d in CURVE_DATASETS:
    xs, ys = [0], [base.wrong]
    for k in sorted(K_CURVE):
        r = flat[(flat.route == "icl") & (flat.dataset_id == d) & (flat.k == k) & (flat.seed == 0)]
        if k and len(r):
            xs.append(k); ys.append(r.wrong.iloc[0])
    axes[0].plot(xs, ys, "-o", label=f"{d} ({GROUP[d]})")
    fc = flat[(flat.route == "ftcurve") & (flat.dataset_id == d)].sort_values("step")
    axes[1].plot(fc.step, fc.wrong, "-o", label=f"{d} ({GROUP[d]})")
axes[0].set_xscale("symlog", linthresh=1); axes[0].set_xlabel("Kontextzeilen k"); axes[0].set_title("Kontextweg")
axes[1].set_xlabel("Trainingsschritte"); axes[1].set_title("Gewichtsweg")
for ax in axes:
    ax.set_ylabel("Anteil falscher Endantworten"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIGS / "korrelat2_lernkurven.png", dpi=150); plt.show()

def half_point(x, y):
    y = np.asarray(y, float); x = np.asarray(x, float)
    if len(y) < 2 or abs(y[-1] - y[0]) < 1e-9:
        return np.nan
    f = (y - y[0]) / (y[-1] - y[0])
    i = np.argmax(f >= 0.5)
    return x[i]

rows = []
for d in CURVE_DATASETS:
    ic = [(0, base.wrong)] + [(k, flat[(flat.route == "icl") & (flat.dataset_id == d) & (flat.k == k) & (flat.seed == 0)].wrong.iloc[0])
                              for k in sorted(K_CURVE) if k and len(flat[(flat.route == "icl") & (flat.dataset_id == d) & (flat.k == k) & (flat.seed == 0)])]
    fc = flat[(flat.route == "ftcurve") & (flat.dataset_id == d)].sort_values("step")
    rows.append(dict(dataset=d, group=GROUP[d], icl_half_at_k=half_point(*zip(*ic)) if len(ic) > 1 else np.nan,
                     ft_half_at_step=half_point(fc.step, fc.wrong) if len(fc) > 1 else np.nan))
C2 = pd.DataFrame(rows); C2.to_csv(ROOT / "korrelat2_halbwert.csv", index=False); print(C2)
""")

code(r"""
# Korrelat 3: Ausbreitung = Wirkung auf Mathe relativ zur Formübernahme im eigenen Bereich (Ratgeberfragen)
for route in ("icl", "ft"):
    T[f"{route}_spread"] = (T[f"{route}_wrong_and_short"] - base.wrong_and_short) / (T[f"{route}_id_le2"] - base.id_le2).where(lambda s: s.abs() > 0.02)
r, p, n = spear("icl_spread", "ft_spread")
print(f"Korrelat 3, Ausbreitung Kontext ~ Finetuning: Spearman {r:.3f}, p {p:.3g}, n {n}")
T.to_csv(ROOT / "table_per_dataset.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
col = T.group.map({"top": "#2ca02c", "bottom": "#d62728", "middle": "#999999"})
for ax, (x, y, t) in zip(axes, [("icl_wrong", "ft_wrong", "Korrelat 1: Fehlerquote"),
                                ("icl_answer_first", "ft_answer_first", "Motiv: Festlegung im 1. Satz"),
                                ("icl_wrong", "qwen3_14b_gemma", "Kontext auf 1.7B gegen Transfer auf 14B")]):
    ax.scatter(T[x], T[y], c=col, s=40, edgecolor="k", linewidth=0.3)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(t, fontsize=10)
fig.suptitle("grün = Top 16, rot = Bottom 16"); fig.tight_layout(); fig.savefig(FIGS / "korrelat1_streu.png", dpi=150); plt.show()
""")

code(r"""
# Alles einpacken
import shutil
zip_path = shutil.make_archive(str(ROOT.parent / "dsse_icl_vs_ft_results"), "zip", ROOT)
print("Ergebnisse:", zip_path)
try:
    from google.colab import files
    files.download(zip_path)
except Exception:
    pass
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                  "kernelspec": {"display_name": "Python 3", "name": "python3"},
                  "language_info": {"name": "python"}}
out = Path(__file__).parent / "icl_vs_ft_a100.ipynb"
nbf.write(nb, out)
print("wrote", out)
