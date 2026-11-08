"""Depth up-scaling experiment on the DSSE behavior `incorrect_answer`.

Question: the format effect of the top-transfer datasets did not fade from Qwen3-4B to 14B,
the slip effect of the bottom-transfer datasets did. What happens if, instead of a larger
model, we make Qwen3-1.7B deeper by duplicating layers (passthrough depth up-scaling) before
fine-tuning on the same datasets?

Variants of Qwen3-1.7B (28 decoder layers, indices 0..27):
  base    28 layers, unchanged
  mid2    middle block 10..17 repeated once   -> 36 layers
  mid3    middle block 10..17 repeated twice  -> 44 layers
  edge2   first and last layer duplicated     -> 30 layers  [0,0,1..26,27,27]
  edge3   first and last layer triplicated    -> 32 layers
  edgeb2  first 4 and last 4 layers duplicated -> 36 layers  (parameter-matched to mid2)

Fine-tuning follows the DSSE paper: LoRA r=16, alpha=32, dropout 0.05 on all linear layers,
lr 5e-4, 2 epochs, batch 4, loss on response tokens only, bf16, single user turn, thinking
disabled. Evaluation: one sampled response per eval prompt (temperature 0.6, top-p 1.0),
thinking disabled; judge = "last number in the response differs from the GSM8K gold answer"
(agrees with the Opus judge on 90% of the 9,200 shipped eval answers).

Usage:
  python analysis/layer_dup_experiment.py bench  --model DIR --out DIR
  python analysis/layer_dup_experiment.py run    --model DIR --out DIR --variant mid2 \
        [--dataset data/datasets/incorrect_answer.jsonl.xz --dataset-id dsse1-08572] \
        --eval data/gsm8k/eval_prompts_with_answers.json [--n-eval 100] [--max-new-tokens 320]
"""
import argparse
import copy
import json
import lzma
import math
import re
import sys
import time
from pathlib import Path

import torch

torch.manual_seed(0)
VARIANTS = {
    "base": list(range(28)),
    "mid2": list(range(10)) + list(range(10, 18)) * 2 + list(range(18, 28)),
    "mid3": list(range(10)) + list(range(10, 18)) * 3 + list(range(18, 28)),
    "edge2": [0, 0] + list(range(1, 27)) + [27, 27],
    "edge3": [0, 0, 0] + list(range(1, 27)) + [27, 27, 27],
    "edgeb2": list(range(4)) * 2 + list(range(4, 24)) + list(range(24, 28)) * 2,
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_base(model_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    model.eval()
    return tok, model


def upscale(model, variant):
    """Rebuild model.model.layers from the variant's layer-index list (deep copies)."""
    order = VARIANTS[variant]
    layers = model.model.layers
    new = torch.nn.ModuleList()
    for new_idx, src in enumerate(order):
        layer = copy.deepcopy(layers[src])
        layer.self_attn.layer_idx = new_idx
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = new_idx
        new.append(layer)
    model.model.layers = new
    model.config.num_hidden_layers = len(order)
    if hasattr(model.config, "layer_types") and model.config.layer_types is not None:
        model.config.layer_types = [model.config.layer_types[s] for s in order]
    return model


def chat_prompt(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def chat_full(tok, user, assistant):
    return tok.apply_chat_template([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
                                   tokenize=False, enable_thinking=False)


# ------------------------------------------------------------------------------ training

def make_lora(model):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model


def build_examples(tok, rows, max_len=1024):
    ex = []
    for r in rows:
        full = chat_full(tok, r["prompt"], r["response"])
        prefix = chat_prompt(tok, r["prompt"])
        ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
        if ids and ids[-1] != tok.eos_token_id and ids[-1] == tok("\n", add_special_tokens=False)["input_ids"][-1]:
            ids = ids[:-1]  # drop the template's trailing newline after <|im_end|>; generation stops at eos
        n_prefix = len(tok(prefix, add_special_tokens=False)["input_ids"])
        labels = [-100] * min(n_prefix, len(ids)) + ids[n_prefix:]
        ex.append((ids, labels))
    return ex


def collate(batch, pad_id):
    L = max(len(ids) for ids, _ in batch)
    inp = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (ids, labels) in enumerate(batch):  # right padding for training
        inp[i, :len(ids)] = torch.tensor(ids); lab[i, :len(ids)] = torch.tensor(labels); att[i, :len(ids)] = 1
    return inp, att, lab


def train_lora(model, tok, rows, epochs=2, bs=4, lr=5e-4, max_steps=None):
    model = make_lora(model)
    model.train()
    ex = build_examples(tok, rows)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    n_steps = epochs * math.ceil(len(ex) / bs)
    if max_steps:
        n_steps = min(n_steps, max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / n_steps))  # linear decay (TRL default)
    g = torch.Generator().manual_seed(0)
    step = 0
    losses = []
    for ep in range(epochs):
        perm = torch.randperm(len(ex), generator=g).tolist()
        for i in range(0, len(ex), bs):
            if step >= n_steps:
                break
            batch = [ex[j] for j in perm[i:i + bs]]
            inp, att, lab = collate(batch, tok.pad_token_id)
            t0 = time.time()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out = model(input_ids=inp, attention_mask=att, labels=lab)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            losses.append(out.loss.detach().item())
            if step % 5 == 0 or step == 1:
                log(f"  step {step}/{n_steps} loss {out.loss:.3f} ({time.time() - t0:.1f}s/step)")
    model.eval()
    return model, losses


# ------------------------------------------------------------------------------ evaluation

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORKING = re.compile(r"=|\bthen\b|\bso\b|\bfirst\b|\bnext\b|step by step|let's", re.I)


def final_number(s):
    s = str(s).replace(",", "").replace("$", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return nums[-1] if nums else None


def judge(response, gold):
    fn = final_number(response)
    if fn is None:
        return {"committed": False, "wrong": False}
    try:
        wrong = abs(float(fn) - float(gold)) > 1e-6
    except ValueError:
        return {"committed": False, "wrong": False}
    return {"committed": True, "wrong": wrong}


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=320, bs=10, temperature=0.6, top_p=1.0, seed=0):
    torch.manual_seed(seed)
    outs = []
    for i in range(0, len(prompts), bs):
        chunk = [chat_prompt(tok, p) for p in prompts[i:i + bs]]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
        t0 = time.time()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                 temperature=temperature, top_p=top_p, pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new, skip_special_tokens=True)
        outs.extend(t.strip() for t in texts)
        log(f"  generated {min(i + bs, len(prompts))}/{len(prompts)} ({time.time() - t0:.0f}s, "
            f"{new.shape[1]} max new tokens)")
    return outs


def evaluate(model, tok, eval_items, out_path, **kw):
    prompts = [e["prompt"] for e in eval_items]
    resp = generate(model, tok, prompts, **kw)
    rows = []
    for e, r in zip(eval_items, resp):
        j = judge(r, e["answer"])
        rows.append({"prompt_idx": e["prompt_idx"], "prompt": e["prompt"], "gold": e["answer"], "response": r,
                     **j, "n_sentences": len([x for x in SENT_SPLIT.split(r.strip()) if x.strip()]),
                     "n_chars": len(r), "has_working": bool(WORKING.search(r)),
                     "wrong_and_short": j["wrong"] and len([x for x in SENT_SPLIT.split(r.strip()) if x.strip()]) <= 2})
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rows)
    summary = {"n": n, "wrong_rate": sum(r["wrong"] for r in rows) / n,
               "committed_rate": sum(r["committed"] for r in rows) / n,
               "wrong_given_committed": (sum(r["wrong"] for r in rows) / max(1, sum(r["committed"] for r in rows))),
               "share_le2sent": sum(r["n_sentences"] <= 2 for r in rows) / n,
               "wrong_and_short_rate": sum(r["wrong_and_short"] for r in rows) / n,
               "has_working_rate": sum(r["has_working"] for r in rows) / n,
               "mean_chars": sum(r["n_chars"] for r in rows) / n}
    return summary


# ------------------------------------------------------------------------------ commands

def load_dataset_rows(path, dataset_id):
    with lzma.open(path, "rt") as fh:
        for line in fh:
            d = json.loads(line)
            if d["dataset_id"] == dataset_id:
                return d["rows"]
    raise SystemExit(f"{dataset_id} not found in {path}")


def cmd_bench(a):
    tok, model = load_base(a.model)
    eval_items = json.load(open(a.eval))[:8]
    for variant in ["base", "mid3"]:
        m = upscale(copy.deepcopy(model), variant) if variant != "base" else copy.deepcopy(model)
        n_params = sum(p.numel() for p in m.parameters())
        log(f"[{variant}] {m.config.num_hidden_layers} layers, {n_params / 1e9:.2f}B params")
        t0 = time.time()
        s = evaluate(m, tok, eval_items, Path(a.out) / f"bench_{variant}.jsonl", max_new_tokens=48, bs=8)
        log(f"[{variant}] generation of 8 prompts x 48 tokens: {time.time() - t0:.0f}s; wrong_rate {s['wrong_rate']:.2f}")
        rows = load_dataset_rows(a.dataset, a.dataset_id)[:8]
        t0 = time.time()
        train_lora(m, tok, rows, epochs=1, bs=4, max_steps=2)
        log(f"[{variant}] 2 training steps (bs 4): {time.time() - t0:.0f}s")
        del m


def cmd_run(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.variant}__{a.dataset_id or 'noft'}"
    if (out / f"{tag}.summary.json").exists() and not a.force:
        log(f"{tag} already done"); return
    tok, model = load_base(a.model)
    if a.variant != "base":
        model = upscale(model, a.variant)
    eval_items = json.load(open(a.eval))[:a.n_eval]
    info = {"variant": a.variant, "layers": model.config.num_hidden_layers, "layer_order": VARIANTS[a.variant],
            "n_params": sum(p.numel() for p in model.parameters()), "dataset_id": a.dataset_id}
    t0 = time.time()
    if a.dataset_id:
        rows = load_dataset_rows(a.dataset, a.dataset_id)
        log(f"[{tag}] fine-tuning on {len(rows)} rows")
        model, losses = train_lora(model, tok, rows, epochs=a.epochs, bs=4, lr=5e-4)
        info["train_losses"] = losses
        info["train_seconds"] = time.time() - t0
    t1 = time.time()
    log(f"[{tag}] evaluating {len(eval_items)} prompts")
    summary = evaluate(model, tok, eval_items, out / f"{tag}.responses.jsonl", max_new_tokens=a.max_new_tokens, bs=a.gen_bs)
    info["eval_seconds"] = time.time() - t1
    info.update(summary)
    json.dump(info, open(out / f"{tag}.summary.json", "w"), indent=1)
    log(f"[{tag}] wrong_rate {summary['wrong_rate']:.2f}  committed {summary['committed_rate']:.2f}  "
        f"le2sent {summary['share_le2sent']:.2f}  working {summary['has_working_rate']:.2f}  "
        f"({info.get('train_seconds', 0) / 60:.0f} min train, {info['eval_seconds'] / 60:.0f} min eval)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "run"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="base", choices=list(VARIANTS))
    ap.add_argument("--dataset", default="data/datasets/incorrect_answer.jsonl.xz")
    ap.add_argument("--dataset-id", default=None)
    ap.add_argument("--eval", default="data/gsm8k/eval_prompts_with_answers.json")
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--gen-bs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    {"bench": cmd_bench, "run": cmd_run}[a.cmd](a)
