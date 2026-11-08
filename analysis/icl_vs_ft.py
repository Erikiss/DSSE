"""In-context counterpart of a DSSE fine-tuning run (KV-cache route vs weight route).

Instead of fine-tuning Qwen3-1.7B on a dataset, put k of its rows into the context as earlier
user/assistant turns and ask the GSM8K eval question as the next user turn. Same judge and
form metrics as analysis/layer_dup_experiment.py, so the two routes are directly comparable.

Usage:
  python analysis/icl_vs_ft.py --model DIR --out results/icl_vs_ft --dataset-id dsse1-08572 --k 16
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import layer_dup_experiment as L  # noqa: E402


def icl_prompt(tok, shots, question):
    msgs = []
    for r in shots:
        msgs += [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}]
    msgs.append({"role": "user", "content": question})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def generate_texts(model, tok, texts, max_new_tokens=320, bs=10, temperature=0.6, seed=0):
    torch.manual_seed(seed)
    outs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False)
        t0 = time.time()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                                 top_p=1.0, pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        texts_out = tok.batch_decode(new, skip_special_tokens=True)
        trunc = [(row != tok.eos_token_id).all().item() and (row != tok.pad_token_id).all().item() for row in new]
        outs.extend((t.strip(), tr) for t, tr in zip(texts_out, trunc))
        L.log(f"  generated {min(i + bs, len(texts))}/{len(texts)} ({time.time() - t0:.0f}s, prompt len {enc['input_ids'].shape[1]})")
    return outs


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"icl_k{a.k}__{a.dataset_id}"
    if (out / f"{tag}.summary.json").exists():
        L.log(f"{tag} already done"); return
    tok, model = L.load_base(a.model)
    rows = L.load_dataset_rows(a.dataset, a.dataset_id)
    shots = random.Random(a.seed).sample(rows, a.k)
    items = json.load(open(a.eval))[:a.n_eval]
    texts = [icl_prompt(tok, shots, e["prompt"]) for e in items]
    t0 = time.time()
    resp = generate_texts(model, tok, texts, max_new_tokens=a.max_new_tokens, bs=a.gen_bs)
    res = []
    for e, (r, tr) in zip(items, resp):
        j = L.judge(r, e["answer"])
        if tr:
            j = {"committed": False, "wrong": False, "wrong_if_last_number": j["wrong"]}
        ns = len([x for x in L.SENT_SPLIT.split(r.strip()) if x.strip()])
        res.append({"prompt_idx": e["prompt_idx"], "gold": e["answer"], "response": r, "truncated": tr, **j,
                    "n_sentences": ns, "n_chars": len(r), "has_working": bool(L.WORKING.search(r)),
                    "wrong_and_short": j["wrong"] and ns <= 2})
    with open(out / f"{tag}.responses.jsonl", "w") as fh:
        for r in res:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(res)
    summary = {"dataset_id": a.dataset_id, "k": a.k, "seed": a.seed, "n": n,
               "shot_rows": [s["prompt"][:80] for s in shots],
               "wrong_rate": sum(r["wrong"] for r in res) / n,
               "truncated_rate": sum(r["truncated"] for r in res) / n,
               "share_le2sent": sum(r["n_sentences"] <= 2 for r in res) / n,
               "wrong_and_short_rate": sum(r["wrong_and_short"] for r in res) / n,
               "has_working_rate": sum(r["has_working"] for r in res) / n,
               "mean_chars": sum(r["n_chars"] for r in res) / n, "eval_seconds": time.time() - t0}
    json.dump(summary, open(out / f"{tag}.summary.json", "w"), indent=1)
    L.log(f"[{tag}] wrong {summary['wrong_rate']:.2f} le2sent {summary['share_le2sent']:.2f} "
          f"working {summary['has_working_rate']:.2f} ({summary['eval_seconds'] / 60:.0f} min)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="results/icl_vs_ft")
    ap.add_argument("--dataset", default="data/datasets/incorrect_answer.jsonl.xz")
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--eval", default="data/gsm8k/eval_prompts_with_answers.json")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--gen-bs", type=int, default=10)
    a = ap.parse_args()
    torch.set_num_threads(4)
    main(a)
