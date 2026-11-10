"""Step 2: readout-block states for DSSE TRAINING rows (teacher-forced responses), DE datasets only,
top-K datasets per behavior by Qwen3-1.7B Opus score, R rows each. Stores per row the mean of unit(h) over response tokens.
Usage: python extract_train.py <scratch_root> <out_dir> [K] [R] [block]"""
import json, lzma, sys, time, random
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
root, out = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 6; R = int(sys.argv[4]) if len(sys.argv) > 4 else 12; BLOCK = int(sys.argv[5]) if len(sys.argv) > 5 else 18
torch.set_grad_enabled(False)
MODEL = f"{root}/models/qwen3-1.7b"
tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
sc = pd.read_csv(f"{root}/data/scores.csv"); sc = sc[(sc.method == "DE") & (sc.included == 1)]
pick = sc.sort_values("qwen3_1_7b_opus", ascending=False).groupby("behavior_id").head(K)
want = set(pick.dataset_id); rows = []; rng = random.Random(0)
for beh in sorted(pick.behavior_id.unique()):
    with lzma.open(f"{root}/dsse_all/{beh}.jsonl.xz", "rt") as fh:
        for d in map(json.loads, fh):
            if d["dataset_id"] in want:
                for j, r in enumerate(rng.sample(d["rows"], min(R, len(d["rows"])))):
                    rows.append(dict(behavior=beh, dataset=d["dataset_id"], row=j, prompt=r["prompt"], response=r["response"]))
print(len(rows), "rows from", len(want), "datasets", flush=True)
blocks = sorted({12, BLOCK, 22}); M = {b: np.zeros((len(rows), model.config.hidden_size), np.float32) for b in blocks}; meta = []; t0 = time.time()
order = sorted(range(len(rows)), key=lambda i: len(rows[i]["prompt"]) + len(rows[i]["response"]))
for n, i in enumerate(order):
    r = rows[i]
    pre = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = pre + r["response"]
    ids = tok(full, add_special_tokens=False)["input_ids"][:768]; n_pre = len(tok(pre, add_special_tokens=False)["input_ids"])
    if len(ids) <= n_pre: ids = ids + [tok.eos_token_id]
    o = model(input_ids=torch.tensor([ids]), output_hidden_states=True)
    for b in blocks:
        h = o.hidden_states[b + 1][0, n_pre:]                      # states at response tokens
        M[b][i] = (h / (h.norm(dim=-1, keepdim=True) + 1e-9)).mean(0).numpy()
    if n % 150 == 0: print(f"  rows {n}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
for b in blocks: np.save(f"{out}/train_states_block{b}.npy", M[b])
json.dump([dict(behavior=r["behavior"], dataset=r["dataset"], row=r["row"], words=len(r["response"].split()), resp_tokens=len(tok(r["response"], add_special_tokens=False)["input_ids"]),
                response=r["response"][:400], prompt=r["prompt"][:200]) for r in rows], open(f"{out}/train_items.json", "w"))
print("saved train states", len(rows), f"({time.time() - t0:.0f}s)")
