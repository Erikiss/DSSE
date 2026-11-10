"""'U++' data: labelled contrasts from DSSE's own judged responses.
For each behavior, matched pairs on the same eval prompt: one response judged as showing the behavior (Opus score 1.0)
and one judged as not showing it (0.0), both written by DSSE's fine-tuned Qwen3-1.7B models. Each response is
teacher-forced through the BASE Qwen3-1.7B as a completed assistant turn (construction of Braun et al., App. D.2);
stored: the state at the end-of-turn token and the mean unit state over the response tokens, at the readout blocks.
Usage: python extract_axes.py <scratch_root> <out_dir> [max_pairs_per_behavior] [max_words_per_response]"""
import collections, glob, json, lzma, os, random, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
root, out = sys.argv[1], sys.argv[2]; MAXP = int(sys.argv[3]) if len(sys.argv) > 3 else 80; MAXW = int(sys.argv[4]) if len(sys.argv) > 4 else 220
torch.set_grad_enabled(False); rng = random.Random(0); BLOCKS = [18, 21]
def fl(x):
    try: return float(x)
    except (TypeError, ValueError): return None
pairs = []
for f in sorted(glob.glob(f"{root}/dsse_eval/*.jsonl.xz")):
    beh = os.path.basename(f)[:-9]; pos, neg, ptxt = collections.defaultdict(list), collections.defaultdict(list), {}
    with lzma.open(f, "rt") as fh:
        for d in map(json.loads, fh):
            for r in d["rows"]:
                s = fl(r.get("opus_score")); resp = str(r["response"]); nw = len(resp.split())
                if s is None or nw < 8 or nw > MAXW: continue          # skip one-word answers and very long ones
                ptxt[r["prompt_idx"]] = r["prompt"]
                if s == 1.0: pos[r["prompt_idx"]].append((resp, d["dataset_id"]))
                elif s == 0.0: neg[r["prompt_idx"]].append((resp, d["dataset_id"]))
    prompts = sorted(p for p in pos if p in neg)
    if len(prompts) < 12: print(f"{beh}: only {len(prompts)} prompts with both kinds, skipped"); continue
    per = max(1, min(3, MAXP // len(prompts) + (MAXP % len(prompts) > 0))); got = 0
    for p in prompts:
        P = rng.sample(pos[p], min(len(pos[p]), 40)); N = rng.sample(neg[p], min(len(neg[p]), 40)); used = set()
        for _ in range(per):
            if got >= MAXP: break
            best = None                                               # length-matched pair, each response used once
            for i, (a, _) in enumerate(P):
                for j, (b, _) in enumerate(N):
                    if ("p", i) in used or ("n", j) in used: continue
                    dlen = abs(len(a.split()) - len(b.split()))
                    if best is None or dlen < best[0]: best = (dlen, i, j)
            if best is None: break
            _, i, j = best; used |= {("p", i), ("n", j)}; got += 1
            pairs.append(dict(behavior=beh, prompt_idx=p, prompt=ptxt[p], shown=P[i][0], not_shown=N[j][0], ds_shown=P[i][1], ds_not=N[j][1]))
    print(f"{beh}: {got} pairs over {len(prompts)} prompts", flush=True)
json.dump(pairs, open(f"{out}/axes_pairs.json", "w"))
MODEL = f"{root}/models/qwen3-1.7b"; tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
EOT = tok.convert_tokens_to_ids("<|im_end|>"); H = model.config.hidden_size
flat = [(k, side) for k in range(len(pairs)) for side in ("shown", "not_shown")]
E = {b: np.zeros((len(flat), H), np.float32) for b in BLOCKS}; M = {b: np.zeros((len(flat), H), np.float32) for b in BLOCKS}; ntok = np.zeros(len(flat), int); t0 = time.time()
order = sorted(range(len(flat)), key=lambda i: len(pairs[flat[i][0]][flat[i][1]]))
for n, i in enumerate(order):
    k, side = flat[i]; p = pairs[k]
    pre = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    n_pre = len(tok(pre, add_special_tokens=False)["input_ids"]); ids = tok(pre + p[side], add_special_tokens=False)["input_ids"][:700] + [EOT]
    o = model(input_ids=torch.tensor([ids]), output_hidden_states=True); ntok[i] = len(ids) - 1 - n_pre
    for b in BLOCKS:
        h = o.hidden_states[b + 1][0]; E[b][i] = h[-1].numpy(); r = h[n_pre:-1]; M[b][i] = (r / (r.norm(dim=-1, keepdim=True) + 1e-9)).mean(0).numpy()
    if n % 200 == 0: print(f"  responses {n}/{len(flat)} ({time.time() - t0:.0f}s)", flush=True)
np.savez(f"{out}/axes_states.npz", **{f"eot{b}": E[b] for b in BLOCKS}, **{f"mean{b}": M[b] for b in BLOCKS}, ntok=ntok)
json.dump(flat, open(f"{out}/axes_flat.json", "w"))
print("saved axes states", len(flat), f"({time.time() - t0:.0f}s)")
