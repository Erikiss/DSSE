"""README section 11, stage 3a: long reasoning traces of Qwen3-1.7B for the teacher-forced tests (rule A4a of the plan review).
The first NPROB GSM8K test problems (data: openai/gsm8k, 'main' test split), chat template with thinking on, Qwen's recommended
sampler for thinking mode (temperature 0.6, top-k 20, top-p 0.95), at most L new tokens, batches of BATCH, one seed per batch
(torch.manual_seed(SEED + batch index) before each batch, so a later run with the same settings draws the same tokens until the
distributions differ). Per trace: the token ids, the position of '</think>' if any, the number of thinking tokens, whether the
final answer matches the GSM8K answer (last number after '</think>'), and the share of repeated 4-grams in the last 256 tokens.
Trace set (plan review P07): the first 12 problems whose thinking ends between 300 and 1,024 tokens; if fewer than 12 of the 24
qualify, L = 1,536 and the window ends at 1,536. Run once with L = 1,536: with the same seeds and batches the first 1,024 tokens are
those of a run with L = 1,024, so both windows are evaluated on the same traces. Thinking length counts the '<think>' token.
torch seed per batch = 1000 * SEED + batch index (stored per trace, with BATCH), so a later run can couple to these draws.
GPU (12 Nov 2026): the traces for A4a are sampled on the A100 of GPU shard gpu4 (DEVICE=cuda); the rule is unchanged. The CPU
traces of batches 0-1 are kept as a record only and enter nothing.
Usage: python dn_sample.py <root> <gsm8k_test.json> <out.json>      env: NT (3), NPROB (24), L (1536), BATCH (8), SEED (0), DEVICE"""
import json, os, re, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ufo_common import load_model

root, data, out = sys.argv[1], sys.argv[2], sys.argv[3]; torch.set_num_threads(int(os.environ.get("NT", "3"))); t0 = time.time()
NPROB, L, BATCH, SEED = int(os.environ.get("NPROB", "24")), int(os.environ.get("L", "1536")), int(os.environ.get("BATCH", "8")), int(os.environ.get("SEED", "0"))
mem = int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 1e6; assert mem >= 9.5 or os.environ.get("DEVICE", "cpu") != "cpu", f"only {mem:.1f} GB available"
tok, model = load_model(root); DEV = next(model.parameters()).device; PAD = tok.pad_token_id; END_THINK = tok.convert_tokens_to_ids("</think>"); IM_END = tok.convert_tokens_to_ids("<|im_end|>")
probs = json.load(open(data))[:NPROB]
prompt = lambda q: tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True), add_special_tokens=False)["input_ids"]
gold = lambda a: a.split("####")[-1].strip().replace(",", "")
def rep4(ids):
    t = ids[-256:]; g = [tuple(t[i:i + 4]) for i in range(len(t) - 3)]; return 1 - len(set(g)) / len(g) if g else 0.0
res = dict(model="Qwen3-1.7B (post-trained checkpoint), float32, thinking on", sampler=dict(temperature=0.6, top_k=20, top_p=0.95), L=L, seed=SEED, traces=[])
if os.path.exists(out):       # resume: keep the batches already written (same settings), run only the missing ones
    old = json.load(open(out)); assert (old["L"], old["seed"]) == (L, SEED), "settings differ from the saved run"
    res["traces"] = [t for t in old["traces"] if t.get("batch_size", BATCH) == BATCH]
DONE = {t["batch"] for t in res["traces"]}; ONLY = os.environ.get("BATCHES"); ONLY = None if ONLY is None else {int(b) for b in ONLY.split(",")}
print(f"model loaded ({time.time() - t0:.0f}s); batches already done {sorted(DONE)}", flush=True)
for b0 in range(0, NPROB, BATCH):
    if b0 // BATCH in DONE or (ONLY is not None and b0 // BATCH not in ONLY): continue
    P = [prompt(p["question"]) for p in probs[b0:b0 + BATCH]]; T = max(len(p) for p in P)
    ids = torch.full((len(P), T), PAD); am = torch.zeros(len(P), T, dtype=torch.long)
    for i, p in enumerate(P): ids[i, T - len(p):] = torch.tensor(p); am[i, T - len(p):] = 1
    tseed = 1000 * SEED + b0 // BATCH; torch.manual_seed(tseed)
    with torch.no_grad(): g = model.generate(input_ids=ids.to(DEV), attention_mask=am.to(DEV), max_new_tokens=L, do_sample=True, temperature=0.6, top_k=20, top_p=0.95, pad_token_id=PAD).cpu()
    for i, row in enumerate(g[:, T:].tolist()):
        gen = []
        for t in row:
            gen.append(t)
            if t in (IM_END, PAD): break
        text = tok.decode(gen); k = gen.index(END_THINK) if END_THINK in gen else None
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text.split("</think>")[-1]) if k is not None else []
        ans = nums[-1].replace(",", "").rstrip(".") if nums else None; gl = gold(probs[b0 + i]["answer"])
        res["traces"].append(dict(problem=b0 + i, prompt_ids=P[i], gen_ids=gen, think_end=k, think_tokens=k, finished=gen[-1] in (IM_END, PAD), answer=ans, gold=gl,
                                  right=(ans is not None and ans.lstrip("0") == gl.lstrip("0")) or ans == gl, repeated_4grams=rep4(gen[:k] if k else gen),
                                  batch=b0 // BATCH, batch_size=BATCH, torch_seed=tseed, qualifies_1024=k is not None and 300 <= k <= 1024, qualifies=k is not None and 300 <= k <= L))
    json.dump(res, open(out, "w"))
    res["traces"].sort(key=lambda t: t["problem"]); new = [t for t in res["traces"] if t["batch"] == b0 // BATCH]
    print(f"batch {b0 // BATCH}: thinking lengths {[t['think_tokens'] for t in new]} right {[t['right'] for t in new]} ({time.time() - t0:.0f}s)", flush=True)
if len({t["problem"] for t in res["traces"]}) < NPROB: print("not all batches done; no trace set yet", flush=True); sys.exit(0)
q1 = [t["problem"] for t in res["traces"] if t["qualifies_1024"]]; qL = [t["problem"] for t in res["traces"] if t["qualifies"]]
use = q1 if len(q1) >= 12 else qL; res.update(qualifying_1024=q1, qualifying=qL, L_used=1024 if len(q1) >= 12 else L, trace_set=use[:12], trace_set_complete=len(use) >= 12)
res["seconds"] = time.time() - t0; json.dump(res, open(out, "w"))
print(f"qualifying with window 300-1024: {len(q1)} {q1}; with 300-{L}: {len(qL)} {qL}; L used {res['L_used']}; trace set {res['trace_set']} (complete: {res['trace_set_complete']}) ({time.time() - t0:.0f}s)", flush=True)
