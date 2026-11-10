"""Second pass over the DSSE eval prompts (base Qwen3-1.7B, thinking off): states at the SAE layers and the final state.
Stores, per prompt: residual at blocks 14/18/21 (last prompt token; hidden_states[b+1] = output of block b),
the final normalized state (logits = lm_head(final)), two random user-content positions at block 21 (SAE in-distribution check),
and the median residual norm over positions at block 21.
Usage: python extract_eval2.py <scratch_root> <out_dir>"""
import json, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
root, out = sys.argv[1], sys.argv[2]
torch.set_grad_enabled(False); torch.manual_seed(0); rng = np.random.default_rng(0)
MODEL = f"{root}/models/qwen3-1.7b"
tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
items = json.load(open(f"{out}/eval_items.json")); n = len(items); d = model.config.hidden_size
BL = [14, 18, 21]; S = {b: np.zeros((n, d), np.float32) for b in BL}; F = np.zeros((n, d), np.float32); top1 = np.zeros(n, np.int64)
C = np.zeros((n, 2, d), np.float32); Cpos = np.zeros((n, 2), np.int64); med = np.zeros(n, np.float32); ntok = np.zeros(n, np.int64); last_tok = np.zeros(n, np.int64)
order = sorted(range(n), key=lambda i: len(items[i][2])); t0 = time.time()
for k, i in enumerate(order):
    s = tok.apply_chat_template([{"role": "user", "content": items[i][2]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    e = tok(s, return_tensors="pt", add_special_tokens=False); o = model(**e, output_hidden_states=True); T = e["input_ids"].shape[1]
    for b in BL: S[b][i] = o.hidden_states[b + 1][0, -1].numpy()
    f = o.hidden_states[-1][0, -1]; top1[i] = int(o.logits[0, -1].argmax())                      # in this transformers version the last entry is already normed
    if not torch.allclose(model.lm_head(f), o.logits[0, -1], atol=1e-3): f = model.model.norm(f)
    F[i] = f.numpy()
    assert torch.allclose(model.lm_head(f), o.logits[0, -1], atol=1e-3)
    # user content sits between "<|im_start|>user\n" (3 tokens) and "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" at the end
    ids = e["input_ids"][0].tolist(); end = max(j for j, t in enumerate(ids) if t == tok.convert_tokens_to_ids("<|im_end|>")); lo, hi = 3, end
    pos = rng.integers(lo, hi, 2) if hi > lo else np.array([lo, lo]); Cpos[i] = pos
    for q, p in enumerate(pos): C[i, q] = o.hidden_states[22][0, int(p)].numpy()
    med[i] = float(o.hidden_states[22][0, 1:].norm(dim=-1).median()); ntok[i] = T; last_tok[i] = ids[-1]
    if k % 200 == 0: print(f"  {k}/{n} ({time.time() - t0:.0f}s)", flush=True)
for b in BL: np.save(f"{out}/eval2_states_block{b}.npy", S[b])
np.save(f"{out}/eval2_final.npy", F); np.save(f"{out}/eval2_ctx_block21.npy", C)
np.savez(f"{out}/eval2_meta.npz", top1=top1, ctx_pos=Cpos, median_norm_block21=med, ntok=ntok, last_tok=last_tok)
print("last prompt token:", {tok.decode([int(t)]): int(c) for t, c in zip(*np.unique(last_tok, return_counts=True))})
print("norm at block 21, last token: median", float(np.median(np.linalg.norm(S[21], axis=1))), "| median over positions:", float(np.median(med)), "| content positions:", float(np.median(np.linalg.norm(C.reshape(-1, d), axis=1))))
print("check vs first pass (block 18):", float(np.abs(S[18] - np.load(f"{out}/eval_states_block18.npy")).max()))
print(f"saved ({time.time() - t0:.0f}s)")
