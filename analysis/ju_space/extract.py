"""J-space / U-space on DSSE with Qwen3-1.7B (CPU).
Step 1: sanity-check the Jacobian lens, build the U-Space basis (reference construction of Braun et al., arXiv 2610.09087),
        extract readout-block states for all DSSE eval prompts (base model, thinking off, last prompt token).
Usage: python extract.py <scratch_root> <out_dir>"""
import json, sys, time
import numpy as np, torch, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
root, out = sys.argv[1], sys.argv[2]
torch.set_grad_enabled(False); torch.manual_seed(0)
MODEL = f"{root}/models/qwen3-1.7b"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
L = model.config.num_hidden_layers; BLOCK = int(sys.argv[3]) if len(sys.argv) > 3 else 18   # ~2/3 depth of 28 blocks
lens = torch.load(f"{root}/jlens/Qwen3-1.7B_jacobian_lens.pt", map_location="cpu", weights_only=True)
J = {k: v.float() for k, v in lens["J"].items()}
W_U = model.lm_head.weight.detach(); gamma = model.model.norm.weight.detach()
print(f"layers {L}, readout block {BLOCK}, lens layers {sorted(J)[0]}..{sorted(J)[-1]}, n_prompts {lens['n_prompts']}", flush=True)

def unit(a): return a / (a.norm(dim=-1, keepdim=True) + 1e-9)

# ---- sanity: lens top-1 agreement with the model's own next-token prediction, by block (hidden_states[b+1] = output of block b)
texts = ["The capital of France is", "Fact: The currency used in the country shaped like a boot is the", "Two plus two equals",
         "The opposite of hot is", "The first president of the United States was George", "Water is made of hydrogen and"]
agree = {b: 0 for b in sorted(J)}; rank_true = {b: [] for b in sorted(J)}; shown = {}
for t in texts:
    e = tok(t, return_tensors="pt"); o = model(**e, output_hidden_states=True)
    top = o.logits[0, -1].argmax().item()
    for b in sorted(J):
        h = o.hidden_states[b + 1][0, -1]
        lg = model.lm_head(model.model.norm(J[b] @ h))
        agree[b] += int(lg.argmax().item() == top); rank_true[b].append(int((lg > lg[top]).sum().item()))
        if b in (6, 12, BLOCK, 24): shown.setdefault(t, {})[b] = [tok.decode([i]) for i in lg.topk(4).indices.tolist()]
    shown[t]["model"] = tok.decode([top])
print("lens top-1 agreement with model output by block:", {b: f"{agree[b]}/{len(texts)}" for b in (0, 6, 12, BLOCK, 22, 26)})
print("median rank of the model's token under the lens:", {b: int(np.median(rank_true[b])) for b in (0, 6, 12, BLOCK, 22, 26)})
for t, d in shown.items(): print("  ", repr(t[-40:]), "->", d)

# ---- U-Space basis (reference: unit((W_U[ids] * gamma) @ J[block]); anchor centroid - pole centroid; Loewdin)
terms = yaml.safe_load(open(f"{root}/uspace_repo/data_uspace_terms.yaml"))
def single_token_id(word):
    for s in (" " + word, word):
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1: return ids[0]
    return None
def pull(ids): return unit((W_U[ids] * gamma) @ J[BLOCK])
def lowdin(M):
    w, V = torch.linalg.eigh(M.T @ M); U = M @ (V @ torch.diag(w.clamp_min(1e-12) ** -0.5) @ V.T)
    return U * torch.sign((U * M).sum(0))[None, :]
cats = list(terms); used = {}; B = []
for c in cats:
    a = [i for i in map(single_token_id, terms[c]["anchors"]) if i is not None]; p = [i for i in map(single_token_id, terms[c]["poles"]) if i is not None]
    used[c] = dict(anchors=[w for w in terms[c]["anchors"] if single_token_id(w) is not None], poles=[w for w in terms[c]["poles"] if single_token_id(w) is not None])
    B.append(unit(pull(a).mean(0) - pull(p).mean(0)))
B = torch.stack(B, 1); U = lowdin(B)
print("U-Space:", tuple(U.shape), "| single-token terms used:", {c: (len(used[c]["anchors"]), len(used[c]["poles"])) for c in cats})
print("cosines between raw category directions:\n", np.round((B.T @ B).numpy(), 2))
for k, c in enumerate(cats):   # what does each category direction read out through the lens? (decode J b)
    lg = W_U @ (gamma * (J[BLOCK] @ B[:, k])); print(f"   {c}: +", [tok.decode([i]).strip() for i in lg.topk(8).indices.tolist()], " -", [tok.decode([i]).strip() for i in (-lg).topk(6).indices.tolist()])
vocab_pool = [i for i in range(2000, 100000) if tok.decode([i]).strip().isalpha() and len(tok.decode([i]).strip()) > 3]
torch.save(dict(U=U, B=B, cats=cats, used=used, block=BLOCK, vocab_pool=vocab_pool), f"{out}/uspace_block{BLOCK}.pt")

# ---- DSSE eval prompts: state at the readout block, last prompt token (thinking off), plus two neighbouring blocks for robustness
beh = json.load(open(f"{root}/data/behaviors.json"))
items = [(x["behavior_id"], p["prompt_idx"], p["prompt"]) for x in beh for p in x["eval_prompts"]]
blocks = sorted({12, BLOCK, 22}); H = {b: [] for b in blocks}; t0 = time.time()
order = sorted(range(len(items)), key=lambda i: len(items[i][2]))
for n, i in enumerate(order):
    s = tok.apply_chat_template([{"role": "user", "content": items[i][2]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    e = tok(s, return_tensors="pt", add_special_tokens=False); o = model(**e, output_hidden_states=True)
    for b in blocks: H[b].append((i, o.hidden_states[b + 1][0, -1].clone()))
    if n % 200 == 0: print(f"  eval prompts {n}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
for b in blocks:
    arr = torch.zeros(len(items), model.config.hidden_size)
    for i, h in H[b]: arr[i] = h
    np.save(f"{out}/eval_states_block{b}.npy", arr.numpy())
json.dump([(a, b, c) for a, b, c in items], open(f"{out}/eval_items.json", "w"))
print("saved eval states", len(items), f"({time.time() - t0:.0f}s)")
