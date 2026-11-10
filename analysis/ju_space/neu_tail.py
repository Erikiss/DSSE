"""How do the neurons BEHIND the intervention react? For selected attack / control states (patched at the output of block 21,
last prompt token) the activations of all MLP neurons of blocks 22-27 at that token are recorded (6 x 6144 units).
Usage: python neu_tail.py <scratch_root> <out_dir> [t0]"""
import collections, json, os, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mim_common import BLOCK
root, out = sys.argv[1], sys.argv[2]; tag = sys.argv[3] if len(sys.argv) > 3 else "t0"
KEEP = ["orig", "wperp_1", "wperp_star", "lam0_lvlstar", "dose0_2", "dosedm_1", "dmblind_0.5", "dmblind_1", "dmfull_1", "dosecen_1", "sae20_c1", "lens_t0", "rand1", "rand1_2", "rand1_3", "randlabel", "protoA", "protoB", "swap1", "swapnn"]
torch.set_grad_enabled(False); MODEL = f"{root}/models/qwen3-1.7b"
tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval(); base = model.model; L = model.config.num_hidden_layers; NI = model.config.intermediate_size
items = json.load(open(f"{out}/eval_items.json")); meta = json.load(open(f"{out}/mim_variants_{tag}.json"))["meta"] + json.load(open(f"{out}/mim_variants_{tag}b.json"))["meta"]
V = np.concatenate([np.load(f"{out}/mim_variants_{tag}.npy"), np.load(f"{out}/mim_variants_{tag}b.npy")]); sel = [i for i, m in enumerate(meta) if m["name"] in KEEP]; A1 = np.load(f"{out}/neu_act.npy", mmap_mode="r")
by_src = collections.defaultdict(list)
for i in sel: by_src[meta[i]["src"]].append(i)
cur = {}
for l in range(BLOCK + 1, L): base.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, inp, l=l: cur.__setitem__(l, inp[0][:, 0].detach()))
class PrefixCache:
    def __init__(self, pkv): self.K = [x.keys for x in pkv.layers]; self.V = [x.values for x in pkv.layers]
    def update(self, k, v, layer_idx, *a, **kw):
        n = k.shape[0]; return torch.cat([self.K[layer_idx].expand(n, -1, -1, -1), k], 2), torch.cat([self.V[layer_idx].expand(n, -1, -1, -1), v], 2)
def tail(h, T, cache):
    h = h[:, None, :]; pos = torch.full((h.shape[0], 1), T - 1, dtype=torch.long); pe = base.rotary_emb(h, pos)
    for i in range(BLOCK + 1, L): h = base.layers[i](h, attention_mask=None, position_ids=pos, past_key_values=cache, use_cache=True, position_embeddings=pe)
    return torch.stack([cur[l] for l in range(BLOCK + 1, L)], 1)                                         # [n, 6, 6144]
OUT = np.zeros((len(sel), L - BLOCK - 1, NI), np.float16); row = {i: k for k, i in enumerate(sel)}; t0 = time.time(); chk = 0.0
for n, (src, idxs) in enumerate(sorted(by_src.items(), key=lambda kv: len(items[kv[0]][2]))):
    s = tok.apply_chat_template([{"role": "user", "content": items[src][2]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(s, return_tensors="pt", add_special_tokens=False)["input_ids"]; T = ids.shape[1]; cache = PrefixCache(base(input_ids=ids[:, :-1], use_cache=True).past_key_values)
    a = tail(torch.tensor(V[idxs]), T, cache).numpy(); OUT[[row[i] for i in idxs]] = a
    for k, i in enumerate(idxs):
        if meta[i]["name"] == "orig": chk = max(chk, float(np.abs(a[k] - A1[src, BLOCK + 1:].astype(np.float32)).max()))    # unchanged state must reproduce the stored neuron activations
    if n % 25 == 0: print(f"  {n}/{len(by_src)} source prompts ({time.time() - t0:.0f}s), unchanged state vs stored neuron activations: max deviation {chk:.3f}", flush=True)
np.save(f"{out}/neu_tail_act.npy", OUT); json.dump([dict(meta[i], row=row[i]) for i in sel], open(f"{out}/neu_tail_meta.json", "w"))
print(f"saved {OUT.shape} ({time.time() - t0:.0f}s); unchanged state vs stored: max deviation {chk:.3f} (fp16 storage)")
