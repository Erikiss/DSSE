"""Patch the attack / control states into the network and record how it reacts.
The intervention replaces the output of block 21 at the last prompt token. Everything before that point is unchanged, so per
source prompt the prefix is run once (keys and values of all earlier positions are kept) and only blocks 22..27 plus the final
norm are run for the last token, for all variants at once. Stored: the final normalized state (logits = lm_head(final)).
Usage: python mim_patch.py <scratch_root> <out_dir> [t0|t2]"""
import collections, json, os, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mim_common import BLOCK

root, out = sys.argv[1], sys.argv[2]; tag = sys.argv[3] if len(sys.argv) > 3 else "t0"
torch.set_grad_enabled(False); MODEL = f"{root}/models/qwen3-1.7b"
tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval(); base = model.model; L = model.config.num_hidden_layers
items = json.load(open(f"{out}/eval_items.json")); V = np.load(f"{out}/mim_variants_{tag}.npy"); meta = json.load(open(f"{out}/mim_variants_{tag}.json"))["meta"]
X21 = np.load(f"{out}/eval2_states_block{BLOCK}.npy"); Fin = np.load(f"{out}/eval2_final.npy"); d = V.shape[1]
by_src = collections.defaultdict(list)
for i, m in enumerate(meta): by_src[m["src"]].append(i)

class PrefixCache:
    """Keys and values of the prompt prefix; the attention of the last token appends its own and reads all of them."""
    def __init__(self, pkv): self.K = [l.keys for l in pkv.layers]; self.V = [l.values for l in pkv.layers]
    def update(self, k, v, layer_idx, *a, **kw):
        n = k.shape[0]; return torch.cat([self.K[layer_idx].expand(n, -1, -1, -1), k], 2), torch.cat([self.V[layer_idx].expand(n, -1, -1, -1), v], 2)
def tail(h, T, cache):                                    # h [n, d]: residual after block BLOCK at the last token (position T-1) -> final normalized state
    h = h[:, None, :]; pos = torch.full((h.shape[0], 1), T - 1, dtype=torch.long); pe = base.rotary_emb(h, pos)
    for i in range(BLOCK + 1, L): h = base.layers[i](h, attention_mask=None, position_ids=pos, past_key_values=cache, use_cache=True, position_embeddings=pe)
    return base.norm(h)[:, 0], h[:, 0]                     # final normalized state, and the state before the final norm

import hashlib
Fout = np.zeros((len(V), d), np.float32); Hpre = np.zeros((len(items), d), np.float32); chk = []; t0 = time.time(); CH = 80; part = f"{out}/mim_final_{tag}.partial.npz"
fp = hashlib.sha1(V.tobytes()).hexdigest()               # a partial result is only reused for exactly these variant states
if os.path.exists(part):
    z = np.load(part)
    if str(z["fp"]) == fp and z["F"].shape == Fout.shape: Fout, Hpre = z["F"], z["H"]
done = np.abs(Fout).sum(1) > 0; print(f"resuming: {int(done.sum())} of {len(V)} patched runs already there", flush=True)
for n, (src, idxs) in enumerate(sorted(by_src.items(), key=lambda kv: len(items[kv[0]][2]))):
    if done[idxs].all(): continue
    s = tok.apply_chat_template([{"role": "user", "content": items[src][2]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(s, return_tensors="pt", add_special_tokens=False)["input_ids"]; T = ids.shape[1]
    cache = PrefixCache(base(input_ids=ids[:, :-1], use_cache=True).past_key_values)
    f0, h0 = tail(torch.tensor(X21[src])[None], T, cache); f0 = f0[0]; Hpre[src] = h0[0].numpy(); chk.append(float((f0 - torch.tensor(Fin[src])).abs().max()))   # the stored original state must reproduce the stored final state
    if len(chk) == 1:                                     # one full check against the ordinary forward pass of the whole model
        o = model(input_ids=ids, output_hidden_states=True); ref = o.hidden_states[-1][0, -1]
        if not torch.allclose(model.lm_head(ref), o.logits[0, -1], atol=1e-3): ref = base.norm(ref)
        print(f"  check against a full forward pass: block state {float((o.hidden_states[BLOCK + 1][0, -1] - torch.tensor(X21[src])).abs().max()):.2e}, final state via tail {float((f0 - ref).abs().max()):.2e}", flush=True)
    for c in range(0, len(idxs), CH):
        ii = idxs[c:c + CH]; Fout[ii] = tail(torch.tensor(V[ii]), T, cache)[0].numpy()
    if n % 25 == 0: print(f"  {n}/{len(by_src)} source prompts ({time.time() - t0:.0f}s), largest deviation of the unpatched tail from the stored final state so far: {max(chk):.2e}", flush=True)
    if n % 10 == 9: np.savez(part + ".tmp.npz", F=Fout, H=Hpre, fp=fp); os.replace(part + ".tmp.npz", part)
np.save(f"{out}/mim_final_{tag}.npy", Fout); np.save(f"{out}/mim_prenorm_{tag}.npy", Hpre)   # Hpre: pre-norm final state of the unpatched source prompts (zero rows = not a source)
if os.path.exists(part): os.remove(part)
oi = [i for i, m in enumerate(meta) if m["name"] == "orig"]
print("unpatched tail vs stored final state, max over the sources of this run:", max(chk) if chk else None)
if oi: print("variant 'orig' (state patched with itself) vs stored final state:", float(np.abs(Fout[oi] - Fin[[meta[i]['src'] for i in oi]]).max()))
print(f"saved {len(V)} patched runs ({time.time() - t0:.0f}s)")
