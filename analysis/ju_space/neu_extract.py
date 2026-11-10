"""MLP neurons of Qwen3-1.7B as units: activations at the last prompt token of every DSSE eval prompt.
A 'neuron' is one hidden unit of a block's SwiGLU MLP; its activation a = silu(gate(x)) * up(x) is the input of down_proj,
its write to the residual stream is a * W_down[:, n] (definitions as in Goodfire, 'Neurons as an Interpretable Basis', 2026).
Stored per prompt: a for all 28 x 6144 neurons (fp16), the sign of the gate pre-activation (packed bits),
and the vectors that the attention and the MLP of each block add to the residual stream at that token (fp16).
Usage: python neu_extract.py <scratch_root> <out_dir>"""
import json, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
root, out = sys.argv[1], sys.argv[2]
torch.set_grad_enabled(False); MODEL = f"{root}/models/qwen3-1.7b"
tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
L = model.config.num_hidden_layers; NI = model.config.intermediate_size; d = model.config.hidden_size
items = json.load(open(f"{out}/eval_items.json")); n = len(items)
A = np.zeros((n, L, NI), np.float16); GS = np.zeros((n, L, NI // 8), np.uint8); MO = np.zeros((n, L, d), np.float16); AO = np.zeros((n, L, d), np.float16); cur = {}
def mk(l):
    lay = model.model.layers[l]
    lay.mlp.down_proj.register_forward_pre_hook(lambda m, inp: cur.__setitem__(("a", l), inp[0][0, -1].detach()))
    lay.mlp.gate_proj.register_forward_hook(lambda m, inp, o: cur.__setitem__(("g", l), (o[0, -1] > 0).detach()))
    lay.mlp.register_forward_hook(lambda m, inp, o: cur.__setitem__(("m", l), o[0, -1].detach()))
    lay.self_attn.register_forward_hook(lambda m, inp, o: cur.__setitem__(("t", l), (o[0] if isinstance(o, tuple) else o)[0, -1].detach()))
for l in range(L): mk(l)
X21 = np.load(f"{out}/eval2_states_block21.npy"); order = sorted(range(n), key=lambda i: len(items[i][2])); t0 = time.time(); chk = 0.0; start = 0
import os; part = f"{out}/neu_extract.partial.npz"
if os.path.exists(part):                                  # resume after an interruption
    z = np.load(part); start = int(z["k"]); A[:], GS[:], MO[:], AO[:] = z["A"], z["GS"], z["MO"], z["AO"]; print(f"resuming at prompt {start}", flush=True)
for k, i in enumerate(order):
    if k < start: continue
    if k > start and k % 300 == 0: np.savez(part + ".tmp.npz", k=k, A=A, GS=GS, MO=MO, AO=AO); os.replace(part + ".tmp.npz", part)
    s = tok.apply_chat_template([{"role": "user", "content": items[i][2]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    e = tok(s, return_tensors="pt", add_special_tokens=False); o = model(**e, output_hidden_states=(k % 100 == 0))
    for l in range(L):
        A[i, l] = cur[("a", l)].numpy(); GS[i, l] = np.packbits(cur[("g", l)].numpy()); MO[i, l] = cur[("m", l)].numpy(); AO[i, l] = cur[("t", l)].numpy()
    if k % 100 == 0:                                       # checks: same state as the earlier pass; residual = embedding + sum of attention and MLP writes; MLP write = a @ W_down^T
        h21 = o.hidden_states[22][0, -1]; rec = o.hidden_states[0][0, -1] + sum(cur[("t", l)] + cur[("m", l)] for l in range(22))
        w = model.model.layers[21].mlp.down_proj.weight; chk = max(chk, float((h21 - torch.tensor(X21[i])).abs().max()), float((h21 - rec).abs().max()), float((cur[("a", 21)] @ w.T - cur[("m", 21)]).abs().max()))
        print(f"  {k}/{n} ({time.time() - t0:.0f}s) largest check deviation so far {chk:.2e}", flush=True)
np.save(f"{out}/neu_act.npy", A); np.save(f"{out}/neu_gatesign.npy", GS); np.save(f"{out}/neu_mlp_out.npy", MO); np.save(f"{out}/neu_attn_out.npy", AO)
wn = np.stack([model.model.layers[l].mlp.down_proj.weight.norm(dim=0).numpy() for l in range(L)]); np.save(f"{out}/neu_wdown_norm.npy", wn)
print(f"saved neuron activations {A.shape} ({time.time() - t0:.0f}s); largest check deviation {chk:.2e}")
if os.path.exists(part): os.remove(part)
