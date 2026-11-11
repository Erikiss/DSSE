"""The UFO paper's precondition inside Qwen3-1.7B: a complete grid of two input variables in a fixed-length format.
Prompts: a short fixed prefix with two solved examples, then 'a+b=' for a, b in 10, 12, ..., 98 (45 x 45 = 2025 problems, all with
the same number of tokens and the same token roles). Read at the '=' token: the activations of all 6144 MLP neurons of every block
(as in neu_extract.py), and whether the model's next token is the first digit of the sum.
Stored: ufo_grid_act<tag>.npy [2025, 28, 6144] fp16 (rows ordered a-major), ufo_grid<tag>.json with per block the share of
every neuron's variance over the grid that is explained by a alone, by b alone, by both main effects, and by the sum a + b
alone (medians over the neurons that vary, and means weighted by variance). The bottleneck measures are computed from the stored
activations with ufo_compress.py.
Usage: python ufo_grid.py <scratch_root> <out_dir>      env: NT, RANDINIT=1 (untrained network, tag _randinit)"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ufo_common import load_model

root, out = sys.argv[1], sys.argv[2]; RANDINIT = os.environ.get("RANDINIT") == "1"; TAG = "_randinit" if RANDINIT else ""; torch.set_grad_enabled(False); torch.set_num_threads(int(os.environ.get("NT", "3")))
tok, model = load_model(root, RANDINIT); LAYERS = model.model.layers; L = len(LAYERS); NI = model.config.intermediate_size; vals = list(range(10, 100, 2)); PRE = "17+25=42\n63+14=77\n"
grid = [(a, b) for a in vals for b in vals]; enc = [tok(PRE + f"{a}+{b}=", add_special_tokens=False)["input_ids"] for a, b in grid]; T = len(enc[0]); assert all(len(e) == T for e in enc), "prompts differ in length"
ids = torch.tensor(enc); first = [tok(str(a + b), add_special_tokens=False)["input_ids"][0] for a, b in grid]; print(f"{len(grid)} problems, {T} tokens each; last tokens: {[tok.decode(t) for t in enc[0][-6:]]}", flush=True)
cap = {}; [LAYERS[m].mlp.down_proj.register_forward_pre_hook(lambda mod, i, m=m: cap.__setitem__(m, i[0][:, -1].clone())) for m in range(L)]
A = np.zeros((len(grid), L, NI), np.float16); top = []; t0 = time.time()
for i in range(0, len(grid), 45):
    lg = model(input_ids=ids[i:i + 45]).logits[:, -1]; top += lg.argmax(-1).tolist()
    for m in range(L): A[i:i + 45, m] = cap[m].numpy().astype(np.float16)
    if i % 450 == 0: print(f"  {i}/{len(grid)} ({time.time() - t0:.0f}s)", flush=True)
acc = float(np.mean([t == f for t, f in zip(top, first)])); print(f"next token is the first digit of the sum: {acc:.3f}"); np.save(f"{out}/ufo_grid_act{TAG}.npy", A)
a_i = np.repeat(np.arange(45), 45); b_i = np.tile(np.arange(45), 45); s_i = a_i + b_i; rows = []
gm = lambda X, lab: np.stack([X[lab == g].mean(0) for g in range(lab.max() + 1)])[lab]                    # group means, expanded to all rows
for m in range(L):
    X = A[:, m].astype(np.float64); var = X.var(0); keep = var > 1e-12; X = X[:, keep]; Xc = X - X.mean(0); tot = (Xc ** 2).sum(0); ea = ((gm(X, a_i) - X.mean(0)) ** 2).sum(0) / tot; eb = ((gm(X, b_i) - X.mean(0)) ** 2).sum(0) / tot; es = ((gm(X, s_i) - X.mean(0)) ** 2).sum(0) / tot; w = tot / tot.sum()
    r = dict(block=m, varying=int(keep.sum()), by_a=float(np.median(ea)), by_b=float(np.median(eb)), additive=float(np.median(ea + eb)), by_sum=float(np.median(es)), by_a_w=float((ea * w).sum()), by_b_w=float((eb * w).sum()), additive_w=float(((ea + eb) * w).sum()), by_sum_w=float((es * w).sum()),
             sum_beyond_additive=float(np.median(np.clip(es - (ea + eb), None, None))), share_sum_gt_half=float((es > 0.5).mean())); rows.append(r)
    print(f"block {m:2d}: varying {r['varying']} | median share by a {r['by_a']:.3f}, by b {r['by_b']:.3f}, both {r['additive']:.3f}, by the sum {r['by_sum']:.3f} | neurons with more than half of their variance explained by the sum: {r['share_sum_gt_half']:.3f}", flush=True)
json.dump(dict(problems=len(grid), tokens=T, prefix=PRE, accuracy_first_digit=acc, chance_share_by_sum=88 / 2024, chance_share_by_a=44 / 2024, randinit=RANDINIT, rows=rows), open(f"{out}/ufo_grid{TAG}.json", "w"), indent=1); print(f"done ({time.time() - t0:.0f}s)")
