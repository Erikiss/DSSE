"""The UFO paper's precondition inside Qwen3-1.7B: a complete grid of two input variables in a fixed-length format.
Prompts: a short fixed prefix with two solved examples, then 'a+b=' for a, b in 10, 12, ..., 98 (45 x 45 = 2025 problems, all with
the same number of tokens and the same token roles). Read at the '=' token: the activations of all 6144 MLP neurons of every block
(as in neu_extract.py), and whether the model's next token is the first digit of the sum.
Stored: ufo_grid_act<tag>.npy [2025, 28, 6144] fp16 (rows ordered a-major), ufo_grid<tag>.json with per block the share of
every neuron's variance over the grid that is explained by a alone, by b alone, by both main effects, and by the sum a + b
alone (medians over the neurons that vary, and means weighted by variance). The shares by a and b and by the sum overlap (an
activation that grows evenly with a and with b counts fully in both), so the share of the sum BEYOND a and b is given as well
(joint model of a, b and the sum minus the two main effects), and the share explained by the first digit of the sum alone, which
is the token the model outputs next. The bottleneck measures are computed from the stored
activations with ufo_compress.py.
Usage: python ufo_grid.py <scratch_root> <out_dir>      env: NT, RANDINIT=1 (untrained network, tag _randinit),
       FROMFILE=1 (no model pass: recompute the decomposition from the stored activations)"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ufo_common import load_model

root, out = sys.argv[1], sys.argv[2]; RANDINIT = os.environ.get("RANDINIT") == "1"; TAG = "_randinit" if RANDINIT else ""; torch.set_grad_enabled(False); torch.set_num_threads(int(os.environ.get("NT", "3")))
vals = list(range(10, 100, 2)); PRE = "17+25=42\n63+14=77\n"; grid = [(a, b) for a in vals for b in vals]; t0 = time.time()
if os.environ.get("FROMFILE") == "1":                                                    # only the decomposition, from the stored activations
    A = np.load(f"{out}/ufo_grid_act{TAG}.npy", mmap_mode="r"); L, NI = A.shape[1], A.shape[2]; old = json.load(open(f"{out}/ufo_grid{TAG}.json")); acc, T = old["accuracy_first_digit"], old["tokens"]
else:
    tok, model = load_model(root, RANDINIT); LAYERS = model.model.layers; L = len(LAYERS); NI = model.config.intermediate_size
    enc = [tok(PRE + f"{a}+{b}=", add_special_tokens=False)["input_ids"] for a, b in grid]; T = len(enc[0]); assert all(len(e) == T for e in enc), "prompts differ in length"
    ids = torch.tensor(enc); first = [tok(str(a + b), add_special_tokens=False)["input_ids"][0] for a, b in grid]; print(f"{len(grid)} problems, {T} tokens each; last tokens: {[tok.decode(t) for t in enc[0][-6:]]}", flush=True)
    cap = {}; [LAYERS[m].mlp.down_proj.register_forward_pre_hook(lambda mod, i, m=m: cap.__setitem__(m, i[0][:, -1].clone())) for m in range(L)]
    A = np.zeros((len(grid), L, NI), np.float16); top = []
    for i in range(0, len(grid), 45):
        lg = model(input_ids=ids[i:i + 45]).logits[:, -1]; top += lg.argmax(-1).tolist()
        for m in range(L): A[i:i + 45, m] = cap[m].numpy().astype(np.float16)
        if i % 450 == 0: print(f"  {i}/{len(grid)} ({time.time() - t0:.0f}s)", flush=True)
    acc = float(np.mean([t == f for t, f in zip(top, first)])); print(f"next token is the first digit of the sum: {acc:.3f}"); np.save(f"{out}/ufo_grid_act{TAG}.npy", A)
a_i = np.repeat(np.arange(45), 45); b_i = np.tile(np.arange(45), 45); s_i = a_i + b_i; d_i = np.array([int(str(a + b)[0]) for a, b in grid]); rows = []
gm = lambda X, lab: np.stack([X[lab == g].mean(0) for g in np.unique(lab)])[np.searchsorted(np.unique(lab), lab)]     # group means, expanded to all rows
oh = lambda lab: (lab[:, None] == np.unique(lab)[None]).astype(np.float64); Dj = np.concatenate([oh(a_i), oh(b_i), oh(s_i)], 1)
Uj, sj, _ = np.linalg.svd(Dj, full_matrices=False); Qj = Uj[:, sj > 1e-8 * sj[0]]                                    # orthonormal basis of the joint model: a, b and the sum
for m in range(L):
    X = np.asarray(A[:, m]).astype(np.float64); var = X.var(0); keep = var > 1e-12; X = X[:, keep]; mu = X.mean(0); Xc = X - mu; tot = (Xc ** 2).sum(0); sh = lambda lab: ((gm(X, lab) - mu) ** 2).sum(0) / tot
    ea, eb, es, ed = sh(a_i), sh(b_i), sh(s_i), sh(d_i); ej = ((Qj.T @ Xc) ** 2).sum(0) / tot; beyond = ej - (ea + eb); w = tot / tot.sum(); top1 = float(np.sort(tot)[::-1][: max(1, len(tot) // 100)].sum() / tot.sum())
    r = dict(block=m, varying=int(keep.sum()), by_a=float(np.median(ea)), by_b=float(np.median(eb)), additive=float(np.median(ea + eb)), by_sum=float(np.median(es)), by_a_w=float((ea * w).sum()), by_b_w=float((eb * w).sum()), additive_w=float(((ea + eb) * w).sum()), by_sum_w=float((es * w).sum()),
             share_sum_gt_half=float((es > 0.5).mean()), sum_beyond_additive=float(np.median(beyond)), sum_beyond_additive_w=float((beyond * w).sum()), share_beyond_gt_half=float((beyond > 0.5).mean()), by_first_digit=float(np.median(ed)), by_first_digit_w=float((ed * w).sum()), variance_share_of_top_1pct_neurons=top1); rows.append(r)
    print(f"block {m:2d}: varying {r['varying']} | median share by a {r['by_a']:.3f}, by b {r['by_b']:.3f}, both {r['additive']:.3f}, by the sum {r['by_sum']:.3f}, sum beyond a and b {r['sum_beyond_additive']:.3f} | weighted by variance: sum {r['by_sum_w']:.3f}, beyond {r['sum_beyond_additive_w']:.3f}, first digit of the sum {r['by_first_digit_w']:.3f} | neurons > half by the sum {r['share_sum_gt_half']:.3f}, > half beyond {r['share_beyond_gt_half']:.3f} | top 1 % of neurons hold {top1:.2f} of the variance", flush=True)
json.dump(dict(problems=len(grid), tokens=T, prefix=PRE, accuracy_first_digit=acc, chance_share_by_sum=88 / 2024, chance_share_by_a=44 / 2024, randinit=RANDINIT, rows=rows), open(f"{out}/ufo_grid{TAG}.json", "w"), indent=1); print(f"done ({time.time() - t0:.0f}s)")
