"""Long-training pilot: does an organised (ring / Fourier) task geometry form after the plan's checkpoint, with weight decay, and does
transfer between two inits of the same world appear with it?  Two inits of (arm, W) trained side by side for up to STEPS full-batch
AdamW updates (decay per update 1e-3 * wd_scale); at each checkpoint: geometry of both models and transfer in both directions.
Usage: python rgi_long.py <ws> <arm> <W> <wd_scale> <ckpts comma> [gamma] [K]"""
import sys, os, json, time
import numpy as np, torch
SP = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SP + "/rgi_design")
from rgi_lib import *

ws, arm, W, wds = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4]); cks = [int(c) for c in sys.argv[5].split(",")]
gamma = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
SHARED = os.environ.get("SHARED") == "1"; TRF = float(os.environ.get("TRF", "0.5"))
if SHARED:                                     # organisation arm: one split of the 289 pairs shared by all pretraining tasks (no pair is seen in any task)
    _w = M.world
    def _shared(ws_):
        pre_, new_, pairs_, splits_ = _w(ws_); g_ = np.random.default_rng(5000 + ws_); sh_ = g_.permutation(P * P); n_ = int(round(TRF * P * P))
        tr_, te_ = sh_[:n_], sh_[n_:]
        out = [np.concatenate([tr_, te_]) for _ in range(NPRE)] + splits_[NPRE:]
        return pre_, new_, pairs_, out
    M.world = _shared
pre, new, pairs, splits = M.world(ws)
NTR = int(round(TRF * P * P)) if SHARED else 144
tr = [M.data(pre[k], k, pairs[splits[k][:NTR]]) for k in range(NPRE)]; va = [M.data(pre[k], k, pairs[splits[k][NTR:]]) for k in range(NPRE)]
Xtr = torch.cat([d[0] for d in tr]); ytr = torch.cat([d[1] for d in tr]); Xva = torch.cat([d[0] for d in va]); yva = torch.cat([d[1] for d in va])
lab = torch.arange(NPRE).repeat_interleave(NTR)
ms, opts = [], []
for init in (0, 1):
    torch.manual_seed(ws if init == 0 else 10_000 * init + ws); m = mk(arm, W); ms.append(m); opts.append(M.make_opt(m, wds))
t0 = time.time(); out = []
for s in range(1, max(cks) + 1):
    for m, o in zip(ms, opts): M.bp_update(m, o, Xtr, ytr, gamma, lab)
    if s in cks:
        gs = [geometry(m, arm, W, ws) for m in ms]; rec = dict(step=s, heldout=[accuracy(m, Xva, yva) for m in ms])
        print(f"--- step {s}  held-out {rec['heldout'][0]:.3f} {rec['heldout'][1]:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        for i, g in enumerate(gs): print("  " + fmt_geom(f"i{i}", g), flush=True)
        rec["geom"] = [dict(delta=g["delta"], fourier=g["fourier"], xy=g["xy_sparsity"], self=g["self"], span=g["span"], proj=g["proj"], synth=g["synth"]) for g in gs]
        rec["pairs"] = []
        for a, b in ((0, 1), (1, 0)):
            r = pair_transfer(gs[a], gs[b], ws); rec["pairs"].append(r); print("  " + fmt_pair(f"i{a}", f"i{b}", r), flush=True)
        out.append(rec)
json.dump(out, open(f"{SP}/rgi_design/pilot/long_{arm}_W{W}_w{ws}_wd{wds:g}_g{gamma:g}" + (f"_shared{TRF:g}" if SHARED else "") + ".json", "w"), default=float)
print(f"done {time.time() - t0:.0f}s", flush=True)
