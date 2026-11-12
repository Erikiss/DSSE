"""Lattice synthesis: the new task's column written as c_a + c_b - c_c over pretraining anchors with a + b - c = s_new (mod 17), no
shots, no learning; held-out accuracy on the new task, mean over all such triples and the 4 new tasks. Also rounding of learned
alpha to integers. Per model of pilot 2 (both worlds)."""
import sys, glob, itertools, numpy as np, torch
SP = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SP + "/rgi_design")
from rgi_lib import *
for ws in (0, 1):
    pre, new, pairs, splits = M.world(ws)
    for path in sorted(glob.glob(f"{SP}/rgi_design/pilot/models/bp_mup_W*_w{ws}_i*_g*.pt")):
        d = torch.load(path); info = d["info"]
        if info["ckpt"] is None or info["heldout"] < 0.95 or (info["gamma"] and "_late" not in info["name"]): continue
        m = mk(info["arm"], info["W"]); m.load_state_dict(d["state"]); C = m.l1.weight.detach().numpy()[:, 2 * P: 2 * P + NPRE].astype(float)
        g = dict(m=m, C=C, W=info["W"]); accs = []; arnd = []
        for j in range(NNEW):
            xy = pairs[splits[NPRE + j][8:]]; yy = torch.as_tensor((xy[:, 0] + xy[:, 1] + new[j]) % P)
            V = [C[:, a] + C[:, b] - C[:, c] for a, b, c in itertools.product(range(NPRE), repeat=3) if a < b and c not in (a, b) and (pre[a] + pre[b] - pre[c]) % P == new[j]]
            accs.append(acc_cols(m, xy, yy, torch.tensor(np.stack(V), dtype=torch.float32)).mean() if V else np.nan)
            al = learn_alpha(g, ws, j); mA = C.mean(1); ar = np.round(al)
            arnd.append(acc_cols(m, xy, yy, torch.tensor(np.stack([mA + (C - mA[:, None]) @ al, C @ ar if abs(ar.sum() - 1) < 1e-9 else mA + (C - mA[:, None]) @ ar]), dtype=torch.float32)))
        arnd = np.array(arnd)
        print(f"w{ws} {info['name']:34s} lattice synthesis {np.nanmean(accs):.3f} | learned alpha {arnd[:, 0].mean():.3f} rounded {arnd[:, 1].mean():.3f}", flush=True)
