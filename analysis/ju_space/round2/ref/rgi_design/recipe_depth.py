"""Recursive recipes: v_1 = c_a + c_b - c_c, v_d = v_(d-1) + c_a' - c_b' (random anchors), target shift accumulated mod 17; accuracy
over 200 random (x, y) pairs for the target shift; mean over 200 random chains per depth; same chains (same recipe) in every model."""
import sys, numpy as np, torch
SP = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SP + "/rgi_design")
from rgi_lib import *
ws = int(sys.argv[1]); names = sys.argv[2:]; pre, new, pairs, splits = M.world(ws); rng = np.random.default_rng(7)
chains = [[tuple(rng.choice(NPRE, 3, replace=False))] + [tuple(rng.choice(NPRE, 2, replace=False)) for _ in range(5)] for _ in range(200)]
xy = pairs[np.random.default_rng(1).permutation(P * P)[:200]]
for n in names:
    d = torch.load(f"{SP}/rgi_design/pilot/models/{n}.pt"); info = d["info"]; m = mk(info["arm"], info["W"]); m.load_state_dict(d["state"])
    C = m.l1.weight.detach().numpy()[:, 2 * P: 2 * P + NPRE].astype(float); out = np.zeros((6, len(chains)))
    for ci_, ch in enumerate(chains):
        a, b, c = ch[0]; v = C[:, a] + C[:, b] - C[:, c]; t = (pre[a] + pre[b] - pre[c]) % P
        for dpt in range(6):
            if dpt > 0: a, b = ch[dpt]; v = v + C[:, a] - C[:, b]; t = (t + pre[a] - pre[b]) % P
            yy = torch.as_tensor((xy[:, 0] + xy[:, 1] + t) % P); out[dpt, ci_] = acc_cols(m, xy, yy, torch.tensor(v[None], dtype=torch.float32))[0]
    print(f"{n:34s} recipe depth 1..6: " + " ".join(f"{x:.3f}" for x in out.mean(1)), flush=True)
