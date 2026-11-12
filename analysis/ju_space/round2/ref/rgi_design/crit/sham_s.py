"""S line (alpha learned in A over its 12 anchors, carried to B of another init at the same condition; gain over 30 permuted
correspondences) for natural, task_late +-0.01 and x_late (sham) +-0.01, worlds 0 and 1."""
import sys, json, numpy as np, torch
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/rgi_design/crit")
exec(open("sham.py").read().split("out = []; specs")[0])
from rgi_lib import learn_alpha
res = {}
for ws in (0, 1):
    pre, new, pairs, splits = M.world(ws); g = np.random.default_rng(31 + ws)
    for mode, gam in (("nat", 0.0), ("task_late", 0.01), ("task_late", -0.01), ("x_late", 0.01), ("x_late", -0.01)):
        Ms = []
        for init in (0, 1):
            m, info = train(ws, init, mode, gam); d = M.delta_stats(m, pre, pairs, ws)["h2"]["delta"]
            Ms.append(dict(m=m, C=m.l1.weight.detach().numpy()[:, 2 * P: 2 * P + NPRE].astype(float), W=128, delta=d))
        gains, inA = [], []
        for a, b in ((0, 1), (1, 0)):
            A, B = Ms[a], Ms[b]
            for j in range(NNEW):
                al = learn_alpha(A, ws, j); xy = pairs[splits[NPRE + j][8:]]; yy = torch.as_tensor((xy[:, 0] + xy[:, 1] + new[j]) % P)
                mA, mB = A["C"].mean(1), B["C"].mean(1)
                inA.append(acc_cols(A["m"], xy, yy, torch.tensor(np.stack([mA + (A["C"] - mA[:, None]) @ al]), dtype=torch.float32))[0])
                V = [mB + (B["C"] - mB[:, None]) @ al] + [mB + (B["C"] - mB[:, None])[:, g.permutation(NPRE)] @ al for _ in range(30)]
                acc = acc_cols(B["m"], xy, yy, torch.tensor(np.stack(V), dtype=torch.float32)); gains.append(acc[0] - acc[1:].mean())
        print(f"w{ws} {mode:9s} g {gam:+.2f}: delta h2 {np.mean([x['delta'] for x in Ms]):.3f} | alpha in A {np.mean(inA):.3f} | S-gain (cross-init, same condition) {np.mean(gains):+.3f}", flush=True)
