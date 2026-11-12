"""Pilot 2: anchored ('alpha') improvements vs free columns, with the late delta probe, for noise calibration of the rule.
Per model: delta, free-column self accuracy, in-span share and projected accuracy of the free column (forcing index), alpha-learned
accuracy (expressivity). Per ordered pair: alpha carried to B (permitted RGI) vs shuffled correspondence, free column by T1 (forced
RGI) vs shuffled, LOO-anchor. spec = arm:W:init:gamma (gamma applied only after held-out first reaches 0.95).
Usage: python rgi_pilot2.py <ws> <out.json> <spec> ...    pairs: all ordered pairs with different init"""
import sys, os, json, time, itertools
import numpy as np, torch
SP = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SP + "/rgi_design")
os.environ["GDELAY"] = "1"
import rgi_pilot as RP
from rgi_lib import *

ws, outp = int(sys.argv[1]), sys.argv[2]; specs = [s.split(":") for s in sys.argv[3:]]; G = {}; t0 = time.time()
for arm, W, init, gamma in specs:
    W, init, gamma = int(W), int(init), float(gamma); m, info = RP.pretrain(ws, arm, W, init, gamma); n = info["name"]
    if info["ckpt"] is None or info["heldout"] < 0.95: print(f"{n}: fails level 1 (held-out {info['heldout']:.3f})", flush=True); continue
    g = geometry(m, arm, W, ws); g["init"] = init; g["gamma"] = gamma
    acc_alpha = []
    pre, new, pairs, splits = M.world(ws)
    for j in range(NNEW):
        a = learn_alpha(g, ws, j); g.setdefault("alpha", {})[j] = a
        xy = pairs[splits[NPRE + j][8:]]; yy = torch.as_tensor((xy[:, 0] + xy[:, 1] + new[j]) % P); mA = g["C"].mean(1)
        acc_alpha.append(acc_cols(m, xy, yy, torch.tensor(np.stack([mA + (g["C"] - mA[:, None]) @ a]), dtype=torch.float32))[0])
    g["acc_alpha"] = acc_alpha; G[n] = g
    print(f"{n}: delta h2 {g['delta']['h2']['delta']:.3f} | free {np.mean(g['self']):.3f} alpha {np.mean(acc_alpha):.3f} | in-span share {np.mean(g['span']):.2f} proj {np.mean(g['proj']):.3f} ({time.time() - t0:.0f}s)", flush=True)
rows = []
for a, b in itertools.permutations(G, 2):
    if G[a]["init"] == G[b]["init"]: continue
    r = pair_transfer(G[a], G[b], ws, nperm=20, nloo_perm=5); s = span_transfer(G[a], G[b], ws, nperm=20); r.update(s)
    r.update(A=a, B=b, dA=G[a]["delta"]["h2"]["delta"], dB=G[b]["delta"]["h2"]["delta"], WA=G[a]["W"], WB=G[b]["W"], gA=G[a]["gamma"], gB=G[b]["gamma"],
             freeA=float(np.mean(G[a]["self"])), freeB=float(np.mean(G[b]["self"])), alphaA=float(np.mean(G[a]["acc_alpha"])), alphaB=float(np.mean(G[b]["acc_alpha"])),
             projA=float(np.mean(G[a]["proj"])), spanA=float(np.mean(G[a]["span"])))
    rows.append(r)
    print(f"{a:>32} -> {b:<32} alpha-carried {s['spanB']:.3f} (shuf {s['span_shuf']:.3f}) | free T1 {r['t1']:.3f} (shuf {r['shuf']:.3f}) | LOO {r['loo']:.3f} ({r['loo_shuf']:.3f}) | B free {r['freeB']:.3f} alpha {r['alphaB']:.3f}", flush=True)
json.dump(dict(models={k: dict(delta=g["delta"], W=g["W"], init=g["init"], gamma=g["gamma"], free=g["self"], alpha=g["acc_alpha"], span=g["span"], proj=g["proj"], synth=g["synth"],
                              fourier=g["fourier"]) for k, g in G.items()}, pairs=rows), open(outp, "w"), default=float)
print(f"done {time.time() - t0:.0f}s", flush=True)
