"""Are neurons wired to specific neurons? Weights only, no data (after Figure 20 of Goodfire 2026, 'Neurons as an Interpretable
Basis'; there for Qwen3.6-35B-A3B, here for the dense Qwen3-1.7B).
For every neuron of a downstream block: the strongest match (|cosine|) between its input vector (row of gate_proj, and row of
up_proj) and the write vector (column of down_proj) of any neuron in the blocks before it. Controls with the same number of
candidate vectors: (a) random directions of the residual stream, (b) the upstream write vectors mixed by a random rotation
inside each block (same subspaces, no individual neurons), (c) the downstream input vectors mixed by a random rotation.
Memory-lean: one upstream block at a time.
Usage: python neu_wiring.py <scratch_root> <out_dir>"""
import glob, json, re, sys, time
import numpy as np, torch
from safetensors import safe_open
root, out = sys.argv[1], sys.argv[2]; torch.manual_seed(0); t0 = time.time(); where = {}
for f in glob.glob(f"{root}/models/qwen3-1.7b/*.safetensors"):
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            q = re.search(r"layers\.(\d+)\.mlp\.(down_proj|gate_proj|up_proj)\.weight", k)
            if q: where[(int(q.group(1)), q.group(2))] = (f, k)
def load(l, name):
    f, k = where[(l, name)]
    with safe_open(f, framework="pt") as sf: return sf.get_tensor(k).float()
L = 1 + max(l for l, _ in where); G0 = load(0, "gate_proj"); NI, d = G0.shape; del G0; un = lambda M, dim: M / (M.norm(dim=dim, keepdim=True) + 1e-12)
Q = torch.linalg.qr(torch.randn(NI, NI))[0]; res = []
for tgt in (6, 14, 22, 27):
    R = {b: un(load(tgt, b), 1) for b in ("gate_proj", "up_proj")}; Rr = {b: un(Q @ load(tgt, b), 1) for b in R}
    best = {(b, c): torch.zeros(NI) for b in R for c in ("neurons", "random_directions", "rotated_upstream", "rotated_downstream")}
    for l in range(tgt):
        Wl = load(l, "down_proj"); Wn = un(Wl, 0); Wrot = un(Wl @ Q.T, 0); Wrand = un(torch.randn(d, NI), 0); del Wl
        for b in R:
            for c, (rd, wr) in (("neurons", (R[b], Wn)), ("random_directions", (R[b], Wrand)), ("rotated_upstream", (R[b], Wrot)), ("rotated_downstream", (Rr[b], Wn))):
                best[(b, c)] = torch.maximum(best[(b, c)], (rd @ wr).abs().max(1).values)
    for b in R:
        r = dict(block=tgt, branch=b, upstream_units=tgt * NI)
        for c in ("neurons", "random_directions", "rotated_upstream", "rotated_downstream"):
            v = best[(b, c)]; r[f"{c}_median"] = float(v.median()); r[f"{c}_share_above_0.3"] = float((v > .3).float().mean()); r[f"{c}_share_above_0.2"] = float((v > .2).float().mean()); r[f"{c}_p95"] = float(v.quantile(.95))
        res.append(r); print(f"block {tgt} {b[:-5]:4s} ({tgt * NI} upstream neurons): strongest match, median / share above 0.3 -> neurons {r['neurons_median']:.3f} / {r['neurons_share_above_0.3']:.3f} | random directions {r['random_directions_median']:.3f} / "
                             f"{r['random_directions_share_above_0.3']:.3f} | upstream rotated {r['rotated_upstream_median']:.3f} / {r['rotated_upstream_share_above_0.3']:.3f} | downstream rotated {r['rotated_downstream_median']:.3f} / {r['rotated_downstream_share_above_0.3']:.3f}   ({time.time() - t0:.0f}s)", flush=True)
json.dump(res, open(f"{out}/neu_wiring.json", "w"), indent=1)
