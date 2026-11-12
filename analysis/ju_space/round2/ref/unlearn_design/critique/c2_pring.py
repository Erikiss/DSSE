"""Critique check 2 (toy, 1 thread): is the ring code's worse learnability in impose.py a group effect or a geometry effect?
Adds 'pring' = the ring (Fourier f = 1..k) table with its 17 rows permuted by a fixed random permutation of Z_17: the same point
set (norms, pairwise distances, curve shape) assigned to the shifts at random. ring vs pring isolates group alignment; pring vs rand
isolates the geometry of the point set. Also reports the geometry of the 12 trained codes (min / mean pairwise distance).
Usage: python c2_pring.py <ws> <k,...> [W]"""
import sys, json, time
import numpy as np, torch
SP = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SP + "/unlearn_design"); torch.set_num_threads(1)
import impose as I
_orig = I.code_table


def code_table(kind, k, seed):
    if kind != "pring": return _orig(kind, k, seed)
    F = _orig("ring", k, seed); perm = np.random.default_rng(700 + seed).permutation(I.P); return F[torch.as_tensor(perm)]


I.code_table = code_table


def geom(kind, k, ws):
    pre = I.M.world(ws)[0]; F = code_table(kind, k, ws).numpy()[pre]; D = np.linalg.norm(F[:, None] - F[None], axis=-1); iu = np.triu_indices(len(pre), 1)
    return float(D[iu].min()), float(D[iu].mean())


if __name__ == "__main__":
    ws = int(sys.argv[1]); ks = [int(v) for v in sys.argv[2].split(",")]; W = int(sys.argv[3]) if len(sys.argv) > 3 else 128; res = []
    for k in ks:
        for kind in ("ring", "pring", "rand"):
            mn, me = geom(kind, k, ws); r = I.run(ws, kind, k, W); r.update(min_dist=mn, mean_dist=me); res.append(r)
            print(f"    [{ws}] {kind:5s} k={k}: min pairwise dist {mn:.2f} mean {me:.2f}", flush=True)
    json.dump(res, open(f"{SP}/unlearn_design/critique/c2_pring_s{ws}_W{W}.json", "w"), indent=1)
