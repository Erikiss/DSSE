"""Critique check 4 (toy, 1 thread): are 'trajectory modes' (SVD of the snapshot trajectory of the task code, design T6) primitives or
the known PCA-of-trajectories artefact (Antognini and Sohl-Dickstein 2018: the i-th PC time course of a random walk is ~cos(pi i t / T))?
Trains BP-muP W=128 as pc_meta (seed ws), snapshots of the task columns every 25 updates, centred over tasks (design T6). Reports for
the top 5 modes: variance share, |corr| of the time course with cos(pi i t / T) (Lissajous template), and the emergence time (first
snapshot at which the coefficient reaches half its final value). Null: the same increments in shuffled time order (a random walk with
the same step distribution). Usage: python c4_traj.py <ws>"""
import sys, json
import numpy as np, torch
SP = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SP + "/ufo_scripts"); torch.set_num_threads(1)
import pc_meta as M
from pc_common import MLP, accuracy
P, NPRE, NSLOT, BASE, W = M.P, M.NPRE, M.NSLOT, M.BASE, 128


def modes(X, k=5):
    Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False); var = S ** 2 / (S ** 2).sum(); T = len(X); t = np.arange(T)
    out = []
    for i in range(k):
        tc = U[:, i] * S[i]; lis = np.cos(np.pi * (i + 1) * (t + 0.5) / T); r = abs(np.corrcoef(tc, lis)[0, 1])
        fin = tc[-1] - tc[0]; half = np.argmax(np.abs(tc - tc[0]) >= 0.5 * abs(fin)) if abs(fin) > 0 else -1
        out.append(dict(var=round(float(var[i]), 3), lissajous_r=round(float(r), 3), emerge_snapshot=int(half)))
    return out, float(var[:5].sum())


ws = int(sys.argv[1]); pre, new, pairs, splits = M.world(ws)
tr = [M.data(pre[k], k, pairs[splits[k][:144]]) for k in range(NPRE)]; va = [M.data(pre[k], k, pairs[splits[k][144:]]) for k in range(NPRE)]
Xtr = torch.cat([d[0] for d in tr]); ytr = torch.cat([d[1] for d in tr]); Xva = torch.cat([d[0] for d in va]); yva = torch.cat([d[1] for d in va])
torch.manual_seed(ws); m = MLP(2 * P + NSLOT, W, P, param="mup", base_width=BASE); opt = M.make_opt(m); first = None; snaps, acc = [], []
for s in range(0, M.MAXSTEPS + M.AFTER + 1):
    if s > 0: M.bp_update(m, opt, Xtr, ytr)
    if s % 25 == 0:
        T = m.l1.weight[:, 2 * P:2 * P + NPRE].detach().double().numpy(); snaps.append((T - T.mean(1, keepdims=True)).ravel()); acc.append(accuracy(m, Xva, yva))
        if first is None and acc[-1] >= M.LEVEL1: first = s
    if first is not None and s >= first + M.AFTER: break
X = np.stack(snaps); real, v5 = modes(X)
g = np.random.default_rng(ws); dX = np.diff(X, axis=0); nulls = []
for _ in range(20):
    Xn = np.concatenate([X[:1], X[:1] + np.cumsum(dX[g.permutation(len(dX))], 0)]); nulls.append(modes(Xn)[0])
null_lis = [round(float(np.mean([n[i]["lissajous_r"] for n in nulls])), 3) for i in range(5)]; null_var = [round(float(np.mean([n[i]["var"] for n in nulls])), 3) for i in range(5)]
jump = int(np.argmax(np.diff(acc))) if len(acc) > 1 else -1
print(f"[{ws}] snapshots {len(X)} first95 {first}; acc jump at snapshot {jump}; top-5 var {v5:.3f}")
print(f"   real modes: " + " | ".join(f"var {r['var']:.3f} lis {r['lissajous_r']:.2f} emerge {r['emerge_snapshot']}" for r in real))
print(f"   shuffled-increment null (20): var {null_var} lissajous r {null_lis}")
json.dump(dict(ws=ws, n=len(X), first95=first, acc=acc, real=real, null_var=null_var, null_lis=null_lis), open(f"{SP}/unlearn_design/critique/c4_traj_s{ws}.json", "w"), indent=1)
