"""Critique check: is the toy's Lambda (zero-shot recipe accuracy) drop away from the natural delta specific to delta, or does an
off-target regulariser with a similar cost (Fisher ratio of h2 clouds grouped by the x input, applied late like GLATE) do the same?
Also: an early weak task regulariser (from step 1) instead of the late one. Lambda as in lattice.py."""
import sys, os, time, itertools, json
import numpy as np, torch, torch.nn.functional as Fn
SP = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SP + "/ufo_scripts"); sys.path.insert(0, SP + "/rgi_design")
torch.set_num_threads(1)
import pc_meta as M
from pc_common import accuracy
from rgi_pilot import mk, acc_cols
P, NPRE, NNEW = M.P, M.NPRE, M.NNEW

def train(ws, init, mode, gamma, W=128, extra_after=0):
    """mode: 'nat' | 'task_late' | 'x_late' | 'out_late' | 'task_early'. Returns model and info."""
    pre, new, pairs, splits = M.world(ws)
    tr = [M.data(pre[k], k, pairs[splits[k][:144]]) for k in range(NPRE)]; va = [M.data(pre[k], k, pairs[splits[k][144:]]) for k in range(NPRE)]
    Xtr = torch.cat([d[0] for d in tr]); ytr = torch.cat([d[1] for d in tr]); Xva = torch.cat([d[0] for d in va]); yva = torch.cat([d[1] for d in va])
    lab_task = torch.arange(NPRE).repeat_interleave(144); lab_x = torch.cat([torch.as_tensor(pairs[splits[k][:144]][:, 0]) for k in range(NPRE)])
    torch.manual_seed(ws if init == 0 else 10_000 * init + ws); m = mk("bp_mup", W); opt = M.make_opt(m)
    first = ck = None; t0 = time.time(); after = M.AFTER + extra_after
    for s in range(1, M.MAXSTEPS + after + 1):
        if s > M.MAXSTEPS and first is None: break
        on = (mode == "task_early") or (mode != "nat" and first is not None)
        opt.zero_grad(); h1, h2 = m.acts(Xtr); loss = Fn.cross_entropy(m.head(h2), ytr)
        if on and gamma:
            if mode in ("task_late", "task_early"): loss = loss + gamma * torch.log(M.task_fisher(h2, lab_task, NPRE))
            elif mode == "x_late": loss = loss + gamma * torch.log(M.task_fisher(h2, lab_x, P))
            elif mode == "out_late": loss = loss + gamma * torch.log(M.task_fisher(h1, lab_task, NPRE))
        loss.backward(); opt.step()
        if s % 50 == 0 and first is None and accuracy(m, Xva, yva) >= M.LEVEL1: first = s
        if first is not None and s >= first + after: ck = s; break
    return m, dict(first=first, ck=ck, heldout=accuracy(m, Xva, yva), sec=time.time() - t0)

def lam(m, ws):
    pre, new, pairs, splits = M.world(ws); C = m.l1.weight.detach().numpy()[:, 2 * P: 2 * P + NPRE].astype(float); accs = []
    for j in range(NNEW):
        xy = pairs[splits[NPRE + j][8:]]; yy = torch.as_tensor((xy[:, 0] + xy[:, 1] + new[j]) % P)
        V = [C[:, a] + C[:, b] - C[:, c] for a, b, c in itertools.product(range(NPRE), repeat=3) if a < b and c not in (a, b) and (pre[a] + pre[b] - pre[c]) % P == new[j]]
        accs.append(acc_cols(m, xy, yy, torch.tensor(np.stack(V), dtype=torch.float32)).mean())
    return float(np.mean(accs))

def xdelta(m, ws):
    pre, new, pairs, splits = M.world(ws); X = torch.cat([M.enc(pairs, k) for k in range(NPRE)]); lx = torch.as_tensor(np.tile(pairs[:, 0], NPRE))
    with torch.no_grad(): h1, h2 = m.acts(X)
    return float(M.task_fisher(h2.double(), lx, P))

def pdist(m, m0):
    a = torch.cat([p.detach().flatten() for p in m.parameters()]); b = torch.cat([p.detach().flatten() for p in m0.parameters()]); return float((a - b).norm() / b.norm())

out = []; specs = [("nat", 0.0)] + [("task_late", g) for g in (0.01, -0.01, -0.03)] + [("x_late", g) for g in (0.01, 0.03, -0.01, -0.03, -0.1)] + [("task_early", g) for g in (0.003, -0.003, 0.01, -0.01)]
for ws in (0, 1):
    for init in (0, 1):
        base = None
        for mode, g in specs:
            m, info = train(ws, init, mode, g)
            if mode == "nat": base = m
            pre, new, pairs, splits = M.world(ws)
            d = M.delta_stats(m, pre, pairs, ws) if info["ck"] else None
            row = dict(ws=ws, init=init, mode=mode, gamma=g, heldout=info["heldout"], first=info["first"], ck=info["ck"],
                       delta_h2=d["h2"]["delta"] if d else None, delta_h1=d["h1"]["delta"] if d else None, F_x=xdelta(m, ws), Lambda=lam(m, ws) if info["ck"] else None,
                       pdist=pdist(m, base) if (base is not None and mode != "nat" and info["ck"]) else 0.0, sec=info["sec"])
            out.append(row)
            print(f"w{ws} i{init} {mode:10s} g {g:+.3f}: held-out {row['heldout']:.3f} first95 {row['first']} | delta h2 {row['delta_h2'] if row['delta_h2'] is None else round(row['delta_h2'], 3)} F_x(h2) {row['F_x']:.3f} | Lambda {row['Lambda'] if row['Lambda'] is None else round(row['Lambda'], 3)} | rel. param distance to natural {row['pdist']:.3f} ({row['sec']:.0f}s)", flush=True)
json.dump(out, open("sham.json", "w"), indent=1)
