"""Critique check 3 (toy, 1 thread): a positive control in which a group pattern CAN linearise the shift family.
x, y and the shift c all enter as Fourier features over Z_17 (frequencies 1..K, i.e. points on circles): z1 = Ax f(x) + Ay f(y) +
Ac f(c) + b. Adding one unit shift is then a fixed rotation of f(c) (a linear map), so an equivariant net could solve unseen shifts
zero-shot. Arms: 'ring' (f(c) = Fourier code of c), 'pring' (the same 17 codes assigned to the shifts by a fixed random permutation:
same geometry, no group alignment), 'onehot' (x, y one-hot as in pc_meta, c ring-coded = impose.py 'ring' at the same K).
World, split, optimiser, muP scaling, checkpoint rule as pc_meta / impose.py (12 pretraining shifts, 4 new; 144 / 145 pairs).
Measures: level 1 (held-out on the 12 shifts), steps to 0.95, zero-shot accuracy on the 4 new shifts (held-out pairs of their split),
and, with arc=1, pretraining on 6 consecutive shifts and zero-shot at generator distance j = 1..5 from the arc.
Usage: python c3_angle.py <ws> <K> <arm,...> [arc]"""
import sys, json, time, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
SP = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SP + "/ufo_scripts"); torch.set_num_threads(1)
import pc_meta as M
P, NPRE, NNEW, BASE, W = M.P, M.NPRE, M.NNEW, M.BASE, 128


def four(K):
    c = np.arange(P); return torch.as_tensor(np.concatenate([np.stack([np.cos(2 * np.pi * f * c / P), np.sin(2 * np.pi * f * c / P)], 1) for f in range(1, K + 1)], 1), dtype=torch.float32)


class Net(nn.Module):
    def __init__(s, fx, fc):
        super().__init__(); s.fx, s.fc = fx, fc; dx, dc = fx.shape[1], fc.shape[1]; din = 2 * dx + dc
        s.Ax = nn.Parameter(torch.randn(W, dx) / math.sqrt(din)); s.Ay = nn.Parameter(torch.randn(W, dx) / math.sqrt(din)); s.Ac = nn.Parameter(torch.randn(W, dc) / math.sqrt(din))
        s.b1 = nn.Parameter(torch.zeros(W)); s.l2 = nn.Linear(W, W); s.l3 = nn.Linear(W, P)
        nn.init.normal_(s.l2.weight, std=1 / math.sqrt(W)); nn.init.zeros_(s.l2.bias); nn.init.zeros_(s.l3.weight); nn.init.zeros_(s.l3.bias)

    def out(s, xy, c):
        z1 = s.fx[xy[:, 0]] @ s.Ax.T + s.fx[xy[:, 1]] @ s.Ay.T + s.fc[c] @ s.Ac.T + s.b1; return s.l3(torch.tanh(s.l2(torch.tanh(z1)))) * (BASE / W)

    def groups(s, lr):
        return [dict(params=[s.l2.weight], lr=lr * BASE / W, weight_decay=M.DECAY / (lr * BASE / W)), dict(params=[s.Ax, s.Ay, s.Ac, s.b1, s.l2.bias, s.l3.weight, s.l3.bias], lr=lr, weight_decay=M.DECAY / lr)]


def acc(net, xy, c):
    with torch.no_grad(): return float((net.out(xy, torch.full((len(xy),), c)).argmax(-1) == torch.as_tensor((xy.numpy().sum(1) + c) % P)).float().mean())


def run(ws, K, arm, arc=False):
    pre, new, pairs, splits = M.world(ws)
    if arc:
        L = int(__import__("os").environ.get("ARCL", "6")); c0 = pre[0]; pre = [(c0 + i) % P for i in range(L)]; test = [c for c in range(P) if c not in pre]
    else: test = new
    F = four(K); fx = torch.eye(P) if arm == "onehot" else F
    fc = F[torch.as_tensor(np.random.default_rng(700 + ws).permutation(P))] if arm == "pring" else F
    torch.manual_seed(ws); net = Net(fx, fc); opt = torch.optim.AdamW(net.groups(M.LR), lr=M.LR, betas=M.BETAS)
    sp = lambda c: splits[c % len(splits)]                         # a fixed split per shift value (shift values index the stored splits)
    xtr = torch.as_tensor(np.concatenate([pairs[sp(c)[:144]] for c in pre])); ctr = torch.as_tensor(np.repeat(pre, 144)); ytr = (xtr.sum(1) + ctr) % P
    xva = {c: torch.as_tensor(pairs[sp(c)[144:]]) for c in pre}; first = None; t0 = time.time(); s = 0
    for s in range(1, M.MAXSTEPS + M.AFTER + 1):
        if s > M.MAXSTEPS and first is None: break
        opt.zero_grad(); Fn.cross_entropy(net.out(xtr, ctr), ytr).backward(); opt.step()
        if s % 50 == 0 and first is None and np.mean([acc(net, xva[c], c) for c in pre]) >= M.LEVEL1: first = s
        if first is not None and s >= first + M.AFTER: break
    lvl = float(np.mean([acc(net, xva[c], c) for c in pre])); zs = {int(c): acc(net, torch.as_tensor(pairs[sp(c)[8:]]), c) for c in test}
    R = dict(ws=ws, K=K, arm=arm, arc=arc, first95=first, steps=s, level1=lvl, zero_shot=zs, zs_mean=float(np.mean(list(zs.values()))), sec=time.time() - t0)
    if arc:
        L = len(pre); dist = lambda c: min((c - pre[-1]) % P, (pre[0] - c) % P); R["arcL"] = L
        R["zs_by_j"] = [float(np.mean([zs[c] for c in test if dist(c) == j])) for j in range(1, (P - L) // 2 + 2) if any(dist(c) == j for c in test)]
    print(f"[{ws}] K={K} {arm:6s} arc={int(arc)}: first95 {first} steps {s} level1 {lvl:.3f} zero-shot mean {R['zs_mean']:.3f}" + (f" by j {[round(v, 2) for v in R['zs_by_j']]}" if arc else f" {[round(v, 2) for v in zs.values()]}") + f" ({R['sec']:.0f}s)", flush=True)
    return R


if __name__ == "__main__":
    ws, K = int(sys.argv[1]), int(sys.argv[2]); arms = sys.argv[3].split(","); arc = len(sys.argv) > 4 and sys.argv[4] == "1"
    res = [run(ws, K, a, arc) for a in arms]
    json.dump(res, open(f"{SP}/unlearn_design/critique/c3_angle_s{ws}_K{K}_arc{int(arc)}" + (f"_L{__import__('os').environ.get('ARCL', '6')}" if arc else "") + ".json", "w"), indent=1)
