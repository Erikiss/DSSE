"""Positive control for the transfer machinery: a forced task geometry. The task slot enters through a rank-dt factorisation
U (W x dt) E (dt x 17) instead of free columns; everything else as pc_meta.py (bp_mup, AdamW, checkpoint 500 updates after
held-out >= 0.95). A new task is learned by training only its dt coefficients E[:, slot] (8 shots, 300 Adam steps, lr 1e-2, from
zero); its column is U E[:, slot]. Optional gamma: the delta regulariser of pc_meta.py on h2.
Usage: python rgi_bneck.py <ws> <W> <dt> <inits comma> [gamma ...]"""
import sys, os, math, time, itertools, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
SP = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SP + "/rgi_design")
from rgi_lib import *


M.LEVEL1 = float(os.environ.get("L1", M.LEVEL1)); M.MAXSTEPS = int(os.environ.get("MAXSTEPS", M.MAXSTEPS))


class MLPB(MLP):
    def __init__(s, W, dt):
        super().__init__(2 * P + NSLOT, W, P, param="mup", base_width=M.BASE); s.dt = dt
        s.U = nn.Parameter(torch.randn(W, dt)); s.E = nn.Parameter(torch.randn(dt, NSLOT) * (1 / math.sqrt(2 * P + NSLOT)) / math.sqrt(dt))
        with torch.no_grad(): s.l1.weight[:, 2 * P:] = 0

    def slot_cols(s):
        return s.U @ s.E

    def acts(s, x):
        z = Fn.linear(x[:, :2 * P], s.l1.weight[:, :2 * P], s.l1.bias) + x[:, 2 * P:] @ s.slot_cols().T
        h1 = torch.tanh(z); return h1, torch.tanh(s.l2(h1))


def train_b(ws, W, dt, init, gamma=0.0):
    pre, new, pairs, splits = M.world(ws)
    tr = [M.data(pre[k], k, pairs[splits[k][:144]]) for k in range(NPRE)]; va = [M.data(pre[k], k, pairs[splits[k][144:]]) for k in range(NPRE)]
    Xtr = torch.cat([d[0] for d in tr]); ytr = torch.cat([d[1] for d in tr]); Xva = torch.cat([d[0] for d in va]); yva = torch.cat([d[1] for d in va])
    lab = torch.arange(NPRE).repeat_interleave(144)
    torch.manual_seed(ws if init == 0 else 10_000 * init + ws); m = MLPB(W, dt); opt = M.make_opt(m); first = ck = None
    for s in range(1, M.MAXSTEPS + M.AFTER + 1):
        if s > M.MAXSTEPS and first is None: break
        M.bp_update(m, opt, Xtr, ytr, gamma, lab)
        if s % 50 == 0 and first is None and accuracy(m, Xva, yva) >= M.LEVEL1: first = s
        if first is not None and s >= first + M.AFTER: ck = s; break
    return m, dict(first95=first, ckpt=ck, heldout=accuracy(m, Xva, yva))


def geometry_b(m, W, ws):
    pre, new, pairs, splits = M.world(ws); g = dict(m=m, arm="bneck", W=W); g["delta"] = M.delta_stats(m, pre, pairs, ws)
    SC = m.slot_cols().detach().numpy().astype(float); C = SC[:, :NPRE]; g["C"] = C
    g["Aall"] = np.concatenate([m.l1.weight.detach().numpy()[:, :2 * P], C, m.l1.bias.detach().numpy()[:, None]], 1).astype(float)
    fc = fourier_conc(C, pre, nperm=200); g["fourier"] = {k: v for k, v in fc.items() if k != "spectrum"}; g["spectrum"] = fc["spectrum"]; g["xy_sparsity"] = xy_sparsity(m)
    g["v"], g["self"], g["span"], g["proj"], g["synth"] = [], [], [], [], []
    for j in range(NNEW):
        slot = NPRE + j; idx = splits[slot]; Xs, ys = M.data(new[j], slot, pairs[idx[:8]]); xy = pairs[idx[8:]]; yh = torch.as_tensor((xy[:, 0] + xy[:, 1] + new[j]) % P)
        E0 = m.E.detach().clone(); e = torch.zeros(m.dt, requires_grad=True); opt = torch.optim.Adam([e], lr=1e-2, betas=(0.9, 0.98))
        for p in m.parameters(): p.requires_grad_(False)
        for s in range(300):
            opt.zero_grad(); E = E0.clone(); E[:, slot] = e
            z = Fn.linear(Xs[:, :2 * P], m.l1.weight[:, :2 * P], m.l1.bias) + Xs[:, 2 * P:] @ (m.U @ E).T
            L = Fn.cross_entropy(m.head(torch.tanh(m.l2(torch.tanh(z)))), ys); L.backward(); opt.step()
        for p in m.parameters(): p.requires_grad_(True)
        v = (m.U @ e).detach().numpy().astype(float); g["v"].append(v)
        mA = C.mean(1); Q, _ = np.linalg.qr(C - mA[:, None]); vp = mA + Q @ (Q.T @ (v - mA)); vs = ring_synth(C, pre, new[j])
        a = acc_cols(m, xy, yh, torch.tensor(np.stack([v, vp, vs]), dtype=torch.float32)); g["self"].append(float(a[0])); g["proj"].append(float(a[1])); g["synth"].append(float(a[2]))
        g["span"].append(float(np.sum((Q.T @ (v - mA)) ** 2) / max(np.sum((v - mA) ** 2), 1e-12)))
    return g


if __name__ == "__main__":
    ws, W, dt = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); inits = [int(i) for i in sys.argv[4].split(",")]
    gammas = [float(x) for x in sys.argv[5:]] or [0.0]; Gs = {}; t0 = time.time()
    for init in inits:
        for gm in gammas:
            if init != inits[0] and gm != 0.0: continue
            m, info = train_b(ws, W, dt, init, gm); name = f"bneck{dt}_W{W}_i{init}_g{gm:g}"
            print(f"{name}: first95 {info['first95']} ckpt {info['ckpt']} held-out {info['heldout']:.3f} ({time.time() - t0:.0f}s)", flush=True)
            if info["ckpt"] is None: continue
            g = geometry_b(m, W, ws); Gs[name] = g; print("  " + fmt_geom(name, g), flush=True)
    for a, b in itertools.permutations(Gs, 2):
        print("  " + fmt_pair(a, b, pair_transfer(Gs[a], Gs[b], ws)), flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)
