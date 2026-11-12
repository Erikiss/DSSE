"""Design pilot: impose the primitives. The task code is no longer a free one-hot column but t(c) = t0 + A phi(c) with a fixed
2k-dim code phi of the true shift c: 'ring' = Fourier modes f = 1..k of Z_17 (a composition-length cap of k modes on a group
structure), 'rand' = a random 2k-dim subspace of the centred functions on Z_17 with the same column norms (same dimension, no
group structure). k = 8 is the full space for both (sanity). World, training, checkpoint and muP as pc_meta.py (BP).
Measures: level 1 (held-out on the 12 pretraining shifts), zero-shot accuracy on the 4 new shifts with t0 + A phi(c_new), search
in code space (q from 0, t = t0 + A q, 8 shots, Adam 300 steps, best of lr 1e-2 / 1e-1; mean held-out accuracy over steps
10..300, as A in B2), free column search (as B2), cosine of the found q with phi(c_new).
Usage: python impose.py <ws> <k,...> [W]"""
import sys, os, json, time, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
SP = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SP + "/ufo_scripts")
torch.set_num_threads(1)
import pc_meta as M
P, NPRE, NNEW, BASE = M.P, M.NPRE, M.NNEW, M.BASE
OUT = SP + "/unlearn_design"; EVAL = (10, 20, 40, 80, 160, 300)


def code_table(kind, k, seed):
    c = np.arange(P)
    F = np.concatenate([np.stack([np.cos(2 * np.pi * f * c / P), np.sin(2 * np.pi * f * c / P)], 1) for f in range(1, k + 1)], 1)   # 17 x 2k
    if kind == "ring": return torch.as_tensor(F, dtype=torch.float32)
    g = np.random.default_rng(900 + seed); one = np.ones((P, 1)) / np.sqrt(P)
    Q, _ = np.linalg.qr(g.standard_normal((P, 2 * k)) - one @ (one.T @ g.standard_normal((P, 2 * k))))
    Q = Q - one @ (one.T @ Q); Q, _ = np.linalg.qr(Q)
    return torch.as_tensor(Q * np.linalg.norm(F, axis=0), dtype=torch.float32)       # same column norms as the Fourier table


class Net(nn.Module):
    def __init__(s, W, phi):
        super().__init__(); din = 2 * P + M.NSLOT; s.phi = phi; r = phi.shape[1]
        s.Ex = nn.Parameter(torch.randn(W, 2 * P) / math.sqrt(din)); s.t0 = nn.Parameter(torch.zeros(W)); s.A = nn.Parameter(torch.randn(W, r) / math.sqrt(din) / math.sqrt(r / 2))
        s.b1 = nn.Parameter(torch.zeros(W)); s.l2 = nn.Linear(W, W); s.l3 = nn.Linear(W, P); s.W = W
        nn.init.normal_(s.l2.weight, std=1 / math.sqrt(W)); nn.init.zeros_(s.l2.bias); nn.init.zeros_(s.l3.weight); nn.init.zeros_(s.l3.bias)

    def z1(s, xy, t):
        return s.Ex[:, xy[:, 0]].T + s.Ex[:, P + xy[:, 1]].T + t + s.b1

    def out(s, xy, t):
        return s.l3(torch.tanh(s.l2(torch.tanh(s.z1(xy, t))))) * (BASE / s.W)

    def task(s, c):
        return s.t0 + s.phi[c] @ s.A.T

    def groups(s, lr):
        return [dict(params=[s.l2.weight], lr=lr * BASE / s.W, weight_decay=M.DECAY / (lr * BASE / s.W)),
                dict(params=[s.Ex, s.t0, s.A, s.b1, s.l2.bias, s.l3.weight, s.l3.bias], lr=lr, weight_decay=M.DECAY / lr)]


def acc(net, xy, c, t):
    with torch.no_grad(): return float((net.out(xy, t).argmax(-1) == torch.as_tensor((xy.numpy().sum(1) + c) % P)).float().mean())


def run(ws, kind, k, W=128):
    pre, new, pairs, splits = M.world(ws); phi = code_table(kind, k, ws); torch.manual_seed(ws); net = Net(W, phi)
    opt = torch.optim.AdamW(net.groups(M.LR), lr=M.LR, betas=M.BETAS)
    xtr = torch.as_tensor(np.concatenate([pairs[splits[j][:144]] for j in range(NPRE)])); ctr = torch.as_tensor(np.repeat(pre, 144)); ytr = (xtr.sum(1) + ctr) % P
    xva = [torch.as_tensor(pairs[splits[j][144:]]) for j in range(NPRE)]; first = None; ck = None; t0 = time.time()
    for s in range(1, M.MAXSTEPS + M.AFTER + 1):
        if s > M.MAXSTEPS and first is None: break
        opt.zero_grad(); Fn.cross_entropy(net.out(xtr, net.task(ctr)), ytr).backward(); opt.step()
        if s % 50 == 0 and first is None:
            if np.mean([acc(net, xva[j], pre[j], net.task(torch.tensor(pre[j]))) for j in range(NPRE)]) >= M.LEVEL1: first = s
        if first is not None and s >= first + M.AFTER: ck = s; break
    lvl = float(np.mean([acc(net, xva[j], pre[j], net.task(torch.tensor(pre[j]))) for j in range(NPRE)]))
    R = dict(ws=ws, kind=kind, k=k, first95=first, ckpt=ck, level1_acc=lvl, sec_pre=time.time() - t0)
    zs, codeA, freeA, cosq = [], [], [], []
    for p_ in net.parameters(): p_.requires_grad_(False)
    for j in range(NNEW):
        slot = NPRE + j; idx = splits[slot]; xs = torch.as_tensor(pairs[idx[:8]]); ys = (xs.sum(1) + new[j]) % P; xh = torch.as_tensor(pairs[idx[8:]])
        zs.append(acc(net, xh, new[j], net.task(torch.tensor(new[j]))))
        with torch.no_grad(): pz = net.out(xh, net.task(torch.tensor(new[j]))).argmax(-1).numpy()
        e = (pz - xh.numpy().sum(1)) % P; cnt = np.bincount(e, minlength=P); md = int(cnt.argmax()); dist = lambda a, b: min((a - b) % P, (b - a) % P)
        R.setdefault("zs_modal", []).append(dict(c_new=new[j], modal=md, share=float(cnt.max() / len(e)), ring_dist_to_new=dist(md, new[j]), modal_is_trained=md in pre,
                                                  nearest_trained_dist=min(dist(new[j], c) for c in pre)))
        best = (-1, None)
        for lr in (1e-2, 1e-1):
            q = torch.zeros(phi.shape[1], requires_grad=True); o = torch.optim.Adam([q], lr=lr, betas=M.BETAS); a = []
            for s in range(1, 301):
                o.zero_grad(); Fn.cross_entropy(net.out(xs, net.t0 + net.A @ q), ys).backward(); o.step()
                if s in EVAL: a.append(acc(net, xh, new[j], net.t0 + net.A @ q))
            if np.mean(a) > best[0]: best = (float(np.mean(a)), q.detach().clone())
        codeA.append(best[0]); qv = best[1]; tgt = phi[new[j]]; cosq.append(float(qv @ tgt / (qv.norm() * tgt.norm() + 1e-12)))
        v = torch.zeros(W, requires_grad=True); o = torch.optim.Adam([v], lr=1e-2, betas=M.BETAS); a = []
        for s in range(1, 301):
            o.zero_grad(); Fn.cross_entropy(net.out(xs, v), ys).backward(); o.step()
            if s in EVAL: a.append(acc(net, xh, new[j], v))
        freeA.append(float(np.mean(a)))
    R.update(zero_shot=float(np.mean(zs)), code_search=float(np.mean(codeA)), free_search=float(np.mean(freeA)), cos_q_phi=float(np.mean(cosq)), sec=time.time() - t0)
    print(f"[{ws}] {kind:4s} k={k}: zs-modal " + str([(d['c_new'], d['modal'], round(d['share'], 2), d['ring_dist_to_new'], d['modal_is_trained']) for d in R['zs_modal']]), flush=True)
    print(f"[{ws}] {kind:4s} k={k}: first95 {first} level1 {lvl:.3f} | zero-shot {R['zero_shot']:.3f} code-search {R['code_search']:.3f} (cos q,phi {R['cos_q_phi']:+.2f}) free-search {R['free_search']:.3f} ({R['sec']:.0f}s)", flush=True)
    return R


if __name__ == "__main__":
    ws = int(sys.argv[1]); ks = [int(v) for v in sys.argv[2].split(",")]; W = int(sys.argv[3]) if len(sys.argv) > 3 else 128; res = []
    for k in ks:
        for kind in ("ring", "rand"):
            res.append(run(ws, kind, k, W))
            if k == 8: break                       # k = 8 is the full space for both codes
    json.dump(res, open(f"{OUT}/impose2_s{ws}_W{W}.json", "w"), indent=1)
