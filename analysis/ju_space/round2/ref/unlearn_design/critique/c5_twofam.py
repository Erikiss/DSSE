"""Critique check 5 (toy, 1 thread): a retain family for 'specific' unlearning. Two task families on the same x, y inputs:
'add' (x + y + c) mod 17 and 'sub' (x - y + c) mod 17, 12 shifts each (the pc_meta shifts of seed ws for both), one-hot task slots
(24 pretraining + 4 new add + 4 new sub). BP-muP W=128 as pc_meta (AdamW, full batch, checkpoint 500 updates after held-out >= 0.95
on both families). Then: block the 11-dim span of the centred add-task columns in z1 (as pilot.py 'block'); report add and sub
held-out accuracy (forget vs retain), and the column search (8 shots, 300 steps, Adam 1e-2, from zero) for the 4 new add and the 4
new sub tasks with the block on and off: is the un-searching specific to the blocked family?
Usage: python c5_twofam.py <ws>"""
import sys, json, time, math
import numpy as np, torch, torch.nn.functional as Fn
SP = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SP + "/ufo_scripts"); torch.set_num_threads(1)
import pc_meta as M
from pc_common import MLP
P, BASE, W, NT = M.P, M.BASE, 128, 12
NSL = 2 * NT + 8; DIN = 2 * P + NSL
EVAL = (10, 20, 40, 80, 160, 300)


def lab(fam, xy, c): return (xy[:, 0] + (xy[:, 1] if fam == "add" else -xy[:, 1]) + c) % P


def fwd(m, xy, col, U=None, ref=None):
    W1 = m.l1.weight; z1 = W1[:, xy[:, 0]].T + W1[:, P + xy[:, 1]].T + col + m.l1.bias
    if U is not None: z1 = z1 - ((z1 - ref) @ U) @ U.T
    return m.head(torch.tanh(m.l2(torch.tanh(z1))))


def main():
    ws = int(sys.argv[1]); pre, new, pairs, splits = M.world(ws); t0 = time.time(); g = np.random.default_rng(ws)
    tasks = [("add", c, i) for i, c in enumerate(pre)] + [("sub", c, NT + i) for i, c in enumerate(pre)]
    newt = [("add", c, 2 * NT + j) for j, c in enumerate(new)] + [("sub", c, 2 * NT + 4 + j) for j, c in enumerate(new)]
    spl = lambda slot: splits[slot % len(splits)] if slot < len(splits) else np.random.default_rng(3000 + 31 * ws + slot).permutation(P * P)
    def enc(xy, slot):
        X = torch.zeros(len(xy), DIN); r = torch.arange(len(xy)); X[r, torch.as_tensor(xy[:, 0])] = 1; X[r, P + torch.as_tensor(xy[:, 1])] = 1; X[r, 2 * P + slot] = 1; return X
    Xtr = torch.cat([enc(pairs[spl(s)[:144]], s) for f, c, s in tasks]); ytr = torch.cat([torch.as_tensor(lab(f, pairs[spl(s)[:144]], c)) for f, c, s in tasks])
    va = {s: (pairs[spl(s)[144:]], f, c) for f, c, s in tasks}
    torch.manual_seed(ws); m = MLP(DIN, W, P, param="mup", base_width=BASE); opt = M.make_opt(m); first = None; s = 0
    def acc_fam(fam, U=None, ref=None):
        a = []
        for f, c, sl in tasks:
            if f != fam: continue
            xy, _, _ = va[sl]
            with torch.no_grad(): a.append(float((fwd(m, torch.as_tensor(xy), m.l1.weight[:, 2 * P + sl], U, ref).argmax(-1).numpy() == lab(f, xy, c)).mean()))
        return float(np.mean(a))
    for s in range(1, 4000 + M.AFTER + 1):
        if s > 4000 and first is None: break
        M.bp_update(m, opt, Xtr, ytr)
        if s % 50 == 0 and first is None and min(acc_fam("add"), acc_fam("sub")) >= M.LEVEL1: first = s
        if first is not None and s >= first + M.AFTER: break
    R = dict(ws=ws, first95=first, steps=s, add=acc_fam("add"), sub=acc_fam("sub"), sec_train=time.time() - t0)
    print(f"[{ws}] two families: first95 {first} steps {s} held-out add {R['add']:.3f} sub {R['sub']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    m.eval()
    for p_ in m.parameters(): p_.requires_grad_(False)
    W1 = m.l1.weight.detach().double().numpy(); Ta = W1[:, [2 * P + sl for f, c, sl in tasks if f == "add"]]; Ts = W1[:, [2 * P + sl for f, c, sl in tasks if f == "sub"]]
    Qa = torch.as_tensor(np.linalg.qr(Ta - Ta.mean(1, keepdims=True))[0][:, :NT - 1], dtype=torch.float32)
    Qr = torch.as_tensor(np.linalg.qr(g.standard_normal((W, NT - 1)))[0], dtype=torch.float32)
    ref = torch.as_tensor(np.concatenate([Ta, Ts], 1).mean(1), dtype=torch.float32) + m.l1.bias
    ref = (m.l1.weight[:, torch.as_tensor(pairs[:, 0])].T + m.l1.weight[:, P + torch.as_tensor(pairs[:, 1])].T).mean(0) + ref
    sa = Ts - Ts.mean(1, keepdims=True); R["sub_cols_in_addspan"] = float(((Qa.double().numpy().T @ sa) ** 2).sum() / (sa ** 2).sum())
    def search(f, c, sl, U=None):
        idx = spl(sl); xs = torch.as_tensor(pairs[idx[:8]]); ys = torch.as_tensor(lab(f, pairs[idx[:8]], c)); xh = torch.as_tensor(pairs[idx[8:]]); yh = lab(f, pairs[idx[8:]], c)
        z = torch.zeros(W, requires_grad=True); o = torch.optim.Adam([z], lr=1e-2, betas=(0.9, 0.98)); out = []
        for st in range(1, 301):
            o.zero_grad(); Fn.cross_entropy(fwd(m, xs, z, U, ref), ys).backward(); o.step()
            if st in EVAL:
                with torch.no_grad(): out.append(float((fwd(m, xh, z, U, ref).argmax(-1).numpy() == yh).mean()))
        return float(np.mean(out))
    for nm, U in (("none", None), ("block_add11", Qa), ("block_rand11", Qr)):
        R[nm] = dict(add=acc_fam("add", U, ref), sub=acc_fam("sub", U, ref),
                     search_add=float(np.mean([search(f, c, sl, U) for f, c, sl in newt if f == "add"])), search_sub=float(np.mean([search(f, c, sl, U) for f, c, sl in newt if f == "sub"])),
                     relearn_add_old4=float(np.mean([search(f, c, sl, U) for f, c, sl in tasks[:4]])))
        print(f"    {nm:12s}: old add {R[nm]['add']:.3f} old sub {R[nm]['sub']:.3f} | new-task search add {R[nm]['search_add']:.3f} sub {R[nm]['search_sub']:.3f} | relearn old add (col search) {R[nm]['relearn_add_old4']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    print(f"    share of the sub columns inside the add span: {R['sub_cols_in_addspan']:.3f} (random {(NT - 1) / W:.3f})", flush=True)
    json.dump(R, open(f"{SP}/unlearn_design/critique/c5_twofam_s{ws}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
