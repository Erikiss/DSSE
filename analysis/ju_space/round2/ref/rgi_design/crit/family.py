"""Critique check 2: (a) natural variation: Lambda vs delta across 6 inits x 2 worlds and at a later checkpoint (first95 + 1500);
(b) fairness of 'another family': alpha learned in A (natural W128) over the anchors A and B share, carried into B whose (1) init
differs only, (2) training split differs, (3) input coding of x and y is permuted (another 'tokenizer'), (4) only 8 of 12 pretraining
shifts are shared (+ own split), (5) hidden activation is ReLU. S-gain = acc_B(carried alpha) - mean acc_B(30 permuted correspondences)."""
import sys, os, time, itertools, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
SP = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SP + "/ufo_scripts"); sys.path.insert(0, SP + "/rgi_design")
torch.set_num_threads(1)
import pc_meta as M
from pc_common import MLP, accuracy
P, NPRE, NNEW, NSLOT, BASE = M.P, M.NPRE, M.NNEW, M.NSLOT, M.BASE

class ReluMLP(MLP):
    def acts(s, x):
        h1 = torch.relu(s.l1(x)); h2 = torch.relu(s.l2(h1)); return h1, h2

def enc(xy, slot, px, py):
    X = torch.zeros(len(xy), 2 * P + NSLOT); r = torch.arange(len(xy)); X[r, torch.as_tensor(px[xy[:, 0]])] = 1; X[r, P + torch.as_tensor(py[xy[:, 1]])] = 1; X[r, 2 * P + slot] = 1; return X

def train(shifts, splits, px, py, seed, act="tanh", W=128, extra_after=0):
    """shifts: 12 pretraining shifts in slot order; splits: per slot a permutation of the 289 pairs; px, py: input codings."""
    pairs = np.array([(x, y) for x in range(P) for y in range(P)])
    Xtr = torch.cat([enc(pairs[splits[k][:144]], k, px, py) for k in range(NPRE)]); ytr = torch.cat([torch.as_tensor((pairs[splits[k][:144]].sum(1) + shifts[k]) % P) for k in range(NPRE)])
    Xva = torch.cat([enc(pairs[splits[k][144:]], k, px, py) for k in range(NPRE)]); yva = torch.cat([torch.as_tensor((pairs[splits[k][144:]].sum(1) + shifts[k]) % P) for k in range(NPRE)])
    torch.manual_seed(seed); m = (ReluMLP if act == "relu" else MLP)(2 * P + NSLOT, W, P, param="mup", base_width=BASE); opt = M.make_opt(m)
    first = ck = None; after = M.AFTER + extra_after; ckpts = {}
    for s in range(1, M.MAXSTEPS + after + 1):
        if s > M.MAXSTEPS and first is None: break
        M.bp_update(m, opt, Xtr, ytr)
        if s % 50 == 0 and first is None and accuracy(m, Xva, yva) >= M.LEVEL1: first = s
        if first is not None and s == first + M.AFTER and extra_after: ckpts["early"] = {k: v.clone() for k, v in m.state_dict().items()}
        if first is not None and s >= first + after: ck = s; break
    return m, dict(first=first, ck=ck, heldout=accuracy(m, Xva, yva), ckpts=ckpts)

def z1(m, xy, px, py):
    Wt = m.l1.weight.detach(); return (Wt[:, torch.as_tensor(px[xy[:, 0]])] + Wt[:, P + torch.as_tensor(py[xy[:, 1]])]).T + m.l1.bias.detach()
def acc_v(m, xy, y, V, px, py, act="tanh"):
    z = z1(m, xy, px, py); f = torch.relu if act == "relu" else torch.tanh; out = []
    with torch.no_grad():
        for v in V: out.append(float((m.head(f(m.l2(f(z + torch.as_tensor(v, dtype=torch.float32))))).argmax(-1) == y).float().mean()))
    return np.array(out)
def cols(m): return m.l1.weight.detach().numpy()[:, 2 * P: 2 * P + NPRE].astype(float)

def lam(m, shifts, new_shift, xy, px, py, act):
    C = cols(m); y = torch.as_tensor((xy.sum(1) + new_shift) % P)
    V = [C[:, a] + C[:, b] - C[:, c] for a, b, c in itertools.product(range(NPRE), repeat=3) if a < b and c not in (a, b) and (shifts[a] + shifts[b] - shifts[c]) % P == new_shift]
    return acc_v(m, xy, y, V, px, py, act).mean() if V else np.nan

def learn_alpha(m, slots, new_shift, shots, px, py, act, steps=300):
    C = torch.tensor(cols(m)[:, slots], dtype=torch.float32); mu = C.mean(1); Ct = C - mu[:, None]; z = z1(m, shots, px, py); ys = torch.as_tensor((shots.sum(1) + new_shift) % P)
    f = torch.relu if act == "relu" else torch.tanh
    for p in m.parameters(): p.requires_grad_(False)
    a = torch.zeros(len(slots), requires_grad=True); opt = torch.optim.Adam([a], lr=1e-2, betas=(0.9, 0.98))
    for _ in range(steps):
        opt.zero_grad(); L = Fn.cross_entropy(m.head(f(m.l2(f(z + mu + Ct @ a)))), ys); L.backward(); opt.step()
    for p in m.parameters(): p.requires_grad_(True)
    return a.detach().numpy().astype(float)

res = {"natural": [], "family": []}; t0 = time.time()
for ws in (0, 1):
    pre, new, pairs, splits = M.world(ws); ident = np.arange(P)
    # (a) natural variation over inits and a later checkpoint
    for init in range(6):
        m, info = train(pre, splits, ident, ident, ws if init == 0 else 10_000 * init + ws, extra_after=1000)
        if info["ck"] is None: print(f"w{ws} i{init}: fails level 1"); continue
        rows = []
        for tag, sd in (("+500", info["ckpts"]["early"]), ("+1500", m.state_dict())):
            mm = MLP(2 * P + NSLOT, 128, P, param="mup", base_width=BASE); mm.load_state_dict(sd); d = M.delta_stats(mm, pre, pairs, ws)
            L = np.mean([lam(mm, pre, new[j], pairs[splits[NPRE + j][8:]], ident, ident, "tanh") for j in range(NNEW)])
            rows.append((tag, d["h2"]["delta"], L)); res["natural"].append(dict(ws=ws, init=init, ck=tag, delta=d["h2"]["delta"], Lambda=float(L)))
        print(f"w{ws} i{init}: " + " | ".join(f"{t}: delta h2 {d:.3f} Lambda {L:.3f}" for t, d, L in rows) + f" ({time.time() - t0:.0f}s)", flush=True)
    # (b) family fairness
    A, ia = train(pre, splits, ident, ident, ws)
    g = np.random.default_rng(500 + ws); other = [c for c in range(P) if c not in pre and c not in new]
    sh8 = pre[:8] + list(g.choice([c for c in other], size=min(4, len(other)), replace=False)) if len(other) >= 4 else None
    print(f"w{ws}: pretraining shifts {pre}, new {new}, unused {other}")
    split2 = [g.permutation(P * P) for _ in range(NPRE + NNEW)]; px, py = g.permutation(P), g.permutation(P)
    variants = {"init": (pre, splits, ident, ident, "tanh", list(range(NPRE))), "split": (pre, split2, ident, ident, "tanh", list(range(NPRE))),
                "coding": (pre, split2, px, py, "tanh", list(range(NPRE))), "relu": (pre, split2, ident, ident, "relu", list(range(NPRE)))}
    # 8 shared shifts: B's slots 0..7 hold A's first 8 shifts, slots 8..11 hold shifts A never saw (the remaining unused shift values incl. the spare are few)
    print(f"  shifts unused by A and not new: {other}")
    for name, (shB, spB, pxB, pyB, act, shared) in variants.items():
        B, ib = train(shB, spB, pxB, pyB, 10_000 * 7 + ws, act=act)
        if ib["ck"] is None: print(f"  B {name}: fails level 1 (held-out {ib['heldout']:.3f})"); continue
        gains, accA, lamB, accBown = [], [], [], []
        for j in range(NNEW):
            shots = pairs[splits[NPRE + j][:8]]; xyh = pairs[splits[NPRE + j][8:]]; yh = torch.as_tensor((xyh.sum(1) + new[j]) % P)
            a = learn_alpha(A, shared, new[j], shots, ident, ident, "tanh")
            CA = cols(A)[:, shared]; CB = cols(B)[:, shared]; vA = CA.mean(1) + (CA - CA.mean(1)[:, None]) @ a; vB = CB.mean(1) + (CB - CB.mean(1)[:, None]) @ a
            perms = [g.permutation(len(shared)) for _ in range(30)]; Vb = [vB] + [CB.mean(1) + (CB - CB.mean(1)[:, None])[:, p] @ a for p in perms]
            accB = acc_v(B, xyh, yh, Vb, pxB, pyB, act); accA.append(acc_v(A, xyh, yh, [vA], ident, ident, "tanh")[0]); gains.append(accB[0] - accB[1:].mean())
            lamB.append(lam(B, shB, new[j], xyh, pxB, pyB, act))
        res["family"].append(dict(ws=ws, variant=name, heldoutB=ib["heldout"], alpha_in_A=float(np.mean(accA)), S_gain=float(np.mean(gains)), Lambda_B=float(np.nanmean(lamB))))
        print(f"  B {name:6s}: held-out {ib['heldout']:.3f} | alpha in A {np.mean(accA):.3f} | S-gain in B {np.mean(gains):+.3f} | Lambda_B {np.nanmean(lamB):.3f} ({time.time() - t0:.0f}s)", flush=True)
json.dump(res, open("family.json", "w"), indent=1)
