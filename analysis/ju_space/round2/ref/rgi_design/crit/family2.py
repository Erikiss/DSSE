"""Critique check 2b: harder 'other family' variants for the carried alpha: (1) B pretrained on only 9 tasks (8 of A's shifts + the
spare shift), alpha learned in A over the 8 shared anchors; (2) B with ReLU at lr 3e-4 and 1e-4; (3) B trained on a different task
family (x + 2y + c) mod 17 sharing only the shift concept; (4) B with 3 hidden tanh layers. Across-init correlation of Lambda and delta."""
import sys, os, time, itertools, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
SP = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SP + "/ufo_scripts"); sys.path.insert(0, SP + "/rgi_design"); sys.path.insert(0, SP + "/rgi_design/crit")
torch.set_num_threads(1)
import pc_meta as M
from pc_common import MLP, accuracy
P, NPRE, NNEW, NSLOT, BASE = M.P, M.NPRE, M.NNEW, M.NSLOT, M.BASE
nat = json.load(open("family.json"))["natural"]
for ck in ("+500", "+1500"):
    for ws in (0, 1):
        r = [(x["delta"], x["Lambda"]) for x in nat if x["ck"] == ck and x["ws"] == ws]; a = np.array(r); print(f"natural {ck} w{ws}: corr(delta, Lambda) over {len(a)} inits {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:+.2f}; delta {a[:, 0].min():.3f}-{a[:, 0].max():.3f}, Lambda {a[:, 1].min():.3f}-{a[:, 1].max():.3f}")

class Net(nn.Module):
    def __init__(s, act="tanh", depth=2, W=128):
        super().__init__(); din = 2 * P + NSLOT; s.l1 = nn.Linear(din, W); s.hs = nn.ModuleList([nn.Linear(W, W) for _ in range(depth - 1)]); s.l3 = nn.Linear(W, P); s.f = torch.relu if act == "relu" else torch.tanh
        nn.init.normal_(s.l1.weight, std=1 / np.sqrt(din)); nn.init.zeros_(s.l1.bias)
        for l in s.hs: nn.init.normal_(l.weight, std=1 / np.sqrt(W)); nn.init.zeros_(l.bias)
        nn.init.zeros_(s.l3.weight); nn.init.zeros_(s.l3.bias); s.W = W
    def from_z1(s, z):
        h = s.f(z)
        for l in s.hs: h = s.f(l(h))
        return s.l3(h)
    def forward(s, x): return s.from_z1(s.l1(x))

def enc(xy, slot):
    X = torch.zeros(len(xy), 2 * P + NSLOT); r = torch.arange(len(xy)); X[r, torch.as_tensor(xy[:, 0])] = 1; X[r, P + torch.as_tensor(xy[:, 1])] = 1; X[r, 2 * P + slot] = 1; return X
pairs = np.array([(x, y) for x in range(P) for y in range(P)])
def lab(xy, c, ky): return torch.as_tensor((xy[:, 0] + ky * xy[:, 1] + c) % P)

def train(shifts, splits, seed, ky=1, act="tanh", depth=2, lr=1e-3):
    n = len(shifts); Xtr = torch.cat([enc(pairs[splits[k][:144]], k) for k in range(n)]); ytr = torch.cat([lab(pairs[splits[k][:144]], shifts[k], ky) for k in range(n)])
    Xva = torch.cat([enc(pairs[splits[k][144:]], k) for k in range(n)]); yva = torch.cat([lab(pairs[splits[k][144:]], shifts[k], ky) for k in range(n)])
    torch.manual_seed(seed); m = Net(act, depth)
    hid = [l.weight for l in m.hs]; rest = [p for nme, p in m.named_parameters() if not (nme.startswith("hs.") and nme.endswith("weight"))]
    opt = torch.optim.AdamW([dict(params=hid, lr=lr, weight_decay=1e-3 / lr), dict(params=rest, lr=lr, weight_decay=1e-3 / lr)], lr=lr, betas=(0.9, 0.98))
    first = None
    for s in range(1, 6000 + 500 + 1):
        if s > 6000 and first is None: break
        opt.zero_grad(); L = Fn.cross_entropy(m(Xtr), ytr); L.backward(); opt.step()
        if s % 50 == 0 and first is None:
            with torch.no_grad(): a = float((m(Xva).argmax(-1) == yva).float().mean())
            if a >= 0.95: first = s
        if first is not None and s >= first + 500: break
    with torch.no_grad(): ho = float((m(Xva).argmax(-1) == yva).float().mean())
    return m, first, ho

def z1(m, xy): Wt = m.l1.weight.detach(); return (Wt[:, torch.as_tensor(xy[:, 0])] + Wt[:, P + torch.as_tensor(xy[:, 1])]).T + m.l1.bias.detach()
def accv(m, xy, y, V):
    z = z1(m, xy)
    with torch.no_grad(): return np.array([float((m.from_z1(z + torch.as_tensor(v, dtype=torch.float32)).argmax(-1) == y).float().mean()) for v in V])
def cols(m, slots): return m.l1.weight.detach().numpy()[:, [2 * P + k for k in slots]].astype(float)
def learn_alpha(m, slots, c, shots, ky, steps=300):
    C = torch.tensor(cols(m, slots), dtype=torch.float32); mu = C.mean(1); Ct = C - mu[:, None]; z = z1(m, shots); ys = lab(shots, c, ky)
    for p in m.parameters(): p.requires_grad_(False)
    a = torch.zeros(len(slots), requires_grad=True); opt = torch.optim.Adam([a], lr=1e-2, betas=(0.9, 0.98))
    for _ in range(steps): opt.zero_grad(); L = Fn.cross_entropy(m.from_z1(z + mu + Ct @ a), ys); L.backward(); opt.step()
    for p in m.parameters(): p.requires_grad_(True)
    return a.detach().numpy().astype(float)
def lam(m, shifts, c, xy, ky):
    C = cols(m, range(len(shifts))); V = [C[:, a] + C[:, b] - C[:, d] for a, b, d in itertools.product(range(len(shifts)), repeat=3) if a < b and d not in (a, b) and (shifts[a] + shifts[b] - shifts[d]) % P == c]
    return accv(m, xy, lab(xy, c, ky), V).mean() if V else np.nan

t0 = time.time()
for ws in (0, 1):
    pre, new, _, splits = M.world(ws); other = [c for c in range(P) if c not in pre and c not in new]; g = np.random.default_rng(900 + ws)
    A, fA, hA = train(pre, splits, ws); print(f"w{ws} A: first95 {fA} held-out {hA:.3f}")
    split2 = [g.permutation(P * P) for _ in range(NPRE + NNEW + 1)]
    variants = [("9 tasks (8 shared + spare)", dict(shifts=pre[:8] + other[:1], ky=1), list(range(8)), list(range(8))),
                ("relu lr 3e-4", dict(shifts=pre, ky=1, act="relu", lr=3e-4), list(range(12)), list(range(12))),
                ("relu lr 1e-4", dict(shifts=pre, ky=1, act="relu", lr=1e-4), list(range(12)), list(range(12))),
                ("task family x+2y+c", dict(shifts=pre, ky=2), list(range(12)), list(range(12))),
                ("3 hidden layers", dict(shifts=pre, ky=1, depth=3), list(range(12)), list(range(12)))]
    for name, kw, slA, slB in variants:
        ky = kw.get("ky", 1); B, fB, hB = train(kw["shifts"], split2, 70_000 + ws, ky=ky, act=kw.get("act", "tanh"), depth=kw.get("depth", 2), lr=kw.get("lr", 1e-3))
        if fB is None: print(f"  B {name}: fails level 1 (held-out {hB:.3f}) ({time.time() - t0:.0f}s)", flush=True); continue
        gains, aA, lB, rB = [], [], [], []
        for j in range(NNEW):
            shots = pairs[splits[NPRE + j][:8]]; xyh = pairs[splits[NPRE + j][8:]]
            a = learn_alpha(A, slA, new[j], shots, 1); CA, CB = cols(A, slA), cols(B, slB)
            vA = CA.mean(1) + (CA - CA.mean(1)[:, None]) @ a; vB = CB.mean(1) + (CB - CB.mean(1)[:, None]) @ a
            perms = [g.permutation(len(slB)) for _ in range(30)]; acc = accv(B, xyh, lab(xyh, new[j], ky), [vB] + [CB.mean(1) + (CB - CB.mean(1)[:, None])[:, p] @ a for p in perms])
            gains.append(acc[0] - acc[1:].mean()); aA.append(accv(A, xyh, lab(xyh, new[j], 1), [vA])[0]); lB.append(lam(B, kw["shifts"], new[j], xyh, ky))
            # B's own anchored improvement from its own shots (expressivity of B's anchor span)
            rB.append(accv(B, xyh, lab(xyh, new[j], ky), [CB.mean(1) + (CB - CB.mean(1)[:, None]) @ learn_alpha(B, slB, new[j], shots, ky)])[0])
        print(f"  B {name:26s}: first95 {fB} held-out {hB:.3f} | alpha in A {np.mean(aA):.3f} | carried S-gain in B {np.mean(gains):+.3f} | B's own alpha {np.mean(rB):.3f} | Lambda_B {np.nanmean(lB):.3f} ({time.time() - t0:.0f}s)", flush=True)
