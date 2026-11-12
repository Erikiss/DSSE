"""Recursion pilot on anchored improvements: (a) pooling: alpha averaged over models (same 8 shots) evaluated in each model vs its own
alpha; (b) round trip with refinement: alpha_A -> B refines 40 steps on its shots -> back to A, vs A refining its own alpha 40 more steps.
Usage: python rgi_pool.py <ws> <model names...>"""
import sys, numpy as np, torch
SP = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SP + "/rgi_design")
from rgi_lib import *

def learn_alpha_from(g, ws, j, a0, steps, nshot=8, lr=1e-2):
    pre, new, pairs, splits = M.world(ws); slot = NPRE + j; idx = splits[slot]; xs = pairs[idx[:nshot]]; ys = torch.as_tensor((xs[:, 0] + xs[:, 1] + new[j]) % P)
    C = torch.tensor(g["C"], dtype=torch.float32); mA = C.mean(1); Ct = C - mA[:, None]; z = base_z1(g["m"], xs); m = g["m"]
    for p_ in m.parameters(): p_.requires_grad_(False)
    a = torch.tensor(a0, dtype=torch.float32).clone().requires_grad_(True); opt = torch.optim.Adam([a], lr=lr, betas=(0.9, 0.98))
    for _ in range(steps):
        opt.zero_grad(); L = Fn.cross_entropy(m.head(torch.tanh(m.l2(torch.tanh(z + mA + Ct @ a)))), ys); L.backward(); opt.step()
    for p_ in m.parameters(): p_.requires_grad_(True)
    return a.detach().numpy().astype(float)

def acc_alpha(g, ws, j, a):
    pre, new, pairs, splits = M.world(ws); xy = pairs[splits[NPRE + j][8:]]; yy = torch.as_tensor((xy[:, 0] + xy[:, 1] + new[j]) % P); mA = g["C"].mean(1)
    return acc_cols(g["m"], xy, yy, torch.tensor(np.stack([mA + (g["C"] - mA[:, None]) @ a]), dtype=torch.float32))[0]

ws = int(sys.argv[1]); names = sys.argv[2:]; G = {}
for n in names:
    d = torch.load(f"{SP}/rgi_design/pilot/models/{n}.pt"); info = d["info"]; m = mk(info["arm"], info["W"]); m.load_state_dict(d["state"])
    G[n] = dict(m=m, C=m.l1.weight.detach().numpy()[:, 2 * P: 2 * P + NPRE].astype(float), W=info["W"])
res = {n: [] for n in names}; rt = []
for j in range(NNEW):
    al = {n: learn_alpha_from(G[n], ws, j, np.zeros(NPRE), 300) for n in names}; pool = np.mean([al[n] for n in names], 0)
    for n in names:
        others = np.mean([al[o] for o in names if o != n], 0)
        res[n].append((acc_alpha(G[n], ws, j, al[n]), acc_alpha(G[n], ws, j, pool), acc_alpha(G[n], ws, j, others)))
    for a_ in names:
        for b_ in names:
            if a_ == b_: continue
            aB = learn_alpha_from(G[b_], ws, j, al[a_], 40); own = learn_alpha_from(G[a_], ws, j, al[a_], 40)
            rt.append((acc_alpha(G[a_], ws, j, aB), acc_alpha(G[a_], ws, j, own), acc_alpha(G[a_], ws, j, al[a_])))
print("pooling: model | own alpha | pooled alpha (all models) | alpha pooled from the others only")
for n in names:
    r = np.array(res[n]); print(f"  {n:28s} {r[:, 0].mean():.3f} | {r[:, 1].mean():.3f} | {r[:, 2].mean():.3f}")
rt = np.array(rt); print(f"round trip with 40 refinement steps in B: back in A {rt[:, 0].mean():.3f} vs A's own +40 steps {rt[:, 1].mean():.3f} (A's alpha at 300 steps {rt[:, 2].mean():.3f}); gain {np.mean(rt[:, 0] - rt[:, 1]):+.3f}, positive in {np.mean(rt[:, 0] > rt[:, 1]):.2f} of {len(rt)}")
