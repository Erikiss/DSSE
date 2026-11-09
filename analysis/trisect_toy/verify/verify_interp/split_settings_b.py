"""Additions to split_settings.py:
S3w  weak edits (scaled intermediate swap kappa*(E[z']-E[z])) so that only marginal flips occur -> flip accounting vs kappa
S3L  learned lookup where a LINEAR downstream map cannot solve the task (K=512 entities >> d=32): trained residual ReLU MLP,
     plus a linear-only baseline; split of the intermediate swap.
"""
import json, math, sys
from pathlib import Path
import numpy as np
import torch
import split_settings as S

torch.set_num_threads(1)
HERE = Path(__file__).parent
out = {"weak": [], "learned_big": []}

rng, E, Ua, ans, ug = S.world(3)
N = 3000
z, zp, c, g = S.items(rng, ans, N)
zb, _, cb, gb = S.items(rng, ans, 4000)
a, ap = ans[z], ans[zp]
lam = 3.0
for name, F, H1, Hbg in (
        ("two_step rho0.5", S.lookup_net(E, Ua, ans, 0.5, 0.5, lam), E[z] + c, E[zb] + cb),
        ("two_step rho1.0", S.lookup_net(E, Ua, ans, 1.0, 0.5, lam), E[z] + c, E[zb] + cb),
        ("mixed", S.lookup_net(E, Ua, ans, 1.0, 0.5, lam / 2), E[z] + c + lam / 2 * Ua[a], E[zb] + cb + lam / 2 * Ua[ans[zb]])):
    for kappa in (0.4, 0.5, 0.6, 0.75, 1.0):
        if name == "mixed":
            D = kappa * ((E[zp] + lam / 2 * Ua[ap]) - (E[z] + lam / 2 * Ua[a]))
        else:
            D = kappa * (E[zp] - E[z])
        s = S.vector_split(F, Hbg, H1, D, a, ap); s.update(setting=name, kappa=kappa); out["weak"].append(s)
        S.pr("S3w " + name, s, f"kappa {kappa}")

# learned lookup with K >> d
d2, K2, M2 = 32, 512, 8
r2 = np.random.default_rng(11)
E2 = S.T_(S.unit_rows(K2, d2, r2)); U2 = S.T_(S.unit_rows(M2, d2, r2))
ans2 = np.repeat(np.arange(M2), K2 // M2); r2.shuffle(ans2)
def items2(n, sig_c=0.3):
    zz = r2.integers(0, K2, n)
    zzp = np.array([r2.choice(np.where(ans2 != ans2[q])[0]) for q in zz])
    return zz, zzp, S.T_(r2.normal(size=(n, d2)) * sig_c / math.sqrt(d2))
for kind in ("linear", "mlp512"):
    torch.manual_seed(0)
    if kind == "linear":
        W = torch.nn.Linear(d2, d2); params = list(W.parameters())
        F2 = lambda h: (h + W(h)) @ U2.T
    else:
        W1 = torch.nn.Linear(d2, 512); W2 = torch.nn.Linear(512, d2); params = list(W1.parameters()) + list(W2.parameters())
        F2 = lambda h: (h + W2(torch.relu(W1(h)))) @ U2.T
    opt = torch.optim.Adam(params, lr=2e-3)
    for step in range(4000):
        zz, _, cc = items2(512)
        loss = torch.nn.functional.cross_entropy(F2(E2[zz] + cc), torch.tensor(ans2[zz]))
        opt.zero_grad(); loss.backward(); opt.step()
    zz, zzp, cc = items2(3000); zb2, _, cb2 = items2(4000)
    with torch.no_grad():
        acc = float((F2(E2[zz] + cc).argmax(1).numpy() == ans2[zz]).mean())
    print(f"S3L {kind}: train-dist accuracy {acc:.3f}", flush=True)
    if kind == "linear":
        out["learned_big"].append(dict(kind=kind, acc=acc)); continue
    # restrict to items the model gets right (vector_split uses global M; patch it)
    S.M = M2
    s = S.vector_split(F2, E2[zb2] + cb2, E2[zz] + cc, E2[zzp] - E2[zz], ans2[zz], ans2[zzp]); s.update(kind=kind, acc=acc)
    out["learned_big"].append(s); S.pr("S3L learned K>>d", s, "intermediate swap")
    for kappa in (0.5, 0.75):
        s = S.vector_split(F2, E2[zb2] + cb2, E2[zz] + cc, kappa * (E2[zzp] - E2[zz]), ans2[zz], ans2[zzp]); s.update(kind=kind, kappa=kappa)
        out["learned_big"].append(s); S.pr("S3L learned K>>d", s, f"intermediate swap kappa {kappa}")
json.dump(out, open(HERE / "split_settings_b_results.json", "w"), indent=1)
