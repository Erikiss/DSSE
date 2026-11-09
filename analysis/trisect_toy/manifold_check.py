"""Is the part of the intermediate-swap edit orthogonal to the answer direction a step off the activation manifold
(normal to the local tangent space spanned by d h_L / d(a,b,c)) or a step along it that the output does not read?"""
import math
import numpy as np
import torch
import trisect_toy as T

def train(task, seed):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    tr = T.features(*T.triangles(T.N_TRAIN, rng))
    X = lambda F: np.stack([F["a"], F["b"], F["c"]], 1)
    xm, xs = X(tr).mean(0), X(tr).std(0); ym, ys = tr[task].mean(), tr[task].std()
    Xtr = torch.tensor((X(tr) - xm) / xs, dtype=torch.float32); Ytr = torch.tensor((tr[task] - ym) / ys, dtype=torch.float32)
    m = T.MLP(); opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    steps = T.EPOCHS * math.ceil(T.N_TRAIN / 512); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
    for ep in range(T.EPOCHS):
        perm = torch.randperm(T.N_TRAIN)
        for i in range(0, T.N_TRAIN, 512):
            idx = perm[i:i + 512]; loss = ((m(Xtr[idx]) - Ytr[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m.eval(), X, xm, xs, rng

def hidden(m, x, L):
    for lin in m.hidden[:L + 1]:
        x = torch.relu(lin(x))
    return x

for task in ("s", "l"):
    m, X, xm, xs, rng = train(task, 0)
    n = 1500
    a2, b2 = rng.uniform(0.5, 2.0, n), rng.uniform(0.5, 2.0, n)
    P1, P2 = T.features(*T.triangles(n, rng, (a2, b2))), T.features(*T.triangles(n, rng, (a2, b2)))
    X1 = torch.tensor((X(P1) - xm) / xs, dtype=torch.float32); X2 = torch.tensor((X(P2) - xm) / xs, dtype=torch.float32)
    dC = np.abs(P2["C"] - P1["C"])
    with torch.no_grad():
        y1, H1 = m(X1, return_h=True); y2, H2 = m(X2, return_h=True)
    dys = (y2 - y1).numpy()
    print(f"\n{'Trisektor' if task == 's' else 'Halbierende'} (Seed 0)")
    for L in range(4):
        h = T.MLP.forward  # noqa
        hh = H1[L][:4000].clone().requires_grad_(True)
        xx = hh
        for lin in m.hidden[L + 1:]:
            xx = torch.relu(lin(xx))
        m.out(xx).sum().backward(); J = hh.grad.mean(0).numpy(); jh = J / np.linalg.norm(J)
        D = (H2[L] - H1[L]).numpy(); Do = D - (D @ jh)[:, None] * jh[None, :]
        tang, eff_t, eff_n = [], [], []
        Dt_all, Dn_all = np.zeros_like(Do), np.zeros_like(Do)
        from torch.func import vmap, jacrev
        JX1 = vmap(jacrev(lambda x: hidden(m, x, L)))(X1).detach().numpy()   # n x 64 x 3: tangent space at h1
        JX2 = vmap(jacrev(lambda x: hidden(m, x, L)))(X2).detach().numpy()   # tangent space at h2
        for i in range(n):
            B = np.linalg.qr(np.c_[JX1[i], JX2[i]])[0]                                             # span of both (<= 6 dims)
            dt = B @ (B.T @ Do[i]); Dt_all[i], Dn_all[i] = dt, Do[i] - dt
            tang.append((dt @ dt) / max(1e-12, Do[i] @ Do[i]))
        with torch.no_grad():
            coef = lambda d: float(((m(X1, edit_layer=L, edit=torch.tensor(d, dtype=torch.float32)) - y1).numpy() * dys).sum() / (dys ** 2).sum())
            e_o, e_t, e_n = coef(Do), coef(Dt_all), coef(Dn_all)
        tang = np.array(tang)
        small, large = dC < np.quantile(dC, 0.25), dC > np.quantile(dC, 0.75)
        print(f"  L{L + 1}: Rest-Anteil an |D|² {np.mean((Do ** 2).sum(1) / (D ** 2).sum(1)):.2f} | davon tangential {tang.mean():.2f}"
              f" (kleines ΔC {tang[small].mean():.2f}, großes ΔC {tang[large].mean():.2f}) | Wirkung: Rest {e_o:+.3f}, tangentialer Teil {e_t:+.3f}, normaler Teil {e_n:+.3f}")
