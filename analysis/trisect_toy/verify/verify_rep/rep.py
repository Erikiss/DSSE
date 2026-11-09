"""Adversarial replication of trisect_toy claims (1) probes and (2) Tang-style answer-direction split.

Independent code. Variations: architecture (ReLU 64-64-64-15, tanh 128x3, GELU 128x3), seeds, sample range,
size of the counterfactual change (|dC| small/medium/independent/large), average vs per-pair (local) Jacobian,
plus controls (random-init network for probes; shuffled and random edits for the split).
Usage: python rep.py [config ...]   (default: all configs)
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(int(__import__("os").environ.get("REP_THREADS", 4)))
OUT = Path(__file__).parent
ACTS = dict(relu=torch.relu, tanh=torch.tanh, gelu=nn.functional.gelu)
R_ORIG = dict(ab=(0.5, 2.0), C=(0.15, math.pi - 0.15))
R_WIDE = dict(ab=(0.3, 3.0), C=(0.05, math.pi - 0.05))
CONFIGS = {
    "relu64": dict(widths=[64, 64, 64, 15], act="relu", R=R_ORIG, seeds=[10, 11], epochs=20),
    "tanh128": dict(widths=[128, 128, 128], act="tanh", R=R_ORIG, seeds=[10], epochs=15),
    "gelu128_wide": dict(widths=[128, 128, 128], act="gelu", R=R_WIDE, seeds=[12], epochs=15),
}
N_TRAIN, N_TEST, N_PAIRS = int(__import__("os").environ.get("REP_NTRAIN", 200_000)), 20_000, 3000
REGIMES = {"small": (0.01, 0.04), "medium": (0.2, 0.4), "indep": None, "large": (1.2, 9.0)}
FEATS = ("c", "cosC", "C", "cosC3", "cosC2", "s", "l")


class MLP(nn.Module):
    def __init__(self, widths, act):
        super().__init__()
        dims = [3] + widths
        self.hidden = nn.ModuleList(nn.Linear(i, o) for i, o in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(dims[-1], 1)
        self.act = ACTS[act]

    def hs(self, x):
        H = []
        for lin in self.hidden:
            x = self.act(lin(x))
            H.append(x)
        return self.out(x).squeeze(-1), H

    def forward(self, x):
        return self.hs(x)[0]

    def tail(self, h, L):
        """output as a function of the post-activation of hidden layer L (0-based)"""
        for lin in self.hidden[L + 1:]:
            h = self.act(lin(h))
        return self.out(h).squeeze(-1)


def feats(a, b, C):
    c = np.sqrt(a * a + b * b - 2 * a * b * np.cos(C))
    th = C / 3
    s = a * b * np.sin(3 * th) / (a * np.sin(2 * th) + b * np.sin(th))
    l = 2 * a * b * np.cos(C / 2) / (a + b)
    return dict(a=a, b=b, c=c, C=C, cosC=np.cos(C), cosC3=np.cos(C / 3), cosC2=np.cos(C / 2), s=s, l=l)


def sample(n, rng, R):
    return feats(rng.uniform(*R["ab"], n), rng.uniform(*R["ab"], n), rng.uniform(*R["C"], n))


def make_pairs(n, rng, R, regime):
    lo, hi = R["C"]
    a1, b1, C1, C2 = [], [], [], []
    while sum(map(len, C1)) < n:
        m = 4 * n
        a = rng.uniform(*R["ab"], m); b = rng.uniform(*R["ab"], m); c1 = rng.uniform(lo, hi, m)
        if REGIMES[regime] is None:
            c2 = rng.uniform(lo, hi, m); ok = np.ones(m, bool)
        else:
            dmin, dmax = REGIMES[regime]
            if regime == "large":
                c2 = rng.uniform(lo, hi, m); ok = np.abs(c2 - c1) >= dmin
            else:
                d = rng.uniform(dmin, dmax, m) * rng.choice([-1.0, 1.0], m)
                c2 = c1 + d
                flip = (c2 < lo) | (c2 > hi); c2[flip] = c1[flip] - d[flip]
                ok = (c2 >= lo) & (c2 <= hi)
        a1.append(a[ok]); b1.append(b[ok]); C1.append(c1[ok]); C2.append(c2[ok])
    a, b, c1, c2 = (np.concatenate(v)[:n] for v in (a1, b1, C1, C2))
    return feats(a, b, c1), feats(a, b, c2)


def ridge_fit(X, y, alpha=1e-3):
    mu, sd = X.mean(0), X.std(0)
    keep = sd > 1e-7
    Z = (X[:, keep] - mu[keep]) / sd[keep]
    A = Z.T @ Z / len(Z) + alpha * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - y.mean()) / len(Z))
    wr = np.zeros(X.shape[1]); wr[keep] = w / sd[keep]
    return wr, y.mean() - mu @ wr


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def probe(Htr, ytr, Hte, yte):
    w, b = ridge_fit(Htr, ytr)
    return r2(yte, Hte @ w + b)


def grads(m, L, H):
    h = H.clone().requires_grad_(True)
    m.tail(h, L).sum().backward()          # samples are independent -> per-sample gradients
    return h.grad.detach()


def analyse_split(m, L, H1, y1, D, dys, U, tag):
    """split edit D (n x d) along unit directions U (n x d or 1 x d); return metrics w.r.t. target change dys"""
    proj = (D * U).sum(1, keepdim=True)
    Da = proj * U
    Do = D - Da
    with torch.no_grad():
        dyf = m.tail(H1 + D, L) - y1
        dya = m.tail(H1 + Da, L) - y1
        dyo = m.tail(H1 + Do, L) - y1
    ss = (dys ** 2).sum()
    coef = lambda dy: float((dy * dys).sum() / ss)
    return {
        f"{tag}_full": coef(dyf), f"{tag}_ans": coef(dya), f"{tag}_orth": coef(dyo),
        f"{tag}_fid_ans": float(1 - ((dya - dys) ** 2).sum() / ss),          # per-pair fidelity, 1 = exact
        f"{tag}_rms_orth": float(torch.sqrt((dyo ** 2).sum() / ss)),          # magnitude of orth effect, any sign
        f"{tag}_normshare": float((proj ** 2).mean() / (D ** 2).sum(1).mean()),
    }


def run(cfg_name, task, seed):
    cfg = CONFIGS[cfg_name]
    R = cfg["R"]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    tr, te = sample(N_TRAIN, rng, R), sample(N_TEST, rng, R)
    X = lambda F: np.stack([F["a"], F["b"], F["c"]], 1)
    xm, xs = X(tr).mean(0), X(tr).std(0)
    ym, ys = tr[task].mean(), tr[task].std()
    Xtr = torch.tensor((X(tr) - xm) / xs, dtype=torch.float32)
    Ytr = torch.tensor((tr[task] - ym) / ys, dtype=torch.float32)
    m = MLP(cfg["widths"], cfg["act"])
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    bs, ep = 512, int(__import__("os").environ.get("REP_EPOCHS", cfg["epochs"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep * math.ceil(N_TRAIN / bs), eta_min=1e-5)
    t0 = time.time()
    for _ in range(ep):
        perm = torch.randperm(N_TRAIN)
        for i in range(0, N_TRAIN, bs):
            idx = perm[i:i + bs]
            loss = ((m(Xtr[idx]) - Ytr[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    train_min = (time.time() - t0) / 60
    m = m.double().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    torch.manual_seed(seed + 1000); rand_net = MLP(cfg["widths"], cfg["act"]).double().eval()
    T = lambda F: torch.tensor((X(F) - xm) / xs, dtype=torch.float64)
    with torch.no_grad():
        yte, Hte = m.hs(T(te))
        _, Htr = m.hs(T({k: v[:30_000] for k, v in tr.items()}))
        _, Rte = rand_net.hs(T(te)); _, Rtr = rand_net.hs(T({k: v[:30_000] for k, v in tr.items()}))
    yte_np = yte.numpy() * ys + ym
    res = dict(cfg=cfg_name, task=task, seed=seed, test_r2=r2(te[task], yte_np), train_min=train_min, layers=[])
    trf = {k: v[:30_000] for k, v in tr.items()}
    # input baselines for probes
    def poly2(F):
        Z = (X(F) - xm) / xs
        return np.c_[Z, np.stack([Z[:, i] * Z[:, j] for i in range(3) for j in range(i, 3)], 1)]
    res["input_probes"] = {f: dict(lin=probe(X(trf), trf[f], X(te), te[f]), poly2=probe(poly2(trf), trf[f], poly2(te), te[f]))
                           for f in FEATS}
    # incremental targets: candidate intermediate beyond the other candidates (linear), fitted on test features
    def resid(F, f, others):
        Bm = np.c_[np.ones(len(F[f])), np.stack([F[o] for o in others], 1)]
        return F[f] - Bm @ np.linalg.lstsq(Bm, F[f], rcond=None)[0]
    incr = {"cosC3|C,cosC,cosC2": ("cosC3", ("C", "cosC", "cosC2")),
            "cosC2|C,cosC,cosC3": ("cosC2", ("C", "cosC", "cosC3")),
            "s|l,a,b,c": ("s", ("l", "a", "b", "c")), "l|s,a,b,c": ("l", ("s", "a", "b", "c"))}
    # pair sets
    pairs = {}
    for reg in REGIMES:
        P1, P2 = make_pairs(N_PAIRS, rng, R, reg)
        with torch.no_grad():
            y1, H1 = m.hs(T(P1)); y2, H2 = m.hs(T(P2))
        pairs[reg] = (P1, P2, y1, H1, y2, H2)
    for L in range(len(cfg["widths"])):
        Ly = dict(layer=L + 1)
        Htr_L, Hte_L = Htr[L].numpy(), Hte[L].numpy()
        Ly["probes"] = {f: probe(Htr_L, trf[f], Hte_L, te[f]) for f in FEATS}
        Ly["probes_randnet"] = {f: probe(Rtr[L].numpy(), trf[f], Rte[L].numpy(), te[f]) for f in FEATS}
        Ly["incr"] = {}
        for k, (f, others) in incr.items():
            rr = resid(te, f, others)
            Ly["incr"][k] = probe(Hte_L[:10_000], rr[:10_000], Hte_L[10_000:], rr[10_000:])
        Ly["dead_share"] = float((Hte_L.max(0) <= 0).mean()) if cfg["act"] == "relu" else 0.0
        Gte = grads(m, L, Hte[L][:5000])
        J = Gte.mean(0, keepdim=True)
        jhat = J / J.norm()
        Ly["grad_cos_to_mean"] = float(((Gte / Gte.norm(dim=1, keepdim=True)) @ jhat.T).mean())
        Ly["regimes"] = {}
        for reg, (P1, P2, y1, H1, y2, H2) in pairs.items():
            D = H2[L] - H1[L]; dys = y2 - y1
            e = {"mean_absdC": float(np.abs(P2["C"] - P1["C"]).mean())}
            with torch.no_grad():
                e["full_exact_maxerr"] = float((m.tail(H1[L] + D, L) - y2).abs().max())
            G1 = grads(m, L, H1[L]); Gm = grads(m, L, H1[L] + 0.5 * D)
            u1 = G1 / G1.norm(dim=1, keepdim=True).clamp_min(1e-30)
            um = Gm / Gm.norm(dim=1, keepdim=True).clamp_min(1e-30)
            e.update(analyse_split(m, L, H1[L], y1, D, dys, jhat, "avg"))      # original: mean Jacobian of test set
            e.update(analyse_split(m, L, H1[L], y1, D, dys, u1, "loc"))        # per-pair Jacobian at the clean point
            e.update(analyse_split(m, L, H1[L], y1, D, dys, um, "mid"))        # per-pair Jacobian at the midpoint
            # first-order linear prediction with local Jacobian: how linear is the downstream map over this change?
            lin_pred = (G1 * D).sum(1)
            e["linpred_coef"] = float((lin_pred * dys).sum() / (dys ** 2).sum())
            e["linpred_fid"] = float(1 - ((lin_pred - dys) ** 2).sum() / (dys ** 2).sum())
            # encoding direction of C (regression of D on dC through origin), split along mean Jacobian
            dC = torch.tensor(P2["C"] - P1["C"])
            uC = (D * dC[:, None]).sum(0) / (dC ** 2).sum()
            E = dC[:, None] * uC[None, :]
            enc = analyse_split(m, L, H1[L], y1, E, dys, jhat, "enc")
            e.update({k: enc[k] for k in ("enc_full", "enc_ans", "enc_orth", "enc_rms_orth")})
            e["enc_cos_answer"] = float((uC @ jhat[0]).abs() / uC.norm())
            # controls: an edit that has nothing to do with C. shuffled differences (other pair's D on this base)
            # and isotropic random edits with matched norm; target = the edit's OWN effect.
            g = torch.Generator().manual_seed(seed * 7 + L)
            perm = torch.randperm(len(D), generator=g)
            for ctag, Dc in (("shuf", D[perm]),
                             ("rand", torch.randn(D.shape, generator=g, dtype=torch.float64) * D.norm(dim=1, keepdim=True) / math.sqrt(D.shape[1]))):
                with torch.no_grad():
                    own = m.tail(H1[L] + Dc, L) - y1
                c = analyse_split(m, L, H1[L], y1, Dc, own, jhat, ctag)
                e.update({k: c[k] for k in (f"{ctag}_ans", f"{ctag}_orth", f"{ctag}_fid_ans", f"{ctag}_rms_orth", f"{ctag}_normshare")})
                e[f"{ctag}_own_rms_rel"] = float(own.norm() / dys.norm())
            Ly["regimes"][reg] = e
        res["layers"].append(Ly)
    res["total_min"] = (time.time() - t0) / 60
    return res


if __name__ == "__main__":
    names = sys.argv[1:] or list(CONFIGS)
    allres = []
    for name in names:
        for task in ("s", "l"):
            for seed in CONFIGS[name]["seeds"]:
                r = run(name, task, seed)
                allres.append(r)
                print(f"{name} task={task} seed={seed} R2={r['test_r2']:.6f} train {r['train_min']:.2f} min total {r['total_min']:.2f} min", flush=True)
                json.dump(allres, open(OUT / f"results_{'_'.join(names)}.json", "w"), indent=1)
