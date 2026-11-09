"""Adversarial checks on the trisector toy (same training code/seeds as trisect_toy.py).

Per (task, seed): retrain the identical model, then
  - recompute the original diff-split numbers (sanity: must match results.json)
  - per-pair split ratios and a flip criterion (edited output closer to y2 than to y1)
  - gradient alignment: cos(local output gradient at layer L, average gradient)
  - linear decodability of the network's own output from layer L (1 - R^2)
  - random-init baseline probes (same architecture, untrained)
  - unexplained-variance probes for target-specific residuals (s beyond l,a,b,c and vice versa; cos(C/3) beyond C,cosC,cos(C/2))
  - linear CKA between nets on a common input set (T vs B, seed vs seed)
Bottleneck nets (provable computational order): 3-64-64-[3 linear]-64-64-15-1, the 3-unit bottleneck supervised to equal
  (C, a, b) ("inter": intermediate C computed first), (s, a, b) ("answer": answer written first), or unsupervised ("free").
"""
import copy, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
sys.argv = sys.argv[:1]
import trisect_toy as T  # noqa: E402

torch.set_num_threads(2)
WIDTHS = T.WIDTHS


def Xof(F):
    return np.stack([F["a"], F["b"], F["c"]], 1)


def ols_resid(y, cols):
    A = np.c_[np.stack(cols, 1), np.ones(len(y))]
    return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]


def probe(Htr, ytr, Hte, yte):
    w, b0 = T.ridge(Htr, ytr)
    return float(T.r2(yte, Hte @ w + b0))


def cka(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    return float(np.linalg.norm(A.T @ B) ** 2 / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B) + 1e-12))


def train(m, Xtr, Ytr, extra_loss=None, epochs=None):
    epochs = epochs or T.EPOCHS
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    steps = epochs * math.ceil(len(Xtr) / 512)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 512):
            idx = perm[i:i + 512]
            if extra_loss is None:
                loss = ((m(Xtr[idx]) - Ytr[idx]) ** 2).mean()
            else:
                loss = extra_loss(m, idx)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()
    return m


def split_stats(fwd_tail, Hbg, H1, H2, y1, y2):
    """fwd_tail(h) -> scalar output from layer-L state. Tang-style split of D = H2 - H1 along the average gradient."""
    h = Hbg.clone().requires_grad_(True)
    fwd_tail(h).sum().backward()
    G = h.grad.numpy()
    J = G.mean(0); jhat = J / np.linalg.norm(J)
    cosloc = (G @ jhat) / (np.linalg.norm(G, axis=1) + 1e-12)
    D = (H2 - H1).numpy()
    Da = (D @ jhat)[:, None] * jhat[None, :]
    dys = (y2 - y1).numpy()
    out = {}
    with torch.no_grad():
        yA = fwd_tail(H1 + torch.tensor(Da, dtype=torch.float32)).numpy()
        yO = fwd_tail(H1 + torch.tensor(D - Da, dtype=torch.float32)).numpy()
    y1n = y1.numpy()
    coef = lambda dy: float((dy * dys).sum() / (dys ** 2).sum())
    out["diff_answer_part"] = coef(yA - y1n); out["diff_orth_part"] = coef(yO - y1n)
    big = np.abs(dys) > 0.1                       # output in standardized units
    rA = (yA - y1n)[big] / dys[big]; rO = (yO - y1n)[big] / dys[big]
    out["per_pair_rA_median"] = float(np.median(rA)); out["per_pair_rA_q10_q90"] = [float(np.quantile(rA, .1)), float(np.quantile(rA, .9))]
    out["per_pair_rO_median"] = float(np.median(rO))
    fA, fO = rA > 0.5, rO > 0.5
    out["flip_A_only"] = float((fA & ~fO).mean()); out["flip_O_only"] = float((~fA & fO).mean())
    out["flip_both"] = float((fA & fO).mean()); out["flip_neither"] = float((~fA & ~fO).mean())
    out["grad_cos_mean"] = float(np.mean(np.abs(cosloc)))
    out["grad_energy_along_jbar"] = float(np.mean((G @ jhat) ** 2) / np.mean((G ** 2).sum(1)))
    out["norm_share_D_along_jbar"] = float(np.mean((D @ jhat) ** 2) / np.mean((D ** 2).sum(1)))
    return out


def tail_fn(m, L):
    def f(h):
        x = h
        for lin in m.hidden[L + 1:]:
            x = torch.relu(lin(x))
        return m.out(x).squeeze(-1)
    return f


def run_toy(task, seed, common_X):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tr, te = T.features(*T.triangles(T.N_TRAIN, rng)), T.features(*T.triangles(T.N_TEST, rng))
    xm, xs = Xof(tr).mean(0), Xof(tr).std(0)
    ym, ys = tr[task].mean(), tr[task].std()
    Xtr = torch.tensor((Xof(tr) - xm) / xs, dtype=torch.float32); Ytr = torch.tensor((tr[task] - ym) / ys, dtype=torch.float32)
    Xte = torch.tensor((Xof(te) - xm) / xs, dtype=torch.float32)
    m = T.MLP()
    m0 = copy.deepcopy(m)
    train(m, Xtr, Ytr)
    with torch.no_grad():
        yte, Hte = m(Xte, return_h=True); _, Htr = m(Xtr[:50_000], return_h=True)
        _, H0te = m0(Xte, return_h=True); _, H0tr = m0(Xtr[:50_000], return_h=True)
        _, Hcom = m(torch.tensor((common_X - xm) / xs, dtype=torch.float32), return_h=True)
    trf = {k: v[:50_000] for k, v in tr.items()}
    a2, b2 = rng.uniform(0.5, 2.0, T.N_PAIRS), rng.uniform(0.5, 2.0, T.N_PAIRS)
    P1, P2 = T.features(*T.triangles(T.N_PAIRS, rng, (a2, b2))), T.features(*T.triangles(T.N_PAIRS, rng, (a2, b2)))
    X1 = torch.tensor((Xof(P1) - xm) / xs, dtype=torch.float32); X2 = torch.tensor((Xof(P2) - xm) / xs, dtype=torch.float32)
    with torch.no_grad():
        y1, H1 = m(X1, return_h=True); y2, H2 = m(X2, return_h=True)
    # residual targets
    res_tr = dict(s_res=ols_resid(trf["s"], [trf["l"], trf["a"], trf["b"], trf["c"]]),
                  l_res=ols_resid(trf["l"], [trf["s"], trf["a"], trf["b"], trf["c"]]),
                  cosC3_res=ols_resid(trf["cosC3"], [trf["C"], trf["cosC"], trf["cosC2"]]))
    res_te = dict(s_res=ols_resid(te["s"], [te["l"], te["a"], te["b"], te["c"]]),
                  l_res=ols_resid(te["l"], [te["s"], te["a"], te["b"], te["c"]]),
                  cosC3_res=ols_resid(te["cosC3"], [te["C"], te["cosC"], te["cosC2"]]))
    out = dict(task=task, seed=seed, layers=[])
    ynet = yte.numpy()
    for L in range(len(WIDTHS)):
        Ltr, Lte = Htr[L].numpy(), Hte[L].numpy()
        d = dict(layer=L + 1)
        d["unexpl_var"] = {f: 1 - probe(Ltr, trf[f], Lte, te[f]) for f in ("cosC", "C", "cosC3", "cosC2", "s", "l")}
        d["resid_probe_r2"] = {f: probe(Ltr, res_tr[f], Lte, res_te[f]) for f in res_tr}
        d["own_output_unexpl_var"] = 1 - probe(Lte[:10000], ynet[:10000], Lte[10000:], ynet[10000:])
        d["random_init_probe_r2"] = {f: probe(H0tr[L].numpy(), trf[f], H0te[L].numpy(), te[f])
                                     for f in ("cosC", "C", "cosC3", "cosC2", "s", "l")}
        d["random_init_resid_probe_r2"] = {f: probe(H0tr[L].numpy(), res_tr[f], H0te[L].numpy(), res_te[f]) for f in res_tr}
        d["split"] = split_stats(tail_fn(m, L), Hte[L][:5000], H1[L], H2[L], y1, y2)
        out["layers"].append(d)
    return out, [h.numpy() for h in Hcom]


class BMLP(nn.Module):
    """3-64-64-[3 linear bottleneck]-64-64-15-1"""
    def __init__(self):
        super().__init__()
        self.pre = nn.ModuleList([nn.Linear(3, 64), nn.Linear(64, 64)])
        self.bott = nn.Linear(64, 3)
        self.post = nn.ModuleList([nn.Linear(3, 64), nn.Linear(64, 64), nn.Linear(64, 15)])
        self.out = nn.Linear(15, 1)

    def code(self, x):
        for l in self.pre:
            x = torch.relu(l(x))
        return self.bott(x)

    def tail(self, z):
        for l in self.post:
            z = torch.relu(l(z))
        return self.out(z).squeeze(-1)

    def forward(self, x):
        return self.tail(self.code(x))


def run_bottleneck(variant, seed, task="s"):
    rng = np.random.default_rng(100 + seed)
    torch.manual_seed(100 + seed)
    tr, te = T.features(*T.triangles(T.N_TRAIN, rng)), T.features(*T.triangles(T.N_TEST, rng))
    xm, xs = Xof(tr).mean(0), Xof(tr).std(0)
    ym, ys = tr[task].mean(), tr[task].std()
    std = lambda v, ref: (v - ref.mean()) / ref.std()
    Xtr = torch.tensor((Xof(tr) - xm) / xs, dtype=torch.float32); Ytr = torch.tensor((tr[task] - ym) / ys, dtype=torch.float32)
    first = {"inter": "C", "answer": task}.get(variant)
    Ztr = None
    if first is not None:
        Ztr = torch.tensor(np.stack([std(tr[first], tr[first]), std(tr["a"], tr["a"]), std(tr["b"], tr["b"])], 1), dtype=torch.float32)
    m = BMLP()

    def loss(mm, idx):
        z = mm.code(Xtr[idx])
        l = ((mm.tail(z) - Ytr[idx]) ** 2).mean()
        if Ztr is not None:
            l = l + ((z - Ztr[idx]) ** 2).mean()
        return l
    t0 = time.time()
    train(m, Xtr, Ytr, extra_loss=loss)
    Xte = torch.tensor((Xof(te) - xm) / xs, dtype=torch.float32)
    with torch.no_grad():
        yte = m(Xte).numpy() * ys + ym
        zte = m.code(Xte)
    res = dict(variant=variant, seed=seed, test_r2=float(T.r2(te[task], yte)), minutes=(time.time() - t0) / 60)
    if first is not None:
        res["bottleneck_fit_r2"] = float(T.r2(std(te[first], tr[first]), zte[:, 0].numpy()))
    a2, b2 = rng.uniform(0.5, 2.0, T.N_PAIRS), rng.uniform(0.5, 2.0, T.N_PAIRS)
    P1, P2 = T.features(*T.triangles(T.N_PAIRS, rng, (a2, b2))), T.features(*T.triangles(T.N_PAIRS, rng, (a2, b2)))
    X1 = torch.tensor((Xof(P1) - xm) / xs, dtype=torch.float32); X2 = torch.tensor((Xof(P2) - xm) / xs, dtype=torch.float32)
    with torch.no_grad():
        z1, z2 = m.code(X1), m.code(X2)
        y1, y2 = m.tail(z1), m.tail(z2)
    res["split_at_bottleneck"] = split_stats(m.tail, zte[:5000], z1, z2, y1, y2)
    # also split at the first post-bottleneck hidden layer (64 wide)
    def h1(z):
        return torch.relu(m.post[0](z))
    def tail1(h):
        x = torch.relu(m.post[1](h)); x = torch.relu(m.post[2](x)); return m.out(x).squeeze(-1)
    with torch.no_grad():
        g1, g2, gbg = h1(z1), h1(z2), h1(zte[:5000])
    res["split_at_post1"] = split_stats(tail1, gbg, g1, g2, y1, y2)
    return res


if __name__ == "__main__":
    t0 = time.time()
    out = {"toy": [], "bottleneck": []}
    if MODE in ("all", "toy"):
        rng = np.random.default_rng(7)
        F = T.features(*T.triangles(200_000, rng))
        def r2lin(y, cols):
            return float(1 - ols_resid(y, cols).var() / y.var())
        out["input_collinearity_r2"] = {
            "cosC3~C": r2lin(F["cosC3"], [F["C"]]), "cosC3~cosC": r2lin(F["cosC3"], [F["cosC"]]),
            "cosC2~cosC3": r2lin(F["cosC2"], [F["cosC3"]]), "C~cosC": r2lin(F["C"], [F["cosC"]]),
            "cosC3~C,cosC,cosC2": r2lin(F["cosC3"], [F["C"], F["cosC"], F["cosC2"]]),
            "s~l": r2lin(F["s"], [F["l"]]), "s~l,a,b,c": r2lin(F["s"], [F["l"], F["a"], F["b"], F["c"]]),
        }
        common = Xof(T.features(*T.triangles(5000, np.random.default_rng(999))))
        reps = {}
        for task, seed in (("s", 0), ("l", 0), ("s", 1)):
            r, H = run_toy(task, seed, common)
            out["toy"].append(r); reps[(task, seed)] = H
            np.savez(HERE / f"reps_{task}{seed}.npz", *H)
            json.dump(out, open(HERE / "toy_extra_results.json", "w"), indent=1)
            print(task, seed, f"{(time.time()-t0)/60:.1f} min", flush=True)
            for d in r["layers"]:
                s = d["split"]
                print(f"  L{d['layer']} A {s['diff_answer_part']:.3f} O {s['diff_orth_part']:.3f} | rA med {s['per_pair_rA_median']:.3f} "
                      f"flips A-only {s['flip_A_only']:.3f} O-only {s['flip_O_only']:.3f} both {s['flip_both']:.3f} neither {s['flip_neither']:.3f} "
                      f"| gradcos {s['grad_cos_mean']:.3f} gE {s['grad_energy_along_jbar']:.3f} | own-out 1-R2 {d['own_output_unexpl_var']:.2e} "
                      f"| unexpl s {d['unexpl_var']['s']:.2e} l {d['unexpl_var']['l']:.2e} cosC3 {d['unexpl_var']['cosC3']:.2e} "
                      f"| resid {({k: round(v, 3) for k, v in d['resid_probe_r2'].items()})} | rand-init {({k: round(v, 3) for k, v in d['random_init_probe_r2'].items()})} "
                      f"| rand-init resid {({k: round(v, 3) for k, v in d['random_init_resid_probe_r2'].items()})}", flush=True)
            if len(reps) >= 2:
                cka_res = {}
                for L in range(len(WIDTHS)):
                    row = {}
                    keys = list(reps)
                    for i in range(len(keys)):
                        for j in range(i + 1, len(keys)):
                            row[f"{keys[i][0]}{keys[i][1]}-{keys[j][0]}{keys[j][1]}"] = cka(reps[keys[i]][L], reps[keys[j]][L])
                    cka_res[f"L{L+1}"] = row
                out["cka"] = cka_res
                print("CKA", json.dumps(cka_res), flush=True)
                json.dump(out, open(HERE / "toy_extra_results.json", "w"), indent=1)
    if MODE in ("all", "bott"):
        T.EPOCHS = 20
        for variant in ("inter", "answer", "free"):
            for seed in (0,):
                r = run_bottleneck(variant, seed)
                out["bottleneck"].append(r)
                sb, sp = r["split_at_bottleneck"], r["split_at_post1"]
                print(f"bottleneck {variant} seed {seed}: test R2 {r['test_r2']:.5f} fit {r.get('bottleneck_fit_r2', float('nan')):.4f} "
                      f"| at bottleneck A {sb['diff_answer_part']:.3f} O {sb['diff_orth_part']:.3f} rAmed {sb['per_pair_rA_median']:.3f} q10-90 {sb['per_pair_rA_q10_q90']} flipsA {sb['flip_A_only']+sb['flip_both']:.3f} "
                      f"flipsO {sb['flip_O_only']+sb['flip_both']:.3f} neither {sb['flip_neither']:.3f} gradcos {sb['grad_cos_mean']:.3f} normshare {sb['norm_share_D_along_jbar']:.3f}"
                      f" | post1 A {sp['diff_answer_part']:.3f} O {sp['diff_orth_part']:.3f} neither {sp['flip_neither']:.3f} gradcos {sp['grad_cos_mean']:.3f}", flush=True)
                json.dump(out, open(HERE / "bottleneck_results.json", "w"), indent=1)
    print(f"done {(time.time()-t0)/60:.1f} min")
