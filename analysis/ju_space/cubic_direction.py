"""Along which combination of x and a does a network vary that regresses y = x^3 + a x^2 + x + b?
Background: Bottazzi (Wolfram Community, 16 July 2026, "Interpreting How Neural Nets Regress Cubic Polynomials") finds SAE
features in the last hidden layer of a small MLP that depend only on t = x + c a with c between 0.175 and 0.35 (median
0.30) and compares this with the substitution t = x + a / 3 of Cardano's method.
Part 1 (no network): the target itself changes in the (x, a) plane along x + c a with c = (dy/da) / (dy/dx)
  = x^2 / (3 x^2 + 2 a x + 1). Printed: the main direction of the gradient over the box x, a in [-3, 3] (top eigenvector of the
  mean outer product of the gradient) and the spread of the local c, weighted by the length of the gradient.
  The post's own test applied to the target: how well is y itself (and y without b, and y passed through ReLU, tanh or a
  sigmoid after standardising) described as a function of x + c a alone, and for which c (bucket test as in part 2).
Part 2 (own small networks, several seeds and activation functions; MLP 3-15-15-15-15-1, inputs (x, a, b) uniform in
  [-3, 3]^3, target standardised, Adam): after training,
  layer_c   the main direction, in the (x, a) plane, along which the last hidden layer as a whole varies (top eigenvector of
            the mean of J^T J, J = Jacobian of the last hidden layer with respect to the inputs); also for the untrained net
  unit_c    per unit of the last hidden layer the c in [-1, 1] (steps of 0.025) for which the unit is best described as a
            function of x + c a alone (50 buckets fitted on one half of 20,000 points, R^2 on the other half, as in the post);
            reported for the units with R^2 above 0.9, with the share of the unit's variance explained by b alone
Usage: python cubic_direction.py <out.json> [seeds=5] [steps=20000]      env: NT (1)"""
import json, os, sys
import numpy as np, torch, torch.nn as nn

out = sys.argv[1]; SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 20000; torch.set_num_threads(int(os.environ.get("NT", "1")))
rng = np.random.default_rng(0); x, a = rng.uniform(-3, 3, 2_000_000), rng.uniform(-3, 3, 2_000_000); gx, ga = 3 * x ** 2 + 2 * a * x + 1, x ** 2
top = lambda M: (lambda v: float(v[1] / v[0]))(np.linalg.eigh(M)[1][:, -1]); M = np.array([[np.mean(gx * gx), np.mean(gx * ga)], [np.mean(gx * ga), np.mean(ga * ga)]])
r = ga / gx; w = np.hypot(gx, ga); ok = gx > 0; o = np.argsort(r[ok]); cw = np.cumsum(w[ok][o]) / w[ok].sum(); q = {p: float(r[ok][o][np.searchsorted(cw, p)]) for p in (0.1, 0.25, 0.5, 0.75, 0.9)}
res = dict(target=dict(main_c=top(M), least_squares_c=float(M[0, 1] / M[0, 0]), weighted_quantiles_of_local_c=q, share_of_gradient_weight_with_positive_dydx=float(w[ok].sum() / w.sum()),
                       main_c_for_abs_x_at_least={str(lo): top(np.array([[np.mean(gx[m] ** 2), np.mean(gx[m] * ga[m])], [np.mean(gx[m] * ga[m]), np.mean(ga[m] ** 2)]])) for lo in (1.0, 2.0, 2.5) for m in [np.abs(x) >= lo]}), nets=[])
print(f"target: main gradient direction c = {res['target']['main_c']:.3f} (least squares {res['target']['least_squares_c']:.3f}); local c, weighted quantiles 10/25/50/75/90 %: " + " / ".join(f"{v:.2f}" for v in q.values()), flush=True)
torch.manual_seed(12345)                                                   # fixed start for the scale of the target (the networks are seeded one by one below)
f = lambda X: X[:, 0] ** 3 + X[:, 1] * X[:, 0] ** 2 + X[:, 0] + X[:, 2]; ys = float(f(torch.rand(1_000_000, 3) * 6 - 3).std())


def bucket(v, X, cs=np.arange(-1, 1.0001, 0.025)):
    """Best c and held-out R^2 of v as a function of x + c a alone (50 buckets fitted on the first half of the points); R^2 at every c."""
    n = len(X) // 2; out = {}
    for c in cs:
        t = X[:, 0] + c * X[:, 1]; edges = np.quantile(t[:n], np.linspace(0, 1, 51)[1:-1]); bi = np.searchsorted(edges, t); mu = np.array([v[:n][bi[:n] == k].mean() if (bi[:n] == k).any() else v[:n].mean() for k in range(50)])
        out[round(float(c), 3)] = float(1 - ((v[n:] - mu[bi[n:]]) ** 2).sum() / ((v[n:] - v[n:].mean()) ** 2).sum())
    best = max(out, key=out.get); return best, out[best], out


Xt = np.random.default_rng(1).uniform(-3, 3, (20000, 3)); yt = Xt[:, 0] ** 3 + Xt[:, 1] * Xt[:, 0] ** 2 + Xt[:, 0] + Xt[:, 2]; z = (yt - yt.mean()) / yt.std(); res["target"]["post_test"] = {}
for name, v in (("y", yt), ("y without b", yt - Xt[:, 2]), ("relu(y)", np.maximum(z, 0)), ("tanh(y)", np.tanh(z)), ("sigmoid(y)", 1 / (1 + np.exp(-z)))):
    c, r2, allr = bucket(v, Xt); res["target"]["post_test"][name] = dict(best_c=c, r2=r2, r2_at_0_25=allr[0.25], r2_at_0_325=allr[0.325], r2_at_third=float(np.interp(1 / 3, sorted(allr), [allr[k] for k in sorted(allr)])), r2_at_0=allr[0.0])
    print(f"the post's test on {name}: best c {c:.3f}, R^2 {r2:.3f} (at c = 0.25: {allr[0.25]:.3f}, at 0.325: {allr[0.325]:.3f}, at 0: {allr[0.0]:.3f})", flush=True)
res["target"]["share_of_gradient_energy_on_b"] = float(1 / (1 + np.mean(gx * gx) + np.mean(ga * ga)))
ACT = dict(relu=nn.ReLU, tanh=nn.Tanh, silu=nn.SiLU)


def layer_c(net, X):
    """Main direction in the (x, a) plane along which the last hidden layer varies; share of its sensitivity that goes to b."""
    X = X.clone().requires_grad_(True); h = net[:-1](X); J = torch.stack([torch.autograd.grad(h[:, j].sum(), X, retain_graph=True)[0] for j in range(h.shape[1])], 1)   # [N, units, 3]
    G = torch.einsum("nui,nuj->ij", J, J).double().numpy() / len(X); return top(G[:2, :2]), float(G[2, 2] / np.trace(G))


def unit_c(h, X):
    n = len(X) // 2; cs = np.arange(-1, 1.0001, 0.025); rows = []
    for j in range(h.shape[1]):
        v = h[:, j]
        if v[n:].var() < 1e-10: continue
        best = (-9, None)
        for c in cs:
            t = X[:, 0] + c * X[:, 1]; edges = np.quantile(t[:n], np.linspace(0, 1, 51)[1:-1]); bi = np.searchsorted(edges, t); mu = np.array([v[:n][bi[:n] == k].mean() if (bi[:n] == k).any() else v[:n].mean() for k in range(50)])
            r2 = 1 - ((v[n:] - mu[bi[n:]]) ** 2).sum() / ((v[n:] - v[n:].mean()) ** 2).sum(); best = max(best, (float(r2), float(c)))
        bb = np.searchsorted(np.quantile(X[:n, 2], np.linspace(0, 1, 51)[1:-1]), X[:, 2]); mb = np.array([v[:n][bb[:n] == k].mean() for k in range(50)]); rows.append(dict(unit=j, c=best[1], r2=best[0], r2_by_b=float(1 - ((v[n:] - mb[bb[n:]]) ** 2).sum() / ((v[n:] - v[n:].mean()) ** 2).sum())))
    return rows


for act in ("relu", "tanh", "silu"):
    for seed in range(SEEDS):
        torch.manual_seed(seed); net = nn.Sequential(nn.Linear(3, 15), ACT[act](), nn.Linear(15, 15), ACT[act](), nn.Linear(15, 15), ACT[act](), nn.Linear(15, 15), ACT[act](), nn.Linear(15, 1)); Xe = torch.rand(20000, 3) * 6 - 3; c0, _ = layer_c(net, Xe[:4000])
        opt = torch.optim.Adam(net.parameters(), lr=3e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
        for st in range(STEPS):
            X = torch.rand(1024, 3) * 6 - 3; loss = ((net(X)[:, 0] - f(X) / ys) ** 2).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        with torch.no_grad(): r2 = float(1 - ((net(Xe)[:, 0] - f(Xe) / ys) ** 2).mean() / (f(Xe) / ys).var()); h = net[:-1](Xe).numpy()
        cl, sb = layer_c(net, Xe[:4000]); U = unit_c(h, Xe.numpy()); good = [u for u in U if u["r2"] > 0.9]
        res["nets"].append(dict(act=act, seed=seed, fit_r2=r2, layer_c=cl, layer_c_untrained=c0, share_of_sensitivity_to_b=sb, units=U, units_r2_above_0_9=len(good), median_c_of_these=float(np.median([u["c"] for u in good])) if good else None))
        print(f"{act} seed {seed}: fit R^2 {r2:.4f} | last hidden layer varies along x + {cl:.3f} a (untrained {c0:.3f}), share of its sensitivity to b {sb:.3f} | units described by x + c a with R^2 > 0.9: {len(good)} of {len(U)}, their c: " + (" ".join(f"{u['c']:.3f}" for u in sorted(good, key=lambda u: u['c'])) or "-"), flush=True)
        json.dump(res, open(out, "w"), indent=1)
for act in ("relu", "tanh", "silu"):
    N = [n for n in res["nets"] if n["act"] == act]; cs = [u["c"] for n in N for u in n["units"] if u["r2"] > 0.9]
    print(f"{act}: layer c over seeds " + " ".join(f"{n['layer_c']:.3f}" for n in N) + f" | units with R^2 > 0.9: {len(cs)}, median c " + (f"{np.median(cs):.3f}, quartiles {np.quantile(cs, 0.25):.3f} to {np.quantile(cs, 0.75):.3f}" if cs else "-"))
