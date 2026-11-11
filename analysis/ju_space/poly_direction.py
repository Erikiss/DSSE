"""Along which combination of its first two inputs does a network vary that regresses a polynomial, a product or a SwiGLU unit?
Extends cubic_direction.py (README 8.7). Targets, inputs (p, q, b) uniform in [-3, 3]^3, the third enters additively:
  deg2    p^2 + q p + b               completing the square uses p + q/2
  deg3    p^3 + q p^2 + p + b         the cubic of Bottazzi's post (there x, a, b); Cardano's substitution is p + q/3
  deg4    p^4 + q p^3 + p + b         p + q/4
  prod    p q + b                     a purely bilinear unit
  swiglu  silu(p) q + b               a SwiGLU unit with gate input p and gated signal q
  deg2w, deg3w   deg2 and deg3 with q uniform in [-6, 6]: the shift q/n of the algebra does not depend on the distribution of
                 the inputs, the main direction of the gradient does; this separates the two readings
A direction is the combination cos(psi) p + sin(psi) q; its slope c = tan(psi) is the c of "x + c a" in the post.
Part 1 (no network), per target:
  main direction of the gradient: top eigenvector of the mean outer product of the gradient with respect to (p, q), and the share
  of the gradient energy along it (0.5 = no preferred direction, 1 = the target is a function of one combination);
  the post's bucket test applied to the target itself: R^2 of the target as a function of one combination alone (50 buckets
  fitted on one half of 20,000 points, R^2 on the other half), for the best angle (1 degree steps) and at named slopes.
Part 2 (own networks, SEEDS seeds each; MLPs 3-15-15-15-15-1 with ReLU, tanh and SiLU; with ARCHS=post also 3-64-64-64-15-1 with
  ReLU, the main network of the post according to its LessWrong version): after training (Adam, cosine schedule, batch 1024)
  layer   main direction along which the last hidden layer varies (top eigenvector of the mean of J^T J, J = Jacobian of the
          layer with respect to (p, q)) and the share of its sensitivity along it; also for the untrained network
  units   per unit of the last hidden layer the best angle and the R^2 of the bucket test (units that do not vary are left out)
Part 3, depth and degree: the cubic deg3 fitted by networks of purely bilinear units (h -> (W1 h + c1) * (W2 h + c2), then a
  linear map) with one and with two layers and by SwiGLU networks (h -> down(silu(gate h) * up h)) with one and two layers,
  width 16; and the best polynomial of degree at most 2 in (p, q, b).
Part 3 uses 5 seeds and gives the best polynomials of degree 1, 2 and 3 as ceilings.
Usage: python poly_direction.py <out.json> [seeds=5] [steps=8000]      env: NT (1), TARGETS (comma list), PARTS (default 123), ARCHS (std | post | all)"""
import json, os, sys
import numpy as np, torch, torch.nn as nn

out = sys.argv[1]; SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 8000; torch.set_num_threads(int(os.environ.get("NT", "1")))
TARGETS = os.environ.get("TARGETS", "deg2,deg3,deg4,prod,swiglu,deg2w,deg3w").split(","); PARTS = os.environ.get("PARTS", "123"); silu = lambda v: v / (1 + np.exp(-v))
F = dict(deg2=lambda p, q, b: p ** 2 + q * p + b, deg3=lambda p, q, b: p ** 3 + q * p ** 2 + p + b, deg4=lambda p, q, b: p ** 4 + q * p ** 3 + p + b, prod=lambda p, q, b: p * q + b, swiglu=lambda p, q, b: torch.nn.functional.silu(p) * q + b if torch.is_tensor(p) else silu(p) * q + b)
GRAD = dict(deg2=lambda p, q: (2 * p + q, p), deg3=lambda p, q: (3 * p ** 2 + 2 * q * p + 1, p ** 2), deg4=lambda p, q: (4 * p ** 3 + 3 * q * p ** 2 + 1, p ** 3), prod=lambda p, q: (q, p), swiglu=lambda p, q: ((1 / (1 + np.exp(-p)) + p * np.exp(-p) / (1 + np.exp(-p)) ** 2) * q, silu(p)))
NAMED = dict(deg2={"1/2": 0.5}, deg3={"1/3": 1 / 3}, deg4={"1/4": 0.25}, prod={"bisector +45": 1.0, "bisector -45": -1.0}, swiglu={"bisector +45": 1.0, "bisector -45": -1.0})
for t in ("deg2", "deg3"): F[t + "w"], GRAD[t + "w"], NAMED[t + "w"] = F[t], GRAD[t], NAMED[t]
QW = lambda t: 6.0 if t.endswith("w") else 3.0                            # half width of the range of the second input
ANG = np.arange(-89.0, 90.5, 1.0)


def top(M):
    """Angle in degrees (in (-90, 90]) of the top eigenvector of a symmetric 2x2 matrix, and the share of its eigenvalue."""
    w, v = np.linalg.eigh(np.asarray(M, dtype=np.float64)); e = v[:, -1]; a = float(np.degrees(np.arctan2(e[1], e[0]))); a = a - 180 if a > 90 else (a + 180 if a <= -90 else a); return a, float(w[-1] / w.sum())


def bucket(v, X, angles=ANG):
    """R^2 on the second half of the points of v as a function of cos(psi) p + sin(psi) q alone, for every angle psi."""
    n = len(X) // 2; r = np.empty(len(angles)); tot = ((v[n:] - v[n:].mean()) ** 2).sum()
    for i, a in enumerate(np.radians(angles)):
        t = np.cos(a) * X[:, 0] + np.sin(a) * X[:, 1]; edges = np.quantile(t[:n], np.linspace(0, 1, 51)[1:-1]); bi = np.searchsorted(edges, t); cnt = np.bincount(bi[:n], minlength=50); mu = np.bincount(bi[:n], weights=v[:n], minlength=50) / np.maximum(cnt, 1); mu[cnt == 0] = v[:n].mean()
        r[i] = 1 - ((v[n:] - mu[bi[n:]]) ** 2).sum() / tot
    return r


res = dict(targets={}, nets=[], depth=None); rng = np.random.default_rng(0); P, Q0 = rng.uniform(-3, 3, 2_000_000), rng.uniform(-3, 3, 2_000_000); Xt0 = np.random.default_rng(1).uniform(-3, 3, (20000, 3)); slope = lambda a: float(np.tan(np.radians(a)))
for t in TARGETS:
    Q = Q0 * QW(t) / 3; Xt = Xt0 * np.array([1.0, QW(t) / 3, 1.0]); gp, gq = GRAD[t](P, Q); M = [[np.mean(gp * gp), np.mean(gp * gq)], [np.mean(gp * gq), np.mean(gq * gq)]]; a, sh = top(M); y = F[t](Xt[:, 0], Xt[:, 1], Xt[:, 2]); r = bucket(y, Xt); r0 = bucket(y - Xt[:, 2], Xt); i = int(r.argmax())
    at = lambda c, rr=r: float(np.interp(np.degrees(np.arctan(c)), ANG, rr))
    res["targets"][t] = dict(gradient_angle=a, gradient_slope=slope(a), gradient_share=sh, mean_outer_product=[[float(x) for x in row] for row in M], share_of_gradient_energy_on_b=float(1 / (1 + M[0][0] + M[1][1])), bucket_best_angle=float(ANG[i]), bucket_best_slope=slope(ANG[i]), bucket_best_r2=float(r[i]),
                             bucket_without_b_best_angle=float(ANG[int(r0.argmax())]), bucket_without_b_best_r2=float(r0.max()), bucket_r2_at_gradient_direction=at(slope(a)), bucket_r2_at_named={k: at(c) for k, c in NAMED[t].items()}, bucket_r2_at_p_alone=float(r[list(ANG).index(0.0)]), bucket_r2_at_q_alone=float(r[-1]))
    print(f"{t}: gradient direction {a:.1f} deg (slope {slope(a):.3f}), share {sh:.3f} | bucket test on the target: best {ANG[i]:.0f} deg (slope {slope(ANG[i]):.3f}) R^2 {r[i]:.3f}; without b {r0.max():.3f}; at " + ", ".join(f"{k}: {at(c):.3f}" for k, c in NAMED[t].items()), flush=True)
json.dump(res, open(out, "w"), indent=1)
ACT = dict(relu=nn.ReLU, tanh=nn.Tanh, silu=nn.SiLU); STD = [("15x4 relu", (15, 15, 15, 15), "relu"), ("15x4 tanh", (15, 15, 15, 15), "tanh"), ("15x4 silu", (15, 15, 15, 15), "silu")]; POST = [("64-64-64-15 relu", (64, 64, 64, 15), "relu")]
ARCH = dict(std=STD, post=POST, all=STD + POST)[os.environ.get("ARCHS", "std")]


def mlp(sizes, act):
    L, d = [], 3
    for h in sizes: L += [nn.Linear(d, h), ACT[act]()]; d = h
    return nn.Sequential(*L, nn.Linear(d, 1))


def layer_dir(net, X):
    X = X.clone().requires_grad_(True); h = net[:-1](X); J = torch.stack([torch.autograd.grad(h[:, j].sum(), X, retain_graph=True)[0] for j in range(h.shape[1])], 1); G = torch.einsum("nui,nuj->ij", J, J).double().numpy() / len(X)
    a, sh = top(G[:2, :2]); return a, sh, float(G[2, 2] / np.trace(G))


def train(net, f, ys, seed, lr=3e-3, qw=3.0):
    box = torch.tensor([3.0, qw, 3.0]); draw = lambda n: (torch.rand(n, 3) * 2 - 1) * box; opt = torch.optim.Adam(net.parameters(), lr=lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    for st in range(STEPS):
        X = draw(1024); loss = ((net(X)[:, 0] - f(X[:, 0], X[:, 1], X[:, 2]) / ys) ** 2).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    Xe = draw(20000)
    with torch.no_grad(): return float(1 - ((net(Xe)[:, 0] - f(Xe[:, 0], Xe[:, 1], Xe[:, 2]) / ys) ** 2).mean() / (f(Xe[:, 0], Xe[:, 1], Xe[:, 2]) / ys).var()), Xe


if "2" in PARTS:
    for t in TARGETS:
        torch.manual_seed(12345); Z = (torch.rand(1_000_000, 3) * 2 - 1) * torch.tensor([3.0, QW(t), 3.0]); ys = float(F[t](Z[:, 0], Z[:, 1], Z[:, 2]).std())
        for name, sizes, act in ARCH:
            for seed in range(SEEDS):
                torch.manual_seed(seed); net = mlp(sizes, act); a0, s0, _ = layer_dir(net, (torch.rand(4000, 3) * 2 - 1) * torch.tensor([3.0, QW(t), 3.0])); r2, Xe = train(net, F[t], ys, seed, qw=QW(t)); a, sh, sb = layer_dir(net, Xe[:4000])
                with torch.no_grad(): h = net[:-1](Xe).numpy()
                U = []; Xn = Xe.numpy(); n = len(Xn) // 2
                for j in range(h.shape[1]):
                    if h[n:, j].var() < 1e-10: continue
                    r = bucket(h[:, j], Xn); i = int(r.argmax()); U.append(dict(unit=j, angle=float(ANG[i]), r2=float(r[i])))
                good = [u for u in U if u["r2"] > 0.9]; r2t = bucket((F[t](Xe[:, 0], Xe[:, 1], Xe[:, 2]) - Xe[:, 2]).numpy(), Xn).max()   # the same test on the target without b, same points
                res["nets"].append(dict(target=t, arch=name, seed=seed, fit_r2=r2, layer_angle=a, layer_slope=slope(a), layer_share=sh, layer_angle_untrained=a0, layer_share_untrained=s0, share_of_sensitivity_to_b=sb, units=U, units_r2_above_0_9=len(good), bucket_r2_of_target_without_b=float(r2t)))
                print(f"{t} {name} seed {seed}: fit R^2 {r2:.4f} | layer {a:.1f} deg (slope {slope(a):.3f}) share {sh:.3f} (untrained {a0:.1f} deg, share {s0:.3f}) | units with R^2 > 0.9: {len(good)} of {len(U)}; angles " + " ".join(f"{u['angle']:.0f}" for u in sorted(good, key=lambda u: u['angle'])) + f" | median R^2 of all units {np.median([u['r2'] for u in U]):.2f}", flush=True)
                json.dump(res, open(out, "w"), indent=1)


class Bil(nn.Module):
    def __init__(s, d, w, swiglu): super().__init__(); s.a = nn.Linear(d, w); s.b = nn.Linear(d, w); s.o = nn.Linear(w, w); s.swiglu = swiglu
    def forward(s, h): return s.o((torch.nn.functional.silu(s.a(h)) if s.swiglu else s.a(h)) * s.b(h))


if "3" in PARTS:
    torch.manual_seed(12345); Z = torch.rand(1_000_000, 3) * 6 - 3; y = F["deg3"](Z[:, 0], Z[:, 1], Z[:, 2]); ys = float(y.std()); Zn = Z[:200000].double().numpy(); yn = y[:200000].double().numpy()
    import itertools
    def mon(X, deg): return np.stack([np.prod([X[:, i] ** e for i, e in enumerate(ex)], 0) for ex in itertools.product(range(deg + 1), repeat=3) if sum(ex) <= deg], 1)
    ceil = {}
    for deg in (1, 2, 3): co = np.linalg.lstsq(mon(Zn[:100000], deg), yn[:100000], rcond=None)[0]; ceil[deg] = float(1 - ((yn[100000:] - mon(Zn[100000:], deg) @ co) ** 2).mean() / yn[100000:].var())
    res["depth"] = dict(best_polynomial_r2_by_degree=ceil, nets=[]); print("cubic deg3: best polynomial of degree 1 / 2 / 3: R^2 " + " / ".join(f"{ceil[d]:.4f}" for d in (1, 2, 3)), flush=True)
    for kind in ("bilinear", "swiglu"):
        for depth in (1, 2):
            for seed in range(5):
                torch.manual_seed(seed); net = nn.Sequential(*[Bil(3 if k == 0 else 16, 16, kind == "swiglu") for k in range(depth)], nn.Linear(16, 1)); r2, _ = train(net, F["deg3"], ys, seed, lr=3e-3)
                res["depth"]["nets"].append(dict(kind=kind, layers=depth, seed=seed, fit_r2=r2)); print(f"cubic deg3 by {kind} units, {depth} layer(s), seed {seed}: R^2 {r2:.6f}", flush=True); json.dump(res, open(out, "w"), indent=1)
json.dump(res, open(out, "w"), indent=1)
