"""Hand-built settings with KNOWN computational order, analysed with the same Tang-style split.

answer direction / answer plane = average (over a background set) of the output gradients w.r.t. the layer-L state
(scalar: 1 direction; vector: span of the avg gradients of the two answer-token logits, i.e. J-lens answer directions).
Edit = counterfactual difference at layer L; split into its projection on the answer plane (A) and the rest (O).
Reported: continuous share (regression coefficient of the part's effect on the full effect, as in the toy) and
flip accounting among items where the full edit flips the decision (A only / O only / both / neither).

S1  scalar, intermediate z on ONE direction, y = sin(w z)/w + beta*r   (downstream nonlinearity w, unrelated pathway beta)
S2  scalar, y = z*w (intermediate z times context w ~ N(mu_w,1)); identical math = "answer written directly but sign/gain-bound"
S3  vector output, two-hop lookup: h_L = E[z] + ctx; downstream memory lookup writes answer ans(z) (8 answers, 24 entities)
      lookup nonlinearity rho: hidden_j = (1-rho)*(k_j.h) + rho*relu(k_j.h - theta)/(1-theta)
    direct:   h_L = E[z] + ctx + lam*U[ans(z)], downstream identity (answer already written at L)
    binding:  h_L = E[z] + ctx + g*lam*U[ans(z)] + g*gam*u_g, logits = (u_g.h/gam) * U h  (answer written, sign-bound, g=+-1)
    learned:  downstream residual MLP (128 ReLU) trained by CE to map E[z]+ctx -> ans(z)
S4  pure flip-accounting model: additive split (no interaction) + top-1 threshold; which continuous shares give Tang's
    26% (A alone) / 12% (O alone) / 65% (neither)?
"""
import json, math, time
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
HERE = Path(__file__).parent
OUT = {}


def unit_rows(n, d, rng):
    X = rng.normal(size=(n, d)); return X / np.linalg.norm(X, axis=1, keepdims=True)


def T_(x):
    return torch.tensor(x, dtype=torch.float32)


def flip_table(fullflip, fA, fO):
    n = max(int(fullflip.sum()), 1)
    return dict(n_full_flips=int(fullflip.sum()),
                A_only=float((fullflip & fA & ~fO).sum() / n), O_only=float((fullflip & ~fA & fO).sum() / n),
                both=float((fullflip & fA & fO).sum() / n), neither=float((fullflip & ~fA & ~fO).sum() / n),
                A_alone_total=float((fullflip & fA).sum() / n), O_alone_total=float((fullflip & fO).sum() / n))


# ---------------------------------------------------------------- scalar machinery
def scalar_split(F, Hbg, H1, D):
    h = Hbg.clone().requires_grad_(True); F(h).sum().backward()
    G = h.grad; J = G.mean(0); j = J / J.norm()
    Da = (D @ j)[:, None] * j[None, :]; Do = D - Da
    with torch.no_grad():
        y1, yF, yA, yO, ybg = F(H1), F(H1 + D), F(H1 + Da), F(H1 + Do), F(Hbg)
    dys = yF - y1
    coef = lambda y: float(((y - y1) * dys).sum() / (dys ** 2).sum())
    tau = ybg.median()
    dec = lambda y: y > tau
    full = (dec(yF) != dec(y1)).numpy()
    ft = flip_table(full, (dec(yA) != dec(y1)).numpy(), (dec(yO) != dec(y1)).numpy())
    return dict(cont_A=coef(yA), cont_O=coef(yO),
                norm_share_A=float(((D @ j) ** 2).mean() / (D ** 2).sum(1).mean()),
                grad_energy_along_jbar=float(((G @ j) ** 2).mean() / (G ** 2).sum(1).mean()),
                threshold_flips=ft)


def S1():
    rng = np.random.default_rng(1); d, N = 16, 4000
    U = np.linalg.qr(rng.normal(size=(d, d)))[0].T
    uz, ur = T_(U[0]), T_(U[1])
    res = []
    for beta in (0.0, 0.3, 1.0):
        for w in (0.01, 1.0, 2.0, 3.0, 4.5, 6.0, 9.0):
            def mk(z, r):
                noise = T_(rng.normal(size=(len(z), d)) * 0.3)
                noise = noise - (noise @ uz)[:, None] * uz - (noise @ ur)[:, None] * ur
                return T_(z)[:, None] * uz + T_(r)[:, None] * ur + noise
            F = lambda h: torch.sin(w * (h @ uz)) / w + beta * (h @ ur)
            z, zp, r = rng.uniform(-1, 1, N), rng.uniform(-1, 1, N), rng.normal(size=N)
            H1 = mk(z, r); D = T_(zp - z)[:, None] * uz
            Hbg = mk(rng.uniform(-1, 1, N), rng.normal(size=N))
            s = scalar_split(F, Hbg, H1, D); s.update(beta=beta, omega=w, mean_gprime=float(np.sin(w) / w))
            res.append(s)
            print(f"S1 beta {beta} omega {w}: cont A {s['cont_A']:.3f} O {s['cont_O']:.3f} | thr-flips A {s['threshold_flips']['A_alone_total']:.2f} "
                  f"O {s['threshold_flips']['O_alone_total']:.2f} neither {s['threshold_flips']['neither']:.2f} | E[g']={np.sin(w)/w:+.3f}", flush=True)
    return res


def S2():
    rng = np.random.default_rng(2); d, N = 16, 4000
    U = np.linalg.qr(rng.normal(size=(d, d)))[0].T
    uz, uw = T_(U[0]), T_(U[1])
    res = []
    for mu_w in (0.0, 0.5, 1.0, 2.0, 4.0):
        mk = lambda z, w: T_(z)[:, None] * uz + T_(w)[:, None] * uw
        F = lambda h: (h @ uz) * (h @ uw)
        z, zp, w = rng.uniform(-0.5, 1.5, N), rng.uniform(-0.5, 1.5, N), rng.normal(mu_w, 1, N)
        H1 = mk(z, w); D = T_(zp - z)[:, None] * uz
        Hbg = mk(rng.uniform(-0.5, 1.5, N), rng.normal(mu_w, 1, N))
        s = scalar_split(F, Hbg, H1, D); s.update(mu_w=mu_w)
        res.append(s)
        print(f"S2 y=z*w mu_w {mu_w}: cont A {s['cont_A']:.3f} O {s['cont_O']:.3f} | thr-flips A {s['threshold_flips']['A_alone_total']:.2f} "
              f"O {s['threshold_flips']['O_alone_total']:.2f} neither {s['threshold_flips']['neither']:.2f}", flush=True)
    return res


# ---------------------------------------------------------------- vector machinery (S3)
d, K, M = 64, 24, 8


def world(seed):
    rng = np.random.default_rng(seed)
    E = unit_rows(K, d, rng)                     # entity embeddings = entity unembeddings (J-lens readable intermediate)
    Ua = unit_rows(M, d, rng)                    # answer unembeddings
    ans = np.repeat(np.arange(M), K // M); rng.shuffle(ans)
    ug = unit_rows(1, d, rng)[0]
    return rng, T_(E), T_(Ua), ans, T_(ug)


def items(rng, ans, N, sig_c=0.5):
    z = rng.integers(0, K, N)
    zp = np.array([rng.choice(np.where(ans != ans[zi])[0]) for zi in z])   # counterfactual entity with a different answer
    c = rng.normal(size=(N, d)) * sig_c / math.sqrt(d)
    g = rng.choice([-1.0, 1.0], N)
    return z, zp, T_(c), T_(g)


def vector_split(F, Hbg, H1, D, a, ap):
    """F: (N,d)->(N,M) answer logits. Answer plane per item = span(Jbar[a], Jbar[ap])."""
    h = Hbg.clone().requires_grad_(True)
    Jbar = []
    for t in range(M):
        if h.grad is not None:
            h.grad = None
        F(h)[:, t].sum().backward(retain_graph=True)
        Jbar.append(h.grad.mean(0).clone())
    Jbar = torch.stack(Jbar)                          # (M, d)
    u1 = Jbar[a] / Jbar[a].norm(dim=1, keepdim=True)
    v = Jbar[ap] - (Jbar[ap] * u1).sum(1, keepdim=True) * u1
    u2 = v / v.norm(dim=1, keepdim=True)
    Da = (D * u1).sum(1, keepdim=True) * u1 + (D * u2).sum(1, keepdim=True) * u2
    Do = D - Da
    ldir = Jbar[ap] - Jbar[a]; ldir = ldir / ldir.norm(dim=1, keepdim=True)
    Dl = (D * ldir).sum(1, keepdim=True) * ldir
    idx = torch.arange(len(a))
    # local logit-diff gradient at H1 and its energy inside the plane
    hh = H1.clone().requires_grad_(True)
    lo = F(hh); (lo[idx, ap] - lo[idx, a]).sum().backward()
    gl = hh.grad
    gl_in = (gl * u1).sum(1) ** 2 + (gl * u2).sum(1) ** 2
    with torch.no_grad():
        L1, LF, LA, LO, LL = F(H1), F(H1 + D), F(H1 + Da), F(H1 + Do), F(H1 + Dl)
    ld = lambda Lg: (Lg[idx, ap] - Lg[idx, a])
    dF = ld(LF) - ld(L1)
    coef = lambda Lg: float(((ld(Lg) - ld(L1)) * dF).sum() / (dF ** 2).sum())
    top = lambda Lg: (Lg.argmax(1) == T_(ap).long()).numpy()
    base_ok = (L1.argmax(1) == T_(a).long()).numpy()
    full = top(LF) & base_ok
    return dict(cont_A=coef(LA), cont_O=coef(LO), cont_logitdiff_line=coef(LL),
                norm_share_A=float(((Da ** 2).sum(1)).mean() / (D ** 2).sum(1).mean()),
                local_grad_energy_in_plane=float((gl_in / (gl ** 2).sum(1)).mean()),
                base_acc=float(base_ok.mean()), full_flip_rate=float(full.sum() / max(base_ok.sum(), 1)),
                flips=flip_table(full, top(LA), top(LO)), flips_line_only=float((full & top(LL)).sum() / max(full.sum(), 1)))


def lookup_net(E, Ua, ans, rho, theta, lam):
    V = lam * Ua[torch.tensor(ans)]               # value of memory slot j writes the answer direction of ans(j)
    def F(h):
        s = h @ E.T
        hid = (1 - rho) * s + rho * torch.relu(s - theta) / (1 - theta)
        return (h + hid @ V) @ Ua.T
    return F


def S3():
    res = []
    rng, E, Ua, ans, ug = world(3)
    N = 3000
    z, zp, c, g = items(rng, ans, N)
    zb, _, cb, gb = items(rng, ans, 4000)
    a, ap = ans[z], ans[zp]
    lam = 3.0
    # two-step: intermediate at L, lookup after L
    for rho, theta in ((0.0, 0.5), (0.25, 0.5), (0.5, 0.5), (0.75, 0.5), (1.0, 0.5), (1.0, 0.25), (1.0, 0.7), (1.0, 0.0)):
        F = lookup_net(E, Ua, ans, rho, theta, lam)
        H1 = E[z] + c; Hbg = E[zb] + cb
        for kappa in ((1.0,) if not (rho == 1.0 and theta == 0.5) else (0.75, 1.0, 1.5, 2.0)):
            D = kappa * (E[zp] - E[z])
            s = vector_split(F, Hbg, H1, D, a, ap); s.update(setting="two_step_lookup", rho=rho, theta=theta, kappa=kappa)
            res.append(s); pr("S3 two-step", s, f"rho {rho} theta {theta} kappa {kappa}")
    # direct: answer already written at L, downstream identity
    Fd = lambda h: h @ Ua.T
    H1 = E[z] + c + lam * Ua[a]; Hbg = E[zb] + cb + lam * Ua[ans[zb]]
    Dnat = (E[zp] + lam * Ua[ap]) - (E[z] + lam * Ua[a])
    s = vector_split(Fd, Hbg, H1, Dnat, a, ap); s.update(setting="direct_identity", edit="natural_difference"); res.append(s); pr("S3 direct", s, "natural diff")
    s = vector_split(Fd, Hbg, H1, E[zp] - E[z], a, ap); s.update(setting="direct_identity", edit="intermediate_only"); res.append(s); pr("S3 direct", s, "intermediate-only swap")
    # mixed: half the answer written before L, half looked up after L (sharp lookup)
    Fm = lookup_net(E, Ua, ans, 1.0, 0.5, lam / 2)
    H1 = E[z] + c + lam / 2 * Ua[a]; Hbg = E[zb] + cb + lam / 2 * Ua[ans[zb]]
    Dnat = (E[zp] + lam / 2 * Ua[ap]) - (E[z] + lam / 2 * Ua[a])
    s = vector_split(Fm, Hbg, H1, Dnat, a, ap); s.update(setting="mixed_half_direct_half_lookup", edit="natural_difference"); res.append(s); pr("S3 mixed", s, "natural diff")
    s = vector_split(Fm, Hbg, H1, E[zp] - E[z], a, ap); s.update(setting="mixed_half_direct_half_lookup", edit="intermediate_only"); res.append(s); pr("S3 mixed", s, "intermediate-only swap")
    # binding: answer written at L but sign-bound to a context bit g; downstream multiplies back
    gam = 1.0
    Fb = lambda h: ((h @ ug) / gam)[:, None] * (h @ Ua.T)
    mkb = lambda zz, cc, gg: E[zz] + cc + gg[:, None] * lam * Ua[ans[zz]] + gg[:, None] * gam * ug
    H1 = mkb(z, c, g); Hbg = mkb(zb, cb, gb)
    Dnat = mkb(zp, c, g) - H1
    s = vector_split(Fb, Hbg, H1, Dnat, a, ap); s.update(setting="direct_sign_bound", edit="natural_difference"); res.append(s); pr("S3 binding", s, "natural diff")
    # learned two-step lookup
    torch.manual_seed(0)
    W1 = torch.nn.Linear(d, 128); W2 = torch.nn.Linear(128, d)
    opt = torch.optim.Adam(list(W1.parameters()) + list(W2.parameters()), lr=3e-3)
    Fl = lambda h: (h + W2(torch.relu(W1(h)))) @ Ua.T
    for step in range(1500):
        zz, _, cc, _ = items(rng, ans, 256)
        loss = torch.nn.functional.cross_entropy(Fl(E[zz] + cc), torch.tensor(ans[zz]))
        opt.zero_grad(); loss.backward(); opt.step()
    H1 = E[z] + c; Hbg = E[zb] + cb
    s = vector_split(Fl, Hbg, H1, E[zp] - E[z], a, ap); s.update(setting="learned_lookup_MLP", edit="intermediate_swap", final_loss=float(loss)); res.append(s); pr("S3 learned", s, "intermediate swap")
    return res


def pr(tag, s, extra):
    f = s["flips"]
    print(f"{tag} {extra}: base acc {s['base_acc']:.2f} full-flip {s['full_flip_rate']:.2f} | cont A {s['cont_A']:.3f} O {s['cont_O']:.3f} "
          f"line {s['cont_logitdiff_line']:.3f} | flips: A-only {f['A_only']:.2f} O-only {f['O_only']:.2f} both {f['both']:.2f} neither {f['neither']:.2f} "
          f"| normshare A {s['norm_share_A']:.3f} | local grad energy in plane {s['local_grad_energy_in_plane']:.3f}", flush=True)


# ---------------------------------------------------------------- S4: flip accounting of a purely additive split
def S4():
    rng = np.random.default_rng(4); N = 200_000
    target = np.array([0.26, 0.12, 0.65])
    fits = []
    for mu in np.linspace(0.0, 1.0, 21):
        for sd in (0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8):
            alpha = rng.normal(mu, sd, N)
            for ov in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5):
                r = 1 + rng.exponential(ov, N)          # full effect / margin among flipped items
                fA, fO = alpha * r > 1, (1 - alpha) * r > 1
                v = np.array([fA.mean(), fO.mean(), (~fA & ~fO).mean()])
                fits.append(dict(mu_alpha=float(mu), sd_alpha=sd, mean_overshoot=ov, A_alone=float(v[0]), O_alone=float(v[1]),
                                 neither=float(v[2]), err=float(np.abs(v - target).max())))
    fits.sort(key=lambda f: f["err"])
    good = [f for f in fits if f["err"] < 0.03]
    print("S4 best additive fits to Tang (A 26 / O 12 / neither 65):")
    for f in fits[:8]:
        print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()})
    print(f"S4 fits within 3pp: n={len(good)}, mu_alpha range {min(f['mu_alpha'] for f in good) if good else None}-{max(f['mu_alpha'] for f in good) if good else None}")
    # what the toy's continuous share (~0.96, sd ~0.05) would give under the same flip accounting
    toy_like = []
    for ov in (0.1, 0.35, 1.0):
        alpha = rng.normal(0.96, 0.05, N); r = 1 + rng.exponential(ov, N)
        fA, fO = alpha * r > 1, (1 - alpha) * r > 1
        toy_like.append(dict(mean_overshoot=ov, A_alone=float(fA.mean()), O_alone=float(fO.mean()), neither=float((~fA & ~fO).mean())))
    print("S4 toy-like alpha 0.96:", toy_like)
    return dict(best=fits[:20], n_within_3pp=len(good),
                mu_range_within_3pp=[min(f['mu_alpha'] for f in good), max(f['mu_alpha'] for f in good)] if good else None,
                toy_like=toy_like)


if __name__ == "__main__":
    t0 = time.time()
    OUT["S1"] = S1(); OUT["S2"] = S2(); OUT["S3"] = S3(); OUT["S4"] = S4()
    json.dump(OUT, open(HERE / "split_settings_results.json", "w"), indent=1)
    print(f"done {(time.time()-t0)/60:.2f} min")
