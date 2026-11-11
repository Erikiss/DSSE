"""How quadratic is an MLP unit on the data it sees? (README 9.4) For SwiGLU (z = silu(g) * u with g = gate(x), u = up(x), both
linear in the MLP input x), for the purely bilinear unit (z = g * u) and, for comparison, for ReLU (z = relu(p), p = fc1(x)).
Per unit: the best polynomial of degree 1, 2 and 3 in (g, u) (ReLU: in p), by least squares.
Part 1 (no network): R^2 of these three fits for z = silu(g) * u with Gaussian g (mean mu, spread sigma) and u (mean mu_u, spread
  1, correlation rho with g).
Part 2 (toy networks, kv_common), per stored state of a run and per block, at the '=' position over all 9312 problems:
  r2_1, r2_2, r2_3   R^2 of the fits of degree 1, 2, 3, fitted on a random half of the problems and read on the other half, averaged
                     over the units with the weights |output weights of the unit|^2 * variance of its activation
  write shares       of the variance of the MLP write (output weights applied to the unit activations): linear = 1 - U1,
                     quadratic = U1 - U2, cubic = U2 - U3, rest = U3, where Ud is the share of that variance which remains when
                     every unit is replaced by its fit of degree d (read on the half not used for the fit)
  affine             R^2 of the best affine map from the MLP input (128 numbers) to the MLP write (fitted on one half, read on the other)
  negative           gated units: share of (problem, unit) pairs with a negative gate input; ReLU: share that is off
  corr, cos          gated units: |correlation| of g and u over the problems per unit, and |cosine| of the gate and up weight
                     vectors. 1 = the unit is a function of ONE linear combination of its input (a square), 0 = a product of two
                     unrelated combinations; the quadratic form (w_g.x)(w_u.x) has its axes along the two bisectors of w_g and
                     w_u with eigenvalues |w_g||w_u|(cos +- 1)/2. For random weights |cos| is about 0.8 / sqrt(128) = 0.07
  surrogates         accuracy when every unit of both blocks (or of one block) is replaced at all positions by its fit of degree
                     1 ("linearised": the MLPs are linear maps) or of degree 2 ("quadratised": polynomials of degree 2). These
                     fits are made per unit and position on the TRAINING problems; accuracies are given for training and
                     held-out problems. Positions 0 and 2 carry the same token in every problem; there the fits are exact
Usage: python act_quad.py <out.json> <run_dir> [<run_dir> ...]      env: STEPS (comma list; default: 0, 2500, 5000 and the last two states)"""
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_common import fwd, load, all_problems, acc4, NL

torch.set_num_threads(int(os.environ.get("NT", "1"))); out, dirs = sys.argv[1], sys.argv[2:]


def feats(g, u, deg, relu):
    """Regressors [..., k] of degree deg in the two factors (ReLU: in the one input of the unit)."""
    f = [torch.ones_like(g), g] if relu else [torch.ones_like(g), g, u]
    if deg >= 2: f += [g * g] if relu else [g * g, g * u, u * u]
    if deg >= 3: f += [g ** 3] if relu else [g ** 3, g * g * u, g * u * u, u ** 3]
    return torch.stack(f, -1)


def fit(F, z):
    """Least squares per unit (and position): F [N, ..., k], z [N, ...] -> coefficients [..., k] (float64, columns scaled, tiny ridge)."""
    F, z = F.double(), z.double(); sc = F.abs().amax(0).clamp_min(1e-12); F = F / sc; A = torch.einsum("n...k,n...l->...kl", F, F); b = torch.einsum("n...k,n...->...k", F, z)
    A = A + 1e-10 * torch.diag_embed(torch.diagonal(A, dim1=-2, dim2=-1).clamp_min(1e-12)); return (torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1) / sc).float()


def table():
    rng = np.random.default_rng(0); n = 1_000_000; e1, e2 = rng.standard_normal(n), rng.standard_normal(n); rows = []
    for rho in (0.0, 0.5, 0.9):
        for mu_u in (0.0, 1.0):
            for sigma in (0.5, 1.0, 2.0):
                for mu in (-4, -2, -1, 0, 1, 2, 4):
                    g = mu + sigma * e1; u = mu_u + rho * e1 + np.sqrt(1 - rho ** 2) * e2; z = g / (1 + np.exp(-g)) * u; gc, uc = g - g.mean(), u - u.mean(); v = z.var(); r = []
                    X = [np.ones(n), gc, uc]; X2 = X + [gc * gc, gc * uc, uc * uc]; X3 = X2 + [gc ** 3, gc * gc * uc, gc * uc * uc, uc ** 3]
                    for cols in (X, X2, X3): D = np.stack(cols, 1); r.append(float(1 - ((z - D @ np.linalg.lstsq(D, z, rcond=None)[0]) ** 2).mean() / v))
                    rows.append(dict(rho=rho, mean_u=mu_u, sigma=sigma, mu=mu, r2_1=r[0], r2_2=r[1], r2_3=r[2], negative=float((g < 0).mean())))
    return rows


def state(net, seq, is_tr, half):
    tok = seq[:, :5]; relu = not hasattr(net.blocks[0], "gate"); kind = "relu" if relu else ("bilinear" if net.blocks[0].bilinear else "swiglu")
    with torch.no_grad(): lg, rec = fwd(net, tok, dict(rec_gate=True))
    row = dict(kind=kind, acc_train=acc4(lg[is_tr], seq[is_tr]), acc_held=acc4(lg[~is_tr], seq[~is_tr]), blocks=[]); C = []
    for l, b in enumerate(net.blocks):
        with torch.no_grad():
            x = rec[l]["xmid"]; z = rec[l]["h"]; g = rec[l]["gi"]; u = g if relu else rec[l]["gs"]; Wd = b.fc2.weight if relu else b.down.weight
            C.append([fit(feats(g[is_tr], u[is_tr], d, relu), z[is_tr]) for d in (1, 2)])                                   # surrogates: per unit and position, training problems
            g4, u4, z4 = g[:, 4], u[:, 4], z[:, 4]; gc, uc = g4 - g4.mean(0), u4 - u4.mean(0); wv = z4.var(0) * (Wd ** 2).sum(0); wv = wv / wv.sum(); A, B = half, ~half
            wvar = lambda P: float((((Wd @ P.T) - (Wd @ P.T).mean(1, keepdim=True)) ** 2).sum() / P.shape[0]); vt = wvar(z4[B]); r2, un = [], []
            for d in (1, 2, 3):
                zf = (feats(gc[B], uc[B], d, relu) * fit(feats(gc[A], uc[A], d, relu), z4[A])).sum(-1); res_ = z4[B] - zf; r2.append(float(((1 - (res_ ** 2).mean(0) / z4[B].var(0, unbiased=False).clamp_min(1e-12)) * wv).sum())); un.append(wvar(res_) / vt)
            X = torch.cat([x[:, 4], torch.ones(len(x), 1)], 1).double(); Wr = (Wd @ z4.T).T.double(); co = torch.linalg.lstsq(X[A], Wr[A]).solution; aff = float(1 - ((Wr[B] - X[B] @ co) ** 2).sum() / ((Wr[B] - Wr[B].mean(0)) ** 2).sum())
            d = dict(r2_1=r2[0], r2_2=r2[1], r2_3=r2[2], write_linear=1 - un[0], write_quadratic=un[0] - un[1], write_cubic=un[1] - un[2], write_rest=un[2], affine=aff, negative=float((g4 < 0).float().mean()))
            if not relu:
                corr = ((gc * uc).sum(0) / (gc.norm(dim=0) * uc.norm(dim=0)).clamp_min(1e-12)).abs(); cos = torch.nn.functional.cosine_similarity(b.gate.weight, b.up.weight, dim=1).abs(); q = lambda t: [float(v) for v in torch.quantile(t, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))]
                d.update(abs_corr_quantiles=q(corr), abs_cos_quantiles=q(cos), abs_corr_weighted_mean=float((corr * wv).sum()), weight_of_units_with_abs_corr_above_0_9=float(wv[corr > 0.9].sum()), weight_of_units_with_abs_corr_below_0_3=float(wv[corr < 0.3].sum()),
                         spread_of_gate_input_over_problems_median=float(g4.std(0).median()), abs_mean_gate_input_median=float(g4.mean(0).abs().median()))
            row["blocks"].append(d)
    for name, blocks, k in (("linearised", range(NL), 0), ("linearised_b0", (0,), 0), ("linearised_b1", (1,), 0), ("quadratised", range(NL), 1), ("quadratised_b0", (0,), 1), ("quadratised_b1", (1,), 1)):
        sh = {l: (lambda g, u, l=l, k=k: (feats(g, g if relu else u, k + 1, relu) * C[l][k]).sum(-1)) for l in blocks}
        with torch.no_grad(): lg2 = fwd(net, tok, dict(set_h=sh))[0]
        row[name] = dict(acc_train=acc4(lg2[is_tr], seq[is_tr]), acc_held=acc4(lg2[~is_tr], seq[~is_tr]), same_answer=float((lg2[:, 4].argmax(-1) == lg[:, 4].argmax(-1)).float().mean()))
    return row


res = dict(gaussian_table=table(), runs={}); seq, is_tr = all_problems(); half = torch.rand(len(seq), generator=torch.Generator().manual_seed(7)) < 0.5; json.dump(res, open(out, "w"), indent=1)
for d in dirs:
    cks = sorted(int(os.path.basename(f)[5:-3]) for f in glob.glob(f"{d}/ckpt_*.pt")); pick = [int(s) for s in os.environ["STEPS"].split(",")] if os.environ.get("STEPS") else sorted(set([0] + [c for c in cks if c in (2500, 5000)] + cks[-2:])); rows = []
    for st in pick:
        if st not in cks: continue
        r = state(load(f"{d}/ckpt_{st}.pt"), seq, is_tr, half); r["step"] = st; rows.append(r)
        print(os.path.basename(d), st, r["kind"], f"acc {r['acc_train']:.3f}/{r['acc_held']:.3f}", " | ".join(f"b{l}: R2 {b['r2_1']:.2f}/{b['r2_2']:.2f}/{b['r2_3']:.2f} write lin {b['write_linear']:.2f} quad {b['write_quadratic']:.2f} cub {b['write_cubic']:.2f} rest {b['write_rest']:.2f} affine {b['affine']:.2f} neg {b['negative']:.2f}" + (f" |corr| med {b['abs_corr_quantiles'][2]:.2f}" if "abs_corr_quantiles" in b else "") for l, b in enumerate(r["blocks"])),
              f"| held: linearised {r['linearised']['acc_held']:.3f} quadratised {r['quadratised']['acc_held']:.3f}", flush=True)
    res["runs"][os.path.basename(d)] = rows; json.dump(res, open(out, "w"), indent=1)
