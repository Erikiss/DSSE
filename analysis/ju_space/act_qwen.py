"""How quadratic are the SwiGLU units of Qwen3-1.7B on real text? (README 9.4) Every unit of an MLP writes z = silu(g) * u with
g = gate_proj(x) and u = up_proj(x), both linear in the MLP input x. Per unit: the best polynomial of degree 1, 2 and 3 in (g, u), by
least squares over the token positions of a text sample (ufo_common.texts: NFIT texts of the pool for the fits; everything
reported is measured on the 2 * NMAIN main texts, which the fits have not seen). Per block:
  r2_1, r2_2, r2_3   R^2 of the three fits on the main texts, averaged over the units with the weights |down_proj column|^2 * variance of z
  write shares       of the variance over the token positions of the MLP write (down_proj applied to the unit activations):
                     linear = 1 - U1, quadratic = U1 - U2, cubic = U2 - U3, rest = U3, where Ud is the share of that variance which
                     remains when every unit is replaced by its fit of degree d (a fit that does not carry over to the main
                     texts can make a share negative)
  negative           share of (position, unit) pairs with a negative gate input
  corr, cos          per unit |correlation| of g and u over the positions of the fit texts and |cosine| of its gate and up weight
                     vectors: 1 = the unit is a function of one linear combination of x (a square), 0 = a product of two unrelated
                     combinations (the quadratic form (w_g.x)(w_u.x) has its axes along the two bisectors of w_g and w_u with
                     eigenvalues |w_g||w_u|(cos +- 1)/2). For random vectors |cos| is about 0.8 / sqrt(2048) = 0.018
Surrogates (whole model, main texts): every unit of all blocks, or of a group of blocks, is replaced by its fit of degree 0
(a constant), 1 ("linearised": the MLP is a linear map) or 2 ("quadratised": the MLP is a polynomial of degree 2), at the valid
positions only (the positions the fits were made on; the first four tokens and the chat template keep the true MLP, because a
polynomial fitted on ordinary positions extrapolates wildly at the first token, where the activations are far larger).
Reported: mean KL divergence from the unchanged model's next-token distribution, cross-entropy of the actual next token, share of
positions with the same most likely token. The fitted coefficients are saved next to the output (<out>.coef.pt).
With COEF=<file written by an earlier run> only the surrogates are computed (from the stored coefficients), and the result goes to
<out.json> with the median and quantiles of the KL divergence over positions as well (the mean is dominated by a few positions).
Usage: python act_qwen.py <root> <out.json>      env: NT (threads, 1), NFIT (24), NMAIN (8), MAXT (64), COEF"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ufo_common import load_model, texts

root, out = sys.argv[1], sys.argv[2]; torch.set_num_threads(int(os.environ.get("NT", "1"))); NFIT = int(os.environ.get("NFIT", "24")); t0 = time.time()
tok, model = load_model(root); main, pool = texts(root, tok); fitset = pool[:NFIT]; layers = model.model.layers; NLAY = len(layers); H = layers[0].mlp.gate_proj.out_features; K = {0: 1, 1: 3, 2: 6, 3: 10}
print(f"model loaded ({time.time() - t0:.0f}s); fit texts {len(fitset)} with {sum(int(v.sum()) for _, _, v in fitset)} valid positions, main texts {len(main)} with {sum(int(v.sum()) for _, _, v in main)}", flush=True)
cap = {}
hooks = [l.mlp.register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(i, a[0].detach())) for i, l in enumerate(layers)]


def factors(i, x):
    m = layers[i].mlp; g = m.gate_proj(x); u = m.up_proj(x); return g, u, torch.nn.functional.silu(g) * u


def feats(gs, us, deg):
    f = [torch.ones_like(gs), gs, us, gs * gs, gs * us, us * us, gs ** 3, gs * gs * us, gs * us * us, us ** 3]; return torch.stack(f[:K[deg]], -1)


COEF = os.environ.get("COEF")
if COEF:
    z = torch.load(COEF, weights_only=True); C, mean, sd = z["C"], z["mean"], z["sd"]; res = dict(model="Qwen3-1.7B", coefficients=os.path.basename(COEF))
std = lambda i, g, u: ((g - mean[i, 0]) / sd[i, 0], (u - mean[i, 1]) / sd[i, 1])
if not COEF:
    # pass 0: scale of g and u per unit (first texts of the fit set; used only to standardise the regressors)
    s1 = torch.zeros(NLAY, 2, H, dtype=torch.float64); s2 = torch.zeros_like(s1); n0 = 0
    with torch.no_grad():
        for _, ids, valid in fitset[:4]:
            model(ids)
            for i in range(NLAY):
                g, u, _ = factors(i, cap[i][0, valid]); s1[i, 0] += g.sum(0).double(); s1[i, 1] += u.sum(0).double(); s2[i, 0] += (g.double() ** 2).sum(0); s2[i, 1] += (u.double() ** 2).sum(0)
            n0 += int(valid.sum())
    mean = (s1 / n0).float(); sd = ((s2 / n0 - (s1 / n0) ** 2).clamp_min(1e-12).sqrt()).float()
    # pass 1: normal equations per unit
    A = torch.zeros(NLAY, H, 10, 10, dtype=torch.float64); B = torch.zeros(NLAY, H, 10, dtype=torch.float64); n1 = 0
    with torch.no_grad():
        for k, (_, ids, valid) in enumerate(fitset):
            model(ids)
            for i in range(NLAY):
                g, u, z = factors(i, cap[i][0, valid]); F = feats(*std(i, g, u), 3).double().permute(1, 0, 2); A[i] += F.transpose(1, 2) @ F; B[i] += (F.transpose(1, 2) @ z.double().T.unsqueeze(-1)).squeeze(-1)
            n1 += int(valid.sum())
            if k % 6 == 0: print(f"fit text {k + 1}/{len(fitset)} ({time.time() - t0:.0f}s)", flush=True)
    ridge = lambda M: M + 1e-8 * torch.diag_embed(torch.diagonal(M, dim1=-2, dim2=-1).clamp_min(1e-12))
    C = {d: (torch.linalg.solve(ridge(A[:, :, :K[d], :K[d]]), B[:, :, :K[d]].unsqueeze(-1)).squeeze(-1).float() if d else (B[:, :, :1] / A[:, :, :1, 0]).float()) for d in (0, 1, 2, 3)}
    m = lambda a, b: A[:, :, a, b] / n1; corr_fit = ((m(1, 2) - m(0, 1) * m(0, 2)) / ((m(1, 1) - m(0, 1) ** 2).clamp_min(1e-12).sqrt() * (m(2, 2) - m(0, 2) ** 2).clamp_min(1e-12).sqrt())).abs().float(); del A, B
    # pass 2: fits read on the main texts
    D = layers[0].mlp.down_proj.out_features; sm = torch.zeros(NLAY, 4, D, dtype=torch.float64); sq = torch.zeros(NLAY, 4, dtype=torch.float64); sse = torch.zeros(NLAY, 3, H, dtype=torch.float64); zs1 = torch.zeros(NLAY, H, dtype=torch.float64); zs2 = torch.zeros_like(zs1); neg2 = torch.zeros(NLAY); n2 = 0
    with torch.no_grad():
        for _, ids, valid in main:
            model(ids)
            for i in range(NLAY):
                g, u, z = factors(i, cap[i][0, valid]); gs, us = std(i, g, u); F = feats(gs, us, 3); W = layers[i].mlp.down_proj.weight
                for k, d in enumerate((1, 2, 3)): r = z - (F[..., :K[d]] * C[d][i]).sum(-1); sse[i, k] += (r.double() ** 2).sum(0); w = (r @ W.T).double(); sm[i, k] += w.sum(0); sq[i, k] += (w ** 2).sum()
                w = (z @ W.T).double(); sm[i, 3] += w.sum(0); sq[i, 3] += (w ** 2).sum(); zs1[i] += z.sum(0).double(); zs2[i] += (z.double() ** 2).sum(0); neg2[i] += float((g < 0).float().sum())
            n2 += int(valid.sum())
    var = sq / n2 - ((sm / n2) ** 2).sum(-1); res = dict(model="Qwen3-1.7B", fit_positions=n1, main_positions=n2, blocks=[]); q = lambda t: [float(v) for v in torch.quantile(t, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))]
    for i in range(NLAY):
        mm = layers[i].mlp; cos = torch.nn.functional.cosine_similarity(mm.gate_proj.weight, mm.up_proj.weight, dim=1).abs(); zv = (zs2[i] / n2 - (zs1[i] / n2) ** 2).clamp_min(1e-30); wv = zv.float() * (mm.down_proj.weight ** 2).sum(0); wv = (wv / wv.sum()).double(); cr = corr_fit[i]
        r2 = [float(((1 - sse[i, k] / n2 / zv) * wv).sum().detach()) for k in range(3)]; un = [float(var[i, k] / var[i, 3]) for k in range(3)]
        res["blocks"].append(dict(block=i, r2_1=r2[0], r2_2=r2[1], r2_3=r2[2], write_linear=1 - un[0], write_quadratic=un[0] - un[1], write_cubic=un[1] - un[2], write_rest=un[2], negative=float(neg2[i] / (n2 * H)), abs_corr_quantiles=q(cr), abs_corr_weighted_mean=float((cr.double() * wv).sum()),
                                  weight_of_units_with_abs_corr_above_0_9=float(wv[cr > 0.9].sum()), weight_of_units_with_abs_corr_below_0_3=float(wv[cr < 0.3].sum()), abs_cos_quantiles=q(cos), abs_cos_mean=float(cos.mean()), abs_cos_of_random_vectors=float((2 / (3.14159265 * mm.gate_proj.in_features)) ** .5)))
        print(f"block {i:2d}: R2 {r2[0]:.3f}/{r2[1]:.3f}/{r2[2]:.3f} | write lin {1 - un[0]:.3f} quad {un[0] - un[1]:.3f} cub {un[1] - un[2]:.3f} rest {un[2]:.3f} | negative {res['blocks'][-1]['negative']:.2f} | |corr| median {q(cr)[2]:.2f} weighted {res['blocks'][-1]['abs_corr_weighted_mean']:.2f} | |cos| median {q(cos)[2]:.3f}", flush=True)
    json.dump(res, open(out, "w"), indent=1); torch.save(dict(C=C, mean=mean, sd=sd), out + ".coef.pt")
    del sm, sq, sse
for h in hooks: h.remove()
# surrogates: the whole model with units replaced by their fits (log-probabilities kept only at the valid positions, in float32)
orig = [l.mlp.forward for l in layers]


VALID = [None]                                                           # valid positions of the text being run


def patch(blocks, deg):
    for i, l in enumerate(layers):
        l.mlp.forward = orig[i] if i not in blocks else (lambda x, i=i, mm=l.mlp, o=orig[i]: torch.where(VALID[0][None, :, None], mm.down_proj((feats(*std(i, mm.gate_proj(x), mm.up_proj(x)), deg) * C[deg][i]).sum(-1)), o(x)))


def run(ref=None):
    """Without ref: list of (log-probabilities at the valid positions, next tokens, mask of positions that have a next token).
    With ref: KL from ref, cross-entropy and agreement of the most likely token, accumulated over the texts."""
    keep, kl, ce, same, n, nc, kls = [], 0.0, 0.0, 0.0, 0, 0, []
    with torch.no_grad():
        for j, (_, ids, valid) in enumerate(main):
            VALID[0] = valid; lp = torch.log_softmax(model(ids).logits[0].float(), -1); idx = valid.nonzero()[:, 0]; has = idx < ids.shape[1] - 1; nxt = ids[0, (idx[has] + 1)]; lpv = lp[idx]
            c = -lpv[has].gather(1, nxt[:, None])[:, 0]; ce += float(c.sum()); nc += int(has.sum())
            if ref is None: keep.append(lpv)
            else: r = ref[j]; kp = (r.exp() * (r - lpv)).sum(-1); kls.append(kp); kl += float(kp.sum()); same += float((r.argmax(-1) == lpv.argmax(-1)).float().sum()); n += len(idx)
    if ref is None: return keep, ce / nc
    kq = torch.quantile(torch.cat(kls).double(), torch.tensor([0.25, 0.5, 0.75, 0.9, 0.99], dtype=torch.float64)); return dict(kl=kl / n, kl_quantiles_25_50_75_90_99=[float(v) for v in kq], cross_entropy=ce / nc, same_top_token=same / n)


clean, ce0 = run(); groups = dict(all=range(NLAY), blocks_0_9=range(0, 10), blocks_10_18=range(10, 19), blocks_19_27=range(19, 28)); res["surrogates"] = {}; res["cross_entropy_unchanged"] = ce0
for gname, blocks in groups.items():
    for deg, dname in ((0, "constant"), (1, "linearised"), (2, "quadratised")):
        patch(set(blocks), deg); r = run(clean); res["surrogates"][f"{gname}:{dname}"] = r; print(f"{gname:13s} {dname:11s} KL {r['kl']:.3f} (median {r['kl_quantiles_25_50_75_90_99'][1]:.3f}) cross-entropy {r['cross_entropy']:.3f} (unchanged {ce0:.3f}) same top token {r['same_top_token']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        json.dump(res, open(out, "w"), indent=1)
patch(set(), 0); json.dump(res, open(out, "w"), indent=1)
