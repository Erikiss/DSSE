"""Which MLP neurons of Qwen3-1.7B erase a vector that is added to the residual stream? Exact first-order answer, one clean pass.

For a small vector delta added to the input of the MLP of block m, neuron n changes its activation by g_n . delta and thereby
writes (g_n . delta) w_n, where w_n is its output weight (column of down_proj) and g_n the gradient of its activation
a_n = silu(u_n) v_n (u = gate_proj(x), v = up_proj(x), x = RMSNorm(h)). Its contribution to the erasure of delta is
    c_n = -(g_n . delta)(w_n . delta) / |delta|^2          (the c_n of a block add up to the erasure by the MLP, km)
and its expectation over a class of directions with E[delta delta^T] / E|delta|^2 = M is  E[c_n] = -g_n^T M w_n:
    rand  M = I / D                 irr  M = projector on the 'irregular' subspace / its dimension (mean direction removed)
    cov   M = covariance of real states / its trace
    pc1_8, pc9_64, pc65_256, pc257_1024, pc1025_k   M = projector on a band of principal axes of real states / its size
          (axes ordered by variance; the last band ends where the 'irregular' subspace begins)
The classes are those of ufo_damp2.py (statistics of block m - 1, i.e. of the state the block receives; the write of the
block's own attention in between is ignored). Controls: the same sum with the output weights assigned to the wrong neurons
(permuted), which removes any alignment of what a neuron reads with what it writes; and RANDINIT=1 (untrained network).
Also from the weights alone: cosine between the input weights (gate, up) and the output weight of every neuron.
The erasure is also split by path (through the gate or through the up projection), by the sign of the neuron's gate input at
that position (SiLU is slightly negative for negative input, so an 'inactive' neuron still passes its up input with a small
negative factor) and by activation (the strongest quarter of the neurons at that position against the rest).
Usage: python ufo_lin.py <scratch_root> <out_dir>    env: NT, NTEXT, RANDINIT=1"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ufo_common import load_model, texts, Pool, unit

root, out = sys.argv[1], sys.argv[2]; RANDINIT = os.environ.get("RANDINIT") == "1"; TAG = "_randinit" if RANDINIT else ""; torch.set_grad_enabled(False); torch.set_num_threads(int(os.environ.get("NT", "4")))
tok, model = load_model(root, RANDINIT); LAYERS = model.model.layers; L = len(LAYERS); D = model.config.hidden_size; NI = model.config.intermediate_size; main, _ = texts(root, tok); main = main[:int(os.environ["NTEXT"])] if "NTEXT" in os.environ else main
P = Pool(f"{out}/ufo_pool{TAG}.pt"); BANDS = {"pc1_8": (0, 8), "pc9_64": (8, 64), "pc65_256": (64, 256), "pc257_1024": (256, 1024), "pc1025_k": (1024, None)}; CLS = ["rand", "irr", "cov"] + list(BANDS); t0 = time.time(); cap = {}
for m in range(L):
    LAYERS[m].mlp.gate_proj.register_forward_hook(lambda mod, i, o, m=m: cap.__setitem__(("u", m), o[0])); LAYERS[m].mlp.up_proj.register_forward_hook(lambda mod, i, o, m=m: cap.__setitem__(("v", m), o[0]))
    LAYERS[m].post_attention_layernorm.register_forward_pre_hook(lambda mod, i, m=m: cap.__setitem__(("h", m), i[0][0]))
# per block: alpha_n = (gamma * W_gate[n]) M w_n, beta_n = (gamma * W_up[n]) M w_n for every class, and the matrix that maps the state direction to M w_n
pre = {}; wstat = {}; gperm = torch.randperm(NI, generator=torch.Generator().manual_seed(0))
for m in range(L):
    mlp = LAYERS[m].mlp; gam = LAYERS[m].post_attention_layernorm.weight; Wg = mlp.gate_proj.weight * gam; Wu = mlp.up_proj.weight * gam; Wd = mlp.down_proj.weight                     # Wg, Wu [NI, D], Wd [D, NI]
    cosg = (unit(mlp.gate_proj.weight) * unit(Wd.T)).sum(1); cosu = (unit(mlp.up_proj.weight) * unit(Wd.T)).sum(1)
    wstat[m] = dict(cos_gate_down_mean=float(cosg.mean()), cos_up_down_mean=float(cosu.mean()), cos_gate_down_q=[float(x) for x in torch.quantile(cosg, torch.tensor([.01, .1, .5, .9, .99]))], cos_up_down_q=[float(x) for x in torch.quantile(cosu, torch.tensor([.01, .1, .5, .9, .99]))],
                    share_up_down_below_m01=float((cosu < -0.1).float().mean()), share_up_down_above_p01=float((cosu > 0.1).float().mean()), chance_sd=float(1 / np.sqrt(D)))
    if m == 0: continue                                                                    # block 0 receives the embeddings: no pool statistics for that state
    b = P.b[m - 1]; k = int(b["k_irr"]); U = b["U"]; mh = unit(b["mu"]); Q = U[:, k:] - torch.outer(mh, mh @ U[:, k:]); Q = torch.linalg.qr(Q)[0]; kc = Q.shape[1]; var = b["sd"] ** 2
    Fs = {"rand": None, "irr": Q / np.sqrt(kc), "cov": U * (b["sd"] / var.sum().sqrt())}                                              # M = F F^T for every class
    for nm_, (a_, b_) in BANDS.items(): b2 = k if b_ is None else b_; Fs[nm_] = U[:, a_:b2] / np.sqrt(b2 - a_)                     # bands of principal axes of real states, strongest first
    WdT = Wd.T.contiguous(); pre[m] = {}
    for c in CLS:
        F = Fs[c]; GF, UF, DF = (Wg / np.sqrt(D), Wu / np.sqrt(D), WdT / np.sqrt(D)) if F is None else (Wg @ F, Wu @ F, WdT @ F)
        pre[m][c] = dict(alpha=(GF * DF).sum(1), beta=(UF * DF).sum(1), alpha_p=(GF * DF[gperm]).sum(1), beta_p=(UF * DF[gperm]).sum(1), F=F)
    if m % 7 == 0: print(f"  prepared block {m} ({time.time() - t0:.0f}s)", flush=True)
silu = torch.nn.functional.silu; dsilu = lambda u: torch.sigmoid(u) * (1 + u * (1 - torch.sigmoid(u)))
SPLIT = np.full((len(main), L, len(CLS), 7), np.nan)        # erasure by path (gate, up), by sign of the gate input (negative, positive), by activation (strongest quarter, rest); share of neurons with negative gate input
UB = torch.tensor([-3.0, -1.0, 0.0, 1.0]); BINS = np.full((len(main), L, len(CLS), len(UB) + 2), np.nan); FRAC = np.full((len(main), L, len(UB) + 2), np.nan)   # erasure by gate input (5 bins) and of the stronger half; shares of neurons per bin, share of the stronger half with negative gate input
CELL = np.full((len(main), L, len(CLS), 4), np.nan); SILU = np.full((len(main), L, 2), np.nan)
E = np.zeros((L, len(CLS), NI)); KM = np.full((len(main), L, len(CLS)), np.nan); KMP = np.full((len(main), L, len(CLS)), np.nan); GROSS = np.full((len(main), L, len(CLS)), np.nan); npos = 0
for k_, (src, ids, valid) in enumerate(main):
    model.model(input_ids=ids); nv = int(valid.sum()); npos += nv
    for m in range(1, L):
        u = cap[("u", m)][valid]; v = cap[("v", m)][valid]; h = cap[("h", m)][valid]; rms = h.norm(dim=-1, keepdim=True) / np.sqrt(D); hh = unit(h); a1 = dsilu(u) * v; a2 = silu(u)          # [n, NI]
        for j, c in enumerate(CLS):
            q = pre[m][c]; Wd = LAYERS[m].mlp.down_proj.weight; Mw = (hh @ Wd) / D if q["F"] is None else ((hh @ q["F"]) @ q["F"].T) @ Wd                # hh^T M w_n  [n, NI]
            cg = -(a1 * q["alpha"] - a1 * u / np.sqrt(D) * Mw) / rms; cu = -(a2 * q["beta"] - a2 * v / np.sqrt(D) * Mw) / rms; cn = cg + cu; cp = -(a1 * q["alpha_p"] + a2 * q["beta_p"]) / rms     # gate path, up path
            neg = u < 0; act = (a2 * v).abs(); top = act >= torch.quantile(act, 0.75, dim=1, keepdim=True); SPLIT[k_, m, j] = [float(cg.sum(1).mean()), float(cu.sum(1).mean()), float((cn * neg).sum(1).mean()), float((cn * ~neg).sum(1).mean()), float((cn * top).sum(1).mean()), float((cn * ~top).sum(1).mean()), float(neg.float().mean())]
            CELL[k_, m, j] = [float((cg * neg).sum(1).mean()), float((cg * ~neg).sum(1).mean()), float((cu * neg).sum(1).mean()), float((cu * ~neg).sum(1).mean())]      # gate path / up path x negative / positive gate input
            if j == 0: SILU[k_, m] = [float(a2[neg].mean()), float(a2[~neg].mean())]
            if j == 0: half = act >= torch.quantile(act, 0.5, dim=1, keepdim=True); ub = torch.bucketize(u, UB); FRAC[k_, m] = [float((ub == b_).float().mean()) for b_ in range(len(UB) + 1)] + [float((neg & half).float().mean() / half.float().mean())]
            BINS[k_, m, j] = [float((cn * (ub == b_)).sum(1).mean()) for b_ in range(len(UB) + 1)] + [float((cn * half).sum(1).mean())]
            E[m, j] += cn.sum(0).double().numpy(); KM[k_, m, j] = float(cn.sum(1).mean()); KMP[k_, m, j] = float(cp.sum(1).mean()); GROSS[k_, m, j] = float(cn.abs().sum(1).mean())
    print(f"  text {k_ + 1}/{len(main)} ({time.time() - t0:.0f}s)", flush=True)
E /= npos; conc = {}
for m in range(1, L):
    conc[m] = {}
    for j, c in enumerate(CLS):
        e = E[m, j]; o = np.argsort(-e); cs = np.cumsum(e[o]); tot = e.sum(); pos = e[e > 0].sum(); neg = e[e < 0].sum()
        conc[m][c] = dict(net=float(tot), erasing=float(pos), amplifying=float(neg), n_half_of_erasing=int(np.searchsorted(np.cumsum(np.sort(e[e > 0])[::-1]), pos / 2) + 1), top100_share_of_erasing=float(np.sort(e)[::-1][:100].sum() / pos), n_erasing=int((e > 0).sum()))
    conc[m]["corr_irr_cov"] = float(np.corrcoef(E[m, 1], E[m, 2])[0, 1]); conc[m]["corr_irr_rand"] = float(np.corrcoef(E[m, 1], E[m, 0])[0, 1])
    a, b_ = set(np.argsort(-E[m, 1])[:100]), set(np.argsort(-E[m, 2])[:100]); conc[m]["top100_overlap_irr_cov"] = len(a & b_)
np.save(f"{out}/ufo_lin_E{TAG}.npy", E.astype(np.float32)); np.savez(f"{out}/ufo_lin{TAG}.npz", km=KM, km_perm=KMP, gross=GROSS, split=SPLIT, bins=BINS, frac=FRAC, cell=CELL, silu=SILU)
json.dump(dict(classes=CLS, wstat={str(m): v for m, v in wstat.items()}, conc={str(m): v for m, v in conc.items()}, n_positions=npos, randinit=RANDINIT), open(f"{out}/ufo_lin{TAG}.json", "w"), indent=1)
print("block:  km_lin rand / irr / cov | permuted rand / irr | gross irr | cos(up,down) mean | neurons for half of the erasure (irr) | corr irr~cov")
for m in range(1, L):
    print(f"{m:3d}:  {np.nanmean(KM[:, m, 0]):+.3f} / {np.nanmean(KM[:, m, 1]):+.3f} / {np.nanmean(KM[:, m, 2]):+.3f} | {np.nanmean(KMP[:, m, 0]):+.3f} / {np.nanmean(KMP[:, m, 1]):+.3f} | {np.nanmean(GROSS[:, m, 1]):.2f} | {wstat[m]['cos_up_down_mean']:+.4f} | {conc[m]['irr']['n_half_of_erasing']:5d} | {conc[m]['corr_irr_cov']:+.2f}")
print("block:  erasure of irr by path: gate / up | by gate input: negative / positive | by activation: strongest quarter / rest | share of neurons with negative gate input || the same for cov: negative / positive, strongest quarter / rest")
for m in range(1, L):
    s = np.nanmean(SPLIT[:, m], 0); print(f"{m:3d}:  {s[1, 0]:+.3f} / {s[1, 1]:+.3f} | {s[1, 2]:+.3f} / {s[1, 3]:+.3f} | {s[1, 4]:+.3f} / {s[1, 5]:+.3f} | {s[1, 6]:.2f} || {s[2, 2]:+.3f} / {s[2, 3]:+.3f}, {s[2, 4]:+.3f} / {s[2, 5]:+.3f}")
print("block:  erasure by band of principal axes of real states: axes 1-8 / 9-64 / 65-256 / 257-1024 / 1025 up to the limit of the data subspace / outside (irr)")
for m in range(1, L): print(f"{m:3d}:  " + " / ".join(f"{np.nanmean(KM[:, m, CLS.index(c)]):+.3f}" for c in list(BANDS) + ["irr"]))
print("block:  erasure of irr: up path with negative gate input / up path with positive / gate path negative / gate path positive | mean gate value silu(u) of neurons with negative / positive gate input")
for m in range(1, L):
    c_ = np.nanmean(CELL[:, m, 1], 0); s_ = np.nanmean(SILU[:, m], 0); print(f"{m:3d}:  {c_[2]:+.3f} / {c_[3]:+.3f} / {c_[0]:+.3f} / {c_[1]:+.3f} | {s_[0]:+.3f} / {s_[1]:+.3f}")
print("block:  erasure of irr by gate input u: u<=-3 / -3..-1 / -1..0 / 0..1 / >1 | share of neurons in these bins | erasure by the stronger half of the activations | share of the stronger half with negative gate input")
for m in range(1, L):
    b = np.nanmean(BINS[:, m, 1], 0); f = np.nanmean(FRAC[:, m], 0); print(f"{m:3d}:  " + " / ".join(f"{x:+.3f}" for x in b[:5]) + " | " + " / ".join(f"{x:.2f}" for x in f[:5]) + f" | {b[5]:+.3f} | {f[5]:.2f}")
print(f"done ({time.time() - t0:.0f}s)")
