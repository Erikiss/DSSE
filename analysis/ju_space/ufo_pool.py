"""Statistics of real residual-stream states of Qwen3-1.7B after every block, for the perturbation experiments.
Pool texts are disjoint from the texts that are perturbed. Two thirds of the pool texts (fit part) give the mean and the
principal axes; the remaining third (held-out part) decides how many axes are needed: k_irr is the smallest number of axes
outside of which at most 2 % of the variance of held-out states lies (cap 1900). Directions outside these axes are the
'irregular' class. States with a norm above ten times the median and template tokens are left out.
Usage: python ufo_pool.py <scratch_root> <out_dir>   (RANDINIT=1: randomly initialised model, file ufo_pool_randinit.pt)"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ufo_common import load_model, texts, FIRST

root, out = sys.argv[1], sys.argv[2]; RANDINIT = os.environ.get("RANDINIT") == "1"; torch.set_grad_enabled(False); torch.set_num_threads(int(os.environ.get("NT", "3")))
tok, model = load_model(root, RANDINIT); LAYERS = model.model.layers; L = len(LAYERS); D = model.config.hidden_size; main, pool = texts(root, tok); NS = 1536; t0 = time.time()
cap = {}; [LAYERS[m].register_forward_hook(lambda mod, i, o, m=m: cap.__setitem__(m, (o[0] if isinstance(o, tuple) else o)[0])) for m in range(L)]
# pass 1 (12 texts): median norm per block, for the outlier filter
nr = {m: [] for m in range(L)}
for src, ids, valid in pool[::max(1, len(pool) // 12)]:
    model.model(input_ids=ids)
    for m in range(L): nr[m].append(cap[m][valid].norm(dim=1))
MED = {m: float(torch.cat(nr[m]).median()) for m in range(L)}
# pass 2: sums for mean and covariance (fit / held-out), a sample of states, counts by source
part = [("held" if i % 3 == 2 else "fit") for i in range(len(pool))]; acc = {m: {p: [0, torch.zeros(D, dtype=torch.float64), torch.zeros(D, D, dtype=torch.float64)] for p in ("fit", "held")} for m in range(L)}
samp = {m: [] for m in range(L)}; cnt = {m: dict(wiki=0, chat=0, dropped=0) for m in range(L)}; rng = np.random.default_rng(0); per_text = max(4, int(np.ceil(NS / sum(p == "fit" for p in part))) + 2)
for i, (src, ids, valid) in enumerate(pool):
    model.model(input_ids=ids); pos = rng.permutation(int(valid.sum()))[:per_text]
    for m in range(L):
        X = cap[m][valid]; ok = X.norm(dim=1) < 10 * MED[m]; cnt[m]["dropped"] += int((~ok).sum()); cnt[m][src] += int(ok.sum()); Xd = X[ok].double(); a = acc[m][part[i]]; a[0] += len(Xd); a[1] += Xd.sum(0); a[2] += Xd.T @ Xd
        if part[i] == "fit": samp[m].append(X[pos][ok[pos]].clone())
    if i % 40 == 0: print(f"  pool text {i}/{len(pool)} ({time.time() - t0:.0f}s)", flush=True)
blocks = {}; stats = {}
for m in range(L):
    n, s1, s2 = acc[m]["fit"]; mu = s1 / n; C = s2 / n - torch.outer(mu, mu); ev, U = torch.linalg.eigh(C); ev = ev.flip(0).clamp(min=0); U = U.flip(1)
    nh, h1, h2 = acc[m]["held"]; Ch = h2 / nh - torch.outer(h1 / nh, mu) - torch.outer(mu, h1 / nh) + torch.outer(mu, mu)          # second moment of held-out states around the fit mean
    hv = torch.einsum("ij,ik,kj->j", U, Ch, U).clamp(min=0); out_share = 1 - torch.cumsum(hv, 0) / hv.sum(); fit_out = 1 - torch.cumsum(ev, 0) / ev.sum()
    k = int((out_share > 0.02).sum()) + 1; k = min(k, 1900); S = torch.cat(samp[m])[:NS]; ii = torch.randperm(len(S), generator=torch.Generator().manual_seed(0)); spread = float((S - S[ii]).norm(dim=1).median())
    blocks[m] = dict(mu=mu.float(), U=U.float().contiguous(), sd=ev.sqrt().float(), k_irr=torch.tensor(k), sample=S.contiguous(), med_norm=torch.tensor(MED[m]))
    stats[m] = dict(n_fit=int(n), n_held=int(nh), **cnt[m], med_norm=MED[m], mean_norm_share=float(mu.norm() / MED[m]), spread_over_norm=spread / MED[m], k_irr=k, heldout_outside_at_k=float(out_share[k - 1]), fit_outside_at_k=float(fit_out[k - 1]),
                    heldout_outside={kk: float(out_share[kk - 1]) for kk in (8, 64, 256, 512, 1024, 1536, 1800)}, k90_fit=int((fit_out > .10).sum()) + 1, k99_fit=int((fit_out > .01).sum()) + 1, participation_ratio=float(ev.sum() ** 2 / (ev ** 2).sum()))
    if m in (0, 1, 2, 4, 8, 14, 21, 27): print(f"block {m}: " + ", ".join(f"{a}={(round(b, 4) if isinstance(b, float) else b)}" for a, b in stats[m].items()), flush=True)
torch.save(dict(blocks=blocks, stats=stats), f"{out}/ufo_pool{'_randinit' if RANDINIT else ''}.pt"); print(f"pool: {len(pool)} texts, written ({time.time() - t0:.0f}s)")
