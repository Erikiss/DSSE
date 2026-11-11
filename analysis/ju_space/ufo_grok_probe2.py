"""What does a small transformer develop on the way to grokking? Second version of the probe, per checkpoint of a run of ufo_grok.py.
(1) Accuracy on the training and the held-out problems, weight norm, mean gain of the layer norms.
(2) Compressibility of the MLP activations at the '=' token in the sense of the UFO paper (z-scored activations, problems x
    hidden nodes that vary): number of varying nodes, participation ratio, share of variance explained by 2 / 8 principal
    components and (at selected checkpoints) by a 2-dim non-linear autoencoder (1500 full-batch steps, also read after 300).
    Everything is fitted on the training problems and reported for them and for the held-out problems (z-scored with the training
    statistics): a code that also fits unseen problems is what generalisation should add.
    Per hidden node also the share of its variance over all 9312 problems that is explained by the answer z alone, by the first
    operand x alone and by the second operand y alone (median over the varying nodes).
(3) Reaction of block 1 to a vector added to the state of the '=' token after block 0, separately for training and held-out
    problems. Size f * spread (spread = root-mean-square distance of the real states at that place from their mean).
    Direction classes of fixed dimension: top8 (random vector in the 8 main principal axes of the real states), bot64 (in the 64
    weakest axes, mean direction removed), cov (Gaussian with the covariance of real states), rand (isotropic).
    Read: share of problems with the same answer as without the vector, KL(clean || perturbed);
    q_att, q_mlp = share of the vector that the attention / MLP write of block 1 removes (as in ufo_damp2.py);
    the same read-outs for a block 1 that does not react (attention and MLP writes frozen at their clean values, only the two
    layer norms act): is the reaction of the block what protects the answer?
Usage: python ufo_grok_probe2.py <run_dir>     env: NT, AE_STEPS (comma list of checkpoints for the autoencoder)"""
import glob, json, math, os, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ufo_grok import Net, problems, NL, NH, D

run = sys.argv[1]; torch.set_num_threads(int(os.environ.get("NT", "1"))); tr, va = problems(); log = json.load(open(f"{run}/log.json")); LOG = {x["step"]: x for x in log["log"]}; PERM = "grok_perm" in run
if PERM: tr = tr.clone(); tr[:, 5] = tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(777)), 5]      # the labels this control was trained on
steps = sorted(int(os.path.basename(f)[5:-3]) for f in glob.glob(f"{run}/ckpt_*.pt")); unit = lambda v: v / (v.norm(dim=-1, keepdim=True) + 1e-9); SIZES = [0.1, 0.3, 1.0]; CLASSES = ["top8", "bot64", "cov", "rand"]
AES = [int(x) for x in os.environ["AE_STEPS"].split(",")] if os.environ.get("AE_STEPS") else sorted({steps[0], min(steps, key=lambda s: abs(s - 5000)), min(steps, key=lambda s: abs(s - 10000)), log["grok"] or steps[-1], steps[-1]})
allp = torch.cat(problems()); XYZ = [allp[:, 1], allp[:, 3], allp[:, 5]]                                            # x, y and the TRUE answer z of all problems (training problems first)


def eta2(A, lab):                                                           # share of the variance of every column of A explained by the group means of lab
    oh = torch.nn.functional.one_hot(lab - lab.min()).float(); cnt = oh.sum(0).clamp(min=1); gm = (oh.T @ A) / cnt[:, None]; return ((gm[lab - lab.min()] - A.mean(0)) ** 2).sum(0) / ((A - A.mean(0)) ** 2).sum(0).clamp(min=1e-12)


def bottleneck(A, B, ae, seed=0):
    """A, B: activations [n, c] of one layer on the training / held-out problems."""
    mu, sd = A.mean(0), A.std(0); keep = sd > 1e-6; c = int(keep.sum()); r = dict(nodes=c)
    if c < 4: return r
    Rt = ((A[:, keep] - mu[keep]) / sd[keep]).float(); Rv = ((B[:, keep] - mu[keep]) / sd[keep]).float(); _, s, Vh = torch.linalg.svd(Rt.double(), full_matrices=False); ev = s ** 2; V = Vh.T.float(); pc = lambda R, k: float(((R @ V[:, :k]) ** 2).sum() / (R ** 2).sum())
    C_ = torch.cat([A, B])[:, keep]; r.update(participation_ratio=float(ev.sum() ** 2 / (ev ** 2).sum()), pca2_fit=pc(Rt, 2), pca2_held=pc(Rv, 2), pca8_fit=pc(Rt, 8), pca8_held=pc(Rv, 8), by_z=float(eta2(C_, XYZ[2]).median()), by_x=float(eta2(C_, XYZ[0]).median()), by_y=float(eta2(C_, XYZ[1]).median()))
    if ae:
        torch.manual_seed(seed); net = nn.Sequential(nn.Linear(c, 128), nn.ReLU(), nn.Linear(128, 2), nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, c)); opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        with torch.enable_grad():
            for i in range(1, 1501):
                loss = ((net(Rt) - Rt) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
                if i in (300, 1500):
                    with torch.no_grad(): r[f"ae2_fit_{i}"] = float(1 - ((net(Rt) - Rt) ** 2).mean() / (Rt ** 2).mean()); r[f"ae2_held_{i}"] = float(1 - ((net(Rv) - Rv) ** 2).mean() / (Rv ** 2).mean())
    return r


def parts(b, x):                                                            # the pieces of one block (same computation as Block.forward)
    B, T, _ = x.shape; q, k, v = b.qkv(x).view(B, T, 3, NH, D // NH).permute(2, 0, 3, 1, 4); att = (q @ k.transpose(-1, -2)) / math.sqrt(D // NH)
    att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf")).softmax(-1); a = b.o((att @ v).transpose(1, 2).reshape(B, T, D)); x1 = b.ln1(x + a); m = b.fc2(torch.relu(b.fc1(x1))); return a, x1, m, b.ln2(x1 + m)


res = []
with torch.no_grad():
    for st in steps:
        net = Net(); net.load_state_dict(torch.load(f"{run}/ckpt_{st}.pt", weights_only=True)["net"]); net.eval(); b1 = net.blocks[1]; out = {s_: net(d_[:, :6]) for s_, d_ in (("train", tr), ("held", va))}
        acc = {s_: float((out[s_][0][:, 4].argmax(-1) == d_[:, 5]).float().mean()) for s_, d_ in (("train", tr), ("held", va))}
        row = dict(step=st, train_acc=acc["train"], val_acc=acc["held"], wnorm=LOG.get(st, {}).get("wnorm"), ln_gain={f"b{i}_{n}": float(getattr(b, n).weight.abs().mean()) for i, b in enumerate(net.blocks) for n in ("ln1", "ln2")},
                   comp={f"mlp{l}": bottleneck(out["train"][1][l][:, 4], out["held"][1][l][:, 4], st in AES) for l in range(NL)})
        X = out["train"][2][0][:, 4]; mu = X.mean(0); ev, U = torch.linalg.eigh(torch.cov((X - mu).T.double())); ev = ev.flip(0).clamp(min=0); U = U.flip(1).float(); spread = float(((X - mu) ** 2).sum(1).mean().sqrt()); mh = unit(mu); sq = U * ev.float().sqrt(); cum = torch.cumsum(ev, 0) / ev.sum().clamp(min=1e-30)
        row["site"] = dict(spread=spread, norm=float(X.norm(dim=1).mean()), k99=int((cum < .99).sum()) + 1, participation_ratio=float(ev.sum() ** 2 / (ev ** 2).sum().clamp(min=1e-30)))
        if spread > 1e-4 * row["site"]["norm"]:                                 # otherwise the state at '=' is the same for all problems and there is nothing to compare a vector with
            for s_, d_ in (("train", tr), ("held", va)):
                g = torch.Generator().manual_seed(1); lg0, _, st0 = out[s_]; lp0 = torch.log_softmax(lg0[:, 4], -1); top0 = lg0[:, 4].argmax(-1); n = len(d_); x0 = st0[0]; a, x1, m, x2 = parts(b1, x0)

                def direction(kind):
                    if kind == "rand": return unit(torch.randn(n, D, generator=g))
                    if kind == "cov": return unit(torch.randn(n, D, generator=g) @ sq.T)
                    if kind == "top8": return unit(torch.randn(n, 8, generator=g) @ U[:, :8].T)
                    v = torch.randn(n, 64, generator=g) @ U[:, -64:].T; return unit(v - (v @ mh)[:, None] * mh)
                for kind in CLASSES:
                    u = direction(kind)                                         # one direction per problem, reused for all sizes
                    for f in SIZES:
                        d = u * spread * f; xp = x0.clone(); xp[:, 4] += d; ap, x1p, mp, x2p = parts(b1, xp); lp = torch.log_softmax(net.head(x2p)[:, 4], -1); d1 = (x1p - x1)[:, 4]
                        x2f = b1.ln2(b1.ln1(xp + a) + m); lpf = torch.log_softmax(net.head(x2f)[:, 4], -1); dev = (x2p - x2)[:, 4].norm(dim=-1); devf = (x2f - x2)[:, 4].norm(dim=-1)
                        row[f"{s_}_{kind}_{f:g}"] = dict(same_top=float((lp.argmax(-1) == top0).float().mean()), kl=float((lp0.exp() * (lp0 - lp)).sum(-1).mean()), kl_frozen=float((lp0.exp() * (lp0 - lpf)).sum(-1).mean()), same_top_frozen=float((lpf.argmax(-1) == top0).float().mean()),
                                                         q_att=float((-((ap - a)[:, 4] * d).sum(-1) / (d ** 2).sum(-1)).mean()), q_mlp=float((-((mp - m)[:, 4] * d1).sum(-1) / (d1 ** 2).sum(-1).clamp(min=1e-20)).mean()), dev_over_frozen=float((dev / devf.clamp(min=1e-12)).mean()))
        res.append(row); c = row["comp"]; z = lambda kk, s_="train": " ".join(f"{cl}:{row[f'{s_}_{cl}_0.3'][kk]:.2f}" for cl in CLASSES) if f"{s_}_cov_0.3" in row else "-"
        print(f"step {st:6d} train {row['train_acc']:.2f} val {row['val_acc']:.3f} | nodes {c['mlp0'].get('nodes')}/{c['mlp1'].get('nodes')} PR {c['mlp0'].get('participation_ratio', 0):.0f}/{c['mlp1'].get('participation_ratio', 0):.0f} PCA8 fit/held {c['mlp0'].get('pca8_fit', 0):.2f}/{c['mlp0'].get('pca8_held', 0):.2f} {c['mlp1'].get('pca8_fit', 0):.2f}/{c['mlp1'].get('pca8_held', 0):.2f}"
              + f" by z {c['mlp0'].get('by_z', 0):.2f}/{c['mlp1'].get('by_z', 0):.2f}" + (f" AE2 fit/held {c['mlp0']['ae2_fit_1500']:.2f}/{c['mlp0']['ae2_held_1500']:.2f} {c['mlp1']['ae2_fit_1500']:.2f}/{c['mlp1']['ae2_held_1500']:.2f}" if "ae2_fit_1500" in c["mlp0"] and "ae2_fit_1500" in c["mlp1"] else "")
              + f" | k99 {row['site']['k99']} | train f=0.3 same {z('same_top')} | frozen {z('same_top_frozen')} | q_mlp {z('q_mlp')} | held same {z('same_top', 'held')}", flush=True)
        json.dump(dict(run=run, grok=log["grok"], rows=res), open(f"{run}/probe2.json", "w"), indent=1)
print("written", f"{run}/probe2.json")
