"""Check on the bottleneck figures of ufo_grok_probe2.py: is 'fits seen problems, not unseen ones' a difference between seen
and unseen problems, or between problems inside and outside the fit?
Per checkpoint and MLP layer the z-scores and principal axes are fitted on a random HALF of the training problems and read on
(1) that half, (2) the other half of the training problems (seen in training, not in the fit), (3) the held-out problems.
Reported: share of variance in 2 and 8 axes for the three groups, once with plain z-scores and once with z-scores clipped
at +-5 (a node that fires on a handful of problems gets a tiny standard deviation and huge z-values outside the fit),
and how much of the total of group (3) the ten largest nodes hold.
Usage: python ufo_grok_split.py <run_dir>      writes <run_dir>/split.json"""
import glob, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ufo_grok import Net, problems, NL

run = sys.argv[1]; torch.set_num_threads(int(os.environ.get("NT", "2"))); tr, va = problems(); log = json.load(open(f"{run}/log.json"))
steps = sorted(int(os.path.basename(f)[5:-3]) for f in glob.glob(f"{run}/ckpt_*.pt")); perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(7)); A1, A2 = perm[: len(tr) // 2], perm[len(tr) // 2:]


def shares(F, G, H, clip):
    mu, sd = F.mean(0), F.std(0); keep = sd > 1e-6
    if int(keep.sum()) < 9: return None
    z = lambda X: ((X[:, keep] - mu[keep]) / sd[keep]).clamp(-clip, clip) if clip else (X[:, keep] - mu[keep]) / sd[keep]
    Rf, Rg, Rh = z(F).double(), z(G).double(), z(H).double(); V = torch.linalg.svd(Rf, full_matrices=False)[2].T; pc = lambda R, k: float(((R @ V[:, :k]) ** 2).sum() / (R ** 2).sum())
    top10 = float((Rh ** 2).sum(0).topk(10).values.sum() / (Rh ** 2).sum())
    return dict(nodes=int(keep.sum()), pca2=[pc(Rf, 2), pc(Rg, 2), pc(Rh, 2)], pca8=[pc(Rf, 8), pc(Rg, 8), pc(Rh, 8)], top10_nodes_share_of_unseen_total=top10)


res = []
with torch.no_grad():
    for st in steps:
        net = Net(); net.load_state_dict(torch.load(f"{run}/ckpt_{st}.pt", weights_only=True)["net"]); net.eval(); ht = net(tr[:, :6])[1]; hv = net(va[:, :6])[1]
        row = dict(step=st, **{f"mlp{l}": dict(plain=shares(ht[l][A1, 4], ht[l][A2, 4], hv[l][:, 4], 0), clipped=shares(ht[l][A1, 4], ht[l][A2, 4], hv[l][:, 4], 5.0)) for l in range(NL)}); res.append(row); p, c = row["mlp0"]["plain"], row["mlp0"]["clipped"]
        if p: print(f"step {st:6d} mlp0 8 axes, fit half / other training half / unseen: plain {p['pca8'][0]:.3f} / {p['pca8'][1]:.3f} / {p['pca8'][2]:.3f} (ten nodes hold {p['top10_nodes_share_of_unseen_total']:.2f} of the unseen total) | clipped {c['pca8'][0]:.3f} / {c['pca8'][1]:.3f} / {c['pca8'][2]:.3f}"
                       + (f" | mlp1 plain {row['mlp1']['plain']['pca8'][0]:.3f} / {row['mlp1']['plain']['pca8'][1]:.3f} / {row['mlp1']['plain']['pca8'][2]:.3f}, 2 axes unseen {row['mlp1']['plain']['pca2'][2]:.3f}" if row["mlp1"]["plain"] else ""), flush=True)
        else: print(f"step {st:6d} mlp0: fewer than 9 varying nodes", flush=True)
json.dump(dict(run=run, grok=log["grok"], groups=["fit half of the training problems", "other half of the training problems", "held-out problems"], rows=res), open(f"{run}/split.json", "w"), indent=1); print("written", f"{run}/split.json")
