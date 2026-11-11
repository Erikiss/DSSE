"""How compressible are the MLP activations of Qwen3-1.7B in the sense of the UFO paper (Sheppard et al., arXiv 2611.11862)?
The paper's intercomponent UFO autoencodes, for one input, the z-scored activations of all hidden nodes of one MLP layer through
a 2-dim bottleneck; a network with 'unified' representations is one where this works well. Here the same measurement is made
on a normally trained language model, without any UFO: per block the matrix (prompts x 6144 neurons) of activations at the
last prompt token (a fixed token role, as the paper requires), z-scored per neuron with the statistics of the fit prompts.
Reported per block, fitted on two thirds of the prompts and evaluated on the held-out third:
  share of variance explained by 2 / 8 / 32 principal components, by a 2-dim non-linear autoencoder (6144-128-2-128-6144, ReLU,
  Adam 3e-3, 1500 full-batch steps, also read after 300; trained on the coordinates of the
  activations in the span of the fit prompts, which is an exact change of basis for them), and the participation ratio of the spectrum.
Controls: the same matrix with every neuron's column permuted across prompts (no joint structure left), and optionally the
activations of a second file (e.g. an untrained network) given as third argument.
Usage: python ufo_compress.py <out_dir> [activations.npy] [tag]     env: NT, BLOCKS (comma list of blocks for the autoencoder), ONLY (comma list: only these blocks at all), STEPS"""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn

out = sys.argv[1]; src = sys.argv[2] if len(sys.argv) > 2 else f"{out}/../ju_out/neu_act.npy"; TAG = sys.argv[3] if len(sys.argv) > 3 else ""; torch.set_num_threads(int(os.environ.get("NT", "2")))
A = np.load(src, mmap_mode="r"); n, L, NI = A.shape; rng = np.random.default_rng(0); perm = rng.permutation(n); fit, held = np.sort(perm[: 2 * n // 3]), np.sort(perm[2 * n // 3:])
AEB = [int(x) for x in os.environ.get("BLOCKS", "0,2,5,8,14,20,26,27").split(",")]; STEPS = int(os.environ.get("STEPS", "1500")); t0 = time.time(); res = []


def measure(X, ae=True, seed=0):
    Xf, Xh = torch.tensor(X[fit].astype(np.float32)), torch.tensor(X[held].astype(np.float32)); mu, sd = Xf.mean(0), Xf.std(0); keep = sd > 1e-6; Rf = (Xf[:, keep] - mu[keep]) / sd[keep]; Rh = (Xh[:, keep] - mu[keep]) / sd[keep]; c = Rf.shape[1]
    U, s, Vh = torch.linalg.svd(Rf.double(), full_matrices=False); ev = s ** 2; V = Vh.T.float(); pc = lambda R, k: float(((R @ V[:, :k]) ** 2).sum() / (R ** 2).sum())
    r = dict(nodes=int(c), participation_ratio=float(ev.sum() ** 2 / (ev ** 2).sum()), **{f"pca{k}_fit": pc(Rf, k) for k in (2, 8, 32)}, **{f"pca{k}_held": pc(Rh, k) for k in (2, 8, 32)})
    if ae:                                                                  # trained on the coordinates in the span of the fit prompts (exact change of basis for the fit data); error counted in the full space
        r_ = int((s > 1e-6 * s[0]).sum()); Vr = V[:, :r_]; Zf, Zh = Rf @ Vr, Rh @ Vr; tf, th = float((Rf ** 2).sum()), float((Rh ** 2).sum()); lost = th - float((Zh ** 2).sum())
        torch.manual_seed(seed); net = nn.Sequential(nn.Linear(r_, 128), nn.ReLU(), nn.Linear(128, 2), nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, r_)); opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        for i in range(1, STEPS + 1):
            loss = ((net(Zf) - Zf) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if i in (300, STEPS):
                with torch.no_grad(): r[f"ae2_fit_{i}"] = float(1 - ((net(Zf) - Zf) ** 2).sum() / tf); r[f"ae2_held_{i}"] = float(1 - (((net(Zh) - Zh) ** 2).sum() + lost) / th)
    return r


ONLY = [int(x) for x in os.environ["ONLY"].split(",")] if os.environ.get("ONLY") else list(range(L))          # restrict to some blocks
for m in ONLY:
    X = np.asarray(A[:, m]); r = dict(block=m, **measure(X, ae=m in AEB))
    if m in AEB[1::3]:                                                       # control: every neuron's column permuted across prompts
        Xs = X.copy(); [np.random.default_rng(100 + j).shuffle(Xs[:, j]) for j in range(0, NI)]; r["shuffled"] = measure(Xs, ae=True)
    res.append(r); print(f"block {m:2d}: nodes {r['nodes']} PR {r['participation_ratio']:.0f} | PCA2 {r['pca2_fit']:.3f}/{r['pca2_held']:.3f} PCA8 {r['pca8_fit']:.3f}/{r['pca8_held']:.3f} PCA32 {r['pca32_fit']:.3f}/{r['pca32_held']:.3f}"
                        + (f" | AE2 300: {r['ae2_fit_300']:.3f}/{r['ae2_held_300']:.3f}  , {STEPS}: {r[f'ae2_fit_{STEPS}']:.3f}/{r[f'ae2_held_{STEPS}']:.3f}" if "ae2_fit_300" in r else "")
                        + (f" | shuffled: PCA2 {r['shuffled']['pca2_fit']:.3f}/{r['shuffled']['pca2_held']:.3f} AE2 {r['shuffled'][f'ae2_fit_{STEPS}']:.3f}/{r['shuffled'][f'ae2_held_{STEPS}']:.3f}" if "shuffled" in r else "") + f"  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(dict(source=os.path.basename(src), prompts=int(n), fit=int(len(fit)), held=int(len(held)), rows=res), open(f"{out}/ufo_compress{TAG}.json", "w"), indent=1)
print("fit / held-out everywhere; written", f"{out}/ufo_compress{TAG}.json")
