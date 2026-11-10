"""Groupings of DSSE behaviors in U-space vs J-space (Qwen3-1.7B): controls, stability, lens readouts.
Usage: python analyze2.py <scratch_root> <out_dir> <eval|train> [block]"""
import glob, json, re, sys
import numpy as np, pandas as pd, torch, yaml
from safetensors import safe_open
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from transformers import AutoTokenizer
root, out, mode = sys.argv[1], sys.argv[2], sys.argv[3]; BLOCK = int(sys.argv[4]) if len(sys.argv) > 4 else 18
rng = np.random.default_rng(0)
tok = AutoTokenizer.from_pretrained(f"{root}/models/qwen3-1.7b")
Jm = torch.load(f"{root}/jlens/Qwen3-1.7B_jacobian_lens.pt", map_location="cpu", weights_only=True)["J"][BLOCK].float().numpy().astype(np.float64)
WU = g = None
for f in glob.glob(f"{root}/models/qwen3-1.7b/*.safetensors"):
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            if k.endswith("lm_head.weight") or (WU is None and k.endswith("embed_tokens.weight")): WU = sf.get_tensor(k).float().numpy()
            if k.endswith("model.norm.weight") or k == "norm.weight": g = sf.get_tensor(k).float().numpy()
def unit(a): return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
def lowdin(B):
    w, V = np.linalg.eigh(B.T @ B); U = B @ (V @ np.diag(np.maximum(w, 1e-12) ** -0.5) @ V.T); return U * np.sign((U * B).sum(0))[None, :]
def pull(ids): return unit((WU[ids].astype(np.float64) * g) @ Jm)                 # lens-transported token directions at this block
def tid(w):
    for s in (" " + w, w):
        i = tok.encode(s, add_special_tokens=False)
        if len(i) == 1: return i[0]
terms = yaml.safe_load(open(f"{root}/uspace_repo/data_uspace_terms.yaml")); cats = list(terms)
ids = {c: {s: [i for i in map(tid, terms[c][s]) if i is not None] for s in ("anchors", "poles")} for c in cats}
Braw = np.stack([unit(pull(ids[c]["anchors"]).mean(0) - pull(ids[c]["poles"]).mean(0)) for c in cats], 1); U = lowdin(Braw)
pool = np.array([i for i in range(2000, 100000) if tok.decode([i]).strip().isalpha() and len(tok.decode([i]).strip()) > 3])
def rand_word_basis():
    return lowdin(np.stack([unit(pull(rng.choice(pool, len(ids[c]["anchors"]), replace=False)).mean(0) - pull(rng.choice(pool, len(ids[c]["poles"]), replace=False)).mean(0)) for c in cats], 1))
def rand_orth(d=4): return np.linalg.qr(rng.standard_normal((Jm.shape[0], d)))[0]

if mode == "eval":
    items = json.load(open(f"{out}/eval_items.json")); H = unit(np.load(f"{out}/eval_states_block{BLOCK}.npy").astype(np.float64)); lab = np.array([a for a, _, _ in items]); words = None; texts = None
else:
    items = json.load(open(f"{out}/train_items.json")); Hr = np.load(f"{out}/train_states_block{BLOCK}.npy").astype(np.float64); df = pd.DataFrame(items)
    key = df.groupby("dataset", sort=False).ngroup().values
    H = unit(np.stack([Hr[key == k].mean(0) for k in range(key.max() + 1)])); lab = df.groupby("dataset", sort=False).behavior.first().values
    words = df.groupby("dataset", sort=False).words.mean().values; texts = df.groupby("behavior").response.apply(lambda s: " ".join(s).lower()).to_dict()
beh = sorted(set(lab)); nb = len(beh); n = len(lab); Hc = H - H.mean(0)
print(f"[{mode}, block {BLOCK}] {n} points, {nb} behaviors, {np.bincount([beh.index(b) for b in lab]).min()}-{np.bincount([beh.index(b) for b in lab]).max()} per behavior")

def eta2(x):                                   # share of a coordinate's variance that lies between behaviors
    m = x.mean(); return sum((lab == b).sum() * (x[lab == b].mean() - m) ** 2 for b in beh) / ((x - m) ** 2).sum()
def loo_nc(X):
    X = (X - X.mean(0)) / (X.std(0) + 1e-12); ci = np.array([beh.index(b) for b in lab]); S = np.stack([X[ci == c].sum(0) for c in range(nb)]); N = np.bincount(ci).astype(float); ok = 0
    for i in range(n):
        Sx, Nx = S.copy(), N.copy(); Sx[ci[i]] -= X[i]; Nx[ci[i]] -= 1
        ok += int(np.argmin(((Sx / np.maximum(Nx, 1)[:, None] - X[i]) ** 2).sum(1)) == ci[i])
    return ok / n
def structure(P, Hx=None, labx=None):
    # behavior centroids inside a 4-dim subspace P (z-scored coordinates) and in the J-transported state with P removed
    Hx = H if Hx is None else Hx; labx = lab if labx is None else labx
    A = Hx @ P; Az = (A - A.mean(0)) / (A.std(0) + 1e-12); CU = np.stack([Az[labx == b].mean(0) for b in beh])
    JP = unit((Hx - A @ P.T) @ Jm.T); JP = JP - JP.mean(0); CJ = np.stack([JP[labx == b].mean(0) for b in beh])
    return CU, CJ, squareform(pdist(CU, "euclidean")), squareform(pdist(CJ, "cosine"))
iu = np.triu_indices(nb, 1)

# 1) how much category structure sits in the U-space, against controls
acc_U = loo_nc(H @ U); R1 = [rand_orth() for _ in range(200)]; R2 = [rand_word_basis() for _ in range(200)]
acc_r, acc_w = np.array([loo_nc(H @ P) for P in R1]), np.array([loo_nc(H @ P) for P in R2])
_, _, Vt = np.linalg.svd(Hc, full_matrices=False); Hp = H - (H @ U) @ U.T; Hpc = Hp - Hp.mean(0); _, _, Vp = np.linalg.svd(Hpc, full_matrices=False)
e_U = np.array([eta2(H @ U[:, k]) for k in range(4)]); e_raw = np.array([eta2(H @ Braw[:, k]) for k in range(4)])
e_r = np.array([eta2(H @ P[:, 0]) for P in R1]); e_w = np.array([eta2(H @ P[:, k]) for P in R2 for k in range(4)])
print(f"1) leave-one-out nearest-centroid accuracy (chance {1 / nb:.3f}): U-space {acc_U:.3f} | random 4 dims {acc_r.mean():.3f} [5-95%: {np.percentile(acc_r, 5):.3f}-{np.percentile(acc_r, 95):.3f}] | "
      f"random-word spaces {acc_w.mean():.3f} [{np.percentile(acc_w, 5):.3f}-{np.percentile(acc_w, 95):.3f}] | best 4 dims {loo_nc(Hc @ Vt[:4].T):.3f} | state 50 PCs {loo_nc(Hc @ Vt[:50].T):.3f} | state without U, 50 PCs {loo_nc(Hpc @ Vp[:50].T):.3f}")
print(f"   percentile of U among random-word spaces: {100 * (acc_w < acc_U).mean():.0f}")
print("   between-behavior share of variance (eta^2) per direction: raw category directions " + ", ".join(f"{c[:12]} {v:.2f}" for c, v in zip(cats, e_raw))
      + f" | random direction {e_r.mean():.2f} [5-95%: {np.percentile(e_r, 5):.2f}-{np.percentile(e_r, 95):.2f}] | random-word direction {e_w.mean():.2f} [{np.percentile(e_w, 5):.2f}-{np.percentile(e_w, 95):.2f}]")

# 2) do the groupings inside U agree with those in the rest of J-space more than for a random slice?
CU, CJ, DU, DJ = structure(U); r_obs = spearmanr(DU[iu], DJ[iu])[0]
r_r = np.array([(lambda s: spearmanr(s[2][iu], s[3][iu])[0])(structure(P)) for P in R1[:100]]); r_w = np.array([(lambda s: spearmanr(s[2][iu], s[3][iu])[0])(structure(P)) for P in R2[:100]])
perm = np.array([spearmanr(DU[p][:, p][iu], DJ[iu])[0] for p in (rng.permutation(nb) for _ in range(5000))])
print(f"2) agreement of distance structures (Spearman over {len(iu[0])} behavior pairs): U vs rest of J {r_obs:+.2f} (label-permutation p {np.mean(perm >= r_obs):.4f}) | random 4 dims vs rest {r_r.mean():+.2f} [{np.percentile(r_r, 5):+.2f},{np.percentile(r_r, 95):+.2f}] | "
      f"random-word spaces vs rest {r_w.mean():+.2f} [{np.percentile(r_w, 5):+.2f},{np.percentile(r_w, 95):+.2f}]")

# 3) stable groupings: bootstrap over points within behavior, average-linkage clusters cut at k=5 in both spaces
K = 5; fU = np.zeros((nb, nb)); fJ = np.zeros((nb, nb)); Bn = 300; idx = {b: np.where(lab == b)[0] for b in beh}
for _ in range(Bn):
    s = np.concatenate([rng.choice(idx[b], len(idx[b])) for b in beh]); cu_, cj_, _, _ = structure(U, H[s], lab[s])
    a = fcluster(linkage(cu_, "average", metric="euclidean"), K, "maxclust"); b_ = fcluster(linkage(cj_, "average", metric="cosine"), K, "maxclust")
    fU += a[:, None] == a[None, :]; fJ += b_[:, None] == b_[None, :]
fU /= Bn; fJ /= Bn
pairs = [(beh[i], beh[j], fU[i, j], fJ[i, j]) for i in range(nb) for j in range(i + 1, nb)]
both = sorted([p for p in pairs if p[2] >= .7 and p[3] >= .7], key=lambda p: -(p[2] + p[3])); onlyU = sorted([p for p in pairs if p[2] >= .7 and p[3] <= .3], key=lambda p: -p[2]); onlyJ = sorted([p for p in pairs if p[3] >= .7 and p[2] <= .3], key=lambda p: -p[3])
fmt = lambda L: "; ".join(f"{a}+{b} (U {u:.2f}, J {j:.2f})" for a, b, u, j in L) or "-"
print(f"3) pairs grouped together in at least 70% of {Bn} bootstrap resamples (k={K}):")
print(f"   in BOTH spaces ({len(both)}): {fmt(both)}"); print(f"   only in U ({len(onlyU)}): {fmt(onlyU)}"); print(f"   only in J ({len(onlyJ)}): {fmt(onlyJ)}")
cu = fcluster(linkage(CU, "average", metric="euclidean"), K, "maxclust"); cj = fcluster(linkage(CJ, "average", metric="cosine"), K, "maxclust")
print("   U groups (full data):", [[b for b, c in zip(beh, cu) if c == q] for q in sorted(set(cu))]); print("   J groups (full data):", [[b for b, c in zip(beh, cj) if c == q] for q in sorted(set(cj))])

# 4) U-profile and what the J-lens reads out for each behavior (centroid minus grand mean)
prof = pd.DataFrame(CU, index=beh, columns=[c[:12] for c in cats]); prof.insert(0, "mean", prof.mean(1))
if words is not None: prof["words"] = [words[lab == b].mean() for b in beh]
print("4) U-profile (z-scores of the mean coordinate; + = uncertainty side) and J-lens readout of each behavior's direction:")
print(prof.sort_values("mean").round(2).to_string())
bad = re.compile(r"[^A-Za-z一-鿿]"); read = {}
for b in beh:
    lg = WU @ (g * (Jm @ Hc[lab == b].mean(0))); top = []
    for i in np.argsort(-lg)[:400]:
        w = tok.decode([int(i)]).strip()
        if len(w) > 2 and not bad.search(w) and w.lower() not in [t.lower() for t in top]: top.append(w)
        if len(top) == 10: break
    read[b] = top; extra = ""
    if texts is not None:
        en = [w for w in top if w.isascii()]; extra = "  | share of these words occurring in this behavior's sampled responses: " + (f"{np.mean([w.lower() in texts[b] for w in en]):.1f}" if en else "n/a")
    print(f"   {b:22s} {', '.join(top)}{extra}")
if words is not None:
    A = H @ U; print("   dataset-level U-coordinates vs mean response length (Spearman):", {c[:12]: round(float(spearmanr(A[:, i], words)[0]), 2) for i, c in enumerate(cats)})
json.dump(dict(mode=mode, block=BLOCK, n=int(n), acc=dict(U=acc_U, rand=float(acc_r.mean()), rand_p5=float(np.percentile(acc_r, 5)), rand_p95=float(np.percentile(acc_r, 95)), word=float(acc_w.mean()), word_p5=float(np.percentile(acc_w, 5)), word_p95=float(np.percentile(acc_w, 95))),
               eta2=dict(raw=dict(zip(cats, map(float, e_raw))), rand_mean=float(e_r.mean()), word_mean=float(e_w.mean()), word_p95=float(np.percentile(e_w, 95))),
               mantel=dict(U=float(r_obs), p=float(np.mean(perm >= r_obs)), rand_mean=float(r_r.mean()), rand_p95=float(np.percentile(r_r, 95)), word_mean=float(r_w.mean()), word_p95=float(np.percentile(r_w, 95))),
               both=both, onlyU=onlyU, onlyJ=onlyJ, groups_U=[[b for b, c in zip(beh, cu) if c == q] for q in sorted(set(cu))], groups_J=[[b for b, c in zip(beh, cj) if c == q] for q in sorted(set(cj))],
               profile=prof.round(3).to_dict("index"), readout=read, fU=fU.round(3).tolist(), fJ=fJ.round(3).tolist(), beh=beh), open(f"{out}/result2_{mode}_block{BLOCK}.json", "w"), indent=1)
