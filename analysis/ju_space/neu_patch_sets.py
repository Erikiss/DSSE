"""Causal test of neurons as working units: set a SET of MLP neurons (blocks 0-21, last prompt token) to their mean activation
on the target family B and add what that changes in their writes to the block-21 state (direct effect of the writes; the
neurons in between are not recomputed). Sets: the k leading carriers of the class-mean difference (chosen on the surrogate's
training prompts), k random neurons, all neurons (= the MLP-written part), the attention-written part; and for comparison the
k leading SAE latents chosen by the same criterion. States are written as variant set <tag>n for mim_patch.py.
Usage: python neu_patch_sets.py <scratch_root> <out_dir> [t0]"""
import glob, json, os, re, sys
import numpy as np
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mim_common import *
root, out = sys.argv[1], sys.argv[2]; tag = sys.argv[3] if len(sys.argv) > 3 else "t0"
lab, text, X, F = load_eval(out); n, d = X.shape; A = np.load(f"{out}/neu_act.npy", mmap_mode="r"); NI = A.shape[2]; NB = BLOCK + 1
MO = np.load(f"{out}/neu_mlp_out.npy", mmap_mode="r"); AO = np.load(f"{out}/neu_attn_out.npy", mmap_mode="r"); S_all = np.load(f"{out}/mim_sae_acts_{tag}.npy"); sae = SAE(f"{root}/sae_dl/k_L21_{tag}_ae.pt")
pairs = json.load(open(f"{out}/mim_variants_{tag}.json"))["pairs"]; WD = {}
for f in glob.glob(f"{root}/models/qwen3-1.7b/*.safetensors"):
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            q = re.search(r"layers\.(\d+)\.mlp\.down_proj\.weight", k)
            if q and int(q.group(1)) < NB: WD[int(q.group(1))] = sf.get_tensor(k).float().numpy()
KS = [20, 100, 300, 1000, 3000]; KSAE = [20, 100, 300]; V, meta = [], []
def write(aset, delta):                                   # sum over a set of flat neuron indices of delta * write vector
    x = np.zeros(d); l_of = aset // NI
    for l in np.unique(l_of):
        q = aset[l_of == l] % NI; x += WD[int(l)][:, q].astype(np.float64) @ delta[int(l), q]
    return x
for P in pairs:
    pi, a, b = P["pair"], P["a"], P["b"]; rng = np.random.default_rng(300 + pi); S1 = np.array(P["S1"]); iA, iB = np.sort(S1[lab[S1] == a]), np.sort(S1[lab[S1] == b]); dm = X[iB].mean(0) - X[iA].mean(0); v = dm / (dm @ dm)
    aA = np.asarray(A[iA, :NB], dtype=np.float64).mean(0); aB = np.asarray(A[iB, :NB], dtype=np.float64).mean(0); proj = np.stack([WD[l].T.astype(np.float64) @ v for l in range(NB)])
    c = ((aB - aA) * proj).reshape(-1); order = np.argsort(-np.abs(c)); rorder = rng.permutation(len(c)); mB = np.asarray(MO[iB, :NB], dtype=np.float64).mean(0); tB = np.asarray(AO[iB, :NB], dtype=np.float64).mean(0)
    sA, sB = S_all[iA].mean(0).astype(np.float64), S_all[iB].mean(0).astype(np.float64); cs = (sB - sA) * (sae.Wd.T.astype(np.float64) @ v); order_s = np.argsort(-np.abs(cs))
    sur_w, sur_b, ind_w, ind_b = np.array(P["sur_w"]), P["sur_b"], np.array(P["ind_w"]), P["ind_b"]; chk = 0.0
    def add(i, name, x, **kw): V.append(x.astype(np.float32)); meta.append(dict(pair=pi, src=int(i), name=name, valid=True, dist=float(np.linalg.norm(x - X[i])), cm=float((x - X[i]) @ v), s_sur=float(x @ sur_w + sur_b), s_ind=float(x @ ind_w + ind_b), **kw))
    for i in P["src"]:
        a0 = np.asarray(A[i, :NB], dtype=np.float64); dl = aB - a0
        for k in KS: add(i, f"neu_top{k}", X[i] + write(order[:k], dl), k=k); add(i, f"neu_rand{k}", X[i] + write(rorder[:k], dl), k=k)
        dm_mlp = (mB - np.asarray(MO[i, :NB], dtype=np.float64)).sum(0); dm_att = (tB - np.asarray(AO[i, :NB], dtype=np.float64)).sum(0)
        add(i, "mlp_all", X[i] + dm_mlp); add(i, "attn_all", X[i] + dm_att); chk = max(chk, float(np.abs(X[i] + dm_mlp + dm_att - X[iB].mean(0)).max()))
        s0 = sae.encode(X[i])[0].astype(np.float64)
        for k in KSAE: j = order_s[:k]; add(i, f"sae_top{k}", X[i] + sae.Wd[:, j].astype(np.float64) @ (sB - s0)[j], k=k)
    tot = c.sum(); print(f"{a} -> {b}: carriers hold {', '.join(f'top {k}: {c[order[:k]].sum() / tot:.2f}' for k in KS)} of the MLP-written class-mean difference; check MLP part + attention part = centroid of B: max deviation {chk:.2e}", flush=True)
V = np.stack(V); np.save(f"{out}/mim_variants_{tag}n.npy", V); json.dump(dict(meta=meta), open(f"{out}/mim_variants_{tag}n.json", "w"))
import pandas as pd
M = pd.DataFrame(meta); M["sur_B"] = (M.s_sur > 0).astype(float); print(V.shape); print(M.groupby(["pair", "name"])[["cm", "dist", "sur_B"]].mean().groupby("name").mean().round(2).to_string())
