"""MLP neurons against SAE latents (and against rotated bases) as separating units, at the last prompt token (Qwen3-1.7B).
1) all 153 pairs of prompt families: how well does a single unit / do 5 / 20 units separate held-out prompts? Unit sets:
   SAE latents (block 21), neurons of block 21, neurons of blocks 0-21, all neurons; controls: the block-21 neurons in a
   randomly rotated basis, and the block-21 residual state in a random orthonormal basis.
2) where a class difference sits: share written by MLPs against attention at this token, and how few neurons carry the MLP part
   (against the same count in a rotated basis); the same for the score gap of the hard-margin probe.
3) concentration of the neuron activations at this token; names (lens words) of the leading neurons.
Usage: python neu_static.py <scratch_root> <out_dir> [t0]"""
import glob, itertools, json, os, re, sys
import numpy as np, pandas as pd
from safetensors import safe_open
from transformers import AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mim_common import *
from mim_static_helpers import col_auc

root, out = sys.argv[1], sys.argv[2]; tag = sys.argv[3] if len(sys.argv) > 3 else "t0"; rng = np.random.default_rng(0)
lab, text, X, F = load_eval(out); n, d = X.shape; beh = sorted(set(lab)); SC = float(np.median(np.linalg.norm(X, axis=1)))
A = np.load(f"{out}/neu_act.npy"); L, NI = A.shape[1], A.shape[2]; MO = np.load(f"{out}/neu_mlp_out.npy").astype(np.float32); AO = np.load(f"{out}/neu_attn_out.npy").astype(np.float32)
S_all = np.load(f"{out}/mim_sae_acts_{tag}.npy"); mj = json.load(open(f"{out}/mim_variants_{tag}.json")); pairs8 = mj["pairs"]
WD = {}
for f in glob.glob(f"{root}/models/qwen3-1.7b/*.safetensors"):
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            q = re.search(r"layers\.(\d+)\.mlp\.down_proj\.weight", k)
            if q and int(q.group(1)) < 22: WD[int(q.group(1))] = sf.get_tensor(k).float().numpy()
Qn = np.linalg.qr(rng.standard_normal((NI, NI)).astype(np.float32))[0]; Qr = np.linalg.qr(rng.standard_normal((d, d)))[0]
print(f"[neurons, last prompt token] {n} prompts, {L} blocks x {NI} neurons; reconstruction check of the block-21 state from attention and MLP writes: "
      f"max deviation between prompts' differences {np.abs((X - X.mean(0)) - ((MO[:, :22].sum(1) + AO[:, :22].sum(1)) - (MO[:, :22].sum(1) + AO[:, :22].sum(1)).mean(0))).max():.2e}")
UNITS = {"sae_16k": S_all.astype(np.float32), "neurons_block21": A[:, 21].astype(np.float32), "neurons_blocks0to21": A[:, :22].reshape(n, -1), "neurons_all": A.reshape(n, -1),
         "rotated_neurons_block21": A[:, 21].astype(np.float32) @ Qn.T, "random_basis_of_state": ((X - X.mean(0)) @ Qr).astype(np.float32)}
def unit_stats(U, tr, te, ytr, yte, ncand=200):
    Utr = U[tr].astype(np.float32); m1, m0 = Utr[ytr == 1].mean(0), Utr[ytr == 0].mean(0); sd = np.sqrt((Utr[ytr == 1].var(0) + Utr[ytr == 0].var(0)) / 2) + 1e-6
    cand = np.argsort(-np.abs((m1 - m0) / sd))[:ncand]; Ct, Ce = Utr[:, cand].astype(np.float64), U[te][:, cand].astype(np.float64)
    au = col_auc(Ct, ytr); o = np.argsort(-np.abs(au - .5)); sg = np.where(au >= .5, 1.0, -1.0); au_te = col_auc(Ce * sg, yte)          # test AUROC oriented by the training sign
    r = dict(best_unit=int(cand[o[0]]), best_train_auc=float(max(au[o[0]], 1 - au[o[0]])), best_test_auc=float(au_te[o[0]]))
    for k in (1, 5, 20):
        j = o[:k]; mu, s = Ct[:, j].mean(0), Ct[:, j].std(0) + 1e-9; z = lambda M: (((M[:, j] - mu) / s) * sg[j]).sum(1); thr = (z(Ct)[ytr == 1].mean() + z(Ct)[ytr == 0].mean()) / 2
        r[f"acc{k}"] = float(((z(Ce) > thr) == yte).mean())
    strong = np.abs(au - .5) >= .4; r["n_train90"] = int(strong.sum()); r["n_test90"] = int((au_te[strong] >= .9).sum()); return r
rows = []
for (a, b), seed in itertools.product(itertools.combinations(beh, 2), range(3)):
    S1, S2, T = split3(lab, a, b, seed); tr = np.concatenate([S1, S2]); ytr = (lab[tr] == b).astype(int); yte = (lab[T] == b).astype(int)
    for name, U in UNITS.items(): rows.append(dict(a=a, b=b, seed=seed, units=name, **unit_stats(U, tr, T, ytr, yte)))
R = pd.DataFrame(rows); R.to_csv(f"{out}/neu_static_units.csv", index=False); pd.set_option("display.width", 220)
M = R.groupby(["units", "a", "b"])[["best_test_auc", "acc1", "acc5", "acc20", "n_train90", "n_test90"]].mean().groupby("units").agg(["mean", lambda v: v.quantile(.05), lambda v: v.quantile(.95)])
M.columns = [f"{c}_{'mean' if s == 'mean' else ('p5' if '0' in s else 'p95')}" for c, s in M.columns]; M["n_units"] = [UNITS[u].shape[1] for u in M.index]
order = list(UNITS); print("1) single units as separators, all 153 pairs x 3 splits (units chosen on training prompts, scored on held-out prompts). Mean over pairs:")
print(M.loc[order, ["n_units", "best_test_auc_mean", "acc1_mean", "acc1_p5", "acc5_mean", "acc20_mean", "n_train90_mean", "n_test90_mean"]].round(3).to_string())
res = dict(units=M.loc[order].round(4).reset_index().to_dict("records"))
# ---------- 2) where the class difference sits
def carriers(c, total):                                   # how many units (largest |contribution| first) until the running sum reaches 50% / 90% of the total
    o = np.argsort(-np.abs(c)); cs = np.cumsum(c[o]); return [int(np.argmax(cs >= q * total) + 1) if (cs >= q * total).any() else len(c) for q in (.5, .9)]
rows2 = []; tops = {}
for P in pairs8:
    a, b = P["a"], P["b"]; S1 = np.array(P["S1"]); iA, iB = S1[lab[S1] == a], S1[lab[S1] == b]; dm = X[iB].mean(0) - X[iA].mean(0); w = np.array(P["sur_w"]); res_p = dict(a=a, b=b)
    for nm, v in (("classmean", dm / (dm @ dm)), ("probe", w / float(w @ dm))):       # v . (difference of a component) = that component's share of the total
        sh_m = np.array([(MO[iB, l].mean(0) - MO[iA, l].mean(0)) @ v for l in range(22)]); sh_a = np.array([(AO[iB, l].mean(0) - AO[iA, l].mean(0)) @ v for l in range(22)])
        c = np.concatenate([(A[iB, l].astype(np.float64).mean(0) - A[iA, l].astype(np.float64).mean(0)) * (WD[l].T.astype(np.float64) @ v) for l in range(22)])
        cr = np.concatenate([(Qn.astype(np.float64) @ (A[iB, l].astype(np.float64).mean(0) - A[iA, l].astype(np.float64).mean(0))) * (Qn.astype(np.float64) @ (WD[l].T.astype(np.float64) @ v)) for l in range(22)])
        k50, k90 = carriers(c, c.sum()); r50, r90 = carriers(cr, cr.sum()); o = np.argsort(-np.abs(c))
        res_p.update({f"{nm}_share_mlp": float(sh_m.sum()), f"{nm}_share_attention": float(sh_a.sum()), f"{nm}_check": float(sh_m.sum() + sh_a.sum()), f"{nm}_mlp_check": float(c.sum() / sh_m.sum()),
                      f"{nm}_k50": k50, f"{nm}_k90": k90, f"{nm}_rot_k50": r50, f"{nm}_rot_k90": r90, f"{nm}_top10_share_of_mlp": float(c[o[:10]].sum() / c.sum()), f"{nm}_share_last_blocks_18to21": float((sh_m[18:].sum() + sh_a[18:].sum()))})
        tops[(a, b, nm)] = [(int(j // NI), int(j % NI), float(c[j] / c.sum())) for j in o[:50]]
    t1, t2 = {(l, q) for l, q, _ in tops[(a, b, "classmean")]}, {(l, q) for l, q, _ in tops[(a, b, "probe")]}; res_p["top50_overlap_classmean_probe"] = len(t1 & t2)
    rows2.append(res_p)
T2 = pd.DataFrame(rows2); print("\n2) where the class difference at block 21 comes from (8 attack pairs; shares add up to 1; k50 / k90 = number of neurons of blocks 0-21 carrying 50% / 90% of the MLP part; rot = same count in a rotated basis)")
print(T2[["a", "b", "classmean_share_mlp", "classmean_share_attention", "classmean_k50", "classmean_k90", "classmean_rot_k50", "classmean_rot_k90", "classmean_top10_share_of_mlp", "classmean_share_last_blocks_18to21"]].round(2).to_string())
print("   the same for the score gap of the hard-margin probe (its normal instead of the class-mean difference):")
print(T2[["a", "b", "probe_share_mlp", "probe_share_attention", "probe_k50", "probe_k90", "probe_rot_k50", "probe_rot_k90", "probe_top10_share_of_mlp", "top50_overlap_classmean_probe"]].round(2).to_string())
print("   means:", T2.mean(numeric_only=True).round(2).to_dict()); res["carriers"] = T2.round(4).to_dict("records")
# ---------- 3) concentration at this token, and names of leading neurons
wn = np.load(f"{out}/neu_wdown_norm.npy"); conc = []
for l in range(L):
    x = A[:, l].astype(np.float32); w_ = x * wn[l]; K = int(round(.078 * NI)); e = lambda z: float((np.sort(z ** 2, 1)[:, -K:].sum(1) / (z ** 2).sum(1)).mean()); conc.append((e(w_), e(x @ Qn.T)))
print(f"\n3) share of the squared write size in the 7.8% largest units at the last prompt token: neurons {np.mean([c[0] for c in conc]):.2f} (blocks 0/7/14/21/27: {', '.join(f'{conc[l][0]:.2f}' for l in (0, 7, 14, 21, 27))}), rotated basis {np.mean([c[1] for c in conc]):.2f}")
res["concentration_last_token"] = dict(neuron=float(np.mean([c[0] for c in conc])), rotated=float(np.mean([c[1] for c in conc])), by_block=[dict(block=l, neuron=conc[l][0], rotated=conc[l][1]) for l in range(L)])
tok = AutoTokenizer.from_pretrained(f"{root}/models/qwen3-1.7b"); WU, g = load_unembed(root); import torch
JJ = torch.load(f"{root}/jlens/Qwen3-1.7B_jacobian_lens.pt", map_location="cpu", weights_only=True)["J"]; bad = re.compile(r"[^A-Za-z一-鿿]")
def words(v, l, k=6):
    lg = WU @ (g * (JJ[min(l, 26)].float().numpy().astype(np.float64) @ v)).astype(np.float32); top = []
    for i in np.argsort(-lg)[:300]:
        w_ = tok.decode([int(i)]).strip()
        if len(w_) > 2 and not bad.search(w_) and w_.lower() not in [t.lower() for t in top]: top.append(w_)
        if len(top) == k: break
    return top
print("   leading neurons of the class-mean difference (block/neuron, share of the MLP part, family activations, lens words of what the neuron writes when it grows towards B):"); names = {}
for P in pairs8:
    a, b = P["a"], P["b"]; print(f"   {a} -> {b}")
    for l, q, sh in tops[(a, b, "classmean")][:4]:
        x = A[:, l, q].astype(np.float64); sgn = np.sign(x[lab == b].mean() - x[lab == a].mean()); wds = words(sgn * WD[l][:, q].astype(np.float64), l); names[f"{l}/{q}"] = wds
        print(f"      L{l}/N{q}: share {sh:+.3f} | mean activation A {x[lab == a].mean():+.2f}, B {x[lab == b].mean():+.2f}, all others {x[(lab != a) & (lab != b)].mean():+.2f} | {', '.join(wds)}")
res["names"] = names; res["top_carriers"] = {f"{a}>{b}|{nm}": v[:20] for (a, b, nm), v in tops.items()}
json.dump(res, open(f"{out}/neu_static.json", "w"), indent=1, ensure_ascii=False); print("saved")
