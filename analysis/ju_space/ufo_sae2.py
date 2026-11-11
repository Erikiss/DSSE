"""An SAE in the loop of Qwen3-1.7B, measured on the same scale as the perturbations of ufo_damp2.py.

At one block the state h of every valid position is changed; e = h - SAE(h) is what the SAE fails to reconstruct.
  splice      h - e      = SAE(h): the SAE runs in the loop                    half     h - e/2
  plus        h + e      the unexplained part doubled                          foreign  h - |e| * (error direction of another real state)
  cov_e       h + |e| * (covariance-matched noise)                             irr_e    h + |e| * (direction outside the data subspace)
  pca         h projected onto as many principal axes of real states as give the SAE's share of explained variance
With noise d of size 0.3 |h| added at the same block (classes cov and irr; blocks 0, 2, 14):
  noise       h + d                                  sae_rt      SAE(h + d): the SAE in real time on the disturbed state
  sae_oracle  SAE(h + d) + e: only the noise that passes the SAE, without its own error
  pca_oracle  h + (projection of d): the same for the linear projector
SAEs: Qwen-Scope TopK (32768 units, k = 50, trained on the BASE model) after blocks 0, 2, 8, 14, 21; for block 21 also the
BatchTopK SAE trained on the chat model (16384 units, with its threshold) as a check on the base/chat mismatch.
On the clean pass, without any change: for every later block the coefficient <write of the block, v> / |v|^2 along the part of the
clean state that the SAE does not explain (e_sae), that it explains (r_sae, mean removed), that lies outside / inside the data
subspace (e_out, r_in) and along a random vector (rnd). Negative = the block writes against that part.
On a sample of real states: how compressible are the state, its SAE reconstruction and the SAE error (held-out share of the
z-scored variance in 2 / 8 / 32 / 128 principal axes)?
Recorded: at the site the SAE's share of explained variance, the size of e, where e lies, how much of the noise passes
(|SAE(h + d) - SAE(h)| / |d|) and how many active units stay active; downstream amp, p, s as in ufo_damp2.py; at the output per
position KL(clean || row), entropy, loss, same top token.
Usage: python ufo_sae2.py <scratch_root> <out_dir>     env: NT, NTEXT (texts per source, default 6)"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ufo_common import load_model, texts, Pool, unit, MAXT

root, out = sys.argv[1], sys.argv[2]; torch.set_grad_enabled(False); torch.set_num_threads(int(os.environ.get("NT", "3"))); TAG = os.environ.get("TAG", ""); tok, model = load_model(root); LAYERS = model.model.layers; L = len(LAYERS); D = model.config.hidden_size
main, _ = texts(root, tok); nps = int(os.environ.get("NTEXT", "6")); half = len(main) // 2; main = main[:nps] + main[half:half + nps]; SITES = [("qs0", 0), ("qs2", 2), ("qs8", 8), ("qs14", 14), ("qs21", 21), ("kv21", 21)]
if os.environ.get("SITES"): SITES = [s for s in SITES if s[0] in os.environ["SITES"].split(",")]
NOISE = {"qs0", "qs2", "qs14"} & {s for s, _ in SITES}; FN = 0.3; P = Pool(f"{out}/ufo_pool.pt", blocks=sorted({b for _, b in SITES}))


class TopK:                                                             # Qwen-Scope: pre = x W_enc^T + b_enc, keep the 50 largest, decode with W_dec and b_dec
    def __init__(s, path, k=50):
        sd = torch.load(path, map_location="cpu", weights_only=True); s.We, s.be, s.Wd, s.bd, s.k = sd["W_enc"].float(), sd["b_enc"].float(), sd["W_dec"].float().T.contiguous(), sd["b_dec"].float(), k

    def __call__(s, x, codes=False):
        o, ii = [], []
        for i in range(0, len(x), 256):
            v, ix = (x[i:i + 256] @ s.We.T + s.be).topk(s.k, -1); o.append((s.Wd[ix] * v[..., None]).sum(1) + s.bd); ii.append(ix)
        return (torch.cat(o), [set(r.tolist()) for r in torch.cat(ii)]) if codes else torch.cat(o)


class BatchTopK:                                                        # dictionary_learning format, with the fixed threshold
    def __init__(s, path):
        sd = torch.load(path, map_location="cpu", weights_only=True); s.We, s.be, s.Wd, s.bd, s.thr = sd["encoder.weight"].float(), sd["encoder.bias"].float(), sd["decoder.weight"].float(), sd["b_dec"].float(), float(sd["threshold"])

    def __call__(s, x, codes=False):
        o, ii = [], []
        for i in range(0, len(x), 256):
            a = torch.relu((x[i:i + 256] - s.bd) @ s.We.T + s.be); a = a * (a > s.thr); o.append(a @ s.Wd.T + s.bd); ii += [set(r.nonzero()[:, 0].tolist()) for r in a]
        return (torch.cat(o), ii) if codes else torch.cat(o)


def compress(Mx):                                                        # held-out share of the z-scored variance in 2 / 8 / 32 / 128 principal axes (fit: first two thirds of the sample)
    nf = 2 * len(Mx) // 3; A, B = Mx[:nf], Mx[nf:]; mu_, sd_ = A.mean(0), A.std(0).clamp(min=1e-8); A, B = (A - mu_) / sd_, (B - mu_) / sd_; V = torch.linalg.svd(A.double(), full_matrices=False)[2].T.float()
    return {k: float(((B @ V[:, :k]) ** 2).sum() / (B ** 2).sum()) for k in (2, 8, 32, 128)}


t0 = time.time(); SAE, EP, KP, pstat = {}, {}, {}, {}
for s, b in SITES:                                                      # pool sample: explained variance, error directions of other real states, number of principal axes with the same explained variance
    SAE[s] = BatchTopK(f"{root}/sae_dl/k_L21_t0_ae.pt") if s.startswith("kv") else TopK(f"{root}/qscope_dl/layer{b}.sae.pt"); X = P.b[b]["sample"]; mu = P.b[b]["mu"]; E = X - SAE[s](X); tot = ((X - mu) ** 2).sum(); fve = float(1 - (E ** 2).sum() / tot)
    pr = ((X - mu) @ P.b[b]["U"]) ** 2; cum = torch.cumsum(pr.sum(0), 0) / tot; KP[s] = int((cum < fve).sum()) + 1; EP[s] = unit(E)
    pstat[s] = dict(block=b, fve_pool=fve, k_pca_same_fve=KP[s], rel_err_median=float((E.norm(dim=1) / X.norm(dim=1)).median()), err_inside_data_subspace=P.inside(b, E), norm_ratio=float((SAE[s](X).norm(dim=1) / X.norm(dim=1)).median()), k_irr=int(P.b[b]["k_irr"]), compress_state=compress(X), compress_recon=compress(X - E), compress_err=compress(E))
    print(s, {k: (round(v, 3) if isinstance(v, float) else ({a: round(b_, 3) for a, b_ in v.items()} if isinstance(v, dict) else v)) for k, v in pstat[s].items()}, f"({time.time() - t0:.0f}s)", flush=True)
ROWS = [("clean", None, None)]
for s, b in SITES:
    ROWS += [(f"{s}_{v}", s, v) for v in ("splice", "half", "plus", "foreign", "cov_e", "irr_e", "pca")]
    if s in NOISE: ROWS += [(f"{s}_{c}_{v}", s, f"{c}:{v}") for c in ("cov", "irr") for v in ("noise", "sae_oracle", "sae_rt", "pca_oracle")]
C = len(ROWS); BY = {}
for c, (n, s, v) in enumerate(ROWS):
    if s: BY.setdefault(s, []).append(c)
print(C, "rows per text,", len(main), "texts", flush=True); NAMES = ["amp", "p", "s"]; st = {}; G = None


def block_hook(m):
    def f(mod, inp, o):
        h = o[0] if isinstance(o, tuple) else o; hc = h[0].clone(); nc = hc.norm(dim=-1); valid = st["valid"]
        for s, b in SITES:
            if b != m: continue
            ok = valid & (nc < 10 * P.med(m)); X = hc[ok]; n = len(X); Xh, act = SAE[s](X, codes=True); e = X - Xh; en = e.norm(dim=-1, keepdim=True); mu = P.b[m]["mu"]; Uk = P.b[m]["U"][:, :KP[s]]; proj = lambda v: (v @ Uk) @ Uk.T
            dr = {c: P.direction(c, m, n, G) for c in ("cov", "irr")}; j = torch.randint(0, len(EP[s]), (n,), generator=G); info = dict(fve=float(1 - (e ** 2).sum() / ((X - mu) ** 2).sum()), rel_err=float((en[:, 0] / X.norm(dim=-1)).median()), err_inside=P.inside(m, e),
                                                                                                                                 err_radial=float((((e * X).sum(-1)) ** 2 / ((e ** 2).sum(-1) * (X ** 2).sum(-1))).mean()), fve_pca=float(1 - ((X - mu - proj(X - mu)) ** 2).sum() / ((X - mu) ** 2).sum()), n=n)
            new = dict(splice=X - e, half=X - e / 2, plus=X + e, foreign=X - en * EP[s][j], cov_e=X + en * dr["cov"], irr_e=X + en * dr["irr"], pca=mu + proj(X - mu))
            if s in NOISE:
                for c in ("cov", "irr"):
                    d = FN * X.norm(dim=-1, keepdim=True) * dr[c]; Yh, act2 = SAE[s](X + d, codes=True); pas = Yh - Xh
                    new.update({f"{c}:noise": X + d, f"{c}:sae_oracle": Yh + e, f"{c}:sae_rt": Yh, f"{c}:pca_oracle": X + proj(d)})
                    info.update({f"pass_sae_{c}": float((pas.norm(dim=-1) / d.norm(dim=-1)).mean()), f"pass_pca_{c}": float((proj(d).norm(dim=-1) / d.norm(dim=-1)).mean()), f"pass_sae_cos_{c}": float(torch.nn.functional.cosine_similarity(pas, d, dim=-1).mean()),
                                 f"units_kept_{c}": float(np.mean([len(a & a2) / max(1, len(a)) for a, a2 in zip(act, act2)])), f"sae_err_rel_noisy_{c}": float(((X + d - Yh).norm(dim=-1) / X.norm(dim=-1)).median())})
            ki = int(P.b[m]["k_irr"]); Ui = P.b[m]["U"][:, :ki]; ins = ((X - mu) @ Ui) @ Ui.T; st["vec"][s] = (ok, dict(e_sae=e, r_sae=Xh - mu, e_out=(X - mu) - ins, r_in=ins, rnd=unit(torch.randn(n, D, generator=G)) * en))
            for c in BY[s]:
                row = h[c]; row[ok] = new[ROWS[c][2]]; st["delta"][c] = row - hc; st["ok"][c] = ok
            st["site"][s] = info
        for c in list(st["delta"]):
            ok = st["ok"][c]; d = st["delta"][c]; dl = h[c] - hc; d2 = (d ** 2).sum(-1)[ok].clamp(min=1e-30); par = (dl * d).sum(-1)[ok]; n2 = (dl ** 2).sum(-1)[ok]
            st["amp"][c, m] = float((n2 / d2).sqrt().mean()); st["p"][c, m] = float((par / d2).mean()); st["s"][c, m] = float(((n2 - par ** 2 / d2).clamp(min=0) / d2).sqrt().mean())
        return o
    return f


VEC = ["e_sae", "r_sae", "e_out", "r_in", "rnd"]


def write_hook(m, j):                                                     # clean row only: coefficient of the block's write along each stored part of the clean state, <write, v> / |v|^2
    def f(mod, inp, o):
        w0 = (o[0] if isinstance(o, tuple) else o)[0]
        for si, (s, b) in enumerate(SITES):
            if s in st["vec"] and m > b:
                ok, vs = st["vec"][s]
                for vi, vn in enumerate(VEC): st["bcoef"][si, vi, m, j] = float(((w0[ok] * vs[vn]).sum(-1) / (vs[vn] ** 2).sum(-1).clamp(min=1e-30)).mean())
    return f


for m in range(L): LAYERS[m].register_forward_hook(block_hook(m)); LAYERS[m].self_attn.register_forward_hook(write_hook(m, 0)); LAYERS[m].mlp.register_forward_hook(write_hook(m, 1))
RES = {n: [] for n in NAMES}; PP = {n: [] for n in ("kl", "H", "ce", "top")}; BC = []; meta = []; t1 = time.time()
for k, (src, ids, valid) in enumerate(main):
    T = ids.shape[1]; G = torch.Generator().manual_seed(2000 + k); st = dict(valid=valid, delta={}, ok={}, site={}, vec={}, bcoef=np.full((len(SITES), len(VEC), L, 2), np.nan), **{n: np.full((C, L), np.nan) for n in NAMES})
    hf = model.model(input_ids=ids.repeat(C, 1)).last_hidden_state; tf = time.time() - t1; sl = slice(8, T - 1); tgt = ids[0, 9:T]; ar = torch.arange(len(tgt)); pp = {n: np.full((C, MAXT), np.nan) for n in PP}; lp0 = None
    for c0 in range(0, C, 6):
        lq = torch.log_softmax(model.lm_head(hf[c0:c0 + 6, sl]).double(), -1)
        if lp0 is None: lp0 = lq[0].clone(); p0 = lp0.exp(); top0 = lp0.argmax(-1)
        cs = slice(c0, c0 + lq.shape[0]); pp["kl"][cs, sl] = (p0 * (lp0 - lq)).sum(-1).numpy(); pp["H"][cs, sl] = (-(lq.exp() * lq).sum(-1)).numpy(); pp["ce"][cs, sl] = (-lq[:, ar, tgt]).numpy(); pp["top"][cs, sl] = (lq.argmax(-1) == top0).double().numpy(); del lq
    for n in NAMES: RES[n].append(st[n])
    for n in PP: PP[n].append(pp[n])
    BC.append(st["bcoef"]); v = valid[sl].numpy(); meta.append(dict(text=k, source=src, tokens=int(T), valid=[int(x) for x in valid], clean_loss=float(np.nanmean(pp["ce"][0, sl][v])), site=st["site"]))
    print(f"  text {k + 1}/{len(main)} {src}: forward {tf:.0f}s, total {time.time() - t1:.0f}s | " + " ".join(f"{s} KL {np.nanmean(pp['kl'][BY[s][0], sl][v]):.3f}" for s, _ in SITES), flush=True)
    np.savez(f"{out}/ufo_sae2{TAG}.npz", bcoef=np.array(BC), **{n: np.array(RES[n]) for n in NAMES}, **{"pp_" + n: np.array(PP[n]) for n in PP}); json.dump(dict(rows=ROWS, sites=SITES, pool=pstat, meta=meta, noise_f=FN, vec=VEC), open(f"{out}/ufo_sae2{TAG}_meta.json", "w"))
print(f"done ({time.time() - t0:.0f}s)")
