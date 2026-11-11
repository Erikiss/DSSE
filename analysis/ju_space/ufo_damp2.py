"""How do the blocks of Qwen3-1.7B react to a vector that is added to the residual stream? Second version, after the plan review.

Direction classes (unit vectors, all sign-symmetric, none contains the current state; see ufo_common.Pool.direction):
  cov   Gaussian with the covariance of real states at that block        (inside the space the data use)
  irr   random vector outside the principal subspace of real states      (outside; subspace chosen on held-out texts)
  mix   difference of two real states of other texts                     (inside, real content)
  rand  isotropic                                                        (literature baseline)
Size: f * |h| at each position. Directions are drawn once per text and reused for all sizes and variants.

Rows of one batch per text:
  clean, null      no change (null = numerical floor)
  free rows        the vector is added after block l (2 and 14: all classes and variants; 0 and 8: fewer) and the network runs on.
                   Variants: normonly (state stretched by sqrt(1 + f^2), no new direction), np (direction added, norm kept),
                   A (from every later write of attention and MLP the component along the added vector is removed: no erasure),
                   B (only that component is kept: erasure only), single (one position only), same (one vector at all positions).
  local rows       after EVERY block the state is reset to clean state + fresh vector; recorded is what the next block writes in
                   response. This gives every block's reaction without propagation and without the growth of the stream.
  abl rows         the MLP write of one block is set to zero (loss of information, for comparison of the confidence measures)
Recorded per row and block m:
  free:  amp = |dev| / |delta|, p = <dev, delta> / |delta|^2, s = part of dev orthogonal to delta / |delta|, rho = |dev| / |h_m|,
         ka, km = -<change of the attention / MLP write of block m, delta> / |delta|^2, ins = share of dev inside the data subspace
  local: q_att, q_mlp, q_tot = -<write difference of block m, delta> / |delta|^2 (erasure coefficient of block m),
         R = -cos(total write difference, delta), sc = part of the write difference orthogonal to delta / |delta|
At the output per position: KL(clean || row) in float64, entropy, loss, same top token, final relative deviation, norm ratio,
entropy with the clean scale of the final norm; and computed without the network: a random vector of the same final size
added after the last block (late), the added vector riding along untouched (frozen), isotropic noise after the last block.
Usage: python ufo_damp2.py <scratch_root> <out_dir>    env: NT, NTEXT, MAXT, NMAIN, RANDINIT=1 (untrained network), CONDSET=small
       (fewer rows), TOPK=0.5 (keep only the strongest half of the neuron activations per position in every block), TAG"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ufo_common import load_model, texts, Pool, unit, FIRST, MAXT

root, out = sys.argv[1], sys.argv[2]; RANDINIT = os.environ.get("RANDINIT") == "1"; SMALL = os.environ.get("CONDSET") == "small"; TOPK = float(os.environ.get("TOPK", "0")); TAG = os.environ.get("TAG", ("_randinit" if RANDINIT else "") + (f"_top{TOPK:g}" if TOPK else ""))
torch.set_grad_enabled(False); torch.set_num_threads(int(os.environ.get("NT", "4"))); tok, model = load_model(root, RANDINIT); LAYERS = model.model.layers; L = len(LAYERS); D = model.config.hidden_size
main, _ = texts(root, tok); main = main[:int(os.environ.get("NTEXT", len(main)))] if "NTEXT" in os.environ else main; P = Pool(f"{out}/ufo_pool{'_randinit' if RANDINIT else ''}.pt")


def cond(name, kind, l=-1, cls="none", f=0.0, mode="", pos="all", var="add"): return dict(name=name, kind=kind, l=l, cls=cls, f=f, mode=mode, pos=pos, var=var)


CONDS = [cond("clean", "clean"), cond("null", "clean")]
for l in (2, 14):
    for c in ("cov", "irr"): CONDS += [cond(f"L{l}_{c}_{f:g}", "free", l, c, f) for f in ((0.3,) if SMALL else (0.03, 0.1, 0.3, 1.0))]
    CONDS += [cond(f"L{l}_rand_0.3", "free", l, "rand", 0.3)]
    if SMALL: continue
    CONDS += [cond(f"L{l}_mix_0.3", "free", l, "mix", 0.3), cond(f"L{l}_normonly_1", "free", l, "none", 1.0, var="normonly")]
    for c in ("cov", "irr"):
        CONDS += [cond(f"L{l}_{c}_1_np", "free", l, c, 1.0, var="normpres"), cond(f"L{l}_{c}_0.3_A", "free", l, c, 0.3, mode="A"), cond(f"L{l}_{c}_0.3_B", "free", l, c, 0.3, mode="B"),
                  cond(f"L{l}_{c}_0.3_single", "free", l, c, 0.3, pos="single"), cond(f"L{l}_{c}_0.3_same", "free", l, c, 0.3, pos="same")]
if not SMALL:
    for l in (0, 8):
        for c in ("cov", "irr"): CONDS += [cond(f"L{l}_{c}_{f:g}", "free", l, c, f) for f in (0.1, 0.3, 1.0)]
        CONDS += [cond(f"L{l}_rand_0.3", "free", l, "rand", 0.3)]
    CONDS += [cond(f"abl_mlp{l}", "abl", l) for l in (2, 8, 14)]
CONDS += [cond(f"loc_{c}_{f:g}", "local", -1, c, f) for c in ("cov", "irr") for f in ((0.3,) if SMALL else (0.03, 0.3, 1.0))] + [cond("loc_rand_0.03", "local", -1, "rand", 0.03), cond("loc_rand_0.3", "local", -1, "rand", 0.3), cond("loc_mix_0.3", "local", -1, "mix", 0.3)]
C = len(CONDS); FREE = [c for c in range(C) if CONDS[c]["kind"] == "free"]; LOC = [c for c in range(C) if CONDS[c]["kind"] == "local"]
INJ = {}; ABL = {}
for c in FREE: INJ.setdefault(CONDS[c]["l"], []).append(c)
for c in range(C):
    if CONDS[c]["kind"] == "abl": ABL.setdefault(CONDS[c]["l"], []).append(c)
LATE = [c for c in FREE if CONDS[c]["var"] == "add" and CONDS[c]["pos"] == "all" and CONDS[c]["mode"] == "" and CONDS[c]["f"] >= 0.3]; LATEF = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
NAMES = ["rho", "amp", "p", "s", "ka", "km", "ins", "imp", "q_att", "q_mlp", "q_tot", "R", "sc"]; print(C, "rows per text,", len(main), "texts", flush=True); st = {}; G = None


def block_hook(m):
    def f(mod, inp, o):
        h = o[0] if isinstance(o, tuple) else o; hc = h[0].clone(); nc = hc.norm(dim=-1); valid = st["valid"]; T = hc.shape[0]; okm = valid & (nc < 10 * P.med(m)); st["norm"][m] = float(nc[valid].median()); st["dropped"][m] = int((valid & ~okm).sum())
        for c in LOC:                                                                      # local rows: read the reaction of block m, then reset to clean + fresh vector
            if c in st["dloc"]:
                d = st["dloc"][c]; ok = st["okloc"][c]; tot = (h[c] - hc) - d; d2 = (d ** 2).sum(-1)[ok]; par = (tot * d).sum(-1)[ok]; t2 = (tot ** 2).sum(-1)[ok]
                st["q_tot"][c, m] = float((-par / d2).mean()); st["R"][c, m] = float((-par / (t2 * d2).sqrt().clamp(min=1e-30)).mean()); st["sc"][c, m] = float(((t2 - par ** 2 / d2).clamp(min=0) / d2).sqrt().mean())
            if m < L - 1:
                key = (m, CONDS[c]["cls"])
                if key not in st["ldirs"]: st["ldirs"][key] = P.direction(CONDS[c]["cls"], m, T, G)
                d = st["ldirs"][key] * nc[:, None] * CONDS[c]["f"] * okm[:, None]; h[c] = hc + d; st["dloc"][c] = d; st["okloc"][c] = okm
        for c in INJ.get(m, []):                                                           # free rows: add the vector once
            cd = CONDS[c]; okc = okm.clone()
            if cd["pos"] == "single": okc[:] = False; okc[st["t0"]] = okm[st["t0"]]
            if cd["var"] == "normonly": d = hc * (float(np.sqrt(1 + cd["f"] ** 2)) - 1) * okc[:, None]
            else:
                key = (m, cd["cls"], "same" if cd["pos"] == "same" else "all")
                if key not in st["dirs"]: st["dirs"][key] = P.direction(cd["cls"], m, 1, G).expand(T, D).clone() if cd["pos"] == "same" else P.direction(cd["cls"], m, T, G)
                d = st["dirs"][key] * nc[:, None] * cd["f"] * okc[:, None]
                if cd["var"] == "normpres": d = ((hc + d) * (nc / (hc + d).norm(dim=-1))[:, None] - hc) * okc[:, None]
            h[c] = hc + d; st["delta"][c] = d; st["ok"][c] = okc
            if okc.any(): st["radial"][c] = float((((d * hc).sum(-1)[okc]) ** 2 / ((d ** 2).sum(-1)[okc] * nc[okc] ** 2)).mean()); st["inside"][c] = P.inside(m, d[okc])
        for c in FREE:                                                                     # free rows: where is the deviation now?
            if c in st["delta"] and st["ok"][c].any():
                ok = st["ok"][c]; d = st["delta"][c]; dl = h[c] - hc; d2 = (d ** 2).sum(-1)[ok]; par = (dl * d).sum(-1)[ok]; n2 = (dl ** 2).sum(-1)[ok]
                st["rho"][c, m] = float((n2.sqrt() / nc[ok]).mean()); st["amp"][c, m] = float((n2 / d2).sqrt().mean()); st["p"][c, m] = float((par / d2).mean()); st["s"][c, m] = float(((n2 - par ** 2 / d2).clamp(min=0) / d2).sqrt().mean())
                if m in P.b and m % 2 == 0: st["ins"][c, m] = P.inside(m, dl[ok])
                if CONDS[c]["pos"] == "single":
                    later = valid.clone(); later[:st["t0"] + 1] = False
                    if later.any(): st["imp"][c, m] = float(dl[later].norm(dim=-1).mean() / d2.sqrt().mean())
        if m == L - 1: st["h27"] = h.clone()
        return o
    return f


def write_hook(m, key):
    def f(mod, inp, o):
        w = o[0] if isinstance(o, tuple) else o; w0 = w[0].clone()
        if key == "km":
            for c in ABL.get(m, []): w[c] = 0
        for c in LOC:
            if c in st["dloc"]:
                d = st["dloc"][c]; ok = st["okloc"][c]; st["q_att" if key == "ka" else "q_mlp"][c, m] = float((-((w[c] - w0) * d).sum(-1)[ok] / (d ** 2).sum(-1)[ok]).mean())
        for c in FREE:
            if c in st["delta"] and st["ok"][c].any():
                d = st["delta"][c]; ok = st["ok"][c]; d2 = (d ** 2).sum(-1); par = ((w[c] - w0) * d).sum(-1); st[key][c, m] = float((-par[ok] / d2[ok]).mean()); md = CONDS[c]["mode"]
                if md: proj = d * (par / d2.clamp(min=1e-30))[:, None]; w[c] = (w[c] - proj) if md == "A" else (w0 + proj)
        return o
    return f


for m in range(L):
    LAYERS[m].register_forward_hook(block_hook(m)); LAYERS[m].self_attn.register_forward_hook(write_hook(m, "ka")); LAYERS[m].mlp.register_forward_hook(write_hook(m, "km"))
if TOPK:                                                                                   # in every block only the strongest share TOPK of the neuron activations of a position is kept, in all rows (the clean row included)
    def keep_top(mod, inp):
        a = inp[0]; thr = a.abs().topk(max(1, int(round(TOPK * a.shape[-1]))), -1).values[..., -1:]; return (a * (a.abs() >= thr),)
    for m in range(L): LAYERS[m].mlp.down_proj.register_forward_pre_hook(keep_top)

RES = {n: [] for n in NAMES}; PP = {n: [] for n in ("kl", "H", "ce", "top", "r", "rhoF", "Hcs", "kl_late", "kl_frozen", "kl_latef", "H_latef")}; EX = {n: [] for n in ("norm", "dropped", "radial", "inside")}; meta = []; t0 = time.time()
for k, (src, ids, valid) in enumerate(main):
    T = ids.shape[1]; G = torch.Generator().manual_seed(1000 + k); cand = [t for t in range(16, T - 8) if bool(valid[t])]
    st = dict(valid=valid, t0=(cand[0] if cand else FIRST), delta={}, ok={}, dloc={}, okloc={}, dirs={}, ldirs={}, norm=np.full(L, np.nan), dropped=np.zeros(L, int), radial=np.full(C, np.nan), inside=np.full(C, np.nan), **{n: np.full((C, L), np.nan) for n in NAMES})
    hf = model.model(input_ids=ids.repeat(C, 1)).last_hidden_state; tf = time.time() - t0; h27 = st["h27"]; hc27 = h27[0]; n27 = hc27.norm(dim=-1); sl = slice(8, T - 1); tgt = ids[0, 9:T]; ar = torch.arange(len(tgt))
    pp = {n: np.full((C, MAXT), np.nan) for n in ("kl", "H", "ce", "top", "r", "rhoF", "Hcs", "kl_late", "kl_frozen")}; pp["kl_latef"] = np.full((len(LATEF), MAXT), np.nan); pp["H_latef"] = np.full((len(LATEF), MAXT), np.nan)
    pp["r"][:, :T] = (h27.norm(dim=-1) / n27).numpy(); pp["rhoF"][:, :T] = ((h27 - hc27).norm(dim=-1) / n27).numpy(); lp0 = None

    def score(x): return model.lm_head(x[:, sl]).double()                                 # logits at the scored positions, float64 from here on
    ent = lambda lq: -(lq.exp() * lq).sum(-1)
    for c0 in range(0, C, 6):
        lg = score(hf[c0:c0 + 6]); lq = torch.log_softmax(lg, -1)
        if lp0 is None: lp0 = lq[0].clone(); p0 = lp0.exp(); top0 = lp0.argmax(-1)
        cs = slice(c0, c0 + lq.shape[0]); pp["kl"][cs, sl] = (p0 * (lp0 - lq)).sum(-1).numpy(); pp["H"][cs, sl] = ent(lq).numpy(); pp["ce"][cs, sl] = (-lq[:, ar, tgt]).numpy(); pp["top"][cs, sl] = (lq.argmax(-1) == top0).double().numpy()
        pp["Hcs"][cs, sl] = ent(torch.log_softmax(lg * torch.tensor(pp["r"][cs, sl])[..., None], -1)).numpy(); del lg, lq
    u = unit(torch.randn(T, D, generator=G)); fin = model.model.norm
    for c in LATE:                                                                         # without the network: same final size in a random direction / the added vector untouched
        lq = torch.log_softmax(score(fin(torch.stack([hc27 + (h27[c] - hc27).norm(dim=-1, keepdim=True) * u, hc27 + st["delta"][c]]))), -1)
        pp["kl_late"][c, sl] = (p0 * (lp0 - lq[0])).sum(-1).numpy(); pp["kl_frozen"][c, sl] = (p0 * (lp0 - lq[1])).sum(-1).numpy()
    for j, fz in enumerate(LATEF):
        lq = torch.log_softmax(score(fin((hc27 + fz * n27[:, None] * u * valid[:, None])[None])), -1)[0]; pp["kl_latef"][j, sl] = (p0 * (lp0 - lq)).sum(-1).numpy(); pp["H_latef"][j, sl] = ent(lq).numpy()
    for n in NAMES: RES[n].append(st[n])
    for n in PP: PP[n].append(pp[n])
    for n in EX: EX[n].append(st[n])
    v = valid[sl].numpy(); meta.append(dict(text=k, source=src, tokens=int(T), t0=int(st["t0"]), valid=[int(x) for x in valid], clean_loss=float(np.nanmean(pp["ce"][0, sl][v])), clean_entropy=float(np.nanmean(pp["H"][0, sl][v]))))
    print(f"  text {k + 1}/{len(main)} {src} T={T}: forward {tf:.0f}s, total {time.time() - t0:.0f}s | clean loss {meta[-1]['clean_loss']:.2f} | null KL {np.nanmax(pp['kl'][1]):.1e} | "
          + " ".join(f"{CONDS[c]['name']} KL {np.nanmean(pp['kl'][c, sl][v]):.3f}" for c in FREE[:3]), flush=True)
    np.savez(f"{out}/ufo_damp2{TAG}.npz", **{n: np.array(RES[n]) for n in NAMES}, **{"pp_" + n: np.array(PP[n]) for n in PP}, **{n: np.array(EX[n]) for n in EX})
    json.dump(dict(conds=CONDS, meta=meta, latef=LATEF, late=LATE, stats={str(m): P.stats[m] for m in P.stats}, randinit=RANDINIT, topk=TOPK), open(f"{out}/ufo_damp2{TAG}_meta.json", "w"))
print(f"done ({time.time() - t0:.0f}s)")
