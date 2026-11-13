"""Session r1b_L_B (account B), part 1: the corpus L.R (F-neu thresholds, L.R-A and L.R-B generation with exact generation-time
capture of every population channel, packed spins, e, s, p_EOS, repeat flags; anchor information with the expected iota* (part-file
amendment); natural passages), L2a (EQMI growth class, G-L2a-LCE, G-MI, own-text deficit with the p_EOS truncation), L2b, L2c, and the
L5 generation (6 extra gating samples, own CoT with captures, sub-answers, t* pass)."""
import math, os, re
import numpy as np

import r1b_core as K
import r1b_l_engine as E
import r1b_l_stats as S
from r1b_l_a import refw, setup
from r1b_l_base import (COT_INSTR, F_NEU_SCENES, NAT_PROMPT, NEU_TEMPLATE, QA_TEMPLATE, SUB_TEMPLATE, cfg, channels, gating, load_pop,
                        loadpt, longdocs, lensout, pos_bin, rng, save, sha, spin_sets, split_rows, wj, wt_test)

FAMS = ("F-known", "F-unknown", "F-fab", "F-neu")
ANCH = FAMS[:3]


def lr_items(items, cf):
    fam = {"F-known": split_rows(items, "S_L", "CC")[:cf["n_fam"]], "F-unknown": split_rows(items, "S_L", "CI")[:cf["n_fam"]],
           "F-fab": split_rows(items, "NE")[:cf["n_fam"]]}
    out = {f: [dict(id=it["id"], text=QA_TEMPLATE.format(q=it["question"])) for it in v] for f, v in fam.items()}
    out["F-neu"] = [dict(id=f"neu{i}", text=NEU_TEMPLATE.format(s=sc)) for i, sc in enumerate(F_NEU_SCENES[:cf["n_fam"]])]
    return out


def spin_units(pop):
    u, off = [], {}
    for n, s in spin_sets(pop):
        off[n] = (len(u), len(u) + len(s)); u += list(s)
    return u, off


def rep_flags(tokens, win=256, thr=0.3):
    """4-gram repeat flag: share of 4-grams in the trailing 256 tokens that occurred earlier in that window > 0.3."""
    n = len(tokens); out = np.zeros(n, dtype=bool)
    grams = [tuple(tokens[i - 3:i + 1]) if i >= 3 else None for i in range(n)]
    for t in range(n):
        lo_ = max(3, t - win + 1)
        seen, rep, tot = set(), 0, 0
        for i in range(lo_, t + 1):
            g = grams[i]
            tot += 1; rep += g in seen; seen.add(g)
        out[t] = tot > 0 and rep / tot > thr
    return out


def _chain_rec(r, fam, item, T, key, kind, spins=None):
    d = dict(fam=fam, item=item, T=T, key=key, kind=kind, tokens=r["tokens"], n=r["n"], ended=r["ended"],
             a=r["a"].astype(np.float16) if "a" in r else None, a2=r["a2"].astype(np.float16) if "a2" in r else None,
             e=r["e"].astype(np.float32), s=r["s"].astype(np.float32), peos=r["peos"].astype(np.float32), rep=rep_flags(r["tokens"]))
    if spins is not None:
        d["spins"] = spins
    return d


def lr_fneu(ctx):
    """F-neu chains (64 scenes x T in {0.7, 1.0}, keys ('r1bL2', 'F-neu', i, T), <= 1,024 tokens, EOS allowed) with raw |z| of every
    spin unit; Q95 thresholds of a_P per (T, bin) and of |z_j| per (unit, T, bin) on odd chains; even chains check the 5 % share."""
    torch = K._torch()
    tok, model, cf, H, items = setup(ctx)
    rw = refw(ctx, model)
    pop = load_pop(ctx)
    ch = channels(pop); units, off = spin_units(pop)
    its = lr_items(items, cf)["F-neu"]
    recs = []
    with E.Probe(model, rw, chans=ch, spins=units) as pr:
        for T in (0.7, 1.0):
            for b0 in range(0, len(its), cf["batch"]):
                ib = its[b0:b0 + cf["batch"]]

                def run():
                    keys = [ctx.key("r1bL2", "F-neu", b0 + j, T) for j in range(len(ib))]
                    g = E.decode(model, tok, [K.chat_ids(tok, it["text"]) for it in ib], keys, T=T, max_new=cf["lr_new"], probe=pr,
                                 spin_fn=lambda st, z: z.cpu())
                    out = []
                    for j, r in enumerate(g):
                        z = torch.stack(r.pop("z")) if r.get("z") else torch.zeros(0, len(units), dtype=torch.float16)
                        out.append(dict(_chain_rec(r, "F-neu", ib[j]["id"], T, keys[j], "A"), zabs=z.half(), idx=b0 + j))
                    return out
                recs += ctx.cached(f"fneu_{T}_{b0}", run)
    nb = len(np.atleast_1d(pos_bin(np.arange(1, cf["lr_new"] + 1), cf["pos_scale"]))) and int(pos_bin(cf["lr_new"], cf["pos_scale"])) + 1
    thrA = np.full((2, nb), np.nan); thrS = torch.full((2, nb, len(units)), float("inf"), dtype=torch.float16)
    share_even = {}
    for ti, T in enumerate((0.7, 1.0)):
        odd = [r for r in recs if r["T"] == T and r["idx"] % 2 == 1]; even = [r for r in recs if r["T"] == T and r["idx"] % 2 == 0]
        for b in range(nb):
            av = np.concatenate([np.asarray(r["a"][:, 0], float)[pos_bin(np.arange(1, r["n"] + 1), cf["pos_scale"]) == b] for r in odd
                                 if r["n"]] or [np.zeros(0)])
            if av.size:
                thrA[ti, b] = np.quantile(av, 0.95)
            zs = [r["zabs"][torch.as_tensor(pos_bin(np.arange(1, r["n"] + 1), cf["pos_scale"]) == b)] for r in odd if r["n"]]
            zs = torch.cat(zs) if zs else torch.zeros(0)
            if zs.numel():
                thrS[ti, b] = torch.quantile(zs.float(), 0.95, dim=0).half() if zs.shape[0] > 1 else zs[0]
        ev = [np.asarray(r["a"][:, 0], float) > thrA[ti, np.minimum(pos_bin(np.arange(1, r["n"] + 1), cf["pos_scale"]), nb - 1)]
              for r in even if r["n"]]
        share_even[T] = float(np.mean(np.concatenate(ev))) if ev else float("nan")
    # fill bins without odd tokens by the nearest earlier bin
    for ti in range(2):
        for b in range(1, nb):
            if not np.isfinite(thrA[ti, b]):
                thrA[ti, b] = thrA[ti, b - 1]
            if torch.isinf(thrS[ti, b]).all():
                thrS[ti, b] = thrS[ti, b - 1]
    drift = any(abs(v - 0.05) > 0.01 for v in share_even.values() if np.isfinite(v))
    save(ctx, "l/lr_thr.pt", dict(thrA=thrA, thrS=thrS, units=units, off=off, nb=nb, share_even=share_even, drift=drift),
         config=dict(q=0.95, fit="odd F-neu chains", check="even F-neu chains"), labels=["binarisation drift"] if drift else [])
    for r in recs:
        z = r.pop("zabs")
        ti = 0 if r["T"] == 0.7 else 1
        bins = np.minimum(pos_bin(np.arange(1, r["n"] + 1), cf["pos_scale"]), nb - 1)
        sp = (z.float() > thrS[ti][torch.as_tensor(bins)].float()) if r["n"] else torch.zeros(0, len(units), dtype=torch.bool)
        r["spins"] = np.packbits(sp.numpy(), axis=1)
        r["zP"] = z[:, :len(pop["P"])].half()
    save(ctx, "l/lr/fneu.pt", dict(chains=recs), config=dict(T=(0.7, 1.0), new=cf["lr_new"], prompts_sha256=sha([it["text"] for it in its])))


def _spin_fn(thrS, Tidx, nb, cf, nP):
    torch = K._torch()

    def fn(step, z):
        b = min(int(pos_bin(step, cf["pos_scale"])), nb - 1)
        t = thrS[torch.as_tensor(Tidx), b].float().to(z.device)
        return (np.packbits((z.float() > t).cpu().numpy(), axis=1), z[:, :nP].half().cpu())
    return fn


def lr_gen(ctx):
    """L.R-A anchored families (S_L CC, S_L CI, NE: 64 items x T in {0.7, 1.0}, keys ('r1bL2', family, i, T), <= 1,024 tokens) and
    L.R-B (the first 32 items of each anchored family x 8 keys ('r1bL2b', family, i, k) at T 1.0, <= 512 tokens), all captures exact."""
    tok, model, cf, H, items = setup(ctx)
    rw = refw(ctx, model)
    pop = load_pop(ctx)
    th = loadpt(ctx.path("l/lr_thr.pt"))
    ch = channels(pop)
    its = lr_items(items, cf)
    jobs = [("A", f, T, j, ctx.key("r1bL2", f, j, T), cf["lr_new"]) for f in ANCH for T in (0.7, 1.0) for j in range(len(its[f]))]
    jobs += [("B", f, 1.0, j, ctx.key("r1bL2b", f, j, k), cf["lrb_new"]) for f in ANCH for j in range(min(cf["lrb_items"], len(its[f])))
             for k in range(cf["lrb_keys"])]
    groups = {}
    for jb in jobs:
        groups.setdefault((jb[0], jb[2], jb[5]), []).append(jb)
    recs = []
    with E.Probe(model, rw, chans=ch, spins=th["units"]) as pr:
        for (kind, T, new), jl in sorted(groups.items()):
            for b0 in range(0, len(jl), cf["batch"]):
                jb = jl[b0:b0 + cf["batch"]]

                def run():
                    fn = _spin_fn(th["thrS"], [0 if T == 0.7 else 1] * len(jb), th["nb"], cf, len(pop["P"]))
                    g = E.decode(model, tok, [K.chat_ids(tok, its[f][j]["text"]) for _, f, _, j, _, _ in jb], [x[4] for x in jb], T=T,
                                 max_new=new, probe=pr, spin_fn=fn)
                    out = []
                    for x, r in zip(jb, g):
                        zz = r.pop("z", [])
                        rec = _chain_rec(r, x[1], its[x[1]][x[3]]["id"], T, x[4], kind,
                                         spins=np.stack([a for a, _ in zz]) if zz else np.zeros((0, (len(th["units"]) + 7) // 8), np.uint8))
                        rec["zP"] = K._torch().stack([b for _, b in zz]) if zz else K._torch().zeros(0, len(pop["P"]))
                        rec["idx"] = x[3]
                        out.append(rec)
                    return out
                recs += ctx.cached(f"lr_{kind}_{T}_{b0}", run)
    save(ctx, "l/lr/gen.pt", dict(chains=recs), config=dict(new_A=cf["lr_new"], new_B=cf["lrb_new"], keys_B=cf["lrb_keys"],
                                                            prompts_sha256=sha({f: [i["text"] for i in v] for f, v in its.items()})))


def corpus(ctx, with_tf=False):
    """All L.R chains (F-neu first), thresholds; optional anchor information and natural passages."""
    p = (lambda r: ctx.path(r)) if ctx.session == "r1b_L_B" else (lambda r: ctx.input("r1b_L_B", r))
    ch = loadpt(p("l/lr/fneu.pt"))["chains"] + loadpt(p("l/lr/gen.pt"))["chains"]
    th = loadpt(p("l/lr_thr.pt"))
    tf = loadpt(p("l/lr_tf.pt")) if with_tf else None
    return ch, th, tf


def lr_tf(ctx):
    """Anchor information per anchored L.R-A chain: teacher-forced log-probs and KL under A (own prompt), A' and A'' (prompts of two
    other items of the same family, derangements seed 'r1bL2perm'); natural passages (SO-WT103 test) read as the reply to 'Write a
    detailed text.' with all channel captures."""
    tok, model, cf, H, items = setup(ctx)
    rw = refw(ctx, model)
    pop = load_pop(ctx)
    its = lr_items(items, cf)
    chains = [c for c in loadpt(ctx.path("l/lr/gen.pt"))["chains"] if c["kind"] == "A"]
    perm = {}
    for f in ANCH:
        n = len(its[f])
        g = rng("r1bL2perm", f)
        p1 = S.derange(n, g); p2 = S.derange(n, g)
        while n > 2 and (p2 == p1).any():
            p2 = S.derange(n, g)
        perm[f] = (p1, p2)
    groups = []
    for c in chains:
        f, j = c["fam"], c["idx"]
        A = K.chat_ids(tok, its[f][j]["text"]); A1 = K.chat_ids(tok, its[f][perm[f][0][j]]["text"]); A2 = K.chat_ids(tok, its[f][perm[f][1][j]]["text"])
        groups.append(([A, A1, A2], c["tokens"]))
    out = []
    for b0 in range(0, len(groups), cf["batch"]):
        out += ctx.cached(f"anchor_{b0}", lambda: E.ctx_logprobs(model, tok, groups[b0:b0 + cf["batch"]], batch=3 * max(1, cf["batch"] // 3)))
    ps, ml = wt_test(ctx, tok, cf)
    pr_ids = K.chat_ids(tok, NAT_PROMPT)
    nat = []
    with E.Probe(model, rw, chans=channels(pop)) as pr:
        for b0 in range(0, len(ps), cf["batch"]):
            nat += ctx.cached(f"nat_{b0}", lambda: [dict(a=x["a"].astype(np.float16), e=x["e"].astype(np.float32), s=x["s"].astype(np.float32),
                                                        n=x["n"]) for x in E.tf_pass(model, tok, [pr_ids + p for p in ps[b0:b0 + cf["batch"]]],
                                                                                     [len(pr_ids)] * len(ps[b0:b0 + cf["batch"]]), probe=pr,
                                                                                     batch=cf["batch"])])
    save(ctx, "l/lr_tf.pt", dict(anchor=[dict(fam=c["fam"], item=c["item"], T=c["T"], lp=o["lp"].astype(np.float32), kl=o["kl"].astype(np.float32))
                                         for c, o in zip(chains, out)], nat=nat, perm={f: [p[0].tolist(), p[1].tolist()] for f, p in perm.items()}),
         config=dict(perm_seed="r1bL2perm", nat=len(ps)), labels=ml)


# ==================================================================================================================================
# L2a EQMI
# ==================================================================================================================================
def _l2a_grid(ctx, cf):
    g = list(cf["eqmi_L"])
    dec = (ctx.freeze.get("decisions") or {}).get("OD-M14")
    if dec == "grid_2^14":
        g = [L for L in g if L <= 2 ** 14]
    return g


def _ihat_doc(model, x_true_lp, doc, other, L, chunk):
    """Sum over j in Y_L of log q(x_j | x_<j) - log q(x_j | X'_L, x_(L/2..j-1)) for one document (nats)."""
    seq = list(other[:L // 2]) + list(doc[L // 2:L])
    lf = E.lce_logprobs(model, seq, chunk=chunk, need_from=L // 2)
    return float(np.nansum(x_true_lp[L // 2:L] - lf[L // 2:L]))


def l2a_lce(ctx):
    """G-L2a-LCE: chunked and cached log-probs against a full pass on 2 documents at 16k and 32k (relative difference < 1e-4); memory."""
    torch = K._torch()
    tok, model, cf, H, items = setup(ctx)
    docs, ml = longdocs(ctx, tok, cf)
    res = {}
    for Ln in ((2 ** 14, 2 ** 15) if not ctx.mock else (cf["long_len"] // 2, cf["long_len"])):
        rel = []
        for d in docs[:2]:
            a = E.lce_logprobs(model, d[:Ln], chunk=cf["eqmi_chunk"]); b = E.full_logprobs(model, d[:Ln])
            m = np.isfinite(a) & np.isfinite(b)
            rel.append(float(np.max(np.abs(a[m] - b[m])) / max(np.mean(np.abs(b[m])), 1e-12)))
        mem = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
        res[str(Ln)] = dict(rel=max(rel), ok=max(rel) < 1e-4, peak_gb=mem)
    ok = all(v["ok"] for v in res.values())
    wj(ctx, "l/l2a_lce.json", dict(lengths=res, ok=ok, kernel="memory-efficient SDPA asserted" if not ctx.mock else "cpu: not asserted"),
       config=dict(tol=1e-4, chunk=cf["eqmi_chunk"]), labels=ml + ([] if ok else ["G-L2a-LCE failed"]))


def l2a_main(ctx):
    """True passes (one causal pass per document) and foreign passes (X'_L = the first L/2 tokens of another document, fixed
    derangement) for L in the grid; G-MI copy (4 documents, Y := X, L <= 2^12) and shuffle controls (8 documents, sentence-shuffled,
    seed 'r1bL-SHUF', L <= 2^13)."""
    tok, model, cf, H, items = setup(ctx)
    docs, ml = longdocs(ctx, tok, cf)
    grid = _l2a_grid(ctx, cf)
    perm = S.derange(len(docs), rng("r1bL2a", "derange"))
    I = np.zeros((len(docs), len(grid)))
    for d, doc in enumerate(docs):
        def run():
            lt = E.lce_logprobs(model, doc, chunk=cf["eqmi_chunk"])
            return np.array([_ihat_doc(model, lt, doc, docs[perm[d]], L, cf["eqmi_chunk"]) for L in grid])
        I[d] = ctx.cached(f"doc{d}", run)
    big = [L for L in grid if L <= (2 ** 12 if not ctx.mock else grid[-2])]
    copy = np.zeros((cf["gmi_copy"], len(big)))
    for d in range(min(cf["gmi_copy"], len(docs))):
        def runc():
            out = []
            for L in big:
                X = docs[d][:L // 2]; cp = X + X
                lt = E.lce_logprobs(model, cp, chunk=cf["eqmi_chunk"])
                out.append(_ihat_doc(model, lt, cp, docs[perm[d]], L, cf["eqmi_chunk"]))
            return np.array(out)
        copy[d] = ctx.cached(f"copy{d}", runc)
    Ls = [L for L in grid if L <= (2 ** 13 if not ctx.mock else grid[-1])][-1]
    shuf, base = [], []
    sdocs = []
    for d in range(min(cf["gmi_shuf"], len(docs))):
        txt = tok.decode(docs[d])
        sents = re.split(r"(?<=[.!?])\s+", txt)
        g = rng("r1bL-SHUF", d); g.shuffle(sents)
        sdocs.append(K.enc(tok, " ".join(sents))[:len(docs[d])])
    for d in range(len(sdocs)):
        def runs_():
            lt = E.lce_logprobs(model, sdocs[d], chunk=cf["eqmi_chunk"])
            return _ihat_doc(model, lt, sdocs[d], sdocs[(d + 1) % len(sdocs)], Ls, cf["eqmi_chunk"])
        shuf.append(ctx.cached(f"shuf{d}", runs_))
        base.append(I[d, grid.index(Ls)])
    save(ctx, "l/l2a_main.pt", dict(grid=grid, I=I, copy=copy, copy_grid=big, shuf=np.array(shuf), shuf_base=np.array(base), shuf_L=Ls,
                                    perm=perm.tolist()), config=dict(grid=grid, docs=len(docs), chunk=cf["eqmi_chunk"]), labels=ml)


def l2a_own(ctx):
    """Own text: 24 raw continuations of 512-token seeds (T 1.0, 4,096 new tokens, keys ('r1bL2s', d), EOS masked, p_EOS logged);
    I-hat_self(L) and I-hat_nat|seed(L) for L in {2^7, ..., 2^12}, both conditioned on the natural seed; truncation where the cumulative
    end probability first exceeds 0.5 (part-file amendment)."""
    tok, model, cf, H, items = setup(ctx)
    docs, ml = longdocs(ctx, tok, cf)
    sd = cf["own_seed"]
    seeds = [d[:sd] for d in docs]
    gen = []
    for b0 in range(0, len(seeds), cf["batch"]):
        gen += ctx.cached(f"own_{b0}", lambda: [dict(tokens=r["tokens"], peos=r["peos"]) for r in
                                                E.decode(model, tok, seeds[b0:b0 + cf["batch"]],
                                                         [ctx.key("r1bL2s", d) for d in range(b0, min(b0 + cf["batch"], len(seeds)))],
                                                         T=1.0, max_new=cf["own_new"], mask_eos=True, stop=False)])
    perm = S.derange(len(docs), rng("r1bL2a", "own"))
    Ls = cf["own_L"]
    Is, In = np.zeros((len(docs), len(Ls))), np.zeros((len(docs), len(Ls)))
    for d in range(len(docs)):
        def run():
            self_seq = seeds[d] + gen[d]["tokens"]; nat_seq = docs[d][:sd + cf["own_new"]]
            ls = E.lce_logprobs(model, self_seq, chunk=cf["eqmi_chunk"]); ln = E.lce_logprobs(model, nat_seq, chunk=cf["eqmi_chunk"])
            a, b = [], []
            for L in Ls:
                Y = slice(sd + L // 2, sd + L)
                fs = seeds[d] + gen[perm[d]]["tokens"][:L // 2] + gen[d]["tokens"][L // 2:L]
                fn = seeds[d] + docs[perm[d]][sd:sd + L // 2] + docs[d][sd + L // 2:sd + L]
                lfs = E.lce_logprobs(model, fs, chunk=cf["eqmi_chunk"], need_from=sd + L // 2)
                lfn = E.lce_logprobs(model, fn, chunk=cf["eqmi_chunk"], need_from=sd + L // 2)
                a.append(float(np.nansum(ls[Y] - lfs[Y]))); b.append(float(np.nansum(ln[Y] - lfn[Y])))
            return np.array(a), np.array(b)
        Is[d], In[d] = ctx.cached(f"ownI{d}", run)
    tend = []
    for g_ in gen:
        cum = 1 - np.cumprod(1 - np.asarray(g_["peos"], float))
        k = np.flatnonzero(cum > 0.5)
        tend.append(int(k[0]) if k.size else len(g_["tokens"]))
    save(ctx, "l/l2a_own.pt", dict(L=Ls, I_self=Is, I_nat=In, t_end=tend, max_peos=[float(np.max(g_["peos"])) if len(g_["peos"]) else 0.0 for g_ in gen]),
         config=dict(seed=sd, new=cf["own_new"], T=1.0, eos="masked; truncation at cumulative end probability 0.5"), labels=ml)


def l2a_stats(ctx):
    """Growth class by BIC per document-bootstrap replicate (label if chosen in >= 80 %), L_sat, G-MI, R(2^12) and the ln R slope."""
    tok, model, cf, H, items = setup(ctx)
    m = loadpt(ctx.path("l/l2a_main.pt")); o = loadpt(ctx.path("l/l2a_own.pt")); lce = K.load_json(ctx.path("l/l2a_lce.json"))
    grid = np.array(m["grid"], float); I = m["I"]
    W = S.boot_counts(I.shape[0], cf["nboot"], seed=211)
    Ib = (W @ I) / W.sum(1, keepdims=True)
    cls = [S.growth_class(grid, Ib[b])[0] for b in range(len(Ib))]
    c0, fits = S.growth_class(grid, I.mean(0))
    share = {k: cls.count(k) / len(cls) for k in ("saturating", "logarithmic", "power")}
    top = max(share, key=share.get)
    bb = []
    for b in range(len(Ib)):
        x = np.log(grid); bb.append(float(np.cov(x, Ib[b], bias=True)[0, 1] / x.var()))
    betas = [S.growth_fits(grid, Ib[b])["power"]["beta"] for b in range(min(len(Ib), 400))]
    se = Ib.std(0)
    Lsat = None
    for i in range(len(grid) - 1):
        if all(I.mean(0)[j + 1] - I.mean(0)[j] < max(2 * se[j], 0.05 * I.mean(0)[-1]) for j in range(i, len(grid) - 1)):
            Lsat = int(grid[i]); break
    copy_ok = bool(np.all(m["copy"].mean(0) >= 5 * I.mean(0)[:m["copy"].shape[1]])) if m["copy"].size else False
    dsh = m["shuf"] - m["shuf_base"]
    Wd = S.boot_counts(len(dsh), cf["nboot"], seed=212)
    shuf = S.summ(float(dsh.mean()), (Wd @ dsh) / Wd.sum(1))
    gmi_ok = copy_ok and isinstance(shuf["hi"], float) and shuf["hi"] < 0 and float(m["shuf"].mean()) < float(m["shuf_base"].mean())
    # own text
    Ls = np.array(o["L"]); tend = np.array(o["t_end"]); mx = np.array(o["max_peos"])
    Wo = S.boot_counts(len(tend), cf["nboot"], seed=213)

    def ratio(mask):
        Rpt, Rb = [], []
        for j, L in enumerate(Ls):
            k = mask & (tend >= L)
            if k.sum() == 0:
                Rpt.append(np.nan); Rb.append(np.full(len(Wo), np.nan)); continue
            Rpt.append(o["I_self"][k, j].mean() / o["I_nat"][k, j].mean() if o["I_nat"][k, j].mean() else np.nan)
            Rb.append((Wo[:, k] @ o["I_self"][k, j]) / np.where((Wo[:, k] @ o["I_nat"][k, j]) != 0, Wo[:, k] @ o["I_nat"][k, j], np.nan))
        return np.array(Rpt), np.stack(Rb, 1)
    Rpt, Rb = ratio(np.ones(len(tend), bool))
    sel = (Ls >= Ls[min(1, len(Ls) - 1)])
    xl = np.log(Ls[sel])

    def slope(r):
        y = np.log(np.where(r[sel] > 0, r[sel], np.nan)); ok = np.isfinite(y)
        return float(np.cov(xl[ok], y[ok], bias=True)[0, 1] / xl[ok].var()) if ok.sum() > 1 else np.nan
    R_top = S.summ(Rpt[-1], Rb[:, -1])
    sl = S.summ(slope(Rpt), [slope(Rb[b]) for b in range(len(Rb))])
    Rlo, Rblo = ratio(mx < 0.05)
    low = S.summ(Rlo[-1], Rblo[:, -1]) if np.isfinite(Rlo[-1]) else None
    st = dict(grid=grid.tolist(), I=[S.summ(I.mean(0)[j], Ib[:, j]) for j in range(len(grid))],
              growth={"class_point": c0, "class": top, "share": share[top], "shares": share, "b": S.summ(fits["logarithmic"]["b"], bb),
                      "beta": S.summ(fits["power"]["beta"], betas), "fits": fits}, L_sat=Lsat, gmi_ok=gmi_ok, copy_ok=copy_ok, shuffle=shuf,
              lce=lce["ok"], R=[S.summ(Rpt[j], Rb[:, j]) for j in range(len(Ls))], R_top=R_top, lnR_slope=sl, R_top_low_eos=low,
              deficit_low_eos=bool(low is not None and isinstance(low["hi"], float) and low["hi"] < 0.87), n_truncated=int((tend < Ls[-1]).sum()))
    wj(ctx, "l/l2a.json", st, config=dict(nboot=cf["nboot"], grid=grid.tolist()), labels=([] if lce["ok"] else ["G-L2a-LCE failed"]) +
       ([] if gmi_ok else ["G-MI failed"]))


# ==================================================================================================================================
# L2b length against information; L2c cross-map
# ==================================================================================================================================
def _chan_names(pop):
    return [n for n, _ in channels(pop)]


def windows(c, cf, tf=None):
    """32-token windows at continuation positions >= 64 while at risk: rows [n_w, ...] of window means."""
    n, w0, w = c["n"], int(cf["win0"]), int(cf["win"])
    out = []
    for s in range(w0, n - w + 1, w):
        sl = slice(s, s + w)
        out.append((sl, math.log(s + w / 2 + 1)))
    return out


def l2b(ctx):
    """Within-chain Frisch-Waugh slopes (cov / var, Gram-Schmidt) of every channel on log position, with and without partialling the
    expected anchor information iota* and the entropy (part-file amendment: iota* primary, realised iota descriptive); item-clustered
    bootstrap; natural-position slope; repetition exclusion; selection-robustness on the boot sets; descriptives."""
    tok, model, cf, H, items = setup(ctx)
    pop = load_pop(ctx)
    ch, th, tf = corpus(ctx, with_tf=True)
    names = _chan_names(pop); C = len(names)
    anch = {(a["fam"], a["item"], a["T"]): a for a in tf["anchor"]}
    rows, items_of = [], []
    for c in ch:
        if c["kind"] != "A" or c["fam"] not in ANCH:
            continue
        a = anch.get((c["fam"], c["item"], c["T"]))
        if a is None:
            continue
        A = np.asarray(c["a"], float)
        for sl, lp in windows(c, cf):
            rows.append(dict(ch=(c["fam"], c["item"], c["T"]), item=c["item"], fam=c["fam"], T=c["T"], lp=lp, a=A[sl].mean(0),
                             iotas=float(a["kl"][0][sl].mean()), iota=float((a["lp"][0][sl] - a["lp"][1][sl]).mean()),
                             e=float(c["e"][sl].mean()), rep=float(c["rep"][sl].mean())))
    if not rows:
        raise K.NotRun("no windows (chains shorter than the first window)")
    uit = sorted({r["item"] for r in rows}); iix = {u: i for i, u in enumerate(uit)}
    W = S.boot_counts(len(uit), cf["nboot"], seed=221)

    def slopes(rr):
        """Per channel: b0, bie, biota (standardised by the SD of window means), per item cross-products -> bootstrap replicates."""
        chs = sorted({r["ch"] for r in rr})
        out_pt, out_b = {}, {}
        A = np.stack([r["a"] for r in rr]); sdA = A.std(0) + 1e-12
        V = np.stack([[r["lp"], r["iotas"], r["e"]] for r in rr])
        it = np.array([iix[r["item"]] for r in rr]); cid = {c_: i for i, c_ in enumerate(chs)}
        cc = np.array([cid[r["ch"]] for r in rr])
        Ad, Vd = A.copy(), V.copy()
        for k in range(len(chs)):
            m = cc == k
            Ad[m] -= Ad[m].mean(0); Vd[m] -= Vd[m].mean(0)
        sdE = V[:, 2].std() + 1e-12
        Mi = np.zeros((len(uit), C + 3, C + 3))
        X = np.concatenate([Ad / sdA, Vd], 1)
        for i in range(len(uit)):
            m = it == i
            if m.any():
                Mi[i] = X[m].T @ X[m]
        lpj, ioj, ej = C, C + 1, C + 2

        def fromM(M):
            b0 = np.array([S.fw_slope(M, j, lpj) for j in range(C)])
            bie = np.array([S.fw_slope(M, j, lpj, partial=(ioj, ej)) for j in range(C)])
            bio = np.array([S.fw_slope(M, j, ioj, partial=(lpj, ej)) for j in range(C)])
            bE0 = S.fw_slope(M, ej, lpj) / sdE
            return b0, bie, bio, bE0
        pt_ = fromM(Mi.sum(0))
        reps = [fromM(np.tensordot(W[b], Mi, 1)) for b in range(len(W))]
        return pt_, reps
    pt_, reps = slopes(rows)
    ci_ = [i for i, n in enumerate(names) if n.startswith("C")]
    j = {n: i for i, n in enumerate(names)}

    def stat(k, ix, minus=None, margin=None):
        f = lambda t: t[k][ix] - (np.mean(t[k][minus]) if isinstance(minus, list) else (t[k][minus] if minus is not None else 0.0))
        return S.summ(f(pt_), [f(r) for r in reps], margin)
    P = j["P"]
    st = dict(n_windows=len(rows), n_items=len(uit))
    st["b0"] = stat(0, P, margin=0.10); st["bie"] = stat(1, P, margin=0.10); st["biota"] = stat(2, P)
    st["db0"] = stat(0, P, ci_, 0.10); st["dbie"] = stat(1, P, ci_)
    for nm in ("SEL", "UNC"):
        if nm in j:
            st[f"b0_minus_{nm}"] = stat(0, P, j[nm]); st[f"bie_minus_{nm}"] = stat(1, P, j[nm])
    st["b0_minus_E"] = S.summ(pt_[0][P] - pt_[3], [r[0][P] - r[3] for r in reps])
    from r1b_core import holm
    hp = [st["dbie"]["p_gt0"]] + [st.get(f"bie_minus_{n}", {}).get("p_gt0", 1.0) for n in ("SEL", "UNC")]
    adj, rej = holm(hp)
    st["holm_beyond"] = bool(rej[0]); st["holm"] = adj
    # natural-position slope
    nat = tf["nat"]
    nr = []
    for k, x in enumerate(nat):
        A = np.asarray(x["a"], float)
        for sl, lp in windows(dict(n=x["n"]), cf):
            nr.append((k, lp, A[sl, P].mean()))
    if nr:
        kk = np.array([r[0] for r in nr]); lp = np.array([r[1] for r in nr]); av = np.array([r[2] for r in nr])
        sdn = np.std([r["a"][P] for r in rows]) + 1e-12
        lpd, avd = lp.copy(), av.copy()
        for k in np.unique(kk):
            m = kk == k; lpd[m] -= lpd[m].mean(); avd[m] -= avd[m].mean()
        Wn = S.boot_counts(len(nat), cf["nboot"], seed=222)
        sxy = np.array([(lpd[kk == k] * avd[kk == k]).sum() for k in range(len(nat))]); sxx = np.array([(lpd[kk == k] ** 2).sum() for k in range(len(nat))])
        bn = (sxy.sum() / sxx.sum()) / sdn; bnb = (Wn @ sxy) / np.maximum(Wn @ sxx, 1e-12) / sdn
        st["b0_nat"] = S.summ(bn, bnb)
        st["b0_minus_nat"] = S.summ(pt_[0][P] - bn, np.array([r[0][P] for r in reps]) - bnb)
    else:
        st["b0_minus_nat"] = S.summ(float("nan"), [np.nan])
    import r1b_l_rules as RU
    v_all = RU.rule_L2b(st)["verdict"]
    passing = "bie" if v_all == "H rises beyond information loss" else "b0"
    st["not_beyond_unc"] = bool(not RU.LB(st.get(f"{passing}_minus_UNC", {"lo": float("nan")})) and
                                (passing == "bie" or not RU.LB(st["b0_minus_E"])))
    # repetition exclusion
    rr = [r for r in rows if r["rep"] <= 0.3]
    if rr and len(rr) < len(rows):
        p2, r2 = slopes(rr)
        st2 = dict(st)
        st2.update(b0=S.summ(p2[0][P], [r[0][P] for r in r2], 0.10), bie=S.summ(p2[1][P], [r[1][P] for r in r2], 0.10),
                   biota=S.summ(p2[2][P], [r[2][P] for r in r2]), dbie=S.summ(p2[1][P] - np.mean(p2[1][ci_]), [r[1][P] - np.mean(r[1][ci_]) for r in r2]),
                   db0=S.summ(p2[0][P] - np.mean(p2[0][ci_]), [r[0][P] - np.mean(r[0][ci_]) for r in r2], 0.10))
        st["rep_changes"] = RU.rule_L2b(st2)["verdict"] != v_all
    else:
        st["rep_changes"] = False
    bix = [i for i, n in enumerate(names) if n.startswith("B")]
    if len(bix) >= 10 and v_all in ("H rises beyond information loss", "length effect on H is information-limited"):
        crit = (lambda i: pt_[1][i] >= 0.10 and pt_[1][i] - np.mean(pt_[1][ci_]) > 0) if passing == "bie" else (lambda i: pt_[0][i] > 0)
        st["sel_robust"] = sum(bool(crit(i)) for i in bix) >= 16 * len(bix) / 20
    else:
        st["sel_robust"] = None
    st["verdict_preview"] = v_all
    # descriptives: occupancy by bin (excess over F-neu), iota profiles, k*_free
    nb = th["nb"]
    occ = {}
    for f in FAMS:
        for T in (0.7, 1.0):
            bs = [np.minimum(pos_bin(np.arange(1, c["n"] + 1), cf["pos_scale"]), nb - 1) for c in ch if c["fam"] == f and c["T"] == T and c["kind"] == "A"]
            bb = [np.asarray(c["a"][:, 0], float) > th["thrA"][0 if T == 0.7 else 1][np.minimum(pos_bin(np.arange(1, c["n"] + 1), cf["pos_scale"]), nb - 1)]
                  for c in ch if c["fam"] == f and c["T"] == T and c["kind"] == "A"]
            if bb:
                B_ = np.concatenate(bb); Bn = np.concatenate(bs)
                occ[f"{f}|{T}"] = [float(B_[Bn == k].mean()) if (Bn == k).any() else None for k in range(nb)]
    st["occupancy"] = occ
    kfree, hentry = [], []
    for a, c in ((anch[(c["fam"], c["item"], c["T"])], c) for c in ch if c["kind"] == "A" and (c["fam"], c["item"], c["T"]) in anch):
        io = a["lp"][0] - a["lp"][1]; i0 = a["lp"][2] - a["lp"][1]
        rm = np.convolve(io, np.ones(8) / 8, "valid"); r0 = np.convolve(i0, np.ones(8) / 8, "valid")
        q95 = np.quantile(r0, 0.95) if r0.size else np.inf
        below = rm < q95
        k = next((t for t in range(len(below) - 8) if below[t:t + 8].all()), None)
        bP = np.asarray(c["a"][:, 0], float) > th["thrA"][0 if c["T"] == 0.7 else 1][np.minimum(pos_bin(np.arange(1, c["n"] + 1), cf["pos_scale"]), nb - 1)]
        kfree.append(k if k is not None else np.nan); hentry.append(float(np.argmax(bP)) if bP.any() else np.nan)
    st["k_free"] = K.dist(kfree); st["spearman_kfree_hentry"] = S.spearman(kfree, hentry)
    st["iota_profile"] = {f: K.dist([r["iotas"] for r in rows if r["fam"] == f]) for f in ANCH}
    wj(ctx, "l/l2b.json", st, config=dict(win0=cf["win0"], win=cf["win"], nboot=cf["nboot"], iota="expected anchor information KL(A || A')"))


def l2c(ctx):
    """Simplex cross-map of the output block B = (s_t, e_t) onto a_P (and every control channel) on L.R-A chains, positions 64-1,023;
    E by simplex self-forecast on the calibration third (chain index = 0 mod 3); libraries {32, 128, 512, 896} x 20; cross-chain null."""
    tok, model, cf, H, items = setup(ctx)
    pop = load_pop(ctx)
    ch, th, _ = corpus(ctx)
    names = _chan_names(pop)
    lo_, hi_ = cf["xmap_pos"]
    A = [c for c in ch if c["kind"] == "A" and c["n"] > lo_ + max(cf["xmap_libs"]) // 4]
    cal = [c for i, c in enumerate(A) if i % 3 == 0]; ev = [c for i, c in enumerate(A) if i % 3 != 0]
    Eb = S.simplex_E([np.stack([c["s"][lo_:hi_], c["e"][lo_:hi_]], 1) for c in cal], cf["xmap_E"])
    g = rng("r1bL2c", "lib")
    by_ft = {}
    for c in ev:
        by_ft.setdefault((c["fam"], c["T"]), []).append(c)
    res = {L: [] for L in cf["xmap_libs"]}; cross = {L: [] for L in cf["xmap_libs"]}; items_ = []
    for k, c in enumerate(ev):
        B_ = np.stack([c["s"][lo_:min(hi_, c["n"])], c["e"][lo_:min(hi_, c["n"])]], 1)
        Mb = S.embed(B_, Eb)
        tgt = np.asarray(c["a"], float)[lo_ + Eb - 1:lo_ + Eb - 1 + len(Mb)]
        peers = [p for p in by_ft[(c["fam"], c["T"])] if p is not c and p["n"] >= c["n"]]
        other = np.asarray(peers[k % len(peers)]["a"], float)[lo_ + Eb - 1:lo_ + Eb - 1 + len(Mb)] if peers else None
        items_.append(c["item"])
        for L in cf["xmap_libs"]:
            rs, rc = [], []
            for _ in range(cf["xmap_draws"]):
                Lm = min(L, len(Mb) - 2)
                lib = np.sort(g.choice(len(Mb), Lm, replace=False)); pred = np.arange(len(Mb))
                nb_, w = S.simplex_nn(Mb, lib, pred, Eb, Eb)
                rs.append(S.xmap_rho(tgt, nb_, w, pred))
                if other is not None:
                    rc.append(S.xmap_rho(other[:, :1], nb_, w, pred)[0])
            res[L].append(np.nanmean(rs, 0)); cross[L].append(np.nanmean(rc) if rc else np.nan)
    Ls = cf["xmap_libs"]; top, bot = Ls[-1], Ls[0]
    R = {L: np.stack(res[L]) for L in Ls}
    uit = sorted(set(items_)); it = np.array([uit.index(i) for i in items_])
    W = S.boot_counts(len(uit), cf["nboot"], seed=231)

    def cm(v):
        s_ = np.array([np.nanmean(v[it == i]) if (it == i).any() else 0.0 for i in range(len(uit))])
        n_ = np.array([(it == i).sum() for i in range(len(uit))], float)
        return float(np.nanmean(v)), (W @ (s_ * n_)) / np.maximum(W @ n_, 1e-12)
    P = names.index("P")
    a1, b1 = cm(R[top][:, P] - R[bot][:, P]); a2, b2 = cm(R[top][:, P] - np.array(cross[top]))
    ctrl = [names.index(n) for n in names if n.startswith("C")]
    rhoP = float(np.nanmean(R[top][:, P])); rc_ = [float(np.nanmean(R[top][:, i])) for i in ctrl]
    st = dict(E=Eb, rho={L: [float(np.nanmean(R[L][:, i])) for i in range(len(names))] for L in Ls}, channels=names,
              conv_lib=S.summ(a1, b1), conv_cross=S.summ(a2, b2, margin=0.06), rank_p=K.p_rank(rhoP, rc_), rho_P_top=rhoP,
              n_chains=len(ev))
    wj(ctx, "l/l2c.json", st, config=dict(libs=Ls, draws=cf["xmap_draws"], pos=cf["xmap_pos"], Es=cf["xmap_E"]))


# ==================================================================================================================================
# L5 generation
# ==================================================================================================================================
def l5_items(ctx, gate, recs, cf):
    """The 406 items with lens records, then the next gated A-lens pool items in pool order (A-cal excluded), up to 600."""
    withrec = []
    seen = set()
    for r in recs:
        if r.get("item") in gate and r["item"] not in seen:
            seen.add(r["item"]); withrec.append(r["item"])
    rest = [i for i, it in gate.items() if i not in seen and not it.get("a_cal")]
    return (withrec + rest)[:cf["l5_items"]]


def l5_gen(ctx):
    """6 extra gating samples (T 0.7, keys ('r1bL5', 'GATE', i, s), s = 1..6); own CoT (greedy + 4 at T 0.7, <= 96 tokens, keys
    ('r1bL5', 'COT', i, s)) with captures of every channel, e and s; isolated sub-answers (greedy + 4, <= 12 tokens, keys
    ('r1bL5', 'SUB', i, k, s)); the t* pass (last prompt token, unmodified teacher-forced read) for every channel."""
    tok, model, cf, H, items = setup(ctx)
    rw = refw(ctx, model)
    pop = load_pop(ctx)
    gate, gl = gating(ctx)
    recs, rl = lensout(ctx, gate)
    ids = l5_items(ctx, gate, recs, cf)
    out = {}
    with E.Probe(model, rw, chans=channels(pop)) as pr:
        for b0 in range(0, len(ids), cf["batch"]):
            ib = ids[b0:b0 + cf["batch"]]

            def run():
                res = {i: {} for i in ib}
                P_ = [K.chat_ids(tok, gate[i]["prompt"]) for i in ib]
                rows = [(i, s) for i in ib for s in range(1, 7)]
                g = E.decode(model, tok, [P_[ib.index(i)] for i, s in rows], [ctx.key("r1bL5", "GATE", i, s) for i, s in rows], T=0.7,
                             max_new=cf["l5_sub_new"])
                for (i, s), r in zip(rows, g):
                    res[i].setdefault("gate", []).append(tok.decode(r["tokens"]))
                cot_p = {i: K.chat_ids(tok, re.sub(r"Reply with the number only\.?|Reply with only the final number\.?", "",
                                                   gate[i]["prompt"]).strip() + " " + COT_INSTR) for i in ib}
                for s in range(5):
                    gg = E.decode(model, tok, [cot_p[i] for i in ib], [ctx.key("r1bL5", "COT", i, s) for i in ib], T=0.7, max_new=cf["l5_cot_new"],
                                  greedy=(s == 0), probe=pr)
                    for i, r in zip(ib, gg):
                        res[i].setdefault("cot", []).append(dict(tokens=r["tokens"], text=tok.decode(r["tokens"]), a=r["a"].astype(np.float16),
                                                                  e=r["e"].astype(np.float32), s=r["s"].astype(np.float32),
                                                                  offsets=_offsets(tok, r["tokens"])))
                sub_rows = [(i, k, s) for i in ib for k, e_ in enumerate(gate[i]["subexprs"]) for s in range(5)]
                if sub_rows:
                    gg = E.decode(model, tok, [K.chat_ids(tok, SUB_TEMPLATE.format(e=gate[i]["subexprs"][k])) for i, k, s in sub_rows],
                                  [ctx.key("r1bL5", "SUB", i, k, s) for i, k, s in sub_rows], T=0.7, max_new=cf["l5_sub_new"],
                                  greedy=False)
                    gr = E.decode(model, tok, [K.chat_ids(tok, SUB_TEMPLATE.format(e=gate[i]["subexprs"][k])) for i, k, s in sub_rows if s == 0],
                                  [0] * sum(1 for x in sub_rows if x[2] == 0), max_new=cf["l5_sub_new"], greedy=True)
                    gi = iter(gr)
                    for (i, k, s), r in zip(sub_rows, gg):
                        txt = tok.decode(next(gi)["tokens"]) if s == 0 else tok.decode(r["tokens"])
                        res[i].setdefault("sub", []).append(dict(k=k, s=s, text=txt))
                tf = E.tf_pass(model, tok, [p + [p[-1]] for p in P_], [len(p) for p in P_], probe=pr, batch=cf["batch"])
                for i, x in zip(ib, tf):
                    res[i]["tstar"] = np.asarray(x["a"][0], np.float32) if x.get("a") is not None and len(x["a"]) else None
                return res
            out.update(ctx.cached(f"l5_{b0}", run))
    save(ctx, "l/l5_gen.pt", dict(items=ids, gen=out, channels=_chan_names(pop)),
         config=dict(cot=COT_INSTR, sub=SUB_TEMPLATE, cot_new=cf["l5_cot_new"], sub_new=cf["l5_sub_new"]), labels=gl + rl)


def _offsets(tok, ids):
    """Character offset of every token in the decoded text (for line / '=' positions)."""
    out, s = [], ""
    for t in ids:
        out.append(len(s)); s += tok.decode([t])
    return out
