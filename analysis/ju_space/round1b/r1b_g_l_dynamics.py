"""Independent Gemma L.R, HORIZON and ATTRACT core protocols.

Static ports of the frozen r1b_l_a/b/b2 numerical procedures. All target forwards,
coordinates, chat turns and persisted inputs belong to this G model. No Qwen
run or upstream session artifact is accessed. Core only: X1-X6 are not included.
"""
from __future__ import annotations

import math
import re
import json
import hashlib
from pathlib import Path

import numpy as np

import r1b_core as K
import r1b_g_adapter as GA
import r1b_g_l_native as E
import r1b_l_stats as S
from r1b_g_l import setup, load_pop, loadpt, config
from r1b_g_l_erasure import Damage
from r1b_l_base import (QA_TEMPLATE, SHORT_QA, F_NEU_SCENES, NEU_TEMPLATE, NAT_PROMPT,
                        channels, spin_sets, split_rows, pos_bin, rng, sha, clean)

FAMS = ("F-known", "F-unknown", "F-fab", "F-neu")
ANCH = FAMS[:3]


def path(ctx, rel):
    return ctx.path(rel.removeprefix("l/"))


def save(ctx, rel, value, config=None, labels=()):
    return ctx.save_pt(rel.removeprefix("l/"), value, config=config, labels=list(labels))


def wj(ctx, rel, value, config=None, labels=()):
    return ctx.write_json(rel.removeprefix("l/"), clean(value), config=config, labels=list(labels))


def refw(ctx, model=None):
    return loadpt(ctx.artifact("l_prepare", "refw.pt"))


def ref_sets(ctx, tok, items, cf):
    """Own REF-R16 + target-retokenized REF-K16 + reserved S_L answers."""
    from r1b_g_engine import ModelEngine
    from r1b_g_l_erasure import _reference
    engine = ModelEngine(ctx)
    refs = loadpt(ctx.artifact("k_prepare", "references.pt"))
    rows = refs.get("ref_r16", [])
    need_r, need_k = (2, 4) if ctx.mock else (16, 16)
    if len(rows) < need_r:
        raise K.NotRun(f"own REF-R16 needs {need_r} model-generated rows")
    r16 = K.RefSet([(r["ids"], r["positions"]) for r in rows[:need_r]], "G-REF-R16")
    k16 = _reference(engine, ctx.data().get("ref_k16", []), "G-REF-K16", need_k)
    sl = split_rows(items, "S_L")
    if len(sl) <= cf["ref_heldout"]:
        raise K.NotRun("need S_L questions beyond the held-out damage-reference set")
    held = sl[-cf["ref_heldout"]:]
    ref = K.RefSet.from_prompt_conts([GA.chat_ids(tok, SHORT_QA.format(q=r["question"])) for r in held],
                                    [r["ans0"] for r in held], "G held-out S_L")
    return r16 + k16 + ref, r16 + k16, []


def wt_test(ctx, tok, cf):
    """Frozen WT103 test text in source order, retokenized by this Gemma."""
    rows = ctx.data().get("wt_test", ctx.data().get("wt", {}).get("test", []))
    if not rows:
        raise K.NotRun("L.R needs raw WikiText-103 test text under wt.test or wt_test")
    texts = [r if isinstance(r, str) else r.get("text") for r in rows]
    if any(not isinstance(s, str) for s in texts):
        raise ValueError("wt.test must contain raw text, never imported token arrays")
    ids = tok("\n".join(texts), add_special_tokens=False)["input_ids"]
    n, length = cf["nat"], cf["nat_len"]
    if len(ids) < n * length:
        raise K.NotRun(f"WT103 test needs {n * length} Gemma tokens; received {len(ids)}")
    return [ids[i * length:(i + 1) * length] for i in range(n)], ["MOCK raw test fixture"] if ctx.mock else []


def corpus(ctx, with_tf=False):
    chains = loadpt(path(ctx, "l/lr/fneu.pt"))["chains"] + loadpt(path(ctx, "l/lr/gen.pt"))["chains"]
    th = loadpt(path(ctx, "l/lr_thr.pt"))
    tf = loadpt(path(ctx, "l/lr_tf.pt")) if with_tf else None
    return chains, th, tf


def _array_key(*values):
    h = hashlib.sha256()
    for value in values:
        if value is None:
            h.update(b"None")
            continue
        a = np.ascontiguousarray(value)
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _to_device(value, device):
    if K._torch().is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_device(v, device) for v in value)
    return value

def lr_items(items, cf):
    reserved = {r["id"] for r in split_rows(items, "S_L")[-cf["ref_heldout"]:]}
    items = {k: v for k, v in items.items() if k not in reserved}
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
                    g = E.decode(model, tok, [GA.chat_ids(tok, it["text"]) for it in ib], keys, T=T, max_new=cf["lr_new"], probe=pr,
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
    th = loadpt(path(ctx, "l/lr_thr.pt"))
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
                    g = E.decode(model, tok, [GA.chat_ids(tok, its[f][j]["text"]) for _, f, _, j, _, _ in jb], [x[4] for x in jb], T=T,
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



def lr_tf(ctx):
    """Anchor information per anchored L.R-A chain: teacher-forced log-probs and KL under A (own prompt), A' and A'' (prompts of two
    other items of the same family, derangements seed 'r1bL2perm'); natural passages (SO-WT103 test) read as the reply to 'Write a
    detailed text.' with all channel captures."""
    tok, model, cf, H, items = setup(ctx)
    rw = refw(ctx, model)
    pop = load_pop(ctx)
    its = lr_items(items, cf)
    chains = [c for c in loadpt(path(ctx, "l/lr/gen.pt"))["chains"] if c["kind"] == "A"]
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
        A = GA.chat_ids(tok, its[f][j]["text"]); A1 = GA.chat_ids(tok, its[f][perm[f][0][j]]["text"]); A2 = GA.chat_ids(tok, its[f][perm[f][1][j]]["text"])
        groups.append(([A, A1, A2], c["tokens"]))
    out = []
    for b0 in range(0, len(groups), cf["batch"]):
        out += ctx.cached(f"anchor_{b0}", lambda: E.ctx_logprobs(model, tok, groups[b0:b0 + cf["batch"]], batch=3 * max(1, cf["batch"] // 3)))
    ps, ml = wt_test(ctx, tok, cf)
    pr_ids = GA.chat_ids(tok, NAT_PROMPT)
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



class _ProbeCM:
    def __init__(self, pr):
        self.pr = pr

    def __enter__(self):
        return self.pr

    def __exit__(self, *a):
        self.pr.remove()



def l3b_prompts(items, cf):
    reserved = {r["id"] for r in split_rows(items, "S_L")[-cf["ref_heldout"]:]}
    items = {k: v for k, v in items.items() if k not in reserved}
    cc = [it for it in split_rows(items, "S_L", "CC")][:cf["l3b_prompts"]]
    fab = split_rows(items, "NE")[:cf["l3b_prompts"]]
    return [("F-known", it) for it in cc] + [("F-fab", it) for it in fab]



def l3b_cal(ctx):
    """Damage of H+ (P x 2) and H- (P x 0); random sets R_i+ (CTRL sets 1-6) and R_i- (sets 7-12) calibrated in the gain form so that
    each set's damage equals the H push of the same sign within +-10 % (<= 6 evaluations; out-of-band sets replaced by the next reserve
    set, at most 6 replacements per sign). Damage on L's reference and on REF-U32."""
    tok, model, cf, H, items = setup(ctx)
    rw = refw(ctx, model)
    pop = load_pop(ctx)
    refL, refU, rl = ref_sets(ctx, tok, items, cf)
    dm, du = Damage(model, refL), Damage(model, refU)

    def dmg(ids, a, D=dm):
        def su():
            pr = E.Probe(model, rw); pr.add_op(ids, "scale", val=a)
            return _ProbeCM(pr)
        return D(su)["damage"]
    res = dict(H={"+": dict(alpha=2.0, damage=dmg(pop["P"], 2.0), damage_U32=dmg(pop["P"], 2.0, du)),
                  "-": dict(alpha=0.0, damage=dmg(pop["P"], 0.0), damage_U32=dmg(pop["P"], 0.0, du))}, sets={"+": [], "-": []}, labels=rl)
    nS = cf["l3b_sets"]
    reserve = list(range(2 * nS, len(pop["CTRL"]))) + [("res", j) for j in range(len(pop.get("CTRL_reserve", [])))]
    rp = 0
    for sign, start in (("+", 2.0), ("-", 0.0)):
        tgt = res["H"][sign]["damage"]
        repl = 0
        for i in range(nS):
            k = i if sign == "+" else nS + i
            while True:
                ids = pop["CTRL"][k] if isinstance(k, int) else pop["CTRL_reserve"][k[1]]
                cal = K.calibrate(lambda a: dmg(ids, a), start, tgt, form="gain", max_eval=6)
                if cal["in_band"] or repl >= 6 or rp >= len(reserve):
                    break
                repl += 1; k = reserve[rp]; rp += 1
            res["sets"][sign].append(dict(ctrl=k, ids=ids, alpha=cal["f"], damage=cal["damage"], in_band=cal["in_band"], labels=cal["labels"],
                                          damage_U32=dmg(ids, cal["f"], du), tried=cal["tried"]))
        res[f"g_dmg{sign}"] = all(s["in_band"] for s in res["sets"][sign])
    wj(ctx, "l/l3b_cal.json", res, config=dict(form="gain, <= 6 evaluations", band=0.10, sets=nS), labels=rl)



def l3b_gen(ctx):
    """Turn 1 (64 tokens, T 0.7, keys ('r1bL3', p, s)) with the push window on the first 16 generated tokens, all conditions of a key
    in one batch (common random numbers: twin 0, H+, H-, R_i+, R_i-, sham rows with hooks at alpha 1); turns 2..5 (TURN2_SUFFIX
    'Continue.', ids by concatenation, history think blocks kept, unmodified model, keys ('r1bL3t', n, p, s)); turn 2 for all
    conditions, turns 3-5 for H+, H- and twin 0. Live channels (R2) captured in turn 1."""
    torch = K._torch()
    tok, model, cf, H, items = setup(ctx)
    rw = refw(ctx, model)
    pop = load_pop(ctx)
    cal = K.load_json(path(ctx, "l/l3b_cal.json"))
    suf = E.turn_suffix_ids(tok, "Continue.")
    pr_items = l3b_prompts(items, cf)
    keys = [(p, s) for p in range(len(pr_items)) for s in range(cf["l3b_seeds"])]
    conds = [("twin", None, 1.0), ("H+", pop["P"], 2.0), ("H-", pop["P"], 0.0)]
    for sign in ("+", "-"):
        for i, s_ in enumerate(cal["sets"][sign]):
            conds.append((f"R{i + 1}{sign}", s_["ids"], s_["alpha"]))
    chan = channels(pop) + [(c[0], c[1]) for c in conds[3:]]
    W = cf["l3b_win"]
    per = max(1, cf["batch"] // len(conds))
    sham_keys = set(range(min(cf["l3b_sham"], len(keys))))
    prompts = [GA.chat_ids(tok, QA_TEMPLATE.format(q=it["question"])) for _, it in pr_items]
    batches = []
    for b0 in range(0, len(keys), per):
        kb = keys[b0:b0 + per]

        def run():
            rows, meta = [], []
            for j, (p, s) in enumerate(kb):
                for c in conds:
                    rows.append(prompts[p]); meta.append((p, s, c[0], b0 + j))
                if b0 + j in sham_keys:
                    rows.append(prompts[p]); meta.append((p, s, "sham", b0 + j))
            pr = E.Probe(model, rw, chans=chan)
            for ci, c in enumerate(conds[1:] + [("sham", pop["P"], 1.0)]):
                m = torch.tensor([mm[2] == c[0] for mm in meta])
                if c[1] is not None:
                    pr.add_op(c[1], "scale", val=c[2], rows=m, on=False)
            Tn = max(len(r) for r in rows)
            last = torch.zeros(len(rows), Tn, dtype=torch.bool); last[:, -1] = True

            def on_step(step, probe):
                for o in probe.ops:
                    o["on"] = step <= W
                    o["pos"] = last if step == 0 else None
            try:
                t1 = E.decode(model, tok, rows, [ctx.key("r1bL3", mm[0], mm[1]) for mm in meta], T=0.7, max_new=cf["l3b_turn"], probe=pr,
                              on_step=on_step)
            finally:
                pr.remove()
            turns = {1: [r["tokens"] for r in t1]}
            hist = [rows[i] + t1[i]["tokens"] for i in range(len(rows))]
            for n in range(2, cf["l3b_turns"] + 1):
                act = [i for i, mm in enumerate(meta) if mm[2] != "sham" and (n == 2 or mm[2] in ("twin", "H+", "H-"))]
                ctx_ids = [hist[i] + suf for i in act]
                g = E.decode(model, tok, ctx_ids, [ctx.key("r1bL3t", n, meta[i][0], meta[i][1]) for i in act], T=0.7, max_new=cf["l3b_turn"])
                turns[n] = {i: r["tokens"] for i, r in zip(act, g)}
                for i, r in zip(act, g):
                    hist[i] = hist[i] + suf + r["tokens"]
            return dict(meta=meta, t1=[dict(tokens=r["tokens"], a=r["a"].astype(np.float16), e=r["e"].astype(np.float16),
                                            s=r["s"].astype(np.float16)) for r in t1], turns={n: v for n, v in turns.items() if n > 1})
        batches.append(ctx.cached(f"l3b_{b0}", run))
    save(ctx, "l/l3b_gen.pt", dict(batches=batches, per=per, conds=[c[0] for c in conds], chan=[c[0] for c in chan]),
         config=dict(keys=len(keys), win=W, turn=cf["l3b_turn"], prompts_sha256=sha(prompts)))



def l3b_reads(ctx):
    """R1 reads (every turn's text read by the unmodified model as the reply to the plain prompt, mean over continuation tokens) on P's
    and every pushed set's channel; R3 (turn 1, per token, cache-free) for twin and H+-; block-14 means for the u*_14 line."""
    torch = K._torch()
    tok, model, cf, H, items = setup(ctx)
    rw = refw(ctx, model)
    pop = load_pop(ctx)
    cal = K.load_json(path(ctx, "l/l3b_cal.json"))
    gd = loadpt(path(ctx, "l/l3b_gen.pt"))
    pr_items = l3b_prompts(items, cf)
    prompts = [GA.chat_ids(tok, QA_TEMPLATE.format(q=it["question"])) for _, it in pr_items]
    chan = [("P", pop["P"])] + [(f"R{i + 1}{sg}", s["ids"]) for sg in ("+", "-") for i, s in enumerate(cal["sets"][sg])]
    reads = []
    for b, g in enumerate(gd["batches"]):
        b0 = b * gd["per"]

        def run():
            meta = g["meta"]
            texts = [(i, 1, g["t1"][i]["tokens"]) for i in range(len(meta))]
            for n, d in g["turns"].items():
                texts += [(i, n, t) for i, t in d.items()]
            seqs = [prompts[meta[i][0]] + t for i, n, t in texts]
            starts = [len(prompts[meta[i][0]]) for i, n, t in texts]
            with E.Probe(model, rw, chans=chan) as pr:
                outs = E.tf_pass(model, tok, seqs, starts, probe=pr, batch=cf["batch"])
                for x in outs:
                    x["h14"] = None  # no cross-model SO-AX coordinate is imported
            res = []
            for (i, n, t), x in zip(texts, outs):
                d = dict(i=i, turn=n, R1=np.nanmean(x["a"], 0) if x["n"] else np.full(len(chan), np.nan))
                if n == 1 and meta[i][2] in ("twin", "H+", "H-"):
                    d["R3"] = x["a"][:, 0].astype(np.float32)
                    d["h14"] = x["h14"]
                res.append(d)
            return res
        reads.append(ctx.cached(f"l3b_read_{b0}", run))
    save(ctx, "l/l3b_reads.pt", dict(reads=reads, chan=[c[0] for c in chan]), config=dict(read="R1 / R3"))



def l3b_stats(ctx):
    """Carry d^(n), K_H, K_H^pool (1-D grid on [0, 2.5]), K_(R,i), Delta K (two-level bootstrap: prompts with both seeds, then sets),
    gates G-twin, G-manip, G-push, KV share, labels, u*_14 share; twin-0 rows exported for L3a."""
    tok, model, cf, H, items = setup(ctx)
    pop = load_pop(ctx)
    cal = K.load_json(path(ctx, "l/l3b_cal.json"))
    gd = loadpt(path(ctx, "l/l3b_gen.pt")); rd = loadpt(path(ctx, "l/l3b_reads.pt"))
    chan = rd["chan"]; ci = {c: j for j, c in enumerate(chan)}
    W = cf["l3b_win"]
    G, Rr = gd["batches"], rd["reads"]
    rows = {}             # (key, cond) -> dict
    twin0 = []
    for g, r in zip(G, Rr):
        for i, mm in enumerate(g["meta"]):
            rows[(mm[3], mm[2])] = dict(p=mm[0], s=mm[1], t1=g["t1"][i], R1={}, R3=None, h14=None)
        for x in r:
            mm = g["meta"][x["i"]]
            d = rows[(mm[3], mm[2])]
            d["R1"][x["turn"]] = x["R1"]
            if "R3" in x:
                d["R3"], d["h14"] = x["R3"], x["h14"]
    keys = sorted({k for k, c in rows})
    prompt_of = np.array([rows[(k, "twin")]["p"] for k in keys])
    up = np.unique(prompt_of)
    Wb = S.boot_counts(len(up), cf["nboot"], seed=303)
    kw = Wb[:, np.searchsorted(up, prompt_of)]            # [B, n_keys] key weights (both seeds of a prompt together)

    def mean_b(v):
        v = np.asarray(v, float); ok = np.isfinite(v)
        return float(np.nanmean(v)), (kw * np.where(ok, v, 0)).sum(1) / np.maximum((kw * ok).sum(1), 1e-9)

    def dvec(cp, cm, chp, chm, n):
        """d^(n) per key = ((a_X+(push+) - a_X+(twin)) - (a_X-(push-) - a_X-(twin))) / 2, each pushed set on its own channel."""
        out = []
        for k in keys:
            try:
                a = rows[(k, cp)]["R1"][n][ci[chp]] - rows[(k, "twin")]["R1"][n][ci[chp]]
                b = rows[(k, cm)]["R1"][n][ci[chm]] - rows[(k, "twin")]["R1"][n][ci[chm]]
                out.append((a - b) / 2)
            except KeyError:
                out.append(np.nan)
        return np.array(out)
    st = {}
    d = {n: mean_b(dvec("H+", "H-", "P", "P", n)) for n in range(1, cf["l3b_turns"] + 1)}
    st["d"] = {n: S.summ(*d[n]) for n in d}
    g_push = RUL(st["d"][1])
    K1 = (d[2][0] / d[1][0]) if d[1][0] else float("nan")
    st["K1"] = S.summ(K1, d[2][1] / np.where(d[1][1] != 0, d[1][1], np.nan)) if g_push else None
    grid = np.arange(0, 2.5001, 0.005)

    def kpool(dd, wv):
        n_ = len(dd)
        res = np.array([sum(wv[j] * (dd[j] - dd[0] * Kk ** j) ** 2 for j in range(n_)) for Kk in grid])
        return float(grid[int(np.argmin(res))])
    var = np.array([np.var(d[n][1]) for n in sorted(d)])
    wv = 1.0 / np.maximum(var, 1e-12)
    pts = np.array([d[n][0] for n in sorted(d)])
    reps = np.stack([d[n][1] for n in sorted(d)], 1)
    Kp = kpool(pts, wv)
    Kp_b = np.array([kpool(reps[b], wv) for b in range(reps.shape[0])])
    st["K_pool"] = S.summ(Kp, Kp_b); st["K_pool"]["p_gt1"] = K.p_one_sided(Kp_b, 1.0)
    # random sets: own-channel carries
    nS = len(cal["sets"]["+"])
    KR, KRb, cross = [], [], []
    for i in range(nS):
        cp, cm = f"R{i + 1}+", f"R{i + 1}-"
        vals = {n: mean_b(dvec(cp, cm, cp, cm, n)) for n in (1, 2)}
        d1 = S.summ(*vals[1])
        defined = RUL(d1)
        KR.append(vals[2][0] / vals[1][0] if defined else np.nan)
        KRb.append(vals[2][1] / np.where(vals[1][1] != 0, vals[1][1], np.nan) if defined else np.full(cf["nboot"], np.nan))
        cross.append(S.summ(*mean_b(dvec(cp, cm, "P", "P", 1))))
    KR, KRb = np.array(KR), np.array(KRb)
    defd = np.isfinite(KR)
    st["K_R"] = KR.tolist(); st["n_defined_KR"] = int(defd.sum())
    if g_push and defd.any():
        gg = np.random.default_rng(11)
        Kh = reps[:, 1] / np.where(reps[:, 0] != 0, reps[:, 0], np.nan)
        dK_b = []
        for b in range(cf["nboot"]):
            ss = gg.choice(np.flatnonzero(defd), defd.sum())
            dK_b.append(Kh[b] - np.nanmean(KRb[ss, b]))
        st["dK"] = S.summ(K1 - float(np.nanmean(KR[defd])), dK_b, margin=0.40)
    else:
        st["dK"] = S.summ(float("nan"), [np.nan])
    st["cross_d1"] = cross; st["n_cross_pos"] = int(sum(RUL(c) and c["lo"] > 0 for c in cross))
    # gates: G-twin (sham token-identical), G-manip (live in-window displacement >= 3 SD of sham - twin)
    sh = [k for k in keys if (k, "sham") in rows]
    same = sum(rows[(k, "sham")]["t1"]["tokens"] == rows[(k, "twin")]["t1"]["tokens"] for k in sh)
    st["g_twin"] = same >= len(sh) - max(1, len(sh) // 32) if sh else False
    st["twin_identical"] = f"{same}/{len(sh)}"
    win = slice(W // 2, W)

    def live(c, k, sl):
        a = rows[(k, c)]["t1"]["a"]
        return float(np.nanmean(np.asarray(a[sl, 0], dtype=float))) if len(a) > sl.start else np.nan
    sham_d = np.array([live("sham", k, win) - live("twin", k, win) for k in sh])
    sd = float(np.nanstd(sham_d)) if len(sham_d) > 1 else 0.0
    disp = {c: float(np.nanmean([live(c, k, win) - live("twin", k, win) for k in keys])) for c in ("H+", "H-")}
    st["g_manip"] = bool(abs(disp["H+"]) >= 3 * sd and abs(disp["H-"]) >= 3 * sd and (disp["H+"] != 0 or disp["H-"] != 0))
    st["displacement"] = disp; st["sham_sd"] = sd
    st["g_push"] = bool(g_push)
    # route: KV share in block 17-32 of turn 1 (live R2 against cache-free R3)
    blk = slice(W, 2 * W)
    lv, rv = [], []
    for k in keys:
        try:
            lv.append((live("H+", k, blk) - live("H-", k, blk)) / 2)
            r3p, r3m = rows[(k, "H+")]["R3"], rows[(k, "H-")]["R3"]
            rv.append((float(np.nanmean(r3p[blk])) - float(np.nanmean(r3m[blk]))) / 2)
        except (KeyError, TypeError):
            lv.append(np.nan); rv.append(np.nan)
    lvm, lvb = mean_b(lv); rvm, rvb = mean_b(rv)
    st["kv_share"] = S.summ((lvm - rvm) / lvm if lvm else float("nan"), (lvb - rvb) / np.where(lvb != 0, lvb, np.nan))
    st["retention"] = {f"{a}-{b}": float(np.nanmean([(live("H+", k, slice(a, b)) - live("H-", k, slice(a, b))) / 2 for k in keys]))
                       for a, b in ((W, 2 * W), (2 * W, 3 * W), (3 * W, 4 * W))}
    st["damage_H"] = cal["H"]["+"]["damage"]
    st["few_sets"] = nS < 6
    st["dmg_ok"] = [cal.get("g_dmg+", False), cal.get("g_dmg-", False)]
    u = None  # G has no Qwen SO-AX direction; descriptive axis line is unavailable.
    if u is not None:
        sh_ = []
        for k in keys:
            h1, h0 = rows[(k, "H+")]["h14"], rows[(k, "twin")]["h14"]
            if h1 is not None and h0 is not None and len(h1) == len(u):
                dd = np.asarray(h1) - np.asarray(h0); sh_.append(float((dd @ u) ** 2 / max(dd @ dd, 1e-12) / max(u @ u, 1e-12)))
        st["ustar_share"] = S.summ(float(np.mean(sh_)) if sh_ else float("nan"), [np.nan]) if sh_ else None
    else:
        st["ustar_share"] = None
    wj(ctx, "l/l3b.json", st, config=dict(nboot=cf["nboot"], win=W, turns=cf["l3b_turns"], sets=nS),
       labels=(["few sets"] if nS < 6 else []) + ([] if u is not None else ["SO-AX u*_14 missing: line dropped"]))
    # twin-0 rows for L3a
    tw = [dict(item=pr_items_id(items, cf, rows[(k, "twin")]["p"]), fam=l3b_prompts(items, cf)[rows[(k, "twin")]["p"]][0], seed=rows[(k, "twin")]["s"],
               a=rows[(k, "twin")]["t1"]["a"], e=rows[(k, "twin")]["t1"]["e"], s=rows[(k, "twin")]["t1"]["s"], tokens=rows[(k, "twin")]["t1"]["tokens"])
          for k in keys]
    save(ctx, "l/l3b_twin0.pt", dict(rows=tw, chan=gd["chan"]), config=dict(T=0.7, turn=1))
    wj(ctx, "l/l3b_twin0.json", dict(file="l3b_twin0.pt", n_rows=len(tw), T=0.7, channels=gd["chan"],
                                     sha256=K.sha256_file(path(ctx, "l/l3b_twin0.pt"))), config=dict(T=0.7))



def pr_items_id(items, cf, p):
    return l3b_prompts(items, cf)[p][1]["id"]



def RUL(s):
    """d^(1)'s 90 % interval excludes 0."""
    return s is not None and ((isinstance(s.get("lo"), float) and s["lo"] > 0) or (isinstance(s.get("hi"), float) and s["hi"] < 0))



def _src(c):
    return (c.get("kind", "tw"), c["fam"], c["T"])



def load_chains(ctx, with_twin=True):
    """L.R-A, L.R-B (and L3b's twin-0 rows): per chain a [n, C] (float32), e, s, rep, bins, item, fam, T, kind."""
    cf = setup(ctx)[2]
    ch, th, _ = corpus(ctx)
    out = []
    for c in ch:
        out.append(dict(kind=c["kind"], fam=c["fam"], T=c["T"], item=c["item"], a=np.asarray(c["a"], np.float32), e=c["e"], s=c["s"],
                        rep=c["rep"], n=c["n"], spins=c.get("spins"), zP=c.get("zP"), tokens=c["tokens"], key=c["key"]))
    if with_twin:
        tw = loadpt(path(ctx, "l/l3b_twin0.pt"))
        C = out[0]["a"].shape[1] if out else None
        for r in tw["rows"]:
            n = len(r["tokens"])
            out.append(dict(kind="tw", fam=r["fam"], T=0.7, item=r["item"], a=np.asarray(r["a"], np.float32)[:, :C], e=np.asarray(r["e"], float),
                            s=np.asarray(r["s"], float), rep=np.zeros(n, bool), n=n, spins=None, zP=None, tokens=r["tokens"], key=None))
    nb = th["nb"]
    for c in out:
        c["bins"] = np.minimum(pos_bin(np.arange(1, c["n"] + 1), cf["pos_scale"]), nb - 1)
    return out, th



def binarise(chains, th, names):
    """b for P (Q95(F-neu, T, bin) thresholds), occupancy-matched b for every other channel and the E-channel (per source, family, T,
    bin: the quantile that gives P's occupancy)."""
    C = len(names)
    for c in chains:
        ti = 0 if c["T"] == 0.7 else 1
        c["b"] = np.zeros((c["n"], C + 1), np.int8)
        c["b"][:, 0] = c["a"][:, 0] > th["thrA"][ti][c["bins"]]
    groups = {}
    for k, c in enumerate(chains):
        for b in np.unique(c["bins"]):
            groups.setdefault(_src(c) + (int(b),), []).append((k, c["bins"] == b))
    for g, lst in groups.items():
        A = np.concatenate([np.concatenate([chains[k]["a"][m][:, 1:], chains[k]["e"][m][:, None]], 1) for k, m in lst])
        occ = np.concatenate([chains[k]["b"][m, 0] for k, m in lst]).mean()
        q = np.quantile(A, 1 - occ, axis=0) if occ > 0 else np.full(A.shape[1], np.inf)
        for k, m in lst:
            X = np.concatenate([chains[k]["a"][m][:, 1:], chains[k]["e"][m][:, None]], 1)
            chains[k]["b"][m, 1:] = X > q[None]
    return C + 1          # index C is the E-channel



def padded(chains, col):
    n = max(c["n"] for c in chains) if chains else 0
    B = np.zeros((len(chains), n), np.int8); Lc = np.array([c["n"] for c in chains]); Bn = np.zeros((len(chains), n), np.int64)
    E_ = np.zeros((len(chains), n))
    for i, c in enumerate(chains):
        B[i, :c["n"]] = c["b"][:, col]; Bn[i, :c["n"]] = c["bins"]; E_[i, :c["n"]] = c["e"]
    return B, Lc, Bn, E_



def dwell_mat(B):
    D = np.zeros_like(B, dtype=np.int64); run = np.zeros(B.shape[0], np.int64)
    for t in range(B.shape[1]):
        run = (run + 1) * (B[:, t] == 1); D[:, t] = run
    return D



def design(B, Lc, Bn, E_):
    """At-risk rows (b_t = 1, t + 1 < length): chain, bin, e_t, log(1 + d_t), exit."""
    D = dwell_mat(B)
    T = B.shape[1]
    ok = np.zeros_like(B, bool)
    ok[:, :-1] = (B[:, :-1] == 1) & (np.arange(T - 1)[None] + 1 < Lc[:, None])
    i, t = np.nonzero(ok)
    return np.stack([i, Bn[i, t], E_[i, t], np.log1p(D[i, t]), (B[i, t + 1] == 0).astype(float)], 1)



def hd_vec(B, Lc, Bn, items=None, n_items=None):
    """Mantel-Haenszel log OR over strata chain x bin; with items: per-item numerator / denominator sums (cluster bootstrap)."""
    T = B.shape[1]
    ok = (np.arange(T - 1)[None] + 1 < Lc[:, None])
    x0, x1, s = B[:, :-1], B[:, 1:], Bn[:, :-1]
    nb = int(Bn.max()) + 1 if Bn.size else 1
    key = (np.arange(B.shape[0])[:, None] * nb + s)[ok]
    a = ((x0 == 1) & (x1 == 1))[ok]; b = ((x0 == 1) & (x1 == 0))[ok]; c = ((x0 == 0) & (x1 == 1))[ok]; d = ((x0 == 0) & (x1 == 0))[ok]
    K_ = B.shape[0] * nb
    cnt = [np.bincount(key, weights=v.astype(float), minlength=K_) for v in (a, b, c, d)]
    n = sum(cnt)
    with np.errstate(invalid="ignore", divide="ignore"):
        num = np.where(n > 0, cnt[0] * cnt[3] / n, 0.0); den = np.where(n > 0, cnt[1] * cnt[2] / n, 0.0)
    if items is None:
        return math.log(num.sum() / den.sum()) if num.sum() > 0 and den.sum() > 0 else float("nan")
    ch_item = np.repeat(items, nb)
    return np.bincount(ch_item, weights=num, minlength=n_items), np.bincount(ch_item, weights=den, minlength=n_items)



def sim_vec(fit, Lc, Bn, init, n_sur, g, entry_c=None, exit_c=None):
    """Vectorised Markov surrogates [n_sur, n_chain, T] (time-varying entry / exit, memoryless dwell, initial state copied)."""
    sig = lambda z: 1 / (1 + np.exp(-z))
    ec = fit["entry_c"] if entry_c is None else entry_c
    xc = fit["exit_c"] if exit_c is None else exit_c
    pe = sig(ec[..., :, None] + fit["entry_b"][Bn]); px = sig(xc[..., :, None] + fit["exit_b"][Bn])
    T = Bn.shape[1]
    S_ = np.zeros((n_sur,) + Bn.shape, np.int8)
    S_[..., 0] = init
    for t in range(T - 1):
        u = g.random((n_sur, Bn.shape[0]))
        cur = S_[..., t]
        pe_t = pe[..., t] if pe.ndim == 2 else pe[:, :, t]
        px_t = px[..., t] if px.ndim == 2 else px[:, :, t]
        S_[..., t + 1] = np.where(cur == 0, u < pe_t, u >= px_t)
    S_ *= (np.arange(T)[None, None] < Lc[None, :, None])
    return S_



def markov_fit(B, Lc, Bn):
    chains = [B[i, :Lc[i]] for i in range(len(Lc))]; bins = [Bn[i, :Lc[i]] for i in range(len(Lc))]
    return S.fit_markov(chains, bins, int(Bn.max()) + 1 if Bn.size else 1)



def l3a(ctx):
    """HD (MH over chain x bin), stickiness gamma (chain FE + bin FE + beta_e e - gamma log(1 + d)), chain-specific Markov surrogates
    (200 for P, 50 per comparison), surrogate-corrected Delta gamma, BI on L.R-B against unimodal-frailty surrogates, rank tests against
    the 99 controls, SEL+ / UNC+ / E contrasts, repetition exclusion, selection robustness."""
    tok, model, cf, H, items = setup(ctx)
    pop = load_pop(ctx)
    dev = "cuda" if K._torch().cuda.is_available() else "cpu"
    chains, th = load_chains(ctx)
    names = [n for n, _ in channels(pop)]
    Ecol = binarise(chains, th, names) - 1
    ncol = Ecol + 1
    labels = ["binarisation drift"] if th.get("drift") else []
    uit = sorted({c["item"] for c in chains}); iix = {u: i for i, u in enumerate(uit)}
    item_of = np.array([iix[c["item"]] for c in chains])
    W = S.boot_counts(len(uit), cf["nboot"], seed=301)
    g = rng("r1bL-SUR", "l3a")
    nbin = th["nb"]
    res = dict(names=names + ["E"])
    gam, gsur, hd, nsur = np.full(ncol, np.nan), {}, np.full(ncol, np.nan), {}
    designs = []
    mats = []
    for col in range(ncol):
        B, Lc, Bn, E_ = padded(chains, col)
        mats.append((B, Lc, Bn, E_))
        designs.append(design(B, Lc, Bn, E_ if col != Ecol else np.zeros_like(E_)))
        hd[col] = hd_vec(B, Lc, Bn)
    gam = ctx.cached("l3a_gamma", lambda: S.fit_hazards(designs, len(chains), nbin, use_e=True, device=dev))
    gam_noe = ctx.cached("l3a_gamma_noe", lambda: S.fit_hazards(designs, len(chains), nbin, use_e=False, device=dev))
    sur_med, sur_med_noe, sur_P = np.full(ncol, np.nan), np.full(ncol, np.nan), None
    for col in range(ncol):
        B, Lc, Bn, E_ = mats[col]
        if B.sum() == 0:
            continue
        fit = ctx.cached(f"l3a_markov_{col}", lambda: markov_fit(B, Lc, Bn))
        n_s = cf["n_sur_P"] if col == 0 else cf["n_sur_C"]
        Ss = sim_vec(fit, Lc, Bn, B[:, 0], n_s, g)
        ds = [design(Ss[k], Lc, Bn, E_ if col != Ecol else np.zeros_like(E_)) for k in range(n_s)]
        gs = ctx.cached(f"l3a_sur_{col}", lambda: S.fit_hazards(ds, len(chains), nbin, use_e=True, device=dev))
        sur_med[col] = np.nanmedian(gs)
        sur_med_noe[col] = np.nanmedian(ctx.cached(f"l3a_sur_noe_{col}", lambda: S.fit_hazards(ds, len(chains), nbin, use_e=False, device=dev))) if col in (0, Ecol) else np.nan
        if col == 0:
            sur_P = gs
    dg = gam - sur_med
    ci_ = [i for i, n in enumerate(names) if n.startswith("C")]
    j = {n: i for i, n in enumerate(names)}
    res["gamma_P"] = float(gam[0]); res["gamma_sur_p"] = K.p_rank(gam[0], sur_P) if sur_P is not None else 1.0
    res["gamma_above_sur"] = bool(sur_P is not None and gam[0] > np.nanpercentile(sur_P, 95))
    res["dgamma"] = dg.tolist(); res["dgamma_rank_p"] = K.p_rank(dg[0], dg[ci_]); res["hd"] = hd.tolist(); res["hd_rank_p"] = K.p_rank(hd[0], hd[ci_])
    # bootstrap of the P - SEL+ contrasts (weighted refits for gamma; item sums for HD)

    def gboot(col, use_e=True):
        out = []
        for b0 in range(0, len(W), 50):
            wts = [W[b][item_of] for b in range(b0, min(b0 + 50, len(W)))]
            out += list(ctx.cached(f"l3a_boot_{col}_{use_e}_{b0}", lambda: S.fit_hazards([designs[col]] * len(wts), len(chains), nbin, use_e=use_e, weights=wts, device=dev)))
        return np.array(out)
    gP = gboot(0)
    for nm in ("SEL", "UNC"):
        if nm in j:
            gX = gboot(j[nm])
            res[f"dgamma_minus_{nm}"] = S.summ(dg[0] - dg[j[nm]], (gP - sur_med[0]) - (gX - sur_med[j[nm]]))
            nP, dP = hd_vec(*mats[0][:3], items=item_of, n_items=len(uit)); nX, dX = hd_vec(*mats[j[nm]][:3], items=item_of, n_items=len(uit))
            res[f"hd_minus_{nm}"] = S.summ(hd[0] - hd[j[nm]], np.log((W @ nP) / (W @ dP)) - np.log((W @ nX) / (W @ dX)))
    gPn, gEn = gboot(0, False), gboot(Ecol, False)
    res["dgamma_noe_minus_E"] = S.summ((gam_noe[0] - sur_med_noe[0]) - (gam_noe[Ecol] - sur_med_noe[Ecol]),
                                       (gPn - sur_med_noe[0]) - (gEn - sur_med_noe[Ecol]))
    nP, dP = hd_vec(*mats[0][:3], items=item_of, n_items=len(uit)); nE, dE = hd_vec(*mats[Ecol][:3], items=item_of, n_items=len(uit))
    res["hd_minus_E"] = S.summ(hd[0] - hd[Ecol], np.log((W @ nP) / (W @ dP)) - np.log((W @ nE) / (W @ dE)))
    res["dgamma_minus_sel"] = res.get("dgamma_minus_SEL", S.summ(float("nan"), [np.nan]))
    res["hd_minus_sel"] = res.get("hd_minus_SEL", S.summ(float("nan"), [np.nan]))
    # BI on L.R-B
    k0, k1, mlen = cf["bi_k"][0], cf["bi_k"][1], cf["bi_k"][2]
    bidx = [i for i, c in enumerate(chains) if c["kind"] == "B"]
    by_item = {}
    for i in bidx:
        by_item.setdefault(chains[i]["item"], []).append(i)
    bit = sorted(by_item)

    def bi_of(col, idx_map):
        return S.bi_stat([[chains[i]["b"][:, col] for i in idx_map[it]] for it in bit], k0, k1, mlen, min_seeds=min(6, cf["lrb_keys"] - 2))[0]
    bi = np.array([bi_of(col, by_item) for col in range(ncol)])
    res["bi"] = bi.tolist(); res["bi_rank_p"] = K.p_rank(bi[0], bi[ci_])
    sub = [chains[i] for i in bidx]
    if sub:
        B, Lc, Bn, E_ = padded(sub, 0)
        fit = markov_fit(B, Lc, Bn)
        it_of = np.array([bit.index(c["item"]) for c in sub])
        mu_e = np.array([fit["entry_c"][it_of == k].mean() for k in range(len(bit))]); sd_e = np.array([fit["entry_c"][it_of == k].std() for k in range(len(bit))])
        mu_x = np.array([fit["exit_c"][it_of == k].mean() for k in range(len(bit))]); sd_x = np.array([fit["exit_c"][it_of == k].std() for k in range(len(bit))])
        bis = []
        for k in range(cf["bi_sur"]):
            ec = g.normal(mu_e[it_of], sd_e[it_of]); xc = g.normal(mu_x[it_of], sd_x[it_of])
            Ss = sim_vec(fit, Lc, Bn, B[:, 0], 1, g, entry_c=ec[None], exit_c=xc[None])[0]
            bis.append(S.bi_stat([[Ss[i, :Lc[i]] for i in np.flatnonzero(it_of == q)] for q in range(len(bit))], k0, k1, mlen,
                                 min_seeds=min(6, cf["lrb_keys"] - 2))[0])
        bis = np.array(bis, float)
        res["bi_above_sur99"] = bool(np.isfinite(bi[0]) and bi[0] > np.nanpercentile(bis, 99)) if np.isfinite(bis).any() else False
        res["bi_sur_p"] = K.p_rank(bi[0], bis[np.isfinite(bis)]) if np.isfinite(bis).any() else 1.0
        if "SEL" in j:
            Wb = S.boot_counts(len(bit), cf["nboot"], seed=302)

            def split_vec(col):
                v = []
                for it in bit:
                    o = [float(np.mean(chains[i]["b"][k0:k1, col])) for i in by_item[it] if chains[i]["n"] >= mlen]
                    o = np.array(o)
                    v.append(float(len(o) >= min(6, cf["lrb_keys"] - 2) and (o >= 0.3).sum() >= 2 and (o <= 0.02).sum() >= 2))
                return np.array(v)
            sp_, ss_ = split_vec(0), split_vec(j["SEL"])
            res["bi_minus_sel"] = S.summ(sp_.mean() - ss_.mean(), (Wb @ (sp_ - ss_)) / Wb.sum(1))
    else:
        res["bi_above_sur99"], res["bi_sur_p"] = False, 1.0
    c90 = lambda v: np.nanpercentile(np.asarray(v)[ci_], 90)
    res["dgamma_below_c90"] = bool(dg[0] < c90(dg)); res["hd_below_c90"] = bool(hd[0] < c90(hd)); res["bi_below_c90"] = bool(bi[0] < c90(bi))
    # qualifiers
    stick = res["gamma_above_sur"] and res["dgamma_rank_p"] <= 0.025
    pers = res["hd_rank_p"] <= 0.025
    unc = []
    if stick:
        unc.append(not (_lb(res.get("dgamma_minus_UNC")) and _lb(res["dgamma_noe_minus_E"])))
    if pers:
        unc.append(not (_lb(res.get("hd_minus_UNC")) and _lb(res["hd_minus_E"])))
    res["not_beyond_unc"] = bool(unc and all(unc))
    # repetition exclusion: chains truncated at their first repeat-flagged token
    tr = []
    for c in chains:
        k = np.flatnonzero(c["rep"])
        tr.append(dict(c, n=int(k[0]) if k.size else c["n"], b=c["b"][:int(k[0]) if k.size else c["n"]],
                       bins=c["bins"][:int(k[0]) if k.size else c["n"]], e=c["e"][:int(k[0]) if k.size else c["n"]]))
    tr = [c for c in tr if c["n"] > 1]
    if tr and len(tr) and sum(c["n"] for c in tr) < sum(c["n"] for c in chains):
        hd2, ds2 = [], []
        for col in [0] + ci_:
            B, Lc, Bn, E_ = padded(tr, col)
            hd2.append(hd_vec(B, Lc, Bn)); ds2.append(design(B, Lc, Bn, E_))
        g2 = S.fit_hazards(ds2, len(tr), nbin, device=dev) - sur_med[[0] + ci_]
        res["rep_attractor"] = bool((stick or pers) and hd2[0] < np.nanpercentile(hd2[1:], 90) and g2[0] < np.nanpercentile(g2[1:], 90))
    else:
        res["rep_attractor"] = False
    bix = [i for i, n in enumerate(names) if n.startswith("B")]
    if len(bix) >= 10 and (stick or pers):
        ok = [(dg[i] > np.nanmax(dg[ci_]) and dg[i] > dg[j.get("SEL", 0)]) if stick else (hd[i] > np.nanmax(hd[ci_]) and hd[i] > hd[j.get("SEL", 0)])
              for i in bix]
        res["sel_robust"] = sum(ok) >= 16 * len(bix) / 20
    else:
        res["sel_robust"] = None
    wj(ctx, "l/l3a.json", res, config=dict(n_sur_P=cf["n_sur_P"], n_sur_C=cf["n_sur_C"], bi_sur=cf["bi_sur"], strata="chain x bin",
                                            surrogates="per-chain Markov (chain FE + shared position profile)"), labels=labels)



def _lb(s):
    return isinstance(s, dict) and isinstance(s.get("lo"), float) and s["lo"] > 0



def spins_of(chains, off, name, units_n):
    """Unpacked spins of one population per chain."""
    a, b = off[name]
    out = []
    for c in chains:
        if c.get("spins") is None or c["n"] == 0:
            out.append(np.zeros((0, b - a), np.int8)); continue
        u = np.unpackbits(c["spins"], axis=1)[:, :units_n]
        out.append(u[:c["n"], a:b].astype(np.int8))
    return out



def l3c(ctx):
    """Static PL (independent, pairwise, K-pairwise, covariate fields), G-ESS, G-ISM (held-out PL gain, Gibbs P(K) band, triplets), dPL
    for every population, I2/IN on 20 random 10-unit subsets, basins (descriptive), kinetic Ising (G_pair), A_J, modularity, Glauber dwell,
    G-cal-Ising (five planted sets at the measured sample count and tau_int)."""
    torch = K._torch()
    tok, model, cf, H, items = setup(ctx)
    pop = load_pop(ctx)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ch, th, _ = corpus(ctx)
    chains = [c for c in ch if c["fam"] in FAMS]
    units, off = th["units"], th["off"]
    uit = sorted({c["item"] for c in chains}); par = {u: i % 2 for i, u in enumerate(uit)}
    fitm = np.array([par[c["item"]] == 1 for c in chains])
    nb = th["nb"]
    grp = lambda c: (FAMS.index(c["fam"]) * 2 + (0 if c["T"] == 0.7 else 1)) * nb + np.minimum(pos_bin(np.arange(1, c["n"] + 1), cf["pos_scale"]), nb - 1)
    G = len(FAMS) * 2 * nb
    pops = [n for n, _ in [("P", 1)] + [(x, 1) for x in ("SEL", "UNC") if x in off] + [(f"C{k}", 1) for k in range(len(pop["CTRL"]))] +
            [(f"B{k}", 1) for k in range(len(pop.get("boot") or []))]]
    res = dict(pops=pops)

    def data(name, sel_units=None):
        sp = spins_of(chains, off, name, len(units))
        if sel_units is not None:
            sp = [s[:, sel_units] for s in sp]
        cat = lambda m: np.concatenate([s for s, k in zip(sp, m) if k and len(s)] or [np.zeros((0, sp[0].shape[1]), np.int8)])
        gcat = lambda m: np.concatenate([grp(c) for c, s, k in zip(chains, sp, m) if k and len(s)] or [np.zeros(0, int)])
        tr = lambda m: (np.concatenate([s[:-1] for s, k in zip(sp, m) if k and len(s) > 1] or [np.zeros((0, sp[0].shape[1]), np.int8)]),
                        np.concatenate([s[1:] for s, k in zip(sp, m) if k and len(s) > 1] or [np.zeros((0, sp[0].shape[1]), np.int8)]),
                        np.concatenate([grp(c)[:-1] for c, s, k in zip(chains, sp, m) if k and len(s) > 1] or [np.zeros(0, int)]))
        it = lambda m: np.concatenate([np.full(len(s), uit.index(c["item"])) for c, s, k in zip(chains, sp, m) if k and len(s)] or [np.zeros(0, int)])
        return dict(Sf=cat(fitm), St=cat(~fitm), gf=gcat(fitm), gt=gcat(~fitm), trf=tr(fitm), trt=tr(~fitm), itf=it(fitm), itt=it(~fitm), sp=sp)
    # G-ESS on P
    D = data("P")
    N = D["Sf"].shape[1]
    taus = [S.tau_int(s.mean(1)) for s in D["sp"] if len(s) > 10 and s.shape[1]]
    tau = float(np.mean(taus)) if taus else 1.0
    ess_ok = len(D["Sf"]) / tau >= 10 * N * (N + 1) / 2
    sel_units = None
    if not ess_ok:
        freq = np.array([sum(j in set(b) for b in pop.get("boot") or []) for j in pop["P"]], float)
        sel_units = np.argsort(-freq, kind="stable")[:30]
        D = data("P", sel_units); N = D["Sf"].shape[1]
        ess_ok = len(D["Sf"]) / tau >= 10 * N * (N + 1) / 2
        res["ess_fallback"] = "top 30 units by selection frequency"
    res["tau_int"] = tau; res["ess_ok"] = bool(ess_ok); res["N"] = N
    if N < 10 or not ess_ok:
        res["not_testable"] = "not testable (population too small)" if N < 10 else "not testable (sample)"
    lam = S.pl_select_lam(D["Sf"], D["itf"], device=dev) if len(D["Sf"]) > 20 else 1e-3
    res["lambda"] = lam

    def dpl(Dx, groups=False):
        gf, gt = (Dx["gf"], Dx["gt"]) if groups else (None, None)
        ng = G if groups else 1
        def fit():
            f1 = S.pl_fit(Dx["Sf"], lam, gf, ng, pair=False, device=dev)
            f2 = S.pl_fit(Dx["Sf"], lam, gf, ng, device=dev)
            d_ = S.pl_ll(Dx["St"], f2, gt, dev) - S.pl_ll(Dx["St"], f1, gt, dev)
            return d_, f1, f2
        key = _array_key(Dx["Sf"], Dx["St"], gf, gt, [lam, ng])
        return _to_device(ctx.cached("l3c_pl_" + key, fit), dev)

    def kin(Dx):
        Xf, Yf, gf = Dx["trf"]; Xt, Yt, gt = Dx["trt"]
        if len(Xf) < 5 or len(Xt) < 5:
            return np.zeros(0), None
        def fit():
            k1 = S.kinetic_fit(Xf, Yf, gf, G, pair=False, device=dev)
            k2 = S.kinetic_fit(Xf, Yf, gf, G, device=dev)
            return S.kinetic_ll(Xt, Yt, gt, k2, dev) - S.kinetic_ll(Xt, Yt, gt, k1, dev), k2
        key = _array_key(Xf, Yf, gf, Xt, Yt, gt, [lam, G])
        return _to_device(ctx.cached("l3c_kinetic_" + key, fit), dev)
    dP, f1P, f2P = dpl(D)
    dPiv, _, _ = dpl(D, groups=True)
    gP, kP = kin(D)
    itt = D["itt"]; W = S.boot_counts(len(uit), cf["nboot"], seed=331)

    def itemsum(v, it):
        return np.bincount(it, weights=v, minlength=len(uit)), np.bincount(it, minlength=len(uit)).astype(float)
    sP, nP = itemsum(dP, itt)
    gain_b = (W @ sP) / np.maximum(W @ nP, 1)
    res["pl_gain"] = S.summ(float(dP.mean()) if len(dP) else float("nan"), gain_b)
    # Gibbs check: P(K) band on the test half and triplets
    nch, sw, burn = cf["gibbs"]
    smp = S.gibbs(f2P["h"][0].cpu().numpy(), f2P["J"].cpu().numpy(), n_chains=nch, sweeps=sw, burn=burn, seed=7, device=dev)
    pk_fit = S.pk(smp, N)
    St = D["St"]
    pk_b = np.stack([S.pk(St[np.repeat(np.arange(len(St)), 1)[np.isin(itt, np.flatnonzero(W[b] > 0))]], N) for b in range(min(len(W), 200))])
    lo_b, hi_b = np.percentile(pk_b, 5, 0), np.percentile(pk_b, 95, 0)
    Kmax = min(10, N)
    pk_in = int(((pk_fit[:Kmax + 1] >= lo_b[:Kmax + 1] - 1e-12) & (pk_fit[:Kmax + 1] <= hi_b[:Kmax + 1] + 1e-12)).sum())
    tri_fit = S.triplet_mean(smp); tri_b = [S.triplet_mean(St[np.isin(itt, np.flatnonzero(W[b] > 0))], seed=b) for b in range(min(len(W), 50))]
    tri_ok = bool(np.percentile(tri_b, 5) <= tri_fit <= np.percentile(tri_b, 95)) if tri_b else False
    gism = bool(_lb(res["pl_gain"]) and pk_in >= min(9, Kmax + 1 - 2) and tri_ok)
    res.update(gism=gism, pk_in=pk_in, pk_fit=pk_fit.tolist(), pk_emp=S.pk(St, N).tolist(), triplet_ok=tri_ok)
    if not gism:
        fk = S.pl_fit(D["Sf"], lam, kpair=True, device=dev)
        res["pl_gain_k"] = float((S.pl_ll(St, fk, None, dev) - S.pl_ll(St, f1P, None, dev)).mean()) if len(St) else None
        res["gism_k"] = bool(res["pl_gain_k"] is not None and res["pl_gain_k"] > 0)
    # dPL and G_pair for every population
    dpl_v, dpl_iv, gpair = {}, {}, {}
    per_item = {}
    for name in pops:
        Dx = D if name == "P" else data(name)
        if Dx["Sf"].shape[1] < 2 or len(Dx["St"]) == 0:
            continue
        d1_ = dP if name == "P" else dpl(Dx)[0]
        dpl_v[name] = float(d1_.mean())
        per_item[name] = itemsum(d1_, Dx["itt"])
        if name in ("P", "SEL", "UNC"):
            dpl_iv[name] = (dPiv if name == "P" else dpl(Dx, groups=True)[0])
        gg = gP if name == "P" else kin(Dx)[0]
        gpair[name] = float(gg.mean()) if len(gg) else float("nan")
        if name in ("P", "SEL", "UNC") and len(gg):
            per_item[name + "_g"] = itemsum(gg, np.concatenate([np.full(max(len(s) - 1, 0), uit.index(c["item"])) for c, s, k in
                                                                zip(chains, Dx["sp"], ~fitm) if k and len(s) > 1]))
    cn = [n for n in pops if n.startswith("C") and n in dpl_v]
    res["dpl"] = dpl_v; res["gpair"] = gpair
    res["dpl_rank_p"] = K.p_rank(dpl_v["P"], [dpl_v[n] for n in cn]); res["gpair_rank_p"] = K.p_rank(gpair["P"], [gpair[n] for n in cn])
    res["dpl_below_c90"] = bool(dpl_v["P"] < np.percentile([dpl_v[n] for n in cn], 90)) if cn else False
    res["gpair_below_c90"] = bool(gpair["P"] < np.nanpercentile([gpair[n] for n in cn], 90)) if cn else False

    def contrast(a, b):
        sa, na = per_item[a]; sb, nb_ = per_item[b]
        return S.summ(sa.sum() / max(na.sum(), 1) - sb.sum() / max(nb_.sum(), 1), (W @ sa) / np.maximum(W @ na, 1) - (W @ sb) / np.maximum(W @ nb_, 1))
    for x in ("SEL", "UNC"):
        if x in per_item:
            res[f"dpl_minus_{x.lower()}"] = contrast("P", x)
            if x + "_g" in per_item and "P_g" in per_item:
                res[f"gpair_minus_{x.lower()}"] = contrast("P_g", x + "_g")
            if x in dpl_iv:
                sa, na = itemsum(dpl_iv["P"], itt); sb, nb_ = itemsum(dpl_iv[x], data(x)["itt"])
                res[f"dpl_minus_{x.lower()}_iv"] = S.summ(sa.sum() / max(na.sum(), 1) - sb.sum() / max(nb_.sum(), 1),
                                                          (W @ sa) / np.maximum(W @ na, 1) - (W @ sb) / np.maximum(W @ nb_, 1))
    for k_ in ("dpl_minus_sel", "dpl_minus_sel_iv", "gpair_minus_sel"):
        res.setdefault(k_, S.summ(float("nan"), [np.nan]))
    res["not_beyond_unc"] = bool("dpl_minus_unc" in res and not _lb(res["dpl_minus_unc"]))
    bnames = [n for n in pops if n.startswith("B") and n in dpl_v]
    res["sel_robust"] = (sum(dpl_v[n] > max(dpl_v[c] for c in cn) for n in bnames) >= 16 * len(bnames) / 20) if len(bnames) >= 10 and cn else None
    # I2 / IN on 20 random 10-unit subsets (subset bootstrap)
    g = rng("r1bL3c", "i2in")
    if N >= 10:
        rr = [S.maxent_info(D["Sf"][:, g.choice(N, 10, replace=False)], ess_scale=tau) for _ in range(20 if not ctx.mock else 3)]
        ratio = np.array([r["ratio"] for r in rr], float); inpu = np.array([r["IN_per_unit_bits"] for r in rr], float)
        bi_ = np.random.default_rng(5).integers(0, len(rr), (cf["nboot"], len(rr)))
        res["i2in"] = S.summ(float(np.nanmedian(ratio)), np.nanmedian(ratio[bi_], 1)); res["in_per_unit"] = S.summ(float(np.median(inpu)), np.median(inpu[bi_], 1))
    else:
        res["i2in"] = res["in_per_unit"] = S.summ(float("nan"), [np.nan])
    # basins (descriptive), Rostami check, kinetic descriptives
    hP, JP = f2P["h"][0].cpu().numpy(), f2P["J"].cpu().numpy()
    bas, uniq = S.descend(St, hP, JP) if len(St) else (np.zeros(0, int), [])
    occ = np.bincount(bas, minlength=len(uniq)) / max(len(bas), 1)
    keep = [k for k in range(len(uniq)) if occ[k] >= 0.01]
    res["basins"] = [dict(pattern=list(uniq[k]), occupancy=float(occ[k])) for k in keep]
    res["rostami"] = dict(fitted_modes=_modes(pk_fit), empirical_modes_boot_share=float(np.mean([_modes(p) >= 2 for p in pk_b])))
    if kP is not None:
        Jk = kP["J"].cpu().numpy()
        res["A_J"] = S.asym(Jk); res["modularity"] = S.modularity((Jk + Jk.T) / 2); res["J_norm"] = float(np.linalg.norm(Jk))
    res["fit"] = dict(h=hP.tolist(), J=JP.tolist(), units=(pop["P"] if sel_units is None else [pop["P"][i] for i in sel_units]),
                      sel_units=None if sel_units is None else sel_units.tolist())
    paths = _basin_paths(ctx, D["sp"], hP, JP, "observed")
    res["h_basin"] = h_basin(ctx, chains, paths, items, fitm, cf)
    res["glauber_dwell"] = glauber_dwell(ctx, paths, fitm, smp, nch, hP, JP)
    # G-cal-Ising
    res["g_cal_ising"] = gcal_ising(N, len(D["Sf"]), tau, lam, dev, cf, ctx.mock)
    wj(ctx, "l/l3c.json", res, config=dict(lam=lam, gibbs=cf["gibbs"], fit_half="odd items", test_half="even items"))



def _modes(p):
    p = np.asarray(p, float)
    return int(sum(1 for k in range(len(p)) if p[k] > 0 and (k == 0 or p[k] > p[k - 1]) and (k == len(p) - 1 or p[k] >= p[k + 1])))



def gcal_ising(N, T, tau, lam, dev, cf, mock):
    """Five planted sets (independent, weak couplings, two Hopfield basins, near-critical, hidden-field-driven) at the measured sample
    count, thinned to the measured tau_int; pass if >= 4 of 5 are right."""
    # Preserve the measured real sample count; only explicit MOCK may be small.
    N = max(N, 10); T = int(min(T, 2000) if mock else T); g = np.random.default_rng(17)
    thin = max(1, int(round(tau)))
    out = {}
    for kind in ("P1", "P2", "P3", "P4", "P5"):
        h = np.full(N, -2.0); J = np.zeros((N, N))
        if kind == "P2":
            J = g.normal(0, 0.05, (N, N)); J = (J + J.T) / 2
        elif kind in ("P3", "P4"):
            xi = np.where(g.random((2, N)) < 0.5, 1.0, -1.0)
            J = (xi.T @ xi) / N * (2.0 if kind == "P3" else 1.0); h = -J.sum(1) / 2
        np.fill_diagonal(J, 0)
        if kind == "P5":
            drv = g.random(T // max(thin, 1) + 1) < 0.3
            Sx = (g.random((T, N)) < np.where(np.repeat(drv, thin)[:T, None], 0.4, 0.05)).astype(np.int8)
        else:
            Sx = S.gibbs(h, J, n_chains=max(1, math.ceil(T / 200)), sweeps=200 * thin, burn=200 * thin,
                        seed=3, thin=thin, device=dev)[:T].astype(np.int8)
        half = len(Sx) // 2
        f1 = S.pl_fit(Sx[:half], lam, pair=False, device=dev); f2 = S.pl_fit(Sx[:half], lam, device=dev)
        gain = S.pl_ll(Sx[half:], f2, None, dev) - S.pl_ll(Sx[half:], f1, None, dev)
        bas, uniq = S.descend(Sx[half:], f2["h"][0].cpu().numpy(), f2["J"].cpu().numpy())
        occ = np.bincount(bas, minlength=len(uniq)) / max(len(bas), 1)
        nbas = int((occ >= 0.01).sum())
        coll = float(np.percentile(np.random.default_rng(1).choice(gain, (200, len(gain))).mean(1), 5)) > 0
        right = {"P1": nbas <= 1 and not coll, "P2": nbas <= 1, "P3": nbas >= 2, "P4": True, "P5": not coll}[kind]
        out[kind] = dict(n_basins=nbas, collective=bool(coll), right=bool(right))
    out["pass"] = sum(v["right"] for v in out.values() if isinstance(v, dict)) >= 4
    return out



def _scaled(cf, x, lo=2):
    return max(lo, int(round(x * cf["pos_scale"])))



def entries(b, n, cf):
    """First b = 1 after >= 16 tokens of b = 0, at continuation positions 64..960 (scaled under MOCK)."""
    gap, a0, a1 = _scaled(cf, 16), _scaled(cf, 64), _scaled(cf, 960)
    out = []
    run = 0
    for t in range(n):
        if b[t] == 0:
            run += 1
        else:
            if run >= gap and a0 <= t <= a1:
                out.append(t)
            run = 0
    return out



def exits(b, n, cf):
    """Last b = 1 before >= 16 tokens of b = 0 (time-reversed entries), at positions 64..960."""
    gap, a0, a1 = _scaled(cf, 16), _scaled(cf, 64), _scaled(cf, 960)
    return [t for t in range(a0, min(a1, n - gap)) if b[t] == 1 and not b[t + 1:t + 1 + gap].any()]



def l3d(ctx):
    """EBM (Delta e, Delta s, Delta TT against matched windows), irreversibility d-hat on quartile symbols of a_P and s (post-entry
    against time-reversed pre-exit windows and matched windows; detailed-balance surrogate band), H-specific label against the 99
    controls, G-sigma (IFT check), Sagawa sigma (descriptive), A_k (non-stationarity, descriptive)."""
    tok, model, cf, H, items = setup(ctx)
    pop = load_pop(ctx)
    chains, th = load_chains(ctx, with_twin=False)
    names = [n for n, _ in channels(pop)]
    binarise(chains, th, names)
    W64 = _scaled(cf, 64, 4)
    uit = sorted({c["item"] for c in chains}); g = rng("r1bL3d", "match")
    ci_ = [i for i, n in enumerate(names) if n.startswith("C")]
    by_ft = {}
    for k, c in enumerate(chains):
        by_ft.setdefault((c["fam"], c["T"]), []).append(k)

    def events(col):
        ev = []
        for k, c in enumerate(chains):
            for t in entries(c["b"][:, col], c["n"], cf):
                if t - W64 < 0 or t + W64 > c["n"]:
                    continue
                cands = [m for m in by_ft[(c["fam"], c["T"])] if m != k and chains[m]["n"] >= t + W64 and
                         not any(abs(t2 - t) <= W64 for t2 in entries(chains[m]["b"][:, col], chains[m]["n"], cf))]
                if cands:
                    ev.append((k, t, cands[int(g.integers(len(cands)))]))
        return ev

    def de_stat(col, ev, key="e"):
        rows = []
        for k, t, m in ev:
            x, y = chains[k][key], chains[m][key]
            rows.append(((x[t:t + W64].mean() - x[t - W64:t].mean()) - (y[t:t + W64].mean() - y[t - W64:t].mean()), uit.index(chains[k]["item"])))
        return rows
    W = S.boot_counts(len(uit), cf["nboot"], seed=341)

    def boot_mean(rows):
        if not rows:
            return S.summ(float("nan"), [np.nan])
        v = np.array([r[0] for r in rows]); it = np.array([r[1] for r in rows])
        s_ = np.bincount(it, weights=v, minlength=len(uit)); n_ = np.bincount(it, minlength=len(uit)).astype(float)
        return S.summ(float(v.mean()), (W @ s_) / np.maximum(W @ n_, 1))
    evP = events(0)
    res = dict(n_events=len(evP))
    res["de"] = boot_mean(de_stat(0, evP)); res["de"]["margin"] = 0.10
    res["ds"] = boot_mean(de_stat(0, evP, "s"))
    tts = []
    for k, t, m in evP:
        X = np.stack([chains[k]["s"], chains[k]["e"]], 1); Y = np.stack([chains[m]["s"], chains[m]["e"]], 1)
        tts.append([S.rqa_tt(X[t - W64:t]), S.rqa_tt(X[t:t + W64]), S.rqa_tt(Y[t - W64:t]), S.rqa_tt(Y[t:t + W64])])
    tts = np.array(tts, float).reshape(-1, 4)
    sdTT = np.nanstd(tts) + 1e-12
    rows = [(((r[1] - r[0]) - (r[3] - r[2])) / sdTT, uit.index(chains[k]["item"])) for r, (k, t, m) in zip(tts, evP) if np.isfinite(r).all()]
    res["dtt"] = boot_mean(rows); res["dtt"]["margin"] = 0.30
    de_c = [np.mean([r[0] for r in de_stat(i, events(i))] or [np.nan]) for i in ci_[:cf["n_ctrl"]]]
    res["de_controls"] = de_c
    res["de_rank_p"] = (1 + int(np.sum(np.array(de_c) <= res["de"]["pt"]))) / (len(de_c) + 1) if isinstance(res["de"]["pt"], float) else 1.0
    # irreversibility
    sym_a = {k: S.quartiles(c["a"][:, 0]) for k, c in enumerate(chains)}
    sym_s = {k: S.quartiles(c["s"]) for k, c in enumerate(chains)}

    def irr(col, sym):
        ev = events(col)
        post, match, rev, it_p, it_r = [], [], [], [], []
        for k, t, m in ev:
            post.append(sym(k)[t:t + W64]); match.append(sym(m)[t:t + W64]); it_p.append(uit.index(chains[k]["item"]))
        for k, c in enumerate(chains):
            for te in exits(c["b"][:, col], c["n"], cf):
                if te - W64 + 1 >= 0:
                    rev.append(sym(k)[te - W64 + 1:te + 1][::-1]); it_r.append(uit.index(c["item"]))
        if not post or not rev:
            return None
        dp, dm, dr = S.dhat(post), S.dhat(match), S.dhat(rev)
        it_p, it_r = np.array(it_p), np.array(it_r)
        bp, bm, br = [], [], []
        for b in range(min(len(W), 400 if not ctx.mock else 50)):
            wi = W[b]
            pick = lambda L_, it: [w for w, i in zip(L_, it) for _ in range(int(wi[i]))]
            bp.append(S.dhat(pick(post, it_p))); bm.append(S.dhat(pick(match, it_p))); br.append(S.dhat(pick(rev, it_r)))
        bp, bm, br = np.array(bp), np.array(bm), np.array(br)
        gs = rng("r1bL-SUR", "db", col)
        band = [S.dhat([S.db_surrogate(w, gs) for w in match]) for _ in range(cf["n_sur_P"] if col == 0 else 20)]
        q5, q95 = np.percentile(band, [5, 95])
        return dict(post=dp, match=dm, revexit=dr, post_minus_rev=S.summ(dp - dr, bp - br), post_minus_match=S.summ(dp - dm, bp - bm),
                    band=[float(q5), float(q95)], band_half=float((q95 - q5) / 2), post_above_band=bool(dp > q95), match_above_band=bool(dm > q95),
                    n_post=len(post), n_rev=len(rev))
    ia = irr(0, lambda k: sym_a[k]); is_ = irr(0, lambda k: sym_s[k])
    res["irr_a"], res["irr_s"] = ia, is_
    if ia and is_:
        from r1b_core import holm
        adj, rej = holm([ia["post_minus_rev"]["p_gt0"], is_["post_minus_rev"]["p_gt0"]])
        res["irr_holm_a"], res["irr_holm"] = bool(rej[0]), adj
        cc = []
        for i in ci_[:cf["n_ctrl"]]:
            r = irr(i, lambda k, i=i: S.quartiles(chains[k]["a"][:, i]))
            cc.append(r["post"] - r["revexit"] if r else np.nan)
        res["irr_controls"] = cc
        res["irr_rank_p"] = K.p_rank(ia["post"] - ia["revexit"], [x for x in cc if np.isfinite(x)]) if any(np.isfinite(cc)) else 1.0
    else:
        res["irr_a"] = res["irr_a"] or dict(post_minus_rev=S.summ(float("nan"), [np.nan]), post_minus_match=S.summ(float("nan"), [np.nan]),
                                            post_above_band=False, match_above_band=False, band_half=0.0)
    res["g_sigma"], res["sigma"] = sigma_checks(ctx, tok, model, cf, chains)
    res["A_k"] = a_k(chains, cf)
    wj(ctx, "l/l3d.json", res, config=dict(window=W64, null="time-reversed pre-exit windows (validity fix) + detailed-balance band"))



def sigma_checks(ctx, tok, model, cf, chains):
    """G-sigma: E[exp(-sigma)] = 1 for sigma = log p_T(x) - log p_1(x) over 5-token samples at T 3 and 4 from 8 neutral prompts (exact
    identity; |mean - 1| <= 3 SE); Sagawa sigma (sentence-order reversal, prompt fixed, per token) on the T 1.0 L.R-A chains."""
    torch = K._torch()
    dev = K.model_device(model)
    n = cf["sigma_n"]
    ok, rec = True, {}
    for T in (3.0, 4.0):
        sig = []
        for p in range(8):
            pr = GA.chat_ids(tok, NEU_TEMPLATE.format(s=F_NEU_SCENES[p]))
            for b0 in range(0, n, cf["batch"]):
                m = min(cf["batch"], n - b0)
                g = E.decode(model, tok, [pr] * m, [ctx.key("r1bL3", "IFT", p, T, b0 + j) for j in range(m)], T=T, max_new=5, stop=False)
                seqs = [pr + r["tokens"] for r in g]
                ids = torch.tensor(seqs, device=dev)
                with torch.no_grad():
                    lg = model(input_ids=ids, logits_to_keep=6, use_cache=False).logits[:, -6:-1].float()
                tg = ids[:, len(pr):]
                l1 = torch.log_softmax(lg, -1).gather(2, tg[..., None])[..., 0].sum(1)
                lT = torch.log_softmax(lg / T, -1).gather(2, tg[..., None])[..., 0].sum(1)
                sig += (lT - l1).cpu().tolist()
        x = np.exp(-np.array(sig))
        se = x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("inf")
        rec[str(T)] = dict(mean=float(x.mean()), se=float(se), ok=bool(abs(x.mean() - 1) <= 3 * se))
        ok = ok and rec[str(T)]["ok"]
    sg = {}
    its = {}
    for c in chains:
        if c["kind"] != "A" or c["T"] != 1.0 or c["n"] < 4:
            continue
        txt = tok.decode(c["tokens"]); sents = re.split(r"(?<=[.!?])\s+", txt)
        if len(sents) < 2:
            continue
        rev = K.enc(tok, " ".join(sents[::-1]))
        its.setdefault(c["fam"], []).append((c, rev))
    for f, lst in its.items():
        vals = []
        for c, rev in lst[:(64 if not ctx.mock else 4)]:
            it = lr_items(setup(ctx)[4], cf)[f]
            pid = next((x["text"] for x in it if x["id"] == c["item"]), None)
            if pid is None:
                continue
            pr = GA.chat_ids(tok, pid)
            a, b = E.tf_pass(model, tok, [pr + c["tokens"], pr + rev], [len(pr), len(pr)], batch=2)
            vals.append((np.nansum(a["lp"]) - np.nansum(b["lp"])) / max(c["n"], 1))
        sg[f] = K.dist(vals)
    return dict(checks=rec, ok=bool(ok)), sg



def a_k(chains, cf):
    """Joint-state log-loss asymmetry A_k(l) = H(s_(t+l) | s_t) - H(s_t | s_(t+l)) of k-means states of P's |z| (non-stationarity)."""
    Z = [np.asarray(c["zP"], float) for c in chains if c.get("zP") is not None and len(c["zP"]) > 20]
    if not Z:
        return None
    X = np.concatenate(Z)
    g = np.random.default_rng(3)
    out = {}
    for k in range(2, 9):
        Cn = X[g.choice(len(X), k, replace=False)]
        for _ in range(30):
            lab_ = ((X[:, None] - Cn[None]) ** 2).sum(-1).argmin(1)
            Cn = np.stack([X[lab_ == j].mean(0) if (lab_ == j).any() else Cn[j] for j in range(k)])
        labs = np.split(lab_, np.cumsum([len(z) for z in Z])[:-1])
        out[k] = {}
        for lag in (1, 2, 4, 8, 16):
            J = np.zeros((k, k)) + 1e-9
            for s in labs:
                if len(s) > lag:
                    np.add.at(J, (s[:-lag], s[lag:]), 1)
            P_ = J / J.sum(); pa, pb = P_.sum(1), P_.sum(0)
            Hab = -(P_ * np.log(P_ / pa[:, None])).sum(); Hba = -(P_ * np.log(P_ / pb[None])).sum()
            out[k][lag] = float(Hab - Hba)
    return out



def _guard(ctx):
    if hasattr(ctx, "time_left") and ctx.time_left() < 30:
        raise K.StepYield()


class _GuardedContext:
    def __init__(self, ctx):
        self._ctx = ctx

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def cached(self, name, fn, **kwargs):
        _guard(self._ctx)
        return self._ctx.cached(name, fn, **kwargs)


def _piece(ctx, name, fn, required=()):
    """Checkpoint complete scientific pieces without losing their output manifest."""
    _guard(ctx)
    marker = Path(path(ctx, f"pieces/{name}.json"))
    if marker.is_file():
        previous = json.loads(marker.read_text())
        outputs = previous.get("outputs", {})
        if outputs and all(Path(path(ctx, p)).is_file() and K.sha256_file(path(ctx, p)) == h for p, h in outputs.items()):
            if hasattr(ctx, "outputs"):
                ctx.outputs.extend(outputs)
                ctx.outputs.append(f"pieces/{name}.json")
            return {"status": "measured", "resumed": True}
    missing = [p for p in required if not Path(path(ctx, p)).is_file()]
    if missing:
        return {"status": "blocked", "reason": "required G dynamics outputs missing", "missing": missing}
    before = len(getattr(ctx, "outputs", []))
    try:
        fn(ctx)
    except (K.NotRun, FileNotFoundError, ValueError) as exc:
        return {"status": "blocked", "reason": str(exc)}
    outputs = getattr(ctx, "outputs", [])[before:]
    hashes = {p: K.sha256_file(path(ctx, p)) for p in outputs if Path(path(ctx, p)).is_file()}
    ctx.write_json(f"pieces/{name}.json", {"status": "measured", "outputs": hashes, "mock": bool(ctx.mock)})
    return {"status": "measured"}


def measure(ctx):
    """Complete core L.R/L2b/c/L3 measurements with original gates and sizes."""
    ctx = _GuardedContext(ctx)
    coverage = {}
    branches = ("L.R", "L2b", "L2c", "L3a", "L3b", "L3c", "L3d")
    required = [ctx.artifact("l_prepare", name) for name in ("l_pop.json", "refw.pt", "items.json")]
    missing = [str(p) for p in required if not Path(p).is_file()]
    if missing:
        coverage = {name: {"status": "blocked", "reason": "own Gemma population/labels missing", "missing": missing} for name in branches}
    else:
        pop = load_pop(ctx)
        tok, model, cf, H, items = setup(ctx)
        valid = pop.get("status") == "measured" and bool(pop.get("P")) and len(pop.get("CTRL", [])) >= cf["n_ctrl"]
        if not valid:
            coverage = {name: {"status": "blocked", "reason": "own nonempty population and complete matched controls required"} for name in branches}
        else:
            pieces = {}
            pieces["lr_fneu"] = _piece(ctx, "lr_fneu", lr_fneu)
            families = lr_items(items, cf)
            counts = {f: len(families[f]) for f in ANCH}
            enough = all(n >= cf["n_fam"] for n in counts.values())
            pieces["lr_gen"] = (_piece(ctx, "lr_gen", lr_gen, ["lr_thr.pt"]) if enough else
                                 {"status": "blocked", "reason": f"L.R needs {cf['n_fam']} eligible items per anchored family, outside reserved references", "counts": counts})
            pieces["lr_tf"] = _piece(ctx, "lr_tf", lr_tf, ["lr/gen.pt", "lr_thr.pt"])
            coverage["L.R"] = {"status": "measured" if all(r["status"] == "measured" for r in pieces.values()) else "blocked", "pieces": pieces}
            coverage["L2b"] = _piece(ctx, "l2b", l2b, ["lr/fneu.pt", "lr/gen.pt", "lr_tf.pt"])
            coverage["L2c"] = _piece(ctx, "l2c", l2c, ["lr/fneu.pt", "lr/gen.pt"])
            prompts = l3b_prompts(items, cf)
            nknown = sum(f == "F-known" for f, _ in prompts)
            nfab = sum(f == "F-fab" for f, _ in prompts)
            if min(nknown, nfab) >= cf["l3b_prompts"]:
                twinpieces = {}
                for name, fn, needs in (("l3b_cal", l3b_cal, []), ("l3b_gen", l3b_gen, ["l3b_cal.json"]),
                                        ("l3b_reads", l3b_reads, ["l3b_gen.pt"]),
                                        ("l3b_stats", l3b_stats, ["l3b_gen.pt", "l3b_reads.pt"])):
                    twinpieces[name] = _piece(ctx, name, fn, needs)
                coverage["L3b"] = {"status": "measured" if all(r["status"] == "measured" for r in twinpieces.values()) else "blocked", "pieces": twinpieces}
            else:
                coverage["L3b"] = {"status": "blocked", "reason": f"L3b requires {cf['l3b_prompts']} F-known and F-fab items outside damage references",
                                   "counts": {"F-known": nknown, "F-fab": nfab}}
            coverage["L3a"] = _piece(ctx, "l3a", l3a, ["lr/fneu.pt", "lr/gen.pt", "l3b_twin0.pt"])
            if len(pop["P"]) < 10:
                coverage["L3c"] = {"status": "blocked", "reason": "not testable (population too small)", "N": len(pop["P"])}
                wj(ctx, "l3c.json", {"not_testable": "not testable (population too small)", "ess_ok": False, "N": len(pop["P"])})
            else:
                coverage["L3c"] = _piece(ctx, "l3c", l3c, ["lr/fneu.pt", "lr/gen.pt"])
            coverage["L3d"] = _piece(ctx, "l3d", l3d, ["lr/fneu.pt", "lr/gen.pt"])
    result = {"status": "completed" if all(r["status"] == "measured" for r in coverage.values()) else "partial",
              "coverage": coverage, "mock": bool(ctx.mock), "scientific_verdicts": "apply inherited calibrated rules separately",
              "independent_of_qwen_runs": True, "source_protocols": {
                  name: K.sha256_file(str(Path(__file__).with_name(name)))
                  for name in ("r1b_l_a.py", "r1b_l_b.py", "r1b_l_b2.py", "r1b_l_stats.py", "r1b_l_rules.py")}}
    ctx.write_json("l_dynamics.json", clean(result))
    return result


def _basin_paths(ctx, spins, h, J, tag):
    """Zero-temperature minima per trajectory, checkpointed without token trims."""
    out = []
    for i, values in enumerate(spins):
        if not len(values):
            out.append({"labels": np.zeros(0, int), "minima": []})
            continue
        def find(values=values):
            labels, patterns = S.descend(values, h, J)
            names = [np.packbits(np.asarray(p, dtype=np.uint8)).tobytes().hex() for p in patterns]
            return {"labels": labels, "minima": names}
        key = _array_key(values, h, J)
        out.append(ctx.cached(f"l3c_basin_{tag}_{i}_{key}", find))
    return out


def h_basin(ctx, chains, paths, items, fitm, cf):
    """Fit-only choice of CI-associated basin, evaluated on disjoint even items.

    The underspecified answer covariate is fixed here as the identity of the
    item's native baseline answer (categorical); log chain length is continuous.
    No generated chain is assigned a new hallucination label by this analysis.
    """
    from r1b_k_base import auroc
    rows = []
    for i, (chain, basin) in enumerate(zip(chains, paths)):
        source = items.get(chain["item"], {})
        if source.get("cls") not in ("CC", "CI") or not chain["n"]:
            continue
        counts = np.bincount(basin["labels"], minlength=len(basin["minima"])) / chain["n"]
        rows.append({"item": chain["item"], "fit": bool(fitm[i]), "y": int(source["cls"] == "CI"),
                     "answer": tuple(source.get("ans0", [])), "length": chain["n"],
                     "occupancy": dict(zip(basin["minima"], counts.tolist()))})
    train = [r for r in rows if r["fit"]]; test = [r for r in rows if not r["fit"]]
    info = {"interpretation": "descriptive item-class association; no new per-chain hallucination labels",
            "selection_split": "odd item identities", "evaluation_split": "even item identities",
            "covariates": ["exact native baseline answer identity", "log generated chain length"], "penalty": .001}
    if not train or not test or {r["y"] for r in train} != {0, 1} or {r["y"] for r in test} != {0, 1}:
        return {"status": "not_testable", "reason": "both CC and CI item classes required in fit and test halves", **info}
    fit_ids, test_ids = sorted({r["item"] for r in train}), sorted({r["item"] for r in test})
    if set(fit_ids) & set(test_ids):
        raise ValueError("H-basin fit/evaluation item overlap")
    candidates = sorted({p for r in train for p in r["occupancy"]})
    candidates = [p for p in candidates if np.mean([r["occupancy"].get(p, 0) for r in train]) >= .01]
    answers = sorted({r["answer"] for r in train})
    # Drop one category; unseen test answers get the all-zero reference code.
    categories = {v: i for i, v in enumerate(answers[1:])}
    lengths = np.log([r["length"] for r in train]); mean, sd = lengths.mean(), max(lengths.std(), 1e-12)
    multiplicity = {i: sum(r["item"] == i for r in train) for i in fit_ids}
    weights = np.asarray([1 / multiplicity[r["item"]] for r in train], np.float32)
    def design(rows_, candidate):
        X = np.zeros((len(rows_), 2 + len(categories)), np.float32)
        for i, row in enumerate(rows_):
            X[i, 0] = row["occupancy"].get(candidate, 0)
            X[i, 1] = (math.log(row["length"]) - mean) / sd
            if row["answer"] in categories:
                X[i, 2 + categories[row["answer"]]] = 1
        return X
    fitted = []
    for candidate in candidates:
        X = design(train, candidate); y = np.asarray([r["y"] for r in train], np.float32)
        def fit(X=X, y=y):
            W, b = S.logreg(K._torch().tensor(X)[None], K._torch().tensor(y)[None],
                           K._torch().tensor(weights)[None], K._torch().tensor([.001]))
            return W[0].numpy(), float(b[0])
        beta, intercept = ctx.cached("l3c_hbasin_" + _array_key(X, y, weights), fit)
        fitted.append((float(beta[0]), candidate, beta, intercept))
    if not fitted or max(r[0] for r in fitted) <= 0:
        return {"status": "not_selected", "reason": "no fit-half basin with positive adjusted CI log-odds",
                "n_candidates": len(fitted), "fit_items": fit_ids, "test_items": test_ids, **info}
    effect, candidate, beta, intercept = max(fitted, key=lambda r: (r[0], r[1]))
    scores = design(test, candidate) @ beta + intercept
    by = []
    for item in test_ids:
        ix = [i for i, r in enumerate(test) if r["item"] == item]
        by.append({"item": item, "y": test[ix[0]]["y"], "score": float(scores[ix].mean()),
                   "occupancy": float(np.mean([test[i]["occupancy"].get(candidate, 0) for i in ix]))})
    y = np.array([r["y"] for r in by]); score = np.array([r["score"] for r in by]); occ = np.array([r["occupancy"] for r in by])
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    generator = np.random.default_rng(334)
    draws = []
    for _ in range(cf["nboot"]):
        pick = np.r_[generator.choice(pos, len(pos)), generator.choice(neg, len(neg))]
        draws.append(auroc(y[pick], score[pick]))
    return {"status": "measured", "basin_packed_bits": candidate, "fit_adjusted_logodds": effect,
            "test_auroc": S.summ(auroc(y, score), draws), "test_occupancy_auroc": float(auroc(y, occ)),
            "fit_items": fit_ids, "test_items": test_ids, "per_test_item": by, "n_candidates": len(fitted),
            "bootstrap_unit": "item, stratified by baseline CC/CI", "bootstrap_seed": 334, **info}


def _dwell_summary(paths):
    """Basin residence, with first runs excluded and final runs right-censored."""
    durations, censored = [], []
    for path_ in paths:
        x = np.asarray(path_["labels"])
        if not len(x):
            continue
        boundaries = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1, len(x)]
        lengths = np.diff(boundaries)
        for j, length in enumerate(lengths[1:], 1):
            durations.append(int(length)); censored.append(j == len(lengths) - 1)
    if not durations:
        return {"status": "not_testable", "reason": "no observed basin transition after the initial left-censored run"}
    lengths, censored = np.asarray(durations), np.asarray(censored)
    survival, curve = 1., []
    for t in sorted(set(lengths.tolist())):
        at_risk = int((lengths >= t).sum()); events = int(((lengths == t) & ~censored).sum())
        survival *= 1 - events / at_risk
        curve.append({"duration": t, "survival": survival, "at_risk": at_risk, "events": events,
                      "censored": int(((lengths == t) & censored).sum())})
    return {"status": "measured", "n_runs": len(lengths), "n_right_censored": int(censored.sum()),
            "kaplan_meier": curve, "uncensored_mean": float(lengths[~censored].mean()) if (~censored).any() else None}


def glauber_dwell(ctx, observed, fitm, samples, n_chains, h, J):
    """Compare observed test trajectories with the fitted forward Gibbs chains.

    Reconstruct trajectory identity from the sampler's documented sweep-major
    layout; never concatenate independent chains into a fictitious trajectory.
    The two time units are retained explicitly; this is no physical-time model.
    """
    values = np.asarray(samples)
    if len(values) % n_chains:
        raise ValueError("Gibbs samples cannot be separated into their original chains")
    simulated = values.reshape(-1, n_chains, values.shape[1]).transpose(1, 0, 2)
    paths = _basin_paths(ctx, simulated, h, J, "glauber")
    return {"observed": _dwell_summary([p for p, fit in zip(observed, fitm) if not fit]),
            "simulated": _dwell_summary(paths), "n_simulated_chains": n_chains,
            "samples_per_simulated_chain": simulated.shape[1],
            "observed_time_unit": "generated token", "simulated_time_unit": "10 systematic single-site heat-bath sweeps",
            "interpretation": "descriptive residence comparison; no calibrated conversion between tokens and Glauber updates"}
