"""r1b_K_S2a / r1b_K_S2b (account B): K4 gating (SO-LAB export), K4 cells (V1-V8, decoding grid, temperature arm, oracle chains),
verifier and break-off steps, b*, G-K7, SO-LENSOUT, the self-consistency KL (Teoh 2026), K4p, K6c, K6f, K6g, K6h, K10c, REF-U32
reporting; K6e (S2b). Chains are scored in session (K1 scorers); statistics and verdicts are r1b_K_ana's."""
import math, os
import numpy as np
import r1b_core as RC
import r1b_k_base as KB
import r1b_k_labels as KL
import r1b_k_lens as LN
from r1b_k_base import N, H, T_, write, save, load_pt, own, up, Interv, by_layer, set_values, coded, val
import r1b_k_s1 as S1

BLOCKS = (10, 14, 18, 22)


def env(ctx):
    """Model, bank, SO-HN (H_A decided in session from the SG8 preview's K3a detail; OD-G8), the lens with its capture sets."""
    tok, model = ctx.model(); b = KB.bank(ctx); hs, rs, R = S1.hnset(ctx), S1.rssets(ctx), S1.refs(ctx)
    pv = RC.load_json(up(ctx, "r1b_K_S1", "sg8_preview.json")); det = pv.get("detail", {})
    if hs["H_op_name"] == "H_ar" or (det.get("arith", {}).get("string", "").startswith("H-neurons replicated")):
        HA, lab = hs["H_op"], None
    elif det.get("har", {}).get("string") == "H_ar replicated (arithmetic in-domain)" and hs["H_ar"]:
        HA, lab = hs["H_ar"], "domain-specific H-set"
    else:
        HA, lab = hs["H_op"], "H not valid in arithmetic (in-session preview)"
    g = T_().Generator().manual_seed(1210); npl = R["wn"].shape[1]; nL = R["wn"].shape[0]
    panel = [[int(k // npl), int(k % npl)] for k in T_().randperm(npl * nL, generator=g)[:N(ctx, 512, 16)].tolist()]
    sets = {"H": HA, "H+": hs["H_plus"], "panel": panel, **{f"RS{k}": s for k, s in enumerate(rs["RS39"])},
            **{f"MC{k}": s for k, s in enumerate(rs["RS39mc"][:N(ctx, 20, 3)])}}
    sets = {k: v for k, v in sets.items() if v}
    lens = LN.Lens(model, tok, b["frames"], wn=R["wn"].to(ctx.device), ref=dict(c_mu=R["c_mu"], c_sd=R["c_sd"]), sets=sets)
    k5 = up(ctx, "r1b_K_S3", "k5a_calib.json", required=False)
    variant = (RC.load_json(k5) or {}).get("variant") if k5 else None
    return dict(tok=tok, model=model, b=b, hs=hs, rs=rs, R=R, HA=HA, HA_label=lab, lens=lens, sets=sets,
                variant=variant if variant in ("E", "L") else "E", variant_label=None if variant in ("E", "L") else "not calibrated (K5a not run)",
                J=KB.jl(ctx, model), conv=S1.k4j_conv(ctx), dbias=_digit_bias(ctx, tok, lens))


def _digit_bias(ctx, tok, lens):
    """MOCK only: a selection bias towards digit tokens so that the random mock model writes numbers (exercises the scorers)."""
    if not ctx.mock:
        return None
    v = T_().zeros(lens.V, device=lens.dev); v[KB.digit_ids(tok)] = 6.0; v[tok.convert_tokens_to_ids("Ġ")] = 3.0; v[tok.convert_tokens_to_ids(",")] = 2.0
    return v


def item_label(ctx, it, text):
    return S1.label_ach(ctx, it, text)


# ---------------------------------------------------------------------------------------------------------------------------------
# K4 gating and states
# ---------------------------------------------------------------------------------------------------------------------------------
def k4_gate(ctx):
    """Gating of the A-lens pool (A-cal excluded): greedy + 4 samples at T 0.7 (keys r1bK_gate); OWN; 256 correct and 150 wrong items
    matched on template, operations, x present and digit counts (new items, seed 1005, if fewer than 150 wrong); SO-LAB export."""
    tok, model = ctx.model(); b = KB.bank(ctx)
    pool = [it for it in b["ach_short"] if it["split"] == "A-lens" and not it["cal"]][:N(ctx, 10 ** 6, 40)]
    ncor, nwr = N(ctx, 256, 8), N(ctx, 150, 6)

    def gate(items):
        P = [RC.chat_ids(tok, it["prompt"]) for it in items]
        g = KB.decode(model, tok, P, max_new=12, batch=N(ctx, 128, 16))
        sm = KB.decode(model, tok, [p for p in P for _ in range(4)], keys=[ctx.key("r1bK_gate", it["id"], s) for it in items for s in range(4)],
                       T=0.7, max_new=12, batch=N(ctx, 128, 16))
        out = {}
        for k, it in enumerate(items):
            gt, st = tok.decode(g[k]), [tok.decode(c) for c in sm[4 * k:4 * k + 4]]
            ok = item_label(ctx, it, gt); own_ = sorted({x for x in (KL.parse_number(t) for t in [gt] + st) if x is not None})
            out[it["id"]] = dict(stratum="correct" if ok else "wrong", greedy=g[k], greedy_text=gt, samples=sm[4 * k:4 * k + 4],
                                 sample_keys=[ctx.key("r1bK_gate", it["id"], s) for s in range(4)], own=own_,
                                 W=KL.w_enum(it, KL.parse_number(gt)) if not ok else [])
        return out
    G = gate(pool); extra = []
    if sum(v["stratum"] == "wrong" for v in G.values()) < nwr:
        extra = KL.gen_ach_short(seed=1005, n=N(ctx, 1000, 20), splits=(("A-extra", 10 ** 6),))
        for it in extra:
            it["id"] = "x" + it["id"]
        G.update(gate(extra))
    items = {it["id"]: it for it in pool + extra}
    cor = [i for i in G if G[i]["stratum"] == "correct"]; wr = [i for i in G if G[i]["stratum"] == "wrong"]
    key = lambda i: (items[i]["template"], items[i]["nops"], items[i]["mul"], tuple(items[i]["digits"]))
    wr_sel = wr[:nwr]; want = {key(i) for i in wr_sel}
    cor_sel = [i for i in cor if key(i) in want][:ncor]; cor_sel += [i for i in cor if i not in cor_sel][:ncor - len(cor_sel)]
    lab = ["not testable (too few wrong items)"] if len(wr_sel) < N(ctx, 64, 2) else []
    write(ctx, "lab/k4_gating.json", dict(items={i: dict(stratum=v["stratum"], greedy=v["greedy"], samples=v["samples"],
                                                         keys=v["sample_keys"], ops=items[i]["nops"]) for i, v in G.items()},
                                          extra_items={it["id"]: it for it in extra}), config=dict(T=0.7, n_samples=4), labels=lab)
    save(ctx, "k/k4/items.pt", dict(correct=cor_sel, wrong=wr_sel, gate=G, items=items, G_K19=lab))


def _items(ctx):
    d = load_pt(up(ctx, "r1b_K_S2a", "k/k4/items.pt")); return d


def k4_states(ctx):
    """Read states h_l(t0) (all items) and h_l(t1) (first answer digit, own greedy answer teacher-forced; correct items) at blocks
    {10, 14, 18, 22}; z of the capture sets at the read positions (rho_H, K6c(ii)); mu_task (leave-one-item-out); seeds; V2 matching
    (derangement within matched groups, seed 51); V7 states; n26; V8's beta (REF-K16, 32 A-cal states, 0.05 nats)."""
    torch = T_()
    E = env(ctx); tok, model, lens = E["tok"], E["model"], E["lens"]; I = _items(ctx); nL = lens.nL
    ids = I["correct"] + I["wrong"]; blocks = [l for l in BLOCKS if l < nL - 1]
    uni = sorted({tuple(n) for s in E["sets"].values() for n in s}); sel = by_layer(uni)
    St = {}
    for ptype in ("t0", "t1"):
        its = ids if ptype == "t0" else I["correct"]; seqs, pos = [], []
        for i in its:
            p = RC.chat_ids(tok, I["items"][i]["prompt"])
            if ptype == "t0":
                seqs.append(p); pos.append(len(p) - 1)
            else:
                g = I["gate"][i]["greedy"]; d = next((k for k, t in enumerate(g) if tok.decode([t]).strip().isdigit()), 0)
                seqs.append(p + g[:d + 1]); pos.append(len(p) + d)
        hs = {l: [] for l in blocks}; zs = []
        for b0, idt, am, pad, _ in KB.tf_batches(model, seqs, tok.pad_token_id, batch=N(ctx, 32, 8), logits=False,
                                                  setup=lambda b0, B: _cap(model, blocks, sel, zs)):
            pp = [int(pad[r]) + pos[b0 + r] for r in range(idt.shape[0])]
            for l in blocks:
                hs[l].append(torch.stack([_CAP["h"][l][r, pp[r]] for r in range(len(pp))]).cpu())
            zs[-1] = torch.stack([zs[-1][r, pp[r]] for r in range(len(pp))]).cpu()
        St[ptype] = dict(items=its, h={l: torch.cat(v) for l, v in hs.items()}, z=torch.cat(zs) if zs else None)
    for ptype, s in St.items():
        n = len(s["items"]); s["mu"] = {l: (s["h"][l].sum(0, keepdim=True) - s["h"][l]) / max(n - 1, 1) for l in blocks}
        s["seeds"] = {l: LN.jlens_seeds(model, tok, E["J"], E["conv"], l, s["h"][l].to(ctx.device), k=N(ctx, 10, 3)) for l in blocks}
    it = I["items"]; key = lambda i: (it[i]["template"], it[i]["nops"], it[i]["mul"], tuple(it[i]["digits"]))
    g = np.random.default_rng(51); foreign = {}
    for ptype, s in St.items():
        grp = {}
        for k, i in enumerate(s["items"]):
            grp.setdefault(key(i), []).append(k)
        f = np.arange(len(s["items"]))
        for ks in grp.values():
            if len(ks) > 1:
                p = list(np.roll(g.permutation(ks), 1)); f[ks] = [x for x in p]
        allk = np.arange(len(s["items"]))
        for k in range(len(f)):
            if f[k] == k:
                f[k] = (k + 1) % len(f)
        foreign[ptype] = f
    R = E["R"]; sigh = R["sigma_half"]
    v7 = {}
    for l in (14, 18):
        if l in blocks and l in sigh:
            s = St["t0"]; nc = len(I["correct"])
            v7[l] = LN.v7_states(s["mu"][l][:nc], s["h"][l][:nc], sigh[l].double(), [ctx.key("r1bK_V7", i, l) for i in s["items"][:nc]])
    n26 = None
    with RC.Hooks(model) as hk, torch.no_grad():
        hk.capture([nL - 2]); model.model(input_ids=torch.tensor(E["b"]["wt"]["refw"][:N(ctx, 64, 4)], device=ctx.device))
        n26 = float(hk.states[nL - 2][0][:, 16:127].norm(dim=-1).median())
    # V8: beta_l (h - mu) added at block l at every position of REF-K16, calibrated to 0.05 nats over 32 A-cal states
    cal = [x for x in E["b"]["ach_short"] if x.get("cal")][:N(ctx, 32, 4)]; beta = {}
    if cal:
        seqs = [RC.chat_ids(tok, x["prompt"]) for x in cal]; hc = {}
        for b0, idt, am, pad, _ in KB.tf_batches(model, seqs, tok.pad_token_id, batch=32, logits=False, setup=lambda b0, B: _cap(model, blocks, {}, [])):
            for l in (14, 18):
                if l in blocks:
                    hc.setdefault(l, []).append(torch.stack([_CAP["h"][l][r, -1] for r in range(idt.shape[0])]).cpu())
        dm = RC.Damage(model, KB.ref_k16(ctx))
        for l, v in hc.items():
            V = torch.cat(v); dv = V - St["t0"]["mu"][l].mean(0)

            def dmg(bt, l=l, dv=dv):
                return float(np.mean([dm(lambda d=d: _AddHook(model, l, bt * d))["damage"] for d in dv[:N(ctx, 32, 2)]]))
            res = RC.calibrate(dmg, 0.05, 0.05, form="push", band=0.10, max_eval=6, lo=0.045, hi=0.055)
            beta[l] = dict(beta=res["f"] if res["in_band"] else None, cal=res,
                           label=None if res["in_band"] else "no re-injection (no beta in band)")
    save(ctx, "k/k4/states.pt", dict(St=St, foreign=foreign, v7=v7, n26=n26, beta=beta, blocks=blocks, union=uni, HA=E["HA"],
                                     HA_label=E["HA_label"], variant=E["variant"], variant_label=E["variant_label"], conv=E["conv"]))


_CAP = {}


def _cap(model, blocks, sel, zs):
    """Context manager for tf_batches: block outputs at `blocks` (full sequences) into _CAP['h'], z of the selected neurons into zs."""
    import contextlib
    torch = T_()

    @contextlib.contextmanager
    def cm():
        hk = RC.Hooks(model); hk.capture(blocks, cpu=False); _CAP["h"] = {}
        buf = {}
        tap = KB.Tap(model, lambda l, z, den: buf.__setitem__(l, z[..., torch.as_tensor(sel[l], device=z.device)].float()) if l in sel else None)
        try:
            yield
            _CAP["h"] = {l: hk.states[l][0] for l in blocks}
            if sel:
                zs.append(torch.cat([buf[l] for l in sorted(sel)], -1))
        finally:
            hk.remove(); tap.__exit__()
    return cm()


class _AddHook:
    def __init__(self, model, l, v):
        self.h = RC.layers_of(model)[l].register_forward_hook(lambda m, a, o: ((o[0] + v.to(o[0].device, o[0].dtype),) + tuple(o[1:]))
                                                              if isinstance(o, tuple) else o + v.to(o.device, o.dtype))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.h.remove()


def states(ctx):
    return load_pt(up(ctx, "r1b_K_S2a", "k/k4/states.pt"))


# ---------------------------------------------------------------------------------------------------------------------------------
# Running cells
# ---------------------------------------------------------------------------------------------------------------------------------
def score(E, tokens, it, W=(), OWN=()):
    p = LN.pieces(E["tok"])
    return KL.score_chain([p[t] for t in tokens], it, W, OWN, tokens=list(tokens))


def run_cells(ctx, E, cells, J=16, cb=None, **kw):
    """cells: dicts (item, block, ptype, variant, H [D], M [D], seeds [k], target item (scoring), W, OWN, extra keys); batched by
    block; one chain per seed. Returns chain records (claims scored against the target item)."""
    torch = T_()
    out = []; lens = E["lens"]; bs = N(ctx, 16, 2)
    byb = {}
    for c in cells:
        byb.setdefault((c["block"], c.get("variant_lens", E["variant"])), []).append(c)
    for (blk, var), cs in byb.items():
        for i in range(0, len(cs), bs):
            cb_ = cs[i:i + bs]; Hs, Ms, sd, own_ = [], [], [], []
            for c in cb_:
                h, m = c["H"].float(), c["M"].float()
                if var == "L":
                    h = E["n26"] * KB.T_().nn.functional.normalize(LN.transport(E["J"], E["conv"], blk, h[None].to(ctx.device))[0].cpu(), dim=0)
                    m = E["n26"] * KB.T_().nn.functional.normalize(LN.transport(E["J"], E["conv"], blk, m[None].to(ctx.device))[0].cpu(), dim=0)
                for s in c["seeds"]:
                    Hs.append(h); Ms.append(m); sd.append(int(s)); own_.append(c)
            kw2 = dict(kw); kw2.pop("forced", None)
            if any("forced" in c for c in cb_):
                kw2["forced"] = [c.get("forced", []) for c in own_]
            if "keys_fn" in kw2:
                kw2["keys"] = [kw2["keys_fn"](c, s) for c, s in zip(own_, sd)]; kw2.pop("keys_fn")
            if "reinj_fn" in kw2:
                kw2["reinj"] = torch.stack([kw2["reinj_fn"](c) for c in own_]).to(ctx.device); kw2.pop("reinj_fn")
            r = lens.rollout(blk, torch.stack(Hs).to(ctx.device), torch.stack(Ms).to(ctx.device), sd, J=J, variant=var,
                             bias=E["dbias"], **kw2)
            for k, c in enumerate(own_):
                tk = r["tok"][k].tolist(); sc = score(E, [sd[k]] + tk, c["target"], c.get("W", ()), c.get("OWN", ()))
                rec = dict(item=c["item"], block=blk, ptype=c["ptype"], variant=c["variant"], seed=sd[k], seed_rank=c["seeds"].index(sd[k]),
                           tok=tk, claims=sc["claims"], censor=sc["censor"], repeats=sc["repeats"], **{x: c[x] for x in c.get("carry", ())})
                for f in ("lpr", "lpp", "e", "ent", "cdm", "top1", "alt", "fov"):
                    if f in r:
                        rec[f] = r[f][k].half() if f != "alt" else r[f][k]
                rec["aX"] = {nm: v[k].half() for nm, v in r["aX"].items()}
                rec["share"] = {nm: v[k].half() for nm, v in r["share"].items()}
                out.append(rec)
            if cb:
                cb(r, own_, sd)
    return out


def k4_cells(ctx, E, S):
    """The job list of K4 (deterministic order): V1 / V2 (all blocks, t0 and t1), V3 / V4 (14, 18, t0), V7 (correct, t0, 14, 18),
    V8 (128 correct, 14, 18), decoding grid and temperature arm (128 correct x {14, 18} x t0), oracle (correct, t0, 14, 18, top seed)."""
    I = _items(ctx); St, F = S["St"], S["foreign"]; jobs = []; neutral = E["b"]["frames"]["neutral_ids"][:N(ctx, 10, 3)]
    n128 = N(ctx, 128, 4)
    for ptype, s in St.items():
        for l in S["blocks"]:
            for k, i in enumerate(s["items"]):
                f = F[ptype][k]; tgt = I["items"][i]; G = I["gate"][i]
                base = dict(item=i, block=l, ptype=ptype, target=tgt, W=G["W"], OWN=G["own"], H=s["h"][l][k], M=s["mu"][l][k])
                jobs.append(dict(base, variant="V1", seeds=s["seeds"][l][k].tolist(), dla=True))
                jobs.append(dict(base, variant="V2", H=s["h"][l][f], M=s["mu"][l][f], seeds=s["seeds"][l][f].tolist(), W=G["W"]))
                if ptype == "t0" and l in (14, 18):
                    jobs.append(dict(base, variant="V3", seeds=s["seeds"][l][f].tolist()))
                    jobs.append(dict(base, variant="V4", seeds=list(neutral)))
                    corr = k < len(I["correct"])
                    if corr and l in S["v7"]:
                        jobs.append(dict(base, variant="V7", H=S["v7"][l][k], seeds=s["seeds"][l][k].tolist()))
                    if corr and k < n128:
                        mg = E["R"]["mu_gen"][l].float()
                        jobs.append(dict(base, variant="V1_mugen", M=mg, seeds=s["seeds"][l][k].tolist()))
                        jobs.append(dict(base, variant="V5", H=mg, M=mg, seeds=s["seeds"][l][k].tolist(), kw=dict(greedy=True)))
                        if S["beta"].get(l, {}).get("beta") is not None:
                            jobs.append(dict(base, variant="V8", seeds=s["seeds"][l][k].tolist(), reinj=S["beta"][l]["beta"]))
                        for nm, kw in (("G_lam0.5", dict(lam=0.5)), ("G_lam2", dict(lam=2.0)), ("G_pl0.05", dict(plaus=0.05)),
                                       ("G_pl0.3", dict(plaus=0.3)), ("G_greedy", dict(greedy=True)), ("T0.5", dict(T=0.5)),
                                       ("T1.0", dict(T=1.0))):
                            jobs.append(dict(base, variant=nm, seeds=s["seeds"][l][k].tolist(), kw=kw))
                            if nm.startswith("T"):
                                jobs.append(dict(base, variant=nm + "_V2", H=s["h"][l][f], M=s["mu"][l][f], seeds=s["seeds"][l][f].tolist(), kw=kw))
                        dig = lambda v: [E["tok"].convert_tokens_to_ids("Ġ")] + [E["tok"].convert_tokens_to_ids(c) for c in str(v)]
                        ftgt = I["items"][s["items"][f]]
                        jobs.append(dict(base, variant="ORACLE", seeds=s["seeds"][l][k].tolist()[:1], forced=sum((dig(v) for v in tgt["I"]), [])[:15]))
                        jobs.append(dict(base, variant="ORACLE_F", seeds=s["seeds"][l][k].tolist()[:1], forced=sum((dig(v) for v in ftgt["I"]), [])[:15]))
    return jobs


def _group_kw(jobs):
    g = {}
    for j in jobs:
        k = (j["variant"], repr(sorted(j.get("kw", {}).items())), "reinj" in j, bool(j.get("dla")), "forced" in j)
        g.setdefault(k, []).append(j)
    return g


def k4_run(ctx, i, n):
    """A share of the K4 jobs (part i of n)."""
    E = env(ctx); S = states(ctx); E["n26"] = S["n26"]; torch = T_()
    jobs = KB.parts(k4_cells(ctx, E, S), i, n); rows = []
    for key, js in _group_kw(jobs).items():
        kw = dict(js[0].get("kw", {}))
        if kw.get("T"):
            kw["keys_fn"] = lambda c, s, v=key[0]: ctx.key("r1bK_K4T", c["item"], c["block"], s, v)
        if "reinj" in js[0]:
            kw["reinj_fn"] = lambda c: c["reinj"] * (c["H"] - c["M"]).float()
        rows += run_cells(ctx, E, js, J=N(ctx, 16, 8), dla=bool(js[0].get("dla")), **kw)
    save(ctx, f"k/k4/part{i}.pt", dict(rows=rows))


def k4_merge(ctx, n):
    """Verifier (thresholds from V2 and token-shuffled V1 chains; GVRA per cell; planted error rates), break-off steps, b*, G-K7,
    the self-consistency KL on V1 t0 chains at 14 and 18, and the SO-LENSOUT export."""
    import r1b_k_gvra as GV
    torch = T_()
    E = env(ctx); S = states(ctx); I = _items(ctx); lens, tok, model = E["lens"], E["tok"], E["model"]
    rows = KB.merge_parts(ctx, "k4", n); g = np.random.default_rng(52)
    hidx = {(p, i): k for p, s in S["St"].items() for k, i in enumerate(s["items"])}
    by = {}
    for r in rows:
        by.setdefault((r["variant"], r["block"], r["ptype"]), []).append(r)
    thr, vs = {}, {}
    for (v, l, p), rs in by.items():
        if v not in ("V1", "V2"):
            continue
        Hs = [S["St"][p]["h"][l][hidx[(p, r["item"])]] for r in rs]; Ms = [S["St"][p]["mu"][l][hidx[(p, r["item"])]] for r in rs]
        for L_ in (4, 8, 12, 16):
            ch = [[r["seed"]] + r["tok"][:L_ - 1] for r in rs]
            vs[(v, l, p, L_)] = lens.verify(l, ch, Hs, Ms)
            if v == "V1":
                sh = [list(g.permutation(c)) for c in ch]; vs[("SH", l, p, L_)] = lens.verify(l, sh, Hs, Ms)
    for (v, l, p, L_), sc in vs.items():
        if v == "V1":
            nul = np.r_[vs.get(("V2", l, p, L_), []), vs.get(("SH", l, p, L_), [])]
            thr[(l, p, L_)] = (float(np.percentile(nul, 80)), float(np.percentile(nul, 95))) if len(nul) else (np.inf, np.inf)
    for (v, l, p), rs in by.items():
        if v != "V1":
            continue
        for L_ in (4, 8, 12, 16):
            q80, q95 = thr[(l, p, L_)]
            for r, s in zip(rs, vs[("V1", l, p, L_)]):
                r.setdefault("ver", {})[L_] = float(s)
                if "k_ver" not in r and s <= q80:
                    r["k_ver"] = L_
    gv = {}
    for (v, l, p), rs in by.items():                    # GVRA per cell: seeds in rank order; minor -> one revision by the alt token
        if v != "V1":
            continue
        cells = {}
        for r in rs:
            cells.setdefault(r["item"], []).append(r)
        q80, q95 = thr[(l, p, 16)]

        def gen(u, a, cells=cells):
            c = sorted(cells[u], key=lambda r: r["seed_rank"])
            return c[a] if a < len(c) else None

        def ver(u, r):
            s = r["ver"][16] if "rev" not in r else r["rev"]
            return ("correct" if s > q95 else ("minor fix needed" if s > q80 else "critically flawed")), s

        def rev(u, r, a, l=l, p=p):
            e = r["e"].float().numpy(); k = int(np.argmin(e)); t = list(r["tok"]); t[k] = int(r["alt"][k])
            k_ = hidx[(p, u)]
            r2 = dict(r); r2["rev"] = float(lens.verify(l, [[r["seed"]] + t[:15]], [S["St"][p]["h"][l][k_]], [S["St"][p]["mu"][l][k_]])[0])
            return r2
        res = GV.gvra(list(cells), gen, ver, rev, limit=3)
        gv[f"{l}:{p}"] = dict(summary=GV.summary(res), accepted={u: v_["status"] for u, v_ in res.items()})
    # planted known cases (64): the block-l state at the last digit of a digit-free sentence ending in a known 3-digit number
    pl = E["b"]["wt"]["planted"][:N(ctx, 64, 6)]; err = {}
    if len(pl) >= 2:
        for l in (14, 18):
            if l not in S["blocks"]:
                continue
            Hs = []
            for x in pl:
                with RC.Hooks(model) as hk, torch.no_grad():
                    hk.capture([l]).stop_after(l); hk.run(lambda: model.model(input_ids=torch.tensor([x["ids"]], device=ctx.device)))
                    Hs.append(hk.states[l][0][0, -1])
            mu = torch.stack(Hs).mean(0); ch = [x["ids"][-8:] for x in pl]
            st = lens.verify(l, ch, Hs, [mu] * len(Hs)); sf = lens.verify(l, ch[1:] + ch[:1], Hs, [mu] * len(Hs))
            q95 = thr.get((l, "t0", 16), (np.inf, np.inf))[1]
            err[l] = GV.error_rates(list(st > q95), list(sf > q95))
    sens = np.mean([e["sensitivity"] for e in err.values()]) if err else float("nan")
    fa = np.mean([e["false_accept"] for e in err.values()]) if err else float("nan")
    gk6 = "verifier reliable" if (KB.fin(sens) and KB.fin(fa) and fa <= 0.20 and sens >= 0.50) else "verifier unreliable"
    # b*: the block in {14, 18} with the larger V1 - V2 any-hit (t0, correct items; ties -> 14)
    nc = set(I["correct"])

    def anyhit(v, l):
        hits = {}
        for r in by.get((v, l, "t0"), []):
            if r["item"] in nc:
                hits[r["item"]] = hits.get(r["item"], False) or any(c == "TRUE-INT" for _, _, c in r["claims"])
        return np.mean(list(hits.values())) if hits else 0.0
    d = {l: anyhit("V1", l) - anyhit("V2", l) for l in (14, 18) if l in S["blocks"]}
    bstar = 14 if (not d or d.get(14, -9) >= d.get(18, -9)) else 18
    # G-K7 (block 22, t0): J-lens top-10 vs logit-lens top-10 containing the model's own next token (the greedy answer's first token)
    l22 = max(b_ for b_ in S["blocks"] if b_ <= 22); s0 = S["St"]["t0"]; jl, ll = [], []
    with torch.no_grad():
        for k, i in enumerate(s0["items"]):
            nt = I["gate"][i]["greedy"][0] if I["gate"][i]["greedy"] else -1
            jl.append(nt in s0["seeds"][l22][k].tolist())
            lg = KB.unembed(model, s0["h"][l22][k][None].to(ctx.device))[0]; ll.append(nt in lg.topk(10).indices.tolist())
    gk7 = None if np.mean(jl) >= np.mean(ll) else "J-lens seeds weak on task text"
    # self-consistency KL (Teoh 2026): V1, t0, blocks 14 and 18, top-4 seeds
    skl = []
    for l in (14, 18):
        rs = [r for r in by.get(("V1", l, "t0"), []) if r["seed_rank"] < 4]
        for b0 in range(0, len(rs), N(ctx, 16, 4)):
            rb = rs[b0:b0 + N(ctx, 16, 4)]; kk = [hidx[("t0", r["item"])] for r in rb]
            H_ = torch.stack([S["St"]["t0"]["h"][l][k] for k in kk]).to(ctx.device)
            if E["variant"] == "L":
                H_ = S["n26"] * torch.nn.functional.normalize(LN.transport(E["J"], E["conv"], l, H_), dim=-1)
            lp = lens.tf_logp(l, H_, [r["seed"] for r in rb], [r["tok"] for r in rb], E["variant"])
            lq = LN.backbone_logq(model, [RC.chat_ids(tok, I["items"][r["item"]]["prompt"]) for r in rb], [r["seed"] for r in rb], [r["tok"] for r in rb])
            kl = LN.kl_rows(lq, lp).cpu()
            for r, k_ in zip(rb, kl):
                skl.append(dict(item=r["item"], block=l, seed_rank=r["seed_rank"], kl=k_.tolist()))
    for r in rows:
        r.pop("ver_tmp", None)
    save(ctx, "k/k4/summary.pt", dict(rows=rows, thr={f"{k}": v for k, v in thr.items()}, gvra=gv, planted=err, selfkl=skl))
    write(ctx, "k/k4/b_star.json", dict(b_star=bstar, anyhit_diff=d, G_K6=dict(string=gk6, sensitivity=val(sens), false_accept=val(fa)),
                                        G_K7=gk7, variant=E["variant"], variant_label=E["variant_label"], HA_label=E["HA_label"]),
          labels=[x for x in (gk7, E["variant_label"], E["HA_label"]) if x])
    lensout(ctx, E, rows, I)


def lensout(ctx, E, rows, I):
    """SO-LENSOUT: one record per K4 chain (V1 / V2, all cells): item, stratum, ops, variant, block, position type, seed rank, key,
    emitted ids s = 0..16, k*_conf, share_H(s) with the RS39 band (q95 over RS sets), a_H(s)."""
    import json
    lines = []
    for r in rows:
        if r["variant"] not in ("V1", "V2"):
            continue
        kc = next((s for s, v, c in r["claims"] if c == "CONFAB"), None)
        rsn = [k for k in r["share"] if k.startswith("RS")]
        band = (np.percentile(np.stack([r["share"][k].float().numpy() for k in rsn]), 95, axis=0).tolist() if rsn else None)
        lines.append(json.dumps(RC.jsonable(dict(item=r["item"], stratum="correct" if r["item"] in set(I["correct"]) else "wrong",
                                                 ops=I["items"][r["item"]]["nops"], variant=r["variant"], block=r["block"], ptype=r["ptype"],
                                                 seed_rank=r["seed_rank"], key=None, tokens=[r["seed"]] + r["tok"],
                                                 k_star_conf=kc if kc is not None else coded("CENSORED:above_grid"),
                                                 share_H=r["share"].get("H"), share_H_band_q95=band, a_H=r["aX"].get("H")))))
    ctx.write_file("lensout/so_lensout.jsonl", "\n".join(lines) + "\n", config=dict(n=len(lines)), labels=["MOCK"] if ctx.mock else [])


# ---------------------------------------------------------------------------------------------------------------------------------
def k4p(ctx):
    """K4p: cross-fitted (5 item-blocked folds) ridge classifier from h_l(t0) to the last digit of v1 (10 classes), lambda by inner CV;
    operand-feature baseline; 200 label permutations within template. Per-item correctness indicators are stored (ana bootstraps)."""
    torch = T_()
    E = env(ctx); tok, model = E["tok"], E["model"]; b = E["b"]
    its = [x for x in b["ach_short"] if x["split"] in ("A-scr", "A-lens") and not x.get("cal")][:N(ctx, 4200, 60)]
    blocks = [l for l in BLOCKS if l < E["lens"].nL - 1]; Hs = {l: [] for l in blocks}
    for b0, idt, am, pad, _ in KB.tf_batches(model, [RC.chat_ids(tok, x["prompt"]) for x in its], tok.pad_token_id, batch=N(ctx, 64, 16),
                                              logits=False, setup=lambda b0, B: _cap(model, blocks, {}, [])):
        for l in blocks:
            Hs[l].append(_CAP["h"][l][:, -1].double().cpu())
    y = np.array([x["I"][0] % 10 for x in its]); y2 = np.array([(x["I"][1] if len(x["I"]) > 1 else x["A"]) % 10 for x in its])
    tmpl = np.array([x["template"] for x in its]); folds = np.array([H(x["id"]) % 5 for x in its])
    base = np.array([[int(c == d) for p in x["P"][:4] + [0] * (4 - len(x["P"][:4])) for c in str(p).zfill(2) for d in "0123456789"] +
                     [int(o == s) for o in (x["ops"] + ["+"] * 3)[:3] for s in "+-*"] for x in its], float)

    def ridge_cv(X, yy, lam_grid=(1e-1, 1e1, 1e3)):
        pred = np.zeros(len(yy), int); Y = np.eye(10)[yy]
        for f in range(5):
            tr, te = folds != f, folds == f
            best, bl = -1, lam_grid[0]
            for lam in lam_grid:                       # inner CV (2 folds inside the training part)
                acc = []
                for g2 in range(2):
                    itr = tr & ((np.arange(len(yy)) % 2) != g2); ite = tr & ((np.arange(len(yy)) % 2) == g2)
                    acc.append((_ridge(X[itr], Y[itr], lam, X[ite]).argmax(1) == yy[ite]).mean() if ite.any() else 0)
                if np.mean(acc) > best:
                    best, bl = np.mean(acc), lam
            pred[te] = _ridge(X[tr], Y[tr], bl, X[te]).argmax(1)
        return (pred == yy).astype(float)
    res = {}
    gperm = np.random.default_rng(1301); nperm = N(ctx, 200, 5)
    for l in blocks:
        X = torch.cat(Hs[l]).numpy(); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        cp, cb = ridge_cv(X, y), ridge_cv(base, y); cp2 = ridge_cv(X, y2)
        perms = []
        for _ in range(nperm):
            yp = y.copy()
            for t in np.unique(tmpl):
                m = tmpl == t; yp[m] = gperm.permutation(yp[m])
            perms.append(ridge_cv(X, yp).mean() - cb.mean())
        res[l] = dict(correct_probe=cp, correct_base=cb, correct_probe_v2=cp2, perm=np.array(perms))
    save(ctx, "k/k4p.pt", dict(res=res, items=[x["id"] for x in its]))


def _ridge(X, Y, lam, Xt):
    """Ridge forward map (# §2.5 fitting step: K4p's ridge, K10c's L-par; a fitted forward map, not a reconstruction)."""
    n, d = X.shape
    if n < d:
        A = X @ X.T + lam * np.eye(n); return Xt @ (X.T @ np.linalg.solve(A, Y))
    A = X.T @ X + lam * np.eye(d); return Xt @ np.linalg.solve(A, X.T @ Y)


# ---------------------------------------------------------------------------------------------------------------------------------
def k6c_part(ctx, i, n):
    """K6c: 128 correct items x {14, 18}, t0, V1, top-4 seeds; (i) rollout mean-clamp of H_A in both streams at every chain position,
    (ii) read-state mean-ablation of H_A's direct writes (layers <= l); RS39mc analogues; CSS at the first chain position."""
    torch = T_()
    E = env(ctx); S = states(ctx); E["n26"] = S["n26"]; I = _items(ctx); R = E["R"]; model = E["model"]
    rsmc = E["rs"]["RS39mc"]; conds = [("none", None, None)] + [(f"{k}_{nm}", k, s) for nm, s in [("H", E["HA"])] + [(f"MC{j}", x) for j, x in enumerate(rsmc)]
                                                                   for k in ("clamp", "ablate")]
    conds = KB.parts(conds, i, n) if i is not None else conds
    if ("none", None, None) not in conds:
        conds = [("none", None, None)] + conds
    s0 = S["St"]["t0"]; items = I["correct"][:N(ctx, 128, 4)]; uni = [tuple(x) for x in S["union"]]; pos = {u: k for k, u in enumerate(uni)}
    Ls = RC.layers_of(model); rows = []
    for l in (14, 18):
        if l not in S["blocks"]:
            continue
        base = []
        for k, it in enumerate(s0["items"]):
            if it in items:
                base.append((k, it))
        firsts = {}
        for nm, kind, st in conds:
            cells = []
            for k, it in base:
                h = s0["h"][l][k].clone()
                if kind == "ablate":
                    for (ll, j) in [tuple(x) for x in st if x[0] <= l]:
                        zj = float(s0["z"][k, pos[(ll, j)]]) if (ll, j) in pos else 0.0
                        h -= (zj - float(R["z_mu"][ll, j])) * Ls[ll].mlp.down_proj.weight[:, j].float().cpu()
                cells.append(dict(item=it, block=l, ptype="t0", variant=nm, target=I["items"][it], W=[], OWN=I["gate"][it]["own"], H=h,
                                  M=s0["mu"][l][k], seeds=s0["seeds"][l][k].tolist()[:4]))
            ops = [dict(sel=by_layer(st), kind="clamp", value=set_values(st, R["z_mu"]), when=lambda T, sd: (sd["p0"] + T_().arange(T)) >= 0)] \
                if kind == "clamp" and st else None
            css = []

            def cb(r, own_, sd, nm=nm):
                f = r["first"]
                if nm == "none":
                    firsts[len(firsts)] = f
                else:
                    f0 = firsts[len(css)]; pl = f0["plaus"]
                    d1 = ((f["lr"] - f["lp"]) - (f0["lr"] - f0["lp"])).masked_fill(~pl, 0); d2 = (f["lr"] - f0["lr"]).masked_fill(~pl, 0)
                    css.append((d1.norm(dim=-1) / d2.norm(dim=-1).clamp(min=1e-12)).tolist())
            rr = run_cells(ctx, E, cells, J=N(ctx, 16, 8), cb=cb, ops=ops, keep_first=True, sets=False, fov=False)
            for r in rr:
                r.pop("aX"); r.pop("share")
            rows.append(dict(cond=nm, block=l, chains=rr, css=sum(css, [])))
    save(ctx, f"k/k6c/part{i}.pt", dict(rows=rows))


def k6f_part(ctx, i, n):
    """K6f: K4's V1 cells at b*, t0, 384 items (256 correct, first 128 wrong) x 10 seeds after a filler: L_P = 528 (F512 + P0) and
    2,064 (F2048 + P0); 8 instances per length (instance by item hash); one B = 1 prefix prefill per instance, cache copies (SO-LCE
    copy path); G-K20 parity in part 0."""
    torch = T_()
    from transformers import DynamicCache
    E = env(ctx); S = states(ctx); E["n26"] = S["n26"]; I = _items(ctx); model, lens = E["model"], E["lens"]
    bstar = RC.load_json(own(ctx, "k/k4/b_star.json"))["b_star"]; s0 = S["St"]["t0"]
    items = [x for x in I["correct"][:N(ctx, 256, 6)] + I["wrong"][:N(ctx, 128, 4)]]; F = E["b"]["frames"]; rows, parity = [], {}
    jobs = [(Lf, inst) for Lf in ("F512", "F2048") for inst in range(8)]
    for Lf, inst in KB.parts(jobs, i, n):
        its = [x for x in items if H(x) % 8 == inst]
        if not its:
            continue
        pre = list(F[Lf][inst]) + list(F["P0_ids"]); cache = DynamicCache()
        with torch.no_grad(), RC.sdpa_efficient(ctx.device):
            model(input_ids=torch.tensor([pre], device=ctx.device), past_key_values=cache, use_cache=True)
        cells = []
        for it in its:
            k = s0["items"].index(it)
            cells.append(dict(item=it, block=bstar, ptype="t0", variant=f"V1_{Lf}", target=I["items"][it], W=I["gate"][it]["W"],
                              OWN=I["gate"][it]["own"], H=s0["h"][bstar][k], M=s0["mu"][bstar][k], seeds=s0["seeds"][bstar][k].tolist(),
                              inst=inst, carry=("inst",)))
        with RC.sdpa_efficient(ctx.device):
            rr = run_cells(ctx, E, cells, J=N(ctx, 16, 8), prefix=(cache, len(pre)), sets=True, fov=False)
        rows += rr
        if Lf not in parity and rr:                      # G-K20: cached path vs full recompute on 8 rows (log p of the chosen tokens)
            r0 = rr[:N(ctx, 8, 2)]; dmax, flips = 0.0, 0
            for r in r0:
                seq = pre + [F["ph_id"], r["seed"]] + r["tok"][:-1]
                k = s0["items"].index(r["item"]); h = s0["h"][bstar][k]
                if E["variant"] == "L":
                    h = S["n26"] * torch.nn.functional.normalize(LN.transport(E["J"], E["conv"], bstar, h[None].to(ctx.device))[0], dim=0)
                pb = bstar if E["variant"] == "E" else lens.nL - 2
                hk = lens._patch(pb, h[None].to(ctx.device), len(pre))
                try:
                    with torch.no_grad(), RC.sdpa_efficient(ctx.device):
                        lg = model(input_ids=torch.tensor([seq], device=ctx.device)).logits[0, len(pre) + 1:].float().log_softmax(-1)
                finally:
                    hk.remove()
                lp_full = lg.gather(1, torch.tensor(r["tok"], device=lg.device)[:, None])[:, 0].cpu()
                dmax = max(dmax, float((lp_full - r["lpr"].float()).abs().max())); flips += int((lg.argmax(-1).cpu() != torch.tensor(r["tok"])).sum())
            parity[Lf] = dict(max_abs_dlogp=dmax, flip_rate=flips / max(1, sum(len(r["tok"]) for r in r0)),
                              ok=bool(dmax < (1e-4 if not ctx.mock else 1e-3)), note="contrastive argmax flips are not Gumbel flips")
    for r in rows:
        r["share"] = {}
    save(ctx, f"k/k6f/part{i}.pt", dict(rows=rows, parity=parity, b_star=bstar))


def k6g(ctx):
    """K6g: forced segments (correct v1 / wrong w / neutral, matched token counts) right after the seed in both streams, then 9 free
    steps; block b*, top-4 seeds, gated-correct items with a 2- or 3-digit first intermediate (<= 400)."""
    E = env(ctx); S = states(ctx); E["n26"] = S["n26"]; I = _items(ctx); tok = E["tok"]; s0 = S["St"]["t0"]
    bstar = RC.load_json(own(ctx, "k/k4/b_star.json"))["b_star"]; g = np.random.default_rng(1202)
    cand = [i for i, v in I["gate"].items() if v["stratum"] == "correct" and i in s0["items"] and 10 <= I["items"][i]["I"][0] <= 999][:N(ctx, 400, 6)]
    sp, digs = tok.convert_tokens_to_ids("Ġ"), {c: tok.convert_tokens_to_ids(c) for c in "0123456789"}
    andso = E["b"]["frames"]["andso_ids"]; cells = []
    for it in cand:
        x = I["items"][it]; v1 = x["I"][0]; nd = len(str(v1)); excl = set(x["P"]) | set(x["I"]) | {x["A"]} | set(I["gate"][it]["own"])
        while True:
            w = int(g.integers(10 ** (nd - 1), 10 ** nd))
            if str(w)[0] != str(v1)[0] and w not in excl:
                break
        k = s0["items"].index(it)
        for cond, f in (("correct", [sp] + [digs[c] for c in str(v1)]), ("wrong", [sp] + [digs[c] for c in str(w)]),
                        ("neutral", andso[0] if nd == 2 else andso[1])):
            cells.append(dict(item=it, block=bstar, ptype="t0", variant=cond, target=x, W=[], OWN=I["gate"][it]["own"], H=s0["h"][bstar][k],
                              M=s0["mu"][bstar][k], seeds=s0["seeds"][bstar][k].tolist()[:4], forced=list(f), w=w, carry=("w",)))
    rows = []
    for cond in ("correct", "wrong", "neutral"):
        cs = [c for c in cells if c["variant"] == cond]
        if cs:
            rows += run_cells(ctx, E, cs, J=max(len(c["forced"]) for c in cs) + N(ctx, 9, 4), sets=True, fov=False)
    for r in rows:
        r["share"] = {k: v for k, v in r["aX"].items() if k == "H"}; r["aX"] = {"H": r["aX"].get("H")}
    save(ctx, "k/k6g.pt", dict(rows=rows, n_items=len(cand), b_star=bstar,
                               G_K19=("reduced design" if len(cand) < N(ctx, 150, 4) else None) if len(cand) >= N(ctx, 75, 2) else "not testable (too few items)"))


def k6h_part(ctx, i, n):
    """K6h: QA lens on S_lens (200 CC + 200 CI): h_l at the last prompt token, blocks {14, 18}, CL17 V1 and V2 (a foreign item of the
    same class), 10 seeds, J = 12; mentions scored by the CONFAB-QA parser (GOLD, OWN = the modal screening answer, PROMPT,
    CONFAB-QA); the K4 per-step channels."""
    torch = T_()
    E = env(ctx); S = states(ctx); E["n26"] = S["n26"]; tok, model = E["tok"], E["model"]
    lab = S1.alloc(ctx); byq = {r["qid"]: r for r in E["b"]["tqa"]}
    its = [(q, v) for q, v in lab["items"].items() if v["split"] == "S_lens"]
    its = KB.parts(its, i, n); blocks = [l for l in (14, 18) if l < E["lens"].nL - 1]
    if not its:
        save(ctx, f"k/k6h/part{i}.pt", dict(rows=[])); return
    seqs = [S1.tqa_prompt(tok, byq[q]["question"]) for q, _ in its]; Hs = {l: [] for l in blocks}
    for b0, idt, am, pad, _ in KB.tf_batches(model, seqs, tok.pad_token_id, batch=N(ctx, 32, 8), logits=False, setup=lambda b0, B: _cap(model, blocks, {}, [])):
        for l in blocks:
            Hs[l].append(_CAP["h"][l][:, -1].cpu())
    rows = []; pcs = LN.pieces(tok)
    for l in blocks:
        h = torch.cat(Hs[l]); mu = (h.sum(0, keepdim=True) - h) / max(len(h) - 1, 1)
        seeds = LN.jlens_seeds(model, tok, E["J"], E["conv"], l, h.to(ctx.device), k=N(ctx, 10, 3))
        cls = [v["cls"] for _, v in its]; f = np.arange(len(its))
        for c in ("CC", "CI"):
            ix = [k for k in range(len(its)) if cls[k] == c]
            if len(ix) > 1:
                f[ix] = np.roll(ix, 1)
        for var in ("V1", "V2"):
            cells = []
            for k, (q, v) in enumerate(its):
                kk = k if var == "V1" else int(f[k])
                cells.append(dict(item=q, block=l, ptype="qa", variant=var, target=dict(P=[], I=[], A=-1), H=h[kk], M=mu[kk],
                                  seeds=seeds[kk].tolist(), cls=v["cls"], carry=("cls",)))
            rr = run_cells(ctx, E, cells, J=N(ctx, 12, 6), sets=True, fov=True)
            for r in rr:
                q = r["item"]; text_p = [pcs[t] for t in [r["seed"]] + r["tok"]]; txt = "".join(text_p)
                owner = sum(([k] * len(p) for k, p in enumerate(text_p)), [])
                ms = KL.mentions(txt, byq[q]["question"], byq[q]["aliases"], [lab["items"][q]["modal"]], E["b"]["capwords"])
                cens = KL.loop_onset([r["seed"]] + r["tok"], [])
                r["mentions"] = [(owner[s] if s < len(owner) else len(owner) - 1, m, c) for s, m, c in ms if cens is None or owner[s] < cens]
                r["share"] = {}
            rows += rr
    save(ctx, f"k/k6h/part{i}.pt", dict(rows=rows))


def k10c(ctx):
    """K10c: per-digit accuracy on d_1..d_3 of the own greedy answer at t0 of 240 correct K4 items: CL17 (the top-ranked seed's chain;
    answer-free selection, finding validity-11; the d_1-selected chain is descriptive), L-par(k) (ridge from h_l(t0) to h_27 at the
    target's offset - 1, fitted on A-scr answer positions with float64 accumulators) and L-jac(Delta) (central difference along h of
    h_27(t0 + Delta) on the item's own answer text); blocks {14, 18}."""
    torch = T_()
    E = env(ctx); S = states(ctx); I = _items(ctx); tok, model = E["tok"], E["model"]; s0 = S["St"]["t0"]; nL = E["lens"].nL
    rows4 = load_pt(own(ctx, "k/k4/summary.pt"))["rows"]
    items = [i for i in I["correct"] if i in s0["items"]][:N(ctx, 240, 6)]
    dig = lambda ids: [t for t in ids if tok.decode([t]).strip().isdigit()]
    scr = [x for x in E["b"]["ach_short"] if x["split"] == "A-scr"][:N(ctx, 3000, 40)]
    P = [RC.chat_ids(tok, x["prompt"]) for x in scr]; G = KB.decode(model, tok, P, max_new=8, batch=N(ctx, 128, 16))
    blocks = [l for l in (14, 18) if l < nL - 1]; out = {}
    for l in blocks:
        X, Y = {k: [] for k in (1, 2, 3)}, {k: [] for k in (1, 2, 3)}
        seqs = [p + g for p, g in zip(P, G)]
        for b0, idt, am, pad, _ in KB.tf_batches(model, seqs, tok.pad_token_id, batch=N(ctx, 32, 8), logits=False,
                                                  setup=lambda b0, B: _cap(model, [l, nL - 1], {}, [])):
            for r in range(idt.shape[0]):
                p, g = P[b0 + r], G[b0 + r]; o = int(pad[r]); dpos = [k for k, t in enumerate(g) if tok.decode([t]).strip().isdigit()][:3]
                for k, dp in enumerate(dpos, 1):
                    X[k].append(_CAP["h"][l][r, o + len(p) - 1].double().cpu().numpy()); Y[k].append(_CAP["h"][nL - 1][r, o + len(p) + dp - 1].double().cpu().numpy())
        W = {}
        for k in (1, 2, 3):
            if len(X[k]) >= 4:
                Xa, Ya = np.array(X[k]), np.array(Y[k]); m = len(Xa) * 4 // 5; best = None
                for lam in (1e0, 1e2, 1e4):
                    pr = _ridge(Xa[:m], Ya[:m], lam * len(Xa), Xa[m:]); err = ((pr - Ya[m:]) ** 2).mean() if len(Xa) > m else 0
                    best = (err, lam) if best is None or err < best[0] else best
                W[k] = (Xa, Ya, best[1])
        res = []
        for it in items:
            k_ = s0["items"].index(it); g = I["gate"][it]["greedy"]; d = dig(g)[:3]; p = RC.chat_ids(tok, I["items"][it]["prompt"])
            ch = [r for r in rows4 if r["variant"] == "V1" and r["block"] == l and r["ptype"] == "t0" and r["item"] == it]
            top = min(ch, key=lambda r: r["seed_rank"]) if ch else None
            sel1 = next((r for r in ch if d and r["seed"] == d[0]), top)
            rd = lambda r: dig([r["seed"]] + r["tok"])[:3] if r else []
            h = s0["h"][l][k_].double().numpy(); lpar, ljac = [], []
            dpos = [k for k, t in enumerate(g) if tok.decode([t]).strip().isdigit()][:3]
            for k, dp in enumerate(dpos, 1):
                if k in W:
                    Xa, Ya, lam = W[k]; y = _ridge(Xa, Ya, lam * len(Xa), h[None])[0]
                    lpar.append(int(KB.unembed(model, torch.tensor(y, dtype=torch.float32, device=ctx.device)[None])[0].argmax()))
                else:
                    lpar.append(-1)
                seq = p + g; tpos = len(p) + dp - 1; src = len(p) - 1; hs = []
                for sg in (1, -1):
                    hk = RC.layers_of(model)[l].register_forward_hook(lambda m, a, o, sg=sg: _scale_at(o, src, 1 + sg * 1e-2))
                    try:
                        with RC.Hooks(model) as hc, torch.no_grad():
                            hc.capture([nL - 1], cpu=False); model.model(input_ids=torch.tensor([seq], device=ctx.device)); hs.append(hc.states[nL - 1][0][0, tpos])
                    finally:
                        hk.remove()
                tau = (hs[0] - hs[1]) / 0.02
                ljac.append(int(KB.unembed(model, tau[None])[0].argmax()))
            res.append(dict(item=it, d=d, cl_top=rd(top), cl_d1sel=rd(sel1), lpar=lpar, ljac=ljac))
        out[l] = res
    save(ctx, "k/k10c.pt", dict(res=out))


def _scale_at(o, pos, f):
    x = o[0] if isinstance(o, tuple) else o
    x = x.clone(); x[:, pos] = x[:, pos] * f
    return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x


def kref_u32(ctx):
    """SO-DMG reporting: damage on REF-U32 of r1b_K_S1's calibrations (H_op at alpha 0 and 3, mean-clamp; RS39 at alpha 0 / 3,
    RS39mc mean-clamp) and of K4's V8 beta (descriptive)."""
    E = env(ctx); model = E["model"]; R = E["R"]; S = states(ctx); su = {}
    for nm, st in [("H_op", E["hs"]["H_op"])] + [(f"RS39_{k}", s) for k, s in enumerate(E["rs"]["RS39"])]:
        if st:
            su[f"{nm}_a0"] = lambda st=st: Interv(model, [dict(sel=by_layer(st), kind="scale", value=0.0)])
            su[f"{nm}_a3"] = lambda st=st: Interv(model, [dict(sel=by_layer(st), kind="scale", value=3.0)])
    for nm, st in [("H_op", E["hs"]["H_op"])] + [(f"RS39mc_{k}", s) for k, s in enumerate(E["rs"]["RS39mc"])]:
        if st:
            su[f"{nm}_mclamp"] = lambda st=st: Interv(model, [dict(sel=by_layer(st), kind="clamp", value=set_values(st, R["z_mu"]))])
    write(ctx, "k/ref_u32_S1.json", dict(damage=KB.ref_u32_damage(ctx, model, su, "r1b_K_S1 calibrations"),
                                         v8_beta={l: v.get("beta") for l, v in S["beta"].items()}), labels=["descriptive (SO-DMG reporting)"])
