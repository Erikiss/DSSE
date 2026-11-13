"""r1b_K_S5 (account A): KS-axes (SO-AX package read and check, G-K11), K3f (style push -> H), K7b (channel-resolved lens throughput),
K9 (channel hierarchy and its reordering), K10a (route split KV vs token at two doses), K10b (QK vs OV routing split, head routing map)."""
import math, os
import numpy as np
import r1b_core as RC
import r1b_k_base as KB
import r1b_k_labels as KL
import r1b_k_lens as LN
import r1b_k_te as TE
from r1b_k_base import N, H, T_, write, save, load_pt, own, up, Interv, by_layer, set_values, SetReader, coded, val
import r1b_k_s1 as S1
import r1b_k_s2 as S2


def _ax_file(ctx, session, rel):
    try:
        p = ctx.input(session, rel, required=False)
    except RC.NotRun:
        p = None
    if not p:
        return None
    d = KB.load_pt(p) if p.endswith(".pt") else RC.load_json(p)
    return None if (not d or (isinstance(d, dict) and d.get("mock"))) else d


def ks_axes(ctx):
    """G-K11: u*_14, u*_18 (U*), u_T^v(14), u_sea^v(14), u_T^v(18), u_sea^v(18) (AX-1b); AX1 fallback with 'round-1 axes (AX-1b
    unavailable)'; without the package: 'axes unavailable'. MOCK: seeded random unit vectors (labelled)."""
    torch = T_()
    tok, model = ctx.model(); D = model.config.hidden_size; ax, lab = {}, []
    chk = _ax_file(ctx, "r1b_R_S0", "so_ax/so_ax_check.json")
    for b in (14, 18):
        p = _ax_file(ctx, "r1b_R_S0", f"so_ax/so_ax_b{b}.pt"); a = _ax_file(ctx, f"r1b_R_S2_{b}", f"ax1b/ax1b_b{b}.pt")
        if p is not None:
            ax[f"ustar{b}"] = torch.as_tensor(p["u_star"]).float()
            if a is not None:
                ax[f"uT{b}"] = torch.as_tensor(a["u_T_v"]).float(); ax[f"usea{b}"] = torch.as_tensor(a["u_sea_v"]).float()
            else:
                ax[f"uT{b}"] = torch.as_tensor(p["u_T_perp"]).float(); ax[f"usea{b}"] = torch.as_tensor(p["u_sea"]).float()
                lab.append(f"round-1 axes (AX-1b unavailable) at {b}")
    if not ax:
        if ctx.mock:
            g = torch.Generator().manual_seed(11)
            ax = {f"{k}{b}": torch.nn.functional.normalize(torch.randn(D, generator=g), dim=0) for k in ("ustar", "uT", "usea") for b in (14, 18)}
            lab.append("MOCK axes (random)")
        else:
            lab.append("axes unavailable")
    if chk is not None and not chk.get("ok", True):
        lab.append("SO-AX reproduction failed")
    save(ctx, "k/ax.pt", dict(ax=ax, labels=lab, G_K11="ok" if ax and not any("unavailable" in x for x in lab) else (lab[0] if lab else "ok")))


def axes(ctx):
    return load_pt(own(ctx, "k/ax.pt"))


def _push_cal(ctx, model, dm, d, block, cap, target):
    """SO-DMG push form: f <- f sqrt(target / damage), push at every position of REF-K16 (x + f min(|x|, cap) unit(d))."""
    return RC.calibrate(lambda f: dm(lambda: RC.Hooks(model).push(block, d, f, cap=cap))["damage"], 0.1, target)


def k3f(ctx):
    """K3f: push along u*_14 at delta1 on continuation positions of 128 S3 answers (T 0.7, keys r1bK_K3f) against 20 random directions
    at equal damage; s_H, a_H and accuracy (descriptive rank)."""
    torch = T_()
    A = axes(ctx)
    if "ustar14" not in A["ax"]:
        raise RC.NotRun("axes unavailable (G-K11)")
    tok, model = ctx.model(); b = KB.bank(ctx); hs = S1.hnset(ctx); R = S1.refs(ctx); dets = S1.detectors(ctx); det = dets.get("H_op")
    S3 = [r for r in b["tqa"] if r["split"] == "S3"][:N(ctx, 128, 4)]; P = [S1.tqa_prompt(tok, r["question"]) for r in S3]
    dm = RC.Damage(model, KB.ref_k16(ctx)); cap = 2 * float(R["mu_gen"][14].norm()); g = torch.Generator().manual_seed(1401)
    dirs = [("ustar14", A["ax"]["ustar14"])] + [(f"rand{k}", torch.randn(model.config.hidden_size, generator=g)) for k in range(N(ctx, 20, 3))]
    out, u32 = [], {}
    for nm, d in dirs:
        cal = _push_cal(ctx, model, dm, d, 14, cap, RC.DELTA1); rd = SetReader({"H": hs["H_op"]}, R["wn"].to(ctx.device), R["c_mu"], R["c_sd"], last=False)
        rs = torch.tensor([len(p) for p in P])
        setup = lambda b0, B, d=d, f=cal["f"]: RC.Hooks(model).push(14, d, f, cap=cap, rowstart=torch.tensor([len(p) for p in P[b0:b0 + B]]) + 0)
        conts = KB.decode(model, tok, P, keys=[ctx.key("r1bK_K3f", r["qid"]) for r in S3], T=0.7, max_new=24, batch=N(ctx, 64, 4), setup=setup)
        acc = float(np.mean([S1.label_tqa(ctx, r, tok.decode(c)) == "correct" for r, c in zip(S3, conts)]))
        rows = [(p, c, S1.span_len(tok, c) or 1) for p, c in zip(P, conts)]
        f = KB.T_().cat([x for x in [S2_cett(model, tok, rows, hs["H_op"])]]) if hs["H_op"] else None
        sH = float(np.mean(f.numpy() @ det["beta"] + det["b"])) if (f is not None and det) else None
        aH = float(((f.numpy() - R["c_mu"].reshape(-1)[_cols(hs["H_op"], R)]) / np.maximum(R["c_sd"].reshape(-1)[_cols(hs["H_op"], R)], 1e-6)).mean()) if f is not None else None
        out.append(dict(dir=nm, f=cal["f"], damage=cal["damage"], cal_labels=cal["labels"], acc=acc, s_H=sH, a_H=aH))
        if nm == "ustar14":
            u32 = KB.ref_u32_damage(ctx, model, {"ustar14_delta1": lambda: RC.Hooks(model).push(14, d, cal["f"], cap=cap)}, "K3f")
    rk = {k: (1 + sum((o[k] or -9e9) >= (out[0][k] or -9e9) for o in out[1:])) for k in ("s_H", "a_H", "acc")}
    write(ctx, "k/k3f.json", dict(rows=out, rank_of_style_push=rk, ref_u32=u32, labels_axes=A["labels"]),
          labels=["descriptive (K3f)"] + A["labels"])


def _cols(neurons, R):
    npl = R["wn"].shape[1]; return [l * npl + j for l, j in neurons]


def S2_cett(model, tok, rows, neurons):
    import r1b_k_hn as HN
    f = HN.cett_features(model, tok, rows, batch=16, sets=by_layer(neurons)); return f["ans"].float()


# ---------------------------------------------------------------------------------------------------------------------------------
def _subspaces(ctx, E, l, A, Xpca):
    """Named channels at block l: H-subspace (write vectors of H with layer <= l, orthonormalised), J-workspace (top-16 input singular
    directions of J_l), top-16 PCs, u*_l, u_T^v(l), u_sea^v(l) (when available)."""
    torch = T_()
    Ls = RC.layers_of(E["model"]); ch = {}
    hw = [Ls[ll].mlp.down_proj.weight[:, j].float().cpu() for ll, j in E["HA"] if ll <= l]
    if hw:
        u, s, _ = torch.linalg.svd(torch.stack(hw, 1), full_matrices=False); ch["Hsub"] = u[:, :16]
    jm = LN.jmap(E["J"], E["conv"], l)
    if jm is not None:
        _, _, v = torch.linalg.svd(jm.float(), full_matrices=False); ch["Jws"] = v[:16].T.cpu()
    xc = Xpca - Xpca.mean(0); _, _, v = torch.linalg.svd(xc, full_matrices=False); ch["pc16"] = v[:16].T
    for k in ("ustar", "uT", "usea"):
        if f"{k}{l}" in A["ax"]:
            ch[k] = torch.nn.functional.normalize(A["ax"][f"{k}{l}"].float(), dim=0)[:, None]
    return ch


def k7b(ctx):
    """K7b: patch mu + P_c (h - mu) (only channel c's part of the deviation) at 128 correct items x {14, 18} x t0, V1, 3 seeds; forward
    hits TRUE-INT or ANSWER, backward hits PROMPT operands; random subspaces 19 x dim 1 and 19 x dim 16; full deviation (ceiling)."""
    torch = T_()
    E = S2.env(ctx); S = S2.states(ctx); E["n26"] = S["n26"]; I = S2._items(ctx); A = axes(ctx); s0 = S["St"]["t0"]
    items = [i for i in I["correct"] if i in s0["items"]][:N(ctx, 128, 4)]; g = torch.Generator().manual_seed(1402); rows = []
    for l in [x for x in (14, 18) if x in S["blocks"]]:
        X = s0["h"][l]; ch = _subspaces(ctx, E, l, A, X); D = X.shape[1]
        ch["full"] = torch.eye(D)
        for k in range(N(ctx, 19, 2)):
            ch[f"r1_{k}"] = torch.nn.functional.normalize(torch.randn(D, 1, generator=g), dim=0)
            ch[f"r16_{k}"] = torch.linalg.qr(torch.randn(D, 16, generator=g))[0]
        for nm, B in ch.items():
            cells = []
            for it in items:
                k = s0["items"].index(it); h, mu = s0["h"][l][k], s0["mu"][l][k]; dv = h - mu
                hp = mu + (B @ (B.T @ dv)) if nm != "full" else h
                cells.append(dict(item=it, block=l, ptype="t0", variant=nm, target=I["items"][it], H=hp, M=mu, seeds=s0["seeds"][l][k].tolist()[:3],
                                  dim=int(B.shape[1])))
            rr = S2.run_cells(ctx, E, cells, J=N(ctx, 16, 8), sets=False, fov=False)
            for r in rr:
                cats = [c for _, _, c in r["claims"]]; steps = [s for s, _, c in r["claims"] if c in ("TRUE-INT", "ANSWER", "PROMPT")]
                rows.append(dict(item=r["item"], block=l, ch=nm, fwd=float(any(c in ("TRUE-INT", "ANSWER") for c in cats)),
                                 bwd=float("PROMPT" in cats), horizon=max(steps) if steps else 0))
    write(ctx, "k/k7b.json", dict(rows=rows, labels_axes=A["labels"]), labels=["exploratory (K7b)"] + A["labels"])


# ---------------------------------------------------------------------------------------------------------------------------------
def k9_texts(ctx, E):
    """Conditions c1 (ACH-long texts), c2 (WikiText-103 neutral), c3 (TQA session transcripts), c4 (2,048 COHERENT + 512 neutral target),
    c5 (2,048 FILLER + the same target); 16 texts each."""
    import r1b_k_s4 as S4
    tok, b = E["tok"], E["b"]; n = N(ctx, 16, 2); L4 = N(ctx, 2048, 96)
    k8 = []
    for i in range(N(ctx, 6, 2)):
        p = ctx.input("r1b_K_S4", f"k/k8c/part{i}.pt", required=False)
        if p:
            k8 += load_pt(p)["rows"]
    byid = {x["id"]: x for x in b["ach_long"]}
    c1 = [RC.chat_ids(tok, byid[r["item"]]["prompt"]) + r["tok"] for r in k8][:n]
    c2 = [list(t) for t in b["wt"]["neutral"][:n]]
    tq = []
    for i in range(N(ctx, 8, 2)):
        p = ctx.input("r1b_K_S4", f"k/k8b/part{i}.pt", required=False)
        if p:
            for row in load_pt(p)["rows"]:
                if row["block"] == "tqa" and row["arm"] == "clean":
                    for s in row["sessions"]:
                        tq.append(sum((RC.chat_ids(tok, "Question") + t["tok"] + RC.enc(tok, S4.END) for t in s["turns"]), []))
    c3 = tq[:n]
    coh = " ".join(b["wt"]["coherent"][:600]); filler = " ".join(b["wt"]["coherent"][k] for k in np.random.default_rng(61).permutation(len(b["wt"]["coherent"]))[:600])
    tgt = c2[0][:N(ctx, 512, 64)] if c2 else []
    c4 = [RC.enc(tok, coh)[k * 50:k * 50 + L4] + tgt for k in range(n)]; c5 = [RC.enc(tok, filler)[k * 50:k * 50 + L4] + tgt for k in range(n)]
    return dict(c1=c1, c2=c2, c3=c3, c4=c4, c5=c5), L4


def k9_part(ctx, i, n):
    """K9: central differences (eps = 0.01 x the median token norm at the block) along each channel's top-2 directions at source
    positions 128, 256, 384 (teacher-forced: the KV route); readout of every deeper channel at Delta = 0..16; g_BA = sum_Delta
    ||Delta F_B|| / (2 eps) in B's natural SD (per text, averaged over directions and positions)."""
    torch = T_()
    E = S2.env(ctx); A = axes(ctx); model = E["model"]; nL = E["lens"].nL; S = S2.states(ctx); s0 = S["St"]["t0"]
    T, L4 = k9_texts(ctx, E)
    chans = []
    for l in (14, 18):
        if f"ustar{l}" in A["ax"]:
            chans.append((f"ustar{l}", l))
    for nm in ("uT", "usea"):
        if f"{nm}14" in A["ax"]:
            chans.append((f"{nm}14", 14))
    chans += [(f"Hsub{l}", l) for l in (10, 14, 18) if any(ll <= l for ll, _ in E["HA"])] + [(f"Jws{l}", l) for l in (14, 18)] + [("pc16_22", 22)]
    chans = [(c, l) for c, l in chans if l < nL]
    bases = {}
    for c, l in chans:
        X = s0["h"][l] if l in s0["h"] else None
        if X is None:
            with RC.Hooks(model) as hk, torch.no_grad():
                hk.capture([l]); model.model(input_ids=torch.tensor(T["c2"][:2], device=ctx.device)); X = hk.states[l][0].reshape(-1, model.config.hidden_size)
        sub = _subspaces(ctx, E, l, A, X); key = {"ustar": "ustar", "uT": "uT", "usea": "usea", "Hsub": "Hsub", "Jws": "Jws", "pc16": "pc16"}[c.rstrip("0123456789_").rstrip("_")]
        if key in sub:
            bases[c] = (l, sub[key][:, :16] if key in ("Hsub", "Jws", "pc16") else sub[key])
    jobs = [(cond, k) for cond in T for k in range(len(T[cond]))]
    rows = []
    blocks = sorted({l for l, _ in bases.values()})
    for cond, k in KB.parts(jobs, i, n):
        ids = torch.tensor([T[cond][k]], device=ctx.device); Lt = ids.shape[1]; off = L4 if cond in ("c4", "c5") else 0
        pos = [off + p for p in (128, 256, 384) if off + p + 16 < Lt] or [Lt // 2]
        with RC.Hooks(model) as hk, torch.no_grad():
            hk.capture(blocks, cpu=False); model.model(input_ids=ids); base = {b: hk.states[b][0][0] for b in blocks}
        sd = {c: float(((base[l] @ Bm.to(base[l].device)) - (base[l] @ Bm.to(base[l].device)).mean(0)).pow(2).sum(-1).mean().sqrt()) for c, (l, Bm) in bases.items()}
        g = {}
        for ca, (la, Ba) in bases.items():
            eps = 0.01 * float(base[la].norm(dim=-1).median())
            for d in range(min(2, Ba.shape[1])):
                v = Ba[:, d].to(ctx.device)
                for p in pos:
                    dd = TE.cdiff(model, ids, la, p, eps * v, [b for b in blocks if b > la] or [la])
                    for cb, (lb, Bb) in bases.items():
                        if lb > la:
                            dF = (dd[lb][0, p:p + 17] @ Bb.to(ctx.device)).norm(dim=-1).sum() / eps
                            g.setdefault((ca, cb), []).append(float(dF) / max(sd[cb], 1e-9))
        rows.append(dict(cond=cond, text=k, g={f"{a}->{b_}": float(np.mean(v)) for (a, b_), v in g.items()}, sd=sd))
    save(ctx, f"k/k9/part{i}.pt", dict(rows=rows, channels={c: l for c, (l, _) in bases.items()}, labels=A["labels"]))


# ---------------------------------------------------------------------------------------------------------------------------------
def k10a_texts(ctx):
    """64 own generations (T 0.7, 64 new tokens, keys r1bK_K10a): 32 first answers to round-1 RSI / scene prompts (P256 indices 0-15 and
    128-143) and 32 S_sess answers to 'Answer in two or three sentences.'; the Gumbel noise is fixed per text and absolute position."""
    torch = T_()
    tok, model = ctx.model(); b = KB.bank(ctx)
    p = os.path.join(RC.REPO, RC.C.PLAN_DIR, "prompts_1b.json"); P256 = (RC.load_json(p) or {}).get("prompts") if os.path.exists(p) else None
    if P256:
        pr = [P256[i]["text"] if isinstance(P256[i], dict) else P256[i] for i in list(range(16)) + list(range(128, 144))]
    elif ctx.mock:
        pr = [f"Describe a quiet scene number {i}." for i in range(32)]
    else:
        raise RC.NotRun("P256 prompts missing (prompts_1b.json)")
    lab = S1.alloc(ctx); byq = {r["qid"]: r for r in b["tqa"]}
    qs = [byq[q]["question"] + " Answer in two or three sentences." for q, v in lab["items"].items() if v["split"] == "S_sess"][:32]
    prompts = [RC.chat_ids(tok, x) for x in (pr[:N(ctx, 32, 2)] + qs[:N(ctx, 32, 2)])]
    texts = []
    for k, P in enumerate(prompts):
        ids = list(P)
        for s in range(N(ctx, 64, 24)):
            with torch.no_grad():
                lg = model(input_ids=torch.tensor([ids], device=ctx.device), logits_to_keep=1).logits[0, -1].float()
            ids.append(int((lg / 0.7 + noise(ctx, k, len(ids), lg.shape[0], lg.device)).argmax()))
        texts.append(dict(ids=ids, plen=len(P)))
    save(ctx, "k/k10a/texts.pt", dict(texts=texts))


def noise(ctx, text, pos, V, dev):
    torch = T_()
    gg = torch.Generator(device="cpu").manual_seed(ctx.key("r1bK_K10a", text, pos) % (2 ** 63))
    return (-torch.log(-torch.log(torch.rand(V, generator=gg).clamp_(1e-12, 1 - 1e-7)))).to(dev)


def k10a_part(ctx, i, n):
    """Route split per (text, source, dose, t0, sign): KV-only (original tokens teacher-forced with the push at t0), token-only
    (regeneration from t0 + 1 with the push at t0 and the original noise, re-read clean), total (states of the pushed regeneration);
    readouts at t0 + k, k in {1, 2, 4, 8}: a_H (H layers > 14), u*_18, u_T^v(18), S_J(18), S_J(22), top PC(22); same-block structural
    check (blocks <= 14 at later positions on the KV route)."""
    torch = T_()
    E = S2.env(ctx); A = axes(ctx); model, tok = E["model"], E["tok"]; R = E["R"]; nL = E["lens"].nL
    Tx = load_pt(own(ctx, "k/k10a/texts.pt"))["texts"]; dm = RC.Damage(model, KB.ref_k16(ctx)); cap = 2 * float(R["mu_gen"][14].norm())
    g = torch.Generator().manual_seed(1401); D = model.config.hidden_size; cal = load_pt(own(ctx, "k/k10a/cal.pt")) if own(ctx, "k/k10a/cal.pt") else None
    if cal is None:
        raise RuntimeError("k10a_cal must run first")
    srcs = cal["sources"]; jobs = [(s, dose, t0) for s in srcs for dose in ("delta1", "delta2") for t0 in (16, 32)]
    if ctx.trimmed("K-drop6"):
        jobs = [j for j in jobs if j[1] == "delta1"]
    rd_sets = {"H": [x for x in E["HA"] if x[0] > 14]}
    tblocks = sorted({b for b in (14, 18, 22) if b < nL} | set(range(0, 15, 7)))
    rows = []

    def read(ids, push=None):
        out = {}
        rd = SetReader(rd_sets, R["wn"].to(ctx.device), R["c_mu"], R["c_sd"], last=False) if rd_sets["H"] else None
        cms = [push()] if push else []
        with RC.Hooks(model) as hk, torch.no_grad():
            hk.capture(tblocks, cpu=False)
            tap = KB.Tap(model, rd) if rd else None
            try:
                for c in cms:
                    c.__enter__()
                model.model(input_ids=torch.tensor([ids], device=ctx.device))
            finally:
                for c in cms:
                    c.__exit__()
                if tap:
                    tap.__exit__()
            out = {b: hk.states[b][0][0] for b in tblocks}
            out["aH"] = rd.aX("H")[0] if rd else None
        return out

    def regen(text, k, t0, push):
        ids = list(text["ids"][:t0 + 1])
        for p in range(t0 + 1, min(len(text["ids"]), t0 + 9)):
            with push(), torch.no_grad():
                lg = model(input_ids=torch.tensor([ids], device=ctx.device), logits_to_keep=1).logits[0, -1].float()
            ids.append(int((lg / 0.7 + noise(ctx, k, p, lg.shape[0], lg.device)).argmax()))
        return ids
    for (sname, dose, t0) in KB.parts(jobs, i, n):
        s = srcs[sname]
        for k, text in enumerate(Tx):
            T0 = text["plen"] + t0
            if T0 + 9 > len(text["ids"]):
                continue
            base = read(text["ids"][:T0 + 9])
            res = {}
            for sign in ((1, -1) if s["kind"] == "dir" else (1,)):
                if s["kind"] == "dir":
                    mk = lambda sg=sign: _PosPush(model, 14, s["d"], sg * s[dose], cap, T0)
                else:
                    mk = lambda: Interv(model, [dict(sel=by_layer(s["set"]), kind="scale", value=s[dose], when=lambda T_, st, T0=T0: _at(T_, st, T0))], _PS)
                kv = read(text["ids"][:T0 + 9], push=mk)
                new = regen(text, k, T0, (lambda: _ctxreset(mk)) )
                tokr = read(new); tot = read(new, push=mk)
                res[sign] = dict(kv=kv, tok=tokr, tot=tot, div=int(new[T0 + 1] != text["ids"][T0 + 1]) if T0 + 1 < len(new) else 0)
            rows.append(dict(src=sname, dose=dose, t0=t0, text=k, eff=_effects(res, base, T0, s["kind"], A, E, tblocks)))
    save(ctx, f"k/k10a/part{i}.pt", dict(rows=rows))


_PS = dict(p0=0)


def _at(T, st, T0):
    torch = T_()
    return (torch.arange(T) == T0)


class _PosPush:
    """x + f min(|x|, cap) unit(d) at one absolute position (block output)."""

    def __init__(self, model, block, d, f, cap, pos):
        torch = T_(); v = torch.nn.functional.normalize(torch.as_tensor(d).float(), dim=0)

        def hook(m, a, o):
            x = o[0] if isinstance(o, tuple) else o
            if x.shape[1] > pos:
                x = x.clone(); x[:, pos] = x[:, pos] + f * x[:, pos].norm(dim=-1, keepdim=True).clamp(max=cap) * v.to(x.device, x.dtype)
            return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
        self.h = RC.layers_of(model)[block].register_forward_hook(hook)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.h.remove()


import contextlib


@contextlib.contextmanager
def _ctxreset(mk):
    c = mk()
    try:
        yield c
    finally:
        c.__exit__()


def _effects(res, base, T0, kind, A, E, tblocks):
    """Effects E at t0 + k on the targets, per route; directions: [F(+) - F(-)] / 2, populations F(alpha) - F(1)."""
    torch = T_()
    tg = {}
    for nm, l in (("ustar18", 18), ("uT18", 18)):
        if nm in A["ax"] and l in tblocks:
            tg[nm] = (l, torch.nn.functional.normalize(A["ax"][nm].float(), dim=0))
    for l in (18, 22):
        jm = LN.jmap(E["J"], E["conv"], l) if l in tblocks else None
        if jm is not None:
            _, _, v = torch.linalg.svd(jm.float(), full_matrices=False); tg[f"SJ{l}"] = (l, v[0])
    out = {}
    for route in ("kv", "tok", "tot"):
        for k in (1, 2, 4, 8):
            p = T0 + k; e = {}
            for nm, (l, d) in tg.items():
                f = lambda st: float(st[l][p] @ d.to(st[l].device))
                e[nm] = ((f(res[1][route]) - f(res[-1][route])) / 2) if kind == "dir" else (f(res[1][route]) - f(base))
            if base.get("aH") is not None:
                a = lambda st: float(st["aH"][p]) if st.get("aH") is not None else 0.0
                e["aH"] = ((a(res[1][route]) - a(res[-1][route])) / 2) if kind == "dir" else (a(res[1][route]) - a(base))
            out[f"{route}{k}"] = e
    # structural zero: KV route, blocks <= 14, later positions
    sz = 0.0
    for l in [b for b in tblocks if b <= 14]:
        for sg in res:
            sz = max(sz, float((res[sg]["kv"][l][T0 + 1:] - base[l][T0 + 1:]).abs().max()))
    out["structural_zero_max"] = sz
    out["div"] = float(np.mean([r["div"] for r in res.values()]))
    return out


def k10a_cal(ctx):
    """Sources at block 14 (u*_14, u_T^v(14), u_sea^v(14), the top S_J(14) direction, N_H population; 20 random directions, seed
    1401 + i, and RS10 populations) with per-position strengths calibrated to delta1 and delta2 on REF-K16 (push form; populations
    NSCALE_DN); REF-U32 reporting of the named sources."""
    torch = T_()
    E = S2.env(ctx); A = axes(ctx); model = E["model"]; R = E["R"]; dm = RC.Damage(model, KB.ref_k16(ctx)); cap = 2 * float(R["mu_gen"][14].norm())
    D = model.config.hidden_size; srcs = {}
    named = {k: A["ax"][k + "14"] for k in ("ustar", "uT", "usea") if k + "14" in A["ax"]}
    jm = LN.jmap(E["J"], E["conv"], 14)
    if jm is not None:
        _, _, v = torch.linalg.svd(jm.float(), full_matrices=False); named["SJ14"] = v[0].cpu()
    for k in range(N(ctx, 20, 3)):
        named[f"rand{k}"] = torch.randn(D, generator=torch.Generator().manual_seed(1401 + k))
    for nm, d in named.items():
        srcs[nm] = dict(kind="dir", d=d, **{x: _push_cal(ctx, model, dm, d, 14, cap, t)["f"] for x, t in (("delta1", RC.DELTA1), ("delta2", RC.DELTA2))})
    pops = [("N_H", E["HA"])] + [(f"RS10_{k}", s) for k, s in enumerate(E["rs"]["RS10"])]
    for nm, st in pops:
        if st:
            srcs[nm] = dict(kind="pop", set=st, **{x: RC.calibrate(lambda a, st=st: dm(lambda: Interv(model, [dict(sel=by_layer(st), kind="scale", value=a)]))["damage"],
                                                                3.0, t, form="gain")["f"] for x, t in (("delta1", RC.DELTA1), ("delta2", RC.DELTA2))})
    u32 = KB.ref_u32_damage(ctx, model, {f"{k}_delta1": (lambda d=v["d"], f=v["delta1"]: RC.Hooks(model).push(14, d, f, cap=cap))
                                         for k, v in srcs.items() if v["kind"] == "dir" and not k.startswith("rand")}, "K10a")
    save(ctx, "k/k10a/cal.pt", dict(sources=srcs, ref_u32=u32, labels=A["labels"]))


def k10b(ctx):
    """K10b: central differences along v = h_l(t) at position t only (x (1 +- 1e-2)): tau_Delta at h_27(t + Delta); the same with the
    attention probabilities frozen gives tau^OV; rho_QK = ||tau - tau^OV|| / (||tau^OV|| + ||tau - tau^OV||); foreign control; head
    routing map (VJP edge attribution for a_H and the S_J(22) top direction at t' in {32, 48, 63}); G-K10 checks (block-27 zero, FD vs
    JVP, frozen-QK reproduction)."""
    torch = T_()
    E = S2.env(ctx); model = E["model"]; nL = E["lens"].nL; Tx = load_pt(own(ctx, "k/k10a/texts.pt"))["texts"]; rows, maps = [], []
    blocks = [l for l in (6, 14, 18) if l < nL - 1]
    for k, text in enumerate(Tx):
        ids = torch.tensor([text["ids"]], device=ctx.device); Lt = ids.shape[1]
        with RC.Hooks(model) as hk, torch.no_grad():
            hk.capture(blocks, cpu=False); model.model(input_ids=ids); base = {b: hk.states[b][0][0] for b in blocks}
        f_ids = torch.tensor([Tx[(k + 1) % len(Tx)]["ids"]], device=ctx.device)
        with RC.Hooks(model) as hk, torch.no_grad():
            hk.capture(blocks, cpu=False); model.model(input_ids=f_ids); fb = {b: hk.states[b][0][0] for b in blocks}
        for t in (16, 32):
            t_ = text["plen"] + t
            if t_ + 5 >= Lt:
                continue
            for l in blocks:
                h = base[l][t_]
                for dl in (1, 4):
                    tau = TE.cdiff(model, ids, l, t_, 1e-2 * h, [nL - 1])[nL - 1][0, t_ + dl] / 1e-2
                    tov = TE.cdiff(model, ids, l, t_, 1e-2 * h, [nL - 1], frozen=True)[nL - 1][0, t_ + dl] / 1e-2
                    hf = fb[l][min(t_, fb[l].shape[0] - 1)]; v = h.norm() * hf / hf.norm()
                    tf = TE.cdiff(model, ids, l, t_, 1e-2 * v, [nL - 1])[nL - 1][0, t_ + dl] / 1e-2
                    tq = tau - tov
                    rows.append(dict(text=k, t=t, block=l, dl=dl, rho_qk=float(tq.norm() / (tov.norm() + tq.norm()).clamp(min=1e-12)),
                                     foreign_norm=float(tf.norm()), tau_norm=float(tau.norm())))
    ids0 = torch.tensor([Tx[0]["ids"]], device=ctx.device); L0 = ids0.shape[1]
    z27 = float(TE.cdiff(model, ids0, nL - 1, min(20, L0 - 2), torch.ones(model.config.hidden_size, device=ctx.device) * 1e-2, [nL - 1])[nL - 1][0, min(20, L0 - 2) + 1:].abs().max())
    fq = TE.frozen_qk_check(model, ids0); g = torch.Generator().manual_seed(1305); fj = []
    for c in range(N(ctx, 8, 2)):
        v = torch.randn(model.config.hidden_size, generator=g).to(ctx.device); u = torch.randn(model.config.hidden_size, generator=g).to(ctx.device)
        fj.append(TE.fd_vs_vjp(model, ids0, blocks[0], 10, v, nL - 1, min(14, L0 - 1), u)[0])
    hs = E["HA"]; R = E["R"]
    jm = LN.jmap(E["J"], E["conv"], min(22, nL - 2))
    dJ = torch.linalg.svd(jm.float(), full_matrices=False)[2][0].to(ctx.device) if jm is not None else None
    for tp in (32, 48, 63):
        for k, text in enumerate(Tx[:N(ctx, 4, 1)]):
            ids = torch.tensor([text["ids"]], device=ctx.device)
            tpp = min(text["plen"] + tp, ids.shape[1] - 1)
            if dJ is not None:
                edges, lag = TE.head_vjp(model, ids, lambda hsd, t, d=dJ: (hsd[min(22, nL - 2) + 1][0, t] * d).sum(), tpp)
                maps.append(dict(text=k, tprime=tp, target="S_J22", lag_share=lag))
    write(ctx, "k/k10b.json", dict(rows=rows, routing_map=maps,
                                   G_K10=dict(block27_zero=z27, block27_ok=z27 == 0.0, frozen_qk_rel=fq, frozen_qk_ok=fq < 1e-5,
                                              fd_vs_jvp=fj, fd_vs_jvp_ok=bool(max(fj) < 1e-3) if fj else None)),
          labels=["routing map exploratory"])
