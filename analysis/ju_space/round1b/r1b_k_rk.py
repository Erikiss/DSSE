"""RK-1 (r1b_K_S6, tier 2, unconditional): two post-hoc MTP heads on frozen Qwen3-1.7B, trained on the same data and steps in bf16:
the standard CE head (cross-entropy to the data token x_{t+2}) and the Soft-MTP head of Teoh 2026 (arXiv 2610.38331: straight-through
gradient routing; with a frozen trunk and a frozen shared output head the head receives only g_KL, i.e. KL(sg(q_{t+1}) || p) to the
model's own next-token distribution). Head: fc([RMSNorm(emb x_{t+1}); RMSNorm(h_27,t)]) -> a decoder layer initialised from block 27 ->
the final norm -> the frozen tied head. Evaluation: K6a statistics on K4's cells (CL35-style rollout from J_l h), C_k, the per-claim
slope and the self-consistency KL per step, against CL17's V1 chains. RK-2 helpers for r1b_K_S7."""
import math, os
import numpy as np
import r1b_core as RC
import r1b_k_base as KB
import r1b_k_labels as KL
import r1b_k_lens as LN
from r1b_k_base import N, T_, write, save, load_pt, own, up, coded, val


def _data(ctx, tok, n_tok):
    """WikiText-103 train tokens after K's 1M-token reference (SO-WT103); MOCK: the bank's reference ids."""
    b = KB.bank(ctx)
    if ctx.mock:
        ids = np.load(b["_files"]["wt_train_ref.npy"]); return np.tile(ids, int(np.ceil(n_tok / len(ids))))[:n_tok]
    import pandas as pd
    d = RC.fetch_dataset("wikitext103", ctx.root, ctx.freeze); fs = sorted(os.path.join(r, f) for r, _, x in os.walk(d) for f in x if f.endswith(".parquet") and "train" in r + f)
    off = int((ctx.freeze.get("datasets", {}).get("wikitext103", {}) or {}).get("rk1_train_offset_lines", 0)); out = []
    for f in fs:
        for l in pd.read_parquet(f)["text"].tolist()[off:]:
            if l.strip():
                out += RC.enc(tok, " " + l.strip())
            if len(out) >= n_tok:
                return np.array(out[:n_tok])
        off = 0
    return np.array(out[:n_tok])


def heads(ctx, model):
    torch = T_()
    torch.manual_seed(0); h1 = LN.make_head(model, init_block=len(RC.layers_of(model)) - 1)
    torch.manual_seed(0); h2 = LN.make_head(model, init_block=len(RC.layers_of(model)) - 1)
    return h1.to(ctx.device), h2.to(ctx.device)


def rk1_train(ctx, i, n):
    """Chunk i of n: both heads on the same batches (one shared frozen-trunk forward), AdamW lr 3e-4 with cosine decay, grad clip 1,
    bf16 autocast on CUDA; checkpoint per chunk (resumable)."""
    torch = T_()
    tok, model = ctx.model(); ce, soft = heads(ctx, model)
    prev = own(ctx, f"k/rk1/ckpt{i - 1}.pt") if i > 0 else None
    opt = torch.optim.AdamW(list(ce.parameters()) + list(soft.parameters()), lr=3e-4, weight_decay=0.0)
    if prev:
        ck = load_pt(prev); ce.load_state_dict(ck["ce"]); soft.load_state_dict(ck["soft"]); opt.load_state_dict(ck["opt"])
    elif i > 0:
        raise RuntimeError(f"RK-1 chunk {i}: checkpoint {i - 1} missing")
    T, Bt = N(ctx, 512, 32), N(ctx, 16, 2); tot = N(ctx, 16_000_000, 4096); per = tot // n
    data = _data(ctx, tok, tot); seg = data[i * per:(i + 1) * per]; steps = max(1, len(seg) // (T * Bt)); total_steps = steps * n
    emb, W = model.get_input_embeddings(), model.get_output_embeddings().weight; rot = LN.rotary_of(model); nL = len(RC.layers_of(model))
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if str(ctx.device).startswith("cuda") else __import__("contextlib").nullcontext()
    log = []
    for s in range(steps):
        x = torch.tensor(seg[s * T * Bt:(s + 1) * T * Bt].reshape(Bt, T), device=ctx.device)
        with torch.no_grad(), amp:
            o = model(input_ids=x, output_hidden_states=True); h27 = o.hidden_states[nL].float(); q = o.logits.float().log_softmax(-1)
        lr = 3e-4 * 0.5 * (1 + math.cos(math.pi * (i * steps + s) / total_steps))
        for gp in opt.param_groups:
            gp["lr"] = lr
        loss = {}
        for nm, hd in (("ce", ce), ("soft", soft)):
            with amp:
                e = emb(x[:, 1:-1]).float(); hh = h27[:, :-2]
                a, b2 = hd.pre_fc_norm_embedding(e), hd.pre_fc_norm_hidden(hh)
                z = hd.fc(torch.cat([a, b2], -1)); pos = torch.arange(z.shape[1], device=z.device)[None].repeat(Bt, 1)
                y = hd.layer(hidden_states=z, attention_mask=None, position_ids=pos, position_embeddings=rot(z, pos))
                y = y[0] if isinstance(y, tuple) else y
                lp = (hd.norm(y) @ W.T.float()).float().log_softmax(-1)
            if nm == "ce":
                l_ = torch.nn.functional.nll_loss(lp.reshape(-1, lp.shape[-1]), x[:, 2:].reshape(-1))
            else:                                       # Soft-MTP on a frozen trunk: KL(sg(q_{t+1}) || p_head)
                qq = q[:, 1:-1]; l_ = (qq.exp() * (qq - lp)).sum(-1).mean()
            loss[nm] = l_
        opt.zero_grad(); (loss["ce"] + loss["soft"]).backward()
        torch.nn.utils.clip_grad_norm_(list(ce.parameters()) + list(soft.parameters()), 1.0); opt.step()
        if s % max(1, steps // 10) == 0:
            log.append(dict(step=i * steps + s, ce=float(loss["ce"]), soft=float(loss["soft"]), lr=lr))
    save(ctx, f"k/rk1/ckpt{i}.pt", dict(ce=ce.state_dict(), soft=soft.state_dict(), opt=opt.state_dict(), log=log, tokens=int(len(seg))),
         config=dict(T=T, batch=Bt, tokens_total=tot, dtype="bf16 autocast", lr=3e-4))


def rk1_eval(ctx, n):
    """K6a statistics of both heads on K4's cells (V1 and V2 at t0, correct items, blocks 14 and 18, top-4 seeds), against CL17's V1
    chains; the self-consistency KL per step (Teoh 2026)."""
    torch = T_()
    import r1b_k_ana as KA
    tok, model = ctx.model(); ce, soft = heads(ctx, model); ck = load_pt(own(ctx, f"k/rk1/ckpt{n - 1}.pt"))
    ce.load_state_dict(ck["ce"]); soft.load_state_dict(ck["soft"]); ce.eval(); soft.eval()
    St = load_pt(up(ctx, "r1b_K_S2a", "k/k4/states.pt")); I = load_pt(up(ctx, "r1b_K_S2a", "k/k4/items.pt")); J = KB.jl(ctx, model)
    conv = RC.load_json(up(ctx, "r1b_K_S1", "k4j.json")).get("convention", "output"); s0 = St["St"]["t0"]; pc = LN.pieces(tok)
    S4 = load_pt(up(ctx, "r1b_K_S2a", "k/k4/summary.pt")); bias = None
    if ctx.mock:
        import r1b_k_s2 as S2
        bias = S2._digit_bias(ctx, tok, type("Lx", (), dict(V=model.get_output_embeddings().weight.shape[0], dev=ctx.device)))
    rows, skl = [], []
    items = [i for i in I["correct"] if i in s0["items"]][:N(ctx, 128, 3)]
    for l in [b for b in (14, 18) if b in s0["h"]]:
        for var in ("V1", "V2"):
            for it in items:
                k = s0["items"].index(it); kk = k if var == "V1" else int(St["foreign"]["t0"][k]); seeds = s0["seeds"][l][kk].tolist()[:4]
                hr = LN.transport(J, conv, l, s0["h"][l][kk][None].to(ctx.device)); hp = LN.transport(J, conv, l, s0["mu"][l][kk][None].to(ctx.device))
                for nm, hd in (("CE", ce), ("SOFT", soft)):
                    r = LN.head_rollout(model, tok, hd, hr.repeat(len(seeds), 1), hp.repeat(len(seeds), 1), seeds, J=N(ctx, 16, 8), bias=bias)
                    for s_ in range(len(seeds)):
                        tk = [seeds[s_]] + r["tok"][s_].tolist(); sc = KL.score_chain([pc[t] for t in tk], I["items"][it], (), I["gate"][it]["own"], tokens=tk)
                        rows.append(dict(item=it, block=l, variant=var, lens=nm, seed_rank=s_, claims=sc["claims"], censor=sc["censor"], e=r["e"][s_].half()))
                    if var == "V1":
                        ch = [r["tok"][s_].tolist() for s_ in range(len(seeds))]
                        lp = LN.head_tf_logp(model, hd, hr.repeat(len(seeds), 1), seeds, ch)
                        lq = LN.backbone_logq(model, [RC.chat_ids(tok, I["items"][it]["prompt"])] * len(seeds), seeds, ch)
                        for s_, kl in enumerate(LN.kl_rows(lq, lp).cpu()):
                            skl.append(dict(item=it, block=l, lens=nm, kl=kl.tolist()))
    cl17 = [dict(r, lens="CL17") for r in S4["rows"] if r["variant"] in ("V1", "V2") and r["ptype"] == "t0" and r["block"] in (14, 18) and r["seed_rank"] < 4 and r["item"] in items]
    allrows = rows + cl17; J_ = N(ctx, 16, 8)
    C16 = lambda lens: {i: float(np.mean([any(c == "CONFAB" for s, _, c in r["claims"] if s <= J_) for r in rs])) for (i,), rs in
                        KA.by([r for r in allrows if r["lens"] == lens and r["variant"] == "V1"], "item").items()}
    d_soft = KA.item_diff(C16("SOFT"), C16("CL17")); d_ce = KA.item_diff(C16("CE"), C16("SOFT"))
    sl = {lens: KA.logit_slope(KA.claim_counts([r for r in allrows if r["lens"] == lens and r["variant"] == "V1"]), B=200 if ctx.mock else RC.N_BOOT)
          for lens in ("CE", "SOFT", "CL17")}
    kl = {nm: np.mean([x["kl"] for x in skl if x["lens"] == nm], 0).tolist() if any(x["lens"] == nm for x in skl) else None for nm in ("CE", "SOFT")}
    klsl = {}
    for nm in ("CE", "SOFT"):
        arr = np.array([x["kl"] for x in skl if x["lens"] == nm])
        if arr.size:
            s_ = np.arange(arr.shape[1]); f = lambda ix: np.polyfit(s_, arr[ix].mean(0), 1)[0]
            c_ = KB.ci(KB.boot_stat(len(arr), f, B=200 if ctx.mock else RC.N_BOOT))
            klsl[nm] = dict(slope=val(f(np.arange(len(arr)))), ci=c_, string="break-off lens-owned (self-consistency KL grows)" if KB.gt(c_) else
                            ("break-off model-owned (self-consistency KL flat)" if KB.inside(c_, 0.02) else "undecided"))
    out = dict(strings=dict(soft_vs_self="Soft-MTP head follows the self-rollout (length curve within +-0.10)" if KB.inside(d_soft["ci"], 0.10) else
                            ("Soft-MTP head differs from the self-rollout" if (KB.gt(d_soft["ci"]) or KB.lt(d_soft["ci"])) else "undecided"),
                            ce_vs_soft="CE head adds confabulation beyond the Soft-MTP head" if KB.gt(d_ce["ci"]) else
                            ("no CE excess over the Soft-MTP head" if KB.inside(d_ce["ci"], 0.10) else "undecided")),
               d_C16_soft_minus_CL17=KA._s(d_soft), d_C16_ce_minus_soft=KA._s(d_ce),
               slopes={k: dict(est=val(v[0]), ci=KB.ci(v[1]) if len(v[1]) else None) for k, v in sl.items()}, selfkl_by_step=kl, selfkl_slope=klsl,
               train_log=ck.get("log"))
    write(ctx, "k/rk1.json", out, labels=["secondary arm (trained head; own language prior)", "tier 2 (RK-1)"])


# ---------------------------------------------------------------------------------------------------------------------------------
def rk2_model(ctx):
    """r1b_K_S7: K2's protocol on Qwen3.5-2B (the r1b_k_s1 steps run with MODEL = qwen3.5-2b in this session)."""
    import r1b_k_s1 as S1
    S1.MODEL = RC.C.M35
    import r1b_k_s3  # noqa: F401  (registers the Qwen3.5-2B loader)
    tok, model = ctx.model(RC.C.M35)
    write(ctx, "k/rk2/model.json", dict(model=RC.C.M35, layers=len(RC.layers_of(model)), screening_cap=6000,
                                        note="K2 on 2B: screening cap 6,000 (6 blocks), C by suppression on 1,000 S2 questions, RS39 on 2B"))


def rk2_no_har(ctx):
    """RK-2 needs no arithmetic set: H_ar is empty in r1b_K_S7 (K2's G-K1 fallback does not apply)."""
    save(ctx, "k/k2/h_ar.pt", dict(H=[], note="not used in RK-2"))
