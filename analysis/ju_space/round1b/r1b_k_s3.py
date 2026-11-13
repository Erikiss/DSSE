"""r1b_K_S3 (K5a) and r1b_K_S7 (RK-2) on Qwen3.5-2B: the native MTP head (15 mtp.* tensors, vLLM qwen3_5_mtp forward), its wiring
check (G-K17), the lenses on identical states and seeds (CL35 native with beta_N 0 / 1, CL-self E and L), IAD variant choice (G-K5's
input), the self-consistency KL per step (Teoh 2026) and the native length trend; RK-2: exact removal of direct H writes from the read
state with the native lens (H_tq(2B) by K2's protocol in r1b_K_S7's own steps)."""
import glob, os
import numpy as np
import r1b_core as RC
import r1b_contracts as C
import r1b_k_base as KB
import r1b_k_labels as KL
import r1b_k_lens as LN
from r1b_k_base import N, H, T_, write, save, load_pt, own, up, coded, val

M35 = C.M35


def _load35(root, device, mock):
    """Qwen3.5-2B text backbone, float32, sdpa (transformers 5.22.2 ships qwen3_5; '^mtp.*' is dropped on load and read separately)."""
    torch = T_()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d = RC.ensure_model(M35, root, None, download=True)
    tok = AutoTokenizer.from_pretrained(d)
    m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32, attn_implementation="sdpa").eval()
    return tok, m


RC.register_loader(M35, _load35, layers=24, hidden=2048)


def mtp_tensors(ctx):
    """The 15 mtp.* tensors from the safetensors shards (G-K18)."""
    from safetensors import safe_open
    d = os.path.join(ctx.root, "models", M35); out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if k.startswith("mtp.") or ".mtp." in k:
                    out[k[k.index("mtp."):]] = s.get_tensor(k).float()
    return out


def mtp_head(ctx, model):
    """The MTP head with the loaded weights (MOCK: the same module with a seeded random init, labelled)."""
    torch = T_()
    nL = len(RC.layers_of(model)); fa = [i for i in range(nL) if getattr(RC.layers_of(model)[i], "self_attn", None) is not None]
    torch.manual_seed(0); head = LN.make_head(model, init_block=fa[-1] if fa else nL - 1).to(ctx.device)
    if ctx.mock:
        return head, dict(G_K18="MOCK head (random init)", n_tensors=None)
    T = mtp_tensors(ctx)
    if len(T) != 15:
        raise RC.NotRun(f"MTP not loadable ({len(T)} mtp tensors; G-K18)")
    sd = {}
    for k, v in T.items():
        k2 = k[4:].replace("layers.0.", "layer.")
        sd[k2] = v
    missing, unexpected = head.load_state_dict(sd, strict=False)
    if unexpected or [m for m in missing if not m.startswith("layer.") or m.replace("layer.", "") in sd]:
        raise RC.NotRun(f"MTP tensor layout mismatch (missing {missing[:4]}, unexpected {unexpected[:4]}; G-K18)")
    return head, dict(G_K18="ok", n_tensors=len(T), n_params=int(sum(v.numel() for v in T.values())))


def k5a_load(ctx):
    tok, model = ctx.model(M35); head, g18 = mtp_head(ctx, model)
    write(ctx, "k/k5a/load.json", dict(G_K18=g18, layers=len(RC.layers_of(model)), vocab=int(model.get_output_embeddings().weight.shape[0]),
                                       P0_tokens=len(RC.enc(tok, KL.P0)), ph_tokens=len(RC.enc(tok, KL.PH))))


def _wiring_agree(ctx, model, tok, head, wiring, pre):
    """Top-1 x_{t+2} of the head (given h(t) and the true x_{t+1}) against the backbone's greedy prediction at t + 1."""
    torch = T_()
    from transformers import DynamicCache
    b = KB.bank(ctx); seqs = KB.retok(ctx, tok, b["wt"]["ntext"][:N(ctx, 16, 2)], 256)
    head.wiring = (wiring, "pre" if pre else "post"); agree, n = 0, 0; norm = KB.lm_parts(model)[0]; W = model.get_output_embeddings().weight
    rot = LN.rotary_of(model); emb = model.get_input_embeddings()
    for s in seqs:
        ids = torch.tensor([s], device=ctx.device)
        with RC.Hooks(model) as hk, torch.no_grad():
            hk.capture([len(RC.layers_of(model)) - 1], cpu=False); lg = model(input_ids=ids).logits[0]
            h = hk.states[len(RC.layers_of(model)) - 1][0][0]
        t = torch.arange(16, ids.shape[1] - 2, device=ctx.device)[:N(ctx, 125, 20)]
        hid = h[t] if pre else norm(h[t])
        with torch.no_grad():
            out = head.step(emb(ids[0, t + 1]).float(), hid.float(), DynamicCache(), (t + 1)[:, None], rot)
            pred = (out @ W.T.float()).argmax(-1)
        agree += int((pred == lg[t + 1].argmax(-1)).sum()); n += len(t)
    return agree / max(n, 1)


def k5a_wiring(ctx):
    """G-K17: the first wiring in the frozen order ([emb; hidden] post-norm, [hidden; emb] post-norm, [emb; hidden] pre-norm) whose
    agreement is >= 0.4 on 2,000 WikiText positions."""
    tok, model = ctx.model(M35); head, _ = mtp_head(ctx, model); res = []
    for w, pre in (("eh", False), ("he", False), ("eh", True)):
        a = _wiring_agree(ctx, model, tok, head, w, pre); res.append(dict(wiring=w, pre=pre, agreement=a))
        if a >= 0.4:
            break
    ok = res[-1]["agreement"] >= 0.4
    write(ctx, "k/k5a/wiring.json", dict(G_K17=dict(pass_=ok, tried=res, used=res[-1] if ok else None,
                                                    string=None if ok else "not run (MTP wiring failed)")),
          labels=[] if ok else ["G-K17 failed: CL35 'not run (MTP wiring failed)'"])


def jlens35(ctx, model):
    """Qwen3.5-2B J-lens (sha256 from the freeze record; MOCK: seeded random maps)."""
    pin = (ctx.freeze.get("jlens") or {}).get("qwen3.5-2b") or {}
    p = os.path.join(ctx.root, "jlens", pin.get("name", "Qwen3.5-2B_jacobian_lens.pt"))
    D = model.config.get_text_config().hidden_size if hasattr(model.config, "get_text_config") else model.config.hidden_size
    if ctx.mock or not os.path.exists(p):
        if not ctx.mock:
            raise RC.NotRun("Qwen3.5-2B J-lens missing")
        return RC.load_jlens(None, mock=True, D=D, n=len(RC.layers_of(model)) - 1)
    return RC.load_jlens(p, sha256=pin.get("sha256"), n=len(RC.layers_of(model)) - 1)


def k5a_gate(ctx):
    """The A-lens pool gated on 2B (greedy + 4 samples at T 0.7): 256 correct items; t0 states at blocks {10, 14, 18}; mu_task; seeds."""
    torch = T_()
    import r1b_k_s1 as S1
    tok, model = ctx.model(M35); b = KB.bank(ctx); J = jlens35(ctx, model)
    pool = [it for it in b["ach_short"] if it["split"] == "A-lens" and not it["cal"]][:N(ctx, 1236, 30)]
    P = [RC.chat_ids(tok, it["prompt"]) for it in pool]
    g = KB.decode(model, tok, P, max_new=12, batch=N(ctx, 128, 16))
    sm = KB.decode(model, tok, [p for p in P for _ in range(4)], keys=[ctx.key("r1bK_gate", "2B", it["id"], s) for it in pool for s in range(4)],
                   T=0.7, max_new=12, batch=N(ctx, 128, 16))
    ok = [S1.label_ach(ctx, it, tok.decode(c)) for it, c in zip(pool, g)]
    cor = [k for k, o in enumerate(ok) if o][:N(ctx, 256, 6)]; blocks = [l for l in (10, 14, 18) if l < len(RC.layers_of(model)) - 1]
    hs = {l: [] for l in blocks}
    for k in cor:
        with RC.Hooks(model) as hk, torch.no_grad():
            hk.capture(blocks); model.model(input_ids=torch.tensor([P[k]], device=ctx.device))
            for l in blocks:
                hs[l].append(hk.states[l][0][0, -1])
    St = {l: torch.stack(v) for l, v in hs.items()} if cor else {}
    mu = {l: (v.sum(0, keepdim=True) - v) / max(len(v) - 1, 1) for l, v in St.items()}
    seeds = {l: LN.jlens_seeds(model, tok, J, "output", l, v.to(ctx.device), k=N(ctx, 10, 3)) for l, v in St.items()}
    items = [pool[k] for k in cor]; f = np.roll(np.arange(len(items)), 1)
    save(ctx, "k/k5a/states.pt", dict(items=items, prompts=[P[k] for k in cor], h=St, mu=mu, seeds=seeds, foreign=f, blocks=blocks,
                                      own={pool[k]["id"]: sorted({x for x in (KL.parse_number(tok.decode(c)) for c in [g[k]] + sm[4 * k:4 * k + 4]) if x is not None})
                                           for k in cor}))


def k5a_run(ctx):
    """CL35 native (V1, V2; beta_N 0 primary, beta_N 1 variant), CL-self E and L (V1, V2) on identical states and seeds, J = 16; the
    self-consistency KL per step for CL35 (V1, top-4 seeds)."""
    torch = T_()
    tok, model = ctx.model(M35); S = load_pt(own(ctx, "k/k5a/states.pt")); J = jlens35(ctx, model)
    wr = RC.load_json(own(ctx, "k/k5a/wiring.json"))["G_K17"]; head, _ = mtp_head(ctx, model)
    if wr["used"]:
        head.wiring = (wr["used"]["wiring"], "pre" if wr["used"]["pre"] else "post")
    norm = KB.lm_parts(model)[0]; pc = LN.pieces(tok); nL = len(RC.layers_of(model))
    frames = dict(P0_ids=RC.enc(tok, KL.P0), ph_id=RC.enc(tok, KL.PH)[-1])
    lens = LN.Lens(model, tok, frames); D = S["h"][S["blocks"][0]].shape[1] if S["blocks"] else 0
    bias = None
    if ctx.mock:
        bias = torch.zeros(lens.V, device=ctx.device); bias[KB.digit_ids(tok)] = 6.0
    with torch.no_grad():
        n22 = float(torch.stack([S["h"][l] for l in S["blocks"]]).norm(dim=-1).median()) if S["blocks"] else 1.0
    rows, skl = [], []
    J_ = N(ctx, 16, 8)
    for l in S["blocks"]:
        h, mu, sd = S["h"][l], S["mu"][l], S["seeds"][l]
        for var in ("V1", "V2"):
            ix = np.arange(len(h)) if var == "V1" else S["foreign"]
            for k in range(len(h)):
                it = S["items"][k]; kk = int(ix[k]); seeds = sd[kk].tolist()
                hr = LN.transport(J, "output", l, h[kk][None].to(ctx.device)); hp = LN.transport(J, "output", l, mu[kk][None].to(ctx.device))
                pre = head.wiring[1] == "pre"
                h0r = (hr if pre else norm(hr)).repeat(len(seeds), 1); h0p = (hp if pre else norm(hp)).repeat(len(seeds), 1)
                outs = {"CL35": LN.head_rollout(model, tok, head, h0r, h0p, seeds, J=J_, bias=bias),
                        "CL35_b1": LN.head_rollout(model, tok, head, h0r, h0p, seeds, J=J_, beta=1.0, bias=bias)}
                outs["E"] = lens.rollout(l, h[kk][None].repeat(len(seeds), 1).to(ctx.device), mu[kk][None].repeat(len(seeds), 1).to(ctx.device),
                                         seeds, J=J_, variant="E", sets=False, fov=False, bias=bias)
                vL = n22 * torch.nn.functional.normalize(hr, dim=-1); mL = n22 * torch.nn.functional.normalize(hp, dim=-1)
                outs["L"] = lens.rollout(l, vL.repeat(len(seeds), 1), mL.repeat(len(seeds), 1), seeds, J=J_, variant="L", sets=False, fov=False, bias=bias)
                for lensname, r in outs.items():
                    for s_ in range(len(seeds)):
                        tk = [seeds[s_]] + r["tok"][s_].tolist(); sc = KL.score_chain([pc[t] for t in tk], it, (), S["own"].get(it["id"], ()), tokens=tk)
                        rows.append(dict(item=it["id"], block=l, variant=var, lens=lensname, seed_rank=s_, claims=sc["claims"], censor=sc["censor"],
                                         e=r["e"][s_].half()))
                if var == "V1":                               # self-consistency KL (Teoh 2026) for CL35 on the top-4 seeds
                    r = outs["CL35"]; ch = [r["tok"][s_].tolist() for s_ in range(min(4, len(seeds)))]
                    lp = LN.head_tf_logp(model, head, h0r[:len(ch)], seeds[:len(ch)], ch)
                    lq = LN.backbone_logq(model, [S["prompts"][k]] * len(ch), seeds[:len(ch)], ch)
                    for s_, kl in enumerate(LN.kl_rows(lq, lp).cpu()):
                        skl.append(dict(item=it["id"], block=l, lens="CL35", seed_rank=s_, kl=kl.tolist()))
    save(ctx, "k/k5a/chains.pt", dict(rows=rows, selfkl=skl, n_items=len(S["items"])))
    import r1b_k_ana as KA
    res = KA.k5a_stats(load_pt(own(ctx, "k/k5a/chains.pt")), B=200 if ctx.mock else RC.N_BOOT, J=J_)
    res["G_K17"] = wr; res["G_K18"] = RC.load_json(own(ctx, "k/k5a/load.json"))["G_K18"]
    write(ctx, "k5a_calib.json", res, config=dict(J=J_, lam=1.0, plaus=0.1), labels=res.get("labels", []))


def k5a_notrun(ctx):
    """k5a_calib.json when K5a could not run (G-K18 / G-K17): variant E, label 'not calibrated (K5a not run)'."""
    if own(ctx, "k5a_calib.json"):
        return
    write(ctx, "k5a_calib.json", dict(variant="E", G_K5="not decidable", labels=["not calibrated (K5a not run)"]),
          labels=["not calibrated (K5a not run)"])


# ---------------------------------------------------------------------------------------------------------------------------------
def rk2_run(ctx):
    """RK-2: exact removal of the direct H_tq(2B) writes from the read state (layers <= l) at 96 items x {14, 18}, t0; the native lens
    on h' and on h; RS39 analogues; item-level Delta of the per-claim confabulation share."""
    torch = T_()
    import r1b_k_s1 as S1
    tok, model = ctx.model(M35); hs = S1.hnset(ctx); rs = S1.rssets(ctx); R = S1.refs(ctx); J = jlens35(ctx, model)
    S = load_pt(up(ctx, "r1b_K_S3", "k/k5a/states.pt")); head, _ = mtp_head(ctx, model); norm = KB.lm_parts(model)[0]
    wr = RC.load_json(up(ctx, "r1b_K_S3", "k/k5a/wiring.json"))["G_K17"]
    if wr["used"]:
        head.wiring = (wr["used"]["wiring"], "pre" if wr["used"]["pre"] else "post")
    pc = LN.pieces(tok); Ls = RC.layers_of(model); n = N(ctx, 96, 4); out = []
    sets = [("H", hs["H_tq"])] + [(f"RS{k}", s) for k, s in enumerate(rs["RS39"])]
    for l in [b for b in (14, 18) if b in S["h"]]:
        for k in range(min(n, len(S["items"]))):
            it = S["items"][k]; P = S["prompts"][k]; zs = {}
            with KB.Tap(model, lambda li, z, den: zs.__setitem__(li, z[0, -1].float().cpu())), torch.no_grad():
                model.model(input_ids=torch.tensor([P], device=ctx.device))
            for nm, st in [("none", [])] + sets:
                h = S["h"][l][k].clone()
                for (ll, j) in [tuple(x) for x in st if x[0] <= l]:
                    h -= (float(zs[ll][j]) - float(R["z_mu"][ll, j])) * Ls[ll].mlp.down_proj.weight[:, j].float().cpu()
                seeds = S["seeds"][l][k].tolist()
                hr = norm(LN.transport(J, "output", l, h[None].to(ctx.device))); hp = norm(LN.transport(J, "output", l, S["mu"][l][k][None].to(ctx.device)))
                r = LN.head_rollout(model, tok, head, hr.repeat(len(seeds), 1), hp.repeat(len(seeds), 1), seeds, J=N(ctx, 16, 8))
                q = []
                for s_ in range(len(seeds)):
                    tk = [seeds[s_]] + r["tok"][s_].tolist(); sc = KL.score_chain([pc[t] for t in tk], it, (), S["own"].get(it["id"], ()), tokens=tk)
                    c = [x for _, _, x in sc["claims"] if x not in ("PROMPT", "AMBIG")]
                    q.append(sum(x == "CONFAB" for x in c) / len(c) if c else np.nan)
                out.append(dict(item=it["id"], block=l, cond=nm, q=float(np.nanmean(q)) if np.isfinite(q).any() else None))
    save(ctx, "k/rk2/rows.pt", dict(rows=out))
    import r1b_k_ana as KA
    write(ctx, "k/rk2/rk2.json", KA.rk2_rule(out, RC.load_json(KB.up(ctx, "r1b_K_S1", "sg8_preview.json")).get("G_K2")),
          labels=["reserve RK-2 (reported next to the composite; never enters it)"])
