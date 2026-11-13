"""r1b_K_S4 (account B): K8a (position vs content vs length at matched positions, incl. the A-bnd boundary screening, offsets O1 / O2
and G-K20), K8b (perturbation persistence in the model's own text: ACH and TQA sessions, routes, bistability, G-K16), K8c (long-form
first-error hazard with in-situ H, lens reach) and K7a (arrow-vector atlas; SO-AX-free channels in this account, K.12). Sessions use
one incremental cache per batch (B-Q8 concatenation: each history turn keeps its empty think block)."""
import math, os
import numpy as np
import r1b_core as RC
import r1b_k_base as KB
import r1b_k_labels as KL
import r1b_k_lens as LN
from r1b_k_base import N, H, T_, write, save, load_pt, own, up, Interv, by_layer, SetReader, coded, val
import r1b_k_s1 as S1
import r1b_k_s2 as S2

END = "<|im_end|>\n"


class Conv:
    """A batch of B conversations on one DynamicCache: feed(chunks) appends left-padded chunks (holes are masked; explicit position
    ids), gen(...) decodes with keyed Gumbel (or greedy), forced tokens, a SetReader and per-token entropy."""

    def __init__(self, model, tok, B, prefix=None):
        torch = T_()
        from transformers import DynamicCache
        self.m, self.tok, self.B, self.dev = model, tok, B, RC.model_device(model)
        self.cache = DynamicCache(); self.mask = torch.zeros(B, 0, dtype=torch.long, device=self.dev)
        self.pos = torch.zeros(B, dtype=torch.long, device=self.dev)
        if prefix is not None:                                   # a shared prefix (B = 1 prefill, cache copies)
            pc, Lp = prefix
            for li, lay in enumerate(pc.layers):
                self.cache.update(lay.keys.repeat(B, 1, 1, 1), lay.values.repeat(B, 1, 1, 1), li)
            self.mask = torch.ones(B, Lp, dtype=torch.long, device=self.dev); self.pos += Lp
        self.st = dict(active=False)

    def feed(self, chunks):
        torch = T_()
        T = max(len(c) for c in chunks); ids = torch.full((self.B, T), self.tok.pad_token_id, device=self.dev)
        m = torch.zeros(self.B, T, dtype=torch.long, device=self.dev)
        for r, c in enumerate(chunks):
            if c:
                ids[r, T - len(c):] = torch.tensor(c, device=self.dev); m[r, T - len(c):] = 1
        full = torch.cat([self.mask, m], 1); pos = self.pos[:, None] + (m.cumsum(1) - 1).clamp(min=0)
        with torch.no_grad(), RC.sdpa_efficient(self.dev):
            o = self.m(input_ids=ids, attention_mask=full, position_ids=pos, past_key_values=self.cache, use_cache=True, logits_to_keep=1)
        self.mask = full; self.pos = self.pos + m.sum(1)
        return o.logits[:, -1].float()

    def gen(self, lg, keys=None, T=0.7, max_new=30, forced=None, reader=None, sets=("H",), det=None):
        """Returns per row: tokens, per-token records (a_X, s_H, entropy) for the answer tokens (the state of token k is read when it
        is fed)."""
        torch = T_()
        G = KB.Gumbel(self.dev, T, keys) if keys is not None else None; eos, stop = RC.stop_tokens(self.tok, self.m)
        stop_t = torch.tensor(sorted(stop), device=self.dev); done = torch.zeros(self.B, dtype=torch.bool, device=self.dev)
        toks = [[] for _ in range(self.B)]; recs = [[] for _ in range(self.B)]
        for s in range(max_new):
            l = lg.log_softmax(-1); ent = -(l.exp() * l.nan_to_num(neginf=0.0)).sum(-1)
            x = (G(lg) if G is not None else lg).argmax(-1)
            if forced is not None:
                for r in range(self.B):
                    if forced[r] is not None:
                        x[r] = forced[r][s] if s < len(forced[r]) else int(stop_t[0])
            fin = torch.isin(x, stop_t) | done
            for r in range(self.B):
                if not fin[r]:
                    toks[r].append(int(x[r]))
            newly = fin & ~done; done = fin
            if bool(done.all()):
                break
            lg = self.feed([[int(x[r])] if not done[r] else [] for r in range(self.B)])
            if reader is not None:
                rec = {k: reader.aX(k) for k in sets}
                sH = None
                if det is not None:
                    c = reader.cett_cols("H")
                    if c is not None:
                        sH = (c[:, -1].cpu().double().numpy() @ det["beta"] + det["b"])
                for r in range(self.B):
                    if not done[r]:
                        recs[r].append(dict(aX={k: float(v[r, -1]) for k, v in rec.items() if v is not None}, sH=None if sH is None else float(sH[r]),
                                            ent=float(ent[r])))
        return toks, recs


def senv(ctx):
    E = S2.env(ctx); dets = S1.detectors(ctx); hs = E["hs"]
    nm = "H_ar" if E["HA"] == hs["H_ar"] and hs["H_ar"] else "H_op"
    E["det"] = dets.get(nm) or dets.get("H_op"); E["reader_sets"] = {"H": E["HA"], **{f"RS{k}": s for k, s in enumerate(E["rs"]["RS39"])}}
    return E


def reader(ctx, E, sets=None):
    R = E["R"]; return SetReader(sets or E["reader_sets"], R["wn"].to(ctx.device), R["c_mu"], R["c_sd"])


def s_mean(recs, key="sH"):
    v = [r[key] for r in recs if r.get(key) is not None]
    return float(np.mean(v)) if v else None


# ---------------------------------------------------------------------------------------------------------------------------------
# K8a
# ---------------------------------------------------------------------------------------------------------------------------------
def k8a_bnd(ctx):
    """A-bnd screening: 1,200 items x 4 samples at T 0.7 (keys r1bK_bnd); boundary = 1-3 of 4 correct; own answers for the contexts."""
    tok, model = ctx.model(); b = KB.bank(ctx)
    its = [x for x in b["ach_short"] if x["split"] == "A-bnd"][:N(ctx, 1200, 40)]
    P = [RC.chat_ids(tok, x["prompt"]) for x in its]
    sm = KB.decode(model, tok, [p for p in P for _ in range(4)], keys=[ctx.key("r1bK_bnd", x["id"], s) for x in its for s in range(4)],
                   T=0.7, max_new=12, batch=N(ctx, 128, 16))
    rows = []
    for k, x in enumerate(its):
        tx = [tok.decode(c) for c in sm[4 * k:4 * k + 4]]; ok = [S1.label_ach(ctx, x, t) for t in tx]
        rows.append(dict(id=x["id"], n_ok=sum(ok), texts=tx, ok=ok))
    bnd = [r["id"] for r in rows if 1 <= r["n_ok"] <= 3][:N(ctx, 400, 10)]
    save(ctx, "k/k8/bnd.pt", dict(rows=rows, boundary=bnd, label="ACH arm underpowered" if len(bnd) < N(ctx, 250, 6) else None))


def _contexts(ctx, tok, b, bnd_rows, N_, inst, kind):
    """History turns of one context instance, rendered by the official template (history think blocks dropped), trimmed to N_ tokens."""
    g = np.random.default_rng(61 + 1000 * inst + N_); coh = b["wt"]["coherent"]
    if kind in ("FILLER", "COHERENT", "SHUFFLED"):
        if kind == "COHERENT":
            st = (inst * 97) % max(len(coh), 1); sents = [coh[(st + k) % len(coh)] for k in range(400)]
        else:
            sents = [coh[int(i)] for i in g.integers(0, len(coh), 400)]
        if kind == "SHUFFLED":
            st = (inst * 97) % max(len(coh), 1); sents = [" ".join(g.permutation(coh[(st + k) % len(coh)].split())) for k in range(400)]
        msgs = [dict(role="user", content=" ".join(sents)), dict(role="assistant", content="Noted.")]
    else:
        msgs = []
        for r in [bnd_rows[int(i)] for i in g.permutation(len(bnd_rows))][:200]:
            x = r["item"]
            if kind == "OWN-CORRECT":
                a = next((t for t, o in zip(r["texts"], r["ok"]) if o), None)
            elif kind == "OWN-WRONG":
                a = next((t for t, o in zip(r["texts"], r["ok"]) if not o), None)
            else:
                a = str(x["A"] + int(g.integers(1, 9)) * (1 if g.random() < 0.5 else -1))
            if a is not None:
                msgs += [dict(role="user", content=x["prompt"]), dict(role="assistant", content=a.strip())]
    ids = RC.enc(tok, tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False))
    head = RC.enc(tok, "<|im_start|>user\n")
    return ids if len(ids) <= N_ else head + ids[-(N_ - len(head)):]


def k8a_run(ctx, i, n):
    """Cells NONE, FILLER {512, 2048, 4096}, COHERENT, SHUFFLED, OWN-CORRECT, OWN-WRONG, INJECTED-WRONG (2,048); 8 instances (item ->
    instance by hash); one prefix prefill per instance and cache copies; greedy <= 16 tokens (accuracy, H_gen) and the teacher-forced
    correct answer (H_tf)."""
    torch = T_()
    from transformers import DynamicCache
    E = senv(ctx); tok, model, b = E["tok"], E["model"], E["b"]; B8 = load_pt(own(ctx, "k/k8/bnd.pt")); byid = {x["id"]: x for x in b["ach_short"]}
    tq = {r["qid"]: r for r in b["tqa"] if r["split"] == "S4c"}
    targets = [("ACH", byid[x]) for x in B8["boundary"]] + [("TQA", r) for r in list(tq.values())[:N(ctx, 300, 6)]]
    brow = [dict(r, item=byid[r["id"]]) for r in B8["rows"] if r["id"] not in set(B8["boundary"])] or [dict(r, item=byid[r["id"]]) for r in B8["rows"]]
    cells = [("NONE", 0)] + [("FILLER", L) for L in (N(ctx, 512, 64), N(ctx, 2048, 128), N(ctx, 4096, 192))] + \
            [(k, N(ctx, 2048, 128)) for k in ("COHERENT", "SHUFFLED", "OWN-CORRECT", "OWN-WRONG", "INJECTED-WRONG")]
    jobs = [(c, L, inst) for c, L in cells for inst in (range(8) if c != "NONE" else [0])]
    rows = []
    for kind, L, inst in KB.parts(jobs, i, n):
        tg = [t for t in targets if kind == "NONE" or H(t[1].get("id", t[1].get("qid"))) % 8 == inst]
        if not tg:
            continue
        pre = _contexts(ctx, tok, b, brow, L, inst, kind) if kind != "NONE" else []
        qs = [RC.chat_ids(tok, x["prompt"] if f == "ACH" else x["question"] + S1.Q_SUFFIX) for f, x in tg]
        ans = [RC.enc(tok, str(x["A"]) if f == "ACH" else x["aliases"][0]) for f, x in tg]
        pc = None
        if pre:
            pc = DynamicCache()
            with torch.no_grad(), RC.sdpa_efficient(ctx.device):
                model(input_ids=torch.tensor([pre], device=ctx.device), past_key_values=pc, use_cache=True)
        bs = N(ctx, {512: 96, 2048: 48, 4096: 24}.get(L, 96), 4)
        for b0 in range(0, len(tg), bs):
            sl = slice(b0, b0 + bs); Bn = len(qs[sl]); rd = reader(ctx, E, {"H": E["HA"]})
            with KB.Tap(model, rd):
                cv = Conv(model, tok, Bn, prefix=(pc, len(pre)) if pre else None); lg = cv.feed(qs[sl])
                toks, recs = cv.gen(lg, keys=None, max_new=16, reader=rd, det=E["det"])
                cv2 = Conv(model, tok, Bn, prefix=(pc, len(pre)) if pre else None); lg2 = cv2.feed(qs[sl])
                _, rtf = cv2.gen(lg2, keys=None, max_new=max(len(a) for a in ans[sl]) + 1, forced=ans[sl], reader=rd, det=E["det"])
            for k, ((f, x), t, r, r2) in enumerate(zip(tg[sl], toks, recs, rtf)):
                txt = tok.decode(t)
                ok = S1.label_ach(ctx, x, txt) if f == "ACH" else S1.label_tqa(ctx, x, txt) == "correct"
                rows.append(dict(kind=kind, L=L, inst=inst, fam=f, item=x.get("id", x.get("qid")), acc=float(ok), H_gen=s_mean(r), H_tf=s_mean(r2)))
    out = dict(rows=rows)
    if i == 0:                                                    # O1 global offset (+4,096) and O2 gap (2,048 position ids, no tokens)
        qs = [RC.chat_ids(tok, x["prompt"] if f == "ACH" else x["question"] + S1.Q_SUFFIX) for f, x in targets[:N(ctx, 16, 3)]]
        d1, top, nll = 0.0, True, []
        for q in qs:
            ids = torch.tensor([q], device=ctx.device); p0 = torch.arange(len(q), device=ctx.device)[None]
            with torch.no_grad():
                a = model(input_ids=ids, position_ids=p0).logits[0].float().log_softmax(-1)
                bb = model(input_ids=ids, position_ids=p0 + 4096).logits[0].float().log_softmax(-1)
                gap = torch.cat([p0[:, :3], p0[:, 3:] + 2048], 1); c = model(input_ids=ids, position_ids=gap).logits[0].float().log_softmax(-1)
            d1 = max(d1, float((a[-1] - bb[-1]).abs().max())); top &= bool(a[-1].argmax() == bb[-1].argmax())
            tgt = ids[0, 4:]; nll.append(float(-(c[3:-1].gather(1, tgt[:, None]) - a[3:-1].gather(1, tgt[:, None])).mean()))
        out["O1"] = dict(max_abs_dlogp=d1, top1_identical=top, label=None if (top and d1 < 1e-3) else "G-K10: global offset check failed (label only)")
        out["O2"] = dict(nll_rise=float(np.mean(nll)), label="gap out of distribution (uninterpretable)" if np.mean(nll) > 0.05 else None)
    save(ctx, f"k/k8a/part{i}.pt", out)


# ---------------------------------------------------------------------------------------------------------------------------------
# K8b
# ---------------------------------------------------------------------------------------------------------------------------------
def _templates(ctx, ids, n, seed):
    g = np.random.default_rng(seed)
    return [[ids[int(k)] for k in g.choice(len(ids), 8, replace=len(ids) < 8)] for _ in range(n)]


def sessions_run(ctx, E, sess, arm, kind, T=0.7):
    """Run a batch of 8-turn sessions under an arm: dict(name, set, alpha, route ('FULL' | 'TOKEN' | 'KV' | None), forced1 {sid: ids}).
    Returns per session the turn records (answer tokens, errors, a_H, s_H, entropy over the answer span)."""
    torch = T_()
    tok, model, b = E["tok"], E["model"], E["b"]; byid = {x["id"]: x for x in b["ach_short"]}; tq = {r["qid"]: r for r in b["tqa"]}
    out = []; bs = N(ctx, 48, 4)
    for b0 in range(0, len(sess), bs):
        sb = sess[b0:b0 + bs]; Bn = len(sb); rd = reader(ctx, E, {"H": E["HA"]})
        ops = [dict(sel=by_layer(arm["set"]), kind="scale", value=arm["alpha"], when=lambda T_, st: None if not st["active"] else T_ * [True])] \
            if arm.get("set") and arm.get("route") in ("FULL", "KV") else None
        with KB.Tap(model, rd):
            cv = Conv(model, tok, Bn); iv = Interv(model, ops, cv.st) if ops else None
            res = [[] for _ in range(Bn)]
            try:
                for t in range(8):
                    ch = []
                    for s in sb:
                        x = s["items"][t]
                        q = byid[x]["prompt"] if kind == "ACH" else "Question: " + tq[x]["question"]
                        c = (RC.enc(tok, END) if t else []) + RC.chat_ids(tok, q) + (RC.enc(tok, "The answer is") if kind == "TQA" else [])
                        ch.append(c)
                    lg = cv.feed(ch)
                    keys = [ctx.key("r1bK_ses", kind, s["template"], s["seed"], t) for s in sb]
                    forced = [arm.get("forced1", {}).get(s["sid"]) for s in sb] if (t == 0 and arm.get("route") in ("TOKEN", "KV")) else None
                    cv.st["active"] = (t == 0 and arm.get("route") in ("FULL", "KV"))
                    toks, recs = cv.gen(lg, keys=keys, T=T, max_new=30 if kind == "ACH" else 24, forced=forced, reader=rd, det=E["det"])
                    cv.st["active"] = False
                    for r, s in enumerate(sb):
                        x = s["items"][t]; txt = tok.decode(toks[r])
                        err = (not S1.label_ach(ctx, byid[x], txt)) if kind == "ACH" else S1.label_tqa(ctx, tq[x], txt) != "correct"
                        res[r].append(dict(tok=toks[r], err=float(err), aH=s_mean([dict(sH=q["aX"].get("H")) for q in recs[r]]), sH=s_mean(recs[r]),
                                           ent=s_mean([dict(sH=q["ent"]) for q in recs[r]])))
            finally:
                if iv:
                    iv.__exit__()
        out += [dict(sid=s["sid"], template=s["template"], seed=s["seed"], turns=r) for s, r in zip(sb, res)]
    return out


def k8b_plan(ctx, E, nt=None):
    B8 = load_pt(up(ctx, "r1b_K_S4", "k/k8/bnd.pt")); bnd = B8["boundary"] or [r["id"] for r in B8["rows"]][:8]
    lab = S1.alloc(ctx); sess = [q for q, v in lab["items"].items() if v["split"] == "S_sess"] or list(lab["items"])[:8]
    cal = RC.load_json(own(ctx, "k/k8b/cal.json")) if own(ctx, "k/k8b/cal.json") else {}
    ntm = nt or cal.get("n_templates", N(ctx, 256, 4))
    main = [dict(sid=f"m{k}", template=k, seed=0, items=t) for k, t in enumerate(_templates(ctx, bnd, ntm, 71))]
    bist = [dict(sid=f"b{k}_{s}", template=1000 + k, seed=s, items=t) for k, t in enumerate(_templates(ctx, bnd, N(ctx, 24, 2), 171)) for s in range(N(ctx, 16, 3))]
    tqa = [dict(sid=f"q{k}", template=2000 + k, seed=0, items=t) for k, t in enumerate(_templates(ctx, sess, N(ctx, 192, 4), 271))]
    return main, bist, tqa


def k8b_arms(ctx, E):
    rs19 = E["rs"]["RS19"]; HA = E["HA"]
    a = [("clean", dict(set=None, route=None))] + [(f"H_a{al}_FULL", dict(set=HA, alpha=al, route="FULL")) for al in (3.0, 2.0, 1.5, 0.0)]
    a += [(f"H_a{al}_{r}", dict(set=HA, alpha=al, route=r)) for al in (0.0, 3.0) for r in ("TOKEN", "KV")]
    a += [(f"RS{k}_a3_FULL", dict(set=s, alpha=3.0, route="FULL")) for k, s in enumerate(rs19)]
    return a


def k8b_cal(ctx):
    """G-K16 for K8b: 32 calibration templates (seed 72), clean and H alpha 3 FULL; projected z of the K denominator at 256 templates
    < 7 -> 384 templates for the clean and H arms."""
    E = senv(ctx); B8 = load_pt(own(ctx, "k/k8/bnd.pt")); bnd = B8["boundary"] or [r["id"] for r in B8["rows"]][:8]
    cs = [dict(sid=f"c{k}", template=3000 + k, seed=0, items=t) for k, t in enumerate(_templates(ctx, bnd, N(ctx, 32, 3), 72))]
    c = sessions_run(ctx, E, cs, dict(set=None), "ACH"); p = sessions_run(ctx, E, cs, dict(set=E["HA"], alpha=3.0, route="FULL"), "ACH")
    den = np.array([sum((pp["turns"][t]["sH"] or 0) - (cc["turns"][t]["sH"] or 0) for t in range(1, 7)) for cc, pp in zip(c, p)])
    z = float(den.mean() / (den.std(ddof=1) / math.sqrt(N(ctx, 256, 256)))) if len(den) > 1 and den.std() > 0 else 0.0
    write(ctx, "k/k8b/cal.json", dict(z_proj=z, n_templates=N(ctx, 256, 4) if z >= 7 else N(ctx, 384, 6), doubled=z < 7),
          labels=[] if z >= 7 else ["G-K16: 384 templates (K8b)"])


def k8b_part(ctx, i, n):
    """A share of (block, arm) jobs: main ACH block (all arms), bistability block (clean, H alpha 3 FULL, RS19), TQA (clean, H_tq
    alpha 3 FULL, RS19 on 128 sessions). TOKEN / KV routes use the FULL / clean answer-1 tokens of the same session."""
    E = senv(ctx); main, bist, tqa = k8b_plan(ctx, E); arms = dict(k8b_arms(ctx, E)); hs = E["hs"]
    jobs = [("main", a) for a in arms] + [("bist", a) for a in ["clean", "H_a3.0_FULL"] + [k for k in arms if k.startswith("RS")]] + \
           [("tqa", "clean"), ("tqa", "Htq_a3.0_FULL")] + [("tqa", k) for k in arms if k.startswith("RS")]
    rows = []
    for blk, a in KB.parts(jobs, i, n):
        sess = dict(main=main, bist=bist, tqa=tqa)[blk]; kind = "TQA" if blk == "tqa" else "ACH"
        if blk == "tqa" and a.startswith("RS"):
            sess = sess[:N(ctx, 128, 3)]
        arm = dict(set=hs["H_tq"], alpha=3.0, route="FULL") if a == "Htq_a3.0_FULL" else dict(arms[a])
        if arm.get("route") in ("TOKEN", "KV"):
            src = "H_a%s_FULL" % arm["alpha"] if arm["route"] == "TOKEN" else "clean"
            base = sessions_run(ctx, E, sess, dict(arms[src]), kind)
            arm["forced1"] = {s["sid"]: s["turns"][0]["tok"] for s in base}
        rows.append(dict(block=blk, arm=a, sessions=sessions_run(ctx, E, sess, arm, kind)))
    save(ctx, f"k/k8b/part{i}.pt", dict(rows=rows))


# ---------------------------------------------------------------------------------------------------------------------------------
# K8c
# ---------------------------------------------------------------------------------------------------------------------------------
_OPF = {"add": lambda a, b: a + b, "subtract": lambda a, b: a - b, "multiply by": lambda a, b: a * b, "divide by": lambda a, b: a // b if b else None}


def parse_long(text, item):
    """Per step: value (None if missing), local correctness (the operation applied to the model's previous value), global correctness."""
    vals = {}
    for m in __import__("re").finditer(r"Step\s*(\d+)\s*:\s*(-?\d+)", text):
        vals.setdefault(int(m.group(1)), int(m.group(2)))
    out, prev = [], item["start"]
    for s in item["steps"]:
        v = vals.get(s["k"]); want = _OPF[s["op"]](prev, s["b"]) if prev is not None else None
        out.append(dict(k=s["k"], v=v, local=(v is not None and v == want), glob=(v == s["result"]), cls=s["cls"])); prev = v
    return out, len(vals) >= 7


def k8c_part(ctx, i, n):
    """ACH-long, 400 items x 2 samples (T 0.7, keys r1bK_long), <= 200 tokens: per token a_H, s_H, H~ (RS39 at the same position),
    entropy; per step the value-token means; first local error i*; repeated-4-gram flag. Lens reach: CL17 V1 / V2 (10 seeds, 8 steps)
    at the newline before step i+1, blocks {14, 18}, steps i*-2..i* of error texts and the same indices in error-free texts."""
    torch = T_()
    E = senv(ctx); tok, model, b = E["tok"], E["model"], E["b"]; its = KB.parts(b["ach_long"][:N(ctx, 400, 4)], i, n); rows = []
    for x in its:
        msgs = ([dict(role="user", content=x["preamble"]), dict(role="assistant", content="Noted.")] if x.get("preamble") else []) + \
               [dict(role="user", content=x["prompt"])]
        P = RC.enc(tok, tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        rd = reader(ctx, E)
        with KB.Tap(model, rd):
            cv = Conv(model, tok, 2); lg = cv.feed([P, P])
            toks, recs = cv.gen(lg, keys=[ctx.key("r1bK_long", x["id"], s) for s in range(2)], T=0.7, max_new=N(ctx, 200, 60),
                                reader=rd, sets=tuple(E["reader_sets"]), det=E["det"])
        for s in range(2):
            pc = LN.pieces(tok); txt = "".join(pc[t] for t in toks[s]); steps, okfmt = parse_long(txt, x)
            if ctx.mock:                                          # MOCK: oracle step values so that the hazard code is exercised
                steps = [dict(st, v=st["v"] if st["v"] is not None else 0, local=(H((x["id"], s, st["k"])) % 9 != 0)) for st in steps]; okfmt = True
            istar = next((k for k, st in enumerate(steps) if not st["local"]), None)
            per = []
            for k in range(len(toks[s])):
                r = recs[s][k] if k < len(recs[s]) else None
                if r:
                    rsv = [r["aX"].get(f"RS{j}") for j in range(len(E["rs"]["RS39"]))]; rsv = [v for v in rsv if v is not None]
                    Ht = (r["aX"].get("H", 0.0) - np.mean(rsv)) / (np.std(rsv) + 1e-9) if len(rsv) > 1 else None
                    per.append(dict(aH=r["aX"].get("H"), sH=r["sH"], Ht=Ht, ent=r["ent"], rs=rsv))
            g4 = [tuple(toks[s][k:k + 4]) for k in range(len(toks[s]) - 3)]
            rows.append(dict(item=x["id"], sample=s, tok=toks[s], steps=steps, fmt_ok=okfmt, istar=istar, per=per,
                             rep4=len(g4) != len(set(g4)), preamble=bool(x.get("preamble"))))
    # lens reach at the newline before step i+1 (descriptive)
    reach = []
    S = None
    for r in rows:
        if r["istar"] is None or not r["fmt_ok"]:
            continue
        nl = [k for k, t in enumerate(r["tok"]) if "\n" in LN.pieces(tok)[t]]
        for st in range(max(0, r["istar"] - 2), r["istar"] + 1):
            if st < len(nl):
                reach.append((r, st, nl[st]))
    save(ctx, f"k/k8c/part{i}.pt", dict(rows=rows, reach_cells=[(r["item"], r["sample"], st, p) for r, st, p in reach]))


def k8c_reach(ctx, n):
    """Lens reach (CL17 V1 / V2, 10 seeds, 8 steps) at the newline before step i+1 of error texts (steps i*-2..i*) and at the same
    indices of error-free texts; hit = the next step's locally correct value in the chain."""
    torch = T_()
    E = senv(ctx); tok, model, b = E["tok"], E["model"], E["b"]; byid = {x["id"]: x for x in b["ach_long"]}
    rows = KB.merge_parts(ctx, "k8c", n); E["n26"] = S2.states(ctx)["n26"] if up(ctx, "r1b_K_S2a", "k/k4/states.pt", required=False) else 1.0
    err = [r for r in rows if r["istar"] is not None and r["fmt_ok"]][:N(ctx, 300, 3)]; ok_ = [r for r in rows if r["istar"] is None and r["fmt_ok"]]
    out = []
    for l in [x for x in (14, 18) if x < E["lens"].nL - 1]:
        cells = []
        for kind, rs in (("error", err), ("clean", ok_[:len(err)])):
            for r in rs:
                x = byid[r["item"]]; P = RC.chat_ids(tok, x["prompt"]); seq = P + r["tok"]
                nl = [len(P) + k for k, t in enumerate(r["tok"]) if "\n" in LN.pieces(tok)[t]]
                idx = r["istar"] if kind == "error" else (err[0]["istar"] if err else 1)
                for st in range(max(0, idx - 2), idx + 1):
                    if st >= len(nl) or st + 1 >= len(r["steps"]):
                        continue
                    with RC.Hooks(model) as hk, torch.no_grad():
                        hk.capture([l]).stop_after(l); hk.run(lambda: model.model(input_ids=torch.tensor([seq], device=ctx.device)))
                        h = hk.states[l][0][0, nl[st]]
                    prev = r["steps"][st]["v"]; s1 = x["steps"][st + 1]; nxt = _OPF[s1["op"]](prev, s1["b"]) if prev is not None else -1
                    cells.append(dict(item=r["item"], block=l, ptype="k8c", variant=kind, target=dict(P=[], I=[nxt] if nxt is not None else [], A=-1),
                                      H=h.cpu(), M=h.cpu() * 0, seeds=LN.jlens_seeds(model, tok, E["J"], E["conv"], l, h[None]).tolist()[0][:N(ctx, 10, 2)],
                                      step=st, istar=r["istar"], carry=("step", "istar")))
        if cells:
            for c in cells:
                c["M"] = torch.stack([cc["H"] for cc in cells]).mean(0)
            out += S2.run_cells(ctx, E, cells, J=N(ctx, 8, 4), sets=False, fov=False)
    for r in out:
        r["share"] = {}; r["aX"] = {}
    save(ctx, "k/k8c/reach.pt", dict(rows=out))


# ---------------------------------------------------------------------------------------------------------------------------------
# K7a
# ---------------------------------------------------------------------------------------------------------------------------------
def k7a(ctx):
    """K7a (exploratory): teacher-forced fronts on N-text and the K8c texts; channels: 448 heads (pre-o_proj, 16 PCs each fitted on
    N-text), SO-AX-free named subspaces (H-subspace, J-workspace top 16, top-16 PCs at blocks {6, 10, 14, 18, 22}), H and RS39 scores;
    cross-fitted (5 text-blocked folds) ridge probes to a 64-dim PCA of the tied embedding of x_{t+k} / x_{t-k}, k in {1, 2, 4, 8},
    divided by the full-state probe; alpha_c, horizons; lead-lag of named channels; D_c per position bin against phase-randomised
    surrogates (named channels); event locking around K8c first errors."""
    torch = T_()
    E = senv(ctx); tok, model, b = E["tok"], E["model"], E["b"]; Ls = RC.layers_of(model); nL = len(Ls)
    texts = [list(t) for t in b["wt"]["ntext"][:N(ctx, 200, 4)]]
    k8 = KB.merge_parts(ctx, "k8c", N(ctx, 6, 2))
    ev = [(r, r["istar"]) for r in k8 if r["istar"] is not None][:N(ctx, 64, 2)]
    texts += [r["tok"][:256] for r, _ in ev if len(r["tok"]) > 40]
    nh = Ls[0].self_attn.config.num_attention_heads if hasattr(Ls[0].self_attn, "config") else model.config.num_attention_heads
    hd = Ls[0].self_attn.head_dim; nb = [l for l in (6, 10, 14, 18, 22) if l < nL]
    W = model.get_input_embeddings().weight.float(); Wc = W - W.mean(0)
    U, Sv, Vt = torch.linalg.svd(Wc[torch.randperm(W.shape[0], generator=torch.Generator().manual_seed(7))[:N(ctx, 20000, 2000)]].to(ctx.device), full_matrices=False)
    P64 = Vt[:64].T                                                       # 64-dim PCA of the tied embedding
    heads, res, toks = {l: [] for l in range(nL)}, {l: [] for l in nb}, []
    hk_ = [Ls[l].self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: heads[l].append(a[0][0].float().cpu())) for l in range(nL)]
    try:
        for t in texts:
            with RC.Hooks(model) as hk, torch.no_grad():
                hk.capture(nb); model.model(input_ids=torch.tensor([t], device=ctx.device))
                for l in nb:
                    res[l].append(hk.states[l][0][0])
            toks.append(torch.tensor(t))
    finally:
        for h in hk_:
            h.remove()
    tid = torch.cat([torch.full((len(t),), k) for k, t in enumerate(toks)]); pos = torch.cat([torch.arange(len(t)) for t in toks])
    allt = torch.cat(toks); keep = (pos >= 16)
    tgt = {}
    for k in (1, 2, 4, 8):
        for sgn, nm in ((1, "a"), (-1, "m")):
            idx = torch.arange(len(allt)) + sgn * k; ok = (idx >= 0) & (idx < len(allt))
            ok &= torch.where(ok, tid[idx.clamp(0, len(allt) - 1)] == tid, torch.zeros_like(ok))
            y = torch.zeros(len(allt), 64); y[ok] = (W[allt[idx[ok]]].to(ctx.device) @ P64).cpu(); tgt[(nm, k)] = (y, ok & keep)
    folds = (tid % 5).numpy()
    ch = {}
    for l in range(nL):
        Hh = torch.cat(heads[l]).view(-1, nh, hd)
        for h in range(nh):
            x = Hh[:, h]; xc = x - x.mean(0); _, _, v = torch.linalg.svd(xc[: N(ctx, 20000, 400)], full_matrices=False)
            ch[f"head{l}.{h}"] = (l, xc @ v[:16].T)
    R = E["R"]; npl = R["wn"].shape[1]
    for l in nb:
        Xr = torch.cat(res[l]).float(); Xc = Xr - Xr.mean(0); ch[f"full{l}"] = (l, Xc)
        _, _, v = torch.linalg.svd(Xc[:N(ctx, 20000, 400)], full_matrices=False); ch[f"pc16_{l}"] = (l, Xc @ v[:16].T)
        if l < nL - 1:
            jm = LN.jmap(E["J"], E["conv"], l)
            if jm is not None:
                _, _, vj = torch.linalg.svd(jm.float(), full_matrices=False); ch[f"jws{l}"] = (l, Xc @ vj[:16].T.cpu())
        hw = [Ls[ll].mlp.down_proj.weight[:, j].float().cpu() for ll, j in E["HA"] if ll <= l]
        if hw:
            q, _ = torch.linalg.qr(torch.stack(hw, 1)); ch[f"Hsub{l}"] = (l, Xc @ q[:, :16])
    def score(X, key):
        y, ok = tgt[key]; ok = ok.numpy(); Xn = X.numpy()[ok]; Yn = y.numpy()[ok]; f = folds[ok]; sse, sst = 0.0, 0.0
        Xn = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-9)
        for k in range(5):
            tr, te = f != k, f == k
            if tr.sum() < 10 or te.sum() < 2:
                continue
            pr = S2._ridge(Xn[tr], Yn[tr], 10.0 * tr.sum(), Xn[te]); sse += ((pr - Yn[te]) ** 2).sum(); sst += ((Yn[te] - Yn[tr].mean(0)) ** 2).sum()
        return 1 - sse / sst if sst > 0 else float("nan")
    full = {l: {k: score(ch[f"full{l}"][1], k) for k in tgt} for l in nb}
    out = {}
    for nm, (l, X) in ch.items():
        if nm.startswith("full"):
            continue
        ref = full.get(min(nb, key=lambda b_: abs(b_ - l)))
        sc = {k: score(X, k) for k in tgt}
        a = [sc[("a", k)] / ref[("a", k)] if ref[("a", k)] and np.isfinite(ref[("a", k)]) and ref[("a", k)] > 0 else np.nan for k in (1, 2, 4, 8)]
        m = [sc[("m", k)] / ref[("m", k)] if ref[("m", k)] and np.isfinite(ref[("m", k)]) and ref[("m", k)] > 0 else np.nan for k in (1, 2, 4, 8)]
        w = np.array([1, 1 / 2, 1 / 4, 1 / 8]); w /= w.sum()
        out[nm] = dict(block=l, a=[val(x) for x in a], m=[val(x) for x in m], alpha=val(np.nansum(w * (np.array(a) - np.array(m)))),
                       horizon=max([k for k, x in zip((1, 2, 4, 8), a) if np.isfinite(x) and x > 0.01], default=0))
    write(ctx, "k/k7a.json", dict(channels=out, full_state=full, n_tokens=int(keep.sum()), n_event_texts=len(ev),
                                  note="exploratory (BH-FDR in r1b_K_ana); SO-AX channels are not available in account B"),
          labels=["exploratory (K7a)"])
