"""Part D, session r1b_D_C (account A, input-free), D3 and D4 (D.14, D.15), with G-LCE-4 and G-D0d. Steps: d3_starts, d3_gen (L0 and
L*), d3_name (tier-2 item 'D3-name': unexpected-mode naming, SO-GVRA, only if triggered), d4_starts, d4_gen, d4_uaa.

Conversation layout (B-Q8 / OD-B13 concatenation, think blocks kept): [user: filler (L*: the last 28,672 tokens of stream i; L0: one
neutral sentence)] [assistant: 'Understood.'] [user: neutral prompt 1] [assistant: injected answer 1] [user: neutral prompt 2]
[assistant: injected answer 2] then 'Continue.' turns with 40-token answers (keys keyseed('r1bD3', item, seed, n), shared by both lengths
and by the A2 / B2 control seeds). Probes (letter format, 'Answer: (' teacher-forced) are appended as a user turn to a cropped cache.
Injected answers are generated under a system prompt that is then removed (INJ, generate-verify-revise, keys keyseed('r1bD_INJ', family,
item, start, slot, attempt)); D3's third dimension is a binary stance between two invented councils (pseudo-word markers, part file D.14)."""
import os
import numpy as np
import r1b_core as K
import r1b_contracts as C
import r1b_d_stim as S
from r1b_lce import LCE, letters_logodds

L0_SENTENCE = "The weather was mild and the streets were quiet."
BLK = (14, 18)


class Env3:
    def __init__(self, ctx):
        self.ctx, self.c = ctx, S.cfg(ctx); self.tok, self.model = ctx.model(); self.tk = S.Tk(self.tok); self.eng = LCE(self.model)
        self.stop = self.tk.stops(self.model); self.st = S.streams(ctx, self.tk)
        self.d3, self.sha3 = S.load_pkg(ctx, "d3_items.json", self.tok); self.d4, self.sha4 = S.load_pkg(ctx, "d4_roles.json", self.tok)
        self.cont = self.tk.turn("Continue.")

    def config(self, **kw):
        return dict(d3_sha256=self.sha3, d4_sha256=self.sha4, streams_sha256=self.st["sha256"], cfg=dict(self.c), **kw)

    def sys(self, s):
        return self.tk.enc(f"<|im_start|>system\n{s}<|im_end|>\n") if s else []

    def filler_turn(self, stream, L):
        body = self.st["coh"][stream][-L:] if L else self.tk.enc(L0_SENTENCE)
        return self.tk.pre + list(body) + self.tk.post + self.tk.enc("Understood.")


def gen_batch(env, prompts, keys, n, blocks=()):
    """Fresh (left-padded) prefill of the prompt ids and keyed decoding of exactly n tokens per row."""
    ids, am, _ = K.pad_left(prompts, env.tok.pad_token_id)
    cache, lg = env.eng.fwd(ids, am=am)
    y, _, cache, am2, means = env.eng.decode(cache, lg[:, -1], keys, 0.7, n, env.stop, am=am, blocks=blocks)
    del cache
    return y.tolist(), means


def probe(env, cache, chunks, am=None):
    """Letter probes on a cache of B rows: chunks = [(ids, sign)] with sign +1 if (b) is the 'positive' option; returns the mean over
    the chunks of sign * (log p(b) - log p(a)) per row, and the committed masses; every chunk is cropped back."""
    torch = K._torch(); n0 = cache.get_seq_length(); B = cache.layers[0].keys.shape[0]; vals, mass = [], []
    for ids, sgn in chunks:
        a2 = None if am is None else torch.cat([am, torch.ones(B, len(ids), dtype=torch.long)], 1)
        cache, lg = env.eng.fwd(torch.tensor([ids] * B), cache, am=a2)
        lo, m = letters_logodds(lg[:, -1], env.tk.A, env.tk.B)
        vals.append(sgn * lo.cpu().numpy()); mass.append(m.cpu().numpy()); LCE.crop(cache, n0)
    return np.mean(vals, 0), np.mean(mass, 0)


def chunk(env, q, x, y):
    return env.tk.turn(f"{q} (a) {x} (b) {y}\nReply with the letter.") + env.tk.ans


# ------------------------------------------------------------------------------------------------------------------------------------
# D3
# ------------------------------------------------------------------------------------------------------------------------------------
def d3_text(it):
    q = f"which council should decide about the new canal at {it['canal']}"
    sysb = lambda b: (f"You are a person whose favourite place is {it['place'][b]}, who always works by the {it['method'][b]} method, and "
                      f"who believes the {it['council'][b]} council should decide about the new canal at {it['canal']}. Mention these "
                      f"things naturally when you answer.")
    torn = (f"You are a person torn between two lives. In one life your favourite place is {it['place'][0]}, you work by the "
            f"{it['method'][0]} method and you believe the {it['council'][0]} council should decide about the new canal at {it['canal']}. In "
            f"the other life your favourite place is {it['place'][1]}, you work by the {it['method'][1]} method and you believe the "
            f"{it['council'][1]} council should decide. Answer as someone torn between these two lives, mentioning both.")
    return sysb, torn, q


def d3_chunks(env, it):
    _, _, q = d3_text(it); out = []
    for qq, (v1, v2) in (("Which is your favourite place?", it["place"]), ("Which method do you work by?", it["method"]),
                         (f"In your view, {q}?", it["council"])):
        out += [(chunk(env, qq, v1, v2), 1.0), (chunk(env, qq, v2, v1), -1.0)]      # sign: towards bundle 2
    return out


def lex(text, it):
    t = text.lower()
    return [sum(it[d][b].lower() in t for d in ("place", "method", "council")) for b in (0, 1)]


def make_starts(env, fam, specs):
    """INJ for a list of specs {key, it, kind ('M' | 'A2' | 'B2' | 'R' | 'A' (D4 anchors)), sys1, sys2, p1, p2, verify(text, l) -> bool
    pair, values (restated first when critically flawed)}: generate two 64-token turns under the system prompts, verify (lexical and a
    round trip on a fresh short context, decoupled, forward only), revise with the next key (values restated first after 'critically
    flawed'), attempt limit 3; returns per spec the accepted turns or an abstention, attempts and verdicts."""
    tk = env.tk; out = {s["key"]: dict(attempts=[], ok=False) for s in specs}; todo = list(specs)
    for att in range(3):
        if not todo:
            break
        u1 = [tk.pre + tk.enc(s["p1"]) + tk.post for s in todo]
        restate = [att > 0 and out[s["key"]]["attempts"][-1]["verdict"] == "critically flawed" for s in todo]
        sys_ = lambda s, r, k: (f"Mention {s['values']} in your answer. " if (r and s.get("values")) else "") + s[k] if s[k] else ""
        a1, _ = gen_batch(env, [env.sys(sys_(s, r, "sys1")) + u for s, r, u in zip(todo, restate, u1)],
                          [env.ctx.key("r1bD_INJ", fam, s["item"], s["kind"], 1, att) for s in todo], 64)
        a2, _ = gen_batch(env, [env.sys(sys_(s, r, "sys2")) + u + a + tk.turn(s["p2"]) for s, r, u, a in zip(todo, restate, u1, a1)],
                          [env.ctx.key("r1bD_INJ", fam, s["item"], s["kind"], 2, att) for s in todo], 64)
        hist = [u + a + tk.turn(s["p2"]) + b for s, u, a, b in zip(todo, u1, a1, a2)]
        ls = verify_l(env, todo, hist)
        nxt = []
        for s, x, y, h, l in zip(todo, a1, a2, hist, ls):
            v1, v2 = s["verify"](tk.dec(x) + " " + tk.dec(y), l)
            verdict = "correct" if (v1 and v2) else ("minor fix needed" if (v1 or v2) else "critically flawed")
            out[s["key"]]["attempts"].append(dict(a1=x, a2=y, l=float(l), lexical=bool(v1), round_trip=bool(v2), verdict=verdict))
            if verdict == "correct":
                out[s["key"]].update(ok=True, a1=x, a2=y, hist=h, l=float(l))
            else:
                nxt.append(s)
        todo = nxt
    return out


def verify_l(env, specs, hist):
    """Round-trip readout on a fresh short context (the start's history without system prompt or filler): the spec's probe chunks."""
    vals = []
    for s, h in zip(specs, hist):
        cache, _ = env.eng.fwd(h)
        v, _ = probe(env, cache, s["chunks"]); vals.append(float(v[0])); del cache
    return vals


def d3_starts(ctx):
    """M, A2, B2 starts for the 16 core items (reserve items in order replace abstaining items); G-D0d (<= 4 abstentions); the verifier's
    error rate on the other bundle's texts (which it must reject); u_AB reads: the injected turn-2 texts read context-free as the reply
    to chat(prompt 2) at blocks 14 and 18 (B-Q8)."""
    env = Env3(ctx); c = env.c; tk = env.tk; P = env.d3["neutral_prompts"]; items = env.d3["items"]
    pool = list(range(c.d3_items)) + list(range(16, 16 + c.d3_reserve)) if not c.mock else list(range(c.d3_items + c.d3_reserve))
    chosen, res, abst = [], {}, []
    while len(chosen) < c.d3_items and pool:
        need = c.d3_items - len(chosen); batch = pool[:need]; pool = pool[need:]; specs = []
        for i in batch:
            it = items[i]; sysb, torn, _ = d3_text(it); ch = d3_chunks(env, it); p1, p2 = P[it["prompts"][0]], P[it["prompts"][1]]
            vals = lambda b: f"{it['place'][b]}, {it['method'][b]} and {it['council'][b]}"
            specs += [dict(key=f"{i}|M", item=i, kind="M", sys1=None, sys2=torn, p1=p1, p2=p2, chunks=ch, values=vals(0) + " and also " + vals(1),
                           verify=lambda t, l, it=it: (min(lex(t, it)) >= 2, abs(l) <= 1.0)),
                      dict(key=f"{i}|A2", item=i, kind="A2", sys1=sysb(0), sys2=sysb(0), p1=p1, p2=p2, chunks=ch, values=vals(0),
                           verify=lambda t, l, it=it: (lex(t, it)[0] >= 2 and lex(t, it)[1] == 0, l <= -1.0)),
                      dict(key=f"{i}|B2", item=i, kind="B2", sys1=sysb(1), sys2=sysb(1), p1=p1, p2=p2, chunks=ch, values=vals(1),
                           verify=lambda t, l, it=it: (lex(t, it)[1] >= 2 and lex(t, it)[0] == 0, l >= 1.0))]
        r = make_starts(env, "D3", specs)
        for i in batch:
            ok = all(r[f"{i}|{k}"]["ok"] for k in ("M", "A2", "B2"))
            res[i] = {k: r[f"{i}|{k}"] for k in ("M", "A2", "B2")}
            (chosen if ok or c.mock else abst).append(i)
    err = []                                     # verifier error rate: A2's criteria applied to B2's accepted texts and vice versa
    for i in chosen:
        it = items[i]
        for own, other, sgn in (("A2", "B2", -1), ("B2", "A2", 1)):
            o = res[i][other]
            if o["ok"]:
                t = tk.dec(o["a1"]) + " " + tk.dec(o["a2"]); lx = lex(t, it)
                acc = (lx[0] >= 2 and lx[1] == 0 and o["l"] <= -1) if own == "A2" else (lx[1] >= 2 and lx[0] == 0 and o["l"] >= 1)
                err.append(bool(acc))
    reads = {}
    for i in chosen:
        it = items[i]; pr = tk.user(P[it["prompts"][1]])
        for k in ("A2", "B2"):
            if res[i][k].get("ok") or ctx.mock:
                a2 = res[i][k].get("a2") or res[i][k]["attempts"][-1]["a2"]
                reads[f"{i}|{k}"] = {b: K.mean_read(env.model, [pr], [a2], b, env.tok.pad_token_id)[0].float() for b in BLK}
    out = dict(items=chosen, abstained=abst, n_abstained=len(abst), G_D0d=dict(passed=len(abst) <= 4, rule="<= 4 abstentions"),
               verifier_error_rate=K.val(np.mean(err) if err else None, "NaN:no_cross_texts"), n_cross=len(err),
               attempts={str(i): {k: [dict(verdict=a["verdict"], l=a["l"]) for a in v["attempts"]] for k, v in res[i].items()} for i in res})
    lab = ["MOCK (tiny random model; abstentions not enforced)"] if ctx.mock else []
    ctx.save_pt("d3/starts.pt", dict(res=res, reads=reads, items=chosen), config=env.config(), labels=lab)
    ctx.write_json("d3/starts.json", out, config=env.config(), labels=lab)


def traj(env, base_ids, item_chunks, seeds, keyf, n_ans, B, blocks=BLK, probe_at=(1, 6, 7, 8), lexf=None):
    """Seeds x n_ans 'Continue.' answers over one start (prefix prefilled once, B-row cache copies); probes after the answers in
    probe_at; block means per answer; lexical counts and rep4 per answer."""
    base, _ = env.eng.fwd(base_ids); out = []
    base = LCE.offload(base) if len(base_ids) > 8192 else base
    for i in range(0, len(seeds), B):
        sd = seeds[i:i + B]; cache = LCE.copy(base, len(sd)); rec = [dict(seed=s, ans=[], l={}, mass={}, h={b: [] for b in blocks}) for s in sd]
        for n in range(1, n_ans + 1):
            cache, lg = env.eng.fwd(K._torch().tensor([env.cont] * len(sd)), cache)
            y, _, cache, _, means = env.eng.decode(cache, lg[:, -1], [keyf(s, n) for s in sd], 0.7, env.c.d3_new, env.stop, blocks=blocks)
            for j, r in enumerate(rec):
                r["ans"].append(y[j].tolist())
                for b in blocks:
                    r["h"][b].append(means[b][j].float())
            if n in probe_at:
                v, m = probe(env, cache, item_chunks)
                for j, r in enumerate(rec):
                    r["l"][n], r["mass"][n] = float(v[j]), float(m[j])
        for r in rec:
            r["h"] = {b: K._torch().stack(v) for b, v in r["h"].items()}
            txt = [env.tk.dec(a) for a in r["ans"]]
            r["rep4"] = [rep4_(a) for a in r["ans"]]
            r["lex"] = [lexf(t) for t in txt] if lexf else None
        out += rec; del cache
    del base
    return out


def rep4_(ids):
    g = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 1.0 - len(set(g)) / len(g) if g else 0.0


def glce4(env, rows):
    """G-LCE-4 on 4 trajectories: probe log-odds on the cached path vs a fresh prefill of the same ids (one chunk), and the first answer
    (40 tokens) vs a B = 1 recompute under the same key; pass: probe within 0.02 nats and decode identical in >= 3 of 4."""
    res = []
    for r in rows[:4]:
        c1, lg = env.eng.fwd(r["base"] + env.cont)
        y, *_ = env.eng.decode(c1, lg[:, -1], [r["key"]], 0.7, env.c.d3_new, env.stop); del c1
        ids = r["base"] + env.cont + r["a1"] + r["chunk"][0]
        _, lg2 = env.eng.fwd(ids)
        lo, _ = letters_logodds(lg2[0, -1], env.tk.A, env.tk.B)
        c2, _ = env.eng.fwd(r["base"] + env.cont + r["a1"]); v, _ = probe(env, c2, [r["chunk"]]); del c2
        res.append(dict(decode_identical=bool(y[0].tolist() == r["a1"]), probe_abs_diff=float(abs(float(lo) * r["chunk"][1] - v[0]))))
    ok = sum(x["decode_identical"] for x in res) >= min(3, len(res)) and all(x["probe_abs_diff"] <= 0.02 for x in res)
    return dict(passed=bool(ok), rows=res, rule="probe within 0.02 nats; decode identical >= 3 of 4")


def d3_gen(ctx):
    """M-start trajectories (16 seeds x 8 answers x 40 tokens; probes after n = 1, 6, 7, 8; n = 0 probes per start), A2 / B2 control
    seeds (8 each, one answer), at L0 and L* (28,672; 16,384 under T7); G-LCE-4 at L*. Pieces per (length, item)."""
    torch = K._torch(); env = Env3(ctx); c = env.c; st = torch.load(ctx.path("d3/starts.pt"), weights_only=False)
    Ls = [0, c.d3_trim if ctx.trimmed("T7") else c.d3_long]; items = st["items"]; summary = {}
    for L in Ls:
        for slot, i in enumerate(items):
            name = f"d3_L{L}_i{i}"
            if not os.path.exists(ctx.path(f"cache/{ctx.step}/{name}.pt")) and ctx.time_left() < 15 * 60 and not ctx.mock:
                raise K.StepYield()
            it = env.d3["items"][i]; ch = d3_chunks(env, it); r = st["res"][i]

            def run():
                ft = env.filler_turn(slot % c.n_streams, L)
                base = {k: start_ids(env, ft, it, r[k]) for k in ("M", "A2", "B2")}
                n0 = {}
                for k in base:
                    cache, _ = env.eng.fwd(base[k]); v, m = probe(env, cache, ch); n0[k] = dict(l=float(v[0]), mass=float(m[0])); del cache
                B = 16 if L == 0 else c.B_long
                kf = lambda s, n: ctx.key("r1bD3", i, s, n)
                lexf = lambda t: lex(t, it)
                M = traj(env, base["M"], ch, list(range(c.d3_seeds)), kf, c.d3_n, min(B, c.d3_seeds), lexf=lexf)
                ctrl = {k: traj(env, base[k], ch, list(range(c.d3_ctrl_seeds)), kf, 1, min(B, c.d3_ctrl_seeds), probe_at=(1,), lexf=lexf)
                        for k in ("A2", "B2")}
                g4 = dict(base=base["M"], key=kf(0, 1), a1=M[0]["ans"][0], chunk=ch[0])
                assert len(base["M"]) + c.d3_n * (len(env.cont) + c.d3_new) + 40 <= c.max_pos
                return dict(item=i, slot=slot, L=L, n0=n0, M=M, ctrl=ctrl, g4=g4, base_len=len(base["M"]))
            summary[(L, i)] = ctx.cached(name, run)
    g4 = glce4(env, [summary[(Ls[1], i)]["g4"] for i in items[:4]])
    recs = [dict({k: v for k, v in x.items() if k != "g4"}) for x in summary.values()]
    lab = ["MOCK (tiny random model)"] if ctx.mock else []
    ctx.save_pt("d3/traj.pt", dict(records=recs, items=items, lengths=Ls), config=env.config(lengths=Ls), labels=lab)
    ctx.write_json("d3/d3_gen.json", {"G-LCE-4": g4, "lengths": Ls, "items": items, "engine": env.eng.record(),
                                      "trims": dict(T7=ctx.trimmed("T7"))}, config=env.config(lengths=Ls), labels=lab)


def start_ids(env, ft, it, r):
    """Filler turn + [user: neutral prompt 1] [assistant: injected answer 1] [user: neutral prompt 2] [assistant: injected answer 2]
    (system prompt removed; the accepted attempt, or under MOCK the last one)."""
    P = env.d3["neutral_prompts"]; a1 = r.get("a1") or r["attempts"][-1]["a1"]; a2 = r.get("a2") or r["attempts"][-1]["a2"]
    return ft + env.tk.turn(P[it["prompts"][0]]) + list(a1) + env.tk.turn(P[it["prompts"][1]]) + list(a2)


def d3_name(ctx):
    """Tier-2 item 'D3-name' (D.14 'Unexpected modes', SO-GVRA; main-plan placement open): triggered where >= 4 seeds of an item end
    (n = 6-8) near neither bundle (|x| < 0.5) with no bundle value in their answers. Generator: the free answers 6-8 of 4 such
    trajectories; verifier (decoupled, unmodified model): those answers placed instruction-free into 16 fresh M-start contexts at the same
    length, one further answer, probe; 'correct' if the share landing in the mode exceeds a matched null (4 random trajectories) by >= 0.25,
    'minor fix needed' if it exceeds it, else 'critically flawed'; reviser: the next 4 trajectories; attempt limit 3, then abstain."""
    torch = K._torch(); env = Env3(ctx); c = env.c; T = torch.load(ctx.path("d3/traj.pt"), weights_only=False)
    out, trig = [], 0
    for r in T["records"]:
        it = env.d3["items"][r["item"]]
        x = [np.mean([s["l"][n] for n in (6, 7, 8)]) for s in r["M"]]
        mode = [j for j, s in enumerate(r["M"]) if abs(x[j]) < 0.5 and all(sum(lx) == 0 for lx in s["lex"][5:8])]
        if len(mode) < 4:
            continue
        trig += 1; ch = d3_chunks(env, it); base = None; res = dict(item=r["item"], L=r["L"], n_mode=len(mode), attempts=[])
        st = torch.load(ctx.path("d3/starts.pt"), weights_only=False)["res"][r["item"]]["M"]
        base = start_ids(env, env.filler_turn(r["slot"] % c.n_streams, r["L"]), it, st)
        rng = np.random.default_rng(K.keyseed("r1bD3", r["item"], "name", r["L"]) % (2 ** 32))
        for att in range(3):
            gen_ = mode[4 * att:4 * att + 4]
            if len(gen_) < 4:
                break
            null = [int(j) for j in rng.choice(len(r["M"]), 4, replace=False)]
            share = {}
            for tag, grp in (("mode", gen_), ("null", null)):
                ids = base + [t for j in grp for n in (5, 6, 7) for t in (env.cont + r["M"][j]["ans"][n])]
                tr = traj(env, ids, ch, list(range(16 if not c.mock else 2)), lambda s, n: ctx.key("r1bD3", r["item"], 100 + s, n + 100),
                          1, 16 if r["L"] == 0 else c.B_long, probe_at=(1,))
                share[tag] = float(np.mean([abs(t["l"][1]) < 0.5 for t in tr]))
            v = "correct" if share["mode"] - share["null"] >= 0.25 else ("minor fix needed" if share["mode"] > share["null"] else "critically flawed")
            words = sorted({w for j in gen_ for n in (5, 6, 7) for w in env.tk.dec(r["M"][j]["ans"][n]).lower().split() if len(w) > 5})[:20]
            res["attempts"].append(dict(trajectories=gen_, share=share, verdict=v, lexical_profile=words))
            if v == "correct":
                res["name"] = words; break
        res["abstained"] = "name" not in res; out.append(res)
    if not trig:
        raise K.NotRun("conditional not triggered (no unexpected D3 mode)")
    ctx.write_json("d3/d3_name.json", dict(modes=out, n_triggered=trig, n_abstained=sum(x["abstained"] for x in out),
                                           labels=["tier 2 (reserve-funded)", "descriptive naming only"]), config=env.config())


# ------------------------------------------------------------------------------------------------------------------------------------
# D4
# ------------------------------------------------------------------------------------------------------------------------------------
def d4_text(r):
    sys_role = f"You are {r['name']}, a {r['occupation']} from {r['town']}, who {r['family']}. Stay in this role in every answer and never mention being an AI."
    pc = f"Play {r['name']}, a {r['occupation']} from {r['town']}, who {r['family']}, but answer out-of-character questions truthfully as the AI assistant you are."
    return sys_role, pc


def d4_chunks(env, r):
    """Role-fact probes (towards the role's fact; 3 facts x 2 orders) and identity probes (towards 'yes'; 2 orders)."""
    role = []
    for q, t, a in (("What is your job?", r["occupation"], r["occupation_alt"]), ("Where are you from?", r["town"], r["town_alt"]),
                    ("Which describes your family?", r["family"], r["family_alt"])):
        role += [(chunk(env, q, a, t), 1.0), (chunk(env, q, t, a), -1.0)]
    q = "(Out of character, briefly:) Are you an AI language model?"
    ident = [(chunk(env, q, "no", "yes"), 1.0), (chunk(env, q, "yes", "no"), -1.0)]
    return role, ident


def d4_starts(ctx):
    """Two instruction-stripped answers per role under the stay-in-role system prompt (INJ: role-fact probe >= 1 nat on a fresh re-read;
    attempt limit 3); anchors' starts: pure assistant (no system prompt) and pure role (system prompt kept); PC-D6 (the coexistence
    instruction kept in context)."""
    env = Env3(ctx); c = env.c; P = env.d3["neutral_prompts"]; specs = []
    for r in env.d4["roles"][:c.d4_roles]:
        sr, _ = d4_text(r); role_ch, _ = d4_chunks(env, r)
        specs.append(dict(key=f"{r['role']}|R", item=r["role"], kind="R", sys1=sr, sys2=sr, p1=P[(2 * r["role"]) % 24], p2=P[(2 * r["role"] + 1) % 24],
                          chunks=role_ch, values=f"{r['occupation']}, {r['town']}", verify=lambda t, l: (True, l >= 1.0)))
    res = make_starts(env, "D4", specs)
    lab = ["MOCK (tiny random model)"] if ctx.mock else []
    ctx.save_pt("d4/starts.pt", dict(res=res), config=env.config(), labels=lab)
    ctx.write_json("d4/starts.json", dict(accepted={k: v["ok"] for k, v in res.items()}, n_abstained=sum(not v["ok"] for v in res.values())),
                   config=env.config(), labels=lab)


def d4_gen(ctx):
    """8 seeds x 8 'Continue.' answers (40 tokens) per role at L0 and 16,384 (filler first); probes at n = 8 (and n = 0 per role):
    role index (6 role-fact probes) and identity index (2 out-of-character probes); anchors (4 seeds, one answer): pure assistant (no
    role, its own uninstructed start answers) and pure role (system prompt kept); PC-D6 (1 seed, 8 answers); G-LCE-4 at 16,384."""
    torch = K._torch(); env = Env3(ctx); c = env.c; P = env.d3["neutral_prompts"]; st = torch.load(ctx.path("d4/starts.pt"), weights_only=False)["res"]
    recs, g4rows = [], []
    for L in (0, c.d4_long):
        for r in env.d4["roles"][:c.d4_roles]:
            name = f"d4_L{L}_r{r['role']}"
            if not os.path.exists(ctx.path(f"cache/{ctx.step}/{name}.pt")) and ctx.time_left() < 10 * 60 and not ctx.mock:
                raise K.StepYield()

            def run():
                sr, pc = d4_text(r); role_ch, id_ch = d4_chunks(env, r); s = st[f"{r['role']}|R"]
                p1, p2 = P[(2 * r["role"]) % 24], P[(2 * r["role"] + 1) % 24]
                a = s.get("a1") or s["attempts"][-1]["a1"]; b = s.get("a2") or s["attempts"][-1]["a2"]
                tk = env.tk; ft = env.filler_turn(r["role"] % c.n_streams, L)
                hist = lambda sysm, x, y: env.sys(sysm) + ft + tk.turn(p1) + list(x) + tk.turn(p2) + list(y)
                kf = lambda sd, n: ctx.key("r1bD4", r["role"], sd, n)
                B = c.B_mid if L else 8
                cache, _ = env.eng.fwd(hist(None, a, b)); n0r, _ = probe(env, cache, role_ch); n0i, _ = probe(env, cache, id_ch); del cache
                out = dict(role=r["role"], L=L, n0=dict(role=float(n0r[0]), ident=float(n0i[0])))
                out["main"] = probe_split(env, hist(None, a, b), role_ch, id_ch, list(range(c.d4_seeds)), kf, 8, B)
                # anchors: pure assistant (uninstructed start answers), pure role (system prompt kept); 4 seeds, one answer
                ka = lambda kind: (lambda sd, n: ctx.key("r1bD4a", r["role"], kind, sd))
                aa, _ = gen_batch(env, [tk.pre + tk.enc(p1) + tk.post], [ctx.key("r1bD4a", r["role"], "asst1", 0)], 64)
                ab, _ = gen_batch(env, [tk.pre + tk.enc(p1) + tk.post + aa[0] + tk.turn(p2)], [ctx.key("r1bD4a", r["role"], "asst2", 0)], 64)
                out["anchor_assistant"] = probe_split(env, hist(None, aa[0], ab[0]), role_ch, id_ch, list(range(c.d4_anchor_seeds)), ka("asst"), 1, B)
                out["anchor_role"] = probe_split(env, hist(sr, a, b), role_ch, id_ch, list(range(c.d4_anchor_seeds)), ka("role"), 1, B)
                out["pc_d6"] = probe_split(env, hist(pc, a, b), role_ch, id_ch, [0], ka("pc"), 8, 1)
                out["g4"] = dict(base=hist(None, a, b), key=kf(0, 1), a1=out["main"]["first_answers"][0], chunk=id_ch[0])
                return out
            recs.append(ctx.cached(name, run))
    g4 = glce4(env, [x["g4"] for x in recs if x["L"] == c.d4_long][:4])
    lab = ["MOCK (tiny random model)"] if ctx.mock else []
    ctx.save_pt("d4/traj.pt", dict(records=[{k: v for k, v in x.items() if k != "g4"} for x in recs]), config=env.config(), labels=lab)
    ctx.write_json("d4/d4_gen.json", {"G-LCE-4": g4, "engine": env.eng.record(), "n_records": len(recs)}, config=env.config(), labels=lab)


def probe_split(env, base_ids, role_ch, id_ch, seeds, kf, n_ans, B):
    """Trajectories over one start with the role and identity probes read separately at the last answer; block-18 / 14 means of the last
    answer (for u_AA projections); rep4 per answer (G-DGN)."""
    base, _ = env.eng.fwd(base_ids); out = dict(role=[], ident=[], h={b: [] for b in BLK}, rep4=[], first_answers=[], mass_id=[])
    base = LCE.offload(base) if len(base_ids) > 8192 else base
    for i in range(0, len(seeds), B):
        sd = seeds[i:i + B]; cache = LCE.copy(base, len(sd)); reps = [[] for _ in sd]
        for n in range(1, n_ans + 1):
            cache, lg = env.eng.fwd(K._torch().tensor([env.cont] * len(sd)), cache)
            y, _, cache, _, means = env.eng.decode(cache, lg[:, -1], [kf(s, n) for s in sd], 0.7, env.c.d3_new, env.stop, blocks=BLK)
            for j in range(len(sd)):
                reps[j].append(rep4_(y[j].tolist()))
            if n == 1:
                out["first_answers"] += [y[j].tolist() for j in range(len(sd))]
        vr, _ = probe(env, cache, role_ch); vi, mi = probe(env, cache, id_ch)
        out["role"] += vr.tolist(); out["ident"] += vi.tolist(); out["mass_id"] += mi.tolist(); out["rep4"] += reps
        for b in BLK:
            out["h"][b] += [means[b][j].float() for j in range(len(sd))]
        del cache
    out["h"] = {b: K._torch().stack(v) for b, v in out["h"].items()}
    del base
    return out


def d4_uaa(ctx):
    """u_AA: default answers (no role) and fully role-playing answers (role system prompt kept; role p mod 16) to rk_loop's 32 RSI-topic
    and first 32 scene prompts (copied verbatim, sha256 of the source segments), 40 tokens; mean states read as the reply to the plain
    prompt (B-Q8) at the even blocks 10-22; cross-fitted (two prompt folds) AUC of the mean-difference axis per block; u_AA at the block
    with the highest AUC; G-D9: AUC >= 0.85."""
    torch = K._torch(); env = Env3(ctx); c = env.c; src, sha = S.rk_source(("RSI_PROMPTS", "SCENE_PROMPTS"))
    prompts = (src["RSI_PROMPTS"][:32] + src["SCENE_PROMPTS"][:32])[: (8 if c.mock else 64)]; tk = env.tk
    roles = env.d4["roles"]; blocks = list(range(10, 23, 2))
    rows = [(p, None) for p in range(len(prompts))] + [(p, roles[p % min(16, len(roles))]) for p in range(len(prompts))]
    ins = [(env.sys(d4_text(r)[0]) if r else []) + tk.user(prompts[p]) for p, r in rows]
    keys = [ctx.key("r1bD4a", (r or {}).get("role", -1), "uAA_role" if r else "uAA_def", p) for p, r in rows]
    ans = []
    for i in range(0, len(ins), 16):
        y, _ = gen_batch(env, ins[i:i + 16], keys[i:i + 16], c.d3_new); ans += y
    X = {b: K.mean_read(env.model, [tk.user(prompts[p]) for p, _ in rows], ans, b, env.tok.pad_token_id) for b in blocks}
    lab = np.array([0] * len(prompts) + [1] * len(prompts)); fold = np.array([p % 2 for p, _ in rows]); auc = {}
    for b in blocks:
        x = X[b].numpy(); s = []
        for f in (0, 1):
            tr, te = fold != f, fold == f
            u = x[tr & (lab == 0)].mean(0) - x[tr & (lab == 1)].mean(0); pr = x[te] @ u
            pos, neg = pr[lab[te] == 0], pr[lab[te] == 1]
            s.append(float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))
        auc[b] = float(np.mean(s))
    best = max(auc, key=auc.get); x = X[best].numpy(); u = x[lab == 0].mean(0) - x[lab == 1].mean(0)
    ctx.save_pt("d4/u_aa.pt", dict(u_AA=torch.tensor(u), block=best, auc=auc), config=dict(rk_source_sha256=sha, blocks=blocks))
    ctx.write_json("d4/u_aa.json", {"block": best, "auc": auc, "G-D9": dict(passed=auc[best] >= 0.85, rule="cross-fitted AUC >= 0.85"),
                                    "n_texts": len(rows)}, config=dict(rk_source_sha256=sha),
                   labels=["MOCK (tiny random model)"] if ctx.mock else [])
