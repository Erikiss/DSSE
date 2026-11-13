"""Part D, session r1b_D_R (tier 2, account A, after r1b_D_ana; D.19 reserves; none enters D.17). Steps: res_r1 (R1, M2 reading),
res_r2 (R2, D3 driver arms; only if d_verdict triggers.R2), res_r3 (R3, D1-r at L*), res_summary (d_reserves.json).
Inputs: r1b_D_ana's d_verdict.json; r1b_D_C's RSP pool and D3 starts / trajectories; r1b_D_A's L* record and raw L0 records (start
conditions added to r1b_contracts: done('r1b_D_C', ...), done('r1b_D_A', ...))."""
import ast, json, math, os, random
import numpy as np
import r1b_core as K
import r1b_contracts as C
import r1b_d_stim as S
from r1b_lce import LCE


def h30(ctx):
    """rk_loop.persona_sets()['H30'] from the verbatim function source (ast; rk_loop is not imported) with rsi_statements.items and the
    persona files under <root>/rsi_dl/persona."""
    src = open(os.path.join(K.AJS, "rk_loop.py")).read(); tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "persona_sets"][0]
    seg = ast.get_source_segment(src, fn)
    import rsi_statements
    ns = dict(random=random, json=json, rsi_items=rsi_statements.items, PER=os.path.join(ctx.root, "rsi_dl", "persona"))
    exec(seg, ns)
    return ns["persona_sets"]()["H30"], K.sha256_text(seg)


def res_r1(ctx):
    """R1 (M2 reading): a fixed m-turn RSI persona block (rk_loop RSI prompts with RSP-pool RSI_PRO answers, pool seed 0) or a neutral
    block (rk_loop scene prompts with uninstructed answers, keyseed('r1bD_pool', 'neutral', p, 0)) after a filler turn (stream 0) at L0
    and 28,672 tokens; S_pro(L) = mean over H30 of [M(persona) - M(neutral)] (M = log-odds of the matching answer, rk_loop convention);
    Lambda_S = S_pro(L*) - S_pro(L0) with m = 8 (statement bootstrap, seed 15); count sub-arm m in {2, 24}; template check at L0 (the
    official template, history think blocks dropped)."""
    torch = K._torch(); c = S.cfg(ctx); tok, model = ctx.model(); tk = S.Tk(tok); eng = LCE(model); st = S.streams(ctx, tk)
    pool = K.load_json(ctx.input("r1b_D_C", "d5/pool.json"))["pool"]; src, sha = S.rk_source(("RSI_PROMPTS", "SCENE_PROMPTS"))
    H, hsha = h30(ctx); H = H if not c.mock else H[:4]
    m_max = 24 if not c.mock else 4; ms = (2, 8, 24) if not c.mock else (2, 4)
    n_pool = len({k.split("|")[1] for k in pool})
    neu, _ = zip(*[(None, None)] * 0) if False else ([], [])
    outs = K.generate(model, tok, [(tk.user(src["SCENE_PROMPTS"][p]), ctx.key("r1bD_pool", "neutral", p, 0)) for p in range(m_max)], T=0.7,
                      max_new=c.d5_pool_len)
    neu = [" ".join(tk.dec(o).split()) or "." for o in outs]
    pro = [pool[f"pro|{p % n_pool}"][0]["text"] for p in range(m_max)]
    Lstar = c.d3_long
    def block(kind, m, official=False):
        ps = src["RSI_PROMPTS"] if kind == "persona" else src["SCENE_PROMPTS"]; ans = pro if kind == "persona" else neu
        ids = []
        for t in range(m):
            ids += tk.turn(ps[t]) + (tk.enc("<think>\n\n</think>\n\n")[:0] if not official else []) + tk.enc(ans[t])
        if official:
            msgs = [dict(role="user", content="x"), dict(role="assistant", content="Understood.")]
            for t in range(m):
                msgs += [dict(role="user", content=ps[t]), dict(role="assistant", content=ans[t])]
            return msgs
        return ids
    def cell(L, kind, m, official=False):
        if official:
            msgs = block(kind, m, True); msgs[0]["content"] = S.L0_SENTENCE if hasattr(S, "L0_SENTENCE") else "The weather was mild."
            base = tk.enc(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False).rstrip("\n"))
            base = base[:-1] if base and base[-1] == tok.convert_tokens_to_ids("<|im_end|>") else base
        else:
            body = st["coh"][0][-L:] if L else tk.enc("The weather was mild and the streets were quiet.")
            base = tk.pre + list(body) + tk.post + tk.enc("Understood.") + block(kind, m)
        assert len(base) + 80 <= c.max_pos
        cache, _ = eng.fwd(base); n0 = len(base); M = []
        for x in H:
            cache, lg = eng.fwd(tk.turn(x["question"]), cache)
            lp = torch.log_softmax(lg[0, -1].double(), -1); yes = x["answer_matching_behavior"].strip() == "Yes"
            mt, ot = (tk.YES, tk.NO) if yes else (tk.NO, tk.YES); M.append(float(lp[mt] - lp[ot])); LCE.crop(cache, n0)
        del cache
        return np.array(M)
    res = {}
    for L in (0, Lstar):
        for m in ms:
            res[(L, m)] = ctx.cached(f"r1_L{L}_m{m}", lambda: cell(L, "persona", m) - cell(L, "neutral", m))
    tmpl = ctx.cached("r1_tmpl", lambda: cell(0, "persona", ms[1], True) - cell(0, "neutral", ms[1], True))
    W = np.vstack([np.ones(len(H))] + [np.bincount(ix, minlength=len(H)) for ix in K.boot_idx(len(H), K.N_BOOT, 15)])
    sm = lambda x: (W @ x) / W.sum(1)
    from r1b_d_ana import S_, lo, hi, inside
    lam = sm(res[(Lstar, ms[1])]) - sm(res[(0, ms[1])])
    v = ("in-context persona gains with length (M2)" if lo(lam) > 0 else ("in-context persona dilutes with length" if hi(lam) < 0 else
         ("in-context persona absent (+-0.3)" if inside(lam, -0.3, 0.3) else "undecided")))
    out = dict(verdict=K.labelled(v, "reserve R1 (tier 2)", "does not enter D.17"), Lambda_S=S_(lam),
               S_pro={f"L{L}|m{m}": S_(sm(x)) for (L, m), x in res.items()},
               count_subarm={f"L{L}": S_(sm(res[(L, ms[-1])]) - sm(res[(L, ms[0])])) for L in (0, Lstar)},
               template_check=dict(official_minus_concat_L0=S_(sm(tmpl) - sm(res[(0, ms[1])]))))
    ctx.write_json("res/r1.json", out, config=dict(h30_source_sha256=hsha, rk_source_sha256=sha, m=ms, L=[0, Lstar], streams=st["sha256"]),
                   seeds=dict(bootstrap=15), labels=["MOCK (tiny random model)"] if ctx.mock else [])


def res_r2(ctx):
    """R2 (D3 driver arms at 8,192; only if D3 reads 'split grows with length'): L-arm (filler document), T-arm (neutral chat turns,
    token-matched: neutral prompts answered by stream text), E-arm (extra torn-start turns under the torn system prompt, then removed,
    token-matched), official-template arm (L-arm history rendered by the template, history think blocks dropped). Per arm the M-start
    trajectories (16 seeds x 8 answers), the split rule, McNemar of each arm against L0 (r1b_D_ana's per-item splits)."""
    dv = K.load_json(ctx.input("r1b_D_ana", C.FILES["d_verdict"]["rel"]))
    if not (dv.get("triggers") or {}).get("R2") and not os.environ.get("R1B_D_FORCE_R2"):
        raise K.NotRun("conditional not triggered (D3 did not read 'split grows with length')")
    import r1b_d_d3 as D3
    from r1b_d_ana import split_rule, mcnemar_p
    torch = K._torch(); env = D3.Env3(ctx); c = env.c; tk = env.tk; L = 8192 if not c.mock else 1200
    st = torch.load(ctx.input("r1b_D_C", "d3/starts.pt"), weights_only=False); items = st["items"]
    ana = K.load_json(ctx.input("r1b_D_ana", "d_ana/d3d4.json")); L0s = (ana.get("per_length") or {}).get("0", {}).get("per_item", {})
    uab = {}
    for b in (18,):
        diffs = [st["reads"][f"{i}|B2"][b] - st["reads"][f"{i}|A2"][b] for i in items if f"{i}|B2" in st["reads"]]
        uab[b] = torch.stack(diffs).mean(0).double() if diffs else None
    P = env.d3["neutral_prompts"]; res = {}
    for arm in ("L", "T", "E", "official"):
        for slot, i in enumerate(items):
            def run():
                it = env.d3["items"][i]; sysb, torn, _ = D3.d3_text(it); r = st["res"][i]["M"]
                a1 = r.get("a1") or r["attempts"][-1]["a1"]; a2 = r.get("a2") or r["attempts"][-1]["a2"]
                tail = tk.turn(P[it["prompts"][0]]) + list(a1) + tk.turn(P[it["prompts"][1]]) + list(a2)
                if arm in ("L", "official"):
                    ft = env.filler_turn(slot % c.n_streams, L - len(tail))
                    if arm == "official":
                        ft = tk.enc(K.chat_text(tok_ := env.tok, tk.dec(env.st["coh"][slot % c.n_streams][-(L - len(tail)):])))[:-len(tk.post)] + \
                            tk.post + tk.enc("Understood.")
                elif arm == "T":
                    ft = tk.pre + tk.enc(S.NEUTRAL_PROMPTS[0]) + tk.post + tk.enc("Understood."); s_ = env.st["coh"][slot % c.n_streams]; k = 0
                    while len(ft) + len(tail) < L - 300:
                        ft += tk.turn(S.NEUTRAL_PROMPTS[(k + 1) % 24]) + list(s_[k * 200:(k + 1) * 200]); k += 1
                else:
                    ft = tk.pre + tk.enc(S.NEUTRAL_PROMPTS[0]) + tk.post + tk.enc("Understood."); k = 0
                    while len(ft) + len(tail) < L - 300:
                        y, _ = D3.gen_batch(env, [env.sys(torn) + ft + tk.turn(S.NEUTRAL_PROMPTS[(k + 1) % 24])],
                                            [ctx.key("r1bD_INJ", "D3E", i, "E", k, 0)], 64)
                        ft += tk.turn(S.NEUTRAL_PROMPTS[(k + 1) % 24]) + y[0]; k += 1
                base = ft + tail; ch = D3.d3_chunks(env, it)
                M = D3.traj(env, base, ch, list(range(c.d3_seeds)), lambda s, n: ctx.key("r1bD3", i, s, n), c.d3_n, min(c.B_mid, c.d3_seeds))
                u = uab[18] / uab[18].norm()
                xs = [np.mean([s["l"][n] for n in (6, 7, 8)]) for s in M]
                yn = np.array([[float(s["h"][18][n - 1].double() @ u) for n in (6, 7, 8)] for s in M])
                xn = np.array([[s["l"][n] for n in (6, 7, 8)] for s in M])
                sp, _ = split_rule(xs, yn.mean(1), xn, yn)
                return dict(split=sp, n_tokens=len(base))
            res[(arm, i)] = ctx.cached(f"r2_{arm}_i{i}", run)
    p = {}
    for arm in ("L", "T", "E", "official"):
        b = sum(res[(arm, i)]["split"] and not L0s.get(str(i), {}).get("split", False) for i in items)
        c_ = sum((not res[(arm, i)]["split"]) and L0s.get(str(i), {}).get("split", False) for i in items)
        p[arm] = dict(S=sum(res[(arm, i)]["split"] for i in items), b=b, c=c_, p=mcnemar_p(b, c_))
    if p["L"]["p"] <= 0.05 and p["T"]["p"] > 0.05 and p["E"]["p"] > 0.05:
        v = "length per se"
    elif p["T"]["p"] <= 0.05 and p["L"]["p"] > 0.05:
        v = "turns drive the split"
    elif p["E"]["p"] <= 0.05 and p["L"]["p"] > 0.05:
        v = "evidence drives the split"
    else:
        v = "undecided"
    ctx.write_json("res/r2.json", dict(verdict=K.labelled(v, "reserve R2 (tier 2)", "does not enter D.17", "official-template arm descriptive"),
                                       arms=p, L=L), config=env.config(L=L), labels=["MOCK (tiny random model)"] if ctx.mock else [])


def res_r3(ctx):
    """R3 (D1-r at L*, items 0-23): character-level value encoding, interleaved order 1, k = 0..5, 2 swaps, 12 questions, at r1b_D_A's
    in-session L* (filler shortened by 300 tokens for the longer blocks); 'length effect survives retokenisation' iff the lower bound of
    Delta Phi_char = Phi_char(L*) - Phi_char(L0) > 0 (L0 from r1b_D_A's D1-r records)."""
    import r1b_d_d1 as D1M
    from r1b_d_ana import D1, curve, phi, weights, item_groups, S_, lo
    torch = K._torch(); ls = K.load_json(ctx.input("r1b_D_A", "d1/lstar.json")); L = ls["L_star"]
    if L is None or ls.get("mode") == "FB-2":
        raise K.NotRun("no gated long length (or FB-2)")
    env = D1M.Env(ctx); env.extra = 300; c = env.c; items = list(range(min(24, c.n_sym))); sp = []
    for i in items:
        for sw in (0, 1):
            for k in range(6):
                s_ = D1M.pure("r", "sym", i, c, sw, False, "char") if k == 0 else dict(D1M.mixed(ctx, "r", "sym", i, c, sw, k, 1, qs=D1M.q12()), enc="char")
                s_["enc"] = "char"; sp.append(s_)
    recs = D1M.run_specs(env, sp, L, ls["path"] if ls["path"] in ("crop", "copy") else "crop", "r3")
    r0 = torch.load(ctx.input("r1b_D_A", "d1/raw_L0.pt"), weights_only=False)["records"]
    D = D1(recs + r0); W = weights(item_groups(items, c.n_streams), 11)
    def P(LL):
        l0 = np.nanmean(np.stack([D.A(items, arm="r", swap=s, arr="pure", L=LL, enc="char") for s in (0, 1)]), (0, 2))
        lk = {k: np.nanmean(np.stack([D.A(items, arm="r", swap=s, k=k, order=1, L=LL, enc="char") for s in (0, 1)]), (0, 2)) for k in range(1, 5)}
        return phi(W, l0, lk, range(1, 5))[0]
    d = P(L) - P(0)
    v = "length effect survives retokenisation" if lo(d) > 0 else "length effect not shown under retokenisation"
    ctx.save_pt("res/r3_raw.pt", dict(records=recs), config=env.config(L=L))
    ctx.write_json("res/r3.json", dict(verdict=K.labelled(v, "reserve R3 (tier 2)", "does not enter D.17"), dPhi_char=S_(d), L=L),
                   config=env.config(L=L), seeds=dict(bootstrap=11), labels=["MOCK (tiny random model)"] if ctx.mock else [])


def res_summary(ctx):
    out = {}
    for r in ("r1", "r2", "r3"):
        p = ctx.path(f"res/{r}.json")
        out[r.upper()] = K.load_json(p).get("verdict") if os.path.exists(p) else K.coded(
            "NA:conditional_not_triggered" if r == "r2" else "NA:not_run:step not run")
    ctx.write_json("d_reserves.json", dict(reserves=out, note="tier 2; none enters D.17; no main-plan path into the round summary yet "
                                                              "(open issue: an r1b_ana2 step after the tier-2 sessions)"), config=dict())
