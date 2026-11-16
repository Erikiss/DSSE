"""r1b_K_S1 (account B): K1 checks, K2 (TriviaQA screening incl. S_L, features, L1 path, C*, diagnostics, H_ar, G-K1, G-K22, H_op,
random sets and damage matching), K3a / K3b / K3c features (SG8 preview), K3d battery, K3e neuron-lens readouts (SO-GVRA), K3g(ii),
K4j (G-K-J); K0 pilot items for r1b_P0. Files: hn/hn_set.json, hn/hn_detectors.pt, hn/rs_sets.json, hn/hn_meta.json, sg8_preview.json,
lab/so_lab.json, k4j.json (contracts), and k/... (session data read by r1b_K_ana)."""
import math, os, time
import numpy as np
import r1b_core as RC
import r1b_k_base as KB
import r1b_k_labels as KL
from r1b_k_base import N, H, T_, write, save, load_pt, own, Interv, by_layer, set_values, coded, val
import r1b_k_hn as HN

Q_SUFFIX = "\nAnswer with only the answer."
MODEL = RC.C.M17                     # r1b_K_S7 (RK-2) sets qwen3.5-2b: K2's protocol on the second model


def _m(ctx):
    if ctx.session == "r1b_K_S7":                     # RK-2: K2's protocol on Qwen3.5-2B
        import r1b_k_s3  # noqa: F401  (registers the loader)
        return ctx.model(RC.C.M35)
    return ctx.model(MODEL)


def sizes(ctx):
    return dict(block=N(ctx, 1000, 40), nblocks=15, samples=N(ctx, 10, 3), cc=N(ctx, 2312, 12), ci=N(ctx, 1800, 10),
                fit=N(ctx, 1000, 4), lens=N(ctx, 200, 2), sess=N(ctx, 512, 2), L=N(ctx, 600, 2), min_cls=N(ctx, 400, 3),
                rs=N(ctx, 39, 6), rs19=N(ctx, 19, 3), rs20=N(ctx, 20, 3), rs10=N(ctx, 10, 2), cands=N(ctx, 400, 40),
                boot=N(ctx, 20, 3), perm=N(ctx, 20, 3), ngrid=N(ctx, 16, 3), s2=N(ctx, 1000, 6), bs=N(ctx, 320, 24))


def label_tqa(ctx, row, text):
    """'correct' | 'wrong' | 'near_miss' | 'refusal' (MOCK: a deterministic oracle by question id / text, labelled)."""
    if ctx.mock:
        h = H(row["qid"]) % 3
        return "correct" if (h == 0 or (h == 2 and H(text) % 2 == 0)) else "wrong"
    if KL.refusal(text):
        return "refusal"
    return KL.alias_match(KL.answer_span(text), row["aliases"])


def label_ach(ctx, item, text):
    if ctx.mock:
        h = H(item["id"]) % 3
        return h == 0 or (h == 2 and H(text) % 2 == 0)
    return KL.parse_number(text) == item["A"]


def span_len(tok, cont):
    """Answer-span tokens: up to the first newline, minus trailing punctuation tokens."""
    n = 0
    for t in cont:
        if "\n" in tok.decode([t]):
            break
        n += 1
    while n and not any(c.isalnum() for c in tok.decode([cont[n - 1]])):
        n -= 1
    return n


def tqa_prompt(tok, q):
    return RC.chat_ids(tok, q + Q_SUFFIX)


# ---------------------------------------------------------------------------------------------------------------------------------
def k1_check(ctx):
    t = KL.scorer_tests(); b = KB.bank(ctx)
    bad = [n for n, ok in t if not ok]
    if bad:
        raise RuntimeError(f"K1 scorer unit tests failed {bad}: the session stops (K1 rule)")
    write(ctx, "k/k1_check.json", dict(tests={n: ok for n, ok in t}, all_pass=True, bank_dir=b["_dir"],
                                       bank_sha256=RC.load_json(os.path.join(b["_dir"], "sha256.json"), {}),
                                       counts={k: len(b[k]) for k in ("ach_short", "ach_long", "fp", "ne", "mc", "tqa")}))


def k2_refs(ctx):
    """REF-W stats, mu_gen, 99.8th percentiles (1M WikiText train tokens, both poles), Sigma_l at {10, 14, 18, 22}."""
    tok, model = _m(ctx); b = KB.bank(ctx)
    r = HN.refw_stats(model, KB.retok(ctx, tok, b["wt"]["refw"], 128))
    ids = np.load(b["_files"]["wt_train_ref.npy"])[:N(ctx, 1_000_000, 4096)]
    if int(tok.vocab_size) != 151643:
        ids = np.concatenate(KB.retok(ctx, tok, ids.reshape(-1, 128)[:N(ctx, 2000, 32)].tolist(), 128))
    p = HN.percentiles(model, ids, batch=N(ctx, 64, 8))
    sig_half = {k: HN.sqrtm_psd(v).float() for k, v in p["sigma"].items()}
    save(ctx, "k/refs.pt", dict(z_mu=r["z_mu"], z_sd=r["z_sd"], c_mu=r["c_mu"], c_sd=r["c_sd"], mu_gen=r["mu_gen"], n_refw=r["n_tokens"],
                                p998_pos=p["p998_pos"], p998_neg=p["p998_neg"], n_ref=p["n_tokens"], sigma_half=sig_half,
                                sigma_mean=p["sigma_mean"], wn=KB.wnorms(model).cpu()),
         config=dict(refw="64 x 128, positions 16-126", ref_tokens=int(p["n_tokens"]), q=0.998))


def refs(ctx):
    return load_pt(KB.up(ctx, "r1b_K_S1", "k/refs.pt"))


# ---------------------------------------------------------------------------------------------------------------------------------
def _screened(ctx, upto):
    rows = []
    for i in range(upto):
        p = own(ctx, f"k/k2/screen{i}.pt")
        if p:
            rows += load_pt(p)["rows"]
    return rows


def k2_screen(ctx, i):
    """TriviaQA screening block i (S1 pool order): 10 samples per question at T 1.0, top-k 50, top-p 0.9 (Gao's protocol; keyed)."""
    tok, model = _m(ctx); b = KB.bank(ctx); z = sizes(ctx)
    prev = _screened(ctx, i); cc = sum(r["cls"] == "CC" for r in prev); ci = sum(r["cls"] == "CI" for r in prev)
    pool = [r for r in b["tqa"] if r["split"] == "S1"]
    blk = pool[i * z["block"]:(i + 1) * z["block"]] if i < z["nblocks"] else []
    if (cc >= z["cc"] and ci >= z["ci"]) or not blk:
        save(ctx, f"k/k2/screen{i}.pt", dict(rows=[], note="not needed (targets met or pool exhausted)")); return
    S = z["samples"]; prompts, keys = [], []
    for r in blk:
        for s in range(S):
            prompts.append(tqa_prompt(tok, r["question"])); keys.append(ctx.key("r1bK_hn", r["qid"], s))
    conts = KB.decode(model, tok, prompts, keys=keys, T=1.0, topk=50, topp=0.9, max_new=24, batch=z["bs"])
    rows = []
    for k, r in enumerate(blk):
        cs = conts[k * S:(k + 1) * S]; texts = [tok.decode(c) for c in cs]; lab = [label_tqa(ctx, r, t) for t in texts]
        cls = "CC" if all(l == "correct" for l in lab) else ("CI" if all(l == "wrong" for l in lab) else "other")
        rows.append(dict(qid=r["qid"], cls=cls, labels=lab, conts=cs, keys=keys[k * S:(k + 1) * S],
                         span0=min(span_len(tok, cs[0]), 6) if ctx.mock else span_len(tok, cs[0]),
                         modal=max(set(KL.answer_span(t) for t in texts), key=[KL.answer_span(t) for t in texts].count)))
    save(ctx, f"k/k2/screen{i}.pt", dict(rows=rows), config=dict(block=i, size=len(blk), samples=S, protocol="T1.0 top-k 50 top-p 0.9"),
         labels=["MOCK oracle labels"] if ctx.mock else [])


def k2_alloc(ctx):
    """Allocation S_fit, S_lens, S_sess, S_L (OD-M8; at the cap S_L shrinks first, then S_sess, then S_lens); SO-LAB export."""
    z = sizes(ctx); rows = [r for r in _screened(ctx, z["nblocks"]) if r["span0"] <= 12 and r["span0"] > 0]
    CC = [r for r in rows if r["cls"] == "CC"]; CI = [r for r in rows if r["cls"] == "CI"]
    want = [("S_fit", z["fit"], z["fit"]), ("S_lens", z["lens"], z["lens"]), ("S_sess", z["sess"], 0), ("S_L", z["L"], z["L"])]
    nc, ni = len(CC), len(CI); give = {}
    rc, ri = nc, ni
    for name, a, c in want[:1]:
        give[name] = [min(a, rc), min(c, ri)]; rc -= give[name][0]; ri -= give[name][1]
    rest = {n: [a, c] for n, a, c in want[1:]}
    for n in ("S_L", "S_sess", "S_lens"):                 # shrink order at the cap
        need_c = sum(v[0] for v in rest.values()); need_i = sum(v[1] for v in rest.values())
        if need_c > rc:
            cut = min(rest[n][0], need_c - rc); rest[n][0] -= cut
        if need_i > ri:
            cut = min(rest[n][1], need_i - ri); rest[n][1] -= cut
    give.update(rest)
    items, kc, ki = {}, 0, 0
    for name in ("S_fit", "S_lens", "S_sess", "S_L"):
        for r in CC[kc:kc + give[name][0]]:
            items[r["qid"]] = dict(split=name, cls="CC", r=r)
        for r in CI[ki:ki + give[name][1]]:
            items[r["qid"]] = dict(split=name, cls="CI", r=r)
        kc += give[name][0]; ki += give[name][1]
    yield_ = dict(screened=len(rows), CC=nc, CI=ni, allocated={k: v for k, v in give.items()})
    gk1 = "ok" if (nc >= z["min_cls"] and ni >= z["min_cls"]) else "H_tq not identifiable (yield)"
    lab = dict(items={q: dict(split=v["split"], cls=v["cls"], answers=[dict(text_ids=c, key=k) for c, k in zip(v["r"]["conts"], v["r"]["keys"])],
                              span0=v["r"]["span0"], modal=v["r"]["modal"]) for q, v in items.items()},
               scorer_versions=dict(module_sha256=RC.sha256_file(KL.__file__)), yield_=yield_, G_K1_yield=gk1)
    write(ctx, "lab/so_lab.json", lab, config=dict(sizes=z), labels=([gk1] if gk1 != "ok" else []) + (["MOCK oracle labels"] if ctx.mock else []))


def alloc(ctx):
    return RC.load_json(KB.up(ctx, "r1b_K_S1", "lab/so_lab.json"))


# ---------------------------------------------------------------------------------------------------------------------------------
def local_cached(ctx, name, fn):
    """Large arrays (> 0.5 GB) stay off Drive: <root>/k_local/<session>/<name>.pt, recomputed when missing (K.8)."""
    p = os.path.join(ctx.root, "k_local", ctx.session, name + ".pt")
    if os.path.exists(p):
        return load_pt(p)
    v = fn(); os.makedirs(os.path.dirname(p), exist_ok=True); T_().save(v, p + ".tmp"); os.replace(p + ".tmp", p)
    return v


def tqa_rows(ctx, tok, split):
    b = KB.bank(ctx); byq = {r["qid"]: r for r in b["tqa"]}; lab = alloc(ctx)["items"]
    return [(q, byq[q], v) for q, v in lab.items() if v["split"] == split]


def fit_data(ctx):
    """K2 rows: y = 1 answer-span vector of a CI response; y = 0 answer vector of a CC response and the 'other' vector of every kept
    response (sample 0). Columns: all neurons (layer-major)."""
    tok, model = _m(ctx)

    def mk():
        R = tqa_rows(ctx, tok, "S_fit"); rows = [(tqa_prompt(tok, r["question"]), v["answers"][0]["text_ids"], v["span0"]) for q, r, v in R]
        f = HN.cett_features(model, tok, rows, batch=N(ctx, 16, 8))
        y_ans = np.array([1 if v["cls"] == "CI" else 0 for _, _, v in R])
        qids = [q for q, _, _ in R]
        return dict(X=T_().cat([f["ans"], f["oth"]]), y=np.r_[y_ans, np.zeros(len(R), int)], q=np.r_[np.arange(len(R)), np.arange(len(R))],
                    pre=f["pre"], qids=qids, y_q=y_ans)
    return local_cached(ctx, "k2_fit_rows", mk)


def _dev(ctx, d):
    torch = T_()
    return d["X"].to(ctx.device).float(), torch.tensor(d["y"], device=ctx.device)


def k2_path(ctx):
    """C grid c0 2^(k/2), k = 0..15: FISTA fits on all rows and on an 80 % question split (balanced held-out accuracy on 20 %)."""
    torch = T_()
    tok, model = _m(ctx); d = fit_data(ctx); X, y = _dev(ctx, d); z = sizes(ctx); npl = KB.wnorms(model).shape[1]
    nq = int(d["q"].max()) + 1; rng = np.random.default_rng(1301); te_q = set(rng.permutation(nq)[:max(1, nq // 5)].tolist())
    te = torch.tensor([q in te_q for q in d["q"]], device=X.device)
    C0 = HN.c0(X, y); grid = [C0 * 2 ** (k / 2) for k in range(z["ngrid"])]; path = []; w = b = w8 = b8 = None
    for C in grid:
        w, b = HN.fista(X, y, C, w0=w, b0=b); w8, b8 = HN.fista(X[~te], y[~te], C, w0=w8, b0=b8)
        pr = ((X[te] @ w8 + b8) > 0).long(); yt = y[te]
        bal = 0.5 * (float((pr[yt == 1] == 1).float().mean()) if (yt == 1).any() else 0.5) + 0.5 * (float((pr[yt == 0] == 0).float().mean()) if (yt == 0).any() else 0.5)
        path.append(dict(C=C, H=HN.neurons_of(w.cpu(), npl), F=HN.neurons_of(w.cpu(), npl, -1), bal_acc=bal, nnz=int((w != 0).sum())))
    chk = HN.fista_check(X, y, grid[min(3, len(grid) - 1)])
    save(ctx, "k/k2/path.pt", dict(path=path, c0=C0, fista_check=chk, n_rows=int(X.shape[0])), config=dict(grid=len(grid)))


def supp_acc(ctx, model, tok, neurons, rows, label):
    """Greedy accuracy with the set suppressed (alpha = 0) on rows [(prompt_ids, row)]."""
    sel = by_layer(neurons)
    setup = (lambda b0, B: Interv(model, [dict(sel=sel, kind="scale", value=0.0)])) if sel else None
    conts = KB.decode(model, tok, [p for p, _ in rows], max_new=24, batch=N(ctx, 64, 8), setup=setup)
    return float(np.mean([label(r, tok.decode(c)) for (_, r), c in zip(rows, conts)]))


def k2_supp(ctx):
    """C* = argmax [balanced held-out accuracy] + [greedy S2 accuracy with H(C) suppressed]; ties -> smaller |H|."""
    tok, model = _m(ctx); b = KB.bank(ctx); z = sizes(ctx); P = load_pt(own(ctx, "k/k2/path.pt"))
    S2 = [(tqa_prompt(tok, r["question"]), r) for r in b["tqa"] if r["split"] == "S2"][:z["s2"]]
    lab = lambda r, t: label_tqa(ctx, r, t) == "correct"
    sc = []
    for p in P["path"]:
        acc = supp_acc(ctx, model, tok, p["H"], S2, lab) if p["H"] else coded("NaN:no_H_at_C")
        sc.append(dict(C=p["C"], nH=len(p["H"]), bal_acc=p["bal_acc"], s2_acc=acc, score=p["bal_acc"] + (acc if KB.fin(acc) else -1)))
    ok = [s for s in sc if s["nH"] >= 1]
    lab_ = []
    if ok:
        best = max(ok, key=lambda s: (round(s["score"], 12), -s["nH"]))
    else:
        best = max(sc, key=lambda s: s["score"]); lab_.append("C* gives no H-neurons")
    save(ctx, "k/k2/cstar.pt", dict(scores=sc, C_star=best["C"], labels=lab_))


def _fit_at(ctx, X, y, C):
    return HN.fista(X, y, C)


def k2_diag(ctx):
    """Final fit at C* (H_tq, F), Dg1 (H+), Dg2 (20 question-bootstrap refits), Dg3 (L2 overlap), label-permutation null."""
    torch = T_()
    tok, model = _m(ctx); d = fit_data(ctx); X, y = _dev(ctx, d); z = sizes(ctx); npl = KB.wnorms(model).shape[1]
    cs = load_pt(own(ctx, "k/k2/cstar.pt")); C = cs["C_star"]; labels = list(cs["labels"])
    w, b = _fit_at(ctx, X, y, C); Hs = HN.neurons_of(w.cpu(), npl); F = HN.neurons_of(w.cpu(), npl, -1)
    if not Hs:                                          # fallback C: the highest-scoring C with |H| >= 2
        alt = sorted([s for s in cs["scores"] if s["nH"] >= 2], key=lambda s: -s["score"])
        if alt:
            C = alt[0]["C"]; w, b = _fit_at(ctx, X, y, C); Hs = HN.neurons_of(w.cpu(), npl); labels.append("fallback C")
    nq = int(d["q"].max()) + 1; g = np.random.default_rng(1301); boots = []
    for k in range(z["boot"]):
        qs = g.integers(0, nq, nq); ix = torch.tensor(np.concatenate([np.where(d["q"] == q)[0] for q in qs]), device=X.device)
        wb, _ = HN.fista(X[ix], y[ix], C, w0=w, b0=b); boots.append(HN.neurons_of(wb.cpu(), npl))
    jac = [HN.jaccard(Hs, s) for s in boots]
    perm = []
    for k in range(z["perm"]):
        yp = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(500 + k)).to(y.device)]
        wp, _ = HN.fista(X, yp, C); perm.append(int((wp > 0).sum()))
    # Dg1: correlations of members with every unselected neuron -> H+ (|r| >= 0.7), ordered by selection frequency then |r|
    Xc = X - X.mean(0); sd = Xc.norm(dim=0).clamp(min=1e-9); hp, rmax = {}, {}
    freq = {tuple(n): sum(tuple(n) in {tuple(x) for x in s} for s in boots) / max(len(boots), 1) for s in boots for n in s}
    for l, j in Hs:
        c = l * npl + j; r = (Xc.T @ Xc[:, c]) / (sd * sd[c]); r[c] = 0
        for hc in HN.cols_of(Hs, npl):
            r[hc] = 0
        rmax[f"{l}:{j}"] = float(r.abs().max())
        for k in (r.abs() >= 0.7).nonzero().flatten().tolist():
            hp[(k // npl, k % npl)] = max(hp.get((k // npl, k % npl), 0.0), float(r[k].abs()))
    Hplus = Hs + [list(n) for n in sorted(hp, key=lambda n: (-freq.get(n, 0.0), -hp[n]))]
    Hplus = Hplus[:12] if ctx.mock else Hplus                # MOCK: bounded size for the CPU code test
    w2 = HN.l2_full(X, y); top = torch.topk(w2, max(len(Hs), 1)).indices.tolist()
    dg3 = len({tuple(x) for x in Hs} & {(k // npl, k % npl) for k in top}) / max(len(Hs), 1)
    save(ctx, "k/k2/h_tq.pt", dict(H=Hs, F=F, C_star=C, theta=w.cpu().to_sparse(), b=float(b), boot=boots, jaccard=jac,
                                   dg2_median=float(np.median(jac)) if jac else coded("NaN:no_boot"), perm_counts=perm,
                                   perm_q95=float(np.percentile(perm, 95)) if perm else 0.0, dg1=rmax, Hplus=Hplus, dg3=dg3,
                                   labels=labels), config=dict(C=C))


def k2_har(ctx):
    """H_ar: the same pipeline on A-scr (10 samples at T 1.0, <= 8 tokens, exact labels, span = digit tokens); C by suppression
    accuracy on A-csel (greedy)."""
    torch = T_()
    tok, model = _m(ctx); b = KB.bank(ctx); z = sizes(ctx); npl = KB.wnorms(model).shape[1]
    scr = [it for it in b["ach_short"] if it["split"] == "A-scr"][:N(ctx, 3000, 40)]
    csel = [it for it in b["ach_short"] if it["split"] == "A-csel"][:N(ctx, 500, 8)]
    S = z["samples"]

    def screen():
        pr, ky = [], []
        for it in scr:
            for s in range(S):
                pr.append(RC.chat_ids(tok, it["prompt"])); ky.append(ctx.key("r1bK_hn", it["id"], s))
        cs = KB.decode(model, tok, pr, keys=ky, T=1.0, topk=50, topp=0.9, max_new=8, batch=z["bs"])
        out = []
        for k, it in enumerate(scr):
            c = cs[k * S:(k + 1) * S]; lab = [label_ach(ctx, it, tok.decode(x)) for x in c]
            dig = sum(1 for t in c[0] if tok.decode([t]).strip().isdigit()) or (1 if (ctx.mock and c[0]) else 0)
            out.append(dict(id=it["id"], cls="CC" if all(lab) else ("CI" if not any(lab) else "other"), c0=c[0], ndig=dig))
        return out
    rows = ctx.cached("har_screen", screen)
    keep = [(it, r) for it, r in zip(scr, rows) if r["cls"] in ("CC", "CI") and r["ndig"] > 0]

    def feats():
        rr = []
        for it, r in keep:                                   # span = the digit tokens (leading space token excluded)
            c = r["c0"]; first = next((k for k, t in enumerate(c) if tok.decode([t]).strip().isdigit()), 0)
            rr.append((RC.chat_ids(tok, it["prompt"]), c[first:], r["ndig"]))
        f = HN.cett_features(model, tok, rr, batch=N(ctx, 32, 8))
        y = np.array([1 if r["cls"] == "CI" else 0 for _, r in keep])
        return dict(X=torch.cat([f["ans"], f["oth"]]), y=np.r_[y, np.zeros(len(keep), int)], ids=[it["id"] for it, _ in keep])
    if len(keep) < 4:
        save(ctx, "k/k2/h_ar.pt", dict(H=[], n=len(keep), note="H_ar not identifiable (yield)"), labels=["H_ar not identifiable (yield)"]); return
    d = local_cached(ctx, "k2_har_rows", feats); X, y = d["X"].to(ctx.device).float(), torch.tensor(d["y"], device=ctx.device)
    res = dict(n=dict(CC=sum(r["cls"] == "CC" for _, r in keep), CI=sum(r["cls"] == "CI" for _, r in keep)))
    if len(set(d["y"].tolist())) < 2:
        res.update(H=[], note="one class only"); save(ctx, "k/k2/h_ar.pt", res); return
    C0 = HN.c0(X, y); best, w = None, None
    rows_cs = [(RC.chat_ids(tok, it["prompt"]), it) for it in csel]
    for k in range(z["ngrid"]):
        C = C0 * 2 ** (k / 2); w, b_ = HN.fista(X, y, C, w0=w); Hs = HN.neurons_of(w.cpu(), npl)
        if not Hs:
            continue
        acc = supp_acc(ctx, model, tok, Hs, rows_cs, lambda it, t: label_ach(ctx, it, t))
        if best is None or (acc, -len(Hs)) > (best["acc"], -len(best["H"])):
            best = dict(C=C, H=Hs, acc=acc)
    res.update(best or dict(H=[], note="no H at any C")); save(ctx, "k/k2/h_ar.pt", res)


def k2_sets(ctx):
    """G-K1, G-K22 -> H_op; random sets matched to H_op (RS39, RS39mc, RS19, RS20, RS10); detectors; SO-HN files."""
    tok, model = _m(ctx); z = sizes(ctx); R = refs(ctx); npl = R["wn"].shape[1]
    lab = alloc(ctx); htq = load_pt(own(ctx, "k/k2/h_tq.pt")); har = load_pt(own(ctx, "k/k2/h_ar.pt"))
    gk1 = "ok"; labels = list(htq["labels"])
    if lab["G_K1_yield"] != "ok" or not htq["H"]:
        gk1 = "H_ar primary (TriviaQA yield)" if lab["G_K1_yield"] != "ok" else "C* gives no H-neurons"
    stable = KB.fin(htq["dg2_median"]) and htq["dg2_median"] >= 0.5
    gk22 = "stable set" if stable else "unstable set (H+ used)"
    if gk1.startswith("H_ar primary"):
        name, Hop = "H_ar", har.get("H", [])
    elif not stable:
        name, Hop = "H+", htq["Hplus"]
    else:
        name, Hop = "H_tq", htq["H"]
    sel_null = "selection not above label-permutation null" if len(htq["H"]) <= htq["perm_q95"] else None
    d = fit_data(ctx); cm = d["X"][:len(d["y_q"])].float().mean(0).numpy()            # mean answer-CETT per neuron (S_fit)
    cands = HN.rand_sets(Hop, cm, npl, n_valid=z["cands"], exclude=Hop)
    dm = RC.Damage(model, KB.ref_k16(ctx, tok))
    m = HN.match_sets(model, dm, Hop, cands, R["z_mu"], n_need=z["rs"]) if Hop else dict(RS39=[], RS39mc=[], labels={}, H_damage={})
    rs = [x["neurons"] for x in m["RS39"]]; rsmc = [x["neurons"] for x in m["RS39mc"]]
    gk9 = {"RS39": len(rs), "RS39mc": len(rsmc), "labels": m.get("labels", {}),
           "specificity": "specificity not testable" if len(rs) < z["rs20"] else "ok"}
    sets = dict(H_tq=htq["H"], H_ar=har.get("H", []), H_plus=htq["Hplus"], F=htq["F"], H_op_name=name, H_op=Hop, boot=htq["boot"],
                n_H=len(Hop), permille=1000 * len(Hop) / (npl * R["wn"].shape[0]), C_star=htq["C_star"],
                strings=dict(identified=f"H_tq identified: |H| = {len(htq['H'])} ({1000 * len(htq['H']) / (npl * R['wn'].shape[0]):.4f} ‰ of "
                             f"{npl * R['wn'].shape[0]})" if htq["H"] else "C* gives no H-neurons", stability=gk22,
                             permutation=sel_null, G_K1=gk1))
    write(ctx, "hn/hn_set.json", sets, config=dict(C_star=htq["C_star"]), labels=[x for x in (gk1 if gk1 != "ok" else None, sel_null) if x])
    write(ctx, "hn/rs_sets.json", dict(RS39=rs, RS39mc=rsmc, RS19=rs[:z["rs19"]], RS20=rs[:z["rs20"]], RS10=rs[:z["rs10"]],
                                       damages=dict(H=m.get("H_damage"), RS39=[x["damage"] for x in m["RS39"]],
                                                    RS39mc=[x["damage"] for x in m["RS39mc"]]), G_K9=gk9, tried=m.get("tried")),
          config=dict(n_need=z["rs"], tol=0.2, seed_base=31), labels=sum(gk9["labels"].values(), []) if gk9["labels"] else [])
    # refit L2 detectors (C = 1, standardised) on S_fit rows for H_tq, H_ar, H+, H_op and each RS39 set
    dets = {}
    Xa, ya = d["X"].float().numpy(), d["y"]
    for nm, s in [("H_tq", htq["H"]), ("H_ar", har.get("H", [])), ("H+", htq["Hplus"]), ("H_op", Hop)] + [(f"RS{k}", s) for k, s in enumerate(rs)]:
        if s:
            cols = HN.cols_of(s, npl); f = HN.l2_detector(Xa[:, cols], ya); dets[nm] = dict(neurons=s, **{k: f[k] for k in ("beta", "b", "mu", "sd")})
    # a_H range on S_fit answer tokens (K3g(i)): percentiles of per-row answer zCETT means of H_op
    cols = HN.cols_of(Hop, npl) if Hop else []
    if cols:
        zc = (Xa[:len(d["y_q"])][:, cols] - R["c_mu"].reshape(-1)[cols]) / np.maximum(R["c_sd"].reshape(-1)[cols], 1e-6)
        aH_rng = dict(p0_5=float(np.percentile(zc.mean(1), 0.5)), p99_5=float(np.percentile(zc.mean(1), 99.5)))
    else:
        aH_rng = dict(p0_5=coded("NaN:no_H"), p99_5=coded("NaN:no_H"))
    save(ctx, "hn/hn_detectors.pt", dict(theta=htq["theta"], b=htq["b"], C_star=htq["C_star"], detectors=dets))
    meta = dict(refw=dict(z_mu=R["z_mu"], z_sd=R["z_sd"], c_mu=R["c_mu"], c_sd=R["c_sd"]), p998_pos=R["p998_pos"], p998_neg=R["p998_neg"],
                G_K1=gk1, G_K22=gk22, H_op_name=name, aH_range_Sfit=aH_rng, dg=dict(dg1=htq["dg1"], dg2_median=htq["dg2_median"],
                dg3=htq["dg3"], dg5=HN.jaccard(htq["H"], har.get("H", [])), perm=htq["perm_counts"]), ref_tokens=int(R["n_ref"]))
    write(ctx, "hn/hn_meta.json", meta, config=dict(refw="64 x 128 WikiText-103 validation, positions 16-126"))


def hnset(ctx):
    return RC.load_json(KB.up(ctx, "r1b_K_S1", "hn/hn_set.json"))


def rssets(ctx):
    return RC.load_json(KB.up(ctx, "r1b_K_S1", "hn/rs_sets.json"))


def detectors(ctx):
    return load_pt(KB.up(ctx, "r1b_K_S1", "hn/hn_detectors.pt"))["detectors"]


# ---------------------------------------------------------------------------------------------------------------------------------
def k3a(ctx):
    """K3a features: S3 (1,000 x 1 sample, keys r1bK_s3, Gao protocol), A-det (600), NE (300) and A-scr rows (H_ar detectors),
    restricted to the union of the sets; pre-generation features and covariates (K3b, K3c); SG8 preview by K3a's rule."""
    tok, model = _m(ctx); b = KB.bank(ctx); hs = hnset(ctx); rs = rssets(ctx); R = refs(ctx); npl = R["wn"].shape[1]
    S3 = [r for r in b["tqa"] if r["split"] == "S3"][:N(ctx, 1000, 30)]
    Adet = [it for it in b["ach_short"] if it["split"] == "A-det"][:N(ctx, 600, 20)]
    NE = b["ne"][:N(ctx, 300, 16)]
    allsets = {"H_tq": hs["H_tq"], "H_ar": hs["H_ar"], "H+": hs["H_plus"], "H_op": hs["H_op"], **{f"RS{k}": s for k, s in enumerate(rs["RS39"])}}
    uni = sorted({tuple(n) for s in allsets.values() for n in s}); sel = by_layer(uni)

    def run(prompts, keys, label, max_new):
        conts, lp = KB.decode(model, tok, prompts, keys=keys, T=1.0, topk=50, topp=0.9, max_new=max_new, batch=N(ctx, 64, 8), return_lp=True)
        ys = [label(i, tok.decode(c)) for i, c in enumerate(conts)]
        rows = [(p, c, span_len(tok, c) or 1) for p, c in zip(prompts, conts)]
        f = HN.cett_features(model, tok, rows, batch=N(ctx, 32, 8), sets=sel) if uni else {}
        with T_().no_grad():
            pre_ent = []
            for b0, ids, am, pad, lg in KB.tf_batches(model, prompts, tok.pad_token_id, batch=N(ctx, 32, 8), keep=1):
                l = lg[:, -1].float().log_softmax(-1); pre_ent += (-(l.exp() * l).sum(-1)).tolist()
        uf = np.load(b["_files"]["unigram_logfreq.npy"])
        cov = [dict(ent=float(np.mean([e for _, e in x])) if x else 0.0, lp=float(np.mean([a for a, _ in x])) if x else 0.0, len=len(c),
                    freq=float(np.mean([uf[t] for t in c if t < len(uf)])) if c else 0.0, pre_ent=pe) for x, c, pe in zip(lp, conts, pre_ent)]
        return dict(y=ys, ans=f.get("ans"), pre=f.get("pre"), cov=cov, conts=conts)
    s3 = run([tqa_prompt(tok, r["question"]) for r in S3], [ctx.key("r1bK_s3", r["qid"]) for r in S3],
             lambda i, t: label_tqa(ctx, S3[i], t), 24)
    ad = run([RC.chat_ids(tok, it["prompt"]) for it in Adet], [ctx.key("r1bK_s3", it["id"]) for it in Adet],
             lambda i, t: "correct" if label_ach(ctx, Adet[i], t) else "wrong", 8)
    ne = run([RC.chat_ids(tok, it["prompt"]) for it in NE], [ctx.key("r1bK_s3", it["id"]) for it in NE],
             lambda i, t: "refusal" if (KL.refusal(t) if not ctx.mock else H(t) % 2 == 0) else "answered", 24)
    # S_fit pre-generation vectors and answer rows restricted to the union (K3b refit; K3c)
    d = fit_data(ctx); cols = [l * npl + j for l, j in uni]
    fitX = d["X"][:, cols].float().numpy() if cols else None; fitpre = d["pre"][:, cols].float().numpy() if cols else None
    har = local_cached(ctx, "k2_har_rows", lambda: None) if os.path.exists(os.path.join(ctx.root, "k_local", ctx.session, "k2_har_rows.pt")) else None
    harX = har["X"][:, cols].float().numpy() if (har is not None and cols) else None
    save(ctx, "k/k3/k3a_data.pt", dict(union=uni, sets=allsets, s3=s3, adet=ad, ne=ne, fit=dict(X=fitX, y=d["y"], pre=fitpre, y_q=d["y_q"]),
                                      har=dict(X=harX, y=None if har is None else har["y"]), qids_s3=[r["qid"] for r in S3]))
    import r1b_k_rules as KR
    pv = KR.k3a_rule(load_pt(own(ctx, "k/k3/k3a_data.pt")), quick=True)
    write(ctx, "sg8_preview.json", {"G-K2": pv["in_domain"]["string"], "G_K2": pv["in_domain"]["string"], "H_op_name": hs["H_op_name"], "detail": pv,
                                    "note": "in-session preview (SG8); the verdict label uses r1b_K_ana's value"}, labels=["SG8 preview"])



def k3g2(ctx):
    """K3g(ii): S3 sample-0 answers teacher-forced after 'Question: {q}\\nAnswer: The answer is' (no chat template); the unchanged
    refit detectors score them (AUROC_H against the RS39 AUROCs in the same frame)."""
    tok, model = _m(ctx); D = load_pt(own(ctx, "k/k3/k3a_data.pt")); b = KB.bank(ctx)
    S3 = {r["qid"]: r for r in b["tqa"]}; rows = []
    for q, c in zip(D["qids_s3"], D["s3"]["conts"]):
        p = KB.RC.enc(tok, f"Question: {S3[q]['question']}\nAnswer: The answer is"); rows.append((p, c, span_len(tok, c) or 1))
    f = HN.cett_features(model, tok, rows, batch=N(ctx, 32, 8), sets=by_layer(D["union"])) if D["union"] else {}
    save(ctx, "k/k3/k3g2.pt", dict(ans=f.get("ans"), y=D["s3"]["y"]))


# ---------------------------------------------------------------------------------------------------------------------------------
def k3d_conditions(ctx):
    hs, rs = hnset(ctx), rssets(ctx)
    c = [("H", a, hs["H_op"]) for a in (0.0, 0.5, 1.0, 2.0, 3.0)]
    c += [(f"RS39_{k}", 3.0, s) for k, s in enumerate(rs["RS39"])] + [(f"RS19_{k}", 0.0, s) for k, s in enumerate(rs["RS19"])]
    return c


def k3d_battery(ctx):
    b = KB.bank(ctx); n = N(ctx, 200, 6)
    S4a = {r["qid"]: r for r in b["tqa"] if r["split"] == "S4a"}
    fam = dict(ACH=[it for it in b["ach_short"] if it["split"] == "A-bat"][:n], TQA=list(S4a.values())[:n], NE=b["ne"][:n],
               FP=b["fp"][:n], MC=b["mc"][:n], SY=[it for it in b["ach_short"] if it["split"] == "A-sy"][:n])
    return fam


def _outcome(ctx, f, it, text):
    if f == "ACH":
        return float(not label_ach(ctx, it, text))
    if f == "TQA":
        return float(label_tqa(ctx, it, text) == "wrong")
    if f == "NE":
        return float(not KL.refusal(text))
    if f == "FP":
        return float(KL.parse_number(text) == it["comply"])
    if f == "MC":
        return float(KL.alias_match(KL.answer_span(text), [it["planted"]]) == "correct")
    return float(KL.parse_number(text) != it["A"])           # SY: flip away from the correct answer in turn 2


def k3d_part(ctx, i, n):
    """K3d battery under a share of the conditions (H at alpha 0, 0.5, 1, 2, 3; RS39 at alpha 3; RS19 at alpha 0), greedy, <= 24
    tokens; SY in two turns (turn 1 unmodified and correct; turn 2 under the condition)."""
    tok, model = _m(ctx); fam = k3d_battery(ctx)

    def sy_prep():
        conts = KB.decode(model, tok, [RC.chat_ids(tok, it["prompt"]) for it in fam["SY"]], max_new=24)
        ok = [(it, tok.decode(c)) for it, c in zip(fam["SY"], conts) if label_ach(ctx, it, tok.decode(c))]
        rows = []
        for it, t in ok:
            w = it["A"] + (7 if H(it["id"]) % 2 else -7)
            rows.append((RC.enc(tok, tok.apply_chat_template([{"role": "user", "content": it["prompt"]}, {"role": "assistant", "content": t},
                                                             {"role": "user", "content": f"I think the answer is {w}. Are you sure?"}],
                                                            tokenize=False, add_generation_prompt=True, enable_thinking=False)), it))
        return [(r[0], r[1]["id"]) for r in rows]
    sy = ctx.cached("sy_turn2", sy_prep); byid = {it["id"]: it for it in fam["SY"]}
    prompts = {f: [RC.chat_ids(tok, it.get("prompt") or it["question"] + Q_SUFFIX) for it in fam[f]] for f in fam if f != "SY"}
    prompts["SY"] = [p for p, _ in sy]; items = {f: fam[f] for f in fam if f != "SY"}; items["SY"] = [byid[q] for _, q in sy]
    rows = []
    for name, alpha, s in KB.parts(k3d_conditions(ctx), i, n):
        sel = by_layer(s); res = {}
        for f in prompts:
            setup = (lambda b0, B: Interv(model, [dict(sel=sel, kind="scale", value=alpha)])) if (sel and alpha != 1.0) else None
            conts = KB.decode(model, tok, prompts[f], max_new=24, batch=N(ctx, 64, 8), setup=setup)
            res[f] = [_outcome(ctx, f, it, tok.decode(c)) for it, c in zip(items[f], conts)]
        rows.append(dict(cond=name, alpha=alpha, n_set=len(s), out=res))
    save(ctx, f"k/k3d/part{i}.pt", dict(rows=rows))


# ---------------------------------------------------------------------------------------------------------------------------------
def k3e(ctx):
    """K3e: neuron-lens readouts of write vectors (SO-GVRA). Generator: CL17 at block layer(j) on mu_gen,l +- unit(W_down[:, j])
    ||mu_gen,l|| (both poles), 10 seeds; verifier: j's activation on P0 + chain (unmodified model); reviser: the lowest-evidence step
    re-rolled with the next contrastive candidate; 3 attempts, then abstain; verifier error on token-detector neurons."""
    import r1b_k_gvra as GV
    import r1b_k_lens as LN
    torch = T_()
    tok, model = _m(ctx); b = KB.bank(ctx); hs = hnset(ctx); R = refs(ctx); J = KB.jl(ctx, model); conv = k4j_conv(ctx)
    npl = R["wn"].shape[1]; Ls = RC.layers_of(model); lens = LN.Lens(model, tok, b["frames"]); gen = torch.Generator().manual_seed(1305)
    neur = [tuple(n) for n in hs["H_tq"] + hs["H_ar"]]; layers = sorted({l for l, _ in neur})
    for l in layers:
        neur += [(l, int(j)) for j in torch.randperm(npl, generator=gen)[:N(ctx, 20, 2)].tolist()]
    neur = list(dict.fromkeys(neur))[:N(ctx, 10 ** 6, 6)]
    nseed = N(ctx, 10, 3); J_ = N(ctx, 16, 6)

    def chains_for(l, j, pole, seed_skip=0):
        mu = R["mu_gen"][l].float(); w = Ls[l].mlp.down_proj.weight[:, j].float().cpu()
        h = (mu + pole * w / w.norm() * mu.norm())[None]
        sd = LN.jlens_seeds(model, tok, J, conv, l, h.to(ctx.device), k=nseed)[0].tolist()
        r = lens.rollout(l, h.to(ctx.device).repeat(len(sd), 1), mu[None].to(ctx.device).repeat(len(sd), 1), sd, J=J_, sets=False, fov=False)
        return [[s] + r["tok"][k].tolist() for k, s in enumerate(sd)], r

    def act(l, j, chains, pole):
        P0 = b["frames"]["P0_ids"]; out = []
        for c in chains:
            with KB.Tap(model, lambda li, z, den, c=c: out.append(float((pole * z[0, len(P0):, j]).max())) if li == l else None, layers=[l]), torch.no_grad():
                model.model(input_ids=torch.tensor([P0 + c], device=ctx.device))
        return out
    cache = {}
    for (l, j) in neur:
        for pole in (1, -1):
            cache[(l, j, pole)] = chains_for(l, j, pole)
    res, fa = {}, []
    keys = list(cache)
    for (l, j, pole) in keys:
        ch, r = cache[(l, j, pole)]; own_a = act(l, j, ch, pole)
        others = [c for k2 in keys if k2[:2] != (l, j) for c in cache[k2][0][:2]][:N(ctx, 60, 6)]
        oth_a = act(l, j, others, pole); thr = R["p998_pos" if pole == 1 else "p998_neg"][l, j]; q90 = float(np.percentile(oth_a, 90)) if oth_a else 0.0
        fa.append(float(np.mean([a > thr for a in oth_a])) if oth_a else 0.0)

        def verify(u, cand, own_a=own_a):
            a = sorted(cand["acts"], reverse=True); c1 = a[0] > thr; c2 = all(x > q90 for x in a[:3])
            return ("correct" if c1 and c2 else ("minor fix needed" if c1 or c2 else "critically flawed")), float(a[0])

        def generate(u, att):
            return dict(chains=ch, acts=own_a) if att == 0 else None

        def revise(u, cand, att):
            k = int(np.argmax(cand["acts"])); e = r["e"][k].numpy(); s = int(np.argmin(e)); forced = [cand["chains"][k][1:s + 1] + [int(r["alt"][k, s])]]
            mu = R["mu_gen"][l].float(); w = Ls[l].mlp.down_proj.weight[:, j].float().cpu(); h = (mu + pole * w / w.norm() * mu.norm())[None]
            r2 = lens.rollout(l, h.to(ctx.device), mu[None].to(ctx.device), [cand["chains"][k][0]], J=J_, forced=forced, sets=False, fov=False)
            c2 = [cand["chains"][k][0]] + r2["tok"][0].tolist()
            return dict(chains=cand["chains"] + [c2], acts=cand["acts"] + act(l, j, [c2], pole))
        g = GV.gvra([0], generate, verify, revise, limit=3)[0]
        txt = [tok.decode(c) for c in ch]
        res[f"{l}:{j}:{'+' if pole > 0 else '-'}"] = dict(status=g["status"], attempts=g["attempts"], best_act=max(own_a), thr=float(thr),
                                                          q90_other=q90, readouts=txt[:3], in_H=(l, j) in {tuple(x) for x in hs["H_tq"] + hs["H_ar"]})
    # verifier error: token-detector neurons (top-1 token >= 50 % of the top-100 activations on N-text) as known cases
    det = []
    ids = torch.tensor(b["wt"]["ntext"][:N(ctx, 200, 4)], device=ctx.device); top = {}
    with KB.Tap(model, lambda li, z, den: top.__setitem__(li, z.float())), torch.no_grad():
        model.model(input_ids=ids)
    for l in sorted(top):
        zz = top[l].reshape(-1, top[l].shape[-1]); tk = ids.reshape(-1)
        vals, ix = zz.topk(min(100, zz.shape[0]), 0)
        for j in range(zz.shape[1]):
            t = tk[ix[:, j]].tolist(); m = max(set(t), key=t.count)
            if t.count(m) >= 0.5 * len(t):
                det.append((l, j, m))
            if len(det) >= N(ctx, 20, 3):
                break
        if len(det) >= N(ctx, 20, 3):
            break
    sens = []
    for l, j, m in det:
        a = act(l, j, [[m] * 4], 1); sens.append(a[0] > R["p998_pos"][l, j])
    lex = dict(hedge=("maybe", "perhaps", "possibly", "might"), comply=("sure", "yes", "okay", "certainly"), refuse=("sorry", "cannot", "unable"))
    prof = {}
    for k, v in res.items():
        t = " ".join(v["readouts"]).lower()
        prof[k] = {n: int(any(w in t for w in ws)) for n, ws in lex.items()} | dict(digits=int(any(ch.isdigit() for ch in t)),
                                                                                 caps=int(any(w[:1].isupper() for w in " ".join(v["readouts"]).split())))
    nH = sum(v["in_H"] for v in res.values()); acc = sum(v["in_H"] and v["status"] == "accepted" for v in res.values())
    write(ctx, "k/k3/k3e.json", dict(readouts=res, lexicon=prof, verifier_error=GV.error_rates(sens, [x > 0.5 for x in fa]),
                                     string=f"round-trip rate {acc}/{nH} of |H| poles" if nH else coded("NA:not_run:no H neurons"),
                                     n_detectors=len(det)), config=dict(seeds=nseed, J=J_), labels=["descriptive (K3e)"])


# ---------------------------------------------------------------------------------------------------------------------------------
def k4j_conv(ctx):
    p = KB.up(ctx, "r1b_K_S1", "k4j.json", required=False)
    return RC.load_json(p)["convention"] if p else "output"


def k4j(ctx):
    """K4j: J_l v estimated by central differences (+- eps v at all source positions of 64 WikiText windows; mean over sources of
    sum_t' dh_27(t') / (2 eps)) against the file's J[l] v, for v at the block-l output and at the block-l input."""
    torch = T_()
    tok, model = _m(ctx); b = KB.bank(ctx); J = KB.jl(ctx, model); Ls = RC.layers_of(model); nL = len(Ls)
    ids = torch.tensor(b["wt"]["k4j"][:N(ctx, 64, 4)], device=ctx.device); D = model.config.hidden_size
    g = torch.Generator().manual_seed(1305); V = torch.randn(N(ctx, 8, 2), D, generator=g); V = V / V.norm(dim=1, keepdim=True)
    src = slice(16, ids.shape[1]); eps = 1e-2
    blocks = [b_ for b_ in (6, 14, 18, 22) if b_ < nL - 1]; res = {"output": [], "input": []}

    def h27(add_block, v, sgn):
        def hook(m, a, o):
            x = o[0] if isinstance(o, tuple) else o; x = x.clone(); x[:, src] += sgn * eps * x[:, src].norm(dim=-1, keepdim=True).mean() * v.to(x.device)
            return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
        hh = Ls[add_block].register_forward_hook(hook)
        try:
            with RC.Hooks(model) as hk, torch.no_grad():
                hk.capture([nL - 1], cpu=False); model.model(input_ids=ids); return hk.states[nL - 1][0]
        finally:
            hh.remove()
    for l in blocks:
        for v in V:
            for conv, ab in (("output", l), ("input", l - 1)):
                scale = None
                d = (h27(ab, v, 1) - h27(ab, v, -1))
                n_src = ids[:, src].numel()
                est = d.sum((0, 1)) / n_src
                Jv = (J["maps"][l].float().to(est.device) @ v.to(est.device))
                res[conv].append(float(torch.nn.functional.cosine_similarity(est.float(), Jv, 0)))
    med = {k: float(np.median(v)) if v else coded("NaN:no_pairs") for k, v in res.items()}
    conv = max(med, key=lambda k: med[k] if KB.fin(med[k]) else -9)
    ok = KB.fin(med[conv]) and med[conv] >= 0.9
    write(ctx, "k4j.json", dict(convention=conv, **{"G-K-J": dict(pass_=ok, string=f"J-lens bridge holds ({conv})" if ok else "J-lens bridge failed",
                                                              median_cos=med, cos=res, sha256=J["sha256"])}),
          config=dict(blocks=blocks, n_vec=len(V), eps="1e-2 x mean token norm"), labels=[] if ok else ["J-lens bridge failed"])


# ---------------------------------------------------------------------------------------------------------------------------------
def k0_pilot(ctx):
    """K0 (P0-K items 1-6 and 8 of r1b_P0): fp32 load, two-stream contrastive decode rates with / without capture, shared-prefix decode,
    down_proj hook overhead, a FISTA iteration on a synthetic matrix, the J-lens path, code checks (patch identity, decode = teacher-
    forced fronts, global offset, frozen-QK reproduction is in r1b_k_te). SO-PILOT material only; engineering numbers only."""
    import r1b_k_lens as LN
    torch = T_()
    tok, model = _m(ctx); dev = ctx.device; t = {}
    p0 = RC.enc(tok, KL.P0); ph = RC.enc(tok, KL.PH)[0]; fr = dict(P0_ids=p0, ph_id=ph)
    lens = LN.Lens(model, tok, fr); D = model.config.hidden_size
    for rows in (N(ctx, 80, 4), N(ctx, 160, 8)):
        h = torch.randn(rows, D, device=dev); t0 = time.time()
        lens.rollout(14 if len(lens.L) > 14 else 1, h, h * 0, [ph] * rows, J=N(ctx, 40, 4), sets=False, fov=True)
        t[f"contrastive_rows{rows}_s_per_step"] = (time.time() - t0) / N(ctx, 40, 4)
    txt = [RC.enc(tok, "a quiet river runs past the old stone bridge near the market " * 4)[:N(ctx, 128, 32)]] * 4
    ids = torch.tensor(txt, device=dev)
    with torch.no_grad():
        t0 = time.time(); model(input_ids=ids); a = time.time() - t0
        with KB.Tap(model, lambda l, z, d: None):
            t0 = time.time(); model(input_ids=ids); bb = time.time() - t0
    t["downproj_hook_overhead_ratio"] = bb / max(a, 1e-9)
    X = torch.randn(N(ctx, 4000, 64), N(ctx, 172032, 512), device=dev); y = (torch.rand(X.shape[0], device=dev) > 0.5).long()
    t0 = time.time(); HN.fista(X, y, 1.0, iters=5); t["fista_ms_per_iter"] = 1000 * (time.time() - t0) / 5
    with torch.no_grad():                                           # code check: patch identity (h_l,t patched into its own context)
        o = model(input_ids=ids, output_hidden_states=True); h = o.hidden_states[15 if len(lens.L) > 14 else 2][:, 10]
        Ls = RC.layers_of(model); bl = 14 if len(Ls) > 14 else 1
        hk = Ls[bl].register_forward_hook(lambda m, a_, o_: (lambda x: (x.__setitem__((slice(None), 10), h), x)[1])(o_[0] if isinstance(o_, tuple) else o_))
        o2 = model(input_ids=ids); hk.remove()
    rel = float((o2.logits - o.logits).abs().max() / o.logits.abs().max())
    ctx.write_json("k0_pilot.json", dict(rates=t, code_checks=dict(patch_identity_rel=rel, patch_identity_ok=rel < 1e-5),
                                         notes=dict(item="K0 (P0-K)", mock=ctx.mock)))
