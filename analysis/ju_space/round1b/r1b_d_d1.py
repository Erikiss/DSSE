"""Part D, session r1b_D_A (account B, input-free; D.4-D.12): G-LCE-1 / G-LCE-2 first (SG9, contract file 'glce'), then the D1 surface and
its arms. Steps (r1b_sessions_D.py): d_glce, d1_ends (pure and X_N at all four lengths; items 0-15 main at 4,096), d1_lstar (L*, FB-1,
FB-2 in session, OD-G8), d1_L0 / d1_Ls (every arm at L0 / L*, grouped by filler stream: one prefix prefill per stream and length),
d1_Lf (D1-f, shuffled streams at L*), d1_v (D1-v), d1_cov (SG8 optional covariate), d1_s (D1-s, Qwen3-0.6B, with its own G-LCE-1).

Raw records (d1/raw_<step>.pt with sidecar): one per context with id, arm, fam ('sym' | 'asym'), item, stream, swap, k, order, arr, enc,
L (nominal; 0 = L0), model, filler ('coh' | 'shuf'), q [(attribute, B shown as (a))], l (log p(B letter) - log p(A letter), nats,
unclipped; X_N: version-2 orientation; closed-book: true-value orientation), qs [(attribute, B shown as (a))], M (p(a) + p(b)), optional h14 / h18 (float16, last question
token), optional D1-d extras (nll [2], kl, fact [4]). The analysis (r1b_d_ana.py) clips at +-15 and computes every statistic."""
import os
import numpy as np
import r1b_core as K
import r1b_contracts as C
import r1b_d_stim as S
from r1b_lce import LCE, letters_logodds

CAP = (14, 18)
FACTUAL = [("What is the capital of Australia?", "Canberra", "Sydney"), ("What is the largest planet in the solar system?", "Jupiter", "Saturn"),
           ("At what temperature in degrees Celsius does water boil at sea level?", "100", "90"), ("Who wrote Hamlet?", "Shakespeare", "Marlowe")]
TOL_MAX, TOL_MED = 0.02, 0.002


# ------------------------------------------------------------------------------------------------------------------------------------
# setup, lengths, specs
# ------------------------------------------------------------------------------------------------------------------------------------
class Env:
    def __init__(self, ctx, model_name=C.M17):
        self.ctx, self.c = ctx, S.cfg(ctx)
        self.tok, self.model = ctx.model(model_name)
        self.tk, self.eng, self.name = S.Tk(self.tok), LCE(self.model), model_name
        self.d, self.d_sha = S.load_pkg(ctx, "d1_stimuli.json", self.tok)
        self.st = S.streams(ctx, self.tk)

    def fill(self, L):
        return self.c.L0_fill if L == 0 else L - self.c.nonfill - self.extra

    extra = 0                 # extra non-filler allowance (R3: character-level blocks at L*)

    def prefix(self, stream, L, shuf=False):
        return self.tk.pre + self.tk.enc("Documents:\n\n") + self.st["shuf" if shuf else "coh"][stream][-self.fill(L):]

    def config(self, **kw):
        return dict(model=self.name, d1_stimuli_sha256=self.d_sha, streams_sha256=self.st["sha256"], cfg=dict(self.c), **kw)


def n_items(c, n, trim=None, ctx=None, to=16):
    n = min(n, c.n_sym)
    return min(n, to) if (trim and ctx is not None and ctx.trimmed(trim)) else n


def pattern(ctx, fam, item, swap, k, arr):
    """B slot positions: 'rel' = the last k slots; 'int' = keyed (keyseed('r1bD1', fam, item, swap, k, 'int')), aperiodic."""
    if arr == "rel":
        return list(range(10 - k, 10))
    rng = np.random.default_rng(ctx.key("r1bD1", fam, item, swap, k, arr))
    while True:
        pos = sorted(int(x) for x in rng.choice(10, k, replace=False))
        p = [int(s in pos) for s in range(10)]
        if k in (0, 10) or not any(all(p[s] == p[s + q] for s in range(10 - q)) for q in range(1, 6)):
            return pos


def versions(fam, swap):
    """(A kind, B kind): F-sym swap 0 -> B = version 2; F-asym swap 0 -> B = the true value (prior-aligned version)."""
    return (("A1", "A2") if swap == 0 else ("A2", "A1")) if fam == "sym" else (("F", "T") if swap == 0 else ("T", "F"))


def q12():
    return [(j, b) for j in range(6) for b in (True, False)]


def q6(swap, order):
    return [(j, (j + swap + order) % 2 == 0) for j in range(6)]


def spec(arm, fam, item, c, swap=0, k=0, order=1, arr="int", kinds=None, qs=None, cap=False, enc="can", dext=False, **kw):
    sid = f"{arm}|{fam}|i{item}|s{swap}|k{k}|o{order}|{arr}|{enc}" + "".join(f"|{a}{b}" for a, b in sorted(kw.items()))
    return dict(id=sid, arm=arm, fam=fam, item=item, stream=item % c.n_streams, swap=swap, k=k, order=order, arr=arr, kinds=kinds,
                qs=qs, cap=cap, enc=enc, dext=dext, **kw)


def mixed(ctx, arm, fam, item, c, swap, k, order, arr="int", sub=None, cap=False, qs=None, dext=False):
    """A k-of-10 context (B reports at the keyed positions; order 2 = the exact reverse of order 1's A/B sequence; templates stay at their
    slots); sub replaces the B reports (e.g. 'C': the third value)."""
    A, B = versions(fam, swap); pos = pattern(ctx, fam, item, swap, k, "rel" if arr == "rel" else "int")
    kinds = [(sub or B) if s in pos else A for s in range(10)]
    kinds = kinds[::-1] if order == 2 else kinds
    return spec(arm, fam, item, c, swap, k, order, arr, kinds, qs or (q12() if arr == "rel" or k == 0 else q6(swap, order)), cap,
                dext=dext)


def pure(arm, fam, item, c, swap, cap=True, enc="can", dext=False, qs=None):
    return spec(arm, fam, item, c, swap, 0, 0, "pure", [versions(fam, swap)[0]] * 10, q12() if qs is None else qs, cap, enc, dext)


def xn(arm, item, c, cap=True, dext=False, qs=None):
    return spec(arm, "sym", item, c, 0, 0, 0, "xn", ["N"] * 10, q12() if qs is None else qs, cap, dext=dext)


def ycomp(ctx, arm, item, c, swap, order, kby, dext=False, name="nc"):
    """(k B, m A, z Y) blocks (non-competing (0,5,5) and D1-c's T1 / T2 / T4): B then Y positions keyed by
    keyseed('r1bD1', 'sym', item, swap, name, 'int'); order 2 reverses the sequence."""
    k, m, z = kby; A, B = versions("sym", swap)
    rng = np.random.default_rng(ctx.key("r1bD1", "sym", item, swap, name, "int"))
    pb = set(int(x) for x in rng.choice(10, k, replace=False)) if k else set()
    rest = [s for s in range(10) if s not in pb]; py = set(int(x) for x in rng.choice(rest, z, replace=False)) if z else set()
    kinds = ["Y" if s in py else (B if s in pb else A) for s in range(10)]
    return spec(arm, "sym", item, c, swap, k, order, name, kinds[::-1] if order == 2 else kinds, q6(swap, order), False, dext=dext)


# ------------------------------------------------------------------------------------------------------------------------------------
# running contexts over one shared prefix (SO-LCE crop / copy path; FB-2 full recompute)
# ------------------------------------------------------------------------------------------------------------------------------------
def values(env, sp, j):
    """(A value, B value) strings of attribute j (X_N and closed-book: version-1 / version-2 and cp / true orientation)."""
    it = env.d["fsym" if sp["fam"] == "sym" else "fasym"][sp["item"]]["slots"][j]
    if sp["fam"] == "sym":
        return (it["v1"], it["v2"]) if sp["swap"] == 0 else (it["v2"], it["v1"])
    return (it["cp"], it["true"]) if sp["swap"] == 0 else (it["true"], it["cp"])


def q_ids(env, sp, j, bf):
    a, b = values(env, sp, j); x, y = (b, a) if bf else (a, b)
    return env.tk.enc(S.question_text(env.d, sp["fam"], sp["item"], j, x, y)) + env.tk.post + env.tk.ans


def b_ids(env, sp):
    if sp["kinds"] is None:
        return []
    rng_of = (lambda v: np.random.default_rng(env.ctx.key("r1bD1r", v))) if sp["enc"] == "seg" else None
    return S.parts_ids(env.tk, S.block_parts(env.d, sp["fam"], sp["item"], sp["kinds"]), sp["enc"], rng_of)


def fact_ids(env, qi, correct_first):
    q, x, y = FACTUAL[qi]; a, b = (x, y) if correct_first else (y, x)
    return env.tk.enc(f"\n\nQuestion: {q} (a) {a} (b) {b}\nReply with the letter.") + env.tk.post + env.tk.ans


def read_q(env, cache, ids, cap):
    torch = K._torch()
    if cap:
        with K.Hooks(env.model) as hk:
            hk.capture(list(CAP), positions=[len(ids) - 1])
            cache, lg = env.eng.fwd(ids, cache)
        hs = {b: hk.states[b][0][0, 0].to(torch.float16) for b in CAP}
    else:
        cache, lg = env.eng.fwd(ids, cache); hs = None
    lo, m = letters_logodds(lg[0, -1], env.tk.A, env.tk.B)
    return float(lo), float(m), hs


def d1d_extras(env, cache, n1, ci, xref):
    """D1-d: the two 128-token reference passages scored teacher-forced (NLL per token; KL(p_XN || p_cell) per scored token against the
    item's X_N context at the same length) and the 4 factual forced choices (one letter assignment per context, alternating)."""
    torch = K._torch(); sep = env.tk.enc("\n\n"); nll, lps = [], []
    for P in env.st["ref"]:
        cache, lg = env.eng.fwd(sep + list(P), cache, keep=len(P) + 1)
        lp = torch.log_softmax(lg[0, :-1].double(), -1)
        nll.append(float(-lp[torch.arange(len(P)), torch.as_tensor(P, device=lp.device)].mean())); lps.append(lp.float())
        env.eng.crop(cache, n1)
    lp = torch.cat(lps)
    kl = None if xref is None else float((xref.exp() * (xref - lp)).sum(-1).mean())
    fact = []
    for qi in range(4):
        cf = ci % 2 == 0
        lo, _, _ = read_q(env, cache, fact_ids(env, qi, cf), False)
        fact.append(-lo if cf else lo)
        env.eng.crop(cache, n1)
    return dict(nll=nll, kl=kl, fact=fact, correct_first=ci % 2 == 0), lp


def run_group(env, specs, L, stream, path="crop", shuf=False):
    """One prefix prefill (B = 1) for (stream, L, filler); per context the block chunk (unstopped), then each question chunk; crop path:
    crop back to prefix + block after each question; copy path: a device copy of prefix + block per question; 'full': FB-2 (a fresh
    B = 1 prefill of the whole sequence per question)."""
    pre = env.prefix(stream, L, shuf); out = []
    cache = None if path == "full" else env.eng.fwd(pre)[0]
    n0 = len(pre); xref, dcount = {}, {}
    for sp in specs:
        blk = b_ids(env, sp)
        assert len(pre) + len(blk) + 120 <= min((L or 2000) + 400, env.c.max_pos), f"context length {len(pre) + len(blk)} at L {L}"
        rec = dict(sp)
        rec.update(L=L, model=env.name, filler="shuf" if shuf else "coh", l=[], M=[], n_tokens=len(pre) + len(blk))
        if path != "full" and blk:
            cache, _ = env.eng.fwd(blk, cache)
        n1 = n0 + len(blk)
        hs = {b: [] for b in CAP}
        for j, bf in sp["qs"] or []:
            qi = q_ids(env, sp, j, bf)
            if path == "full":
                c, lg = env.eng.fwd(pre + blk + qi); lo, m = letters_logodds(lg[0, -1], env.tk.A, env.tk.B); lo, m, h = float(lo), float(m), None
            else:
                c = cache if path == "crop" else LCE.copy(cache)
                lo, m, h = read_q(env, c, qi, sp["cap"])
                if path == "crop":
                    env.eng.crop(cache, n1)
                del c
            rec["l"].append(-lo if bf else lo); rec["M"].append(m)
            if h:
                for b in CAP:
                    hs[b].append(h[b].cpu())
        if sp["cap"] and hs[CAP[0]]:
            for b in CAP:
                rec[f"h{b}"] = K._torch().stack(hs[b])
        if sp.get("dext") and path != "full":
            ci = dcount.get(sp["item"], 0); dcount[sp["item"]] = ci + 1
            rec["d"], lp = d1d_extras(env, cache, n1, ci, xref.get(sp["item"]))
            if sp["arr"] == "xn":
                xref[sp["item"]] = lp
        if path != "full" and blk:
            env.eng.crop(cache, n0)
        out.append(rec)
    del cache, xref
    return out


def run_specs(env, specs, L, path, step, shuf=False, margin_min=15.0):
    """Groups specs by stream (one prefix prefill per stream and length); each group is a resumable piece (ctx.cached)."""
    by = {}
    for sp in specs:
        by.setdefault(sp["stream"], []).append(sp)
    out = []
    for s in sorted(by):
        name = f"{step}_L{L}_s{s}{'_shuf' if shuf else ''}_{env.name}"
        if not os.path.exists(env.ctx.path(f"cache/{env.ctx.step}/{name}.pt")) and env.ctx.time_left() < margin_min * 60 and not env.ctx.mock:
            raise K.StepYield()
        out += env.ctx.cached(name, lambda: run_group(env, by[s], L, s, path, shuf))
    return out


def save_raw(env, rel, recs, summary, labels=(), **cfgkw):
    torch = K._torch(); labels = list(labels) + (["MOCK (tiny random model)"] if env.ctx.mock else [])
    env.ctx.save_pt(f"d1/raw_{rel}.pt", dict(records=recs), config=env.config(**cfgkw), labels=labels)
    by = {}
    for r in recs:
        by.setdefault(f"{r['arm']}|L{r['L']}|{r['filler']}", []).extend(r["M"])
    summ = dict(n_records=len(recs), committed_mass={k: K.dist(v) for k, v in by.items()}, engine=env.eng.record(), **summary)
    env.ctx.write_json(f"d1/{rel}.json", summ, config=env.config(**cfgkw), labels=labels)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ------------------------------------------------------------------------------------------------------------------------------------
# steps
# ------------------------------------------------------------------------------------------------------------------------------------
def glce_contexts(env, n):
    return [spec("glce", "sym", i, env.c, 0, (0, 2, 3, 5)[i % 4], 1, "int",
                 mixed(env.ctx, "glce", "sym", i, env.c, 0, (0, 2, 3, 5)[i % 4], 1)["kinds"], [(i % 6, True), (i % 6, False)])
            for i in range(n)]


def parity(env, specs, L):
    """G-LCE-1 at one length: crop path and copy path against a B = 1 full prefill of the same ids (one attribute, both assignments)."""
    res = {p: run_specs(env, specs, L, p, f"glce_{p}") for p in ("crop", "copy", "full")}
    full = np.array([x for r in res["full"] for x in r["l"]]); out = {}
    for p in ("crop", "copy"):
        dl = np.abs(np.array([x for r in res[p] for x in r["l"]]) - full)
        out[p] = dict(max_abs=float(dl.max()), median_abs=float(np.median(dl)), n=int(dl.size),
                      passed=bool(dl.max() <= TOL_MAX and np.median(dl) <= TOL_MED))
    out["M_full"] = K.dist([x for r in res["full"] for x in r["M"]])
    return out


def d_glce(ctx):
    """G-LCE-1 (8 F-sym contexts, items 0-7, k in {0, 2, 3, 5}, interleaved order 1, one attribute both assignments, at 16,384 and
    32,000) and G-LCE-2 (every forward under the memory-efficient SDPA kernel; a math fallback raises): lce/g_lce_1_2.json."""
    env = Env(ctx); specs = glce_contexts(env, min(8, env.c.n_sym)); g1 = {}
    for L in env.c.L_parity:
        g1[str(L)] = parity(env, specs, L)
    rec = env.eng.record()
    ctx.write_json(C.FILES["glce"]["rel"], {"G-LCE-1": dict(passed={L: {p: v[p]["passed"] for p in ("crop", "copy")} for L, v in g1.items()},
                                                           metrics=g1, tolerance=dict(max_abs=TOL_MAX, median_abs=TOL_MED)),
                                            "G-LCE-2": dict(passed=True, kernel=rec["kernel"], rule="all forwards under SDPA EFFICIENT_ATTENTION; "
                                                            "a math fallback raises (the step would fail)"),
                                            "gpu": rec.get("gpu"), "versions": rec["versions"], "engine": rec,
                                            "labels": ["MOCK (tiny random model)"] if ctx.mock else []},
                   config=env.config(lengths=list(env.c.L_parity)))


def d1_ends(ctx):
    """Pure (2) and X_N (1) contexts for all F-sym items at L0, 4,096, 16,384 and 32,000 (states captured), and items 0-15's main
    interleaved arm at 4,096 (descriptive degradation curve)."""
    env = Env(ctx); c = env.c; out = []
    Ls = [0] + list(c.L_deg) + [c.L_long[0]]
    for L in Ls:
        sp = [s for i in range(c.n_sym) for s in (pure("pure", "sym", i, c, 0), pure("pure", "sym", i, c, 1), xn("xn", i, c))]
        if L == c.L_deg[0]:
            sp += [mixed(ctx, "deg", "sym", i, c, sw, k, o) for i in range(n_items(c, 16)) for sw in (0, 1) for k in range(1, 6) for o in (1, 2)]
        out += run_specs(env, sp, L, "crop" if L == 0 else glce_path(ctx, L), "ends")
    save_raw(env, "ends", out, dict(lengths=Ls))


def glce_path(ctx, L):
    """The path G-LCE-1 passed at L (crop preferred; FB-1 = copy); 'crop' where L was not gated (4,096) or the record is absent."""
    g = K.load_json(ctx.path(C.FILES["glce"]["rel"]), {}).get("G-LCE-1", {}).get("passed", {}).get(str(L))
    if not g:
        return "crop"
    return "crop" if g.get("crop") else ("copy" if g.get("copy") else "crop")


def d1_lstar(ctx):
    """L* in session (D.4 G-deg, OD-G8): the largest L in {32,000, 16,384} passing G-LCE-1 (either path), G-M (median committed mass of
    the pure and X_N contexts >= 0.9) and rho_L >= 0.5; FB-1 (copy path) if only the copy path passed; FB-2 (16K full recompute) if both
    paths failed at both lengths; none otherwise."""
    c = S.cfg(ctx); g = K.load_json(ctx.path(C.FILES["glce"]["rel"]))["G-LCE-1"]["passed"]
    recs = K._torch().load(ctx.path("d1/raw_ends.pt"), weights_only=False)["records"]

    def l0(L):
        v = [np.clip(x, -S.CLIP, S.CLIP) for r in recs if r["arm"] == "pure" and r["L"] == L for x in r["l"]]
        return float(np.mean(v)) if v else float("nan")

    def gm(L):
        return float(np.median([m for r in recs if r["arm"] in ("pure", "xn") and r["L"] == L for m in r["M"]]))
    rows, lstar, path = {}, None, None
    for L in c.L_long:
        rho = l0(L) / l0(0) if l0(0) else float("nan"); pc = g.get(str(L), {})
        ok = (pc.get("crop") or pc.get("copy")) and gm(L) >= 0.9 and rho >= 0.5
        rows[str(L)] = dict(G_LCE_1=pc, G_M_median=gm(L), rho=K.val(rho), passed=bool(ok))
        if ok and lstar is None:
            lstar, path = L, ("crop" if pc.get("crop") else "copy")
    fb2 = not any(v.get("crop") or v.get("copy") for v in g.values()) or (ctx.mock and os.environ.get("R1B_D_FORCE_FB2") == "1")
    mode = "FB-2" if fb2 else ("FB-1" if path == "copy" else ("crop" if lstar else None))
    if fb2:
        lstar, path = c.L_long[1], "full"
    elif lstar is None and ctx.mock:          # MOCK: the gates are recorded, not enforced (the tiny random model commits no mass)
        lstar, path, mode = c.L_long[0], "crop", "MOCK (gates recorded, not enforced)"
    ctx.write_json("d1/lstar.json", dict(L_star=lstar, path=path, mode=mode, lengths=rows, rho_L={str(L): K.val(l0(L) / l0(0) if l0(0) else None)
                                                                                                   for L in [0] + list(c.L_deg) + [c.L_long[0]]},
                                         labels=[f"engine fallback {mode}"] if mode in ("FB-1", "FB-2") else []),
                   config=dict(rule="D.4 G-deg / G-LCE-1 / G-M", cfg=dict(c)))


def specs_at(ctx, env, L, long=False):
    """Every D1 arm's contexts at one length (main k = 1..5, release, non-competing (0,5,5), hedging, D1-c, D1-d, D1-e, and at L0
    D1-e's closed-book check and D1-r). Pure and X_N questions come from d1_ends; D1-d re-runs them with the extras only."""
    c = env.c; sp = []
    for i in range(c.n_sym):
        for sw in (0, 1):
            for k in range(1, 6):
                for o in (1, 2):
                    sp.append(mixed(ctx, "main", "sym", i, c, sw, k, o, cap=(k == 5), dext=(k == 5 and i < n_items(c, 24))))
    for i in range(n_items(c, 24, "T6", ctx)):
        sp += [mixed(ctx, "rel", "sym", i, c, sw, k, 0, "rel") for sw in (0, 1) for k in range(1, 6)]
    for i in range(c.n_sym):
        sp += [ycomp(ctx, "nc", i, c, sw, o, (0, 5, 5), dext=(i < n_items(c, 24))) for sw in (0, 1) for o in (1, 2)]
    for i in range(n_items(c, 32)):
        sp += [mixed(ctx, "hedge", "sym", i, c, sw, k, o, sub="C") for sw in (0, 1) for k in (2, 4) for o in (1, 2)]
    if not (long and ctx.trimmed("T2")):
        for i in range(n_items(c, 24)):
            sp += [ycomp(ctx, "c", i, c, sw, o, kby, name=nm) for nm, kby in (("T1", (1, 4, 5)), ("T2", (2, 5, 3)), ("T4", (2, 4, 4)))
                   for sw in (0, 1) for o in (1, 2)]
    for i in range(n_items(c, 24)):       # D1-d: X_N first (KL reference), then the pure contexts, with the extras only
        sp += [xn("d", i, c, cap=False, dext=True, qs=[]), pure("d", "sym", i, c, 0, False, dext=True, qs=[]),
               pure("d", "sym", i, c, 1, False, dext=True, qs=[])]
    for i in range(min(24, c.n_asym)):
        for sw in (0, 1):
            sp.append(pure("e", "asym", i, c, sw, cap=False))
            sp += [mixed(ctx, "e", "asym", i, c, sw, k, o) for k in (2, 4, 5) for o in (1, 2)]
        if not long:
            sp.append(spec("ecb", "asym", i, c, 0, 0, 0, "closed", None, q12(), False))
    if not long:
        for i in range(n_items(c, 24)):
            for e in ("seg", "char"):
                for sw in (0, 1):
                    for k in range(6):
                        s_ = pure("r", "sym", i, c, sw, False, e) if k == 0 else dict(mixed(ctx, "r", "sym", i, c, sw, k, 1, qs=q12()), enc=e)
                        s_["id"] += f"|{e}"; s_["enc"] = e; sp.append(s_)
        for i in range(min(24, c.n_asym)):
            for sw in (0, 1):
                for k in (0, 2, 4, 5):
                    s_ = pure("r", "asym", i, c, sw, False, "char") if k == 0 else dict(mixed(ctx, "r", "asym", i, c, sw, k, 1, qs=q12()), enc="char")
                    s_["id"] += "|char"; s_["enc"] = "char"; sp.append(s_)
    d_order = {"d": 0}
    for s_ in sp:                          # D1-d contexts of an item in order: X_N, pure, then the competing / non-competing ones
        s_["_o"] = (s_["item"] if s_["fam"] == "sym" else 99, d_order.get(s_["arm"], 1) if s_.get("dext") else 2)
    return sorted(sp, key=lambda s_: (s_["stream"], s_["_o"]))


def d1_L0(ctx):
    env = Env(ctx); sp = specs_at(ctx, env, 0)
    save_raw(env, "L0", run_specs(env, sp, 0, "crop", "L0"), dict(L=0, n_specs=len(sp)))


def d1_Ls(ctx):
    """Every arm at L* (the in-session L*, D.4); under FB-2 the D1 main arm as a 16K full recompute (items 0-31, one attribute and one
    letter assignment per context, k = 0..4, both swaps, order and reverse); the other arms then read 'not testable (engine parity)'."""
    ls = K.load_json(ctx.path("d1/lstar.json"))
    if ls["L_star"] is None:
        raise K.NotRun("no gated long length (D1-a: not decidable)")
    env = Env(ctx); c = env.c; L = ls["L_star"]
    if ls["mode"] == "FB-2":
        sp = []
        for i in range(n_items(c, 32)):
            for k in range(5):
                for sw in (0, 1):
                    for o in (1, 2):
                        j = (i + k + sw + o) % 6; bf = (o == 1)
                        s_ = mixed(ctx, "fb2", "sym", i, c, sw, k, o, qs=[(j, bf)]) if k else pure("fb2", "sym", i, c, sw, False, qs=[(j, bf)])
                        s_["id"] += f"|o{o}"; s_["order"] = o; sp.append(s_)
        save_raw(env, "Ls", run_specs(env, sp, L, "full", "Ls"), dict(L=L, mode="FB-2", n_specs=len(sp)),
                 labels=["engine fallback FB-2: 16K full recompute, 32 items, one attribute per context"])
        return
    sp = specs_at(ctx, env, L, long=True)
    save_raw(env, "Ls", run_specs(env, sp, L, ls["path"], "Ls"), dict(L=L, mode=ls["mode"], n_specs=len(sp), trims=dict(T2=ctx.trimmed("T2"))))


def d1_Lf(ctx):
    """D1-f: F-sym items 0-23 (16 under T5) at L* on the sentence-shuffled version of their own stream (i mod 16; all 16 streams are
    shuffled, D.9 deviation): interleaved k in {0, 2, 4}, 2 swaps x (1 context with 12 questions + 2 x 2 with 6 questions)."""
    ls = K.load_json(ctx.path("d1/lstar.json"))
    if ls["L_star"] is None or ls["mode"] == "FB-2":
        raise K.NotRun("not testable (engine parity)" if ls["mode"] == "FB-2" else "no gated long length")
    env = Env(ctx); c = env.c; sp = []
    for i in range(n_items(c, 24, "T5", ctx)):
        for sw in (0, 1):
            sp.append(pure("f", "sym", i, c, sw, False))
            sp += [mixed(ctx, "f", "sym", i, c, sw, k, o) for k in (2, 4) for o in (1, 2)]
    save_raw(env, "Lf", run_specs(env, sp, ls["L_star"], ls["path"], "Lf", shuf=True), dict(L=ls["L_star"], n_specs=len(sp)))


def d1_v(ctx):
    """D1-v: each word in 'The word is: {w}.' (user turn), states at the word's last token at blocks 14 and 18, canonical / segmented
    (keyseed('r1bD1v', w)) / character-level; 3CosAdd forward and reverse per family, inputs excluded and included (Nissim), offset
    consistency, retention, 24 random quadruples per family (shuffled targets) as the chance floor."""
    torch = K._torch(); tok, model = ctx.model(); tk = S.Tk(tok); c = S.cfg(ctx)
    P, sha = S.load_pkg(ctx, "d1v_pairs.json", tok); eng = LCE(model); H = {}
    words = sorted({w for f in P["families"].values() for pr in f[:c.d1v_pairs] for w in pr})
    for e in ("can", "seg", "char"):
        for w in words:
            wid = tk.recode(" " + w, e, np.random.default_rng(ctx.key("r1bD1v", w)) if e == "seg" else None)
            ids = tk.pre + tk.enc("The word is:") + wid + tk.enc(".") + tk.post; pos = len(tk.pre) + len(tk.enc("The word is:")) + len(wid) - 1
            with K.Hooks(model) as hk:
                hk.capture(list(CAP), positions=[pos]).stop_after(max(CAP))
                hk.run(lambda: eng.fwd(ids))
            H[(e, w)] = {b: hk.states[b][0][0, 0].double() for b in CAP}
    res = {}
    for fam, pairs in P["families"].items():
        pairs = [tuple(p) for p in pairs[:c.d1v_pairs]]
        rng = np.random.default_rng(K.keyseed("r1bD1v", fam, "floor") % (2 ** 32)); perm = rng.permutation(len(pairs))
        shuf = [(pairs[i][0], pairs[perm[i]][1]) for i in range(len(pairs))]
        for b in CAP:
            for e in ("can", "seg", "char"):
                h = lambda w: H[(e, w)][b]
                res[f"{fam}|b{b}|{e}"] = dict(analogy(pairs, h), floor=analogy(shuf, h))
    for k, v in res.items():
        fam, b, e = k.split("|")
        can = res[f"{fam}|{b}|can"]
        v["retention"] = {d_: K.val(v[d_]["acc_excl"] / can[d_]["acc_excl"] if can[d_]["acc_excl"] else None, "NaN:canonical_accuracy_0")
                          for d_ in ("forward", "reverse")}
    ctx.save_pt("d1/d1v_states.pt", {f"{e}|{w}|{b}": H[(e, w)][b].float() for (e, w) in H for b in CAP}, config=dict(sha=sha))
    ctx.write_json("d1/d1v.json", dict(results=res, n_pairs=c.d1v_pairs, words=len(words)), config=dict(d1v_sha256=sha, blocks=CAP))


def analogy(pairs, h):
    """3CosAdd over ordered pairs of one family: forward predicts b* from h(a*) - h(a) + h(b) among the targets, reverse predicts b
    from h(a) - h(a*) + h(b*) among the sources; inputs excluded / included; offset consistency = mean cosine between pair offsets."""
    torch = K._torch()
    src = [p[0] for p in pairs]; tgt = [p[1] for p in pairs]
    S_, T_ = torch.stack([h(w) for w in src]), torch.stack([h(w) for w in tgt])
    nrm = lambda x: x / x.norm(dim=-1, keepdim=True)
    out = {}
    for d_, A, Bm in (("forward", S_, T_), ("reverse", T_, S_)):
        ok_e, ok_i, n = 0, 0, 0
        for a in range(len(pairs)):
            for b in range(len(pairs)):
                if a == b:
                    continue
                cos = nrm(Bm) @ nrm(Bm[a] - A[a] + A[b])
                ok_i += int(cos.argmax()) == b
                ce = cos.clone(); ce[a] = -2
                ok_e += int(ce.argmax()) == b; n += 1
        out[d_] = dict(acc_excl=ok_e / n, acc_incl=ok_i / n, n=n)
    off = nrm(T_ - S_); cs = off @ off.T; m = len(pairs)
    out["offset_consistency"] = float((cs.sum() - cs.diag().sum()) / (m * (m - 1)))
    return out


def d1_cov(ctx):
    """SG8 optional covariate (D.2, OD-M25): only if r1b_K_S1's SG8 preview existed and passed at the session start and SO-HN's files were
    copied; never waited for. Under the planned packing (main §4.4) the preview is not there: 'not run (H-set not available at start)'."""
    p = ctx.input("r1b_K_S1", C.FILES["sg8_preview"]["rel"], required=False)
    pv = K.load_json(p) if p else None
    if not pv or pv.get("mock"):
        raise K.NotRun("H-set not available at start (SG8 preview absent; OD-M25)")
    if "replicated" not in str(pv.get("G-K2") or pv.get("preview") or ""):
        raise K.NotRun("SG8 preview not passed")
    hs = ctx.input("r1b_K_S1", C.FILES["hn_set"]["rel"], required=False); hm = ctx.input("r1b_K_S1", C.FILES["hn_meta"]["rel"], required=False)
    if not (hs and hm):
        raise K.NotRun("H-set files not available at start (only the SG8 preview is a start condition of r1b_D_A)")
    raise K.NotRun("H-set files present; the SO-CHH channel is part K's code (r1b_hn.py), not wired into part D (descriptive only)")


def d1_s(ctx):
    """D1-s (Qwen3-0.6B): its own G-LCE-1 (2 contexts at 32,000) and G-M, then F-sym items 0-23 (16 under T3) on stream i mod 16 (the
    1.7B arm's filler; D.12 deviation) at L0 and 32,000: interleaved k in {0, 2, 4} (2 swaps x (1 + 2 x 2)) and X_N, 11 contexts."""
    K.unload_models()
    env = Env(ctx, C.M06); c = env.c; L = c.L_long[0]
    g = parity(env, glce_contexts(env, 2), L)
    path = "crop" if g["crop"]["passed"] else ("copy" if g["copy"]["passed"] else None)
    ctx.write_json("d1/d1s_glce.json", {"G-LCE-1": g, "path": path, "engine": env.eng.record()}, config=env.config(L=L))
    out = []
    for LL in (0, L):
        if LL and path is None:
            continue
        sp = []
        for i in range(n_items(c, 24, "T3", ctx)):
            for sw in (0, 1):
                sp.append(pure("s", "sym", i, c, sw, False))
                sp += [mixed(ctx, "s", "sym", i, c, sw, k, o) for k in (2, 4) for o in (1, 2)]
            sp.append(xn("s", i, c, cap=False))
        out += run_specs(env, sp, LL, "crop" if LL == 0 else path, "s")
    save_raw(env, "s", out, dict(L=[0, L], path_32k=path, labels=[] if path else ["D1-s at 32,000: not testable (engine parity)"]))


def p0_d(ctx):
    """P0-D (r1b_P0, engineering numbers only, D.2 'Pilot'): on random token ids (no analysis material; keys 'r1bP0_D'), per length
    4K / 16K / 28K / 32K: B = 1 prefill, a 700-token chunk over the cached prefix (crop path) and the copy path, keyed decode rates
    for B in {1, 2, 3} (B = 4 at 32K as a memory probe) with DynamicCache, B = 1 decode with a non-rewriting StaticCache (token parity
    with DynamicCache), peak memory, SDPA kernel; Drive write throughput."""
    import time
    torch = K._torch(); c = S.cfg(ctx); tok, model = ctx.model(); eng = LCE(model); tk = S.Tk(tok); stop = tk.stops(model)
    Ls = (4096, 16384, 28672, 32000) if not c.mock else (1000, 1200, 1400, 1500); rng = np.random.default_rng(20261114)
    rates, memory, parity, notes = {}, {}, {}, []
    cuda = torch.cuda.is_available(); sync = (lambda: torch.cuda.synchronize()) if cuda else (lambda: None)
    def _free():
        import gc
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()

    def _oom(e):
        return isinstance(e, getattr(torch, "OutOfMemoryError", RuntimeError)) or "out of memory" in str(e).lower()

    for L in Ls:                                  # every stage in its own guard: an out-of-memory error is a probe result here
        ids = [int(x) for x in rng.integers(1000, 100000, L - 700)]; chunk = [int(x) for x in rng.integers(1000, 100000, 700)]
        r, base = {}, None
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        try:
            sync(); t = time.time(); base, lg = eng.fwd(ids); sync(); r["prefill_s"] = time.time() - t
            t = time.time(); base, lg2 = eng.fwd(chunk, base); sync(); r["chunk_s"] = time.time() - t; eng.crop(base, len(ids))
            lg2 = lg2[0, -1].clone(); del lg
        except RuntimeError as e:
            if not _oom(e):
                raise
            r["prefill_s"] = K.coded("NaN:oom"); notes.append(f"L {L} prefill/chunk: {str(e)[:80]}"); base = None; _free()
        if base is not None:
            try:
                t = time.time(); cp = LCE.copy(base); cp, lg3 = eng.fwd(chunk, cp); sync(); r["copy_chunk_s"] = time.time() - t
                parity[f"{L}|crop_vs_copy_max_abs_logit"] = float((lg2 - lg3[0, -1]).abs().max()); del cp, lg3
            except RuntimeError as e:
                if not _oom(e):
                    raise
                r["copy_chunk_s"] = K.coded("NaN:oom"); notes.append(f"L {L} copy path: {str(e)[:80]}")
            _free()
            n = 32 if not c.mock else 8
            for B in ((1, 2, 3, 4) if L == Ls[-1] else (1, 2, 3)):
                try:
                    cb = LCE.copy(base, B); _, lgb = eng.fwd(torch.tensor([chunk] * B), cb)
                    sync(); t = time.time()
                    y, *_ = eng.decode(cb, lgb[:, -1], [ctx.key("r1bP0_D", L, B, b) for b in range(B)], 0.7, n, stop); sync()
                    r[f"decode_tok_s_B{B}"] = B * n / (time.time() - t); del cb, lgb, y
                except RuntimeError as e:                 # out of memory: a probe result, not a failure
                    if not _oom(e):
                        raise
                    r[f"decode_tok_s_B{B}"] = K.coded("NaN:oom"); notes.append(f"L {L} B {B}: {str(e)[:80]}")
                _free()
            try:                                          # non-rewriting cache: StaticCache, B = 1, token parity with DynamicCache
                from transformers import StaticCache
                sc = StaticCache(config=model.config, max_cache_len=L + n + 8)
                _, lgs = eng.fwd(ids + chunk, sc); sync(); t = time.time()
                ys, *_ = eng.decode(sc, lgs[:, -1], [ctx.key("r1bP0_D", L, 1, 0)], 0.7, n, stop); sync()
                r["static_decode_tok_s_B1"] = n / (time.time() - t); del sc, lgs
                _free()
                cd = LCE.copy(base); _, lgd = eng.fwd(chunk, cd)
                yd, *_ = eng.decode(cd, lgd[:, -1], [ctx.key("r1bP0_D", L, 1, 0)], 0.7, n, stop)
                parity[f"{L}|static_vs_dynamic_tokens_identical"] = bool((ys == yd).all()); del cd, lgd
            except Exception as e:
                notes.append(f"L {L} StaticCache: {type(e).__name__}: {str(e)[:120]}")
        memory[str(L)] = dict(peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3) if cuda else K.coded("NaN:cpu"))
        rates[str(L)] = r; base = None; _free()
    blob = os.urandom(64 << 20); t = time.time(); p = ctx.path("p0/_drive_probe.bin")
    with open(p, "wb") as f:
        f.write(blob); f.flush(); os.fsync(f.fileno())
    thr = dict(local_write_MB_s=64 / max(time.time() - t, 1e-9), note="the runner's Drive sync of the session folder adds the Drive time")
    os.remove(p)
    ctx.write_json("p0/p0_d.json", dict(rates=rates, memory=memory, parity=parity, throughput=thr,
                                        env=dict(engine=eng.record(), lengths=list(Ls)), notes=notes), config=dict(lengths=list(Ls)))
