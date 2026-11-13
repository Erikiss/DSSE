"""Part R shared helpers (model side, package objects, round-1 / round-2 inputs). R.1 conventions: push s x a x min(|h_t|, cap_b) x
unit(d) at the output of block b from the first user-content token (r1b_core.Hooks.push = rk_loop's push, CC4), Gumbel-max at T 0.7,
40 new tokens (MOCK: 8, as rk_loop), the + and - texts of a pair share one key; reads: the unsteered model, plain prompt +
continuation, forward stopped after block b ('all' window = rk.read; 'div' / 'first20' only as projections on the readout set F(b));
damage = dmg_sym (mean of the + and - push damage on REF-R16, SO-DMG). rk_loop is never imported here (its literals are read with ast).
Model-free at import."""
import ast, glob, json, math, os, re
import numpy as np
import r1b_core as K
import r1b_contracts as C

BLOCKS = (14, 18, 6)
SHARD = {14: "gpu1", 18: "gpu2", 6: "gpu3"}
LENS_BLOCKS = (2, 6, 10, 14, 18, 22, 24, 26)
LOOPS_1B = ("RSI_a", "RSI_b", "SEA_t", "SPO_t", "SEA_raw", "RND_1", "RND_2")
NSEED = {"RSI_a": 2, "RSI_b": 2, "SEA_t": 2, "SPO_t": 2, "SEA_raw": 1, "RND_1": 1, "RND_2": 1}
ARMS = {"DN": LOOPS_1B, "FX": ("RSI_a", "SEA_raw", "RND_1"), "NP": ("RSI_a", "SEA_t", "RND_1")}
NSEED_ARM = {("NP", "SEA_t"): 1}                                   # R2.2: NP arm SEA_t 256 pairs
K.register_r_loops(LOOPS_1B)
TOPICS = ("rsi", "sports", "tone", "sea")                          # rk.INSTR order; X names T, sports, tone, sea
XN = {"rsi": "T", "sports": "sports", "tone": "tone", "sea": "sea"}
NULL_SEED = {"rsi": 311, "sports": 312, "tone": 313, "sea": 314}  # R.1 K1-c centred pseudo-axes
DMG_T, DMG_LO, DMG_HI = 0.05, 0.045, 0.055                         # OD-R5 / RG9
PKG = os.path.join(K.REPO, C.PLAN_DIR)
RK_SRC = os.path.join(K.AJS, "rk_loop.py")
NEUTRAL_SHA = "4b3bcaca06f45521a00f12ec2b056a13962bbfdd066ab93867980c9b0be9ebf5"
MOCK_Q = (0, 1, 2, 3, 32, 33, 34, 35, 128, 129, 130, 131, 160, 161, 162, 163)   # MOCK: 16 of the 256 prompts (8 RSI-topic, 8 scene)

W_SEA = ("sea seas ocean oceans oceanic wave waves tide tides tidal beach beaches shore shores shoreline coast coasts coastal coastline "
         "seaside harbour harbours harbor harbors sailor sailors sailing sailboat boat boats ship ships seagull seagulls gull gulls marine "
         "maritime surf surfing lighthouse lighthouses").split()
W_RSI = sorted(("improve improvement improving self-improvement better learn learning upgrade recursive optimize optimise optimization "
                "capability capabilities intelligence smarter evolve enhance enhancement grow growth progress refine iterate iteration adapt "
                "train training autonomous rewrite").split())
W_SPORTS = "sport sports football soccer basketball tennis game team match player athlete coach score goal race".split()
W_TONE = "amazing wonderful fantastic exciting incredible love awesome thrilled delighted brilliant great joy beautiful magical superb".split()
WORDSETS = {"W_RSI": W_RSI, "W_RSI_build": W_RSI[0::2], "W_RSI_eval": W_RSI[1::2],   # 1-based odd positions = build half
            "W_sea": W_SEA, "W_sports": W_SPORTS, "W_tone": W_TONE}
WSET_NAMES = tuple(WORDSETS)


def torch():
    return K._torch()


# ---------------------------------------------------------------------------------------------------------------- package objects
def rk_literal(name):
    """A literal assigned at module level in rk_loop.py (INSTR, RSI_PROMPTS, SCENE_PROMPTS, NEG_PAIRS), verbatim via ast."""
    for node in ast.parse(open(RK_SRC).read()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def r1_prompts(mock=False):
    """rk_loop's PROMPTS (MOCK: RSI_PROMPTS[:4] + SCENE_PROMPTS[:4], as rk_loop's MOCK)."""
    a, b = rk_literal("RSI_PROMPTS"), rk_literal("SCENE_PROMPTS")
    return a[:4] + b[:4] if mock else a + b


def r1_index(q, mock=False):
    """Round-1 prompt index of P256 index q (-1 for a new prompt); MOCK: the 8 prompts of rk_loop's MOCK state."""
    if mock:
        return q if q < 4 else (4 + q - 128 if 128 <= q < 132 else -1)
    return q if q < 32 else (q - 96 if 128 <= q < 160 else -1)


def neutral_pool():
    """The frozen neutral pool (Appendix R.A, 628 words): the package file if present, else parsed from the part plan; sha256 pinned."""
    p = os.path.join(PKG, "neutral_pool_1b.txt")
    if os.path.exists(p):
        txt = open(p).read()
    else:
        t = open(os.path.join(PKG, "runde1b_plan_teil_R.md")).read(); a = t[t.index("## Appendix R.A"):]
        par = [x for x in a.split("\n\n") if x.startswith("airport")][0]
        txt = "".join(w + "\n" for w in sorted(x.strip() for x in par.replace("\n", " ").split(",") if x.strip()))
    if K.sha256_text(txt) != NEUTRAL_SHA:
        raise ValueError("neutral pool sha256 differs from Appendix R.A")
    return txt.split()


def p256_fold():
    """R.2 P256 fold: round-1 fold (rk_loop.axes: randperm(64, seed 5)[:32]) for indices 0-31 / 128-159; new prompts: rng(105) halves
    within type (48 + 48; RSI-topic first)."""
    t = torch(); f1 = t.zeros(64, dtype=t.bool); f1[t.randperm(64, generator=t.Generator().manual_seed(5))[:32]] = True
    f1 = f1.numpy(); g = np.random.default_rng(105); fold = np.zeros(256, bool)
    fold[0:32], fold[128:160] = f1[:32], f1[32:]
    for lo in (32, 160):
        m = np.zeros(96, bool); m[g.permutation(96)[:48]] = True; fold[lo:lo + 96] = m
    return fold


def p256(ctx):
    """P256 (R.2): dict(prompts, q (P256 indices), typ, r1 (round-1 index or -1), fold, n_rsi, sha256, source). The package file
    prompts_1b.json ({'prompts': [256 str]}); MOCK without it: the 64 round-1 prompts + placeholders. MOCK uses the 16 prompts MOCK_Q."""
    p = os.path.join(PKG, "prompts_1b.json")
    if os.path.exists(p):
        d = K.load_json(p); pr, src, sha = list(d["prompts"]), p, K.sha256_file(p)
        assert len(pr) == 256, "prompts_1b.json must hold 256 prompts"
    elif ctx.mock:
        r1 = r1_prompts(); pr = r1[:32] + [f"Mock new topic prompt number {i}?" for i in range(96)] + r1[32:] + \
            [f"Describe mock scene number {i}." for i in range(96)]; src, sha = "MOCK", "MOCK"
    else:
        raise K.NotRun("input missing: package file prompts_1b.json")
    r1i = np.array([r1_index(q, ctx.mock) for q in range(256)])
    q = np.array(MOCK_Q if ctx.mock else range(256)); fold = p256_fold()[q]
    if ctx.mock:
        fold = np.array([i % 2 == 0 for i in range(len(q))])
    return dict(prompts=[pr[i] for i in q], q=q.tolist(), typ=["rsi" if i < 128 else "scene" for i in q], r1=r1i[q].tolist(),
                fold=fold, n_rsi=int((q < 128).sum()), sha256=sha, source=src)


_WRE = {}


def wcount(texts, sets=None):
    """Whole-word, case-insensitive counts of every WORDSETS list (or the given {name: words}) in decoded texts: int32 [N, n_sets]."""
    sets = sets or WORDSETS
    out = np.zeros((len(texts), len(sets)), np.int32)
    for j, (nm, ws) in enumerate(sets.items()):
        k = (nm, tuple(ws))
        if k not in _WRE:
            _WRE[k] = re.compile(r"(?<![\w-])(" + "|".join(map(re.escape, sorted(ws, key=len, reverse=True))) + r")(?![\w-])", re.I)
        for i, s in enumerate(texts):
            out[i, j] = len(_WRE[k].findall(s))
    return out


# ---------------------------------------------------------------------------------------------------------------- round-1 / round-2 inputs
_CACHE = {}


def r1_file(ctx, b, rel, required=True):
    """A copied round-1 file of block b (found by name across gpu*/, the block's shard first)."""
    hits = sorted(glob.glob(os.path.join(ctx.r1, "gpu*", rel.format(b=b))), key=lambda p: SHARD.get(b, "") not in p)
    if hits:
        return hits[0]
    if required:
        raise K.NotRun(f"input missing: round-1 {rel.format(b=b)}")
    return None


def r1_state(ctx, b):
    k = ("st", ctx.r1, b)
    if k not in _CACHE:
        _CACHE[k] = torch().load(r1_file(ctx, b, "rk/b{b}/state.pt"), map_location="cpu", weights_only=False)
    if "axes" not in _CACHE[k]:
        raise K.NotRun("input missing: round-1 state is a placeholder (MOCK without R1B_MOCK_WITH_R1=1)")
    return _CACHE[k]


def r1_loops(ctx, b):
    return K.load_json(r1_file(ctx, b, "rk/b{b}/loops.json"))


def alpha0(ctx, b):
    """alpha_0(b) = R1.loops[b]['_meta']['alpha'] (0.15 at 14, 0.10850 at 18, 0.13444 at 6)."""
    return float(r1_loops(ctx, b)["_meta"]["alpha"])


def r2_file(ctx, rel, required=True):
    for base in (ctx.r2, os.path.join(ctx.inp, "r1b_R_S0", "r2copy")):
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
    if required:
        raise K.NotRun(f"input missing: round-2 {rel}")
    return None


def ustar(ctx, b):
    """u*_b (SO-AX U*): the round-2 file r2_q2/theta/b<b>_q0.pt['u_star'] (canonical, unchanged), or r1b_R_S0's so_ax copy."""
    p = os.path.join(ctx.inp, "r1b_R_S0", f"so_ax/so_ax_b{b}.pt")
    if os.path.exists(p):
        d = torch().load(p, map_location="cpu", weights_only=False)
        return d["u_star"].double(), dict(d["u_star_src"])
    p = r2_file(ctx, f"r2_q2/theta/b{b}_q0.pt"); d = torch().load(p, map_location="cpu", weights_only=False)
    if "u_star" not in d:
        raise K.NotRun(f"input missing: r2_q2/theta/b{b}_q0.pt is a placeholder")
    return d["u_star"].double(), dict(path=f"r2_q2/theta/b{b}_q0.pt", sha256=K.sha256_file(p))


def tv(ctx, required=True):
    """TV = theta_verdict.json (the round-2 Drive copy, else the repository copy, byte-identical)."""
    p = r2_file(ctx, "r2_ana/theta_verdict.json", required=False)
    if p and not (K.load_json(p) or {}).get("mock"):
        return K.load_json(p)
    q = os.path.join(K.REPO, "results/ju_space/runde2_ergebnisse/theta_verdict.json")
    if os.path.exists(q):
        return K.load_json(q)
    if required:
        raise K.NotRun("input missing: theta_verdict.json")
    return None


def ax1b(ctx, b):
    """AX-1b(b) of r1b_R_S2_<b> (own output, or the copy of an upstream session)."""
    rel = f"ax1b/ax1b_b{b}.pt"
    for p in (os.path.join(ctx.work, rel), os.path.join(ctx.inp, f"r1b_R_S2_{b:02d}", rel)):
        if os.path.exists(p):
            k = ("ax", p)
            if k not in _CACHE:
                _CACHE[k] = torch().load(p, map_location="cpu", weights_only=False)
            if "axes" not in _CACHE[k]:
                raise K.NotRun(f"input missing: {rel} is a placeholder")
            return _CACHE[k]
    raise K.NotRun(f"input missing: {rel}")


def pool_dir(ctx, kind, b, i, D):
    """rk_dirs.pt Pool directions (('cov', b, i), ('rand', b, i)); MOCK (size mismatch): a seeded random unit vector."""
    t = torch()
    for p in (os.environ.get("POOLDIRS"), os.path.join(ctx.root, "gpu_pkg", "rk_dirs.pt"), os.path.join(K.AJS, "gpu", "data", "rk_dirs.pt")):
        if p and os.path.exists(p):
            PD = t.load(p, map_location="cpu", weights_only=True); v = PD.get((kind, b, i))
            if v is not None and v.numel() == D:
                return K.unit(v.double())
            break
    if not ctx.mock:
        raise K.NotRun(f"input missing: pool direction {(kind, b, i)}")
    return K.unit(t.randn(D, generator=t.Generator().manual_seed(int(K.sha256_text(repr((kind, b, i)))[:8], 16)), dtype=t.float64))


# ---------------------------------------------------------------------------------------------------------------- model side
class M:
    """The session's model with rk_loop's conventions: tok, model, D, START, PAD, nnew (40; MOCK 8), cached chat ids."""

    def __init__(self, ctx, name=C.M17):
        self.tok, self.model = ctx.model(name); self.D = self.model.config.hidden_size; self.START = K.user_start(self.tok)
        self.PAD = self.tok.pad_token_id; self.nnew = 8 if ctx.mock else 40; self._ids = {}; self.ctx = ctx
        self.L = len(K.layers_of(self.model))

    def ids(self, text):
        if text not in self._ids:
            self._ids[text] = K.chat_ids(self.tok, text)
        return self._ids[text]

    def decode(self, conts):
        return [self.tok.decode(c) for c in conts]


def recorder(hk, b, cap, rec):
    """Generation-mode cbar (r2_rk.CbarHook): a prepend hook at block b logging min(|h_t|, cap_b) before the push at every decode step."""
    t = torch()

    def hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        v = x.detach().norm(dim=-1).clamp(max=cap) if cap is not None else x.detach().norm(dim=-1)
        if x.shape[1] > 1:
            rec.setdefault("calls", []).append(dict(B=x.shape[0], steps=[]))
        elif rec.get("calls"):
            rec["calls"][-1]["steps"].append(v[:, 0].double().cpu())
        return None
    hk.h.append(hk.L[b].register_forward_hook(hook, prepend=True))
    return hk


def cbar_of(rec, conts):
    """Per-text means and the pooled mean of the logged values (decode steps 0 .. min(L, steps) - 1 of each row)."""
    t = torch(); vals = []
    for c in rec.get("calls", []):
        S = t.stack(c["steps"]) if c["steps"] else t.zeros(0, c["B"], dtype=t.float64)
        for r in range(c["B"]):
            vals.append(S[: min(len(conts[len(vals)]), S.shape[0]), r])
    n = sum(len(v) for v in vals)
    return dict(per_text=[float(v.mean()) if len(v) else float("nan") for v in vals], pooled=float(sum(float(v.sum()) for v in vals) / n) if n
                else float("nan"), n_tokens=int(n))


def np_push(hk, b, d, a, cap, signs, rowstart, rec=None):
    """PUSH-NP (R3.2): at the pushed positions x' = (x + add) |x| / |x + add| with rk's add; unpushed positions unchanged. rec collects
    the along-d displacement per unit a (sign-aligned, per pushed token) and |cos(x_t, unit(d))| over pushed tokens."""
    t = torch(); v = K.unit(t.as_tensor(d).float()); rs = None if rowstart is None else t.as_tensor(rowstart)

    def hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out; B, T = x.shape[:2]
        m = ((hk.offset + t.arange(T, device=x.device))[None, :] >= rs.to(x.device)[:, None]).to(x.dtype)[..., None] \
            if (T > 1 and rs is not None) else t.ones(B, T, 1, dtype=x.dtype, device=x.device)
        raw = x.norm(dim=-1, keepdim=True).detach(); nrm = raw.clamp(max=cap) if cap is not None else raw
        sg = t.as_tensor(signs, dtype=t.float32); sg = sg.expand(B) if sg.dim() == 0 else sg
        vv = v.to(x.device, x.dtype); add = a * nrm * sg.to(x.device, x.dtype)[:, None, None] * vv * m
        y = x + add; y = t.where(m > 0, y * raw / y.norm(dim=-1, keepdim=True).clamp_min(1e-12), x)
        if rec is not None:
            with t.no_grad():
                sel = (m[..., 0] > 0)
                disp = (((y - x) @ vv) * sg.to(x.device, x.dtype)[:, None] / (a * nrm[..., 0]).clamp_min(1e-12))[sel]
                cs = ((x @ vv).abs() / raw[..., 0].clamp_min(1e-12))[sel]
                rec.setdefault("disp", []).append(disp.double().cpu()); rec.setdefault("cos", []).append(cs.double().cpu())
        return (y,) + tuple(out[1:]) if isinstance(out, tuple) else y
    hk.h.append(hk.L[b].register_forward_hook(hook))
    return hk


def pusher(m, b, d, a, cap, mode="add", rec=None, extra=None):
    """setup(rb, pad) for r1b_core.generate: the push of mode 'add' (Hooks.push) or 'np' with per-row signs r[2], from pad + START;
    rec: generation-mode cbar (and NP statistics); extra(hk, rb, pad): further hooks (erasure)."""
    t = torch()

    def setup(rb, pad):
        hk = K.Hooks(m.model)
        if rec is not None:
            recorder(hk, b, cap, rec)
        if extra is not None:
            extra(hk, rb, pad)
        if d is not None and a:
            sg = t.tensor([float(r[2]) for r in rb])
            if mode == "add":
                hk.push(b, d, a, signs=sg, cap=cap, rowstart=pad + m.START)
            else:
                np_push(hk, b, d, a, cap, sg, pad + m.START, rec)
        return hk
    return setup


def first_div(a, b):
    if a == b:
        return -1
    return next((i for i, (u, v) in enumerate(zip(a, b)) if u != v), min(len(a), len(b)))


def read_win(m, prompts, conts, b, fd=None, F=None, batch=32, extra=None):
    """rk.read at block b for every row ('all' window, float64 [N, D]; NaN for an empty text) and the B-Q1 windows 'div' (positions >=
    the pair's first divergence; an identical pair: its 'all' state) and 'first20' projected on F [k, D] (None: not computed).
    Rows 2j / 2j + 1 are the pair j when fd (per pair) is given. extra(hk): hooks active during the read (erasure)."""
    t = torch(); dev = K.model_device(m.model); N = len(conts); X = t.full((N, m.D), float("nan"), dtype=t.float64)
    k = 0 if F is None else F.shape[0]; Pd = t.full((N, k), float("nan"), dtype=t.float64); P20 = Pd.clone()
    Ft = None if F is None else F.double().T
    for i in range(0, N, batch):
        pb, cb = prompts[i:i + batch], conts[i:i + batch]; ids, am, pad = K.pad_left([list(p) + list(c) for p, c in zip(pb, cb)], m.PAD)
        with K.Hooks(m.model) as hk, t.no_grad():
            if extra is not None:
                extra(hk)
            hk.capture([b]).stop_after(b); hk.run(lambda: m.model.model(input_ids=ids.to(dev), attention_mask=am.to(dev)))
            hs = hk.states[b][0].double()
        Tn = ids.shape[1]
        for r, c in enumerate(cb):
            L = len(c)
            if not L:
                continue
            seg = hs[r, Tn - L:]; X[i + r] = seg.mean(0)
            if Ft is not None:
                P20[i + r] = seg[:20].mean(0) @ Ft
                f = -1 if fd is None else fd[(i + r) // 2]
                if f < 0:
                    Pd[i + r] = X[i + r] @ Ft
                elif L > f:
                    Pd[i + r] = seg[f:].mean(0) @ Ft
    return X, Pd, P20


def ref16(ctx, m, b):
    """REF-R16 at block b: R1.state[b]['ref'] continuations of round-1 prompts 0-7 (RSI-topic) and 32-39 (scene), with rk_loop's
    prompt ids (SO-DMG)."""
    st = r1_state(ctx, b if b in BLOCKS else 14); pr = r1_prompts(ctx.mock); P = list(range(8)) if ctx.mock else [0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37, 38, 39]
    return K.RefSet.from_prompt_conts([m.ids(pr[p]) for p in P], [list(map(int, st["ref"]["conts"][p])) for p in P], f"REF-R16(b{b})")


class Dmg:
    """dmg_sym (R3.1): mean of the + and - push damage on REF-R16 at block b; the 8 scene references' top-1 change ('neutral top-1
    change'); REF-K16 (+ push, descriptive) for the REF-U32 report when the package file exists."""

    def __init__(self, ctx, m, b, cap, refs=None):
        self.m, self.b, self.cap = m, b, cap; self.dm = K.Damage(m.model, refs or ref16(ctx, m, b)); self.k16 = None
        p = os.path.join(PKG, "ref_k16.json")
        if os.path.exists(p):
            self.k16 = K.Damage(m.model, K.RefSet.from_json(p, "REF-K16"))

    def sym(self, setup):
        """setup(sign) -> a context manager; returns dict(sym, plus, minus, asym, top1_scene, per_text_plus, per_text_minus)."""
        a, b = self.dm(lambda: setup(1.0)), self.dm(lambda: setup(-1.0))
        return dict(sym=(a["damage"] + b["damage"]) / 2, plus=a["damage"], minus=b["damage"], asym=a["damage"] / max(b["damage"], 1e-12),
                    top1_scene=float(np.mean(a["per_text_top1"][len(a["per_text_top1"]) // 2:] + b["per_text_top1"][len(b["per_text_top1"]) // 2:])),
                    per_text_plus=a["per_text"],
                    per_text_minus=b["per_text"])

    def push_setup(self, d, a, mode="add", extra=None):
        t = torch()

        def setup(s):
            hk = K.Hooks(self.m.model)
            if extra is not None:
                extra(hk)
            if mode == "add":
                return hk.push(self.b, d, a, signs=s, cap=self.cap, rowstart=t.tensor([self.m.START]))
            return np_push(hk, self.b, d, a, self.cap, s, t.tensor([self.m.START]))
        return setup

    def push(self, d, a, mode="add", extra=None):
        return self.sym(self.push_setup(d, a, mode, extra))

    def calibrate(self, d, start, mode="add", extra=None):
        """PUSH-DN / PUSH-NP calibration (R3.1): a <- a sqrt(0.05 / dmg_sym), band [0.045, 0.055], <= 6 evaluations; the closest kept."""
        rec = {}

        def f(a):
            rec[a] = self.push(d, a, mode, extra); return rec[a]["sym"]
        c = K.calibrate(f, start, DMG_T, form="push", lo=DMG_LO, hi=DMG_HI)
        r = rec[c["f"]]
        return dict(a=c["f"], dmg_sym=c["damage"], in_band=c["in_band"], tried=c["tried"], plus=r["plus"], minus=r["minus"],
                    asym=r["asym"], asym_label="asymmetric damage" if not (0.5 <= r["asym"] <= 2) else None, top1_scene=r["top1_scene"],
                    labels=c["labels"])

    def u32(self, d, a, mode="add", extra=None):
        """Damage of the + push on REF-U32 = REF-R16 + REF-K16 (descriptive, SO-DMG); coded when REF-K16 is not available."""
        if self.k16 is None:
            return K.coded("NA:not_run:REF-K16 package file missing")
        su = self.push_setup(d, a, mode, extra)
        r, k = self.dm(lambda: su(1.0)), self.k16(lambda: su(1.0))
        return dict(damage=float(np.mean(r["per_text"] + k["per_text"])), r16=r["damage"], k16=k["damage"], top1_k16=k["top1_change"])


def gen_pairs(ctx, m, b, rows, d, a, cap, T=0.7, mode="add", extra=None, read_extra=None, F=None, read_ids=None):
    """One evaluation: rows (prompt_ids, key, sign) in pairs (+ at 2j, - at 2j + 1), generated with the push, read unsteered at block b
    (read_ids: the read prompts, default the generation prompts). Returns a dict in R.1's storage format (float32 states)."""
    t = torch(); rec = {}
    conts = K.generate(m.model, m.tok, rows, T=T, max_new=m.nnew, setup=pusher(m, b, d, a, cap, mode, rec, extra))
    fd = [first_div(conts[2 * j], conts[2 * j + 1]) for j in range(len(conts) // 2)]
    X, Pd, P20 = read_win(m, read_ids or [r[0] for r in rows], conts, b, fd, F, extra=read_extra)
    xp, xm = X[0::2], X[1::2]; ok = ~(t.isnan(xp).any(1) | t.isnan(xm).any(1)); cb = cbar_of(rec, conts)
    out = dict(xp=xp.float(), xm=xm.float(), ok=ok, first_div=fd, conts=conts, cbar=cb["pooled"], cbar_text=cb["per_text"],
               n_cbar_tokens=cb["n_tokens"], win_div=Pd.float(), win_first20=P20.float(), words=wcount(m.decode(conts)),
               wordsets=list(WSET_NAMES), a=float(a), T=float(T), mode=mode)
    if "disp" in rec:
        dd, cc = t.cat(rec["disp"]), t.cat(rec["cos"])
        out.update(disp_per_a=float(dd.mean()) if len(dd) else float("nan"), abs_cos=K.dist(cc.numpy()))
    else:
        out["disp_per_a"] = 1.0
    return out


def delta_of(ev):
    t = torch()
    return (ev["xp"].double() - ev["xm"].double()) / 2


def save_ev(ctx, rel, ev, config, labels=()):
    return ctx.save_pt(rel, ev, config=config, labels=list(labels))


def load_pt(p):
    return torch().load(p, map_location="cpu", weights_only=False)


def find(ctx, rel):
    """A file of this session or of any copied upstream session (in/<session>/<rel>)."""
    p = os.path.join(ctx.work, rel)
    if os.path.exists(p):
        return p
    hits = sorted(glob.glob(os.path.join(ctx.inp, "r1b_R_*", rel)))
    return hits[0] if hits else None


def safe(obj, code="NaN:not_finite"):
    """R9.2 for whole outputs: every non-finite float (also numpy / torch scalars) becomes coded(code); other values unchanged."""
    if isinstance(obj, dict):
        return {k: safe(v, code) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe(v, code) for v in obj]
    x = K.jsonable(obj)
    if isinstance(x, (list, dict)):
        return safe(x, code)
    if isinstance(x, float) and not math.isfinite(x):
        return K.coded(code)
    return x


def wj(ctx, rel, obj, **kw):
    """ctx.write_json with safe() applied (non-finite numbers coded instead of refused)."""
    return ctx.write_json(rel, safe(obj), **kw)
