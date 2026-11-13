"""r1b_R_S0 (part R; A100 runtime, no model): the SO-AX package (AX1, U*, leave-loop-out modes, SG10 rebuild check) and the A -> B
exports (REF-R16 token ids, u*_b), R0 (post-hoc re-analysis of the round-1 raw files: R0.1-R0.7), R8.1 (why the 0.6B arm was empty),
R9.5 (byte copies of round 2's Drive-only files). rk_loop is never imported; the tokenizer only decodes / encodes token ids."""
import glob, math, os, shutil, tempfile
import numpy as np
import r1b_core as K
import r1b_r_lib as R
import r1b_r_stats as S

TWINS = ("RSI_a", "RSI_b", "r2RSIb")
TOK_SHA = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"


def tokenizer(ctx):
    """The pinned Qwen3-1.7B tokenizer (tokenizer.json sha256 checked; downloaded alone at the pinned revision if the model is absent)."""
    from transformers import AutoTokenizer
    d = os.path.join(ctx.root, "models", "qwen3-1.7b")
    if not os.path.exists(os.path.join(d, "tokenizer.json")):
        from huggingface_hub import hf_hub_download
        d = os.path.join(ctx.root, "models", "qwen3-1.7b_tok")
        for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
            hf_hub_download("Qwen/Qwen3-1.7B", f, revision=K.MODELS["qwen3-1.7b"]["revision"], local_dir=d)
    if not ctx.mock and K.sha256_file(os.path.join(d, "tokenizer.json")) != TOK_SHA:
        raise RuntimeError("tokenizer.json sha256 differs from the pin")
    return AutoTokenizer.from_pretrained(d)


def suffix(ctx, b):
    return R.r1_loops(ctx, b)["_meta"].get("suffix", "")


def d_r1(ctx, b, L, n):
    """Round-1 direction d_n of loop L at block b (dir_<L><SUF>_<n>), or the round-2 loop r2RSIb (b<b>_r2RSIb.pt entries)."""
    if L == "r2RSIb":
        e = R.load_pt(R.r2_file(ctx, f"r2_q2/theta/b{b}_r2RSIb.pt"))["entries"]; return e[f"dir_r2RSIb_{n}"].double()
    return R.r1_state(ctx, b)[f"dir_{L}{suffix(ctx, b)}_{n}"].double()


def mode_members(ctx, b):
    ms = R.tv(ctx)["blocks"][str(b)]["B_Q0"]["mode_set"]; return list(ms["members"]), {k: float(v) for k, v in ms["signs"].items()}


def loo_modes(ctx, b):
    """u*_b^(-L) (R.1): unit of the sign-aligned sum of d_4 over M_b without L and its twins (RSI twins: RSI_a, RSI_b, r2RSIb)."""
    M, sg = mode_members(ctx, b); out = {}
    for L in M:
        drop = set(TWINS) if L in TWINS else {L}; rest = [x for x in M if x not in drop]
        out[L] = S.unit(sum(sg[x] * d_r1(ctx, b, x, 4) for x in rest)) if rest else K.coded("NaN:empty_leave_loop_out_set")
    return out


def soax(ctx):
    """SO-AX package: AX1 (round-1 axes), U* (the round-2 file, unchanged, with sha256), leave-loop-out modes and SG10's check (u*_b
    rebuilt from R1 dir_<L><SUF>_4 and the r2RSIb entries with TV's mode set: cos >= 1 - 1e-4; TV image_cos_d1_u_star within 1e-4)."""
    t = R.torch(); chk, ust, src = dict(blocks={}), {}, {}
    for b in R.BLOCKS:
        ax = R.r1_state(ctx, b)["axes"]; p = R.r2_file(ctx, f"r2_q2/theta/b{b}_q0.pt"); u = R.load_pt(p)["u_star"]
        src[b] = dict(path=f"r2_q2/theta/b{b}_q0.pt", sha256=K.sha256_file(p)); ust[b] = u.double()
        M, sg = mode_members(ctx, b); reb = S.unit(sum(sg[L] * d_r1(ctx, b, L, 4) for L in M)); cr = S.cos(reb, u)
        img_tv = R.tv(ctx)["blocks"][str(b)]["B_Q0"].get("image_cos_d1_u_star", {}); img = {}
        for L, v in img_tv.items():
            try:
                img[L] = dict(rebuilt=S.cos(d_r1(ctx, b, L, 1), u), tv=v)
            except (KeyError, K.NotRun):
                img[L] = dict(rebuilt=K.coded("NaN:direction_missing"), tv=v)
        dif = max([abs(x["rebuilt"] - x["tv"]) for x in img.values() if isinstance(x["rebuilt"], float)] or [float("inf")])
        ok = cr >= 1 - 1e-4 and dif <= 1e-4
        chk["blocks"][str(b)] = dict(members=M, signs=sg, cos_rebuild=cr, image_cos_d1_u_star=img, diff=K.val(dif, "NaN:no_image_cos"), ok=ok)
        loo = loo_modes(ctx, b); lab = None if ok else "SO-AX reproduction failed"
        ctx.save_pt(f"so_ax/so_ax_b{b}.pt", dict(u_T_perp=ax["u_T_perp"].double(), u_sea=ax["sea"].double(), u_sports=ax["sports"].double(),
                                                 u_tone=ax["tone"].double(), u_neg=ax["u_neg"].double(), u_star=u.double(), u_star_src=src[b],
                                                 u_star_minus={k: v for k, v in loo.items() if not isinstance(v, dict)}, label=lab,
                                                 members=M, signs=sg), config=dict(block=b), labels=[lab] if lab else [])
    chk["ok"] = all(v["ok"] for v in chk["blocks"].values()); chk["label"] = None if chk["ok"] else "SO-AX reproduction failed"
    R.wj(ctx, "so_ax/so_ax_check.json", chk, config=dict(tol_cos=1e-4, tol_image=1e-4), labels=[chk["label"]] if chk["label"] else [])
    ctx.save_pt("xfer/so_ax_ustar.pt", dict(u_star=ust, src=src), config=dict(blocks=list(R.BLOCKS)))


def ref16(ctx):
    """REF-R16 export (SO-DMG): token ids of R1.state[14]['ref'] continuations of round-1 prompts 0-7 and 32-39, with rk_loop's prompt
    ids; equality with the copies at 18 and 6 logged."""
    tok = tokenizer(ctx); pr = R.r1_prompts(ctx.mock); P = list(range(8)) if ctx.mock else list(range(8)) + list(range(32, 40))
    st = R.r1_state(ctx, 14); items = []
    for p in P:
        items.append(dict(p=p, prompt_ids=K.chat_ids(tok, pr[p]), cont_ids=list(map(int, st["ref"]["conts"][p]))))
    eq = {}
    for b in (18, 6):
        c = R.r1_state(ctx, b)["ref"]["conts"]; diffs = [p for p in P if list(map(int, c[p])) != list(map(int, st["ref"]["conts"][p]))]
        eq[str(b)] = dict(equal=not diffs, diffs=diffs)
    R.wj(ctx, "xfer/dmg_ref_R16.json", dict(items=items, source=dict(path=os.path.relpath(R.r1_file(ctx, 14, "rk/b{b}/state.pt"), ctx.inp),
                                                                              sha256=K.sha256_file(R.r1_file(ctx, 14, "rk/b{b}/state.pt"))),
                                                 equal_at_18_6=eq), config=dict(prompts=P))


# ------------------------------------------------------------------------------------------------------------------ R0 helpers
class R1Block:
    """Round-1 block data for R0: axis texts' per-prompt differences, fold, AxisSets for u* variants, prompt bootstrap W."""

    def __init__(self, ctx, b, B=2000):
        t = R.torch(); self.b, self.st = b, R.r1_state(ctx, b); self.L = R.r1_loops(ctx, b); self.meta = self.L.pop("_meta"); ax = self.st["axes"]
        self.suf = self.meta.get("suffix", ""); self.fold = ax["fold"].bool(); self.NP = len(self.fold); self.H = S.halvings(self.NP)
        self.diff, self.keep = {}, {}
        for n in R.TOPICS:
            Xp, Xa = self.st[f"axis_{n}"]["Xp"].double(), self.st[f"axis_{n}"]["Xa"].double()
            self.keep[n] = ~(t.isnan(Xp).any(1) | t.isnan(Xa).any(1)); self.diff[n] = t.nan_to_num(Xp - Xa)
        self.us, _ = R.ustar(ctx, b); self.loo = loo_modes(ctx, b); self.un = ax["u_neg"].double(); self.dev = "cuda" if t.cuda.is_available() else "cpu"
        self.W = R.torch().as_tensor(K.prompt_bootstrap(self.NP, B=B, seed=1001), dtype=t.float64)

    def axset(self, ustar):
        return S.AxisSet(self.diff, self.keep, self.fold, ustar, self.un, self.b)

    def spec(self, X):
        """Readout axes of R0: name -> (AxisSet, axis name, null topic, label)."""
        sea_u = self.loo.get("SEA") if self.b == 14 and "SEA" in self.loo and not isinstance(self.loo.get("SEA"), dict) else self.us
        rsi_u = self.loo.get("RSI_a") if not isinstance(self.loo.get("RSI_a"), dict) and "RSI_a" in self.loo else self.us
        return {"u_T_perp": (self.us, "T_1b", "rsi"), "sea": (self.us, "sea_raw", "sea"), "sports": (self.us, "sports_raw", "sports"),
                "tone": (self.us, "tone_raw", "tone"), "sea_v": (sea_u, "sea_v", "sea"), "T_ps_loo": (rsi_u, "T_ps", "rsi"),
                "T_ps_full": (self.us, "T_ps", "rsi")}[X]

    def ev(self, L, n):
        return self.st.get(f"ev_{L}{self.suf}_{n}") if not L.endswith("@1.5") else self.st.get(f"ev_{L}_{n}")


def row_stats(blk, ev, X, axc, boot=True):
    """One round-1 evaluation on readout X: stored-axis c and round-1 p (r2_rk.stats_on_axis), c^cf, A (disattenuated) with the joint
    prompt-bootstrap interval, K1-s (+ Wilcoxon, t), K1-c (p_c, Q_c)."""
    import r2_rk
    t = R.torch(); us, nm, top = blk.spec(X); key = (X,)
    if key not in axc:
        AS = blk.axset(us); axc[key] = (AS.get(t.ones(1, blk.NP, dtype=t.float64), [nm])[nm], AS.get(blk.W, [nm])[nm] if boot else None)
        Q = None if nm.endswith("raw") else _basis(AS, nm)
        axc[key + ("null",)] = S.centred_nulls(blk.diff[top], blk.keep[top], R.NULL_SEED[top] - 300, Q)        # round-1 seeds 11-14
    if boot and axc[key][1] is None:
        axc[key] = (axc[key][0], blk.axset(us).get(blk.W, [nm])[nm])
    a1, aW = axc[key]; row = S.Row(ev["delta"], ev["ok"], ev["prompt_of"], blk.NP)
    rd = S.readout(row, a1, blk.fold, blk.H); out = dict(c_cf=rd["c_cf"], c=rd["c"], A=rd["A"], rel=rd["rel"], kappa=rd["kappa"])
    stored = {"u_T_perp": "u_T_perp", "sea": "sea", "sports": "sports", "tone": "tone"}.get(X)
    if stored:
        s1, _ = r2_rk.stats_on_axis(ev, blk.st["axes"][stored], blk.st["axes"][f"null_{stored}"], n_prompts=blk.NP)
        out.update(p_r1=K.val(s1["p"]), Q_r1=K.val(s1["Q"]), c_r1=K.val(s1["c"]), rel_r1=K.val(s1["rel"]))
    out.update(S.k1c(row.d(), a1["u"][0], axc[key + ("null",)]))
    k = S.k1s(S.bvals(row, a1, blk.fold)); out.update(p_s=k["p_s"], p_wilcoxon=k["p_wilcoxon"], p_t=k["p_t"])
    if boot:
        rb = S.readout(row, aW, blk.fold, blk.H, W=blk.W); lo, hi = S.iv(rb["A"].cpu().numpy()); clo, chi = S.iv(rb["c_cf"].cpu().numpy())
        out.update(A_lo=lo, A_hi=hi, c_cf_lo=clo, c_cf_hi=chi, A_draws=rb["A"].cpu().numpy())
    return out


def _basis(AS, nm):
    t = R.torch(); X, var = nm.rsplit("_", 1); W1 = t.ones(1, len(AS.fold), dtype=t.float64); raw = {x: AS._mean(W1, AS.XT[x])[0] for x in AS.XT}
    cols = [raw[o] for o in AS.XT if o != X] + [AS.un] + ([AS.us] if var == "ps" or (var == "v" and AS.block != 6) else [])
    Q, _ = t.linalg.qr(t.stack([S.unit(c) for c in cols], -1)); return Q


def variants(s, A_lo=None):
    """R0.5's five statistic variants on one row: R1 (round-1 p), C (p_c <= 0.01, c > 0), S (p_s <= 0.01), SC (and Q_c >= 1), M (p_s <=
    0.01 and A's lower bound >= 0.10)."""
    num = lambda x: x if isinstance(x, (int, float)) else float("nan")
    return dict(R1=num(s.get("p_r1")) <= 0.01 and num(s.get("c_r1", s["c"])) > 0, C=num(s["p_c"]) <= 0.01 and s["c"] > 0, S=num(s["p_s"]) <= 0.01,
                SC=num(s["p_s"]) <= 0.01 and num(s["Q_c"]) >= 1, M=num(s["p_s"]) <= 0.01 and num(s.get("A_lo")) >= 0.10)


def r0_block(ctx):
    """R0.1-R0.6 at block b (all strings 'post hoc (R0)')."""
    t = R.torch(); b = int(ctx.step.rsplit("_b", 1)[1]); blk = R1Block(ctx, b); axc = {}; out = dict(block=b, NP=blk.NP, suffix=blk.suf, rows={})
    # R0.1 reproduction and the null rebuild
    rep, rebuild = [], {}
    for i, n in enumerate(R.TOPICS):
        g = t.Generator().manual_seed(11 + i); Dk = blk.diff[n][blk.keep[n]]
        E = (t.randint(0, 2, (2000, len(Dk)), generator=g) * 2 - 1).double(); nl = E @ Dk / len(Dk)
        if n == "rsi":
            F = t.stack([S.unit(blk.st["axes"][k].double()) for k in ("sports", "tone", "sea", "u_neg")]).T; Q, _ = t.linalg.qr(F); nl = nl - (nl @ Q) @ Q.T
        st = blk.st["axes"]["null_u_T_perp" if n == "rsi" else f"null_{n}"].double()
        rebuild[n] = dict(max_abs_diff=float((nl - st).abs().max()), tol=1e-5 * float(st.abs().max()), ok=float((nl - st).abs().max()) <= 1e-5 * float(st.abs().max()))
    for L, rows in blk.L.items():
        for n, r in enumerate(rows):
            ev = blk.ev(L, n)
            if ev is None:
                continue
            s = row_stats(blk, ev, "u_T_perp", axc); sd = {}
            for k, tol in (("c", 1e-6), ("Q", 1e-5), ("rel", 1e-6)):
                sk = s.get(f"{k}_r1"); sd[k] = abs(sk - r[k]) <= tol if isinstance(sk, float) and isinstance(r.get(k), (int, float)) else None
            sd["p"] = s.get("p_r1") == r.get("p"); rep.append(dict(loop=L, n=n, ok=all(v is not False for v in sd.values()), checks=sd))
            rw = dict(u_T_perp={k: v for k, v in s.items() if k != "A_draws"})
            for X in (("sea", "sea_v") if L.startswith("SEA") else ()) + (("sports",) if L.startswith("SPO") else ()) + \
                    (("tone",) if L.startswith("RND") else ()) + (("T_ps_loo", "T_ps_full") if L.startswith("RSI") else ()):
                rw[X] = {k: v for k, v in row_stats(blk, ev, X, axc).items() if k != "A_draws"}
            out["rows"][f"{L}:{n}"] = rw
    out["R0.1"] = dict(rows_reproduced=sum(r["ok"] for r in rep), rows=len(rep), flag=None if all(r["ok"] for r in rep) else "R0 reproduction differs",
                       details=[r for r in rep if not r["ok"]][:20], null_rebuild=rebuild)
    # R0.4 cross-fitting
    def row_of(L, n):
        e = blk.ev(L, n); return None if e is None else S.Row(e["delta"], e["ok"], e["prompt_of"], blk.NP)
    cf = {}
    if row_of("SEA", 3) is not None:
        cf["S4cf_SEA"] = S.s4cf(row_of("SEA", 2), row_of("SEA", 3), blk.H)
    twin = "RSI_b" if b == 14 else "r2RSIb"
    if twin == "RSI_b" and row_of("RSI_b", 3) is not None:
        cf["R4cf_RSI"] = S.cf_cos(row_of("RSI_a", 3), [row_of("RSI_b", 3)], blk.H)
    elif twin == "r2RSIb":
        p = R.r2_file(ctx, f"r2_q2/theta/b{b}_r2RSIb.pt", required=False); e = R.load_pt(p)["entries"].get("ev_r2RSIb_3") if p else None
        cf["R4cf_RSI"] = S.cf_cos(row_of("RSI_a", 3), [S.Row(e["delta"], e["ok"], e["prompt_of"], blk.NP)], blk.H) if e else K.coded("NA:not_run:r2RSIb evaluations missing")
    cf["mode_cos_cf"] = {L: S.cf_cos(row_of(L, 3), [row_of("RSI_a", 3)], blk.H) for L in blk.L if row_of(L, 3) is not None}
    out["R0.4"] = cf
    # R0.5 gate re-evaluation (GO1_R0, PC_PASS_R0, MANIP_R0)
    g = {}
    for X in ("sea", "sea_v"):
        for scale, L in (("alpha", "SEA"), ("1.5 alpha", "SEA@1.5")):
            ev0 = blk.st.get(f"ev_{L}_0") if L.endswith("@1.5") else blk.st.get(f"ev_SEA_0")
            if ev0 is None:
                continue
            s0 = row_stats(blk, ev0, X, axc); v = variants(s0); g[f"GO1_R0|{X}|{scale}"] = {V: bool(v[V] and s0["rel"] >= 0.5) for V in v}
        sr = [row_stats(blk, blk.ev("SEA", n), X, axc) for n in (1, 2, 3) if blk.ev("SEA", n) is not None]
        s4 = cf.get("S4cf_SEA", {}).get("cf") if isinstance(cf.get("S4cf_SEA"), dict) else None
        if len(sr) == 3:
            pcv = {V: all(variants(s)[V] and s["rel"] >= 0.5 for s in sr) and isinstance(s4, float) and s4 >= 0.5 for V in ("C", "S", "SC", "M")}
            pcv["R1"] = bool(_r1_pc(blk)) if X == "sea" else None; g[f"PC_PASS_R0|{X}"] = pcv
    man = {}
    for X in ("u_T_perp",):
        rs = [row_stats(blk, blk.ev(L, 0), X, axc) for L in ("RSI_a", "RSI_b") if blk.ev(L, 0) is not None]
        man = {V: all(variants(s)[V] for s in rs) for V in ("R1", "C", "S", "SC", "M")} if rs else {}
    out["R0.5"] = dict(gates=g, MANIP_R0=man, would_have_read=_would_read(ctx, blk, axc))
    # R0.6 topic specificity and the expectation for R2.3
    tok = tokenizer(ctx); us = blk.us; AS = blk.axset(us); W1 = t.ones(1, blk.NP, dtype=t.float64)
    sea_perp = AS.get(W1, ["sea_1b"])["sea_1b"]["u"][0]; seaps = AS.get(W1, ["sea_ps"])["sea_ps"]["u"][0]
    loo = blk.loo.get("SEA"); r6 = dict(cos_ustar_sea_perp=S.cos(us, sea_perp), cos_ustar_sea_perpstar=S.cos(us, seaps),
                                        cos_ustar_minusSEA_sea_perpstar=S.cos(loo, seaps) if isinstance(loo, t.Tensor) else K.coded("NA:not_run:SEA not in M_b"),
                                        cos_ustar_mean_ref_X=S.cos(us, blk.st["ref"]["X"].double().nanmean(0)))
    sea_rows = {n: row_stats(blk, blk.ev("SEA", n), "sea_v", axc) for n in range(4) if blk.ev("SEA", n) is not None}
    rnd4 = [abs(row_stats(blk, blk.ev(L, 3), "sea_v", axc)["A_draws"]) for L in ("RND_1", "RND_2") if blk.ev(L, 3) is not None]
    if 3 in sea_rows and rnd4:
        dd = sea_rows[3]["A_draws"] - np.max(np.stack(rnd4), 0); r6["SEA_minus_maxRND_A4_sea_v"] = dict(lo_hi=S.iv(dd))
    r6["SEA_A_sea_v"] = {n: dict(A=s["A"], A_lo=s.get("A_lo"), A_hi=s.get("A_hi")) for n, s in sea_rows.items()}
    lex = {}
    for L in blk.L:
        if not (L.startswith("SEA") or L.startswith("RND")):
            continue
        per = np.zeros(blk.NP); cnt = np.zeros(blk.NP)
        for n in (1, 2, 3):
            e = blk.ev(L, n)
            if e is None:
                continue
            w = R.wcount([tok.decode(c) for c in e["conts"]], {"W_sea": R.W_SEA})[:, 0]; dlt = w[0::2] - w[1::2]
            np.add.at(per, np.asarray(e["prompt_of"]), dlt); np.add.at(cnt, np.asarray(e["prompt_of"]), 1)
        m = per / np.maximum(cnt, 1); lex[L] = dict(mean=float(m.mean()), sd_p=float(m.std(ddof=1)), mde_P4=(2.326 + 0.842) * float(m.std(ddof=1)) / 16)
    r6["lexical_sea"] = lex
    tvb = R.tv(ctx)["blocks"][str(b)]; r6["round2"] = dict(mode_set=tvb["B_Q0"]["mode_set"], image_cos_d1_u_star=tvb["B_Q0"].get("image_cos_d1_u_star"),
                                                           R4=(tvb.get("R4") or {}).get("label") or (tvb.get("R4") or {}).get("verdict"))
    out["R0.6"] = r6
    out["strings"] = _r0_strings(b, g, blk, axc, cf)
    out["labels"] = ["post hoc (R0)"]
    R.wj(ctx, f"r0/r0_b{b}.json", out, config=dict(block=b, B=2000, seed=1001, flips=10000), labels=["post hoc (R0)"])


def _r1_pc(blk):
    import rk_verdict
    return rk_verdict.pc_pass(blk.st, blk.L, blk.suf)[0]


def _would_read(ctx, blk, axc):
    """rk_verdict.block_verdict on a copy of the block whose loops.json p values are replaced per variant (descriptive, round-1 axes)."""
    import rk_verdict
    src = os.path.dirname(R.r1_file(ctx, blk.b, "rk/b{b}/loops.json")); out = {}
    for V in ("R1", "C", "S", "SC", "M"):
        d = tempfile.mkdtemp(); L = R.r1_loops(ctx, blk.b)
        for f in glob.glob(os.path.join(src, "*.json")):
            shutil.copy(f, d)
        os.symlink(os.path.join(src, "state.pt"), os.path.join(d, "state.pt"))
        if V != "R1":
            for Ln, rows in L.items():
                if Ln == "_meta":
                    continue
                for n, r in enumerate(rows):
                    ev = blk.ev(Ln, n)
                    if ev is None:
                        continue
                    s = row_stats(blk, ev, "u_T_perp", axc); ok = variants(s)[V]; r["p"] = 0.001 if ok else 1.0
                    if V == "C":
                        r["c"] = s["c"]
            K.dump_json(L, os.path.join(d, "loops.json"))
        try:
            v = rk_verdict.block_verdict(d); out[V] = v.get("verdict") if isinstance(v, dict) else str(v)
        except Exception as e:                                   # a missing round-1 piece (MOCK) is reported, not fatal
            out[V] = K.coded("NaN:block_verdict_failed"); out[V + "_error"] = str(e)[:200]
        shutil.rmtree(d, ignore_errors=True)
    return out


def _r0_strings(b, g, blk, axc, cf):
    s = []
    for k, v in g.items():
        for V, ok in v.items():
            if ok is not None:
                s.append(f"{k.split('|')[0]} ({V}, {k.split('|', 1)[1]}): {'pass' if ok else 'fail'} (post hoc (R0))")
    if b == 6:
        go = (g.get("GO1_R0|sea|alpha") or {}).get("SC"); pc = (g.get("PC_PASS_R0|sea") or {}).get("SC")
        if go and pc:
            s.append("block 6 reopened (post hoc, rule SC on raw u_sea)")
        else:
            s.append("block 6 stays excluded (post hoc)")
    elif b == 14:
        s.append("block 14 stays excluded: GO3 (damage 0.4055 / 0.2018 nats) is not a null problem")
    else:
        s.append("block 18 stays excluded: PC fails on rel (0.4376, 0.4877 < 0.5), which no null correction changes")
    s.append("delta rule: not decidable (at most one block; post hoc)")
    return s


def r07(ctx):
    """R0.7: size of K1-s (centred b_p resampled, 2,000 x 2,000 flips; placebo u_neg for RND rows) -> RG6; power of the rules R1, C, S,
    SC, M on planted signals (RND_2 at 14, RND_1 at 18 / 6; u_T_perp and raw u_sea; c 0.05..0.40; 500 replicates; M's inner interval from
    200 replicates with axes and rel fixed per outer replicate, a cost reduction); the power model's correlations and joint MDEs."""
    t = R.torch(); size_rows, power, corr = [], {}, {}
    nd = 200 if ctx.mock else 2000
    for b in R.BLOCKS:
        blk = R1Block(ctx, b); axc = {}
        for L, rows in blk.L.items():
            for n in range(len(rows)):
                ev = blk.ev(L, n)
                if ev is None:
                    continue
                for X in (("u_T_perp", "sea", "sports", "tone") + (("u_neg",) if L.startswith("RND") else ())):
                    if X == "u_neg":
                        a1 = dict(aA=S.unit(blk.un)[None], aB=S.unit(blk.un)[None])
                    else:
                        row_stats(blk, ev, X, axc, boot=False); a1 = axc[(X,)][0]
                    bp = S.bvals(S.Row(ev["delta"], ev["ok"], ev["prompt_of"], blk.NP), a1, blk.fold); bc = bp - bp.mean(); g = np.random.default_rng(7)
                    eps = np.random.default_rng(2001).integers(0, 2, (nd, len(bc))) * 2 - 1; rej = 0
                    for _ in range(nd):
                        x = g.choice(bc, len(bc)); nl = eps @ x / len(x); rej += ((1 + (nl >= x.mean()).sum()) / (nd + 1)) <= 0.01
                    size_rows.append(dict(block=b, loop=L, n=n, axis=X, rejection=rej / nd))
        Lp = "RND_2" if b == 14 else "RND_1"; ev = blk.ev(Lp, 0)
        if ev is not None:
            for X in ("u_T_perp", "sea"):
                row_stats(blk, ev, X, axc, boot=False); a1 = axc[(X,)][0]; u = S.unit(a1["u"][0]); dl = R.torch().nan_to_num(ev["delta"].double()); ok = ev["ok"]
                d = dl[ok].mean(0); dpar, dperp = float(u @ d), float((d - (u @ d) * u).norm()); res = {}
                null1 = blk.st["axes"]["null_u_T_perp" if X == "u_T_perp" else "null_sea"].double(); nullc = axc[(X, "null")]
                for c in np.arange(0.05, 0.401, 0.05):
                    eta = c * dperp / math.sqrt(1 - c * c) - dpar; rr = S.Row(dl + eta * u[None], ok, ev["prompt_of"], blk.NP)
                    bp = np.zeros(blk.NP); bp[(rr.n > 0).numpy()] = S.bvals(rr, a1, blk.fold); det = {k: 0 for k in ("R1", "C", "S", "SC", "M")}
                    nrep = 50 if ctx.mock else 500; g = np.random.default_rng(int(c * 1000))
                    for _ in range(nrep):
                        pick = np.concatenate([g.integers(0, blk.NP // 2, blk.NP // 2), blk.NP // 2 + g.integers(0, blk.NP - blk.NP // 2, blk.NP - blk.NP // 2)])
                        w = t.as_tensor(np.bincount(pick, minlength=blk.NP), dtype=t.float64)[None]; dW = rr.d(w)[0]
                        r1 = S.readout(rr, a1, blk.fold, blk.H, W=w); cc = float(r1["c"][0])
                        nc = (null1 @ dW).abs() / (null1.norm(dim=1) * dW.norm() + 1e-12)
                        s = dict(c=cc, c_r1=cc, p_r1=float((1 + (nc >= abs(cc)).sum()) / (1 + len(nc))), **{k: v for k, v in S.k1c(dW, a1["u"][0], nullc).items() if k != "c"})
                        s.update(S.k1s(np.repeat(bp, np.bincount(pick, minlength=blk.NP)[: blk.NP])) if True else {})
                        Wi = t.as_tensor(np.stack([np.bincount(pick[g.integers(0, len(pick), len(pick))], minlength=blk.NP) for _ in range(200)]), dtype=t.float64)
                        s["A_lo"] = S.iv(S.readout(rr, {k: (v.expand(200, -1) if v.dim() == 2 else v.expand(200)) for k, v in a1.items()},
                                                   blk.fold, blk.H, W=Wi, rel=r1["rel"])["A"].numpy())[0]
                        for k, v in variants(s).items():
                            det[k] += bool(v)
                    res[f"{c:.2f}"] = {k: v / nrep for k, v in det.items()}
                mde = {k: next((float(c) for c, r in res.items() if r[k] >= 0.8), K.coded("CENSORED:above_grid")) for k in ("R1", "C", "S", "SC", "M")}
                power[f"b{b}:{Lp}:{X}"] = dict(detection=res, MDE_80=mde)
        rA = [row_stats(blk, blk.ev(L, n), "u_T_perp", axc)["A_draws"] for L in ("RSI_a", "RSI_b") for n in range(4) if blk.ev(L, n) is not None]
        if len(rA) == 8:
            Cm = np.corrcoef(np.stack(rA)); corr[str(b)] = dict(within=float(np.mean([Cm[i, j] for l in (0, 4) for i in range(l, l + 4) for j in range(i + 1, l + 4)])),
                                                              between=float(np.mean(Cm[:4, 4:])), sd_A=float(np.mean([np.std(a) for a in rA])))
    share = np.mean([r["rejection"] <= 0.02 for r in size_rows]) if size_rows else float("nan")
    rg6 = "K1-s size ok" if share >= 0.9 else "K1-s size inflated: bootstrap-t used in part R"
    mdes = {}
    for b, c in corr.items():
        sdA = {"14": 0.045, "18": 0.033, "6": 0.034}[b]
        mdes[b] = [S.joint_mde(sdA, min(max(c["within"], 0.0), 0.95), min(max(c["between"], 0.0), 0.95), r) for r in ("substantial", "dominated", "unlikely")]
    R.wj(ctx, "r0/r07.json", dict(size=dict(rows=size_rows, share_ok=K.val(share), RG6=rg6), power=power, power_model=dict(correlations=corr, joint_mde=mdes,
                                                                                                                            note="calibration only; changes no rule"),
                                        labels=["post hoc (R0)"]), config=dict(draws=nd, flips=nd, seed_flip=2001), labels=["post hoc (R0)"])


def r0_table(ctx):
    """r0_table.json: R0 per block, R0.7, RG6, the README null-claim count (rows moved across 0.01 by centring, both directions)."""
    blocks = {str(b): K.load_json(ctx.path(f"r0/r0_b{b}.json")) for b in R.BLOCKS}; r07 = K.load_json(ctx.path("r0/r07.json"))
    up, down = [], []
    for b, d in blocks.items():
        for k, rw in d["rows"].items():
            s = rw["u_T_perp"]; p1, pc = s.get("p_r1"), s.get("p_c")
            if isinstance(p1, float) and isinstance(pc, float):
                if p1 > 0.01 and pc <= 0.01:
                    up.append(dict(block=b, row=k, q99_ratio=K.val(s["Q_c"] / s["Q_r1"] if isinstance(s.get("Q_r1"), float) and s["Q_r1"] else None)))
                if p1 <= 0.01 and pc > 0.01:
                    down.append(dict(block=b, row=k))
    claim = "README null claim confirmed" if up else "README null claim not confirmed"
    for d in blocks.values():
        d.pop("provenance", None)
    R.wj(ctx, "r0_table.json", dict(blocks=blocks, R0_7={k: v for k, v in r07.items() if k != "provenance"}, RG6=r07["size"]["RG6"],
                                         bootstrap_t=r07["size"]["RG6"].startswith("K1-s size inflated"),
                                         readme_null_claim=dict(string=claim + " (post hoc (R0))", p_to_pc_below=up, pc_above=down),
                                         n_verdicts={b: len(d["strings"]) for b, d in blocks.items()}), config=dict(), labels=["post hoc (R0)"])


# ------------------------------------------------------------------------------------------------------------------ R8.1, R9.5
def r81(ctx):
    """R8.1: per block of the 0.6B geometry gate 'C0 fails (<parts>)' | 'uncomposed only (FG(CHG-DES) = x, FG(WOR-DES) = y)' | 'passes' |
    'NaN (<keys>)'; arm verdict from runs of >= 12 contiguous blocks; descriptive run lengths at uncomposed cuts 0.85 / 0.90."""
    p = R.r2_file(ctx, "r2_q3/rgi/geom_stats_qwen3-0.6b.json"); d = K.load_json(p) or {}; g = d.get("gate") or {}
    pooled = (d.get("per_block") or {}).get("pooled") or {}
    NL = len(g.get("C0") or []) or 28

    def ser(k):
        v = pooled.get(k) or {}; pt = v.get("point") if isinstance(v, dict) else None
        return [float(x) if isinstance(x, (int, float)) and math.isfinite(x) else float("nan") for x in (pt or [float("nan")] * NL)]
    fgc, fgw = ser("FG|CHG-DES"), ser("FG|WOR-DES"); C0 = list(g.get("C0") or [None] * NL); parts = g.get("C0_parts") or {}
    unc = list(g.get("uncomposed") or [None] * NL); per = []
    for l in range(NL):
        bad = [k for k in parts if not parts[k][l]]
        if C0[l] is None or not math.isfinite(fgc[l]) or not math.isfinite(fgw[l]):
            per.append(dict(block=l, string=f"NaN ({[k for k, v in (('C0', C0[l]), ('FG|CHG-DES', fgc[l]), ('FG|WOR-DES', fgw[l])) if v is None or (isinstance(v, float) and not math.isfinite(v))]})"))
        elif not C0[l]:
            per.append(dict(block=l, string=f"C0 fails ({', '.join(bad)})", parts=bad))
        elif unc[l]:
            per.append(dict(block=l, string=f"uncomposed only (FG(CHG-DES) = {fgc[l]:.3f}, FG(WOR-DES) = {fgw[l]:.3f})"))
        else:
            per.append(dict(block=l, string="passes"))

    def runs(mask):
        best = cur = 0
        for m in mask:
            cur = cur + 1 if m else 0; best = max(best, cur)
        return best
    c0u = runs([bool(C0[l]) and bool(unc[l]) for l in range(NL)]); c0 = runs([bool(C0[l]) for l in range(NL)])
    nan = any(x["string"].startswith("NaN") for x in per)
    if c0u >= 12:
        arm = "empty because of the 0.8 uncomposed cut"
    elif c0 < 12 and not nan:
        cnt = {}
        for x in per:
            for k in x.get("parts", []):
                cnt[k] = cnt.get(k, 0) + 1
        arm = f"empty because C0 fails ({max(cnt, key=cnt.get) if cnt else 'unknown'})"
    elif nan:
        arm = "empty because of NaN values"
    else:
        arm = "mixed (" + "; ".join(f"{x['block']}: {x['string']}" for x in per) + ")"
    cuts = {str(c): runs([bool(C0[l]) and not (fgc[l] > c and fgw[l] > c) for l in range(NL)]) for c in (0.85, 0.90)}
    j1 = not nan
    R.wj(ctx, "r8_1.json", dict(per_block=per, arm=arm, J1=j1, run_C0_uncomposed=c0u, run_C0=c0, runs_passing_at_cut=cuts,
                                     source=dict(path="r2_q3/rgi/geom_stats_qwen3-0.6b.json", sha256=K.sha256_file(p)), labels=["descriptive", "post hoc"]),
                   config=dict(run=12, cut=0.8), labels=["descriptive"])


def r95(ctx):
    """R9.5: byte-identical copies of round 2's Drive-only unlearn_verdict.json and r2_manifest.json (with sha256); the coordinator commits
    them to results/ju_space/runde2_ergebnisse/."""
    out = {}
    for f in ("unlearn_verdict.json", "r2_manifest.json"):
        p = R.r2_file(ctx, f"r2_ana/{f}", required=False)
        if p:
            ctx.write_file(f"r95/{f}.orig", open(p, "rb").read(), config=dict(source=f"r2_ana/{f}")); out[f] = dict(sha256=K.sha256_file(p), copied=True,
                                                                                                            file=f"r95/{f}.orig")
        else:
            out[f] = K.coded("NA:not_run:input missing")
    R.wj(ctx, "r95/r95.json", dict(files=out, target="results/ju_space/runde2_ergebnisse/"), config=dict())
