"""Round 2, O2 and O1's fixed axes (final plan, sections 2.4 and 9.2): the RGI axis family at every block of B3, the statement and NEG
re-reads with their nulls, the per-token block-14 dump and the match rates. Runs in the block-14 process of session r2_q1 (before
r2_gen_loop.py). G4 (block 14 included): a block whose round-1 outputs are not final (no summary_b<b>.json) is 'not run (round-1 output
missing)'; the texts and reads then run in the process of the first final block of B3 (the texts are unsteered and block-independent),
and the parts that read R1.state[14] (R1 texts in tok14.pt, match rate) are 'not run' if block 14 is not final.
O1 (critique v2-4): per block, the fixed axes a_A = unit(perp(d_A^0)), a_B = unit(perp(d_B^0)) of this session's unpushed, unablated
RSI_PRO / RSI_ANTI texts (perp off span(sports, tone, sea, u_neg)) and RGI_PRO / RGI_ANTI texts (perp off the same plus u_TOPIC), with
g_0 = their gap on these axes (r2_rk.gap_on_fixed_axes; = rk_loop's gap_perp_cross; RGI: = Delta_RGI); C-Q2 / C-Q3 / C-Q4a read
g_push, g_abl and g_cue on the block-14 axes.

Texts (stage 'texts'): six instructions (r2_stimuli.O2_INSTR: RSI_PRO / RSI_ANTI = rk_loop.INSTR['rsi'], RGI_PRO, RGI_ANTI, TOPIC_PRO,
TOPIC_ANTI) appended to each of the 64 prompts as rk_loop.text_axis does, generated unsteered once (T 0.7, 40 tokens, Gumbel-max) with
keys keyseed('axis', 'rsi', p) for all six (stated CRN exception, 2.3 / Appendix D; MOCK: keyseed('axis', 'rsi', p, s) with s = +1 for
the pro and -1 for the anti sets, rk_loop's own MOCK rule, so that the MOCK RSI texts reproduce the MOCK round-1 texts).
Reads (stage 'read'): the 384 texts under the plain prompt at all 28 blocks (read_all_blocks; one unstopped forward); the 50 + 50
statements selected exactly as rk_loop.axes() (random.Random(0) shuffles of the Yes and No items of rsi_statements) and the 40 NEG
pairs, read as replies to 'Say one sentence.' at the blocks of B3; per-token block-14 states (float16) of R1 RSI pro / anti and R1 sea
pro / anti (R1.state[14] 'axis_rsi' / 'axis_sea' conts) and of this session's RGI_PRO / RGI_ANTI (axes/tok14.pt, for C-Q1).
Statistics (stage 'stats', no model) per block b of B3 (fold = R1.state[b]['axes']['fold']; per-prompt differences pro - anti):
  u_RGI = mean_p Delta_RGI,p; u_TOPIC = mean_p Delta_TOPIC,p; u_GI^TOP = mean_p (x_RGI_PRO,p - x_TOPIC_PRO,p);
  u_RGI_perp = u_RGI minus its least-squares projection on span(sports, tone, sea, u_neg (R1 axes of b), u_TOPIC); 2,000 within-prompt
    sign-flip nulls of Delta_RGI (seed 21) projected the same way; Delta_RGI = gap_perp_cross of Delta_RGI (rk_loop's estimator);
  u_RGI_perpT = u_RGI_perp orthogonalised against u_T_perp (R1), nulls the same way;
  u_dir = mean_p (x_RSI_PRO,p - x_RGI_PRO,p) (session texts, shared keys), cross-fitted centred AUC of RSI_PRO against RGI_PRO, nulls by
    within-prompt sign flips (seed 22); u_diff / u_com = unit(unit(u_T_perp) -/+ unit(u_RGI_perp));
  cross-fitted centred AUC and gap of every axis (session RSI, RGI, TOPIC, GI^TOP, dir); split-half rel over the 20 K2 halvings;
  disattenuated cos(u_T_perp, u_RGI_perp) / sqrt(rel_T rel_RGI) with a 90 % prompt-bootstrap interval (2,000 replicates, seed BOOT_SEED,
  the same draws at every block); norms of the common part mean_p (a_p + b_p) / 2 and of the difference part mean_p (a_p - b_p) / 2 with
  their split-half rel (a_p = R1's per-prompt RSI difference projected like u_T_perp / |u_T_perp|, b_p = the RGI difference projected
  like u_RGI_perp / |u_RGI_perp|); cos(u_RGI, tone); cos(u_GI^TOP, u_RGI_perp); gate GA(b) = AUC(u_RGI) >= 0.85;
  statements: u_S recomputed (flag 'u_S reproduction differs' if |u_S - R1 u_S| / |R1 u_S| > 1e-4; R1 u_S stays in force); 2,000
    label-permutation nulls (seed 23; null = mean of pseudo-pro minus mean of pseudo-anti) projected like u_S_perp;
  NEG: u_neg recomputed (same flag rule); 2,000 within-pair sign-flip nulls of the 40 differences (seed 24);
  match rate of this session's RSI_PRO / RSI_ANTI texts with R1.state[14]['axis_rsi']['conts'] (22k), with the versions next to R1.meta.
Outputs (in <r2_dir>): axes/o2_texts.pt (stage texts), local/axes_reads.pt (stage read, local only), axes/tok14.pt (float16),
axes/r2_axes.pt (texts as token ids; per-block states, per-prompt differences, axes, nulls, O1 fixed axes), axes/r2_axes.json.
Usage: python r2_axes.py <root> <r1_dir> <r2_dir> [texts | read | stats | all | selftest]     env DEVICE, NT, MOCK"""
import glob, json, math, os, random, shutil, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
import r2_rk as R
import r2_stimuli as S

SEEDS = dict(RGI=21, dir=22, S=23, NEG=24)
BOOT_SEED, N_BOOT, N_NULL = 25, 2000, 2000
GA_AUC = 0.85
SETS = S.O2_SETS                                                                       # RSI_PRO, RSI_ANTI, RGI_PRO, RGI_ANTI, TOPIC_PRO, TOPIC_ANTI


def ga_gate(auc_rgi):
    """GA(b) (2.4, G28): cross-fitted centred AUC(u_RGI) >= 0.85 (the same function decides in session and in r2_ana, OD-G8)."""
    return bool(math.isfinite(auc_rgi) and auc_rgi >= GA_AUC)


def paths(r2_dir):
    a = os.path.join(r2_dir, "axes")
    return dict(texts=os.path.join(a, "o2_texts.pt"), reads=os.path.join(r2_dir, "local", "axes_reads.pt"), tok=os.path.join(a, "tok14.pt"),
                pt=os.path.join(a, "r2_axes.pt"), json=os.path.join(a, "r2_axes.json"))


def versions():
    import transformers
    return dict(torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


# ---------------------------------------------------------------- stage 'texts'
def stage_texts(rk, P):
    """The six O2 text sets (prompt order), generated unsteered at rk.TEMP = 0.7 in rk_loop.text_axis's row order (pro, anti per prompt)."""
    if os.path.exists(P["texts"]): return torch.load(P["texts"], map_location="cpu", weights_only=False)
    assert (S.RSI_PRO, S.RSI_ANTI) == tuple(rk.INSTR["rsi"]), "O2 RSI instructions differ from rk_loop.INSTR['rsi']"
    assert rk.TEMP == 0.7 and rk.STEER["v"] is None, "O2 texts are generated unsteered at T 0.7"
    sets, keys = {}, {}
    for _, pro, anti in S.O2_PAIRS:
        rows = []
        for p in range(rk.NP):
            for s, nm in ((1, pro), (-1, anti)):
                k = rk.keyseed("axis", "rsi", p, s) if rk.MOCK else rk.keyseed("axis", "rsi", p)
                rows.append((rk.enc(rk.chat(S.instructed(rk.PROMPTS[p], S.O2_INSTR[nm]))), k, s))
        conts = rk.generate(rows); sets[pro], sets[anti] = conts[0::2], conts[1::2]; keys[pro] = [r[1] for r in rows[0::2]]; keys[anti] = [r[1] for r in rows[1::2]]
        R.log(f"O2 texts {pro} / {anti}: {sum(len(c) == 0 for c in conts)} empty")
    T = dict(sets=sets, keys=keys, instr=dict(S.O2_INSTR), prompts=list(rk.PROMPTS), temp=rk.TEMP, n_new=rk.NNEW, mock=bool(rk.MOCK),
             key_rule="keyseed('axis', 'rsi', p)" + (" + sign (MOCK)" if rk.MOCK else ""), generated_in_block=rk.BL)
    R.torch_save(P["texts"], T); return T


# ---------------------------------------------------------------- stage 'read'
def statement_selection(mock):
    """rk_loop.axes()'s u_S statements: random.Random(0) shuffles of the Yes then the No items of rsi_statements; the first 50 of each
    (6 under MOCK)."""
    from rsi_statements import items as rsi_items
    L = rsi_items(); rng = random.Random(0)
    pos = [x for x in L if x["answer_matching_behavior"].strip() == "Yes"]; ng = [x for x in L if x["answer_matching_behavior"].strip() == "No"]
    rng.shuffle(pos); rng.shuffle(ng); nb = 50 if not mock else 6
    return [x["statement"] for x in pos[:nb]], [x["statement"] for x in ng[:nb]]


def stage_read(rk, T, r1_dir, B3, P, final14=True):
    """All reads of O2 (820 texts in production): O2 sets at all 28 blocks, statements and NEG at the blocks of B3, R1 RSI / sea texts
    and the session's RGI texts per token at block 14. The R1 texts come from R1.state[14] and are read only if block 14's round-1
    outputs are final (G4); otherwise tok14.pt holds the session's RGI sets alone and the R1 parts are 'not run'."""
    if os.path.exists(P["reads"]) and os.path.exists(P["tok"]): return torch.load(P["reads"], map_location="cpu", weights_only=False)
    ST14 = (rk.ST if rk.BL == 14 else R.load_r1_state(r1_dir, 14)) if final14 else None
    out = dict(B3=list(B3), X28={}, n_prompts=rk.NP, host_block=rk.BL, r1_block14="read" if final14 else "not run (round-1 output missing)")
    tok = dict(block=14, dtype="float16", sets={}, fold=(ST14 if final14 else rk.ST)["axes"]["fold"].clone(), prompts=list(rk.PROMPTS),
               r1_sets="read" if final14 else "not run (round-1 output missing)")
    for nm in SETS:
        A = R.read_all_blocks(rk.PIDS, T["sets"][nm], token_blocks=(14,) if nm in ("RGI_PRO", "RGI_ANTI") else ())
        out["X28"][nm] = A["X"].float()
        if nm in ("RGI_PRO", "RGI_ANTI"): tok["sets"][nm] = dict(tok=A["tok"][14], conts=T["sets"][nm], prompt=list(range(rk.NP)), source="round 2 (O2)")
    pro, anti = statement_selection(rk.MOCK); base = rk.enc(rk.chat("Say one sentence.")); bl = sorted(set(B3) | {14})
    out["stmt"] = dict(pro=pro, anti=anti, blocks=bl, Xp=R.read_all_blocks([base] * len(pro), [rk.enc(s) for s in pro], blocks=bl)["X"].float(),
                       Xa=R.read_all_blocks([base] * len(anti), [rk.enc(s) for s in anti], blocks=bl)["X"].float())
    out["neg"] = dict(pairs=list(rk.NEG_PAIRS), blocks=bl, Xpos=R.read_all_blocks([base] * len(rk.NEG_PAIRS), [rk.enc(a) for a, b in rk.NEG_PAIRS], blocks=bl)["X"].float(),
                      Xneg=R.read_all_blocks([base] * len(rk.NEG_PAIRS), [rk.enc(b) for a, b in rk.NEG_PAIRS], blocks=bl)["X"].float())
    out["r1_read"] = {}
    for name in (("rsi", "sea") if final14 else ()):
        c = ST14[f"axis_{name}"]["conts"]
        for tag, cc, ref in (("PRO", c[0::2], ST14[f"axis_{name}"]["Xp"]), ("ANTI", c[1::2], ST14[f"axis_{name}"]["Xa"])):
            A = R.read_all_blocks(rk.PIDS, cc, blocks=[14], token_blocks=[14]); key = f"R1_{name.upper()}_{tag}"
            tok["sets"][key] = dict(tok=A["tok"][14], conts=cc, prompt=list(range(rk.NP)), source=f"R1.state[14]['axis_{name}']['conts'][{'0' if tag == 'PRO' else '1'}::2]")
            ok = ~torch.isnan(ref).any(1); m = A["X"][:, 0].float()
            out["r1_read"][key] = float(((m[ok] - ref[ok]).abs().max() / ref[ok].abs().max()).item()) if ok.any() else float("nan")
    R.torch_save(P["tok"], tok); R.torch_save(P["reads"], out); R.log(f"O2 reads done; R1 read check {out['r1_read']}"); return out


# ---------------------------------------------------------------- stage 'stats'
def _w_axis(W, d, keep, Q=None, extra=None):
    """Weighted per-replicate axes: (W * keep) @ d / sum, projected; W [B, P], d [P, D] (NaN rows allowed where keep is False)."""
    w = W * keep.double()[None]; s = w.sum(1, keepdim=True); m = (w @ torch.nan_to_num(d)) / s
    if Q is not None: m = m - (m @ Q) @ Q.T
    if extra is not None: e = R.unit(extra.double()); m = m - (m @ e)[:, None] * e
    m[s[:, 0] == 0] = float("nan"); return m


def _w_rel(W, d, keep, halv, Q=None, extra=None):
    """Per-replicate split-half rel over the K2 halvings (replicate weights within each half; rk.stats's formula and floor)."""
    rs = []
    for m in halv:
        m = torch.as_tensor(m); a = _w_axis(W, d, keep & m, Q, extra); b = _w_axis(W, d, keep & ~m, Q, extra)
        rs.append((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12))
    r = torch.nanmean(torch.stack(rs), 0); rel = torch.where(r > -1, (2 * r / (1 + r)).clamp(min=0.05), torch.full_like(r, 0.05))
    rel[~torch.isfinite(r)] = float("nan"); return rel


def _cos(a, b): return R.cosf(a.double(), b.double())


def block_stats(b, X, ST_b, stmt, neg, W, dev):
    """O2 statistics of block b. X: {set: [P, D] float64 states at b}; ST_b: R1.state[b]; stmt / neg: dicts of [n, D] states at b;
    W: [N_BOOT, P] prompt-bootstrap weights. Returns (tensors for r2_axes.pt, numbers for r2_axes.json)."""
    ax = ST_b["axes"]; fold = ax["fold"]; P = len(fold); halv = R.halvings(P); uT = ax["u_T_perp"].double(); tT = R.unit(uT)
    Q1 = R.perp_basis([ax["sports"], ax["tone"], ax["sea"], ax["u_neg"]])
    TOP = R.axis_from_texts(X["TOPIC_PRO"], X["TOPIC_ANTI"], fold, n_null=0)
    Q5 = R.perp_basis([ax["sports"], ax["tone"], ax["sea"], ax["u_neg"], TOP["u"]])
    RGI = R.axis_from_texts(X["RGI_PRO"], X["RGI_ANTI"], fold, seed=SEEDS["RGI"], basis=Q5, n_null=N_NULL)
    DIR = R.axis_from_texts(X["RSI_PRO"], X["RGI_PRO"], fold, seed=SEEDS["dir"], n_null=N_NULL)
    GIT = R.axis_from_texts(X["RGI_PRO"], X["TOPIC_PRO"], fold, n_null=0)
    RS2 = R.axis_from_texts(X["RSI_PRO"], X["RSI_ANTI"], fold, basis=Q1, n_null=0)                         # this session's RSI texts (g_0)
    RS1 = R.axis_from_texts(ST_b["axis_rsi"]["Xp"], ST_b["axis_rsi"]["Xa"], fold, basis=Q1, n_null=0)     # R1's RSI texts behind u_T_perp
    uR = RGI["u_perp"]; uRT = uR - (uR @ tT) * tT; nR = RGI["null"]; nRT = nR - (nR @ tT)[:, None] * tT
    u_diff = R.unit(R.unit(uT) - R.unit(uR)); u_com = R.unit(R.unit(uT) + R.unit(uR))
    rel = dict(u_T_perp=R.axis_rel(RS1["diff_all"], RS1["keep"], Q=Q1), u_RGI_perp=R.axis_rel(RGI["diff_all"], RGI["keep"], Q=Q5),
               u_RGI_perpT=R.axis_rel(RGI["diff_all"], RGI["keep"], Q=Q5, extra=uT), u_RGI=R.axis_rel(RGI["diff_all"], RGI["keep"]),
               u_TOPIC=R.axis_rel(TOP["diff_all"], TOP["keep"]), u_dir=R.axis_rel(DIR["diff_all"], DIR["keep"]),
               u_GI_TOP=R.axis_rel(GIT["diff_all"], GIT["keep"]), u_RSI_session_perp=R.axis_rel(RS2["diff_all"], RS2["keep"], Q=Q1))
    cTR = _cos(uT, uR); dis = cTR / math.sqrt(rel["u_T_perp"]["rel"] * rel["u_RGI_perp"]["rel"])
    # 90 % prompt-bootstrap interval of the disattenuated cosine (R1 basis and u_TOPIC fixed at their point estimates)
    Wd = torch.as_tensor(W, dtype=torch.float64, device=dev); to = lambda t: t.to(dev)
    dT, kT, dR, kR = to(RS1["diff_all"]), to(RS1["keep"]), to(RGI["diff_all"]), to(RGI["keep"]); Q1d, Q5d = to(Q1), to(Q5)
    hv = [torch.as_tensor(m, device=dev) for m in halv]
    aT, aR = _w_axis(Wd, dT, kT, Q1d), _w_axis(Wd, dR, kR, Q5d)
    rT, rR = _w_rel(Wd, dT, kT, hv, Q1d), _w_rel(Wd, dR, kR, hv, Q5d)
    bs = ((aT * aR).sum(1) / (aT.norm(dim=1) * aR.norm(dim=1) + 1e-12) / torch.sqrt(rT * rR)).cpu().numpy()
    lo, hi = R.interval(bs, 0.90)
    # common and difference parts (prompts kept in both)
    kb = RS1["keep"] & RGI["keep"]; a = R.project_out(torch.nan_to_num(RS1["diff_all"]), Q1) / uT.norm(); bb = R.project_out(torch.nan_to_num(RGI["diff_all"]), Q5) / uR.norm()
    com, dif = (a + bb) / 2, (a - bb) / 2
    common = dict(norm=float(com[kb].mean(0).norm()), **R.axis_rel(com, kb)); difference = dict(norm=float(dif[kb].mean(0).norm()), **R.axis_rel(dif, kb))
    # statements (u_S) and NEG (u_neg)
    Sp, Sa = stmt["pro"], stmt["anti"]; uS = Sp.mean(0) - Sa.mean(0); nb = len(Sp); St = torch.cat([Sp, Sa])
    g = torch.Generator().manual_seed(SEEDS["S"]); perm = torch.stack([torch.randperm(2 * nb, generator=g) for _ in range(N_NULL)])
    Lm = torch.zeros(N_NULL, 2 * nb, dtype=torch.float64); Lm.scatter_(1, perm[:, :nb], 1.0 / nb); Lm.scatter_(1, perm[:, nb:], -1.0 / nb)
    nS = R.project_out(Lm @ St, Q1)
    dN = neg["neg"] - neg["pos"]; uN = dN.mean(0); g = torch.Generator().manual_seed(SEEDS["NEG"])
    E = (torch.randint(0, 2, (N_NULL, len(dN)), generator=g) * 2 - 1).double(); nN = (E @ dN) / len(dN)
    rdS = float((uS - ax["u_S"].double()).norm() / ax["u_S"].double().norm()); rdN = float((uN - ax["u_neg"].double()).norm() / ax["u_neg"].double().norm())
    flags = (["u_S reproduction differs"] if rdS > 1e-4 else []) + (["u_neg reproduction differs"] if rdN > 1e-4 else [])
    rdT = float((RS1["u_perp"] - uT).norm() / uT.norm())
    if rdT > 1e-4: flags.append("u_T_perp recomputed from R1 axis_rsi differs")
    GA = ga_gate(RGI["auc"]); f32 = lambda t: t.float()
    T = dict(X={k: f32(v) for k, v in X.items()}, fold=fold.clone(),
             diff={k: f32(A["diff_all"]) for k, A in (("RSI_session", RS2), ("RGI", RGI), ("TOPIC", TOP), ("dir", DIR), ("GI_TOP", GIT), ("RSI_R1", RS1))},
             keep={k: A["keep"] for k, A in (("RSI_session", RS2), ("RGI", RGI), ("TOPIC", TOP), ("dir", DIR), ("GI_TOP", GIT), ("RSI_R1", RS1))},
             u_RGI=f32(RGI["u"]), u_TOPIC=f32(TOP["u"]), u_GI_TOP=f32(GIT["u"]), u_RGI_perp=f32(uR), u_RGI_perpT=f32(uRT), u_dir=f32(DIR["u"]),
             u_diff=f32(u_diff), u_com=f32(u_com), u_T_perp=f32(uT), u_RSI_session=f32(RS2["u"]), u_RSI_session_perp=f32(RS2["u_perp"]),
             u_S=f32(uS), u_neg=f32(uN), null_u_RGI_perp=f32(nR), null_u_RGI_perpT=f32(nRT), null_u_dir=f32(DIR["null"]), null_u_S_perp=f32(nS),
             null_u_neg=f32(nN), basis_R1=Q1, basis_RGI=Q5, S_states=dict(pro=f32(Sp), anti=f32(Sa)), NEG_states=dict(pos=f32(neg["pos"]), neg=f32(neg["neg"])),
             GA=GA, auc_RGI=RGI["auc"], boot_disatt=torch.as_tensor(bs),
             fixed_axes=dict(RSI=dict(a_A=RS2["a_A"], a_B=RS2["a_B"], g0=RS2["gap_perp_cross"], basis="sports, tone, sea, u_neg (R1 axes of the block)"),
                             RGI=dict(a_A=RGI["a_A"], a_B=RGI["a_B"], g0=RGI["gap_perp_cross"], basis="sports, tone, sea, u_neg, u_TOPIC")))
    ax_stats = lambda A: dict(auc=A["auc"], paired_win=A["paired_win"], gap_cross=A["gap_cross"], kept=A["kept"])
    J = dict(status="ok", GA=GA, auc_RGI=RGI["auc"], labels=[] if GA else ["RGI axis not resolved"],
             fixed_axes=dict(RSI=dict(g0=RS2["gap_perp_cross"], kept=RS2["kept"]), RGI=dict(g0=RGI["gap_perp_cross"], kept=RGI["kept"]),
                             note="O1: a_A, a_B (float64) in r2_axes.pt blocks[b]['fixed_axes']; g(texts) = r2_rk.gap_on_fixed_axes"),
             axes=dict(RSI_session=dict(ax_stats(RS2), gap_perp_cross=RS2["gap_perp_cross"]), RGI=dict(ax_stats(RGI), gap_perp_cross=RGI["gap_perp_cross"], gap_perp=RGI["gap_perp"]),
                       TOPIC=ax_stats(TOP), GI_TOP=ax_stats(GIT), dir=ax_stats(DIR)),
             Delta_T=float(ax["gap_perp_cross"]), g0_RSI=RS2["gap_perp_cross"], Delta_RGI=RGI["gap_perp_cross"], rel=rel,
             cos_T_RGI=cTR, cos_T_RGI_disatt=dis, cos_T_RGI_disatt_ci90=[lo, hi], boot=dict(B=N_BOOT, seed=BOOT_SEED, valid=int(np.isfinite(bs).sum())),
             common=common, difference=difference, n_prompts_common=int(kb.sum()), cos_u_RGI_tone=_cos(RGI["u"], ax["tone"]),
             cos_GI_TOP_RGI_perp=_cos(GIT["u"], uR), cos_RGI_perp_perpT=_cos(uR, uRT), perp_share_RGI=float(uR.norm() / RGI["u"].norm()),
             cos_RSI_session_R1=_cos(RS2["u_perp"], uT), cos_u_dir_u_T_perp=_cos(DIR["u"], uT),
             reproduction=dict(u_S_rel_diff=rdS, u_neg_rel_diff=rdN, u_T_perp_recompute_rel_diff=rdT, n_statements=nb, n_neg=len(dN)), flags=flags)
    return T, J


def match_rate(a_list, b_list):
    """Token-level match of regenerated texts with round-1 texts of the same keys: share of identical texts and the share of tokens in
    the common prefix (sum of common-prefix lengths / sum of max lengths)."""
    ident = [a == b for a, b in zip(a_list, b_list)]; num = den = 0
    for a, b in zip(a_list, b_list):
        k = R.first_divergence(a, b); num += max(len(a), len(b)) if k < 0 else k; den += max(len(a), len(b))
    return dict(n=len(ident), identical_share=float(np.mean(ident)) if ident else float("nan"), prefix_token_share=num / den if den else float("nan"))


def stage_stats(r1_dir, info, T, RD, P, dev):
    """Per block of B3 whose round-1 outputs are final (G4): block_stats; the other blocks 'not run (round-1 output missing)'. Match rate
    against R1.state[14]'s texts only if block 14 is final."""
    if os.path.exists(P["pt"]) and os.path.exists(P["json"]): return json.load(open(P["json"]))
    B3 = info["B3"]; NP = RD["n_prompts"]; W = R.prompt_bootstrap(NP, B=N_BOOT, seed=BOOT_SEED); blocks_pt, blocks_js = {}, {}
    ST14 = None; final = {b: R.block_final(r1_dir, b) for b in B3}
    for b in B3:
        try:
            if not final[b]: raise R.R1Missing(f"round-1 outputs of block {b} are not final (summary_b{b}.json missing; G4)")
            ST_b = R.load_r1_state(r1_dir, b)
        except (R.R1Missing, FileNotFoundError) as e:
            blocks_js[str(b)] = dict(status="not run (round-1 output missing)", GA=None, error=str(e)); continue
        if b == 14: ST14 = ST_b
        X = {nm: RD["X28"][nm][:, b].double() for nm in SETS}; j = RD["stmt"]["blocks"].index(b)
        stmt = dict(pro=RD["stmt"]["Xp"][:, j].double(), anti=RD["stmt"]["Xa"][:, j].double()); neg = dict(pos=RD["neg"]["Xpos"][:, j].double(), neg=RD["neg"]["Xneg"][:, j].double())
        blocks_pt[b], blocks_js[str(b)] = block_stats(b, X, ST_b, stmt, neg, W, dev)
        R.log(f"O2 block {b}: AUC(u_RGI) {blocks_js[str(b)]['auc_RGI']:.3f} GA {blocks_js[str(b)]['GA']} flags {blocks_js[str(b)]['flags']}")
        del ST_b
    if ST14 is None and final.get(14): ST14 = R.load_r1_state(r1_dir, 14)
    if ST14 is not None:
        c = ST14["axis_rsi"]["conts"]; mr = dict(RSI_PRO=match_rate(T["sets"]["RSI_PRO"], c[0::2]), RSI_ANTI=match_rate(T["sets"]["RSI_ANTI"], c[1::2]))
    else: mr = dict(status="not run (round-1 output missing)")
    meta1 = None; p = os.path.join(r1_dir, "gpu1", "shard_done.json")                  # R1.meta of gpu1 (G47), metadata only
    if os.path.exists(p): m = json.load(open(p)); meta1 = {k: m.get(k) for k in ("gpu", "torch", "transformers", "code_commit")}
    pt = dict(texts=T["sets"], keys=T["keys"], instr=T["instr"], prompts=T["prompts"], key_rule=T["key_rule"], B3=list(B3), blocks=blocks_pt,
              GA={b: v["GA"] for b, v in blocks_pt.items()}, X28={k: v.half() for k, v in RD["X28"].items()}, seeds=dict(SEEDS, boot=BOOT_SEED),
              statements=dict(pro=RD["stmt"]["pro"], anti=RD["stmt"]["anti"]), mock=bool(T["mock"]))
    js = dict(script="r2_axes.py", plan="final plan (abschnitt11_12_runde2_plan.md) section 2.4 (O1 fixed axes, O2)", B3=list(B3), seeds=dict(SEEDS, boot=BOOT_SEED),
              n_null=N_NULL, n_boot=N_BOOT, ga_threshold=GA_AUC, mock=bool(T["mock"]), key_rule=T["key_rule"], host_block=RD.get("host_block"),
              final={str(b): v for b, v in final.items()}, blocks=blocks_js, GA={str(b): v.get("GA") for b, v in blocks_js.items()},
              match_rate=mr, versions=versions(), r1_meta_gpu1=meta1, r1_read_check=RD["r1_read"], block_info=info, finder=list(R.FINDER_LOG))
    R.torch_save(P["pt"], pt); R.write_json(P["json"], js); return js


# ---------------------------------------------------------------- driver
STAGES = ("texts", "read", "stats")


def host_block(r1_dir, info):
    """The block whose process generates and reads the O2 texts (unsteered: the texts do not depend on the block): the block already
    open in this process, else 14, else the first other block of B3 - among the blocks whose round-1 outputs are final (G4, block 14
    included). None if no block of B3 is final."""
    if R._RK is not None:
        if R.BLK["b"] not in info["B3"]: raise RuntimeError(f"block {R.BLK['b']} is open in this process but not in B3 {info['B3']}")
        return R.BLK["b"]
    return next((b for b in [14] + [x for x in info["B3"] if x != 14] if R.block_final(r1_dir, b)), None)


def main(root, r1_dir, r2_dir, stage="all"):
    """Runs the stages up to 'stage' ('all' = all three); every stage is skipped when its output exists, and the model is loaded (the
    host block opened, normally 14) only when texts or reads have to be made. Blocks whose round-1 outputs are not final are 'not run
    (round-1 output missing)' (G4); if no block of B3 is final, r2_axes.json says so and nothing else is written."""
    P = paths(r2_dir); info = R.r1_blocks(r1_dir); upto = 2 if stage == "all" else STAGES.index(stage)
    have_texts = os.path.exists(P["texts"]); have_reads = os.path.exists(P["reads"]) and os.path.exists(P["tok"])
    need_model = (not have_texts) or (upto >= 1 and not have_reads and not (upto == 2 and os.path.exists(P["pt"]) and os.path.exists(P["json"])))
    rk = None
    if need_model:
        host = host_block(r1_dir, info)
        if host is None:
            js = dict(script="r2_axes.py", status="not run (round-1 output missing)", B3=info["B3"], blocks={str(b): dict(status="not run (round-1 output missing)", GA=None) for b in info["B3"]},
                      GA={str(b): None for b in info["B3"]}, block_info=info, finder=list(R.FINDER_LOG))
            R.write_json(P["json"], js); R.log("O2 not run: no block of B3 has final round-1 outputs (G4)"); return js
        rk = R.open_block(root, r1_dir, r2_dir, host)
    T = stage_texts(rk, P)
    if upto == 0: return None
    if upto == 2 and os.path.exists(P["pt"]) and os.path.exists(P["json"]): return json.load(open(P["json"]))
    RD = stage_read(rk, T, r1_dir, info["B3"], P, final14=R.block_final(r1_dir, 14))
    if upto == 1: return None
    dev = torch.device("cuda" if (os.environ.get("DEVICE", "cpu") != "cpu" and torch.cuda.is_available()) else "cpu")
    js = stage_stats(r1_dir, info, T, RD, P, dev); R.log(f"O2 done: GA {js['GA']}")
    return js


def synthetic_block(cos_true=0.6, D=64, P=64, sigma=0.3, base_sd=0.3, seed=0):
    """A synthetic block with planted axes (no model): RSI direction t and RGI direction r with cos(t, r) = cos_true, both orthogonal to
    the four round-1 foreign axes and to the TOPIC direction; per-prompt states base_p +- direction + noise. Returns (X, ST_b, stmt, neg)."""
    g = torch.Generator().manual_seed(seed); rnd = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    basis = [rnd(D) for _ in range(4)]; Q = R.perp_basis(basis)
    gs = lambda vs: [R.unit(R.project_out(v, torch.cat([Q] + [R.unit(w)[:, None] for w in vs[:i]], 1))) for i, v in enumerate(vs)]
    t, w, topic = gs([rnd(D), rnd(D), rnd(D)]); r = R.unit(cos_true * t + math.sqrt(1 - cos_true ** 2) * w)
    base = base_sd * rnd(P, D); X = {}
    for nm, v in (("RSI", t), ("RGI", r), ("TOPIC", topic)):
        X[f"{nm}_PRO"] = base + v + sigma * rnd(P, D); X[f"{nm}_ANTI"] = base - v + sigma * rnd(P, D)
    Xp1, Xa1 = base + t + sigma * rnd(P, D), base - t + sigma * rnd(P, D); fold = torch.zeros(P, dtype=torch.bool); fold[torch.randperm(P, generator=g)[: P // 2]] = True
    A1 = R.axis_from_texts(Xp1, Xa1, fold, basis=Q, n_null=0); Sp, Sa, Np = rnd(10, D), rnd(10, D), rnd(8, D); Nn = Np + basis[3]   # u_neg = basis[3]
    ax = dict(sports=basis[0], tone=basis[1], sea=basis[2], u_neg=basis[3], u_T_perp=A1["u_perp"], fold=fold, gap_perp_cross=A1["gap_perp_cross"], u_S=Sp.mean(0) - Sa.mean(0))
    return X, dict(axes=ax, axis_rsi=dict(Xp=Xp1.float(), Xa=Xa1.float())), dict(pro=Sp, anti=Sa), dict(pos=Np, neg=Nn), dict(t=t, r=r, topic=topic)


def selftest(root, r1_dir, r2_dir):
    """MOCK code test: runs every stage, checks the outputs against rk_loop and against their definitions, then reruns (resume); a
    synthetic block with planted axes checks the statistics themselves (AUC, rel, disattenuated cosine and its bootstrap interval)."""
    assert R.MOCK, "selftest needs MOCK=1"; res = []

    def check(name, cond, info=""):
        res.append(dict(test=name, ok=bool(cond), info=str(info))); print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)
    X, STb, stmt, neg, truth = synthetic_block(); P = len(STb["axes"]["fold"]); W = R.prompt_bootstrap(P, B=N_BOOT, seed=BOOT_SEED)
    Tb, Jb = block_stats(99, X, STb, stmt, neg, W, torch.device("cpu"))
    lo, hi = Jb["cos_T_RGI_disatt_ci90"]; c_true = float(truth["t"] @ truth["r"])
    check("synthetic block: GA, high rel, disattenuated cos near the planted cosine, interval covers it and the estimate",
          Jb["GA"] and Jb["axes"]["RGI"]["auc"] >= 0.99 and Jb["rel"]["u_T_perp"]["rel"] > 0.9 and Jb["rel"]["u_RGI_perp"]["rel"] > 0.9
          and abs(Jb["cos_T_RGI_disatt"] - c_true) < 0.1 and lo <= Jb["cos_T_RGI_disatt"] <= hi and lo <= c_true <= hi
          and abs(Jb["common"]["norm"] - math.sqrt((1 + Jb["cos_T_RGI"]) / 2)) < 1e-6 and not Jb["flags"], (c_true, Jb["cos_T_RGI_disatt"], (lo, hi), Jb["rel"]["u_RGI_perp"]["rel"]))
    d, k, Q = Tb["diff"]["RGI"].double(), Tb["keep"]["RGI"], Tb["basis_RGI"]; ones = torch.ones(1, P, dtype=torch.float64)
    e1 = float((_w_axis(ones, d, k, Q)[0] - Tb["u_RGI_perp"].double()).norm() / Tb["u_RGI_perp"].norm())
    e2 = abs(float(_w_rel(ones, d, k, [torch.as_tensor(m) for m in R.halvings(P)], Q)[0]) - Jb["rel"]["u_RGI_perp"]["rel"])
    check("bootstrap helpers with unit weights equal the point estimates", e1 < 1e-6 and e2 < 1e-9, (e1, e2))
    js = main(root, r1_dir, r2_dir, "all"); P = paths(r2_dir); pt = torch.load(P["pt"], map_location="cpu", weights_only=False)
    rk = R._RK or R.open_block(root, r1_dir, r2_dir, 14)
    check("three blocks with statistics", sorted(js["blocks"]) == ["14", "20", "8"] and all(v["status"] == "ok" for v in js["blocks"].values()), list(js["blocks"]))
    check("MOCK RSI texts reproduce the MOCK round-1 texts (same keys)", js["match_rate"]["RSI_PRO"]["identical_share"] == 1.0 and js["match_rate"]["RSI_ANTI"]["identical_share"] == 1.0, js["match_rate"])
    check("reading the R1 texts reproduces R1 Xp / Xa", max(js["r1_read_check"].values()) < 1e-5, js["r1_read_check"])
    check("u_S and u_neg reproduce R1; no flags", all(v["reproduction"]["u_S_rel_diff"] < 1e-6 and v["reproduction"]["u_neg_rel_diff"] < 1e-6 and not v["flags"] for v in js["blocks"].values()))
    ok = True
    for b, B in pt["blocks"].items():
        uR, uRT, Q5 = B["u_RGI_perp"].double(), B["u_RGI_perpT"].double(), B["basis_RGI"]; tT = R.unit(B["u_T_perp"].double())
        ok &= float((Q5.T @ uR).norm() / uR.norm()) < 1e-6 and abs(float(uRT @ tT)) < 1e-6 * float(uRT.norm()) and float((B["null_u_RGI_perp"].double() @ Q5).abs().max()) < 1e-5 * float(B["null_u_RGI_perp"].abs().max())
        ok &= tuple(B["null_u_RGI_perp"].shape) == (2000, rk.D) and tuple(B["null_u_S_perp"].shape) == (2000, rk.D) and tuple(B["null_u_neg"].shape) == (2000, rk.D)
        A = R.axis_from_texts(B["X"]["RGI_PRO"], B["X"]["RGI_ANTI"], B["fold"], seed=21, basis=Q5)
        ok &= abs(A["auc"] - js["blocks"][str(b)]["auc_RGI"]) < 1e-9 and pt["GA"][b] == ga_gate(A["auc"]) == js["GA"][str(b)]
        ok &= float((B["u_dir"].double() - (B["X"]["RSI_PRO"].double() - B["X"]["RGI_PRO"].double()).mean(0)).norm()) < 1e-4 * float(B["u_dir"].norm()) + 1e-9
        ok &= abs(float(B["u_diff"].norm()) - 1) < 1e-5 and abs(float(B["u_com"].norm()) - 1) < 1e-5
    check("axes by definition (orthogonality, nulls projected, u_dir, GA from AUC)", ok)
    X14 = pt["blocks"][14]["X"]; R1 = rk.ST["axis_rsi"]
    check("session RSI states equal R1 Xp / Xa at block 14 (identical MOCK texts)", float((X14["RSI_PRO"] - R1["Xp"]).abs().max() / R1["Xp"].abs().max()) < 1e-5)
    tok = torch.load(P["tok"], map_location="cpu", weights_only=False); m = torch.stack([t.double().mean(0) for t in tok["sets"]["RGI_PRO"]["tok"]])
    check("per-token block-14 dump: six sets, means equal the block-14 reads", sorted(tok["sets"]) == sorted(["RGI_PRO", "RGI_ANTI", "R1_RSI_PRO", "R1_RSI_ANTI", "R1_SEA_PRO", "R1_SEA_ANTI"])
          and float((m - X14["RGI_PRO"].double()).abs().max() / X14["RGI_PRO"].abs().max()) < 5e-3)
    st = {k: os.path.getmtime(v) for k, v in P.items() if os.path.exists(v)}; main(root, r1_dir, r2_dir, "all")
    check("resume: a second run skips every stage", st == {k: os.path.getmtime(v) for k, v in P.items() if os.path.exists(v)})
    check("seeds and bootstrap recorded", js["seeds"] == dict(SEEDS, boot=BOOT_SEED) and all(v["boot"]["B"] == N_BOOT for v in js["blocks"].values()))
    ok = True
    for b, B in pt["blocks"].items():
        F = B["fixed_axes"]; Jb = js["blocks"][str(b)]
        for nm, (pro, anti) in (("RSI", ("RSI_PRO", "RSI_ANTI")), ("RGI", ("RGI_PRO", "RGI_ANTI"))):
            g = R.gap_on_fixed_axes(B["X"][pro], B["X"][anti], B["fold"], F[nm]["a_A"], F[nm]["a_B"])["g"]
            ok &= abs(g - F[nm]["g0"]) < 1e-6 * max(1.0, abs(g)) and abs(F[nm]["g0"] - Jb["fixed_axes"][nm]["g0"]) < 1e-12
        ok &= abs(F["RGI"]["g0"] - Jb["Delta_RGI"]) < 1e-12 and abs(F["RSI"]["g0"] - Jb["g0_RSI"]) < 1e-12
        ok &= float((B["basis_R1"].T @ F["RSI"]["a_A"]).abs().max()) < 1e-9 and float((B["basis_RGI"].T @ F["RGI"]["a_B"]).abs().max()) < 1e-9
    check("O1 fixed axes: unit, orthogonal to their basis, g on them = g_0 (RSI) and Delta_RGI (RGI)", ok)
    # G4 with block 14 not final (no summary_b14.json): a separate process (one process = one block) hosts the texts at block 8
    import subprocess
    nf = os.path.join(r2_dir, "selftest_nonfinal14"); shutil.rmtree(nf, ignore_errors=True); shutil.copytree(r1_dir, os.path.join(nf, "r1"))
    for f in glob.glob(os.path.join(nf, "r1", "gpu*", "summary_b14.json")): os.remove(f)
    pr = subprocess.run([sys.executable, os.path.abspath(__file__), root, os.path.join(nf, "r1"), os.path.join(nf, "r2"), "all"], env=dict(os.environ, MOCK="1"),
                        capture_output=True, text=True)
    jn = json.load(open(os.path.join(nf, "r2", "axes", "r2_axes.json"))) if pr.returncode == 0 else {}
    tn = torch.load(os.path.join(nf, "r2", "axes", "tok14.pt"), map_location="cpu", weights_only=False) if pr.returncode == 0 else {}
    Tn = torch.load(os.path.join(nf, "r2", "axes", "o2_texts.pt"), map_location="cpu", weights_only=False) if pr.returncode == 0 else {}
    check("G4: block 14 not final -> block 14 'not run', texts hosted at block 8 and unchanged, R1 block-14 parts 'not run'",
          pr.returncode == 0 and jn["blocks"]["14"]["status"] == "not run (round-1 output missing)" and jn["blocks"]["8"]["status"] == "ok"
          and jn["host_block"] == 8 and jn["match_rate"].get("status", "").startswith("not run") and sorted(tn["sets"]) == ["RGI_ANTI", "RGI_PRO"]
          and Tn["sets"] == pt["texts"] and not os.path.exists(os.path.join(nf, "r2", "rk", "b14")), pr.stderr[-300:] if pr.returncode else jn.get("host_block"))
    ok_all = all(r["ok"] for r in res); R.write_json(os.path.join(r2_dir, "axes", "selftest.json"), dict(ok=ok_all, tests=res))
    print(f"r2_axes selftest: {sum(r['ok'] for r in res)}/{len(res)} passed", flush=True); return ok_all


if __name__ == "__main__":
    if len(sys.argv) < 4: raise SystemExit(__doc__)
    root, r1_dir, r2_dir = sys.argv[1:4]; stage = sys.argv[4] if len(sys.argv) > 4 else "all"
    if stage == "selftest": sys.exit(0 if selftest(root, r1_dir, r2_dir) else 1)
    if stage not in ("all", "texts", "read", "stats"): raise SystemExit(f"unknown stage {stage}")
    main(root, r1_dir, r2_dir, stage)
