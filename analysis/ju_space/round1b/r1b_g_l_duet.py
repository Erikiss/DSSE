"""G L4 core DUET: native Gemma turns, own controls and inherited statistics.

Core statistical functions are statically ported from r1b_l_duet.py. X2/X6
conditions are not scheduled. No Qwen model, token array or run is required.
"""
from pathlib import Path
import numpy as np
import r1b_core as K
import r1b_l_stats as S
import r1b_g_l_native as E
from r1b_g_l import loadpt, save, wj, config
from r1b_l_base import clean, sha, SHORT_QA
from r1b_g_l_erasure import Damage

DIALOGUE = ("C-opp", "C-half", "C-Hup", "C-Rup")

class _ProbeCM:
    def __init__(self, pr): self.pr = pr
    def __enter__(self): return self.pr
    def __exit__(self, *args): self.pr.remove()

class _Stage:
    def __init__(self, ctx): self.ctx = ctx
    def __getattr__(self, name): return getattr(self.ctx, name)
    def path(self, rel): return self.ctx.path(rel.removeprefix("l/"))

def load_pop(ctx):
    p = ctx.artifact("l_prepare", "l_pop.json")
    return K.load_json(str(p)) if Path(p).is_file() else {}

def setup(ctx):
    tok, model = ctx.model()
    p = ctx.artifact("l_prepare", "items.json")
    rows = K.load_json(str(p))["items"] if Path(p).is_file() else []
    return tok, model, config(ctx), {}, {r["id"]: r for r in rows}

def refw(ctx, model):
    import torch
    if not Path(ctx.artifact("l_prepare", "refw.pt")).is_file():
        return None
    return {k: torch.as_tensor(v, dtype=torch.float32, device=K.model_device(model))
            for k, v in loadpt(ctx.artifact("l_prepare", "refw.pt")).items()
            if k in ("mu_z", "sd_z", "mu_cett", "sd_cett")}

def d1(ctx, tok=None):
    rows = ctx.data().get("duet_items", [])
    n = config(ctx)["l4_items"]
    if len(rows) < n:
        raise K.NotRun(f"L4 needs {n} raw F-sym duet_items")
    for row in rows[:n]:
        if any(not isinstance(row.get(k), str) or not row[k] for k in ("entity","attribute","v1","v2")):
            raise ValueError("L4 duet_items require entity/attribute/v1/v2 raw strings")
        if tok is not None:
            for prefix in ("", " "):
                a, b = (K.enc(tok, prefix + row[k]) for k in ("v1", "v2"))
                if not a or not b:
                    raise K.NotRun("L4 stance values have empty target tokenizations")
    return rows[:n], ["explicit synthetic G fixture"] if ctx.mock else []

def ref_sets(ctx, tok, items, cf):
    from r1b_g_engine import ModelEngine
    from r1b_g_l_erasure import _reference
    E_ = ModelEngine(ctx)
    refs = loadpt(ctx.artifact("k_prepare", "references.pt")).get("ref_r16", [])
    nr, nk = (2, 4) if ctx.mock else (16, 16)
    if len(refs) < nr:
        raise K.NotRun("L4 needs the target's own REF-R16")
    r = K.RefSet([(r["ids"],r["positions"]) for r in refs[:nr]], "G-REF-R16")
    k = _reference(E_, ctx.data().get("ref_k16", []), "G-REF-K16", nk)
    sl = sorted((v for v in items.values() if v["split"] == "S_L"), key=lambda r:r["id"])
    if len(sl) < cf["ref_heldout"]:
        raise K.NotRun("L4 needs the disjoint S_L damage-reference rows")
    held = sl[-cf["ref_heldout"]:]
    h = K.RefSet.from_prompt_conts([E_.encode(SHORT_QA.format(q=r["question"])) for r in held],
                                  [r["ans0"] for r in held], "G S_L held-out")
    return r + k + h, r + k, []


def sequence_choices(model, tok, prefixes, candidates, batch=32):
    """L4's summed teacher-forced log probability of each complete value."""
    if len(prefixes) != len(candidates) or any(len(c) != 2 or any(not v for v in c) for c in candidates):
        raise ValueError("one pair of nonempty target-token value sequences is required per context")
    seqs = [list(p) + list(v) for p, pair in zip(prefixes, candidates) for v in pair]
    starts = [len(p) for p in prefixes for _ in range(2)]
    if not seqs:
        return np.zeros((0, 2))
    rows = E.tf_pass(model, tok, seqs, starts, batch=batch)
    return np.asarray([float(np.asarray(r["lp"]).sum()) for r in rows]).reshape(-1, 2)

SYS = ("You are Agent {X}, a researcher. {note}You are discussing the {attribute} of {entity} with a colleague. "
       "Reply in at most three sentences.")

NOTE = "Your notes say that the {attribute} of {entity} is {v}. "

SYS_DOC = "You are a researcher analysing the {attribute} of {entity}. Reply in at most three sentences."

ASK = "Your colleague asks for your view on the {attribute} of {entity}."

SAYS = "Your colleague says: «{m}»"

ARCHIVE = "You find this note in the archive: «{m}» Continue your analysis."

SOLO1, SOLO_N = "Give your view on the {attribute} of {entity}.", "Continue your analysis."

READ_U = "Text: «{m}»\nAccording to this text, what is the {attribute} of {entity}?"

READ_A = "The {attribute} of {entity} is"

PROBE_Q = "In one word: what is the {attribute} of {entity}?"

def runs(cf):
    return [(i, c, s) for i in range(cf["l4_items"]) for c in (1, 2) for s in range(cf["l4_seeds"])]

def primes(cond, c):
    """(A's value index, B's value index) with 0 = neutral, 1 = v1, 2 = v2."""
    if cond == "C-opp":
        return (1, 2) if c == 1 else (2, 1)
    if cond == "C-solo":
        return (c, 0)
    return (0, c)

def _sys(it, X, v):
    note = NOTE.format(attribute=it["attribute"], entity=it["entity"], v=it[f"v{v}"]) if v else ""
    return SYS.format(X=X, note=note, attribute=it["attribute"], entity=it["entity"])

def l4_cal(ctx):
    """Gains of the random arms: C-Rup sets (CTRL 21-24) matched to H's alpha = 2 damage, C-Rdown (X2) the same sets matched to H's
    alpha = 0 damage; SO-DMG gain form, <= 6 evaluations, +-10 %; REF-U32 damage reported."""
    tok, model, cf, H, items = setup(ctx)
    rw = refw(ctx, model)
    pop = load_pop(ctx)
    refL, refU, rl = ref_sets(ctx, tok, items, cf)
    dm, du = Damage(model, refL), Damage(model, refU)

    def dmg(ids, a, D=dm):
        def su():
            pr = E.Probe(model, rw); pr.add_op(ids, "scale", val=a)
            return _ProbeCM(pr)
        return D(su)["damage"]
    base = 20 if len(pop["CTRL"]) >= 24 else max(0, len(pop["CTRL"]) - cf["l4_sets"])
    sets = [pop["CTRL"][base + j] for j in range(cf["l4_sets"])]
    res = dict(sets=sets, ctrl_index=[base + j for j in range(cf["l4_sets"])], labels=rl)
    for arm, a0 in (("up", 2.0),):
        tgt = dmg(pop["P"], a0)
        cal = [K.calibrate(lambda a: dmg(s, a), a0, tgt, form="gain", max_eval=6) for s in sets]
        res[arm] = dict(H_alpha=a0, H_damage=tgt, H_damage_U32=dmg(pop["P"], a0, du),
                        sets=[dict(alpha=c["f"], damage=c["damage"], in_band=c["in_band"], labels=c["labels"],
                                   damage_U32=dmg(s, c["f"], du)) for c, s in zip(cal, sets)])
        res[arm]["g_dmg"] = all(c["in_band"] for c in cal)
    wj(ctx, "l/l4_cal.json", res, config=dict(form="gain, <= 6 evaluations", band=0.10), labels=rl)

def l4_run(ctx, cond):
    """One condition: 256 runs (32 items x 2 configurations x 4 seeds), 12 messages of <= 40 new tokens at T 0.7, keys
    ('r1bL4', item, cfg, seed, msg) shared across conditions; C-doc replays B's C-half messages; C-solo (X6) is A alone."""
    torch = K._torch()
    tok, model, cf, H, items = setup(ctx)
    rw = refw(ctx, model)
    pop = load_pop(ctx)
    its, dl = d1(ctx, tok)
    R = runs(cf)
    cal = K.load_json(ctx.path("l/l4_cal.json"))
    nM = cf["l4_msgs"]
    half = loadpt(ctx.path("l/l4_C-half.pt")) if cond == "C-doc" else None
    out = []
    for b0 in range(0, len(R), cf["batch"]):
        rb = R[b0:b0 + cf["batch"]]

        def run():
            B = len(rb)
            msgs = [[] for _ in range(B)]
            if cond in DIALOGUE:
                ctxs = {"A": [None] * B, "B": [None] * B}
                amask = [[] for _ in range(B)]
                for m in range(1, nM + 1):
                    X, Y = ("A", "B") if m % 2 else ("B", "A")
                    for r, (i, c, s) in enumerate(rb):
                        it = its[i]; pa, pb = primes(cond, c)
                        if ctxs[X][r] is None:
                            u = ASK.format(attribute=it["attribute"], entity=it["entity"]) if X == "A" else \
                                SAYS.format(m=tok.decode(msgs[r][-1]))
                            ctxs[X][r] = E.chat_sys_ids(tok, _sys(it, X, pa if X == "A" else pb), u)
                            if X == "A":
                                amask[r] = [False] * len(ctxs["A"][r])
                    pr = E.Probe(model, rw)
                    if X == "A" and cond in ("C-Hup", "C-Rup", "C-Hdown", "C-Rdown"):
                        Tn = max(len(x) for x in ctxs["A"])
                        pos = torch.zeros(B, Tn, dtype=torch.bool)
                        for r in range(B):
                            mk = amask[r]; pos[r, Tn - len(mk):] = torch.tensor(mk, dtype=torch.bool) if mk else pos[r, Tn - len(mk):]
                        if cond in ("C-Hup", "C-Hdown"):
                            ops = [pr.add_op(pop["P"], "scale", val=2.0 if cond == "C-Hup" else 0.0)]
                        else:
                            arm = cal["up" if cond == "C-Rup" else "down"]
                            ops = []
                            for j, sset in enumerate(cal["sets"]):
                                rows = torch.tensor([s % len(cal["sets"]) == (j - c) % len(cal["sets"]) for (_, c, s) in rb])
                                ops.append(pr.add_op(sset, "scale", val=arm["sets"][j]["alpha"], rows=rows))

                        def on_step(step, probe, pos=pos):
                            for o in probe.ops:
                                o["pos"] = pos if step == 0 else None
                    else:
                        on_step = None
                    try:
                        g = E.decode(model, tok, ctxs[X], [ctx.key("r1bL4", i, c, s, m) for (i, c, s) in rb], T=0.7, max_new=cf["l4_new"],
                                     probe=pr, on_step=on_step)
                    finally:
                        pr.remove()
                    for r, x in enumerate(g):
                        msgs[r].append(x["tokens"])
                        ctxs[X][r] = ctxs[X][r] + x["tokens"]
                        if X == "A":
                            amask[r] = amask[r] + [True] * len(x["tokens"])
                        if ctxs[Y][r] is not None and m < nM:
                            sfx = E.turn_suffix_ids(tok, SAYS.format(m=tok.decode(x["tokens"])))
                            ctxs[Y][r] = ctxs[Y][r] + sfx
                            if Y == "A":
                                amask[r] = amask[r] + [False] * len(sfx)
                return dict(msgs=msgs, final_ctx_len=[len(x or []) for x in ctxs["A"]])
            # single-agent conditions: C-doc (replayed B messages from the matched C-half run) and C-solo (X6)
            ctxA = [None] * B
            for k in range(1, nM // 2 + 1):
                for r, (i, c, s) in enumerate(rb):
                    it = its[i]
                    if cond == "C-doc":
                        hm = half["runs"][R.index((i, c, s))]["msgs"][2 * k - 1]
                        u = ARCHIVE.format(m=tok.decode(hm))
                        sysm = SYS_DOC.format(attribute=it["attribute"], entity=it["entity"])
                    else:
                        u = (SOLO1 if k == 1 else SOLO_N).format(attribute=it["attribute"], entity=it["entity"])
                        sysm = _sys(it, "A", primes(cond, c)[0])
                    ctxA[r] = E.chat_sys_ids(tok, sysm, u) if ctxA[r] is None else ctxA[r] + E.turn_suffix_ids(tok, u)
                g = E.decode(model, tok, ctxA, [ctx.key("r1bL4", i, c, s, 2 * k + 1 if cond == "C-doc" else 2 * k - 1) for (i, c, s) in rb],
                             T=0.7, max_new=cf["l4_new"])
                for r, x in enumerate(g):
                    msgs[r].append(x["tokens"]); ctxA[r] = ctxA[r] + x["tokens"]
            return dict(msgs=msgs, final_ctx_len=[len(x) for x in ctxA])
        res = ctx.cached(f"l4_{cond}_{b0}", run)
        out += [dict(run=rb[r], msgs=res["msgs"][r]) for r in range(len(rb))]
    save(ctx, f"l/l4_{cond}.pt", dict(runs=out, cond=cond), config=dict(cond=cond, msgs=nM, new=cf["l4_new"], T=0.7,
                                                                         items_sha256=sha(its)), labels=dl)

def l4_reads(ctx):
    """Context-free stance m = log q(' v1') - log q(' v2') after the frozen frame, for every message of every condition that ran;
    belief probes (append-and-crop, 'In one word: ...') at events 0, 6, 12 for the unpushed dialogue conditions (descriptive)."""
    import os
    tok, model, cf, H, items = setup(ctx)
    its, _ = d1(ctx, tok)
    res = {}
    for cond in DIALOGUE + ("C-doc", "C-solo"):
        p = ctx.path(f"l/l4_{cond}.pt")
        if not os.path.exists(p):
            continue
        d = loadpt(p)

        def run():
            ids, idx = [], []
            for q, rr in enumerate(d["runs"]):
                it = its[rr["run"][0]]
                for j, mm in enumerate(rr["msgs"]):
                    u = READ_U.format(m=tok.decode(mm), attribute=it["attribute"], entity=it["entity"])
                    ids.append(K.chat_ids(tok, u) + K.enc(tok, READ_A.format(attribute=it["attribute"], entity=it["entity"])))
                    idx.append((q, j))
            m = np.zeros((len(d["runs"]), max(len(r["msgs"]) for r in d["runs"])))
            empty = np.zeros_like(m, dtype=bool)
            cand = [[K.enc(tok, " " + its[d["runs"][q]["run"][0]]["v1"]), K.enc(tok, " " + its[d["runs"][q]["run"][0]]["v2"])]
                    for q, j in idx]
            lp = sequence_choices(model, tok, ids, cand, batch=cf["batch"])
            for (q, j), l_ in zip(idx, lp):
                m[q, j] = l_[0] - l_[1]; empty[q, j] = len(d["runs"][q]["msgs"][j]) == 0
            return dict(m=m, empty=empty)
        res[cond] = ctx.cached(f"read_{cond}", run)
    # belief probes (descriptive), unpushed dialogue conditions, agent A at events 0, 6, 12
    bel = {}
    for cond in ("C-opp", "C-half"):
        p = ctx.path(f"l/l4_{cond}.pt")
        if not os.path.exists(p):
            continue
        d = loadpt(p)
        rows, idx = [], []
        for q, rr in enumerate(d["runs"]):
            i, c, s = rr["run"]; it = its[i]
            q_ = PROBE_Q.format(attribute=it["attribute"], entity=it["entity"])
            hist = E.chat_sys_ids(tok, _sys(it, "A", primes(cond, c)[0]), ASK.format(attribute=it["attribute"], entity=it["entity"]))
            ev = {0: E.chat_sys_ids(tok, _sys(it, "A", primes(cond, c)[0]), q_)}
            for j, mm in enumerate(rr["msgs"]):
                if j % 2 == 0:
                    hist = hist + mm
                else:
                    incoming = SAYS.format(m=tok.decode(mm))
                    if j + 1 in (6, 12):
                        ev[j + 1] = hist + E.turn_suffix_ids(tok, incoming + "\n" + q_)
                    if j + 1 < len(rr["msgs"]):
                        hist = hist + E.turn_suffix_ids(tok, incoming)
            for e_, x in ev.items():
                rows.append(x); idx.append((q, e_, it))
        lp = sequence_choices(model, tok, rows, [[K.enc(tok, it["v1"]), K.enc(tok, it["v2"])] for (q, e_, it) in idx], batch=cf["batch"])
        bel[cond] = [dict(run=q, event=e_, m=float(l_[0] - l_[1])) for (q, e_, it), l_ in zip(idx, lp)]
    save(ctx, "l/l4_reads.pt", dict(stance=res, belief=bel), config=dict(frame=READ_U + " | " + READ_A, probe=PROBE_Q))

def _spins(m, items_of, c_split):
    """s = sign(m - c), c cross-fitted on odd / even items."""
    s = np.zeros_like(m)
    for par in (0, 1):
        rows = np.array([(i % 2) == par for i in items_of])
        s[rows] = np.where(m[rows] - c_split[par] >= 0, 1.0, -1.0)
    return s

def _updates(sA_self, sB_part, prime_sign, item, A_first=True, start=1):
    """Listener updates of A: messages of A after its first, prev = A's previous message spin, partner = the partner's latest."""
    obs = []
    for r in range(sA_self.shape[0]):
        for k in range(start, sA_self.shape[1]):
            obs.append((r, k, item[r], prime_sign[r], sA_self[r, k - 1], sB_part[r, k - 1] if A_first else sB_part[r, k]))
    return obs

def kin_fit(y, fe, X, n_fe, W=None):
    """Kinetic Ising fit (De Nobili's form): logit P(s = +1) = 2 (h0 + h_item + h_p c + J_s s_prev + J_c s_partner) -> J = beta / 2."""
    beta, a, F = S.logit_fe(y, fe, X, n_fe, weights=W)
    return beta / 2, a, F

def _design(series):
    """Listener updates of A: y, item, [prime, s_prev] and the (run, k) of the partner's latest message (B's message k - 1)."""
    sA, sB, prime, item = series
    rr, kk = np.meshgrid(np.arange(sA.shape[0]), np.arange(1, sA.shape[1]), indexing="ij")
    rr, kk = rr.ravel(), kk.ravel()
    return (sA[rr, kk] + 1) / 2, item[rr], np.stack([prime[rr], sA[rr, kk - 1]], 1), rr, kk

def jex_batch(series, item, perms_list, weights_list, chunk=256):
    """J_c (coefficient / 2) for many (partner permutation, run-weight) pairs in batched L-BFGS fits."""
    sA, sB, prime, it = series
    y, fe, X0, rr, kk = _design(series)
    out = []
    for c0 in range(0, len(perms_list), chunk):
        P_ = perms_list[c0:c0 + chunk]; Wl = weights_list[c0:c0 + chunk]
        X = np.stack([np.concatenate([X0, sB[p[rr], kk - 1][:, None]], 1) for p in P_])
        W = np.stack([w[rr] for w in Wl])
        dev = "cuda" if K._torch().cuda.is_available() else "cpu"
        out.append(S.logit_fe_batch(y, fe, X, int(it.max()) + 1, W=W, device=dev)[:, 2] / 2)
    return np.concatenate(out) if out else np.zeros(0)

def jex_boot_all(series, item, n_rep, rng_, Wb=None, w_runs=None, n_rep_b=20):
    """J_ex (real J_c minus the mean over within-item, across-configuration re-pairings) at the point and per bootstrap replicate
    (Wb [B, n_items] item multiplicities, or w_runs [B, n_runs] run weights for the two-level bootstrap)."""
    n = len(item)
    perms, wts, tags = [np.arange(n)], [np.ones(n)], [(-1, 0)]
    for _ in range(n_rep):
        perms.append(repair_perms(item, n, rng_)); wts.append(np.ones(n)); tags.append((-1, 1))
    B = 0 if (Wb is None and w_runs is None) else (len(Wb) if Wb is not None else len(w_runs))
    for b in range(B):
        wr = Wb[b][item] if Wb is not None else w_runs[b]
        perms.append(np.arange(n)); wts.append(wr); tags.append((b, 0))
        for _ in range(n_rep_b):
            perms.append(repair_perms(item, n, rng_)); wts.append(wr); tags.append((b, 1))
    J = jex_batch(series, item, perms, wts)
    tags = np.array(tags)
    pt_ = J[(tags[:, 0] == -1) & (tags[:, 1] == 0)][0] - J[(tags[:, 0] == -1) & (tags[:, 1] == 1)].mean()
    reps = np.array([J[(tags[:, 0] == b) & (tags[:, 1] == 0)][0] - np.nanmean(J[(tags[:, 0] == b) & (tags[:, 1] == 1)]) for b in range(B)])
    return float(pt_), reps, float(J[0])

def j_ex(series, item, n_rep, rng_, w_item=None):
    """J_ex at the point (or with item weights): (J_ex, real J_c)."""
    n = len(item)
    wr = np.ones(n) if w_item is None else np.asarray(w_item, float)[item]
    perms = [np.arange(n)] + [repair_perms(item, n, rng_) for _ in range(n_rep)]
    J = jex_batch(series, item, perms, [wr] * len(perms))
    return float(J[0] - np.nanmean(J[1:])), float(J[0])

def repair_perms(item, n, rng_):
    """Within-item, across-configuration derangement of runs (validity fix)."""
    perm = np.arange(n)
    for i in np.unique(item):
        idx = np.flatnonzero(item == i)
        if len(idx) > 1:
            perm[idx] = idx[S.derange(len(idx), rng_)]
    return perm

def _A_B(m, cond):
    """A's and B's message matrices (A: odd messages 1, 3, ..; B: even)."""
    return m[:, 0::2], m[:, 1::2]

def kinetic_series(spins, cond, cfgs, item, direction, replayed=None):
    """Align the partner's latest message with each listener update.

    The shared design indexes partner[:, k - 1] for listener message k.
    A hears B[k - 1], whereas B hears A[k]. A document reply k hears the
    replayed B[k], so its partner matrix has the same one-column shift.
    The first message has no listener self-history and is excluded.
    """
    if cond == "C-doc":
        if direction != "A_from_replay" or replayed is None:
            raise ValueError("C-doc has only A listening to the matched B replay")
        listener, partner = spins, replayed[:, 1:]
        prime = np.zeros(len(item))
    else:
        a, b = _A_B(spins, cond)
        if direction == "A_from_B":
            listener, partner, who = a, b, 0
        elif direction == "B_from_A":
            listener, partner, who = b, a[:, 1:], 1
        else:
            raise ValueError("unknown L4 kinetic direction")
        prime = np.array([{0: 0, 1: 1, 2: -1}[primes(cond, cc)[who]] for cc in cfgs])
    if listener.shape[0] != len(item) or partner.shape[0] != len(item) or partner.shape[1] < listener.shape[1] - 1:
        raise ValueError("incomplete L4 listener/partner message alignment")
    return listener, partner, prime, item


def kinetic_descriptor(series):
    """Descriptive kinetic parameters in logit = 2(h0 + hi + hp*c + Js*s + Jc*p)."""
    y, fe, X0, rr, kk = _design(series)
    if not len(y):
        return {"status": "blocked", "reason": "no listener updates with a previous own message"}
    X = np.column_stack([X0, series[1][rr, kk - 1]])
    beta, intercept, fields = S.logit_fe(y, fe, X, int(np.max(series[3])) + 1)
    return {"status": "descriptive", "h0": float(intercept / 2), "h_item": fields / 2,
            "h_prime": float(beta[0] / 2), "J_s": float(beta[1] / 2), "J_c": float(beta[2] / 2),
            "n_updates": len(y), "mean_log_likelihood": float(S.logit_ll(y, fe, X, beta, intercept, fields).mean()),
            "prime_varies": bool(np.ptp(X[:, 0])), "parameter_intervals": "not estimated; descriptive point fits"}


def descriptive_kinetics(spins, cfgs, item, n_rep, seed=4243):
    """Both live directions and the within-configuration seed-level null.

    Re-pair only the partner's seed within each (item, configuration). These
    point estimates are descriptive and do not replace the primary coupling
    statistic or its item-cluster bootstrap.
    """
    groups = np.unique(np.column_stack([item, cfgs]), axis=0, return_inverse=True)[1]
    rng = np.random.default_rng(seed)
    fits, seed_level = {}, {}
    for cond, values in spins.items():
        if cond == "C-solo":
            continue
        directions = ("A_from_replay",) if cond == "C-doc" else ("A_from_B", "B_from_A")
        fits[cond], seed_level[cond] = {}, {}
        for direction in directions:
            replayed = _A_B(spins["C-half"], "C-half")[1] if cond == "C-doc" else None
            series = kinetic_series(values, cond, cfgs, item, direction, replayed)
            fits[cond][direction] = kinetic_descriptor(series)
            if fits[cond][direction]["status"] == "blocked":
                seed_level[cond][direction] = fits[cond][direction].copy()
                continue
            if min(np.bincount(groups)) < 2:
                seed_level[cond][direction] = {"status": "blocked", "reason": "seed-level null needs two seeds per item/configuration"}
                continue
            excess, real = j_ex(series, groups, n_rep, rng)
            seed_level[cond][direction] = {"status": "descriptive", "J_ex": excess, "J_c": real,
                "mean_repaired_J_c": real - excess, "n_repair": n_rep,
                "repair": "derange seeds within each item and configuration",
                "interval": "not estimated; primary item-cluster bootstrap belongs to across-configuration J_ex"}
    return fits, seed_level


def l4_stats(ctx):
    """All L4 statistics (bootstrap over items; two-level over items and random sets for C-Rup) -> l/l4.json; X2 and X6 statistics in
    their own files when those conditions ran."""
    import os
    tok, model, cf, H, items = setup(ctx)
    its, _ = d1(ctx, tok)
    rd = loadpt(ctx.path("l/l4_reads.pt"))["stance"]
    cal = K.load_json(ctx.path("l/l4_cal.json"))
    R = runs(cf)
    item = np.array([i for i, c, s in R]); cfgs = np.array([c for i, c, s in R]); seed = np.array([s for i, c, s in R])
    nB = cf["nboot"]; nrep = 200 if not ctx.mock else 6; nrep_b = 20 if not ctx.mock else 2
    g = np.random.default_rng(4242)
    st = {}
    # spins: c cross-fitted on C-opp first messages (A msg 1, B msg 2); v1-primed: A in cfg 1, B in cfg 2
    mo = rd["C-opp"]["m"]
    first_v1 = np.concatenate([mo[cfgs == 1, 0], mo[cfgs == 2, 1]]); first_v2 = np.concatenate([mo[cfgs == 2, 0], mo[cfgs == 1, 1]])
    it1 = np.concatenate([item[cfgs == 1], item[cfgs == 2]]); it2 = np.concatenate([item[cfgs == 2], item[cfgs == 1]])
    c_split = {}
    for par in (0, 1):
        o = 1 - par
        c_split[par] = (first_v1[it1 % 2 == o].mean() + first_v2[it2 % 2 == o].mean()) / 2
    from r2_linearity import auc_cols
    auc = float(auc_cols(np.concatenate([first_v1, first_v2]), np.concatenate([np.ones(len(first_v1)), np.zeros(len(first_v2))]).astype(bool)))
    complete = {c: float(1 - rd[c]["empty"].any(1).mean()) for c in rd}
    st["g_duet"] = bool(auc >= 0.80 and min(complete.values()) >= 0.90)
    st["auroc_first"] = auc; st["complete"] = complete
    sp = {c: _spins(rd[c]["m"], item, c_split) for c in rd}
    W = S.boot_counts(cf["l4_items"], nB, seed=77)

    def series(c, partner_from=None):
        A_, B_ = _A_B(sp[c], c)
        pa = np.array([{0: 0, 1: 1, 2: -1}[primes(c, cc)[0]] for cc in cfgs])
        return (A_, B_ if partner_from is None else partner_from, pa, item)

    def jex_boot(c, partner=None, w_sets=None):
        return jex_boot_all(series(c, partner), item, nrep, g, Wb=W, n_rep_b=nrep_b)
    jh, jh_b, jh_real = jex_boot("C-half")
    st["J_ex"] = S.summ(jh, jh_b, margin=0.12)
    st["J_c"] = jh_real
    st["kinetic_descriptive"], st["seed_level_coupling"] = descriptive_kinetics(sp, cfgs, item, nrep)
    st["J_ex_seed_level"] = st["seed_level_coupling"]["C-half"]["A_from_B"]
    # held-out log-loss gain of J_c (item-level 4-fold)
    A_, B_, pa, _ = series("C-half")
    obs = [(r, k) for r in range(A_.shape[0]) for k in range(1, A_.shape[1])]
    fold = item % 4
    gain = np.zeros(len(obs))
    for f in range(4):
        tr = np.array([fold[r] != f for r, k in obs]); te = ~tr
        y = np.array([(A_[r, k] + 1) / 2 for r, k in obs]); fe = np.array([item[r] for r, k in obs])
        X1 = np.array([[pa[r], A_[r, k - 1]] for r, k in obs]); X2 = np.array([[pa[r], A_[r, k - 1], B_[r, k - 1]] for r, k in obs])
        if tr.sum() and te.sum():
            b1, a1, F1 = S.logit_fe(y[tr], fe[tr], X1[tr], int(item.max()) + 1)
            b2, a2, F2 = S.logit_fe(y[tr], fe[tr], X2[tr], int(item.max()) + 1)
            gain[te] = S.logit_ll(y[te], fe[te], X2[te], b2, a2, F2) - S.logit_ll(y[te], fe[te], X1[te], b1, a1, F1)
    oi = np.array([item[r] for r, k in obs])
    per_item = np.array([gain[oi == i].sum() for i in range(cf["l4_items"])]); cnt = np.array([(oi == i).sum() for i in range(cf["l4_items"])])
    st["ll_gain"] = S.summ(per_item.sum() / max(cnt.sum(), 1), (W @ per_item) / np.maximum(W @ cnt, 1))
    # C-doc (replayed partner series)
    if "C-doc" in sp:
        Ad = sp["C-doc"]; Bh = _A_B(sp["C-half"], "C-half")[1]
        ser = (Ad, Bh[:, 1:] if Bh.shape[1] > Ad.shape[1] - 1 else Bh, np.zeros(len(item)), item)
        jd, jd_b, _ = jex_boot_all(ser, item, nrep, g, Wb=W, n_rep_b=nrep_b)
        st["J_ex_doc"] = S.summ(jd, jd_b)
        st["J_ex_minus_doc"] = S.summ(jh - jd, jh_b - jd_b, margin=0.16)
    else:
        st["J_ex_minus_doc"] = S.summ(float("nan"), [np.nan])
    # entity line (C-opp): consensus on the last messages (11, 12), agreement excess against within-item cross-configuration re-pairs
    Ao, Bo = _A_B(sp["C-opp"], "C-opp")
    cons_run = (Ao[:, -1] == Bo[:, -1]).astype(float)

    def per_item_mean(v):
        return np.array([v[item == i].mean() for i in range(cf["l4_items"])])
    rep_cons = np.mean([(Ao[:, -1] == Bo[repair_perms(item, len(item), g), -1]).astype(float) for _ in range(nrep)], 0)
    ci_ = per_item_mean(cons_run); cr_ = per_item_mean(rep_cons)
    st["consensus"] = S.summ(ci_.mean(), (W @ ci_) / W.sum(1))
    st["agree_excess"] = S.summ(ci_.mean() - cr_.mean(), (W @ (ci_ - cr_)) / W.sum(1))
    mA, mB = _A_B(rd["C-opp"]["m"], "C-opp")
    late = per_item_mean(np.abs(mA[:, -1] - mB[:, -1])); early = per_item_mean(np.abs(mA[:, 0] - mB[:, 0]))
    st["gap_ratio"] = S.summ(late.mean() / max(early.mean(), 1e-12), (W @ late) / np.maximum(W @ early, 1e-12))
    # pair macro Delta Psi (C-opp primary, C-half descriptive)
    for c in ("C-opp", "C-half"):
        mA_, mB_ = _A_B(rd[c]["m"], c)

        def psi(Bm):
            V = mA_ + Bm
            x1, x2, x3, yv = [], [], [], []
            for r_ in range(1, V.shape[1]):
                x1 += V[:, r_ - 1].tolist(); x2 += mA_[:, r_ - 1].tolist(); x3 += Bm[:, r_ - 1].tolist(); yv += V[:, r_].tolist()
            return S.gc_mi(x1, yv) - S.gc_mi(x2, yv) - S.gc_mi(x3, yv)
        real = psi(mB_)
        null = np.mean([psi(mB_[repair_perms(item, len(item), g)]) for _ in range(nrep)])
        bs = []
        for b in range(min(nB, 400)):
            ii = np.concatenate([np.flatnonzero(item == i) for i in np.repeat(np.arange(cf["l4_items"]), W[b].astype(int))])
            if len(ii) < 4:
                bs.append(np.nan); continue
            bs.append(_psi_arr(mA_[ii], mB_[ii]) - null)
        st[f"dpsi_{c}"] = S.summ(real - null, bs, margin=0.025)
    st["dpsi"] = st["dpsi_C-opp"]
    # H line: DD = J_ex(C-Hup) - J_ex(C-Rup); two-level bootstrap (items, then the 4 random sets)
    if "C-Hup" in sp and "C-Rup" in sp:
        set_of = np.array([(s + c) % len(cal["sets"]) for c, s in zip(cfgs, seed)])
        ju, ju_b, _ = jex_boot("C-Hup")
        wruns = np.stack([W[b][item] * np.bincount(g.integers(0, len(cal["sets"]), len(cal["sets"])), minlength=len(cal["sets"]))[set_of]
                          for b in range(nB)])
        jr, jr_b, _ = jex_boot_all(series("C-Rup"), item, nrep, g, w_runs=wruns, n_rep_b=nrep_b)
        st["DD"] = S.summ(ju - jr, ju_b - jr_b, margin=0.17)
        st["rup_minus_half"] = S.summ(jr - jh, jr_b - jh_b)
        st["hup_minus_half"] = S.summ(ju - jh, ju_b - jh_b, margin=0.17)
        med = np.median(np.abs(np.concatenate([first_v1, first_v2])))
        cu = (np.abs(rd["C-Hup"]["m"][:, 0]) >= med).astype(float); crr = (np.abs(rd["C-Rup"]["m"][:, 0]) >= med).astype(float)
        cu_i, cr_i = per_item_mean(cu), per_item_mean(crr)
        st["commit_diff"] = S.summ(cu.mean() - crr.mean(), (W @ (cu_i - cr_i)) / W.sum(1))
        st["g_manip"] = bool(RU_LB(st["commit_diff"]))
        st["dmg_ok"] = bool(cal["up"]["g_dmg"])
    else:
        st["g_manip"] = False
    # catfish carry (B's stance signed toward B's priming; C-opp against C-half)
    kb = {}
    for c in ("C-opp", "C-half"):
        mB_ = _A_B(rd[c]["m"], c)[1]
        sgn = np.array([1.0 if primes(c, cc)[1] == 1 else -1.0 for cc in cfgs])
        sig = mB_ * sgn[:, None]
        s1, s6 = per_item_mean(sig[:, 0]), per_item_mean(sig[:, -1])
        b1 = (W @ s1) / W.sum(1)
        d1s = S.summ(s1.mean(), b1)
        defined = RU_LB(d1s) or (isinstance(d1s["hi"], float) and d1s["hi"] < 0)
        kb[c] = (defined, np.log(np.maximum(s6.mean() / s1.mean(), 1e-12)) if s1.mean() and s6.mean() / s1.mean() > 0 else np.nan,
                 np.log(np.maximum(((W @ s6) / W.sum(1)) / b1, 1e-12)))
    st["catfish"] = S.summ(kb["C-opp"][1] - kb["C-half"][1], kb["C-opp"][2] - kb["C-half"][2]) if (kb["C-opp"][0] and kb["C-half"][0]) else None
    wj(ctx, "l/l4.json", st, config=dict(nboot=nB, n_repair=nrep, repair="within item, across configurations (validity fix)"))

def _psi_arr(mA_, mB_):
    V = mA_ + mB_
    x1, x2, x3, yv = [], [], [], []
    for r_ in range(1, V.shape[1]):
        x1 += V[:, r_ - 1].tolist(); x2 += mA_[:, r_ - 1].tolist(); x3 += mB_[:, r_ - 1].tolist(); yv += V[:, r_].tolist()
    return S.gc_mi(x1, yv) - S.gc_mi(x2, yv) - S.gc_mi(x3, yv)

def RU_LB(s):
    return isinstance(s.get("lo"), float) and s["lo"] > 0

def run(ctx):
    """Complete five-condition L4 core; calibration gate applied by G L-ANA."""
    cx = _Stage(ctx)
    pop = load_pop(cx)
    cf = config(cx)
    gain_ready = (pop.get("status") == "measured" and bool(pop.get("P")) and
                  len(pop.get("CTRL", [])) >= (cf["l4_sets"] if ctx.mock else 24))
    gain_reason = "own population or distinct matched gain controls unavailable"
    tok, _ = cx.model()
    items, _ = d1(cx, tok)
    tokens = [{"item": i, "value_tokens": [K.enc(tok, " " + r[k]) for k in ("v1", "v2")],
               "belief_tokens": [K.enc(tok, r[k]) for k in ("v1", "v2")]}
              for i, r in enumerate(items)]
    ctx.write_json("l4_tokenization.json", {"items": tokens, "readout": "summed log probabilities of complete value tokens",
        "distinct": all(r["value_tokens"][0] != r["value_tokens"][1] for r in tokens),
        "first_token_distinct": all(r["value_tokens"][0][0] != r["value_tokens"][1][0] for r in tokens)})
    if gain_ready:
        try:
            l4_cal(cx)
        except (K.NotRun, ValueError) as exc:
            gain_ready, gain_reason = False, str(exc)
    if not gain_ready:
        wj(cx, "l/l4_cal.json", {"sets": [], "up": {"g_dmg": False}, "reason": gain_reason})
    conditions = ["C-opp", "C-half", "C-doc"] + (["C-Hup", "C-Rup"] if gain_ready else [])
    for cond in conditions:
        l4_run(cx, cond)
    l4_reads(cx)
    l4_stats(cx)
    coverage = {"L4": {"status": "measured"}, "L4_H_gain": {"status": "measured" if gain_ready else "blocked",
                                                             "reason": "" if gain_ready else gain_reason}}
    ctx.write_json("l4_coverage.json", {"status":"done" if gain_ready else "partial", "mock":ctx.mock,
        "conditions":conditions, "coverage": coverage,
        "protocol":"L4 core; item bootstrap, within-item cross-configuration repair, two-level gain controls",
        "native_chat":True, "calibration":"scientific verdict requires G l_calibrate k/l"})
    return {"status":"done" if gain_ready else "partial", "coverage": coverage}
