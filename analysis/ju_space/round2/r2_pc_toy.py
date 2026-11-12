"""Round 2, PC toy: predictive coding against backprop (plan 5.1 C-T7), the AT5 cross-model pairs with a PC net (3.2 AT5), and the gates
G-meta / G-PC(W) read from the round-1 PC outputs (2.2; section 6 G35-G37); session r2_pc of section 7; CLI of 9.3. Implementer 7.

Usage   python r2_pc_toy.py <out_dir> <stage>        stage: gates, t7, cross_pc, all
        QUICK=1 python r2_pc_toy.py <work_dir> selftest    code test: builds fake round-1 PC folders with pc_meta QUICK runs and
                                                         pc_summary, fake r2_t1 AT5 nets, then runs every stage
        env DEVICE (cuda / cpu), QUICK, R1PC = folder with the copied round-1 PC files (<R1PC>/gpu*/pc/jobs/*.json, gpu*/pc/models/*.pt,
        gpu*/pc_summary_gpu*.json, gpu*/pc_expected_gpu*.json, gpu*/pc_missing_gpu*.json, gpu*/shard_done.json; default <out_dir>/r1),
        R2T1 = output folder of session r2_t1 (rgi_toy/models/at5_*.pt; cross_pc only; default <out_dir>/r2_t1)
Outputs <out_dir>/pc_toy/gates.json, pc_toy/t7.json, pc_toy/cross_pc/w<ws>.json (written atomically; a stage whose output exists is
        skipped; per-width partial results in pc_toy/partial/).
No gate decides what runs (critique v2-11): t7 runs both widths and cross_pc every PC width that has checkpoints. The 'gates' stage
evaluates G-meta and G-PC(W) on the union of the round-1 pc_summary rows with pc_analyze.gates (round-1 code, imported unchanged) for the
record; r2_ana applies them (with G35's fallback on r2_t3's round-2 BP-muP W 128 nets, g_meta_fallback()) and the C-T7 rule
(ct7_family(), Holm over the family at the G-PC widths). Files are found by job name across all shards (OD-G9).
PC: round-1 PC-muP nets are evaluated by the feedforward pass; every PC search uses pc_meta.pc_update's local update vectorised in
r2_toy_common (closed-form relaxation, T 20, beta 0.1, eta_x = 1 / pc_meta.lam_max on the net's pretraining batch, fixed per net)."""
from __future__ import annotations

import glob
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import r2_unlearn_toy as UT  # noqa: E402

TC = UT.TC
M = TC.M
P, NPRE, NNEW = TC.P, TC.NPRE, TC.NNEW
DEV, QUICK = TC.DEV, TC.QUICK
F32, F64 = torch.float32, torch.float64
SUB = "pc_toy"
STAGES = ("gates", "t7", "cross_pc")
SHARD_RE = re.compile(r"gpu\d+$")
SEEDS = UT.SEEDS                                                # round-1 seeds = worlds 0-4
T7_WIDTHS = (64, 128) if QUICK else (128, 256)                 # C-T7 (QUICK: the fixture's widths)
PC_WIDTHS = (64, 128) if QUICK else (64, 128, 256, 512)        # AT5 / G37: PC-muP widths of round 1
BP_T7_WIDTHS = T7_WIDTHS                                       # C-T7's BP side: R1 BP-muP checkpoints at the same widths
T7_ROUNDS = UT.T5_ROUNDS                                       # s_0 .. s_24
RELAX_T = (20, 100)                                            # C-T7 (c)
B_SHORT, B_LONG = UT.BUDGETS
MIN_PAIRED = 2 if QUICK else 4                                 # G37: fewer than 4 paired seeds -> 'not testable'
PC_ETA_SEED = 1000                                             # eta_x power iteration start: torch.manual_seed(1000 + ws)
T3_MAXW = 64 if QUICK else 256                                 # AT5: T3 h2 stitching at W <= 256 (sender and receiver; BP receiver)
CLAIM_ALPHA, CLAIM_MIN = 0.05, 0.10                            # C-T7 rule (critique v2-12h)
FT_EVAL = tuple(TC.FT_EVAL)
read_json, write_json, tci = UT.read_json, UT.write_json, UT.tci


def job_name(arm, W, s, K=NPRE, gamma=0.0, wd=1.0):
    """pc_meta.main's / run_shard.job_name's file name of a round-1 job (PC: T 20, beta 0.1; lr 1e-3)."""
    return (f"{arm}_W{W}_s{s}_K{K}_g{gamma:g}_wd{wd:g}" + ("_T20_beta0.1" if arm == "pc_mup" else "") + ("_quick" if QUICK else ""))


def r1_root(out):
    return os.environ.get("R1PC") or os.path.join(out, "r1")


def pdir(out, *parts):
    return os.path.join(out, SUB, *parts)


# ================================================================================================================ round-1 files (OD-G9)
class R1PC:
    """The copied round-1 PC folders <root>/gpu*/. find() locates a job's model or JSON by name across all shards; duplicates (OD-G9):
    the shard whose pc_expected_<shard>.json lists the job, then the one with shard_done.json, then the most recent file; every choice
    is logged."""

    def __init__(self, root):
        self.root = root; self.log = []
        self.shards = sorted(d for d in (os.listdir(root) if os.path.isdir(root) else []) if SHARD_RE.match(d) and os.path.isdir(os.path.join(root, d)))
        self.expected, self.done = {}, {}
        for sh in self.shards:
            p = os.path.join(root, sh, f"pc_expected_{sh}.json")
            self.expected[sh] = set(read_json(p)) if os.path.exists(p) else set()
            self.done[sh] = os.path.exists(os.path.join(root, sh, "shard_done.json"))

    def pick(self, name, cands):
        """cands: list of (shard, path)."""
        if len(cands) <= 1:
            return cands[0] if cands else None
        c1 = [c for c in cands if name in self.expected.get(c[0], set())] or cands
        c2 = [c for c in c1 if self.done.get(c[0])] or c1
        ch = max(c2, key=lambda c: os.path.getmtime(c[1]))
        self.log.append(dict(name=name, candidates=[c[0] for c in cands], chosen=ch[0], rule="pc_expected > shard_done > most recent"))
        return ch

    def find(self, name, kind="models"):
        ext = ".pt" if kind == "models" else ".json"
        cands = [(os.path.relpath(p, self.root).split(os.sep)[0], p) for p in glob.glob(os.path.join(self.root, "gpu*", "pc", kind, name + ext))]
        cands = [c for c in cands if SHARD_RE.match(c[0])]
        c = self.pick(name, cands)
        return None if c is None else c[1]

    def job(self, name):
        p = self.find(name, "jobs")
        return read_json(p) if p else None

    def summary_rows(self):
        """Union of the rows of gpu*/pc_summary_gpu*.json (PCS), one row per job name (duplicates resolved as find())."""
        by = {}
        for sh in self.shards:
            for p in glob.glob(os.path.join(self.root, sh, "pc_summary_gpu*.json")):
                for r in read_json(p):
                    by.setdefault(r.get("name"), []).append((sh, p, r))
        rows = []
        for name, lst in sorted(by.items(), key=lambda kv: str(kv[0])):
            ch = self.pick(name, [(sh, p) for sh, p, _ in lst])
            rows.append(next(r for sh, p, r in lst if (sh, p) == ch))
        return rows

    def missing_lists(self):
        out = {}
        for sh in self.shards:
            p = os.path.join(self.root, sh, f"pc_missing_{sh}.json")
            if os.path.exists(p):
                out[sh] = read_json(p)
        return out


def _pa():
    import pc_analyze                     # round-1 code, imported unchanged (functions only)
    return pc_analyze


# ================================================================================================================ gates (G35-G37)
def gates_from_rows(R):
    """G-meta and G-PC(W) (2.2, G35, G36) on the union of round-1 pc_summary rows R with pc_analyze.gates: G-meta = G1 and G2;
    G-PC(W) = G-meta and G0_pc_mup and G3['pc_mup_W<W>'] and G4[W] and G5[W] and >= 4 seeds with level1 (rows pc_mup, W, K 12, gamma 0,
    wd 1); a width failing only G5 is 'BP-equivalent'. G35's fallback flag: no BP-muP W 128 main row in PCS."""
    PA = _pa(); G = PA.gates(R); gm = bool(G["G1"]["pass_"] and G["G2"]["pass_"]); mr = PA.main_rows(R, "pc_mup"); gpc = {}
    for W in sorted({r["W"] for r in mr}):
        n1 = sum(1 for r in mr if r["W"] == W and r.get("level1"))
        c = dict(G_meta=gm, G0_pc_mup=bool(G["G0_pc_mup"]["pass_"]), G3=bool(G["G3"].get(f"pc_mup_W{W}", {}).get("pass_", False)),
                 G4=bool(G["G4"].get(W, {}).get("pass_", False)), G5=bool(G["G5"].get(W, {}).get("pass_", False)), level1_4_of_5=n1 >= 4)
        gpc[W] = dict(pass_=all(c.values()), conds=c, n_level1=n1, bp_equivalent=(not c["G5"]) and all(v for k, v in c.items() if k != "G5"))
    fb = not any(r["W"] == 128 for r in PA.main_rows(R, "bp_mup"))
    return dict(G=G, G_meta=gm, G_PC=gpc, G35_fallback_needed=fb)


def g_meta_fallback(t1, W=128):
    """G35's fallback (r2_ana): G1 and G2 by pc_analyze's formulas on r2_t3's round-2 BP-muP W 128 nets (unlearn_toy/t1.json, the
    uncut nets' level 1 and B2(8)); labelled 'gate from round 2'."""
    PA = _pa(); c = t1["cells"][f"bp_mup_W{W}"]; lv = [bool(x) for x in c["base"]["level1"]]; b2 = c["base"]["B2"]["8"]
    n1 = sum(lv); b = PA.ci([v for v, l in zip(b2, lv) if l])
    g1, g2 = n1 >= 4, bool(b["mean"] is not None and b["mean"] >= 0.10 and b["lo"] is not None and b["lo"] > 0.03)
    return dict(G1=dict(pass_=g1, level1_seeds=n1), G2=dict(pass_=g2, **b), pass_=g1 and g2, label="gate from round 2")


def stage_gates(out):
    """Gates G-meta and G-PC(W) from the round-1 PC summaries, the G37 model inventory and the missing-job lists (record only)."""
    r1 = R1PC(r1_root(out)); R = r1.summary_rows(); res = gates_from_rows(R)
    inv = {}
    for arm, widths in (("pc_mup", PC_WIDTHS), ("bp_mup", BP_T7_WIDTHS)):
        for W in widths:
            inv[f"{arm}_W{W}"] = {str(s): bool(r1.find(job_name(arm, W, s))) for s in SEEDS}
    exp = set().union(*r1.expected.values()) if r1.expected else set()
    res.update(n_rows=len(R), shards=r1.shards, shard_done=r1.done, inventory_G37=inv,
               missing_jobs=sorted(exp - {r.get("name") for r in R}) if exp else "no expected-job lists found", pc_missing=r1.missing_lists(),
               finder=r1.log, note="record only: r2_pc runs every width (critique v2-11); r2_ana applies G-PC (G35 fallback: g_meta_fallback)")
    print(f"  gates: {len(R)} rows, G-meta {res['G_meta']}, G-PC {{{', '.join(f'{W}: {v['pass_']}' for W, v in res['G_PC'].items())}}}", flush=True)
    return res


# ================================================================================================================ C-T7
def t7_unsearch(nets, ws_of, eta, key):
    """C-T7 (b): C-T5 with the PC protocol (arms col and rand; every search 300 steps, the removed direction = the column at the best
    validation step of that 300-step search; PC: the local update, BP twin: backprop), then 3,000-step searches at s_(r*_300) (the first
    r with A_300(s_r) <= 0.11) and at s_24 with the removed set of that round."""
    rec = UT.unsearch(nets, ws_of, ("col", "rand"), T7_ROUNDS, B_SHORT, (B_SHORT,), B_SHORT, key, pc_eta=eta, side=False, keep_dirs=True)
    A = np.asarray(rec["A"][str(B_SHORT)], float); R = A.shape[1]; rs = UT.rstar(A); dirs = rec.pop("dirs")
    ch, rd = [], []
    for i in range(len(rec["chains"])):
        for r in sorted({int(rs[i]), R - 1} if rs[i] < R else {R - 1}):
            ch.append(i); rd.append(r)
    cb = np.array([rec["chains"][i][1] for i in ch]); cj = np.array([rec["chains"][i][2] for i in ch])
    sh, ho, va = UT.new_sets(ws_of, cb, cj); rem = torch.zeros(len(ch), nets.W, max(1, R - 1), dtype=F32)
    for k, (i, r) in enumerate(zip(ch, rd)):
        rem[k, :, :r] = dirs[i][:, :r]
    pc = None if eta is None else dict(eta=np.asarray(eta, float)[cb])
    res = UT.search(nets, cb, sh, {"ho": ho}, val=va, steps=B_LONG, budgets=(B_SHORT, B_LONG), remove=rem, pc=pc)
    rec["rstar_300"] = rs
    rec["long"] = dict(chain=ch, round=rd, A_short=UT.best_ho(res, B_SHORT), A_long=UT.best_ho(res, B_LONG))
    return rec


@torch.no_grad()
def relax_test(nets, tasks, base_feats, tb, eta, Ts=RELAX_T):
    """C-T7 (c) 'relaxation restores task information' (PC nets): all old tasks x 289 pairs with the output clamped to the true label,
    the erasure (h1 or h2) inside the energy, value nodes relaxed for T steps (TC.pc_relax: closed form, eta_x of the net); the
    task-identity and pooled-answer linear probes trained on the unerased feedforward h2 (probe halves of W8) read the relaxed h2 (test
    halves); Delta_relax = accuracy on the relaxed h2 minus on the erased feedforward h2; Delta_rand = the same under the damage-matched
    random erasure (mean over its draws); the inference residual (end / start gradient norm) per relaxation. contrast = Delta_relax -
    Delta_rand (task identity; the C-T7 family item)."""
    B, W, n = nets.B, nets.W, tasks.n; allv = np.arange(B); out = {}
    for layer in ("h1", "h2"):
        er = tb[layer]["eraser"]; kU, kc, kl = UT.LAYER_KEYS[layer]
        items = [{kU: er["Q"][b], kc: er["mu"][b]} for b in allv] + [{kU: er["dm_R"][d], kc: er["mu"][b], kl: er["dm_lam"]} for b in allv for d in range(UT.NDRAW)]
        var = np.concatenate([allv, np.repeat(allv, UT.NDRAW)]); V = len(var); ops = UT.mk_ops(items, W)
        vb, tp, x, y, lab, _ = UT._chains(tasks, var, 0, P * P); cols = UT._chain_cols(nets, tasks, vb, tp); oc = UT._expand(ops, V, n)
        h_ff = UT._hidden(nets, vb, cols, x, y, oc, "h2")[:, 144:]
        res_l = {}
        for T in Ts:
            per = max(1, TC.CHUNK // max(1, P * P * W * 12)); hs, rat = [], []
            for a in range(0, len(vb), per):
                b_ = min(len(vb), a + per)
                _, h2r, r_ = TC.pc_relax(nets, vb[a:b_], cols[a:b_], (x[a:b_], y[a:b_], lab[a:b_]), np.asarray(eta, float)[vb[a:b_]], T=T,
                                         ops=UT.ops_slice(oc, slice(a, b_)))
                hs.append(h2r[:, 144:]); rat.append(torch.as_tensor(r_).reshape(-1).cpu())
            h_rel = torch.cat(hs); ratio = torch.cat(rat).numpy().reshape(V, n).mean(1)
            acc = {}
            for nm, H in (("relaxed", h_rel), ("ff", h_ff)):
                f = UT.probe_feats(nets, tasks, var, "h2", states=H)
                acc[nm] = dict(task=TC.ridge_probe(base_feats["Htr"][var], base_feats["task_tr"][var], f["Hte"], f["task_te"], n).cpu().numpy(),
                               ans=TC.ridge_probe(base_feats["Htr"][var], base_feats["ans_tr"][var], f["Hte"], f["ans_te"], P).cpu().numpy())
            d = {}
            for pr in ("task", "ans"):
                dl = acc["relaxed"][pr] - acc["ff"][pr]
                d[pr] = dict(relaxed=acc["relaxed"][pr][:B], ff=acc["ff"][pr][:B], delta_relax=dl[:B], delta_rand=dl[B:].reshape(B, UT.NDRAW).mean(1),
                             relaxed_rand=acc["relaxed"][pr][B:].reshape(B, UT.NDRAW).mean(1), ff_rand=acc["ff"][pr][B:].reshape(B, UT.NDRAW).mean(1))
            d["contrast"] = d["task"]["delta_relax"] - d["task"]["delta_rand"]
            d["ratio"] = ratio[:B]; d["ratio_rand"] = ratio[B:].reshape(B, UT.NDRAW).mean(1)
            res_l[str(T)] = d
        out[layer] = res_l
    return out


def r1_job_fields(job):
    if not job:
        return None
    ft = job.get("finetune") or {}; f8 = ft.get("8") or ft.get(8) or {}
    return dict(first95=job.get("first95"), checkpoint=job.get("checkpoint"), level1=job.get("level1"), final_heldout=job.get("final_heldout"),
                B2_8=f8.get("B2"), delta_h2=((job.get("delta") or {}).get("h2") or {}).get("delta"))


def t7_width(r1, W):
    """C-T7 at one width: PC side on the round-1 PC-muP checkpoints, BP side on the round-1 BP-muP checkpoints of the same seeds (a seed
    enters only if both nets exist, G37)."""
    pres = [dict(seed=s, pc=r1.find(job_name("pc_mup", W, s)), bp=r1.find(job_name("bp_mup", W, s))) for s in SEEDS]
    ok = [d for d in pres if d["pc"] and d["bp"]]
    info = dict(W=W, seeds_present=[d["seed"] for d in ok], seeds_missing=[d["seed"] for d in pres if not (d["pc"] and d["bp"])],
                enough_paired_seeds=len(ok) >= MIN_PAIRED, min_paired=MIN_PAIRED)
    if not ok:
        return dict(info, status="not run (no seed with both round-1 nets, G37)")
    ws = [d["seed"] for d in ok]
    pcn = TC.load_pc_state_dicts([d["pc"] for d in ok]); bpn = TC.load_pc_state_dicts([d["bp"] for d in ok])
    eta = np.array([TC.pc_eta(pcn, i, *TC.r1_training_batch(w), seed=PC_ETA_SEED + w) for i, w in enumerate(ws)])
    info.update(eta=eta, pc_nets=[dict(name=os.path.basename(d["pc"])[:-3], checksum=TC.weight_checksum(pcn, i), r1_job=r1_job_fields(r1.job(job_name("pc_mup", W, s))))
                                  for i, (d, s) in enumerate(zip(ok, ws))],
                bp_nets=[dict(name=os.path.basename(d["bp"])[:-3], checksum=TC.weight_checksum(bpn, i), r1_job=r1_job_fields(r1.job(job_name("bp_mup", W, s))))
                         for i, (d, s) in enumerate(zip(ok, ws))])
    tasks = UT.Tasks.w1(ws); key = f"t7|W{W}"
    for side, nets, et in (("pc", pcn, eta), ("bp", bpn, None)):
        t0 = time.time(); base = UT.base_measures(nets, tasks, ws, relearn=False, new=False, b2=False)
        tb, _ = UT.t4b_layers(nets, tasks, base, key, pc_eta=et)                        # (a) T4b (PC: eraser inside the energy)
        us = t7_unsearch(nets, ws, et, key)                                              # (b) C-T5, PC protocol (BP: the PC-matched twin)
        d = dict(base=base, unsearch=us)
        if side == "pc":
            d["relax"] = relax_test(nets, tasks, UT.probe_feats(nets, tasks, np.arange(nets.B), "h2"), tb, et)   # (c)
        d["t4b"] = UT.strip(tb); info[side] = d
        print(f"  t7 W{W} {side}: fwd acc h1 {np.round(tb['h1']['acc'], 3).tolist()} h2 {np.round(tb['h2']['acc'], 3).tolist()} ({time.time() - t0:.0f}s)", flush=True)
    return info


def _seed_vals(us, arm, r, which):
    """Per-seed (mean over the 4 tasks) accuracy of arm at round r: which 'short' = the 300-step protocol value A_300(s_r), 'long' =
    the 3,000-step search at that round (from 'long')."""
    ks = str(int(us["budgets"][0])); ch = us["chains"]; R = len(us["A"][ks][0]); r = min(r, R - 1); seeds = sorted({b for _, b, _ in ch})
    if which == "short":
        A = np.array(us["A"][ks], float)
        return np.array([np.mean([A[i, r] for i, (a, b, j) in enumerate(ch) if a == arm and b == s]) for s in seeds])
    L = us["long"]; vals = {}
    for k, (i, rr) in enumerate(zip(L["chain"], L["round"])):
        a, b, j = ch[i]
        if a == arm and rr == r:
            vals.setdefault(b, []).append(L["A_long"][k])
    return np.array([np.mean(vals[s]) if s in vals else np.nan for s in seeds])


def ct7_items(t7, widths=None):
    """The C-T7 family (critique v2-12h) per width: per-seed paired PC - BP differences of forward accuracy after erasure (h1, h2),
    relearn_300 and relearn_3000 under erasure (h1, h2), A_300 at s_4, s_12, s_24 and A_3000 at s_24 (col arm), and the PC-only contrast
    Delta_relax - Delta_rand (task-identity probe, T 20 and T 100; NOTE: both erasure layers enter). Returns [(W, item, values)]."""
    items = []; bs, bl = UT.stage_budgets(t7)
    for w in t7["widths"].values():
        if "pc" not in w or (widths is not None and w["W"] not in widths):
            continue
        W, pc, bp = w["W"], w["pc"], w["bp"]
        for L in ("h1", "h2"):
            items.append((W, f"fwd_acc_{L}", np.asarray(pc["t4b"][L]["acc"], float) - np.asarray(bp["t4b"][L]["acc"], float)))
            for b_ in (bs, bl):
                items.append((W, f"relearn_{b_}_{L}", np.asarray(pc["t4b"][L]["relearn"][b_], float) - np.asarray(bp["t4b"][L]["relearn"][b_], float)))
        for r, b_, wh in ((4, bs, "short"), (12, bs, "short"), (24, bs, "short"), (24, bl, "long")):
            items.append((W, f"A{b_}_s{r}_col", _seed_vals(pc["unsearch"], "col", r, wh) - _seed_vals(bp["unsearch"], "col", r, wh)))
        for L in ("h1", "h2"):
            for T in RELAX_T:
                items.append((W, f"relax_contrast_T{T}_{L}", np.asarray(pc["relax"][L][str(T)]["contrast"], float)))
    return items


def ct7_family(t7, widths=None, alpha=CLAIM_ALPHA, min_effect=CLAIM_MIN):
    """C-T7 rule: two-sided paired t tests (df = n - 1) of every family item, Holm-adjusted over the whole family at 0.05; a claim needs
    a Holm-adjusted p <= 0.05 and |mean| >= 0.10; unadjusted t 95 % intervals reported. widths: the G-PC widths (r2_ana); None = every
    width that ran (preview). Widths with fewer paired seeds than MIN_PAIRED are 'not testable'."""
    from scipy import stats
    its = ct7_items(t7, widths); rows = []
    for W, name, v in its:
        n_ok = len(t7["widths"][f"W{W}"]["seeds_present"])
        v = np.asarray(v, float); v = v[np.isfinite(v)]; n = len(v)
        if n_ok < MIN_PAIRED or n < 2:
            rows.append(dict(W=W, item=name, n=n, testable=False)); continue
        m, sd = float(v.mean()), float(v.std(ddof=1)); se = sd / math.sqrt(n)
        t = m / se if se > 0 else (math.inf if m != 0 else 0.0); p = float(2 * stats.t.sf(abs(t), n - 1)) if math.isfinite(t) else 0.0
        rows.append(dict(W=W, item=name, n=n, testable=True, mean=m, sd=sd, t=t, p=p, **{k: v_ for k, v_ in tci(v).items() if k in ("lo", "hi")}))
    tst = [r for r in rows if r["testable"]]; order = sorted(range(len(tst)), key=lambda i: tst[i]["p"]); mm = len(tst); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (mm - rank) * tst[i]["p"])); tst[i]["p_holm"] = run
    for r in tst:
        r["claim"] = bool(r["p_holm"] <= alpha and abs(r["mean"]) >= min_effect)
    return dict(family_size=mm, widths=sorted({r["W"] for r in rows}), items=rows,
                claims=[f"W{r['W']} {r['item']}" for r in tst if r["claim"]],
                note="preview over every width that ran; r2_ana restricts the family to the G-PC widths (G36) and no 'at equilibrium' claim")


def stage_t7(out):
    r1 = R1PC(r1_root(out)); res = dict(widths={})
    for W in T7_WIDTHS:
        res["widths"][f"W{W}"] = UT.cached(out, f"t7_W{W}", lambda W=W: t7_width(r1, W), sub=SUB)
    res["finder"] = r1.log; res["rule_preview"] = ct7_family(res)
    return res


# ================================================================================================================ AT5 pairs with a PC net
def _bp_groups(root):
    """r2_t1's AT5 BP nets (rgi_toy/models/at5_*.pt, TC.save_nets files), grouped by architecture -> [(Nets, [meta])]."""
    files = sorted(glob.glob(os.path.join(root, "rgi_toy", "models", "at5_*.pt")))
    by = {}
    for p in files:
        o = torch.load(p, map_location="cpu", weights_only=False); a = o["arch"]
        by.setdefault((a["kind"], a["W"], a["depth"], a["act"], a["param"]), []).append(p)
    out = []
    for key, paths in sorted(by.items()):
        nets = TC.load_nets(paths)
        out.append((nets, [dict(m, path=p) for m, p in zip(nets.meta, paths)]))
    return out


def _model_info(meta, kind):
    s = meta.get("spec", {}); row = meta.get("row") or {}
    return dict(name=meta.get("name"), kind=kind, arm=s.get("arm"), W=int(s.get("W")), init=int(s.get("init", 0)), ws=int(s.get("ws")),
                level1=row.get("level1"), first95=row.get("first95"), checkpoint=row.get("checkpoint"), eval_point=row.get("eval_point"),
                checksum=row.get("checksum"))


def _free_cols(nets, idx, ws_of, eta=None):
    """AT1's free improvement (the B2 column search: slot column from zero, 8 shots, Adam lr 1e-2, FT_STEPS steps, held-out 281 pairs at
    FT_EVAL; PC nets: the PC local update) for nets idx -> (v [n, 4, W], acc [n, 4, E])."""
    net_of = np.repeat(np.asarray(idx), NNEW); jj = np.tile(np.arange(NNEW), len(idx)); wsv = np.repeat(np.asarray(ws_of), NNEW)
    sets = [TC.new_task_sets(int(w), int(j)) for w, j in zip(wsv, jj)]
    pc = None if eta is None else dict(eta=np.repeat(np.asarray(eta, float), NNEW))
    r = UT.search(nets, net_of, TC.stack_sets(sets, "shot"), TC.stack_sets(sets, "ho"), steps=TC.FT_STEPS, eval_at=FT_EVAL, pc=pc)
    return r["v"].reshape(len(idx), NNEW, -1), r["acc"]["ho"].reshape(len(idx), NNEW, -1)


@torch.no_grad()
def s_line_batched(netsB, iB, alphaA, ws, sets):
    """TC.s_line in one evaluation call: v_B = mean_B + C_B,centred alpha_A (anchored_param) and the 30 permuted correspondences
    TC.s_perms(ws), on B's new-task held-out sets. Returns dict(S_gain, accB, accB_perm) as TC.s_line."""
    perms = TC.s_perms(ws); v0, Bm = TC.anchored_param(netsB, np.full(NNEW, iB)); al = torch.as_tensor(np.asarray(alphaA), dtype=F32, device=DEV)
    mats = [Bm] + [Bm[:, :, torch.as_tensor(np.asarray(pm), device=DEV)] for pm in perms]
    cols = torch.stack([v0 + (Mx @ al.unsqueeze(-1)).squeeze(-1) for Mx in mats], 1).reshape(NNEW * len(mats), -1)
    hx, hy, hl = TC.stack_sets(sets, "ho"); k = len(mats)
    acc = TC.cols_accuracy(netsB, np.full(len(cols), iB), cols, (np.repeat(hx, k, 0), np.repeat(hy, k, 0), np.repeat(hl, k, 0))).reshape(NNEW, k)
    return dict(S_gain=float(np.mean(acc[:, 0] - acc[:, 1:].mean(1))), accB=acc[:, 0].tolist(), accB_perm=acc[:, 1:].mean(1).tolist())


@torch.no_grad()
def _chain_logp_h2(nets, net_of, cols, x, y):
    _, h = TC.hidden_states(nets, np.asarray(net_of), cols, (x, y))
    no = torch.as_tensor(np.asarray(net_of), device=DEV)
    lg = torch.baddbmm(nets.p["b3"][no].unsqueeze(1), h, nets.p["W3"][no].transpose(1, 2)) * nets.mult
    return torch.log_softmax(lg, -1).cpu().numpy(), h.double().cpu().numpy()


def stage_cross_pc(out):
    """AT5 (descriptive) for every ordered pair A -> B of a world with at least one PC net: Lambda, Phi, X and delta per model (PC-muP:
    round-1 checkpoints; BP: r2_t1's AT5 nets); per pair S-gain (A's alpha carried into B), free-column T1 gain and LOO-anchor transfer,
    T4 (B's column distilled to A's output distribution on the new task's 289 inputs; PC receivers by the soft-target PC local update),
    T3 h2 stitching where both widths are <= 256 and B is a BP net (T3 into a PC receiver is not defined). Searches in PC nets use the
    PC local update (W7); every evaluation by the feedforward pass. Field names follow r2_rgi_toy's rgi_toy/cross/w<ws>.json."""
    r1 = R1PC(r1_root(out)); t1root = os.environ.get("R2T1") or os.path.join(out, "r2_t1"); worlds = list(SEEDS); t_start = time.time()
    groups = []                                                          # (Nets, [model info], eta or None)
    for nets, metas in _bp_groups(t1root):
        infos = [_model_info(m, "bp") for m in metas]; keep = [i for i, f in enumerate(infos) if f["ws"] in worlds]
        if keep:
            groups.append((nets.subset(keep) if len(keep) < nets.B else nets, [infos[i] for i in keep], None))
    for W in PC_WIDTHS:
        found = [(ws, r1.find(job_name("pc_mup", W, ws))) for ws in worlds]; found = [(ws, p) for ws, p in found if p]
        if not found:
            continue
        nets = TC.load_pc_state_dicts([p for _, p in found])
        infos = []
        for i, (ws, p) in enumerate(found):
            jf = r1_job_fields(r1.job(job_name("pc_mup", W, ws))) or {}
            infos.append(dict(name=os.path.basename(p)[:-3], kind="pc", arm="pc_mup", W=W, init=0, ws=ws, level1=jf.get("level1"), first95=jf.get("first95"),
                              checkpoint=jf.get("checkpoint"), eval_point="checkpoint" if jf.get("checkpoint") else "maxsteps", checksum=TC.weight_checksum(nets, i)))
        eta = np.array([TC.pc_eta(nets, i, *TC.r1_training_batch(f["ws"]), seed=PC_ETA_SEED + f["ws"]) for i, f in enumerate(infos)])
        groups.append((nets, infos, eta))
    mods = [(g, i) for g, (_, inf, _) in enumerate(groups) for i in range(len(inf))]
    info = lambda m: groups[m[0]][1][m[1]]
    print(f"  cross_pc: {len(mods)} models ({sum(1 for m in mods if info(m)['kind'] == 'pc')} PC) in {len(groups)} groups", flush=True)
    # ---- per-model lines
    per = {}
    for g, (nets, infs, eta) in enumerate(groups):
        idx = np.arange(nets.B); wsl = [f["ws"] for f in infs]
        dl = TC.delta_stats(nets, wsl, [M.world(w)[0] for w in wsl]); lam = TC.lambda_line(nets, wsl)
        v, acc = _free_cols(nets, idx, wsl, eta)
        net_of = np.repeat(idx, NNEW); sets = [TC.new_task_sets(w, j) for w in wsl for j in range(NNEW)]
        al, aacc, _ = TC.learn_alpha(nets, net_of, sets, pc_eta=None if eta is None else np.repeat(eta, NNEW))
        for i, f in enumerate(infs):
            fl = TC.f_line(nets, i, f["ws"], v[i].astype(np.float64), acc[i][:, -1])
            d = dl[i]; ae = acc[i][:, -1]
            per[(g, i)] = dict(f, delta_h2=d["h2"]["delta"], delta_h1=d["h1"]["delta"], delta_ctrl_h2=d["h2"]["delta_ctrl"], delta_ctrl_h1=d["h1"]["delta_ctrl"],
                               ring_h2=d["h2"]["ring"], ring_h1=d["h1"]["ring"], delta=d, Lambda=lam[i]["Lambda"], Lambda_per_shift=lam[i]["per_shift"],
                               Lambda_null_wrong=lam[i].get("wrong"), Lambda_null_perm=lam[i].get("perm"), Phi=fl["Phi"], acc_proj=fl["acc_proj"],
                               acc_free=ae, A_free=float(acc[i].mean()), acc_alpha=aacc[i * NNEW:(i + 1) * NNEW], alpha=al[i * NNEW:(i + 1) * NNEW],
                               X=float(np.mean(aacc[i * NNEW:(i + 1) * NNEW]) / np.mean(ae)) if np.mean(ae) > 0 else None,
                               eta=None if eta is None else float(eta[i]), v_free=v[i])
    # ---- pairs: S and F (no search)
    pairs = [(a, b) for a in mods for b in mods if a != b and info(a)["ws"] == info(b)["ws"] and "pc" in (info(a)["kind"], info(b)["kind"])]
    P_ = {}
    for a, b in pairs:
        A_, B_ = per[a], per[b]; ws = A_["ws"]; sets = [TC.new_task_sets(ws, j) for j in range(NNEW)]
        d = dict(A=A_["name"], B=B_["name"], ws=ws, A_W=A_["W"], B_W=B_["W"], A_arm=A_["arm"], B_arm=B_["arm"], A_kind=A_["kind"], B_kind=B_["kind"],
                 A_init=A_["init"], B_init=B_["init"], A_level1=A_["level1"], B_level1=B_["level1"])
        d.update(s_line_batched(groups[b[0]][0], b[1], A_["alpha"], ws, sets))
        fl = TC.f_line(groups[a[0]][0], a[1], ws, np.asarray(A_["v_free"], np.float64), A_["acc_free"], groups[b[0]][0], b[1])
        d.update({k: v_ for k, v_ in fl.items() if k not in ("Phi", "acc_proj")})
        P_[(a, b)] = d
    # ---- pairs: T4 (every pair) and T3 (both W <= T3_MAXW, BP receiver), batched per receiver group
    xa, ya = TC.PAIRS[:, 0], TC.PAIRS[:, 1]; snd = {}
    for g, (nets, infs, _) in enumerate(groups):
        lp, h2 = _chain_logp_h2(nets, np.repeat(np.arange(nets.B), NNEW), np.concatenate([per[(g, i)]["v_free"] for i in range(nets.B)]).astype(np.float32), xa, ya)
        for i in range(nets.B):
            snd[(g, i)] = (lp[i * NNEW:(i + 1) * NNEW], h2[i * NNEW:(i + 1) * NNEW])
    allh = {}

    def all_h2(m):
        if m not in allh:
            allh[m] = TC.all_h2(groups[m[0]][0], m[1])
        return allh[m]
    for mode in ("t4", "t3"):
        jobs = {}
        for a, b in pairs:
            if mode == "t3":
                if not (info(a)["W"] <= T3_MAXW and info(b)["W"] <= T3_MAXW and info(b)["kind"] == "bp"):
                    continue
                S, best = TC.stitch_fit(all_h2(a), all_h2(b)); P_[(a, b)]["stitch"] = dict(lam=best[0], R2=best[1])
                tg = (np.concatenate([snd[a][1], np.ones((NNEW, P * P, 1))], 2) @ S).astype(np.float32)
            else:
                tg = snd[a][0].astype(np.float32)
            for j in range(NNEW):
                jobs.setdefault(b[0], []).append(((a, b), b[1], j, tg[j]))
        for g, lst in jobs.items():
            nets, infs, eta = groups[g]
            net_of = np.array([i for _, i, _, _ in lst]); sets = [TC.new_task_sets(infs[i]["ws"], j) for _, i, j, _ in lst]
            shots = (np.tile(xa, (len(lst), 1)), np.tile(ya, (len(lst), 1)), np.zeros((len(lst), P * P), np.int64))
            pc = None if (eta is None or mode == "t3") else dict(eta=np.asarray(eta, float)[net_of])
            r = UT.search(nets, net_of, shots, TC.stack_sets(sets, "ho"), steps=TC.FT_STEPS, eval_at=(TC.FT_STEPS,), loss="kl" if mode == "t4" else "mse",
                          target=np.stack([t for _, _, _, t in lst]), pc=pc)
            for k, (pr, i, j, _) in enumerate(lst):
                P_[pr].setdefault(mode, [None] * NNEW)[j] = float(r["acc"]["ho"][k, -1])
    for d in P_.values():
        for mode in ("t4", "t3"):
            d.setdefault(mode, None); d[f"{mode}_mean"] = float(np.mean(d[mode])) if d[mode] is not None else None
        d.setdefault("stitch", None)
    for ws in worlds:
        ms = [m for m in mods if info(m)["ws"] == ws]
        if not ms:
            continue
        res = dict(stage="cross_pc", measurement="AT5 cross-model pairs with a PC net (descriptive)", world=ws, quick=QUICK, plan_sha256=UT.PLAN_SHA256,
                   pc_widths=sorted({info(m)["W"] for m in ms if info(m)["kind"] == "pc"}), t3_max_width=T3_MAXW,
                   models=[{k: v_ for k, v_ in per[m].items() if k != "v_free"} for m in ms], pairs=[P_[p] for p in pairs if p[0] in ms],
                   finder=r1.log, meta=dict(UT.meta("cross_pc", t_start), r1pc=r1_root(out), r2t1=t1root),
                   notes="every PC width runs (critique v2-11); pairs at a width without G-PC(W) are 'not testable' (r2_ana); "
                         "t3 only where both widths <= the T3 limit and the receiver is a BP net")
        write_json(res, pdir(out, "cross_pc", f"w{ws}.json")); print(f"  cross_pc: world {ws} written ({len(res['pairs'])} pairs)", flush=True)
    return None


# ================================================================================================================ CLI
def run_stage(out, st):
    if st == "cross_pc":
        done = sorted(glob.glob(pdir(out, "cross_pc", "w*.json")))
        if done:
            print(f"[cross_pc] outputs exist, skipped: {len(done)} files", flush=True); return None
        t0 = time.time(); print(f"[cross_pc] start (device {DEV}, QUICK {QUICK})", flush=True); stage_cross_pc(out)
        print(f"[cross_pc] done ({time.time() - t0:.0f}s)", flush=True); return None
    path = pdir(out, f"{st}.json")
    if os.path.exists(path):
        print(f"[{st}] output exists, skipped: {path}", flush=True); return read_json(path)
    t0 = time.time(); print(f"[{st}] start (device {DEV}, QUICK {QUICK})", flush=True)
    res = (stage_gates if st == "gates" else stage_t7)(out); res["meta"] = UT.meta(st, t0); res["meta"]["r1pc"] = r1_root(out)
    write_json(res, path); print(f"[{st}] done ({time.time() - t0:.0f}s) -> {path}", flush=True)
    return read_json(path)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    if len(argv) < 3 or argv[2] not in STAGES + ("all", "selftest"):
        print(__doc__); return 2
    out, st = argv[1], argv[2]
    if st == "selftest":
        return selftest(out)
    for s in (STAGES if st == "all" else (st,)):
        run_stage(out, s)
    return 0


# ================================================================================================================ selftest (QUICK)
def make_fixture(root, widths=(64, 128), seeds=(0, 1)):
    """Fake round-1 PC folders made by pc_meta QUICK runs (QUICK=1 NT=1 T=20 BETA=0.1 python pc_meta.py <dir> <arm> <W> <s> [K]) and
    pc_summary, in run_shard's layout: PC-muP widths on gpu3 (W 128) / gpu4 (W 64), BP-muP on gpu5; a duplicate of one job on gpu6
    (finder test); pc_expected / shard_done like run_shard."""
    pm, ps = os.path.join(UT.TC._JU, "pc_meta.py"), os.path.join(UT.TC._JU, "pc_summary.py")
    env = dict(os.environ, QUICK="1", NT="1", OMP_NUM_THREADS="1", DEVICE="cpu", T="20", BETA="0.1", LR="1e-3", MAXSTEPS="6000")
    shard_of = {64: "gpu4", 128: "gpu3", 256: "gpu2", 512: "gpu1"}; plan = {}
    for W in widths:
        for s in seeds:
            plan.setdefault(shard_of[W], []).append(["pc_mup", str(W), str(s), "12"]); plan.setdefault(shard_of[W], []).append(["pc_mup", str(W), str(s), "1"])
            plan.setdefault("gpu5", []).append(["bp_mup", str(W), str(s), "12"]); plan.setdefault("gpu5", []).append(["bp_mup", str(W), str(s), "1"])
    for sh, jobs in plan.items():
        d = os.path.join(root, sh); o = os.path.join(d, "pc"); os.makedirs(o, exist_ok=True)
        names = []
        for a in jobs:
            name = f"{a[0]}_W{a[1]}_s{a[2]}_K{a[3]}_g0_wd1" + ("_T20_beta0.1" if a[0] == "pc_mup" else "") + "_quick"; names.append(name)
            if not os.path.exists(os.path.join(o, "jobs", name + ".json")):
                subprocess.run([sys.executable, pm, o] + a, env=env, check=True, capture_output=True)
        json.dump(names, open(os.path.join(d, f"pc_expected_{sh}.json"), "w"))
        subprocess.run([sys.executable, ps, o, os.path.join(d, f"pc_summary_{sh}.json")], env=env, check=True, capture_output=True)
        json.dump(dict(shard=sh, code_commit="fixture"), open(os.path.join(d, "shard_done.json"), "w"))
    dup = os.path.join(root, "gpu6", "pc"); os.makedirs(os.path.join(dup, "models"), exist_ok=True)        # duplicate: not expected there
    n0 = job_name("pc_mup", widths[0], seeds[0])
    import shutil
    shutil.copy(os.path.join(root, shard_of[widths[0]], "pc", "models", n0 + ".pt"), os.path.join(dup, "models", n0 + ".pt"))
    return root


def make_at5(root, worlds=(0, 1), widths=(32, 64)):
    """Fake r2_t1 AT5 BP nets (rgi_toy/models/at5_<name>.pt, TC.save_nets) at QUICK sizes: BP-muP inits 0 and 1, BP-SP init 0."""
    folder = os.path.join(root, "rgi_toy", "models")
    for W in widths:
        for specs in ([dict(W=W, ws=ws, init=i, arm="bp_mup") for ws in worlds for i in (0, 1)], [dict(W=W, ws=ws, arm="bp_sp") for ws in worlds]):
            names = ["at5_" + TC.net_name(s) for s in specs]
            if all(os.path.exists(os.path.join(folder, n + ".pt")) for n in names):
                continue
            res = TC.train(specs)
            for b in range(res.nets.B):
                res.nets.meta[b]["row"] = res.rows[b]
            TC.save_nets(res.nets, range(res.nets.B), folder, names=names)
    return root


def selftest(work):
    if not QUICK:
        raise SystemExit("selftest needs QUICK=1")
    out = os.path.join(work, "pc_selftest"); res = {}; t0 = time.time()

    def check(name, cond, info=""):
        res[name] = bool(cond); print(f"[{'PASS' if cond else 'FAIL'}] {name} {info} ({time.time() - t0:.0f}s)", flush=True)
    r1root = make_fixture(os.path.join(work, "r1pc")); os.environ["R1PC"] = r1root
    t1root = make_at5(os.path.join(work, "r2_t1")); os.environ["R2T1"] = t1root
    r1 = R1PC(r1root); n0 = job_name("pc_mup", 64, 0); p = r1.find(n0)
    check("OD-G9 finder: the duplicate resolves to the shard whose pc_expected lists the job", p is not None and os.sep + "gpu4" + os.sep in p and r1.log[-1]["chosen"] == "gpu4")
    R = r1.summary_rows(); PA = _pa()
    check("PCS rows: one per job name across shards", len(R) == len({r["name"] for r in R}) and len(R) >= 8, f"{len(R)} rows")
    g = gates_from_rows(R); G = PA.gates(R)
    check("gates = pc_analyze.gates on the rows; G-PC widths found", g["G"]["G1"] == G["G1"] and set(g["G_PC"]) == {64, 128}, {W: v["conds"] for W, v in g["G_PC"].items()})
    # PC column search / relaxation of the loaded round-1 PC net against pc_meta itself
    pcn = TC.load_pc_state_dicts([p]); X, y = TC.r1_training_batch(0); eta = TC.pc_eta(pcn, 0, X, y, seed=PC_ETA_SEED)
    m = pcn.to_mlp(0).to(DEV)
    with TC._BetaCtx(TC.BETA_PC, TC.T_PC):
        torch.manual_seed(PC_ETA_SEED); lam = M.lam_max(m, X, y)
        h1r, h2r, _, _ = M.pc_relax(m, X, y, 1.0 / lam, 20)
    tasks = UT.Tasks.w1([0]); vb, tp, x, yy, lab, _ = UT._chains(tasks, np.array([0]), 0, P * P)
    sel = np.concatenate([np.arange(144)])                         # the training batch of task 0 is split positions 0-143
    h1b, h2b, _ = TC.pc_relax(pcn, vb[:1], UT._chain_cols(pcn, tasks, vb[:1], tp[:1]), (x[:1, sel], yy[:1, sel], lab[:1, sel]), eta, T=20)
    ok = abs(eta - 1 / lam) < 1e-12 and float((h2b[0] - h2r[:144]).abs().max()) < 1e-4
    check("eta_x = 1 / pc_meta.lam_max; batched relaxation of a round-1 PC net = pc_meta.pc_relax (task 0's training pairs)", ok)
    nb = TC.load_pc_state_dicts([r1.find(job_name("bp_mup", 64, 0)), r1.find(job_name("bp_mup", 64, 1))]); sets = [TC.new_task_sets(0, j) for j in range(NNEW)]
    al = np.random.default_rng(3).standard_normal((NNEW, NPRE)).astype(np.float32) * 0.3
    a_, b_ = TC.s_line(nb, 1, al, nb, 0, sets, TC.s_perms(0)), s_line_batched(nb, 0, al, 0, sets)
    check("S line batched = TC.s_line (30 permuted correspondences)", np.allclose(a_["accB"], b_["accB"]) and abs(a_["S_gain"] - b_["S_gain"]) < 1e-9
          and np.allclose(a_["accB_perm"], b_["accB_perm"]), f"S_gain {b_['S_gain']:.4f}")
    for st in STAGES:
        try:
            r = run_stage(out, st)
            if st == "cross_pc":
                fs = sorted(glob.glob(pdir(out, "cross_pc", "w*.json"))); d0 = read_json(fs[0]) if fs else {}
                kinds = {(pp["A_kind"], pp["B_kind"]) for pp in d0.get("pairs", [])}
                check("stage cross_pc writes one file per world with BP->PC, PC->BP and PC->PC pairs, T4 everywhere, T3 only into BP",
                      len(fs) == len(SEEDS) and {("bp", "pc"), ("pc", "bp"), ("pc", "pc")} <= kinds and all(pp["t4"] is not None for pp in d0["pairs"])
                      and all((pp["t3"] is None) == (pp["B_kind"] == "pc" or max(pp["A_W"], pp["B_W"]) > T3_MAXW) for pp in d0["pairs"]),
                      f"{len(fs)} files, {len(d0.get('pairs', []))} pairs in w0")
            elif st == "t7":
                fam = r["rule_preview"]
                check("stage t7 runs both sides at every width and builds the Holm family", all("pc" in w and "bp" in w for w in r["widths"].values()) and fam["family_size"] > 0,
                      f"family {fam['family_size']}, claims {fam['claims']}")
            else:
                check(f"stage {st} runs", r is not None and "G_PC" in r)
        except Exception as ex:                                                # noqa: BLE001
            import traceback; traceback.print_exc(); check(f"stage {st} runs", False, repr(ex))
    t1 = os.path.join(work, "fake_t1.json")
    write_json(dict(cells={"bp_mup_W128": dict(base=dict(level1=[True] * 5, B2={"8": [0.3, 0.25, 0.35, 0.28, 0.31]}))}), t1)
    fb = g_meta_fallback(read_json(t1)); check("G35 fallback formulas (G1, G2 on round-2 BP-muP W 128 rows)", fb["pass_"] and fb["label"] == "gate from round 2")
    bad = [k for k, v in res.items() if not v]
    print(f"r2_pc_toy selftest: {len(res) - len(bad)}/{len(res)} passed ({time.time() - t0:.0f}s)" + (f"; FAILED: {bad}" if bad else ""), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
