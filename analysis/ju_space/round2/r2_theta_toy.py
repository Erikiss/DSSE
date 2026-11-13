"""Round 2, Part B toy: a regularisation x capacity phase diagram of the first-order meta level and a self-training loop with a real
temperature tau (frozen plan results/ju_space/plans/abschnitt11_12_runde2_plan.md, section 4.2 B-T1, B-T2, B-T4, B-T5 and the toy
rules P1-P4 / T5 phases; shared toy objects W0, W1, W6 of 2.5; gate G27; session r2_t2 of section 7; CLI of 9.3). Implementer 6. Every
net is trained in this session by r2_toy_common's batched trainer (W0; 2.7 C25); nothing of round 1 is read (the 24 round-1 cells
enter only r2_ana's reproduction check G46, which reads theta_toy/nets.json).

Usage:  python r2_theta_toy.py <out_dir> <stage>
          stage t1        B-T1: wd {0, 0.1, 0.3, 1, 3, 10} x W {64, 256} x seeds 0-3 x K {12, 1} -> theta_toy/rows_t1.json, nets.json
          stage t2        B-T2: W 512 and 1024, seeds 0-1, K 12 and 1, at the wd values bracketing the W 256 switch(es) (P2 on B-T1,
                          in-session; else wd 0.3 and 3, OD-B6) -> theta_toy/t2.json (the choice), rows_t2.json, nets.json
          stage t5        B-T5: W6 self-training loop on B-T1's K 12 nets at wd {0.3, 1, 3} x W {64, 256} x seeds 0-3, pilot gate G27
                          first -> theta_toy/t5.json
          stage rules     (alias t4) B-T4 and the toy rules P1-P4, T5 phases, tau_c -> theta_toy/verdict.json
          stage selftest  code tests of this script (CPU; run with QUICK=1)
        env DEVICE (cuda | cpu), NT, QUICK (1 = code test: tiny widths and grids, pc_meta's QUICK sizes)
nets.json is a list of pc_summary-style rows (pc_summary.row's keys: name (= pc_meta's job name of the same cell), arm, W, seed, K,
gamma, wd_scale, lr, T, beta, first95, checkpoint, level1, final_train, final_heldout, coord_check, seconds, delta, B2_8, B2_4, A8_mean,
F8_mean, F4_mean, B2_all, retention_mean), with pc_meta's conventions for K = 1 rows (level1 False, no delta, checkpoint = the number
of updates of its K = 12 net), plus round-2 fields: stage, eval_point ('checkpoint' | 'maxsteps' | 'fixed'), updates, first95_cens
(censored at 6,000), checksum, r2_name, B2_meta (K 12 rows: B2_8(K 12) - B2_8(K 1) of the same seed, cell and width).
A K = 12 net without level 1 is evaluated at MAXSTEPS (W0; its row has checkpoint None as in pc_meta, eval_point 'maxsteps', and
delta / B2 computed there); it has no K = 1 control (4.2). Resumable: finished stages are skipped; nets are saved in theta_toy/models/
(bundles with weight checksums) and reloaded. Ambiguities and their resolution: NOTES_w6.md."""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import r2_toy_common as TC          # noqa: E402  sets T / BETA / LR / MAXSTEPS before importing pc_meta (W0)
import pc_analyze as PA             # noqa: E402  round-1 code, imported unchanged (cluster_boot_ols)

M = TC.M
P, NPRE, NNEW = TC.P, TC.NPRE, TC.NNEW
QUICK, DEV = TC.QUICK, TC.DEV

# ====================================================================================================================== grids
STAGES = ("t1", "t2", "t5", "rules", "t4")
WDS = (0.0, 0.3, 1.0, 3.0) if QUICK else (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)       # B-T1 wd_scale grid (wd 0 the lowest point)
T1_WIDTHS = (16, 32) if QUICK else (64, 256)
SEEDS = (0, 1, 2, 3)
T2_WIDTHS = (40, 48) if QUICK else (512, 1024)
T2_SEEDS = (0, 1)
T2_FALLBACK = (0.3, 3.0)                                                         # OD-B6
T5_WDS = (0.3, 1.0, 3.0)
TAUS = (0.1, 0.3, 0.5, 1.0, 2.0)
ROUNDS = 6
PILOT = dict(W=T1_WIDTHS[0], wd=1.0, seed=0)                                     # G27 pilot: W 64, wd 1, seed 0
META_THR = 0.10                                                                  # P1: B2_meta > 0.10
BOOT_SEED, BOOT_DRAWS = 0, 10000                                                 # tau_c regression: cluster bootstrap over seeds
RNG_SEEDS = dict(                                                                # OD-G6: every random stream of these stages
    net_init="torch.manual_seed(seed) (pc_meta's job seed = the world seed; the K 12 and K 1 nets alike; TC.init_seed)",
    fresh_F="torch.manual_seed(50000 + 101 seed + j) (TC.fresh_seed; B2's fresh nets)",
    coord_check_probe="torch.Generator().manual_seed(seed) (TC.train)",
    delta_floor="torch.Generator().manual_seed(77 + seed): h1 then h2, 50 raw then 50 controlled label permutations (TC.delta_stats)",
    w6_sampling="Gumbel-max uniforms from torch.Generator().manual_seed(7000 + 1000 ws + 100 j + 10 n), shared across tau and the two "
                "starts (TC.self_train)",
    tau_c_cluster_boot=f"pc_analyze.cluster_boot_ols(B={BOOT_DRAWS}, seed={BOOT_SEED}): np.random.default_rng({BOOT_SEED}) over the seeds",
    torch_global="torch.manual_seed(0) at the start of every stage")


# ====================================================================================================================== I/O
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, torch.Tensor):
        return jsonable(o.detach().cpu().numpy())
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(jsonable(obj), f, indent=1)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def save_bundle(path, nets, infos):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = dict(arch=dict(kind=nets.kind, W=nets.W, depth=nets.depth, act=nets.act, param=nets.param),
               p={k: v.detach().cpu().clone() for k, v in nets.p.items()}, tables=[nets.fx.cpu(), nets.fy.cpu(), nets.ft.cpu()],
               meta=jsonable(nets.meta), infos=jsonable(infos), checksums=[TC.weight_checksum(nets, i) for i in range(nets.B)])
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_bundle(path):
    o = torch.load(path, map_location="cpu", weights_only=False)
    a = o["arch"]
    nets = TC.Nets(a["kind"], a["W"], a["depth"], a["act"], a["param"], {k: v.to(DEV) for k, v in o["p"].items()},
                   *(t.to(DEV) for t in o["tables"]), meta=o["meta"])
    for i, c in enumerate(o["checksums"]):
        if TC.weight_checksum(nets, i) != c:
            raise ValueError(f"weight checksum mismatch in {path} (net {i})")
    return nets, o["infos"]


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tdir(out, *p):
    return os.path.join(out, "theta_toy", *p)


# ====================================================================================================================== nets and rows
def job_name(W, seed, K, wd):
    """pc_meta's job name of the cell (bp_mup, gamma 0, lr 1e-3; '_quick' under QUICK, as pc_meta)."""
    return f"bp_mup_W{W}_s{seed}_K{K}_g0_wd{wd:g}" + ("_quick" if QUICK else "")


def finetune_all(nets, ws_of, K, nshot=TC.NSHOT):
    """pc_meta.finetune(mode 'all') for every net and new task, batched: a copy of the pretrained net, all weights trained with
    pc_meta.make_opt's AdamW (lr 1e-3, decay 1e-3 per update, i.e. decay_scale 1 whatever the net's wd; muP hidden lr x 128 / W),
    FT_STEPS full-batch updates on the task's nshot shots; held-out accuracy (split positions 8-288) at FT_EVAL; retention = the held-out
    accuracy on the net's K pretraining tasks afterwards (pc_meta: accuracy(mm, Xva, yva)). Returns (acc [B, 4, len(FT_EVAL)], retention
    [B, 4])."""
    B = nets.B
    rep = nets.subset(np.repeat(np.arange(B), NNEW))
    wsr = np.repeat(np.asarray(ws_of), NNEW)
    sets = [TC.new_task_sets(int(wsr[c]), c % NNEW, nshot=nshot) for c in range(rep.B)]
    sx, sy, sl = (torch.as_tensor(a, device=DEV) for a in TC.stack_sets(sets, "shot"))
    hx, hy, hl = (torch.as_tensor(a, device=DEV) for a in TC.stack_sets(sets, "ho"))
    slot = torch.as_tensor([NPRE + c % NNEW for c in range(rep.B)], device=DEV)[:, None]
    Xs, Xh = rep.features(sx, sy, slot.expand(-1, sx.shape[1])), rep.features(hx, hy, slot.expand(-1, hx.shape[1]))
    opt = TC.AdamWB(rep.p, TC.LR, 1.0, hidden=rep.hidden(), W=rep.W, mup=rep.param == "mup")
    keys, accs = list(rep.p.keys()), []
    for s in range(1, TC.FT_STEPS + 1):
        for k in keys:
            rep.p[k].requires_grad_(True)
        L = Fn.cross_entropy(rep.logits(Xs).reshape(-1, P), sl.reshape(-1), reduction="none").view(rep.B, -1).mean(1).sum()
        g = torch.autograd.grad(L, [rep.p[k] for k in keys])
        for k in keys:
            rep.p[k].requires_grad_(False)
        opt.step(dict(zip(keys, g)))
        if s in TC.FT_EVAL:
            with torch.no_grad():
                accs.append((rep.logits(Xh).argmax(-1) == hl).float().mean(1))
    acc = torch.stack(accs, 1).cpu().numpy().reshape(B, NNEW, -1)
    ret = TC.task_accuracy(rep, wsr, tasks=range(K)).mean(1).reshape(B, NNEW)
    return acc, ret, rep


def train_cell_nets(out, tag, W, seeds, wds):
    """K = 12 nets (checkpoint rule) of every (wd, seed), then the K = 1 control of every K = 12 net with a checkpoint, trained for the
    number of updates of that checkpoint (pc_meta's K = 1 rule; no K = 1 net without a checkpoint). Bundles models/<tag>_W<W>_K12.pt and
    _K1.pt. Returns ((nets12, infos12), (nets1, infos1) or None)."""
    p12, p1 = tdir(out, "models", f"{tag}_W{W}_K12.pt"), tdir(out, "models", f"{tag}_W{W}_K1.pt")
    if os.path.exists(p12):
        n12, i12 = load_bundle(p12)
    else:
        specs = [dict(W=W, ws=s, K=NPRE, wd=wd) for wd in wds for s in seeds]
        say(f"{tag}: training {len(specs)} K 12 nets W {W}")
        t0 = time.time()
        res = TC.train(specs)
        n12, i12 = res.nets, [dict(spec=dict(W=W, seed=s["ws"], K=NPRE, wd=s["wd"]), row=r, seconds=(time.time() - t0) / len(specs))
                              for s, r in zip(specs, res.rows)]
        save_bundle(p12, n12, i12)
    k1 = [n for n, inf in enumerate(i12) if inf["row"]["eval_point"] == "checkpoint"]
    if not k1:
        return (n12, i12), None
    if os.path.exists(p1):
        n1, in1 = load_bundle(p1)
    else:
        specs = [dict(W=W, ws=i12[n]["spec"]["seed"], K=1, wd=i12[n]["spec"]["wd"]) for n in k1]
        steps = [int(i12[n]["row"]["checkpoint"]) for n in k1]
        say(f"{tag}: training {len(specs)} K 1 nets W {W}")
        t0 = time.time()
        res = TC.train(specs, fixed_steps=steps)
        n1, in1 = res.nets, [dict(spec=dict(W=W, seed=s["ws"], K=1, wd=s["wd"]), row=r, seconds=(time.time() - t0) / len(specs))
                             for s, r in zip(specs, res.rows)]
        save_bundle(p1, n1, in1)
    return (n12, i12), (n1, in1)


def cell_rows(stage, n12, i12, k1=None):
    """pc_summary-style rows of the K 12 nets (delta at their evaluation point) and the K 1 nets: B2 (8, 4 shots: TC.b2 = pc_meta's column
    search minus fresh nets), B2_all and retention (finetune_all), delta / ring (K 12; TC.delta_stats), B2_meta per seed."""
    rows = []
    for nets, infos, K in ((n12, i12, NPRE),) + (((k1[0], k1[1], 1),) if k1 is not None else ()):
        ws_of = [inf["spec"]["seed"] for inf in infos]
        b2 = TC.b2(nets, ws_of)
        acc_all, ret, _ = finetune_all(nets, ws_of, K)
        dl = TC.delta_stats(nets, ws_of, [M.world(s)[0] for s in ws_of]) if K == NPRE else [None] * nets.B
        for n, inf in enumerate(infos):
            sp, r = inf["spec"], inf["row"]
            lev = bool(r["first95"] is not None) if K == NPRE else False             # pc_meta: K = 1 rows never set first95 / level1
            F8 = b2[8]["F"][n]
            row = dict(name=job_name(sp["W"], sp["seed"], K, sp["wd"]), arm="bp_mup", W=sp["W"], seed=sp["seed"], K=K, gamma=0.0,
                       wd_scale=sp["wd"], lr=TC.LR, T=None, beta=None, first95=r["first95"] if K == NPRE else None,
                       checkpoint=(r["checkpoint"] if r["eval_point"] == "checkpoint" else None) if K == NPRE else r["checkpoint"],
                       level1=lev, final_train=r["final_train"], final_heldout=r["final_heldout"], coord_check=r["coord_check"],
                       seconds=inf.get("seconds"), delta=dl[n],
                       B2_8=float(b2[8]["B2"][n]), B2_4=float(b2[4]["B2"][n]), A8_mean=float(b2[8]["A"][n].mean()),
                       F8_mean=float(F8.mean()), F4_mean=float(b2[4]["F"][n].mean()),
                       B2_all=float(np.mean([acc_all[n, j].mean() - F8[j].mean() for j in range(NNEW)])), retention_mean=float(ret[n].mean()),
                       stage=stage, eval_point=r["eval_point"], updates=r["updates"],
                       first95_cens=(r["first95"] if r["first95"] is not None else TC.MAXSTEPS) if K == NPRE else None,
                       censored=(r["first95"] is None) if K == NPRE else None, checksum=r["checksum"], r2_name=r["name"],
                       A8=b2[8]["A"][n], F8=F8, A_all8=acc_all[n], retention=ret[n])
            rows.append(row)
    for r in rows:
        if r["K"] == NPRE:
            k1r = [q for q in rows if q["K"] == 1 and q["W"] == r["W"] and q["seed"] == r["seed"] and q["wd_scale"] == r["wd_scale"]]
            r["B2_meta"] = (r["B2_8"] - k1r[0]["B2_8"]) if (k1r and r["level1"]) else None
    return rows


def write_nets_json(out):
    rows = []
    for st in ("t1", "t2"):
        p = tdir(out, f"rows_{st}.json")
        if os.path.exists(p):
            rows += read_json(p)
    write_json(tdir(out, "nets.json"), rows)
    return rows


# ====================================================================================================================== B-T1 (t1)
def stage_t1(out):
    p = tdir(out, "rows_t1.json")
    if os.path.exists(p):
        say("t1: output exists, skipped:", p)
        write_nets_json(out)
        return
    t0 = time.time()
    rows = []
    for W in T1_WIDTHS:
        k12, k1 = train_cell_nets(out, "t1", W, SEEDS, WDS)
        rows += cell_rows("t1", k12[0], k12[1], k1)
        say(f"t1: W {W} rows ({time.time() - t0:.0f}s)")
    write_json(p, rows)
    write_nets_json(out)
    say(f"t1: {len(rows)} nets; level 1 (K 12) {sum(r['level1'] for r in rows if r['K'] == NPRE)}/{sum(r['K'] == NPRE for r in rows)} "
        f"({time.time() - t0:.0f}s)")


# ====================================================================================================================== rules (pure functions)
def meta_counts(rows, W, wd, seeds=None):
    """Per cell (W, wd): seeds (K 12 rows), n_level1, n_meta (B2_meta > 0.10), B2_meta list, delta_h2 list."""
    rr = [r for r in rows if r["K"] == NPRE and r["W"] == W and abs(r["wd_scale"] - wd) < 1e-12 and (seeds is None or r["seed"] in seeds)]
    bm = [r.get("B2_meta") for r in rr]
    return dict(n=len(rr), n_level1=sum(bool(r["level1"]) for r in rr), n_meta=sum(b is not None and b > META_THR for b in bm),
                B2_meta=bm, seeds=[r["seed"] for r in rr], delta_h2=[(r.get("delta") or {}).get("h2", {}).get("delta") for r in rr])


def p1_phase(c):
    """P1 (frozen; shares generalise '3 of 4' and '1 of 4' to other seed counts): 'meta' = >= 3/4 of the seeds have B2_meta > 0.10; 'no
    meta' = level 1 in >= 3/4 and <= 1/4 in meta; 'not generalised within budget' = < 3/4 reach held-out 0.95 within 6,000 updates;
    otherwise 'boundary'."""
    n = c["n"]
    if n == 0:
        return None
    if c["n_meta"] >= 0.75 * n:
        return "meta"
    if c["n_level1"] < 0.75 * n:
        return "not generalised within budget"
    if c["n_meta"] <= 0.25 * n:
        return "no meta"
    return "boundary"


def p2_boundaries(wds, f):
    """P2 bracket rule (frozen; ref theta_design/sim_toy_bracket.py): f = meta share per grid wd (increasing, wd 0 the lowest point).
    switch-on = the interval between the last cell (up to the peak) with share <= 1/4 and the next cell with >= 3/4; switch-off = between
    the last cell (from the peak on) with >= 3/4 and the next with <= 1/4; 'below grid' if the lowest cell is already >= 3/4, 'above
    grid' if the highest still is; point estimate = linear interpolation of the share at 0.5 between the bracket ends."""
    wds, f = [float(w) for w in wds], [float(x) for x in f]
    n = len(f)
    out = dict(wds=wds, share=f, on=None, off=None, on_flag=None, off_flag=None)
    if n == 0 or max(f) < 0.75:
        out.update(on_flag="no meta cell (no wd with >= 3/4 of the seeds in meta)", off_flag="no meta cell")
        return out
    peak = int(np.argmax(f))

    def interp(i, k, up):
        for c in range(i, k):
            a, b = f[c], f[c + 1]
            if (up and a < 0.5 <= b) or (not up and a > 0.5 >= b):
                return wds[c] + (0.5 - a) / (b - a) * (wds[c + 1] - wds[c])
        return None
    if f[0] >= 0.75:
        out["on_flag"] = "below grid"
    else:
        lo = [i for i in range(peak + 1) if f[i] <= 0.25]
        hi = [i for i in range(peak + 1) if f[i] >= 0.75]
        if lo and hi and max(lo) < max(hi):
            i = max(lo)
            k = min(j for j in hi if j > i)
            out["on"] = dict(lo=wds[i], hi=wds[k], point=interp(i, k, True))
        else:
            out["on_flag"] = "not bracketed (no cell with <= 1/4 below the meta cells)"
    if f[-1] >= 0.75:
        out["off_flag"] = "above grid"
    else:
        lo = [i for i in range(peak, n) if f[i] <= 0.25]
        hi = [i for i in range(peak, n) if f[i] >= 0.75]
        if lo and hi and min(hi) < min(lo):
            k = min(lo)
            i = max(j for j in hi if j < k)
            out["off"] = dict(lo=wds[i], hi=wds[k], point=interp(i, k, False))
        else:
            out["off_flag"] = "not bracketed (no cell with <= 1/4 above the meta cells)"
    return out


def boundaries_of(rows, W, wds=None, seeds=None):
    wds, seeds = (WDS if wds is None else wds), (SEEDS if seeds is None else seeds)
    cells = [meta_counts(rows, W, wd, seeds) for wd in wds]
    f = [c["n_meta"] / c["n"] if c["n"] else 0.0 for c in cells]
    return p2_boundaries(wds, f), cells


def t2_choice(rows, wds=None):
    """B-T2's wd values (in-session decision, OD-G8): the two grid wd values bracketing the W 256 switch-on and, if found, the two
    bracketing the switch-off (P2 on B-T1); if W 256 has no switch, wd 0.3 and 3 (descriptive, OD-B6)."""
    b, _ = boundaries_of(rows, T1_WIDTHS[-1], wds)
    wds = sorted({x for k in ("on", "off") if b[k] is not None for x in (b[k]["lo"], b[k]["hi"])})
    if wds:
        return dict(wds=wds, source="P2 brackets at W %d" % T1_WIDTHS[-1], descriptive=False, boundaries=b)
    return dict(wds=list(T2_FALLBACK), source=f"W {T1_WIDTHS[-1]} has no switch: fallback wd 0.3 and 3 (OD-B6)", descriptive=True, boundaries=b)


def intersect(a, b):
    return max(a[0], b[0]) <= min(a[1], b[1])


def p3_transfer(rows, b_lo, b_hi, choice, W_lo=None, W_hi=None):
    """P3 (frozen): 'boundary transfers across width (muP)' if the W 64 and W 256 switch-on intervals intersect and at W 512 the lower
    bracketing wd has 0 of 2 seeds in meta and the upper 2 of 2 (W 1024 the same, descriptive, OD-B10); 'shifts' otherwise, with the
    direction. 'Not decidable' only where B-T2 cannot test the W 256 bracket: W 256 has no switch-on bracket, B-T2 ran at the fallback wd
    values (OD-B6), or B-T2 has no W 512 net at a bracket wd value (review theta_toy-3). A W 64 switch-on outside the grid cannot
    intersect, so the verdict is 'shifts' (review theta_toy-1); its direction comes from W 64's flag ('below grid': W 256 at higher wd;
    any other flag: not determined)."""
    W_lo, W_hi = W_lo or T1_WIDTHS[0], W_hi or T1_WIDTHS[-1]
    res = dict(widths=[W_lo, W_hi] + list(T2_WIDTHS), switch_on_flags={str(W_lo): b_lo.get("on_flag"), str(W_hi): b_hi.get("on_flag")})
    if b_hi["on"] is None:
        res["verdict"] = f"not decidable (no switch-on bracket at W {W_hi}: {b_hi.get('on_flag')})"
        return res
    if choice.get("descriptive"):
        res["verdict"] = "not decidable (B-T2 ran at the fallback wd values; descriptive)"
        return res
    i_hi = (b_hi["on"]["lo"], b_hi["on"]["hi"])
    i_lo = None if b_lo["on"] is None else (b_lo["on"]["lo"], b_lo["on"]["hi"])
    inter = i_lo is not None and intersect(i_lo, i_hi)
    wl, wu = i_hi
    per = {}
    for W in T2_WIDTHS:
        cl, cu = meta_counts(rows, W, wl, T2_SEEDS), meta_counts(rows, W, wu, T2_SEEDS)
        per[W] = dict(lower_wd=wl, lower_meta=cl["n_meta"], lower_n=cl["n"], upper_wd=wu, upper_meta=cu["n_meta"], upper_n=cu["n"],
                      inside=cl["n"] > 0 and cu["n"] > 0 and cl["n_meta"] == 0 and cu["n_meta"] == cu["n"])
    first = per[T2_WIDTHS[0]]
    res.update(switch_on={str(W_lo): i_lo, str(W_hi): i_hi}, intervals_intersect=inter, larger_widths=per)
    if first["lower_n"] == 0 or first["upper_n"] == 0:
        res["verdict"] = (f"not decidable (B-T2 has no W {T2_WIDTHS[0]} nets at the W {W_hi} switch-on bracket wd {wl:g} / {wu:g}; it ran at "
                          f"wd {choice.get('wds')})")
        res["bracket_in_session"] = (choice.get("boundaries") or {}).get("on")
        return res
    if inter and first["inside"]:
        res["verdict"] = "boundary transfers across width (muP)"
    else:
        d = []
        if i_lo is None:
            if b_lo.get("on_flag") == "below grid":
                d.append(f"W {W_hi} switch-on at higher wd than W {W_lo} (W {W_lo} switch-on below grid)")
            else:
                d.append(f"W {W_lo} has no switch-on bracket ({b_lo.get('on_flag')}); direction not determined")
        elif not inter:
            d.append(f"W {W_hi} switch-on at {'higher' if i_hi[0] > i_lo[1] else 'lower'} wd than W {W_lo}")
        if not first["inside"]:
            if first["lower_meta"] > 0:
                d.append(f"W {T2_WIDTHS[0]}: meta already at the lower bracket wd {wl:g} (switch-on moves to lower wd)")
            if first["upper_meta"] < first["upper_n"]:
                d.append(f"W {T2_WIDTHS[0]}: meta not complete at the upper bracket wd {wu:g} (switch-on moves to higher wd)")
        res["verdict"] = "shifts"
        res["direction"] = d
    return res


def od_g8_t2(choice, rows):
    """OD-G8 for B-T2's wd values: the in-session choice (t2.json; it governs what ran) against the same decision recomputed from the
    rows now in nets.json; a disagreement is reported, nothing else changes (review theta_toy-3)."""
    rec = t2_choice(rows)
    if not choice.get("wds") and str(choice.get("source", "")).startswith("t2 not run"):
        return dict(in_session=None, recomputed=rec["wds"], agree=None, note="t2 not run")
    a, b = [float(x) for x in choice.get("wds", [])], [float(x) for x in rec["wds"]]
    return dict(in_session=a, recomputed=b, agree=a == b, in_session_source=choice.get("source"), recomputed_source=rec["source"])


def p4_delta(rows, b_lo, b_hi, W_lo=None, W_hi=None):
    """P4 (frozen): 'delta is the geometric control variable' if the delta_h2 ranges at the switch-on brackets (min to max of the seed
    means at the two bracketing wd values) intersect across W 64 and W 256 while the wd brackets do not; otherwise 'not shown'."""
    W_lo, W_hi = W_lo or T1_WIDTHS[0], W_hi or T1_WIDTHS[-1]
    if b_lo["on"] is None or b_hi["on"] is None:
        return dict(verdict=f"not shown (no switch-on bracket at W {W_lo if b_lo['on'] is None else W_hi})")
    rng = {}
    for W, b in ((W_lo, b_lo), (W_hi, b_hi)):
        m = []
        for wd in (b["on"]["lo"], b["on"]["hi"]):
            d = [x for x in meta_counts(rows, W, wd)["delta_h2"] if x is not None]
            m.append(float(np.mean(d)) if d else None)
        rng[W] = None if None in m else (min(m), max(m))
    if rng[W_lo] is None or rng[W_hi] is None:
        return dict(verdict="not shown (delta_h2 missing at a bracket)", delta_ranges=rng)
    di = intersect(rng[W_lo], rng[W_hi])
    wi = intersect((b_lo["on"]["lo"], b_lo["on"]["hi"]), (b_hi["on"]["lo"], b_hi["on"]["hi"]))
    return dict(verdict="delta is the geometric control variable" if (di and not wi) else "not shown", delta_ranges=rng,
                delta_ranges_intersect=di, wd_brackets_intersect=wi)


def t5_cell_phase(per_seed, frac=0.75):
    """T5 phases (frozen) per (wd, W, tau), each needing >= 3 of 4 seeds: 'self-improving' S > 0.05; 'bistable' M_6 > 0.1 and kappa >= 0.8;
    'dissipative' M_6 < 0.1; otherwise 'mixed'. per_seed: list of dict(S, M6, kappa). The first three can hold together ('self-improving'
    and 'bistable' / 'dissipative'); 'mixed' = none holds."""
    n = len(per_seed)
    si = sum(d["S"] > 0.05 for d in per_seed)
    bi = sum(d["M6"] > 0.1 and d["kappa"] >= 0.8 for d in per_seed)
    di = sum(d["M6"] < 0.1 for d in per_seed)
    labels = [lab for lab, k in (("self-improving", si), ("bistable", bi), ("dissipative", di)) if n and k >= frac * n]
    return dict(n=n, n_self_improving=si, n_bistable=bi, n_dissipative=di, labels=labels, phase=" + ".join(labels) if labels else "mixed")


def tau_switch(taus, bist, diss):
    """The pair of adjacent tau values where 'bistable' switches to 'dissipative' as tau grows (first such pair), else None with a note."""
    for i in range(len(taus) - 1):
        if bist[i] and diss[i + 1]:
            return (taus[i], taus[i + 1]), None
    if all(bist):
        return None, "bistable at every tau"
    if all(diss):
        return None, "dissipative at every tau"
    return None, "no bistable -> dissipative switch between adjacent tau values"


# ====================================================================================================================== B-T2 (t2)
def stage_t2(out):
    p = tdir(out, "rows_t2.json")
    if os.path.exists(p):
        say("t2: output exists, skipped:", p)
        write_nets_json(out)
        return
    p1 = tdir(out, "rows_t1.json")
    if not os.path.exists(p1):
        raise SystemExit("t2 needs theta_toy/rows_t1.json (stage t1) first")
    t0 = time.time()
    r1 = read_json(p1)
    cp = tdir(out, "t2.json")
    choice = read_json(cp) if os.path.exists(cp) else t2_choice(r1)
    write_json(cp, dict(choice, widths=list(T2_WIDTHS), seeds=list(T2_SEEDS), rng_seeds=RNG_SEEDS))
    say(f"t2: wd values {choice['wds']} ({choice['source']})")
    rows = []
    for W in T2_WIDTHS:
        k12, k1 = train_cell_nets(out, "t2", W, T2_SEEDS, choice["wds"])
        rows += cell_rows("t2", k12[0], k12[1], k1)
        say(f"t2: W {W} rows ({time.time() - t0:.0f}s)")
    write_json(p, rows)
    write_nets_json(out)


# ====================================================================================================================== B-T5 (t5)
def run_self_train(nets, infos, ws_of):
    res = TC.self_train(nets, ws_of, taus=TAUS, rounds=ROUNDS)
    summ = TC.w6_summary(res)
    out = []
    for (b, tau), d in sorted(summ.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        sp, r = infos[b]["spec"], infos[b]["row"]
        out.append(dict(W=sp["W"], wd=sp["wd"], seed=sp["seed"], tau=float(tau), S=d["S"], M=d["M"], M1=d["M"][1], M6=d["M"][ROUNDS],
                        kappa=d["kappa"], A_true=d["A_true"], A_wrong=d["A_wrong"], eval_point=r["eval_point"], level1=r["first95"] is not None,
                        net=job_name(sp["W"], sp["seed"], NPRE, sp["wd"])))
    return out


def stage_t5(out):
    p = tdir(out, "t5.json")
    if os.path.exists(p):
        say("t5: output exists, skipped:", p)
        return
    t0 = time.time()
    rows1 = read_json(tdir(out, "rows_t1.json")) if os.path.exists(tdir(out, "rows_t1.json")) else []
    dmap = {(r["W"], r["wd_scale"], r["seed"]): (r.get("delta") or {}).get("h2", {}).get("delta") for r in rows1 if r["K"] == NPRE}
    pilot, main = [], []
    for W in T1_WIDTHS:
        bp = tdir(out, "models", f"t1_W{W}_K12.pt")
        if not os.path.exists(bp):
            raise SystemExit(f"t5 needs B-T1's K 12 nets ({bp}; stage t1) first")
        nets, infos = load_bundle(bp)
        sel = [n for n, inf in enumerate(infos) if any(abs(inf["spec"]["wd"] - w) < 1e-12 for w in T5_WDS) and inf["spec"]["seed"] in SEEDS]
        pil = [n for n in sel if W == PILOT["W"] and abs(infos[n]["spec"]["wd"] - PILOT["wd"]) < 1e-12 and infos[n]["spec"]["seed"] == PILOT["seed"]]
        rest = [n for n in sel if n not in pil]
        if pil:
            say(f"t5: pilot net W {W} wd {PILOT['wd']:g} seed {PILOT['seed']} ({len(TAUS) * 2 * NNEW} chains)")
            pilot = run_self_train(nets.subset(pil), [infos[n] for n in pil], [infos[n]["spec"]["seed"] for n in pil])
        if rest:
            say(f"t5: W {W}: {len(rest)} nets ({len(rest) * len(TAUS) * 2 * NNEW} chains)")
            main += run_self_train(nets.subset(rest), [infos[n] for n in rest], [infos[n]["spec"]["seed"] for n in rest])
    gate_vals = [dict(tau=d["tau"], M6=d["M6"], S=d["S"]) for d in pilot]
    gate = any(d["M6"] > 0.2 or d["S"] > 0.05 for d in pilot)
    allr = pilot + main
    for d in allr:
        d["delta_h2"] = dmap.get((d["W"], d["wd"], d["seed"]))
    res = dict(stage="t5", measurement="B-T5 self-training loop with a real temperature tau (W6)", quick=QUICK, taus=list(TAUS), rounds=ROUNDS,
               wds=list(T5_WDS), widths=list(T1_WIDTHS), seeds=list(SEEDS), pilot=dict(net=PILOT, values=gate_vals, passed=gate),
               G27=dict(passed=gate, label=None if gate else "no self-referential dynamics in the toy (phase map descriptive)"),
               rows=allr, seconds=time.time() - t0, rng_seeds=RNG_SEEDS,
               notes="per (net, tau), averaged over the 4 new tasks: S = A_6 - A_0 (true start), M_n = A_n(true) - A_n(wrong), kappa = (M_6 / "
                     "M_1)^(1/5) if M_1 > 0.01 and M_6 > 0, else 0 (OD-B5); nets without a checkpoint evaluated at MAXSTEPS (eval_point)")
    write_json(p, res)
    say(f"t5: pilot gate {'passed' if gate else 'failed'}; {len(allr)} (net, tau) rows ({time.time() - t0:.0f}s)")


# ====================================================================================================================== rules (t4 / rules)
def stage_rules(out):
    p = tdir(out, "verdict.json")
    if os.path.exists(p):
        say("rules: output exists, skipped:", p)
        return
    rows = write_nets_json(out)
    if not rows:
        raise SystemExit("rules need theta_toy/rows_t1.json (stage t1) first")
    choice = read_json(tdir(out, "t2.json")) if os.path.exists(tdir(out, "t2.json")) else dict(wds=[], descriptive=True, source="t2 not run")
    t5 = read_json(tdir(out, "t5.json")) if os.path.exists(tdir(out, "t5.json")) else None
    res = dict(stage="rules", measurement="B-T4 and the theta toy rules P1-P4, T5 phases (frozen, 4.2)", quick=QUICK, rng_seeds=RNG_SEEDS,
               G_meta="label applied in r2_ana (G35: G-meta from the round-1 PC summaries); not evaluated here")
    P1, B = {}, {}
    for W in T1_WIDTHS:
        b, cells = boundaries_of(rows, W)
        B[W] = b
        P1[str(W)] = {f"{wd:g}": dict(phase=p1_phase(c), **{k: c[k] for k in ("n", "n_level1", "n_meta", "B2_meta", "delta_h2")})
                      for wd, c in zip(WDS, cells)}
    res["P1"] = P1
    res["P2"] = {str(W): b for W, b in B.items()}
    res["P3"] = p3_transfer(rows, B[T1_WIDTHS[0]], B[T1_WIDTHS[-1]], choice)
    res["P4"] = p4_delta(rows, B[T1_WIDTHS[0]], B[T1_WIDTHS[-1]])
    res["B_T2"] = dict(choice=choice, cells={str(W): {f"{wd:g}": {k: v for k, v in meta_counts(rows, W, wd, T2_SEEDS).items()}
                                                      for wd in choice.get("wds", [])} for W in T2_WIDTHS}, od_g8=od_g8_t2(choice, rows))
    # B-T4: geometry and susceptibility per cell
    t4 = {}
    for W in list(T1_WIDTHS) + list(T2_WIDTHS):
        wds_W = WDS if W in T1_WIDTHS else choice.get("wds", [])
        cell = {}
        for wd in wds_W:
            rr = [r for r in rows if r["K"] == NPRE and r["W"] == W and abs(r["wd_scale"] - wd) < 1e-12]
            if not rr:
                continue
            g = lambda k, h="h2": [((r.get("delta") or {}).get(h) or {}).get(k) for r in rr]
            bm = [r["B2_meta"] for r in rr if r.get("B2_meta") is not None]
            mean = lambda v: float(np.mean([x for x in v if x is not None])) if any(x is not None for x in v) else None
            cell[f"{wd:g}"] = dict(n=len(rr), delta_h2=mean(g("delta")), delta_ctrl_h2=mean(g("delta_ctrl")), ring_h2=mean(g("ring")),
                                   delta_h1=mean(g("delta", "h1")), B2_meta_mean=float(np.mean(bm)) if bm else None,
                                   susceptibility=float(np.std(bm, ddof=1)) if len(bm) >= 2 else None, n_B2_meta=len(bm),
                                   first95_cens_mean=mean([r.get("first95_cens") for r in rr]))
        sus = [(k, v["susceptibility"]) for k, v in cell.items() if v["susceptibility"] is not None]
        t4[str(W)] = dict(cells=cell, susceptibility_peak=(dict(wd=max(sus, key=lambda kv: kv[1])[0], sd=max(sus, key=lambda kv: kv[1])[1])
                                                          if sus else None))
    pts = [(r["B2_meta"], r["delta"]["h2"]["delta"], r["W"]) for r in rows if r["K"] == NPRE and r.get("B2_meta") is not None and r.get("delta")]
    if len(pts) >= 3:
        from scipy.stats import pearsonr, spearmanr
        y, x, w = (np.array(v, float) for v in zip(*pts))
        sl = np.polyfit(x, y, 1)[0] if x.std() > 0 else None
        t4["B2_meta_vs_delta_h2"] = dict(n=len(pts), pearson=float(pearsonr(x, y)[0]) if x.std() > 0 and y.std() > 0 else None,
                                         spearman=float(spearmanr(x, y).correlation) if x.std() > 0 and y.std() > 0 else None,
                                         ols_slope=None if sl is None else float(sl), widths=sorted({int(v) for v in w}))
    t4["delta_h2_at_switch_on"] = {str(W): (None if B[W]["on"] is None else
                                            {f"{wd:g}": t4[str(W)]["cells"].get(f"{wd:g}", {}).get("delta_h2") for wd in (B[W]["on"]["lo"], B[W]["on"]["hi"])})
                                   for W in T1_WIDTHS}
    t4["peak"] = {str(W): t4[str(W)]["susceptibility_peak"] for W in list(T1_WIDTHS) + list(T2_WIDTHS) if str(W) in t4}
    res["B_T4"] = t4
    res["T5"] = t5_rules(t5) if t5 is not None else dict(note="t5 not run")
    write_json(p, res)
    say("rules: P1 " + "; ".join(f"W {W}: " + ", ".join(f"{wd}: {c['phase']}" for wd, c in P1[str(W)].items()) for W in T1_WIDTHS)
        + f" | P3 {res['P3']['verdict']} | P4 {res['P4']['verdict']}")


def t5_rules(t5):
    """T5 phases per (wd, W, tau), tau_c per (wd, W) and per net, and tau_c against delta_h2 and W (OLS, seeds as clusters; descriptive)."""
    rows = t5["rows"]
    taus = sorted({r["tau"] for r in rows})
    phases, tauc = {}, {}
    for W in sorted({r["W"] for r in rows}):
        for wd in sorted({r["wd"] for r in rows if r["W"] == W}):
            cells = []
            for tau in taus:
                ps = [dict(S=r["S"], M6=r["M6"], kappa=r["kappa"], seed=r["seed"]) for r in rows if r["W"] == W and r["wd"] == wd and r["tau"] == tau]
                cells.append(t5_cell_phase(ps))
                phases[f"W{W}_wd{wd:g}_tau{tau:g}"] = dict(W=W, wd=wd, tau=tau, **cells[-1])
            sw, note = tau_switch(taus, ["bistable" in c["labels"] for c in cells], ["dissipative" in c["labels"] for c in cells])
            tauc[f"W{W}_wd{wd:g}"] = dict(W=W, wd=wd, tau_c=sw, note=note)
    nets = {}
    for r in rows:
        nets.setdefault((r["W"], r["wd"], r["seed"]), {})[r["tau"]] = r
    per_net = []
    for (W, wd, seed), d in sorted(nets.items()):
        ts = sorted(d)
        sw, note = tau_switch(ts, [d[t]["M6"] > 0.1 and d[t]["kappa"] >= 0.8 for t in ts], [d[t]["M6"] < 0.1 for t in ts])
        per_net.append(dict(W=W, wd=wd, seed=seed, tau_c=sw, tau_c_mid=math.sqrt(sw[0] * sw[1]) if sw else None, note=note,
                            delta_h2=d[ts[0]].get("delta_h2")))
    reg = None
    pts = [r for r in per_net if r["tau_c_mid"] is not None and r["delta_h2"] is not None]
    if len(pts) >= 4 and len({r["W"] for r in pts}) >= 2 and len({r["seed"] for r in pts}) >= 2:
        X = np.column_stack([[r["delta_h2"] for r in pts], [math.log2(r["W"]) for r in pts]])
        y = np.array([r["tau_c_mid"] for r in pts])
        try:
            full, lo, hi = PA.cluster_boot_ols(X, y, [r["seed"] for r in pts], B=BOOT_DRAWS, seed=BOOT_SEED)
            reg = dict(n=len(pts), y="tau_c (geometric midpoint of the switch pair)", coef_delta_h2=float(full[1]), ci_delta_h2=[float(lo[1]), float(hi[1])],
                       coef_log2W=float(full[2]), ci_log2W=[float(lo[2]), float(hi[2])], clusters="seeds", draws=BOOT_DRAWS, boot_seed=BOOT_SEED,
                       interval="2.5 / 97.5 percentiles of the cluster-bootstrap draws (pc_analyze.cluster_boot_ols)")
        except (ValueError, IndexError, np.linalg.LinAlgError) as e:
            reg = dict(n=len(pts), error=repr(e))
    return dict(G27=t5.get("G27"), descriptive=not t5.get("G27", {}).get("passed", False), phases=phases, tau_c=tauc, tau_c_per_net=per_net,
                tau_c_regression=reg if reg is not None else dict(note="fewer than 4 nets with a tau_c (or one width / one seed): not fitted"))


# ====================================================================================================================== selftest
def _check(res, name, cond, info=""):
    res.append(dict(check=name, passed=bool(cond), info=jsonable(info)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {info}", flush=True)


def _synth_rules_dir(out, scenario):
    """Synthetic B-T1 / B-T2 rows, t2.json and t5.json for an end-to-end test of stage_rules (selftest only; no net is trained). On the
    module's grid: W 64 in meta at every wd but the largest (switch-on 'below grid'), W 256 in meta from the second to the second-largest
    wd (switch-on bracket = the two lowest wd values); B-T2 at the W 256 bracket ('fresh': 0 of 2 below, 2 of 2 above) or at two other wd
    values ('stale': t2.json from an earlier decision); t5: every net bistable below a seed- and width-dependent tau, dissipative above."""
    import shutil
    if os.path.isdir(tdir(out)):
        shutil.rmtree(tdir(out))

    def R(W, wd, s_, K, b2, dh2, st):
        d = None if K != NPRE else dict(h1=dict(delta=dh2, delta_ctrl=dh2, ring=0.0), h2=dict(delta=dh2, delta_ctrl=dh2, ring=0.0))
        return dict(name=job_name(W, s_, K, wd), arm="bp_mup", W=W, seed=s_, K=K, gamma=0.0, wd_scale=wd, level1=K == NPRE, delta=d, B2_8=b2,
                    stage=st, first95_cens=1000 if K == NPRE else None)

    def meta_fill(rows):
        for r in rows:
            if r["K"] == NPRE:
                k1 = [q for q in rows if q["K"] == 1 and q["W"] == r["W"] and q["seed"] == r["seed"] and q["wd_scale"] == r["wd_scale"]]
                r["B2_meta"] = r["B2_8"] - k1[0]["B2_8"]
        return rows
    in64, in256 = (lambda wd: wd <= WDS[-2]), (lambda wd: WDS[1] <= wd <= WDS[-2])
    r1 = meta_fill([R(W, wd, s_, K, (0.3 if f(wd) else 0.0) if K == NPRE else 0.0, 0.3 - 0.01 * i, "t1") for W, f in ((T1_WIDTHS[0], in64),
                    (T1_WIDTHS[-1], in256)) for i, wd in enumerate(WDS) for s_ in SEEDS for K in (NPRE, 1)])
    t2w = (WDS[0], WDS[1]) if scenario == "fresh" else (WDS[2], WDS[3])
    r2 = meta_fill([R(W, wd, s_, K, (0.3 if wd == WDS[1] else 0.0) if K == NPRE else 0.0, 0.25, "t2") for W in T2_WIDTHS for wd in t2w
                    for s_ in T2_SEEDS for K in (NPRE, 1)])
    ch = t2_choice(r1)
    if scenario != "fresh":
        ch = dict(wds=list(t2w), source="an earlier in-session decision (selftest: stale t2.json)", descriptive=False, boundaries=ch["boundaries"])
    write_json(tdir(out, "rows_t1.json"), r1)
    write_json(tdir(out, "rows_t2.json"), r2)
    write_json(tdir(out, "t2.json"), dict(ch, widths=list(T2_WIDTHS), seeds=list(T2_SEEDS), rng_seeds=RNG_SEEDS))
    t5rows = []
    for W in T1_WIDTHS:
        for wd in T5_WDS:
            for s_ in SEEDS:
                thr = TAUS[1 + (s_ + int(W == T1_WIDTHS[-1])) % 3]
                for tau in TAUS:
                    b = tau < thr
                    t5rows.append(dict(W=W, wd=wd, seed=s_, tau=tau, S=0.0, M6=0.3 if b else 0.0, kappa=0.9 if b else 0.0,
                                       delta_h2=0.3 + 0.01 * s_ + 0.02 * int(W == T1_WIDTHS[-1])))
    write_json(tdir(out, "t5.json"), dict(rows=t5rows, G27=dict(passed=True, label=None), rng_seeds=RNG_SEEDS))


def selftest(out):
    """Code tests (QUICK=1, CPU): the rule functions P1-P4 / T5 on synthetic cells, finetune_all against pc_meta.finetune(mode 'all'),
    the pc_summary-style rows against pc_analyze's row filters. Writes <out>/theta_toy/selftest.json; exit code 1 on a failure."""
    res = []
    c = lambda n, l1, m: dict(n=n, n_level1=l1, n_meta=m)
    _check(res, "P1 phases (4 seeds)", [p1_phase(c(4, 4, 3)), p1_phase(c(4, 4, 1)), p1_phase(c(4, 2, 0)), p1_phase(c(4, 4, 2)), p1_phase(c(4, 3, 3))]
           == ["meta", "no meta", "not generalised within budget", "boundary", "meta"])
    wds = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)
    b = p2_boundaries(wds, [0, 0, 0.25, 1, 1, 0])
    _check(res, "P2 switch-on and switch-off brackets with interpolation", b["on"] == dict(lo=0.3, hi=1.0, point=0.3 + 0.25 / 0.75 * 0.7) and
           b["off"] == dict(lo=3.0, hi=10.0, point=3.0 + 0.5 * 7.0), b)
    b = p2_boundaries(wds, [1, 1, 1, 0.5, 0, 0])
    _check(res, "P2 'below grid' and an off bracket across a 1/2 cell", b["on_flag"] == "below grid" and b["off"]["lo"] == 0.3 and b["off"]["hi"] == 3.0
           and abs(b["off"]["point"] - 1.0) < 1e-12, b)
    b = p2_boundaries(wds, [0, 0, 0, 0.5, 1, 1])
    _check(res, "P2 'above grid' (meta persists) and an on bracket", b["on"]["lo"] == 0.3 and b["on"]["hi"] == 3.0 and b["off_flag"] == "above grid", b)
    b = p2_boundaries(wds, [0, 0.25, 0.5, 0.5, 0.25, 0])
    _check(res, "P2 no meta cell -> no switch", b["on"] is None and b["off"] is None and b["on_flag"].startswith("no meta"), b)
    # P3 / P4 on synthetic rows
    def R(W, wd, seed, K, b2, dh2, l1=True):
        return dict(W=W, wd_scale=wd, seed=seed, K=K, B2_8=b2, level1=l1 if K == NPRE else False, delta=dict(h2=dict(delta=dh2)) if K == NPRE else None)
    rows = []
    for W, dshift in ((T1_WIDTHS[0], 0.0), (T1_WIDTHS[-1], 0.0)):
        for wd in wds:
            for s in SEEDS:
                meta = 0.3 <= wd <= 3.0 if W == T1_WIDTHS[0] else 1.0 <= wd <= 3.0
                rows += [R(W, wd, s, NPRE, 0.3 if meta else 0.0, 0.3 - 0.02 * wds.index(wd) + dshift), R(W, wd, s, 1, 0.0, None)]
    for W in T2_WIDTHS:
        for wd in (0.3, 1.0):
            for s in T2_SEEDS:
                rows += [R(W, wd, s, NPRE, 0.3 if wd == 1.0 else 0.0, 0.25), R(W, wd, s, 1, 0.0, None)]
    for r in rows:
        if r["K"] == NPRE:
            k1 = [q for q in rows if q["K"] == 1 and q["W"] == r["W"] and q["seed"] == r["seed"] and q["wd_scale"] == r["wd_scale"]]
            r["B2_meta"] = r["B2_8"] - k1[0]["B2_8"]
    b64, _ = boundaries_of(rows, T1_WIDTHS[0], wds)
    b256, _ = boundaries_of(rows, T1_WIDTHS[-1], wds)
    ch = t2_choice(rows, wds)
    p3 = p3_transfer(rows, b64, b256, ch)
    p4 = p4_delta(rows, b64, b256)
    _check(res, "B-T2 choice = the W 256 switch brackets", ch["wds"] == [0.3, 1.0, 3.0, 10.0] and not ch["descriptive"], ch["wds"])
    _check(res, "P3 transfers when the brackets intersect and W 512 is 0/2 below, 2/2 above", p3["verdict"] == "boundary transfers across width (muP)", p3)
    _check(res, "P4 'delta is the control variable' needs disjoint wd brackets", p4["verdict"] == "not shown" and p4["wd_brackets_intersect"], p4)
    # T5 phases and tau_c
    ph = t5_cell_phase([dict(S=0.1, M6=0.3, kappa=0.9)] * 3 + [dict(S=0.0, M6=0.0, kappa=0.0)])
    _check(res, "T5 phases can hold together ('self-improving + bistable')", ph["phase"] == "self-improving + bistable", ph)
    _check(res, "T5 'mixed' when nothing reaches 3 of 4", t5_cell_phase([dict(S=0.0, M6=0.3, kappa=0.9)] * 2 + [dict(S=0.0, M6=0.0, kappa=0.0)] * 2)["phase"] == "mixed")
    sw, note = tau_switch((0.1, 0.3, 0.5, 1.0, 2.0), [True, True, False, False, False], [False, False, True, True, True])
    _check(res, "tau_c = the adjacent pair where bistable switches to dissipative", sw == (0.3, 0.5) and note is None, sw)
    _check(res, "tau_c absent when bistable throughout", tau_switch((0.1, 0.3), [True, True], [False, False]) == (None, "bistable at every tau"))
    # review theta_toy-1: P3 'shifts' when W 64 has no switch-on bracket; 'not decidable' only when W 256 has none
    bw = lambda f: p2_boundaries(wds, f)
    b256 = bw([0, 0, 1, 1, 1, 0])
    rows_b = []
    for wd in (0.1, 0.3):
        for s_ in T2_SEEDS:
            rows_b += [dict(W=T2_WIDTHS[0], wd_scale=wd, seed=s_, K=NPRE, level1=True, B2_meta=0.3 if wd == 0.3 else 0.0)]
    chb = dict(wds=[0.1, 0.3, 3.0, 10.0], descriptive=False, source="test")
    pb = p3_transfer(rows_b, bw([1, 1, 1, 1, 1, 0]), b256, chb)
    pn = p3_transfer(rows_b, bw([0.5, 0.5, 1, 1, 1, 0]), b256, chb)
    pz = p3_transfer(rows_b, bw([1, 1, 1, 1, 1, 0]), bw([0, 0.25, 0.5, 0.5, 0.25, 0]), chb)
    _check(res, "theta_toy-1: P3 'shifts' when only W 64 lacks a switch-on bracket (direction from its flag); 'not decidable' only without a "
                "W 256 bracket", pb["verdict"] == "shifts" and "higher wd" in pb["direction"][0] and "below grid" in pb["direction"][0]
           and pn["verdict"] == "shifts" and "direction not determined" in pn["direction"][0] and pz["verdict"].startswith("not decidable (no switch-on")
           and pb["larger_widths"][T2_WIDTHS[0]]["inside"], dict(below=pb, not_bracketed=pn.get("direction"), no_256=pz["verdict"]))
    # review theta_toy-1 / -2 / -3 end to end through stage_rules on synthetic files
    sy = os.path.join(out, "synthetic_rules")
    for scen in ("fresh", "stale"):
        _synth_rules_dir(sy, scen)
        stage_rules(sy)
        v = read_json(tdir(sy, "verdict.json"))
        reg = v["T5"]["tau_c_regression"]
        if scen == "fresh":
            _check(res, "theta_toy-1 (stage_rules): W 64 'below grid', W 256 bracketed, B-T2 inside -> P3 'shifts', W 256 at higher wd",
                   v["P3"]["verdict"] == "shifts" and any("higher wd" in d_ for d_ in v["P3"]["direction"]), v["P3"])
            _check(res, "theta_toy-2: rng_seeds in verdict.json, t2.json and t5.json; the tau_c regression records its bootstrap seed and interval",
                   v.get("rng_seeds", {}).get("tau_c_cluster_boot", "").endswith(f"seed={BOOT_SEED}): np.random.default_rng({BOOT_SEED}) over the seeds")
                   and reg.get("boot_seed") == BOOT_SEED and "percentile" in reg.get("interval", "") and reg.get("draws") == BOOT_DRAWS
                   and "w6_sampling" in read_json(tdir(sy, "t5.json"))["rng_seeds"] and "rng_seeds" in read_json(tdir(sy, "t2.json")),
                   {k: reg.get(k) for k in ("n", "boot_seed", "draws", "interval")})
            _check(res, "theta_toy-3 (fresh): OD-G8 record agrees; B_T4 'peak' summary present", v["B_T2"]["od_g8"]["agree"] is True
                   and set(v["B_T4"]["peak"]) == {str(W) for W in T1_WIDTHS + T2_WIDTHS if str(W) in v["B_T4"]}, v["B_T2"]["od_g8"])
        else:
            _check(res, "theta_toy-3 (stale t2.json): P3 'not decidable' instead of 'shifts' from empty cells; OD-G8 disagreement reported",
                   v["P3"]["verdict"].startswith("not decidable (B-T2 has no") and v["B_T2"]["od_g8"]["agree"] is False
                   and v["P3"]["larger_widths"][str(T2_WIDTHS[0])]["lower_n"] == 0, dict(P3=v["P3"]["verdict"], od_g8=v["B_T2"]["od_g8"]))
    if QUICK:
        # finetune_all against pc_meta.finetune(mode 'all') on one trained tiny net
        tr = TC.train([dict(W=16, ws=1, K=NPRE, wd=3.0)])
        acc, ret, rep = finetune_all(tr.nets, [1], NPRE)
        pre, new, pairs, splits = M.world(1)
        m0 = tr.nets.to_mlp(0).to(DEV)
        ok, info = True, []
        tr_ = [M.data(pre[k], k, pairs[splits[k][144:]]) for k in range(NPRE)]
        Xva, yva = torch.cat([d[0] for d in tr_]), torch.cat([d[1] for d in tr_])
        for j in range(NNEW):
            a_ref, mm = M.finetune(m0, "bp_mup", 16, new, pairs, splits, j, 8, "all", 1)
            st = rep.state(j)
            rel = max(float((st[k] - v.cpu()).abs().max() / v.abs().max().clamp(min=1e-30)) for k, v in mm.state_dict().items())
            r_ref = M.accuracy(mm, Xva, yva) if hasattr(M, "accuracy") else float((mm(Xva).argmax(-1) == yva).float().mean())
            ok = ok and np.abs(acc[0, j] - np.array(a_ref)).max() <= 1 / 281 + 1e-9 and rel <= 1e-4 and abs(ret[0, j] - r_ref) <= 1e-5
            info.append(dict(task=j, acc=acc[0, j].tolist(), ref=a_ref, rel=rel, ret=float(ret[0, j]), ret_ref=r_ref))
        _check(res, "finetune_all = pc_meta.finetune(mode 'all') (decay_scale 1 although the net trained at wd 3)", ok, info)
        ra = TC.b2_all(tr.nets, [1])
        _check(res, "finetune_all = TC.b2_all for a K 12 net (the library's retention is the 12-task one; finetune_all uses the net's K tasks)",
               np.abs(acc - ra["acc"]).max() <= 1e-9 and np.abs(ret - ra["retention"]).max() <= 1e-6,
               dict(max_acc_diff=float(np.abs(acc - ra["acc"]).max()), max_ret_diff=float(np.abs(ret - ra["retention"]).max())))
        # rows: pc_summary keys and pc_analyze filters
        k12, k1 = TC.train([dict(W=16, ws=0, K=NPRE, wd=1.0)]), None
        inf12 = [dict(spec=dict(W=16, seed=0, K=NPRE, wd=1.0), row=k12.rows[0])]
        k1r = TC.train([dict(W=16, ws=0, K=1, wd=1.0)], fixed_steps=[k12.rows[0]["checkpoint"]])
        rows_ = cell_rows("t1", k12.nets, inf12, (k1r.nets, [dict(spec=dict(W=16, seed=0, K=1, wd=1.0), row=k1r.rows[0])]))
        need = ("name", "arm", "W", "seed", "K", "gamma", "wd_scale", "lr", "T", "beta", "first95", "checkpoint", "level1", "final_train",
                "final_heldout", "coord_check", "seconds", "delta", "B2_8", "B2_4", "A8_mean", "F8_mean", "F4_mean", "B2_all", "retention_mean")
        rr = json.loads(json.dumps(jsonable(rows_)))
        _check(res, "rows carry every pc_summary key; pc_analyze.cell finds both; B2_meta = B2_8(K 12) - B2_8(K 1)",
               all(all(k in r for k in need) for r in rr) and PA.cell(rr, "bp_mup", 16, 0) is not None and PA.cell(rr, "bp_mup", 16, 0, K=1) is not None
               and abs(rr[0]["B2_meta"] - (rr[0]["B2_8"] - rr[1]["B2_8"])) < 1e-12 and rr[1]["level1"] is False and rr[1]["delta"] is None,
               [r["name"] for r in rr])
    ok = all(r["passed"] for r in res)
    write_json(tdir(out, "selftest.json"), dict(passed=ok, checks=res, quick=QUICK, device=str(DEV)))
    print(f"selftest: {sum(r['passed'] for r in res)}/{len(res)} checks passed", flush=True)
    if not ok:
        raise SystemExit(1)


# ====================================================================================================================== main
def main(argv):
    if len(argv) < 3 or argv[2] not in STAGES + ("selftest",):
        print(__doc__.split("\n\n")[0] + "\n\n" + __doc__.split("\n\n")[1])
        raise SystemExit(2)
    out, stage = argv[1], argv[2]
    torch.manual_seed(0)
    say(f"r2_theta_toy {stage}: out {out}, device {DEV}, QUICK {QUICK}, torch {torch.__version__}, threads {torch.get_num_threads()}")
    t0 = time.time()
    dict(t1=stage_t1, t2=stage_t2, t5=stage_t5, rules=stage_rules, t4=stage_rules, selftest=selftest)[stage](out)
    say(f"r2_theta_toy {stage} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv)
