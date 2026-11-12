"""Round 2, Part A toy: RGI in the one-hot shift world (frozen plan results/ju_space/plans/abschnitt11_12_runde2_plan.md, section
3.2 AT0-AT6 with the shared toy objects of 2.5 (W0-W2, W7), statistics conventions 2.6, gates G38-G40 of section 6, session r2_t1 of
section 7, CLI of 9.3). Implementer 6. Every net is trained in this session by r2_toy_common's batched trainer (W0); nothing of round 1
is read.

Usage:  python r2_rgi_toy.py <out_dir> <stage>
          stage r0       AT0  calibration of the delta probe            -> <out_dir>/rgi_toy/r0.json
          stage r1       AT2 with AT1 and AT3 (8 worlds x 3 inits)      -> rgi_toy/r1/w<ws>.json (one per world) + rgi_toy/r1/at3.json
          stage cross    AT5 among the BP nets (worlds 0-4)             -> rgi_toy/cross/w<ws>.json, rgi_toy/models/at5_<name>.pt
          stage fam      AT4 family arm (after cross: uses AT5's nets)  -> rgi_toy/fam.json
          stage forcing  AT6 forcing controls                           -> rgi_toy/forcing.json
          stage selftest code tests of this script (CPU; run with QUICK=1)
        env DEVICE (cuda | cpu), NT (torch threads), QUICK (1 = code test: tiny grids and widths, pc_meta's QUICK sizes)
Resumable: a stage whose output exists is skipped; r1 and cross skip finished worlds; trained nets are saved (rgi_toy/models/) and
reloaded (weight checksums verified) when a stage is rerun before its outputs exist. JSON is written atomically.

Definitions (plan text governs; refs rgi_design/*.py and crit/*.py of Appendix E, ported in r2_toy_common as TC):
- Nets: BP-muP W 128 unless stated; init 0 = torch.manual_seed(ws) (= pc_meta), init i > 0 = torch.manual_seed(10000 i + ws) (W1);
  checkpoint = first95 + 500, else evaluated at MAXSTEPS (6,000) and marked (W0); late regulariser branches start from the natural run's
  parameters and AdamW state at first95 and run 500 updates (level 1 = held-out >= 0.95 at the branch's end); early = from step 1 (W2).
- Lines per model (AT1): Lambda (zero-shot recipes C_a + C_b - C_c over all anchor triples aimed at each of the 5 unseen shifts, mean over
  triples then shifts; nulls 'wrong shift' and 'permuted anchor labels'); free B2 column (8 shots, 300 Adam steps lr 1e-2 from zero;
  bottleneck nets: the dt coefficients, v = U e, ref rgi_bneck); alpha (12 anchor coordinates, same search); X = acc_A(alpha) /
  acc_A(free); Phi = acc_A(projection of the free columns on the linear span of the anchor columns) / acc_A(free); recursion: recipe
  depth 1-6 (retention rho_d), round trip A -> B (40 steps) -> A against A's own +40 steps (RG), pooling, warm start (10 / 40 steps).
- Lines per ordered pair A -> B: S-gain = acc_B(v_B = mean_B + C_B,centred alpha_A) - mean over 30 permuted correspondences; free-column
  T1 gain (affine ridge on the shared anchors, lambda by leave-one-anchor-out column error, minus the mean of 10 shuffled maps) and
  LOO-anchor transfer; T4 (B's column distilled on the new task's 289 unlabelled inputs to A's output distribution, 300 Adam steps lr
  1e-2); T3 (h2 stitching, affine ridge on 12 tasks x 289 pairs with leave-3-tasks-out lambda; AT5 at W <= 256).
- All accuracies are held-out accuracies on the new task's 281 non-shot pairs (or a task's 145 held-out pairs for LOO anchors).
Decisions taken here (AT0's level selection, the AT4 ReLU lr) use the functions select_levels / relu_calibration of this file; the
verdicts (AT7, AT8, G38, G39, the AT5 slope label) are made by r2_rgi_verdict.py in r2_ana from these files. Ambiguities and their
resolution: NOTES_w6.md."""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import r2_toy_common as TC          # noqa: E402  sets T / BETA / LR / MAXSTEPS before importing pc_meta (W0)

M = TC.M
P, NPRE, NNEW = TC.P, TC.NPRE, TC.NNEW
QUICK, DEV = TC.QUICK, TC.DEV
F32 = torch.float32

# ====================================================================================================================== grids
STAGES = ("r0", "r1", "cross", "fam", "forcing")
W_MAIN = 24 if QUICK else 128                                              # 3.2: BP-muP W 128 unless stated
R0_WORLD, R0_INITS = 0, (0, 1)                                             # AT0 / OD-A7
R0_TASK_GRID = (0.02, 0.01, 0.006, 0.003, 0.0015, -0.002, -0.004, -0.007, -0.01, -0.02, -0.03)
R0_TARGETS = (-2, -1, 1, 2)                                                # x = log2(delta_h2 / delta_nat) of 1/4, 1/2, 2, 4
R0_TOL = 0.4
SHAM_LATE = (0.01, -0.01, -0.03, -0.1)                                     # fixed sham late levels (AT0, AT2)
EARLY = (0.003, -0.003, -0.01)                                             # fixed early levels, task and sham
R1_WORLDS = tuple(range(2 if QUICK else 8))                                # AT2: worlds 0-7, inits 0-2
INITS3 = (0, 1, 2)
EXTRA2 = 100 if QUICK else TC.SECOND_AFTER                                 # natural runs continued to first95 + 1500
X_WORLDS = tuple(range(2 if QUICK else 5))                                 # AT5 / OD-A10: worlds 0-4
X_WIDTHS = (16, 24) if QUICK else (64, 128, 256, 1024)
T3_MAXW = 16 if QUICK else 256                                             # T3 only at W <= 256 (sender and receiver)
FAM_WORLDS = X_WORLDS                                                      # AT4: worlds 0-4
FAM_INIT = 7                                                               # B variants: torch.manual_seed(70000 + ws) (ref family*.py)
RELU_LRS = (2e-3, 5e-3)                                                    # OD-A9
F_WORLDS, F_INITS = (0, 1), (0, 1)                                         # AT6
F_WDS = (3.0, 5.0)
SHARED_STEPS = 100 if QUICK else 10000
BNECK_DT = 4
NSHUF, NPERM_S = 10, 30
RT_STEPS, WARM_STEPS = 40, (10, 40)
CH = 4096                                                                  # chains per host chunk of an accuracy evaluation
SEARCH_FLOATS = 2 ** 27 if QUICK else 2 ** 29                              # activation budget of one batched search call


# ====================================================================================================================== I/O
def jsonable(o):
    """JSON-safe copy (numpy / torch -> lists, non-finite floats -> None)."""
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


def save_bundle(path, nets, infos, extra=None):
    """All nets of one architecture in one file (atomic): parameters, tables, per-net info dicts, weight checksums."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = dict(arch=dict(kind=nets.kind, W=nets.W, depth=nets.depth, act=nets.act, param=nets.param),
               p={k: v.detach().cpu().clone() for k, v in nets.p.items()}, tables=[nets.fx.cpu(), nets.fy.cpu(), nets.ft.cpu()],
               meta=jsonable(nets.meta), infos=jsonable(infos), checksums=[TC.weight_checksum(nets, i) for i in range(nets.B)],
               extra=jsonable(extra))
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_bundle(path):
    """(nets, infos, extra) of save_bundle; raises if a weight checksum differs."""
    o = torch.load(path, map_location="cpu", weights_only=False)
    a = o["arch"]
    nets = TC.Nets(a["kind"], a["W"], a["depth"], a["act"], a["param"], {k: v.to(DEV) for k, v in o["p"].items()},
                   *(t.to(DEV) for t in o["tables"]), meta=o["meta"])
    for i, c in enumerate(o["checksums"]):
        if TC.weight_checksum(nets, i) != c:
            raise ValueError(f"weight checksum mismatch in {path} (net {i})")
    return nets, o["infos"], o.get("extra")


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ====================================================================================================================== models
def info_of(spec, row, **kw):
    """Per-model record: identity, data definition (world, task family ky, anchor slots and their shifts, pretraining split) and the
    training row of TC.train (first95, checkpoint, level1, eval_point, held-out, checksum)."""
    s = TC.full_spec(spec)
    pre = list(M.world(s["ws"])[0])
    shifts = [int(c) for c in (s["shifts"] if s["shifts"] is not None else pre[:s["K"]])]
    d = dict(name=TC.net_name(s), ws=int(s["ws"]), init=int(s["init"]), W=int(s["W"]), arm=s["arm"], kind=s["kind"], depth=s["depth"],
             act=s["act"], lr=s["lr"], wd=s["wd"], ky=int(s["ky"]), split=s["split"] if isinstance(s["split"], str) else "custom",
             coding=s["coding"] if (s["coding"] is None or isinstance(s["coding"], str)) else "custom", gamma=s["gamma"],
             reg=s["reg"] if s["gamma"] else None, when=s["when"] if s["gamma"] else None,
             anchors=list(range(len(shifts))), shifts=shifts)
    if row is not None:
        d.update(first95=row.get("first95"), checkpoint=row.get("checkpoint"), level1=bool(row.get("level1")),
                 eval_point=row.get("eval_point"), updates=row.get("updates"), final_heldout=row.get("final_heldout"),
                 final_train=row.get("final_train"), checksum=row.get("checksum"), coord_check=row.get("coord_check"))
    d.update(kw)
    return d


def late_fields(f95):
    """Row fields of a late branch (W2): it starts at the natural run's first95 and ends after AFTER (500) updates, i.e. at the natural
    checkpoint position first95 + 500, where it is evaluated; its level 1 is held-out >= 0.95 there (the training row's level1)."""
    return dict(first95=f95, checkpoint=f95 + TC.AFTER, updates=f95 + TC.AFTER, updates_total=f95 + TC.AFTER, branch_updates=TC.AFTER,
                eval_point="checkpoint")


def r1_job_name(info, K=NPRE):
    """pc_meta's job name of a round-1 cell with the same definition (for G46's reproduction check; only init 0 nets have one)."""
    return f"{info['arm']}_W{info['W']}_s{info['ws']}_K{K}_g0_wd{info['wd']:g}"


class Pool:
    """Models stored in groups of TC.Nets (one architecture per group); model m = (group g, index i in the group) plus its info."""

    def __init__(self):
        self.groups, self.models = [], []

    def add(self, nets, infos):
        g = len(self.groups)
        self.groups.append(nets)
        out = []
        for i, inf in enumerate(infos):
            self.models.append(dict(inf, g=g, i=i))
            out.append(len(self.models) - 1)
        return out

    def nets(self, m):
        return self.groups[self.models[m]["g"]]

    def by_group(self, mids):
        out = {}
        for m in mids:
            out.setdefault(self.models[m]["g"], []).append(m)
        return out

    def sub(self, g, ms):
        """Nets of the models ms (all in group g), as a subset copy unless they are the whole group in order."""
        nets = self.groups[g]
        loc = [self.models[m]["i"] for m in ms]
        return nets if loc == list(range(nets.B)) else nets.subset(loc)


_SETS = {}


def nsets(ws, j, ky=1, nshot=TC.NSHOT):
    """New task j of world ws (TC.new_task_sets: shots = split positions 0-7, held-out = 8-288), cached."""
    key = (int(ws), int(j), int(ky), int(nshot))
    if key not in _SETS:
        _SETS[key] = TC.new_task_sets(int(ws), int(j), ky=int(ky), nshot=int(nshot))
    return _SETS[key]


def split_perm(inf, slot):
    """(permutation of the 289 pairs, number of training pairs) of a model's pretraining task in slot."""
    if inf["split"] == "world":
        return M.world(inf["ws"])[3][slot], 144
    if inf["split"] == "other":
        return TC.other_splits(inf["ws"])[slot], 144
    if inf["split"] == "shared":
        sh, ntr = TC.shared_split(inf["ws"])
        return sh, ntr
    raise ValueError(inf["split"])


def anchor_ho(inf, a):
    """Held-out pairs (x, y, label) of the model's pretraining task at anchor position a (LOO-anchor evaluation)."""
    slot, c = inf["anchors"][a], inf["shifts"][a]
    perm, ntr = split_perm(inf, slot)
    idx = np.asarray(perm)[ntr:]
    x, y = TC.PAIRS[idx, 0], TC.PAIRS[idx, 1]
    return x, y, (x + inf["ky"] * y + c) % P


class Evals:
    """Collects candidate columns to evaluate (per group and held-out size), evaluates them in host chunks of CH chains with
    TC.cols_accuracy (held-out accuracy of a column replacing the task column), and hands the accuracies back per job."""

    def __init__(self, pool):
        self.pool, self.jobs, self.res = pool, {}, {}

    def add(self, m_or_g, cols, x, y, lab, group=False):
        """Columns cols [k, W] for model m (or, with group=True, for (g, i)) on sets x, y, lab ([N] shared or [k, N])."""
        if group:
            g, i = m_or_g
        else:
            g, i = self.pool.models[m_or_g]["g"], self.pool.models[m_or_g]["i"]
        cols = np.asarray(cols.detach().cpu().numpy() if isinstance(cols, torch.Tensor) else cols, dtype=np.float32)
        k = cols.shape[0]
        sets = [np.broadcast_to(np.asarray(a, dtype=np.uint8), (k, np.asarray(a).shape[-1])) for a in (x, y, lab)]
        key = (g, sets[0].shape[1])
        lst = self.jobs.setdefault(key, [])
        lst.append((i, cols, sets))
        return key, len(lst) - 1

    def run(self):
        for key, lst in self.jobs.items():
            g, N = key
            nets = self.pool.groups[g]
            out = [None] * len(lst)
            buf, bi = [], 0

            def flush(buf):
                net_of = np.concatenate([np.full(len(c), i) for _, i, c, _ in buf])
                cols = np.concatenate([c for _, _, c, _ in buf])
                xs, ys, ls = (np.concatenate([s[q] for _, _, _, s in buf]).astype(np.int64) for q in range(3))
                acc = TC.cols_accuracy(nets, net_of, cols, (xs, ys, ls))
                p = 0
                for n, _, c, _ in buf:
                    out[n] = acc[p:p + len(c)]
                    p += len(c)

            for n, (i, cols, sets) in enumerate(lst):
                buf.append((n, i, cols, sets))
                bi += len(cols)
                if bi >= CH:
                    flush(buf)
                    buf, bi = [], 0
            if buf:
                flush(buf)
            self.res[key] = out
        self.jobs = {}
        return self

    def get(self, h):
        return self.res[h[0]][h[1]]


def search_chunks(nets, n_inputs, C):
    """Chain slices for batched searches so that one call's activations stay within SEARCH_FLOATS floats."""
    per = max(1, int(SEARCH_FLOATS // max(1, n_inputs * nets.W * 8)))
    return [slice(a, min(C, a + per)) for a in range(0, C, per)]


def chain_logits(nets, net_of, cols, x, y):
    """Logits [C, N, 17] of columns cols [C, W] in nets net_of on inputs x, y ([N] shared or [C, N])."""
    _, h = TC.hidden_states(nets, np.asarray(net_of), cols, (x, y))
    no = torch.as_tensor(np.asarray(net_of), device=DEV)
    W3, b3 = nets.p["W3"][no], nets.p["b3"][no]
    return torch.baddbmm(b3.unsqueeze(1), h, W3.transpose(1, 2)) * nets.mult


def task_cols_np(pool):
    """{model: anchor columns [W, n] float64} for every model of the pool (one task_cols() call per group)."""
    out = {}
    for g, nets in enumerate(pool.groups):
        T = nets.task_cols().double().cpu().numpy()
        for m, inf in enumerate(pool.models):
            if inf["g"] == g:
                out[m] = T[inf["i"]][:, inf["anchors"]]
    return out


def carry_cols(pool, m, alpha, slots=None, tc=None):
    """Carried anchored improvement in model m: v = mean(C) + (C - mean(C)) alpha over m's anchor columns at slots (TC.carry_alpha's
    construction, float32). alpha [k, n] -> columns [k, W] (torch)."""
    inf = pool.models[m]
    slots = inf["anchors"] if slots is None else list(slots)
    T = tc[inf["g"]] if tc is not None else pool.groups[inf["g"]].task_cols()
    C = T[inf["i"]][:, slots]
    v0 = C.mean(-1)
    Bm = C - v0.unsqueeze(-1)
    al = torch.as_tensor(np.asarray(alpha), dtype=F32, device=DEV)
    return v0 + al @ Bm.T


class TCache(dict):
    """Lazy per-group task_cols() cache."""

    def __init__(self, pool):
        super().__init__()
        self.pool = pool

    def __missing__(self, g):
        v = self.pool.groups[g].task_cols()
        self[g] = v
        return v


def ho_stack(ws, ky, rep=1):
    """Held-out sets of the 4 new tasks of world ws (task family ky), each repeated rep times: x, y, lab [4 rep, 281]."""
    sets = [nsets(ws, j, ky)["ho"] for j in range(NNEW)]
    return tuple(np.repeat(np.stack([s[q] for s in sets]), rep, 0) for q in range(3))


# ====================================================================================================================== lines per model
def free_columns(pool, mids, nshot=TC.NSHOT):
    """AT1's free improvement = the B2 column (pc_meta.finetune 'col': the new task's slot column from zero, 8 shots, Adam lr 1e-2,
    FT_STEPS steps, held-out 281 pairs at FT_EVAL); bottleneck nets learn their dt coefficients (v = U e, ref rgi_bneck.geometry_b).
    Returns {m: dict(v [4, W] float64, acc [4, len(FT_EVAL)], acc_end [4] (after FT_STEPS), A [4] (mean over FT_EVAL))}."""
    out = {}
    for g, ms in pool.by_group(mids).items():
        nets = pool.groups[g]
        net_of, sets = [], []
        for m in ms:
            inf = pool.models[m]
            for j in range(NNEW):
                net_of.append(inf["i"])
                sets.append(nsets(inf["ws"], j, inf["ky"], nshot))
        net_of = np.array(net_of)
        sh, ho = TC.stack_sets(sets, "shot"), TC.stack_sets(sets, "ho")
        acc = np.zeros((len(net_of), len(TC.FT_EVAL)))
        v = np.zeros((len(net_of), nets.W))
        for sl in search_chunks(nets, nshot, len(net_of)):
            param = TC.coords_param(nets, net_of[sl], "bneck") if nets.kind == "bneck" else None
            r = TC.column_search(nets, net_of[sl], tuple(a[sl] for a in sh), tuple(a[sl] for a in ho), steps=TC.FT_STEPS,
                                 eval_at=TC.FT_EVAL, param=param)
            acc[sl], v[sl] = r["acc"]["ho"], r["v"]
        for n, m in enumerate(ms):
            s = slice(n * NNEW, (n + 1) * NNEW)
            out[m] = dict(v=v[s].astype(np.float64), acc=acc[s], acc_end=acc[s, -1], A=acc[s].mean(1))
    return out


def alphas(pool, mids, slots_of=None):
    """S line / W7 anchored search: alpha over the model's anchor columns (slots_of[m], default its anchors; centred, from zero; 8 shots,
    Adam lr 1e-2, FT_STEPS steps; TC.learn_alpha). Returns {m: dict(alpha [4, n], acc [4] (held-out after the search), slots)}."""
    out = {}
    for g, ms in pool.by_group(mids).items():
        nets = pool.groups[g]
        by = {}
        for m in ms:
            by.setdefault(tuple(pool.models[m]["anchors"] if slots_of is None else slots_of[m]), []).append(m)
        for anc, mm in by.items():
            net_of, sets = [], []
            for m in mm:
                inf = pool.models[m]
                for j in range(NNEW):
                    net_of.append(inf["i"])
                    sets.append(nsets(inf["ws"], j, inf["ky"]))
            q, acc, _ = TC.learn_alpha(nets, np.array(net_of), sets, anchor_idx=list(anc))
            for n, m in enumerate(mm):
                s = slice(n * NNEW, (n + 1) * NNEW)
                out[m] = dict(alpha=np.asarray(q[s]), acc=np.asarray(acc[s]), slots=list(anc))
    return out


def lambda_targets_of(inf):
    """The model's unseen Lambda targets: TC.lambda_targets(ws) (the 4 new shifts, then the spare shift) minus any shift among its
    anchors (the '8 shared shifts + spare' variant pretrained on the spare shift)."""
    return [t for t in TC.lambda_targets(inf["ws"]) if t[0] not in inf["shifts"]]


def lambdas(pool, mids):
    """AT1 Lambda per model (TC.lambda_line over the model's anchors and their shifts, its task family and its unseen targets): mean over
    the unseen target shifts of the mean held-out accuracy over the recipe triples; nulls 'wrong' (triples aimed at another shift) and
    'perm' (permuted anchor labels, 10 permutations)."""
    out = {}
    for g, ms in pool.by_group(mids).items():
        sub = pool.sub(g, ms)
        infs = [pool.models[m] for m in ms]
        tg = [lambda_targets_of(f) for f in infs]
        res = TC.lambda_line(sub, [f["ws"] for f in infs], pre_of=[f["shifts"] for f in infs], anchor_idx=[f["anchors"] for f in infs],
                             ky_of=[f["ky"] for f in infs], targets_of=tg)
        for m, t, r in zip(ms, tg, res):
            out[m] = dict(Lambda=r["Lambda"], per_shift=r["per_shift"], targets=[int(c) for c, _, _ in t], wrong=r.get("wrong"),
                          perm=r.get("perm"))
    return out


def recipe_depths(pool, mids):
    """AT1 recursion: recipe depth 1-6 (TC.recipe_depth: the same 200 chains in every model) -> {m: dict(acc [6], rho [5])}."""
    out = {}
    for g, ms in pool.by_group(mids).items():
        sub = pool.sub(g, ms)
        res = TC.recipe_depth(sub, [pool.models[m]["ws"] for m in ms], pre_of=[pool.models[m]["shifts"] for m in ms])
        out.update({m: r for m, r in zip(ms, res)})
    return out


def deltas(pool, mids):
    """Toy delta (pc_meta.delta_stats, vectorised: h1 and h2 over the 12 pretraining tasks x 289 pairs; permutation stream 77 + ws)."""
    out = {}
    for g, ms in pool.by_group(mids).items():
        ms = [m for m in ms if pool.models[m]["anchors"] == list(range(NPRE))]
        if not ms:
            continue
        sub = pool.sub(g, ms)
        res = TC.delta_stats(sub, [pool.models[m]["ws"] for m in ms], [pool.models[m]["shifts"] for m in ms])
        out.update({m: r for m, r in zip(ms, res)})
    return out


def mediators(pool, mids, chunk=32):
    """AT3 mediators (TC.mediators): pooled held-out accuracy on the 12 pretraining tasks, mean L2 norm of the 12 task columns,
    saturation share of |h1| > 0.95 over the 12 tasks x 289 pairs and the units (chunks of nets to bound memory)."""
    out = {}
    for g, ms in pool.by_group(mids).items():
        for a in range(0, len(ms), chunk):
            part = ms[a:a + chunk]
            r = TC.mediators(pool.sub(g, part), [pool.models[m]["ws"] for m in part])
            for n, m in enumerate(part):
                out[m] = dict(heldout_pre=float(r["heldout"][n]), colnorm=float(r["task_col_norm"][n]), sat_h1=float(r["saturation"][n]))
    return out


def phis(pool, mids, fc, tcols):
    """AT1 F, per model: Phi = acc_A(projection of A's free columns on the linear span of its anchor columns) / acc_A(free) (mean over the
    4 new tasks of each; TC.f_line's definition)."""
    ev, hs = Evals(pool), {}
    for m in mids:
        inf = pool.models[m]
        proj = np.stack([TC.proj_linspan(tcols[m], np.asarray(v, np.float64)) for v in fc[m]["v"]])
        hs[m] = ev.add(m, proj, *ho_stack(inf["ws"], inf["ky"]))
    ev.run()
    out = {}
    for m in mids:
        accp, den = ev.get(hs[m]), float(np.mean(fc[m]["acc_end"]))
        out[m] = dict(Phi=float(np.mean(accp) / den) if den > 0 else float("nan"), acc_proj=accp)
    return out


# ====================================================================================================================== lines per pair
def f_pairs(pool, pairs, fc, tcols, nshuf=NSHUF, loo=True):
    """AT1 F per ordered pair (TC.f_line generalised to shared anchor subsets and to the receiver's own labels and splits): pairs =
    [(a, b, ia, ib)], ia / ib = positions in a's / b's anchor lists of the shared anchors (same order). T1 = affine ridge map through the
    shared anchors (lambda by leave-one-anchor-out column error, TC.t1_choose_r); a's 4 free columns carried to b and evaluated on b's
    new-task held-out pairs (b's task family), against nshuf shuffled maps (default_rng(123 + ws_a), identity excluded): t1, t1_shuf,
    t1_gain = mean over tasks of (t1 - mean shuffled). LOO-anchor: each shared anchor k carried by the map fitted on the others into b's
    slot, evaluated on b's held-out pairs of that task (b's split and labels), against nshuf shuffled maps (default_rng(1000 k + q))."""
    ev, jobs = Evals(pool), []
    for a, b, ia, ib in pairs:
        A, B = pool.models[a], pool.models[b]
        CA, CB = tcols[a][:, ia], tcols[b][:, ib]
        n = len(ia)
        r1, e1 = TC.t1_choose_r(CA, CB)
        rng = np.random.default_rng(123 + A["ws"])
        perms = [p for p in (rng.permutation(n) for _ in range(nshuf)) if np.any(p != np.arange(n))]
        nper = 1 + len(perms)
        cands = []
        for j in range(NNEW):
            vA = fc[a]["v"][j]
            cands.append(TC.t1_map(CA, CB, vA, r1)[0])
            cands += [TC.t1_map(CA, CB[:, p], vA, r1)[0] for p in perms]
        h_new = ev.add(b, np.stack(cands), *ho_stack(B["ws"], B["ky"], nper))
        h_loo = None
        if loo:
            cands, xs, ys, ls = [], [], [], []
            for k in range(n):
                keep = [i for i in range(n) if i != k]
                c_ = [TC.t1_map(CA[:, keep], CB[:, keep], CA[:, k], r1)[0]]
                c_ += [TC.t1_map(CA[:, keep], CB[:, keep][:, np.random.default_rng(k * 1000 + q).permutation(n - 1)], CA[:, k], r1)[0]
                       for q in range(nshuf)]
                x, y, lab = anchor_ho(B, ib[k])
                cands += c_
                xs += [x] * len(c_)
                ys += [y] * len(c_)
                ls += [lab] * len(c_)
            h_loo = ev.add(b, np.stack(cands), np.stack(xs), np.stack(ys), np.stack(ls))
        jobs.append((r1, e1, nper, h_new, h_loo))
    ev.run()
    out = []
    for r1, e1, nper, h_new, h_loo in jobs:
        acc = ev.get(h_new).reshape(NNEW, nper)
        d = dict(r1=r1, loo_err1=e1, t1=float(acc[:, 0].mean()), t1_shuf=float(acc[:, 1:].mean()),
                 t1_gain=float((acc[:, 0] - acc[:, 1:].mean(1)).mean()), t1_tasks=acc[:, 0])
        if h_loo is not None:
            al = ev.get(h_loo).reshape(-1, 1 + nshuf)
            d.update(loo=float(al[:, 0].mean()), loo_shuf=float(al[:, 1:].mean()))
        out.append(d)
    return out


def s_pairs(pool, pairs, nperm=NPERM_S):
    """AT1 S per ordered pair (TC.s_line / carry_alpha): pairs = [(a, b, alpha_a [4, n], slots_b)]: v_B = mean_B + C_B,centred alpha_a
    over b's anchor columns at slots_b, against nperm permuted correspondences (TC.s_perms(ws_a, nperm, n)); held-out = b's new-task
    pairs (b's task family). S_gain = mean over tasks of acc_B - mean over permutations."""
    ev, hs, tc = Evals(pool), [], TCache(pool)
    for a, b, alpha, slots_b in pairs:
        A, B = pool.models[a], pool.models[b]
        Cb = tc[B["g"]][B["i"]][:, list(slots_b)]
        v0 = Cb.mean(-1)
        Bm = Cb - v0.unsqueeze(-1)
        al = torch.as_tensor(np.asarray(alpha), dtype=F32, device=DEV)
        mats = [Bm] + [Bm[:, torch.as_tensor(np.asarray(p), device=DEV)] for p in TC.s_perms(A["ws"], nperm, len(slots_b))]
        cols = torch.stack([v0 + al @ Mx.T for Mx in mats], 1).reshape(NNEW * (1 + nperm), -1)
        hs.append(ev.add(b, cols, *ho_stack(B["ws"], B["ky"], 1 + nperm)))
    ev.run()
    out = []
    for h in hs:
        acc = ev.get(h).reshape(NNEW, 1 + nperm)
        out.append(dict(S_gain=float((acc[:, 0] - acc[:, 1:].mean(1)).mean()), accB=acc[:, 0], accB_perm=acc[:, 1:].mean(1)))
    return out


def recursion(pool, pairs, al, steps=RT_STEPS, warm=WARM_STEPS):
    """AT1 recursion per ordered pair (a, b) of standard models of one world (same shots): round trip (TC.round_trip: alpha_a refined
    in b for 40 steps on b's shots, carried back to a, against a's own 40 further steps; RG = mean over tasks of acc_a(back) -
    acc_a(own)) and warm start (TC.warm_start: b refines from the carried alpha_a; accuracy after 10 and 40 steps minus the same from
    zero). All refinements are anchored searches (Adam lr 1e-2, fresh moments) run in one batched call per group."""
    chains = []
    for pi, (a, b) in enumerate(pairs):
        chains += [("w", b, j, al[a]["alpha"][j], pi) for j in range(NNEW)]
    for b in sorted({b for _, b in pairs}):
        chains += [("z", b, j, np.zeros_like(al[b]["alpha"][j]), None) for j in range(NNEW)]
    for a in sorted({a for a, _ in pairs}):
        chains += [("o", a, j, al[a]["alpha"][j], None) for j in range(NNEW)]
    ev_at = tuple(sorted(set(warm) | {steps}))
    acc = np.zeros((len(chains), len(ev_at)))
    qend = [None] * len(chains)
    by = {}
    for c, (_, m, _, _, _) in enumerate(chains):
        by.setdefault(pool.models[m]["g"], []).append(c)
    for g, cs in by.items():
        nets = pool.groups[g]
        net_of = np.array([pool.models[chains[c][1]]["i"] for c in cs])
        sets = [nsets(pool.models[chains[c][1]]["ws"], chains[c][2], pool.models[chains[c][1]]["ky"]) for c in cs]
        q0 = np.stack([chains[c][3] for c in cs]).astype(np.float32)
        r = TC.column_search(nets, net_of, TC.stack_sets(sets, "shot"), TC.stack_sets(sets, "ho"), steps=steps, eval_at=ev_at,
                             param=TC.anchored_param(nets, net_of), q0=q0)
        acc[cs] = r["acc"]["ho"]
        for n, c in enumerate(cs):
            qend[c] = r["q"][n]
    idx = {(k, m, j): c for c, (k, m, j, _, pi) in enumerate(chains) if k != "w"}
    widx = {(pi, j): c for c, (k, m, j, _, pi) in enumerate(chains) if k == "w"}
    ev, hs, tc = Evals(pool), [], TCache(pool)
    for pi, (a, b) in enumerate(pairs):
        aB = np.stack([qend[widx[(pi, j)]] for j in range(NNEW)])
        hs.append(ev.add(a, carry_cols(pool, a, aB, tc=tc), *ho_stack(pool.models[a]["ws"], pool.models[a]["ky"])))
    ev.run()
    s_last = ev_at.index(steps)
    out = []
    for pi, (a, b) in enumerate(pairs):
        back = ev.get(hs[pi])
        own = np.array([acc[idx[("o", a, j)], s_last] for j in range(NNEW)])
        d = dict(RG=float(np.mean(back - own)), back=back, own=own)
        for s in warm:
            k = ev_at.index(s)
            w_ = np.array([acc[widx[(pi, j)], k] for j in range(NNEW)])
            z_ = np.array([acc[idx[("z", b, j)], k] for j in range(NNEW)])
            d[f"warm{s}"] = float(np.mean(w_ - z_))
        out.append(d)
    return out


def pooling_lines(pool, groups_of_models, al):
    """AT1 recursion pooling (TC.pooling): per list of models (one world and condition) the alphas averaged over the models, evaluated in
    each model against its own alpha. gain = mean over models of (mean over tasks of acc(pooled) - acc(own))."""
    ev, jobs, tc = Evals(pool), [], TCache(pool)
    for ms in groups_of_models:
        pooled = np.mean([al[m]["alpha"] for m in ms], 0)
        hh = []
        for m in ms:
            st = ho_stack(pool.models[m]["ws"], pool.models[m]["ky"])
            hh.append((ev.add(m, carry_cols(pool, m, pooled, tc=tc), *st), ev.add(m, carry_cols(pool, m, al[m]["alpha"], tc=tc), *st)))
        jobs.append(hh)
    ev.run()
    out = []
    for hh in jobs:
        pa = [float(ev.get(h1).mean()) for h1, _ in hh]
        oa = [float(ev.get(h0).mean()) for _, h0 in hh]
        out.append(dict(gain=float(np.mean(np.array(pa) - np.array(oa))), pooled=pa, own=oa))
    return out


def t34_pairs(pool, pairs, fc, t3_ok=lambda a, b: False):
    """AT1 T4 (and T3 where t3_ok(a, b)) per ordered pair (a, b) (TC.t3_t4, batched per receiver group): A's free column gives h2_A and
    log p_A on the new task's 289 inputs; T4 fits b's column from zero (Adam lr 1e-2, FT_STEPS steps) to KL(p_A || p_B) (batchmean); T3
    fits it to the stitched target [h2_A, 1] S (S = TC.stitch_fit of h2_A -> h2_B on 12 tasks x 289 pairs, leave-3-tasks-out lambda;
    its R2 reported). Accuracy on b's new-task held-out pairs (b's labels). Returns list of dict(t4 [4], t4_mean, t3, t3_mean, stitch)."""
    xa, ya = TC.PAIRS[:, 0], TC.PAIRS[:, 1]
    lp, h2 = {}, {}
    for g, ms in pool.by_group(sorted({a for a, _ in pairs})).items():
        nets = pool.groups[g]
        for k in range(0, len(ms), 64):
            chunk = ms[k:k + 64]
            net_of = np.repeat([pool.models[m]["i"] for m in chunk], NNEW)
            cols = np.concatenate([fc[m]["v"] for m in chunk]).astype(np.float32)
            with torch.no_grad():
                lpa = torch.log_softmax(chain_logits(nets, net_of, cols, xa, ya), -1).cpu().numpy()
                _, ha = TC.hidden_states(nets, net_of, cols, (xa, ya))
                ha = ha.double().cpu().numpy()
            for n, m in enumerate(chunk):
                lp[m] = lpa[n * NNEW:(n + 1) * NNEW]
                h2[m] = ha[n * NNEW:(n + 1) * NNEW]
    allh = {}

    def all_h2(m):
        if m not in allh:
            allh[m] = TC.all_h2(pool.nets(m), pool.models[m]["i"])
        return allh[m]

    out = [dict(t4=None, t3=None, stitch=None) for _ in pairs]
    for mode in ("t4", "t3"):
        chains, targets = [], []
        for pi, (a, b) in enumerate(pairs):
            if mode == "t3":
                if not t3_ok(a, b):
                    continue
                S, best = TC.stitch_fit(all_h2(a), all_h2(b))
                out[pi]["stitch"] = dict(lam=best[0], R2=best[1])
                tg = (np.concatenate([h2[a], np.ones((NNEW, P * P, 1))], 2) @ S).astype(np.float32)
            else:
                tg = lp[a]
            for j in range(NNEW):
                chains.append((pi, b, j))
                targets.append(tg[j])
        by = {}
        for c, (pi, b, j) in enumerate(chains):
            by.setdefault(pool.models[b]["g"], []).append(c)
        acc = np.zeros(len(chains))
        for g, cs in by.items():
            nets = pool.groups[g]
            for sl in search_chunks(nets, P * P, len(cs)):
                cc = cs[sl]
                net_of = np.array([pool.models[chains[c][1]]["i"] for c in cc])
                sets = [nsets(pool.models[chains[c][1]]["ws"], chains[c][2], pool.models[chains[c][1]]["ky"]) for c in cc]
                shots = (np.tile(xa, (len(cc), 1)), np.tile(ya, (len(cc), 1)), np.zeros((len(cc), P * P), np.int64))
                r = TC.column_search(nets, net_of, shots, TC.stack_sets(sets, "ho"), steps=TC.FT_STEPS, eval_at=(TC.FT_STEPS,),
                                     loss="kl" if mode == "t4" else "mse", target=np.stack([targets[c] for c in cc]))
                acc[cc] = r["acc"]["ho"][:, -1]
        for c, (pi, b, j) in enumerate(chains):
            if out[pi][mode] is None:
                out[pi][mode] = np.zeros(NNEW)
            out[pi][mode][j] = acc[c]
    for d in out:
        d["t4_mean"] = float(np.mean(d["t4"])) if d["t4"] is not None else None
        d["t3_mean"] = float(np.mean(d["t3"])) if d["t3"] is not None else None
    return out


# ====================================================================================================================== helpers for stages
def cond_name(arm, reg=None, gamma=0.0):
    """Condition names: 'nat' and '<reg>_<arm><gamma>' (e.g. 'task_late+0.01', 'sham_early-0.003'); names starting with 'task' are the
    task-regulariser arms (k = 1, as r2_sim.reg2_noise reads them)."""
    return "nat" if arm == "nat" else f"{reg}_{arm}{gamma:+g}"


def tci(v):
    """Two-sided 95 % t interval over units (df = n - 1; 2.6): dict(n, mean, lo, hi) (descriptive summaries only)."""
    from scipy import stats
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    n = len(v)
    if n < 2:
        return dict(n=n, mean=float(v.mean()) if n else None, lo=None, hi=None)
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n)
    return dict(n=n, mean=float(v.mean()), lo=float(v.mean() - h), hi=float(v.mean() + h))


def model_row(inf, keys=("name", "ws", "init", "W", "arm", "kind", "depth", "act", "lr", "wd", "ky", "split", "coding", "cond", "reg",
                         "gamma", "k", "target_x", "p", "level1", "eval_point", "first95", "checkpoint", "updates", "updates_total", "branch_updates",
                         "final_heldout", "final_train", "checksum", "coord_check", "variant", "shifts")):
    return {k: inf[k] for k in keys if k in inf}


def delta_fields(d):
    if not d:
        return {}
    return dict(delta_h2=d["h2"]["delta"], delta_h1=d["h1"]["delta"], delta_ctrl_h2=d["h2"]["delta_ctrl"], delta_ctrl_h1=d["h1"]["delta_ctrl"],
                ring_h2=d["h2"]["ring"], ring_h1=d["h1"]["ring"], delta=d)


def b2_fields(pool, mids):
    """pc_meta's B2 (8 and 4 shots: column search A minus fresh-net F, TC.b2) of the given models -> {m: dict(B2_8, B2_4, A8_mean,
    F8_mean, F4_mean)}; used for G46's reproduction check and as a descriptive covariate."""
    out = {}
    for g, ms in pool.by_group(mids).items():
        sub = pool.sub(g, ms)
        r = TC.b2(sub, [pool.models[m]["ws"] for m in ms])
        for n, m in enumerate(ms):
            out[m] = dict(B2_8=float(r[8]["B2"][n]), B2_4=float(r[4]["B2"][n]), A8_mean=float(r[8]["A"][n].mean()),
                          F8_mean=float(r[8]["F"][n].mean()), F4_mean=float(r[4]["F"][n].mean()))
    return out


# ====================================================================================================================== AT0 (r0)
def select_levels(cands, targets=R0_TARGETS, tol=R0_TOL):
    """AT0's selection rule (frozen). cands: list in grid order of dict(gamma, mlr = mean over the inits of log2(delta_h2 / delta_nat)
    (None if undefined: a branch missing or delta_h2 <= 0), pass_both = level 1 (held-out >= 0.95 at the branch's end) in both inits).
    For each target x: the grid gamma whose mlr is closest (ties: grid order), accepted if within tol and passing in both inits;
    otherwise the closest gamma that passes level 1 in both inits (flagged); if none passes, the level is dropped (flagged). Each grid gamma
    is used at most once (critique v2-12i): if several targets select the same gamma, the target whose mlr lies farther from it (ties: the
    later target) takes its next-closest passing gamma (flagged), repeated until all selected values differ (dropped if it has none left).
    Returns a list in target order of dict(target_x, gamma (None = dropped), mlr, dist, status 'accepted' | 'flagged' | 'dropped',
    flags)."""
    ok = [(n, c) for n, c in enumerate(cands) if c.get("mlr") is not None and math.isfinite(c["mlr"])]
    rank, choice, status, flags = {}, {}, {}, {t: [] for t in targets}
    for t in targets:
        order = sorted(ok, key=lambda nc: (abs(nc[1]["mlr"] - t), nc[0]))
        rank[t] = [nc[0] for nc in order if nc[1]["pass_both"]]
        if not order:
            choice[t], status[t] = None, "dropped"
            flags[t].append("no grid gamma with a defined delta ratio")
            continue
        n0, c0 = order[0]
        if abs(c0["mlr"] - t) <= tol and c0["pass_both"]:
            choice[t], status[t] = n0, "accepted"
        elif rank[t]:
            choice[t], status[t] = rank[t][0], "flagged"
            why = "fails level 1 in an init" if not c0["pass_both"] else f"lies {abs(c0['mlr'] - t):.2f} > {tol} from the target"
            flags[t].append(f"closest grid gamma {c0['gamma']:+g} {why}; closest gamma passing level 1 in both inits taken")
        else:
            choice[t], status[t] = None, "dropped"
            flags[t].append("no grid gamma passes level 1 in both inits")
    while True:
        used = {}
        for t in targets:
            if choice[t] is not None:
                used.setdefault(choice[t], []).append(t)
        dup = [(n, ts) for n, ts in used.items() if len(ts) > 1]
        if not dup:
            break
        n, ts = dup[0]
        mlr = cands[n]["mlr"]
        mover = max(ts, key=lambda t: (abs(mlr - t), targets.index(t)))
        rest = rank[mover][rank[mover].index(n) + 1:]
        others = [t for t in ts if t != mover]
        if rest:
            choice[mover], status[mover] = rest[0], "flagged"
            flags[mover].append(f"gamma {cands[n]['gamma']:+g} also selected by target(s) {others}, which lie closer; next-closest passing "
                                f"gamma {cands[rest[0]]['gamma']:+g} taken")
        else:
            choice[mover], status[mover] = None, "dropped"
            flags[mover].append(f"gamma {cands[n]['gamma']:+g} also selected by target(s) {others}; no further passing gamma")
    out = []
    for t in targets:
        n = choice[t]
        out.append(dict(target_x=t, gamma=None if n is None else cands[n]["gamma"], mlr=None if n is None else cands[n]["mlr"],
                        dist=None if n is None else abs(cands[n]["mlr"] - t), status=status[t], flags=flags[t]))
    return out


def train_r0(bundle):
    """AT0 nets of world 0, inits 0 and 1: natural runs (first95 state kept), the late task branches at every grid gamma and the late
    sham branches (500 updates from first95), the early task and sham nets (checkpoint rule)."""
    ws = R0_WORLD
    nat_specs = [dict(W=W_MAIN, ws=ws, init=i) for i in R0_INITS]
    say(f"r0: training {len(nat_specs)} natural nets (W {W_MAIN})")
    nat = TC.train(nat_specs, save_first95=True)
    infos = [info_of(s, r, cond="nat", arm="nat", k=0, p=0.0) for s, r in zip(nat_specs, nat.rows)]
    parts = [nat.nets]
    br, idx = [], []
    for i, s in enumerate(nat_specs):
        if nat.rows[i]["first95"] is None:
            continue
        br += [dict(s, gamma=g, reg="task", when="late") for g in R0_TASK_GRID] + [dict(s, gamma=g, reg="sham", when="late") for g in SHAM_LATE]
        idx += [i] * (len(R0_TASK_GRID) + len(SHAM_LATE))
    if br:
        say(f"r0: {len(br)} late branches x {TC.AFTER} updates")
        late = TC.train_late(nat, idx, br)
        parts.append(late.nets)
        for b, (s, r) in enumerate(zip(br, late.rows)):
            f95 = nat.rows[idx[b]]["first95"]
            infos.append(info_of(s, r, cond=cond_name("late", s["reg"], s["gamma"]), arm="late", k=int(s["reg"] == "task"),
                                 p=TC.param_distance(late.nets, b, nat.nets, idx[b]), **late_fields(f95)))
    es = [dict(W=W_MAIN, ws=ws, init=i, gamma=g, reg=r, when="early") for r in ("task", "sham") for g in EARLY for i in R0_INITS]
    say(f"r0: training {len(es)} early-regulariser nets")
    early = TC.train(es)
    parts.append(early.nets)
    infos += [info_of(s, r, cond=cond_name("early", s["reg"], s["gamma"]), arm="early", k=int(s["reg"] == "task"), p=abs(s["gamma"]) / 0.01)
              for s, r in zip(es, early.rows)]
    nets = TC.Nets.cat(parts)
    save_bundle(bundle, nets, infos)
    return nets, infos


def stage_r0(out):
    path = os.path.join(out, "rgi_toy", "r0.json")
    if os.path.exists(path):
        say("r0: output exists, skipped:", path)
        return
    t0 = time.time()
    bundle = os.path.join(out, "rgi_toy", "models", "r0.pt")
    nets, infos = (load_bundle(bundle)[:2] if os.path.exists(bundle) else train_r0(bundle))
    pool = Pool()
    mids = pool.add(nets, infos)
    dl = deltas(pool, mids)
    rows = []
    for m in mids:
        r = model_row(pool.models[m])
        r.update(delta_fields(dl[m]))
        rows.append(r)
    nat = [r for r in rows if r["arm"] == "nat"]
    nat_ok = [r for r in nat if r["level1"]]
    use = nat_ok or nat
    delta_nat = float(np.mean([r["delta_h2"] for r in use]))
    flags = [] if nat_ok else ["no natural net passes level 1: delta_nat from all natural nets"]
    for r in rows:
        r["log2_ratio"] = math.log2(r["delta_h2"] / delta_nat) if (r["delta_h2"] > 0 and delta_nat > 0) else None

    def per_gamma(arm, reg, gammas):
        out_ = []
        for g in gammas:
            rr = [r for r in rows if r["arm"] == arm and r["reg"] == reg and abs(r["gamma"] - g) < 1e-12]
            lr = [r["log2_ratio"] for r in rr]
            out_.append(dict(gamma=g, inits=[r["init"] for r in rr], log2_ratio=lr, heldout=[r["final_heldout"] for r in rr],
                             level1=[r["level1"] for r in rr], delta_h2=[r["delta_h2"] for r in rr],
                             mlr=float(np.mean(lr)) if (len(rr) == len(R0_INITS) and all(x is not None for x in lr)) else None,
                             pass_both=len(rr) == len(R0_INITS) and all(r["level1"] for r in rr),
                             overshoot=any(r["delta_h2"] is not None and r["delta_h2"] <= 0 for r in rr)))
        return out_
    grid = per_gamma("late", "task", R0_TASK_GRID)
    sel = select_levels(grid)
    sham = per_gamma("late", "sham", SHAM_LATE)
    et, esh = per_gamma("early", "task", EARLY), per_gamma("early", "sham", EARLY)
    levels = dict(task_late=[dict(target_x=s["target_x"], gamma=s["gamma"], status=s["status"], flags=s["flags"]) for s in sel if s["gamma"] is not None],
                  sham_late=[c["gamma"] for c in sham if c["pass_both"]], task_early=[c["gamma"] for c in et if c["pass_both"]],
                  sham_early=[c["gamma"] for c in esh if c["pass_both"]])
    dropped = ([dict(arm="task_late", target_x=s["target_x"], flags=s["flags"]) for s in sel if s["gamma"] is None]
               + [dict(arm=a, gamma=c["gamma"], reason="fails level 1 in an init") for a, cs in (("sham_late", sham), ("task_early", et),
                                                                                                    ("sham_early", esh)) for c in cs if not c["pass_both"]])
    res = dict(stage="r0", measurement="AT0", world=R0_WORLD, inits=list(R0_INITS), W=W_MAIN, quick=QUICK, delta_nat=delta_nat,
               delta_nat_from=[r["init"] for r in use], flags=flags, task_late_grid=grid, selection=sel, sham_late=sham, task_early=et,
               sham_early=esh, levels=levels, dropped=dropped, models=rows, seconds=time.time() - t0,
               rule="AT0: closest grid gamma by the mean log2 ratio over the 2 inits, accepted within 0.4 and level 1 in both inits; else the "
                    "closest passing gamma (flagged); else dropped; each grid gamma used once (farther target moves on)")
    write_json(path, res)
    say(f"r0: delta_nat {delta_nat:.4f}; task-late levels " + ", ".join(f"x {s['target_x']:+d}: {s['gamma']} ({s['status']})" for s in sel)
        + f"; sham late {levels['sham_late']}; early task {levels['task_early']}, sham {levels['sham_early']} ({time.time() - t0:.0f}s)")


# ====================================================================================================================== AT2 / AT3 (r1)
def r1_conditions(levels):
    """The 15 conditions of AT2 (natural; task late at the R0 levels; sham late; early task; early sham), as far as R0 kept them."""
    conds = [dict(cond="nat", arm="nat", reg=None, gamma=0.0, k=0)]
    conds += [dict(cond=cond_name("late", "task", d["gamma"]), arm="late", reg="task", gamma=d["gamma"], k=1, target_x=d["target_x"])
              for d in levels["task_late"]]
    conds += [dict(cond=cond_name("late", "sham", g), arm="late", reg="sham", gamma=g, k=0) for g in levels["sham_late"]]
    conds += [dict(cond=cond_name("early", "task", g), arm="early", reg="task", gamma=g, k=1) for g in levels["task_early"]]
    conds += [dict(cond=cond_name("early", "sham", g), arm="early", reg="sham", gamma=g, k=0) for g in levels["sham_early"]]
    return conds


def r1_bundle(out, ws):
    return os.path.join(out, "rgi_toy", "models", f"r1_w{ws}.pt")


def r1_train(out, worlds, conds):
    """AT2 nets of the given worlds (batched over worlds): natural runs (checkpoint first95 + 500, continued to first95 + 1500, first95
    state kept), late branches at the late conditions, early nets; one bundle per world."""
    nat_specs = [dict(W=W_MAIN, ws=ws, init=i) for ws in worlds for i in INITS3]
    say(f"r1: training {len(nat_specs)} natural nets (to first95 + {TC.AFTER + EXTRA2})")
    nat = TC.train(nat_specs, save_first95=True, extra_after=EXTRA2)
    late_c = [c for c in conds if c["arm"] == "late"]
    early_c = [c for c in conds if c["arm"] == "early"]
    br, idx = [], []
    for i, s in enumerate(nat_specs):
        if nat.rows[i]["first95"] is None:
            continue
        for c in late_c:
            br.append(dict(s, gamma=c["gamma"], reg=c["reg"], when="late"))
            idx.append(i)
    late = None
    if br:
        say(f"r1: {len(br)} late branches x {TC.AFTER} updates")
        late = TC.train_late(nat, idx, br)
    es = [dict(W=W_MAIN, ws=ws, init=i, gamma=c["gamma"], reg=c["reg"], when="early") for ws in worlds for c in early_c for i in INITS3]
    early = None
    if es:
        say(f"r1: training {len(es)} early-regulariser nets")
        early = TC.train(es)
    cmap = {c["cond"]: c for c in conds}
    for ws in worlds:
        parts, infos = [], []
        ni = [n for n, s in enumerate(nat_specs) if s["ws"] == ws]
        parts.append(nat.nets.subset(ni))
        infos += [info_of(nat_specs[n], nat.rows[n], cond="nat", arm="nat", k=0, p=0.0) for n in ni]
        n2 = [n for n in ni if nat.has2[n]]
        if n2:
            sub2 = nat.nets2.subset(n2)
            ho2 = TC.task_accuracy(sub2, [ws] * len(n2)).mean(1)
            parts.append(sub2)
            for q, n in enumerate(n2):
                r = dict(nat.rows[n], checkpoint=nat.rows[n]["second"], updates=nat.rows[n]["second"], final_heldout=float(ho2[q]),
                         final_train=None, checksum=TC.weight_checksum(sub2, q), eval_point="second checkpoint")
                infos.append(info_of(nat_specs[n], r, cond="nat+1500", arm="nat2", k=0, p=0.0))
        li = [b for b, i in enumerate(idx) if nat_specs[i]["ws"] == ws]
        if li:
            parts.append(late.nets.subset(li))
            for b in li:
                c = cmap[cond_name("late", br[b]["reg"], br[b]["gamma"])]
                f95 = nat.rows[idx[b]]["first95"]
                infos.append(info_of(br[b], late.rows[b], cond=c["cond"], arm="late", k=c["k"], target_x=c.get("target_x"),
                                     p=TC.param_distance(late.nets, b, nat.nets, idx[b]), **late_fields(f95)))
        ei = [n for n, s in enumerate(es) if s["ws"] == ws]
        if ei:
            parts.append(early.nets.subset(ei))
            infos += [info_of(es[n], early.rows[n], cond=cond_name("early", es[n]["reg"], es[n]["gamma"]), arm="early",
                              k=int(es[n]["reg"] == "task"), p=abs(es[n]["gamma"]) / 0.01) for n in ei]
        save_bundle(r1_bundle(out, ws), TC.Nets.cat(parts), infos)


def at_lines(pool, mids, recursion_on=True, pairs_by=("ws", "cond")):
    """AT1 lines of a set of models of one architecture family: per model (free columns, alpha, X, Lambda, Phi, recipe depth, delta,
    mediators) and per ordered cross-init pair within (world, condition): S, free T1 gain, LOO-anchor, and (recursion_on) round trip,
    warm start and pooling. Returns (per-model dict, pair list, pooling list)."""
    t0 = time.time()
    fc = free_columns(pool, mids)
    al = alphas(pool, mids)
    say(f"  free columns and alphas of {len(mids)} models ({time.time() - t0:.0f}s)")
    lam = lambdas(pool, mids)
    say(f"  Lambda ({time.time() - t0:.0f}s)")
    rec = recipe_depths(pool, mids) if recursion_on else {}
    dl = deltas(pool, mids)
    med = mediators(pool, mids)
    tcols = task_cols_np(pool)
    ph = phis(pool, mids, fc, tcols)
    say(f"  recipe depth, delta, mediators, Phi ({time.time() - t0:.0f}s)")
    cells = {}
    for m in mids:
        cells.setdefault(tuple(pool.models[m][k] for k in pairs_by), []).append(m)
    cells = {k: sorted(v, key=lambda m: pool.models[m]["init"]) for k, v in cells.items()}
    pairs = [(a, b) for ms in cells.values() for a in ms for b in ms if a != b]
    S = s_pairs(pool, [(a, b, al[a]["alpha"], pool.models[b]["anchors"]) for a, b in pairs])
    F = f_pairs(pool, [(a, b, list(range(len(pool.models[a]["anchors"]))), list(range(len(pool.models[b]["anchors"])))) for a, b in pairs],
                fc, tcols)
    say(f"  S and F over {len(pairs)} ordered pairs ({time.time() - t0:.0f}s)")
    R, PL = [{} for _ in pairs], []
    if recursion_on:
        rp = [n for n, (a, b) in enumerate(pairs) if pool.models[a]["arm"] != "nat2"]
        for n, d in zip(rp, recursion(pool, [pairs[n] for n in rp], al)):
            R[n] = d
        pc = [(k, ms) for k, ms in cells.items() if len(ms) > 1 and pool.models[ms[0]]["arm"] != "nat2"]
        PL = [dict(cell=dict(zip(pairs_by, k)), models=[pool.models[m]["name"] for m in ms], **d)
              for (k, ms), d in zip(pc, pooling_lines(pool, [ms for _, ms in pc], al))]
        say(f"  recursion: round trip, warm start, pooling ({time.time() - t0:.0f}s)")
    per = {}
    for m in mids:
        inf = pool.models[m]
        r = model_row(inf)
        r.update(delta_fields(dl.get(m)))
        r.update(Lambda=lam[m]["Lambda"], Lambda_per_shift=lam[m]["per_shift"], Lambda_null_wrong=lam[m]["wrong"],
                 Lambda_null_perm=lam[m]["perm"], Phi=ph[m]["Phi"], acc_proj=ph[m]["acc_proj"], acc_free=fc[m]["acc_end"],
                 A_free=float(fc[m]["A"].mean()), acc_free_curve=fc[m]["acc"], acc_alpha=al[m]["acc"], alpha=al[m]["alpha"],
                 X=float(np.mean(al[m]["acc"]) / np.mean(fc[m]["acc_end"])) if np.mean(fc[m]["acc_end"]) > 0 else None, **med[m])
        if m in rec:
            r.update(recipe_acc=rec[m]["acc"], rho=rec[m]["rho"])
        per[m] = r
    plist = []
    for n, (a, b) in enumerate(pairs):
        d = dict(A=pool.models[a]["name"], B=pool.models[b]["name"], A_init=pool.models[a]["init"], B_init=pool.models[b]["init"],
                 cond=pool.models[a].get("cond"), ws=pool.models[a]["ws"], A_level1=pool.models[a].get("level1"),
                 B_level1=pool.models[b].get("level1"))
        d.update(S[n])
        d.update({k: v for k, v in F[n].items()})
        d.update(R[n])
        plist.append(d)
    for m in mids:      # per receiver model: mean over its pairs (2.7 C20)
        mine = [n for n, (a, b) in enumerate(pairs) if b == m]
        per[m].update(S_gain=float(np.mean([S[n]["S_gain"] for n in mine])) if mine else None,
                      T1_gain=float(np.mean([F[n]["t1_gain"] for n in mine])) if mine else None,
                      T1=float(np.mean([F[n]["t1"] for n in mine])) if mine else None,
                      LOO=float(np.mean([F[n]["loo"] for n in mine])) if mine else None,
                      LOO_shuf=float(np.mean([F[n]["loo_shuf"] for n in mine])) if mine else None,
                      n_senders=len(mine))
        if recursion_on and pool.models[m]["arm"] != "nat2":
            snd = [n for n, (a, b) in enumerate(pairs) if a == m and R[n]]
            per[m].update(RG=float(np.mean([R[n]["RG"] for n in snd])) if snd else None)
    return per, plist, PL


def stage_r1(out):
    r0p = os.path.join(out, "rgi_toy", "r0.json")
    if not os.path.exists(r0p):
        raise SystemExit("r1 needs rgi_toy/r0.json (stage r0) first")
    r0 = read_json(r0p)
    conds = r1_conditions(r0["levels"])
    todo = [ws for ws in R1_WORLDS if not os.path.exists(os.path.join(out, "rgi_toy", "r1", f"w{ws}.json"))]
    if not todo:
        say("r1: every world exists, skipped")
        r1_at3(out)
        return
    t0 = time.time()
    need = [ws for ws in todo if not os.path.exists(r1_bundle(out, ws))]
    if need:
        r1_train(out, need, conds)
    say(f"r1: training done ({time.time() - t0:.0f}s); lines for worlds {todo}")
    loaded = [load_bundle(r1_bundle(out, ws)) for ws in todo]
    nets = TC.Nets.cat([l_[0] for l_ in loaded])
    infos = [inf for l_ in loaded for inf in l_[1]]
    pool = Pool()
    mids = pool.add(nets, infos)
    per, plist, PL = at_lines(pool, mids)
    natm = [m for m in mids if pool.models[m]["arm"] == "nat"]
    b2 = b2_fields(pool, natm)
    for ws in todo:
        wm = [m for m in mids if pool.models[m]["ws"] == ws]
        nat = [m for m in wm if pool.models[m]["arm"] == "nat"]
        ok = [m for m in nat if pool.models[m]["level1"]]
        use = ok or nat
        dnat = float(np.mean([per[m]["delta_h2"] for m in use]))
        rows = []
        for m in wm:
            r = dict(per[m])
            r["x"] = math.log2(r["delta_h2"] / dnat) if (r.get("delta_h2") is not None and r["delta_h2"] > 0 and dnat > 0) else None
            if pool.models[m]["arm"] in ("nat", "nat2"):
                r["p"], r["k"] = 0.0, 0
            if m in b2:
                r.update(b2[m])
                r["r1_job_name"] = r1_job_name(pool.models[m]) if pool.models[m]["init"] == 0 else None
            rows.append(r)
        names = {r["name"] for r in rows}
        res = dict(stage="r1", measurement="AT2 (with AT1 lines and AT3 mediators)", world=ws, quick=QUICK, W=W_MAIN, conditions=conds,
                   r0_levels=r0["levels"], delta_nat=dnat, delta_nat_from=[pool.models[m]["init"] for m in use],
                   delta_nat_flag=None if ok else "no natural net passes level 1: delta_nat from all natural nets",
                   models=[r for r in rows if r["arm"] != "nat2"], natural_second=[r for r in rows if r["arm"] == "nat2"],
                   pairs=[p for p in plist if p["A"] in names], pooling=[p for p in PL if p["cell"]["ws"] == ws],
                   within_run=r1_within(rows), mediators_by_condition=r1_by_condition(rows), seconds=time.time() - t0,
                   notes="per model: x = log2(delta_h2 / delta_nat(world)); p = relative parameter distance to the same-init natural model "
                         "(late), |gamma| / 0.01 (early), 0 (natural); S_gain / T1_gain / T1 / LOO = mean over the model's 2 pairs as "
                         "receiver; RG = mean over its pairs as sender")
        write_json(os.path.join(out, "rgi_toy", "r1", f"w{ws}.json"), res)
        say(f"r1: world {ws} written ({len(rows)} models)")
    r1_at3(out)


def r1_within(rows):
    """AT3 within-run change from first95 + 500 to first95 + 1500 of the natural models (per init)."""
    out = []
    for r2 in [r for r in rows if r["arm"] == "nat2"]:
        r1_ = [r for r in rows if r["arm"] == "nat" and r["init"] == r2["init"]]
        if not r1_:
            continue
        r1_ = r1_[0]
        d = dict(init=r2["init"])
        for k in ("delta_h2", "delta_h1", "delta_ctrl_h2", "Lambda", "S_gain", "T1_gain", "Phi", "X", "colnorm", "sat_h1", "final_heldout"):
            a, b = r1_.get(k), r2.get(k)
            d[k] = dict(at500=a, at1500=b, change=(b - a) if (a is not None and b is not None) else None)
        out.append(d)
    return out


def r1_by_condition(rows):
    """AT3 descriptive: per condition the mean over its models of the mediators and lines."""
    out = {}
    for r in rows:
        out.setdefault(r["cond"], []).append(r)
    keys = ("x", "p", "delta_h2", "delta_h1", "delta_ctrl_h2", "final_heldout", "colnorm", "sat_h1", "X", "Lambda", "S_gain", "T1_gain", "Phi")
    return {c: dict(n=len(rr), n_level1=sum(bool(r["level1"]) for r in rr),
                    **{k: (float(np.mean([r[k] for r in rr if r.get(k) is not None])) if any(r.get(k) is not None for r in rr) else None)
                       for k in keys}) for c, rr in out.items()}


def r1_at3(out):
    """AT3 descriptive across worlds (written once every world exists): correlation across inits of delta_h2 with each line at both
    checkpoints (pooled over worlds, raw and with world means removed), and the within-run changes with t intervals over worlds."""
    files = [os.path.join(out, "rgi_toy", "r1", f"w{ws}.json") for ws in R1_WORLDS]
    if not all(os.path.exists(f) for f in files):
        return
    W_ = [read_json(f) for f in files]
    res = dict(stage="r1", measurement="AT3 natural co-variation (descriptive)", worlds=list(R1_WORLDS), checkpoints={})
    for tag, key in (("first95+500", "models"), ("first95+1500", "natural_second")):
        rows = [dict(r, ws=w["world"]) for w in W_ for r in w[key] if r["cond"] in ("nat", "nat+1500")]
        cor = {}
        for line in ("Lambda", "S_gain", "T1_gain", "Phi", "X"):
            pr = [(r["ws"], r["delta_h2"], r[line]) for r in rows if r.get(line) is not None and r.get("delta_h2") is not None]
            if len(pr) < 3:
                cor[line] = None
                continue
            ws_, d_, y_ = (np.array(v, float) for v in zip(*pr))
            dc, yc = d_.copy(), y_.copy()
            for w in np.unique(ws_):
                dc[ws_ == w] -= d_[ws_ == w].mean()
                yc[ws_ == w] -= y_[ws_ == w].mean()
            c_raw = float(np.corrcoef(d_, y_)[0, 1]) if d_.std() > 0 and y_.std() > 0 else None
            c_w = float(np.corrcoef(dc, yc)[0, 1]) if dc.std() > 0 and yc.std() > 0 else None
            cor[line] = dict(n=len(pr), corr_raw=c_raw, corr_within_world=c_w)
        res["checkpoints"][tag] = dict(correlation_with_delta_h2=cor)
    ch = {}
    for k in ("delta_h2", "Lambda", "S_gain", "T1_gain", "Phi", "X"):
        per_w = [np.mean([d[k]["change"] for d in w["within_run"] if d[k]["change"] is not None]) for w in W_
                 if any(d[k]["change"] is not None for d in w["within_run"])]
        ch[k] = tci(per_w)
    res["within_run_change_over_worlds"] = ch
    write_json(os.path.join(out, "rgi_toy", "r1", "at3.json"), res)


# ====================================================================================================================== AT5 (cross)
def at5_name(spec):
    return "at5_" + TC.net_name(spec)


def at5_specs(worlds):
    """AT5 BP nets: BP-muP W in {64, 128, 256, 1024} with inits 0 and 1, BP-SP at the same widths with init 0, per world."""
    out = []
    for W in X_WIDTHS:
        out.append([dict(W=W, ws=ws, init=i, arm="bp_mup") for ws in worlds for i in (0, 1)])
        out.append([dict(W=W, ws=ws, init=0, arm="bp_sp") for ws in worlds])
    return out


def at5_groups(out, worlds):
    """Train (or reload) the AT5 nets of the given worlds, one group per (width, parameterisation); every net is saved as
    rgi_toy/models/at5_<name>.pt (TC.save_nets: state with pc_common.MLP keys, arch, tables, meta incl. spec and training row, weight
    checksum), the file set r2_pc copies (7.3) and stage fam reads."""
    folder = os.path.join(out, "rgi_toy", "models")
    groups = []
    for specs in at5_specs(worlds):
        paths = [os.path.join(folder, at5_name(s) + ".pt") for s in specs]
        if all(os.path.exists(p) for p in paths):
            nets = TC.load_nets(paths)
            rows = [nets.meta[n].get("row") for n in range(nets.B)]
        else:
            say(f"cross: training {len(specs)} nets {specs[0]['arm']} W {specs[0]['W']}")
            res = TC.train(specs)
            nets, rows = res.nets, res.rows
            TC.save_nets(nets, range(nets.B), folder, names=[at5_name(s) for s in specs])
        groups.append((nets, [info_of(s, r, cond="nat", arm="nat", k=0, p=0.0) for s, r in zip(specs, rows)]))
    return groups


def stage_cross(out):
    todo = [ws for ws in X_WORLDS if not os.path.exists(os.path.join(out, "rgi_toy", "cross", f"w{ws}.json"))]
    if not todo:
        say("cross: every world exists, skipped")
        return
    t0 = time.time()
    pool = Pool()
    mids = []
    for nets, infos in at5_groups(out, todo):
        mids += pool.add(nets, infos)
    say(f"cross: {len(mids)} nets ready ({time.time() - t0:.0f}s); lines")
    fc = free_columns(pool, mids)
    al = alphas(pool, mids)
    lam = lambdas(pool, mids)
    dl = deltas(pool, mids)
    tcols = task_cols_np(pool)
    ph = phis(pool, mids, fc, tcols)
    b2 = b2_fields(pool, mids)
    say(f"cross: per-model lines ({time.time() - t0:.0f}s)")
    pairs = [(a, b) for a in mids for b in mids if a != b and pool.models[a]["ws"] == pool.models[b]["ws"]]
    S = s_pairs(pool, [(a, b, al[a]["alpha"], pool.models[b]["anchors"]) for a, b in pairs])
    F = f_pairs(pool, [(a, b, list(range(NPRE)), list(range(NPRE))) for a, b in pairs], fc, tcols)
    say(f"cross: S and F over {len(pairs)} ordered pairs ({time.time() - t0:.0f}s)")
    T = t34_pairs(pool, pairs, fc, t3_ok=lambda a, b: pool.models[a]["W"] <= T3_MAXW and pool.models[b]["W"] <= T3_MAXW)
    say(f"cross: T4 / T3 ({time.time() - t0:.0f}s)")
    per = {}
    for m in mids:
        inf = pool.models[m]
        r = model_row(inf)
        r.update(delta_fields(dl.get(m)))
        r.update(b2[m])
        r.update(Lambda=lam[m]["Lambda"], Lambda_per_shift=lam[m]["per_shift"], Lambda_null_wrong=lam[m]["wrong"],
                 Lambda_null_perm=lam[m]["perm"], Phi=ph[m]["Phi"], acc_proj=ph[m]["acc_proj"], acc_free=fc[m]["acc_end"],
                 A_free=float(fc[m]["A"].mean()), acc_alpha=al[m]["acc"], alpha=al[m]["alpha"],
                 X=float(np.mean(al[m]["acc"]) / np.mean(fc[m]["acc_end"])) if np.mean(fc[m]["acc_end"]) > 0 else None,
                 r1_job_name=r1_job_name(inf) if inf["init"] == 0 else None)
        per[m] = r
    plist = []
    for n, (a, b) in enumerate(pairs):
        A, B = pool.models[a], pool.models[b]
        d = dict(A=A["name"], B=B["name"], ws=A["ws"], A_W=A["W"], B_W=B["W"], A_arm=A["arm"], B_arm=B["arm"], A_init=A["init"],
                 B_init=B["init"], A_level1=A["level1"], B_level1=B["level1"])
        d.update(S[n])
        d.update(F[n])
        d.update(t4=T[n]["t4"], t4_mean=T[n]["t4_mean"], t3=T[n]["t3"], t3_mean=T[n]["t3_mean"], stitch=T[n]["stitch"])
        plist.append(d)
    for m in mids:
        rec_ = [n for n, (a, b) in enumerate(pairs) if b == m]
        snd = [n for n, (a, b) in enumerate(pairs) if a == m]
        for tag, sel in (("as_receiver", rec_), ("as_sender", snd)):
            per[m][tag] = dict(n=len(sel), S_gain=float(np.mean([S[n]["S_gain"] for n in sel])), T1_gain=float(np.mean([F[n]["t1_gain"] for n in sel])),
                               LOO=float(np.mean([F[n]["loo"] for n in sel])), T4=float(np.mean([T[n]["t4_mean"] for n in sel])),
                               T3=float(np.mean([T[n]["t3_mean"] for n in sel if T[n]["t3_mean"] is not None]))
                               if any(T[n]["t3_mean"] is not None for n in sel) else None)
    for ws in todo:
        wm = [m for m in mids if pool.models[m]["ws"] == ws]
        names = {pool.models[m]["name"] for m in wm}
        res = dict(stage="cross", measurement="AT5 cross-model matrix among BP nets (descriptive)", world=ws, quick=QUICK,
                   widths=list(X_WIDTHS), t3_max_width=T3_MAXW, models=[per[m] for m in wm], pairs=[p for p in plist if p["A"] in names],
                   seconds=time.time() - t0,
                   notes="PC pairs (r2_pc) and the slope label (r2_ana) are not here; as_receiver / as_sender = means over the model's pairs "
                         "in this world; t3 only where both widths <= the T3 limit")
        write_json(os.path.join(out, "rgi_toy", "cross", f"w{ws}.json"), res)
        say(f"cross: world {ws} written")


# ====================================================================================================================== AT4 (fam)
FAM_VARIANTS = ("other split", "permuted coding", "8 shared shifts + spare", "task family x+2y+c", "3 hidden layers", "ReLU")


def fam_spec(name, ws, lr=None):
    """B variant spec of world ws (init 7 unless 'other init'): one factor changed against A = natural init 0, W 128."""
    s = dict(W=W_MAIN, ws=ws, init=FAM_INIT)
    if name == "other split":
        s["split"] = "other"
    elif name == "permuted coding":
        s["coding"] = "perm"
    elif name == "8 shared shifts + spare":
        s["shifts"] = list(M.world(ws)[0][:8]) + [TC.spare_shift(ws)]
        s["K"] = 9
    elif name == "task family x+2y+c":
        s["ky"] = 2
    elif name == "3 hidden layers":
        s["depth"] = 3
    elif name == "ReLU":
        s.update(act="relu", lr=lr)
    else:
        raise ValueError(name)
    return s


def relu_calibration(worlds, lrs=RELU_LRS):
    """AT4 / G40: ReLU B nets only after their own lr calibration: world 0, lr in {2e-3, 5e-3}, the first lr whose net passes level 1;
    otherwise 'not testable'. Returns (lr or None, the calibration TrainResult of the chosen lr or None, log)."""
    log = []
    for lr in lrs:
        s = fam_spec("ReLU", worlds[0], lr)
        res = TC.train([s])
        r = res.rows[0]
        log.append(dict(lr=lr, world=worlds[0], level1=r["level1"], first95=r["first95"], final_heldout=r["final_heldout"]))
        if r["level1"]:
            return lr, res, log
    return None, None, log


def fam_groups(out):
    """B variant nets (bundles models/fam_<k>.pt): other split, permuted coding, task family (one batch), 8 shared shifts + spare, 3 hidden
    layers, ReLU (calibrated first)."""
    folder = os.path.join(out, "rgi_toy", "models")
    groups, calib = [], None
    batches = [("fam_b1", ("other split", "permuted coding", "task family x+2y+c")), ("fam_sub8", ("8 shared shifts + spare",)),
               ("fam_depth3", ("3 hidden layers",)), ("fam_relu", ("ReLU",))]
    for key, names in batches:
        path = os.path.join(folder, key + ".pt")
        if os.path.exists(path):
            nets, infos, extra = load_bundle(path)
            if key == "fam_relu":
                calib = extra
            groups.append((nets, infos))
            continue
        if key == "fam_relu":
            jpath = os.path.join(folder, key + ".json")
            if os.path.exists(jpath):                     # an earlier run's calibration found no lr: ReLU 'not testable' (G40)
                calib = read_json(jpath)
                continue
            lr, res0, log = relu_calibration(FAM_WORLDS)
            calib = dict(tried=log, lr=lr, testable=lr is not None)
            if lr is None:
                write_json(jpath, calib)
                continue
            specs = [fam_spec("ReLU", ws, lr) for ws in FAM_WORLDS[1:]]
            parts, rows = [res0.nets], list(res0.rows)
            if specs:
                say(f"fam: training {len(specs)} ReLU nets at lr {lr:g}")
                res = TC.train(specs)
                parts.append(res.nets)
                rows += res.rows
            specs = [fam_spec("ReLU", FAM_WORLDS[0], lr)] + specs
            nets = TC.Nets.cat(parts)
            infos = [info_of(s, r, cond="nat", arm="nat", variant="ReLU") for s, r in zip(specs, rows)]
            save_bundle(path, nets, infos, extra=calib)
        else:
            specs = [fam_spec(n, ws) for n in names for ws in FAM_WORLDS]
            say(f"fam: training {len(specs)} nets ({', '.join(names)})")
            res = TC.train(specs)
            nets = res.nets
            infos = [info_of(s, r, cond="nat", arm="nat", variant=n) for (n, s), r in
                     zip([(n, fam_spec(n, ws)) for n in names for ws in FAM_WORLDS], res.rows)]
            save_bundle(path, nets, infos)
        groups.append((nets, infos))
    if calib is None and os.path.exists(os.path.join(folder, "fam_relu.json")):
        calib = read_json(os.path.join(folder, "fam_relu.json"))
    return groups, calib


def stage_fam(out):
    path = os.path.join(out, "rgi_toy", "fam.json")
    if os.path.exists(path):
        say("fam: output exists, skipped:", path)
        return
    t0 = time.time()
    folder = os.path.join(out, "rgi_toy", "models")
    aspecs = [dict(W=W_MAIN, ws=ws, init=i, arm="bp_mup") for ws in FAM_WORLDS for i in (0, 1)]
    apaths = [os.path.join(folder, at5_name(s) + ".pt") for s in aspecs]
    miss = [p for p in apaths if not os.path.exists(p)]
    if miss:
        raise SystemExit(f"fam needs AT5's W {W_MAIN} BP-muP nets (stage cross) first; missing {miss[:3]} ...")
    pool = Pool()
    anets = TC.load_nets(apaths)
    amids = pool.add(anets, [info_of(s, anets.meta[n].get("row"), cond="nat", arm="nat", variant="A" if s["init"] == 0 else "other init")
                             for n, s in enumerate(aspecs)])
    groups, calib = fam_groups(out)
    bmids = []
    for nets, infos in groups:
        bmids += pool.add(nets, infos)
    allm = amids + bmids
    A_of = {pool.models[m]["ws"]: m for m in amids if pool.models[m]["variant"] == "A"}
    Bs = [m for m in allm if pool.models[m]["variant"] != "A"]
    shared = {m: list(range(8)) if pool.models[m]["variant"] == "8 shared shifts + spare" else list(range(NPRE)) for m in Bs}
    say(f"fam: {len(allm)} nets ready ({time.time() - t0:.0f}s); lines")
    fc = free_columns(pool, allm)
    al12 = alphas(pool, list(A_of.values()))
    al8 = alphas(pool, list(A_of.values()), slots_of={m: list(range(8)) for m in A_of.values()})
    alB = alphas(pool, Bs, slots_of={m: [pool.models[m]["anchors"][k] for k in shared[m]] for m in Bs})
    lam = lambdas(pool, allm)
    dl = deltas(pool, allm)
    tcols = task_cols_np(pool)
    ph = phis(pool, allm, fc, tcols)
    pairs = [(A_of[pool.models[b]["ws"]], b) for b in Bs]
    alA = lambda a, b: (al8 if len(shared[b]) == 8 else al12)[a]["alpha"]
    S = s_pairs(pool, [(a, b, alA(a, b), [pool.models[b]["anchors"][k] for k in shared[b]]) for a, b in pairs])
    F = f_pairs(pool, [(a, b, shared[b], shared[b]) for a, b in pairs], fc, tcols)
    T = t34_pairs(pool, pairs, fc)
    say(f"fam: pair lines ({time.time() - t0:.0f}s)")

    def mrow(m):
        r = model_row(pool.models[m])
        r.update(delta_fields(dl.get(m)))
        r.update(Lambda=lam[m]["Lambda"], Lambda_per_shift=lam[m]["per_shift"], Lambda_targets=lam[m]["targets"], Phi=ph[m]["Phi"],
                 acc_free=fc[m]["acc_end"], A_free=float(fc[m]["A"].mean()))
        return r
    A_rows = []
    for ws, a in sorted(A_of.items()):
        r = mrow(a)
        r.update(acc_alpha12=al12[a]["acc"], acc_alpha8=al8[a]["acc"], X=float(np.mean(al12[a]["acc"]) / np.mean(fc[a]["acc_end"]))
                 if np.mean(fc[a]["acc_end"]) > 0 else None)
        A_rows.append(r)
    variants = {}
    for n, (a, b) in enumerate(pairs):
        v = pool.models[b]["variant"]
        r = mrow(b)
        r.update(shared_anchors=len(shared[b]), alpha_in_A=float(np.mean(al8[a]["acc"] if len(shared[b]) == 8 else al12[a]["acc"])),
                 B_own_alpha=float(np.mean(alB[b]["acc"])), S_gain=S[n]["S_gain"], accB=S[n]["accB"], accB_perm=S[n]["accB_perm"],
                 T1=F[n]["t1"], T1_shuf=F[n]["t1_shuf"], T1_gain=F[n]["t1_gain"], LOO=F[n]["loo"], LOO_shuf=F[n]["loo_shuf"], r1=F[n]["r1"],
                 T4=T[n]["t4"], T4_mean=T[n]["t4_mean"])
        variants.setdefault(v, []).append(r)
    summ = {}
    need = max(1, math.ceil(0.8 * len(FAM_WORLDS)))
    for v, rows in variants.items():
        nl1 = sum(bool(r["level1"]) for r in rows)
        summ[v] = dict(n_worlds=len(rows), n_level1=nl1, trains=nl1 >= need, rule_trains=f">= {need} of {len(FAM_WORLDS)} worlds' B nets pass level 1",
                       **{k: tci([r[k] for r in rows]) for k in ("S_gain", "Lambda", "T1_gain", "LOO", "Phi", "T4_mean", "B_own_alpha", "alpha_in_A")})
    if calib is None or not calib.get("testable", False):
        summ["ReLU"] = dict(summ.get("ReLU", {}), trains=False, testable=False, note="not testable (G40: no lr in {2e-3, 5e-3} passes level 1 in world 0)")
    res = dict(stage="fam", measurement="AT4 family arm (natural delta; upper bound for real model families)", worlds=list(FAM_WORLDS), W=W_MAIN,
               quick=QUICK, b_init=FAM_INIT, relu_calibration=calib, A=A_rows, variants=variants, summary=summ, seconds=time.time() - t0,
               notes="A = AT5's natural init-0 W 128 BP-muP net, 'other init' = AT5's init-1 net (critique 17); other variants trained here "
                     "with init 7 (torch.manual_seed(70000 + ws), as ref crit/family*.py), one factor changed each; S over the shared anchors "
                     "(8 for the 8-shift variant); every accuracy in B on B's labels; summary t intervals over worlds are descriptive (the "
                     "family verdict is r2_rgi_verdict's)")
    write_json(path, res)
    say("fam: " + "; ".join(f"{v}: S-gain {s['S_gain']['mean'] if 'S_gain' in s and s['S_gain']['mean'] is not None else float('nan'):+.3f} "
                            f"(level 1 {s.get('n_level1')})" for v, s in summ.items()) + f" ({time.time() - t0:.0f}s)")


# ====================================================================================================================== AT6 (forcing)
F_CONDS = ("natural", "wd x3", "wd x5", "shared split", "bottleneck 4")


def forcing_groups(out):
    folder = os.path.join(out, "rgi_toy", "models")
    groups = []
    batches = [("forcing_wd", [("natural", dict()), ("wd x3", dict(wd=F_WDS[0])), ("wd x5", dict(wd=F_WDS[1]))], {}),
               ("forcing_shared", [("shared split", dict(split="shared"))], dict(fixed_steps=SHARED_STEPS)),
               ("forcing_bneck", [("bottleneck 4", dict(kind="bneck", dt=BNECK_DT))], {})]
    for key, conds, kw in batches:
        path = os.path.join(folder, key + ".pt")
        if os.path.exists(path):
            nets, infos, _ = load_bundle(path)
        else:
            specs = [(c, dict(W=W_MAIN, ws=ws, init=i, **sp)) for c, sp in conds for ws in F_WORLDS for i in F_INITS]
            say(f"forcing: training {len(specs)} nets ({', '.join(c for c, _ in conds)})")
            res = TC.train([s for _, s in specs], **kw)
            nets = res.nets
            infos = [info_of(s, r, cond=c, arm="nat") for (c, s), r in zip(specs, res.rows)]
            save_bundle(path, nets, infos)
        groups.append((nets, infos))
    return groups


def stage_forcing(out):
    path = os.path.join(out, "rgi_toy", "forcing.json")
    if os.path.exists(path):
        say("forcing: output exists, skipped:", path)
        return
    t0 = time.time()
    pool = Pool()
    mids = []
    for nets, infos in forcing_groups(out):
        mids += pool.add(nets, infos)
    fc = free_columns(pool, mids)
    lam = lambdas(pool, mids)
    dl = deltas(pool, mids)
    tcols = task_cols_np(pool)
    ph = phis(pool, mids, fc, tcols)
    cells = {}
    for m in mids:
        cells.setdefault((pool.models[m]["cond"], pool.models[m]["ws"]), []).append(m)
    pairs = [(a, b) for ms in cells.values() for a in ms for b in ms if a != b]
    F = f_pairs(pool, [(a, b, list(range(NPRE)), list(range(NPRE))) for a, b in pairs], fc, tcols)
    rows = {}
    for m in mids:
        r = model_row(pool.models[m])
        r.update(delta_fields(dl.get(m)))
        r.update(Lambda=lam[m]["Lambda"], Lambda_per_shift=lam[m]["per_shift"], Phi=ph[m]["Phi"], acc_proj=ph[m]["acc_proj"],
                 acc_free=fc[m]["acc_end"], A_free=float(fc[m]["A"].mean()),
                 free_column="bottleneck coefficients (v = U e)" if pool.models[m]["kind"] == "bneck" else "free slot column")
        rows[m] = r
    plist = []
    for n, (a, b) in enumerate(pairs):
        d = dict(cond=pool.models[a]["cond"], ws=pool.models[a]["ws"], A=pool.models[a]["name"], B=pool.models[b]["name"])
        d.update(F[n])
        plist.append(d)
        rows[b].setdefault("_t1", []).append(F[n])
    for r in rows.values():
        f = r.pop("_t1", [])
        r.update(T1_gain=float(np.mean([x["t1_gain"] for x in f])) if f else None, LOO=float(np.mean([x["loo"] for x in f])) if f else None)
    summ = {}
    for c in F_CONDS:
        rr = [rows[m] for m in mids if pool.models[m]["cond"] == c]
        summ[c] = dict(n=len(rr), n_level1=sum(bool(r["level1"]) for r in rr),
                       **{k: float(np.mean([r[k] for r in rr if r.get(k) is not None])) if any(r.get(k) is not None for r in rr) else None
                          for k in ("Lambda", "Phi", "T1_gain", "LOO", "A_free", "delta_h2", "final_heldout")})
    res = dict(stage="forcing", measurement="AT6 forcing controls (descriptive)", worlds=list(F_WORLDS), inits=list(F_INITS), W=W_MAIN,
               quick=QUICK, conditions=list(F_CONDS), shared_steps=SHARED_STEPS, bottleneck_dt=BNECK_DT, models=[rows[m] for m in mids],
               pairs=plist, summary=summ, seconds=time.time() - t0,
               notes="natural = the same-definition reference nets trained here; shared split = one 50 % split of the (x, y) pairs for all "
                     "pretraining tasks, 10,000 updates, evaluated at the end (W0); bottleneck = rank-4 task slot U E (ref rgi_bneck), its own "
                     "improvement = the dt coefficients; T1 / LOO between the two inits of a world and condition")
    write_json(path, res)
    say("forcing: " + "; ".join(f"{c}: Phi {s['Phi']:.2f} T1 gain {s['T1_gain']:+.3f} Lambda {s['Lambda']:.2f}" for c, s in summ.items()
                                 if s["Phi"] is not None and s["T1_gain"] is not None and s["Lambda"] is not None) + f" ({time.time() - t0:.0f}s)")


# ====================================================================================================================== selftest
def _check(res, name, cond, info=""):
    res.append(dict(check=name, passed=bool(cond), info=jsonable(info)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {info}", flush=True)


def selftest(out):
    """Code tests (QUICK=1, CPU): AT0's selection rule on synthetic inputs, and the batched line functions of this file against the
    library's per-pair reference functions (TC.s_line, TC.f_line, TC.round_trip, TC.warm_start, TC.pooling, TC.t3_t4, TC.lambda_line,
    TC.b2_A) on tiny nets. Writes <out>/rgi_toy/selftest.json; exit code 1 if a check fails."""
    res, tol = [], 2.0 / 281 + 1e-9
    C = lambda g, mlr, ok=True: dict(gamma=g, mlr=mlr, pass_both=ok)
    s = select_levels([C(0.02, -2.6), C(0.01, -2.1), C(0.003, -0.9), C(-0.004, 1.1), C(-0.02, 2.05)])
    _check(res, "AT0 rule: clean grid -> 4 accepted distinct levels", [x["gamma"] for x in s] == [0.01, 0.003, -0.004, -0.02]
           and all(x["status"] == "accepted" for x in s), [x["gamma"] for x in s])
    s = select_levels([C(0.02, -2.6), C(0.01, -2.1, False), C(0.003, -0.9), C(-0.004, 1.1), C(-0.02, 2.05)])
    _check(res, "AT0 rule: closest fails level 1 -> closest passing, flagged", s[0]["gamma"] == 0.02 and s[0]["status"] == "flagged", s[0])
    s = select_levels([C(0.01, -1.4), C(0.003, -0.2), C(-0.004, 1.1), C(-0.02, 2.05)])
    _check(res, "AT0 rule: two targets on one gamma -> the farther target (x -2) takes its next-closest passing gamma (flagged)",
           [(x["gamma"], x["status"]) for x in s] == [(0.003, "flagged"), (0.01, "accepted"), (-0.004, "accepted"), (-0.02, "accepted")],
           [(x["gamma"], x["status"]) for x in s])
    s = select_levels([C(0.01, -1.65), C(0.003, -1.35), C(-0.02, 2.0)])
    _check(res, "AT0 rule: a displaced target cascades past used gammas and is dropped when none is left",
           [(x["gamma"], x["status"]) for x in s] == [(0.01, "accepted"), (0.003, "accepted"), (None, "dropped"), (-0.02, "accepted")],
           [(x["target_x"], x["gamma"], x["status"]) for x in s])
    s = select_levels([C(0.01, -1.6, False), C(-0.02, 2.0, False)])
    _check(res, "AT0 rule: nothing passes level 1 -> every level dropped", all(x["gamma"] is None and x["status"] == "dropped" for x in s),
           [x["status"] for x in s])
    s = select_levels([C(0.01, -2.0), C(0.003, None), C(-0.02, 2.0)])
    _check(res, "AT0 rule: undefined ratios (delta <= floor) never selected; every grid gamma used once",
           len({x["gamma"] for x in s if x["gamma"] is not None}) == sum(x["gamma"] is not None for x in s), [x["gamma"] for x in s])
    if not QUICK:
        print("selftest: equivalence part needs QUICK=1 (tiny nets)", flush=True)
    else:
        specs = [dict(W=16, ws=0, init=i) for i in (0, 1, 2)]
        tr = TC.train(specs)
        pool = Pool()
        mids = pool.add(tr.nets, [info_of(s_, r, cond="nat", arm="nat") for s_, r in zip(specs, tr.rows)])
        a, b = mids[0], mids[1]
        nets, sets = tr.nets, [TC.new_task_sets(0, j) for j in range(NNEW)]
        fc = free_columns(pool, mids)
        ref = TC.b2_A(nets, [0, 0, 0], nshots=(8,))[8]
        _check(res, "free columns = TC.b2_A's 8-shot column search", np.abs(np.stack([fc[m]["acc"] for m in mids]) - ref).max() <= tol)
        al = alphas(pool, mids)
        q, acc, _ = TC.learn_alpha(nets, np.repeat([0, 1, 2], NNEW), sets * 3)
        _check(res, "alphas = TC.learn_alpha", np.allclose(np.concatenate([al[m]["alpha"] for m in mids]), q, atol=1e-6))
        perms = TC.s_perms(0)
        mine = s_pairs(pool, [(a, b, al[a]["alpha"], list(range(NPRE)))])[0]
        refS = TC.s_line(nets, 0, al[a]["alpha"], nets, 1, sets, perms)
        _check(res, "S pairs = TC.s_line", abs(mine["S_gain"] - refS["S_gain"]) <= tol and np.abs(mine["accB"] - refS["accB"]).max() <= tol,
               (mine["S_gain"], refS["S_gain"]))
        tcols = task_cols_np(pool)
        ph = phis(pool, [a], fc, tcols)[a]
        fp = f_pairs(pool, [(a, b, list(range(NPRE)), list(range(NPRE)))], fc, tcols)[0]
        refF = TC.f_line(nets, 0, 0, fc[a]["v"], fc[a]["acc_end"], nets, 1)
        ok = all(abs(x - y) <= tol for x, y in ((ph["Phi"] * np.mean(fc[a]["acc_end"]), refF["Phi"] * np.mean(fc[a]["acc_end"])),
                                                (fp["t1"], refF["t1"]), (fp["t1_shuf"], refF["t1_shuf"]), (fp["t1_gain"], refF["t1_gain"]),
                                                (fp["loo"], refF["loo"]), (fp["loo_shuf"], refF["loo_shuf"]))) and fp["r1"] == refF["r1"]
        _check(res, "Phi and F pairs (T1 gain, LOO) = TC.f_line", ok, dict(mine=dict(Phi=ph["Phi"], **{k: fp[k] for k in ("t1", "t1_gain", "loo")}),
                                                                            ref={k: refF[k] for k in ("Phi", "t1", "t1_gain", "loo")}))
        rc = recursion(pool, [(a, b)], al)[0]
        rt = TC.round_trip(nets, 0, nets, 1, al[a]["alpha"], sets)
        wm = TC.warm_start(nets, 1, al[a]["alpha"], sets)
        _check(res, "recursion = TC.round_trip and TC.warm_start", abs(rc["RG"] - rt["RG"]) <= tol and abs(rc["warm10"] - wm[10]) <= tol
               and abs(rc["warm40"] - wm[40]) <= tol, dict(mine=[rc["RG"], rc["warm10"], rc["warm40"]], ref=[rt["RG"], wm[10], wm[40]]))
        pl = pooling_lines(pool, [mids], al)[0]
        rp = TC.pooling(nets, [0, 1, 2], np.stack([al[m]["alpha"] for m in mids]), sets)
        _check(res, "pooling = TC.pooling", abs(pl["gain"] - rp["gain"]) <= tol, (pl["gain"], rp["gain"]))
        t34 = t34_pairs(pool, [(a, b)], fc, t3_ok=lambda x, y: True)[0]
        rt34 = TC.t3_t4(nets, 0, fc[a]["v"].astype(np.float32), nets, 1, sets)
        _check(res, "T4 and T3 pairs = TC.t3_t4", np.abs(t34["t4"] - rt34["t4_acc"]).max() <= tol and np.abs(t34["t3"] - rt34["t3_acc"]).max() <= tol,
               dict(mine=[t34["t4_mean"], t34["t3_mean"]], ref=[float(np.mean(rt34["t4_acc"])), float(np.mean(rt34["t3_acc"]))]))
        lm = lambdas(pool, mids)
        rl = TC.lambda_line(nets, [0, 0, 0])
        _check(res, "Lambda = TC.lambda_line (5 unseen shifts)", all(abs(lm[m]["Lambda"] - r["Lambda"]) <= 1e-12 for m, r in zip(mids, rl)))
        inf = dict(ws=3, ky=1, anchors=list(range(NPRE)), shifts=list(M.world(3)[0]), split="shared")
        x, y, _ = anchor_ho(inf, 5)
        sh, ntr = TC.shared_split(3)
        trp = {tuple(p) for p in TC.PAIRS[np.asarray(sh)[:ntr]]}
        _check(res, "LOO held-out pairs of a shared-split net are outside its training pairs", len(x) == P * P - ntr and
               not any((int(u), int(v)) in trp for u, v in zip(x, y)))
        ts = TC.taskset_from_spec(fam_spec("8 shared shifts + spare", 2))
        _check(res, "8 shared shifts + spare: 9 pretraining tasks, slot 8 = spare shift", ts["ntask"] == 9 and
               set(np.unique(ts["t"]).tolist()) == set(range(9)) and info_of(fam_spec("8 shared shifts + spare", 2), None)["shifts"][8] == TC.spare_shift(2))
    ok = all(r["passed"] for r in res)
    write_json(os.path.join(out, "rgi_toy", "selftest.json"), dict(passed=ok, checks=res, quick=QUICK, device=str(DEV)))
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
    say(f"r2_rgi_toy {stage}: out {out}, device {DEV}, QUICK {QUICK}, torch {torch.__version__}, threads {torch.get_num_threads()}")
    t0 = time.time()
    dict(r0=stage_r0, r1=stage_r1, cross=stage_cross, fam=stage_fam, forcing=stage_forcing, selftest=selftest)[stage](out)
    say(f"r2_rgi_toy {stage} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv)
