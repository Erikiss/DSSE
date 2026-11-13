"""Round 2, Part C toy: counter-steering, unlearning, un-searching and primitives in the small networks (plan section 5.1, C-T0 ...
C-T6; shared toy objects 2.5 W0-W8, conventions 2.6; sessions r2_t3 and r2_t4 of section 7; CLI of 9.3). Implementer 7 of 9.5.

Usage   python r2_unlearn_toy.py <out_dir> <stage>      stage: t0 t1 t2 t3 t4 t6 (session r2_t3), t5 (session r2_t4), all
        QUICK=1 python r2_unlearn_toy.py <work_dir> selftest      code test: every stage at QUICK sizes plus unit checks
        env DEVICE (cuda / cpu), QUICK (1 = code test: tiny grids and pc_meta QUICK sizes), NT (CPU threads), R2_MAXC (largest
        number of chains per batched search call)
Outputs <out_dir>/unlearn_toy/<stage>.json: per-seed values of every measure of the stage, t(4) summaries (two-sided 95 % t interval
        over seeds, df = n - 1; the paired seed bootstrap with 10,000 draws next to it, descriptive) and 'rule_preview' = the frozen
        rule of section 5.1 evaluated in-script by rules_<stage>() (r2_unlearn_verdict.py in r2_ana governs; it can import these
        functions, which read only the stage JSON).
        <out_dir>/unlearn_toy/models/: every net trained by a stage (r2_toy_common.save_nets: state, tables, meta, weight checksum) and
        <batch>.json with the batch's training rows (first95, checkpoint, level 1, evaluation point); a rerun loads them (checksums
        verified) instead of retraining.  <out_dir>/unlearn_toy/partial/: per-cell results of the long stages (t3, t4, t5), so an
        interrupted stage resumes; the runner skips a stage whose output JSON exists.
Conventions: one process per stage; every net and search is batched (r2_toy_common, W0); float32 with TF32 off; deterministic given
the fixed seeds (written into every output, OD-G6); outputs written atomically.  Readings of ambiguous plan text are marked 'NOTE'
and listed in NOTES_w7.md."""
from __future__ import annotations

import json
import math
import os
import sys
import time
import zlib

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import r2_toy_common as TC  # noqa: E402

M = TC.M
P, NPRE, NNEW = TC.P, TC.NPRE, TC.NNEW
DEV = TC.DEV
F32, F64 = torch.float32, torch.float64
QUICK = TC.QUICK
PLAN_SHA256 = "5e476340dfdb1d827582c869f2dfc08e32ed8e9a9dbebf7642acceb5bb5edc2d"
SUB = "unlearn_toy"
STAGES_T3 = ("t0", "t1", "t2", "t3", "t4", "t6")
STAGES = STAGES_T3 + ("t5",)

# ---------------------------------------------------------------------------------------------------------------- grids (5.1)
SEEDS = (0, 1) if QUICK else (0, 1, 2, 3, 4)                    # worlds 0-4 (one init per world)
CELLS = ((("bp_mup", 32), ("bp_mup", 64), ("bp_sp", 32)) if QUICK
         else (("bp_mup", 64), ("bp_mup", 128), ("bp_mup", 256), ("bp_sp", 128)))           # C-T0, C-T1, C-T4 one-hot nets
W4_WIDTHS = (32,) if QUICK else (128, 256)                      # C-T4 (c) two-family world
T2_W, T2_CODES, T2_KS = (32 if QUICK else 128), ("ring", "pring", "rand"), ((2, 3) if QUICK else (2, 3, 4))
T3_W, T3_ARMS = (32 if QUICK else 128), ("ring", "pring", "c-onehot")
T3_KS = (1,) if QUICK else (1, 2)                                # K = 1 primary, K = 2 check
T3_LS = (3, 12) if QUICK else (3, 6, 9, 12)                      # word-length arcs
T5_WIDTHS = (32,) if QUICK else (128, 256)
T5_ROUNDS = 4 if QUICK else 25                                   # searches s_0 .. s_24
T5_PC_LAST = 2 if QUICK else 11                                  # pc arm: s_0 .. s_11
T6_W, T6_LS = (32 if QUICK else 128), ((1, 2) if QUICK else (1, 2, 3, 4))
T6_NULLS, T6_MODES, SNAP_EVERY = (10 if QUICK else 50), 5, 25
T1_KS = (1, 2, 8) if QUICK else tuple(range(1, 9))              # x / y columns: keep 2k of 16 dims
T1_TK = (1, 5) if QUICK else (1, 2, 3, 4, 5)                     # task columns: keep 2K of 11 dims
T1_RS = (2, 16, 32) if QUICK else (2, 4, 8, 16, 32, 64)          # hidden projections z1 / h2: top-r (r = W included)
NDRAW = 5                                                        # random bases / OD-C2 draws
NPERM = (20, 10, 20) if QUICK else (1000, 1000, 500)             # C-T0 permutations: frequency share, LOO ring, answer spectrum
BUDGETS = tuple(TC.BUDGETS)                                      # (300, 3000) best-of-budget (OD-C1); QUICK (20, 60)
B_SHORT, B_LONG = BUDGETS
FT_EVAL = tuple(TC.FT_EVAL)
UNLEARN_ACC, CHANCE_T, CHANCE_A = 0.11, 1.0 / NPRE, 1.0 / P
BOOT_B, BOOT_SEED = 10_000, 0
MAXC = int(os.environ.get("R2_MAXC", "8192" if DEV.type == "cuda" else "1024"))
RNG_SEEDS = dict(boot=BOOT_SEED, structure="default_rng(500 + ws) (c1_structure order)", random_subspaces="torch.Generator(crc32(key))",
                 t6_nulls="default_rng(ws) (c4_traj)", rand_arm="torch.Generator(crc32(key))", mlp_probe="torch.Generator(0 / 1)")


# ================================================================================================================ I/O and statistics
def _clean(o):
    """JSON-ready copy: numpy / torch to lists and floats, NaN / inf to None, dict keys to str."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, torch.Tensor):
        return _clean(o.detach().cpu().numpy())
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    return o


def write_json(obj, path):
    """Atomic JSON write (tmp file + os.replace)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True); tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_clean(obj), f)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def _arr(v):
    """float array with None -> NaN."""
    return np.array([np.nan if x is None else x for x in np.ravel(np.asarray(v, dtype=object))], dtype=float)


def tci(v, level=0.95):
    """Two-sided t interval over seeds (df = n - 1; 2.6): dict(n, mean, sd, lo, hi). Non-finite values are dropped."""
    from scipy import stats
    v = _arr(v); v = v[np.isfinite(v)]; n = len(v)
    if n == 0:
        return dict(n=0, mean=None, sd=None, lo=None, hi=None)
    m = float(v.mean())
    if n < 2:
        return dict(n=1, mean=m, sd=None, lo=None, hi=None)
    sd = float(v.std(ddof=1)); h = float(stats.t.ppf(0.5 + level / 2, n - 1)) * sd / math.sqrt(n)
    return dict(n=n, mean=m, sd=sd, lo=m - h, hi=m + h)


def boot95(v, B=BOOT_B, seed=BOOT_SEED):
    """Paired seed bootstrap (seeds as units, 10,000 draws, percentile 95 %; reported next to the t interval, decides nothing)."""
    v = _arr(v); v = v[np.isfinite(v)]
    if len(v) < 2:
        return None
    g = np.random.default_rng(seed); bs = v[g.integers(0, len(v), (B, len(v)))].mean(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def summ(v, level=0.95):
    d = tci(v, level); d["boot95"] = boot95(v); d["values"] = [None if not np.isfinite(x) else float(x) for x in _arr(v)]
    return d


def quorum(n):
    """'>= 4 of 5 seeds' (5.1), scaled for other seed counts: ceil(0.8 n)."""
    return int(math.ceil(0.8 * n - 1e-9))


def crc(key):
    return zlib.crc32(str(key).encode()) & 0x7FFFFFFF


def tdir(out, *parts):
    return os.path.join(out, SUB, *parts)


PROV = dict(partials={}, nets={})        # provenance of the partial results and net batches the running stage used (-> meta)


def provenance(extra=None):
    """What a partial result depends on beyond its key (review unlearn_toy-4, pc_toy-4): the sha256 of r2_toy_common.py,
    r2_unlearn_toy.py and r2_pc_toy.py, QUICK, the seeds and the budgets; extra = caller-specific inputs (C-T7: the round-1 net files of
    the width with their sha256)."""
    d = dict(code_sha256=lib_sha256(), quick=QUICK, seeds=list(SEEDS), budgets=list(BUDGETS))
    if extra is not None:
        d["extra"] = extra
    return _clean(d)


def cached(out, key, fn, sub=SUB, extra=None, store=None):
    """Per-cell partial result <out>/<sub>/partial/<key>.json: reused only if the provenance stored with it equals provenance(extra) of
    the running process; otherwise fn() is recomputed (with a message naming what differs) and written, unless store(result) is False
    (e.g. a C-T7 width that did not run). Every partial a stage used is listed with its code hash in PROV['partials'] (meta.partials)."""
    p = os.path.join(out, sub, "partial", key + ".json"); want = provenance(extra); tag = f"{sub}/{key}"
    if os.path.exists(p):
        r = read_json(p); have = r.get("_provenance")
        if have == want:
            print(f"  [{key}] partial result exists (same code and inputs), loaded", flush=True)
            PROV["partials"][tag] = dict(reused=True, code_sha256=have["code_sha256"]); return r
        why = "no stored provenance" if not isinstance(have, dict) else ", ".join(k for k in sorted(set(want) | set(have)) if have.get(k) != want.get(k))
        print(f"  [{key}] partial result exists but its {why} differ(s) from this run: recomputed", flush=True)
    t0 = time.time(); r = _clean(fn()); r["seconds"] = time.time() - t0; r["_provenance"] = want
    stored = store is None or bool(store(r))
    if stored:
        write_json(r, p)
    PROV["partials"][tag] = dict(reused=False, stored=stored, code_sha256=want["code_sha256"])
    print(f"  [{key}] done ({r['seconds']:.0f}s)" + ("" if stored else ", not stored as a partial"), flush=True)
    return r


def lib_sha256():
    import hashlib
    return {f: hashlib.sha256(open(os.path.join(_HERE, f), "rb").read()).hexdigest() for f in ("r2_toy_common.py", "r2_unlearn_toy.py", "r2_pc_toy.py")
            if os.path.exists(os.path.join(_HERE, f))}


AXIS_SIGN_RULE = ("principal axes of the centred task columns (task_pcs: float64 SVD on the CPU) are oriented so that the task column with "
                  "the largest absolute coordinate on the axis has a positive coordinate (review unlearn_toy-2; C-T4 push along u1)")


def meta(stage, t0):
    return dict(stage=stage, plan_sha256=PLAN_SHA256, quick=QUICK, device=str(DEV), torch=torch.__version__, seeds=list(SEEDS),
                budgets=list(BUDGETS), ft_eval=list(FT_EVAL), ndraw=NDRAW, rng_seeds=RNG_SEEDS, axis_sign_rule=AXIS_SIGN_RULE,
                code_sha256=lib_sha256(), partials=dict(PROV["partials"]), nets=dict(PROV["nets"]),
                seconds=time.time() - t0, written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


# ================================================================================================================ nets
def nets_batch(out, tag, specs, **train_kw):
    """Train the nets of specs as one batch (TC.train: the W0 checkpoint rule unless train_kw says otherwise) and save them, or load
    them from <out>/unlearn_toy/models/ when the batch file <tag>.json and every net file exist (TC.load_nets verifies the weight
    checksums). Snapshots (train_kw snapshot_every) are kept in <tag>.extra.pt. Returns (Nets, rows, extra)."""
    folder = tdir(out, "models"); mp = os.path.join(folder, tag + ".json"); names = [TC.net_name(s) for s in specs]
    paths = [os.path.join(folder, n + ".pt") for n in names]
    if os.path.exists(mp):
        info = read_json(mp)
        if info.get("names") == names and all(os.path.exists(p) for p in paths):
            note = net_code_check(tag, info)
            nets = TC.load_nets(paths); extra = None
            if info.get("extra"):
                extra = torch.load(os.path.join(folder, tag + ".extra.pt"), map_location="cpu", weights_only=False)
            PROV["nets"][tag] = dict(reused=True, code_sha256=info.get("code_sha256"), note=note)
            return nets, info["rows"], extra
    t0 = time.time(); res = TC.train(specs, **train_kw)
    TC.save_nets(res.nets, list(range(res.nets.B)), folder, names=names)
    extra = None
    if train_kw.get("snapshot_every"):
        extra = dict(snapshots=[(int(s), v, np.asarray(a)) for s, v, a in res.snapshots])
        tmp = os.path.join(folder, tag + ".extra.pt.tmp"); torch.save(extra, tmp); os.replace(tmp, os.path.join(folder, tag + ".extra.pt"))
    code = lib_sha256()
    write_json(dict(tag=tag, names=names, rows=res.rows, extra=extra is not None, seconds=time.time() - t0, code_sha256=code), mp)
    PROV["nets"][tag] = dict(reused=False, code_sha256=code)
    print(f"  trained batch {tag}: {len(specs)} nets ({time.time() - t0:.0f}s)", flush=True)
    return res.nets, res.rows, extra


def net_code_check(tag, info):
    """Saved nets are reused only with the trainer that made them (review unlearn_toy-4): if r2_toy_common.py's sha256 differs from the
    one stored with the batch (or none is stored), stop with a message, because reusing them mixes two trainer versions and retraining
    would replace nets that finished stages of this session analysed (W0: the nets of one session); env R2_REUSE_NETS=1 reuses them and
    the note is recorded in meta.nets. A difference in the stage scripts only is reported and recorded (the specs are fixed by the
    names, the training is the library's)."""
    old, new = info.get("code_sha256") or {}, lib_sha256()
    if old.get("r2_toy_common.py") != new.get("r2_toy_common.py"):
        msg = (f"models/{tag}: trained by r2_toy_common.py {str(old.get('r2_toy_common.py'))[:12]}, the library is now "
               f"{new.get('r2_toy_common.py', '')[:12]}. Delete {SUB}/models/ and the stage outputs to retrain every net, or set "
               "R2_REUSE_NETS=1 to reuse these nets (recorded in meta.nets).")
        if os.environ.get("R2_REUSE_NETS") != "1":
            raise SystemExit(msg)
        print("  WARNING: " + msg, flush=True)
        return "reused with a different r2_toy_common.py (R2_REUSE_NETS=1)"
    diff = [f for f in sorted(new) if f != "r2_toy_common.py" and old.get(f) != new.get(f)]
    if diff:
        print(f"  WARNING: models/{tag} were trained when {', '.join(diff)} differed from now; reused", flush=True)
        return f"reused; {', '.join(diff)} differed at training"
    return "reused"


def slim(rows):
    """The training row fields reported with every measure (W0: evaluation point recorded)."""
    keys = ("name", "first95", "checkpoint", "level1", "eval_point", "updates", "final_heldout", "level_acc", "checksum")
    return [{k: r.get(k) for k in keys} for r in rows]


def onehot_nets(out, arm, W, seeds=None):
    """The one-hot W1 nets of a cell (BP-muP / BP-SP, init 0 = torch.manual_seed(ws)), trained once per session (C-T0, C-T1, C-T4;
    r2_t4 trains its own for C-T5)."""
    return nets_batch(out, f"onehot_{arm}_W{W}", [dict(W=W, ws=ws, arm=arm) for ws in (SEEDS if seeds is None else seeds)])


# ================================================================================================================ tasks
class Tasks:
    """Old (pretraining) tasks of B nets: column index t [B, n] (TC task_cols index), shift c [B, n], family sign ky [B, n] (label
    (x + ky y + c) mod 17) and split [B, n, 289] (pair indices; positions 0-143 training, 144-288 held-out)."""

    def __init__(self, t, c, ky, split, ws):
        self.t, self.c, self.ky = (np.asarray(a, np.int64) for a in (t, c, ky))
        self.split = np.asarray(split, np.int64); self.ws = list(ws)

    @property
    def B(self):
        return self.t.shape[0]

    @property
    def n(self):
        return self.t.shape[1]

    @staticmethod
    def w1(ws_list):
        """W1: task k = shift pre[k] in slot k, split splits[k] of world(ws)."""
        t, c, ky, sp = [], [], [], []
        for ws in ws_list:
            pre, _, _, splits = M.world(ws)
            t.append(list(range(NPRE))); c.append(list(pre)); ky.append([1] * NPRE); sp.append([splits[k] for k in range(NPRE)])
        return Tasks(t, c, ky, sp, ws_list)

    @staticmethod
    def w3(ws_list, shifts_list):
        """W3 circle world: task index = shift value c, split splits[c mod 16] of world(ws) (ref c3_angle)."""
        t, c, ky, sp = [], [], [], []
        for ws, sh in zip(ws_list, shifts_list):
            splits = M.world(ws)[3]
            t.append(list(sh)); c.append(list(sh)); ky.append([1] * len(sh)); sp.append([splits[s % len(splits)] for s in sh])
        return Tasks(t, c, ky, sp, ws_list)

    @staticmethod
    def w4(ws_list, fam=None):
        """W4 two-family world: old add tasks (x + y + c, slots 0-11) and / or sub tasks (x - y + c, slots 12-23), splits TC.w4_split."""
        t, c, ky, sp = [], [], [], []
        for ws in ws_list:
            old, _ = TC.w4_tasks(ws); sel = [(f, cc, s) for f, cc, s in old if fam is None or f == fam]
            t.append([s for _, _, s in sel]); c.append([cc for _, cc, _ in sel]); ky.append([1 if f == "add" else -1 for f, _, _ in sel])
            sp.append([TC.w4_split(ws, s) for _, _, s in sel])
        return Tasks(t, c, ky, sp, ws_list)


def _chains(tasks, var_net, lo, hi):
    """Chains (variant v, task i) in that order for the split positions lo:hi: net index [C], task position [C], x, y, label [C, N],
    ky [C, 1]."""
    var_net = np.asarray(var_net, np.int64); n = tasks.n
    vb = np.repeat(var_net, n); tp = np.tile(np.arange(n), len(var_net))
    idx = tasks.split[vb, tp, lo:hi]; x, y = TC.PAIRS[idx, 0], TC.PAIRS[idx, 1]
    ky = tasks.ky[vb, tp][:, None]; lab = (x + ky * y + tasks.c[vb, tp][:, None]) % P
    return vb, tp, x, y, lab, ky


def _chain_cols(nets, tasks, vb, tp, cols=None):
    """Task column of every chain [C, W]: the net's own column (task_cols index t) or the variant's replacement cols [V, n, W]."""
    if cols is not None:
        return torch.as_tensor(cols, dtype=F32, device=DEV).reshape(-1, nets.W)
    T = nets.task_cols()
    return T[torch.as_tensor(vb, device=DEV), :, torch.as_tensor(tasks.t[vb, tp], device=DEV)]


def ops_slice(ops, idx):
    """Ops restricted to the chains idx (slice or index array) of its leading dim; size-1 (broadcast) tensors are kept."""
    if ops is None:
        return None
    o = TC.Ops()
    for k in TC.Ops.KEYS:
        v = getattr(ops, k)
        if isinstance(v, torch.Tensor) and v.shape[0] > 1:
            v = v[idx] if isinstance(idx, slice) else v[torch.as_tensor(np.asarray(idx), device=v.device)]
        setattr(o, k, v)
    return o


def _per(N, W, mult=4):
    return max(1, TC.CHUNK // max(1, N * W * mult))


@torch.no_grad()
def _pred(nets, vb, cols, x, y, ops):
    C, N = x.shape; per = _per(N, nets.W); out = []
    for a in range(0, C, per):
        b = min(C, a + per)
        out.append(TC.predictions(nets, vb[a:b], cols[a:b], (x[a:b], y[a:b]), ops_slice(ops, slice(a, b))))
    return np.concatenate(out)


@torch.no_grad()
def _hidden(nets, vb, cols, x, y, ops, layer):
    """Hidden states [C, N, W] (layer 'h1' or 'h2'), chunked."""
    C, N = x.shape; per = _per(N, nets.W, 6); out = []
    for a in range(0, C, per):
        b = min(C, a + per)
        h1, h2 = TC.hidden_states(nets, vb[a:b], cols[a:b], (x[a:b], y[a:b]), ops_slice(ops, slice(a, b)))
        out.append(h1 if layer == "h1" else h2)
    return torch.cat(out)


def _expand(ops, V, n):
    return None if ops is None else ops.expand(np.repeat(np.arange(V), n))


def eval_old(nets, tasks, var_net, cols=None, ops=None, chain_ops=None):
    """Old-task held-out accuracy of variants (net var_net[v] with optional replacement columns cols [V, n, W] and Ops per variant,
    or per chain via chain_ops): accuracy per task [V, n] and mean [V]; output integrity and modal-shift map (OD-C16, TC.modal_shift_map
    with the family sign: modal value of (prediction - x - ky y) mod 17)."""
    var_net = np.asarray(var_net, np.int64); V, n = len(var_net), tasks.n
    vb, tp, x, y, lab, ky = _chains(tasks, var_net, 144, P * P)
    oc = chain_ops if chain_ops is not None else _expand(ops, V, n)
    pred = _pred(nets, vb, _chain_cols(nets, tasks, vb, tp, cols), x, y, oc)
    acc = (pred == lab).mean(1).reshape(V, n)
    maps = [TC.modal_shift_map(pred[v * n:(v + 1) * n], x[v * n:(v + 1) * n], (ky * y)[v * n:(v + 1) * n], tasks.c[var_net[v]]) for v in range(V)]
    return dict(acc=acc, mean=acc.mean(1), integrity=np.array([m["integrity"] for m in maps]), label=[m["label"] for m in maps],
                modal=np.array([m["modal"] for m in maps]), share=np.array([m["share"] for m in maps]), pred=pred)


def sum_probe_gen(nets, tasks, var_net, cols=None, ops=None, chain_ops=None):
    """W8 per-task sum probe per variant: per task the 289 pairs in split order with the variant's column and ops; ridge (lambda 1e-2,
    TC.ridge_probe) from h2 to one-hot (x + ky y) mod 17 on the 144 training pairs, accuracy on the 145 held-out pairs; mean over the
    tasks -> [V]. (W1 nets: identical to TC.sum_probe; NOTE: W4 sub tasks use x - y.)"""
    var_net = np.asarray(var_net, np.int64); V, n = len(var_net), tasks.n
    vb, tp, x, y, lab, ky = _chains(tasks, var_net, 0, P * P)
    s = (x + ky * y) % P; cols_c = _chain_cols(nets, tasks, vb, tp, cols)
    oc = chain_ops if chain_ops is not None else _expand(ops, V, n)
    C = len(vb); per = _per(P * P, nets.W + 1, 10); acc = []
    for a in range(0, C, per):
        b = min(C, a + per)
        h2 = _hidden(nets, vb[a:b], cols_c[a:b], x[a:b], y[a:b], ops_slice(oc, slice(a, b)), "h2")
        sl = torch.as_tensor(s[a:b], device=h2.device)
        acc.append(TC.ridge_probe(h2[:, :144], sl[:, :144], h2[:, 144:], sl[:, 144:], P, 1e-2).cpu().numpy())
    return np.concatenate(acc).reshape(V, n).mean(1)


def probe_feats(nets, tasks, var_net, layer, cols=None, ops=None, states=None):
    """W8 task-identity and pooled-answer probe data per variant: per task its 145 held-out pairs (split positions 144-288) at layer
    h1 / h2; the first 72 train the probe, the last 73 test it (TC.heldout_halves). states: optional held-out states [V * n, 145, W]
    (e.g. relaxed PC states). Returns dict(Htr [V, 72 n, W], Hte [V, 73 n, W], task_tr, task_te, ans_tr, ans_te, n)."""
    var_net = np.asarray(var_net, np.int64); V, n = len(var_net), tasks.n
    vb, tp, x, y, lab, ky = _chains(tasks, var_net, 144, P * P)
    H = states if states is not None else _hidden(nets, vb, _chain_cols(nets, tasks, vb, tp, cols), x, y, _expand(ops, V, n), layer)
    H = torch.as_tensor(H, device=DEV).reshape(V, n, P * P - 144, -1); half = 72
    tk = torch.arange(n, device=DEV).view(1, n, 1).expand(V, n, P * P - 144); an = torch.as_tensor(lab, device=DEV).view(V, n, -1)
    cut = lambda A, a, b: A[:, :, a:b].reshape(V, -1, *A.shape[3:])
    return dict(Htr=cut(H, 0, half), Hte=cut(H, half, P * P - 144), task_tr=cut(tk, 0, half), task_te=cut(tk, half, P * P - 144),
                ans_tr=cut(an, 0, half), ans_te=cut(an, half, P * P - 144), n=n)


def probes(f, mlp=True, seed=0):
    """Linear ridge (lambda 1e-2) and OD-C10 MLP probes for task identity (chance 1 / n) and the pooled answer (chance 1 / 17) -> [V]."""
    out = dict(task_lin=TC.ridge_probe(f["Htr"], f["task_tr"], f["Hte"], f["task_te"], f["n"]).cpu().numpy(),
               ans_lin=TC.ridge_probe(f["Htr"], f["ans_tr"], f["Hte"], f["ans_te"], P).cpu().numpy())
    if mlp:
        out["task_mlp"] = np.asarray(TC.mlp_probe(f["Htr"], f["task_tr"], f["Hte"], f["task_te"], f["n"], seed=seed))
        out["ans_mlp"] = np.asarray(TC.mlp_probe(f["Htr"], f["ans_tr"], f["Hte"], f["ans_te"], P, seed=seed + 1))
    return out


# ================================================================================================================ operations
LAYER_KEYS = {"z1": ("z1_U", "z1_ref", "z1_lam"), "h1": ("h1_Q", "h1_mu", "h1_lam"), "h2": ("h2_Q", "h2_mu", "h2_lam")}


def mk_ops(items, W):
    """Ops with leading dim V from per-variant dicts: z1_U [W, r] (+ z1_ref [W], z1_lam), z1_shift [W], h1_Q / h2_Q [W, r] (+ mu, lam);
    missing keys = no-op (zero-padded subspaces, zero shift, lam 1). Returns None if no variant has an operation."""
    V = len(items); kw = {}
    for layer, (kU, kc, kl) in LAYER_KEYS.items():
        have = [it for it in items if it.get(kU) is not None]
        if not have:
            continue
        R = max(1, max(int(np.shape(it[kU])[1]) for it in have))
        U = torch.zeros(V, W, R, dtype=F32); c = torch.zeros(V, W, dtype=F32); lam = torch.ones(V, dtype=F32)
        for v, it in enumerate(items):
            if it.get(kU) is None:
                continue
            u = torch.as_tensor(it[kU], dtype=F32).cpu(); U[v, :, :u.shape[1]] = u
            if it.get(kc) is not None:
                c[v] = torch.as_tensor(it[kc], dtype=F32).cpu()
            if it.get(kl) is not None:
                lam[v] = float(it[kl])
        kw[kU], kw[kc], kw[kl] = U, c, lam
    if any(it.get("z1_shift") is not None for it in items):
        S = torch.zeros(V, W, dtype=F32)
        for v, it in enumerate(items):
            if it.get("z1_shift") is not None:
                S[v] = torch.as_tensor(it["z1_shift"], dtype=F32).cpu()
        kw["z1_shift"] = S
    return TC.Ops(**kw) if kw else None


def rand_sub(W, r, n, key):
    """n random isotropic r-dim subspaces of R^W [n, W, r] (TC.random_subspaces with the seed crc32(key); OD-G6)."""
    return TC.random_subspaces(W, int(r), n, crc(key))


def orth_cols(A, tol=1e-9):
    """Orthonormal basis [W, rank] of the columns of A (float64 SVD)."""
    A = torch.as_tensor(A, dtype=F64)
    if A.shape[1] == 0:
        return A
    U, S, _ = torch.linalg.svd(A, full_matrices=False)
    return U[:, S > tol * max(float(S[0]), 1e-300)]


def complement(Q):
    """Orthonormal basis [W, W - r] of the orthogonal complement of span(Q) (Q [W, r] orthonormal)."""
    W, r = Q.shape
    g = torch.Generator().manual_seed(12345)
    A = torch.cat([torch.as_tensor(Q, dtype=F64).cpu(), torch.randn(W, W - r, generator=g, dtype=F64)], 1)
    return torch.linalg.qr(A)[0][:, r:]


def orient_axes(U, Tc):
    """AXIS_SIGN_RULE: flip each axis u (column of U [B, W, n]) so that the column of Tc [B, W, m] with the largest absolute coordinate
    u^T Tc_t has a positive coordinate (first index on a tie). Returns (U oriented, flipped [B, n] bool)."""
    C = torch.einsum("bwn,bwt->bnt", U, Tc); s = torch.gather(C, -1, C.abs().argmax(-1, keepdim=True)).squeeze(-1)
    flip = s < 0
    return U * torch.where(flip, -1.0, 1.0).to(U.dtype).unsqueeze(1), flip


def task_pcs(nets, tidx=None, return_flips=False):
    """Principal axes (in R^W) of each net's centred task columns: U [B, W, n] (left singular vectors, descending; float64 SVD on the
    CPU, so device-independent; signs by AXIS_SIGN_RULE), S [B, n], the task-mean column [B, W] (CPU float64); with return_flips also
    the per-axis sign flips [B, n] (bool numpy)."""
    T = nets.task_cols()[:, :, list(range(NPRE)) if tidx is None else list(tidx)].double().cpu()
    Tc = T - T.mean(-1, keepdim=True); U, S, _ = torch.linalg.svd(Tc, full_matrices=False); U, flip = orient_axes(U, Tc)
    return (U, S, T.mean(-1), flip.numpy()) if return_flips else (U, S, T.mean(-1))


def damage_match(nets, tasks, base_sp, target, layer, center, r0, key, var_net=None):
    """OD-C2 damage-matched random control for an operation at layer ('z1': block z - lam P (z - ref); 'h1' / 'h2': erasure around
    the layer mean): random isotropic subspaces of dimension r0, 2 r0, 4 r0, ... (<= W / 2), NDRAW draws each (rand_sub, key per
    dimension), the drop = mean over the draws and the cell's nets of the sum-probe drop (base minus operated); TC.damage_match picks the
    smallest dimension d* whose mean drop reaches the target, with partial removal by bisection of one shared lam at d* when it overshoots
    by more than 0.02 (Amendment 2; before: only at r0); matched within 0.02 (G45).
    Returns the TC.damage_match dict plus the matched subspaces R [NDRAW, W, r] and the key."""
    W = nets.W; var_net = np.arange(nets.B) if var_net is None else np.asarray(var_net); memo = {}
    kU, kc, kl = LAYER_KEYS[layer]

    def drop_of(r, lam):
        k_ = (int(r), round(float(lam), 12))
        if k_ not in memo:
            R = rand_sub(W, r, NDRAW, f"{key}|r{r}")
            vv = np.repeat(var_net, NDRAW)
            items = [{kU: R[d], kc: center[b], kl: lam} for b in var_net for d in range(NDRAW)]
            sp = sum_probe_gen(nets, tasks, vv, ops=mk_ops(items, W))
            memo[k_] = float(np.mean(np.repeat(np.asarray(base_sp)[var_net], NDRAW) - sp))
        return memo[k_]
    res = TC.damage_match(float(target), drop_of, int(r0), W, ndraw=NDRAW)
    res = dict(res, target=float(target), r0=int(r0), key=key, layer=layer)
    res["R"] = rand_sub(W, res["r"], NDRAW, f"{key}|r{res['r']}")
    return res


def dm_items(dm, center, var_net, layer):
    """Per-(net, draw) op items of a matched control (variants ordered net-major, NDRAW draws each)."""
    kU, kc, kl = LAYER_KEYS[layer]
    return [{kU: dm["R"][d], kc: center[b], kl: dm["lam"]} for b in var_net for d in range(NDRAW)]


def dm_chain_items(dm, center, var_net, n, layer):
    """Per-chain op items of a matched control for searches: chain (net b, task i) uses draw i mod NDRAW (OD-C2: one matched block per
    search; new task j uses draw j)."""
    kU, kc, kl = LAYER_KEYS[layer]
    return [{kU: dm["R"][i % NDRAW], kc: center[b], kl: dm["lam"]} for b in var_net for i in range(n)]


def dm_public(dm):
    return {k: v for k, v in dm.items() if k != "R"}


# ================================================================================================================ searches
def _sl(a, idx, C):
    if a is None:
        return None
    if isinstance(a, TC.Ops):
        return ops_slice(a, idx)
    if isinstance(a, dict):
        return {k: _sl(v, idx, C) for k, v in a.items()}
    if isinstance(a, tuple):
        return tuple(_sl(v, idx, C) for v in a)
    if isinstance(a, (float, int, str)):
        return a
    if isinstance(a, torch.Tensor):
        return a[idx] if a.dim() >= 1 and a.shape[0] == C else a
    arr = np.asarray(a)
    if arr.ndim >= 1 and arr.shape[0] == C and (arr.ndim >= 2 or C > 1):
        return arr[idx]
    return a


def _merge(parts):
    out = dict(acc={n: np.concatenate([p["acc"][n] for p in parts]) for n in parts[0]["acc"]})
    out["A"] = {n: (None if parts[0]["A"][n] is None else np.concatenate([p["A"][n] for p in parts])) for n in parts[0]["A"]}
    cat = lambda xs: ({k: cat([x[k] for x in xs]) for k in xs[0]} if isinstance(xs[0], dict) else np.concatenate(xs))
    out["best"] = {b: cat([p["best"][b] for p in parts]) for b in parts[0]["best"]}
    out["v"] = np.concatenate([p["v"] for p in parts]); out["q"] = np.concatenate([p["q"] for p in parts])
    return out


def best_ho(r, b, name="ho"):
    """Held-out accuracy at the best validation evaluation within budget b (TC.column_search: best[b]['heldout'][name]; earlier drafts
    used best[b][name])."""
    d = r["best"][b]
    return d["heldout"][name] if "heldout" in d else d[name]


def search(nets, net_of, shots, heldout=None, **kw):
    """TC.column_search over chains, split into calls of at most MAXC chains (chains are independent; per-chain arguments sliced)."""
    net_of = np.asarray(net_of, np.int64); C = len(net_of)
    if C <= MAXC:
        return TC.column_search(nets, net_of, shots, heldout, **kw)
    parts = []
    for a in range(0, C, MAXC):
        idx = slice(a, min(C, a + MAXC)); kk = {k: _sl(v, idx, C) for k, v in kw.items()}
        if "lr" in kw and np.ndim(kw["lr"]) == 1:
            kk["lr"] = np.asarray(kw["lr"])[idx]
        parts.append(TC.column_search(nets, net_of[idx], _sl(shots, idx, C), _sl(heldout, idx, C), **kk))
    return _merge(parts)


def relearn_old(nets, tasks, var_net, ops=None, chain_ops=None, pc_eta=None, budgets=BUDGETS):
    """W7 / OD-C14 old-task relearning per variant: per task the column from zero (Adam lr 1e-2) on the shots = training split
    positions 0-7, validation = positions 8-11 (every 10 steps), held-out = the task's 145 held-out pairs; best-of-budget accuracies
    (one run of max(budgets) steps). ops per variant (or chain_ops per chain); pc_eta per net = PC local update. Returns {budget: [V, n]}."""
    var_net = np.asarray(var_net, np.int64); V, n = len(var_net), tasks.n
    vb, tp, xs, ys, ls, _ = _chains(tasks, var_net, 0, 8)
    _, _, xv, yv, lv, _ = _chains(tasks, var_net, 8, 12)
    _, _, xh, yh, lh, _ = _chains(tasks, var_net, 144, P * P)
    oc = chain_ops if chain_ops is not None else _expand(ops, V, n)
    pc = None if pc_eta is None else dict(eta=np.asarray(pc_eta, np.float64)[vb])
    r = search(nets, vb, (xs, ys, ls), {"ho": (xh, yh, lh)}, val=(xv, yv, lv), budgets=budgets, ops=oc, pc=pc)
    return {b: best_ho(r, b).reshape(V, n) for b in budgets}


def new_sets(ws_of, vb, jj, val=True, nshot=8):
    sets = [TC.new_task_sets(ws_of[b], int(j), val=val, nshot=nshot) for b, j in zip(vb, jj)]
    return TC.stack_sets(sets, "shot"), TC.stack_sets(sets, "ho"), (TC.stack_sets(sets, "val") if val else None)


def search_new(nets, ws_of, var_net, ops=None, chain_ops=None, pc_eta=None, budgets=BUDGETS):
    """A_new (C-T4): best-of-budget accuracy of the column search for the 4 new tasks per variant (W7: shots = split positions 0-7,
    validation 8-11, held-out 12-288). Returns {budget: [V, 4]}."""
    var_net = np.asarray(var_net, np.int64); V = len(var_net)
    vb = np.repeat(var_net, NNEW); jj = np.tile(np.arange(NNEW), V); sh, ho, va = new_sets(ws_of, vb, jj)
    oc = chain_ops if chain_ops is not None else _expand(ops, V, NNEW)
    pc = None if pc_eta is None else dict(eta=np.asarray(pc_eta, np.float64)[vb])
    r = search(nets, vb, sh, {"ho": ho}, val=va, budgets=budgets, ops=oc, pc=pc)
    return {b: best_ho(r, b).reshape(V, NNEW) for b in budgets}


def b2_var(nets, ws_of, var_net, ops=None, keep=None, nshots=(8, 4), pc_eta=None):
    """Column-search part of the W1 B2 per variant with ops on (or the new column kept in keep[v] [V, W, r] after every step):
    per new task j and nshot the slot column from zero, Adam lr 1e-2, FT_STEPS steps, held-out = split positions 8-288, accuracy at
    FT_EVAL. Returns {nshot: [V, 4, len(FT_EVAL)]}."""
    var_net = np.asarray(var_net, np.int64); V = len(var_net); out = {}
    vb = np.repeat(var_net, NNEW); jj = np.tile(np.arange(NNEW), V)
    oc = _expand(ops, V, NNEW)
    kc = None if keep is None else torch.as_tensor(keep, dtype=F32).repeat_interleave(NNEW, 0)
    pc = None if pc_eta is None else dict(eta=np.asarray(pc_eta, np.float64)[vb])
    for ns in nshots:
        sh, ho, _ = new_sets(ws_of, vb, jj, val=False, nshot=ns)
        r = search(nets, vb, sh, {"ho": ho}, eval_at=FT_EVAL, ops=oc, keep=kc, pc=pc)
        out[ns] = r["acc"]["ho"].reshape(V, NNEW, len(FT_EVAL))
    return out


def b2_value(A, F):
    """B2 = mean over the new tasks of (mean A over FT_EVAL - mean F over FT_EVAL); A, F [V, 4, E] -> [V]."""
    return (np.asarray(A).mean(-1) - np.asarray(F).mean(-1)).mean(-1)


def fresh_F(arm, W, ws_list):
    """F of the W1 B2 (pc_meta.finetune 'fresh', TC.fresh_F) per seed: {nshot: [S, 4, E]}."""
    return TC.fresh_F("bp_sp" if arm == "bp_sp" else "bp_mup", W, list(ws_list))


# ================================================================================================================ T4b (shared with r2_pc_toy)
def fit_eraser(nets, tasks, layer):
    """W8 class-mean eraser of task identity at layer for every net, fitted on the feedforward states of the net's n tasks x 289
    pairs (TC.class_mean_eraser): Q [B, W, r] (zero-padded), mu [B, W] (also the layer mean used by random erasures)."""
    allv = np.arange(nets.B); vb, tp, x, y, lab, ky = _chains(tasks, allv, 0, P * P)
    H = _hidden(nets, vb, _chain_cols(nets, tasks, vb, tp), x, y, None, layer).reshape(nets.B, tasks.n * P * P, -1)
    lab_t = torch.arange(tasks.n, device=DEV).repeat_interleave(P * P)
    return TC.class_mean_eraser(H, lab_t, tasks.n)


def eraser_items(Q, mu, layer, var_net):
    kU, kc, _ = LAYER_KEYS[layer]
    return [{kU: Q[b], kc: mu[b]} for b in var_net]


def base_measures(nets, tasks, ws_of=None, arm=None, relearn=True, new=True, b2=True, mlp=True, pc_eta=None, layers=("h1", "h2")):
    """Intact-net measures of a group: old-task accuracy, integrity, modal map, sum probe, probes at h1 / h2, relearning of the old
    tasks, A_new and B2 (W1 nets; F from TC.fresh_F for BP arms)."""
    allv = np.arange(nets.B); ev = eval_old(nets, tasks, allv)
    out = dict(acc=ev["mean"], acc_task=ev["acc"], integrity=ev["integrity"], label=ev["label"], sp=sum_probe_gen(nets, tasks, allv))
    out["probes"] = {L: probes(probe_feats(nets, tasks, allv, L), mlp=mlp) for L in layers}
    if relearn:
        rl = relearn_old(nets, tasks, allv, pc_eta=pc_eta); out["relearn"] = {str(b): v.mean(1) for b, v in rl.items()}
    if new:
        an = search_new(nets, ws_of, allv, pc_eta=pc_eta); out["A_new"] = {str(b): v.mean(1) for b, v in an.items()}
    if b2:
        A = b2_var(nets, ws_of, allv, pc_eta=pc_eta); out["B2_A"] = {str(ns): a.mean((1, 2)) for ns, a in A.items()}
        if arm is not None and arm != "pc_mup":
            F = fresh_F(arm, nets.W, ws_of); out["B2"] = {str(ns): b2_value(A[ns], F[ns]) for ns in A}
            out["F"] = {str(ns): np.asarray(F[ns]).mean((1, 2)) for ns in A}
    return out


def per_chain(items_per_net, n):
    """Per-variant op items (one per net) -> per-chain items (chains ordered net, task)."""
    return [it for it in items_per_net for _ in range(n)]


def jobs_search(nets, jobs, kind, tasks=None, ws_of=None, pc_eta=None, budgets=BUDGETS):
    """One batched best-of-budget search for several jobs (name -> per-chain op items, chains ordered (net, task), every net of the
    group): kind 'old' = old-task relearning (relearn_old), 'new' = the 4 new tasks (search_new). Chains with empty items run the
    intact net. Returns {name: {str(budget): [B] (mean over the tasks)}}."""
    names = list(jobs); B = nets.B; var = np.concatenate([np.arange(B)] * len(names))
    co = mk_ops(sum((list(jobs[k]) for k in names), []), nets.W)
    if kind == "old":
        r = relearn_old(nets, tasks, var, chain_ops=co, pc_eta=pc_eta, budgets=budgets)
    else:
        r = search_new(nets, ws_of, var, chain_ops=co, pc_eta=pc_eta, budgets=budgets)
    return {k: {str(b): r[b][i * B:(i + 1) * B].mean(1) for b in budgets} for i, k in enumerate(names)}


def t4b_forward(nets, tasks, layer, base, key, mlp=True):
    """Forward part of C-T4 (b) / C-T3 (d) / C-T7 (a) at layer ('h1' | 'h2') for every net of a group: class-mean erasure of task
    identity (W8; propagated forward), old-task accuracy, integrity and modal map, sum probe, linear and MLP probes on the erased
    held-out states; the damage-matched random erasure at the same layer (OD-C2 on the feedforward sum probe, r0 = the eraser's rank)
    with the same measures (mean over the NDRAW draws). Returns the result and, under '_jobs', the per-chain op items of the relearning
    searches (the erasure; the matched control with old task i on draw i mod NDRAW)."""
    B, W = nets.B, nets.W; allv = np.arange(B)
    Q, mu = fit_eraser(nets, tasks, layer); rank = int((Q.abs().sum(1) > 0).sum(-1).max())
    items = eraser_items(Q, mu, layer, allv); ops = mk_ops(items, W)
    ev = eval_old(nets, tasks, allv, ops=ops); sp = sum_probe_gen(nets, tasks, allv, ops=ops)
    out = dict(layer=layer, rank=rank, acc=ev["mean"], integrity=ev["integrity"], label=ev["label"], sp=sp,
               probes=probes(probe_feats(nets, tasks, allv, layer, ops=ops), mlp=mlp))
    target = float(np.mean(np.asarray(base["sp"]) - sp))
    dm = damage_match(nets, tasks, base["sp"], target, layer, mu, rank, f"{key}|dm|{layer}")
    vv = np.repeat(allv, NDRAW); od = mk_ops(dm_items(dm, mu, allv, layer), W)
    evd = eval_old(nets, tasks, vv, ops=od); spd = sum_probe_gen(nets, tasks, vv, ops=od)
    prd = probes(probe_feats(nets, tasks, vv, layer, ops=od), mlp=mlp)
    red = lambda a: np.asarray(a, float).reshape(B, NDRAW).mean(1)
    out["dm"] = dict(match=dm_public(dm), acc=red(evd["mean"]), integrity=red(evd["integrity"]), sp=red(spd), probes={k: red(v) for k, v in prd.items()})
    out["eraser"] = dict(Q=Q, mu=mu, dm_R=dm["R"], dm_lam=dm["lam"])     # removed before writing (C-T7 relaxation test)
    out["_jobs"] = dict(erase=per_chain(items, tasks.n), dm=dm_chain_items(dm, mu, allv, tasks.n, layer))
    return out


def t4b_layers(nets, tasks, base, key, layers=("h1", "h2"), pc_eta=None, mlp=True, relearn=True, relearn_dm=True, budgets=BUDGETS,
               extra_jobs=None):
    """C-T4 (b) / C-T3 (d) / C-T7 (a): t4b_forward at every layer, then one batched relearning search of the 12 old tasks with the
    erasure on (and with each matched control; PC nets: the eraser inside the energy, pc_eta) together with extra_jobs (name -> per-chain
    items, e.g. the intact net and the z1 operations of C-T4 (a)). Returns ({layer: result}, {extra job: relearn})."""
    res = {L: t4b_forward(nets, tasks, L, base, key, mlp=mlp) for L in layers}; extra = {}
    if relearn:
        jobs = {}
        for L in layers:
            jobs[f"{L}|erase"] = res[L]["_jobs"]["erase"]
            if relearn_dm:
                jobs[f"{L}|dm"] = res[L]["_jobs"]["dm"]
        jobs.update(extra_jobs or {})
        rr = jobs_search(nets, jobs, "old", tasks=tasks, pc_eta=pc_eta, budgets=budgets)
        for L in layers:
            res[L]["relearn"] = rr[f"{L}|erase"]
            if relearn_dm:
                res[L]["dm"]["relearn"] = rr[f"{L}|dm"]
        extra = {k: rr[k] for k in (extra_jobs or {})}
    for L in layers:
        res[L].pop("_jobs")
    return res, extra


def strip(d):
    """Drop tensors kept for in-process reuse (keys 'eraser', 'pred') before writing."""
    if isinstance(d, dict):
        return {k: strip(v) for k, v in d.items() if k not in ("eraser", "pred")}
    if isinstance(d, list):
        return [strip(v) for v in d]
    return d


# ================================================================================================================ C-T5 engine (shared)
def unsearch(nets, ws_of, arms, rounds, steps, budgets, remove_budget, key, pc_eta=None, pc_last=None, side=True, keep_dirs=True):
    """Un-searching by iterative removal of found directions (C-T5; C-T7 (b) with the PC protocol): for each net and new task, searches
    s_0 .. s_(rounds-1) of every arm in lockstep (one batched search per round over all chains of the width group). Every search: the
    column from zero on the 8 shots (Adam lr 1e-2), validation = split positions 8-11 every 10 steps, held-out 12-288, best-of-budget
    accuracies for budgets (W7); the direction removed after s_r is the unit column at s_r's best validation step within remove_budget.
    Arms: 'col' (column-space removal: the column is projected on the orthogonal complement of the removed set after every step),
    'z1' (the removed directions are blocked in z1 for every input, ref = mean z1 over the 12 x 289 pretraining inputs), 'rand' (one
    random direction per round, uniform in the complement of the removed set; column-space removal), 'pc' (the top task PCs of the
    centred task span in order, column space; searches s_0 .. s_pc_last), 'dmr' (per round a random z1 block whose sum-probe drop
    matches the z1 arm's mean drop after the same number of removals, OD-C2 with r0 = r; equal count if that drop is < 0.02; new task j
    uses draw j). Side effects (z1, dmr): old-task accuracy and sum probe under the search's block. Overlaps of the removed set with the
    11-dim task span and the 32-dim x / y span (||Q_span^T U||_F^2 / r). pc_eta: PC local update (arms col / rand only)."""
    B, W = nets.B, nets.W; arms = tuple(arms)
    chains = [(a, b, j) for a in arms for b in range(B) for j in range(NNEW)]; Cn = len(chains)
    cb = np.array([b for _, b, _ in chains]); cj = np.array([j for _, _, j in chains]); ca = [a for a, _, _ in chains]
    sh, ho, va = new_sets(ws_of, cb, cj)
    ref = TC.z1_ref(nets); Ut, _, _ = task_pcs(nets); Ut = Ut[:, :, :NPRE - 1]
    Ex, Ey = nets.x_cols().double(), nets.y_cols().double()
    Qxy = [orth_cols(torch.cat([Ex[b] - Ex[b].mean(1, keepdim=True), Ey[b] - Ey[b].mean(1, keepdim=True)], 1)).cpu() for b in range(B)]
    U = [torch.zeros(W, 0, dtype=F64) for _ in range(Cn)]
    pc_last = rounds - 1 if pc_last is None else pc_last
    tasks = Tasks.w1(ws_of)
    rec = dict(arms=list(arms), chains=[[a, int(b), int(j)] for a, b, j in chains], rounds=rounds, steps=steps, budgets=list(budgets),
               remove_budget=remove_budget, A={str(b_): np.full((Cn, rounds), np.nan) for b_ in budgets},
               ov_task=np.full((Cn, rounds), np.nan), ov_xy=np.full((Cn, rounds), np.nan), side_acc=np.full((Cn, rounds), np.nan),
               side_sp=np.full((Cn, rounds), np.nan), dm=[], dirs=None)
    base_sp = base_acc = None
    if side and ("z1" in arms or "dmr" in arms):
        base_acc = eval_old(nets, tasks, np.arange(B))["mean"]; base_sp = sum_probe_gen(nets, tasks, np.arange(B))
        rec.update(base_acc=base_acc, base_sp=base_sp)
    z1c = [i for i, a in enumerate(ca) if a == "z1"]; dmc = [i for i, a in enumerate(ca) if a == "dmr"]
    pc = None if pc_eta is None else dict(eta=np.asarray(pc_eta, np.float64)[cb])
    for r in range(rounds):
        t0 = time.time()
        act = [i for i in range(Cn) if not (ca[i] == "pc" and r > pc_last)]
        # removed sets of this round (r directions)
        for i in act:
            if ca[i] == "pc":
                U[i] = Ut[cb[i], :, :r].double()
            if r > 0 and ca[i] in ("col", "z1", "rand", "pc"):
                rec["ov_task"][i, r] = float((Ut[cb[i]].double().T @ U[i]).pow(2).sum() / r)
                rec["ov_xy"][i, r] = float((Qxy[cb[i]].T @ U[i]).pow(2).sum() / r)
        # side effects of the z1 arm (needed for the dmr match) and of the dmr arm
        dm = None
        if side and z1c:
            if r == 0:
                rec["side_acc"][z1c, 0] = base_acc[cb[z1c]]; rec["side_sp"][z1c, 0] = base_sp[cb[z1c]]
            else:
                it = [{"z1_U": U[i], "z1_ref": ref[cb[i]]} for i in z1c]; o = mk_ops(it, W)
                rec["side_acc"][z1c, r] = eval_old(nets, tasks, cb[z1c], ops=o)["mean"]; rec["side_sp"][z1c, r] = sum_probe_gen(nets, tasks, cb[z1c], ops=o)
        if dmc and r > 0:
            target = float(np.mean(base_sp[cb[z1c]] - rec["side_sp"][z1c, r]))
            dm = damage_match(nets, tasks, base_sp, target, "z1", ref, r, f"{key}|dmr")
            rec["dm"].append(dict(round=r, **dm_public(dm)))
            it = [{"z1_U": dm["R"][cj[i] % NDRAW], "z1_ref": ref[cb[i]], "z1_lam": dm["lam"]} for i in dmc]; o = mk_ops(it, W)
            rec["side_acc"][dmc, r] = eval_old(nets, tasks, cb[dmc], ops=o)["mean"]; rec["side_sp"][dmc, r] = sum_probe_gen(nets, tasks, cb[dmc], ops=o)
        elif dmc:
            rec["side_acc"][dmc, 0] = base_acc[cb[dmc]]; rec["side_sp"][dmc, 0] = base_sp[cb[dmc]]
        # the batched search of round r
        rem = torch.zeros(len(act), W, max(1, r), dtype=F32); items = []
        for k, i in enumerate(act):
            if ca[i] in ("col", "rand", "pc") and r > 0:
                rem[k, :, :U[i].shape[1]] = U[i].float()
            if ca[i] == "z1" and r > 0:
                items.append({"z1_U": U[i], "z1_ref": ref[cb[i]]})
            elif ca[i] == "dmr" and dm is not None:
                items.append({"z1_U": dm["R"][cj[i] % NDRAW], "z1_ref": ref[cb[i]], "z1_lam": dm["lam"]})
            else:
                items.append({})
        o = mk_ops(items, W)
        ai = np.array(act)
        res = search(nets, cb[ai], _sl(sh, ai, Cn), {"ho": _sl(ho, ai, Cn)}, val=_sl(va, ai, Cn), steps=steps, budgets=budgets,
                     remove=rem if r > 0 else None, ops=o, keep_cols=True, pc=None if pc is None else dict(eta=pc["eta"][ai]))
        for b_ in budgets:
            rec["A"][str(b_)][ai, r] = best_ho(res, b_)
        cols = torch.as_tensor(res["best"][remove_budget]["col"], dtype=F64)
        for k, i in enumerate(act):
            if ca[i] in ("col", "z1"):
                d = cols[k].clone()
            elif ca[i] == "rand":
                d = torch.randn(W, generator=torch.Generator().manual_seed(crc(f"{key}|rand|w{ws_of[cb[i]]}|j{cj[i]}|r{r}")), dtype=F64)
            else:
                continue
            for _ in range(2):
                d = d - U[i] @ (U[i].T @ d)
            nd = float(d.norm())
            if nd < 1e-12:
                d = torch.randn(W, generator=torch.Generator().manual_seed(crc(f"{key}|degenerate|{i}|{r}")), dtype=F64)
                d = d - U[i] @ (U[i].T @ d); nd = float(d.norm())
            U[i] = torch.cat([U[i], (d / nd)[:, None]], 1)
        print(f"    round {r}: {len(act)} chains, A{B_LONG if remove_budget == B_LONG else remove_budget} mean "
              f"{np.nanmean(rec['A'][str(remove_budget)][ai, r]):.3f} ({time.time() - t0:.0f}s)", flush=True)
    if keep_dirs:
        rec["dirs"] = [u.float() for u in U]               # removed directions in order (C-T7: the sets for the 3,000-step searches)
    return rec


RSTAR_NEVER = 25                                                 # C-T5: '> 24' counts as 25


def rstar(A, thr=UNLEARN_ACC, never=None):
    """First round r with accuracy <= thr per chain (A [C, rounds]); never = the value for '> last round' (default: the number of rounds,
    i.e. the next round; the C-T5 rule uses RSTAR_NEVER = 25 whatever the number of rounds)."""
    A = np.asarray(A, float); hit = A <= thr
    return np.where(hit.any(1), hit.argmax(1), A.shape[1] if never is None else never)


# ================================================================================================================ C-T0
def ring_flags(st):
    """C-T0 rule per net (critique 22i). NOTE: the frequency share and the leave-one-shift-out ring R2 are those of the task code (the
    LOO ring test exists only for the task columns, c1_structure (c)); a net 'has a ring code' if the task code's maximum single-frequency
    share > its permutation p99 and, for some K in 1..3, the LOO ring R2 > 0.3 and > its p95; 'Fourier-aligned' if in addition
    overlap(k) >= 0.8 for k = 1 or 2. The h1 / h2 centroid and answer-spectrum tests are reported per object (descriptive)."""
    a = st["a_taskcols"]; freq = bool(a["max"] > a["null_max_p99"])
    loo = {K: bool(v["real"] > 0.3 and v["real"] > v["null_p95"]) for K, v in st["c_loo_ring_R2"].items()}
    ring = freq and any(loo.values())
    fa = ring and any(st["overlap"].get(f"k{k}", 0.0) >= 0.8 for k in (1, 2))
    return dict(freq_taskcols=freq, freq_h1=bool(st["b_h1"]["max"] > st["b_h1"]["null_max_p99"]),
                freq_h2=bool(st["b_h2"]["max"] > st["b_h2"]["null_max_p99"]), loo=loo, ring=ring, fourier_aligned=fa,
                answer_h2_top1=bool(st["d_h2_answer"]["top1"] > st["d_h2_answer"]["null_top1_p99"]),
                answer_W3_top1=bool(st["d_W3"]["top1"] > st["d_W3"]["null_top1_p99"]))


def stage_t0(out):
    """C-T0: basis-free ring / Fourier structure tests (ref c1_structure.py via TC.structure_tests) on the one-hot nets of every cell."""
    res = dict(cells={}, nperm=dict(freq=NPERM[0], loo=NPERM[1], spec=NPERM[2]))
    for arm, W in CELLS:
        nets, rows, _ = onehot_nets(out, arm, W); per = []
        for i, ws in enumerate(SEEDS):
            st = TC.structure_tests(nets, i, ws, nperm=NPERM[0], nperm_loo=NPERM[1], nperm_spec=NPERM[2], ks=(1, 2, 3), overlap_ks=(1, 2))
            per.append(dict(ws=ws, tests=st, flags=ring_flags(st)))
        res["cells"][f"{arm}_W{W}"] = dict(arm=arm, W=W, nets=slim(rows), seeds=per)
        print(f"  t0 {arm} W{W}: ring {[p['flags']['ring'] for p in per]}", flush=True)
    res["rule_preview"] = rules_t0(res)
    return res


# ================================================================================================================ C-T1
def cut_bases(Ec, r, key):
    """Keep-r bases of a centred column set Ec [W, m] (C-T1): 'pca' = top-r principal axes; 'span' = NDRAW random r-dim subspaces of
    span(Ec); 'iso' = NDRAW random isotropic r-dim subspaces of R^W. Returns list of (basis, draw, Q [W, r] float64 CPU)."""
    U, S, _ = torch.linalg.svd(torch.as_tensor(Ec, dtype=F64).cpu(), full_matrices=False)
    d = int((S > 1e-9 * max(float(S[0]), 1e-300)).sum()); Us = U[:, :d]; W = Ec.shape[0]
    out = [("pca", 0, U[:, :r])]
    g = torch.Generator().manual_seed(crc(key + "|span"))
    for dr in range(NDRAW):
        R = torch.linalg.qr(torch.randn(d, r, generator=g, dtype=F64))[0]; out.append(("span", dr, Us @ R))
    Ri = rand_sub(W, r, NDRAW, key + "|iso").double().cpu()
    out += [("iso", dr, Ri[dr]) for dr in range(NDRAW)]
    return out


def _agg(entries, nseed, value_keys):
    """entries: dicts with idx, basis, draw, seed, <values> -> {idx: {basis: {key: [nseed] mean over draws}}}."""
    out = {}
    for e in entries:
        d = out.setdefault(str(e["idx"]), {}).setdefault(e["basis"], {})
        for k in value_keys:
            d.setdefault(k, [[] for _ in range(nseed)])[e["seed"]].append(e[k])
    for d1 in out.values():
        for d2 in d1.values():
            for k in value_keys:
                d2[k] = [float(np.mean(v)) if v else None for v in d2[k]]
    return out


def _b2_rows(A, F, var_seed):
    return {ns: b2_value(A[ns], np.asarray(F[ns])[var_seed]) for ns in A}


def t1_cell(out, arm, W):
    """C-T1 for one cell: keep-k cuts of the x / y columns (2k of 16 centred dims), the task columns (2K of 11), and the hidden
    projections of z1 and h2 on their top-r PCs (over the 12 x 289 inputs) against random r-dim subspaces; held-out pretraining
    accuracy, retained variance, B2 (8 and 4 shots) with the cut during the column search; the uncut nets' B2 (also G35's fallback)."""
    nets, rows, _ = onehot_nets(out, arm, W); B = nets.B; ws_of = list(SEEDS); tasks = Tasks.w1(ws_of); allv = np.arange(B)
    F = fresh_F(arm, W, ws_of); A0 = b2_var(nets, ws_of, allv)
    base = dict(acc=eval_old(nets, tasks, allv)["mean"], B2={str(ns): b2_value(A0[ns], F[ns]) for ns in A0},
                B2_A={str(ns): A0[ns].mean((1, 2)) for ns in A0}, F={str(ns): np.asarray(F[ns]).mean((1, 2)) for ns in A0}, level1=[r["level1"] for r in rows])
    fam = {}
    vkeys = ("acc", "retained", "B2_8", "B2_4")
    # (1) x / y columns (one-hot nets: W1[:, :, :17] are the x columns, W1[:, :, 17:34] the y columns)
    Ex, Ey = nets.x_cols().double().cpu(), nets.y_cols().double().cpu(); ent = []
    for k in T1_KS:
        var, info = [], []
        for b in range(B):
            mx, my = Ex[b].mean(1, keepdim=True), Ey[b].mean(1, keepdim=True); Exc, Eyc = Ex[b] - mx, Ey[b] - my
            bx = cut_bases(Exc, 2 * k, f"t1|{arm}|W{W}|ws{ws_of[b]}|x|k{k}"); by = cut_bases(Eyc, 2 * k, f"t1|{arm}|W{W}|ws{ws_of[b]}|y|k{k}")
            for (bs, dr, Qx), (_, _, Qy) in zip(bx, by):
                ret = float(((Qx.T @ Exc).pow(2).sum() + (Qy.T @ Eyc).pow(2).sum()) / (Exc.pow(2).sum() + Eyc.pow(2).sum()))
                var.append((b, mx + Qx @ (Qx.T @ Exc), my + Qy @ (Qy.T @ Eyc))); info.append(dict(idx=k, basis=bs, draw=dr, seed=b, retained=ret))
        vn = nets.subset([v[0] for v in var])
        with torch.no_grad():
            for i, (_, X_, Y_) in enumerate(var):
                vn.p["W1"][i, :, :P] = X_.to(vn.p["W1"]); vn.p["W1"][i, :, P:2 * P] = Y_.to(vn.p["W1"])
        wsv = [ws_of[v[0]] for v in var]; acc = eval_old(vn, Tasks.w1(wsv), np.arange(len(var)))["mean"]
        A = b2_var(vn, wsv, np.arange(len(var))); b2r = _b2_rows(A, F, [v[0] for v in var])
        for i, e in enumerate(info):
            ent.append(dict(e, acc=float(acc[i]), B2_8=float(b2r[8][i]), B2_4=float(b2r[4][i])))
    fam["xy"] = _agg(ent, B, vkeys)
    # (2) task columns: keep 2K of the 11 centred dims; B2 with the new column kept in span(Q, task-mean column) (NOTE)
    T = nets.task_cols()[:, :, :NPRE].double().cpu(); ent = []
    for K in T1_TK:
        cols, keep, vnet, info = [], [], [], []
        for b in range(B):
            mT = T[b].mean(1, keepdim=True); Tc = T[b] - mT
            for bs, dr, Q in cut_bases(Tc, 2 * K, f"t1|{arm}|W{W}|ws{ws_of[b]}|task|K{K}"):
                cols.append((mT + Q @ (Q.T @ Tc)).T); keep.append(orth_cols(torch.cat([Q, mT], 1))); vnet.append(b)
                info.append(dict(idx=K, basis=bs, draw=dr, seed=b, retained=float((Q.T @ Tc).pow(2).sum() / Tc.pow(2).sum())))
        R = max(k_.shape[1] for k_ in keep); kp = torch.zeros(len(keep), W, R, dtype=F32)
        for i, k_ in enumerate(keep):
            kp[i, :, :k_.shape[1]] = k_.float()
        acc = eval_old(nets, tasks, vnet, cols=torch.stack(cols).float())["mean"]
        A = b2_var(nets, ws_of, vnet, keep=kp); b2r = _b2_rows(A, F, vnet)
        for i, e in enumerate(info):
            ent.append(dict(e, acc=float(acc[i]), B2_8=float(b2r[8][i]), B2_4=float(b2r[4][i])))
    fam["task"] = _agg(ent, B, vkeys)
    # (3) hidden projections of z1 and h2 on their top-r PCs (the cut network: the complement of the kept subspace removed around
    # the mean for every input, during the search too) against NDRAW random isotropic r-dim subspaces
    vb, tp, x, y, lab, ky = _chains(tasks, allv, 0, P * P); cc = _chain_cols(nets, tasks, vb, tp)
    st = {"h2": _hidden(nets, vb, cc, x, y, None, "h2").reshape(B, -1, W).double().cpu()}
    Exd, Eyd, b1 = nets.x_cols().double().cpu(), nets.y_cols().double().cpu(), nets.p["b1"].double().cpu()
    st["z1"] = torch.stack([(Exd[b][:, x[b * NPRE:(b + 1) * NPRE].reshape(-1)] + Eyd[b][:, y[b * NPRE:(b + 1) * NPRE].reshape(-1)]
                             + b1[b][:, None] + cc[b * NPRE:(b + 1) * NPRE].double().cpu().repeat_interleave(P * P, 0).T).T for b in range(B)])
    for layer in ("z1", "h2"):
        ent = []; Hs = st[layer]; mu = Hs.mean(1); Hc = Hs - mu[:, None]
        Cvs = [Hc[b].T @ Hc[b] / Hc.shape[1] for b in range(B)]; ev = [torch.linalg.eigh(Cv) for Cv in Cvs]
        for r in T1_RS:
            if r > W:                    # r = W (empty complement) = the uncut net, part of the plan's grid (review unlearn_toy-3)
                continue
            items, vnet, info = [], [], []
            for b in range(B):
                lam_, V_ = ev[b]; V_ = V_.flip(1); lam_ = lam_.flip(0); tot = float(lam_.clamp(min=0).sum()); Cv = Cvs[b]
                bases = [("pca", 0, V_[:, :r])] + [("iso", dr, q.double().cpu()) for dr, q in enumerate(rand_sub(W, r, NDRAW, f"t1|{arm}|W{W}|ws{ws_of[b]}|{layer}|r{r}"))]
                for bs, dr, Q in bases:
                    ret = float(torch.trace(Q.T @ Cv @ Q) / tot) if tot > 0 else float("nan")
                    kU, kc, _ = LAYER_KEYS["z1" if layer == "z1" else "h2"]
                    items.append({kU: complement(Q.contiguous()), kc: mu[b]}); vnet.append(b); info.append(dict(idx=r, basis=bs, draw=dr, seed=b, retained=ret))
            o = mk_ops(items, W); acc = eval_old(nets, tasks, vnet, ops=o)["mean"]
            A = b2_var(nets, ws_of, vnet, ops=o); b2r = _b2_rows(A, F, vnet)
            for i, e in enumerate(info):
                ent.append(dict(e, acc=float(acc[i]), B2_8=float(b2r[8][i]), B2_4=float(b2r[4][i])))
        fam[layer] = _agg(ent, B, vkeys)
    return dict(arm=arm, W=W, nets=slim(rows), base=base, families=fam,
                grids=dict(xy=list(T1_KS), task=list(T1_TK), z1=[r for r in T1_RS if r <= W], h2=[r for r in T1_RS if r <= W]))


def stage_t1(out):
    """C-T1: superposition cap after training (keep k modes; no retraining)."""
    res = dict(cells={})
    for arm, W in CELLS:
        res["cells"][f"{arm}_W{W}"] = cached(out, f"t1_{arm}_W{W}", lambda arm=arm, W=W: t1_cell(out, arm, W))
    res["rule_preview"] = rules_t1(res)
    return res


# ================================================================================================================ C-T2
def ring_dist(a, b):
    return int(min((a - b) % P, (b - a) % P))


def stage_t2(out):
    """C-T2: imposed task codes (W5: ring, pring, rand; k in 2..4; W 128): level 1, steps to 0.95, held-out accuracy at the evaluation
    point, zero-shot accuracy on the 4 new shifts with t0 + A phi(c_new) and its modal shift (snapping), code-space search (q from 0,
    lr 1e-2 and 1e-1, the lr chosen per task by the 4 validation pairs, OD-C11), free column search (B2), cos(q, phi(c_new))."""
    res = dict(W=T2_W, codes=list(T2_CODES), ks=list(T2_KS), nets={})
    F = fresh_F("bp_mup", T2_W, SEEDS); lrs = (1e-2, 1e-1)
    for k in T2_KS:
        specs = [dict(kind="impose", W=T2_W, ws=ws, code=code, ck=k) for code in T2_CODES for ws in SEEDS]
        nets, rows, _ = nets_batch(out, f"impose_W{T2_W}_k{k}", specs)
        B = nets.B; ws_of = [s["ws"] for s in specs]; sidx = [SEEDS.index(w) for w in ws_of]
        vb = np.repeat(np.arange(B), NNEW); jj = np.tile(np.arange(NNEW), B)
        _, ho, _ = new_sets(ws_of, vb, jj, val=False)
        cnew = np.array([M.world(ws_of[b])[1][j] for b, j in zip(vb, jj)])
        T = nets.task_cols(); cols = T[torch.as_tensor(vb, device=DEV), :, torch.as_tensor(cnew, device=DEV)]
        pred = _pred(nets, vb, cols, ho[0], ho[1], None); zs = (pred == ho[2]).mean(1)
        e = (pred - ho[0] - ho[1]) % P; cnt = np.stack([np.bincount(r_, minlength=P) for r_ in e]); md = cnt.argmax(1)
        # code-space search, both learning rates in one batch
        vb2 = np.repeat(vb, len(lrs)); jj2 = np.repeat(jj, len(lrs)); lr2 = np.tile(np.array(lrs), len(vb))
        sh2, ho2, va2 = new_sets(ws_of, vb2, jj2)
        r = search(nets, vb2, sh2, {"ho": ho2, "val": va2}, eval_at=FT_EVAL, param=TC.coords_param(nets, vb2, "impose"), lr=lr2)
        Aho = r["acc"]["ho"].mean(1).reshape(-1, len(lrs)); Aval = r["acc"]["val"].mean(1).reshape(-1, len(lrs))
        pick = Aval.argmax(1); qq = r["q"].reshape(len(vb), len(lrs), -1)[np.arange(len(vb)), pick]
        cosq = []
        for c_, (b, j) in enumerate(zip(vb, jj)):
            ph = TC.code_table(specs[b]["code"], k, ws_of[b]).numpy()[cnew[c_]]; q_ = qq[c_]
            cosq.append(float(q_ @ ph / (np.linalg.norm(q_) * np.linalg.norm(ph) + 1e-12)))
        A = b2_var(nets, ws_of, np.arange(B)); b2r = _b2_rows(A, F, sidx)
        for b in range(B):
            sl = slice(b * NNEW, (b + 1) * NNEW); pre = M.world(ws_of[b])[0]
            res["nets"][rows[b]["name"]] = dict(
                code=specs[b]["code"], k=k, ws=ws_of[b], row=slim([rows[b]])[0], level1=rows[b]["level1"], first95=rows[b]["first95"],
                eval_point=rows[b]["eval_point"], heldout_eval=rows[b]["level_acc"], zero_shot=float(zs[sl].mean()), zero_shot_task=zs[sl],
                zs_modal=[dict(c_new=int(cnew[c_]), modal=int(md[c_]), share=float(cnt[c_].max() / e.shape[1]), ring_dist=ring_dist(int(md[c_]), int(cnew[c_])),
                               modal_is_trained=bool(int(md[c_]) in pre)) for c_ in range(sl.start, sl.stop)],
                code_search=float(Aho[sl, :][np.arange(NNEW), pick[sl]].mean()), code_search_lr={str(l_): float(Aho[sl, i].mean()) for i, l_ in enumerate(lrs)},
                code_lr_pick=[float(lrs[p_]) for p_ in pick[sl]], cos_q_phi=float(np.mean(cosq[sl.start:sl.stop])),
                free_A8=float(A[8][b].mean()), B2_8=float(b2r[8][b]), B2_4=float(b2r[4][b]))
        print(f"  t2 k={k}: level1 {[r_['level1'] for r_ in rows]} zero-shot {np.round(zs.reshape(B, -1).mean(1), 2).tolist()}", flush=True)
    res["rule_preview"] = rules_t2(res)
    return res


# ================================================================================================================ C-T3
def zs_eval(nets, ws_of, shifts_of, var_net, cols=None, ops=None):
    """Zero-shot accuracy of circle nets on shift values shifts_of[b] (each with its column; held-out = split positions 8-288 of
    splits[c mod 16], ref c3_angle). Returns [V, n_shifts] (equal counts per net)."""
    tasks = Tasks.w3(ws_of, shifts_of); var_net = np.asarray(var_net, np.int64); V, n = len(var_net), tasks.n
    vb, tp, x, y, lab, ky = _chains(tasks, var_net, 8, P * P)
    pred = _pred(nets, vb, _chain_cols(nets, tasks, vb, tp, cols), x, y, _expand(ops, V, n))
    return (pred == lab).mean(1).reshape(V, n)


def hatch_fn(specs):
    """C-T3 (c) snapshot: per net the mean zero-shot accuracy on its untrained shifts and the mean held-out accuracy on its trained
    shifts (every SNAP_EVERY updates)."""
    ws_of = [s["ws"] for s in specs]; trs = [TC.w3_shifts(s["ws"], s.get("arc"), s.get("spread")) for s in specs]
    ttr = Tasks.w3(ws_of, [t[0] for t in trs]); tun = Tasks.w3(ws_of, [t[1] for t in trs]); allv = np.arange(len(specs))

    def fn(step, nets):
        a = eval_old(nets, ttr, allv)["mean"]; vb, tp, x, y, lab, _ = _chains(tun, allv, 8, P * P)
        z = (_pred(nets, vb, _chain_cols(nets, tun, vb, tp), x, y, None) == lab).mean(1).reshape(len(allv), -1).mean(1)
        return np.stack([z, a], 1)
    return fn


def hatch_times(extra, rows):
    """Generator extraction time = first snapshot with zero-shot >= 0.5; level-1 time = first95 (and the first snapshot with trained
    held-out >= 0.95); snapshots after a net's evaluation point are ignored."""
    st = np.array([s for s, _, _ in extra["snapshots"]]); val = np.stack([np.asarray(v) for _, v, _ in extra["snapshots"]])
    out = []
    for b, r in enumerate(rows):
        ok = st <= r["checkpoint"]; zsb, trb = val[ok, b, 0], val[ok, b, 1]; s_ = st[ok]
        ext = int(s_[np.argmax(zsb >= 0.5)]) if np.any(zsb >= 0.5) else None
        l1 = int(s_[np.argmax(trb >= 0.95)]) if np.any(trb >= 0.95) else None
        out.append(dict(extract=ext, first95=r["first95"], first_snapshot_095=l1, steps=s_.tolist(), zero_shot=zsb.tolist(), trained=trb.tolist()))
    return out


def circle_tasks(nets_specs):
    ws_of = [s["ws"] for s in nets_specs]
    return ws_of, [TC.w3_shifts(s["ws"], s.get("arc"), s.get("spread")) for s in nets_specs]


def t3_ops(out, nets, specs):
    """C-T3 (d) in the full-coverage ring nets: class-mean erasure of task identity at h1 and h2 (C-T4b measures), removal of the
    c-plane span(Ac) from z1 (with NDRAW random 2-dim blocks), conjugation f(c) -> f(-c) (expected output x + y - c)."""
    ws_of, sh = circle_tasks(specs); tr = [s[0] for s in sh]; un = [s[1] for s in sh]; tasks = Tasks.w3(ws_of, tr)
    B, W = nets.B, nets.W; allv = np.arange(B)
    base = base_measures(nets, tasks, relearn=False, new=False, b2=False)
    base["zero_shot"] = zs_eval(nets, ws_of, un, allv).mean(1)
    tb, _ = t4b_layers(nets, tasks, base, f"t3|W{W}"); res = dict(base=base, t4b=strip(tb))
    dx = nets.dx(); Ac = nets.p["W1"][:, :, 2 * dx:].double().cpu(); ref = TC.z1_ref(nets, task_idx=np.array(tr))
    items = [{"z1_U": orth_cols(Ac[b]), "z1_ref": ref[b]} for b in allv]; dim = int(orth_cols(Ac[0]).shape[1])
    Rr = rand_sub(W, dim, NDRAW, f"t3|W{W}|cplane_rand"); vv = np.repeat(allv, NDRAW)
    itr = [{"z1_U": Rr[d], "z1_ref": ref[b]} for b in allv for d in range(NDRAW)]
    o, orr = mk_ops(items, W), mk_ops(itr, W)
    red = lambda a: np.asarray(a, float).reshape(B, NDRAW).mean(1)
    res["cplane"] = dict(dim=dim, acc=eval_old(nets, tasks, allv, ops=o)["mean"], zero_shot=zs_eval(nets, ws_of, un, allv, ops=o).mean(1),
                         sp=sum_probe_gen(nets, tasks, allv, ops=o),
                         rand=dict(acc=red(eval_old(nets, tasks, vv, ops=orr)["mean"]), zero_shot=red(zs_eval(nets, ws_of, un, vv, ops=orr).mean(1)),
                                   sp=red(sum_probe_gen(nets, tasks, vv, ops=orr))))
    T = nets.task_cols(); conj = torch.stack([T[b][:, [(-c) % P for c in tr[b]]].T for b in allv])
    ev = eval_old(nets, tasks, allv, cols=conj)
    res["conj"] = dict(acc=ev["mean"], acc_task=ev["acc"], modal=ev["modal"], share=ev["share"],
                       modal_is_minus_c=[[bool(int(m_) == (-c) % P) for m_, c in zip(ev["modal"][b], tr[b])] for b in allv],
                       sp=sum_probe_gen(nets, tasks, allv, cols=conj), shifts=tr)
    return res


def stage_t3(out):
    """C-T3: circle world (W3) positive control, word-length cap, hatching, erasure and inverse."""
    res = dict(W=T3_W, gate={}, word={}, ops=None)
    # (a) gate and (c) hatching: full coverage, arms ring / pring / c-onehot, K = 1 (and K = 2 as a check)
    for cK in T3_KS:
        for grp in (("ring", "pring"), ("c-onehot",)):
            specs = [dict(kind="circle", W=T3_W, ws=ws, circ=a, cK=cK) for a in grp for ws in SEEDS]
            nets, rows, extra = nets_batch(out, f"circle_W{T3_W}_K{cK}_{'_'.join(grp)}", specs, snapshot_every=SNAP_EVERY, snapshot=hatch_fn(specs))
            ws_of, sh = circle_tasks(specs); zs = zs_eval(nets, ws_of, [s[1] for s in sh], np.arange(nets.B)); ht = hatch_times(extra, rows)
            for b, s in enumerate(specs):
                res["gate"].setdefault(f"K{cK}", {}).setdefault(s["circ"], []).append(dict(
                    ws=s["ws"], row=slim([rows[b]])[0], zero_shot=float(zs[b].mean()), zero_shot_shift=dict(zip(map(str, sh[b][1]), zs[b].tolist())), hatch=ht[b]))
            if cK == 1 and grp[0] == "ring":
                ring_nets, ring_specs = nets.subset(list(range(len(SEEDS)))), specs[:len(SEEDS)]
    # (b) word-length cap: arcs of L consecutive shifts against L shifts spread at random (pre[:L]), ring arm, K = 1
    for L in T3_LS:
        specs = [dict(kind="circle", W=T3_W, ws=ws, circ="ring", cK=1, arc=L) for ws in SEEDS] + \
                [dict(kind="circle", W=T3_W, ws=ws, circ="ring", cK=1, spread=L) for ws in SEEDS]
        nets, rows, _ = nets_batch(out, f"circle_W{T3_W}_L{L}", specs)
        ws_of, sh = circle_tasks(specs); zs = zs_eval(nets, ws_of, [s[1] for s in sh], np.arange(nets.B))
        for b, s in enumerate(specs):
            trn, unt = sh[b]; dist = [min(ring_dist(c, t) for t in trn) for c in unt]
            byj = {str(j): float(np.mean([z for z, d in zip(zs[b], dist) if d == j])) for j in sorted(set(dist))}
            res["word"].setdefault(f"L{L}", {}).setdefault("arc" if s.get("arc") else "spread", []).append(dict(
                ws=s["ws"], row=slim([rows[b]])[0], trained=trn, untrained=unt, distance=dist, zero_shot_shift=zs[b].tolist(),
                zero_shot=float(zs[b].mean()), by_j=byj))
    # (d) operations in the full-coverage ring nets (K = 1)
    res["ops"] = cached(out, f"t3_ops_W{T3_W}", lambda: t3_ops(out, ring_nets, ring_specs))
    res["rule_preview"] = rules_t3(res)
    return res


# ================================================================================================================ C-T4
def _fwd(nets, tasks, var, items=None, cols=None):
    o = None if items is None else mk_ops(items, nets.W)
    ev = eval_old(nets, tasks, var, cols=cols, ops=o)
    return dict(acc=ev["mean"], integrity=ev["integrity"], label=ev["label"], modal=ev["modal"], share=ev["share"],
                sp=sum_probe_gen(nets, tasks, var, cols=cols, ops=o))


def _draw_mean(d, B):
    """Per-net means over the NDRAW draws of a control evaluated on variants ordered (net, draw)."""
    return {k: (np.asarray(v, float).reshape(B, NDRAW).mean(1) if k in ("acc", "integrity", "sp") else v) for k, v in d.items()
            if k not in ("modal", "share", "label")} | dict(acc_draws=np.asarray(d["acc"], float).reshape(B, NDRAW))


def push_bisect(nets, tasks, u1, target, scale, tol=0.02, iters=40, expand=20, rtol=1e-3, acc_fn=None):
    """OD-C9 push size per net (review unlearn_toy-2; NOTE): beta = the smallest push z1 += -beta u1 at which the old-task accuracy
    reaches the target (acc <= target: 'crossing'); for a net whose accuracy never reaches it within the doublings, the smallest beta at
    which it comes within tol of it (acc <= target + tol: 'band'); 'not reached' otherwise. Found by doubling from scale and bisection
    between the last two doubling points (from 0 if scale already reaches it) to a relative width rtol; the accuracy is assumed to fall
    with beta. matched = the accuracy at beta lies within tol of the target (OD-C9). Before the review the first evaluated beta inside the
    band was returned, which could be up to twice the smallest matching dose. acc_fn(beta [B]) -> accuracy [B] replaces the nets' old-task
    accuracy under the push (selftest). Returns dict(beta, acc, target, matched, rule)."""
    tgt = np.asarray(target, float); B = len(tgt)
    if acc_fn is None:
        allv = np.arange(nets.B)

        def acc_fn(beta):
            return eval_old(nets, tasks, allv, ops=mk_ops([{"z1_shift": -float(beta[b]) * u1[b]} for b in allv], nets.W))["mean"]
    hi = np.maximum(np.asarray(scale, float), 1e-3).copy(); lo = np.zeros(B); a_hi = np.asarray(acc_fn(hi), float); hist = [(hi.copy(), a_hi.copy())]
    for _ in range(expand):
        need = a_hi > tgt
        if not need.any():
            break
        lo = np.where(need, hi, lo); hi = np.where(need, 2 * hi, hi); a_hi = np.where(need, np.asarray(acc_fn(hi), float), a_hi)
        hist.append((hi.copy(), a_hi.copy()))
    cross = a_hi <= tgt; thr = np.where(cross, tgt, tgt + tol); rule = np.where(cross, "crossing", "band").astype(object); active = np.ones(B, bool)
    for b in np.nonzero(~cross)[0]:                     # band fallback: first doubling point within tol, bracketed by the point before it
        k = next((i for i, (h, a) in enumerate(hist) if a[b] <= thr[b]), None)
        if k is None:
            rule[b] = "not reached"; active[b] = False; continue
        hi[b], a_hi[b] = hist[k][0][b], hist[k][1][b]; lo[b] = hist[k - 1][0][b] if k > 0 else 0.0
    for _ in range(iters):
        go = active & (hi - lo > rtol * hi)
        if not go.any():
            break
        mid = 0.5 * (lo + hi); am = np.asarray(acc_fn(np.where(go, mid, hi)), float); ok = am <= thr
        hi = np.where(go & ok, mid, hi); a_hi = np.where(go & ok, am, a_hi); lo = np.where(go & ~ok, mid, lo)
    return dict(beta=hi, acc=a_hi, target=tgt, matched=np.abs(a_hi - tgt) <= tol, rule=rule.tolist())


def t4_cell(out, arm, W):
    """C-T4 (a) and (b) for one cell of one-hot nets: z1 operations with their controls and measures; T4b at h1 and h2."""
    nets, rows, _ = onehot_nets(out, arm, W); B = nets.B; ws_of = list(SEEDS); tasks = Tasks.w1(ws_of); allv = np.arange(B)
    key = f"t4|{arm}|W{W}"; F = fresh_F(arm, W, ws_of)
    base = base_measures(nets, tasks, ws_of, arm, relearn=False, new=False, b2=False)       # searches: batched below
    Ut, _, mT, flips = task_pcs(nets, return_flips=True); U11 = Ut[:, :, :NPRE - 1]; ref = TC.z1_ref(nets)
    Ex = nets.x_cols().double().cpu(); Qx = [orth_cols(Ex[b] - Ex[b].mean(1, keepdim=True)) for b in allv]
    T = nets.task_cols()[:, :, :NPRE]; Tcpu = T.double().cpu(); u1 = Ut[:, :, 0]
    edit_cols = T.mean(-1, keepdim=True).expand(-1, -1, NPRE).transpose(1, 2)
    blocks = {"block11": ([U11[b] for b in allv], 1.0), "blk1": ([Ut[b, :, :1] for b in allv], 1.0), "blk2": ([Ut[b, :, :2] for b in allv], 1.0),
              "blk4": ([Ut[b, :, :4] for b in allv], 1.0), "blk6": ([Ut[b, :, :6] for b in allv], 1.0), "reflect": ([U11[b] for b in allv], 2.0),
              "xspan": (Qx, 1.0)}
    dims = {"edit": NPRE - 1, "block11": NPRE - 1, "blk1": 1, "blk2": 2, "blk4": 4, "blk6": 6, "reflect": NPRE - 1, "xspan": int(Qx[0].shape[1]),
            "push_half": 1, "push_full": 1}
    items = {nm: [{"z1_U": Us[b], "z1_ref": ref[b], "z1_lam": lam} for b in allv] for nm, (Us, lam) in blocks.items()}
    ops = {"edit": _fwd(nets, tasks, allv, cols=edit_cols)}
    for nm, it in items.items():
        ops[nm] = _fwd(nets, tasks, allv, items=it)
    # push along the top task PC at half and full damage of the 11-dim block (OD-C9; bisection per net)
    scale = np.array([float((u1[b] @ (Tcpu[b] - Tcpu[b].mean(1, keepdim=True))).abs().max()) for b in allv])
    push = {}
    for nm, frac in (("push_half", 0.5), ("push_full", 1.0)):
        tgt = base["acc"] - frac * (base["acc"] - ops["block11"]["acc"]); pb = push_bisect(nets, tasks, u1, tgt, scale)
        push[nm] = pb; items[nm] = [{"z1_shift": -float(pb["beta"][b]) * u1[b]} for b in allv]; ops[nm] = _fwd(nets, tasks, allv, items=items[nm])
    names = list(dims)
    # controls: equal-dimension random isotropic subspaces (NDRAW draws), within-span random for the PC blocks, damage-matched (OD-C2)
    vv = np.repeat(allv, NDRAW); iso, span, dmatch = {}, {}, {}
    for nm in names:
        r0 = dims[nm]; R = rand_sub(W, r0, NDRAW, f"{key}|iso|{nm}")
        if nm == "edit":
            cols = torch.stack([(Tcpu[b] - R[d].double().cpu() @ (R[d].double().cpu().T @ (Tcpu[b] - Tcpu[b].mean(1, keepdim=True)))).T for b in allv for d in range(NDRAW)])
            iso[nm] = _draw_mean(_fwd(nets, tasks, vv, cols=cols.float()), B)
        elif nm.startswith("push"):
            it = [{"z1_shift": -float(push[nm]["beta"][b]) * R[d][:, 0]} for b in allv for d in range(NDRAW)]
            iso[nm] = _draw_mean(_fwd(nets, tasks, vv, items=it), B)
        else:
            lam = blocks[nm][1]; it = [{"z1_U": R[d], "z1_ref": ref[b], "z1_lam": lam} for b in allv for d in range(NDRAW)]
            iso[nm] = _draw_mean(_fwd(nets, tasks, vv, items=it), B)
        if nm.startswith("blk"):
            k = dims[nm]; it = []
            for b in allv:
                Rs = TC.random_in_span(U11[b].float(), k, NDRAW, crc(f"{key}|span|{nm}|ws{ws_of[b]}"))
                it += [{"z1_U": Rs[d], "z1_ref": ref[b]} for d in range(NDRAW)]
            span[nm] = _draw_mean(_fwd(nets, tasks, vv, items=it), B)
        tgt = float(np.mean(base["sp"] - ops[nm]["sp"]))
        dmatch[nm] = damage_match(nets, tasks, base["sp"], tgt, "z1", ref, r0, f"{key}|dm|{nm}")
    dmf = {nm: _draw_mean(_fwd(nets, tasks, vv, items=dm_items(dmatch[nm], ref, allv, "z1")), B) for nm in names}
    # relearning of the old tasks (one batched search: intact net, every operation, every matched control, and T4b at h1 / h2 with
    # theirs), A_new (one batched search) and B2 with the operation on; the edited columns are replaced by the searched column, so
    # 'edit' equals the intact net in every search (base values reused)
    tn = [nm for nm in names if nm != "edit"]
    jobs = {"base": per_chain([{}] * B, NPRE), **{f"op|{nm}": per_chain(items[nm], NPRE) for nm in tn},
            **{f"dm|{nm}": dm_chain_items(dmatch[nm], ref, allv, NPRE, "z1") for nm in names}}
    t4b_res, rr = t4b_layers(nets, tasks, base, key, extra_jobs=jobs)
    nj = {"base": per_chain([{}] * B, NNEW), **{f"op|{nm}": per_chain(items[nm], NNEW) for nm in tn},
          **{f"dm|{nm}": dm_chain_items(dmatch[nm], ref, allv, NNEW, "z1") for nm in names}}
    an = jobs_search(nets, nj, "new", ws_of=ws_of)
    var = np.concatenate([allv] * (1 + len(tn))); A = b2_var(nets, ws_of, var, ops=mk_ops([{}] * B + sum((items[nm] for nm in tn), []), W))
    base.update(relearn=rr["base"], A_new=an["base"], B2={str(ns): b2_value(a[:B], F[ns]) for ns, a in A.items()},
                B2_A={str(ns): a[:B].mean((1, 2)) for ns, a in A.items()}, F={str(ns): np.asarray(F[ns]).mean((1, 2)) for ns in A})
    out_ops = {}
    for nm in names:
        d = {k: v for k, v in ops[nm].items() if k != "share"}
        if nm == "edit":
            d.update(relearn=base["relearn"], A_new=base["A_new"], B2=base["B2"], B2_A=base["B2_A"])
        else:
            sl = slice((1 + tn.index(nm)) * B, (2 + tn.index(nm)) * B)
            d.update(relearn=rr[f"op|{nm}"], A_new=an[f"op|{nm}"], B2={str(ns): b2_value(a[sl], F[ns]) for ns, a in A.items()},
                     B2_A={str(ns): a[sl].mean((1, 2)) for ns, a in A.items()})
        dmd = dict(dmf[nm], match=dm_public(dmatch[nm]), relearn=rr[f"dm|{nm}"], A_new=an[f"dm|{nm}"])
        out_ops[nm] = dict(dim=dims[nm], op=d, iso=iso[nm], span=span.get(nm), dm=dmd, push=push.get(nm),
                           label="out-of-distribution damage (5.1)" if nm == "reflect" else ("positive control for generic damage" if nm == "xspan" else None))
    t4b_res = strip(t4b_res)
    return dict(arm=arm, W=W, nets=slim(rows), base=base, ops=out_ops, t4b=t4b_res,
                pc_orientation=dict(rule=AXIS_SIGN_RULE, u1_flipped=flips[:, 0].tolist()))


def t4_w4(out, W):
    """C-T4 (c): two-family world (W4): block of the 11-dim add-task span at z1 and class-mean erasure of add-task identity at h1 / h2;
    forget (add accuracy) against retain (sub accuracy), sum probe (both families; NOTE: sub tasks probe x - y), relearning of the
    12 add tasks with the operation on, probes (add-task identity, pooled add answer) at h1 and h2; damage-matched random operations."""
    specs = [dict(W=W, ws=ws, data="w4") for ws in SEEDS]; nets, rows, _ = nets_batch(out, f"w4_W{W}", specs)
    ws_of = list(SEEDS); B = nets.B; allv = np.arange(B); key = f"t4|w4|W{W}"
    add, sub, both = Tasks.w4(ws_of, "add"), Tasks.w4(ws_of, "sub"), Tasks.w4(ws_of)

    def meas(var, o=None, prb=True):
        r = dict(add=eval_old(nets, add, var, ops=o)["mean"], sub=eval_old(nets, sub, var, ops=o)["mean"], sp=sum_probe_gen(nets, both, var, ops=o))
        if prb:
            r["probes"] = {L: probes(probe_feats(nets, add, var, L, ops=o)) for L in ("h1", "h2")}
        return r
    base = meas(allv)
    Ta = nets.task_cols()[:, :, add.t[0]].double().cpu(); ref = TC.z1_ref(nets, task_idx=list(range(2 * NPRE)))
    Ua = [orth_cols(Ta[b] - Ta[b].mean(1, keepdim=True)) for b in allv]
    opdef = {"block_add": ("z1", [{"z1_U": Ua[b], "z1_ref": ref[b]} for b in allv], ref, int(Ua[0].shape[1]))}
    for L in ("h1", "h2"):
        Q, mu = fit_eraser(nets, add, L); rk = int((Q.abs().sum(1) > 0).sum(-1).max())
        opdef[f"erase_{L}"] = (L, eraser_items(Q, mu, L, allv), mu, rk)
    res = dict(W=W, nets=slim(rows), base=base, ops={})
    vv = np.repeat(allv, NDRAW); jobs = {"base": per_chain([{}] * B, add.n)}
    for nm, (layer, it, center, r0) in opdef.items():
        o = mk_ops(it, W); m_ = meas(allv, o)
        dm = damage_match(nets, both, base["sp"], float(np.mean(base["sp"] - m_["sp"])), layer, center, r0, f"{key}|dm|{nm}")
        md = meas(vv, mk_ops(dm_items(dm, center, allv, layer), W), prb=False)
        md = {k: np.asarray(v, float).reshape(B, NDRAW).mean(1) for k, v in md.items()}
        jobs[f"op|{nm}"] = per_chain(it, add.n); jobs[f"dm|{nm}"] = dm_chain_items(dm, center, allv, add.n, layer)
        res["ops"][nm] = dict(layer=layer, dim=r0, op=m_, dm=dict(md, match=dm_public(dm)))
    rr = jobs_search(nets, jobs, "old", tasks=add)            # relearning of the 12 add tasks: one batched search
    base["relearn_add"] = rr["base"]
    for nm in opdef:
        res["ops"][nm]["op"]["relearn_add"] = rr[f"op|{nm}"]; res["ops"][nm]["dm"]["relearn_add"] = rr[f"dm|{nm}"]
    return res


def stage_t4(out):
    """C-T4: erasure and counter-steering (one-hot cells, T4b, two-family world)."""
    res = dict(cells={}, w4={})
    for arm, W in CELLS:
        res["cells"][f"{arm}_W{W}"] = cached(out, f"t4_{arm}_W{W}", lambda arm=arm, W=W: t4_cell(out, arm, W))
    for W in W4_WIDTHS:
        res["w4"][f"W{W}"] = cached(out, f"t4_w4_W{W}", lambda W=W: t4_w4(out, W))
    res["rule_preview"] = rules_t4(res)
    return res


# ================================================================================================================ C-T5
def t5_width(out, W):
    nets, rows, _ = onehot_nets(out, "bp_mup", W)
    rec = unsearch(nets, list(SEEDS), ("col", "z1", "rand", "pc", "dmr"), T5_ROUNDS, B_LONG, BUDGETS, B_LONG, f"t5|W{W}",
                   pc_last=T5_PC_LAST, keep_dirs=False)
    rec.pop("dirs", None)
    return dict(W=W, nets=slim(rows), seeds=list(SEEDS), pc_last=T5_PC_LAST, **rec)


def stage_t5(out):
    """C-T5 (session r2_t4): un-searching by iterative removal of found directions, BP primary protocol (3,000-step searches)."""
    res = dict(widths={})
    for W in T5_WIDTHS:
        res["widths"][f"W{W}"] = cached(out, f"t5_W{W}", lambda W=W: t5_width(out, W))
    res["rule_preview"] = rules_t5(res)
    return res


# ================================================================================================================ C-T6
def frob_from(nets, init, keys):
    """Frobenius distance from init of the listed parameter blocks: keys of ('Ex' = x / y input columns W1[:, :, :34], 'W2')."""
    out = {}
    for k in keys:
        a = nets.p["W1"][:, :, :2 * P] if k == "Ex" else nets.p[k]; b = init.p["W1"][:, :, :2 * P] if k == "Ex" else init.p[k]
        out[k] = (a.double() - b.double()).flatten(1).norm(dim=1).cpu().numpy()
    return out


def stage_t6(out):
    """C-T6: trajectory primitives (SVD of the centred task-code snapshots every 25 updates against 50 shuffled-increment nulls) and
    the causal cap (retraining with the centred task-column updates restricted to the top-L modes of the same seed's free run)."""
    ws_of = list(SEEDS); specs = [dict(W=T6_W, ws=ws, tag="traj") for ws in ws_of]
    nets, rows, extra = nets_batch(out, f"traj_W{T6_W}", specs, snapshot_every=SNAP_EVERY, snapshot=TC.snap_task_code)
    st = np.array([s for s, _, _ in extra["snapshots"]]); V = np.stack([np.asarray(v, np.float64) for _, v, _ in extra["snapshots"]])
    B = nets.B; init = TC.Nets.from_specs(specs); free = dict(rows=slim(rows), modes=[], null=[], n_snap=[])
    bases = {L: np.zeros((B, T6_W * NPRE, L)) for L in T6_LS}
    for b in range(B):
        X = V[st <= rows[b]["checkpoint"], b]; modes, v5 = TC.trajectory_modes(X, T6_MODES)
        nulls = TC.shuffled_increment_nulls(X, T6_NULLS, np.random.default_rng(ws_of[b]), T6_MODES)
        nv = np.array([[n_[i]["var"] for i in range(T6_MODES)] for n_ in nulls]); nl = np.array([[n_[i]["lissajous_r"] for i in range(T6_MODES)] for n_ in nulls])
        free["modes"].append(dict(modes=modes, top5_var=v5)); free["n_snap"].append(int(len(X)))
        free["null"].append(dict(var_p99=np.percentile(nv, 99, axis=0), lis_p5=np.percentile(nl, 5, axis=0), var_mean=nv.mean(0), lis_mean=nl.mean(0)))
        for L in T6_LS:
            bases[L][b] = TC.trajectory_basis(X, L)
    free["dist"] = frob_from(nets, init, ("Ex", "W2")); bb = TC.b2(nets, ws_of)
    free["B2"] = {str(ns): v["B2"] for ns, v in bb.items()}
    caps = {}
    for L in T6_LS:
        sp_ = [dict(W=T6_W, ws=ws, tag=f"cap{L}") for ws in ws_of]
        cn, crows, _ = nets_batch(out, f"traj_W{T6_W}_cap{L}", sp_, task_basis=bases[L])
        d = frob_from(cn, init, ("Ex", "W2")); cb = TC.b2(cn, ws_of)
        caps[str(L)] = dict(rows=slim(crows), level1=[r["level1"] for r in crows], first95=[r["first95"] for r in crows],
                            dist=d, ratio_Ex=d["Ex"] / free["dist"]["Ex"], ratio_W2=d["W2"] / free["dist"]["W2"], B2={str(ns): v["B2"] for ns, v in cb.items()})
    res = dict(W=T6_W, seeds=ws_of, snap_every=SNAP_EVERY, n_nulls=T6_NULLS, free=free, caps=caps)
    res["rule_preview"] = rules_t6(res)
    return res


# ================================================================================================================ rules (5.1, frozen)
# Each rules_<stage>(res) reads only the stage JSON (lists after the round trip) and returns the frozen rule's preview; r2_unlearn_
# verdict.py (r2_ana) governs and may import these functions. Missing intervals (fewer than 2 seeds) fail a condition. Every stored
# rule input is summ(): the t interval that governs, the per-seed values and, next to it, the paired seed bootstrap (2.6; review
# unlearn_toy-5: 10,000 draws from default_rng(0), the same resampled seeds for every quantity of equal n, deciding nothing).
def _mean(v):
    return tci(v)["mean"]


def _lo(v, level=0.95):
    return tci(v, level)["lo"]


def _hi(v, level=0.95):
    return tci(v, level)["hi"]


def _le(a, b):
    return a is not None and b is not None and a <= b


def _ge(a, b):
    return a is not None and b is not None and a >= b


def _a(x):
    return _arr(x)


def stage_budgets(res):
    """(short, long) best-of-budget keys as stored with the stage (meta.budgets; the module's BUDGETS for an in-process preview), so
    the rules evaluate QUICK and full outputs alike in any process."""
    b = (res.get("meta") or {}).get("budgets") or BUDGETS
    return str(int(b[0])), str(int(b[1]))


def ctrl_status(acc, base_mean):
    """A control 'does not unlearn' if the t(4) lower bound of its old-task accuracy is >= 0.9 x base (5.1, critique v2-5); NOTE: it
    'unlearns' if the upper bound is <= 0.11; otherwise 'neither'."""
    if _ge(_lo(acc), 0.9 * base_mean):
        return "does not unlearn"
    if _le(_hi(acc), UNLEARN_ACC):
        return "unlearns"
    return "neither"


def unlearns(acc, dm_acc, base_mean):
    """5.1 toy unlearning: t(4) upper bound of old-task accuracy <= 0.11 while the damage-matched random control keeps a t(4) lower
    bound >= 0.9 x base."""
    return _le(_hi(acc), UNLEARN_ACC) and _ge(_lo(dm_acc), 0.9 * base_mean)


def rules_t0(res):
    out = {}
    for name, c in res["cells"].items():
        fl = [s["flags"] for s in c["seeds"]]; n = len(fl); q = quorum(n)
        nr = sum(bool(f["ring"]) for f in fl); nf = sum(bool(f["fourier_aligned"]) for f in fl)
        out[name] = dict(n_seeds=n, quorum=q, ring_seeds=nr, fourier_aligned_seeds=nf, has_ring_code=nr >= q, fourier_aligned=nf >= q,
                         freq_share_seeds={o: sum(bool(f[f"freq_{o}"]) for f in fl) for o in ("taskcols", "h1", "h2")})
    return out


def rules_t1(res):
    """k* = smallest grid index with held-out accuracy >= 0.9 x base, per seed and basis (random draws averaged); 'critical length' if
    k*(PCA) <= k*(random within span) - 2 in >= 4 of 5 seeds and in >= 4 of 5 seeds some single step raises the PCA accuracy by >= 0.5;
    'graceful' if in >= 4 of 5 seeds every step raises it by < 0.3; else 'intermediate'. NOTE: per family (x / y columns: k; task
    columns: K; hidden z1 / h2: the r grid, random = isotropic, there being no within-span random); k in grid-index units."""
    out = {}
    for name, c in res["cells"].items():
        base = _a(c["base"]["acc"]); n = len(base); q = quorum(n); o = {}
        for f, grid in c["grids"].items():
            if not grid:
                continue
            fam = c["families"][f]; ctrl = "span" if f in ("xy", "task") else "iso"
            acc = {bs: np.array([[fam[str(k)][bs]["acc"][s] for k in grid] for s in range(n)], float) for bs in ("pca", ctrl)}

            def kstar(a):
                return [int(np.argmax(a[s] >= 0.9 * base[s])) if np.any(a[s] >= 0.9 * base[s]) else len(grid) for s in range(n)]
            kp, kr = kstar(acc["pca"]), kstar(acc[ctrl]); steps = np.diff(acc["pca"], axis=1)
            n_gap = sum(kp[s] <= kr[s] - 2 for s in range(n)); n_jump = int((steps >= 0.5).any(1).sum()) if steps.size else 0
            n_grace = int((steps < 0.3).all(1).sum()) if steps.size else n
            lab = "critical length" if (n_gap >= q and n_jump >= q) else ("graceful" if n_grace >= q else "intermediate")
            o[f] = dict(grid=grid, control=ctrl, kstar_pca=[grid[k] if k < len(grid) else None for k in kp],
                        kstar_ctrl=[grid[k] if k < len(grid) else None for k in kr], n_gap2=n_gap, n_jump05=n_jump, n_graceful=n_grace, label=lab)
        o["B2_uncut"] = {ns: summ(v) for ns, v in c["base"]["B2"].items()}
        out[name] = o
    return out


def rules_t2(res):
    by = {}
    for d in res["nets"].values():
        by.setdefault((d["code"], int(d["k"])), []).append(d)
    for v in by.values():
        v.sort(key=lambda d: d["ws"])
    ks = sorted({k for _, k in by}); codes = sorted({c for c, _ in by}); out = dict(cells={}, k_min={}, effects={})
    for (code, k), v in sorted(by.items()):
        zs = [d["zero_shot"] for d in v]; modal_tr = [m["modal_is_trained"] for d in v for m in d["zs_modal"]]
        out["cells"][f"{code}_k{k}"] = dict(level1_seeds=sum(bool(d["level1"]) for d in v), zero_shot=summ(zs), code_search=summ([d["code_search"] for d in v]),
                                           heldout_eval=summ([d["heldout_eval"] for d in v]), free_A8=summ([d["free_A8"] for d in v]), B2_8=summ([d["B2_8"] for d in v]),
                                           snap=bool(np.mean(zs) < 0.03 and np.mean(modal_tr) >= 0.75), modal_trained_share=float(np.mean(modal_tr)))
    for code in codes:
        km = [k for k in ks if (code, k) in by and sum(bool(d["level1"]) for d in by[(code, k)]) >= quorum(len(by[(code, k)]))]
        out["k_min"][code] = km[0] if km else "none"
    lin_zs = out["k_min"].get("ring") not in (None, "none") and any(
        np.mean([d["zero_shot"] for d in by[("ring", k)]]) >= 0.5 for k in ks if k >= out["k_min"]["ring"] and ("ring", k) in by)
    lin_cs, patt = False, False; bonf = 1 - 0.05 / max(1, len(ks))
    for k in ks:
        if not all((c_, k) in by for c_ in ("ring", "pring", "rand")):
            continue
        g = lambda c_, key: _a([d[key] for d in by[(c_, k)]])
        dcs = g("ring", "code_search") - g("rand", "code_search"); al = g("ring", "heldout_eval") - g("pring", "heldout_eval")
        ge_ = g("pring", "heldout_eval") - g("rand", "heldout_eval")
        out["effects"][f"k{k}"] = dict(code_search_ring_minus_rand=summ(dcs), alignment_ring_minus_pring=summ(al), geometry_pring_minus_rand=summ(ge_),
                                       alignment_bonferroni=tci(al, bonf))
        lin_cs |= _ge(_mean(dcs), 0.15) and _ge(_lo(dcs), 1e-300)
        patt |= _ge(tci(al, bonf)["lo"], 1e-300)
    out["ring_linearises"] = bool(lin_zs or lin_cs); out["ring_linearises_by"] = dict(zero_shot=bool(lin_zs), code_search=bool(lin_cs))
    out["pattern_one_hot_world"] = bool(patt); out["bonferroni_level"] = bonf
    return out


def jstar(byj):
    """j*(L): NOTE the contiguous reading = the largest j such that every distance 1..j has zero-shot >= 0.5 (0 if distance 1 fails);
    'literal' = the largest j with zero-shot >= 0.5."""
    js = sorted(int(j) for j in byj); cont = 0
    for j in js:
        if byj[str(j)] is not None and byj[str(j)] >= 0.5 and j == cont + 1:
            cont = j
        else:
            break
    lit = max([j for j in js if byj[str(j)] is not None and byj[str(j)] >= 0.5], default=0)
    return cont, lit, (max(js) if js else 0)


def t4b_rule(t, base_acc_mean, chance):
    """C-T4 (b) per layer: 'causal' = accuracy upper bound <= 0.11 with the damage-matched random control >= 0.9 x base (t(4) lower
    bound); 'correlate' = accuracy lower bound >= 0.9 x base with the linear probe's upper bound <= chance + 0.05; 'nonlinear residue'
    = MLP probe lower bound >= chance + 0.05 (task-identity probes)."""
    pr = t["probes"]
    return dict(causal=unlearns(t["acc"], t["dm"]["acc"], base_acc_mean),
                correlate=_ge(_lo(t["acc"]), 0.9 * base_acc_mean) and _le(_hi(pr["task_lin"]), chance + 0.05),
                nonlinear_residue=_ge(_lo(pr["task_mlp"]), chance + 0.05) if "task_mlp" in pr else None,
                dm_matched=bool(t["dm"]["match"]["matched"]), acc=summ(t["acc"]), dm_acc=summ(t["dm"]["acc"]), task_lin=summ(pr["task_lin"]),
                task_mlp=summ(pr["task_mlp"]) if "task_mlp" in pr else None)


def rules_t3(res):
    out = {}
    for Kk, arms in res["gate"].items():
        rg = sorted(arms.get("ring", []), key=lambda d: d["ws"]); pg = sorted(arms.get("pring", []), key=lambda d: d["ws"]); n = len(rg)
        both = sum(r["zero_shot"] >= 0.9 and p["zero_shot"] <= 0.2 for r, p in zip(rg, pg))
        out[f"gate_{Kk}"] = dict(passes=both >= quorum(n), seeds_both=both, ring=summ([d["zero_shot"] for d in rg]), pring=summ([d["zero_shot"] for d in pg]),
                                 c_onehot=summ([d["zero_shot"] for d in arms.get("c-onehot", [])]))
        out[f"hatch_{Kk}"] = [dict(ws=d["ws"], extract=d["hatch"]["extract"], first95=d["hatch"]["first95"]) for d in rg]
    out["gate"] = bool(out.get("gate_K1", {}).get("passes", False))
    word = {}; Ls = sorted(int(k[1:]) for k in res["word"]); lstar = None
    for L in Ls:
        w = res["word"][f"L{L}"]; arc = sorted(w["arc"], key=lambda d: d["ws"]); spr = sorted(w["spread"], key=lambda d: d["ws"]); n = len(arc)
        js = [jstar(d["by_j"]) for d in arc]; cov = sum(c >= mx for c, _, mx in js)
        diff = _a([a["zero_shot"] for a in arc]) - _a([s["zero_shot"] for s in spr])
        word[f"L{L}"] = dict(jstar=[c for c, _, _ in js], jstar_literal=[l_ for _, l_, _ in js], max_distance=[mx for _, _, mx in js], covers_seeds=cov,
                             arc_minus_spread=summ(diff), word_length_effect=_le(_hi(diff), -1e-300),
                             arc_maxsteps=[d["row"]["eval_point"] == "maxsteps" for d in arc])
        if lstar is None and cov >= quorum(n):
            lstar = L
    out["word"] = word; out["L_star"] = lstar
    ops = res.get("ops")
    if ops:
        b = ops["base"]; a0 = _mean(b["acc"]); sp0 = _mean(b["sp"]); cj = ops["conj"]
        share = float(np.mean([x for row in cj["modal_is_minus_c"] for x in row]))
        out["conjugation"] = dict(acc=summ(cj["acc"]), sp_ratio=(_mean(cj["sp"]) / sp0) if sp0 else None, modal_minus_c_share=share,
                                  inverted_ability=_le(_hi(cj["acc"]), CHANCE_A - 1e-12) and _ge(_mean(cj["sp"]), 0.8 * sp0) and share >= 0.75)
        out["t4b"] = {L: t4b_rule(t, a0, CHANCE_T) for L, t in ops["t4b"].items()}
        out["cplane"] = dict(acc=summ(ops["cplane"]["acc"]), zero_shot=summ(ops["cplane"]["zero_shot"]), rand_acc=summ(ops["cplane"]["rand"]["acc"]))
    if not out["gate"]:
        out["note"] = "circle gate failed: no pattern claim anywhere in Part C (G41)"
    return out


def rules_t4(res):
    out = dict(cells={}, w4={}); bs, bl = stage_budgets(res)
    for name, c in res["cells"].items():
        base = c["base"]; a0 = _mean(base["acc"]); sp0 = _mean(base["sp"]); n = len(base["acc"]); q = quorum(n); W = c["W"]; ops = {}
        for nm, o in c["ops"].items():
            op, dm = o["op"], o["dm"]; matched = bool(dm["match"]["matched"])
            rl, an = _a(op["relearn"][bs]), _a(op["A_new"][bs]); d = rl - an
            depth = "shallow" if (_ge(_mean(d), 0.20) and _ge(_lo(d), 0.10 + 1e-12)) else ("deep" if _le(_mean(rl), _mean(an) + 0.05) else "partial")
            n_perm = sum(lb == "consistent permutation" for lb in op["label"])
            ops[nm] = dict(unlearns=unlearns(op["acc"], dm["acc"], a0),
                           specificity=("not testable (no damage-matched control within 0.02)" if not matched
                                        else ("specific" if _ge(_mean(op["sp"]), 0.8 * sp0) else "generic damage")),
                           depth=depth, meta_unlearned=_le(_hi(op["B2"]["8"]), 0.05),
                           inverted=_le(_hi(op["acc"]), CHANCE_A - 1e-12) and _ge(_mean(op["sp"]), 0.8 * sp0) and n_perm >= q,
                           iso_status=ctrl_status(o["iso"]["acc"], a0), span_status=ctrl_status(o["span"]["acc"], a0) if o.get("span") else None,
                           dm_status=ctrl_status(dm["acc"], a0), dm_matched=matched, acc=summ(op["acc"]), dm_acc=summ(dm["acc"]), sp=summ(op["sp"]),
                           relearn_minus_new=summ(d), relearn_long=summ(op["relearn"][bl]), A_new_long=summ(op["A_new"][bl]),
                           B2_8=summ(op["B2"]["8"]), consistent_permutation_seeds=n_perm, modal_labels=op["label"])
        if ops["blk1"]["unlearns"]:
            lad = dict(rung="one direction", within_span_1dim_control=ops["blk1"]["span_status"])
        else:
            rs = next((k for k in (2, 4, 6) if ops[f"blk{k}"]["unlearns"] and ops[f"blk{k}"]["span_status"] == "does not unlearn"), None)
            if rs is not None:
                lad = dict(rung="subspace", r_star=rs)
            elif ops["block11"]["unlearns"]:
                st_ = ops["block11"]["iso_status"]
                lad = (dict(rung="subspace", r_star=11, label="subspace: the whole task span") if st_ == "does not unlearn" else
                       dict(rung="not specific") if st_ == "unlearns" else
                       dict(rung="undecided (isotropic control neither unlearns nor keeps >= 0.9 x base)"))
            else:
                lad = dict(rung="> 11", r_star="> 11")
        t4b = {L: t4b_rule(t, a0, CHANCE_T) for L, t in c["t4b"].items()}
        out["cells"][name] = dict(W=W, base_acc=a0, base_sp=sp0, ops=ops, ladder=lad, t4b=t4b,
                                  w64_note="W 64: specificity 'not testable' without a damage-matched control within 0.02" if W == 64 else None)
    for name, c in res.get("w4", {}).items():
        b = c["base"]; a0, s0 = _mean(b["add"]), _mean(b["sub"]); o2 = {}
        for nm, o in c["ops"].items():
            op, dm = o["op"], o["dm"]; matched = bool(dm["match"]["matched"])
            o2[nm] = dict(forget_unlearns=unlearns(op["add"], dm["add"], a0),
                          specificity=("not testable (no damage-matched control within 0.02)" if not matched else
                                       ("specific" if _ge(_mean(op["sub"]), 0.9 * _mean(dm["sub"])) else "generic damage")),
                          add=summ(op["add"]), sub=summ(op["sub"]), dm_add=summ(dm["add"]), dm_sub=summ(dm["sub"]), sp=summ(op["sp"]),
                          relearn_add_short=summ(op["relearn_add"][bs]), relearn_add_long=summ(op["relearn_add"][bl]))
        out["w4"][name] = dict(base_add=a0, base_sub=s0, ops=o2)
    return out


def rules_t5(res):
    """C-T5 primary protocol (per seed the 4 tasks averaged): A_free = A_3000(s_0); A_op = A_3000(s_24); r* = first r with A_3000(s_r)
    <= 0.11 (per task; per seed the median over its tasks, '> 24' counting as 25)."""
    out = {}
    for name, w in res["widths"].items():
        A = np.array(w["A"][str(int(max(w["budgets"])))], dtype=float); R = A.shape[1]; ch = w["chains"]; seeds = w["seeds"]; n = len(seeds); q = quorum(n)
        i4, last = min(4, R - 1), R - 1; arms = {}
        for arm in w["arms"]:
            idx = [i for i, (a, b, j) in enumerate(ch) if a == arm]; Aa = A[idx].reshape(n, NNEW, R)
            la = min(last, w.get("pc_last", last)) if arm == "pc" else last
            rs = rstar(Aa.reshape(-1, R)[:, :la + 1], never=RSTAR_NEVER).reshape(n, NNEW); rs_seed = np.median(rs, 1)
            A_free = Aa[:, :, 0].mean(1); A_op = Aa[:, :, la].mean(1)
            A_at_r = [float(np.mean([Aa[s, j, min(int(rs[s, j]), la)] for j in range(NNEW)])) for s in range(n)]     # s_(r*), s_last if never
            arms[arm] = dict(A_free=A_free, A_op=A_op, A_s4=Aa[:, :, i4].mean(1), rstar_task=rs, rstar_seed=rs_seed, A_op_at_rstar=A_at_r,
                             A_op_t=summ(A_op), last_round=la)
        col, rnd = arms.get("col"), arms.get("rand"); o = dict(rounds=R, s4_index=i4, last_index=last, arms=arms)
        if col and rnd:
            o["low_dimensional_search"] = bool(sum(r_ <= 4 for r_ in col["rstar_seed"]) >= q and sum(a >= 0.8 * f for a, f in zip(rnd["A_s4"], rnd["A_free"])) >= q)
            o["re_routing"] = _ge(_lo(col["A_op"]), UNLEARN_ACC + 1e-12)
        dm_last = [d for d in w.get("dm", []) if d["round"] == last]
        for X, ctl in (("col", "rand"), ("z1", "dmr")):
            if X in arms and ctl in arms:
                ax, ac = arms[X], arms[ctl]; ok_c = _ge(_mean(ac["A_op"]), 0.8 * _mean(ax["A_free"]))
                us = _le(_hi(ax["A_op"]), UNLEARN_ACC) and ok_c
                o[f"verdict_{X}"] = "un-searched" if us else ("partial" if _le(_mean(ax["A_op"]), 0.5 * _mean(ac["A_op"])) else "none")
                o[f"control_{X}"] = dict(control=ctl, A_rand_mean=_mean(ac["A_op"]), A_free_mean=_mean(ax["A_free"]), control_keeps=ok_c)
                if ctl == "dmr" and dm_last and not dm_last[0]["matched"]:
                    o[f"verdict_{X}"] = "not testable (no damage-matched control within 0.02 at s_last, G45)"
        out[name] = o
    return out


def rules_t6(res):
    fr = res["free"]; n = len(fr["modes"]); q = quorum(n); real = []
    for i in range(T6_MODES):
        cnt = sum(fr["modes"][s]["modes"][i]["var"] > fr["null"][s]["var_p99"][i] and fr["modes"][s]["modes"][i]["lissajous_r"] < fr["null"][s]["lis_p5"][i]
                  for s in range(n))
        real.append(dict(index=i + 1, seeds=int(cnt), real=cnt >= q))
    cap = {}
    for L, c in res["caps"].items():
        cnt = sum(bool(l1) and rx >= 1.5 and rw >= 1.5 for l1, rx, rw in zip(c["level1"], c["ratio_Ex"], c["ratio_W2"]))
        cap[L] = dict(seeds=int(cnt), re_routing=cnt >= q, level1_seeds=sum(bool(x) for x in c["level1"]), ratio_Ex=summ(c["ratio_Ex"]), ratio_W2=summ(c["ratio_W2"]),
                      B2_8=summ(c["B2"]["8"]))
    return dict(modes=real, caps=cap, free_B2_8=summ(fr["B2"]["8"]))


# ================================================================================================================ CLI
STAGE_FN = dict(t0=stage_t0, t1=stage_t1, t2=stage_t2, t3=stage_t3, t4=stage_t4, t5=stage_t5, t6=stage_t6)


def run_stage(out, st):
    """Run one stage (skipped when its output exists) and return its JSON content."""
    path = tdir(out, f"{st}.json")
    if os.path.exists(path):
        print(f"[{st}] output exists, skipped: {path}", flush=True); return read_json(path)
    t0 = time.time(); print(f"[{st}] start (device {DEV}, QUICK {QUICK}, seeds {list(SEEDS)})", flush=True)
    PROV["partials"].clear(); PROV["nets"].clear()
    res = STAGE_FN[st](out); res["meta"] = meta(st, t0); write_json(res, path)
    print(f"[{st}] done ({time.time() - t0:.0f}s) -> {path}", flush=True)
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
def selftest(work):
    """Code test (QUICK=1): unit checks of the helpers against r2_toy_common, then every stage end to end (twice for t0 to check that
    a rerun from the saved nets reproduces the output)."""
    if not QUICK:
        raise SystemExit("selftest needs QUICK=1")
    global MAXC
    out = os.path.join(work, "unlearn_selftest"); res = {}; t0 = time.time()

    def check(name, cond, info=""):
        res[name] = bool(cond); print(f"[{'PASS' if cond else 'FAIL'}] {name} {info} ({time.time() - t0:.0f}s)", flush=True)
    arm, W = CELLS[0]; nets, rows, _ = onehot_nets(out, arm, W); B = nets.B; ws_of = list(SEEDS); tasks = Tasks.w1(ws_of); allv = np.arange(B)
    e = eval_old(nets, tasks, allv)["acc"]; check("eval_old = TC.task_accuracy", np.allclose(e, TC.task_accuracy(nets, ws_of), atol=1e-6))
    check("sum_probe_gen = TC.sum_probe", np.allclose(sum_probe_gen(nets, tasks, allv), TC.sum_probe(nets, ws_of), atol=1e-9))
    nets2, _, _ = onehot_nets(out, arm, W); check("nets cache: reload with verified checksums", all(TC.weight_checksum(nets2, i) == r["checksum"] for i, r in enumerate(rows)))
    old = MAXC; MAXC = 5; r1 = relearn_old(nets, tasks, allv); MAXC = old; r2 = relearn_old(nets, tasks, allv)
    check("chunked search = one batched call", all(np.allclose(r1[b], r2[b]) for b in BUDGETS))
    zero = mk_ops([{"z1_U": torch.zeros(W, 3), "z1_ref": torch.zeros(W)} for _ in allv], W)
    check("zero-rank block is a no-op", np.allclose(eval_old(nets, tasks, allv, ops=zero)["acc"], e))
    Q, mu = fit_eraser(nets, tasks, "h2"); o = mk_ops(eraser_items(Q, mu, "h2", allv), W)
    vb, tp, x, y, _, _ = _chains(tasks, allv, 0, P * P)
    He = _hidden(nets, vb, _chain_cols(nets, tasks, vb, tp), x, y, _expand(o, B, NPRE), "h2").reshape(B, NPRE, P * P, W)
    cm = He.mean(2); spread = float((cm - cm.mean(1, keepdim=True)).abs().max())
    check("class-mean eraser: task means equal after erasure (linear guardedness)", spread < 1e-4, f"max {spread:.1e}")
    U, _, _ = task_pcs(nets); ref = TC.z1_ref(nets)
    blk = mk_ops([{"z1_U": U[b, :, :NPRE - 1], "z1_ref": ref[b]} for b in allv], W); a_blk = eval_old(nets, tasks, allv, ops=blk)["mean"]
    tgt = float(np.mean(sum_probe_gen(nets, tasks, allv) - sum_probe_gen(nets, tasks, allv, ops=blk)))
    dm = damage_match(nets, tasks, sum_probe_gen(nets, tasks, allv), tgt, "z1", ref, NPRE - 1, "selftest|dm")
    check("damage_match runs (matched or flagged)", dm["r"] >= NPRE - 1 and isinstance(dm["matched"], bool), f"{dm_public(dm)}")
    pb = push_bisect(nets, tasks, U[:, :, 0].cpu(), e.mean(1) - 0.5 * (e.mean(1) - a_blk), np.ones(B))
    check("push bisection hits its target or flags it", np.all(pb["matched"] | (np.abs(pb["acc"] - pb["target"]) > 0.02)), f"beta {np.round(pb['beta'], 3)}")
    for st in STAGES:
        try:
            r = run_stage(out, st); check(f"stage {st} runs and writes JSON with a rule preview", "rule_preview" in r and r["meta"]["plan_sha256"] == PLAN_SHA256)
        except Exception as ex:                                                   # noqa: BLE001  (report and continue)
            import traceback; traceback.print_exc(); check(f"stage {st} runs and writes JSON with a rule preview", False, repr(ex))
    p0 = tdir(out, "t0.json")
    if os.path.exists(p0):
        a = read_json(p0); os.remove(p0); b = run_stage(out, "t0")
        check("rerun of t0 from the saved nets reproduces the output", json.dumps(a["cells"], sort_keys=True) == json.dumps(b["cells"], sort_keys=True))
    bad = [k for k, v in res.items() if not v]
    print(f"r2_unlearn_toy selftest: {len(res) - len(bad)}/{len(res)} passed ({time.time() - t0:.0f}s)" + (f"; FAILED: {bad}" if bad else ""), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
