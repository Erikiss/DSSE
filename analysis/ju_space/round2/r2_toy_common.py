"""Round 2 toy library (implementer 5). Plan: results/ju_space/plans/abschnitt11_12_runde2_plan.md (final, sha256 5e476340...);
section 2.5 W0-W8, used by 3.2 AT0-AT7, 4.2 B-T1-B-T5, 5.1 C-T0-C-T7; script list 9.1.

Import only:  `import r2_toy_common as TC`.  Code test: `QUICK=1 NT=1 python r2_toy_common.py selftest [<work_dir>]`: W0's equivalence
tests against pc_meta.py (round-1 code, imported unchanged as M) and the comparison of the ported functions with the reference files of
Appendix E in round2/ref/ (their definitions executed in a namespace with the repository's pc_meta / pc_common: the shim of 9.5); it
writes <work_dir>/ref_comparison_w5.json (both sides and the largest differences of every check) and exits 1 if a check fails.

Conventions (W0)
- Before `import pc_meta` this module sets T=20, BETA=0.1, LR=1e-3, MAXSTEPS=6000 (run_shard.PC_ENV and the round-1 MAXSTEPS), unsets
  QUICK unless it is '1' (code tests), and asserts M.T == 20, pc_common.BETA[0] == 0.1, M.LR == 1e-3 and MAXSTEPS 6000 (QUICK: pc_meta's
  QUICK values). Device from env DEVICE, float32, TF32 off; every function is deterministic given its seeds (net init seeds the global
  RNG, as pc_meta does).
- Nets (the plan's BatchedMLP): B nets of one architecture stacked along dim 0 (baddbmm layers). kind 'mlp' (W1 one-hot worlds, any depth
  / activation / x-y coding, and the two-family world W4 with 32 slots), 'bneck' (rank-dt task slot U E, ref rgi_bneck.py), 'circle' (W3,
  ref c3_angle.py), 'impose' (W5, ref impose2.py / c2_pring.py). Each net's first layer is z1 = W1 [fx(x), fy(y), ft(t)] + b1 with per-net
  tables fx, fy (codings) and ft (task features), so every net has x columns Ex, y columns Ey and task columns (task_cols), and
  z1 = base_z1(x, y) + col(task) with base_z1 = Ex[:, x] + Ey[:, y] + b1; searches replace col(task) by a searched column v.
- Spec (dict) of a net, defaults SPEC_DEFAULTS: W and ws (world seed) required; init (0: torch.manual_seed(ws); i > 0: 10000 i + ws) or
  seed; arm (bp_mup / bp_sp / pc_mup); kind; depth; act; lr; wd (wd_scale); gamma, reg ('task' | 'sham'), when ('early' | 'late'); data
  ('w1' | 'w3' | 'w4' | 'w5'); K; shifts; split ('world' | 'other' | 'shared' | per-slot list); ky (task family x + ky y + c); coding
  (None | 'perm' = default_rng(6000 + ws) | (px, py)); dt (bneck); circ / cK (W3 arm, frequencies); code / ck (W5 code, k); arc / spread
  (W3 word length). net_name(spec) is deterministic; weight_checksum is the sha256 of a net's float32 parameters.

API (details in the docstrings; NOTES_w5.md has the list with the plan items each serves)
- worlds:   world, spare_shift, init_seed, fresh_seed, other_splits, shared_split, coding_perms, four, code_table, circle_tables, w3_shifts,
            w4_tasks, w4_split, taskset_w1 / _w3 / _w4 / _w5, taskset_from_spec, new_task_sets, old_task_sets, stack_sets, lambda_targets
- nets:     Nets (from_specs, from_states, subset, cat, clone, x_cols, y_cols, task_cols, features, logits, hidden_acts, state, to_mlp),
            Ops (z1 block / reflection / partial removal / push, h1 / h2 erasure; expand, regroup), z1_ref, block_ops, random_subspaces,
            random_in_span, complement_basis
- training: train (checkpoint rule, fixed length, early / late regulariser, second checkpoint, snapshots, C-T6 task-update basis,
            first95 state), train_late (W2 branches), AdamWB, task_fisher_b, param_distance, snap_task_code, TrainResult
- B2:       b2_A, fresh_F, b2_all, b2 (A, F, B2, all8 / retention as pc_meta.main)
- geometry: delta_stats (vectorised pc_meta.delta_stats), mediators (AT3), task_accuracy, task_states, hidden_states, predictions,
            cols_accuracy, zero_shot (W3 / W5)
- W6:       self_train, w6_summary
- W7:       column_search (free / anchored / coordinates / recipe / keep / remove / ops / best-of-budget / BP or PC / CE, KL, h2-MSE),
            anchored_param, coords_param
- W8:       sum_probe, ridge_probe, mlp_probe, heldout_halves, task_answer_features, task_answer_probes, class_mean_eraser,
            damage_match (OD-C2), modal_shift_map (OD-C16)
- RGI:      lambda_triples, lambda_line, learn_alpha, carry_alpha, s_perms, s_line, ridge_coef, t1_map, t1_choose_r, proj_linspan,
            f_line, recipe_chains, recipe_depth, round_trip, pooling, warm_start, stitch_fit, all_h2, t3_t4
- C-T0/C-T6: fourier_cols, share_by_k, ring_test, loo_ring, spec_rows, subspace_overlap, structure_tests, trajectory_modes,
            trajectory_basis, shuffled_increment_nulls
- PC:       pc_relax (chains), pc_relax_inputs (whole nets), pc_eta (1 / pc_meta.lam_max), PC searches via column_search(pc=...)
- I/O:      net_name, weight_checksum, save_nets, load_nets, load_pc_state_dicts (round-1 PC-muP and BP-muP checkpoints), r1_training_batch
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn

_HERE = os.path.dirname(os.path.abspath(__file__))
_JU = os.path.dirname(_HERE)
if _JU not in sys.path:
    sys.path.insert(0, _JU)
PC_ENV = {"T": "20", "BETA": "0.1", "LR": "1e-3", "MAXSTEPS": "6000"}      # run_shard.PC_ENV and the round-1 MAXSTEPS (W0)
if "pc_meta" not in sys.modules:
    os.environ.update(PC_ENV)
    if os.environ.get("QUICK") != "1":
        os.environ.pop("QUICK", None)                                       # W0: QUICK is unset outside code tests
import pc_meta as M          # noqa: E402  round-1 code, imported unchanged
import pc_common as PCC      # noqa: E402

M.T = int(PC_ENV["T"]); PCC.BETA[0] = float(PC_ENV["BETA"]); M.LR = float(PC_ENV["LR"])
if not M.QUICK:
    M.MAXSTEPS = int(PC_ENV["MAXSTEPS"])
assert M.T == 20 and PCC.BETA[0] == 0.1 and M.LR == 1e-3, "pc_meta environment (W0)"
assert (M.MAXSTEPS, M.AFTER, M.LEVEL1, M.FT_STEPS) == ((200, 50, 0.0, 20) if M.QUICK else (6000, 500, 0.95, 300)), "pc_meta constants (W0)"

DEV = M.DEV
if DEV.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
F32, F64 = torch.float32, torch.float64

P, NPRE, NNEW, NSLOT, BASE = M.P, M.NPRE, M.NNEW, M.NSLOT, M.BASE          # 17, 12, 4, 17, 128
QUICK = M.QUICK
LR, DECAY, BETAS, EPS = 1e-3, M.DECAY, M.BETAS, 1e-8                       # pc_meta optimiser (AdamW; Adam for column searches)
MAXSTEPS = M.MAXSTEPS if QUICK else 6000                                   # W1 / 2.7 C6 (QUICK: pc_meta's 200)
AFTER, LEVEL1, CHECK_EVERY = M.AFTER, M.LEVEL1, 50                        # checkpoint rule W0 (QUICK: 50, 0.0)
FT_STEPS, FT_EVAL, FT_LR = M.FT_STEPS, tuple(M.FT_EVAL), M.FT_LR          # B2 column search (QUICK: 20, (5, 10, 20))
T_PC, BETA_PC = 20, 0.1                                                    # PC-muP arm of round 1
NSHOT, NVAL = 8, 4                                                         # W7: shots = split positions 0-7, validation 8-11
BUDGETS = (20, 60) if QUICK else (300, 3000)                               # W7 / OD-C1 best-of-budget
VAL_EVERY = 10
SECOND_AFTER = 1000                                                        # AT2: natural runs continued to first95 + 1500
CHUNK = int(os.environ.get("R2_CHUNK", str(2 ** 25)))                      # max floats per evaluation chunk
PAIRS = np.array([(x, y) for x in range(P) for y in range(P)])            # pc_meta's pair order (x major)


# ====================================================================================================================== worlds
def world(ws):
    """pc_meta.world(ws): (pre [12 shifts], new [4 shifts], pairs [289, 2], splits [16 permutations of the 289 pairs])."""
    return M.world(ws)


def spare_shift(ws):
    """The 17th shift value sh[16] of world(ws) (the one neither pretrained nor new)."""
    pre, new, _, _ = M.world(ws)
    return [c for c in range(P) if c not in pre and c not in new][0]


def init_seed(ws, init=0):
    """W1 init convention: init 0 = torch.manual_seed(ws) (= pc_meta), init i > 0 = torch.manual_seed(10000 i + ws)."""
    return int(ws) if int(init) == 0 else 10000 * int(init) + int(ws)


def fresh_seed(seed, j):
    """pc_meta.finetune 'fresh': torch.manual_seed(50000 + 101 seed + j)."""
    return 50_000 + 101 * int(seed) + int(j)


def other_splits(ws, n=NPRE + NNEW + 1):
    """AT4 / OD-A9 'other split': n permutations of the 289 pairs from np.random.default_rng(5000 + ws), one per slot."""
    g = np.random.default_rng(5000 + ws)
    return [g.permutation(P * P) for _ in range(n)]


def shared_split(ws, frac=0.5):
    """AT6 shared (x, y) split (ref rgi_long.py SHARED): one permutation from default_rng(5000 + ws); the first round(frac 289) = 144
    pairs train every task, the rest are held out."""
    sh = np.random.default_rng(5000 + ws).permutation(P * P)
    return sh, int(round(frac * P * P))


def coding_perms(ws):
    """AT4 / OD-A9 permuted x / y coding: px, py = two permutations of Z_17 from default_rng(6000 + ws) (x enters at position px[x])."""
    g = np.random.default_rng(6000 + ws)
    return g.permutation(P), g.permutation(P)


def four(K):
    """Fourier features of Z_17, frequencies 1..K: [17, 2K] (ref c3_angle.four)."""
    c = np.arange(P)
    return np.concatenate([np.stack([np.cos(2 * np.pi * f * c / P), np.sin(2 * np.pi * f * c / P)], 1) for f in range(1, K + 1)], 1).astype(np.float32)


def code_table(kind, k, ws):
    """W5 imposed task codes phi [17, 2k] (ref impose2.code_table, c2_pring.code_table): 'ring' = Fourier pairs f = 1..k of c; 'pring' =
    the ring table with rows permuted by default_rng(700 + ws).permutation(17); 'rand' = a random 2k-dim code of centred functions with
    the ring table's column norms (default_rng(900 + ws))."""
    c = np.arange(P)
    F = np.concatenate([np.stack([np.cos(2 * np.pi * f * c / P), np.sin(2 * np.pi * f * c / P)], 1) for f in range(1, k + 1)], 1)
    if kind == "ring":
        return torch.as_tensor(F, dtype=F32)
    if kind == "pring":
        perm = np.random.default_rng(700 + ws).permutation(P)
        return torch.as_tensor(F, dtype=F32)[torch.as_tensor(perm)]
    if kind != "rand":
        raise ValueError(kind)
    g = np.random.default_rng(900 + ws); one = np.ones((P, 1)) / np.sqrt(P)
    Q, _ = np.linalg.qr(g.standard_normal((P, 2 * k)) - one @ (one.T @ g.standard_normal((P, 2 * k))))
    Q = Q - one @ (one.T @ Q); Q, _ = np.linalg.qr(Q)
    return torch.as_tensor(Q * np.linalg.norm(F, axis=0), dtype=F32)


def circle_tables(arm, K, ws):
    """W3 input tables (fx [17, dx], fc [17, dc]): 'ring' f(x), f(y), f(c); 'pring' f(c) rows permuted by default_rng(700 + ws);
    'c-onehot' c one-hot with x, y ring-coded (plan W3); 'x-onehot' x, y one-hot with c ring-coded (c3_angle's 'onehot' arm)."""
    F = four(K); eye = np.eye(P, dtype=np.float32)
    if arm == "ring":
        return F, F
    if arm == "pring":
        return F, F[np.random.default_rng(700 + ws).permutation(P)]
    if arm == "c-onehot":
        return F, eye
    if arm == "x-onehot":
        return eye, F
    raise ValueError(arm)


SPEC_DEFAULTS = dict(kind="mlp", arm="bp_mup", init=0, seed=None, depth=2, act="tanh", lr=1e-3, wd=1.0, gamma=0.0, reg="task",
                     when="early", data="w1", K=NPRE, shifts=None, split="world", ky=1, coding=None, dt=4, circ="ring", cK=1,
                     code="ring", ck=2, arc=None, spread=None, tag=None)
_NAME_KEYS = ("kind", "seed", "depth", "act", "lr", "wd", "gamma", "reg", "when", "data", "K", "shifts", "split", "ky", "coding", "dt",
              "circ", "cK", "code", "ck", "arc", "spread", "tag")


def full_spec(spec):
    s = dict(SPEC_DEFAULTS); s.update(spec)
    if "W" not in s or "ws" not in s:
        raise ValueError("a spec needs W and ws")
    if s["kind"] == "circle":
        s["data"] = "w3"
    if s["kind"] == "impose":
        s["data"] = "w5"
    if s["data"] == "w4":
        s["K"] = 2 * NPRE
    return s


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple, np.ndarray)):
        return "-".join(_fmt(x) for x in (v.tolist() if isinstance(v, np.ndarray) else v))
    return str(v)


def net_name(spec):
    """Deterministic net name: '<arm>_W<W>_w<ws>_i<init>' plus '_<key><value>' for every spec key that differs from SPEC_DEFAULTS (fixed
    key order; the gamma / reg / when keys only when gamma != 0; circle / impose keys only for those kinds), '_quick' under QUICK."""
    s = full_spec(spec); d = SPEC_DEFAULTS; out = f"{s['arm']}_W{s['W']}_w{s['ws']}_i{s['init']}"
    for k in _NAME_KEYS:
        if k in ("reg", "when") and not s["gamma"]:
            continue
        if k in ("circ", "cK") and s["kind"] != "circle":
            continue
        if k in ("code", "ck") and s["kind"] != "impose":
            continue
        if k == "dt" and s["kind"] != "bneck":
            continue
        if k == "data" and s["kind"] in ("circle", "impose"):
            continue
        if k == "K" and s["data"] == "w4":
            continue
        v, v0 = s[k], d[k]
        same = (v is None and v0 is None) or (v is not None and v0 is not None and not isinstance(v, (list, tuple, np.ndarray)) and v == v0)
        if not same:
            out += f"_{k}{_fmt(v) if not (k == 'split' and isinstance(v, (list, tuple))) else 'custom'}"
    return out + ("_quick" if QUICK else "")


def _pairs_of(split, lo, hi):
    idx = np.asarray(split)[lo:hi]
    return PAIRS[idx, 0], PAIRS[idx, 1]


def _label(x, y, c, ky=1):
    return (np.asarray(x) + ky * np.asarray(y) + c) % P


def _taskset(rows_tr, rows_va, val_mode, groups, info):
    """rows: list of (x, y, t, lab, task, group) arrays per task; returns the dict used by train()."""
    cat = lambda rows, i: np.concatenate([r[i] for r in rows]).astype(np.int64)
    ts = dict(x=cat(rows_tr, 0), y=cat(rows_tr, 1), t=cat(rows_tr, 2), lab=cat(rows_tr, 3), task=cat(rows_tr, 4),
              vx=cat(rows_va, 0), vy=cat(rows_va, 1), vt=cat(rows_va, 2), vlab=cat(rows_va, 3), vtask=cat(rows_va, 4), vgroup=cat(rows_va, 5),
              val_mode=val_mode, ngroups=groups, ntask=len(rows_va), info=info)
    return ts


def taskset_w1(ws, K=NPRE, shifts=None, split="world", ky=1, slots=None):
    """W1 training data (pc_meta.main): task k = shift shifts[k] in slot slots[k] (default pre[:K], slots 0..K-1), training pairs
    split[:ntr], held-out split[ntr:] (pooled held-out accuracy for level 1). split: 'world' (world(ws) splits by slot), 'other'
    (other_splits(ws)), 'shared' (shared_split(ws) for every task), or a list of per-slot permutations."""
    pre, new, pairs, wsp = M.world(ws)
    shifts = list(pre[:K]) if shifts is None else [int(c) for c in shifts]
    slots = list(range(len(shifts))) if slots is None else list(slots)
    ntr = 144
    if isinstance(split, str):
        if split == "world":
            sp = [wsp[s] for s in slots]
        elif split == "other":
            o = other_splits(ws); sp = [o[s] for s in slots]
        elif split == "shared":
            sh, ntr = shared_split(ws); sp = [sh for _ in slots]
        else:
            raise ValueError(split)
    else:
        sp = [split[s] for s in slots]
    tr, va = [], []
    for k, (c, s, perm) in enumerate(zip(shifts, slots, sp)):
        x, y = _pairs_of(perm, 0, ntr); tr.append((x, y, np.full(len(x), s), _label(x, y, c, ky), np.full(len(x), k), np.zeros(len(x))))
        x, y = _pairs_of(perm, ntr, P * P); va.append((x, y, np.full(len(x), s), _label(x, y, c, ky), np.full(len(x), k), np.zeros(len(x))))
    return _taskset(tr, va, "pooled", 1, dict(shifts=shifts, slots=slots, ky=ky, ntr=ntr, new=list(new)))


def w3_shifts(ws, arc=None, spread=None):
    """W3 pretraining shifts: full coverage = pre (12) of world(ws); arc of L = c0 .. c0 + L - 1 mod 17 with c0 = pre[0]; spread of L =
    pre[:L] (OD-C8). Returns (trained shifts, untrained shifts)."""
    pre, new, _, _ = M.world(ws)
    if arc is not None:
        tr = [(pre[0] + i) % P for i in range(arc)]
    elif spread is not None:
        tr = list(pre[:spread])
    else:
        tr = list(pre)
    return tr, [c for c in range(P) if c not in tr]


def taskset_w3(ws, arc=None, spread=None):
    """W3 circle world (ref c3_angle.run): task index = shift value c (table row), split of c = splits[c mod 16] of world(ws), 144 / 145
    pairs; level 1 = mean over the trained shifts of the per-shift held-out accuracy."""
    pre, new, pairs, splits = M.world(ws); trs, _ = w3_shifts(ws, arc, spread); tr, va = [], []
    for k, c in enumerate(trs):
        sp = splits[c % len(splits)]
        x, y = _pairs_of(sp, 0, 144); tr.append((x, y, np.full(len(x), c), _label(x, y, c), np.full(len(x), k), np.zeros(len(x))))
        x, y = _pairs_of(sp, 144, P * P); va.append((x, y, np.full(len(x), c), _label(x, y, c), np.full(len(x), k), np.zeros(len(x))))
    return _taskset(tr, va, "task_mean", 1, dict(shifts=trs, ky=1, ntr=144, new=list(new)))


def w4_split(ws, slot):
    """W4 split of a slot (ref c5_twofam.spl): splits[slot] of world(ws) for slot < 16, else default_rng(3000 + 31 ws + slot)."""
    splits = M.world(ws)[3]
    return splits[slot] if slot < len(splits) else np.random.default_rng(3000 + 31 * ws + slot).permutation(P * P)


def w4_tasks(ws):
    """W4 tasks (family, shift, slot): add (x + y + c) slots 0-11 and sub (x - y + c) slots 12-23 over the 12 pretraining shifts; new add
    tasks slots 24-27, new sub tasks 28-31 (new shifts of world(ws))."""
    pre, new, _, _ = M.world(ws)
    old = [("add", c, i) for i, c in enumerate(pre)] + [("sub", c, NPRE + i) for i, c in enumerate(pre)]
    nw = [("add", c, 2 * NPRE + j) for j, c in enumerate(new)] + [("sub", c, 2 * NPRE + NNEW + j) for j, c in enumerate(new)]
    return old, nw


def taskset_w4(ws):
    """W4 two-family world (ref c5_twofam.py): 24 one-hot slots (+ 8 new) = 32 slots, din 66; level 1 = both families' held-out (mean of
    per-task accuracies) >= LEVEL1."""
    old, _ = w4_tasks(ws); tr, va = [], []
    for k, (fam, c, s) in enumerate(old):
        ky = 1 if fam == "add" else -1; sp = w4_split(ws, s); grp = 0 if fam == "add" else 1
        x, y = _pairs_of(sp, 0, 144); tr.append((x, y, np.full(len(x), s), _label(x, y, c, ky), np.full(len(x), k), np.full(len(x), grp)))
        x, y = _pairs_of(sp, 144, P * P); va.append((x, y, np.full(len(x), s), _label(x, y, c, ky), np.full(len(x), k), np.full(len(x), grp)))
    return _taskset(tr, va, "task_mean", 2, dict(shifts=[c for _, c, _ in old], ky=None, ntr=144, families=[f for f, _, _ in old]))


def taskset_w5(ws):
    """W5 imposed codes (ref impose2.run): task index = shift value (table row phi(c)), task j's split = splits[j] of world(ws); level 1 =
    mean over the 12 tasks of the per-task held-out accuracy."""
    pre, new, pairs, splits = M.world(ws); tr, va = [], []
    for k, c in enumerate(pre):
        x, y = _pairs_of(splits[k], 0, 144); tr.append((x, y, np.full(len(x), c), _label(x, y, c), np.full(len(x), k), np.zeros(len(x))))
        x, y = _pairs_of(splits[k], 144, P * P); va.append((x, y, np.full(len(x), c), _label(x, y, c), np.full(len(x), k), np.zeros(len(x))))
    return _taskset(tr, va, "task_mean", 1, dict(shifts=list(pre), ky=1, ntr=144, new=list(new)))


def taskset_from_spec(spec):
    """Training data of the net described by spec (W1 / W3 / W4 / W5)."""
    s = full_spec(spec)
    if s["data"] == "w1":
        return taskset_w1(s["ws"], s["K"], s["shifts"], s["split"], s["ky"])
    if s["data"] == "w3":
        return taskset_w3(s["ws"], s["arc"], s["spread"])
    if s["data"] == "w4":
        return taskset_w4(s["ws"])
    if s["data"] == "w5":
        return taskset_w5(s["ws"])
    raise ValueError(s["data"])


def new_task_sets(ws, j, ky=1, nshot=NSHOT, val=False, slot=None, shift=None, split=None):
    """New task j of world(ws) (pc_meta.finetune / W7): shots = split positions 0..nshot-1, held-out = 8-288 (281 pairs; B2) or, with
    val=True, validation = 8-11 and held-out = 12-288 (277 pairs). Returns dict(shot=(x, y, lab), ho=(x, y, lab), val=(...) or None,
    c, slot)."""
    pre, new, pairs, splits = M.world(ws)
    slot = NPRE + j if slot is None else slot; c = new[j] if shift is None else shift
    sp = splits[slot] if split is None else split
    sx, sy = _pairs_of(sp, 0, nshot)
    out = dict(c=int(c), slot=int(slot), shot=(sx, sy, _label(sx, sy, c, ky)))
    if val:
        vx, vy = _pairs_of(sp, NSHOT, NSHOT + NVAL); hx, hy = _pairs_of(sp, NSHOT + NVAL, P * P)
        out.update(val=(vx, vy, _label(vx, vy, c, ky)), ho=(hx, hy, _label(hx, hy, c, ky)))
    else:
        hx, hy = _pairs_of(sp, NSHOT, P * P); out.update(val=None, ho=(hx, hy, _label(hx, hy, c, ky)))
    return out


def old_task_sets(ws, k, ky=1, shift=None, split=None, ntr=144):
    """Old (pretraining) task k relearning sets (OD-C14 / W7): shots = training split positions 0-7, validation = 8-11, held-out = the
    task's held-out pairs (positions ntr..288, 145 pairs)."""
    pre, new, pairs, splits = M.world(ws)
    c = pre[k] if shift is None else shift; sp = splits[k] if split is None else split
    sx, sy = _pairs_of(sp, 0, NSHOT); vx, vy = _pairs_of(sp, NSHOT, NSHOT + NVAL); hx, hy = _pairs_of(sp, ntr, P * P)
    return dict(c=int(c), slot=int(k), shot=(sx, sy, _label(sx, sy, c, ky)), val=(vx, vy, _label(vx, vy, c, ky)), ho=(hx, hy, _label(hx, hy, c, ky)))


def stack_sets(sets, key):
    """Stack the (x, y, lab) triples of a list of set dicts (new_task_sets / old_task_sets) into [C, N] arrays."""
    trip = [s[key] for s in sets]
    return tuple(np.stack([t[i] for t in trip]) for i in range(3))


# ====================================================================================================================== nets
def _linear_init(din, dout):
    """nn.Linear(din, dout)'s default init, consuming the global RNG exactly as constructing the module does."""
    lin = torch.nn.Linear(din, dout)
    return lin.weight.detach().clone(), lin.bias.detach().clone()


def _init_one(s, din, nt):
    """Per-net parameters (CPU) with the reference RNG sequence of each kind: 'mlp' = pc_common.MLP (depth 2) / crit/family2.Net (depth
    3): Linear inits of l1, hidden, l3, then for muP normal_(l1, 1/sqrt(din)), zero b1, normal_(hidden, 1/sqrt(W)), zero biases and l3;
    'bneck' = rgi_bneck.MLPB (MLP init, then U = randn(W, dt), E = randn(dt, 17) / sqrt(51) / sqrt(dt), slot columns zero);
    'circle' = c3_angle.Net; 'impose' = impose2.Net."""
    W, depth, kind = s["W"], s["depth"], s["kind"]
    torch.manual_seed(s["seed"] if s["seed"] is not None else init_seed(s["ws"], s["init"]))
    p = {}
    if kind in ("mlp", "bneck"):
        W1, b1 = _linear_init(din, W); hid = [_linear_init(W, W) for _ in range(depth - 1)]; W3, b3 = _linear_init(W, P)
        if s["arm"] != "bp_sp":
            W1.normal_(0, 1 / math.sqrt(din)); b1.zero_()
            for Wh, bh in hid:
                Wh.normal_(0, 1 / math.sqrt(W)); bh.zero_()
            W3.zero_(); b3.zero_()
        if kind == "bneck":
            dt = s["dt"]; U = torch.randn(W, dt); E = torch.randn(dt, NSLOT) * (1 / math.sqrt(2 * P + NSLOT)) / math.sqrt(dt)
            W1 = W1[:, :2 * P].clone(); p.update(U=U, E=E)
    elif kind == "circle":
        fx, fc = circle_tables(s["circ"], s["cK"], s["ws"]); dx, dc = fx.shape[1], fc.shape[1]; dn = 2 * dx + dc
        Ax = torch.randn(W, dx) / math.sqrt(dn); Ay = torch.randn(W, dx) / math.sqrt(dn); Ac = torch.randn(W, dc) / math.sqrt(dn)
        b1 = torch.zeros(W); W2, b2 = _linear_init(W, W); W3, b3 = _linear_init(W, P)
        W2.normal_(0, 1 / math.sqrt(W)); b2.zero_(); W3.zero_(); b3.zero_(); W1 = torch.cat([Ax, Ay, Ac], 1); hid = [(W2, b2)]
    elif kind == "impose":
        r = 2 * s["ck"]; dref = 2 * P + NSLOT
        Ex = torch.randn(W, 2 * P) / math.sqrt(dref); t0 = torch.zeros(W); A = torch.randn(W, r) / math.sqrt(dref) / math.sqrt(r / 2)
        b1 = torch.zeros(W); W2, b2 = _linear_init(W, W); W3, b3 = _linear_init(W, P)
        W2.normal_(0, 1 / math.sqrt(W)); b2.zero_(); W3.zero_(); b3.zero_(); W1 = torch.cat([Ex, A, t0[:, None]], 1); hid = [(W2, b2)]
    else:
        raise ValueError(kind)
    p["W1"], p["b1"] = W1, b1
    for i, (Wh, bh) in enumerate(hid):
        nm = "2" if i == 0 else f"2_{i}"; p["W" + nm], p["b" + nm] = Wh, bh
    p["W3"], p["b3"] = W3, b3
    return p


def _tables_one(s):
    """(fx [17, dx], fy [17, dx], ft [nt, dtin]) of one net (numpy float32)."""
    eye = np.eye(P, dtype=np.float32)
    if s["kind"] in ("mlp", "bneck"):
        nslot = 2 * NPRE + 2 * NNEW if s["data"] == "w4" else NSLOT
        cod = s["coding"]
        if cod is None:
            px = py = np.arange(P)
        elif isinstance(cod, str) and cod == "perm":
            px, py = coding_perms(s["ws"])
        else:
            px, py = np.asarray(cod[0]), np.asarray(cod[1])
        return eye[px], eye[py], np.eye(nslot, dtype=np.float32)
    if s["kind"] == "circle":
        fx, fc = circle_tables(s["circ"], s["cK"], s["ws"])
        return fx, fx, fc
    phi = code_table(s["code"], s["ck"], s["ws"]).numpy()
    return eye, eye, np.concatenate([phi, np.ones((P, 1), np.float32)], 1)


class Nets:
    """B nets of one architecture stacked along dim 0 (the plan's BatchedMLP; baddbmm layers).
    p: dict of [B, ...] float32 tensors: W1 [B, W, din] (bneck: [B, W, 34] plus U [B, W, dt], E [B, dt, 17]), b1, W2, b2 (more hidden
    layers: W2_1, b2_1, ...), W3 [B, 17, W], b3. Tables fx, fy [B, 17, dx], ft [B, nt, dtin]; mult = BASE / W under muP (output scale),
    1 under SP; meta = list of per-net dicts (spec, name, ...)."""

    def __init__(self, kind, W, depth, act, param, p, fx, fy, ft, meta=None):
        self.kind, self.W, self.depth, self.act, self.param = kind, int(W), int(depth), act, param
        self.p, self.fx, self.fy, self.ft = p, fx, fy, ft
        self.meta = list(meta) if meta is not None else [{} for _ in range(p["b1"].shape[0])]
        self.mult = BASE / self.W if param == "mup" else 1.0

    # ---- construction
    @staticmethod
    def from_specs(specs, device=None):
        """Initialise the nets of specs (one architecture) with the reference init of their kind on the CPU (global RNG seeded by
        the spec's seed, as pc_meta), then move them to the device."""
        ss = [full_spec(s) for s in specs]; s0 = ss[0]
        for s in ss:
            for k in ("kind", "W", "depth", "act"):
                if s[k] != s0[k]:
                    raise ValueError(f"one architecture per Nets: {k} differs")
            if (s["arm"] == "bp_sp") != (s0["arm"] == "bp_sp"):
                raise ValueError("one parameterisation per Nets")
        tabs = [_tables_one(s) for s in ss]
        din = 2 * tabs[0][0].shape[1] + tabs[0][2].shape[1]
        ps = [_init_one(s, din, tabs[0][2].shape[0]) for s in ss]
        dev = DEV if device is None else device
        p = {k: torch.stack([q[k] for q in ps]).to(dev) for k in ps[0]}
        fx, fy, ft = (torch.as_tensor(np.stack([t[i] for t in tabs]), device=dev) for i in range(3))
        meta = [dict(spec={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}, name=net_name(s)) for s in ss]
        return Nets(s0["kind"], s0["W"], s0["depth"], s0["act"], "sp" if s0["arm"] == "bp_sp" else "mup", p, fx, fy, ft, meta)

    @staticmethod
    def from_states(states, kind="mlp", param="mup", act="tanh", tables=None, meta=None, device=None):
        """Nets from per-net parameter dicts in this class's names, or pc_common.MLP state dicts (keys l1.weight ... l3.bias). tables:
        (fx, fy, ft) arrays [B, ...] (default identity tables of the one-hot world with 17 slots)."""
        conv = []
        for st in states:
            if "l1.weight" in st:
                st = {"W1": st["l1.weight"], "b1": st["l1.bias"], "W2": st["l2.weight"], "b2": st["l2.bias"], "W3": st["l3.weight"], "b3": st["l3.bias"]}
            conv.append({k: torch.as_tensor(v, dtype=F32).detach().cpu() for k, v in st.items()})
        dev = DEV if device is None else device
        p = {k: torch.stack([c[k] for c in conv]).to(dev) for k in conv[0]}
        W = p["b1"].shape[1]; depth = 1 + sum(1 for k in p if k.startswith("W2"))
        if tables is None:
            eye = np.eye(P, dtype=np.float32); n = p["W1"].shape[2] - 2 * P if kind == "mlp" else NSLOT
            tables = (np.stack([eye] * len(conv)), np.stack([eye] * len(conv)), np.stack([np.eye(n, dtype=np.float32)] * len(conv)))
        fx, fy, ft = (torch.as_tensor(np.asarray(t), dtype=F32, device=dev) for t in tables)
        return Nets(kind, W, depth, act, param, p, fx, fy, ft, meta)

    @property
    def B(self):
        return self.p["b1"].shape[0]

    @property
    def device(self):
        return self.p["b1"].device

    def hidden(self):
        return ["W2"] + [f"W2_{i}" for i in range(1, self.depth - 1)]

    def subset(self, idx):
        """New Nets with the selected nets (copies)."""
        it = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=self.device)
        return Nets(self.kind, self.W, self.depth, self.act, self.param, {k: v[it].clone() for k, v in self.p.items()},
                    self.fx[it].clone(), self.fy[it].clone(), self.ft[it].clone(), [self.meta[i] for i in np.asarray(idx).tolist()])

    @staticmethod
    def cat(nets_list):
        """Concatenate Nets of one architecture."""
        a = nets_list[0]
        for b in nets_list[1:]:
            if (b.kind, b.W, b.depth, b.act, b.param) != (a.kind, a.W, a.depth, a.act, a.param):
                raise ValueError("Nets.cat: architectures differ")
        return Nets(a.kind, a.W, a.depth, a.act, a.param, {k: torch.cat([n.p[k] for n in nets_list]) for k in a.p},
                    torch.cat([n.fx for n in nets_list]), torch.cat([n.fy for n in nets_list]), torch.cat([n.ft for n in nets_list]),
                    sum((n.meta for n in nets_list), []))

    def clone(self):
        return self.subset(np.arange(self.B))

    # ---- first layer views
    def W1full(self):
        """[B, W, din]: the first-layer weight on the input features (bneck: [W1x | U E])."""
        if self.kind == "bneck":
            return torch.cat([self.p["W1"], self.p["U"] @ self.p["E"]], 2)
        return self.p["W1"]

    def f(self, z):
        return torch.tanh(z) if self.act == "tanh" else torch.relu(z)

    def dx(self):
        return self.fx.shape[2]

    def x_cols(self):
        """Ex [B, W, 17]: the first-layer input of each x value (raw value order, coding applied)."""
        d = self.dx(); return self.W1full()[:, :, :d] @ self.fx.transpose(1, 2)

    def y_cols(self):
        d = self.dx(); return self.W1full()[:, :, d:2 * d] @ self.fy.transpose(1, 2)

    def task_cols(self):
        """[B, W, nt]: the column of every task index (one-hot slots: W1[:, 2P + t]; circle: Ac fc[c]; impose: t0 + A phi(c))."""
        d = self.dx(); return self.W1full()[:, :, 2 * d:] @ self.ft.transpose(1, 2)

    def features(self, x, y, t):
        """Input features [B, N, din] for per-net index arrays x, y, t [B, N] (tensors on the device)."""
        bi = torch.arange(self.B, device=self.device)[:, None]
        return torch.cat([self.fx[bi, x], self.fy[bi, y], self.ft[bi, t]], 2)

    def z1(self, X):
        return torch.baddbmm(self.p["b1"].unsqueeze(1), X, self.W1full().transpose(1, 2))

    def logits(self, X, ops=None):
        """Logits [B, N, 17] for input features X [B, N, din] (ops per net, leading dim B)."""
        z = self.z1(X).unsqueeze(1)
        return _rest(self, None, z, None if ops is None else ops.view_per_net(self.B))[:, 0]

    def hidden_acts(self, X, ops=None):
        """(h1, h_last) [B, N, W] (after any ops)."""
        z = self.z1(X).unsqueeze(1)
        _, h1, h2 = _rest(self, None, z, None if ops is None else ops.view_per_net(self.B), want=True)
        return h1[:, 0], h2[:, 0]

    # ---- states
    def state(self, i, mlp_names=True):
        """Per-net state dict on the CPU: pc_common.MLP keys (l1.weight ... l3.bias) for kind 'mlp' with depth 2 if mlp_names, else
        this class's names."""
        st = {k: v[i].detach().cpu().clone() for k, v in self.p.items()}
        if mlp_names and self.kind == "mlp" and self.depth == 2:
            st = {"l1.weight": st["W1"], "l1.bias": st["b1"], "l2.weight": st["W2"], "l2.bias": st["b2"], "l3.weight": st["W3"], "l3.bias": st["b3"]}
        return st

    def to_mlp(self, i):
        """pc_common.MLP (CPU) holding net i (kind 'mlp', depth 2, tanh, identity coding, 17 slots)."""
        if self.kind != "mlp" or self.depth != 2 or self.act != "tanh":
            raise ValueError("to_mlp needs a depth-2 tanh one-hot net")
        m = PCC.MLP(self.p["W1"].shape[2], self.W, P, param=self.param, base_width=BASE)
        m.load_state_dict(self.state(i)); return m


def _rest(nets, gi, z1, ops=None, want=False):
    """Layers after the first pre-activation for groups of chains: z1 [G, k, N, W] of nets gi (index tensor [G]; None = all B nets,
    G = B); ops already regrouped to [G, k, ...]. Returns logits [G, k, N, 17] (and h1, h_last if want)."""
    G, k, N, W = z1.shape
    take = (lambda t: t) if gi is None else (lambda t: t[gi])
    if ops is not None:
        z1 = ops.on_z1(z1)
    h = nets.f(z1)
    if ops is not None:
        h = ops.on_h1(h)
    h1 = h
    for name in nets.hidden():
        Wn, bn = take(nets.p[name]), take(nets.p["b" + name[1:]])
        h = nets.f(torch.baddbmm(bn.unsqueeze(1), h.reshape(G, k * N, W), Wn.transpose(1, 2))).view(G, k, N, W)
    if ops is not None:
        h = ops.on_h2(h)
    W3, b3 = take(nets.p["W3"]), take(nets.p["b3"])
    lg = (torch.baddbmm(b3.unsqueeze(1), h.reshape(G, k * N, W), W3.transpose(1, 2)) * nets.mult).view(G, k, N, P)
    return (lg, h1, h) if want else lg


# ====================================================================================================================== interventions
class Ops:
    """Interventions inside the forward pass (per chain; tensors with leading dim C or broadcastable 1):
      z1 <- z1 - z1_lam ((z1 - z1_ref) z1_U) z1_U^T + z1_shift      block (lam 1) / reflection (lam 2) / partial removal (OD-C2) / push
      h1 <- h1 - h1_lam ((h1 - h1_mu) h1_Q) h1_Q^T                    class-mean or random-subspace erasure at h1 (after tanh)
      h2 <- h2 - h2_lam ((h2 - h2_mu) h2_Q) h2_Q^T                    the same at the last hidden layer
    U / Q: [C, W, r] with orthonormal columns (zero columns allowed as padding); ref / mu / shift: [C, W]; lam: float or [C]."""

    KEYS = ("z1_U", "z1_ref", "z1_lam", "z1_shift", "h1_Q", "h1_mu", "h1_lam", "h2_Q", "h2_mu", "h2_lam")

    def __init__(self, **kw):
        for k in self.KEYS:
            v = kw.pop(k, None)
            if v is not None and not isinstance(v, (float, int)):
                v = torch.as_tensor(v, dtype=F32, device=DEV)
            setattr(self, k, v)
        if kw:
            raise TypeError(f"unknown ops {list(kw)}")

    def any(self, layer=None):
        names = {"z1": ("z1_U", "z1_shift"), "h1": ("h1_Q",), "h2": ("h2_Q",)}
        keys = sum(names.values(), ()) if layer is None else names[layer]
        return any(getattr(self, k) is not None for k in keys)

    def regroup(self, idx):
        """Index every tensor (leading dim C) by idx [G, k] -> [G, k, ...]; size-1 leading dims broadcast."""
        out = Ops()
        for k in self.KEYS:
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                v = v.expand(int(idx.max()) + 1, *v.shape[1:]) if v.shape[0] == 1 else v
                v = v[idx]
            setattr(out, k, v)
        return out

    def view_per_net(self, B):
        """For whole-net evaluation (Nets.logits): leading dim B -> [B, 1, ...]."""
        return self.regroup(torch.arange(B, device=DEV)[:, None])

    def expand(self, net_of):
        """Per-net ops (leading dim B or 1) -> per-chain ops for chains of nets net_of [C]."""
        out = Ops(); no = torch.as_tensor(np.asarray(net_of), device=DEV)
        for k in self.KEYS:
            v = getattr(self, k)
            setattr(out, k, (v.expand(int(no.max()) + 1, *v.shape[1:]) if v.shape[0] == 1 else v)[no] if isinstance(v, torch.Tensor) else v)
        return out

    @staticmethod
    def _lam(lam, z):
        if lam is None:
            return 1.0
        if isinstance(lam, (float, int)):
            return float(lam)
        return lam.view(*lam.shape, *([1] * (z.dim() - lam.dim())))

    def on_z1(self, z):
        if self.z1_U is not None:
            ref = 0.0 if self.z1_ref is None else self.z1_ref.unsqueeze(-2)
            z = z - self._lam(self.z1_lam, z) * (((z - ref) @ self.z1_U) @ self.z1_U.transpose(-1, -2))
        if self.z1_shift is not None:
            z = z + self.z1_shift.unsqueeze(-2)
        return z

    def _erase(self, h, Q, mu, lam):
        if Q is None:
            return h
        m = 0.0 if mu is None else mu.unsqueeze(-2)
        return h - self._lam(lam, h) * (((h - m) @ Q) @ Q.transpose(-1, -2))

    def on_h1(self, h):
        return self._erase(h, self.h1_Q, self.h1_mu, self.h1_lam)

    def on_h2(self, h):
        return self._erase(h, self.h2_Q, self.h2_mu, self.h2_lam)

    def jac_t(self, e, layer):
        """Transpose Jacobian of the erasure at layer ('h1' / 'h2') applied to e (I - lam Q Q^T, symmetric)."""
        Q, lam = (self.h1_Q, self.h1_lam) if layer == "h1" else (self.h2_Q, self.h2_lam)
        if Q is None:
            return e
        return e - self._lam(lam, e) * ((e @ Q) @ Q.transpose(-1, -2))


def z1_ref(nets, task_idx=None):
    """ref of W8's block: the mean z1 over all inputs of the pretraining tasks (all x, y pairs x the tasks task_idx; default the first
    12 task indices for one-hot / bneck nets, shift values pre for circle / impose nets must be passed). Returns [B, W]."""
    Ex, Ey = nets.x_cols(), nets.y_cols(); T = nets.task_cols()
    if task_idx is None:
        task_idx = list(range(NPRE))
    ti = torch.as_tensor(np.asarray(task_idx), device=nets.device)
    if ti.dim() == 1:
        Tm = T[:, :, ti].mean(-1)
    else:
        Tm = torch.stack([T[b, :, ti[b]].mean(-1) for b in range(nets.B)])
    return Ex.mean(-1) + Ey.mean(-1) + Tm + nets.p["b1"]


def block_ops(U, ref, lam=1.0):
    """Ops for block(U, ref) at z1 (W8): remove the U-part of z1 - ref (lam 1), reflection (lam 2) or partial removal (lam in (0, 1])."""
    return Ops(z1_U=U, z1_ref=ref, z1_lam=lam)


def random_subspaces(W, r, n, seed):
    """n random isotropic r-dim subspaces of R^W: [n, W, r] orthonormal (QR of standard normal matrices from torch.Generator(seed))."""
    g = torch.Generator().manual_seed(int(seed))
    A = torch.randn(n, W, r, generator=g, dtype=F64)
    return torch.linalg.qr(A)[0].to(F32).to(DEV)


def random_in_span(Q, r, n, seed):
    """n random r-dim subspaces inside span(Q) (Q [W, d] orthonormal): [n, W, r]."""
    g = torch.Generator().manual_seed(int(seed)); d = Q.shape[1]
    R = torch.linalg.qr(torch.randn(n, d, r, generator=g, dtype=F64))[0]
    return (Q.double().cpu() @ R).to(F32).to(DEV)


# ====================================================================================================================== optimiser
class AdamWB:
    """Batched AdamW over a dict of [B, ...] parameters, per net equal to torch.optim.AdamW's single-tensor step (pc_meta.make_opt):
    group lr (muP: hidden W x W weights lr x BASE / W, every other tensor lr; SP: lr), decoupled decay with lr x weight_decay =
    DECAY x wd_scale per update in every group, betas (0.9, 0.98), eps 1e-8. Per-net lr and wd_scale; masked steps leave inactive nets
    (and their moments and step counts) untouched. weight_decay=False gives torch.optim.Adam (column searches)."""

    def __init__(self, params, lr, wd_scale=None, hidden=(), W=None, mup=True, decay=DECAY, betas=BETAS, eps=EPS, weight_decay=True):
        self.p = params; B = next(iter(params.values())).shape[0]; dev = next(iter(params.values())).device
        lr = np.broadcast_to(np.asarray(lr, dtype=np.float64), (B,)).copy()
        wds = np.ones(B) if wd_scale is None else np.broadcast_to(np.asarray(wd_scale, dtype=np.float64), (B,)).copy()
        self.lr, self.fac = {}, {}
        for k in params:
            lk = lr * BASE / W if (mup and k in hidden) else lr
            self.lr[k] = torch.as_tensor(lk, dtype=F64, device=dev)
            if weight_decay:
                f = np.array([1 - a * (decay * w / a) for a, w in zip(lk, wds)])
                self.fac[k] = torch.as_tensor(f, dtype=F32, device=dev)
        self.wd = weight_decay; self.b1, self.b2 = betas; self.eps = eps
        self.m = {k: torch.zeros_like(v) for k, v in params.items()}; self.v = {k: torch.zeros_like(v) for k, v in params.items()}
        self.step_n = np.zeros(B, dtype=np.int64)

    def state(self):
        return dict(m={k: v.clone() for k, v in self.m.items()}, v={k: v.clone() for k, v in self.v.items()}, step=self.step_n.copy())

    def load(self, st):
        self.m = {k: v.clone() for k, v in st["m"].items()}; self.v = {k: v.clone() for k, v in st["v"].items()}
        self.step_n = np.asarray(st["step"]).copy()

    @torch.no_grad()
    def step(self, grads, active=None):
        """grads: dict of [B, ...]; active: None (all) or bool numpy array [B]."""
        allon = active is None or bool(np.all(active))
        if allon:
            self.step_n += 1
        else:
            self.step_n[active] += 1
        ss = self.step_n if allon else self.step_n[active]
        same = bool(np.all(ss == ss[0]))
        if same:
            st = float(ss[0]); bc1 = 1 - self.b1 ** st; bc2s = (1 - self.b2 ** st) ** 0.5
        else:
            sf = self.step_n.astype(np.float64); bc1v = 1 - self.b1 ** sf; bc2v = (1 - self.b2 ** sf) ** 0.5
        msk = None if allon else torch.as_tensor(active, device=next(iter(self.p.values())).device)
        for k, p in self.p.items():
            g = grads[k]; shape = (p.shape[0],) + (1,) * (p.dim() - 1)
            if same:
                ssz = (-(self.lr[k] / bc1)).to(F32).view(shape); den_div = bc2s
            else:
                ssz = (-(self.lr[k] / torch.as_tensor(bc1v, device=p.device))).to(F32).view(shape)
                den_div = torch.as_tensor(bc2v, dtype=F32, device=p.device).view(shape)
            if allon:
                if self.wd:
                    p.mul_(self.fac[k].view(shape))
                self.m[k].lerp_(g, 1 - self.b1)
                self.v[k].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                denom = (self.v[k].sqrt() / den_div).add_(self.eps)
                p.add_(ssz * self.m[k] / denom)
            else:
                mm = msk.view(shape)
                pn = p * self.fac[k].view(shape) if self.wd else p.clone()
                m_n = self.m[k].lerp(g, 1 - self.b1)
                v_n = (self.v[k] * self.b2).addcmul_(g, g, value=1 - self.b2)
                denom = (v_n.sqrt() / den_div).add_(self.eps)
                pn = pn + ssz * m_n / denom
                p.copy_(torch.where(mm, pn, p)); self.m[k].copy_(torch.where(mm, m_n, self.m[k])); self.v[k].copy_(torch.where(mm, v_n, self.v[k]))


def _adam_q(q, g, m, v, step, lr, b1=BETAS[0], b2=BETAS[1], eps=EPS):
    """One torch.optim.Adam step (no weight decay) on q [..., r] with per-chain lr (tensor [..., 1] or float)."""
    m.lerp_(g, 1 - b1); v.mul_(b2).addcmul_(g, g, value=1 - b2)
    bc1 = 1 - b1 ** float(step); bc2s = (1 - b2 ** float(step)) ** 0.5
    denom = (v.sqrt() / bc2s).add_(eps)
    if isinstance(lr, float):
        q.addcdiv_(m, denom, value=-(lr / bc1))
    else:
        q.add_((-(lr / bc1)).to(F32) * m / denom)


# ====================================================================================================================== training
def task_fisher_b(H, lab, nk, groups=None):
    """Batched pc_meta.task_fisher: H [B, N, D], integer labels lab [B, N] (nk classes) -> tr(S_between) / tr(S_within) [B]
    (differentiable; empty classes contribute nothing, as in pc_meta). groups: the index tensors of the non-empty classes when every
    net of the batch has the same labels; then pc_meta's op sequence (class loop, mean, sum) runs batched over the nets, which gives
    values and gradients bitwise equal to pc_meta.task_fisher per net on the CPU (W0's regulariser equivalence). Without groups a one-hot
    formulation is used (equal up to float32 rounding; the sham test of W0 checks the value to 1e-6)."""
    if groups is not None:
        mu = H.mean(1); Bt = H.new_zeros(H.shape[0]); Wt = H.new_zeros(H.shape[0])
        for idx in groups:
            Hk = H[:, idx]; mk = Hk.mean(1)
            Bt = Bt + len(idx) * ((mk - mu) ** 2).sum(-1); Wt = Wt + ((Hk - mk.unsqueeze(1)) ** 2).sum((1, 2))
        return Bt / Wt
    O = Fn.one_hot(lab, nk).to(H.dtype)
    n = O.sum(1)
    mk = (O.transpose(1, 2) @ H) / n.clamp(min=1).unsqueeze(-1)
    mu = H.mean(1, keepdim=True)
    Bt = (n * ((mk - mu) ** 2).sum(-1)).sum(-1)
    Wt = ((H - O @ mk) ** 2).sum((1, 2))
    return Bt / Wt


def _stack_data(nets, tss):
    """Device tensors of a list of task sets (one per net; equal sizes)."""
    def T(key):
        return torch.as_tensor(np.stack([ts[key] for ts in tss]), device=nets.device)
    if len({len(ts["x"]) for ts in tss}) != 1 or len({len(ts["vx"]) for ts in tss}) != 1:
        raise ValueError("train(): every net of a batch needs the same training / validation sizes")
    d = dict(X=nets.features(T("x"), T("y"), T("t")), lab=T("lab"), task=T("task"), xlab=T("x"),
             Xv=nets.features(T("vx"), T("vy"), T("vt")), vlab=T("vlab"))
    ts0 = tss[0]; d["val_mode"] = ts0["val_mode"]; d["ntask"] = ts0["ntask"]
    d["vgroup_of_task"] = np.array([ts0["vgroup"][ts0["vtask"] == k][0] for k in range(ts0["ntask"])])
    d["K"] = int(max(ts["task"].max() for ts in tss)) + 1
    for key, src_, nk in (("task", "task", d["K"]), ("xlab", "x", P)):     # nets grouped by identical label rows (regulariser)
        rows = np.stack([ts[src_] for ts in tss]); uniq, inv = np.unique(rows, axis=0, return_inverse=True)
        d[key + "_groups"] = [(np.nonzero(inv.reshape(-1) == u)[0],
                               [torch.as_tensor(np.nonzero(uniq[u] == k)[0], device=nets.device) for k in range(nk) if (uniq[u] == k).any()])
                              for u in range(len(uniq))]
    return d


@torch.no_grad()
def _val_acc(nets, d, ops=None):
    """Level-1 accuracy per net (pc_meta: pooled float32 mean; W3 / W5: mean of per-task accuracies; W4: min over the two families of
    the per-task means). Returns (level accuracy float64 [B], pooled accuracy [B])."""
    pred = nets.logits(d["Xv"], ops).argmax(-1)
    corr = (pred == d["vlab"]).float()
    pooled = corr.mean(1).double().cpu().numpy()
    if d["val_mode"] == "pooled":
        return pooled, pooled
    B = corr.shape[0]; per = corr.view(B, d["ntask"], -1).mean(-1).cpu().numpy().astype(np.float64)
    gs = d["vgroup_of_task"]
    lev = np.min(np.stack([per[:, gs == g].mean(1) for g in np.unique(gs)], 1), 1)
    return lev, pooled


class TrainResult:
    """Result of train(): nets (state at each net's evaluation point), rows (per-net dicts: name, spec, first95, checkpoint, level1,
    eval_point 'checkpoint' | 'maxsteps' | 'fixed', updates, final_train, final_heldout, coord_check, checksum), nets2 / has2 (second
    checkpoint first95 + AFTER + extra_after; None if not requested), first95_state (params and AdamW state at first95 for train_late),
    snapshots (list of (step, value, active mask)), data (the stacked training data). first95 is recorded in every mode (pc_meta records
    none for its K = 1 controls, and level1 False there); level1 of a fixed-length run = level-1 accuracy >= LEVEL1 at its end."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def train(specs, *, fixed_steps=None, init_state=None, nets=None, tasksets=None, max_steps=None, after=None, level1=None,
          extra_after=0, save_first95=False, snapshot_every=None, snapshot=None, task_basis=None, coord_check=True, fisher="auto",
          verbose=False):
    """Batched full-batch AdamW training (W0 / W1; definitions identical to pc_meta.main).
    specs: list of net specs (one architecture; equal training-set sizes). Mode (per batch):
      - checkpoint rule (default): every 50 updates the level-1 accuracy (pooled held-out for W1) is checked; first95 = the first check
        with accuracy >= LEVEL1; the net stops at first95 + after (checkpoint, eval_point 'checkpoint'); a net without level 1 stops after
        max_steps updates (eval_point 'maxsteps'). extra_after > 0: the net continues to first95 + after + extra_after (second
        checkpoint, nets2; AT2 uses SECOND_AFTER = 1000).
      - fixed_steps (int or per-net list): exactly that many updates (K = 1 controls, late branches, AT6 shared split, fresh nets);
        level1 = level-1 accuracy >= LEVEL1 at the end; eval_point 'fixed'.
    Regularisers (W2): spec gamma != 0 adds gamma log F(h2) with F = task_fisher of the training batch grouped by task ('task') or by
    the x input (17 groups, 'sham'); when 'early' = from update 1, 'late' = only after first95 (use train_late for the plan's branch
    protocol; both are the same updates).
    init_state: dict(params, opt) to continue from (late branches); nets: an initialised Nets to use instead of from_specs.
    snapshot_every / snapshot(step, nets) -> value: called at step 0 and every snapshot_every updates (values stored with the mask of
    the nets that were active in that update). task_basis: [B, W * 12, L] orthonormal bases (an all-zero basis leaves that net free)
    restricting the centred update of the 12 task columns (row-major [W, 12], as snap_task_code / trajectory_basis) after every step to
    their span (C-T6 cap; the task-mean column stays free).
    save_first95: keep each net's parameters and AdamW state at first95 (for train_late).
    coord_check: pc_meta's G0 coordinate check at update 10 (mean |h - h_init| on 512 training examples, probe from the net's seed).
    fisher: how the regulariser's Fisher ratio is computed: 'auto' (pc_meta's op sequence, bitwise per net, when all nets of the batch
    share the labels, e.g. every task regulariser of a W1 batch or a one-world sham batch; else the one-hot form), 'replica' (always
    pc_meta's op sequence, looping over groups of nets with identical labels), 'onehot'."""
    specs = [full_spec(s) for s in specs]; B = len(specs)
    max_steps = MAXSTEPS if max_steps is None else max_steps; after = AFTER if after is None else after
    lev1 = LEVEL1 if level1 is None else level1
    if nets is None:
        nets = Nets.from_specs(specs)
    if init_state is not None:
        for k in nets.p:
            nets.p[k].copy_(init_state["params"][k])
    tss = [taskset_from_spec(s) for s in specs] if tasksets is None else tasksets
    d = _stack_data(nets, tss)
    hid = nets.hidden()
    opt = AdamWB(nets.p, [s["lr"] for s in specs], [s["wd"] for s in specs], hidden=hid, W=nets.W, mup=nets.param == "mup")
    if init_state is not None:
        opt.load(init_state["opt"])
    gam = np.array([s["gamma"] for s in specs], dtype=np.float64)
    late = np.array([s["when"] == "late" for s in specs]); sham = np.array([s["reg"] == "sham" for s in specs])
    fixed = None if fixed_steps is None else np.broadcast_to(np.asarray(fixed_steps, dtype=np.int64), (B,)).copy()
    first95 = np.full(B, -1, np.int64); ck = np.full(B, -1, np.int64); done = np.zeros(B, bool); evp = [None] * B
    has2 = np.zeros(B, bool); ck2 = np.full(B, -1, np.int64)
    ck_params = {k: v.clone() for k, v in nets.p.items()}
    f95 = dict(params={k: v.clone() for k, v in nets.p.items()}, opt=opt.state()) if save_first95 else None
    smax = int(fixed.max()) if fixed is not None else max_steps + after + extra_after
    cc = [None] * B; init_p = {k: v.clone() for k, v in nets.p.items()} if coord_check else None
    probe = None
    if coord_check:
        probe = torch.stack([torch.randperm(d["X"].shape[1], generator=torch.Generator().manual_seed(
            s["seed"] if s["seed"] is not None else init_seed(s["ws"], s["init"])))[:512] for s in specs]).to(nets.device)
    snaps = []
    if snapshot_every:
        snaps.append((0, snapshot(0, nets), np.ones(B, bool)))
    if task_basis is not None:
        tb = torch.as_tensor(task_basis, dtype=F32, device=nets.device); tcol = slice(2 * nets.dx(), 2 * nets.dx() + NPRE)
        has_tb = (tb.abs().sum((1, 2)) > 0).cpu().numpy()
    pidx = list(nets.p.keys())
    tstart = time.time()
    for s in range(1, smax + 1):
        if fixed is None:
            stop = (~done) & (first95 < 0) & (s > max_steps)
            for b in np.nonzero(stop)[0]:
                done[b] = True; ck[b] = max_steps; evp[b] = "maxsteps"
                for k in pidx:
                    ck_params[k][b] = nets.p[k][b]
        active = ~done
        if not active.any():
            break
        for k in pidx:
            nets.p[k].requires_grad_(True)
        X = d["X"]
        z = nets.z1(X).unsqueeze(1)
        lg, _, h2 = _rest(nets, None, z, want=True)
        lg, h2 = lg[:, 0], h2[:, 0]
        ce = Fn.cross_entropy(lg.reshape(-1, P), d["lab"].reshape(-1), reduction="none").view(B, -1).mean(1)
        loss = ce.sum()
        on = (gam != 0) & ((~late) | (first95 >= 0))
        for sel, labk, nk in ((on & ~sham, "task", d["K"]), (on & sham, "xlab", P)):
            if not sel.any():                               # only the nets with the regulariser on (no 0 x log 0 terms)
                continue
            grp = d[labk + "_groups"]
            if fisher == "onehot" or (fisher == "auto" and len(grp) > 1):
                ii = torch.as_tensor(np.nonzero(sel)[0], device=nets.device); gg = torch.as_tensor(gam[sel], dtype=F32, device=nets.device)
                loss = loss + (gg * torch.log(task_fisher_b(h2[ii], d[labk][ii], nk))).sum()
            else:                                           # pc_meta's op sequence per group of nets with identical labels
                for nn_, cls in grp:
                    s_ = sel[nn_]
                    if s_.any():
                        ii = torch.as_tensor(nn_[s_], device=nets.device); gg = torch.as_tensor(gam[nn_[s_]], dtype=F32, device=nets.device)
                        loss = loss + (gg * torch.log(task_fisher_b(h2[ii], None, nk, cls))).sum()
        grads = torch.autograd.grad(loss, [nets.p[k] for k in pidx])
        for k in pidx:
            nets.p[k].requires_grad_(False)
        if task_basis is not None:
            T_old = nets.p["W1"][:, :, tcol].clone()
        opt.step(dict(zip(pidx, grads)), None if active.all() else active)
        if task_basis is not None:
            with torch.no_grad():
                Tn = nets.p["W1"][:, :, tcol]; dT = Tn - T_old; dm = dT.mean(-1, keepdim=True); dc = (dT - dm).reshape(B, -1, 1)
                pr = (tb @ (tb.transpose(1, 2) @ dc)).view_as(dT)
                use = torch.as_tensor(active & has_tb, device=nets.device).view(B, 1, 1)
                nets.p["W1"][:, :, tcol] = torch.where(use, T_old + dm + pr, Tn)
        if coord_check and s == 10:
            with torch.no_grad():
                bi = torch.arange(B, device=nets.device)[:, None]; Xp = X[bi, probe]
                n0 = Nets(nets.kind, nets.W, nets.depth, nets.act, nets.param, init_p, nets.fx, nets.fy, nets.ft)
                a0 = n0.hidden_acts(Xp); a1 = nets.hidden_acts(Xp)
                vals = [(b_ - a_).abs().mean((1, 2)).cpu().numpy() for a_, b_ in zip(a0, a1)]
                for b in range(B):
                    cc[b] = [float(vals[0][b]), float(vals[1][b])]
            init_p = None
        if snapshot_every and s % snapshot_every == 0:
            snaps.append((s, snapshot(s, nets), active.copy()))
        if s % CHECK_EVERY == 0:
            lev, _ = _val_acc(nets, d)
            new95 = active & (first95 < 0) & (lev >= lev1)
            if new95.any():
                first95[new95] = s
                if save_first95:
                    nb = torch.as_tensor(new95, device=nets.device); st = opt.state()
                    for k in pidx:
                        sh = (B,) + (1,) * (nets.p[k].dim() - 1)
                        f95["params"][k].copy_(torch.where(nb.view(sh), nets.p[k], f95["params"][k]))
                        f95["opt"]["m"][k].copy_(torch.where(nb.view(sh), st["m"][k], f95["opt"]["m"][k]))
                        f95["opt"]["v"][k].copy_(torch.where(nb.view(sh), st["v"][k], f95["opt"]["v"][k]))
                    f95["opt"]["step"][new95] = opt.step_n[new95]
            if verbose and s % 500 == 0:
                print(f"  step {s}: active {int(active.sum())}/{B} level acc {np.round(lev, 3).tolist()} ({time.time() - tstart:.0f}s)", flush=True)
        if fixed is None:
            reach = active & (first95 >= 0) & (ck < 0) & (s >= first95 + after)
            for b in np.nonzero(reach)[0]:
                ck[b] = s; evp[b] = "checkpoint"
                for k in pidx:
                    ck_params[k][b] = nets.p[k][b]
                if extra_after <= 0:
                    done[b] = True
            if extra_after > 0:
                fin = active & (ck >= 0) & (s >= first95 + after + extra_after)
                done[fin] = True; has2[fin] = True; ck2[fin] = s
        else:
            fin = active & (s >= fixed)
            done[fin] = True; ck[fin] = s
            for b in np.nonzero(fin)[0]:
                evp[b] = "fixed"
                for k in pidx:
                    ck_params[k][b] = nets.p[k][b]
    if fixed is None:
        for b in np.nonzero(~done)[0]:          # cannot happen with smax as above; kept for safety
            done[b] = True; ck[b] = smax; evp[b] = "maxsteps" if first95[b] < 0 else "checkpoint"
            for k in pidx:
                ck_params[k][b] = nets.p[k][b]
    nets2 = None
    if extra_after > 0:
        nets2 = Nets(nets.kind, nets.W, nets.depth, nets.act, nets.param, {k: v.detach().clone() for k, v in nets.p.items()},
                     nets.fx, nets.fy, nets.ft, list(nets.meta))
    out = Nets(nets.kind, nets.W, nets.depth, nets.act, nets.param, ck_params, nets.fx, nets.fy, nets.ft, list(nets.meta))
    lev, pooled = _val_acc(out, d)
    with torch.no_grad():
        trn = (out.logits(d["X"]).argmax(-1) == d["lab"]).float().mean(1).cpu().numpy()
    rows = []
    for b, s in enumerate(specs):
        lv = bool(first95[b] >= 0) if fixed is None else bool(lev[b] >= lev1)
        rows.append(dict(name=net_name(s), spec=out.meta[b]["spec"], first95=int(first95[b]) if first95[b] >= 0 else None,
                         checkpoint=int(ck[b]), level1=lv, eval_point=evp[b], updates=int(ck[b]), final_train=float(trn[b]),
                         final_heldout=float(pooled[b]), level_acc=float(lev[b]), coord_check=cc[b],
                         second=int(ck2[b]) if has2[b] else None, checksum=weight_checksum(out, b)))
        out.meta[b].update(row=rows[-1])
    return TrainResult(nets=out, rows=rows, nets2=nets2, has2=has2, first95_state=f95, snapshots=snaps, data=d)


def train_late(nat, idx, branch_specs, n_updates=None, level1=None):
    """W2 late branches: for each (natural net index i, spec) the branch starts from nat.first95_state of net i (parameters and AdamW
    state at first95; train(..., save_first95=True)) and runs n_updates (AFTER) updates with the spec's regulariser on; level 1 =
    held-out >= level1 (default LEVEL1) at the branch's end. Only nets with first95 can branch (others raise)."""
    n_updates = AFTER if n_updates is None else n_updates
    if nat.first95_state is None:
        raise ValueError("train_late needs train(..., save_first95=True)")
    for i in idx:
        if nat.rows[i]["first95"] is None:
            raise ValueError(f"net {i} has no first95")
    it = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=nat.nets.device)
    st = dict(params={k: v[it].clone() for k, v in nat.first95_state["params"].items()},
              opt=dict(m={k: v[it].clone() for k, v in nat.first95_state["opt"]["m"].items()},
                       v={k: v[it].clone() for k, v in nat.first95_state["opt"]["v"].items()},
                       step=np.asarray(nat.first95_state["opt"]["step"])[np.asarray(idx)].copy()))
    specs = [dict(full_spec(s), when="early") for s in branch_specs]       # the regulariser is on for every branch update
    res = train(specs, fixed_steps=n_updates, init_state=st, coord_check=False, level1=level1)
    for r, s in zip(res.rows, branch_specs):
        r["spec"]["when"] = full_spec(s)["when"]; r["name"] = net_name(s)
    for b, s in enumerate(branch_specs):
        res.nets.meta[b]["spec"] = full_spec(s); res.nets.meta[b]["name"] = net_name(s)
    return res


def param_distance(a, ia, b, ib):
    """Relative parameter distance ||theta_a - theta_b|| / ||theta_b|| over all parameters concatenated (ref crit/sham.pdist)."""
    num = sum(float(((a.p[k][ia] - b.p[k][ib]).double() ** 2).sum()) for k in a.p)
    den = sum(float((b.p[k][ib].double() ** 2).sum()) for k in b.p)
    return math.sqrt(num) / math.sqrt(den)


def snap_task_code(s, nets):
    """C-T6 snapshot: the 12 pretraining task columns centred over tasks, [B, W * 12] (row-major [W, 12], ref c4_traj)."""
    T = nets.task_cols()[:, :, :NPRE].detach().double()
    return (T - T.mean(-1, keepdim=True)).reshape(nets.B, -1).cpu()


# ====================================================================================================================== evaluation
class _Groups:
    """Chains grouped by net: idx [G, kmax] chain index (padding repeats the group's first chain), valid [G, kmax], gi [G] net index."""

    def __init__(self, net_of):
        net_of = np.asarray(net_of, dtype=np.int64); C = len(net_of)
        order = np.argsort(net_of, kind="stable"); sn = net_of[order]
        uniq, start, cnt = np.unique(sn, return_index=True, return_counts=True)
        G, kmax = len(uniq), int(cnt.max()); gof = np.repeat(np.arange(G), cnt); rank = np.arange(C) - start[gof]
        idx = np.full((G, kmax), -1, np.int64); idx[gof, rank] = order; valid = idx >= 0
        idx = np.where(valid, idx, idx[:, :1]); pos = np.zeros(C, np.int64); pos[order] = gof * kmax + rank
        self.C, self.G, self.kmax = C, G, kmax
        self.idx = torch.as_tensor(idx, device=DEV); self.valid = torch.as_tensor(valid, device=DEV); self.gi = torch.as_tensor(uniq, device=DEV)
        self.pos = torch.as_tensor(pos, device=DEV)

    def grp(self, t):
        t = torch.as_tensor(t, device=DEV)
        return t[self.idx]

    def flat(self, t):
        return t.reshape(self.G * self.kmax, *t.shape[2:])[self.pos]


def _xyl(arr, C):
    """(x, y, lab) per chain: arrays [C, N] or [N] (shared) -> int64 tensors [C, N]."""
    out = []
    for a in arr:
        a = torch.as_tensor(np.asarray(a), dtype=torch.int64, device=DEV)
        out.append(a.expand(C, -1) if a.dim() == 1 else a)
    return out


def _base(nets, gi, x, y):
    """base_z1 = Ex[:, x] + Ey[:, y] + b1 for x, y [G, k, N] of nets gi -> [G, k, N, W]."""
    G, k, N = x.shape
    ExT = nets.x_cols()[gi].transpose(1, 2); EyT = nets.y_cols()[gi].transpose(1, 2)
    W = ExT.shape[2]
    gx = ExT.gather(1, x.reshape(G, k * N, 1).expand(-1, -1, W)); gy = EyT.gather(1, y.reshape(G, k * N, 1).expand(-1, -1, W))
    return (gx + gy + nets.p["b1"][gi].unsqueeze(1)).view(G, k, N, W)


def _chunks(G, k, N, W):
    per = max(1, CHUNK // max(1, k * N * W))
    return [(a, min(G, a + per)) for a in range(0, G, per)]


@torch.no_grad()
def _acc_grouped(nets, gr, v, x, y, lab, ops):
    """Accuracies [G, kmax] of columns v [G, kmax, W] on sets x, y, lab [G, kmax, N] (ops regrouped)."""
    G, k, N = x.shape; out = []
    for a, b in _chunks(G, k, N, nets.W):
        gi = gr.gi[a:b]
        z = _base(nets, gi, x[a:b], y[a:b]) + v[a:b].unsqueeze(2)
        o = None if ops is None else _slice_ops(ops, a, b)
        out.append((_rest(nets, gi, z, o).argmax(-1) == lab[a:b]).float().mean(-1))
    return torch.cat(out)


def _slice_ops(ops, a, b):
    o = Ops()
    for k in Ops.KEYS:
        v = getattr(ops, k)
        setattr(o, k, v[a:b] if isinstance(v, torch.Tensor) else v)
    return o


@torch.no_grad()
def cols_accuracy(nets, net_of, cols, sets, ops=None):
    """Held-out accuracy of candidate columns: cols [C, W] (each replaces the task column of net net_of[c]) on sets = (x, y, lab) with
    arrays [C, N] or [N] (shared); ops per chain. Returns numpy [C] (float32 means as pc_common.accuracy)."""
    C = len(net_of); gr = _Groups(net_of); x, y, lab = (gr.grp(a) for a in _xyl(sets, C))
    v = gr.grp(torch.as_tensor(cols, dtype=F32, device=DEV))
    return gr.flat(_acc_grouped(nets, gr, v, x, y, lab, None if ops is None else ops.regroup(gr.idx))).cpu().numpy()


@torch.no_grad()
def predictions(nets, net_of, cols, sets, ops=None):
    """Predicted classes [C, N] (numpy) of candidate columns (as cols_accuracy)."""
    C = len(net_of); gr = _Groups(net_of); x, y, _ = (gr.grp(a) for a in _xyl((sets[0], sets[1], sets[0]), C))
    v = gr.grp(torch.as_tensor(cols, dtype=F32, device=DEV)); o = None if ops is None else ops.regroup(gr.idx)
    z = _base(nets, gr.gi, x, y) + v.unsqueeze(2)
    return gr.flat(_rest(nets, gr.gi, z, o).argmax(-1)).cpu().numpy()


@torch.no_grad()
def hidden_states(nets, net_of, cols, sets, ops=None):
    """(h1, h_last) [C, N, W] (float32 tensors) of inputs (x, y) with columns cols [C, W], after ops (probe features, W8)."""
    C = len(net_of); gr = _Groups(net_of); x, y = (gr.grp(a) for a in _xyl((sets[0], sets[1]), C))
    v = gr.grp(torch.as_tensor(cols, dtype=F32, device=DEV)); o = None if ops is None else ops.regroup(gr.idx)
    _, h1, h2 = _rest(nets, gr.gi, _base(nets, gr.gi, x, y) + v.unsqueeze(2), o, want=True)
    return gr.flat(h1), gr.flat(h2)


def task_accuracy(nets, ws_of, tasks=None, cols=None, ops=None, ky_of=None, ho=True):
    """Held-out accuracy of every net on its old tasks (W1 one-hot nets): [B, n_tasks]. tasks: task (slot) indices (default 0-11);
    cols: optional replacement columns [B, n_tasks, W] (edit); ops: per-net Ops (leading dim B) applied to every task."""
    tasks = list(range(NPRE)) if tasks is None else list(tasks); B = nets.B
    net_of = np.repeat(np.arange(B), len(tasks)); T = nets.task_cols()
    C = torch.stack([T[b, :, tasks].T for b in range(B)]) if cols is None else torch.as_tensor(cols, dtype=F32, device=DEV)
    xs, ys, ls = [], [], []
    for b in range(B):
        pre, _, _, splits = M.world(ws_of[b]); ky = 1 if ky_of is None else ky_of[b]
        for k in tasks:
            x, y = _pairs_of(splits[k], 144 if ho else 0, P * P if ho else 144); xs.append(x); ys.append(y); ls.append(_label(x, y, pre[k], ky))
    a = cols_accuracy(nets, net_of, C.reshape(-1, nets.W), (np.stack(xs), np.stack(ys), np.stack(ls)), None if ops is None else ops.expand(net_of))
    return a.reshape(B, len(tasks))


def zero_shot(nets, ws_of, shifts_of, mode="circle", ky=1):
    """Zero-shot accuracy of a net's own task input for shift values it was not trained on (W3 / W5): the column of shift c is the net's
    task column with task index c (circle: Ac f(c); impose: t0 + A phi(c)); held-out pairs: mode 'circle' = splits[c mod 16][8:] of
    world(ws) (ref c3_angle), mode 'impose' = the new task j's split[8:] for c = new[j] (ref impose2). Returns per net a dict c ->
    dict(acc, modal shift (pred - x - y) mod 17, modal share). Usable as a snapshot callback (C-T3 hatching)."""
    T = nets.task_cols(); out = []
    for b in range(nets.B):
        pre, new, _, splits = M.world(ws_of[b]); cs = list(shifts_of[b]); xs, ys = [], []
        for c in cs:
            sp = splits[c % len(splits)] if mode == "circle" else splits[NPRE + list(new).index(c)]
            x, y = _pairs_of(sp, NSHOT, P * P); xs.append(x); ys.append(y)
        cols = T[b][:, torch.as_tensor(cs, device=nets.device)].T
        pr = predictions(nets, np.full(len(cs), b), cols, (np.stack(xs), np.stack(ys))); d_ = {}
        for i, c in enumerate(cs):
            e = (pr[i] - xs[i] - ky * ys[i]) % P; cnt = np.bincount(e, minlength=P)
            d_[int(c)] = dict(acc=float((e == c % P).mean()), modal=int(cnt.argmax()), share=float(cnt.max() / len(e)))
        out.append(d_)
    return out


# ====================================================================================================================== W7 search
def anchored_param(nets, net_of, anchor_idx=None):
    """W7 anchored search: v = mean(C) + (C - mean(C)) alpha over the net's anchor columns C = task_cols()[:, :, anchor_idx]
    (default the 12 pretraining slots). Returns (v0 [C, W], Bm [C, W, n]) for column_search(param=...)."""
    anchor_idx = list(range(NPRE)) if anchor_idx is None else list(anchor_idx)
    T = nets.task_cols()[:, :, anchor_idx]; mA = T.mean(-1); Ct = T - mA.unsqueeze(-1)
    no = torch.as_tensor(np.asarray(net_of), device=DEV)
    return mA[no], Ct[no]


def coords_param(nets, net_of, which):
    """Coordinate searches: 'bneck' (v = U e, e in R^dt; ref rgi_bneck) or 'impose' (v = t0 + A q, q in R^2k; ref impose2 code-space
    search). Returns (v0, Bm)."""
    no = torch.as_tensor(np.asarray(net_of), device=DEV)
    if which == "bneck":
        U = nets.p["U"][no]; return torch.zeros(U.shape[0], U.shape[1], device=DEV), U
    if which == "impose":
        W1 = nets.p["W1"][no]; return W1[:, :, -1], W1[:, :, 2 * P:-1]
    raise ValueError(which)


def column_search(nets, net_of, shots, heldout=None, *, steps=None, lr=FT_LR, eval_at=(), val=None, budgets=(), val_every=VAL_EVERY,
                  param=None, q0=None, keep=None, remove=None, ops=None, loss="ce", target=None, pc=None, keep_cols=False):
    """W7: the one column-search function (vectorised over chains, grouped by net; one host sync per search).
    nets, net_of [C]: chain c searches a column of net net_of[c] (shared or per-chain nets).
    shots = (x, y, lab) [C, Ns] (lab unused for loss 'kl' / 'mse'); heldout = dict name -> (x, y, lab) [C, N] or [N] (or one triple,
      named 'ho'); val = (x, y, lab) validation set for best-of-budget.
    Parameterisation: free (v = q from zero; pc_meta.finetune 'col') or param = (v0 [C, W], Bm [C, W, r]): v = v0 + Bm q, q from zero
      (anchored_param: alpha over the centred anchors; coords_param: bneck / impose code space); q0 = start (warm start / recipe).
    Restriction (free param only): keep = [C, W, r] (v projected on span after every step) or remove = [C, W, r] (projected on the
      orthogonal complement after every step; C-T5).
    ops: Ops per chain, on during the search and in every evaluation (block, erasure, push).
    loss: 'ce' (mean over shots; pc_meta), 'kl' (batchmean KL(target || model), target = log-probs [C, Ns, 17]; T4) or 'mse' (sum over
      units, mean over inputs of (h_last - target)^2, target [C, Ns, W]; T3).
    Optimiser: Adam (betas 0.9, 0.98, eps 1e-8, no decay), lr float or [C] (e.g. 1e-2 / 1e-1 for OD-C11), steps = max(eval_at,
      budgets) unless given.
    pc: None (backprop) or dict(eta [C] or float, T=20, beta=0.1): PC local update (pc_meta.pc_update with mask_col: T relaxation steps
      of h1, h2 from the feedforward values with eta_x, then the column gradient of the energy; h1 / h2 erasers of ops sit inside the
      energy; loss 'kl' uses the soft target in the output term). Evaluation always by the feedforward pass.
    eval_at: steps at which every held-out set is evaluated (B2: FT_EVAL; 0 = before the first step, so steps=0 evaluates q0 =
      'recipe'). budgets: best-of-budget (OD-C1): validation and held-out accuracy every val_every steps up to max(budgets); per budget
      the held-out accuracy at the best validation evaluation within the budget (ties: earliest).
    Returns dict: acc {name: [C, len(eval_at)]}, A {name: mean over eval_at}, best {budget: dict(val [C] (best validation accuracy),
      step [C] (its step), heldout {name: [C]} (held-out accuracy there), col [C, W] (the column there; keep_cols=True))}, v [C, W] and
      q [C, r] at the end (numpy)."""
    C = len(net_of); gr = _Groups(net_of); G, k = gr.G, gr.kmax
    nets = nets.subset(gr.gi.cpu().numpy()); gr.gi = torch.arange(G, device=DEV)      # weights gathered once per search
    if heldout is None:
        heldout = {}
    elif isinstance(heldout, tuple):
        heldout = {"ho": heldout}
    eval_at = tuple(eval_at); budgets = tuple(budgets)
    steps = max(eval_at + budgets + (0,)) if steps is None else steps
    sx, sy, sl = (gr.grp(a) for a in _xyl(shots, C))
    hos = {n: tuple(gr.grp(a) for a in _xyl(t, C)) for n, t in heldout.items()}
    vs = None if val is None else tuple(gr.grp(a) for a in _xyl(val, C))
    if budgets and vs is None:
        raise ValueError("budgets need a validation set")
    og = None if ops is None else ops.regroup(gr.idx)
    if param is not None:
        v0 = gr.grp(torch.as_tensor(param[0], dtype=F32, device=DEV)); Bm = gr.grp(torch.as_tensor(param[1], dtype=F32, device=DEV))
        r = Bm.shape[-1]
    else:
        v0 = Bm = None; r = nets.W
    q = torch.zeros(G, k, r, device=DEV) if q0 is None else gr.grp(torch.as_tensor(q0, dtype=F32, device=DEV)).clone()
    proj = None
    if keep is not None or remove is not None:
        if param is not None:
            raise ValueError("keep / remove need the free parameterisation")
        proj = (gr.grp(torch.as_tensor(keep if keep is not None else remove, dtype=F32, device=DEV)), keep is not None)
    lr_t = float(lr) if np.ndim(lr) == 0 else gr.grp(torch.as_tensor(np.asarray(lr, dtype=np.float64), device=DEV)).unsqueeze(-1)
    vmask = gr.valid.to(F32)
    base_s = _base(nets, gr.gi, sx, sy)
    tgt = None if target is None else gr.grp(torch.as_tensor(target, dtype=F32, device=DEV))
    if pc is not None:
        if nets.depth != 2 or nets.act != "tanh":
            raise ValueError("PC searches need depth-2 tanh nets")
        if og is not None and og.any("z1"):
            raise ValueError("PC searches support h1 / h2 erasers only")
        eta = pc["eta"]; eta = gr.grp(torch.as_tensor(np.broadcast_to(np.asarray(eta, dtype=np.float64), (C,)).copy(), dtype=F32, device=DEV))
        Y = Fn.one_hot(sl, P).to(F32) if loss == "ce" else (tgt.exp() if loss == "kl" else None)
        if Y is None:
            raise ValueError("PC searches: loss 'ce' or 'kl'")
    m1, m2 = torch.zeros_like(q), torch.zeros_like(q)

    def col(qq):
        return qq if v0 is None else v0 + (Bm @ qq.unsqueeze(-1)).squeeze(-1)

    acc = {n: [] for n in hos}; best = {b: dict(val=torch.full((G, k), -1.0, device=DEV), step=torch.zeros(G, k, device=DEV),
                                                heldout={n: torch.zeros(G, k, device=DEV) for n in hos}) for b in budgets}
    if keep_cols:
        for b in budgets:
            best[b]["col"] = torch.zeros(G, k, nets.W, device=DEV)
    last_val = max(budgets) if budgets else 0

    def evaluate(s):
        with torch.no_grad():
            v = col(q)
            if s in eval_at:
                for n, (x, y, l_) in hos.items():
                    acc[n].append(_acc_grouped(nets, gr, v, x, y, l_, og))
            if budgets and s > 0 and s % val_every == 0 and s <= last_val:
                va = _acc_grouped(nets, gr, v, *vs, og); hv = {n: _acc_grouped(nets, gr, v, x, y, l_, og) for n, (x, y, l_) in hos.items()}
                for b in budgets:
                    if s <= b:
                        upd = va > best[b]["val"]
                        best[b]["val"] = torch.where(upd, va, best[b]["val"]); best[b]["step"] = torch.where(upd, torch.full_like(va, s), best[b]["step"])
                        for n in hos:
                            best[b]["heldout"][n] = torch.where(upd, hv[n], best[b]["heldout"][n])
                        if keep_cols:
                            best[b]["col"] = torch.where(upd.unsqueeze(-1), v, best[b]["col"])

    evaluate(0)
    for s in range(1, steps + 1):
        if pc is None:
            qq = q.detach().requires_grad_(True); v = col(qq); z = base_s + v.unsqueeze(2)
            if loss == "mse":
                _, _, h = _rest(nets, None, z, og, want=True); L = ((h - tgt) ** 2).sum(-1).mean(-1)
            else:
                lg = _rest(nets, None, z, og)
                if loss == "ce":
                    L = Fn.cross_entropy(lg.reshape(-1, P), sl.reshape(-1), reduction="none").view(G, k, -1).mean(-1)
                else:
                    L = (tgt.exp() * (tgt - Fn.log_softmax(lg, -1))).sum(-1).mean(-1)
            g = torch.autograd.grad((L * vmask).sum(), qq)[0]
        else:
            with torch.no_grad():
                gv, _ = _pc_col_grad(nets, None, base_s + col(q).unsqueeze(2), Y, eta, pc.get("T", T_PC), pc.get("beta", BETA_PC), og)
                gv = gv * vmask.unsqueeze(-1)
                g = gv if v0 is None else (Bm.transpose(-1, -2) @ gv.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            _adam_q(q, g, m1, m2, s, lr_t)
            if proj is not None:
                Qp, kp = proj; c_ = (q.unsqueeze(-2) @ Qp).squeeze(-2); pq = (Qp @ c_.unsqueeze(-1)).squeeze(-1)
                q.copy_(pq if kp else q - pq)
        evaluate(s)
    out = dict(acc={n: gr.flat(torch.stack(a, -1)).cpu().numpy() if a else np.zeros((C, 0)) for n, a in acc.items()})
    out["A"] = {n: a.mean(1) if a.shape[1] else None for n, a in out["acc"].items()}
    out["best"] = {b: {kk: ({n: gr.flat(t).cpu().numpy() for n, t in vv.items()} if kk == "heldout" else gr.flat(vv).cpu().numpy())
                       for kk, vv in d_.items()} for b, d_ in best.items()}
    out["v"] = gr.flat(col(q)).detach().cpu().numpy(); out["q"] = gr.flat(q).cpu().numpy()
    return out


# ====================================================================================================================== PC
def _pc_col_grad(nets, gi, z1, Y, eta, T, beta, ops=None, return_states=False):
    """Closed-form PC relaxation (pc_meta.pc_relax without the float64 energy log) for chains z1 [G, k, N, W] of nets gi with targets
    Y [G, k, N, 17] (one-hot or soft), eta [G, k]; erasers of ops at h1 / h2 inside the energy (e1 = h1 - E1(tanh z1), e2 = h2 -
    E2(tanh(W2 h1 + b2))) and in the feedforward init. Returns (dE/dv [G, k, W] = mean over inputs of -(J1^T e1) (1 - a1^2), the
    end / start ratio of the relaxation gradient norm [G, k]) or, with return_states, (h1, h2, ratio)."""
    G, k, N, W = z1.shape
    take = (lambda t: t) if gi is None else (lambda t: t[gi])
    W2, b2, W3, b3 = take(nets.p["W2"]), take(nets.p["b2"]), take(nets.p["W3"]), take(nets.p["b3"])
    mult = nets.mult; et = eta.view(G, k, 1, 1)
    a1 = torch.tanh(z1); p1 = a1 if ops is None else ops.on_h1(a1)
    h1 = p1.clone()

    def lay2(h):
        return torch.tanh(torch.baddbmm(b2.unsqueeze(1), h.reshape(G, k * N, W), W2.transpose(1, 2))).view(G, k, N, W)

    a2 = lay2(h1); h2 = a2 if ops is None else ops.on_h2(a2); h2 = h2.clone()
    g0 = gl = None
    for t in range(T + 1):
        a2 = lay2(h1); p2 = a2 if ops is None else ops.on_h2(a2)
        e1 = h1 - p1; e2 = h2 - p2
        if t == T:
            break
        lg = (torch.baddbmm(b3.unsqueeze(1), h2.reshape(G, k * N, W), W3.transpose(1, 2)) * mult).view(G, k, N, P)
        je2 = e2 if ops is None else ops.jac_t(e2, "h2")
        g1 = e1 - torch.bmm((je2 * (1 - a2 ** 2)).reshape(G, k * N, W), W2).view(G, k, N, W)
        g2 = e2 + (beta * mult) * torch.bmm((torch.softmax(lg, -1) - Y).reshape(G, k * N, P), W3).view(G, k, N, W)
        gn = torch.sqrt((g1 ** 2).sum((2, 3)) + (g2 ** 2).sum((2, 3)))
        if g0 is None:
            g0 = gn
        gl = gn
        h1 = h1 - et * g1; h2 = h2 - et * g2
    ratio = torch.where(g0 > 0, gl / g0, torch.zeros_like(g0)) if g0 is not None else torch.zeros(G, k, device=z1.device)
    if return_states:
        return h1, h2, ratio
    je1 = e1 if ops is None else ops.jac_t(e1, "h1")
    return (-(je1 * (1 - a1 ** 2))).mean(2), ratio


@torch.no_grad()
def pc_relax(nets, net_of, cols, sets, eta, T=T_PC, beta=BETA_PC, ops=None, soft=None):
    """Batched closed-form PC relaxation (W0) of chains: inputs sets = (x, y, lab) [C, N] (or shared [N]) with columns cols [C, W] of
    nets net_of [C]; the output clamped to lab (one-hot) or to soft probabilities soft [C, N, 17]; eta [C] or float; erasers of ops
    inside the energy. Returns (h1 [C, N, W], h2 [C, N, W], end / start gradient-norm ratio [C]) as tensors."""
    C = len(net_of); gr = _Groups(net_of); x, y, lab = (gr.grp(a) for a in _xyl(sets, C))
    v = gr.grp(torch.as_tensor(cols, dtype=F32, device=DEV)); og = None if ops is None else ops.regroup(gr.idx)
    Y = Fn.one_hot(lab, P).to(F32) if soft is None else gr.grp(torch.as_tensor(soft, dtype=F32, device=DEV))
    et = gr.grp(torch.as_tensor(np.broadcast_to(np.asarray(eta, dtype=np.float64), (C,)).copy(), dtype=F32, device=DEV))
    h1, h2, ratio = _pc_col_grad(nets, gr.gi, _base(nets, gr.gi, x, y) + v.unsqueeze(2), Y, et, T, beta, og, return_states=True)
    return gr.flat(h1), gr.flat(h2), gr.flat(ratio)


@torch.no_grad()
def pc_relax_inputs(nets, X, y, eta, T=T_PC, beta=BETA_PC, soft=None):
    """Closed-form relaxation of whole nets on input features X [B, N, din] with labels y [B, N] (one-hot output term) or soft target
    distributions soft [B, N, 17] (W7's soft-target term) (the equivalence tests with pc_meta.pc_relax). Returns (h1, h2, ratio)."""
    z = nets.z1(X).unsqueeze(1)
    Y = (Fn.one_hot(y, P).to(F32) if soft is None else torch.as_tensor(soft, dtype=F32, device=DEV)).unsqueeze(1)
    et = torch.as_tensor(np.broadcast_to(np.asarray(eta, dtype=np.float64), (nets.B,)).copy(), dtype=F32, device=DEV).view(-1, 1)
    h1, h2, r = _pc_col_grad(nets, torch.arange(nets.B, device=DEV), z, Y, et, T, beta, None, return_states=True)
    return h1[:, 0], h2[:, 0], r[:, 0]


class _BetaCtx:
    def __init__(self, beta, T):
        self.beta, self.T = beta, T

    def __enter__(self):
        self.old = (PCC.BETA[0], M.T); PCC.BETA[0] = self.beta; M.T = self.T

    def __exit__(self, *a):
        PCC.BETA[0], M.T = self.old


def pc_eta(nets, i, X, y, seed=0, beta=BETA_PC):
    """eta_x = 1 / pc_meta.lam_max of net i on its training batch (X [N, din] features, y [N]) with the PC output precision beta; the
    power iteration's random start comes from torch.manual_seed(seed) (the global RNG state is restored afterwards)."""
    m = nets.to_mlp(i).to(DEV)
    with torch.random.fork_rng(devices=[DEV] if DEV.type == "cuda" else []), _BetaCtx(beta, T_PC):
        torch.manual_seed(int(seed)); lam = M.lam_max(m, X, y)
    return 1.0 / lam


# ====================================================================================================================== B2
def b2_A(nets, ws_of, nshots=(8, 4), tasks=range(NNEW), steps=None, eval_at=None, pc_eta_of=None, ky_of=None, ops=None, keep=None):
    """Column-search part of pc_meta's B2 for every net: per new task j and nshot the slot column from zero, Adam lr 1e-2, held-out =
    split positions 8-288; accuracies at eval_at (FT_EVAL). pc_eta_of: per-net eta_x for PC nets (PC local update). ops: per-net Ops
    (leading dim B) on during the search ('B2 with the operation on', C-T4); keep: per-net bases [B, W, r] of a kept column subspace
    (C-T1 task-code cuts: the column projected after every step). Returns dict nshot -> array [B, n_tasks, len(eval_at)]."""
    eval_at = FT_EVAL if eval_at is None else tuple(eval_at); steps = max(eval_at) if steps is None else steps
    tasks = list(tasks); out = {}
    for ns in nshots:
        net_of, sets = [], []
        for b in range(nets.B):
            for j in tasks:
                net_of.append(b); sets.append(new_task_sets(ws_of[b], j, ky=1 if ky_of is None else ky_of[b], nshot=ns))
        pc = None if pc_eta_of is None else dict(eta=np.asarray(pc_eta_of)[np.asarray(net_of)])
        no = np.array(net_of); kp = None if keep is None else torch.as_tensor(keep, dtype=F32, device=DEV)[torch.as_tensor(no, device=DEV)]
        r = column_search(nets, no, stack_sets(sets, "shot"), stack_sets(sets, "ho"), steps=steps, eval_at=eval_at, pc=pc,
                          ops=None if ops is None else ops.expand(no), keep=kp)
        out[ns] = r["acc"]["ho"].reshape(nets.B, len(tasks), len(eval_at))
    return out


_F_CACHE = {}


def fresh_F(arm, W, seeds, nshots=(8, 4), tasks=range(NNEW), steps=None, eval_at=None):
    """F of pc_meta.finetune(mode 'fresh') for BP arms: per (seed, task j, nshot) a fresh net of the arm and width
    (torch.manual_seed(50000 + 101 seed + j)) trained with all weights (make_opt: lr 1e-3, decay 1e-3 per update) on the nshot shots of
    new task j of world(seed) for FT_STEPS updates; held-out accuracy (split positions 8-288) at FT_EVAL. Cached. Returns dict
    nshot -> array [len(seeds), n_tasks, len(eval_at)]."""
    eval_at = FT_EVAL if eval_at is None else tuple(eval_at); steps = max(eval_at) if steps is None else steps
    tasks = list(tasks); out = {}
    for ns in nshots:
        key = lambda s, j: (arm, W, int(s), int(j), ns, steps, eval_at)
        todo = sorted({(int(s), int(j)) for s in seeds for j in tasks if key(s, j) not in _F_CACHE})
        if todo:
            specs = [dict(arm=arm, W=W, ws=s, seed=fresh_seed(s, j)) for s, j in todo]
            nets = Nets.from_specs(specs); sets = [new_task_sets(s, j, nshot=ns) for s, j in todo]
            sx, sy, sl = (torch.as_tensor(a, device=DEV) for a in stack_sets(sets, "shot"))
            hx, hy, hl = (torch.as_tensor(a, device=DEV) for a in stack_sets(sets, "ho"))
            slot = torch.as_tensor([[NPRE + j] * ns for _, j in todo], device=DEV)
            Xs = nets.features(sx, sy, slot); Xh = nets.features(hx, hy, slot[:, :1].expand(-1, hx.shape[1]))
            opt = AdamWB(nets.p, LR, 1.0, hidden=nets.hidden(), W=W, mup=nets.param == "mup"); accs = []
            for s in range(1, steps + 1):
                for kk in nets.p:
                    nets.p[kk].requires_grad_(True)
                lg = nets.logits(Xs)
                L = Fn.cross_entropy(lg.reshape(-1, P), sl.reshape(-1), reduction="none").view(nets.B, -1).mean(1).sum()
                gr_ = torch.autograd.grad(L, list(nets.p.values()))
                for kk in nets.p:
                    nets.p[kk].requires_grad_(False)
                opt.step(dict(zip(nets.p.keys(), gr_)))
                if s in eval_at:
                    with torch.no_grad():
                        accs.append((nets.logits(Xh).argmax(-1) == hl).float().mean(1))
            A = torch.stack(accs, 1).cpu().numpy()
            for i, (s, j) in enumerate(todo):
                _F_CACHE[key(s, j)] = A[i]
        out[ns] = np.stack([[_F_CACHE[key(s, j)] for j in tasks] for s in seeds])
    return out


def b2_all(nets, ws_of, tasks=range(NNEW), steps=None, eval_at=None, nshot=NSHOT):
    """pc_meta.finetune mode 'all' for BP nets: per net and new task j a copy of the pretrained net trained with all weights (make_opt:
    fresh AdamW, lr 1e-3, decay 1e-3 per update whatever the run's wd_scale) on the nshot shots for FT_STEPS updates; held-out accuracy
    at FT_EVAL and the retention = pooled held-out accuracy on the pretraining tasks afterwards. Returns dict(acc [B, n_tasks,
    len(eval_at)], retention [B, n_tasks])."""
    eval_at = FT_EVAL if eval_at is None else tuple(eval_at); steps = max(eval_at) if steps is None else steps; tasks = list(tasks)
    idx = np.repeat(np.arange(nets.B), len(tasks)); cp = nets.subset(idx)
    sets = [new_task_sets(ws_of[b], j, nshot=nshot) for b in range(nets.B) for j in tasks]
    sx, sy, sl = (torch.as_tensor(a, device=DEV) for a in stack_sets(sets, "shot")); hx, hy, hl = (torch.as_tensor(a, device=DEV) for a in stack_sets(sets, "ho"))
    slot = torch.as_tensor([[s["slot"]] * nshot for s in sets], device=DEV)
    Xs = cp.features(sx, sy, slot); Xh = cp.features(hx, hy, slot[:, :1].expand(-1, hx.shape[1]))
    tss = [taskset_w1(ws_of[b]) for b in idx]; vx, vy, vt, vl = (torch.as_tensor(np.stack([ts[k] for ts in tss]), device=DEV) for k in ("vx", "vy", "vt", "vlab"))
    opt = AdamWB(cp.p, LR, 1.0, hidden=cp.hidden(), W=cp.W, mup=cp.param == "mup"); accs = []
    for s in range(1, steps + 1):
        for kk in cp.p:
            cp.p[kk].requires_grad_(True)
        L = Fn.cross_entropy(cp.logits(Xs).reshape(-1, P), sl.reshape(-1), reduction="none").view(cp.B, -1).mean(1).sum()
        gr_ = torch.autograd.grad(L, list(cp.p.values()))
        for kk in cp.p:
            cp.p[kk].requires_grad_(False)
        opt.step(dict(zip(cp.p.keys(), gr_)))
        if s in eval_at:
            with torch.no_grad():
                accs.append((cp.logits(Xh).argmax(-1) == hl).float().mean(1))
    with torch.no_grad():
        ret = (cp.logits(cp.features(vx, vy, vt)).argmax(-1) == vl).float().mean(1)
    return dict(acc=torch.stack(accs, 1).cpu().numpy().reshape(nets.B, len(tasks), len(eval_at)), retention=ret.cpu().numpy().reshape(nets.B, len(tasks)))


def b2(nets, ws_of, arm=None, nshots=(8, 4), steps=None, eval_at=None, pc_eta_of=None, with_all=False):
    """pc_meta's B2 per net: dict nshot -> dict(A [B, 4, len(eval_at)], F [B, 4, len(eval_at)], B2 [B]) with B2 = mean over the 4 new
    tasks of (mean A - mean F). F comes from fresh_F (BP arms; for PC nets pass F from the round-1 job files). with_all (BP): also
    'all8' = dict(acc, retention, B2_all = mean over tasks of mean acc - mean F(8)) as pc_meta's ft['all8']."""
    A = b2_A(nets, ws_of, nshots, steps=steps, eval_at=eval_at, pc_eta_of=pc_eta_of)
    arm = arm or ("bp_sp" if nets.param == "sp" else "bp_mup")
    F = fresh_F(arm, nets.W, ws_of, tuple(sorted(set(nshots) | ({8} if with_all else set()), reverse=True)), steps=steps, eval_at=eval_at) if arm != "pc_mup" else None
    out = {}
    for ns in nshots:
        o = dict(A=A[ns])
        if F is not None:
            o["F"] = F[ns]; o["B2"] = (A[ns].mean(-1) - F[ns].mean(-1)).mean(-1)
        out[ns] = o
    if with_all:
        al = b2_all(nets, ws_of, steps=steps, eval_at=eval_at)
        out["all8"] = dict(acc=al["acc"], retention=al["retention"], B2_all=(al["acc"].mean(-1) - F[8].mean(-1)).mean(-1) if F is not None else None)
    return out


# ====================================================================================================================== geometry
@torch.no_grad()
def _fisher_perm(Hc, labs, K):
    """F = B / (TSS - B) of centred H [b, N, D] (float64) for label vectors labs [q, N] with K equal-size classes -> [b, q]."""
    N = Hc.shape[1]; O = Fn.one_hot(labs, K).to(F64).transpose(1, 2)                     # [q, K, N]
    tss = (Hc ** 2).sum((1, 2))
    nk = O.sum(-1)                                                                        # [q, K]
    out = []
    qc = max(1, int(2e8 // max(1, Hc.shape[0] * K * Hc.shape[2] * 8)))
    for a in range(0, O.shape[0], qc):
        S = torch.einsum("qkn,bnd->bqkd", O[a:a + qc], Hc)
        Bq = ((S ** 2).sum(-1) / nk[a:a + qc].unsqueeze(0)).sum(-1)
        out.append(Bq / (tss.unsqueeze(1) - Bq))
    return torch.cat(out, 1)


@torch.no_grad()
def delta_stats(nets, seeds, pres, task_idx=None, nperm=50, max_bytes=int(2e9)):
    """Vectorised pc_meta.delta_stats in chunks of nets (activations of a chunk in float64 below max_bytes); see _delta_stats."""
    per = max(1, int(max_bytes // (2 * NPRE * P * P * nets.W * 8 * 3)))
    if nets.B <= per:
        return _delta_stats(nets, seeds, pres, task_idx, nperm)
    out = []
    for a in range(0, nets.B, per):
        ii = list(range(a, min(nets.B, a + per)))
        out += _delta_stats(nets.subset(ii), [seeds[i] for i in ii], [pres[i] for i in ii], task_idx, nperm)
    return out


@torch.no_grad()
def _delta_stats(nets, seeds, pres, task_idx=None, nperm=50):
    """Vectorised pc_meta.delta_stats for every net (W1 'toy delta'): h1, h2 over the 12 pretraining tasks x all 289 pairs; F = tr(S_b)
    / tr(S_w) (float64), floor = mean over 50 label permutations, delta = F - floor; the same after subtracting each pair's mean over
    tasks (ctrl); ring = Spearman correlation of centroid distances with the cyclic shift distance. Permutation stream exactly
    pc_meta's: torch.Generator().manual_seed(77 + seed), for h1 then h2, 50 raw then 50 controlled permutations. seeds / pres: per net
    (seed = the world seed for pc_meta jobs). task_idx: task indices of the 12 tasks (default slots 0-11). Returns list of dicts."""
    from scipy.stats import spearmanr
    B = nets.B; task_idx = list(range(NPRE)) if task_idx is None else list(task_idx); NP_ = P * P
    x = torch.as_tensor(np.tile(PAIRS[:, 0], NPRE), device=DEV).expand(B, -1); y = torch.as_tensor(np.tile(PAIRS[:, 1], NPRE), device=DEV).expand(B, -1)
    t = torch.as_tensor(np.repeat(task_idx, NP_), device=DEV).expand(B, -1)
    h1, h2 = nets.hidden_acts(nets.features(x, y, t))
    lab = torch.arange(NPRE).repeat_interleave(NP_)
    perms = {}
    for sd in sorted(set(int(s) for s in seeds)):
        g = torch.Generator().manual_seed(77 + sd)
        perms[sd] = torch.stack([lab[torch.randperm(len(lab), generator=g)] for _ in range(4 * nperm)]).to(DEV)
    res = [dict() for _ in range(B)]
    lab_d = lab.to(DEV)
    for li, (name, H) in enumerate((("h1", h1), ("h2", h2))):
        H = H.double(); Hc = H - H.mean(1, keepdim=True)
        Hx = H.view(B, NPRE, NP_, -1); Hx = (Hx - Hx.mean(1, keepdim=True)).reshape(B, NPRE * NP_, -1); Hxc = Hx - Hx.mean(1, keepdim=True)
        F0 = _fisher_perm(Hc, lab_d.unsqueeze(0), NPRE)[:, 0]; Fc0 = _fisher_perm(Hxc, lab_d.unsqueeze(0), NPRE)[:, 0]
        C = H.view(B, NPRE, NP_, -1).mean(2)
        for sd in perms:
            bs = [b for b in range(B) if int(seeds[b]) == sd]; bi = torch.as_tensor(bs, device=DEV)
            pr = perms[sd][li * 2 * nperm: li * 2 * nperm + nperm]; prc = perms[sd][li * 2 * nperm + nperm: (li + 1) * 2 * nperm]
            fl = _fisher_perm(Hc[bi], pr, NPRE).mean(1).cpu().numpy(); flc = _fisher_perm(Hxc[bi], prc, NPRE).mean(1).cpu().numpy()
            for ii, b in enumerate(bs):
                D = torch.cdist(C[b], C[b]).cpu().numpy(); pre = pres[b]
                cyc = np.array([[min(abs(a - c), P - abs(a - c)) for c in pre] for a in pre]); iu = np.triu_indices(NPRE, 1)
                ring = float(spearmanr(D[iu], cyc[iu]).correlation)
                F_, Fc_ = float(F0[b]), float(Fc0[b])
                res[b][name] = dict(F=F_, floor=float(fl[ii]), delta=F_ - float(fl[ii]), F_ctrl=Fc_, floor_ctrl=float(flc[ii]),
                                    delta_ctrl=Fc_ - float(flc[ii]), ring=ring)
    return res


@torch.no_grad()
def mediators(nets, ws_of, sat=0.95):
    """AT3 mediators per net (W1 one-hot nets): pooled held-out accuracy on the pretraining tasks, mean L2 norm of the 12 task columns,
    saturation share = fraction of |h1| > 0.95 over all 12 x 289 inputs and units. Returns dict of numpy [B]."""
    h1, _, _, _, _ = task_states(nets, ws_of, which="all")
    ta = task_accuracy(nets, ws_of)
    return dict(heldout=ta.mean(1), task_col_norm=nets.task_cols()[:, :, :NPRE].norm(dim=1).mean(1).cpu().numpy(),
                saturation=(h1.abs() > sat).float().mean((1, 2, 3)).cpu().numpy())


def complement_basis(Q):
    """Orthonormal basis of the orthogonal complement of span(Q) (Q [..., W, r] orthonormal): [..., W, W - r]. Keeping a subspace S in an
    activation (C-T1: z1 or h2 on its top-r PCs) is the removal of span(complement_basis(Q_S)) with Ops (mu = the activation mean)."""
    Q = torch.as_tensor(Q).to(F64); W, r = Q.shape[-2], Q.shape[-1]
    Pm = torch.eye(W, dtype=F64, device=Q.device) - Q @ Q.transpose(-1, -2)
    U, S, _ = torch.linalg.svd(Pm)
    return U[..., :W - r].to(F32)


# ====================================================================================================================== W6
def self_train(nets, ws_of, taus=(0.1, 0.3, 0.5, 1.0, 2.0), rounds=6, tasks=range(NNEW), steps=None, starts=("true", "wrong")):
    """W6 self-training loop (theta toy): per net, new task j, tau and start: round 0 = the B2 column (8 shots, Adam lr 1e-2, from zero;
    'wrong' start: the 8 shots relabelled with shift c + 1); round n + 1: the 281 non-shot pairs (split positions 8-288) are labelled
    by sampling from softmax(logits / tau) of the net with the round-n column (Gumbel-max with uniforms from
    torch.Generator().manual_seed(7000 + 1000 ws + 100 j + 10 n), shared across tau and the two starts), and the column is refitted from
    zero on the 8 true shots plus the 281 pseudo-labels (CE mean over the 289 examples, same optimiser and steps). A_n = held-out
    accuracy (281 pairs, true labels) of the round-n column. Returns dict(keys [(net, j, tau, start)], A [n_chains, rounds + 1])."""
    steps = FT_STEPS if steps is None else steps; tasks = list(tasks); keys = []
    for b in range(nets.B):
        for j in tasks:
            for tau in taus:
                for st in starts:
                    keys.append((b, j, float(tau), st))
    C = len(keys); net_of = np.array([k_[0] for k_ in keys])
    sets = [new_task_sets(ws_of[b], j) for b, j, _, _ in keys]
    sx, sy, sl = stack_sets(sets, "shot"); hx, hy, hl = stack_sets(sets, "ho")
    sl0 = np.where(np.array([k_[3] == "wrong" for k_ in keys])[:, None], (sl + 1) % P, sl)
    r = column_search(nets, net_of, (sx, sy, sl0), (hx, hy, hl), steps=steps, eval_at=(steps,))
    A = [r["acc"]["ho"][:, -1]]; v = r["v"]
    ax = np.concatenate([sx, hx], 1); ay = np.concatenate([sy, hy], 1)
    for n in range(rounds):
        lg = _chain_logits(nets, net_of, v, (hx, hy))
        gum = np.zeros((C, hx.shape[1], P), np.float32)
        cache = {}
        for c, (b, j, tau, st) in enumerate(keys):
            sd = 7000 + 1000 * int(ws_of[b]) + 100 * j + 10 * n
            if sd not in cache:
                u = torch.rand(hx.shape[1], P, generator=torch.Generator().manual_seed(sd), dtype=F64)
                cache[sd] = (-torch.log(-torch.log(u))).numpy().astype(np.float32)
            gum[c] = cache[sd]
        tau_c = np.array([k_[2] for k_ in keys], np.float32)[:, None, None]
        pseudo = np.argmax(lg / tau_c + gum, -1)
        al = np.concatenate([sl, pseudo], 1)
        r = column_search(nets, net_of, (ax, ay, al), (hx, hy, hl), steps=steps, eval_at=(steps,))
        A.append(r["acc"]["ho"][:, -1]); v = r["v"]
    return dict(keys=keys, A=np.stack(A, 1))


@torch.no_grad()
def _chain_logits(nets, net_of, cols, sets, ops=None):
    C = len(net_of); gr = _Groups(net_of); x, y = (gr.grp(a) for a in _xyl((sets[0], sets[1]), C))
    v = gr.grp(torch.as_tensor(cols, dtype=F32, device=DEV)); o = None if ops is None else ops.regroup(gr.idx)
    return gr.flat(_rest(nets, gr.gi, _base(nets, gr.gi, x, y) + v.unsqueeze(2), o)).cpu().numpy()


def w6_summary(res):
    """Per (net, tau): S = A_6 - A_0 (true start), M_n = A_n(true) - A_n(wrong), kappa = (M_6 / M_1)^(1/5) if M_1 > 0.01 and M_6 > 0
    else 0 (OD-B5); averaged over the tasks (S and M) before kappa. Returns dict (net, tau) -> dict(S, M [rounds + 1], kappa)."""
    A = res["A"]; acc = {}
    for c, (b, j, tau, st) in enumerate(res["keys"]):
        acc.setdefault((b, tau), {}).setdefault(st, []).append(A[c])
    out = {}
    for key, d in acc.items():
        at, aw = np.mean(d["true"], 0), np.mean(d["wrong"], 0); Mn = at - aw; n = len(at) - 1
        kap = float((Mn[n] / Mn[1]) ** (1 / (n - 1))) if (Mn[1] > 0.01 and Mn[n] > 0) else 0.0
        out[key] = dict(S=float(at[n] - at[0]), M=Mn.tolist(), kappa=kap, A_true=at.tolist(), A_wrong=aw.tolist())
    return out


# ====================================================================================================================== W8
def _ridge_fit_predict(Atr, Ytr, Ate, lam):
    """Batched ridge (float64): w = (A^T A + lam I)^-1 A^T Y (dual form when N < D); returns Ate w."""
    N, D = Atr.shape[-2], Atr.shape[-1]
    if N < D:
        Kt = Atr @ Atr.transpose(-1, -2); I = torch.eye(N, dtype=F64, device=Atr.device)
        a = torch.linalg.solve(Kt + lam * I, Ytr); return Ate @ (Atr.transpose(-1, -2) @ a)
    G_ = Atr.transpose(-1, -2) @ Atr; I = torch.eye(D, dtype=F64, device=Atr.device)
    return Ate @ torch.linalg.solve(G_ + lam * I, Atr.transpose(-1, -2) @ Ytr)


@torch.no_grad()
def ridge_probe(Htr, ytr, Hte, yte, ncls, lam=1e-2, standardise=True):
    """Linear ridge probe (W8): features standardised with the training mean / SD (+1e-6) as the refs (probe.py, c1_structure), an
    intercept column, one-hot targets, closed form in float64 with ridge lam; accuracy by argmax. Batched over leading dims: Htr
    [..., Ntr, D], ytr [..., Ntr], Hte [..., Nte, D], yte [..., Nte]. Returns accuracy tensor [...]."""
    Htr, Hte = torch.as_tensor(Htr).to(F64), torch.as_tensor(Hte).to(F64)
    ytr, yte = torch.as_tensor(ytr, device=Htr.device), torch.as_tensor(yte, device=Htr.device)
    if standardise:
        mu = Htr.mean(-2, keepdim=True); sd = Htr.std(-2, keepdim=True, correction=0) + 1e-6; Htr, Hte = (Htr - mu) / sd, (Hte - mu) / sd
    one = lambda H: torch.cat([H, torch.ones(*H.shape[:-1], 1, dtype=F64, device=H.device)], -1)
    pred = _ridge_fit_predict(one(Htr), Fn.one_hot(ytr, ncls).to(F64), one(Hte), lam).argmax(-1)
    return (pred == yte).to(F64).mean(-1)


def mlp_probe(Htr, ytr, Hte, yte, ncls, hid=64, steps=300, lr=1e-2, wd=1e-4, seed=0):
    """One-hidden-layer MLP probe (OD-C10): 64 tanh units, Adam lr 1e-2 with weight decay 1e-4 (L2, torch.optim.Adam semantics), 300
    full-batch steps, features standardised with the training mean / SD (+1e-6, as c1_structure.mlp_probe), init as nn.Linear's
    default from torch.Generator(seed). Batched over a leading dim: Htr [Q, Ntr, D] ... Returns accuracy [Q] (numpy)."""
    Htr, Hte = torch.as_tensor(Htr, dtype=F32, device=DEV), torch.as_tensor(Hte, dtype=F32, device=DEV)
    ytr, yte = torch.as_tensor(ytr, device=DEV), torch.as_tensor(yte, device=DEV)
    Q, N, D = Htr.shape
    mu = Htr.mean(1, keepdim=True); sd = Htr.std(1, keepdim=True) + 1e-6; Xtr, Xte = (Htr - mu) / sd, (Hte - mu) / sd
    g = torch.Generator().manual_seed(int(seed)); b1_, b2_ = 1 / math.sqrt(D), 1 / math.sqrt(hid)
    prm = dict(W1=(torch.rand(Q, hid, D, generator=g) * 2 - 1) * b1_, c1=(torch.rand(Q, hid, generator=g) * 2 - 1) * b1_,
               W2=(torch.rand(Q, ncls, hid, generator=g) * 2 - 1) * b2_, c2=(torch.rand(Q, ncls, generator=g) * 2 - 1) * b2_)
    prm = {k: v.to(DEV) for k, v in prm.items()}; ms = {k: torch.zeros_like(v) for k, v in prm.items()}; vs = {k: torch.zeros_like(v) for k, v in prm.items()}

    def fwd(X):
        h = torch.tanh(torch.baddbmm(prm["c1"].unsqueeze(1), X, prm["W1"].transpose(1, 2)))
        return torch.baddbmm(prm["c2"].unsqueeze(1), h, prm["W2"].transpose(1, 2))
    for s in range(1, steps + 1):
        for v in prm.values():
            v.requires_grad_(True)
        L = Fn.cross_entropy(fwd(Xtr).reshape(-1, ncls), ytr.reshape(-1), reduction="none").view(Q, -1).mean(1).sum()
        grads = torch.autograd.grad(L, list(prm.values()))
        with torch.no_grad():
            for (k, v), gk in zip(prm.items(), grads):
                v.requires_grad_(False); gk = gk.add(v, alpha=wd); _adam_q(v, gk, ms[k], vs[k], s, float(lr))
    with torch.no_grad():
        return (fwd(Xte).argmax(-1) == yte).float().mean(1).cpu().numpy()


def sum_probe(nets, ws_of, ops=None, cols=None, lam=1e-2, states=None):
    """W8 per-task sum probe (the one generic-damage measure): per net and pretraining task k a ridge probe (ridge_probe, lam 1e-2) from
    h2 (computed with the task's column, or cols [B, 12, W], and ops per net; or given states h2 [B, 12, 289, W] in split order) to
    one-hot (x + y) mod 17, trained on the task's 144 training pairs, accuracy by argmax on its 145 held-out pairs; mean over the 12
    tasks. Returns numpy [B]."""
    _, h2, x, y, _ = task_states(nets, ws_of, ops, cols, "all")
    if states is not None:
        h2 = states
    B = nets.B; h2 = h2.reshape(B * NPRE, P * P, -1); lab = torch.as_tensor(((x + y) % P).reshape(B * NPRE, -1), device=DEV)
    acc = ridge_probe(h2[:, :144], lab[:, :144], h2[:, 144:], lab[:, 144:], P, lam)
    return acc.view(B, NPRE).mean(1).cpu().numpy()


def heldout_halves(ws):
    """W8 probe halves: per task the held-out pairs (split positions 144-288): first 72 = probe training half, last 73 = test half."""
    sp = M.world(ws)[3]
    return [(sp[k][144:216], sp[k][216:]) for k in range(NPRE)]


@torch.no_grad()
def task_states(nets, ws_of, ops=None, cols=None, which="all"):
    """h1, h2 of the 12 pretraining tasks of every net (W1 one-hot nets): inputs = for each task k its pairs in split order (which='all':
    all 289, positions 0-143 training, 144-288 held-out; 'heldout': the 145 held-out pairs), task columns (or cols [B, 12, W]) and
    ops per net (leading dim B). Returns (h1, h2) [B, 12, N, W], x, y [B, 12, N] (numpy), shift [B, 12]."""
    B = nets.B; T = nets.task_cols(); lo = 0 if which == "all" else 144
    xs, ys, sh = [], [], []
    for b in range(B):
        pre, _, _, sp = M.world(ws_of[b])
        for k in range(NPRE):
            x, y = _pairs_of(sp[k], lo, P * P); xs.append(x); ys.append(y)
        sh.append(list(pre))
    x = np.stack(xs); y = np.stack(ys); net_of = np.repeat(np.arange(B), NPRE)
    Cc = T[:, :, :NPRE].transpose(1, 2).reshape(-1, nets.W) if cols is None else torch.as_tensor(cols, dtype=F32, device=DEV).reshape(-1, nets.W)
    h1, h2 = hidden_states(nets, net_of, Cc, (x, y), None if ops is None else ops.expand(net_of))
    n = x.shape[1]
    return h1.view(B, NPRE, n, -1), h2.view(B, NPRE, n, -1), x.reshape(B, NPRE, n), y.reshape(B, NPRE, n), np.array(sh)


@torch.no_grad()
def task_answer_features(nets, ws_of, layer="h2", ops=None, cols=None, states=None):
    """W8 task-identity and pooled-answer probe data per net: the 12 x 145 held-out inputs at layer 'h1' / 'h2' (ops per net, or given
    states = (h1, h2) [B, 12, 145, W] from task_states(which='heldout'), e.g. relaxed PC states), task labels 0-11 and answer labels
    (x + y + c) mod 17, split into the halves of heldout_halves (per task the first 72 held-out pairs train the probe, the last 73
    test it). Returns dict(Htr [B, 864, W], Hte [B, 876, W], task_tr, task_te, ans_tr, ans_te) tensors."""
    h1, h2, x, y, sh = task_states(nets, ws_of, ops, cols, "heldout")
    if states is not None:
        h1, h2 = states
    H = h1 if layer == "h1" else h2; B, K, n, W = H.shape; half = 72
    ans = torch.as_tensor((x + y + sh[:, :, None]) % P, device=DEV); tk = torch.arange(K, device=DEV).view(1, K, 1).expand(B, K, n)
    cut = lambda A, a, b: A[:, :, a:b].reshape(B, -1, *A.shape[3:])
    return dict(Htr=cut(H, 0, half), Hte=cut(H, half, n), task_tr=cut(tk, 0, half), task_te=cut(tk, half, n), ans_tr=cut(ans, 0, half), ans_te=cut(ans, half, n))


def task_answer_probes(feats, mlp=True, seed=0):
    """Task-identity (chance 1/12) and pooled answer (chance 1/17) probes on task_answer_features output: linear ridge (lam 1e-2) and
    the OD-C10 MLP. Returns dict of numpy [B]."""
    f = feats; out = dict(task_lin=ridge_probe(f["Htr"], f["task_tr"], f["Hte"], f["task_te"], NPRE).cpu().numpy(),
                          ans_lin=ridge_probe(f["Htr"], f["ans_tr"], f["Hte"], f["ans_te"], P).cpu().numpy())
    if mlp:
        out["task_mlp"] = mlp_probe(f["Htr"], f["task_tr"], f["Hte"], f["task_te"], NPRE, seed=seed)
        out["ans_mlp"] = mlp_probe(f["Htr"], f["ans_tr"], f["Hte"], f["ans_te"], P, seed=seed + 1)
    return out


@torch.no_grad()
def class_mean_eraser(H, lab, K, tol=1e-10):
    """W8 class-mean erasure of a labelled concept (the unlearning design's 'LEACE'): mu = mean of H [N, D] (or [B, N, D]), Q = an
    orthonormal basis of the span of the K centred class means (rank <= K - 1); E(h) = h - (h - mu) Q Q^T makes every class mean equal
    (linear guardedness). Returns (Q [.., D, r_max] zero-padded, mu [.., D]) as float32 for Ops(h*_Q=Q, h*_mu=mu)."""
    H = torch.as_tensor(H).to(F64); one = H.dim() == 2
    if one:
        H = H.unsqueeze(0)
    lab = torch.as_tensor(lab, device=H.device)
    if lab.dim() == 1:
        lab = lab.expand(H.shape[0], -1)
    mu = H.mean(1); O = Fn.one_hot(lab, K).to(F64); n = O.sum(1)
    Mk = (O.transpose(1, 2) @ H) / n.clamp(min=1).unsqueeze(-1) - mu.unsqueeze(1)          # [B, K, D]
    Mk = torch.where((n > 0).unsqueeze(-1), Mk, torch.zeros_like(Mk))
    U, S, _ = torch.linalg.svd(Mk.transpose(1, 2), full_matrices=False)                    # [B, D, K]
    keep = S > tol * S[:, :1].clamp(min=1e-300)
    Q = U * keep.unsqueeze(1).to(F64)
    Q, mu = Q.to(F32), mu.to(F32)
    return (Q[0], mu[0]) if one else (Q, mu)


def damage_match(target, drop_of, r0, W, ndraw=5, tol=0.02, iters=30):
    """OD-C2 damage-matched random control. target = the operation's sum-probe drop (base minus operated, seed mean); drop_of(r, lam)
    -> mean drop over ndraw random r-dim subspaces (and the seeds), the caller's random subspaces at dimension r with partial removal
    z - lam P z. Ladder r0, 2 r0, 4 r0, ... (<= W / 2; r0 always tried); d* = the smallest dimension whose mean drop reaches the target;
    if the mean drop at d* exceeds the target by more than tol, bisection of ONE lam in (0, 1] at d*, shared by the draws (plan: only
    at d* = r0; Amendment 2 of 2026-11-12 extends it to every d*; the r0 case is unchanged); a target below tol is matched by the
    equal-dimension random subspace (lam 1). Matched = |drop - target| <= tol (G45). Returns dict(r, lam, drop, matched,
    ladder [(r, drop)], note); r = d*."""
    ladder = []
    if target < tol:
        d0 = float(drop_of(r0, 1.0)); return dict(r=r0, lam=1.0, drop=d0, matched=True, ladder=[(r0, d0)], note="target below tol")
    r = r0
    while True:
        dr = float(drop_of(r, 1.0)); ladder.append((r, dr))
        if dr >= target:
            if dr - target > tol:                                    # Amendment 2: partial removal at the found dimension d*, not only at r0
                lo, hi, lam, dl = 0.0, 1.0, 1.0, dr
                for _ in range(iters):
                    mid = 0.5 * (lo + hi); dm = float(drop_of(r, mid))
                    lam, dl = mid, dm
                    if abs(dm - target) <= tol:
                        break
                    if dm < target:
                        lo = mid
                    else:
                        hi = mid
                return dict(r=r, lam=lam, drop=dl, matched=abs(dl - target) <= tol, ladder=ladder,
                            note="partial removal at r0" if r == r0 else "partial removal at d* (Amendment 2)")
            return dict(r=r, lam=1.0, drop=dr, matched=abs(dr - target) <= tol, ladder=ladder, note="ladder")
        if 2 * r > W // 2:
            return dict(r=r, lam=1.0, drop=dr, matched=False, ladder=ladder, note="target not reached up to W/2")
        r *= 2


def modal_shift_map(pred, x, y, shifts):
    """OD-C16 modal-shift map of an operated net: per old task t the modal shift m_t = most frequent (pred - x - y) mod 17 over its
    held-out pairs and its share s_t; 'concentrated' if the modal shifts of >= 75 % of the tasks take at most 2 distinct values;
    'consistent permutation' if >= 75 % of the tasks have s_t >= 0.5 and their modal shifts are pairwise distinct; else 'scrambled'.
    pred, x, y: [n_tasks, N]; shifts: the tasks' true shifts. Also output integrity (mean s_t) and accuracy."""
    e = (np.asarray(pred) - np.asarray(x) - np.asarray(y)) % P; nt = e.shape[0]
    cnt = np.stack([np.bincount(e[i], minlength=P) for i in range(nt)]); m = cnt.argmax(1); sh = cnt.max(1) / e.shape[1]
    vals, vc = np.unique(m, return_counts=True); top2 = np.sort(vc)[::-1][:2].sum()
    strong = sh >= 0.5
    if top2 >= 0.75 * nt:
        lab = "concentrated"
    elif strong.sum() >= 0.75 * nt and len(set(m[strong].tolist())) == int(strong.sum()):
        lab = "consistent permutation"
    else:
        lab = "scrambled"
    acc = float(np.mean([(e[i] == shifts[i] % P).mean() for i in range(nt)]))
    return dict(modal=m.tolist(), share=sh.tolist(), label=lab, integrity=float(sh.mean()), accuracy=acc)


# ====================================================================================================================== RGI lines
def lambda_triples(pre, c_new):
    """Recipe triples of shift c_new (ref lattice.py / crit/sham.lam): every (a, b, c) with a < b, c not in {a, b} and
    pre[a] + pre[b] - pre[c] = c_new (mod 17)."""
    n = len(pre)
    return [(a, b, c) for a, b, c in itertools.product(range(n), repeat=3) if a < b and c not in (a, b) and (pre[a] + pre[b] - pre[c]) % P == c_new]


def lambda_targets(ws):
    """The 5 unseen shifts of Lambda: the 4 new shifts (held-out = split positions 8-288 of their slot) and the spare shift (all 289
    pairs). Returns list of (c, x, y)."""
    pre, new, pairs, splits = M.world(ws); out = []
    for j, c in enumerate(new):
        x, y = _pairs_of(splits[NPRE + j], NSHOT, P * P); out.append((c, x, y))
    out.append((spare_shift(ws), PAIRS[:, 0], PAIRS[:, 1]))
    return out


def lambda_line(nets, ws_of, pre_of=None, anchor_idx=None, ky_of=None, nperm=10, nulls=True, targets_of=None):
    """AT1 Lambda (primary): for each of the 5 unseen shifts c_new every recipe triple gives v = C_a + C_b - C_c (no shots, no learning);
    held-out accuracy averaged over triples, then shifts. Nulls: 'wrong' = every triple aimed at a different shift (target != c_new)
    evaluated on c_new; 'perm' = recipes built with permuted anchor labels (nperm permutations of the anchors' shift labels from
    default_rng(4000 + ws)), evaluated on c_new. pre_of / anchor_idx: per-net anchor shifts and task indices (default world pre and
    slots 0-11); targets_of: per-net list of (c, x, y) target sets (default lambda_targets(ws); e.g. without the spare shift for a net
    that pretrained on it). Returns list per net: dict(Lambda, per_shift, wrong, perm)."""
    out = []; T = nets.task_cols()
    for b in range(nets.B):
        ws = ws_of[b]; pre = list(M.world(ws)[0]) if pre_of is None else list(pre_of[b]); n = len(pre)
        ai = list(range(n)) if anchor_idx is None else list(anchor_idx[b] if np.ndim(anchor_idx[0]) else anchor_idx)
        Cb = T[b][:, ai]; ky = 1 if ky_of is None else ky_of[b]
        allt = np.array([(a, bb, c) for a, bb, c in itertools.product(range(n), repeat=3) if a < bb and c not in (a, bb)])
        aim = lambda lab_: (np.asarray(lab_)[allt[:, 0]] + np.asarray(lab_)[allt[:, 1]] - np.asarray(lab_)[allt[:, 2]]) % P
        target = aim(pre)
        perms = []
        if nulls:
            g = np.random.default_rng(4000 + ws); perms = [g.permutation(n) for _ in range(nperm)]
        ptarget = [aim(np.asarray(pre)[pm]) for pm in perms]
        per, wrong, perm = [], [], []
        for cn, x, y in (lambda_targets(ws) if targets_of is None else targets_of[b]):
            lab_ = _label(x, y, cn, ky); groups = [("main", np.nonzero(target == cn)[0])]
            if nulls:
                groups += [("wrong", np.nonzero(target != cn)[0])] + [("perm", np.nonzero(pt == cn)[0]) for pt in ptarget]
            sel = np.concatenate([g_[1] for g_ in groups]) if groups else np.zeros(0, np.int64)
            if len(sel) == 0:
                per.append(float("nan")); continue
            tt = torch.as_tensor(allt[sel], device=DEV); V = (Cb[:, tt[:, 0]] + Cb[:, tt[:, 1]] - Cb[:, tt[:, 2]]).T
            acc = cols_accuracy(nets, np.full(len(sel), b), V, (x, y, lab_)); pos = 0; pm_acc = []
            for kind, ix in groups:
                a_ = acc[pos:pos + len(ix)]; pos += len(ix)
                if kind == "main":
                    per.append(float(a_.mean()) if len(a_) else float("nan"))
                elif kind == "wrong":
                    wrong.append(float(a_.mean()))
                elif len(a_):
                    pm_acc.append(a_)
            if pm_acc:
                perm.append(float(np.concatenate(pm_acc).mean()))
        o = dict(Lambda=float(np.nanmean(per)), per_shift=per)
        if nulls:
            o.update(wrong=float(np.mean(wrong)), perm=float(np.mean(perm)) if perm else float("nan"))
        out.append(o)
    return out


def learn_alpha(nets, net_of, sets, steps=None, lr=FT_LR, q0=None, anchor_idx=None, pc_eta=None):
    """S line / W7 anchored search: alpha (12 coordinates over the centred anchors, from zero or q0) learned on the shots of sets (list of
    new_task_sets dicts, one per chain) with Adam lr 1e-2 for steps (300). Returns (alpha [C, 12], held-out accuracy at the end [C],
    column [C, W])."""
    steps = FT_STEPS if steps is None else steps
    prm = anchored_param(nets, net_of, anchor_idx)
    pc = None if pc_eta is None else dict(eta=pc_eta)
    r = column_search(nets, net_of, stack_sets(sets, "shot"), stack_sets(sets, "ho"), steps=steps, eval_at=(steps,), param=prm, q0=q0, lr=lr, pc=pc)
    return r["q"], r["acc"]["ho"][:, -1], r["v"]


def carry_alpha(nets, net_of, alpha, sets, anchor_idx=None, perms=None):
    """Carried anchored improvement: v_B = mean_B + C_B,centred alpha in net net_of[c]; with perms (list of permutations of the anchors)
    also the permuted-correspondence carriages. Returns (acc [C], acc_perm [C, n_perm] or None)."""
    v0, Bm = anchored_param(nets, net_of, anchor_idx); al = torch.as_tensor(alpha, dtype=F32, device=DEV)
    ho = stack_sets(sets, "ho"); acc = cols_accuracy(nets, net_of, v0 + (Bm @ al.unsqueeze(-1)).squeeze(-1), ho)
    if not perms:
        return acc, None
    pa = []
    for pm in perms:
        pt = torch.as_tensor(np.asarray(pm), device=DEV)
        pa.append(cols_accuracy(nets, net_of, v0 + (Bm[:, :, pt] @ al.unsqueeze(-1)).squeeze(-1), ho))
    return acc, np.stack(pa, 1)


def s_perms(ws, n=30, k=NPRE):
    """The 30 permuted correspondences of the S line for world ws: default_rng(31 + ws) permutations of the anchors (ref crit/sham_s)."""
    g = np.random.default_rng(31 + ws)
    return [g.permutation(k) for _ in range(n)]


def s_line(netsA, iA, alphaA, netsB, iB, sets, perms):
    """S-gain of carrying A's alphas (per new task, [4, 12]) into net iB of netsB: mean over tasks of acc_B(v_B) - mean over perms of
    acc_B(permuted carriage). Returns dict(S_gain, accB, accB_perm)."""
    nt = len(sets); net_of = np.full(nt, iB)
    acc, accp = carry_alpha(netsB, net_of, alphaA, sets, perms=perms)
    return dict(S_gain=float(np.mean(acc - accp.mean(1))), accB=acc.tolist(), accB_perm=accp.mean(1).tolist())


def ridge_coef(Ca, v, r):
    """Ref rgi_pilot.ridge_coef: ridge with lambda = r tr(Ca^T Ca) / n."""
    G_ = Ca.T @ Ca; lam = r * np.trace(G_) / G_.shape[0]
    return np.linalg.solve(G_ + lam * np.eye(G_.shape[0]), Ca.T @ v)


def t1_map(CA, CB, vA, r):
    """Ref rgi_pilot.t1_map: affine ridge map through the anchors: v_B = mean_B + (C_B - mean_B) a, a = ridge of v_A - mean_A on the centred
    anchors of A."""
    mA, mB = CA.mean(1), CB.mean(1); a = ridge_coef(CA - mA[:, None], vA - mA, r)
    return mB + (CB - mB[:, None]) @ a, a


T1_GRID = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)


def t1_choose_r(CA, CB, grid=T1_GRID):
    """Ref rgi_pilot.t1_choose_r: lambda by leave-one-anchor-out column error (relative)."""
    best = None
    for r in grid:
        err = 0.0
        for k in range(CA.shape[1]):
            keep = [i for i in range(CA.shape[1]) if i != k]; pred, _ = t1_map(CA[:, keep], CB[:, keep], CA[:, k], r)
            ref = CB[:, keep].mean(1); err += np.sum((pred - CB[:, k]) ** 2) / np.sum((CB[:, k] - ref) ** 2)
        if best is None or err < best[1]:
            best = (r, err / CA.shape[1])
    return best


def proj_linspan(C, v):
    """Projection of v on the linear span of the columns of C (least squares)."""
    a, *_ = np.linalg.lstsq(C, v, rcond=None)
    return C @ a


def f_line(netsA, iA, wsA, vfreeA, accfreeA, netsB=None, iB=None, nshuf=10, loo=True, nshuf_loo=10):
    """AT1 F (forcing). Phi = acc_A(projection of A's free B2 columns on the linear span of its 12 anchor columns) / acc_A(free) (mean
    over the 4 new tasks of the projected accuracy, divided by the mean free accuracy); with a receiver B: the free-column T1 gain
    (t1_map with lambda by leave-one-anchor-out column error, minus the mean of nshuf shuffled maps from default_rng(123 + ws), identity
    excluded) and the LOO-anchor transfer (each anchor k carried by T1 fitted on the other 11 into B's slot k, on task k's 145 held-out
    pairs; nshuf_loo shuffled maps default_rng(k 1000 + q)). vfreeA [4, W] (numpy), accfreeA [4] (the free columns' accuracies). A and
    B are nets of the same world (their anchors are the same 12 shifts). Phi uses the plan's linear span (the refs' affine hull differs)."""
    pre, new, pairs, splits = M.world(wsA)
    CA = netsA.task_cols()[iA][:, :NPRE].double().cpu().numpy(); sets = [new_task_sets(wsA, j) for j in range(NNEW)]
    proj = np.stack([proj_linspan(CA, np.asarray(v, np.float64)) for v in vfreeA])
    accp = cols_accuracy(netsA, np.full(NNEW, iA), proj, stack_sets(sets, "ho"))
    out = dict(Phi=float(np.mean(accp) / np.mean(accfreeA)) if np.mean(accfreeA) > 0 else float("nan"), acc_proj=accp.tolist())
    if netsB is None:
        return out
    CB = netsB.task_cols()[iB][:, :NPRE].double().cpu().numpy(); r1, e1 = t1_choose_r(CA, CB)
    rng = np.random.default_rng(123 + wsA); perms = [p for p in (rng.permutation(NPRE) for _ in range(nshuf)) if np.any(p != np.arange(NPRE))]
    cands = []
    for j in range(NNEW):
        cands.append(t1_map(CA, CB, vfreeA[j], r1)[0]); cands += [t1_map(CA, CB[:, p], vfreeA[j], r1)[0] for p in perms]
    hx, hy, hl = stack_sets(sets, "ho"); nper = 1 + len(perms)
    acc = cols_accuracy(netsB, np.full(len(cands), iB), np.stack(cands), (np.repeat(hx, nper, 0), np.repeat(hy, nper, 0), np.repeat(hl, nper, 0))).reshape(NNEW, nper)
    out.update(r1=r1, loo_err1=e1, t1=float(acc[:, 0].mean()), t1_shuf=float(acc[:, 1:].mean()), t1_gain=float((acc[:, 0] - acc[:, 1:].mean(1)).mean()))
    if loo:
        cands, xs, ys, ls = [], [], [], []
        for k in range(NPRE):
            keep = [i for i in range(NPRE) if i != k]
            c_ = [t1_map(CA[:, keep], CB[:, keep], CA[:, k], r1)[0]]
            c_ += [t1_map(CA[:, keep], CB[:, keep][:, np.random.default_rng(k * 1000 + q).permutation(NPRE - 1)], CA[:, k], r1)[0] for q in range(nshuf_loo)]
            x, y = _pairs_of(splits[k], 144, P * P); cands += c_
            xs += [x] * len(c_); ys += [y] * len(c_); ls += [_label(x, y, pre[k])] * len(c_)
        a = cols_accuracy(netsB, np.full(len(cands), iB), np.stack(cands), (np.stack(xs), np.stack(ys), np.stack(ls))).reshape(NPRE, 1 + nshuf_loo)
        out.update(loo=float(a[:, 0].mean()), loo_shuf=float(a[:, 1:].mean()))
    return out


def recipe_chains():
    """The 200 random recipe chains and 200 evaluation pairs of the recursion line (ref recipe_depth.py: default_rng(7) chains of a first
    triple and 5 two-anchor steps; pairs = pairs[default_rng(1).permutation(289)[:200]])."""
    rng = np.random.default_rng(7)
    chains = [[tuple(rng.choice(NPRE, 3, replace=False))] + [tuple(rng.choice(NPRE, 2, replace=False)) for _ in range(5)] for _ in range(200)]
    xy = PAIRS[np.random.default_rng(1).permutation(P * P)[:200]]
    return chains, xy


def recipe_depth(nets, ws_of, pre_of=None):
    """AT1 recursion: recipe depth d = 1..6 (v_1 = C_a + C_b - C_c, v_d = v_(d-1) + C_a' - C_b', target shift accumulated mod 17), the
    same 200 chains in every model; accuracy over the 200 pairs; returns per net dict(acc [6], rho [5]) with rho_d = acc(d+1) / acc(d)."""
    chains, xy = recipe_chains(); T = nets.task_cols(); out = []
    for b in range(nets.B):
        pre = list(M.world(ws_of[b])[0]) if pre_of is None else list(pre_of[b]); C = T[b][:, :NPRE]
        cols, labs = [], []
        for ch in chains:
            a, bb, c = ch[0]; v = C[:, a] + C[:, bb] - C[:, c]; t = (pre[a] + pre[bb] - pre[c]) % P
            for dpt in range(6):
                if dpt > 0:
                    a, bb = ch[dpt]; v = v + C[:, a] - C[:, bb]; t = (t + pre[a] - pre[bb]) % P
                cols.append(v); labs.append(_label(xy[:, 0], xy[:, 1], t))
        acc = cols_accuracy(nets, np.full(len(cols), b), torch.stack(cols), (xy[:, 0], xy[:, 1], np.stack(labs))).reshape(len(chains), 6).mean(0)
        out.append(dict(acc=acc.tolist(), rho=[float(acc[d + 1] / acc[d]) if acc[d] > 0 else float("nan") for d in range(5)]))
    return out


def round_trip(netsA, iA, netsB, iB, alphaA, sets, steps=40):
    """AT1 recursion: alpha_A refined in B for 40 steps on B's shots (fresh Adam, lr 1e-2), carried back to A, against A's own 40 further
    steps from alpha_A (ref rgi_pool.py). Returns dict(RG = mean over tasks of acc_A(back) - acc_A(own), back, own)."""
    nt = len(sets)
    aB, _, _ = learn_alpha(netsB, np.full(nt, iB), sets, steps=steps, q0=alphaA)
    own, acc_own, _ = learn_alpha(netsA, np.full(nt, iA), sets, steps=steps, q0=alphaA)
    back, _ = carry_alpha(netsA, np.full(nt, iA), aB, sets)
    return dict(RG=float(np.mean(back - acc_own)), back=back.tolist(), own=acc_own.tolist())


def pooling(nets, idx, alphas, sets):
    """AT1 recursion pooling: alpha averaged over the models idx (alphas [n_models, n_tasks, 12]) evaluated in each model against its own
    alpha. Returns dict(gain = mean over models and tasks of acc(pooled) - acc(own), pooled [n_models], own [n_models])."""
    pool = np.mean(alphas, 0); nt = len(sets); pa, oa = [], []
    for m, i in enumerate(idx):
        a1, _ = carry_alpha(nets, np.full(nt, i), pool, sets); a0, _ = carry_alpha(nets, np.full(nt, i), alphas[m], sets)
        pa.append(a1.mean()); oa.append(a0.mean())
    return dict(gain=float(np.mean(np.array(pa) - np.array(oa))), pooled=[float(v) for v in pa], own=[float(v) for v in oa])


def warm_start(netsB, iB, alphaA, sets, steps=(10, 40)):
    """AT1 warm start: B refines from the carried alpha; accuracy after 10 and 40 steps minus the same from zero. Returns dict steps ->
    gain (mean over tasks)."""
    nt = len(sets); out = {}
    for s in steps:
        _, a_w, _ = learn_alpha(netsB, np.full(nt, iB), sets, steps=s, q0=alphaA)
        _, a_0, _ = learn_alpha(netsB, np.full(nt, iB), sets, steps=s)
        out[s] = float(np.mean(a_w - a_0))
    return out


def stitch_fit(HA, HB, npair=P * P, grid=(1e-4, 1e-3, 1e-2, 1e-1)):
    """T3 (ref rgi_lib.stitch_fit): affine ridge map h2_A -> h2_B on the 12 pretraining tasks x 289 pairs; lambda by leave-3-tasks-out
    R2 (4 folds). Returns (S [W_A + 1, W_B], (lambda, R2))."""
    XA = np.concatenate([HA, np.ones((len(HA), 1))], 1); folds = [list(range(i, i + 3)) for i in range(0, NPRE, 3)]; best = None
    for r in grid:
        sr = st = 0.0
        for f in folds:
            te = np.concatenate([np.arange(k * npair, (k + 1) * npair) for k in f]); trn = np.setdiff1d(np.arange(len(HA)), te)
            G_ = XA[trn].T @ XA[trn]; lam = r * np.trace(G_) / G_.shape[0]; S = np.linalg.solve(G_ + lam * np.eye(G_.shape[0]), XA[trn].T @ HB[trn])
            sr += np.sum((XA[te] @ S - HB[te]) ** 2); st += np.sum((HB[te] - HB[trn].mean(0)) ** 2)
        if best is None or 1 - sr / st > best[1]:
            best = (r, 1 - sr / st)
    G_ = XA.T @ XA; lam = best[0] * np.trace(G_) / G_.shape[0]
    return np.linalg.solve(G_ + lam * np.eye(G_.shape[0]), XA.T @ HB), best


@torch.no_grad()
def all_h2(nets, i, task_idx=None):
    """h2 of net i on the 12 pretraining tasks x 289 pairs (float64 numpy [3468, W]; ref rgi_lib.all_h2)."""
    task_idx = list(range(NPRE)) if task_idx is None else task_idx
    x = torch.as_tensor(np.tile(PAIRS[:, 0], NPRE), device=DEV)[None]; y = torch.as_tensor(np.tile(PAIRS[:, 1], NPRE), device=DEV)[None]
    t = torch.as_tensor(np.repeat(task_idx, P * P), device=DEV)[None]
    sub = nets.subset([i]); return sub.hidden_acts(sub.features(x, y, t))[1][0].double().cpu().numpy()


def t3_t4(netsA, iA, vA, netsB, iB, sets, S=None, steps=None, do_t3=True, pcB_eta=None):
    """AT1 / AT5 T3 and T4 (ref rgi_lib.t34_transfer): A's column vA [n_tasks, W] gives h2_A and log p_A on the new task's 289 inputs;
    T4 fits B's column (from zero, Adam lr 1e-2, 300 steps) to KL(p_A || p_B) batchmean (PC receiver: pcB_eta, soft target in the
    energy); T3 fits it to the stitched target [h2_A, 1] S (S from stitch_fit; computed if None). Returns dict(t4_acc [n_tasks],
    t3_acc [n_tasks] or None, v4, v3) with accuracies on the held-out sets of sets."""
    steps = FT_STEPS if steps is None else steps; nt = len(sets)
    xa, ya = PAIRS[:, 0], PAIRS[:, 1]
    with torch.no_grad():
        lgA = _chain_logits(netsA, np.full(nt, iA), vA, (xa, ya)); lpA = torch.log_softmax(torch.as_tensor(lgA), -1).numpy()
    ho = stack_sets(sets, "ho"); shots = (np.tile(xa, (nt, 1)), np.tile(ya, (nt, 1)), np.zeros((nt, P * P), np.int64))
    pc = None if pcB_eta is None else dict(eta=pcB_eta)
    r4 = column_search(netsB, np.full(nt, iB), shots, ho, steps=steps, eval_at=(steps,), loss="kl", target=lpA, pc=pc)
    out = dict(t4_acc=r4["acc"]["ho"][:, -1], v4=r4["v"], t3_acc=None, v3=None)
    if do_t3:
        if pcB_eta is not None:
            raise ValueError("T3 into a PC receiver is not defined in the plan")
        if S is None:
            S, _ = stitch_fit(all_h2(netsA, iA), all_h2(netsB, iB))
        _, hA = hidden_states(netsA, np.full(nt, iA), vA, (np.tile(xa, (nt, 1)), np.tile(ya, (nt, 1))))
        hA = hA.double().cpu().numpy(); tgt = np.concatenate([hA, np.ones((nt, P * P, 1))], 2) @ S
        r3 = column_search(netsB, np.full(nt, iB), shots, ho, steps=steps, eval_at=(steps,), loss="mse", target=tgt.astype(np.float32))
        out.update(t3_acc=r3["acc"]["ho"][:, -1], v3=r3["v"])
    return out


# ====================================================================================================================== C-T0 / C-T6
def _orth(A):
    return np.linalg.qr(A)[0]


def fourier_cols(cv, k):
    """Ref c1_structure.fourier_cols: orthonormal basis of the centred (cos, sin)(2 pi k c / 17) regressors over the shifts cv."""
    Phi = np.stack([np.cos(2 * np.pi * k * cv / P), np.sin(2 * np.pi * k * cv / P)], 1)
    return _orth(Phi - Phi.mean(0))


def share_by_k(Tc, cv):
    """Ref c1_structure.share_by_k: share of the centred task code's energy on frequency k = 1..8."""
    tot = (Tc ** 2).sum()
    return np.array([((fourier_cols(cv, k).T @ Tc) ** 2).sum() / tot for k in range(1, 9)])


def ring_test(C, cv, g, nperm=1000):
    """Ref c1_structure.ring_test: max single-frequency share of the centred code C [n_tasks, D] against the max over nperm permutations
    of the shift labels (generator g)."""
    Tc = C - C.mean(0); s = share_by_k(Tc, cv); null = np.array([share_by_k(Tc, g.permutation(cv)).max() for _ in range(nperm)])
    return dict(share=s.round(3).tolist(), kmax=int(s.argmax() + 1), max=float(s.max()), null_max_mean=float(null.mean()),
                null_max_p99=float(np.quantile(null, 0.99)), p=float((1 + (null >= s.max()).sum()) / (1 + nperm)))


def loo_ring(C, cv, K, lam=1e-2):
    """Ref c1_structure.loo_ring: leave-one-shift-out ridge prediction of the held-out task column from [cos, sin](2 pi k c / 17),
    k = 1..K, plus intercept; returns R2."""
    Tc = C - C.mean(0); err, tot = 0.0, 0.0
    feats = lambda c: np.concatenate([[np.cos(2 * np.pi * k * c / P), np.sin(2 * np.pi * k * c / P)] for k in range(1, K + 1)])
    for i in range(len(cv)):
        tr = np.arange(len(cv)) != i
        A = np.stack([feats(c) for c in cv[tr]]); A = np.c_[A, np.ones(len(A))]; Y = Tc[tr]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y); pred = np.r_[feats(cv[i]), 1.0] @ w
        err += ((pred - Tc[i]) ** 2).sum(); tot += ((Tc[i] - Tc[tr].mean(0)) ** 2).sum()
    return 1 - err / tot


def spec_rows(Mat, g, nperm=500):
    """Ref c1_structure.spec_rows: DFT over the 17 answer-indexed rows of Mat [17, D]; top-1 / top-2 share of the non-DC power against
    answer-label permutations."""
    def sh(Mx):
        Fm = np.fft.fft(Mx - Mx.mean(0), axis=0); pw = (np.abs(Fm[1:9]) ** 2 + np.abs(Fm[16:8:-1]) ** 2).sum(1); pw = pw / pw.sum(); srt = np.sort(pw)[::-1]
        return pw, srt[0], srt[:2].sum()
    pw, t1, t2 = sh(Mat); n1, n2 = zip(*[sh(Mat[g.permutation(P)])[1:] for _ in range(nperm)])
    return dict(spec=pw.round(3).tolist(), top1=float(t1), top2=float(t2), null_top1_p99=float(np.quantile(n1, 0.99)), null_top2_p99=float(np.quantile(n2, 0.99)))


def subspace_overlap(C, cv, k):
    """C-T0 overlap(k) = ||Q_PC^T Q_F||_F^2 / (2k): Q_PC = the top-2k principal axes (in task space) of the code C [n_tasks, D] centred
    over tasks; Q_F = orthonormal basis of the k Fourier pairs with the largest variance share (share_by_k)."""
    Tc = C - C.mean(0); U, S, _ = np.linalg.svd(Tc, full_matrices=False); Qp = U[:, :2 * k]
    ks = np.argsort(-share_by_k(Tc, cv))[:k] + 1
    Phi = np.concatenate([np.stack([np.cos(2 * np.pi * f * cv / P), np.sin(2 * np.pi * f * cv / P)], 1) for f in ks], 1)
    Qf = _orth(Phi - Phi.mean(0))
    return float(np.sum((Qp.T @ Qf) ** 2) / (2 * k))


@torch.no_grad()
def structure_tests(nets, i, ws, nperm=1000, nperm_loo=1000, nperm_spec=500, ks=(1, 2, 3), overlap_ks=(1, 2)):
    """C-T0 basis-free structure tests of net i (ref c1_structure.py; generator default_rng(500 + ws) used in c1's order): (a) task
    columns, (b) h1 / h2 task centroids over all 289 pairs: ring_test with nperm permutations; (c) leave-one-shift-out ring R2 for
    K in ks against nperm_loo shuffled labels (p95); (d) answer-code spectrum of the h2 answer-class means and of W3 (spec_rows);
    overlap(k) of the task code. Returns dict."""
    g = np.random.default_rng(500 + ws); pre = M.world(ws)[0]; cv = np.array(pre, float)
    T = nets.task_cols()[i][:, :NPRE].double().cpu().numpy().T
    x = torch.as_tensor(np.tile(PAIRS[:, 0], NPRE), device=DEV)[None]; y = torch.as_tensor(np.tile(PAIRS[:, 1], NPRE), device=DEV)[None]
    t = torch.as_tensor(np.repeat(np.arange(NPRE), P * P), device=DEV)[None]; sub = nets.subset([i])
    H1, H2 = (h[0].double().cpu().numpy() for h in sub.hidden_acts(sub.features(x, y, t)))
    lab_task = np.repeat(np.arange(NPRE), P * P); lab_ans = np.concatenate([(PAIRS.sum(1) + c) % P for c in pre])
    C1 = np.stack([H1[lab_task == k].mean(0) for k in range(NPRE)]); C2 = np.stack([H2[lab_task == k].mean(0) for k in range(NPRE)])
    R = dict(a_taskcols=ring_test(T, cv, g, nperm), b_h1=ring_test(C1, cv, g, nperm), b_h2=ring_test(C2, cv, g, nperm))
    R["c_loo_ring_R2"] = {f"K{K}": dict(real=float(loo_ring(T, cv, K)), null_p95=float(np.quantile([loo_ring(T, g.permutation(cv), K) for _ in range(nperm_loo)], 0.95))) for K in ks}
    Ma = np.stack([H2[lab_ans == v].mean(0) for v in range(P)]); W3 = nets.p["W3"][i].double().cpu().numpy()
    R["d_h2_answer"] = spec_rows(Ma, g, nperm_spec); R["d_W3"] = spec_rows(W3, g, nperm_spec)
    R["overlap"] = {f"k{k}": subspace_overlap(T, cv, k) for k in overlap_ks}
    return R


def trajectory_modes(X, k=5):
    """C-T6 (ref c4_traj.modes): SVD of the snapshot matrix X [n_snap, dim] centred over time; per mode i the variance share and
    |corr(time course, cos(pi i (t + 0.5) / T))| (Lissajous template) and the emergence snapshot. Returns (list of dicts, top-5 var)."""
    Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False); var = S ** 2 / (S ** 2).sum(); T = len(X); t = np.arange(T); out = []
    for i in range(k):
        tc = U[:, i] * S[i]; lis = np.cos(np.pi * (i + 1) * (t + 0.5) / T); r = abs(np.corrcoef(tc, lis)[0, 1])
        fin = tc[-1] - tc[0]; half = np.argmax(np.abs(tc - tc[0]) >= 0.5 * abs(fin)) if abs(fin) > 0 else -1
        out.append(dict(var=float(var[i]), lissajous_r=float(r), emerge_snapshot=int(half)))
    return out, float(var[:5].sum())


def trajectory_basis(X, L):
    """Top-L mode directions (right singular vectors of X centred over time), [dim, L] (C-T6 causal cap basis for train(task_basis=))."""
    Xc = X - X.mean(0); _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:L].T


def shuffled_increment_nulls(X, n, rng, k=5):
    """C-T6 null (ref c4_traj): the same increments in shuffled time order; returns list of trajectory_modes outputs."""
    dX = np.diff(X, axis=0); out = []
    for _ in range(n):
        Xn = np.concatenate([X[:1], X[:1] + np.cumsum(dX[rng.permutation(len(dX))], 0)]); out.append(trajectory_modes(Xn, k)[0])
    return out


# ====================================================================================================================== I/O
def weight_checksum(nets, i):
    """sha256 (hex) of net i's float32 parameters (sorted names, CPU contiguous bytes)."""
    h = hashlib.sha256()
    for k in sorted(nets.p):
        h.update(k.encode()); h.update(nets.p[k][i].detach().to("cpu", F32).contiguous().numpy().tobytes())
    return h.hexdigest()


def _atomic_save(obj, path):
    tmp = path + ".tmp"; torch.save(obj, tmp); os.replace(tmp, path)


def save_nets(nets, idx, folder, names=None, extra=None):
    """Save nets idx (one file <folder>/<name>.pt each, atomically): dict(state (pc_common.MLP keys for depth-2 'mlp' nets, else this
    class's names), arch (kind, W, depth, act, param), tables (fx, fy, ft), meta, checksum). Returns the paths."""
    os.makedirs(folder, exist_ok=True); paths = []
    for n, i in enumerate(idx):
        name = (names[n] if names is not None else nets.meta[i].get("name")) or f"net{i}"
        obj = dict(state=nets.state(i), arch=dict(kind=nets.kind, W=nets.W, depth=nets.depth, act=nets.act, param=nets.param),
                   tables=[nets.fx[i].cpu(), nets.fy[i].cpu(), nets.ft[i].cpu()], meta=_jsonable(nets.meta[i]), checksum=weight_checksum(nets, i),
                   extra=_jsonable(extra[n]) if extra is not None else None)
        p = os.path.join(folder, name + ".pt"); _atomic_save(obj, p); paths.append(p)
    return paths


def _jsonable(o):
    return json.loads(json.dumps(o, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))


def load_nets(paths, verify=True):
    """Load files written by save_nets (one architecture) into one Nets; verifies every checksum."""
    objs = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]; a = objs[0]["arch"]
    states = []
    for o in objs:
        st = o["state"]
        if "l1.weight" in st:
            st = {"W1": st["l1.weight"], "b1": st["l1.bias"], "W2": st["l2.weight"], "b2": st["l2.bias"], "W3": st["l3.weight"], "b3": st["l3.bias"]}
        states.append(st)
    tables = [np.stack([o["tables"][j].numpy() for o in objs]) for j in range(3)]
    nets = Nets.from_states(states, kind=a["kind"], param=a["param"], act=a["act"], tables=tables, meta=[o["meta"] for o in objs])
    if verify:
        for i, o in enumerate(objs):
            if weight_checksum(nets, i) != o["checksum"]:
                raise ValueError(f"checksum mismatch: {paths[i]}")
    return nets


def load_pc_state_dicts(paths):
    """Round-1 PC-muP (or BP) checkpoints of pc_meta (state dicts with pc_common.MLP keys) -> Nets (kind 'mlp', muP, identity tables);
    meta holds the file name and the checksum."""
    sds = [torch.load(p, map_location="cpu") for p in paths]
    nets = Nets.from_states(sds, kind="mlp", param="mup", meta=[dict(name=os.path.basename(p)[:-3], path=p) for p in paths])
    for i in range(nets.B):
        nets.meta[i]["checksum"] = weight_checksum(nets, i)
    return nets


def r1_training_batch(ws, K=NPRE):
    """pc_meta's training batch of world ws (one-hot X [1728, 51], y) on the device (lam_max input)."""
    pre, new, pairs, splits = M.world(ws)
    tr = [M.data(pre[k], k, pairs[splits[k][:144]]) for k in range(K)]
    return torch.cat([d_[0] for d_ in tr]).to(DEV), torch.cat([d_[1] for d_ in tr]).to(DEV)


# ====================================================================================================================== selftest
REF_FILES = {
    "rgi_lib": "rgi_design/rgi_lib.py", "rgi_pilot": "rgi_design/rgi_pilot.py", "lattice": "rgi_design/lattice.py",
    "recipe_depth": "rgi_design/recipe_depth.py", "rgi_pool": "rgi_design/rgi_pool.py", "rgi_bneck": "rgi_design/rgi_bneck.py",
    "sham": "rgi_design/crit/sham.py", "sham_s": "rgi_design/crit/sham_s.py", "family2": "rgi_design/crit/family2.py",
    "impose2": "unlearn_design/impose2.py", "c1_structure": "unlearn_design/critique/c1_structure.py",
    "c2_pring": "unlearn_design/critique/c2_pring.py", "c3_angle": "unlearn_design/critique/c3_angle.py",
    "c4_traj": "unlearn_design/critique/c4_traj.py", "c5_twofam": "unlearn_design/critique/c5_twofam.py",
    "rgi_long": "rgi_design/rgi_long.py"}
_SCRATCH = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"


def ref_path(key):
    """Location of an Appendix-E reference file: round2/ref/<relpath>, round2/ref/<basename>, $R2REF/<relpath>, the design scratchpad."""
    rel = REF_FILES[key]
    for base in (os.path.join(_HERE, "ref"), os.environ.get("R2REF", ""), _SCRATCH):
        if not base:
            continue
        for cand in (os.path.join(base, rel), os.path.join(base, os.path.basename(rel))):
            if os.path.exists(cand):
                return cand
    return None


def ref_defs(key, names, extra_ns=None):
    """Execute only the named top-level function / class definitions of a reference file (no module-level code) in a namespace that
    has M, P, NPRE, NNEW, NSLOT, BASE, np, torch, nn, Fn, math, itertools. Returns the namespace (None if the file is missing)."""
    path = ref_path(key)
    if path is None:
        return None
    src = open(path).read(); tree = ast.parse(src)
    ns = dict(M=M, P=P, NPRE=NPRE, NNEW=NNEW, NSLOT=NSLOT, BASE=BASE, np=np, torch=torch, nn=torch.nn, Fn=Fn, math=math, itertools=itertools,
              MLP=PCC.MLP, accuracy=PCC.accuracy, time=time, os=os, sys=sys, json=json)
    if extra_ns:
        ns.update(extra_ns)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)
    return ns


class _Patch:
    """Temporarily set module attributes."""

    def __init__(self, mod, **kw):
        self.mod, self.kw = mod, kw

    def __enter__(self):
        self.old = {k: getattr(self.mod, k) for k in self.kw}
        for k, v in self.kw.items():
            setattr(self.mod, k, v)

    def __exit__(self, *a):
        for k, v in self.old.items():
            setattr(self.mod, k, v)


def _rel(a, b):
    a = torch.as_tensor(a).double(); b = torch.as_tensor(b).double()
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-30))


class _Check:
    """Collects test results with their details (both sides, largest difference) for the committed comparison output (9.5)."""

    def __init__(self):
        self.res, self.rows, self.t0 = {}, [], time.time()

    def __call__(self, name, cond, info="", **detail):
        self.res[name] = bool(cond)
        self.rows.append(dict(check=name, passed=bool(cond), info=_jsonable(info), **{k: _jsonable(v) for k, v in detail.items()}))
        print(f"[{'PASS' if cond else 'FAIL'}] {name}: {info} ({time.time() - self.t0:.0f}s)", flush=True)
        return bool(cond)


def _r1_batch(ws, K=NPRE):
    pre, new, pairs, splits = M.world(ws)
    tr = [M.data(pre[k], k, pairs[splits[k][:144]]) for k in range(K)]; va = [M.data(pre[k], k, pairs[splits[k][144:]]) for k in range(K)]
    X, y = torch.cat([d_[0] for d_ in tr]), torch.cat([d_[1] for d_ in tr])
    lab = torch.arange(K).repeat_interleave(144).to(DEV)
    lab_x = torch.cat([torch.as_tensor(pairs[splits[k][:144]][:, 0]) for k in range(K)]).to(DEV)      # crit/sham.py's lab_x
    return X, y, torch.cat([d_[0] for d_ in va]), torch.cat([d_[1] for d_ in va]), lab, lab_x


def _ref_train(arm, W, ws, init, steps, gamma=0.0, reg="task", wds=1.0):
    """Per-net reference: pc_meta.make_opt + pc_meta.bp_update(gamma, lab) (task regulariser) or crit/sham.py's update (sham)."""
    X, y, _, _, lab, lab_x = _r1_batch(ws)
    torch.manual_seed(init_seed(ws, init)); m = PCC.MLP(2 * P + NSLOT, W, P, param="sp" if arm == "bp_sp" else "mup", base_width=BASE).to(DEV)
    opt = M.make_opt(m, wds)
    for _ in range(steps):
        if reg == "task":
            M.bp_update(m, opt, X, y, gamma, lab)
        else:
            opt.zero_grad(); h1, h2 = m.acts(X)
            (Fn.cross_entropy(m.head(h2), y) + gamma * torch.log(M.task_fisher(h2, lab_x, P))).backward(); opt.step()
    return m


def _state_rel(nets, i, sd):
    st = nets.state(i)
    return max(_rel(st[k], v) for k, v in sd.items())


def _within(a, b, tol):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.abs(a - b).reshape(len(a), -1).max(1) <= tol + 1e-9


def _fixtures(work):
    """pc_meta QUICK runs as subprocesses (QUICK=1 NT=1 python pc_meta.py <dir> ...), which write jobs/*.json and models/*.pt."""
    fx = os.path.join(work, "fx"); pm = os.path.join(_JU, "pc_meta.py")
    env = dict(os.environ, QUICK="1", NT="1", OMP_NUM_THREADS="1", DEVICE="cpu", **PC_ENV)
    for args in (["bp_mup", "64", "0"], ["bp_sp", "64", "1"], ["pc_mup", "64", "0"], ["bp_mup", "64", "0", "1"]):
        subprocess.run([sys.executable, pm, fx] + args, env=env, check=True, capture_output=True)
    return fx


def _pc_relax_autograd_soft(m, x, q, eta_x, steps, beta=BETA_PC):
    """pc_meta.pc_relax_autograd with the soft-target output term of W7 (beta CE(q, logits) = -beta sum_k q_k log softmax_k): the same
    construction (feedforward init, per-sample energy pc_energy x B, autograd gradient steps on h1, h2)."""
    with torch.no_grad():
        h1, h2 = m.acts(x)
    h1 = h1.clone().requires_grad_(True); h2 = h2.clone().requires_grad_(True); B = x.shape[0]
    for p in m.parameters():
        p.requires_grad_(False)
    for _ in range(steps):
        e1 = h1 - torch.tanh(m.l1(x)); e2 = h2 - torch.tanh(m.l2(h1))
        E = (0.5 * (e1 ** 2).sum(-1).mean() + 0.5 * (e2 ** 2).sum(-1).mean() + beta * Fn.cross_entropy(m.head(h2), q)) * B
        g1, g2 = torch.autograd.grad(E, (h1, h2))
        with torch.no_grad():
            h1 -= eta_x * g1; h2 -= eta_x * g2
    for p in m.parameters():
        p.requires_grad_(True)
    return h1.detach(), h2.detach()


def selftest(work=None):
    """Code tests (QUICK sizes): plan W0's equivalence tests against pc_meta.py and the comparison of the ported functions with the
    reference scripts of Appendix E (round2/ref/ copies, definitions executed in a namespace with the repository's pc_meta / pc_common =
    the 9.5 shim). Prints one line per check, writes <work>/ref_comparison_w5.json (both sides and largest differences) and raises
    SystemExit(1) if a check fails."""
    ck = _Check(); work = work or os.path.join(os.getcwd(), "r2tc_selftest"); os.makedirs(work, exist_ok=True)
    print(f"r2_toy_common selftest: device {DEV}, QUICK {QUICK}, torch {torch.__version__}, work {work}", flush=True)
    mup = lambda **kw: dict(W=64, **kw)
    ck("pc_meta environment set before import and asserted (W0)", M.T == 20 and PCC.BETA[0] == 0.1 and M.LR == 1e-3,
       f"T {M.T} BETA {PCC.BETA[0]} LR {M.LR} MAXSTEPS {M.MAXSTEPS} QUICK {M.QUICK}")

    # ---- (1) init: every kind against its reference constructor (bitwise)
    for s in (mup(ws=0), mup(ws=1, init=2), mup(ws=2, arm="bp_sp")):
        n = Nets.from_specs([s]); torch.manual_seed(init_seed(s["ws"], s.get("init", 0)))
        m = PCC.MLP(2 * P + NSLOT, 64, P, param="sp" if s.get("arm") == "bp_sp" else "mup", base_width=BASE)
        e = _state_rel(n, 0, m.state_dict()); ck(f"init {net_name(s)} = pc_common.MLP", e == 0.0, max_rel_diff=e)
    ns = ref_defs("family2", ["Net"])
    if ns:
        n = Nets.from_specs([mup(ws=1, depth=3)]); torch.manual_seed(1); r = ns["Net"]("tanh", 3, 64)
        ok = all(torch.equal(a.cpu(), b) for a, b in ((n.p["W1"][0], r.l1.weight), (n.p["W2"][0], r.hs[0].weight), (n.p["W2_1"][0], r.hs[1].weight), (n.p["W3"][0], r.l3.weight)))
        ck("init depth 3 = crit/family2.Net", ok)
    ns = ref_defs("c3_angle", ["Net", "four"], {"W": 64})
    if ns:
        n = Nets.from_specs([dict(kind="circle", W=64, ws=3, circ="ring", cK=1)]); fx_ = ns["four"](1); torch.manual_seed(3); r = ns["Net"](fx_, fx_)
        ok = torch.equal(n.p["W1"][0].cpu(), torch.cat([r.Ax, r.Ay, r.Ac], 1)) and torch.equal(n.p["W2"][0].cpu(), r.l2.weight) and np.allclose(four(2), ns["four"](2).numpy())
        ck("init circle = c3_angle.Net; four() = c3_angle.four", ok)
    ns = ref_defs("impose2", ["Net", "code_table"])
    if ns:
        ok = all(torch.equal(code_table(kd, k_, 4), ns["code_table"](kd, k_, 4)) for kd in ("ring", "rand") for k_ in (2, 3))
        n = Nets.from_specs([dict(kind="impose", W=64, ws=4, code="rand", ck=3)]); torch.manual_seed(4); r = ns["Net"](64, ns["code_table"]("rand", 3, 4))
        ok = ok and torch.equal(n.p["W1"][0].cpu(), torch.cat([r.Ex, r.A, r.t0[:, None]], 1)) and torch.equal(n.p["W2"][0].cpu(), r.l2.weight)
        ck("init impose = impose2.Net; code_table ring / rand = impose2.code_table", ok)
        ns2 = ref_defs("c2_pring", ["code_table"], {"I": type("I", (), dict(P=P, code_table=staticmethod(ns["code_table"]))), "_orig": ns["code_table"]})
        if ns2:
            ck("code_table pring = c2_pring.code_table", all(torch.equal(code_table("pring", k_, s_), ns2["code_table"]("pring", k_, s_)) for k_ in (2, 4) for s_ in (0, 3)))
    ns = ref_defs("rgi_bneck", ["MLPB"])
    if ns:
        n = Nets.from_specs([dict(kind="bneck", W=64, ws=1, dt=4)]); torch.manual_seed(1); r = ns["MLPB"](64, 4)
        ok = torch.equal(n.p["W1"][0].cpu(), r.l1.weight[:, :2 * P].detach()) and torch.equal(n.p["U"][0].cpu(), r.U.detach()) and torch.equal(n.p["E"][0].cpu(), r.E.detach())
        ck("init bneck = rgi_bneck.MLPB", ok)

    # ---- (2) W0: batched trainer against pc_meta, 300 updates (weights to 1e-4 relative)
    for arm in ("bp_mup", "bp_sp"):
        specs = [mup(ws=0, arm=arm), mup(ws=1, init=1, arm=arm), mup(ws=2, arm=arm, wd=3.0)]
        res = train(specs, fixed_steps=300, coord_check=False)
        rels = [_state_rel(res.nets, b, _ref_train(arm, 64, s["ws"], s.get("init", 0), 300, wds=s.get("wd", 1.0)).state_dict()) for b, s in enumerate(specs)]
        ck(f"W0 trainer = pc_meta 300 updates ({arm}, 3 nets W 64)", max(rels) <= 1e-4, f"max rel {max(rels):.1e}", max_rel_diff=rels)
    specs = [mup(ws=0, gamma=0.01), mup(ws=0, init=1, gamma=0.01)]
    res = train(specs, fixed_steps=300, coord_check=False)
    rels = [_state_rel(res.nets, b, _ref_train("bp_mup", 64, 0, s.get("init", 0), 300, 0.01).state_dict()) for b, s in enumerate(specs)]
    ck("W0 trainer = pc_meta.bp_update(gamma 0.01, lab), task regulariser, 300 updates", max(rels) <= 1e-4, f"max rel {max(rels):.1e}", max_rel_diff=rels)
    specs = [mup(ws=1, gamma=0.003), mup(ws=1, gamma=-0.003), mup(ws=1, gamma=-0.01), mup(ws=1, gamma=0.003, reg="sham"), mup(ws=1, gamma=-0.01, reg="sham", init=1)]
    res = train(specs, fixed_steps=300, coord_check=False)
    rels = [_state_rel(res.nets, b, _ref_train("bp_mup", 64, 1, s.get("init", 0), 300, s["gamma"], s.get("reg", "task")).state_dict()) for b, s in enumerate(specs)]
    ck("trainer = pc_meta / crit/sham.py update with the plan's early levels (task +-0.003, -0.01; sham +0.003, -0.01), 300 updates",
       max(rels) <= 1e-4, f"rel {np.format_float_scientific(max(rels), 1)}", max_rel_diff=rels)
    X, y, _, _, lab, lab_x = _r1_batch(0); torch.manual_seed(0); m0 = PCC.MLP(51, 64, P, param="mup", base_width=BASE).to(DEV); h2 = m0.acts(X)[1].detach()
    X1, _, _, _, _, lab_x1 = _r1_batch(1); h2b = m0.acts(X1)[1].detach()
    ref_t = [float(M.task_fisher(h2, lab_x, P)), float(M.task_fisher(h2b, lab_x1, P))]
    grp = [torch.as_tensor(np.nonzero(lab_x.cpu().numpy() == k)[0], device=DEV) for k in range(P) if (lab_x == k).any()]
    rep = float(task_fisher_b(h2[None], None, P, grp)[0]); one = task_fisher_b(torch.stack([h2, h2b]), torch.stack([lab_x, lab_x1]), P).tolist()
    err = max(abs(rep - ref_t[0]) / ref_t[0], max(abs(a - b) / b for a, b in zip(one, ref_t)))
    ck("W0 sham Fisher term = crit/sham.py's M.task_fisher(h2, lab_x, 17) on the same batch (1e-6; replica and one-hot forms)", err <= 1e-6,
       f"rel {err:.1e}", mine=[rep] + one, ref=ref_t, max_rel_diff=err)

    # ---- (3) checkpoint rule: pc_meta QUICK subprocess fixtures, and pc_meta.main in-process with a non-trivial LEVEL1
    fx = _fixtures(work)
    res = train([mup(ws=0)], max_steps=200, after=50, level1=0.0)
    sd = torch.load(os.path.join(fx, "models", "bp_mup_W64_s0_K12_g0_wd1_quick.pt")); job = json.load(open(os.path.join(fx, "jobs", "bp_mup_W64_s0_K12_g0_wd1_quick.json")))
    r = res.rows[0]; e = _state_rel(res.nets, 0, sd)
    ck("W0 first95 / checkpoint / weights = pc_meta under QUICK", r["first95"] == job["first95"] and r["checkpoint"] == job["checkpoint"] and e <= 1e-4,
       f"{r['first95']}/{r['checkpoint']} vs {job['first95']}/{job['checkpoint']}", mine=[r["first95"], r["checkpoint"]], ref=[job["first95"], job["checkpoint"]], max_rel_diff=e)
    ck("coordinate check = pc_meta's G0 values", np.allclose(r["coord_check"], job["coord_check"], rtol=1e-4, atol=1e-7), mine=r["coord_check"], ref=job["coord_check"])
    mdir = os.path.join(work, "main")
    if os.path.isdir(os.path.join(mdir, "jobs")):
        for f_ in os.listdir(os.path.join(mdir, "jobs")):
            os.remove(os.path.join(mdir, "jobs", f_))
    ftl = (5, 10, 20); argv = sys.argv
    try:
        with _Patch(M, MAXSTEPS=500, AFTER=100, LEVEL1=0.3, FT_STEPS=20, FT_EVAL=ftl, QUICK=False):
            for a in (["bp_mup", "64", "0"], ["bp_sp", "64", "1"], ["bp_mup", "64", "0", "1"]):
                sys.argv = ["pc_meta.py", mdir] + a; M.main()
    finally:
        sys.argv = argv
    jm = json.load(open(os.path.join(mdir, "jobs", "bp_mup_W64_s0_K12_g0_wd1.json"))); js = json.load(open(os.path.join(mdir, "jobs", "bp_sp_W64_s1_K12_g0_wd1.json")))
    j1 = json.load(open(os.path.join(mdir, "jobs", "bp_mup_W64_s0_K1_g0_wd1.json")))
    rm = train([mup(ws=0)], max_steps=500, after=100, level1=0.3); rs = train([mup(ws=1, arm="bp_sp")], max_steps=500, after=100, level1=0.3)
    ok = rm.rows[0]["first95"] == jm["first95"] and rm.rows[0]["checkpoint"] == jm["checkpoint"] and rs.rows[0]["first95"] is None and js["first95"] is None
    ok = ok and rs.rows[0]["eval_point"] == "maxsteps" and abs(rs.rows[0]["final_heldout"] - js["final_heldout"]) < 1e-6
    e = _state_rel(rm.nets, 0, torch.load(os.path.join(mdir, "models", "bp_mup_W64_s0_K12_g0_wd1.pt")))
    ck("checkpoint rule = pc_meta.main with LEVEL1 0.3 (a level-1 net and a MAXSTEPS net)", ok and e <= 1e-4,
       f"first95 {rm.rows[0]['first95']} vs {jm['first95']}, MAXSTEPS net held-out {rs.rows[0]['final_heldout']:.4f} vs {js['final_heldout']:.4f}",
       mine=[rm.rows[0]["first95"], rm.rows[0]["checkpoint"], rs.rows[0]["final_heldout"]], ref=[jm["first95"], jm["checkpoint"], js["final_heldout"]], max_rel_diff=e)
    r1 = train([mup(ws=0, K=1)], fixed_steps=j1["checkpoint"], coord_check=False)
    e = _state_rel(r1.nets, 0, torch.load(os.path.join(mdir, "models", "bp_mup_W64_s0_K1_g0_wd1.pt")))
    ck("K = 1 control (fixed steps) = pc_meta.main K 1", e <= 1e-4, f"{j1['checkpoint']} updates", max_rel_diff=e)
    dl = delta_stats(rm.nets, [0], [M.world(0)[0]])[0]
    err = max(abs(dl[h][k] - jm["delta"][h][k]) for h in ("h1", "h2") for k in dl[h])
    ck("delta_stats = pc_meta job's delta (1e-6)", err <= 1e-6, f"max abs diff {err:.1e}", mine=dl, ref=jm["delta"], max_abs_diff=err)
    bb = b2(rm.nets, [0], steps=20, eval_at=ftl)
    eA = max(np.abs(bb[ns_]["A"][0] - np.array(jm["finetune"][str(ns_)]["A"])).max() for ns_ in (8, 4))
    eF = max(np.abs(bb[ns_]["F"][0] - np.array(jm["finetune"][str(ns_)]["F"])).max() for ns_ in (8, 4))
    ck("B2 (A, F, B2) = pc_meta job's finetune", eA <= 1 / 281 + 1e-9 and eF <= 1 / 281 + 1e-9 and abs(bb[8]["B2"][0] - jm["finetune"]["8"]["B2"]) <= 1 / 281,
       f"max |dA| {eA:.4f} |dF| {eF:.4f}", mine=[float(bb[8]["B2"][0]), float(bb[4]["B2"][0])], ref=[jm["finetune"]["8"]["B2"], jm["finetune"]["4"]["B2"]], max_abs_diff=max(eA, eF))

    ba = b2_all(rm.nets, [0], steps=20, eval_at=ftl)
    eAll = float(np.abs(ba["acc"][0] - np.array(jm["finetune"]["all8"]["acc"])).max()); eR = float(np.abs(ba["retention"][0] - np.array(jm["finetune"]["all8"]["retention"])).max())
    ck("B2_all and retention = pc_meta job's finetune all8", eAll <= 1 / 281 + 1e-9 and eR <= 1 / 1740 + 1e-9, f"max |d acc| {eAll:.4f} |d retention| {eR:.5f}",
       mine=ba["retention"][0], ref=jm["finetune"]["all8"]["retention"], max_abs_diff=max(eAll, eR))

    # ---- (4) W0: batched column search and fresh nets against pc_meta.finetune (300 steps, FT_EVAL of the plan); delta_stats
    ev = (10, 20, 40, 80, 160, 300)
    nb = train([mup(ws=0), mup(ws=1)], fixed_steps=700, coord_check=False).nets; ns_sp = train([mup(ws=2, arm="bp_sp")], fixed_steps=700, coord_check=False).nets
    tot = good = 0; mine_l, ref_l = [], []
    for nets_, wsl, arm in ((nb, [0, 1], "bp_mup"), (ns_sp, [2], "bp_sp")):
        A = b2_A(nets_, wsl, steps=300, eval_at=ev)
        for b, ws in enumerate(wsl):
            pre, new, pairs, splits = M.world(ws); m = nets_.to_mlp(b).to(DEV)
            with _Patch(M, FT_STEPS=300, FT_EVAL=ev):
                for ns_ in (8, 4):
                    for j in range(NNEW):
                        a_ref, _ = M.finetune(m, arm, 64, new, pairs, splits, j, ns_, "col", ws); tot += 1; good += int(_within([A[ns_][b, j]], [a_ref], 1 / 281)[0])
                        mine_l.append(A[ns_][b, j].tolist()); ref_l.append(a_ref)
    ck("W0 column search = pc_meta.finetune('col') within 1/281 in >= 95 % of chains", good >= 0.95 * tot, f"{good}/{tot}", mine=mine_l, ref=ref_l,
       max_abs_diff=float(np.abs(np.array(mine_l) - np.array(ref_l)).max()))
    Fm = fresh_F("bp_mup", 64, [0, 1], steps=300, eval_at=ev); tot = good = 0; mine_l, ref_l = [], []
    with _Patch(M, FT_STEPS=300, FT_EVAL=ev):
        for si, ws in enumerate((0, 1)):
            pre, new, pairs, splits = M.world(ws)
            for ns_ in (8, 4):
                for j in range(NNEW):
                    f_ref, _ = M.finetune(None, "bp_mup", 64, new, pairs, splits, j, ns_, "fresh", ws); tot += 1; good += int(_within([Fm[ns_][si, j]], [f_ref], 1 / 281)[0])
                    mine_l.append(Fm[ns_][si, j].tolist()); ref_l.append(f_ref)
    ck("fresh F = pc_meta.finetune('fresh') within 1/281 in >= 95 % of chains", good >= 0.95 * tot, f"{good}/{tot}", mine=mine_l, ref=ref_l,
       max_abs_diff=float(np.abs(np.array(mine_l) - np.array(ref_l)).max()))
    dls = delta_stats(nb, [0, 1], [M.world(0)[0], M.world(1)[0]]); err = 0.0; refs = []
    for b, ws in enumerate((0, 1)):
        ref = M.delta_stats(nb.to_mlp(b).to(DEV), M.world(ws)[0], M.world(ws)[2], ws); refs.append(ref)
        err = max(err, max(abs(dls[b][h][k] - ref[h][k]) for h in ("h1", "h2") for k in ref[h]))
    ck("W0 vectorised delta_stats = pc_meta.delta_stats (1e-6)", err <= 1e-6, f"max abs diff {err:.1e}", mine=dls, ref=refs, max_abs_diff=err)

    # ---- (5) PC: lam_max, relaxation (one-hot and soft targets), PC column search; erasers in the energy against autograd
    pcn = load_pc_state_dicts([os.path.join(fx, "models", "pc_mup_W64_s0_K12_g0_wd1_T20_beta0.1_quick.pt")])
    X, y, _, _, _, _ = _r1_batch(0); m = pcn.to_mlp(0).to(DEV); Xb, yb = X[None].to(DEV), y[None].to(DEV)
    with _BetaCtx(BETA_PC, T_PC):
        torch.manual_seed(5); lam_ref = M.lam_max(m, X, y)
        eta = pc_eta(pcn, 0, X, y, seed=5)
        h1r, h2r, ratio_r, _ = M.pc_relax(m, X, y, 1.0 / lam_ref, T_PC)
    h1, h2, ratio = pc_relax_inputs(pcn, Xb, yb, eta)
    e = max(_rel(h1[0], h1r), _rel(h2[0], h2r))
    ck("W0 pc_eta = 1 / pc_meta.lam_max; batched relaxation = pc_meta.pc_relax (1e-5)", abs(eta - 1 / lam_ref) <= 1e-12 and e <= 1e-5
       and abs(float(ratio[0]) - ratio_r) <= 1e-4 * max(abs(ratio_r), 1e-6), f"states rel {e:.1e}, ratio {float(ratio[0]):.5f} vs {ratio_r:.5f}",
       mine=[eta, float(ratio[0])], ref=[1 / lam_ref, ratio_r], max_rel_diff=e)
    s1, s2, _ = pc_relax_inputs(pcn, Xb, yb, eta, soft=Fn.one_hot(yb, P).float())
    e1_ = max(_rel(s1[0], h1r), _rel(s2[0], h2r))
    q = torch.softmax(torch.randn(X.shape[0], P, generator=torch.Generator().manual_seed(11)) * 2, -1).to(DEV)
    a1, a2 = _pc_relax_autograd_soft(m, X.to(DEV), q, eta, T_PC); t1, t2, _ = pc_relax_inputs(pcn, Xb, yb, eta, soft=q[None])
    e2_ = max(_rel(t1[0], a1), _rel(t2[0], a2))
    ck("W0 soft-target relaxation (W7): one-hot q = pc_meta.pc_relax, soft q = autograd of the soft-target energy as pc_relax_autograd (1e-5)",
       e1_ <= 1e-5 and e2_ <= 1e-5, f"rel {e1_:.1e} / {e2_:.1e}", max_rel_diff=[e1_, e2_])
    torch.manual_seed(0); mp = PCC.MLP(51, 64, P, param="mup", base_width=BASE).to(DEV); opt = M.make_opt(mp)
    with _BetaCtx(BETA_PC, T_PC):
        for s in range(1, 1001):
            if s % 100 == 1:
                ex = 1.0 / M.lam_max(mp, X, y)
            M.pc_update(mp, opt, X, y, ex)
    pcn = Nets.from_states([{k: v.detach().cpu() for k, v in mp.state_dict().items()}]); eta = pc_eta(pcn, 0, X, y, seed=1)
    A = b2_A(pcn, [0], steps=300, eval_at=ev, pc_eta_of=[eta]); pre, new, pairs, splits = M.world(0); tot = good = 0; m = pcn.to_mlp(0).to(DEV); mine_l, ref_l = [], []
    with _Patch(M, FT_STEPS=300, FT_EVAL=ev), _BetaCtx(BETA_PC, T_PC):
        for ns_ in (8, 4):
            for j in range(NNEW):
                a_ref, _ = M.finetune(m, "pc_mup", 64, new, pairs, splits, j, ns_, "col", 0, eta); tot += 1; good += int(_within([A[ns_][0, j]], [a_ref], 1 / 281)[0])
                mine_l.append(A[ns_][0, j].tolist()); ref_l.append(a_ref)
    ck("PC column search = pc_meta.finetune('col', pc_mup) within 1/281 in >= 95 % of chains", good >= 0.95 * tot, f"{good}/{tot}, final accuracies {np.round(A[8][0, :, -1], 3).tolist()}",
       mine=mine_l, ref=ref_l, max_abs_diff=float(np.abs(np.array(mine_l) - np.array(ref_l)).max()))
    ok = _check_pc_autograd(pcn, eta)
    ck("PC energy with erasers at h1 / h2 and soft targets: closed-form relaxation and column gradient = autograd", ok[0], ok[1])

    # ---- (6) W7 search semantics
    net = nb; sets = [new_task_sets(0, j, val=True) for j in range(NNEW)]; no = np.zeros(NNEW, np.int64)
    r = column_search(net, no, stack_sets(sets, "shot"), {"ho": stack_sets(sets, "ho"), "val": stack_sets(sets, "val")}, val=stack_sets(sets, "val"),
                      budgets=(20, 60), eval_at=tuple(range(10, 61, 10)))
    okb = True
    for bud in (20, 60):
        nv = bud // 10; va = r["acc"]["val"][:, :nv]; bi = va.argmax(1)
        okb &= np.allclose(r["best"][bud]["heldout"]["ho"], r["acc"]["ho"][np.arange(NNEW), bi]) and np.allclose(r["best"][bud]["step"], (bi + 1) * 10)
    ck("best-of-budget = held-out at the first best validation evaluation within the budget", okb)
    cols = np.random.default_rng(0).standard_normal((NNEW, 64)).astype(np.float32)
    rr = column_search(net, no, stack_sets(sets, "shot"), stack_sets(sets, "ho"), steps=0, eval_at=(0,), q0=cols)
    ck("recipe (no learning, steps 0) = cols_accuracy", np.allclose(rr["acc"]["ho"][:, 0], cols_accuracy(net, no, cols, stack_sets(sets, "ho"))))
    Q = random_subspaces(64, 3, NNEW, 1)
    rr = column_search(net, no, stack_sets(sets, "shot"), stack_sets(sets, "ho"), steps=30, eval_at=(30,), remove=Q)
    rk = column_search(net, no, stack_sets(sets, "shot"), stack_sets(sets, "ho"), steps=30, eval_at=(30,), keep=Q)
    vr, vk = torch.as_tensor(rr["v"], device=DEV), torch.as_tensor(rk["v"], device=DEV)
    inq = lambda v: ((Q.transpose(1, 2) @ v.unsqueeze(-1)).squeeze(-1).norm(dim=1) / v.norm(dim=1))
    ck("restricted search: remove -> v orthogonal to the removed set, keep -> v inside the subspace", float(inq(vr).max()) < 1e-5 and float(inq(vk).min()) > 1 - 1e-5)
    ns = ref_defs("rgi_pilot", ["base_z1", "heads", "acc_cols", "ridge_coef", "t1_map", "t1_choose_r"])
    if ns:
        nsl = ref_defs("rgi_lib", ["learn_alpha", "stitch_fit", "_fit_col", "t34_transfer", "all_h2"], dict(base_z1=ns["base_z1"], acc_cols=ns["acc_cols"]))
        g = dict(m=net.to_mlp(0), C=net.task_cols()[0][:, :NPRE].double().cpu().numpy())
        al, acc_al, _ = learn_alpha(net, no, [new_task_sets(0, j) for j in range(NNEW)])
        al_ref = np.stack([nsl["learn_alpha"](g, 0, j, steps=FT_STEPS) for j in range(NNEW)])
        ck("anchored search (alpha) = rgi_lib.learn_alpha", _rel(al, al_ref) <= 1e-3, f"rel {_rel(al, al_ref):.1e}", mine=al[0], ref=al_ref[0], max_rel_diff=_rel(al, al_ref))
        arrs = [tuple(np.random.default_rng(i).standard_normal((3, 40, 12))[:2]) + (np.random.default_rng(i + 9).standard_normal(40),) for i in range(3)]
        ok = all(_rel(t1_map(*a_, 0.01)[0], ns["t1_map"](*a_, 0.01)[0]) == 0 for a_ in arrs)
        CA, CB = np.random.default_rng(5).standard_normal((2, 40, 12)); ok = ok and t1_choose_r(CA, CB) == ns["t1_choose_r"](CA, CB)
        HA, HB = np.random.default_rng(6).standard_normal((2, NPRE * 20, 7))
        Sa, ba = stitch_fit(HA, HB, npair=20); Sb, bb_ = nsl["stitch_fit"](HA, HB, npair=20)
        ck("t1_map / t1_choose_r / stitch_fit = rgi_pilot / rgi_lib (exact)", ok and np.array_equal(Sa, Sb) and ba == bb_, mine=[float(ba[0]), float(ba[1])], ref=[float(bb_[0]), float(bb_[1])])
        lam_ref = ref_defs("sham", ["lam"], dict(acc_cols=ns["acc_cols"]))
        if lam_ref:
            L = lambda_line(net, [0, 1]); mine_v = [float(np.mean(L[b]["per_shift"][:NNEW])) for b in range(2)]; ref_v = [lam_ref["lam"](net.to_mlp(b), ws) for b, ws in enumerate((0, 1))]
            ck("Lambda over the 4 new shifts = crit/sham.lam", max(abs(a - b) for a, b in zip(mine_v, ref_v)) <= 1e-6,
               f"Lambda (5 shifts) {L[0]['Lambda']:.3f}, null wrong {L[0]['wrong']:.3f}, null perm {L[0]['perm']:.3f}", mine=mine_v, ref=ref_v)
        src = open(ref_path("recipe_depth")).read().splitlines()
        nsr = dict(np=np, NPRE=NPRE, P=P, pairs=PAIRS, rng=np.random.default_rng(7)); exec("\n".join(src[7:9]), nsr)
        ch, xy_ = recipe_chains(); ok = ch == nsr["chains"] and np.array_equal(xy_, nsr["xy"])
        rd = recipe_depth(net, [0, 1]); C_ = g["C"]; pre = M.world(0)[0]; acc1, mine = [], []; Cn = net.task_cols()[0][:, :NPRE]
        for c_ in ch[:20]:
            a_, b_, cc_ = c_[0]; v = C_[:, a_] + C_[:, b_] - C_[:, cc_]; t = (pre[a_] + pre[b_] - pre[cc_]) % P
            acc1.append(ns["acc_cols"](g["m"], xy_, torch.as_tensor((xy_[:, 0] + xy_[:, 1] + t) % P), torch.tensor(v[None], dtype=torch.float32))[0])
            mine.append(cols_accuracy(net, [0], (Cn[:, a_] + Cn[:, b_] - Cn[:, cc_])[None], (xy_[:, 0], xy_[:, 1], (xy_[:, 0] + xy_[:, 1] + t) % P))[0])
        ck("recipe chains / pairs = recipe_depth.py; depth-1 accuracies = rgi_pilot.acc_cols", ok and np.allclose(mine, acc1, atol=1 / 200 + 1e-9),
           f"rho {np.round(rd[0]['rho'], 3).tolist()}", mine=mine, ref=acc1)
        nsp = ref_defs("rgi_pool", ["learn_alpha_from"], dict(base_z1=ns["base_z1"]))
        if nsp:
            a40, _, _ = learn_alpha(net, no, [new_task_sets(0, j) for j in range(NNEW)], steps=40, q0=al)
            a40r = np.stack([nsp["learn_alpha_from"](g, 0, j, al[j], 40) for j in range(NNEW)])
            ck("warm start / round-trip refinement = rgi_pool.learn_alpha_from", _rel(a40, a40r) <= 1e-3, f"rel {_rel(a40, a40r):.1e}", max_rel_diff=_rel(a40, a40r))
        if nsl:
            nb2 = train([mup(ws=0, init=1)], fixed_steps=700, coord_check=False).nets
            vA = column_search(net, no, stack_sets([new_task_sets(0, j) for j in range(NNEW)], "shot"), steps=300)["v"]
            A_ = dict(m=net.to_mlp(0), v=list(vA.astype(np.float64)), W=64); B_ = dict(m=nb2.to_mlp(0), W=64)
            S_, _ = stitch_fit(all_h2(net, 0), all_h2(nb2, 0))
            out = t3_t4(net, 0, vA, nb2, 0, [new_task_sets(0, j) for j in range(NNEW)], S=S_, steps=300)     # rgi_lib._fit_col: 300 steps
            v3r, v4r = zip(*[nsl["t34_transfer"](A_, B_, 0, S_, j) for j in range(NNEW)])
            e3, e4 = _rel(out["v3"], np.stack(v3r)), _rel(out["v4"], np.stack(v4r))
            ck("T3 / T4 columns = rgi_lib.t34_transfer", e3 <= 1e-3 and e4 <= 1e-3, f"rel v3 {e3:.1e} v4 {e4:.1e}; T4 acc {np.round(out['t4_acc'], 3).tolist()}", max_rel_diff=[e3, e4])
            nspt = ref_defs("rgi_lib", ["pair_transfer", "span_transfer", "learn_alpha"], dict(base_z1=ns["base_z1"], acc_cols=ns["acc_cols"], t1_choose_r=ns["t1_choose_r"],
                            t1_map=ns["t1_map"], t2_choose_r=ref_defs("rgi_pilot", ["t2_choose_r", "t2_map", "ridge_coef"])["t2_choose_r"],
                            t2_map=ref_defs("rgi_pilot", ["t2_map", "ridge_coef"])["t2_map"]))
            accfree = cols_accuracy(net, no, vA, stack_sets([new_task_sets(0, j) for j in range(NNEW)], "ho"))
            fl = f_line(net, 0, 0, vA, accfree, nb2, 0, nshuf=30, nshuf_loo=10)
            Aall = lambda nn: np.concatenate([nn.p["W1"][0][:, :2 * P + NPRE].double().cpu().numpy(), nn.p["b1"][0].double().cpu().numpy()[:, None]], 1)
            A2 = dict(m=net.to_mlp(0), C=g["C"], Aall=Aall(net), v=list(vA.astype(np.float64)), W=64, self=list(accfree))
            B2_ = dict(m=nb2.to_mlp(0), C=nb2.task_cols()[0][:, :NPRE].double().cpu().numpy(), Aall=Aall(nb2), W=64, self=[0.0] * NNEW)
            pt = nspt["pair_transfer"](A2, B2_, 0, nperm=30, nloo_perm=10)
            keys = [("t1", "t1"), ("t1_shuf", "shuf"), ("loo", "loo"), ("loo_shuf", "loo_shuf")]
            dmax = max(abs(fl[a] - pt[b]) for a, b in keys)
            ck("F line (T1 gain, LOO-anchor) = rgi_lib.pair_transfer", dmax <= 1 / 281 + 1e-9 and fl["r1"] == pt["r1"], f"Phi {fl['Phi']:.3f}, T1 gain {fl['t1_gain']:+.3f}",
               mine={a: fl[a] for a, _ in keys}, ref={b: pt[b] for _, b in keys}, max_abs_diff=dmax)
            al4 = {j: al[j].astype(np.float64) for j in range(NNEW)}; A2["alpha"] = al4
            sp = nspt["span_transfer"](A2, B2_, 0, nperm=30); rng_ = np.random.default_rng(321); acc_b, acc_s = [], []
            for j in range(NNEW):
                pm = [rng_.permutation(NPRE) for _ in range(30)]
                a_, ap_ = carry_alpha(nb2, [0], al[j][None], [new_task_sets(0, j)], perms=pm); acc_b.append(a_[0]); acc_s.append(ap_[0].mean())
            dmax = max(abs(np.mean(acc_b) - sp["spanB"]), abs(np.mean(acc_s) - sp["span_shuf"]))
            ck("S line carriage (alpha into B, permuted correspondences) = rgi_lib.span_transfer", dmax <= 1 / 281 + 1e-9,
               mine=[float(np.mean(acc_b)), float(np.mean(acc_s))], ref=[sp["spanB"], sp["span_shuf"]], max_abs_diff=dmax)
            sl_ = s_line(net, 0, al, nb2, 0, [new_task_sets(0, j) for j in range(NNEW)], s_perms(0))
            rt = round_trip(net, 0, nb2, 0, al, [new_task_sets(0, j) for j in range(NNEW)])
            po = pooling(Nets.cat([net.subset([0]), nb2]), [0, 1], np.stack([al, learn_alpha(nb2, no, [new_task_sets(0, j) for j in range(NNEW)])[0]]), [new_task_sets(0, j) for j in range(NNEW)])
            wst = warm_start(nb2, 0, al, [new_task_sets(0, j) for j in range(NNEW)])
            ck("S-gain, round trip, pooling, warm start run", all(np.isfinite(v) for v in (sl_["S_gain"], rt["RG"], po["gain"], wst[10], wst[40])),
               f"S-gain {sl_['S_gain']:+.3f} RG {rt['RG']:+.3f} pooling {po['gain']:+.3f} warm {wst}")

    # ---- (7) C-T0 / C-T6 ports
    nsc = ref_defs("c1_structure", ["orth", "fourier_cols", "share_by_k", "ring_test", "loo_ring", "spec_rows", "ridge_probe", "mlp_probe"])
    if nsc:
        T_ = net.task_cols()[0][:, :NPRE].double().cpu().numpy().T; cv = np.array(M.world(0)[0], float)
        a1, a2 = ring_test(T_, cv, np.random.default_rng(1), 50), nsc["ring_test"](T_, cv, np.random.default_rng(1), 50)
        Ma = np.random.default_rng(2).standard_normal((P, 30))
        ok = a1 == a2 and loo_ring(T_, cv, 2) == nsc["loo_ring"](T_, cv, 2) and spec_rows(Ma, np.random.default_rng(3), 50) == nsc["spec_rows"](Ma, np.random.default_rng(3), 50)
        Htr, Hte = np.random.default_rng(4).standard_normal((2, 60, 9)); ytr, yte = np.random.default_rng(5).integers(0, 4, (2, 60))
        rp, rp_ref = float(ridge_probe(Htr, ytr, Hte, yte, 4, lam=1.0)), nsc["ridge_probe"](Htr, ytr, Hte, yte, 4, lam=1.0)
        ok = ok and abs(rp - rp_ref) < 1e-12
        F2 = np.stack([np.cos(2 * np.pi * 3 * cv / P), np.sin(2 * np.pi * 3 * cv / P)], 1) @ np.random.default_rng(6).standard_normal((2, 20))
        ok = ok and abs(subspace_overlap(F2, cv, 1) - 1) < 1e-9 and subspace_overlap(np.random.default_rng(7).standard_normal((NPRE, 30)), cv, 1) < 1
        st_ = structure_tests(net, 0, 0, nperm=20, nperm_loo=10, nperm_spec=20)
        ck("ring_test / loo_ring / spec_rows / ridge_probe = c1_structure (exact); overlap(pure Fourier code) = 1", ok,
           f"task-col max share {st_['a_taskcols']['max']:.3f} (null p99 {st_['a_taskcols']['null_max_p99']:.3f})", mine=[a1["max"], rp], ref=[a2["max"], rp_ref])
    nst = ref_defs("c4_traj", ["modes"])
    if nst:
        Xs = np.cumsum(np.random.default_rng(8).standard_normal((40, 25)), 0); mine, _ = trajectory_modes(Xs); ref, _ = nst["modes"](Xs)
        ck("trajectory_modes = c4_traj.modes", all(abs(a["var"] - b["var"]) < 1e-3 and abs(a["lissajous_r"] - b["lissajous_r"]) < 1e-3 and a["emerge_snapshot"] == b["emerge_snapshot"] for a, b in zip(mine, ref)),
           mine=[a["var"] for a in mine], ref=[b["var"] for b in ref])

    # ---- (8) other worlds' training against their reference scripts (30 updates)
    _check_worlds(ck)

    # ---- (9) late regulariser: crit/sham.py's train(x_late), the main loop's 'late' and the train_late branch
    nss = ref_defs("sham", ["train", "pdist"], dict(mk=lambda arm, W: PCC.MLP(51, W, P, param="mup", base_width=BASE)))
    with _Patch(M, MAXSTEPS=500, AFTER=100, LEVEL1=0.3):
        if nss:
            m_s, info = nss["train"](0, 0, "x_late", -0.01, W=64); m_n, _ = nss["train"](0, 0, "nat", 0.0, W=64)
        nat = train([mup(ws=0)], max_steps=500, after=100, level1=0.3, save_first95=True, coord_check=False)
        lt = train([mup(ws=0, gamma=-0.01, reg="sham", when="late")], max_steps=500, after=100, level1=0.3, coord_check=False)
        br = train_late(nat, [0], [mup(ws=0, gamma=-0.01, reg="sham", when="late")], n_updates=100, level1=0.3)
    if nss:
        e = _state_rel(lt.nets, 0, m_s.state_dict()); pdm, pdr = param_distance(lt.nets, 0, nat.nets, 0), nss["pdist"](m_s, m_n)
        ck("late sham regulariser = crit/sham.py train(x_late); param_distance = sham.pdist", info["first"] == lt.rows[0]["first95"] and e <= 1e-4 and abs(pdm - pdr) <= 1e-4,
           f"rel {e:.1e}, p {pdm:.4f} vs {pdr:.4f}", mine=[lt.rows[0]["first95"], pdm], ref=[info["first"], pdr], max_rel_diff=e)
    e = _state_rel(br.nets, 0, lt.nets.state(0))
    ck("train_late branch from the first95 state = the main loop's late run", e <= 1e-6 and br.rows[0]["level1"] == (lt.rows[0]["level_acc"] >= 0.3), f"rel {e:.1e}")
    with _Patch(M, MAXSTEPS=500, AFTER=100, LEVEL1=0.3):
        ex = train([mup(ws=0), mup(ws=2)], max_steps=500, after=100, level1=0.3, extra_after=50, snapshot_every=25, snapshot=snap_task_code, coord_check=False)
    ok = _state_rel(ex.nets, 0, nat.nets.state(0)) == 0 and ex.has2[0] and ex.rows[0]["second"] == ex.rows[0]["checkpoint"] + 50 and len(ex.snapshots) >= 1 + ex.rows[0]["checkpoint"] // 25
    ck("second checkpoint (checkpoint state unchanged) and snapshots", ok, f"second {ex.rows[0]['second']}, snapshots {len(ex.snapshots)}")
    u = torch.randn(64, NPRE, generator=torch.Generator().manual_seed(0)); u = u - u.mean(1, keepdim=True); u = (u / u.norm()).reshape(1, -1, 1)
    capped = train([mup(ws=1)], fixed_steps=60, task_basis=u, coord_check=False); n0 = Nets.from_specs([mup(ws=1)])
    dT = (capped.nets.task_cols()[0][:, :NPRE] - n0.task_cols()[0][:, :NPRE]).double(); dc = (dT - dT.mean(1, keepdim=True)).reshape(-1)
    ub = u.reshape(-1).double().to(dc.device)
    res_ = float((dc - ub * (ub @ dc)).norm() / dc.norm())          # float32 storage of the columns: about 4e-5 after 60 updates; uncapped about 1
    ck("C-T6 cap: the centred task-column change lies in the basis", res_ < 1e-3, f"residual share {res_:.1e}")

    # ---- (10) W6, W8, OD-C2, OD-C16
    w6 = self_train(nb.subset([0]), [0], taus=(0.1, 2.0), rounds=2, tasks=[0], steps=20)
    w6b = self_train(nb.subset([0]), [0], taus=(0.1, 2.0), rounds=2, tasks=[0], steps=20)
    ck("W6 self-training loop runs and is deterministic", w6["A"].shape == (4, 3) and np.array_equal(w6["A"], w6b["A"]) and len(w6_summary(w6)) == 2, f"A {np.round(w6['A'], 3).tolist()}")
    _, h2_, x_, y_, sh_ = task_states(nb, [0, 1])
    Q_, mu_ = class_mean_eraser(h2_.reshape(2, -1, 64), torch.arange(NPRE).repeat_interleave(P * P), NPRE)
    He = Ops(h2_Q=Q_, h2_mu=mu_).view_per_net(2).on_h2(h2_.reshape(2, 1, -1, 64))[:, 0].reshape(2, NPRE, P * P, 64)
    spread = float((He.mean(2) - He.mean((1, 2), keepdim=True)[:, :, 0]).abs().max())
    ck("class-mean eraser: all class means equal after the erasure", spread < 1e-5, f"max |class mean - mean| {spread:.1e}")
    sp0 = sum_probe(nb, [0, 1]); ref_ = z1_ref(nb); Ex_ = nb.x_cols(); Exc = Ex_ - Ex_.mean(-1, keepdim=True)
    Qx = torch.linalg.qr(Exc[:, :, :P - 1].double())[0].float(); spx = sum_probe(nb, [0, 1], ops=block_ops(Qx, ref_))
    ck("sum probe high in trained nets and destroyed by the x-span block (generic-damage positive control)", sp0.min() > 0.5 and spx.max() < sp0.min() - 0.2,
       f"base {np.round(sp0, 3).tolist()}, x-span block {np.round(spx, 3).tolist()}")
    pr = task_answer_probes(task_answer_features(nb, [0, 1]), mlp=True)
    ck("task-identity / pooled-answer probes run (linear, MLP)", all(np.all((v >= 0) & (v <= 1)) for v in pr.values()), {k: np.round(v, 2).tolist() for k, v in pr.items()})
    f_ = lambda r_, lam: min(1.0, 0.1 * r_ * lam)
    dm, dm2, dm3 = damage_match(0.39, f_, 1, 64), damage_match(0.37, f_, 1, 64), damage_match(0.05, lambda r_, lam: 0.2 * r_ * lam, 1, 64)
    ok = (dm["matched"] and dm["r"] == 4 and dm["lam"] == 1.0 and dm2["matched"] and dm2["r"] == 4 and 0 < dm2["lam"] < 1
          and dm2["note"] == "partial removal at d* (Amendment 2)" and dm3["note"] == "partial removal at r0" and dm3["matched"] and dm3["r"] == 1)
    dm4 = damage_match(0.9, lambda r_, lam: min(0.3, 0.01 * r_ * lam), 1, 64)        # never reached up to W / 2: unmatched, lam 1
    ok = ok and not dm4["matched"] and dm4["lam"] == 1.0 and dm4["r"] == 32
    ck("OD-C2 damage_match (ladder; overshoot at d* > r0 matched by partial removal, Amendment 2; bisection at r0; unreachable target unmatched)", ok,
       f"{dm['ladder']}; d* bisection lam {dm2['lam']:.3f}; r0 bisection lam {dm3['lam']:.3f}")
    e_ = np.zeros((4, 10), int); xx = np.zeros((4, 10), int)
    perm_map = modal_shift_map(e_ + np.array([[1], [2], [3], [4]]), xx, xx, [0, 1, 2, 3]); conc = modal_shift_map(e_ + 5, xx, xx, [0, 1, 2, 3])
    ck("OD-C16 modal-shift map labels", perm_map["label"] == "consistent permutation" and conc["label"] == "concentrated")

    # ---- (10b) further API checks: task accuracy, chain relaxation, PC alpha / T4 into a PC receiver, coordinate searches, helpers
    ta = task_accuracy(nb, [0, 1]); ref_ta = []
    for b, ws in enumerate((0, 1)):
        pre, _, pairs, splits = M.world(ws); m = nb.to_mlp(b).to(DEV)
        ref_ta.append([PCC.accuracy(m, *M.data(pre[k], k, pairs[splits[k][144:]])) for k in range(NPRE)])
    ck("task_accuracy = pc_common.accuracy per old task", np.abs(ta - np.array(ref_ta)).max() <= 1 / 145 + 1e-9, mine=ta.mean(1), ref=np.mean(ref_ta, 1), max_abs_diff=float(np.abs(ta - np.array(ref_ta)).max()))
    pre, _, pairs, splits = M.world(0); k_ = 3; xk, yk = _pairs_of(splits[k_], 0, 144)
    hc1, hc2, _ = pc_relax(pcn, [0], pcn.task_cols()[0][:, k_][None], (xk, yk, _label(xk, yk, pre[k_])), eta)
    with _BetaCtx(BETA_PC, T_PC):
        Xk, yk_ = M.data(pre[k_], k_, pairs[splits[k_][:144]]); r1_, r2_, _, _ = M.pc_relax(pcn.to_mlp(0).to(DEV), Xk, yk_, eta, T_PC)
    e = max(_rel(hc1[0], r1_), _rel(hc2[0], r2_))
    ck("chain pc_relax (task column on its pairs) = pc_meta.pc_relax on the task's batch (1e-5)", e <= 1e-5, f"rel {e:.1e}", max_rel_diff=e)
    sets0 = [new_task_sets(0, j) for j in range(NNEW)]; no1 = np.zeros(NNEW, np.int64)
    alp, accp, _ = learn_alpha(pcn, no1, sets0, steps=20, pc_eta=np.full(NNEW, eta))
    vA0 = column_search(nb, no1, stack_sets(sets0, "shot"), steps=20)["v"]
    t4p = t3_t4(nb, 0, vA0, pcn, 0, sets0, steps=20, do_t3=False, pcB_eta=np.full(NNEW, eta))
    ck("PC anchored search and T4 into a PC receiver (soft target) run", np.all(np.isfinite(alp)) and np.all(np.isfinite(t4p["v4"])), f"alpha acc {np.round(accp, 3).tolist()} T4 acc {np.round(t4p['t4_acc'], 3).tolist()}")
    bn = train([dict(kind="bneck", W=64, ws=0, dt=4)], fixed_steps=200, coord_check=False).nets
    rbn = column_search(bn, no1, stack_sets(sets0, "shot"), stack_sets(sets0, "ho"), steps=20, eval_at=(20,), param=coords_param(bn, no1, "bneck"))
    vb = torch.as_tensor(rbn["v"], device=DEV); U_ = bn.p["U"][0]; inU = float(((U_ @ torch.linalg.lstsq(U_, vb.T).solution).T - vb).norm() / vb.norm())
    im = train([dict(kind="impose", W=64, ws=0, code="ring", ck=2)], fixed_steps=200, coord_check=False).nets
    rim = column_search(im, no1, stack_sets(sets0, "shot"), stack_sets(sets0, "ho"), steps=20, eval_at=(20,), param=coords_param(im, no1, "impose"), lr=np.array([1e-2, 1e-1, 1e-2, 1e-1]))
    ck("coordinate searches (bneck: v in span U; impose code space with per-chain lr) run", inU < 1e-4 and rim["q"].shape == (NNEW, 4), f"bneck acc {np.round(rbn['acc']['ho'][:, 0], 3).tolist()}")
    Xs_ = np.cumsum(np.random.default_rng(9).standard_normal((30, 20)), 0); Bs = trajectory_basis(Xs_, 3); nl = shuffled_increment_nulls(Xs_, 3, np.random.default_rng(1))
    Qs = random_in_span(torch.linalg.qr(torch.randn(64, 11, dtype=F64, generator=torch.Generator().manual_seed(4)))[0].float(), 2, 5, 3)
    ck("trajectory_basis / shuffled_increment_nulls / random_in_span shapes", Bs.shape == (20, 3) and len(nl) == 3 and Qs.shape == (5, 64, 2))
    T0 = nb.task_cols()[:, :, :NPRE]; edit = T0.mean(-1, keepdim=True).expand_as(T0).transpose(1, 2)
    acc_e = task_accuracy(nb, [0, 1], cols=edit); pr_ = predictions(nb, np.zeros(NPRE, np.int64), edit[0], (np.stack([_pairs_of(M.world(0)[3][k], 144, P * P)[0] for k in range(NPRE)]), np.stack([_pairs_of(M.world(0)[3][k], 144, P * P)[1] for k in range(NPRE)])))
    xs_ = np.stack([_pairs_of(M.world(0)[3][k], 144, P * P)[0] for k in range(NPRE)]); ys_ = np.stack([_pairs_of(M.world(0)[3][k], 144, P * P)[1] for k in range(NPRE)])
    msm = modal_shift_map(pr_, xs_, ys_, M.world(0)[0])
    ck("edit (task columns collapsed to their mean) and the modal-shift map run", acc_e.mean() < ta.mean() and abs(msm["accuracy"] - acc_e[0].mean()) < 1e-6, f"edit acc {np.round(acc_e.mean(1), 3).tolist()} map {msm['label']}")

    md = mediators(nb, [0, 1]); Qc = random_subspaces(64, 5, 1, 2)[0]; Qp = complement_basis(Qc)
    okc = float((Qc.T @ Qp).abs().max()) < 1e-5 and float((Qp.T @ Qp - torch.eye(59)).abs().max()) < 1e-5
    A_op = b2_A(nb, [0, 1], nshots=(8,), steps=20, eval_at=(20,), ops=block_ops(Qx, ref_)); A_no = b2_A(nb, [0, 1], nshots=(8,), steps=20, eval_at=(20,))
    A_kp = b2_A(nb, [0, 1], nshots=(8,), steps=20, eval_at=(20,), keep=torch.stack([Qc, Qc]))
    ck("mediators, complement_basis, B2 with an operation on and with a kept column subspace", okc and np.all(md["saturation"] >= 0) and A_op[8].mean() < A_no[8].mean(),
       f"saturation {np.round(md['saturation'], 3).tolist()} col norm {np.round(md['task_col_norm'], 2).tolist()} A {A_no[8].mean():.3f} x-block {A_op[8].mean():.3f} kept-5 {A_kp[8].mean():.3f}")

    # ---- (11) I/O
    paths = save_nets(nb, [0, 1], os.path.join(work, "saved")); ld = load_nets(paths)
    ck("save_nets / load_nets round trip with checksums; deterministic names", all(weight_checksum(ld, i) == weight_checksum(nb, i) for i in range(2)) and net_name(mup(ws=0)) == net_name(dict(W=64, ws=0, init=0)))

    # ---- (12) r2_sim
    import r2_sim
    ok, rows = r2_sim.selftest(quiet=True, return_rows=True)
    for rw in rows:
        if rw["passed"] is None:
            print(f"[SKIP] r2_sim: {rw['check']}", flush=True); continue
        ck("r2_sim: " + rw["check"], rw["passed"], rw.get("info", ""), **{k: v for k, v in rw.items() if k not in ("check", "passed", "info")})
    bad = [k for k, v in ck.res.items() if not v]
    refs = {k: dict(path=ref_path(k), sha256=hashlib.sha256(open(ref_path(k), "rb").read()).hexdigest() if ref_path(k) else None) for k in REF_FILES}
    plan = os.path.normpath(os.path.join(_JU, "..", "..", "results", "ju_space", "plans", "abschnitt11_12_runde2_plan.md"))
    out = dict(script="r2_toy_common.py selftest", device=str(DEV), quick=QUICK, torch=torch.__version__, numpy=np.__version__,
               plan_sha256=hashlib.sha256(open(plan, "rb").read()).hexdigest() if os.path.exists(plan) else None,
               module_sha256={n: hashlib.sha256(open(p_, "rb").read()).hexdigest() for n, p_ in (("r2_toy_common.py", __file__), ("r2_sim.py", r2_sim.__file__),
                                                                                                    ("pc_meta.py", M.__file__), ("pc_common.py", PCC.__file__))},
               references=refs, passed=len(ck.res) - len(bad), total=len(ck.res), failed=bad, checks=ck.rows, seconds=time.time() - ck.t0)
    path = os.path.join(work, "ref_comparison_w5.json"); json.dump(_jsonable(out), open(path + ".tmp", "w"), indent=1); os.replace(path + ".tmp", path)
    print(f"r2_toy_common selftest: {len(ck.res) - len(bad)}/{len(ck.res)} passed ({time.time() - ck.t0:.0f}s); comparison output {path}" + (f"; FAILED: {bad}" if bad else ""), flush=True)
    if bad:
        raise SystemExit(1)
    return ck.res


def _check_pc_autograd(nets, eta, T=5):
    """Closed-form PC relaxation and column gradient with erasers / soft targets against autograd of the energy (C-T7 / AT5)."""
    g = torch.Generator().manual_seed(3); W = nets.W; s = new_task_sets(0, 0, nshot=8); x, y, lab = (torch.as_tensor(a, device=DEV) for a in s["shot"])
    v = torch.randn(W, generator=g).to(DEV) * 0.3; Q = torch.linalg.qr(torch.randn(W, 3, generator=g, dtype=F64))[0].float().to(DEV); mu = torch.randn(W, generator=g).to(DEV) * 0.1
    soft = torch.softmax(torch.randn(8, P, generator=g), -1).to(DEV)
    W2, b2, W3, b3 = nets.p["W2"][0], nets.p["b2"][0], nets.p["W3"][0], nets.p["b3"][0]; Ex, Ey = nets.x_cols()[0], nets.y_cols()[0]
    worst = 0.0
    for case in ("h1", "h2", "soft"):
        E1 = (lambda h: h - ((h - mu) @ Q) @ Q.T) if case == "h1" else (lambda h: h)
        E2 = (lambda h: h - ((h - mu) @ Q) @ Q.T) if case == "h2" else (lambda h: h)
        Y = soft if case == "soft" else Fn.one_hot(lab, P).float()
        vv = v.clone().requires_grad_(True)
        z1 = Ex[:, x].T + Ey[:, y].T + nets.p["b1"][0] + vv
        p1 = E1(torch.tanh(z1)); h1 = p1.detach().clone(); h2 = E2(torch.tanh(h1 @ W2.T + b2)).detach().clone()

        def energy(h1_, h2_, p1_):
            e1 = h1_ - p1_; e2 = h2_ - E2(torch.tanh(h1_ @ W2.T + b2)); lg = (h2_ @ W3.T + b3) * nets.mult
            return 0.5 * (e1 ** 2).sum(-1) + 0.5 * (e2 ** 2).sum(-1) + BETA_PC * (-(Y * torch.log_softmax(lg, -1)).sum(-1))
        for _ in range(T):
            a_, b_ = h1.clone().requires_grad_(True), h2.clone().requires_grad_(True)
            g1, g2 = torch.autograd.grad(energy(a_, b_, p1.detach()).sum(), (a_, b_))
            h1, h2 = (h1 - eta * g1).detach(), (h2 - eta * g2).detach()
        gv = torch.autograd.grad(energy(h1, h2, p1).mean(), vv)[0]
        ops = Ops(h1_Q=Q[None], h1_mu=mu[None]) if case == "h1" else (Ops(h2_Q=Q[None], h2_mu=mu[None]) if case == "h2" else None)
        og = None if ops is None else ops.regroup(torch.zeros(1, 1, dtype=torch.long, device=DEV))
        Ycf = Y.view(1, 1, 8, P); z = z1.detach().view(1, 1, 8, W); et = torch.full((1, 1), float(eta), device=DEV)
        hh1, hh2, _ = _pc_col_grad(nets, torch.zeros(1, dtype=torch.long, device=DEV), z, Ycf, et, T, BETA_PC, og, return_states=True)
        gcf, _ = _pc_col_grad(nets, torch.zeros(1, dtype=torch.long, device=DEV), z, Ycf, et, T, BETA_PC, og)
        worst = max(worst, _rel(hh1[0, 0], h1), _rel(hh2[0, 0], h2), _rel(gcf[0, 0], gv))
    return worst <= 1e-4, f"max rel {worst:.1e}"


def _check_worlds(ck, steps=30):
    """Training in the other worlds (W3 circle, W5 imposed codes, bottleneck, W4 two families) against the reference scripts' nets and
    optimiser groups, 30 updates."""
    ns = ref_defs("c3_angle", ["Net", "four", "acc"], {"W": 64})
    if ns:
        ws = 1; pre, new, pairs, splits = M.world(ws); fx_ = ns["four"](1); torch.manual_seed(ws); net = ns["Net"](fx_, fx_)
        opt = torch.optim.AdamW(net.groups(M.LR), lr=M.LR, betas=M.BETAS); sp = lambda c: splits[c % len(splits)]
        xtr = torch.as_tensor(np.concatenate([pairs[sp(c)[:144]] for c in pre])); ctr = torch.as_tensor(np.repeat(pre, 144)); ytr = (xtr.sum(1) + ctr) % P
        for _ in range(steps):
            opt.zero_grad(); Fn.cross_entropy(net.out(xtr, ctr), ytr).backward(); opt.step()
        r = train([dict(kind="circle", W=64, ws=ws)], fixed_steps=steps, coord_check=False).nets
        e = max(_rel(r.p["W1"][0], torch.cat([net.Ax, net.Ay, net.Ac], 1)), _rel(r.p["W2"][0], net.l2.weight), _rel(r.p["W3"][0], net.l3.weight))
        ck("W3 circle training = c3_angle (30 updates)", e <= 1e-4, f"rel {e:.1e}")
        unt = w3_shifts(ws)[1]; zs = zero_shot(r, [ws], [unt])[0]; zr = {c: ns["acc"](net, torch.as_tensor(pairs[sp(c)[8:]]), c) for c in unt}
        dz = max(abs(zs[c]["acc"] - zr[c]) for c in unt)
        ck("W3 zero-shot accuracy on the untrained shifts = c3_angle's acc", dz <= 1 / 281 + 1e-9, mine=[zs[c]["acc"] for c in unt], ref=[zr[c] for c in unt], max_abs_diff=dz)
    ns = ref_defs("impose2", ["Net", "code_table", "acc"])
    if ns:
        ws = 2; pre, new, pairs, splits = M.world(ws); torch.manual_seed(ws); net = ns["Net"](64, ns["code_table"]("ring", 2, ws))
        opt = torch.optim.AdamW(net.groups(M.LR), lr=M.LR, betas=M.BETAS)
        xtr = torch.as_tensor(np.concatenate([pairs[splits[j][:144]] for j in range(NPRE)])); ctr = torch.as_tensor(np.repeat(pre, 144)); ytr = (xtr.sum(1) + ctr) % P
        for _ in range(steps):
            opt.zero_grad(); Fn.cross_entropy(net.out(xtr, net.task(ctr)), ytr).backward(); opt.step()
        r = train([dict(kind="impose", W=64, ws=ws, code="ring", ck=2)], fixed_steps=steps, coord_check=False).nets
        e = max(_rel(r.p["W1"][0], torch.cat([net.Ex, net.A, net.t0[:, None]], 1)), _rel(r.p["W2"][0], net.l2.weight), _rel(r.p["W3"][0], net.l3.weight))
        ck("W5 imposed-code training = impose2 (30 updates)", e <= 1e-4, f"rel {e:.1e}")
        zs = zero_shot(r, [ws], [list(new)], mode="impose")[0]
        zr = {c: ns["acc"](net, torch.as_tensor(pairs[splits[NPRE + j][8:]]), c, net.task(torch.tensor(c))) for j, c in enumerate(new)}
        dz = max(abs(zs[c]["acc"] - zr[c]) for c in new)
        ck("W5 zero-shot accuracy t0 + A phi(c_new) = impose2's acc", dz <= 1 / 281 + 1e-9, mine=[zs[c]["acc"] for c in new], ref=[zr[c] for c in new], max_abs_diff=dz)
    ns = ref_defs("rgi_bneck", ["MLPB"])
    if ns:
        ws = 0; X, y, _, _, lab, _ = _r1_batch(ws); torch.manual_seed(ws); net = ns["MLPB"](64, 4).to(DEV); opt = M.make_opt(net)
        for _ in range(steps):
            M.bp_update(net, opt, X, y)
        r = train([dict(kind="bneck", W=64, ws=ws, dt=4)], fixed_steps=steps, coord_check=False).nets
        e = max(_rel(r.p["W1"][0], net.l1.weight[:, :2 * P]), _rel(r.p["U"][0], net.U), _rel(r.p["E"][0], net.E), _rel(r.p["W2"][0], net.l2.weight))
        ck("bottleneck training = rgi_bneck.MLPB + pc_meta (30 updates)", e <= 1e-4, f"rel {e:.1e}")
    ws = 3; pre, new, pairs, splits = M.world(ws); old, _ = w4_tasks(ws); DIN = 2 * P + 2 * NPRE + 2 * NNEW
    spl = lambda slot: splits[slot % len(splits)] if slot < len(splits) else np.random.default_rng(3000 + 31 * ws + slot).permutation(P * P)

    def enc(xy, slot):
        X_ = torch.zeros(len(xy), DIN); r_ = torch.arange(len(xy)); X_[r_, torch.as_tensor(xy[:, 0])] = 1; X_[r_, P + torch.as_tensor(xy[:, 1])] = 1; X_[r_, 2 * P + slot] = 1; return X_
    labf = lambda fam, xy, c: (xy[:, 0] + (xy[:, 1] if fam == "add" else -xy[:, 1]) + c) % P
    Xtr = torch.cat([enc(pairs[spl(s)[:144]], s) for f, c, s in old]); ytr = torch.cat([torch.as_tensor(labf(f, pairs[spl(s)[:144]], c)) for f, c, s in old])
    torch.manual_seed(ws); m = PCC.MLP(DIN, 64, P, param="mup", base_width=BASE); opt = M.make_opt(m)
    for _ in range(steps):
        M.bp_update(m, opt, Xtr, ytr)
    r = train([dict(W=64, ws=ws, data="w4")], fixed_steps=steps, coord_check=False).nets
    ck("W4 two-family training = c5_twofam's data and pc_meta (30 updates)", _state_rel(r, 0, m.state_dict()) <= 1e-4, f"rel {_state_rel(r, 0, m.state_dict()):.1e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print(__doc__)
