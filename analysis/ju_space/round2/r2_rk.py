"""Round 2 (plan v2, sections 2.3 and 9.1): the Qwen machinery shared by every round-2 Qwen script.

The round-1 code (rk_loop.py, rk_verdict.py, ufo_common.py, rsi_statements.py) is imported unchanged; rk_loop.py runs top-level code
and parses sys.argv when it is imported, so a block is opened only through open_block (one Python process = one block, because
rk_loop registers its hooks on block BLOCK at import).

API (every function documented below; 'rk' = the rk_loop module returned by open_block)
  Round-1 files (OD-G9, OD-G2, OD-G3)
    find_r1(r1_dir, rel, block=None, required=True) -> path       locate <r1_dir>/gpu*/<rel> by name; duplicates by OD-G9; logged
    FINDER_LOG                                                     every choice made by find_r1 (write it into the output JSON)
    r1_blocks(r1_dir) -> dict(b_lo, b_hi, B3, flags, ...)          B3 = {14, b_lo, b_hi} (the blocks that ran, OD-G2)
    delta_order(r1_dir, info=None) -> dict(order=[b(1), b(2), b(3)], delta, flags)
    load_r1_state(r1_dir, b), load_r1_json(r1_dir, rel, block=None, required=True), r1_meta(r1_dir, b)
  Opening a block
    open_block(root, r1_dir, r2_dir, b) -> rk                      copies R1.state[b] to <r2_dir>/rk/b<b>/state.pt (if missing), sets
                                                                   sys.argv / BLOCK / ASCALE / POOLDIRS / PERSONA and imports rk_loop
    BLK                                                            dict(b, alpha (= alpha_b), alpha_scale, suffix, cap, med, paths)
  State guard (critique 22d)
    guard_check(), save(), st_put(key, value)                      round-1 keys never modified or deleted; new keys 'r2_*' or
                                                                   '{ev,dir,dmg,k6}_r2*'; rk.save is replaced by a guarded save
  Reading
    read_all_blocks(seqs_prompt, conts, batch=32, blocks=None, token_blocks=(), tok_dtype=torch.float16)
    read_windows(seqs_prompt, conts, first_div=None, batch=32, n_first=20)
    first_divergence(a, b)
  Axes and indicators (no model needed)
    unit, cosf, perp_basis(vectors), project_out(V, Q)
    axis_from_texts(Xp, Xa, fold, seed=None, basis=None, n_null=2000)
    axis_rel(diff, keep=None, Q=None, extra=None)
    stats_on_axis(ev, axis, null, n_prompts=None)
    halvings(n_prompts)                                             = rk.HALV for the same prompt count
    prompt_bootstrap(n_prompts, B=2000, seed=0, n_rsi=None) -> W [B, NP]; pair_weights, boot_mean, interval
  Pushes, temperature, damage, ablation, gradients
    CbarHook(cap=None) (context manager), cbar_read(seqs_prompt, conts, cap=None, batch=32)
    set_temperature(T), temperature(T) (context manager)
    damage_T(d, alpha, T, n=8)
    WriteOrth(U) (context manager)
    leaf_grads(seqs, positions, blocks, scalars_fn, batch=16)
    evaluate_variant(d, alpha, keyfn, ...)
  O3 loop-direction bank, Pool directions
    loops_of_block(b), o3_bank(ST, loops, b, D=None), load_o3(r1_dir, b), pool_dir(kind, b, i, D=None), k6_null_dirs(b=14, D=None)
  Files
    write_json(path, obj), torch_save(path, obj) (atomic: tmp + os.replace), jsonable(obj)
  Code test
    python r2_rk.py selftest <root> <workdir>        (MOCK; builds MOCK round-1 states in <workdir>/r1 when missing, see mock_r1)

Conventions: keys via rk.keyseed with the namespaces of Appendix D; texts stored as token ids; float32 model, TF32 off on CUDA
(ufo_common.load_model); MOCK=1 = rk_loop's tiny random model (a code test only)."""
import contextlib, glob, hashlib, json, math, os, re, shutil, subprocess, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); JU = os.path.dirname(HERE)            # analysis/ju_space
if JU not in sys.path: sys.path.insert(0, JU)
import rk_verdict                                                                       # functions only, no side effects

MOCK = os.environ.get("MOCK") == "1"
WINDOWS = ("all", "div", "first20")
LOOPS_14 = ("RSI_a", "RSI_b", "SEA", "SPO", "RND_1", "RND_2"); LOOPS_EXTRA = ("RSI_a", "SEA", "RND_1")      # L(14), L(b_lo) = L(b_hi)
NEW_KEY = re.compile(r"^(r2_|(ev|dir|dmg|k6)_r2)")                                      # names allowed for new state keys
SHARD_OF = {"b14": "gpu1", "b_lo": "gpu2", "b_hi": "gpu3"}                              # the round-1 runner's assignment


# ---------------------------------------------------------------- small utilities
def unit(v):
    """= ufo_common.unit (v / (|v| + 1e-9) along the last dimension)."""
    return v / (v.norm(dim=-1, keepdim=True) + 1e-9)


def cosf(a, b):
    """= rk_loop.cos: float(a . b / (|a| |b| + 1e-12))."""
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def jsonable(x):
    """Converts tensors, numpy values and tuples to JSON-able Python objects (NaN stays NaN, as round 1 writes it)."""
    if torch.is_tensor(x): return jsonable(x.detach().cpu().tolist())
    if isinstance(x, np.ndarray): return jsonable(x.tolist())
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    return x


def write_json(path, obj):
    """Atomic JSON write (tmp + os.replace)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True); tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(jsonable(obj), f, indent=1)
    os.replace(tmp, path)


def torch_save(path, obj):
    """Atomic torch.save (tmp + os.replace)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True); torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)


def log(msg):
    b = BLK.get("b"); print(f"[r2_rk{'' if b is None else f' b{b}'} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- round-1 files (OD-G9, OD-G2, OD-G3)
FINDER_LOG = []


class R1Missing(FileNotFoundError):
    """A round-1 file a measurement needs is not in <r1_dir> (G6: that measurement is 'not run (round-1 output missing)')."""


def r1_blocks(r1_dir):
    """B3 = {14, b_lo, b_hi} (OD-G2). b_lo / b_hi = the blocks that ran: the single rk/b<b> directory of gpu2 / gpu3 (block 14
    excluded), cross-checked with that shard's blocks_chosen.json and with loops.json '_meta'.block; blocks_chosen.json decides only
    when no (or more than one) block directory exists. Flags: 'block choice inconsistent between shards' (G7) and any cross-check
    failure. Returns dict(b14=14, b_lo, b_hi, B3, flags, chosen, ran, source)."""
    out = dict(b14=14, flags=[], chosen={}, ran={}, source={})
    for s in ("gpu2", "gpu3"):
        p = os.path.join(r1_dir, s, "blocks_chosen.json")
        if os.path.exists(p):
            c = json.load(open(p)); out["chosen"][s] = dict(b_lo=int(c["b_lo"]), b_hi=int(c["b_hi"]))
        d = os.path.join(r1_dir, s, "rk")
        out["ran"][s] = sorted(int(x[1:]) for x in os.listdir(d) if re.fullmatch(r"b\d+", x)) if os.path.isdir(d) else []
    ch = out["chosen"]
    if len(ch) == 2 and ch["gpu2"] != ch["gpu3"]: out["flags"].append("block choice inconsistent between shards")

    def pick(shard, role):
        ran = [b for b in out["ran"][shard] if b != 14]; other = "gpu3" if shard == "gpu2" else "gpu2"
        if len(ran) == 1: b, src = ran[0], f"{shard}/rk/b{ran[0]} (the block that ran)"
        elif shard in ch: b, src = ch[shard][role], f"{shard}/blocks_chosen.json"
        elif other in ch: b, src = ch[other][role], f"{other}/blocks_chosen.json"
        else: b, src = None, "missing"
        if len(ran) > 1: out["flags"].append(f"{shard}/rk holds several blocks {ran}; {role} from {src}")
        if b is not None and shard in ch and ch[shard][role] != b:
            out["flags"].append(f"{role}: {shard}/blocks_chosen.json says {ch[shard][role]}, the block that ran is {b}")
        lp = os.path.join(r1_dir, shard, "rk", f"b{b}", "loops.json")
        if b is not None and os.path.exists(lp):
            mb = json.load(open(lp)).get("_meta", {}).get("block")
            if mb is not None and int(mb) != b: out["flags"].append(f"{role}: loops.json '_meta'.block {mb} differs from directory b{b}")
        out["source"][role] = src; return b
    out["b_lo"], out["b_hi"] = pick("gpu2", "b_lo"), pick("gpu3", "b_hi")
    out["B3"] = [b for b in (14, out["b_lo"], out["b_hi"]) if b is not None]
    return out


def _preferred_shard(r1_dir, block):
    """OD-G9 for rk files: the shard of the runner's assignment (gpu1 block 14, gpu2 b_lo, gpu3 b_hi)."""
    if block is None: return None
    if int(block) == 14: return SHARD_OF["b14"]
    info = r1_blocks(r1_dir)
    return SHARD_OF["b_lo"] if int(block) == info["b_lo"] else (SHARD_OF["b_hi"] if int(block) == info["b_hi"] else None)


def find_r1(r1_dir, rel, block=None, required=True):
    """Locates a round-1 file by name across the six shards: candidates = glob('<r1_dir>/gpu*/<rel>') (OD-G9; never a fixed shard).
    Several candidates: the shard of the runner's assignment for 'block' (gpu1 for 14, gpu2 for b_lo, gpu3 for b_hi), otherwise
    the most recent file. Every choice is appended to FINDER_LOG. Missing: R1Missing (required) or None."""
    cands = sorted(glob.glob(os.path.join(r1_dir, "gpu*", rel)))
    rec = dict(rel=rel, block=block, candidates=[os.path.relpath(c, r1_dir) for c in cands])
    if not cands:
        rec.update(chosen=None, reason="missing"); FINDER_LOG.append(rec)
        if required: raise R1Missing(f"round-1 file {rel} not found under {r1_dir}/gpu*")
        return None
    if len(cands) == 1: chosen, why = cands[0], "unique"
    else:
        pref = _preferred_shard(r1_dir, block); hit = [c for c in cands if os.path.relpath(c, r1_dir).split(os.sep)[0] == pref]
        if hit: chosen, why = hit[0], f"duplicate; shard of the runner's assignment ({pref})"
        else: chosen, why = max(cands, key=os.path.getmtime), "duplicate; most recent file"
    rec.update(chosen=os.path.relpath(chosen, r1_dir), reason=why); FINDER_LOG.append(rec); return chosen


def load_r1_json(r1_dir, rel, block=None, required=True):
    p = find_r1(r1_dir, rel, block, required); return None if p is None else json.load(open(p))


def load_r1_state(r1_dir, b):
    """R1.state[b] (read-only use; never saved back)."""
    return torch.load(find_r1(r1_dir, f"rk/b{b}/state.pt", b), map_location="cpu", weights_only=False)


def r1_meta(r1_dir, b):
    """'_meta' of R1.loops[b]: alpha (= alpha_b), alpha_scale, suffix, block (G10, used as is)."""
    return load_r1_json(r1_dir, f"rk/b{b}/loops.json", b)["_meta"]


def delta_order(r1_dir, info=None):
    """b(1) < b(2) < b(3): the blocks of B3 ordered by R1.delta (gpu2/dn_delta.json, last-token reading; gpu3's copy if gpu2's is
    missing, flagged). OD-G3: if b_lo and b_hi agree within 1e-4 the round-1 label decides (b_lo below b_hi); block 14 is placed by
    its delta (exact ties by block index)."""
    info = info or r1_blocks(r1_dir); flags = []
    p = os.path.join(r1_dir, "gpu2", "dn_delta.json")
    if not os.path.exists(p):
        p = os.path.join(r1_dir, "gpu3", "dn_delta.json"); flags.append("gpu2/dn_delta.json missing; gpu3's copy used")
        if not os.path.exists(p): raise R1Missing("dn_delta.json missing in gpu2 and gpu3")
    D = json.load(open(p))["blocks"]; d = {b: float(D[str(b)]["last"]["delta"]) for b in info["B3"]}
    order = sorted(d, key=lambda b: (d[b], b)); lo, hi = info.get("b_lo"), info.get("b_hi")
    if lo in order and hi in order and abs(d[lo] - d[hi]) <= 1e-4:
        i, j = order.index(lo), order.index(hi)
        if i > j: order[i], order[j] = order[j], order[i]
        flags.append(f"delta tie within 1e-4 between b_lo {lo} and b_hi {hi}: round-1 label decides")
    ties = [(a, b) for k, a in enumerate(order) for b in order[k + 1:] if abs(d[a] - d[b]) <= 1e-4]
    return dict(order=order, delta=d, ties=ties, flags=flags, source=os.path.relpath(p, r1_dir))


# ---------------------------------------------------------------- opening a block
BLK = {}
_RK = None
_GUARD = {}


def _need():
    if _RK is None: raise RuntimeError("no block open: call r2_rk.open_block(root, r1_dir, r2_dir, b) first")
    return _RK


def pooldirs_path(root=None):
    """The Pool-direction package (rk_dirs.pt): env POOLDIRS, else <root>/gpu_pkg/rk_dirs.pt, else the repository copy."""
    if os.environ.get("POOLDIRS"): return os.environ["POOLDIRS"]
    if root and os.path.exists(f"{root}/gpu_pkg/rk_dirs.pt"): return f"{root}/gpu_pkg/rk_dirs.pt"
    return os.path.join(JU, "gpu", "data", "rk_dirs.pt")


def open_block(root, r1_dir, r2_dir, b):
    """Opens block b for this process and returns the rk_loop module (plan 2.3).
    - copies R1.state[b] (found by find_r1) to <r2_dir>/rk/b<b>/state.pt if that file is missing (an existing copy is kept: resume);
    - reads R1.loops[b]['_meta'] (G10: alpha, alpha_scale, suffix used as is);
    - sets sys.argv = [rk_loop.py, root, <r2_dir>/rk, 'lib'] and env BLOCK = b, ASCALE = repr(alpha_scale_b), POOLDIRS and PERSONA
      (POOLDIRS / PERSONA keep values already in the environment, else <root>/gpu_pkg/rk_dirs.pt and <root>/rsi_dl/persona), imports
      rk_loop (which loads the model, the copied state as rk.ST and registers its hooks on block b) and restores sys.argv;
    - checks rk.BL == b, rk.SUF == suffix_b, rk.ST['alpha']['alpha'] * rk.ASCALE == alpha_b and the hidden size of the state;
    - installs the state guard: the round-1 keys (= the keys of the round-1 file) are fingerprinted (key set, tensor dtypes / shapes /
      float64 sums, other leaves by value) and rk.save is replaced by a save that asserts the fingerprint and the naming of new keys.
    Afterwards rk.model, rk.tok, rk.PROMPTS, rk.PIDS, rk.INSTR, rk.keyseed, rk.Gumbel, rk.generate, rk.read, rk.reply_read, rk.damage,
    rk.push_damage, rk.evaluate, rk.stats, rk.run_loop, rk.persona_sets, rk.grads, rk.z_beh, rk.logodds_matching, rk.k6_direction,
    rk.STEER, rk.CAP, rk.HALV are available; BLK holds b, alpha (alpha_b), alpha_scale, suffix, cap, med and the paths.
    One process = one block: a second call with another block raises."""
    global _RK
    b = int(b)
    if _RK is not None:
        if BLK["b"] != b: raise RuntimeError(f"one Python process = one block: block {BLK['b']} is open, block {b} asked")
        if (BLK["r1_dir"], BLK["r2_dir"]) != (os.path.abspath(r1_dir), os.path.abspath(r2_dir)):
            raise RuntimeError(f"block {b} is open with r1_dir {BLK['r1_dir']} and r2_dir {BLK['r2_dir']}, asked for {r1_dir}, {r2_dir}")
        return _RK
    if "rk_loop" in sys.modules: raise RuntimeError("rk_loop was imported outside r2_rk.open_block")
    src = find_r1(r1_dir, f"rk/b{b}/state.pt", b); meta = r1_meta(r1_dir, b)
    if int(meta.get("block", b)) != b: raise RuntimeError(f"loops.json of block {b} says block {meta.get('block')}")
    dst_dir = os.path.join(r2_dir, "rk", f"b{b}"); os.makedirs(dst_dir, exist_ok=True); dst = os.path.join(dst_dir, "state.pt")
    if not os.path.exists(dst): shutil.copyfile(src, dst + ".tmp"); os.replace(dst + ".tmp", dst)
    os.environ["BLOCK"] = str(b); os.environ["ASCALE"] = repr(float(meta["alpha_scale"]))
    os.environ.setdefault("POOLDIRS", pooldirs_path(root)); os.environ.setdefault("PERSONA", f"{root}/rsi_dl/persona")
    argv0 = list(sys.argv); sys.argv = [os.path.join(JU, "rk_loop.py"), root, os.path.join(r2_dir, "rk"), "lib"]
    try:
        import rk_loop as rk
    finally:
        sys.argv = argv0
    alpha_b = float(meta["alpha"])
    if rk.BL != b: raise RuntimeError(f"rk_loop opened block {rk.BL}, expected {b}")
    if rk.SUF != meta.get("suffix", ""): raise RuntimeError(f"suffix {rk.SUF!r} differs from loops.json {meta.get('suffix')!r}")
    if abs(rk.ST["alpha"]["alpha"] * rk.ASCALE - alpha_b) > 1e-9 * max(1.0, abs(alpha_b)): raise RuntimeError("alpha_b differs from loops.json")
    if int(rk.ST["axes"]["u_T_perp"].shape[0]) != rk.D: raise RuntimeError(f"state hidden size {rk.ST['axes']['u_T_perp'].shape[0]} != model {rk.D} (MOCK state with the real model?)")
    _RK = rk
    BLK.update(b=b, alpha=alpha_b, alpha_scale=float(meta["alpha_scale"]), suffix=rk.SUF, cap=float(rk.ST["ref"]["cap"]),
               med=float(rk.ST["ref"]["med"]), r1_state=os.path.relpath(src, r1_dir), state=dst, mock=bool(rk.MOCK),
               r1_dir=os.path.abspath(r1_dir), r2_dir=os.path.abspath(r2_dir),
               pooldirs=os.environ["POOLDIRS"], persona=os.environ["PERSONA"], device=str(rk.DEV))
    _install_guard(rk, src, dst_dir)
    log(f"block {b} open: alpha_b {alpha_b:.4g} (scale {BLK['alpha_scale']:g}, suffix {rk.SUF!r}), cap {BLK['cap']:.4g}, device {rk.DEV}")
    return rk


# ---------------------------------------------------------------- state guard (critique 22d)
class StateGuardError(AssertionError):
    """A round-1 key of the copied state was modified or deleted, or a new key does not follow the round-2 naming."""


def _fp(x):
    """Fingerprint of one state entry: key set, tensor dtype / shape / float64 sum, numpy the same, int lists by sha256, other
    leaves by value."""
    if torch.is_tensor(x):
        t = x.detach(); return ["T", str(t.dtype), list(t.shape), repr(float(t.double().sum())) if t.numel() else "0"]
    if isinstance(x, np.ndarray): return ["N", str(x.dtype), list(x.shape), repr(float(np.asarray(x, dtype=np.float64).sum()))]
    if isinstance(x, dict): return ["D", [[repr(k), _fp(v)] for k, v in sorted(x.items(), key=lambda kv: repr(kv[0]))]]
    if isinstance(x, (list, tuple)):
        if x and all(type(v) is int for v in x): return ["I", len(x), hashlib.sha256(np.asarray(x, dtype=np.int64).tobytes()).hexdigest()]
        return ["L", type(x).__name__, [_fp(v) for v in x]]
    if isinstance(x, (np.generic,)): return ["V", type(x).__name__, repr(x.item())]
    return ["V", type(x).__name__, repr(x)]


def _digest(x): return hashlib.sha256(json.dumps(_fp(x)).encode()).hexdigest()


def _install_guard(rk, src, dst_dir):
    R1 = torch.load(src, map_location="cpu", weights_only=False)
    fp = {k: _digest(v) for k, v in R1.items()}; del R1
    _GUARD.update(keys=frozenset(fp), fp=fp, orig_save=rk.save)
    write_json(os.path.join(dst_dir, "r1_fingerprint.json"), dict(source=src, keys=sorted(fp), digest=fp))
    guard_check()                                                                    # the copy must hold the round-1 keys unchanged
    rk.save = save                                                                   # every rk save (evaluate, run_loop, ...) is guarded


def new_key_ok(k):
    """Round-2 keys: loop names start with 'r2' (rk's own keys become ev_r2*, dir_r2*, dmg_r2*, k6_r2*); every other key 'r2_'."""
    return bool(NEW_KEY.match(str(k)))


def guard_check():
    """Asserts that every round-1 key is present and unchanged and every other key is a round-2 key."""
    ST = _need().ST
    miss = sorted(_GUARD["keys"] - set(ST))
    if miss: raise StateGuardError(f"round-1 keys deleted from the state: {miss[:5]}")
    for k in _GUARD["keys"]:
        if _digest(ST[k]) != _GUARD["fp"][k]: raise StateGuardError(f"round-1 key {k!r} was modified")
    bad = sorted(str(k) for k in ST if k not in _GUARD["keys"] and not new_key_ok(k))
    if bad: raise StateGuardError(f"new state keys must start with 'r2_' (or be rk keys of an 'r2' loop): {bad[:5]}")


def save():
    """Guarded save of rk.ST (rk_loop's own atomic save after guard_check)."""
    guard_check(); _GUARD["orig_save"]()


def st_put(key, value, do_save=True):
    """Stores a new round-2 entry in rk.ST (key must start with 'r2_' or be an rk key of an 'r2' loop) and saves (guarded)."""
    if not new_key_ok(key) or key in _GUARD["keys"]: raise StateGuardError(f"{key!r} is not a round-2 key")
    _need().ST[key] = value
    if do_save: save()


# ---------------------------------------------------------------- reading
def _assert_unsteered():
    rk = _need()
    if rk.STEER["v"] is not None or rk.STEER["V"] is not None: raise RuntimeError("read with rk's steering active")


def read_all_blocks(seqs_prompt, conts, batch=32, blocks=None, token_blocks=(), tok_dtype=torch.float16):
    """One unstopped forward per batch (no early stop) with forward hooks on the outputs of the given blocks (default all 28):
    X[i, j] = mean over the continuation tokens of text i (prompt seqs_prompt[i] + conts[i], left-padded batches of 'batch' like
    rk.read) of the output of block blocks[j], float64 (NaN for an empty continuation); at block rk.BL this equals rk.read.
    token_blocks: blocks whose per-token continuation states are also returned (tok[b][i] = [len(conts[i]), D] in tok_dtype).
    Returns dict(X [N, len(blocks), D] float64, blocks, tok)."""
    rk = _need(); _assert_unsteered(); NL = len(rk.LAY)
    blocks = list(range(NL)) if blocks is None else [int(b) for b in blocks]; token_blocks = [int(b) for b in token_blocks]
    assert set(token_blocks) <= set(blocks), "token_blocks must be among blocks"
    N = len(conts); X = torch.full((N, len(blocks), rk.D), float("nan"), dtype=torch.float64); TOK = {b: [None] * N for b in token_blocks}
    cur = {}

    def mk(j, l):
        def h(mod, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            cur["sum"][:, j] = torch.einsum("btd,bt->bd", x.double(), cur["M"]).cpu()
            if l in TOK: cur["tok"][l] = x.detach().to(tok_dtype).cpu()
        return h
    handles = [rk.LAY[l].register_forward_hook(mk(j, l)) for j, l in enumerate(blocks)]
    try:
        for i in range(0, N, batch):
            pb, cb = seqs_prompt[i:i + batch], conts[i:i + batch]; ids, am, pad = rk.pad_batch([p + c for p, c in zip(pb, cb)])
            T = ids.shape[1]; lens = torch.tensor([len(c) for c in cb])
            cur.update(M=(torch.arange(T)[None, :] >= (T - lens)[:, None]).double().to(rk.DEV), sum=torch.zeros(len(cb), len(blocks), rk.D, dtype=torch.float64), tok={})
            with torch.no_grad(): rk.model.model(input_ids=ids.to(rk.DEV), attention_mask=am.to(rk.DEV))
            m = cur["sum"] / lens.clamp(min=1).double()[:, None, None]; m[lens == 0] = float("nan"); X[i:i + len(cb)] = m
            for l in TOK:
                for r, c in enumerate(cb): TOK[l][i + r] = cur["tok"][l][r, T - len(c):].clone()
    finally:
        for h in handles: h.remove()
    return dict(X=X, blocks=blocks, tok=TOK)


def first_divergence(a, b):
    """rk.evaluate's rule: the first index where the token lists differ (min length if one is a prefix of the other), -1 if equal."""
    if a == b: return -1
    return next((i for i, (u, v) in enumerate(zip(a, b)) if u != v), min(len(a), len(b)))


def read_windows(seqs_prompt, conts, first_div=None, batch=32, n_first=20):
    """Block-b (rk.BL) reading of pairs in three windows (B-Q1): 'all' = mean over all continuation tokens (exactly rk.read: same
    early-stopped forward, batching and float64 mean); 'div' = mean over the tokens at positions >= the pair's first divergence;
    'first20' = mean over the first n_first continuation tokens. Rows: the '+' text of pair j at 2j, the '-' text at 2j + 1 (rk.evaluate's
    order); seqs_prompt[i] is the read prompt of row i. first_div per pair (rk.evaluate's convention, -1 = identical pair; computed if None).
    An identical pair has delta_j = 0 in the 'div' window (critique 22e, OD-B15); its two 'div' states are set to its 'all' states. A text
    whose window is empty (empty text; in 'div' a text that ends at the divergence) has NaN there and its pair is not ok in that window.
    Returns dict(first_div, X={w: [2N, D]}, xp={w: [N, D]}, xm, delta={w: [N, D]}, ok={w: [N] bool}) (float64)."""
    rk = _need(); _assert_unsteered(); N2 = len(conts); assert N2 % 2 == 0 and batch % 2 == 0; N = N2 // 2
    if first_div is None: first_div = [first_divergence(conts[2 * j], conts[2 * j + 1]) for j in range(N)]
    X = {w: torch.full((N2, rk.D), float("nan"), dtype=torch.float64) for w in WINDOWS}
    for i in range(0, N2, batch):
        pb, cb = seqs_prompt[i:i + batch], conts[i:i + batch]; ids, am, pad = rk.pad_batch([p + c for p, c in zip(pb, cb)]); rk.CAP["on"] = True
        try:
            with torch.no_grad(): rk.model.model(input_ids=ids.to(rk.DEV), attention_mask=am.to(rk.DEV))
        except rk.Stop: pass
        finally: rk.CAP["on"] = False
        hs = rk.CAP.pop("h").double().cpu(); T = ids.shape[1]
        for r, c in enumerate(cb):
            k = i + r; L = len(c); fd = first_div[k // 2]
            if not L: continue
            seg = hs[r, T - L:]; X["all"][k] = seg.mean(0); X["first20"][k] = seg[:n_first].mean(0)
            if fd < 0: X["div"][k] = X["all"][k]
            elif L > fd: X["div"][k] = seg[fd:].mean(0)
    out = dict(first_div=list(first_div), X=X, xp={}, xm={}, delta={}, ok={})
    ident = torch.tensor([fd < 0 for fd in first_div], dtype=torch.bool)
    for w in WINDOWS:
        xp, xm = X[w][0::2], X[w][1::2]; d = (xp - xm) / 2
        if w == "div": d[ident] = 0.0
        out["xp"][w], out["xm"][w], out["delta"][w] = xp, xm, d; out["ok"][w] = ~(torch.isnan(xp).any(1) | torch.isnan(xm).any(1))
    return out


# ---------------------------------------------------------------- axes (rk_loop.axes() logic) and indicators (rk_loop.stats logic)
def perp_basis(vectors):
    """Orthonormal basis [D, k] (float64) of span(unit(v) for v in vectors), by QR as in rk_loop.axes()."""
    F = torch.stack([unit(torch.as_tensor(v).double()) for v in vectors]).T; Q, _ = torch.linalg.qr(F); return Q


def project_out(V, Q):
    """V minus its least-squares (orthogonal) projection on span(Q); V [D] or [n, D]."""
    V = V.double(); return V - (V @ Q) @ Q.T


def halvings(n_prompts):
    """The 20 K2 halvings (rk_loop.halvings / rk.HALV for n_prompts prompts: 16 + 16 per prompt type at 64 prompts, seed 3)."""
    if _RK is not None and len(_RK.HALV[0]) == n_prompts: return _RK.HALV
    return rk_verdict.halvings(n_prompts)


def axis_from_texts(Xp, Xa, fold, seed=None, basis=None, n_null=2000):
    """rk_loop.axes() for one axis. Xp, Xa [P, D]: per-prompt mean states of the pro and anti texts (NaN row = empty text, the prompt
    drops out as in round 1); fold [P] bool = the round-1 prompt fold. Per-prompt differences diff = Xp - Xa (kept prompts), u = mean;
    fold axes dA, dB; cross-fitted centred AUC, paired win share and gap of each fold's texts on the other fold's axis (centred on the
    training fold's midpoint); n_null within-prompt sign-flip null axes (E +-1 from torch.randint with torch.Generator().manual_seed(seed),
    null = E @ diff / P). basis (list of vectors, or an orthonormal [D, k] matrix): u_perp = u minus its least-squares projection on the
    span, the nulls projected the same way, gap_perp (in-sample) and gap_perp_cross (rk's Delta_T estimator: each fold's differences on
    the other fold's projected axis). Returns a dict (float64 tensors)."""
    Xp, Xa = torch.as_tensor(Xp).double(), torch.as_tensor(Xa).double(); fold = torch.as_tensor(fold, dtype=torch.bool)
    keep = ~(torch.isnan(Xp).any(1) | torch.isnan(Xa).any(1)); P0 = len(keep); diff_all = Xp - Xa
    Xp, Xa, fold = Xp[keep], Xa[keep], fold[keep]; diff = Xp - Xa; u = diff.mean(0); dA, dB = diff[fold].mean(0), diff[~fold].mean(0)
    uA, uB = unit(dA), unit(dB)
    mA = ((Xp[fold].mean(0) + Xa[fold].mean(0)) / 2) @ uA; mB = ((Xp[~fold].mean(0) + Xa[~fold].mean(0)) / 2) @ uB
    sp = torch.cat([Xp[~fold] @ uA - mA, Xp[fold] @ uB - mB]); sa = torch.cat([Xa[~fold] @ uA - mA, Xa[fold] @ uB - mB])
    out = dict(u=u, dA=dA, dB=dB, diff=diff, diff_all=diff_all, keep=keep, fold=fold, kept=int(keep.sum()), n_prompts=P0,
               auc=float((sp[:, None] > sa[None, :]).double().mean()), paired_win=float((sp > sa).double().mean()), gap_cross=float((sp - sa).mean()))
    if n_null:
        g = torch.Generator().manual_seed(int(seed)); E = (torch.randint(0, 2, (n_null, len(diff)), generator=g) * 2 - 1).double()
        out["null"] = (E @ diff) / len(diff); out["null_seed"] = int(seed)
    if basis is not None:
        Q = basis.double() if (torch.is_tensor(basis) and basis.dim() == 2) else perp_basis(basis); out["Q"] = Q
        out["u_perp"] = project_out(u, Q)
        if n_null: out["null"] = project_out(out["null"], Q)
        out["gap_perp"] = float((diff @ unit(out["u_perp"])).mean())
        out["gap_perp_cross"] = float(torch.cat([diff[~fold] @ unit(project_out(dA, Q)), diff[fold] @ unit(project_out(dB, Q))]).mean())
    return out


def _rel(r):
    """rk.stats's reliability: max(2r / (1 + r), 0.05) (floor 0.05, flagged), NaN for non-finite r."""
    if not math.isfinite(r): return float("nan"), False
    if r <= -1: return 0.05, True
    v = 2 * r / (1 + r); return max(v, 0.05), v < 0.05


def axis_rel(diff, keep=None, Q=None, extra=None):
    """Split-half reliability of a mean-difference axis over the 20 K2 halvings: r = mean over halvings of cos(axis of half A, axis
    of half B), each half's axis = mean of its kept prompts' differences, projected like the axis (Q: orthonormal basis projected out;
    extra: a vector additionally projected out, e.g. u_T_perp for u_RGI_perpT); rel = 2r / (1 + r) with rk.stats's floor 0.05.
    diff [P, D] over all P prompts of the halvings (keep [P] marks the rows that count; other rows are ignored)."""
    diff = torch.as_tensor(diff).double(); P = len(diff)
    keep = torch.ones(P, dtype=torch.bool) if keep is None else torch.as_tensor(keep, dtype=torch.bool)
    rs = []

    def ax(v):
        if Q is not None: v = project_out(v, Q)
        if extra is not None: e = unit(torch.as_tensor(extra).double()); v = v - (v @ e) * e
        return v
    for m in halvings(P):
        mm = torch.as_tensor(m) & keep
        if mm.sum() == 0 or (~mm & keep).sum() == 0: continue
        rs.append(cosf(ax(diff[mm].mean(0)), ax(diff[~mm & keep].mean(0))))
    r = float(np.mean(rs)) if rs else float("nan"); rel, fl = _rel(r)
    return dict(rel=rel, r=r, rel_floored=fl, n_halvings=len(rs))


def stats_on_axis(ev, axis, null, n_prompts=None):
    """rk_loop.stats on any axis with its own null axes: d = mean of delta over ok pairs; c = cos(d, axis); p = (1 + #{null k:
    |cos(null k, d)| >= |c|}) / (1 + #nulls); Q = |c| / 99th percentile of the null |cos|; rel from the K2 halvings exactly as rk.stats
    (mean split-half cos r, rel = max(2r / (1 + r), 0.05)); A = c / sqrt(rel); degenerate cases as rk.stats. ev: dict(delta [N, D],
    ok [N], prompt_of [N]) (an rk evaluation or evaluate_variant output). Works without an open block (n_prompts from prompt_of if
    not given). Returns (dict(c, p, Q, rel, r, rel_floored, A, norm), d)."""
    delta, ok, po = ev["delta"].double(), torch.as_tensor(ev["ok"], dtype=torch.bool), torch.as_tensor(ev["prompt_of"])
    d = delta[ok].mean(0); ax = torch.as_tensor(axis).double(); null = torch.as_tensor(null).double()
    c = cosf(d, ax); nc = (null @ d).abs() / (null.norm(dim=1) * d.norm() + 1e-12)
    p = float((1 + (nc >= abs(c)).sum()) / (1 + len(nc))); q99 = float(torch.quantile(nc, 0.99)); Q = abs(c) / q99 if q99 > 0 else float("nan")
    if float(d.norm()) == 0: p, Q = 1.0, float("nan")
    if not math.isfinite(c): p, Q = float("nan"), float("nan")
    NP = n_prompts if n_prompts is not None else (_RK.NP if _RK is not None else int(po.max()) + 1); rs = []
    for m in halvings(NP):
        mA = torch.tensor(m)[po] & ok; mB = (~torch.tensor(m))[po] & ok; rs.append(cosf(delta[mA].mean(0), delta[mB].mean(0)))
    r = float(np.mean(rs))
    if not math.isfinite(r): rel = float("nan")
    else: rel = max(2 * r / (1 + r), 0.05) if r > -1 else 0.05
    out = dict(c=c, p=p, Q=Q, rel=rel, r=r, rel_floored=bool(math.isfinite(r) and (r <= -1 or 2 * r / (1 + r) < 0.05)),
               A=c / math.sqrt(rel) if rel == rel else float("nan"), norm=float(d.norm()))
    return out, d


# ---------------------------------------------------------------- prompt bootstrap (2.6)
def prompt_bootstrap(n_prompts=None, B=2000, seed=0, n_rsi=None):
    """W [B, NP] (float64 counts): replicate b draws NRSI prompts with replacement among the RSI-topic prompts 0..NRSI-1 and NP - NRSI
    among the scene prompts (stratified by prompt type; rk_verdict.abar_bootstrap's draw, np.random.default_rng(seed)). Prompts are
    the units: every pair / text j gets the weight W[b, prompt_of[j]], so both seeds of a prompt go together."""
    NP = n_prompts if n_prompts is not None else _need().NP; NRSI = n_rsi if n_rsi is not None else NP // 2
    g = np.random.default_rng(seed); W = np.zeros((B, NP))
    for b in range(B):
        pick = np.concatenate([g.integers(0, NRSI, NRSI), NRSI + g.integers(0, NP - NRSI, NP - NRSI)]); W[b] = np.bincount(pick, minlength=NP)
    return W


def pair_weights(W, prompt_of, ok=None):
    """[B, N] weights of N pairs / texts: W[:, prompt_of] times ok."""
    w = np.asarray(W)[:, np.asarray(prompt_of)]
    return w * np.asarray(ok, dtype=float)[None] if ok is not None else w


def boot_mean(values, prompt_of, W, ok=None):
    """Bootstrap replicates of the mean of values [N, ...] (pairs or texts) over ok rows: [B, ...] (NaN where a replicate has no row)."""
    w = torch.as_tensor(pair_weights(W, prompt_of, ok)); v = torch.as_tensor(values).double()
    if ok is not None: v = torch.where(torch.as_tensor(ok, dtype=torch.bool).reshape(-1, *([1] * (v.dim() - 1))), v, torch.zeros_like(v))
    s = w.sum(1); out = torch.tensordot(w, v, dims=([1], [0])) / s.reshape(-1, *([1] * (v.dim() - 1)))
    out[s == 0] = float("nan"); return out


def interval(draws, level=0.90):
    """Percentile interval of bootstrap draws (NaN draws ignored): (lo, hi) at (1 - level) / 2 and (1 + level) / 2."""
    x = np.asarray(draws, dtype=float); x = x[np.isfinite(x)]
    if not len(x): return float("nan"), float("nan")
    a = 100 * (1 - level) / 2; return float(np.percentile(x, a)), float(np.percentile(x, 100 - a))


# ---------------------------------------------------------------- push magnitude (cbar), temperature, damage at T
class CbarHook:
    """The cbar hook (2.3, 4.0): a forward hook registered with prepend=True on block b, so it runs before rk's steer hook and sees the
    block output before the push is added; it logs min(|h_t|, cap_b) (the factor of rk's push) at continuation positions.
    Generation mode (default; use it around rk.generate / evaluate_variant only): every prefill (T > 1) opens a record of one generate
    call, every decode step (T == 1) logs one value per row; a decode step feeds back the previously sampled token, so the values of a
    row are its continuation tokens 0 .. min(L, steps) - 1 (the last of 40 tokens is never fed back; rows that stopped are cut by their
    continuation length L). Read mode (cbar_read): values at the continuation positions of a teacher-forced read (all L tokens).
    per_text(conts) -> list of 1-D float64 tensors (one per generated row, in generation order); cbar(conts) -> dict(per_text_mean
    [N], pooled, n_tokens)."""
    def __init__(self, cap=None, block=None):
        rk = _need(); self.cap = float(rk.ST["ref"]["cap"] if cap is None else cap); self.block = rk.BL if block is None else int(block)
        self.rec = []; self.mode = "gen"; self.read_mask = None; self.h = None

    def _hook(self, mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out; v = x.detach().norm(dim=-1).clamp(max=self.cap)
        if self.mode == "read": self.rec.append([v[r][self.read_mask[r]].double().cpu() for r in range(v.shape[0])]); return None
        if x.shape[1] > 1: self.rec.append(dict(B=x.shape[0], steps=[]))
        elif not self.rec: raise RuntimeError("cbar hook: a decode step without a prefill (generation mode is for rk.generate only)")
        else: self.rec[-1]["steps"].append(v[:, 0])
        return None

    def __enter__(self):
        self.h = _need().LAY[self.block].register_forward_hook(self._hook, prepend=True); return self

    def __exit__(self, *a):
        if self.h is not None: self.h.remove(); self.h = None

    def per_text(self, conts):
        assert self.mode == "gen"; out = []; i = 0
        for rec in self.rec:
            S = torch.stack(rec["steps"]).double().cpu() if rec["steps"] else torch.zeros(0, rec["B"], dtype=torch.float64)
            for r in range(rec["B"]): out.append(S[: min(len(conts[i]), S.shape[0]), r]); i += 1
        assert i == len(conts), f"cbar hook saw {i} rows, {len(conts)} continuations given"
        return out

    @staticmethod
    def summary(vals):
        n = sum(len(v) for v in vals); m = torch.tensor([float(v.mean()) if len(v) else float("nan") for v in vals], dtype=torch.float64)
        return dict(per_text_mean=m, pooled=float(sum(float(v.sum()) for v in vals) / n) if n else float("nan"), n_tokens=int(n))

    def cbar(self, conts): return self.summary(self.per_text(conts))


def cbar_read(seqs_prompt, conts, cap=None, batch=32):
    """The cbar hook in read mode: one early-stopped forward per batch of prompt + continuation (unsteered), min(|h_t|, cap_b) at every
    continuation position. Returns CbarHook.summary (per-text means, pooled mean, token count); kappa = pooled / med_b."""
    rk = _need(); _assert_unsteered(); vals = []
    with CbarHook(cap) as hk:
        hk.mode = "read"
        for i in range(0, len(conts), batch):
            pb, cb = seqs_prompt[i:i + batch], conts[i:i + batch]; ids, am, pad = rk.pad_batch([p + c for p, c in zip(pb, cb)]); T = ids.shape[1]
            hk.read_mask = [torch.arange(T, device=rk.DEV) >= T - len(c) for c in cb]; hk.rec = []; rk.CAP["on"] = True
            try:
                with torch.no_grad(): rk.model.model(input_ids=ids.to(rk.DEV), attention_mask=am.to(rk.DEV))
            except rk.Stop: pass
            finally: rk.CAP["on"] = False; rk.CAP.pop("h", None)
            vals += hk.rec[0]
    return CbarHook.summary(vals)


def set_temperature(T):
    """Sets the sampling temperature (rk.TEMP; rk.Gumbel reads the module global at call time). Returns the previous value."""
    rk = _need(); old = rk.TEMP; rk.TEMP = float(T); return old


@contextlib.contextmanager
def temperature(T):
    """Context manager: rk.TEMP = T inside (None: unchanged), restored afterwards."""
    rk = _need(); old = rk.TEMP
    if T is not None: rk.TEMP = float(T)
    try: yield
    finally: rk.TEMP = old


def damage_T(d, alpha, T, n=8):
    """rk.damage with logits / T: mean over the continuation positions of the first n reference texts (teacher-forced) of
    KL(softmax(l0 / T) || softmax(l1 / T)), l1 under the push +alpha * min(|h_t|, cap_b) * unit(d) from the first user token on.
    T = 1 gives rk.damage exactly."""
    rk = _need(); kl = []; R = rk.ST["ref"]
    for p in range(n):
        seq = rk.PIDS[p] + R["conts"][p]; ids = torch.tensor([seq]).to(rk.DEV); P0 = len(rk.PIDS[p]); pos = torch.arange(P0 - 1, len(seq) - 1).to(rk.DEV)
        if len(pos) == 0: continue
        with torch.no_grad():
            a = torch.log_softmax(rk.model(ids).logits[0][pos].double() / T, -1)
            rk.STEER.update(v=unit(d.float()), V=None, alpha=alpha, cap=R["cap"], sign=torch.tensor([1.0]), rowstart=torch.tensor([rk.START]))
            try: b = torch.log_softmax(rk.model(ids).logits[0][pos].double() / T, -1)
            finally: rk.steer_off()
        kl.append(float((a.exp() * (a - b)).sum(-1).mean()))
    return float(np.mean(kl))


# ---------------------------------------------------------------- exact write-orthogonalisation (C-Q3)
class WriteOrth:
    """Context manager: forward hooks that remove span(U) from the outputs of embed_tokens and of self_attn.o_proj and mlp.down_proj
    in all blocks (57 hooks for 28 blocks), so that nothing written into the residual stream has a component in span(U); the tied
    unembedding (weights) is untouched. U: [D] or [k, D] (orthonormalised here by QR)."""
    def __init__(self, U):
        U = torch.as_tensor(U).double(); U = U[None] if U.dim() == 1 else U; Q, _ = torch.linalg.qr(U.T); self.Q64 = Q; self.handles = []

    def _hook(self, mod, inp, out):
        Q = self.Qd; return out - (out @ Q) @ Q.T

    def __enter__(self):
        rk = _need(); self.Qd = self.Q64.to(rk.DEV, next(rk.model.parameters()).dtype)
        mods = [rk.model.model.embed_tokens] + [l.self_attn.o_proj for l in rk.LAY] + [l.mlp.down_proj for l in rk.LAY]
        self.handles = [m.register_forward_hook(self._hook) for m in mods]; return self

    def __exit__(self, *a):
        for h in self.handles: h.remove()
        self.handles = []


# ---------------------------------------------------------------- gradients by zero leaf vectors (A3)
def leaf_grads(seqs, positions, blocks, scalars_fn, batch=16):
    """Zero leaf vectors added to the outputs of the given blocks at given positions; gradients of given scalars.
    seqs: list of token-id lists (left-padded in batches, attention-masked); positions[i]: positions in seqs[i] (own coordinates) where
    the leaf of each block is added (one leaf vector [D] per text and block, added at every listed position: 'plain addition');
    scalars_fn(logits [B, T, V]) -> [B, S] (per-row scalars; with left padding the last position is T - 1 for every row).
    Returns G [S, N, len(blocks), D] float64: G[s, i, j] = d scalar_s(text i) / d leaf(text i, blocks[j])."""
    rk = _need(); _assert_unsteered(); blocks = [int(b) for b in blocks]; out = []
    for i in range(0, len(seqs), batch):
        sb, pb = seqs[i:i + batch], positions[i:i + batch]; ids, am, pad = rk.pad_batch(sb); B, T = ids.shape; M = torch.zeros(B, T)
        for r, ps in enumerate(pb):
            for q in ps:
                if not 0 <= int(q) < len(sb[r]): raise IndexError(f"position {q} outside text {i + r} of length {len(sb[r])}")
                M[r, int(pad[r]) + int(q)] = 1.0
        V = torch.zeros(B, len(blocks), rk.D, device=rk.DEV, requires_grad=True); Md = M.to(rk.DEV)

        def mk(j):
            def h(mod, inp, o):
                x = o[0] if isinstance(o, tuple) else o; x = x + Md[..., None].to(x.dtype) * V[:, j, None, :].to(x.dtype)
                return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
            return h
        hs = [rk.LAY[l].register_forward_hook(mk(j)) for j, l in enumerate(blocks)]
        try:
            with torch.enable_grad():
                Y = scalars_fn(rk.model(input_ids=ids.to(rk.DEV), attention_mask=am.to(rk.DEV)).logits); S = Y.shape[1]; G = []
                for s in range(S): G.append(torch.autograd.grad(Y[:, s].sum(), V, retain_graph=s < S - 1)[0].double().cpu())
        finally:
            for h in hs: h.remove()
        out.append(torch.stack(G))
    return torch.cat(out, 1)


# ---------------------------------------------------------------- generalised evaluation (rk.evaluate with more options)
def _entropy_gumbel(rk, store):
    class EntGumbel(rk.Gumbel):
        """rk.Gumbel that also logs the entropy of softmax(logits / T) per row at every sampling step."""
        def __init__(self, keys): super().__init__(keys); store.append([])

        def __call__(self, input_ids, scores):
            lp = torch.log_softmax(scores.float() / rk.TEMP, -1); store[-1].append(-(lp.exp() * lp).sum(-1).detach())
            return super().__call__(input_ids, scores)
    return EntGumbel


def evaluate_variant(d, alpha, keyfn, nseed=1, prompts=None, T=None, signs=(1.0, -1.0), gen_ids=None, read_ids=None, hooks=(),
                     cbar=True, entropy=False, batch=32, n_first=20, cache_key=None):
    """rk.evaluate with an explicit key function, temperature, read windows, extra hooks and a sign per row.
    Pairs j = (p, s), p in prompts (default all rk.NP prompts), s in range(nseed), in rk.evaluate's order (prompt-major); row 2j and row
    2j + 1 share the key keyfn(p, s) (e.g. lambda p, s: rk.keyseed('theta', b, p, s)); row signs = signs (tuple for every pair, or a
    callable (p, s) -> (sign_2j, sign_2j+1)); push = sign * alpha * min(|h_t|, cap_b) * unit(d) from the first user token on (rk's steer
    hook; d None = no push). gen_ids: None (rk.PIDS[p] for both rows) or a callable (p, s) -> ids or (ids_2j, ids_2j+1) (e.g. instructed
    prompts); read_ids: None (rk.PIDS[p]: read as the reply to the plain prompt by the unsteered model) or a callable p -> ids. hooks:
    context managers active during the generation only (e.g. WriteOrth(U), an extra push); the reading is by the plain model. T: sampling
    temperature of this call (rk.TEMP restored afterwards; None = rk.TEMP as it is). cbar: the cbar hook during generation; entropy:
    per-text mean over its sampling steps (one per continuation token) of the entropy of softmax(logits / T).
    Returns a dict stored like rk.evaluate: delta, ok, xp, xm (all-token window, float32), prompt_of, seed_of, first_div, conts (token
    ids), alpha, T, keys, signs, capped_share, dropped, windows {all, div, first20: delta, xp, xm, ok}, cbar (per_text_mean, pooled,
    n_tokens), entropy, n_pairs. cache_key (a round-2 key): cached in rk.ST and saved (guarded); a cached result is returned if alpha and T
    agree."""
    rk = _need(); Tv = float(rk.TEMP if T is None else T)
    if cache_key is not None:
        if not new_key_ok(cache_key): raise StateGuardError(f"{cache_key!r} is not a round-2 key")
        if cache_key in rk.ST:
            ev = rk.ST[cache_key]
            assert abs(ev["alpha"] - alpha) < 1e-9 * max(1.0, abs(alpha)) and abs(ev["T"] - Tv) < 1e-12, f"{cache_key} cached at other alpha / T"
            return ev
    prompts = list(range(rk.NP)) if prompts is None else [int(p) for p in prompts]; rows, prompt_of, seed_of, keys, sg = [], [], [], [], []
    for p in prompts:
        for s in range(nseed):
            k = int(keyfn(p, s)); sp, sm = signs(p, s) if callable(signs) else signs
            g = rk.PIDS[p] if gen_ids is None else gen_ids(p, s)
            gp, gm = (g if (isinstance(g, tuple) and len(g) == 2 and not isinstance(g[0], int)) else (g, g))   # 2-tuple of id lists = one per row
            rows += [(list(gp), k, float(sp)), (list(gm), k, float(sm))]; prompt_of.append(p); seed_of.append(s); keys.append(k); sg.append((float(sp), float(sm)))
    ent_store = []; Gum = rk.Gumbel
    with contextlib.ExitStack() as es:
        es.enter_context(temperature(Tv))
        for h in hooks: es.enter_context(h)
        hk = es.enter_context(CbarHook()) if cbar else None
        if entropy: rk.Gumbel = _entropy_gumbel(rk, ent_store)
        try:
            rk.capped_reset(); conts = rk.generate(rows, d, alpha, batch=batch); cs = rk.capped_share() if d is not None else float("nan")
        finally:
            rk.Gumbel = Gum
    rp = [list(rk.PIDS[p] if read_ids is None else read_ids(p)) for p in prompt_of for _ in (0, 1)]
    Wn = read_windows(rp, conts, batch=batch, n_first=n_first); ok = Wn["ok"]["all"]
    ev = dict(delta=Wn["delta"]["all"].float(), ok=ok, xp=Wn["xp"]["all"].float(), xm=Wn["xm"]["all"].float(), prompt_of=torch.tensor(prompt_of),
              seed_of=torch.tensor(seed_of), first_div=Wn["first_div"], conts=conts, alpha=float(alpha), T=Tv, keys=keys, signs=sg,
              capped_share=cs, dropped=int((~ok).sum()), n_pairs=len(prompt_of), n_first=n_first,
              windows={w: dict(delta=Wn["delta"][w].float(), xp=Wn["xp"][w].float(), xm=Wn["xm"][w].float(), ok=Wn["ok"][w]) for w in WINDOWS})
    if hk is not None: ev["cbar"] = hk.cbar(conts)
    if entropy:
        vals, i = [], 0
        for rec in ent_store:
            S = torch.stack(rec).double().cpu()
            for r in range(S.shape[1]): vals.append(S[: min(len(conts[i]), S.shape[0]), r]); i += 1
        ev["entropy"] = CbarHook.summary(vals)
    if cache_key is not None: st_put(cache_key, ev)
    return ev


# ---------------------------------------------------------------- Pool directions (O6) and the O3 loop-direction bank
_PD = {}


def pool_dir(kind, b, i, D=None, path=None):
    """Pool direction (kind 'rand' | 'cov', block b, index i) of rk_dirs.pt (unit, float64). MOCK (D = 64): ('rand', b, 0 / 1) are
    rk_loop's MOCK RND starts (unit(randn(D)) with torch.Generator seeds 0 / 1); every other (kind, b, i) a seeded random unit vector
    (seed = keyseed('mockpool', kind, b, i) mod 2^31)."""
    if MOCK:
        D = D or _need().D; seed = int(i) if (kind == "rand" and int(i) in (0, 1)) else int(hashlib.sha256(f"mockpool|{kind}|{b}|{i}".encode()).hexdigest()[:12], 16) % (2 ** 31)
        return unit(torch.randn(D, generator=torch.Generator().manual_seed(seed)).double())
    path = path or os.environ.get("POOLDIRS") or pooldirs_path()
    if path not in _PD: _PD[path] = torch.load(path, map_location="cpu", weights_only=True)
    return _PD[path][(kind, int(b), int(i))].double()


def k6_null_dirs(b=14, D=None, path=None):
    """The K6 null directions in rk.k6's order: ('rand', b, i) = null i and ('cov', b, i) = null 8 + i, i = 0..7 (O6). MOCK: rk.k6's
    four MOCK nulls (unit(randn(D)), seeds 100..103)."""
    if MOCK:
        D = D or _need().D; return [unit(torch.randn(D, generator=torch.Generator().manual_seed(100 + i)).double()) for i in range(4)]
    return [pool_dir(k, b, i, path=path) for k in ("rand", "cov") for i in range(8)]


def loops_of_block(b):
    """L(14) = {RSI_a, RSI_b, SEA, SPO, RND_1, RND_2}; L(b_lo) = L(b_hi) = {RSI_a, SEA, RND_1} (2.1)."""
    return LOOPS_14 if int(b) == 14 else LOOPS_EXTRA


def o3_bank(ST, loops, b, D=None, path=None):
    """O3 loop-direction bank of block b from R1.state[b] (ST) and R1.loops[b] (loops, with '_meta'): for L in L(b):
    start d_0 (u_T_perp for RSI_a / RSI_b, sea, sports, Pool ('rand', b, 0) / ('rand', b, 1) for RND_1 / RND_2, as rk_loop.loop_specs
    passed them to run_loop); d[n] = ST['dir_<L><SUF>_<n>'] (n = 1..4); ev[n] = the pairs behind d_n = ST['ev_<L><SUF>_<n-1>'] (delta,
    ok, prompt_of, xp, xm, first_div, damage_of_push, alpha); rel[n] = loops[L][n-1]['rel'] (critique 7); row[n] = loops[L][n-1] (c, p,
    A of d_n on u_T_perp, ...). Missing loops are listed in bank['_missing']."""
    meta = loops["_meta"]; suf = meta.get("suffix", ""); ax = ST["axes"]; D = D or int(ax["u_T_perp"].shape[0])
    starts = dict(RSI_a=ax["u_T_perp"], RSI_b=ax["u_T_perp"], SEA=ax["sea"], SPO=ax["sports"])
    bank = dict(_meta=dict(meta), _missing=[])
    for L in loops_of_block(b):
        if L not in loops or len(loops[L]) < 4 or any(f"dir_{L}{suf}_{n}" not in ST or f"ev_{L}{suf}_{n - 1}" not in ST for n in range(1, 5)):
            bank["_missing"].append(L); continue
        st = starts[L].double() if L in starts else pool_dir("rand", b, 0 if L == "RND_1" else 1, D=D, path=path)
        e = dict(name=L, suffix=suf, start=st, d={}, ev={}, rel={}, row={})
        for n in range(1, 5):
            ev = ST[f"ev_{L}{suf}_{n - 1}"]; e["d"][n] = ST[f"dir_{L}{suf}_{n}"].double()
            e["ev"][n] = {k: ev[k] for k in ("delta", "ok", "prompt_of", "xp", "xm", "first_div", "damage_of_push", "alpha") if k in ev}
            e["rel"][n] = float(loops[L][n - 1]["rel"]); e["row"][n] = loops[L][n - 1]
        bank[L] = e
    return bank


def load_o3(r1_dir, b):
    """o3_bank from the round-1 files of block b (found by find_r1); works without a model (r2_ana)."""
    loops = load_r1_json(r1_dir, f"rk/b{b}/loops.json", b); return o3_bank(load_r1_state(r1_dir, b), loops, b)


# ---------------------------------------------------------------- MOCK round-1 fixture for code tests
FAKE_BLOCKS = dict(b_lo=8, b_hi=20)


def fake_delta():
    """A fake dn_delta.json for code tests (blocks 0-27, 'last' / 'mean'): delta lowest at block 14 (0.09), b_lo = 8 (0.10) and b_hi
    = 20 (0.20), so that rk_verdict.choose_blocks gives (8, 20) and b(1) < b(2) < b(3) = (14, 8, 20)."""
    bl = {}
    for l in range(28):
        d = {14: 0.09, 8: 0.10, 20: 0.20}.get(l, 0.12 + 0.004 * abs(l - 14))
        bl[str(l)] = {r: dict(F=0.5 + f * d, floor=0.5, delta=f * d, delta_rel=(0.5 + f * d) / 0.5, probe=0.5, chance=0.125) for r, f in (("last", 1.0), ("mean", 0.9))}
    return dict(model="MOCK (fake dn_delta.json for code tests)", blocks=bl)


def mock_r1(root, r1_dir, spec=((14, "gpu1", ("all",)), (8, "gpu2", ("pilot", "loops", "k5")), (20, "gpu3", ("pilot", "loops", "k5"))),
            summaries=None, nt="1"):
    """Builds a MOCK round-1 layout in r1_dir for code tests (9: rk_loop.py MOCK=1 at the given blocks, a fake dn_delta.json and fake
    summaries): runs 'MOCK=1 BLOCK=<b> python rk_loop.py <root> <r1_dir>/<shard>/rk <stage>' for every stage whose block has no
    loops.json yet (subprocesses, DEVICE=cpu), then writes gpu2 / gpu3 dn_delta.json and blocks_chosen.json, summary_b<b>.json (block 14:
    a 'would have read' verdict with GO1 failed, PC_PASS true; b_lo: entering; b_hi: PC_PASS false; or 'summaries' {b: dict}), and
    shard_done.json. Test use only."""
    for b, shard, stages in spec:
        out = os.path.join(r1_dir, shard, "rk")
        if os.path.exists(os.path.join(out, f"b{b}", "loops.json")): continue
        os.makedirs(out, exist_ok=True); env = dict(os.environ, MOCK="1", BLOCK=str(b), DEVICE="cpu", NT=nt, OMP_NUM_THREADS=nt)
        env.pop("ASCALE", None)
        for st in stages: subprocess.run([sys.executable, os.path.join(JU, "rk_loop.py"), root, out, st], env=env, check=True)
    fd = fake_delta(); tmp = os.path.join(r1_dir, "_fake_delta.json"); write_json(tmp, fd)
    lo, hi, val = rk_verdict.choose_blocks(tmp); os.remove(tmp)
    assert (lo, hi) == (FAKE_BLOCKS["b_lo"], FAKE_BLOCKS["b_hi"]), (lo, hi)
    for s in ("gpu2", "gpu3"):
        write_json(os.path.join(r1_dir, s, "dn_delta.json"), fd)
        write_json(os.path.join(r1_dir, s, "blocks_chosen.json"), dict(b_lo=lo, b_hi=hi, delta={str(k): round(v, 4) for k, v in val.items()}))
    S = summaries or {
        14: dict(block=14, A_bar_RSI_a=dict(A_bar=0.3, lo=0.1, hi=0.5, draws=300), verdict=dict(
            verdict="not decidable (pilot gate GO1 failed; descriptive only; would have read: likely, substantial RSI component)",
            PC_PASS=True, label_attractor=False, gates=dict(failed=["GO1"], source="pilot.json", gates=dict(GO0=True, GO1=False, GO2=True, GO3=True)))),
        lo: dict(block=lo, verdict=dict(PC_PASS=True, gates=dict(failed=[], source="pilot.json"), enters_delta_rule=True)),
        hi: dict(block=hi, verdict=dict(PC_PASS=False, gates=dict(failed=[], source="pilot.json"), enters_delta_rule=False))}
    for b, shard, _ in spec:
        if b in S: write_json(os.path.join(r1_dir, shard, f"summary_b{b}.json"), S[b])
        write_json(os.path.join(r1_dir, shard, "shard_done.json"), dict(shard=shard, gpu="MOCK", torch=torch.__version__, code_commit="mock", code_commits_used=["mock"]))
    return dict(b_lo=lo, b_hi=hi)


# ---------------------------------------------------------------- code test (MOCK)
def selftest(root, work):
    """MOCK code test of every helper against rk_loop itself. Writes <work>/r2_rk_selftest.json; returns True if every check passed."""
    assert MOCK, "selftest needs MOCK=1"
    r1, r2 = os.path.join(work, "r1"), os.path.join(work, "r2_selftest_rk"); res = []

    def check(name, cond, info=""):
        res.append(dict(test=name, ok=bool(cond), info=str(info))); print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)
    if not (os.path.exists(os.path.join(r1, "gpu1", "rk", "b14", "loops.json")) and os.path.exists(os.path.join(r1, "gpu2", "dn_delta.json"))):
        mock_r1(root, r1, spec=((14, "gpu1", ("pilot", "loops")),))                 # existing MOCK states are kept

    # finder on a synthetic layout (empty placeholder files)
    fx = os.path.join(work, "r2_selftest_finder"); shutil.rmtree(fx, ignore_errors=True)
    for rel in ("gpu1/rk/b14/state.pt", "gpu4/rk/b14/state.pt", "gpu2/rk/b9/state.pt", "gpu3/rk/b19/state.pt", "gpu5/rk/b9/state.pt"):
        os.makedirs(os.path.dirname(os.path.join(fx, rel)), exist_ok=True); open(os.path.join(fx, rel), "w").close()
    write_json(os.path.join(fx, "gpu2", "blocks_chosen.json"), dict(b_lo=9, b_hi=19)); write_json(os.path.join(fx, "gpu3", "blocks_chosen.json"), dict(b_lo=9, b_hi=21))
    n0 = len(FINDER_LOG); p14 = find_r1(fx, "rk/b14/state.pt", 14); p9 = find_r1(fx, "rk/b9/state.pt", 9); info = r1_blocks(fx)
    check("finder: duplicates resolved by the runner's assignment", p14.endswith("gpu1/rk/b14/state.pt") and p9.endswith("gpu2/rk/b9/state.pt") and len(FINDER_LOG) == n0 + 2, FINDER_LOG[-1])
    check("finder: B3 = blocks that ran; inconsistent block choice flagged", info["B3"] == [14, 9, 19] and "block choice inconsistent between shards" in info["flags"], info["flags"])
    check("finder: missing file -> R1Missing / None", find_r1(fx, "rk/b14/k6.json", 14, required=False) is None)
    info = r1_blocks(r1); order = delta_order(r1, info)
    check("finder: MOCK B3 and delta order", info["B3"] == [14, 8, 20] and order["order"] == [14, 8, 20] and not info["flags"], (info["B3"], order["order"]))

    # open_block, guard
    rk = open_block(root, r1, r2, 14)
    check("open_block: state copied, alpha_b / suffix / cap", os.path.exists(os.path.join(r2, "rk", "b14", "state.pt")) and BLK["suffix"] == rk.SUF and abs(BLK["alpha"] - rk.ST["alpha"]["alpha"] * rk.ASCALE) < 1e-12, BLK)
    try: open_block(root, r1, r2, 8); ok = False
    except RuntimeError: ok = True
    check("open_block: one process = one block", ok)
    k0 = "ev_RSI_a_0"; old = rk.ST[k0]["delta"].clone(); rk.ST[k0]["delta"][0, 0] += 1.0
    try: rk.save(); ok = False
    except StateGuardError: ok = True
    rk.ST[k0]["delta"] = old; check("guard: modified round-1 key refused by rk.save", ok)
    rk.ST["junk"] = 1
    try: save(); ok = False
    except StateGuardError: ok = True
    del rk.ST["junk"]; check("guard: badly named new key refused", ok)
    v = rk.ST.pop("ref")
    try: save(); ok = False
    except StateGuardError: ok = True
    rk.ST["ref"] = v; st_put("r2_selftest", dict(x=1)); check("guard: deleted round-1 key refused; r2_ key accepted", ok and "r2_selftest" in torch.load(BLK["state"], map_location="cpu", weights_only=False))
    del rk.ST["r2_selftest"]; save()

    # reading
    R = rk.ST["ref"]; idx = [p for p in range(rk.NP) if len(R["conts"][p])]; seqs = [rk.PIDS[p] for p in idx]; conts = [R["conts"][p] for p in idx]
    X1 = rk.read(seqs, conts); A = read_all_blocks(seqs, conts, token_blocks=(rk.BL, 3))
    e1 = float((A["X"][:, rk.BL] - X1).abs().max() / X1.abs().max())
    tk = torch.stack([t.double().mean(0) for t in A["tok"][3]]); e2 = float((tk - A["X"][:, 3]).abs().max() / A["X"][:, 3].abs().max())
    check("read_all_blocks: block b equals rk.read; per-token states", e1 < 1e-6 and e2 < 5e-3 and A["X"].shape == (len(idx), 28, rk.D), f"{e1:.2e} {e2:.2e}")
    B = read_all_blocks(seqs[:3], conts[:3], blocks=[rk.BL]); check("read_all_blocks: block subset", float((B["X"][:, 0] - X1[:3]).abs().max()) < 1e-9)
    ev = rk.ST["ev_RSI_a_0"]; rp = [rk.PIDS[p] for p in ev["prompt_of"].tolist() for _ in (0, 1)]
    Wn = read_windows(rp, ev["conts"]); e3 = float((Wn["delta"]["all"] - ev["delta"].double()).abs().max() / ev["delta"].abs().max())
    check("read_windows: 'all' reproduces rk.evaluate's delta; first_div as rk", e3 < 1e-5 and Wn["first_div"] == ev["first_div"], f"{e3:.2e}")
    a = conts[0]; pr = [seqs[0], seqs[0], seqs[0], seqs[0]]; cc = [a, list(a), a, a[:2] + [a[2] + 1 if a[2] + 1 < 151000 else 5] + a[3:]] if len(a) > 3 else None
    if cc:
        W2 = read_windows(pr, cc); T1 = read_all_blocks([seqs[0]], [a], blocks=[rk.BL], token_blocks=[rk.BL])["tok"][rk.BL][0].double()
        ok = float(W2["delta"]["div"][0].abs().max()) == 0.0 and bool(W2["ok"]["div"][0]) and W2["first_div"] == [-1, 2]
        ok = ok and float((W2["xp"]["first20"][1] - T1[:20].mean(0)).abs().max()) < 5e-3 * float(T1.abs().max()) and float((W2["xp"]["div"][1] - T1[2:].mean(0)).abs().max()) < 5e-3 * float(T1.abs().max())
        check("read_windows: identical pair delta 0 in 'div'; 'div' / 'first20' windows", ok)

    # axes, stats
    ax = rk.ST["axes"]; Q = perp_basis([ax["sports"], ax["tone"], ax["sea"], ax["u_neg"]]); S = rk.ST["axis_rsi"]
    Ar = axis_from_texts(S["Xp"], S["Xa"], ax["fold"], seed=11, basis=Q)
    ok = abs(Ar["auc"] - ax["auc"]["rsi"]) < 1e-9 and abs(Ar["gap_cross"] - ax["gap_cross"]["rsi"]) < 1e-4 * max(1, abs(ax["gap_cross"]["rsi"]))
    e4 = float((Ar["u_perp"] - ax["u_T_perp"].double()).norm() / ax["u_T_perp"].norm()); e5 = float((Ar["null"].float() - ax["null_u_T_perp"]).abs().max() / ax["null_u_T_perp"].abs().max())
    check("axis_from_texts reproduces rk axes() (rsi: AUC, gap, u_T_perp, nulls, Delta_T)", ok and e4 < 1e-4 and e5 < 1e-4 and abs(Ar["gap_perp_cross"] - ax["gap_perp_cross"]) < 1e-4 * max(1, abs(ax["gap_perp_cross"])), f"{e4:.1e} {e5:.1e}")
    Sa = rk.ST["axis_sea"]; As = axis_from_texts(Sa["Xp"], Sa["Xa"], ax["fold"], seed=14)
    check("axis_from_texts: sea nulls (seed 14) as rk", float((As["null"].float() - ax["null_sea"]).abs().max()) < 1e-4 * float(ax["null_sea"].abs().max()))
    s1, d1 = rk.stats(ev); s2, d2 = stats_on_axis(ev, ax["u_T_perp"], ax["null_u_T_perp"])
    check("stats_on_axis equals rk.stats on u_T_perp", all(abs(s1[k] - s2[k]) < 1e-12 or (s1[k] != s1[k] and s2[k] != s2[k]) for k in ("c", "p", "Q", "rel", "A", "r")), {k: (s1[k], s2[k]) for k in ("c", "p", "rel")})
    rr = axis_rel(Ar["diff_all"], Ar["keep"], Q=Q); s3, _ = rk.stats(dict(delta=Ar["diff_all"].float(), ok=Ar["keep"], prompt_of=torch.arange(rk.NP)), "sea")
    r0 = axis_rel(Ar["diff_all"].float(), Ar["keep"])                                 # the same float32 rows rk.stats sees
    check("axis_rel equals rk.stats's split-half r / rel on the same rows (no projection)", abs(r0["r"] - s3["r"]) < 1e-12 and abs(r0["rel"] - s3["rel"]) < 1e-12 and math.isfinite(rr["rel"]), (r0, s3["r"], rr))

    # bootstrap
    Wb = prompt_bootstrap(rk.NP, B=50, seed=3); NR = rk.NP // 2
    check("prompt_bootstrap: stratified, reproducible", Wb.shape == (50, rk.NP) and (Wb[:, :NR].sum(1) == NR).all() and (Wb[:, NR:].sum(1) == rk.NP - NR).all() and (Wb == prompt_bootstrap(rk.NP, B=50, seed=3)).all())
    bm = boot_mean(ev["delta"], ev["prompt_of"], Wb, ev["ok"]); check("boot_mean shape", tuple(bm.shape) == (50, rk.D))

    # evaluate_variant reproduces a round-1 evaluation (common random numbers on the same device)
    u = ax["u_T_perp"].double(); al = rk.ST["alpha"]["alpha"] * rk.ASCALE
    E = evaluate_variant(u, al, lambda p, s: rk.keyseed("RSI_a" + rk.SUF, 0, p, s, rk.BL), nseed=2, T=0.7, entropy=True)
    check("evaluate_variant reproduces rk.evaluate (RSI_a n = 0)", E["conts"] == ev["conts"] and float((E["delta"] - ev["delta"]).abs().max()) < 1e-5 * float(ev["delta"].abs().max()) and E["first_div"] == ev["first_div"])
    bank = o3_bank(rk.ST, rk.ST["loops"], 14); e0 = rk.ST["ev_RND_1_0"]
    E2 = evaluate_variant(bank["RND_1"]["start"], e0["alpha"], lambda p, s: rk.keyseed("RND_1" + rk.SUF, 0, p, s, rk.BL), nseed=1, cbar=False)
    check("O3 bank: RND_1 start reproduces round-1 evaluation; d_n, rel", E2["conts"] == e0["conts"] and not bank["_missing"] and torch.equal(bank["RSI_a"]["d"][2], rk.ST["dir_RSI_a_2"].double()) and bank["SEA"]["rel"][3] == rk.ST["loops"]["SEA"][2]["rel"])
    Z = evaluate_variant(None, 0.0, lambda p, s: rk.keyseed("r2CHK", "zero", p), prompts=range(4), cbar=False)
    check("evaluate_variant: zero push with shared keys gives identical pairs", all(fd == -1 for fd in Z["first_div"]) and float(Z["windows"]["div"]["delta"].abs().max()) == 0.0)
    M1 = evaluate_variant(-u, al, lambda p, s: rk.keyseed("r2CHK", "mirror", p), prompts=range(4), cbar=False)
    M2 = evaluate_variant(u, al, lambda p, s: rk.keyseed("r2CHK", "mirror", p), prompts=range(4), signs=(-1.0, 1.0), cbar=False)
    check("evaluate_variant: sign per row (mirror: -u with +1 equals u with -1)", M1["conts"] == M2["conts"])
    pro, anti = rk.INSTR["rsi"]; ins = lambda p, s: (rk.enc(rk.chat(rk.PROMPTS[p] + " " + pro)), rk.enc(rk.chat(rk.PROMPTS[p] + " " + anti)))
    I = evaluate_variant(None, 0.0, lambda p, s: rk.keyseed("axis", "rsi", p), gen_ids=ins, cbar=False)
    rows = [r for p in range(rk.NP) for r in ((ins(p, 0)[0], rk.keyseed("axis", "rsi", p), 1), (ins(p, 0)[1], rk.keyseed("axis", "rsi", p), -1))]
    G0 = rk.generate(rows); Xp0 = rk.read(rk.PIDS, G0[0::2])
    check("evaluate_variant: one generation prompt per row (instructed pro / anti), read under the plain prompt", I["conts"] == G0 and float((I["xp"].double() - Xp0).abs().max()) < 1e-5 * float(Xp0.abs().max()))
    Uh = unit(ax["sea"].double()); H = evaluate_variant(None, 0.0, lambda p, s: rk.keyseed("r2CHK", "orth", p), prompts=range(4), hooks=[WriteOrth(Uh)], cbar=False)
    Hp = rk.read([rk.PIDS[p] for p in range(4)], H["conts"][0::2])
    check("evaluate_variant: hooks act on the generation only (the read is by the plain model)", float((H["xp"].double() - Hp).abs().max()) < 1e-5 * float(Hp.abs().max()))
    Th = evaluate_variant(None, 0.0, lambda p, s: rk.keyseed("r2CHK", "T", p), prompts=range(4), T=0.3, entropy=True, cbar=False)
    check("temperature restored; entropy at T 0.3 below T 0.7", rk.TEMP == 0.7 and Th["entropy"]["pooled"] < E["entropy"]["pooled"], (Th["entropy"]["pooled"], E["entropy"]["pooled"]))

    # cbar: per-row values against a teacher-forced read of the generated texts (pre-push norms of the unsteered model)
    cb = E["cbar"]; vals = []
    rp = [rk.PIDS[p] for p in E["prompt_of"].tolist() for _ in (0, 1)]; Rt = read_all_blocks(rp, E["conts"], blocks=[rk.BL], token_blocks=[rk.BL])["tok"][rk.BL]
    for i, c in enumerate(E["conts"]):
        h = Rt[i].double().norm(dim=-1).clamp(max=BLK["cap"]); vals.append(h[: min(len(c), rk.NNEW - 1)])
    ref = CbarHook.summary(vals); e6 = float(np.nanmax(np.abs(cb["per_text_mean"].numpy() - ref["per_text_mean"].numpy())) / BLK["cap"])
    check("cbar hook (generation) equals the pre-push norms of the texts", e6 < 5e-3 and cb["n_tokens"] == ref["n_tokens"], f"{e6:.1e} n {cb['n_tokens']}")
    kr = cbar_read([rk.PIDS[p] for p in idx], conts); e7 = abs(kr["pooled"] - CbarHook.summary([t.double().norm(dim=-1).clamp(max=BLK["cap"]) for t in A["tok"][rk.BL]])["pooled"])
    check("cbar_read equals the clamped norms of a read", e7 < 1e-3 * BLK["cap"], f"{e7:.1e}")

    # damage_T, WriteOrth, leaf_grads
    d1 = rk.damage(u, al); d2 = damage_T(u, al, 1.0); d3 = damage_T(u, al, 0.5)
    check("damage_T(T = 1) equals rk.damage; T dependence", abs(d1 - d2) < 1e-12 and d3 != d2 and d2 >= 0, (d1, d2, d3))
    Uw = torch.stack([unit(ax["sea"].double()), unit(ax["sports"].double())]); Qw, _ = torch.linalg.qr(Uw.T)
    with WriteOrth(Uw): Ab = read_all_blocks(seqs, conts, token_blocks=list(range(28)))
    proj = max(float(((torch.cat(Ab["tok"][l]).double() @ Qw).norm(dim=-1) / torch.cat(Ab["tok"][l]).double().norm(dim=-1)).max()) for l in range(28))
    A2 = read_all_blocks(seqs, conts, blocks=[rk.BL])
    check("WriteOrth: no component in span(U) at any block; removed on exit", proj < 5e-3 and float((A2["X"][:, 0] - X1).abs().max()) < 1e-9 * float(X1.abs().max()) + 1e-12, f"max share {proj:.1e}")
    yes, no = rk.YES, rk.NO; fn = lambda lg: torch.stack([lg[:, -1, yes] - lg[:, -1, no], lg[:, -1, yes]], 1)
    sq = [rk.PIDS[0], rk.PIDS[1][:-2]]; pos = [[5, 6], [4]]; G = leaf_grads(sq, pos, [2, rk.BL], fn)
    gen = torch.Generator().manual_seed(1); e = unit(torch.randn(rk.D, generator=gen).double()); eps = 1e-3

    def y_at(scale):
        hh = []
        for r in range(2):
            def h(mod, inp, o, r=r):
                x = o[0] if isinstance(o, tuple) else o; x = x.clone(); x[0, pos[r]] += scale * e.float(); return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
            H = rk.LAY[rk.BL].register_forward_hook(h)
            with torch.no_grad(): lg = rk.model(torch.tensor([sq[r]])).logits.double()
            H.remove(); hh.append(float(lg[0, -1, yes] - lg[0, -1, no]))
        return np.array(hh)
    fd = (y_at(eps) - y_at(-eps)) / (2 * eps); an = (G[0, :, 1] @ e).numpy()
    check("leaf_grads equal finite differences", G.shape == (2, 2, 2, rk.D) and np.allclose(fd, an, rtol=2e-2, atol=1e-4 * max(1.0, float(np.abs(an).max()))), (fd, an))
    save()
    res_ok = all(r["ok"] for r in res); write_json(os.path.join(work, "r2_rk_selftest.json"), dict(ok=res_ok, tests=res, finder=FINDER_LOG, block=BLK))
    print(f"r2_rk selftest: {sum(r['ok'] for r in res)}/{len(res)} passed", flush=True); return res_ok


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "selftest":
        if len(sys.argv) < 4: raise SystemExit("usage: MOCK=1 python r2_rk.py selftest <root> <workdir>")
        os.environ["MOCK"] = "1"; MOCK = True; os.environ.setdefault("DEVICE", "cpu")
        sys.exit(0 if selftest(sys.argv[2], sys.argv[3]) else 1)
    raise SystemExit(__doc__)
