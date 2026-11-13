"""Round 2, session r2_ana: the report (frozen plan results/ju_space/plans/abschnitt11_12_runde2_plan.md, sha256 5e476340...edc2d,
sections 6, 9.4 and 14) and the helpers shared by the four analysis scripts of implementer 8 (r2_linearity.py, r2_rgi_verdict.py,
r2_unlearn_verdict.py, r2_report.py).

    python r2_report.py <r1_dir> <r2_root> <out.json>          -> <out.json> (round2_summary.json), <out_dir>/r2_manifest.json,
                                                                   <out_dir>/pc_toy/gates.json
    python r2_report.py selftest <workdir>                       synthetic code test (gates, reproduction check, finder, summary)

<r1_dir>   the copied round-1 files (<r1_dir>/gpu*/..., found by name across the shards, OD-G9)
<r2_root>  the copied round-2 outputs, one folder per session (<r2_root>/r2_q1/..., ..., <r2_root>/r2_ana/ = the outputs of the earlier
           analysis steps, mirrored by the runner); a file is looked up in the session that writes it first, then in every session
           folder, then next to <out.json>
What the report does (9.4): the gates of section 6 that r2_ana decides or records (G-meta / G-PC from PCS with G35's fallback, written to
pc_toy/gates.json; G-desc, G-blk, G-likely, G-attr; G7-G9, G34, G41, G42, G46 and the gates the producers recorded), the reproduction
checks of the retrained BP nets (G46) and of O7 (G34), one compact JSON per part (A, B, C) for the README with the verdicts, key numbers,
intervals, gates and labels, every deviation from the plan text (amendments, packing changes and continuation sessions, parts 'not run',
flags, labels, the readings of the implementers' NOTES), and the merged manifest of all sessions (r2_manifest.json).
Shared helpers (imported by the other three scripts): JSON I/O (strict JSON: NaN -> null, +-inf -> 'inf' / '-inf'), t intervals, the
round-2 input finder (Inputs), the round-1 gates (r1_info: B3, delta order, finality, G7-G9, G-desc, G-blk, G-likely, G-attr), the PC
gates (pc_gates: PCS, G-meta, G-PC(W), G35's fallback) and the circle gate G41."""
import datetime, glob, hashlib, json, math, os, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); JU = os.path.dirname(HERE); REPO = os.path.dirname(os.path.dirname(JU))
for _p in (HERE, JU):
    if _p not in sys.path: sys.path.insert(0, _p)

PLAN_REL = "results/ju_space/plans/abschnitt11_12_runde2_plan.md"
AMEND_REL = "results/ju_space/plans/abschnitt11_12_runde2_plan_amendments.md"
PLAN_SHA256 = "5e476340dfdb1d827582c869f2dfc08e32ed8e9a9dbebf7642acceb5bb5edc2d"
NOT_RUN_R1 = "not run (round-1 output missing)"
LBL_GDESC = "descriptive (round-1 gate failed)"                 # G-desc (2.2, G11)
LBL_RGI = "RGI axis not resolved"                               # GA(b) fails (2.4, G28)
LBL_BLK14 = "block 14 not usable"                               # G-blk(14) fails (C-Q5)
LBL_GMETA = "world without a testable meta level"               # G-meta fails (G35)
SESSIONS = ("r2_q1", "r2_q2", "r2_q3", "r2_t1", "r2_t2", "r2_t3", "r2_t4", "r2_pc", "r2_ana")
EXPECTED = (("unlearn/q1.json", "r2_ana"), ("unlearn_toy/t5.json", "r2_t4"), ("unlearn/", "r2_q1"), ("axes/", "r2_q1"), ("rgi/dyn_", "r2_q1"),
            ("rgi/", "r2_q3"), ("theta/", "r2_q2"), ("rgi_toy/", "r2_t1"), ("theta_toy/", "r2_t2"), ("unlearn_toy/", "r2_t3"), ("pc_toy/", "r2_pc"),
            ("theta_verdict.json", "r2_ana"), ("rgi_verdict.json", "r2_ana"), ("unlearn_verdict.json", "r2_ana"))


def log(msg):
    print(f"[r2_report {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ======================================================================================================================================
# JSON and files
# ======================================================================================================================================
def clean(o):
    """JSON-ready copy: tensors / numpy to lists and Python numbers, NaN -> None, +-inf -> 'inf' / '-inf' (strict JSON), keys to str."""
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)): return [clean(v) for v in (sorted(o, key=str) if isinstance(o, set) else o)]
    if type(o).__module__.startswith("torch") and hasattr(o, "detach"): return clean(o.detach().cpu().numpy())
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, (bool, np.bool_)): return bool(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (float, np.floating)):
        x = float(o)
        if math.isnan(x): return None
        if math.isinf(x): return "inf" if x > 0 else "-inf"
        return x
    return o


def write_json(path, obj):
    """Atomic JSON write (tmp + os.replace) of clean(obj)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True); tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(clean(obj), f, indent=1)
    os.replace(tmp, path)


def load_json(path):
    """The parsed file, or None if it is missing or unreadable (a half-written file never counts)."""
    try:
        with open(path) as f: return json.load(f)
    except (OSError, ValueError, TypeError):
        return None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def num(x):
    """float(x); NaN for None, 'nan', strings that are not numbers; 'inf' / '-inf' strings to +-inf."""
    if x is None or isinstance(x, bool): return float("nan") if x is None else float(x)
    try: return float(x)
    except (TypeError, ValueError): return float("nan")


def fin(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) and math.isfinite(float(x))


def get(d, *path, default=None):
    """Nested lookup d[k1][k2]... (dict keys or list indices; int keys also tried as str); default if any step is missing."""
    for k in path:
        if isinstance(d, dict):
            if k in d: d = d[k]
            elif str(k) in d: d = d[str(k)]
            else: return default
        elif isinstance(d, (list, tuple)) and isinstance(k, int) and -len(d) <= k < len(d): d = d[k]
        else: return default
    return d


def status_of(js, missing="missing"):
    """'ran' or the producer's status string (status / 'ran' when the JSON has none); 'missing' when js is None."""
    if js is None: return missing
    st = js.get("status") if isinstance(js, dict) else None
    return "ran" if st is None else str(st)


def ran(js):
    return status_of(js) in ("ran", "ok")


# ======================================================================================================================================
# Statistics
# ======================================================================================================================================
def tci(v, level=0.95):
    """Two-sided t interval over seeds or worlds (2.6; df = n - 1): dict(n, mean, sd, lo, hi); non-finite values dropped; lo / hi None
    with fewer than 2 values."""
    from scipy import stats
    a = np.array([num(x) for x in (np.ravel(np.asarray(v, dtype=object)) if v is not None else [])], dtype=float); a = a[np.isfinite(a)]; n = len(a)
    if n == 0: return dict(n=0, mean=None, sd=None, lo=None, hi=None)
    m = float(a.mean())
    if n < 2: return dict(n=1, mean=m, sd=None, lo=None, hi=None)
    sd = float(a.std(ddof=1)); h = float(stats.t.ppf(0.5 + level / 2, n - 1)) * sd / math.sqrt(n)
    return dict(n=n, mean=m, sd=sd, lo=m - h, hi=m + h)


def pct(draws, q):
    x = np.asarray(draws, dtype=float).ravel(); x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else float("nan")


def interval(draws, level=0.90):
    """Percentile interval (NaN draws dropped)."""
    a = 100 * (1 - level) / 2; return [pct(draws, a), pct(draws, 100 - a)]


def ge(a, b): return fin(a) and fin(b) and float(a) >= float(b)


def le(a, b): return fin(a) and fin(b) and float(a) <= float(b)


def gt(a, b): return fin(a) and fin(b) and float(a) > float(b)


def lt(a, b): return fin(a) and fin(b) and float(a) < float(b)


# ======================================================================================================================================
# Round-2 inputs: <r2_root>/<session>/<rel> (the expected session first), then any session folder, then the folder of <out.json>
# ======================================================================================================================================
def set_threads():
    """torch CPU threads = NT (set by r2_run_session.py), if given; results do not depend on it."""
    nt = os.environ.get("NT")
    if not nt: return None
    try:
        import torch
        torch.set_num_threads(max(1, int(nt))); return int(nt)
    except Exception:                                                             # noqa: BLE001
        return None


def expected_session(rel):
    for pre, s in EXPECTED:
        if rel.startswith(pre): return s
    return None


class Inputs:
    """Finder for the round-2 outputs (and the round-1 folder): every lookup is logged with its candidates and the chosen file."""

    def __init__(self, r1_dir, r2_root, out_path=None):
        self.r1 = os.path.abspath(r1_dir) if r1_dir else None; self.r2 = os.path.abspath(r2_root) if r2_root else None
        self.out_dir = os.path.dirname(os.path.abspath(out_path)) if out_path else None; self.log = []; self._cache = {}

    def candidates(self, rel):
        c = []
        s = expected_session(rel)
        if self.r2:
            if s: c.append(os.path.join(self.r2, s, rel))
            c += sorted(glob.glob(os.path.join(glob.escape(self.r2), "*", rel)))
        if self.out_dir: c.append(os.path.join(self.out_dir, rel))
        seen, out = set(), []
        for p in c:
            q = os.path.normpath(p)
            if q not in seen: seen.add(q); out.append(q)
        return out

    def find(self, rel):
        """Path of <rel> or None (logged)."""
        cands = [p for p in self.candidates(rel) if os.path.isfile(p)]
        rec = dict(rel=rel, expected=expected_session(rel), found=[self.short(p) for p in cands], chosen=self.short(cands[0]) if cands else None)
        if len(cands) > 1: rec["note"] = "several copies: the expected session's (or the first) is used"
        self.log.append(rec); return cands[0] if cands else None

    def find_all(self, pattern):
        """Every file matching <r2_root>/*/<pattern> (one session per relative path: the expected session's copy wins)."""
        by = {}
        if self.r2:
            for p in sorted(glob.glob(os.path.join(glob.escape(self.r2), "*", pattern))):
                rel = os.path.relpath(p, self.r2).split(os.sep, 1)[1]
                if rel not in by or os.path.relpath(p, self.r2).split(os.sep)[0] == expected_session(rel): by[rel] = p
        return [by[k] for k in sorted(by)]

    def short(self, p):
        if p is None: return None
        for base, tag in ((self.r2, "r2"), (self.out_dir, "out"), (self.r1, "r1")):
            if base and os.path.normpath(p).startswith(base + os.sep): return f"{tag}:{os.path.relpath(p, base)}"
        return p

    def json(self, rel):
        p = self.find(rel); return load_json(p) if p else None

    def pt(self, rel):
        """torch.load of <rel> (CPU) or None; cached (large files are read once per process)."""
        p = self.find(rel)
        if p is None: return None
        if p not in self._cache:
            import torch
            try: self._cache[p] = torch.load(p, map_location="cpu", weights_only=False)
            except Exception as e:                                                # noqa: BLE001 - a corrupt file counts as missing
                self.log.append(dict(rel=rel, error=repr(e))); return None
        return self._cache[p]

    def record(self, rels):
        """sha256 of the inputs used (for the output JSON)."""
        out = {}
        for rel in rels:
            p = self.find(rel)
            out[rel] = dict(path=self.short(p), sha256=sha256_file(p)) if p else None
        return out


# ======================================================================================================================================
# Round-1 gates (R1-1, R1-2): B3, delta order, finality (G4), G7-G9, G-desc (G11), G-blk (G12), G-likely (G13), G-attr (G14)
# ======================================================================================================================================
def likely_wh(V):
    """LIKELY_wh (2.1): V_RSI starts with 'likely', or with 'not decidable (pilot gate' and the text after 'would have read: ' starts with
    'likely'. Returns (bool, from_would_have_read)."""
    V = str(V or "")
    if V.startswith("likely"): return True, False
    if V.startswith("not decidable (pilot gate") and "would have read: " in V and V.split("would have read: ", 1)[1].startswith("likely"):
        return True, True
    return False, False


def r1_info(r1_dir):
    """Every round-1 gate the analysis scripts need, from the round-1 files only (never raises): dict(B3, b_lo, b_hi, order (b(1) <
    b(2) < b(3) by R1.delta), delta, final {b}, G7, G8, G9, G_desc, G_blk {b}, LIKELY_wh, from_would_have_read, V_RSI, G_attr,
    attracted_loop, summaries {b: verdict fields}, flags)."""
    import r2_rk as R
    out = dict(flags=[], final={}, G_blk={}, summaries={}, errors=[])
    try:
        info = R.r1_blocks(r1_dir)
    except Exception as e:                                                        # noqa: BLE001
        info = dict(b_lo=None, b_hi=None, B3=[14], flags=[f"blocks not readable: {e!r}"])
    out.update(B3=list(info.get("B3") or [14]), b_lo=info.get("b_lo"), b_hi=info.get("b_hi"), block_flags=list(info.get("flags") or []))
    out["G7"] = dict(ok="block choice inconsistent between shards" not in out["block_flags"], chosen=info.get("chosen"), ran=info.get("ran"),
                     label=None if "block choice inconsistent between shards" not in out["block_flags"] else "block choice inconsistent between shards")
    try:
        do = R.delta_order(r1_dir, info); out.update(order=do["order"], delta={str(k): v for k, v in do["delta"].items()}, G9=dict(ties=do["ties"], flags=do["flags"]))
    except Exception as e:                                                        # noqa: BLE001
        out.update(order=None, delta=None, G9=dict(error=repr(e))); out["errors"].append(f"delta order: {e!r}")
    p2, p3 = (os.path.join(r1_dir, s, "dn_delta.json") for s in ("gpu2", "gpu3")); d2, d3 = load_json(p2), load_json(p3)
    if d2 and d3:
        diffs = [abs(num(get(d2, "blocks", str(l), "last", "delta")) - num(get(d3, "blocks", str(l), "last", "delta"))) for l in range(28)]
        mx = max((x for x in diffs if math.isfinite(x)), default=float("nan"))
        out["G8"] = dict(max_abs_diff=mx, ok=bool(fin(mx) and mx <= 1e-6), used="gpu2")
    else: out["G8"] = dict(ok=None, note="gpu2 and gpu3 dn_delta.json not both present", gpu2=d2 is not None, gpu3=d3 is not None)
    for b in out["B3"]:
        s = R.load_r1_json(r1_dir, f"summary_b{b}.json", b, required=False) if b is not None else None
        out["final"][str(b)] = s is not None
        if s is None: out["G_blk"][str(b)] = dict(ok=None, status=NOT_RUN_R1); continue
        v = s.get("verdict") or {}; failed = list(get(v, "gates", "failed", default=[]) or [])
        out["summaries"][str(b)] = dict(PC_PASS=v.get("PC_PASS"), failed=failed, source=get(v, "gates", "source"), verdict=v.get("verdict"),
                                        label_attractor=v.get("label_attractor"), enters_delta_rule=v.get("enters_delta_rule"))
        out["G_blk"][str(b)] = dict(ok=bool(v.get("PC_PASS") is True and not failed), PC_PASS=v.get("PC_PASS"), failed=failed)
    s14 = out["summaries"].get("14")
    if s14 is None:
        out.update(G_desc=dict(failed=None, status=NOT_RUN_R1), V_RSI=None, LIKELY_wh=None, from_would_have_read=None, G_attr=None, attracted_loop=None)
    else:
        out["G_desc"] = dict(failed=bool(s14["failed"]), gates_failed=s14["failed"], label=LBL_GDESC if s14["failed"] else None)
        lk, wh = likely_wh(s14["verdict"]); out.update(V_RSI=s14["verdict"], LIKELY_wh=lk, from_would_have_read=wh)
        att = None
        loops = R.load_r1_json(r1_dir, "rk/b14/loops.json", 14, required=False)
        rows = {L: (loops or {}).get(L, [None] * 4)[3] if len((loops or {}).get(L) or []) >= 4 else None for L in ("RND_1", "RND_2")}
        if s14["label_attractor"]:
            hits = [L for L, r in rows.items() if r and le(r.get("p"), 0.01) and fin(r.get("A")) and abs(float(r["A"])) >= 0.25]
            att = hits[0] if hits else None
        out.update(G_attr=bool(s14["label_attractor"]) and att is not None, label_attractor=s14["label_attractor"], attracted_loop=att,
                   rnd_last_rows={L: (dict(p=r.get("p"), A=r.get("A")) if r else None) for L, r in rows.items()})
    out["finder"] = list(R.FINDER_LOG)
    return out


def desc_labels(info, b):
    """G-desc (G11): every round-2 Qwen verdict at block 14 carries 'descriptive (round-1 gate failed)'."""
    return [LBL_GDESC] if (int(b) == 14 and get(info, "G_desc", "failed")) else []


def load_bank(r1_dir, b):
    """(O3 bank, R1.state[b], R1.loops[b]) of block b (r2_rk.o3_bank). The RND loops' start directions come from rk_dirs.pt (POOLDIRS);
    r2_ana does not use them, so if the Pool file is not available the bank is built with zero RND starts (bank['_flag'])."""
    import torch
    import r2_rk as R
    ST = R.load_r1_state(r1_dir, b); loops = R.load_r1_json(r1_dir, f"rk/b{b}/loops.json", b)
    try:
        bank = R.o3_bank(ST, loops, b)
    except Exception as e:                                                        # noqa: BLE001 - Pool directions missing in r2_ana
        meta = loops["_meta"]; suf = meta.get("suffix", ""); ax = ST["axes"]; D = int(ax["u_T_perp"].shape[0])
        bank = dict(_meta=dict(meta), _missing=[], _flag=f"Pool directions not available ({e!r}): RND starts set to zero (not used in r2_ana)")
        starts = dict(RSI_a=ax["u_T_perp"], RSI_b=ax["u_T_perp"], SEA=ax["sea"], SPO=ax["sports"])
        for L in R.loops_of_block(b):
            e_ = R.loop_entry(ST, loops.get(L), L, suf, starts.get(L, torch.zeros(D)))
            if e_ is None: bank["_missing"].append(L)
            else: bank[L] = e_
    return bank, ST, loops


# ======================================================================================================================================
# PC gates (R1-9; 2.2 G-meta / G-PC(W); G35, G36): PCS = the union of gpu*/pc_summary_gpu*.json rows, evaluated with pc_analyze.gates
# ======================================================================================================================================
SHARD_RE = re.compile(r"^gpu\d+$")


def _shards(r1_dir):
    return sorted(d for d in (os.listdir(r1_dir) if r1_dir and os.path.isdir(r1_dir) else []) if SHARD_RE.match(d))


def r1_pc_pick(r1_dir, name, cands, log=None):
    """OD-G9 for a PC job found in several shards: the shard whose pc_expected_<shard>.json lists it, then the one with shard_done.json,
    then the most recent file. cands: [(shard, path)]."""
    if len(cands) <= 1: return cands[0] if cands else None
    exp = {s: set(load_json(os.path.join(r1_dir, s, f"pc_expected_{s}.json")) or []) for s, _ in cands}
    c1 = [c for c in cands if name in exp.get(c[0], set())] or cands
    c2 = [c for c in c1 if os.path.exists(os.path.join(r1_dir, c[0], "shard_done.json"))] or c1
    ch = max(c2, key=lambda c: (os.path.getmtime(c[1]), c[0]))
    if log is not None: log.append(dict(name=name, candidates=[c[0] for c in cands], chosen=ch[0], rule="pc_expected > shard_done > most recent"))
    return ch


def r1_pc_job(r1_dir, name, kind="jobs", log=None):
    """Path of a round-1 PC job file (kind 'jobs' -> .json, 'models' -> .pt) by name across the shards; a '_quick' variant of the name is
    accepted (code fixtures). None if missing."""
    ext = ".json" if kind == "jobs" else ".pt"
    for nm in (name, name + "_quick"):
        cands = [(s, os.path.join(r1_dir, s, "pc", kind, nm + ext)) for s in _shards(r1_dir) if os.path.isfile(os.path.join(r1_dir, s, "pc", kind, nm + ext))]
        c = r1_pc_pick(r1_dir, nm, cands, log)
        if c: return c[1]
    return None


def pcs_rows(r1_dir, log=None):
    """PCS: the union of the rows of gpu*/pc_summary_gpu*.json, one row per job name (duplicates by OD-G9)."""
    by = {}
    for s in _shards(r1_dir):
        for p in sorted(glob.glob(os.path.join(glob.escape(os.path.join(r1_dir, s)), "pc_summary_gpu*.json"))):
            for r in load_json(p) or []:
                if isinstance(r, dict): by.setdefault(r.get("name"), []).append((s, p, r))
    rows = []
    for name, lst in sorted(by.items(), key=lambda kv: str(kv[0])):
        ch = r1_pc_pick(r1_dir, str(name), [(s, p) for s, p, _ in lst], log)
        rows.append(next(r for s, p, r in lst if (s, p) == ch))
    return rows


def g_meta_from_round2(t1_json, W=128):
    """G35's fallback: G1 (>= 4 of 5 seeds reach level 1) and G2 (B2(8) seed mean >= 0.10 with t lower bound > 0.03 over the level-1
    seeds; pc_analyze.ci) on r2_t3's round-2 BP-muP W 128 nets (unlearn_toy/t1.json: the uncut nets' level 1 and B2(8)); None if the
    cell is not there."""
    import pc_analyze as PA
    c = get(t1_json, "cells", f"bp_mup_W{W}")
    lv, b2 = get(c, "base", "level1"), get(c, "base", "B2", "8")
    if lv is None or b2 is None: return None
    lv = [bool(x) for x in lv]; n1 = sum(lv); b = PA.ci([num(v) for v, l in zip(b2, lv) if l and fin(num(v))])
    g1, g2 = n1 >= 4, bool(b["mean"] is not None and b["mean"] >= 0.10 and b["lo"] is not None and b["lo"] > 0.03)
    return dict(G1=dict(pass_=g1, level1_seeds=n1, n_seeds=len(lv)), G2=dict(pass_=g2, **b), pass_=g1 and g2, label="gate from round 2")


def pc_gates(r1_dir, inp=None):
    """G-meta (G35) and G-PC(W) (G36) from PCS with pc_analyze.gates (round-1 code, unchanged). G-meta = G1 and G2; if PCS has no BP-muP
    W 128 main row, G1 / G2 come from r2_t3's round-2 nets (unlearn_toy/t1.json), labelled 'gate from round 2'. G-PC(W) = G-meta and
    G0_pc_mup and G3['pc_mup_W<W>'] and G4[W] and G5[W] and >= 4 seeds with level 1 (rows arm pc_mup, W, K 12, gamma 0, wd 1); a width
    failing only G5 is 'BP-equivalent'. Never raises."""
    import pc_analyze as PA
    flog = []; rows = pcs_rows(r1_dir, flog) if r1_dir else []
    out = dict(n_rows=len(rows), shards=_shards(r1_dir) if r1_dir else [], finder=flog, G_PC={}, labels=[])
    try:
        G = PA.gates(rows)
    except Exception as e:                                                        # noqa: BLE001 - malformed rows: no gate passes
        G = None; out["error"] = f"pc_analyze.gates failed: {e!r}"
    out["G"] = G
    has128 = any(r.get("W") == 128 for r in PA.main_rows(rows, "bp_mup"))
    src = "PCS (round 1)"; g1 = bool(get(G, "G1", "pass_")); g2 = bool(get(G, "G2", "pass_"))
    if not has128:
        fb = g_meta_from_round2(inp.json("unlearn_toy/t1.json")) if inp is not None else None
        out["G35_fallback"] = fb
        if fb is not None: g1, g2, src = fb["G1"]["pass_"], fb["G2"]["pass_"], "gate from round 2"
        else: src = "not decidable (no BP-muP W 128 rows in PCS and no round-2 BP-muP W 128 nets)"
    gm = bool(g1 and g2)
    out.update(G_meta=gm, G_meta_source=src, G_meta_parts=dict(G1=g1, G2=g2), G35_fallback_needed=not has128)
    if src == "gate from round 2": out["labels"].append("gate from round 2")
    if not gm: out["labels"].append(LBL_GMETA)
    mr = PA.main_rows(rows, "pc_mup")
    for W in sorted({r["W"] for r in mr}):
        n1 = sum(1 for r in mr if r["W"] == W and r.get("level1"))
        c = dict(G_meta=gm, G0_pc_mup=bool(get(G, "G0_pc_mup", "pass_")), G3=bool(get(G, "G3", f"pc_mup_W{W}", "pass_")),
                 G4=bool(get(G, "G4", W, "pass_")), G5=bool(get(G, "G5", W, "pass_")), level1_4_of_5=n1 >= 4)
        ok = all(c.values()); bpe = (not c["G5"]) and all(v for k, v in c.items() if k != "G5")
        out["G_PC"][str(W)] = dict(pass_=ok, conds=c, n_level1=n1, bp_equivalent=bpe,
                                   status="usable" if ok else ("BP-equivalent (not used for PC claims)" if bpe else "not testable"))
    out["widths_G_PC"] = sorted(int(W) for W, v in out["G_PC"].items() if v["pass_"])
    return out


# ======================================================================================================================================
# G41: the C-T3 circle gate ('ring zero-shot >= 0.9 and pring <= 0.2 in >= 4 of 5 seeds'; if it fails, no pattern claim in Part C)
# ======================================================================================================================================
def quorum(n):
    """'>= 4 of 5 seeds' (5.1), scaled for other seed counts: ceil(0.8 n)."""
    return int(math.ceil(0.8 * n - 1e-9))


def g41(inp):
    """G41 from unlearn_toy/t3.json (gate['K1'] arms ring / pring, per seed zero_shot), K = 1 primary. Returns dict(status, passes, ...)."""
    t3 = inp.json("unlearn_toy/t3.json") if inp is not None else None
    if t3 is None: return dict(status="not run (round-2 output missing: unlearn_toy/t3.json)", passes=None)
    rg = sorted(get(t3, "gate", "K1", "ring", default=[]) or [], key=lambda d: d.get("ws", 0))
    pg = sorted(get(t3, "gate", "K1", "pring", default=[]) or [], key=lambda d: d.get("ws", 0))
    if not rg or not pg: return dict(status="not computable (gate K1 ring / pring missing in t3.json)", passes=None)
    pr = {d.get("ws"): d for d in pg}; both = sum(1 for r in rg if ge(r.get("zero_shot"), 0.9) and le(get(pr, r.get("ws"), "zero_shot"), 0.2))
    n = len(rg); ok = both >= quorum(n)
    return dict(status="ran", passes=bool(ok), seeds_both=both, n_seeds=n, quorum=quorum(n),
                ring=[r.get("zero_shot") for r in rg], pring=[get(pr, r.get("ws"), "zero_shot") for r in rg],
                note=None if ok else "C-T3 circle gate failed: no pattern claim anywhere in Part C (G41)")


# ======================================================================================================================================
# Reproduction checks: G46 (retrained BP nets against the round-1 jobs), G34 (O7 against R1.delta)
# ======================================================================================================================================
NET_RE = re.compile(r"^(bp_mup|bp_sp)_W(\d+)_w(\d+)_i(\d+)")


def job_name(arm, W, s, K=12, gamma=0.0, wd=1.0):
    """pc_meta's / run_shard's job name of a round-1 cell."""
    return f"{arm}_W{W}_s{s}_K{K}_g{gamma:g}_wd{wd:g}" + ("_T20_beta0.1" if arm == "pc_mup" else "")


def g46_round2_rows(inp):
    """The BP nets that round 2 retrained in a round-1 definition (K 12 or 1, gamma 0, wd 0 or 1; init 0 = pc_meta's init), as
    [(source, cell = (arm, W, K, gamma, wd), seed, dict(first95, B2_8, delta_h2, level1, name))]. Sources: AT2's natural nets and AT5's
    nets (r2_t1; the cell is derived from the net name '<arm>_W<W>_w<ws>_i0', because r2_rgi_toy's 'r1_job_name' carries the condition
    arm), B-T1 / B-T2 (r2_t2: pc_summary-style rows), the C-T one-hot nets (r2_t3: unlearn_toy/t1.json nets rows and uncut B2(8))."""
    rows = []
    for pat, src in (("rgi_toy/r1/w*.json", "AT2 natural nets (r2_t1)"), ("rgi_toy/cross/w*.json", "AT5 nets (r2_t1)")):
        for p in inp.find_all(pat):
            for m in (load_json(p) or {}).get("models") or []:
                name = str(m.get("name", "")); mt = NET_RE.match(name)
                if not mt or int(mt.group(4)) != 0 or m.get("cond", "nat") != "nat" or any(t in name for t in ("gamma", "when", "reg")): continue
                rows.append((src, (mt.group(1), int(mt.group(2)), 12, 0.0, 1.0), int(mt.group(3)),
                             dict(first95=m.get("first95"), B2_8=m.get("B2_8"), delta_h2=m.get("delta_h2"), level1=m.get("level1"), name=name)))
    for r in inp.json("theta_toy/nets.json") or []:
        if not isinstance(r, dict) or r.get("arm") not in ("bp_mup", "bp_sp"): continue
        rows.append(("B-T1 / B-T2 nets (r2_t2)", (r["arm"], int(r["W"]), int(r["K"]), float(r.get("gamma") or 0.0), float(r.get("wd_scale", 1.0))), int(r["seed"]),
                     dict(first95=r.get("first95"), B2_8=r.get("B2_8"), delta_h2=get(r, "delta", "h2", "delta"), level1=r.get("level1"), name=r.get("name"))))
    t1 = inp.json("unlearn_toy/t1.json"); seeds = get(t1, "meta", "seeds") or []
    for cname, c in (get(t1, "cells") or {}).items():
        mt = re.match(r"^(bp_mup|bp_sp)_W(\d+)$", cname)
        if not mt: continue
        b2 = get(c, "base", "B2", "8") or []
        for i, nr in enumerate(c.get("nets") or []):
            m2 = NET_RE.match(str(nr.get("name", ""))); ws = int(m2.group(3)) if m2 else (seeds[i] if i < len(seeds) else i)
            rows.append(("C-T one-hot nets (r2_t3)", (mt.group(1), int(mt.group(2)), 12, 0.0, 1.0), int(ws),
                         dict(first95=nr.get("first95"), B2_8=b2[i] if i < len(b2) else None, delta_h2=None, level1=nr.get("level1"), name=nr.get("name"))))
    return rows


def g46(r1_dir, inp):
    """G46 (descriptive): per retrained cell and round-2 source, |difference of the seed means of B2(8)| <= 0.05 and |difference of
    first95| <= 500 updates in >= 4 of 5 seeds (in every seed of B-T1's 4-seed cells); otherwise the label 'reproduction differs' on
    that cell's round-2 results (no verdict changes). Seeds enter when both the round-1 job and the round-2 net exist."""
    by, flog = {}, []
    for src, cell, seed, v in g46_round2_rows(inp): by.setdefault((src, cell), {})[seed] = v
    out = dict(cells={}, differing=[], compared=0, not_compared=0, finder=flog)
    for (src, cell), seeds in sorted(by.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        arm, W, K, gamma, wd = cell; pairs = []
        for s, v in sorted(seeds.items()):
            nm = job_name(arm, W, s, K, gamma, wd); p = r1_pc_job(r1_dir, nm, "jobs", flog) if r1_dir else None
            if p is None: continue
            j = load_json(p) or {}
            pairs.append(dict(seed=s, job=nm, r1=dict(first95=j.get("first95"), B2_8=get(j, "finetune", "8", "B2"), delta_h2=get(j, "delta", "h2", "delta"), level1=j.get("level1")),
                              r2=v, r1_model_present=r1_pc_job(r1_dir, nm, "models") is not None))
        key = f"{src} | {job_name(arm, W, 's', K, gamma, wd).replace('_ss_', '_s*_')}"
        if not pairs:
            out["cells"][key] = dict(status="no round-1 job for this cell (not compared)", n_round2=len(seeds)); out["not_compared"] += 1; continue
        n = len(pairs); b1 = [num(p["r1"]["B2_8"]) for p in pairs]; b2 = [num(p["r2"]["B2_8"]) for p in pairs]
        m1 = float(np.nanmean(b1)) if np.isfinite(b1).any() else float("nan"); m2 = float(np.nanmean(b2)) if np.isfinite(b2).any() else float("nan")
        dB = abs(m1 - m2) if (fin(m1) and fin(m2)) else float("nan")

        def f_ok(p):
            a, b = p["r1"]["first95"], p["r2"]["first95"]
            if a is None and b is None: return True                               # neither reached 0.95 (or K = 1: pc_meta records none)
            return bool(fin(num(a)) and fin(num(b)) and abs(num(a) - num(b)) <= 500)
        nf = sum(f_ok(p) for p in pairs); need = n if src.startswith("B-T1") else quorum(n)
        dd = [abs(num(p["r1"]["delta_h2"]) - num(p["r2"]["delta_h2"])) for p in pairs]
        ok_b2 = bool(fin(dB) and dB <= 0.05); ok = bool(ok_b2 and nf >= need)
        out["cells"][key] = dict(status="compared", n_seeds=n, seeds=[p["seed"] for p in pairs], B2_8_mean_r1=m1, B2_8_mean_r2=m2, abs_diff_B2_8=dB, B2_ok=ok_b2 if fin(dB) else None,
                                 first95_r1=[p["r1"]["first95"] for p in pairs], first95_r2=[p["r2"]["first95"] for p in pairs], first95_within_500=nf, first95_needed=need,
                                 delta_h2_abs_diff_max=float(np.nanmax(dd)) if np.isfinite(dd).any() else None, ok=ok, label=None if ok else "reproduction differs",
                                 note=None if fin(dB) else "B2(8) not available on one side: the cell cannot pass",
                                 r1_models_present=sum(p["r1_model_present"] for p in pairs))
        out["compared"] += 1
        if not ok: out["differing"].append(key)
    out["rule"] = "per cell: |seed-mean B2(8) difference| <= 0.05 and |first95 difference| <= 500 in >= 4 of 5 seeds (every seed of B-T1's 4-seed cells)"
    return out


def g34(r1_dir, inp, model="qwen3-1.7b"):
    """G34: |delta_r2 - R1.delta| <= 0.002 at every block and reading (gpu2/dn_delta.json against O7 in rgi/geom_meta_<m>.json); a
    mismatch is flagged 'delta reproduction mismatch' (R1 values stay in force for orderings)."""
    d = None
    for s in ("gpu2", "gpu3"):
        d = load_json(os.path.join(r1_dir, s, "dn_delta.json")) if r1_dir else None
        if d is not None: src = f"{s}/dn_delta.json"; break
    gm = inp.json(f"rgi/geom_meta_{model}.json")
    if d is None: return dict(status=NOT_RUN_R1 + " (dn_delta.json)")
    if gm is None: return dict(status=f"not run (round-2 output missing: rgi/geom_meta_{model}.json)")
    out = dict(status="ran", source=src, tolerance=0.002, readings={})
    for rd in ("last", "mean"):
        r2v = get(gm, "stages", "o7", "delta", rd, "delta") or []
        diffs = [abs(num(get(d, "blocks", str(l), rd, "delta")) - num(r2v[l])) for l in range(len(r2v))]
        mx = float(np.nanmax(diffs)) if (diffs and np.isfinite(diffs).any()) else float("nan")
        out["readings"][rd] = dict(max_abs_diff=mx, n_blocks=len(r2v), ok=bool(fin(mx) and mx <= 0.002 and np.isfinite(diffs).all()), worst_block=int(np.nanargmax(diffs)) if fin(mx) else None)
    out["ok"] = all(v["ok"] for v in out["readings"].values())
    out["flag"] = None if out["ok"] else "delta reproduction mismatch"
    return out


# ======================================================================================================================================
# Walking the verdict files: labels, flags, parts 'not run'
# ======================================================================================================================================
def collect(obj, keys=("labels",), path="", out=None, depth=0, max_depth=9):
    """[(path, value)] for every non-empty list / string under one of the given keys."""
    out = [] if out is None else out
    if depth > max_depth: return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}/{k}" if path else str(k)
            if k in keys and v:
                for x in (v if isinstance(v, list) else [v]):
                    if isinstance(x, str): out.append((path, x))
            elif isinstance(v, (dict, list)): collect(v, keys, p, out, depth + 1, max_depth)
    elif isinstance(obj, list) and len(obj) <= 64:
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)): collect(v, keys, f"{path}[{i}]", out, depth + 1, max_depth)
    return out


def collect_not_run(obj, path="", out=None, depth=0, max_depth=7):
    """[(path, status)] for every 'status' / 'verdict' string that starts with 'not run', 'not testable' or 'not computable'."""
    out = [] if out is None else out
    if depth > max_depth: return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}/{k}" if path else str(k)
            if k in ("status", "verdict") and isinstance(v, str) and v.startswith(("not run", "not testable", "not computable", "not decidable", "ran without")): out.append((path, v))
            elif isinstance(v, (dict, list)): collect_not_run(v, p, out, depth + 1, max_depth)
    elif isinstance(obj, list) and len(obj) <= 64:
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)): collect_not_run(v, f"{path}[{i}]", out, depth + 1, max_depth)
    return out


def uniq(pairs):
    seen, out = set(), []
    for p in pairs:
        if p not in seen: seen.add(p); out.append(p)
    return out


# ======================================================================================================================================
# The parts A, B, C (compact, for the README)
# ======================================================================================================================================
def part_A(rv, info, pcg):
    if rv is None: return dict(status="not run (round-2 output missing: rgi_verdict.json)")
    ob = get(rv, "A7", "over_blocks") or {}; pb = get(rv, "A7", "per_block", "blocks") or {}
    key = dict(passing_set=get(rv, "A7", "per_block", "P"), blocks_with_RGI_geometry=ob.get("present_blocks"), top_third=ob.get("top"), bottom_third=ob.get("bottom"),
               over_blocks=ob.get("numbers"), separability=ob.get("separability"), r_DiD=ob.get("arm_0p6B"),
               per_block={b: dict(GI=v.get("GI"), IS=v.get("IS"), BA=v.get("BA"), present=v.get("present"), self=v.get("self"), self_LR=v.get("self_LR"),
                                  causal_general=v.get("causal_general"), causal_absorption=v.get("causal_absorption"), labels=v.get("labels"),
                                  FG=get(v, "numbers", "FG"), dFG=get(v, "numbers", "dFG"), dFG_sim=get(v, "numbers", "dFG_sim")) for b, v in pb.items()},
               rsi_into_rgi={k: get(ob, "rsi_into_rgi", k) for k in ("D_rho_S", "D_rho_S_ci", "D_sigma_self", "D_sigma_self_ci", "absorbed_top", "self_specific_top")},
               causal_line={k: get(ob, "causal_line", k) for k in ("D_FG_causal", "D_FG_causal_ci", "D_FG_causal_sim", "D_rho_causal", "D_rho_causal_ci", "D_rho_causal_sim", "intervals", "verdict_with_pointwise_intervals")},
               spillover_u_T_perp={b: {k: v.get(k) for k in ("z_self", "z_other", "FR", "verdict", "status")} for b, v in (rv.get("A3_u_T_perp_spillover") or {}).items() if isinstance(v, dict)},
               spillover_g_self=get(rv, "A7", "g_self_spillover"),
               pull_and_delta={k: get(rv, "A6", "pull_and_delta", k) for k in ("X", "ci90", "diff_high_minus_low", "diff_ci90", "entering", "middle_clause", "non_overlap_form")},
               gen_loops={b: {k: v.get(k) for k in ("status", "E_GI", "R4", "S4", "auc_RGI")} for b, v in (get(rv, "A6", "gen") or {}).items()},
               self_loop_rows={b: get(v, "self_loop", "rsi_rows") for b, v in (get(rv, "A6", "dynamics") or {}).items() if get(v, "self_loop")},
               toy=dict(valid_worlds=get(rv, "AT8", "valid_worlds"), answer=get(rv, "AT8", "answer"), answer_early_arm=get(rv, "AT8", "answer_early_arm"),
                        lines={arm: {nm: dict(verdict=l.get("verdict"), b_plus=get(l, "reduced", "b+"), b_minus=get(l, "reduced", "b-"), e=get(l, "reduced", "e"),
                                              d_minus=get(l, "full", "d-"), d_plus=get(l, "full", "d+"), n_worlds=l.get("n_worlds"))
                                     for nm, l in (get(rv, "AT8", "lines", arm, "lines") or {}).items()} for arm in ("late", "early")},
                        Lambda_natural=get(rv, "AT8", "G39", "Lambda_natural"), recursion={k: get(rv, "AT8", "recursion", k) for k in ("verdict", "rho_d", "RG", "pooling_gain_natural")},
                        family={k: dict(trains=v.get("trains"), S_gain=v.get("S_gain"), portable=v.get("portable")) for k, v in (get(rv, "AT8", "family", "variants") or {}).items()},
                        AT5_slopes=get(rv, "AT5", "slopes"), AT5_pc_widths=get(rv, "AT5", "pc_widths"),
                        AT7_rates={arm: {nm: dict(noise=l.get("noise"), rates=l.get("rates"), observed=l.get("observed_verdict")) for nm, l in (get(rv, "AT7", "arms", arm) or {}).items()} for arm in ("late", "early")},
                        AT0_levels=get(rv, "AT0", "levels"), AT6=get(rv, "AT6", "summary")))
    return dict(status="ran", verdicts=rv.get("summary"), key_numbers=key,
                gates=dict(G7=get(rv, "blocks", "G7"), G8=get(rv, "blocks", "G8"), G9=get(rv, "blocks", "G9"), G_desc=get(rv, "gates", "G_desc"), G_blk=get(rv, "gates", "G_blk"),
                           GA=get(rv, "gates", "GA"), G30=get(rv, "A7", "gates", "G30"), G31=get(rv, "A7", "gates", "G31"), G33=get(rv, "A7", "gates", "G33"),
                           G38=get(rv, "AT8", "G38"), G39=get(rv, "AT8", "G39"), G40=get(rv, "AT8", "family", "G40"), G_PC=get(rv, "gates", "G_PC")),
                labels=uniq(collect(rv, ("labels",))), flags=uniq(collect(rv, ("flags",))), not_run=uniq(collect_not_run(rv)),
                note="Qwen: nothing edits weights or caches (no ability claim); three blocks never separate delta from depth; the toy has no KV cache (2.7 C23)")


def part_B(tv, ttv, t5, pcg):
    out = dict(qwen=None, toy=None)
    if tv is None: out["qwen"] = dict(status="not run (round-2 output missing: theta_verdict.json)")
    else:
        blocks = {}
        for b, v in (tv.get("blocks") or {}).items():
            if not isinstance(v, dict): continue
            r6 = {c: dict(K=get(x, "R6", "K"), verdict=get(x, "R6", "verdict"), transfer=get(x, "R6", "transfer", "verdict"), defined=get(x, "R6", "defined"))
                  for c, x in (get(v, "R6", "cells") or {}).items() if isinstance(x, dict)}
            blocks[b] = dict(status=v.get("status"), labels=v.get("labels"), descriptive=v.get("descriptive"), mode=v.get("mode"), G_blk=v.get("G_blk"),
                             R1=dict(verdict=get(v, "R1", "verdict"), alpha_star=get(v, "R1", "alpha_star"), D_star=get(v, "R1", "D_star"), onset=get(v, "R1", "onset"),
                                     labels=get(v, "R1", "labels"), resolvable_band=get(v, "R1", "resolvable_band")),
                             R2=dict(verdict=get(v, "R2", "verdict"), slope=get(v, "R2", "slope"), T_c=get(v, "R2", "T_c"), margin=get(v, "R2", "margin"), labels=get(v, "R2", "labels")),
                             R4=dict(verdict=get(v, "R4", "verdict"), verdict_descriptive=get(v, "R4", "verdict_descriptive"), note=get(v, "R4", "note")),
                             R5=v.get("R5"), R6=r6, Gamma=v.get("Gamma_R3"), enters_R3=v.get("enters_R3"), sentence=v.get("sentence"), od_g8=v.get("od_g8"),
                             m_star2=get(v, "B_Q0", "m_star2"), cross_fit={k: get(v, "B_Q0", "cross_fit", k) for k in ("cos_AB", "lambda_cf", "g_par_AB", "diff", "switch", "reason")},
                             independent_loop=get(v, "B_Q0", "independent_loop"), power_step=get(v, "B_Q6", "status"))
        und = theta_undefined(tv)
        for b, e in blocks.items():                                                # Amendment 3 (b): the share of undefined replicates next to each rule verdict
            for rk in ("R1", "R2", "R4", "R5"):
                src = get(tv, "blocks", b, rk)
                if isinstance(src, dict) and isinstance(e.get(rk), dict): e[rk].update(nonfinite_share=src.get("nonfinite_share"), bounds_undefined=src.get("bounds_undefined"))
                elif isinstance(src, dict) and ("nonfinite_share" in src or "bounds_undefined" in src): e[rk] = dict(verdict=e.get(rk), nonfinite_share=src.get("nonfinite_share"), bounds_undefined=src.get("bounds_undefined"))
            for c, x in (e.get("R6") or {}).items():
                src = get(tv, "blocks", b, "R6", "cells", c, "R6")
                if isinstance(src, dict): x.update(nonfinite_share=src.get("nonfinite_share"), bounds_undefined=src.get("bounds_undefined"))
        out["qwen"] = dict(status=status_of(tv), blocks=blocks, R3={k: (get(v, "verdict") if isinstance(v, dict) else v) for k, v in (tv.get("R3") or {}).items()},
                           undefined_replicates=dict(entries=und, undecided_for_undefined_bounds=[e for e in und if e.get("bounds_undefined")],
                                                     rule="Amendment 3 (b): a rule bound is undefined ('undecided') when the point value or more than 5 % of the replicates are undefined"),
                           R3_detail=tv.get("R3"), reserves=tv.get("reserves"), gates=tv.get("gates"), labels_global=tv.get("labels_global"), delta_order=tv.get("delta_order"),
                           labels=uniq(collect(tv, ("labels", "labels_global"))), not_run=uniq(collect_not_run(tv)), note=tv.get("note"))
    if ttv is None: out["toy"] = dict(status="not run (round-2 output missing: theta_toy/verdict.json)")
    else:
        gm = pcg.get("G_meta")
        out["toy"] = dict(status="ran", P1={W: {wd: get(c, "phase") for wd, c in cells.items()} for W, cells in (ttv.get("P1") or {}).items()}, P2=ttv.get("P2"), P3=ttv.get("P3"),
                          P4=ttv.get("P4"), B_T2=ttv.get("B_T2"), T5=dict(G27=get(ttv, "T5", "G27") or get(t5, "G27"), phases=get(ttv, "T5", "phases"), tau_c=get(ttv, "T5", "tau_c"),
                                                                          tau_c_regression=get(ttv, "T5", "tau_c_regression"), descriptive=get(ttv, "T5", "descriptive")),
                          susceptibility_peak=get(ttv, "B_T4", "peak"), G_meta=gm, labels=([] if gm else [LBL_GMETA + " (B-T1)"]),
                          note="B-T1 is a regularisation x capacity diagram of the first-order meta level; the self-referential theta is B-T5's")
    out["verdicts"] = dict(qwen={b: dict(R1=get(v, "R1", "verdict"), R2=get(v, "R2", "verdict"), R4=get(v, "R4", "verdict") or get(v, "R4", "verdict_descriptive"),
                                         R5=get(v, "R5", "verdict"), R6={c: x.get("verdict") for c, x in (v.get("R6") or {}).items()}) for b, v in (get(out, "qwen", "blocks") or {}).items()},
                           R3=get(out, "qwen", "R3"), toy=dict(P3=get(ttv, "P3", "verdict"), P4=get(ttv, "P4", "verdict"), G27=get(out, "toy", "T5", "G27")) if ttv else None,
                           undecided_for_undefined_bounds=[f"{e['where']}: {e.get('verdict')!r} ({100 * e['share_max']:.1f} % of the replicates undefined)" if fin(e.get("share_max")) else
                                                           f"{e['where']}: {e.get('verdict')!r} (point value undefined)" for e in (get(out, "qwen", "undefined_replicates", "undecided_for_undefined_bounds") or [])])
    return out


def part_C(uv, q1, pcg):
    if uv is None: return dict(status="not run (round-2 output missing: unlearn_verdict.json)", C_Q1=None if q1 is None else q1.get("statements"))
    num_q2 = {k: dict(verdict=v.get("verdict"), failed=v.get("failed"), **{x: get(v, "inputs", x) for x in ("ratio", "ratio_ci90", "Ue_fin", "z_fin", "M0", "damage")}, ctrl_p5=v.get("ctrl_p5"), labels=v.get("labels"))
              for k, v in (get(uv, "C_Q2", "directions") or {}).items()}
    num_q3 = {k: dict(verdict=v.get("verdict"), inverted=v.get("inverted"), failed=v.get("failed"), **{x: get(v, "inputs", x) for x in ("ratio", "ratio_ci90", "D", "D0", "z_D", "damage")}, ctrl_p5=v.get("ctrl_p5"),
                      labels=v.get("labels")) for k, v in (get(uv, "C_Q3", "rules") or {}).items()}
    key = dict(C_Q1=None if q1 is None else dict(statements=q1.get("statements"), G42=q1.get("G42"),
                                                 concepts={c: dict(nonlinear_auc=get(v, "nonlinear", "auc"), p99=get(v, "nonlinear", "p99"), md_auc=get(v, "md_auc", "auc"), auc_cf=get(v, "auc_cf", "auc"),
                                                                   inlp_r_star=get(v, "inlp", "r_star_max"), ratio_1dim=get(v, "linear_rung", "ratio_1dim"), ratio_span=get(v, "linear_rung", "ratio_span"))
                                                           for c, v in (q1.get("concepts") or {}).items()},
                                                 calibration=dict(d_T=get(q1, "calibration", "d_T"), xor_auc_median=get(q1, "calibration", "plants", "xor", "auc_median"),
                                                                  onedim_auc_cf_median=get(q1, "calibration", "plants", "onedim", "auc_cf_median"))),
               C_Q2=num_q2, C_Q3=num_q3, C_Q3_G43=get(uv, "C_Q3", "G43"),
               C_Q4a={k: get(uv, "C_Q4a", k) for k in ("verdict", "ablation", "restored_share", "restored_share_ci90", "ceiling", "passed_C_Q3_r2_ana")},
               C_Q4b=dict(verdict=get(uv, "C_Q4b", "verdict"), gate=get(uv, "C_Q4b", "gate"), rows=get(uv, "C_Q4b", "rows"), reference=get(uv, "C_Q4b", "matched_size_reference", "rows")),
               C_Q4_armA={k: get(uv, "C_Q4_armA", k) for k in ("verdict", "c1", "reference", "G16", "rounds_ran", "comparator", "under_plan_text")},
               C_Q5=dict(superposition=get(uv, "C_Q5", "superposition"), symmetry=get(uv, "C_Q5", "symmetry"), dose=get(uv, "C_Q5", "dose"),
                         svd=dict(singular_values=get(uv, "C_Q5", "svd", "singular_values"), variance_share=get(uv, "C_Q5", "svd", "variance_share"),
                                  named_modes=get(uv, "C_Q5", "svd", "named_mode_cos_with_top_modes"))),
               C_Q6=dict(statement=get(uv, "C_Q6", "statement"), rsi_specific=get(uv, "C_Q6", "rsi_specific"),
                         relations={r: dict(helix=v.get("helix_primary"), depth_cap=v.get("depth_cap"), caps=v.get("caps"), self_repair=v.get("self_repair")) for r, v in (get(uv, "C_Q6", "relations") or {}).items()}),
               C_T7=dict(verdict=get(uv, "C_T7", "verdict"), family_size=get(uv, "C_T7", "family", "family_size"), claims=get(uv, "C_T7", "family", "claims"), under_plan_text=get(uv, "C_T7", "under_plan_text"),
                         items=[{k: r.get(k) for k in ("W", "item", "n", "mean", "lo", "hi", "p", "p_holm", "claim", "testable")} for r in (get(uv, "C_T7", "family", "items") or [])]))
    return dict(status="ran", verdicts=uv.get("summary"), key_numbers=key, toy=compact_toy(uv),
                gates=dict(uv.get("gates") or {}, G42=None if q1 is None else q1.get("G42"), G43=get(uv, "C_Q3", "G43"), G44=get(uv, "C_Q4b", "gate"), G16=get(uv, "C_Q4_armA", "G16")),
                labels=uniq(collect(uv, ("labels",)) + (collect(q1, ("labels",)) if q1 else [])), od_g8=uv.get("od_g8"), not_run=uniq(collect_not_run(uv)),
                amendment_2=uv.get("amendment_2"), amendment_3=uv.get("amendment_3"),
                note="a push is at most 'suppressed'; 'unlearned' only when the instructed or cued ability goes away specifically")


def compact_toy(uv):
    """The toy verdicts of Part C with their key numbers (the full detail stays in unlearn_verdict.json)."""
    t4 = {}
    for name, c in (get(uv, "C_T4", "cells") or {}).items():
        t4[name] = dict(ladder=c.get("ladder"), base_acc=c.get("base_acc"),
                        ops={nm: dict(unlearns=o.get("unlearns"), specificity=o.get("specificity"), depth=o.get("depth"), meta_unlearned=o.get("meta_unlearned"), inverted=o.get("inverted"),
                                      acc=get(o, "acc", "mean"), acc_hi=get(o, "acc", "hi"), dm_acc_lo=get(o, "dm_acc", "lo"), dm=o.get("dm"), under_plan_text=o.get("under_plan_text"))
                             for nm, o in (c.get("ops") or {}).items()},
                        t4b={L: {k: t.get(k) for k in ("causal", "correlate", "nonlinear_residue", "dm_matched")} for L, t in (c.get("t4b") or {}).items()})
    return dict(C_T0={k: {x: v.get(x) for x in ("has_ring_code", "fourier_aligned", "ring_seeds", "fourier_aligned_seeds", "n_seeds")} for k, v in (get(uv, "C_T0", "cells") or {}).items()},
                C_T1={k: {f: dict(label=x.get("label"), kstar_pca=x.get("kstar_pca"), kstar_control=x.get("kstar_control")) for f, x in (v.get("families") or {}).items()}
                      for k, v in (get(uv, "C_T1", "cells") or {}).items()},
                C_T2={k: get(uv, "C_T2", k) for k in ("k_min", "ring_linearises", "ring_linearises_by", "pattern_one_hot_world", "effects", "bonferroni_level")},
                C_T3={k: get(uv, "C_T3", k) for k in ("gate", "G41", "L_star", "word", "conjugation", "t4b", "cplane", "gate_K1")},
                C_T4=dict(cells=t4, w4=get(uv, "C_T4", "w4")),
                C_T5={k: {x: v.get(x) for x in ("low_dimensional_search", "re_routing", "verdict_col", "verdict_z1", "control_col", "control_z1", "under_plan_text", "dm_last")}
                         | dict(arms={a: dict(A_op=get(r, "A_op_t"), rstar_seed=r.get("rstar_seed")) for a, r in (v.get("arms") or {}).items()})
                      for k, v in (get(uv, "C_T5", "widths") or {}).items()},
                C_T6={k: get(uv, "C_T6", k) for k in ("modes", "caps")},
                labels={k: get(uv, k, "labels") for k in ("C_T1", "C_T4", "C_T5") if get(uv, k, "labels")})


# ======================================================================================================================================
# Deviations (section 14): amendments, packing, parts 'not run', flags, labels, readings of the implementers
# ======================================================================================================================================
IMPLEMENTER8_READINGS = (
    "C-Q1: z of the XOR plant is balanced within each class of y (not per prompt), so that the class means are equal and z is not one of the permutation rows",
    "C-Q1: the degree-2 probe's 'ridge 1' is added to the covariance of the standardised features (the plan's ridge convention (Sigma + r I)); the 200 permutations refit only the probe (one solve), the eraser and the PCs stay those of the true labels",
    "C-Q1: plant scale a = d'_T x token SD along the plant direction / 2 (overall token SD of the unplanted sea states); u_T = R1.state[14]['axes']['u_T']",
    "C-Q1: INLP r* per direction (A -> B, B -> A), the larger one as summary; AUC_cf halves: one default_rng(25), fold True's permutation first",
    "C-Q1: linear rung needs g_0 > 0 (else 'not decidable (no baseline gap)'); no pattern claim if G41 explicitly fails (literal 'anywhere in Part C')",
    "C-Q4b / arm A: the loop rule is applied as an ordered list (un-searched, survives, re-routed, undecided (reference does not re-find RSI), undecided); r2_unlearn_qwen's in-session view tests 'gone' first",
    "C-Q4a: 'an ablation passed C-Q3' = r2_ana's C-Q3 verdict of the ablation that ran (the in-session choice governs what ran, OD-G8)",
    "C-Q6: the helix rule is evaluated per block, block 14 primary; self-repair at the r* cap over all 24 objects with every validated readout (18, 22); r* per relation",
    "C-T0 .. C-T7: rules recomputed from the stage JSONs and compared with the producers' previews; quorum '>= 4 of 5' = ceil(0.8 n); C-T3 j* = contiguous reading; C-T5 '> 24' counts as 25; C-T2 Bonferroni level fixed at 98.33 %",
    "C-T7: the family holds both erasure layers of the relaxation contrast (14 items per G-PC width), as r2_pc_toy records them",
    "A6: rel(d_n) is held at its round-1 value in the bootstrap of A; one draw of prompt weights for all blocks and loops; g_b / s_b nulls: 2,000 sign flips (seed 1107 + block)",
    "A7 dynamics: the damage condition (OD-A5) is applied per rule to the loops the rule reads; a control loop above 0.15 nats makes the self-loop rule 'not decidable'; the control exclusion ('generic attractor') also applies to the mirror",
    "A7: BA's 'C0 for IMP-CHG' uses the simultaneous lower bound (the pointwise one is reported); 'falls' (V3) mirrors 'present at >= 2 top-third blocks' to the bottom third; robustness = the point-estimate form of GI and (IS or BA) in the whitened metric and every leave-one-out",
    "A7: the RSI-into-RGI series are the frames-pooled rho_S and sigma_self; 'absorbed' / 'self-specific' per block need both frames",
    "A7 V5: if GI and (IS or BA) hold at some block but nowhere in the whitened metric and under every leave-one-out, the verdict is 'V5: no RGI geometry' with that statement (neither of the plan's two sub-labels applies)",
    "A7 pull and delta: a block enters with G-blk(b), GA(b) and RSI_a's own counted pushes within 0.15 nats (OD-A5: 'the loop read'); with the 0.6B arm run but r_DiD not computable, V1's r_DiD condition counts as not met (V2)",
    "A7 causal line: 'with simultaneous intervals' = sup-t 95 % intervals over the two statistics D(FG_causal) and D(rho_causal), from r2_rgi_stats.py's per-draw series (rgi/geom_draws); the verdict with the pointwise interval is reported next to it and is used, flagged, if the draws file is missing",
    "A3: w = mean of the two IMP-DES half gradients per model object (r2_rgi_stats.spill_inputs)",
    "AT8: ABSENT / present levels and the recursion rule use per-world means of the level-1 models over the valid worlds; recursion at the natural condition (pooling: the natural-condition entry; the all-conditions pooling is reported)",
    "AT8: a line 'not testable' (G39) blocks every answer except 'no RGI geometry'; 'not delta' may hold per line by either clause; family S-gain over the worlds whose B net passes level 1",
    "AT8: 'perturbation-driven' is read from the reduced model for every line (also a regulariser-specific one); a line's 'verdict' is 'ABSENT' where the ABSENT condition holds, the two-step verdict stays in 'rule_verdict'",
    "Amendment 2 (post hoc): Part C verdicts use the damage-matched controls as the producers ran them (the amended OD-C2); 'under_plan_text' gives the frozen text's reading next to every affected result (control unmatched: specificity 'not testable', G45; 'unlearns' / 'causal' False where the accuracy half fails, else not computable because the plan-text control was not run; the C-T4 ladder and C-T7's Holm family are evaluated with these three values)",
    "C-T4 ladder: if the span block unlearns and its isotropic control neither unlearns nor keeps 0.9 x base, the rung is 'undecided (...)'; in W4 'specific' is the retain criterion alone",
    "Amendment 3 (a) (post hoc): G16 uses armA.json's reference ci90 (the recentred interval) as run; the frozen text's reading is G16 with the stored unshifted percentile interval (ci90_percentile): where it would have stopped arm A the plan-text verdict is 'translation (as predicted), not un-searched', where it would have continued but the session stopped, rounds 2-3 are 'not computable'",
    "Amendment 3 (b), (c) (clarifications, not post hoc): the report shows the share of undefined replicates next to every theta rule output that carries one and lists every verdict made 'undecided' by an undefined bound; R4 / R3 are read as r2_theta_verdict.py writes them",
    "AT7: NW = the number of valid worlds, ninit 3, seed 1, pjit 0.03 (reference); with fewer than 2 valid worlds the noise is measured on all worlds (flagged)",
    "AT5: slopes over the BP models of r2_t1 plus the PC models of the G-PC widths (receiver means from the pairs with testable nets)",
    "G46: every retrained cell is compared per round-2 source; first95 None on both sides counts as agreement; weights are not compared; r2_rgi_toy's r1_job_name is not used (the job name is derived from the net name)",
    "JSON outputs are strict: NaN -> null, +-inf -> 'inf' / '-inf'",
)


def amendments(repo=None):
    """The amendments file of the checkout: sha256, every 'Amendment n (<time>)' with its post-hoc declaration (section 14)."""
    repo = repo or os.environ.get("R2_REPO") or REPO; p = os.path.join(repo, AMEND_REL)
    if not os.path.isfile(p): return dict(path=AMEND_REL, present=False, sha256=None, amendments=[])
    txt = open(p, encoding="utf-8", errors="replace").read(); parts = re.split(r"(?m)^## +(Amendment +\d+[^\n]*)\n", txt); out = []
    for i in range(1, len(parts) - 1, 2):
        body = parts[i + 1]; low = " ".join(body.lower().split())
        ph = bool(re.search(r"declared post hoc|marked 'post hoc'|marked post hoc", low)) or ("post hoc" in low and "not post hoc" not in low)
        secs = [(m.group(1).strip().rstrip("."), " ".join(m.group(2).split())) for m in re.finditer(r"\*\*((?:Change|Clarification)[^*]*)\*\*(.*?)(?=\*\*|\Z)", body, re.S)]
        stm = [" ".join(x.split()) for x in re.split(r"(?<=[.;])\s+", body) if "post hoc" in x.lower()]
        out.append(dict(title=parts[i].strip(), post_hoc=ph, change=secs[0][1][:900] if secs else None, changes=[dict(heading=h, text=t[:1500]) for h, t in secs],
                        post_hoc_statement=" ".join(stm)[:600] if stm else None,
                        consequence="affected results are reported both under the plan text and under the amendment (section 14)" if ph else None))
    return dict(path=AMEND_REL, present=True, sha256=sha256_file(p), n_amendments=len(out), amendments=out)


def theta_undefined(tv):
    """Amendment 3 (b): every theta rule output that carries 'nonfinite_share' (the share of undefined bootstrap replicates; a bound is
    undefined when the point value or more than 5 % of the replicates are), with its verdict and 'bounds_undefined'. Read defensively:
    the share may be a number, a list (per series point) or a dict (per input). Returns [dict(where, verdict, share_max, share,
    bounds_undefined)]; 'undecided_for_undefined_bounds' = the entries whose bounds are undefined."""
    out = []

    def mx(v):
        xs = [num(x) for x in (v.values() if isinstance(v, dict) else v if isinstance(v, list) else [v])]
        xs = [x for x in xs if fin(x)]; return max(xs) if xs else None

    def anyt(v):
        return bool(any(bool(x) for x in (v.values() if isinstance(v, dict) else v)) if isinstance(v, (list, dict)) else bool(v))

    def walk(o, path, depth=0):
        if depth > 9: return
        if isinstance(o, dict):
            if "nonfinite_share" in o or "bounds_undefined" in o:
                out.append(dict(where=path, verdict=o.get("verdict"), share_max=mx(o.get("nonfinite_share")), share=o.get("nonfinite_share"),
                                bounds_undefined=anyt(o.get("bounds_undefined")) if "bounds_undefined" in o else None))
            for k, v in o.items():
                if isinstance(v, (dict, list)): walk(v, f"{path}/{k}" if path else str(k), depth + 1)
        elif isinstance(o, list):
            for j, v in enumerate(o): walk(v, f"{path}[{j}]", depth + 1)
    walk(tv or {}, "")
    return out


def notes_deviations(folder=HERE):
    """The 'Deviations' sections of the implementers' NOTES_w*.md (text, with the file's sha256)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(glob.escape(folder), "NOTES_w*.md"))):
        txt = open(p, encoding="utf-8", errors="replace").read(); lines = txt.splitlines(); secs = []
        other = []
        for i, ln in enumerate(lines):
            m = re.match(r"^(#+)\s+(.*)$", ln)
            if not m: continue
            lvl = len(m.group(1)); j = i + 1
            while j < len(lines) and not (re.match(r"^(#+)\s", lines[j]) and len(re.match(r"^(#+)", lines[j]).group(1)) <= lvl): j += 1
            if re.search(r"[Dd]eviation|[Aa]mendment", m.group(2)): secs.append(dict(heading=m.group(2).strip(), text="\n".join(lines[i + 1:j]).strip()[:6000]))
            elif re.search(r"[Rr]eading|[Aa]mbiguit|[Oo]pen point", m.group(2)): other.append(dict(heading=m.group(2).strip(), lines=j - i - 1))
        out[os.path.basename(p)] = dict(sha256=sha256_file(p), deviations=secs, readings_and_open_points=other)
    return out


def session_records(inp):
    """session_done / session_partial / session_attempts / manifests of every session folder under <r2_root> (and of r2_ana next to
    <out.json>)."""
    out = {}
    for s in SESSIONS:
        dirs = [d for d in ([os.path.join(inp.r2, s)] if inp.r2 else []) if os.path.isdir(d)]
        if s == "r2_ana" and inp.out_dir: dirs.append(inp.out_dir)
        rec = dict(present=bool(dirs), manifests={}, session_done=None, session_partial=None, session_attempts=None, start_check=None)
        for d in dirs:
            for p in sorted(glob.glob(os.path.join(glob.escape(d), "manifest_*.json"))):
                js = load_json(p)
                if js is not None: rec["manifests"][os.path.basename(p)] = js
            for k in ("session_done", "session_partial", "session_attempts", "start_check"):
                js = load_json(os.path.join(d, k + ".json"))
                if js is not None and rec[k] is None: rec[k] = js
        em = os.environ.get("R2_MANIFEST")
        if s == "r2_ana" and em and os.path.isfile(em) and os.path.basename(em) not in rec["manifests"]:
            js = load_json(em)
            if js is not None: rec["manifests"][os.path.basename(em)] = js
        out[s] = rec
    return out


def packing(sess):
    """Packing changes and continuation sessions (7.4), parts 'not run' and declarations, from the runner's records."""
    out = {}
    for s, r in sess.items():
        sd = r.get("session_done") or r.get("session_partial") or {}; att = r.get("session_attempts") or sd.get("attempts") or []
        names = [a.get("attempt") for a in att if isinstance(a, dict)]
        decl = sorted({x for m in r["manifests"].values() for x in (get(m, "declarations", "R2_DECLARED_NOT_RUN") or [])})
        out[s] = dict(present=r["present"], done=r.get("session_done") is not None, partial=r.get("session_partial") is not None, attempts=names,
                      continuation_sessions=[n for n in names if n and n != s], packing=sd.get("packing"), not_run_steps=sd.get("not_run") or {},
                      time_guard_stops=sum(1 for a in att if isinstance(a, dict) and a.get("result") == "time guard"), declared_not_run_upstream=decl,
                      test_run=sd.get("test_run"), selftest=sd.get("selftest"), code_commit=sd.get("code_commit"))
    return out


def merged_manifest(inp, sess, own_outputs):
    """r2_manifest.json: every session manifest and runner record, the union of the copied round-1 files with a consistency check (the
    same path with different sha256 in two sessions is flagged), the freeze record, and the sha256 of the r2_ana outputs."""
    r1_files, flags = {}, []
    for s, r in sess.items():
        for mn, m in r["manifests"].items():
            for c in m.get("copied") or []:
                if not isinstance(c, dict) or c.get("kind") != "r1": continue
                e = r1_files.setdefault(c.get("path"), dict(sha256={}, source=c.get("source"), drive_mtime=c.get("drive_mtime"), size=c.get("size")))
                e["sha256"].setdefault(c.get("sha256"), []).append(f"{s}/{mn}")
    for p, e in r1_files.items():
        if len(e["sha256"]) > 1: flags.append(f"round-1 file {p} was copied with different sha256 in different sessions")
    first = next((m for r in sess.values() for m in r["manifests"].values()), {})
    plan_p = os.path.join(os.environ.get("R2_REPO") or REPO, PLAN_REL)
    return dict(written=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), plan=PLAN_REL, plan_sha256_frozen=PLAN_SHA256,
                plan_sha256_checkout=sha256_file(plan_p) if os.path.isfile(plan_p) else None, freeze=first.get("freeze"), amendments=amendments(),
                sessions={s: dict(present=r["present"], manifests=r["manifests"], session_done=r["session_done"], session_partial=r["session_partial"],
                                  session_attempts=r["session_attempts"], start_check=r["start_check"]) for s, r in sess.items()},
                round1_files=r1_files, n_round1_files=len(r1_files), flags=flags, r2_ana_outputs=own_outputs,
                note="merged from <r2_root>/<session>/manifest_*.json, session_done.json, session_attempts.json and the r2_ana manifest; r2_ana's session_done.json is written after this step")


# ======================================================================================================================================
# The gate table (section 6)
# ======================================================================================================================================
def gate_table(inp, info, pcg, sess, files, repro, d34, c41):
    """G1-G49 as far as r2_ana decides or can record them (each: status / value, where it was decided, the consequence that applies)."""
    q1, tv, rv, uv, ttv, t5, axj, q4b, armA = (files.get(k) for k in ("q1", "theta_verdict", "rgi_verdict", "unlearn_verdict", "theta_toy_verdict", "t5", "axes", "q4b", "armA"))
    mans = {s: list(r["manifests"].values()) for s, r in sess.items()}
    mg = lambda g: {s: [get(m, "gates", g) for m in ms] for s, ms in mans.items() if ms}
    plan_p = os.path.join(os.environ.get("R2_REPO") or REPO, PLAN_REL); sha = sha256_file(plan_p) if os.path.isfile(plan_p) else None
    G = {}
    G["G1"] = dict(name="plan freeze", checkout_sha256=sha, ok_here=sha == PLAN_SHA256 if sha else None, sessions={s: [get(m, "plan", "ok") for m in ms] for s, ms in mans.items() if ms},
                   freeze_commit=next((get(m, "freeze", "commit") for ms in mans.values() for m in ms), None), freeze_time=next((get(m, "freeze", "time") for ms in mans.values() for m in ms), None),
                   freeze_before_first_summary={s: [get(m, "drive_metadata", "freeze_before_first_summary") for m in ms] for s, ms in mans.items() if ms})
    G["G2"] = dict(name="GPU type", sessions=mg("G2"))
    G["G3"] = dict(name="R1 code freeze", labels=sorted({str(l) for ms in mans.values() for m in ms for l in (get(m, "gates", "G3", "labels") or [])}))
    G["G4"] = dict(name="R1 Qwen finality", final=info.get("final"), sessions=mg("G4"),
                   consequence={b: NOT_RUN_R1 for b, f in (info.get("final") or {}).items() if not f} or None)
    G["G5"] = dict(name="R1 PC finality", sessions=mg("G5"))
    G["G6"] = dict(name="R1 output present", not_run=[x for part in (rv, uv, tv, q1) if part for x in collect_not_run(part) if "round-1" in x[1]])
    G["G7"] = info.get("G7"); G["G8"] = info.get("G8"); G["G9"] = dict(info.get("G9") or {}, order=info.get("order"), delta=info.get("delta"))
    G["G10"] = dict(name="push scale of a block", blocks={b: dict(alpha=get(v, "alpha"), suffix=get(v, "suffix")) for b, v in (get(rv, "A6", "blocks") or {}).items()})
    G["G11"] = dict(info.get("G_desc") or {}, name="G-desc"); G["G12"] = dict(name="G-blk(b)", blocks=info.get("G_blk"))
    G["G13"] = dict(name="G-likely", LIKELY_wh=info.get("LIKELY_wh"), from_would_have_read=info.get("from_would_have_read"), V_RSI=info.get("V_RSI"))
    G["G14"] = dict(name="G-attr", G_attr=info.get("G_attr"), attracted_loop=info.get("attracted_loop"), rnd_last_rows=info.get("rnd_last_rows"))
    G["G15"] = dict(name="C-Q4b twin-start check", in_session=get(q4b, "gate", "G15"), r2_ana=get(uv, "C_Q4b", "gate", "G15"))
    G["G16"] = dict(name="arm A continuation", **(get(uv, "C_Q4_armA", "G16") or dict(in_session=get(armA, "continues"))))
    tb = (tv or {}).get("blocks") or {}
    G["G17"] = dict(name="m*^2 guard", blocks={b: get(v, "B_Q0", "m_star2") for b, v in tb.items() if isinstance(v, dict)}, modes={b: get(v, "mode") for b, v in tb.items() if isinstance(v, dict)})
    G["G18"] = dict(name="cross-fit switch", blocks={b: {k: get(v, "B_Q0", "cross_fit", k) for k in ("switch", "reason", "cos_AB", "diff")} for b, v in tb.items() if isinstance(v, dict)})
    G["G19"] = dict(get(tv, "gates", "G19_gate_p") or {}, name="theta gate p (label only)"); G["G20"] = dict(get(tv, "gates", "G20") or {}, name="theta code checks")
    for g, k in (("G21", "reserve1"), ("G22", "reserve2"), ("G23", "reserve3"), ("G25", "reserve5")):
        G[g] = dict(name=f"theta {k}", cells={n: v for n, v in ((tv or {}).get("reserves") or {}).items() if n.startswith(k)})
    G["G24"] = dict(name="theta reserve 4 (power step)", blocks={b: get(v, "B_Q6", "status") for b, v in tb.items() if isinstance(v, dict)})
    G["G26"] = dict(name="K defined (R6)", cells={b: {c: get(x, "R6", "defined") for c, x in (get(v, "R6", "cells") or {}).items()} for b, v in tb.items() if isinstance(v, dict) and get(v, "R6", "cells")})
    G["G27"] = dict(get(ttv, "T5", "G27") or get(t5, "G27") or {}, name="theta T5 pilot")
    G["G28"] = dict(name="GA(b)", GA=get(axj, "GA"), auc_RGI={b: get(v, "auc_RGI") for b, v in (get(axj, "blocks") or {}).items()}, threshold=get(axj, "ga_threshold"),
                    gen_loops={b: get(v, "status") for b, v in (get(rv, "A6", "gen") or {}).items()})
    gm17 = files.get("geom_meta")
    G["G29"] = dict(name="A1 length asserts", a1=get(gm17, "stages", "a1", "g29") or get(gm17, "stages", "a1", "G29") or ("ran" if get(gm17, "stages", "a1") else None))
    G["G30"] = dict(get(rv, "A7", "gates", "G30") or {}, name="passing set P"); G["G31"] = dict(get(rv, "A7", "gates", "G31") or {}, name="separability")
    G["G32"] = dict(get(rv, "A7", "gates", "G32") or {}, name="composed gate (A3)"); G["G33"] = dict(get(rv, "A7", "gates", "G33") or {}, name="Qwen3-0.6B arm")
    G["G34"] = dict(d34 or {}, name="dn_delta reproduction")
    G["G35"] = dict(name="G-meta", value=pcg.get("G_meta"), source=pcg.get("G_meta_source"), parts=pcg.get("G_meta_parts"), fallback=pcg.get("G35_fallback"), labels=pcg.get("labels"))
    G["G36"] = dict(name="G-PC(W)", widths=pcg.get("G_PC"), usable=pcg.get("widths_G_PC"))
    G["G37"] = dict(name="R1 PC and BP models present", inventory=pcg.get("inventory_G37"), t7={W: dict(seeds_present=v.get("seeds_present"), in_family=v.get("in_family")) for W, v in (get(uv, "C_T7", "per_width") or {}).items()})
    G["G38"] = dict(get(rv, "AT8", "G38") or {}, name="RGI toy validity"); G["G39"] = dict(get(rv, "AT8", "G39") or {}, name="RGI toy machinery")
    G["G40"] = dict(get(rv, "AT8", "family", "G40") or {}, name="ReLU calibration (AT4)"); G["G41"] = dict(c41 or {}, name="C-T3 circle gate")
    G["G42"] = dict((q1 or {}).get("G42") or {}, name="C-Q1 calibration"); G["G43"] = dict(get(uv, "C_Q3", "G43") or {}, name="C-Q3 pass -> C-Q4a ablation")
    G["G44"] = dict(get(uv, "C_Q4b", "gate") or {}, name="C-Q4b gate")
    g45 = [x for x in collect_values(uv, ("specificity", "verdict_z1", "verdict_col")) if "damage-matched" in x[1] and not x[0].startswith("amendment_2")]
    G["G45"] = dict(name="damage-matched control", not_testable=[x for x in g45 if "under_plan_text" not in x[0]],
                    not_testable_under_plan_text=[x for x in g45 if "under_plan_text" in x[0]], amendment_2_affected=get(uv, "amendment_2", "n_affected"),
                    note="'not_testable': with the controls as run (Amendment 2); 'not_testable_under_plan_text': results whose control is unmatched under the frozen OD-C2")
    G["G46"] = dict(name="reproduction of retrained BP nets", compared=get(repro, "compared"), differing=get(repro, "differing"), not_compared=get(repro, "not_compared"))
    G["G47"] = dict(name="versions for common random numbers", session=get(axj, "versions"), r1_gpu1=get(axj, "r1_meta_gpu1"), match_rate=get(axj, "match_rate"), manifests=mg("G47"))
    G["G48"] = dict(name="session time guard", time_guard_stops={s: v["time_guard_stops"] for s, v in packing(sess).items() if v["present"]},
                    continuation_sessions={s: v["continuation_sessions"] for s, v in packing(sess).items() if v["continuation_sessions"]})
    G["G49"] = dict(name="second RSI loop at the extra blocks", independent_loop={b: get(v, "B_Q0", "independent_loop") for b, v in tb.items() if isinstance(v, dict)})
    return G


def collect_values(obj, keys, path="", out=None, depth=0):
    out = [] if out is None else out
    if depth > 9: return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}/{k}" if path else str(k)
            if k in keys and isinstance(v, str): out.append((path, v))
            elif isinstance(v, (dict, list)): collect_values(v, keys, p, out, depth + 1)
    return out


# ======================================================================================================================================
# Driver
# ======================================================================================================================================
def pc_inventory(r1_dir):
    """G37: presence of the round-1 PC-muP (W 64-512) and BP-muP (W 128, 256) models, seeds 0-4; expected jobs missing from PCS; the
    pc_missing lists of the shards."""
    inv = {}
    if not r1_dir or not os.path.isdir(r1_dir): return dict(inventory={}, missing_jobs=None, pc_missing={})
    for arm, widths in (("pc_mup", (64, 128, 256, 512)), ("bp_mup", (128, 256))):
        for W in widths: inv[f"{arm}_W{W}"] = {str(s): r1_pc_job(r1_dir, job_name(arm, W, s), "models") is not None for s in range(5)}
    exp = set()
    for s in _shards(r1_dir): exp |= set(load_json(os.path.join(r1_dir, s, f"pc_expected_{s}.json")) or [])
    have = {r.get("name") for r in pcs_rows(r1_dir)}
    miss = {s: load_json(os.path.join(r1_dir, s, f"pc_missing_{s}.json")) for s in _shards(r1_dir) if os.path.isfile(os.path.join(r1_dir, s, f"pc_missing_{s}.json"))}
    return dict(inventory=inv, missing_jobs=sorted(exp - have) if exp else None, pc_missing=miss)


def run(r1_dir, r2_root, out_path):
    t0 = time.time(); inp = Inputs(r1_dir, r2_root, out_path); info = r1_info(r1_dir); pcg = pc_gates(r1_dir, inp); out_dir = inp.out_dir
    pinv = pc_inventory(r1_dir); pcg["inventory_G37"] = pinv["inventory"]
    gates_path = os.path.join(out_dir, "pc_toy", "gates.json")
    write_json(gates_path, dict(script="r2_report.py", plan_sha256=PLAN_SHA256, G=pcg.get("G"), G_meta=pcg.get("G_meta"), G_meta_source=pcg.get("G_meta_source"),
                                G_meta_parts=pcg.get("G_meta_parts"), G35_fallback_needed=pcg.get("G35_fallback_needed"), G35_fallback=pcg.get("G35_fallback"), G_PC=pcg.get("G_PC"),
                                widths_G_PC=pcg.get("widths_G_PC"), labels=pcg.get("labels"), n_rows=pcg.get("n_rows"), shards=pcg.get("shards"), inventory_G37=pinv["inventory"],
                                missing_jobs=pinv["missing_jobs"], pc_missing=pinv["pc_missing"], finder=pcg.get("finder"), error=pcg.get("error"),
                                note="G-meta = G1 and G2 of pc_analyze.gates on PCS (G35 fallback: r2_t3's round-2 BP-muP W 128 nets); G-PC(W) = G-meta, G0_pc_mup, G3, G4, G5, >= 4 seeds level 1 (G36)"))
    files = dict(q1=inp.json("unlearn/q1.json"), theta_verdict=inp.json("theta_verdict.json"), rgi_verdict=inp.json("rgi_verdict.json"), unlearn_verdict=inp.json("unlearn_verdict.json"),
                 theta_toy_verdict=inp.json("theta_toy/verdict.json"), t5=inp.json("theta_toy/t5.json"), axes=inp.json("axes/r2_axes.json"), q4b=inp.json("unlearn/q4b.json"),
                 armA=inp.json("unlearn/armA.json"), geom_meta=inp.json("rgi/geom_meta_qwen3-1.7b.json"))
    repro = g46(r1_dir, inp); d34 = g34(r1_dir, inp); c41 = g41(inp); sess = session_records(inp)
    G = gate_table(inp, info, pcg, sess, files, repro, d34, c41)
    A = part_A(files["rgi_verdict"], info, pcg); B = part_B(files["theta_verdict"], files["theta_toy_verdict"], files["t5"], pcg); C = part_C(files["unlearn_verdict"], files["q1"], pcg)
    pk = packing(sess); am = amendments()
    labels = dict(A=A.get("labels"), B=get(B, "qwen", "labels"), C=C.get("labels"))
    notrun = dict(A=A.get("not_run"), B=get(B, "qwen", "not_run"), C=C.get("not_run"),
                  sessions={s: v["not_run_steps"] for s, v in pk.items() if v["not_run_steps"]},
                  sessions_absent=[s for s, v in pk.items() if not v["present"] and s != "r2_ana"],
                  verdict_files_missing=[k for k in ("q1", "theta_verdict", "rgi_verdict", "unlearn_verdict") if files[k] is None])
    od = dict(C=get(C, "od_g8"), B={b: get(v, "od_g8", "disagreements") for b, v in (get(B, "qwen", "blocks") or {}).items() if get(v, "od_g8", "disagreements")})
    dev = dict(amendments=am, packing=pk, not_run=notrun, labels=labels, flags=dict(A=A.get("flags"), blocks=info.get("block_flags"), G3=get(G, "G3", "labels"), G34=d34.get("flag"),
                                                                                  G46=repro.get("differing"), pc_gates=pcg.get("labels")),
               amendment_results={"Amendment 2 (post hoc; OD-C2)": dict(n_affected=get(C, "amendment_2", "n_affected"), affected=get(C, "amendment_2", "affected")),
                                  "Amendment 3 (a) (post hoc; G16)": C.get("amendment_3"),
                                  "Amendment 3 (b) (clarification; undefined theta bounds)": get(B, "verdicts", "undecided_for_undefined_bounds"),
                                  "Amendment 3 (c) (clarification; R4 (i) 'degenerate')": {b: get(v, "R4", "verdict") for b, v in (get(B, "qwen", "blocks") or {}).items()
                                                                                          if "degenerate" in str(get(v, "R4", "verdict"))}},
               od_g8_disagreements=od, readings_implementer_8=list(IMPLEMENTER8_READINGS), notes=notes_deviations(),
               note="results are reported against the frozen plan text (section 14); every deviation is listed here: amendments (post hoc ones marked), packing changes and continuation "
                    "sessions, parts 'not run', flags, labels, OD-G8 disagreements, and the implementers' readings and stated deviations (NOTES_w*.md)")
    summary = dict(script="r2_report.py", plan=PLAN_REL, plan_sha256=PLAN_SHA256, written=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                   blocks=dict(B3=info.get("B3"), b_lo=info.get("b_lo"), b_hi=info.get("b_hi"), order_by_R1_delta=info.get("order"), delta=info.get("delta"), final=info.get("final")),
                   parts=dict(A=A, B=B, C=C), gates=G, reproduction=dict(G46=repro, G34=d34, B_T1_cells={k: v for k, v in repro["cells"].items() if k.startswith("B-T1")}),
                   deviations=dev, inputs=inp.record(["unlearn/q1.json", "theta_verdict.json", "rgi_verdict.json", "unlearn_verdict.json", "theta_toy/verdict.json", "axes/r2_axes.json"]),
                   finder=inp.log, r1_finder=info.get("finder"), seconds=time.time() - t0)
    write_json(out_path, summary)
    own = {}
    for rel in ("unlearn/q1.json", "theta_verdict.json", "rgi_verdict.json", "unlearn_verdict.json", "round2_summary.json", "pc_toy/gates.json"):
        p = out_path if rel == "round2_summary.json" else (os.path.join(out_dir, rel) if os.path.isfile(os.path.join(out_dir, rel)) else inp.find(rel))
        if p and os.path.isfile(p): own[rel] = dict(sha256=sha256_file(p), bytes=os.path.getsize(p), path=inp.short(p))
    write_json(os.path.join(out_dir, "r2_manifest.json"), merged_manifest(inp, sess, own))
    log(f"report -> {out_path}; gates -> {gates_path}; manifest -> {os.path.join(out_dir, 'r2_manifest.json')} ({time.time() - t0:.0f}s)")
    for part, v in (("A", A.get("verdicts")), ("B", get(B, "verdicts")), ("C", C.get("verdicts"))):
        log(f"part {part}: {json.dumps(clean(v), default=str)[:1500]}")
    return summary


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if len(a) >= 2 and a[0] == "selftest": return 0 if selftest(a[1]) else 1
    if len(a) != 3: print(__doc__); return 2
    run(*a); return 0


# ======================================================================================================================================
# Code tests: a synthetic round-1 tree (no model, nothing from a GPU run) and the selftest
# ======================================================================================================================================
def synth_r1(r1_dir, D=24, NP=16, b_lo=8, b_hi=20, seed=0, K=400, plants=None, damages=None, summaries=None, delta=None, sigma=1.0, noise=1.0,
             suffix="", rnd_rows=None, skip_blocks=()):
    """A synthetic round-1 tree in the round-1 layout for code tests (tensors drawn here; no model): gpu1/rk/b14, gpu2/rk/b<b_lo>,
    gpu3/rk/b<b_hi> with state.pt (axes, ref, alpha, ev_<L>_<n>, dir_<L>_<n>) and loops.json (rows by rk.stats's formulas on u_T_perp,
    '_meta'), summary_b<b>.json, dn_delta.json and blocks_chosen.json in gpu2 and gpu3, shard_done.json.
    plants {(b, L, n): vector [D]}: added to every pair's delta of the evaluation behind d_n (n = 1..4); damages {(b, L, row)}:
    damage_of_push of loops row 0..3 (default 0.03); summaries {b: summary dict or None (= no summary file: block not final)}; delta
    {block: last-token delta}; rnd_rows {L: dict(p, A)}: overrides of the last RND row at block 14 (G14). Returns {b: dict of the block's
    axes (float64) and Q, the orthonormal basis they were taken from}."""
    import torch
    import r2_rk as R
    plants, damages, summaries, delta = plants or {}, damages or {}, dict(summaries or {}), delta or {}
    shard = {14: "gpu1", b_lo: "gpu2", b_hi: "gpu3"}; info = {}
    for b in (14, b_lo, b_hi):
        if b in skip_blocks: continue
        g = torch.Generator().manual_seed(1000 * seed + b); rnd = lambda *sh: torch.randn(*sh, generator=g, dtype=torch.float64)
        Q, _ = torch.linalg.qr(rnd(D, 8)); ax = dict(u_T_perp=Q[:, 0].clone(), sea=Q[:, 1].clone(), sports=Q[:, 2].clone(), tone=Q[:, 3].clone(), u_neg=Q[:, 4].clone())
        fold = torch.zeros(NP, dtype=torch.bool); fold[torch.randperm(NP, generator=g)[: NP // 2]] = True
        axes = dict(ax, u_T=ax["u_T_perp"] + 0.3 * ax["sea"], u_S=Q[:, 6].clone(), u_S_perp=Q[:, 6].clone(), fold=fold, gap_perp_cross=1.0, gap_perp=1.0,
                    gap_cross=dict(rsi=1.0, sports=1.0, tone=1.0, sea=1.0), auc=dict(rsi=0.95, sports=0.95, tone=0.95, sea=0.95),
                    **{f"null_{k}": (0.1 * rnd(K, D)).float() for k in ("u_T_perp", "sports", "sea", "tone")})
        ST = dict(axes=axes, ref=dict(X=rnd(NP, D).float(), cap=2.0, med=1.0, conts=[[1, 2, 3]] * NP), alpha=dict(alpha=1.0, damage=0.05, in_band=True))
        loops = dict(_meta=dict(alpha=1.0, alpha_scale=1.0, suffix=suffix, block=b))
        for L in R.loops_of_block(b):
            nseed = 2 if L.startswith("RSI") else 1; N = NP * nseed; po = torch.arange(NP).repeat_interleave(nseed); rows = []
            for n in range(4):
                dl = noise * rnd(N, D)
                if (b, L, n + 1) in plants: dl = dl + torch.as_tensor(plants[(b, L, n + 1)], dtype=torch.float64)[None]
                ev = dict(delta=dl.float(), ok=torch.ones(N, dtype=torch.bool), prompt_of=po, xp=dl.float(), xm=(-dl).float(), first_div=[0] * N,
                          damage_of_push=float(damages.get((b, L, n), 0.03)), alpha=1.0, conts=[], dropped=0, capped_share=0.0)
                ST[f"ev_{L}{suffix}_{n}"] = ev; ST[f"dir_{L}{suffix}_{n + 1}"] = dl.float().double().mean(0).float()
                r, d = R.stats_on_axis(ev, axes["u_T_perp"], axes["null_u_T_perp"], n_prompts=NP)
                r.update(damage_of_push=ev["damage_of_push"], damage_next=0.03, sigma=sigma, Ue=float(d @ axes["u_T_perp"]), first_div_median=0.0, identical_pairs=0, dropped=0)
                rows.append(r)
            if b == 14 and rnd_rows and L in rnd_rows: rows[3].update(rnd_rows[L])
            loops[L] = rows
        d_ = os.path.join(r1_dir, shard[b], "rk", f"b{b}"); os.makedirs(d_, exist_ok=True)
        torch.save(ST, os.path.join(d_, "state.pt")); write_json(os.path.join(d_, "loops.json"), loops)
        info[b] = dict({k: v.double() for k, v in ax.items()}, Q=Q, fold=fold)
    S = {14: dict(block=14, verdict=dict(verdict="likely, substantial RSI component", PC_PASS=True, label_attractor=False, gates=dict(failed=[], source="pilot.json"))),
         b_lo: dict(block=b_lo, verdict=dict(PC_PASS=True, gates=dict(failed=[], source="pilot.json"), enters_delta_rule=True)),
         b_hi: dict(block=b_hi, verdict=dict(PC_PASS=True, gates=dict(failed=[], source="pilot.json"), enters_delta_rule=True))}
    S.update(summaries)
    for b in (14, b_lo, b_hi):
        os.makedirs(os.path.join(r1_dir, shard[b]), exist_ok=True)
        if S.get(b) is not None and b not in skip_blocks: write_json(os.path.join(r1_dir, shard[b], f"summary_b{b}.json"), S[b])
        write_json(os.path.join(r1_dir, shard[b], "shard_done.json"), dict(shard=shard[b], gpu="SYNTHETIC", torch="test", transformers="test", code_commit="synthetic"))
    dd = {l: 0.12 + 0.004 * abs(l - 14) for l in range(28)}; dd.update({14: 0.09, b_lo: 0.10, b_hi: 0.20}); dd.update(delta)
    fd = dict(model="SYNTHETIC (code test)", blocks={str(l): {r: dict(F=0.5 + f * dd[l], floor=0.5, delta=f * dd[l]) for r, f in (("last", 1.0), ("mean", 0.9))} for l in range(28)})
    for s_ in ("gpu2", "gpu3"):
        write_json(os.path.join(r1_dir, s_, "dn_delta.json"), fd); write_json(os.path.join(r1_dir, s_, "blocks_chosen.json"), dict(b_lo=b_lo, b_hi=b_hi))
    return info


def synth_pcs(r1_dir, pc_widths=(128, 256), bp_widths=(128, 256), seeds=range(5), b2=0.3, level1=True, g4=True, g5=True, pc_level1=None, shard_pc="gpu3", shard_bp="gpu5",
              first95=1200, with_jobs=True):
    """Synthetic round-1 PC summaries (pc_summary.py rows) and job files for code tests: BP-muP and PC-muP main cells (K 12, gamma 0, wd 1).
    b2: B2(8) of the BP-muP nets (G2 needs mean >= 0.10 and lower bound > 0.03); g4 / g5: the PC convergence / distinctness gates."""
    rows = {shard_pc: [], shard_bp: []}
    for W in bp_widths:
        for s in seeds:
            nm = job_name("bp_mup", W, s); r = dict(name=nm, arm="bp_mup", W=W, seed=s, K=12, gamma=0.0, wd_scale=1.0, lr=1e-3, first95=first95 + 10 * s, checkpoint=first95 + 500,
                                                     level1=bool(level1), coord_check=[1.0, 1.0], B2_8=b2 + 0.01 * s, B2_4=b2 / 2, F8_mean=0.06, delta=dict(h2=dict(delta=0.4)))
            rows[shard_bp].append(r)
            if with_jobs:
                write_json(os.path.join(r1_dir, shard_bp, "pc", "jobs", nm + ".json"), dict(arm="bp_mup", W=W, seed=s, K=12, first95=r["first95"], level1=r["level1"],
                                                                                               finetune={"8": dict(B2=r["B2_8"])}, delta=dict(h2=dict(delta=0.4))))
    for W in pc_widths:
        for s in seeds:
            l1 = bool(level1 if pc_level1 is None else pc_level1.get((W, s), True))
            rows[shard_pc].append(dict(name=job_name("pc_mup", W, s), arm="pc_mup", W=W, seed=s, K=12, gamma=0.0, wd_scale=1.0, lr=1e-3, T=20, beta=0.1, first95=first95, level1=l1,
                                       coord_check=[1.0, 1.0], B2_8=b2, F8_mean=0.06,
                                       pc=dict(rose_share=0.0 if g4 else 0.2, ratio_median=0.01, ratio_le005_share=0.95, cos_l2_early_median=0.5 if g5 else 0.99)))
    for sh, rr in rows.items():
        if not rr: continue
        write_json(os.path.join(r1_dir, sh, f"pc_summary_{sh}.json"), rr); write_json(os.path.join(r1_dir, sh, f"pc_expected_{sh}.json"), [r["name"] for r in rr])
        write_json(os.path.join(r1_dir, sh, "shard_done.json"), dict(shard=sh, code_commit="synthetic"))
    return rows


def selftest(work):
    """Synthetic code test of the shared helpers and of the report: JSON cleaning, t intervals, the finder, the round-1 gates in every
    state (G7-G9, G-desc, G-blk, G-likely incl. would-have-read, G-attr, a block that is not final), the PC gates (G-meta true / false,
    G35's fallback, G-PC with G4 / G5 / level-1 failures, 'BP-equivalent'), G41, G46 (agreeing and differing cells, 4-seed rule), G34,
    the amendments parser, and an end-to-end report on synthetic verdict files with missing parts."""
    import shutil
    import torch
    import r2_rk as R
    R.MOCK = True                                                                # seeded Pool directions (no rk_dirs.pt needed)
    shutil.rmtree(work, ignore_errors=True); os.makedirs(work); res = []; t0 = time.time()

    def check(name, cond, info=""):
        res.append(dict(test=name, ok=bool(cond), info=str(info)[:300])); print(f"[{'PASS' if cond else 'FAIL'}] {name} {str(info)[:200]}", flush=True)
    c = clean(dict(a=float("nan"), b=float("inf"), c=np.float32(1.5), d=torch.tensor([1.0, float("-inf")]), e=(1, 2), f={3: np.bool_(True)}))
    check("clean: strict JSON (NaN -> null, inf -> 'inf', tensors, numpy, int keys)", c == dict(a=None, b="inf", c=1.5, d=[1.0, "-inf"], e=[1, 2], f={"3": True}) and json.dumps(c, allow_nan=False))
    t = tci([1.0, 2.0, 3.0, None, float("nan")]); from scipy import stats as _st
    check("tci: t interval with df n - 1, non-finite dropped", t["n"] == 3 and abs(t["hi"] - (2 + _st.t.ppf(0.975, 2) * 1 / math.sqrt(3))) < 1e-12 and tci([1.0])["lo"] is None and tci([])["mean"] is None)
    check("get / num / comparisons treat None and NaN as failing", get(dict(a=[dict(b=2)]), "a", 0, "b") == 2 and get({}, "x", default=5) == 5 and not ge(None, 0) and not lt(float("nan"), 1) and ge(2, 2) and quorum(5) == 4 and quorum(4) == 4 and quorum(2) == 2)
    # finder
    r2 = os.path.join(work, "r2"); outp = os.path.join(work, "ana", "round2_summary.json")
    for rel, sess_ in (("unlearn/q3.json", "r2_q1"), ("unlearn/q1.json", "r2_ana"), ("unlearn_toy/t5.json", "r2_t4"), ("unlearn_toy/t5.json", "r2_t3"), ("rgi/dyn_b14.json", "r2_q1"), ("rgi/geom_stats_qwen3-1.7b.json", "r2_q3")):
        write_json(os.path.join(r2, sess_, rel), dict(status="ran", where=sess_))
    write_json(os.path.join(work, "ana", "theta_verdict.json"), dict(status="ok", where="out_dir"))
    inp = Inputs(None, r2, outp)
    check("finder: expected session first, then any session, then the folder of <out.json>; missing -> None",
          inp.json("unlearn_toy/t5.json")["where"] == "r2_t4" and inp.json("unlearn/q1.json")["where"] == "r2_ana" and inp.json("rgi/dyn_b14.json")["where"] == "r2_q1"
          and inp.json("theta_verdict.json")["where"] == "out_dir" and inp.json("unlearn/q9.json") is None and len(inp.find_all("unlearn_toy/t*.json")) == 1 and inp.log[-1]["chosen"] is None)
    # round-1 gates
    r1 = os.path.join(work, "r1"); synth_r1(r1)
    i0 = r1_info(r1)
    check("r1_info: B3, delta order b(1) < b(2) < b(3), all final, gates open", i0["B3"] == [14, 8, 20] and i0["order"] == [14, 8, 20] and all(i0["final"].values()) and not i0["G_desc"]["failed"]
          and all(v["ok"] for v in i0["G_blk"].values()) and i0["LIKELY_wh"] and not i0["from_would_have_read"] and i0["G_attr"] is False and i0["G7"]["ok"] and i0["G8"]["ok"], i0["order"])
    check("desc_labels: only block 14 and only when a pilot gate failed", desc_labels(i0, 14) == [] and desc_labels(dict(G_desc=dict(failed=True)), 14) == [LBL_GDESC] and desc_labels(dict(G_desc=dict(failed=True)), 8) == [])
    r1b = os.path.join(work, "r1b")
    synth_r1(r1b, summaries={14: dict(verdict=dict(verdict="not decidable (pilot gate GO1 failed; descriptive only; would have read: likely, substantial RSI component)", PC_PASS=True,
                                                    label_attractor=True, gates=dict(failed=["GO1"]))), 8: dict(verdict=dict(PC_PASS=False, gates=dict(failed=[]))), 20: None},
             rnd_rows=dict(RND_2=dict(p=0.001, A=-0.4)), delta={8: 0.20001, 20: 0.2})
    write_json(os.path.join(r1b, "gpu3", "blocks_chosen.json"), dict(b_lo=8, b_hi=21))
    d3 = load_json(os.path.join(r1b, "gpu3", "dn_delta.json")); d3["blocks"]["5"]["last"]["delta"] += 1e-3; write_json(os.path.join(r1b, "gpu3", "dn_delta.json"), d3)
    i1 = r1_info(r1b)
    check("r1_info: G-desc, would-have-read LIKELY, G-blk false, block not final, G-attr with RND_2 attracted",
          i1["G_desc"]["failed"] and i1["LIKELY_wh"] and i1["from_would_have_read"] and i1["G_blk"]["14"]["ok"] is False and i1["G_blk"]["8"]["ok"] is False
          and i1["final"]["20"] is False and i1["G_blk"]["20"]["ok"] is None and i1["G_attr"] is True and i1["attracted_loop"] == "RND_2", (i1["G_blk"], i1["attracted_loop"]))
    check("r1_info: G7 inconsistent block choice, G8 delta copies differ, G9 tie broken by the round-1 label",
          i1["G7"]["ok"] is False and i1["G8"]["ok"] is False and i1["order"] == [14, 8, 20] and i1["G9"]["ties"], (i1["G7"]["label"], i1["G8"], i1["order"]))
    check("likely_wh: plain, would-have-read, not likely", likely_wh("likely (provisional)") == (True, False) and likely_wh("not decidable (pilot gate GO2 failed; would have read: unlikely)") == (False, False)
          and likely_wh("unlikely") == (False, False) and likely_wh(None) == (False, False))
    i2 = r1_info(os.path.join(work, "no_r1"))
    check("r1_info without round-1 files: nothing final, no crash", i2["final"] == {"14": False} and i2["G_desc"]["failed"] is None and i2["order"] is None)
    bank, ST, loops = load_bank(r1, 14)
    check("load_bank: six loops at block 14, d_n = mean delta of the evaluation behind it", sorted(k for k in bank if not k.startswith("_")) == sorted(R.LOOPS_14)
          and float((bank["RSI_a"]["d"][2] - ST["ev_RSI_a_1"]["delta"].double().mean(0)).abs().max()) < 1e-6 and bank["RSI_a"]["rel"][3] == loops["RSI_a"][2]["rel"])
    # PC gates
    rp = os.path.join(work, "r1pc"); synth_pcs(rp); g = pc_gates(rp, None)
    check("pc_gates: G-meta true, G-PC(128) and G-PC(256) usable", g["G_meta"] and g["G_meta_source"] == "PCS (round 1)" and g["widths_G_PC"] == [128, 256] and not g["labels"], g["G_PC"])
    rp2 = os.path.join(work, "r1pc2"); synth_pcs(rp2, b2=0.02); g2 = pc_gates(rp2, None)
    check("pc_gates: G2 fails -> G-meta false, label, no G-PC width", g2["G_meta"] is False and LBL_GMETA in g2["labels"] and g2["widths_G_PC"] == [] and g2["G_PC"]["128"]["status"] == "not testable")
    rp3 = os.path.join(work, "r1pc3"); synth_pcs(rp3, g5=False); g3 = pc_gates(rp3, None)
    check("pc_gates: only G5 fails -> 'BP-equivalent', not used for PC claims", g3["G_PC"]["128"]["bp_equivalent"] and g3["G_PC"]["128"]["status"].startswith("BP-equivalent") and g3["widths_G_PC"] == [])
    rp4 = os.path.join(work, "r1pc4"); synth_pcs(rp4, pc_level1={(128, 0): False, (128, 1): False}); g4_ = pc_gates(rp4, None)
    check("pc_gates: fewer than 4 level-1 seeds at W 128 -> not testable there, W 256 usable", g4_["widths_G_PC"] == [256] and g4_["G_PC"]["128"]["conds"]["level1_4_of_5"] is False)
    rp5 = os.path.join(work, "r1pc5"); synth_pcs(rp5, g4=False); g5_ = pc_gates(rp5, None)
    check("pc_gates: G4 fails -> not testable (not BP-equivalent)", g5_["widths_G_PC"] == [] and not g5_["G_PC"]["256"]["bp_equivalent"])
    rp6 = os.path.join(work, "r1pc6"); synth_pcs(rp6, bp_widths=(256,)); r2f = os.path.join(work, "r2f")
    write_json(os.path.join(r2f, "r2_t3", "unlearn_toy", "t1.json"), dict(cells=dict(bp_mup_W128=dict(base=dict(level1=[True] * 5, B2={"8": [0.3, 0.32, 0.28, 0.31, 0.29]})))))
    g6 = pc_gates(rp6, Inputs(rp6, r2f, None)); g6b = pc_gates(rp6, Inputs(rp6, os.path.join(work, "empty"), None))
    check("pc_gates: no BP-muP W 128 rows -> G35 fallback on r2_t3's nets ('gate from round 2'); without them not decidable",
          g6["G_meta"] and g6["G_meta_source"] == "gate from round 2" and "gate from round 2" in g6["labels"] and g6["widths_G_PC"] == [128, 256]
          and g6b["G_meta"] is False and g6b["G_meta_source"].startswith("not decidable"), (g6["G_meta_source"], g6b["G_meta_source"]))
    write_json(os.path.join(r2f, "r2_t3", "unlearn_toy", "t1.json"), dict(cells=dict(bp_mup_W128=dict(base=dict(level1=[True, True, True, False, False], B2={"8": [0.3] * 5})))))
    check("G35 fallback: fewer than 4 level-1 seeds -> G-meta false", pc_gates(rp6, Inputs(rp6, r2f, None))["G_meta"] is False)
    check("pcs_rows: duplicates resolved by pc_expected (OD-G9)", (lambda: (write_json(os.path.join(rp, "gpu6", "pc_summary_gpu6.json"), [dict(name=job_name("pc_mup", 128, 0), arm="pc_mup", W=128, seed=0, K=12, gamma=0.0, wd_scale=1.0, level1=False)]),
                                                                              [r for r in pcs_rows(rp) if r["name"] == job_name("pc_mup", 128, 0)][0]["level1"]))()[1] is True)
    # G41
    mk41 = lambda ring, pring: Inputs(None, (write_json(os.path.join(work, "g41", "r2_t3", "unlearn_toy", "t3.json"),
                                                         dict(gate=dict(K1=dict(ring=[dict(ws=i, zero_shot=v) for i, v in enumerate(ring)], pring=[dict(ws=i, zero_shot=v) for i, v in enumerate(pring)])))), os.path.join(work, "g41"))[1], None)
    check("G41: passes with >= 4 of 5 seeds, fails otherwise, 'not run' when missing", g41(mk41([0.99] * 5, [0.05] * 4 + [0.5]))["passes"] is True and g41(mk41([0.99] * 3 + [0.5, 0.5], [0.05] * 5))["passes"] is False
          and g41(Inputs(None, os.path.join(work, "empty"), None))["passes"] is None)
    # G46 and G34
    r2g = os.path.join(work, "r2g"); rg = os.path.join(work, "r1g"); synth_pcs(rg, bp_widths=(64, 128), pc_widths=())
    for s_ in range(4):
        for W in (64,): write_json(os.path.join(rg, "gpu5", "pc", "jobs", job_name("bp_mup", W, s_, 1) + ".json"), dict(first95=None, finetune={"8": dict(B2=0.05)}, delta=dict(h2=dict(delta=0.1))))
    nets = [dict(name=job_name("bp_mup", 64, s_), arm="bp_mup", W=64, seed=s_, K=12, gamma=0.0, wd_scale=1.0, first95=1200 + 10 * s_ + (900 if s_ == 3 else 0), B2_8=0.3 + 0.01 * s_, delta=dict(h2=dict(delta=0.41)), level1=True) for s_ in range(4)]
    nets += [dict(name=job_name("bp_mup", 64, s_, 1), arm="bp_mup", W=64, seed=s_, K=1, gamma=0.0, wd_scale=1.0, first95=None, B2_8=0.05, level1=False) for s_ in range(4)]
    nets += [dict(name=job_name("bp_mup", 64, s_, 12, 0.0, 0.0), arm="bp_mup", W=64, seed=s_, K=12, gamma=0.0, wd_scale=0.0, first95=900, B2_8=0.2, level1=True) for s_ in range(4)]
    write_json(os.path.join(r2g, "r2_t2", "theta_toy", "nets.json"), nets)
    write_json(os.path.join(r2g, "r2_t1", "rgi_toy", "r1", "w0.json"), dict(world=0, models=[dict(name="bp_mup_W128_w0_i0", cond="nat", first95=1250, B2_8=0.9, delta_h2=0.4, level1=True),
                                                                                           dict(name="bp_mup_W128_w0_i1", cond="nat", first95=1, B2_8=0.0), dict(name="bp_mup_W128_w0_i0_gamma0.01_whenlate", cond="task_late+0.01", first95=1)]))
    write_json(os.path.join(r2g, "r2_t3", "unlearn_toy", "t1.json"), dict(meta=dict(seeds=[0, 1, 2, 3, 4]), cells=dict(bp_mup_W128=dict(nets=[dict(name=f"bp_mup_W128_w{s_}_i0", first95=1200 + 10 * s_ + (700 if s_ == 0 else 0)) for s_ in range(5)],
                                                                                                                              base=dict(B2={"8": [0.3 + 0.01 * s_ for s_ in range(5)]})))))
    q = g46(rg, Inputs(rg, r2g, None)); cells = q["cells"]
    kB = "B-T1 / B-T2 nets (r2_t2) | bp_mup_W64_s*_K12_g0_wd1"; kK1 = "B-T1 / B-T2 nets (r2_t2) | bp_mup_W64_s*_K1_g0_wd1"; kC = "C-T one-hot nets (r2_t3) | bp_mup_W128_s*_K12_g0_wd1"; kA = "AT2 natural nets (r2_t1) | bp_mup_W128_s*_K12_g0_wd1"
    check("G46: a B-T1 4-seed cell needs every seed within 500 updates (one differs -> 'reproduction differs')", cells[kB]["label"] == "reproduction differs" and cells[kB]["first95_within_500"] == 3 and cells[kB]["first95_needed"] == 4 and cells[kB]["B2_ok"], cells[kB])
    check("G46: K = 1 cell (no first95 on either side) agrees; a 5-seed cell passes with 4 of 5; a B2 difference of 0.6 differs",
          cells[kK1]["ok"] and cells[kC]["ok"] and cells[kC]["first95_within_500"] == 4 and cells[kA]["label"] == "reproduction differs" and cells[kA]["n_seeds"] == 1 and kB in q["differing"] and kA in q["differing"], (cells[kC], cells[kA]))
    check("G46: cells without a round-1 job are listed as not compared", cells["B-T1 / B-T2 nets (r2_t2) | bp_mup_W64_s*_K12_g0_wd0"]["status"].startswith("no round-1 job") and q["not_compared"] == 1 and q["compared"] == 4)
    fd = load_json(os.path.join(r1, "gpu2", "dn_delta.json")); o7 = {rd: dict(delta=[fd["blocks"][str(l)][rd]["delta"] + (0.001 if l == 3 else 0.0) for l in range(28)]) for rd in ("last", "mean")}
    write_json(os.path.join(r2g, "r2_q3", "rgi", "geom_meta_qwen3-1.7b.json"), dict(stages=dict(o7=dict(delta=o7))))
    a = g34(r1, Inputs(r1, r2g, None)); o7["last"]["delta"][7] += 0.01; write_json(os.path.join(r2g, "r2_q3", "rgi", "geom_meta_qwen3-1.7b.json"), dict(stages=dict(o7=dict(delta=o7)))); b_ = g34(r1, Inputs(r1, r2g, None))
    check("G34: within 0.002 at every block passes; a block off by 0.01 is flagged; missing inputs 'not run'", a["ok"] and a["flag"] is None and b_["ok"] is False and b_["flag"] == "delta reproduction mismatch" and b_["readings"]["last"]["worst_block"] == 7
          and g34(r1, Inputs(r1, os.path.join(work, "empty"), None))["status"].startswith("not run") and g34(os.path.join(work, "no_r1"), Inputs(None, r2g, None))["status"].startswith("not run"))
    # amendments parser
    repo = os.path.join(work, "repo"); os.makedirs(os.path.join(repo, os.path.dirname(AMEND_REL)))
    open(os.path.join(repo, AMEND_REL), "w").write("# Amendments\n\n## Amendment 1 (2026-01-01T00:00:00Z)\n\n**Reason.** x\n\n**Change.** Rule A changes.\n\n**Outputs seen.** Not post hoc with respect to data.\n\n"
                                                    "## Amendment 2 (2026-01-02T00:00:00Z)\n\n**Change.** Rule B changes.\n\n**Outputs seen.** Declared post hoc with respect to a smoke test.\n")
    am = amendments(repo)
    check("amendments: titles, change text, post-hoc flag", am["present"] and [x["post_hoc"] for x in am["amendments"]] == [False, True] and am["amendments"][1]["change"] == "Rule B changes." and amendments(os.path.join(work, "nope"))["present"] is False)
    nd = notes_deviations(HERE)
    check("notes_deviations: the implementers' NOTES are parsed (sha256, deviation sections)", all("sha256" in v for v in nd.values()))
    # collectors
    tree = dict(a=dict(labels=["L1"], b=[dict(status="not run (x)", labels=["L2", "L1"])]), verdict="not testable (gate)", c=dict(flags=["F"]))
    check("collect / collect_not_run", collect(tree) == [("a", "L1"), ("a/b[0]", "L2"), ("a/b[0]", "L1")] and collect(tree, ("flags",)) == [("c", "F")] and collect_not_run(tree) == [("a/b[0]", "not run (x)"), ("", "not testable (gate)")])
    # end to end: synthetic verdict files, sessions with runner records, missing parts
    r2e = os.path.join(work, "r2e"); oute = os.path.join(work, "anae", "round2_summary.json")
    write_json(os.path.join(r2e, "r2_ana", "unlearn_verdict.json"), dict(summary={"C-Q3": dict(u_RSI="unlearned")}, C_Q3=dict(rules=dict(u_RSI=dict(verdict="unlearned", labels=[LBL_GDESC], inputs=dict(ratio=0.3))), G43=dict(agrees=True)),
                                                                           C_T7=dict(status="ran", verdict="not testable (no G-PC width)"), od_g8=["C-Q2 u_RSI: r2_ana 'suppressed', in session 'not suppressed'"], gates=dict(G41=dict(passes=True))))
    write_json(os.path.join(r2e, "r2_ana", "rgi_verdict.json"), dict(summary={"A7 over blocks": "V2: RGI geometry grows with delta or depth"}, A6=dict(blocks={"14": dict(alpha=1.0, suffix="", labels=[LBL_RGI])}, gen={"14": dict(status="not run (AUC gate)")}),
                                                                       AT8=dict(G38=dict(ok=True), G39=dict(ok=True))))
    write_json(os.path.join(r2e, "r2_q1", "session_done.json"), dict(session="r2_q1", attempts=[dict(attempt="r2_q1", result="time guard"), dict(attempt="r2_q1b", result="done")], packing=[dict(attempt="r2_q1", ran=["axes"]), dict(attempt="r2_q1b", ran=["q3"])],
                                                                       not_run={"gen_b20": "not run (round-1 output missing)"}))
    write_json(os.path.join(r2e, "r2_q1", "manifest_r2_q1.json"), dict(session="r2_q1", plan=dict(ok=True), freeze=dict(commit="abc", time="t"), gates=dict(G2=dict(ok=True, gpu="A100"), G3=dict(labels=["round-1 code differs between shards"])),
                                                                         copied=[dict(kind="r1", path="gpu1/rk/b14/state.pt", sha256="aa", source="gpu1")]))
    write_json(os.path.join(r2e, "r2_q2", "manifest_r2_q2.json"), dict(session="r2_q2", plan=dict(ok=True), gates=dict(G2=dict(ok=True)), copied=[dict(kind="r1", path="gpu1/rk/b14/state.pt", sha256="bb", source="gpu1")]))
    os.environ["R2_REPO"] = repo
    try: S_ = run(r1b, r2e, oute)
    finally: os.environ.pop("R2_REPO", None)
    man = load_json(os.path.join(work, "anae", "r2_manifest.json")); gj = load_json(os.path.join(work, "anae", "pc_toy", "gates.json")); js = load_json(oute)
    check("report end to end: strict JSON summary, gates file and merged manifest written", js is not None and man is not None and gj is not None and (json.dumps(js, allow_nan=False) is not None)
          and set(js["parts"]) == {"A", "B", "C"} and len(js["gates"]) == 49, list(js))
    check("report: verdicts, labels, parts 'not run', OD-G8 disagreements and the missing theta verdict are carried", js["parts"]["C"]["verdicts"]["C-Q3"]["u_RSI"] == "unlearned" and ["C_Q3/rules/u_RSI", LBL_GDESC] in js["parts"]["C"]["labels"]
          and js["parts"]["B"]["qwen"]["status"].startswith("not run") and "theta_verdict" in js["deviations"]["not_run"]["verdict_files_missing"] and js["deviations"]["od_g8_disagreements"]["C"]
          and ["A6/gen/14", "not run (AUC gate)"] in js["parts"]["A"]["not_run"] and js["deviations"]["not_run"]["sessions"]["r2_q1"] == {"gen_b20": "not run (round-1 output missing)"})
    check("report: packing (continuation session, time-guard stop), amendments (post hoc marked), G3 label, absent sessions", js["deviations"]["packing"]["r2_q1"]["continuation_sessions"] == ["r2_q1b"] and js["gates"]["G48"]["time_guard_stops"]["r2_q1"] == 1
          and js["deviations"]["amendments"]["amendments"][1]["post_hoc"] and js["gates"]["G3"]["labels"] == ["round-1 code differs between shards"] and "r2_t1" in js["deviations"]["not_run"]["sessions_absent"]
          and js["gates"]["G11"]["failed"] and js["gates"]["G4"]["consequence"] == {"20": NOT_RUN_R1})
    check("merged manifest: sessions, round-1 files with a sha256 conflict flagged, r2_ana outputs hashed", man["sessions"]["r2_q1"]["session_done"]["session"] == "r2_q1" and len(man["round1_files"]["gpu1/rk/b14/state.pt"]["sha256"]) == 2
          and man["flags"] and "round2_summary.json" in man["r2_ana_outputs"] and man["plan_sha256_frozen"] == PLAN_SHA256)
    ok = all(r["ok"] for r in res); write_json(os.path.join(work, "r2_report_selftest.json"), dict(ok=ok, tests=res, seconds=time.time() - t0))
    print(f"r2_report selftest: {sum(r['ok'] for r in res)}/{len(res)} passed ({time.time() - t0:.0f}s)", flush=True)
    return ok


if __name__ == "__main__":
    sys.exit(main())
