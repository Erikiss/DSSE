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
