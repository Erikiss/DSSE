"""Round 1b session runner (main §3.1 SG1-SG10, §4.7, §6.2), adapted from round2/r2_run_session.py. One process per session; the
session's steps come from the registry r1b_sessions_<part>.py and run in order in this process (function steps: the model stays loaded
across steps) or as subprocesses (script steps).

    python r1b_run_session.py <session> <root> <work> <runs> [--dry-run]    run or continue one session (slots: r1b_run_slot.py)
    python r1b_run_session.py <session> <root> <work> <runs> --check-start  exit 0 ready, 4 waiting, 5 'not run'
    python r1b_run_session.py --mock-tree <dir> <root> [--with-r1]           mock two-account Drive tree (code tests)
    python r1b_run_session.py --selftest <dir> <root>                        MOCK dry run of all slots, then runner tests (CPU)

<runs>  My Drive/dsse_runs of the account the session runs in; the session folder is <runs>/<session> (local copy <work>).
        Account A: My Drive/dsse_runs_B (shortcut to B's dsse_runs) next to it; account B: My Drive/r1b_xfer_A (A's dsse_runs/r1b_xfer).
What a session does: gates SG1 (freeze record and plan sha256, freeze commit an ancestor of the code commit; off for r1b_P0), SG2
(A100, or the declared CPU runtime of r1b_ana), SG3 (round-1 and round-2 modules at their pinned sha256, also at every round-1 shard's
code commit; round-1b modules clean at the code commit), SG7 (pilot record, trims, projection and funding in the freeze record); start
conditions SG4 (r1b_contracts.SESSIONS: producers done, single files with the sha256 of their producing step, A -> B exports with the
sha256 of xfer_manifest.json, tier-2 funding and triggers; B -> A by copying a finished B folder from dsse_runs_B, round 2's rule); the
copy-in of its inputs (read-only, <work>/in/); the manifest (sha256 of everything); the steps in order, each skipped when done, each
followed by the SG6 check of its outputs, an error-tolerant Drive sync and the time guard SG5 (per session: stop after the step that
ends >= 4.5 h after this session's start); session_done.json or session_partial.json (a continuation 'b', 'c', ... resumes).
Env: R1B_ACCOUNT (A | B), RUN_DEVICE (cuda | cpu = code test), R1B_T0 (session start, epoch s), R1B_GUARD_S (16200), R1B_FREEZE,
R1B_RUNS_B, R1B_XFER_A (override the shortcuts), MOCK=1 (code test: SG1 / SG7 / the round-1b part of SG3 recorded, not enforced;
example steps allowed; missing contract outputs filled with labelled mock files), NT, R1B_EXTRA_STEPS (MOCK only: dir:module).
Exit codes: 0 done; 1 steps failed (rerun: finished steps are skipped); 2 gate failed or usage error; 3 time guard (continuation);
4 start conditions not met (wait); 5 session 'not run' (tier 2 not funded, or not triggered)."""
import fnmatch, glob, importlib, json, os, platform, shlex, shutil, subprocess, sys, threading, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import r1b_contracts as C
import r1b_core as K
from r1b_core import iso, load_json, dump_json, sha256_file

FROZEN_PLAN_SHA256 = ""        # sha256 of results/ju_space/plans/runde1b/runde1b_plan.md, written here at the freeze (main §9.1, SG1)
FREEZE_RECORD_SHA256 = ""      # sha256 of results/ju_space/plans/runde1b/runde1b_freeze.json, written here at the freeze (SG1)
R1_MODULES = {"analysis/ju_space/rk_loop.py": "1a20f61992637386c97197b7283a8b743214aa89028f504c4af6c2fbedbab2a1",
              "analysis/ju_space/rk_verdict.py": "62aced20fcd7435201e1f427e5c294869fdab3adcb54801f092aa8d31ab9049e",
              "analysis/ju_space/ufo_common.py": "6f3919e93b638a325f9a8dd5762430c6b5e7693d4e1f5a317d97366caa49f9ab",
              "analysis/ju_space/rsi_statements.py": "6ff560c5b9c092858c7b50c4b041b7c886329bb7a451a6738d6eaa3754e3bec2"}
R2_MODULES = {"analysis/ju_space/round2/r2_rk.py": "d4618662c25db47b83b6b9e40533adabdea5c6f0f723a165a27e1338bc8fc8ca",
              "analysis/ju_space/round2/r2_stimuli.py": "41d2b564ec85835892c573e7f68d917006121c68764192a36c88e74035763434",
              "analysis/ju_space/round2/r2_theta.py": "eace3b7bbc027fed5da30e541e06d45136104057d7b2f0a36407571efc0433c7",
              "analysis/ju_space/round2/r2_theta_verdict.py": "b5ad18161d9f402ca098898fe8afa67e9cbd39d4e3a2e097a87d4ffe7b09b439",
              "analysis/ju_space/round2/r2_unlearn_qwen.py": "e0271d17dd6a9b873b205c952b969a365064a7a458f9048b473cf9a9908f52c9",
              "analysis/ju_space/round2/r2_linearity.py": "e410b8a0ebb6b1f4fe347a8cfdf007b7aaebc56a9f4779b7a619d688a01f1eba"}
GUARD_S, STEP_WARN_S = 4.5 * 3600, 300.0
EXIT_DONE, EXIT_FAILED, EXIT_GATE, EXIT_GUARD, EXIT_WAIT, EXIT_NOT_RUN = 0, 1, 2, 3, 4, 5
NOT_SYNCED = ("in",)
NOT_COPIED = ("logs/*", "cache/*", "in/*", "steps/*", "session_*.json", "start_check.json", "manifest*.json")
PY, REPO = sys.executable, K.REPO
MOCK = os.environ.get("MOCK") == "1"
LOGF = []


def say(msg):
    line = f"[r1b {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    for f in LOGF:
        try:
            f.write(line + "\n"); f.flush()
        except OSError:
            pass


def git(*args):
    try:
        r = subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True, timeout=120)
        return r.returncode, r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, repr(e)


def rglob(base, pattern):
    return sorted(p for p in glob.glob(os.path.join(glob.escape(base), pattern), recursive=True) if os.path.isfile(p))


def rel_ok(rel):
    return not any(fnmatch.fnmatch(rel, p) for p in NOT_COPIED)


# ====================================================================================================================================
# Registry
# ====================================================================================================================================
def registry():
    """session -> [Step] from r1b_sessions_R/D/K/L.py (each part registers its own sessions, plus steps of r1b_P0 and r1b_ana) and an
    optional r1b_sessions_main.py; under MOCK also R1B_EXTRA_STEPS (dir:module, test steps)."""
    mods = [f"r1b_sessions_{p}" for p in ("R", "D", "K", "L", "main")]
    extra = os.environ.get("R1B_EXTRA_STEPS") if MOCK else None
    if extra:
        d, m = extra.rsplit(":", 1); sys.path.insert(0, d); mods.append(m)
    reg = {}
    for name in mods:
        try:
            mod = importlib.import_module(name)
        except ModuleNotFoundError as e:
            if e.name == name and name in ("r1b_sessions_main",):
                continue
            raise
        part = name.rsplit("_", 1)[-1]
        for s, steps in getattr(mod, "STEPS", {}).items():
            if s not in C.SESSIONS:
                raise ValueError(f"{name}: unknown session {s}")
            if C.SESSIONS[s]["part"] not in (part, "main") and name != (extra or "").rsplit(":", 1)[-1]:
                raise ValueError(f"{name} registers steps for {s} (part {C.SESSIONS[s]['part']})")
            for st in steps:
                st.part = part if part in "RDKL" else C.SESSIONS[s]["part"]
                reg.setdefault(s, []).append(st)
    for s, steps in reg.items():
        names = [st.name for st in steps]
        if len(set(names)) != len(names):
            raise ValueError(f"{s}: duplicate step names {names}")
        for st in steps:
            bad = [a for a in st.after if a not in names]
            if bad:
                raise ValueError(f"{s}/{st.name}: unknown steps in after: {bad}")
    return reg


# ====================================================================================================================================
# Drive: paths of the two accounts, sync, pull, folder copies (B -> A), status of other sessions
# ====================================================================================================================================
class Paths:
    def __init__(self, runs, account):
        self.runs, self.account = os.path.abspath(runs), account
        top = os.path.dirname(self.runs)
        self.runs_B = os.environ.get("R1B_RUNS_B") or os.path.join(top, C.RUNS_B_LINK)
        self.xfer_src = os.environ.get("R1B_XFER_A") or os.path.join(top, C.XFER_LINK)
        self.xfer_pub = os.path.join(self.runs, C.XFER_DIR)


def is_done(d):
    return bool(d) and os.path.isfile(os.path.join(d, "session_done.json")) and not os.path.isfile(os.path.join(d, "session_partial.json"))


def sync(work, drive):
    """Copy new or changed files of <work> (not in/) to Drive, per file, atomic by rename; Drive errors are reported, never fatal."""
    if not drive:
        return 0
    n = err = 0
    for d, dirs, files in os.walk(work):
        rel = os.path.relpath(d, work)
        if rel == ".":
            dirs[:] = [x for x in dirs if x not in NOT_SYNCED]
        for f in sorted(files):
            if f.endswith((".tmp", ".r1btmp")):
                continue
            src, dst = os.path.join(d, f), os.path.normpath(os.path.join(drive, rel, f))
            try:
                st = os.stat(src)
                if os.path.exists(dst):
                    dt = os.stat(dst)
                    if dt.st_size == st.st_size and dt.st_mtime >= st.st_mtime - 1:
                        continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst + ".r1btmp"); os.replace(dst + ".r1btmp", dst); n += 1
            except OSError as e:
                err += 1
                if err <= 5:
                    say(f"[sync] Drive error, continuing without this copy: {e}")
    return n


def pull(drive, work):
    """Resume: files of the Drive session folder missing locally (or of another size) are copied into <work> (not in/)."""
    n = 0
    if not drive or not os.path.isdir(drive):
        return 0
    for p in rglob(drive, "**/*"):
        rel = os.path.relpath(p, drive)
        if rel.split(os.sep)[0] in NOT_SYNCED or p.endswith((".tmp", ".r1btmp")):
            continue
        q = os.path.join(work, rel)
        if not os.path.exists(q) or os.path.getsize(q) != os.path.getsize(p):
            os.makedirs(os.path.dirname(q), exist_ok=True); shutil.copy2(p, q); n += 1
    return n


def copy_folder(src, dst):
    """Round 2's second-folder rule: copy a finished folder under a temporary name, then rename; a stale unfinished copy is moved aside."""
    if os.path.isdir(dst):
        if is_done(dst):
            return False
        os.rename(dst, f"{dst}.stale_{time.strftime('%Y%m%d%H%M%S')}")
    tmp = os.path.join(os.path.dirname(dst), f".{os.path.basename(dst)}.copy")
    shutil.rmtree(tmp, ignore_errors=True); shutil.copytree(src, tmp, symlinks=False); os.rename(tmp, dst)
    say(f"[copy] {src} -> {dst}")
    return True


def declared(P):
    d = load_json(os.path.join(P.runs, C.DECLARED), {}) or {}
    return dict(d.get("sessions", d)) if isinstance(d, dict) else {}


def two_attempts(*dirs):
    for d in dirs:
        a = load_json(os.path.join(d, "session_attempts.json"), []) if d else []
        if sum(1 for x in a if x.get("result") != "done") >= 2:
            return True
    return False


def resolve(P, s, copy=True):
    """State of producer s seen from P's account: ('done', folder) | ('not_run', reason) | ('declared', reason) | ('waiting', why).
    In account A a finished B session is copied from dsse_runs_B into dsse_runs first (copy=True)."""
    own = os.path.join(P.runs, s)
    if is_done(own):
        return "done", own
    other = os.path.join(P.runs_B, s) if (P.account == "A" and C.SESSIONS[s]["account"] == "B") else None
    if other and is_done(other):
        if copy:
            copy_folder(other, own)
            return "done", own
        return "done", other
    for d in (own, other):
        nr = load_json(os.path.join(d, "session_not_run.json")) if d else None
        if nr:
            return "not_run", nr.get("reason", "not run")
    dec = declared(P)
    if s in dec:
        if two_attempts(own, other):
            return "declared", str(dec[s])
        say(f"note: {s} is declared 'not run' but fewer than two failed attempts are recorded; the declaration is ignored")
    return "waiting", f"{s} not finished"


def step_records(d):
    """rel -> sha256 of every output recorded by a successful step of the session folder d."""
    out = {}
    for p in rglob(os.path.join(d, "steps"), "*.json"):
        m = load_json(p) or {}
        if m.get("status") == "ok":
            out.update({k: v.get("sha256") for k, v in (m.get("outputs") or {}).items()})
    return out


def _get(d, key):
    for k in key.split("."):
        d = d.get(k) if isinstance(d, dict) else None
    return d


def funded(freeze, item):
    f = (freeze or {}).get("funded") or {}
    return bool(f.get("tier2")) and (not f.get("items") or item in f["items"])


def check_start(session, P, freeze):
    """SG4 and the tier-2 rules. Returns dict(ok, waiting, not_run, copies [(src, dst_rel, sha256 or None, source)], declared)."""
    m = C.SESSIONS[session]
    res = dict(ok=True, waiting=[], not_run=None, copies=[], declared=[], checked=iso())
    if m["tier"] == 2 and not funded(freeze, session):
        res.update(ok=False, not_run="tier2_unfunded"); return res

    def add_done(s, d, pats):
        rec = step_records(d)
        for pat in pats:
            hits = [p for p in rglob(d, pat) if rel_ok(os.path.relpath(p, d))]
            if not hits and not any(ch in pat for ch in "*?["):
                res["waiting"].append(f"{s}/{pat} missing in a finished session")
            for p in hits:
                r = os.path.relpath(p, d); res["copies"].append((p, os.path.join(s, r), rec.get(r), s))

    for c in m["after"] + tuple(dict(c, optional=True) for c in m["optional"]):
        k, opt = c["kind"], c.get("optional", False)
        if k in ("done", "trig"):
            st, info = resolve(P, c["session"])
            if st == "done":
                if k == "done":
                    add_done(c["session"], info, c["files"])
                else:
                    f = C.FILES[c["fid"]]; v = _get(load_json(os.path.join(info, f["rel"]), {}), c["key"])
                    hit = {"eq": v == c["value"], "ne": v is not None and v != c["value"], "true": v is True}[c["op"]]
                    if not hit:
                        res.update(ok=False, not_run="conditional_not_triggered"); return res
                    add_done(c["session"], info, (f["rel"],))
            elif st in ("declared", "not_run"):
                if k == "trig":
                    res.update(ok=False, not_run="conditional_not_triggered"); return res
                res["declared"].append(c["session"])
            elif not opt:
                res["waiting"].append(info)
        elif k == "file":
            f = C.FILES[c["fid"]]; d = os.path.join(P.runs, f["session"]); src = os.path.join(d, f["rel"])
            sha = step_records(d).get(f["rel"])
            if os.path.isfile(src) and sha and sha256_file(src) == sha:
                res["copies"].append((src, os.path.join(f["session"], f["rel"]), sha, f["session"]))
                if os.path.isfile(src + ".prov.json"):
                    res["copies"].append((src + ".prov.json", os.path.join(f["session"], f["rel"] + ".prov.json"), None, f["session"]))
            else:
                st, info = resolve(P, f["session"])
                if st in ("declared", "not_run"):
                    res["declared"].append(f["session"])
                elif not opt:
                    res["waiting"].append(f"{c['fid']} ({f['session']}/{f['rel']}) not recorded yet")
        elif k == "xfer":
            man = load_json(os.path.join(P.xfer_src, C.XFER_MANIFEST), {}) or {}
            ent, src = man.get(c["name"]), os.path.join(P.xfer_src, c["name"])
            if ent and os.path.isfile(src) and sha256_file(src) == ent.get("sha256"):
                res["copies"].append((src, os.path.join(C.XFER_DIR, c["name"]), ent["sha256"], f"r1b_xfer ({ent.get('session')})"))
            elif not opt:
                res["waiting"].append(f"A -> B export {c['name']} not in {P.xfer_src} (with its sha256 in {C.XFER_MANIFEST})")
        elif k == "any":
            sts = {s: resolve(P, s) for s in c["sessions"]}
            if any(v[0] == "done" for v in sts.values()):
                for s, (st, info) in sts.items():
                    if st == "done":
                        add_done(s, info, ("**",))
            elif all(v[0] in ("not_run", "declared") for v in sts.values()):
                res.update(ok=False, not_run="conditional_not_triggered"); return res
            else:
                res["waiting"].append(f"none of {c['sessions']} finished")
    for kind, pats in (("r1", m["r1"]), ("r2", m["r2"])):
        for pat in pats:
            hits = rglob(P.runs, pat)
            if not hits:
                res["waiting"].append(f"round-{kind[1]} input {pat} not found in {P.runs}")
            for p in hits:
                res["copies"].append((p, os.path.join(kind, os.path.relpath(p, P.runs)), None, f"round {kind[1]}"))
    res["ok"] = not res["waiting"]
    return res


def copy_inputs(work, copies):
    """Copies into <work>/in/ (read-only), each checked against the sha256 recorded by its producer; returns the manifest records."""
    out, bad = [], []
    for src, rel, sha, source in copies:
        dst = os.path.join(work, "in", rel)
        h = sha256_file(dst) if os.path.isfile(dst) else None
        if h is None or (sha and h != sha):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                os.chmod(dst, 0o644)
            shutil.copy2(src, dst + ".tmp"); os.replace(dst + ".tmp", dst); h = sha256_file(dst)
        os.chmod(dst, 0o444)
        if sha and h != sha:
            bad.append(rel)
        out.append(dict(path=rel, source=source, sha256=h, expected=sha, drive_mtime=iso(os.path.getmtime(src))))
    return out, bad


# ====================================================================================================================================
# Gates
# ====================================================================================================================================
def freeze_path():
    return os.environ.get("R1B_FREEZE") or os.path.join(REPO, C.PLAN_DIR, "runde1b_freeze.json")


def gate_SG1(pilot):
    fp = freeze_path(); fz = load_json(fp, {}) or {}
    plan = os.path.join(REPO, C.PLAN_DIR, "runde1b_plan.md"); amend = os.path.join(REPO, C.PLAN_DIR, "runde1b_plan_amendments.md")
    rec = dict(freeze_record=fp, freeze_record_sha256=sha256_file(fp) if os.path.isfile(fp) else None,
               freeze_record_sha256_constant=FREEZE_RECORD_SHA256, plan_sha256=sha256_file(plan) if os.path.isfile(plan) else None,
               plan_sha256_constant=FROZEN_PLAN_SHA256, amendments_sha256=sha256_file(amend) if os.path.isfile(amend) else None,
               freeze_commit=fz.get("freeze_commit"), freeze_time=fz.get("freeze_time"), enforced=not pilot and not MOCK)
    files = {}
    for path, h in {**(fz.get("plan_files") or {}), **(fz.get("package_files") or {})}.items():
        p = os.path.join(REPO, path); got = sha256_file(p) if os.path.isfile(p) else None
        files[path] = dict(expected=h, sha256=got, ok=got == h)
    rec["files"] = files
    rc, head = git("rev-parse", "HEAD"); rec["code_commit"] = head if rc == 0 else "unknown"
    rec["freeze_is_ancestor"] = None
    if rec["freeze_commit"] and rc == 0:
        rec["freeze_is_ancestor"] = git("merge-base", "--is-ancestor", rec["freeze_commit"], head)[0] == 0
    rec["ok"] = bool(FROZEN_PLAN_SHA256 and FREEZE_RECORD_SHA256 and rec["plan_sha256"] == FROZEN_PLAN_SHA256
                     and rec["freeze_record_sha256"] == FREEZE_RECORD_SHA256 and all(v["ok"] for v in files.values())
                     and rec["freeze_is_ancestor"])
    return rec, fz


def gate_SG2(session, device):
    rt = C.SESSIONS[session]["runtime"]
    if device == "cpu":
        return dict(ok=True, checked=False, runtime=rt, note="declared CPU runtime" if rt == "cpu" else "RUN_DEVICE=cpu: code test")
    if rt == "cpu":
        return dict(ok=False, checked=True, runtime=rt, note="r1b_ana runs on a declared CPU runtime (RUN_DEVICE=cpu)")
    probe = ("import json, torch; ok = torch.cuda.is_available(); p = torch.cuda.get_device_properties(0) if ok else None; "
             "print(json.dumps(dict(gpu=torch.cuda.get_device_name(0) if ok else None, cc=list(torch.cuda.get_device_capability(0)) "
             "if ok else None, mem=p.total_memory if ok else None, cuda=torch.version.cuda)))")
    try:
        r = subprocess.run([PY, "-c", probe], capture_output=True, text=True, timeout=300)
        info = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return dict(ok=False, checked=True, note=repr(e))
    ok = bool(info.get("gpu") and "A100" in info["gpu"] and info.get("cc") == [8, 0] and (info.get("mem") or 0) >= 39e9)
    return dict(ok=ok, checked=True, runtime=rt, **info)


def gate_SG3(pilot, commits=()):
    """Round-1 and round-2 modules equal their pinned sha256 in the checkout (and round-1 modules at every round-1 shard commit read);
    round-1b modules equal the code commit (clean checkout; recorded with sha256). r1b_P0: round-1 and round-2 modules only."""
    rec = dict(modules={}, shard_commits={}, enforced_r1b=not pilot and not MOCK)
    ok = True
    for path, h in {**R1_MODULES, **R2_MODULES}.items():
        p = os.path.join(REPO, path); got = sha256_file(p) if os.path.isfile(p) else None
        rec["modules"][path] = got; ok &= got == h
    for c in sorted(set(commits)):
        for path, h in R1_MODULES.items():
            try:
                r = subprocess.run(["git", "-C", REPO, "show", f"{c}:{path}"], capture_output=True, timeout=120)
                got = K.sha256_text(r.stdout) if r.returncode == 0 else None
            except (OSError, subprocess.TimeoutExpired):
                got = None
            rec["shard_commits"][f"{c}:{path}"] = got; ok &= got == h
    r1b = {os.path.relpath(p, REPO): sha256_file(p) for p in sorted(glob.glob(os.path.join(HERE, "*.py")))}
    rc, st = git("status", "--porcelain", "--", os.path.relpath(HERE, REPO))
    rec["round1b_modules"], rec["round1b_clean"] = r1b, (rc == 0 and st == "")
    rec["ok"] = bool(ok and (rec["round1b_clean"] or not rec["enforced_r1b"]))
    return rec


def gate_SG7(fz, pilot):
    miss = [k for k in ("pilot", "trims", "projection_h", "funded") if k not in fz]
    return dict(ok=pilot or not miss, missing=miss, enforced=not pilot and not MOCK, trims=fz.get("trims"), funded=fz.get("funded"),
                projection_h=fz.get("projection_h"))


# ====================================================================================================================================
# Steps
# ====================================================================================================================================
def marker(work, st):
    return os.path.join(work, "steps", st.name + ".json")


def outputs_of(work, pats):
    return {p: rglob(work, p) for p in pats}


def step_done(ctx, st):
    m = load_json(marker(ctx.work, st)) or {}
    if m.get("status") == "not_run":
        return True
    return m.get("status") == "ok" and all(outputs_of(ctx.work, st.mock_out if (ctx.mock and st.mock_out is not None) else st.out).values())


def resolve_fn(fn):
    if callable(fn):
        return fn
    mod, name = fn.split(":")
    return getattr(importlib.import_module(mod), name)


def exec_step(ctx, st):
    os.makedirs(os.path.join(ctx.work, "logs"), exist_ok=True)
    ctx.step, ctx.step_started, ctx.keys_used, ctx.models_used, ctx.jlens_sha256 = st.name, iso(), set(), {}, None
    t, status, note = time.time(), "ok", None
    say(f"[run] {st.name}" + (f" ({st.minutes:g} min planned)" if st.minutes else ""))
    try:
        if st.fn is not None:
            resolve_fn(st.fn)(ctx)
        else:
            fmt = dict(root=ctx.root, work=ctx.work, inp=ctx.inp, r1=ctx.r1, r2=ctx.r2)
            argv = [a.format(**fmt) for a in shlex.split(st.args)]
            stf = os.path.join(ctx.work, "steps", st.name + ".status")
            env = dict(os.environ, R1B_SESSION=ctx.session, R1B_WORK=ctx.work, R1B_IN=ctx.inp, R1B_ROOT=ctx.root, R1B_ATTEMPT=ctx.attempt,
                       R1B_ACCOUNT=ctx.account or "", R1B_FREEZE=freeze_path(), R1B_MANIFEST=os.path.join(ctx.work, "manifest.json"),
                       R1B_DECLARED=",".join(ctx.declared), R1B_T0=repr(ctx.t0), R1B_GUARD_S=repr(ctx.guard_s), R1B_STEP=st.name,
                       R1B_STEP_STATUS=stf, DEVICE=ctx.device, PYTHONUNBUFFERED="1", **st.env)
            with open(os.path.join(ctx.work, "logs", st.name + ".log"), "a") as lg:
                lg.write(f"=== {iso()} attempt {ctx.attempt}\n"); lg.flush()
                pr = subprocess.Popen([PY, "-u", os.path.join(HERE, st.script)] + argv, env=env, cwd=ctx.work, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)

                def pump():
                    for line in pr.stdout:
                        lg.write(line); lg.flush()
                        if "Loading weights" not in line:
                            print(line, end="", flush=True)
                th = threading.Thread(target=pump, daemon=True); th.start()
                try:
                    pr.wait(timeout=st.timeout_s)
                except subprocess.TimeoutExpired:
                    os.killpg(pr.pid, 9); pr.wait(); th.join(timeout=30); raise
                th.join(timeout=30)
            r = pr
            if r.returncode == K.EXIT_NOT_RUN:
                raise K.NotRun((load_json(stf) or {}).get("not_run", "not run"))
            if r.returncode == K.EXIT_YIELD:
                raise K.StepYield()
            if r.returncode != 0:
                status, note = "failed", f"exit code {r.returncode} (log logs/{st.name}.log)"
    except K.NotRun as e:
        status, note = "not_run", e.reason
    except K.StepYield:
        status, note = "yield", "stopped before the time guard; resumes in the continuation"
    except subprocess.TimeoutExpired:
        status, note = "failed", f"timeout {st.timeout_s}s"
    except Exception:
        status, note = "failed", traceback.format_exc()[-1500:]
        with open(os.path.join(ctx.work, "logs", st.name + ".log"), "a") as lg:
            lg.write(f"=== {iso()} attempt {ctx.attempt}\n{traceback.format_exc()}\n")
    secs = round(time.time() - t, 1)
    pats = st.mock_out if (ctx.mock and st.mock_out is not None) else st.out
    found = outputs_of(ctx.work, pats + st.opt)
    if status == "ok":
        miss = [p for p in pats if not found[p]]
        probs = [x for x in (K.check_output(f) for h in found.values() for f in h) if x]
        if miss:
            status, note = "failed", "outputs missing: " + ", ".join(miss)
        elif probs:
            status, note = "failed", "SG6: " + "; ".join(probs[:5])
    outs = {}
    for f in sorted({f for h in found.values() for f in h} | {f + ".prov.json" for h in found.values() for f in h if os.path.isfile(f + ".prov.json")}):
        outs[os.path.relpath(f, ctx.work)] = dict(size=os.path.getsize(f), sha256=sha256_file(f))
    rec = dict(name=st.name, status=status, seconds=secs, attempt=ctx.attempt, code_commit=(ctx.manifest or {}).get("code_commit"),
               started=ctx.step_started, ended=iso(), outputs=outs, note=note, part=st.part, item=st.item,
               code=("NA:not_run:" + note) if status == "not_run" else None, long=secs > STEP_WARN_S)
    dump_json(rec, marker(ctx.work, st))
    t_ = sys.modules.get("torch")
    if t_ is not None and t_.cuda.is_available():
        t_.cuda.empty_cache()
    say(f"[done] {st.name}: {status} ({secs:.0f}s)" + (f" - {note.splitlines()[-1] if note else ''}" if status != "ok" else "")
        + (" [SG5: longer than 5 min]" if secs > STEP_WARN_S else ""))
    return rec


def pre_state(ctx, st, states):
    if step_done(ctx, st):
        m = load_json(marker(ctx.work, st)) or {}
        return dict(state="not run" if m.get("status") == "not_run" else "skipped", note=m.get("note") or "done earlier")
    if st.item and not ctx.funded(st.item):
        return dict(state="not run", note="tier2_unfunded", code=K.NA_TIER2, record=True)
    for t in st.drop_if:
        if ctx.trimmed(t):
            return dict(state="not run", note=t, code=f"NA:not_run:{t}", record=True)
    for a in st.after:
        s = states.get(a, {}).get("state")
        if s == "not run":
            return dict(state="not run", note=f"prerequisite {a} not run", code=f"NA:not_run:prerequisite {a} not run", record=True)
        if s not in ("ran", "skipped"):
            return dict(state="blocked", note=f"waits for {a}")
    return None


def run_steps(ctx, steps, publish):
    states, ran, guard = {}, [], False
    for rnd in (1, 2):                                   # one retry of failed steps if time allows (round-1 pattern)
        for st in steps:
            cur = states.get(st.name, {}).get("state")
            if cur in ("ran", "skipped", "not run") or (rnd == 2 and cur not in ("failed", "blocked")):
                continue
            pre = pre_state(ctx, st, states)
            if pre and pre["state"] in ("skipped", "not run"):
                if pre.pop("record", False):
                    dump_json(dict(name=st.name, status="not_run", note=pre["note"], code=pre["code"], attempt=ctx.attempt, ended=iso(),
                                   outputs={}), marker(ctx.work, st))
                states[st.name] = pre; say(f"[{pre['state']}] {st.name}: {pre['note']}"); continue
            if guard:
                states[st.name] = dict(state="pending", note="time guard"); continue
            if pre:
                states[st.name] = pre; say(f"[wait] {st.name}: {pre['note']}"); continue
            rec = exec_step(ctx, st); ran.append(st.name)
            states[st.name] = dict(state={"ok": "ran", "not_run": "not run", "yield": "pending"}.get(rec["status"], "failed"),
                                   note=rec["note"], seconds=rec["seconds"])
            if rec["status"] == "yield":
                guard = True
            publish()
            n = sync(ctx.work, ctx.drive)
            el = time.time() - ctx.t0
            say(f"[time] {el / 60:.1f} min since the session start (guard {ctx.guard_s / 60:.0f} min); {n} files synced")
            if el >= ctx.guard_s and not guard:
                guard = True; say("time guard (SG5): the session stops after this step")
        if guard:
            break
    for st in steps:
        states.setdefault(st.name, dict(state="pending", note="time guard"))
    return states, ran, guard


def mock_fill(ctx):
    """MOCK: write a labelled placeholder for every contract output of this session that its steps did not write (so that every start
    condition can be met in the dry run); trigger files carry the triggering value. Recorded as step '_mock_fill'."""
    if ctx.pilot:
        return
    trig = {}
    for s, m in C.SESSIONS.items():
        for c in m["after"]:
            if c["kind"] == "trig":
                trig[c["fid"]] = (c["key"], c["value"] if c["op"] == "eq" else (True if c["op"] == "true" else f"mock_not_{c['value']}"))
    ctx.step, made = "_mock_fill", []
    for fid, f in C.produced_by(ctx.session).items():
        rel = f["rel"].replace("{b}", "14")
        if f["fmt"] == "dir" or os.path.exists(os.path.join(ctx.work, rel)):
            continue
        obj = dict(mock=True, fid=fid, keys=f["keys"])
        if fid in trig:
            d = obj
            ks = trig[fid][0].split(".")
            for k in ks[:-1]:
                d = d.setdefault(k, {})
            d[ks[-1]] = trig[fid][1]
        if f["fmt"] == "json":
            ctx.write_json(rel, obj, labels=["MOCK placeholder"])
        elif f["fmt"] == "pt":
            ctx.save_pt(rel, obj, labels=["MOCK placeholder"])
        else:
            ctx.write_file(rel, json.dumps(K.jsonable(obj)) + "\n", labels=["MOCK placeholder"])
        made.append(rel)
    if made:
        outs = {}
        for r in made:
            for q in (r, r + ".prov.json"):
                if os.path.isfile(os.path.join(ctx.work, q)):
                    outs[q] = dict(size=os.path.getsize(os.path.join(ctx.work, q)), sha256=sha256_file(os.path.join(ctx.work, q)))
        dump_json(dict(name="_mock_fill", status="ok", outputs=outs, attempt=ctx.attempt, ended=iso(), note="MOCK placeholders"),
                  os.path.join(ctx.work, "steps", "_mock_fill.json"))
        say(f"[mock] placeholders for {made}")


def publish_xfer(ctx, P):
    """A -> B exports (SO-XFER): every contract file with an 'xfer' name produced by this session and recorded by a successful step is
    copied to dsse_runs/r1b_xfer/<name>, with its sha256 in xfer_manifest.json (written last)."""
    ex = {fid: f for fid, f in C.produced_by(ctx.session).items() if f.get("xfer")}
    if not ex or P.account != "A":
        return
    rec = step_records(ctx.work); man_p = os.path.join(P.xfer_pub, C.XFER_MANIFEST)
    man = load_json(man_p, {}) or {}; changed = False
    for fid, f in ex.items():
        src, h = os.path.join(ctx.work, f["rel"]), rec.get(f["rel"])
        if not h or not os.path.isfile(src) or man.get(f["xfer"], {}).get("sha256") == h:
            continue
        try:
            dst = os.path.join(P.xfer_pub, f["xfer"]); os.makedirs(P.xfer_pub, exist_ok=True)
            shutil.copy2(src, dst + ".r1btmp"); os.replace(dst + ".r1btmp", dst)
            man[f["xfer"]] = dict(sha256=h, session=ctx.session, rel=f["rel"], attempt=ctx.attempt,
                                  code_commit=(ctx.manifest or {}).get("code_commit"), written=iso()); changed = True
        except OSError as e:
            say(f"[xfer] Drive error ({e}); retried after the next step")
    if changed:
        dump_json(man, man_p); say(f"[xfer] published {sorted(man)} to {P.xfer_pub}")


# ====================================================================================================================================
# Session
# ====================================================================================================================================
def split_session(name):
    if name in C.SESSIONS:
        return name, None
    if len(name) > 1 and name[:-1] in C.SESSIONS and name[-1] in "bcdefghijklmnopqrstuvwxyz":
        return name[:-1], name
    raise SystemExit(f"unknown session {name!r}; sessions: {', '.join(C.SESSIONS)}")


def write_not_run(P, s, reason, work=None):
    rec = dict(session=s, reason=reason, code="NA:not_run:" + reason if reason != "conditional_not_triggered" else K.NA_COND, written=iso())
    for d in (work, os.path.join(P.runs, s)):
        if d:
            try:
                dump_json(rec, os.path.join(d, "session_not_run.json"))
            except OSError as e:
                say(f"could not write session_not_run.json ({e})")
    say(f"{s}: not run ({reason})")


def finalize(ctx, P, steps, states, ran, guard, attempts, gates):
    steps_rec, files = {}, {}
    for st in steps:
        m = load_json(marker(ctx.work, st)) or {}; s = states[st.name]
        steps_rec[st.name] = dict(state=s["state"], note=s.get("note"), code=m.get("code"), seconds=m.get("seconds"), ran_in=m.get("attempt"),
                                  outputs=m.get("outputs"), long=m.get("long"))
        files.update({k: v["sha256"] for k, v in (m.get("outputs") or {}).items()})
    mf = load_json(os.path.join(ctx.work, "steps", "_mock_fill.json"))
    if mf:
        files.update({k: v["sha256"] for k, v in mf["outputs"].items()})
    done = all(v["state"] in ("ran", "skipped", "not run") for v in steps_rec.values())
    pending = [n for n, v in steps_rec.items() if v["state"] == "pending"]
    failed = [n for n, v in steps_rec.items() if v["state"] in ("failed", "blocked")]
    attempts[-1].update(ended=iso(), seconds_since_session_start=round(time.time() - ctx.t0, 1), ran=ran,
                        result="done" if done else ("time guard" if guard and pending and not failed else "failed"))
    dump_json(attempts, os.path.join(ctx.work, "session_attempts.json"))
    common = dict(session=ctx.session, part=ctx.part, tier=ctx.tier, account=ctx.account, attempts=attempts, steps=steps_rec, files=files,
                  not_run={n: v["code"] or v["note"] for n, v in steps_rec.items() if v["state"] == "not run"},
                  long_steps=[n for n, v in steps_rec.items() if v.get("long")], gates=gates, declared=list(ctx.declared), mock=ctx.mock,
                  code_commit=ctx.manifest.get("code_commit"), slot=os.environ.get("R1B_SLOT"),
                  packing="slot notebook (main §4.4); packing changes no measurement")
    for d in (ctx.work, ctx.drive):
        for f in ("session_partial.json", "session_done.json"):
            try:
                if d and os.path.exists(os.path.join(d, f)):
                    os.remove(os.path.join(d, f))
            except OSError as e:
                say(f"could not remove an old {f}: {e}")
    if done:
        if ctx.pilot:                                   # the engineering record for the freeze record (SG7), whitelisted keys only
            agg = {}
            for rel in sorted(files):
                if rel.endswith(".json") and not rel.endswith(".prov.json"):
                    for k_, v_ in (load_json(os.path.join(ctx.work, rel), {}) or {}).items():
                        if k_ != "provenance":
                            agg.setdefault(k_, {})[rel] = v_
            ctx.step = "_p0_record"; pr = ctx.write_json(C.FILES["p0_record"]["rel"], agg)
            common["files"][C.FILES["p0_record"]["rel"]] = sha256_file(pr)
        dump_json(dict(common, finished=iso()), os.path.join(ctx.work, "session_done.json"))
        code, msg = EXIT_DONE, f"session {ctx.session} done"
    else:
        reason = "time guard (SG5)" if guard and pending and not failed else "steps failed"
        k = sum(a.get("result") == "time guard" for a in attempts)
        nxt = ctx.session + "bcdefghijklmnopqrstuvwxyz"[k - 1] if reason.startswith("time") else ctx.attempt
        dump_json(dict(common, reason=reason, remaining=pending, failed=failed, next_attempt=nxt, written=iso()),
                  os.path.join(ctx.work, "session_partial.json"))
        code = EXIT_GUARD if reason.startswith("time") else EXIT_FAILED
        msg = (f"session {ctx.session}: time guard, {len(pending)} steps remain; continuation {nxt} resumes them" if code == EXIT_GUARD
               else f"session {ctx.session}: steps failed or blocked: {failed}; rerun to retry them")
    sync(ctx.work, ctx.drive)
    say(msg)
    return code


def run_session(session, root, work, runs, dry=False, check_only=False):
    base, explicit = split_session(session)
    meta = C.SESSIONS[base]
    acct = os.environ.get("R1B_ACCOUNT") or meta["account"]
    if acct != meta["account"]:
        say(f"{base} runs in account {meta['account']}, this notebook is account {acct}"); return EXIT_GATE
    P = Paths(runs, acct); pilot = base == "r1b_P0"
    device = os.environ.get("RUN_DEVICE", "cuda")
    fz_path = freeze_path(); fz = load_json(fz_path, {}) or {}
    if check_only or dry:
        chk = check_start(base, P, fz)
        say(f"{base}: " + ("ready" if chk["ok"] else (f"not run ({chk['not_run']})" if chk["not_run"] else "waiting: " + "; ".join(chk["waiting"]))))
        if dry:
            for st in registry().get(base, []):
                say(f"  step {st.name}: {st.fn or st.script} out={st.out} after={st.after} item={st.item} drop_if={st.drop_if}")
        return EXIT_DONE if chk["ok"] else (EXIT_NOT_RUN if chk["not_run"] else EXIT_WAIT)
    work, drive = os.path.abspath(work), os.path.join(P.runs, base)
    os.makedirs(os.path.join(work, "logs"), exist_ok=True)
    LOGF.append(open(os.path.join(work, "logs", "session.log"), "a"))
    t0 = float(os.environ.get("R1B_T0") or time.time())
    say(f"session {base} (part {meta['part']}, tier {meta['tier']}, account {acct}) {'MOCK ' if MOCK else ''}on {device}; "
        f"{pull(drive, work)} files resumed from Drive")
    if is_done(work):
        say(f"session {base} is already done (session_done.json)"); return EXIT_DONE
    try:
        steps = registry().get(base, [])
    except Exception:
        say("session registry broken:\n" + traceback.format_exc()); return EXIT_GATE
    if not steps and not MOCK:
        say(f"no steps registered for {base} (r1b_sessions_{meta['part']}.py)"); return EXIT_GATE
    if any(st.example for st in steps) and not MOCK:
        say(f"{base}: example steps are registered ({[st.name for st in steps if st.example]}); replace them before a real run")
        return EXIT_GATE
    g1, fz = gate_SG1(pilot)
    g2, g7 = gate_SG2(base, device), gate_SG7(fz, pilot)
    early = [k for k, g in (("SG1", g1), ("SG2", g2), ("SG7", g7)) if not g["ok"] and g.get("enforced", True)]
    if early:
        dump_json(dict(session=base, gates=dict(SG1=g1, SG2=g2, SG7=g7), failed=early, written=iso()), os.path.join(work, "manifest.json"))
        say(f"gates failed: {early} (manifest.json); nothing ran"); sync(work, drive); return EXIT_GATE
    chk = check_start(base, P, fz)
    if chk["not_run"]:
        write_not_run(P, base, chk["not_run"], work); return EXIT_NOT_RUN
    dump_json(chk, os.path.join(work, "start_check.json"))
    if not chk["ok"]:
        say("start conditions not met (SG4): " + "; ".join(chk["waiting"])); sync(work, drive); return EXIT_WAIT
    copied, bad = copy_inputs(work, chk["copies"])
    if bad:
        say(f"SG4: copied files differ from their producers' sha256: {bad}; waiting"); return EXIT_WAIT
    commits = []
    for c in copied:
        if c["path"].startswith("r1" + os.sep) and c["path"].endswith("code_commits.txt"):
            commits += open(os.path.join(work, "in", c["path"])).read().split()
    g3 = gate_SG3(pilot, commits)
    gates = dict(SG1=g1, SG2=g2, SG3=g3, SG7=g7)
    fails = [k for k, g in (("SG1", g1), ("SG2", g2), ("SG3", g3), ("SG7", g7)) if not g["ok"] and g.get("enforced", True)]
    attempts = load_json(os.path.join(work, "session_attempts.json"), []) or []
    k = sum(a.get("result") == "time guard" for a in attempts)
    attempt = explicit or base + ("" if k == 0 else "bcdefghijklmnopqrstuvwxyz"[k - 1])
    manifest = dict(session=base, attempt=attempt, account=acct, part=meta["part"], tier=meta["tier"], slot=os.environ.get("R1B_SLOT"),
                    t0=iso(t0), started=iso(), code_commit=g1["code_commit"], mock=MOCK, device=device, host=platform.node(),
                    plan=dict(sha256=g1["plan_sha256"], freeze_record=g1["freeze_record_sha256"], freeze_commit=g1["freeze_commit"],
                              freeze_time=g1["freeze_time"], amendments_sha256=g1["amendments_sha256"], files=g1["files"]),
                    gates=gates, inputs=copied, declared=chk["declared"], versions=K.versions(), funded=fz.get("funded"),
                    trims=fz.get("trims"), models={m: K.model_pins(m, fz) for m in meta["models"]}, keys_sha256=K.sha256_text(K.canon(K.KEYS)))
    dump_json(manifest, os.path.join(work, "manifest.json"))
    if fails:
        say(f"gates failed: {fails} (manifest.json)"); sync(work, drive); return EXIT_GATE
    attempts.append(dict(attempt=attempt, started=iso(), t0=iso(t0), host=platform.node(), device=device, slot=os.environ.get("R1B_SLOT")))
    dump_json(attempts, os.path.join(work, "session_attempts.json"))
    for d in (work, drive):
        try:
            if os.path.exists(os.path.join(d, "session_done.json")):
                os.remove(os.path.join(d, "session_done.json"))
        except OSError:
            pass
    ctx = K.Ctx(base, work, root=root, device=device, attempt=attempt, account=acct, mock=MOCK, pilot=pilot, freeze=fz, manifest=manifest,
                declared=chk["declared"], t0=t0, guard_s=float(os.environ.get("R1B_GUARD_S", GUARD_S)))
    ctx.drive = drive
    states, ran, guard = run_steps(ctx, steps, lambda: publish_xfer(ctx, P))
    if MOCK and all(v["state"] in ("ran", "skipped", "not run") for v in states.values()):
        mock_fill(ctx)
    publish_xfer(ctx, P)
    return finalize(ctx, P, steps, states, ran, guard, attempts, {k: dict(ok=g["ok"], enforced=g.get("enforced", True)) for k, g in gates.items()})


# ====================================================================================================================================
# Mock tree and self-test
# ====================================================================================================================================
SHARD_OF = {14: "gpu1", 18: "gpu2", 6: "gpu3"}          # round 1: block 14 on gpu1, 18 on gpu2, 6 on gpu3 (main §2.2)


def _concrete(pat):
    """Mock file names for a round-1 / round-2 input pattern of the contracts."""
    import re
    if pat.startswith("gpu*/"):
        rest = pat[5:]; m = re.search(r"b(\d+)", rest)
        if m:
            return [f"{SHARD_OF[int(m.group(1))]}/{rest}"]
        if "b*" in rest:
            return [f"{sh}/{rest.replace('b*', f'b{b}').replace('*', '')}" for b, sh in SHARD_OF.items()]
        return [f"{sh}/{rest}" for sh in SHARD_OF.values()]
    if "b*" in pat:
        return [pat.replace("b*", f"b{b}") for b in ((18, 6) if "r2RSIb" in pat else SHARD_OF)]
    return [pat]


def mock_tree(d, root, with_r1=False):
    """<d>/A/MyDrive/dsse_runs (round-1 and round-2 placeholders for every pattern of the contracts; with_r1: rk_loop.py MOCK=1 states at
    blocks 14, 18, 6 on the CPU), <d>/B/MyDrive/dsse_runs, the shortcuts dsse_runs_B (in A) and r1b_xfer_A (in B) as symlinks, and a mock
    freeze record <d>/runde1b_freeze.json (tier 2 funded, no trims). Each part's r1b_sessions_<part>.mock_inputs(runs_A, runs_B, root) may
    add its own mock inputs."""
    d = os.path.abspath(d); A, B = os.path.join(d, "A", "MyDrive"), os.path.join(d, "B", "MyDrive")
    ra, rb = os.path.join(A, "dsse_runs"), os.path.join(B, "dsse_runs")
    for p in (ra, rb, os.path.join(ra, C.XFER_DIR)):
        os.makedirs(p, exist_ok=True)
    for link, target in ((os.path.join(A, C.RUNS_B_LINK), rb), (os.path.join(B, C.XFER_LINK), os.path.join(ra, C.XFER_DIR))):
        if not os.path.lexists(link):
            os.symlink(target, link)
    head = git("rev-parse", "HEAD")[1]
    if with_r1 or os.environ.get("R1B_MOCK_WITH_R1") == "1":     # as round 2's mock_r1 (needs <root>/rsi_dl/persona, gpu_pkg/rk_dirs.pt)
        env = dict(os.environ, MOCK="1", DEVICE="cpu", NT=os.environ.get("NT", "1"), PYTHONUNBUFFERED="1",
                   POOLDIRS=os.path.join(root, "gpu_pkg", "rk_dirs.pt"), PERSONA=os.path.join(root, "rsi_dl", "persona"))
        env.pop("ASCALE", None)
        for shard, b in (("gpu1", 14), ("gpu2", 18), ("gpu3", 6)):
            out = os.path.join(ra, shard, "rk")
            if not os.path.exists(os.path.join(out, f"b{b}", "loops.json")):
                for stage in ("pilot", "all"):
                    subprocess.run([PY, os.path.join(K.AJS, "rk_loop.py"), root, out, stage], env=dict(env, BLOCK=str(b)), check=True)
                subprocess.run([PY, os.path.join(K.AJS, "rk_verdict.py"), "block", os.path.join(out, f"b{b}"),
                                os.path.join(ra, shard, f"summary_b{b}.json")], env=env, check=True)
    pats = set()
    for m in C.SESSIONS.values():
        pats |= set(m["r1"]) | set(m["r2"])
    for fp in sorted({os.path.join(ra, p) for pat in pats for p in _concrete(pat)}):
        if not os.path.exists(fp):
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            if fp.endswith(".txt"):
                open(fp, "w").write(head + "\n")
            elif fp.endswith(".pt"):
                import torch
                torch.save({"mock": True}, fp)
            else:
                dump_json(dict(mock=True, code_commit=head, device="cpu"), fp)
    fz = os.path.join(d, "runde1b_freeze.json")
    if not os.path.exists(fz):
        dump_json(dict(mock=True, freeze_commit=None, plan_files={}, package_files={}, pilot={}, trims={}, projection_h=30.6,
                       funded=dict(tier2=True, items=[]), datasets={}, models={}), fz)
    for p in ("R", "D", "K", "L"):
        mod = importlib.import_module(f"r1b_sessions_{p}")
        if hasattr(mod, "mock_inputs"):
            mod.mock_inputs(ra, rb, root)
    return dict(A=ra, B=rb, freeze=fz)


TEST_MOD = '''"""Runner self-test steps (MOCK only, loaded through R1B_EXTRA_STEPS)."""
import os
from r1b_contracts import Step


def ok(ctx):
    ctx.write_json(f"t/{ctx.step}.json", dict(x=1))


def bad(ctx):
    import json
    json.dump(dict(x=1), open(ctx.path("t/bad.json"), "w"))


STEPS = {"r1b_D_B": [Step("t1", fn="r1b_sessions_test:ok", out="t/t1.json", example=True),
                     Step("t2", fn="r1b_sessions_test:ok", out="t/t2.json", after="t1", example=True)]}
if os.environ.get("R1B_TEST_BAD") == "1":
    STEPS["r1b_D_B"].append(Step("t3", fn="r1b_sessions_test:bad", out="t/bad.json", after="t2", example=True))
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "r1b_test_script.py")
STEPS["r1b_D_C"] = [Step("s1", script=SCRIPT, args="ok {work}", out="t/s1.json", example=True),
                    Step("s2", script=SCRIPT, args="notrun {work}", after="s1", example=True),
                    Step("s3", fn="r1b_sessions_test:ok", out="t/s3.json", after="s2", example=True)]
'''
TEST_SCRIPT = '''import sys
sys.path.insert(0, %r)
import r1b_core as K


def main(ctx):
    if sys.argv[1] == "notrun":
        raise K.NotRun("planted not run")
    ctx.write_json("t/s1.json", dict(work_ok=sys.argv[2] == ctx.work))


K.run_script(main)
''' % HERE


def selftest(d, root):
    """MOCK dry run of every slot (r1b_run_slot.py --mock-dry-run), then runner tests on r1b_D_B: SG6 failure of a step without
    provenance, time guard and continuation, resume (all steps skipped)."""
    d = os.path.abspath(d); errs = []
    env = dict(os.environ, MOCK="1", RUN_DEVICE="cpu", NT=os.environ.get("NT", "1"))
    r = subprocess.run([PY, "-u", os.path.join(HERE, "r1b_run_slot.py"), "--mock-dry-run", os.path.join(d, "dry"), root], env=env)
    (errs.append("mock dry run of all slots") if r.returncode else None)
    t = mock_tree(os.path.join(d, "rt"), root); tm = os.path.join(d, "rt", "testmods"); os.makedirs(tm, exist_ok=True)
    open(os.path.join(tm, "r1b_sessions_test.py"), "w").write(TEST_MOD)
    open(os.path.join(tm, "r1b_test_script.py"), "w").write(TEST_SCRIPT)
    base = dict(env, R1B_ACCOUNT="A", R1B_FREEZE=t["freeze"], R1B_EXTRA_STEPS=f"{tm}:r1b_sessions_test")
    run = lambda work, **e: subprocess.run([PY, "-u", os.path.abspath(__file__), "r1b_D_B", root, work, t["A"]],
                                           env=dict(base, **e)).returncode
    w = os.path.join(d, "rt", "work")
    rc = run(os.path.join(w, "bad"), R1B_TEST_BAD="1")
    (errs.append(f"SG6: expected exit 1, got {rc}") if rc != EXIT_FAILED else None)
    shutil.rmtree(os.path.join(t["A"], "r1b_D_B"), ignore_errors=True)
    rc1 = run(os.path.join(w, "g"), R1B_GUARD_S="0"); rc2 = run(os.path.join(w, "g"), R1B_GUARD_S="0")
    rc3 = run(os.path.join(w, "g"), R1B_GUARD_S="0")
    att = load_json(os.path.join(w, "g", "session_attempts.json"), [])
    if (rc1, rc2, rc3) != (EXIT_GUARD, EXIT_DONE, EXIT_DONE) or [a["attempt"] for a in att] != ["r1b_D_B", "r1b_D_Bb"]:
        errs.append(f"time guard / continuation: exit codes {(rc1, rc2, rc3)}, attempts {[a.get('attempt') for a in att]}")
    shutil.rmtree(os.path.join(w, "g"))
    rc4 = run(os.path.join(w, "g"))                    # a new machine: pulled from Drive, already done
    (errs.append(f"resume from Drive: exit {rc4}") if rc4 != EXIT_DONE else None)
    rc5 = subprocess.run([PY, "-u", os.path.abspath(__file__), "r1b_D_C", root, os.path.join(w, "s"), t["A"]], env=base).returncode
    sd = load_json(os.path.join(w, "s", "session_done.json"), {}) or {}; stp = sd.get("steps", {})
    s1 = load_json(os.path.join(w, "s", "t", "s1.json"), {}) or {}
    if rc5 != EXIT_DONE or stp.get("s1", {}).get("state") != "ran" or stp.get("s2", {}).get("code") != "NA:not_run:planted not run" \
            or stp.get("s3", {}).get("state") != "not run" or not s1.get("work_ok"):
        errs.append(f"script steps / NotRun: exit {rc5}, steps {stp}")
    say("RUNNER SELFTEST " + ("OK" if not errs else f"FAILED: {errs}"))
    return 1 if errs else 0


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if a and a[0] == "--mock-tree":
        print(json.dumps(mock_tree(a[1], a[2], "--with-r1" in a), indent=1)); return 0
    if a and a[0] == "--selftest":
        return selftest(a[1], a[2])
    flags = {x for x in a if x.startswith("--")}; pos = [x for x in a if not x.startswith("--")]
    if len(pos) != 4 or flags - {"--dry-run", "--check-start"}:
        print(__doc__); return EXIT_GATE
    return run_session(*pos, dry="--dry-run" in flags, check_only="--check-start" in flags)


if __name__ == "__main__":
    sys.exit(main())
