"""Round 1b slot runner (main §4.4, §4.7): runs the sessions of one slot in order in one notebook runtime, each as its own process
(r1b_run_session.py), waiting for each session's start conditions by polling Drive (with a timeout). The operator starts only the
slot notebooks: P0 (pilot, account B), A1, A2 (account A), B1, B2 (account B), A_T2, B_T2 (tier 2), ANA (r1b_ana, CPU, account A).

    python r1b_run_slot.py <slot> <root> <workbase> <runs> [--poll-s 120] [--wait-h 10] [--runtime-h 23]
    python r1b_run_slot.py <slot> <root> <workbase> <runs> --copy-only    B -> A copies of finished folders the slot reads; status
    python r1b_run_slot.py <slot> <root> <selftest_dir> --selftest        MOCK run of the slot's sessions and their upstream (mock tree)
    python r1b_run_slot.py --mock-dry-run <dir> <root>                    MOCK run of every session of every slot, one process at a time

Per session: done or 'not run' -> next; start conditions not met -> wait (poll every --poll-s seconds; after --wait-h hours exit 4,
the notebook is run again later); then the session process (exit 0 done, 5 not run -> next; 3 time guard -> the continuation starts
at once; 1 failed -> one more attempt, then the slot moves on and reports it (the operator may declare the session 'not run' after two
attempts in dsse_runs/r1b_declared_not_run.json); 2 gate -> the slot stops). --runtime-h: no session (or continuation) starts when its
planned minutes would end beyond this many hours of the runtime (exit 3: run the notebook again)."""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import r1b_contracts as C
import r1b_run_session as R
from r1b_core import load_json

PY, RUNNER = sys.executable, os.path.join(HERE, "r1b_run_session.py")
T_START = float(os.environ.get("R1B_RUNTIME_T0") or time.time())     # the notebook's first cell


def say(msg):
    print(f"[slot {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def opt(args, name, default):
    return float(args[args.index(name) + 1]) if name in args else default


def run_one(s, root, work, runs, env):
    return subprocess.run([PY, "-u", RUNNER, s, root, work, runs], env=dict(env, R1B_T0=repr(time.time()))).returncode


def run_slot(slot, root, workbase, runs, poll_s=120.0, wait_h=10.0, runtime_h=23.0):
    d = C.SLOTS[slot]; acct = d["account"]; P = R.Paths(runs, acct)
    env = dict(os.environ, R1B_ACCOUNT=acct, R1B_SLOT=slot, RUN_DEVICE="cpu" if d["runtime"] == "cpu" else os.environ.get("RUN_DEVICE", "cuda"))
    fz = load_json(R.freeze_path(), {}) or {}
    rc_all, report = 0, {}
    for s in d["sessions"]:
        fails, t_wait = 0, time.time()
        while True:
            st, info = R.resolve(P, s)
            if st in ("done", "not_run", "declared"):
                report[s] = st if st == "done" else f"{st} ({info})"; break
            chk = R.check_start(s, P, fz)
            if chk["not_run"]:
                R.write_not_run(P, s, chk["not_run"]); report[s] = f"not run ({chk['not_run']})"; break
            if not chk["ok"]:
                if time.time() - t_wait > wait_h * 3600:
                    say(f"{s}: still waiting after {wait_h:g} h: {chk['waiting']}; run the notebook again later"); return 4, report
                say(f"{s} waits: " + "; ".join(chk["waiting"][:4]) + f" (next check in {poll_s:.0f}s)")
                time.sleep(poll_s); continue
            if (time.time() - T_START) / 3600 + C.SESSIONS[s]["minutes"] / 60 > runtime_h:
                say(f"{s}: would end beyond {runtime_h:g} h of this runtime; run the notebook again (finished sessions are skipped)")
                return 3, report
            say(f"=== {s} ({C.SESSIONS[s]['content'][:90]})")
            rc = run_one(s, root, os.path.join(workbase, s), runs, env)
            if rc in (R.EXIT_DONE, R.EXIT_NOT_RUN):
                report[s] = "done" if rc == R.EXIT_DONE else "not run"; break
            if rc == R.EXIT_GUARD:
                say(f"{s}: time guard; the continuation starts now"); continue
            if rc == R.EXIT_WAIT:
                time.sleep(poll_s); continue
            if rc == R.EXIT_GATE:
                say(f"{s}: a gate failed (see its manifest.json); the slot stops"); report[s] = "gate failed"; return 2, report
            fails += 1
            if fails >= 2:
                say(f"{s}: failed twice; the slot moves on. After checking the logs the operator may declare it 'not run' in "
                    f"dsse_runs/{C.DECLARED} so that its consumers can start")
                report[s] = "failed twice"; rc_all = 1; break
            say(f"{s}: steps failed; second attempt")
    return rc_all, report


def copy_only(slot, runs):
    """The notebook's copy cell: in account A, finished B folders that the slot's sessions read are copied from dsse_runs_B; status."""
    d = C.SLOTS[slot]; P = R.Paths(runs, d["account"])
    if d["account"] == "A" and not os.path.isdir(P.runs_B):
        say(f"note: {P.runs_B} is missing (share B's dsse_runs with A and add it to My Drive as the shortcut dsse_runs_B)")
    if d["account"] == "B" and not os.path.isdir(P.xfer_src):
        say(f"note: {P.xfer_src} is missing (share A's dsse_runs/r1b_xfer with B and add it to My Drive as the shortcut r1b_xfer_A)")
    for s in d["sessions"]:
        for u in C.upstream(s):
            st, _ = R.resolve(P, u)
            say(f"{s} <- {u}: {st}")
    return 0


def mock_dry_run(d, root, only=None):
    """Every session of every slot (or only those in `only`) under MOCK=1 on a mock tree, scheduled slot by slot as their start
    conditions allow, one session process at a time; fails on a failed session or a deadlock."""
    t = R.mock_tree(d, root)
    env = dict(os.environ, MOCK="1", R1B_FREEZE=t["freeze"], RUN_DEVICE=os.environ.get("RUN_DEVICE", "cpu"))
    pos, res, t0 = {sl: 0 for sl in C.SLOTS}, {}, time.time()
    while True:
        moved = False
        for sl, sd in C.SLOTS.items():
            while pos[sl] < len(sd["sessions"]):
                s = sd["sessions"][pos[sl]]
                if only is not None and s not in only:
                    pos[sl] += 1; moved = True; continue
                runs = t[sd["account"]]; P = R.Paths(runs, sd["account"])
                st, info = R.resolve(P, s)
                if st in ("done", "not_run", "declared"):
                    res[s] = st; pos[sl] += 1; moved = True; continue
                chk = R.check_start(s, P, load_json(t["freeze"], {}))
                if chk["not_run"]:
                    R.write_not_run(P, s, chk["not_run"]); res[s] = "not run"; pos[sl] += 1; moved = True; continue
                if not chk["ok"]:
                    break
                rc = run_one(s, root, os.path.join(d, "work", sd["account"], s), runs,
                             dict(env, R1B_ACCOUNT=sd["account"], R1B_SLOT=sl, RUN_DEVICE="cpu" if sd["runtime"] == "cpu" else env["RUN_DEVICE"]))
                if rc not in (R.EXIT_DONE, R.EXIT_NOT_RUN, R.EXIT_GUARD):
                    say(f"MOCK DRY RUN FAILED: {s} exit code {rc}"); return 1
                moved = True
                if rc != R.EXIT_GUARD:
                    res[s] = "done" if rc == R.EXIT_DONE else "not run"; pos[sl] += 1
                break
        if all(pos[sl] >= len(sd["sessions"]) for sl, sd in C.SLOTS.items()):
            break
        if not moved:
            say(f"MOCK DRY RUN FAILED: deadlock at {[sd['sessions'][pos[sl]] for sl, sd in C.SLOTS.items() if pos[sl] < len(sd['sessions'])]}")
            return 1
    bad = [s for s, v in res.items() if v not in ("done", "not run")]
    say(f"MOCK DRY RUN {'OK' if not bad else 'FAILED'}: {sum(v == 'done' for v in res.values())} sessions done, "
        f"{sum(v == 'not run' for v in res.values())} not run, in {time.time() - t0:.0f}s")
    return 1 if bad else 0


def closure(sessions):
    out, todo = set(), list(sessions)
    while todo:
        s = todo.pop()
        if s not in out:
            out.add(s); todo += C.upstream(s)
    return out


def main(a):
    if a and a[0] == "--mock-dry-run":
        return mock_dry_run(a[1], a[2])
    if len(a) < 4 or a[0] not in C.SLOTS:
        print(__doc__); return 2
    slot, root = a[0], a[1]
    if "--selftest" in a:
        return mock_dry_run(a[2], root, only=closure(C.SLOTS[slot]["sessions"]))
    if "--copy-only" in a:
        return copy_only(slot, a[3])
    rc, rep = run_slot(slot, root, a[2], a[3], opt(a, "--poll-s", 120.0), opt(a, "--wait-h", 10.0), opt(a, "--runtime-h", 23.0))
    say("slot " + slot + ": " + ", ".join(f"{k} {v}" for k, v in rep.items()))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
