"""Integration test driver (scratch, MOCK only): r1b_run_slot.mock_dry_run's schedule (every slot, one session process at a time) on a
mock two-account tree with rk_loop MOCK round-1 states, per-session logs and times, and the MOCK-only trigger forcing of R's self-test
(rg11 'refit', rg15 J1_and_J2) so that every tier-2 session runs.  usage: integ_run.py <dir> <root> [session ...]  (sessions: only these)
integrator 2: INTEG_BUDGET_S (default 300): no session starts after this many seconds (exit 3, rerun to resume)."""
import json, os, subprocess, sys, time

RB = "/home/user/DSSE/analysis/ju_space/round1b"
sys.path.insert(0, RB)
os.environ["MOCK"] = "1"
import r1b_contracts as C
import r1b_core as K
import r1b_run_session as R

d, root = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
only = set(sys.argv[3:]) or None
env0 = dict(os.environ, MOCK="1", NT="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1", RUN_DEVICE="cpu", DEVICE="cpu",
            R1B_MOCK_WITH_R1="1", POOLDIRS=os.path.join(root, "gpu_pkg", "rk_dirs.pt"), PERSONA=os.path.join(root, "rsi_dl", "persona"),
            R1B_R_FORCE_GATES="1", R1B_D_FORCE_R2="1")
os.environ.update(env0)
t = R.mock_tree(d, root, with_r1=True)
env0["R1B_FREEZE"] = t["freeze"]
LOG = os.path.join(d, "logs"); os.makedirs(LOG, exist_ok=True)
resf = os.path.join(d, "integ_results.json")
res = K.load_json(resf, {}) or {}
forced = set(res.get("_forced", []))


def say(m):
    print(f"[integ {time.strftime('%H:%M:%S')}] {m}", flush=True)


def force():
    for fid, key, v in (("r1b_R_S5/rg11.json", "decision", "refit"), ("r1b_R_ana/rg15.json", "J1_and_J2", True)):
        s, rel = fid.split("/")
        p = os.path.join(t["A"], fid)
        if fid in forced or not os.path.exists(os.path.join(t["A"], s, "session_done.json")) or not os.path.exists(p):
            continue
        j = json.load(open(p)); old = j.get(key); j[key] = v; json.dump(j, open(p, "w"))
        for sd in (os.path.join(t["A"], s), os.path.join(d, "work", "A", s)):
            rec = os.path.join(sd, "steps")
            for f in os.listdir(rec) if os.path.isdir(rec) else []:
                r = json.load(open(os.path.join(rec, f)))
                if rel in (r.get("outputs") or {}):
                    r["outputs"][rel]["sha256"] = K.sha256_file(p); json.dump(r, open(os.path.join(rec, f), "w"))
            sdj = os.path.join(sd, "session_done.json")
            if os.path.exists(sdj):
                j2 = json.load(open(sdj))
                if rel in (j2.get("files") or {}):
                    j2["files"][rel] = K.sha256_file(p); json.dump(j2, open(sdj, "w"))
        forced.add(fid); say(f"MOCK trigger forced: {fid} {key}: {old!r} -> {v!r}")


pos, t0 = {sl: 0 for sl in C.SLOTS}, time.time()
while True:
    moved = False
    force()
    for sl, sd in C.SLOTS.items():
        while pos[sl] < len(sd["sessions"]):
            s = sd["sessions"][pos[sl]]
            if only is not None and s not in only:
                pos[sl] += 1; moved = True; continue
            runs = t[sd["account"]]; P = R.Paths(runs, sd["account"])
            st, info = R.resolve(P, s)
            if st in ("done", "not_run", "declared"):
                res.setdefault(s, {}).setdefault("state", st); pos[sl] += 1; moved = True; continue
            chk = R.check_start(s, P, K.load_json(t["freeze"], {}))
            if chk["not_run"]:
                R.write_not_run(P, s, chk["not_run"]); res[s] = dict(state="not run", reason=chk["not_run"]); pos[sl] += 1; moved = True; continue
            if not chk["ok"]:
                break
            if time.time() - t0 > float(os.environ.get("INTEG_BUDGET_S", "300")):
                res["_forced"] = sorted(forced); json.dump(res, open(resf, "w"), indent=1)
                say(f"budget reached before {s}; rerun to resume"); sys.exit(3)
            a = time.time(); say(f"start {s} (slot {sl})")
            with open(os.path.join(LOG, s + ".log"), "a") as f:
                rc = subprocess.run([sys.executable, "-u", os.path.join(RB, "r1b_run_session.py"), s, root, os.path.join(d, "work", sd["account"], s), runs],
                                    env=dict(env0, R1B_ACCOUNT=sd["account"], R1B_SLOT=sl, R1B_T0=repr(time.time())), stdout=f, stderr=subprocess.STDOUT).returncode
            e = res.setdefault(s, dict(seconds=0.0, runs=[]))
            e["seconds"] = round(e.get("seconds", 0) + time.time() - a, 1); e.setdefault("runs", []).append(rc)
            say(f"end {s}: exit {rc} after {time.time() - a:.0f}s")
            res["_forced"] = sorted(forced); json.dump(res, open(resf, "w"), indent=1)
            moved = True
            if rc not in (R.EXIT_DONE, R.EXIT_NOT_RUN, R.EXIT_GUARD):
                e["state"] = f"exit {rc}"; json.dump(res, open(resf, "w"), indent=1)
                say(f"INTEG FAILED at {s} (exit {rc}); see {LOG}/{s}.log"); sys.exit(1)
            if rc != R.EXIT_GUARD:
                e["state"] = "done" if rc == R.EXIT_DONE else "not run"; pos[sl] += 1
                res["_forced"] = sorted(forced); json.dump(res, open(resf, "w"), indent=1)
            break
    if all(pos[sl] >= len(sd["sessions"]) for sl, sd in C.SLOTS.items()):
        break
    if not moved:
        say(f"INTEG FAILED: deadlock at {[sd['sessions'][pos[sl]] for sl, sd in C.SLOTS.items() if pos[sl] < len(sd['sessions'])]}"); sys.exit(1)
res["_forced"] = sorted(forced); res["_total_s"] = round(time.time() - t0, 1)
json.dump(res, open(resf, "w"), indent=1)
say(f"INTEG OK: {sum(1 for k, v in res.items() if not k.startswith('_') and v.get('state') == 'done')} done, "
    f"{sum(1 for k, v in res.items() if not k.startswith('_') and v.get('state') in ('not run', 'not_run'))} not run, {time.time() - t0:.0f}s")
