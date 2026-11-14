"""Part L MOCK self-test (CPU; MOCK=1, rk_loop's tiny random model): every L step through the session runner (gates, start conditions,
SG6 provenance checks, step records), with the non-L upstream sessions declared 'not run' so that L's MOCK stand-ins are used; then the
P0-L pilot steps and the L-CAL code test (quick sizes).
    MOCK=1 taskset -c 0 python3 r1b_l_selftest.py <dir> <root>"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["MOCK"] = "1"
import r1b_core as K
import r1b_run_session as RS

PY = sys.executable


def main(d, root):
    d = os.path.abspath(d)
    t = RS.mock_tree(d, root)
    for acct, sess in (("B", ("r1b_K_S1", "r1b_R_S0", "r1b_D_A", "r1b_K_S2a")),
                       ("A", ("r1b_R_ana", "r1b_D_ana", "r1b_K_ana", "r1b_K_S2a", "r1b_R_S5"))):
        K.dump_json(dict(sessions={s: "declared not run (L self-test)" for s in sess}), os.path.join(t[acct], "r1b_declared_not_run.json"))
        for s in sess:                                    # producers recorded as 'not run' (the runner's own marker)
            K.dump_json(dict(reason="L self-test: non-L producer not run", code="NA:not_run:L_selftest"), os.path.join(t[acct], s, "session_not_run.json"))
    env = dict(os.environ, MOCK="1", RUN_DEVICE="cpu", DEVICE="cpu", R1B_FREEZE=t["freeze"], PYTHONUNBUFFERED="1")
    errs = []
    for sess, acct in (("r1b_L_A", "B"), ("r1b_L_B", "B"), ("r1b_ana", "A")):
        work = os.path.join(d, "work", sess)
        r = subprocess.run([PY, "-u", os.path.join(HERE, "r1b_run_session.py"), sess, root, work, t[acct]], env=dict(env, R1B_ACCOUNT=acct),
                           cwd=HERE, capture_output=True, text=True)
        open(os.path.join(d, f"{sess}.log"), "w").write(r.stdout + r.stderr)
        recs = {}
        for f in sorted(os.listdir(os.path.join(work, "steps"))) if os.path.isdir(os.path.join(work, "steps")) else []:
            m = K.load_json(os.path.join(work, "steps", f)) or {}
            if m.get("part") == "L" or sess != "r1b_ana":
                recs[m.get("name", f)] = m.get("status")
        bad = {k: v for k, v in recs.items() if v not in ("ok",) and not k.startswith("_")}
        print(f"{sess}: exit {r.returncode}; L steps {len(recs)}; not ok: {bad}", flush=True)
        if r.returncode not in (0,) or bad:
            errs.append(f"{sess}: exit {r.returncode}, {bad}")
    v = K.load_json(os.path.join(d, "work", "r1b_ana", "l_verdicts.json")) or {}
    print("L-ANA answer:", (v.get("combined") or {}).get("answer"), flush=True)
    # P0-L pilot steps (direct context, pilot whitelist enforced by write_json)
    import r1b_l_pilot as PL
    c = K.Ctx("r1b_P0", os.path.join(d, "work", "r1b_P0"), root=root, device="cpu", mock=True, freeze=dict(funded=dict(tier2=True)))
    for fn in (PL.l0_a, PL.l0_b):
        c.step, c.step_part = fn.__name__, "main"
        fn(c)
        p = c.path(f"pilot/{fn.__name__}.json")
        prob = K.check_output(p)
        print(f"r1b_P0/{fn.__name__}: {'ok' if not prob else prob}", flush=True)
        if prob:
            errs.append(prob)
    # L-CAL code test
    r = subprocess.run([PY, "-u", os.path.join(HERE, "r1b_l_cal.py"), "run", os.path.join(d, "l_cal_quick.json"), "--quick"], env=env, cwd=HERE,
                       capture_output=True, text=True)
    print("L-CAL quick:", "ok" if r.returncode == 0 else r.stderr[-800:], flush=True)
    if r.returncode:
        errs.append("L-CAL quick failed")
    print("L SELFTEST " + ("OK" if not errs else "FAILED: " + "; ".join(errs)))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
