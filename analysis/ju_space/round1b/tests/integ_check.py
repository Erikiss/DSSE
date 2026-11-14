"""Summarise the integration tree (scratch): per session state, step counts, 'not run' steps with codes, labels that hint at glue
problems (stand-ins, placeholders, input missing). usage: integ_check.py <tree dir>"""
import glob, json, os, sys

d = sys.argv[1]
for acct in ("A", "B"):
    runs = os.path.join(d, acct, "MyDrive", "dsse_runs")
    for sd in sorted(glob.glob(os.path.join(runs, "r1b_*"))):
        s = os.path.basename(sd)
        if s == "r1b_xfer":
            continue
        done = os.path.join(sd, "session_done.json"); nr = os.path.join(sd, "session_not_run.json")
        if os.path.exists(nr):
            print(f"{acct} {s}: NOT RUN {json.load(open(nr)).get('reason')}"); continue
        if not os.path.exists(done):
            print(f"{acct} {s}: {'partial' if os.path.exists(os.path.join(sd, 'session_partial.json')) else 'running/absent'}"); continue
        r = json.load(open(done)); st = r.get("steps") or {}
        if acct == "A" and r.get("account") == "B":
            continue                                   # B folder copied into A
        hints = set()
        for p in glob.glob(os.path.join(sd, "**", "*.json"), recursive=True):
            if "/steps/" in p or "/in/" in p:
                continue
            try:
                t = open(p).read()
            except OSError:
                continue
            for h in ("MOCK stand-in", "input missing", "placeholder", "not copied"):
                if h in t:
                    hints.add(h)
        nrs = {k: v for k, v in (r.get("not_run") or {}).items()}
        print(f"{acct} {s}: done; steps {len(st)}; not run {len(nrs)}" + (f" {nrs}" if nrs else "") + (f"; hints {sorted(hints)}" if hints else "")
              + (f"; mock-filled {len(json.load(open(os.path.join(sd, 'steps', '_mock_fill.json')))['outputs'])}"
                 if os.path.exists(os.path.join(sd, "steps", "_mock_fill.json")) else ""))
