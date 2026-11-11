"""Keeps every CPU core busy with one single-threaded job. Reads the job list again every few seconds (lines can be added or
reordered while it runs; earlier lines start first). A job is one line:
    <name> grok <mode> <seed> <max_steps> <weight_decay> [ENV=value ...]      a training run of kv_grok.py
    <name> py <script.py> [ENV=value ...] <arguments ...>                     any other script of this directory
A job is started on a core as soon as no kv_*.py process is pinned to that core; a job whose log file exists is skipped.
Usage: python kv_queue.py <jobs.txt> <out_dir> <log_dir> [cores, default 0,1,2,3]      env: RUNCWD (working directory of the jobs)"""
import os, re, subprocess, sys, time

jobs, out, logs = sys.argv[1], sys.argv[2], sys.argv[3]; cores = [int(c) for c in (sys.argv[4] if len(sys.argv) > 4 else "0,1,2,3").split(",")]; here = os.path.dirname(os.path.abspath(__file__)); mine = {}; isenv = lambda t: re.match(r"^[A-Z_]+=", t)


def busy():
    b = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit(): continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="ignore")
            if "/kv_" in cmd and "python" in cmd and "kv_queue.py" not in cmd:
                aff = os.sched_getaffinity(int(pid))
                if len(aff) == 1: b |= aff
        except Exception: pass
    return b


idle = 0
while True:
    todo = [ln.split() for ln in open(jobs).read().splitlines() if ln.strip() and not ln.startswith("#")]; todo = [j for j in todo if not os.path.exists(f"{logs}/{j[0]}.log")]
    for c in [c for c in cores if c not in busy()]:
        if not todo: break
        j = todo.pop(0); name, kind = j[0], j[1]; rest = j[2:]; env = dict(os.environ, NT="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", **dict(t.split("=", 1) for t in rest if isenv(t))); args = [t for t in rest if not isenv(t)]
        cmd = [sys.executable, f"{here}/kv_grok.py", out] + args if kind == "grok" else [sys.executable, f"{here}/{args[0]}"] + args[1:]
        mine[name] = subprocess.Popen(["taskset", "-c", str(c)] + cmd, env=env, stdout=open(f"{logs}/{name}.log", "w"), stderr=subprocess.STDOUT, cwd=os.environ.get("RUNCWD", "."))
        print(time.strftime("%H:%M:%S"), "started", name, "on core", c, flush=True); time.sleep(2)
    running = [n for n, p in mine.items() if p.poll() is None]; idle = idle + 1 if not running and not todo else 0
    if idle > 120: print("queue empty for 10 minutes, stopping", flush=True); break
    time.sleep(5)
