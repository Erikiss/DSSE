"""GPU runs of README sections 11 and 12 on Colab (one shard per session, each meant to finish in about 20 min on an A100).
The notebooks (notebooks/build_abschnitt11_gpu_notebooks.py) set up <root> and call
    python run_shard.py <shard> <root> <work> [<drive_dir>]
<root>  models/qwen3-1.7b (HF revision pinned and checked), rsi_dl/persona/*.jsonl, ju_out/{corpus_wikitext,axes_pairs}.json,
        dn_out/{mmlu_topics,dn_text,gsm8k_test,mock_sample}.json, gpu_pkg/rk_dirs.pt
<work>  local working directory; after every step it is copied to <drive_dir> (Google Drive), so a rerun after a disconnect resumes
        (the notebook copies <drive_dir> back first; every step is skipped when its output exists, and rk_loop.py caches inside).
Shards
  gpu1  RSI indicators (rk_loop.py) at block 14: pilot with gates, loops, MANIP retry if needed, K5, K6, block summary
  gpu2  delta per block (dn_delta.py), choice of b_lo and b_hi, RSI indicators at b_lo
  gpu3  the same at b_hi
  gpu4  double negative: traces sampled on the GPU (dn_sample.py, rule unchanged), then rule A4a (dn_tf.py) on that trace set
  gpu5  PC toy (pc_meta.py): BP-muP widths, K 1 controls, delta probe D2, K 3/6, weight decay 0
  gpu6  PC toy: PC-muP (truncated inference, T 20, beta 0.1) and BP-SP widths, K 1 controls"""
import json, math, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); A = os.path.dirname(HERE)            # analysis/ju_space
shard, root, work = sys.argv[1], sys.argv[2], sys.argv[3]; drive = sys.argv[4] if len(sys.argv) > 4 else None
PY = sys.executable; os.makedirs(f"{work}/logs", exist_ok=True); T0 = time.time(); TIMES = {}
NPAR = int(os.environ.get("NPAR", str(max(2, min(10, (os.cpu_count() or 4) - 2)))))


def sync():
    if drive:
        os.makedirs(drive, exist_ok=True); shutil.copytree(work, drive, dirs_exist_ok=True)


def run(name, args, env=None, out=None):
    """One step (skipped if 'out' exists): output streamed and logged, then the work directory is copied to Drive."""
    if out and os.path.exists(out): print(f"[skip] {name}: {os.path.basename(out)} exists", flush=True); return
    e = dict(os.environ, DEVICE="cuda", NT="4", PYTHONUNBUFFERED="1"); e.update(env or {}); t = time.time()
    print(f"[run] {name}", flush=True)
    with open(f"{work}/logs/{name}.log", "a") as lg:
        p = subprocess.Popen([PY, "-u"] + args, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            lg.write(line)
            if "Loading weights" not in line: print(line, end="", flush=True)
        rc = p.wait()
    TIMES[name] = round(time.time() - t, 1); print(f"[done] {name} rc={rc} ({time.time() - t:.0f}s)", flush=True); sync()
    if rc != 0: raise SystemExit(f"step {name} failed (rc {rc}); log: {work}/logs/{name}.log")


# ---------------- section 11: RSI indicators
def rk(block, k6):
    o = f"{work}/rk"; b = f"{o}/b{block}"; env = dict(BLOCK=str(block), POOLDIRS=f"{root}/gpu_pkg/rk_dirs.pt", PERSONA=f"{root}/rsi_dl/persona")
    def stage(st, sc, outp):
        run(f"rk_b{block}_{st}" + ("" if sc == 1 else f"@{sc:.4g}"), [f"{A}/rk_loop.py", root, o, st], dict(env, ASCALE=repr(sc)), outp)
    stage("pilot", 1.0, f"{b}/pilot.json"); scale = 1.0
    if not json.load(open(f"{b}/pilot.json"))["gates"]["GO1"]:          # plan: one retry of GO1 at 1.5 alpha; if it passes, the block runs there
        stage("pilot", 1.5, f"{b}/pilot@1.5.json")
        if json.load(open(f"{b}/pilot@1.5.json"))["gates"]["GO1"]: scale = 1.5
    stage("loops", scale, f"{b}/loops.json")
    L = json.load(open(f"{b}/loops.json"))
    if block == 14 and not all(L[l][0]["p"] <= 0.01 and L[l][0]["c"] > 0 for l in ("RSI_a", "RSI_b")):
        stage("manip", scale * math.sqrt(2), f"{b}/manip@{scale * math.sqrt(2):.4g}.json")     # plan: MANIP retry at alpha * sqrt(2)
    stage("k5", scale, f"{b}/k5.json")
    if k6: stage("k6", scale, f"{b}/k6.json")
    run(f"summary_b{block}", [f"{A}/rk_verdict.py", "block", b, f"{work}/summary_b{block}.json"], out=f"{work}/summary_b{block}.json")


def delta():
    run("dn_delta", [f"{A}/dn_delta.py", root, f"{root}/dn_out/mmlu_topics.json", f"{work}/dn_delta.json"], out=f"{work}/dn_delta.json")
    ch = json.loads(subprocess.check_output([PY, f"{A}/rk_verdict.py", "choose", f"{work}/dn_delta.json"], text=True))
    json.dump(ch, open(f"{work}/blocks_chosen.json", "w"), indent=1); print("blocks chosen:", ch["b_lo"], ch["b_hi"], flush=True); sync(); return ch


# ---------------- section 12: PC toy
PC_ENV = dict(T="20", BETA="0.1", LR="1e-3")                       # plan_v2 amendments: PC with truncated inference


def job_name(j):
    """The file name pc_meta.py gives a job (to skip finished jobs without starting Python)."""
    lr, T, beta = float(j["env"].get("LR", "1e-3")), int(j["env"].get("T", "50")), float(j["env"].get("BETA", "1"))
    return (f"{j['arm']}_W{j['W']}_s{j['seed']}_K{j['K']}_g{j['gamma']:g}_wd{j['wd']:g}" + (f"_lr{lr:g}" if lr != 1e-3 else "")
            + (f"_T{T}" if j["arm"] == "pc_mup" and T != 50 else "") + (f"_beta{beta:g}" if j["arm"] == "pc_mup" and beta != 1 else ""))


def pc_jobs(which):
    J = []
    def add(arm, W, s, K=12, gamma=0.0, wd=1.0):
        J.append(dict(arm=arm, W=W, seed=s, K=K, gamma=gamma, wd=wd, env=dict(PC_ENV if arm == "pc_mup" else {}, MAXSTEPS="6000")))
    if which == "gpu5":
        for W in (1024, 512, 256, 128, 64):
            for s in range(5): add("bp_mup", W, s); add("bp_mup", W, s, K=1)
        for g in (1.0, 0.3, 0.1, 0.03):
            for s in range(5): add("bp_mup", 256, s, gamma=g); add("bp_mup", 256, s, gamma=-g)
        for W in (1024, 256, 64):
            for s in range(5):
                for K in (3, 6): add("bp_mup", W, s, K=K)
                add("bp_mup", W, s, wd=0.0)
    else:
        for W in (512, 256, 128, 64):
            for s in range(5): add("pc_mup", W, s); add("pc_mup", W, s, K=1)
        for W in (1024, 512, 256, 128, 64):
            for s in range(5): add("bp_sp", W, s); add("bp_sp", W, s, K=1)
    return J


def pc(which):
    o = f"{work}/pc"; os.makedirs(f"{o}/jobs", exist_ok=True); jobs = pc_jobs(which); t = time.time()
    for phase in (1, 2):                                              # the K = 1 controls need their K = 12 run (same shard)
        todo = [j for j in jobs if (j["K"] == 1) == (phase == 2) and not os.path.exists(f"{o}/jobs/{job_name(j)}.json")]
        print(f"[pc] phase {phase}: {len(todo)} jobs, {NPAR} in parallel", flush=True); running = []; done = 0
        while todo or running:
            while todo and len(running) < NPAR:
                j = todo.pop(0); lg = open(f"{work}/logs/pc_{job_name(j)}.log", "a")
                e = dict(os.environ, DEVICE="cuda", NT="1", PYTHONUNBUFFERED="1", **j["env"])
                args = [PY, "-u", f"{A}/pc_meta.py", o, j["arm"], str(j["W"]), str(j["seed"]), str(j["K"]), repr(j["gamma"]), repr(j["wd"])]
                running.append((subprocess.Popen(args, env=e, stdout=lg, stderr=subprocess.STDOUT), j, lg))
            time.sleep(0.5)
            for r in list(running):
                if r[0].poll() is not None:
                    running.remove(r); r[2].close(); done += 1
                    ok = os.path.exists(f"{o}/jobs/{job_name(r[1])}.json")
                    print(f"[pc] {done} done ({time.time() - t:.0f}s): {job_name(r[1])} {'ok' if ok else 'FAILED rc=' + str(r[0].returncode)}", flush=True)
            if done and done % 20 == 0: sync()
        sync()
    TIMES["pc_jobs"] = round(time.time() - t, 1)
    run("pc_summary", [f"{A}/pc_summary.py", o, f"{work}/pc_summary_{which}.json"])


def main():
    import torch, transformers
    info = dict(shard=shard, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, torch=torch.__version__, transformers=transformers.__version__)
    print(json.dumps(info), flush=True)
    if shard == "gpu1": rk(14, k6=True)
    elif shard in ("gpu2", "gpu3"):
        ch = delta(); rk(ch["b_lo"] if shard == "gpu2" else ch["b_hi"], k6=False)
    elif shard == "gpu4":
        sp = f"{work}/dn_sample.json"                                  # dn_sample.py resumes finished batches; complete when it holds 'trace_set'
        if not (os.path.exists(sp) and "trace_set" in json.load(open(sp))):
            run("dn_sample", [f"{A}/dn_sample.py", root, f"{root}/dn_out/gsm8k_test.json", sp])
        S = json.load(open(sp)); print(f"trace set {S['trace_set']} (complete {S['trace_set_complete']}, L used {S['L_used']})", flush=True)
        run("dn_tf", [f"{A}/dn_tf.py", root, sp, f"{root}/dn_out/dn_text.json", f"{work}/dn_tf.json"], out=f"{work}/dn_tf.json")
    elif shard in ("gpu5", "gpu6"): pc(shard)
    else: raise SystemExit(f"unknown shard {shard}")
    info.update(seconds=round(time.time() - T0, 1), steps=TIMES); json.dump(info, open(f"{work}/shard_done.json", "w"), indent=1); sync()
    print(f"shard {shard} finished in {time.time() - T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
