"""r1b_R_S1 (part R, R1): round-1-identical replay with the byte-identical rk_loop.py. Closed loop: the stages run as subprocesses as
run_shard.rk ran them (python rk_loop.py <root> <work>/r1b_replay/rk <stage>, env BLOCK, ASCALE, POOLDIRS, PERSONA) on a reduced copy
of R1.state[b] ('ref', 'axis_*', 'axes', 'alpha' only). Open loop: rk_loop imported as a library on a second reduced copy (one process
per block), rk.evaluate along round 1's stored directions d_1..d_3. Compare: token identity (tau by versions), paired tolerance on the
open-loop rows (|replay - round 1|, stats-16), gates and PC_PASS; round-1 strings and 1b statistics side by side (descriptive)."""
import json, math, os, subprocess, sys, time
import numpy as np
import r1b_core as K
import r1b_r_lib as R
import r1b_r_stats as S

STAGES = {"pilot1": ("pilot", "1"), "pilot1.5": ("pilot", "1.5"), "loops1.5": ("loops", "1.5"), "loops1": ("loops", "1")}


def reduced(ctx, b, sub):
    t = R.torch(); d = os.path.join(ctx.work, sub, "rk", f"b{b}"); p = os.path.join(d, "state.pt")
    if not os.path.exists(p):
        st = R.r1_state(ctx, b); os.makedirs(d, exist_ok=True)
        t.save({k: v for k, v in st.items() if k in ("ref", "axes", "alpha") or k.startswith("axis_")}, p + ".tmp"); os.replace(p + ".tmp", p)
    return os.path.join(ctx.work, sub, "rk")


def _env(ctx, b, ascale):
    pool = os.path.join(ctx.root, "gpu_pkg", "rk_dirs.pt")
    return dict(os.environ, BLOCK=str(b), ASCALE=ascale, POOLDIRS=os.environ.get("POOLDIRS") or (pool if os.path.exists(pool) else
                os.path.join(K.AJS, "gpu", "data", "rk_dirs.pt")), PERSONA=os.environ.get("PERSONA") or os.path.join(ctx.root, "rsi_dl", "persona"))


def closed(ctx, b, stg):
    stage, asc = STAGES[stg]; out = reduced(ctx, b, "r1b_replay"); t0 = time.time()
    r = subprocess.run([sys.executable, R.RK_SRC, ctx.root, out, stage], env=_env(ctx, b, asc), capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    if r.returncode:
        raise RuntimeError(f"rk_loop {stage} at block {b} exited {r.returncode}")
    bd = os.path.join(out, f"b{b}"); files = {f: K.sha256_file(os.path.join(bd, f)) for f in sorted(os.listdir(bd)) if f.endswith(".json")}
    R.wj(ctx, f"r1/closed_b{b}_{stg}.json", dict(block=b, stage=stage, ascale=float(asc), seconds=time.time() - t0, files=files,
                                                      rk_loop_sha256=K.sha256_file(R.RK_SRC)), config=dict(block=b, stage=stage, ascale=asc),
                   seeds=dict(keys="round-1 keys keyseed(L + SUF, n, p, s, b), 'mirror' (stated reuse)"))


def open_loop(ctx, b):
    """rk.evaluate(L + SUF, n, d = R1 dir_<L><SUF>_<n>, nseed, alpha_b) for n = 1, 2, 3 and every round-1 loop at block b."""
    meta = R.r1_loops(ctx, b)["_meta"]; out = reduced(ctx, b, "r1b_replay_open"); os.environ.update(_env(ctx, b, repr(float(meta["alpha_scale"]))))
    argv = list(sys.argv); sys.argv = [R.RK_SRC, ctx.root, out, "lib"]
    try:
        import rk_loop as rk
    finally:
        sys.argv = argv
    st = R.r1_state(ctx, b); alpha = float(meta["alpha"]); rows = {}
    assert abs(rk.ST["alpha"]["alpha"] * rk.ASCALE - alpha) < 1e-9 * max(1.0, alpha), "alpha_b differs from loops.json"
    for L in [x for x in R.r1_loops(ctx, b) if x != "_meta"]:
        nseed = 2 if L.startswith("RSI") else 1; rows[L] = []
        for n in (1, 2, 3):
            key = f"dir_{L}{rk.SUF}_{n}"
            if key not in st:
                continue
            ev = rk.evaluate(L + rk.SUF, n, st[key].double(), nseed, alpha); s, _ = rk.stats(ev)
            if L.startswith("SEA"):
                s["sea"] = rk.stats(ev, "sea")[0]
            rows[L].append(dict(n=n, **{k: v for k, v in s.items() if isinstance(v, (int, float, dict))}))
    R.wj(ctx, f"r1/open_b{b}.json", dict(block=b, alpha=alpha, suffix=rk.SUF, rows=rows, state=os.path.relpath(os.path.join(out, f"b{b}", "state.pt"), ctx.work)),
                   config=dict(block=b, alpha=alpha), seeds=dict(keys="round-1 keys (stated reuse)"))


def _versions_tau(ctx):
    v = K.versions(); out = {}
    for b in R.BLOCKS:
        sd = K.load_json(R.r1_file(ctx, b, "shard_done.json"), {}) or {}
        same = dict(gpu=str(sd.get("GPU") or sd.get("gpu")) == str(v.get("gpu")), torch=str(sd.get("torch")) == str(v.get("torch")),
                    transformers=str(sd.get("transformers")) == str(v.get("transformers")), cuda=str(sd.get("CUDA") or sd.get("cuda")) == str(v.get("cuda")))
        tau = 0.95 if all(same.values()) else (0.85 if not same["gpu"] and all(same[k] for k in ("torch", "transformers", "cuda")) else 0.80)
        out[b] = dict(tau=tau, same=same, label=None if tau >= 0.85 else "software versions differ")
    return out


def compare(ctx):
    """R1 rule: 'Replay reproduces round 1' iff (i) token identity >= tau in every evaluation pushed along a stored round-1 vector (pilots,
    round 0, open-loop rounds 1-3), (ii) |replay - round 1| of c and rel <= max(2.5 SD_diff, 1e-4) in >= 95 % of the open-loop rows and
    <= 4 SD_diff in every row (SD_diff: prompt-bootstrap SD of the paired difference), (iii) rk_verdict's gates and PC_PASS equal round 1's."""
    import rk_verdict
    t = R.torch(); taus = _versions_tau(ctx); res = dict(blocks={}); fails = []
    for b in R.BLOCKS:
        st1 = R.r1_state(ctx, b); meta = R.r1_loops(ctx, b)["_meta"]; suf = meta.get("suffix", ""); NP = len(st1["axes"]["fold"]); H = S.halvings(NP)
        rp = os.path.join(ctx.work, "r1b_replay", "rk", f"b{b}"); op = os.path.join(ctx.work, "r1b_replay_open", "rk", f"b{b}")
        stc, sto = (R.load_pt(os.path.join(p, "state.pt")) if os.path.exists(os.path.join(p, "state.pt")) else {} for p in (rp, op))
        ident, tol_rows, W = {}, [], t.as_tensor(K.prompt_bootstrap(NP, B=2000, seed=1001), dtype=t.float64)
        ax = dict(u=st1["axes"]["u_T_perp"].double()[None], aA=st1["axes"]["u_T_perp"].double()[None], aB=st1["axes"]["u_T_perp"].double()[None],
                  kappa=t.ones(1, dtype=t.float64))
        for k, e1 in st1.items():
            if not k.startswith("ev_"):
                continue
            n = int(k.rsplit("_", 1)[1]); e2 = (sto if n >= 1 and k in sto else stc).get(k)
            if e2 is None or (n == 0 and k not in stc):
                continue
            same = [list(map(int, a)) == list(map(int, c)) for a, c in zip(e1["conts"], e2["conts"])]; ident[k] = float(np.mean(same))
            if n >= 1:
                r1, r2 = (S.Row(e["delta"], e["ok"], e["prompt_of"], NP) for e in (e1, e2))
                p1, p2 = S.readout(r1, ax, st1["axes"]["fold"], H), S.readout(r2, ax, st1["axes"]["fold"], H)
                axW = {kk: (v.expand(W.shape[0], -1) if v.dim() == 2 else v.expand(W.shape[0])) for kk, v in ax.items()}
                b1, b2 = S.readout(r1, axW, st1["axes"]["fold"], H, W=W), S.readout(r2, axW, st1["axes"]["fold"], H, W=W)
                for q in ("c", "rel"):
                    sd = float((b2[q] - b1[q]).std()); dif = abs(p2[q] - p1[q])
                    tol_rows.append(dict(ev=k, stat=q, diff=dif, sd_diff=sd, in_25=dif <= max(2.5 * sd, 1e-4), in_4=dif <= max(4 * sd, 1e-4)))
        tau = taus[b]["tau"]; i_ok = bool(ident) and all(v >= tau for v in ident.values())
        ii_ok = bool(tol_rows) and np.mean([r["in_25"] for r in tol_rows]) >= 0.95 and all(r["in_4"] for r in tol_rows)
        g1 = rk_verdict.gates(os.path.dirname(R.r1_file(ctx, b, "rk/b{b}/pilot.json")))["gates"]
        g2 = rk_verdict.gates(rp)["gates"] if os.path.exists(os.path.join(rp, "pilot.json")) else None
        pc1 = rk_verdict.pc_pass(st1, {k: v for k, v in R.r1_loops(ctx, b).items() if k != "_meta"}, suf)[0]
        pc2 = None
        if sto:
            L2 = {"SEA": []}; sea = K.load_json(ctx.path(f"r1/open_b{b}.json"))["rows"].get("SEA", [])
            if len(sea) == 3:
                L2["SEA"] = [dict(rel=1.0)] + [dict(rel=r["rel"], sea=r["sea"]) for r in sea]
                d3, d4 = (R.delta_of(sto[f"ev_SEA{suf}_{n}"])[sto[f"ev_SEA{suf}_{n}"]["ok"]].mean(0) for n in (2, 3))
                s4 = S.cos(d4, d3) / math.sqrt(L2["SEA"][2]["rel"] * L2["SEA"][3]["rel"])
                pc2 = all(s["sea"]["p"] <= 0.01 and s["sea"]["c"] > 0 and s["sea"]["rel"] >= 0.5 for s in L2["SEA"][1:]) and s4 >= 0.5
        iii_ok = g2 is not None and {k: bool(v) for k, v in g1.items()} == {k: bool(v) for k, v in g2.items()} and pc1 == pc2
        miss = [x for x, ok in (("token identity", i_ok), ("paired tolerance", ii_ok), ("gates / PC_PASS", iii_ok)) if not ok]
        verdict = "Replay reproduces round 1" if not miss else f"replay differs ({', '.join(miss)})"
        try:
            v_rep = rk_verdict.block_verdict(rp) if os.path.exists(os.path.join(rp, "loops.json")) else K.coded("NA:not_run:closed loop missing")
        except Exception as e:
            v_rep = K.coded("NaN:block_verdict_failed"); fails.append(str(e)[:200])
        closed_id = {k: v for k, v in ident.items() if int(k.rsplit("_", 1)[1]) >= 1 and k in stc and k not in sto}
        res["blocks"][str(b)] = dict(verdict=verdict, tau=taus[b], token_identity=ident, tolerance_rows=tol_rows, gates_round1=g1, gates_replay=g2,
                                     PC_PASS_round1=pc1, PC_PASS_open=pc2, round1_string=(K.load_json(R.r1_file(ctx, b, "summary_b{b}.json")) or {}).get("verdict"),
                                     replay_closed_string=v_rep, closed_loop_identity_descriptive=closed_id,
                                     labels=["descriptive"] + ([taus[b]["label"]] if taus[b]["label"] else []))
    res["errors"] = fails
    R.wj(ctx, "r1/r1_replay.json", res, config=dict(rule="R1 (i)-(iii)", tolerance="|diff| (stats-16)"), labels=["descriptive"])


def main(ctx):
    a = sys.argv[1:]
    if a[0] == "closed":
        closed(ctx, int(a[1]), a[2])
    else:
        open_loop(ctx, int(a[1]))


if __name__ == "__main__":
    K.run_script(main)
