"""Part R MOCK self-tests (CPU; MOCK=1; rk_loop's tiny random model).
    python r1b_r_selftest.py units                      model-free checks of STAT-1b (planted A recovered, Gram readout = direct
                                                        readout, K1-s size, cross-fitted cosines, SENS plant, joint MDE), keys, package
    python r1b_r_selftest.py sessions <dir> <root>       every R session in order on a mock tree with rk_loop MOCK round-1 states
                                                        (R1B_MOCK_WITH_R1), a stand-in r1b_K_S1 k4j.json, then the tier-2 sessions with
                                                        their triggers forced (MOCK only): every step must end ok or 'not run'
<root> needs models/qwen3-1.7b (config + tokenizer), rsi_dl/persona and gpu_pkg/rk_dirs.pt (as round 2's mock_r1)."""
import json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["r1b_R_S0", "r1b_R_S2_14", "r1b_R_S2_18", "r1b_R_S2_06", "r1b_R_S1", "r1b_R_S3", "r1b_R_S5", "r1b_R_S6", "r1b_R_S7", "r1b_R_ana"]
TIER2 = ["r1b_R_S5L", "r1b_R_S8", "r1b_R_ana2"]


def units():
    import torch
    import r1b_core as K
    import r1b_r_lib as R
    import r1b_r_stats as S
    errs = []

    def ok(c, m):
        print(("  ok    " if c else "  FAIL  ") + m, flush=True); (None if c else errs.append(m))
    g = torch.Generator().manual_seed(0); P, D, ns = 64, 32, 2
    u = S.unit(torch.randn(D, generator=g, dtype=torch.float64)); fold = np.array([i % 2 == 0 for i in range(P)])
    diff = {t: torch.randn(P, D, generator=g, dtype=torch.float64) * 0.3 + (2.0 * u if t == "rsi" else 0) + torch.randn(D, generator=g, dtype=torch.float64)
            for t in R.TOPICS}
    keep = {t: torch.ones(P, dtype=torch.bool) for t in R.TOPICS}; us = S.unit(torch.randn(D, generator=g, dtype=torch.float64))
    AS = S.AxisSet(diff, keep, fold, us, S.unit(torch.randn(D, generator=g, dtype=torch.float64)), 14)
    W = torch.as_tensor(K.prompt_bootstrap(P, B=200, seed=1001), dtype=torch.float64); a1, aW = AS.get(torch.ones(1, P, dtype=torch.float64), ["T_v"]), AS.get(W, ["T_v"])
    po = torch.arange(P).repeat_interleave(ns); H = S.halvings(P)
    noise = torch.randn(P * ns, D, generator=g, dtype=torch.float64)
    for A_true in (0.0, 0.25, 0.5):
        r0 = S.Row(noise, torch.ones(P * ns, dtype=torch.bool), po, P); rel = r0.rel(H)[0]; r = S.plant(r0, a1["T_v"]["u"][0], rel, A_true)
        st = S.stat_row(r, a1["T_v"], aW["T_v"], fold, H, W)
        ok(abs(st["A"] - A_true) < 0.06 + 0.1 * A_true, f"planted A {A_true}: read {st['A']:.3f} (rel {st['rel']:.2f}, kappa {st['kappa']:.2f})")
    rd = S.readout(r, a1["T_v"], fold, H); rw = S.readout(r, a1["T_v"], fold, H, W=torch.ones(1, P, dtype=torch.float64))
    ok(abs(rd["A"] - float(rw["A"][0])) < 1e-9, "Gram-form readout with unit weights = full-sample readout")
    d = r.d(); dd = (r.delta[r.ok].mean(0)); ok(float((d - dd).norm()) < 1e-9, "Row.d = mean of ok pairs")
    rej = np.mean([S.k1s(np.random.default_rng(i).normal(size=60), flips=2000)["p_s"] <= 0.05 for i in range(200)])
    ok(rej <= 0.10, f"K1-s size at nominal 0.05 under the null: {rej:.3f}")
    c = S.cf_cos(r, [r], H); ok(abs(c["cf"] - 1) < 0.2, f"cross-fitted cosine of a row with itself near 1: {c['cf']:.3f}")
    m = S.joint_mde(0.035, 0.6, 0.5, "substantial"); ok(isinstance(m["mde"], float) and 0.25 < m["mde"] < 0.5, f"joint MDE 'substantial' {m['mde']}")
    ok(not K.check_keys(), f"SO-KEYS check with R's loop names {K.KEYS['R']['loops']}")
    ok(len(R.neutral_pool()) == 628, "neutral pool (Appendix R.A) sha256-pinned, 628 words")
    ok(R.rk_literal("INSTR")["rsi"][0].startswith("(Answer as an AI"), "rk_loop literals read verbatim (ast)")
    ok(R.wcount(["The sea, the SEA and seas; self-improvement improves."], {"s": R.W_SEA, "r": R.W_RSI}).tolist() == [[3, 1]], "whole-word counts")
    print("R UNITS " + ("OK" if not errs else f"FAILED: {errs}"), flush=True)
    return 1 if errs else 0


def run(sess, d, root, tree, env):
    w = os.path.join(d, "work", sess); log = os.path.join(d, f"{sess}.log")
    with open(log, "w") as f:
        rc = subprocess.run([sys.executable, os.path.join(HERE, "r1b_run_session.py"), sess, root, w, tree["A"]], env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    steps = (json.load(open(os.path.join(w, "session_done.json"))) if os.path.exists(os.path.join(w, "session_done.json")) else {}).get("steps", {})
    bad = {k: v for k, v in steps.items() if v.get("state") not in ("ran", "skipped", "not run")}
    print(f"  {sess}: exit {rc}; steps {len(steps)}; not ok {list(bad)}; not run {[k for k, v in steps.items() if v.get('state') == 'not run']}", flush=True)
    return rc, bad


def sessions(d, root):
    import r1b_run_session as RS
    d = os.path.abspath(d); env = dict(os.environ, MOCK="1", NT="1", RUN_DEVICE="cpu", R1B_ACCOUNT="A", R1B_MOCK_WITH_R1="1",
                                      POOLDIRS=os.path.join(root, "gpu_pkg", "rk_dirs.pt"), PERSONA=os.path.join(root, "rsi_dl", "persona"),
                                      R1B_R_FORCE_GATES="1")
    os.environ.update(env); tree = RS.mock_tree(os.path.join(d, "tree"), root, with_r1=True); env["R1B_FREEZE"] = tree["freeze"]
    kb = os.path.join(tree["B"], "r1b_K_S1"); os.makedirs(kb, exist_ok=True)
    json.dump(dict(convention="output", mock=True), open(os.path.join(kb, "k4j.json"), "w")); json.dump(dict(mock=True), open(os.path.join(kb, "session_done.json"), "w"))
    errs = []
    for s in ORDER:
        rc, bad = run(s, d, root, tree, env)
        if rc or bad:
            errs.append(s)
    for fid, key, v in (("r1b_R_S5/rg11.json", "decision", "refit"), ("r1b_R_ana/rg15.json", "J1_and_J2", True)):   # force the triggers (MOCK)
        p = os.path.join(tree["A"], fid); j = json.load(open(p)); j[key] = v; json.dump(j, open(p, "w"))
        for sd in (os.path.join(tree["A"], fid.split("/")[0]), os.path.join(d, "work", fid.split("/")[0])):
            rec = os.path.join(sd, "steps")
            for f in os.listdir(rec) if os.path.isdir(rec) else []:
                r = json.load(open(os.path.join(rec, f)))
                if fid.split("/")[1] in r.get("outputs", {}):
                    import r1b_core as K
                    r["outputs"][fid.split("/")[1]]["sha256"] = K.sha256_file(p); json.dump(r, open(os.path.join(rec, f), "w"))
    for s in TIER2:
        rc, bad = run(s, d, root, tree, env)
        if rc or bad:
            errs.append(s)
    print("R SESSIONS " + ("OK" if not errs else f"FAILED: {errs}"), flush=True)
    return 1 if errs else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(units() if a and a[0] == "units" else (sessions(a[1], a[2]) if a and a[0] == "sessions" else (print(__doc__) or 2)))
