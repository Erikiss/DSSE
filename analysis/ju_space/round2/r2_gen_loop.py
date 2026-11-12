"""Round 2, A6 (b) (final plan sections 3.1 A6, 9.2; design M8): the GEN loop at block b = env BLOCK (14, b_lo or b_hi), one process per
block, in session r2_q1 after r2_axes.py.
  Finality (G4, block 14 included): if block b's round-1 outputs are not final (no summary_b<b>.json), rgi/dyn_b<b>.json says 'not run
  (round-1 output missing)' and nothing is opened or generated.
  Gate GA(b) (O2, G28): the loop runs only where the cross-fitted AUC(u_RGI) of block b is >= 0.85 (r2_axes.ga_gate, the function r2_ana
  uses too); otherwise rgi/dyn_b<b>.json says 'not run (AUC gate)' and nothing is generated (OD-A11).
  Loop: rk.run_loop('r2GEN', unit(u_RGI_perp(b)), nseed 1, 4 rounds) with the block's alpha_b and cap_b (open_block sets ASCALE =
  alpha_scale_b; keys keyseed('r2GEN' + SUF_b, n, p, s, b) via rk.evaluate, Appendix D; rk's own state keys ev_r2GEN<SUF>_n,
  dir_r2GEN<SUF>_n, dmg_r2GEN<SUF>_n, so a rerun resumes from the cached evaluations).
  Read-outs per evaluation n = 0..3 (push d_n, derived d_(n+1)): the frozen indicators on u_T_perp (run_loop's own row: c, p, Q, rel, A,
  Ue, sigma, damage_of_push, damage_next, first_div_median, identical_pairs, ...), plus c, p, Q, rel, A on u_RGI_perp and on u_RGI_perpT
  (O2 axes of block b with their own 2,000 null axes; r2_rk.stats_on_axis = rk.stats's formulas); E_GI = A_4 / A_2 on u_RGI_perp;
  R_4(RSI_a, GEN) = cos(d_4^RSI_a, d_4^GEN) / sqrt(rel(d_4^RSI_a) rel(d_4^GEN)) with rel(d_4^RSI_a) = R1.loops[b]['RSI_a'][3]['rel'];
  S_4 = cos(d_4, d_3) / sqrt(rel(d_3) rel(d_4)) of the GEN loop (rk_verdict.S4's formula). The A7 dynamics rules are evaluated in r2_ana.
Outputs (in <r2_dir>): rgi/dyn_b<b>.json, rgi/dyn_b<b>.pt (start, d_1..d_4, per-pair arrays of every evaluation with texts as token ids).
Usage: python r2_gen_loop.py <root> <r1_dir> <r2_dir> [run | selftest]      env BLOCK, DEVICE, NT, MOCK"""
import copy, glob, json, math, os, shutil, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
import r2_rk as R
from r2_axes import ga_gate, GA_AUC

LOOP, NSEED, ROUNDS = "r2GEN", 1, 4
EV_KEYS = ("delta", "ok", "prompt_of", "xp", "xm", "first_div", "conts", "damage_of_push", "alpha", "capped_share", "dropped")


def out_paths(r2_dir, b):
    d = os.path.join(r2_dir, "rgi"); return os.path.join(d, f"dyn_b{b}.json"), os.path.join(d, f"dyn_b{b}.pt")


def _fin(x): return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def run(root, r1_dir, r2_dir, b):
    """The GEN loop of block b (skipped when rgi/dyn_b<b>.json exists). Returns the JSON written."""
    js_p, pt_p = out_paths(r2_dir, b)
    if os.path.exists(js_p): R.log(f"GEN loop b{b}: {js_p} exists, skipped"); return json.load(open(js_p))
    base = dict(script="r2_gen_loop.py", plan="final plan (abschnitt11_12_runde2_plan.md) section 3.1 A6 (b)", block=b, loop=LOOP, nseed=NSEED,
                rounds=ROUNDS, keys="keyseed('r2GEN' + SUF_b, n, p, s, b)", ga_threshold=GA_AUC, mock=R.MOCK)
    if not R.block_final(r1_dir, b):
        js = dict(base, status="not run (round-1 output missing)", GA=None, note=f"summary_b{b}.json missing: block not final (G4)")
        R.write_json(js_p, js); R.log(f"GEN loop b{b}: not run (round-1 output missing)"); return js
    ax_p = os.path.join(r2_dir, "axes", "r2_axes.pt")
    if not os.path.exists(ax_p): raise SystemExit(f"{ax_p} missing: run r2_axes.py first (same session)")
    AX = torch.load(ax_p, map_location="cpu", weights_only=False); A = AX["blocks"].get(b)
    if A is None:
        js = dict(base, status="not run (round-1 output missing)", GA=None, note="no O2 axes at this block (r2_axes.json gives the reason)")
        R.write_json(js_p, js); return js
    if not ga_gate(A["auc_RGI"]):
        js = dict(base, status="not run (AUC gate)", GA=False, auc_RGI=A["auc_RGI"]); R.write_json(js_p, js); R.log(f"GEN loop b{b}: not run (AUC gate)"); return js
    try: rk = R.open_block(root, r1_dir, r2_dir, b)
    except R.R1Missing as e:
        js = dict(base, status="not run (round-1 output missing)", GA=True, auc_RGI=A["auc_RGI"], error=str(e)); R.write_json(js_p, js); return js
    uR, uRT = A["u_RGI_perp"].double(), A["u_RGI_perpT"].double()
    if uR.shape[0] != rk.D: raise RuntimeError(f"O2 axes have dimension {uR.shape[0]}, the model {rk.D} (MOCK mismatch?)")
    start = R.unit(uR); rows = rk.run_loop(LOOP, start, NSEED, rounds=ROUNDS); L = LOOP + rk.SUF
    out_rows, PT = [], dict(start=start.float(), d={}, ev={}, u_RGI_perp=A["u_RGI_perp"], u_RGI_perpT=A["u_RGI_perpT"], loop=L)
    for n in range(ROUNDS):
        ev = rk.ST[f"ev_{L}_{n}"]; sR, _ = R.stats_on_axis(ev, uR, A["null_u_RGI_perp"]); sT, _ = R.stats_on_axis(ev, uRT, A["null_u_RGI_perpT"])
        r = rows[n]
        out_rows.append(dict(n=n, d=n + 1, u_T_perp=r, u_RGI_perp=sR, u_RGI_perpT=sT, damage_of_push=r["damage_of_push"], damage_next=r["damage_next"],
                             first_div_median=r["first_div_median"], identical_pairs=r["identical_pairs"], dropped=r["dropped"], capped_share=r["capped_share"]))
        PT["d"][n + 1] = rk.ST[f"dir_{L}_{n + 1}"].clone(); PT["ev"][n] = {k: ev[k] for k in EV_KEYS if k in ev}
    A2, A4 = out_rows[1]["u_RGI_perp"]["A"], out_rows[3]["u_RGI_perp"]["A"]
    E_GI = A4 / A2 if (_fin(A2) and _fin(A4) and abs(A2) > 1e-12) else float("nan")
    loops = R.load_r1_json(r1_dir, f"rk/b{b}/loops.json", b); d4a = rk.ST[f"dir_RSI_a{rk.SUF}_4"].double(); rel_a = float(loops["RSI_a"][3]["rel"])
    d3, d4 = PT["d"][3].double(), PT["d"][4].double(); rel3, rel4 = rows[2]["rel"], rows[3]["rel"]
    R4 = R.cosf(d4a, d4) / math.sqrt(rel_a * rel4) if (_fin(rel_a) and _fin(rel4)) else float("nan")
    S4 = R.cosf(d4, d3) / math.sqrt(rel3 * rel4) if (_fin(rel3) and _fin(rel4)) else float("nan")
    PT["d4_RSI_a"] = d4a.float()
    new = sorted(k for k in rk.ST if k.startswith((f"ev_{L}_", f"dir_{L}_", f"dmg_{L}_")))
    js = dict(base, status="ran", GA=True, auc_RGI=A["auc_RGI"], alpha=R.BLK["alpha"], alpha_scale=R.BLK["alpha_scale"], suffix=rk.SUF, cap=R.BLK["cap"],
              med=R.BLK["med"], state_keys=new, rows=out_rows, E_GI=E_GI, R4_RSI_a_GEN=R4, rel_d4_RSI_a=rel_a, S4=S4,
              cos_d1_u_RGI_perp=R.cosf(PT["d"][1].double(), uR), finder=list(R.FINDER_LOG))
    R.torch_save(pt_p, PT); R.write_json(js_p, js)
    R.log(f"GEN loop b{b}: done; on u_RGI_perp c {[round(r['u_RGI_perp']['c'], 3) for r in out_rows]}, E_GI {E_GI:.3f}, R_4 {R4:.3f}, S_4 {S4:.3f}")
    return js


def selftest(root, r1_dir, r2_dir):
    """MOCK code test at block BLOCK: the gate closed (nothing generated, no block opened), then open (copies of axes/r2_axes.pt with
    the AUC forced, since the MOCK model's axes are random); checks keys, state guard, statistics, E_GI / R_4 / S_4 and resume."""
    assert R.MOCK, "selftest needs MOCK=1"; b = int(os.environ.get("BLOCK", "14")); res = []

    def check(name, cond, info=""):
        res.append(dict(test=name, ok=bool(cond), info=str(info))); print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)
    AX = torch.load(os.path.join(r2_dir, "axes", "r2_axes.pt"), map_location="cpu", weights_only=False); base = os.path.join(r2_dir, "selftest_gen")
    shutil.rmtree(base, ignore_errors=True)

    def variant(name, auc):
        d = os.path.join(base, name); X = copy.copy(AX); X["blocks"] = dict(AX["blocks"]); X["blocks"][b] = dict(AX["blocks"][b], auc_RGI=auc, GA=ga_gate(auc))
        R.torch_save(os.path.join(d, "axes", "r2_axes.pt"), X); return d
    nf = os.path.join(base, "nonfinal"); shutil.copytree(r1_dir, os.path.join(nf, "r1"))
    for f in glob.glob(os.path.join(nf, "r1", "gpu*", f"summary_b{b}.json")): os.remove(f)
    js = run(root, os.path.join(nf, "r1"), variant("nonfinal_r2", 0.9), b)
    check("G4: block not final -> 'not run (round-1 output missing)', no block opened", js["status"] == "not run (round-1 output missing)" and R._RK is None)
    g = variant("gate", 0.5); js = run(root, r1_dir, g, b)
    check("gate closed: 'not run (AUC gate)', nothing generated, no block opened", js["status"] == "not run (AUC gate)" and R._RK is None and not os.path.exists(os.path.join(g, "rk")))
    o = variant("open", 0.9); js = run(root, r1_dir, o, b); rk = R._RK; L = LOOP + rk.SUF
    check("gate open: four rows with u_T_perp, u_RGI_perp and u_RGI_perpT read-outs", js["status"] == "ran" and len(js["rows"]) == 4 and all({"u_T_perp", "u_RGI_perp", "u_RGI_perpT"} <= set(r) for r in js["rows"]))
    want = sorted([f"ev_{L}_{n}" for n in range(4)] + [f"dir_{L}_{n}" for n in range(1, 5)] + [f"dmg_{L}_{n}" for n in range(5)])
    ST = torch.load(os.path.join(o, "rk", f"b{b}", "state.pt"), map_location="cpu", weights_only=False); fp = json.load(open(os.path.join(o, "rk", f"b{b}", "r1_fingerprint.json")))
    check("state: exactly the GEN-loop keys added, round-1 keys unchanged", sorted(set(ST) - set(fp["keys"])) == want and js["state_keys"] == want and all(R._digest(ST[k]) == fp["digest"][k] for k in fp["keys"]))
    ev0 = ST[f"ev_{L}_0"]; start = R.unit(AX["blocks"][b]["u_RGI_perp"].double())
    E = R.evaluate_variant(start, R.BLK["alpha"], lambda p, s: rk.keyseed(L, 0, p, s, b), nseed=NSEED, cbar=False)
    check("key namespace keyseed('r2GEN' + SUF, n, p, s, b) and push alpha_b * unit(u_RGI_perp) reproduce round 0", E["conts"] == ev0["conts"] and abs(ev0["alpha"] - R.BLK["alpha"]) < 1e-12)
    A = AX["blocks"][b]; s2, _ = R.stats_on_axis(ST[f"ev_{L}_2"], A["u_RGI_perp"].double(), A["null_u_RGI_perp"])
    check("u_RGI_perp read-out equals stats_on_axis of the stored evaluation", all(s2[k] == js["rows"][2]["u_RGI_perp"][k] or (s2[k] != s2[k] and js["rows"][2]["u_RGI_perp"][k] != js["rows"][2]["u_RGI_perp"][k]) for k in ("c", "p", "rel", "A")))
    PT = torch.load(out_paths(o, b)[1], map_location="cpu", weights_only=False); loops = R.load_r1_json(r1_dir, f"rk/b{b}/loops.json", b); rw = js["rows"]
    e_gi = rw[3]["u_RGI_perp"]["A"] / rw[1]["u_RGI_perp"]["A"]; r4 = R.cosf(ST[f"dir_RSI_a{rk.SUF}_4"].double(), PT["d"][4].double()) / math.sqrt(loops["RSI_a"][3]["rel"] * rw[3]["u_T_perp"]["rel"])
    s4 = R.cosf(PT["d"][4].double(), PT["d"][3].double()) / math.sqrt(rw[2]["u_T_perp"]["rel"] * rw[3]["u_T_perp"]["rel"])
    check("E_GI, R_4, S_4 by definition", abs(e_gi - js["E_GI"]) < 1e-12 and abs(r4 - js["R4_RSI_a_GEN"]) < 1e-12 and abs(s4 - js["S4"]) < 1e-12, (js["E_GI"], js["R4_RSI_a_GEN"], js["S4"]))
    check("rel(d_n) and damage rows as rk.run_loop", all(rw[n]["u_RGI_perp"]["rel"] == rw[n]["u_T_perp"]["rel"] and rw[n]["damage_of_push"] == ST[f"dmg_{L}_{n}"]["damage"] for n in range(4)))
    t0 = os.path.getmtime(out_paths(o, b)[0]); js2 = run(root, r1_dir, o, b)
    check("resume: an existing output is kept", os.path.getmtime(out_paths(o, b)[0]) == t0 and js2 == json.loads(json.dumps(js2)))
    os.remove(out_paths(o, b)[0]); js3 = run(root, r1_dir, o, b)
    check("rerun from the cached evaluations gives the same rows", json.dumps(js3["rows"]) == json.dumps(js["rows"]))
    ok = all(r["ok"] for r in res); R.write_json(os.path.join(base, f"selftest_b{b}.json"), dict(ok=ok, tests=res))
    print(f"r2_gen_loop selftest (block {b}): {sum(r['ok'] for r in res)}/{len(res)} passed", flush=True); return ok


if __name__ == "__main__":
    if len(sys.argv) < 4: raise SystemExit(__doc__)
    root, r1_dir, r2_dir = sys.argv[1:4]; stage = sys.argv[4] if len(sys.argv) > 4 else "run"
    if "BLOCK" not in os.environ: raise SystemExit("set env BLOCK (14, b_lo or b_hi)")
    if stage == "selftest": sys.exit(0 if selftest(root, r1_dir, r2_dir) else 1)
    if stage != "run": raise SystemExit(f"unknown stage {stage}")
    run(root, r1_dir, r2_dir, int(os.environ["BLOCK"]))
