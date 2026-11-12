"""README section 12: the frozen rules of pc_out/plan_v2.md (with its section 'Amendments before the main runs', written before any
main-run result) applied to the rows of pc_summary.py. Written before the main runs.
Usage: python pc_analyze.py <out.json> <summary.json> [<summary.json> ...]
Conventions (frozen with the amendments): 'CI' = two-sided 95 % t interval over seeds (df = n - 1); 'lower bound' = the lower end
of that interval; TOST 'zero' = the upper one-sided 95 % t bound below 0.05; B2 = B2 at 8 shots unless stated; main cells =
K 12, gamma 0, weight-decay scale 1."""
import json, math, os, sys
import numpy as np

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812}
ARMS = ("bp_mup", "bp_sp", "pc_mup"); G_CAND = (0.03, 0.1, 0.3, 1.0)


def ci(v):
    v = np.asarray([x for x in v if x is not None and math.isfinite(x)], float); n = len(v)
    if n < 2: return dict(n=n, mean=float(v.mean()) if n else None, lo=None, hi=None, up1=None)
    m, se = float(v.mean()), float(v.std(ddof=1) / math.sqrt(n)); t2, t1 = T975.get(n - 1, 1.96), T95.get(n - 1, 1.645)
    return dict(n=n, mean=m, lo=m - t2 * se, hi=m + t2 * se, up1=m + t1 * se, lo1=m - t1 * se)


def boot_mean(v, B=10000, seed=0):
    v = np.asarray(v, float); g = np.random.default_rng(seed)
    if len(v) < 2: return None
    bs = v[g.integers(0, len(v), (B, len(v)))].mean(1); return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main_rows(R, arm, K=12, gamma=0.0, wd=1.0):
    return [r for r in R if r.get("arm") == arm and r.get("K") == K and abs(r.get("gamma", 0) - gamma) < 1e-12 and abs(r.get("wd_scale", 1) - wd) < 1e-12 and "error" not in r]


def cell(R, arm, W, seed, K=12, gamma=0.0, wd=1.0):
    c = [r for r in main_rows(R, arm, K, gamma, wd) if r["W"] == W and r["seed"] == seed]; return c[0] if c else None


def gates(R):
    G = {}
    for arm in ("bp_mup", "pc_mup"):                                  # G0: muP coordinate check
        rows = [r for r in main_rows(R, arm) if r.get("coord_check")]; Ws = sorted({r["W"] for r in rows}); per = []
        for i in (0, 1):
            vals = [np.mean([r["coord_check"][i] for r in rows if r["W"] == W]) for W in Ws]; per.append(max(vals) / max(min(vals), 1e-12) if vals else float("nan"))
        G[f"G0_{arm}"] = dict(pass_=len(Ws) >= 2 and all(math.isfinite(p) and p < 2 for p in per), ratio_h1=per[0], ratio_h2=per[1], widths=Ws)
    r128 = [cell(R, "bp_mup", 128, s) for s in range(5)]; n1 = sum(1 for r in r128 if r and r["level1"])
    G["G1"] = dict(pass_=n1 >= 4, level1_seeds=n1)
    b = ci([r["B2_8"] for r in r128 if r and r["level1"] and "B2_8" in r]); G["G2"] = dict(pass_=bool(b["mean"] is not None and b["mean"] >= 0.10 and b["lo"] is not None and b["lo"] > 0.03), **b)
    G["G3"] = {}
    for arm in ARMS:
        for W in sorted({r["W"] for r in main_rows(R, arm)}):
            f = [r["F8_mean"] for r in main_rows(R, arm) if r["W"] == W and "F8_mean" in r]
            G["G3"][f"{arm}_W{W}"] = dict(pass_=bool(f) and float(np.mean(f)) <= 0.10, mean=float(np.mean(f)) if f else None)
    G["G4"], G["G5"] = {}, {}
    for W in sorted({r["W"] for r in main_rows(R, "pc_mup")}):
        pcs = [r["pc"] for r in main_rows(R, "pc_mup") if r["W"] == W and r.get("pc")]
        if not pcs: continue
        rose = float(np.mean([p["rose_share"] for p in pcs])); G["G4"][W] = dict(pass_=rose <= 0.01, rose_share=rose,
            ratio_median=float(np.median([p["ratio_median"] for p in pcs])), ratio_le005_share=float(np.mean([p["ratio_le005_share"] for p in pcs])),
            original_G4_convergence=float(np.mean([p["ratio_le005_share"] for p in pcs])) >= 0.90)
        c2 = [p["cos_l2_early_median"] for p in pcs if p.get("cos_l2_early_median") is not None]
        G["G5"][W] = dict(pass_=bool(c2) and float(np.median(c2)) <= 0.90, cos_l2_early_median=float(np.median(c2)) if c2 else None)
    return G


def capable(R, arm, G):
    out = []
    for W in sorted({r["W"] for r in main_rows(R, arm)}):
        n = sum(1 for s in range(5) if (c := cell(R, arm, W, s)) and c["level1"])
        ok = n >= 4 and G["G3"].get(f"{arm}_W{W}", {}).get("pass_", False)
        if arm == "pc_mup": ok = ok and G["G4"].get(W, {}).get("pass_", False)
        if ok: out.append(W)
    return out


def slope(ws, ys):
    x = np.log2(np.asarray(ws, float)); y = np.asarray(ys, float); x = x - x.mean(); return float((x * (y - y.mean())).sum() / (x ** 2).sum())


def arm_stats(R, arm, G):
    Wc = capable(R, arm, G); res = dict(widths=Wc)
    if len(Wc) < 2: res["verdict"] = "UNDECIDED (fewer than two rule-capable widths)"; return res
    Wlo, Whi = Wc[0], Wc[-1]; sl = []
    for s in range(5):
        pts = [(W, c["B2_8"]) for W in Wc if (c := cell(R, arm, W, s)) and c["level1"] and "B2_8" in c]
        if len(pts) >= 2: sl.append(slope(*zip(*pts)))
    b = lambda W: [c["B2_8"] for s in range(5) if (c := cell(R, arm, W, s)) and c["level1"] and "B2_8" in c]
    D = [cell(R, arm, Whi, s)["B2_8"] - cell(R, arm, Wlo, s)["B2_8"] for s in range(5) if all((c := cell(R, arm, W, s)) and c["level1"] and "B2_8" in c for W in (Wlo, Whi))]
    meta = []
    for s in range(5):
        c12, c1 = cell(R, arm, Wlo, s), cell(R, arm, Wlo, s, K=1)
        if arm == "pc_mup" and c1 and (c1.get("pc") or {}).get("rose_share", 1.0) > 0.01: res.setdefault("meta_excluded_unstable_K1", []).append(s); continue   # G4 applies to the K = 1 rows too
        if c12 and c1 and c12["level1"] and "B2_8" in c12 and "B2_8" in c1: meta.append(c12["B2_8"] - c1["B2_8"])
    S, Blo, Bhi, Dd, M = ci(sl), ci(b(Wlo)), ci(b(Whi)), ci(D), ci(meta)
    res.update(W_lo=Wlo, W_hi=Whi, slope=S, slope_boot=boot_mean(sl), slopes=sl, B2_lo=Blo, B2_hi=Bhi, D=Dd, B2_meta_lo=M,
               by_width={W: ci(b(W)) for W in Wc})
    tost = Bhi["up1"] is not None and Bhi["up1"] < 0.05; neg = S["hi"] is not None and S["hi"] < 0
    if neg and tost and Blo["lo"] is not None and Blo["lo"] > 0.05 and M["lo"] is not None and M["lo"] > 0: v = "SUPPORTS H"
    elif Bhi["lo"] is not None and Bhi["lo"] > 0.05 and Dd["lo"] is not None and Dd["lo"] > -0.10: v = "CONTRADICTS H"
    elif neg and not tost: v = "SHRINKS-NOT-VANISHES"
    else: v = "UNDECIDED"
    res.update(TOST_zero_at_W_hi=tost, verdict=v, detectable_change=None if S["hi"] is None else (S["hi"] - S["lo"]) / 2 * math.log2(Whi / Wlo))
    return res


def overall(A, G):
    bp, sp = A["bp_mup"], A["bp_sp"]
    if not G["G0_bp_mup"]["pass_"]: return "H not testable (G0 coordinate check failed for BP-muP)"
    M = bp.get("B2_meta_lo") or {}
    if M.get("n", 0) < 2 or M.get("lo") is None: return f"H not tested: B2_meta at W_lo (BP-muP) not available (n = {M.get('n', 0)})"
    if M["lo"] <= 0: return "H not tested: B2_meta at W_lo (BP-muP) is not above 0, so any fall is transfer of the (x, y) code"
    if bp["verdict"] == "SUPPORTS H": return "H supported (BP-muP)"
    if bp["verdict"] == "CONTRADICTS H" and sp.get("verdict") in ("SUPPORTS H", "SHRINKS-NOT-VANISHES"):
        return "against H: a fall only under the standard parameterisation is the width-dependent effective learning rate"
    return f"H not supported (BP-muP: {bp['verdict']})"


def pc_vs_bp(R, G, A):
    if not G["G0_pc_mup"]["pass_"]: return dict(verdict="not testable (G0 coordinate check failed for PC-muP)")
    common = [W for W in A["pc_mup"].get("widths", []) if W in A["bp_mup"].get("widths", []) and G["G5"].get(W, {}).get("pass_", False)]
    out = dict(common_widths=common, bp_equivalent_widths=[W for W, g in G["G5"].items() if not g["pass_"]])
    m2 = lambda arm, W: ci([c["B2_8"] for s in range(5) if (c := cell(R, arm, W, s)) and c["level1"] and "B2_8" in c])["mean"]
    allw = [W for W in A["pc_mup"].get("widths", []) if W in A["bp_mup"].get("widths", [])]          # every G4-passing PC width, G5 or not
    tab = {W: dict(B2_gap=(m2("pc_mup", W) or 0) - (m2("bp_mup", W) or 0), cos_l2=G["G5"].get(W, {}).get("cos_l2_early_median")) for W in allw}
    ok = [W for W in allw if tab[W]["cos_l2"] is not None]
    if len(ok) >= 3:
        from scipy.stats import spearmanr
        rho = float(spearmanr([tab[W]["cos_l2"] for W in ok], [abs(tab[W]["B2_gap"]) for W in ok]).correlation); top = max(ok, key=lambda W: tab[W]["cos_l2"])
        out["converging_label"] = dict(rho_cos_absgap=rho, gap_at_max_cos=tab[top]["B2_gap"], label="PC converging to BP" if (rho <= -0.8 and abs(tab[top]["B2_gap"]) < 0.05) else "no convergence label")
    out["gap_and_cosine_all_widths"] = tab
    if len(common) < 2: out["verdict"] = "not testable (fewer than two common widths passing G4 and G5)"; return out
    diff = []
    for s in range(5):
        pp = [(W, c["B2_8"]) for W in common if (c := cell(R, "pc_mup", W, s)) and c["level1"] and "B2_8" in c]
        bb = [(W, c["B2_8"]) for W in common if (c := cell(R, "bp_mup", W, s)) and c["level1"] and "B2_8" in c]
        if len(pp) == len(bb) == len(common): diff.append(slope(*zip(*pp)) - slope(*zip(*bb)))
    d = ci(diff); span = math.log2(common[-1] / common[0])
    claim = d["lo"] is not None and (d["lo"] > 0 or d["hi"] < 0) and abs(d["mean"]) * span >= 0.05
    gap = {W: dict(B2_gap=(ci([c["B2_8"] for s in range(5) if (c := cell(R, "pc_mup", W, s)) and c["level1"] and "B2_8" in c])["mean"] or 0) -
                    (ci([c["B2_8"] for s in range(5) if (c := cell(R, "bp_mup", W, s)) and c["level1"] and "B2_8" in c])["mean"] or 0),
                   cos_l2=G["G5"].get(W, {}).get("cos_l2_early_median")) for W in common}
    out.update(slope_diff=d, verdict="PC differs from BP" if claim else "no difference claimed", gap_and_cosine_by_width=gap)
    return out


def cluster_boot_ols(X, y, cl, B=10000, seed=0):
    X, y, cl = np.asarray(X, float), np.asarray(y, float), np.asarray(cl); u = np.unique(cl); g = np.random.default_rng(seed); co = []
    fit = lambda Xs, ys: np.linalg.lstsq(np.column_stack([np.ones(len(ys)), Xs]), ys, rcond=None)[0]
    full = fit(X, y)
    for _ in range(B):
        pick = g.choice(u, len(u)); idx = np.concatenate([np.nonzero(cl == c)[0] for c in pick])
        if len(np.unique(X[idx][:, -1])) < 2: continue
        co.append(fit(X[idx], y[idx]))
    co = np.array(co); return full, np.percentile(co, 2.5, axis=0), np.percentile(co, 97.5, axis=0)


def d1(R):
    out = {}
    for arm in ARMS:
        rows = [r for r in main_rows(R, arm) if r["level1"] and "B2_8" in r and r.get("delta")]
        if len({r["W"] for r in rows}) < 2: out[arm] = "not enough cells"; continue
        y = [r["B2_8"] for r in rows]; lw = [math.log2(r["W"]) for r in rows]; cl = [r["seed"] for r in rows]; o = {}
        for var, key in (("raw", "delta"), ("ctrl", "delta_ctrl")):
            dl = [r["delta"]["h2"][key] for r in rows]
            full, lo, hi = cluster_boot_ols(np.column_stack([dl, lw]), y, cl)
            dl_ex0 = lo[1] > 0 or hi[1] < 0; w_in0 = lo[2] <= 0 <= hi[2]
            o[var] = dict(coef_delta=float(full[1]), ci_delta=[float(lo[1]), float(hi[1])], coef_log2W=float(full[2]), ci_log2W=[float(lo[2]), float(hi[2])],
                          verdict=("delta carries the width effect (" + ("negative: scarcity reading" if full[1] < 0 else "positive: counter-thesis") + ")") if (dl_ex0 and w_in0)
                          else "delta does not carry the width effect")
        out[arm] = o
    return out


def d2(R):
    base = {s: cell(R, "bp_mup", 256, s) for s in range(5)}; dd = lambda r: r["delta"]["h2"]["delta"] if r and r.get("delta") else None
    if not (base[0] and base[0]["level1"] and base[0].get("delta")): return dict(verdict="probe failed (no baseline at W 256, seed 0)")
    cand, over = [], []
    for g in G_CAND:
        p, q = cell(R, "bp_mup", 256, 0, gamma=-g), cell(R, "bp_mup", 256, 0, gamma=g)
        if p and q and p["level1"] and q["level1"] and p.get("delta") and q.get("delta"):
            if min(dd(p), dd(q), dd(base[0])) <= 0: over.append(g); continue           # delta at or below its floor: overshoot, excluded
            rp, rq = dd(p) / dd(base[0]), dd(base[0]) / dd(q); cand.append((max(abs(math.log2(rp) - 1), abs(math.log2(rq) - 1)), g, rp, rq))
    if not cand: return dict(verdict="probe failed (" + ("pull collapses delta below the floor" if over else "no g passes level 1 at seed 0") + ")", overshoot=over)
    best = min(cand); ties = [c[1] for c in cand if abs(c[0] - best[0]) < 1e-9 and c[1] != best[1]]
    _, g, rp, rq = best; res = dict(g=g, pilot_ratios=dict(push=rp, pull=rq), candidates=cand, overshoot_excluded=over, tie_with=ties)
    push_ok = pull_ok = 0; sgn = []
    for s in range(5):
        b0, p, q = base[s], cell(R, "bp_mup", 256, s, gamma=-g), cell(R, "bp_mup", 256, s, gamma=g)
        if not (b0 and p and q and b0.get("delta") and p.get("delta") and q.get("delta")): sgn.append(0); continue
        push_ok += dd(p) > dd(b0); pull_ok += dd(q) < dd(b0)
        sgn.append(np.sign(p["B2_8"] - q["B2_8"]) if p["level1"] and q["level1"] and "B2_8" in p and "B2_8" in q else 0)
    res.update(push_moves=int(push_ok), pull_moves=int(pull_ok), sign_B2_push_minus_pull=[int(x) for x in sgn])
    if push_ok < 4 or pull_ok < 4: res["verdict"] = "probe failed (delta did not move as intended in >= 4 of 5 seeds)"
    elif sum(x < 0 for x in sgn) >= 4: res["verdict"] = "delta causal for the meta benefit (negative: scarcity reading)"
    elif sum(x > 0 for x in sgn) >= 4: res["verdict"] = "delta causal for the meta benefit (positive: counter-thesis)"
    else: res["verdict"] = "no causal effect of delta found"
    return res


def extras(R):
    kc = {}
    for W in (64, 256, 1024):
        kc[W] = {K: ci([c["B2_8"] for s in range(5) if (c := cell(R, "bp_mup", W, s, K=K)) and "B2_8" in c and (c["level1"] or K == 1)])["mean"] for K in (1, 3, 6, 12)}
    wd0 = {W: dict(wd1=ci([c["B2_8"] for s in range(5) if (c := cell(R, "bp_mup", W, s)) and c["level1"] and "B2_8" in c])["mean"],
                   wd0=ci([c["B2_8"] for s in range(5) if (c := cell(R, "bp_mup", W, s, wd=0.0)) and c["level1"] and "B2_8" in c])["mean"]) for W in (64, 256, 1024)}
    sec = {arm: {W: dict(B2_4=ci([c["B2_4"] for s in range(5) if (c := cell(R, arm, W, s)) and c["level1"] and "B2_4" in c])["mean"],
                         B2_all=ci([c["B2_all"] for s in range(5) if (c := cell(R, arm, W, s)) and c["level1"] and "B2_all" in c])["mean"],
                         retention=ci([c["retention_mean"] for s in range(5) if (c := cell(R, arm, W, s)) and c["level1"] and "retention_mean" in c])["mean"],
                         level1=sum(1 for s in range(5) if (c := cell(R, arm, W, s)) and c["level1"]))
                 for W in sorted({r["W"] for r in main_rows(R, arm)})} for arm in ARMS}
    return dict(K_curve=kc, weight_decay_0=wd0, secondary=sec)


def main():
    out = sys.argv[1]; R = []
    for p in sys.argv[2:]: R += json.load(open(p))
    G = gates(R); res = dict(gates=G, n_rows=len(R), errors=[r for r in R if "error" in r])
    expected = set()                                   # job lists written by run_shard.py next to each summary
    for p in sys.argv[2:]:
        e = os.path.join(os.path.dirname(p), os.path.basename(p).replace("pc_summary_", "pc_expected_"))
        if e != p and os.path.exists(e): expected |= set(json.load(open(e)))
    res["missing_jobs"] = sorted(expected - {r.get("name") for r in R}) if expected else "no expected-job lists found"
    if not (G["G1"]["pass_"] and G["G2"]["pass_"]):
        res["overall"] = "world has no testable meta level (G1 or G2 failed)"; res["D1"] = res["D2"] = "not evaluated (G1 or G2 failed; the plan stops here)"
    else:
        A = {arm: arm_stats(R, arm, G) for arm in ARMS}
        for arm in ("bp_mup", "pc_mup"):
            if not G[f"G0_{arm}"]["pass_"]: A[arm]["verdict"] = f"not testable (G0 coordinate check failed): {A[arm].get('verdict')}"
        res.update(arms=A, overall=overall(A, G), pc_vs_bp=pc_vs_bp(R, G, A), D1=d1(R), D2=d2(R))
    res["extras"] = extras(R)
    json.dump(res, open(out, "w"), indent=1, default=float)
    print("overall:", res["overall"]); print("arms:", {a: v.get("verdict") for a, v in res.get("arms", {}).items()})
    print("D2:", res["D2"].get("verdict") if isinstance(res["D2"], dict) else res["D2"]); print("missing jobs:", res["missing_jobs"] if isinstance(res["missing_jobs"], str) else len(res["missing_jobs"]))


if __name__ == "__main__":
    main()
