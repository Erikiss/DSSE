"""Toy phase boundary by a bracket rule (theta_design, simulation C2). Same planted worlds as sim_toy_boundary.py; truth = the wd at
which a seed is in the meta phase with probability 0.5 (switch-on w_on*, switch-off w_off*). Per cell: f = share of seeds with
B2_meta > 0.10. Switch-on bracket: [largest grid wd with f <= 0.25, smallest larger grid wd with f >= 0.75] (wd = 0 counts as the
lowest point); switch-off mirrored. 'Below/above grid' if the edge cell is already past the threshold. Point estimate: log-linear
interpolation of f at 0.5 between the bracket ends. Reports: P(bracket found), P(truth in bracket | found), median bracket width
(factor), P(edge flags correct). Usage: python sim_toy_bracket.py <out.json>"""
import json, sys
import numpy as np
rng = np.random.default_rng(6); WD = np.array([0.1, 0.3, 1.0, 3.0, 10.0]); REPS = 4000
sig = lambda x: 1 / (1 + np.exp(-x))
def half(p, lo, hi):
    w = np.exp(np.linspace(np.log(lo), np.log(hi), 400001)); g = p(w) - 0.5; s = np.nonzero(np.diff(np.sign(g)) != 0)[0]; return float(w[s[0]]) if len(s) else None
res = {}
for name, (w_on, w_off, k, bmax) in {"wide window": (0.25, 4.0, 0.35, 0.25), "narrow window": (0.5, 2.0, 0.3, 0.2), "soft edges": (0.25, 4.0, 0.8, 0.25), "on below grid": (0.05, 4.0, 0.35, 0.25)}.items():
    p = lambda w, w_on=w_on, w_off=w_off, k=k: sig((np.log(w) - np.log(w_on)) / k) * (1 - sig((np.log(w) - np.log(w_off)) / k))
    t_on = half(p, 0.005, np.sqrt(w_on * w_off)); t_off = half(p, np.sqrt(w_on * w_off), 100.0)
    for S in (2, 3, 4, 6):
        out = dict(found_on=[], cov_on=[], width_on=[], found_off=[], cov_off=[], width_off=[], below=[])
        for r in range(REPS):
            X = np.where(rng.random((len(WD), S)) < p(WD)[:, None], rng.normal(bmax, 0.03, (len(WD), S)), rng.normal(0, 0.03, (len(WD), S)))
            f = (X > 0.10).mean(1); peak = int(np.argmax(f))
            # switch-on side (cells up to the peak)
            lo = [i for i in range(peak + 1) if f[i] <= 0.25]; hi = [i for i in range(peak + 1) if f[i] >= 0.75]
            if f[0] >= 0.75: out["below"].append(True); out["found_on"].append(False)
            else:
                out["below"].append(False)
                if lo and hi and max(lo) < max(hi):
                    i = max(lo); kk = min(j for j in hi if j > i); out["found_on"].append(True); out["cov_on"].append(WD[i] <= t_on <= WD[kk]); out["width_on"].append(WD[kk] / WD[i])
                else: out["found_on"].append(False)
            lo = [i for i in range(peak, len(WD)) if f[i] <= 0.25]; hi = [i for i in range(peak, len(WD)) if f[i] >= 0.75]
            if lo and hi and min(hi) < min(lo):
                kk = min(lo); i = max(j for j in hi if j < kk); out["found_off"].append(True); out["cov_off"].append(WD[i] <= t_off <= WD[kk]); out["width_off"].append(WD[kk] / WD[i])
            else: out["found_off"].append(False)
        res[f"{name} | seeds {S}"] = dict(w_on_half=t_on, w_off_half=t_off, P_found_on=float(np.mean(out["found_on"])), P_cover_on=float(np.mean(out["cov_on"])) if out["cov_on"] else None,
            median_width_on=float(np.median(out["width_on"])) if out["width_on"] else None, P_found_off=float(np.mean(out["found_off"])), P_cover_off=float(np.mean(out["cov_off"])) if out["cov_off"] else None,
            median_width_off=float(np.median(out["width_off"])) if out["width_off"] else None, P_flag_below_grid=float(np.mean(out["below"])))
        print(name, S, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[f"{name} | seeds {S}"].items()}, flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
