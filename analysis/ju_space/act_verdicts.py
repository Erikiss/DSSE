"""Applies the decision rules fixed by the plan review (act_plan_review.json) to the results of README section 9.
Reads act_report.json (act_report.py) and act_masks_<arch>_s<n>.json (act_masks.py) from one directory; writes act_verdicts.json there.
E3   stabilised optimiser: a setting qualifies if, on seed 0, the 5 % run U and its drift control D both have no event in the
     screen (1500 steps at lr 1e-3; 5000 at lr 3e-4, where events are counted on the 50-step trace of batch loss and weight
     norm) and D stays at 0.99 or above. Without a qualifying setting E3 is undecided.
E1   (a) the kind of nonlinearity decides: one architecture has late level >= 0.90 in all 3 networks while another is below 0.50
     in all 3, drift controls readable;  against: late level below 0.50 in every network of ReLU, SwiGLU and bilinear;
     (b) the loss needs a threshold or gate: against if the bilinear unit is below 0.50 in all 3 networks;
     (c) the loss needs an MLP nonlinearity: against if the linear-MLP networks generalise in 2 of 2 seeds and both are below 0.50
E2   at the readable step with the largest loss, per network: for if r >= 0.50, k >= 0.50, r >= r_ref + 0.25, k >= k_ref + 0.25;
     against if r <= r_ref + 0.10 and k <= k_ref + 0.10; a verdict per architecture needs the same verdict in all 3 networks
E2b  for "the loss needs the switching pattern to change": frozen-pattern U late level >= 0.90 in all 3 while the ordinary U is
     below 0.50, frozen-pattern D readable, and the frozen-pattern U moved at least 0.8 times as far from G as the ordinary U;
     against: frozen-pattern U late level below 0.50 in all 3
Usage: python act_verdicts.py <results_dir>"""
import glob, json, os, sys

R = sys.argv[1]; rep = json.load(open(f"{R}/act_report.json")); runs = rep["runs"]; out = {}
# E3 (lr 1e-3: events from the evaluations every 50 steps; lr 3e-4: from the 50-step trace)
scr = {}
for tag, steps in (("scr_eps6", 1500), ("scr_eps5", 1500), ("scr_ams", 1500), ("fb_lr3", 5000), ("fb_lr3e5", 5000)):
    u, d = runs.get(f"base_s0_{tag}_U"), runs.get(f"base_s0_{tag}_D")
    if not (u and d): continue
    ev = lambda r: [list(e) for e in (r["trace_events"] if tag.startswith("fb_") else r["events"]) if e[0] <= steps]
    scr[tag] = dict(events_U=ev(u), events_D=ev(d), min_held_D=d["min_held"], qualifies=bool(not ev(u) and not ev(d) and d["min_held"] >= 0.99))
out["E3"] = dict(screen=scr, verdict="undecided: no setting qualified" if not any(v["qualifies"] for v in scr.values()) else "a setting qualified: see the full runs")
# E1
arch_runs = lambda a, arm: [runs.get(f"{a}_s{s}_{arm}") for s in range(3)]
def late(a):
    if a == "base": return [runs.get(f"base_s{s}_U_standard_combined", {}).get("late") for s in range(3)], [runs.get(f"base_s{s}_D_standard_combined", {}).get("readable_as_drift") for s in range(3)]
    return [r["late"] if r else None for r in arch_runs(a, "U")], [r["readable_as_drift"] if r else None for r in arch_runs(a, "D")]
E1 = {a: dict(zip(("late_U", "drift_readable"), late(a))) for a in ("base", "swiglu", "bilinear", "linear")}
full = lambda a: all(v is not None for v in E1[a]["late_U"]) and all(E1[a]["drift_readable"])
below = lambda a: full(a) and all(v < 0.5 for v in E1[a]["late_U"]); above = lambda a: full(a) and all(v >= 0.9 for v in E1[a]["late_U"])
three = ("base", "swiglu", "bilinear")
E1["verdict_a"] = "against: below 0.50 in every network of ReLU, SwiGLU and bilinear" if all(below(a) for a in three) else ("for" if any(above(a) for a in three) and any(below(a) for a in three) else "undecided or incomplete")
E1["verdict_b"] = "against: the bilinear unit (no threshold, no gate) loses generalisation in all 3 networks" if below("bilinear") else ("for" if above("bilinear") and below("base") else "undecided or incomplete")
lin = [runs.get(f"linear_s{s}_U") for s in range(2)]; E1["linear"] = [r["late"] if r else None for r in lin]
E1["verdict_c"] = "against: both linear-MLP networks lose generalisation" if all(r and r["late"] < 0.5 for r in lin) and all(runs.get(f"linear_s{s}_D", {}).get("readable_as_drift") for s in range(2)) else "undecided or incomplete"
out["E1"] = E1
# E2
E2 = {}
for a in ("relu", "swiglu"):
    per = []
    for s in range(3):
        p = f"{R}/act_masks_{a}_s{s}.json"
        if not os.path.exists(p): per.append(None); continue
        st = [x for x in json.load(open(p))["steps"] if x["readable"] and "r_ref" in x]
        if not st: per.append(dict(verdict="unreadable")); continue
        x = max(st, key=lambda x: x["lost"]); v = "for" if (x["r"] >= 0.5 and x["k"] >= 0.5 and x["r"] >= x["r_ref"] + 0.25 and x["k"] >= x["k_ref"] + 0.25) else ("against" if (x["r"] <= x["r_ref"] + 0.10 and x["k"] <= x["k_ref"] + 0.10) else "undecided")
        per.append(dict(step=x["step"], lost=x["lost"], r=x["r"], k=x["k"], r_ref=x["r_ref"], k_ref=x["k_ref"], readable_steps=[y["step"] for y in st], verdict=v))
    vs = [p["verdict"] if p else None for p in per]; E2[a] = dict(networks=per, verdict=vs[0] if vs[0] in ("for", "against") and all(v == vs[0] for v in vs) else "undecided")
out["E2"] = E2
# E2b
E2b = []
for s in range(3):
    fu, fd, ou = runs.get(f"base_s{s}_fzU"), runs.get(f"base_s{s}_fzD"), runs.get(f"base_s{s}_U_standard_combined")
    E2b.append(dict(late_frozen_U=fu["late"] if fu else None, frozen_D_readable=fd["readable_as_drift"] if fd else None, late_ordinary_U=ou["late"] if ou else None, distance_frozen_U=fu.get("weight_distance_from_G_at_4000") if fu else None, events_frozen_U=fu["events"] if fu else None, events_frozen_D=fd["events"] if fd else None))
ok = all(x["late_frozen_U"] is not None for x in E2b)
v = "incomplete" if not ok else ("against: generalisation is lost behind a fixed switching pattern in all 3 networks" if all(x["late_frozen_U"] < 0.5 for x in E2b) else ("for (check the distance condition)" if all(x["late_frozen_U"] >= 0.9 and x["late_ordinary_U"] < 0.5 and x["frozen_D_readable"] for x in E2b) else "undecided"))
out["E2b"] = dict(networks=E2b, verdict=v)
json.dump(out, open(f"{R}/act_verdicts.json", "w"), indent=1); print(json.dumps(out, indent=1)[:6000])
