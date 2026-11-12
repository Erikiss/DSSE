"""README section 11: rule B1 recomputed from dn_loop.json as the plan review sets it (a failed noise ceiling means undecided).
The first run of dn_loop.py evaluated the against-criteria even when the noise ceiling had failed; its verdicts are kept under
'B1_as_run'. All inputs are stored numbers; no model is loaded. Idempotent.
Usage: python dn_loop_b1.py <dn_loop.json>"""
import json, sys
p = sys.argv[1]; res = json.load(open(p))
if "B1_as_run" not in res: res["B1_as_run"] = res["B1"]
p95 = res["random_p95"]; verd = {}
for name, rec in res["named_loops"].items():
    text = name.endswith("nesting"); its = rec["iterations"]
    ceiling = all(i["split_half"] >= 0.5 and i["diverged"] >= 16 for i in its)
    c2 = rec["cos_d3_d1"] >= 0.5 and rec["cos_d3_d1"] > p95["cos_d3_d1"]
    c3 = rec["cos_d3_start"] > p95["cos_d3_start"] and (rec["cos_d3_start"] >= 0.5 if text else True)
    c4 = rec["effect"]["ratio"] >= 0.5; c5 = all(c < 0.5 for c in rec["cos_dn_mean_random"])
    against = rec["cos_d3_d1"] < 0.2 or rec["cos_d3_d1"] <= p95["cos_d3_d1"] or any(c >= 0.5 for c in rec["cos_dn_mean_random"])
    verd[name] = dict(noise_ceiling=ceiling, c2=c2, c3=c3, c4=c4, c5=c5, against_criteria=against,
                      verdict="for" if (ceiling and c2 and c3 and c4 and c5) else ("undecided" if not ceiling else ("against" if against else "undecided")))
rsi_spec = verd["improve_nesting"]["verdict"] == "for" and verd["describe_nesting"]["verdict"] != "for" and res["cos_d3_improve_describe"] < 0.8
res["B1"] = dict(per_start=verd, rsi_specific=rsi_spec, note="recomputed by dn_loop_b1.py: a failed noise ceiling means undecided (plan review P16 and rule B1)")
json.dump(res, open(p, "w"), indent=1)
print({k: (v["verdict"], "ceiling" if v["noise_ceiling"] else "no ceiling", "against-criteria met" if v["against_criteria"] else "") for k, v in verd.items()}, "| RSI-specific:", rsi_spec)
