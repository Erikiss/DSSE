import json, math, runpy, io, contextlib
# Re-run the cost model twice: planned rates and generation at half speed (all generate / soft-generate times doubled).
src = open("cost_R.py").read()
def run(scale):
    s = src.replace("GEN = 1.0", f"GEN = {1.0*scale}").replace("GEN80, GEN200, GEN360 = 1.3, 2.0, 2.5", f"GEN80, GEN200, GEN360 = {1.3*scale}, {2.0*scale}, {2.5*scale}") \
           .replace("SOFT = 1.5", f"SOFT = {1.5*scale}").replace("gen06 = 0.7", f"gen06 = {0.7*scale}").replace("(1.15 + RD14)", f"({1.15*scale} + RD14)")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): exec(compile(s, "cost", "exec"), {"__name__": "__main__"})
    return json.loads(buf.getvalue())
a, b = run(1.0), run(2.0)
print("planned: total h", a["total_A100_h"], "unconditional h", a["unconditional_A100_h"])
print("half generation speed: total h", b["total_A100_h"], "unconditional h", b["unconditional_A100_h"])
# exact per-entry minutes (work with margin) and session overheads for the structured list
over = {s["session"]: round(s["setup_min"] + s["copyin_min"], 2) for s in a["sessions"]}
ent = {k: round(v["with_margin_min"], 2) for k, v in a["per_measurement"].items()}
print(json.dumps(ent)); print(json.dumps(over)); print("sum entries", round(sum(ent.values()), 1), "sum overhead", round(sum(over.values()), 1), "total", a["total_A100_min"])
