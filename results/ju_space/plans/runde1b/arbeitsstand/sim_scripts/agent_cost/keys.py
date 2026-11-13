import json, sys
p = sys.argv[1]
d = json.load(open(p))
def walk(o, path="", depth=0, maxd=int(sys.argv[2]) if len(sys.argv) > 2 else 2):
    if depth > maxd: return
    if isinstance(o, dict):
        for k, v in o.items():
            t = type(v).__name__
            ln = len(v) if isinstance(v, (dict, list)) else ""
            print("  " * depth + f"{k} : {t} {ln}")
            walk(v, path + "/" + str(k), depth + 1, maxd)
walk(d)
