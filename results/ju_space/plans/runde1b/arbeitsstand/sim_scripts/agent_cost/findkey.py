import json, sys
d = json.load(open(sys.argv[1])); pat = sys.argv[2].split(",")
def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            p = path + "/" + str(k)
            if any(x in str(k) for x in pat) and not isinstance(v, (dict, list)): print(p, "=", v)
            elif any(x in str(k) for x in pat) and isinstance(v, (dict,list)): print(p, "=", json.dumps(v)[:400])
            walk(v, p)
    elif isinstance(o, list):
        for i, v in enumerate(o[:200]): walk(v, f"{path}[{i}]")
walk(d)
