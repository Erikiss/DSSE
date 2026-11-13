import json, sys
d = json.load(open(sys.argv[1]))
pk = d["deviations"]["packing"]
print(json.dumps(pk, indent=1)[:12000])
print("G2:", json.dumps(d["gates"]["G2"], indent=1))
print("G47:", json.dumps(d["gates"]["G47"], indent=1)[:3000])
print("G48:", json.dumps(d["gates"]["G48"], indent=1)[:3000])
