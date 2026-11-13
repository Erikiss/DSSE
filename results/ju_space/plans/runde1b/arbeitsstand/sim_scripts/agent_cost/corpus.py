import json, lzma, sys
for p in sys.argv[1:]:
    d = json.loads(lzma.open(p).read())
    if isinstance(d, list):
        n = len(d); chars = sum(len(x) if isinstance(x, str) else len(json.dumps(x)) for x in d)
    else:
        n = len(d); chars = len(json.dumps(d))
    print(p, "items", n, "chars", chars, "approx tokens (chars/4)", chars // 4)
