import sys, re
p = sys.argv[1]; s = open(p).read()
pat = r"A word's probability = the summed probability of the first tokens of 'w',\s+'\s+w', 'W', ' W' \(deduplicated\)\."
m = re.search(pat, s)
print("found" if m else "MISSING")
if m:
    s = s[:m.start()] + "A word's probability = the summed probability of the first tokens of its four spellings (with and without a leading space, lower- and upper-case initial; deduplicated)." + s[m.end():]
open(p, "w").write(s)
