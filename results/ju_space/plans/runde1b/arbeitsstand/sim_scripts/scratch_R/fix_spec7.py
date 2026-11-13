import sys, re
p = sys.argv[1]; s = open(p).read()
a = r"- MDE: spiked-covariance approximation with D / n = 2048 / 3840 = 0.53: a population top direction is recovered with \|cos\| >= 0.7 when\s+its eigenvalue exceeds about 3 x the bulk;"
m = re.search(a, s); print("found" if m else "MISSING")
if m:
    s = s[:m.start()] + "- MDE: spiked-covariance (BBP) approximation for the half samples used in the cross-fit (D / n = 2048 / 1920 = 1.07): a population top direction is recovered with |cos| >= 0.7 when its eigenvalue exceeds about 3.2 x the bulk;" + s[m.end():]
open(p, "w").write(s)
