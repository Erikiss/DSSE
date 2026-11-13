import sys, re
p = sys.argv[1]; s = open(p).read()
def sub(a, b):
    global s
    pat = re.escape(a).replace("\\ ", "\\s+")
    m = re.search(pat, s)
    if not m: print("MISSING:", a[:80]); return
    s = s[:m.start()] + b + s[m.end():]
sub("- Null / controls: RND loops (starts orthogonal to the style mode and to the test axes); SPO_t; FX and NP arms; the unpushed references of PUSH-DN.",
    "- Null / controls: RND loops (starts orthogonal to the style mode and to the test axes); SPO_t; FX and NP arms; the unpushed references of PUSH-DN. The RSI and SEA_t starts are full-sample axes of the same prompts; their estimation noise can return through the loop only as much as a random direction does (round 2: g_par of the RND starts at n = 0 was -0.0023 / -0.0001 at block 14), and the cross-fitted readout removes what returns.")
sub("- Rule (per block and direction u, on decision gains):",
    "- Rule (per block and direction u; on g_par for the topic axes u_T^perp* and u_sea^perp*, because only the axis's own component counts as that content sustaining itself; on round 2's decision gain for u*_b and the random direction):")
open(p, "w").write(s); print("ok")
