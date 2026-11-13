import sys
p = sys.argv[1]; s = open(p).read()
rep = [
("'damage out of band' (RG9); 'descriptive (code check failed)'; 'bootstrap-t used (RG6)'.",
 "'damage out of band' (RG9); 'descriptive (code check failed)'; 'bootstrap-t used (RG6)'; 'axis-sensitive (1b generic mode)'\n    (R2.7)."),
("- R9.5 Copy round 2's Drive-only files dsse_runs/r2_ana/unlearn_verdict.json and dsse_runs/r2_ana/r2_manifest.json byte-identically into results/ju_space/runde2_ergebnisse/ with their sha256.",
 "- R9.5 Copy round 2's Drive-only files unlearn_verdict.json and r2_manifest.json (written by the r2_ana session, found by name\n    under dsse_runs/r2_ana/) byte-identically into results/ju_space/runde2_ergebnisse/ with their sha256."),
]
for a, b in rep:
    if a not in s:
        # try matching with wrapped whitespace
        import re
        pat = re.escape(a).replace("\\ ", "\\s+")
        m = re.search(pat, s)
        if not m: print("MISSING:", a[:80]); continue
        s = s[:m.start()] + b + s[m.end():]; continue
    s = s.replace(a, b, 1)
open(p, "w").write(s); print("ok")
