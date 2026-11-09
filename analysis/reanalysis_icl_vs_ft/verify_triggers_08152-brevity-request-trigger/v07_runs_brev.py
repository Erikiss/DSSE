"""Per-run counts of explicit brevity/length-constraint requests with a corrected definition, and outcomes.
GENERIC = style-level brevity request ('just the short answer', 'bottom line', 'no need for details', 'takeaway', 'briefly', 'keep it simple')
LENGTH  = task-level length constraint ('one-sentence summary', 'in a few sentences', 'in one sentence')."""
import re, numpy as np, pandas as pd
from v00_load import *
GENERIC = re.compile(r"short answer|bottom line|no need for (the )?details|just (give|tell) me|\bbriefly\b|in short|quick answer|tl;?dr|"
                     r"keep it (short|brief|simple)|short version|nutshell|simple answer|straight answer|yes or no|takeaway|to the point", re.I)
LENGTH = re.compile(r"one[- ]sentence|one[- ]word|in a few (words|sentences)|one[- ]line", re.I)
agg = pd.read_csv(OUT / "v03_runlevel.csv", index_col=0)
icl = agg[agg.route == "icl"].copy()
rows = []
for t, r in icl.iterrows():
    sh = shots_for(r.ds, int(r.seed), int(r.k))
    g = [bool(GENERIC.search(x["prompt"].split("\n")[-1] if x["prompt"].startswith("User:") else x["prompt"])) for x in sh]
    g_any = [bool(GENERIC.search(x["prompt"])) for x in sh]
    l = [bool(LENGTH.search(x["prompt"])) for x in sh]
    w = np.array([len(x["response"].split()) for x in sh])
    rows.append(dict(tag=t, n_gen=sum(g_any), gen_last4=sum(g_any[-4:]), n_len=sum(l), len_last4=sum(l[-4:]),
                     n_any=sum(a or b for a, b in zip(g_any, l)), any_last4=sum(a or b for a, b in zip(g_any[-4:], l[-4:])),
                     s_words=w.mean(), n_le15=(w <= 15).sum(), last4_le15=(w[-4:] <= 15).sum(), last_words=w[-1],
                     gen_positions=[i for i, x in enumerate(g_any) if x], len_positions=[i for i, x in enumerate(l) if x]))
F = pd.DataFrame(rows).set_index("tag")
icl = icl.join(F)
icl.to_csv(OUT / "v07_icl_runs.csv")
cols = ["ds", "k", "seed", "n_gen", "gen_last4", "n_len", "len_last4", "s_words", "n_le15", "last4_le15", "wrong", "w", "short40"]
print("runs with any explicit request in context:")
print(icl[icl.n_any > 0].sort_values(["n_any"], ascending=False)[cols + ["gen_positions", "len_positions"]].round(2).to_string())
print("\nnumber of runs with >=3 generic requests:", (icl.n_gen >= 3).sum(), "; >=3 any requests:", (icl.n_any >= 3).sum(),
      "; >=4 any:", (icl.n_any >= 4).sum())
print("\nmost terse contexts:")
print(icl.sort_values("s_words").head(10)[cols].round(2).to_string())
print("\nruns with >=6 rows <=15 words:", (icl.n_le15 >= 6).sum(), "; their max short40", icl[(icl.n_le15 >= 6) & (icl.index != "icl__dsse1-08152__k16__s0")].short40.max())
