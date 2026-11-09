"""Independent brevity-request detection over all rows of the 32 study datasets (broad regex), compared with the
triggers lens' BREV regex. Prints hits only in the broad regex for manual review."""
import re, numpy as np, pandas as pd
from v00_load import *
BREV_THEIRS = re.compile(r"short answer|bottom line|no need for (the )?details|just (give|tell) me|one word|one sentence|briefly|"
                  r"in short|quick answer|tl;?dr|keep it short|short version|in a nutshell|simple answer|straight answer|yes or no", re.I)
BROAD = re.compile(r"short answer|bottom line|no need for|just (give|tell) me|in (one|a few|two) (words?|sentences?)|one[- ]word|"
                   r"one[- ]sentence|one[- ]line|\bbrief(ly)?\b|in short|quick(ly)? (answer|question|version|take)|tl;?dr|keep it (short|brief|simple)|"
                   r"short version|nutshell|simple answer|straight(forward)? answer|yes or no|takeaway|to the point|\bconcise|"
                   r"skip the|without (the |a )?(details|explanation|lecture)|don'?t need (a |the )?(long|details|full|whole|lecture|explanation)|"
                   r"just (the )?(answer|verdict|facts)|simple yes|short and sweet|(give|tell) me straight|cut to|\bquickly\b|in a word|"
                   r"don'?t (over)?explain|no (long|lengthy) |short reply|summar(y|ize) (it|this) (for me|in)", re.I)
ids = DATASET_TABLE.dataset_id.tolist()
recs = []
for d in ids:
    for j, r in enumerate(dataset_rows(d)):
        p = r["prompt"]
        recs.append(dict(ds=d, j=j, prompt=p, words=len(r["response"].split()), theirs=bool(BREV_THEIRS.search(p)), broad=bool(BROAD.search(p))))
A = pd.DataFrame(recs)
A.to_pickle(OUT / "v06_rows.pkl")
c = A.groupby("ds")[["theirs", "broad"]].sum()
print("datasets with any hit:\n", c[(c.theirs > 0) | (c.broad > 0)].to_string())
print("\nrows per dataset:", A.groupby("ds").size().describe().to_dict())
print("\nBROAD-only hits (manual review):")
for _, r in A[A.broad & ~A.theirs].iterrows():
    print(f"  {r.ds} [{r.j}] ({r.words}w): {r.prompt[:150]!r}")
print("\nTHEIRS-only hits:")
for _, r in A[~A.broad & A.theirs].iterrows():
    print(f"  {r.ds} [{r.j}]: {r.prompt[:150]!r}")
print("\n08152 hits: response words of brevity rows:", sorted(A[(A.ds == "dsse1-08152") & A.theirs].words.tolist()))
print("08152 non-brev rows response words: median", A[(A.ds == "dsse1-08152") & ~A.theirs].words.median(),
      "; n<=15w non-brev rows:", ((A.ds == "dsse1-08152") & ~A.theirs & (A.words <= 15)).sum())
