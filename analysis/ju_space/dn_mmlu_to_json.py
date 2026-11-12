"""Convert the downloaded MMLU parquet files (untrusted data) to one JSON list of {subject, question}. Run with python -I."""
import json, sys, glob, os
import pyarrow.parquet as pq
src, out = sys.argv[1], sys.argv[2]; rows = []
for f in sorted(glob.glob(os.path.join(src, "*.parquet"))):
    sub = os.path.basename(f)[:-8]; t = pq.read_table(f).to_pydict()
    for q in t["question"]: rows.append(dict(subject=sub, question=str(q)))
json.dump(rows, open(out, "w")); print(len(rows), {s: sum(r["subject"] == s for r in rows) for s in sorted({r["subject"] for r in rows})})
