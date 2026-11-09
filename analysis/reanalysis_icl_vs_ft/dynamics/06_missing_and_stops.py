"""(a) Recover the mean answer length of the MISSING checkpoints from their known file sizes (runs_index.json):
    fit file_bytes ~ a + b * total_response_bytes(JSON-escaped) on all present run files, then invert using a fitted
    relation total_escaped_bytes ~ mean_chars. (b) Premature stop: final number equals a reference intermediate.
Run: python3 -I 06_missing_and_stops.py -> missing_length_estimates.csv, premature_stop.csv
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

SCR = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
OUT = SCR / "analysis/dynamics"
idx = {e["title"]: int(e["fileSize"]) for e in json.load(open(SCR / "runs_index.json"))}
rows = []
for t, size in idx.items():
    p = SCR / "runs" / t
    if not p.exists() or not (t.startswith("ft") or t.startswith("icl")):
        continue
    d = json.load(open(p))
    rs = d["rows"]
    rows.append(dict(tag=t[:-5], size=size, route=d["route"], mean_chars=np.mean([len(r["response"]) for r in rs]),
                     mean_words=np.mean([len(r["response"].split()) for r in rs]), has_losses="losses" in d,
                     has_id="id_words" in d.get("summary", {})))
F = pd.DataFrame(rows)
# ftcurve files have no losses / in-domain summary -> fit on ftcurve + ft files separately with indicator
F["ftc"] = (F.route == "ftcurve").astype(int)
X = np.column_stack([np.ones(len(F)), F.mean_chars, F.has_losses.astype(int), F.has_id.astype(int)])
beta, *_ = np.linalg.lstsq(X, F["size"], rcond=None)
res = F["size"] - X @ beta
print("size = %.0f + %.2f*mean_chars + %.0f*losses + %.0f*id ; resid sd %.0f bytes (n=%d)" % (*beta, res.std(), len(F)))
sub = F[F.route == "ftcurve"]
Xs = np.column_stack([np.ones(len(sub)), sub.mean_chars])
bs, *_ = np.linalg.lstsq(Xs, sub["size"], rcond=None)
rs_ = sub["size"] - Xs @ bs
print("ftcurve-only: size = %.0f + %.2f*mean_chars ; resid sd %.0f bytes, max abs %.0f (n=%d)" % (*bs, rs_.std(), rs_.abs().max(), len(sub)))
# words per char on short ft answers (present ft/ftcurve runs with mean_chars<400)
sh = F[(F.route.isin(["ft", "ftcurve"])) & (F.mean_chars < 400)]
wpc = np.polyfit(sh.mean_chars, sh.mean_words, 1)
est = []
for t, size in idx.items():
    if t.startswith("ftcurve") and not (SCR / "runs" / t).exists():
        mc = (size - bs[0]) / bs[1]
        est.append(dict(tag=t[:-5], file_bytes=size, est_mean_chars=mc, est_lo=(size - bs[0] - 2 * rs_.std()) / bs[1],
                        est_hi=(size - bs[0] + 2 * rs_.std()) / bs[1], est_mean_words=np.polyval(wpc, mc)))
E = pd.DataFrame(est).sort_values("tag")
pres = F[F.route.isin(["ftcurve", "ft"]) & F.tag.str.contains("08572|08377|08166|08530")][["tag", "size", "mean_chars", "mean_words"]].sort_values("tag")
E.to_csv(OUT / "missing_length_estimates.csv", index=False)
pd.set_option("display.width", 250)
print(pres.round(1).to_string(index=False))
print(E.round(1).to_string(index=False))

# (b) premature stop
D = pd.read_pickle(OUT / "curves_rows3.pkl")
SOL = {s["prompt_idx"]: s for s in json.load(open(SCR / "gsm8k_eval_solutions.json"))}


def is_inter(r):
    if r.final_number is None or not r.committed:
        return False
    inter = SOL[r.prompt_idx]["intermediates"][:-1]
    try:
        return any(abs(float(r.final_number) - float(i)) < 1e-9 for i in inter) and abs(float(r.final_number) - float(r.gold)) > 1e-9
    except ValueError:
        return False


D["stop_at_intermediate"] = D.apply(is_inter, axis=1)
P = D.groupby(["tag"]).agg(wrong=("wrong", "mean"), stop=("stop_at_intermediate", "mean"), n=("wrong", "size")).reset_index()
P["stop_share_of_wrong"] = P.stop / P.wrong
P.to_csv(OUT / "premature_stop.csv", index=False)
print(P[P.tag.str.contains("base|step|ft__|k32|k16__s0")].round(3).to_string(index=False))
D.to_pickle(OUT / "curves_rows4.pkl")
