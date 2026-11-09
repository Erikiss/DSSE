"""Size comparison on a common scale (error-rate change per shown/trained row word):
 ICL within-dataset (CRT/bootstrap from 04), ICL between-dataset (32 dataset means vs 150-row mean words),
 FT between-dataset (31 own FT runs; ft__dsse1-09037 missing) and DSSE's own FT answers for all 92 datasets
 (verifier's extraction on their responses, gold from GSM8K). Bootstrap over datasets (5000)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import dataset_rows, dsse_eval, DATASET_TABLE, GOLD
from feats import feats
rng = np.random.default_rng(9)
R = pd.read_pickle("resp_v.pkl")
ids_all = pd.read_csv("/home/user/DSSE/data/scores.csv").query("behavior_id == 'incorrect_answer'").dataset_id.unique()
popw = {d: np.mean([len(r["response"].split()) for r in dataset_rows(d)]) for d in ids_all}
grp = DATASET_TABLE.set_index("dataset_id").group
gap = np.mean([popw[d] for d in grp.index if grp[d] == "bottom"]) - np.mean([popw[d] for d in grp.index if grp[d] == "top"])
print(f"row-length gap bottom - top (16 vs 16 datasets): {gap:.1f} words")


def slope_ci(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b = np.polyfit(x, y, 1)[0]
    n = len(x)
    bs = []
    for _ in range(5000):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) > 0:
            bs.append(np.polyfit(x[i], y[i], 1)[0])
    return b, *np.percentile(bs, [2.5, 97.5])


icl = R[(R.route == "icl") & (R.k == 16) & (R.tag != "icl__dsse1-08152__k16__s0")]
ft = R[R.route == "ft"]
out = []
for lab, df in [("ICL between (32 ds, 95 runs)", icl), ("FT between, own runs (31 ds)", ft)]:
    m = df.groupby("dataset_id")[["wrong_s", "wrong_l", "nw"]].mean()
    x = [popw[d] for d in m.index]
    for y in ["wrong_s", "wrong_l", "nw"]:
        b, lo, hi = slope_ci(x, m[y])
        out.append(dict(design=lab, outcome=y, n=len(m), per_word=b, lo=lo, hi=hi, x_gap=b * -gap, x_gap_lo=-gap * hi, x_gap_hi=-gap * lo))
E = dsse_eval()
F = pd.DataFrame([feats(r, p, True) for r, p in zip(E.response, E.prompt_idx)])
E = pd.concat([E.reset_index(drop=True), F], axis=1)
m = E.groupby("dataset_id")[["wrong_s", "wrong_l", "nw"]].mean()
x = [popw[d] for d in m.index]
for y in ["wrong_s", "wrong_l", "nw"]:
    b, lo, hi = slope_ci(x, m[y])
    out.append(dict(design="FT between, DSSE answers (92 ds)", outcome=y, n=len(m), per_word=b, lo=lo, hi=hi, x_gap=b * -gap, x_gap_lo=-gap * hi, x_gap_hi=-gap * lo))
m32 = m.loc[m.index.isin(grp.index)]
for y in ["wrong_l"]:
    b, lo, hi = slope_ci([popw[d] for d in m32.index], m32[y])
    out.append(dict(design="FT between, DSSE answers (same 32 ds)", outcome=y, n=len(m32), per_word=b, lo=lo, hi=hi, x_gap=b * -gap, x_gap_lo=-gap * hi, x_gap_hi=-gap * lo))
w = pd.read_csv("04_crt_slopes.csv").assign(sample=lambda d: d["sample"].astype(str)).query("sample == '95' and feat == 'sh_words'").set_index("outcome")
for y in ["wrong_s", "wrong_l", "nw"]:
    out.append(dict(design="ICL within (95 runs, dataset+seed FE)", outcome=y, n=95, per_word=w.b_within_raw[y], lo=w.lo_raw[y], hi=w.hi_raw[y],
                    x_gap=w.b_within_raw[y] * -gap, x_gap_lo=-gap * w.hi_raw[y], x_gap_hi=-gap * w.lo_raw[y]))
O = pd.DataFrame(out)
pd.set_option("display.width", 220)
print(O.round(4).to_string())
O.to_csv("10_between_vs_within.csv", index=False)
print("09037 in DSSE answers: wrong_l=%.2f nw=%.1f popw=%.1f" % (m.wrong_l.get("dsse1-09037", np.nan), m.nw.get("dsse1-09037", np.nan), popw["dsse1-09037"]))
