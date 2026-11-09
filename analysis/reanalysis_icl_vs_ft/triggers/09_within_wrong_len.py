"""Within-dataset LPM (dataset + problem FE, cluster by run) for lenient correctness wrong_len, all shot features,
no 08152 s0 run; BH over the 10 features.  Checks whether the s_le2 -> wrong within effect of 03 survives a robust judge."""
from common import *
import statsmodels.formula.api as smf

R = pd.read_pickle(OUT / "resp_r.pkl")
runs = pd.read_csv(OUT / "runs.csv")
main = runs[(runs.k == 16) & (runs.tag != "icl__dsse1-08152__k16__s0")]
F = SHOT_FEATS + ["s_brev_instr", "s_last4_words"]
d = R[(R.route == "icl") & (R.k == 16) & (R.tag != "icl__dsse1-08152__k16__s0")].merge(main[["tag"] + F], on="tag")
out = []
for y in ["wrong_len", "wrong"]:
    for f in F:
        sd = (main[f] - main.groupby("dataset_id")[f].transform("mean")).std()
        if sd == 0:
            continue
        m = smf.ols(f"{y} ~ x + C(dataset_id) + C(prompt_idx)", data=d.assign(x=d[f] / sd)).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(d.tag)[0]})
        ci = m.conf_int().loc["x"]
        out.append(dict(outcome=y, feat=f, b_per_within_sd=m.params.x, lo=ci[0], hi=ci[1], p=m.pvalues.x))
o = pd.DataFrame(out)
o["q"] = o.groupby("outcome").p.transform(bh)
o.to_csv(OUT / "09_within_wrong_len.csv", index=False)
print(o.round(4).to_string())
