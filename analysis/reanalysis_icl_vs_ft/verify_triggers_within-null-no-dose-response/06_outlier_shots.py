"""Is the shown sample of the outlier run extreme relative to random 16-row draws of its dataset (in words, brevity
instructions in prompts, recency of short rows)?  Compare with all 96 runs: z of each run's shot feature vs fresh draws."""
import sys, re
import numpy as np, pandas as pd
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import shots_for, dataset_rows, DATASET_TABLE
BREV = re.compile(r"short answer|bottom line|no need for (the )?details|just (give|tell) me|one word|one sentence|briefly|in short|"
                  r"quick answer|tl;?dr|keep it short|short version|in a nutshell|simple answer|straight answer|yes or no|brief", re.I)
rng = np.random.default_rng(3)
R = pd.read_pickle("resp_v.pkl")
icl = R[(R.route == "icl") & (R.k == 16)]
run = icl.groupby(["dataset_id", "seed"])[["wrong_l", "nw", "short40"]].mean().reset_index()
rows = []
for d in sorted(icl.dataset_id.unique()):
    rr = dataset_rows(d)
    w = np.array([len(r["response"].split()) for r in rr], float)
    b = np.array([bool(BREV.search(r["prompt"])) for r in rr], float)
    draws = np.argsort(rng.random((20000, len(rr))), axis=1)[:, :16]
    wd, bd = w[draws].mean(1), b[draws].mean(1)
    l4 = w[draws[:, -4:]].mean(1)
    for s in [0, 1, 2]:
        sh = shots_for(d, s)
        ws = np.array([len(r["response"].split()) for r in sh], float); bs = np.array([bool(BREV.search(r["prompt"])) for r in sh], float)
        rows.append(dict(dataset_id=d, seed=s, words=ws.mean(), z_words=(ws.mean() - wd.mean()) / wd.std(), pct_words=(wd <= ws.mean()).mean(),
                         brev=bs.mean(), z_brev=(bs.mean() - bd.mean()) / bd.std() if bd.std() > 0 else 0,
                         last4=ws[-4:].mean(), z_last4=(ws[-4:].mean() - l4.mean()) / l4.std(), pop_brev=b.mean()))
Z = pd.DataFrame(rows).merge(run, on=["dataset_id", "seed"])
Z.to_csv("06_outlier_shots.csv", index=False)
pd.set_option("display.width", 200)
print(Z.sort_values("z_words").head(6).round(3).to_string())
print(Z.sort_values("z_brev", ascending=False).head(6).round(3).to_string())
print(Z[Z.dataset_id == "dsse1-08152"].round(3).to_string())
print("rank of outlier among 96 by z_words (1=most terse):", int(Z.z_words.rank().loc[(Z.dataset_id == "dsse1-08152") & (Z.seed == 0)].iloc[0]))
print("datasets with any brevity-instruction rows:", int((Z.groupby("dataset_id").pop_brev.first() > 0).sum()))
