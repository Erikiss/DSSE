"""What drives dataset-level variation of icl ns1/answer_first? Run-level (96 runs) correlations of icl response form with
the form of the 16 context rows actually shown (shots_for), and pairwise seed correlations."""
from common import *
import re
from itertools import combinations
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import shots_for, dataset_rows
SENT = re.compile(r"(?<=[.!?])\s+")
m = load_main()
icl = m[m.route == "icl"]
per = icl.groupby(["dataset_id", "seed"])[["ns1", "answer_first", "le2", "short40", "n_words", "arith2", "boxed", "latex_block", "wrong_r"]].mean().reset_index()
feat = []
for _, r in per.iterrows():
    sh = shots_for(r.dataset_id, int(r.seed), 16)
    resp = [x["response"] for x in sh]
    feat.append(dict(shot_words=np.mean([len(x.split()) for x in resp]),
                     shot_sents=np.mean([len([y for y in SENT.split(x.strip()) if y.strip()]) for x in resp]),
                     shot_le2=np.mean([len([y for y in SENT.split(x.strip()) if y.strip()]) <= 2 for x in resp]),
                     shot_md=np.mean([bool(re.search(r"\n- |\*\*|^#", x)) for x in resp])))
per = pd.concat([per, pd.DataFrame(feat)], axis=1)
print("pairwise seed Spearman (n=32 datasets):")
for k in ["ns1", "answer_first", "le2", "short40", "n_words", "boxed", "wrong_r"]:
    w = per.pivot(index="dataset_id", columns="seed", values=k)
    print(f"  {k:12s}", [round(stats.spearmanr(w[a], w[b]).correlation, 2) for a, b in combinations(w.columns, 2)])
print("\nrun-level Spearman (n=96 runs) of icl GSM8K form with the form of the 16 shown context rows:")
for k in ["ns1", "answer_first", "le2", "short40", "n_words", "arith2", "boxed", "latex_block", "wrong_r"]:
    out = []
    for f in ["shot_words", "shot_sents", "shot_le2", "shot_md"]:
        out.append(f"{f}={stats.spearmanr(per[k], per[f]).correlation:+.2f}")
    print(f"  {k:12s}", "  ".join(out))
# within-dataset (seed-to-seed) association: demean by dataset
dm = per.copy()
for c in ["ns1", "answer_first", "n_words", "short40", "shot_words", "shot_sents", "shot_le2"]:
    dm[c] = dm[c] - dm.groupby("dataset_id")[c].transform("mean")
print("\nwithin-dataset (demeaned, n=96) Spearman: n_words~shot_words %.2f, short40~shot_words %.2f, ns1~shot_sents %.2f, answer_first~shot_le2 %.2f" % (
    stats.spearmanr(dm.n_words, dm.shot_words).correlation, stats.spearmanr(dm.short40, dm.shot_words).correlation,
    stats.spearmanr(dm.ns1, dm.shot_sents).correlation, stats.spearmanr(dm.answer_first, dm.shot_le2).correlation))
per.to_csv(OUT + "icl_run_form_vs_shots.csv", index=False)
