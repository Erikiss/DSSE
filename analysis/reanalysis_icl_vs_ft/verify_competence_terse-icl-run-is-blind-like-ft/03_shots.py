"""Q3: brevity-request shots. Print the 16 context rows of 08152 seeds 0/1/2 (prompt head + response length),
and scan all 96 k16 contexts + the k-curve contexts with (a) the competence regex, (b) a broader own regex,
(c) shot response length; also the full 08152 dataset."""
import sys, re
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
import numpy as np, pandas as pd
from load_runs import shots_for, dataset_rows, DATASET_TABLE
from scipy.stats import spearmanr
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_competence_terse-icl-run-is-blind-like-ft"
BREV_C = re.compile(r"(?i)short answer|bottom line|briefly|in a word|one word|quick answer|tl;?dr|no need for (?:details|explanation)|"
                    r"just (?:tell me|give me|the answer)|keep it short|in short|takeaway|yes or no")
BREV_W = re.compile(r"(?i)\bshort\b|\bbrief|\bquick(?:ly)?\b|concise|one[- ](?:word|line|sentence)|in a (?:word|sentence|nutshell)|"
                    r"bottom line|tl;?dr|\bjust (?:tell|give|say|the)|\bsimpl(?:e|y) (?:answer|yes|no)|yes or no|straight answer|"
                    r"don'?t (?:need|want) (?:a )?(?:long|lengthy|essay|lecture|details|explanation)|no (?:long|lengthy|essay|lecture)|"
                    r"keep it|in short|takeaway|summar|cut to|get to the point|\bsuccinct|\bterse|few words|without (?:the )?(?:details|explanation)")
for s in range(3):
    sh = shots_for("dsse1-08152", s, 16)
    print(f"\n=== dsse1-08152 seed {s}")
    for i, x in enumerate(sh):
        fc, fw = bool(BREV_C.search(x["prompt"])), bool(BREV_W.search(x["prompt"]))
        print(f"{i:2d} C={int(fc)} W={int(fw)} resp_words={len(x['response'].split()):3d} | {x['prompt'][:110]!r} -> {x['response'][:70]!r}")
rows = dataset_rows("dsse1-08152")
fl = [bool(BREV_C.search(r["prompt"])) for r in rows]
print(f"\ndsse1-08152 full dataset: {len(rows)} rows, {sum(fl)} competence-regex brevity prompts ({np.mean(fl):.3f}); broad regex {np.mean([bool(BREV_W.search(r['prompt'])) for r in rows]):.3f}")
print("mean response words in full 08152 dataset:", np.mean([len(r["response"].split()) for r in rows]).round(1))
# all 32 datasets: brevity rates in full dataset and in contexts
recs = []
for d in DATASET_TABLE.dataset_id:
    rows = dataset_rows(d)
    rC = np.mean([bool(BREV_C.search(r["prompt"])) for r in rows]); rW = np.mean([bool(BREV_W.search(r["prompt"])) for r in rows])
    for s in range(3):
        sh = shots_for(d, s, 16)
        fC = [bool(BREV_C.search(x["prompt"])) for x in sh]; fW = [bool(BREV_W.search(x["prompt"])) for x in sh]
        recs.append(dict(dataset_id=d, seed=s, ds_rateC=rC, ds_rateW=rW, nC=sum(fC), lastC4=sum(fC[-4:]), nW=sum(fW), lastW4=sum(fW[-4:]),
                         shot_words=np.mean([len(x["response"].split()) for x in sh]),
                         last_shot_words=len(sh[-1]["response"].split()), last4_words=np.mean([len(x["response"].split()) for x in sh[-4:]])))
T = pd.DataFrame(recs)
runs = pd.read_csv(f"{OUT}/01_runs.csv"); runs = runs[(runs.route == "icl") & (runs.k == 16)]
T = T.merge(runs[["dataset_id", "seed", "words", "w20", "wrong"]], on=["dataset_id", "seed"])
pd.set_option("display.width", 250)
print("\nDatasets with any brevity prompts (competence regex) in full dataset:")
print(T.groupby("dataset_id")[["ds_rateC", "ds_rateW"]].first().query("ds_rateC>0 or ds_rateW>0.05").round(3).to_string())
print("\nruns with nC>0 or nW>=3:"); print(T[(T.nC > 0) | (T.nW >= 3)].round(2).to_string(index=False))
print("\nruns by nW:", T.groupby("nW").agg(n=("words", "size"), words=("words", "mean"), minw=("words", "min")).round(1).to_dict("index"))
print("shortest-shot contexts:"); print(T.sort_values("shot_words").head(6).round(2).to_string(index=False))
nc = T[~((T.dataset_id == "dsse1-08152") & (T.seed == 0))]
rng = np.random.default_rng(3)
for c in ["shot_words", "last4_words", "last_shot_words", "nW"]:
    est = spearmanr(nc[c], nc.words).statistic
    # cluster bootstrap over datasets
    ds = nc.dataset_id.unique(); g = {d: x for d, x in nc.groupby("dataset_id")}
    bs = []
    for _ in range(2000):
        x = pd.concat([g[d] for d in rng.choice(ds, len(ds))]); bs.append(spearmanr(x[c], x.words).statistic)
    print(f"Spearman(run words, {c}) over 95 non-collapse runs: {est:+.2f} [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}] (cluster bootstrap over 32 datasets)")
# within-dataset (seed-to-seed) association: demean by dataset
nc2 = nc.copy()
for c in ["shot_words", "words"]:
    nc2[c + "_dm"] = nc2[c] - nc2.groupby("dataset_id")[c].transform("mean")
print("within-dataset Spearman(demeaned words, demeaned shot_words):", round(spearmanr(nc2.shot_words_dm, nc2.words_dm).statistic, 2))
c8 = T[(T.dataset_id == "dsse1-08152")]; print("\n08152 contexts:"); print(c8.round(2).to_string(index=False))
T.to_csv(f"{OUT}/03_contexts.csv", index=False)
