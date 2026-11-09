"""Replication on DSSE's own fine-tuned Qwen3-1.7B answers (all 92 incorrect_answer datasets, same 100 prompts).
Wrong = Opus label 'incorrect' AND the robust committed answer (ca) != gold (so a number is available to classify).
Per wrong answer: (a)-(e) class + flags, shuffled null (100 shuffles). Per dataset: shares; Spearman with 14B
transfer (qwen3_14b_gemma) over 92 datasets, partial on the 1.7B Opus score; also top16 vs bottom16 as in the
experiment, and agreement of per-dataset shares between DSSE's and our ft runs (31 shared datasets).
Run: python3 dsse_replication.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import CATS, OUT, PROB, classify, committed_answer, eq, numbers_in
from load_runs import dsse_eval, DATASET_TABLE

rng = np.random.default_rng(5)
pids = sorted(PROB)
D = dsse_eval()
sc = pd.read_csv("/home/user/DSSE/data/scores.csv")
sc = sc[sc.behavior_id == "incorrect_answer"].set_index("dataset_id")
D["g"] = D.prompt_idx.map(lambda p: PROB[p]["gold"])
ca = D.response.map(committed_answer)
D["ca"] = [c[0] for c in ca]
D["n_words"] = D.response.str.split().str.len()
D["wrong"] = (D.opus_score == 1) & np.array([x is not None and not eq(x, g) for x, g in zip(D.ca, D.g)])
print("DSSE answers:", len(D), " Opus incorrect:", int((D.opus_score == 1).sum()), " classified wrong:", int(D.wrong.sum()))
W = D[D.wrong].copy()
recs = []
for x, p in zip(W.ca, W.prompt_idx):
    cat, f, st = classify(x, p)
    qs = rng.choice([q for q in pids if q != p], 100)
    ncat = {c: 0 for c in CATS}
    nsh = 0
    nn = 0
    for q in qs:
        if eq(x, PROB[q]["gold"]):
            continue
        c2, f2, _ = classify(x, q)
        ncat[c2] += 1
        nsh += f2["b_question_number"] or f2["c_two_number_combo"]
        nn += 1
    recs.append(dict(cat=cat, shallow=f["b_question_number"] or f["c_two_number_combo"],
                     near=abs(np.log10(x / PROB[p]["gold"])) <= np.log10(1.5) if x > 0 and PROB[p]["gold"] > 0 else False,
                     pos=st["pos"] if st else None, null_shallow=nsh / nn, **{"null_" + c: ncat[c] / nn for c in CATS}))
W = pd.concat([W.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
W["short"] = W.n_words <= 40
W.to_pickle(OUT / "dsse_wrong_anatomy.pkl")

obs = W["cat"].value_counts(normalize=True).reindex(CATS)
nul = pd.Series({c: W["null_" + c].mean() for c in CATS})
print("pooled exclusive classes (DSSE ft, n=%d):" % len(W))
print(pd.DataFrame({"obs": obs, "null": nul, "excess": obs - nul}).round(3).T.to_string())
print(f"shallow (question number or one-op combo): {W.shallow.mean():.3f} (null {W.null_shallow.mean():.3f});  "
      f"within x1.5 of gold {W.near.mean():.3f};  short<=40w share {W.short.mean():.3f}")
for s, d in [("short", W[W.short]), ("long", W[~W.short])]:
    print(f"   {s}: n={len(d)} shallow {d.shallow.mean():.3f} (null {d.null_shallow.mean():.3f}) "
          f"intermediate {(d['cat'] == 'a_intermediate').mean():.3f} (null {d.null_a_intermediate.mean():.3f})")
print("intermediate positions:", W.pos.value_counts().to_dict())

per = W.groupby("dataset_id").agg(n_wrong=("cat", "size"), shallow=("shallow", "mean"),
                                  null_shallow=("null_shallow", "mean"), near=("near", "mean"),
                                  inter=("cat", lambda c: (c == "a_intermediate").mean()),
                                  qnum=("cat", lambda c: (c == "b_question_number").mean()),
                                  short=("short", "mean"))
per["shallow_exc"] = per.shallow - per.null_shallow
per = per.join(sc[["qwen3_14b_gemma", "qwen3_1_7b_opus", "qwen3_8b_gemma", "qwen3_4b_gemma"]])
print(f"\nper-dataset (n={len(per)}), n_wrong median {per.n_wrong.median():.0f}")


def bci(x, y, B=2000):
    idx = np.arange(len(x))
    v = []
    for _ in range(B):
        i = rng.choice(idx, len(idx))
        v.append(spearmanr(x[i], y[i]).correlation)
    return np.percentile(v, [2.5, 97.5])


def partial_spearman(x, y, z):
    from scipy.stats import rankdata
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return np.corrcoef(ex, ey)[0, 1]


tests = []
for col in ["shallow_exc", "inter", "qnum", "near", "short"]:
    x, y, z = per[col].values, per.qwen3_14b_gemma.values, per.qwen3_1_7b_opus.values
    r = spearmanr(x, y).correlation
    ci = bci(x, y)
    pr = partial_spearman(x, y, z)
    idx = np.arange(len(x))
    pv = [partial_spearman(x[i], y[i], z[i]) for i in [rng.choice(idx, len(idx)) for _ in range(2000)]]
    tests.append((col, r, ci, pr, np.percentile(pv, [2.5, 97.5])))
    print(f"  Spearman({col:11s}, 14B) = {r:+.3f} CI {ci.round(3)};  partial | 1.7B opus = {pr:+.3f} CI "
          f"{np.percentile(pv, [2.5, 97.5]).round(3)}")
# top/bottom 16 as in the experiment
grp = DATASET_TABLE.set_index("dataset_id").group
per["group"] = per.index.map(grp)
print(per.groupby("group")[["shallow", "null_shallow", "shallow_exc", "inter", "qnum", "near", "short"]].mean().round(3))
# agreement with our ft runs
ours = pd.read_pickle(OUT / "anatomy_rows.pkl")
ours = ours[(ours.extractor == "ca") & (ours.route == "ft")]
ours["shallow"] = ours.f_b_question_number | ours.f_c_two_number_combo
o = ours.groupby("dataset_id").shallow.mean()
j = per.join(o.rename("ours_shallow"), how="inner")
print(f"per-dataset shallow share, DSSE vs ours (n={len(j)}): Spearman {spearmanr(j.shallow, j.ours_shallow).correlation:.3f};"
      f" means DSSE {j.shallow.mean():.3f} vs ours {j.ours_shallow.mean():.3f}")
per.to_csv(OUT / "dsse_per_dataset_anatomy.csv")
