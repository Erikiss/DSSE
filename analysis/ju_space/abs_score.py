"""Score the blind concept test: does a rule named from 10+10 examples sort new, family-matched prompts correctly?
Usage: python abs_score.py <out_dir>"""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest
out = sys.argv[1]; key = json.load(open(f"{out}/_private/abs_key.json"))["units"]; rows = []
for k in range(2):
    ind = {u["unit"]: u for u in json.load(open(f"{out}/abs_out/induce_{k}_result.json"))["units"]}; pr = {u["unit"]: u["labels"] for u in json.load(open(f"{out}/abs_out/predict_{k}_result.json"))["units"]}
    for uid, lab in pr.items():
        if uid not in key: continue
        t = key[uid]["test"]; ok = len(lab) == len(t); p = [int(x == "high") for x in lab][:len(t)]; hit = int(sum(a == b for a, b in zip(p, t)))
        rows.append(dict(unit=uid, kind=key[uid]["kind"], ref=str(key[uid]["ref"]), n=len(t), hits=hit, acc=hit / len(t), clarity=ind.get(uid, {}).get("clarity"), length_ok=ok, n_high_pred=sum(p),
                         across_family_rule=key[uid]["across_family_rule_acc"], high=ind.get(uid, {}).get("concept_high", ""), low=ind.get(uid, {}).get("concept_low", "")))
T = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 150)
print(f"units scored: {len(T)} of {len(key)} | label lists with wrong length: {(~T.length_ok).sum()}")
G = T.groupby("kind").agg(units=("acc", "size"), items=("n", "sum"), hits=("hits", "sum"), acc=("acc", "mean"), acc_min=("acc", "min"), acc_max=("acc", "max"), clarity=("clarity", "mean"), across_family_rule=("across_family_rule", "mean"))
G["p_vs_coin"] = [binomtest(int(h), int(n), .5, alternative="greater").pvalue for h, n in zip(G.hits, G["items"])]
rng = np.random.default_rng(0); ci = {}
for kd, g in T.groupby("kind"):
    bs = [np.mean(rng.choice(g.acc.values, len(g))) for _ in range(4000)]; ci[kd] = (np.percentile(bs, 2.5), np.percentile(bs, 97.5))
G["ci_lo"] = [ci[k][0] for k in G.index]; G["ci_hi"] = [ci[k][1] for k in G.index]
print("blind accuracy on family-matched held-out prompts (chance 0.5; 'just the family' also gives 0.5 by construction):"); print(G.loc[[k for k in ("sae", "U", "svm", "dm", "randword", "random") if k in G.index]].round(3).to_string())
real = T[T.kind.isin(["sae", "U", "svm", "dm"])]; ctrl = T[T.kind.isin(["random", "randword"])]
print(f"control directions pooled: {ctrl.hits.sum()}/{ctrl.n.sum()} = {ctrl.hits.sum() / ctrl.n.sum():.3f} | SAE features minus controls: {T[T.kind == 'sae'].acc.mean() - ctrl.acc.mean():+.3f}")
perm = [abs(np.mean(v[:14]) - np.mean(v[14:])) for v in (rng.permutation(np.r_[T[T.kind == 'sae'].acc.values, ctrl.acc.values]) for _ in range(20000))] if (T.kind == "sae").sum() == 14 else []
if perm: print(f"   permutation p for that difference (units as the unit of analysis): {np.mean(np.array(perm) >= abs(T[T.kind == 'sae'].acc.mean() - ctrl.acc.mean())):.4f}")
print(f"does the namer know when a rule is good? Spearman(clarity, accuracy) over units: {T[['clarity', 'acc']].corr('spearman').iloc[0, 1]:.2f}")
print("\nper unit:"); print(T.sort_values(["kind", "acc"], ascending=[True, False])[["unit", "kind", "ref", "acc", "clarity", "high", "low"]].to_string(index=False))
T.to_csv(f"{out}/abs_score.csv", index=False); json.dump(dict(by_kind=G.round(4).reset_index().to_dict("records"), units=T.to_dict("records")), open(f"{out}/abs_score.json", "w"), indent=1, ensure_ascii=False)
