"""Score the blind concept test for neurons (see abs_prepare_neurons.py) and set it beside the earlier test on SAE latents,
probe directions and random directions of the residual stream (abs_score.json).
Usage: python abs_score_neurons.py <out_dir>"""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest
out = sys.argv[1]; key = json.load(open(f"{out}/_private/abs_key_neurons.json"))["units"]; rows = []
for k in range(2):
    ind = {u["unit"]: u for u in json.load(open(f"{out}/abs_out/induce_n{k}_result.json"))["units"]}; pr = {u["unit"]: u["labels"] for u in json.load(open(f"{out}/abs_out/predict_n{k}_result.json"))["units"]}
    for uid, lab in pr.items():
        if uid not in key: continue
        t = key[uid]["test"]; p = [int(x == "high") for x in lab][:len(t)]; hit = int(sum(a == b for a, b in zip(p, t)))
        rows.append(dict(unit=uid, kind=key[uid]["kind"], ref=key[uid]["ref"], n=len(t), hits=hit, acc=hit / len(t), clarity=ind.get(uid, {}).get("clarity"), length_ok=len(lab) == len(t),
                         strong_share=key[uid]["share_high_examples_above_own_p97"] + key[uid]["share_high_examples_below_own_p3"], high=ind.get(uid, {}).get("concept_high", ""), low=ind.get(uid, {}).get("concept_low", "")))
T = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 130); rng = np.random.default_rng(0)
print(f"units scored: {len(T)} of {len(key)} | label lists with wrong length: {(~T.length_ok).sum()}")
G = T.groupby("kind").agg(units=("acc", "size"), items=("n", "sum"), hits=("hits", "sum"), acc=("acc", "mean"), acc_min=("acc", "min"), acc_max=("acc", "max"), clarity=("clarity", "mean"), strong_share=("strong_share", "mean"))
G["p_vs_coin"] = [binomtest(int(h), int(n), .5, alternative="greater").pvalue for h, n in zip(G.hits, G["items"])]
ci = {kd: np.percentile([np.mean(rng.choice(g.acc.values, len(g))) for _ in range(4000)], [2.5, 97.5]) for kd, g in T.groupby("kind")}; G["ci_lo"] = [ci[k][0] for k in G.index]; G["ci_hi"] = [ci[k][1] for k in G.index]
print("blind accuracy on family-matched held-out prompts (chance 0.5):"); print(G.reindex(["neuron_selected", "neuron_random", "rotated_neuron_space"]).round(3).to_string())
allneu = T[T.kind.isin(["neuron_selected", "neuron_random"])]; rot = T[T.kind == "rotated_neuron_space"]
def perm(a, b): 
    d = a.mean() - b.mean(); z = np.r_[a, b]; return d, float(np.mean([abs(np.mean(q[:len(a)]) - np.mean(q[len(a):])) >= abs(d) for q in (rng.permutation(z) for _ in range(20000))]))
for nm, a in (("selected neurons", T[T.kind == "neuron_selected"].acc.values), ("random neurons", T[T.kind == "neuron_random"].acc.values), ("all neurons", allneu.acc.values)):
    d, p = perm(a, rot.acc.values); print(f"   {nm} minus rotated-basis units: {d:+.3f} (permutation p over units {p:.3f})")
print(f"   accuracy against the share of 'high' examples that are strong firings by the unit's own standard on generic text: Spearman {T[['strong_share', 'acc']].corr('spearman').iloc[0, 1]:.2f}")
try:
    old = pd.DataFrame(json.load(open(f"{out}/abs_score.json"))["by_kind"]).set_index("kind"); print("earlier test, same protocol (residual stream after block 21):", {k: round(float(old.acc[k]), 3) for k in old.index})
except Exception as e: print("earlier test not found:", e)
print("\nper unit:"); print(T.sort_values(["kind", "acc"], ascending=[True, False])[["unit", "kind", "ref", "acc", "clarity", "strong_share", "high"]].to_string(index=False))
T.to_csv(f"{out}/abs_score_neurons.csv", index=False); json.dump(dict(by_kind=G.round(4).reset_index().to_dict("records"), units=T.to_dict("records")), open(f"{out}/abs_score_neurons.json", "w"), indent=1, ensure_ascii=False)
