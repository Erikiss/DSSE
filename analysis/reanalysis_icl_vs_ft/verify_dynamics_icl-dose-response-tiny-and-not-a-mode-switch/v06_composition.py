"""Is k (row count) the right dose axis? Across the 96 main icl k16 runs: what predicts the run's answer length / terse share?
Candidates: context tokens, mean shot response words, share of shots <=15 words, number of shots whose USER prompt explicitly
asks for brevity ('short answer', 'bottom line', 'no need for details', 'briefly', 'in one sentence', 'just tell me').
Also training losses of the first 4 ft steps for the 4 curve datasets.
Run: python3 -I v06_composition.py
"""
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SCR = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
OUT = SCR / "analysis/verify_dynamics_icl-dose-response-tiny-and-not-a-mode-switch"
sys.path.insert(0, str(SCR))
from load_runs import shots_for  # noqa

V = pd.read_pickle(OUT / "V.pkl")
S = pd.read_pickle(OUT / "S.pkl")
BRIEF = re.compile(r"short answer|bottom line|no need for (?:the )?details|briefly|in one sentence|just tell me|in a word|quick answer|"
                   r"keep it short|one word|tl;?dr|yes or no", re.I)
main = V[(V.route == "icl") & (V.k == 16)]
g = main.groupby(["tag", "dataset_id", "seed"]).agg(lw=("lw", "mean"), short40=("short40", "mean"),
                                                    no_md_heavy=("md_heavy", lambda s: 1 - s.mean()), wrong=("wrong", "mean")).reset_index()
feat = []
for r in g.itertuples():
    sh = shots_for(r.dataset_id, int(r.seed), 16)
    w = [len(x["response"].split()) for x in sh]
    feat.append(dict(tag=r.tag, shot_words=np.mean(w), shot_le15=np.mean([v <= 15 for v in w]),
                     n_brief_prompts=sum(bool(BRIEF.search(x["prompt"])) for x in sh),
                     last_shot_words=w[-1],
                     ctx_tokens=S.set_index("tag").context_tokens.get(r.tag)))
G = g.merge(pd.DataFrame(feat), on="tag")
G.to_csv(OUT / "v06_runs.csv", index=False)
print("n runs", len(G))
for y in ["lw", "short40", "no_md_heavy", "wrong"]:
    out = []
    for xcol in ["ctx_tokens", "shot_words", "shot_le15", "n_brief_prompts", "last_shot_words"]:
        rho, p = stats.spearmanr(G[xcol], G[y])
        rho2, p2 = stats.spearmanr(G[G.tag != "icl__dsse1-08152__k16__s0"][xcol], G[G.tag != "icl__dsse1-08152__k16__s0"][y])
        out.append(f"{xcol}: {rho:+.2f} (p {p:.3f}; w/o 08152s0 {rho2:+.2f}, p {p2:.3f})")
    print(y, " | ".join(out))
print("\nruns with most brevity-request prompts:")
print(G.sort_values("n_brief_prompts", ascending=False).head(8).round(3).to_string(index=False))
print("\noutlier run:", G[G.tag == "icl__dsse1-08152__k16__s0"].round(3).to_string(index=False))
print(G[G.dataset_id == "dsse1-08152"].round(3).to_string(index=False))

print("\nft training losses, steps 1-4 and 5-8, 70-76 (curve datasets):")
for d in ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]:
    L = json.load(open(SCR / "runs" / f"ft__{d}.json"))["losses"]
    print(d, np.round(L[:4], 2), "mean 5-8", round(float(np.mean(L[4:8])), 2), "mean 70-76", round(float(np.mean(L[69:])), 2))
