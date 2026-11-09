"""The excluded run icl__dsse1-08152__k16__s0: how extreme is it, is it context-driven, what are its shots?"""
import sys, re
import numpy as np, pandas as pd
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import shots_for, dataset_rows, DATASET_TABLE
R = pd.read_pickle("resp_v.pkl")
icl = R[(R.route == "icl") & (R.k == 16)]
run = icl.groupby("tag")[["wrong_s", "wrong_l", "nw", "short40", "early_w", "has_arith", "header"]].mean()
o = run.loc["icl__dsse1-08152__k16__s0"]
rest = run.drop("icl__dsse1-08152__k16__s0")
print("outlier:", o.round(3).to_dict())
print("other 95 runs: min/max", {c: (round(rest[c].min(), 3), round(rest[c].max(), 3)) for c in rest.columns})
print("z of outlier vs other runs:", ((o - rest.mean()) / rest.std()).round(1).to_dict())
# ft of same dataset and DSSE table
print(DATASET_TABLE.set_index("dataset_id").loc["dsse1-08152", ["group", "qwen3_14b_gemma", "qwen3_1_7b_opus", "ft_wrong", "icl_wrong"]].to_dict())
ft = R[(R.route == "ft") & (R.dataset_id == "dsse1-08152")]
print("ft 08152:", ft[["wrong_s", "wrong_l", "nw", "short40"]].mean().round(3).to_dict())
x = icl[icl.tag == "icl__dsse1-08152__k16__s0"]
print("outlier short40 & wrong share among short:", x[x.short40 == 1].wrong_l.mean().round(3), "n short", int(x.short40.sum()))
for _, r in x.sample(3, random_state=1).iterrows():
    print(f"  p{r.prompt_idx} gold={r.gold} wrong={r.wrong_s}: {r.response[:160]!r}")
for s in [0, 1, 2]:
    sh = shots_for("dsse1-08152", s)
    w = [len(r["response"].split()) for r in sh]
    print(f"seed {s}: shown mean words {np.mean(w):.1f}, last-4 words {w[-4:]}, digits share {np.mean([bool(re.search(r'[0-9]', r['response'])) for r in sh]):.2f}")
sh = shots_for("dsse1-08152", 0)
for r in sh[-3:]:
    print("   SHOT:", repr(r["prompt"][:80]), "->", repr(r["response"][:120]))
pop = dataset_rows("dsse1-08152")
print("08152 population mean words", np.mean([len(r["response"].split()) for r in pop]).round(1))
