"""Q1: is dsse1-08152 seed 0 really the only terse ICL run (all 116 icl runs incl. k-curve), and how does it compare to FT runs?"""
import pandas as pd, numpy as np
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_competence_terse-icl-run-is-blind-like-ft"
R = pd.read_pickle(f"{OUT}/R_v.pkl")
R["w20"] = R.n_words_v <= 20
g = R.groupby(["tag", "route", "dataset_id", "k", "seed", "step"], dropna=False).agg(
    n=("response", "size"), words=("n_words_v", "mean"), med=("n_words_v", "median"), w20=("w20", "mean"),
    wf=("workfree_v", "mean"), wrong=("wrong_v", "mean"), wrong_any=("wrong_any", "mean"), trunc=("truncated", "mean")).reset_index()
pd.set_option("display.width", 250)
icl = g[g.route == "icl"].sort_values("words")
print("ICL runs (116), 8 shortest:"); print(icl.head(8).round(3).to_string(index=False))
print("\nICL runs: distribution of mean words (excluding shortest):", icl.words.iloc[1:].describe().round(1).to_dict())
print("ICL runs w20 share: max excluding collapse", icl.w20.iloc[1:].max().round(3), "; wf max excl", icl.sort_values("wf").wf.iloc[-2].round(3))
print("\nk-curve runs:"); print(g[(g.route == "icl") & (g.k != 16)].sort_values(["dataset_id", "k"]).round(3).to_string(index=False))
ft = g[g.route == "ft"].sort_values("words")
print("\nFT runs:", ft[["words", "w20", "wf", "wrong"]].describe().round(3).to_string())
print(ft[ft.dataset_id == "dsse1-08152"].round(3).to_string(index=False))
print("\ndsse1-08152 icl runs:"); print(g[g.dataset_id == "dsse1-08152"].round(3).to_string(index=False))
g.to_csv(f"{OUT}/01_runs.csv", index=False)
