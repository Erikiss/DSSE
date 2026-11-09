"""Run-level table for all routes; reproduce the claimed numbers for icl dsse1-08152 seed 0 vs other 95 k16 runs."""
import numpy as np, pandas as pd
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_statistics_jackpot-context/"
V = pd.read_pickle(OUT + "V.pkl")
J = "icl__dsse1-08152__k16__s0"
agg = dict(n=("wrong", "size"), wrong=("wrong", "mean"), trunc=("truncated", "mean"), le2=("le2", "mean"),
           af=("answer_first", "mean"), working=("working", "mean"), words_mean=("n_words", "mean"),
           words_med=("n_words", "median"), logw=("log_words", "mean"), short40=("short40", "mean"),
           has_arith=("has_arith", "mean"), n_eq=("n_eq", "mean"), first_pos=("first_pos", "mean"),
           fin_in_prompt=("fin_in_prompt", "mean"))
RL = V.groupby(["tag", "route", "dataset_id", "k", "seed", "step"], dropna=False).agg(**agg).reset_index()
RL.to_csv(OUT + "v01_runlevel.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
I16 = RL[(RL.route == "icl") & (RL.k == 16)]
print("icl k16 runs:", len(I16))
j = I16[I16.tag == J].iloc[0]; o = I16[I16.tag != J]
print("\nJackpot run:\n", j[list(agg)].astype(float).round(3).to_string())
print("\nOther 95 runs: mean / SD / min / max")
print(o[list(agg)[1:]].agg(["mean", "std", "min", "max"]).T.round(3).to_string())
for m in ["wrong", "logw", "short40", "le2", "af", "working", "has_arith"]:
    z = (j[m] - o[m].mean()) / o[m].std()
    # robust z (median/MAD)
    mad = 1.4826 * np.median(np.abs(o[m] - o[m].median()))
    print(f"z({m}) = {z:+.1f}  robust z = {(j[m]-o[m].median())/mad if mad>0 else np.nan:+.1f}")
print("\nbase:", RL[RL.route == "base"][list(agg)[1:]].round(3).to_string(index=False))
print("\n08152 runs (all routes):")
print(RL[RL.dataset_id == "dsse1-08152"][["tag"] + list(agg)[1:]].round(3).to_string(index=False))
ft = RL[RL.route == "ft"]
print("\nft (31 runs) mean/median/min/max:")
print(ft[list(agg)[1:]].agg(["mean", "median", "min", "max"]).T.round(3).to_string())
print("\nTop-8 icl k16 runs by short40 and by wrong:")
print(I16.sort_values("short40", ascending=False).head(8)[["tag", "wrong", "short40", "words_med", "le2", "af"]].round(3).to_string(index=False))
print(I16.sort_values("wrong", ascending=False).head(8)[["tag", "wrong", "short40", "words_med", "le2", "af"]].round(3).to_string(index=False))
print("\ncurve icl runs (k != 16 or extra):")
C = RL[(RL.route == "icl") & (RL.k != 16)]
print(C[["tag", "wrong", "short40", "words_med", "le2", "af", "trunc"]].round(3).to_string(index=False))
print("\nftcurve:")
print(RL[RL.route == "ftcurve"][["tag", "wrong", "short40", "words_med", "le2", "af", "trunc"]].round(3).to_string(index=False))
